import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import densenet121, DenseNet121_Weights
import json
import random
from tqdm import tqdm

# === DENSENET MODEL ===
class SimpleDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = densenet121(weights="IMAGENET1K_V1")
        base.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        feats = self.flatten(x)
        logits = self.classifier(feats)
        return logits.squeeze(1), x
# class SimpleDenseNet(nn.Module):
#     def __init__(self):
#         super(SimpleDenseNet, self).__init__()
#         weights = DenseNet121_Weights.DEFAULT
#         densenet = densenet121(weights=weights)
#         densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.feature_extractor = densenet.features
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(1024, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         pooled = self.pool(features)
#         logits = self.classifier(pooled)
#         return logits, features

# === UTILS ===
def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(img_path).convert('L')
    tensor = transform(image).unsqueeze(0)
    return image, tensor

def apply_colormap_on_image(org_img, activation, colormap_name='jet', alpha=0.5):
    import matplotlib.cm as cm
    heatmap = cm.get_cmap(colormap_name)(activation)
    heatmap = np.delete(heatmap, 3, 2)
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(org_img.size)
    heatmap = np.array(heatmap) / 255.0
    org_img_rgb = np.array(org_img.convert('RGB')) / 255.0
    blended = org_img_rgb * (1 - alpha) + heatmap * alpha
    return (blended * 255).astype(np.uint8)

# === GRAD-CAM GENERATION ===
def generate_gradcam(model, image_tensor, device):
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    target_layer = model.encoder.denseblock4

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    image_tensor.requires_grad = True
    logits, _ = model(image_tensor)
    score = logits
    model.zero_grad()
    score.backward()

    act = activations[0][0].cpu().numpy()
    grad = gradients[0][0].cpu().numpy()

    weights = np.mean(grad, axis=(1, 2))
    heatmap = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * act[i, :, :]

    heatmap = np.maximum(heatmap, 0)
    vmin = np.percentile(heatmap, 60)
    vmax = np.percentile(heatmap, 99)
    heatmap = np.clip(heatmap, vmin, vmax)
    heatmap -= heatmap.min()
    heatmap /= (heatmap.max() + 1e-8)

    heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze().numpy()
    heatmap = np.clip(heatmap, 0, 1)

    handle_fwd.remove()
    handle_bwd.remove()

    return heatmap


def save_side_by_side_image(orig_pil, overlay_pil, save_path):
    width, height = orig_pil.size
    combined = Image.new("RGB", (2 * width, height))
    combined.paste(orig_pil.convert('RGB'), (0, 0))
    combined.paste(overlay_pil, (width, 0))
    combined.save(save_path)

# === MAIN FUNCTION ===
def run_full_automation(
    model_weights_path,
    json_path,
    image_root_dir,
    output_dir
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    # Load model
    model = SimpleDenseNet().to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    print("✅ Model loaded.")

    # Load JSON
    with open(json_path, 'r') as f:
        label_data = json.load(f)

    # Map from category to saved status
    found_examples = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    os.makedirs(output_dir, exist_ok=True)

    # Helper: turn "good"/"bad" into 1/0
    def label_to_int(label_str):
        return 1 if label_str.lower() == "good" else 0

    # Collect all entries
    all_images = []
    for patient_eye, visits in label_data.items():
        for visit in visits:
            true_label = label_to_int(visit["label"])
            for img_filename in visit["images"]:
                full_path = os.path.join(image_root_dir, img_filename)
                if os.path.exists(full_path):
                    all_images.append( (full_path, true_label) )
                else:
                    print(f"⚠️ Missing file: {full_path}")

    print(f"✅ Total images found: {len(all_images)}")

    # Randomize order to avoid bias
    images_good = [(p, l) for p, l in all_images if l == 1]
    images_bad = [(p, l) for p, l in all_images if l == 0]

    random.shuffle(images_good)
    random.shuffle(images_bad)

    # === Process bad images next (TN, FP) ===
    for img_path, true_label in tqdm(images_bad, desc="Scanning BAD images"):
        orig_img, input_tensor = load_image(img_path)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            logits, _ = model(input_tensor)
            prob = logits.item()
            pred_label = 1 if prob >= 0.5 else 0

        if true_label == 0 and pred_label == 0:
            category = "TN"
        elif true_label == 0 and pred_label == 1:
            category = "FP"
        else:
            category = None

        if category and found_examples[category] < 5:
            heatmap = generate_gradcam(model, input_tensor, device)
            overlay = apply_colormap_on_image(orig_img, heatmap, colormap_name='jet', alpha=0.5)

            out_path = os.path.join(output_dir, f"{category}_{found_examples[category]+1}.png")
            overlay_img = Image.fromarray(overlay)
            save_side_by_side_image(orig_img, overlay_img, out_path)
            found_examples[category] += 1

            print(f"✅ Saved {category} example #{found_examples[category]}: {out_path}")

        if all(count >= 5 for count in found_examples.values()):
            break

            # === Process good images first (TP, FP) ===
    for img_path, true_label in tqdm(images_good, desc="Scanning GOOD images"):
        orig_img, input_tensor = load_image(img_path)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            logits, _ = model(input_tensor)
            prob = logits.item()
            pred_label = 1 if prob >= 0.5 else 0

        if true_label == 1 and pred_label == 1:
            category = "TP"
        elif true_label == 1 and pred_label == 0:
            category = "FN"
        else:
            category = None

        if category and found_examples[category] < 5:
            heatmap = generate_gradcam(model, input_tensor, device)
            overlay = apply_colormap_on_image(orig_img, heatmap, colormap_name='jet', alpha=0.5)

            out_path = os.path.join(output_dir, f"{category}_{found_examples[category]+1}.png")
            overlay_img = Image.fromarray(overlay)
            save_side_by_side_image(orig_img, overlay_img, out_path)
            found_examples[category] += 1

            print(f"✅ Saved {category} example #{found_examples[category]}: {out_path}")

        if all(count >= 5 for count in found_examples.values()):
            break

    if not all(found_examples.values()):
        print("⚠️ Could not find examples for all categories!")


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    run_full_automation(
        model_weights_path=r"D:\MS Computer Engineering\Thesis\Code\AMD\Dense_Results_Yearwise_CV_20250615_214628\Fold1\best_model.pth",
        json_path=r"D:\MS Computer Engineering\Thesis\Code\AMD\AMD_Label_New.json",
        image_root_dir=r"E:\Labeled_PNGs",
        output_dir=r"D:\MS Computer Engineering\Thesis\Code\AMD\Explainability\Static\Examples"
    )
