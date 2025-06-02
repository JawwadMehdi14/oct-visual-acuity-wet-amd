import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# === CONFIGURATION ===
json_path = "flattened_all_years.json"  # 👈 Use full path if running standalone
input_dir = "E:/Labeled_PNGs"           # 👈 Path to your image folder
output_dir = r"D:\MS Computer Engineering\Thesis\Code\New folder\Patientwiseclassificationfullpipeline\No data leakage\GRADCAM"           # 👈 Where to save Grad-CAM visualizations
patient_eye = "ACCORDINI, LIDO EMILIO, LEFT"  # 👈 Patient-Eye ID
model_path = r"D:\MS Computer Engineering\Thesis\Code\AMD\EyeWiseResults_20250531_193203\best_model.pth"  # 👈 Update path

os.makedirs(output_dir, exist_ok=True)

# === MODEL DEFINITION ===
class ResNetForGradCAM(nn.Module):
    def __init__(self, return_nodes=["layer4"]):
        super().__init__()
        base = resnet18(weights="IMAGENET1K_V1")
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = create_feature_extractor(base, return_nodes={"layer4": "feat"})
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feats = self.feature_extractor(x)["feat"]
        pooled = self.flatten(self.pool(feats))
        out = self.classifier(pooled)
        return out, feats

# === GRAD-CAM FUNCTION ===
def generate_gradcam(model, image_tensor, device):
    model.eval()
    image_tensor.requires_grad = True
    output, features = model(image_tensor)
    score = output.squeeze()
    model.zero_grad()
    score.backward()

    cam = features.squeeze(0).detach().cpu().numpy()
    weights = np.mean(cam, axis=(1, 2))
    heatmap = np.zeros(cam.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * cam[i, :, :]

    heatmap = np.maximum(heatmap, 0)
    vmin = np.percentile(heatmap, 60)
    vmax = np.percentile(heatmap, 99)
    heatmap = np.clip(heatmap, vmin, vmax)
    heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze().numpy()
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-8
    return heatmap

# === SIDE-BY-SIDE VISUALIZER ===
def save_side_by_side(image_path, heatmap, output_path):
    orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orig = cv2.resize(orig, (224, 224))
    heatmap_img = np.uint8(255 * heatmap)
    heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR), 0.6, heatmap_img, 0.4, 0)
    combined = np.hstack([cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR), overlay])
    cv2.imwrite(output_path, combined)

# === SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetForGradCAM().to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
transform = transforms.ToTensor()

# === LOAD & FILTER JSON LIST ===
with open(json_path, "r") as f:
    data = json.load(f)

matching_entries = [entry for entry in data if entry.get("patient_eye") == patient_eye]

if not matching_entries:
    print(f"❌ No entries found for patient-eye: {patient_eye}")
else:
    for record in tqdm(matching_entries, desc=f"Processing {patient_eye}"):
        for img_file in record.get("images", []):
            img_path = os.path.join(input_dir, img_file)
            if not os.path.exists(img_path):
                print(f"⚠️ Missing: {img_path}")
                continue
            image = Image.open(img_path).convert("L").resize((224, 224))
            tensor = transform(image).unsqueeze(0).to(device)
            heatmap = generate_gradcam(model, tensor, device)

            filename = os.path.splitext(os.path.basename(img_file))[0]
            out_path = os.path.join(output_dir, f"{filename}_gradcam.png")
            save_side_by_side(img_path, heatmap, out_path)

    print(f"\n✅ Saved all Grad-CAM visualizations to: {output_dir}")
