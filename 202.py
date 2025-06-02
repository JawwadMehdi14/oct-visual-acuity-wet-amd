import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# === CONFIG ===
json_path = "AMD_Label_New.json"
input_dir = "E:/Labeled_PNGs"
output_dir = r"D:\MS Computer Engineering\Thesis\Code\New folder\Year-wise Classification Full Pipeline\No data Leakage"
patient_name = "BEGNONI, NERINO, LEFT"  # 👈 Update this
target_year = 6                           # 👈 Update this (as integer)

os.makedirs(output_dir, exist_ok=True)

# === MODEL ===
class ResNetForGradCAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights="IMAGENET1K_V1")
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = create_feature_extractor(base, return_nodes={"layer5": "feat"})
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

# === GRAD-CAM ===
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

def save_side_by_side(image_path, heatmap, output_path):
    orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orig = cv2.resize(orig, (224, 224))
    heatmap_img = np.uint8(255 * heatmap)
    heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR), 0.6, heatmap_img, 0.4, 0)
    combined = np.hstack([cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR), overlay])
    cv2.imwrite(output_path, combined)

# === MAIN ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetForGradCAM().to(device)
model.load_state_dict(torch.load("Results_TemporalResNet_20250531_225902/best_model.pth", map_location=device), strict=False)

transform = transforms.ToTensor()

# === LOAD JSON ===
with open(json_path, "r") as f:
    data = json.load(f)

if patient_name not in data:
    print(f"❌ Patient '{patient_name}' not found in JSON.")
    exit()

entries = data[patient_name]
found = False

for record in tqdm(entries, desc=f"Processing {patient_name}"):
    if record.get("year") == target_year:
        found = True
        image_list = record.get("images", [])
        for img_file in image_list:
            img_path = os.path.join(input_dir, img_file)
            if not os.path.exists(img_path):
                print(f"⚠️ Skipping missing image: {img_path}")
                continue
            image = Image.open(img_path).convert("L").resize((224, 224))
            tensor = transform(image).unsqueeze(0).to(device)
            heatmap = generate_gradcam(model, tensor, device)

            filename = os.path.splitext(os.path.basename(img_file))[0]
            out_path = os.path.join(output_dir, f"{filename}_gradcam.png")
            save_side_by_side(img_path, heatmap, out_path)

if not found:
    print(f"❌ No scans found for patient {patient_name} in year {target_year}")
else:
    print(f"✅ Done! Visualizations saved to: {output_dir}")
