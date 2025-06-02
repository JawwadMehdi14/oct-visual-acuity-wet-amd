import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os

# === MODEL FOR GRAD-CAM (SINGLE IMAGE EXPLANATION ONLY) ===
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
        feats = self.feature_extractor(x)["feat"]  # B x 512 x H x W
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

    grads = image_tensor.grad  # gradient wrt input
    cam = features.squeeze(0).detach().cpu().numpy()       # shape (512, H, W)
    grads_val = image_tensor.grad.squeeze(0).cpu().numpy() # shape (1, 224, 224)

    pooled_grads = torch.mean(model.classifier[1].weight).item()

    weights = np.mean(cam, axis=(1, 2))  # shape (512,)
    heatmap = np.zeros(cam.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * cam[i, :, :]

    heatmap = np.maximum(heatmap, 0)
    # 🎯 Percentile clipping to clean up background
    vmin = np.percentile(heatmap, 60)
    vmax = np.percentile(heatmap, 99)
    heatmap = np.clip(heatmap, vmin, vmax)
    # 🔄 Interpolation and normalization
    heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap = torch.nn.functional.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze().numpy()
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-8
    return heatmap

def save_gradcam_overlay(image_path, heatmap, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, overlay)

# === LOAD TRAINED MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetForGradCAM().to(device)
model.load_state_dict(torch.load("Results_TemporalResNet_20250531_225902/best_model.pth", map_location=device), strict=False)

# === RUN EXPLANATION ===
sample_path = "E:/Labeled_PNGs/BARBE25O_Barbessi_Ottavio_L_akt1.png"  # Update as needed
image = Image.open(sample_path).convert("L").resize((224, 224))
transform = transforms.ToTensor()
tensor = transform(image).unsqueeze(0).to(device)

heatmap = generate_gradcam(model, tensor, device)
save_gradcam_overlay(sample_path, heatmap, "Results_TemporalResNet_20250531_225902/gradcam_overlay.png")
print("✅ Grad-CAM overlay saved!")
