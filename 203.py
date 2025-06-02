# === GRAD-CAM EXPLAINABILITY SCRIPT ===
# Uses trained model weights to visualize attention on OCT images

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os

# === MODEL DEFINITION (same as training) ===
class SimpleCNN(nn.Module):
       def __init__(self):
        super().__init__()
        base = resnet18(weights="IMAGENET1K_V1")
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        def forward(self, x, lengths):
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            feats = self.flatten(self.encoder(x)).view(B, T, -1)
            mean_feats = torch.stack([
                torch.mean(feats[i, :lengths[i]], dim=0)
                for i in range(B)
            ])
            return self.classifier(mean_feats).squeeze(1)
    # def __init__(self):
    #     super(SimpleCNN, self).__init__()
    #     resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    #     resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #     self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    #     self.flatten = nn.Flatten()
    #     self.classifier = nn.Sequential(
    #         nn.Linear(512, 128),
    #         nn.ReLU(),
    #         nn.Dropout(0.2),
    #         nn.Linear(128, 1)
    #     )

    # def forward(self, x):
    #     x = self.feature_extractor(x)
    #     x = self.flatten(x)
    #     return self.classifier(x)

# === GRAD-CAM FUNCTIONS ===
def generate_gradcam(model, image_tensor, device, target_layer_idx=6):
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)
        output.retain_grad()

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    handle_fwd = model.feature_extractor[target_layer_idx].register_forward_hook(forward_hook)
    handle_bwd = model.feature_extractor[target_layer_idx].register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)
    score = output[0]
    model.zero_grad()
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    fmap = activations[0].squeeze(0).detach().cpu().numpy()
    grads_val = gradients[0].squeeze(0).detach().cpu().numpy()
    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i, :, :]
    cam = np.maximum(cam, 0)

    import torch.nn.functional as F
    cam = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().numpy()
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cam

def save_gradcam_overlay(image_path, heatmap, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    edges = cv2.Canny(image, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)
    overlay = cv2.addWeighted(overlay, 0.9, edges_colored, 0.1, 0)  # light edge map
    cv2.imwrite(output_path, overlay)

# === LOAD TRAINED MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("Results_TemporalResNet_20250531_225902/best_model.pth", map_location=device))

from tqdm import tqdm

# === PROCESS A SAMPLE IMAGE ===
sample_path = "E:/Labeled_PNGs/TOMBA11C_Tomba_Carla_L_akt10.png"  # Change to a valid image path
image = Image.open(sample_path).convert("L").resize((224, 224))
tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

# === GENERATE AND SAVE GRAD-CAM ===
heatmap = generate_gradcam(model, tensor, device)
output_path = "Results_TemporalResNet_20250531_225902/gradcam_sample_overlay.png"
save_gradcam_overlay(sample_path, heatmap, output_path)
print(f"✅ Grad-CAM overlay saved to {output_path}")
