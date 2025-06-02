# === GRAD-CAM++ EXPLAINABILITY SCRIPT ===
# Uses trained model weights to visualize attention on OCT images using Grad-CAM++

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os

# === MODEL DEFINITION (same as training) ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        return self.classifier(x)

# === GRAD-CAM++ FUNCTION ===
def gradcam_plus_plus(model, image_tensor, target_layer_idx=6):
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)
        output.retain_grad()

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fwd = model.feature_extractor[target_layer_idx].register_forward_hook(forward_hook)
    handle_bwd = model.feature_extractor[target_layer_idx].register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    score = output[0]
    model.zero_grad()
    score.backward(retain_graph=True)

    handle_fwd.remove()
    handle_bwd.remove()

    A = activations[0].squeeze(0).detach().cpu().numpy()
    grads = gradients[0].squeeze(0).detach().cpu().numpy()

    numerator = grads ** 2
    denominator = 2 * grads ** 2 + np.sum(A * grads ** 3, axis=(1, 2), keepdims=True)
    denominator = np.where(denominator != 0.0, denominator, 1e-8)
    alpha = numerator / denominator
    weights = np.sum(alpha * np.maximum(grads, 0), axis=(1, 2))

    cam = np.sum(weights[:, None, None] * A, axis=0)
    cam = np.maximum(cam, 0)
    cam = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().numpy()
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cam

# === SAVE HEATMAP ===
def save_gradcam_overlay(image_path, heatmap, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    edges = cv2.Canny(image, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)
    overlay = cv2.addWeighted(overlay, 0.9, edges_colored, 0.1, 0)
    cv2.imwrite(output_path, overlay)

# === LOAD MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))

# === LOAD IMAGE ===
sample_path = "E:/Labeled_PNGs/TOMBA11C_Tomba_Carla_L_akt10.png"  # Change to your image
image = Image.open(sample_path).convert("L").resize((224, 224))
tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

# === GENERATE AND SAVE ===
heatmap = gradcam_plus_plus(model, tensor)
output_path = "gradcam_sample_overlay++.png"
save_gradcam_overlay(sample_path, heatmap, output_path)
print(f"✅ Grad-CAM++ overlay saved to {output_path}")
