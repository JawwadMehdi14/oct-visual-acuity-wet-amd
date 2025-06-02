# === Grad-CAM++ for Yearwise Model on a Single Image ===

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, transforms
from PIL import Image
import cv2
import os

from Continuation9 import SimpleCNN  # Ensure this matches the SimpleCNN from your training code

# === Configuration ===
image_path = "E:/Labeled_PNGs/ACCOR42L_Accordini_Lido Emilio_L_akt8.png"  # Replace with your image
model_path = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load image ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img = Image.open(image_path).convert('L')
tensor = transform(img).unsqueeze(0).to(device)  # shape: [1, 1, 224, 224]

# === Load model ===
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Register hook on last conv layer ===
features = None
grads = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_input, grad_output):
    global grads
    grads = grad_output[0]

# Hook last layer
target_layer = model.feature_extractor[-1]  # AdaptiveAvgPool2d
handle_fwd = target_layer.register_forward_hook(forward_hook)
handle_bwd = target_layer.register_backward_hook(backward_hook)

# === Forward and backward ===
output = model(tensor)
label_index = torch.sigmoid(output).item()
model.zero_grad()
output.backward()

# === Grad-CAM++ computation ===
grads_val = grads[0].cpu().numpy()
features_val = features[0].detach().cpu().numpy()

weights = np.mean(grads_val, axis=(1, 2))
cam = np.zeros(features_val.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * features_val[i, :, :]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224, 224))
cam -= np.min(cam)
cam /= np.max(cam) if np.max(cam) != 0 else 1

# === Overlay ===
heatmap = (cam * 255).astype(np.uint8)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
img_np = np.array(img.resize((224, 224)).convert('RGB'))
img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

# === Save and show ===
out_path = "gradcam_overlay.png"
cv2.imwrite(out_path, overlay)
print(f"✅ Grad-CAM saved to: {out_path}")

# === Cleanup ===
handle_fwd.remove()
handle_bwd.remove()

plt.imshow(overlay)
plt.title(f"Grad-CAM++ Output (Prob={label_index:.2f})")
plt.axis(False)
plt.show()
