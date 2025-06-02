# === YEAR-WISE IMAGE CLASSIFIER FOR GOOD/BAD LABELS ===
# Based on AMD_Label_New.json with image-level BCVA-based labels

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
import json, os
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_fscore_support
import matplotlib.pyplot as plt
import csv
import pandas as pd

# === DATASET ===
class YearwiseImageDataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        with open(json_path, 'r') as f:
            data = json.load(f)

        for patient_key, records in data.items():
            for record in records:
                label = 1 if record['BCVA'] >= 0.5 else 0
                for img_name in record['images']:
                    img_path = os.path.join("E:/Labeled_PNGs", img_name)
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, label, patient_key))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, patient_id = self.samples[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image)  # Already (1, H, W) for grayscale  # (1, H, W)
        return image, label, patient_id

# === MODEL ===
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

# === SETUP ===
json_file = "AMD_Label_New.json"
dataset = YearwiseImageDataset(json_file)
labels = [s[1] for s in dataset.samples]

train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.3, stratify=labels, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=42)

train_set = torch.utils.data.Subset(dataset, train_idx)
val_set = torch.utils.data.Subset(dataset, val_idx)
test_set = torch.utils.data.Subset(dataset, test_idx)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

print(f"Training images: {len(train_set)}, Validation: {len(val_set)}, Testing: {len(test_set)}")

# === TRAINING ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
best_auc = 0.0
patience = 3
patience_counter = 0

train_results, val_results, test_results = [], [], []

for epoch in range(10):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for images, labels, _ in loop:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    # === VALIDATION ===
    model.eval()
    preds, targets, ids = [], [], []
    with torch.no_grad():
        for image, label, pid in val_loader:
            image = image.to(device)
            logit = model(image)
            prob = torch.sigmoid(logit).item()
            preds.append(prob)
            targets.append(label.item())
            ids.append(pid[0])

    pred_labels = [1 if p >= 0.5 else 0 for p in preds]
    acc = accuracy_score(targets, pred_labels)
    prec = precision_score(targets, pred_labels)
    rec = recall_score(targets, pred_labels)
    f1 = f1_score(targets, pred_labels)
    auc = roc_auc_score(targets, preds)
    print(f"Val Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    val_results.extend(zip(ids, targets, pred_labels, preds))

    if auc > best_auc:
        torch.save(model.state_dict(), "best_model.pth")
        best_auc = auc
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

# === EVALUATE ON TRAINING ===
model.eval()
with torch.no_grad():
    for image, label, pid in train_loader:
        image = image.to(device)
        logit = model(image)
        prob = torch.sigmoid(logit).cpu().numpy().flatten()
        for i, p in enumerate(pid):
            train_results.append((p, int(label[i]), int(prob[i] >= 0.5), float(prob[i])))

# === EVALUATE ON TEST ===
with torch.no_grad():
    for image, label, pid in tqdm(test_loader, desc="Testing"):
        image = image.squeeze(1).to(device)
        logit = model(image)
        prob = torch.sigmoid(logit).item()
        test_results.append((pid[0], int(label.item()), int(prob >= 0.5), float(prob)))

# === SAVE ALL TO EXCEL WITH 3 SHEETS ===
with pd.ExcelWriter("all_predictions.xlsx") as writer:
    pd.DataFrame(train_results, columns=["PatientID", "TrueLabel", "PredLabel", "Probability"]).to_excel(writer, sheet_name="Train", index=False)
    pd.DataFrame(val_results, columns=["PatientID", "TrueLabel", "PredLabel", "Probability"]).to_excel(writer, sheet_name="Validation", index=False)
    pd.DataFrame(test_results, columns=["PatientID", "TrueLabel", "PredLabel", "Probability"]).to_excel(writer, sheet_name="Test", index=False)
print("📁 Saved predictions to all_predictions.xlsx")

# === PLOT CONFUSION MATRIX & ROC CURVE ===
y_true = [row[1] for row in test_results]
y_pred = [row[2] for row in test_results]
y_prob = [row[3] for row in test_results]

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.savefig("confusion_matrix.png")
plt.close()

RocCurveDisplay.from_predictions(y_true, y_prob)
plt.title("ROC Curve (Test Set)")
plt.savefig("roc_curve.png")
plt.close()
print("📊 Confusion matrix and ROC curve saved.")
