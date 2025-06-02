# === IMAGE-LEVEL VOTING CLASSIFIER ===
# Replaces patient-level averaging with per-image classification
# Aggregates predictions per patient via majority voting during validation/test

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
import json, os
import numpy as np
from PIL import Image
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ImageLevelDataset(Dataset):
    def __init__(self, json_path, image_root, transform=None):
        self.samples = []
        self.transform = transform
        with open(json_path, 'r') as f:
            data = json.load(f)
        for entry in data:
            patient_key = entry['patient_eye']
            label = 1 if entry['label'] == 'improved' else 0
            for img_path in entry['images']:
                if os.path.isfile(img_path):
                    self.samples.append((img_path, label, patient_key))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, patient_id = self.samples[idx]
        image = Image.open(img_path).convert('L')
        image = image.resize((224, 224))
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)
        return image, label, patient_id

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
json_file = "flattened_all_years.json"
image_root = "E:/Labeled_PNGs"
num_epochs = 10
batch_size = 16
lr = 1e-4
seed = 42

print("📊 Counting samples...")
dataset = ImageLevelDataset(json_file, image_root)
labels = [s[1] for s in dataset.samples]
patient_ids = [s[2] for s in dataset.samples]
from sklearn.model_selection import train_test_split
train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.3, stratify=labels, random_state=seed)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=seed)
train_set = torch.utils.data.Subset(dataset, train_idx)
val_set = torch.utils.data.Subset(dataset, val_idx)
test_set = torch.utils.data.Subset(dataset, test_idx)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
print(f"Training images: {len(train_set)}, Validation images: {len(val_set)}, Testing images: {len(test_set)}")

# === MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# === TRAIN ===
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_fscore_support
import matplotlib.pyplot as plt

best_auc = 0.0
patience = 3
patience_counter = 0
train_val_log = open("train_val_metrics.csv", "w", newline="")
log_writer = csv.writer(train_val_log)
log_writer.writerow(["Epoch", "Split", "Accuracy", "Precision", "Recall", "F1", "AUC"])

for epoch in range(num_epochs):
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for images, labels, _ in loop:
        images = images.squeeze(1).to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    print("🔍 Evaluating on training set...")
    model.eval()
    train_preds_per_patient = defaultdict(list)
    train_labels_per_patient = {}
    with torch.no_grad():
        for image, label, pid in train_loader:
            image = image.squeeze(1).to(device)
            logit = model(image)
            prob = torch.sigmoid(logit).cpu().numpy().flatten()
            for i, p in enumerate(pid):
                train_preds_per_patient[p].append(prob[i])
                train_labels_per_patient[p] = label[i].item()
    y_true_train = [int(train_labels_per_patient[pid]) for pid in train_preds_per_patient]
    y_pred_train = [int(np.mean(train_preds_per_patient[pid]) >= 0.5) for pid in train_preds_per_patient]
    acc_train = accuracy_score(y_true_train, y_pred_train)
    prec_train = precision_score(y_true_train, y_pred_train)
    rec_train = recall_score(y_true_train, y_pred_train)
    f1_train = f1_score(y_true_train, y_pred_train)
    auc_train = roc_auc_score(y_true_train, list(map(np.mean, train_preds_per_patient.values())))
    print(f"Train Accuracy: {acc_train:.4f} | Precision: {prec_train:.4f} | Recall: {rec_train:.4f} | F1: {f1_train:.4f} | AUC: {auc_train:.4f}")
    class_prec, class_rec, class_f1, _ = precision_recall_fscore_support(y_true_train, y_pred_train, average=None)
    print(f"Per-Class Precision: {class_prec}, Recall: {class_rec}, F1: {class_f1}")
    log_writer.writerow([epoch+1, "Train", acc_train, prec_train, rec_train, f1_train, auc_train])

    if auc_train >= best_auc:
        best_auc = auc_train
        torch.save(model.state_dict(), "best_model.pth")
        print("💾 Saved best model based on training AUC.")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience and epoch >= 3:
            print("⏹️ Early stopping triggered.")
            break
    model.train()

train_val_log.close()

# === TEST: Majority Voting ===
model.eval()
preds_per_patient = defaultdict(list)
labels_per_patient = {}
with torch.no_grad():
    for image, label, pid in tqdm(test_loader, desc="Testing"):
        image = image.squeeze(1).to(device)
        logit = model(image)
        prob = torch.sigmoid(logit).item()
        preds_per_patient[pid[0]].append(prob)
        labels_per_patient[pid[0]] = label.item()

final_preds = {}
for pid, probs in preds_per_patient.items():
    avg_prob = np.mean(probs)
    final_preds[pid] = 1 if avg_prob >= 0.5 else 0

y_true = [int(labels_per_patient[pid]) for pid in final_preds]
y_pred = [int(pred) for pred in final_preds.values()]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, list(final_preds.values()))

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.savefig("confusion_matrix.png")
plt.show()

RocCurveDisplay.from_predictions(y_true, list(final_preds.values()))
plt.title("ROC Curve (Test Set)")
plt.savefig("roc_curve.png")
plt.show()

torch.save(model.state_dict(), "best_model.pth")
print("✅ Patient-Level Evaluation")

with open("test_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "true_label", "predicted_label", "probability"])
    for pid in final_preds:
        writer.writerow([pid, int(labels_per_patient[pid]), int(final_preds[pid]), round(np.mean(preds_per_patient[pid]), 4)])
print("📁 Saved test predictions to test_predictions.csv")
print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
