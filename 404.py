import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from PatientEyeDatasetLoader1 import PatientEyeDataset
import torchvision.models as models

class PatientClassifier(nn.Module):
    def __init__(self):
        super(PatientClassifier, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the final FC layer
        self.feature_extractor = nn.Sequential(*modules)  # Output: (512, 1, 1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

from tqdm import tqdm
import os
import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, f1_score, precision_recall_fscore_support

# === FIXED SEED FOR REPRODUCIBILITY ===
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# === CONFIGURATION ===
json_file = "flattened_all_years.json"
image_dir = "E:\\Labeled_PNGs"
save_path = "best_model.pth"
num_epochs = 10
learning_rate = 1e-4
train_ratio = 0.7
val_ratio = 0.15

# === LOAD DATASET ===
dataset = PatientEyeDataset(json_file, image_root=image_dir)
labels = dataset.labels if hasattr(dataset, 'labels') else [item[1] for item in dataset]  # Optimized label access if available

# === STRATIFIED SPLIT ===
split_file = "split_indices.pth"
if os.path.exists(split_file):
    print("🔁 Loading precomputed stratified split...")
    split = torch.load(split_file)
    train_idx, val_idx, test_idx = split['train'], split['val'], split['test']
else:
    print("🔄 Creating new stratified split and saving...")
    indices = list(range(len(dataset)))
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices, labels, stratify=labels, test_size=1 - train_ratio, random_state=seed
    )
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, stratify=temp_labels, test_size=1 - val_ratio_adjusted, random_state=seed
    )
    torch.save({'train': train_idx, 'val': val_idx, 'test': test_idx}, split_file)

train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
test_set = Subset(dataset, test_idx)

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# === MODEL SETUP ===
model = PatientClassifier()
for param in model.feature_extractor.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === MANUAL CLASS WEIGHT ADJUSTMENT ===
pos_weight = torch.tensor([3.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float("inf")

# === TRAIN LOOP ===
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", leave=False)

    for image_tensor, label in loop:
        torch.cuda.empty_cache()  # Clear unused memory
        image_tensor = image_tensor.squeeze(0).to(device)
        label = label.float().unsqueeze(0).to(device)
        optimizer.zero_grad()

        chunk_size = 2
        features = []
        for i in range(0, image_tensor.size(0), chunk_size):
            chunk = image_tensor[i:i+chunk_size].to(device)
            chunk = chunk.repeat(1, 3, 1, 1)
            f = model.feature_extractor(chunk)
            f = model.flatten(f)
            features.append(f)

        features = torch.cat(features, dim=0)
        del chunk, f
        patient_repr = torch.mean(features, dim=0)
        output = model.classifier(patient_repr).unsqueeze(0)
        output = torch.sigmoid(output)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} → Train Loss: {avg_train_loss:.4f}")

    # === VALIDATION ===
    model.eval()
    val_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for image_tensor, label in val_loader:
            image_tensor = image_tensor.squeeze(0).to(device)
            label = label.float().unsqueeze(0).to(device)

            features = []
            for i in range(0, image_tensor.size(0), chunk_size):
                chunk = image_tensor[i:i+chunk_size].to(device)
                chunk = chunk.repeat(1, 3, 1, 1)
                f = model.feature_extractor(chunk)
                f = model.flatten(f)
                features.append(f)

            features = torch.cat(features, dim=0)
            patient_repr = torch.mean(features, dim=0)
            output = model.classifier(patient_repr).unsqueeze(0)
            output = torch.sigmoid(output)

            loss = criterion(output, label)
            val_loss += loss.item()

            preds.append(output.item())
            targets.append(label.item())

    avg_val_loss = val_loss / len(val_loader)

    # === THRESHOLD SEARCH ===
    best_thresh, best_f1 = 0.5, 0.0
    thresholds = np.linspace(0.1, 0.9, 9)
    for t in thresholds:
        test_preds = [1 if p >= t else 0 for p in preds]
        f1_t = f1_score(targets, test_preds)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t

    print(f"🔎 Optimal Threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")
    bin_preds = [1 if p >= best_thresh else 0 for p in preds]
    acc = accuracy_score(targets, bin_preds)
    prec = precision_score(targets, bin_preds)
    rec = recall_score(targets, bin_preds)
    auc = roc_auc_score(targets, preds)
    f1 = f1_score(targets, bin_preds)
    class_prec, class_rec, class_f1, _ = precision_recall_fscore_support(targets, bin_preds, average=None)

    print(f"Epoch {epoch+1} → Val Loss: {avg_val_loss:.4f}")
    print("Predicted label counts:", {0: bin_preds.count(0), 1: bin_preds.count(1)})
    print(f"Validation Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
    print(f"Per-Class Precision: {class_prec}, Recall: {class_rec}, F1: {class_f1}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print("✅ Best model saved.")

print("\n🔍 Evaluating on Test Set...")
model.load_state_dict(torch.load(save_path))
model.eval()
preds, targets = [], []

with torch.no_grad():
    for image_tensor, label in tqdm(test_loader, desc="Testing"):
        image_tensor = image_tensor.squeeze(0).to(device)
        label = label.float().unsqueeze(0).to(device)

        features = []
        for i in range(0, image_tensor.size(0), chunk_size):
            chunk = image_tensor[i:i+chunk_size].to(device)
            chunk = chunk.repeat(1, 3, 1, 1)
            f = model.feature_extractor(chunk)
            f = model.flatten(f)
            features.append(f)

        features = torch.cat(features, dim=0)
        patient_repr = torch.mean(features, dim=0)
        output = model.classifier(patient_repr).unsqueeze(0)
        output = torch.sigmoid(output)

        preds.append(output.item())
        targets.append(label.item())

bin_preds = [1 if p >= 0.5 else 0 for p in preds]
acc = accuracy_score(targets, bin_preds)
prec = precision_score(targets, bin_preds)
rec = recall_score(targets, bin_preds)
auc = roc_auc_score(targets, preds)
print(f"\nTest Set Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUC: {auc:.4f}")

# === CONFUSION MATRIX & ROC CURVE ===
cm = confusion_matrix(targets, bin_preds)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.savefig("confusion_matrix.png")
plt.show()

RocCurveDisplay.from_predictions(targets, preds)
plt.title("ROC Curve (Test Set)")
plt.savefig("roc_curve.png")
plt.show()

# === SAVE TEST PREDICTIONS TO CSV ===
with open("test_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "true_label", "predicted_label", "probability"])
    for i, (p, t) in enumerate(zip(preds, targets)):
        writer.writerow([i, int(t), int(p >= 0.5), round(p, 4)])

print("\n📁 Predictions saved to: test_predictions.csv")
