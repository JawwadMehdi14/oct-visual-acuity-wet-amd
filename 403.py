import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from PatientEyeDatasetLoader1 import PatientEyeDataset
from DensePatientClassifier2 import PatientClassifier
from tqdm import tqdm
import os
import random
import numpy as np
import csv

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
labels = [item[1] for item in dataset]  # Extract labels from dataset

# === STRATIFIED SPLIT ===
indices = list(range(len(dataset)))
train_idx, temp_idx, train_labels, temp_labels = train_test_split(
    indices, labels, stratify=labels, test_size=1 - train_ratio, random_state=seed
)
val_ratio_adjusted = val_ratio / (1 - train_ratio)
val_idx, test_idx, _, _ = train_test_split(
    temp_idx, temp_labels, stratify=temp_labels, test_size=1 - val_ratio_adjusted, random_state=seed
)

train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
test_set = Subset(dataset, test_idx)

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# === MODEL SETUP ===
model = PatientClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float("inf")

# === TRAIN LOOP ===
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", leave=False)

    for image_tensor, label in loop:
        image_tensor = image_tensor.squeeze(0).to(device)
        label = label.float().unsqueeze(0).to(device)
        optimizer.zero_grad()

        # Chunked feature extraction
        chunk_size = 8
        features = []
        for i in range(0, image_tensor.size(0), chunk_size):
            chunk = image_tensor[i:i+chunk_size].to(device)
            chunk = chunk.repeat(1, 3, 1, 1)
            with torch.no_grad():
                f = model.feature_extractor(chunk)
                f = model.pool(f)
                f = model.flatten(f)
            features.append(f.detach())

        features = torch.cat(features, dim=0)
        patient_repr = torch.mean(features, dim=0)
        output = model.classifier(patient_repr).unsqueeze(0)

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
                f = model.pool(f)
                f = model.flatten(f)
                features.append(f)

            features = torch.cat(features, dim=0)
            patient_repr = torch.mean(features, dim=0)
            output = model.classifier(patient_repr).unsqueeze(0)

            loss = criterion(output, label)
            val_loss += loss.item()

            preds.append(output.item())
            targets.append(label.item())

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} → Val Loss: {avg_val_loss:.4f}")

    bin_preds = [1 if p >= 0.5 else 0 for p in preds]
    acc = accuracy_score(targets, bin_preds)
    prec = precision_score(targets, bin_preds)
    rec = recall_score(targets, bin_preds)
    auc = roc_auc_score(targets, preds)

    print(f"Validation Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUC: {auc:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print("✅ Best model saved.")

print("\nTraining complete. Best model stored at:", save_path)

# === TEST SET EVALUATION ===
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
            f = model.pool(f)
            f = model.flatten(f)
            features.append(f)

        features = torch.cat(features, dim=0)
        patient_repr = torch.mean(features, dim=0)
        output = model.classifier(patient_repr).unsqueeze(0)

        preds.append(output.item())
        targets.append(label.item())

bin_preds = [1 if p >= 0.5 else 0 for p in preds]
acc = accuracy_score(targets, bin_preds)
prec = precision_score(targets, bin_preds)
rec = recall_score(targets, bin_preds)
auc = roc_auc_score(targets, preds)
print(f"\nTest Set Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUC: {auc:.4f}")

# === SAVE TEST PREDICTIONS TO CSV ===
with open("test_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "true_label", "predicted_label", "probability"])
    for i, (p, t) in enumerate(zip(preds, targets)):
        writer.writerow([i, int(t), int(p >= 0.5), round(p, 4)])

print("\n📁 Predictions saved to: test_predictions.csv")
