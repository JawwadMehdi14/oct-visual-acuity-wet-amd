# === COMPLETE IMAGE CLASSIFIER WITH METRICS TRACKING ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import models
import json, os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           ConfusionMatrixDisplay, RocCurveDisplay)
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

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
        image = self.transform(image)
        return image, label, patient_id

# === ResNET MODEL ===
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         #resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
#         resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
#         self.flatten = nn.Flatten()
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = self.flatten(x)
#         return self.classifier(x)

# === DenseNET MODEL ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        weights = DenseNet121_Weights.DEFAULT
        densenet = densenet121(weights=weights)
        
        # Modify first conv layer to accept grayscale (1-channel) input
        densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Use the feature extractor up to the classification head
        self.feature_extractor = densenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Final pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 128),  # DenseNet121 final feature size = 1024
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        return self.classifier(x)

# === SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
json_file = "AMD_Label_New.json"
dataset = YearwiseImageDataset(json_file)
labels = [s[1] for s in dataset.samples]

# Split dataset (70% train, 15% val, 15% test)
train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.3, stratify=labels, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=42)

train_set = torch.utils.data.Subset(dataset, train_idx)
val_set = torch.utils.data.Subset(dataset, val_idx)
test_set = torch.utils.data.Subset(dataset, test_idx)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

print(f"\n📊 Dataset Splits:")
print(f"Training images: {len(train_set)}")
print(f"Validation images: {len(val_set)}")
print(f"Testing images: {len(test_set)}")

# Initialize model and training parameters
model = SimpleCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
best_auc = 0.0
patience = 3
patience_counter = 0

# Create metrics storage
metrics_history = {
    'epoch': [],
    'train_loss': [],
    'train_acc': [],
    'train_prec': [],
    'train_rec': [],
    'train_f1': [],
    'train_auc': [],
    'val_acc': [],
    'val_prec': [],
    'val_rec': [],
    'val_f1': [],
    'val_auc': []
}

# === TRAINING LOOP ===
print("\n🚀 Starting Training...")
for epoch in range(10):
    model.train()
    total_loss = 0
    train_preds, train_targets, train_probs = [], [], []
    
    # Training phase
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for images, labels, _ in loop:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        # Collect training metrics
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        train_preds.extend(preds)
        train_targets.extend(labels.cpu().numpy().flatten().astype(int))
        train_probs.extend(probs)
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    # Calculate training metrics
    train_acc = accuracy_score(train_targets, train_preds)
    train_prec = precision_score(train_targets, train_preds)
    train_rec = recall_score(train_targets, train_preds)
    train_f1 = f1_score(train_targets, train_preds)
    train_auc = roc_auc_score(train_targets, train_probs)
    epoch_loss = total_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_preds, val_targets, val_probs = [], [], []
    with torch.no_grad():
        for image, label, _ in val_loader:
            image = image.to(device)
            logit = model(image)
            prob = torch.sigmoid(logit).item()
            val_preds.append(int(prob >= 0.5))
            val_targets.append(label.item())
            val_probs.append(prob)
    
    # Calculate validation metrics
    val_acc = accuracy_score(val_targets, val_preds)
    val_prec = precision_score(val_targets, val_preds)
    val_rec = recall_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds)
    val_auc = roc_auc_score(val_targets, val_probs)
    
    # Store metrics
    metrics_history['epoch'].append(epoch+1)
    metrics_history['train_loss'].append(epoch_loss)
    metrics_history['train_acc'].append(train_acc)
    metrics_history['train_prec'].append(train_prec)
    metrics_history['train_rec'].append(train_rec)
    metrics_history['train_f1'].append(train_f1)
    metrics_history['train_auc'].append(train_auc)
    metrics_history['val_acc'].append(val_acc)
    metrics_history['val_prec'].append(val_prec)
    metrics_history['val_rec'].append(val_rec)
    metrics_history['val_f1'].append(val_f1)
    metrics_history['val_auc'].append(val_auc)
    
    # Print comprehensive metrics
    print(f"\n📊 Epoch {epoch+1} Metrics:")
    print("              Accuracy  Precision  Recall    F1       AUC")
    print(f"Training:     {train_acc:.4f}    {train_prec:.4f}     {train_rec:.4f}    {train_f1:.4f}  {train_auc:.4f}")
    print(f"Validation:   {val_acc:.4f}    {val_prec:.4f}     {val_rec:.4f}    {val_f1:.4f}  {val_auc:.4f}")
    print(f"Epoch Loss: {epoch_loss:.4f}")
    
    # Early stopping check
    if val_rec > best_auc:
        torch.save(model.state_dict(), "Year_Wise_Classification_28052025/best_model.pth")
        best_auc = val_rec
        patience_counter = 0
        print("💾 Saved new best model")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹️ Early stopping triggered (no improvement for {patience} epochs)")
            break


# === FINAL EVALUATION ===
print("\n🔍 Final Evaluation...")

# Load best model
best_model_path = "Year_Wise_Classification_28052025/best_model.pth"
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("✅ Loaded best model from early stopping.")
else:
    print("⚠️ No best model found (early stopping didn't save one), using last epoch model.")

model.eval()

# Test Evaluation
test_preds, test_targets, test_probs = [], [], []
test_results = []
with torch.no_grad():
    for image, label, pid in tqdm(test_loader, desc="Testing"):
        image = image.to(device)
        logit = model(image)
        prob = torch.sigmoid(logit).item()
        test_preds.append(int(prob >= 0.5))
        test_targets.append(label.item())
        test_probs.append(prob)
        test_results.append((pid[0], label.item(), int(prob >= 0.5), prob))

# Calculate test metrics
test_acc = accuracy_score(test_targets, test_preds)
test_prec = precision_score(test_targets, test_preds)
test_rec = recall_score(test_targets, test_preds)
test_f1 = f1_score(test_targets, test_preds)
test_auc = roc_auc_score(test_targets, test_probs)

# Print final results
print("\n✅ Final Test Results:")
print("              Accuracy  Precision  Recall    F1       AUC")
print(f"Test:        {test_acc:.4f}    {test_prec:.4f}     {test_rec:.4f}    {test_f1:.4f}  {test_auc:.4f}")

# === SAVE RESULTS ===
# Save all metrics to text file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"Year_Wise_Classification_28052025/training_metrics_{timestamp}.txt", "w") as f:
    f.write("=== TRAINING METRICS HISTORY ===\n")
    f.write(f"{'Epoch':<6} {'Loss':<8} {'Trn_Acc':<8} {'Trn_Prec':<8} {'Trn_Rec':<8} {'Trn_F1':<8} {'Trn_AUC':<8} ")
    f.write(f"{'Val_Acc':<8} {'Val_Prec':<8} {'Val_Rec':<8} {'Val_F1':<8} {'Val_AUC':<8}\n")
    
    for i in range(len(metrics_history['epoch'])):
        f.write(f"{metrics_history['epoch'][i]:<6} {metrics_history['train_loss'][i]:<8.4f} ")
        f.write(f"{metrics_history['train_acc'][i]:<8.4f} {metrics_history['train_prec'][i]:<8.4f} ")
        f.write(f"{metrics_history['train_rec'][i]:<8.4f} {metrics_history['train_f1'][i]:<8.4f} ")
        f.write(f"{metrics_history['train_auc'][i]:<8.4f} {metrics_history['val_acc'][i]:<8.4f} ")
        f.write(f"{metrics_history['val_prec'][i]:<8.4f} {metrics_history['val_rec'][i]:<8.4f} ")
        f.write(f"{metrics_history['val_f1'][i]:<8.4f} {metrics_history['val_auc'][i]:<8.4f}\n")
    
    f.write("\n=== FINAL TEST RESULTS ===\n")
    f.write(f"Accuracy:  {test_acc:.4f}\n")
    f.write(f"Precision: {test_prec:.4f}\n")
    f.write(f"Recall:    {test_rec:.4f}\n")
    f.write(f"F1 Score:  {test_f1:.4f}\n")
    f.write(f"AUC:       {test_auc:.4f}\n")

# Save predictions to files
pd.DataFrame(test_results, columns=["PatientID", "TrueLabel", "PredLabel", "Probability"]).to_csv(f"Year_Wise_Classification_28052025/test_predictions_{timestamp}.csv", index=False)

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics_history)
metrics_df.to_csv(f"Year_Wise_Classification_28052025/training_metrics_{timestamp}.csv", index=False)

# === VISUALIZATION ===
# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training CM
cm_train = confusion_matrix(train_targets, train_preds)
ConfusionMatrixDisplay(cm_train).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("Training Confusion Matrix")

# Validation CM
cm_val = confusion_matrix(val_targets, val_preds)
ConfusionMatrixDisplay(cm_val).plot(ax=axes[1], cmap="Blues")
axes[1].set_title("Validation Confusion Matrix")

# Test CM
cm_test = confusion_matrix(test_targets, test_preds)
ConfusionMatrixDisplay(cm_test).plot(ax=axes[2], cmap="Blues")
axes[2].set_title("Test Confusion Matrix")
plt.savefig(f"Year_Wise_Classification_28052025/confusion_matrices_{timestamp}.png")
plt.close()

# ROC Curve
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_predictions(test_targets, test_probs)
plt.title("Test ROC Curve")
plt.savefig(f"Year_Wise_Classification_28052025/roc_curve_{timestamp}.png")
plt.close()

print("\n💾 Saved all results:")
print(f"- Training metrics: training_metrics_{timestamp}.txt/.csv")
print(f"- Test predictions: test_predictions_{timestamp}.csv")
print(f"- Visualization files: confusion_matrices_{timestamp}.png, roc_curve_{timestamp}.png")