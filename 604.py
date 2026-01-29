# === COMPLETE IMAGE CLASSIFIER WITH METRICS TRACKING ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights, resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
from torchvision import models
import json, os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
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
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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

# === DenseNET MODEL ===
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         weights = DenseNet121_Weights.DEFAULT
#         densenet = densenet121(weights=weights)
        
#         # Modify first conv layer to accept grayscale (1-channel) input
#         densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
#         # Use the feature extractor up to the classification head
#         self.feature_extractor = densenet.features
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Final pooling
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(1024, 128),  # DenseNet121 final feature size = 1024
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = self.pool(x)
#         return self.classifier(x)

# === SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
json_file = "AMD_Label_New.json"
dataset = YearwiseImageDataset(json_file)
labels = [s[1] for s in dataset.samples]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# To collect fold-wise results
all_fold_metrics = []
all_fold_test_predictions = []

for fold, (trainval_idx, test_idx) in enumerate(skf.split(range(len(dataset)), labels)):
    print(f"\n==============================")
    print(f"✅ Fold {fold+1}/3")
    print(f"==============================")

    # Extract test set
    test_set = torch.utils.data.Subset(dataset, test_idx)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Now split trainval into train and val
    trainval_labels = [labels[i] for i in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.18,   # ~18% of 67% ≈ 12% of full
        stratify=trainval_labels,
        random_state=fold
    )

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    print(f"Split sizes — Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # ============ Training as in your code ============

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_auc = 0.0
    patience = 3
    patience_counter = 0

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

    print("\n🚀 Starting Training...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        train_preds, train_targets, train_probs = [], [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, labels_, _ in loop:
            images = images.to(device)
            labels_ = labels_.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels_)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            train_preds.extend(preds)
            train_targets.extend(labels_.cpu().numpy().flatten().astype(int))
            train_probs.extend(probs)
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_acc = accuracy_score(train_targets, train_preds)
        train_prec = precision_score(train_targets, train_preds)
        train_rec = recall_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds)
        train_auc = roc_auc_score(train_targets, train_probs)
        epoch_loss = total_loss / len(train_loader)

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

        val_acc = accuracy_score(val_targets, val_preds)
        val_prec = precision_score(val_targets, val_preds)
        val_rec = recall_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds)
        val_auc = roc_auc_score(val_targets, val_probs)

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

        print(f"\n📊 Epoch {epoch+1} Metrics:")
        print(f"Training:     ACC {train_acc:.4f}  PREC {train_prec:.4f}  REC {train_rec:.4f}  F1 {train_f1:.4f}  AUC {train_auc:.4f}")
        print(f"Validation:   ACC {val_acc:.4f}  PREC {val_prec:.4f}  REC {val_rec:.4f}  F1 {val_f1:.4f}  AUC {val_auc:.4f}")

        if val_rec > best_auc:
            os.makedirs(f"Year_Wise_Classification_Fold{fold+1}", exist_ok=True)
            torch.save(model.state_dict(), f"Year_Wise_Classification_Fold{fold+1}/best_model.pth")
            best_auc = val_rec
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered (no improvement for {patience} epochs)")
                break

    model.load_state_dict(torch.load(f"Year_Wise_Classification_Fold{fold+1}/best_model.pth"))
    model.eval()

    # TEST EVALUATION
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

    test_acc = accuracy_score(test_targets, test_preds)
    test_prec = precision_score(test_targets, test_preds)
    test_rec = recall_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds)
    test_auc = roc_auc_score(test_targets, test_probs)

    fold_metrics = {
        "train_acc": train_acc, "train_prec": train_prec, "train_rec": train_rec, "train_f1": train_f1, "train_auc": train_auc,
        "val_acc": val_acc, "val_prec": val_prec, "val_rec": val_rec, "val_f1": val_f1, "val_auc": val_auc,
        "test_acc": test_acc, "test_prec": test_prec, "test_rec": test_rec, "test_f1": test_f1, "test_auc": test_auc
    }
    all_fold_metrics.append(fold_metrics)
    all_fold_test_predictions.extend(test_results)

    # SAVE PER FOLD
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"Year_Wise_Classification_Fold{fold+1}", exist_ok=True)
    pd.DataFrame(metrics_history).to_csv(f"Year_Wise_Classification_Fold{fold+1}/training_metrics_{timestamp}.csv", index=False)
    pd.DataFrame(test_results, columns=["PatientID", "TrueLabel", "PredLabel", "Probability"]).to_csv(f"Year_Wise_Classification_Fold{fold+1}/test_predictions_{timestamp}.csv", index=False)

    # Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ConfusionMatrixDisplay(confusion_matrix(train_targets, train_preds)).plot(ax=axes[0], cmap="Blues")
    axes[0].set_title("Training Confusion Matrix")
    ConfusionMatrixDisplay(confusion_matrix(val_targets, val_preds)).plot(ax=axes[1], cmap="Blues")
    axes[1].set_title("Validation Confusion Matrix")
    ConfusionMatrixDisplay(confusion_matrix(test_targets, test_preds)).plot(ax=axes[2], cmap="Blues")
    axes[2].set_title("Test Confusion Matrix")
    plt.savefig(f"Year_Wise_Classification_Fold{fold+1}/confusion_matrices_{timestamp}.png")
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(test_targets, test_probs)
    plt.title("Test ROC Curve")
    plt.savefig(f"Year_Wise_Classification_Fold{fold+1}/roc_curve_{timestamp}.png")
    plt.close()

# === Save mean and std across folds ===
df_results = pd.DataFrame(all_fold_metrics)
df_results.to_csv("Year_Wise_Classification_CV_All_Folds.csv", index=False)
mean_metrics = df_results.mean()
std_metrics = df_results.std()
mean_std = pd.DataFrame({"Mean": mean_metrics, "Std": std_metrics})
mean_std.to_csv("Year_Wise_Classification_CV_Mean_Std.csv")
print("\n✅ Saved mean and std across folds.")