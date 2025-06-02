import os, json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import pandas as pd
from tqdm import tqdm

class PatientEyeTemporalDataset(Dataset):
    def __init__(self, json_path, image_root, transform=None, max_years=13):
        self.image_root = image_root
        self.max_years = max_years
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.samples = []
        for patient_eye, records in data.items():
            years = sorted(records, key=lambda x: x['year'])
            labels_found = [r['label'] for r in years if 'label' in r]
            if not labels_found:
                continue
            label = 1 if labels_found[0].lower() == 'good' else 0

            year_features = []
            for r in years:
                imgs = [os.path.join(self.image_root, i) for i in r.get("images", []) if os.path.isfile(os.path.join(self.image_root, i))]
                if imgs:
                    year_features.append(imgs)

            if year_features:
                self.samples.append({
                    'patient_eye': patient_eye,
                    'image_paths': year_features,
                    'length': len(year_features),
                    'label': label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tensors = []
        for year_imgs in sample['image_paths']:
            imgs = [self.transform(Image.open(p).convert('L')) for p in year_imgs]
            year_tensor = torch.mean(torch.stack(imgs), dim=0)
            tensors.append(year_tensor)

        padded = tensors + [torch.zeros_like(tensors[0])] * (self.max_years - len(tensors))
        sequence_tensor = torch.stack(padded)
        return sequence_tensor, sample['length'], sample['label'], sample['patient_eye']

class ResNetLSTMClassifier(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=128):
        super().__init__()
        # ✅ Use pretrained weights
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(
            input_size=feature_dim, hidden_size=hidden_dim, 
            num_layers=2, batch_first=True, dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.BatchNorm1d(1)
        )

    def forward(self, x_seq, lengths):
        B, T, C, H, W = x_seq.shape
        x_seq = x_seq.view(B * T, C, H, W)
        feats = self.flatten(self.cnn(x_seq)).view(B, T, -1)
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        return self.classifier(hn[-1]).squeeze(1)

def collate_fn(batch):
    sequences, lengths, labels, ids = zip(*batch)
    seq_tensor = torch.stack(sequences)
    lengths_tensor = torch.tensor(lengths)
    labels_tensor = torch.tensor(labels).float()
    return seq_tensor, lengths_tensor, labels_tensor, ids

# === SETUP ===
image_root = "E:/Labeled_PNGs"
json_path = "AMD_Label_New.json"
save_dir = f"Temporal_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PatientEyeTemporalDataset(json_path, image_root)

# === SPLIT ===
all_labels = [s['label'] for s in dataset.samples]
train_ids, temp_ids = train_test_split(list(range(len(dataset))), test_size=0.3, stratify=all_labels, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, stratify=[all_labels[i] for i in temp_ids], random_state=42)

train_set = torch.utils.data.Subset(dataset, train_ids)
val_set = torch.utils.data.Subset(dataset, val_ids)
test_set = torch.utils.data.Subset(dataset, test_ids)

train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

# === PRINT SPLIT INFO ===
print(f"\n📊 Dataset Splitting Info:")
print(f"Total samples: {len(dataset)}")
print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")

def label_dist(subset):
    labels = [dataset.samples[i]['label'] for i in subset.indices]
    return sum(labels), len(labels) - sum(labels)

print(f"Train Label Dist: {label_dist(train_set)}")
print(f"Val Label Dist:   {label_dist(val_set)}")
print(f"Test Label Dist:  {label_dist(test_set)}")

# === MODEL + TRAINING ===
model = ResNetLSTMClassifier().to(device)
for param in model.cnn.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()
patience, best_f1, patience_counter = 3, 0, 0
history = []

for epoch in range(15):
    # 🔓 Unfreeze CNN after 3 epochs
    if epoch == 2:
        print("🔓 Unfreezing CNN layers for fine-tuning...")
        for param in model.cnn.parameters():
            param.requires_grad = True
    model.train()
    train_losses, train_probs, train_targets = [], [], []

    for x, lengths, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(x, lengths)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_probs.extend(torch.sigmoid(out).detach().cpu().numpy())
        train_targets.extend(labels.cpu().numpy())

    model.eval()
    val_probs, val_targets = [], []
    with torch.no_grad():
        for x, lengths, labels, _ in val_loader:
            x, lengths = x.to(device), lengths.to(device)
            out = model(x, lengths)
            val_probs.append(torch.sigmoid(out).item())
            val_targets.append(labels.item())

    train_preds = [int(p >= 0.5) for p in train_probs]
    val_preds = [int(p >= 0.5) for p in val_probs]

    train_metrics = {
        "acc": accuracy_score(train_targets, train_preds),
        "prec": precision_score(train_targets, train_preds),
        "rec": recall_score(train_targets, train_preds),
        "f1": f1_score(train_targets, train_preds),
        "auc": roc_auc_score(train_targets, train_probs)
    }

    val_metrics = {
        "acc": accuracy_score(val_targets, val_preds),
        "prec": precision_score(val_targets, val_preds),
        "rec": recall_score(val_targets, val_preds),
        "f1": f1_score(val_targets, val_preds),
        "auc": roc_auc_score(val_targets, val_probs)
    }

    history.append({
        "epoch": epoch+1, "loss": sum(train_losses)/len(train_losses), **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}
    })

    print(f"\n📊 Epoch {epoch+1} Metrics")
    print("Train → ACC: {:.4f}, PREC: {:.4f}, REC: {:.4f}, F1: {:.4f}, AUC: {:.4f}".format(train_metrics['acc'], train_metrics['prec'], train_metrics['rec'], train_metrics['f1'], train_metrics['auc']))
    print("Val   → ACC: {:.4f}, PREC: {:.4f}, REC: {:.4f}, F1: {:.4f}, AUC: {:.4f}".format(val_metrics['acc'], val_metrics['prec'], val_metrics['rec'], val_metrics['f1'], val_metrics['auc']))

    if val_metrics['f1'] > best_f1:
        best_f1 = val_metrics['f1']
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping")
            break

# === TESTING ===
model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
model.eval()

test_probs, test_targets, test_ids = [], [], []
with torch.no_grad():
    for x, lengths, labels, ids in tqdm(test_loader, desc="Testing"):
        x, lengths = x.to(device), lengths.to(device)
        out = model(x, lengths)
        prob = torch.sigmoid(out).item()
        test_probs.append(prob)
        test_targets.append(labels.item())
        test_ids.append(ids[0])

test_preds = [int(p >= 0.5) for p in test_probs]
test_metrics = {
    "acc": accuracy_score(test_targets, test_preds),
    "prec": precision_score(test_targets, test_preds),
    "rec": recall_score(test_targets, test_preds),
    "f1": f1_score(test_targets, test_preds),
    "auc": roc_auc_score(test_targets, test_probs)
}

print("\n✅ Test Results:")
for k, v in test_metrics.items():
    print(f"{k.upper()}: {v:.4f}")

# === SAVE ===
pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_metrics.csv"), index=False)
pd.DataFrame({"PatientEyeID": test_ids, "True": test_targets, "Pred": test_preds, "Prob": test_probs}).to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)
with open(os.path.join(save_dir, "test_metrics.txt"), "w") as f:
    for k, v in test_metrics.items():
        f.write(f"{k.upper()}: {v:.4f}\n")
