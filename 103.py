import os, json, torch, pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

# 📦 Dataset
class EyeWiseTemporalDataset(Dataset):
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
        for eye_id, records in data.items():
            records = sorted(records, key=lambda x: x['year'])
            labels = [r['label'] for r in records if 'label' in r]
            if not labels:
                continue
            label = 1 if labels[0].lower() == 'good' else 0

            scan_groups = []
            for r in records:
                images = [os.path.join(self.image_root, img) for img in r.get("images", [])]
                images = [img for img in images if os.path.isfile(img)]
                if images:
                    scan_groups.append(images)
            if scan_groups:
                self.samples.append({'id': eye_id, 'scans': scan_groups, 'length': len(scan_groups), 'label': label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        scan_tensors = []
        for group in item['scans']:
            imgs = [self.transform(Image.open(p).convert('L')) for p in group]
            scan_tensor = torch.mean(torch.stack(imgs), dim=0)
            scan_tensors.append(scan_tensor)
        padded = scan_tensors + [torch.zeros_like(scan_tensors[0])] * (self.max_years - len(scan_tensors))
        return torch.stack(padded), item['length'], item['label'], item['id']

def collate_fn(batch):
    xs, lens, ys, ids = zip(*batch)
    return torch.stack(xs), torch.tensor(lens), torch.tensor(ys).float(), ids

# 🧠 Model
class TemporalResNet(nn.Module):
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

# 📈 Metrics
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob)
    }

def plot_conf_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm, index=["Bad", "Good"], columns=["Bad", "Good"])
    plt.figure(figsize=(5,4))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True"), plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 🚀 Training
def train_pipeline(json_path, image_root, save_dir, max_epochs=30):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EyeWiseTemporalDataset(json_path, image_root)
    all_labels = [s['label'] for s in dataset.samples]
    all_groups = [s['id'] for s in dataset.samples]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(splitter.split(all_labels, groups=all_groups))
    val_idx, test_idx = next(GroupShuffleSplit(test_size=0.5, n_splits=1).split(
        [all_labels[i] for i in temp_idx], groups=[all_groups[i] for i in temp_idx]))
    val_idx = [temp_idx[i] for i in val_idx]
    test_idx = [temp_idx[i] for i in test_idx]

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = TemporalResNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.BCELoss()

    best_f1 = 0
    history, patience, patience_counter = [], 7, 0

    for epoch in range(max_epochs):
        model.train()
        train_probs, train_targets = [], []
        for x, lens, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, lens)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_probs.extend(out.detach().cpu().numpy())
            train_targets.extend(y.cpu().numpy())

        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for x, lens, y, _ in val_loader:
                x, lens = x.to(device), lens.to(device)
                out = model(x, lens)
                val_probs.append(out.item())
                val_targets.append(y.item())

        train_preds = [int(p >= 0.5) for p in train_probs]
        val_preds = [int(p >= 0.5) for p in val_probs]
        train_metrics = compute_metrics(train_targets, train_preds, train_probs)
        val_metrics = compute_metrics(val_targets, val_preds, val_probs)
        scheduler.step()

        history.append({
            "epoch": epoch+1,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        print(f"\n📊 Epoch {epoch+1} Metrics")
        print("Train → " + ", ".join([f"{k.upper()}: {v:.4f}" for k, v in train_metrics.items()]))
        print("Val   → " + ", ".join([f"{k.upper()}: {v:.4f}" for k, v in val_metrics.items()]))

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping")
                break

    # ✅ Testing
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    model.eval()
    test_probs, test_targets, test_ids = [], [], []
    with torch.no_grad():
        for x, lens, y, ids in tqdm(test_loader, desc="Testing"):
            x, lens = x.to(device), lens.to(device)
            out = model(x, lens)
            test_probs.append(out.item())
            test_targets.append(y.item())
            test_ids.append(ids[0])
    test_preds = [int(p >= 0.5) for p in test_probs]
    test_metrics = compute_metrics(test_targets, test_preds, test_probs)

    print("\n✅ Test Results:")
    for k, v in test_metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
    pd.DataFrame({"ID": test_ids, "True": test_targets, "Pred": test_preds, "Prob": test_probs}).to_csv(
        os.path.join(save_dir, "test_predictions.csv"), index=False)
    plot_conf_matrix(test_targets, test_preds, os.path.join(save_dir, "confusion_matrix.png"))

# 🚀 Run
save_dir = f"Results_TemporalResNet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
train_pipeline("AMD_Label_New.json", "E:/Labeled_PNGs", save_dir)
