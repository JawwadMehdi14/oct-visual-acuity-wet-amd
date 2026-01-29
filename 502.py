# 📚 Imports
import os, json, torch, pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torchvision.models import densenet121
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from torch.nn.utils.rnn import pad_sequence

# 📦 Temporal Dataset with Delta-BCVA Labeling (gap-tolerant)
class TemporalOCTDataset(Dataset):
    def __init__(self, json_path, image_root, transform=None):
        self.image_root = image_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.samples = []
        for eye_id, records in data.items():
            valid_records = [r for r in sorted(records, key=lambda x: x['year']) if 'BCVA' in r and r['BCVA'] is not None]
            if len(valid_records) < 2:
                continue

            for i in range(1, len(valid_records)):
                for j in range(i - 1, -1, -1):
                    prev = valid_records[j]
                    curr = valid_records[i]
                    if 'BCVA' in prev and 'BCVA' in curr:
                        delta_bcva = curr['BCVA'] - prev['BCVA']
                        label = 1 if delta_bcva >= 0.1 else 0

                        images = [os.path.join(self.image_root, img) for img in curr.get('images', []) if os.path.isfile(os.path.join(self.image_root, img))]
                        if not images:
                            continue

                        self.samples.append({
                            'id': eye_id,
                            'year': curr['year'],
                            'years_back': [r['year'] for r in valid_records[:i]],
                            'images_per_year': [r['images'] for r in valid_records[:i]],
                            'label': label,
                            'length': len(valid_records[:i])
                        })
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        seq_tensors = []
        for image_list in item['images_per_year']:
            imgs = [self.transform(Image.open(os.path.join(self.image_root, img)).convert('L'))
                    for img in image_list if os.path.isfile(os.path.join(self.image_root, img))]
            if not imgs:
                imgs = [torch.zeros(1, 224, 224)]
            avg_img = torch.mean(torch.stack(imgs), dim=0)
            seq_tensors.append(avg_img)

        return torch.stack(seq_tensors), torch.tensor(item['label']).float(), item['id'], item['year'], item['length']


# Collate Function for Padded Sequences
def collate_fn(batch):
    xs, ys, ids, years, lengths = zip(*batch)
    padded = pad_sequence(xs, batch_first=True)
    return padded, torch.tensor(ys).float(), list(ids), list(years), list(lengths)


# Model Definition: DenseNet + LSTM
class DenseLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=1):
        super().__init__()
        base = densenet121(weights="IMAGENET1K_V1")
        base.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.flatten(self.pool(self.encoder(x)))
        feats = feats.view(B, T, -1)
        _, (h_n, _) = self.lstm(feats)
        return self.classifier(h_n[-1]).squeeze(1)


# Evaluation Metrics

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
    df = pd.DataFrame(cm, index=["Not Improved", "Improved"], columns=["Not Improved", "Improved"])
    plt.figure(figsize=(5,4))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Training with Cross-Validation

def train_cv(json_path, image_root, save_base, max_epochs=30):
    os.makedirs(save_base, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TemporalOCTDataset(json_path, image_root)
    if len(dataset.samples) < 3:
        raise ValueError(f"Insufficient samples: only {len(dataset.samples)} valid sequences found. Check image availability and BCVA entries.")

    all_labels = [s['label'] for s in dataset.samples]
    all_groups = [s['id'] for s in dataset.samples]
    gkf = GroupKFold(n_splits=3)
    fold_results = []

    for fold, (trainval_idx, test_idx) in enumerate(gkf.split(all_labels, groups=all_groups)):
        print(f"\n=== Fold {fold+1}/3 ===")
        test_ids = [dataset.samples[i]['id'] for i in test_idx]
        print(f"Test  Eyes: {len(set(test_ids))}, Years: {len(test_idx)}")

        save_dir = os.path.join(save_base, f"Fold{fold+1}")
        os.makedirs(save_dir, exist_ok=True)

        trainval_labels = [all_labels[i] for i in trainval_idx]
        trainval_groups = [all_groups[i] for i in trainval_idx]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=fold)
        train_idx_rel, val_idx_rel = next(splitter.split(trainval_labels, groups=trainval_groups))
        train_idx = [trainval_idx[i] for i in train_idx_rel]
        val_idx = [trainval_idx[i] for i in val_idx_rel]

        train_labels_only = [dataset.samples[i]['label'] for i in train_idx]
        pos_weight = torch.tensor([sum(torch.tensor(train_labels_only) == 0) / sum(torch.tensor(train_labels_only) == 1)]).to(device)

        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=4, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=1, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=1, shuffle=False, collate_fn=collate_fn)

        model = DenseLSTMClassifier().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_f1 = 0
        patience = 5
        patience_counter = 0
        history = []

        for epoch in range(max_epochs):
            model.train()
            train_probs, train_targets = [], []
            for x, y, _, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                probs = torch.sigmoid(logits)
                train_probs.extend(probs.detach().cpu().numpy())
                train_targets.extend(y.cpu().numpy())

            scheduler.step()
            train_preds = [int(p >= 0.5) for p in train_probs]
            train_metrics = compute_metrics(train_targets, train_preds, train_probs)

            model.eval()
            val_probs, val_targets = [], []
            with torch.no_grad():
                for x, y, _, _, _ in val_loader:
                    x = x.to(device)
                    logits = model(x)
                    prob = torch.sigmoid(logits)
                    val_probs.append(prob.item())
                    val_targets.append(y.item())
            val_preds = [int(p >= 0.5) for p in val_probs]
            val_metrics = compute_metrics(val_targets, val_preds, val_probs)

            history.append({"epoch": epoch+1, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
            print(f"Train → " + ", ".join([f"{k.upper()}: {v:.4f}" for k, v in train_metrics.items()]))
            print(f"Val   → " + ", ".join([f"{k.upper()}: {v:.4f}" for k, v in val_metrics.items()]))

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
        model.eval()
        test_probs, test_targets, test_ids = [], [], []
        with torch.no_grad():
            for x, y, ids, _, _ in test_loader:
                x = x.to(device)
                logits = model(x)
                prob = torch.sigmoid(logits)
                test_probs.append(prob.item())
                test_targets.append(y.item())
                test_ids.append(ids[0])

        test_preds = [int(p >= 0.5) for p in test_probs]
        test_metrics = compute_metrics(test_targets, test_preds, test_probs)
        print("\nTest → " + ", ".join([f"{k.upper()}: {v:.4f}" for k, v in test_metrics.items()]))

        pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
        pd.DataFrame({"ID": test_ids, "True": test_targets, "Pred": test_preds, "Prob": test_probs}).to_csv(
            os.path.join(save_dir, "test_predictions.csv"), index=False)

        plot_conf_matrix(train_targets, train_preds, os.path.join(save_dir, "train_conf_matrix.png"))
        plot_conf_matrix(val_targets, val_preds, os.path.join(save_dir, "val_conf_matrix.png"))
        plot_conf_matrix(test_targets, test_preds, os.path.join(save_dir, "test_conf_matrix.png"))

        fold_results.append({
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()}
        })

    print("\n=== Cross-Validation Summary ===")
    df = pd.DataFrame(fold_results)
    print("Mean Metrics Across Folds:")
    for phase in ['train', 'val', 'test']:
        print(f"{phase.upper()} METRICS")
        phase_df = df[[col for col in df.columns if col.startswith(phase)]]
        for metric, value in phase_df.mean().items():
            print(f"{metric.upper()}: {value:.4f}")
    df.to_csv(os.path.join(save_base, "cv_results.csv"), index=False)


# 🚀 Run
if __name__ == '__main__':
    save_base = f"Results_Temporal_DenseNet_LSTM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_cv("AMD_Label_New.json", "E:/Labeled_PNGs", save_base)
