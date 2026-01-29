# 🚀 Temporal Classification with Pretrained Biomarker Encoder (DenseNet)
import os, json, torch, pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torchvision.models import densenet121
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.nn.utils.rnn import pad_sequence

# 📦 Temporal Dataset with Biomarker-encoded Features + Metadata
class BiomarkerTemporalDataset(Dataset):
    def __init__(self, json_path, image_root, encoder_weights, target_years=range(7, 14), transform=None):
        self.image_root = image_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Load pretrained encoder
        base = densenet121(weights=None)
        base.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(base.features, nn.AdaptiveAvgPool2d((1, 1)))

        state_dict = torch.load(encoder_weights)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("features."):
                new_key = k.replace("features.", "")
                new_state_dict[new_key] = v

        missing, unexpected = self.encoder[0].load_state_dict(new_state_dict, strict=False)
        print("[INFO] Encoder Load State:")
        print(" - Missing keys:", missing)
        print(" - Unexpected keys:", unexpected)

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.samples = []
        for eye_id, records in data.items():
            valid = [r for r in sorted(records, key=lambda x: x['year']) if 'BCVA' in r and r['BCVA'] is not None and r['scan_date']]
            if len(valid) < 2:
                continue

            for i in range(1, len(valid)):
                prev = valid[i - 1]
                curr = valid[i]
                if curr['year'] not in target_years:
                    continue

                delta_bcva = curr['BCVA'] - prev['BCVA']
                if delta_bcva >= 0.2:
                    label = 1
                elif delta_bcva <= -0.2:
                    label = 0
                else:
                    continue

                sequence = []
                for j in range(i):
                    entry = valid[j]
                    imgs = [os.path.join(self.image_root, p) for p in entry.get('images', []) if os.path.isfile(os.path.join(self.image_root, p))]
                    if not imgs:
                        continue
                    img_tensors = [self.transform(Image.open(img).convert('L')) for img in imgs]
                    cnn_input = torch.stack(img_tensors)
                    with torch.no_grad():
                        features = [self.encoder(img.unsqueeze(0)).squeeze().flatten() for img in cnn_input]
                    cnn_feat = torch.mean(torch.stack(features), dim=0)

                    bcva = torch.tensor([entry.get('BCVA', 0) / 2.0])
                    inj = torch.tensor([entry.get('n_inj', 0) / 12.0])
                    delta_days = (datetime.strptime(valid[j+1]['scan_date'], "%d/%m/%Y") - datetime.strptime(entry['scan_date'], "%d/%m/%Y")).days / 365.0 if j+1 < len(valid) else 0.0
                    scan_gap = torch.tensor([delta_days])

                    sequence.append(torch.cat([cnn_feat, bcva, inj, scan_gap]))

                if not sequence:
                    continue

                self.samples.append({
                    'id': eye_id,
                    'year': curr['year'],
                    'sequence': torch.stack(sequence),
                    'label': label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item['sequence'], torch.tensor(item['label']).float(), item['id'], item['year'], item['sequence'].shape[0]


def collate_fn(batch):
    xs, ys, ids, years, lengths = zip(*batch)
    padded = pad_sequence(xs, batch_first=True)
    return padded, torch.tensor(ys).float(), list(ids), list(years), list(lengths)

# 🧠 Model: BiLSTM + Attention
class TemporalAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1).unsqueeze(-1)
        return torch.sum(x * weights, dim=1)

class BiomarkerTemporalClassifier(nn.Module):
    def __init__(self, input_dim=1024+3, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = TemporalAttention(hidden_dim * 2)
        self.out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attn(lstm_out)
        return self.out(context).squeeze(1)

# 📊 Metrics

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
    }

def plot_conf_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm, index=["Worsened", "Improved"], columns=["Worsened", "Improved"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 🏋️ Training Pipeline

def train_temporal(json_path, image_root, encoder_weights, save_base, max_epochs=30):
    os.makedirs(save_base, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BiomarkerTemporalDataset(json_path, image_root, encoder_weights)
    print(f"Total samples loaded: {len(dataset)}")

    all_labels = [s['label'] for s in dataset.samples]
    all_groups = [s['id'] for s in dataset.samples]
    pos, neg = sum(all_labels), len(all_labels) - sum(all_labels)
    print(f"Positive: {pos}, Negative: {neg}, Pos weight: {neg / max(pos,1):.2f}")

    gkf = GroupKFold(n_splits=3)
    fold_results = []

    for fold, (trainval_idx, test_idx) in enumerate(gkf.split(all_labels, groups=all_groups)):
        print(f"\n=== Fold {fold+1}/3 ===")
        test_ids = [dataset.samples[i]['id'] for i in test_idx]
        print(f"Test Eyes: {len(set(test_ids))}, Years: {len(test_idx)}")

        save_dir = os.path.join(save_base, f"Fold{fold+1}")
        os.makedirs(save_dir, exist_ok=True)

        trainval_labels = [all_labels[i] for i in trainval_idx]
        trainval_groups = [all_groups[i] for i in trainval_idx]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=fold)
        train_idx_rel, val_idx_rel = next(splitter.split(trainval_labels, groups=trainval_groups))
        train_idx = [trainval_idx[i] for i in train_idx_rel]
        val_idx = [trainval_idx[i] for i in val_idx_rel]

        pos_weight = torch.tensor([sum(torch.tensor([dataset.samples[i]['label'] for i in train_idx]) == 0) / max(sum(torch.tensor([dataset.samples[i]['label'] for i in train_idx]) == 1), 1)]).to(device)

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=4, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=1, shuffle=False, collate_fn=collate_fn)

        model = BiomarkerTemporalClassifier().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_f1 = 0
        patience, patience_counter = 5, 0
        history = []

        for epoch in range(max_epochs):
            model.train()
            train_probs, train_targets = [], []
            for x, y, *_ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
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
                for x, y, *_ in val_loader:
                    x = x.to(device)
                    prob = torch.sigmoid(model(x))
                    val_probs.append(prob.item())
                    val_targets.append(y.item())

            val_preds = [int(p >= 0.5) for p in val_probs]
            val_metrics = compute_metrics(val_targets, val_preds, val_probs)

            history.append({"epoch": epoch+1, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
            print("Train →", ", ".join([f"{k.upper()}: {v:.4f}" for k, v in train_metrics.items()]))
            print("Val   →", ", ".join([f"{k.upper()}: {v:.4f}" for k, v in val_metrics.items()]))

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Evaluate on Test
        model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
        model.eval()
        test_probs, test_targets, test_ids = [], [], []
        with torch.no_grad():
            for x, y, ids, *_ in test_loader:
                x = x.to(device)
                prob = torch.sigmoid(model(x))
                test_probs.append(prob.item())
                test_targets.append(y.item())
                test_ids.append(ids[0])

        test_preds = [int(p >= 0.5) for p in test_probs]
        test_metrics = compute_metrics(test_targets, test_preds, test_probs)
        print("\nTest →", ", ".join([f"{k.upper()}: {v:.4f}" for k, v in test_metrics.items()]))

        pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
        pd.DataFrame({"ID": test_ids, "True": test_targets, "Pred": test_preds, "Prob": test_probs}).to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)

        plot_conf_matrix(train_targets, train_preds, os.path.join(save_dir, "train_conf_matrix.png"))
        plot_conf_matrix(val_targets, val_preds, os.path.join(save_dir, "val_conf_matrix.png"))
        plot_conf_matrix(test_targets, test_preds, os.path.join(save_dir, "test_conf_matrix.png"))

        fold_results.append({**{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}, **{f"test_{k}": v for k, v in test_metrics.items()}})

    print("\n=== Cross-Validation Summary ===")
    df = pd.DataFrame(fold_results)
    for phase in ['train', 'val', 'test']:
        print(f"{phase.upper()} METRICS")
        for metric, value in df[[col for col in df.columns if col.startswith(phase)]].mean().items():
            print(f"{metric.upper()}: {value:.4f}")
    df.to_csv(os.path.join(save_base, "cv_results.csv"), index=False)

# 🚀 Run
if __name__ == '__main__':
    encoder_path = r"D:\MS Computer Engineering\Thesis\Code\Weights\pretrained_amd_encoder.pth"
    save_base = f"Results_Temporal_Biomarker_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_temporal("AMD_Label_New.json", "E:/Labeled_PNGs", encoder_path, save_base)