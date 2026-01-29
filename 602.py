import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import models, transforms
from collections import Counter
import seaborn as sns
import pandas as pd
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --- Focal Loss with Label Smoothing ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None, smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.smoothing = smoothing

    def forward(self, input, target):
        num_classes = input.size(1)
        one_hot = F.one_hot(target, num_classes=num_classes).float()
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        logp = F.log_softmax(input, dim=1)
        p = torch.exp(logp)
        loss = -(one_hot * ((1 - p) ** self.gamma) * logp)
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)
        return loss.sum(dim=1).mean()

# --- Dataset ---
class TemporalDataset(Dataset):
    def __init__(self, label_file, image_root, target_year):
        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        self.data, self.ids, self.raw_deltas = [], [], []
        self.image_root = image_root
        self.target_year = target_year

        for pid_eye, yearly_list in self.labels.items():
            try:
                yearly = {str(entry['year']): entry for entry in yearly_list}
                input_paths = []
                for y in range(1, target_year):
                    y_str = str(y)
                    if y_str not in yearly or 'images' not in yearly[y_str] or not yearly[y_str]['images']:
                        raise ValueError("Missing images")
                    input_paths += yearly[y_str]['images']

                prev = str(target_year - 1)
                curr = str(target_year)
                if prev not in yearly or curr not in yearly:
                    raise ValueError("Missing BCVA years")

                bcva_prev = yearly[prev]['BCVA']
                bcva_curr = yearly[curr]['BCVA']
                delta = bcva_curr - bcva_prev
                self.raw_deltas.append(delta)
                n_inj = yearly[curr].get('n_inj', 0)
                label = 1 if delta >= 0.1 else 0
                self.data.append((input_paths, label, pid_eye, bcva_prev, bcva_curr, delta, n_inj))
                self.ids.append(pid_eye)
            except Exception:
                continue

        print(f"[Dataset] Using {len(self.data)} samples for year {target_year} prediction")

        self.scaler = MinMaxScaler()
        if self.data:
            meta_features = [[d[3], d[4], d[5], d[6]] for d in self.data]
            self.scaler.fit(meta_features)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paths, label, pid_eye, bcva_prev, bcva_curr, delta, n_inj = self.data[idx]
        imgs = []
        for p in paths:
            img = plt.imread(os.path.join(self.image_root, p))
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            img = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img
            img = Image.fromarray(img)
            img = self.transform(img)
            imgs.append(img)
        imgs = pad_sequence(imgs, batch_first=True)
        meta = torch.tensor(self.scaler.transform([[bcva_prev, bcva_curr, delta, n_inj]])[0], dtype=torch.float32)
        return imgs, label, pid_eye, meta

# --- Model ---
# class TemporalModel(nn.Module):
#     def __init__(self, hidden_dim=256, num_classes=2):
#         super().__init__()
#         base = models.densenet121(pretrained=True)
#         self.encoder = nn.Sequential(base.features, nn.AdaptiveAvgPool2d((1, 1)))
#         self.flatten = nn.Flatten()
#         self.dropout = nn.Dropout(0.5)
#         self.lstm = nn.LSTM(input_size=1024, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
#         self.attn = nn.Linear(2 * hidden_dim, 1)
#         self.fc = nn.Linear(2 * hidden_dim + 4, num_classes)

#     def forward(self, x, meta):
#         B, T, C, H, W = x.shape
#         x = x.view(B*T, C, H, W)
#         feats = self.encoder(x)
#         feats = self.flatten(feats)
#         feats = feats.view(B, T, 1024)
#         out, _ = self.lstm(feats)
#         weights = F.softmax(self.attn(out), dim=1)
#         context = torch.sum(out * weights, dim=1)
#         out = self.dropout(context)
#         out = torch.cat([out, meta], dim=1)
#         return self.fc(out)
class TemporalModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=2):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden_dim, 1)
        self.fc = nn.Linear(2 * hidden_dim + 4, num_classes)

    def forward(self, x, meta):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.encoder(x)
        feats = self.flatten(feats)
        feats = feats.view(B, T, 512)
        out, _ = self.lstm(feats)
        weights = F.softmax(self.attn(out), dim=1)
        context = torch.sum(out * weights, dim=1)
        out = self.dropout(context)
        out = torch.cat([out, meta], dim=1)
        return self.fc(out)

# --- Collate Function ---
def custom_collate(batch):
    xs, ys, ids, metas = zip(*batch)
    max_len = max([x.size(0) for x in xs])
    padded = []
    for x in xs:
        pad = torch.zeros(max_len - x.size(0), *x.shape[1:])
        padded.append(torch.cat([x, pad], dim=0))
    x_tensor = torch.stack(padded)
    y_tensor = torch.tensor(ys)
    meta_tensor = torch.stack(metas)
    return x_tensor, y_tensor, ids, meta_tensor

# --- Save Confusion Matrix PNG ---
def save_confusion_matrix_png(matrix, save_path_prefix):
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred: Not Improved', 'Pred: Improved'],
                yticklabels=['Actual: Not Improved', 'Actual: Improved'])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    plt.savefig(f"{save_path_prefix}_confusion_matrix.png")
    plt.close()

# --- Save All Metrics to Single CSV ---
def save_metrics(report, matrix, auc, save_path_prefix):
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    flat_metrics = {}
    for k, v in report.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_metrics[f"{k}_{sub_k}"] = sub_v
        else:
            flat_metrics[k] = v
    flat_metrics["AUC"] = auc
    flat_metrics["Confusion_TN"] = matrix[0][0]
    flat_metrics["Confusion_FP"] = matrix[0][1]
    flat_metrics["Confusion_FN"] = matrix[1][0]
    flat_metrics["Confusion_TP"] = matrix[1][1]
    pd.DataFrame([flat_metrics]).to_csv(f"{save_path_prefix}_all_metrics.csv", index=False)
    save_confusion_matrix_png(matrix, save_path_prefix)

# --- Compute Metrics ---
def compute_metrics(y_true, y_pred, y_prob, label_names):
    report = classification_report(y_true, y_pred, target_names=label_names, digits=4, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob[:,1]) if y_prob is not None else 0
    print("\nPer-class Metrics:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print(f"Confusion Matrix:\n{matrix}")
    print(f"AUC: {auc:.4f}")
    return report, matrix, auc

# --- Training Loop ---
def run_kfold_training(label_file, image_root, target_year, save_dir, batch_size=1, epochs=10):
    dataset = TemporalDataset(label_file, image_root, target_year)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=custom_collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=custom_collate)

    model = TemporalModel().to(device)
    label_counts = Counter([dataset[i][1] for i in train_set.indices])
    total = sum(label_counts.values())
    weights = torch.tensor([total / label_counts[i] for i in range(2)], dtype=torch.float32).to(device)
    criterion = FocalLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_f1 = 0
    patience = 0
    best_model_path = os.path.join(save_dir, f"best_model_year{target_year}.pt")

    for epoch in range(epochs):
        model.train()
        all_train_preds, all_train_targets, all_train_probs = [], [], []
        running_loss = 0.0
        for xb, yb, _, meta in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            xb, yb, meta = xb.to(device), yb.to(device), meta.to(device)
            optimizer.zero_grad()
            out = model(xb, meta)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            probs = F.softmax(out, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_train_preds.extend(preds)
            all_train_targets.extend(yb.cpu().numpy())
            all_train_probs.extend(probs)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
        compute_metrics(all_train_targets, all_train_preds, np.array(all_train_probs), ['Not Improved', 'Improved'])

        model.eval()
        torch.cuda.empty_cache()
        all_val_preds, all_val_targets, all_val_probs = [], [], []
        with torch.no_grad():
            for xb, yb, _, meta in val_loader:
                xb, yb, meta = xb.to(device), yb.to(device), meta.to(device)
                out = model(xb, meta)
                probs = F.softmax(out, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_val_preds.extend(preds)
                all_val_targets.extend(yb.cpu().numpy())
                all_val_probs.extend(probs)
        print(f"Epoch {epoch+1}: Validation Metrics")
        val_report, _, _ = compute_metrics(all_val_targets, all_val_preds, np.array(all_val_probs), ['Not Improved', 'Improved'])
        val_f1 = val_report['macro avg']['f1-score']

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            patience = 0
        else:
            patience += 1
            print(f"No F1 improvement. Patience: {patience}/3")
            if patience >= 3:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb, _, meta in test_loader:
            xb, yb, meta = xb.to(device), yb.to(device), meta.to(device)
            out = model(xb, meta)
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_targets.extend(yb.cpu().numpy())
            all_probs.extend(probs)

    save_path = os.path.join(save_dir, f"final_test")
    report, matrix, auc = compute_metrics(all_targets, all_preds, np.array(all_probs), ['Not Improved', 'Improved'])
    save_metrics(report, matrix, auc, save_path)

# --- Entry ---
if __name__ == '__main__':
    label_file = "AMD_Label_Delta.json"
    image_root = "E:/Labeled_PNGs"
    target_year = 12
    save_dir = r"D:\MS Computer Engineering\Thesis\Code\AMD\Temporal Final\Results_Y12"
    run_kfold_training(label_file, image_root, target_year, save_dir)
