import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import models, transforms
from collections import Counter
import seaborn as sns
import pandas as pd
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=3.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=1)
        prob = torch.exp(log_prob)
        ce_loss = F.nll_loss(log_prob, target, weight=self.weight, reduction='none')
        focal_loss = ((1 - prob.gather(1, target.unsqueeze(1)).squeeze()) ** self.gamma) * ce_loss
        return focal_loss.mean()

# --- Dataset ---
class TemporalDataset(Dataset):
    def __init__(self, label_file, image_root, target_year):
        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        self.data, self.ids = [], []
        self.image_root = image_root
        self.target_year = target_year
        self.scaler = MinMaxScaler()

        for pid_eye, yearly_list in self.labels.items():
            try:
                yearly = {str(entry['year']): entry for entry in yearly_list}
                input_paths = []
                for y in range(1, target_year):
                    # y_str = str(y)
                    y_str = str(target_year - 1)
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
                n_inj = yearly[curr].get('n_inj', 0)
                label = 1 if delta >= 0.1 else 0
                self.data.append((input_paths, label, pid_eye, bcva_prev, bcva_curr, delta, n_inj))
                self.ids.append(pid_eye)
            except Exception:
                continue

        print(f"[Dataset] Using {len(self.data)} samples for year {target_year} prediction")

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
class TemporalModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=2):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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

# --- Compute and Print Metrics ---
def compute_and_save_metrics(y_true, y_pred, y_prob, ids, output_dir, fold_name):
    os.makedirs(output_dir, exist_ok=True)
    precision = precision_score(y_true, y_pred, pos_label=0)
    recall = recall_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred, pos_label=0)
    auc = roc_auc_score(1 - np.array(y_true), y_prob[:, 0])
    acc = (np.array(y_true) == np.array(y_pred)).mean()
    cm = confusion_matrix(y_true, y_pred)

    print(f"Fold {fold_name} Metrics:")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f} | Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'precision': precision, 'recall': recall, 'f1_score': f1, 'accuracy': acc, 'auc': auc,
        'tn': cm[0][0], 'fp': cm[0][1], 'fn': cm[1][0], 'tp': cm[1][1]
    }])
    metrics_df.to_csv(os.path.join(output_dir, f"{fold_name}_metrics.csv"), index=False)

    # Save confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Improved', 'Improved'],
                yticklabels=['Not Improved', 'Improved'])
    plt.title(f"Confusion Matrix - {fold_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{fold_name}_confusion_matrix.png"))
    plt.close()

    # Save ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])  # use prob of class 1 (Improved)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {fold_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{fold_name}_roc_curve.png"))
    plt.close()

    # Save predictions
    pd.DataFrame({
        'PatientID': ids,
        'True': y_true,
        'Pred': y_pred,
        'Prob_NotImproved': y_prob[:, 0],
        'Prob_Improved': y_prob[:, 1]
    }).to_csv(os.path.join(output_dir, f"{fold_name}_predictions.csv"), index=False)


# --- Aggregate Fold Results ---
def summarize_kfold_results(output_dir, k):
    all_metrics = []
    for i in range(1, k+1):
        path = os.path.join(output_dir, f"fold_{i}", "test_metrics.csv")
        df = pd.read_csv(path)
        all_metrics.append(df)
    df_all = pd.concat(all_metrics, ignore_index=True)
    summary = df_all.agg(['mean', 'std'])
    summary.to_csv(os.path.join(output_dir, "kfold_summary.csv"))
    print("\nCross-Validation Summary:")
    print(summary)

# --- Run K-Fold Training ---
def run_kfold_training(model_class, dataset, output_dir, k=3, batch_size=1, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    indices = np.arange(len(dataset))

    for fold, (train_idx, test_idx) in enumerate(kf.split(indices), 1):
        print(f"\n===== Fold {fold} =====")
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

        model = model_class().to(device)
        alpha = 2
        label_counts = Counter([dataset[i][1] for i in train_idx])
        total = sum(label_counts.values())
        weights = torch.tensor([(total / label_counts[i])*alpha for i in range(2)], dtype=torch.float32).to(device)
        criterion = FocalLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_combined_score = -1.0
        F1_WEIGHT = 0.4
        AUC_WEIGHT = 0.6
        patience = 0
        best_model_path = os.path.join(fold_dir, "best_model.pt")

        for epoch in range(epochs):
            model.train()
            train_preds, train_targets, train_probs, train_ids = [], [], [], []
            total_loss = 0
            for xb, yb, ids, meta in tqdm(train_loader, desc=f"Fold {fold} - Epoch {epoch+1}"):
                xb, yb, meta = xb.to(device), yb.to(device), meta.to(device)
                optimizer.zero_grad()
                out = model(xb, meta)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                probs = F.softmax(out, dim=1).detach().cpu().numpy()
                preds = np.argmax(probs, axis=1)
                train_preds.extend(preds)
                train_targets.extend(yb.cpu().numpy())
                train_probs.extend(probs)
                train_ids.extend(ids)

            print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f}")
            compute_and_save_metrics(train_targets, train_preds, np.array(train_probs), train_ids, fold_dir, f"train_epoch{epoch+1}")

            # Validation on test set (early stopping)
            model.eval()
            val_preds, val_targets, val_probs, val_ids = [], [], [], []
            with torch.no_grad():
                for xb, yb, ids, meta in test_loader:
                    xb, yb, meta = xb.to(device), yb.to(device), meta.to(device)
                    out = model(xb, meta)
                    probs = F.softmax(out, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                    val_preds.extend(preds)
                    val_targets.extend(yb.cpu().numpy())
                    val_probs.extend(probs)
                    val_ids.extend(ids)

            val_f1 = f1_score(val_targets, val_preds, pos_label=0)
            val_auc = roc_auc_score(1 - np.array(val_targets), np.array(val_probs)[:, 0])
            # print(f"Epoch {epoch+1}: Test F1 = {val_f1:.4f}")
            # print(f"Epoch {epoch+1}: Test AUC = {val_auc:.4f}")
            compute_and_save_metrics(val_targets, val_preds, np.array(val_probs), val_ids, fold_dir, f"test_epoch{epoch+1}")

            current_combined_score = (F1_WEIGHT * val_f1) + (AUC_WEIGHT * val_auc)
            print(f"Epoch {epoch+1}: Test Combined Score = {current_combined_score:.4f}")

            if current_combined_score > best_combined_score:
                best_combined_score = current_combined_score
                torch.save(model.state_dict(), best_model_path)
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    print("Early stopping.")
                    break

        # Load best model and evaluate on test set again for final metric saving
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_preds, test_targets, test_probs, test_ids = [], [], [], []
        with torch.no_grad():
            for xb, yb, ids, meta in test_loader:
                xb, yb, meta = xb.to(device), yb.to(device), meta.to(device)
                out = model(xb, meta)
                probs = F.softmax(out, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                test_preds.extend(preds)
                test_targets.extend(yb.cpu().numpy())
                test_probs.extend(probs)
                test_ids.extend(ids)

        compute_and_save_metrics(test_targets, test_preds, np.array(test_probs), test_ids, fold_dir, "test")

    summarize_kfold_results(output_dir, k)

if __name__ == "__main__":
    label_file = "AMD_Label_Delta.json"  # path to your JSON label file
    image_root = "E:/Labeled_PNGs"       # folder where your OCT scan images are
    target_year = 10                   # prediction year

    dataset = TemporalDataset(label_file, image_root, target_year)
    output_dir = r"D:\MS Computer Engineering\Thesis\Code\AMD\Final_KFold_Lastyear_Y10"

    run_kfold_training(
        model_class=TemporalModel,
        dataset=dataset,
        output_dir=output_dir,
        k=3,
        batch_size=1,
        epochs=20
    )