import os
import json
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


class YearPairDeltaDataset(Dataset):
    def __init__(self, json_path, image_root, transform=None):
        self.image_root = image_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.samples = []

        with open(json_path, 'r') as f:
            data = json.load(f)

        for eye_id, records in data.items():
            records = sorted(records, key=lambda x: x['year'])
            bcvas = [r['BCVA'] for r in records if 'BCVA' in r]
            if len(bcvas) != len(records):
                continue

            for i in range(1, len(records)):
                prev, curr = records[i - 1], records[i]
                delta = curr['BCVA'] - prev['BCVA']
                if delta >= 0.05:
                    label = 2
                elif delta <= -0.05:
                    label = 0
                else:
                    label = 1

                imgs_prev = [os.path.join(image_root, p) for p in prev.get("images", []) if os.path.isfile(os.path.join(image_root, p))]
                imgs_curr = [os.path.join(image_root, p) for p in curr.get("images", []) if os.path.isfile(os.path.join(image_root, p))]
                if imgs_prev and imgs_curr:
                    self.samples.append({
                        'id': eye_id,
                        'prev_imgs': imgs_prev,
                        'curr_imgs': imgs_curr,
                        'label': torch.tensor(label)
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        def extract_features(image_paths):
            features = []
            for img_path in image_paths:
                feature_path = img_path.replace(".png", "_feat.pt")
                if os.path.exists(feature_path):
                    feat = torch.load(feature_path)
                else:
                    img = self.transform(Image.open(img_path).convert('L')).unsqueeze(0)
                    with torch.no_grad():
                        feat = dataset_encoder(img).squeeze(0)
                    torch.save(feat, feature_path)
                features.append(feat)
            return torch.stack(features)

        prev_tensor = extract_features(item['prev_imgs'])
        curr_tensor = extract_features(item['curr_imgs'])

        return (prev_tensor, curr_tensor), item['label'], item['id']


def collate_fn(batch):
    pairs, labels, ids = zip(*batch)
    prevs, currs = zip(*pairs)

    def pad_sequence(tensors):
        max_len = max(t.shape[0] for t in tensors)
        padded = torch.zeros(len(tensors), max_len, *tensors[0].shape[1:])
        lengths = torch.tensor([t.shape[0] for t in tensors])
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0]] = t
        return padded, lengths

    prevs_pad, prev_lens = pad_sequence(prevs)
    currs_pad, curr_lens = pad_sequence(currs)

    return (prevs_pad, currs_pad), torch.stack(labels), [0] * len(labels), (prev_lens, curr_lens)


base = models.resnet18(weights="IMAGENET1K_V1")
base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
dataset_encoder = nn.Sequential(*list(base.children())[:-1], nn.Flatten()).eval()


class DeltaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_prev = nn.LSTM(512, 256, batch_first=True)
        self.lstm_curr = nn.LSTM(512, 256, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, x, lengths):
        prev, curr = x
        prev_len, curr_len = lengths

        packed_prev = nn.utils.rnn.pack_padded_sequence(prev, prev_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_curr = nn.utils.rnn.pack_padded_sequence(curr, curr_len.cpu(), batch_first=True, enforce_sorted=False)

        _, (h_prev, _) = self.lstm_prev(packed_prev)
        _, (h_curr, _) = self.lstm_curr(packed_curr)

        feats = torch.cat([h_prev[-1], h_curr[-1]], dim=1)
        return self.classifier(feats)


def compute_metrics(y_true, y_pred):
    print("Per-class Metrics:")
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "rec": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "auc": roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), multi_class='ovo')
    }


def train_pipeline(json_path, image_root, save_dir, max_epochs=30):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = YearPairDeltaDataset(json_path, image_root)
    print("Class distribution:")
    counts = Counter(s['label'].item() for s in dataset.samples)
    for cls, count in sorted(counts.items()):
        print(f"Class {cls} → {count} samples")

    weights = torch.tensor([1.0 / counts[i] for i in range(3)], dtype=torch.float32).to(device)

    all_labels = [s['label'].item() for s in dataset.samples]
    all_groups = [s['id'] for s in dataset.samples]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(splitter.split(all_labels, groups=all_groups))
    val_idx, test_idx = next(GroupShuffleSplit(test_size=0.5, n_splits=1).split(
        [all_labels[i] for i in temp_idx], groups=[all_groups[i] for i in temp_idx]))
    val_idx = [temp_idx[i] for i in val_idx]
    test_idx = [temp_idx[i] for i in test_idx]

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = DeltaClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_f1 = 0
    patience, patience_counter = 5, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_preds, train_targets = [], []
        for (prevs, currs), y, _, (prev_lens, curr_lens) in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            x = (prevs.to(device), currs.to(device))
            lengths = (prev_lens.to(device), curr_lens.to(device))
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x, lengths)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_preds.extend(out.argmax(1).cpu().tolist())
            train_targets.extend(y.cpu().tolist())

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for (prevs, currs), y, _, (prev_lens, curr_lens) in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                x = (prevs.to(device), currs.to(device))
                lengths = (prev_lens.to(device), curr_lens.to(device))
                y = y.to(device)
                out = model(x, lengths)
                val_preds.extend(out.argmax(1).cpu().tolist())
                val_targets.extend(y.cpu().tolist())

        train_metrics = compute_metrics(train_targets, train_preds)
        val_metrics = compute_metrics(val_targets, val_preds)

        print(f"Epoch {epoch} Metrics:")
        print("Train → " + ", ".join([f"{k.upper()}: {v:.4f}" for k, v in train_metrics.items()]))
        print("Val   → " + ", ".join([f"{k.upper()}: {v:.4f}" for k, v in val_metrics.items()]))

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for (prevs, currs), y, _, (prev_lens, curr_lens) in tqdm(test_loader, desc="Testing"):
            x = (prevs.to(device), currs.to(device))
            lengths = (prev_lens.to(device), curr_lens.to(device))
            y = y.to(device)
            out = model(x, lengths)
            test_preds.extend(out.argmax(1).cpu().tolist())
            test_targets.extend(y.cpu().tolist())

    test_metrics = compute_metrics(test_targets, test_preds)
    print("Test Metrics:")
    for k, v in test_metrics.items():
        print(f"{k.upper()}: {v:.4f}")


if __name__ == '__main__':
    save_dir = f"Results_DeltaClassifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_pipeline("AMD_Label_Delta.json", "E:/Labeled_PNGs", save_dir)
