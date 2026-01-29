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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# 🔍 Visualize Biomarker Features via t-SNE or PCA

def visualize_feature_space(dataset, method='tsne', save_path="feature_projection.png"):
    print("Extracting features for visualization...")
    features, labels = [], []
    for i in tqdm(range(len(dataset)), desc="Extracting features"):
        try:
            x, y, *_ = dataset[i]
            if x.shape[0] == 0:
                continue
            features.append(torch.mean(x, dim=0).numpy())  # Mean across time
            labels.append(int(y))
        except Exception as e:
            print(f"[Warning] Skipping index {i} due to error: {e}")
            continue

    if not features:
        print("No valid features to visualize.")
        return

    print(f"Running {method.upper()} projection...")
    features = np.stack(features)
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2)
    reduced = reducer.fit_transform(features)

    df = pd.DataFrame(reduced, columns=["x", "y"])
    df["label"] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="Set2")
    plt.title(f"{method.upper()} of Biomarker + Metadata Features")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

# 🧠 Visualize LSTM Hidden State via t-SNE or PCA

def visualize_lstm_hidden_space(model, dataset, method='tsne', save_path="hidden_state_projection.png", device='cpu'):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Extracting hidden states"):
            try:
                x, y, *_ = dataset[i]
                x = x.to(device).unsqueeze(0)  # (1, T, D)
                logits = model(x)
                if hasattr(model, 'last_hidden'):  # expected to be set inside forward pass
                    hidden = model.last_hidden.squeeze().cpu().numpy()
                    features.append(hidden)
                    labels.append(int(y))
            except Exception as e:
                print(f"[Warning] Skipping index {i} due to error: {e}")
                continue

    if not features:
        print("No valid hidden states to visualize.")
        return

    print(f"Running {method.upper()} projection...")
    features = np.stack(features)
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2)
    reduced = reducer.fit_transform(features)

    df = pd.DataFrame(reduced, columns=["x", "y"])
    df["label"] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="Set2")
    plt.title(f"{method.upper()} of LSTM Hidden States")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved hidden state visualization to {save_path}")

# 🧱 Actual BiLSTM with Attention used in training (for loading trained weights)
class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, lstm_out):
        scores = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(lstm_out * weights, dim=1)
        return context

class CNNBiLSTMAttentionClassifier(nn.Module):
    def __init__(self, feature_dim=1027, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.attn = AttentionModule(hidden_dim * 2)
        self.out = nn.Linear(hidden_dim * 2, 1)
        self.last_hidden = None  # for feature visualization

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attn(lstm_out)
        self.last_hidden = context.detach()  # capture last hidden state
        return self.out(context).squeeze(1)

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
        if idx >= len(self.samples):
            raise IndexError("Sample index out of range")
        item = self.samples[idx]
        return item['sequence'], torch.tensor(item['label']).float(), item['id'], item['year'], item['sequence'].shape[0]


model = CNNBiLSTMAttentionClassifier()
model.load_state_dict(torch.load(r"D:\MS Computer Engineering\Thesis\Code\AMD\Results_Temporal_Biomarker_20250619_202246\Fold1\best_model.pth"))
model.eval()

dataset = BiomarkerTemporalDataset(r"D:\MS Computer Engineering\Thesis\Code\AMD\AMD_Label_New.json", r"E:\Labeled_PNGs", r"D:\MS Computer Engineering\Thesis\Code\Weights\pretrained_amd_encoder.pth")
# visualize_feature_space(dataset, method='tsne')  # or 'pca'
visualize_lstm_hidden_space(model, dataset, method='tsne')  # or method='pca'
