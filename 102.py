# === IMPORTS ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
import json, os
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             ConfusionMatrixDisplay, RocCurveDisplay)
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def main():
    # === CONFIG ===
    json_file = "flattened_all_years.json"
    image_root = "E:/Labeled_PNGs"
    batch_size = 16
    num_epochs = 30
    lr = 3e-5
    patience = 5
    weight_decay = 1e-4

    # === DATA TRANSFORMS ===
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # === DATASET CLASS ===
    class ImageLevelDataset(Dataset):
        def __init__(self, json_path, image_root, transform=None):
            self.samples = []
            self.transform = transform
            with open(json_path, 'r') as f:
                data = json.load(f)
            for entry in data:
                patient_eye_id = entry['patient_eye']
                label = 1 if entry['label'] == 'improved' else 0
                for img_path in entry['images']:
                    full_path = os.path.join(image_root, img_path)
                    if os.path.isfile(full_path):
                        self.samples.append((full_path, label, patient_eye_id))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label, patient_eye_id = self.samples[idx]
            image = Image.open(img_path).convert('L')
            image = self.transform(image)
            return image, label, patient_eye_id

    # === MODEL ARCHITECTURE ===
    class EnhancedResNet(nn.Module):
        def __init__(self):
            super().__init__()
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.features = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                nn.Dropout2d(0.2)
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 1)
            )
            self.skip = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.SiLU()
            )

        def forward(self, x):
            x = self.features(x)
            identity = self.skip(x)
            x = x + identity
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x).squeeze(-1)

    # === EVALUATION FUNCTION ===
    def evaluate_model(loader, model, device):
        model.eval()
        preds_per_eye = defaultdict(list)
        labels_per_eye = {}
        with torch.no_grad():
            for images, labels, pids in loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                labels = labels.cpu().numpy()
                pids = np.array(pids)
                for i in range(len(pids)):
                    pid = pids[i]
                    prob = probs[i] if probs.ndim > 0 else probs.item()
                    label = labels[i] if labels.ndim > 0 else labels.item()
                    preds_per_eye[pid].append(float(prob))
                    labels_per_eye[pid] = int(label)
        y_true = [labels_per_eye[pid] for pid in preds_per_eye]
        y_prob = [np.mean(preds_per_eye[pid]) for pid in preds_per_eye]
        y_pred = [int(prob >= 0.5) for prob in y_prob]
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        return acc, prec, rec, f1, auc, y_true, y_pred, y_prob, preds_per_eye, labels_per_eye

    # === DATA SPLITTING AND LOADING ===
    full_dataset = ImageLevelDataset(json_file, image_root, transform=val_transform)
    all_patient_eyes = list({sample[2] for sample in full_dataset.samples})
    labels_dict = {pid: next(sample[1] for sample in full_dataset.samples if sample[2] == pid) for pid in all_patient_eyes}
    all_labels = [labels_dict[pid] for pid in all_patient_eyes]

    train_ids, temp_ids = train_test_split(all_patient_eyes, test_size=0.3, stratify=all_labels, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, stratify=[labels_dict[pid] for pid in temp_ids], random_state=42)

    def filter_samples_by_ids(dataset, sample_ids):
        return [i for i, sample in enumerate(dataset.samples) if sample[2] in sample_ids]

    train_idx = filter_samples_by_ids(full_dataset, train_ids)
    val_idx = filter_samples_by_ids(full_dataset, val_ids)
    test_idx = filter_samples_by_ids(full_dataset, test_ids)

    train_set = torch.utils.data.Subset(ImageLevelDataset(json_file, image_root, transform=train_transform), train_idx)
    val_set = torch.utils.data.Subset(ImageLevelDataset(json_file, image_root, transform=val_transform), val_idx)
    test_set = torch.utils.data.Subset(ImageLevelDataset(json_file, image_root, transform=val_transform), test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    print(f"\n📊 Dataset Splits:")
    print(f"Training images: {len(train_set)}")
    print(f"Validation images: {len(val_set)}")
    print(f"Testing images: {len(test_set)}")
    print(f"Training patient-eyes: {len(train_ids)}")
    print(f"Validation patient-eyes: {len(val_ids)}")
    print(f"Testing patient-eyes: {len(test_ids)}")

    # === CONTINUE WITH TRAINING LOOP, EARLY STOPPING, EVALUATION AND SAVING AS IN PREVIOUS BLOCK ===

if __name__ == '__main__':
    main()
