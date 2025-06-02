# === TEST SCRIPT: Reload best model and evaluate on test set only ===

import torch
import pandas as pd
import json, os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

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

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = torch.nn.Flatten()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        return self.classifier(x)

# === SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Load dataset
full_dataset = YearwiseImageDataset("AMD_Label_New.json")
labels = [s[1] for s in full_dataset.samples]
from sklearn.model_selection import train_test_split
_, temp_idx = train_test_split(range(len(full_dataset)), test_size=0.3, stratify=labels, random_state=42)
_, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=42)
test_set = torch.utils.data.Subset(full_dataset, test_idx)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# === TEST ===
test_results = []
with torch.no_grad():
    for image, label, pid in tqdm(test_loader, desc="Testing"):
        image = image.to(device)
        logit = model(image)
        prob = torch.sigmoid(logit).item()
        test_results.append((pid, int(label), int(prob >= 0.5), prob))

# === METRICS & OUTPUT ===
y_true = [row[1] for row in test_results]
y_pred = [row[2] for row in test_results]
y_prob = [row[3] for row in test_results]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

print(f"\n✅ Test Results:")
print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.savefig("test_confusion_matrix.png")
plt.close()

# ROC Curve
RocCurveDisplay.from_predictions(y_true, y_prob)
plt.title("ROC Curve (Test Set)")
plt.savefig("test_roc_curve.png")
plt.close()

# Save results
df = pd.DataFrame(test_results, columns=["PatientID", "TrueLabel", "PredLabel", "Probability"])
df.to_csv("test_only_predictions.csv", index=False)
print("📁 Saved test_only_predictions.csv + PNG plots.")
