import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path

class PatientEyeDataset(Dataset):
    def __init__(self, json_path, image_root, transform=None):
        self.data = self._load_json(json_path)
        self.image_root = Path(image_root)
        self.transform = transform if transform else self.default_transforms()
        self.label_map = {"improved": 1, "not_improved": 0}

    def _load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def default_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        images = []
        for img_name in item['images']:
            img_path = self.image_root / img_name
            try:
                img = Image.open(img_path).convert('L')  # convert to grayscale
                img = self.transform(img)
                images.append(img)
            except Exception as e:
                continue

        if not images:
            raise ValueError(f"No valid images for item at index {idx}")

        image_tensor = torch.stack(images)  # shape: (N, C, H, W)
        label = torch.tensor(self.label_map[item['label']], dtype=torch.long)

        return image_tensor, label

# dataset = PatientEyeDataset("flattened_all_years.json", image_root="/path/to/images")
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)