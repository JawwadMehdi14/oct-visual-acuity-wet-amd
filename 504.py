# 🚧 DenseNet-based U-Net for AMD-SD Biomarker Segmentation
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm
import numpy as np

class DenseUNet(nn.Module):
    def __init__(self, n_classes=5):
        super(DenseUNet, self).__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.encoder = base.features

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512 + 1024, 512, kernel_size=3, padding=1),  # x8 corrected to 1024
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1),  # x6 corrected to 512
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),  # x4 is 256
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),  # x2 is 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.encoder[0](x)
        x1 = self.encoder[1](x0)
        x2 = self.encoder[2](x1)
        x3 = self.encoder[3](x2)
        x4 = self.encoder[4](x3)
        x5 = self.encoder[5](x4)
        x6 = self.encoder[6](x5)
        x7 = self.encoder[7](x6)
        x8 = self.encoder[8](x7)
        x9 = self.encoder[9](x8)
        x10 = self.encoder[10](x9)
        x11 = self.encoder[11](x10)

        u1 = self.up1(x11)
        u1 = torch.cat([u1, x8], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x6], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x4], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, x2], dim=1)
        u4 = self.conv4(u4)

        return nn.functional.interpolate(self.final(u4), size=(224, 224), mode='bilinear', align_corners=False)

class AMDSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        img_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        image = self.transform(image)
        mask = self.transform(mask).squeeze(0).long()

        return image, mask

def dice_score(pred, target, num_classes):
    pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    dice_scores = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item()
        dice = (2. * intersection) / (union + 1e-5) if union != 0 else 1.0
        dice_scores.append(dice)
    return np.mean(dice_scores), dice_scores

def dice_loss(pred, target, smooth=1e-5):
    pred = torch.softmax(pred, dim=1)
    target_1h = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * target_1h).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_1h.sum(dim=(2, 3))
    return 1 - ((2. * intersection + smooth) / (union + smooth)).mean()

def train_segmentation():
    image_dir = r"D:\MS Computer Engineering\Thesis\Code\images"
    mask_dir = r"D:\MS Computer Engineering\Thesis\Code\masks"
    save_path = r"D:\MS Computer Engineering\Thesis\Code\Weights\pretrained_amd_encoder.pth"

    dataset = AMDSegmentationDataset(image_dir, mask_dir)
    val_size = int(0.15 * len(dataset))
    test_size = int(0.10 * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

    model = DenseUNet(n_classes=5).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ce_loss = nn.CrossEntropyLoss()
    best_dice = 0

    for epoch in range(10):
        model.train()
        running_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            imgs, masks = imgs.cuda(), masks.cuda()
            logits = model(imgs)
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1} Train Loss: {running_loss / len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            dice_total = 0
            class_dices = np.zeros(5)
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                imgs, masks = imgs.cuda(), masks.cuda()
                logits = model(imgs)
                mean_dice, per_class_dice = dice_score(logits, masks, num_classes=5)
                dice_total += mean_dice
                class_dices += per_class_dice

            mean_dice_epoch = dice_total / len(val_loader)
            per_class_dice_epoch = class_dices / len(val_loader)
            print(f"Epoch {epoch+1} Val Dice: {mean_dice_epoch:.4f}, Per-class: {np.round(per_class_dice_epoch, 4)}")

            if mean_dice_epoch > best_dice:
                best_dice = mean_dice_epoch
                torch.save(model.encoder.state_dict(), save_path)
                print(f"✅ Saved new best encoder with Val Dice: {best_dice:.4f}")

    # Final Evaluation on Test Set
    print("\n=== Final Evaluation on Test Set ===")
    model.eval()
    with torch.no_grad():
        dice_total = 0
        class_dices = np.zeros(5)
        for imgs, masks in tqdm(test_loader, desc="Testing"):
            imgs, masks = imgs.cuda(), masks.cuda()
            logits = model(imgs)
            mean_dice, per_class_dice = dice_score(logits, masks, num_classes=5)
            dice_total += mean_dice
            class_dices += per_class_dice

        mean_test_dice = dice_total / len(test_loader)
        per_class_test_dice = class_dices / len(test_loader)
        print(f"Test Mean Dice: {mean_test_dice:.4f}")
        for i, d in enumerate(per_class_test_dice):
            print(f"Class {i} Dice: {d:.4f}")

# Run training if needed
if __name__ == '__main__':
    train_segmentation()
