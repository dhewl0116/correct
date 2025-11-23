import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.optim as optim


class CushionSegDataset(Dataset):
    """
    posture_seg_dataset_augmented.csv (또는 중복제거+최종본) 읽어서
    x: [1,6,6] float
    y: [1,6,6] float (0/1 mask)
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        px_cols   = [f"px_{i}"   for i in range(36)]
        mask_cols = [f"mask_{i}" for i in range(36)]

        self.X = df[px_cols].to_numpy(dtype=np.float32)    # shape: [N,36]
        self.Y = df[mask_cols].to_numpy(dtype=np.float32)  # shape: [N,36]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_flat = self.X[idx]  # (36,)
        y_flat = self.Y[idx]  # (36,)

        x_img = torch.tensor(x_flat, dtype=torch.float32).reshape(1,6,6)
        y_img = torch.tensor(y_flat, dtype=torch.float32).reshape(1,6,6)

        return x_img, y_img


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(
                x1,
                [diffX//2, diffX - diffX//2,
                 diffY//2, diffY - diffY//2]
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet6(nn.Module):
    """
    작은 입력(6x6)에 맞춘 경량 U-Net 변형.
    encoder: 6x6 -> 3x3
    bottleneck: channels up
    decoder: 3x3 -> 6x6
    최종 출력: 1채널 (binary mask logits)
    """
    def __init__(self, in_channels=1, out_channels=1, base_ch=16, dropout=0.0):
        super().__init__()
        ch = base_ch
        self.inc = DoubleConv(in_channels, ch, dropout=dropout)     # -> (ch,6,6)
        self.down1 = Down(ch, ch*2, dropout=dropout)                # -> (2ch,3,3)
        self.bottleneck = DoubleConv(ch*2, ch*4, dropout=dropout)   # -> (4ch,3,3)
        self.up1 = Up(ch*4, ch*2, skip_ch=ch, dropout=dropout)      # -> (2ch,6,6)
        self.final_conv = nn.Conv2d(ch*2, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)        # (B,ch,6,6)
        x2 = self.down1(x1)     # (B,2ch,3,3)
        x3 = self.bottleneck(x2)# (B,4ch,3,3)
        x  = self.up1(x3, x1)   # (B,2ch,6,6)
        out = self.final_conv(x)# (B,1,6,6)
        return out


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        # logits: [B,1,6,6], target: [B,1,6,6]
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (probs_flat * target_flat).sum(1)
        dice = (2. * intersection + self.smooth) / (
            probs_flat.sum(1) + target_flat.sum(1) + self.smooth
        )
        return 1 - dice.mean()

def combined_loss(logits, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = DiceLoss()(logits, target)
    return bce_weight * bce + (1 - bce_weight) * dice

@torch.no_grad()
def dice_score(logits, target, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * target).sum()
    union = preds.sum() + target.sum()
    return (2 * intersection) / (union + 1e-6)

@torch.no_grad()
def accuracy(logits, target, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == target).float().sum()
    return correct / target.numel()


class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoints/best_unet6.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.delta:
            # 성능 개선 안 됨
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 개선됨 -> 저장하고 카운터 리셋
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)



device = torch.device("cuda" if torch.cuda.is_available() else "mps")

csv_path = "posture_seg_dataset_augmented.csv"

full_dataset = CushionSegDataset(csv_path)

num_samples = len(full_dataset)
val_ratio = 0.2
num_val = int(num_samples * val_ratio)
num_train = num_samples - num_val

train_dataset, val_dataset = random_split(
    full_dataset,
    [num_train, num_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

model = UNet6(in_channels=1, out_channels=1, base_ch=16, dropout=0.2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
early_stopping = EarlyStopping(patience=10, path="checkpoints/best_unet6.pth")

num_epochs = 50

train_loss_list, val_loss_list = [], []
train_dice_list, val_dice_list = [], []
train_acc_list,  val_acc_list  = [], []


for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_acc  = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)  
        loss = combined_loss(logits, yb)

        loss.backward()
        optimizer.step()

        batch_size = xb.size(0)
        epoch_loss += loss.item() * batch_size
        epoch_dice += dice_score(logits, yb).item() * batch_size
        epoch_acc  += accuracy(logits, yb).item() * batch_size

    train_loss = epoch_loss / len(train_loader.dataset)
    train_dice = epoch_dice / len(train_loader.dataset)
    train_acc  = epoch_acc  / len(train_loader.dataset)

    train_loss_list.append(train_loss)
    train_dice_list.append(train_dice)
    train_acc_list.append(train_acc)

    model.eval()
    val_loss_sum = 0.0
    val_dice_sum = 0.0
    val_acc_sum  = 0.0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = combined_loss(logits, yb)

            batch_size = xb.size(0)
            val_loss_sum += loss.item() * batch_size
            val_dice_sum += dice_score(logits, yb).item() * batch_size
            val_acc_sum  += accuracy(logits, yb).item() * batch_size

    val_loss = val_loss_sum / len(val_loader.dataset)
    val_dice = val_dice_sum / len(val_loader.dataset)
    val_acc  = val_acc_sum  / len(val_loader.dataset)

    val_loss_list.append(val_loss)
    val_dice_list.append(val_dice)
    val_acc_list.append(val_acc)

    print(f"[Epoch {epoch+1:02d}] "
          f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
          f"Dice={val_dice:.4f}, Acc={val_acc:.4f}")

    # EarlyStopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("⏹ Early stopping triggered.")
        break


os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/final_unet6.pth")
print("✅ Training complete. Best model saved at 'checkpoints/best_unet6.pth'")

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list,   label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.subplot(1,3,2)
plt.plot(train_dice_list, label="Train Dice")
plt.plot(val_dice_list,   label="Val Dice")
plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.legend()

plt.subplot(1,3,3)
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list,   label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

plt.tight_layout()
plt.savefig("training_metrics_unet6.png")
plt.show()
