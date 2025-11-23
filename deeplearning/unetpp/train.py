import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from unetpp.Unetpp import UNetPP6

# ===== DiceLoss 정의 =====
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (probs_flat * target_flat).sum(1)
        dice = (2. * intersection + self.smooth) / (probs_flat.sum(1) + target_flat.sum(1) + self.smooth)
        return 1 - dice.mean()

def combined_loss(logits, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = DiceLoss()(logits, target)
    return bce_weight * bce + (1 - bce_weight) * dice

# ===== 평가지표 =====
def dice_score(logits, target, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * target).sum()
    union = preds.sum() + target.sum()
    return (2 * intersection) / (union + 1e-6)

def accuracy(logits, target, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == target).float().sum()
    return correct / target.numel()

# ===== EarlyStopping 클래스 =====
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoints/best_model.pth'):
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
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)

# ===== 데이터 준비 예시 =====
x = torch.rand(100,1,6,6)
y = torch.randint(0,2,(100,1,6,6)).float()
dataset = TensorDataset(x, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ===== 학습 설정 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetPP6().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
early_stopping = EarlyStopping(patience=10, path="checkpoints/best_unet6.pth")

num_epochs = 50
train_loss_list, val_loss_list = [], []
train_dice_list, val_dice_list = [], []
train_acc_list, val_acc_list = [], []

# ===== 학습 loop =====
for epoch in range(num_epochs):
    # ---- Train ----
    model.train()
    epoch_loss, epoch_dice, epoch_acc = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = combined_loss(logits, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)
        epoch_dice += dice_score(logits, yb).item() * xb.size(0)
        epoch_acc += accuracy(logits, yb).item() * xb.size(0)

    train_loss = epoch_loss / len(train_loader.dataset)
    train_dice = epoch_dice / len(train_loader.dataset)
    train_acc = epoch_acc / len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_dice_list.append(train_dice)
    train_acc_list.append(train_acc)

    # ---- Validation ----
    model.eval()
    val_loss, val_dice, val_acc = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = combined_loss(logits, yb)
            val_loss += loss.item() * xb.size(0)
            val_dice += dice_score(logits, yb).item() * xb.size(0)
            val_acc += accuracy(logits, yb).item() * xb.size(0)

    val_loss /= len(val_loader.dataset)
    val_dice /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)
    val_loss_list.append(val_loss)
    val_dice_list.append(val_dice)
    val_acc_list.append(val_acc)

    print(f"[Epoch {epoch+1:02d}] "
          f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
          f"Dice={val_dice:.4f}, Acc={val_acc:.4f}")

    # Early stopping & Checkpoint
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("⏹ Early stopping triggered.")
        break

# ===== 최종 모델 저장 =====
torch.save(model.state_dict(), "checkpoints/final_unet6.pth")
print("✅ Training complete. Best model saved at 'checkpoints/best_unet6.pth'")

# ===== 학습 그래프 저장 =====
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.subplot(1,3,2)
plt.plot(train_dice_list, label="Train Dice")
plt.plot(val_dice_list, label="Val Dice")
plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.legend()

plt.subplot(1,3,3)
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

plt.tight_layout()
plt.savefig("training_metrics_unet6.png")
plt.show()
