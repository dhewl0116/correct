import time
import ast
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import serial  

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
            x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                            diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet6(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=16, dropout=0.0):
        super().__init__()
        ch = base_ch
        self.inc = DoubleConv(in_channels, ch, dropout=dropout)     # (ch,6,6)
        self.down1 = Down(ch, ch*2, dropout=dropout)                # (2ch,3,3)
        self.bottleneck = DoubleConv(ch*2, ch*4, dropout=dropout)   # (4ch,3,3)
        self.up1 = Up(ch*4, ch*2, skip_ch=ch, dropout=dropout)      # (2ch,6,6)
        self.final_conv = nn.Conv2d(ch*2, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.bottleneck(x2)
        x  = self.up1(x3, x1)
        out = self.final_conv(x)  # logits [B,1,6,6]
        return out


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 모델 로드
MODEL_PATH = "/Users/dhewl/Desktop/Correct_연두/source/checkpoints/best_unet6.pth"
model = UNet6(in_channels=1, out_channels=1)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

THRESHOLD = 0.5 

SERIAL_PORT = '/dev/cu.usbserial-10'  
BAUD = 9600
TIMEOUT = 1.0

plt.ion()
try:
    plt.rcParams['font.family'] = 'AppleGothic'
except Exception:
    pass


plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 12})

fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(12, 6))  # 🔥 창 크게
im_in  = ax_in.imshow(np.zeros((6,6), dtype=np.float32), vmin=0, vmax=1, cmap="viridis")
im_out = ax_out.imshow(np.zeros((6,6), dtype=np.float32), vmin=0, vmax=1, cmap="magma")

ax_in.set_title("Pressure Input (RAW)", fontsize=14)
ax_out.set_title(f"Model Output (0/1)  Thr={THRESHOLD}", fontsize=14)

ax_in.set_xlabel("← 앞(다리)          뒤(허리) →", fontsize=12)
ax_out.set_xlabel("← 앞(다리)          뒤(허리) →", fontsize=12)

ax_in.set_ylabel("왼쪽 엉덩이  ↕  오른쪽 엉덩이", fontsize=12)
ax_out.set_ylabel("왼쪽 엉덩이  ↕  오른쪽 엉덩이", fontsize=12)

fig.tight_layout(pad=3.0)
fig.canvas.draw()
fig.canvas.flush_events()

def parse_bracket_list_36(line: str) -> np.ndarray | None:
    s = line.strip()
    if not s:
        return None
    try:
        obj = ast.literal_eval(s)
        if not isinstance(obj, (list, tuple)):
            return None
        if len(obj) != 36:
            return None
        arr = np.array(obj, dtype=np.float32)
        return arr
    except Exception:
        return None

def main():
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=TIMEOUT)
    time.sleep(2.0) 
    print(f"✅ Serial connected: {SERIAL_PORT} @ {BAUD}bps")

    try:
        while True:
            try:
                raw = ser.readline().decode('utf-8', errors='ignore').strip()
                if not raw:
                    continue

                arr = parse_bracket_list_36(raw)
                if arr is None:
                    print(f"⚠ invalid data: {raw}")
                    continue

                x = torch.tensor(arr.reshape(1, 1, 6, 6), dtype=torch.float32, device=device)

           
                with torch.no_grad():
                    logits = model(x)                 
                    probs  = torch.sigmoid(logits) 
                    pred   = (probs > THRESHOLD).float()

  
                pressure = x[0, 0].detach().cpu().numpy() 
                pred_map = pred[0, 0].detach().cpu().numpy() 

        
                pmin, pmax = pressure.min(), pressure.max()
                pressure_vis = (pressure - pmin) / (pmax - pmin) if pmax > pmin else pressure

                pressure_vis_disp = np.flipud(pressure_vis)
                pred_map_disp     = np.flipud(pred_map)

                im_in.set_data(pressure_vis_disp)
                im_out.set_data(pred_map_disp)

                im_in.set_clim(vmin=0.0, vmax=1.0)
                im_out.set_clim(vmin=0.0, vmax=1.0)

                ax_in.set_title(f"Pressure Input (RAW)\nmin={pmin:.2f}, max={pmax:.2f}", fontsize=13)
                ax_out.set_title(f"Model Output (0/1)  Thr={THRESHOLD}", fontsize=13)

                fig.canvas.draw()
                fig.canvas.flush_events()

                on_cnt = int(pred_map.sum())
                peak_idx = int(np.argmax(pressure))
                print(f"[serial] active_cells={on_cnt}/36 | peak={np.unravel_index(peak_idx,(6,6))}  "
                      f"(col=0=앞/다리)")

            except Exception as e:
                print(f"Error (loop): {e}")
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n⏹ Stopped by user.")
    finally:
        try:
            ser.close()
        except Exception:
            pass
        plt.ioff()
        plt.close(fig)

if __name__ == "__main__":
    main()
