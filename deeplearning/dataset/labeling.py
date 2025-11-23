import pandas as pd
import numpy as np
import random
import cv2

def generate_segmentation_map(label):
    mask = np.zeros((6,6), dtype=np.uint8)
    
    if label == 'front':
        mask[0,5] = 1
        mask[1,4:6] = 1
        mask[2,4:6] = 1
        mask[3,4:6] = 1
        mask[4,4:6] = 1
        mask[5,5] = 1

    elif label == 'sleep':
        mask[0,0] = 1
        mask[1,0:2] = 1
        mask[2,0:2] = 1
        mask[3,0:2] = 1
        mask[4,0:2] = 1
        mask[5,0] = 1

    elif label == 'twist_left':
        mask[0:2,1:5] = 1

    elif label == 'twist_right':
        mask[4:6,1:5] = 1

    return mask

def augment_data(data, mask):
    aug_data, aug_mask = data.copy(), mask.copy()
    
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.1, data.shape)
        aug_data = np.clip(aug_data + noise, 0, None)
    
    # 2) Brightness scale
    if random.random() < 0.5:
        scale = random.uniform(0.9, 1.1)
        aug_data = np.clip(aug_data * scale, 0, None)
    
    # 3) Pixel shift
    if random.random() < 0.5:
        shift_x, shift_y = random.choice([-1,0,1]), random.choice([-1,0,1])
        M = np.float32([[1,0,shift_x],[0,1,shift_y]])
        aug_data = cv2.warpAffine(aug_data, M, (6,6), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        aug_mask = cv2.warpAffine(aug_mask, M, (6,6), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return aug_data, aug_mask

# =========================
# 3. CSV 처리 + 증강 + 중복 제거
# =========================
def process_csv(path, n_augment=1):
    df = pd.read_csv(path)
    all_data, all_masks, all_labels = [], [], []
    
    for _, row in df.iterrows():
        vals = row[:-1].values.astype(float)
        label = row[-1]
        data = vals.reshape(6,6, order='F')  # 앞쪽이 1,7,13... 순서
        mask = generate_segmentation_map(label)
        
        all_data.append(data.flatten())
        all_masks.append(mask.flatten())
        all_labels.append(label)
        

        for _ in range(n_augment):
            aug_data, aug_mask = augment_data(data, mask)
            all_data.append(aug_data.flatten())
            all_masks.append(aug_mask.flatten())
            all_labels.append(label + "_aug")
    

    out_df = pd.DataFrame(all_data)
    out_df['mask'] = [m.tolist() for m in all_masks]
    out_df['label'] = all_labels


    out_df = out_df.drop_duplicates(subset=out_df.columns[:-2], keep='first')  # 데이터 중복 제거
    out_df.reset_index(drop=True, inplace=True)

    out_df.to_csv(f'{label}_data.csv', index=False)
    print(f"✅ 저장 완료: {len(out_df)} 샘플 -> {label}_data.csv (중복 제거 완료)")


process_csv("pressure_normal.csv")

