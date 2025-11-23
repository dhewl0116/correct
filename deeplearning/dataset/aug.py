import pandas as pd
import numpy as np
import torch
import random

# ===== 1️⃣ 2D 시프트 함수 =====
def shift_tensor_2d(arr_2d, dy, dx, fill_val=0.0):
    if isinstance(arr_2d, np.ndarray):
        arr_2d = torch.tensor(arr_2d)
    H, W = arr_2d.shape
    out = torch.full((H, W), fill_val, dtype=arr_2d.dtype)
    src_y0, src_y1 = max(0, -dy), min(H, H - dy)
    src_x0, src_x1 = max(0, -dx), min(W, W - dx)
    dst_y0, dst_y1 = max(0, dy), min(H, H + dy)
    dst_x0, dst_x1 = max(0, dx), min(W, W + dx)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr_2d[src_y0:src_y1, src_x0:src_x1]
    return out

# ===== 2️⃣ 단일 샘플 증강 =====
def augment_single(px_flat, mask_flat,
                   num_shift_variants=1,
                   num_noise_variants=1,
                   noise_std=0.05):
    """
    한 샘플에서 여러 버전을 생성:
      - 원본
      - shift n개
      - noise n개
    """
    results = []

    # ① 원본 그대로 추가
    results.append((px_flat.copy(), mask_flat.copy(), "original"))

    # ② 시프트 증강
    for i in range(num_shift_variants):
        while True:
            dy, dx = random.randint(-1, 1), random.randint(-1, 1)
            if dy != 0 or dx != 0:
                break
        px_2d = torch.tensor(px_flat.reshape(6,6))
        mask_2d = torch.tensor(mask_flat.reshape(6,6))
        px_s = shift_tensor_2d(px_2d, dy, dx, fill_val=0.0)
        mask_s = shift_tensor_2d(mask_2d, dy, dx, fill_val=0.0)
        results.append((px_s.reshape(-1).numpy(), mask_s.reshape(-1).numpy(), f"shift({dy},{dx})"))

    # ③ 노이즈 증강
    for j in range(num_noise_variants):
        px_2d = torch.tensor(px_flat.reshape(6,6))
        noise = torch.randn_like(px_2d) * noise_std
        px_n = torch.clamp(px_2d + noise, min=0.0)
        results.append((px_n.reshape(-1).numpy(), mask_flat.copy(), f"noise({j})"))

    return results


# ===== 3️⃣ 전체 데이터셋 증강 =====
def build_augmented_dataset(
    in_csv="posture_seg_dataset.csv",
    out_csv="posture_seg_dataset_augmented.csv",
    num_shift_variants=1,
    num_noise_variants=1,
    noise_std=0.05,
    shuffle_seed=123
):
    df = pd.read_csv(in_csv)
    px_cols   = [f"px_{i}" for i in range(36)]
    mask_cols = [f"mask_{i}" for i in range(36)]

    px_data   = df[px_cols].to_numpy(np.float32)
    mask_data = df[mask_cols].to_numpy(np.float32)
    metas = df["posture_meta"].tolist() if "posture_meta" in df.columns else ["unknown"] * len(df)

    new_rows = []

    for idx in range(len(df)):
        px_flat, mask_flat, meta = px_data[idx], mask_data[idx], metas[idx]
        aug_list = augment_single(px_flat, mask_flat,
                                  num_shift_variants=num_shift_variants,
                                  num_noise_variants=num_noise_variants,
                                  noise_std=noise_std)

        for px_aug, mask_aug, aug_type in aug_list:
            rec = {f"px_{i}": float(px_aug[i]) for i in range(36)}
            rec.update({f"mask_{i}": float(mask_aug[i]) for i in range(36)})
            rec["posture_meta"] = meta
            rec["aug_type"] = aug_type  # ← 어떤 변형인지 기록
            new_rows.append(rec)

    out_df = pd.DataFrame(new_rows)
    out_df = out_df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)

    print(f"✅ 증강 데이터셋 저장 완료: {out_csv}")
    print(f"원본 샘플 수: {len(df)} → 최종 {len(out_df)}개  (x{len(out_df)/len(df):.1f})")
    print(out_df.head())

# ===== 4️⃣ 실행 예시 =====
build_augmented_dataset(
    in_csv="posture_seg_dataset.csv",
    out_csv="posture_seg_dataset_augmented.csv",
    num_shift_variants=1,   # shift 1개
    num_noise_variants=1,   # noise 1개
    noise_std=0.05
)
