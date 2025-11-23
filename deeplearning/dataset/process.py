import pandas as pd
import numpy as np

# === 1. 세그멘테이션 마스크 템플릿 정의 ===
# 좌표계 기준 (중요):
# - 데이터를 (6,6)으로 reshape 했을 때 arr[row, col]
# - col 0 (값 1,7,13,...) 이 사람 다리 쪽(앞쪽)
# - row 0은 몸의 왼 엉덩이 쪽, row 5는 몸의 오른 엉덩이 쪽이라고 본다
# ⇒ 따라서:
#   front  = 앞쪽으로 쏠림 → col 0~1 전체 1
#   left   = 왼 엉덩이 쏠림 → row 0~1 전체 1
#   right  = 오른 엉덩이 쏠림 → row 4~5 전체 1
#   normal = 교정 불필요 → 전부 0

def mask_front():
    m = np.zeros((6,6), dtype=np.float32)
    m[:, 0:2] = 1.0  # 앞쪽(다리 방향) 영역 강조
    return m.reshape(-1)  # (36,)

def mask_left():
    m = np.zeros((6,6), dtype=np.float32)
    m[0:2, :] = 1.0  # 몸의 왼쪽 골반/엉덩이 쪽
    return m.reshape(-1)

def mask_right():
    m = np.zeros((6,6), dtype=np.float32)
    m[4:6, :] = 1.0  # 몸의 오른쪽 골반/엉덩이 쪽
    return m.reshape(-1)

def mask_normal():
    m = np.zeros((6,6), dtype=np.float32)
    return m.reshape(-1)

MASK_BY_POSTURE = {
    "front":  mask_front(),
    "left":   mask_left(),
    "right":  mask_right(),
    "normal": mask_normal(),
}

# === 2. CSV 로드 & 전처리 함수 ===
def load_pressure_csv(path):
    """
    1) CSV 읽기 (header 없음 가정)
    2) 앞 36개 컬럼(센서값)만 남기고 float로 변환
    3) 완전 빈 행 제거
    4) 중복 패턴 제거
    """
    df = pd.read_csv(path, header=None)
    df36 = df.iloc[:, :36].copy()
    df36 = df36.apply(pd.to_numeric, errors="coerce")
    df36 = df36.dropna(how="all").reset_index(drop=True)
    df36 = df36.drop_duplicates().reset_index(drop=True)
    return df36  # shape: (N_unique, 36)

# === 3. posture별 샘플들에 마스크를 붙여서 row dict로 변환 ===
def expand_with_mask(df_src, posture_tag):
    """
    df_src: (N,36) DataFrame (중복 제거 완료된 전체 샘플)
    posture_tag: "front" / "left" / "right" / "normal"
    return: list of dicts with:
      px_0..px_35  -> 입력 압력값
      mask_0..mask_35 -> 같은 posture 공통 segmentation GT
      posture_meta -> 원 posture 태그 (디버깅용)
    """
    rows = []
    fixed_mask = MASK_BY_POSTURE[posture_tag]  # shape (36,)
    for _, row in df_src.iterrows():
        vals = row.to_numpy(dtype=float)  # (36,)
        rec = {}
        # 입력 (압력)
        for i in range(36):
            v = vals[i]
            rec[f"px_{i}"] = float(v) if not np.isnan(v) else 0.0
        # 출력 (세그멘테이션 라벨)
        for i in range(36):
            rec[f"mask_{i}"] = float(fixed_mask[i])
        # 추가로 posture 이름은 기록만
        rec["posture_meta"] = posture_tag
        rows.append(rec)
    return rows

# === 4. 실제로 불러오기 ===
front_df  = load_pressure_csv("pressure_front.csv")
left_df   = load_pressure_csv("pressure_left.csv")
right_df  = load_pressure_csv("pressure_right.csv")
normal_df = load_pressure_csv("pressure_normal.csv")

# 여기서 더 이상 샘플 줄이지 않음 (전부 사용)
# take_representative(...) 삭제

# === 5. posture별로 마스크 붙여서 하나로 합치기 ===
rows_all = []
rows_all += expand_with_mask(front_df,  "front")
rows_all += expand_with_mask(left_df,   "left")
rows_all += expand_with_mask(right_df,  "right")
rows_all += expand_with_mask(normal_df, "normal")

final_df = pd.DataFrame(rows_all)

# 섞기 (학습 시 batch마다 posture 섞이게 하려고)
final_df = final_df.sample(frac=1, random_state=123).reset_index(drop=True)

# === 6. 최종 데이터셋 저장 ===
final_df.to_csv("posture_seg_dataset.csv", index=False)

print("완료: posture_seg_dataset.csv 생성됨")
print(final_df.head())
print("샘플 개수:", len(final_df))
print("posture_meta 분포:")
print(final_df["posture_meta"].value_counts())
