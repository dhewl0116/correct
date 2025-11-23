import requests
import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ===== 설정 =====
url = "https://correct.lyj.kr/data"  # 센서 데이터 GET URL
samples_per_label = 100               # 한 라벨당 수집 샘플 수
output_csv = "pressure_left_diagonal.csv"     # 저장 파일명
all_data = []

# ===== 수집할 라벨 입력 =====
labels = input("수집할 라벨 이름을 ,로 구분하여 입력하세요: ").split(',')

for label in labels:
    label = label.strip()
    print(f"\n[Collecting] 라벨: {label}")
    for i in range(samples_per_label):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()  # JSON 배열
            data = np.array(data, dtype=np.float32)
            if len(data) != 36:
                print(f"Warning: length {len(data)} != 36, skipped")
                continue
            # 라벨 추가
            all_data.append(np.append(data, label))
            print(i)
        except Exception as e:
            print(f"Error fetching sample {i}: {e}")
        time.sleep(0.5)  # 요청 간 딜레이

# ===== CSV 저장 =====
columns = [f"s{i+1}" for i in range(36)] + ["label"]
df = pd.DataFrame(all_data, columns=columns)
df.to_csv(output_csv, index=False)
print(f"\n[Done] CSV 저장 완료: {output_csv}")

