# -*- coding: utf-8 -*-
"""
ratio(과점 비율) vs 기존 품질 피처 상관관계 분석
- input1: 사과 예측 데이터 정리(충주 청송 무처리 30%).csv
- input2: results.csv (img_path, ratio, ...)
- join key:
    기존 CSV의 첫 컬럼(예: 1,2,3...)  <->  results.csv img_path에서 추출한 파일명 숫자(예: 1.jpg -> 1)
- output:
    merged_with_ratio.csv
    corr_pearson.csv / corr_spearman.csv
    corr_bar_pearson.png / corr_bar_spearman.png
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# 경로 설정
# ---------------------------
BASE_CSV = "사과 예측 데이터 정리(충주 청송 무처리 30%).csv"
RESULTS_CSV = "results.csv"
OUT_DIR = "./corr_out"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------
# 유틸: results.csv의 img_path에서 숫자 id 추출
#  - ".../10.jpg" -> 10
#  - 확장자 jpg/png/webp 등 대응
# ---------------------------
def extract_id_from_img_path(p: str):
    if pd.isna(p):
        return np.nan
    s = str(p)
    # 파일명만 분리
    fname = os.path.basename(s)
    # 10.jpg -> 10 (확장자 제거)
    stem = os.path.splitext(fname)[0]
    # stem이 "10"이면 10, stem이 "img_10" 같은 형태면 마지막 숫자 추출
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else np.nan


# ---------------------------
# 1) 데이터 로드
# ---------------------------
# base csv는 맨 앞에 빈 컬럼처럼 보일 수 있으니 자동 처리
base = pd.read_csv(BASE_CSV, encoding="utf-8-sig")
res = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig")

# base의 "첫 컬럼"을 id로 사용 (예시에서 헤더가 비어있거나, 이름이 이상할 수 있음)
base_id_col = base.columns[0]
base = base.rename(columns={base_id_col: "id"})
base["id"] = pd.to_numeric(base["id"], errors="coerce")

# results에서 id 추출
res["id"] = res["img_path"].apply(extract_id_from_img_path)
res["id"] = pd.to_numeric(res["id"], errors="coerce")

# ratio 숫자화
res["ratio"] = pd.to_numeric(res["ratio"], errors="coerce")

# ---------------------------
# 2) 조인 (id 기준)
#    - base에 같은 id가 여러 번 있으면(지금처럼 1이 두 줄) ratio가 동일하게 붙음
# ---------------------------
merged = base.merge(res[["id", "ratio"]], on="id", how="left")

# 저장
merged_path = os.path.join(OUT_DIR, "merged_with_ratio.csv")
merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
print(f"[OK] merged 저장: {merged_path}")

# ---------------------------
# 3) 상관관계 분석 (ratio vs numeric features)
# ---------------------------
# 숫자 컬럼만 후보로 잡기 (id, ratio 제외)
numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in ("id", "ratio")]

# ratio가 없는 행 제거(매칭 실패한 행)
df = merged.dropna(subset=["ratio"]).copy()

# Pearson / Spearman 계산
pearson = df[feature_cols].corrwith(df["ratio"], method="pearson").sort_values(key=lambda s: s.abs(), ascending=False)
spearman = df[feature_cols].corrwith(df["ratio"], method="spearman").sort_values(key=lambda s: s.abs(), ascending=False)

pearson_df = pearson.reset_index()
pearson_df.columns = ["feature", "corr_pearson"]

spearman_df = spearman.reset_index()
spearman_df.columns = ["feature", "corr_spearman"]

pearson_out = os.path.join(OUT_DIR, "corr_pearson.csv")
spearman_out = os.path.join(OUT_DIR, "corr_spearman.csv")

pearson_df.to_csv(pearson_out, index=False, encoding="utf-8-sig")
spearman_df.to_csv(spearman_out, index=False, encoding="utf-8-sig")

print(f"[OK] Pearson 저장: {pearson_out}")
print(f"[OK] Spearman 저장: {spearman_out}")

# ---------------------------
# 4) 시각화(절대값 큰 순 Top N)
# ---------------------------
TOPN = min(15, len(feature_cols))

def plot_bar(corr_series, title, out_path):
    top = corr_series.iloc[:TOPN][::-1]  # 보기 좋게 역순
    plt.figure(figsize=(10, 0.5 * TOPN + 2))
    plt.barh(top.index, top.values)
    plt.title(title)
    plt.xlabel("correlation with ratio")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] plot 저장: {out_path}")

plot_bar(pearson, f"Top {TOPN} Pearson corr with ratio", os.path.join(OUT_DIR, "corr_bar_pearson.png"))
plot_bar(spearman, f"Top {TOPN} Spearman corr with ratio", os.path.join(OUT_DIR, "corr_bar_spearman.png"))

# ---------------------------
# 5) 콘솔 요약 출력
# ---------------------------
print("\n=== Top Pearson ===")
print(pearson.head(TOPN))

print("\n=== Top Spearman ===")
print(spearman.head(TOPN))

print("\n완료! merged_with_ratio.csv / corr_*.csv / corr_bar_*.png 를 확인하세요.")
