#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="예: test_input.csv")
    parser.add_argument("--base", default="melon", help="webp 탐색 시작 폴더 (기본: melon)")
    parser.add_argument("--col", default="cutout_webp_0", help="추가/덮어쓸 컬럼명 (기본: cutout_webp_0)")
    parser.add_argument("--backup", action="store_true", help="백업 파일(.bak.csv)도 저장")
    args = parser.parse_args()

    base = Path(".") / args.base

    # melon 아래 full_webp 폴더들 찾기
    full_webp_dirs = sorted([p for p in base.rglob("full_webp") if p.is_dir()])

    # 파일명이 "_0.webp" 로 끝나는 것만 수집 (폴더 순서 -> 파일명 순서)
    webp_names_0 = []
    for d in full_webp_dirs:
        webp_names_0.extend([p.name for p in sorted(d.glob("*_0.webp"))])

    # CSV 읽고 새 컬럼에 순서대로 채우기
    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)

    df[args.col] = ""
    n = min(len(df), len(webp_names_0))
    if n > 0:
        df.loc[: n - 1, args.col] = webp_names_0[:n]

    # 저장
    if args.backup:
        backup_path = csv_path.with_suffix(".bak.csv")
        df.to_csv(backup_path, index=False)
        print(f"backup saved: {backup_path}")

    df.to_csv(csv_path, index=False)

    print(f"found '*_0.webp' files: {len(webp_names_0)}")
    print(f"rows in csv: {len(df)}")
    print(f"filled rows: {n}")
    print(f"updated saved: {csv_path}")

if __name__ == "__main__":
    main()


"""
python .\find_child_webp.py test_input.csv
이러면 test_input.csv 파일에 cutout_webp_0 컬럼이 추가되고
melon 폴더 아래에서 찾은 *_0.webp 파일명이 순서대로 채움
이 test_input.csv 파일을 run_quality_pipeline_flat.py 에서 사용하게 할거임
제대로 작동하는지 보려면 test_input.csv 파일에서 cutout_webp_0에 있는 데이터 오염시키고 다시 실행하면
다시 덮어써지는지 확인하면 됨
"""