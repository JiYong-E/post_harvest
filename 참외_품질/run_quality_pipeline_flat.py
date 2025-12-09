# -*- coding: utf-8 -*-
import os
import re
import argparse
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# =========================
# Utils
# =========================
def triage_from_pred(pred: float, pass_t: float, fail_t: float) -> str:
    if pred >= pass_t:
        return "PASS"
    if pred < fail_t:
        return "FAIL"
    return "GRAY"


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def infer_mask_name_from_webp_same_stem(webp_name: str) -> str:
    # 20240612_..._1_0.webp -> 20240612_..._1_0.png  (stem 유지)
    bn = os.path.basename(str(webp_name).strip())
    stem, _ = os.path.splitext(bn)
    return stem + ".png"


# =========================
# Shape model (ResNet18 2-class)
# label: 0=상, 1=특
# =========================
def load_shape_model(pt_path: str, device: torch.device):
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, 2)

    ckpt = torch.load(pt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    return model, tfm


def load_rgb(path: str):
    if not path or (not os.path.exists(path)):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def load_mask(path: str):
    if not path or (not os.path.exists(path)):
        return None
    try:
        return Image.open(path).convert("L")
    except Exception:
        return None


def apply_mask(rgb_img: Image.Image, mask_img: Image.Image) -> Image.Image:
    rgb = np.array(rgb_img, dtype=np.uint8)
    m = np.array(mask_img, dtype=np.uint8)

    if (rgb.shape[0] != m.shape[0]) or (rgb.shape[1] != m.shape[1]):
        mask_img = mask_img.resize((rgb.shape[1], rgb.shape[0]), resample=Image.NEAREST)
        m = np.array(mask_img, dtype=np.uint8)

    m01 = (m > 0).astype(np.uint8)
    rgb2 = rgb * m01[..., None]
    return Image.fromarray(rgb2.astype(np.uint8))


def predict_shape_one(
    model,
    tfm,
    img_path: str,
    device: torch.device,
    use_mask: bool = False,
    mask_path: str = "",
    swap_label: bool = False,
):
    rgb_img = load_rgb(img_path)
    if rgb_img is None:
        return "NOIMG", None

    if use_mask:
        m = load_mask(mask_path)
        if m is None:
            return "NOMASK", None
        rgb_img = apply_mask(rgb_img, m)

    x = tfm(rgb_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(prob.argmax())
        conf = float(prob[pred])

    if not swap_label:
        label = "상" if pred == 0 else "특"
    else:
        label = "특" if pred == 0 else "상"

    return label, conf


# =========================
# Sweet model
# =========================
def _required_cols_from_sklearn_pipe(pipe):
    pre = pipe.named_steps.get("pre", None)
    if pre is None:
        for _, step in pipe.steps:
            if step.__class__.__name__ == "ColumnTransformer":
                pre = step
                break
    if pre is None:
        raise ValueError("Pipeline에서 ColumnTransformer(pre)를 찾지 못했습니다.")

    cols = []
    trans_list = pre.transformers_ if hasattr(pre, "transformers_") else pre.transformers
    for name, _trans, colspec in trans_list:
        if name == "remainder":
            continue
        if isinstance(colspec, (list, tuple)):
            cols.extend(list(colspec))
        else:
            try:
                cols.extend(list(colspec))
            except Exception:
                pass
    return sorted(set([str(c) for c in cols]))


def predict_sweet(sweet_pipe, df: pd.DataFrame) -> np.ndarray:
    need = _required_cols_from_sklearn_pipe(sweet_pipe)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) for sweet model: {missing}")
    X = df[need].copy()
    return sweet_pipe.predict(X).astype(float)


# =========================
# FLAT index (recurse)
# =========================
def build_file_index(flat_root: str, exts=(".webp", ".png")):
    """
    flat_root 아래 전부 돌면서
    - 전체 파일명
    - '__' 뒤 suffix 파일명
    둘 다 키로 저장해서 매칭률 올림.
    """
    idx = {}
    n = 0

    def add_key(key, fullpath):
        if key not in idx:
            idx[key] = [fullpath]
        else:
            idx[key].append(fullpath)

    for dirpath, _dirnames, filenames in os.walk(flat_root):
        for fn in filenames:
            lf = fn.lower()
            if not any(lf.endswith(e) for e in exts):
                continue

            full = os.path.join(dirpath, fn)
            n += 1

            # 1) 전체 파일명 키
            add_key(fn, full)

            # 2) '__' 뒤 suffix 키 (있으면)
            if "__" in fn:
                suffix = fn.split("__", 1)[1]
                add_key(suffix, full)

    return idx, n



def pick_first(idx_map, filename: str):
    if not filename:
        return ""

    fn = os.path.basename(str(filename).strip())
    if not fn:
        return ""

    # 1) 정확히 파일명으로 찾기
    lst = idx_map.get(fn, [])
    if lst:
        return lst[0]

    # 2) 혹시 CSV에 경로가 포함되어 있고 그 안에 '__' 가 있으면 suffix로도 시도
    if "__" in fn:
        suf = fn.split("__", 1)[1]
        lst = idx_map.get(suf, [])
        if lst:
            return lst[0]

    # 3) 마지막 fallback: stem(확장자 제거)로 유사검색 (느릴 수 있지만 안전)
    stem = os.path.splitext(fn)[0]
    # idx_map 키들 중 stem으로 끝나는 첫 번째를 반환
    for k, paths in idx_map.items():
        if os.path.splitext(k)[0].endswith(stem):
            return paths[0]

    return ""



# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--sweet_model", required=True)
    ap.add_argument("--shape_model_pt", required=True)

    ap.add_argument("--pass_t", type=float, default=12.65)
    ap.add_argument("--fail_t", type=float, default=11.25)
    ap.add_argument("--target", type=float, default=12.0)

    ap.add_argument("--img_col", default="cutout_webp_0")     # 파일명만 있어도 됨
    ap.add_argument("--mask_col", default="")                 # 보통 비워둠

    ap.add_argument("--flat_root", required=True)             # ex) D:\과일\_FLAT_OUT_FULL
    ap.add_argument("--use_mask", action="store_true")

    ap.add_argument("--second_pass_mode", default="avg", choices=["avg", "max", "min", "second"])
    ap.add_argument("--sweet_pred2_col", default="")

    ap.add_argument("--encoding", default="utf-8-sig")
    ap.add_argument("--swap_shape_label", action="store_true")

    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding=args.encoding)
    df.columns = [str(c).strip() for c in df.columns]

    # 1) sweet
    sweet_pipe = joblib.load(args.sweet_model)
    df["sweet_pred_1"] = predict_sweet(sweet_pipe, df)

    # 2) triage 1
    df["triage_1"] = df["sweet_pred_1"].apply(lambda v: triage_from_pred(float(v), args.pass_t, args.fail_t))

    # 3) GRAY second pass (optional)
    gray_mask = (df["triage_1"].values == "GRAY")
    df["sweet_pred_2"] = np.nan
    df["sweet_pred_final"] = df["sweet_pred_1"].astype(float)

    if gray_mask.any() and args.sweet_pred2_col and (args.sweet_pred2_col in df.columns):
        df.loc[gray_mask, "sweet_pred_2"] = df.loc[gray_mask, args.sweet_pred2_col].apply(safe_float).values
        p1 = df.loc[gray_mask, "sweet_pred_1"].astype(float).values
        p2 = df.loc[gray_mask, "sweet_pred_2"].astype(float).values

        def combine(a, b):
            if not (b == b):
                b = a
            if args.second_pass_mode == "avg":
                return 0.5 * (a + b)
            if args.second_pass_mode == "max":
                return max(a, b)
            if args.second_pass_mode == "min":
                return min(a, b)
            return b  # second

        df.loc[gray_mask, "sweet_pred_final"] = np.array([combine(float(a), float(b)) for a, b in zip(p1, p2)], dtype=float)

    # 4) final triage
    df["triage_final"] = df["sweet_pred_final"].apply(lambda v: triage_from_pred(float(v), args.pass_t, args.fail_t))

    # 5) build index once
    print(f"Indexing flat_root: {args.flat_root}")
    idx_map, nfiles = build_file_index(args.flat_root, exts=(".webp", ".png"))
    print(f"Indexed files: {nfiles} (unique names: {len(idx_map)})")

    # 6) shape model (PASS only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    shape_model, shape_tfm = load_shape_model(args.shape_model_pt, device)

    df["shape_pred"] = ""
    df["shape_conf"] = np.nan
    df["img_path"] = ""
    if args.use_mask:
        df["mask_path"] = ""

    pass_mask = (df["triage_final"].values == "PASS")
    idxs = np.where(pass_mask)[0]

    # 통계
    noimg = 0
    nomask = 0

    for i in tqdm(idxs, desc="Shape(PASS only)"):
        img_name = str(df.at[i, args.img_col]).strip()
        img_p = pick_first(idx_map, img_name)
        df.at[i, "img_path"] = img_p

        mask_p = ""
        if args.use_mask:
            if args.mask_col and (args.mask_col in df.columns):
                mask_name = str(df.at[i, args.mask_col]).strip()
            else:
                mask_name = infer_mask_name_from_webp_same_stem(img_name)
            mask_p = pick_first(idx_map, mask_name)
            df.at[i, "mask_path"] = mask_p

        pred, conf = predict_shape_one(
            shape_model, shape_tfm, img_p, device,
            use_mask=args.use_mask, mask_path=mask_p,
            swap_label=args.swap_shape_label
        )

        df.at[i, "shape_pred"] = pred
        df.at[i, "shape_conf"] = conf if conf is not None else np.nan

        if pred == "NOIMG":
            noimg += 1
        if pred == "NOMASK":
            nomask += 1

    # 7) summary
    total = len(df)
    c1 = df["triage_1"].value_counts()
    cf = df["triage_final"].value_counts()
    print("\n=== Triage counts ===")
    print(f"Total: {total}")
    print("[Stage1]", c1.to_dict())
    print("[Final ]", cf.to_dict())
    print(f"Coverage(final decided) = {((cf.get('PASS', 0) + cf.get('FAIL', 0)) / total):.3%}")
    if pass_mask.any():
        print(f"Shape NOIMG: {noimg}/{len(idxs)}   NOMASK: {nomask}/{len(idxs)}")

    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()



"""
사용법:
python .\run_quality_pipeline_flat.py `
  --csv .\test_input.csv `
  --out .\test_output.csv `
  --sweet_model .\best_abs_sweet_model_min.joblib `
  --shape_model_pt .\참외_shape_model_2class_color_new.pt `
  --img_col cutout_webp_0 `
  --flat_root "D:\과일\_FLAT_OUT_FULL" `
  --use_mask

여기서 중요한건 --csv, flat_root 옵션
왜냐면 --csv 파일에서 cutout_webp_0 컬럼에 들어있는 파일명들을
flat_root 폴더 아래에서 찾아서 shape 모델에 넣어주기 때문임
flat_root 폴더에서 full_mask랑 full_webp 폴더들에서 마스크랑 참외 사진 가져와서
모델을 돌리는거임
따라서 유동적으로 사용할려면

--csv {csv 파일 경로} <-- 여기에 우리가 원하는 csv 파일 경로 넣어주고, 또는 이름
--flat_root {flat_root 폴더 경로} <-- 여기에 우리가 사용하는 flat_root 폴더 경로

를 바꿔주면 됨
근데 앞에서 참외 배경 제거 모델을 돌리면 규칙이 melon 폴더 아래에 여러 하위 폴더 또는 바로
폴더에 full_webp, full_mask 폴더가 생기는데
그 폴더들에서 파일들을 찾아야 하므로
내가 보기에는 flat_root 폴더를 melon으로 고정하는게 좋아보임
--flat_root "melon"
이렇게 하면 melon 폴더 아래에서 full_webp, full_mask 폴더들을 다 뒤져서 파일들을 찾게 됨

find_child_webp.py 같은 경우 앞에서 참외 배경 제거 모델 돌린 후에
melon 폴더 아래에서 *_0.webp 파일들을 찾아서
미리 해당 준비된 화방이나 LAB 값들이 들어있는 csv 파일에
cutout_webp_0 컬럼으로 추가해주는 스크립트임

전체 파이프 라인을 말하자면
1) 참외 배경 제거 모델 돌려서 melon 폴더 아래에 full_webp, full_mask 폴더들 생성
2) find_child_webp.py 스크립트로 csv 파일에 cutout_webp_0 컬럼 추가
3) run_quality_pipeline_flat.py 스크립트로 최종 품질 판정 실행
"""