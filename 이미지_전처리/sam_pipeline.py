# -*- coding: utf-8 -*-
"""
YOLO → SAM 컷아웃 자동화 (dual mode + directory mirroring)
==========================================================
1️⃣ YOLO가 fruit box/mask 탐지
2️⃣ SAM이 YOLO box를 seed로 받아 정밀 경계 생성
3️⃣ 두 가지 버전 동시 저장:
    - crop 중심형 (과일 중심)
    - full-mask형 (원본 크기 유지)
4️⃣ 원본 images_all의 디렉토리 구조를 cutout_results에도 그대로 유지
----------------------------------------------------------
폴더 구조 예시:
models/
 ├── melon/
 │    ├── yolo_melon.pt
 │    ├── sam_vit_b_01ec64.pth
 │    └── sam_mask_decoder_melon_final.pth
 ├── apple/
 │    ├── yolo_apple.pt
 │    ├── sam_vit_b_01ec64.pth
 │    └── sam_mask_decoder_apple_final.pth
"""

import os, cv2, torch, numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

# ===============================================
# 설정
# ===============================================

# 현재 스크립트 경로 기준
current_dir = os.path.dirname(os.path.abspath(__file__))

# C# 또는 커맨드라인에서 --fruit 옵션으로 전달 가능하게
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fruit", type=str, default=None, help="Fruit name (melon/apple/mandarin/garlic/onion...)")
args = parser.parse_args()

if args.fruit:
    fruit = args.fruit.lower()
else:
    # C#에서 전달되지 않으면 상위 폴더명 사용 (예: ./melon/sam_pipeline.py)
    fruit = os.path.basename(os.path.dirname(current_dir)).lower()

# 모델 디렉토리
MODEL_DIR = os.path.join(current_dir, "models", fruit)
if not os.path.exists(MODEL_DIR):
    print(f"⚠️ Model folder not found: {MODEL_DIR}")
    print("❌ Please create models/{fruit}/ and put .pt/.pth files inside.")
    exit(1)

YOLO_WEIGHT = os.path.join(MODEL_DIR, f"yolo_{fruit}.pt")
SAM_BASE_CKPT = os.path.join(MODEL_DIR, "sam_vit_b_01ec64.pth")
SAM_DECODER_PTH = os.path.join(MODEL_DIR, f"sam_mask_decoder_{fruit}_final.pth")

INPUT_DIR = os.path.join(current_dir, "images_all")
OUTPUT_DIR = os.path.join(current_dir, "cutout_results")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================================
# 모델 로드
# ===============================================

print(f"🚀 Loading YOLO + SAM models for '{fruit}'...")

# YOLO 로드
if not os.path.isfile(YOLO_WEIGHT):
    print(f"❌ YOLO weight not found: {YOLO_WEIGHT}")
    exit(1)

yolo = YOLO(YOLO_WEIGHT)

# SAM 모델 로드
SAM_MODEL_TYPE = "vit_b"
if not os.path.isfile(SAM_BASE_CKPT):
    print(f"❌ SAM base checkpoint not found: {SAM_BASE_CKPT}")
    exit(1)

sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_BASE_CKPT)

# fine-tuned decoder 불러오기
if os.path.isfile(SAM_DECODER_PTH):
    try:
        state = torch.load(SAM_DECODER_PTH, map_location="cpu")
        sam.mask_decoder.load_state_dict(state, strict=True)
        print(f"✅ Loaded fine-tuned SAM decoder: {SAM_DECODER_PTH}")
    except Exception as e:
        print(f"⚠️ Decoder load failed ({SAM_DECODER_PTH}): {e}")
else:
    print(f"⚠️ Fine-tuned decoder for '{fruit}' not found, using base SAM only.")

sam.to(device)
predictor = SamPredictor(sam)

# ===============================================
# 유틸 함수
# ===============================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def mirror_structure(image_path, root_input=INPUT_DIR, root_output=OUTPUT_DIR):
    """
    images_all 내부 구조를 그대로 cutout_results 하위에 반영
    """
    rel_path = os.path.relpath(os.path.dirname(image_path), root_input)
    crop_dir = os.path.join(root_output, fruit, "crop", rel_path)
    full_dir = os.path.join(root_output, fruit, "full", rel_path)

    ensure_dir(os.path.join(crop_dir, "cutout_webp"))
    ensure_dir(os.path.join(crop_dir, "crop_mask"))
    ensure_dir(os.path.join(full_dir, "full_webp"))
    ensure_dir(os.path.join(full_dir, "full_mask"))

    return crop_dir, full_dir

# ===============================================
# YOLO → SAM 컷아웃
# ===============================================

def yolo_sam_cutout(image_path):
    print(f"▶ Processing {os.path.basename(image_path)}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Failed to read: {image_path}")
        return 0

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = yolo.predict(source=img_rgb, conf=0.3, verbose=False)

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print(f"⚠️ No boxes detected: {image_path}")
        return 0

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    print(f"🔎 Detected {len(boxes)} boxes")

    crop_dir, full_dir = mirror_structure(image_path)
    count = 0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        predictor.set_image(img_rgb)
        input_box = np.array([x1, y1, x2, y2])
        masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
        best_mask = masks[np.argmax(scores)].astype(np.uint8)

        base = os.path.splitext(os.path.basename(image_path))[0]

        # 1️⃣ Crop 중심형
        crop_rgb = img[y1:y2, x1:x2]
        mask_crop = best_mask[y1:y2, x1:x2] * 255
        rgba_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2BGRA)
        rgba_crop[:, :, 3] = mask_crop

        cv2.imwrite(
            f"{crop_dir}/cutout_webp/{base}_{i}.webp",
            rgba_crop,
            [cv2.IMWRITE_WEBP_QUALITY, 95]
        )
        cv2.imwrite(f"{crop_dir}/crop_mask/{base}_{i}.png", mask_crop)

        # 2️⃣ Full-mask형
        mask_full = best_mask * 255
        rgba_full = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba_full[:, :, 3] = mask_full

        cv2.imwrite(
            f"{full_dir}/full_webp/{base}_{i}.webp",
            rgba_full,
            [cv2.IMWRITE_WEBP_QUALITY, 95]
        )
        cv2.imwrite(f"{full_dir}/full_mask/{base}_{i}.png", mask_full)

        count += 1

    print(f"✅ Saved {count} cutouts (crop+full) for {os.path.basename(image_path)}")
    return count

# ===============================================
# 실행
# ===============================================

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input folder not found: {INPUT_DIR}")
        return

    images = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".png")):
                images.append(os.path.join(root, f))

    if not images:
        print(f"⚠️ No images found under {INPUT_DIR}")
        return

    total = 0
    for f in tqdm(images, desc=f"Processing ({fruit})"):
        n = yolo_sam_cutout(f)
        total += n

    print(f"✅ Total cutouts saved for '{fruit}': {total}")

if __name__ == "__main__":
    main()
