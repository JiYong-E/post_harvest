# -*- coding: utf-8 -*-
"""
YOLO → SAM 컷아웃 자동화 (dual mode + directory mirroring)
======================================================
1️⃣ YOLO가 fruit box/mask 탐지
2️⃣ SAM이 YOLO box를 seed로 받아 정밀 경계 생성
3️⃣ 두 가지 버전 동시 저장:
    - crop 중심형 (과일 중심)
    - full-mask형 (원본 크기 유지)
4️⃣ 원본 images_all의 디렉토리 구조를 cutout_results에도 그대로 유지
"""

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

# ===============================================
# 설정
# ===============================================

# 현재 스크립트가 있는 디렉토리 (예: ~/sam_yolo/melon)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 폴더 이름을 fruit 이름으로 사용 (예: melon)
fruit = os.path.basename(current_dir)  # ← sam_yolo가 아니라 melon이 되도록 수정

# 가중치 / 체크포인트 경로
YOLO_WEIGHT = "./runs/segment/train/weights/yolo_melon.pt"
SAM_BASE_CKPT = "./checkpoints/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
SAM_DECODER_PTH = f"./checkpoints/sam_mask_decoder_{fruit}_final.pth"

# 입력 / 출력 디렉토리 (WSL에서 Windows D: 드라이브 사용)
# INPUT_DIR = "./images_all"
INPUT_DIR = "images_all"
# OUTPUT_DIR = "./cutout_results"
OUTPUT_DIR = "images_all_cutout"

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("Selected device            =", device)

# ===============================================
# 모델 로드
# ===============================================
print(f"🚀 Loading YOLO + SAM models for '{fruit}'...")

# YOLO 로드
yolo = YOLO(YOLO_WEIGHT)
# 필요 시, 명시적으로 GPU로 올리기 (에러 방지를 위해 try/except)
if device == "cuda":
    try:
        yolo.to(device)
    except Exception as e:
        print("⚠️ YOLO .to(cuda) 실패, 기본 설정으로 진행:", e)

# SAM 로드
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_BASE_CKPT)

# Fine-tuned decoder가 있으면 로드
if os.path.isfile(SAM_DECODER_PTH):
    state = torch.load(SAM_DECODER_PTH, map_location="cpu")
    sam.mask_decoder.load_state_dict(state, strict=True)
    print(f"✅ Loaded fine-tuned SAM decoder: {SAM_DECODER_PTH}")
else:
    print(f"⚠️ Fine-tuned decoder for '{fruit}' not found, using base SAM only.")

# SAM을 device로 올리기
sam.to(device)
predictor = SamPredictor(sam)

# 실제로 어떤 디바이스에 올라갔는지 확인용 출력
try:
    yolo_device = next(yolo.model.parameters()).device
except Exception:
    # ultralytics 버전/구조에 따라 model 접근이 다를 수 있으므로 예외 처리
    yolo_device = "unknown"

sam_device = next(sam.parameters()).device

print("YOLO device:", yolo_device)
print("SAM  device:", sam_device)
print("==============================================")


# ===============================================
# 폴더 생성 유틸 (원본 디렉토리 구조 복제)
# ===============================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def mirror_structure(image_path, root_input=INPUT_DIR, root_output=OUTPUT_DIR):
    """
    INPUT_DIR 내부 구조를 그대로 OUTPUT_DIR 하위에 반영
    예)
      INPUT_DIR/2025참외/10도/xxx.png
      → OUTPUT_DIR/melon/crop/2025참외/10도/...
    """
    rel_path = os.path.relpath(os.path.dirname(image_path), root_input)
    crop_dir = os.path.join(root_output, fruit, "crop", rel_path)
    full_dir = os.path.join(root_output, fruit, "full", rel_path)

    ensure_dir(os.path.join(crop_dir, "cutout_webp"))
    ensure_dir(os.path.join(crop_dir, "crop_rgb"))
    ensure_dir(os.path.join(crop_dir, "crop_mask"))
    ensure_dir(os.path.join(full_dir, "full_webp"))
    ensure_dir(os.path.join(full_dir, "full_mask"))

    return crop_dir, full_dir


# ===============================================
# YOLO → SAM 컷아웃
# ===============================================
def yolo_sam_cutout(image_path):
    print(f"▶ Processing {os.path.basename(image_path)}")

    # 이미지 로딩
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Failed to read: {image_path}")
        return 0

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # YOLO 추론 (device 명시)
    results = yolo.predict(
        source=img_rgb,
        conf=0.3,
        verbose=False,
        device=device  # cuda 또는 cpu
    )

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print(f"⚠️ No boxes detected: {image_path}")
        return 0

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    print(f"🔎 Detected {len(boxes)} boxes in {os.path.basename(image_path)}")

    crop_dir, full_dir = mirror_structure(image_path)
    count = 0

    # 한 이미지에서 감지된 box들에 대해 SAM 세그멘테이션
    predictor.set_image(img_rgb)  # ← 이미지마다 한 번만 세팅

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        input_box = np.array([x1, y1, x2, y2])
        masks, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=True
        )
        best_mask = masks[np.argmax(scores)].astype(np.uint8)

        base = os.path.splitext(os.path.basename(image_path))[0]

        # 1️⃣ Crop 중심형 (box 영역만 잘라서 RGBA)
        crop_rgb = img[y1:y2, x1:x2]
        mask_crop = best_mask[y1:y2, x1:x2] * 255
        rgba_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2BGRA)
        rgba_crop[:, :, 3] = mask_crop

        cv2.imwrite(
            f"{crop_dir}/cutout_webp/{base}_{i}.webp",
            rgba_crop,
            [cv2.IMWRITE_WEBP_QUALITY, 95]
        )
        # 필요하면 RGB도 저장하고 싶을 때 주석 해제
        # cv2.imwrite(f"{crop_dir}/crop_rgb/{base}_{i}.png", crop_rgb)
        cv2.imwrite(f"{crop_dir}/crop_mask/{base}_{i}.png", mask_crop)

        # 2️⃣ Full-mask형 (원본 크기 유지, 알파만 마스크로)
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
    images = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".png")):
                images.append(os.path.join(root, f))

    print(f"📂 Found {len(images)} images under INPUT_DIR = {INPUT_DIR}")
    total = 0

    for f in tqdm(images, desc=f"Processing ({fruit})"):
        n = yolo_sam_cutout(f)
        total += n

    print(f"✅ Total cutouts saved for '{fruit}': {total}")

if __name__ == "__main__":
    main()
