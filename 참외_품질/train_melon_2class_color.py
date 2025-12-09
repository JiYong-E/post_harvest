# -*- coding: utf-8 -*-
"""
참외 모양 + 색상 분류 (2-class: 상 vs 특)
========================================
폴더 구조 예시:

D:\과일\참외_cutout\melon\crop\2025 참외 고해상도영상\1,2,3화방\4도\250703 4도_1w\스마트팜(특)\2024_yy_...\crop_mask
D:\과일\참외_cutout\melon\crop\2025 참외 고해상도영상\1,2,3화방\4도\250703 4도_1w\스마트팜(특)\2024_yy_...\cutout_webp

- 마스크: crop_mask\20250703_152137_843_0_0.png
- RGB  : cutout_webp\20250703_152137_843_0_0.webp
"""

import os
import random
from collections import Counter

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
from tqdm import tqdm


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """한글/공백 경로에서도 안전하게 이미지를 읽기 위한 함수."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception as e:
        print(f"⚠ 이미지 읽기 실패(imread_unicode): {path} ({e})")
        return None


# ============================================================
# 설정
# ============================================================

# crop 루트 (여기 아래를 돌면서 crop_mask를 찾음)
ROOT_DIR = r"D:\과일\_FLAT_OUT_FULL"

SAVE_PATH = r"D:\과일\참외_shape_model_2class_color_new.pt"

NUM_EPOCHS = 20
BATCH_SIZE = 64
VAL_RATIO = 0.2
LR = 1e-4
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 등급 파싱
# ============================================================

def parse_grade_from_path(path: str):
    """
    경로에서 등급(상/특)을 파싱.
    - ...\상\... 이면 '상'
    - ...\특\... 이면 '특'
    """
    parts = os.path.normpath(path).split(os.sep)
    if "특" in parts:
        return "특"
    if "상" in parts:
        return "상"
    return None

# """
#     flat 구조에서 full_mask와 full_webp를 파일명으로 매칭한다.
#     - mask: ...\상\full_mask\NAME.png
#     - rgb : ...\상\full_webp\NAME.webp
#     label: 0(상), 1(특)
# """

# ============================================================
# 샘플 수집: crop_mask → cutout_webp 매핑
# ============================================================

def collect_samples_flat_full(root_dir):


    samples = []
    skip_no_webp = 0
    skip_no_mask = 0
    debug_printed = 0

    for grade in ["상", "특"]:
        mask_dir = os.path.join(root_dir, grade, "full_mask")
        webp_dir = os.path.join(root_dir, grade, "full_webp")

        if not os.path.isdir(mask_dir) or not os.path.isdir(webp_dir):
            print(f"⚠ 폴더 없음 스킵: {grade}")
            print("  mask_dir:", mask_dir)
            print("  webp_dir:", webp_dir)
            continue

        label = 0 if grade == "상" else 1

        # 1) 마스크 기준 매칭
        for fname in os.listdir(mask_dir):
            if not fname.lower().endswith(".png"):
                continue

            base = os.path.splitext(fname)[0]
            mask_path = os.path.join(mask_dir, fname)
            rgb_path = os.path.join(webp_dir, base + ".webp")

            if not os.path.isfile(rgb_path):
                skip_no_webp += 1
                if debug_printed < 10:
                    print("⚠ webp 없음(마스크만 존재) 예시:")
                    print("  mask:", mask_path)
                    print("  webp:", rgb_path)
                    debug_printed += 1
                continue

            samples.append((rgb_path, mask_path, label))

        # 2) (옵션) webp만 있고 mask 없는 것 카운트(품질 점검용)
        mask_bases = set(os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.lower().endswith(".png"))
        for f in os.listdir(webp_dir):
            if not f.lower().endswith(".webp"):
                continue
            base = os.path.splitext(f)[0]
            if base not in mask_bases:
                skip_no_mask += 1

    print("매칭 실패(마스크는 있는데 webp 없음):", skip_no_webp)
    print("매칭 실패(webp는 있는데 마스크 없음):", skip_no_mask)
    return samples


# 사용
samples = collect_samples_flat_full(ROOT_DIR)
print(f"총 샘플 수: {len(samples)}")
label_counts = Counter([lbl for _, _, lbl in samples])
print("클래스 분포 (0:상, 1:특):", label_counts)

if len(samples) == 0:
    raise RuntimeError("수집된 샘플이 없습니다. ROOT_DIR / flat 폴더 구조를 확인하세요.")

# ============================================================
# Train / Val split
# ============================================================

random.shuffle(samples)
n_total = len(samples)
n_val = int(n_total * VAL_RATIO)
n_train = n_total - n_val

train_samples = samples[:n_train]
val_samples = samples[n_train:]

print(f"train: {len(train_samples)}, val: {len(val_samples)}")


# ============================================================
# Dataset / DataLoader
# ============================================================

transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


class MelonColorMaskDataset(Dataset):
    def __init__(self, samples, transform_img=None):
        self.samples = samples
        self.transform_img = transform_img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, mask_path, label = self.samples[idx]

        # ---------- RGB (.webp) ----------
        if not os.path.exists(rgb_path):
            print(f"⚠ RGB 파일 없음, 더미 사용: {rgb_path}")
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            rgb = imread_unicode(rgb_path, cv2.IMREAD_COLOR)
            if rgb is None:
                print(f"⚠ RGB imread 실패, 더미 사용: {rgb_path}")
                rgb = np.zeros((224, 224, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # ---------- 마스크 (.png, 그레이) ----------
        if not os.path.exists(mask_path):
            print(f"⚠ mask 파일 없음, 더미 사용: {mask_path}")
            mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        else:
            m = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                print(f"⚠ mask imread 실패, 더미 사용: {mask_path}")
                mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
            else:
                mask = m

        # ---------- 리사이즈 ----------
        rgb = cv2.resize(rgb, (224, 224))
        mask = cv2.resize(mask, (224, 224))

        # ---------- 마스크 적용 (배경 제거) ----------
        mask01 = (mask > 0).astype(np.float32)   # 0 or 1
        masked_rgb = rgb.astype(np.float32) * mask01[..., None]

        img_pil = Image.fromarray(masked_rgb.astype(np.uint8))

        if self.transform_img is not None:
            img_tensor = self.transform_img(img_pil)
        else:
            img = masked_rgb.transpose(2, 0, 1) / 255.0
            img_tensor = torch.from_numpy(img).float()

        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor


train_dataset = MelonColorMaskDataset(train_samples, transform_img=transform_img)
val_dataset = MelonColorMaskDataset(val_samples, transform_img=transform_img)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=torch.cuda.is_available())

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())



# ============================================================
# 모델 정의 (ResNet18 + 2-class)
# ============================================================

def create_model(num_classes=2):
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


model = create_model(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ============================================================
# Train / Val 루프
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def main():
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f}")
        print(f"val_loss  : {val_loss:.4f}  val_acc  : {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "epoch": epoch,
            }, SAVE_PATH)
            print(f"✅ Best 모델 갱신! (val_acc={best_val_acc:.4f}) → {SAVE_PATH} 저장")


if __name__ == "__main__":
    main()
