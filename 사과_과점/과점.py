import torch
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def detect_apple_with_lenticels(
        img_path,
        model_path,
        output_dir,
        # KMeans 파라미터
        K_lenticel=6,
        lenticel_h_ranges=((0, 180),),
        lenticel_v_min=180,
        lenticel_s_max=100,
        # 로컬 밝기 차이 파라미터
        local_ksize=51,
        local_diff_thresh=10,
        # 정제 파라미터
        refine_ksize=61,
        refine_diff_thresh=6,
        # 형태학적 연산
        morph_open_kernel=(5, 5),
        morph_close_kernel=(3, 3),
):
    """
    사과 검출 후 색상 클러스터링과 로컬 밝기 차이를 이용한 과점 검출
    ⭐ 핵심 개선:
    - Normalized Convolution으로 배경을 완전히 제외한 로컬 평균 계산
    - 테두리 물리적 제거 없이도 윤곽선 오검출 방지
    """

    print("=" * 60)
    print("과점(Lenticel) 검출 - Normalized Convolution 방식")
    print("=" * 60)

    # 1. 이미지 로드
    print("\n[1단계] 이미지 로드 중...")
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {img_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    print(f"   이미지 크기: {w} x {h}")

    # 2. YOLO로 사과 검출
    print("\n[2단계] YOLO로 사과 검출 중...")
    yolo = YOLO(f"{model_path}/yolo_apple.pt")
    results = yolo(image_rgb)[0]

    if len(results.boxes) == 0:
        print("❌ 사과가 검출되지 않았습니다.")
        return None

    print(f"   ✓ 사과 {len(results.boxes)}개 검출됨")

    # 3. SAM으로 사과 마스킹
    print("\n[3단계] SAM으로 사과 마스킹 중...")
    sam = sam_model_registry["vit_b"](checkpoint=f"{model_path}/sam_vit_b_01ec64.pth")
    sam.mask_decoder.load_state_dict(
        torch.load(f"{model_path}/sam_mask_decoder_apple_final.pth", map_location="cpu")
    )
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    box = results.boxes.xyxy[0].cpu().numpy()
    masks, _, _ = predictor.predict(box=box, multimask_output=False)
    apple_mask = masks[0]
    apple_mask_uint8 = (apple_mask * 255).astype(np.uint8)

    # 마스킹된 사과 이미지 생성
    masked_image = image.copy()
    masked_image[~apple_mask] = 0

    masked_rgb = image_rgb.copy()
    masked_rgb[~apple_mask] = 0

    print(f"   ✓ 사과 영역 마스킹 완료")

    # 4. 과점 검출 - 방법 1: 색상 클러스터링 (KMeans)
    print("\n[4단계] KMeans 클러스터링으로 과점 검출 중...")

    # 사과 영역만 추출
    apple_pixels = masked_rgb[apple_mask].reshape(-1, 3).astype(np.float32)

    # KMeans 클러스터링
    kmeans = KMeans(n_clusters=K_lenticel, random_state=42, n_init=10)
    kmeans.fit(apple_pixels)

    # 전체 이미지에 대한 라벨 생성
    labels_full = np.zeros((h, w), dtype=np.int32) - 1
    labels_full[apple_mask] = kmeans.predict(apple_pixels)

    # 클러스터 중심을 HSV로 변환
    centers_rgb = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)
    centers_bgr = centers_rgb[:, ::-1].reshape((-1, 1, 3))
    centers_hsv = cv2.cvtColor(centers_bgr, cv2.COLOR_BGR2HSV).reshape((-1, 3))

    H_centers = centers_hsv[:, 0]
    S_centers = centers_hsv[:, 1]
    V_centers = centers_hsv[:, 2]

    print(f"   클러스터별 HSV 값:")
    for i in range(K_lenticel):
        print(f"     클러스터 {i}: H={H_centers[i]:3.0f}, S={S_centers[i]:3.0f}, V={V_centers[i]:3.0f}")

    # 과점 후보 클러스터 찾기
    def in_ranges(v, ranges):
        v = int(v)
        return any(lo <= v <= hi for (lo, hi) in ranges)

    lenticel_candidates = [
        i for i in range(K_lenticel)
        if in_ranges(H_centers[i], lenticel_h_ranges)
           and V_centers[i] >= lenticel_v_min
           and S_centers[i] <= lenticel_s_max
    ]

    print(f"   과점 후보 클러스터: {lenticel_candidates}")

    # 과점 마스크 생성 (KMeans 기반)
    mask_lenticel_kmeans = np.zeros((h, w), dtype=np.uint8)
    for idx in lenticel_candidates:
        cluster_mask = (labels_full == idx).astype(np.uint8) * 255
        mask_lenticel_kmeans |= cluster_mask

    # 5. 과점 검출 - 방법 2: 로컬 밝기 차이 (개선된 방법)
    print("\n[5단계] 로컬 밝기 차이로 과점 검출 중...")
    print(f"   파라미터: local_diff_thresh={local_diff_thresh} (민감도 향상)")

    # RGB를 BGR로 변환 후 HSV 변환
    rgb_bgr = masked_rgb[:, :, ::-1]
    hsv_full = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    V = hsv_full[:, :, 2].astype(np.float32)

    # ⭐⭐ 핵심 개선: Normalized Convolution으로 배경 제외한 로컬 평균 계산
    print(f"   ✓ Normalized Convolution 방식 사용 (배경 완전 제외)")

    # 사과 영역을 float 마스크로 변환
    mask_float = apple_mask.astype(np.float32)

    # 1. 값의 가중 합 계산 (배경은 자동으로 0이므로 제외됨)
    V_weighted = V * mask_float
    sum_blurred = cv2.GaussianBlur(V_weighted, (local_ksize, local_ksize), 0)

    # 2. 가중치 합 계산 (각 픽셀 주변의 유효 이웃 개수)
    weight_blurred = cv2.GaussianBlur(mask_float, (local_ksize, local_ksize), 0)

    # 3. 정규화된 로컬 평균 (0으로 나누기 방지)
    local_mean = np.zeros_like(V)
    valid_mask = weight_blurred > 1e-5  # 유효한 이웃이 있는 영역만
    local_mean[valid_mask] = sum_blurred[valid_mask] / weight_blurred[valid_mask]

    # 사과 영역에서만 차이 계산
    diff_local = np.zeros_like(V)
    diff_local[apple_mask] = V[apple_mask] - local_mean[apple_mask]

    # 주변보다 밝은 작은 영역 검출
    local_bright = (diff_local > local_diff_thresh).astype(np.uint8) * 255
    local_bright = cv2.bitwise_and(local_bright, apple_mask_uint8)

    # 형태학적 연산으로 얇은 라인 구조 제거
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_open_kernel)
    local_bright = cv2.morphologyEx(local_bright, cv2.MORPH_OPEN, kernel_open, iterations=1)

    print(f"   ✓ 로컬 밝기 차이 검출 완료 (배경 영향 0%)")
    print(f"   ✓ 형태학적 연산으로 얇은 구조 제거 (커널: {morph_open_kernel})")

    # 6. 정제 - 더 세밀한 로컬 밝기 차이
    print("\n[6단계] 과점 마스크 정제 중...")

    # Normalized Convolution으로 정제 단계도 배경 제외
    sum_blurred_refine = cv2.GaussianBlur(V_weighted, (refine_ksize, refine_ksize), 0)
    weight_blurred_refine = cv2.GaussianBlur(mask_float, (refine_ksize, refine_ksize), 0)

    local_mean_refine = np.zeros_like(V)
    valid_mask_refine = weight_blurred_refine > 1e-5
    local_mean_refine[valid_mask_refine] = sum_blurred_refine[valid_mask_refine] / weight_blurred_refine[
        valid_mask_refine]

    diff_refine = np.zeros_like(V)
    diff_refine[apple_mask] = V[apple_mask] - local_mean_refine[apple_mask]

    local_bright_refine = (diff_refine > refine_diff_thresh).astype(np.uint8) * 255
    local_bright_refine = cv2.bitwise_and(local_bright_refine, apple_mask_uint8)

    # KMeans 결과를 로컬 밝기 차이로 정제
    mask_lenticel_refined = cv2.bitwise_and(mask_lenticel_kmeans, local_bright_refine)

    # 7. 최종 마스크: 두 방법 결합
    print("\n[7단계] 최종 마스크 생성 중...")

    final_lenticel_mask = cv2.bitwise_or(local_bright, mask_lenticel_refined)

    # 작은 노이즈 제거
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_close_kernel)
    final_lenticel_mask = cv2.morphologyEx(
        final_lenticel_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1
    )

    # 사과 영역 내에서만 최종 마스크 생성
    final_lenticel_mask = cv2.bitwise_and(final_lenticel_mask, apple_mask_uint8)

    # 8. 통계 계산
    print("\n[8단계] 통계 계산 중...")

    total_apple_pixels = np.count_nonzero(apple_mask)
    lenticel_pixels = np.count_nonzero(final_lenticel_mask)
    lenticel_ratio = lenticel_pixels / total_apple_pixels if total_apple_pixels > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"결과 통계")
    print(f"{'=' * 60}")
    print(f"  사과 영역 픽셀 수    : {total_apple_pixels:,}")
    print(f"  과점 픽셀 수         : {lenticel_pixels:,}")
    print(f"  과점 비율            : {lenticel_ratio:.4f} ({lenticel_ratio * 100:.2f}%)")
    print(f"{'=' * 60}\n")

    # 9. 결과 저장
    print("[9단계] 결과 저장 중...")

    # 9-1. 마스킹된 사과 이미지
    cv2.imwrite(f"{output_dir}/apple_masked.jpg", masked_image)
    print(f"   ✓ 저장: apple_masked.jpg")

    # 9-2. 과점 마스크 (흑백)
    cv2.imwrite(f"{output_dir}/apple_lenticels_mask.jpg", final_lenticel_mask)
    print(f"   ✓ 저장: apple_lenticels_mask.jpg")

    # 9-3. 시각화 이미지 (과점을 초록색으로 표시)
    visualization = masked_image.copy()
    visualization[final_lenticel_mask > 0] = [0, 255, 0]

    cv2.imwrite(f"{output_dir}/apple_lenticels_visualization.jpg", visualization)
    print(f"   ✓ 저장: apple_lenticels_visualization.jpg")

    # 9-4. 중간 단계 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'과점 검출 결과 (비율: {lenticel_ratio:.4f})', fontsize=16, fontweight='bold')

    axes[0, 0].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. 원본 (마스킹)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask_lenticel_kmeans, cmap='gray')
    axes[0, 1].set_title("2. KMeans 클러스터링")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(local_bright, cmap='gray')
    axes[0, 2].set_title(f"3. 로컬 밝기 (thresh={local_diff_thresh})")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(mask_lenticel_refined, cmap='gray')
    axes[1, 0].set_title("4. KMeans + 정제")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(final_lenticel_mask, cmap='gray')
    axes[1, 1].set_title("5. 최종 마스크")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("6. 시각화")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/apple_lenticels_process.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ 저장: apple_lenticels_process.jpg")

    print(f"\n✅ 모든 처리 완료! 결과는 {output_dir}에 저장되었습니다.\n")

    return {
        'masked_image': masked_image,
        'final_mask': final_lenticel_mask,
        'kmeans_mask': mask_lenticel_kmeans,
        'local_bright_mask': local_bright,
        'ratio': lenticel_ratio,
        'total_pixels': total_apple_pixels,
        'lenticel_pixels': lenticel_pixels
    }


if __name__ == "__main__":
    # 경로 설정
    IMG_PATH = r"C:\code\apple\images\1.jpg"
    MODEL_PATH = r"C:\code\apple\models\apple"
    OUTPUT_DIR = r"C:\code\apple"

    # ⭐ Normalized Convolution 방식:
    #
    # 문제 해결 원리:
    # 1. 기존: GaussianBlur(V_masked) → 배경(0값)이 평균에 포함됨
    # 2. 개선: sum_blurred / weight_blurred → 배경 완전 제외!
    #
    # 수식:
    #   local_mean[x,y] = Σ(V[i,j] * mask[i,j]) / Σ(mask[i,j])
    #   단, (i,j)는 (x,y) 주변 이웃
    #
    # 장점:
    # - 테두리 물리적 제거 불필요 (전체 영역 사용 가능)
    # - 윤곽선에서 배경 영향 0% → 오검출 원천 차단
    # - 민감도 향상으로 실제 과점 검출률 UP

    result = detect_apple_with_lenticels(
        img_path=IMG_PATH,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        K_lenticel=4,
        lenticel_h_ranges=((0, 180),),
        lenticel_v_min=180,
        lenticel_s_max=100,
        local_ksize=51,
        local_diff_thresh=12,
        refine_ksize=61,
        refine_diff_thresh=6,
        morph_open_kernel=(5, 5),
        morph_close_kernel=(3, 3),
    )

    if result:
        print(f"\n⭐ Normalized Convolution 방식 적용:")
        print(f"  - 배경 영향: 0% (완전 제외)")
        print(f"  - local_diff_thresh: 10 (민감도 향상)")
        print(f"  - refine_diff_thresh: 6 (정밀도 향상)")
        print(f"\n과점 비율: {result['ratio']:.4f}")