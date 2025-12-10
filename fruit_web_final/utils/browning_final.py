import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans


def combined_browning_mask(
    image_path: str,
    fixed_size=(800, 800),
    K_brown=5,
    v_thresh_brown=210,
    brown_h_ranges=((0, 50),),
    morph_kernel=(7, 7),
    change_ksize_local=91,
    change_diff_thresh=10,
    refine_ksize_local=101,
    refine_diff_thresh=5,
):
    img_ori = Image.open(image_path)
    img_ori.thumbnail(fixed_size, Image.LANCZOS)

    img_rgba = Image.open(image_path).convert("RGBA")
    img_rgba.thumbnail(fixed_size, Image.LANCZOS)

    img = Image.open(image_path).convert("RGB")
    img.thumbnail(fixed_size, Image.LANCZOS)

    rgb = np.asarray(img, dtype=np.uint8)
    H, W = rgb.shape[:2]
    Z = rgb.reshape(-1, 3).astype(np.float32)

    def to_centers_hsv(centers_rgb_uint8):
        bgr = centers_rgb_uint8[:, ::-1].reshape((-1, 1, 3))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape((-1, 3))
        return hsv

    def in_ranges(v, ranges):
        v = int(v)
        return any(lo <= v <= hi for (lo, hi) in ranges)

    def mask_from_alpha(img_rgba, shrink_px=0, top_extra_shrink=0, bottom_extra_shrink=0):
        rgba = np.array(img_rgba)
        alpha = rgba[:, :, 3]

        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        if shrink_px > 0:
            kernel_shrink = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (shrink_px, shrink_px)
            )
            mask = cv2.erode(mask, kernel_shrink, iterations=1)

        if top_extra_shrink > 0:
            h, w = mask.shape
            top_h = int(h * 0.4)
            top_region = mask[:top_h, :]
            kernel_top = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (top_extra_shrink, top_extra_shrink)
            )
            mask[:top_h, :] = cv2.erode(top_region, kernel_top, iterations=1)

        if bottom_extra_shrink > 0:
            h, w = mask.shape
            bottom_h = int(h * 0.4)
            bottom_region = mask[h - bottom_h :, :]
            kernel_bottom = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (bottom_extra_shrink, bottom_extra_shrink)
            )
            mask[h - bottom_h :, :] = cv2.erode(bottom_region, kernel_bottom, iterations=1)

        return mask

    mask_alpha = mask_from_alpha(img_rgba)

    rgb_bgr = rgb[:, :, ::-1]
    hsv_full = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    V = hsv_full[:, :, 2].astype(np.float32)

    local_mean_change = cv2.GaussianBlur(
        V, (change_ksize_local, change_ksize_local), 0
    )
    diff_change = local_mean_change - V
    local_dark_change = (diff_change > change_diff_thresh).astype(np.uint8) * 255
    local_dark_change = cv2.bitwise_and(local_dark_change, mask_alpha)
    local_dark_change = cv2.morphologyEx(
        local_dark_change,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        1,
    )

    mask_melon = mask_from_alpha(
        img_rgba,
        shrink_px=7,
        top_extra_shrink=0,
        bottom_extra_shrink=0,
    )

    final_mask_change = cv2.bitwise_and(local_dark_change, mask_melon)
    final_mask_change = cv2.bitwise_and(final_mask_change, mask_alpha)

    km_b = KMeans(n_clusters=K_brown, random_state=42, n_init=10)
    labels_b = km_b.fit_predict(Z)
    centers_b = km_b.cluster_centers_.astype(np.float32)
    centers_b_u8 = np.clip(centers_b, 0, 255).astype(np.uint8)
    hsv_b = to_centers_hsv(centers_b_u8)
    H_bv, S_bv, V_bv = hsv_b[:, 0], hsv_b[:, 1], hsv_b[:, 2]

    cand = [
        i
        for i in range(K_brown)
        if in_ranges(H_bv[i], brown_h_ranges) and V_bv[i] < v_thresh_brown
    ]
    cand_sorted = sorted(cand, key=lambda i: V_bv[i])

    if len(cand_sorted) >= 3 and V_bv[cand_sorted[0]] < 60:
        browning_idxs = cand_sorted[:4]
    else:
        browning_idxs = cand_sorted[:3]

    mask_browning = np.zeros((H, W), dtype=np.uint8)
    for idx in browning_idxs:
        mask_browning |= (labels_b.reshape(H, W) == idx).astype(np.uint8) * 255

    local_mean_refine = cv2.GaussianBlur(
        V, (refine_ksize_local, refine_ksize_local), 0
    )
    diff_refine = local_mean_refine - V
    local_dark_refine = (diff_refine > refine_diff_thresh).astype(np.uint8) * 255
    local_dark_refine = cv2.bitwise_and(local_dark_refine, mask_alpha)

    mask_browning_refined = cv2.bitwise_and(mask_browning, local_dark_refine)

    browning_mask_kmeans = cv2.bitwise_and(mask_browning, mask_melon)
    final_mask_kmeans = cv2.bitwise_and(mask_browning_refined, mask_melon)
    final_mask_kmeans = cv2.bitwise_and(final_mask_kmeans, mask_alpha)

    final_mask = cv2.bitwise_or(final_mask_change, final_mask_kmeans)

    return rgb, final_mask, mask_alpha