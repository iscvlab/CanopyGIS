"""Canopy V2 native-resolution tiling and RGB/SAM reconciliation."""

import contextlib
import inspect
import sys
from typing import Callable, Optional

import cv2
import numpy as np
from scipy.ndimage import find_objects
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _compute_scene_haze_offset(image_rgb: np.ndarray) -> tuple[float, float]:
    """Return (scene_gray_p1, haze_offset) for wide-dynamic-range hazy satellite scenes."""
    if image_rgb.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    p1 = float(np.percentile(gray, 1))
    p99 = float(np.percentile(gray, 99))
    if p1 > 32.0 and p99 > 120.0:
        return p1, max(0.0, p1 - 10.0)
    return p1, 0.0


def enhance_shadow_luminance_cielab(
    image_rgb: np.ndarray,
    scene_gray_p1: Optional[float] = None,
) -> np.ndarray:
    """
    Stream A (AI Detector Radiometric Stream):
    - On low-haze imagery (scene_gray_p1 <= 32.0): Enhance local luminance contrast and lift
      deep shadow details exclusively on the CIELAB L-channel while keeping chromaticity (a, b)
      100% unchanged (zero Hue or Saturation shift).
    - On hazy satellite imagery (scene_gray_p1 > 32.0): Perform physical path-radiance subtraction
      (neutralizing Rayleigh blue scatter) and atmospheric transmission contrast/chroma restoration
      so dark-green crowns are not washed into the hazy background.
    """
    if image_rgb.size == 0:
        return image_rgb
    haze_offset = (
        max(0.0, float(scene_gray_p1) - 10.0)
        if (scene_gray_p1 is not None and float(scene_gray_p1) > 32.0)
        else 0.0
    )
    if haze_offset > 0.0:
        h0 = 0.85 * haze_offset
        rgb_f = image_rgb.astype(np.float32)
        r_c = rgb_f[:, :, 0] - h0 * 0.92
        g_c = rgb_f[:, :, 1] - h0 * 0.95
        b_c = rgb_f[:, :, 2] - h0 * 1.08
        scale = 250.0 / max(120.0, 255.0 - h0)
        dehazed = np.clip(
            np.dstack([r_c, g_c, b_c]) * scale, 0.0, 255.0
        ).astype(np.uint8)
        lab_h = cv2.cvtColor(dehazed, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab_h)
        clahe_h = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_out_h = clahe_h.apply(l_ch)
        a_f = 128.0 + (a_ch.astype(np.float32) - 128.0) * 1.30
        b_f = 128.0 + (b_ch.astype(np.float32) - 128.0) * 1.25
        return cv2.cvtColor(
            cv2.merge(
                [
                    l_out_h,
                    np.clip(a_f, 0.0, 255.0).astype(np.uint8),
                    np.clip(b_f, 0.0, 255.0).astype(np.uint8),
                ]
            ),
            cv2.COLOR_LAB2RGB,
        )

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_f = l_chan.astype(np.float32)

    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_chan).astype(np.float32)
    l_gamma = np.power(np.clip(l_f / 255.0, 0.0, 1.0), 0.78) * 255.0
    l_enh = 0.55 * l_gamma + 0.45 * l_clahe

    w_shadow = np.clip((118.0 - l_f) / 92.0, 0.0, 0.70)
    l_out = np.clip(
        l_f * (1.0 - w_shadow) + l_enh * w_shadow, 0.0, 255.0
    ).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_out, a_chan, b_chan]), cv2.COLOR_LAB2RGB)


def compute_strict_foliage_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Stream B (Physical Anti-Fake Filter Stream):
    Compute strict pixel-level foliage mask on Raw RGB rejecting bare soil, roofs,
    water, marsh mud, and cars, with physical Rayleigh path-radiance compensation
    when scene_gray_p1 > 32.0.
    """
    if image_rgb.size == 0:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    _, haze_offset = _compute_scene_haze_offset(image_rgb)

    rgb_f = image_rgb.astype(np.float32)
    r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
    exg = 2.0 * g - r - b
    gli = exg / (2.0 * g + r + b + 1e-5)
    vari = (g - r) / (g + r - b + 1e-5)
    bright = (r + g + b) / 3.0
    foliage_cond = (
        ((g >= r - 1.0) | ((vari >= -0.008) & (exg >= 11.0)))
        & (
            (g >= b + 3.0)
            | ((g >= b + 1.5) & (vari >= 0.14) & (g >= r + 4.5))
        )
        & (gli >= 0.038)
        & ((exg >= 6.5) | ((vari >= 0.15) & (gli >= 0.055)))
        & (bright >= 13.0)
        & (bright <= 190.0)
    )
    if haze_offset > 0.0:
        bg_tol = min(3.8, 0.08 * haze_offset)
        gli_min_haze = max(0.024, 0.038 - 0.00030 * haze_offset)
        haze_foliage = (
            (g >= b - bg_tol)
            & (g >= r + 13.0)
            & (exg >= 11.5)
            & (gli >= gli_min_haze)
            & (bright >= 13.0)
            & (bright <= 190.0)
        )
        gray_fol = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap_fol = cv2.blur(
            np.abs(cv2.Laplacian(gray_fol, cv2.CV_32F)), (5, 5)
        )
        mean_g = cv2.blur(gray_fol, (7, 7))
        sq_g = cv2.blur(gray_fol * gray_fol, (7, 7))
        std_fol = np.sqrt(np.maximum(0.0, sq_g - mean_g * mean_g))
        rayleigh_olive_canopy = (
            (g >= b - min(8.5, 0.185 * haze_offset))
            & (
                (
                    (g >= r + 11.5)
                    & (r <= 56.0 + 1.05 * haze_offset)
                    & (exg >= 5.5)
                    & (lap_fol >= 6.5)
                    & (std_fol >= 5.8)
                )
                | (
                    (g >= r + 20.0)
                    & (r <= 48.0 + 1.05 * haze_offset)
                    & (exg >= 14.0)
                    & (lap_fol >= 4.2)
                    & (std_fol >= 3.5)
                )
            )
            & (bright >= 13.0)
            & (bright <= 185.0)
        )
        dusty_textured_canopy = (
            (g >= b - min(5.5, 0.12 * haze_offset))
            & (g >= r + 4.5)
            & (r <= 65.0 + 1.05 * haze_offset)
            & (exg >= 1.5)
            & (lap_fol >= 9.5)
            & (std_fol >= 8.8)
            & (bright >= 18.0)
            & (bright <= 175.0)
        )
        foliage_cond = (
            foliage_cond | haze_foliage | rayleigh_olive_canopy | dusty_textured_canopy
        )

    foliage = foliage_cond.astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foliage = cv2.morphologyEx(foliage, cv2.MORPH_OPEN, k3, iterations=1)
    foliage = cv2.morphologyEx(foliage, cv2.MORPH_CLOSE, k3, iterations=1)

    # Tight (~0.7m, 5x5) self-shaded chlorophyll rim directly bordering sunlit crown foliage
    k_shade = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    near_sunlit = cv2.dilate(foliage, k_shade, iterations=1)
    r_shade_max = 42.0 + 1.05 * haze_offset
    gb_shade_tol = 1.5 + min(4.5, 0.10 * haze_offset)
    shaded_rim = (
        (near_sunlit > 0)
        & (bright >= 8.5)
        & (r <= r_shade_max)
        & (g >= 13.0)
        & (g >= r + 4.0)
        & (g >= b - gb_shade_tol)
        & (exg >= (4.5 if haze_offset > 0.0 else 8.0))
        & (vari >= (0.08 if haze_offset == 0.0 else 0.01))
    ).astype(np.uint8) * 255
    foliage = cv2.bitwise_or(foliage, shaded_rim)
    foliage = cv2.morphologyEx(foliage, cv2.MORPH_CLOSE, k3, iterations=1)
    return foliage


def compute_woody_canopy_relief_mask(
    image_rgb: np.ndarray,
    strict_foliage: Optional[np.ndarray] = None,
    gsd_m: float = 0.10,
) -> np.ndarray:
    """
    Compute a pixel-level 3D woody canopy relief mask that rejects flat green
    duckweed/algae ponds, smooth lawns, and uniform low scrub lacking 3D crown
    luminance variance, Laplacian texture energy, and inter-crown shadow relief.
    """
    if image_rgb.size == 0:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    if strict_foliage is None:
        strict_foliage = compute_strict_foliage_mask(image_rgb)
    _, haze_offset = _compute_scene_haze_offset(image_rgb)

    lab_l = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(
        np.float32
    )
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    k = max(7, int(round(2.2 / max(gsd_m, 0.03))) | 1)
    k_shadow = max(11, int(round(3.2 / max(gsd_m, 0.03))) | 1)

    mean_l = cv2.blur(lab_l, (k, k))
    sq_l = cv2.blur(lab_l * lab_l, (k, k))
    local_std = np.sqrt(np.maximum(0.0, sq_l - mean_l * mean_l))

    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    local_lap = cv2.blur(lap, (k, k))

    se_shadow = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k_shadow, k_shadow)
    )
    local_max_l = cv2.dilate(lab_l, se_shadow)
    local_min_l = cv2.erode(lab_l, se_shadow)
    local_dyn_l = local_max_l - local_min_l

    if haze_offset > 0.0:
        haze_gain = 1.0 + min(0.55, 0.012 * haze_offset)
        local_std = local_std * haze_gain
        local_lap = local_lap * haze_gain
        local_dyn_l = local_dyn_l * haze_gain

    relief = (
        (strict_foliage > 0)
        & (local_std >= 9.5)
        & (local_lap >= 7.5)
        & (local_dyn_l >= 28.0)
    ).astype(np.uint8) * 255

    if haze_offset > 0.0:
        rgb_f = image_rgb.astype(np.float32)
        r_f, g_f, b_f = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
        deep_chlorophyll_relief = (
            (strict_foliage > 0)
            & ((g_f - r_f) >= 18.0)
            & ((2.0 * g_f - r_f - b_f) >= 12.0)
            & (local_std >= 5.8)
            & (local_lap >= 5.0)
            & (local_dyn_l >= 18.0)
        ).astype(np.uint8) * 255
        relief = cv2.bitwise_or(relief, deep_chlorophyll_relief)
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        relief = cv2.morphologyEx(relief, cv2.MORPH_CLOSE, k7, iterations=1)
    elif gsd_m >= 0.22:
        rgb_f = image_rgb.astype(np.float32)
        r_f, g_f, b_f = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
        deep_chlorophyll_relief = (
            (strict_foliage > 0)
            & (r_f <= 68.0)
            & ((g_f - r_f) >= 24.0)
            & ((b_f - r_f) >= 8.0)
            & (local_std >= 5.0)
            & (local_lap >= 5.0)
            & (local_dyn_l >= 18.0)
        ).astype(np.uint8) * 255
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        deep_chlorophyll_relief = cv2.morphologyEx(
            deep_chlorophyll_relief, cv2.MORPH_CLOSE, k5, iterations=1
        )
        relief = cv2.bitwise_or(relief, deep_chlorophyll_relief)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    relief = cv2.morphologyEx(relief, cv2.MORPH_OPEN, k3, iterations=1)
    relief = cv2.morphologyEx(relief, cv2.MORPH_CLOSE, k3, iterations=1)
    relief = cv2.bitwise_and(relief, strict_foliage)
    return relief


def compute_crown_confidence(
    crop_rgb: np.ndarray,
    cnt_mask: np.ndarray,
    circularity: float,
    detector_score: Optional[float] = 0.5,
    geometry_quality: Optional[str] = None,
    haze_offset: float = 0.0,
) -> tuple[float, str, str]:
    """
    Compute continuous anti-cheat calibrated confidence_pct (18..98),
    confidence_tier, and confidence_label from 5 physical + AI indicators.
    """
    mask_bool = cnt_mask > 0
    if not np.any(mask_bool) or crop_rgb.size == 0:
        return (
            50.0,
            "moderate_watershed",
            "Moderate Confidence — Closed-Canopy Watershed Split",
        )

    haze_gain = (
        1.0 + min(0.55, 0.011 * float(haze_offset))
        if haze_offset > 0.0
        else 1.0
    )
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ring_mask = cv2.subtract(
        cv2.dilate(cnt_mask, k3, iterations=1),
        cv2.erode(cnt_mask, k3, iterations=1),
    )
    gray_f = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    grad_mag = cv2.magnitude(gx, gy)
    ring_bool = ring_mask > 0
    if np.any(ring_bool):
        perimeter_grad = float(grad_mag[ring_bool].mean()) * haze_gain
    else:
        perimeter_grad = float(grad_mag[mask_bool].mean()) * haze_gain
    edge_contrast = float(np.clip(perimeter_grad / 20.0, 0.0, 1.0))

    gray_px = gray_f[mask_bool]
    std_score = float(
        np.clip((float(gray_px.std()) * haze_gain - 7.0) / 18.0, 0.0, 1.0)
    )
    p90_p10 = float(np.percentile(gray_px, 90) - np.percentile(gray_px, 10))
    dr_score = float(
        np.clip((p90_p10 * haze_gain - 18.0) / 36.0, 0.0, 1.0)
    )

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner_mask = cv2.erode(cnt_mask, k5, iterations=1)
    if not np.any(inner_mask > 0):
        inner_mask = cv2.erode(cnt_mask, k3, iterations=1)
    outer_rim = cv2.subtract(cnt_mask, inner_mask)
    if np.any(inner_mask > 0) and np.any(outer_rim > 0):
        inner_l = float(gray_f[inner_mask > 0].mean())
        rim_l = float(gray_f[outer_rim > 0].mean())
        radial_score = float(
            np.clip(((inner_l - rim_l) * haze_gain + 2.0) / 14.0, 0.0, 1.0)
        )
    else:
        radial_score = 0.5
    dome_relief = float(
        0.45 * std_score + 0.35 * dr_score + 0.20 * radial_score
    )

    px = crop_rgb[mask_bool].astype(np.float32)
    r_p, g_p, b_p = px[:, 0], px[:, 1], px[:, 2]
    exg_p = 2.0 * g_p - r_p - b_p
    pure_frac = float(np.mean((g_p > r_p) & (exg_p >= 6.5)))
    r_m, g_m, b_m = float(r_p.mean()), float(g_p.mean()), float(b_p.mean())
    gli_m = (2.0 * g_m - r_m - b_m) / (2.0 * g_m + r_m + b_m + 1e-5)
    gli_norm = min(1.0, max(0.0, (gli_m * haze_gain) / 0.12))
    foliage_purity = float(
        np.clip(0.55 * pure_frac + 0.45 * gli_norm, 0.0, 1.0)
    )

    cnts_m, _ = cv2.findContours(
        cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if cnts_m:
        c_main = max(cnts_m, key=cv2.contourArea)
        cnt_area = float(cv2.contourArea(c_main))
        hull_area = float(cv2.contourArea(cv2.convexHull(c_main)))
        solidity = (
            float(np.clip(cnt_area / max(1.0, hull_area), 0.0, 1.0))
            if hull_area > 0.0
            else 0.75
        )
    else:
        solidity = 0.75
    shape_score = float(
        0.60 * np.clip(float(circularity) / 0.62, 0.0, 1.0)
        + 0.40 * solidity
    )

    det_s = float(
        np.clip(
            detector_score if detector_score is not None else 0.5, 0.0, 1.0
        )
    )
    gq = (geometry_quality or "").strip().lower()

    if gq == "sam3_native" or "sam3" in gq:
        ai_prior = float(np.clip(0.80 + 0.18 * det_s, 0.80, 0.98))
    elif gq == "deepforest_watershed" or "deepforest" in gq:
        ai_prior = float(np.clip(0.58 + 0.22 * det_s, 0.58, 0.78))
    else:
        ai_prior = float(np.clip(0.36 + 0.24 * det_s, 0.36, 0.58))

    if (
        haze_offset > 0.0
        and gq in ("watershed_forest_crown", "deepforest_watershed")
        and foliage_purity >= 0.42
    ):
        dome_relief = max(dome_relief, 0.38)

    physical_quality = (
        0.35 * dome_relief
        + 0.25 * edge_contrast
        + 0.20 * shape_score
        + 0.20 * foliage_purity
    )
    raw_conf = 100.0 * (0.64 * ai_prior + 0.36 * physical_quality)
    if dome_relief < 0.26:
        raw_conf *= 0.44 + 1.5 * dome_relief
    elif dome_relief < 0.38:
        raw_conf *= 0.78 + 0.55 * dome_relief

    conf_pct = float(np.clip(raw_conf, 18.0, 98.0))
    if conf_pct >= 80.0:
        confidence_tier = "high_individual"
        confidence_label = "High Confidence — Distinct Crown (SAM 3)"
    elif conf_pct >= 60.0:
        confidence_tier = "good_treetop"
        confidence_label = "Good Confidence — Verified Treetop + Crown Relief"
    elif conf_pct >= 45.0:
        confidence_tier = "moderate_watershed"
        confidence_label = (
            "Moderate Confidence — Closed-Canopy Watershed Split"
        )
    else:
        confidence_tier = "low_ambiguous"
        confidence_label = (
            "Low Confidence — Ambiguous Dense / Scrub Partition"
        )

    return conf_pct, confidence_tier, confidence_label


def compute_polygon_physical_scores(
    crop_rgb: np.ndarray,
    cnt_mask: np.ndarray,
    haze_offset: float = 0.0,
) -> dict[str, float]:
    """
    Compute scene-normalized [0.0..1.0] physical indicators for <30ms client-side
    post-processing filters:
    - foliage_prob: Chlorophyll / GLI / VARI vegetation vigor [0.0..1.0]
    - relief_3d_score: 3D woody canopy texture & internal shadow relief vs. flat turf [0.0..1.0]
    - artificial_cyan_score: Synthetic cyan-green paint/roof/truck signature [0.0..1.0]
    """
    mask_bool = cnt_mask > 0
    if not np.any(mask_bool) or crop_rgb.size == 0:
        return {
            "foliage_prob": 0.5,
            "relief_3d_score": 0.5,
            "artificial_cyan_score": 0.0,
        }

    haze_gain = (
        1.0 + min(0.55, 0.011 * float(haze_offset))
        if haze_offset > 0.0
        else 1.0
    )
    gray_f = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_px = gray_f[mask_bool]
    px = crop_rgb[mask_bool].astype(np.float32)
    r_p, g_p, b_p = px[:, 0], px[:, 1], px[:, 2]
    r_m, g_m, b_m = float(r_p.mean()), float(g_p.mean()), float(b_p.mean())
    exg_p = 2.0 * g_p - r_p - b_p
    pure_frac = float(np.mean((g_p > r_p) & (exg_p >= 6.5)))
    gli_m = (2.0 * g_m - r_m - b_m) / (2.0 * g_m + r_m + b_m + 1e-5)
    vari_m = (g_m - r_m) / (g_m + r_m - b_m + 1e-5)
    gli_norm = float(np.clip((gli_m * haze_gain) / 0.12, 0.0, 1.0))
    vari_norm = float(np.clip((vari_m + 0.05) / 0.25, 0.0, 1.0))
    foliage_prob = round(
        float(
            np.clip(
                0.45 * pure_frac + 0.35 * gli_norm + 0.20 * vari_norm,
                0.0,
                1.0,
            )
        ),
        3,
    )

    std_score = float(
        np.clip((float(gray_px.std()) * haze_gain - 6.0) / 18.0, 0.0, 1.0)
    )
    p90_p10 = float(np.percentile(gray_px, 90) - np.percentile(gray_px, 10))
    dr_score = float(
        np.clip((p90_p10 * haze_gain - 16.0) / 38.0, 0.0, 1.0)
    )
    shadow_thresh = 46.0 + 1.15 * float(haze_offset)
    shadow_frac = float(np.mean(gray_px < shadow_thresh))
    shadow_score = float(np.clip(shadow_frac / 0.22, 0.0, 1.0))
    r_corr = max(0.0, r_m - 0.75 * float(haze_offset))
    g_corr = max(0.0, g_m - 0.80 * float(haze_offset))
    b_corr = max(0.0, b_m - 0.88 * float(haze_offset))
    chl_depth = (
        float(np.clip(((g_corr - r_corr) - 10.0) / 22.0, 0.0, 1.0))
        if r_corr <= 72.0
        else 0.0
    )
    relief_3d_score = round(
        float(
            np.clip(
                0.35 * std_score
                + 0.30 * dr_score
                + 0.20 * shadow_score
                + 0.15 * chl_depth,
                0.0,
                1.0,
            )
        ),
        3,
    )

    br_excess = (b_corr - r_corr) - 8.0
    bg_ratio = (b_corr - (g_corr - 12.0)) / 18.0
    cyan_chroma = float(
        np.clip(
            0.60 * max(0.0, br_excess / 18.0) + 0.40 * max(0.0, bg_ratio),
            0.0,
            1.0,
        )
    )
    flat_bright_factor = float(
        np.clip(
            0.50 * (1.0 - shadow_score)
            + 0.30 * (1.0 - std_score)
            + 0.20 * np.clip((r_corr - 42.0) / 45.0, 0.0, 1.0),
            0.15,
            1.0,
        )
    )
    artificial_cyan_score = round(
        float(np.clip(cyan_chroma * flat_bright_factor, 0.0, 1.0)), 3
    )

    return {
        "foliage_prob": foliage_prob,
        "relief_3d_score": relief_3d_score,
        "artificial_cyan_score": artificial_cyan_score,
    }


def derive_feature_physical_scores(
    props: dict, haze_offset: float = 0.0
) -> dict[str, float]:
    """
    Deterministic fallback physical score calculator from GeoJSON feature properties
    when raw raster pixels are not loaded in memory.
    """
    if (
        isinstance(props.get("relief_3d_score"), (int, float))
        and isinstance(props.get("artificial_cyan_score"), (int, float))
        and isinstance(props.get("foliage_prob"), (int, float))
    ):
        return {
            "foliage_prob": round(float(props["foliage_prob"]), 3),
            "relief_3d_score": round(float(props["relief_3d_score"]), 3),
            "artificial_cyan_score": round(
                float(props["artificial_cyan_score"]), 3
            ),
        }

    mean_rgb = props.get("mean_rgb")
    if isinstance(mean_rgb, (list, tuple)) and len(mean_rgb) >= 3:
        r_m, g_m, b_m = (
            float(mean_rgb[0]),
            float(mean_rgb[1]),
            float(mean_rgb[2]),
        )
    else:
        r_m, g_m, b_m = 52.0, 82.0, 62.0

    gli_m = float(
        props.get(
            "foliar_gli",
            (2.0 * g_m - r_m - b_m) / (2.0 * g_m + r_m + b_m + 1e-5),
        )
    )
    vari_m = float(
        props.get("raw_vari", (g_m - r_m) / (g_m + r_m - b_m + 1e-5))
    )
    conf_pct = float(props.get("confidence_pct", 70.0))
    r_corr = max(0.0, r_m - 0.75 * float(haze_offset))
    g_corr = max(0.0, g_m - 0.80 * float(haze_offset))
    b_corr = max(0.0, b_m - 0.88 * float(haze_offset))

    gli_norm = float(np.clip(gli_m / 0.14, 0.0, 1.0))
    vari_norm = float(np.clip((vari_m + 0.05) / 0.25, 0.0, 1.0))
    foliage_prob = round(
        float(np.clip(0.60 * gli_norm + 0.40 * vari_norm, 0.0, 1.0)), 3
    )

    conf_norm = float(np.clip((conf_pct - 25.0) / 65.0, 0.0, 1.0))
    chl_depth = float(np.clip(((g_corr - r_corr) - 8.0) / 24.0, 0.0, 1.0))
    dark_woody = float(np.clip((92.0 - r_corr) / 55.0, 0.0, 1.0))
    relief_3d_score = round(
        float(
            np.clip(
                0.50 * conf_norm + 0.30 * chl_depth + 0.20 * dark_woody,
                0.0,
                1.0,
            )
        ),
        3,
    )

    br_excess = (b_corr - r_corr) - 8.0
    bg_ratio = (b_corr - (g_corr - 12.0)) / 18.0
    cyan_chroma = float(
        np.clip(
            0.60 * max(0.0, br_excess / 18.0) + 0.40 * max(0.0, bg_ratio),
            0.0,
            1.0,
        )
    )
    bright_factor = float(
        np.clip(
            0.55 * (1.0 - dark_woody) + 0.45 * (1.0 - 0.5 * conf_norm),
            0.15,
            1.0,
        )
    )
    artificial_cyan_score = round(
        float(np.clip(cyan_chroma * bright_factor, 0.0, 1.0)), 3
    )
    return {
        "foliage_prob": foliage_prob,
        "relief_3d_score": relief_3d_score,
        "artificial_cyan_score": artificial_cyan_score,
    }


def compute_scene_auto_tune_profile(
    image_rgb: np.ndarray,
    gsd_m: float = 0.15,
) -> dict:
    """Compute image-driven auto-tuning parameters from scene GSD, haze, contrast, and foliage density."""
    gsd_val = max(0.01, float(gsd_m or 0.15))
    if image_rgb is None or image_rgb.size == 0:
        return {
            "gsd_m": round(gsd_val, 4),
            "effective_gsd_m": round(gsd_val, 4),
            "haze_offset": 0.0,
            "scene_gray_p1": 0.0,
            "scene_gray_p99": 255.0,
            "contrast_span": 255.0,
            "foliage_pct": 0.0,
            "cyan_industrial_pct": 0.0,
            "v2_instance_threshold": 0.18,
            "v2_coverage_threshold": 0.43,
            "v2_tile_size": 1280,
            "min_area_m2": 0.5,
            "max_area_m2": 260,
            "default_height_m": 10.0,
            "atmosphere_regime": "Clear-Sky",
            "summary_text": (
                "Scene-Calibrated (Clear-Sky · GSD 0.15m/px) → "
                "conf 0.18 · RGB 0.43 · Tile 1280px · Area 0.5–260 m²"
            ),
        }

    h, w = image_rgb.shape[:2]
    max_dim = max(h, w)
    effective_gsd_m = (
        gsd_val * (float(max_dim) / 5200.0) if max_dim > 5200 else gsd_val
    )

    scene_gray_p1, haze_offset = _compute_scene_haze_offset(image_rgb)

    stride = max(1, max_dim // 512)
    sample = image_rgb[::stride, ::stride]
    gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    p1 = float(np.percentile(gray, 1))
    p99 = float(np.percentile(gray, 99))
    contrast_span = max(1.0, p99 - p1)

    r = sample[:, :, 0].astype(np.float32)
    g = sample[:, :, 1].astype(np.float32)
    b = sample[:, :, 2].astype(np.float32)
    r_c = np.clip(r - 0.75 * haze_offset, 0.0, 255.0)
    g_c = np.clip(g - 0.80 * haze_offset, 0.0, 255.0)
    b_c = np.clip(b - 0.88 * haze_offset, 0.0, 255.0)

    exg = 2.0 * g_c - r_c - b_c
    foliage_mask = (
        (g_c > r_c + 4.0)
        & (g_c >= b_c - 6.0)
        & (exg >= 11.0)
        & (gray >= 16)
        & (gray <= 218)
    )
    foliage_pct = round(100.0 * float(np.mean(foliage_mask)), 1)

    cyan_mask = (
        (b_c - r_c > 12.0)
        & (g_c - r_c > 8.0)
        & (gray >= 55)
        & (gray <= 225)
    )
    cyan_industrial_pct = round(100.0 * float(np.mean(cyan_mask)), 1)

    # 1. Tile Size (px) scaled to effective GSD so tiles span ~115-350m physical receptive field
    if effective_gsd_m <= 0.095:
        v2_tile_size = 1536
    elif effective_gsd_m <= 0.34:
        v2_tile_size = 1280
    else:
        v2_tile_size = 1024

    # 2. Min & Max Single-Crown Area (m²) scaled to physical pixel resolution
    if effective_gsd_m <= 0.12:
        min_area_m2 = 0.5
        max_area_m2 = 260
    elif effective_gsd_m < 0.22:
        min_area_m2 = 0.5
        max_area_m2 = 380 if haze_offset > 0.0 else 260
    elif effective_gsd_m <= 0.34:
        min_area_m2 = 1.2
        max_area_m2 = 380
    else:
        min_area_m2 = 2.0
        max_area_m2 = 450

    # 3. SAM 3 Instance & RGB Coverage thresholds adapted to contrast and foliage/urban balance
    # Note: Stream A & Stream B already subtract Rayleigh path-radiance when haze_offset > 0.
    if contrast_span < 140.0 and haze_offset == 0.0:
        v2_instance_threshold = 0.16
        v2_coverage_threshold = 0.39
        atmosphere_regime = f"Low-Contrast ({int(round(contrast_span))} span)"
    elif foliage_pct < 5.0 and cyan_industrial_pct > 12.0:
        v2_instance_threshold = 0.20
        v2_coverage_threshold = 0.45
        atmosphere_regime = "High-Glare Urban/Industrial"
    elif haze_offset > 0.0:
        v2_instance_threshold = 0.18
        v2_coverage_threshold = 0.43
        atmosphere_regime = f"Rayleigh Haze (+{haze_offset:.0f} Compensated)"
    else:
        v2_instance_threshold = 0.18
        v2_coverage_threshold = 0.43
        atmosphere_regime = "Clear-Sky VHR"

    summary_text = (
        f"Scene-Calibrated ({atmosphere_regime} · GSD {effective_gsd_m:.2f}m/px · "
        f"{foliage_pct:.1f}% foliage) → conf {v2_instance_threshold:.2f} · "
        f"RGB {v2_coverage_threshold:.2f} · Tile {v2_tile_size}px · "
        f"Area {min_area_m2:.1f}–{int(max_area_m2)}m²"
    )

    return {
        "gsd_m": round(gsd_val, 4),
        "effective_gsd_m": round(effective_gsd_m, 4),
        "haze_offset": round(float(haze_offset), 2),
        "scene_gray_p1": round(float(scene_gray_p1), 2),
        "scene_gray_p99": round(float(p99), 2),
        "contrast_span": round(float(contrast_span), 1),
        "foliage_pct": foliage_pct,
        "cyan_industrial_pct": cyan_industrial_pct,
        "v2_instance_threshold": round(float(v2_instance_threshold), 2),
        "v2_coverage_threshold": round(float(v2_coverage_threshold), 2),
        "v2_tile_size": int(v2_tile_size),
        "min_area_m2": round(float(min_area_m2), 1),
        "max_area_m2": int(max_area_m2),
        "default_height_m": 10.0,
        "atmosphere_regime": atmosphere_regime,
        "summary_text": summary_text,
    }


def carve_non_overlapping_crowns(
    items: list[dict],
    height: int,
    width: int,
    iom_reject: float = 0.28,
    min_area_px: int = 25,
    initial_occupied: Optional[np.ndarray] = None,
) -> tuple[list[dict], np.ndarray]:
    """Carve mutually exclusive, non-overlapping crown polygons sorted by score."""
    occupied = (
        initial_occupied.copy()
        if initial_occupied is not None
        else np.zeros((height, width), dtype=np.uint8)
    )
    if not items:
        return [], occupied

    prepared = []
    for item in items:
        contour = item.get("contour")
        if contour is None:
            continue
        raw_area = item.get("area")
        area = (
            float(raw_area)
            if raw_area is not None
            else float(cv2.contourArea(contour))
        )
        if area < min_area_px:
            continue
        score = float(item.get("score", 0.0))
        out_item = dict(item)
        out_item["area"] = area
        prepared.append((score, area, out_item))

    prepared.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sep_kernel = np.ones((3, 3), dtype=np.uint8)

    carved: list[dict] = []
    for _, _, cand in prepared:
        contour = np.asarray(cand["contour"]).copy()
        rx, ry, rw, rh = cv2.boundingRect(contour)
        bx = max(0, rx - 2)
        by = max(0, ry - 2)
        bx2 = min(width, rx + rw + 2)
        by2 = min(height, ry + rh + 2)
        bw = bx2 - bx
        bh = by2 - by
        if bw <= 0 or bh <= 0:
            continue

        cand_roi = np.zeros((bh, bw), dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, 0, 0] -= bx
        shifted[:, 0, 1] -= by
        cv2.drawContours(cand_roi, [shifted], -1, 255, -1)
        cand_roi = cv2.morphologyEx(cand_roi, cv2.MORPH_CLOSE, close_kernel)

        occ_roi = occupied[by:by2, bx:bx2]
        cand_px = cv2.countNonZero(cand_roi)
        if cand_px < min_area_px:
            continue
        inter_px = cv2.countNonZero(cv2.bitwise_and(cand_roi, occ_roi))
        if inter_px / max(1.0, float(cand_px)) >= iom_reject:
            continue

        if inter_px > 0:
            cand_roi = cv2.bitwise_and(cand_roi, cv2.bitwise_not(occ_roi))
            cand_roi = cv2.morphologyEx(cand_roi, cv2.MORPH_OPEN, open_kernel)
            if cv2.countNonZero(cand_roi) < min_area_px:
                continue

        contours, _ = cv2.findContours(
            cand_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        best_c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best_c) < min_area_px:
            continue

        approx_c = cv2.approxPolyDP(
            best_c, max(1.0, 0.0035 * cv2.arcLength(best_c, True)), True
        )
        final_c = best_c
        if len(approx_c) >= 3 and cv2.contourArea(approx_c) >= min_area_px:
            test_roi = np.zeros_like(cand_roi)
            cv2.drawContours(test_roi, [approx_c], -1, 255, -1)
            if cv2.countNonZero(cv2.bitwise_and(test_roi, occ_roi)) == 0:
                final_c = approx_c

        final_roi = np.zeros_like(cand_roi)
        cv2.drawContours(final_roi, [final_c], -1, 255, -1)
        if cv2.countNonZero(cv2.bitwise_and(final_roi, occ_roi)) > 0:
            final_roi = cv2.bitwise_and(final_roi, cv2.bitwise_not(occ_roi))
            sub_cnts, _ = cv2.findContours(
                final_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not sub_cnts:
                continue
            final_c = max(sub_cnts, key=cv2.contourArea)
            if cv2.contourArea(final_c) < min_area_px:
                continue
            final_roi.fill(0)
            cv2.drawContours(final_roi, [final_c], -1, 255, -1)
            ov_hole = cv2.bitwise_and(final_roi, occ_roi)
            if cv2.countNonZero(ov_hole) > 0:
                hx, hy, hw, hh = cv2.boundingRect(ov_hole)
                hcx = hx + hw // 2
                final_roi = cv2.bitwise_and(final_roi, cv2.bitwise_not(occ_roi))
                final_roi[:, max(0, hcx - 1) : min(bw, hcx + 2)] = 0
                sub_cnts, _ = cv2.findContours(
                    final_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not sub_cnts:
                    continue
                final_c = max(sub_cnts, key=cv2.contourArea)
                if cv2.contourArea(final_c) < min_area_px:
                    continue
                final_roi.fill(0)
                cv2.drawContours(final_roi, [final_c], -1, 255, -1)
                if cv2.countNonZero(cv2.bitwise_and(final_roi, occ_roi)) > 0:
                    continue

        out_contour = final_c.copy()
        out_contour[:, 0, 0] += bx
        out_contour[:, 0, 1] += by
        occupied[by:by2, bx:bx2] = cv2.bitwise_or(
            occ_roi, cv2.dilate(final_roi, sep_kernel, iterations=1)
        )
        out_item = dict(cand)
        out_item["contour"] = out_contour
        out_item["area"] = float(cv2.contourArea(out_contour))
        carved.append(out_item)

    return carved, occupied


def deduplicate_contours_iom(
    items: list[dict], iom_threshold: float = 0.65
) -> list[dict]:
    """Suppress overlapping duplicate contours by Intersection-over-Minimum (IoM)."""
    if len(items) <= 1:
        return list(items)

    prepared = []
    for item in items:
        contour = item.get("contour")
        if contour is None:
            continue
        raw_area = item.get("area")
        area = (
            float(raw_area)
            if raw_area is not None
            else float(cv2.contourArea(contour))
        )
        if area <= 0.0:
            continue
        bbox = cv2.boundingRect(contour)
        score = float(item.get("score", 0.0))
        out_item = dict(item)
        out_item["area"] = area
        prepared.append((score, area, bbox, out_item))

    prepared.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    kept: list[tuple[tuple[int, int, int, int], float, dict]] = []
    for _, cand_area, (bx, by, bw, bh), cand in prepared:
        cand_contour = cand["contour"]
        duplicate = False
        for (ox, oy, ow, oh), acc_area, accepted in kept:
            ix0, iy0 = max(bx, ox), max(by, oy)
            ix1, iy1 = min(bx + bw, ox + ow), min(by + bh, oy + oh)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            roi_width, roi_height = ix1 - ix0, iy1 - iy0
            smaller = min(cand_area, acc_area)
            if smaller <= 0.0 or (roi_width * roi_height) < iom_threshold * smaller:
                continue
            first = np.zeros((roi_height, roi_width), dtype=np.uint8)
            second = np.zeros_like(first)
            shifted = cand_contour.copy()
            shifted[:, 0, 0] -= ix0
            shifted[:, 0, 1] -= iy0
            other_shifted = accepted["contour"].copy()
            other_shifted[:, 0, 0] -= ix0
            other_shifted[:, 0, 1] -= iy0
            cv2.drawContours(first, [shifted], -1, 255, -1)
            cv2.drawContours(second, [other_shifted], -1, 255, -1)
            intersection = cv2.countNonZero(cv2.bitwise_and(first, second))
            if intersection / smaller >= iom_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(((bx, by, bw, bh), cand_area, cand))

    return [entry[2] for entry in kept]


def _axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    step = tile_size - overlap
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = max(0, length - tile_size)
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def iter_tile_windows(
    width: int, height: int, tile_size: int, overlap: int
):
    """Yield tile and non-overlapping centroid-ownership bounds."""
    if tile_size <= overlap or overlap < 0:
        raise ValueError("tile_size must be greater than overlap >= 0")
    xs = _axis_starts(width, tile_size, overlap)
    ys = _axis_starts(height, tile_size, overlap)
    for iy, y0 in enumerate(ys):
        y1 = min(height, y0 + tile_size)
        core_y0 = 0 if iy == 0 else (ys[iy - 1] + tile_size + y0) // 2
        core_y1 = (
            height
            if iy == len(ys) - 1
            else (y1 + ys[iy + 1]) // 2
        )
        for ix, x0 in enumerate(xs):
            x1 = min(width, x0 + tile_size)
            core_x0 = 0 if ix == 0 else (xs[ix - 1] + tile_size + x0) // 2
            core_x1 = (
                width
                if ix == len(xs) - 1
                else (x1 + xs[ix + 1]) // 2
            )
            yield {
                "tile": (x0, y0, x1, y1),
                "core": (core_x0, core_y0, core_x1, core_y1),
                "index": iy * len(xs) + ix,
                "count": len(xs) * len(ys),
            }


def _tile_vegetation_probability(tile_rgb: np.ndarray) -> np.ndarray:
    rgb = tile_rgb.astype(np.float32) / 255.0
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    eps = 1e-5

    exg = np.clip((2.0 * green - red - blue + 0.10) / 0.55, 0.0, 1.0)
    vari = (green - red) / (green + red - blue + eps)
    vari = np.clip((vari + 0.08) / 0.42, 0.0, 1.0)

    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1] / 255.0
    green_hue = np.clip(1.0 - np.abs(hue - 55.0) / 42.0, 0.0, 1.0)

    lab = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_green = np.clip((138.0 - lab[:, :, 1]) / 42.0, 0.0, 1.0)

    probability = (
        0.30 * exg
        + 0.25 * vari
        + 0.20 * lab_green
        + 0.15 * green_hue * saturation
        + 0.10 * saturation
    )
    brightness = rgb.mean(axis=2)
    probability[(brightness < 0.06) | (brightness > 0.94)] *= 0.15
    return np.clip(probability, 0.0, 1.0)


def visible_vegetation_support(
    image_rgb: np.ndarray,
    tile_size: int = 1536,
    overlap: int = 192,
    threshold: float = 0.45,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a seam-free visible-band vegetation probability and support mask."""
    height, width = image_rgb.shape[:2]
    prob_full = np.zeros((height, width), dtype=np.float32)
    scene_gray_p1, haze_offset = _compute_scene_haze_offset(image_rgb)
    h0 = 0.82 * haze_offset
    h_rgb = (
        np.array([0.75, 0.80, 0.88], dtype=np.float32) * haze_offset
        if haze_offset > 0.0
        else None
    )
    haze_scale = 245.0 / max(130.0, 255.0 - h0) if haze_offset > 0.0 else 1.0
    windows = list(iter_tile_windows(width, height, tile_size, overlap))

    for item in windows:
        x0, y0, x1, y1 = item["tile"]
        cx0, cy0, cx1, cy1 = item["core"]
        tile_crop = image_rgb[y0:y1, x0:x1]
        if haze_offset > 0.0 and h_rgb is not None:
            tile_eval = np.clip(
                (tile_crop.astype(np.float32) - h_rgb) * haze_scale, 0.0, 255.0
            ).astype(np.uint8)
        else:
            tile_eval = tile_crop
        tile_probability = _tile_vegetation_probability(tile_eval)

        tx0, ty0 = cx0 - x0, cy0 - y0
        tx1, ty1 = cx1 - x0, cy1 - y0
        prob_full[cy0:cy1, cx0:cx1] = tile_probability[ty0:ty1, tx0:tx1]

        if progress and (
            item["index"] == 0
            or (item["index"] + 1) % 20 == 0
            or item["index"] + 1 == item["count"]
        ):
            progress(
                f"RGB canopy support: tile {item['index'] + 1}/{item['count']}"
            )

    probability = (prob_full * 255).astype(np.uint8)
    strong = (prob_full >= min(0.95, threshold + 0.12)).astype(np.uint8)
    weak = (prob_full >= threshold).astype(np.uint8)
    _, labels = cv2.connectedComponents(weak, connectivity=8)
    strong_labels = np.unique(labels[strong > 0])
    strong_labels = strong_labels[strong_labels > 0]
    support = np.isin(labels, strong_labels).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    support = cv2.morphologyEx(
        support, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    support = cv2.morphologyEx(
        support, cv2.MORPH_OPEN, kernel, iterations=1
    )
    return probability, support


def resolve_compute_device() -> str:
    """
    Resolve best available PyTorch compute device across CUDA, Apple Silicon MPS,
    and CPU with a live 2x2 tensor multiplication smoke check.
    """
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        try:
            test_tensor = torch.randn(2, 2, device="cuda")
            _ = test_tensor * 2
            return "cuda"
        except Exception:  # nosec B110
            pass
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        try:
            test_tensor = torch.randn(2, 2, device="mps")
            _ = test_tensor * 2
            return "mps"
        except Exception:  # nosec B110
            pass
    return "cpu"


def _sam_autocast_context(sam):
    if (
        torch is not None
        and torch.cuda.is_available()
        and str(getattr(sam, "device", "")).startswith("cuda")
    ):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _invoke_sam_generate_masks(
    sam,
    prompt: str,
    min_area_px: int = 80,
    max_area_px: Optional[int] = None,
    confidence: Optional[float] = None,
):
    if confidence is not None:
        if hasattr(sam, "set_confidence_threshold"):
            sam.set_confidence_threshold(confidence)
        else:
            sam.confidence_threshold = confidence
    return sam.generate_masks(prompt, min_size=min_area_px, quiet=True)


def run_sam3_tiled(
    image_rgb: np.ndarray,
    sam,
    prompt: str,
    tile_size: int = 1536,
    overlap: int = 192,
    confidence: float = 0.35,
    min_area_px: int = 80,
    max_area_px: int = 100000,
    max_tiles: int = 0,
    progress: Optional[Callable[[str], None]] = None,
    vegetation_support: Optional[np.ndarray] = None,
    probability_map: Optional[np.ndarray] = None,
    gsd_m: float = 0.0725,
    strict_foliage: Optional[np.ndarray] = None,
    woody_relief_mask: Optional[np.ndarray] = None,
    enable_car_prompt: bool = True,
) -> tuple[list[dict], np.ndarray, dict]:
    """Run SAM 3 per native-resolution tile with centroid ownership and negative car veto."""
    height, width = image_rgb.shape[:2]
    if strict_foliage is None:
        strict_foliage = compute_strict_foliage_mask(image_rgb)
    if woody_relief_mask is None:
        woody_relief_mask = compute_woody_canopy_relief_mask(
            image_rgb, strict_foliage, gsd_m=gsd_m
        )
    scene_gray_p1, haze_offset = _compute_scene_haze_offset(image_rgb)
    if gsd_m >= 0.22 and tile_size > 768 and max_tiles == 0:
        tile_size = max(704, int(round(tile_size * (0.165 / gsd_m))))
        overlap = max(96, min(overlap, tile_size // 4))

    instances = []
    union_indiv_mask = np.zeros((height, width), dtype=np.uint8)
    global_vehicle_exclusion = np.zeros((height, width), dtype=np.uint8)
    global_cluster_masks: list[tuple[int, int, int, int, np.ndarray]] = []
    windows = list(iter_tile_windows(width, height, tile_size, overlap))
    if max_tiles > 0:
        windows = windows[:max_tiles]

    single_crown_max_m2 = (
        380.0 if (gsd_m >= 0.22 or haze_offset > 0.0) else 260.0
    )
    default_single_px = max(
        min_area_px + 1,
        int(round(single_crown_max_m2 / max(gsd_m ** 2, 1e-6))),
    )
    single_crown_max_px = (
        min(int(max_area_px), default_single_px)
        if int(max_area_px) == 160000
        else max(min_area_px + 1, int(max_area_px))
    )
    large_crown_thresh_px = int(round(85.0 / max(gsd_m ** 2, 1e-6)))
    max_border_touch_px = max(24, int(tile_size * 0.038))
    max_car_area_px = int(28.0 / max(gsd_m ** 2, 1e-4))
    gli_min_sam = (
        max(0.012, 0.038 - 0.00055 * haze_offset)
        if haze_offset > 0.0
        else 0.038
    )
    bg_tol_sam = 1.0 + min(7.5, 0.16 * haze_offset)
    r_dark_sam = 34.0 + 0.95 * haze_offset
    r_low_score_max = 85.0 + 0.95 * haze_offset
    vari_low_score_min = max(0.08, 0.16 - 0.0018 * haze_offset)

    effective_prompt = (
        "trees"
        if (prompt or "").strip().lower() == "tree"
        else ((prompt or "").strip() or "trees")
    )

    if hasattr(sam, "set_confidence_threshold"):
        sam.set_confidence_threshold(confidence)
    else:
        sam.confidence_threshold = confidence

    failed_tiles = 0
    skipped_barren_tiles = 0
    vehicle_vetoed = 0
    for run_index, item in enumerate(windows, start=1):
        x0, y0, x1, y1 = item["tile"]
        cx0, cy0, cx1, cy1 = item["core"]

        if vegetation_support is not None and probability_map is not None:
            veg_tile = vegetation_support[y0:y1, x0:x1]
            prob_tile = probability_map[y0:y1, x0:x1]
            if veg_tile.size > 0 and prob_tile.size > 0:
                veg_px = cv2.countNonZero(veg_tile)
                p99 = float(np.percentile(prob_tile, 99))
                if veg_px < 16 and p99 < 56.0:
                    skipped_barren_tiles += 1
                    continue

        tile = np.ascontiguousarray(image_rgb[y0:y1, x0:x1])
        ai_tile = np.ascontiguousarray(
            enhance_shadow_luminance_cielab(tile, scene_gray_p1=scene_gray_p1)
        )
        tile_foliage = strict_foliage[y0:y1, x0:x1]
        foliage_tile_ratio = float(cv2.countNonZero(tile_foliage)) / float(
            max(1, tile_foliage.size)
        )
        try:
            vehicle_exclusion = np.zeros(tile.shape[:2], dtype=np.uint8)
            with _sam_autocast_context(sam):
                sam.set_image(ai_tile)
                if enable_car_prompt and foliage_tile_ratio < 0.80:
                    try:
                        _invoke_sam_generate_masks(
                            sam,
                            prompt="car",
                            min_area_px=25,
                            confidence=0.24,
                        )
                        car_masks = list(sam.masks or [])
                        for cmask in car_masks:
                            cm_np = np.asarray(cmask).squeeze()
                            if cm_np.shape != tile.shape[:2]:
                                cm_np = cv2.resize(
                                    cm_np.astype(np.float32),
                                    (tile.shape[1], tile.shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            cm_bool = cm_np > 0.5
                            cm_area = int(np.count_nonzero(cm_bool))
                            if 25 <= cm_area <= int(max_car_area_px * 2.0):
                                cm_fol_ratio = float(
                                    np.count_nonzero(tile_foliage[cm_bool])
                                ) / float(cm_area)
                                cm_rgb = tile[cm_bool].astype(np.float32)
                                cm_r = float(cm_rgb[:, 0].mean())
                                cm_g = float(cm_rgb[:, 1].mean())
                                cm_b = float(cm_rgb[:, 2].mean())
                                cm_exg = float(2.0 * cm_g - cm_r - cm_b)
                                if (
                                    cm_area <= max_car_area_px
                                    and (cm_fol_ratio < 0.70 or cm_exg < 35.0)
                                ) or (
                                    cm_b >= cm_r + 3.0
                                    and cm_r >= 75.0
                                ):
                                    vehicle_exclusion[cm_bool] = 255
                        if np.any(vehicle_exclusion > 0):
                            veh_kernel = cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE, (3, 3)
                            )
                            vehicle_exclusion = cv2.dilate(
                                vehicle_exclusion, veh_kernel, iterations=1
                            )
                            global_vehicle_exclusion[y0:y1, x0:x1] = (
                                cv2.bitwise_or(
                                    global_vehicle_exclusion[y0:y1, x0:x1],
                                    vehicle_exclusion,
                                )
                            )
                            tile_foliage = cv2.bitwise_and(
                                tile_foliage,
                                cv2.bitwise_not(vehicle_exclusion),
                            )
                            strict_foliage[y0:y1, x0:x1] = tile_foliage
                    except Exception:  # nosec B110
                        pass
                    finally:
                        sam.masks = None
                        sam.boxes = None
                        sam.scores = None

                _invoke_sam_generate_masks(
                    sam,
                    prompt=effective_prompt,
                    min_area_px=min_area_px,
                    confidence=confidence,
                )
            masks = list(sam.masks or [])
            scores = (
                np.asarray(sam.scores).reshape(-1).tolist()
                if sam.scores is not None
                else [confidence] * len(masks)
            )
            tile_instances = []
            for mask_index, mask in enumerate(masks):
                mask_np = np.asarray(mask).squeeze()
                if mask_np.shape != tile.shape[:2]:
                    mask_np = cv2.resize(
                        mask_np.astype(np.float32),
                        (tile.shape[1], tile.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask_bool = mask_np > 0.5
                if not np.any(mask_bool):
                    continue

                mask_px = int(np.count_nonzero(mask_bool))
                if np.any(vehicle_exclusion > 0):
                    veh_inter = int(
                        np.count_nonzero(vehicle_exclusion[mask_bool])
                    )
                    veh_px = int(np.count_nonzero(vehicle_exclusion))
                    iom_denom = (
                        min(mask_px, veh_px)
                        if mask_px <= max_car_area_px * 2.5
                        else mask_px
                    )
                    if veh_inter / float(max(1, iom_denom)) >= 0.25:
                        vehicle_vetoed += 1
                        continue

                score = (
                    float(scores[mask_index])
                    if mask_index < len(scores)
                    else confidence
                )
                foliage_ratio = float(
                    np.count_nonzero(tile_foliage[mask_bool])
                ) / float(max(1, mask_px))
                rgb_px = tile[mask_bool].astype(np.float32)
                r_m = float(rgb_px[:, 0].mean())
                g_m = float(rgb_px[:, 1].mean())
                b_m = float(rgb_px[:, 2].mean())
                brightness = (r_m + g_m + b_m) / 3.0
                exg = 2.0 * g_m - r_m - b_m
                vari = (g_m - r_m) / (g_m + r_m - b_m + 1e-5)
                gli = (2.0 * g_m - r_m - b_m) / (2.0 * g_m + r_m + b_m + 1e-5)
                if haze_offset > 0.0 and score >= 0.65:
                    min_fol_ratio = 0.22
                elif score >= 0.48 and mask_px <= single_crown_max_px:
                    min_fol_ratio = 0.34
                elif mask_px > large_crown_thresh_px:
                    min_fol_ratio = 0.38
                else:
                    min_fol_ratio = 0.48
                exg_min_sam = (
                    2.5 if (haze_offset > 0.0 and score >= 0.65) else 7.0
                )
                vari_ok = (
                    vari >= 0.0
                    or (vari >= -0.005 and exg >= 10.0)
                    or (haze_offset > 0.0 and vari >= -0.045 and exg >= 7.0)
                    or (
                        haze_offset > 0.0
                        and score >= 0.65
                        and g_m >= r_m + 5.0
                        and exg >= 2.5
                    )
                )
                allow_trimmed_foliage_gate = (
                    gsd_m >= 0.22
                    and haze_offset == 0.0
                    and foliage_ratio >= min_fol_ratio
                    and 13.0 <= brightness <= 185.0
                )
                if not allow_trimmed_foliage_gate and (
                    foliage_ratio < min_fol_ratio
                    or g_m <= r_m
                    or not vari_ok
                    or gli < gli_min_sam
                    or exg < exg_min_sam
                    or brightness < 13.0
                    or brightness > 185.0
                ):
                    continue

                if (
                    probability_map is not None
                    and vegetation_support is not None
                ):
                    prob_crop = probability_map[y0:y1, x0:x1]
                    supp_crop = vegetation_support[y0:y1, x0:x1]
                    if (
                        prob_crop.shape == mask_bool.shape
                        and supp_crop.shape == mask_bool.shape
                    ):
                        mean_prob = float(prob_crop[mask_bool].mean()) / 255.0
                        supp_ratio = float(
                            np.count_nonzero(supp_crop[mask_bool])
                        ) / max(1.0, float(mask_px))
                        if (
                            mean_prob < 0.22
                            and supp_ratio < 0.18
                            and score < 0.48
                        ):
                            continue

                mask_u8 = mask_bool.astype(np.uint8) * 255
                raw_contours, _ = cv2.findContours(
                    mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not raw_contours:
                    continue
                raw_contours_sorted = sorted(
                    raw_contours, key=cv2.contourArea, reverse=True
                )[:64]
                max_raw_contour_area = float(
                    cv2.contourArea(raw_contours_sorted[0])
                )
                for raw_cnt in raw_contours_sorted:
                    raw_area = float(cv2.contourArea(raw_cnt))
                    if raw_area < min_area_px:
                        break
                    rx, ry, rw, rh = cv2.boundingRect(raw_cnt)
                    if rw <= 0 or rh <= 0:
                        continue
                    shifted_raw = raw_cnt.copy()
                    shifted_raw[:, 0, 0] -= rx
                    shifted_raw[:, 0, 1] -= ry
                    local_raw_mask = np.zeros((rh, rw), dtype=np.uint8)
                    cv2.drawContours(
                        local_raw_mask, [shifted_raw], -1, 255, -1
                    )
                    sub_fol = tile_foliage[ry : ry + rh, rx : rx + rw]
                    sub_bin = cv2.bitwise_and(local_raw_mask, sub_fol)
                    sub_bin_px = int(cv2.countNonZero(sub_bin))
                    filled_px = max(1, int(cv2.countNonZero(local_raw_mask)))
                    if sub_bin_px < min_area_px:
                        continue
                    sub_rel_frac = 0.0
                    if woody_relief_mask is not None:
                        sub_rel_crop = woody_relief_mask[
                            y0 + ry : y0 + ry + rh, x0 + rx : x0 + rx + rw
                        ]
                        sub_rel_frac = float(
                            np.count_nonzero(sub_rel_crop[sub_bin > 0])
                        ) / float(max(1, sub_bin_px))

                    c_rgb = tile[ry : ry + rh, rx : rx + rw][
                        sub_bin > 0
                    ].astype(np.float32)
                    c_r = float(c_rgb[:, 0].mean())
                    c_g = float(c_rgb[:, 1].mean())
                    c_b = float(c_rgb[:, 2].mean())
                    c_exg = 2.0 * c_g - c_r - c_b
                    c_gli = c_exg / (2.0 * c_g + c_r + c_b + 1e-5)
                    c_vari = (c_g - c_r) / (c_g + c_r - c_b + 1e-5)
                    bg_exceed = (
                        c_b >= c_g + bg_tol_sam
                        if (
                            haze_offset > 0.0
                            and (
                                c_g >= c_r + 12.0
                                or (score >= 0.65 and c_g >= c_r + 5.0)
                            )
                        )
                        else c_b >= c_g + 1.0
                    )
                    c_exg_min = (
                        2.5 if (haze_offset > 0.0 and score >= 0.65) else 7.0
                    )
                    if (
                        c_g <= c_r
                        or c_exg < c_exg_min
                        or c_gli < gli_min_sam
                        or bg_exceed
                        or (
                            c_r < r_dark_sam
                            and c_b > c_r + 3.0
                            and c_exg < 18.0
                            and (haze_offset == 0.0 or (c_g - c_r) < 12.0)
                        )
                    ):
                        continue
                    sunlit_woody_ok = (
                        gsd_m >= 0.22
                        and haze_offset == 0.0
                        and c_r <= 102.0
                        and (c_g - c_r) >= 7.5
                        and c_exg >= 14.0
                        and sub_rel_frac >= 0.25
                        and (float(sub_bin_px) / float(filled_px)) >= 0.65
                    )
                    if (
                        score < 0.24
                        and not sunlit_woody_ok
                        and (
                            c_r > r_low_score_max or c_vari < vari_low_score_min
                        )
                    ):
                        continue

                    top_touch = (
                        int(np.count_nonzero(sub_bin[: max(1, 3 - ry), :] > 0))
                        if (y0 > 0 and ry < 3)
                        else 0
                    )
                    bot_touch = (
                        int(
                            np.count_nonzero(
                                sub_bin[
                                    max(0, rh - (3 - (tile.shape[0] - (ry + rh)))) :,
                                    :,
                                ]
                                > 0
                            )
                        )
                        if (y1 < height and (ry + rh) > tile.shape[0] - 3)
                        else 0
                    )
                    left_touch = (
                        int(np.count_nonzero(sub_bin[:, : max(1, 3 - rx)] > 0))
                        if (x0 > 0 and rx < 3)
                        else 0
                    )
                    right_touch = (
                        int(
                            np.count_nonzero(
                                sub_bin[
                                    :,
                                    max(0, rw - (3 - (tile.shape[1] - (rx + rw)))) :,
                                ]
                                > 0
                            )
                        )
                        if (x1 < width and (rx + rw) > tile.shape[1] - 3)
                        else 0
                    )
                    border_touch_px = max(
                        top_touch, bot_touch, left_touch, right_touch
                    )

                    is_multi_crown_cluster = (
                        mask_px > single_crown_max_px
                        or raw_area > single_crown_max_px
                    )
                    if (
                        is_multi_crown_cluster
                        and gsd_m >= 0.22
                        and haze_offset == 0.0
                        and (
                            (
                                max_raw_contour_area <= single_crown_max_px * 2.8
                                and (
                                    (
                                        raw_area <= int(large_crown_thresh_px * 0.35)
                                        and sub_rel_frac >= 0.25
                                        and c_r <= 82.0
                                    )
                                    or (
                                        raw_area <= large_crown_thresh_px
                                        and sub_rel_frac >= 0.85
                                        and c_r <= 75.0
                                    )
                                )
                            )
                            or (
                                raw_area <= large_crown_thresh_px
                                and sub_rel_frac >= 0.45
                                and c_r <= 70.0
                                and (c_g - c_r) >= 22.0
                                and (c_b - c_r) >= 4.0
                            )
                        )
                    ):
                        is_multi_crown_cluster = False
                    if (
                        not is_multi_crown_cluster
                        and raw_area > large_crown_thresh_px
                    ):
                        raw_hull = cv2.convexHull(raw_cnt)
                        raw_hull_area = max(
                            1.0, float(cv2.contourArea(raw_hull))
                        )
                        if (float(raw_area) / raw_hull_area) < 0.60:
                            is_multi_crown_cluster = True

                    if is_multi_crown_cluster:
                        # Reject narrow tile-border-sliced strips (e.g. edge of a large marsh cut by tile boundary)
                        touches_left = x0 > 0 and rx <= 3
                        touches_right = x1 < width and (rx + rw) >= tile.shape[1] - 3
                        touches_top = y0 > 0 and ry <= 3
                        touches_bot = y1 < height and (ry + rh) >= tile.shape[0] - 3
                        if (
                            (touches_left or touches_right)
                            and rw < int(tile.shape[1] * 0.18)
                            and rh > int(tile.shape[0] * 0.24)
                        ) or (
                            (touches_top or touches_bot)
                            and rh < int(tile.shape[0] * 0.18)
                            and rw > int(tile.shape[1] * 0.24)
                        ):
                            keep_woody_strip = False
                            if (
                                (gsd_m >= 0.22 or haze_offset > 0.0)
                                and woody_relief_mask is not None
                            ):
                                strip_rel = woody_relief_mask[
                                    y0 + ry : y0 + ry + rh,
                                    x0 + rx : x0 + rx + rw,
                                ]
                                strip_rel_frac = float(
                                    np.count_nonzero(strip_rel[sub_bin > 0])
                                ) / float(max(1, sub_bin_px))
                                min_strip_rel = (
                                    0.38 if haze_offset > 0.0 else 0.55
                                )
                                if strip_rel_frac >= min_strip_rel:
                                    keep_woody_strip = True
                            if not keep_woody_strip:
                                continue
                        global_cluster_masks.append(
                            (
                                x0 + rx,
                                y0 + ry,
                                x0 + rx + rw,
                                y0 + ry + rh,
                                sub_bin,
                            )
                        )
                        continue

                    min_fill_ratio = (
                        0.32
                        if (haze_offset > 0.0 and score >= 0.65)
                        else (
                            0.36
                            if (
                                gsd_m >= 0.22
                                and haze_offset == 0.0
                                and score >= 0.55
                                and sub_rel_frac >= 0.40
                            )
                            else (0.44 if score >= 0.45 else 0.52)
                        )
                    )
                    if float(sub_bin_px) / float(filled_px) < min_fill_ratio:
                        continue

                    if (
                        score < 0.30
                        and c_r < (32.0 + 0.95 * haze_offset)
                        and c_b > c_r + 4.5
                        and c_exg < 18.0
                        and (haze_offset == 0.0 or (c_g - c_r) < 12.0)
                    ):
                        continue
                    if score < 0.28:
                        gray_roi = cv2.cvtColor(
                            tile[ry : ry + rh, rx : rx + rw],
                            cv2.COLOR_RGB2GRAY,
                        )
                        sub_filled = local_raw_mask > 0
                        min_std_low_score = (
                            8.5
                            if (
                                gsd_m >= 0.22
                                and haze_offset == 0.0
                                and sub_rel_frac >= 0.25
                            )
                            else 11.5
                        )
                        if (
                            np.any(sub_filled)
                            and float(gray_roi[sub_filled].std())
                            < min_std_low_score
                        ):
                            continue

                    if border_touch_px >= max_border_touch_px:
                        continue

                    sub_cnts, _ = cv2.findContours(
                        sub_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if not sub_cnts:
                        continue
                    contour = max(sub_cnts, key=cv2.contourArea)
                    area = float(cv2.contourArea(contour))
                    if area < min_area_px:
                        continue
                    if (
                        gsd_m >= 0.22
                        and haze_offset == 0.0
                        and area < int(large_crown_thresh_px * 1.65)
                    ):
                        cnt_perim = cv2.arcLength(contour, True)
                        cnt_circ = (
                            (4.0 * np.pi * area) / (cnt_perim ** 2)
                            if cnt_perim > 0
                            else 0.0
                        )
                        cnt_hull = float(
                            cv2.contourArea(cv2.convexHull(contour))
                        )
                        cnt_sol = area / max(1.0, cnt_hull)
                        is_deep_chl_crown = (
                            c_r <= 70.0
                            and c_g >= 85.0
                            and (c_g - c_r) >= 25.0
                            and (c_b - c_r) >= 7.0
                        )
                        if not is_deep_chl_crown and (
                            (cnt_circ >= 0.70 and cnt_sol >= 0.92)
                            or (
                                cnt_sol >= 0.85
                                and c_r >= 86.0
                                and (c_b - c_r) >= -5.0
                            )
                        ):
                            pad_r = 8
                            ry0_r = max(0, ry - pad_r)
                            ry1_r = min(tile.shape[0], ry + rh + pad_r)
                            rx0_r = max(0, rx - pad_r)
                            rx1_r = min(tile.shape[1], rx + rw + pad_r)
                            ring_m = np.zeros(
                                (ry1_r - ry0_r, rx1_r - rx0_r), dtype=np.uint8
                            )
                            ring_m[
                                ry - ry0_r : ry - ry0_r + rh,
                                rx - rx0_r : rx - rx0_r + rw,
                            ] = sub_bin
                            dil_m = cv2.dilate(
                                ring_m,
                                cv2.getStructuringElement(
                                    cv2.MORPH_ELLIPSE, (11, 11)
                                ),
                            )
                            ext_b = (dil_m > 0) & (ring_m == 0)
                            if np.any(ext_b):
                                ext_rgb_patch = tile[ry0_r:ry1_r, rx0_r:rx1_r]
                                ext_gray = cv2.cvtColor(
                                    ext_rgb_patch,
                                    cv2.COLOR_RGB2GRAY,
                                ).astype(np.float32)[ext_b]
                                ext_rgb_vals = ext_rgb_patch[ext_b].astype(
                                    np.float32
                                )
                                ext_gr_diff = float(
                                    (
                                        ext_rgb_vals[:, 1] - ext_rgb_vals[:, 0]
                                    ).mean()
                                )
                                if float(np.mean(ext_gray < 42.0)) < 0.01 and (
                                    (
                                        cnt_circ >= 0.70
                                        and cnt_sol >= 0.92
                                        and float(np.percentile(ext_gray, 10))
                                        >= 92.0
                                        and ext_gr_diff <= 2.0
                                    )
                                    or (
                                        cnt_sol >= 0.85
                                        and c_r >= 86.0
                                        and float(np.percentile(ext_gray, 10))
                                        >= 80.0
                                        and ext_gr_diff <= 0.0
                                    )
                                ):
                                    global_vehicle_exclusion[
                                        y0 + ry : y0 + ry + rh,
                                        x0 + rx : x0 + rx + rw,
                                    ][sub_bin > 0] = 255
                                    continue

                    union_indiv_mask[
                        y0 + ry : y0 + ry + rh, x0 + rx : x0 + rx + rw
                    ] = cv2.bitwise_or(
                        union_indiv_mask[
                            y0 + ry : y0 + ry + rh, x0 + rx : x0 + rx + rw
                        ],
                        sub_bin,
                    )

                    full_contour = contour.copy()
                    full_contour[:, 0, 0] += x0 + rx
                    full_contour[:, 0, 1] += y0 + ry
                    moments = cv2.moments(contour)
                    if moments["m00"] <= 0:
                        continue
                    centre_x = x0 + rx + moments["m10"] / moments["m00"]
                    centre_y = y0 + ry + moments["m01"] / moments["m00"]
                    seam_buf = max(12, min(32, overlap // 12))
                    if not (
                        max(0, cx0 - seam_buf) <= centre_x < min(width, cx1 + seam_buf)
                        and max(0, cy0 - seam_buf) <= centre_y < min(height, cy1 + seam_buf)
                    ):
                        continue
                    tile_instances.append(
                        {
                            "contour": full_contour,
                            "score": score,
                            "detector_score": score,
                            "area": area,
                            "geometry_quality": "sam3_native",
                            "source_model": "sam3",
                        }
                    )

            # SAM 3 can return nested concept masks. Suppress near-duplicates
            # per tile and then globally across tile boundaries.
            kept = deduplicate_contours_iom(tile_instances, iom_threshold=0.80)
            instances.extend(kept)
        except Exception as error:
            failed_tiles += 1
            if progress:
                progress(
                    f"SAM 3 tile {run_index}/{len(windows)} failed: {error}"
                )
        finally:
            sam.masks = None
            sam.boxes = None
            sam.scores = None

        if progress and (
            run_index == 1
            or run_index % 10 == 0
            or run_index == len(windows)
        ):
            progress(
                f"SAM 3 native tiles: {run_index}/{len(windows)} "
                f"({len(instances)} instances)"
            )

    if np.any(global_vehicle_exclusion > 0):
        not_veh = cv2.bitwise_not(global_vehicle_exclusion)
        union_indiv_mask = cv2.bitwise_and(union_indiv_mask, not_veh)
        strict_foliage[:] = cv2.bitwise_and(strict_foliage, not_veh)
        if woody_relief_mask is not None:
            woody_relief_mask[:] = cv2.bitwise_and(woody_relief_mask, not_veh)
    union_mask = union_indiv_mask.copy()
    if global_cluster_masks and (np.any(union_indiv_mask > 0) or haze_offset > 0.0):
        reach_r = max(14, min(140, int(round(12.5 / max(gsd_m, 0.03)))))
        indiv_reach = (
            cv2.dilate(
                union_indiv_mask,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * reach_r + 1, 2 * reach_r + 1)
                ),
                iterations=1,
            )
            if np.any(union_indiv_mask > 0)
            else np.zeros_like(union_indiv_mask)
        )
        k_rel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        max_stand_px = int(
            round(single_crown_max_px * (7.5 + 0.45 * haze_offset))
        )
        r_max_2b = 38.0 + 1.15 * haze_offset
        shd18_thresh = 18.0 + 1.55 * haze_offset
        shd25_thresh = 25.0 + 1.60 * haze_offset
        r_max_2c = (
            68.0
            if (gsd_m >= 0.22 and haze_offset == 0.0)
            else (46.5 + 1.10 * haze_offset)
        )
        rel_min_2b = 0.42 if haze_offset > 0.0 else 0.70
        br_min_2b = 0.0 if haze_offset > 0.0 else 8.4
        gr_min_2b = 10.0 if haze_offset > 0.0 else 20.0
        shd18_min_2b = 0.06 if haze_offset > 0.0 else 0.175
        shd25_min_2b = 0.12 if haze_offset > 0.0 else 0.235
        rel_min_2c = (
            0.45
            if (gsd_m >= 0.22 and haze_offset == 0.0)
            else (0.52 if haze_offset > 0.0 else 0.74)
        )
        gr_min_2c = (
            22.0
            if (gsd_m >= 0.22 and haze_offset == 0.0)
            else (12.0 if haze_offset > 0.0 else 18.0)
        )
        br_min_2c = (
            6.0
            if (gsd_m >= 0.22 and haze_offset == 0.0)
            else (0.0 if haze_offset > 0.0 else 1.5)
        )

        for item_cl in global_cluster_masks:
            x0, y0, x1, y1, cl_mask = item_cl[:5]
            cl_px = max(1, int(cv2.countNonZero(cl_mask)))
            if cl_px < min_area_px:
                continue
            can_run_2a = True
            if cl_px > max_stand_px and woody_relief_mask is not None:
                sub_rel_all = woody_relief_mask[y0:y1, x0:x1]
                rel_all = float(
                    np.count_nonzero(sub_rel_all[cl_mask > 0])
                ) / float(cl_px)
                if rel_all < (0.60 if haze_offset > 0.0 else 0.85):
                    can_run_2a = False
            sub_indiv = union_indiv_mask[y0:y1, x0:x1]
            inter_px = int(
                cv2.countNonZero(cv2.bitwise_and(cl_mask, sub_indiv))
            )
            overlap_frac = float(inter_px) / float(cl_px)
            sub_reach = indiv_reach[y0:y1, x0:x1]
            reach_frac = float(
                cv2.countNonZero(cv2.bitwise_and(cl_mask, sub_reach))
            ) / float(cl_px)

            # Step 2A: Crown-rim & woodland stand expansion around confirmed Pass 1 crowns
            # Note: Do not `continue` after Step 2A so large forest masks also run Step 2B
            if can_run_2a and (
                (overlap_frac >= 0.10 and reach_frac >= 0.56)
                or (
                    overlap_frac >= 0.05
                    and cl_px <= single_crown_max_px * 4
                    and reach_frac >= 0.70
                )
            ):
                clipped_cl = cv2.bitwise_and(cl_mask, sub_reach)
                new_b_2a = (clipped_cl > 0) & (sub_indiv == 0)
                new_px_2a = int(np.count_nonzero(new_b_2a))
                if new_px_2a >= min_area_px and woody_relief_mask is not None:
                    sub_rel_2a = woody_relief_mask[y0:y1, x0:x1]
                    rel_2a = float(
                        np.count_nonzero(sub_rel_2a[new_b_2a])
                    ) / float(max(1, new_px_2a))
                    if rel_2a >= (0.50 if haze_offset > 0.0 else 0.65):
                        rel_closed = cv2.morphologyEx(
                            sub_rel_2a, cv2.MORPH_CLOSE, k_rel_close, iterations=1
                        )
                        clipped_cl = cv2.bitwise_or(
                            cv2.bitwise_and(clipped_cl, sub_indiv),
                            cv2.bitwise_and(clipped_cl, rel_closed),
                        )
                        union_mask[y0:y1, x0:x1] = cv2.bitwise_or(
                            union_mask[y0:y1, x0:x1], clipped_cl
                        )
                elif new_px_2a < min_area_px:
                    union_mask[y0:y1, x0:x1] = cv2.bitwise_or(
                        union_mask[y0:y1, x0:x1], clipped_cl
                    )

            # Step 2B: Closed-canopy woody tree stands with deep 3D shadow & chlorophyll absorption
            if cl_px <= max_stand_px and woody_relief_mask is not None:
                sub_rel_cl = woody_relief_mask[y0:y1, x0:x1]
                cl_b = cl_mask > 0
                rel_cl = float(
                    np.count_nonzero(sub_rel_cl[cl_b])
                ) / float(cl_px)
                sub_box_rgb = image_rgb[y0:y1, x0:x1]
                cl_rgb = sub_box_rgb[cl_b].astype(np.float32)
                r_cl = float(cl_rgb[:, 0].mean())
                g_cl = float(cl_rgb[:, 1].mean())
                b_cl = float(cl_rgb[:, 2].mean())
                if (
                    gsd_m >= 0.22
                    and haze_offset == 0.0
                    and (
                        (
                            cl_px <= int(single_crown_max_px * 0.72)
                            and rel_cl >= 0.35
                            and (g_cl - r_cl) >= 12.0
                            and 55.0 <= r_cl <= 88.0
                        )
                        or (
                            rel_cl >= 0.42
                            and r_cl <= 68.0
                            and (g_cl - r_cl) >= 24.0
                            and (b_cl - r_cl) >= 8.0
                        )
                    )
                ):
                    rel_closed = cv2.morphologyEx(
                        sub_rel_cl, cv2.MORPH_CLOSE, k_rel_close, iterations=1
                    )
                    union_mask[y0:y1, x0:x1] = cv2.bitwise_or(
                        union_mask[y0:y1, x0:x1],
                        cv2.bitwise_and(cl_mask, rel_closed),
                    )
                    continue
                if rel_cl < rel_min_2b:
                    continue
                if (
                    r_cl <= r_max_2b
                    and (b_cl - r_cl) >= br_min_2b
                    and (g_cl - r_cl) >= gr_min_2b
                ):
                    box_gray = cv2.cvtColor(sub_box_rgb, cv2.COLOR_RGB2GRAY)
                    shd18_cl = float(np.mean(box_gray < shd18_thresh))
                    shd25_cl = float(np.mean(box_gray < shd25_thresh))
                    if (
                        shd18_cl >= shd18_min_2b and shd25_cl >= shd25_min_2b
                    ) or (
                        haze_offset > 0.0
                        and rel_cl >= 0.48
                        and (g_cl - r_cl) >= 14.0
                    ):
                        rel_closed = cv2.morphologyEx(
                            sub_rel_cl, cv2.MORPH_CLOSE, k_rel_close, iterations=1
                        )
                        union_mask[y0:y1, x0:x1] = cv2.bitwise_or(
                            union_mask[y0:y1, x0:x1],
                            cv2.bitwise_and(cl_mask, rel_closed),
                        )

        # Step 2C: Multi-Sided Woodland Interior Enclosure Fill (4-Way Cardinal Rotational Symmetry)
        # Promotes continuous forest pockets inside large SAM 3 masks that are enclosed on at least
        # 3 of the 4 cardinal directions ((int(has_n) + int(has_s) + int(has_w) + int(has_e)) >= 3).
        if woody_relief_mask is not None:
            base_union_2c = union_mask.copy()
            blk_2c = max(110, min(240, int(round(26.0 / max(gsd_m, 0.03)))))
            stride_2c = max(48, blk_2c // 2)
            pad_2c = max(60, min(160, int(round(18.0 / max(gsd_m, 0.03)))))
            min_px_2c = max(min_area_px * 2, int(round(blk_2c * blk_2c * 0.20)))
            for item_cl in global_cluster_masks:
                x0, y0, x1, y1, cl_mask = item_cl[:5]
                cl_px = max(1, int(cv2.countNonZero(cl_mask)))
                if cl_px < min_px_2c:
                    continue
                sub_rel_cl = woody_relief_mask[y0:y1, x0:x1]
                rel_closed_all = cv2.morphologyEx(
                    sub_rel_cl, cv2.MORPH_CLOSE, k_rel_close, iterations=1
                )
                sub_box_rgb = image_rgb[y0:y1, x0:x1]
                bh_cl, bw_cl = cl_mask.shape[:2]
                ys_2c = list(range(0, max(1, bh_cl - blk_2c + 1), stride_2c))
                if bh_cl > blk_2c and (bh_cl - blk_2c) not in ys_2c:
                    ys_2c.append(bh_cl - blk_2c)
                xs_2c = list(range(0, max(1, bw_cl - blk_2c + 1), stride_2c))
                if bw_cl > blk_2c and (bw_cl - blk_2c) not in xs_2c:
                    xs_2c.append(bw_cl - blk_2c)
                for by0 in ys_2c:
                    by1 = min(bh_cl, by0 + blk_2c)
                    gy0, gy1 = y0 + by0, y0 + by1
                    for bx0 in xs_2c:
                        bx1 = min(bw_cl, bx0 + blk_2c)
                        gx0, gx1 = x0 + bx0, x0 + bx1
                        w_mask = cl_mask[by0:by1, bx0:bx1]
                        w_px = int(cv2.countNonZero(w_mask))
                        if w_px < min_px_2c:
                            continue
                        w_b = w_mask > 0
                        w_rel = float(
                            np.count_nonzero(sub_rel_cl[by0:by1, bx0:bx1][w_b])
                        ) / float(w_px)
                        if w_rel < rel_min_2c:
                            continue
                        w_rgb = sub_box_rgb[by0:by1, bx0:bx1][w_b].astype(
                            np.float32
                        )
                        r_w = float(w_rgb[:, 0].mean())
                        g_w = float(w_rgb[:, 1].mean())
                        b_w = float(w_rgb[:, 2].mean())
                        if (
                            r_w > r_max_2c
                            or (g_w - r_w) < gr_min_2c
                            or (b_w - r_w) < br_min_2c
                        ):
                            continue
                        n_strip = base_union_2c[
                            max(0, gy0 - pad_2c) : gy0, gx0:gx1
                        ]
                        s_strip = base_union_2c[
                            gy1 : min(height, gy1 + pad_2c), gx0:gx1
                        ]
                        w_strip = base_union_2c[
                            gy0:gy1, max(0, gx0 - pad_2c) : gx0
                        ]
                        e_strip = base_union_2c[
                            gy0:gy1, gx1 : min(width, gx1 + pad_2c)
                        ]
                        has_n = (
                            n_strip.size > 0
                            and float(np.mean(n_strip > 0)) >= 0.06
                        )
                        has_s = (
                            s_strip.size > 0
                            and float(np.mean(s_strip > 0)) >= 0.06
                        )
                        has_w = (
                            w_strip.size > 0
                            and float(np.mean(w_strip > 0)) >= 0.06
                        )
                        has_e = (
                            e_strip.size > 0
                            and float(np.mean(e_strip > 0)) >= 0.06
                        )
                        if (
                            int(has_n) + int(has_s) + int(has_w) + int(has_e)
                        ) >= 3:
                            union_mask[gy0:gy1, gx0:gx1] = cv2.bitwise_or(
                                union_mask[gy0:gy1, gx0:gx1],
                                cv2.bitwise_and(
                                    w_mask, rel_closed_all[by0:by1, bx0:bx1]
                                ),
                            )
    elif global_cluster_masks and not np.any(union_indiv_mask > 0):
        for item_cl in global_cluster_masks:
            x0, y0, x1, y1, cl_mask = item_cl[:5]
            if woody_relief_mask is not None:
                supported_cl = cv2.bitwise_and(
                    cl_mask, woody_relief_mask[y0:y1, x0:x1]
                )
            else:
                supported_cl = cl_mask
            if np.any(supported_cl > 0):
                union_mask[y0:y1, x0:x1] = cv2.bitwise_or(
                    union_mask[y0:y1, x0:x1], supported_cl
                )

    instances = deduplicate_contours_iom(instances, iom_threshold=0.65)
    sam_carve_min_px = (
        max(min_area_px, int(round(4.2 / max(gsd_m ** 2, 1e-6))))
        if gsd_m >= 0.22
        else max(16, int(min_area_px // 2))
    )
    instances, _ = carve_non_overlapping_crowns(
        instances,
        height=height,
        width=width,
        iom_reject=0.35,
        min_area_px=sam_carve_min_px,
    )
    return instances, union_mask, {
        "tiles_processed": len(windows),
        "tiles_total": next(
            iter(iter_tile_windows(width, height, tile_size, overlap))
        )["count"],
        "failed_tiles": failed_tiles,
        "skipped_barren_tiles": skipped_barren_tiles,
        "vehicle_vetoed": vehicle_vetoed,
    }


def run_deepforest_crowns(
    image_rgb: np.ndarray,
    df_model,
    vegetation_support: Optional[np.ndarray] = None,
    probability_map: Optional[np.ndarray] = None,
    gsd_m: float = 0.0725,
    score_threshold: float = 0.20,
    progress: Optional[Callable[[str], None]] = None,
    strict_foliage: Optional[np.ndarray] = None,
    woody_relief_mask: Optional[np.ndarray] = None,
    sam_union: Optional[np.ndarray] = None,
) -> tuple[list[dict], np.ndarray, dict]:
    """Run Weecology DeepForest 2.1.0 nadir tree crown detector and verify foliage."""
    height, width = image_rgb.shape[:2]
    df_support_mask = np.zeros((height, width), dtype=np.uint8)
    if df_model is None:
        return [], df_support_mask, {"df_boxes": 0, "df_verified": 0}
    if strict_foliage is None:
        strict_foliage = compute_strict_foliage_mask(image_rgb)
    if woody_relief_mask is None:
        woody_relief_mask = compute_woody_canopy_relief_mask(
            image_rgb, strict_foliage, gsd_m=gsd_m
        )

    scene_gray_p1, haze_offset = _compute_scene_haze_offset(image_rgb)
    gli_min_df = (
        max(0.028, 0.038 - 0.00020 * haze_offset)
        if haze_offset > 0.0
        else 0.038
    )
    shadow_floor_df = 52.0 + 1.15 * haze_offset
    df_patch_size = 360 if gsd_m >= 0.22 else 600

    has_sam_union = bool(sam_union is not None and np.any(sam_union > 0))
    min_box_px = max(10, int(round(1.6 / max(gsd_m, 0.03))))
    max_box_area_px = max(
        2500, int(round(400.0 / max(gsd_m ** 2, 1e-4)))
    )

    ai_rgb = enhance_shadow_luminance_cielab(
        image_rgb, scene_gray_p1=scene_gray_p1
    )
    try:
        preds = df_model.predict_tile(
            image=ai_rgb,
            patch_size=df_patch_size,
            patch_overlap=0.25,
            iou_threshold=0.18,
        )
    except Exception as error:
        if progress:
            progress(f"DeepForest predict_tile failed: {error}")
        return [], df_support_mask, {"df_boxes": 0, "df_verified": 0}

    raw_rows: list[tuple[float, float, float, float, float]] = []
    if preds is not None:
        if hasattr(preds, "columns") and hasattr(preds, "__len__"):
            if len(preds) > 0:
                for row in preds[
                    ["xmin", "ymin", "xmax", "ymax", "score"]
                ].to_numpy():
                    raw_rows.append(
                        (
                            float(row[0]),
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                        )
                    )
        elif isinstance(preds, (list, tuple)):
            for item in preds:
                if isinstance(item, dict):
                    raw_rows.append(
                        (
                            float(item["xmin"]),
                            float(item["ymin"]),
                            float(item["xmax"]),
                            float(item["ymax"]),
                            float(item.get("score", 0.5)),
                        )
                    )
                else:
                    raw_rows.append(tuple(float(v) for v in item[:5]))

    verified: list[dict] = []
    for xmin, ymin, xmax, ymax, score in raw_rows:
        if score < score_threshold:
            continue
        x1 = max(0, int(round(xmin)))
        y1 = max(0, int(round(ymin)))
        x2 = min(width, int(round(xmax)))
        y2 = min(height, int(round(ymax)))
        bw, bh = x2 - x1, y2 - y1
        if bw < min_box_px or bh < min_box_px or (bw * bh) > max_box_area_px:
            continue

        el = np.zeros((bh, bw), dtype=np.uint8)
        cv2.ellipse(
            el,
            (bw // 2, bh // 2),
            (max(2, int(round(bw * 0.42))), max(2, int(round(bh * 0.42)))),
            0,
            0,
            360,
            255,
            -1,
        )
        el_b = el > 0
        if not np.any(el_b):
            continue

        el_px = max(1.0, float(np.count_nonzero(el_b)))
        sub_foliage = strict_foliage[y1:y2, x1:x2]
        foliage_ratio = (
            float(np.count_nonzero(sub_foliage[el_b])) / el_px
        )

        if vegetation_support is not None:
            cov_crop = vegetation_support[y1:y2, x1:x2]
            cov_ratio = float(np.count_nonzero(cov_crop[el_b])) / el_px
        else:
            cov_ratio = 1.0

        if probability_map is not None:
            prob_crop = probability_map[y1:y2, x1:x2]
            mean_p = float(prob_crop[el_b].mean()) / 255.0
        else:
            mean_p = 1.0

        roi = image_rgb[y1:y2, x1:x2]
        px = roi[el_b].astype(np.float32)
        r_m = float(px[:, 0].mean())
        g_m = float(px[:, 1].mean())
        b_m = float(px[:, 2].mean())
        exg = 2.0 * g_m - r_m - b_m
        gli = exg / (2.0 * g_m + r_m + b_m + 1e-5)
        bright = (r_m + g_m + b_m) / 3.0

        is_green_veg = (
            foliage_ratio >= (0.40 if haze_offset > 0.0 else 0.45)
            and (cov_ratio >= 0.25 or mean_p >= 0.25)
            and g_m > r_m
            and gli >= gli_min_df
            and exg >= 7.0
            and 16.0 <= bright <= 185.0
        )
        if not is_green_veg:
            continue

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g_std = float(gray_roi.std())
        g_dyn = float(
            np.percentile(gray_roi, 90) - np.percentile(gray_roi, 10)
        )
        dark_shadow_frac = float(np.mean(gray_roi < shadow_floor_df))

        if has_sam_union:
            sub_sam = sam_union[y1:y2, x1:x2]
            sam_overlap_ratio = (
                float(np.count_nonzero(sub_sam[el_b])) / el_px
            )
            if sam_overlap_ratio >= 0.20:
                if g_std < (8.5 if haze_offset > 0.0 else 9.5) or g_dyn < (
                    19.0 if haze_offset > 0.0 else 22.0
                ):
                    continue
            else:
                rel_ratio = 0.0
                if woody_relief_mask is not None:
                    sub_rel = woody_relief_mask[y1:y2, x1:x2]
                    rel_ratio = (
                        float(np.count_nonzero(sub_rel[el_b])) / el_px
                    )
                if haze_offset > 0.0:
                    if (
                        score < 0.28
                        or g_std < 9.5
                        or g_dyn < 20.0
                        or dark_shadow_frac < 0.04
                        or rel_ratio < 0.45
                    ):
                        continue
                else:
                    if (
                        score < 0.56
                        or g_std < 18.0
                        or g_dyn < 28.0
                        or dark_shadow_frac < 0.08
                        or rel_ratio < 0.60
                    ):
                        continue
        else:
            if g_std < 11.5 or g_dyn < 28.0:
                continue
            if score < 0.42:
                if g_std < 13.5 or dark_shadow_frac < 0.03:
                    continue
                if woody_relief_mask is not None:
                    sub_rel = woody_relief_mask[y1:y2, x1:x2]
                    rel_ratio = (
                        float(np.count_nonzero(sub_rel[el_b])) / el_px
                    )
                    if rel_ratio < 0.45:
                        continue

        valid_px = el_b & (sub_foliage > 0)
        df_support_mask[y1:y2, x1:x2][valid_px] = 255
        verified.append(
            {
                "box": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "radius_px": max(6, int(round(0.25 * ((x2 - x1) + (y2 - y1))))),
                "crown_radius_px": max(
                    8, int(round(0.46 * ((x2 - x1) + (y2 - y1))))
                ),
                "score": float(score),
            }
        )

    if verified:
        df_support_mask = cv2.morphologyEx(
            df_support_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        df_support_mask = cv2.bitwise_and(df_support_mask, strict_foliage)
    if progress:
        progress(
            f"DeepForest 2.1.0: verified {len(verified)}/{len(raw_rows)} tree crown boxes."
        )
    return (
        verified,
        df_support_mask,
        {"df_boxes": len(raw_rows), "df_verified": len(verified)},
    )


def _mask_contours(mask: np.ndarray, min_area_px: int) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [c for c in contours if cv2.contourArea(c) >= min_area_px]


def reconcile_canopy_outputs(
    image_rgb: np.ndarray,
    vegetation_support: np.ndarray,
    sam_union: np.ndarray,
    min_group_area_px: int = 250,
    strict_foliage: Optional[np.ndarray] = None,
    woody_relief_mask: Optional[np.ndarray] = None,
    gsd_m: float = 0.10,
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Fuse semantic canopy with nearby RGB support and flag review groups."""
    if strict_foliage is None:
        strict_foliage = compute_strict_foliage_mask(image_rgb)
    if woody_relief_mask is None:
        woody_relief_mask = compute_woody_canopy_relief_mask(
            image_rgb, strict_foliage, gsd_m=gsd_m
        )
    scene_gray_p1, haze_offset = _compute_scene_haze_offset(image_rgb)
    kernel_sz = max(5, min(15, int(round(0.9 / max(gsd_m, 0.03))) | 1))
    growth_sz = max(9, min(35, int(round(1.8 / max(gsd_m, 0.03))) | 1))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_sz, kernel_sz)
    )
    growth_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (growth_sz, growth_sz)
    )
    sam_neighbourhood = cv2.dilate(sam_union, growth_kernel, iterations=1)
    supported_growth = cv2.bitwise_and(
        vegetation_support, sam_neighbourhood
    )
    supported_growth = cv2.bitwise_and(supported_growth, strict_foliage)
    if (
        gsd_m >= 0.22
        and haze_offset == 0.0
        and woody_relief_mask is not None
    ):
        supported_growth = cv2.bitwise_and(
            supported_growth, woody_relief_mask
        )
        r_sg = image_rgb[:, :, 0].astype(np.float32)
        g_sg = image_rgb[:, :, 1].astype(np.float32)
        b_sg = image_rgb[:, :, 2].astype(np.float32)
        dark_chl_woodland = (
            (strict_foliage > 0)
            & (woody_relief_mask > 0)
            & (r_sg <= 68.0)
            & ((g_sg - r_sg) >= 24.0)
            & ((b_sg - r_sg) >= 8.0)
        ).astype(np.uint8) * 255
        woodland_reach_px = max(25, min(85, int(round(18.0 / max(gsd_m, 0.05))) | 1))
        k_wl = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (woodland_reach_px, woodland_reach_px)
        )
        sam_woodland_reach = cv2.dilate(sam_union, k_wl, iterations=1)
        supported_growth = cv2.bitwise_or(
            supported_growth,
            cv2.bitwise_and(dark_chl_woodland, sam_woodland_reach),
        )
        bridge_len = max(35, min(125, int(round(34.0 / max(gsd_m, 0.05)))))
        bridge_w = max(15, min(65, int(round(18.5 / max(gsd_m, 0.05))) | 1))
        k_v_bridge = np.ones((bridge_len, bridge_w), dtype=np.uint8)
        reach_from_n = cv2.dilate(
            sam_union, k_v_bridge, anchor=(bridge_w // 2, 0), iterations=1
        )
        reach_from_s = cv2.dilate(
            sam_union,
            k_v_bridge,
            anchor=(bridge_w // 2, bridge_len - 1),
            iterations=1,
        )
        ns_corridor = cv2.bitwise_and(reach_from_n, reach_from_s)
        ns_corridor = cv2.bitwise_and(ns_corridor, strict_foliage)
        ns_corridor = cv2.bitwise_and(
            ns_corridor, cv2.bitwise_not(sam_union)
        )
        if np.any(ns_corridor > 0):
            num_nc, nc_labels, nc_stats, _ = cv2.connectedComponentsWithStats(
                (ns_corridor > 0).astype(np.uint8), connectivity=8
            )
            for ncid in range(1, num_nc):
                nca = int(nc_stats[ncid, cv2.CC_STAT_AREA])
                if nca < 40 or nca > 9000:
                    continue
                nx0 = int(nc_stats[ncid, cv2.CC_STAT_LEFT])
                ny0 = int(nc_stats[ncid, cv2.CC_STAT_TOP])
                nw = int(nc_stats[ncid, cv2.CC_STAT_WIDTH])
                nh = int(nc_stats[ncid, cv2.CC_STAT_HEIGHT])
                if nh < int(nw * 1.05):
                    continue
                nc_m = nc_labels[ny0 : ny0 + nh, nx0 : nx0 + nw] == ncid
                nc_rel_box = woody_relief_mask[ny0 : ny0 + nh, nx0 : nx0 + nw] > 0
                nc_rel = nc_rel_box[nc_m]
                if float(np.mean(nc_rel)) < 0.28:
                    continue
                nc_rgb = image_rgb[ny0 : ny0 + nh, nx0 : nx0 + nw][nc_m].astype(
                    np.float32
                )
                nc_r = float(nc_rgb[:, 0].mean())
                nc_g = float(nc_rgb[:, 1].mean())
                nc_b = float(nc_rgb[:, 2].mean())
                if nca > 4500 and (
                    nc_r <= nc_b + 0.5 or (nc_g - nc_r) > 20.5
                ):
                    continue
                if (
                    (nc_g - nc_r) < 7.5
                    or nc_r > 104.0
                    or (
                        nc_r >= 70.0
                        and nc_b > nc_r + 2.5
                        and (nc_g - nc_r) > 24.5
                    )
                ):
                    continue
                supported_growth[ny0 : ny0 + nh, nx0 : nx0 + nw][
                    nc_m & nc_rel_box
                ] = 255
        k_h_bridge = np.ones((bridge_w, bridge_len), dtype=np.uint8)
        reach_from_w = cv2.dilate(
            sam_union, k_h_bridge, anchor=(0, bridge_w // 2), iterations=1
        )
        reach_from_e = cv2.dilate(
            sam_union,
            k_h_bridge,
            anchor=(bridge_len - 1, bridge_w // 2),
            iterations=1,
        )
        ew_corridor = cv2.bitwise_and(reach_from_w, reach_from_e)
        ew_corridor = cv2.bitwise_and(ew_corridor, strict_foliage)
        ew_corridor = cv2.bitwise_and(
            ew_corridor, cv2.bitwise_not(sam_union)
        )
        if np.any(ew_corridor > 0):
            num_ec, ec_labels, ec_stats, _ = cv2.connectedComponentsWithStats(
                (ew_corridor > 0).astype(np.uint8), connectivity=8
            )
            for ecid in range(1, num_ec):
                eca = int(ec_stats[ecid, cv2.CC_STAT_AREA])
                if eca < 40 or eca > 4500:
                    continue
                ex0 = int(ec_stats[ecid, cv2.CC_STAT_LEFT])
                ey0 = int(ec_stats[ecid, cv2.CC_STAT_TOP])
                ew = int(ec_stats[ecid, cv2.CC_STAT_WIDTH])
                eh = int(ec_stats[ecid, cv2.CC_STAT_HEIGHT])
                if ew < int(eh * 1.45):
                    continue
                ec_m = ec_labels[ey0 : ey0 + eh, ex0 : ex0 + ew] == ecid
                ec_rel_box = woody_relief_mask[ey0 : ey0 + eh, ex0 : ex0 + ew] > 0
                ec_rel = ec_rel_box[ec_m]
                if float(np.mean(ec_rel)) < 0.55:
                    continue
                ec_rgb = image_rgb[ey0 : ey0 + eh, ex0 : ex0 + ew][ec_m].astype(
                    np.float32
                )
                ec_r = float(ec_rgb[:, 0].mean())
                ec_g = float(ec_rgb[:, 1].mean())
                ec_b = float(ec_rgb[:, 2].mean())
                if (
                    ec_r > 64.0
                    or (ec_g - ec_r) < 26.0
                    or (ec_b - ec_r) > 8.5
                ):
                    continue
                supported_growth[ey0 : ey0 + eh, ex0 : ex0 + ew][
                    ec_m & ec_rel_box
                ] = 255
    coverage = cv2.bitwise_or(sam_union, supported_growth)
    close_iters = 1 if (gsd_m >= 0.22 and haze_offset == 0.0) else 2
    coverage = cv2.morphologyEx(
        coverage, cv2.MORPH_CLOSE, kernel, iterations=close_iters
    )
    coverage = cv2.bitwise_and(coverage, strict_foliage)

    residual_seed = vegetation_support
    if gsd_m >= 0.22 and haze_offset == 0.0:
        r_res = image_rgb[:, :, 0].astype(np.float32)
        g_res = image_rgb[:, :, 1].astype(np.float32)
        b_res = image_rgb[:, :, 2].astype(np.float32)
        dark_chl_res = (
            (strict_foliage > 0)
            & (r_res <= 68.0)
            & ((g_res - r_res) >= 25.0)
            & ((b_res - r_res) >= 8.0)
        ).astype(np.uint8) * 255
        residual_seed = cv2.bitwise_or(vegetation_support, dark_chl_res)
    residual = cv2.bitwise_and(
        residual_seed, cv2.bitwise_not(sam_neighbourhood)
    )
    residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, kernel, iterations=1)
    residual = cv2.bitwise_and(residual, strict_foliage)
    group_contours = []
    for contour in _mask_contours(residual, min_group_area_px):
        x, y, width, height = cv2.boundingRect(contour)
        local_mask = np.zeros((height, width), dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, 0, 0] -= x
        shifted[:, 0, 1] -= y
        cv2.drawContours(local_mask, [shifted], -1, 255, -1)
        roi_rgb = image_rgb[y : y + height, x : x + width]
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        mask_bool = local_mask > 0
        values = gray[mask_bool]
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        texture = laplacian[mask_bool]
        rgb_vals = roi_rgb[mask_bool].astype(np.float32)
        if not rgb_vals.size:
            continue
        r_m = float(rgb_vals[:, 0].mean())
        g_m = float(rgb_vals[:, 1].mean())
        b_m = float(rgb_vals[:, 2].mean())
        exg = 2.0 * g_m - r_m - b_m
        if g_m <= r_m or exg < (5.0 if haze_offset > 0.0 else 7.5):
            continue
        extent = cv2.contourArea(contour) / max(1.0, width * height)
        if (
            values.size
            and float(values.std()) >= 11.0
            and texture.size
            and float(texture.mean()) >= 9.0
            and (
                exg >= 10.0
                or (haze_offset > 0.0 and (exg >= 7.0 or (g_m - r_m) >= 11.0))
            )
            and extent >= 0.12
        ):
            if (
                haze_offset > 0.0 or gsd_m >= 0.22
            ) and woody_relief_mask is not None:
                sub_rel = woody_relief_mask[y : y + height, x : x + width]
                mask_px = max(1, int(np.count_nonzero(mask_bool)))
                rel_ratio = float(
                    np.count_nonzero(sub_rel[mask_bool])
                ) / float(mask_px)
                r_stand_max = 52.0 + 1.15 * haze_offset
                if (
                    haze_offset > 0.0
                    and rel_ratio >= 0.42
                    and (g_m - r_m) >= 10.0
                    and r_m <= r_stand_max
                ) or (
                    haze_offset == 0.0
                    and gsd_m >= 0.22
                    and rel_ratio >= 0.42
                    and r_m <= 68.0
                    and (g_m - r_m) >= 24.0
                    and (b_m - r_m) >= 8.0
                ):
                    rel_closed = cv2.morphologyEx(
                        sub_rel, cv2.MORPH_CLOSE, kernel, iterations=1
                    )
                    promoted_stand = (
                        local_mask
                        if (
                            haze_offset == 0.0
                            and gsd_m >= 0.22
                            and mask_px <= int(round(120.0 / max(gsd_m ** 2, 1e-6)))
                        )
                        else cv2.bitwise_and(local_mask, rel_closed)
                    )
                    coverage[y : y + height, x : x + width] = cv2.bitwise_or(
                        coverage[y : y + height, x : x + width], promoted_stand
                    )
            group_contours.append(contour)

    coverage = cv2.bitwise_and(coverage, strict_foliage)
    group_pixels = int(sum(cv2.contourArea(c) for c in group_contours))
    return coverage, group_contours, {
        "coverage_pixels": int(cv2.countNonZero(coverage)),
        "sam_pixels": int(cv2.countNonZero(sam_union)),
        "rgb_supported_pixels": int(cv2.countNonZero(supported_growth)),
        "review_group_pixels": group_pixels,
        "unresolved_group_count": len(group_contours),
    }


def split_dense_canopy_watershed(
    image_rgb: np.ndarray,
    coverage_mask: np.ndarray,
    instance_contours: list[np.ndarray],
    gsd_m: float = 0.10,
    tile_size: int = 1536,
    overlap: int = 192,
    min_dense_ratio: float = 0.20,
    min_area_px: int = 150,
    max_area_px: int = 30000,
    progress: Optional[Callable[[str], None]] = None,
    deepforest_boxes: Optional[list[dict]] = None,
    strict_foliage: Optional[np.ndarray] = None,
    woody_relief_mask: Optional[np.ndarray] = None,
) -> tuple[list[dict], dict]:
    """Partition unresolved dense canopy via Global Compact Geodesic Watershed."""
    height, width = image_rgb.shape[:2]
    if strict_foliage is None:
        strict_foliage = compute_strict_foliage_mask(image_rgb)
    if woody_relief_mask is None:
        woody_relief_mask = compute_woody_canopy_relief_mask(
            image_rgb, strict_foliage, gsd_m=gsd_m
        )
    scene_gray_p1, haze_offset = _compute_scene_haze_offset(image_rgb)
    coarse_or_hazy = bool(gsd_m >= 0.22 or haze_offset > 0.0)
    gli_min_ws = (
        max(0.022, 0.035 - 0.00028 * haze_offset)
        if haze_offset > 0.0
        else 0.035
    )
    individual_mask = np.zeros((height, width), dtype=np.uint8)
    if instance_contours:
        cv2.drawContours(individual_mask, instance_contours, -1, 255, -1)
    excl_sz = (
        max(5, min(15, int(round(1.35 / max(gsd_m, 0.03))) | 1))
        if coarse_or_hazy
        else 3
    )
    exclusion = cv2.dilate(
        individual_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (excl_sz, excl_sz)),
        iterations=1,
    )
    unresolved = cv2.bitwise_and(
        coverage_mask, cv2.bitwise_not(exclusion)
    )
    unresolved = cv2.bitwise_and(unresolved, strict_foliage)
    if coarse_or_hazy:
        k5_unres = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k3_unres = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        unresolved = cv2.morphologyEx(
            unresolved, cv2.MORPH_CLOSE, k5_unres, iterations=1
        )
        unresolved = cv2.bitwise_and(
            unresolved, cv2.bitwise_and(coverage_mask, cv2.bitwise_not(exclusion))
        )
        unresolved = cv2.morphologyEx(
            unresolved, cv2.MORPH_OPEN, k3_unres, iterations=1
        )
    spacing_px = int(round(5.5 / max(gsd_m, 0.02))) | 1
    spacing_px = max(21, min(121, spacing_px))
    if coarse_or_hazy:
        min_dist_px = max(16, int(round(5.0 / max(gsd_m, 0.02))))
    else:
        min_dist_floor = 11 if gsd_m >= 0.22 else 16
        min_dist_px = max(min_dist_floor, int(round(3.6 / max(gsd_m, 0.02))))

    if cv2.countNonZero(unresolved) < min_area_px:
        return [], {
            "dense_tiles": 0,
            "candidate_count": 0,
            "spacing_px": spacing_px,
            "marker_spacing_px": spacing_px,
            "method": "marker_controlled_rgb_watershed",
        }

    windows = list(iter_tile_windows(width, height, tile_size, overlap))
    dense_tiles = 0
    for item in windows:
        cx0, cy0, cx1, cy1 = item["core"]
        core = unresolved[cy0:cy1, cx0:cx1]
        if cv2.countNonZero(core) / max(1, core.size) >= min_dense_ratio:
            dense_tiles += 1

    dist_to_bg = cv2.distanceTransform(
        (unresolved > 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    rgb_f = image_rgb.astype(np.float32)
    exg = 2.0 * rgb_f[:, :, 1] - rgb_f[:, :, 0] - rgb_f[:, :, 2]
    lab_l = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(
        np.float32
    )
    max_crown_rad_px = max(12.0, 5.5 / max(gsd_m, 0.02))
    dist_norm = np.clip(dist_to_bg / max_crown_rad_px, 0.0, 1.5) * 100.0
    surface = (
        0.45 * dist_norm
        + 0.35 * lab_l
        + 0.20 * np.clip(exg + 80.0, 0.0, 255.0)
    )
    sigma = max(2.0, 0.45 / max(gsd_m, 0.02))
    surface = cv2.GaussianBlur(surface, (0, 0), sigma)

    dyn_win = max(9, min(45, int(round(2.5 / max(gsd_m, 0.03))))) | 1
    dyn_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dyn_win, dyn_win))
    gray_f = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_dyn_l = cv2.dilate(gray_f, dyn_k) - cv2.erode(gray_f, dyn_k)

    occ_seeds = np.zeros((height, width), dtype=np.uint8)
    markers = np.zeros((height, width), dtype=np.int32)
    seed_scores: dict[int, float] = {}
    seed_sources: dict[int, str] = {}
    next_id = 1

    # Seed Priority 1: DeepForest 2.1.0 verified tree crown centers
    if deepforest_boxes:
        df_sorted = sorted(
            deepforest_boxes,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        for box in df_sorted:
            cx, cy = box["center"]
            if not (0 <= cx < width and 0 <= cy < height):
                continue
            y0_w, y1_w = max(0, cy - 4), min(height, cy + 5)
            x0_w, x1_w = max(0, cx - 4), min(width, cx + 5)
            win_unres = unresolved[y0_w:y1_w, x0_w:x1_w]
            if not np.any(win_unres > 0):
                continue
            win_surf = surface[y0_w:y1_w, x0_w:x1_w].copy()
            win_surf[win_unres == 0] = -1e9
            dy, dx = np.unravel_index(int(np.argmax(win_surf)), win_surf.shape)
            sy, sx = y0_w + int(dy), x0_w + int(dx)
            if occ_seeds[sy, sx] > 0:
                continue
            crown_rad = int(
                box.get(
                    "crown_radius_px", box.get("radius_px", min_dist_px)
                )
            )
            occ_rad = max(4, int(round(crown_rad * 0.80)))
            cv2.circle(occ_seeds, (sx, sy), occ_rad, 255, -1)
            cv2.circle(markers, (sx, sy), 2, next_id, -1)
            seed_scores[next_id] = float(
                np.clip(float(box.get("score", 0.45)), 0.20, 0.98)
            )
            seed_sources[next_id] = "deepforest_watershed"
            next_id += 1

    # Seed Priority 2: Radiometric + distance local peaks inside unresolved canopy
    min_bg_dist = (
        max(1.5, 0.45 / max(gsd_m, 0.02))
        if gsd_m >= 0.22
        else max(3.0, 0.55 / max(gsd_m, 0.02))
    )
    rem_mask = (
        (unresolved > 0)
        & (occ_seeds == 0)
        & (dist_to_bg >= min_bg_dist)
    )
    if np.any(rem_mask):
        coords = peak_local_max(
            surface,
            min_distance=min_dist_px,
            labels=rem_mask.astype(np.uint8),
            exclude_border=False,
        )
        for py, px in coords:
            py_i, px_i = int(py), int(px)
            if occ_seeds[py_i, px_i] > 0:
                continue
            dyn_min_p2 = 4.5 if haze_offset > 0.0 else 8.0
            exg_min_p2 = 12.0 if haze_offset > 0.0 else 18.0
            if (
                local_dyn_l[py_i, px_i] < dyn_min_p2
                and exg[py_i, px_i] < exg_min_p2
            ):
                continue
            cv2.circle(
                occ_seeds, (px_i, py_i), int(min_dist_px * 0.72), 255, -1
            )
            cv2.circle(markers, (px_i, py_i), 2, next_id, -1)
            seed_scores[next_id] = float(
                np.clip(surface[py_i, px_i] / 210.0, 0.35, 0.85)
            )
            seed_sources[next_id] = "watershed_forest_crown"
            next_id += 1

    # Seed Priority 3: Guarantee every unresolved canopy component >= min_area_px is seeded
    # and large components (> 140 m2 per seed) receive secondary local-maxima seeds
    num_comps, comp_labels, comp_stats, _ = cv2.connectedComponentsWithStats(
        (unresolved > 0).astype(np.uint8), connectivity=8
    )
    target_basin_m2 = 165.0 if coarse_or_hazy else 135.0
    target_basin_px = max(
        min_area_px * 2, int(round(target_basin_m2 / max(gsd_m ** 2, 1e-6)))
    )
    sec_dist_mult = 0.76 if coarse_or_hazy else 0.58
    sec_dist_px = max(10, int(round(min_dist_px * sec_dist_mult)))
    for comp_id in range(1, num_comps):
        c_area = int(comp_stats[comp_id, cv2.CC_STAT_AREA])
        if c_area < min_area_px:
            continue
        cx0 = int(comp_stats[comp_id, cv2.CC_STAT_LEFT])
        cy0 = int(comp_stats[comp_id, cv2.CC_STAT_TOP])
        cw = int(comp_stats[comp_id, cv2.CC_STAT_WIDTH])
        ch = int(comp_stats[comp_id, cv2.CC_STAT_HEIGHT])
        sub_comp = comp_labels[cy0 : cy0 + ch, cx0 : cx0 + cw] == comp_id
        sub_markers = markers[cy0 : cy0 + ch, cx0 : cx0 + cw]
        existing_seeds = len(np.unique(sub_markers[sub_comp & (sub_markers > 0)]))
        if existing_seeds == 0 or (c_area > target_basin_px * max(1, existing_seeds)):
            sub_occ = occ_seeds[cy0 : cy0 + ch, cx0 : cx0 + cw]
            sub_surf = surface[cy0 : cy0 + ch, cx0 : cx0 + cw]
            cand_mask = sub_comp & (sub_occ == 0)
            if np.any(cand_mask):
                sec_coords = peak_local_max(
                    sub_surf,
                    min_distance=sec_dist_px,
                    labels=cand_mask.astype(np.uint8),
                    exclude_border=False,
                )
                for lpy, lpx in sec_coords:
                    py_i, px_i = cy0 + int(lpy), cx0 + int(lpx)
                    if occ_seeds[py_i, px_i] > 0:
                        continue
                    cv2.circle(
                        occ_seeds, (px_i, py_i), int(sec_dist_px * 0.70), 255, -1
                    )
                    cv2.circle(markers, (px_i, py_i), 2, next_id, -1)
                    seed_scores[next_id] = float(
                        np.clip(surface[py_i, px_i] / 210.0, 0.35, 0.85)
                    )
                    seed_sources[next_id] = "watershed_forest_crown"
                    next_id += 1
                    existing_seeds += 1
            if existing_seeds == 0:
                sub_dist = dist_to_bg[cy0 : cy0 + ch, cx0 : cx0 + cw].copy()
                sub_dist[~sub_comp] = -1.0
                ly, lx = np.unravel_index(int(np.argmax(sub_dist)), sub_dist.shape)
                py_i, px_i = cy0 + int(ly), cx0 + int(lx)
                cv2.circle(markers, (px_i, py_i), 2, next_id, -1)
                seed_scores[next_id] = 0.45
                seed_sources[next_id] = "watershed_forest_crown"
                next_id += 1

    if next_id == 1:
        return [], {
            "dense_tiles": dense_tiles,
            "candidate_count": 0,
            "spacing_px": spacing_px,
            "marker_spacing_px": spacing_px,
            "method": "marker_controlled_rgb_watershed",
        }

    seeded_comps = np.unique(comp_labels[markers > 0])
    seeded_comps = seeded_comps[seeded_comps > 0]
    ws_mask = np.isin(comp_labels, seeded_comps)
    labels_ws = watershed(
        -surface, markers, mask=ws_mask, compactness=0.006
    )
    default_ws_max_px = int(round(320.0 / max(gsd_m ** 2, 1e-6)))
    max_region_px = (
        min(int(max_area_px), default_ws_max_px)
        if int(max_area_px) == 140000
        else max(min_area_px + 1, int(max_area_px))
    )
    if coarse_or_hazy:
        split_basin_px = min(
            max_region_px,
            max(
                min_area_px * 3,
                int(round(185.0 / max(gsd_m ** 2, 1e-6))),
            ),
        )
        for split_pass in range(2):
            raw_slices = find_objects(labels_ws)
            any_split = False
            sub_step = max(
                13,
                int(
                    round(
                        min_dist_px
                        * (0.82 if split_pass == 0 else 0.72)
                    )
                ),
            )
            for b_id, b_sl in enumerate(raw_slices, start=1):
                if b_sl is None:
                    continue
                b_view = labels_ws[b_sl]
                b_mask = b_view == b_id
                b_area = int(np.count_nonzero(b_mask))
                if b_area <= (
                    split_basin_px if split_pass == 0 else max_region_px
                ):
                    continue
                b_u8 = b_mask.astype(np.uint8)
                b_dist = cv2.distanceTransform(b_u8, cv2.DIST_L2, 5)
                b_surf = surface[b_sl] + 2.2 * b_dist
                b_markers = np.zeros(b_mask.shape, dtype=np.int32)
                b_occ = np.zeros(b_mask.shape, dtype=np.uint8)
                local_k = 0
                if split_pass == 0:
                    b_peaks = peak_local_max(
                        b_surf,
                        min_distance=max(12, int(round(min_dist_px * 0.74))),
                        labels=b_u8,
                        exclude_border=False,
                    )
                    for py_b, px_b in b_peaks:
                        py_i, px_i = int(py_b), int(px_b)
                        if b_occ[py_i, px_i] > 0:
                            continue
                        local_k += 1
                        cv2.circle(
                            b_occ,
                            (px_i, py_i),
                            max(6, sub_step // 2),
                            255,
                            -1,
                        )
                        cv2.circle(b_markers, (px_i, py_i), 1, local_k, -1)
                min_needed = max(
                    2,
                    int(
                        np.ceil(
                            float(b_area) / float(split_basin_px * 0.78)
                        )
                    ),
                )
                if local_k < min_needed or split_pass > 0:
                    bh, bw = b_mask.shape
                    for gy in range(sub_step // 2, bh, sub_step):
                        for gx in range(sub_step // 2, bw, sub_step):
                            if b_mask[gy, gx] and b_occ[gy, gx] == 0:
                                local_k += 1
                                cv2.circle(
                                    b_occ,
                                    (gx, gy),
                                    max(6, sub_step // 2),
                                    255,
                                    -1,
                                )
                                cv2.circle(
                                    b_markers, (gx, gy), 1, local_k, -1
                                )
                if local_k >= 2:
                    b_ws = watershed(
                        -b_surf, b_markers, mask=b_mask, compactness=0.012
                    )
                    base_score = float(seed_scores.get(b_id, 0.52))
                    for k_idx in range(2, local_k + 1):
                        k_mask = b_ws == k_idx
                        if np.any(k_mask):
                            b_view[k_mask] = next_id
                            seed_scores[next_id] = base_score
                            seed_sources[next_id] = "watershed_forest_crown"
                            next_id += 1
                            any_split = True
            if not any_split:
                break

        # Region-Adjacency-Graph (RAG) small-basin coalescing:
        # Absorb over-segmented shoulder shards (< 13.5 m2) into the adjacent watershed
        # basin with the longest shared boundary, preserving isolated standalone small trees.
        coalesce_min_px = max(
            min_area_px * 2,
            int(round(13.5 / max(gsd_m ** 2, 1e-6))),
        )
        collar_min_px = max(
            min_area_px,
            int(round(6.5 / max(gsd_m ** 2, 1e-6))),
        )
        k3_rag = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _rag_pass in range(3):
            rag_slices = find_objects(labels_ws)
            small_basins: list[tuple[int, int, tuple[slice, slice]]] = []
            for b_id, b_sl in enumerate(rag_slices, start=1):
                if b_sl is None:
                    continue
                b_cnt = int(np.count_nonzero(labels_ws[b_sl] == b_id))
                if 0 < b_cnt < coalesce_min_px:
                    small_basins.append((b_cnt, b_id, b_sl))
            if not small_basins:
                break
            small_basins.sort(key=lambda item: item[0])
            merged_any = False
            for b_cnt, b_id, b_sl in small_basins:
                y0_r = max(0, b_sl[0].start - 2)
                y1_r = min(height, b_sl[0].stop + 2)
                x0_r = max(0, b_sl[1].start - 2)
                x1_r = min(width, b_sl[1].stop + 2)
                sub_ws = labels_ws[y0_r:y1_r, x0_r:x1_r]
                b_mask = sub_ws == b_id
                cur_cnt = int(np.count_nonzero(b_mask))
                if cur_cnt == 0 or cur_cnt >= coalesce_min_px:
                    continue
                dil = cv2.dilate(b_mask.astype(np.uint8), k3_rag, iterations=1) > 0
                border_labels = sub_ws[dil & (~b_mask) & (sub_ws > 0)]
                if border_labels.size > 0:
                    uniq_l, counts_l = np.unique(border_labels, return_counts=True)
                    best_nbr = int(uniq_l[int(np.argmax(counts_l))])
                    nbr_total = int(np.count_nonzero(labels_ws == best_nbr))
                    if nbr_total + cur_cnt <= max_region_px:
                        sub_ws[b_mask] = best_nbr
                        merged_any = True
                        continue
                if cur_cnt < collar_min_px:
                    sub_excl = exclusion[y0_r:y1_r, x0_r:x1_r] > 0
                    if np.any(dil & sub_excl):
                        sub_ws[b_mask] = 0
                        merged_any = True
            if not merged_any:
                break

    k5_reg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    slices = find_objects(labels_ws)
    candidates: list[dict] = []
    for label_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        sub_labels = labels_ws[sl]
        reg_mask = (sub_labels == label_id).astype(np.uint8) * 255
        if coarse_or_hazy:
            closed_reg = cv2.morphologyEx(
                reg_mask, cv2.MORPH_CLOSE, k5_reg, iterations=1
            )
            reg_mask = np.where(
                (sub_labels == 0) & (closed_reg > 0), 255, reg_mask
            ).astype(np.uint8)
        area = int(cv2.countNonZero(reg_mask))
        if area < min_area_px or area > max_region_px:
            continue
        contours, _ = cv2.findContours(
            reg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        iter_contours = [max(contours, key=cv2.contourArea)]
        for contour in iter_contours:
            cnt_area = float(cv2.contourArea(contour))
            if cnt_area < min_area_px:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = (
                (4.0 * np.pi * cnt_area) / (perimeter ** 2)
                if perimeter > 0
                else 0.0
            )
            contour = cv2.approxPolyDP(
                contour, max(1.0, 0.003 * perimeter), True
            )
            if len(contour) < 3:
                continue

            local_cnt_mask = np.zeros_like(reg_mask)
            cv2.drawContours(local_cnt_mask, [contour], -1, 255, -1)
            eval_mask = (
                cv2.bitwise_and(local_cnt_mask, reg_mask)
                if coarse_or_hazy
                else local_cnt_mask
            )
            if cv2.countNonZero(eval_mask) < min_area_px:
                eval_mask = local_cnt_mask
            crop_rgb = image_rgb[sl]
            crop_fol = strict_foliage[sl]
            cnt_px = max(1, int(cv2.countNonZero(eval_mask)))
            fol_px = int(
                cv2.countNonZero(cv2.bitwise_and(eval_mask, crop_fol))
            )
            if float(fol_px) / float(cnt_px) < 0.42:
                continue
            mean_rgb = cv2.mean(crop_rgb, mask=eval_mask)[:3]
            r_m, g_m, b_m = (
                float(mean_rgb[0]),
                float(mean_rgb[1]),
                float(mean_rgb[2]),
            )
            bright_m = (r_m + g_m + b_m) / 3.0
            exg_m = 2.0 * g_m - r_m - b_m
            gli_m = exg_m / (2.0 * g_m + r_m + b_m + 1e-5)
            if (
                g_m <= r_m
                or gli_m < gli_min_ws
                or (exg_m < 6.0 and gli_m < 0.06)
                or bright_m < 7.5
                or bright_m > 190.0
            ):
                continue
            if gsd_m >= 0.22 and haze_offset == 0.0:
                if (g_m - r_m) < 4.5:
                    continue
                crop_gray = cv2.cvtColor(
                    crop_rgb, cv2.COLOR_RGB2GRAY
                ).astype(np.float32)
                eval_b = eval_mask > 0
                if np.any(eval_b):
                    gr_vals = crop_gray[eval_b]
                    in_p10 = float(np.percentile(gr_vals, 10))
                    in_shd38 = float(np.mean(gr_vals < 38.0))
                    in_lap = float(
                        np.abs(cv2.Laplacian(crop_gray, cv2.CV_32F))[
                            eval_b
                        ].mean()
                    )
                    is_dark_chl_ws = (
                        r_m <= 68.0
                        and (g_m - r_m) >= 24.0
                        and (b_m - r_m) >= 8.0
                    )
                    if not is_dark_chl_ws and (
                        (
                            in_shd38 < 0.06
                            and in_p10 >= 50.0
                            and b_m >= r_m
                            and (
                                in_lap < 22.0
                                or (
                                    circularity < 0.42
                                    and (g_m - r_m) >= 23.5
                                    and b_m >= r_m + 1.0
                                )
                            )
                        )
                        or (
                            r_m >= 95.0
                            and b_m < r_m - 5.0
                            and circularity < 0.35
                            and in_shd38 < 0.05
                        )
                        or (
                            r_m >= 105.0
                            and in_p10 >= 85.0
                            and in_shd38 < 0.01
                        )
                    ):
                        continue

            gq = seed_sources.get(label_id, "watershed_forest_crown")
            det_score = float(seed_scores.get(label_id, 0.45))
            conf_pct, conf_tier, conf_label = compute_crown_confidence(
                crop_rgb=crop_rgb,
                cnt_mask=local_cnt_mask,
                circularity=circularity,
                detector_score=det_score,
                geometry_quality=gq,
                haze_offset=haze_offset,
            )

            contour = contour.copy()
            contour[:, 0, 0] += sl[1].start
            contour[:, 0, 1] += sl[0].start
            candidates.append(
                {
                    "contour": contour,
                    "detector_score": round(det_score, 4),
                    "score": round(float(conf_pct / 100.0), 4),
                    "confidence_pct": round(float(conf_pct), 1),
                    "confidence_tier": conf_tier,
                    "confidence_label": conf_label,
                    "area": float(cv2.contourArea(contour)),
                    "geometry_quality": gq,
                    "source_model": (
                        "deepforest_2.1"
                        if gq == "deepforest_watershed"
                        else "compact_watershed"
                    ),
                }
            )

    carve_min_px = (
        max(min_area_px, int(round(4.2 / max(gsd_m ** 2, 1e-6))))
        if coarse_or_hazy
        else max(20, int(min_area_px // 2))
    )
    candidates, occ_after_ws = carve_non_overlapping_crowns(
        candidates,
        height=height,
        width=width,
        iom_reject=0.50,
        min_area_px=carve_min_px,
        initial_occupied=individual_mask if gsd_m >= 0.22 else exclusion,
    )
    if (
        gsd_m >= 0.22
        and haze_offset == 0.0
        and woody_relief_mask is not None
    ):
        bg_blur = cv2.GaussianBlur(gray_f, (25, 25), 6.0)
        r_f = rgb_f[:, :, 0]
        g_f = rgb_f[:, :, 1]
        b_f = rgb_f[:, :, 2]
        k3_orn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k13_orn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        unocc_orn = occ_after_ws == 0
        warm_fol_orn = (
            (strict_foliage > 0)
            & unocc_orn
            & ((r_f - b_f) >= 1.5)
            & ((r_f >= 80.0) | ((bg_blur - gray_f) >= 5.0))
        ).astype(np.uint8) * 255
        warm_bed_orn = cv2.morphologyEx(
            warm_fol_orn, cv2.MORPH_OPEN, k3_orn, iterations=1
        )
        warm_bed_orn = cv2.morphologyEx(
            warm_bed_orn, cv2.MORPH_CLOSE, k13_orn, iterations=1
        )
        warm_bed_orn = cv2.bitwise_and(
            warm_bed_orn, unocc_orn.astype(np.uint8) * 255
        )
        num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
            (warm_bed_orn > 0).astype(np.uint8), connectivity=8
        )
        min_courtyard_px = max(
            1800, int(round(200.0 / max(gsd_m ** 2, 1e-6)))
        )
        max_courtyard_px = max(
            12000, int(round(4500.0 / max(gsd_m ** 2, 1e-6)))
        )
        orn_candidates: list[dict] = []
        for cid in range(1, num_cc):
            ca = int(cc_stats[cid, cv2.CC_STAT_AREA])
            if ca < min_courtyard_px or ca > max_courtyard_px:
                continue
            x0_c = int(cc_stats[cid, cv2.CC_STAT_LEFT])
            y0_c = int(cc_stats[cid, cv2.CC_STAT_TOP])
            w_c = int(cc_stats[cid, cv2.CC_STAT_WIDTH])
            h_c = int(cc_stats[cid, cv2.CC_STAT_HEIGHT])
            bed_m = (
                cc_labels[y0_c : y0_c + h_c, x0_c : x0_c + w_c] == cid
            )
            comp_m = bed_m & (
                strict_foliage[y0_c : y0_c + h_c, x0_c : x0_c + w_c] > 0
            )
            if np.count_nonzero(comp_m) < int(min_courtyard_px * 0.65):
                continue
            comp_r = r_f[y0_c : y0_c + h_c, x0_c : x0_c + w_c][comp_m]
            comp_b = b_f[y0_c : y0_c + h_c, x0_c : x0_c + w_c][comp_m]
            comp_gr = gray_f[y0_c : y0_c + h_c, x0_c : x0_c + w_c][comp_m]
            if (
                float(comp_r.mean()) < 100.0
                or float(comp_gr.mean()) < 102.5
                or float((comp_b - comp_r).mean()) > -13.0
                or float(comp_gr.std()) < 18.0
                or float(comp_gr.std()) > 29.5
            ):
                continue
            sub_gr = gray_f[y0_c : y0_c + h_c, x0_c : x0_c + w_c]
            sub_bg = bg_blur[y0_c : y0_c + h_c, x0_c : x0_c + w_c]
            sub_r = r_f[y0_c : y0_c + h_c, x0_c : x0_c + w_c]
            sub_g = g_f[y0_c : y0_c + h_c, x0_c : x0_c + w_c]
            sub_b = b_f[y0_c : y0_c + h_c, x0_c : x0_c + w_c]
            sub_exg = exg[y0_c : y0_c + h_c, x0_c : x0_c + w_c]
            sub_rel = (
                woody_relief_mask[y0_c : y0_c + h_c, x0_c : x0_c + w_c] > 0
            )
            sub_occ = (
                occ_after_ws[y0_c : y0_c + h_c, x0_c : x0_c + w_c] > 0
            )
            tree_domes = (
                bed_m
                & (~sub_occ)
                & ((sub_bg - sub_gr) >= 5.5)
                & (sub_gr <= 98.0)
                & (sub_r <= 88.0)
                & (sub_b <= sub_r + 2.0)
                & ((sub_g - sub_r) >= 3.0)
                & (sub_exg >= 7.0)
                & (sub_rel | ((sub_bg - sub_gr) >= 8.0))
            ).astype(np.uint8) * 255
            if cv2.countNonZero(tree_domes) < 16:
                continue
            dist_orn = cv2.distanceTransform(
                (tree_domes > 0).astype(np.uint8), cv2.DIST_L2, 5
            )
            peaks_orn = peak_local_max(
                dist_orn,
                min_distance=7,
                labels=(tree_domes > 0).astype(np.uint8),
                exclude_border=False,
            )
            if len(peaks_orn) < 6:
                continue
            markers_orn = np.zeros(tree_domes.shape, dtype=np.int32)
            for idx_p, (py_o, px_o) in enumerate(peaks_orn, start=1):
                cv2.circle(
                    markers_orn, (int(px_o), int(py_o)), 1, idx_p, -1
                )
            ws_orn = watershed(
                -dist_orn,
                markers_orn,
                mask=(tree_domes > 0),
                compactness=0.01,
            )
            for bid_o in range(1, len(peaks_orn) + 1):
                bm_o = (ws_orn == bid_o).astype(np.uint8) * 255
                if cv2.countNonZero(bm_o) < 4:
                    continue
                bm_o = cv2.dilate(bm_o, k3_orn, iterations=1)
                bm_o = cv2.bitwise_and(
                    bm_o,
                    (bed_m & (~sub_occ)).astype(np.uint8) * 255,
                )
                cnts_o, _ = cv2.findContours(
                    bm_o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not cnts_o:
                    continue
                c_o = max(cnts_o, key=cv2.contourArea)
                area_o = float(cv2.contourArea(c_o))
                if not (12.0 <= area_o <= 500.0):
                    continue
                (_, (rw_o, rh_o), _) = cv2.minAreaRect(c_o)
                rot_aspect = max(rw_o, rh_o) / max(1.0, min(rw_o, rh_o))
                hull_o = max(1.0, float(cv2.contourArea(cv2.convexHull(c_o))))
                sol_o = area_o / hull_o
                if rot_aspect > 2.35 or sol_o < 0.58:
                    continue
                perim_o = cv2.arcLength(c_o, True)
                circ_o = (
                    (4.0 * np.pi * area_o) / (perim_o ** 2)
                    if perim_o > 0
                    else 0.65
                )
                bx_o, by_o, bw_o, bh_o = cv2.boundingRect(c_o)
                local_m_o = np.zeros((bh_o, bw_o), dtype=np.uint8)
                shifted_o = c_o.copy()
                shifted_o[:, 0, 0] -= bx_o
                shifted_o[:, 0, 1] -= by_o
                cv2.drawContours(local_m_o, [shifted_o], -1, 255, -1)
                crop_rgb_o = image_rgb[
                    y0_c + by_o : y0_c + by_o + bh_o,
                    x0_c + bx_o : x0_c + bx_o + bw_o,
                ]
                conf_pct_o, conf_tier_o, conf_label_o = (
                    compute_crown_confidence(
                        crop_rgb=crop_rgb_o,
                        cnt_mask=local_m_o,
                        circularity=circ_o,
                        detector_score=0.58,
                        geometry_quality="watershed_forest_crown",
                        haze_offset=haze_offset,
                    )
                )
                c_full_o = c_o.copy()
                c_full_o[:, 0, 0] += x0_c
                c_full_o[:, 0, 1] += y0_c
                orn_candidates.append(
                    {
                        "contour": c_full_o,
                        "detector_score": 0.58,
                        "score": round(float(conf_pct_o / 100.0), 4),
                        "confidence_pct": round(float(conf_pct_o), 1),
                        "confidence_tier": conf_tier_o,
                        "confidence_label": conf_label_o,
                        "area": area_o,
                        "geometry_quality": "watershed_forest_crown",
                        "source_model": "compact_watershed",
                    }
                )
        if orn_candidates:
            orn_carved, _ = carve_non_overlapping_crowns(
                orn_candidates,
                height=height,
                width=width,
                iom_reject=0.35,
                min_area_px=12,
                initial_occupied=occ_after_ws,
            )
            candidates.extend(orn_carved)
    if progress:
        progress(
            f"Global compact watershed stage: {len(candidates)} crown candidates "
            f"from {next_id - 1} seeds at ~{spacing_px}px spacing."
        )
    return candidates, {
        "dense_tiles": dense_tiles,
        "candidate_count": len(candidates),
        "spacing_px": spacing_px,
        "marker_spacing_px": spacing_px,
        "method": "marker_controlled_rgb_watershed",
    }


def contours_from_mask(mask: np.ndarray, min_area_px: int = 250):
    return _mask_contours(mask, min_area_px)


DISTINCT_CROWN_PALETTE: list[str] = [
    "#10b981",
    "#3b82f6",
    "#f59e0b",
    "#f43f5e",
    "#8b5cf6",
    "#06b6d4",
    "#f97316",
    "#84cc16",
    "#ec4899",
    "#14b8a6",
]


def _min_ring_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Compute minimum Euclidean distance between two polygon rings (vertices + segments)."""
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return float("inf")
    if pts_a.shape[0] > 48:
        step_a = max(1, pts_a.shape[0] // 48)
        pts_a = np.vstack([pts_a[::step_a], pts_a[-1:]])
    if pts_b.shape[0] > 48:
        step_b = max(1, pts_b.shape[0] // 48)
        pts_b = np.vstack([pts_b[::step_b], pts_b[-1:]])

    def _pt_to_seg_min_sq(p: np.ndarray, q: np.ndarray) -> float:
        if q.shape[0] < 2:
            diff = p[:, None, :] - q[None, :, :]
            return float(np.min(np.sum(diff * diff, axis=2)))
        q0 = q[:-1]
        q1 = q[1:]
        v = q1 - q0
        w = p[:, None, :] - q0[None, :, :]
        c1 = np.sum(w * v[None, :, :], axis=2)
        c2 = np.sum(v * v, axis=1)[None, :]
        t = np.clip(c1 / np.maximum(c2, 1e-18), 0.0, 1.0)
        proj = q0[None, :, :] + t[:, :, None] * v[None, :, :]
        d2 = np.sum((p[:, None, :] - proj) ** 2, axis=2)
        return float(np.min(d2))

    d2_ab = _pt_to_seg_min_sq(pts_a, pts_b)
    if d2_ab <= 1e-20:
        return 0.0
    d2_ba = _pt_to_seg_min_sq(pts_b, pts_a)
    return float(np.sqrt(min(d2_ab, d2_ba)))


def assign_adjacent_distinct_colors(
    features: list[dict],
    palette: Optional[list[str]] = None,
    buffer_px: int = 10,
) -> list[int]:
    """
    Assign adjacent-distinct colors to crown polygon features via DSatur graph coloring
    with a deterministic randomized tie-breaker seeded by len(features).
    """
    active_palette = list(palette) if palette else list(DISTINCT_CROWN_PALETTE)
    num_colors = len(active_palette)
    n = len(features)
    if n == 0 or num_colors == 0:
        return []

    pboxes: list[Optional[tuple[float, float, float, float]]] = []
    rings: list[Optional[np.ndarray]] = []
    ring_bboxes: list[Optional[tuple[float, float, float, float]]] = []
    ring_is_geo: list[bool] = []
    diams: list[float] = []

    for feat in features:
        props = feat.setdefault("properties", {})
        pb = props.get("pixel_bounds")
        if (
            isinstance(pb, (list, tuple))
            and len(pb) == 4
            and all(isinstance(v, (int, float)) for v in pb)
        ):
            pboxes.append((float(pb[0]), float(pb[1]), float(pb[2]), float(pb[3])))
        else:
            pboxes.append(None)

        diam = props.get("crown_diameter_m")
        diams.append(
            float(diam) if isinstance(diam, (int, float)) and diam > 0 else 4.0
        )

        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        raw_ring = (
            coords[0]
            if (
                coords
                and isinstance(coords[0], (list, tuple))
                and len(coords[0]) >= 3
            )
            else None
        )
        if raw_ring is not None:
            try:
                pts = np.asarray(raw_ring, dtype=np.float64)
                if pts.ndim == 2 and pts.shape[1] >= 2 and pts.shape[0] >= 3:
                    pts = pts[:, :2]
                    rx0, ry0 = float(pts[:, 0].min()), float(pts[:, 1].min())
                    rx1, ry1 = float(pts[:, 0].max()), float(pts[:, 1].max())
                    span = max(rx1 - rx0, ry1 - ry0)
                    is_geo = (
                        span < 0.05
                        and abs(rx0) <= 180.0
                        and abs(rx1) <= 180.0
                        and abs(ry0) <= 90.0
                        and abs(ry1) <= 90.0
                    )
                    rings.append(pts)
                    ring_bboxes.append((rx0, ry0, rx1, ry1))
                    ring_is_geo.append(is_geo)
                else:
                    rings.append(None)
                    ring_bboxes.append(None)
                    ring_is_geo.append(False)
            except Exception:
                rings.append(None)
                ring_bboxes.append(None)
                ring_is_geo.append(False)
        else:
            rings.append(None)
            ring_bboxes.append(None)
            ring_is_geo.append(False)

    adj: list[set[int]] = [set() for _ in range(n)]
    buf_f = float(buffer_px)

    for i in range(n):
        pb_i = pboxes[i]
        ring_i = rings[i]
        rb_i = ring_bboxes[i]
        geo_i = ring_is_geo[i]
        di = diams[i]
        for j in range(i + 1, n):
            pb_j = pboxes[j]
            ring_j = rings[j]
            rb_j = ring_bboxes[j]
            geo_j = ring_is_geo[j]
            dj = diams[j]
            min_diam_m = min(di, dj)
            is_adjacent = False

            if pb_i is not None and pb_j is not None:
                xi, yi, wi, hi = pb_i
                xj, yj, wj, hj = pb_j
                if (
                    xi - buf_f <= xj + wj + buf_f
                    and xj - buf_f <= xi + wi + buf_f
                    and yi - buf_f <= yj + hj + buf_f
                    and yj - buf_f <= yi + hi + buf_f
                ):
                    if ring_i is not None and ring_j is not None:
                        if geo_i and geo_j:
                            thresh_m = max(1.8, 0.15 * min_diam_m)
                            dist_m = _min_ring_distance(ring_i, ring_j) * 111320.0
                            is_adjacent = dist_m <= thresh_m
                        else:
                            dist_px = _min_ring_distance(ring_i, ring_j)
                            is_adjacent = dist_px <= buf_f
                    else:
                        dx = max(0.0, xi - (xj + wj), xj - (xi + wi))
                        dy = max(0.0, yi - (yj + hj), yj - (yi + hi))
                        is_adjacent = (dx * dx + dy * dy) <= (buf_f * buf_f)
            elif ring_i is not None and ring_j is not None and rb_i is not None and rb_j is not None:
                rx0_i, ry0_i, rx1_i, ry1_i = rb_i
                rx0_j, ry0_j, rx1_j, ry1_j = rb_j
                if geo_i and geo_j:
                    thresh_m = max(2.0, 0.15 * min_diam_m)
                    deg_pad = thresh_m / 111320.0
                    if (
                        rx0_i - deg_pad <= rx1_j + deg_pad
                        and rx0_j - deg_pad <= rx1_i + deg_pad
                        and ry0_i - deg_pad <= ry1_j + deg_pad
                        and ry0_j - deg_pad <= ry1_i + deg_pad
                    ):
                        dist_m = _min_ring_distance(ring_i, ring_j) * 111320.0
                        is_adjacent = dist_m <= thresh_m
                else:
                    if (
                        rx0_i - buf_f <= rx1_j + buf_f
                        and rx0_j - buf_f <= rx1_i + buf_f
                        and ry0_i - buf_f <= ry1_j + buf_f
                        and ry0_j - buf_f <= ry1_i + buf_f
                    ):
                        dist_px = _min_ring_distance(ring_i, ring_j)
                        is_adjacent = dist_px <= buf_f

            if is_adjacent:
                adj[i].add(j)
                adj[j].add(i)

    rng = np.random.default_rng(n)
    tie_rank = rng.permutation(n).tolist()
    vertex_color_orders = [
        rng.permutation(num_colors).tolist() for _ in range(n)
    ]

    assigned: list[int] = [-1] * n
    neighbor_colors: list[set[int]] = [set() for _ in range(n)]
    uncolored: set[int] = set(range(n))

    while uncolored:
        u = max(
            uncolored,
            key=lambda v: (len(neighbor_colors[v]), len(adj[v]), tie_rank[v]),
        )
        uncolored.remove(u)
        used = neighbor_colors[u]
        chosen_color = -1
        for c_idx in vertex_color_orders[u]:
            if c_idx not in used:
                chosen_color = int(c_idx)
                break
        if chosen_color < 0:
            counts = {c_idx: 0 for c_idx in range(num_colors)}
            for nb in adj[u]:
                nb_c = assigned[nb]
                if nb_c >= 0:
                    counts[nb_c] = counts.get(nb_c, 0) + 1
            chosen_color = int(
                min(vertex_color_orders[u], key=lambda c_idx: counts[c_idx])
            )
        assigned[u] = chosen_color
        for nb in adj[u]:
            if nb in uncolored:
                neighbor_colors[nb].add(chosen_color)

    for idx, feat in enumerate(features):
        c_idx = int(assigned[idx])
        props = feat.setdefault("properties", {})
        props["color_index"] = c_idx
        props["distinct_color"] = str(active_palette[c_idx])

    return assigned
