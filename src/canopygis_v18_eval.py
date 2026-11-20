"""
CanopyGIS v1.8.1 Public Zero-Shot Planar Tree Crown Delineation Engine (MIT License).

Self-contained public evaluation module combining:
  - Meta SAM 3 full-frame + scale-adaptive tiled zero-shot proposal generation with
    Option 1 G3 shared vision_embeds reuse, immediate last-use VRAM cache eviction
    (<= 4.8 GB peak VRAM), and pre-upsample 288x288 logit filtering + tight RoI
    F.grid_sample(align_corners=False) extraction.
  - Locked CanopyGIS v1.8 physics & geometry partitioner:
      * CF2: Separable O(HW) MORPH_RECT 3D micro-relief gate + all-score cast-umbra /
        radial dome convexity veto + unfilled super-clump canopy support.
      * CF3-Scoped: Frame-invariant SI emergent allometry shield with bisecting
        shadow-crevice guard.
      * CF4: Polsby-Popper 4*pi*A/P^2 roundness priority (0.85/0.15), >= 2 inner-seam
        corner-clip veto, 1D-LUT shadow-barred 3-pass RAG coalescing, post-carve
        sliver/crescent rind pruning, and global planar Euler==1 closure
        (overlap_px == 0, holes == 0, single 4-connected component).
  - Decision 2A post-carve Tier-3 watershed score remap (strictly after planar carving
    so DeltaMask == 0 px and DeltaF1 == 0.0000 while improving COCO AP40/AP60 ranking).
  - Topology-preserving planar label-map resizing (resize_planar_predictions).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, find_objects
from shapely import STRtree, box as shapely_box
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import torch
import torch.nn.functional as F


__version__ = "1.8.1"

__all__ = [
    "__version__",
    "enhance_shadow_luminance_cielab",
    "pack_sparse_masks",
    "unpack_cached_group",
    "compute_adaptive_sam_windows",
    "_extract_sparse_items_preupsample",
    "run_sam3_sparse_batch",
    "prepare_tile_precomp_from_groups",
    "prepare_tile_precomp",
    "compute_mask_shape_metrics",
    "compute_radial_dome_gradient",
    "strtree_dedup_with_meta",
    "count_inner_window_seam_touches",
    "is_valid_foliage_patch_cf",
    "crop_solidity_and_clean4",
    "estimate_physical_crown_radius_px",
    "compute_si_shadow_crevice_topography",
    "_rag_coalesce_1d_lut",
    "build_planar_euler1_v18",
    "run_canopygis_v18_tile",
    "remap_post_carve_scores",
    "resize_planar_predictions",
    "sparse_predictions_to_tensor_dict",
    "CanopyGISv18Predictor",
]


# ==============================================================================
# 1. Input Coercion & Stream-A CIELAB Shadow-Luminance Enhancement
# ==============================================================================


def _coerce_rgb_uint8(img_rgb: np.ndarray | Image.Image | torch.Tensor) -> np.ndarray:
    """Coerce an RGB image (uint8 HWC, float32 [0,1] HWC/CHW, PIL.Image, or Tensor) to contiguous uint8 HWC."""
    if isinstance(img_rgb, Image.Image):
        return np.ascontiguousarray(np.asarray(img_rgb.convert("RGB"), dtype=np.uint8))

    if isinstance(img_rgb, torch.Tensor):
        arr = img_rgb.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
    else:
        arr = np.asarray(img_rgb)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {arr.shape}")

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size > 0 and float(np.nanmax(arr)) <= 1.0 + 1e-4:
            arr = np.clip(np.round(arr * 255.0), 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(np.round(arr), 0.0, 255.0).astype(np.uint8)
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def enhance_shadow_luminance_cielab(
    image_rgb: np.ndarray,
    scene_gray_p1: float | None = None,
) -> np.ndarray:
    """
    Stream A (AI Detector Radiometric Stream):
    - On low-haze imagery (scene_gray_p1 <= 32.0 or None): Enhance local luminance contrast
      and lift deep shadow details exclusively on the CIELAB L-channel while keeping
      chromaticity (a, b) 100% unchanged (zero Hue or Saturation shift).
    - On hazy satellite imagery (scene_gray_p1 > 32.0): Perform physical path-radiance
      subtraction (neutralizing Rayleigh blue scatter) and atmospheric transmission
      contrast/chroma restoration so dark-green crowns are not washed into haze.
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


# ==============================================================================
# 2. Sparse Mask Packing & Option 1 G3 SAM 3 Inference
# ==============================================================================


def pack_sparse_masks(
    items: list[tuple[tuple[int, int, int, int], np.ndarray, float, int]],
) -> dict[str, np.ndarray]:
    if not items:
        return {
            "bboxes": np.zeros((0, 4), dtype=np.int32),
            "scores": np.zeros((0,), dtype=np.float32),
            "seam": np.zeros((0,), dtype=np.uint8),
            "offsets": np.zeros((0, 2), dtype=np.int64),
            "packed": np.zeros((0,), dtype=np.uint8),
        }
    bboxes = np.zeros((len(items), 4), dtype=np.int32)
    scores = np.zeros((len(items),), dtype=np.float32)
    seam = np.zeros((len(items),), dtype=np.uint8)
    offsets = np.zeros((len(items), 2), dtype=np.int64)
    chunks = []
    cursor = 0
    for idx, ((y0, y1, x0, x1), crop_b, sc, sm) in enumerate(items):
        pk = np.packbits(np.ascontiguousarray(crop_b, dtype=np.uint8).ravel())
        bboxes[idx] = [y0, y1, x0, x1]
        scores[idx] = sc
        seam[idx] = sm
        offsets[idx] = [cursor, len(pk)]
        chunks.append(pk)
        cursor += len(pk)
    packed = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.uint8)
    return {
        "bboxes": bboxes,
        "scores": scores,
        "seam": seam,
        "offsets": offsets,
        "packed": packed,
    }


def unpack_cached_group(
    npz_data: Any,
    prefix: str,
) -> list[tuple[tuple[int, int, int, int], np.ndarray, float, int, int]]:
    """Returns list of ((y0, y1, x0, x1), crop_bool, score, touches_seam, area_px)."""
    bboxes = npz_data[f"{prefix}_bboxes"]
    scores = npz_data[f"{prefix}_scores"]
    seam = npz_data[f"{prefix}_seam"]
    offsets = npz_data[f"{prefix}_offsets"]
    packed = npz_data[f"{prefix}_packed"]
    out = []
    for i in range(len(scores)):
        y0, y1, x0, x1 = int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 2]), int(bboxes[i, 3])
        st, ln = int(offsets[i, 0]), int(offsets[i, 1])
        if ln <= 0 or y1 <= y0 or x1 <= x0:
            continue
        hc, wc = y1 - y0, x1 - x0
        sub = np.unpackbits(packed[st:st + ln])[:hc * wc].reshape((hc, wc)).astype(bool)
        area = int(sub.sum())
        if area > 0:
            out.append(((y0, y1, x0, x1), sub, float(scores[i]), int(seam[i]), area))
    return out


def _extract_sparse_items_preupsample(
    outputs: Any,
    post_thr: float,
    mask_thr: float,
    orig_h: int,
    orig_w: int,
    proc_h: int,
    proc_w: int,
    ox: int = 0,
    oy: int = 0,
    full_w: int | None = None,
    full_h: int | None = None,
    min_px: int = 16,
) -> list[tuple[tuple[int, int, int, int], np.ndarray, float, int]]:
    """
    Score-filter on raw 288x288 logits BEFORE upsampling and interpolate ONLY tight per-mask
    bounding RoIs via exact align_corners=False grid_sample (saves multi-GB full-frame VRAM).
    """
    scores = outputs.pred_logits.sigmoid()[0]
    if getattr(outputs, "presence_logits", None) is not None:
        scores = scores * outputs.presence_logits.sigmoid()[0]
    keep = scores > post_thr
    if not bool(keep.any().item()):
        return []

    kept_scores = scores[keep].float().detach().cpu().numpy()
    masks_lr = outputs.pred_masks[0][keep].sigmoid()
    hm, wm = int(masks_lr.shape[-2]), int(masks_lr.shape[-1])
    device = masks_lr.device
    need_rescale = (proc_h != orig_h) or (proc_w != orig_w)
    sy = float(orig_h) / float(proc_h)
    sx = float(orig_w) / float(proc_w)

    items: list[tuple[tuple[int, int, int, int], np.ndarray, float, int]] = []
    for idx_m in range(len(kept_scores)):
        sc = float(kept_scores[idx_m])
        if sc < 0.10:
            continue
        prob_lr = masks_lr[idx_m]
        nz_lr = torch.nonzero(prob_lr > 0.05, as_tuple=False)
        if nz_lr.numel() == 0:
            continue
        ly0 = max(0, int(nz_lr[:, 0].min().item()) - 2)
        ly1 = min(hm, int(nz_lr[:, 0].max().item()) + 3)
        lx0 = max(0, int(nz_lr[:, 1].min().item()) - 2)
        lx1 = min(wm, int(nz_lr[:, 1].max().item()) + 3)

        ty0 = (
            0
            if ly0 == 0
            else max(0, min(proc_h - 1, int(math.floor((ly0 + 0.5) * float(proc_h) / float(hm) - 0.5))))
        )
        ty1 = (
            proc_h
            if ly1 >= hm
            else max(ty0 + 1, min(proc_h, int(math.ceil((ly1 + 0.5) * float(proc_h) / float(hm)))))
        )
        tx0 = (
            0
            if lx0 == 0
            else max(0, min(proc_w - 1, int(math.floor((lx0 + 0.5) * float(proc_w) / float(wm) - 0.5))))
        )
        tx1 = (
            proc_w
            if lx1 >= wm
            else max(tx0 + 1, min(proc_w, int(math.ceil((lx1 + 0.5) * float(proc_w) / float(wm)))))
        )

        gy = 2.0 * (torch.arange(ty0, ty1, device=device, dtype=torch.float32) + 0.5) / float(proc_h) - 1.0
        gx = 2.0 * (torch.arange(tx0, tx1, device=device, dtype=torch.float32) + 0.5) / float(proc_w) - 1.0
        rh, rw = int(gy.shape[0]), int(gx.shape[0])
        grid = torch.stack([gx[None, :].expand(rh, rw), gy[:, None].expand(rh, rw)], dim=-1)[None]
        up_roi = (
            F.grid_sample(
                prob_lr[None, None].float(),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )[0, 0]
            > mask_thr
        )
        nz = torch.nonzero(up_roi, as_tuple=False)
        if nz.numel() == 0:
            continue
        by0 = int(nz[:, 0].min().item())
        by1 = int(nz[:, 0].max().item()) + 1
        bx0 = int(nz[:, 1].min().item())
        bx1 = int(nz[:, 1].max().item()) + 1
        py0, py1, px0, px1 = ty0 + by0, ty0 + by1, tx0 + bx0, tx0 + bx1
        crop_p = up_roi[by0:by1, bx0:bx1].detach().cpu().numpy()

        if need_rescale:
            ry0 = max(0, min(orig_h - 1, int(round(py0 * sy))))
            ry1 = max(ry0 + 1, min(orig_h, int(round(py1 * sy))))
            rx0 = max(0, min(orig_w - 1, int(round(px0 * sx))))
            rx1 = max(rx0 + 1, min(orig_w, int(round(px1 * sx))))
            crop = cv2.resize(
                crop_p.astype(np.uint8),
                (rx1 - rx0, ry1 - ry0),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        else:
            ry0, ry1, rx0, rx1 = py0, py1, px0, px1
            crop = crop_p.copy()

        if int(crop.sum()) < min_px:
            continue

        touches_inner_seam = 0
        if full_w is not None and full_h is not None:
            if (
                (ox > 0 and rx0 <= 2)
                or (ox + orig_w < full_w and rx1 >= orig_w - 2)
                or (oy > 0 and ry0 <= 2)
                or (oy + orig_h < full_h and ry1 >= orig_h - 2)
            ):
                touches_inner_seam = 1
        items.append(((oy + ry0, oy + ry1, ox + rx0, ox + rx1), crop, sc, touches_inner_seam))
    return items


def run_sam3_sparse_batch(
    sam_model: Any,
    sam_processor: Any,
    pil_images: list[Image.Image],
    prompts: list[str],
    device: torch.device,
    post_thr: float = 0.10,
    mask_thr: float = 0.50,
    max_post_side: int = 2200,
    meta_list: list[tuple[int, int, int, int, int | None, int | None]] | None = None,
) -> list[list[tuple[tuple[int, int, int, int], np.ndarray, float, int]]]:
    """
    Option 1 G3 SAM 3 inference:
      - Shares sam_model.get_vision_features(pixel_values=pv) across prompts for the same PIL image.
      - Evicts vision_cache.pop(im_key, None) immediately after its last prompt use
        (last_use[im_key] == idx_b) so at most one image's vision features reside on GPU
        at a time (peak VRAM <= 4.8 GB).
      - Applies pre-upsample 288x288 logit filtering and tight per-mask F.grid_sample extraction.
    """
    if not pil_images:
        return []
    target_sizes = []
    orig_sizes = []
    last_use: dict[int, int] = {}
    for idx_b, im in enumerate(pil_images):
        last_use[id(im)] = idx_b
        oh, ow = im.height, im.width
        orig_sizes.append((oh, ow))
        mx = max(oh, ow)
        if mx > max_post_side:
            scale = float(max_post_side) / float(mx)
            target_sizes.append([max(64, int(round(oh * scale))), max(64, int(round(ow * scale)))])
        else:
            target_sizes.append([oh, ow])

    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    sparse_batches: list[list[tuple[tuple[int, int, int, int], np.ndarray, float, int]]] = []
    vision_cache: dict[int, object] = {}
    try:
        txt_enc = sam_processor(text=prompts, return_tensors="pt").to(device)
        with torch.no_grad():
            for idx_b, im in enumerate(pil_images):
                im_key = id(im)
                if im_key not in vision_cache:
                    pv = sam_processor(images=im, return_tensors="pt")["pixel_values"].to(device)
                    if use_cuda:
                        with torch.autocast("cuda", dtype=amp_dtype):
                            vision_cache[im_key] = sam_model.get_vision_features(pixel_values=pv)
                    else:
                        vision_cache[im_key] = sam_model.get_vision_features(pixel_values=pv)
                    del pv

                if use_cuda:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        outputs = sam_model(
                            vision_embeds=vision_cache[im_key],
                            input_ids=txt_enc["input_ids"][idx_b:idx_b + 1],
                            attention_mask=txt_enc["attention_mask"][idx_b:idx_b + 1],
                        )
                else:
                    outputs = sam_model(
                        vision_embeds=vision_cache[im_key],
                        input_ids=txt_enc["input_ids"][idx_b:idx_b + 1],
                        attention_mask=txt_enc["attention_mask"][idx_b:idx_b + 1],
                    )

                if last_use.get(im_key) == idx_b:
                    vision_cache.pop(im_key, None)

                oh, ow = orig_sizes[idx_b]
                ph, pw = target_sizes[idx_b]
                if meta_list is not None:
                    ox, oy, tw, th, fw, fh = meta_list[idx_b]
                    sp = _extract_sparse_items_preupsample(
                        outputs,
                        post_thr=post_thr,
                        mask_thr=mask_thr,
                        orig_h=th,
                        orig_w=tw,
                        proc_h=ph,
                        proc_w=pw,
                        ox=ox,
                        oy=oy,
                        full_w=fw,
                        full_h=fh,
                        min_px=16,
                    )
                else:
                    sp = _extract_sparse_items_preupsample(
                        outputs,
                        post_thr=post_thr,
                        mask_thr=mask_thr,
                        orig_h=oh,
                        orig_w=ow,
                        proc_h=ph,
                        proc_w=pw,
                        min_px=16,
                    )
                sparse_batches.append(sp)
                del outputs
    finally:
        vision_cache.clear()
        if use_cuda:
            torch.cuda.empty_cache()

    return sparse_batches


def compute_adaptive_sam_windows(
    H: int,
    W: int,
    gsd_m: float | None,
) -> list[tuple[int, int, int, int]]:
    min_dim = min(H, W)
    if min_dim <= 600:
        win = max(224, int(round(0.62 * min_dim)))
    elif gsd_m is not None and gsd_m > 0:
        win = int(np.clip(round(32.0 / gsd_m), min(768, int(round(0.56 * min_dim))), 1152))
    else:
        win = int(np.clip(round(0.56 * min_dim), 512, 1152))

    win_w = min(W, win)
    win_h = min(H, win)
    if win_w >= W and win_h >= H:
        return []

    step_x = max(64, int(round(win_w * 0.62)))
    step_y = max(64, int(round(win_h * 0.62)))
    xs = sorted(set(list(range(0, max(1, W - win_w + 1), step_x)) + [max(0, W - win_w)]))
    ys = sorted(set(list(range(0, max(1, H - win_h + 1), step_y)) + [max(0, H - win_h)]))
    return [(x0, y0, min(W, x0 + win_w), min(H, y0 + win_h)) for y0 in ys for x0 in xs]


# ==============================================================================
# 3. Precomputation, Shape / Dome Physics, Spatial NMS & 1D-LUT RAG Coalescing
# ==============================================================================


def prepare_tile_precomp_from_groups(
    img_rgb: np.ndarray,
    groups: dict[str, list[tuple[tuple[int, int, int, int], np.ndarray, float, int, int]]],
    gsd_m: float | None,
) -> dict[str, Any]:
    """Precomputes per-tile color/relief channels around in-memory sparse proposal groups."""
    H, W = img_rgb.shape[:2]
    lab_full = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_l, lab_a = lab_full[:, :, 0], lab_full[:, :, 1]
    gray_f = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap_full = np.abs(cv2.Laplacian(gray_f, cv2.CV_32F))
    rgb_f = img_rgb.astype(np.float32)
    exg_full = 2.0 * rgb_f[:, :, 1] - rgb_f[:, :, 0] - rgb_f[:, :, 2]
    nonblack_mask = (rgb_f[:, :, 0] > 3.0) | (rgb_f[:, :, 1] > 3.0) | (rgb_f[:, :, 2] > 3.0)
    valid_img_px = max(1, int(np.count_nonzero(nonblack_mask)))
    scene_l_p12 = float(np.percentile(lab_l[nonblack_mask], 12)) if np.any(nonblack_mask) else 20.0
    scene_l_p25 = float(np.percentile(lab_l[nonblack_mask], 25)) if np.any(nonblack_mask) else 35.0
    scene_l_p50 = float(np.percentile(lab_l[nonblack_mask], 50)) if np.any(nonblack_mask) else 80.0

    return {
        "H": H,
        "W": W,
        "lab_l": lab_l,
        "lab_a": lab_a,
        "lap_full": lap_full,
        "rgb_f": rgb_f,
        "exg_full": exg_full,
        "nonblack_mask": nonblack_mask,
        "valid_img_px": valid_img_px,
        "scene_l_p12": scene_l_p12,
        "scene_l_p25": scene_l_p25,
        "scene_l_p50": scene_l_p50,
        "adaptive_wins": compute_adaptive_sam_windows(H, W, gsd_m),
        "groups": groups,
    }


def prepare_tile_precomp(
    img_rgb: np.ndarray,
    npz_cache: Any,
    gsd_m: float | None,
) -> dict[str, Any]:
    """Precomputes per-tile color/relief channels and unpacked sparse proposals once per tile."""
    groups = {
        k: unpack_cached_group(npz_cache, k)
        for k in ("ff_tree", "ff_ind", "ff_can", "ff_enh", "ti_tree", "ti_ind", "ti_can")
    }
    return prepare_tile_precomp_from_groups(img_rgb, groups, gsd_m)


def compute_mask_shape_metrics(crop_bool: np.ndarray) -> tuple[float, float, float, int]:
    """
    Returns (circularity, solidity, aspect_ratio, area_px) on a boolean crop mask.
    Polsby-Popper circularity = 4*pi*cnt_area / perimeter^2.
    """
    area_px = int(crop_bool.sum())
    if area_px <= 0:
        return 0.0, 0.0, 99.0, 0
    ay = crop_bool.any(axis=1)
    ax = crop_bool.any(axis=0)
    if not ay.any() or not ax.any():
        return 0.0, 0.0, 99.0, 0
    ry0, ry1 = int(np.argmax(ay)), int(len(ay) - np.argmax(ay[::-1]))
    rx0, rx1 = int(np.argmax(ax)), int(len(ax) - np.argmax(ax[::-1]))
    bh = max(1, ry1 - ry0)
    bw = max(1, rx1 - rx0)
    aspect = float(max(bw, bh)) / float(min(bw, bh))

    u8 = crop_bool.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0, aspect, area_px
    c = max(cnts, key=cv2.contourArea)
    cnt_a = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    circ = (4.0 * math.pi * cnt_a) / (perim * perim) if perim > 1e-6 else 0.0
    hull_a = max(1.0, float(cv2.contourArea(cv2.convexHull(c))))
    sol = cnt_a / hull_a
    return float(circ), float(sol), float(aspect), area_px


def compute_radial_dome_gradient(l_crop: np.ndarray, crop_bool: np.ndarray) -> float:
    """
    Computes L*_inner_50pct - L*_outer_ring using Euclidean distance transform median split.
    Sunlit convex 3D tree crowns have positive dome gradient (> 0); cast shadow umbras &
    flat clearings surrounded by brighter borders have <= 0 dome gradient.
    """
    if int(crop_bool.sum()) < 16:
        return 0.0
    dt = cv2.distanceTransform(crop_bool.astype(np.uint8), cv2.DIST_L2, 5)
    d_vals = dt[crop_bool]
    d_med = float(np.median(d_vals))
    inner = crop_bool & (dt > d_med)
    if int(inner.sum()) < int(0.15 * d_vals.size):
        inner = crop_bool & (dt >= d_med)
    outer = crop_bool & (~inner)
    if not np.any(inner) or not np.any(outer):
        return 0.0
    return float(l_crop[inner].mean() - l_crop[outer].mean())


def strtree_dedup_with_meta(
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray, float, int, int, dict]],
    iou_thr: float = 0.55,
    iom_thr: float = 0.72,
) -> list[tuple[tuple[int, int, int, int], np.ndarray, float, int, int, dict]]:
    """Deduplicates sparse mask proposals via shapely.STRtree while carrying per-proposal metadata."""
    n = len(candidates)
    if n <= 1:
        return list(candidates)
    order = sorted(
        range(n),
        key=lambda i: (1 if candidates[i][3] == 0 else 0, round(candidates[i][2], 5), candidates[i][4]),
        reverse=True,
    )
    sorted_cands = [candidates[i] for i in order]
    geoms = [shapely_box(c[0][2], c[0][0], c[0][3], c[0][1]) for c in sorted_cands]
    tree = STRtree(geoms)
    pairs = tree.query(geoms, predicate="intersects")
    adj_earlier: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(pairs[0].tolist(), pairs[1].tolist()):
        if a < b:
            adj_earlier[b].append(a)

    suppressed = [False] * n
    kept = []
    for idx in range(n):
        if suppressed[idx]:
            continue
        (by0, by1, bx0, bx1), bcrop, _, _, barea, _ = sorted_cands[idx]
        is_dup = False
        for prev in adj_earlier[idx]:
            if suppressed[prev]:
                continue
            (ay0, ay1, ax0, ax1), acrop, _, _, aarea, _ = sorted_cands[prev]
            oy0, oy1 = max(ay0, by0), min(ay1, by1)
            ox0, ox1 = max(ax0, bx0), min(ax1, bx1)
            if oy1 <= oy0 or ox0 >= ox1:
                continue
            inter = int(
                np.count_nonzero(
                    acrop[oy0 - ay0:oy1 - ay0, ox0 - ax0:ox1 - ax0]
                    & bcrop[oy0 - by0:oy1 - by0, ox0 - bx0:ox1 - bx0]
                )
            )
            if inter == 0:
                continue
            iom = float(inter) / float(min(aarea, barea))
            iou = float(inter) / float(aarea + barea - inter)
            size_ratio = float(max(aarea, barea)) / float(max(1, min(aarea, barea)))
            if iou >= iou_thr or (size_ratio <= 1.65 and iom >= iom_thr):
                is_dup = True
                break
        if is_dup:
            suppressed[idx] = True
        else:
            kept.append(sorted_cands[idx])
    return kept


def count_inner_window_seam_touches(
    bbox: tuple[int, int, int, int],
    adaptive_windows: list[tuple[int, int, int, int]],
    H: int,
    W: int,
) -> tuple[int, list[str]]:
    """Checks if a tiled SAM mask bbox coincides with inner window boundaries (not outer image edges)."""
    y0, y1, x0, x1 = bbox
    inner_xs = sorted({wx for win in adaptive_windows for wx in (win[0], win[2]) if 0 < wx < W})
    inner_ys = sorted({wy for win in adaptive_windows for wy in (win[1], win[3]) if 0 < wy < H})
    touched = []
    for ix in inner_xs:
        if abs(x0 - ix) <= 2:
            touched.append(f"x0~{ix}")
        if abs(x1 - ix) <= 2:
            touched.append(f"x1~{ix}")
    for iy in inner_ys:
        if abs(y0 - iy) <= 2:
            touched.append(f"y0~{iy}")
        if abs(y1 - iy) <= 2:
            touched.append(f"y1~{iy}")
    return len(touched), touched


def is_valid_foliage_patch_cf(
    img_rgb: np.ndarray,
    lab_l: np.ndarray,
    lap_full: np.ndarray,
    green_px_mask: np.ndarray | None,
    scene_l_p12: float,
    scene_l_p25: float,
    bbox: tuple[int, int, int, int],
    crop_bool: np.ndarray,
    score: float,
    cf_green_only: bool = False,
    cf_umbra_veto: bool = False,
) -> bool:
    y0, y1, x0, x1 = bbox
    px_rgb = img_rgb[y0:y1, x0:x1][crop_bool].astype(np.float32)
    if px_rgb.shape[0] < 16:
        return False

    if cf_green_only and green_px_mask is not None:
        g_frac = float(np.count_nonzero(green_px_mask[y0:y1, x0:x1][crop_bool])) / float(px_rgb.shape[0])
        if g_frac < 0.50:
            return False

    r_m = float(px_rgb[:, 0].mean())
    g_m = float(px_rgb[:, 1].mean())
    b_m = float(px_rgb[:, 2].mean())
    l_vals = lab_l[y0:y1, x0:x1][crop_bool]
    l_m = float(l_vals.mean())
    l_std = float(l_vals.std())

    if max(r_m, g_m, b_m) < 6.0:
        return False

    exg = 2.0 * g_m - r_m - b_m
    if r_m > g_m + 10.0 and exg < -8.0 and score < (0.78 if cf_umbra_veto else 0.72):
        return False

    if not cf_umbra_veto:
        if score < 0.50:
            if l_m < max(22.0, scene_l_p12 * 0.95) and l_std < 9.5 and exg < 6.0:
                return False
            if g_m < r_m - 3.0 and exg < -2.0:
                return False
        return True

    # CF2(a): Illumination-Aware Cast-Umbra + Flat-Patch Veto across ALL SAM scores
    lap_m = float(lap_full[y0:y1, x0:x1][crop_bool].mean())
    dome_grad = compute_radial_dome_gradient(lab_l[y0:y1, x0:x1], crop_bool)

    if l_m < max(58.0, scene_l_p25 * 1.05) and (l_std < 22.0 or lap_m < 14.5) and dome_grad <= 0.0:
        return False

    if score < 0.52 and l_m < max(64.0, scene_l_p25 * 1.08) and l_std < 24.0 and lap_m < 15.0:
        if dome_grad <= 6.0 or exg < 18.0:
            return False

    if l_m < max(46.0, scene_l_p12 * 1.08) and l_std < 15.0 and lap_m < 10.0:
        return False

    if score < 0.54 and g_m < r_m - 3.0 and exg < -2.0:
        return False

    return True


def crop_solidity_and_clean4(crop_bool: np.ndarray) -> tuple[float, float, np.ndarray, int]:
    """
    Extracts largest 4-connected component, fills internal holes, and returns
    (solidity, aspect_ratio, clean_crop_bool, clean_area).
    """
    filled = binary_fill_holes(crop_bool)
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(filled.astype(np.uint8), connectivity=4)
    if n_lbl <= 1:
        return 0.0, 99.0, filled, 0
    best_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = lbl == best_id
    clean_area = int(stats[best_id, cv2.CC_STAT_AREA])
    w = max(1, int(stats[best_id, cv2.CC_STAT_WIDTH]))
    h = max(1, int(stats[best_id, cv2.CC_STAT_HEIGHT]))
    aspect = float(max(w, h)) / float(min(w, h))
    u8 = clean.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, aspect, clean, clean_area
    c = max(cnts, key=cv2.contourArea)
    hull_a = max(1.0, float(cv2.contourArea(cv2.convexHull(c))))
    sol = float(cv2.contourArea(c)) / hull_a
    return sol, aspect, clean, clean_area


def estimate_physical_crown_radius_px(
    H: int,
    W: int,
    gsd_m: float | None,
    proto_pool: list[tuple[tuple[int, int, int, int], np.ndarray, float, int, int]],
) -> tuple[float, float, float]:
    """
    Continuous SI-grounded crown radius estimator:
    Uses GeoTIFF GSD when present or a smooth resolution power-law prior otherwise,
    combined with the 30th percentile of compact single-crown SAM 3 prototypes.
    Returns (r_min_px, r_med_px, r_max_px).
    """
    min_dim = float(min(H, W))
    if gsd_m is not None and gsd_m > 0:
        r_phys_lo = max(5.5, 0.42 / gsd_m)
        r_proto_hi = min(0.090 * min_dim, 4.5 / gsd_m)
        r_prior = float(np.clip(2.15 / gsd_m, max(14.0, r_phys_lo * 1.4), max(24.0, 0.080 * min_dim)))
    else:
        r_phys_lo = max(6.0, 0.010 * min_dim)
        r_proto_hi = 0.085 * min_dim
        r_prior = float(np.clip(23.5 * ((min_dim / 1000.0) ** 1.65), 18.0, 0.075 * min_dim))

    proto_rads = []
    for _bbox, crop, sc, seam, area in proto_pool:
        if seam != 0 or sc < 0.48:
            continue
        r_eq = math.sqrt(float(area) / math.pi)
        if not (r_phys_lo <= r_eq <= r_proto_hi):
            continue
        sol, aspect, _, clean_a = crop_solidity_and_clean4(crop)
        if clean_a > 0 and sol >= 0.80 and aspect <= 1.68:
            proto_rads.append(math.sqrt(float(clean_a) / math.pi))

    if proto_rads:
        r_emp = float(np.percentile(proto_rads, 30))
        w_emp = min(1.0, float(len(proto_rads)) / 14.0)
        r_med = w_emp * r_emp + (1.0 - w_emp) * r_prior
    else:
        r_med = r_prior

    r_med = float(np.clip(r_med, max(14.0, r_phys_lo * 1.15), max(30.0, 0.085 * min_dim)))
    r_min = max(5.0, min(r_phys_lo, 0.32 * r_med))
    r_max = max(0.185 * min_dim, 4.50 * r_med)
    return r_min, r_med, r_max


def compute_si_shadow_crevice_topography(
    img_rgb: np.ndarray,
    canopy_support: np.ndarray,
    r_med_px: float,
    foliage_ref_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale-Invariant Shadow-Crevice & Dome Relief Surface with Pixel-Level Bare-Soil /
    Non-Foliage Self-Calibration.
    """
    rgb_f = img_rgb.astype(np.float32)
    r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
    nodata = (r <= 3.0) & (g <= 3.0) & (b <= 3.0)

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_ch, a_ch = lab[:, :, 0], lab[:, :, 1]
    exg = 2.0 * g - r - b

    if foliage_ref_mask is not None and int(np.count_nonzero(foliage_ref_mask)) >= 400:
        ref_a = a_ch[foliage_ref_mask]
        ref_exg = exg[foliage_ref_mask]
        ref_l = l_ch[foliage_ref_mask]
        a_hi = float(np.percentile(ref_a, 98)) + 5.0
        exg_lo = float(np.percentile(ref_exg, 2)) - 8.0
        l_hi = float(np.percentile(ref_l, 98)) + 14.0
        bare_soil_px = (
            ((a_ch > a_hi) & (exg < exg_lo))
            | ((l_ch > l_hi) & (exg < exg_lo))
            | ((r > g + 12.0) & (exg < -10.0))
        )
    else:
        bare_soil_px = (r > g + 8.0) & (exg < -6.0)

    valid_px = (~nodata) & (~bare_soil_px) & canopy_support
    if np.count_nonzero(valid_px) < 64:
        valid_px = (~nodata) & (~bare_soil_px)
    if np.count_nonzero(valid_px) < 64:
        valid_px = ~nodata

    l_vals = l_ch[valid_px]
    exg_vals = exg[valid_px]
    l_p12 = float(np.percentile(l_vals, 12)) if l_vals.size else 22.0
    l_p22 = float(np.percentile(l_vals, 22)) if l_vals.size else 32.0
    l_p42 = float(np.percentile(l_vals, 42)) if l_vals.size else 55.0
    exg_p12 = float(np.percentile(exg_vals, 12)) if exg_vals.size else 0.0

    deep_floor_gap = (
        nodata
        | bare_soil_px
        | (l_ch < max(10.0, l_p12 * 0.86))
        | ((l_ch < l_p22) & (exg < max(1.5, exg_p12)))
        | ((b > g + 3.5) & (l_ch < l_p42))
    )
    active_canopy = canopy_support & (~deep_floor_gap)

    sig_smooth = max(0.8, 0.10 * r_med_px)
    l_smooth = cv2.GaussianBlur(l_ch, (0, 0), sig_smooth)
    exg_smooth = cv2.GaussianBlur(exg, (0, 0), sig_smooth)

    k_bh_sz = max(5, min(121, int(round(0.52 * r_med_px)) | 1))
    k_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bh_sz, k_bh_sz))
    blackhat = cv2.morphologyEx(l_smooth, cv2.MORPH_BLACKHAT, k_bh)

    sig_loc = max(2.5, 0.45 * r_med_px)
    local_mean_l = cv2.GaussianBlur(l_smooth, (0, 0), sig_loc)
    local_sq_l = cv2.GaussianBlur(l_smooth ** 2, (0, 0), sig_loc)
    local_std_l = np.sqrt(np.maximum(local_sq_l - local_mean_l ** 2, 1.0))

    bh_vals = blackhat[active_canopy] if np.any(active_canopy) else blackhat.ravel()
    bh_thr = float(np.percentile(bh_vals, 76)) if bh_vals.size else 7.0

    shadow_crevices = (
        (~active_canopy)
        | (blackhat >= max(4.0, bh_thr))
        | (l_smooth < (local_mean_l - 0.52 * local_std_l))
    )
    crevice_free = active_canopy & (~shadow_crevices)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    crevice_free_u8 = cv2.morphologyEx(crevice_free.astype(np.uint8), cv2.MORPH_OPEN, k_open, iterations=1)
    crevice_free = crevice_free_u8 > 0

    dist_crevice = cv2.distanceTransform(crevice_free_u8, cv2.DIST_L2, 5)
    dist_active = cv2.distanceTransform(active_canopy.astype(np.uint8), cv2.DIST_L2, 5)

    def _norm01(arr: np.ndarray) -> np.ndarray:
        vals = arr[active_canopy] if np.any(active_canopy) else arr.ravel()
        if vals.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        p5, p98 = float(np.percentile(vals, 5)), float(np.percentile(vals, 98))
        return np.clip((arr - p5) / max(p98 - p5, 1e-5), 0.0, 1.0)

    elevation_map = (
        0.48 * _norm01(dist_crevice)
        + 0.20 * _norm01(dist_active)
        + 0.22 * _norm01(l_smooth)
        + 0.10 * _norm01(exg_smooth)
    )
    return active_canopy, shadow_crevices, crevice_free, elevation_map


def _union_slice_pair(
    sl_a: tuple[slice, slice],
    sl_b: tuple[slice, slice],
) -> tuple[slice, slice]:
    return (
        slice(min(sl_a[0].start, sl_b[0].start), max(sl_a[0].stop, sl_b[0].stop)),
        slice(min(sl_a[1].start, sl_b[1].start), max(sl_a[1].stop, sl_b[1].stop)),
    )


def _rag_coalesce_1d_lut(
    ws_labels: np.ndarray,
    shadow_crevices: np.ndarray,
    coalesce_min_px: int,
    max_crown_px: int,
    use_shadow_barrier: bool = True,
) -> np.ndarray:
    """
    3-Pass Region Adjacency Graph (RAG) Coalescing powered by 1D LUT (np.bincount)
    + scipy.ndimage.find_objects + Boundary Shadow Barrier.
    """
    labels = ws_labels.copy()
    H, W = labels.shape
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for _pass in range(3):
        area_lut = np.bincount(labels.ravel())
        if len(area_lut) <= 1:
            break
        cur_slices = list(find_objects(labels))
        small_basins = []
        for b_id, sl in enumerate(cur_slices, start=1):
            if sl is None or b_id >= len(area_lut):
                continue
            cnt = int(area_lut[b_id])
            if 0 < cnt < coalesce_min_px:
                small_basins.append((cnt, b_id))
        if not small_basins:
            break
        small_basins.sort(key=lambda x: x[0])
        merged_any = False

        for _, b_id in small_basins:
            cur_cnt = int(area_lut[b_id])
            sl = cur_slices[b_id - 1]
            if cur_cnt == 0 or cur_cnt >= coalesce_min_px or sl is None:
                continue
            y0 = max(0, sl[0].start - 3)
            y1 = min(H, sl[0].stop + 3)
            x0 = max(0, sl[1].start - 3)
            x1 = min(W, sl[1].stop + 3)
            sub_lbl = labels[y0:y1, x0:x1]
            b_mask = sub_lbl == b_id
            if not np.any(b_mask):
                continue
            dil = cv2.dilate(b_mask.astype(np.uint8), k5, iterations=1) > 0
            rim = dil & (~b_mask) & (sub_lbl > 0)
            if not np.any(rim):
                continue

            border_nbrs = sub_lbl[rim]
            uniq_n, counts_n = np.unique(border_nbrs, return_counts=True)
            order_n = np.argsort(counts_n)[::-1]

            for idx_n in order_n:
                cand_nbr = int(uniq_n[idx_n])
                if cand_nbr <= 0 or cand_nbr >= len(area_lut):
                    continue
                if int(area_lut[cand_nbr]) + cur_cnt > max_crown_px:
                    continue
                if use_shadow_barrier:
                    shared_rim = rim & (sub_lbl == cand_nbr)
                    crevice_frac = (
                        float(shadow_crevices[y0:y1, x0:x1][shared_rim].mean())
                        if np.any(shared_rim)
                        else 0.0
                    )
                    if crevice_frac > 0.75 and cur_cnt >= int(0.65 * coalesce_min_px):
                        continue
                sub_lbl[b_mask] = cand_nbr
                area_lut[cand_nbr] += cur_cnt
                area_lut[b_id] = 0
                if cur_slices[cand_nbr - 1] is not None:
                    cur_slices[cand_nbr - 1] = _union_slice_pair(cur_slices[cand_nbr - 1], sl)
                else:
                    cur_slices[cand_nbr - 1] = sl
                cur_slices[b_id - 1] = None
                merged_any = True
                break
        if not merged_any:
            break
    return labels


# ==============================================================================
# 4. Locked CanopyGIS v1.8 Engine (Bit-Identical to Verified Holdout Harness)
# ==============================================================================


def build_planar_euler1_v18(
    H: int,
    W: int,
    anchor_items: list[tuple[tuple[int, int, int, int], np.ndarray, float, dict]],
    secondary_items: list[tuple[tuple[int, int, int, int], np.ndarray, float, dict]],
    min_area_px: int,
    iom_reject_anchor: float = 0.50,
    iom_reject_secondary: float = 0.52,
) -> tuple[list[tuple[tuple[int, int, int, int], np.ndarray, int]], list[float], list[dict]]:
    """
    Locked v1.8 Planar Symmetric-Voronoi & Euler==1 Carve Engine:
      - CF4(a): Locked priority = score * (0.85 + 0.15 * min(1.0, pre_circ / 0.55))
      - CF4(c): Post-carve crescent/rind & interstitial sliver pruning while
        guaranteeing overlap_px == 0 & Euler == 1.
    """
    occ_id = np.zeros((H, W), dtype=np.int32)
    is_protected = [False]
    id_scores = [0.0]
    id_meta: list[dict] = [{}]

    def _priority_key(item):
        _, _, sc, meta = item
        circ = float(meta.get("pre_circ", 0.5))
        return float(sc) * (0.85 + 0.15 * min(1.0, circ / 0.55))

    def _place_candidates(cands, protect_flag: bool, iom_rej: float):
        cands_sorted = sorted(cands, key=_priority_key, reverse=True)
        for (y0, y1, x0, x1), crop_b, sc, meta in cands_sorted:
            raw_a = int(crop_b.sum())
            if raw_a < min_area_px:
                continue
            sub_occ = occ_id[y0:y1, x0:x1]
            occ_mask = sub_occ > 0
            ov = int(np.count_nonzero(crop_b & occ_mask))
            if float(ov) / float(raw_a) >= iom_rej:
                continue

            present_ids = [int(pid) for pid in np.unique(sub_occ[crop_b & occ_mask]) if pid > 0]
            if ov > 0 and not protect_flag:
                prot_mask = np.zeros_like(occ_mask)
                unprot_mask = np.zeros_like(occ_mask)
                for pid in present_ids:
                    if is_protected[pid]:
                        prot_mask |= sub_occ == pid
                    else:
                        unprot_mask |= sub_occ == pid

                avail = crop_b & (~prot_mask)
                if int(avail.sum()) < min_area_px:
                    continue
                ov_unprot = avail & unprot_mask
                if np.any(ov_unprot):
                    core_new = avail & (~unprot_mask)
                    core_old = unprot_mask & (~avail)
                    if np.any(core_new) and np.any(core_old):
                        d_new = cv2.distanceTransform((~core_new).astype(np.uint8), cv2.DIST_L2, 3)
                        d_old = cv2.distanceTransform((~core_old).astype(np.uint8), cv2.DIST_L2, 3)
                        win_new = ov_unprot & (d_new < d_old)
                        sub_occ[win_new] = 0
                    occ_mask = sub_occ > 0
                place_mask = crop_b & (~(sub_occ > 0))
            else:
                place_mask = crop_b & (~occ_mask)

            if int(place_mask.sum()) < min_area_px:
                continue

            cid = len(id_scores)
            occ_id[y0:y1, x0:x1][place_mask] = cid
            is_protected.append(protect_flag)
            id_scores.append(float(sc))
            m_rec = dict(meta)
            m_rec["raw_area"] = raw_a
            id_meta.append(m_rec)

    _place_candidates(anchor_items, protect_flag=True, iom_rej=iom_reject_anchor)
    _place_candidates(secondary_items, protect_flag=False, iom_rej=iom_reject_secondary)

    slices = find_objects(occ_id)
    for cid, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        sub_occ = occ_id[sl]
        sub_m = sub_occ == cid
        if not np.any(sub_m):
            continue

        n_c, lbl_c, stats_c, _ = cv2.connectedComponentsWithStats(sub_m.astype(np.uint8), connectivity=4)
        if n_c <= 1:
            sub_occ[sub_m] = 0
            continue
        best_c = 1 + int(np.argmax(stats_c[1:, cv2.CC_STAT_AREA]))
        if int(stats_c[best_c, cv2.CC_STAT_AREA]) < min_area_px:
            sub_occ[sub_m] = 0
            continue
        if n_c > 2:
            drop_islands = sub_m & (lbl_c != best_c)
            sub_occ[drop_islands] = 0
            sub_m = lbl_c == best_c

        filled = binary_fill_holes(sub_m)
        holes = filled & (~sub_m)
        if np.any(holes):
            free_holes = holes & (sub_occ == 0)
            if np.any(free_holes):
                sub_occ[free_holes] = cid
                sub_m = sub_occ == cid
                holes = binary_fill_holes(sub_m) & (~sub_m)
            if np.any(holes):
                pad_m = np.pad(sub_m.astype(np.uint8), 1, constant_values=0)
                inv_pad = 1 - pad_m
                n_bg, lbl_bg = cv2.connectedComponents(inv_pad, connectivity=4)
                outer_bg = (lbl_bg == lbl_bg[0, 0])[1:-1, 1:-1]
                for bg_id in range(1, n_bg):
                    if bg_id == int(lbl_bg[0, 0]):
                        continue
                    hole_mask = lbl_bg[1:-1, 1:-1] == bg_id
                    hy, hx = np.where(hole_mask)
                    if hy.size == 0:
                        continue
                    cy_h, cx_h = int(hy.mean()), int(hx.mean())
                    oy, ox = np.where(outer_bg)
                    if oy.size > 0:
                        d2 = (oy - cy_h) ** 2 + (ox - cx_h) ** 2
                        idx_near = int(np.argmin(d2))
                        ty, tx = int(oy[idx_near]), int(ox[idx_near])
                    else:
                        ty, tx = 0, cx_h
                    cut_canvas = np.zeros(sub_m.shape, dtype=np.uint8)
                    cv2.line(cut_canvas, (cx_h, cy_h), (tx, ty), 1, thickness=2)
                    sub_occ[(cut_canvas > 0) & (sub_occ == cid)] = 0
                sub_m = sub_occ == cid
                n_c2, lbl_c2, stats_c2, _ = cv2.connectedComponentsWithStats(sub_m.astype(np.uint8), connectivity=4)
                if n_c2 <= 1:
                    continue
                best_c2 = 1 + int(np.argmax(stats_c2[1:, cv2.CC_STAT_AREA]))
                sub_occ[sub_m & (lbl_c2 != best_c2)] = 0
                sub_m = sub_occ == cid
                if int(sub_m.sum()) < min_area_px:
                    sub_occ[sub_m] = 0
                    continue

        ay = sub_m.any(axis=1)
        ax = sub_m.any(axis=0)
        ry0, ry1 = int(np.argmax(ay)), int(len(ay) - np.argmax(ay[::-1]))
        rx0, rx1 = int(np.argmax(ax)), int(len(ax) - np.argmax(ax[::-1]))
        bh, bw = ry1 - ry0, rx1 - rx0
        cur_a = int(sub_m.sum())
        if bh >= 4 and bw >= 4 and float(cur_a) / float(bh * bw) > 0.915:
            yy_loc, xx_loc = np.ogrid[0:bh, 0:bw]
            cy_loc, cx_loc = 0.5 * (bh - 1), 0.5 * (bw - 1)
            ell = ((xx_loc - cx_loc) / max(0.5 * bw, 1.0)) ** 2 + ((yy_loc - cy_loc) / max(0.5 * bh, 1.0)) ** 2 <= 1.04
            crop_view = sub_occ[ry0:ry1, rx0:rx1]
            trim = (crop_view == cid) & (~ell)
            if cur_a - int(np.count_nonzero(trim)) >= min_area_px:
                crop_view[trim] = 0
                sub_m3 = sub_occ == cid
                n_c3, lbl_c3, st_c3, _ = cv2.connectedComponentsWithStats(sub_m3.astype(np.uint8), connectivity=4)
                if n_c3 > 2:
                    b3 = 1 + int(np.argmax(st_c3[1:, cv2.CC_STAT_AREA]))
                    sub_occ[sub_m3 & (lbl_c3 != b3)] = 0

    final_slices = find_objects(occ_id)
    pred_sparse = []
    pred_scores = []
    pred_metas = []

    for cid, sl in enumerate(final_slices, start=1):
        if sl is None:
            continue
        sub_m = occ_id[sl] == cid
        area = int(sub_m.sum())
        if area < min_area_px:
            continue
        n8, lbl8, st8, _ = cv2.connectedComponentsWithStats(sub_m.astype(np.uint8), connectivity=8)
        if n8 > 2:
            b8 = 1 + int(np.argmax(st8[1:, cv2.CC_STAT_AREA]))
            sub_m = lbl8 == b8
            area = int(sub_m.sum())
            if area < min_area_px:
                continue
        ay = sub_m.any(axis=1)
        ax = sub_m.any(axis=0)
        ry0, ry1 = int(np.argmax(ay)), int(len(ay) - np.argmax(ay[::-1]))
        rx0, rx1 = int(np.argmax(ax)), int(len(ax) - np.argmax(ax[::-1]))
        crop = sub_m[ry0:ry1, rx0:rx1].copy()
        post_area = int(crop.sum())
        meta = dict(id_meta[cid])
        raw_area = max(1, int(meta.get("raw_area", post_area)))
        keep_ratio = float(post_area) / float(raw_area)

        p_circ, p_sol, p_asp, _ = compute_mask_shape_metrics(crop)
        drop_carved_rind = (
            (keep_ratio < 0.45)
            or (keep_ratio < 0.68 and (p_circ < 0.28 or p_sol < 0.68 or (p_asp > 2.20 and p_sol < 0.78)))
            or (keep_ratio < 0.84 and (p_circ < 0.19 or p_sol < 0.60 or (p_asp > 2.45 and p_sol < 0.76)))
        )
        drop_extreme_sliver = post_area >= 900 and (
            (p_circ < 0.14 and p_sol < 0.62) or p_sol < 0.50 or (p_asp > 2.80 and p_sol < 0.74)
        )
        if drop_carved_rind or drop_extreme_sliver:
            continue

        y0, x0 = sl[0].start + ry0, sl[1].start + rx0
        y1, x1 = sl[0].start + ry1, sl[1].start + rx1
        meta["cid"] = cid
        meta["keep_ratio"] = round(keep_ratio, 4)
        meta["post_circ"] = round(p_circ, 4)
        meta["post_sol"] = round(p_sol, 4)
        pred_sparse.append(((y0, y1, x0, x1), crop, post_area))
        pred_scores.append(id_scores[cid])
        pred_metas.append(meta)

    return pred_sparse, pred_scores, pred_metas


def run_canopygis_v18_tile(
    img_rgb: np.ndarray,
    precomp: dict[str, Any],
    gsd_m: float | None,
) -> tuple[list[tuple[tuple[int, int, int, int], np.ndarray, int]], list[float], list[dict], dict]:
    """
    Pure Zero-Shot CanopyGIS v1.8 (CF2 + CF3-Scoped + CF4):
      - Zero green-only masking (preserves dry-season olive/tan/autumn/shaded crowns).
      - Zero per-tile or biome switches.
      - Frame-invariant SI Emergent Shield (CF3-Scoped: triggered strictly when
        clean_a > max_single_crown_px and _has_bisecting_shadow_crevice == False,
        capped at min(0.16*H*W, pi*(20/gsd_m)^2)).
      - Locked CF4 roundness priority: score * (0.85 + 0.15 * min(1.0, pre_circ / 0.55)).
    """
    H, W = precomp["H"], precomp["W"]
    lab_l, lab_a = precomp["lab_l"], precomp["lab_a"]
    lap_full = precomp["lap_full"]
    rgb_f, exg_full = precomp["rgb_f"], precomp["exg_full"]
    nonblack_mask = precomp["nonblack_mask"]
    valid_img_px = precomp["valid_img_px"]
    scene_l_p12 = precomp["scene_l_p12"]
    scene_l_p25 = precomp["scene_l_p25"]
    scene_l_p50 = precomp["scene_l_p50"]
    groups = precomp["groups"]
    adaptive_wins = precomp["adaptive_wins"]

    ff_tree = groups["ff_tree"]
    ff_ind = groups["ff_ind"]
    ff_can = groups["ff_can"]
    ff_enh = groups["ff_enh"]

    # 1. CF2(a): All-Score Cast-Umbra + Non-Convex Dome Veto on Full-Frame & Tiled SAM 3 Proposals
    ff_crown_pool = []
    for grp_name, lst, w_sc in [("ff_tree", ff_tree, 1.00), ("ff_ind", ff_ind, 0.98), ("ff_enh", ff_enh, 0.93)]:
        for idx_p, (bbox, crop, sc, seam, area) in enumerate(lst):
            sc_w = sc * w_sc
            if sc_w >= 0.16 and is_valid_foliage_patch_cf(
                img_rgb,
                lab_l,
                lap_full,
                None,
                scene_l_p12,
                scene_l_p25,
                bbox,
                crop,
                sc_w,
                cf_green_only=False,
                cf_umbra_veto=True,
            ):
                ff_crown_pool.append(
                    (bbox, crop, sc_w, seam, area, {"src_group": grp_name, "src_idx": idx_p, "raw_sc": sc})
                )

    ff_dedup_meta = strtree_dedup_with_meta(ff_crown_pool, iou_thr=0.52, iom_thr=0.72)
    ff_dedup_pure = [(c[0], c[1], c[2], c[3], c[4]) for c in ff_dedup_meta]

    ti_pool = []
    for grp_name, lst, w_sc in [("ti_tree", groups["ti_tree"], 1.00), ("ti_ind", groups["ti_ind"], 0.97)]:
        for idx_p, (bbox, crop, sc, seam, area) in enumerate(lst):
            sc_w = sc * w_sc
            if sc_w >= 0.42 and is_valid_foliage_patch_cf(
                img_rgb,
                lab_l,
                lap_full,
                None,
                scene_l_p12,
                scene_l_p25,
                bbox,
                crop,
                sc_w,
                cf_green_only=False,
                cf_umbra_veto=True,
            ):
                ti_pool.append(
                    (bbox, crop, sc_w, seam, area, {"src_group": grp_name, "src_idx": idx_p, "raw_sc": sc})
                )
    ti_dedup_meta = strtree_dedup_with_meta(ti_pool, iou_thr=0.50, iom_thr=0.72)
    ti_dedup_pure = [(c[0], c[1], c[2], c[3], c[4]) for c in ti_dedup_meta]

    # 2. Continuous SI Crown Scale & CF3-Scoped Frame-Invariant Emergent Allometry Ceiling
    proto_pool = ff_dedup_pure + [c for c in ti_dedup_pure if c[2] >= 0.52]
    r_min_px, r_med_px, r_max_px = estimate_physical_crown_radius_px(H, W, gsd_m, proto_pool)
    med_crown_area = math.pi * (r_med_px ** 2)
    min_crown_px = max(22, int(round(math.pi * (r_min_px ** 2))))

    max_single_crown_px = min(int(0.095 * H * W), int(round(math.pi * (r_max_px ** 2))))
    if gsd_m is not None and gsd_m > 0:
        r_emerg_px = min(20.0 / gsd_m, math.sqrt(0.16 * H * W / math.pi))
        emergent_max_px = min(
            int(0.16 * H * W),
            int(round(math.pi * (max(r_max_px, r_emerg_px) ** 2))),
        )
    else:
        emergent_max_px = int(0.16 * H * W)

    # In-Situ Tree Foliage Reference Mask
    foliage_ref_mask = np.zeros((H, W), dtype=bool)
    n_high_ff_tree = 0
    for (y0, y1, x0, x1), crop, sc, _, area in ff_tree:
        ref_ceil = emergent_max_px if sc >= 0.68 else max_single_crown_px
        if sc >= 0.52 and min_crown_px <= area <= ref_ceil:
            foliage_ref_mask[y0:y1, x0:x1] |= crop
            n_high_ff_tree += 1
    use_ref = foliage_ref_mask if n_high_ff_tree >= 6 else None
    if use_ref is not None:
        ref_a_hi = float(np.percentile(lab_a[use_ref], 96)) + 4.5
        ref_exg_lo = float(np.percentile(exg_full[use_ref], 4)) - 6.0
        ref_l_hi = float(np.percentile(lab_l[use_ref], 98)) + 14.0
        bare_px_est = nonblack_mask & (
            ((lab_a > ref_a_hi) & (exg_full < ref_exg_lo))
            | ((lab_l > ref_l_hi) & (exg_full < ref_exg_lo))
            | ((rgb_f[:, :, 0] > rgb_f[:, :, 1] + 10.0) & (exg_full < -8.0))
        )
    else:
        ref_a_hi = 136.0
        ref_exg_lo = -6.0
        bare_px_est = nonblack_mask & (rgb_f[:, :, 0] > rgb_f[:, :, 1] + 8.0) & (exg_full < -6.0)
    scene_bare_fraction = float(np.count_nonzero(bare_px_est)) / float(valid_img_px)

    # 3. Neural Canopy Support Mask
    canopy_support = np.zeros((H, W), dtype=bool)
    for (y0, y1, x0, x1), crop, sc, _, _ in ff_dedup_pure:
        if sc >= 0.42:
            canopy_support[y0:y1, x0:x1] |= crop

    min_sheet_area = max(int(0.24 * valid_img_px), int(round(8.0 * med_crown_area)))
    for (y0, y1, x0, x1), crop, sc, _, area in ff_can:
        if sc >= 0.48 and area >= min_sheet_area:
            canopy_support[y0:y1, x0:x1] |= crop

    for (y0, y1, x0, x1), crop, sc, _, area in groups["ti_can"]:
        if sc >= 0.56 and area >= max(int(0.10 * valid_img_px), int(round(6.0 * med_crown_area))):
            canopy_support[y0:y1, x0:x1] |= crop

    if n_high_ff_tree < 6 and float(np.count_nonzero(canopy_support)) / float(valid_img_px) < 0.25:
        for (y0, y1, x0, x1), crop, sc, _, _ in ff_dedup_pure:
            if sc >= 0.16:
                canopy_support[y0:y1, x0:x1] |= crop
        for (y0, y1, x0, x1), crop, sc, _, _ in ti_dedup_pure:
            if sc >= 0.44:
                canopy_support[y0:y1, x0:x1] |= crop

    # 4. Parent-vs-Children Clump Decomposition with CF3-Scoped Emergent Shield
    min_child_area = max(min_crown_px, int(round(0.28 * med_crown_area)))
    ff_child_pool = [c for c in ff_dedup_pure if c[2] >= 0.48 and c[3] == 0 and c[4] >= min_child_area]
    all_child_pool = [
        c for c in (ff_dedup_pure + ti_dedup_pure)
        if c[2] >= 0.48 and c[3] == 0 and c[4] >= min_child_area
    ]

    def _count_disjoint_children(
        pool,
        py0: int,
        py1: int,
        px0: int,
        px1: int,
        pcrop: np.ndarray,
        parea: int,
    ) -> int:
        if parea < int(round(1.65 * med_crown_area)):
            return 0
        accepted_children = []
        for (cy0, cy1, cx0, cx1), ccrop, _, _, carea in pool:
            if carea > int(0.68 * parea) or cy1 <= py0 or cy0 >= py1 or cx1 <= px0 or cx0 >= px1:
                continue
            oy0, oy1 = max(py0, cy0), min(py1, cy1)
            ox0, ox1 = max(px0, cx0), min(px1, cx1)
            inter = int(
                np.count_nonzero(
                    pcrop[oy0 - py0:oy1 - py0, ox0 - px0:ox1 - px0]
                    & ccrop[oy0 - cy0:oy1 - cy0, ox0 - cx0:ox1 - cx0]
                )
            )
            if float(inter) / float(carea) < 0.76:
                continue
            sol_c, asp_c, _, _ = crop_solidity_and_clean4(ccrop)
            if sol_c < 0.78 or asp_c > 1.75:
                continue
            is_sib_dup = False
            for (sy0, sy1, sx0, sx1), scrop, sarea in accepted_children:
                iy0, iy1 = max(cy0, sy0), min(cy1, sy1)
                ix0, ix1 = max(cx0, sx0), min(cx1, sx1)
                if iy1 > iy0 and ix1 > ix0:
                    sib_ov = int(
                        np.count_nonzero(
                            ccrop[iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0]
                            & scrop[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0]
                        )
                    )
                    if float(sib_ov) / float(min(carea, sarea)) >= 0.22:
                        is_sib_dup = True
                        break
            if not is_sib_dup:
                accepted_children.append(((cy0, cy1, cx0, cx1), ccrop, carea))
                if len(accepted_children) >= 2:
                    return 2
        return len(accepted_children)

    def _has_bisecting_shadow_crevice(
        py0: int,
        py1: int,
        px0: int,
        px1: int,
        pcrop: np.ndarray,
        parea: int,
    ) -> bool:
        crev = lab_l[py0:py1, px0:px1] < max(20.0, scene_l_p12 * 1.02)
        crev_dil = cv2.dilate(crev.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
        split_m = pcrop & (~crev_dil)
        n_l, _, st_l, _ = cv2.connectedComponentsWithStats(split_m.astype(np.uint8), connectivity=4)
        if n_l <= 2:
            return False
        major = sum(1 for k in range(1, n_l) if int(st_l[k, cv2.CC_STAT_AREA]) >= int(0.20 * parea))
        return major >= 2

    anchor_crowns: list[tuple[tuple[int, int, int, int], np.ndarray, float, dict]] = []
    secondary_crowns: list[tuple[tuple[int, int, int, int], np.ndarray, float, dict]] = []
    claimed_pass1 = np.zeros((H, W), dtype=bool)
    n_superclumps_routed = 0
    deferred_mid_ff = []

    ff_iter_items = []
    for (y0, y1, x0, x1), crop, sc, seam, raw_a, pmeta in ff_dedup_meta:
        if sc < 0.25:
            continue
        if use_ref is not None and sc < 0.50:
            m_a = float(lab_a[y0:y1, x0:x1][crop].mean())
            m_exg = float(exg_full[y0:y1, x0:x1][crop].mean())
            if m_a > ref_a_hi and m_exg < ref_exg_lo:
                continue
        sol, aspect, clean_crop, clean_a = crop_solidity_and_clean4(crop)
        if clean_a < min_crown_px:
            continue
        circ_c, _, _, _ = compute_mask_shape_metrics(clean_crop)
        ff_iter_items.append(((y0, y1, x0, x1), crop, clean_crop, clean_a, raw_a, sol, aspect, circ_c, sc, seam, pmeta))

    # Locked CF4(a) Priority: score * (0.85 + 0.15 * min(1.0, pre_circ / 0.55))
    ff_iter_items.sort(
        key=lambda it: (1 if it[8] >= 0.44 else 0, it[8] * (0.85 + 0.15 * min(1.0, it[7] / 0.55))),
        reverse=True,
    )

    for (y0, y1, x0, x1), crop, clean_crop, clean_a, raw_a, sol, aspect, circ_c, sc, seam, pmeta in ff_iter_items:
        is_emergent_shielded = (
            clean_a > max_single_crown_px
            and clean_a <= emergent_max_px
            and sc >= 0.68
            and sol >= 0.74
            and circ_c >= 0.28
            and not _has_bisecting_shadow_crevice(y0, y1, x0, x1, clean_crop, clean_a)
        )
        active_child_pool = ff_child_pool if is_emergent_shielded else all_child_pool
        n_ch = _count_disjoint_children(active_child_pool, y0, y1, x0, x1, clean_crop, clean_a)

        trig_area = clean_a > (emergent_max_px if is_emergent_shielded else max_single_crown_px)
        trig_lowsol = clean_a > 4.50 * med_crown_area and sol < 0.52
        trig_children = n_ch >= 2
        if trig_area or trig_lowsol or trig_children:
            canopy_support[y0:y1, x0:x1] |= crop
            n_superclumps_routed += 1
            continue

        if sc >= 0.44:
            if sol < 0.64 or circ_c < 0.16 or (aspect > 2.50 and sol < 0.76):
                if clean_a >= int(round(1.6 * med_crown_area)):
                    canopy_support[y0:y1, x0:x1] |= crop
                continue

            ov_p1 = int(np.count_nonzero(clean_crop & claimed_pass1[y0:y1, x0:x1]))
            if float(ov_p1) / float(clean_a) >= 0.48:
                continue
            anchor_crowns.append((
                (y0, y1, x0, x1),
                clean_crop,
                sc,
                {
                    "tier": "Tier1A_ff",
                    "src_group": pmeta["src_group"],
                    "seam": seam,
                    "sam_raw_area": raw_a,
                    "pre_circ": round(circ_c, 4),
                    "pre_sol": round(sol, 4),
                },
            ))
            claimed_pass1[y0:y1, x0:x1] |= clean_crop
            canopy_support[y0:y1, x0:x1] |= clean_crop
        else:
            deferred_mid_ff.append(
                ((y0, y1, x0, x1), crop, clean_crop, clean_a, raw_a, sol, aspect, circ_c, sc, seam, pmeta)
            )

    # Tier 1B — High-Res Tiled SAM 3 Gap-Filling with >= 2 Inner-Seam Corner Veto (CF4)
    if ti_dedup_meta:
        min_ti_area = max(min_crown_px, int(round(0.24 * med_crown_area)))
        max_ti_area = max_single_crown_px
        req_base = 0.54 if scene_bare_fraction >= 0.25 else 0.48

        ti_iter = []
        for (y0, y1, x0, x1), crop, sc, seam, raw_a, pmeta in ti_dedup_meta:
            req_sc = max(req_base, 0.58) if seam == 1 else req_base
            if sc < req_sc:
                continue
            if use_ref is not None:
                m_a = float(lab_a[y0:y1, x0:x1][crop].mean())
                m_exg = float(exg_full[y0:y1, x0:x1][crop].mean())
                if m_a > ref_a_hi and m_exg < ref_exg_lo:
                    continue
            sol, aspect, clean_crop, clean_a = crop_solidity_and_clean4(crop)
            if clean_a < min_ti_area:
                continue
            circ_c, _, _, _ = compute_mask_shape_metrics(clean_crop)
            ti_iter.append(((y0, y1, x0, x1), crop, clean_crop, clean_a, raw_a, sol, aspect, circ_c, sc, seam, pmeta))

        ti_iter.sort(key=lambda it: it[8] * (0.85 + 0.15 * min(1.0, it[7] / 0.55)), reverse=True)

        for (y0, y1, x0, x1), crop, clean_crop, clean_a, raw_a, sol, aspect, circ_c, sc, seam, pmeta in ti_iter:
            if clean_a > max_ti_area or (clean_a > 4.20 * med_crown_area and sol < 0.55):
                if sc >= 0.54:
                    canopy_support[y0:y1, x0:x1] |= crop
                    n_superclumps_routed += 1
                continue

            if sol < 0.74 or aspect > 1.90 or circ_c < 0.28:
                continue

            n_seam_edges, seam_desc = count_inner_window_seam_touches((y0, y1, x0, x1), adaptive_wins, H, W)
            if seam == 1 and (n_seam_edges >= 2 or circ_c < 0.48 or sol < 0.85):
                continue

            ov_p1 = int(np.count_nonzero(clean_crop & claimed_pass1[y0:y1, x0:x1]))
            if float(ov_p1) / float(clean_a) >= 0.32:
                continue

            ti_meta = {
                "tier": "Tier1B_ti",
                "src_group": pmeta["src_group"],
                "seam": seam,
                "inner_seam_touches": seam_desc,
                "sam_raw_area": raw_a,
                "pre_circ": round(circ_c, 4),
                "pre_sol": round(sol, 4),
            }
            if seam == 1:
                secondary_crowns.append(((y0, y1, x0, x1), clean_crop, sc * 0.94, ti_meta))
            else:
                anchor_crowns.append(((y0, y1, x0, x1), clean_crop, sc * 0.96, ti_meta))
            claimed_pass1[y0:y1, x0:x1] |= clean_crop
            canopy_support[y0:y1, x0:x1] |= clean_crop

    # Tier 1C — Mid-Confidence Full-Frame Secondary Proposals
    for (y0, y1, x0, x1), crop, clean_crop, clean_a, raw_a, sol, aspect, circ_c, sc, seam, pmeta in deferred_mid_ff:
        if sol < 0.64 or circ_c < 0.18 or (aspect > 2.35 and sol < 0.76):
            continue
        sup_ov = int(np.count_nonzero(crop & canopy_support[y0:y1, x0:x1]))
        if float(sup_ov) / float(raw_a) < 0.60:
            continue
        ov_p1 = int(np.count_nonzero(clean_crop & claimed_pass1[y0:y1, x0:x1]))
        if float(ov_p1) / float(clean_a) >= 0.36:
            continue
        secondary_crowns.append((
            (y0, y1, x0, x1),
            clean_crop,
            sc,
            {
                "tier": "Tier1C_mid_ff",
                "src_group": pmeta["src_group"],
                "seam": seam,
                "sam_raw_area": raw_a,
                "pre_circ": round(circ_c, 4),
                "pre_sol": round(sol, 4),
            },
        ))
        claimed_pass1[y0:y1, x0:x1] |= clean_crop
        canopy_support[y0:y1, x0:x1] |= clean_crop

    # 5. SI Shadow-Crevice Topography + CF2(b) Separable O(HW) MORPH_RECT 3D Micro-Relief Gate
    active_canopy, shadow_crevices, crevice_free, elevation_map = compute_si_shadow_crevice_topography(
        img_rgb, canopy_support, r_med_px, foliage_ref_mask=use_ref
    )

    k_rel = max(9, min(75, int(round(0.55 * r_med_px)) | 1))
    mean_l_loc = cv2.blur(lab_l, (k_rel, k_rel))
    sq_l_loc = cv2.blur(lab_l * lab_l, (k_rel, k_rel))
    loc_std_l = np.sqrt(np.maximum(0.0, sq_l_loc - mean_l_loc * mean_l_loc))
    loc_lap = cv2.blur(lap_full, (k_rel, k_rel))
    se_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (k_rel, k_rel))
    loc_dyn_l = cv2.dilate(lab_l, se_rect) - cv2.erode(lab_l, se_rect)

    ref_3d = use_ref if (use_ref is not None and int(np.count_nonzero(use_ref)) >= 400) else (
        claimed_pass1 if int(np.count_nonzero(claimed_pass1)) >= 400 else None
    )
    if ref_3d is not None:
        std_floor = min(22.0, max(9.0, 0.65 * float(np.percentile(loc_std_l[ref_3d], 15))))
        lap_floor = min(15.5, max(7.0, 0.62 * float(np.percentile(loc_lap[ref_3d], 15))))
        dyn_floor = min(42.0, max(20.0, 0.65 * float(np.percentile(loc_dyn_l[ref_3d], 15))))
        l_umbra_floor = max(46.0, min(scene_l_p25 * 0.98, float(np.percentile(lab_l[ref_3d], 8)) * 0.82))
    else:
        std_floor, lap_floor, dyn_floor = 9.5, 7.0, 22.0
        l_umbra_floor = max(24.0, scene_l_p12 * 0.95)

    woody_3d = (
        (((loc_std_l >= std_floor) & (loc_lap >= lap_floor)) | (loc_dyn_l >= dyn_floor * 1.35))
        & (mean_l_loc >= l_umbra_floor)
    )
    k_bridge = max(5, min(35, int(round(0.24 * r_med_px)) | 1))
    se_br = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bridge, k_bridge))
    woody_3d_u8 = cv2.morphologyEx(woody_3d.astype(np.uint8), cv2.MORPH_CLOSE, se_br, iterations=1)
    active_canopy = (active_canopy & (woody_3d_u8 > 0)) | claimed_pass1
    crevice_free = crevice_free & active_canopy

    # 6. Tier 3 — Component-Wise Shadow-Crevice Geodesic Watershed
    collar_px = max(3, int(round(0.26 * r_med_px)))
    k_collar = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * collar_px + 1, 2 * collar_px + 1))
    claimed_buffered = cv2.dilate(claimed_pass1.astype(np.uint8), k_collar, iterations=1) > 0

    unclaimed_raw = active_canopy & (~claimed_pass1)
    core_open_r = max(2, int(round(0.24 * r_med_px)))
    k_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * core_open_r + 1, 2 * core_open_r + 1))
    unclaimed_cores = cv2.morphologyEx(unclaimed_raw.astype(np.uint8), cv2.MORPH_OPEN, k_core, iterations=1) > 0

    min_clump_px = max(int(1.6 * min_crown_px), int(round(0.62 * med_crown_area)))
    n_cc, cc_lbl, cc_stats, _ = cv2.connectedComponentsWithStats(unclaimed_raw.astype(np.uint8), connectivity=8)
    valid_comp_ids = [
        cid for cid in range(1, n_cc)
        if int(cc_stats[cid, cv2.CC_STAT_AREA]) >= min_clump_px
        and np.any(
            unclaimed_cores[
                cc_stats[cid, cv2.CC_STAT_TOP]:cc_stats[cid, cv2.CC_STAT_TOP] + cc_stats[cid, cv2.CC_STAT_HEIGHT],
                cc_stats[cid, cv2.CC_STAT_LEFT]:cc_stats[cid, cv2.CC_STAT_LEFT] + cc_stats[cid, cv2.CC_STAT_WIDTH],
            ]
            & (
                cc_lbl[
                    cc_stats[cid, cv2.CC_STAT_TOP]:cc_stats[cid, cv2.CC_STAT_TOP] + cc_stats[cid, cv2.CC_STAT_HEIGHT],
                    cc_stats[cid, cv2.CC_STAT_LEFT]:cc_stats[cid, cv2.CC_STAT_LEFT] + cc_stats[cid, cv2.CC_STAT_WIDTH],
                ]
                == cid
            )
        )
    ]

    ws_count = 0
    if valid_comp_ids:
        ws_target_mask = np.isin(cc_lbl, valid_comp_ids)
        seed_search_mask = ws_target_mask & crevice_free & (~claimed_buffered)
        seed_min_dist = max(9, int(round(0.92 * r_med_px)))

        seed_points: list[tuple[int, int, float]] = []
        occ_seed = claimed_buffered.astype(np.uint8) * 255

        rem_search = seed_search_mask & (occ_seed == 0)
        if np.any(rem_search):
            peaks = peak_local_max(
                elevation_map,
                min_distance=seed_min_dist,
                threshold_rel=0.20,
                labels=rem_search.astype(np.uint8),
                exclude_border=False,
            )
            for py, px in peaks:
                py_i, px_i = int(py), int(px)
                if occ_seed[py_i, px_i] > 0:
                    continue
                seed_points.append((px_i, py_i, 0.56))
                cv2.circle(occ_seed, (px_i, py_i), max(8, int(round(0.88 * r_med_px))), 255, -1)

        for cid in valid_comp_ids:
            cx0 = int(cc_stats[cid, cv2.CC_STAT_LEFT])
            cy0 = int(cc_stats[cid, cv2.CC_STAT_TOP])
            cw = int(cc_stats[cid, cv2.CC_STAT_WIDTH])
            ch = int(cc_stats[cid, cv2.CC_STAT_HEIGHT])
            sub_cc = cc_lbl[cy0:cy0 + ch, cx0:cx0 + cw] == cid
            has_s = any(
                (cx0 <= sx < cx0 + cw) and (cy0 <= sy < cy0 + ch) and sub_cc[sy - cy0, sx - cx0]
                for (sx, sy, _) in seed_points
            )
            if not has_s:
                sub_el = elevation_map[cy0:cy0 + ch, cx0:cx0 + cw].copy()
                cand_m = sub_cc & (~claimed_buffered[cy0:cy0 + ch, cx0:cx0 + cw])
                if np.any(cand_m):
                    sub_el[~cand_m] = -1e9
                    ly, lx = np.unravel_index(int(np.argmax(sub_el)), sub_el.shape)
                    seed_points.append((cx0 + int(lx), cy0 + int(ly), 0.50))

        if seed_points:
            markers = np.zeros((H, W), dtype=np.int32)
            seed_rad = max(2, int(round(0.16 * r_med_px)))
            for idx_s, (sx, sy, _) in enumerate(seed_points, start=1):
                cv2.circle(markers, (sx, sy), seed_rad, idx_s, -1)

            inv_seeds = np.ones((H, W), dtype=np.uint8)
            for sx, sy, _ in seed_points:
                inv_seeds[sy, sx] = 0
            dist_from_seed = cv2.distanceTransform(inv_seeds, cv2.DIST_L2, 5)
            radial_env = dist_from_seed <= (1.72 * r_med_px)
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cont_canopy = cv2.morphologyEx(ws_target_mask.astype(np.uint8), cv2.MORPH_CLOSE, k_close) > 0
            ws_domain = (cont_canopy & (~claimed_pass1) & radial_env) | (markers > 0)

            ws_labels = watershed(-elevation_map, markers, mask=ws_domain, compactness=0.012)
            coalesce_min_px = max(min_crown_px, int(round(0.40 * med_crown_area)))
            max_ws_crown_px = int(round(2.20 * med_crown_area))
            ws_labels = _rag_coalesce_1d_lut(
                ws_labels, shadow_crevices, coalesce_min_px, max_ws_crown_px, use_shadow_barrier=True
            )

            min_ws_basin_px = max(min_crown_px, int(round(0.30 * med_crown_area)))
            ws_slices = find_objects(ws_labels)
            for b_id, sl in enumerate(ws_slices, start=1):
                if sl is None:
                    continue
                sub_b = ws_labels[sl] == b_id
                if int(sub_b.sum()) < min_ws_basin_px:
                    continue
                filled_b = binary_fill_holes(sub_b)
                if float(np.count_nonzero(filled_b & claimed_pass1[sl])) / float(filled_b.sum()) > 0.16:
                    continue
                clean_b = filled_b & (~claimed_pass1[sl])
                ca_b = int(clean_b.sum())
                if ca_b < min_ws_basin_px:
                    continue
                y0, y1 = sl[0].start, sl[0].stop
                x0, x1 = sl[1].start, sl[1].stop

                b_l = lab_l[y0:y1, x0:x1][clean_b]
                b_lap = lap_full[y0:y1, x0:x1][clean_b]
                b_dome = compute_radial_dome_gradient(lab_l[y0:y1, x0:x1], clean_b)
                b_lm, b_lstd, b_lapm = float(b_l.mean()), float(b_l.std()), float(b_lap.mean())
                is_umbra_basin = b_lm < max(58.0, scene_l_p25 * 1.05) and b_dome <= 2.0
                is_flat_clearing_basin = (
                    (b_lapm < lap_floor * 1.08 and b_lstd < max(26.0, std_floor * 1.35))
                    or (b_dome <= -2.0 and b_lapm < lap_floor * 1.25)
                ) and (b_lm < scene_l_p50 * 0.96)
                if is_umbra_basin or is_flat_clearing_basin:
                    continue

                b_circ, b_sol, _, _ = compute_mask_shape_metrics(clean_b)
                sc_b = seed_points[b_id - 1][2] if (b_id - 1) < len(seed_points) else 0.54
                secondary_crowns.append((
                    (y0, y1, x0, x1),
                    clean_b,
                    sc_b,
                    {
                        "tier": "Tier3_ws",
                        "src_group": "watershed",
                        "seam": 0,
                        "sam_raw_area": int(filled_b.sum()),
                        "pre_circ": round(b_circ, 4),
                        "pre_sol": round(b_sol, 4),
                    },
                ))
                ws_count += 1

    # 7. Tier 4 — Global Planar & Euler==1 Closure (CF4)
    pred_sparse, pred_scores, pred_metas = build_planar_euler1_v18(
        H, W, anchor_crowns, secondary_crowns, min_area_px=min_crown_px
    )
    diag = {
        "r_med_px": round(r_med_px, 2),
        "max_single_crown_px": max_single_crown_px,
        "emergent_max_px": emergent_max_px,
        "n_anchors": len(anchor_crowns),
        "n_superclumps_routed": n_superclumps_routed,
        "n_ws_basins": ws_count,
        "active_canopy_bool": active_canopy,
    }
    return pred_sparse, pred_scores, pred_metas, diag


# ==============================================================================
# 5. Decision 2A Post-Carve Tier-3 Watershed Score Remap & Planar Resize Helper
# ==============================================================================


def remap_post_carve_scores(
    pred_scores: list[float],
    pred_metas: list[dict],
) -> list[float]:
    """
    Post-carve Tier-3 watershed score remap (Decision 2A).

    Called strictly AFTER build_planar_euler1_v18 returns (never mutating carving
    priority before planar closure) so DeltaMask == 0 px and DeltaF1 == 0.0000 while
    ranking Tier-1/Tier-2 SAM 3 crowns ahead of Tier-3 watershed basins for AP40/AP60.
    """
    remapped: list[float] = []
    for sc, meta in zip(pred_scores, pred_metas):
        if meta.get("tier") == "Tier3_ws" or meta.get("src_group") == "watershed":
            sol = float(meta.get("pre_sol", 0.70))
            circ = float(meta.get("pre_circ", 0.45))
            new_sc = 0.15 + 0.22 * (
                0.5 * min(1.0, max(0.0, sol))
                + 0.5 * min(1.0, max(0.0, circ / 0.65))
            )
            remapped.append(float(new_sc))
        else:
            remapped.append(float(sc))
    return remapped


def _resize_int32_label_nearest(occ_id: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    """Nearest-neighbor resize of an int32 planar label map with OpenCV dtype safety."""
    try:
        return cv2.resize(occ_id, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
    except cv2.error:
        max_id = int(occ_id.max()) if occ_id.size else 0
        if max_id <= 65535:
            return cv2.resize(
                occ_id.astype(np.uint16), (dst_w, dst_h), interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)
        return cv2.resize(
            occ_id.astype(np.float64), (dst_w, dst_h), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)


def resize_planar_predictions(
    pred_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
    pred_scores: list[float],
    pred_metas: list[dict],
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
    min_area_px: int = 16,
) -> tuple[list[tuple[tuple[int, int, int, int], np.ndarray, int]], list[float], list[dict]]:
    """
    Topology-preserving planar resize of sparse crown predictions from src_hw to dst_hw.

    Paints an integer occ_id (int32) label canvas, resizes via cv2.INTER_NEAREST to dst_hw,
    and re-runs 4-connected largest component + binary_fill_holes (+ radial slit cut for
    enclosed foreign crowns & corner chamfer) so the resized partition strictly preserves
    overlap_px == 0 and Euler == 1.
    """
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])

    if not pred_sparse:
        return [], [], []
    if (src_h, src_w) == (dst_h, dst_w):
        return (
            [(bbox, crop.copy(), int(area)) for (bbox, crop, area) in pred_sparse],
            [float(s) for s in pred_scores],
            [dict(m) for m in pred_metas],
        )

    occ_id = np.zeros((src_h, src_w), dtype=np.int32)
    for cid, ((y0, y1, x0, x1), crop_b, _area) in enumerate(pred_sparse, start=1):
        occ_id[y0:y1, x0:x1][crop_b] = cid

    occ_rs = _resize_int32_label_nearest(occ_id, dst_h, dst_w)

    # Topological closure passes on resized planar label map (guarantees overlap==0 & Euler==1)
    for _topo_pass in range(2):
        modified = False
        slices = find_objects(occ_rs)
        for cid, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            sub_occ = occ_rs[sl]
            sub_m = sub_occ == cid
            if not np.any(sub_m):
                continue

            # 1. Keep strictly largest 4-connected component
            n_c, lbl_c, stats_c, _ = cv2.connectedComponentsWithStats(sub_m.astype(np.uint8), connectivity=4)
            if n_c <= 1:
                sub_occ[sub_m] = 0
                modified = True
                continue
            best_c = 1 + int(np.argmax(stats_c[1:, cv2.CC_STAT_AREA]))
            if int(stats_c[best_c, cv2.CC_STAT_AREA]) < min_area_px:
                sub_occ[sub_m] = 0
                modified = True
                continue
            if n_c > 2:
                drop_islands = sub_m & (lbl_c != best_c)
                sub_occ[drop_islands] = 0
                sub_m = lbl_c == best_c
                modified = True

            # 2. Fill unclaimed holes; if an enclosed foreign crown remains, cut a 2px radial slit
            filled = binary_fill_holes(sub_m)
            holes = filled & (~sub_m)
            if np.any(holes):
                free_holes = holes & (sub_occ == 0)
                if np.any(free_holes):
                    sub_occ[free_holes] = cid
                    sub_m = sub_occ == cid
                    holes = binary_fill_holes(sub_m) & (~sub_m)
                    modified = True
                if np.any(holes):
                    pad_m = np.pad(sub_m.astype(np.uint8), 1, constant_values=0)
                    inv_pad = 1 - pad_m
                    n_bg, lbl_bg = cv2.connectedComponents(inv_pad, connectivity=4)
                    outer_bg = (lbl_bg == lbl_bg[0, 0])[1:-1, 1:-1]
                    for bg_id in range(1, n_bg):
                        if bg_id == int(lbl_bg[0, 0]):
                            continue
                        hole_mask = lbl_bg[1:-1, 1:-1] == bg_id
                        hy, hx = np.where(hole_mask)
                        if hy.size == 0:
                            continue
                        cy_h, cx_h = int(hy.mean()), int(hx.mean())
                        oy, ox = np.where(outer_bg)
                        if oy.size > 0:
                            d2 = (oy - cy_h) ** 2 + (ox - cx_h) ** 2
                            idx_near = int(np.argmin(d2))
                            ty, tx = int(oy[idx_near]), int(ox[idx_near])
                        else:
                            ty, tx = 0, cx_h
                        cut_canvas = np.zeros(sub_m.shape, dtype=np.uint8)
                        cv2.line(cut_canvas, (cx_h, cy_h), (tx, ty), 1, thickness=2)
                        sub_occ[(cut_canvas > 0) & (sub_occ == cid)] = 0
                    modified = True
                    sub_m = sub_occ == cid
                    n_c2, lbl_c2, stats_c2, _ = cv2.connectedComponentsWithStats(sub_m.astype(np.uint8), connectivity=4)
                    if n_c2 <= 1:
                        continue
                    best_c2 = 1 + int(np.argmax(stats_c2[1:, cv2.CC_STAT_AREA]))
                    sub_occ[sub_m & (lbl_c2 != best_c2)] = 0
                    sub_m = sub_occ == cid
                    if int(sub_m.sum()) < min_area_px:
                        sub_occ[sub_m] = 0
                        continue

            # 3. Chamfer degenerate rectangle fill > 0.915 on downscaled small masks
            ay = sub_m.any(axis=1)
            ax = sub_m.any(axis=0)
            ry0, ry1 = int(np.argmax(ay)), int(len(ay) - np.argmax(ay[::-1]))
            rx0, rx1 = int(np.argmax(ax)), int(len(ax) - np.argmax(ax[::-1]))
            bh, bw = ry1 - ry0, rx1 - rx0
            cur_a = int(sub_m.sum())
            if bh >= 4 and bw >= 4 and float(cur_a) / float(bh * bw) > 0.915:
                yy_loc, xx_loc = np.ogrid[0:bh, 0:bw]
                cy_loc, cx_loc = 0.5 * (bh - 1), 0.5 * (bw - 1)
                ell = (
                    ((xx_loc - cx_loc) / max(0.5 * bw, 1.0)) ** 2
                    + ((yy_loc - cy_loc) / max(0.5 * bh, 1.0)) ** 2
                    <= 1.04
                )
                crop_view = sub_occ[ry0:ry1, rx0:rx1]
                trim = (crop_view == cid) & (~ell)
                if cur_a - int(np.count_nonzero(trim)) >= min_area_px:
                    crop_view[trim] = 0
                    modified = True
                    sub_m3 = sub_occ == cid
                    n_c3, lbl_c3, st_c3, _ = cv2.connectedComponentsWithStats(sub_m3.astype(np.uint8), connectivity=4)
                    if n_c3 > 2:
                        b3 = 1 + int(np.argmax(st_c3[1:, cv2.CC_STAT_AREA]))
                        sub_occ[sub_m3 & (lbl_c3 != b3)] = 0
        if not modified:
            break

    final_slices = find_objects(occ_rs)
    out_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]] = []
    out_scores: list[float] = []
    out_metas: list[dict] = []

    for cid, sl in enumerate(final_slices, start=1):
        if sl is None:
            continue
        sub_m = occ_rs[sl] == cid
        area = int(sub_m.sum())
        if area < min_area_px:
            continue
        n4, lbl4, st4, _ = cv2.connectedComponentsWithStats(sub_m.astype(np.uint8), connectivity=4)
        if n4 > 2:
            b4 = 1 + int(np.argmax(st4[1:, cv2.CC_STAT_AREA]))
            sub_m = lbl4 == b4
            area = int(sub_m.sum())
            if area < min_area_px:
                continue
        ay = sub_m.any(axis=1)
        ax = sub_m.any(axis=0)
        ry0, ry1 = int(np.argmax(ay)), int(len(ay) - np.argmax(ay[::-1]))
        rx0, rx1 = int(np.argmax(ax)), int(len(ax) - np.argmax(ax[::-1]))
        crop = sub_m[ry0:ry1, rx0:rx1].copy()
        post_area = int(crop.sum())
        if post_area < min_area_px:
            continue
        y0, x0 = sl[0].start + ry0, sl[1].start + rx0
        y1, x1 = sl[0].start + ry1, sl[1].start + rx1
        out_sparse.append(((y0, y1, x0, x1), crop, post_area))
        out_scores.append(float(pred_scores[cid - 1]))
        out_metas.append(dict(pred_metas[cid - 1]))

    return out_sparse, out_scores, out_metas


def sparse_predictions_to_tensor_dict(
    pred_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
    pred_scores: list[float],
    hw: tuple[int, int],
    score_threshold: float = 0.10,
) -> dict[str, torch.Tensor]:
    """
    Convert sparse crown predictions to MillionTrees-compatible dense tensor dictionary:
      {"y": uint8 Tensor[N, H, W], "labels": int64 Tensor[N] (zeros), "scores": float32 Tensor[N]}.
    """
    out_h, out_w = int(hw[0]), int(hw[1])
    kept = [
        (sp, float(sc))
        for sp, sc in zip(pred_sparse, pred_scores)
        if float(sc) > float(score_threshold)
    ]
    n = len(kept)
    if n == 0:
        return {
            "y": torch.zeros((0, out_h, out_w), dtype=torch.uint8),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "scores": torch.zeros((0,), dtype=torch.float32),
        }

    masks_np = np.zeros((n, out_h, out_w), dtype=np.uint8)
    scores_np = np.zeros((n,), dtype=np.float32)
    for idx, (((y0, y1, x0, x1), crop_b, _area), sc) in enumerate(kept):
        masks_np[idx, y0:y1, x0:x1] = crop_b.astype(np.uint8)
        scores_np[idx] = sc

    return {
        "y": torch.from_numpy(masks_np),
        "labels": torch.zeros((n,), dtype=torch.int64),
        "scores": torch.from_numpy(scores_np),
    }


# ==============================================================================
# 6. Public Predictor Class (Live SAM 3 + Cached NPZ Parity API)
# ==============================================================================


class CanopyGISv18Predictor:
    """
    Zero-shot CanopyGIS v1.8.1 public predictor.

    Wraps Meta SAM 3 (Option 1 G3 shared vision_embeds + immediate last-use eviction)
    and the locked v1.8 planar Euler==1 physics partitioner.
    """

    def __init__(
        self,
        sam_model: Any,
        sam_processor: Any,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.sam_model = sam_model
        self.sam_processor = sam_processor
        self.sam_model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "facebook/sam3",
        device: str = "auto",
        hf_token: str | None = None,
    ) -> "CanopyGISv18Predictor":
        """Load Meta SAM 3 model and processor (prefers local cache, falls back to Hub download)."""
        if device == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device(device)

        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        from transformers import Sam3Model, Sam3Processor

        try:
            sam_model = Sam3Model.from_pretrained(model_id, local_files_only=True)
            sam_processor = Sam3Processor.from_pretrained(model_id, local_files_only=True)
        except OSError:
            load_kwargs: dict[str, Any] = {}
            if token:
                load_kwargs["token"] = token
            sam_model = Sam3Model.from_pretrained(model_id, **load_kwargs)
            sam_processor = Sam3Processor.from_pretrained(model_id, **load_kwargs)

        sam_model = sam_model.to(torch_device)
        sam_model.eval()
        return cls(sam_model=sam_model, sam_processor=sam_processor, device=torch_device)

    @staticmethod
    def _raw_to_group_tuples(
        raw_items: list[tuple[tuple[int, int, int, int], np.ndarray, float, int]],
    ) -> list[tuple[tuple[int, int, int, int], np.ndarray, float, int, int]]:
        out = []
        for bbox, crop, sc, seam in raw_items:
            area = int(crop.sum())
            if area > 0:
                out.append((bbox, crop, float(np.float32(sc)), int(seam), area))
        return out

    def extract_proposals(
        self,
        img_rgb: np.ndarray,
        gsd_m: float | None = None,
    ) -> dict[str, list[tuple[tuple[int, int, int, int], np.ndarray, float, int, int]]]:
        """
        Live in-memory generation of the exact 7 proposal groups
        (ff_tree, ff_ind, ff_can, ff_enh, ti_tree, ti_ind, ti_can) without writing .npz files.
        """
        img_u8 = _coerce_rgb_uint8(img_rgb)
        H, W = img_u8.shape[:2]
        pil_img = Image.fromarray(img_u8)
        enh_rgb = enhance_shadow_luminance_cielab(img_u8)
        pil_enh = Image.fromarray(enh_rgb)

        # 1. Full-Frame SAM 3 (shared vision_embeds across 3 prompts on pil_img, then evicted)
        raw_ff_batches = run_sam3_sparse_batch(
            self.sam_model,
            self.sam_processor,
            [pil_img, pil_img, pil_img, pil_enh],
            ["tree", "individual tree canopy", "tree canopy", "tree"],
            self.device,
            post_thr=0.10,
            mask_thr=0.50,
            max_post_side=2200,
        )
        ff_tree_raw, ff_ind_raw, ff_can_raw, ff_enh_raw = raw_ff_batches

        # 2. Scale-Adaptive Multi-Scale Tiled SAM 3 (one 3-prompt batch per window)
        windows = compute_adaptive_sam_windows(H, W, gsd_m)
        ti_tree_raw: list[tuple[tuple[int, int, int, int], np.ndarray, float, int]] = []
        ti_ind_raw: list[tuple[tuple[int, int, int, int], np.ndarray, float, int]] = []
        ti_can_raw: list[tuple[tuple[int, int, int, int], np.ndarray, float, int]] = []

        for (x0, y0, x1, y1) in windows:
            crop_raw = Image.fromarray(img_u8[y0:y1, x0:x1])
            crop_enh = Image.fromarray(enh_rgb[y0:y1, x0:x1])
            tw, th = x1 - x0, y1 - y0
            meta_win = [
                (x0, y0, tw, th, W, H),
                (x0, y0, tw, th, W, H),
                (x0, y0, tw, th, W, H),
            ]
            sub_batches = run_sam3_sparse_batch(
                self.sam_model,
                self.sam_processor,
                [crop_raw, crop_raw, crop_enh],
                ["tree", "tree canopy", "individual tree canopy"],
                self.device,
                post_thr=0.12,
                mask_thr=0.50,
                max_post_side=2200,
                meta_list=meta_win,
            )
            ti_tree_raw.extend(sub_batches[0])
            ti_can_raw.extend(sub_batches[1])
            ti_ind_raw.extend(sub_batches[2])

        return {
            "ff_tree": self._raw_to_group_tuples(ff_tree_raw),
            "ff_ind": self._raw_to_group_tuples(ff_ind_raw),
            "ff_can": self._raw_to_group_tuples(ff_can_raw),
            "ff_enh": self._raw_to_group_tuples(ff_enh_raw),
            "ti_tree": self._raw_to_group_tuples(ti_tree_raw),
            "ti_ind": self._raw_to_group_tuples(ti_ind_raw),
            "ti_can": self._raw_to_group_tuples(ti_can_raw),
        }

    def predict_tile_sparse(
        self,
        img_rgb: np.ndarray,
        gsd_m: float | None = None,
        remap_ws_scores: bool = True,
    ) -> tuple[list[tuple[tuple[int, int, int, int], np.ndarray, int]], list[float], list[dict], dict]:
        """Run full live CanopyGIS v1.8.1 inference and return sparse planar crowns + scores + metadata."""
        img_u8 = _coerce_rgb_uint8(img_rgb)
        groups = self.extract_proposals(img_u8, gsd_m=gsd_m)
        precomp = prepare_tile_precomp_from_groups(img_u8, groups, gsd_m)
        pred_sparse, pred_scores, pred_metas, diag = run_canopygis_v18_tile(img_u8, precomp, gsd_m)
        if remap_ws_scores:
            pred_scores = remap_post_carve_scores(pred_scores, pred_metas)
        return pred_sparse, pred_scores, pred_metas, diag

    def predict_tile(
        self,
        img_rgb: np.ndarray,
        gsd_m: float | None = None,
        score_threshold: float = 0.10,
        remap_ws_scores: bool = True,
        output_hw: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Run full live CanopyGIS v1.8.1 inference and return MillionTrees-compatible tensors:
          {"y": uint8 Tensor[N, H, W], "labels": int64 Tensor[N] (zeros), "scores": float32 Tensor[N]}.
        """
        img_u8 = _coerce_rgb_uint8(img_rgb)
        src_h, src_w = img_u8.shape[:2]
        pred_sparse, pred_scores, pred_metas, _diag = self.predict_tile_sparse(
            img_u8, gsd_m=gsd_m, remap_ws_scores=remap_ws_scores
        )
        if output_hw is not None and (int(output_hw[0]), int(output_hw[1])) != (src_h, src_w):
            pred_sparse, pred_scores, pred_metas = resize_planar_predictions(
                pred_sparse,
                pred_scores,
                pred_metas,
                src_hw=(src_h, src_w),
                dst_hw=(int(output_hw[0]), int(output_hw[1])),
                min_area_px=16,
            )
            target_hw = (int(output_hw[0]), int(output_hw[1]))
        else:
            target_hw = (src_h, src_w)

        return sparse_predictions_to_tensor_dict(
            pred_sparse,
            pred_scores,
            hw=target_hw,
            score_threshold=score_threshold,
        )

    @staticmethod
    def predict_from_cached_npz_sparse(
        img_rgb: np.ndarray,
        npz_cache: Any,
        gsd_m: float | None = None,
        remap_ws_scores: bool = True,
    ) -> tuple[list[tuple[tuple[int, int, int, int], np.ndarray, int]], list[float], list[dict], dict]:
        """Zero-copy cached-.npz parity path returning sparse planar crowns + scores + metadata."""
        img_u8 = _coerce_rgb_uint8(img_rgb)
        opened = False
        npz_obj = npz_cache
        if isinstance(npz_cache, (str, Path)):
            npz_obj = np.load(npz_cache, allow_pickle=False)
            opened = True
        try:
            precomp = prepare_tile_precomp(img_u8, npz_obj, gsd_m)
        finally:
            if opened:
                npz_obj.close()
        pred_sparse, pred_scores, pred_metas, diag = run_canopygis_v18_tile(img_u8, precomp, gsd_m)
        if remap_ws_scores:
            pred_scores = remap_post_carve_scores(pred_scores, pred_metas)
        return pred_sparse, pred_scores, pred_metas, diag

    @staticmethod
    def predict_from_cached_npz(
        img_rgb: np.ndarray,
        npz_cache: Any,
        gsd_m: float | None = None,
        score_threshold: float = 0.10,
        remap_ws_scores: bool = True,
        output_hw: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Cached-.npz parity path returning MillionTrees-compatible dense tensors."""
        img_u8 = _coerce_rgb_uint8(img_rgb)
        src_h, src_w = img_u8.shape[:2]
        pred_sparse, pred_scores, pred_metas, _diag = CanopyGISv18Predictor.predict_from_cached_npz_sparse(
            img_u8, npz_cache, gsd_m=gsd_m, remap_ws_scores=remap_ws_scores
        )
        if output_hw is not None and (int(output_hw[0]), int(output_hw[1])) != (src_h, src_w):
            pred_sparse, pred_scores, pred_metas = resize_planar_predictions(
                pred_sparse,
                pred_scores,
                pred_metas,
                src_hw=(src_h, src_w),
                dst_hw=(int(output_hw[0]), int(output_hw[1])),
                min_area_px=16,
            )
            target_hw = (int(output_hw[0]), int(output_hw[1]))
        else:
            target_hw = (src_h, src_w)

        return sparse_predictions_to_tensor_dict(
            pred_sparse,
            pred_scores,
            hw=target_hw,
            score_threshold=score_threshold,
        )
