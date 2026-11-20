"""
CanopyGIS v1.8.1 Verification Suite: Native Parity, Decision 2A AP40/AP60 Delta,
Decision 3A Pure-Image (gsd_m=None) Parity, and Live GPU Mode A vs Mode B A/B Gate.

Modes:
  --mode parity-native
      Loads frozen .npz proposal caches across Holdout_v18 (Holdout_OOD_12 + Official_TLS_59,
      incl. Official TLS Unseen-51) and evaluates:
        (a) gsd_m=manifest, remap=False  (parity check vs TLS-59=0.6201, Unseen-51=0.6311, OOD-12=0.7524)
        (b) gsd_m=manifest, remap=True   (asserts DeltaMask == 0 px & DeltaF1 == 0.0000; measures AP40/AP60 delta)
        (c) gsd_m=None,     remap=True   (Decision 3A pure-image parity vs manifest gsd_m)

  --mode ab-gate
      Runs live GPU CanopyGISv18Predictor.from_pretrained(...) across Holdout_v18 (gsd_m=None):
        - Mode A (1024 direct): resizes RGB, GT, and tree_coverage_mask to 1024x1024 identically to
          TreePolygonsDataset(..., image_size=1024), runs predict_tile_sparse on 1024x1024 (reporting
          both remap_ws_scores=True and remap_ws_scores=False on identical masks for AP40/AP60 delta).
        - Mode B (native -> 1024 INTER_NEAREST): runs live inference on native RGB (gsd_m=None),
          resizes planar predictions to 1024x1024 via resize_planar_predictions, and scores against
          the identical 1024x1024 GT and tree_coverage_mask.

Uses official milliontrees.common.metrics.all_metrics (MaskAccuracy, MaskAwareMaskPrecision,
DetectionMAP / make_mean_average_precision at IoU 0.40 & 0.60 with max_detection_thresholds=[1, 10, 1000])
when installed, with a mathematically equivalent standalone fallback otherwise.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from shapely import from_wkt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from canopygis_v18_eval import (  # noqa: E402
    CanopyGISv18Predictor,
    remap_post_carve_scores,
    resize_planar_predictions,
    sparse_predictions_to_tensor_dict,
)

try:
    from milliontrees.common.metrics.all_metrics import (
        DetectionMAP,
        MaskAccuracy,
        MaskAwareMaskPrecision,
        make_mean_average_precision,
    )

    HAS_MILLIONTREES = True
except Exception:
    DetectionMAP = None
    MaskAccuracy = None
    MaskAwareMaskPrecision = None
    make_mean_average_precision = None
    HAS_MILLIONTREES = False

try:
    from torchmetrics.detection import MeanAveragePrecision as _TMMeanAveragePrecision

    HAS_TORCHMETRICS = True
except Exception:
    _TMMeanAveragePrecision = None
    HAS_TORCHMETRICS = False


LOCKED_PARITY_F1_TARGETS = {
    "part_1a_ood_12": 0.7524,
    "part_1b_tls_unseen_51": 0.6311,
    "part_1b_tls_complete_59": 0.6201,
}


# ==============================================================================
# 1. File Integrity, GT Rasterization & Topology Verification Helpers
# ==============================================================================


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_dataset_dir(cli_val: str | None, env_var: str, fallback_rel: Path) -> Path:
    if cli_val:
        p = Path(cli_val).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified path does not exist: {p}")
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        p = Path(env_val).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"Environment variable {env_var} points to missing path: {p}")
    candidate = REPO_ROOT / fallback_rel
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not resolve directory. Pass --{env_var.lower().replace('canopygis_', '').replace('_', '-')} "
        f"or set {env_var}."
    )


def load_gt_sparse_native(
    H: int,
    W: int,
    gt_rows: list[dict[str, str]],
) -> list[tuple[tuple[int, int, int, int], np.ndarray, int]]:
    """Rasterize GT WKT polygons at native (H, W) into tight sparse boolean crops."""
    gt_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]] = []
    for row in gt_rows:
        wkt_str = row.get("geometry") or row.get("polygon")
        if not wkt_str:
            continue
        try:
            geom = from_wkt(wkt_str)
            if geom is None or geom.is_empty:
                continue
            polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
            if not polys:
                continue
            minx, miny, maxx, maxy = geom.bounds
            x0 = max(0, min(W - 1, int(np.floor(minx))))
            y0 = max(0, min(H - 1, int(np.floor(miny))))
            x1 = max(x0 + 1, min(W, int(np.ceil(maxx)) + 1))
            y1 = max(y0 + 1, min(H, int(np.ceil(maxy)) + 1))
            sub = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            for poly in polys:
                if poly.is_empty:
                    continue
                pts = np.asarray(poly.exterior.coords, dtype=np.float32)
                pts_i = np.round(pts).astype(np.int32)
                pts_i[:, 0] = np.clip(pts_i[:, 0], 0, W - 1) - x0
                pts_i[:, 1] = np.clip(pts_i[:, 1], 0, H - 1) - y0
                cv2.fillPoly(sub, [pts_i], 1)
            crop = sub > 0
            ay = crop.any(axis=1)
            if not ay.any():
                continue
            ax = crop.any(axis=0)
            ry0, ry1 = int(np.argmax(ay)), int(len(ay) - np.argmax(ay[::-1]))
            rx0, rx1 = int(np.argmax(ax)), int(len(ax) - np.argmax(ax[::-1]))
            tight = crop[ry0:ry1, rx0:rx1].copy()
            area = int(tight.sum())
            if area > 0:
                gt_sparse.append(((y0 + ry0, y0 + ry1, x0 + rx0, x0 + rx1), tight, area))
        except Exception:
            continue
    return gt_sparse


def load_gt_sparse_resized_tree_polygons(
    orig_h: int,
    orig_w: int,
    dst_h: int,
    dst_w: int,
    gt_rows: list[dict[str, str]],
) -> list[tuple[tuple[int, int, int, int], np.ndarray, int]]:
    """
    Rasterize GT WKT polygons at (dst_h, dst_w) matching TreePolygonsDataset.__getitem__
    and create_polygon_mask bit-for-bit:
      - scale_x = dst_w / orig_w, scale_y = dst_h / orig_h
      - pts = np.asarray(coords, float64); pts[:,0]*=scale_x; pts[:,1]*=scale_y; pts.astype(int32)
      - cv2.fillPoly(mask_img, [pts], 255)
      - Filters empty masks and degenerate bboxes ((x2 > x1) & (y2 > y1)).
    """
    scale_x = float(dst_w) / float(orig_w)
    scale_y = float(dst_h) / float(orig_h)
    gt_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]] = []
    for row in gt_rows:
        wkt_str = row.get("geometry") or row.get("polygon")
        if not wkt_str:
            continue
        try:
            geom = from_wkt(wkt_str)
            if geom is None or geom.is_empty:
                continue
            polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
            if not polys:
                continue
            mask_img = np.zeros((dst_h, dst_w), dtype=np.uint8)
            for poly in polys:
                if poly.is_empty:
                    continue
                coords = poly.exterior.coords
                if scale_x == 1.0 and scale_y == 1.0:
                    pts = np.asarray(coords, dtype=np.int32)
                else:
                    pts = np.asarray(coords, dtype=np.float64)
                    pts[:, 0] *= scale_x
                    pts[:, 1] *= scale_y
                    pts = pts.astype(np.int32)
                cv2.fillPoly(mask_img, [pts], 255)
            crop_full = mask_img > 0
            ay = crop_full.any(axis=1)
            if not ay.any():
                continue
            ax = crop_full.any(axis=0)
            y0, y1 = int(np.argmax(ay)), int(len(ay) - np.argmax(ay[::-1]))
            x0, x1 = int(np.argmax(ax)), int(len(ax) - np.argmax(ax[::-1]))
            # Match TreePolygonsDataset degenerate bbox filter: (x2 > x1) & (y2 > y1)
            if (x1 - 1) <= x0 or (y1 - 1) <= y0:
                continue
            tight = crop_full[y0:y1, x0:x1].copy()
            area = int(tight.sum())
            if area > 0:
                gt_sparse.append(((y0, y1, x0, x1), tight, area))
        except Exception:
            continue
    return gt_sparse


def sparse_list_to_dense_bool(
    sparse_list: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
    H: int,
    W: int,
) -> torch.Tensor:
    n = len(sparse_list)
    if n == 0:
        return torch.zeros((0, H, W), dtype=torch.bool)
    out = np.zeros((n, H, W), dtype=bool)
    for idx, ((y0, y1, x0, x1), crop_b, _area) in enumerate(sparse_list):
        out[idx, y0:y1, x0:x1] = crop_b
    return torch.from_numpy(out)


def audit_overlap_and_topology(
    pred_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
    H: int,
    W: int,
) -> dict[str, int]:
    """Verify planar mutual exclusion (overlap_px==0), Euler==1, and no boxy rectangles >0.92."""
    stack = np.zeros((H, W), dtype=np.int32)
    n_with_holes = 0
    n_multi_comp = 0
    n_euler_neq_1 = 0
    n_rect_gt_092 = 0

    for (y0, y1, x0, x1), crop_b, _area in pred_sparse:
        stack[y0:y1, x0:x1] += crop_b.astype(np.int32)

        u8 = crop_b.astype(np.uint8)
        n4, _ = cv2.connectedComponents(u8, connectivity=4)
        n_comp = max(0, int(n4) - 1)
        holes_mask = binary_fill_holes(crop_b) & (~crop_b)
        if np.any(holes_mask):
            nh, _ = cv2.connectedComponents(holes_mask.astype(np.uint8), connectivity=4)
            n_holes = max(0, int(nh) - 1)
        else:
            n_holes = 0

        if n_holes > 0:
            n_with_holes += 1
        if n_comp != 1:
            n_multi_comp += 1
        if (n_comp - n_holes) != 1 or n_comp != 1 or n_holes != 0:
            n_euler_neq_1 += 1

        bh, bw = crop_b.shape
        a_px = int(crop_b.sum())
        if bh >= 4 and bw >= 4 and (float(a_px) / float(bh * bw)) > 0.92:
            n_rect_gt_092 += 1

    return {
        "overlap_px": int(np.count_nonzero(stack > 1)),
        "max_stack_depth": int(stack.max()) if stack.size else 0,
        "n_with_holes": n_with_holes,
        "n_multi_component": n_multi_comp,
        "n_euler_neq_1": n_euler_neq_1,
        "n_rect_gt_092": n_rect_gt_092,
    }


def compute_sparse_mask_delta_px(
    sparse_a: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
    sparse_b: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
    H: int,
    W: int,
) -> int:
    """Return exact pixel difference count between two planar sparse partitions."""
    if len(sparse_a) == len(sparse_b):
        exact_match = True
        for (ba, ca, aa), (bb, cb, ab) in zip(sparse_a, sparse_b):
            if ba != bb or aa != ab or not np.array_equal(ca, cb):
                exact_match = False
                break
        if exact_match:
            return 0
    occ_a = np.zeros((H, W), dtype=np.int32)
    occ_b = np.zeros((H, W), dtype=np.int32)
    for i, ((y0, y1, x0, x1), crop_b, _) in enumerate(sparse_a, start=1):
        occ_a[y0:y1, x0:x1][crop_b] = i
    for i, ((y0, y1, x0, x1), crop_b, _) in enumerate(sparse_b, start=1):
        occ_b[y0:y1, x0:x1][crop_b] = i
    return int(np.count_nonzero(occ_a != occ_b))


# ==============================================================================
# 2. MillionTrees-Exact IoU, Elementwise Metrics & Streaming AP40/AP60
# ==============================================================================


def _pure_torch_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return torch.where(union > 0, inter / union.clamp(min=1e-6), torch.zeros_like(inter))


def _greedy_iou_match(iou: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    n_gt, n_pred = iou.shape
    gt_to_pred = torch.full((n_gt,), -1, dtype=torch.long)
    if n_gt == 0 or n_pred == 0:
        return gt_to_pred
    iou_cpu = iou.detach().cpu()
    valid = iou_cpu > iou_threshold
    if not valid.any():
        return gt_to_pred
    vals = iou_cpu[valid]
    ij = torch.nonzero(valid, as_tuple=False)
    order = vals.argsort(descending=True)
    pred_used = np.zeros(n_pred, dtype=bool)
    for k in order.tolist():
        gi = int(ij[k, 0])
        pj = int(ij[k, 1])
        if gt_to_pred[gi] >= 0 or pred_used[pj]:
            continue
        gt_to_pred[gi] = pj
        pred_used[pj] = True
    return gt_to_pred


def sparse_to_224_and_boxes(
    sparse_list: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
    H: int,
    W: int,
    target_size: int = 224,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-copy conversion of sparse crops to exact torchvision masks_to_boxes [N, 4]
    and nearest-sampled [N, 224, 224] bool masks (identical to MaskAccuracy._mask_iou).
    """
    n = len(sparse_list)
    use_ds = H > target_size
    out_h = target_size if use_ds else H
    out_w = target_size if use_ds else W
    if n == 0:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0, out_h, out_w), dtype=torch.bool),
        )

    boxes = np.zeros((n, 4), dtype=np.float32)
    masks_small = np.zeros((n, out_h, out_w), dtype=bool)

    if use_ds:
        iy = (np.arange(target_size, dtype=np.int64) * H) // target_size
        ix = (np.arange(target_size, dtype=np.int64) * W) // target_size
    else:
        iy = None
        ix = None

    for idx, ((y0, y1, x0, x1), crop_b, _area) in enumerate(sparse_list):
        ay = crop_b.any(axis=1)
        ax = crop_b.any(axis=0)
        if not ay.any() or not ax.any():
            continue
        by0 = y0 + int(np.argmax(ay))
        by1 = y0 + int(len(ay) - 1 - np.argmax(ay[::-1]))
        bx0 = x0 + int(np.argmax(ax))
        bx1 = x0 + int(len(ax) - 1 - np.argmax(ax[::-1]))
        boxes[idx] = [float(bx0), float(by0), float(bx1), float(by1)]

        if use_ds:
            ym = (iy >= y0) & (iy < y1)
            xm = (ix >= x0) & (ix < x1)
            if ym.any() and xm.any():
                dy = np.nonzero(ym)[0]
                dx = np.nonzero(xm)[0]
                sub_sampled = crop_b[iy[dy] - y0][:, ix[dx] - x0]
                masks_small[idx][np.ix_(dy, dx)] = sub_sampled
        else:
            masks_small[idx, y0:y1, x0:x1] = crop_b

    return torch.from_numpy(boxes), torch.from_numpy(masks_small)


def compute_mt224_iou_from_sparse(
    gt_boxes: torch.Tensor,
    gt_224: torch.Tensor,
    p_boxes: torch.Tensor,
    p_224: torch.Tensor,
) -> torch.Tensor:
    """Bit-identical to MillionTrees MaskAccuracy._mask_iou (hybrid bbox + 224x224 mask IoU)."""
    n_gt, n_pred = len(gt_boxes), len(p_boxes)
    if n_gt == 0 or n_pred == 0:
        return torch.zeros((n_gt, n_pred), dtype=torch.float32)

    bbox_iou = _pure_torch_box_iou(gt_boxes, p_boxes)
    iou = bbox_iou.clone()
    iou[bbox_iou < 0.1] = 0.0
    amb = (bbox_iou >= 0.1) & (bbox_iou <= 0.9)
    if amb.any():
        amb_g, amb_p = torch.where(amb)
        u_g, inv_g = torch.unique(amb_g, return_inverse=True)
        u_p, inv_p = torch.unique(amb_p, return_inverse=True)
        g_flat = gt_224[u_g].flatten(1)
        p_flat = p_224[u_p].flatten(1)
        g_areas = g_flat.float().sum(dim=1)
        p_areas = p_flat.float().sum(dim=1)
        num_amb = len(amb_g)
        chunk_size = 2048
        for st in range(0, num_amb, chunk_size):
            en = min(st + chunk_size, num_amb)
            gl = inv_g[st:en]
            pl = inv_p[st:en]
            inter = (g_flat[gl] & p_flat[pl]).float().sum(dim=1)
            union = g_areas[gl] + p_areas[pl] - inter
            vals = torch.where(union > 0, inter / union.clamp(min=1e-6), torch.zeros_like(inter))
            iou[amb_g[st:en], amb_p[st:en]] = vals
    return iou


def _compute_exact_sparse_pixel_iou_matrix(
    pred_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
    gt_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
) -> np.ndarray:
    """Exact pixel-level mask IoU [n_pred, n_gt] without allocating dense full-frame arrays."""
    n_p = len(pred_sparse)
    n_g = len(gt_sparse)
    out = np.zeros((n_p, n_g), dtype=np.float64)
    if n_p == 0 or n_g == 0:
        return out
    gt_bboxes = [g[0] for g in gt_sparse]
    for pi, ((py0, py1, px0, px1), pcrop, parea) in enumerate(pred_sparse):
        if parea <= 0:
            continue
        for gi, (gy0, gy1, gx0, gx1) in enumerate(gt_bboxes):
            oy0 = max(py0, gy0)
            oy1 = min(py1, gy1)
            if oy1 <= oy0:
                continue
            ox0 = max(px0, gx0)
            ox1 = min(px1, gx1)
            if ox1 <= ox0:
                continue
            gcrop = gt_sparse[gi][1]
            garea = gt_sparse[gi][2]
            inter = int(
                np.count_nonzero(
                    pcrop[oy0 - py0:oy1 - py0, ox0 - px0:ox1 - px0]
                    & gcrop[oy0 - gy0:oy1 - gy0, ox0 - gx0:ox1 - gx0]
                )
            )
            if inter > 0:
                union = parea + garea - inter
                if union > 0:
                    out[pi, gi] = float(inter) / float(union)
    return out


class _PerSourceAPAccumulator:
    """
    Streaming per-source AP40 / AP60 accumulator matching MillionTrees DetectionMAP
    (iou_type='segm', max_detection_thresholds=[1, 10, 1000], single class).
    """

    def __init__(self, iou_threshold: float) -> None:
        self.iou_threshold = float(iou_threshold)
        self._metric = None
        if HAS_MILLIONTREES and make_mean_average_precision is not None:
            self.backend = "milliontrees"
            self._metric = make_mean_average_precision(
                iou_type="segm",
                iou_thresholds=[self.iou_threshold],
                max_detection_thresholds=[1, 10, 1000],
                class_metrics=False,
            )
        elif HAS_TORCHMETRICS and _TMMeanAveragePrecision is not None:
            self.backend = "torchmetrics"
            self._metric = _TMMeanAveragePrecision(
                iou_type="segm",
                iou_thresholds=[self.iou_threshold],
                max_detection_thresholds=[1, 10, 1000],
                class_metrics=False,
            )
        else:
            self.backend = "numpy_fallback"
            self._scores_list: list[np.ndarray] = []
            self._tp_list: list[np.ndarray] = []
            self._fp_list: list[np.ndarray] = []
            self._total_gt = 0

        self._n_updates = 0

    def update(
        self,
        pred_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
        pred_scores: list[float],
        gt_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
        H: int,
        W: int,
        score_threshold: float,
        pred_dense_bool: torch.Tensor | None = None,
        gt_dense_bool: torch.Tensor | None = None,
    ) -> None:
        self._n_updates += 1
        scores_t = torch.as_tensor(pred_scores, dtype=torch.float32)
        keep_mask = scores_t > float(score_threshold)
        keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1).tolist()

        if self._metric is not None:
            if pred_dense_bool is not None:
                p_masks = pred_dense_bool[keep_mask]
            else:
                kept_sp = [pred_sparse[i] for i in keep_idx]
                p_masks = sparse_list_to_dense_bool(kept_sp, H, W)
            p_scores = scores_t[keep_mask]
            g_masks = gt_dense_bool if gt_dense_bool is not None else sparse_list_to_dense_bool(gt_sparse, H, W)

            pred_dict = {
                "masks": p_masks.bool(),
                "scores": p_scores.float(),
                "labels": torch.zeros(len(p_scores), dtype=torch.long),
            }
            gt_dict = {
                "masks": g_masks.bool(),
                "labels": torch.zeros(len(g_masks), dtype=torch.long),
            }
            self._metric.update([pred_dict], [gt_dict])
            return

        # Exact 101-point COCO AP fallback (max_detections=1000, exact pixel mask IoU)
        n_gt = len(gt_sparse)
        self._total_gt += n_gt
        if not keep_idx:
            return
        kept_sp = [pred_sparse[i] for i in keep_idx]
        kept_sc = np.array([float(pred_scores[i]) for i in keep_idx], dtype=np.float64)
        order = np.argsort(-kept_sc, kind="mergesort")[:1000]
        kept_sp = [kept_sp[i] for i in order]
        kept_sc = kept_sc[order]
        n_p = len(kept_sp)

        tp = np.zeros(n_p, dtype=np.float64)
        fp = np.zeros(n_p, dtype=np.float64)
        if n_gt == 0:
            fp[:] = 1.0
        else:
            iou_mat = _compute_exact_sparse_pixel_iou_matrix(kept_sp, gt_sparse)
            gt_matched = np.zeros(n_gt, dtype=bool)
            thr = self.iou_threshold
            for pi in range(n_p):
                best_g = -1
                best_iou = thr - 1e-12
                row = iou_mat[pi]
                for gi in range(n_gt):
                    if gt_matched[gi]:
                        continue
                    v = float(row[gi])
                    if v >= thr and v > best_iou:
                        best_iou = v
                        best_g = gi
                if best_g >= 0:
                    gt_matched[best_g] = True
                    tp[pi] = 1.0
                else:
                    fp[pi] = 1.0

        self._scores_list.append(kept_sc)
        self._tp_list.append(tp)
        self._fp_list.append(fp)

    def compute(self) -> float:
        if self._n_updates == 0:
            return 0.0
        if self._metric is not None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self._metric._to_sync = False
            res = self._metric.compute()
            val = float(res["map"].item())
            return max(0.0, val)

        if self._total_gt == 0 or not self._scores_list:
            return 0.0
        all_sc = np.concatenate(self._scores_list, axis=0)
        all_tp = np.concatenate(self._tp_list, axis=0)
        all_fp = np.concatenate(self._fp_list, axis=0)
        order = np.argsort(-all_sc, kind="mergesort")
        tp_sum = np.cumsum(all_tp[order])
        fp_sum = np.cumsum(all_fp[order])
        rc = tp_sum / float(self._total_gt)
        pr = tp_sum / (tp_sum + fp_sum + np.spacing(1))
        for i in range(len(pr) - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]
        rec_thrs = np.linspace(0.0, 1.0, 101)
        inds = np.searchsorted(rc, rec_thrs, side="left")
        q = np.zeros(len(rec_thrs), dtype=np.float64)
        for ri, pi in enumerate(inds):
            if pi < len(pr):
                q[ri] = pr[pi]
        return float(np.mean(q))


def evaluate_single_tile_elementwise(
    tile_meta: dict[str, Any],
    gt_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
    gt_boxes: torch.Tensor,
    gt_224: torch.Tensor,
    tm_bool: np.ndarray | None,
    pred_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
    pred_scores: list[float],
    H: int,
    W: int,
    score_threshold: float = 0.10,
    gt_dense_bool: torch.Tensor | None = None,
    pred_dense_bool: torch.Tensor | None = None,
    mt_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute per-tile MaskRecall, MaskAwarePrecision, PlainPrecision, MaskAccuracy,
    and planar/topological audit metrics.
    """
    keep_idx = [i for i, sc in enumerate(pred_scores) if float(sc) > float(score_threshold)]
    kept_sparse = [pred_sparse[i] for i in keep_idx]

    n_gt = len(gt_sparse)
    n_pred = len(kept_sparse)

    use_official_mt = (
        mt_metrics is not None
        and HAS_MILLIONTREES
        and gt_dense_bool is not None
        and pred_dense_bool is not None
        and (H * W <= 1500 * 1500)
    )

    if use_official_mt:
        scores_t = torch.as_tensor(pred_scores, dtype=torch.float32)
        p_masks = pred_dense_bool[scores_t > float(score_threshold)]
        g_masks = gt_dense_bool
        tm_t = torch.from_numpy(tm_bool) if tm_bool is not None else None
        acc_m = mt_metrics["accuracy"]
        rec_m = mt_metrics["recall"]
        prec_m = mt_metrics["maskaware_precision"]

        if n_gt > 0 and n_pred > 0:
            iou_mt = acc_m._mask_iou(g_masks, p_masks)
        else:
            iou_mt = None

        recall = float(rec_m._recall(g_masks, p_masks, 0.40, iou=iou_mt).item())
        mask_acc = float(acc_m._accuracy(g_masks, p_masks, 0.40, iou=iou_mt).item())
        prec_mask = float(prec_m._precision(g_masks, p_masks, tm_t, iou=iou_mt).item())
        prec_plain = float(prec_m._precision(g_masks, p_masks, None, iou=iou_mt).item())
        if n_gt > 0 and n_pred > 0 and iou_mt is not None:
            gt_to_pred = _greedy_iou_match(iou_mt, 0.40)
            tp = int((gt_to_pred >= 0).sum().item())
        else:
            tp = 0
    else:
        p_boxes, p_224 = sparse_to_224_and_boxes(kept_sparse, H, W)
        iou_mt = compute_mt224_iou_from_sparse(gt_boxes, gt_224, p_boxes, p_224)

        if n_gt == 0 and n_pred == 0:
            tp, recall, mask_acc, prec_mask, prec_plain = 0, 1.0, 1.0, 1.0, 1.0
        elif n_pred == 0:
            tp, recall, mask_acc, prec_mask, prec_plain = 0, 0.0, 0.0, 0.0, 0.0
        elif n_gt == 0:
            tp, recall, mask_acc = 0, 0.0, 0.0
            ign = 0
            if tm_bool is not None:
                for (py0, py1, px0, px1), p_crop, p_area in kept_sparse:
                    tm_ov = np.count_nonzero(p_crop & tm_bool[py0:py1, px0:px1])
                    if p_area > 0 and float(tm_ov) / float(p_area) >= 0.50:
                        ign += 1
            counted_fp = n_pred - ign
            prec_mask = 1.0 if counted_fp == 0 else 0.0
            prec_plain = 0.0
        else:
            gt_to_pred = _greedy_iou_match(iou_mt, 0.40)
            matched = gt_to_pred[gt_to_pred >= 0]
            tp = int(len(matched))
            recall = float(tp) / float(n_gt)
            fp_raw = n_pred - tp
            fn = n_gt - tp
            mask_acc = float(tp) / float(tp + fp_raw + fn)

            matched_set = set(matched.unique().tolist())
            unmatched_indices = [i for i in range(n_pred) if i not in matched_set]
            max_iou_per_pred = iou_mt.max(dim=0).values.numpy()

            ign = 0
            for p_idx in unmatched_indices:
                if float(max_iou_per_pred[p_idx]) > 0.40:
                    continue
                (py0, py1, px0, px1), p_crop, p_area = kept_sparse[p_idx]
                if p_area <= 0:
                    continue
                if tm_bool is not None:
                    ov_tm = int(np.count_nonzero(p_crop & tm_bool[py0:py1, px0:px1]))
                    if float(ov_tm) / float(p_area) >= 0.50:
                        ign += 1
            counted_fp = len(unmatched_indices) - ign
            denom = tp + counted_fp
            prec_mask = 1.0 if denom == 0 else float(tp) / float(denom)
            prec_plain = float(tp) / float(n_pred)

    topo = audit_overlap_and_topology(kept_sparse, H, W)
    return {
        "short_id": tile_meta["short_id"],
        "source": tile_meta["source"],
        "part": tile_meta["part"],
        "in_dev20": bool(tile_meta.get("in_dev20", False)),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": tp,
        "recall": float(recall),
        "precision_maskaware": float(prec_mask),
        "precision_plain": float(prec_plain),
        "mask_accuracy": float(mask_acc),
        "overlap_px": topo["overlap_px"],
        "max_stack_depth": topo["max_stack_depth"],
        "n_with_holes": topo["n_with_holes"],
        "n_multi_component": topo["n_multi_component"],
        "n_euler_neq_1": topo["n_euler_neq_1"],
        "n_rect_gt_092": topo["n_rect_gt_092"],
    }


class SubsetEvaluator:
    """Tracks per-tile metrics, per-source AP40/AP60, timing, and peak VRAM for one split subset."""

    def __init__(self, score_threshold: float = 0.10) -> None:
        self.score_threshold = float(score_threshold)
        self.rows: list[dict[str, Any]] = []
        self.ap40_by_src: dict[str, _PerSourceAPAccumulator] = {}
        self.ap60_by_src: dict[str, _PerSourceAPAccumulator] = {}
        self.elapsed_s_list: list[float] = []
        self.peak_vram_mb_list: list[float] = []
        self.ap_backend: str = "milliontrees" if HAS_MILLIONTREES else (
            "torchmetrics" if HAS_TORCHMETRICS else "numpy_fallback"
        )

    def add_tile(
        self,
        row: dict[str, Any],
        pred_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
        pred_scores: list[float],
        gt_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
        H: int,
        W: int,
        elapsed_s: float,
        peak_vram_mb: float,
        pred_dense_bool: torch.Tensor | None = None,
        gt_dense_bool: torch.Tensor | None = None,
    ) -> None:
        src = row["source"]
        self.rows.append(row)
        self.elapsed_s_list.append(float(elapsed_s))
        self.peak_vram_mb_list.append(float(peak_vram_mb))

        if src not in self.ap40_by_src:
            self.ap40_by_src[src] = _PerSourceAPAccumulator(0.40)
            self.ap60_by_src[src] = _PerSourceAPAccumulator(0.60)
            self.ap_backend = self.ap40_by_src[src].backend

        self.ap40_by_src[src].update(
            pred_sparse,
            pred_scores,
            gt_sparse,
            H,
            W,
            self.score_threshold,
            pred_dense_bool=pred_dense_bool,
            gt_dense_bool=gt_dense_bool,
        )
        self.ap60_by_src[src].update(
            pred_sparse,
            pred_scores,
            gt_sparse,
            H,
            W,
            self.score_threshold,
            pred_dense_bool=pred_dense_bool,
            gt_dense_bool=gt_dense_bool,
        )

    def summarize(self) -> dict[str, Any]:
        if not self.rows:
            return {
                "n_tiles": 0,
                "n_sources": 0,
                "metrics_backend": self.ap_backend,
                "source_balanced": {},
                "topology": {
                    "total_overlap_px": 0,
                    "total_euler_neq_1": 0,
                    "total_rect_gt_092": 0,
                    "total_with_holes": 0,
                    "total_multi_component": 0,
                },
                "runtime": {"mean_s_per_tile": 0.0, "total_elapsed_s": 0.0, "peak_vram_mb": 0.0},
                "per_source": {},
            }

        by_src: dict[str, list[dict[str, Any]]] = {}
        for r in self.rows:
            by_src.setdefault(r["source"], []).append(r)

        per_source: dict[str, Any] = {}
        src_r, src_pm, src_pp, src_acc, src_ap40, src_ap60 = [], [], [], [], [], []
        for src, srows in sorted(by_src.items()):
            sr = float(np.mean([x["recall"] for x in srows]))
            spm = float(np.mean([x["precision_maskaware"] for x in srows]))
            spp = float(np.mean([x["precision_plain"] for x in srows]))
            sacc = float(np.mean([x["mask_accuracy"] for x in srows]))
            sap40 = self.ap40_by_src[src].compute()
            sap60 = self.ap60_by_src[src].compute()
            sf1_m = (2.0 * sr * spm / (sr + spm)) if (sr + spm) > 0 else 0.0
            sf1_p = (2.0 * sr * spp / (sr + spp)) if (sr + spp) > 0 else 0.0
            per_source[src] = {
                "n_tiles": len(srows),
                "n_gt_crowns": int(sum(x["n_gt"] for x in srows)),
                "n_pred_crowns": int(sum(x["n_pred"] for x in srows)),
                "mask_recall": round(sr, 4),
                "maskaware_precision": round(spm, 4),
                "f1_official": round(sf1_m, 4),
                "plain_precision": round(spp, 4),
                "f1_plain": round(sf1_p, 4),
                "mask_accuracy": round(sacc, 4),
                "ap40": round(sap40, 4),
                "ap60": round(sap60, 4),
            }
            src_r.append(sr)
            src_pm.append(spm)
            src_pp.append(spp)
            src_acc.append(sacc)
            src_ap40.append(sap40)
            src_ap60.append(sap60)

        avg_r = float(np.mean(src_r))
        avg_pm = float(np.mean(src_pm))
        avg_pp = float(np.mean(src_pp))
        avg_acc = float(np.mean(src_acc))
        avg_ap40 = float(np.mean(src_ap40))
        avg_ap60 = float(np.mean(src_ap60))
        f1_off = (2.0 * avg_r * avg_pm / (avg_r + avg_pm)) if (avg_r + avg_pm) > 0 else 0.0
        f1_pln = (2.0 * avg_r * avg_pp / (avg_r + avg_pp)) if (avg_r + avg_pp) > 0 else 0.0

        return {
            "n_tiles": len(self.rows),
            "n_sources": len(by_src),
            "total_gt_crowns": int(sum(x["n_gt"] for x in self.rows)),
            "total_pred_crowns": int(sum(x["n_pred"] for x in self.rows)),
            "total_tp": int(sum(x["tp"] for x in self.rows)),
            "metrics_backend": self.ap_backend,
            "source_balanced": {
                "mask_recall": round(avg_r, 4),
                "maskaware_precision": round(avg_pm, 4),
                "f1_official": round(f1_off, 4),
                "plain_precision": round(avg_pp, 4),
                "f1_plain": round(f1_pln, 4),
                "mask_accuracy": round(avg_acc, 4),
                "ap40": round(avg_ap40, 4),
                "ap60": round(avg_ap60, 4),
            },
            "topology": {
                "total_overlap_px": int(sum(x["overlap_px"] for x in self.rows)),
                "total_euler_neq_1": int(sum(x["n_euler_neq_1"] for x in self.rows)),
                "total_rect_gt_092": int(sum(x["n_rect_gt_092"] for x in self.rows)),
                "total_with_holes": int(sum(x["n_with_holes"] for x in self.rows)),
                "total_multi_component": int(sum(x["n_multi_component"] for x in self.rows)),
            },
            "runtime": {
                "mean_s_per_tile": round(float(np.mean(self.elapsed_s_list)), 3),
                "total_elapsed_s": round(float(np.sum(self.elapsed_s_list)), 2),
                "peak_vram_mb": round(float(np.max(self.peak_vram_mb_list)), 1) if self.peak_vram_mb_list else 0.0,
            },
            "per_source": per_source,
        }


class ThreeSplitTracker:
    """Routes each Holdout_v18 tile into OOD-12, TLS Unseen-51, and TLS Complete-59."""

    def __init__(self, score_threshold: float = 0.10) -> None:
        self.splits = {
            "part_1a_ood_12": SubsetEvaluator(score_threshold=score_threshold),
            "part_1b_tls_unseen_51": SubsetEvaluator(score_threshold=score_threshold),
            "part_1b_tls_complete_59": SubsetEvaluator(score_threshold=score_threshold),
        }

    def add_tile(
        self,
        tile_meta: dict[str, Any],
        row: dict[str, Any],
        pred_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
        pred_scores: list[float],
        gt_sparse: list[tuple[tuple[int, int, int, int], np.ndarray, int]],
        H: int,
        W: int,
        elapsed_s: float,
        peak_vram_mb: float,
        pred_dense_bool: torch.Tensor | None = None,
        gt_dense_bool: torch.Tensor | None = None,
    ) -> None:
        part = tile_meta["part"]
        in_dev20 = bool(tile_meta.get("in_dev20", False))
        if part == "Holdout_OOD_12":
            self.splits["part_1a_ood_12"].add_tile(
                row, pred_sparse, pred_scores, gt_sparse, H, W, elapsed_s, peak_vram_mb,
                pred_dense_bool=pred_dense_bool, gt_dense_bool=gt_dense_bool,
            )
        else:
            self.splits["part_1b_tls_complete_59"].add_tile(
                row, pred_sparse, pred_scores, gt_sparse, H, W, elapsed_s, peak_vram_mb,
                pred_dense_bool=pred_dense_bool, gt_dense_bool=gt_dense_bool,
            )
            if not in_dev20:
                self.splits["part_1b_tls_unseen_51"].add_tile(
                    row, pred_sparse, pred_scores, gt_sparse, H, W, elapsed_s, peak_vram_mb,
                    pred_dense_bool=pred_dense_bool, gt_dense_bool=gt_dense_bool,
                )

    def summarize(self) -> dict[str, Any]:
        return {k: v.summarize() for k, v in self.splits.items()}


def _build_official_mt_metrics(score_threshold: float) -> dict[str, Any] | None:
    if not HAS_MILLIONTREES or MaskAccuracy is None or MaskAwareMaskPrecision is None:
        return None
    return {
        "accuracy": MaskAccuracy(
            iou_threshold=0.40, score_threshold=score_threshold, geometry_name="y", metric="accuracy"
        ),
        "recall": MaskAccuracy(
            iou_threshold=0.40, score_threshold=score_threshold, geometry_name="y", metric="recall"
        ),
        "maskaware_precision": MaskAwareMaskPrecision(
            iou_threshold=0.40, score_threshold=score_threshold, geometry_name="y"
        ),
    }


def _load_holdout_manifest_and_gt(
    v18_dir: Path,
    skip_sha256_check: bool = False,
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]], Path]:
    gt_locked_dir = v18_dir / "gt_locked"
    manifest_p = gt_locked_dir / "v18_holdout_manifest.json"
    gt_csv_p = gt_locked_dir / "v18_gt_annotations.csv"
    img_only_p = v18_dir / "v18_image_only_manifest.json"

    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    if not skip_sha256_check:
        if "sha256_gt_annotations_csv" in manifest:
            assert sha256_file(gt_csv_p) == manifest["sha256_gt_annotations_csv"], (
                "SHA-256 mismatch on v18_gt_annotations.csv!"
            )
        if "sha256_image_only_manifest" in manifest and img_only_p.exists():
            assert sha256_file(img_only_p) == manifest["sha256_image_only_manifest"], (
                "SHA-256 mismatch on v18_image_only_manifest.json!"
            )

    gt_by_fn: dict[str, list[dict[str, str]]] = {}
    with open(gt_csv_p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt_by_fn.setdefault(row["filename"], []).append(row)
    return manifest, gt_by_fn, gt_locked_dir


def _print_comparison_table(title: str, configs: list[tuple[str, dict[str, Any]]]) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'Config':34s} | {'F1':>6s} | {'R':>6s} | {'MaskP':>6s} | {'PlainP':>6s} | "
        f"{'Acc':>6s} | {'AP40':>6s} | {'AP60':>6s} | {'OvPx':>4s} | {'Eul!1':>5s} | "
        f"{'Rect':>4s} | {'s/tile':>6s} | {'VRAM_MB':>7s}"
    )
    print(header)
    print("-" * len(header))
    for label, summary in configs:
        if summary.get("n_tiles", 0) == 0:
            print(f"{label:34s} | (0 tiles)")
            continue
        sb = summary["source_balanced"]
        tp = summary["topology"]
        rt = summary["runtime"]
        print(
            f"{label:34s} | {sb['f1_official']:6.4f} | {sb['mask_recall']:6.4f} | "
            f"{sb['maskaware_precision']:6.4f} | {sb['plain_precision']:6.4f} | "
            f"{sb['mask_accuracy']:6.4f} | {sb['ap40']:6.4f} | {sb['ap60']:6.4f} | "
            f"{tp['total_overlap_px']:4d} | {tp['total_euler_neq_1']:5d} | "
            f"{tp['total_rect_gt_092']:4d} | {rt['mean_s_per_tile']:6.2f} | {rt['peak_vram_mb']:7.1f}"
        )


# ==============================================================================
# 3. Mode 1: parity-native (Cached NPZ Verification)
# ==============================================================================


def run_parity_native(
    dev20_dir: Path,
    v18_dir: Path,
    out_json: Path,
    score_threshold: float = 0.10,
    max_tiles: int | None = None,
    skip_sha256_check: bool = False,
) -> dict[str, Any]:
    t_all = time.perf_counter()
    manifest, gt_by_fn, gt_locked_dir = _load_holdout_manifest_and_gt(
        v18_dir, skip_sha256_check=skip_sha256_check
    )
    tiles = manifest["tiles"]
    if max_tiles is not None:
        tiles = tiles[: max(1, int(max_tiles))]

    mt_metrics = _build_official_mt_metrics(score_threshold)
    backend_name = "milliontrees" if HAS_MILLIONTREES else ("torchmetrics" if HAS_TORCHMETRICS else "numpy_fallback")
    print(
        f"[parity-native] Evaluating {len(tiles)} Holdout_v18 tiles | "
        f"metrics_backend={backend_name} | score_threshold={score_threshold}"
    )

    tr_a = ThreeSplitTracker(score_threshold=score_threshold)
    tr_b = ThreeSplitTracker(score_threshold=score_threshold)
    tr_c = ThreeSplitTracker(score_threshold=score_threshold)

    total_delta_mask_px_ab = 0
    needs_dense_for_ap = HAS_MILLIONTREES or HAS_TORCHMETRICS

    for idx, t in enumerate(tiles, start=1):
        sid = t["short_id"]
        fn = t["filename"]
        in_dev20 = bool(t["in_dev20"])
        gsd_m = t["gsd_m"]

        img_p = (dev20_dir / "images" / fn) if in_dev20 else (v18_dir / "images" / fn)
        msk_p = (
            (dev20_dir / "masks" / f"{Path(fn).stem}.png")
            if in_dev20
            else (gt_locked_dir / "masks" / f"{Path(fn).stem}.png")
        )
        npz_p = (dev20_dir / "frozen_cache" / f"{sid}.npz") if in_dev20 else (v18_dir / "frozen_cache" / f"{sid}.npz")

        img_rgb = np.asarray(Image.open(img_p).convert("RGB"), dtype=np.uint8)
        H, W = img_rgb.shape[:2]
        tm_bool = (np.asarray(Image.open(msk_p).convert("L"), dtype=np.uint8) > 0) if msk_p.exists() else None
        gt_sparse = load_gt_sparse_native(H, W, gt_by_fn.get(fn, []))
        gt_boxes, gt_224 = sparse_to_224_and_boxes(gt_sparse, H, W)
        gt_dense = sparse_list_to_dense_bool(gt_sparse, H, W) if needs_dense_for_ap else None

        with np.load(npz_p, allow_pickle=False) as npz_cache:
            # (a) gsd_m=manifest, remap=False
            t0 = time.perf_counter()
            sp_a, sc_a, meta_a, _ = CanopyGISv18Predictor.predict_from_cached_npz_sparse(
                img_rgb, npz_cache, gsd_m=gsd_m, remap_ws_scores=False
            )
            dt_a = time.perf_counter() - t0

            # (b) gsd_m=manifest, remap=True (verify exact zero-pixel mask invariance!)
            t0 = time.perf_counter()
            sp_b, sc_b, _meta_b, _ = CanopyGISv18Predictor.predict_from_cached_npz_sparse(
                img_rgb, npz_cache, gsd_m=gsd_m, remap_ws_scores=True
            )
            dt_b = time.perf_counter() - t0

            # (c) gsd_m=None, remap=True (Decision 3A pure-image parity)
            t0 = time.perf_counter()
            sp_c, sc_c, _meta_c, _ = CanopyGISv18Predictor.predict_from_cached_npz_sparse(
                img_rgb, npz_cache, gsd_m=None, remap_ws_scores=True
            )
            dt_c = time.perf_counter() - t0

        delta_px = compute_sparse_mask_delta_px(sp_a, sp_b, H, W)
        total_delta_mask_px_ab += delta_px
        assert delta_px == 0, f"Decision 2A violated on {sid}: DeltaMask = {delta_px} px != 0!"
        # Also verify direct remap_post_carve_scores parity
        assert sc_b == remap_post_carve_scores(sc_a, meta_a), f"Score remap mismatch on {sid}!"

        pred_dense_ab = sparse_list_to_dense_bool(sp_a, H, W) if needs_dense_for_ap else None
        row_a = evaluate_single_tile_elementwise(
            t, gt_sparse, gt_boxes, gt_224, tm_bool, sp_a, sc_a, H, W,
            score_threshold=score_threshold,
            gt_dense_bool=gt_dense, pred_dense_bool=pred_dense_ab, mt_metrics=mt_metrics,
        )
        row_b = evaluate_single_tile_elementwise(
            t, gt_sparse, gt_boxes, gt_224, tm_bool, sp_b, sc_b, H, W,
            score_threshold=score_threshold,
            gt_dense_bool=gt_dense, pred_dense_bool=pred_dense_ab, mt_metrics=mt_metrics,
        )
        tr_a.add_tile(
            t, row_a, sp_a, sc_a, gt_sparse, H, W, dt_a, 0.0,
            pred_dense_bool=pred_dense_ab, gt_dense_bool=gt_dense,
        )
        tr_b.add_tile(
            t, row_b, sp_b, sc_b, gt_sparse, H, W, dt_b, 0.0,
            pred_dense_bool=pred_dense_ab, gt_dense_bool=gt_dense,
        )
        del pred_dense_ab

        pred_dense_c = sparse_list_to_dense_bool(sp_c, H, W) if needs_dense_for_ap else None
        row_c = evaluate_single_tile_elementwise(
            t, gt_sparse, gt_boxes, gt_224, tm_bool, sp_c, sc_c, H, W,
            score_threshold=score_threshold,
            gt_dense_bool=gt_dense, pred_dense_bool=pred_dense_c, mt_metrics=mt_metrics,
        )
        tr_c.add_tile(
            t, row_c, sp_c, sc_c, gt_sparse, H, W, dt_c, 0.0,
            pred_dense_bool=pred_dense_c, gt_dense_bool=gt_dense,
        )
        del pred_dense_c, gt_dense

        if H * W > 1500 * 1500:
            gc.collect()

        if idx % 10 == 0 or t["part"] == "Holdout_OOD_12" or idx == len(tiles):
            print(
                f"  [{idx:2d}/{len(tiles)}] {t['part']:15s} | {sid:18s} | "
                f"a(raw) R={row_a['recall']:.3f} P={row_a['precision_maskaware']:.3f} | "
                f"c(noGSD) R={row_c['recall']:.3f} P={row_c['precision_maskaware']:.3f} | "
                f"dMask={delta_px}",
                flush=True,
            )

    sum_a = tr_a.summarize()
    sum_b = tr_b.summarize()
    sum_c = tr_c.summarize()

    # Assertions & Parity Checks
    delta_f1_ab: dict[str, float] = {}
    delta_ap_ab: dict[str, dict[str, float]] = {}
    delta_c_vs_b: dict[str, dict[str, float]] = {}
    parity_checks: dict[str, Any] = {}

    for sec_k in ("part_1a_ood_12", "part_1b_tls_unseen_51", "part_1b_tls_complete_59"):
        sa, sb, sc = sum_a[sec_k], sum_b[sec_k], sum_c[sec_k]
        if sa["n_tiles"] == 0:
            continue
        for tag, s_obj in (("a", sa), ("b", sb), ("c", sc)):
            assert s_obj["topology"]["total_overlap_px"] == 0, f"Overlap > 0 in {sec_k}/{tag}: {s_obj['topology']}"
            assert s_obj["topology"]["total_euler_neq_1"] == 0, f"Euler != 1 in {sec_k}/{tag}: {s_obj['topology']}"
            assert s_obj["topology"]["total_rect_gt_092"] == 0, f"Rect > 0.92 in {sec_k}/{tag}: {s_obj['topology']}"

        df1 = round(sb["source_balanced"]["f1_official"] - sa["source_balanced"]["f1_official"], 6)
        delta_f1_ab[sec_k] = df1
        assert abs(df1) == 0.0, f"Decision 2A violated on {sec_k}: DeltaF1 = {df1} != 0.0000!"

        delta_ap_ab[sec_k] = {
            "delta_ap40": round(sb["source_balanced"]["ap40"] - sa["source_balanced"]["ap40"], 4),
            "delta_ap60": round(sb["source_balanced"]["ap60"] - sa["source_balanced"]["ap60"], 4),
        }
        delta_c_vs_b[sec_k] = {
            "delta_f1": round(sc["source_balanced"]["f1_official"] - sb["source_balanced"]["f1_official"], 4),
            "delta_ap40": round(sc["source_balanced"]["ap40"] - sb["source_balanced"]["ap40"], 4),
            "delta_ap60": round(sc["source_balanced"]["ap60"] - sb["source_balanced"]["ap60"], 4),
        }

        if max_tiles is None and sec_k in LOCKED_PARITY_F1_TARGETS:
            target = LOCKED_PARITY_F1_TARGETS[sec_k]
            actual = sa["source_balanced"]["f1_official"]
            diff = round(actual - target, 4)
            passed = abs(diff) <= 1e-4
            parity_checks[sec_k] = {"target_f1": target, "actual_f1": actual, "diff": diff, "pass": passed}
            assert passed, f"Parity mismatch on {sec_k}: actual={actual:.4f} vs target={target:.4f} (diff={diff:+.4f})"

    for title, sec_k in [
        ("PART 1A: Holdout_OOD_12", "part_1a_ood_12"),
        ("PART 1B-BLIND: Official TLS Unseen-51", "part_1b_tls_unseen_51"),
        ("PART 1B-FULL: Official TLS Complete-59", "part_1b_tls_complete_59"),
    ]:
        _print_comparison_table(
            f"[parity-native] {title}",
            [
                ("(a) gsd=manifest, remap=False", sum_a[sec_k]),
                ("(b) gsd=manifest, remap=True", sum_b[sec_k]),
                ("(c) gsd=None,     remap=True", sum_c[sec_k]),
            ],
        )
        if sec_k in delta_ap_ab:
            print(
                f"  Decision 2A Delta (b - a): DeltaMask={total_delta_mask_px_ab} px, "
                f"DeltaF1={delta_f1_ab[sec_k]:+.4f}, "
                f"DeltaAP40={delta_ap_ab[sec_k]['delta_ap40']:+.4f}, "
                f"DeltaAP60={delta_ap_ab[sec_k]['delta_ap60']:+.4f}"
            )
            print(
                f"  Decision 3A Delta (c - b): "
                f"DeltaF1={delta_c_vs_b[sec_k]['delta_f1']:+.4f}, "
                f"DeltaAP40={delta_c_vs_b[sec_k]['delta_ap40']:+.4f}, "
                f"DeltaAP60={delta_c_vs_b[sec_k]['delta_ap60']:+.4f}"
            )

    receipt = {
        "mode": "parity-native",
        "n_tiles_evaluated": len(tiles),
        "metrics_backend": backend_name,
        "score_threshold": score_threshold,
        "elapsed_s": round(time.perf_counter() - t_all, 2),
        "decision_2a_invariants": {
            "total_delta_mask_px": total_delta_mask_px_ab,
            "delta_f1_by_split": delta_f1_ab,
            "delta_ap_by_split": delta_ap_ab,
        },
        "decision_3a_nogsd_vs_manifest": delta_c_vs_b,
        "locked_v18_parity_checks": parity_checks,
        "configs": {
            "a_manifest_gsd_raw": sum_a,
            "b_manifest_gsd_remapped": sum_b,
            "c_nogsd_remapped": sum_c,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\nSaved parity-native receipt -> {out_json}")
    return receipt


# ==============================================================================
# 4. Mode 2: ab-gate (Live GPU Mode A 1024 Direct vs Mode B Native -> 1024)
# ==============================================================================


def _eval_ab_pass(
    pass_name: str,
    mode_kind: str,
    predictor: CanopyGISv18Predictor,
    tiles: list[dict[str, Any]],
    dev20_dir: Path,
    v18_dir: Path,
    gt_locked_dir: Path,
    gt_by_fn: dict[str, list[dict[str, str]]],
    score_threshold: float,
    mt_metrics: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tr_remap = ThreeSplitTracker(score_threshold=score_threshold)
    tr_raw = ThreeSplitTracker(score_threshold=score_threshold)
    use_cuda = predictor.device.type == "cuda"
    needs_dense = HAS_MILLIONTREES or HAS_TORCHMETRICS
    target_size = 1024

    if use_cuda:
        torch.cuda.empty_cache()

    print(f"\n[{pass_name}] Running live inference on {len(tiles)} tiles (device={predictor.device})...")
    for idx, t in enumerate(tiles, start=1):
        sid = t["short_id"]
        fn = t["filename"]
        in_dev20 = bool(t["in_dev20"])

        img_p = (dev20_dir / "images" / fn) if in_dev20 else (v18_dir / "images" / fn)
        msk_p = (
            (dev20_dir / "masks" / f"{Path(fn).stem}.png")
            if in_dev20
            else (gt_locked_dir / "masks" / f"{Path(fn).stem}.png")
        )

        img_rgb = np.asarray(Image.open(img_p).convert("RGB"), dtype=np.uint8)
        orig_h, orig_w = img_rgb.shape[:2]

        # Identical 1024x1024 target preprocessing to TreePolygonsDataset(..., image_size=1024)
        if orig_h != target_size or orig_w != target_size:
            img_1024 = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_1024 = img_rgb

        if msk_p.exists():
            tm_orig = np.asarray(Image.open(msk_p).convert("L"), dtype=np.uint8)
            if tm_orig.shape[:2] != (target_size, target_size):
                tm_1024 = cv2.resize(tm_orig, (target_size, target_size), interpolation=cv2.INTER_NEAREST) > 0
            else:
                tm_1024 = tm_orig > 0
        else:
            tm_1024 = None

        gt_1024_sparse = load_gt_sparse_resized_tree_polygons(
            orig_h, orig_w, target_size, target_size, gt_by_fn.get(fn, [])
        )
        gt_boxes, gt_224 = sparse_to_224_and_boxes(gt_1024_sparse, target_size, target_size)
        gt_dense = sparse_list_to_dense_bool(gt_1024_sparse, target_size, target_size) if needs_dense else None

        if use_cuda:
            torch.cuda.reset_peak_memory_stats(predictor.device)
            torch.cuda.synchronize(predictor.device)
        t0 = time.perf_counter()

        if mode_kind == "mode_a":
            sp_eval, sc_raw, metas_eval, _ = predictor.predict_tile_sparse(
                img_1024, gsd_m=None, remap_ws_scores=False
            )
            sc_remap = remap_post_carve_scores(sc_raw, metas_eval)
        else:
            sp_nat, sc_nat_raw, metas_nat, _ = predictor.predict_tile_sparse(
                img_rgb, gsd_m=None, remap_ws_scores=False
            )
            sp_eval, sc_raw, metas_eval = resize_planar_predictions(
                sp_nat,
                sc_nat_raw,
                metas_nat,
                src_hw=(orig_h, orig_w),
                dst_hw=(target_size, target_size),
                min_area_px=16,
            )
            sc_remap = remap_post_carve_scores(sc_raw, metas_eval)

        if use_cuda:
            torch.cuda.synchronize(predictor.device)
            peak_vram_mb = float(torch.cuda.max_memory_allocated(predictor.device)) / (1024.0 * 1024.0)
        else:
            peak_vram_mb = 0.0
        dt = time.perf_counter() - t0

        # Verify public tensor-dict contract on tile 1 (and empty-safe contract)
        if idx == 1:
            td = sparse_predictions_to_tensor_dict(
                sp_eval, sc_remap, hw=(target_size, target_size), score_threshold=score_threshold
            )
            assert td["y"].dtype == torch.uint8 and td["y"].shape[1:] == (target_size, target_size)
            assert td["labels"].dtype == torch.int64 and td["scores"].dtype == torch.float32
            assert td["y"].shape[0] == td["labels"].shape[0] == td["scores"].shape[0]
            del td

        pred_dense = sparse_list_to_dense_bool(sp_eval, target_size, target_size) if needs_dense else None

        row_remap = evaluate_single_tile_elementwise(
            t, gt_1024_sparse, gt_boxes, gt_224, tm_1024, sp_eval, sc_remap, target_size, target_size,
            score_threshold=score_threshold,
            gt_dense_bool=gt_dense, pred_dense_bool=pred_dense, mt_metrics=mt_metrics,
        )
        row_raw = evaluate_single_tile_elementwise(
            t, gt_1024_sparse, gt_boxes, gt_224, tm_1024, sp_eval, sc_raw, target_size, target_size,
            score_threshold=score_threshold,
            gt_dense_bool=gt_dense, pred_dense_bool=pred_dense, mt_metrics=mt_metrics,
        )

        tr_remap.add_tile(
            t, row_remap, sp_eval, sc_remap, gt_1024_sparse, target_size, target_size, dt, peak_vram_mb,
            pred_dense_bool=pred_dense, gt_dense_bool=gt_dense,
        )
        tr_raw.add_tile(
            t, row_raw, sp_eval, sc_raw, gt_1024_sparse, target_size, target_size, dt, peak_vram_mb,
            pred_dense_bool=pred_dense, gt_dense_bool=gt_dense,
        )

        del pred_dense, gt_dense
        if idx % 10 == 0 or t["part"] == "Holdout_OOD_12" or idx == len(tiles):
            print(
                f"  [{idx:2d}/{len(tiles)}] {t['part']:15s} | {sid:18s} | "
                f"R={row_remap['recall']:.3f} MaskP={row_remap['precision_maskaware']:.3f} "
                f"PlainP={row_remap['precision_plain']:.3f} | Ov={row_remap['overlap_px']} "
                f"Eul!1={row_remap['n_euler_neq_1']} | {dt:.2f}s | {peak_vram_mb:.0f} MB",
                flush=True,
            )

    return tr_remap.summarize(), tr_raw.summarize()


def run_ab_gate(
    dev20_dir: Path,
    v18_dir: Path,
    out_json: Path,
    device: str = "auto",
    model_id: str = "facebook/sam3",
    hf_token: str | None = None,
    score_threshold: float = 0.10,
    max_tiles: int | None = None,
    skip_sha256_check: bool = False,
) -> dict[str, Any]:
    t_all = time.perf_counter()
    manifest, gt_by_fn, gt_locked_dir = _load_holdout_manifest_and_gt(
        v18_dir, skip_sha256_check=skip_sha256_check
    )
    tiles = manifest["tiles"]
    if max_tiles is not None:
        tiles = tiles[: max(1, int(max_tiles))]

    mt_metrics = _build_official_mt_metrics(score_threshold)
    backend_name = "milliontrees" if HAS_MILLIONTREES else ("torchmetrics" if HAS_TORCHMETRICS else "numpy_fallback")
    print(
        f"[ab-gate] Loading CanopyGISv18Predictor ({model_id}, device={device}) | "
        f"{len(tiles)} tiles | metrics_backend={backend_name}"
    )
    predictor = CanopyGISv18Predictor.from_pretrained(model_id=model_id, device=device, hf_token=hf_token)

    mode_a_remap, mode_a_raw = _eval_ab_pass(
        "Mode A (1024 direct)",
        "mode_a",
        predictor,
        tiles,
        dev20_dir,
        v18_dir,
        gt_locked_dir,
        gt_by_fn,
        score_threshold,
        mt_metrics,
    )
    mode_b_remap, mode_b_raw = _eval_ab_pass(
        "Mode B (native -> 1024 INTER_NEAREST)",
        "mode_b",
        predictor,
        tiles,
        dev20_dir,
        v18_dir,
        gt_locked_dir,
        gt_by_fn,
        score_threshold,
        mt_metrics,
    )

    # Assert strict planar & topological invariants across all modes & splits
    for tag, summ in (
        ("mode_a_remap", mode_a_remap),
        ("mode_a_raw", mode_a_raw),
        ("mode_b_remap", mode_b_remap),
        ("mode_b_raw", mode_b_raw),
    ):
        for sec_k, sec_v in summ.items():
            if sec_v["n_tiles"] == 0:
                continue
            assert sec_v["topology"]["total_overlap_px"] == 0, f"Overlap > 0 in {tag}/{sec_k}"
            assert sec_v["topology"]["total_euler_neq_1"] == 0, f"Euler != 1 in {tag}/{sec_k}"

    mode_a_ap_delta: dict[str, dict[str, float]] = {}
    mode_b_vs_a_delta: dict[str, dict[str, float]] = {}
    for title, sec_k in [
        ("PART 1A: Holdout_OOD_12 (1024x1024 Eval)", "part_1a_ood_12"),
        ("PART 1B-BLIND: Official TLS Unseen-51 (1024x1024 Eval)", "part_1b_tls_unseen_51"),
        ("PART 1B-FULL: Official TLS Complete-59 (1024x1024 Eval)", "part_1b_tls_complete_59"),
    ]:
        _print_comparison_table(
            f"[ab-gate] {title}",
            [
                ("Mode A (1024 direct, remap=True)", mode_a_remap[sec_k]),
                ("Mode A (1024 direct, remap=False)", mode_a_raw[sec_k]),
                ("Mode B (native->1024, remap=True)", mode_b_remap[sec_k]),
                ("Mode B (native->1024, remap=False)", mode_b_raw[sec_k]),
            ],
        )
        if mode_a_remap[sec_k]["n_tiles"] > 0:
            sb_a_rem = mode_a_remap[sec_k]["source_balanced"]
            sb_a_raw = mode_a_raw[sec_k]["source_balanced"]
            sb_b_rem = mode_b_remap[sec_k]["source_balanced"]
            da40 = round(sb_a_rem["ap40"] - sb_a_raw["ap40"], 4)
            da60 = round(sb_a_rem["ap60"] - sb_a_raw["ap60"], 4)
            df1_a = round(sb_a_rem["f1_official"] - sb_a_raw["f1_official"], 6)
            assert abs(df1_a) == 0.0, f"Mode A Decision 2A violated on {sec_k}: DeltaF1 = {df1_a}"
            mode_a_ap_delta[sec_k] = {"delta_f1": df1_a, "delta_ap40": da40, "delta_ap60": da60}

            mode_b_vs_a_delta[sec_k] = {
                "delta_f1": round(sb_b_rem["f1_official"] - sb_a_rem["f1_official"], 4),
                "delta_ap40": round(sb_b_rem["ap40"] - sb_a_rem["ap40"], 4),
                "delta_ap60": round(sb_b_rem["ap60"] - sb_a_rem["ap60"], 4),
            }
            print(
                f"  Mode A Remap Delta: DeltaF1={df1_a:+.4f}, DeltaAP40={da40:+.4f}, DeltaAP60={da60:+.4f} | "
                f"Mode B - Mode A: DeltaF1={mode_b_vs_a_delta[sec_k]['delta_f1']:+.4f}, "
                f"DeltaAP40={mode_b_vs_a_delta[sec_k]['delta_ap40']:+.4f}"
            )

    receipt = {
        "mode": "ab-gate",
        "device": str(predictor.device),
        "model_id": model_id,
        "n_tiles_evaluated": len(tiles),
        "metrics_backend": backend_name,
        "score_threshold": score_threshold,
        "elapsed_s": round(time.perf_counter() - t_all, 2),
        "mode_a_remap_ap_delta": mode_a_ap_delta,
        "mode_b_minus_mode_a": mode_b_vs_a_delta,
        "configs": {
            "mode_a_1024_direct_remapped": mode_a_remap,
            "mode_a_1024_direct_raw": mode_a_raw,
            "mode_b_native_to_1024_remapped": mode_b_remap,
            "mode_b_native_to_1024_raw": mode_b_raw,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\nSaved ab-gate receipt -> {out_json}")
    return receipt


def main() -> None:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Verify CanopyGIS v1.8.1 native parity, Decision 2A AP40/AP60 delta, and live 1024 A/B gate."
    )
    parser.add_argument("--mode", choices=["parity-native", "ab-gate"], required=True)
    parser.add_argument(
        "--dev20-dir", type=str, default=None, help="Path to Dev-20 dataset directory (or CANOPYGIS_DEV20_DIR)."
    )
    parser.add_argument(
        "--v18-dir", type=str, default=None, help="Path to Holdout_v18 directory (or CANOPYGIS_V18_DIR)."
    )
    parser.add_argument("--out-json", type=str, default=None, help="Output JSON receipt path.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--model-id", type=str, default="facebook/sam3")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--max-tiles", type=int, default=None, help="Optional smoke limit on number of tiles.")
    parser.add_argument("--skip-sha256-check", action="store_true", help="Skip SHA-256 verification on manifest files.")
    args = parser.parse_args()

    dev20_dir = _resolve_dataset_dir(args.dev20_dir, "CANOPYGIS_DEV20_DIR", Path("benchmarks/mt_v16"))
    v18_dir = _resolve_dataset_dir(args.v18_dir, "CANOPYGIS_V18_DIR", Path("benchmarks/mt_v18"))
    default_out = REPO_ROOT / f"v181_{args.mode.replace('-', '_')}_receipt.json"
    out_json = Path(args.out_json) if args.out_json else default_out

    if args.mode == "parity-native":
        run_parity_native(
            dev20_dir=dev20_dir,
            v18_dir=v18_dir,
            out_json=out_json,
            score_threshold=args.score_threshold,
            max_tiles=args.max_tiles,
            skip_sha256_check=args.skip_sha256_check,
        )
    else:
        run_ab_gate(
            dev20_dir=dev20_dir,
            v18_dir=v18_dir,
            out_json=out_json,
            device=args.device,
            model_id=args.model_id,
            hf_token=args.hf_token,
            score_threshold=args.score_threshold,
            max_tiles=args.max_tiles,
            skip_sha256_check=args.skip_sha256_check,
        )


if __name__ == "__main__":
    main()
