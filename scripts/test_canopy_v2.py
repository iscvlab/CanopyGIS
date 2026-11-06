"""Small dependency-light checks for Canopy V2 tiling and reconciliation."""

import os
import sys

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from canopy_v2 import (  # noqa: E402
    DISTINCT_CROWN_PALETTE,
    assign_adjacent_distinct_colors,
    carve_non_overlapping_crowns,
    compute_crown_confidence,
    compute_strict_foliage_mask,
    compute_woody_canopy_relief_mask,
    deduplicate_contours_iom,
    enhance_shadow_luminance_cielab,
    iter_tile_windows,
    reconcile_canopy_outputs,
    run_deepforest_crowns,
    run_sam3_tiled,
    split_dense_canopy_watershed,
    visible_vegetation_support,
)
from main import (  # noqa: E402
    _image_only_preview_bounds,
    _normalize_preset_key,
    build_roi_pixel_mask,
    get_dynamic_zoom,
    get_latest_session,
    health_models,
    parse_roi_polygon,
    process_contour_enhanced,
)


assert get_dynamic_zoom(120, 120) == 22
assert get_dynamic_zoom(200, 200) == 21
assert get_dynamic_zoom(360, 300) == 21
assert get_dynamic_zoom(711, 422) == 20
assert get_dynamic_zoom(1200, 1200) == 20
assert get_dynamic_zoom(2000, 2000) == 19

preview_bounds = _image_only_preview_bounds(1000, 1000, assumed_gsd_m=0.10)
assert abs(((preview_bounds["south"] + preview_bounds["north"]) / 2.0) - 13.8785514) < 1e-6
assert abs(((preview_bounds["west"] + preview_bounds["east"]) / 2.0) - 100.6195068) < 1e-6

# Verify deduplicate_contours_iom suppresses overlapping duplicate contours
dup_mask_1 = np.zeros((160, 160), dtype=np.uint8)
dup_mask_2 = np.zeros((160, 160), dtype=np.uint8)
distinct_mask = np.zeros((160, 160), dtype=np.uint8)
cv2.circle(dup_mask_1, (50, 50), 25, 255, -1)
cv2.circle(dup_mask_2, (52, 51), 24, 255, -1)
cv2.circle(distinct_mask, (120, 120), 22, 255, -1)
cnt_1 = cv2.findContours(dup_mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
cnt_2 = cv2.findContours(dup_mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
cnt_3 = cv2.findContours(distinct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]

deduped = deduplicate_contours_iom(
    [
        {"contour": cnt_2, "score": 0.72},
        {"contour": cnt_1, "score": 0.91},
        {"contour": cnt_3, "score": 0.84},
    ],
    iom_threshold=0.65,
)
assert len(deduped) == 2
assert deduped[0]["score"] == 0.91
assert deduped[1]["score"] == 0.84

# Verify carve_non_overlapping_crowns guarantees 0 pixel overlap between output contours
ov_mask_a = np.zeros((160, 160), dtype=np.uint8)
ov_mask_b = np.zeros((160, 160), dtype=np.uint8)
ov_mask_c = np.zeros((160, 160), dtype=np.uint8)
cv2.circle(ov_mask_a, (60, 80), 26, 255, -1)
cv2.circle(ov_mask_b, (95, 80), 26, 255, -1)
cv2.circle(ov_mask_c, (62, 81), 24, 255, -1)
ov_cnt_a = cv2.findContours(ov_mask_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
ov_cnt_b = cv2.findContours(ov_mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
ov_cnt_c = cv2.findContours(ov_mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]

carved_items, _ = carve_non_overlapping_crowns(
    [
        {"contour": ov_cnt_a, "score": 0.92},
        {"contour": ov_cnt_b, "score": 0.85},
        {"contour": ov_cnt_c, "score": 0.60},
    ],
    height=160,
    width=160,
    iom_reject=0.28,
    min_area_px=25,
)
assert len(carved_items) == 2
m0 = np.zeros((160, 160), dtype=np.uint8)
m1 = np.zeros((160, 160), dtype=np.uint8)
cv2.drawContours(m0, [carved_items[0]["contour"]], -1, 255, -1)
cv2.drawContours(m1, [carved_items[1]["contour"]], -1, 255, -1)
assert cv2.countNonZero(cv2.bitwise_and(m0, m1)) == 0

image = np.full((300, 420, 3), 100, dtype=np.uint8)
rng = np.random.default_rng(7)
texture = rng.integers(-25, 26, size=(300, 420, 1), dtype=np.int16)
green = np.array([45, 140, 50], dtype=np.int16) + texture
green = np.clip(green, 0, 255).astype(np.uint8)
mask_a = np.zeros(image.shape[:2], dtype=np.uint8)
mask_b = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.circle(mask_a, (100, 150), 60, 255, -1)
cv2.circle(mask_b, (260, 150), 55, 255, -1)
image[mask_a > 0] = green[mask_a > 0]
image[mask_b > 0] = green[mask_b > 0]

windows = list(iter_tile_windows(420, 300, 128, 24))
assert windows[0]["core"][:2] == (0, 0)
assert windows[-1]["core"][2:] == (420, 300)

probability, support = visible_vegetation_support(image, 128, 24, 0.4)
prob_single, support_single = visible_vegetation_support(image, 512, 32, 0.4)
assert support[150, 100] == 255
assert support[10, 10] == 0
assert probability[150, 100] > probability[10, 10]
# Global hysteresis produces identical seam-free support regardless of tile size
assert np.array_equal(probability, prob_single)
assert np.array_equal(support, support_single)


class _DummySam:
    device = "cpu"

    def __init__(self):
        self.masks = None
        self.boxes = None
        self.scores = None
        self.calls = 0
        self.last_prompt = None

    def set_confidence_threshold(self, confidence: float):
        self.confidence = confidence

    def set_image(self, tile: np.ndarray):
        self._tile_shape = tile.shape[:2]

    def generate_masks(self, prompt: str, min_size: int = 80, quiet: bool = True):
        self.calls += 1
        self.last_prompt = prompt
        h, w = self._tile_shape
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(m, (w // 2, h // 2), 18, 1, -1)
        self.masks = [m]
        self.scores = [0.88]


dummy_sam = _DummySam()
sam_instances, sam_tiled_union, sam_tiled_stats = run_sam3_tiled(
    image,
    dummy_sam,
    prompt="tree",
    tile_size=96,
    overlap=16,
    confidence=0.25,
    min_area_px=30,
    max_area_px=20000,
    vegetation_support=support,
    probability_map=probability,
)
assert dummy_sam.last_prompt == "trees"
assert "skipped_barren_tiles" in sam_tiled_stats
assert sam_tiled_stats["skipped_barren_tiles"] >= 1
assert len(sam_instances) >= 1

# Verify roof rejection gate filters non-green masks
roof_img = np.full((128, 128, 3), 230, dtype=np.uint8)
roof_instances, roof_union, _ = run_sam3_tiled(
    roof_img,
    dummy_sam,
    prompt="tree",
    tile_size=128,
    overlap=16,
    confidence=0.18,
    min_area_px=30,
    max_area_px=20000,
)
assert len(roof_instances) == 0
assert cv2.countNonZero(roof_union) == 0

sam_union = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.circle(sam_union, (100, 150), 55, 255, -1)
coverage, groups, stats = reconcile_canopy_outputs(
    image, support, sam_union, min_group_area_px=100
)
assert cv2.countNonZero(coverage) >= cv2.countNonZero(sam_union)
assert stats["sam_pixels"] == cv2.countNonZero(sam_union)
assert len(groups) >= 1

dense_candidates, dense_stats = split_dense_canopy_watershed(
    image,
    support,
    [],
    gsd_m=0.10,
    tile_size=256,
    overlap=32,
    min_dense_ratio=0.05,
    min_area_px=100,
    max_area_px=20000,
)
assert dense_stats["method"] == "marker_controlled_rgb_watershed"
assert dense_stats["spacing_px"] == 55
assert all(cv2.contourArea(item["contour"]) >= 50 for item in dense_candidates)

# Verify DeepForest 2.1.0 crown verification and Watershed priority seeding
df_none_boxes, df_none_mask, df_none_stats = run_deepforest_crowns(image, None)
assert df_none_boxes == []
assert cv2.countNonZero(df_none_mask) == 0
assert df_none_stats == {"df_boxes": 0, "df_verified": 0}


class _DummyDeepForest:
    def predict_tile(self, image, patch_size=600, patch_overlap=0.25, iou_threshold=0.18):
        return [
            {"xmin": 70, "ymin": 120, "xmax": 130, "ymax": 180, "score": 0.62},
            {"xmin": 230, "ymin": 120, "xmax": 290, "ymax": 180, "score": 0.58},
            {"xmin": 5, "ymin": 5, "xmax": 40, "ymax": 40, "score": 0.70},
        ]


df_boxes, df_support_mask, df_stats = run_deepforest_crowns(
    image,
    _DummyDeepForest(),
    vegetation_support=support,
    probability_map=probability,
    gsd_m=0.10,
)
assert df_stats["df_boxes"] == 3
assert df_stats["df_verified"] == 2
assert cv2.countNonZero(df_support_mask) > 0

ws_with_df, ws_df_stats = split_dense_canopy_watershed(
    image,
    support,
    [],
    gsd_m=0.10,
    tile_size=256,
    overlap=32,
    min_dense_ratio=0.05,
    min_area_px=100,
    max_area_px=20000,
    deepforest_boxes=df_boxes,
)
assert len(ws_with_df) >= 2
assert any(
    60.0 <= item["confidence_pct"] <= 79.5
    and item["confidence_tier"] == "good_treetop"
    for item in ws_with_df
)

# Verify strict foliage mask carves out bare brown dirt inside forest & cars
clearing_img = np.full((120, 120, 3), [38, 128, 42], dtype=np.uint8)
cv2.circle(clearing_img, (60, 60), 22, (118, 96, 72), -1)  # Bare brown dirt clearing (R > G)
strict_fol = compute_strict_foliage_mask(clearing_img)
assert strict_fol[10, 10] == 255
assert strict_fol[60, 60] == 0

# Verify Negative Vehicle Exclusion Pass ("car") vetoes parked cars in run_sam3_tiled
class _CarTreeSam(_DummySam):
    def generate_masks(self, prompt: str, min_size: int = 80, quiet: bool = True):
        self.calls += 1
        self.last_prompt = prompt
        h, w = self._tile_shape
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(m, (w // 2, h // 2), 14, 1, -1)
        self.masks = [m]
        self.scores = [0.85]


road_car_img = np.full((128, 128, 3), 90, dtype=np.uint8)
cv2.circle(road_car_img, (64, 64), 14, (72, 78, 66), -1)  # Olive-grey car
car_sam = _CarTreeSam()
car_inst, car_union, car_stats = run_sam3_tiled(
    road_car_img,
    car_sam,
    prompt="tree",
    tile_size=128,
    overlap=16,
    confidence=0.20,
    min_area_px=30,
    max_area_px=20000,
    gsd_m=0.10,
)
assert len(car_inst) == 0
assert car_stats.get("vehicle_vetoed", 0) >= 1

# Verify process_contour_enhanced keeps dark-green and sunlit green crowns,
# rejects bare brown dirt / yellow cars (G <= R) and open water, and calibrates confidence_pct
test_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
yy_t, xx_t = np.mgrid[0:50, 0:50]
# Add realistic 3D dome shading + branch shadow relief to both tree crown quadrants
dome_1 = np.clip(
    24.0 - 1.1 * np.hypot(xx_t - 25.0, yy_t - 25.0) + 22.0 * np.sin(xx_t * 1.1) * np.cos(yy_t * 1.1),
    -25.0,
    35.0,
)
test_rgb[:50, :50, 0] = np.clip(32.0 + 0.8 * dome_1, 6, 95).astype(np.uint8)
test_rgb[:50, :50, 1] = np.clip(68.0 + 1.4 * dome_1, 16, 155).astype(np.uint8)
test_rgb[:50, :50, 2] = np.clip(24.0 + 0.6 * dome_1, 4, 80).astype(np.uint8)

test_rgb[:50, 50:, 0] = np.clip(68.0 + 0.85 * dome_1, 24, 130).astype(np.uint8)
test_rgb[:50, 50:, 1] = np.clip(104.0 + 1.35 * dome_1, 42, 175).astype(np.uint8)
test_rgb[:50, 50:, 2] = np.clip(48.0 + 0.65 * dome_1, 16, 105).astype(np.uint8)

test_rgb[50:, :50] = [30, 52, 65]   # Open water (B > R + 4, G >= R, low ExG)
test_rgb[50:, 50:] = [98, 90, 62]   # Bare brown dirt / yellow car (G <= R -> must be rejected!)
r_f = test_rgb[:, :, 0].astype(np.float32)
g_f = test_rgb[:, :, 1].astype(np.float32)
b_f = test_rgb[:, :, 2].astype(np.float32)
test_vari = np.clip((g_f - r_f) / (g_f + r_f - b_f + 1e-6), -1.0, 1.0)

# Verify woody canopy relief mask keeps 3D textured crowns and rejects flat duckweed ponds
flat_pond_patch = np.full((100, 100, 3), [72, 92, 64], dtype=np.uint8)
pond_fol = compute_strict_foliage_mask(flat_pond_patch)
pond_relief = compute_woody_canopy_relief_mask(flat_pond_patch, pond_fol, gsd_m=0.10)
assert cv2.countNonZero(pond_relief) == 0

# Verify anti-cheat confidence penalizes flat uniform green pond/scrub (< 45%, low_ambiguous)
pond_cnt_mask = np.zeros((40, 40), dtype=np.uint8)
cv2.circle(pond_cnt_mask, (20, 20), 15, 255, -1)
pond_conf, pond_tier, _ = compute_crown_confidence(
    crop_rgb=flat_pond_patch[:40, :40],
    cnt_mask=pond_cnt_mask,
    circularity=0.80,
    detector_score=0.55,
    geometry_quality="sam3_native",
)
assert pond_conf < 45.0
assert pond_tier == "low_ambiguous"

dark_cnt = np.array([[[10, 10]], [[40, 10]], [[40, 40]], [[10, 40]]], dtype=np.int32)
sunlit_cnt = np.array([[[60, 10]], [[90, 10]], [[90, 40]], [[60, 40]]], dtype=np.int32)
water_cnt = np.array([[[10, 60]], [[40, 60]], [[40, 90]], [[10, 90]]], dtype=np.int32)
dirt_cnt = np.array([[[60, 60]], [[90, 60]], [[90, 90]], [[60, 90]]], dtype=np.int32)

out_feats = []
for idx, (cnt, gq) in enumerate(
    [
        (dark_cnt, "sam3_native"),
        (sunlit_cnt, "deepforest_watershed"),
        (water_cnt, "sam3_native"),
        (dirt_cnt, "watershed_forest_crown"),
    ],
    start=1,
):
    process_contour_enhanced(
        contour=cnt,
        tree_id=idx,
        img_rgb=test_rgb,
        vari_img=test_vari,
        height_map=None,
        ndvi_src=None,
        transform=None,
        crs=None,
        is_georeferenced=False,
        geo_transformer=None,
        geod=None,
        min_area_m2=0.5,
        max_area_m2=500.0,
        vari_threshold=0.02,
        circularity_threshold=0.0,
        default_height_m=10.0,
        features=out_feats,
        reject_water=True,
        detector_score=0.82,
        geometry_quality=gq,
    )
assert len(out_feats) == 2
assert out_feats[0]["properties"]["ndvi_health"] >= 0.25
assert out_feats[1]["properties"]["ndvi_health"] >= 0.15
assert 78.0 <= out_feats[0]["properties"]["confidence_pct"] <= 98.0
assert out_feats[0]["properties"]["confidence_tier"] == "high_individual"
assert 60.0 <= out_feats[1]["properties"]["confidence_pct"] <= 79.5
assert out_feats[1]["properties"]["confidence_tier"] == "good_treetop"
assert "raw_vari" in out_feats[0]["properties"]
assert "foliar_gli" in out_feats[0]["properties"]

hm = health_models()
assert "deepforest" in hm

# Verify Planar Crown Adjacency Graph Coloring (DSatur) on a 4x4 grid of 16 touching crowns
assert len(DISTINCT_CROWN_PALETTE) == 10
touching_features = []
grid_n = 4
cell_px = 30
base_lon = 100.6190
base_lat = 13.8780
step_deg = 0.00004  # ~4.45m per crown
for r in range(grid_n):
    for c in range(grid_n):
        x0 = c * cell_px
        y0 = r * cell_px
        lon0 = base_lon + c * step_deg
        lon1 = lon0 + step_deg
        lat0 = base_lat + r * step_deg
        lat1 = lat0 + step_deg
        touching_features.append(
            {
                "type": "Feature",
                "id": len(touching_features) + 1,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [lon0, lat0],
                            [lon1, lat0],
                            [lon1, lat1],
                            [lon0, lat1],
                            [lon0, lat0],
                        ]
                    ],
                },
                "properties": {
                    "tree_id": len(touching_features) + 1,
                    "pixel_bounds": [x0, y0, cell_px, cell_px],
                    "pixel_cx": float(x0 + cell_px / 2.0),
                    "pixel_cy": float(y0 + cell_px / 2.0),
                    "crown_diameter_m": 4.4,
                },
            }
        )

assigned_colors = assign_adjacent_distinct_colors(touching_features)
assert len(assigned_colors) == 16
for idx, feat in enumerate(touching_features):
    c_idx = feat["properties"]["color_index"]
    c_hex = feat["properties"]["distinct_color"]
    assert c_idx == assigned_colors[idx]
    assert 0 <= c_idx < len(DISTINCT_CROWN_PALETTE)
    assert c_hex == DISTINCT_CROWN_PALETTE[c_idx]

shared_adj_conflicts = 0
for r in range(grid_n):
    for c in range(grid_n):
        idx_cur = r * grid_n + c
        if c + 1 < grid_n:
            idx_right = r * grid_n + (c + 1)
            if assigned_colors[idx_cur] == assigned_colors[idx_right]:
                shared_adj_conflicts += 1
        if r + 1 < grid_n:
            idx_down = (r + 1) * grid_n + c
            if assigned_colors[idx_cur] == assigned_colors[idx_down]:
                shared_adj_conflicts += 1
assert shared_adj_conflicts == 0

# Verify parse_roi_polygon and build_roi_pixel_mask
assert parse_roi_polygon(None) is None
assert parse_roi_polygon("") is None
assert parse_roi_polygon("[[13.87, 100.61], [13.88, 100.62]]") is None
roi_pts = parse_roi_polygon(
    "[[13.8790, 100.6190], [13.8790, 100.6200], [13.8780, 100.6195]]"
)
assert roi_pts is not None and len(roi_pts) == 3
roi_bounds = {
    "south": 13.8775,
    "west": 100.6185,
    "north": 13.8795,
    "east": 100.6205,
}
roi_mask = build_roi_pixel_mask(roi_pts, 200, 200, roi_bounds)
assert roi_mask is not None and roi_mask.shape == (200, 200)
# Centroid of the triangle (lat=13.87867, lon=100.6195 -> py~83, px~100) must be inside (255)
assert roi_mask[83, 100] == 255
# Far corners outside the triangle must be 0
assert roi_mask[5, 5] == 0
assert roi_mask[195, 195] == 0
assert roi_mask[195, 5] == 0

latest_sess = get_latest_session()
assert isinstance(latest_sess, dict) and "has_session" in latest_sess
assert _normalize_preset_key("nong_khaem_waste_center") == "nongkhaem"
assert _normalize_preset_key("nongkhaem") == "nongkhaem"
assert _normalize_preset_key("on_nut_waste_center") == "onnut"
assert _normalize_preset_key("ramintra_sports_park") == "ramintra"

# Verify V2.8.8 Coarse-GSD (gsd_m >= 0.22, haze_offset == 0.0) RAG Small-Basin Coalescing
# At z=19 (gsd_m = 0.29 m/px, haze_offset = 0.0), a main crown touching a small <13.5 m2 rim basin
# must coalesce into 1 cohesive outer contour with zero < 4.0 m2 micro-shards.
coarse_ws_img = np.full((128, 128, 3), [34, 88, 38], dtype=np.uint8)
yy_cw, xx_cw = np.mgrid[0:128, 0:128]
main_dome = np.clip(36.0 - 1.4 * np.hypot(xx_cw - 56.0, yy_cw - 64.0), -18.0, 36.0)
rim_dome = np.clip(18.0 - 2.2 * np.hypot(xx_cw - 82.0, yy_cw - 64.0), -18.0, 18.0)
dome_comb = np.maximum(main_dome, rim_dome)
coarse_ws_img[:, :, 1] = np.clip(74.0 + 1.5 * dome_comb, 22, 168).astype(np.uint8)
coarse_ws_img[:, :, 0] = np.clip(32.0 + 0.7 * dome_comb, 10, 95).astype(np.uint8)
coarse_ws_img[:, :, 2] = np.clip(26.0 + 0.5 * dome_comb, 8, 80).astype(np.uint8)
coarse_sup = np.zeros((128, 128), dtype=np.uint8)
cv2.circle(coarse_sup, (56, 64), 22, 255, -1)
cv2.circle(coarse_sup, (82, 64), 9, 255, -1)  # ~80 px * 0.0841 m2/px = ~6.7 m2 (< 13.5 m2)
coarse_crowns, coarse_ws_stats = split_dense_canopy_watershed(
    coarse_ws_img,
    coarse_sup,
    [],
    gsd_m=0.29,
    tile_size=128,
    overlap=16,
    min_dense_ratio=0.02,
    min_area_px=25,
    max_area_px=20000,
)
assert len(coarse_crowns) == 1, f"Expected 1 coalesced crown at gsd_m=0.29, got {len(coarse_crowns)}"
assert float(cv2.contourArea(coarse_crowns[0]["contour"])) * (0.29 ** 2) >= 18.0

# Verify V2.7: Dark-green chlorophyll crowns (R < 35, B > R + 3, G > B + 8) are accepted (not confused with water)
dark_chl_img = np.full((128, 128, 3), 90, dtype=np.uint8)
yy_c, xx_c = np.mgrid[0:128, 0:128]
chl_circle = ((xx_c - 64) ** 2 + (yy_c - 64) ** 2) <= 22 ** 2
tex_c = (12.0 * np.sin(xx_c * 0.9) * np.cos(yy_c * 0.9)).astype(np.int16)
dark_chl_img[chl_circle, 0] = np.clip(22 + tex_c[chl_circle] // 2, 8, 34).astype(np.uint8)   # R ~ 22 (< 35)
dark_chl_img[chl_circle, 1] = np.clip(58 + tex_c[chl_circle], 36, 85).astype(np.uint8)      # G ~ 58
dark_chl_img[chl_circle, 2] = np.clip(34 + tex_c[chl_circle] // 2, 18, 48).astype(np.uint8)  # B ~ 34 (> R + 3)

chl_sam = _DummySam()
chl_inst, chl_union, _ = run_sam3_tiled(
    dark_chl_img,
    chl_sam,
    prompt="tree",
    tile_size=128,
    overlap=16,
    confidence=0.18,
    min_area_px=30,
    max_area_px=20000,
    gsd_m=0.10,
)
assert len(chl_inst) >= 1, "Dark-green chlorophyll crown must not be rejected as water"
assert cv2.countNonZero(chl_union) > 200

# Verify V2.8 Dual-Stream Stream A: CIELAB shadow luminance lift preserves chromaticity while lifting dark crowns
shadow_crown_rgb = np.full((64, 64, 3), [18, 36, 26], dtype=np.uint8)
enh_rgb = enhance_shadow_luminance_cielab(shadow_crown_rgb)
raw_lab = cv2.cvtColor(shadow_crown_rgb, cv2.COLOR_RGB2LAB)
enh_lab = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2LAB)
assert float(enh_lab[:, :, 0].mean()) > float(raw_lab[:, :, 0].mean()) + 3.0, "Shadow L must be lifted"
assert float(np.abs(enh_lab[:, :, 1].astype(float) - raw_lab[:, :, 1].astype(float)).mean()) <= 2.5

# Verify V2.8 Stream B: Self-shaded rim adjacent to sunlit foliage passes strict foliage without 17x17 marsh mud dilation
rayleigh_patch = np.full((60, 60, 3), [20, 48, 30], dtype=np.uint8)
rayleigh_patch[:, 30:60] = [14, 28, 28]
rayleigh_fol = compute_strict_foliage_mask(rayleigh_patch)
assert cv2.countNonZero(rayleigh_fol[:, 30:32]) == 60 * 2
assert cv2.countNonZero(rayleigh_fol[:, 40:60]) == 0, "Must not dilate deep into unlit marsh/water"

# Verify V2.6 fallback snapshot and Dual-OS restore scripts exist
fallback_dir = os.path.join(PROJECT_DIR, "backups", "v2_6_fallback")
for req_file in (
    "canopy_v2.py",
    "main.py",
    "script.js",
    "index.html",
    "style.css",
    "restore_v2_6.bat",
    "restore_v2_6.sh",
):
    assert os.path.exists(os.path.join(fallback_dir, req_file)), f"Missing fallback file: {req_file}"

# Verify V2.8.9 Nong Khaem (NK) 5-Case Visual Audit Regression on vhr_1793762267_result.geojson
nk_tif = os.path.join(PROJECT_DIR, "src", "uploads", "vhr_1793762267.tif")
nk_gj_path = os.path.join(PROJECT_DIR, "src", "uploads", "vhr_1793762267_result.geojson")
nk_audit_verified = False
if os.path.exists(nk_tif) and os.path.exists(nk_gj_path):
    import json
    import rasterio

    with rasterio.open(nk_tif) as src_nk:
        inv_nk = ~src_nk.transform
    with open(nk_gj_path, "r", encoding="utf-8") as f_nk:
        nk_gj = json.load(f_nk)
    nk_feats = nk_gj.get("features", [])
    assert len(nk_feats) >= 1800, f"Expected >= 1800 NK crowns, got {len(nk_feats)}"

    def _count_in_box(x0: float, y0: float, x1: float, y1: float) -> int:
        cnt = 0
        for feat in nk_feats:
            p = feat.get("properties", {})
            px, py = inv_nk * (p["centroid_lon"], p["centroid_lat"])
            if x0 <= px <= x1 and y0 <= py <= y1:
                cnt += 1
        return cnt

    assert _count_in_box(880, 825, 1085, 955) >= 20, "Case 1 Upper courtyard garden missed"
    assert _count_in_box(675, 980, 1090, 1095) >= 40, "Case 1 Lower courtyard garden missed"
    assert _count_in_box(398, 1200, 474, 1280) == 1, "Case 2 Single tree must not be over-split"
    assert _count_in_box(2170, 1215, 2202, 1295) >= 1, "Case 3 Left roadside row missed"
    assert _count_in_box(2212, 1220, 2255, 1305) >= 3, "Case 3 Right roadside row missed"
    assert _count_in_box(2585, 1180, 2638, 1298) >= 4, "Case 4 Concrete slab roadside row missed"
    assert _count_in_box(1690, 1873, 1730, 1915) == 1, "Case 5 Red circle solitary tree missed"
    assert _count_in_box(1525, 1855, 1585, 1915) == 0, "Case 5 Left purple lawn circle false positive"
    assert _count_in_box(1762, 1680, 1825, 1745) == 0, "Case 5 Top purple lawn circle false positive"
    assert _count_in_box(1755, 1850, 1830, 1915) == 0, "Case 5 Bottom purple lawn circle false positive"

    # Batch 2 (13 target regions across 5 screenshots on NK vhr_1793762267.tif)
    assert _count_in_box(1280, 460, 1365, 540) == 0, "Batch 2 Img 1 Box 1A compost ring false positive"
    assert _count_in_box(1465, 435, 1520, 475) == 0, "Batch 2 Img 1 Box 1B compost ring false positive"
    assert _count_in_box(1465, 615, 1500, 645) == 0, "Batch 2 Img 1 Box 1C green roof false positive"
    assert _count_in_box(2465, 1655, 2560, 1755) >= 15, "Batch 2 Img 2 Box 2A left nursery plot missed"
    assert _count_in_box(2600, 1600, 2940, 1735) >= 35, "Batch 2 Img 2 Box 2B right nursery plot missed"
    assert _count_in_box(2575, 2290, 2660, 2475) >= 10, "Batch 2 Img 3 Box 3A left shaded woodland missed"
    assert _count_in_box(2740, 2330, 2840, 2430) >= 6, "Batch 2 Img 3 Box 3B right shaded woodland missed"
    assert _count_in_box(2302, 2055, 2350, 2102) == 1, "Batch 2 Img 4 Box 4A dark solitary tree missed"
    assert _count_in_box(2875, 905, 2985, 965) == 0, "Batch 2 Img 5 Box 5A green trucks false positive"
    assert _count_in_box(3080, 935, 3175, 985) == 0, "Batch 2 Img 5 Box 5B green trucks false positive"
    assert _count_in_box(2950, 980, 3295, 1115) == 0, "Batch 2 Img 5 Box 5C green trucks false positive"
    assert _count_in_box(3020, 1210, 3135, 1268) == 0, "Batch 2 Img 5 Box 5D green trucks false positive"
    assert _count_in_box(3165, 1240, 3255, 1290) == 0, "Batch 2 Img 5 Box 5E green trucks false positive"
    sample_p = nk_feats[0].get("properties", {})
    assert "relief_3d_score" in sample_p, "Missing V1.0 relief_3d_score on NK GeoJSON"
    assert "artificial_cyan_score" in sample_p, "Missing V1.0 artificial_cyan_score on NK GeoJSON"
    assert "foliage_prob" in sample_p, "Missing V1.0 foliage_prob on NK GeoJSON"
    nk_audit_verified = True

# Verify V1.0 Pillar 1+2 Coupled Physical Scores & Compute Device Resolution
from canopy_v2 import (
    compute_polygon_physical_scores,
    derive_feature_physical_scores,
    resolve_compute_device,
)

dev = resolve_compute_device()
assert dev in ("cuda", "mps", "cpu"), f"Unexpected compute device: {dev}"

# Synthetic 3D crown vs flat turf vs cyan-painted vehicle crop tests
rng_v3 = np.random.default_rng(42)
crown_crop = np.zeros((32, 32, 3), dtype=np.uint8)
crown_crop[:, :, 0] = rng_v3.integers(18, 55, size=(32, 32), dtype=np.uint8)
crown_crop[:, :, 1] = rng_v3.integers(58, 115, size=(32, 32), dtype=np.uint8)
crown_crop[:, :, 2] = rng_v3.integers(24, 62, size=(32, 32), dtype=np.uint8)
mask_32 = np.ones((32, 32), dtype=np.uint8) * 255
s_crown = compute_polygon_physical_scores(crown_crop, mask_32, haze_offset=0.0)

turf_crop = np.zeros((32, 32, 3), dtype=np.uint8)
turf_crop[:, :, 0] = 86
turf_crop[:, :, 1] = 104
turf_crop[:, :, 2] = 78
s_turf = compute_polygon_physical_scores(turf_crop, mask_32, haze_offset=0.0)

cyan_crop = np.zeros((32, 32, 3), dtype=np.uint8)
cyan_crop[:, :, 0] = 68
cyan_crop[:, :, 1] = 112
cyan_crop[:, :, 2] = 108
s_cyan = compute_polygon_physical_scores(cyan_crop, mask_32, haze_offset=0.0)

assert s_crown["relief_3d_score"] > s_turf["relief_3d_score"] + 0.30, (
    f"3D crown relief ({s_crown['relief_3d_score']}) must exceed flat turf ({s_turf['relief_3d_score']})"
)
assert s_cyan["artificial_cyan_score"] > s_crown["artificial_cyan_score"] + 0.30, (
    f"Cyan vehicle score ({s_cyan['artificial_cyan_score']}) must exceed natural crown ({s_crown['artificial_cyan_score']})"
)
derived_s = derive_feature_physical_scores(
    {"mean_rgb": [42, 84, 52], "foliar_gli": 0.16, "raw_vari": 0.18, "confidence_pct": 88.0}
)
assert 0.0 <= derived_s["foliage_prob"] <= 1.0
assert 0.0 <= derived_s["relief_3d_score"] <= 1.0
assert 0.0 <= derived_s["artificial_cyan_score"] <= 1.0

from canopy_v2 import compute_scene_auto_tune_profile

prof_fine = compute_scene_auto_tune_profile(image, gsd_m=0.08)
assert prof_fine["v2_tile_size"] == 1536
assert prof_fine["min_area_m2"] == 0.5
assert prof_fine["max_area_m2"] == 260

hazy_test_scene = np.clip(image.astype(np.int32) + 45, 52, 240).astype(np.uint8)
prof_wide = compute_scene_auto_tune_profile(hazy_test_scene, gsd_m=0.29)
assert prof_wide["v2_tile_size"] == 1280
assert prof_wide["min_area_m2"] == 1.2
assert prof_wide["max_area_m2"] == 380
assert prof_wide["haze_offset"] > 0.0
assert "Rayleigh Haze" in prof_wide["atmosphere_regime"]

print(
    {
        "tiles": len(windows),
        "support_pixels": int(cv2.countNonZero(support)),
        "groups": len(groups),
        "dense_candidates": len(dense_candidates),
        "df_verified": df_stats["df_verified"],
        "sam_skipped_barren_tiles": sam_tiled_stats["skipped_barren_tiles"],
        "car_vetoed": car_stats["vehicle_vetoed"],
        "touching_crowns_colored": len(assigned_colors),
        "shared_adj_conflicts": shared_adj_conflicts,
        "roi_mask_nonzero": int(cv2.countNonZero(roi_mask)),
        "dark_chlorophyll_crowns": len(chl_inst),
        "rayleigh_foliage_px": int(cv2.countNonZero(rayleigh_fol)),
        "v2_6_fallback_verified": True,
        "nk_v2_8_9_audit_verified": nk_audit_verified,
    }
)
