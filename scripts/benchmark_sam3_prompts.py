"""Compare SAM 3 canopy prompts on representative native M1 tiles."""

import json
import os
import sys
import time

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
sys.path.insert(0, SRC_DIR)
os.chdir(SRC_DIR)

import main  # noqa: E402
from canopy_v2 import iter_tile_windows  # noqa: E402


PROMPTS = [
    "tree",
    "tree canopy",
    "tree crown viewed from above",
    "individual tree crown in an aerial image",
]
TILE_INDICES = [0, 43, 50]
CONFIDENCE = 0.25


def run():
    image_bgr = cv2.imread(os.path.join("uploads", "M1.jpg"))
    if image_bgr is None:
        raise FileNotFoundError("src/uploads/M1.jpg")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    windows = list(
        iter_tile_windows(
            image_rgb.shape[1], image_rgb.shape[0], tile_size=1536, overlap=192
        )
    )
    sam = main.get_sam3_model()
    sam.set_confidence_threshold(CONFIDENCE)
    results = []

    for tile_index in TILE_INDICES:
        item = windows[tile_index]
        x0, y0, x1, y1 = item["tile"]
        tile = np.ascontiguousarray(image_rgb[y0:y1, x0:x1])
        sam.set_image(tile)
        for prompt in PROMPTS:
            started = time.perf_counter()
            sam.generate_masks(prompt, min_size=80, quiet=True)
            masks = list(sam.masks or [])
            union = np.zeros(tile.shape[:2], dtype=np.uint8)
            for mask in masks:
                union[np.asarray(mask).squeeze() > 0.5] = 255
            safe_prompt = prompt.replace(" ", "_")
            overlay = tile.copy()
            overlay[union > 0] = (
                0.55 * overlay[union > 0] + 0.45 * np.array([0, 255, 180])
            ).astype(np.uint8)
            cv2.imwrite(
                os.path.join(
                    "uploads", f"sam3_tile_{tile_index}_{safe_prompt}.jpg"
                ),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
            results.append(
                {
                    "tile": tile_index,
                    "prompt": prompt,
                    "instances": len(masks),
                    "mask_percent": round(
                        100.0
                        * int(np.count_nonzero(union))
                        / (tile.shape[0] * tile.shape[1]),
                        3,
                    ),
                    "seconds": round(time.perf_counter() - started, 3),
                }
            )
            sam.masks = None
            sam.boxes = None
            sam.scores = None

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
