"""Run a bounded M1 Canopy V2 smoke test without the browser."""

import asyncio
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
sys.path.insert(0, SRC_DIR)
os.chdir(SRC_DIR)

import main  # noqa: E402


async def run():
    max_tiles = int(os.environ.get("CANOPY_V2_QA_MAX_TILES", "4"))
    result = await main.segment_canopy(
        file_path=os.path.join("uploads", "M1.jpg"),
        model_type="canopy_v2",
        conf_threshold=0.25,
        min_area_m2=1.0,
        max_area_m2=1000.0,
        vari_threshold=0.05,
        circularity_threshold=0.25,
        nms_threshold=0.5,
        default_height_m=10.0,
        sam3_prompt="tree",
        use_auto=True,
        v2_tile_size=1536,
        v2_overlap=192,
        v2_coverage_threshold=0.45,
        v2_instance_threshold=0.25,
        v2_max_tiles=max_tiles,
    )
    summary = result["properties"]["v2_summary"]
    assert summary["pipeline_version"] == "canopy_v2_prototype"
    expected_tiles = summary["tiles_total"] if max_tiles == 0 else max_tiles
    assert summary["tiles_processed"] == expected_tiles
    assert summary["coverage_percent"] > 0
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
