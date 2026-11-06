import os
import shutil
import numpy as np
import cv2
import torch
import requests
import math
import time
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import rasterio
from rasterio.warp import transform_bounds
from pyproj import Transformer, Geod

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

load_env()


def _fix_windows_ssl_cert_store():
    """
    Some Windows installs break ssl.create_default_context when loading the
    system cert store (ASN1: NOT_ENOUGH_DATA). Point TLS at certifi instead so
    torch hub / huggingface downloads for SAM 3 can succeed.
    """
    try:
        import ssl
        import certifi

        ca = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", ca)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
        os.environ.setdefault("CURL_CA_BUNDLE", ca)

        def _create_default_context(
            purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None
        ):
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            if cafile or capath or cadata:
                ctx.load_verify_locations(cafile=cafile, capath=capath, cadata=cadata)
            else:
                ctx.load_verify_locations(cafile=ca)
            return ctx

        ssl._create_default_https_context = _create_default_context
        ssl.create_default_context = _create_default_context
        print(f"SSL: using certifi CA bundle ({ca})")
    except Exception as e:
        print(f"SSL certifi patch skipped: {e}")


_fix_windows_ssl_cert_store()

def save_sidecar_metadata(file_path: str, metadata: dict):
    sidecar_path = os.path.splitext(file_path)[0] + ".json"
    to_save = dict(metadata)
    if to_save.get("file_path"):
        to_save["file_path"] = os.path.join(
            "uploads", os.path.basename(str(to_save["file_path"]))
        ).replace("\\", "/")
    if to_save.get("ndvi_file_path"):
        to_save["ndvi_file_path"] = os.path.join(
            "uploads", os.path.basename(str(to_save["ndvi_file_path"]))
        ).replace("\\", "/")
    with open(sidecar_path, "w") as f:
        json.dump(to_save, f, indent=2)

def load_sidecar_metadata(file_path: str) -> Optional[dict]:
    sidecar_path = os.path.splitext(file_path)[0] + ".json"
    if os.path.exists(sidecar_path):
        with open(sidecar_path, "r") as f:
            return json.load(f)
    return None

def generate_geotiff_preview(tif_path: str, out_name: str, upload_dir: str) -> Optional[str]:
    try:
        with rasterio.open(tif_path) as src:
            if src.count >= 3:
                img_rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
            else:
                img_rgb = src.read(1)
                if len(img_rgb.shape) == 2:
                    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
            
            if img_rgb.dtype != np.uint8:
                img_rgb = ((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-6) * 255).astype(np.uint8)
            
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Resize if too large for browser performance
            max_dim = 2048
            h, w = img_bgr.shape[:2]
            if h > max_dim or w > max_dim:
                scale = max_dim / max(h, w)
                img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                
            out_path = os.path.join(upload_dir, out_name)
            cv2.imwrite(out_path, img_bgr)
            return f"/static_uploads/{out_name}"
    except Exception as e:
        print(f"Error generating preview for geotiff: {e}")
        return None


app = FastAPI(title="Plant Canopy Extraction API")

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(SCRIPT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def log_progress(msg: str):
    # Windows consoles may be cp1252 — avoid crashing on unicode arrows etc.
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    try:
        print(f"[PROGRESS] {safe}")
    except Exception:  # nosec B110
        pass
    log_path = os.path.join(UPLOAD_DIR, "extraction.log")
    try:
        timestamp = time.strftime("%H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except Exception as e:
        try:
            print(f"Error writing progress log: {e}")
        except Exception:  # nosec B110
            pass

def clear_progress_log():
    log_path = os.path.join(UPLOAD_DIR, "extraction.log")
    try:
        if os.path.exists(log_path):
            os.remove(log_path)
    except Exception as e:
        print(f"Error clearing progress log: {e}")

# Mount uploads directory to serve images directly
app.mount("/static_uploads", StaticFiles(directory=UPLOAD_DIR), name="static_uploads")

# Check CUDA / Apple Silicon MPS / CPU availability with a live test to catch architecture capability mismatches
from canopy_v2 import resolve_compute_device

device = resolve_compute_device()
print(f"Compute device resolved: {device.upper()}")

def _module_available(name: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


DEEPFOREST_AVAILABLE = _module_available("deepforest")
sam3_model = None
deepforest_model = None


def get_sam3_model():
    global sam3_model
    if sam3_model is None:
        import samgeo.samgeo3 as samgeo3_mod
        from samgeo import SamGeo3

        meta_ok = bool(getattr(samgeo3_mod, "SAM3_META_AVAILABLE", False))
        backend_type = "meta" if (meta_ok and os.name != "nt") else "transformers"
        sam3_model = SamGeo3(backend=backend_type, device=device)
    return sam3_model


def get_deepforest_model():
    global deepforest_model
    if deepforest_model is None:
        try:
            from deepforest import main as df_main

            df = df_main.deepforest(config_args={"score_thresh": 0.18})
            if device in ("cuda", "mps") and hasattr(df, "model") and df.model is not None:
                df.model.to(device)
            if hasattr(df, "model") and df.model is not None:
                df.model.eval()
            deepforest_model = df
        except Exception as e:
            print(f"DeepForest 2.1.0 unavailable: {e}")
            return None
    return deepforest_model


_NO_CACHE_HEADERS = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}


@app.get("/")
def read_root(request: Request = None):
    if request is not None and "text/html" in request.headers.get("accept", ""):
        return FileResponse(os.path.join(SCRIPT_DIR, "index.html"), headers=_NO_CACHE_HEADERS)
    return {"status": "running", "device": device}


@app.get("/health/status")
def read_health_status():
    return {"status": "running", "device": device}


@app.get("/index.html")
def serve_index_html():
    return FileResponse(os.path.join(SCRIPT_DIR, "index.html"), headers=_NO_CACHE_HEADERS)


@app.get("/style.css")
def serve_style_css():
    return FileResponse(os.path.join(SCRIPT_DIR, "style.css"), media_type="text/css", headers=_NO_CACHE_HEADERS)


@app.get("/script.js")
def serve_script_js():
    return FileResponse(os.path.join(SCRIPT_DIR, "script.js"), media_type="application/javascript", headers=_NO_CACHE_HEADERS)


@app.get("/health/models")
def health_models():
    """Report Canopy V2 (SAM 3 + DeepForest + RGB/Watershed) readiness."""
    samgeo_ok = _module_available("samgeo")
    df_ok = bool(DEEPFOREST_AVAILABLE or _module_available("deepforest"))
    return {
        "device": device,
        "cuda": device == "cuda",
        "samgeo": samgeo_ok,
        "deepforest": df_ok,
        "primary_ready": True,
        "pipeline_ready": samgeo_ok or df_ok,
        "pipeline_version": "canopy_v2",
        "message": (
            "SAM 3 + DeepForest + Canopy V2 ready"
            if (samgeo_ok and df_ok)
            else (
                "SAM 3 + Canopy V2 ready"
                if samgeo_ok
                else "SAM 3 (samgeo) not installed — Canopy V2 will run in RGB support + Watershed fallback mode."
            )
        ),
    }


def _write_last_roi_session(metadata: dict):
    """Persist active session metadata to both src/uploads and root uploads."""
    for session_dir in (UPLOAD_DIR, os.path.join(os.path.dirname(SCRIPT_DIR), "uploads")):
        try:
            os.makedirs(session_dir, exist_ok=True)
            with open(os.path.join(session_dir, "last_roi_session.json"), "w", encoding="utf-8") as sf:
                json.dump(metadata, sf, indent=2)
        except Exception:  # nosec B110
            pass


_PRESET_KEY_ALIASES = {
    "on_nut_waste_center": "onnut",
    "onnut": "onnut",
    "nong_khaem_waste_center": "nongkhaem",
    "nongkhaem": "nongkhaem",
    "ramintra_sports_park": "ramintra",
    "ramintra": "ramintra",
    "saphan_sung_park": "sapansung",
    "saphansung": "sapansung",
    "sapansung": "sapansung",
    "piyaphirom_park": "piyaphirom",
    "piyaphirom": "piyaphirom",
}


def _normalize_preset_key(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    k = str(key).strip().lower()
    return _PRESET_KEY_ALIASES.get(k, k)


def _ensure_geojson_physical_scores(res_geojson_path: str) -> None:
    """
    Ensure cached *_result.geojson files have V3.0 physical scores
    (foliage_prob, relief_3d_score, artificial_cyan_score) populated for <30ms live sliders.
    """
    if not os.path.isfile(res_geojson_path):
        return
    try:
        with open(res_geojson_path, "r", encoding="utf-8") as gf:
            gj = json.load(gf)
        feats = gj.get("features") or []
        if not feats:
            return
        first_props = feats[0].get("properties") or {}
        if (
            "relief_3d_score" in first_props
            and "artificial_cyan_score" in first_props
            and "foliage_prob" in first_props
        ):
            return
        from canopy_v2 import derive_feature_physical_scores

        for feat in feats:
            props = feat.setdefault("properties", {})
            scores = derive_feature_physical_scores(props)
            props.update(scores)
        tmp_path = f"{res_geojson_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as wf:
            json.dump(gj, wf)
        os.replace(tmp_path, res_geojson_path)
    except Exception as exc:
        print(f"Warning: could not backfill physical scores on {res_geojson_path}: {exc}")


def _ensure_auto_tune_metadata(meta: dict, json_path: Optional[str] = None) -> dict:
    """Compute and attach scene-calibrated auto_tune parameters to session metadata."""
    if not isinstance(meta, dict):
        return meta
    if isinstance(meta.get("auto_tune"), dict) and "v2_instance_threshold" in meta["auto_tune"]:
        return meta
    try:
        from canopy_v2 import compute_scene_auto_tune_profile

        fp = meta.get("file_path") or meta.get("filename")
        if not fp:
            return meta
        basename = os.path.basename(str(fp))
        stem = os.path.splitext(basename)[0]
        cand_paths = [
            os.path.join(UPLOAD_DIR, basename),
            os.path.join(UPLOAD_DIR, f"{stem}_preview.jpg"),
        ]
        img_rgb = None
        for cp in cand_paths:
            if os.path.isfile(cp):
                bgr = cv2.imread(cp)
                if bgr is not None and bgr.size > 0:
                    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    break
        if img_rgb is None:
            return meta
        gsd_val = float(meta.get("gsd_m") or meta.get("gsd_m_nominal") or 0.15)
        meta["auto_tune"] = compute_scene_auto_tune_profile(img_rgb, gsd_m=gsd_val)
        if json_path and os.path.isfile(json_path):
            tmp_p = json_path + ".autotune_tmp"
            with open(tmp_p, "w", encoding="utf-8") as wf:
                json.dump(meta, wf, indent=2)
            os.replace(tmp_p, json_path)
    except Exception as exc:
        print(f"Warning: could not compute auto_tune profile: {exc}")
    return meta


def _find_matching_vhr_session(
    site_preset: Optional[str] = None,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
) -> Optional[dict]:
    """Find the most recent vhr_*.json in UPLOAD_DIR matching a site_preset or center coordinates."""
    if not os.path.isdir(UPLOAD_DIR):
        return None
    norm_preset = _normalize_preset_key(site_preset)
    candidates = []
    for fname in os.listdir(UPLOAD_DIR):
        if (
            fname.startswith("vhr_")
            and fname.endswith(".json")
            and not fname.endswith("_preview.json")
        ):
            full_json = os.path.join(UPLOAD_DIR, fname)
            stem = os.path.splitext(fname)[0]
            full_tif = os.path.join(UPLOAD_DIR, f"{stem}.tif")
            res_geojson = os.path.join(UPLOAD_DIR, f"{stem}_result.geojson")
            if os.path.isfile(full_json) and (os.path.isfile(full_tif) or os.path.isfile(res_geojson)):
                has_res = 1 if os.path.isfile(res_geojson) else 0
                candidates.append((has_res, os.path.getmtime(full_json), full_json, stem))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    for _, _, full_json, stem in candidates:
        try:
            with open(full_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if norm_preset and _normalize_preset_key(meta.get("site_preset")) == norm_preset:
            res_geojson = os.path.join(UPLOAD_DIR, f"{stem}_result.geojson")
            if os.path.isfile(res_geojson):
                _ensure_geojson_physical_scores(res_geojson)
                if not meta.get("result_geojson_url"):
                    meta["result_geojson_url"] = f"/static_uploads/{stem}_result.geojson"
            return _ensure_auto_tune_metadata(meta, full_json)
        if (
            center_lat is not None
            and center_lon is not None
            and isinstance(meta.get("center_lat"), (int, float))
            and isinstance(meta.get("center_lon"), (int, float))
        ):
            if (
                abs(float(meta["center_lat"]) - float(center_lat)) <= 0.008
                and abs(float(meta["center_lon"]) - float(center_lon)) <= 0.008
            ):
                res_geojson = os.path.join(UPLOAD_DIR, f"{stem}_result.geojson")
                if os.path.isfile(res_geojson):
                    _ensure_geojson_physical_scores(res_geojson)
                    if not meta.get("result_geojson_url"):
                        meta["result_geojson_url"] = f"/static_uploads/{stem}_result.geojson"
                return _ensure_auto_tune_metadata(meta, full_json)

    if site_preset is None and center_lat is None and center_lon is None:
        _, _, newest_json, stem = candidates[0]
        with open(newest_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        res_geojson = os.path.join(UPLOAD_DIR, f"{stem}_result.geojson")
        if os.path.isfile(res_geojson):
            _ensure_geojson_physical_scores(res_geojson)
            if not meta.get("result_geojson_url"):
                meta["result_geojson_url"] = f"/static_uploads/{stem}_result.geojson"
        return _ensure_auto_tune_metadata(meta, newest_json)
    return None


@app.get("/latest_session")
def get_latest_session(site_preset: Optional[str] = None):
    """Return the most recent persisted VHR GeoTIFF session metadata and ROI polygon if available."""
    try:
        if site_preset:
            matched = _find_matching_vhr_session(site_preset=site_preset)
            if matched:
                return {"has_session": True, "metadata": matched}
            return {"has_session": False}

        session_file = os.path.join(UPLOAD_DIR, "last_roi_session.json")
        if os.path.isfile(session_file):
            with open(session_file, "r", encoding="utf-8") as sf:
                meta = json.load(sf)
            if isinstance(meta, dict) and (meta.get("center_lat") is not None or meta.get("file_path")):
                fp = meta.get("file_path")
                if fp:
                    stem = os.path.splitext(os.path.basename(str(fp)))[0]
                    res_geojson = os.path.join(UPLOAD_DIR, f"{stem}_result.geojson")
                    if os.path.isfile(res_geojson):
                        _ensure_geojson_physical_scores(res_geojson)
                        if not meta.get("result_geojson_url"):
                            meta["result_geojson_url"] = f"/static_uploads/{stem}_result.geojson"
                meta = _ensure_auto_tune_metadata(meta, session_file)
                return {"has_session": True, "metadata": meta}

        fallback = _find_matching_vhr_session()
        if fallback:
            return {"has_session": True, "metadata": fallback}
        return {"has_session": False}
    except Exception as e:
        print(f"Error loading latest session: {e}")
        return {"has_session": False}


@app.post("/save_boundary_session")
async def save_boundary_session(
    site_preset: Optional[str] = Form(None),
    center_lat: float = Form(...),
    center_lon: float = Form(...),
    width_m: float = Form(...),
    length_m: float = Form(...),
    roi_polygon: Optional[str] = Form(None),
):
    """Persist active site preset, location, and polygon boundary to last_roi_session.json."""
    parsed_roi = parse_roi_polygon(roi_polygon)
    matched = _find_matching_vhr_session(
        site_preset=site_preset,
        center_lat=center_lat,
        center_lon=center_lon,
    )
    if matched:
        metadata = dict(matched)
        metadata["center_lat"] = float(center_lat)
        metadata["center_lon"] = float(center_lon)
        metadata["width_m"] = float(width_m)
        metadata["length_m"] = float(length_m)
        if site_preset:
            metadata["site_preset"] = site_preset
        if parsed_roi and len(parsed_roi) >= 3:
            metadata["roi_polygon"] = parsed_roi
        metadata["updated_at"] = int(time.time() * 1000)
        if metadata.get("filename"):
            tif_path = os.path.join(UPLOAD_DIR, os.path.basename(str(metadata["filename"])))
            if os.path.isfile(tif_path):
                save_sidecar_metadata(tif_path, metadata)
    else:
        EarthRadius = 6378137.0
        c_lat_rad = float(center_lat) * math.pi / 180.0
        d_lat = (float(length_m) / 2.0 / EarthRadius) * (180.0 / math.pi)
        d_lon = (float(width_m) / 2.0 / (EarthRadius * math.cos(c_lat_rad))) * (180.0 / math.pi)
        metadata = {
            "site_preset": site_preset,
            "center_lat": float(center_lat),
            "center_lon": float(center_lon),
            "width_m": float(width_m),
            "length_m": float(length_m),
            "bounds": {
                "south": float(center_lat) - d_lat,
                "west": float(center_lon) - d_lon,
                "north": float(center_lat) + d_lat,
                "east": float(center_lon) + d_lon,
            },
            "crs": "EPSG:4326",
            "pipeline_mode": "geo",
            "roi_polygon": parsed_roi if (parsed_roi and len(parsed_roi) >= 3) else None,
            "updated_at": int(time.time() * 1000),
        }
    _write_last_roi_session(metadata)
    return {"status": "saved", "has_raster": bool(metadata.get("file_path")), "metadata": metadata}



def _image_only_preview_bounds(width: int, height: int, assumed_gsd_m: float = 0.10):
    """
    Place a non-georeferenced image on the map for visual QA only.
    Uses assumed GSD so crown shapes are relatively sized; location is NOT survey truth.
    Centered at Ramintra Sports Park (สวนกีฬารามอินทรา) to match the default basemap.
    """
    center_lon = 100.6195068
    center_lat = 13.8785514
    half_w_m = (width * assumed_gsd_m) / 2.0
    half_h_m = (height * assumed_gsd_m) / 2.0
    earth = 6378137.0
    d_lat = (half_h_m / earth) * (180.0 / math.pi)
    d_lon = (half_w_m / (earth * math.cos(math.radians(center_lat)))) * (180.0 / math.pi)
    return {
        "south": center_lat - d_lat,
        "west": center_lon - d_lon,
        "north": center_lat + d_lat,
        "east": center_lon + d_lon,
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload RGB GeoTIFF (GEO mode) or plain image (IMAGE-ONLY mode).
    Shared detection models; modes differ in georef honesty, GSD, and export semantics.
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use GeoTIFF, PNG, or JPG.")

    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    metadata = {
        "filename": file.filename,
        "file_path": os.path.join("uploads", os.path.basename(temp_file_path)).replace("\\", "/"),
        "is_geotiff": False,
        "width": 0,
        "height": 0,
        "bounds": None,
        "crs": None,
        "preview_url": None,
        "pipeline_mode": "image_only",
        "georef_quality": "none",
        "assumed_gsd_m": None,
        "gsd_m": None,
        "mode_message": "",
    }

    has_real_crs = False
    try:
        with rasterio.open(temp_file_path) as src:
            metadata["width"] = src.width
            metadata["height"] = src.height
            # JPEG/PNG may open via GDAL without being a GeoTIFF
            if file_ext in [".tif", ".tiff"]:
                metadata["is_geotiff"] = True
            if src.crs is not None:
                # Reject pure pixel / identity georef without CRS
                try:
                    left, bottom, right, top = src.bounds
                    # Identity bounds in pixel space (0..W, 0..H) are not geographic
                    looks_like_pixels = (
                        abs(left) < 1 and abs(top) < 1 and right > 100 and bottom > 100
                    ) or (src.crs is None)
                    if src.crs and not looks_like_pixels:
                        metadata["crs"] = str(src.crs)
                        try:
                            wgs84_bounds = transform_bounds(
                                src.crs, "EPSG:4326", left, bottom, right, top
                            )
                            metadata["bounds"] = {
                                "south": wgs84_bounds[1],
                                "west": wgs84_bounds[0],
                                "north": wgs84_bounds[3],
                                "east": wgs84_bounds[2],
                            }
                            has_real_crs = True
                            # Approximate GSD from transform
                            if src.crs.is_projected:
                                metadata["gsd_m"] = abs(float(src.transform.a))
                            else:
                                c_lat = (wgs84_bounds[1] + wgs84_bounds[3]) / 2.0
                                metadata["gsd_m"] = abs(float(src.transform.a)) * 111320.0 * math.cos(
                                    math.radians(c_lat)
                                )
                        except Exception as e:
                            print(f"Error transforming bounds: {e}")
                except Exception as e:
                    print(f"Bounds read error: {e}")
    except Exception as e:
        print(f"Could not read with rasterio: {e}")
        img = cv2.imread(temp_file_path)
        if img is not None:
            metadata["width"] = img.shape[1]
            metadata["height"] = img.shape[0]

    if file_ext in [".tif", ".tiff"]:
        preview_name = f"{os.path.splitext(file.filename)[0]}_preview.png"
        metadata["preview_url"] = generate_geotiff_preview(
            temp_file_path, preview_name, UPLOAD_DIR
        )
    else:
        metadata["preview_url"] = f"/static_uploads/{file.filename}"

    if has_real_crs and metadata.get("bounds"):
        metadata["pipeline_mode"] = "geo"
        metadata["georef_quality"] = "survey"
        metadata["mode_message"] = (
            "GEO mode: real CRS/bounds. Suitable for map export and Rhino/MCDA."
        )
    else:
        # IMAGE-ONLY: plain aerial/drone JPG or non-georeferenced raster
        assumed = 0.10  # metres/pixel assumption for relative crown size only
        metadata["pipeline_mode"] = "image_only"
        metadata["georef_quality"] = "none"
        metadata["assumed_gsd_m"] = assumed
        metadata["gsd_m"] = assumed
        metadata["crs"] = "EPSG:4326"
        metadata["bounds"] = _image_only_preview_bounds(
            metadata["width"], metadata["height"], assumed_gsd_m=assumed
        )
        metadata["mode_message"] = (
            "IMAGE-ONLY mode: no real georeference. Detection QA only — "
            f"map placement is a preview; areas use assumed GSD {assumed:.2f} m/px "
            "(not survey-grade). Prefer a GeoTIFF for Rhino/MCDA."
        )
        print(f"Upload classified as image_only: {file.filename}")

    metadata = _ensure_auto_tune_metadata(metadata)
    save_sidecar_metadata(temp_file_path, metadata)
    return metadata


def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_to_latlon(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


from requests.adapters import HTTPAdapter

_HTTP_SESSION = requests.Session()
_HTTP_ADAPTER = HTTPAdapter(pool_connections=24, pool_maxsize=24)
_HTTP_SESSION.mount("https://", _HTTP_ADAPTER)
_HTTP_SESSION.mount("http://", _HTTP_ADAPTER)


def download_google_tile(x, y, z, scale=1):
    sub = f"mt{(x + y) % 4}"
    scale_q = f"&scale={int(scale)}" if int(scale) > 1 else ""
    url = f"https://{sub}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}{scale_q}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    for attempt in range(3):
        try:
            r = _HTTP_SESSION.get(url, headers=headers, timeout=6)
            if r.status_code == 200:
                arr = np.asarray(bytearray(r.content), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return (x, y, img)
            if r.status_code == 404:
                return (x, y, None)
            print(
                f"[RETRY] Attempt {attempt + 1} for tile {x}, {y} "
                f"returned status {r.status_code}"
            )
        except Exception as e:
            print(f"[RETRY] Attempt {attempt + 1} for tile {x}, {y} failed: {e}")
        time.sleep(0.25 * (attempt + 1))
    return (x, y, None)


def get_dynamic_zoom(width_m: float, length_m: float) -> int:
    """Choose Web Mercator zoom for VHR mosaic based on AOI span."""
    max_dim = max(float(width_m), float(length_m))
    if max_dim <= 160:
        return 22
    if max_dim <= 360:
        return 21
    if max_dim <= 1200:
        return 20
    return 19


def web_mercator_gsd_m(lat_deg: float, zoom: int, tile_px: int = 512) -> float:
    """Approximate meters per pixel at latitude for a given zoom."""
    m_per_px_256 = 156543.03392804097 * math.cos(math.radians(lat_deg)) / (2 ** zoom)
    return m_per_px_256 * (256.0 / float(tile_px))


def write_rgb_geotiff(path: str, img_bgr: np.ndarray, west: float, south: float, east: float, north: float):
    """Write georeferenced RGB GeoTIFF (EPSG:4326) from OpenCV BGR array."""
    from rasterio.transform import from_bounds

    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image for GeoTIFF write")
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = from_bounds(west, south, east, north, w, h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        compress="lzw",
        photometric="RGB",
    ) as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)
    return w, h


def stitch_google_satellite(
    west,
    south,
    east,
    north,
    zoom=21,
    high_dpi: bool = True,
    return_meta: bool = False,
):
    """
    Mosaic Google Satellite VHR tiles over a WGS84 bbox and crop to exact bounds.
    At target_z >= 20 with high_dpi=True, requests scale=2 (512×512) tiles at z_req = target_z - 1,
    which yields the exact same Web Mercator pixel grid and GSD as scale=1 (256×256) at target_z
    with 4× fewer HTTP requests, and automatically falls back to lower zoom if tiles 404.
    """
    from concurrent.futures import ThreadPoolExecutor

    def lat_to_y(lat):
        lat_rad = math.radians(lat)
        return math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad)))

    target_z = min(max(int(zoom), 18), 22)
    min_try_z = min(19, target_z)
    last_error = None

    for cand_z in range(target_z, min_try_z - 1, -1):
        if high_dpi and cand_z >= 19:
            z_req = cand_z - 1
            scale = 2
            tile_px = 512
        else:
            z_req = cand_z
            scale = 1
            tile_px = 256

        x_min, y_max = latlon_to_tile(south, west, z_req)
        x_max, y_min = latlon_to_tile(north, east, z_req)
        width_tiles = x_max - x_min + 1
        height_tiles = y_max - y_min + 1

        if width_tiles * height_tiles > 400:
            last_error = (
                f"AOI too large at zoom {cand_z} "
                f"({width_tiles}x{height_tiles} tiles at z_req={z_req})"
            )
            continue

        tile_coords = [
            (x, y)
            for x in range(x_min, x_max + 1)
            for y in range(y_min, y_max + 1)
        ]
        tiles = {}
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = executor.map(
                lambda c: download_google_tile(c[0], c[1], z_req, scale=scale),
                tile_coords,
            )
            for x, y, img in results:
                if img is not None:
                    tiles[(x, y)] = img

        if not tile_coords or len(tiles) != len(tile_coords):
            last_error = (
                f"Incomplete Google tile set at z={cand_z} "
                f"(got {len(tiles)}/{len(tile_coords)} tiles)"
            )
            continue

        stitched = np.zeros(
            (height_tiles * tile_px, width_tiles * tile_px, 3), dtype=np.uint8
        )
        for (x, y), img in tiles.items():
            if img.shape[0] != tile_px or img.shape[1] != tile_px:
                img = cv2.resize(
                    img, (tile_px, tile_px), interpolation=cv2.INTER_AREA
                )
            dx = (x - x_min) * tile_px
            dy = (y - y_min) * tile_px
            stitched[dy : dy + tile_px, dx : dx + tile_px] = img

        top_lat, left_lon = tile_to_latlon(x_min, y_min, z_req)
        bottom_lat, right_lon = tile_to_latlon(x_max + 1, y_max + 1, z_req)
        h_stitched, w_stitched = stitched.shape[:2]

        x1 = int((west - left_lon) / (right_lon - left_lon) * w_stitched)
        x2 = int((east - left_lon) / (right_lon - left_lon) * w_stitched)
        y_top = lat_to_y(top_lat)
        y_bottom = lat_to_y(bottom_lat)
        y1 = int((y_top - lat_to_y(north)) / (y_top - y_bottom) * h_stitched)
        y2 = int((y_top - lat_to_y(south)) / (y_top - y_bottom) * h_stitched)
        x1 = max(0, min(w_stitched - 1, x1))
        x2 = max(0, min(w_stitched, x2))
        y1 = max(0, min(h_stitched - 1, y1))
        y2 = max(0, min(h_stitched, y2))
        if x2 <= x1 or y2 <= y1:
            last_error = "Google stitch crop is empty"
            continue

        cropped_bgr = stitched[y1:y2, x1:x2]
        if return_meta:
            return cropped_bgr, cand_z, tile_px
        return cropped_bgr

    raise ValueError(
        last_error or "Failed to stitch Google Satellite VHR tiles."
    )


def parse_roi_polygon(roi_polygon_raw: Optional[str]) -> Optional[list[list[float]]]:
    """Parse and validate a drawable ROI polygon of [[lat, lon], ...] vertices."""
    if roi_polygon_raw is None:
        return None
    raw_obj = roi_polygon_raw
    if isinstance(roi_polygon_raw, str):
        s = roi_polygon_raw.strip()
        if not s or s.lower() == "null":
            return None
        try:
            raw_obj = json.loads(s)
        except Exception:
            return None
    if not isinstance(raw_obj, (list, tuple)) or len(raw_obj) < 3:
        return None

    parsed: list[list[float]] = []
    for pt in raw_obj:
        try:
            if isinstance(pt, dict):
                lat_v = pt.get("lat")
                lon_v = pt.get("lon", pt.get("lng"))
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                lat_v, lon_v = pt[0], pt[1]
            else:
                continue
            lat_f = float(lat_v)
            lon_f = float(lon_v)
            if (
                math.isfinite(lat_f)
                and math.isfinite(lon_f)
                and -90.0 <= lat_f <= 90.0
                and -180.0 <= lon_f <= 180.0
            ):
                parsed.append([lat_f, lon_f])
        except (TypeError, ValueError):
            continue

    if len(parsed) >= 2 and parsed[0] == parsed[-1]:
        unique_check = parsed[:-1]
    else:
        unique_check = parsed
    if len(unique_check) < 3:
        return None
    return parsed


def build_roi_pixel_mask(
    roi_polygon: Optional[list[list[float]]],
    width: int,
    height: int,
    bounds: Optional[dict],
) -> Optional[np.ndarray]:
    """Rasterize a WGS84 [[lat, lon], ...] ROI polygon into an 8-bit (height, width) mask."""
    if (
        not roi_polygon
        or len(roi_polygon) < 3
        or not bounds
        or int(width) <= 0
        or int(height) <= 0
    ):
        return None
    try:
        south = float(bounds["south"])
        west = float(bounds["west"])
        north = float(bounds["north"])
        east = float(bounds["east"])
    except (KeyError, TypeError, ValueError):
        return None

    w_int = int(width)
    h_int = int(height)
    lon_span = max(east - west, 1e-12)
    lat_span = max(north - south, 1e-12)

    pts_list = []
    for pair in roi_polygon:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        lat, lon = float(pair[0]), float(pair[1])
        px = int(round((lon - west) / lon_span * w_int))
        py = int(round((north - lat) / lat_span * h_int))
        pts_list.append([px, py])

    if len(pts_list) < 3:
        return None

    mask = np.zeros((h_int, w_int), dtype=np.uint8)
    pts = np.asarray(pts_list, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    if cv2.countNonZero(mask) == 0:
        return None
    return mask


def _contour_centroid_in_mask(contour: np.ndarray, roi_mask: np.ndarray) -> bool:
    """Return True if contour centroid (cx, cy) lies inside a non-zero pixel of roi_mask."""
    cnt = np.asarray(contour)
    if cnt.size == 0:
        return False
    m = cv2.moments(cnt)
    if m["m00"] > 0:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
    else:
        pts = cnt.reshape(-1, 2)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
    h_m, w_m = roi_mask.shape[:2]
    cx_i = min(max(0, int(round(cx))), w_m - 1)
    cy_i = min(max(0, int(round(cy))), h_m - 1)
    return bool(roi_mask[cy_i, cx_i] > 0)


@app.post("/fetch_imagery")
async def fetch_imagery(
    center_lat: float = Form(...),
    center_lon: float = Form(...),
    width_m: float = Form(...),
    length_m: float = Form(...),
    include_ndvi: bool = Form(False),
    preferred_source: str = Form("google"),
    target_zoom: int = Form(0),
    roi_polygon: Optional[str] = Form(None),
    site_preset: Optional[str] = Form(None),
):
    """
    Downloads Google Satellite VHR (z=18..22 via scale=2 tiles),
    writes a georeferenced GeoTIFF, and returns metadata.
    """
    def _unwrap(val):
        return val.default if hasattr(val, "default") else val

    center_lat = float(_unwrap(center_lat))
    center_lon = float(_unwrap(center_lon))
    width_m = float(_unwrap(width_m))
    length_m = float(_unwrap(length_m))
    target_zoom = int(_unwrap(target_zoom) or 0)
    parsed_roi = parse_roi_polygon(_unwrap(roi_polygon))
    site_preset_val = _unwrap(site_preset)

    # Calculate geographic bounding box in WGS84
    EarthRadius = 6378137.0
    c_lat_rad = center_lat * math.pi / 180.0
    d_lat = (length_m / 2.0 / EarthRadius) * (180.0 / math.pi)
    d_lon = (width_m / 2.0 / (EarthRadius * math.cos(c_lat_rad))) * (180.0 / math.pi)

    west = center_lon - d_lon
    south = center_lat - d_lat
    east = center_lon + d_lon
    north = center_lat + d_lat

    timestamp = int(time.time())
    zoom = (
        int(target_zoom)
        if 18 <= int(target_zoom) <= 22
        else get_dynamic_zoom(width_m, length_m)
    )
    imagery_source = "google.satellite.vhr"

    try:
        img_bgr, actual_zoom, _ = stitch_google_satellite(
            west,
            south,
            east,
            north,
            zoom=zoom,
            high_dpi=True,
            return_meta=True,
        )
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Stitched Google VHR image is empty")
        zoom = actual_zoom
        gsd_m = web_mercator_gsd_m(center_lat, actual_zoom, tile_px=256)
        print(f"Google satellite VHR mosaic OK at z={zoom}, shape={img_bgr.shape}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed Google VHR fetch: {e}",
        )

    # GeoTIFF (authoritative path for segmentation) + JPG preview for Leaflet
    vhr_basename = f"vhr_{timestamp}"
    geotiff_filename = f"{vhr_basename}.tif"
    preview_filename = f"{vhr_basename}_preview.jpg"
    geotiff_path = os.path.join(UPLOAD_DIR, geotiff_filename)
    preview_path = os.path.join(UPLOAD_DIR, preview_filename)

    img_w, img_h = write_rgb_geotiff(geotiff_path, img_bgr, west, south, east, north)
    # Preview: cap long edge for browser overlay performance
    preview_bgr = img_bgr
    max_prev = 2048
    ph, pw = preview_bgr.shape[:2]
    if max(ph, pw) > max_prev:
        scale = max_prev / max(ph, pw)
        preview_bgr = cv2.resize(
            preview_bgr, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA
        )
    cv2.imwrite(preview_path, preview_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # Also report measured GSD from written raster
    span_m_x = width_m
    span_m_y = length_m
    gsd_m_x = span_m_x / max(img_w, 1)
    gsd_m_y = span_m_y / max(img_h, 1)
    gsd_m_measured = (gsd_m_x + gsd_m_y) / 2.0

    # Sidecar metadata (path points at GeoTIFF used for segmentation)
    metadata = {
        "filename": geotiff_filename,
        "file_path": os.path.join("uploads", geotiff_filename).replace("\\", "/"),
        "is_geotiff": True,
        "width": img_w,
        "height": img_h,
        "bounds": {
            "south": south,
            "west": west,
            "north": north,
            "east": east,
        },
        "crs": "EPSG:4326",
        "preview_url": f"/static_uploads/{preview_filename}",
        "ndvi_file_path": None,
        "ndvi_preview_url": None,
        "ndvi_date": None,
        "include_ndvi": False,
        "imagery_source": imagery_source,
        "zoom": zoom,
        "gsd_m": round(gsd_m_measured, 4),
        "gsd_m_nominal": round(gsd_m, 4),
        "high_dpi": True,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "width_m": width_m,
        "length_m": length_m,
        "pipeline_mode": "geo",
        "georef_quality": "satellite_approx",
        "mode_message": (
            f"GEO mode ({imagery_source} z{zoom}, GSD ~{gsd_m_measured:.3f} m/px): "
            "real WGS84 bounds ready for SAM 3 + Watershed."
        ),
    }
    if site_preset_val:
        metadata["site_preset"] = str(site_preset_val)
    if parsed_roi and len(parsed_roi) >= 3:
        metadata["roi_polygon"] = parsed_roi
    metadata["updated_at"] = int(time.time() * 1000)
    metadata = _ensure_auto_tune_metadata(metadata)
    save_sidecar_metadata(geotiff_path, metadata)
    # Also mirror sidecar next to preview and last_roi_session.json for convenience
    save_sidecar_metadata(preview_path, metadata)
    _write_last_roi_session(metadata)

    return metadata


@app.post("/segment")
async def segment_canopy(
    file_path: str = Form(...),
    model_type: str = Form("canopy_v2"),
    v2_tile_size: int = Form(1280),
    v2_overlap: int = Form(384),
    v2_coverage_threshold: float = Form(0.43),
    v2_instance_threshold: float = Form(0.18),
    v2_max_tiles: int = Form(0),
    sam3_prompt: Optional[str] = Form("tree"),
    min_area_m2: float = Form(0.3),
    max_area_m2: float = Form(2000.0),
    default_height_m: float = Form(10.0),
    conf_threshold: Optional[float] = Form(None),
    vari_threshold: Optional[float] = Form(None),
    circularity_threshold: Optional[float] = Form(None),
    nms_threshold: Optional[float] = Form(None),
    use_auto: Optional[bool] = Form(None),
    roi_polygon: Optional[str] = Form(None),
):
    """
    Unified Canopy V2 pipeline: native tiled SAM 3 instances + visible RGB
    vegetation support + marker-controlled watershed dense crown candidates +
    unresolved crown groups.
    """
    from canopy_v2 import (
        assign_adjacent_distinct_colors,
        compute_strict_foliage_mask,
        compute_woody_canopy_relief_mask,
        contours_from_mask,
        reconcile_canopy_outputs,
        run_deepforest_crowns,
        run_sam3_tiled,
        split_dense_canopy_watershed,
        visible_vegetation_support,
    )

    def _unwrap(val):
        return val.default if hasattr(val, "default") else val

    file_path = _unwrap(file_path)
    model_type = _unwrap(model_type)
    v2_tile_size = _unwrap(v2_tile_size)
    v2_overlap = _unwrap(v2_overlap)
    v2_coverage_threshold = _unwrap(v2_coverage_threshold)
    v2_instance_threshold = _unwrap(v2_instance_threshold)
    v2_max_tiles = _unwrap(v2_max_tiles)
    sam3_prompt = _unwrap(sam3_prompt)
    min_area_m2 = _unwrap(min_area_m2)
    max_area_m2 = _unwrap(max_area_m2)
    default_height_m = _unwrap(default_height_m)
    conf_threshold = _unwrap(conf_threshold)
    vari_threshold = _unwrap(vari_threshold)
    circularity_threshold = _unwrap(circularity_threshold)
    nms_threshold = _unwrap(nms_threshold)
    use_auto = _unwrap(use_auto)
    roi_polygon = _unwrap(roi_polygon)

    segment_started = time.perf_counter()
    if not isinstance(file_path, str) or not file_path:
        raise HTTPException(status_code=404, detail="Uploaded file not found.")
    if not os.path.exists(file_path):
        candidate_upload = os.path.join(UPLOAD_DIR, os.path.basename(file_path))
        if os.path.exists(candidate_upload):
            file_path = candidate_upload
        else:
            raise HTTPException(status_code=404, detail="Uploaded file not found.")

    if conf_threshold is not None and float(v2_instance_threshold) in (0.18, 0.25):
        v2_instance_threshold = float(conf_threshold)

    if not 768 <= int(v2_tile_size) <= 2048:
        raise HTTPException(
            status_code=422, detail="V2 tile size must be 768–2048 px."
        )
    if not 0 <= int(v2_overlap) < int(v2_tile_size) // 2:
        raise HTTPException(
            status_code=422,
            detail="V2 overlap must be non-negative and less than half the tile size.",
        )
    if not 0.1 <= float(v2_coverage_threshold) <= 0.9:
        raise HTTPException(
            status_code=422,
            detail="V2 coverage threshold must be between 0.1 and 0.9.",
        )
    if not 0.05 <= float(v2_instance_threshold) <= 0.9:
        raise HTTPException(
            status_code=422,
            detail="V2 instance threshold must be between 0.05 and 0.9.",
        )

    clear_progress_log()
    log_progress("Initializing canopy extraction (Algorithm: canopy_v2)")

    # Try loading sidecar JSON metadata first
    log_progress("Loading sidecar metadata and georeferencing...")
    sidecar = load_sidecar_metadata(file_path)
    if sidecar:
        pm = sidecar.get("pipeline_mode", "unknown")
        log_progress(f"Pipeline mode: {pm}")
        if sidecar.get("mode_message"):
            log_progress(sidecar["mode_message"][:200])

    crs = None
    transform = None
    is_georeferenced = False
    ndvi_src = None

    if sidecar and sidecar.get("bounds"):
        try:
            width = sidecar["width"]
            height = sidecar["height"]
            crs = rasterio.crs.CRS.from_string(sidecar["crs"]) if sidecar.get("crs") else rasterio.crs.CRS.from_epsg(4326)
            is_georeferenced = True

            b = sidecar["bounds"]
            from rasterio.transform import from_origin
            pixel_size_lon = (b["east"] - b["west"]) / width
            pixel_size_lat = (b["north"] - b["south"]) / height
            transform = from_origin(b["west"], b["north"], pixel_size_lon, pixel_size_lat)

            geo_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            geod = Geod(ellps="WGS84")

            if sidecar.get("ndvi_file_path") and os.path.exists(sidecar["ndvi_file_path"]):
                ndvi_src = rasterio.open(sidecar["ndvi_file_path"])
            log_progress("Loaded georeferenced metadata from sidecar JSON.")
        except Exception as e:
            print(f"Error loading sidecar georeferencing: {e}")
            is_georeferenced = False

    if not is_georeferenced:
        try:
            log_progress("Reading geospatial headers using Rasterio...")
            with rasterio.open(file_path) as src:
                width = src.width
                height = src.height
                transform = src.transform
                crs = src.crs
                if crs:
                    is_georeferenced = True
                    geo_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    geod = Geod(ellps="WGS84")
                if sidecar and sidecar.get("ndvi_file_path") and os.path.exists(sidecar["ndvi_file_path"]):
                    ndvi_src = rasterio.open(sidecar["ndvi_file_path"])
            log_progress("Loaded georeferenced metadata from TIFF headers.")
        except Exception as e:
            print(f"Failed to read image with rasterio: {e}")

    # Load NDVI to memory immediately to avoid high disk latency inside extraction loops
    ndvi_data = None
    ndvi_transform = None
    if ndvi_src is not None:
        try:
            ndvi_data = ndvi_src.read(1)
            ndvi_transform = ndvi_src.transform
            ndvi_src.close()
            ndvi_src = (ndvi_data, ndvi_transform)
            log_progress("Loaded Sentinel-2 NDVI raster in-memory.")
        except Exception as e:
            print(f"Error loading NDVI to memory: {e}")
            ndvi_src = None

    # Read image pixels (prefer rasterio for GeoTIFF; OpenCV for JPG/PNG)
    log_progress("Reading image pixels...")
    try:
        img_rgb = None
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".tif", ".tiff"]:
            try:
                with rasterio.open(file_path) as src:
                    if src.count >= 3:
                        img_rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
                    else:
                        band = src.read(1)
                        img_rgb = np.stack([band, band, band], axis=-1)
                    if img_rgb.dtype != np.uint8:
                        img_rgb = ((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-6) * 255).astype(np.uint8)
            except Exception as e:
                print(f"rasterio read failed, trying OpenCV: {e}")
        if img_rgb is None:
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                raise HTTPException(status_code=400, detail="Error reading image file.")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading image pixels: {e}")

    # Normalize image to 8-bit RGB if necessary
    if img_rgb.dtype != np.uint8:
        img_rgb = ((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min()) * 255).astype(np.uint8)

    # Compute image-wide VARI greenness (stabilized against Rayleigh shadow B > G+R sign flip)
    log_progress("Computing image-wide VARI greenness index...")
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)
    denom_vari = np.where(
        G >= R,
        np.maximum(G + R - B, 6.0),
        np.where(np.abs(G + R - B) < 6.0, -6.0, G + R - B),
    )
    vari_img = (G - R) / denom_vari
    vari_img = np.clip(vari_img, -1.0, 1.0)
    from canopy_v2 import _compute_scene_haze_offset
    _, scene_haze_offset = _compute_scene_haze_offset(img_rgb)

    features = []
    v2_coverage_features = []
    v2_group_features = []
    v2_dense_candidate_features = []

    def _v2_feature_from_contour(
        contour,
        feature_class: str,
        feature_id: int,
        confidence: float = 0.0,
        geometry_quality_override: Optional[str] = None,
    ):
        """Convert one V2 pixel contour through the shared coordinate logic."""
        converted = []
        geometry_quality = geometry_quality_override or {
            "individual_tree": "sam3_native",
            "dense_crown_candidate": "watershed_dense_review",
            "crown_group": "rgb_review_group",
            "canopy_extent": "semantic_extent",
        }.get(feature_class, "semantic_group")
        is_individual = feature_class == "individual_tree"
        is_crown = feature_class in (
            "individual_tree",
            "dense_crown_candidate",
        ) or geometry_quality in (
            "sam3_native",
            "deepforest_watershed",
            "watershed_forest_crown",
            "watershed_dense_review",
        )
        process_contour_enhanced(
            contour=contour,
            tree_id=feature_id,
            img_rgb=img_rgb,
            vari_img=vari_img,
            height_map=None,
            ndvi_src=ndvi_src,
            transform=transform,
            crs=crs,
            is_georeferenced=is_georeferenced,
            geo_transformer=geo_transformer if is_georeferenced else None,
            geod=geod if is_georeferenced else None,
            min_area_m2=float(min_area_m2) if is_individual else 0.05,
            max_area_m2=float(max_area_m2) if is_individual else 1_000_000_000.0,
            vari_threshold=0.0 if is_crown else -1.0,
            circularity_threshold=0.0,
            default_height_m=default_height_m,
            features=converted,
            reject_water=True if is_crown else False,
            detector_score=confidence,
            geometry_quality=geometry_quality,
            haze_offset=scene_haze_offset,
        )
        if not converted:
            return None
        feature = converted[0]
        feature["id"] = feature_id
        props = feature["properties"]
        props["tree_id"] = feature_id
        props["feature_class"] = feature_class
        props["countable"] = is_individual
        if is_individual:
            props["measurement_note"] = (
                "Crown-scaled allometric estimate anchored to default height (no LiDAR CHM)."
            )
        else:
            props["estimated_height_m"] = None
            props["estimated_dbh_cm"] = None
            props["estimated_volume_m3"] = None
            props["measurement_note"] = (
                "Non-instance canopy layer; allometry intentionally omitted."
            )
        return feature

    def _estimate_gsd_m() -> float:
        if sidecar and isinstance(sidecar.get("gsd_m"), (int, float)):
            return max(float(sidecar["gsd_m"]), 0.01)
        if is_georeferenced and transform is not None:
            try:
                if crs is not None and getattr(crs, "is_projected", False):
                    return abs(float(transform.a))
                c_lat = 0.0
                if sidecar and sidecar.get("bounds"):
                    b = sidecar["bounds"]
                    c_lat = (float(b["south"]) + float(b["north"])) / 2.0
                else:
                    cand_lat = float(transform.f) + float(transform.e) * (
                        float(height) / 2.0
                    )
                    if -85.0 <= cand_lat <= 85.0:
                        c_lat = cand_lat
                return (
                    abs(float(transform.a))
                    * 111320.0
                    * math.cos(math.radians(c_lat))
                )
            except Exception:  # nosec B110
                pass
        return 0.1

    gsd_m = _estimate_gsd_m()
    prompt = sam3_prompt or "tree"

    MAX_V2_DIM = 5200
    work_rgb = img_rgb
    work_w, work_h = width, height
    work_gsd = gsd_m
    scale_back = False
    if max(width, height) > MAX_V2_DIM:
        scale_factor = float(MAX_V2_DIM) / float(max(width, height))
        work_w = max(64, int(round(width * scale_factor)))
        work_h = max(64, int(round(height * scale_factor)))
        work_rgb = cv2.resize(
            img_rgb, (work_w, work_h), interpolation=cv2.INTER_AREA
        )
        work_gsd = gsd_m / scale_factor
        scale_back = True
        log_progress(
            f"Auto-scaling large raster ({width}x{height} @ {gsd_m:.4f} m/px) "
            f"-> ({work_w}x{work_h} @ {work_gsd:.4f} m/px) for optimal SAM 3 receptive field."
        )
    elif (
        is_georeferenced
        and sidecar
        and sidecar.get("georef_quality") == "satellite_approx"
        and gsd_m < 0.05
    ):
        work_w = max(64, int(round(width * 0.5)))
        work_h = max(64, int(round(height * 0.5)))
        work_rgb = cv2.resize(
            img_rgb, (work_w, work_h), interpolation=cv2.INTER_AREA
        )
        work_gsd = gsd_m * 2.0
        scale_back = True
        log_progress(
            f"Auto-scaling z=22 satellite raster ({width}x{height} @ {gsd_m:.4f} m/px) "
            f"-> ({work_w}x{work_h} @ {work_gsd:.4f} m/px) for optimal SAM 3 receptive field."
        )

    log_progress(
        "CANOPY V2: native tiled coverage + instances + unresolved groups."
    )
    log_progress(
        f"V2 settings: tile={v2_tile_size}, overlap={v2_overlap}, "
        f"coverage={v2_coverage_threshold:.2f}, "
        f"instance={v2_instance_threshold:.2f}, GSD~{work_gsd:.3f}."
    )

    probability, vegetation_support = visible_vegetation_support(
        work_rgb,
        tile_size=int(v2_tile_size),
        overlap=int(v2_overlap),
        threshold=float(v2_coverage_threshold),
        progress=log_progress,
    )
    strict_foliage = compute_strict_foliage_mask(work_rgb)
    woody_relief = compute_woody_canopy_relief_mask(
        work_rgb, strict_foliage, gsd_m=work_gsd
    )

    parsed_roi_input = parse_roi_polygon(roi_polygon)
    if parsed_roi_input and sidecar is not None:
        if sidecar.get("roi_polygon") != parsed_roi_input:
            sidecar["roi_polygon"] = parsed_roi_input
            try:
                save_sidecar_metadata(file_path, sidecar)
            except Exception as e:
                print(f"Could not persist updated roi_polygon to sidecar: {e}")
    active_roi = parsed_roi_input or (
        parse_roi_polygon(sidecar.get("roi_polygon")) if sidecar else None
    )
    roi_mask_full = None
    roi_mask_work = None
    if active_roi and sidecar and sidecar.get("bounds"):
        roi_mask_full = build_roi_pixel_mask(
            active_roi, width, height, sidecar["bounds"]
        )
        roi_mask_work = build_roi_pixel_mask(
            active_roi, work_w, work_h, sidecar["bounds"]
        )
        if roi_mask_work is not None:
            vegetation_support = cv2.bitwise_and(
                vegetation_support, roi_mask_work
            )
            strict_foliage = cv2.bitwise_and(strict_foliage, roi_mask_work)
            woody_relief = cv2.bitwise_and(woody_relief, roi_mask_work)
            log_progress(
                f"Applied drawable polygon ROI mask ({len(active_roi)} vertices) to search area."
            )

    log_progress(
        f"RGB support covers {100.0 * cv2.countNonZero(vegetation_support) / (work_w * work_h):.1f}% "
        "of image pixels (vegetation candidate, not yet tree-only)."
    )

    instances = []
    sam_union = np.zeros((work_h, work_w), dtype=np.uint8)
    sam_stats = {
        "tiles_processed": 0,
        "tiles_total": 0,
        "failed_tiles": 0,
        "skipped_barren_tiles": 0,
        "vehicle_vetoed": 0,
    }
    sam_error = None
    try:
        log_progress(f"Loading SAM 3 for native tiled prompt: '{prompt}'...")
        sam = get_sam3_model()
        min_instance_px = max(
            16,
            int(round(max(float(min_area_m2), 0.1) / max(work_gsd ** 2, 1e-6))),
        )
        default_single_max_m2 = 380.0 if work_gsd >= 0.22 else 260.0
        user_max_m2 = float(max_area_m2) if max_area_m2 is not None else 2000.0
        is_default_max_m2 = (
            abs(user_max_m2 - 2000.0) <= 1.0 or abs(user_max_m2 - 450.0) <= 1.0
        )
        single_crown_max_m2 = (
            default_single_max_m2
            if is_default_max_m2
            else max(10.0, user_max_m2)
        )
        max_instance_px = max(
            min_instance_px + 1,
            int(round(single_crown_max_m2 / max(work_gsd ** 2, 1e-6))),
        )
        instances, sam_union, sam_stats = run_sam3_tiled(
            work_rgb,
            sam,
            prompt=prompt,
            tile_size=int(v2_tile_size),
            overlap=int(v2_overlap),
            confidence=float(v2_instance_threshold),
            min_area_px=min_instance_px,
            max_area_px=max_instance_px,
            max_tiles=max(0, int(v2_max_tiles)),
            progress=log_progress,
            vegetation_support=vegetation_support,
            probability_map=probability,
            gsd_m=work_gsd,
            strict_foliage=strict_foliage,
            woody_relief_mask=woody_relief,
        )
        if roi_mask_work is not None and instances:
            instances = [
                item
                for item in instances
                if _contour_centroid_in_mask(item["contour"], roi_mask_work)
            ]
    except Exception as error:
        sam_error = str(error)
        log_progress(
            f"SAM 3 tiled stage unavailable; continuing with coverage/groups only: {error}"
        )

    df_boxes = []
    df_support = np.zeros((work_h, work_w), dtype=np.uint8)
    df_stats = {"df_boxes": 0, "df_verified": 0}
    try:
        df_model = get_deepforest_model()
        if df_model is not None:
            log_progress("Running DeepForest 2.1.0 nadir tree crown detector...")
            df_boxes, df_support, df_stats = run_deepforest_crowns(
                work_rgb,
                df_model,
                vegetation_support=vegetation_support,
                probability_map=probability,
                gsd_m=work_gsd,
                score_threshold=0.20,
                progress=log_progress,
                strict_foliage=strict_foliage,
                woody_relief_mask=woody_relief,
                sam_union=sam_union,
            )
    except Exception as df_error:
        log_progress(f"DeepForest stage skipped: {df_error}")

    min_group_px = max(100, int(round(2.0 / max(work_gsd ** 2, 1e-6))))
    scene_gray_p1 = (
        float(np.percentile(cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY), 1))
        if work_rgb.size > 0
        else 0.0
    )
    if np.any(sam_union > 0):
        sam_df_gate = cv2.dilate(
            sam_union,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        df_support_gated = cv2.bitwise_and(df_support, sam_df_gate)
        if scene_gray_p1 > 32.0 and woody_relief is not None:
            df_support_gated = cv2.bitwise_or(
                df_support_gated, cv2.bitwise_and(df_support, woody_relief)
            )
    else:
        df_support_gated = df_support
    semantic_union = cv2.bitwise_or(sam_union, df_support_gated)
    coverage_mask, group_contours, reconcile_stats = reconcile_canopy_outputs(
        work_rgb,
        vegetation_support,
        semantic_union,
        min_group_area_px=min_group_px,
        strict_foliage=strict_foliage,
        woody_relief_mask=woody_relief,
        gsd_m=work_gsd,
    )
    coverage_contours = contours_from_mask(
        coverage_mask, min_area_px=min_group_px
    )

    dense_proposals = []
    dense_stats = {
        "dense_tiles": 0,
        "candidate_count": 0,
        "method": "marker_controlled_rgb_watershed",
    }
    try:
        log_progress(
            "Splitting unresolved semantic canopy with "
            "Global Compact Geodesic Watershed..."
        )
        proposal_support = coverage_mask.copy()
        if cv2.countNonZero(df_support_gated) > 0:
            proposal_support = cv2.bitwise_or(proposal_support, df_support_gated)
        proposal_support = cv2.bitwise_and(proposal_support, strict_foliage)
        dense_proposals, dense_stats = split_dense_canopy_watershed(
            work_rgb,
            proposal_support,
            [item["contour"] for item in instances],
            gsd_m=work_gsd,
            tile_size=int(v2_tile_size),
            overlap=int(v2_overlap),
            min_dense_ratio=0.20,
            min_area_px=max(
                24, int(round(max(float(min_area_m2), 1.2) / max(work_gsd ** 2, 1e-6)))
            ),
            max_area_px=max(
                100,
                int(
                    round(
                        (
                            320.0
                            if (
                                abs(float(max_area_m2 or 2000.0) - 2000.0) <= 1.0
                                or abs(float(max_area_m2 or 450.0) - 450.0) <= 1.0
                                or abs(
                                    float(max_area_m2 or 2000.0)
                                    - (380.0 if work_gsd >= 0.22 else 260.0)
                                )
                                <= 1.0
                            )
                            else max(15.0, float(max_area_m2))
                        )
                        / max(work_gsd ** 2, 1e-6)
                    )
                ),
            ),
            progress=log_progress,
            deepforest_boxes=df_boxes,
            strict_foliage=strict_foliage,
            woody_relief_mask=woody_relief,
        )
        if roi_mask_work is not None and dense_proposals:
            dense_proposals = [
                item
                for item in dense_proposals
                if _contour_centroid_in_mask(item["contour"], roi_mask_work)
            ]
    except Exception as dense_error:
        dense_stats["error"] = str(dense_error)
        log_progress(f"Dense-crown proposal stage skipped: {dense_error}")

    if scale_back:
        sx = float(width) / float(work_w)
        sy = float(height) / float(work_h)

        def _upscale_contour(cnt: np.ndarray) -> np.ndarray:
            c = np.asarray(cnt, dtype=np.float32).copy()
            c[:, 0, 0] = np.clip(np.round(c[:, 0, 0] * sx), 0, width - 1)
            c[:, 0, 1] = np.clip(np.round(c[:, 0, 1] * sy), 0, height - 1)
            return c.astype(np.int32)

        for inst in instances:
            inst["contour"] = _upscale_contour(inst["contour"])
            inst["area"] = float(cv2.contourArea(inst["contour"]))
        group_contours = [_upscale_contour(c) for c in group_contours]
        coverage_contours = [_upscale_contour(c) for c in coverage_contours]
        for prop in dense_proposals:
            prop["contour"] = _upscale_contour(prop["contour"])
            prop["area"] = float(cv2.contourArea(prop["contour"]))
        coverage_mask = cv2.resize(
            coverage_mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
        if roi_mask_full is not None:
            coverage_mask = cv2.bitwise_and(coverage_mask, roi_mask_full)
        probability = cv2.resize(
            probability, (width, height), interpolation=cv2.INTER_LINEAR
        )
        reconcile_stats["coverage_pixels"] = int(
            cv2.countNonZero(coverage_mask)
        )
        reconcile_stats["review_group_pixels"] = int(
            sum(cv2.contourArea(c) for c in group_contours)
        )

    for instance in instances:
        feature = _v2_feature_from_contour(
            instance["contour"],
            "individual_tree",
            len(features) + 1,
            instance.get("detector_score", instance["score"]),
        )
        if feature:
            feature["properties"]["source_model"] = instance.get(
                "source_model", "sam3"
            )
            features.append(feature)

    promoted_watershed_count = 0
    for proposal in dense_proposals:
        raw_det_score = float(
            proposal.get("detector_score", proposal.get("score", 0.45))
        )
        prop_conf_pct = float(
            proposal.get(
                "confidence_pct", float(proposal.get("score", 0.0)) * 100.0
            )
        )
        if prop_conf_pct >= 36.0:
            gq_override = proposal.get(
                "geometry_quality", "watershed_forest_crown"
            )
            promoted = _v2_feature_from_contour(
                proposal["contour"],
                "individual_tree",
                len(features) + 1,
                raw_det_score,
                geometry_quality_override=gq_override,
            )
            if promoted:
                promoted["properties"]["source_model"] = proposal.get(
                    "source_model",
                    "deepforest_2.1"
                    if gq_override == "deepforest_watershed"
                    else "compact_watershed",
                )
                promoted["properties"]["measurement_note"] = (
                    "Closed-canopy watershed crown partition inside verified forest stand."
                )
                features.append(promoted)
                promoted_watershed_count += 1
                continue
        feature = _v2_feature_from_contour(
            proposal["contour"],
            "dense_crown_candidate",
            len(v2_dense_candidate_features) + 1,
            raw_det_score,
        )
        if feature:
            feature["properties"]["countable"] = False
            feature["properties"]["qa_note"] = (
                "Marker-controlled RGB watershed partition inside "
                "semantic canopy; click-review before counting."
            )
            v2_dense_candidate_features.append(feature)
    if dense_proposals:
        log_progress(
            f"Dense watershed promoted {promoted_watershed_count} forest crowns "
            f"and produced {len(v2_dense_candidate_features)} review candidates."
        )

    for contour in group_contours:
        feature = _v2_feature_from_contour(
            contour,
            "crown_group",
            len(v2_group_features) + 1,
            0.0,
        )
        if feature:
            v2_group_features.append(feature)

    for contour in coverage_contours:
        feature = _v2_feature_from_contour(
            contour,
            "canopy_extent",
            len(v2_coverage_features) + 1,
            0.0,
        )
        if feature:
            v2_coverage_features.append(feature)

    assign_adjacent_distinct_colors(features)
    if v2_dense_candidate_features:
        assign_adjacent_distinct_colors(v2_dense_candidate_features)

    conf_pcts = [
        float(f.get("properties", {}).get("confidence_pct", 0.0))
        for f in features
    ]
    mean_confidence_pct = (
        round(float(np.mean(conf_pcts)), 1) if conf_pcts else 0.0
    )
    high_confidence_count = int(sum(1 for c in conf_pcts if c >= 80.0))
    good_confidence_count = int(
        sum(1 for c in conf_pcts if 60.0 <= c < 80.0)
    )
    moderate_confidence_count = int(
        sum(1 for c in conf_pcts if 45.0 <= c < 60.0)
    )
    low_confidence_count = int(sum(1 for c in conf_pcts if c < 45.0))

    timestamp = int(time.time())
    coverage_name = f"v2_coverage_{timestamp}.png"
    probability_name = f"v2_probability_{timestamp}.png"
    cv2.imwrite(os.path.join(UPLOAD_DIR, coverage_name), coverage_mask)
    cv2.imwrite(os.path.join(UPLOAD_DIR, probability_name), probability)
    coverage_ratio = reconcile_stats["coverage_pixels"] / float(width * height)
    unresolved_pixels = reconcile_stats["review_group_pixels"]
    unresolved_ratio = unresolved_pixels / max(
        1, reconcile_stats["coverage_pixels"] + unresolved_pixels
    )
    v2_summary = {
        "pipeline_version": "canopy_v2_prototype",
        "task": "coverage_instances_dense_candidates_groups",
        "prompt": prompt,
        "tile_size": int(v2_tile_size),
        "tile_overlap": int(v2_overlap),
        "tiles_processed": sam_stats["tiles_processed"],
        "tiles_total": sam_stats["tiles_total"],
        "failed_tiles": sam_stats["failed_tiles"],
        "skipped_barren_tiles": sam_stats.get("skipped_barren_tiles", 0),
        "vehicle_vetoed": sam_stats.get("vehicle_vetoed", 0),
        "individual_count": len(features),
        "mean_confidence_pct": mean_confidence_pct,
        "high_confidence_count": high_confidence_count,
        "good_confidence_count": good_confidence_count,
        "moderate_confidence_count": moderate_confidence_count,
        "low_confidence_count": low_confidence_count,
        "group_count": len(v2_group_features),
        "dense_candidate_count": len(v2_dense_candidate_features),
        "dense_tiles_processed": dense_stats["dense_tiles"],
        "dense_candidate_method": dense_stats.get("method"),
        "dense_marker_spacing_px": dense_stats.get("spacing_px"),
        "deepforest_boxes": df_stats.get("df_boxes", 0),
        "deepforest_verified": df_stats.get("df_verified", 0),
        "coverage_polygon_count": len(v2_coverage_features),
        "coverage_percent": round(100.0 * coverage_ratio, 2),
        "unresolved_canopy_percent": round(100.0 * unresolved_ratio, 2),
        "rgb_supported_coverage_pixels": reconcile_stats[
            "rgb_supported_pixels"
        ],
        "runtime_seconds": round(time.perf_counter() - segment_started, 2),
        "sam_error": sam_error,
        "coverage_preview_url": f"/static_uploads/{coverage_name}",
        "probability_preview_url": f"/static_uploads/{probability_name}",
        "count_note": (
            "Only SAM individual_tree features enter the provisional count. "
            "Dense watershed candidates and RGB crown groups are clickable "
            "review layers and are not silently counted."
        ),
    }
    if active_roi:
        v2_summary["roi_polygon"] = active_roi
    manifest_name = f"v2_run_{timestamp}.json"
    try:
        with open(
            os.path.join(UPLOAD_DIR, manifest_name),
            "w",
            encoding="utf-8",
        ) as manifest_file:
            json.dump(v2_summary, manifest_file, indent=2)
        v2_summary["run_manifest_url"] = (
            f"/static_uploads/{manifest_name}"
        )
    except OSError as manifest_error:
        log_progress(f"V2 run manifest could not be written: {manifest_error}")

    if ndvi_src and hasattr(ndvi_src, "close"):
        ndvi_src.close()

    log_progress(
        f"Extraction complete: {len(features)} provisional crowns "
        f"(mean confidence {mean_confidence_pct:.1f}%) + "
        f"{len(v2_dense_candidate_features)} dense review candidates."
    )
    pipeline_mode = "geo"
    georef_quality = "unknown"
    if sidecar:
        pipeline_mode = sidecar.get("pipeline_mode") or (
            "geo" if sidecar.get("georef_quality") in ("survey", "satellite_approx") else "image_only"
        )
        georef_quality = sidecar.get("georef_quality", "unknown")
    for f in (
        features
        + v2_coverage_features
        + v2_group_features
        + v2_dense_candidate_features
    ):
        f.setdefault("properties", {})
        f["properties"]["pipeline_mode"] = pipeline_mode
        f["properties"]["georef_quality"] = georef_quality
        if pipeline_mode == "image_only":
            f["properties"]["area_is_estimated"] = True
            f["properties"]["coordinate_note"] = (
                "Preview placement / assumed GSD — not survey-grade"
            )
    top_properties = {
        "pipeline_version": "canopy_v2_prototype",
        "pipeline_mode": pipeline_mode,
        "georef_quality": georef_quality,
        "feature_count": len(features),
        "mean_confidence_pct": mean_confidence_pct,
        "high_confidence_count": high_confidence_count,
        "good_confidence_count": good_confidence_count,
        "moderate_confidence_count": moderate_confidence_count,
        "low_confidence_count": low_confidence_count,
        "coverage_feature_count": len(v2_coverage_features),
        "group_feature_count": len(v2_group_features),
        "candidate_feature_count": len(v2_dense_candidate_features),
        "source_file": sidecar.get("filename") if sidecar else None,
        "mode_message": (sidecar or {}).get("mode_message", ""),
        "v2_summary": v2_summary,
        "v2_meta": v2_summary,
    }
    if active_roi:
        top_properties["roi_polygon"] = active_roi
    geojson_response = {
        "type": "FeatureCollection",
        "features": features,
        "coverage_features": v2_coverage_features,
        "group_features": v2_group_features,
        "candidate_features": v2_dense_candidate_features,
        "summary": v2_summary,
        "v2_meta": v2_summary,
        "properties": top_properties,
    }
    try:
        stem = os.path.splitext(os.path.basename(file_path))[0]
        res_filename = f"{stem}_result.geojson"
        res_path = os.path.join(UPLOAD_DIR, res_filename)
        with open(res_path, "w", encoding="utf-8") as rf:
            json.dump(geojson_response, rf)
        if sidecar is not None:
            sidecar["result_geojson_url"] = f"/static_uploads/{res_filename}"
            sidecar["last_segmented_at"] = int(time.time() * 1000)
            save_sidecar_metadata(file_path, sidecar)
            _write_last_roi_session(sidecar)
    except Exception as persist_err:
        print(f"Could not persist result_geojson_url to session: {persist_err}")
    return geojson_response


def process_contour_enhanced(
    contour,
    tree_id: int,
    img_rgb: np.ndarray,
    vari_img: np.ndarray,
    height_map: Optional[np.ndarray],
    ndvi_src,
    transform,
    crs,
    is_georeferenced: bool,
    geo_transformer,
    geod,
    min_area_m2: float,
    max_area_m2: float,
    vari_threshold: float,
    circularity_threshold: float,
    default_height_m: float,
    features: list,
    reject_water: bool = True,
    detector_score: Optional[float] = None,
    geometry_quality: Optional[str] = None,
    haze_offset: float = 0.0,
):
    """Helper function to calculate area, circularity, health (NDVI/VARI), and allometric attributes."""
    # Keep contour inside image so crop/mask sizes always match (ellipse can spill OOB)
    img_h, img_w = img_rgb.shape[:2]
    contour = np.asarray(contour).copy()
    if contour.ndim == 3 and contour.shape[1] == 1:
        contour[:, 0, 0] = np.clip(contour[:, 0, 0], 0, img_w - 1)
        contour[:, 0, 1] = np.clip(contour[:, 0, 1], 0, img_h - 1)
    if geometry_quality in (
        "sam3_native",
        "deepforest_watershed",
        "watershed_forest_crown",
        "watershed_candidate",
    ):
        approx_contour = contour
    else:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx_contour) < 3:
        return

    # Compute pixel centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx_pixel = M["m10"] / M["m00"]
        cy_pixel = M["m01"] / M["m00"]
    else:
        cx_pixel = np.mean(approx_contour[:, 0, 0])
        cy_pixel = np.mean(approx_contour[:, 0, 1])

    # Get polygon coordinate vertices (pixels)
    poly_pixel_coords = [([float(pt[0][0]), float(pt[0][1])]) for pt in approx_contour]
    if poly_pixel_coords[0] != poly_pixel_coords[-1]:
        poly_pixel_coords.append(poly_pixel_coords[0])

    # Geo-coordinate conversion & Area Calculation
    area_m2 = 0.0
    if is_georeferenced:
        mxs = []
        mys = []
        for px, py in poly_pixel_coords:
            mx, my = transform * (px, py)
            mxs.append(float(mx))
            mys.append(float(my))

        crs_is_wgs84 = (
            str(crs).upper() in ("EPSG:4326", "OGC:CRS84")
            or getattr(crs, "to_epsg", lambda: None)() == 4326
        )
        if crs_is_wgs84 or geo_transformer is None:
            lons = mxs
            lats = mys
            mx_c, my_c = transform * (cx_pixel, cy_pixel)
            c_lon = float(mx_c)
            c_lat = float(my_c)
        else:
            lons_res, lats_res = geo_transformer.transform(mxs, mys)
            if isinstance(lons_res, np.ndarray):
                lons = lons_res.tolist()
                lats = lats_res.tolist()
            elif isinstance(lons_res, (list, tuple)):
                lons = list(lons_res)
                lats = list(lats_res)
            else:
                lons = [lons_res]
                lats = [lats_res]
            mx_c, my_c = transform * (cx_pixel, cy_pixel)
            c_lon_res, c_lat_res = geo_transformer.transform(mx_c, my_c)
            c_lon = float(c_lon_res)
            c_lat = float(c_lat_res)

        poly_geo_coords = [[lon, lat] for lon, lat in zip(lons, lats)]

        try:
            area_m2 = abs(geod.polygon_area_perimeter(lons, lats)[0])
        except Exception:
            px_area = cv2.contourArea(contour)
            px_width_deg = abs(transform[0])
            px_height_deg = abs(transform[4])
            px_width_m = px_width_deg * 111320.0 * np.cos(np.radians(c_lat))
            px_height_m = px_height_deg * 110540.0
            area_m2 = px_area * px_width_m * px_height_m
    else:
        poly_geo_coords = poly_pixel_coords
        c_lon, c_lat = cx_pixel, cy_pixel
        # No georeferencing: assume standard 10cm pixel size (0.01 m2 per pixel)
        area_m2 = cv2.contourArea(contour) * 0.01

    # RFC 7946 recommends counter-clockwise exterior rings. Normalizing here
    # also prevents clients from displaying an exterior as an apparent hole.
    signed_twice_area = sum(
        poly_geo_coords[i][0] * poly_geo_coords[i + 1][1]
        - poly_geo_coords[i + 1][0] * poly_geo_coords[i][1]
        for i in range(len(poly_geo_coords) - 1)
    )
    if signed_twice_area < 0:
        poly_geo_coords = list(reversed(poly_geo_coords))

    # Filter by area
    if area_m2 < min_area_m2 or area_m2 > max_area_m2:
        return

    # Filter by circularity (compactness)
    px_area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4.0 * np.pi * px_area) / (perimeter ** 2) if perimeter > 0 else 0.0
    if circularity < circularity_threshold:
        return

    # Filter by VARI (Greenness)
    x_val, y_val, w_val, h_val_rect = cv2.boundingRect(contour)
    if w_val <= 0 or h_val_rect <= 0:
        return
    # Clamp ROI to image
    x_val = int(max(0, x_val))
    y_val = int(max(0, y_val))
    x2 = int(min(img_w, x_val + w_val))
    y2 = int(min(img_h, y_val + h_val_rect))
    w_val = x2 - x_val
    h_val_rect = y2 - y_val
    if w_val <= 0 or h_val_rect <= 0:
        return

    cnt_mask = np.zeros((h_val_rect, w_val), dtype=np.uint8)
    offset_contour = contour.copy()
    offset_contour[:, 0, 0] = offset_contour[:, 0, 0] - x_val
    offset_contour[:, 0, 1] = offset_contour[:, 0, 1] - y_val
    cv2.drawContours(cnt_mask, [offset_contour], -1, 255, -1)

    crop_vari = vari_img[y_val:y2, x_val:x2]
    if crop_vari.shape[:2] != cnt_mask.shape[:2]:
        return
    mean_vari = float(cv2.mean(crop_vari, mask=cnt_mask)[0])

    # RGB foliar metrics & physically grounded shadow / water / roof rejection
    crop_rgb = img_rgb[y_val:y2, x_val:x2]
    if crop_rgb.shape[:2] != cnt_mask.shape[:2]:
        return
    mean_rgb = cv2.mean(crop_rgb, mask=cnt_mask)[:3]
    r_m, g_m, b_m = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
    brightness = (r_m + g_m + b_m) / 3.0
    exg_m = 2.0 * g_m - r_m - b_m
    gli_m = (2.0 * g_m - r_m - b_m) / (2.0 * g_m + r_m + b_m + 1e-5)

    is_crown_gate = bool(
        reject_water
        or geometry_quality
        in (
            "sam3_native",
            "deepforest_watershed",
            "watershed_forest_crown",
            "watershed_dense_review",
            "watershed_candidate",
        )
    )
    if is_crown_gate:
        # Strict Pure-Canopy Gate on ALL Individual & Watershed Crowns
        # (supports dark-green high-chlorophyll and Rayleigh-shaded crowns down to brightness 7.5)
        gli_gate_min = (
            max(0.022, 0.032 - 0.00022 * float(haze_offset))
            if haze_offset > 0.0
            else 0.032
        )
        if (
            g_m <= r_m
            or gli_m < gli_gate_min
            or (exg_m < 3.5 and gli_m < 0.055)
            or brightness < 7.5
            or brightness > 190.0
        ):
            return
        # Reject true open water (blue significantly exceeds green & red, low ExG)
        if (
            b_m > g_m + 4.0
            and b_m > r_m + 4.0
            and exg_m < 15.0
            and brightness < 95.0
        ):
            return
    elif vari_threshold > -0.5:
        if mean_vari < vari_threshold and gli_m < 0.03 and exg_m < 4.0:
            return

    from canopy_v2 import (
        compute_crown_confidence,
        compute_polygon_physical_scores,
    )

    eff_gq = geometry_quality or "sam3_native"
    conf_pct, confidence_tier, confidence_label = compute_crown_confidence(
        crop_rgb=crop_rgb,
        cnt_mask=cnt_mask,
        circularity=circularity,
        detector_score=(
            detector_score if detector_score is not None else 0.75
        ),
        geometry_quality=eff_gq,
        haze_offset=haze_offset,
    )
    phys_scores = compute_polygon_physical_scores(
        crop_rgb=crop_rgb,
        cnt_mask=cnt_mask,
        haze_offset=haze_offset,
    )
    default_source_model = (
        "sam3"
        if eff_gq == "sam3_native"
        else (
            "deepforest_2.1"
            if eff_gq == "deepforest_watershed"
            else "compact_watershed"
        )
    )

    # Estimate crown diameter (CD)
    crown_diameter_m = 2.0 * math.sqrt(area_m2 / math.pi)
    crown_scaled_h = float(
        np.clip(
            float(default_height_m) * ((max(crown_diameter_m, 1.0) / 6.0) ** 0.55),
            2.0,
            35.0,
        )
    )

    # Estimate height (H)
    estimate_method = "allometric_crown_scaled"
    if height_map is not None:
        try:
            crop_height = height_map[y_val:y2, x_val:x2]
            if crop_height.shape[:2] == cnt_mask.shape[:2] and np.any(cnt_mask > 0):
                h_val = float(np.percentile(crop_height[cnt_mask > 0], 95))
                if math.isnan(h_val) or h_val <= 0:
                    h_val = crown_scaled_h
                else:
                    estimate_method = "chm_p95"
            else:
                h_val = crown_scaled_h
        except Exception:
            h_val = crown_scaled_h
    else:
        h_val = crown_scaled_h

    # Calibrated Visible Foliar Greenness Index (VGI) for RGB, or Sentinel-2 NDVI if available
    ngrdi_m = (g_m - r_m) / (g_m + r_m + 1e-5)
    vgi_val = float(
        np.clip(
            max(
                mean_vari,
                0.85 * ngrdi_m,
                0.55 * mean_vari
                + 1.45 * gli_m
                + 0.08 * min(1.0, max(0.0, exg_m / 25.0)),
            ),
            -1.0,
            1.0,
        )
    )
    ndvi_val = vgi_val
    is_true_ndvi = False
    if ndvi_src is not None and is_georeferenced:
        try:
            # ndvi_src is a tuple of (ndvi_data, ndvi_transform) loaded in-memory
            ndvi_data, ndvi_transform = ndvi_src
            from rasterio.transform import rowcol
            r, c = rowcol(ndvi_transform, c_lon, c_lat)
            if 0 <= r < ndvi_data.shape[0] and 0 <= c < ndvi_data.shape[1]:
                val = float(ndvi_data[r, c])
                if not math.isnan(val):
                    ndvi_val = val
                    is_true_ndvi = True
        except Exception as e:
            print(f"Error sampling in-memory NDVI: {e}")

    # Allometric growth model DBH and Volume
    # Equation 7: DBH = 0.3132 * H + 0.3751 * CD + 11.26
    estimated_dbh_cm = 0.3132 * h_val + 0.3751 * crown_diameter_m + 11.26
    # Equation 2: Volume = 0.00005741 * DBH^1.77035219 * H^1.12503045
    estimated_volume_m3 = 0.00005741 * (estimated_dbh_cm ** 1.77035219) * (h_val ** 1.12503045)

    # Add to feature list
    features.append({
        "type": "Feature",
        "id": tree_id,
        "geometry": {
            "type": "Polygon",
            "coordinates": [poly_geo_coords]
        },
        "properties": {
            "tree_id": tree_id,
            "area_m2": round(area_m2, 2),
            "crown_diameter_m": round(crown_diameter_m, 2),
            "estimated_height_m": round(h_val, 2),
            "estimated_dbh_cm": round(estimated_dbh_cm, 2),
            "estimated_volume_m3": round(estimated_volume_m3, 3),
            "ndvi_health": round(ndvi_val, 3),
            "raw_vari": round(float(mean_vari), 3),
            "foliar_gli": round(float(gli_m), 3),
            "health_index_type": "sentinel2_ndvi" if is_true_ndvi else "vgi_rgb",
            "circularity": round(circularity, 3),
            "centroid_lon": float(c_lon),
            "centroid_lat": float(c_lat),
            "pixel_cx": float(cx_pixel),
            "pixel_cy": float(cy_pixel),
            "pixel_bounds": [int(x_val), int(y_val), int(w_val), int(h_val_rect)],
            "mean_rgb": [int(r_m), int(g_m), int(b_m)],
            "estimate_method": estimate_method,
            "detector_score": (
                round(float(detector_score), 4)
                if detector_score is not None
                else None
            ),
            "confidence": round(float(conf_pct / 100.0), 4),
            "confidence_pct": round(float(conf_pct), 1),
            "confidence_tier": confidence_tier,
            "confidence_label": confidence_label,
            "foliage_prob": phys_scores["foliage_prob"],
            "relief_3d_score": phys_scores["relief_3d_score"],
            "artificial_cyan_score": phys_scores["artificial_cyan_score"],
            "source_model": default_source_model,
            "geometry_quality": geometry_quality or "contour",
        }
    })


if __name__ == "__main__":
    import uvicorn
    # Run server locally on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
