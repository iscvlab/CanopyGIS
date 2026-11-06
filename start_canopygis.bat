@echo off
title CanopyGIS - Hybrid Tree Canopy & Centroid Extraction (localhost:8000)
cd /d "%~dp0\src"
echo Starting CanopyGIS Server on http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
