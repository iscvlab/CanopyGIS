@echo off
title Canopy GIS - Cloudflare Share
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File .\share_cloudflare.ps1
pause
