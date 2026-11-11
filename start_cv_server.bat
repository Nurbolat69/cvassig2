@echo off
chcp 65001 >nul
title Assignment 2 — Computer Vision Web Demo Server
echo.
echo ======================================================================
echo    ASTANA IT UNIVERSITY — COMPUTER VISION ASSIGNMENT 2 DEMO SERVER
echo    SOTA Component: Grad-CAM Explainability + MobileNetV2 Transfer
echo ======================================================================
echo.

cd /d "%~dp0"

:: Smart Python Interpreter Detection
set "PY_EXE="
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0..\spandiyar-kobeev-journey\.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0..\spandiyar-kobeev-journey\.venv\Scripts\python.exe"
) else if exist "%~dp0..\cv_assignment\.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0..\cv_assignment\.venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

echo [INFO] Using Python interpreter: "%PY_EXE%"
echo [INFO] Starting Flask backend on http://localhost:5050 ...
echo [INFO] Interactive Web UI: http://localhost:5050/cv_demo.html
echo.
echo Press Ctrl+C in this console window to terminate the server.
echo.

start "" "http://localhost:5050/cv_demo.html"
"%PY_EXE%" cv_project\scripts\app.py

pause
