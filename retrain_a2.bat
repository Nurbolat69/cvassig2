@echo off
chcp 65001 >nul
echo.
echo ============================================================
echo   Assignment 2 - Retrain on CLEANED Dataset
echo   This will take 20-40 min on CPU
echo ============================================================
echo.
echo [STEP 1] Running dataset cleaner...
cd /d "%~dp0"
.venv\Scripts\python.exe cv_project\scripts\clean_dataset.py
echo.
echo [STEP 2] Retraining models on clean data...
.venv\Scripts\python.exe cv_project\scripts\retrain_cleaned.py
echo.
echo DONE! Models saved to cv_project\models\
echo Open: http://localhost:5050/cv_demo.html after starting the server
pause
