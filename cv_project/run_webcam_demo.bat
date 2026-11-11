@echo off
chcp 65001 > nul
cd /d "C:\Users\nurbo\Projects\spandiyar-kobeev-journey"
.\.venv\Scripts\python.exe -u cv_project\scripts\webcam_demo.py
pause
