@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project environment not found.
    echo Run "uv sync" in this folder first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run app.py
pause
