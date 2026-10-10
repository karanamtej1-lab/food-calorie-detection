@echo off
REM Creates the virtual environment on first run, installs packages, then starts the app.
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Creating virtual environment with Python 3.11...
    py -3.11 -m venv venv || (echo Python 3.11 not found. Install it from python.org. & pause & exit /b 1)
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt || (echo Package install failed. & pause & exit /b 1)
)

venv\Scripts\python.exe -c "import torch" || (echo PyTorch failed to load. See README Troubleshooting ^(WinError 1114^). & pause & exit /b 1)
venv\Scripts\python.exe app.py
