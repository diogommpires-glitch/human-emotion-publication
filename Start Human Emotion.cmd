@echo off
setlocal
set "APP_DIR=%~dp0"
set "PYTHONW=%APP_DIR%.venv\Scripts\pythonw.exe"
if not exist "%PYTHONW%" (
    echo The bundled Python runtime was not found.
    echo Expected: "%PYTHONW%"
    echo.
    echo Install uv, then run from this folder:
    echo   uv sync --locked --extra cuda
    echo Use --extra cpu on computers without a supported NVIDIA GPU.
    echo See README.md for setup and development checks.
    pause
    exit /b 1
)
start "Human Emotion" "%PYTHONW%" "%APP_DIR%server.py"
endlocal
