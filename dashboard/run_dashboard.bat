@echo off
REM Sentinel Shield Dashboard Startup Script
REM Starts the Streamlit dashboard on Windows

setlocal enabledelayedexpansion

REM Colors for output
echo.
echo ============================================================
echo     SENTINEL SHIELD - EXECUTIVE DASHBOARD LAUNCHER
echo ============================================================
echo.

REM Check if Streamlit is installed
python -m streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Streamlit is not installed.
    echo.
    echo Please install dependencies:
    echo   pip install -r dashboard/requirements.txt
    echo.
    pause
    exit /b 1
)

REM Get current directory
cd /d "%~dp0\.."

echo [✓] Project root: %cd%
echo.

REM Check if metrics file exists
if not exist ".sentinel_metrics.json" (
    echo [⚠️] WARNING: .sentinel_metrics.json not found
    echo This file is created when the monitor runs.
    echo.
    echo Make sure to start the monitor:
    echo   python scripts/inference/9_gmail_live_monitor.py
    echo.
)

REM Start Streamlit app
echo [*] Starting Streamlit dashboard...
echo.
echo Access the dashboard at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.

python -m streamlit run dashboard/app.py

pause
