#!/usr/bin/env powershell
<#
    Sentinel Shield Dashboard Startup Script
    Starts the Streamlit dashboard on Windows with PowerShell
#>

# Clear screen and show banner
Clear-Host
Write-Host @"
============================================================
    🛡️  SENTINEL SHIELD - EXECUTIVE DASHBOARD LAUNCHER
============================================================
"@ -ForegroundColor Cyan

# Check if Streamlit is installed
Write-Host "[*] Checking Streamlit installation..."
try {
    $streamlit_version = & python -m streamlit --version 2>&1
    Write-Host "[✓] Streamlit found: $streamlit_version" -ForegroundColor Green
} catch {
    Write-Host "[✗] Streamlit is not installed." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install dependencies:" -ForegroundColor Yellow
    Write-Host "  pip install -r dashboard/requirements.txt"
    Write-Host ""
    exit 1
}

# Get project root
$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$project_root = Split-Path -Parent $script_dir
Set-Location $project_root

Write-Host "[✓] Project root: $project_root" -ForegroundColor Green
Write-Host ""

# Check for metrics file
if (-not (Test-Path ".sentinel_metrics.json")) {
    Write-Host "[⚠️] WARNING: .sentinel_metrics.json not found" -ForegroundColor Yellow
    Write-Host "This file is created when the monitor runs." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Make sure to start the monitor in another terminal:" -ForegroundColor Yellow
    Write-Host "  python scripts/inference/9_gmail_live_monitor.py"
    Write-Host ""
}

# Check for log file
if (-not (Test-Path ".sentinel_shield.log")) {
    Write-Host "[⚠️] WARNING: .sentinel_shield.log not found" -ForegroundColor Yellow
    Write-Host "This file is created when the monitor runs." -ForegroundColor Yellow
    Write-Host ""
}

# Check for threat registry
if (-not (Test-Path ".sentinel_threat_registry.json")) {
    Write-Host "[ℹ️] INFO: .sentinel_threat_registry.json not created yet" -ForegroundColor Cyan
    Write-Host "This is created when the first threat is detected." -ForegroundColor Cyan
    Write-Host ""
}

# Start Streamlit
Write-Host "[*] Starting Streamlit dashboard..." -ForegroundColor Green
Write-Host ""
Write-Host "Dashboard URL: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

& python -m streamlit run dashboard/app.py

Write-Host ""
Write-Host "Dashboard closed." -ForegroundColor Gray
