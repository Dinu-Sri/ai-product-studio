# Quick Start Script for Windows PowerShell

Write-Host "========================================"
Write-Host "Product Image Generator - Quick Start"
Write-Host "========================================" 
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    Write-Host ""
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& "venv\Scripts\Activate.ps1"

# Check if requirements are installed
$pyqt6Installed = pip show PyQt6 2>$null
if (-not $pyqt6Installed) {
    Write-Host ""
    Write-Host "Installing dependencies... This may take a few minutes."
    Write-Host ""
    pip install -r requirements.txt
    Write-Host ""
    Write-Host "Installation complete!"
    Write-Host ""
}

# Run the application
Write-Host "Starting Product Image Generator..."
Write-Host ""
python main.py

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "An error occurred. Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
