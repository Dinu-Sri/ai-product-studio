@echo off
echo ========================================
echo Product Image Generator - Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
pip show PyQt6 >nul 2>&1
if errorlevel 1 (
    echo.
    echo Installing dependencies... This may take a few minutes.
    echo.
    pip install -r requirements.txt
    echo.
    echo Installation complete!
    echo.
)

REM Run the application
echo Starting Product Image Generator...
echo.
python main.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause >nul
)
