@echo off
echo ========================================
echo AI Product Studio - Build Executable
echo ========================================
echo.
echo Cleaning previous builds...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"
echo.
echo Building executable...
echo This may take 5-10 minutes...
echo.
pyinstaller --clean build_exe.spec
echo.
if exist "dist\AI Product Studio\AI Product Studio.exe" (
    echo ========================================
    echo BUILD SUCCESSFUL!
    echo ========================================
    echo.
    echo Executable location:
    echo dist\AI Product Studio\AI Product Studio.exe
    echo.
    echo You can now share the entire "AI Product Studio" folder
    echo with others. It contains all necessary files.
    echo.
    pause
) else (
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    echo.
    echo Please check the error messages above.
    echo.
    pause
)
