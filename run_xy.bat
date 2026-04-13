@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ================================================
echo  KOSPI 200 X/Y Generator (Windows)
echo ================================================
echo.

python --version > /dev/null 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo         Install from https://www.python.org
    pause
    exit /b 1
)
echo [OK] Python found

echo.
echo [1/3] Installing packages...
pip install yfinance openpyxl pandas numpy requests -q
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed
    pause
    exit /b 1
)
echo [OK] Packages ready

echo.
echo [2/3] Downloading OHLCV data (10~20 min)...
python 01_download_ohlcv.py
if %errorlevel% neq 0 (
    echo [ERROR] Download failed
    pause
    exit /b 1
)

echo.
echo [3/3] Generating X/Y pairs...
python 02_generate_xy.py
if %errorlevel% neq 0 (
    echo [ERROR] XY generation failed
    pause
    exit /b 1
)

echo.
echo ================================================
echo  Done! xy_pairs\pairs.jsonl created.
echo ================================================
echo.

set /p OPEN=Open viewer in browser? (y/n): 
if /i "%OPEN%"=="y" (
    python 05_xy_viewer.py
)

pause
