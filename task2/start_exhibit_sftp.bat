@echo off
TITLE Stadtmuseum Exhibit - Startup
COLOR 0A

:: ======================================================
:: CONFIGURATION
:: ======================================================
SET "FRONTEND_URL=http://localhost:8080"
SET "ENV_NAME=soi_simple"

:: Set the base directory to where this script is located
SET "BASE_DIR=%~dp0"

echo ======================================================
echo   STADTMUSEUM EXHIBIT - CLEANING PREVIOUS INSTANCES
echo ======================================================

:: [0/4] Kill existing processes to avoid port conflicts
echo Killing existing Python, Node, and Chrome processes...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM node.exe /T >nul 2>&1
taskkill /F /IM chrome.exe /T >nul 2>&1
timeout /t 2 /nobreak > nul

echo ======================================================
echo   STADTMUSEUM EXHIBIT - STARTING
echo ======================================================

:: [1/4] Activate environment
echo [1/4] Activating environment: %ENV_NAME%...
:: Try to find micromamba or conda activation script
SET "ACTIVATE_CMD=micromamba activate %ENV_NAME%"
where micromamba >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] 'micromamba' not in PATH. Trying 'conda'...
    SET "ACTIVATE_CMD=conda activate %ENV_NAME%"
)

:: [2/4] Starting Backend
echo [2/4] Starting Backend (api.py)...
pushd "%BASE_DIR%backend"
if exist api.py (
    start "Exhibit Backend" cmd /k "call %ACTIVATE_CMD% && python api.py"
) else (
    echo [ERROR] Could not find api.py in %CD%
)
popd

:: [3/4] Starting Frontend
echo [3/4] Starting Frontend (Vite)...
pushd "%BASE_DIR%frontend"
if exist package.json (
    start "Exhibit Frontend" cmd /k "npm run dev"
) else (
    echo [ERROR] Could not find package.json in %CD%
)
popd

:: Wait for services to warm up
echo.
echo Waiting for services to initialize (15 seconds)...
timeout /t 15 /nobreak > nul

:: [4/4] Launching Chrome in Kiosk Mode
echo [4/4] Opening Chrome in Kiosk Mode...
:: --use-fake-ui-for-media-stream: Automatically allows camera/mic access without prompts.
:: --autoplay-policy=no-user-gesture-required: Allows video/audio to play immediately.
start chrome --kiosk "%FRONTEND_URL%" ^
    --edge-touch-filtering ^
    --no-first-run ^
    --disable-features=TranslateUI ^
    --user-data-dir="%TEMP%\chrome_kiosk" ^
    --use-fake-ui-for-media-stream ^
    --autoplay-policy=no-user-gesture-required

echo.
echo ======================================================
echo   Exhibit is now running in Kiosk Mode.
echo   - Press ALT+F4 to close Chrome.
echo   - Close the terminal windows to stop services.
echo ======================================================
echo.
pause

