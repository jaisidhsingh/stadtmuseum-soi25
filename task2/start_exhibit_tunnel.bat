@echo off
TITLE Stadtmuseum Exhibit - TUNNEL MODE
COLOR 0B

echo ======================================================
echo   STADTMUSEUM EXHIBIT - STARTING (TUNNEL MODE)
echo ======================================================

:: Change to the script's directory
cd /d "%~dp0"

echo [1/3] Starting Cloudflare Tunnel...
:: Assumes cloudflared.exe is in the same directory as this script or the parent
if exist ".\cloudflared.exe" (
    start "Cloudflare Tunnel" cmd /k ".\cloudflared.exe tunnel --url http://localhost:8000"
) else (
    echo WARNING: cloudflared.exe not found in current folder. Trying PATH...
    start "Cloudflare Tunnel" cmd /k "cloudflared tunnel --url http://localhost:8000"
)

echo Waiting for tunnel to establish...
timeout /t 5 /nobreak > nul

echo [2/3] Starting Backend (api.py --share-mode local)...
cd backend
start "Exhibit Backend" cmd /k "python api.py --share-mode local"

echo Waiting for backend to initialize...
timeout /t 8 /nobreak > nul

echo [3/3] Starting Frontend (Vite)...
cd ..\frontend
start "Exhibit Frontend" cmd /k "npm run dev"

echo.
echo ======================================================
echo   Exhibit is now running in TUNNEL MODE.
echo   Check the Cloudflare terminal for your public URL!
echo ======================================================
echo.
pause
