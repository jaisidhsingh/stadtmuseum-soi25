@echo off
TITLE Stadtmuseum Exhibit - SFTP Mode
COLOR 0A

echo ======================================================
echo   STADTMUSEUM EXHIBIT - STARTING (SFTP MODE)
echo ======================================================

:: Change to the script's directory
cd /d "%~dp0"

echo [1/2] Starting Backend (api.py)...
cd ..\backend
start "Exhibit Backend" cmd /k "python api.py"

echo Waiting for backend to initialize...
timeout /t 8 /nobreak > nul

echo [2/2] Starting Frontend (Vite)...
cd ..\frontend
start "Exhibit Frontend" cmd /k "npm run dev"

echo.
echo ======================================================
echo   Exhibit is now running.
echo   - Backend: http://localhost:8000
echo   - Frontend: Check the console for Vite URL
echo ======================================================
echo.
pause
