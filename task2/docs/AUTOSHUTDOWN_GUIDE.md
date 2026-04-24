# Automatic Shutdown Guide (Windows)

For permanent museum exhibits, it is best practice to have the PC automatically shut down at closing time to save power and clear session data.

## Option 1: Using Windows Task Scheduler (Recommended)

This is the most reliable method as it runs every day automatically.

1.  Press **Windows Key**, type **Task Scheduler**, and press **Enter**.
2.  Click **Create Basic Task...** on the right sidebar.
3.  **Name**: `Daily Shutdown`
4.  **Trigger**: `Daily`
5.  **Time**: Set the time (e.g., `18:00:00` for 6 PM).
6.  **Action**: `Start a program`
7.  **Program/script**: `shutdown`
8.  **Add arguments**: `/s /f /t 60`
    *   `/s`: Shutdown.
    *   `/f`: Force close running applications (like the Exhibit backend/frontend).
    *   `/t 60`: Wait 60 seconds before shutting down (gives a warning).
9.  Click **Finish**.

---

## Option 2: Using a Batch Script (Manual Timer)

If you just want to trigger a shutdown timer manually after starting the exhibit, you can create a `.bat` file:

1.  Create a file named `shutdown_timer.bat`.
2.  Paste the following:
    ```batch
    @echo off
    set /p hours="Enter hours until shutdown: "
    set /a seconds=%hours%*3600
    shutdown /s /f /t %seconds%
    echo Shutdown scheduled in %hours% hours.
    pause
    ```

---

## Option 3: Proximity / Kiosk Mode (Advanced)

If you are using a professional Kiosk management software (like **SiteKiosk** or **FrontFace**), use their built-in scheduler instead, as they can also manage the "Wake-on-LAN" or BIOS Power-On timers to turn the PC back on in the morning.

### Note on "Wake on Power"
To make the exhibit fully automatic, go into your PC's **BIOS/UEFI settings** and look for:
- **Restore on AC Power Loss**: Set to `Power On`.
- This ensures that if the museum turns on the main power switch for the room, the PC starts up automatically.

### Note on "Startup"
To make the Exhibit start as soon as the PC turns on:
1. Press `Win + R`, type `shell:startup`, and press **Enter**.
2. Copy a **Shortcut** of `start_exhibit_sftp.bat` (or `start_exhibit_tunnel.bat`) into this folder.
