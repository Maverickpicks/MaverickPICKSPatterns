@echo off
:: ============================================================
:: MaverickPICKS Daily Runner
:: Runs after market close (3:45 PM IST on weekdays)
:: Place this file in the same folder as your Python scripts
:: ============================================================

:: ── CONFIG — edit these two lines ───────────────────────────
set PYTHON=python
set SCRIPTS_DIR=C:\MaverickPICKS

:: ── Change to scripts directory ─────────────────────────────
cd /d "%SCRIPTS_DIR%"

:: ── Log file for this run ────────────────────────────────────
set LOGFILE=%SCRIPTS_DIR%\logs\maverick_%date:~-4,4%%date:~-7,2%%date:~0,2%.txt
if not exist "%SCRIPTS_DIR%\logs" mkdir "%SCRIPTS_DIR%\logs"

echo. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo MaverickPICKS Run — %date% %time% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

:: ── Skip weekends ────────────────────────────────────────────
:: PowerShell check: 0=Sunday, 6=Saturday
for /f %%d in ('powershell -Command "(Get-Date).DayOfWeek.value__"') do set DOW=%%d
if "%DOW%"=="0" (
    echo Skipping — Sunday >> "%LOGFILE%"
    echo Skipping — Sunday
    exit /b 0
)
if "%DOW%"=="6" (
    echo Skipping — Saturday >> "%LOGFILE%"
    echo Skipping — Saturday
    exit /b 0
)

echo Weekday confirmed. Starting scan... >> "%LOGFILE%"
echo.
echo [1/2] Running pattern scanner...
echo [1/2] Running pattern scanner... >> "%LOGFILE%"

%PYTHON% pattern_detector_v2.py ^
    --csv_file NIFTY500_MASTER.csv ^
    --lookback 90 ^
    --min_score 50 ^
    --workers 4 ^
    --out_csv todays_picks.csv >> "%LOGFILE%" 2>&1

if errorlevel 1 (
    echo [ERROR] Scanner failed. Check log: %LOGFILE%
    echo [ERROR] Scanner failed >> "%LOGFILE%"
    exit /b 1
)

echo [1/2] Scanner complete. >> "%LOGFILE%"
echo [1/2] Scanner complete.
echo.

echo [2/2] Running tracker (import + check)...
echo [2/2] Running tracker... >> "%LOGFILE%"

%PYTHON% pattern_tracker.py ^
    --import_csv todays_picks.csv ^
    --check >> "%LOGFILE%" 2>&1

if errorlevel 1 (
    echo [ERROR] Tracker failed. Check log: %LOGFILE%
    echo [ERROR] Tracker failed >> "%LOGFILE%"
    exit /b 1
)

echo [2/2] Tracker complete. >> "%LOGFILE%"
echo [2/2] Tracker complete.
echo.
echo Done. Log saved to: %LOGFILE%
echo Done >> "%LOGFILE%"
