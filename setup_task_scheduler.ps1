# ============================================================
# setup_task_scheduler.ps1
# Run this ONCE as Administrator to register the daily task.
# Right-click this file → "Run with PowerShell as Administrator"
# ============================================================

# ── CONFIG — edit these to match your setup ─────────────────
$ScriptsDir  = "C:\MaverickPICKS"          # folder with your .py files
$BatchFile   = "$ScriptsDir\run_maverick_daily.bat"
$TaskName    = "MaverickPICKS Daily Runner"

# Run at 3:45 PM IST every weekday
# Task Scheduler uses LOCAL time — adjust if your PC is not in IST
# IST = UTC+5:30, so 3:45 PM IST = 10:15 AM UTC
# If your PC clock is set to IST, use 15:45 below (no change needed)
$RunTime     = "15:45"
# ────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "============================================================"
Write-Host "  MaverickPICKS Task Scheduler Setup"
Write-Host "============================================================"
Write-Host ""

# Check batch file exists
if (-not (Test-Path $BatchFile)) {
    Write-Host "[ERROR] Batch file not found: $BatchFile"
    Write-Host "        Make sure run_maverick_daily.bat is in $ScriptsDir"
    Read-Host "Press Enter to exit"
    exit 1
}

# Remove existing task with same name if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  Removing existing task: $TaskName"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Action: run the batch file
$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatchFile`"" `
    -WorkingDirectory $ScriptsDir

# Trigger: weekdays at 3:45 PM IST
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At $RunTime

# Settings: run even if on battery, wake PC if sleeping (optional)
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RunOnlyIfNetworkAvailable `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew

# Principal: run as current user
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType InteractiveToken `
    -RunLevel Highest

# Register the task
Register-ScheduledTask `
    -TaskName    $TaskName `
    -Action      $Action `
    -Trigger     $Trigger `
    -Settings    $Settings `
    -Principal   $Principal `
    -Description "Runs MaverickPICKS pattern scanner and tracker after NSE market close" `
    -Force | Out-Null

Write-Host "  Task registered successfully!"
Write-Host ""
Write-Host "  Task name  : $TaskName"
Write-Host "  Runs at    : $RunTime IST, Monday to Friday"
Write-Host "  Batch file : $BatchFile"
Write-Host "  Log folder : $ScriptsDir\logs\"
Write-Host ""
Write-Host "  To verify: open Task Scheduler > Task Scheduler Library"
Write-Host "             look for '$TaskName'"
Write-Host ""
Write-Host "  To run manually right now (for testing):"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""

# Offer to run immediately for testing
$test = Read-Host "  Run the task RIGHT NOW for testing? (y/n)"
if ($test -eq "y" -or $test -eq "Y") {
    Write-Host ""
    Write-Host "  Starting task..."
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 3
    $state = (Get-ScheduledTask -TaskName $TaskName).State
    Write-Host "  Task state: $state"
    Write-Host "  Check $ScriptsDir\logs\ for output"
}

Write-Host ""
Write-Host "  Setup complete."
Read-Host "Press Enter to exit"
