@echo off
rem Double-click to start or resume the 3070 training run (see start_3070.ps1).
where pwsh >nul 2>nul && (
  pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_3070.ps1" %*
) || (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_3070.ps1" %*
)
pause
