@echo off
rem Double-click: the live training dashboard at http://localhost:8050 (see scripts\dashboard.py).
rem Add --lan to open it from a phone on the same Wi-Fi. Close this window to stop the page.
title bg-tavern-agent dashboard
"%~dp0..\.venv\Scripts\python.exe" "%~dp0dashboard.py" %*
pause
