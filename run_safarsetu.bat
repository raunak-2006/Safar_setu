@echo off
setlocal
cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:5000/"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$url = '%APP_URL%'; try { $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2; if ($response.StatusCode -eq 200) { exit 0 } } catch { exit 1 }; exit 1"
if errorlevel 1 (
  if not exist ".venv\Scripts\python.exe" (
    echo SafarSetu's Python environment was not found.
    echo Follow the setup instructions in README.md first.
    pause
    exit /b 1
  )

  start "SafarSetu server" cmd /k ""%~dp0.venv\Scripts\python.exe" "%~dp0app.py""
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$url = '%APP_URL%'; for ($attempt = 0; $attempt -lt 60; $attempt++) { try { $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2; if ($response.StatusCode -eq 200) { Start-Process $url; exit 0 } } catch { }; Start-Sleep -Seconds 1 }; Write-Error 'SafarSetu did not start. Check the server window for errors.'; exit 1"
if errorlevel 1 pause
