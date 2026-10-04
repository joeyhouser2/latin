@echo off
rem Desktop launcher for the Latin Library web app.
rem
rem Double-click this (or the Desktop shortcut made by
rem scripts\install_shortcut.ps1) to start the server and open the browser.
rem Closing this window stops the web app *and* any job it is running (the job
rem shares this console). Nothing is lost -- every job commits as it goes and
rem resumes from the last chunk when you requeue it from the Jobs tab.

setlocal
cd /d "%~dp0"

rem Prefer the project venv: it has the CUDA build of torch the jobs need, and
rem whichever interpreter runs the server is the one that runs the jobs.
set "PY=%~dp0latinvenv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

title Latin Library
echo Starting the Latin Library at http://127.0.0.1:8000 ...
echo (close this window to stop the server)
echo.

"%PY%" scripts\serve.py %*

if errorlevel 1 (
  echo.
  echo The server exited with an error. Scroll up for the traceback.
  pause
)
endlocal
