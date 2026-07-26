@echo off
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"
set "PYTHON=python"
if exist "%ROOT_DIR%\..\venv\Scripts\python.exe" set "PYTHON=%ROOT_DIR%\..\venv\Scripts\python.exe"
if exist "%ROOT_DIR%\..\..\venv\Scripts\python.exe" set "PYTHON=%ROOT_DIR%\..\..\venv\Scripts\python.exe"

cd /d "%ROOT_DIR%"
"%PYTHON%" "%ROOT_DIR%\test\test_fixed_best.py" --preset all %*

if errorlevel 1 (
  echo Fixed-best tests failed.
  if not defined NO_PAUSE pause
  exit /b 1
)

if not defined NO_PAUSE pause
