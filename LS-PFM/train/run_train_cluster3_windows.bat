@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"
set "PYTHON=python"
if exist "%ROOT_DIR%\..\venv\Scripts\python.exe" set "PYTHON=%ROOT_DIR%\..\venv\Scripts\python.exe"
if exist "%ROOT_DIR%\..\..\venv\Scripts\python.exe" set "PYTHON=%ROOT_DIR%\..\..\venv\Scripts\python.exe"

if not defined EPOCHS set EPOCHS=6
if not defined BATCH_SIZE set BATCH_SIZE=128
if not defined DEVICE set DEVICE=cpu
if not defined MAX_TRAIN_WINDOWS set MAX_TRAIN_WINDOWS=2048

cd /d "%ROOT_DIR%"
"%PYTHON%" "%ROOT_DIR%\train\train_cluster3.py" ^
  --epochs %EPOCHS% ^
  --batch_size %BATCH_SIZE% ^
  --device %DEVICE% ^
  --max_train_windows %MAX_TRAIN_WINDOWS% ^
  %*

if errorlevel 1 (
  echo Cluster3 training failed.
  if not defined NO_PAUSE pause
  exit /b 1
)

if not defined NO_PAUSE pause
