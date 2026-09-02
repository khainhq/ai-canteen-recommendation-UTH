@echo off
setlocal
rem Use the short path so Python works when the project folder contains Unicode characters.
cd /d "%~sdp0"

set "APP_PYTHON=py"
where py >nul 2>&1
if errorlevel 1 set "APP_PYTHON=python"
where %APP_PYTHON% >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or is not on PATH.
    pause
    exit /b 1
)

rem Use an existing virtual environment when one is already complete.
if exist ".venv\Scripts\python.exe" set "APP_PYTHON=%~sdp0.venv\Scripts\python.exe"

"%APP_PYTHON%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing project dependencies...
    "%APP_PYTHON%" -m pip install --user -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

echo Starting the Streamlit application...
"%APP_PYTHON%" -m streamlit run app.py
endlocal