@echo off
REM install.bat
REM Windows setup for the Liquids DAQ system.
REM Run once from the repo root: install.bat

echo === Liquids DAQ - Install ===

REM --- Python check ---
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found. Install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

REM --- Virtual environment ---
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

REM --- Python packages ---
echo Installing Python dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q

REM --- LabJack LJM driver check & optional install ---
echo.
echo === LabJack LJM Native Driver Check ===

set "DRIVER_FOUND=0"
if exist "C:\Windows\System32\LabJackM.dll" set "DRIVER_FOUND=1"
if exist "C:\Windows\SysWOW64\LabJackM.dll" set "DRIVER_FOUND=1"

if "%DRIVER_FOUND%"=="1" (
    echo [INFO] LabJack LJM driver already detected on this system.
    goto create_data_dir
)

echo [INFO] LabJack LJM driver was not found on this system.
set "install_ljm="
set /p install_ljm="Would you like to download and install the LJM driver now? (y/n): "

if /i "%install_ljm%"=="y" (
    echo Downloading LJM driver installer...
    set "INSTALLER_URL=https://files.labjack.com/installers/LJM/Windows/x86_64/beta/LabJackBasic_2025-02-12.exe"
    set "INSTALLER_EXE=LabJackM_Installer.exe"
    
    curl -L -o "%TEMP%\%INSTALLER_EXE%" "%INSTALLER_URL%"
    
    if exist "%TEMP%\%INSTALLER_EXE%" (
        echo.
        echo Launching LJM Installer...
        echo Please approve the Windows Administrator prompt to complete installation.
        
        REM Run the installer without silent mode so the user sees the installation progress and prompts.
        start /wait "" "%TEMP%\%INSTALLER_EXE%"
        
        del "%TEMP%\%INSTALLER_EXE%"
        echo Driver installer finished.
    ) else (
        echo [WARNING] Failed to download the driver installer automatically.
        goto driver_fallback
    )
) else (
    goto driver_fallback
)

goto create_data_dir

:driver_fallback
echo.
echo === LabJack LJM Driver Manual Setup ===
echo You chose not to install the driver, or the automatic installation failed.
echo To run the hardware, please download and install the driver manually from:
echo   https://labjack.com/pages/support?doc=/software-driver/installer-downloads/ljm-software-installers-t4-t7-t8-digit/
echo or
echo   https://support.labjack.com/docs/windows-setup-basic-driver-only
echo.
echo (Note: The DAQ will fall back to mock mode if the hardware driver is missing.)

:create_data_dir
REM --- Data directory ---
if not exist "data\" mkdir data

echo.
echo === Install complete ===
echo Run the DAQ with:  run.bat
echo Run tests with:    run.bat --test
pause