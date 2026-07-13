@echo off
REM install.bat
REM Windows setup for the Liquids DAQ system.
REM Run once from the repo root: install.bat

cd /d "%~dp0"

echo === Liquids DAQ - Install ===

REM --- Python check ---
REM Prefer 'python' if it resolves; some installs (esp. the official
REM python.org installer) only register the 'py' launcher on PATH instead.
set "PYEXE="
python --version >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE (
    py -3 --version >nul 2>&1 && set "PYEXE=py -3"
)
if not defined PYEXE (
    echo ERROR: no Python interpreter found ^(tried 'python' and 'py -3'^).
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo and make sure to check "Add python.exe to PATH" during setup.
    pause
    exit /b 1
)
echo [INFO] Using interpreter: %PYEXE%

REM --- Virtual environment ---
if not exist ".venv\" (
    echo Creating virtual environment...
    %PYEXE% -m venv .venv
)

call .venv\Scripts\activate.bat

REM --- Python packages ---
echo Installing Python dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies. See output above.
    pause
    exit /b 1
)

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

set "INSTALLER_URL=https://files.labjack.com/installers/LJM/Windows/x86_64/beta/LabJackBasic_2025-02-12.exe"
set "INSTALLER_EXE=LabJackM_Installer.exe"

if /i "%install_ljm%"=="y" goto do_install_ljm
goto driver_fallback

:do_install_ljm
echo Downloading LJM driver installer...
curl -fL -o "%TEMP%\%INSTALLER_EXE%" "%INSTALLER_URL%"

if errorlevel 1 (
    echo [WARNING] Download failed ^(bad URL, network issue, or HTTP error^).
    if exist "%TEMP%\%INSTALLER_EXE%" del "%TEMP%\%INSTALLER_EXE%"
    goto driver_fallback
)

if not exist "%TEMP%\%INSTALLER_EXE%" (
    echo [WARNING] Failed to download the driver installer automatically.
    goto driver_fallback
)

echo Verifying installer signature...
powershell -NoProfile -Command ^
    "$sig = Get-AuthenticodeSignature -FilePath '%TEMP%\%INSTALLER_EXE%'; " ^
    "if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'LabJack') { exit 1 } else { exit 0 }"

if errorlevel 1 (
    echo [WARNING] Installer signature is missing, invalid, or not from LabJack.
    echo For safety, this file will NOT be run automatically.
    del "%TEMP%\%INSTALLER_EXE%"
    goto driver_fallback
)

echo Signature verified - signed by LabJack.
echo.
echo Launching LJM Installer...
echo Please approve the Windows Administrator prompt to complete installation.

start /wait "" "%TEMP%\%INSTALLER_EXE%"

del "%TEMP%\%INSTALLER_EXE%"
echo Driver installer finished.
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
