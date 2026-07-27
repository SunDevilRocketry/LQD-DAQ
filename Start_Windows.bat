@echo off
REM Start_Windows.bat
REM First run:  installs everything into a local .venv (visible progress in
REM             this terminal), checks the LabJack driver, then launches.
REM Every run after: sees .venv already exists, skips straight to launch.
REM
REM Usage:
REM   Start_Windows.bat              - start normally (real hardware if available)
REM   Start_Windows.bat --mock       - force mock hardware
REM   Start_Windows.bat --no-server  - engine only, no HTTP API
REM   Start_Windows.bat --test       - run the test suite (mock hardware only)
REM   Start_Windows.bat --test-live  - run the real-hardware test suite (LabJack T7 must be attached)

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "VENV_DIR=.venv"

REM ====================================================================
REM  HEALTH CHECK -- only install if the venv is missing
REM ====================================================================
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    call :install
    if errorlevel 1 (
        pause
        exit /b 1
    )
) else (
    echo [INFO] Environment already set up -- skipping install.
)

call "%VENV_DIR%\Scripts\activate.bat"

REM ====================================================================
REM  LAUNCH
REM ====================================================================
set "PYEXE="
where python >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE (
    where py >nul 2>&1 && set "PYEXE=py -3"
)
if not defined PYEXE (
    echo ERROR: no Python interpreter found ^(tried 'python' and 'py'^).
    echo Install Python 3.10+ from https://www.python.org/downloads/ and re-run.
    pause
    exit /b 1
)

if "%1"=="--test" goto run_tests
if "%1"=="--test-live" goto run_tests_live

echo === Liquids DAQ ===
%PYEXE% -m daq %*
exit /b %errorlevel%

:run_tests
echo === Running test suite (mock hardware, no LabJack required) ===
%PYEXE% -m pytest tests/software tests/test_calculations.py -v
exit /b %errorlevel%

:run_tests_live
echo === Running live-hardware test suite (LabJack T7 must be attached) ===
%PYEXE% -m pytest tests/hardware_live -v
exit /b %errorlevel%

REM ====================================================================
REM  :install -- first-run setup only
REM ====================================================================
:install
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
    exit /b 1
)
echo [INFO] Using interpreter: %PYEXE%

REM --- Virtual environment ---
if not exist "%VENV_DIR%\" (
    echo Creating virtual environment...
    %PYEXE% -m venv "%VENV_DIR%"
)
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment creation failed.
    echo This can happen if your Python install is missing the 'venv' module,
    echo or if an antivirus/IT policy blocked writing to this folder.
    echo Try running this command manually to see the actual error:
    echo     %PYEXE% -m venv %VENV_DIR%
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"

REM --- Python packages ---
echo Installing Python dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies. See output above.
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
echo.
exit /b 0
