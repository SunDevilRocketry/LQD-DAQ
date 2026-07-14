@echo off
REM run.bat
REM Start the Liquids DAQ service.
REM Usage:
REM   run.bat              - start normally (real hardware if available)
REM   run.bat --mock       - force mock hardware
REM   run.bat --no-server  - engine only, no HTTP API
REM   run.bat --test       - run the test suite (mock hardware only)
REM   run.bat --test-live  - run the real-hardware test suite (LabJack T7 must be attached)

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM --- Resolve a working Python interpreter ---
REM Prefer the 'py' launcher
REM If a venv is active, 'python' inside it takes priority automatically
REM since it's first on PATH after activation.
set "PYEXE="
where python >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE (
    where py >nul 2>&1 && set "PYEXE=py -3"
)
if not defined PYEXE (
    echo ERROR: no Python interpreter found ^(tried 'python' and 'py'^).
    echo Install Python 3.10+ from https://www.python.org/downloads/ and re-run.
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
