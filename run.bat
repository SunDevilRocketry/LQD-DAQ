@echo off
REM run.bat
REM Start the Liquids DAQ service.
REM Usage:
REM   run.bat              - start normally (real hardware if available)
REM   run.bat --mock       - force mock hardware
REM   run.bat --no-server  - engine only, no HTTP API
REM   run.bat --test       - run the test suite

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

echo === Liquids DAQ ===
%PYEXE% -m daq %*
exit /b %errorlevel%

:run_tests
echo === Running test suite ===
%PYEXE% -m pytest tests/ -v
exit /b %errorlevel%
