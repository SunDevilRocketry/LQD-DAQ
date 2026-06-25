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

if "%1"=="--test" goto run_tests

echo === Liquids DAQ ===
python -m daq %*
exit /b %errorlevel%

:run_tests
echo === Running test suite ===
python -m pytest tests/ -v
exit /b %errorlevel%