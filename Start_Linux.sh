#!/usr/bin/env bash
# Start_Linux.sh
# First run:  installs everything into a local .venv, checks the LabJack
#             LJM native driver, then launches.
# Every run after: sees .venv already exists, skips straight to launch.
#
# Usage:
#   bash Start_Linux.sh              - start normally (real hardware if available)
#   bash Start_Linux.sh --mock       - force mock hardware
#   bash Start_Linux.sh --no-server  - engine only, no HTTP API (headless / debug)
#   bash Start_Linux.sh --test       - run the test suite (mock hardware only)
#   bash Start_Linux.sh --test-live  - run the real hardware test suite (LabJack T7 must be attached)
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR=".venv"

install() {
    echo "=== Liquids DAQ - Install ==="

    # --- Python check ---
    # Prefer python3 (the correct convention on macOS/Linux); fall back to
    # bare 'python' for systems where only that name is on PATH.
    PYEXE=""
    if command -v python3 &>/dev/null; then
        PYEXE="python3"
    elif command -v python &>/dev/null; then
        PYEXE="python"
    fi

    if [ -z "$PYEXE" ]; then
        echo "ERROR: no Python interpreter found (tried 'python3' and 'python')."
        echo "Install Python 3.10+ and re-run."
        exit 1
    fi

    PY_VERSION=$("$PYEXE" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Python $PY_VERSION found ($PYEXE)."

    # --- venv module check ---
    if ! "$PYEXE" -c "import venv" &>/dev/null; then
        echo "[ERROR] The 'venv' module is not available for $PYEXE."
        echo "On Debian/Ubuntu, install it with:"
        echo "    sudo apt install python3-venv"
        echo "Then re-run this script."
        exit 1
    fi

    # --- Virtual environment ---
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        "$PYEXE" -m venv "$VENV_DIR"
    fi
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo "[ERROR] Virtual environment creation failed ($VENV_DIR/bin/activate not found)."
        echo "This can happen if your Python install is missing the 'venv' module,"
        echo "or if a permissions/policy issue blocked writing to this folder."
        echo "Try running this command manually to see the actual error:"
        echo "    $PYEXE -m venv $VENV_DIR"
        exit 1
    fi

    source "$VENV_DIR/bin/activate"

    # --- Python packages ---
    echo "Installing Python dependencies..."
    pip install --upgrade pip -q
    if ! pip install -r requirements.txt -q; then
        echo "ERROR: Failed to install Python dependencies."
        exit 1
    fi

    # --- LabJack LJM native driver check ---
    echo ""
    echo "=== LabJack LJM Native Driver Check ==="

    # Detect operating system and architecture
    OS_TYPE=$(uname -s)
    CPU_ARCH=$(uname -m)
    DRIVER_FOUND=0

    if [ "$OS_TYPE" = "Darwin" ]; then
        # macOS standard installation paths
        if [ -f "/usr/local/lib/libLabJackM.dylib" ] || [ -d "/Library/Frameworks/LabJackM.framework" ]; then
            DRIVER_FOUND=1
        fi

        # Intel Vs. Apple Silicon Macs
        if [ "$CPU_ARCH" = "arm64" ]; then
            DOWNLOAD_PAGE="https://support.labjack.com/docs/ljm-software-installer-macos-arm64"
        else
            DOWNLOAD_PAGE="https://support.labjack.com/docs/ljm-software-installer-macos-x64"
        fi

    elif [ "$OS_TYPE" = "Linux" ]; then
        # Linux standard installation paths
        if [ -f "/usr/local/lib/libLabJackM.so" ] || [ -f "/usr/lib/libLabJackM.so" ]; then
            DRIVER_FOUND=1
        fi

        # Standard PC (x64) Vs. ARM
        if [[ "$CPU_ARCH" =~ ^(arm|aarch64) ]]; then
            DOWNLOAD_PAGE="https://support.labjack.com/docs/ljm-software-installer-linux-arm-family"
        else
            DOWNLOAD_PAGE="https://support.labjack.com/docs/ljm-software-installer-linux-x64"
        fi

    else
        # Fallback to the general LJM installer hub page if unable to find OS
        DOWNLOAD_PAGE="https://support.labjack.com/docs/ljm-software-installer-downloads-t4-t7-t8-digit"
    fi

    if [ "$DRIVER_FOUND" -eq 1 ]; then
        echo "[INFO] LabJack LJM library detected on this system."
    else
        echo "[INFO] LabJack LJM library was not found in standard system directories."
        echo "The Python 'labjack-ljm' package requires the native C/C++ LJM library."
        echo ""
        echo "Please download and install LJM for your system ($OS_TYPE / $CPU_ARCH) from:"
        echo "  $DOWNLOAD_PAGE"
        echo ""

        # Prompt user to open browser
        read -p "Would you like to open the download page in your browser now? (y/n): " open_browser
        if [[ "$open_browser" =~ ^[Yy]$ ]]; then
            python3 -m webbrowser "$DOWNLOAD_PAGE"
        fi
        echo ""
        echo "(Note: The DAQ will fall back to mock mode if the library is not installed.)"
    fi

    # --- Data directory ---
    mkdir -p data

    echo ""
    echo "=== Install complete ==="
    echo ""
}

# ========================================================================
#  HEALTH CHECK -- only install if the venv is missing
# ========================================================================
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    install
else
    echo "[INFO] Environment already set up -- skipping install."
    source "$VENV_DIR/bin/activate"
fi

# ========================================================================
#  LAUNCH
# ========================================================================
PYEXE=""
if command -v python3 &>/dev/null; then
    PYEXE="python3"
elif command -v python &>/dev/null; then
    PYEXE="python"
else
    echo "ERROR: no Python interpreter found (tried 'python3' and 'python')."
    echo "Delete the .venv folder and re-run this script."
    exit 1
fi

if [ "$1" = "--test" ]; then
    echo "=== Running test suite (mock hardware, no LabJack required) ==="
    "$PYEXE" -m pytest tests/software tests/test_calculations.py -v
    exit $?
fi

if [ "$1" = "--test-live" ]; then
    echo "=== Running live-hardware test suite (LabJack T7 must be attached) ==="
    "$PYEXE" -m pytest tests/hardware_live -v
    exit $?
fi

echo "=== Liquids DAQ ==="
"$PYEXE" -m daq "$@"
