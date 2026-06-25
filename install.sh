#!/usr/bin/env bash
# install.sh
# macOS / Linux setup for the Liquids DAQ system.
# Run once from the repo root: bash install.sh
set -e

echo "=== Liquids DAQ — Install ==="

# --- Python check ---
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ and re-run."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python $PY_VERSION found."

# --- Virtual environment ---
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# --- Python packages ---
echo "Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

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
echo "Run the DAQ with:  bash run.sh"
echo "Run tests with:    bash run.sh --test"