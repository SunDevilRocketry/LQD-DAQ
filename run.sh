#!/usr/bin/env bash
# run.sh
# Start the Liquids DAQ service.
# Usage:
#   bash run.sh              - start normally (real hardware if available)
#   bash run.sh --mock       - force mock hardware
#   bash run.sh --no-server  - engine only, no HTTP API (headless / debug)
#   bash run.sh --test       - run the test suite (mock hardware only)
#   bash run.sh --test-live  - run the real hardware test suite (LabJack T7 must be attached)

set -e

# Activate venv if the activation script exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Resolve a working interpreter. If the venv is active, 'python' inside it
# takes priority automatically since it's first on PATH after activation.
PYEXE=""
if command -v python3 &>/dev/null; then
    PYEXE="python3"
elif command -v python &>/dev/null; then
    PYEXE="python"
else
    echo "ERROR: no Python interpreter found (tried 'python3' and 'python')."
    echo "Run 'bash install.sh' first, or install Python 3.10+."
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
