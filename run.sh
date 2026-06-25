#!/usr/bin/env bash
# run.sh
# Start the Liquids DAQ service.
# Usage:
#   bash run.sh              - start normally (real hardware if available)
#   bash run.sh --mock       - force mock hardware
#   bash run.sh --no-server  - engine only, no HTTP API (headless / debug)
#   bash run.sh --test       - run the test suite instead

set -e

# Activate venv if the activation script exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

if [ "$1" = "--test" ]; then
    echo "=== Running test suite ==="
    python -m pytest tests/ -v
    exit $?
fi

echo "=== Liquids DAQ ==="
python -m daq "$@"