"""
tests/software/_helpers.py

Shared helpers for the mock-hardware test suite.
"""

from __future__ import annotations

import os

from daq.engine import Engine


def make_engine(logger=None) -> Engine:
    """Create an Engine wired to mock hardware."""
    seq_dir = os.path.join(os.path.dirname(__file__), "..", "..", "sequences")
    return Engine(
        cal_path=None,
        sequence_dir=os.path.abspath(seq_dir),
        logger=logger,
    )
