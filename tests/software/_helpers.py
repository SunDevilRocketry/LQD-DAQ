"""
tests/software/_helpers.py

Shared helpers for the mock-hardware test suite.
"""

from __future__ import annotations

import os

from daq.engine import Engine
from daq.manifest import load_actuators, load_channels


_HERE     = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.abspath(os.path.join(_HERE, "..", ".."))
_FIXTURES = os.path.join(_HERE, "fixtures")

# The fully-wired fixture manifests, not the repo's own channels.yaml.
# The real manifests leave most pins null and most channels inactive
# (nothing is confirmed on the cart yet), which would leave the TC, load
# cell, stepper and photogate paths untested.
FULL_CHANNELS  = os.path.join(_FIXTURES, "channels_full.yaml")
FULL_ACTUATORS = os.path.join(_FIXTURES, "actuators_full.yaml")

# The manifests actually shipped in the repo.
REPO_CHANNELS  = os.path.join(_REPO, "channels.yaml")
REPO_ACTUATORS = os.path.join(_REPO, "actuators.yaml")


def full_channels():
    """The fully-wired fixture channel manifest."""
    return load_channels(FULL_CHANNELS)


def full_actuators():
    """The fully-wired fixture actuator manifest."""
    return load_actuators(FULL_ACTUATORS)


def make_engine(logger=None, thresholds=None) -> Engine:
    """Create an Engine wired to mock hardware and the fixture manifests."""
    seq_dir = os.path.join(_REPO, "sequences")
    return Engine(
        cal_path=None,
        sequence_dir=seq_dir,
        logger=logger,
        thresholds=thresholds,
        channels_path=FULL_CHANNELS,
        actuators_path=FULL_ACTUATORS,
    )
