"""
tests/software/_helpers.py

Shared helpers for the mock-hardware test suite.
"""

from __future__ import annotations

import os
import time

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


def make_engine(
    logger=None,
    thresholds=None,
    sequence_dir=None,
    actuators_path=FULL_ACTUATORS,
) -> Engine:
    """
    Create an Engine wired to mock hardware and the fixture manifests.

    actuators_path defaults to the fully-wired fixture; pass REPO_ACTUATORS
    to exercise the cart as actually shipped, with no pins assigned.
    """
    return Engine(
        cal_path=None,
        sequence_dir=sequence_dir or os.path.join(_REPO, "sequences"),
        logger=logger,
        thresholds=thresholds,
        channels_path=FULL_CHANNELS,
        actuators_path=actuators_path,
    )


def wait_for_first_batch(engine, timeout: float = 5.0) -> None:
    """
    Block until the engine has published its first populated snapshot.

    Raises:
        AssertionError: If no batch has landed within `timeout` seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if engine.snapshot.channels:
            return
        time.sleep(0.01)
    raise AssertionError(
        f"Engine produced no populated snapshot within {timeout:.1f}s - "
        f"the acquisition thread never completed a batch"
    )
