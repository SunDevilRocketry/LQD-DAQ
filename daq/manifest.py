"""
daq/manifest.py

Channel and actuator specs (channels.yaml / actuators.yaml) and their
readings. A null pin is a placeholder: still reported, can't be read or
driven.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import yaml


# Channel types
PT_DIRECT         = "pt_direct"
# Orifice pressure drop; wired and calibrated like pt_direct.
PT_DIFFERENTIAL   = "pt_differential"
TC_DIFFERENTIAL   = "tc_differential"
LC_DIRECT         = "lc_direct"
PHOTOGATE_COUNTER = "photogate_counter"

# Actuator types
BINARY_DIO    = "binary_dio"
PULSE_STEPPER = "pulse_stepper"

# Read via the AIN stream; photogate counters are polled out-of-band.
_STREAMED_TYPES = frozenset(
    {PT_DIRECT, PT_DIFFERENTIAL, TC_DIFFERENTIAL, LC_DIRECT}
)

# AIN16+ refused: extended (Mux80) channels drive MIO0-2, the igniter lines.
MAX_AIN = 15


def check_ain(name: Optional[str]) -> Optional[str]:
    """
    Return an AIN register name unchanged, refusing extended channels.

    Raises:
        ValueError: If the name is AIN16 or higher.
    """
    if name is not None:
        match = re.fullmatch(r"AIN(\d+)", name.strip().upper())
        if match and int(match.group(1)) > MAX_AIN:
            raise ValueError(
                f"{name} is an extended channel (AIN16+), which drives MIO0-2 "
                f"(igniter lines). Use AIN0-AIN{MAX_AIN}."
            )
    return name


# Calibrated under calibration.json "PT"; thresholds authored in psi.
PRESSURE_TYPES = frozenset({PT_DIRECT, PT_DIFFERENTIAL})


@dataclass(frozen=True)
class ChannelSpec:
    """One sensor channel as declared in channels.yaml."""
    id:       str
    type:     str
    unit:     str
    cal_ref:  Optional[str] = None
    active:   bool          = True
    ain:      Optional[str] = None   # pt_direct / pt_differential / lc_direct
    ain_pos:  Optional[str] = None   # tc_differential
    ain_neg:  Optional[str] = None   # tc_differential
    dio:      Optional[str] = None   # photogate_counter

    @property
    def is_streamed(self) -> bool:
        """True if this channel's samples arrive via the AIN stream batch."""
        return self.type in _STREAMED_TYPES

    @property
    def is_wired(self) -> bool:
        """True if the manifest names a physical pin for this channel."""
        if self.type in (PT_DIRECT, PT_DIFFERENTIAL, LC_DIRECT):
            return self.ain is not None
        if self.type == TC_DIFFERENTIAL:
            return self.ain_pos is not None and self.ain_neg is not None
        if self.type == PHOTOGATE_COUNTER:
            return self.dio is not None
        return False


@dataclass(frozen=True)
class ActuatorSpec:
    """One actuator as declared in actuators.yaml."""
    id:            str
    type:          str
    # De-energised state: "open" / "closed" / None. Reported only.
    normal:        Optional[str]   = None
    dio:           Optional[str]   = None   # binary_dio
    step_dio:      Optional[str]   = None   # pulse_stepper
    dir_dio:       Optional[str]   = None
    ena_dio:       Optional[str]   = None   # optional; None = ENA unconnected
    steps_open:    Optional[int]   = None
    steps_close:   Optional[int]   = None
    pulse_freq_hz: Optional[float] = None
    # Photogate channel id; only the mock reads it.
    position_feedback: Optional[str] = None

    @property
    def is_wired(self) -> bool:
        """True if the manifest names the physical pin(s) this type needs."""
        if self.type == BINARY_DIO:
            return self.dio is not None
        if self.type == PULSE_STEPPER:
            return None not in (
                self.step_dio, self.dir_dio,
                self.steps_open, self.steps_close, self.pulse_freq_hz,
            )
        return False


@dataclass(frozen=True)
class ChannelReading:
    """
    One channel's latest value.

    `status`: NOMINAL | CAUTION | WARNING, UNCONFIGURED (no bands), or None
    (bands but no value).
    """
    value:        Optional[float]
    unit:         str
    status:       Optional[str]
    last_updated: Optional[float] # epoch seconds


@dataclass(frozen=True)
class ActuatorReading:
    """
    One actuator's latest state.

    `state`: commanded, 1 = energised, 0 = de-energised.
    `moving`: True while a stepper move is in flight; always False for
    binary_dio.
    """
    state:  int
    moving: bool = False


def _ain_name(value: Any) -> Optional[str]:
    """Normalise an AIN field (2 or 'AIN2') to an LJM name, refusing AIN16+."""
    if value is None:
        return None
    if isinstance(value, int):
        return check_ain(f"AIN{value}")
    return check_ain(str(value))


def _active(value: Any) -> bool:
    """Absent or null = active (still reports); only `false` disables."""
    return True if value is None else bool(value)


def load_channels(path: str) -> list[ChannelSpec]:
    """
    Parse channels.yaml into ChannelSpec objects.

    Raises:
        ValueError: On a duplicate id, an unrecognised channel type, or an
                    extended channel (AIN16+).
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    specs: list[ChannelSpec] = []
    seen: set[str] = set()

    for entry in data.get("channels", []):
        cid   = str(entry["id"])
        ctype = str(entry["type"])

        if cid in seen:
            raise ValueError(f"channels.yaml: duplicate channel id '{cid}'")
        if ctype not in (PT_DIRECT, PT_DIFFERENTIAL, TC_DIFFERENTIAL,
                         LC_DIRECT, PHOTOGATE_COUNTER):
            raise ValueError(
                f"channels.yaml: channel '{cid}' has unknown type '{ctype}'"
            )
        seen.add(cid)

        specs.append(ChannelSpec(
            id      = cid,
            type    = ctype,
            unit    = str(entry.get("unit", "")),
            cal_ref = entry.get("cal_ref"),
            active  = _active(entry.get("active")),
            ain     = _ain_name(entry.get("ain")),
            ain_pos = _ain_name(entry.get("ain_pos")),
            ain_neg = _ain_name(entry.get("ain_neg")),
            dio     = entry.get("dio"),
        ))

    return specs


def load_actuators(path: str) -> list[ActuatorSpec]:
    """
    Parse actuators.yaml into ActuatorSpec objects.

    Raises:
        ValueError: On a duplicate id or an unrecognised actuator type.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    specs: list[ActuatorSpec] = []
    seen: set[str] = set()

    for entry in data.get("actuators", []):
        aid   = str(entry["id"])
        atype = str(entry["type"])

        if aid in seen:
            raise ValueError(f"actuators.yaml: duplicate actuator id '{aid}'")
        if atype not in (BINARY_DIO, PULSE_STEPPER):
            raise ValueError(
                f"actuators.yaml: actuator '{aid}' has unknown type '{atype}'"
            )

        normal = entry.get("normal")
        if normal is not None and normal not in ("open", "closed"):
            raise ValueError(
                f"actuators.yaml: actuator '{aid}' has invalid normal "
                f"'{normal}' (expected 'open', 'closed' or null)"
            )
        seen.add(aid)

        feedback = entry.get("position_feedback") or {}
        specs.append(ActuatorSpec(
            id            = aid,
            type          = atype,
            normal        = normal,
            dio           = entry.get("dio"),
            step_dio      = entry.get("step_dio"),
            dir_dio       = entry.get("dir_dio"),
            ena_dio       = entry.get("ena_dio"),
            steps_open    = entry.get("steps_open"),
            steps_close   = entry.get("steps_close"),
            pulse_freq_hz = entry.get("pulse_freq_hz"),
            position_feedback = (
                feedback.get("source") if feedback.get("enabled") else None
            ),
        ))

    return specs
