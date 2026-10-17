"""
daq/calculations.py

Unit conversions and physics: thermocouple, calibration, LOX properties,
injector flow.
"""

from __future__ import annotations

import math
from typing import Optional


# -- THERMOCOUPLE CONVERSION (Type T) ------------------------
# NIST ITS-90 Type T; E in mV, t in C, constant term first.

# Reference function E(t), -270..0 C.
_TYPE_T_REF_NEG: tuple[float, ...] = (
    0.000000000000E+00,
    0.387481063640E-01,
    0.441944343470E-04,
    0.118443231050E-06,
    0.200329735540E-07,
    0.901380195590E-09,
    0.226511565930E-10,
    0.360711542050E-12,
    0.384939398830E-14,
    0.282135219250E-16,
    0.142515947790E-18,
    0.487686622860E-21,
    0.107955392700E-23,
    0.139450270620E-26,
    0.797951539270E-30,
)

# Reference function E(t), 0..400 C.
_TYPE_T_REF_POS: tuple[float, ...] = (
    0.000000000000E+00,
    0.387481063640E-01,
    0.332922278800E-04,
    0.206182434040E-06,
    -0.218822568460E-08,
    0.109968809280E-10,
    -0.308157587720E-13,
    0.454791352900E-16,
    -0.275129016730E-19,
)

# Inverse function t(E), -200..0 C (-5.603..0 mV).
_TYPE_T_INV_NEG: tuple[float, ...] = (
    0.0000000E+00,
    2.5949192E+01,
    -2.1316967E-01,
    7.9018692E-01,
    4.2527777E-01,
    1.3304473E-01,
    2.0241446E-02,
    1.2668171E-03,
)

# Inverse function t(E), 0..400 C (0..20.872 mV).
_TYPE_T_INV_POS: tuple[float, ...] = (
    0.000000E+00,
    2.592800E+01,
    -7.602961E-01,
    4.637791E-02,
    -2.165394E-03,
    6.048144E-05,
    -7.293422E-07,
)

_TYPE_T_INV_MIN_MV = -5.603
_TYPE_T_INV_MAX_MV = 20.872


def _poly(coeffs: tuple[float, ...], x: float) -> float:
    result = 0.0
    for coeff in reversed(coeffs):
        result = result * x + coeff
    return result


def type_t_celsius_to_mv(temp_c: float) -> float:
    """Type T EMF (mV, 0 C reference) at a junction temperature, -270..400 C."""
    coeffs = _TYPE_T_REF_NEG if temp_c < 0.0 else _TYPE_T_REF_POS
    return _poly(coeffs, temp_c)


def type_t_mv_to_celsius(emf_mv: float) -> Optional[float]:
    """Type T junction C for an EMF (mV), or None outside -200..400 C."""
    if not _TYPE_T_INV_MIN_MV <= emf_mv <= _TYPE_T_INV_MAX_MV:
        return None
    coeffs = _TYPE_T_INV_NEG if emf_mv < 0.0 else _TYPE_T_INV_POS
    return _poly(coeffs, emf_mv)


def thermocouple_type_t(diff_volts: float, cjc_celsius: float) -> Optional[float]:
    """
    Hot-junction C: inverse of (measured EMF + EMF of the cold junction).

    Returns None outside -200..400 C.
    """
    return type_t_mv_to_celsius(diff_volts * 1e3 + type_t_celsius_to_mv(cjc_celsius))


PSI_TO_PA = 6894.757293168361


def psi_to_pa(psi: float) -> float:
    """Convert psi to pascals (SI)."""
    return psi * PSI_TO_PA


def pa_to_psi(pa: float) -> float:
    """Convert pascals to psi."""
    return pa / PSI_TO_PA


# -- LINEAR CALIBRATION (PTs and Load Cells) ------------------

def pt_voltage_to_psi(voltage_v: float, slope: float, intercept: float) -> float:
    """PT pressure in psi: slope [psi/V] * V + intercept [psi]."""
    return slope * voltage_v + intercept


def pt_voltage_to_pa(voltage_v: float, slope: float, intercept: float) -> float:
    """PT pressure in Pa, from a psi-domain calibration."""
    return psi_to_pa(pt_voltage_to_psi(voltage_v, slope, intercept))


def load_cell_voltage_to_force(
    voltage_v: float,
    slope: float,
    intercept: float,
    tare: float = 0.0,
) -> float:
    """Load cell force in lbf: slope [lbf/V] * V + intercept - tare."""
    return slope * voltage_v + intercept - tare


# -- LOX SATURATION DENSITY -----------------------------------
# Table: Rankine -> lbm/ft³. Source UNVERIFIED.

_lox_T_rankine: list[float] = []
_lox_density_lbm_ft3: list[float] = []


def load_lox_table(csv_path: str) -> tuple[int, float, float]:
    """Load the LOX density CSV. Returns (num_points, min_R, max_R)."""
    global _lox_T_rankine, _lox_density_lbm_ft3

    import csv

    temps: list[float] = []
    densities: list[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                temps.append(float(row[0]))
                densities.append(float(row[1]))

    if not temps:
        raise ValueError(f"LOX table at '{csv_path}' contained no data rows.")

    _lox_T_rankine = temps
    _lox_density_lbm_ft3 = densities
    return len(temps), temps[0], temps[-1]


def lox_density_from_celsius(temp_celsius: float) -> Optional[float]:
    """LOX density [lbm/ft³] at temp_celsius; None if unloaded or out of range."""
    if not _lox_T_rankine:
        return None

    t_r = (temp_celsius + 273.15) * (9.0 / 5.0)

    if t_r < _lox_T_rankine[0] or t_r > _lox_T_rankine[-1]:
        return None

    # Binary search for the surrounding bracket
    lo, hi = 0, len(_lox_T_rankine) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _lox_T_rankine[mid] <= t_r:
            lo = mid
        else:
            hi = mid

    t0, t1 = _lox_T_rankine[lo], _lox_T_rankine[hi]
    d0, d1 = _lox_density_lbm_ft3[lo], _lox_density_lbm_ft3[hi]
    frac = (t_r - t0) / (t1 - t0)
    return d0 + frac * (d1 - d0)


# -- MASS FLOW RATES (Orifice Model) --------------------------
# Incompressible orifice: mdot = Cd * A * sqrt(2 * rho * dP), SI units.

def orifice_area_m2(diameter_in: float, count: int = 1) -> float:
    """Total area [m²] of `count` round orifices of diameter_in inches."""
    area_in2 = count * math.pi / 4.0 * diameter_in ** 2
    return area_in2 * 6.4516e-4   # in² -> m²


def _orifice_mdot(
    dp_pa: float,
    area_m2: float,
    cd: float,
    density_kg_m3: float,
) -> Optional[float]:
    """Orifice mass flow [kg/s]; None without a forward dP."""
    if dp_pa is None or dp_pa <= 0.0:
        return None
    return cd * area_m2 * math.sqrt(2.0 * density_kg_m3 * dp_pa)


def lox_mass_flow_rate(
    toi_celsius: float,
    dp_pa: float,
    area_m2: float,
    cd: float,
) -> Optional[float]:
    """
    LOX injector mass flow [kg/s], density from the table at toi_celsius.

    None if the table isn't loaded, toi_celsius is out of range, or dP <= 0.
    """
    if toi_celsius is None:
        return None
    rho_lbm_ft3 = lox_density_from_celsius(toi_celsius)
    if rho_lbm_ft3 is None:
        return None
    rho_kg_m3 = rho_lbm_ft3 * 16.0185   # lbm/ft³ -> kg/m³
    return _orifice_mdot(dp_pa, area_m2, cd, rho_kg_m3)


def fuel_mass_flow_rate(
    dp_pa: float,
    area_m2: float,
    cd: float,
    density_kg_m3: float,
) -> Optional[float]:
    """Fuel injector mass flow [kg/s] at a fixed density (no fuel temp channel)."""
    return _orifice_mdot(dp_pa, area_m2, cd, density_kg_m3)


# -- MIXTURE RATIO (O/F) --------------------------------------

def mixture_ratio(
    lox_mdot_kg_s: Optional[float],
    fuel_mdot_kg_s: Optional[float],
) -> Optional[float]:
    """O/F = LOX / fuel mass flow; None if either is None or fuel <= 0."""
    if lox_mdot_kg_s is None or fuel_mdot_kg_s is None:
        return None
    if fuel_mdot_kg_s <= 0.0:
        return None
    return lox_mdot_kg_s / fuel_mdot_kg_s


# -- TOTAL IMPULSE --------------------------------------------
# Unused until total impulse is restored (handoff "Later").

_G0_M_S2 = 9.80665
_ISP_ESTIMATE_S = 220.0     # Guessed Isp; slated for removal

def impulse_step_load_cell(
    force_lbf_prev: float,
    force_lbf_curr: float,
    dt_seconds: float,
) -> float:
    """Trapezoidal impulse step [lbf*s] between two load-cell samples."""
    return 0.5 * (force_lbf_prev + force_lbf_curr) * dt_seconds


def impulse_step_estimate(
    lox_mdot_kg_s: Optional[float],
    fuel_mdot_kg_s: Optional[float],
    dt_seconds: float,
    isp_seconds: float = _ISP_ESTIMATE_S,
) -> Optional[float]:
    """Impulse step [N*s] from mass flow and a guessed Isp. Slated for removal."""
    if lox_mdot_kg_s is None or fuel_mdot_kg_s is None:
        return None

    total_mdot = lox_mdot_kg_s + fuel_mdot_kg_s
    thrust_n = total_mdot * isp_seconds * _G0_M_S2
    return thrust_n * dt_seconds


# -- LOX SATURATION PRESSURE (Antoine Equation) ---------------
# log10(P [bar]) = A - B / (T [K] + C); oxygen, NIST WebBook.

_ANTOINE_A = 3.9523
_ANTOINE_B = 340.024
_ANTOINE_C = -4.144


_BAR_TO_PA = 100_000.0


def lox_saturation_pressure_pa(temp_celsius: float) -> Optional[float]:
    """LOX saturation pressure [Pa abs], or None outside the fit's range."""
    temp_k = temp_celsius + 273.15

    # NIST's stated range for these constants.
    if not (54.36 <= temp_k <= 154.33):
        return None

    log_p_bar = _ANTOINE_A - _ANTOINE_B / (temp_k + _ANTOINE_C)
    p_bar = 10.0 ** log_p_bar

    return p_bar * _BAR_TO_PA


def lox_below_saturation(
    tank_pressure_pa: float,
    toi_celsius: float,
) -> Optional[bool]:
    """
    True if LOX tank pressure [Pa abs] is below saturation at toi_celsius.

    None if saturation can't be computed. SR 3.2.9.2; unused until the
    LOX-boiling advisory lands.
    """
    p_sat = lox_saturation_pressure_pa(toi_celsius)
    if p_sat is None:
        return None
    return tank_pressure_pa < p_sat
