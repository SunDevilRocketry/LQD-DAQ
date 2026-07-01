"""
daq/calculations.py

Physics calculations for the liquids DAQ system.

Functions:
    type_k_uv_to_celsius()          - Type-K thermocouple conversion
    lm34_voltage_to_celsius()       - LM34 temperature sensor
    software_seebeck_type_k()       - Cold junction compensation
    pt_voltage_to_psi()             - Pressure transducer linear cal
    load_cell_voltage_to_force()    - Load cell with tare offset
    lox_density_from_celsius()      - LOX saturation density lookup
    lox_mass_flow_rate()            - LOX injector mass flow
    fuel_mass_flow_rate()           - Fuel injector mass flow
    mixture_ratio()                 - Oxidizer/fuel ratio
    impulse_step_load_cell()        - Total impulse from load cells
    impulse_step_estimate()         - Total impulse estimate
    lox_saturation_pressure_psia()  - LOX vapor pressure (Antoine)
    lox_below_saturation()          - Boiling alert check

References:
  - NIST Monograph 175, Table 10.5 (Type-K TC)
  - NIST Chemistry WebBook, SRD 69 (LOX density & Antoine Constants)
  - Sutton & Biblarz, "Rocket Propulsion Elements", 9th Ed. (Orifice Flow)
  - Texas Instruments LM34 Datasheet (SNIS155B)
"""

from __future__ import annotations

import math
from typing import Optional


# -- THERMOCOUPLE CONVERSION (Type-K) -------------------------
# NIST Monograph 175, Table 10.5 polynomial: T(°C) = c0 + c1*E + ... + c9*E^9
# E = EMF in microvolts (µV)

_TYPE_K_COEFFS_POS: tuple[float, ...] = (
    0.0,               # c0
    2.508355e-02,      # c1
    7.860106e-08,      # c2
    -2.503131e-10,     # c3
    8.315270e-14,      # c4
    -1.228034e-17,     # c5
    9.804036e-22,      # c6
    -4.413030e-26,     # c7
    1.057734e-30,      # c8
    -1.052755e-35,     # c9
)

_TYPE_K_COEFFS_NEG: tuple[float, ...] = (
    0.0,               # c0
    2.5173462e-02,     # c1
    -1.1662878e-06,    # c2
    -1.0833638e-09,    # c3
    -8.9773540e-13,    # c4
    -3.7342377e-16,    # c5
    -8.6632643e-20,    # c6
    -1.0450598e-23,    # c7
    -5.1920577e-28,    # c8
)


def type_k_uv_to_celsius(emf_uv: float) -> float:
    """
    Convert a Type-K thermocouple EMF to temperature.

    Args:
        emf_uv: Thermocouple EMF in microvolts (µV).

    Returns:
        Hot-junction temperature in degrees Celsius.

    Source:
        NIST Monograph 175, Table 10.5
    """
    coeffs = _TYPE_K_COEFFS_POS if emf_uv >= 0.0 else _TYPE_K_COEFFS_NEG
    result = 0.0
    power = 1.0
    for coeff in coeffs:
        result += coeff * power
        power *= emf_uv
    return result


# -- COLD JUNCTION COMPENSATION (CJC) -------------------------
# CJC conversion uses LM34 (10 mV/°F) and Type-K Seebeck coefficient at 25°C (NIST Monograph 175, Table 2.1).

_TYPE_K_SEEBECK_UV_PER_C = 40.7         # Linear approximation for CJC (µV/°C)

def lm34_voltage_to_celsius(voltage_v: float) -> float:
    """
    Convert an LM34 sensor output voltage to degrees Celsius.

    Args:
        voltage_v:  Raw voltage from the LM34 sensor in volts.

    Returns:
        Ambient (cold junction) temperature in degrees Celsius.

    Source:
        Texas Instruments LM34 Datasheet (SNIS155B)
    """
    fahrenheit = voltage_v * 100.0
    return (fahrenheit - 32.0) * (5.0 / 9.0)


def software_seebeck_type_k(diff_volts: float, cjc_celsius: float) -> float:
    """
    Apply cold junction compensation to a Type-K differential voltage.

    Software Seebeck correction process:
      1. Convert the CJC temperature to an equivalent EMF using the
         linear Seebeck coefficient.
      2. Add that to the measured differential EMF.
      3. Apply the NIST inverse polynomial to get hot-junction °C.

    Args:
        diff_volts:     Differential voltage across the thermocouple in volts.
        cjc_celsius:    Cold junction (ambient) temperature in degrees Celsius.

    Returns:
        Hot-junction temperature in degrees Celsius.
    """
    cjc_uv = cjc_celsius * _TYPE_K_SEEBECK_UV_PER_C   # CJC contribution in µV
    total_uv = diff_volts * 1e6 + cjc_uv               # Total EMF in µV
    return type_k_uv_to_celsius(total_uv)


# -- LINEAR CALIBRATION (PTs and Load Cells) ------------------
# Standard scaling: output = slope * voltage + intercept

def pt_voltage_to_psi(voltage_v: float, slope: float, intercept: float) -> float:
    """
    Convert a raw PT voltage to psi via a linear calibration.

    Args:
        voltage_v:  Raw voltage in volts.
        slope:      Calibration slope in psi/V.
        intercept:  Calibration intercept in psi.

    Returns:
        Pressure in psi.
    """
    return slope * voltage_v + intercept


def load_cell_voltage_to_force(
    voltage_v: float,
    slope: float,
    intercept: float,
    tare: float = 0.0,
) -> float:
    """
    Convert a raw load cell voltage to force in lbf.

    Args:
        voltage_v:  Raw voltage in volts.
        slope:      Calibration slope in lbf/V.
        intercept:  Calibration intercept in lbf.
        tare:       Tare offset in lbf (subtracted from result).

    Returns:
        Force in lbf (tare-corrected).
    """
    return slope * voltage_v + intercept - tare


# -- LOX SATURATION DENSITY -----------------------------------
# NIST SRD 69 saturation density lookup table mapping temperature (Rankine) to density (lbm/ft³).

_lox_T_rankine: list[float] = []
_lox_density_lbm_ft3: list[float] = []


def load_lox_table(csv_path: str) -> tuple[int, float, float]:
    """
    Load LOX saturation density table from CSV.

    Args:
        csv_path: Absolute or relative path to the CSV file.

    Returns:
        Tuple of (num_points, min_temp_R, max_temp_R).

    Source:
        NIST SRD 69, Oxygen Saturation Density
    """
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
    """
    Get LOX saturation density at given temperature via linear interpolation.

    Args:
        temp_celsius: LOX inlet temperature in degrees Celsius (from TOI TC).

    Returns:
        Density in lbm/ft³, or None if the table is not loaded or out of range.

    Source:
        NIST SRD 69, Oxygen Saturation Density
    """
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
# Sutton Eq. 6.15 orifice model: m = Cd * A * sqrt(2 * rho * dP)
# Inputs are converted internally from imperial (psi, in²) to SI (Pa, m²) for kg/s output.

_LOX_Cd = 0.6
_LOX_A_IN2 = 0.02922466566
_LOX_A_M2 = _LOX_A_IN2 * 6.4516e-4


def lox_mass_flow_rate(
    toi_celsius: float,
    poi_psi: float,
    pc_psi: float,
) -> Optional[float]:
    """
    Calculate LOX mass flow rate through the injector orifice.

    Args:
        toi_celsius:    LOX inlet temperature in °C.
        poi_psi:        LOX inlet pressure in psi.
        pc_psi:         Chamber pressure in psi.

    Returns:
        Mass flow rate in kg/s, or None if:
            - The LOX density table is not loaded.
            - toi_celsius is outside the table's valid range.
            - The differential pressure (poi_psi - pc_psi) is <= 0.

    Source:
        Sutton, "Rocket Propulsion Elements", Eq. 6.15
    """
    rho_lbm_ft3 = lox_density_from_celsius(toi_celsius)
    if rho_lbm_ft3 is None:
        return None

    dp_psi = poi_psi - pc_psi
    if dp_psi <= 0.0:
        return None

    rho_kg_m3 = rho_lbm_ft3 * 16.0185          
    dp_pa = dp_psi * 6894.76                    

    return _LOX_Cd * _LOX_A_M2 * math.sqrt(2.0 * rho_kg_m3 * dp_pa)


# -- FUEL (IPA) MASS FLOW RATE --------------------------------
# Orifice model using fixed IPA density (800 kg/m³) at standard operating temperature.

_FUEL_Cd = 0.67
_FUEL_A_IN2 = 0.04526
_FUEL_A_M2 = _FUEL_A_IN2 * 6.4516e-4
_FUEL_RHO_KG_M3 = 800.0


def fuel_mass_flow_rate(pfo_psi: float, pc_psi: float) -> Optional[float]:
    """
    Calculate fuel (IPA) mass flow rate through the injector orifice.

    Args:
        pfo_psi:    Fuel channel outlet pressure in psi.
        pc_psi:     Chamber pressure in psi.

    Returns:
        Mass flow rate in kg/s, or None if the differential pressure is <= 0.

    Source:
        Sutton, "Rocket Propulsion Elements", Eq. 6.15
    """
    dp_psi = pfo_psi - pc_psi
    if dp_psi <= 0.0:
        return None

    dp_pa = dp_psi * 6894.76

    return _FUEL_Cd * _FUEL_A_M2 * math.sqrt(2.0 * _FUEL_RHO_KG_M3 * dp_pa)


# -- MIXTURE RATIO (O/F) --------------------------------------
# Sutton Eq. 2.7: O/F = oxidizer_mass_flow / fuel_mass_flow

def mixture_ratio(
    lox_mdot_kg_s: Optional[float],
    fuel_mdot_kg_s: Optional[float],
) -> Optional[float]:
    """
    Calculate the oxidizer-to-fuel mixture ratio (O/F).

    Args:
        lox_mdot_kg_s:  LOX mass flow rate in kg/s.
        fuel_mdot_kg_s: Fuel mass flow rate in kg/s.

    Returns:
        O/F ratio, or None if either flow rate is None or fuel flow is <= 0.

    Source:
        Sutton, "Rocket Propulsion Elements", Eq. 2.7
    """
    if lox_mdot_kg_s is None or fuel_mdot_kg_s is None:
        return None
    if fuel_mdot_kg_s <= 0.0:
        return None
    return lox_mdot_kg_s / fuel_mdot_kg_s


# -- TOTAL IMPULSE --------------------------------------------
# Sutton Eq. 2.1 total impulse integration (Trapezoidal primary, mathematical fallback secondary).

_G0_M_S2 = 9.80665          
_ISP_ESTIMATE_S = 220.0     # Update with actual design value before hot fire

def impulse_step_load_cell(
    force_lbf_prev: float,
    force_lbf_curr: float,
    dt_seconds: float,
) -> float:
    """
    Compute one trapezoidal integration step for total impulse from load cells.

    Args:
        force_lbf_prev: Thrust force at the previous sample in lbf.
        force_lbf_curr: Thrust force at the current sample in lbf.
        dt_seconds:     Time elapsed since the previous sample in seconds.

    Returns:
        Incremental impulse in lbf*s for this step.

    Source:
        Sutton, "Rocket Propulsion Elements", Eq. 2.1
    """
    return 0.5 * (force_lbf_prev + force_lbf_curr) * dt_seconds


def impulse_step_estimate(
    lox_mdot_kg_s: Optional[float],
    fuel_mdot_kg_s: Optional[float],
    dt_seconds: float,
    isp_seconds: float = _ISP_ESTIMATE_S,
) -> Optional[float]:
    """
    Estimate one impulse step from mass flow rates when load cells are absent.

    Uses F ≈ (m_total) * Isp * g0  then I_step = F * dt.

    Args:
        lox_mdot_kg_s:  LOX mass flow rate in kg/s (or None).
        fuel_mdot_kg_s: Fuel mass flow rate in kg/s (or None).
        dt_seconds:     Time elapsed since previous sample in seconds.
        isp_seconds:    Specific impulse estimate in seconds.

    Returns:
        Incremental impulse in N*s, or None if either flow rate is None.

    Source:
        Sutton, "Rocket Propulsion Elements", Eq. 2.1
    """
    if lox_mdot_kg_s is None or fuel_mdot_kg_s is None:
        return None

    total_mdot = lox_mdot_kg_s + fuel_mdot_kg_s
    thrust_n = total_mdot * isp_seconds * _G0_M_S2
    return thrust_n * dt_seconds


# -- LOX SATURATION PRESSURE (Antoine Equation) ---------------
# NIST SRD 69 Antoine equation: log10(P_sat [bar]) = A - B / (T [K] + C)
# Constants for oxygen: A = 3.9523, B = 340.024, C = -4.144 (valid range: 54.361 K to 154.58 K)

_ANTOINE_A = 3.9523
_ANTOINE_B = 340.024
_ANTOINE_C = -4.144


def lox_saturation_pressure_psia(temp_celsius: float) -> Optional[float]:
    """
    Estimate LOX saturation pressure using the Antoine equation.

    Args:
        temp_celsius: LOX temperature in degrees Celsius.

    Returns:
        Saturation pressure in psia, or None if temperature is out of range.

    Source:
        NIST SRD 69, Oxygen Antoine Equation Constants
    """
    temp_k = temp_celsius + 273.15

    if not (54.361 <= temp_k <= 154.58):
        return None

    log_p_bar = _ANTOINE_A - _ANTOINE_B / (temp_k + _ANTOINE_C)
    p_bar = 10.0 ** log_p_bar
    p_psia = p_bar * 14.5038

    return p_psia


def lox_below_saturation(
    tank_pressure_psia: float,
    toi_celsius: float,
) -> Optional[bool]:
    """
    Check whether the LOX tank pressure is below the saturation pressure.

    Args:
        tank_pressure_psia: Measured LOX tank pressure in psia (from POT).
        toi_celsius:        LOX inlet temperature in °C (from TOI).

    Returns:
        True:   tank pressure is below saturation.
        False:  tank pressure is above saturation.
        None:   saturation pressure could not be computed.

    Source:
        System Requirement 3.2.9.2 (Saturation Pressure Alert)
    """
    p_sat = lox_saturation_pressure_psia(toi_celsius)
    if p_sat is None:
        return None
    return tank_pressure_psia < p_sat
