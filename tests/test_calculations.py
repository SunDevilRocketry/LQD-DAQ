"""
tests/test_calculations.py

Unit tests for daq/calculations.py.

Run with:
    python -m pytest tests/test_calculations.py -v

"""

import math
import sys
import os
import tempfile
import textwrap

# Make the package importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from daq.calculations import (
    type_k_uv_to_celsius,
    lm34_voltage_to_celsius,
    software_seebeck_type_k,
    psi_to_pa,
    pa_to_psi,
    PSI_TO_PA,
    pt_voltage_to_psi,
    pt_voltage_to_pa,
    load_cell_voltage_to_force,
    load_lox_table,
    lox_density_from_celsius,
    lox_mass_flow_rate,
    fuel_mass_flow_rate,
    mixture_ratio,
    impulse_step_load_cell,
    impulse_step_estimate,
    lox_saturation_pressure_pa,
    lox_below_saturation,
)

# -- Helper Utilities ----------------------------------------

def _write_lox_table(rows: list[tuple[float, float]]) -> str:
    """Writes a temporary minimal LOX density table to disk and returns the path."""
    content = "temperature_R,density_lbm_ft3\n"
    for t, d in rows:
        content += f"{t},{d}\n"
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    f.write(content)
    f.close()
    return f.name


# -- Type-K Thermocouple Polynomial (NIST ITS-90) ------------

class TestTypeKInversePolynomial:
    """NIST Monograph 175, Table 10.5 polynomial conversions."""

    def test_zero_emf_gives_zero_celsius(self):
        result = type_k_uv_to_celsius(0.0)
        assert abs(result) < 1e-6, f"Expected 0 °C, got {result}"

    def test_positive_emf_positive_temp(self):
        # NIST: 1000 µV ≈ 24.91 °C for Type K
        result = type_k_uv_to_celsius(1000.0)
        assert abs(result - 24.91) < 0.1, f"Expected ~24.91 °C, got {result}"

    def test_negative_emf_negative_temp(self):
        # NIST: -1000 µV ≈ -25.00 °C for Type K (negative branch)
        result = type_k_uv_to_celsius(-1000.0)
        assert result < 0.0, "Negative EMF should give negative temperature"

    def test_large_positive_emf(self):
        # NIST: ~20 000 µV ≈ 475 °C
        result = type_k_uv_to_celsius(20_000.0)
        assert 460.0 < result < 490.0, f"Expected ~475 °C, got {result}"

    def test_selects_positive_branch_at_zero(self):
        # Positive branch used for emf >= 0
        pos = type_k_uv_to_celsius(0.0)
        neg = type_k_uv_to_celsius(-1e-9)  # just below 0
        # Both should be very close to 0 °C (no discontinuity at origin)
        assert abs(pos - neg) < 0.01


# -- LM34 CJC Temperature Sensor ------------------------------

class TestLM34VoltageToCelsius:
    """LM34 sensor output conversions (10 mV/°F, 0 V = 0 °F)."""

    def test_room_temperature(self):
        # 77 °F = 25 °C; LM34 outputs 10 mV/°F -> 770 mV = 0.770 V
        result = lm34_voltage_to_celsius(0.770)
        assert abs(result - 25.0) < 0.1, f"Expected ~25 °C, got {result}"

    def test_freezing_point_fahrenheit(self):
        # 32 °F = 0 °C; LM34 outputs 320 mV = 0.320 V
        result = lm34_voltage_to_celsius(0.320)
        assert abs(result - 0.0) < 0.1, f"Expected ~0 °C, got {result}"

    def test_zero_volts_is_minus_17_8_celsius(self):
        # 0 V = 0 °F = -17.78 °C
        result = lm34_voltage_to_celsius(0.0)
        assert abs(result - (-17.78)) < 0.1, f"Expected ~-17.78 °C, got {result}"

    def test_linearity(self):
        # Doubling the voltage from 0.5 V to 1.0 V should double °F reading
        t1 = lm34_voltage_to_celsius(0.5)
        t2 = lm34_voltage_to_celsius(1.0)
        # 0.5 V = 50 °F = 10 °C; 1.0 V = 100 °F = 37.78 °C
        assert abs(t1 - 10.0) < 0.1
        assert abs(t2 - 37.78) < 0.1


# -- Cold Junction Compensation (Seebeck) ---------------------

class TestSoftwareSeebeck:
    """CJC offset and differential TC math integrations."""

    def test_zero_differential_zero_cjc_gives_zero(self):
        result = software_seebeck_type_k(0.0, 0.0)
        assert abs(result) < 0.1, f"Expected ~0 °C, got {result}"

    def test_cjc_offset_adds_correctly(self):
        # With 0 V differential and 25 °C CJC, hot junction = CJC temp
        # b/c the TC adds nothing
        result = software_seebeck_type_k(0.0, 25.0)
        # 25 °C * 40.7 µV/°C = 1017.5 µV -> type_k_uv_to_celsius(1017.5) ≈ 25.3 °C
        assert abs(result - 25.3) < 0.5, f"Expected ~25 °C, got {result}"

    def test_positive_differential_increases_temperature(self):
        baseline = software_seebeck_type_k(0.0, 25.0)
        higher   = software_seebeck_type_k(0.001, 25.0)   # 1 mV extra
        assert higher > baseline, "Positive dV should increase inferred temperature"

    def test_output_matches_goondaq_reference(self):
        """
        Cross-check against the GoonDAQ monolith formula:
            cjc_uv  = cjc_c * 40.7
            total   = diff_v * 1e6 + cjc_uv
            result  = type_k_uv_to_celsius(total)
        """
        diff_v = 0.002      # 2 mV differential
        cjc_c  = 23.5

        # Replicate monolith manually
        cjc_uv   = cjc_c * 40.7
        total_uv = diff_v * 1e6 + cjc_uv
        expected = type_k_uv_to_celsius(total_uv)

        result = software_seebeck_type_k(diff_v, cjc_c)
        assert abs(result - expected) < 1e-9, (
            f"Mismatch with GoonDAQ reference: expected {expected}, got {result}"
        )


# -- SI Unit Conversion Helpers --------------------------------

class TestUnitConversion:
    """psi <-> Pa conversion (team convention: internal SI, Jul 2026)."""

    def test_one_psi_in_pa(self):
        assert abs(psi_to_pa(1.0) - 6894.757293168361) < 1e-6

    def test_round_trip(self):
        assert abs(pa_to_psi(psi_to_pa(37.2)) - 37.2) < 1e-9

    def test_zero(self):
        assert psi_to_pa(0.0) == 0.0
        assert pa_to_psi(0.0) == 0.0

    def test_one_atmosphere_sanity_check(self):
        # 14.696 psia is ~1 standard atmosphere (101325 Pa) - a good
        # sanity check that the conversion factor is right.
        assert abs(psi_to_pa(14.696) - 101325.0) < 1.0


# -- Pressure Transducers -------------------------------------

class TestPTConversion:
    """Linear pressure transducer calibrations (raw psi calibration)."""

    def test_default_calibration_midpoint(self):
        # slope=252, intercept=-119.5 -> at 0.5 V: 252*0.5 - 119.5 = 6.5 psi
        result = pt_voltage_to_psi(0.5, slope=252.0, intercept=-119.5)
        assert abs(result - 6.5) < 1e-6

    def test_zero_volts(self):
        result = pt_voltage_to_psi(0.0, slope=252.0, intercept=-119.5)
        assert abs(result - (-119.5)) < 1e-6

    def test_linearity(self):
        s, i = 128.0, -62.8
        r1 = pt_voltage_to_psi(1.0, s, i)
        r2 = pt_voltage_to_psi(2.0, s, i)
        assert abs((r2 - r1) - s) < 1e-6, "Delta should equal slope"


class TestPTConversionPa:
    """Pa-native PT conversion"""

    def test_matches_psi_conversion_times_factor(self):
        v, s, i = 0.5, 252.0, -119.5
        psi_result = pt_voltage_to_psi(v, s, i)
        pa_result  = pt_voltage_to_pa(v, s, i)
        assert abs(pa_result - psi_result * PSI_TO_PA) < 1e-6

    def test_zero_volts(self):
        result = pt_voltage_to_pa(0.0, slope=252.0, intercept=-119.5)
        assert abs(result - psi_to_pa(-119.5)) < 1e-6

    def test_linearity(self):
        s, i = 128.0, -62.8
        r1 = pt_voltage_to_pa(1.0, s, i)
        r2 = pt_voltage_to_pa(2.0, s, i)
        assert abs((r2 - r1) - psi_to_pa(s)) < 1e-6, "Delta should equal slope in Pa"


# -- Load Cells -----------------------------------------------

class TestLoadCellConversion:
    def test_basic_conversion(self):
        result = load_cell_voltage_to_force(1.0, slope=100.0, intercept=0.0)
        assert abs(result - 100.0) < 1e-6

    def test_tare_applied(self):
        result = load_cell_voltage_to_force(
            1.0, slope=100.0, intercept=0.0, tare=10.0
        )
        assert abs(result - 90.0) < 1e-6

    def test_zero_after_tare(self):
        # Tare @ current reading -> output should be 0
        raw = 1.5
        force_untared = load_cell_voltage_to_force(raw, 100.0, 0.0)
        result = load_cell_voltage_to_force(raw, 100.0, 0.0, tare=force_untared)
        assert abs(result) < 1e-9


# -- LOX Saturation Density Lookup Table ----------------------

class TestLOXDensityTable:

    @pytest.fixture(autouse=True)
    def minimal_table(self, tmp_path):
        """Linear ramp 100-200 R, 71-61 lbm/ft³."""
        rows = [(100.0 + i * 10.0, 71.0 - i * 1.0) for i in range(11)]
        path = _write_lox_table(rows)
        load_lox_table(path)
        yield
        os.unlink(path)

    def test_load_returns_count_and_bounds(self, tmp_path):
        rows = [(100.0 + i * 10.0, 71.0 - i * 1.0) for i in range(11)]
        path = _write_lox_table(rows)
        n, lo, hi = load_lox_table(path)
        assert n == 11
        assert lo == 100.0
        assert hi == 200.0
        os.unlink(path)

    def test_exact_table_point_interpolates_correctly(self):
        # At T = 100 R = (100 * 5/9) - 273.15 °C
        t_r = 150.0    # midpoint of our table
        t_c = t_r * (5.0 / 9.0) - 273.15
        result = lox_density_from_celsius(t_c)
        # At 150 R in our linear table: density = 71 - 5 = 66 lbm/ft³
        assert result is not None
        assert abs(result - 66.0) < 0.05

    def test_below_range_returns_none(self):
        # Temp below the minimum in table
        t_c = (90.0 * 5.0 / 9.0) - 273.15  # 90 R, below 100 R minimum
        result = lox_density_from_celsius(t_c)
        assert result is None

    def test_above_range_returns_none(self):
        t_c = (210.0 * 5.0 / 9.0) - 273.15  # 210 R, above 200 R maximum
        result = lox_density_from_celsius(t_c)
        assert result is None

    def test_interpolation_is_linear(self):
        # At exactly halfway between two table points, should be the average
        t_mid_r = 145.0
        t_mid_c = t_mid_r * (5.0 / 9.0) - 273.15
        result = lox_density_from_celsius(t_mid_c)
        # Between 140 R (67 lbm/ft³) and 150 R (66 lbm/ft³): midpoint = 66.5
        assert result is not None
        assert abs(result - 66.5) < 0.05


# -- LOX Mass Flow Rate ---------------------------------------

class TestLOXMassFlowRate:

    @pytest.fixture(autouse=True)
    def real_table(self):
        # Span table across 180-240 R to cover -160 °C (≈203.67 R)
        rows = [(180.0 + i * 5.0, 70.0 - i * 0.2) for i in range(13)]
        path = _write_lox_table(rows)
        load_lox_table(path)
        yield
        os.unlink(path)

    def test_returns_positive_value_for_valid_inputs(self):
        result = lox_mass_flow_rate(-160.0, poi_pa=psi_to_pa(250.0), pc_pa=psi_to_pa(150.0))
        assert result is not None
        assert result > 0.0

    def test_zero_dp_returns_none(self):
        result = lox_mass_flow_rate(-160.0, poi_pa=psi_to_pa(200.0), pc_pa=psi_to_pa(200.0))
        assert result is None

    def test_negative_dp_returns_none(self):
        result = lox_mass_flow_rate(-160.0, poi_pa=psi_to_pa(150.0), pc_pa=psi_to_pa(200.0))
        assert result is None

    def test_higher_dp_gives_higher_flow(self):
        lo = lox_mass_flow_rate(-160.0, poi_pa=psi_to_pa(200.0), pc_pa=psi_to_pa(150.0))
        hi = lox_mass_flow_rate(-160.0, poi_pa=psi_to_pa(300.0), pc_pa=psi_to_pa(150.0))
        assert lo is not None and hi is not None
        assert hi > lo

    def test_sqrt_scaling(self):
        base = lox_mass_flow_rate(-160.0, poi_pa=psi_to_pa(250.0), pc_pa=psi_to_pa(150.0))  # ΔP=100 psi
        quad = lox_mass_flow_rate(-160.0, poi_pa=psi_to_pa(550.0), pc_pa=psi_to_pa(150.0))  # ΔP=400 psi
        assert base is not None and quad is not None
        ratio = quad / base
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio ~2.0, got {ratio}"

    def test_matches_goondaq_formula(self):
        """
        Manually replicate the GoonDAQ lox_mdot() calculation and compare.
        Uses the density value our table returns, not a hardcoded constant.
        """
        toi_c = -160.0
        poi_pa, pc_pa = psi_to_pa(250.0), psi_to_pa(150.0)
        rho_lbm = lox_density_from_celsius(toi_c)
        assert rho_lbm is not None

        Cd = 0.6
        A_m2 = 0.02922466566 * 6.4516e-4
        rho_kg_m3 = rho_lbm * 16.0185
        dp_pa = poi_pa - pc_pa
        expected = Cd * A_m2 * math.sqrt(2.0 * rho_kg_m3 * dp_pa)

        result = lox_mass_flow_rate(toi_c, poi_pa, pc_pa)
        assert result is not None
        assert abs(result - expected) < 1e-6


# -- Fuel Mass Flow Rate --------------------------------------

class TestFuelMassFlowRate:

    def test_returns_positive_for_valid_inputs(self):
        result = fuel_mass_flow_rate(pfo_pa=psi_to_pa(200.0), pc_pa=psi_to_pa(150.0))
        assert result is not None
        assert result > 0.0

    def test_zero_dp_returns_none(self):
        result = fuel_mass_flow_rate(psi_to_pa(200.0), psi_to_pa(200.0))
        assert result is None

    def test_negative_dp_returns_none(self):
        result = fuel_mass_flow_rate(psi_to_pa(150.0), psi_to_pa(200.0))
        assert result is None

    def test_sqrt_scaling(self):
        """Mass flow ∝ sqrt(ΔP). Quadrupling ΔP should double the flow."""
        base = fuel_mass_flow_rate(psi_to_pa(250.0), psi_to_pa(150.0))   # ΔP = 100 psi
        quad = fuel_mass_flow_rate(psi_to_pa(550.0), psi_to_pa(150.0))   # ΔP = 400 psi
        assert base is not None and quad is not None
        ratio = quad / base
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio ~2.0, got {ratio}"

    def test_matches_goondaq_formula(self):
        pfo_pa, pc_pa = psi_to_pa(200.0), psi_to_pa(150.0)
        Cd = 0.67
        A_m2 = 0.04526 * 6.4516e-4
        rho = 800.0
        dp_pa = pfo_pa - pc_pa
        expected = Cd * A_m2 * math.sqrt(2.0 * rho * dp_pa)

        result = fuel_mass_flow_rate(pfo_pa, pc_pa)
        assert result is not None
        assert abs(result - expected) < 1e-6


# -- Oxidizer / Fuel Mixture Ratio ----------------------------
class TestMixtureRatio:

    def test_basic_ratio(self):
        result = mixture_ratio(2.0, 1.0)
        assert abs(result - 2.0) < 1e-9

    def test_stoichiometric_ipa_lox(self):
        # Stoichiometric O/F for LOX/IPA ≈ 2.4
        result = mixture_ratio(2.4, 1.0)
        assert abs(result - 2.4) < 1e-9

    def test_none_lox_returns_none(self):
        assert mixture_ratio(None, 1.0) is None

    def test_none_fuel_returns_none(self):
        assert mixture_ratio(2.0, None) is None

    def test_zero_fuel_returns_none(self):
        assert mixture_ratio(2.0, 0.0) is None

    def test_both_none_returns_none(self):
        assert mixture_ratio(None, None) is None


# -- Total Impulse (Load Cell Method) -------------------------

class TestImpulseLoadCell:

    def test_constant_force_is_force_times_time(self):
        # At constant 100 lbf over 1 s, trapezoidal rule is exact
        result = impulse_step_load_cell(100.0, 100.0, 1.0)
        assert abs(result - 100.0) < 1e-9

    def test_ramp_from_zero(self):
        # Force ramps linearly from 0 to 200 lbf over 1 s
        # Trapezoid: (0 + 200) / 2 * 1 = 100 lbf·s
        result = impulse_step_load_cell(0.0, 200.0, 1.0)
        assert abs(result - 100.0) < 1e-9

    def test_accumulation_gives_total(self):
        # Two equal steps at 100 lbf, dt = 0.5 s each -> total = 100 lbf·s
        step1 = impulse_step_load_cell(100.0, 100.0, 0.5)
        step2 = impulse_step_load_cell(100.0, 100.0, 0.5)
        assert abs(step1 + step2 - 100.0) < 1e-9

    def test_zero_dt_gives_zero(self):
        result = impulse_step_load_cell(500.0, 500.0, 0.0)
        assert result == 0.0


# -- Total Impulse (Mass Flow Method) -------------------------

class TestImpulseEstimate:

    def test_returns_positive_for_valid_inputs(self):
        result = impulse_step_estimate(0.5, 0.2, 0.002)
        assert result is not None
        assert result > 0.0

    def test_none_lox_returns_none(self):
        assert impulse_step_estimate(None, 0.2, 0.002) is None

    def test_none_fuel_returns_none(self):
        assert impulse_step_estimate(0.5, None, 0.002) is None

    def test_manual_calculation(self):
        """Verify against hand-computed value."""
        lox   = 0.5     # kg/s
        fuel  = 0.2     # kg/s
        dt    = 0.002   # s
        isp   = 220.0   # s
        g0    = 9.80665 # m/s²
        total_mdot = lox + fuel
        thrust = total_mdot * isp * g0
        expected = thrust * dt

        result = impulse_step_estimate(lox, fuel, dt, isp_seconds=isp)
        assert result is not None
        assert abs(result - expected) < 1e-9

    def test_higher_flow_gives_higher_impulse(self):
        lo = impulse_step_estimate(0.3, 0.12, 0.002)
        hi = impulse_step_estimate(0.6, 0.24, 0.002)
        assert lo is not None and hi is not None
        assert hi > lo


# -- LOX Saturation Pressure (Antoine Fit) --------------------

class TestLOXSaturationPressure:

    def test_normal_boiling_point(self):
        # LOX normal boiling point: -183 °C, 1 atm = 101325 Pa.
        # The Antoine equation is a curve fit; error at NBP is ~1.6%.
        # Should result in reasonable range & correct direction
        # for saturation check
        result = lox_saturation_pressure_pa(-183.0)
        assert result is not None
        assert psi_to_pa(13.0) < result < psi_to_pa(16.5), (
            f"At NBP expected ~101325 Pa (Antoine fit), got {result}"
        )

    def test_above_valid_range_returns_none(self):
        # Above critical point (~-119 °C / 154 K)
        result = lox_saturation_pressure_pa(-100.0)
        assert result is None

    def test_below_valid_range_returns_none(self):
        # Below triple point (~-219 °C / 54 K)
        result = lox_saturation_pressure_pa(-220.0)
        assert result is None

    def test_higher_temp_gives_higher_pressure(self):
        # Saturation pressure increases with temperature
        p_cold = lox_saturation_pressure_pa(-200.0)
        p_warm = lox_saturation_pressure_pa(-185.0)
        assert p_cold is not None and p_warm is not None
        assert p_warm > p_cold


# -- LOX Saturation Alert Checking ----------------------------

class TestLOXBelowSaturation:

    def test_tank_above_saturation_returns_false(self):
        # At NBP (-183 °C), p_sat ≈ 101325 Pa. Tank at 100 psia -> safe.
        result = lox_below_saturation(psi_to_pa(100.0), -183.0)
        assert result is False

    def test_tank_below_saturation_returns_true(self):
        # At NBP, p_sat ≈ 101325 Pa. Tank at 5 psia -> alert.
        result = lox_below_saturation(psi_to_pa(5.0), -183.0)
        assert result is True

    def test_temp_out_of_range_returns_none(self):
        result = lox_below_saturation(psi_to_pa(100.0), -220.0)
        assert result is None

    def test_exactly_at_saturation_is_not_below(self):
        # Tank pressure exactly equals saturation: not strictly below.
        p_sat = lox_saturation_pressure_pa(-183.0)
        assert p_sat is not None
        result = lox_below_saturation(p_sat, -183.0)
        assert result is False
