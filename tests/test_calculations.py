"""tests/test_calculations.py: unit tests for daq/calculations.py."""

import math
import sys
import os
import tempfile

# Importable from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from daq.calculations import (
    type_t_celsius_to_mv,
    type_t_mv_to_celsius,
    thermocouple_type_t,
    psi_to_pa,
    PSI_TO_PA,
    pt_voltage_to_psi,
    pt_voltage_to_pa,
    load_cell_voltage_to_force,
    load_lox_table,
    lox_density_from_celsius,
    lox_mass_flow_rate,
    fuel_mass_flow_rate,
    orifice_area_m2,
    mixture_ratio,
    impulse_step_load_cell,
    lox_saturation_pressure_pa,
    lox_below_saturation,
)

# -- Helper Utilities ----------------------------------------

def _write_lox_table(rows: list[tuple[float, float]]) -> str:
    """Write a minimal LOX density CSV and return its path."""
    content = "temperature_R,density_lbm_ft3\n"
    for t, d in rows:
        content += f"{t},{d}\n"
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    f.write(content)
    f.close()
    return f.name


# -- Type T Thermocouple (NIST ITS-90) -----------------------
# NIST ITS-90 type T table, mV at a 0 C reference:
# https://its90.nist.gov/downloadFiles/type_t.tab.txt

_NIST_TYPE_T_TABLE = [
    (-200.0, -5.603),
    (-183.0, -5.316),   # about LOX normal boiling point
    (0.0,     0.000),
    (25.0,    0.992),
    (100.0,   4.279),
    (200.0,   9.288),
    (400.0,  20.872),
]


class TestTypeT:
    """NIST ITS-90 Type T reference and inverse functions."""

    @pytest.mark.parametrize("temp_c, emf_mv", _NIST_TYPE_T_TABLE)
    def test_reference_function_matches_nist_table(self, temp_c, emf_mv):
        # The table is rounded to 1 µV.
        assert type_t_celsius_to_mv(temp_c) == pytest.approx(emf_mv, abs=0.0006)

    @pytest.mark.parametrize("temp_c, emf_mv", _NIST_TYPE_T_TABLE)
    def test_inverse_function_matches_nist_table(self, temp_c, emf_mv):
        # NIST inverse error <= 0.04 C, plus up to ~0.03 C from table rounding.
        assert type_t_mv_to_celsius(emf_mv) == pytest.approx(temp_c, abs=0.08)

    @pytest.mark.parametrize("emf_mv", [-5.7, 21.0])
    def test_inverse_refuses_emf_outside_its_range(self, emf_mv):
        assert type_t_mv_to_celsius(emf_mv) is None

    def test_zero_differential_reads_the_cold_junction(self):
        # Bench check: tc0 at room temperature reads room temperature.
        assert thermocouple_type_t(0.0, 22.0) == pytest.approx(22.0, abs=0.05)

    def test_lox_with_room_temperature_cold_junction(self):
        # Hot -183 C, cold 25 C: E(-183) - E(25) = -6.308 mV.
        diff_v = (-5.316 - 0.992) / 1e3
        assert thermocouple_type_t(diff_v, 25.0) == pytest.approx(-183.0, abs=0.1)

    def test_hot_junction_above_cold_junction(self):
        # Hot 100 C, cold 25 C: 4.279 - 0.992 = 3.287 mV.
        diff_v = (4.279 - 0.992) / 1e3
        assert thermocouple_type_t(diff_v, 25.0) == pytest.approx(100.0, abs=0.05)


# -- SI Unit Conversion Helpers --------------------------------

class TestUnitConversion:
    """psi <-> Pa (internal units are SI)."""

    def test_one_psi_in_pa(self):
        assert abs(psi_to_pa(1.0) - 6894.757293168361) < 1e-6

    def test_zero(self):
        assert psi_to_pa(0.0) == 0.0

    def test_one_atmosphere_sanity_check(self):
        # 14.696 psi is 1 atm (101325 Pa).
        assert abs(psi_to_pa(14.696) - 101325.0) < 1.0


# -- Pressure Transducers -------------------------------------

class TestPTConversion:
    """Linear PT calibration in psi."""

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
    """PT calibration returned in Pa."""

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
        # Tare at the current reading -> 0.
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
        # 100 R = 100 * 5/9 - 273.15 C
        t_r = 150.0    # midpoint of our table
        t_c = t_r * (5.0 / 9.0) - 273.15
        result = lox_density_from_celsius(t_c)
        # At 150 R in our linear table: density = 71 - 5 = 66 lbm/ft³
        assert result is not None
        assert abs(result - 66.0) < 0.05

    def test_below_range_returns_none(self):
        t_c = (90.0 * 5.0 / 9.0) - 273.15  # 90 R, below 100 R minimum
        result = lox_density_from_celsius(t_c)
        assert result is None

    def test_above_range_returns_none(self):
        t_c = (210.0 * 5.0 / 9.0) - 273.15  # 210 R, above 200 R maximum
        result = lox_density_from_celsius(t_c)
        assert result is None

    def test_interpolation_is_linear(self):
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
        # 180-240 R covers -160 C (203.67 R).
        rows = [(180.0 + i * 5.0, 70.0 - i * 0.2) for i in range(13)]
        path = _write_lox_table(rows)
        load_lox_table(path)
        yield
        os.unlink(path)

    # The functions take dP directly; geometry comes from config.
    LOX_AREA = orifice_area_m2(0.199)
    LOX_CD   = 0.6

    def _mdot(self, dp_psi, toi_c=-160.0):
        return lox_mass_flow_rate(toi_c, psi_to_pa(dp_psi), self.LOX_AREA, self.LOX_CD)

    def test_returns_positive_value_for_valid_inputs(self):
        result = self._mdot(100.0)
        assert result is not None
        assert result > 0.0

    def test_zero_dp_returns_none(self):
        assert self._mdot(0.0) is None

    def test_negative_dp_returns_none(self):
        assert self._mdot(-50.0) is None

    def test_missing_inlet_temperature_returns_none(self):
        """No LOX inlet temperature means no density, so no flow rate."""
        assert self._mdot(100.0, toi_c=None) is None

    def test_higher_dp_gives_higher_flow(self):
        lo = self._mdot(50.0)
        hi = self._mdot(150.0)
        assert lo is not None and hi is not None
        assert hi > lo

    def test_sqrt_scaling(self):
        base = self._mdot(100.0)
        quad = self._mdot(400.0)
        assert base is not None and quad is not None
        ratio = quad / base
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio ~2.0, got {ratio}"

    def test_matches_goondaq_formula(self):
        """Matches GoonDAQ's lox_mdot() at this table's density."""
        toi_c = -160.0
        dp_pa = psi_to_pa(100.0)
        rho_lbm = lox_density_from_celsius(toi_c)
        assert rho_lbm is not None

        expected = self.LOX_CD * self.LOX_AREA * math.sqrt(
            2.0 * rho_lbm * 16.0185 * dp_pa
        )

        result = lox_mass_flow_rate(toi_c, dp_pa, self.LOX_AREA, self.LOX_CD)
        assert result is not None
        assert abs(result - expected) < 1e-9


# -- Fuel Mass Flow Rate --------------------------------------

class TestFuelMassFlowRate:

    FUEL_AREA = orifice_area_m2(0.280)
    FUEL_CD   = 0.67
    FUEL_RHO  = 800.0

    def _mdot(self, dp_psi):
        return fuel_mass_flow_rate(
            psi_to_pa(dp_psi), self.FUEL_AREA, self.FUEL_CD, self.FUEL_RHO
        )

    def test_returns_positive_for_valid_inputs(self):
        result = self._mdot(50.0)
        assert result is not None
        assert result > 0.0

    def test_zero_dp_returns_none(self):
        assert self._mdot(0.0) is None

    def test_negative_dp_returns_none(self):
        assert self._mdot(-50.0) is None

    def test_sqrt_scaling(self):
        """Mass flow is proportional to sqrt(dP): quadrupling dP doubles flow."""
        base = self._mdot(100.0)
        quad = self._mdot(400.0)
        assert base is not None and quad is not None
        ratio = quad / base
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio ~2.0, got {ratio}"

    def test_matches_goondaq_formula(self):
        dp_pa = psi_to_pa(50.0)
        expected = self.FUEL_CD * self.FUEL_AREA * math.sqrt(
            2.0 * self.FUEL_RHO * dp_pa
        )
        result = self._mdot(50.0)
        assert result is not None
        assert abs(result - expected) < 1e-9


class TestOrificeArea:

    def test_area_from_diameter(self):
        assert abs(orifice_area_m2(0.199) - (math.pi / 4 * 0.199 ** 2) * 6.4516e-4) < 1e-15

    def test_count_scales_area_linearly(self):
        """An N-element injector has N times the flow area of one hole."""
        assert abs(orifice_area_m2(0.199, 4) - 4 * orifice_area_m2(0.199)) < 1e-15


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


# -- LOX Saturation Pressure (Antoine Fit) --------------------

class TestLOXSaturationPressure:

    def test_normal_boiling_point(self):
        # At -183 C the Antoine fit reads ~1.6 % below 101325 Pa.
        result = lox_saturation_pressure_pa(-183.0)
        assert result is not None
        assert psi_to_pa(13.0) < result < psi_to_pa(16.5), (
            f"At NBP expected ~101325 Pa (Antoine fit), got {result}"
        )

    def test_above_valid_range_returns_none(self):
        # Above the critical point (~-119 C / 154 K).
        result = lox_saturation_pressure_pa(-100.0)
        assert result is None

    def test_stops_at_nists_upper_bound(self):
        # NIST fits these constants over 54.36-154.33 K.
        assert lox_saturation_pressure_pa(154.33 - 273.15) is not None
        assert lox_saturation_pressure_pa(154.50 - 273.15) is None

    def test_below_valid_range_returns_none(self):
        # Below the triple point (~-219 C / 54 K).
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
        # At -183 C, p_sat ~ 1 atm; a 100 psia tank is above it.
        result = lox_below_saturation(psi_to_pa(100.0), -183.0)
        assert result is False

    def test_tank_below_saturation_returns_true(self):
        # A 5 psia tank is below it.
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
