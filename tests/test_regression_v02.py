"""
Regression tests for AERIS v0.2 compatibility.

These tests capture the expected outputs from v0.2 and verify that
v0.3 produces the same results (within 0.1% tolerance).

Run with: python -m pytest tests/test_regression_v02.py -v
"""

import pytest
from aeris.modules.airfoil_module import AirfoilEvaluator
from aeris.modules.winglet_module import WingletEvaluator
from aeris.modules.aircraft_module import AircraftEvaluator
from aeris.core.optimizer import optimize
from aeris.core.units import as_float


# Expected v0.2 metrics captured from actual v0.2 runs
# These are the reference values that v0.3 must match within 0.1%

EXPECTED_AIRFOIL = {
    "CL": 1.5107028418353972,
    "CD": 0.008570501062168945,
    "Cm_c4": -0.22466613687013515,
    "CL_max": 1.5107028418353972,
    "stall_AoA_deg": 12.153514209176985,
    "L_D": 176.26773870944277,
    "Reynolds": 121824.42424216852,
}

EXPECTED_WINGLET = {
    "AR_effective": 11.799999999891524,
    "CDi_with_winglet": 0.010066789998993892,
    "CDi_baseline": 0.014848515248379492,
    "CD_total": 0.03024678999899389,
    "L_D": 18.183747763590613,
    "L_D_baseline": 15.782594927787514,
    "L_D_improvement_pct": 15.213929311304359,
}

EXPECTED_AIRCRAFT = {
    "CL": 0.673980126830399,
    "CD": 0.021119916793062514,
    "CD0": 0.01040995619102927,
    "CDi": 0.010559960602033243,
    "L_D": 31.912063548081235,
    "AR": 14.215573355266619,
    "AR_effective": 20.967970699018263,
    "area_m2": 0.22791905180522581,
    "tip_chord_m": 0.05844078251416046,
    "tip_stall_margin": 0.4427001131856223,
    "stability_score": 1.1992204587165327,
    "static_margin": 0.7265224857778869,
    "static_margin_pct": 72.6522485777887,
    "neutral_point_c_root": 0.8030083360160324,
    "cg_position_c_root": 0.2851282051282052,
    "cm_ac": -0.15425085969898594,
    "cm_cg": 0.0,
    "trim_alpha_deg": 2.7160709352143857,
    "trim_cl": 0.29785050728736434,
    "is_stable": True,
}

TOLERANCE = 0.001  # 0.1%


def compare_metrics(actual: dict, expected: dict, prefix: str = "") -> list:
    """Compare actual metrics to expected, return list of mismatches."""
    mismatches = []
    for key, expected_val in expected.items():
        if key not in actual:
            mismatches.append(f"{prefix}{key}: MISSING (expected {expected_val})")
            continue
        actual_val = actual[key]
        if isinstance(actual_val, dict):
            # Skip complex nested structures
            continue
        try:
            actual_f = as_float(actual_val)
            expected_f = as_float(expected_val)
            if expected_f == 0:
                rel_diff = abs(actual_f - expected_f)
            else:
                rel_diff = abs(actual_f - expected_f) / abs(expected_f)
            if rel_diff > TOLERANCE:
                mismatches.append(
                    f"{prefix}{key}: actual={actual_f:.8f}, expected={expected_f:.8f}, "
                    f"rel_diff={rel_diff:.6f} (>{TOLERANCE})"
                )
        except Exception as e:
            mismatches.append(f"{prefix}{key}: comparison error - {e}")
    return mismatches


class TestAirfoilRegression:
    def test_airfoil_optimization_matches_v02(self):
        """Run airfoil optimization with v0.2 seed and compare metrics."""
        evaluator = AirfoilEvaluator(
            alpha_deg=6.0, velocity_mps=12.0, chord_m=0.15, altitude_m=0.0,
            use_biomimicry=True,
        )
        result = optimize(evaluator, objective_fn=lambda m: m["L_D"],
                           maximize=True, popsize=20, maxiter=60, seed=42)
        
        mismatches = compare_metrics(result.metrics, EXPECTED_AIRFOIL, "airfoil.")
        if mismatches:
            pytest.fail("Airfoil regression mismatches:\n" + "\n".join(mismatches))
    
    def test_airfoil_design_variables_match(self):
        """Verify design variables are in expected ranges."""
        evaluator = AirfoilEvaluator(
            alpha_deg=6.0, velocity_mps=12.0, chord_m=0.15, altitude_m=0.0,
            use_biomimicry=True,
        )
        result = optimize(evaluator, objective_fn=lambda m: m["L_D"],
                           maximize=True, popsize=20, maxiter=60, seed=42)
        
        x = result.x
        # Check bounds
        assert 0.0 <= x[0] <= 0.06   # camber_m
        assert 0.2 <= x[1] <= 0.6    # camber_pos_p
        assert 0.06 <= x[2] <= 0.18  # thickness_t
        assert 0.0 <= x[3] <= 0.05   # tubercle_amplitude
        assert 0.1 <= x[4] <= 0.5    # tubercle_wavelength


class TestWingletRegression:
    def test_winglet_optimization_matches_v02(self):
        """Run winglet optimization with v0.2 seed and compare metrics."""
        evaluator = WingletEvaluator(
            base_span_m=1.2, base_area_m2=0.18, operating_CL=0.55, base_CD0=0.02,
        )
        result = optimize(evaluator, objective_fn=lambda m: m["L_D"],
                           maximize=True, popsize=20, maxiter=60, seed=1)
        
        mismatches = compare_metrics(result.metrics, EXPECTED_WINGLET, "winglet.")
        if mismatches:
            pytest.fail("Winglet regression mismatches:\n" + "\n".join(mismatches))
    
    def test_winglet_design_variables_match(self):
        """Verify design variables are in expected ranges."""
        evaluator = WingletEvaluator(
            base_span_m=1.2, base_area_m2=0.18, operating_CL=0.55, base_CD0=0.02,
        )
        result = optimize(evaluator, objective_fn=lambda m: m["L_D"],
                           maximize=True, popsize=20, maxiter=60, seed=1)
        
        x = result.x
        assert 0.0 <= x[0] <= 0.25   # height_ratio
        assert 0.0 <= x[1] <= 75.0   # cant_deg
        assert 0.2 <= x[2] <= 1.0    # taper_ratio


class TestAircraftRegression:
    def test_aircraft_optimization_matches_v02(self):
        """Run aircraft optimization with v0.2 seed and compare metrics."""
        evaluator = AircraftEvaluator(
            cruise_velocity_mps=10.0, altitude_m=0.0, target_CL=0.6,
        )
        result = optimize(evaluator, objective_fn=lambda m: m["L_D"],
                           maximize=True, popsize=30, maxiter=150, seed=7)
        
        mismatches = compare_metrics(result.metrics, EXPECTED_AIRCRAFT, "aircraft.")
        if mismatches:
            pytest.fail("Aircraft regression mismatches:\n" + "\n".join(mismatches))
    
    def test_aircraft_design_variables_match(self):
        """Verify design variables are in expected ranges."""
        evaluator = AircraftEvaluator(
            cruise_velocity_mps=10.0, altitude_m=0.0, target_CL=0.6,
        )
        result = optimize(evaluator, objective_fn=lambda m: m["L_D"],
                           maximize=True, popsize=30, maxiter=150, seed=7)
        
        x = result.x
        assert 0.8 <= x[0] <= 1.8      # span_m
        assert 0.15 <= x[1] <= 0.45    # root_chord_m
        assert 0.3 <= x[2] <= 1.0      # taper_ratio
        assert 0.0 <= x[3] <= 35.0     # sweep_deg
        assert 0.0 <= x[4] <= 10.0     # washout_deg
        assert 0.0 <= x[5] <= 0.06     # root_camber_m
        assert 0.0 <= x[6] <= 0.06     # tip_camber_m
        assert 0.08 <= x[7] <= 0.18    # thickness_t
        assert 0.25 <= x[8] <= 0.5     # camber_pos_p
        assert 0.0 <= x[9] <= 0.25     # winglet_height_ratio
        assert 0.0 <= x[10] <= 60.0    # winglet_cant_deg
        assert 0.0 <= x[11] <= 10.0    # alpha_deg


def test_all_regression():
    """Run all three regressions in sequence (for manual verification)."""
    test_airfoil = TestAirfoilRegression()
    test_airfoil.test_airfoil_optimization_matches_v02()
    test_airfoil.test_airfoil_design_variables_match()
    print("✓ Airfoil regression passed")
    
    test_winglet = TestWingletRegression()
    test_winglet.test_winglet_optimization_matches_v02()
    test_winglet.test_winglet_design_variables_match()
    print("✓ Winglet regression passed")
    
    test_aircraft = TestAircraftRegression()
    test_aircraft.test_aircraft_optimization_matches_v02()
    test_aircraft.test_aircraft_design_variables_match()
    print("✓ Aircraft regression passed")
    
    print("\nAll regression tests passed!")


if __name__ == "__main__":
    test_all_regression()