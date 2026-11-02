"""
Tests for airfoil polar generation and baseline comparison (Phase 2).
"""

import pytest
import numpy as np
from aeris.core import knowledge_base as kb
from aeris.modules.airfoil_module import AirfoilEvaluator


class TestPolarGeneration:
    def test_polar_cl_alpha_basic(self):
        """Test basic CL-alpha polar generation."""
        polar = kb.polar_cl_alpha(0.02, 0.4, 0.12, n_points=20)
        
        assert "alpha_deg" in polar
        assert "CL" in polar
        assert "CL_max" in polar
        assert "stall_alpha_deg" in polar
        assert "alpha_L0_deg" in polar
        assert len(polar["alpha_deg"]) == 20
        assert len(polar["CL"]) == 20
    
    def test_polar_cl_alpha_range(self):
        """Test polar with custom alpha range."""
        polar = kb.polar_cl_alpha(0.02, 0.4, 0.12, alpha_min=-10, alpha_max=20, n_points=31)
        
        assert polar["alpha_deg"][0] == -10.0
        assert polar["alpha_deg"][-1] == 20.0
        assert len(polar["alpha_deg"]) == 31
    
    def test_polar_cl_cd_cm_comprehensive(self):
        """Test comprehensive polar with CL, CD, Cm, L/D."""
        polar = kb.polar_cl_cd_cm(0.02, 0.4, 0.12, Re=1e6, n_points=20)
        
        required_keys = ["alpha_deg", "CL", "CD", "Cm_c4", "LD", 
                         "CL_max", "stall_alpha_deg", "alpha_L0_deg",
                         "best_LD", "best_LD_alpha", "CD0"]
        for key in required_keys:
            assert key in polar, f"Missing key: {key}"
        
        assert len(polar["alpha_deg"]) == 20
        assert len(polar["CL"]) == 20
        assert len(polar["CD"]) == 20
        assert len(polar["Cm_c4"]) == 20
        assert len(polar["LD"]) == 20
        
        # Cm_c4 should be constant (thin airfoil theory)
        cm_vals = polar["Cm_c4"]
        assert all(abs(c - cm_vals[0]) < 1e-10 for c in cm_vals)
        
        # CL_max should match max CL pre-stall
        prestall_CL = [cl for a, cl in zip(polar["alpha_deg"], polar["CL"]) 
                       if a <= polar["stall_alpha_deg"]]
        assert abs(max(prestall_CL) - polar["CL_max"]) < 1e-6
        
        # best_LD should be pre-stall
        assert polar["best_LD_alpha"] <= polar["stall_alpha_deg"]
    
    def test_polar_stall_behavior(self):
        """Test that polar correctly models stall."""
        polar = kb.polar_cl_cd_cm(0.02, 0.4, 0.12, Re=1e6,
                                  alpha_min=-5, alpha_max=25, n_points=31)
        
        # Find stall index (closest alpha to stall_alpha)
        stall_alpha = polar["stall_alpha_deg"]
        stall_idx = min(range(len(polar["alpha_deg"])), 
                        key=lambda i: abs(polar["alpha_deg"][i] - stall_alpha))
        
        # Post-stall CL should be 60% of CL_max
        if stall_idx + 1 < len(polar["CL"]):
            post_stall_CL = polar["CL"][stall_idx + 1]
            assert abs(post_stall_CL - 0.6 * polar["CL_max"]) < 1e-6
        
        # Post-stall CD should be 3x CD0
        if stall_idx + 1 < len(polar["CD"]):
            post_stall_CD = polar["CD"][stall_idx + 1]
            assert abs(post_stall_CD - 3.0 * polar["CD0"]) < 1e-6


class TestBaselineComparison:
    def test_baseline_airfoil_creation(self):
        """Test baseline airfoil has expected values."""
        baseline = kb.create_baseline_airfoil()
        
        assert baseline["camber_m"] == 0.02
        assert baseline["camber_pos_p"] == 0.4
        assert baseline["thickness_t"] == 0.12
    
    def test_compare_airfoil_to_baseline_LD(self):
        """Test L/D comparison with baseline."""
        polar_opt = {
            "best_LD": 100.0,
            "CL_max": 1.5,
            "CD0": 0.01,
            "Cm_c4": [-0.1],
        }
        polar_base = {
            "best_LD": 80.0,
            "CL_max": 1.2,
            "CD0": 0.012,
            "Cm_c4": [-0.05],
        }
        
        comp = kb.compare_airfoil_to_baseline(polar_opt, polar_base, "LD")
        assert comp["opt_value"] == 100.0
        assert comp["base_value"] == 80.0
        assert comp["improvement"] == 20.0
        assert abs(comp["improvement_pct"] - 25.0) < 1e-6
    
    def test_compare_airfoil_to_baseline_CL(self):
        """Test CL_max comparison with baseline."""
        polar_opt = {"CL_max": 1.6}
        polar_base = {"CL_max": 1.4}
        
        comp = kb.compare_airfoil_to_baseline(polar_opt, polar_base, "CL")
        assert abs(comp["improvement"] - 0.2) < 1e-10
        assert abs(comp["improvement_pct"] - (0.2/1.4*100)) < 1e-6
    
    def test_compare_airfoil_to_baseline_CD(self):
        """Test CD0 comparison with baseline (lower is better)."""
        polar_opt = {"CD0": 0.008}
        polar_base = {"CD0": 0.01}
        
        comp = kb.compare_airfoil_to_baseline(polar_opt, polar_base, "CD")
        assert comp["improvement"] == -0.002
        # Negative improvement means opt is better (lower CD)
        assert comp["improvement_pct"] < 0
    
    def test_compare_airfoil_to_baseline_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError):
            kb.compare_airfoil_to_baseline({}, {}, "invalid_metric")


class TestAirfoilEvaluatorPolar:
    def test_evaluate_returns_polar_when_requested(self):
        """Test that evaluate(return_polar=True) returns polar data."""
        evaluator = AirfoilEvaluator(alpha_deg=6.0, velocity_mps=12.0, 
                                      chord_m=0.15, altitude_m=0.0,
                                      use_biomimicry=False)
        
        metrics = evaluator.evaluate([0.02, 0.4, 0.12], return_polar=True)
        
        assert "polar" in metrics
        assert "baseline_polar" in metrics
        
        polar = metrics["polar"]
        assert "alpha_deg" in polar
        assert "CL" in polar
        assert "CD" in polar
        assert "LD" in polar
        assert "best_LD" in polar
        assert "best_LD_alpha" in polar
        
        baseline = metrics["baseline_polar"]
        assert "alpha_deg" in baseline
        assert "CL" in baseline
    
    def test_evaluate_includes_baseline_comparison(self):
        """Test that metrics include baseline comparison."""
        evaluator = AirfoilEvaluator(alpha_deg=6.0, velocity_mps=12.0,
                                      chord_m=0.15, altitude_m=0.0,
                                      use_biomimicry=False)
        
        metrics = evaluator.evaluate([0.02, 0.4, 0.12])
        
        # Check baseline comparison metrics
        assert "baseline_LD_improvement_pct" in metrics
        assert "baseline_CL_improvement_pct" in metrics
        assert "baseline_CD_improvement_pct" in metrics
        assert "baseline_Cm_improvement_pct" in metrics
    
    def test_evaluate_includes_polar_metrics(self):
        """Test that metrics include extracted polar metrics."""
        evaluator = AirfoilEvaluator(alpha_deg=6.0, velocity_mps=12.0,
                                      chord_m=0.15, altitude_m=0.0,
                                      use_biomimicry=False)
        
        metrics = evaluator.evaluate([0.02, 0.4, 0.12])
        
        assert "alpha_L0_deg" in metrics
        assert "best_LD" in metrics
        assert "best_LD_alpha" in metrics
        assert "CD0" in metrics
        assert "polar_source" in metrics
    
    def test_evaluate_matches_direct_computation(self):
        """Test that evaluate() operating point matches direct thin airfoil theory."""
        evaluator = AirfoilEvaluator(alpha_deg=6.0, velocity_mps=12.0,
                                      chord_m=0.15, altitude_m=0.0,
                                      use_biomimicry=False)
        
        metrics = evaluator.evaluate([0.02, 0.4, 0.12])
        
        # Direct computation
        from aeris.core import knowledge_base as kb
        coeffs = kb.thin_airfoil_coeffs(0.02, 0.4)
        alpha_rad = np.radians(6.0)
        cl_direct = kb.thin_airfoil_cl(alpha_rad, coeffs["alpha_L0_rad"])
        cd0 = kb.profile_drag_coefficient(
            kb.reynolds_number(1.225, 12.0, 0.15, 1.81e-5), 0.12)
        cm_direct = kb.thin_airfoil_cm_quarter_chord(coeffs["A1"], coeffs["A2"])
        
        assert abs(metrics["CL"] - cl_direct) < 1e-6
        assert abs(metrics["CD"] - cd0) < 1e-6
        assert abs(metrics["Cm_c4"] - cm_direct) < 1e-6


class TestAirfoilEvaluatorBiomimicry:
    def test_biomimicry_in_polar(self):
        """Test that biomimicry effects are included in polar."""
        evaluator = AirfoilEvaluator(alpha_deg=6.0, velocity_mps=12.0,
                                      chord_m=0.15, altitude_m=0.0,
                                      use_biomimicry=True)
        
        # Without biomimicry
        metrics_no_bio = evaluator.evaluate([0.02, 0.4, 0.12, 0.0, 0.25])
        
        # With biomimicry
        metrics_bio = evaluator.evaluate([0.02, 0.4, 0.12, 0.03, 0.25])
        
        # Biomimicry should increase CL_max and stall alpha
        assert metrics_bio["CL_max"] > metrics_no_bio["CL_max"]
        assert metrics_bio["stall_AoA_deg"] > metrics_no_bio["stall_AoA_deg"]
        # And increase CD0
        assert metrics_bio["CD0"] > metrics_no_bio["CD0"]
    
    def test_biomimicry_in_baseline_comparison(self):
        """Test that baseline comparison works with biomimicry."""
        evaluator = AirfoilEvaluator(alpha_deg=6.0, velocity_mps=12.0,
                                      chord_m=0.15, altitude_m=0.0,
                                      use_biomimicry=True)
        
        metrics = evaluator.evaluate([0.02, 0.4, 0.12, 0.03, 0.25])
        
        # Should have baseline comparison metrics
        assert "baseline_LD_improvement_pct" in metrics
        assert "baseline_CL_improvement_pct" in metrics


class TestPolarEdgeCases:
    def test_polar_symmetric_airfoil(self):
        """Test polar for symmetric airfoil (m=0)."""
        polar = kb.polar_cl_cd_cm(0.0, 0.0, 0.12, Re=1e6)
        
        # Symmetric airfoil should have alpha_L0 = 0
        assert abs(polar["alpha_L0_deg"]) < 1e-6
        
        # CL should be anti-symmetric (check a few points)
        cl_vals = polar["CL"]
        alpha_vals = polar["alpha_deg"]
        for i, a in enumerate(alpha_vals):
            if a != 0:
                # Find index of -a
                neg_idx = min(range(len(alpha_vals)), key=lambda j: abs(alpha_vals[j] + a))
                if abs(alpha_vals[neg_idx] + a) < 1e-6:  # Found exact match
                    assert abs(cl_vals[i] + cl_vals[neg_idx]) < 1e-6
    
    def test_polar_high_camber(self):
        """Test polar for high camber airfoil."""
        polar = kb.polar_cl_cd_cm(0.06, 0.4, 0.12, Re=1e6)
        
        # High camber should have negative alpha_L0 (more negative than low camber)
        low_camber_polar = kb.polar_cl_cd_cm(0.02, 0.4, 0.12, Re=1e6)
        assert polar["alpha_L0_deg"] < low_camber_polar["alpha_L0_deg"]
        
        # Should have higher CL_max than low camber
        assert polar["CL_max"] > low_camber_polar["CL_max"]
    
    def test_polar_low_reynolds(self):
        """Test polar at low Reynolds number."""
        polar = kb.polar_cl_cd_cm(0.02, 0.4, 0.12, Re=5e4)
        
        # CD0 should be higher at low Re (laminar)
        cd0_laminar = kb.profile_drag_coefficient(5e4, 0.12)
        assert abs(polar["CD0"] - cd0_laminar) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])