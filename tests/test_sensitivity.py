"""
Tests for aeris.core.sensitivity
"""

import pytest
import numpy as np
from aeris.core.sensitivity import (
    local_sensitivity, morris_sensitivity, sobol_sensitivity,
    analyze_sensitivity, SensitivityResult
)


class TestLocalSensitivity:
    def test_linear_function(self):
        """Sensitivity of f(x) = 2x1 + 3x2 should be [2, 3]."""
        def f(x):
            return 2 * x[0] + 3 * x[1]
        
        base = np.array([1.0, 2.0])
        bounds = [(0, 10), (0, 10)]
        sens = local_sensitivity(f, base, bounds)
        
        # Relative sensitivity: (df/dx) * x / f
        # f = 2*1 + 3*2 = 8
        # df/dx1 = 2, rel = 2 * 1 / 8 = 0.25
        # df/dx2 = 3, rel = 3 * 2 / 8 = 0.75
        assert abs(sens[0] - 0.25) < 0.01
        assert abs(sens[1] - 0.75) < 0.01
    
    def test_quadratic_function(self):
        """Sensitivity of f(x) = x^2 at x=2 should be 2x * x/f = 2."""
        def f(x):
            return x[0] ** 2
        
        base = np.array([2.0])
        bounds = [(0, 10)]
        sens = local_sensitivity(f, base, bounds)
        
        # f = 4, df/dx = 4, rel = 4 * 2 / 4 = 2
        assert abs(sens[0] - 2.0) < 0.01
    
    def test_zero_at_optimum(self):
        """Sensitivity at minimum of quadratic should be zero."""
        def f(x):
            return (x[0] - 5) ** 2
        
        base = np.array([5.0])
        bounds = [(0, 10)]
        sens = local_sensitivity(f, base, bounds)
        
        # At minimum, derivative is 0
        assert abs(sens[0]) < 1e-6


class TestMorrisSensitivity:
    def test_linear_function(self):
        """Morris mu* should reflect coefficients for linear function."""
        def f(x):
            return 2 * x[0] + 3 * x[1]
        
        bounds = [(0, 10), (0, 10)]
        mu, sigma = morris_sensitivity(f, bounds, n_trajectories=20, n_levels=4)
        
        # x1 coefficient 2, x2 coefficient 3 -> mu* should reflect this
        assert mu[1] > mu[0]  # x2 has larger coefficient
    
    def test_quadratic_function(self):
        """Quadratic should have non-zero sigma (non-linear)."""
        def f(x):
            return x[0] ** 2
        
        bounds = [(0, 5)]
        mu, sigma = morris_sensitivity(f, bounds, n_trajectories=10, n_levels=4)
        
        # Should have some non-linearity
        assert sigma[0] > 0


class TestSobolSensitivity:
    def test_linear_function(self):
        """Sobol indices for linear function: S1 should match coefficient ratios."""
        def f(x):
            return 2 * x[0] + 3 * x[1]
        
        bounds = [(0, 1), (0, 1)]  # Uniform [0,1] for variance calculation
        s1, st = sobol_sensitivity(f, bounds, n_samples=5000)
        
        # Var(f) = 2^2*Var(x1) + 3^2*Var(x2) = 4/12 + 9/12 = 13/12
        # S1 = 4/12 / (13/12) = 4/13 ≈ 0.308
        # S2 = 9/12 / (13/12) = 9/13 ≈ 0.692
        # Using looser tolerance due to Monte Carlo estimation noise
        assert abs(s1[0] - 4/13) < 0.2
        assert abs(s1[1] - 9/13) < 0.2
        # For linear additive, total == first
        assert abs(st[0] - s1[0]) < 0.2
        assert abs(st[1] - s1[1]) < 0.2
    
    def test_interaction_function(self):
        """Function with interaction: f = x1 * x2."""
        def f(x):
            return x[0] * x[1]
        
        bounds = [(0, 1), (0, 1)]
        s1, st = sobol_sensitivity(f, bounds, n_samples=2000)
        
        # For f = x1*x2 with uniform [0,1]:
        # First order should be smaller than total order due to interaction
        assert s1[0] < st[0]
        assert s1[1] < st[1]
        # Total order should capture interaction
        assert st[0] > 0.2
        assert st[1] > 0.2


class TestAnalyzeSensitivity:
    class MockEvaluator:
        def __init__(self):
            self.bounds = [(0, 10), (0, 10)]
            self.var_names = ["x1", "x2"]
        
        def evaluate(self, x):
            return {"f": 2*x[0] + 3*x[1], "_valid": True, "_var_names": ["x1", "x2"]}
    
    def test_analyze_sensitivity_local(self):
        evaluator = self.MockEvaluator()
        base = np.array([1.0, 2.0])
        
        def obj(metrics):
            return metrics["f"]
        
        result = analyze_sensitivity(evaluator, base, obj, method="local")
        
        assert isinstance(result, SensitivityResult)
        assert result.local_sensitivity is not None
        assert "x1" in result.local_sensitivity
        assert "x2" in result.local_sensitivity
    
    def test_analyze_sensitivity_morris(self):
        evaluator = self.MockEvaluator()
        base = np.array([1.0, 2.0])
        
        def obj(metrics):
            return metrics["f"]
        
        result = analyze_sensitivity(evaluator, base, obj, method="morris")
        
        assert result.morris_mu is not None
        assert result.morris_sigma is not None
    
    def test_analyze_sensitivity_summary(self):
        evaluator = self.MockEvaluator()
        base = np.array([1.0, 2.0])
        
        def obj(metrics):
            return metrics["f"]
        
        result = analyze_sensitivity(evaluator, base, obj, method="local")
        summary = result.summary()
        
        assert "SENSITIVITY ANALYSIS" in summary
        assert "x1" in summary
        assert "x2" in summary


class TestSensitivityResult:
    def test_summary_format(self):
        result = SensitivityResult(
            parameter_names=["x1", "x2"],
            local_sensitivity={"x1": 0.25, "x2": 0.75},
            morris_mu={"x1": 1.5, "x2": 2.5},
            morris_sigma={"x1": 0.1, "x2": 0.2},
            sobol_first={"x1": 0.3, "x2": 0.7},
            sobol_total={"x1": 0.3, "x2": 0.7},
            base_point=np.array([1.0, 2.0]),
            base_metrics={"f": 8.0},
        )
        
        summary = result.summary()
        assert "SENSITIVITY ANALYSIS" in summary
        assert "x1" in summary
        assert "x2" in summary