"""
Tests for aeris.core.explain
"""

import pytest
import numpy as np
from aeris.core.explain import (
    DesignExplanation, explain_optimization, explain_pareto_front,
    compare_designs, generate_design_report
)
from aeris.core.sensitivity import SensitivityResult


class MockEvaluator:
    """Mock evaluator for testing."""
    def __init__(self):
        self.var_names = ["camber_m", "camber_pos_p", "thickness_t"]
        self.bounds = [(0.0, 0.06), (0.2, 0.6), (0.06, 0.18)]
    
    def evaluate(self, x):
        m, p, t = x[0], x[1], x[2]
        # Simple mock metrics
        cl = 2 * np.pi * (np.radians(6) - (-2 * np.pi * m))  # simplified
        cd = 0.01 + 0.02 * t
        return {
            "CL": float(cl),
            "CD": float(cd),
            "L_D": float(cl / cd) if cd > 0 else 0,
            "CL_max": float(1.2 + 4 * t),
            "stall_AoA_deg": float(10 + 20 * t),
            "best_LD": float(cl / cd) if cd > 0 else 0,
            "best_LD_alpha": 6.0,
            "CD0": float(cd),
            "_valid": True,
            "_var_names": self.var_names,
        }
    
    def n_vars(self):
        return len(self.bounds)


class TestDesignExplanation:
    def test_to_text(self):
        exp = DesignExplanation(
            design_name="Test Design",
            design_variables={"x1": 1.0, "x2": 2.0},
            objectives={"L_D": 50.0, "CL": 1.0},
            key_changes=["x1 increased for better L/D"],
            tradeoffs=["Higher CL increases CD"],
            constraint_status={"positive_CD": True, "min_CL": True},
        )
        
        text = exp.to_text()
        assert "DESIGN EXPLANATION: Test Design" in text
        assert "x1" in text
        assert "x2" in text
        assert "50.0" in text
        assert "x1 increased for better L/D" in text
        assert "Higher CL increases CD" in text
        assert "SATISFIED" in text


class TestExplainOptimization:
    def test_explain_optimization_basic(self):
        evaluator = MockEvaluator()
        optimal = np.array([0.04, 0.4, 0.12])
        
        def obj(metrics):
            return metrics["L_D"]
        
        explanation = explain_optimization(evaluator, optimal, obj)
        
        assert isinstance(explanation, DesignExplanation)
        assert explanation.design_name == "Optimized Design"
        assert "camber_m" in explanation.design_variables
        assert "L_D" in explanation.objectives
        assert explanation.sensitivity is not None
    
    def test_explain_optimization_with_baseline(self):
        evaluator = MockEvaluator()
        optimal = np.array([0.04, 0.4, 0.12])
        baseline = np.array([0.02, 0.4, 0.12])
        
        def obj(metrics):
            return metrics["L_D"]
        
        explanation = explain_optimization(evaluator, optimal, obj, baseline_design=baseline)
        
        assert explanation.baseline_comparison is not None
        assert "L_D" in explanation.baseline_comparison
        assert "improvement_pct" in explanation.baseline_comparison["L_D"]
    
    def test_explain_optimization_with_constraints(self):
        from aeris.core.constraints import create_airfoil_constraints
        
        evaluator = MockEvaluator()
        optimal = np.array([0.04, 0.4, 0.12])
        
        def obj(metrics):
            return metrics["L_D"]
        
        constraint_set = create_airfoil_constraints()
        explanation = explain_optimization(evaluator, optimal, obj, constraint_set=constraint_set)
        
        assert explanation.constraint_status is not None
        assert "positive_CD" in explanation.constraint_status
        assert "min_CL" in explanation.constraint_status


class TestExplainParetoFront:
    def test_explain_pareto_front(self):
        evaluator = MockEvaluator()
        
        pareto_x = np.array([
            [0.04, 0.4, 0.12],
            [0.05, 0.5, 0.10],
            [0.03, 0.3, 0.14],
        ])
        pareto_metrics = [evaluator.evaluate(x) for x in pareto_x]
        pareto_scores = np.array([[50.0], [45.0], [48.0]])
        
        def obj(metrics):
            return metrics["L_D"]
        
        explanations = explain_pareto_front(
            evaluator, pareto_x, pareto_metrics, pareto_scores,
            ["L_D"], [obj]
        )
        
        assert len(explanations) == 3
        assert explanations[0].design_name == "Pareto Rank 1"
        assert explanations[1].design_name == "Pareto Rank 2"
        assert explanations[2].design_name == "Pareto Rank 3"


class TestCompareDesigns:
    def test_compare_designs(self):
        evaluator = MockEvaluator()
        design_a = np.array([0.04, 0.4, 0.12])
        design_b = np.array([0.02, 0.4, 0.12])
        
        def obj(metrics):
            return metrics["L_D"]
        
        comparison = compare_designs(evaluator, design_a, design_b, "A", "B", obj)
        
        assert "DESIGN COMPARISON: A vs B" in comparison
        assert "camber_m" in comparison
        assert "L_D" in comparison


class TestGenerateDesignReport:
    def test_generate_design_report_single(self):
        evaluator = MockEvaluator()
        optimal = np.array([0.04, 0.4, 0.12])
        
        def obj(metrics):
            return metrics["L_D"]
        
        report = generate_design_report(
            evaluator, optimal, 
            objective_fns=[obj],
            baseline_design=np.array([0.02, 0.4, 0.12]),
        )
        
        assert "AERIS DESIGN REPORT" in report
        assert "DESIGN EXPLANATION" in report
        assert "SENSITIVITIES" in report


class TestSensitivityResult:
    def test_summary_format(self):
        result = SensitivityResult(
            parameter_names=["x1", "x2"],
            local_sensitivity={"x1": 0.25, "x2": 0.75},
            morris_mu=None,
            morris_sigma=None,
            sobol_first=None,
            sobol_total=None,
            base_point=np.array([1.0, 2.0]),
            base_metrics={"f": 8.0},
        )
        
        summary = result.summary()
        assert "SENSITIVITY ANALYSIS" in summary
        assert "x1" in summary
        assert "x2" in summary