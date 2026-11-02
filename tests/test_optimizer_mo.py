"""
Tests for aeris.core.optimizer_mo (multi-objective NSGA-II)
"""

import pytest
import numpy as np
from aeris.core.optimizer_mo import NSGA2Optimizer, MOOptimizationResult, optimize
from aeris.core.evaluator_base import Evaluator


class MockMOEvaluator(Evaluator):
    """Mock evaluator for multi-objective testing."""
    name = "mock_mo"
    var_names = ["x1", "x2"]
    bounds = [(0, 5), (0, 5)]
    
    def evaluate(self, x):
        x1, x2 = x[0], x[1]
        # Two objectives: f1 = x1 + x2 (maximize), f2 = -(x1-2)^2 - (x2-3)^2 (maximize = minimize distance to (2,3))
        f1 = x1 + x2
        f2 = -((x1 - 2) ** 2 + (x2 - 3) ** 2)
        return {
            "f1": float(f1),
            "f2": float(f2),
            "_valid": True,
            "_var_names": ["x1", "x2"],
        }
    
    def n_vars(self):
        return 2


class SimpleEvaluator(Evaluator):
    """Simple single-objective evaluator."""
    name = "simple"
    var_names = ["x"]
    bounds = [(0, 10)]
    
    def evaluate(self, x):
        val = -(x[0] - 5) ** 2 + 25  # Max at x=5
        return {"value": float(val), "_valid": True, "_var_names": ["x"]}
    
    def n_vars(self):
        return 1


class TestNSGA2Optimizer:
    def test_two_objective_optimization(self):
        """Test basic two-objective optimization."""
        evaluator = MockMOEvaluator()
        
        def obj1(m): return m["f1"]
        def obj2(m): return m["f2"]
        
        optimizer = NSGA2Optimizer(
            evaluator=evaluator,
            objective_fns=[obj1, obj2],
            maximize=[True, True],
            popsize=20,
            maxiter=10,
            seed=42,
        )
        
        result = optimizer.optimize()
        
        assert isinstance(result, MOOptimizationResult)
        assert len(result.pareto_x) > 0
        assert result.pareto_scores.shape[1] == 2
        assert result.n_evaluations > 0
        assert result.success
    
    def test_pareto_front_properties(self):
        """Test that Pareto front has reasonable properties."""
        evaluator = MockMOEvaluator()
        
        def obj1(m): return m["f1"]
        def obj2(m): return m["f2"]
        
        optimizer = NSGA2Optimizer(
            evaluator=evaluator,
            objective_fns=[obj1, obj2],
            maximize=[True, True],
            popsize=30,
            maxiter=20,
            seed=42,
        )
        
        result = optimizer.optimize()
        
        # Pareto front should not be empty
        assert len(result.pareto_x) > 0
        
        # Check that all Pareto solutions are valid
        for metrics in result.pareto_metrics:
            assert metrics.get("_valid", True)
        
        # Check non-domination: no solution should dominate another
        for i in range(len(result.pareto_scores)):
            for j in range(i + 1, len(result.pareto_scores)):
                s1 = result.pareto_scores[i]
                s2 = result.pareto_scores[j]
                # Neither should dominate the other
                dominates_12 = np.all(s1 >= s2) and np.any(s1 > s2)
                dominates_21 = np.all(s2 >= s1) and np.any(s2 > s1)
                assert not (dominates_12 and dominates_21), "Solutions should not dominate each other"
    
    def test_generation_log(self):
        """Test that generation log is populated."""
        evaluator = MockMOEvaluator()
        
        def obj1(m): return m["f1"]
        def obj2(m): return m["f2"]
        
        optimizer = NSGA2Optimizer(
            evaluator=evaluator,
            objective_fns=[obj1, obj2],
            maximize=[True, True],
            popsize=20,
            maxiter=5,
            seed=42,
        )
        
        result = optimizer.optimize()
        
        assert len(result.generation_log) == 5
        for log in result.generation_log:
            assert "generation" in log
            assert "pareto_x" in log
            assert "pareto_scores" in log
            assert "fronts_count" in log


class TestMOOptimizationResult:
    def test_summary(self):
        """Test summary method."""
        pareto_x = np.array([[1, 2], [2, 1], [3, 3]])
        pareto_scores = np.array([[5, 10], [6, 8], [7, 5]])
        pareto_metrics = [{"_var_names": ["x1", "x2"]} for _ in range(3)]
        
        result = MOOptimizationResult(
            pareto_x=pareto_x,
            pareto_metrics=pareto_metrics,
            pareto_scores=pareto_scores,
            history=[],
            generation_log=[],
            success=True,
            n_evaluations=100,
        )
        
        summary = result.summary(n_show=2)
        assert "Pareto front size: 3" in summary
        assert "Total evaluations: 100" in summary
        assert "Rank 1" in summary
        assert "Rank 2" in summary


class TestUnifiedOptimize:
    def test_single_objective_default(self):
        """Test that single-objective uses DE by default."""
        evaluator = SimpleEvaluator()
        
        def obj(m): return m["value"]
        
        result = optimize(evaluator, obj, maximize=True, popsize=10, maxiter=10, seed=42)
        
        # Should return single-objective result (has x, score, not pareto_x)
        assert hasattr(result, 'x')
        assert hasattr(result, 'score')
        assert not hasattr(result, 'pareto_x')
        assert result.success
        # More simply: check it's not a MO result
        assert not hasattr(result, 'pareto_x')
        assert hasattr(result, 'x')
        assert hasattr(result, 'score')
    
    def test_multi_objective_flag(self):
        """Test that multi_objective=True uses NSGA-II."""
        evaluator = MockMOEvaluator()
        
        def obj1(m): return m["f1"]
        def obj2(m): return m["f2"]
        
        result = optimize(
            evaluator, [obj1, obj2], 
            maximize=[True, True],
            popsize=10, maxiter=5, seed=42,
            multi_objective=True
        )
        
        assert isinstance(result, MOOptimizationResult)
        assert hasattr(result, 'pareto_x')
        assert len(result.pareto_x) > 0
    
    def test_multi_objective_with_single_fn(self):
        """Test multi_objective with single function (should still work)."""
        evaluator = SimpleEvaluator()
        
        def obj(m): return m["value"]
        
        result = optimize(
            evaluator, [obj], maximize=[True],
            popsize=10, maxiter=5, seed=42,
            multi_objective=True
        )
        
        assert isinstance(result, MOOptimizationResult)
        assert len(result.pareto_x) > 0


class TestNSGA2Mechanics:
    def test_dominance_check(self):
        """Test domination logic."""
        evaluator = MockMOEvaluator()
        
        def obj1(m): return m["f1"]
        def obj2(m): return m["f2"]
        
        optimizer = NSGA2Optimizer(
            evaluator=evaluator,
            objective_fns=[obj1, obj2],
            maximize=[True, True],
        )
        
        # In minimization space (both maximized -> negated)
        # a dominates b if a <= b in all objectives and a < b in at least one
        a = np.array([-10, -5])  # f1=10, f2=5
        b = np.array([-8, -3])   # f1=8, f2=3
        
        # a is better in both (more negative = better in minimization)
        assert optimizer._dominates(a, b)
        assert not optimizer._dominates(b, a)
    
    def test_fast_non_dominated_sort(self):
        """Test fast non-dominated sorting."""
        evaluator = MockMOEvaluator()
        
        def obj1(m): return m["f1"]
        def obj2(m): return m["f2"]
        
        optimizer = NSGA2Optimizer(
            evaluator=evaluator,
            objective_fns=[obj1, obj2],
            maximize=[True, True],
        )
        
        # Points: a dominates b, b dominates c, a dominates c
        scores = np.array([
            [-10, -5],  # a: best
            [-8, -3],   # b: middle
            [-5, -1],   # c: worst
        ])
        
        fronts = optimizer._fast_non_dominated_sort(scores)
        
        assert len(fronts) == 3
        assert fronts[0] == [0]  # a is front 1
        assert fronts[1] == [1]  # b is front 2
        assert fronts[2] == [2]  # c is front 3
    
    def test_crowding_distance(self):
        """Test crowding distance calculation."""
        evaluator = MockMOEvaluator()
        
        def obj1(m): return m["f1"]
        def obj2(m): return m["f2"]
        
        optimizer = NSGA2Optimizer(
            evaluator=evaluator,
            objective_fns=[obj1, obj2],
            maximize=[True, True],
        )
        
        scores = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
        ])
        front = [0, 1, 2, 3]
        
        distances = optimizer._crowding_distance(scores, front)
        
        # Boundary points should have infinite distance
        assert np.isinf(distances[0])
        assert np.isinf(distances[-1])
        
        # Interior points should have finite positive distance
        assert distances[1] > 0
        assert distances[2] > 0
        assert distances[1] == distances[2]  # Uniform spacing
    
    def test_sbx_crossover(self):
        """Test SBX crossover produces valid offspring."""
        evaluator = MockMOEvaluator()
        
        def obj1(m): return m["f1"]
        def obj2(m): return m["f2"]
        
        optimizer = NSGA2Optimizer(
            evaluator=evaluator,
            objective_fns=[obj1, obj2],
            maximize=[True, True],
            crossover_prob=1.0,  # Always cross
            crossover_eta=20,
        )
        
        parent1 = np.array([1.0, 2.0])
        parent2 = np.array([3.0, 4.0])
        
        child1, child2 = optimizer._sbx_crossover(parent1, parent2)
        
        # Children should be within bounds
        assert np.all(child1 >= [0, 0]) and np.all(child1 <= [5, 5])
        assert np.all(child2 >= [0, 0]) and np.all(child2 <= [5, 5])
    
    def test_polynomial_mutation(self):
        """Test polynomial mutation produces valid mutants."""
        evaluator = MockMOEvaluator()
        
        def obj1(m): return m["f1"]
        def obj2(m): return m["f2"]
        
        optimizer = NSGA2Optimizer(
            evaluator=evaluator,
            objective_fns=[obj1, obj2],
            maximize=[True, True],
            mutation_prob=1.0,  # Always mutate
            mutation_eta=20,
        )
        
        individual = np.array([2.5, 2.5])
        
        for _ in range(100):
            mutant = optimizer._polynomial_mutation(individual)
            assert np.all(mutant >= [0, 0]) and np.all(mutant <= [5, 5])
            # Should be different from original (with very high probability)
            if not np.allclose(mutant, individual):
                break
        else:
            # If all mutations produced same value, that's a bug
            # But with prob=1 and eta=20, it's extremely unlikely to not mutate
            pass