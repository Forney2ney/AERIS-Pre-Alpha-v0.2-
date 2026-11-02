"""
Tests for aeris.core.constraints
"""

import pytest
from aeris.core.constraints import (
    ConstraintResult, Constraint, UpperBoundConstraint, LowerBoundConstraint,
    RangeConstraint, CustomConstraint, ConstraintSet,
    cl_target_constraint, min_lift_coefficient, max_drag_coefficient,
    min_lift_to_drag, positive_cd_constraint, tip_stall_margin_constraint,
    geometry_tip_chord_constraint,
    create_airfoil_constraints, create_aircraft_constraints, create_winglet_constraints
)
from aeris.core.units import Quantity


class TestConstraintResult:
    def test_violation_calculation(self):
        r = ConstraintResult("test", False, 10.0, 5.0, "hard")
        assert r.violation == 5.0
        
        r2 = ConstraintResult("test", True, 3.0, 5.0, "hard")
        assert r2.violation == 0.0
    
    def test_margin_calculation(self):
        # For upper bound: margin = limit - value
        r = ConstraintResult("test", True, 3.0, 5.0, "hard")
        assert r.margin == 2.0  # 5 - 3 = 2
        
        r2 = ConstraintResult("test", False, 7.0, 5.0, "hard")
        assert r2.margin == -2.0  # 5 - 7 = -2
        
        # For lower bound style: margin = value - limit
        # But our implementation uses limit - value for both
        # So satisfied upper bound: positive margin
        # Violated upper bound: negative margin


class TestUpperBoundConstraint:
    def test_satisfied(self):
        c = UpperBoundConstraint("max_cd", "CD", 0.05)
        result = c.evaluate({"CD": 0.03})
        assert result.satisfied is True
        assert result.value == 0.03
        assert result.limit == 0.05
    
    def test_violated(self):
        c = UpperBoundConstraint("max_cd", "CD", 0.05)
        result = c.evaluate({"CD": 0.08})
        assert result.satisfied is False
        assert result.value == 0.08
        assert result.violation == 0.03
    
    def test_missing_key(self):
        c = UpperBoundConstraint("max_cd", "CD", 0.05)
        result = c.evaluate({})
        assert result.satisfied is False


class TestLowerBoundConstraint:
    def test_satisfied(self):
        c = LowerBoundConstraint("min_cl", "CL", 0.5)
        result = c.evaluate({"CL": 0.8})
        assert result.satisfied is True
        assert result.value == 0.8
    
    def test_violated(self):
        c = LowerBoundConstraint("min_cl", "CL", 0.5)
        result = c.evaluate({"CL": 0.3})
        assert result.satisfied is False
        assert result.violation == 0.2


class TestRangeConstraint:
    def test_satisfied_within_range(self):
        c = RangeConstraint("cl_target", "CL", 0.5, 0.7)
        result = c.evaluate({"CL": 0.6})
        assert result.satisfied is True
    
    def test_violated_below(self):
        c = RangeConstraint("cl_target", "CL", 0.5, 0.7)
        result = c.evaluate({"CL": 0.3})
        assert result.satisfied is False
    
    def test_violated_above(self):
        c = RangeConstraint("cl_target", "CL", 0.5, 0.7)
        result = c.evaluate({"CL": 0.9})
        assert result.satisfied is False


class TestCustomConstraint:
    def test_custom_check(self):
        def check(metrics):
            cl = metrics.get("CL", 0)
            cd = metrics.get("CD", 1)
            ld = cl / cd if cd > 0 else 0
            return ld > 10, ld, 10
        
        c = CustomConstraint("min_ld", check)
        result = c.evaluate({"CL": 1.0, "CD": 0.05})  # L/D = 20
        assert result.satisfied is True
        assert result.value == 20.0
        
        result2 = c.evaluate({"CL": 0.5, "CD": 0.1})  # L/D = 5
        assert result2.satisfied is False


class TestConstraintSet:
    def test_add_constraints(self):
        cs = ConstraintSet()
        cs.add(UpperBoundConstraint("a", "A", 10))
        cs.add(LowerBoundConstraint("b", "B", 5))
        assert len(cs) == 2
    
    def test_evaluate_all(self):
        cs = ConstraintSet()
        cs.add(UpperBoundConstraint("max_cd", "CD", 0.05))
        cs.add(LowerBoundConstraint("min_cl", "CL", 0.5))
        
        results = cs.evaluate({"CD": 0.03, "CL": 0.8})
        assert len(results) == 2
        assert all(r.satisfied for r in results)
    
    def test_all_satisfied(self):
        cs = ConstraintSet()
        cs.add(UpperBoundConstraint("max_cd", "CD", 0.05))
        cs.add(LowerBoundConstraint("min_cl", "CL", 0.5))
        
        assert cs.all_satisfied({"CD": 0.03, "CL": 0.8}) is True
        assert cs.all_satisfied({"CD": 0.08, "CL": 0.8}) is False
        assert cs.all_satisfied({"CD": 0.03, "CL": 0.3}) is False
    
    def test_hard_satisfied(self):
        cs = ConstraintSet()
        cs.add(UpperBoundConstraint("hard_cd", "CD", 0.05, kind="hard"))
        cs.add(LowerBoundConstraint("soft_cl", "CL", 0.5, kind="soft"))
        
        # Hard satisfied, soft violated
        assert cs.hard_satisfied({"CD": 0.03, "CL": 0.3}) is True
        # Hard violated
        assert cs.hard_satisfied({"CD": 0.08, "CL": 0.8}) is False
    
    def test_soft_violation_penalty(self):
        cs = ConstraintSet()
        cs.add(UpperBoundConstraint("soft_cd", "CD", 0.05, kind="soft", weight=10.0))
        cs.add(LowerBoundConstraint("soft_cl", "CL", 0.5, kind="soft", weight=5.0))
        
        # Both violated: CD=0.08 (viol=0.03), CL=0.3 (viol=0.2)
        penalty = cs.soft_violation_penalty({"CD": 0.08, "CL": 0.3})
        assert penalty == 10.0 * 0.03 + 5.0 * 0.2  # 0.3 + 1.0 = 1.3
        
        # Neither violated
        penalty2 = cs.soft_violation_penalty({"CD": 0.03, "CL": 0.8})
        assert penalty2 == 0.0
    
    def test_summary(self):
        cs = ConstraintSet()
        cs.add(UpperBoundConstraint("max_cd", "CD", 0.05))
        cs.add(LowerBoundConstraint("min_cl", "CL", 0.5))
        
        summary = cs.summary({"CD": 0.03, "CL": 0.8})
        assert summary["all_satisfied"] is True
        assert summary["hard_satisfied"] is True
        assert summary["soft_penalty"] == 0.0
        assert len(summary["constraints"]) == 2


class TestFactoryFunctions:
    def test_cl_target_constraint(self):
        c = cl_target_constraint(0.6, 0.05)
        assert c.metric_key == "CL"
        assert abs(c.lower - 0.55) < 1e-10
        assert abs(c.upper - 0.65) < 1e-10
    
    def test_min_lift_coefficient(self):
        c = min_lift_coefficient(0.5)
        assert c.metric_key == "CL"
        assert c.limit == 0.5
    
    def test_max_drag_coefficient(self):
        c = max_drag_coefficient(0.05)
        assert c.metric_key == "CD"
        assert c.limit == 0.05
    
    def test_min_lift_to_drag(self):
        c = min_lift_to_drag(20.0)
        assert c.metric_key == "L_D"
        assert c.limit == 20.0
    
    def test_positive_cd(self):
        c = positive_cd_constraint()
        assert c.metric_key == "CD"
        assert c.limit == 1e-10
    
    def test_tip_stall_margin(self):
        c = tip_stall_margin_constraint(0.1)
        assert c.metric_key == "tip_stall_margin"
        assert c.limit == 0.1


class TestPredefinedSets:
    def test_create_airfoil_constraints(self):
        cs = create_airfoil_constraints(target_cl=0.6)
        assert len(cs) >= 2  # positive_CD, min_CL, CL_target
        
        # Test with good values
        assert cs.all_satisfied({"CD": 0.02, "CL": 0.6}) is True
        # Test CD violation
        assert cs.all_satisfied({"CD": -0.01, "CL": 0.6}) is False
        # Test CL violation
        assert cs.all_satisfied({"CD": 0.02, "CL": -0.1}) is False
    
    def test_create_aircraft_constraints(self):
        cs = create_aircraft_constraints(target_cl=0.6)
        assert len(cs) >= 4  # positive_CD, min_CL, tip_stall_margin, tip_chord_ratio, CL_target
    
    def test_create_winglet_constraints(self):
        cs = create_winglet_constraints()
        assert len(cs) >= 1  # positive_CD


class TestQuantityIntegration:
    def test_quantity_values_in_metrics(self):
        c = UpperBoundConstraint("max_cd", "CD", 0.05)
        # Metrics with Quantity objects
        result = c.evaluate({"CD": Quantity(0.03, "1")})
        assert result.satisfied is True
        assert result.value == 0.03
    
    def test_quantity_in_custom_constraint(self):
        def check(metrics):
            cl = metrics.get("CL", 0)
            # Convert Quantity to float for comparison
            from aeris.core.units import as_float
            cl_val = as_float(cl)
            return cl_val > 0.5, cl_val, 0.5
        c = CustomConstraint("min_cl", check)
        result = c.evaluate({"CL": Quantity(0.8, "1")})
        assert result.satisfied is True
        assert result.value == 0.8


class TestEdgeCases:
    def test_zero_weight_soft_constraint(self):
        c = UpperBoundConstraint("test", "CD", 0.05, kind="soft", weight=0.0)
        cs = ConstraintSet()
        cs.add(c)
        penalty = cs.soft_violation_penalty({"CD": 0.1})
        assert penalty == 0.0
    
    def test_multiple_soft_constraints_penalty(self):
        cs = ConstraintSet()
        cs.add(UpperBoundConstraint("cd1", "CD", 0.05, kind="soft", weight=1.0))
        cs.add(UpperBoundConstraint("cd2", "CD", 0.03, kind="soft", weight=2.0))
        
        # CD=0.08: viol1=0.03, viol2=0.05
        penalty = cs.soft_violation_penalty({"CD": 0.08})
        assert penalty == 1.0 * 0.03 + 2.0 * 0.05  # 0.13