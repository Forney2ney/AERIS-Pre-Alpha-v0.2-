"""
aeris.core.constraints
======================
Generic constraint infrastructure for optimization.

Provides a structured way to define, evaluate, and report constraints
separate from the optimizer's penalty mechanism. This enables:
- Clear constraint reporting in results
- Future migration to more sophisticated constraint handling
- Hard vs soft constraint distinction
"""

from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Literal, Optional
from abc import ABC, abstractmethod

from aeris.core.units import Quantity, as_float


@dataclass
class ConstraintResult:
    """Result of evaluating a single constraint."""
    name: str
    satisfied: bool
    value: float
    limit: float
    kind: str  # "hard" or "soft"
    weight: float = 1.0
    
    @property
    def violation(self) -> float:
        """Amount of violation (positive = violated)."""
        if self.satisfied:
            return 0.0
        return abs(self.value - self.limit)
    
    @property
    def margin(self) -> float:
        """Positive = satisfied with margin, negative = violated.
        
        For upper bounds: margin = limit - value (positive when value < limit)
        For lower bounds: margin = value - limit (positive when value > limit)
        """
        return self.limit - self.value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "satisfied": self.satisfied,
            "value": self.value,
            "limit": self.limit,
            "kind": self.kind,
            "weight": self.weight,
            "violation": self.violation,
            "margin": self.margin,
        }


class Constraint(ABC):
    """Abstract base class for constraints."""
    
    def __init__(self, name: str, kind: Literal["hard", "soft"] = "hard", weight: float = 1.0):
        self.name = name
        self.kind = kind
        self.weight = weight
    
    @abstractmethod
    def evaluate(self, metrics: Dict[str, Any]) -> ConstraintResult:
        """Evaluate constraint against metrics dict."""
        pass
    
    def __repr__(self):
        return f"Constraint({self.name}, kind={self.kind}, weight={self.weight})"


class UpperBoundConstraint(Constraint):
    """Constraint: metric <= limit"""
    
    def __init__(self, name: str, metric_key: str, limit: float, 
                 kind: Literal["hard", "soft"] = "hard", weight: float = 1.0):
        super().__init__(name, kind, weight)
        self.metric_key = metric_key
        self.limit = limit
    
    def evaluate(self, metrics: Dict[str, Any]) -> ConstraintResult:
        value = metrics.get(self.metric_key)
        if value is None:
            return ConstraintResult(
                name=self.name,
                satisfied=False,
                value=float('nan'),
                limit=self.limit,
                kind=self.kind,
                weight=self.weight
            )
        value = as_float(value)
        satisfied = value <= self.limit
        return ConstraintResult(
            name=self.name,
            satisfied=satisfied,
            value=value,
            limit=self.limit,
            kind=self.kind,
            weight=self.weight
        )


class LowerBoundConstraint(Constraint):
    """Constraint: metric >= limit"""
    
    def __init__(self, name: str, metric_key: str, limit: float,
                 kind: Literal["hard", "soft"] = "hard", weight: float = 1.0):
        super().__init__(name, kind, weight)
        self.metric_key = metric_key
        self.limit = limit
    
    def evaluate(self, metrics: Dict[str, Any]) -> ConstraintResult:
        value = metrics.get(self.metric_key)
        if value is None:
            return ConstraintResult(
                name=self.name,
                satisfied=False,
                value=float('nan'),
                limit=self.limit,
                kind=self.kind,
                weight=self.weight
            )
        value = as_float(value)
        satisfied = value >= self.limit
        return ConstraintResult(
            name=self.name,
            satisfied=satisfied,
            value=value,
            limit=self.limit,
            kind=self.kind,
            weight=self.weight
        )


class RangeConstraint(Constraint):
    """Constraint: lower <= metric <= upper"""
    
    def __init__(self, name: str, metric_key: str, lower: float, upper: float,
                 kind: Literal["hard", "soft"] = "hard", weight: float = 1.0):
        super().__init__(name, kind, weight)
        self.metric_key = metric_key
        self.lower = lower
        self.upper = upper
    
    def evaluate(self, metrics: Dict[str, Any]) -> ConstraintResult:
        value = metrics.get(self.metric_key)
        if value is None:
            return ConstraintResult(
                name=self.name,
                satisfied=False,
                value=float('nan'),
                limit=self.upper,
                kind=self.kind,
                weight=self.weight
            )
        value = as_float(value)
        satisfied = self.lower <= value <= self.upper
        return ConstraintResult(
            name=self.name,
            satisfied=satisfied,
            value=value,
            limit=self.upper,
            kind=self.kind,
            weight=self.weight
        )


class CustomConstraint(Constraint):
    """Constraint with custom evaluation function."""
    
    def __init__(self, name: str, check_fn: Callable[[Dict], tuple[bool, float, float]],
                 kind: Literal["hard", "soft"] = "hard", weight: float = 1.0):
        super().__init__(name, kind, weight)
        self.check_fn = check_fn
    
    def evaluate(self, metrics: Dict[str, Any]) -> ConstraintResult:
        satisfied, value, limit = self.check_fn(metrics)
        return ConstraintResult(
            name=self.name,
            satisfied=satisfied,
            value=value,
            limit=limit,
            kind=self.kind,
            weight=self.weight
        )


class ConstraintSet:
    """Collection of constraints with batch evaluation."""
    
    def __init__(self, constraints: Optional[List[Constraint]] = None):
        self.constraints = constraints or []
    
    def add(self, constraint: Constraint) -> "ConstraintSet":
        self.constraints.append(constraint)
        return self
    
    def evaluate(self, metrics: Dict[str, Any]) -> List[ConstraintResult]:
        """Evaluate all constraints against metrics."""
        return [c.evaluate(metrics) for c in self.constraints]
    
    def all_satisfied(self, metrics: Dict[str, Any]) -> bool:
        """Check if all constraints are satisfied."""
        return all(r.satisfied for r in self.evaluate(metrics))
    
    def hard_satisfied(self, metrics: Dict[str, Any]) -> bool:
        """Check if all hard constraints are satisfied."""
        return all(r.satisfied for r in self.evaluate(metrics) if r.kind == "hard")
    
    def soft_violation_penalty(self, metrics: Dict[str, Any]) -> float:
        """Compute penalty for soft constraint violations (for optimizer)."""
        penalty = 0.0
        for r in self.evaluate(metrics):
            if r.kind == "soft" and not r.satisfied:
                penalty += r.weight * r.violation
        return penalty
    
    def summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary dict for logging/reporting."""
        results = self.evaluate(metrics)
        return {
            "all_satisfied": self.all_satisfied(metrics),
            "hard_satisfied": self.hard_satisfied(metrics),
            "soft_penalty": self.soft_violation_penalty(metrics),
            "constraints": [r.to_dict() for r in results],
        }
    
    def __len__(self) -> int:
        return len(self.constraints)
    
    def __iter__(self):
        return iter(self.constraints)


# Predefined constraint factories for common AERIS constraints
def cl_target_constraint(target_cl: float, tolerance: float = 0.05, 
                         kind: str = "hard") -> RangeConstraint:
    """Constraint: CL within tolerance of target."""
    return RangeConstraint(
        name="CL_target",
        metric_key="CL",
        lower=target_cl - tolerance,
        upper=target_cl + tolerance,
        kind=kind,
        weight=10.0 if kind == "soft" else 1.0
    )


def min_lift_coefficient(min_cl: float, kind: str = "hard") -> LowerBoundConstraint:
    """Constraint: CL >= min_cl"""
    return LowerBoundConstraint(
        name="min_CL",
        metric_key="CL",
        limit=min_cl,
        kind=kind
    )


def max_drag_coefficient(max_cd: float, kind: str = "hard") -> UpperBoundConstraint:
    """Constraint: CD <= max_cd"""
    return UpperBoundConstraint(
        name="max_CD",
        metric_key="CD",
        limit=max_cd,
        kind=kind
    )


def min_lift_to_drag(min_ld: float, kind: str = "hard") -> LowerBoundConstraint:
    """Constraint: L/D >= min_ld"""
    return LowerBoundConstraint(
        name="min_L_D",
        metric_key="L_D",
        limit=min_ld,
        kind=kind
    )


def positive_cd_constraint(kind: str = "hard") -> LowerBoundConstraint:
    """Constraint: CD > 0"""
    return LowerBoundConstraint(
        name="positive_CD",
        metric_key="CD",
        limit=1e-10,
        kind=kind
    )


def tip_stall_margin_constraint(min_margin: float = 0.0, kind: str = "hard") -> LowerBoundConstraint:
    """Constraint: tip_stall_margin >= min_margin"""
    return LowerBoundConstraint(
        name="tip_stall_margin",
        metric_key="tip_stall_margin",
        limit=min_margin,
        kind=kind
    )


def geometry_tip_chord_constraint(min_ratio: float = 0.03, kind: str = "hard") -> CustomConstraint:
    """Constraint: tip_chord / root_chord >= min_ratio"""
    def check(metrics: Dict) -> tuple[bool, float, float]:
        tip = metrics.get("tip_chord_m", 0)
        root = metrics.get("root_chord_m", 1)
        ratio = tip / root if root > 0 else 0
        return ratio >= min_ratio, ratio, min_ratio
    return CustomConstraint(
        name="tip_chord_ratio",
        check_fn=check,
        kind=kind
    )


def create_airfoil_constraints(target_cl: float = None) -> ConstraintSet:
    """Create standard constraint set for airfoil optimization."""
    cs = ConstraintSet()
    cs.add(positive_cd_constraint("hard"))
    cs.add(min_lift_coefficient(0.0, "hard"))
    if target_cl is not None:
        cs.add(cl_target_constraint(target_cl, 0.1, "soft"))
    return cs


def create_aircraft_constraints(target_cl: float = None) -> ConstraintSet:
    """Create standard constraint set for aircraft optimization."""
    cs = ConstraintSet()
    cs.add(positive_cd_constraint("hard"))
    cs.add(min_lift_coefficient(0.05, "hard"))
    cs.add(tip_stall_margin_constraint(0.0, "hard"))
    cs.add(geometry_tip_chord_constraint(0.03, "hard"))
    if target_cl is not None:
        cs.add(cl_target_constraint(target_cl, 0.15, "soft"))
    return cs


def positive_cd_total_constraint(kind: str = "hard") -> LowerBoundConstraint:
    """Constraint: CD_total > 0 (for winglet module)"""
    return LowerBoundConstraint(
        name="positive_CD_total",
        metric_key="CD_total",
        limit=1e-10,
        kind=kind
    )


def create_winglet_constraints() -> ConstraintSet:
    """Create standard constraint set for winglet optimization."""
    cs = ConstraintSet()
    cs.add(positive_cd_total_constraint("hard"))
    return cs