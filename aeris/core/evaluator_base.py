"""
aeris.core.evaluator_base
==========================
The fixed contract every AERIS module (airfoil, winglet, aircraft, and any
future module like propeller or rocket-nozzle) must satisfy. This is what
makes the optimizer engine and logbook completely reusable across projects:
they only ever talk to this interface, never to module-specific details.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from aeris.core.validation import ValidationResult
from aeris.core.model_validity import ValidityCheck


class Evaluator(ABC):
    """Base class for every physics evaluator plug-in.

    Subclasses must set:
        name       : str, short module identifier (e.g. "airfoil")
        var_names  : list[str], one name per design variable, in order
        bounds     : list[(min, max)], one bound pair per design variable,
                     same order as var_names

    Subclasses must implement:
        evaluate(x) -> dict of metrics. The dict MUST include the key
        "_valid" (bool) indicating whether this design is physically/
        geometrically sensible (e.g. tip chord <= root chord). Invalid
        designs are penalized by the optimizer rather than crashing it.

    Optional overrides (Phase 1 additions):
        validate_input(x) -> list[ValidationResult]
        get_model_validity_conditions() -> dict
        get_constraints() -> dict (returned in metrics)
    """

    name = "base"
    var_names = []
    bounds = []

    @abstractmethod
    def evaluate(self, x):
        raise NotImplementedError

    def n_vars(self) -> int:
        return len(self.bounds)

    def describe(self) -> str:
        lines = [f"Evaluator: {self.name}", "Design variables:"]
        for n, b in zip(self.var_names, self.bounds):
            lines.append(f"  {n:22s} bounds={b}")
        return "\n".join(lines)

    # Phase 1: Optional new methods for enhanced functionality
    def validate_input(self, x) -> List[ValidationResult]:
        """Validate design variables before evaluation. Override in subclass.
        Return empty list if all valid, or list of ValidationResult for errors/warnings.
        """
        return []

    def get_model_validity_conditions(self) -> Dict[str, Any]:
        """Return conditions for model validity checking. Override in subclass.
        Should return dict with keys: mach, reynolds, alpha_deg, stall_alpha_deg,
        tip_chord, root_chord, thickness, aspect_ratio (as applicable).
        """
        return {}

    def get_constraints(self) -> Dict[str, Any]:
        """Return constraint evaluation results. Override in subclass.
        Should return dict with constraint names and their satisfied/value/limit.
        """
        return {}