"""
aeris.modules.winglet_module
==============================
Module 2: bio-inspired adaptive winglet optimizer (raptor-wingtip-feather-
spreading-inspired). Takes a FIXED baseline wing (span, area, operating CL)
and searches winglet shape/cant angle for minimum induced drag. Independent
module, but shares the same knowledge base and Evaluator/optimizer contract
as every other module.
"""

from aeris.core import knowledge_base as kb
from aeris.core.evaluator_base import Evaluator
from aeris.core.atmosphere import Atmosphere
from aeris.core.validation import validate_winglet_vars, raise_if_errors
from aeris.core.model_validity import ModelValidityEngine
from aeris.core.constraints import create_winglet_constraints
from aeris.core.units import as_float


class WingletEvaluator(Evaluator):
    """Design variables:
        height_ratio   winglet height / wing semispan     [0, 0.25]
        cant_deg       cant angle from vertical            [0, 75]
        taper_ratio    winglet tip chord / root chord      [0.2, 1.0]
    """

    name = "winglet"
    var_names = ["height_ratio", "cant_deg", "taper_ratio"]
    bounds = [(0.0, 0.25), (0.0, 75.0), (0.2, 1.0)]

    def __init__(self, base_span_m: float, base_area_m2: float,
                 operating_CL: float, base_CD0: float):
        self.base_span = base_span_m
        self.base_area = base_area_m2
        self.CL = operating_CL
        self.base_CD0 = base_CD0
        self.AR = base_span_m ** 2 / base_area_m2
        self.e_base = kb.oswald_efficiency(self.AR, sweep_deg=0.0)
        
        # Initialize shared infrastructure (use sea level atmosphere for winglet)
        self.atmosphere = Atmosphere.from_altitude(0.0)
        self.validity_engine = ModelValidityEngine()
        self.constraint_set = create_winglet_constraints()

    def validate_input(self, x) -> list:
        """Validate design variables using new validation system."""
        return validate_winglet_vars(self.var_names, x)

    def get_model_validity_conditions(self) -> dict:
        """Return conditions for model validity checking."""
        return {
            "mach": 0.0,  # Winglet operates at baseline wing's Mach
            "reynolds": self.atmosphere.reynolds_number(
                self.atmosphere.speed_of_sound, self.base_span / 2
            ).value,
            "alpha_deg": None,  # Not directly applicable
            "stall_alpha_deg": None,
            "tip_chord": self.base_area / self.base_span,  # approximate mean chord
            "root_chord": self.base_area / self.base_span,
            "thickness": 0.12,  # typical
        }

    def get_constraints(self) -> dict:
        return {}

    def evaluate(self, x):
        # Phase 1: Validate input
        validation_results = self.validate_input(x)
        raise_if_errors(validation_results)

        height_ratio, cant_deg, taper_ratio = x

        AR_eff = kb.winglet_effective_AR(self.AR, height_ratio, cant_deg)
        cdi_baseline = kb.induced_drag_coefficient(self.CL, self.AR, self.e_base)
        cdi_with_winglet = kb.induced_drag_coefficient(self.CL, AR_eff, self.e_base)

        # small structural/wetted-area drag penalty for adding a winglet,
        # roughly proportional to its size (a placeholder to calibrate later)
        added_wetted_penalty = 0.0006 * height_ratio * (1.0 + taper_ratio)

        cd_total = self.base_CD0 + cdi_with_winglet + added_wetted_penalty
        cd_baseline_total = self.base_CD0 + cdi_baseline

        l_d = self.CL / cd_total if cd_total > 0 else 0.0
        l_d_baseline = self.CL / cd_baseline_total if cd_baseline_total > 0 else 0.0

        valid = cd_total > 0

        metrics = {
            "AR_effective": float(AR_eff),
            "CDi_with_winglet": float(cdi_with_winglet),
            "CDi_baseline": float(cdi_baseline),
            "CD_total": float(cd_total),
            "L_D": float(l_d),
            "L_D_baseline": float(l_d_baseline),
            "L_D_improvement_pct": float(100 * (l_d - l_d_baseline) / l_d_baseline)
            if l_d_baseline > 0 else 0.0,
            "_valid": bool(valid),
        }

        # Model validity conditions
        validity_conditions = {
            "mach": 0.0,
            "reynolds": self.atmosphere.reynolds_number(
                self.atmosphere.speed_of_sound, self.base_span / 2
            ).value,
            "alpha_deg": None,
            "stall_alpha_deg": None,
            "tip_chord": self.base_area / self.base_span,
            "root_chord": self.base_area / self.base_span,
            "thickness": 0.12,
        }
        metrics["_validity_conditions"] = validity_conditions

        # Evaluate constraints
        constraint_results = self.constraint_set.evaluate(metrics)
        metrics["constraints"] = {r.name: r.to_dict() for r in constraint_results}

        return metrics