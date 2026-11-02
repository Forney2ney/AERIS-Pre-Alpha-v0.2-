"""
aeris.modules.airfoil_module
==============================
Module 1: parametric NACA-4-digit-style airfoil generator + thin-airfoil-
theory evaluator, with an optional biomimicry (tubercle leading-edge)
extension. This is the AERIS core, and its physics functions are reused by
the aircraft module (Module 3) so the whole system stays consistent.

New in Phase 2:
- AoA sweep / polar generation (CL, CD, Cm, L/D vs alpha)
- Baseline comparison (vs NACA 2412-like airfoil)
- Optional XFOIL integration for higher fidelity
- Extracted metrics: CLmax, stall alpha, best L/D, alpha_L0, Cm
"""

import os
import numpy as np
from typing import Optional, Callable, Dict, Any
from aeris.core import knowledge_base as kb
from aeris.core import geometry as geo
from aeris.core import cad_export
from aeris.core.evaluator_base import Evaluator
from aeris.core.atmosphere import Atmosphere
from aeris.core.validation import validate_airfoil_vars, raise_if_errors
from aeris.core.model_validity import ModelValidityEngine
from aeris.core.constraints import create_airfoil_constraints
from aeris.core.units import as_float


# Type alias for XFOIL runner function
XfoilRunner = Optional[Callable[[float, float, float, float], Dict[str, Any]]]


class AirfoilEvaluator(Evaluator):
    """Design variables (with biomimicry ON, the default):
        camber_m            max camber, fraction of chord         [0, 0.06]
        camber_pos_p        camber position, fraction of chord     [0.2, 0.6]
        thickness_t         max thickness, fraction of chord       [0.06, 0.18]
        tubercle_amplitude  amplitude / local chord                [0, 0.05]
        tubercle_wavelength wavelength / local chord                [0.10, 0.50]

    With biomimicry OFF, only the first three variables are used.
    """

    name = "airfoil"

    def __init__(
        self,
        alpha_deg: float,
        velocity_mps: float,
        chord_m: float,
        altitude_m: float = 0.0,
        use_biomimicry: bool = True,
        xfoil_runner: XfoilRunner = None,
        polar_alpha_min: float = -5.0,
        polar_alpha_max: Optional[float] = None,
        polar_n_points: int = 50,
    ):
        self.alpha_deg = alpha_deg
        self.velocity = velocity_mps
        self.chord = chord_m
        self.altitude = altitude_m
        self.use_biomimicry = use_biomimicry
        self.xfoil_runner = xfoil_runner
        self.polar_alpha_min = polar_alpha_min
        self.polar_alpha_max = polar_alpha_max
        self.polar_n_points = polar_n_points

        if use_biomimicry:
            self.var_names = ["camber_m", "camber_pos_p", "thickness_t",
                               "tubercle_amplitude", "tubercle_wavelength"]
            self.bounds = [(0.0, 0.06), (0.2, 0.6), (0.06, 0.18),
                            (0.0, 0.05), (0.10, 0.50)]
        else:
            self.var_names = ["camber_m", "camber_pos_p", "thickness_t"]
            self.bounds = [(0.0, 0.06), (0.2, 0.6), (0.06, 0.18)]

        # Initialize shared infrastructure
        self.atmosphere = Atmosphere.from_altitude(altitude_m)
        self.validity_engine = ModelValidityEngine()
        self.constraint_set = create_airfoil_constraints()

    def validate_input(self, x) -> list:
        """Validate design variables using new validation system."""
        return validate_airfoil_vars(self.var_names, x)

    def get_model_validity_conditions(self) -> dict:
        """Return conditions for model validity checking."""
        return {
            "mach": self.atmosphere.mach_number(self.velocity).value,
            "reynolds": self.atmosphere.reynolds_number(self.velocity, self.chord).value,
            "alpha_deg": self.alpha_deg,
        }

    def get_constraints(self) -> dict:
        return {}

    def _generate_polar(self, m: float, p: float, t: float, Re: float) -> Dict[str, Any]:
        """Generate comprehensive polar for the airfoil.
        
        Uses XFOIL if runner provided, otherwise analytical model.
        """
        if self.xfoil_runner is not None:
            try:
                xfoil_result = self.xfoil_runner(m, p, t, Re)
                # Ensure required keys exist
                if all(k in xfoil_result for k in ["alpha_deg", "CL", "CD"]):
                    # Compute derived quantities
                    alpha = np.array(xfoil_result["alpha_deg"])
                    CL = np.array(xfoil_result["CL"])
                    CD = np.array(xfoil_result["CD"])
                    LD = np.where(CD > 0, CL / CD, 0.0)
                    
                    # Find stall (first alpha where CL drops significantly)
                    cl_max_idx = np.argmax(CL)
                    stall_alpha = alpha[cl_max_idx]
                    cl_max = CL[cl_max_idx]
                    
                    # Find best L/D pre-stall
                    prestall = alpha <= stall_alpha
                    if np.any(prestall):
                        best_ld_idx = np.argmax(LD[prestall])
                        best_LD = LD[prestall][best_ld_idx]
                        best_LD_alpha = alpha[prestall][best_ld_idx]
                    else:
                        best_LD = 0.0
                        best_LD_alpha = 0.0
                    
                    return {
                        "alpha_deg": alpha.tolist(),
                        "CL": CL.tolist(),
                        "CD": CD.tolist(),
                        "LD": LD.tolist(),
                        "CL_max": float(cl_max),
                        "stall_alpha_deg": float(stall_alpha),
                        "alpha_L0_deg": float(xfoil_result.get("alpha_L0_deg", 0.0)),
                        "best_LD": float(best_LD),
                        "best_LD_alpha": float(best_LD_alpha),
                        "CD0": float(xfoil_result.get("CD0", CD[0])),
                        "source": "xfoil",
                    }
            except Exception:
                pass  # Fall back to analytical model
        
        # Analytical model (thin airfoil theory + empirical)
        return kb.polar_cl_cd_cm(
            m, p, t, Re,
            self.polar_alpha_min,
            self.polar_alpha_max,
            self.polar_n_points
        )

    def _generate_baseline_polar(self, Re: float) -> Dict[str, Any]:
        """Generate polar for baseline airfoil (NACA 2412-like)."""
        baseline = kb.create_baseline_airfoil()
        return self._generate_polar(
            baseline["camber_m"],
            baseline["camber_pos_p"],
            baseline["thickness_t"],
            Re
        )

    def evaluate(self, x, return_polar: bool = False):
        # Phase 1: Validate input
        validation_results = self.validate_input(x)
        raise_if_errors(validation_results)

        m, p, t = x[0], x[1], x[2]
        
        # Use centralized atmosphere
        Re = self.atmosphere.reynolds_number(self.velocity, self.chord).value

        # Compute operating point EXACTLY like v0.2 (for regression compatibility)
        coeffs = kb.thin_airfoil_coeffs(m, p)
        alpha_rad = np.radians(self.alpha_deg)
        cl = kb.thin_airfoil_cl(alpha_rad, coeffs["alpha_L0_rad"])
        cm = kb.thin_airfoil_cm_quarter_chord(coeffs["A1"], coeffs["A2"])

        cl_max = kb.empirical_cl_max(t, m)
        stall_alpha = kb.empirical_stall_alpha_deg(t, m)
        cd0 = kb.profile_drag_coefficient(Re, t)

        if self.use_biomimicry and len(x) > 3:
            amp, wl = x[3], x[4]
            eff = kb.tubercle_effect(amp, wl)
            cl_max += eff["d_clmax"]
            stall_alpha += eff["d_stall_alpha_deg"]
            cd0 += eff["d_cd0"]

        stalled = self.alpha_deg >= stall_alpha
        cd = cd0 if not stalled else cd0 * 3.0   # crude post-stall drag rise
        cl_eff = min(cl, cl_max) if not stalled else cl_max * 0.6

        # Generate polar for additional features (baseline comparison, extracted metrics)
        polar = self._generate_polar(m, p, t, Re)
        best_LD = polar["best_LD"]
        best_LD_alpha = polar["best_LD_alpha"]
        alpha_L0 = polar["alpha_L0_deg"]
        polar_source = polar.get("source", "analytical")

        # Baseline comparison
        baseline_polar = self._generate_baseline_polar(Re)
        baseline_comparison = {}
        for metric in ["LD", "CL", "CD", "Cm_c4"]:
            comp = kb.compare_airfoil_to_baseline(polar, baseline_polar, metric)
            baseline_comparison[f"baseline_{metric}"] = comp

        # Model validity conditions
        validity_conditions = {
            "mach": self.atmosphere.mach_number(self.velocity).value,
            "reynolds": Re,
            "alpha_deg": self.alpha_deg,
            "stall_alpha_deg": stall_alpha,
            "tip_chord": self.chord * (1 - t),
            "root_chord": self.chord,
            "thickness": t,
        }

        metrics = {
            "CL": float(cl_eff),
            "CD": float(cd),
            "Cm_c4": float(cm),
            "CL_max": float(cl_max),
            "stall_AoA_deg": float(stall_alpha),
            "L_D": float(cl_eff / cd) if cd > 0 else 0.0,
            "Reynolds": float(Re),
            "alpha_L0_deg": float(alpha_L0),
            "best_LD": float(best_LD),
            "best_LD_alpha": float(best_LD_alpha),
            "CD0": float(cd0),
            "polar_source": polar_source,
            # Baseline comparison
            "baseline_LD_improvement_pct": baseline_comparison["baseline_LD"]["improvement_pct"],
            "baseline_CL_improvement_pct": baseline_comparison["baseline_CL"]["improvement_pct"],
            "baseline_CD_improvement_pct": baseline_comparison["baseline_CD"]["improvement_pct"],
            "baseline_Cm_improvement_pct": baseline_comparison["baseline_Cm_c4"]["improvement_pct"],
            "_valid": True,
            "_validity_conditions": validity_conditions,
        }

        if return_polar:
            metrics["polar"] = polar
            metrics["baseline_polar"] = baseline_polar

        # Evaluate constraints
        constraint_results = self.constraint_set.evaluate(metrics)
        metrics["constraints"] = {r.name: r.to_dict() for r in constraint_results}

        return metrics


def export_airfoil_cad(design_vars: dict, chord_m: float, span_m: float,
                        out_dir: str, base_name: str = "airfoil"):
    """Turn a winning airfoil design (as logged by the Logbook) into a DXF
    (2D profile) and STL (extruded 3D solid, ready to 3D print) file.

    design_vars must contain at least "camber_m", "camber_pos_p",
    "thickness_t" (the keys produced by AirfoilEvaluator.var_names).
    """
    os.makedirs(out_dir, exist_ok=True)
    m = design_vars["camber_m"]
    p = design_vars["camber_pos_p"]
    t = design_vars["thickness_t"]

    xu, yu, xl, yl = geo.naca4_coordinates(m, p, t, chord=chord_m, n_points=60)
    outline = cad_export.outline_from_surfaces(xu, yu, xl, yl)

    dxf_path = os.path.join(out_dir, f"{base_name}.dxf")
    stl_path = os.path.join(out_dir, f"{base_name}.stl")
    cad_export.write_dxf(outline, dxf_path)
    cad_export.export_extruded_stl(outline, span_m, stl_path,
                                    solid_name=base_name)
    return {"dxf": dxf_path, "stl": stl_path}