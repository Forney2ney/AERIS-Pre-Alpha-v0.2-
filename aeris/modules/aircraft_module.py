"""
aeris.modules.aircraft_module
================================
Module 3: the full flying-wing shape generator — the "Neuronautics-style"
module. Takes a wide design-variable set (span, taper, sweep, washout twist,
root/tip camber, thickness, camber position, winglet geometry, operating angle
of attack) and uses strip theory (spanwise integration of the SAME thin-airfoil-
theory physics used by Module 1) plus the SAME winglet effective-aspect-ratio
model used by Module 2, so the whole system stays internally consistent: the
aircraft is literally built out of the airfoil and winglet knowledge, not a
separate re-derivation of the physics.

Design variables (12 total):
    span_m              overall span                          [0.8, 1.8] m
    root_chord_m         root chord                            [0.15, 0.45] m
    taper_ratio          tip_chord / root_chord                [0.3, 1.0]
    sweep_deg            leading-edge sweep                    [0, 35] deg
    washout_deg          tip twist relative to root (washout)  [0, 10] deg
    root_camber_m        root section max camber               [0, 0.06]
    tip_camber_m         tip section max camber                [0, 0.06]
    thickness_t          max thickness / chord                 [0.08, 0.18]
    camber_pos_p         max camber position                   [0.25, 0.5]
    winglet_height_ratio winglet height / semispan              [0, 0.25]
    winglet_cant_deg     winglet cant angle                     [0, 60] deg
    alpha_deg            operating angle of attack               [0, 10] deg
"""

import os
import numpy as np
from aeris.core import knowledge_base as kb
from aeris.core import geometry as geo
from aeris.core import cad_export
from aeris.core.evaluator_base import Evaluator
from aeris.core.atmosphere import Atmosphere
from aeris.core.validation import validate_aircraft_vars, raise_if_errors
from aeris.core.model_validity import ModelValidityEngine
from aeris.core.constraints import create_aircraft_constraints
from aeris.core.units import as_float
from aeris.core.aircraft_stability import (
    analyze_stability, trim_analysis, estimate_cg_position
)

N_STATIONS = 15          # spanwise strip-theory stations (semispan)


"""
aeris.modules.aircraft_module
================================
Module 3: the full flying-wing shape generator — the "Neuronautics-style"
module. Takes a wide design-variable set (span, taper, sweep, washout twist,
root/tip camber, winglet geometry, operating angle of attack, thickness, camber position)
and uses strip theory (spanwise integration of the SAME thin-airfoil-theory physics
used by Module 1) plus the SAME winglet effective-aspect-ratio model used by
Module 2, so the whole system stays internally consistent: the aircraft is
literally built out of the airfoil and winglet knowledge, not a separate
re-derivation of the physics.

Design variables (12 total):
    span_m              overall span                          [0.8, 1.8] m
    root_chord_m         root chord                            [0.15, 0.45] m
    taper_ratio          tip_chord / root_chord                [0.3, 1.0]
    sweep_deg            leading-edge sweep                    [0, 35] deg
    washout_deg          tip twist relative to root (washout)  [0, 10] deg
    root_camber_m        root section max camber               [0, 0.06]
    tip_camber_m         tip section max camber                [0, 0.06]
    thickness_t          max thickness / chord                 [0.08, 0.18]
    camber_pos_p         max camber position                   [0.25, 0.5]
    winglet_height_ratio winglet height / semispan              [0, 0.25]
    winglet_cant_deg     winglet cant angle                     [0, 60] deg
    alpha_deg            operating angle of attack               [0, 10] deg
"""

import os
import numpy as np
from aeris.core import knowledge_base as kb
from aeris.core import geometry as geo
from aeris.core import cad_export
from aeris.core.evaluator_base import Evaluator
from aeris.core.atmosphere import Atmosphere
from aeris.core.validation import validate_aircraft_vars, raise_if_errors
from aeris.core.model_validity import ModelValidityEngine
from aeris.core.constraints import create_aircraft_constraints
from aeris.core.units import as_float
from aeris.core.aircraft_stability import (
    analyze_stability, trim_analysis, estimate_cg_position, compute_aerodynamic_center
)

N_STATIONS = 15          # spanwise strip-theory stations (semispan)


class AircraftEvaluator(Evaluator):
    name = "aircraft"
    var_names = ["span_m", "root_chord_m", "taper_ratio", "sweep_deg",
                 "washout_deg", "root_camber_m", "tip_camber_m",
                 "thickness_t", "camber_pos_p",
                 "winglet_height_ratio", "winglet_cant_deg", "alpha_deg"]
    bounds = [(0.8, 1.8), (0.15, 0.45), (0.3, 1.0), (0.0, 35.0),
              (0.0, 10.0), (0.0, 0.06), (0.0, 0.06),
              (0.08, 0.18), (0.25, 0.5),
              (0.0, 0.25), (0.0, 60.0), (0.0, 10.0)]

    def __init__(self, cruise_velocity_mps: float, altitude_m: float = 0.0,
                 target_CL: float = None, mass_kg: float = 1.5):
        self.V = cruise_velocity_mps
        self.altitude = altitude_m
        self.target_CL = target_CL
        self.mass_kg = mass_kg
        
        # Initialize shared infrastructure
        self.atmosphere = Atmosphere.from_altitude(altitude_m)
        self.validity_engine = ModelValidityEngine()
        self.constraint_set = create_aircraft_constraints(target_CL)

    def validate_input(self, x) -> list:
        """Validate design variables using new validation system."""
        return validate_aircraft_vars(self.var_names, x)

    def get_model_validity_conditions(self) -> dict:
        """Return conditions for model validity checking."""
        # These will be computed in evaluate based on the design
        return {}

    def get_constraints(self) -> dict:
        return {}

    def evaluate(self, x):
        # Phase 1: Validate input
        validation_results = self.validate_input(x)
        raise_if_errors(validation_results)

        (span, root_chord, taper_ratio, sweep_deg, washout_deg,
         root_camber, tip_camber, thickness, camber_pos,
         wl_height_ratio, wl_cant_deg, alpha_deg) = x

        planform = geo.generate_planform(span, root_chord, taper_ratio,
                                          sweep_deg, n_stations=N_STATIONS)
        area, AR = geo.wing_area_and_AR(span, root_chord, taper_ratio)

        y = planform["y"]
        chord = planform["chord"]
        frac = planform["frac"]

        # --- strip theory: local twist, local camber, local 2D CL/CD ---
        local_twist_deg = -washout_deg * frac              # washout = negative tip twist
        local_alpha_deg = alpha_deg + local_twist_deg
        local_camber = root_camber + (tip_camber - root_camber) * frac

        local_cl = np.zeros(N_STATIONS)
        local_cd0 = np.zeros(N_STATIONS)
        local_clmax = np.zeros(N_STATIONS)

        # Use centralized atmosphere
        rho = self.atmosphere.density.value
        mu = self.atmosphere.dynamic_viscosity.value

        for i in range(N_STATIONS):
            coeffs = kb.thin_airfoil_coeffs(local_camber[i], camber_pos)
            alpha_rad = np.radians(local_alpha_deg[i])
            cl_i = kb.thin_airfoil_cl(alpha_rad, coeffs["alpha_L0_rad"])
            clmax_i = kb.empirical_cl_max(thickness, local_camber[i])
            local_cl[i] = min(cl_i, clmax_i)
            local_clmax[i] = clmax_i

            Re_i = self.atmosphere.reynolds_number(self.V, chord[i]).value
            local_cd0[i] = kb.profile_drag_coefficient(Re_i, thickness)

        # area-weighted spanwise integration -> whole-wing coeffs
        weight = chord
        CL_total = 2 * np.trapezoid(local_cl * weight, y) / area
        CD0_total = 2 * np.trapezoid(local_cd0 * weight, y) / area

        # --- winglet effect (reuses Module 2's physics) ---
        e = kb.oswald_efficiency(AR, sweep_deg)
        AR_eff = kb.winglet_effective_AR(AR, wl_height_ratio, wl_cant_deg)
        CDi = kb.induced_drag_coefficient(CL_total, AR_eff, e)
        winglet_wetted_penalty = 0.0006 * wl_height_ratio

        CD_total = CD0_total + CDi + winglet_wetted_penalty
        L_D = CL_total / CD_total if CD_total > 0 else 0.0

        # --- Real stability analysis ---
        stability_results = analyze_stability(
            span=span, root_chord=root_chord, taper_ratio=taper_ratio,
            sweep_deg=sweep_deg, washout_deg=washout_deg,
            root_camber=root_camber, tip_camber=tip_camber,
            thickness=thickness, alpha_deg=alpha_deg,
            mass=self.mass_kg
        )
        
        # Trim analysis for operating CL
        if self.target_CL is not None:
            cg_estimate = estimate_cg_position(
                span, root_chord, taper_ratio, 0,
                self.mass_kg
            )
            trim_alpha_rad, trim_Cm, _ = trim_analysis(
                span, root_chord, taper_ratio, sweep_deg,
                root_camber, tip_camber, thickness,
                cg_estimate, self.target_CL
            )
        else:
            trim_alpha_rad = 0.0
            trim_Cm = 0.0

        # --- Stability proxy (legacy) ---
        stability_score = washout_deg / 10.0 + 0.3 * (1 - taper_ratio)

        tip_stall_margin = np.min(local_clmax - local_cl)

        valid = (
            planform["tip_chord"] > 0.03
            and CD_total > 0
            and tip_stall_margin > -0.05
            and CL_total > 0.05
        )
        if self.target_CL is not None:
            valid = valid and abs(CL_total - self.target_CL) < 0.35

        # Compute Mach and Re for validity checking
        mach = self.atmosphere.mach_number(self.V).value
        reynolds = self.atmosphere.reynolds_number(self.V, span / 2).value

        metrics = {
            "CL": float(CL_total),
            "CD": float(CD_total),
            "CD0": float(CD0_total),
            "CDi": float(CDi),
            "L_D": float(L_D),
            "AR": float(AR),
            "AR_effective": float(AR_eff),
            "area_m2": float(area),
            "tip_chord_m": float(planform["tip_chord"]),
            "tip_stall_margin": float(tip_stall_margin),
            "stability_score": float(stability_score),
            "_valid": bool(valid),
        }

        # Add real stability metrics
        metrics.update({
            "static_margin": stability_results["static_margin"],
            "static_margin_pct": stability_results["stability_margin_pct"],
            "neutral_point_c_root": stability_results["neutral_point_c_root"],
            "cg_position_c_root": stability_results["cg_position_c_root"],
            "cm_ac": stability_results["cm_ac"],
            "cm_cg": stability_results["cm_cg"],
            "trim_alpha_deg": stability_results["trim_alpha_deg"],
            "trim_cl": stability_results["trim_cl"],
            "is_stable": stability_results["is_stable"],
        })

        if self.target_CL is not None:
            metrics["trim_alpha_deg_req"] = np.degrees(trim_alpha_rad)
            metrics["trim_Cm"] = trim_Cm

        # Model validity conditions
        validity_conditions = {
            "mach": mach,
            "reynolds": reynolds,
            "alpha_deg": alpha_deg,
            "stall_alpha_deg": None,  # varies by station
            "tip_chord": planform["tip_chord"],
            "root_chord": root_chord,
            "thickness": thickness,
            "aspect_ratio": AR,
        }
        metrics["_validity_conditions"] = validity_conditions

        # Evaluate constraints
        constraint_results = self.constraint_set.evaluate(metrics)
        metrics["constraints"] = {r.name: r.to_dict() for r in constraint_results}

        return metrics


def export_aircraft_cad(design_vars: dict, out_dir: str,
                         base_name: str = "aircraft", n_points: int = 40):
    """Turn a winning flying-wing design into a 3D-printable STL loft of one
    wing half (root to tip), reusing exactly the same span stations,
    chord/camber/twist schedule the physics evaluation used — so the
    printed shape matches what was actually optimized, not an
    approximation of it.

    design_vars must contain the keys in AircraftEvaluator.var_names.
    NOTE: winglet geometry is not yet included in the exported solid (the
    winglet's aerodynamic effect IS included in the performance numbers,
    but its physical shape is a good next addition — see
    modules/_template_new_module.py for the pattern to follow).
    """
    os.makedirs(out_dir, exist_ok=True)

    span = design_vars["span_m"]
    root_chord = design_vars["root_chord_m"]
    taper_ratio = design_vars["taper_ratio"]
    sweep_deg = design_vars["sweep_deg"]
    washout_deg = design_vars["washout_deg"]
    root_camber = design_vars["root_camber_m"]
    tip_camber = design_vars["tip_camber_m"]
    thickness = design_vars.get("thickness_t", 0.10)
    camber_pos = design_vars.get("camber_pos_p", 0.4)

    planform = geo.generate_planform(span, root_chord, taper_ratio,
                                      sweep_deg, n_stations=N_STATIONS)
    y = planform["y"]
    chord = planform["chord"]
    le_x = planform["le_x"]
    frac = planform["frac"]

    local_twist_deg = -washout_deg * frac
    local_camber = root_camber + (tip_camber - root_camber) * frac

    station_outlines = []
    for i in range(N_STATIONS):
        xu, zu, xl, zl = geo.naca4_coordinates(
            local_camber[i], camber_pos, thickness,
            chord=chord[i], n_points=n_points)
        pivot_x = 0.25 * chord[i]
        xu, zu = geo.rotate_points(xu, zu, pivot_x, 0.0, local_twist_deg[i])
        xl, zl = geo.rotate_points(xl, zl, pivot_x, 0.0, local_twist_deg[i])
        outline = cad_export.outline_from_surfaces(xu, zu, xl, zl)
        station_outlines.append(outline)

    stl_path = os.path.join(out_dir, f"{base_name}.stl")
    cad_export.export_wing_loft_stl(station_outlines, list(y), list(le_x),
                                     stl_path, solid_name=base_name)
    return {"stl": stl_path}