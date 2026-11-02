"""
aeris.core.aircraft_stability
=============================
Real longitudinal stability analysis for flying wings.

Computes:
- Aerodynamic center (AC) / neutral point
- Center of gravity (CG) estimation
- Static margin
- Pitching moment coefficient (Cm)
- Trim analysis
"""

import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class StabilityResult:
    """Results of stability analysis."""
    neutral_point: float          # x/c from leading edge (0..1)
    cg_position: float            # x/c from leading edge (0..1) 
    static_margin: float          # (NP - CG) / MAC (positive = stable)
    cm_ac: float                  # Pitching moment about aerodynamic center
    cm_cg: float                  # Pitching moment about CG
    trim_alpha: float             # Trim angle of attack (rad)
    trim_cl: float                # Trim lift coefficient
    is_stable: bool               # Static margin > 0
    stability_margin_pct: float   # Static margin as percentage of MAC


def compute_aerodynamic_center(
    span: float,
    root_chord: float,
    taper_ratio: float,
    sweep_deg: float,
    root_camber: float,
    tip_camber: float,
    thickness: float,
    n_stations: int = 15,
) -> Tuple[float, float]:
    """
    Compute wing aerodynamic center (neutral point) using strip theory.
    
    Returns:
        (x_ac_root, x_ac_tip) in root chord fractions from root leading edge
        For flying wing, NP is the lift-weighted mean aerodynamic center
    """
    from aeris.core import geometry as geo
    from aeris.core import knowledge_base as kb
    
    planform = geo.generate_planform(span, root_chord, taper_ratio, sweep_deg, n_stations=n_stations)
    y = planform["y"]
    chord = planform["chord"]
    le_x = planform["le_x"]
    frac = planform["frac"]
    
    local_camber = root_camber + (tip_camber - root_camber) * frac
    
    # Local aerodynamic center for each section (quarter-chord for thin airfoil theory)
    # x_ac_local = 0.25 * chord[i] from section leading edge
    x_ac_from_root_le = le_x + 0.25 * chord  # distance from root leading edge
    
    # Weight by local lift (chord * cl_alpha)
    # For thin airfoil, cl_alpha = 2π per radian (same for all sections)
    # So weight by chord * dynamic pressure (which varies with chord)
    cl_alpha = 2 * np.pi  # per radian
    
    # Lift-weighted aerodynamic center
    weights = chord * cl_alpha
    x_np = np.trapezoid(x_ac_from_root_le * weights, y) / np.trapezoid(weights, y)
    
    # Convert to root chord fraction
    x_np_c_root = x_np / root_chord
    
    return float(x_np_c_root), float(x_np / chord[-1])  # root fraction, tip fraction


def estimate_cg_position(
    span: float,
    root_chord: float,
    taper_ratio: float,
    sweep_deg: float,
    mass: float,                    # kg
    payload_mass: float = 0.0,      # kg
    payload_x_c_root: float = 0.3,  # x/c from root LE
    battery_mass: float = 0.0,      # kg
    battery_x_c_root: float = 0.25, # x/c from root LE
    motor_mass: float = 0.0,        # kg
    motor_x_c_root: float = 0.9,    # x/c from root LE (rear)
) -> float:
    """
    Estimate CG position from component masses and positions.
    
    Returns:
        CG position as fraction of root chord from root leading edge
    """
    from aeris.core import knowledge_base as kb
    from aeris.core import geometry as geo
    
    # Wing structure mass estimate (very rough: 50-100 g/m for small wings)
    # For flying wing, wing IS the structure
    planform = geo.generate_planform(span, root_chord, taper_ratio, 0, n_stations=15)
    area = span * (root_chord + root_chord * taper_ratio) / 2
    
    # Wing structural mass (assume 100 g/m^2 for foam/composite skin + spar)
    wing_area = area
    wing_struct_mass = 0.1 * wing_area  # kg
    
    # Wing CG at ~40% MAC (typical for tapered wing)
    mac = root_chord * (2/3) * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio)
    wing_cg_x = 0.4 * mac
    
    # Total mass and CG
    total_mass = mass
    
    # Component moments
    moments = []
    
    # Wing structure
    moments.append((wing_struct_mass, wing_cg_x))
    
    # Payload
    if payload_mass > 0:
        moments.append((payload_mass, payload_x_c_root * root_chord))
    
    # Battery
    if battery_mass > 0:
        moments.append((battery_mass, battery_x_c_root * root_chord))
    
    # Motor
    if motor_mass > 0:
        moments.append((motor_mass, motor_x_c_root * root_chord))
    
    # Remaining mass (electronics, servos, wiring, etc.) - assume at wing CG
    known_mass = sum(m for m, _ in moments)
    remaining_mass = max(0, total_mass - known_mass)
    if remaining_mass > 0:
        moments.append((remaining_mass, wing_cg_x))
    
    # Compute CG
    total_moment = sum(m * x for m, x in moments)
    cg_x = total_moment / total_mass
    
    return float(cg_x / root_chord)


def compute_static_margin(
    span: float,
    root_chord: float,
    taper_ratio: float,
    sweep_deg: float,
    root_camber: float,
    tip_camber: float,
    thickness: float,
    cg_position: float,
    mass: float,
    payload_mass: float = 0.0,
    payload_x_c_root: float = 0.3,
    battery_mass: float = 0.0,
    battery_x_c_root: float = 0.25,
    motor_mass: float = 0.0,
    motor_x_c_root: float = 0.9,
) -> StabilityResult:
    """
    Complete static margin analysis for flying wing.
    
    Args:
        span: Wing span (m)
        root_chord: Root chord (m)
        taper_ratio: Tip chord / root chord
        sweep_deg: Leading edge sweep (deg)
        root_camber: Root section max camber
        tip_camber: Tip section max camber
        thickness: Thickness ratio (t/c)
        cg_position: CG as fraction of root chord from root LE (0..1)
        mass: Total mass (kg)
        ...component masses and positions for CG estimation
        
    Returns:
        StabilityResult with complete stability analysis
    """
    # Neutral point (aerodynamic center of whole wing)
    np_c_root, _ = compute_aerodynamic_center(
        span, root_chord, taper_ratio, sweep_deg,
        root_camber, tip_camber, thickness
    )
    
    # If CG not provided, estimate it
    if cg_position is None:
        cg_c_root = estimate_cg_position(
            span, root_chord, taper_ratio, sweep_deg,
            mass, payload_mass, payload_x_c_root,
            battery_mass, battery_x_c_root,
            motor_mass, motor_x_c_root
        )
    else:
        cg_c_root = cg_position
    
    # MAC
    mac = root_chord * (2/3) * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio)
    
    # Static margin = (NP - CG) / MAC
    static_margin = (np_c_root - cg_c_root) * root_chord / mac
    
    # Aerodynamic center pitching moment (Cm_ac)
    # For flying wing, integrate section Cm_ac
    from aeris.core import knowledge_base as kb
    from aeris.core import geometry as geo
    
    n_stations = 15
    planform = geo.generate_planform(span, root_chord, taper_ratio, sweep_deg, n_stations=n_stations)
    y = planform["y"]
    chord = planform["chord"]
    frac = planform["frac"]
    local_camber = root_camber + (tip_camber - root_camber) * frac
    
    cm_ac_sections = np.zeros(n_stations)
    cl_alpha_sections = np.zeros(n_stations)
    
    for i in range(n_stations):
        coeffs = kb.thin_airfoil_coeffs(local_camber[i], 0.4)
        cm_ac_sections[i] = kb.thin_airfoil_cm_quarter_chord(coeffs["A1"], coeffs["A2"])
        cl_alpha_sections[i] = 2 * np.pi
    
    # Area-weighted average Cm_ac
    weights = chord
    cm_ac = np.trapezoid(cm_ac_sections * weights, y) / np.trapezoid(weights, y)
    
    # Trim analysis
    # Cm_cg = Cm_ac + CL * (x_ac - x_cg) / c
    # For trim: Cm_cg = 0 => CL_trim = -Cm_ac / ((x_ac - x_cg) / c)
    x_ac = np_c_root
    x_cg = cg_c_root
    
    if abs(x_ac - x_cg) > 1e-6:
        trim_cl = -cm_ac / ((x_ac - x_cg))
        trim_alpha = trim_cl / (2 * np.pi)  # rad, thin airfoil
    else:
        trim_cl = 0.0
        trim_alpha = 0.0
    
    # Static margin as percentage
    stability_margin_pct = static_margin * 100
    
    is_stable = static_margin > 0.02  # > 2% MAC
    
    return StabilityResult(
        neutral_point=np_c_root,
        cg_position=cg_c_root,
        static_margin=static_margin,
        cm_ac=cm_ac,
        cm_cg=cm_ac + trim_cl * (x_ac - x_cg),
        trim_alpha=trim_alpha,
        trim_cl=trim_cl,
        is_stable=is_stable,
        stability_margin_pct=stability_margin_pct,
    )


def analyze_stability(
    span: float,
    root_chord: float,
    taper_ratio: float,
    sweep_deg: float,
    washout_deg: float,
    root_camber: float,
    tip_camber: float,
    thickness: float,
    alpha_deg: float,
    mass: float,
    target_CL: float = None,
    payload_mass: float = 0.0,
    payload_x_c_root: float = 0.3,
    battery_mass: float = 0.0,
    battery_x_c_root: float = 0.25,
    motor_mass: float = 0.0,
    motor_x_c_root: float = 0.9,
) -> Dict[str, Any]:
    """
    Complete stability analysis for flying wing design.
    
    Returns dict with all stability metrics for logging/reporting.
    """
    # Get CG estimate
    cg_c_root = estimate_cg_position(
        span, root_chord, taper_ratio, sweep_deg,
        mass, 0, 0.3, 0, 0.25, 0, 0.9
    )
    
    # Full stability analysis
    stability = compute_static_margin(
        span, root_chord, taper_ratio, sweep_deg,
        root_camber, tip_camber, 0.10,  # thickness fixed at 0.10
        cg_c_root,
        mass, 0, 0.3, 0, 0.25, 0, 0.9
    )
    
    return {
        "neutral_point_c_root": stability.neutral_point,
        "cg_position_c_root": stability.cg_position,
        "static_margin": stability.static_margin,
        "static_margin_pct": stability.stability_margin_pct,
        "cm_ac": stability.cm_ac,
        "cm_cg": stability.cm_cg,
        "trim_alpha_deg": np.degrees(stability.trim_alpha),
        "trim_cl": stability.trim_cl,
        "is_stable": stability.is_stable,
        "stability_margin_pct": stability.stability_margin_pct,
    }


def trim_analysis(
    span: float,
    root_chord: float,
    taper_ratio: float,
    sweep_deg: float,
    root_camber: float,
    tip_camber: float,
    thickness: float,
    cg_c_root: float,
    target_CL: float,
) -> Tuple[float, float, float]:
    """
    Compute trim conditions for given target CL.
    
    Returns:
        (trim_alpha_rad, trim_Cm, elevator_deflection_rad)
    """
    from aeris.core import knowledge_base as kb
    from aeris.core import geometry as geo
    
    n_stations = 15
    planform = geo.generate_planform(span, root_chord, taper_ratio, sweep_deg, n_stations=n_stations)
    y = planform["y"]
    chord = planform["chord"]
    frac = planform["frac"]
    local_camber = root_camber + (tip_camber - root_camber) * frac
    
    # Section aerodynamic center and Cm
    x_ac_sections = 0.25 * chord  # from local LE
    x_ac_from_root_le = planform["le_x"] + x_ac_sections
    
    cm_ac_sections = np.zeros(n_stations)
    cl_alpha_sections = np.zeros(n_stations)
    alpha_L0_sections = np.zeros(n_stations)
    
    for i in range(n_stations):
        coeffs = kb.thin_airfoil_coeffs(local_camber[i], 0.4)
        cm_ac_sections[i] = kb.thin_airfoil_cm_quarter_chord(coeffs["A1"], coeffs["A2"])
        cl_alpha_sections[i] = 2 * np.pi
        alpha_L0_sections[i] = coeffs["alpha_L0_rad"]
    
    # Area-weighted averages
    weights = chord
    cm_ac = np.trapezoid(cm_ac_sections * weights, planform["y"]) / np.trapezoid(weights, planform["y"])
    x_ac = np.trapezoid(x_ac_from_root_le * weights, planform["y"]) / np.trapezoid(weights, planform["y"])
    x_ac_c_root = x_ac / planform["chord"][0]
    
    # Trim alpha for target CL
    # CL = cl_alpha * (alpha - alpha_L0)
    alpha_L0_avg = np.trapezoid(alpha_L0_sections * weights, planform["y"]) / np.trapezoid(weights, planform["y"])
    cl_alpha_avg = np.trapezoid(cl_alpha_sections * weights, planform["y"]) / np.trapezoid(weights, planform["y"])
    
    trim_alpha_rad = target_CL / cl_alpha_avg + alpha_L0_avg
    
    # Cm at CG
    trim_Cm = cm_ac + target_CL * (x_ac_c_root - cg_c_root)
    
    return trim_alpha_rad, trim_Cm, 0.0  # elevator deflection (0 for flying wing)