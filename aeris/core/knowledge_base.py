"""
aeris.core.knowledge_base
=========================
The "physics lab" of AERIS.

This module holds every fundamental physical/aerodynamic relationship used by
every evaluator module (airfoil, winglet, aircraft, and future ones like
propeller or rocket-nozzle modules). Nothing in here knows about optimization,
CAD, or any specific project — it is pure physics and empirical engineering
correlations, each one documented with its source/assumption and its known
limitations, so it's obvious what needs calibration against real test data.

Design principle: every evaluator module should import from HERE for any
physics constant or equation instead of re-deriving it. That's what makes the
system "one knowledge base, many modules."
"""

import numpy as np

# Import new infrastructure
from aeris.core import constants as const
from aeris.core.atmosphere import Atmosphere, standard_atmosphere as _standard_atmosphere
from aeris.core.units import as_float

# ---------------------------------------------------------------------------
# Physical constants / standard atmosphere (re-exported from constants)
# ---------------------------------------------------------------------------

# These are kept for backward compatibility - new code should use constants.py
RHO_SL = const.RHO_SL
MU_SL = const.MU_SL
T_SL = const.T_SL
P_SL = const.P_SL
LAPSE_RATE = const.LAPSE_RATE
GAS_CONST_AIR = const.R_GAS_AIR
G0 = const.G0


def standard_atmosphere(altitude_m: float):
    """Simplified ISA troposphere model (valid up to ~11 km).
    Returns (rho, mu, T, P). Good enough for low-altitude student flight
    testing; do not use for high-altitude/space work.
    
    DEPRECATED: Use Atmosphere.from_altitude() for new code.
    """
    return _standard_atmosphere(altitude_m)


def reynolds_number(rho: float, V: float, L: float, mu: float) -> float:
    """Reynolds number Re = rho*V*L/mu."""
    return rho * V * L / mu


def dynamic_pressure(rho: float, V: float) -> float:
    """q = 0.5 * rho * V^2."""
    return 0.5 * rho * V ** 2


def mach_number(V: float, a: float) -> float:
    """Mach number M = V / a."""
    return V / a


def speed_of_sound(T: float) -> float:
    """Speed of sound from temperature: a = sqrt(gamma * R * T)."""
    return np.sqrt(const.GAMMA_AIR * const.R_GAS_AIR * T)


# ---------------------------------------------------------------------------
# Thin airfoil theory (Anderson, "Fundamentals of Aerodynamics")
# ---------------------------------------------------------------------------

def naca4_camber_slope(m: float, p: float, x_over_c: np.ndarray) -> np.ndarray:
    """dz/dx (camber line slope) for a NACA 4-digit camber line.
    m = max camber (fraction of chord), p = location of max camber (fraction
    of chord, 0 < p < 1). Vectorized over x_over_c in [0,1].
    """
    x = np.asarray(x_over_c, dtype=float)
    dzdx = np.zeros_like(x)
    if m == 0 or p == 0:
        return dzdx  # symmetric airfoil, zero camber slope everywhere
    front = x < p
    back = ~front
    dzdx[front] = (2 * m / p ** 2) * (p - x[front])
    dzdx[back] = (2 * m / (1 - p) ** 2) * (p - x[back])
    return dzdx


def thin_airfoil_coeffs(m: float, p: float, n_theta: int = 200):
    """Compute the Fourier coefficients A0 offset term (alpha_L0), A1, A2 of
    thin airfoil theory for a NACA-4-digit-style camber line.

    Returns dict with:
      alpha_L0_rad : zero-lift angle of attack (radians)
      A1, A2       : Fourier coefficients (used for pitching moment)
    """
    theta = np.linspace(1e-6, np.pi - 1e-6, n_theta)
    x_over_c = 0.5 * (1 - np.cos(theta))
    dzdx = naca4_camber_slope(m, p, x_over_c)

    alpha_L0 = -(1.0 / np.pi) * np.trapezoid(dzdx * (np.cos(theta) - 1), theta)
    A1 = (2.0 / np.pi) * np.trapezoid(dzdx * np.cos(theta), theta)
    A2 = (2.0 / np.pi) * np.trapezoid(dzdx * np.cos(2 * theta), theta)
    return {"alpha_L0_rad": alpha_L0, "A1": A1, "A2": A2}


def thin_airfoil_cl(alpha_rad: float, alpha_L0_rad: float) -> float:
    """CL = 2*pi*(alpha - alpha_L0) — thin airfoil theory lift coefficient.
    
    VALIDITY: Incompressible flow (M < 0.3), small angles, 2D, inviscid.
    """
    return 2 * np.pi * (alpha_rad - alpha_L0_rad)


def thin_airfoil_cm_quarter_chord(A1: float, A2: float) -> float:
    """Pitching moment coefficient about the quarter-chord (independent of
    alpha per thin airfoil theory).
    """
    return (np.pi / 4.0) * (A2 - A1)


# ---------------------------------------------------------------------------
# Viscous / profile drag (flat-plate skin friction + form factor)
# ---------------------------------------------------------------------------

def skin_friction_coefficient(Re: float, transition_re: float = 5e5) -> float:
    """Flat-plate skin friction coefficient.
    Laminar (Blasius) below transition_re, turbulent (Schlichting) above.
    NOTE: real transition depends on surface roughness, turbulence
    intensity, and pressure gradient — this is a simplified switch, a prime
    candidate for calibration against your own wind-tunnel data.
    
    VALIDITY: Re > 1e5 for turbulent correlation; transition_re is approximate.
    """
    Re = max(Re, 1.0)
    if Re < transition_re:
        return 1.328 / np.sqrt(Re)
    return 0.074 / Re ** 0.2


def form_factor_thickness(t_over_c: float) -> float:
    """Hoerner-style subsonic form factor for airfoil profile drag.
    FF = 1 + 2*(t/c) + 60*(t/c)^4
    """
    return 1.0 + 2.0 * t_over_c + 60.0 * t_over_c ** 4


def profile_drag_coefficient(Re: float, t_over_c: float) -> float:
    """CD0 referenced to planform area (both surfaces wetted, ~2x planform).
    
    VALIDITY: Subsonic, Re in calibrated range, no separation.
    """
    cf = skin_friction_coefficient(Re)
    ff = form_factor_thickness(t_over_c)
    return 2.0 * cf * ff


# ---------------------------------------------------------------------------
# Empirical stall / CLmax model
# ---------------------------------------------------------------------------

def empirical_cl_max(t_over_c: float, m_camber: float) -> float:
    """Simplified empirical correlation for CLmax of a 2D airfoil section.
    This is NOT a substitute for real stall physics (which needs boundary
    layer / viscous CFD or wind-tunnel data) — it's a documented placeholder
    meant to be replaced/calibrated once you have real test data. Trend:
    thicker + more cambered sections tend to stall at a higher CL, up to a
    point, then thickness beyond ~15% starts to hurt CLmax again.
    
    APPROXIMATION: No physics basis, purely empirical fit. 
    VALIDITY: t/c in [0.06, 0.18], m in [0, 0.06]. Extrapolation unreliable.
    """
    thickness_term = 0.9 + 4.0 * t_over_c - 8.0 * max(t_over_c - 0.15, 0) ** 2
    camber_term = 6.0 * m_camber
    return thickness_term + camber_term


def empirical_stall_alpha_deg(t_over_c: float, m_camber: float) -> float:
    """Simplified empirical stall angle of attack estimate (degrees),
    same caveats as empirical_cl_max.
    
    APPROXIMATION: No physics basis, purely empirical fit.
    VALIDITY: t/c in [0.06, 0.18], m in [0, 0.06]. Extrapolation unreliable.
    """
    return 10.0 + 20.0 * t_over_c + 15.0 * m_camber


# ---------------------------------------------------------------------------
# Biomimicry: tubercle (whale-fin) leading-edge effect model
# ---------------------------------------------------------------------------

def tubercle_effect(amplitude_ratio: float, wavelength_ratio: float):
    """Simplified, PARAMETRIZED model of tubercle effects on a wing/airfoil
    section, based on general trends reported in tubercle aerodynamics
    literature (delayed/softer stall, modest CLmax and stall-angle gains,
    small pre-stall CD penalty). The exact magnitudes are NOT validated —
    they exist so the optimizer has something to search over, and so YOU can
    calibrate the coefficients once you have your own wind-tunnel or
    glide-test data comparing a tuberculed vs. baseline leading edge. That
    calibration step is the actual scientific content of this module.

    amplitude_ratio  : tubercle peak-to-trough amplitude / local chord
    wavelength_ratio : tubercle wavelength / local chord

    Returns dict: d_clmax, d_stall_alpha_deg, d_cd0
    
    APPROXIMATION: Parametrized model with unvalidated constants (k_amp=40, 
    k_wl_center=0.25, k_wl_width=0.20). CALIBRATION REQUIRED.
    """
    amplitude_ratio = max(amplitude_ratio, 0.0)
    wavelength_ratio = max(wavelength_ratio, 1e-3)

    # Effect peaks at a moderate amplitude and a wavelength around ~0.25c,
    # loosely following reported trends; tunable constants below.
    k_amp = 40.0
    k_wl_center = 0.25
    k_wl_width = 0.20

    wl_factor = np.exp(-((wavelength_ratio - k_wl_center) ** 2) / (2 * k_wl_width ** 2))
    d_stall_alpha_deg = k_amp * amplitude_ratio * wl_factor
    d_clmax = 0.5 * d_stall_alpha_deg / 10.0        # modest CLmax gain, tied to stall delay
    d_cd0 = 0.15 * amplitude_ratio                   # small pre-stall drag penalty

    return {
        "d_clmax": d_clmax,
        "d_stall_alpha_deg": d_stall_alpha_deg,
        "d_cd0": d_cd0,
    }


# ---------------------------------------------------------------------------
# Winglets: effective span / induced-drag reduction model
# ---------------------------------------------------------------------------

def winglet_effective_AR(AR: float, height_ratio: float, cant_deg: float,
                          k_wl: float = 1.9) -> float:
    """Simplified effective-aspect-ratio model for a winglet's induced drag
    benefit (loosely following the classic result that winglet height acts
    like a fraction of extra span, discounted by cant angle away from
    vertical). height_ratio = winglet height / (semispan). This is an
    approximation — real winglet benefit depends strongly on load
    distribution and shape, and should be checked against tow-test data.
    
    APPROXIMATION: k_wl=1.9 is empirical. Real benefit depends on load distribution.
    """
    cant_rad = np.radians(cant_deg)
    span_efficiency_gain = k_wl * height_ratio * np.cos(cant_rad)
    return AR * (1.0 + span_efficiency_gain)


def oswald_efficiency(AR: float, sweep_deg: float) -> float:
    """Oswald span efficiency factor estimate (Raymer's correlations).
    Straight/low-sweep wings use one fit; swept wings (>30 deg) use another.
    """
    if sweep_deg <= 30.0:
        e = 1.78 * (1 - 0.045 * AR ** 0.68) - 0.64
    else:
        sweep_rad = np.radians(sweep_deg)
        e = 4.61 * (1 - 0.045 * AR ** 0.68) * (np.cos(sweep_rad) ** 0.15) - 3.1
    return float(np.clip(e, 0.5, 0.98))


def induced_drag_coefficient(CL: float, AR_eff: float, e: float) -> float:
    """CDi = CL^2 / (pi * e * AR_eff)."""
    return CL ** 2 / (np.pi * e * AR_eff)


# ---------------------------------------------------------------------------
# Polar generation utilities
# ---------------------------------------------------------------------------

def polar_cl_alpha(m: float, p: float, t: float, 
                   alpha_min: float = -5.0, alpha_max: float = None,
                   n_points: int = 50):
    """Generate CL-alpha polar for an airfoil.
    
    If alpha_max is None, uses stall_alpha + 10 degrees.
    Returns dict with alpha_deg, CL, CL_max, stall_alpha_deg, alpha_L0_deg.
    """
    coeffs = thin_airfoil_coeffs(m, p)
    alpha_L0 = coeffs["alpha_L0_rad"]
    cl_max = empirical_cl_max(t, m)
    stall_alpha = empirical_stall_alpha_deg(t, m)
    
    if alpha_max is None:
        alpha_max = stall_alpha + 10.0
    
    alphas = np.linspace(alpha_min, alpha_max, n_points)
    cl = np.zeros_like(alphas)
    for i, alpha in enumerate(alphas):
        alpha_rad = np.radians(alpha)
        cl_i = thin_airfoil_cl(alpha_rad, alpha_L0)
        cl[i] = min(cl_i, cl_max) if alpha <= stall_alpha else cl_max * 0.6
    
    return {
        "alpha_deg": alphas.tolist(),
        "CL": cl.tolist(),
        "CL_max": cl_max,
        "stall_alpha_deg": stall_alpha,
        "alpha_L0_deg": np.degrees(alpha_L0),
    }


def polar_cl_cd_cm(m: float, p: float, t: float, Re: float,
                   alpha_min: float = -5.0, alpha_max: float = None,
                   n_points: int = 50):
    """Generate comprehensive polar: CL, CD, Cm, L/D vs alpha.
    
    Uses thin airfoil theory for CL and Cm, empirical models for CLmax/stall,
    and skin friction + form factor for CD0.
    
    Returns dict with:
        alpha_deg: angle of attack array
        CL: lift coefficient array
        CD: drag coefficient array
        Cm_c4: pitching moment coefficient array (constant per thin airfoil theory)
        LD: lift-to-drag ratio array
        CL_max: maximum lift coefficient
        stall_alpha_deg: stall angle
        alpha_L0_deg: zero-lift angle
        best_LD: maximum L/D
        best_LD_alpha: alpha at max L/D
        CD0: profile drag coefficient
    """
    coeffs = thin_airfoil_coeffs(m, p)
    alpha_L0 = coeffs["alpha_L0_rad"]
    cl_max = empirical_cl_max(t, m)
    stall_alpha = empirical_stall_alpha_deg(t, m)
    cm_c4 = thin_airfoil_cm_quarter_chord(coeffs["A1"], coeffs["A2"])
    
    if alpha_max is None:
        alpha_max = stall_alpha + 10.0
    
    alphas = np.linspace(alpha_min, alpha_max, n_points)
    
    CL = np.zeros_like(alphas)
    CD = np.zeros_like(alphas)
    CM = np.full_like(alphas, cm_c4)
    LD = np.zeros_like(alphas)
    
    cd0 = profile_drag_coefficient(Re, t)
    
    for i, alpha in enumerate(alphas):
        alpha_rad = np.radians(alpha)
        cl_i = thin_airfoil_cl(alpha_rad, alpha_L0)
        
        if alpha <= stall_alpha:
            cl_eff = min(cl_i, cl_max)
            cd_eff = cd0
        else:
            cl_eff = cl_max * 0.6
            cd_eff = cd0 * 3.0  # crude post-stall
        
        CL[i] = cl_eff
        CD[i] = cd_eff
        LD[i] = cl_eff / cd_eff if cd_eff > 0 else 0.0
    
    # Find best L/D (pre-stall only)
    prestall_mask = np.array(alphas) <= stall_alpha
    if np.any(prestall_mask):
        ld_prestall = LD[prestall_mask]
        best_ld_idx = np.argmax(ld_prestall)
        best_LD = ld_prestall[best_ld_idx]
        best_LD_alpha = alphas[prestall_mask][best_ld_idx]
    else:
        best_LD = 0.0
        best_LD_alpha = 0.0
    
    return {
        "alpha_deg": alphas.tolist(),
        "CL": CL.tolist(),
        "CD": CD.tolist(),
        "Cm_c4": CM.tolist(),
        "LD": LD.tolist(),
        "CL_max": cl_max,
        "stall_alpha_deg": stall_alpha,
        "alpha_L0_deg": np.degrees(alpha_L0),
        "best_LD": best_LD,
        "best_LD_alpha": best_LD_alpha,
        "CD0": cd0,
    }


def polar_cl_alpha_legacy(m: float, p: float, t: float, n_points: int = 20):
    """Generate CL-alpha polar for an airfoil (legacy interface).
    
    DEPRECATED: Use polar_cl_alpha with explicit alpha_min/alpha_max.
    """
    return polar_cl_alpha(m, p, t, -5.0, None, n_points)


def polar_cl_cd_legacy(m: float, p: float, t: float, Re: float, n_points: int = 20):
    """Generate CL-CD polar for an airfoil (legacy interface).
    
    DEPRECATED: Use polar_cl_cd_cm for comprehensive polar.
    """
    polar = polar_cl_cd_cm(m, p, t, Re, -5.0, None, n_points)
    return {
        "alpha_deg": polar["alpha_deg"],
        "CL": polar["CL"],
        "CD": polar["CD"],
        "CL_max": polar["CL_max"],
        "stall_alpha_deg": polar["stall_alpha_deg"],
    }


# ---------------------------------------------------------------------------
# Baseline comparison utilities
# ---------------------------------------------------------------------------

def create_baseline_airfoil():
    """Create a standard baseline airfoil for comparison.
    
    Returns dict with baseline design variables:
        camber_m=0.02, camber_pos_p=0.4, thickness_t=0.12
        (NACA 2412-like, common general aviation airfoil)
    """
    return {
        "camber_m": 0.02,
        "camber_pos_p": 0.4,
        "thickness_t": 0.12,
    }


def compare_airfoil_to_baseline(polar_opt, polar_base, metric: str = "LD"):
    """Compare optimized airfoil polar to baseline.
    
    Args:
        polar_opt: polar dict from optimized design
        polar_base: polar dict from baseline design
        metric: which metric to compare ("LD", "CL", "CD", "Cm_c4")
    
    Returns dict with:
        opt_value: optimized metric value
        base_value: baseline metric value
        improvement: opt_value - base_value
        improvement_pct: percentage improvement
    """
    if metric == "LD":
        opt_val = polar_opt.get("best_LD", 0.0)
        base_val = polar_base.get("best_LD", 0.0)
    elif metric == "CL":
        opt_val = polar_opt.get("CL_max", 0.0)
        base_val = polar_base.get("CL_max", 0.0)
    elif metric == "CD":
        opt_val = polar_opt.get("CD0", 0.0)
        base_val = polar_base.get("CD0", 0.0)
    elif metric == "Cm_c4":
        opt_val = polar_opt.get("Cm_c4", [0.0])[0]
        base_val = polar_base.get("Cm_c4", [0.0])[0]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    if base_val == 0:
        improvement_pct = 0.0
    else:
        improvement_pct = 100.0 * (opt_val - base_val) / abs(base_val)
    
    return {
        "opt_value": opt_val,
        "base_value": base_val,
        "improvement": opt_val - base_val,
        "improvement_pct": improvement_pct,
    }