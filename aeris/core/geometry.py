"""
aeris.core.geometry
====================
Pure geometry generation — no physics, no optimization. Produces coordinate
arrays that evaluator modules turn into performance numbers, and that a CAD
export step can turn into a DXF/STL file.
"""

import numpy as np


def naca4_thickness(t: float, x_over_c: np.ndarray) -> np.ndarray:
    """Half-thickness distribution for a NACA 4-digit section.
    t = max thickness as a fraction of chord (e.g. 0.12 for a 12% airfoil).
    """
    x = np.asarray(x_over_c, dtype=float)
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 - 0.1015 * x ** 4)


def naca4_camber(m: float, p: float, x_over_c: np.ndarray):
    """Camber line y and slope dy/dx for a NACA 4-digit section."""
    x = np.asarray(x_over_c, dtype=float)
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m == 0 or p == 0:
        return yc, dyc
    front = x < p
    back = ~front
    yc[front] = (m / p ** 2) * (2 * p * x[front] - x[front] ** 2)
    yc[back] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x[back] - x[back] ** 2)
    dyc[front] = (2 * m / p ** 2) * (p - x[front])
    dyc[back] = (2 * m / (1 - p) ** 2) * (p - x[back])
    return yc, dyc


def naca4_coordinates(m: float, p: float, t: float, chord: float = 1.0,
                       n_points: int = 100):
    """Full upper/lower surface coordinates for a NACA 4-digit airfoil,
    scaled to the given chord. Returns (x_upper, y_upper, x_lower, y_lower).
    Cosine-spaced points for good leading-edge resolution.
    """
    beta = np.linspace(0, np.pi, n_points)
    x_over_c = 0.5 * (1 - np.cos(beta))
    yt = naca4_thickness(t, x_over_c)
    yc, dyc = naca4_camber(m, p, x_over_c)
    theta = np.arctan(dyc)

    xu = (x_over_c - yt * np.sin(theta)) * chord
    yu = (yc + yt * np.cos(theta)) * chord
    xl = (x_over_c + yt * np.sin(theta)) * chord
    yl = (yc - yt * np.cos(theta)) * chord
    return xu, yu, xl, yl


def apply_tubercle_leading_edge(chord_at_station: float, amplitude_ratio: float,
                                 wavelength_ratio: float, span_position: float):
    """Returns a local leading-edge x-perturbation (in the same length units
    as chord) for a spanwise station, following a sinusoidal tubercle
    pattern. span_position: 0..1 fraction along the span (or along the
    tuberculed region of the span).
    amplitude_ratio/wavelength_ratio are fractions of local chord, matching
    the design variables used in the biomimicry knowledge-base model.
    """
    amplitude = amplitude_ratio * chord_at_station
    wavelength = max(wavelength_ratio, 1e-3)  # fraction of span-per-cycle
    return amplitude * np.sin(2 * np.pi * span_position / wavelength)


def generate_planform(span: float, root_chord: float, taper_ratio: float,
                       sweep_deg: float, n_stations: int = 21):
    """Simple trapezoidal (straight-taper) planform, one leading-edge sweep
    angle. Returns a dict of arrays over spanwise stations (semispan only,
    station 0 = root, station -1 = tip):
      y        : spanwise position (0..b/2)
      chord    : local chord
      le_x     : local leading-edge x offset due to sweep
    """
    tip_chord = root_chord * taper_ratio
    semispan = span / 2.0
    y = np.linspace(0, semispan, n_stations)
    frac = y / semispan
    chord = root_chord + (tip_chord - root_chord) * frac
    le_x = y * np.tan(np.radians(sweep_deg))
    return {"y": y, "chord": chord, "le_x": le_x, "frac": frac,
            "tip_chord": tip_chord, "semispan": semispan}


def rotate_points(x, z, pivot_x: float, pivot_z: float, angle_deg: float):
    """Rotate 2D points (x, z) about a pivot by angle_deg (degrees).
    Used to apply spanwise twist to a wing section before placing it at its
    spanwise station for CAD export.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    theta = np.radians(angle_deg)
    dx = x - pivot_x
    dz = z - pivot_z
    x_new = pivot_x + dx * np.cos(theta) + dz * np.sin(theta)
    z_new = pivot_z - dx * np.sin(theta) + dz * np.cos(theta)
    return x_new, z_new


def wing_area_and_AR(span: float, root_chord: float, taper_ratio: float) -> tuple:
    """Planform area (trapezoidal) and aspect ratio."""
    tip_chord = root_chord * taper_ratio
    area = span * (root_chord + tip_chord) / 2.0
    AR = span ** 2 / area
    return area, AR
