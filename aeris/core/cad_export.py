"""
aeris.core.cad_export
=======================
Turns AERIS geometry into files a 3D printer or CAD program can actually
open. Pure Python, no external CAD library required (writes DXF and STL by
hand in their plain-text formats) so this doesn't add anything to
requirements.txt.

Two output types:
  - DXF  : a flat 2D outline of an airfoil section (open it in FreeCAD/
           Fusion 360, sketch on top of it, or send it to a laser cutter).
  - STL  : a solid 3D shape ready to open directly in a slicer (Cura,
           PrusaSlicer, Bambu Studio) and 3D print.

Two solids are supported:
  - a simple EXTRUDED airfoil section (constant cross-section, given a
    span/thickness you choose) — good for a wind-tunnel test section.
  - a LOFTED wing half, built by connecting a sequence of station outlines
    along the span (used by the aircraft module, which already computes
    exactly this sequence of stations for its aerodynamics).

Known simplification: the end caps (root/tip of an extrusion, or root/tip
of a loft) are triangulated with a simple centroid fan. This is not a
proper ear-clipping triangulation, so for very thin/concave trailing edges
the cap can occasionally produce a sliver triangle. Always open the file in
your slicer and check it looks right / run "repair mesh" if your slicer
offers one before printing.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared: build a closed 2D outline from upper/lower surface coordinates
# ---------------------------------------------------------------------------

def outline_from_surfaces(xu, yu, xl, yl):
    """Combine upper+lower surface arrays (as returned by
    geometry.naca4_coordinates) into a single closed loop of (x, y) points,
    traversed trailing-edge -> upper surface -> leading edge -> lower
    surface -> trailing edge, with no duplicate leading-edge point.
    """
    xu, yu, xl, yl = map(np.asarray, (xu, yu, xl, yl))
    x = np.concatenate([xu[::-1], xl[1:]])
    y = np.concatenate([yu[::-1], yl[1:]])
    return list(zip(x.tolist(), y.tolist()))


# ---------------------------------------------------------------------------
# DXF export (2D outline only)
# ---------------------------------------------------------------------------

def write_dxf(points_xy, filename, layer="AIRFOIL"):
    """Write a closed 2D polygon as a minimal, broadly-compatible DXF
    (R12-style ENTITIES-only file made of LINE entities). Units are
    whatever units you passed the coordinates in (this project uses
    meters throughout — most CAD packages will ask you to confirm/rescale
    units on import, which is normal).
    """
    lines = ["0", "SECTION", "2", "ENTITIES"]
    n = len(points_xy)
    for i in range(n):
        x1, y1 = points_xy[i]
        x2, y2 = points_xy[(i + 1) % n]
        lines += [
            "0", "LINE",
            "8", layer,
            "10", f"{x1:.6f}", "20", f"{y1:.6f}", "30", "0.0",
            "11", f"{x2:.6f}", "21", f"{y2:.6f}", "31", "0.0",
        ]
    lines += ["0", "ENDSEC", "0", "EOF"]
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    return filename


# ---------------------------------------------------------------------------
# STL export (ASCII format — simplest to write correctly by hand)
# ---------------------------------------------------------------------------

def _facet(f, p1, p2, p3):
    """Write one STL triangle. Normal is left as (0,0,0) — this is valid
    STL; every slicer recomputes real normals on import anyway."""
    f.write("  facet normal 0 0 0\n    outer loop\n")
    for p in (p1, p2, p3):
        f.write(f"      vertex {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    f.write("    endloop\n  endfacet\n")


def _fan_cap(f, ring, z_or_const, axis, reverse=False):
    """Triangulate a ring of 2D points as a flat cap using a centroid fan.
    ring: list of (a, b) 2D points. axis: which 3D axis the cap sits on
    ('z' for extrusion caps, 'y' for spanwise loft caps).
    """
    ring = ring[:-1] if ring[0] == ring[-1] else ring
    cx = sum(p[0] for p in ring) / len(ring)
    cy = sum(p[1] for p in ring) / len(ring)
    n = len(ring)
    order = range(n) if not reverse else range(n - 1, -1, -1)
    order = list(order)
    for idx in range(n):
        a = ring[order[idx]]
        b = ring[order[(idx + 1) % n]]
        if axis == "z":
            p_c = (cx, cy, z_or_const)
            p_a = (a[0], a[1], z_or_const)
            p_b = (b[0], b[1], z_or_const)
        else:  # axis == 'y'
            p_c = (cx, z_or_const, cy)
            p_a = (a[0], z_or_const, a[1])
            p_b = (b[0], z_or_const, b[1])
        _facet(f, p_c, p_a, p_b)


def export_extruded_stl(points_xy, span_m, filename, solid_name="airfoil"):
    """Extrude a closed 2D outline (e.g. from outline_from_surfaces) into a
    straight 3D solid from z=0 to z=span_m. Good for printing a constant-
    section wind-tunnel test piece.
    """
    n = len(points_xy)
    with open(filename, "w") as f:
        f.write(f"solid {solid_name}\n")
        # side walls
        for i in range(n):
            x1, y1 = points_xy[i]
            x2, y2 = points_xy[(i + 1) % n]
            p1 = (x1, y1, 0.0)
            p2 = (x2, y2, 0.0)
            p3 = (x2, y2, span_m)
            p4 = (x1, y1, span_m)
            _facet(f, p1, p2, p3)
            _facet(f, p1, p3, p4)
        # end caps
        _fan_cap(f, points_xy, 0.0, axis="z", reverse=True)
        _fan_cap(f, points_xy, span_m, axis="z", reverse=False)
        f.write(f"endsolid {solid_name}\n")
    return filename


def export_wing_loft_stl(station_outlines, station_y, station_le_x, filename,
                          solid_name="wing"):
    """Build a 3D loft solid from a sequence of 2D station outlines placed
    along the span.

    station_outlines : list of closed 2D outlines (each a list of (x, y)
                        points, all with the SAME number of points and the
                        same point ordering — e.g. all built by
                        outline_from_surfaces with the same n_points)
    station_y         : list of spanwise position for each station (meters)
    station_le_x      : list of leading-edge x-offset for each station
                         (accounts for sweep) — added to every point's local
                         x before writing

    Global axes convention: X = chordwise, Y = spanwise, Z = thickness.
    """
    n_stations = len(station_outlines)
    n_pts = len(station_outlines[0])
    assert all(len(o) == n_pts for o in station_outlines), \
        "all stations must have the same number of outline points"

    def global_point(station_idx, pt_idx):
        x, z = station_outlines[station_idx][pt_idx]
        return (x + station_le_x[station_idx], station_y[station_idx], z)

    with open(filename, "w") as f:
        f.write(f"solid {solid_name}\n")
        # side surface: connect consecutive stations
        for s in range(n_stations - 1):
            for i in range(n_pts):
                j = (i + 1) % n_pts
                p1 = global_point(s, i)
                p2 = global_point(s, j)
                p3 = global_point(s + 1, j)
                p4 = global_point(s + 1, i)
                _facet(f, p1, p2, p3)
                _facet(f, p1, p3, p4)
        # root and tip caps
        root_ring_3d = [global_point(0, i)[::2] for i in range(n_pts)]  # (x,z)
        tip_ring_3d = [global_point(n_stations - 1, i)[::2] for i in range(n_pts)]
        _fan_cap(f, root_ring_3d, station_y[0], axis="y", reverse=True)
        _fan_cap(f, tip_ring_3d, station_y[-1], axis="y", reverse=False)
        f.write(f"endsolid {solid_name}\n")
    return filename
