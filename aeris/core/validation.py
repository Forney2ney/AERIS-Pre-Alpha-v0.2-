"""
aeris.core.validation
=====================
Input validation for design variables and geometry.

Provides clear, actionable error messages instead of cryptic optimizer failures.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from aeris.core.exceptions import ValidationError
from aeris.core import constants as const


@dataclass
class ValidationResult:
    """Result of a validation check."""
    field: str
    message: str
    severity: str = "error"  # "error", "warning"
    value: any = None


def validate_bounds(var_names: List[str], values: List[float], 
                    bounds: List[Tuple[float, float]]) -> List[ValidationResult]:
    """
    Validate that all values are within their bounds.
    
    Args:
        var_names: List of variable names
        values: List of values to check
        bounds: List of (min, max) tuples
        
    Returns:
        List of ValidationResult (empty if all valid)
    """
    results = []
    for name, value, (min_val, max_val) in zip(var_names, values, bounds):
        if value < min_val:
            results.append(ValidationResult(
                field=name,
                message=f"Value {value} below minimum {min_val}",
                severity="error",
                value=value
            ))
        elif value > max_val:
            results.append(ValidationResult(
                field=name,
                message=f"Value {value} above maximum {max_val}",
                severity="error",
                value=value
            ))
    return results


def validate_airfoil_vars(var_names: List[str], values: List[float]) -> List[ValidationResult]:
    """
    Validate airfoil design variables.
    
    Expected order: camber_m, camber_pos_p, thickness_t, [tubercle_amplitude, tubercle_wavelength]
    """
    results = []
    bounds = [
        (0.0, 0.06),      # camber_m
        (0.2, 0.6),       # camber_pos_p
        (0.06, 0.18),     # thickness_t
    ]
    
    # Add biomimicry bounds if present
    if len(values) > 3:
        bounds.extend([
            (0.0, 0.05),      # tubercle_amplitude
            (0.10, 0.50),     # tubercle_wavelength
        ])
    
    results.extend(validate_bounds(var_names, values, bounds))
    
    # Cross-variable checks
    if len(values) >= 3:
        m, p, t = values[0], values[1], values[2]
        
        # Camber position must be reasonable for given camber
        if m > 0 and (p < 0.15 or p > 0.6):
            results.append(ValidationResult(
                field="camber_pos_p",
                message=f"Camber position {p} unrealistic for camber {m}",
                severity="warning",
                value=p
            ))
        
        # Thickness must be sufficient for camber
        if t < m * 2:
            results.append(ValidationResult(
                field="thickness_t",
                message=f"Thickness {t} may be insufficient for camber {m}",
                severity="warning",
                value=t
            ))
    
    return results


def validate_winglet_vars(var_names: List[str], values: List[float]) -> List[ValidationResult]:
    """
    Validate winglet design variables.
    
    Expected order: height_ratio, cant_deg, taper_ratio
    """
    results = []
    bounds = [
        (0.0, 0.25),      # height_ratio
        (0.0, 75.0),      # cant_deg
        (0.2, 1.0),       # taper_ratio
    ]
    
    results.extend(validate_bounds(var_names, values, bounds))
    
    if len(values) >= 3:
        height_ratio, cant_deg, taper_ratio = values[0], values[1], values[2]
        
        # Warn if winglet is very small
        if height_ratio < 0.02:
            results.append(ValidationResult(
                field="height_ratio",
                message=f"Winglet height ratio {height_ratio} very small - negligible effect",
                severity="warning",
                value=height_ratio
            ))
        
        # Warn if cant angle is extreme
        if cant_deg > 60:
            results.append(ValidationResult(
                field="cant_deg",
                message=f"Cant angle {cant_deg}° extreme - structural concerns",
                severity="warning",
                value=cant_deg
            ))
    
    return results


def validate_aircraft_vars(var_names: List[str], values: List[float]) -> List[ValidationResult]:
    """
    Validate aircraft design variables.
    
    Expected order: span_m, root_chord_m, taper_ratio, sweep_deg, washout_deg,
                    root_camber_m, tip_camber_m, winglet_height_ratio, winglet_cant_deg, alpha_deg
    """
    results = []
    bounds = [
        (0.8, 1.8),       # span_m
        (0.15, 0.45),     # root_chord_m
        (0.3, 1.0),       # taper_ratio
        (0.0, 35.0),      # sweep_deg
        (0.0, 10.0),      # washout_deg
        (0.0, 0.06),      # root_camber_m
        (0.0, 0.06),      # tip_camber_m
        (0.0, 0.25),      # winglet_height_ratio
        (0.0, 60.0),      # winglet_cant_deg
        (0.0, 10.0),      # alpha_deg
    ]
    
    results.extend(validate_bounds(var_names, values, bounds))
    
    if len(values) >= 10:
        span, root_chord, taper, sweep, washout = values[0], values[1], values[2], values[3], values[4]
        root_camber, tip_camber = values[5], values[6]
        wl_height, wl_cant, alpha = values[7], values[8], values[9]
        
        tip_chord = root_chord * taper
        
        # Tip chord check
        if tip_chord < const.MIN_TIP_CHORD_RATIO * root_chord:
            results.append(ValidationResult(
                field="taper_ratio",
                message=f"Tip chord {tip_chord:.3f}m below minimum ({const.MIN_TIP_CHORD_RATIO} * root)",
                severity="error",
                value=taper
            ))
        
        # Sweep vs aspect ratio check
        area = span * (root_chord + tip_chord) / 2.0
        AR = span * span / area
        if sweep > 30 and AR > 12:
            results.append(ValidationResult(
                field="sweep_deg",
                message=f"High sweep ({sweep}°) with high AR ({AR:.1f}) - structural weight penalty",
                severity="warning",
                value=sweep
            ))
        
        # Washout check
        if washout > 8:
            results.append(ValidationResult(
                field="washout_deg",
                message=f"Washout {washout}° large - may indicate design issue",
                severity="warning",
                value=washout
            ))
        
        # Angle of attack check
        if alpha > 8:
            results.append(ValidationResult(
                field="alpha_deg",
                message=f"Operating AoA {alpha}° high - near stall for typical airfoils",
                severity="warning",
                value=alpha
            ))
    
    return results


def validate_geometry(
    tip_chord: float, 
    root_chord: float, 
    thickness: float,
    span: Optional[float] = None
) -> List[ValidationResult]:
    """
    Validate geometric constraints.
    
    Args:
        tip_chord: Tip chord length (m)
        root_chord: Root chord length (m)
        thickness: Thickness ratio (t/c)
        span: Wing span (m), optional
        
    Returns:
        List of ValidationResult
    """
    results = []
    
    if tip_chord <= 0:
        results.append(ValidationResult(
            field="tip_chord",
            message=f"Tip chord must be positive, got {tip_chord}",
            severity="error",
            value=tip_chord
        ))
    
    if root_chord <= 0:
        results.append(ValidationResult(
            field="root_chord",
            message=f"Root chord must be positive, got {root_chord}",
            severity="error",
            value=root_chord
        ))
    
    if tip_chord > root_chord:
        results.append(ValidationResult(
            field="taper_ratio",
            message=f"Tip chord ({tip_chord}) exceeds root chord ({root_chord})",
            severity="error",
            value=tip_chord / root_chord
        ))
    
    if thickness < const.MIN_THICKNESS_RATIO:
        results.append(ValidationResult(
            field="thickness",
            message=f"Thickness ratio {thickness} below minimum {const.MIN_THICKNESS_RATIO}",
            severity="error",
            value=thickness
        ))
    
    if span is not None and span <= 0:
        results.append(ValidationResult(
            field="span",
            message=f"Span must be positive, got {span}",
            severity="error",
            value=span
        ))
    
    return results


def raise_if_errors(results: List[ValidationResult]) -> None:
    """Raise ValidationError if any error-severity results exist."""
    errors = [r for r in results if r.severity == "error"]
    if errors:
        msg = "; ".join(f"{e.field}: {e.message}" for e in errors)
        raise ValidationError("validation", msg)