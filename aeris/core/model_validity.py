"""
aeris.core.model_validity
=========================
Model validity checking engine.

Determines whether the current aerodynamic model is appropriate for the 
operating conditions. Each check produces VALID/WARNING/INVALID with 
explicit rationale.

This is the core of the "Physics Health" system - never silently produce
results when the model is outside its validated regime.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
from enum import Enum

from aeris.core.units import Quantity, as_float, dimensionless
from aeris.core import constants as const


class ValidityStatus(Enum):
    VALID = "VALID"
    WARNING = "WARNING"
    INVALID = "INVALID"


@dataclass
class ValidityCheck:
    """Result of a single validity check."""
    name: str
    status: ValidityStatus
    message: str
    threshold: Optional[float] = None
    actual: Optional[float] = None
    
    @property
    def icon(self) -> str:
        if self.status == ValidityStatus.VALID:
            return "✓"
        elif self.status == ValidityStatus.WARNING:
            return "⚠"
        return "✗"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "icon": self.icon,
            "threshold": self.threshold,
            "actual": self.actual,
        }


class ModelValidityEngine:
    """
    Centralized model validity checking.
    
    Checks:
    - Mach number regime (incompressible vs transonic vs supersonic)
    - Reynolds number range (laminar separation vs calibrated vs extrapolation)
    - Angle of attack vs stall (pre-stall vs near-stall vs post-stall)
    - Geometry validity (tip chord, thickness, etc.)
    """
    
    def __init__(self):
        # Mach thresholds (documented rationale)
        self.mach_incompressible = const.MACH_INCOMPRESSIBLE_LIMIT  # 0.3
        self.mach_transonic = const.MACH_TRANSONIC_LIMIT            # 0.7
        
        # Reynolds thresholds
        self.re_laminar_limit = const.RE_LAMINAR_SEPARATION_LIMIT   # 5e4
        self.re_upper_limit = const.RE_MODEL_UPPER_LIMIT            # 1e7
        
        # AoA margins (degrees)
        self.aoa_stall_margin_valid = 2.0    # >2° below stall = VALID
        self.aoa_stall_margin_warn = 0.0     # 0-2° below stall = WARNING
        
    def check_mach(self, mach: float) -> ValidityCheck:
        """
        Check Mach number validity for thin airfoil theory.
        
        Thin airfoil theory assumes incompressible flow (M < 0.3).
        Prandtl-Glauert correction extends to ~0.7.
        Above 0.7: transonic effects, shock formation - theory invalid.
        """
        if mach < self.mach_incompressible:
            return ValidityCheck(
                name="Mach Number",
                status=ValidityStatus.VALID,
                message=f"M = {mach:.3f} < {self.mach_incompressible}: Incompressible flow assumption valid",
                threshold=self.mach_incompressible,
                actual=mach
            )
        elif mach < self.mach_transonic:
            return ValidityCheck(
                name="Mach Number",
                status=ValidityStatus.WARNING,
                message=f"M = {mach:.3f} in [{self.mach_incompressible}, {self.mach_transonic}]: "
                        "Compressibility effects significant. Thin airfoil theory with "
                        "Prandtl-Glauert correction recommended but not implemented.",
                threshold=self.mach_transonic,
                actual=mach
            )
        else:
            return ValidityCheck(
                name="Mach Number",
                status=ValidityStatus.INVALID,
                message=f"M = {mach:.3f} >= {self.mach_transonic}: Transonic/supersonic regime. "
                        "Thin airfoil theory fundamentally invalid. Shock waves, "
                        "wave drag, and flow separation not modeled.",
                threshold=self.mach_transonic,
                actual=mach
            )
    
    def check_reynolds(self, re: float) -> ValidityCheck:
        """
        Check Reynolds number validity for skin friction/stall models.
        
        Below 5e4: Laminar separation bubbles dominate, empirical models unreliable.
        5e4 - 1e7: Calibrated range for skin friction + stall correlations.
        Above 1e7: Extrapolation beyond validated data.
        """
        if re < self.re_laminar_limit:
            return ValidityCheck(
                name="Reynolds Number",
                status=ValidityStatus.WARNING,
                message=f"Re = {re:.0f} < {self.re_laminar_limit:.0f}: "
                        "Low Reynolds regime. Laminar separation bubbles likely. "
                        "Skin friction and stall models not validated here.",
                threshold=self.re_laminar_limit,
                actual=re
            )
        elif re < self.re_upper_limit:
            return ValidityCheck(
                name="Reynolds Number",
                status=ValidityStatus.VALID,
                message=f"Re = {re:.0f} in [{self.re_laminar_limit:.0f}, {self.re_upper_limit:.0f}]: "
                        "Within calibrated model range.",
                threshold=self.re_upper_limit,
                actual=re
            )
        else:
            return ValidityCheck(
                name="Reynolds Number",
                status=ValidityStatus.WARNING,
                message=f"Re = {re:.0f} > {self.re_upper_limit:.0f}: "
                        "High Reynolds extrapolation. Turbulent skin friction "
                        "and stall models may not be accurate.",
                threshold=self.re_upper_limit,
                actual=re
            )
    
    def check_alpha_vs_stall(self, alpha_deg: float, stall_alpha_deg: float) -> ValidityCheck:
        """
        Check angle of attack relative to stall.
        
        More than 2° below stall: VALID
        0° to 2° below stall: WARNING
        At or above stall: INVALID (post-stall model is crude)
        """
        margin = stall_alpha_deg - alpha_deg
        
        if margin > self.aoa_stall_margin_valid:
            return ValidityCheck(
                name="Angle of Attack",
                status=ValidityStatus.VALID,
                message=f"α = {alpha_deg:.1f}°, stall ≈ {stall_alpha_deg:.1f}°: "
                        f"Margin {margin:.1f}° > {self.aoa_stall_margin_valid}°. Pre-stall, model valid.",
                threshold=stall_alpha_deg - self.aoa_stall_margin_valid,
                actual=alpha_deg
            )
        elif margin > self.aoa_stall_margin_warn:
            return ValidityCheck(
                name="Angle of Attack",
                status=ValidityStatus.WARNING,
                message=f"α = {alpha_deg:.1f}°, stall ≈ {stall_alpha_deg:.1f}°: "
                        f"Margin {margin:.1f}° ≤ {self.aoa_stall_margin_valid}°. Near stall - "
                        "nonlinear effects, model accuracy reduced.",
                threshold=stall_alpha_deg - self.aoa_stall_margin_warn,
                actual=alpha_deg
            )
        else:
            return ValidityCheck(
                name="Angle of Attack",
                status=ValidityStatus.INVALID,
                message=f"α = {alpha_deg:.1f}° >= stall ({stall_alpha_deg:.1f}°): "
                        "Post-stall regime. Current model uses crude 3x drag multiplier "
                        "and 60% lift retention - NOT physically accurate.",
                threshold=stall_alpha_deg,
                actual=alpha_deg
            )
    
    def check_geometry(
        self, 
        tip_chord: float, 
        root_chord: float, 
        thickness: float,
        aspect_ratio: Optional[float] = None
    ) -> List[ValidityCheck]:
        """Check geometric validity."""
        checks = []
        
        # Tip chord check
        if tip_chord <= 0:
            checks.append(ValidityCheck(
                name="Tip Chord",
                status=ValidityStatus.INVALID,
                message=f"Tip chord {tip_chord} must be positive",
                actual=tip_chord
            ))
        elif tip_chord < const.MIN_TIP_CHORD_RATIO * root_chord:
            checks.append(ValidityCheck(
                name="Tip Chord",
                status=ValidityStatus.WARNING,
                message=f"Tip/root ratio {tip_chord/root_chord:.3f} < {const.MIN_TIP_CHORD_RATIO}: "
                        "Very pointed tip - structural and manufacturing concerns",
                threshold=const.MIN_TIP_CHORD_RATIO,
                actual=tip_chord / root_chord
            ))
        else:
            checks.append(ValidityCheck(
                name="Tip Chord",
                status=ValidityStatus.VALID,
                message=f"Tip chord {tip_chord:.3f}m acceptable",
                actual=tip_chord
            ))
        
        # Thickness check
        if thickness < const.MIN_THICKNESS_RATIO:
            checks.append(ValidityCheck(
                name="Thickness Ratio",
                status=ValidityStatus.INVALID,
                message=f"t/c = {thickness:.3f} < {const.MIN_THICKNESS_RATIO}: "
                        "Too thin for structural integrity and model validity",
                threshold=const.MIN_THICKNESS_RATIO,
                actual=thickness
            ))
        elif thickness > 0.18:
            checks.append(ValidityCheck(
                name="Thickness Ratio",
                status=ValidityStatus.WARNING,
                message=f"t/c = {thickness:.3f} > 0.18: Very thick - "
                        "stall model and drag correlations less reliable",
                threshold=0.18,
                actual=thickness
            ))
        else:
            checks.append(ValidityCheck(
                name="Thickness Ratio",
                status=ValidityStatus.VALID,
                message=f"t/c = {thickness:.3f} within normal range",
                actual=thickness
            ))
        
        # Aspect ratio check
        if aspect_ratio is not None:
            if aspect_ratio < 4:
                checks.append(ValidityCheck(
                    name="Aspect Ratio",
                    status=ValidityStatus.WARNING,
                    message=f"AR = {aspect_ratio:.1f} < 4: Low aspect ratio - "
                            "3D effects significant, strip theory less accurate",
                    threshold=4.0,
                    actual=aspect_ratio
                ))
            elif aspect_ratio > 25:
                checks.append(ValidityCheck(
                    name="Aspect Ratio",
                    status=ValidityStatus.WARNING,
                    message=f"AR = {aspect_ratio:.1f} > 25: Very high aspect ratio - "
                            "structural flexibility and aeroelastic effects not modeled",
                    threshold=25.0,
                    actual=aspect_ratio
                ))
            else:
                checks.append(ValidityCheck(
                    name="Aspect Ratio",
                    status=ValidityStatus.VALID,
                    message=f"AR = {aspect_ratio:.1f} within normal range",
                    actual=aspect_ratio
                ))
        
        return checks
    
    def check_all(self, conditions: Dict[str, Any]) -> List[ValidityCheck]:
        """
        Run all applicable checks based on provided conditions.
        
        Args:
            conditions: Dict with optional keys:
                - mach: Mach number (float)
                - reynolds: Reynolds number (float)
                - alpha_deg: Angle of attack (deg)
                - stall_alpha_deg: Stall angle (deg)
                - tip_chord: Tip chord (m)
                - root_chord: Root chord (m)
                - thickness: Thickness ratio
                - aspect_ratio: Aspect ratio (optional)
                
        Returns:
            List of ValidityCheck objects
        """
        checks = []
        
        if "mach" in conditions and conditions["mach"] is not None:
            checks.append(self.check_mach(conditions["mach"]))
        
        if "reynolds" in conditions and conditions["reynolds"] is not None:
            checks.append(self.check_reynolds(conditions["reynolds"]))
        
        if "alpha_deg" in conditions and "stall_alpha_deg" in conditions:
            if conditions["alpha_deg"] is not None and conditions["stall_alpha_deg"] is not None:
                checks.append(self.check_alpha_vs_stall(
                    conditions["alpha_deg"], 
                    conditions["stall_alpha_deg"]
                ))
        
        # Geometry checks
        geo_keys = ["tip_chord", "root_chord", "thickness"]
        if all(k in conditions for k in geo_keys):
            aspect_ratio = conditions.get("aspect_ratio")
            checks.extend(self.check_geometry(
                conditions["tip_chord"],
                conditions["root_chord"],
                conditions["thickness"],
                aspect_ratio
            ))
        
        return checks
    
    def overall_status(self, checks: List[ValidityCheck]) -> ValidityStatus:
        """Determine overall validity from individual checks."""
        if any(c.status == ValidityStatus.INVALID for c in checks):
            return ValidityStatus.INVALID
        if any(c.status == ValidityStatus.WARNING for c in checks):
            return ValidityStatus.WARNING
        return ValidityStatus.VALID
    
    def summary(self, checks: List[ValidityCheck]) -> str:
        """Generate human-readable summary."""
        overall = self.overall_status(checks)
        lines = [f"MODEL VALIDITY: {overall.value}"]
        lines.append("")
        for check in checks:
            lines.append(f"  {check.icon} {check.name}: {check.message}")
        return "\n".join(lines)


# Convenience function for quick checks
def check_conditions(mach: float = None, reynolds: float = None, 
                     alpha_deg: float = None, stall_alpha_deg: float = None,
                     tip_chord: float = None, root_chord: float = None,
                     thickness: float = None, aspect_ratio: float = None) -> List[ValidityCheck]:
    """Quick validity check with keyword arguments."""
    engine = ModelValidityEngine()
    conditions = {}
    if mach is not None: conditions["mach"] = mach
    if reynolds is not None: conditions["reynolds"] = reynolds
    if alpha_deg is not None: conditions["alpha_deg"] = alpha_deg
    if stall_alpha_deg is not None: conditions["stall_alpha_deg"] = stall_alpha_deg
    if tip_chord is not None: conditions["tip_chord"] = tip_chord
    if root_chord is not None: conditions["root_chord"] = root_chord
    if thickness is not None: conditions["thickness"] = thickness
    if aspect_ratio is not None: conditions["aspect_ratio"] = aspect_ratio
    return engine.check_all(conditions)