"""
aeris.core.requirements
=======================
Mission requirements → derived operating conditions.

This module translates high-level mission requirements (mass, speed, altitude, etc.)
into the aerodynamic quantities needed for design (q, Re, Mach, required CL, etc.).

This is the "requirements engine" - users specify WHAT they want to achieve,
the system derives the operating conditions.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from aeris.core.atmosphere import Atmosphere
from aeris.core.units import (
    Quantity, meter, kilogram, second, kelvin, pascal, newton,
    m_per_s, kg_per_m3, dimensionless, as_float,
)
from aeris.core import constants as const


@dataclass(frozen=True)
class Requirements:
    """
    High-level mission requirements.
    
    Users specify these directly. All derived quantities are computed
    from these using standard atmospheric physics.
    """
    # Required inputs
    mass: Quantity                      # kg - total aircraft mass
    cruise_speed: Quantity              # m/s - cruise velocity
    altitude: Quantity                  # m - cruise altitude
    max_span: Quantity                  # m - maximum wing span constraint
    
    # Optional inputs with defaults
    endurance: Quantity = field(default_factory=lambda: second(0))  # s - required endurance
    stall_speed: Optional[Quantity] = None      # m/s - stall speed requirement
    payload_mass: Optional[Quantity] = None     # kg - payload mass
    
    # Derived properties (computed on access)
    @property
    def weight(self) -> Quantity:
        """Weight = mass * g0 (N)."""
        return newton(as_float(self.mass) * const.G0, "weight")
    
    @property
    def atmosphere(self) -> Atmosphere:
        """Atmospheric conditions at cruise altitude."""
        return Atmosphere.from_altitude(self.altitude)
    
    @property
    def dynamic_pressure(self) -> Quantity:
        """Dynamic pressure at cruise: q = ½ ρ V²."""
        return self.atmosphere.dynamic_pressure(self.cruise_speed)
    
    @property
    def mach_number(self) -> Quantity:
        """Mach number at cruise: M = V / a."""
        return self.atmosphere.mach_number(self.cruise_speed)
    
    def reynolds_number(self, length: Quantity) -> Quantity:
        """Reynolds number at cruise for given characteristic length."""
        return self.atmosphere.reynolds_number(self.cruise_speed, length)
    
    @property
    def speed_of_sound(self) -> Quantity:
        """Speed of sound at cruise altitude."""
        return self.atmosphere.speed_of_sound
    
    def required_lift_coefficient(self, wing_area: Quantity) -> Quantity:
        """
        Required CL for level flight: CL = W / (q * S).
        
        Args:
            wing_area: Reference wing area (m²)
            
        Returns:
            Required lift coefficient (dimensionless)
        """
        W = as_float(self.weight)
        q = as_float(self.dynamic_pressure)
        S = as_float(wing_area)
        if q * S == 0:
            raise ValueError("Dynamic pressure or wing area cannot be zero")
        CL = W / (q * S)
        return dimensionless(CL, "required_CL")
    
    def wing_loading(self, wing_area: Quantity) -> Quantity:
        """Wing loading: W / S (N/m² = Pa)."""
        W = as_float(self.weight)
        S = as_float(wing_area)
        if S == 0:
            raise ValueError("Wing area cannot be zero")
        return pascal(W / S, "wing_loading")
    
    def aspect_ratio_bounds(self, wing_area: Quantity) -> Tuple[float, float]:
        """
        Aspect ratio bounds from span constraint.
        
        AR = b² / S, so for a given max span and area:
        AR_max = b_max² / S
        
        Returns (min_AR, max_AR) where min_AR is a reasonable lower bound.
        """
        b_max = as_float(self.max_span)
        S = as_float(wing_area)
        if S <= 0:
            raise ValueError("Wing area must be positive")
        
        AR_max = b_max * b_max / S
        # Reasonable minimum AR for fixed-wing aircraft
        AR_min = 4.0
        return (AR_min, AR_max)
    
    def estimate_wing_area(self, target_CL: float = 0.5) -> Quantity:
        """
        Estimate required wing area for a target CL at cruise.
        
        S = W / (q * CL_target)
        
        Args:
            target_CL: Target lift coefficient (default 0.5, typical cruise)
            
        Returns:
            Estimated wing area (m²)
        """
        W = as_float(self.weight)
        q = as_float(self.dynamic_pressure)
        if q * target_CL == 0:
            raise ValueError("Dynamic pressure or target CL cannot be zero")
        S = W / (q * target_CL)
        # Use a Quantity with a generic area unit
        return Quantity(S, "m2", "wing_area")
    
    def estimate_wing_area_m2(self, target_CL: float = 0.5) -> float:
        """Estimate wing area in m² (raw float for convenience)."""
        W = as_float(self.weight)
        q = as_float(self.dynamic_pressure)
        return W / (q * target_CL)


def create_requirements_from_mission(
    mass_kg: float,
    cruise_speed_mps: float,
    altitude_m: float,
    max_span_m: float,
    endurance_min: float = 0,
    stall_speed_mps: Optional[float] = None,
    payload_mass_kg: Optional[float] = None,
) -> Requirements:
    """
    Convenience function to create Requirements from raw values.
    
    All inputs in SI units (or minutes for endurance).
    """
    return Requirements(
        mass=kilogram(mass_kg, "mass"),
        cruise_speed=m_per_s(cruise_speed_mps, "cruise_speed"),
        altitude=meter(altitude_m, "altitude"),
        max_span=meter(max_span_m, "max_span"),
        endurance=second(endurance_min * 60.0, "endurance"),
        stall_speed=m_per_s(stall_speed_mps, "stall_speed") if stall_speed_mps else None,
        payload_mass=kilogram(payload_mass_kg, "payload_mass") if payload_mass_kg else None,
    )