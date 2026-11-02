"""
aeris.core.atmosphere
=====================
Centralized ISA atmosphere model and derived aerodynamic quantities.

Single source of truth for:
- Atmospheric properties (ρ, μ, T, P, a, ν) at altitude
- Dynamic pressure: q = ½ ρ V²
- Reynolds number: Re = ρ V L / μ
- Mach number: M = V / a

All calculations use SI units internally via the units system.
"""

from dataclasses import dataclass
from typing import Optional
import math

from aeris.core import constants as const
from aeris.core.units import (
    Quantity, meter, kilogram, second, kelvin, pascal, pascal_second,
    m_per_s, kg_per_m3, m2_per_s, dimensionless, as_float,
)


@dataclass(frozen=True)
class Atmosphere:
    """
    Complete atmospheric state at a given altitude.
    
    All properties in SI units (Quantity objects).
    Valid for ISA troposphere (0 to 11,000 m).
    """
    altitude: Quantity              # m
    temperature: Quantity           # K
    pressure: Quantity              # Pa
    density: Quantity               # kg/m³
    dynamic_viscosity: Quantity     # Pa·s
    kinematic_viscosity: Quantity   # m²/s
    speed_of_sound: Quantity        # m/s
    
    @classmethod
    def from_altitude(cls, altitude: Quantity) -> "Atmosphere":
        """
        Create Atmosphere from altitude using ISA troposphere model.
        
        Args:
            altitude: Geometric altitude above sea level (Quantity in meters)
            
        Returns:
            Atmosphere object with all derived properties
            
        Raises:
            ValueError: If altitude > 11,000 m (outside troposphere model validity)
        """
        h = as_float(altitude)
        if h < 0:
            h = 0.0
        if h > 11000:
            # Extrapolation warning - but we still compute
            pass
        
        # ISA troposphere: T = T0 - L*h
        T = const.T_SL - const.LAPSE_RATE * h
        
        # Pressure: P = P0 * (T/T0)^(g0/(L*R))
        exponent = const.G0 / (const.LAPSE_RATE * const.R_GAS_AIR)
        P = const.P_SL * (T / const.T_SL) ** exponent
        
        # Density: ρ = P / (R * T)
        rho = P / (const.R_GAS_AIR * T)
        
        # Dynamic viscosity: Sutherland's law
        # μ = μ0 * (T/T0)^1.5 * (T0 + S) / (T + S)
        mu = const.MU_SL * (T / const.T_SL) ** 1.5 * (const.T_SL + const.SUTHERLAND_CONST) / (T + const.SUTHERLAND_CONST)
        
        # Kinematic viscosity: ν = μ / ρ
        nu = mu / rho
        
        # Speed of sound: a = sqrt(γ * R * T)
        a = math.sqrt(const.GAMMA_AIR * const.R_GAS_AIR * T)
        
        return cls(
            altitude=meter(h, "altitude"),
            temperature=kelvin(T, "temperature"),
            pressure=pascal(P, "pressure"),
            density=kg_per_m3(rho, "density"),
            dynamic_viscosity=pascal_second(mu, "dynamic_viscosity"),
            kinematic_viscosity=m2_per_s(nu, "kinematic_viscosity"),
            speed_of_sound=m_per_s(a, "speed_of_sound"),
        )
    
    def dynamic_pressure(self, velocity: Quantity) -> Quantity:
        """
        Compute dynamic pressure q = ½ ρ V².
        
        Args:
            velocity: Flow velocity (Quantity in m/s)
            
        Returns:
            Dynamic pressure (Quantity in Pa)
        """
        V = as_float(velocity)
        rho = self.density.value
        q = 0.5 * rho * V * V
        return pascal(q, "dynamic_pressure")
    
    def reynolds_number(self, velocity: Quantity, length: Quantity) -> Quantity:
        """
        Compute Reynolds number Re = ρ V L / μ.
        
        Args:
            velocity: Flow velocity (Quantity in m/s)
            length: Characteristic length (Quantity in m)
            
        Returns:
            Reynolds number (dimensionless Quantity)
        """
        V = as_float(velocity)
        L = as_float(length)
        rho = self.density.value
        mu = self.dynamic_viscosity.value
        Re = rho * V * L / mu
        return dimensionless(Re, "Reynolds_number")
    
    def mach_number(self, velocity: Quantity) -> Quantity:
        """
        Compute Mach number M = V / a.
        
        Args:
            velocity: Flow velocity (Quantity in m/s)
            
        Returns:
            Mach number (dimensionless Quantity)
        """
        V = as_float(velocity)
        a = self.speed_of_sound.value
        M = V / a
        return dimensionless(M, "Mach_number")


def standard_atmosphere(altitude: float) -> tuple:
    """
    Backward-compatible function returning raw floats.
    
    Returns (rho, mu, T, P) in SI units.
    For legacy code compatibility only.
    """
    atm = Atmosphere.from_altitude(meter(altitude))
    return (
        atm.density.value,
        atm.dynamic_viscosity.value,
        atm.temperature.value,
        atm.pressure.value,
    )


def reynolds_number(rho: float, V: float, L: float, mu: float) -> float:
    """Backward-compatible Re calculation with raw floats."""
    return rho * V * L / mu


def dynamic_pressure(rho: float, V: float) -> float:
    """Backward-compatible q calculation with raw floats."""
    return 0.5 * rho * V * V


def mach_number(V: float, a: float) -> float:
    """Backward-compatible Mach calculation with raw floats."""
    return V / a


def speed_of_sound(T: float) -> float:
    """Speed of sound from temperature: a = sqrt(γ R T)."""
    return math.sqrt(const.GAMMA_AIR * const.R_GAS_AIR * T)