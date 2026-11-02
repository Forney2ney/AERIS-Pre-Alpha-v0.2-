"""
aeris.core.units
================
Lightweight dimensional quantity system for AERIS.

Design principles:
- SI units internally (canonical representation)
- Explicit units on every physical quantity
- Dimensional checking on arithmetic operations
- No silent conversion between incompatible dimensions
- Lightweight, no external dependencies (replaceable with pint later)
- Clear error messages for common mistakes (kg vs lbf, m vs ft)

Base dimensions tracked: L (length), M (mass), T (time), Θ (temperature)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union
import math


# Dimension vectors as (L, M, T, Θ) exponents
DIMENSIONS: Dict[str, tuple] = {
    # Base units
    "m": (1, 0, 0, 0),
    "kg": (0, 1, 0, 0),
    "s": (0, 0, 1, 0),
    "K": (0, 0, 0, 1),
    # Derived units
    "m/s": (1, 0, -1, 0),
    "m/s2": (1, 0, -2, 0),
    "m2": (2, 0, 0, 0),             # area
    "m2/s2": (2, 0, -2, 0),         # velocity squared
    "m3": (3, 0, 0, 0),             # volume
    "kg/m3": (-3, 1, 0, 0),
    "kg/m2": (-2, 1, 0, 0),         # mass per area
    "kg/m2/s": (-2, 1, -1, 0),      # mass flux (rho * V)
    "Pa": (-1, 1, -2, 0),           # N/m^2 = kg/(m·s^2)
    "N": (1, 1, -2, 0),             # kg·m/s^2
    "J": (2, 1, -2, 0),             # N·m = kg·m^2/s^2
    "W": (2, 1, -3, 0),             # J/s = kg·m^2/s^3
    "Pa_s": (-1, 1, -1, 0),         # Pa·s = kg/(m·s)
    "m2/s": (2, 0, -1, 0),          # kinematic viscosity
    "1/s": (0, 0, -1, 0),           # frequency
    "Hz": (0, 0, -1, 0),            # frequency
    "rad": (0, 0, 0, 0),            # dimensionless
    "deg": (0, 0, 0, 0),            # dimensionless
    "1": (0, 0, 0, 0),              # dimensionless
    # Common non-SI (for conversion only, never canonical)
    "ft": (1, 0, 0, 0),
    "ft2": (2, 0, 0, 0),
    "lbf": (1, 1, -2, 0),
    "slug": (0, 1, 0, 0),
    "degC": (0, 0, 0, 1),
    "degF": (0, 0, 0, 1),
}

# Conversion factors TO SI (canonical)
TO_SI: Dict[str, float] = {
    "m": 1.0,
    "kg": 1.0,
    "s": 1.0,
    "K": 1.0,
    "m/s": 1.0,
    "m/s2": 1.0,
    "m2": 1.0,
    "m2/s2": 1.0,
    "m3": 1.0,
    "kg/m3": 1.0,
    "kg/m2": 1.0,
    "kg/m2/s": 1.0,
    "Pa": 1.0,
    "N": 1.0,
    "J": 1.0,
    "W": 1.0,
    "Pa_s": 1.0,
    "m2/s": 1.0,
    "1/s": 1.0,
    "Hz": 1.0,
    "rad": 1.0,
    "deg": math.pi / 180.0,
    "1": 1.0,
    # Non-SI
    "ft": 0.3048,
    "ft2": 0.09290304,
    "lbf": 4.4482216152605,
    "slug": 14.593902937,
    "degC": 1.0,      # Note: offset handled separately
    "degF": 5.0 / 9.0,
}

# Conversion factors FROM SI
FROM_SI: Dict[str, float] = {k: 1.0 / v for k, v in TO_SI.items()}


class DimensionalError(ValueError):
    """Raised when operations involve incompatible dimensions."""
    pass


class UnitError(ValueError):
    """Raised for unknown or unsupported units."""
    pass


@dataclass(frozen=True)
class Quantity:
    """
    A physical quantity with value, unit, and semantic name.
    
    Internally stores value in SI (canonical) units.
    All arithmetic preserves dimensional correctness.
    """
    _value_si: float          # Value in SI (canonical) units
    _unit: str                # Canonical unit string (e.g., "m", "kg/m3", "Pa")
    _name: str = ""           # Semantic name (e.g., "velocity", "dynamic_pressure")
    
    def __post_init__(self):
        if self._unit not in DIMENSIONS:
            raise UnitError(f"Unknown canonical unit: {self._unit}")
        if not isinstance(self._value_si, (int, float)):
            raise TypeError(f"Value must be numeric, got {type(self._value_si)}")
    
    @property
    def value(self) -> float:
        """Value in canonical SI units."""
        return self._value_si
    
    @property
    def unit(self) -> str:
        """Canonical unit string."""
        return self._unit
    
    @property
    def name(self) -> str:
        """Semantic name/description."""
        return self._name
    
    @property
    def dimensions(self) -> tuple:
        """Dimension vector (L, M, T, Θ)."""
        return DIMENSIONS[self._unit]
    
    def to(self, target_unit: str) -> "Quantity":
        """Convert to a different unit (must be dimensionally compatible)."""
        if target_unit not in DIMENSIONS:
            raise UnitError(f"Unknown target unit: {target_unit}")
        if DIMENSIONS[target_unit] != self.dimensions:
            raise DimensionalError(
                f"Cannot convert {self._unit} ({self.dimensions}) to {target_unit} ({DIMENSIONS[target_unit]})"
            )
        # Handle temperature offsets
        if self._unit in ("degC", "degF") or target_unit in ("degC", "degF"):
            raise DimensionalError("Temperature conversions with offsets not supported in lightweight mode. Use Kelvin.")
        
        factor = FROM_SI[target_unit] / FROM_SI[self._unit]
        return Quantity(self._value_si * factor, target_unit, self._name)
    
    def __float__(self) -> float:
        """Allow implicit conversion to float (SI value). Use with caution."""
        return self._value_si
    
    def __repr__(self) -> str:
        return f"Quantity({self._value_si:.6g}, '{self._unit}', '{self._name}')"
    
    def __str__(self) -> str:
        if self._name:
            return f"{self._value_si:.6g} {self._unit} ({self._name})"
        return f"{self._value_si:.6g} {self._unit}"
    
    # Arithmetic with dimensional checking
    def _check_compatible(self, other: "Quantity", op: str) -> None:
        if not isinstance(other, Quantity):
            raise TypeError(f"Unsupported operand type for {op}: 'Quantity' and '{type(other).__name__}'")
        if self.dimensions != other.dimensions:
            raise DimensionalError(
                f"Dimension mismatch in {op}: {self._unit} ({self.dimensions}) vs {other._unit} ({other.dimensions})"
            )
    
    def _check_multipliable(self, other: "Quantity") -> None:
        # Multiplication/division always valid dimensionally, result dimensions add/subtract
        pass
    
    def __add__(self, other: "Quantity") -> "Quantity":
        self._check_compatible(other, "+")
        return Quantity(self._value_si + other._value_si, self._unit, self._name)
    
    def __sub__(self, other: "Quantity") -> "Quantity":
        self._check_compatible(other, "-")
        return Quantity(self._value_si - other._value_si, self._unit, self._name)
    
    def __mul__(self, other: Union["Quantity", float, int]) -> "Quantity":
        if isinstance(other, (int, float)):
            return Quantity(self._value_si * other, self._unit, self._name)
        if not isinstance(other, Quantity):
            return NotImplemented
        # Multiply dimensions
        new_dim = tuple(a + b for a, b in zip(self.dimensions, other.dimensions))
        new_unit = self._find_unit_by_dimensions(new_dim)
        if new_unit is None:
            raise DimensionalError(f"Result dimensions {new_dim} have no registered unit")
        return Quantity(self._value_si * other._value_si, new_unit, self._name)
    
    def __rmul__(self, other: Union[float, int]) -> "Quantity":
        return self.__mul__(other)
    
    def __truediv__(self, other: Union["Quantity", float, int]) -> "Quantity":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return Quantity(self._value_si / other, self._unit, self._name)
        if not isinstance(other, Quantity):
            return NotImplemented
        # Divide dimensions
        new_dim = tuple(a - b for a, b in zip(self.dimensions, other.dimensions))
        new_unit = self._find_unit_by_dimensions(new_dim)
        if new_unit is None:
            raise DimensionalError(f"Result dimensions {new_dim} have no registered unit")
        return Quantity(self._value_si / other._value_si, new_unit, self._name)
    
    def __rtruediv__(self, other: Union[float, int]) -> "Quantity":
        if not isinstance(other, (int, float)):
            return NotImplemented
        new_dim = tuple(-d for d in self.dimensions)
        new_unit = self._find_unit_by_dimensions(new_dim)
        if new_unit is None:
            raise DimensionalError(f"Result dimensions {new_dim} have no registered unit")
        return Quantity(other / self._value_si, new_unit, f"1/{self._name}" if self._name else "")
    
    def __pow__(self, exponent: float) -> "Quantity":
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        new_dim = tuple(d * exponent for d in self.dimensions)
        new_unit = self._find_unit_by_dimensions(new_dim)
        if new_unit is None:
            raise DimensionalError(f"Result dimensions {new_dim} have no registered unit")
        return Quantity(self._value_si ** exponent, new_unit, self._name)
    
    def _find_unit_by_dimensions(self, dim: tuple) -> Optional[str]:
        for unit, d in DIMENSIONS.items():
            if d == dim:
                return unit
        return None
    
    # Comparisons
    def _check_comparable(self, other: "Quantity") -> None:
        if not isinstance(other, Quantity):
            raise TypeError(f"Cannot compare Quantity with {type(other).__name__}")
        if self.dimensions != other.dimensions:
            raise DimensionalError(
                f"Dimension mismatch in comparison: {self._unit} vs {other._unit}"
            )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return False
        self._check_comparable(other)
        return abs(self._value_si - other._value_si) < 1e-12
    
    def __lt__(self, other: "Quantity") -> bool:
        self._check_comparable(other)
        return self._value_si < other._value_si
    
    def __le__(self, other: "Quantity") -> bool:
        self._check_comparable(other)
        return self._value_si <= other._value_si
    
    def __gt__(self, other: "Quantity") -> bool:
        self._check_comparable(other)
        return self._value_si > other._value_si
    
    def __ge__(self, other: "Quantity") -> bool:
        self._check_comparable(other)
        return self._value_si >= other._value_si


# Factory functions for common quantities
def meter(value: float, name: str = "") -> Quantity:
    return Quantity(value, "m", name)

def kilogram(value: float, name: str = "") -> Quantity:
    return Quantity(value, "kg", name)

def second(value: float, name: str = "") -> Quantity:
    return Quantity(value, "s", name)

def kelvin(value: float, name: str = "") -> Quantity:
    return Quantity(value, "K", name)

def pascal(value: float, name: str = "") -> Quantity:
    return Quantity(value, "Pa", name)

def newton(value: float, name: str = "") -> Quantity:
    return Quantity(value, "N", name)

def joule(value: float, name: str = "") -> Quantity:
    return Quantity(value, "J", name)

def watt(value: float, name: str = "") -> Quantity:
    return Quantity(value, "W", name)

def m_per_s(value: float, name: str = "") -> Quantity:
    return Quantity(value, "m/s", name)

def m_per_s2(value: float, name: str = "") -> Quantity:
    return Quantity(value, "m/s2", name)

def kg_per_m3(value: float, name: str = "") -> Quantity:
    return Quantity(value, "kg/m3", name)

def pascal_second(value: float, name: str = "") -> Quantity:
    return Quantity(value, "Pa_s", name)

def m2_per_s(value: float, name: str = "") -> Quantity:
    return Quantity(value, "m2/s", name)

def radian(value: float, name: str = "") -> Quantity:
    return Quantity(value, "rad", name)

def degree(value: float, name: str = "") -> Quantity:
    return Quantity(value * math.pi / 180.0, "rad", name)

def dimensionless(value: float, name: str = "") -> Quantity:
    return Quantity(value, "1", name)


# Common non-SI conversions (explicit, for I/O only)
def foot(value: float, name: str = "") -> Quantity:
    return Quantity(value * 0.3048, "m", name)

def lbf(value: float, name: str = "") -> Quantity:
    return Quantity(value * 4.4482216152605, "N", name)

def slug(value: float, name: str = "") -> Quantity:
    return Quantity(value * 14.593902937, "kg", name)

def degC(value: float, name: str = "") -> Quantity:
    return Quantity(value + 273.15, "K", name)

def degF(value: float, name: str = "") -> Quantity:
    return Quantity((value - 32.0) * 5.0 / 9.0 + 273.15, "K", name)


# Helper to extract SI float from Quantity (for legacy code interface)
def to_si_float(q: Union[Quantity, float], expected_unit: str = None) -> float:
    """Extract SI value, optionally checking unit compatibility."""
    if isinstance(q, Quantity):
        if expected_unit and q.unit != expected_unit:
            # Check dimensional compatibility
            if DIMENSIONS.get(q.unit) != DIMENSIONS.get(expected_unit):
                raise DimensionalError(f"Expected {expected_unit}, got {q.unit}")
        return q.value
    return float(q)


# For backward compatibility: raw float pass-through with warning context
def as_float(value: Union[Quantity, float], context: str = "") -> float:
    """Extract raw float for legacy physics functions."""
    if isinstance(value, Quantity):
        return value.value
    return float(value)