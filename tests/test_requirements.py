"""
Tests for aeris.core.requirements
"""

import pytest
from aeris.core.requirements import Requirements, create_requirements_from_mission
from aeris.core.units import (
    Quantity, kilogram, m_per_s, meter, second, newton, pascal, dimensionless, as_float
)
from aeris.core import constants as const


class TestRequirementsCreation:
    def test_create_from_mission(self):
        req = create_requirements_from_mission(
            mass_kg=1.5,
            cruise_speed_mps=15.0,
            altitude_m=1000.0,
            max_span_m=1.5,
            endurance_min=20.0,
        )
        
        assert as_float(req.mass) == 1.5
        assert as_float(req.cruise_speed) == 15.0
        assert as_float(req.altitude) == 1000.0
        assert as_float(req.max_span) == 1.5
        assert as_float(req.endurance) == 1200.0  # 20 min * 60
    
    def test_create_with_optional_fields(self):
        req = create_requirements_from_mission(
            mass_kg=1.5,
            cruise_speed_mps=15.0,
            altitude_m=1000.0,
            max_span_m=1.5,
            stall_speed_mps=10.0,
            payload_mass_kg=0.3,
        )
        
        assert req.stall_speed is not None
        assert as_float(req.stall_speed) == 10.0
        assert req.payload_mass is not None
        assert as_float(req.payload_mass) == 0.3


class TestDerivedQuantities:
    def test_weight_calculation(self):
        req = create_requirements_from_mission(1.5, 15.0, 1000.0, 1.5)
        W = req.weight
        expected = 1.5 * const.G0
        assert abs(as_float(W) - expected) < 1e-6
        assert W.unit == "N"
    
    def test_atmosphere_derived(self):
        req = create_requirements_from_mission(1.5, 15.0, 1000.0, 1.5)
        atm = req.atmosphere
        
        assert atm.altitude.value == 1000.0
        assert atm.density.value > 0
        assert atm.temperature.value > 0
    
    def test_dynamic_pressure(self):
        req = create_requirements_from_mission(1.5, 15.0, 1000.0, 1.5)
        q = req.dynamic_pressure
        
        # q = 0.5 * ρ * V² at 1000m
        # ρ at 1000m ≈ 1.112 kg/m³
        # q ≈ 0.5 * 1.112 * 225 = 125.1 Pa
        assert q.unit == "Pa"
        assert 120 < as_float(q) < 130
    
    def test_mach_number(self):
        req = create_requirements_from_mission(1.5, 15.0, 1000.0, 1.5)
        M = req.mach_number
        
        # M = V / a at 1000m
        # a at 1000m ≈ 336.4 m/s
        # M ≈ 15 / 336.4 ≈ 0.0446
        assert M.dimensions == (0, 0, 0, 0)
        assert 0.04 < as_float(M) < 0.05
    
    def test_reynolds_number(self):
        req = create_requirements_from_mission(1.5, 15.0, 1000.0, 1.5)
        Re = req.reynolds_number(meter(0.2))
        
        # Re = ρ * V * L / μ at 1000m
        # ρ ≈ 1.112, μ ≈ 1.76e-5
        # Re ≈ 1.112 * 15 * 0.2 / 1.76e-5 ≈ 189,545
        assert Re.dimensions == (0, 0, 0, 0)
        assert 180000 < as_float(Re) < 200000
    
    def test_required_lift_coefficient(self):
        req = create_requirements_from_mission(1.5, 15.0, 1000.0, 1.5)
        S = meter(0.2)**2  # Wait - need proper area unit
        # Actually use the m2 unit
        from aeris.core.units import Quantity
        S = Quantity(0.2, "m2", "wing_area")
        
        CL = req.required_lift_coefficient(S)
        assert CL.dimensions == (0, 0, 0, 0)
        
        # CL = W / (q * S)
        # W = 14.71 N, q ≈ 125.1 Pa, S = 0.2 m²
        # CL = 14.71 / (125.1 * 0.2) = 14.71 / 25.02 = 0.588
        assert 0.55 < as_float(CL) < 0.65
    
    def test_wing_loading(self):
        req = create_requirements_from_mission(1.5, 15.0, 1000.0, 1.5)
        S = Quantity(0.2, "m2", "wing_area")
        
        WL = req.wing_loading(S)
        assert WL.unit == "Pa"
        
        # W/S = 14.71 / 0.2 = 73.55 Pa
        assert 70 < as_float(WL) < 80
    
    def test_aspect_ratio_bounds(self):
        req = create_requirements_from_mission(1.5, 15.0, 1000.0, 1.5)
        S = Quantity(0.2, "m2", "wing_area")
        
        AR_min, AR_max = req.aspect_ratio_bounds(S)
        
        # AR_max = b² / S = 1.5² / 0.2 = 2.25 / 0.2 = 11.25
        assert AR_min == 4.0
        assert abs(AR_max - 11.25) < 0.1
    
    def test_estimate_wing_area(self):
        req = create_requirements_from_mission(1.5, 15.0, 1000.0, 1.5)
        S = req.estimate_wing_area(target_CL=0.6)
        
        # S = W / (q * CL) = 14.71 / (125.1 * 0.6) = 14.71 / 75.06 = 0.196 m²
        assert abs(as_float(S) - 0.196) < 0.01
        assert S.unit == "m2"


class TestEngineeringExample:
    """
    The specified engineering validation example:
    1.5 kg UAV, 15 m/s, 1000 m altitude, 1.5 m max span
    """
    def test_uav_example_complete(self):
        req = create_requirements_from_mission(
            mass_kg=1.5,
            cruise_speed_mps=15.0,
            altitude_m=1000.0,
            max_span_m=1.5,
            endurance_min=20.0,
        )
        
        # Atmospheric properties
        atm = req.atmosphere
        rho = as_float(atm.density)
        T = as_float(atm.temperature)
        P = as_float(atm.pressure)
        a = as_float(atm.speed_of_sound)
        mu = as_float(atm.dynamic_viscosity)
        
        # Verify approximate values
        assert 1.10 < rho < 1.13      # ρ ≈ 1.112 kg/m³
        assert 281 < T < 283           # T ≈ 281.65 K
        assert 89000 < P < 91000       # P ≈ 89875 Pa
        assert 335 < a < 338           # a ≈ 336.4 m/s
        assert 1.74e-5 < mu < 1.78e-5  # μ ≈ 1.76e-5 Pa·s
        
        # Derived quantities
        q = as_float(req.dynamic_pressure)
        M = as_float(req.mach_number)
        Re = as_float(req.reynolds_number(meter(0.2)))
        
        assert 120 < q < 130           # q ≈ 125.1 Pa
        assert 0.044 < M < 0.046       # M ≈ 0.0446
        assert 185000 < Re < 195000    # Re ≈ 189,545
        
        # Required CL for S=0.2 m²
        S = Quantity(0.2, "m2", "wing_area")
        CL = as_float(req.required_lift_coefficient(S))
        assert 0.55 < CL < 0.65        # CL ≈ 0.588
        
        # Model validity: all should be VALID
        # M = 0.0446 < 0.3 → VALID (incompressible)
        # Re = 189,545 > 50,000 → VALID
        assert M < 0.3
        assert Re > 50000


class TestErrorHandling:
    def test_zero_wing_area_raises(self):
        req = create_requirements_from_mission(1.5, 15.0, 1000.0, 1.5)
        S = Quantity(0.0, "m2", "wing_area")
        
        with pytest.raises(ValueError):
            req.required_lift_coefficient(S)
        
        with pytest.raises(ValueError):
            req.wing_loading(S)
        
        with pytest.raises(ValueError):
            req.aspect_ratio_bounds(S)
    
    def test_zero_dynamic_pressure_raises(self):
        req = create_requirements_from_mission(1.5, 0.0, 1000.0, 1.5)
        S = Quantity(0.2, "m2", "wing_area")
        
        with pytest.raises(ValueError):
            req.required_lift_coefficient(S)