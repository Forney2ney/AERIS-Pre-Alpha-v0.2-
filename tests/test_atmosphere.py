"""
Tests for aeris.core.atmosphere
"""

import pytest
import math
from aeris.core.atmosphere import (
    Atmosphere, standard_atmosphere, reynolds_number, 
    dynamic_pressure, mach_number, speed_of_sound
)
from aeris.core.units import (
    meter, m_per_s, kg_per_m3, pascal_second, kelvin, pascal,
    as_float, dimensionless
)
from aeris.core import constants as const


class TestAtmosphereSeaLevel:
    def test_sea_level_values(self):
        atm = Atmosphere.from_altitude(meter(0.0))
        
        # ISA model computes sea-level values from hydrostatic equations
        # Small numerical differences from defined constants are expected (~0.001%)
        assert abs(atm.density.value - const.RHO_SL) < 1e-4
        assert abs(atm.temperature.value - const.T_SL) < 1e-6
        assert abs(atm.pressure.value - const.P_SL) < 1e-2
        assert abs(atm.dynamic_viscosity.value - const.MU_SL) < 1e-8
    
    def test_sea_level_speed_of_sound(self):
        atm = Atmosphere.from_altitude(meter(0.0))
        a_expected = math.sqrt(const.GAMMA_AIR * const.R_GAS_AIR * const.T_SL)
        assert abs(atm.speed_of_sound.value - a_expected) < 0.1
        # ISA sea level speed of sound ≈ 340.3 m/s
        assert abs(atm.speed_of_sound.value - 340.3) < 1.0
    
    def test_sea_level_kinematic_viscosity(self):
        atm = Atmosphere.from_altitude(meter(0.0))
        nu_expected = const.MU_SL / const.RHO_SL
        assert abs(atm.kinematic_viscosity.value - nu_expected) < 1e-8


class TestAtmosphereAltitude:
    def test_altitude_decreases_density(self):
        atm_sl = Atmosphere.from_altitude(meter(0.0))
        atm_1000 = Atmosphere.from_altitude(meter(1000.0))
        atm_5000 = Atmosphere.from_altitude(meter(5000.0))
        
        assert atm_1000.density.value < atm_sl.density.value
        assert atm_5000.density.value < atm_1000.density.value
    
    def test_altitude_decreases_temperature(self):
        atm_sl = Atmosphere.from_altitude(meter(0.0))
        atm_1000 = Atmosphere.from_altitude(meter(1000.0))
        atm_5000 = Atmosphere.from_altitude(meter(5000.0))
        
        assert atm_1000.temperature.value < atm_sl.temperature.value
        assert atm_5000.temperature.value < atm_1000.temperature.value
    
    def test_altitude_decreases_pressure(self):
        atm_sl = Atmosphere.from_altitude(meter(0.0))
        atm_1000 = Atmosphere.from_altitude(meter(1000.0))
        atm_5000 = Atmosphere.from_altitude(meter(5000.0))
        
        assert atm_1000.pressure.value < atm_sl.pressure.value
        assert atm_5000.pressure.value < atm_1000.pressure.value
    
    def test_1000m_values_approximate(self):
        """Verify 1000m values match known ISA approximations."""
        atm = Atmosphere.from_altitude(meter(1000.0))
        # ISA at 1000m: T≈281.65K, P≈89875Pa, ρ≈1.112kg/m³, a≈336.4m/s
        assert abs(atm.temperature.value - 281.65) < 0.5
        assert abs(atm.pressure.value - 89875) < 500
        assert abs(atm.density.value - 1.112) < 0.01
        assert abs(atm.speed_of_sound.value - 336.4) < 1.0
    
    def test_5000m_values_approximate(self):
        """Verify 5000m values match known ISA approximations."""
        atm = Atmosphere.from_altitude(meter(5000.0))
        # ISA at 5000m: T≈255.65K, P≈54048Pa, ρ≈0.736kg/m³, a≈320.5m/s
        assert abs(atm.temperature.value - 255.65) < 0.5
        assert abs(atm.pressure.value - 54048) < 500
        assert abs(atm.density.value - 0.736) < 0.01
        assert abs(atm.speed_of_sound.value - 320.5) < 1.0


class TestDerivedQuantities:
    def test_dynamic_pressure(self):
        atm = Atmosphere.from_altitude(meter(0.0))
        V = m_per_s(15.0)
        q = atm.dynamic_pressure(V)
        
        # q = 0.5 * ρ * V^2, using actual computed density
        expected = 0.5 * atm.density.value * 15.0**2
        assert abs(q.value - expected) < 1e-6
        assert q.unit == "Pa"
    
    def test_reynolds_number(self):
        atm = Atmosphere.from_altitude(meter(0.0))
        V = m_per_s(15.0)
        L = meter(0.2)
        Re = atm.reynolds_number(V, L)
        
        # Re = 1.225 * 15 * 0.2 / 1.81e-5 = 203,315
        expected = const.RHO_SL * 15.0 * 0.2 / const.MU_SL
        assert abs(Re.value - expected) < 1000
        assert Re.dimensions == (0, 0, 0, 0)
    
    def test_mach_number(self):
        atm = Atmosphere.from_altitude(meter(0.0))
        V = m_per_s(15.0)
        M = atm.mach_number(V)
        
        # M = 15 / 340.3 ≈ 0.0441
        expected = 15.0 / atm.speed_of_sound.value
        assert abs(M.value - expected) < 1e-6
        assert M.dimensions == (0, 0, 0, 0)
    
    def test_mach_number_at_1000m(self):
        atm = Atmosphere.from_altitude(meter(1000.0))
        V = m_per_s(15.0)
        M = atm.mach_number(V)
        
        # M = 15 / 336.4 ≈ 0.0446
        expected = 15.0 / 336.4
        assert abs(M.value - expected) < 0.001


class TestBackwardCompatibility:
    def test_standard_atmosphere_returns_floats(self):
        rho, mu, T, P = standard_atmosphere(0.0)
        assert isinstance(rho, float)
        assert isinstance(mu, float)
        assert isinstance(T, float)
        assert isinstance(P, float)
        # ISA model computes from hydrostatic equations - small numerical difference expected
        assert abs(rho - const.RHO_SL) < 1e-4
    
    def test_reynolds_number_floats(self):
        Re = reynolds_number(1.225, 15.0, 0.2, 1.81e-5)
        expected = 1.225 * 15.0 * 0.2 / 1.81e-5
        assert abs(Re - expected) < 1e-6
    
    def test_dynamic_pressure_floats(self):
        q = dynamic_pressure(1.225, 15.0)
        expected = 0.5 * 1.225 * 15.0**2
        assert abs(q - expected) < 1e-6
    
    def test_mach_number_floats(self):
        M = mach_number(15.0, 340.3)
        expected = 15.0 / 340.3
        assert abs(M - expected) < 1e-6
    
    def test_speed_of_sound(self):
        a = speed_of_sound(288.15)
        expected = math.sqrt(const.GAMMA_AIR * const.R_GAS_AIR * 288.15)
        assert abs(a - expected) < 1e-6


class TestUnitsIntegration:
    def test_atmosphere_returns_quantities(self):
        atm = Atmosphere.from_altitude(meter(1000.0))
        
        assert isinstance(atm.altitude, type(atm.density))
        assert atm.density.unit == "kg/m3"
        assert atm.temperature.unit == "K"
        assert atm.pressure.unit == "Pa"
        assert atm.dynamic_viscosity.unit == "Pa_s"
        assert atm.kinematic_viscosity.unit == "m2/s"
        assert atm.speed_of_sound.unit == "m/s"
    
    def test_derived_quantities_return_quantities(self):
        atm = Atmosphere.from_altitude(meter(0.0))
        V = m_per_s(15.0)
        
        q = atm.dynamic_pressure(V)
        Re = atm.reynolds_number(V, meter(0.2))
        M = atm.mach_number(V)
        
        assert isinstance(q, type(atm.density))
        assert q.unit == "Pa"
        assert isinstance(Re, type(atm.density))
        assert Re.dimensions == (0, 0, 0, 0)
        assert isinstance(M, type(atm.density))
        assert M.dimensions == (0, 0, 0, 0)
    
    def test_as_float_extraction(self):
        atm = Atmosphere.from_altitude(meter(0.0))
        rho_float = as_float(atm.density)
        assert isinstance(rho_float, float)
        assert abs(rho_float - const.RHO_SL) < 1e-4


class TestNegativeAltitude:
    def test_negative_altitude_clamped_to_zero(self):
        atm = Atmosphere.from_altitude(meter(-100.0))
        atm_sl = Atmosphere.from_altitude(meter(0.0))
        assert atm.density.value == atm_sl.density.value


class TestEdgeCases:
    def test_high_altitude_extrapolation(self):
        """Above 11km should still compute (with warning in validity engine)."""
        atm = Atmosphere.from_altitude(meter(12000.0))
        assert atm.density.value > 0
        assert atm.temperature.value > 0
        assert atm.pressure.value > 0