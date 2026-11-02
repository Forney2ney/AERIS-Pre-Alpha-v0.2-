"""
aeris.core.constants
====================
Single source of truth for all physical constants used in AERIS.
All values in SI units.
"""

# Universal constants
G0 = 9.80665                    # m/s^2, standard gravity

# Air properties (ISA sea level)
R_GAS_AIR = 287.05              # J/(kg·K), specific gas constant for dry air
GAMMA_AIR = 1.4                 # ratio of specific heats (cp/cv) for air
RHO_SL = 1.225                  # kg/m^3, sea-level density
MU_SL = 1.81e-5                 # Pa·s, sea-level dynamic viscosity
T_SL = 288.15                   # K, sea-level temperature
P_SL = 101325.0                 # Pa, sea-level pressure
LAPSE_RATE = 0.0065             # K/m, ISA tropospheric lapse rate
SUTHERLAND_CONST = 110.4        # K, Sutherland's constant for air viscosity

# Model validity thresholds (documented rationale in model_validity.py)
MACH_INCOMPRESSIBLE_LIMIT = 0.3     # Below: incompressible flow assumption valid
MACH_TRANSONIC_LIMIT = 0.7          # Above: transonic effects significant, thin airfoil theory invalid

RE_LAMINAR_SEPARATION_LIMIT = 5e4   # Below: laminar separation dominates, empirical models unreliable
RE_MODEL_UPPER_LIMIT = 1e7          # Above: extrapolation beyond calibrated range

# Geometry limits
MIN_THICKNESS_RATIO = 0.04          # Minimum t/c for valid airfoil
MIN_TIP_CHORD_RATIO = 0.03          # Minimum tip chord / root chord