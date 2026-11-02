"""
Tests for aeris.core.constants
"""

import pytest
from aeris.core import constants as const


def test_constants_have_correct_values():
    """Verify constants match ISA standard values."""
    assert const.G0 == 9.80665
    assert const.R_GAS_AIR == 287.05
    assert const.GAMMA_AIR == 1.4
    assert const.RHO_SL == 1.225
    assert const.MU_SL == 1.81e-5
    assert const.T_SL == 288.15
    assert const.P_SL == 101325.0
    assert const.LAPSE_RATE == 0.0065
    assert const.SUTHERLAND_CONST == 110.4


def test_validity_thresholds_defined():
    """Verify model validity thresholds are defined and reasonable."""
    assert const.MACH_INCOMPRESSIBLE_LIMIT == 0.3
    assert const.MACH_TRANSONIC_LIMIT == 0.7
    assert const.RE_LAMINAR_SEPARATION_LIMIT == 5e4
    assert const.RE_MODEL_UPPER_LIMIT == 1e7
    assert const.MIN_THICKNESS_RATIO == 0.04
    assert const.MIN_TIP_CHORD_RATIO == 0.03


def test_threshold_ordering():
    """Thresholds should be in logical order."""
    assert const.MACH_INCOMPRESSIBLE_LIMIT < const.MACH_TRANSONIC_LIMIT
    assert const.RE_LAMINAR_SEPARATION_LIMIT < const.RE_MODEL_UPPER_LIMIT