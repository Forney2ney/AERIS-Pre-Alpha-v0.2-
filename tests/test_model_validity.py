"""
Tests for aeris.core.model_validity
"""

import pytest
from aeris.core.model_validity import (
    ModelValidityEngine, ValidityStatus, ValidityCheck, check_conditions
)
from aeris.core import constants as const


class TestMachChecks:
    def test_mach_valid_below_03(self):
        engine = ModelValidityEngine()
        check = engine.check_mach(0.2)
        assert check.status == ValidityStatus.VALID
        assert check.threshold == 0.3
        assert check.actual == 0.2
    
    def test_mach_valid_at_025(self):
        engine = ModelValidityEngine()
        check = engine.check_mach(0.25)
        assert check.status == ValidityStatus.VALID
    
    def test_mach_warning_at_05(self):
        engine = ModelValidityEngine()
        check = engine.check_mach(0.5)
        assert check.status == ValidityStatus.WARNING
        assert check.threshold == 0.7
    
    def test_mach_warning_at_06(self):
        engine = ModelValidityEngine()
        check = engine.check_mach(0.6)
        assert check.status == ValidityStatus.WARNING
    
    def test_mach_invalid_at_08(self):
        engine = ModelValidityEngine()
        check = engine.check_mach(0.8)
        assert check.status == ValidityStatus.INVALID
    
    def test_mach_invalid_at_12(self):
        engine = ModelValidityEngine()
        check = engine.check_mach(1.2)
        assert check.status == ValidityStatus.INVALID
    
    def test_mach_exact_boundary_03(self):
        """At exactly 0.3, should be WARNING (not < 0.3)."""
        engine = ModelValidityEngine()
        check = engine.check_mach(0.3)
        assert check.status == ValidityStatus.WARNING
    
    def test_mach_exact_boundary_07(self):
        """At exactly 0.7, should be INVALID (not < 0.7)."""
        engine = ModelValidityEngine()
        check = engine.check_mach(0.7)
        assert check.status == ValidityStatus.INVALID


class TestReynoldsChecks:
    def test_re_warning_below_5e4(self):
        engine = ModelValidityEngine()
        check = engine.check_reynolds(1e4)
        assert check.status == ValidityStatus.WARNING
        assert check.threshold == 5e4
    
    def test_re_valid_at_1e5(self):
        engine = ModelValidityEngine()
        check = engine.check_reynolds(1e5)
        assert check.status == ValidityStatus.VALID
    
    def test_re_valid_at_1e6(self):
        engine = ModelValidityEngine()
        check = engine.check_reynolds(1e6)
        assert check.status == ValidityStatus.VALID
    
    def test_re_valid_at_5e6(self):
        engine = ModelValidityEngine()
        check = engine.check_reynolds(5e6)
        assert check.status == ValidityStatus.VALID
    
    def test_re_warning_above_1e7(self):
        engine = ModelValidityEngine()
        check = engine.check_reynolds(2e7)
        assert check.status == ValidityStatus.WARNING
        assert check.threshold == 1e7
    
    def test_re_exact_boundary_5e4(self):
        """At exactly 5e4, should be VALID (not < 5e4)."""
        engine = ModelValidityEngine()
        check = engine.check_reynolds(5e4)
        assert check.status == ValidityStatus.VALID
    
    def test_re_exact_boundary_1e7(self):
        """At exactly 1e7, should be WARNING (not < 1e7)."""
        engine = ModelValidityEngine()
        check = engine.check_reynolds(1e7)
        assert check.status == ValidityStatus.WARNING


class TestAlphaChecks:
    def test_alpha_valid_margin_gt_2(self):
        engine = ModelValidityEngine()
        check = engine.check_alpha_vs_stall(5.0, 12.0)  # margin = 7°
        assert check.status == ValidityStatus.VALID
    
    def test_alpha_valid_margin_3(self):
        engine = ModelValidityEngine()
        check = engine.check_alpha_vs_stall(8.0, 12.0)  # margin = 4°
        assert check.status == ValidityStatus.VALID
    
    def test_alpha_warning_margin_2(self):
        engine = ModelValidityEngine()
        check = engine.check_alpha_vs_stall(10.0, 12.0)  # margin = 2°
        assert check.status == ValidityStatus.WARNING
    
    def test_alpha_warning_margin_1(self):
        engine = ModelValidityEngine()
        check = engine.check_alpha_vs_stall(11.0, 12.0)  # margin = 1°
        assert check.status == ValidityStatus.WARNING
    
    def test_alpha_invalid_at_stall(self):
        engine = ModelValidityEngine()
        check = engine.check_alpha_vs_stall(12.0, 12.0)  # margin = 0°
        assert check.status == ValidityStatus.INVALID
    
    def test_alpha_invalid_above_stall(self):
        engine = ModelValidityEngine()
        check = engine.check_alpha_vs_stall(14.0, 12.0)  # margin = -2°
        assert check.status == ValidityStatus.INVALID


class TestGeometryChecks:
    def test_geometry_valid_normal(self):
        engine = ModelValidityEngine()
        checks = engine.check_geometry(
            tip_chord=0.15, root_chord=0.3, thickness=0.12, aspect_ratio=10.0
        )
        # Should have 3 checks: tip_chord, thickness, aspect_ratio
        assert len(checks) == 3
        assert all(c.status == ValidityStatus.VALID for c in checks)
    
    def test_geometry_invalid_zero_tip_chord(self):
        engine = ModelValidityEngine()
        checks = engine.check_geometry(
            tip_chord=0.0, root_chord=0.3, thickness=0.12
        )
        tip_check = next(c for c in checks if c.name == "Tip Chord")
        assert tip_check.status == ValidityStatus.INVALID
    
    def test_geometry_warning_small_tip_chord(self):
        engine = ModelValidityEngine()
        checks = engine.check_geometry(
            tip_chord=0.005, root_chord=0.3, thickness=0.12  # ratio = 0.0167 < 0.03
        )
        tip_check = next(c for c in checks if c.name == "Tip Chord")
        assert tip_check.status == ValidityStatus.WARNING
    
    def test_geometry_invalid_thin(self):
        engine = ModelValidityEngine()
        checks = engine.check_geometry(
            tip_chord=0.15, root_chord=0.3, thickness=0.03  # < 0.04
        )
        thick_check = next(c for c in checks if c.name == "Thickness Ratio")
        assert thick_check.status == ValidityStatus.INVALID
    
    def test_geometry_warning_thick(self):
        engine = ModelValidityEngine()
        checks = engine.check_geometry(
            tip_chord=0.15, root_chord=0.3, thickness=0.20  # > 0.18
        )
        thick_check = next(c for c in checks if c.name == "Thickness Ratio")
        assert thick_check.status == ValidityStatus.WARNING
    
    def test_geometry_warning_low_ar(self):
        engine = ModelValidityEngine()
        checks = engine.check_geometry(
            tip_chord=0.15, root_chord=0.3, thickness=0.12, aspect_ratio=3.0
        )
        ar_check = next(c for c in checks if c.name == "Aspect Ratio")
        assert ar_check.status == ValidityStatus.WARNING
    
    def test_geometry_warning_high_ar(self):
        engine = ModelValidityEngine()
        checks = engine.check_geometry(
            tip_chord=0.15, root_chord=0.3, thickness=0.12, aspect_ratio=30.0
        )
        ar_check = next(c for c in checks if c.name == "Aspect Ratio")
        assert ar_check.status == ValidityStatus.WARNING


class TestOverallStatus:
    def test_all_valid(self):
        engine = ModelValidityEngine()
        checks = [
            ValidityCheck("A", ValidityStatus.VALID, "ok"),
            ValidityCheck("B", ValidityStatus.VALID, "ok"),
        ]
        assert engine.overall_status(checks) == ValidityStatus.VALID
    
    def test_one_warning(self):
        engine = ModelValidityEngine()
        checks = [
            ValidityCheck("A", ValidityStatus.VALID, "ok"),
            ValidityCheck("B", ValidityStatus.WARNING, "warn"),
        ]
        assert engine.overall_status(checks) == ValidityStatus.WARNING
    
    def test_one_invalid(self):
        engine = ModelValidityEngine()
        checks = [
            ValidityCheck("A", ValidityStatus.VALID, "ok"),
            ValidityCheck("B", ValidityStatus.INVALID, "fail"),
        ]
        assert engine.overall_status(checks) == ValidityStatus.INVALID
    
    def test_warning_and_invalid(self):
        engine = ModelValidityEngine()
        checks = [
            ValidityCheck("A", ValidityStatus.WARNING, "warn"),
            ValidityCheck("B", ValidityStatus.INVALID, "fail"),
        ]
        assert engine.overall_status(checks) == ValidityStatus.INVALID


class TestConvenienceFunction:
    def test_check_conditions_mach_only(self):
        checks = check_conditions(mach=0.2)
        assert len(checks) == 1
        assert checks[0].name == "Mach Number"
        assert checks[0].status == ValidityStatus.VALID
    
    def test_check_conditions_multiple(self):
        checks = check_conditions(mach=0.5, reynolds=1e6, alpha_deg=5, stall_alpha_deg=12)
        assert len(checks) == 3
        # Mach=0.5 → WARNING, Re=1e6 → VALID, α=5 stall=12 → margin 7° → VALID
        statuses = {c.name: c.status for c in checks}
        assert statuses["Mach Number"] == ValidityStatus.WARNING
        assert statuses["Reynolds Number"] == ValidityStatus.VALID
        assert statuses["Angle of Attack"] == ValidityStatus.VALID
    
    def test_check_conditions_geometry(self):
        checks = check_conditions(
            tip_chord=0.15, root_chord=0.3, thickness=0.12, aspect_ratio=10
        )
        assert len(checks) == 3
        assert all(c.status == ValidityStatus.VALID for c in checks)


class TestConstantsUsed:
    def test_uses_constants_thresholds(self):
        """Verify the engine uses constants from constants.py."""
        engine = ModelValidityEngine()
        assert engine.mach_incompressible == const.MACH_INCOMPRESSIBLE_LIMIT
        assert engine.mach_transonic == const.MACH_TRANSONIC_LIMIT
        assert engine.re_laminar_limit == const.RE_LAMINAR_SEPARATION_LIMIT
        assert engine.re_upper_limit == const.RE_MODEL_UPPER_LIMIT


class TestSummary:
    def test_summary_format(self):
        engine = ModelValidityEngine()
        checks = [
            ValidityCheck("Mach Number", ValidityStatus.VALID, "M=0.2 valid", 0.3, 0.2),
            ValidityCheck("Reynolds Number", ValidityStatus.WARNING, "Re=1e4 low", 5e4, 1e4),
        ]
        summary = engine.summary(checks)
        assert "MODEL VALIDITY: WARNING" in summary
        assert "✓ Mach Number" in summary
        assert "⚠ Reynolds Number" in summary