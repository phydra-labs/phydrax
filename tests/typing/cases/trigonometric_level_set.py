"""Trigonometric nodal level-set source consumer types."""

from typing import assert_type

import phydrax as phx


def trigonometric_sources(cell: phx.discretization.PeriodicCell) -> None:
    table = phx.geometry.TrigonometricTable.family("gyroid")
    assert_type(table, phx.geometry.TrigonometricTable)
    levels = phx.geometry.certify_trigonometric_critical_levels(
        table, value_resolution=2.0**-10
    )
    assert_type(levels, phx.geometry.TrigonometricCriticalLevels)
    assert_type(levels.enclosure_meeting(0.0), tuple[float, float] | None)
    separation = levels.separate(0.25)
    assert_type(separation, phx.geometry.TrigonometricLevelSeparation)
    assert_type(separation.outcome, phx.geometry.TrigonometricLevelOutcome)
    assert_type(separation.critical_levels, phx.geometry.TrigonometricCriticalLevels)
    assert_type(separation.evidence_value, tuple[float, float] | None)
    phx.geometry.certify_trigonometric_critical_levels(table, value_tolerance=1.0)  # ty: ignore[unknown-argument]
    sheet = phx.geometry.TrigonometricLevelSet(
        table,
        cell,
        region="sheet",
        sheet=phx.geometry.TrigonometricWallThickness(0.1),
        critical_levels=levels,
    )
    assert_type(sheet.half_width, float | None)
    assert_type(sheet.sheet_thickness, phx.geometry.TrigonometricSheetThickness | None)
    thickness = phx.geometry.certify_trigonometric_sheet_thickness(
        table, cell, level=0.0, maximum_half_width=0.4
    )
    assert_type(thickness.half_width_for_thickness(0.1), float)
    assert_type(table.symmetry_sign(((-1, 0, 0), (0, -1, 0), (0, 0, -1)), (0, 0, 0)), int)
    phx.geometry.TrigonometricLevelSet(table, cell, region="solid")  # ty: ignore[invalid-argument-type]
    phx.geometry.TrigonometricTable.family("diamond")  # ty: ignore[invalid-argument-type]
