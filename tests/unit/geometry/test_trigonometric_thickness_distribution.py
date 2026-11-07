#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import PeriodicCell


_G = phx.geometry


def test_schwarz_p_sheet_chords_match_the_closed_form_normal_chords() -> None:
    # With lines at quarter offsets, every sampled mid-surface point of
    # cos X + cos Y + cos Z = 0 has a normal along a cube axis or a cube
    # diagonal. Along the axis the field is cos X, so the band chord is
    # arcsin(tau) / pi; along the diagonal it is 3 sin(2 pi t / sqrt 3), so the
    # chord is sqrt(3) arcsin(tau / 3) / pi.
    tau = 0.3
    sheet = _G.TrigonometricLevelSet(
        _G.TrigonometricTable.family("schwarz_p"),
        PeriodicCell(np.eye(3)),
        region="sheet",
        sheet=_G.TrigonometricSheetBand(tau),
    )

    distribution = sheet.thickness_distribution(lines_per_axis=4)

    axis_chord = math.asin(tau) / math.pi
    diagonal_chord = math.sqrt(3.0) * math.asin(tau / 3.0) / math.pi
    values = distribution.thicknesses
    near_axis = np.abs(values - axis_chord) <= 1.0e-12
    near_diagonal = np.abs(values - diagonal_chord) <= 1.0e-12
    assert distribution.unexited_count == 0
    assert np.all(near_axis | near_diagonal)
    assert np.any(near_axis) and np.any(near_diagonal)
    assert distribution.minimum == pytest.approx(diagonal_chord, abs=1.0e-12)
    assert distribution.maximum == pytest.approx(axis_chord, abs=1.0e-12)
    assert distribution.quantiles == pytest.approx(
        np.quantile(values, distribution.quantile_levels), abs=0.0
    )
    # The certified minimum 2 tau / sup ||grad F|| over the band is below every chord.
    gradient_bound = 2.0 * math.pi * math.sqrt(3.0)
    assert distribution.gradient_norm_upper >= gradient_bound * (1.0 - 1.0e-12)
    assert distribution.uniform_minimum <= 2.0 * tau / gradient_bound
    assert distribution.uniform_minimum == pytest.approx(
        2.0 * tau / distribution.gradient_norm_upper, rel=1.0e-12
    )
    assert distribution.uniform_minimum <= distribution.certified_minimum
    assert distribution.certified_minimum <= distribution.minimum
    # Mid-surface points lie on F = 0.
    angles = 2.0 * math.pi * distribution.points
    assert np.all(np.abs(np.sum(np.cos(angles), axis=1)) <= 1.0e-12)


def test_wall_thickness_target_bounds_every_measured_chord_on_a_sheared_cell() -> None:
    cell = PeriodicCell(np.asarray(((1.0, 0.0, 0.0), (0.25, 1.1, 0.0), (0.0, 0.2, 0.9))))
    target = 0.12
    sheet = _G.TrigonometricLevelSet(
        _G.TrigonometricTable.family("gyroid"),
        cell,
        region="sheet",
        sheet=_G.TrigonometricWallThickness(target, resolution_level=5),
    )

    distribution = sheet.thickness_distribution(lines_per_axis=6)

    assert distribution.sample_count > 100
    assert distribution.unexited_count == 0
    assert distribution.certified_minimum >= target
    assert distribution.minimum >= distribution.certified_minimum
    assert distribution.minimum <= distribution.quantiles[0]
    assert np.all(np.diff(distribution.quantiles) >= 0.0)
    assert distribution.quantiles[-1] <= distribution.maximum
    # The sheet is genuinely nonuniform: the widest chord exceeds the thinnest.
    assert distribution.maximum > 1.2 * distribution.minimum


def test_only_a_sheet_reports_a_thickness_distribution() -> None:
    network = _G.TrigonometricLevelSet(
        _G.TrigonometricTable.family("schwarz_p"), PeriodicCell(np.eye(3))
    )

    with pytest.raises(ValueError, match="Only a sheet region"):
        network.thickness_distribution()
