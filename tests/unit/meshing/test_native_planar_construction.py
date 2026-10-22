#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Uniform-median planar construction admission and bounds."""

import numpy as np
import pytest

from phydrax.meshing.providers._native_planar_construction import (
    LATTICE_FRACTION,
    lattice_site_upper,
    lattice_spacing,
    uniform_construction_size,
    uniform_median_strategy,
)
from tools._meshing_cases import meshing_case, prepare_case


def test_lattice_spacing_follows_the_admitted_uniform_median() -> None:
    _, specification, _ = prepare_case(meshing_case("planar-hole-feature"), 32, 20000, 120.0)
    size = specification.size_controls[0].target_size
    assert uniform_median_strategy(specification, size)
    upper = uniform_construction_size(specification, size)
    assert upper == pytest.approx(1.5 * size)
    assert lattice_spacing(specification, size) == LATTICE_FRACTION * upper
    assert lattice_spacing(specification, 2.0 * size) == 0.0
    assert uniform_construction_size(specification, 2.0 * size) == 2.0 * size


def test_lattice_site_upper_covers_every_visited_row_and_column() -> None:
    lower, upper = np.zeros(2), np.full(2, 2.0)
    spacing = 0.03375
    rows = int(np.floor(2.0 / (spacing * 0.5 * np.sqrt(3.0)))) + 1
    columns = int(np.floor(2.0 / spacing)) + 1
    assert lattice_site_upper(lower, upper, spacing) == columns * rows + rows
    assert lattice_site_upper(lower, upper, 0.0) == 0
