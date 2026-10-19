#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Coordinate scope records validate their map classification without copying banks."""

from fractions import Fraction

import numpy as np
import pytest

from phydrax.geometry import CoordinateMapScope, CoordinateScope


@pytest.mark.parametrize("scope", ("affine", "mapped"))
def test_coordinate_scope_record_preserves_source_and_cell_bank_pointers(
    scope: CoordinateMapScope,
) -> None:
    cells = (
        np.empty((0,), dtype=np.int64)
        if scope == "affine"
        else np.asarray((3, 7), dtype=np.int64)
    )
    source = ((Fraction(1, 3), Fraction(2, 7)),)
    record = CoordinateScope(scope, cells, source)

    assert record.coordinate_scope == scope
    assert record.mapped_cells is cells
    assert record.source_values is source
    assert record.source_values[0] == (Fraction(1, 3), Fraction(2, 7))


@pytest.mark.parametrize(
    ("scope", "cells", "diagnostic"),
    (
        pytest.param(
            "affine",
            np.asarray((1,), dtype=np.int64),
            "CoordinateScope.coordinate_scope",
            id="affine-with-mapped-cell",
        ),
        pytest.param(
            "mapped",
            np.empty((0,), dtype=np.int64),
            "CoordinateScope.coordinate_scope",
            id="mapped-with-no-cell",
        ),
        pytest.param(
            "mapped",
            np.asarray(((1,),), dtype=np.int64),
            "CoordinateScope.mapped_cells",
            id="matrix-not-vector",
        ),
        pytest.param(
            "mapped",
            np.asarray((1,), dtype=np.int32),
            "CoordinateScope.mapped_cells",
            id="wrong-host-dtype",
        ),
    ),
)
def test_coordinate_scope_record_refuses_inconsistent_cell_classification(
    scope: CoordinateMapScope,
    cells: np.ndarray,
    diagnostic: str,
) -> None:
    source = ((Fraction(0), Fraction(1)),)
    with pytest.raises(ValueError, match=diagnostic):
        CoordinateScope(scope, cells, source)
