from fractions import Fraction

import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.geometry.brep import boolean_brep, brep_box, BRepModel
from tools.meshing_qualification.scenarios._rectilinear_oracle import (
    Box,
    validate_rectilinear_result,
)


def _box(
    low: tuple[int | float, int | float, int | float],
    high: tuple[int | float, int | float, int | float],
) -> Box:
    return (
        (Fraction(low[0]), Fraction(low[1]), Fraction(low[2])),
        (Fraction(high[0]), Fraction(high[1]), Fraction(high[2])),
    )


def _model(box: Box, source_id: str) -> BRepModel:
    points = np.asarray(box, dtype=np.float64)
    return brep_box(
        points[0],
        points[1],
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id=source_id,
    )


def test_modified_enclosure_matches_exact_boundary_and_second_moments() -> None:
    outer = _box((0, 0, 0), (4, 3, 2))
    cavity = _box((0.25, 0.25, 0.25), (3.75, 2.75, 2.25))
    slot = _box((1.75, -0.25, 0.5), (2.25, 0.5, 1.5))
    first = boolean_brep(_model(outer, "outer"), _model(cavity, "cavity"), "difference")
    assert first.model is not None
    second = boolean_brep(first.model, _model(slot, "slot"), "difference")
    assert second.model is not None
    reference = validate_rectilinear_result(
        second.model,
        (
            (1, outer),
            (-1, _box((0.25, 0.25, 0.25), (3.75, 2.75, 2))),
            (-1, _box((1.75, 0, 0.5), (2.25, 0.25, 1.5))),
        ),
    )
    assert reference.volume == Fraction(137, 16)
    assert reference.area == Fraction(291, 4)
    assert reference.first_moments == (
        Fraction(137, 8),
        Fraction(833, 64),
        Fraction(851, 128),
    )


def test_oracle_rejects_equal_volume_translated_geometry() -> None:
    expected = _box((0, 0, 0), (1, 1, 1))
    displaced = _model(_box((2, 0, 0), (3, 1, 1)), "displaced")
    with pytest.raises(ValueError, match="source boundary differs"):
        validate_rectilinear_result(displaced, ((1, expected),))


def test_oracle_rejects_equal_volume_equal_centroid_wrong_cavity() -> None:
    outer = _box((-2, -2, -2), (2, 2, 2))
    intended = _box((-1, -0.5, -0.5), (1, 0.5, 0.5))
    rotated = _box((-0.5, -1, -0.5), (0.5, 1, 0.5))
    result = boolean_brep(
        _model(outer, "outer"), _model(rotated, "rotated-cavity"), "difference"
    )
    assert result.model is not None
    with pytest.raises(ValueError, match="source boundary differs"):
        validate_rectilinear_result(result.model, ((1, outer), (-1, intended)))


def test_body_split_matches_two_disconnected_exact_regions() -> None:
    outer = _box((0, 0, 0), (4, 1, 1))
    cutter = _box((1.5, -1, -1), (2.5, 2, 2))
    result = boolean_brep(
        _model(outer, "outer"), _model(cutter, "splitter"), "difference"
    )
    assert result.model is not None
    assert result.model.topology.num_solids == 2
    reference = validate_rectilinear_result(
        result.model,
        (
            (1, _box((0, 0, 0), (1.5, 1, 1))),
            (1, _box((2.5, 0, 0), (4, 1, 1))),
        ),
    )
    assert reference.volume == 3
    assert reference.first_moments == (6, Fraction(3, 2), Fraction(3, 2))
