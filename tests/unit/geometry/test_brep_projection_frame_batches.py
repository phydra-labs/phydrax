from typing import Any, Literal

import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.brep._constructors import (
    brep_planar_face,
    brep_sphere,
    brep_torus,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)


_CONTRACT = phx.SpatialCoordinateContract.si()
_FACE = int(phx.geometry.BRepEntityDimension.FACE)
_FIELDS = (
    "dimensions",
    "indices",
    "queries",
    "points",
    "parameters",
    "residuals",
    "distance_lower_bounds",
    "status",
    "normals",
    "tangents",
)


def _model(kind: str) -> Any:
    if kind == "sphere":
        return brep_sphere(1.0, coordinate_contract=_CONTRACT)
    if kind == "torus":
        return brep_torus(2.0, 0.5, coordinate_contract=_CONTRACT)
    return brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=_CONTRACT,
    )


def _assert_rows_equal(actual: Any, expected: Any, order: np.ndarray) -> None:
    assert actual.source_id == expected.source_id
    assert actual.source_revision == expected.source_revision
    assert actual.frame_id == expected.frame_id
    assert actual.occurrence_paths == tuple(expected.occurrence_paths[i] for i in order)
    for field in _FIELDS:
        left = np.asarray(getattr(actual, field))
        right = np.asarray(getattr(expected, field))[order]
        assert left.shape == right.shape, field
        assert left.dtype == right.dtype, field
        assert left.tobytes() == right.tobytes(), field


@pytest.mark.parametrize(
    ("kind", "points"),
    [
        (
            "sphere",
            [
                [0.3, 0.4, 0.5],
                [-0.5, 0.0, 0.3],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, -0.5],
                [0.0, 0.6, 0.8],
                [0.0, 0.0, 0.0],
            ],
        ),
        (
            "torus",
            [
                [1.5, 0.0, 0.3],
                [-1.0, -2.0, -0.4],
                [0.0, 2.3, 0.2],
                [-3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        ),
        (
            "plane",
            [
                [0.25, 0.25, 0.125],
                [0.75, 0.75, 0.125],
                [0.5, 0.0, 0.125],
                [0.0, 0.0, 0.125],
                [np.nextafter(0.0, 1.0), 0.5, 0.125],
                [0.5, -1.0e-15, 0.125],
                [1.0e8, 1.0e8, 1.0e8],
            ],
        ),
    ],
    ids=["sphere-poles-center", "torus-seams-center", "plane-trim-extreme"],
)
def test_face_projection_frames_are_byte_identical_across_batches(
    kind: str,
    points: Any,
) -> None:
    projection = phx.geometry.prepare_brep_projection(_model(kind))
    queries = np.asarray(points, dtype=np.float64)
    dimensions = np.full(len(queries), _FACE, dtype=np.int8)
    indices = np.zeros(len(queries), dtype=np.int32)
    batch = projection.project(queries, dimensions, indices)

    selections = [
        np.arange(len(queries), dtype=np.int64)[::-1],
        np.arange(0, len(queries), 2, dtype=np.int64),
        np.arange(1, len(queries), 2, dtype=np.int64),
    ]
    selections.extend(np.asarray([row], dtype=np.int64) for row in range(len(queries)))
    for selection in selections:
        selected = projection.project(
            queries[selection], dimensions[selection], indices[selection]
        )
        _assert_rows_equal(selected, batch, selection)


@pytest.mark.parametrize("mode", ["ignore", "warn", "raise", "print"])
def test_regular_projection_frames_preserve_numpy_error_modes(
    mode: Literal["ignore", "warn", "raise", "print"],
) -> None:
    projection = phx.geometry.prepare_brep_projection(_model("torus"))
    points = np.asarray(
        [[1.5, 0.2, 0.3], [-1.0, -2.0, -0.4], [0.3, 2.3, 0.2]],
        dtype=np.float64,
    )
    dimensions = np.full(len(points), _FACE, dtype=np.int8)
    indices = np.zeros(len(points), dtype=np.int64)
    with np.errstate(all=mode):
        batch = projection.project(points, dimensions, indices)
        for row in range(len(points)):
            selection = np.asarray([row], dtype=np.int64)
            scalar = projection.project(
                points[selection], dimensions[selection], indices[selection]
            )
            _assert_rows_equal(scalar, batch, selection)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_projection_rejects_nonfinite_queries_in_batch_and_single_row(
    invalid: float,
) -> None:
    projection = phx.geometry.prepare_brep_projection(_model("plane"))
    points = np.asarray([[0.3, 0.4, 1.0], [invalid, 0.0, 1.0]])
    for queries in (points, points[1:]):
        with pytest.raises(ValueError, match="finite"):
            projection.project(
                queries,
                np.full(len(queries), _FACE),
                np.zeros(len(queries), dtype=np.int64),
            )
