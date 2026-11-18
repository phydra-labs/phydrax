from fractions import Fraction

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="exact layer certification requires meshcore"
)


def _rounded_prefix_sums(values: np.ndarray) -> np.ndarray:
    """Round each exact binary64 addition before starting the next one."""
    result = [0.0]
    for value in values:
        exact = Fraction.from_float(result[-1]) + Fraction.from_float(float(value))
        result.append(float(exact))
    return np.asarray(result, dtype=np.float64)


def test_layer_schedule_accumulates_in_index_order() -> None:
    # Every increment remains representable at its height, but the varied
    # magnitudes make the final binary64 sum depend on the order of addition.
    thicknesses = np.asarray(
        [0.01] + [0.0001, 0.001, 0.0003, 0.002] * 8, dtype=np.float64
    )
    expected = _rounded_prefix_sums(thicknesses)
    reverse = _rounded_prefix_sums(thicknesses[::-1])
    assert expected[-1].view(np.uint64) != reverse[-1].view(np.uint64)
    assert np.all(np.diff(expected) > 0.0)

    wall = phx.discretization.CellMesh(
        np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64),
        (
            phx.discretization.CellBlock(
                "triangles",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int64),
                global_ids=np.asarray((0,), dtype=np.int64),
            ),
        ),
    )
    cells = wall.entity_set(2)
    scope = phx.meshing.MeshingScope(
        wall.mesh_id,
        wall.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        cells.entity_set_id,
        cells.entity_ids,
    )
    control = phx.meshing.BoundaryLayerControl(
        scope,
        phx.meshing.LayerSchedule(thicknesses),
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
    )
    result = phx.meshing.prepare_boundary_layers(wall, control)

    heights, counts = np.unique(
        np.asarray(result.mesh.coordinates, dtype=np.float64)[:, 2], return_counts=True
    )
    np.testing.assert_array_equal(heights.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(counts, np.full(expected.shape, 3, dtype=np.int64))
    assert result.validity.certified_valid_count == thicknesses.size
