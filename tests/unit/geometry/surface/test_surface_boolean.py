#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax._meshcore import meshcore_available
from phydrax.geometry.surface import (
    surface_boolean,
    SurfaceBooleanError,
    SurfaceBooleanOperation,
    SurfaceMetadata,
    SurfaceModel,
)


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(), reason="native phydrax-meshcore library is unavailable"
    ),
]


_VERTICES = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
)
_FACES = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int64)


def _tetrahedron(name: str, faces: np.ndarray = _FACES) -> SurfaceModel:
    metadata = SurfaceMetadata(
        source_id=name,
        source_revision="r1",
        coordinate_contract=SpatialCoordinateContract.si(),
        provenance=("surface-boolean-test",),
        cell_tags=("tetra-face",) * faces.shape[0],
    )
    return SurfaceModel.from_triangles(
        _VERTICES,
        faces,
        metadata,
        cell_global_ids=np.arange(11, 11 + faces.shape[0], dtype=np.int64),
    )


def test_identical_solids_resolve_all_operations_and_keep_face_ancestry() -> None:
    first = _tetrahedron("first")
    second = _tetrahedron("second")

    union = surface_boolean(first, second, SurfaceBooleanOperation.UNION)
    intersection = surface_boolean(first, second, SurfaceBooleanOperation.INTERSECTION)
    difference = surface_boolean(first, second, SurfaceBooleanOperation.DIFFERENCE)

    assert not union.empty and union.surface is not None
    assert union.closure.closed and union.closure.edge_manifold
    assert set(union.source_operand.tolist()) == {0}
    assert set(union.source_cell_global_ids.tolist()) == {11, 12, 13, 14}
    assert union.surface.metadata.cell_tags == ("tetra-face",) * union.triangles.shape[0]
    assert not intersection.empty and intersection.surface is not None
    assert intersection.closure.closed
    assert difference.empty and difference.surface is None
    assert difference.vertices.shape == (0, 3)

    values = _VERTICES @ np.asarray((1.0, -2.0, 0.5))
    transferred = union.source_corner_values(values, values)
    expected = union.vertices[union.triangles] @ np.asarray((1.0, -2.0, 0.5))
    np.testing.assert_allclose(transferred, expected, rtol=0.0, atol=1e-14)


def test_nary_boolean_combines_all_triangle_operands() -> None:
    first = _tetrahedron("first")
    second = _tetrahedron("second")
    third = _tetrahedron("third")

    union = surface_boolean(
        first,
        second,
        SurfaceBooleanOperation.UNION,
        operands=(third,),
    )
    difference = surface_boolean(
        first,
        second,
        SurfaceBooleanOperation.DIFFERENCE,
        operands=(third,),
    )

    assert set(union.source_operand.tolist()) == {0}
    assert union.triangles.shape[0] == _FACES.shape[0]
    assert difference.empty


def test_open_and_inward_operands_are_refused_not_repaired() -> None:
    solid = _tetrahedron("solid")
    open_surface = _tetrahedron("open", _FACES[:-1])
    inward = _tetrahedron("inward", _FACES[:, ::-1].copy())

    with pytest.raises(SurfaceBooleanError) as open_refusal:
        surface_boolean(solid, open_surface, SurfaceBooleanOperation.UNION)
    assert open_refusal.value.status == "open_operand"
    assert open_refusal.value.operand == 1

    with pytest.raises(SurfaceBooleanError) as inward_refusal:
        surface_boolean(solid, inward, SurfaceBooleanOperation.UNION)
    assert inward_refusal.value.status == "operand_not_solid"
    assert inward_refusal.value.operand == 1
