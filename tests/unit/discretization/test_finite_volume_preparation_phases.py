#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Public FV preparation records preserve geometry and boundary evidence."""

from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.discretization import CellBlock, CellMesh, UnstructuredFiniteVolumePlan


# Quadrature and quality preparation use at most 256 eps of accumulated
# floating-point error at unit geometric scale; no approximation error is present.
PREPARATION_ROUNDOFF = 256.0 * np.finfo(np.float64).eps
pytestmark = pytest.mark.strict_jax


@pytest.mark.parametrize(
    ("kind", "vertices", "cells", "volume", "face_measure", "aspect"),
    (
        (
            "quadrilateral",
            np.asarray(
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
            ),
            np.asarray(((0, 1, 2, 3),), dtype=np.int32),
            1.0,
            1.0,
            1.0,
        ),
        (
            "tetrahedron",
            np.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            ),
            np.asarray(((0, 1, 2, 3),), dtype=np.int32),
            1.0 / 6.0,
            0.5,
            np.sqrt(6.0),
        ),
    ),
    ids=("unit-square", "right-tetrahedron"),
)
def test_cell_mesh_quality_matches_independent_geometry(
    kind: Literal["quadrilateral", "tetrahedron"],
    vertices: NDArray[np.float64],
    cells: NDArray[np.int32],
    volume: float,
    face_measure: float,
    aspect: float,
) -> None:
    mesh = CellMesh(vertices, (CellBlock("cells", kind, cells),))
    prepared = UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare()
    quality = prepared.quality
    np.testing.assert_allclose(
        quality.minimum_cell_measure, volume, rtol=0.0, atol=PREPARATION_ROUNDOFF
    )
    np.testing.assert_allclose(
        quality.maximum_cell_measure, volume, rtol=0.0, atol=PREPARATION_ROUNDOFF
    )
    np.testing.assert_allclose(
        quality.minimum_face_measure, face_measure, rtol=0.0, atol=PREPARATION_ROUNDOFF
    )
    np.testing.assert_allclose(
        quality.maximum_aspect_ratio, aspect, rtol=0.0, atol=PREPARATION_ROUNDOFF
    )
    np.testing.assert_allclose(
        quality.maximum_closure_residual, 0.0, rtol=0.0, atol=PREPARATION_ROUNDOFF
    )
    np.testing.assert_array_equal(quality.maximum_nonorthogonality_degrees, 0.0)
    np.testing.assert_array_equal(quality.worst_cell, 0)
    np.testing.assert_allclose(
        np.sum(prepared.cell_quadrature_weights, axis=1),
        prepared.cell_volumes,
        rtol=0.0,
        atol=PREPARATION_ROUNDOFF,
    )


def test_direct_plan_retains_ordered_ids_and_boundary_partition() -> None:
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    vertex_ids = np.asarray((19, 7, 31, 2), dtype=np.int64)
    cell_ids = np.asarray((41,), dtype=np.int64)
    plan = UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=cells,
        vertex_global_ids=vertex_ids,
        cell_global_ids=cell_ids,
        boundary_patches={
            "west": np.asarray(((3, 0),), dtype=np.int32),
            "rest": np.asarray(((0, 1), (1, 2), (2, 3)), dtype=np.int32),
        },
    )
    prepared = plan.prepare()
    np.testing.assert_array_equal(prepared.vertex_global_ids, vertex_ids)
    np.testing.assert_array_equal(prepared.cell_global_ids, cell_ids)
    assert prepared.boundary_patch_names == ("rest", "west")
    np.testing.assert_array_equal(np.sort(prepared.boundary_patch_ids), (0, 0, 0, 1))
    np.testing.assert_array_equal(prepared.neighbor_cells, (-1, -1, -1, -1))
    assert prepared.prepared_id == plan.prepare().prepared_id
