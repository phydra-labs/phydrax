#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import meshio
import numpy as np
from jaxtyping import Array, ArrayLike

from ._generic import FiniteElementDiscretization, FiniteElementRuntimeData


_MESHIO_CELL_KINDS = {
    "triangle": "triangle",
    "quadrilateral": "quad",
    "tetrahedron": "tetra",
}


def write_finite_element_field(
    path: str | Path,
    discretization: FiniteElementDiscretization,
    field_name: str,
    coefficients: ArrayLike,
    /,
    *,
    runtime: FiniteElementRuntimeData | None = None,
) -> None:
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    field_index = discretization._field_index(field_name)
    if discretization.dof_maps[field_index].association not in (
        "vertex",
        "vertex_edge",
    ):
        raise ValueError("Point-data export currently requires an H1 nodal field.")
    values = discretization.field_spaces[field_index].vector_space.validate(coefficients)
    realized = discretization.default_runtime if runtime is None else runtime
    cells = [
        (_MESHIO_CELL_KINDS[block.cell_kind], np.asarray(block.vertices))
        for block in discretization.mesh.blocks
    ]
    vertex_count = int(discretization.mesh.coordinates.shape[0])
    point_values = np.asarray(values[:vertex_count])
    mesh = meshio.Mesh(
        np.asarray(realized.coordinates),
        cells,
        point_data={str(field_name): point_values},
    )
    mesh.write(Path(path))


def evaluate_finite_element_field(
    discretization: FiniteElementDiscretization,
    field_name: str,
    coefficients: ArrayLike,
    block_name: str,
    reference_points: ArrayLike,
    /,
    *,
    runtime: FiniteElementRuntimeData | None = None,
) -> Array:
    return jnp.asarray(
        discretization.reconstruct(
            field_name,
            coefficients,
            block_name,
            reference_points,
            runtime=runtime,
        )
    )


__all__ = ["evaluate_finite_element_field", "write_finite_element_field"]
