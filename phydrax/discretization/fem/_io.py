#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import meshio
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._mesh_file_profiles import resolve_mesh_file_profile
from ..._publication import publish_bytes
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
    file_profile: str | None = None,
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
    vertex_count = discretization.mesh.coordinates.shape[0]
    point_values = np.asarray(values[:vertex_count])
    mesh = meshio.Mesh(
        np.asarray(realized.coordinates),
        cells,
        point_data={str(field_name): point_values},
    )

    destination = Path(path).expanduser().absolute()
    profile = resolve_mesh_file_profile(
        destination,
        file_profile,
        direction="write",
    )
    if profile.carrier != "single-file":
        raise ValueError("Finite-element field export requires a single-file profile.")
    with TemporaryDirectory(prefix="phydrax-fe-field-") as temporary:
        staged = Path(temporary) / destination.name
        mesh.write(staged, file_format=profile.meshio_format)
        decoded = meshio.read(staged, file_format=profile.meshio_format)
        decoded_points = np.asarray(decoded.points)
        expected_points = np.asarray(mesh.points)
        if (
            decoded_points.shape[0] == expected_points.shape[0]
            and decoded_points.shape[1] > expected_points.shape[1]
            and np.all(decoded_points[:, expected_points.shape[1] :] == 0)
        ):
            decoded_points = decoded_points[:, : expected_points.shape[1]]
        if (
            not np.array_equal(decoded_points, expected_points)
            or field_name not in decoded.point_data
            or not np.array_equal(decoded.point_data[field_name], point_values)
        ):
            raise ValueError("Finite-element field codec changed exported data.")
        payload = staged.read_bytes()
    publish_bytes(
        destination,
        payload,
        maximum_bytes=max(len(payload), 1),
        mode="atomic_replace",
    )


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
