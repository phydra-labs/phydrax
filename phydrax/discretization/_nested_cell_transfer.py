#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from .._fingerprint import canonical_fingerprint
from ..linalg import adjoint, ArraySpace, FunctionLinearOperator
from ._transfer import FieldTransfer, TransferProperties
from .finite_volume._unstructured import UnstructuredFiniteVolumeDiscretization


def nested_cell_transfer(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    parent_cells: ArrayLike,
    *,
    tolerance: float = 1e-9,
) -> FieldTransfer:
    """Conservative prolongation on a supplied conforming convex-cell refinement.

    Parent identity is explicit, not inferred from nearest centers. Host checks
    qualify child containment, volume partition and parent-boundary closure.
    Mesh validity/nonoverlap is the input CellMesh contract. Arbitrary overlap,
    partial domains and nonconvex parents require a different transfer builder.
    Geometry/topology derivatives are deliberately not advertised.
    """
    if not isinstance(source, UnstructuredFiniteVolumeDiscretization) or not isinstance(
        target, UnstructuredFiniteVolumeDiscretization
    ):
        raise TypeError("Nested transfer requires prepared unstructured FV meshes.")
    if source.cell_dimension != 3 or target.cell_dimension != 3:
        raise ValueError("Nested cell transfer requires three-dimensional meshes.")
    if source.component_names != target.component_names:
        raise ValueError("Source and target cell components must match exactly.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Geometry tolerance must be positive and finite.")
    parents = np.asarray(parent_cells)
    nc = source.cell_volumes.size
    nt = target.cell_volumes.size
    if parents.shape != (nt,) or not np.issubdtype(parents.dtype, np.integer):
        raise ValueError("One integer parent is required per target cell.")
    if np.any(parents < 0) or np.any(parents >= nc):
        raise ValueError("Target parent indices are out of range.")
    source_volume = np.asarray(source.cell_volumes)
    target_volume = np.asarray(target.cell_volumes)
    summed = np.bincount(parents, weights=target_volume, minlength=nc)
    if not np.allclose(
        summed, source_volume, rtol=tolerance, atol=tolerance * np.min(source_volume)
    ):
        raise ValueError("Child volumes must partition every parent volume.")

    def cell_points(discretization):
        coordinates = np.asarray(discretization.vertices)
        return [
            coordinates[row[mask]]
            for block in discretization.mesh.blocks
            for row, mask in zip(
                np.asarray(block.vertices), np.asarray(block.vertex_valid), strict=True
            )
        ]

    source_points = cell_points(source)
    target_points = cell_points(target)
    if len(source_points) != nc or len(target_points) != nt:
        raise ValueError("Mesh block ordering does not match prepared cell spaces.")
    owner = np.asarray(source.owner_cells)
    neighbour = np.asarray(source.neighbour_cells)
    centers = np.asarray(source.face_centers)
    normal = np.asarray(source.area_vectors) / np.asarray(source.face_measures)[:, None]
    planes = []
    for cell in range(nc):
        faces = np.flatnonzero((owner == cell) | (neighbour == cell))
        outward = normal[faces] * np.where(owner[faces] == cell, 1.0, -1.0)[:, None]
        delta = source_points[cell][:, None, :] - centers[faces][None, :, :]
        length = source_volume[cell] ** (1.0 / 3.0)
        if np.any(np.sum(delta * outward[None, :, :], axis=-1) > tolerance * length):
            raise ValueError("Nested prolongation requires convex parent cells.")
        planes.append((centers[faces], outward, length))
    for cell, parent in enumerate(parents):
        plane_centers, outward, length = planes[parent]
        delta = target_points[cell][:, None, :] - plane_centers[None, :, :]
        if np.any(np.sum(delta * outward[None, :, :], axis=-1) > tolerance * length):
            raise ValueError("A child cell extends outside its declared parent.")
    target_owner = np.asarray(target.owner_cells)
    target_neighbour = np.asarray(target.neighbour_cells)
    target_centers = np.asarray(target.face_centers)
    for face, cell in enumerate(target_owner):
        neighbour_cell = target_neighbour[face]
        if neighbour_cell >= 0 and parents[cell] == parents[neighbour_cell]:
            continue
        for adjacent in (cell,) if neighbour_cell < 0 else (cell, neighbour_cell):
            plane_centers, outward, length = planes[parents[adjacent]]
            distance = np.abs(
                np.sum((target_centers[face] - plane_centers) * outward, axis=-1)
            )
            if not np.any(distance <= tolerance * length):
                raise ValueError(
                    "Child boundary leaves an unmatched internal parent boundary."
                )
    parents_array = jnp.asarray(parents, dtype=jnp.int32)
    source_space = source.cell_space.vector_space
    target_space = target.cell_space.vector_space
    if not isinstance(source_space, ArraySpace) or not isinstance(
        target_space, ArraySpace
    ):
        raise TypeError("Nested cell fields require array-valued vector spaces.")
    operator = FunctionLinearOperator(
        lambda values: values[parents_array],
        source=source_space,
        target=target_space,
        transpose_action=lambda values: (
            source_space.zeros().at[parents_array].add(values)
        ),
        operator_id=canonical_fingerprint(
            {
                "kind": "nested-cell-prolongation",
                "source": source.geometry_id,
                "target": target.geometry_id,
                "parents": parents.tolist(),
            }
        ),
    )
    return FieldTransfer(
        source.cell_space,
        target.cell_space,
        operator,
        dual_pullback_operator=adjoint(operator),
        hilbert_adjoint_operator=adjoint(operator),
        properties=TransferProperties(
            constant_preserving=True,
            conservative=True,
            positivity_preserving=True,
            nested=True,
            adjoint_paired=True,
            differentiable_geometry=False,
            exact_on=("cellwise-constant",),
        ),
    )


__all__ = ["nested_cell_transfer"]
