#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    ConstraintMap,
    DenseLinearOperator,
    FunctionLinearOperator,
)
from ._generic import FiniteElementDiscretization


class FiniteElementDirichletConstraint(StrictModule, NonTrainableState):
    """Strong essential constraint resolved onto one prepared FE field."""

    field_name: str = eqx.field(static=True)
    constraint_map: ConstraintMap
    constrained_dofs: Array
    free_dofs: Array
    dof_coordinates: Array
    constraint_id: str = eqx.field(static=True)

    def lift(
        self,
        values: ArrayLike | Callable[[Array], ArrayLike],
        /,
    ) -> Array:
        evaluated = values(self.dof_coordinates) if callable(values) else values
        raw = jnp.asarray(evaluated)
        full_space = self.constraint_map.full_space
        if not isinstance(full_space, ArraySpace):
            raise TypeError("Finite-element Dirichlet lifts require an ArraySpace.")
        full_shape = full_space.shape
        full_size = int(np.prod(full_shape, dtype=int))
        if raw.shape == ():
            full = jnp.broadcast_to(raw, full_shape).reshape((full_size,))
        elif raw.shape == full_shape:
            full = raw.reshape((full_size,))
        elif raw.shape == (int(self.constrained_dofs.size),):
            full = (
                jnp.zeros((full_size,), dtype=raw.dtype)
                .at[self.constrained_dofs]
                .set(raw)
            )
        else:
            raise ValueError(
                "Dirichlet values must be scalar, full-space shaped, or contain "
                "one value per constrained coordinate."
            )
        zeros = jnp.zeros((full_size,), dtype=full.dtype)
        return (
            zeros.at[self.constrained_dofs]
            .set(full[self.constrained_dofs])
            .reshape(full_shape)
        )


def _validate_component_constraints(
    discretization: FiniteElementDiscretization,
    dof_map,
    mask: np.ndarray,
    /,
) -> None:
    vertex_count = int(discretization.mesh.coordinates.shape[0])
    parents = np.arange(vertex_count, dtype=np.int32)

    def root(value: int) -> int:
        current = int(value)
        while parents[current] != current:
            parents[current] = parents[parents[current]]
            current = int(parents[current])
        return current

    def union(first: int, second: int) -> None:
        first_root = root(first)
        second_root = root(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for block in discretization.mesh.blocks:
        for cell in np.asarray(block.vertices, dtype=np.int32):
            anchor = int(cell[0])
            for vertex in cell[1:]:
                union(anchor, int(vertex))
    component_roots = {root(vertex) for vertex in range(vertex_count)}
    constrained_roots = set()
    for block, routes in zip(
        discretization.mesh.blocks,
        dof_map.cell_dofs,
        strict=True,
    ):
        cells = np.asarray(block.vertices, dtype=np.int32)
        active = np.any(mask[np.asarray(routes, dtype=np.int32)], axis=1)
        constrained_roots.update(
            root(int(cells[cell_index, 0])) for cell_index in np.flatnonzero(active)
        )
    if constrained_roots != component_roots:
        raise ValueError(
            "Dirichlet constraints must anchor every connected mesh component."
        )


def dirichlet_constraint(
    discretization: FiniteElementDiscretization,
    field_name: str,
    /,
    *,
    boundary_mask: ArrayLike | None = None,
    components: Sequence[int] | None = None,
) -> FiniteElementDirichletConstraint:
    """Resolve one reduced-coordinate strong Dirichlet constraint."""

    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    field_index = discretization._field_index(field_name)
    field_space = discretization.field_spaces[field_index]
    dof_map = discretization.dof_maps[field_index]
    if not isinstance(field_space.vector_space, ArraySpace):
        raise ValueError("Finite-element Dirichlet constraints require ArraySpace.")
    node_mask = (
        np.asarray(dof_map.boundary_dof_mask, dtype=bool)
        if boundary_mask is None
        else np.asarray(boundary_mask, dtype=bool)
    )
    full_shape = field_space.vector_space.shape
    component_count = int(np.prod(full_shape[1:], dtype=int)) if full_shape[1:] else 1
    if node_mask.shape == full_shape:
        full_mask = node_mask.reshape((dof_map.global_dof_count, component_count))
        node_mask = np.any(full_mask, axis=1)
    elif node_mask.shape == (dof_map.global_dof_count,):
        selected_components = (
            np.arange(component_count, dtype=np.int32)
            if components is None
            else np.asarray(tuple(int(value) for value in components), dtype=np.int32)
        )
        if (
            selected_components.ndim != 1
            or selected_components.size == 0
            or np.any(selected_components < 0)
            or np.any(selected_components >= component_count)
            or np.unique(selected_components).size != selected_components.size
        ):
            raise ValueError("components must select unique valid flattened components.")
        full_mask = np.zeros(
            (dof_map.global_dof_count, component_count),
            dtype=bool,
        )
        full_mask[:, selected_components] = node_mask[:, None]
    else:
        raise ValueError("boundary_mask must have global-DOF shape or full field shape.")
    _validate_component_constraints(discretization, dof_map, node_mask)
    flattened_mask = full_mask.reshape((-1,))
    constrained = np.flatnonzero(flattened_mask).astype(np.int32)
    free = np.flatnonzero(~flattened_mask).astype(np.int32)
    if constrained.size == 0 or free.size == 0:
        raise ValueError(
            "Dirichlet constraints require a non-empty proper subset of DOFs."
        )
    reduced_space = ArraySpace((free.size,), pairing=None)
    full_space = field_space.vector_space
    free_array = jnp.asarray(free)
    full_size = int(np.prod(full_space.shape, dtype=int))
    prolongation = FunctionLinearOperator(
        lambda reduced: (
            jnp.zeros((full_size,), dtype=reduced.dtype)
            .at[free_array]
            .set(reduced, unique_indices=True)
            .reshape(full_space.shape)
        ),
        source=reduced_space,
        target=full_space,
        transpose_action=lambda full: full.reshape((full_size,))[free_array],
        operator_id=canonical_fingerprint(
            {
                "kind": "finite-element-dirichlet-prolongation",
                "field_space": field_space.field_space_id,
                "free_dofs": free.tolist(),
            }
        ),
    )
    constraint_map = ConstraintMap(
        full_space,
        reduced_space,
        prolongation,
        constraint_id=canonical_fingerprint(
            {
                "kind": "finite-element-dirichlet-constraint",
                "field_space": field_space.field_space_id,
                "constrained_dofs": constrained.tolist(),
            }
        ),
    )
    return FiniteElementDirichletConstraint(
        field_name=field_space.name,
        constraint_map=constraint_map,
        constrained_dofs=jnp.asarray(constrained),
        free_dofs=free_array,
        dof_coordinates=dof_map.dof_coordinates,
        constraint_id=constraint_map.constraint_id,
    )


def affine_dof_constraint(
    discretization: FiniteElementDiscretization,
    field_name: str,
    prolongation_matrix: ArrayLike,
    /,
    *,
    constraint_id: str | None = None,
) -> ConstraintMap:
    """Create periodic, multipoint, or hanging-node affine coordinates."""

    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    field_index = discretization._field_index(field_name)
    full_space = discretization.field_spaces[field_index].vector_space
    matrix = jnp.asarray(prolongation_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != full_space.size:
        raise ValueError(
            "prolongation_matrix must map reduced coordinates to the full field."
        )
    reduced_space = ArraySpace((int(matrix.shape[1]),), dtype=matrix.dtype)
    operator = DenseLinearOperator(
        matrix,
        source=reduced_space,
        target=full_space,
        operator_id=canonical_fingerprint(
            {
                "kind": "finite-element-affine-dof-prolongation",
                "field_space": discretization.field_spaces[field_index].field_space_id,
                "matrix_shape": list(matrix.shape),
            }
        ),
    )
    return ConstraintMap(
        full_space,
        reduced_space,
        operator,
        constraint_id=constraint_id,
    )


__all__ = [
    "FiniteElementDirichletConstraint",
    "affine_dof_constraint",
    "dirichlet_constraint",
]
