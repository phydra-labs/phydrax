#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, ConstraintMap, FunctionLinearOperator
from .._cell_complex import PolygonalConnectivity
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
        if raw.shape == ():
            full = jnp.broadcast_to(raw, full_shape)
        elif raw.shape == full_shape:
            full = raw
        elif raw.shape == (int(self.constrained_dofs.size),) and len(full_shape) == 1:
            full = (
                jnp.zeros(full_shape, dtype=raw.dtype).at[self.constrained_dofs].set(raw)
            )
        else:
            raise ValueError(
                "Dirichlet values must be scalar, full-space shaped, or contain "
                "one scalar per constrained DOF."
            )
        zeros = jnp.zeros(full_shape, dtype=full.dtype)
        return zeros.at[self.constrained_dofs].set(full[self.constrained_dofs])


def _validate_component_constraints(
    discretization: FiniteElementDiscretization,
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
    constrained_roots = {root(int(dof)) for dof in np.flatnonzero(mask[:vertex_count])}
    if mask.size > vertex_count:
        connectivity = discretization.mesh.connectivity
        if not isinstance(connectivity, PolygonalConnectivity):
            raise TypeError("Higher-order constraints require polygonal connectivity.")
        for edge in np.flatnonzero(mask[vertex_count:]):
            constrained_roots.add(
                root(int(np.asarray(connectivity.edges, dtype=np.int32)[edge, 0]))
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
) -> FiniteElementDirichletConstraint:
    """Resolve one reduced-coordinate strong Dirichlet constraint."""

    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    field_index = discretization._field_index(field_name)
    field_space = discretization.field_spaces[field_index]
    dof_map = discretization.dof_maps[field_index]
    if (
        not isinstance(field_space.vector_space, ArraySpace)
        or len(field_space.vector_space.shape) != 1
    ):
        raise ValueError(
            "The initial Dirichlet builder supports scalar FE field coordinates."
        )
    mask = (
        np.asarray(dof_map.boundary_dof_mask, dtype=bool)
        if boundary_mask is None
        else np.asarray(boundary_mask, dtype=bool)
    )
    if mask.shape != (dof_map.global_dof_count,):
        raise ValueError("boundary_mask must contain one value per global DOF.")
    _validate_component_constraints(discretization, mask)
    constrained = np.flatnonzero(mask).astype(np.int32)
    free = np.flatnonzero(~mask).astype(np.int32)
    if constrained.size == 0 or free.size == 0:
        raise ValueError(
            "Dirichlet constraints require a non-empty proper subset of DOFs."
        )
    reduced_space = ArraySpace((free.size,), pairing=None)
    full_space = field_space.vector_space
    free_array = jnp.asarray(free)
    prolongation = FunctionLinearOperator(
        lambda reduced: (
            jnp.zeros(full_space.shape, dtype=reduced.dtype).at[free_array].set(reduced)
        ),
        source=reduced_space,
        target=full_space,
        transpose_action=lambda full: full[free_array],
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


__all__ = ["FiniteElementDirichletConstraint", "dirichlet_constraint"]
