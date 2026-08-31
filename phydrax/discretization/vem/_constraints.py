#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, ConstraintMap, FunctionLinearOperator
from .._integration_domain import IntegrationDomain
from ._space import VirtualElementDiscretization


class VirtualElementDirichletConstraint(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    constraint_map: ConstraintMap
    constrained_dofs: Array
    free_dofs: Array
    dof_coordinates: Array
    constraint_id: str = eqx.field(static=True)

    def lift(self, values: ArrayLike | Callable[[Array], ArrayLike], /) -> Array:
        if callable(values):
            evaluator = cast(Callable[[Array], ArrayLike], values)
            raw = jnp.asarray(evaluator(self.dof_coordinates))
        else:
            raw = jnp.asarray(values)
        full_space = self.constraint_map.full_space
        if not isinstance(full_space, ArraySpace):
            raise TypeError("VEM Dirichlet lifts require ArraySpace.")
        full_size = full_space.size
        if raw.shape == ():
            constrained_values = jnp.broadcast_to(raw, self.constrained_dofs.shape)
        elif raw.shape == self.constrained_dofs.shape:
            constrained_values = raw
        elif raw.shape == full_space.shape:
            constrained_values = raw.reshape((-1,))[self.constrained_dofs]
        else:
            raise ValueError(
                "VEM Dirichlet values must be scalar, full-space, or constrained-size."
            )
        return (
            jnp.zeros((full_size,), dtype=constrained_values.dtype)
            .at[self.constrained_dofs]
            .set(constrained_values)
            .reshape(full_space.shape)
        )


def _component_roots(discretization: VirtualElementDiscretization, /) -> np.ndarray:
    vertex_count = int(discretization.mesh.coordinates.shape[0])
    parents = np.arange(vertex_count, dtype=np.int32)

    def root(value: int) -> int:
        current = int(value)
        while parents[current] != current:
            parents[current] = parents[parents[current]]
            current = int(parents[current])
        return current

    for block in discretization.mesh.blocks:
        for cell in np.asarray(block.vertices, dtype=np.int32):
            first = root(int(cell[0]))
            for vertex in cell[1:]:
                second = root(int(vertex))
                if first != second:
                    parents[second] = first
    return np.asarray([root(index) for index in range(vertex_count)], dtype=np.int32)


def virtual_element_dirichlet_constraint(
    discretization: VirtualElementDiscretization,
    field_name: str,
    /,
    *,
    boundary_mask: ArrayLike | None = None,
    domain: IntegrationDomain | None = None,
) -> VirtualElementDirichletConstraint:
    if not isinstance(discretization, VirtualElementDiscretization):
        raise TypeError("discretization must be VirtualElementDiscretization.")
    if str(field_name) != discretization.field.name:
        raise KeyError(f"Unknown virtual-element field {field_name!r}.")
    if boundary_mask is not None and domain is not None:
        raise ValueError("Specify boundary_mask or domain, not both.")
    mask = np.asarray(discretization.dof_map.boundary_dof_mask, dtype=bool)
    if domain is not None:
        if domain.kind != "exterior_facet":
            raise ValueError("VEM Dirichlet domains must select exterior facets.")
        mask = np.zeros_like(mask)
        edges = np.asarray(discretization.mesh.connectivity.edges, dtype=np.int32)
        selected = np.asarray(domain.entity_indices, dtype=np.int32)
        mask[np.unique(edges[selected].reshape((-1,)))] = True
        edge_width = discretization.field.element.edge_interior_dof_count
        if edge_width:
            offset = discretization.dof_map.vertex_dof_count
            for edge in selected:
                mask[
                    offset + int(edge) * edge_width : offset
                    + (int(edge) + 1) * edge_width
                ] = True
    elif boundary_mask is not None:
        mask = np.asarray(boundary_mask, dtype=bool)
        if mask.shape != (discretization.dof_map.global_dof_count,):
            raise ValueError("boundary_mask must have global VEM DOF shape.")
    constrained = np.flatnonzero(mask).astype(np.int32)
    free = np.flatnonzero(~mask).astype(np.int32)
    if constrained.size == 0 or free.size == 0:
        raise ValueError("VEM Dirichlet constraints require a nonempty proper subset.")
    roots = _component_roots(discretization)
    component_roots = set(int(value) for value in roots)
    constrained_vertices = constrained[
        constrained < discretization.dof_map.vertex_dof_count
    ]
    if {int(roots[index]) for index in constrained_vertices} != component_roots:
        raise ValueError("VEM Dirichlet constraints must anchor every mesh component.")
    full_space = discretization.field_space.vector_space
    reduced_space = ArraySpace((free.size,), pairing=None)
    free_array = jnp.asarray(free)
    prolongation = FunctionLinearOperator(
        lambda reduced: (
            jnp.zeros((full_space.size,), dtype=reduced.dtype)
            .at[free_array]
            .set(reduced, unique_indices=True)
            .reshape(full_space.shape)
        ),
        source=reduced_space,
        target=full_space,
        transpose_action=lambda full: full.reshape((-1,))[free_array],
        operator_id=canonical_fingerprint(
            {
                "kind": "virtual-element-dirichlet-prolongation",
                "field_space": discretization.field_space.field_space_id,
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
                "kind": "virtual-element-dirichlet-constraint",
                "field_space": discretization.field_space.field_space_id,
                "constrained_dofs": constrained.tolist(),
            }
        ),
    )
    points = discretization.dof_map.default_dof_points[jnp.asarray(constrained)]
    return VirtualElementDirichletConstraint(
        field_name=discretization.field.name,
        constraint_map=constraint_map,
        constrained_dofs=jnp.asarray(constrained),
        free_dofs=free_array,
        dof_coordinates=points,
        constraint_id=constraint_map.constraint_id,
    )


__all__ = [
    "VirtualElementDirichletConstraint",
    "virtual_element_dirichlet_constraint",
]
