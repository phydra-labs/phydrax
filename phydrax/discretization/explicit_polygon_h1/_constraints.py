#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...linalg import ArraySpace, ConstraintMap, FunctionLinearOperator
from .._constraints import AbstractDiscreteDirichletConstraint
from .._integration_domain import IntegrationDomain
from ._space import ExplicitPolygonH1Discretization


class ExplicitPolygonH1DirichletConstraint(AbstractDiscreteDirichletConstraint):
    """Strong essential constraint on explicit polygon vertex values."""


def _component_roots(discretization: ExplicitPolygonH1Discretization, /) -> np.ndarray:
    vertex_count = discretization.dof_map.global_dof_count
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


def explicit_polygon_h1_dirichlet_constraint(
    discretization: ExplicitPolygonH1Discretization,
    field_name: str,
    /,
    *,
    boundary_mask: ArrayLike | None = None,
    domain: IntegrationDomain | None = None,
    components: Sequence[int] | None = None,
) -> ExplicitPolygonH1DirichletConstraint:
    """Resolve a component-aware strong constraint on exact polygon traces."""
    if not isinstance(discretization, ExplicitPolygonH1Discretization):
        raise TypeError("discretization must be ExplicitPolygonH1Discretization.")
    discretization._field_index(field_name)
    if boundary_mask is not None and domain is not None:
        raise ValueError("Specify boundary_mask or domain, not both.")
    node_mask = np.asarray(discretization.dof_map.boundary_dof_mask, dtype=bool)
    if domain is not None:
        if domain.kind != "exterior_facet":
            raise ValueError(
                "Explicit polygon Dirichlet domains must select exterior facets."
            )
        if (
            domain.support_id != discretization.support.support_id
            or domain.entity_set_id
            != discretization.mesh.topology.entity_sets[1].entity_set_id
        ):
            raise ValueError(
                "Explicit polygon Dirichlet domain belongs to another support."
            )
        node_mask = np.zeros_like(node_mask)
        edges = np.asarray(discretization.mesh.connectivity.edges, dtype=np.int32)
        node_mask[np.unique(edges[np.asarray(domain.entity_indices)].reshape((-1,)))] = (
            True
        )
    elif boundary_mask is not None:
        candidate = np.asarray(boundary_mask, dtype=bool)
        if candidate.shape == discretization.field_space.vector_space.shape:
            node_mask = np.any(
                candidate.reshape((discretization.dof_map.global_dof_count, -1)),
                axis=1,
            )
        elif candidate.shape == (discretization.dof_map.global_dof_count,):
            node_mask = candidate
        else:
            raise ValueError(
                "boundary_mask must have vertex-DOF shape or full field shape."
            )
        boundary = np.asarray(discretization.dof_map.boundary_dof_mask, dtype=bool)
        if np.any(node_mask & ~boundary):
            raise ValueError("Dirichlet masks may select only exterior vertices.")
    full_space = discretization.field_space.vector_space
    if not isinstance(full_space, ArraySpace):
        raise TypeError("Explicit polygon Dirichlet constraints require ArraySpace.")
    component_count = (
        int(np.prod(full_space.shape[1:], dtype=int)) if full_space.shape[1:] else 1
    )
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
    if boundary_mask is not None:
        candidate = np.asarray(boundary_mask, dtype=bool)
        if candidate.shape == full_space.shape:
            full_mask = candidate.reshape(
                (discretization.dof_map.global_dof_count, component_count)
            )
        else:
            full_mask = np.zeros(
                (discretization.dof_map.global_dof_count, component_count), dtype=bool
            )
            full_mask[:, selected_components] = node_mask[:, None]
    else:
        full_mask = np.zeros(
            (discretization.dof_map.global_dof_count, component_count), dtype=bool
        )
        full_mask[:, selected_components] = node_mask[:, None]
    roots = _component_roots(discretization)
    component_roots = {int(value) for value in roots}
    selected_vertices = np.flatnonzero(np.any(full_mask, axis=1))
    if {int(roots[index]) for index in selected_vertices} != component_roots:
        raise ValueError(
            "Explicit polygon Dirichlet constraints must anchor every mesh component."
        )
    flattened = full_mask.reshape((-1,))
    constrained = np.flatnonzero(flattened).astype(np.int32)
    free = np.flatnonzero(~flattened).astype(np.int32)
    if constrained.size == 0 or free.size == 0:
        raise ValueError(
            "Explicit polygon Dirichlet constraints require a nonempty proper subset."
        )
    reduced_space = ArraySpace((free.size,), pairing=None)
    full_size = full_space.size
    free_array = jnp.asarray(free)
    prolongation = FunctionLinearOperator(
        lambda reduced: (
            jnp.zeros((full_size,), dtype=reduced.dtype)
            .at[free_array]
            .set(reduced, unique_indices=True)
            .reshape(full_space.shape)
        ),
        source=reduced_space,
        target=full_space,
        transpose_action=lambda full: full.reshape((-1,))[free_array],
        operator_id=canonical_fingerprint(
            {
                "kind": "explicit-polygon-h1-dirichlet-prolongation",
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
                "kind": "explicit-polygon-h1-dirichlet-constraint",
                "field_space": discretization.field_space.field_space_id,
                "constrained_dofs": constrained.tolist(),
            }
        ),
    )
    return ExplicitPolygonH1DirichletConstraint(
        field_name=discretization.field.name,
        constraint_map=constraint_map,
        constrained_dofs=jnp.asarray(constrained),
        free_dofs=free_array,
        dof_coordinates=discretization.dof_map.default_dof_points,
        constraint_id=constraint_map.constraint_id,
    )


__all__ = [
    "ExplicitPolygonH1DirichletConstraint",
    "explicit_polygon_h1_dirichlet_constraint",
]
