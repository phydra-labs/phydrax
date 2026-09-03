#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    compose_constraint_maps,
    ConstraintMap,
    DenseLinearOperator,
)
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ._generic import FiniteElementDiscretization
from ._hp_runtime import FiniteElementHPTraceConstraintPlan


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
        if callable(values):
            evaluator = cast(Callable[[Array], ArrayLike], values)
            evaluated = evaluator(self.dof_coordinates)
        else:
            evaluated = values
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


class FiniteElementLinearConstraint(StrictModule, NonTrainableState):
    """Homogeneous finite-element constraint without Dirichlet boundary data."""

    field_name: str = eqx.field(static=True)
    constraint_map: ConstraintMap
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        constraint_map: ConstraintMap,
        /,
    ):
        name = str(field_name)
        if not name or not isinstance(constraint_map, ConstraintMap):
            raise ValueError(
                "Linear finite-element constraints require a field and ConstraintMap."
            )
        self.field_name = name
        self.constraint_map = constraint_map
        self.constraint_id = constraint_map.constraint_id

    def lift(self, /):
        return self.constraint_map.full_space.zeros()


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
    full_space = field_space.vector_space
    reduced_space = ArraySpace((free.size,), dtype=full_space.dtype)
    free_array = jnp.asarray(free)
    full_size = int(np.prod(full_space.shape, dtype=int))
    relation = EdgeRelation(
        np.arange(free.size, dtype=np.int32),
        free,
        source_size=free.size,
        target_size=full_size,
    )
    prolongation = SparseCoordinateOperator(
        relation,
        jnp.ones((free.size,), dtype=full_space.dtype),
        source=reduced_space,
        target=full_space,
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
    prolongation: ArrayLike | AbstractLinearOperator,
    /,
    *,
    constraint_id: str | None = None,
) -> ConstraintMap:
    """Create periodic, multipoint, or hanging-node affine coordinates."""

    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    field_index = discretization._field_index(field_name)
    full_space = discretization.field_spaces[field_index].vector_space
    if isinstance(prolongation, AbstractLinearOperator):
        operator = prolongation
        if not operator.target.compatible(full_space):
            raise ValueError(
                "Constraint prolongation operator must target the full field space."
            )
        if operator.source.size > operator.target.size:
            raise ValueError("Constraint prolongation cannot be coordinate-injective.")
        reduced_space = operator.source
    else:
        matrix = jnp.asarray(prolongation)
        if matrix.ndim != 2 or matrix.shape[0] != full_space.size:
            raise ValueError(
                "prolongation must map reduced coordinates to the full field."
            )
        matrix_host = np.asarray(matrix)
        if (
            matrix.shape[1] == 0
            or np.any(~np.isfinite(matrix_host))
            or np.linalg.matrix_rank(matrix_host) != matrix.shape[1]
        ):
            raise ValueError(
                "Constraint prolongation must be finite and column-injective."
            )
        reduced_space = ArraySpace((int(matrix.shape[1]),), dtype=matrix.dtype)
        operator = DenseLinearOperator(
            matrix,
            source=reduced_space,
            target=full_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "finite-element-affine-dof-prolongation",
                    "field_space": discretization.field_spaces[
                        field_index
                    ].field_space_id,
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


def finite_element_hp_constraint(
    discretization: FiniteElementDiscretization,
    field_name: str,
    plan: FiniteElementHPTraceConstraintPlan,
    /,
) -> ConstraintMap:
    """Lift one scalar trace plan over field components and declare its pairing."""

    if not isinstance(plan, FiniteElementHPTraceConstraintPlan):
        raise TypeError("plan must be FiniteElementHPTraceConstraintPlan.")
    field_index = discretization._field_index(field_name)
    dof_map = discretization.dof_maps[field_index]
    full_space = discretization.field_spaces[field_index].vector_space
    if not isinstance(full_space, ArraySpace):
        raise TypeError("Adaptive hp trace constraints require ArraySpace fields.")
    if plan.full_dof_count != dof_map.global_dof_count:
        raise ValueError("hp trace plan and finite-element DOF map disagree.")
    component_count = int(np.prod(full_space.shape[1:], dtype=int))
    columns = np.asarray(plan.row_columns, dtype=np.int32)
    weights = np.asarray(plan.row_weights)
    valid = np.asarray(plan.row_valid, dtype=bool)
    components = np.arange(component_count, dtype=np.int32)
    source_indices = (columns[..., None] * component_count + components).reshape((-1,))
    target_indices = np.broadcast_to(
        np.arange(plan.full_dof_count, dtype=np.int32)[:, None, None] * component_count
        + components,
        columns.shape + (component_count,),
    ).reshape((-1,))
    route_valid = np.broadcast_to(
        valid[..., None], valid.shape + (component_count,)
    ).reshape((-1,))
    relation = EdgeRelation(
        source_indices,
        target_indices,
        source_size=plan.reduced_dof_count * component_count,
        target_size=plan.full_dof_count * component_count,
        valid=route_valid,
    )
    reduced_space = ArraySpace(
        (plan.reduced_dof_count * component_count,),
        dtype=full_space.dtype,
    )
    operator = SparseCoordinateOperator(
        relation,
        jnp.asarray(
            np.broadcast_to(
                weights[..., None], weights.shape + (component_count,)
            ).reshape((-1,)),
            dtype=full_space.dtype,
        ),
        source=reduced_space,
        target=full_space,
        operator_id=canonical_fingerprint(
            {
                "kind": "finite-element-hp-field-prolongation",
                "field_space": discretization.field_spaces[field_index].field_space_id,
                "trace_plan": plan.plan_id,
            }
        ),
    )
    return affine_dof_constraint(
        discretization,
        field_name,
        operator,
        constraint_id=canonical_fingerprint(
            {
                "kind": "finite-element-hp-field-constraint",
                "field_space": discretization.field_spaces[field_index].field_space_id,
                "trace_plan": plan.plan_id,
            }
        ),
    )


def compose_finite_element_constraints(
    outer: ConstraintMap,
    inner: ConstraintMap,
    /,
) -> ConstraintMap:
    """Compose hanging/master coordinates with a further reduced constraint."""

    if not isinstance(outer, ConstraintMap) or not isinstance(inner, ConstraintMap):
        raise TypeError("Constraint composition requires two ConstraintMap values.")
    return compose_constraint_maps(
        outer,
        inner,
        constraint_id=canonical_fingerprint(
            {
                "kind": "composed-finite-element-constraint",
                "outer": outer.constraint_id,
                "inner": inner.constraint_id,
            }
        ),
    )


__all__ = [
    "FiniteElementDirichletConstraint",
    "FiniteElementLinearConstraint",
    "compose_finite_element_constraints",
    "finite_element_hp_constraint",
    "affine_dof_constraint",
    "dirichlet_constraint",
]
