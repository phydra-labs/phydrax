#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

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
from .._constraints import AbstractDiscreteDirichletConstraint
from .._topology import EntitySelection
from ._boundary import FiniteElementBoundarySet
from ._generic import FiniteElementDiscretization
from ._hp_runtime import FiniteElementHPTraceConstraintPlan


class FiniteElementDirichletConstraint(AbstractDiscreteDirichletConstraint):
    """Strong essential constraint resolved onto one prepared FE field."""


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
    vertex_count = discretization.mesh.coordinates.shape[0]
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
    boundary_selection: EntitySelection | None = None,
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
    if boundary_mask is not None and boundary_selection is not None:
        raise ValueError("Specify boundary_mask or boundary_selection, not both.")
    if boundary_selection is not None:
        if not isinstance(boundary_selection, EntitySelection):
            raise TypeError("boundary_selection must be EntitySelection or None.")
        if (
            boundary_selection.entity_set_id
            != discretization.exterior_facet_domain.entity_set_id
        ):
            raise ValueError("Dirichlet boundary_selection must select mesh facets.")
        node_mask = np.asarray(
            discretization.dof_mask(field_name, boundary_selection),
            dtype=np.bool_,
        )
        selected_facets = np.flatnonzero(
            np.asarray(boundary_selection.mask, dtype=np.bool_)
        )
        exterior_facets = np.asarray(
            discretization.exterior_facet_domain.entity_indices,
            dtype=np.int32,
        )
        if np.any(~np.isin(selected_facets, exterior_facets)):
            raise ValueError(
                "Dirichlet boundary_selection may contain only exterior facets."
            )
    else:
        node_mask = (
            np.asarray(dof_map.boundary_dof_mask, dtype=np.bool_)
            if boundary_mask is None
            else np.asarray(boundary_mask, dtype=np.bool_)
        )
    full_shape = field_space.vector_space.shape
    component_count = (
        int(np.prod(full_shape[1:], dtype=np.int64)) if full_shape[1:] else 1
    )
    if node_mask.shape == full_shape:
        full_mask = node_mask.reshape((dof_map.global_dof_count, component_count))
        node_mask = np.any(full_mask, axis=1)
    elif node_mask.shape == (dof_map.global_dof_count,):
        selected_components = (
            np.arange(component_count, dtype=np.int32)
            if components is None
            else np.asarray(tuple(components), dtype=np.int32)
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
            dtype=np.bool_,
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
    full_size = int(np.prod(full_space.shape, dtype=np.int64))
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
        reduced_space = ArraySpace((matrix.shape[1],), dtype=matrix.dtype)
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


def periodic_constraint(
    discretization: FiniteElementDiscretization,
    field_name: str,
    boundary: FiniteElementBoundarySet,
    /,
) -> FiniteElementLinearConstraint:
    """Identify scalar or componentwise periodic H1 coordinates."""

    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    if not isinstance(boundary, FiniteElementBoundarySet):
        raise TypeError("boundary must be FiniteElementBoundarySet.")
    if boundary.support_id != discretization.support.support_id:
        raise ValueError("Periodic boundary set belongs to a different support.")
    if not boundary.periodic_pairs:
        raise ValueError("Periodic constraints require at least one facet pair.")
    field_index = discretization._field_index(field_name)
    field_space = discretization.field_spaces[field_index]
    dof_map = discretization.dof_maps[field_index]
    if dof_map.association not in ("vertex", "entity"):
        raise ValueError("Periodic constraints require conforming H1 coordinates.")
    full_space = field_space.vector_space
    if not isinstance(full_space, ArraySpace):
        raise TypeError("Periodic constraints require ArraySpace coordinates.")
    node_count = dof_map.global_dof_count
    component_count = (
        int(np.prod(full_space.shape[1:], dtype=np.int64))
        if len(full_space.shape) > 1
        else 1
    )
    parents = np.arange(node_count, dtype=np.int32)

    def root(value: int) -> int:
        current = int(value)
        while parents[current] != current:
            parents[current] = parents[parents[current]]
            current = int(parents[current])
        return current

    def union(first: int, second: int) -> None:
        left = root(first)
        right = root(second)
        if left == right:
            return
        master, slave = sorted((left, right))
        parents[slave] = master

    facet_entities = discretization.mesh.topology.entity_sets[
        discretization.mesh.topological_dimension - 1
    ]
    coordinates = np.asarray(dof_map.dof_coordinates)
    for pair in boundary.periodic_pairs:
        masks = []
        for facet in (pair.owner_facet, pair.neighbor_facet):
            mask = np.zeros((facet_entities.count,), dtype=np.bool_)
            mask[facet] = True
            selection = EntitySelection(facet_entities, mask)
            masks.append(
                np.asarray(discretization.dof_mask(field_name, selection), dtype=np.bool_)
            )
        owner = np.flatnonzero(masks[0])
        neighbor = np.flatnonzero(masks[1])
        if owner.size == 0 or owner.size != neighbor.size:
            raise ValueError("Periodic facets expose incompatible field coordinates.")
        if pair.transform is None:
            mapped = coordinates[owner]
            tolerance = 1.0e-10
        else:
            transform = pair.transform
            components = np.asarray(transform.component_matrix)
            if components.shape not in ((1, 1), (component_count, component_count)):
                raise ValueError("Periodic component transform does not match the field.")
            if not np.allclose(
                components,
                np.eye(components.shape[0]),
                rtol=0.0,
                atol=transform.tolerance,
            ):
                raise ValueError(
                    "Phase-field periodic constraints currently require an identity component transform."
                )
            mapped = coordinates[owner] @ np.asarray(
                transform.coordinate_matrix
            ).T + np.asarray(transform.coordinate_offset)
            tolerance = transform.tolerance
        neighbor_coordinates = coordinates[neighbor]
        distances = np.sqrt(
            np.sum(
                (mapped[:, None, :] - neighbor_coordinates[None, :, :]) ** 2,
                axis=-1,
            )
        )
        matched = np.argmin(distances, axis=1)
        if np.unique(matched).size != owner.size or np.any(
            distances[np.arange(owner.size), matched] > tolerance
        ):
            raise ValueError("Periodic facet coordinates do not match bijectively.")
        for left, right in zip(owner, neighbor[matched], strict=True):
            union(int(left), int(right))

    roots = np.asarray([root(index) for index in range(node_count)], dtype=np.int32)
    masters = np.unique(roots)
    slot_by_master = {int(master): slot for slot, master in enumerate(masters)}
    source = []
    target = []
    for node, master in enumerate(roots):
        for component in range(component_count):
            source.append(slot_by_master[int(master)] * component_count + component)
            target.append(node * component_count + component)
    reduced_space = ArraySpace(
        (masters.size * component_count,),
        dtype=full_space.dtype,
    )
    relation = EdgeRelation(
        np.asarray(source, dtype=np.int32),
        np.asarray(target, dtype=np.int32),
        source_size=reduced_space.size,
        target_size=full_space.size,
    )
    operator = SparseCoordinateOperator(
        relation,
        jnp.ones((len(source),), dtype=full_space.dtype),
        source=reduced_space,
        target=full_space,
        operator_id=canonical_fingerprint(
            {
                "kind": "finite-element-periodic-prolongation",
                "field_space": field_space.field_space_id,
                "boundary": boundary.boundary_set_id,
                "masters": masters.tolist(),
            }
        ),
    )
    constraint = ConstraintMap(
        full_space,
        reduced_space,
        operator,
        constraint_id=canonical_fingerprint(
            {
                "kind": "finite-element-periodic-constraint",
                "field_space": field_space.field_space_id,
                "boundary": boundary.boundary_set_id,
                "roots": roots.tolist(),
            }
        ),
    )
    return FiniteElementLinearConstraint(field_name, constraint)


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
    component_count = int(np.prod(full_space.shape[1:], dtype=np.int64))
    columns = np.asarray(plan.row_columns, dtype=np.int32)
    weights = np.asarray(plan.row_weights)
    valid = np.asarray(plan.row_valid, dtype=np.bool_)
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
    "periodic_constraint",
    "dirichlet_constraint",
]
