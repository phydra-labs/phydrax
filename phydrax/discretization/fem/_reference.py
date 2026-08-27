#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class FiniteElementSpec(StrictModule, NonTrainableState):
    """Immutable scalar reference finite element with explicit entity DOFs."""

    family: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    mapping: str = eqx.field(static=True)
    reference_nodes: Array
    entity_dofs: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: str,
        cell_kind: str,
        degree: int,
        reference_nodes: ArrayLike,
        entity_dofs: tuple[tuple[tuple[int, ...], ...], ...],
        /,
        *,
        conformity: str = "H1",
        mapping: str = "identity",
    ):
        family_ = str(family)
        cell = str(cell_kind)
        order = int(degree)
        conformity_ = str(conformity)
        mapping_ = str(mapping)
        if not family_ or not conformity_ or not mapping_:
            raise ValueError("Finite-element identifiers must be non-empty.")
        if cell not in ("triangle", "quadrilateral", "tetrahedron"):
            raise ValueError("Unsupported reference cell kind.")
        if order <= 0:
            raise ValueError("Finite-element degree must be positive.")
        nodes = np.asarray(reference_nodes, dtype=float)
        dimension = {"triangle": 2, "quadrilateral": 2, "tetrahedron": 3}[cell]
        if nodes.ndim != 2 or nodes.shape[1] != dimension or nodes.shape[0] == 0:
            raise ValueError(
                "Reference nodes must have shape (local_dof_count, cell_dimension)."
            )
        if not np.all(np.isfinite(nodes)):
            raise ValueError("Reference nodes must be finite.")
        normalized_entity_dofs = tuple(
            tuple(tuple(int(dof) for dof in entity) for entity in dimension_entities)
            for dimension_entities in entity_dofs
        )
        if len(normalized_entity_dofs) != dimension + 1:
            raise ValueError("entity_dofs must contain every entity dimension.")
        flattened = tuple(
            dof
            for dimension_entities in normalized_entity_dofs
            for entity in dimension_entities
            for dof in entity
        )
        if tuple(sorted(flattened)) != tuple(range(nodes.shape[0])):
            raise ValueError(
                "Each local DOF must belong to exactly one reference entity."
            )
        self.family = family_
        self.cell_kind = cell
        self.degree = order
        self.conformity = conformity_
        self.mapping = mapping_
        self.reference_nodes = jnp.asarray(nodes)
        self.entity_dofs = normalized_entity_dofs
        self.element_id = canonical_fingerprint(
            {
                "kind": "finite-element-spec",
                "family": family_,
                "cell_kind": cell,
                "degree": order,
                "conformity": conformity_,
                "mapping": mapping_,
                "reference_nodes": array_tree_fingerprint(nodes),
                "entity_dofs": normalized_entity_dofs,
            }
        )

    @property
    def topological_dimension(self) -> int:
        return int(self.reference_nodes.shape[1])

    @property
    def local_dof_count(self) -> int:
        return int(self.reference_nodes.shape[0])

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        """Return basis values and reference gradients at reference points."""

        locations = jnp.asarray(points)
        if locations.ndim != 2 or locations.shape[1] != self.topological_dimension:
            raise ValueError(
                "Reference evaluation points must have shape (point_count, cell_dimension)."
            )
        if self.cell_kind == "triangle" and self.degree == 1:
            return _triangle_p1(locations)
        if self.cell_kind == "triangle" and self.degree == 2:
            return _triangle_p2(locations)
        if self.cell_kind == "quadrilateral" and self.degree == 1:
            return _quadrilateral_q1(locations)
        if self.cell_kind == "tetrahedron" and self.degree == 1:
            return _tetrahedron_p1(locations)
        raise ValueError("Finite-element tabulation is not implemented for this spec.")


def _triangle_p1(points: Array, /) -> tuple[Array, Array]:
    x = points[:, 0]
    y = points[:, 1]
    values = jnp.stack((1.0 - x - y, x, y), axis=-1)
    gradients = jnp.broadcast_to(
        jnp.asarray(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0))),
        (points.shape[0], 3, 2),
    )
    return values, gradients


def _triangle_p2(points: Array, /) -> tuple[Array, Array]:
    lambda_0 = 1.0 - points[:, 0] - points[:, 1]
    lambda_1 = points[:, 0]
    lambda_2 = points[:, 1]
    barycentric = jnp.stack((lambda_0, lambda_1, lambda_2), axis=-1)
    barycentric_gradients = jnp.asarray(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)))
    vertex_values = barycentric * (2.0 * barycentric - 1.0)
    vertex_gradients = (4.0 * barycentric - 1.0)[..., None] * barycentric_gradients[
        None, ...
    ]
    edge_pairs = ((0, 1), (1, 2), (2, 0))
    edge_values = jnp.stack(
        tuple(
            4.0 * barycentric[:, first] * barycentric[:, second]
            for first, second in edge_pairs
        ),
        axis=-1,
    )
    edge_gradients = jnp.stack(
        tuple(
            4.0
            * (
                barycentric[:, first, None] * barycentric_gradients[second]
                + barycentric[:, second, None] * barycentric_gradients[first]
            )
            for first, second in edge_pairs
        ),
        axis=1,
    )
    return (
        jnp.concatenate((vertex_values, edge_values), axis=-1),
        jnp.concatenate((vertex_gradients, edge_gradients), axis=1),
    )


def _quadrilateral_q1(points: Array, /) -> tuple[Array, Array]:
    xi = points[:, 0]
    eta = points[:, 1]
    values = jnp.stack(
        (
            (1.0 - xi) * (1.0 - eta),
            xi * (1.0 - eta),
            xi * eta,
            (1.0 - xi) * eta,
        ),
        axis=-1,
    )
    gradients = jnp.stack(
        (
            jnp.stack((-(1.0 - eta), -(1.0 - xi)), axis=-1),
            jnp.stack((1.0 - eta, -xi), axis=-1),
            jnp.stack((eta, xi), axis=-1),
            jnp.stack((-eta, 1.0 - xi), axis=-1),
        ),
        axis=1,
    )
    return values, gradients


def _tetrahedron_p1(points: Array, /) -> tuple[Array, Array]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    values = jnp.stack((1.0 - x - y - z, x, y, z), axis=-1)
    gradients = jnp.broadcast_to(
        jnp.asarray(
            ((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ),
        (points.shape[0], 4, 3),
    )
    return values, gradients


def lagrange_element(cell_kind: str, degree: int, /) -> FiniteElementSpec:
    """Construct one implemented scalar nodal Lagrange reference element."""

    cell = str(cell_kind)
    order = int(degree)
    if cell == "triangle" and order == 1:
        return FiniteElementSpec(
            "Lagrange",
            cell,
            order,
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            (((0,), (1,), (2,)), ((), (), ()), ((),)),
        )
    if cell == "triangle" and order == 2:
        return FiniteElementSpec(
            "Lagrange",
            cell,
            order,
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.5, 0.0),
                (0.5, 0.5),
                (0.0, 0.5),
            ),
            (((0,), (1,), (2,)), ((3,), (4,), (5,)), ((),)),
        )
    if cell == "quadrilateral" and order == 1:
        return FiniteElementSpec(
            "Lagrange",
            cell,
            order,
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            (((0,), (1,), (2,), (3,)), ((), (), (), ()), ((),)),
        )
    if cell == "tetrahedron" and order == 1:
        return FiniteElementSpec(
            "Lagrange",
            cell,
            order,
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (((0,), (1,), (2,), (3,)), ((),) * 6, ((),) * 4, ((),)),
        )
    raise ValueError(
        "Implemented Lagrange elements are triangle P1/P2, quadrilateral Q1, "
        "and tetrahedron P1."
    )


__all__ = ["FiniteElementSpec", "lagrange_element"]
