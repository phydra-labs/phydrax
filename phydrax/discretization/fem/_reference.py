#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

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
    value_shape: tuple[int, ...] = eqx.field(static=True)
    reference_nodes: Array
    entity_dofs: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    tabulator: Callable | None
    tabulator_id: str | None = eqx.field(static=True)
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
        value_shape: tuple[int, ...] = (),
        tabulator: Callable | None = None,
        tabulator_id: str | None = None,
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
        if order < 0:
            raise ValueError("Finite-element degree must be non-negative.")
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
        values = tuple(int(size) for size in value_shape)
        if any(size <= 0 for size in values):
            raise ValueError("Finite-element value dimensions must be positive.")
        if tabulator is not None and not callable(tabulator):
            raise TypeError("tabulator must be callable or None.")
        resolved_tabulator_id = None if tabulator_id is None else str(tabulator_id)
        if tabulator is not None and not resolved_tabulator_id:
            raise ValueError("Custom tabulators require a non-empty tabulator_id.")
        self.family = family_
        self.cell_kind = cell
        self.degree = order
        self.conformity = conformity_
        self.mapping = mapping_
        self.value_shape = values
        self.reference_nodes = jnp.asarray(nodes)
        self.entity_dofs = normalized_entity_dofs
        self.tabulator = tabulator
        self.tabulator_id = resolved_tabulator_id
        self.element_id = canonical_fingerprint(
            {
                "kind": "finite-element-spec",
                "family": family_,
                "cell_kind": cell,
                "degree": order,
                "conformity": conformity_,
                "mapping": mapping_,
                "value_shape": list(values),
                "reference_nodes": array_tree_fingerprint(nodes),
                "entity_dofs": normalized_entity_dofs,
                "tabulator_id": resolved_tabulator_id,
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
        if self.tabulator is not None:
            values, gradients = self.tabulator(locations)
            values_ = jnp.asarray(values)
            gradients_ = jnp.asarray(gradients)
            if (
                values_.shape[:2]
                != (
                    locations.shape[0],
                    self.local_dof_count,
                )
                or gradients_.shape[:2] != values_.shape[:2]
            ):
                raise ValueError("Custom tabulator returned incompatible leading axes.")
            return values_, gradients_
        if self.family == "DiscontinuousLagrange" and self.degree == 0:
            return (
                jnp.ones((locations.shape[0], 1)),
                jnp.zeros((locations.shape[0], 1, self.topological_dimension)),
            )
        if self.family == "RaviartThomas" and self.cell_kind == "triangle":
            return _triangle_rt0(locations)
        if self.family == "Nedelec" and self.cell_kind == "triangle":
            return _triangle_nedelec0(locations)
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


def _triangle_rt0(points: Array, /) -> tuple[Array, Array]:
    x = points[:, 0]
    y = points[:, 1]
    values = jnp.stack(
        (
            jnp.stack((x, y - 1.0), axis=-1),
            jnp.stack((x, y), axis=-1),
            jnp.stack((x - 1.0, y), axis=-1),
        ),
        axis=1,
    )
    identity = jnp.eye(2)
    gradients = jnp.broadcast_to(
        identity,
        (points.shape[0], 3, 2, 2),
    )
    return values, gradients


def _triangle_nedelec0(points: Array, /) -> tuple[Array, Array]:
    lambda_0 = 1.0 - points[:, 0] - points[:, 1]
    lambda_1 = points[:, 0]
    lambda_2 = points[:, 1]
    barycentric = (lambda_0, lambda_1, lambda_2)
    gradients = jnp.asarray(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)))
    pairs = ((0, 1), (1, 2), (2, 0))
    values = jnp.stack(
        tuple(
            barycentric[first][:, None] * gradients[second]
            - barycentric[second][:, None] * gradients[first]
            for first, second in pairs
        ),
        axis=1,
    )
    derivative = jnp.stack(
        tuple(
            gradients[second][:, None] * gradients[first][None, :]
            - gradients[first][:, None] * gradients[second][None, :]
            for first, second in pairs
        ),
        axis=0,
    )
    return values, jnp.broadcast_to(
        derivative,
        (points.shape[0],) + derivative.shape,
    )


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


def discontinuous_element(cell_kind: str, degree: int = 0, /) -> FiniteElementSpec:
    cell = str(cell_kind)
    order = int(degree)
    if order != 0:
        raise ValueError("Only discontinuous P0 is currently implemented.")
    dimension = {"triangle": 2, "quadrilateral": 2, "tetrahedron": 3}.get(cell)
    if dimension is None:
        raise ValueError("Unsupported discontinuous reference cell.")
    center = {
        "triangle": ((1.0 / 3.0, 1.0 / 3.0),),
        "quadrilateral": ((0.5, 0.5),),
        "tetrahedron": ((0.25, 0.25, 0.25),),
    }[cell]
    entities = {
        "triangle": (((), (), ()), ((), (), ()), ((0,),)),
        "quadrilateral": (((), (), (), ()), ((), (), (), ()), ((0,),)),
        "tetrahedron": (
            ((), (), (), ()),
            ((),) * 6,
            ((),) * 4,
            ((0,),),
        ),
    }[cell]
    return FiniteElementSpec(
        "DiscontinuousLagrange",
        cell,
        0,
        center,
        entities,
        conformity="L2",
    )


def raviart_thomas_element(cell_kind: str, degree: int = 0, /) -> FiniteElementSpec:
    if str(cell_kind) != "triangle" or int(degree) != 0:
        raise ValueError("Only triangular Raviart-Thomas RT0 is implemented.")
    return FiniteElementSpec(
        "RaviartThomas",
        "triangle",
        0,
        ((0.5, 0.0), (0.5, 0.5), (0.0, 0.5)),
        (((), (), ()), ((0,), (1,), (2,)), ((),)),
        conformity="Hdiv",
        mapping="contravariant_piola",
        value_shape=(2,),
    )


def nedelec_element(cell_kind: str, degree: int = 0, /) -> FiniteElementSpec:
    if str(cell_kind) != "triangle" or int(degree) != 0:
        raise ValueError("Only triangular first-kind Nedelec order zero is implemented.")
    return FiniteElementSpec(
        "Nedelec",
        "triangle",
        0,
        ((0.5, 0.0), (0.5, 0.5), (0.0, 0.5)),
        (((), (), ()), ((0,), (1,), (2,)), ((),)),
        conformity="Hcurl",
        mapping="covariant_piola",
        value_shape=(2,),
    )


__all__ = [
    "FiniteElementSpec",
    "discontinuous_element",
    "lagrange_element",
    "nedelec_element",
    "raviart_thomas_element",
]
