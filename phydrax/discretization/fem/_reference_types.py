#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...exterior._form_type import FormValueSpec
from ...typing import checked, parse
from .._reference_cell import reference_cell_topology
from .._spaces import FieldRepresentation
from ._form_basis_types import AbstractReferenceFormBasis


type ElementContinuity = Literal["conforming", "discontinuous"]
type ElementConformity = Literal["H1", "Hcurl", "Hdiv", "L2", "HLambda"]
type ElementMapping = Literal[
    "identity", "covariant_piola", "contravariant_piola", "density", "exterior"
]
type _Tabulator = Callable[[Array], tuple[ArrayLike, ArrayLike]]


class AbstractFiniteElementTabulationSource(StrictModule, NonTrainableState, ABC):
    """Evaluate reference basis values and gradients from immutable source data."""

    @abstractmethod
    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        raise NotImplementedError


class AbstractStoredFiniteElementTabulator(StrictModule, NonTrainableState, ABC):
    """Reconstruct a reference element whose preparation is source-owned."""

    @abstractmethod
    def finite_element_from_stored_fields(self, /) -> FiniteElementSpec:
        raise NotImplementedError


@final
class FiniteElementTabulator(StrictModule, NonTrainableState):
    """Retain a basis source as dynamic numerical leaves for portable evaluation."""

    source: AbstractFiniteElementTabulationSource
    method: str = eqx.field(static=True)

    @checked
    def __init__(
        self, source: AbstractFiniteElementTabulationSource, method: str = "tabulate", /
    ) -> None:
        if method != "tabulate":
            raise TypeError(
                f"FiniteElementTabulator.method must be 'tabulate'; got {method!r}"
            )
        self.source = source
        self.method = method

    def __call__(self, points: ArrayLike, /) -> tuple[Array, Array]:
        return self.source.tabulate(points)


@final
class FiniteElementSpec(AbstractFiniteElementTabulationSource):
    """Reference element with typed form values and explicit entity functionals."""

    family: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    value_spec: FormValueSpec = eqx.field(static=True)
    continuity: ElementContinuity = eqx.field(static=True)
    representation: FieldRepresentation = eqx.field(static=True)
    reference_nodes: Array
    entity_dofs: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    tabulator: _Tabulator | None
    tabulator_id: str | None = eqx.field(static=True)
    form_basis: AbstractReferenceFormBasis | None
    element_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        family: str,
        cell_kind: str,
        degree: int,
        reference_nodes: npt.ArrayLike,
        entity_dofs: tuple[tuple[tuple[int, ...], ...], ...],
        /,
        *,
        value_spec: FormValueSpec,
        continuity: ElementContinuity = "conforming",
        representation: FieldRepresentation = "point_value",
        tabulator: _Tabulator | None = None,
        tabulator_id: str | None = None,
        form_basis: AbstractReferenceFormBasis | None = None,
    ) -> None:
        dimension = reference_cell_topology(cell_kind).dimension
        if dimension != value_spec.form_type.dimension:
            raise ValueError(
                "FiniteElementSpec.value_spec.dimension must match the reference cell; got different dimensions"
            )
        if not family or degree < 0:
            raise ValueError(
                "FiniteElementSpec.family_degree must name a family and have nonnegative degree; got an empty family or negative degree"
            )
        continuity_ = parse(continuity, ElementContinuity, "continuity")
        representation_ = parse(representation, FieldRepresentation, "representation")
        nodes = np.asarray(reference_nodes, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != dimension or nodes.shape[0] == 0:
            raise ValueError(
                "FiniteElementSpec.reference_nodes must have shape (local_dof_count, dimension); got incompatible shape"
            )
        if not np.all(np.isfinite(nodes)):
            raise ValueError(
                "FiniteElementSpec.reference_nodes must be finite; got nonfinite nodes"
            )
        entities = tuple(
            tuple(tuple(int(dof) for dof in entity) for entity in level)
            for level in entity_dofs
        )
        if len(entities) != dimension + 1:
            raise ValueError(
                "FiniteElementSpec.entity_dofs must contain every entity dimension; got incomplete dimension levels"
            )
        flattened = tuple(dof for level in entities for entity in level for dof in entity)
        if tuple(sorted(flattened)) != tuple(range(nodes.shape[0])):
            raise ValueError(
                "FiniteElementSpec.entity_dofs must assign each local DOF to one entity; got missing or repeated DOFs"
            )
        if tabulator is not None and not tabulator_id:
            raise ValueError(
                "FiniteElementSpec.tabulator_id must be nonempty for a custom tabulator; got an empty identifier"
            )
        self.family = family
        self.cell_kind = cell_kind
        self.degree = degree
        self.value_spec = value_spec
        self.continuity = continuity_
        self.representation = representation_
        self.reference_nodes = jax.device_put(nodes)
        self.entity_dofs = entities
        self.tabulator = tabulator
        self.tabulator_id = tabulator_id
        self.form_basis = form_basis
        self.element_id = canonical_fingerprint(
            {
                "kind": "finite-element-spec",
                "family": family,
                "cell_kind": cell_kind,
                "degree": degree,
                "value_spec": value_spec.value_spec_id,
                "continuity": continuity_,
                "representation": representation_,
                "reference_nodes": array_tree_fingerprint(nodes),
                "entity_dofs": entities,
                "tabulator_id": tabulator_id,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(cls, value: FiniteElementSpec, /) -> FiniteElementSpec:
        """Reconstruct a reference element from its retained construction inputs."""
        match value.tabulator:
            case AbstractStoredFiniteElementTabulator():
                return value.tabulator.finite_element_from_stored_fields()
            case _:
                return cls(
                    value.family,
                    value.cell_kind,
                    value.degree,
                    value.reference_nodes,
                    value.entity_dofs,
                    value_spec=value.value_spec,
                    continuity=value.continuity,
                    representation=value.representation,
                    tabulator=value.tabulator,
                    tabulator_id=value.tabulator_id,
                    form_basis=value.form_basis,
                )

    @property
    def value_shape(self) -> tuple[int, ...]:
        return self.value_spec.value_shape

    @property
    def mapping(self) -> ElementMapping:
        match self.value_spec.pullback_rule:
            case "identity":
                return "identity"
            case "covariant":
                return "covariant_piola"
            case "contravariant":
                return "contravariant_piola"
            case "density":
                return "density"
            case "exterior":
                return "exterior"

    @property
    def conformity(self) -> ElementConformity:
        if self.continuity == "discontinuous":
            return "L2"
        form = self.value_spec.form_type
        if form.degree == 0:
            return "H1"
        if form.degree == form.dimension:
            return "L2"
        match self.value_spec.proxy:
            case "circulation":
                return "Hcurl"
            case "flux":
                return "Hdiv"
            case _:
                return "HLambda"

    @property
    def topological_dimension(self) -> int:
        return self.reference_nodes.shape[1]

    @property
    def local_dof_count(self) -> int:
        return self.reference_nodes.shape[0]

    @property
    def entity_vertices(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Vertex ordering paired with each level of ``entity_dofs``."""
        if self.form_basis is not None:
            return self.form_basis.entity_vertices
        return reference_cell_topology(self.cell_kind).entities

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        locations = jnp.asarray(points)
        if locations.ndim != 2 or locations.shape[1] != self.topological_dimension:
            raise ValueError(
                "FiniteElementSpec.points must have shape (point_count, dimension); got incompatible shape"
            )
        if self.tabulator is not None:
            values, gradients = self.tabulator(locations)
            values_, gradients_ = jnp.asarray(values), jnp.asarray(gradients)
            if values_.shape[:2] != (locations.shape[0], self.local_dof_count):
                raise ValueError(
                    "FiniteElementSpec.tabulator.values must have leading axes (point_count, local_dof_count); got incompatible axes"
                )
            if gradients_.shape[:2] != values_.shape[:2]:
                raise ValueError(
                    "FiniteElementSpec.tabulator.gradients must match the value leading axes; got incompatible axes"
                )
            return values_, gradients_
        if self.family == "DiscontinuousLagrange" and self.degree == 0:
            return jnp.ones((locations.shape[0], 1)), jnp.zeros(
                (locations.shape[0], 1, self.topological_dimension)
            )
        if self.cell_kind == "triangle" and self.degree == 1:
            return _triangle_p1(locations)
        if self.cell_kind == "triangle" and self.degree == 2:
            return _triangle_p2(locations)
        if self.cell_kind == "quadrilateral" and self.degree == 1:
            return _quadrilateral_q1(locations)
        if self.cell_kind == "tetrahedron" and self.degree == 1:
            return _tetrahedron_p1(locations)
        if self.cell_kind == "hexahedron" and self.degree == 1:
            return _hexahedron_q1(locations)
        raise ValueError(
            "FiniteElementSpec.tabulation must use an implemented reference element; got an unsupported specification"
        )


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


def _hexahedron_q1(points: Array, /) -> tuple[Array, Array]:
    xi = points[:, 0]
    eta = points[:, 1]
    zeta = points[:, 2]
    one_x = 1.0 - xi
    one_y = 1.0 - eta
    one_z = 1.0 - zeta
    values = jnp.stack(
        (
            one_x * one_y * one_z,
            xi * one_y * one_z,
            xi * eta * one_z,
            one_x * eta * one_z,
            one_x * one_y * zeta,
            xi * one_y * zeta,
            xi * eta * zeta,
            one_x * eta * zeta,
        ),
        axis=-1,
    )
    gradients = jnp.stack(
        (
            jnp.stack((-one_y * one_z, -one_x * one_z, -one_x * one_y), axis=-1),
            jnp.stack((one_y * one_z, -xi * one_z, -xi * one_y), axis=-1),
            jnp.stack((eta * one_z, xi * one_z, -xi * eta), axis=-1),
            jnp.stack((-eta * one_z, one_x * one_z, -one_x * eta), axis=-1),
            jnp.stack((-one_y * zeta, -one_x * zeta, one_x * one_y), axis=-1),
            jnp.stack((one_y * zeta, -xi * zeta, xi * one_y), axis=-1),
            jnp.stack((eta * zeta, xi * zeta, xi * eta), axis=-1),
            jnp.stack((-eta * zeta, one_x * zeta, one_x * eta), axis=-1),
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
