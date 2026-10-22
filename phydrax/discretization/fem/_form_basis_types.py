from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...exterior._form_type import FormTwist
from .._coordinate_budget import CoordinateEnclosureBudget


if TYPE_CHECKING:
    from .._coordinate_enclosure import Expression


type FormElementFamily = Literal[
    "trimmed", "full", "tensor-trimmed", "prism-trimmed", "pyramid-trimmed"
]
type DofLabel = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


class AbstractReferenceTensorFactors(StrictModule, ABC):
    """Provide interval dual factors and their product/component routes."""

    order: eqx.AbstractVar[int]
    zero: eqx.AbstractVar[Array]
    one: eqx.AbstractVar[Array]
    zero_derivatives: eqx.AbstractVar[Array]
    one_derivatives: eqx.AbstractVar[Array]
    indices: eqx.AbstractVar[Array]
    differential_axes: eqx.AbstractVar[Array]
    components: eqx.AbstractVar[Array]

    @abstractmethod
    def tabulate(self, points: Array, /) -> tuple[Array, Array]:
        raise NotImplementedError


class AbstractReferenceHybridFactors(StrictModule, ABC):
    """Provide rational generator sources and numerical moment-dual evidence."""

    source_bank: eqx.AbstractVar[tuple[tuple[Fraction, ...], ...]]
    generators: eqx.AbstractVar[Array]
    dual_rank: eqx.AbstractVar[Array]
    dual_condition: eqx.AbstractVar[Array]
    dual_solve_error: eqx.AbstractVar[Array]
    dual_status: eqx.AbstractVar[Array]
    body_test_exponents: eqx.AbstractVar[tuple[tuple[int, ...], ...]]
    body_test_source_bank: eqx.AbstractVar[tuple[tuple[Fraction, ...], ...]]
    body_test_coefficients: eqx.AbstractVar[Array]


class AbstractReferenceFormBasis(StrictModule, ABC):
    """Provide moment coordinates, entity charts, and reference basis actions."""

    dimension: eqx.AbstractVar[int]
    form_degree: eqx.AbstractVar[int]
    order: eqx.AbstractVar[int]
    family: eqx.AbstractVar[FormElementFamily]
    twist: eqx.AbstractVar[FormTwist]
    exponents: eqx.AbstractVar[tuple[tuple[int, ...], ...]]
    dof_labels: eqx.AbstractVar[tuple[DofLabel, ...]]
    entity_vertices: eqx.AbstractVar[tuple[tuple[tuple[int, ...], ...], ...]]
    coefficients: eqx.AbstractVar[Array]
    tensor_factors: eqx.AbstractVar[AbstractReferenceTensorFactors | None]
    hybrid_factors: eqx.AbstractVar[AbstractReferenceHybridFactors | None]
    functional_points: eqx.AbstractVar[Array]
    functional_weights: eqx.AbstractVar[Array]
    functional_moments: eqx.AbstractVar[Array]
    reference_nodes: eqx.AbstractVar[Array]
    basis_id: eqx.AbstractVar[str]

    @property
    @abstractmethod
    def local_dof_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def tabulate_components(self, points: ArrayLike, /) -> tuple[Array, Array]:
        raise NotImplementedError

    @abstractmethod
    def interpolate(self, values: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def entity_kind(self, face: tuple[int, ...], /) -> str:
        raise NotImplementedError

    @abstractmethod
    def entity_basis(
        self, face: tuple[int, ...], /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> AbstractReferenceFormBasis:
        raise NotImplementedError

    @abstractmethod
    def component_expressions(
        self, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> tuple[tuple[Expression, ...], ...]:
        raise NotImplementedError

    @abstractmethod
    def functional_density_expressions(
        self, face: tuple[int, ...], /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> tuple[tuple[Expression, ...], ...]:
        raise NotImplementedError

    @abstractmethod
    def functional_weights_at(
        self,
        entity_vertices: tuple[int, ...],
        entity_points: ArrayLike,
        quadrature_weights: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        raise NotImplementedError

    @abstractmethod
    def exterior_derivative_matrix(self, target: AbstractReferenceFormBasis, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def permutation_matrix(self, vertices: tuple[int, ...], /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def canonical_permutation(
        self, global_vertices: tuple[int, ...], /
    ) -> tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def entity_permutation_matrix(
        self,
        global_vertices: tuple[int, ...],
        /,
        *,
        entity_bases: tuple[tuple[AbstractReferenceFormBasis | None, ...], ...]
        | None = None,
        cache: EntityMomentCache | None = None,
    ) -> Array:
        raise NotImplementedError


@dataclass(slots=True)
class EntityMomentCache:
    """Caller-scoped entity traces and moment blocks of one reference form basis.

    A block depends only on its local entity, the canonical chart permutation
    of its global vertex IDs and the shared canonical basis. One cache must
    never outlive or cross the basis whose entities it records.
    """

    entities: dict[tuple[int, tuple[int, ...]], AbstractReferenceFormBasis] = field(
        default_factory=dict
    )
    blocks: dict[
        tuple[int, tuple[int, ...], tuple[int, ...], str | None], np.ndarray
    ] = field(default_factory=dict)
