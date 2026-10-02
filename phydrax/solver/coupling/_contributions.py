#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Row-space-aware contributions of coupling laws to a spatial coupled problem.

A contribution states exactly which coordinates it reads and which residual
rows it writes. An endpoint in the ``"full"`` space names a component field:
the assembler composes the component's constraint map once (``P z + g`` on the
state side, ``P^T`` on the row side). An endpoint in the ``"reduced"`` space
names a native block in solve coordinates (a component's own reduced block or
an unknown owned by a law) and is used without further composition. Operator
spaces must equal the resolved endpoint spaces, so a contribution prepared on
one chart cannot be applied on another.

Every contribution carries the imposition identity of the law that produced
it. A law publishes its facet/row impositions separately so that a second law,
or an owner's own boundary law, on the same interface is refused.
"""

from __future__ import annotations

import abc
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...linalg import AbstractLinearOperator, AbstractVectorSpace
from ...typing import checked, parse
from ._parameters import RuntimeInput


ContributionSpace: TypeAlias = Literal["full", "reduced"]


@final
class ContributionEndpoint(StrictModule, NonTrainableState):
    """Exact source or target of one contribution.

    ``owner`` is a component or law name. With ``space="full"`` ``block``
    names a component field (its full coefficient space); with
    ``space="reduced"`` it names a native state or row block in solve
    coordinates.
    """

    owner: str = eqx.field(static=True)
    block: str = eqx.field(static=True)
    space: ContributionSpace = eqx.field(static=True)

    def __init__(
        self, owner: str, block: str, /, *, space: ContributionSpace = "full"
    ) -> None:
        owner_ = canonical_identifier(owner, "owner")
        block_ = canonical_identifier(block, "block")
        space_ = parse(space, ContributionSpace, "space")
        self.owner = owner_
        self.block = block_
        self.space = space_

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.owner, self.block, self.space)


class AbstractContribution(StrictModule):
    """One term of a law in the rows of a coupled problem."""

    law_id: eqx.AbstractVar[str]
    imposition_id: eqx.AbstractVar[str]

    @property
    @abc.abstractmethod
    def targets(self) -> tuple[ContributionEndpoint, ...]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def sources(self) -> tuple[ContributionEndpoint, ...]:
        raise NotImplementedError


def _law_identity(law_id: str, imposition_id: str, /) -> tuple[str, str]:
    return canonical_identifier(law_id, "law_id"), canonical_identifier(
        imposition_id, "imposition_id"
    )


@final
class LinearContribution(AbstractContribution, NonTrainableState):
    """Linear term ``operator(source)`` added to the rows of ``target``.

    ``operator.source`` must equal the source endpoint space and
    ``operator.target`` the target row space (the coordinate dual of a full
    field, or a native row block). Sparse and matrix-free operators are kept
    as given.
    """

    target: ContributionEndpoint
    source: ContributionEndpoint
    operator: AbstractLinearOperator
    law_id: str = eqx.field(static=True)
    imposition_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        target: ContributionEndpoint,
        source: ContributionEndpoint,
        operator: AbstractLinearOperator,
        /,
        *,
        law_id: str,
        imposition_id: str,
    ) -> None:
        if not isinstance(target, ContributionEndpoint) or not isinstance(
            source, ContributionEndpoint
        ):
            raise TypeError("target and source must be ContributionEndpoint values.")
        if operator.batch_shape:
            raise ValueError("Contribution operators must be unbatched.")
        law, imposition = _law_identity(law_id, imposition_id)
        self.target = target
        self.source = source
        self.operator = operator
        self.law_id = law
        self.imposition_id = imposition

    @property
    def targets(self) -> tuple[ContributionEndpoint, ...]:
        return (self.target,)

    @property
    def sources(self) -> tuple[ContributionEndpoint, ...]:
        return (self.source,)


@final
class LoadContribution(AbstractContribution, NonTrainableState):
    """State-independent load ``values`` added to the rows of ``target``.

    The residual convention is ``R(u) = A u - b``, so a law that prescribes
    a data term ``b`` publishes ``-b`` here. ``values`` is a dynamic leaf that
    parameter bindings may replace without repreparing the problem.
    """

    target: ContributionEndpoint
    values: Array
    law_id: str = eqx.field(static=True)
    imposition_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        target: ContributionEndpoint,
        values: ArrayLike,
        /,
        *,
        law_id: str,
        imposition_id: str,
    ) -> None:
        array = jnp.asarray(values)
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            raise TypeError("Load values must be floating point.")
        law, imposition = _law_identity(law_id, imposition_id)
        self.target = target
        self.values = array
        self.law_id = law
        self.imposition_id = imposition

    @property
    def targets(self) -> tuple[ContributionEndpoint, ...]:
        return (self.target,)

    @property
    def sources(self) -> tuple[ContributionEndpoint, ...]:
        return ()


class AbstractContributionResidual(StrictModule):
    """Nonlinear residual term of one law, owned by that law.

    ``evaluate`` receives one array per source endpoint (full coefficients for
    full endpoints) and returns one covector per target endpoint in its row
    space. Its linearization is the JAX derivative of ``evaluate``.
    ``runtime_inputs`` names the component runtime inputs its evaluation reads
    (a learned response bound at every solve); such a residual has no
    argument-free evaluation, so preparation never probes it without
    arguments, and every named input must be the target of a refresh
    ``ParameterBinding`` so its value passes the binding's port and authority
    admission.
    """

    @property
    def runtime_inputs(self) -> tuple[RuntimeInput, ...]:
        return ()

    @abc.abstractmethod
    def evaluate(self, inputs: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        raise NotImplementedError


@final
class ResidualContribution(AbstractContribution, NonTrainableState):
    """Nonlinear contribution evaluated by a law-owned residual.

    ``affine`` states that ``evaluate`` is affine in its inputs; only affine
    residual contributions enter linear assembly, where their action is the
    exact JAX linearization at zero.
    """

    target_endpoints: tuple[ContributionEndpoint, ...]
    source_endpoints: tuple[ContributionEndpoint, ...]
    residual: AbstractContributionResidual
    affine: bool = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    imposition_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        targets: tuple[ContributionEndpoint, ...],
        sources: tuple[ContributionEndpoint, ...],
        residual: AbstractContributionResidual,
        /,
        *,
        affine: bool,
        law_id: str,
        imposition_id: str,
    ) -> None:
        for endpoints, name in ((targets, "targets"), (sources, "sources")):
            if not isinstance(endpoints, tuple) or not all(
                isinstance(endpoint, ContributionEndpoint) for endpoint in endpoints
            ):
                raise TypeError(f"{name} must be a tuple of ContributionEndpoint values.")
        if not targets or not sources:
            raise ValueError("A residual contribution reads and writes at least once.")
        if not isinstance(affine, bool):
            raise TypeError("affine must be a bool.")
        law, imposition = _law_identity(law_id, imposition_id)
        self.target_endpoints = targets
        self.source_endpoints = sources
        self.residual = residual
        self.affine = affine
        self.law_id = law
        self.imposition_id = imposition

    @property
    def targets(self) -> tuple[ContributionEndpoint, ...]:
        return self.target_endpoints

    @property
    def sources(self) -> tuple[ContributionEndpoint, ...]:
        return self.source_endpoints


@final
class EliminationContribution(AbstractContribution, NonTrainableState):
    """Explicit elimination ``u_e[rows] = E u_r[columns]`` of full field rows.

    ``eliminated`` and ``retained`` are full field endpoints of two
    components. ``rows`` are full rows of the eliminated field and
    ``columns`` full rows of the retained field; ``relation`` is the dense
    ``(rows, columns)`` coefficient matrix of the explicit basis relation.
    The eliminated rows are removed from the solve coordinates, and their
    residual rows are summed into the retained rows through ``E^T``. Rows the
    eliminated owner already imposes strongly may not be eliminated.
    """

    eliminated: ContributionEndpoint
    retained: ContributionEndpoint
    rows: Array
    columns: Array
    relation: Array
    law_id: str = eqx.field(static=True)
    imposition_id: str = eqx.field(static=True)
    relation_id: str = eqx.field(static=True)

    def __init__(
        self,
        eliminated: ContributionEndpoint,
        retained: ContributionEndpoint,
        rows: ArrayLike,
        columns: ArrayLike,
        relation: ArrayLike,
        /,
        *,
        law_id: str,
        imposition_id: str,
    ) -> None:
        if not isinstance(eliminated, ContributionEndpoint) or not isinstance(
            retained, ContributionEndpoint
        ):
            raise TypeError(
                "eliminated and retained must be ContributionEndpoint values."
            )
        if eliminated.space != "full" or retained.space != "full":
            raise ValueError("Elimination relates full field rows of two components.")
        if eliminated.owner == retained.owner:
            raise ValueError("Elimination relates two different components.")
        rows_ = np.asarray(rows)
        columns_ = np.asarray(columns)
        matrix = np.asarray(relation, dtype=np.float64)
        for values, name in ((rows_, "rows"), (columns_, "columns")):
            if (
                values.ndim != 1
                or values.size == 0
                or not np.issubdtype(values.dtype, np.integer)
                or np.any(values < 0)
                or np.unique(values).size != values.size
            ):
                raise ValueError(f"{name} must be unique non-negative integer rows.")
        if matrix.shape != (rows_.size, columns_.size) or not np.all(np.isfinite(matrix)):
            raise ValueError("relation must be a finite (rows, columns) matrix.")
        law, imposition = _law_identity(law_id, imposition_id)
        self.eliminated = eliminated
        self.retained = retained
        self.rows = jnp.asarray(rows_.astype(np.int32))
        self.columns = jnp.asarray(columns_.astype(np.int32))
        self.relation = jnp.asarray(matrix)
        self.law_id = law
        self.imposition_id = imposition
        self.relation_id = canonical_fingerprint(
            {
                "kind": "elimination-relation",
                "eliminated": list(eliminated.key),
                "retained": list(retained.key),
                "rows": array_tree_fingerprint(rows_.astype(np.int32)),
                "columns": array_tree_fingerprint(columns_.astype(np.int32)),
                "relation": array_tree_fingerprint(matrix),
            }
        )

    @property
    def targets(self) -> tuple[ContributionEndpoint, ...]:
        return (self.retained,)

    @property
    def sources(self) -> tuple[ContributionEndpoint, ...]:
        return (self.eliminated, self.retained)


@final
class LawBlock(StrictModule, NonTrainableState):
    """One unknown or residual-row block owned by a law (for example a multiplier)."""

    name: str = eqx.field(static=True)
    space: AbstractVectorSpace

    @checked
    def __init__(self, name: str, space: AbstractVectorSpace, /) -> None:
        name_ = canonical_identifier(name, "name")
        self.name = name_
        self.space = space


@final
class LawImposition(StrictModule, NonTrainableState):
    """Facets and rows of one component field on which a law imposes itself.

    ``facets`` are entities of ``entity_set_id`` and ``rows`` the full rows
    whose trace the law constrains or loads. Two laws, or a law and an owner's
    own boundary law, may not act on the same facets.
    """

    component: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    facets: Array
    rows: Array
    imposition_id: str = eqx.field(static=True)

    def __init__(
        self,
        component: str,
        field: str,
        /,
        *,
        field_space_id: str,
        entity_set_id: str,
        facets: ArrayLike,
        rows: ArrayLike,
        imposition_id: str,
    ) -> None:
        facets_ = np.unique(np.asarray(facets, dtype=np.int32))
        rows_ = np.unique(np.asarray(rows, dtype=np.int32))
        if facets_.size == 0:
            raise ValueError("A law imposition acts on at least one facet.")
        self.component = canonical_identifier(component, "component")
        self.field = canonical_identifier(field, "field")
        self.field_space_id = canonical_identifier(field_space_id, "field_space_id")
        self.entity_set_id = canonical_identifier(entity_set_id, "entity_set_id")
        self.facets = jnp.asarray(facets_)
        self.rows = jnp.asarray(rows_)
        self.imposition_id = canonical_identifier(imposition_id, "imposition_id")

    @checked
    def overlaps(self, other: LawImposition, /) -> bool:
        """Whether both impositions act on a common facet of one field."""
        return bool(
            self.component == other.component
            and self.field_space_id == other.field_space_id
            and self.entity_set_id == other.entity_set_id
            and np.intersect1d(np.asarray(self.facets), np.asarray(other.facets)).size
        )


type Contribution = (
    LinearContribution | LoadContribution | ResidualContribution | EliminationContribution
)

__all__ = [
    "AbstractContribution",
    "AbstractContributionResidual",
    "Contribution",
    "ContributionEndpoint",
    "ContributionSpace",
    "EliminationContribution",
    "LawBlock",
    "LawImposition",
    "LinearContribution",
    "LoadContribution",
    "ResidualContribution",
]
