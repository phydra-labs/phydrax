#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Mapping

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._term import AbstractScalarTerm
from ..domain import DomainFunction
from ..integration._api import IntegrationRealization, reduce
from ..transport import AbstractGroundCost


class _MongeCostEvaluator(StrictModule):
    map_evaluator: Any
    cost: AbstractGroundCost

    def __call__(self, *args, key=None):
        source = (
            jnp.asarray(args[0])
            if len(args) == 1
            else jnp.concatenate(tuple(jnp.asarray(item) for item in args), axis=-1)
        )
        target = self.map_evaluator(*args, key=key)
        source_values = source.data if isinstance(source, cx.Field) else source
        target_values = target.data if isinstance(target, cx.Field) else target
        return self.cost.pairwise(source_values, target_values)


def _scalar(value: Any, /) -> Array:
    array = jnp.asarray(value.data if isinstance(value, cx.Field) else value)
    if array.shape != ():
        raise ValueError("transport learning objective contributions must be scalar.")
    return array


class MongeMapTerm(AbstractScalarTerm):
    """Represented Monge cost plus a caller-declared native pushforward discrepancy."""

    source_realization: IntegrationRealization
    cost: AbstractGroundCost
    discrepancy_provider: Any
    transport_weight: Array
    discrepancy_weight: Array
    map_field: str = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        map_field: str,
        source_realization: IntegrationRealization,
        /,
        *,
        cost: AbstractGroundCost,
        discrepancy_provider: Any,
        transport_weight: float = 1.0,
        discrepancy_weight: float = 1.0,
        label: str | None = None,
    ):
        if not isinstance(map_field, str) or not map_field:
            raise ValueError("map_field must be non-empty.")
        if not isinstance(source_realization, IntegrationRealization):
            raise TypeError("source_realization must be an IntegrationRealization.")
        if not isinstance(cost, AbstractGroundCost):
            raise TypeError("cost must be an AbstractGroundCost.")
        if not callable(discrepancy_provider):
            raise TypeError("discrepancy_provider must be callable.")
        self.map_field = map_field
        self.source_realization = source_realization
        self.cost = cost
        self.discrepancy_provider = discrepancy_provider
        self.transport_weight = jnp.asarray(transport_weight, dtype=float)
        self.discrepancy_weight = jnp.asarray(discrepancy_weight, dtype=float)
        self.label = label

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_=None,
        **kwargs,
    ) -> Array:
        del iter_, kwargs
        field = functions[self.map_field]
        cost_field = DomainFunction(
            domain=field.domain,
            deps=field.deps,
            func=_MongeCostEvaluator(field.func, self.cost),
            metadata={"objective": "represented-monge-cost"},
        )
        transport = _scalar(reduce(cost_field, self.source_realization).value)
        discrepancy = _scalar(self.discrepancy_provider(functions, key))
        return self.transport_weight * transport + self.discrepancy_weight * discrepancy


class NeuralDualTransportTerm(AbstractScalarTerm):
    """Finite/empirical Kantorovich dual with explicit pair constraints."""

    source_realization: IntegrationRealization
    target_realization: IntegrationRealization
    cost: AbstractGroundCost
    pair_source: Any
    constraint_weight: Array
    source_potential: str = eqx.field(static=True)
    target_potential: str = eqx.field(static=True)
    full_pair_coverage: bool = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        source_potential: str,
        target_potential: str,
        source_realization: IntegrationRealization,
        target_realization: IntegrationRealization,
        /,
        *,
        cost: AbstractGroundCost,
        pair_source: Any,
        constraint_weight: float = 1.0,
        full_pair_coverage: bool = False,
        label: str | None = None,
    ):
        if not source_potential or not target_potential:
            raise ValueError("potential field names must be non-empty.")
        if not isinstance(source_realization, IntegrationRealization) or not isinstance(
            target_realization, IntegrationRealization
        ):
            raise TypeError(
                "potential realizations must be IntegrationRealization values."
            )
        if not isinstance(cost, AbstractGroundCost) or not callable(pair_source):
            raise TypeError("cost and pair_source must implement their typed contracts.")
        self.source_potential = source_potential
        self.target_potential = target_potential
        self.source_realization = source_realization
        self.target_realization = target_realization
        self.cost = cost
        self.pair_source = pair_source
        self.constraint_weight = jnp.asarray(constraint_weight, dtype=float)
        self.full_pair_coverage = bool(full_pair_coverage)
        self.label = label

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_=None,
        **kwargs,
    ) -> Array:
        del iter_, kwargs
        source_field = functions[self.source_potential]
        target_field = functions[self.target_potential]
        source_expectation = _scalar(reduce(source_field, self.source_realization).value)
        target_expectation = _scalar(reduce(target_field, self.target_realization).value)
        source_points, target_points, pair_weights = self.pair_source(key)
        left = source_field.func(source_points, key=key)
        right = target_field.func(target_points, key=key)
        left_values = left.data if isinstance(left, cx.Field) else left
        right_values = right.data if isinstance(right, cx.Field) else right
        costs = self.cost.matrix(source_points, target_points)
        # Pair sources provide aligned arrays; the matrix diagonal is the selected set.
        violations = jnp.maximum(
            jnp.asarray(left_values) + jnp.asarray(right_values) - jnp.diag(costs),
            0.0,
        )
        weights = jnp.asarray(pair_weights)
        if weights.shape != violations.shape:
            raise ValueError("pair weights must align the explicit constraint pairs.")
        penalty = jnp.sum(weights * violations**2) / jnp.sum(weights)
        # FunctionalSolver minimizes, hence the represented dual is negated.
        return (
            -(source_expectation + target_expectation) + self.constraint_weight * penalty
        )


class LearnedTransportAudit(StrictModule):
    represented_cost: Array
    marginal_discrepancy: Array
    held_out_dual_violation: Array
    valid: Array
    semantics: str = eqx.field(static=True)
    bounded_non_claim: str = eqx.field(static=True)


def audit_transport_map(
    represented_cost: Any,
    marginal_discrepancy: Any,
    held_out_dual_violation: Any,
    /,
    *,
    full_finite_pairs: bool = False,
) -> LearnedTransportAudit:
    """Assemble held-out diagnostics without claiming a global Monge proof."""
    cost = _scalar(represented_cost)
    discrepancy = _scalar(marginal_discrepancy)
    violation = _scalar(held_out_dual_violation)
    valid = (
        jnp.isfinite(cost)
        & jnp.isfinite(discrepancy)
        & jnp.isfinite(violation)
        & (discrepancy >= 0.0)
        & (violation >= 0.0)
    )
    return LearnedTransportAudit(
        represented_cost=cost,
        marginal_discrepancy=discrepancy,
        held_out_dual_violation=violation,
        valid=valid,
        semantics=(
            "finite-dual-audit" if full_finite_pairs else "empirical-held-out-audit"
        ),
        bounded_non_claim=(
            "The audit concerns represented finite/empirical objectives and does not "
            "certify a global Monge map or continuum Kantorovich optimum."
        ),
    )


__all__ = [
    "LearnedTransportAudit",
    "MongeMapTerm",
    "NeuralDualTransportTerm",
    "audit_transport_map",
]
