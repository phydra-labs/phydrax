#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class ApplicabilityStatus(IntEnum):
    APPLICABLE = 0
    OUTSIDE_APPLICABILITY = 1
    MISSING_DATA = 2


class LoadCombination(StrictModule, NonTrainableState):
    combination_id: str = eqx.field(static=True)
    factors: tuple[tuple[str, float], ...] = eqx.field(static=True)
    category: str = eqx.field(static=True)
    clause_id: str = eqx.field(static=True)

    def __init__(
        self,
        combination_id: str,
        factors: Mapping[str, float],
        /,
        *,
        category: str,
        clause_id: str,
    ):
        self.combination_id = str(combination_id)
        self.factors = tuple(
            sorted((str(key), float(value)) for key, value in factors.items())
        )
        self.category = str(category)
        self.clause_id = str(clause_id)

    def combine(self, actions: Mapping[str, ArrayLike], /) -> Array:
        missing = tuple(name for name, _ in self.factors if name not in actions)
        if missing:
            raise KeyError(f"Missing load actions: {missing}.")
        arrays = [
            float(factor) * jnp.asarray(actions[name]) for name, factor in self.factors
        ]
        return sum(arrays[1:], arrays[0])


class ClauseEvidence(StrictModule):
    standard_id: str = eqx.field(static=True)
    edition: str = eqx.field(static=True)
    clause_id: str = eqx.field(static=True)
    applicability: Array
    demand: Array
    nominal_resistance: Array
    resistance_factor: Array
    factored_resistance: Array
    utilization: Array
    governing_case: str = eqx.field(static=True)
    assumptions: tuple[str, ...] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return (self.applicability == int(ApplicabilityStatus.APPLICABLE)) & jnp.all(
            self.utilization <= 1.0
        )


class AbstractStructuralStandard(StrictModule, NonTrainableState):
    organization: str = eqx.field(static=True)
    standard_id: str = eqx.field(static=True)
    edition: str = eqx.field(static=True)
    jurisdiction: str = eqx.field(static=True)

    @abc.abstractmethod
    def load_combinations(self, /) -> tuple[LoadCombination, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def member_resistance(
        self,
        demand: ArrayLike,
        nominal_resistance: ArrayLike,
        /,
        *,
        clause_id: str,
        governing_case: str,
        applicability: ApplicabilityStatus = ApplicabilityStatus.APPLICABLE,
        assumptions: Sequence[str] = (),
    ) -> ClauseEvidence:
        raise NotImplementedError


class GenericLimitStateStandard(AbstractStructuralStandard):
    """Code-neutral user-supplied combinations and resistance factors."""

    combinations: tuple[LoadCombination, ...]
    resistance_factor: float = eqx.field(static=True)

    def __init__(
        self,
        combinations: Sequence[LoadCombination],
        /,
        *,
        resistance_factor: float,
        organization: str = "user-supplied",
        standard_id: str = "generic-limit-state",
        edition: str = "declared",
        jurisdiction: str = "declared",
    ):
        combinations_ = tuple(combinations)
        if not combinations_ or resistance_factor <= 0.0:
            raise ValueError(
                "A standard requires combinations and a positive resistance factor."
            )
        self.combinations = combinations_
        self.resistance_factor = float(resistance_factor)
        self.organization = str(organization)
        self.standard_id = str(standard_id)
        self.edition = str(edition)
        self.jurisdiction = str(jurisdiction)

    def load_combinations(self, /) -> tuple[LoadCombination, ...]:
        return self.combinations

    def member_resistance(
        self,
        demand: ArrayLike,
        nominal_resistance: ArrayLike,
        /,
        *,
        clause_id: str,
        governing_case: str,
        applicability: ApplicabilityStatus = ApplicabilityStatus.APPLICABLE,
        assumptions: Sequence[str] = (),
    ) -> ClauseEvidence:
        demand_ = jnp.asarray(demand)
        nominal = jnp.asarray(nominal_resistance, dtype=demand_.dtype)
        factor = jnp.asarray(self.resistance_factor, dtype=demand_.dtype)
        resistance = factor * nominal
        utilization = jnp.abs(demand_) / jnp.maximum(
            resistance, jnp.finfo(demand_.dtype).tiny
        )
        return ClauseEvidence(
            self.standard_id,
            self.edition,
            str(clause_id),
            jnp.asarray(int(applicability), dtype=jnp.int32),
            demand_,
            nominal,
            factor,
            resistance,
            utilization,
            str(governing_case),
            tuple(str(value) for value in assumptions),
        )


class DirectAnalysisEvidence(StrictModule):
    second_order_displacement: Array
    notional_load: Array
    stiffness_reduction: Array
    imperfection_amplitude: Array
    applicable: Array
    assumptions: tuple[str, ...] = eqx.field(static=True)


def direct_analysis_inputs(
    first_order_displacement: ArrayLike,
    gravity_load: ArrayLike,
    /,
    *,
    notional_load_fraction: float,
    stiffness_reduction: float,
    imperfection_amplitude: float,
) -> DirectAnalysisEvidence:
    displacement = jnp.asarray(first_order_displacement)
    gravity = jnp.asarray(gravity_load, dtype=displacement.dtype)
    notional = float(notional_load_fraction) * gravity
    reduction = jnp.asarray(stiffness_reduction, dtype=displacement.dtype)
    applicable = (
        (reduction > 0.0)
        & (reduction <= 1.0)
        & (notional_load_fraction >= 0.0)
        & (imperfection_amplitude >= 0.0)
    )
    return DirectAnalysisEvidence(
        displacement,
        notional,
        reduction,
        jnp.asarray(imperfection_amplitude, dtype=displacement.dtype),
        applicable,
        (
            "second-order equilibrium required",
            "notional load explicitly applied",
            "stiffness reduction explicitly declared",
        ),
    )


__all__ = [
    "AbstractStructuralStandard",
    "ApplicabilityStatus",
    "ClauseEvidence",
    "DirectAnalysisEvidence",
    "GenericLimitStateStandard",
    "LoadCombination",
    "direct_analysis_inputs",
]
