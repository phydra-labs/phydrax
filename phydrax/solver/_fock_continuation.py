#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..operators.quantum import BosonicFockSpace


class FockContinuationPolicy(StrictModule):
    maximum_cutoffs: tuple[int, ...]
    increments: tuple[int, ...]
    top_probability_tolerance: float
    observable_tolerance: float

    def __init__(
        self,
        maximum_cutoffs: Sequence[int],
        increments: Sequence[int],
        /,
        *,
        top_probability_tolerance: float = 1e-6,
        observable_tolerance: float = 1e-6,
    ):
        maxima = tuple(int(value) for value in maximum_cutoffs)
        increments_ = tuple(int(value) for value in increments)
        top_tolerance = float(top_probability_tolerance)
        observable_tolerance_ = float(observable_tolerance)
        if (
            not maxima
            or len(maxima) != len(increments_)
            or any(value < 2 for value in maxima)
            or any(value < 1 for value in increments_)
            or not isfinite(top_tolerance)
            or top_tolerance < 0.0
            or not isfinite(observable_tolerance_)
            or observable_tolerance_ < 0.0
        ):
            raise ValueError(
                "Fock continuation cutoffs, increments, or tolerances are invalid."
            )
        self.maximum_cutoffs = maxima
        self.increments = increments_
        self.top_probability_tolerance = top_tolerance
        self.observable_tolerance = observable_tolerance_


class FockContinuationStage(StrictModule):
    state: Array
    observable: Array
    top_probabilities: Array
    observable_change: Array
    cutoffs: tuple[int, ...]
    valid: Array

    def __init__(
        self,
        state: ArrayLike,
        observable: ArrayLike,
        top_probabilities: ArrayLike,
        observable_change: ArrayLike,
        /,
        *,
        cutoffs: tuple[int, ...],
        evidence_valid: ArrayLike = True,
    ):
        self.state = jnp.asarray(state)
        self.observable = jnp.asarray(observable)
        self.top_probabilities = jnp.asarray(top_probabilities)
        self.observable_change = jnp.asarray(observable_change)
        self.cutoffs = tuple(cutoffs)
        self.valid = (
            jnp.all(jnp.isfinite(self.state))
            & jnp.all(jnp.isfinite(self.observable))
            & jnp.all(jnp.isfinite(self.top_probabilities))
            & jnp.asarray(evidence_valid, dtype=bool)
        )


class FockContinuationResult(StrictModule):
    stages: tuple[FockContinuationStage, ...]
    converged: Array
    exhausted: Array

    def __init__(
        self,
        stages: Sequence[FockContinuationStage],
        /,
        *,
        converged: ArrayLike,
        exhausted: ArrayLike,
    ):
        self.stages = tuple(stages)
        self.converged = jnp.asarray(converged, dtype=bool)
        self.exhausted = jnp.asarray(exhausted, dtype=bool)


class PreparedFockRefinementPlan(StrictModule):
    """Explicit nested cutoff epochs; topology changes only between solves."""

    spaces: tuple[BosonicFockSpace, ...]
    observable_tolerance: float
    boundary_tolerance: float
    plan_id: str

    def __init__(
        self,
        cutoff_sequence: Sequence[Sequence[int]],
        /,
        *,
        observable_tolerance: float,
        boundary_tolerance: float,
        plan_id: str,
    ):
        spaces = tuple(BosonicFockSpace(cutoffs) for cutoffs in cutoff_sequence)
        if len(spaces) < 2:
            raise ValueError("Prepared Fock refinement requires at least two epochs.")
        for coarse, fine in zip(spaces[:-1], spaces[1:], strict=True):
            if len(coarse.cutoffs) != len(fine.cutoffs) or any(
                right <= left
                for left, right in zip(coarse.cutoffs, fine.cutoffs, strict=True)
            ):
                raise ValueError(
                    "Fock cutoff epochs must be strictly nested in every mode."
                )
        if (
            observable_tolerance < 0.0
            or boundary_tolerance < 0.0
            or not isinstance(plan_id, str)
            or not plan_id
        ):
            raise ValueError("Prepared Fock tolerances/plan_id are invalid.")
        self.spaces = spaces
        self.observable_tolerance = float(observable_tolerance)
        self.boundary_tolerance = float(boundary_tolerance)
        self.plan_id = plan_id


class FockRefinementCertificate(StrictModule):
    stages: tuple[FockContinuationStage, ...]
    observable_differences: Array
    boundary_probabilities: Array
    remainder: Array
    stabilized: Array
    exhausted: Array
    valid: Array
    estimate_kind: str
    plan_id: str
    claim: str


def solve_prepared_fock_refinement(
    initial_state: ArrayLike,
    solve_stage: Callable[[BosonicFockSpace, Array], tuple[Array, Array]],
    plan: PreparedFockRefinementPlan,
    /,
    *,
    certified_tail_bound: Callable[[BosonicFockSpace, Array], Array] | None = None,
) -> FockRefinementCertificate:
    """Execute declared topology epochs and separate boundary/observable evidence."""
    if not isinstance(plan, PreparedFockRefinementPlan):
        raise TypeError("plan must be PreparedFockRefinementPlan.")
    if not callable(solve_stage):
        raise TypeError("solve_stage must be callable.")
    if certified_tail_bound is not None and not callable(certified_tail_bound):
        raise TypeError("certified_tail_bound must be callable or None.")
    state = jnp.asarray(initial_state)
    stages = []
    differences = []
    previous_observable = None
    remainder = jnp.asarray(jnp.nan)
    for index, space in enumerate(plan.spaces):
        if index:
            state = plan.spaces[index - 1].embed(state, space)
        state, observable = solve_stage(space, state)
        evidence = space.cutoff_evidence(state)
        difference = (
            jnp.asarray(jnp.inf)
            if previous_observable is None
            else jnp.sqrt(
                jnp.sum(jnp.abs(jnp.asarray(observable) - previous_observable) ** 2)
            )
        )
        stages.append(
            FockContinuationStage(
                state,
                observable,
                evidence.top_level_probability,
                difference,
                cutoffs=space.cutoffs,
                evidence_valid=evidence.valid,
            )
        )
        differences.append(difference)
        previous_observable = jnp.asarray(observable)
        if certified_tail_bound is not None:
            remainder = jnp.asarray(certified_tail_bound(space, state))
    boundary = jnp.stack([jnp.max(stage.top_probabilities) for stage in stages])
    differences_ = jnp.stack(differences)
    stabilized = (differences_[-1] <= plan.observable_tolerance) & (
        boundary[-1] <= plan.boundary_tolerance
    )
    valid = jnp.all(jnp.stack([stage.valid for stage in stages])) & (
        jnp.isfinite(remainder) if certified_tail_bound is not None else True
    )
    return FockRefinementCertificate(
        stages=tuple(stages),
        observable_differences=differences_,
        boundary_probabilities=boundary,
        remainder=remainder,
        stabilized=stabilized,
        exhausted=~stabilized,
        valid=valid,
        estimate_kind="bound" if certified_tail_bound is not None else "difference",
        plan_id=plan.plan_id,
        claim="stabilized-over-declared-fock-cutoffs-not-unbounded-exactness",
    )


def solve_fock_continuation(
    initial_space: BosonicFockSpace,
    initial_state: ArrayLike,
    solve_stage: Callable[[BosonicFockSpace, Array], tuple[Array, Array]],
    policy: FockContinuationPolicy,
    /,
) -> FockContinuationResult:
    if not isinstance(initial_space, BosonicFockSpace):
        raise TypeError("initial_space must be BosonicFockSpace.")
    if not isinstance(policy, FockContinuationPolicy):
        raise TypeError("policy must be FockContinuationPolicy.")
    if not callable(solve_stage):
        raise TypeError("solve_stage must be callable.")
    if len(initial_space.cutoffs) != len(policy.maximum_cutoffs):
        raise ValueError("Fock space and continuation policy mode counts differ.")
    if any(
        maximum < cutoff
        for cutoff, maximum in zip(
            initial_space.cutoffs,
            policy.maximum_cutoffs,
            strict=True,
        )
    ):
        raise ValueError("Fock continuation maxima cannot be below initial cutoffs.")
    space = initial_space
    state = jnp.asarray(initial_state)
    stages = []
    previous_observable = None
    converged = False
    exhausted = False
    while True:
        state, observable = solve_stage(space, state)
        evidence = space.cutoff_evidence(state)
        change = (
            jnp.asarray(jnp.inf)
            if previous_observable is None
            else jnp.linalg.norm(jnp.asarray(observable) - previous_observable)
        )
        stage = FockContinuationStage(
            state,
            observable,
            evidence.top_level_probability,
            change,
            cutoffs=space.cutoffs,
            evidence_valid=evidence.valid,
        )
        stages.append(stage)
        observable_converged = previous_observable is not None and bool(
            change <= policy.observable_tolerance
        )
        if (
            bool(evidence.valid)
            and bool(
                jnp.all(
                    evidence.top_level_probability <= policy.top_probability_tolerance
                )
            )
            and observable_converged
        ):
            converged = True
            break
        force_validation_refinement = (
            previous_observable is None
            or not observable_converged
            or not bool(evidence.valid)
        )
        next_cutoffs = tuple(
            min(
                cutoff + increment
                if force_validation_refinement
                or float(evidence.top_level_probability[index])
                > policy.top_probability_tolerance
                else cutoff,
                maximum,
            )
            for index, (cutoff, increment, maximum) in enumerate(
                zip(
                    space.cutoffs,
                    policy.increments,
                    policy.maximum_cutoffs,
                    strict=True,
                )
            )
        )
        if next_cutoffs == space.cutoffs:
            exhausted = True
            break
        fine = BosonicFockSpace(next_cutoffs)
        state = space.embed(state, fine)
        previous_observable = jnp.asarray(observable)
        space = fine
    return FockContinuationResult(stages, converged=converged, exhausted=exhausted)


__all__ = [
    "FockRefinementCertificate",
    "PreparedFockRefinementPlan",
    "FockContinuationPolicy",
    "FockContinuationResult",
    "FockContinuationStage",
    "solve_prepared_fock_refinement",
    "solve_fock_continuation",
]
