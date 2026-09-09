#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Implicit point kinetics with an arbitrary fixed delayed-neutron family axis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve_checked,
    TolerancePolicy,
)


class ReactorKineticsState(StrictModule):
    neutron_population: Array
    precursor_populations: Array
    time_s: Array

    def __init__(
        self,
        neutron_population: ArrayLike,
        precursor_populations: ArrayLike,
        time_s: ArrayLike = 0.0,
        /,
    ):
        population = jnp.asarray(neutron_population, dtype=jnp.float64)
        precursors = jnp.asarray(precursor_populations, dtype=population.dtype)
        time = jnp.asarray(time_s, dtype=population.dtype)
        if population.shape != () or precursors.ndim != 1 or time.shape != ():
            raise ValueError(
                "Point kinetics requires scalar population/time and rank-one precursors."
            )
        self.neutron_population = population
        self.precursor_populations = precursors
        self.time_s = time


class ReactorKineticsStepResult(StrictModule):
    candidate_state: ReactorKineticsState
    accepted_state: ReactorKineticsState
    residual_norm: Array
    finite: Array
    domain_valid: Array
    sensitivity_valid: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class DelayedNeutronKineticsPlan:
    delayed_fractions: np.ndarray
    decay_constants_s: np.ndarray
    generation_time_s: float
    source_id: str
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        fractions = np.array(self.delayed_fractions, dtype=np.float64, copy=True)
        decay = np.array(self.decay_constants_s, dtype=np.float64, copy=True)
        generation = float(self.generation_time_s)
        source = str(self.source_id).strip()
        if fractions.ndim != 1 or fractions.size < 1 or decay.shape != fractions.shape:
            raise ValueError(
                "Delayed fractions and decay constants require one shared family axis."
            )
        if (
            np.any(~np.isfinite(fractions))
            or np.any(fractions < 0.0)
            or np.sum(fractions) >= 1.0
        ):
            raise ValueError(
                "Delayed-neutron fractions must be finite, nonnegative, and sum below one."
            )
        if np.any(~np.isfinite(decay)) or np.any(decay <= 0.0):
            raise ValueError("Precursor decay constants must be finite and positive.")
        if not math.isfinite(generation) or generation <= 0.0:
            raise ValueError("generation_time_s must be finite and positive.")
        if not source or source != self.source_id:
            raise ValueError("source_id must be non-empty canonical text.")
        fractions.setflags(write=False)
        decay.setflags(write=False)
        object.__setattr__(self, "delayed_fractions", fractions)
        object.__setattr__(self, "decay_constants_s", decay)
        object.__setattr__(self, "generation_time_s", generation)
        object.__setattr__(self, "source_id", source)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "delayed-neutron-kinetics-plan",
                    "delayed_fractions": array_tree_fingerprint(fractions),
                    "decay_constants_s": array_tree_fingerprint(decay),
                    "generation_time_s": generation,
                    "source": source,
                    "integration": "implicit-euler",
                }
            ),
        )

    @property
    def family_count(self) -> int:
        return int(self.delayed_fractions.size)

    def equilibrium_state(
        self, neutron_population: float, time_s: float = 0.0, /
    ) -> ReactorKineticsState:
        population = float(neutron_population)
        if not math.isfinite(population) or population <= 0.0:
            raise ValueError(
                "Equilibrium neutron population must be finite and positive."
            )
        precursors = (
            self.delayed_fractions
            / (self.generation_time_s * self.decay_constants_s)
            * population
        )
        return ReactorKineticsState(population, precursors, time_s)

    def prepare(self) -> PreparedDelayedNeutronKinetics:
        return PreparedDelayedNeutronKinetics(
            jnp.asarray(self.delayed_fractions),
            jnp.asarray(self.decay_constants_s),
            self.generation_time_s,
            self.plan_id,
        )


class PreparedDelayedNeutronKinetics(StrictModule, NonTrainableState):
    delayed_fractions: Array
    decay_constants_s: Array
    generation_time_s: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def step(
        self,
        state: ReactorKineticsState,
        reactivity: ArrayLike,
        external_source_s: ArrayLike,
        dt_s: ArrayLike,
        /,
    ) -> ReactorKineticsStepResult:
        if not isinstance(state, ReactorKineticsState):
            raise TypeError("state must be ReactorKineticsState.")
        if state.precursor_populations.shape != self.delayed_fractions.shape:
            raise ValueError("Kinetics state does not match delayed-neutron families.")
        rho = jnp.asarray(reactivity, dtype=jnp.float64)
        source = jnp.asarray(external_source_s, dtype=rho.dtype)
        dt = jnp.asarray(dt_s, dtype=rho.dtype)
        if rho.shape != () or source.shape != () or dt.shape != ():
            raise ValueError("Reactivity, external source, and timestep must be scalar.")
        domain_valid = (
            jnp.isfinite(state.neutron_population)
            & jnp.all(jnp.isfinite(state.precursor_populations))
            & jnp.isfinite(rho)
            & jnp.isfinite(source)
            & jnp.isfinite(dt)
            & (state.neutron_population >= 0.0)
            & jnp.all(state.precursor_populations >= 0.0)
            & (source >= 0.0)
            & (dt > 0.0)
        )
        safe_dt = jnp.where(domain_valid, dt, 1.0)
        families = self.delayed_fractions.size
        size = families + 1
        matrix = jnp.eye(size, dtype=rho.dtype)
        beta = jnp.sum(self.delayed_fractions)
        matrix = matrix.at[0, 0].add(-safe_dt * (rho - beta) / self.generation_time_s)
        matrix = matrix.at[0, 1:].set(-safe_dt * self.decay_constants_s)
        matrix = matrix.at[1:, 0].set(
            -safe_dt * self.delayed_fractions / self.generation_time_s
        )
        matrix = matrix.at[1:, 1:].set(jnp.diag(1.0 + safe_dt * self.decay_constants_s))
        rhs = jnp.concatenate(
            (
                (
                    state.neutron_population
                    + safe_dt * jnp.where(jnp.isfinite(source), source, 0.0)
                )[None],
                state.precursor_populations,
            )
        )
        operator = DenseLinearOperator(matrix, operator_id=self.plan_id)
        policy = LinearSolvePolicy(
            DenseLU(),
            tolerance=TolerancePolicy(relative=1.0e-12, absolute=1.0e-14),
            differentiation=DifferentiationPolicy("mathematical"),
            failure=FailurePolicy("status"),
        )
        linear, evidence = solve_checked(LinearSystem(operator), rhs, policy=policy)
        candidate = ReactorKineticsState(
            linear.value[0], linear.value[1:], state.time_s + safe_dt
        )
        finite = jnp.all(jnp.isfinite(linear.value))
        nonnegative = jnp.all(linear.value >= 0.0)
        successful = (
            domain_valid & linear.successful & evidence.valid & finite & nonnegative
        )
        accepted = ReactorKineticsState(
            jnp.where(successful, candidate.neutron_population, state.neutron_population),
            jnp.where(
                successful, candidate.precursor_populations, state.precursor_populations
            ),
            jnp.where(successful, state.time_s + dt, state.time_s),
        )
        return ReactorKineticsStepResult(
            candidate,
            accepted,
            evidence.true_residual_norm,
            finite,
            domain_valid,
            successful,
            successful,
        )


__all__ = [
    "DelayedNeutronKineticsPlan",
    "PreparedDelayedNeutronKinetics",
    "ReactorKineticsState",
    "ReactorKineticsStepResult",
]
