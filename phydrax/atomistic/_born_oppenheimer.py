#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ._hybrid import AbstractExternalAtomisticProvider, ExternalAtomisticEvaluation
from ._system import PreparedAtomisticSystem


ElectronicEvaluator = Callable[
    [PreparedAtomisticSystem, Array, Array | None], ExternalAtomisticEvaluation
]


class CallableBornOppenheimerProvider(AbstractExternalAtomisticProvider):
    """Explicit host-provider boundary for one Born–Oppenheimer surface."""

    evaluator: ElectronicEvaluator
    provider_id: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)

    def __init__(
        self,
        evaluator: ElectronicEvaluator,
        provider_id: str,
        /,
        *,
        conservative: bool = True,
        differentiable: bool = False,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        identifier = str(provider_id).strip()
        if not identifier:
            raise ValueError("provider_id must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = identifier
        self.conservative = bool(conservative)
        self.differentiable = bool(differentiable)

    def evaluate(
        self,
        system: PreparedAtomisticSystem,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        /,
    ) -> ExternalAtomisticEvaluation:
        position = jnp.asarray(positions, dtype=system.plan.coordinate_dtype)
        expected = (system.capacity, 3)
        if position.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        vectors = None if cell_vectors is None else jnp.asarray(cell_vectors)
        result = self.evaluator(system, position, vectors)
        if not isinstance(result, ExternalAtomisticEvaluation):
            raise TypeError(
                "Born–Oppenheimer evaluator must return ExternalAtomisticEvaluation."
            )
        if result.provider_id != self.provider_id:
            raise ValueError("Born–Oppenheimer result provider identity changed.")
        if result.forces.shape != expected or result.energy.shape != ():
            raise ValueError("Born–Oppenheimer energy or force shape is invalid.")
        return result


class BornOppenheimerState(StrictModule):
    positions: Array
    momenta: Array
    forces: Array
    energy: Array
    time: Array
    step_index: Array
    successful: Array
    provider_id: str = eqx.field(static=True)


class BornOppenheimerStep(StrictModule):
    state: BornOppenheimerState
    initial_evaluation: ExternalAtomisticEvaluation
    final_evaluation: ExternalAtomisticEvaluation
    successful: Array


class BornOppenheimerVelocityVerletPlan(StrictModule, NonTrainableState):
    system: PreparedAtomisticSystem
    provider: AbstractExternalAtomisticProvider
    step_size: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: PreparedAtomisticSystem,
        provider: AbstractExternalAtomisticProvider,
        step_size: float,
        /,
    ):
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("system must be PreparedAtomisticSystem.")
        if not isinstance(provider, AbstractExternalAtomisticProvider):
            raise TypeError("provider must implement AbstractExternalAtomisticProvider.")
        step = float(step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        self.system = system
        self.provider = provider
        self.step_size = step
        self.plan_id = canonical_fingerprint(
            {
                "kind": "born-oppenheimer-velocity-verlet",
                "system": system.prepared_id,
                "provider": provider.provider_id,
                "step_size": step,
            }
        )

    def initialize(
        self,
        positions: ArrayLike,
        /,
        *,
        velocity: ArrayLike | None = None,
        momentum: ArrayLike | None = None,
        time: ArrayLike = 0.0,
    ) -> BornOppenheimerState:
        if (velocity is None) == (momentum is None):
            raise ValueError("Supply exactly one of velocity or momentum.")
        position = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        masses = self.system.plan.masses.astype(position.dtype)
        momenta = (
            jnp.asarray(momentum, dtype=position.dtype)
            if momentum is not None
            else masses[:, None] * jnp.asarray(velocity, dtype=position.dtype)
        )
        cell = None if self.system.cell is None else self.system.cell.vectors
        evaluation = self.provider.evaluate(self.system, position, cell)
        successful = evaluation.successful & jnp.all(jnp.isfinite(momenta))
        return BornOppenheimerState(
            position,
            momenta,
            evaluation.forces,
            evaluation.energy,
            jnp.asarray(time, dtype=position.dtype),
            jnp.zeros((), dtype=jnp.int32),
            successful,
            self.provider.provider_id,
        )

    def step(self, state: BornOppenheimerState, /) -> BornOppenheimerStep:
        if state.provider_id != self.provider.provider_id:
            raise ValueError("Born–Oppenheimer state belongs to another provider.")
        initial = ExternalAtomisticEvaluation(
            state.energy,
            state.forces,
            None,
            state.successful,
            self.provider.provider_id,
        )
        dt = jnp.asarray(self.step_size, dtype=state.positions.dtype)
        force_scale = self.system.plan.units.force_to_momentum_rate
        half = state.momenta + 0.5 * dt * force_scale * state.forces
        position = state.positions + dt * half * self.system.inverse_masses[:, None]
        cell = None if self.system.cell is None else self.system.cell.vectors
        final = self.provider.evaluate(self.system, position, cell)
        momentum = half + 0.5 * dt * force_scale * final.forces
        successful = state.successful & final.successful & jnp.all(jnp.isfinite(momentum))
        successor = BornOppenheimerState(
            position,
            momentum,
            final.forces,
            final.energy,
            state.time + dt,
            state.step_index + 1,
            successful,
            self.provider.provider_id,
        )
        return BornOppenheimerStep(
            tree_where(successful, successor, state), initial, final, successful
        )


__all__ = [
    "BornOppenheimerState",
    "BornOppenheimerStep",
    "BornOppenheimerVelocityVerletPlan",
    "CallableBornOppenheimerProvider",
]
