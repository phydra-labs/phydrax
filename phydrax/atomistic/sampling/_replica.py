#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ReplicaExchangeKind(StrEnum):
    TEMPERATURE = "temperature"
    HAMILTONIAN = "hamiltonian"
    LAMBDA = "lambda"
    UMBRELLA = "umbrella"


class AtomisticReplicaEnsemblePlan(StrictModule, NonTrainableState):
    temperatures: Array
    lambda_values: Array
    kind: ReplicaExchangeKind = eqx.field(static=True)
    exchange_interval: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperatures: ArrayLike,
        /,
        *,
        lambda_values: ArrayLike | None = None,
        kind: ReplicaExchangeKind = ReplicaExchangeKind.TEMPERATURE,
        exchange_interval: int = 100,
    ):
        temperature = jnp.asarray(temperatures, dtype=float).reshape((-1,))
        lambdas = (
            jnp.zeros_like(temperature)
            if lambda_values is None
            else jnp.asarray(lambda_values, dtype=float).reshape((-1,))
        )
        if (
            temperature.size < 2
            or lambdas.shape != temperature.shape
            or bool(jnp.any(~jnp.isfinite(temperature) | (temperature <= 0.0)))
            or bool(jnp.any(~jnp.isfinite(lambdas)))
            or not isinstance(kind, ReplicaExchangeKind)
            or int(exchange_interval) <= 0
        ):
            raise ValueError(
                "Replica temperatures, lambdas, kind, or exchange interval are invalid."
            )
        self.temperatures = temperature
        self.lambda_values = lambdas
        self.kind = kind
        self.exchange_interval = int(exchange_interval)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-replica-ensemble",
                "temperatures": np.asarray(temperature).tolist(),
                "lambdas": np.asarray(lambdas).tolist(),
                "exchange_kind": kind.value,
                "interval": self.exchange_interval,
            }
        )

    def should_exchange(self, dynamics_step: ArrayLike, /) -> Array:
        step = jnp.asarray(dynamics_step, dtype=jnp.int64)
        return (step > 0) & (step % self.exchange_interval == 0)


class AtomisticReplicaState(StrictModule):
    positions: Array
    momenta: Array
    energy_matrix: Array
    label_at_slot: Array
    accepted_swaps: Array
    attempted_swaps: Array
    step_index: Array
    exchange_parity: Array
    root_key: Array
    plan_id: str = eqx.field(static=True)


class AtomisticReplicaExchangeEvaluation(StrictModule):
    state: AtomisticReplicaState
    attempted_pairs: Array
    accepted_pairs: Array
    log_acceptance: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class AtomisticReplicaReducer(StrictModule, NonTrainableState):
    def initialize(self, replica_count: int, dtype=float):
        return (
            jnp.zeros((replica_count,), dtype=jnp.int32),
            jnp.zeros((replica_count,), dtype=dtype),
            jnp.zeros((replica_count,), dtype=dtype),
        )

    def update(self, state, values):
        count, first, second = state
        return count + 1, first + values, second + values**2

    def finalize(self, state):
        count, first, second = state
        denominator = jnp.maximum(count, 1)
        mean = first / denominator
        return {
            "count": count,
            "mean": mean,
            "variance": jnp.maximum(second / denominator - mean**2, 0.0),
        }


def initialize_replica_state(
    plan: AtomisticReplicaEnsemblePlan,
    positions: ArrayLike,
    momenta: ArrayLike,
    energy_matrix: ArrayLike,
    root_key,
    /,
) -> AtomisticReplicaState:
    position = jnp.asarray(positions)
    momentum = jnp.asarray(momenta, dtype=position.dtype)
    energy = jnp.asarray(energy_matrix, dtype=position.dtype)
    replica_count = plan.temperatures.size
    if (
        position.shape[0] != replica_count
        or momentum.shape != position.shape
        or energy.shape != (replica_count, replica_count)
        or not bool(jnp.all(jnp.isfinite(position)))
        or not bool(jnp.all(jnp.isfinite(momentum)))
        or not bool(jnp.all(jnp.isfinite(energy)))
    ):
        raise ValueError("Replica state arrays have incompatible or nonfinite values.")
    return AtomisticReplicaState(
        position,
        momentum,
        energy,
        jnp.arange(replica_count, dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
        jr.key_data(root_key),
        plan.plan_id,
    )


def replica_exchange_step(
    plan: AtomisticReplicaEnsemblePlan,
    state: AtomisticReplicaState,
    boltzmann_constant: float,
    /,
) -> AtomisticReplicaExchangeEvaluation:
    if state.plan_id != plan.plan_id:
        raise ValueError("Replica state belongs to another ensemble plan.")
    if not np.isfinite(boltzmann_constant) or float(boltzmann_constant) <= 0.0:
        raise ValueError("Replica exchange requires a positive Boltzmann constant.")
    replica_count = int(plan.temperatures.size)
    parity = state.exchange_parity % 2
    starts = jnp.arange(replica_count - 1, dtype=jnp.int32)
    pair_valid = starts % 2 == parity
    labels = state.label_at_slot
    left_label = labels[:-1]
    right_label = labels[1:]
    left_slot = starts
    right_slot = starts + 1
    beta = 1.0 / (boltzmann_constant * plan.temperatures)
    if plan.kind is ReplicaExchangeKind.TEMPERATURE:
        left_energy = state.energy_matrix[left_label, left_slot]
        right_energy = state.energy_matrix[right_label, right_slot]
        swapped_left = state.energy_matrix[left_label, right_slot]
        swapped_right = state.energy_matrix[right_label, left_slot]
        log_acceptance = -(
            beta[left_label] * swapped_left
            + beta[right_label] * swapped_right
            - beta[left_label] * left_energy
            - beta[right_label] * right_energy
        )
    else:
        current = (
            state.energy_matrix[left_label, left_slot]
            + state.energy_matrix[right_label, right_slot]
        )
        swapped = (
            state.energy_matrix[left_label, right_slot]
            + state.energy_matrix[right_label, left_slot]
        )
        log_acceptance = -(swapped - current)
    base_key = jr.wrap_key_data(state.root_key)
    keys = jax.vmap(
        lambda index: jr.fold_in(jr.fold_in(base_key, state.step_index), index)
    )(starts)
    uniforms = jax.vmap(lambda key: jr.uniform(key, ()))(keys)
    accepted = pair_valid & (jnp.log(uniforms) < jnp.minimum(log_acceptance, 0.0))
    new_labels = labels
    for index in range(replica_count - 1):
        accept = accepted[index]
        left_value, right_value = new_labels[index], new_labels[index + 1]
        new_labels = new_labels.at[index].set(jnp.where(accept, right_value, left_value))
        new_labels = new_labels.at[index + 1].set(
            jnp.where(accept, left_value, right_value)
        )
    momentum = state.momenta
    if plan.kind is ReplicaExchangeKind.TEMPERATURE:
        scales = jnp.sqrt(plan.temperatures[new_labels] / plan.temperatures[labels])
        momentum = momentum * scales.reshape(
            (replica_count,) + (1,) * (momentum.ndim - 1)
        )
    successor = AtomisticReplicaState(
        state.positions,
        momentum,
        state.energy_matrix,
        new_labels,
        state.accepted_swaps + jnp.sum(accepted, dtype=jnp.int32),
        state.attempted_swaps + jnp.sum(pair_valid, dtype=jnp.int32),
        state.step_index + 1,
        1 - parity,
        state.root_key,
        state.plan_id,
    )
    return AtomisticReplicaExchangeEvaluation(
        successor,
        jnp.stack((left_slot, right_slot), axis=-1),
        accepted,
        log_acceptance,
        jnp.all(jnp.isfinite(log_acceptance)),
        plan.plan_id,
    )


def reduced_potential_samples(
    energy_matrix: ArrayLike,
    temperatures: ArrayLike,
    origin_states: ArrayLike,
    boltzmann_constant: float,
    /,
    *,
    source_id: str = "atomistic-reduced-potential",
):
    from ...uq._free_energy import ReducedPotentialSamples

    energy = jnp.asarray(energy_matrix, dtype=float)
    temperature = jnp.asarray(temperatures, dtype=float).reshape((-1,))
    origin = jnp.asarray(origin_states, dtype=jnp.int32).reshape((-1,))
    if (
        energy.shape[0] != temperature.size
        or energy.shape[1] != origin.size
        or not np.isfinite(boltzmann_constant)
        or float(boltzmann_constant) <= 0.0
        or bool(jnp.any(~jnp.isfinite(temperature) | (temperature <= 0.0)))
    ):
        raise ValueError(
            "Reduced-potential energies, temperatures, or units are invalid."
        )
    counts = jnp.bincount(origin, length=temperature.size)
    reduced = energy / (boltzmann_constant * temperature[:, None])
    return ReducedPotentialSamples(reduced, counts, origin, source_id=source_id)


__all__ = [
    "AtomisticReplicaEnsemblePlan",
    "AtomisticReplicaExchangeEvaluation",
    "AtomisticReplicaReducer",
    "AtomisticReplicaState",
    "ReplicaExchangeKind",
    "initialize_replica_state",
    "reduced_potential_samples",
    "replica_exchange_step",
]
