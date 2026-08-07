#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import AbstractDistribution, Normal
from flowjax.flows import coupling_flow, triangular_spline_flow
from flowjax.train import fit_to_data
from jaxtyping import Array


class _ReplayBuffer(NamedTuple):
    values: Array
    size: Array
    seen: Array


class _FlowProposalState(NamedTuple):
    position: Array
    log_target: Array
    log_proposal: Array


class _FlowProposalBlockInfo(NamedTuple):
    accepted: Array
    log_acceptance_ratio: Array
    nonfinite: Array
    proposed_log_target: Array
    proposed_log_density: Array


def _initialize_replay(
    *,
    num_chains: int,
    capacity_per_chain: int,
    dimension: int,
    dtype: Any,
) -> _ReplayBuffer:
    return _ReplayBuffer(
        values=jnp.zeros(
            (int(num_chains), int(capacity_per_chain), int(dimension)),
            dtype=dtype,
        ),
        size=jnp.zeros((int(num_chains),), dtype=jnp.int32),
        seen=jnp.zeros((int(num_chains),), dtype=jnp.int32),
    )


def _update_replay(
    replay: _ReplayBuffer,
    samples: Array,
    keys: Array,
) -> _ReplayBuffer:
    """Apply independent, fixed-capacity reservoir updates for every chain."""
    values = jnp.asarray(samples)
    if values.ndim != 3:
        raise ValueError("Replay samples must have shape (chain, sample, dimension).")
    if values.shape[0] != replay.values.shape[0]:
        raise ValueError("Replay samples have an incompatible chain count.")
    if values.shape[2] != replay.values.shape[2]:
        raise ValueError("Replay samples have an incompatible event dimension.")
    if keys.shape[:2] != values.shape[:2]:
        raise ValueError("Replay keys must match the chain and sample axes.")

    capacity = int(replay.values.shape[1])

    def update_chain(chain_values, chain_size, chain_seen, chain_samples, chain_keys):
        def update_one(carry, item):
            current_values, current_size, current_seen = carry
            sample, key = item
            next_seen = current_seen + jnp.asarray(1, dtype=jnp.int32)
            candidate = jr.randint(
                key,
                (),
                minval=0,
                maxval=next_seen,
                dtype=jnp.int32,
            )
            filling = current_size < capacity
            replace = filling | (candidate < capacity)
            index = jnp.where(
                filling,
                current_size,
                jnp.minimum(candidate, capacity - 1),
            )
            updated_values = jax.lax.cond(
                replace,
                lambda data: data.at[index].set(sample),
                lambda data: data,
                current_values,
            )
            next_size = jnp.minimum(current_size + 1, capacity)
            return (updated_values, next_size, next_seen), None

        (final_values, final_size, final_seen), _ = jax.lax.scan(
            update_one,
            (chain_values, chain_size, chain_seen),
            (chain_samples, chain_keys),
        )
        return final_values, final_size, final_seen

    updated_values, updated_size, updated_seen = jax.vmap(update_chain)(
        replay.values,
        replay.size,
        replay.seen,
        values,
        keys,
    )
    return _ReplayBuffer(updated_values, updated_size, updated_seen)


def _replay_data(replay: _ReplayBuffer) -> Array:
    if not bool(jnp.all(replay.size == replay.size[0])):
        raise RuntimeError("Chain-stratified replay sizes must remain equal.")
    size = int(replay.size[0])
    if size <= 0:
        raise ValueError("Flow training requires at least one replay sample per chain.")
    return replay.values[:, :size, :].reshape((-1, replay.values.shape[-1]))


def _build_default_flow(
    key: Array,
    data: Array,
    /,
    *,
    flow_layers: int,
    num_knots: int,
    nn_width: int,
    nn_depth: int,
) -> AbstractDistribution:
    samples = jnp.asarray(data)
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 1:
        raise ValueError("Flow initialization requires a non-empty matrix of samples.")
    if not jnp.issubdtype(samples.dtype, jnp.floating):
        raise TypeError("Flow samples must use a real floating dtype.")

    location = jnp.mean(samples, axis=0)
    scale = jnp.std(samples, axis=0)
    tolerance = jnp.sqrt(jnp.finfo(samples.dtype).eps) * jnp.maximum(
        jnp.ones_like(location), jnp.abs(location)
    )
    base = Normal(location, jnp.maximum(scale, tolerance))
    dimension = int(samples.shape[1])
    if dimension == 1:
        return triangular_spline_flow(
            key,
            base_dist=base,
            flow_layers=int(flow_layers),
            knots=int(num_knots),
            invert=True,
        )

    transformer = RationalQuadraticSpline(knots=int(num_knots), interval=3.0)
    return coupling_flow(
        key,
        base_dist=base,
        transformer=transformer,
        flow_layers=int(flow_layers),
        nn_width=int(nn_width),
        nn_depth=int(nn_depth),
        invert=True,
    )


def _fit_flow(
    key: Array,
    flow: AbstractDistribution,
    data: Array,
    /,
    *,
    learning_rate: float,
    max_epochs: int,
    max_patience: int,
    batch_size: int,
    validation_fraction: float,
) -> tuple[AbstractDistribution, Array, Array]:
    samples = jnp.asarray(data)
    validation_count = round(float(validation_fraction) * int(samples.shape[0]))
    training_count = int(samples.shape[0]) - validation_count
    if validation_count <= 0 or validation_count >= int(samples.shape[0]):
        raise ValueError(
            "Flow training data must produce non-empty train and validation splits."
        )
    trained, losses = fit_to_data(
        key,
        flow,
        samples,
        learning_rate=float(learning_rate),
        max_epochs=int(max_epochs),
        max_patience=int(max_patience),
        batch_size=min(int(batch_size), training_count),
        val_prop=float(validation_fraction),
        return_best=True,
        show_progress=False,
    )
    training_loss = jnp.asarray(losses["train"])
    validation_loss = jnp.asarray(losses["val"])
    if not bool(jnp.all(jnp.isfinite(training_loss))) or not bool(
        jnp.all(jnp.isfinite(validation_loss))
    ):
        raise FloatingPointError("Flow training produced a nonfinite loss.")
    return trained, training_loss, validation_loss


def _validate_flow(
    flow: AbstractDistribution,
    data: Array,
    key: Array,
    /,
) -> None:
    samples = jnp.asarray(data)
    expected_shape = (int(samples.shape[-1]),)
    if flow.shape != expected_shape:
        raise ValueError(
            f"Flow event shape {flow.shape} does not match {expected_shape}."
        )
    if flow.cond_shape is not None:
        raise ValueError("Flow proposal must be unconditional.")
    data_log_density = flow.log_prob(samples)
    proposed, proposed_log_density = flow.sample_and_log_prob(
        key,
        sample_shape=(min(8, int(samples.shape[0])),),
    )
    if proposed.shape[1:] != expected_shape:
        raise ValueError("Flow proposal samples have an incompatible event shape.")
    if not bool(jnp.all(jnp.isfinite(data_log_density))):
        raise FloatingPointError("Flow density is nonfinite on its training data.")
    if not bool(jnp.all(jnp.isfinite(proposed))) or not bool(
        jnp.all(jnp.isfinite(proposed_log_density))
    ):
        raise FloatingPointError("Flow proposal produced a nonfinite sample or density.")


def _independence_mh_scan(
    initial_state: _FlowProposalState,
    proposed_positions: Array,
    proposed_log_targets: Array,
    proposed_log_densities: Array,
    log_uniforms: Array,
) -> tuple[_FlowProposalState, _FlowProposalBlockInfo]:
    """Apply exact sequential independence-MH decisions to prepared proposals."""

    def transition(state, proposal):
        position, log_target, log_proposal, log_uniform = proposal
        finite = (
            jnp.all(jnp.isfinite(position))
            & jnp.isfinite(log_target)
            & jnp.isfinite(log_proposal)
            & jnp.isfinite(state.log_target)
            & jnp.isfinite(state.log_proposal)
        )
        raw_ratio = log_target - state.log_target + state.log_proposal - log_proposal
        log_acceptance_ratio = jnp.where(
            finite,
            jnp.minimum(jnp.zeros((), dtype=raw_ratio.dtype), raw_ratio),
            -jnp.inf,
        )
        accepted = finite & (log_uniform < log_acceptance_ratio)
        next_state = _FlowProposalState(
            position=jnp.where(accepted, position, state.position),
            log_target=jnp.where(accepted, log_target, state.log_target),
            log_proposal=jnp.where(accepted, log_proposal, state.log_proposal),
        )
        return next_state, (accepted, log_acceptance_ratio, ~finite)

    final_state, (accepted, log_acceptance_ratio, nonfinite) = jax.lax.scan(
        transition,
        initial_state,
        (
            proposed_positions,
            proposed_log_targets,
            proposed_log_densities,
            log_uniforms,
        ),
    )
    return final_state, _FlowProposalBlockInfo(
        accepted=accepted,
        log_acceptance_ratio=log_acceptance_ratio,
        nonfinite=nonfinite,
        proposed_log_target=proposed_log_targets,
        proposed_log_density=proposed_log_densities,
    )


def _run_flow_block(
    key: Array,
    position: Array,
    log_target: Array,
    flow: AbstractDistribution,
    logdensity_fn: Any,
    /,
    *,
    num_steps: int,
) -> tuple[_FlowProposalState, _FlowProposalBlockInfo]:
    proposal_key, acceptance_key = jr.split(key)
    proposed_positions, proposed_log_densities = flow.sample_and_log_prob(
        proposal_key,
        sample_shape=(int(num_steps),),
    )
    proposed_log_targets = jax.vmap(logdensity_fn)(proposed_positions)
    current_log_proposal = flow.log_prob(position)
    tiny = jnp.finfo(proposed_positions.dtype).tiny
    log_uniforms = jnp.log(
        jr.uniform(
            acceptance_key,
            (int(num_steps),),
            minval=tiny,
            maxval=1.0,
            dtype=proposed_positions.dtype,
        )
    )
    return _independence_mh_scan(
        _FlowProposalState(position, log_target, current_log_proposal),
        proposed_positions,
        proposed_log_targets,
        proposed_log_densities,
        log_uniforms,
    )


def _proposal_effective_sample_size(
    proposed_log_target: Array,
    proposed_log_density: Array,
) -> Array:
    log_weights = jnp.ravel(proposed_log_target - proposed_log_density)
    finite = jnp.isfinite(log_weights)

    def finite_ess(values):
        masked = jnp.where(finite, values, -jnp.inf)
        weights = jax.nn.softmax(masked)
        return jnp.reciprocal(jnp.sum(weights**2))

    return jax.lax.cond(
        jnp.any(finite),
        finite_ess,
        lambda values: jnp.zeros((), dtype=values.dtype),
        log_weights,
    )


__all__: list[str] = []
