#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from .._strict import StrictModule
from ._addressing import derive_key, SampleAddress
from ._chain import AbstractChainSampleResult
from ._proposals import AbstractProposal


_PROPOSAL_ADDRESS = SampleAddress(
    "markov",
    "metropolis-hastings",
    target="proposal",
    role="transition",
)
_ACCEPTANCE_ADDRESS = SampleAddress(
    "markov",
    "metropolis-hastings",
    target="acceptance",
    role="transition",
)


def _tree_all_finite(tree: PyTree[Any], /) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("A Markov position must contain at least one array leaf.")
    return jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves])
    )


def _chain_count(positions: PyTree[Any], /) -> int:
    leaves = jax.tree_util.tree_leaves(positions)
    if not leaves:
        raise ValueError("Initial positions must contain at least one array leaf.")
    arrays = tuple(jnp.asarray(leaf) for leaf in leaves)
    if any(array.ndim < 1 for array in arrays):
        raise ValueError("Every initial-position leaf needs a leading chain axis.")
    count = int(arrays[0].shape[0])
    if count < 1 or any(int(array.shape[0]) != count for array in arrays[1:]):
        raise ValueError(
            "Every initial-position leaf must share one nonempty chain axis."
        )
    return count


def _real_scalar(value: Any, /, *, role: str) -> Array:
    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{role} must return one scalar; got shape {array.shape}.")
    if jnp.iscomplexobj(array):
        raise TypeError(f"{role} must be real-valued.")
    return array.astype(jnp.result_type(array, float))


def _tree_select(predicate: Array, accepted: PyTree[Any], rejected: PyTree[Any]):
    return jax.tree_util.tree_map(
        lambda proposed, current: jnp.where(predicate, proposed, current),
        accepted,
        rejected,
    )


def _swap_draw_chain(tree: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree_util.tree_map(lambda value: jnp.swapaxes(value, 0, 1), tree)


class MarkovState(StrictModule):
    """Persistent chain positions and their current target values."""

    position: PyTree[Array]
    log_target: Array
    valid: Array
    step_index: Array

    def __init__(
        self,
        position: PyTree[Any],
        log_target: Array,
        /,
        *,
        valid: Array | None = None,
        step_index: Array | int = 0,
    ):
        count = _chain_count(position)
        positions = jax.tree_util.tree_map(jnp.asarray, position)
        values = jnp.asarray(log_target)
        if values.shape != (count,):
            raise ValueError(
                f"log_target must have shape ({count},); got {values.shape}."
            )
        if jnp.iscomplexobj(values):
            raise TypeError("Markov log targets must be real-valued.")
        validity = (
            jnp.isfinite(values) if valid is None else jnp.asarray(valid, dtype=bool)
        )
        if validity.shape != (count,):
            raise ValueError(f"valid must have shape ({count},); got {validity.shape}.")
        index = jnp.asarray(step_index, dtype=jnp.uint32)
        if index.shape != ():
            raise ValueError("step_index must be scalar.")
        self.position = positions
        self.log_target = values
        self.valid = validity
        self.step_index = index

    @property
    def num_chains(self) -> int:
        return int(self.log_target.shape[0])


class MarkovTransitionInfo(StrictModule):
    """Per-chain evidence from one Metropolis--Hastings transition."""

    accepted: Array
    log_acceptance_ratio: Array
    proposal_valid: Array
    target_valid: Array


class MarkovSampleResult(AbstractChainSampleResult):
    """Chain-preserving Markov draws and complete transition evidence."""

    samples: PyTree[Array]
    log_target: Array
    accepted: Array
    log_acceptance_ratio: Array
    proposal_valid: Array
    target_valid: Array
    final_state: MarkovState
    root_key: Array
    kernel_id: str = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)
    warmup_steps: int = eqx.field(static=True)
    steps_per_draw: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        samples: PyTree[Array],
        log_target: Array,
        accepted: Array,
        log_acceptance_ratio: Array,
        proposal_valid: Array,
        target_valid: Array,
        final_state: MarkovState,
        root_key: Array,
        kernel_id: str,
        proposal_id: str,
        warmup_steps: int,
        steps_per_draw: int,
    ):
        sample_leaves = jax.tree_util.tree_leaves(samples)
        if not sample_leaves:
            raise ValueError("Markov samples must contain at least one array leaf.")
        values = jnp.asarray(log_target)
        if values.ndim != 2:
            raise ValueError("log_target must have leading chain and draw axes.")
        chains, draws = (int(size) for size in values.shape)
        for leaf in sample_leaves:
            if jnp.asarray(leaf).shape[:2] != (chains, draws):
                raise ValueError("Every sample leaf must share chain and draw axes.")
        evidence_shape = (chains, draws, int(steps_per_draw))
        evidence = (
            accepted,
            log_acceptance_ratio,
            proposal_valid,
            target_valid,
        )
        if any(jnp.asarray(value).shape != evidence_shape for value in evidence):
            raise ValueError(
                "Transition evidence must have shape "
                f"{evidence_shape} (chain, draw, transition)."
            )
        if final_state.num_chains != chains:
            raise ValueError("final_state chain count must match samples.")
        if not isinstance(kernel_id, str) or not kernel_id:
            raise ValueError("kernel_id must be non-empty.")
        if not isinstance(proposal_id, str) or not proposal_id:
            raise ValueError("proposal_id must be non-empty.")
        self.samples = samples
        self.log_target = values
        self.accepted = jnp.asarray(accepted, dtype=bool)
        self.log_acceptance_ratio = jnp.asarray(log_acceptance_ratio)
        self.proposal_valid = jnp.asarray(proposal_valid, dtype=bool)
        self.target_valid = jnp.asarray(target_valid, dtype=bool)
        self.final_state = final_state
        self.root_key = jnp.asarray(root_key)
        self.kernel_id = kernel_id
        self.proposal_id = proposal_id
        self.warmup_steps = int(warmup_steps)
        self.steps_per_draw = int(steps_per_draw)

    @property
    def num_chains(self) -> int:
        return int(self.log_target.shape[0])

    @property
    def num_draws(self) -> int:
        return int(self.log_target.shape[1])

    @property
    def chain_provenance(self) -> str:
        return f"markov:{self.kernel_id}:{self.proposal_id}"

    @property
    def acceptance_rate(self) -> Array:
        return jnp.mean(self.accepted.astype(float), axis=(1, 2))


class MetropolisHastings(StrictModule):
    """Fixed-kernel Metropolis--Hastings over an explicit normalized proposal."""

    proposal: AbstractProposal
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        proposal: AbstractProposal,
        /,
        *,
        kernel_id: str = "metropolis-hastings",
    ):
        if not isinstance(proposal, AbstractProposal):
            raise TypeError("proposal must implement AbstractProposal.")
        if not isinstance(kernel_id, str) or not kernel_id:
            raise ValueError("kernel_id must be non-empty.")
        self.proposal = proposal
        self.kernel_id = kernel_id

    def initialize(
        self,
        log_target: Callable[[PyTree[Any]], Any],
        initial_positions: PyTree[Any],
        /,
    ) -> MarkovState:
        if not callable(log_target):
            raise TypeError("log_target must be callable.")
        _chain_count(initial_positions)
        positions = jax.tree_util.tree_map(jnp.asarray, initial_positions)
        values = jax.vmap(
            lambda position: _real_scalar(log_target(position), role="log_target")
        )(positions)
        valid = jnp.isfinite(values) & jax.vmap(_tree_all_finite)(positions)
        values = eqx.error_if(
            values,
            ~jnp.all(valid),
            "Initial Markov positions must have finite positions and log targets.",
        )
        return MarkovState(positions, values, valid=valid)

    def refresh(
        self,
        log_target: Callable[[PyTree[Any]], Any],
        state: MarkovState,
        /,
    ) -> MarkovState:
        if not isinstance(state, MarkovState):
            raise TypeError("state must be a MarkovState.")
        values = jax.vmap(
            lambda position: _real_scalar(log_target(position), role="log_target")
        )(state.position)
        valid = jnp.isfinite(values) & jax.vmap(_tree_all_finite)(state.position)
        values = eqx.error_if(
            values,
            ~jnp.all(valid),
            "Refreshed Markov positions must have finite log targets.",
        )
        return MarkovState(
            state.position,
            values,
            valid=valid,
            step_index=state.step_index,
        )

    def step(
        self,
        log_target: Callable[[PyTree[Any]], Any],
        state: MarkovState,
        key: Key[Array, ""],
        /,
    ) -> tuple[MarkovState, MarkovTransitionInfo]:
        if not isinstance(state, MarkovState):
            raise TypeError("state must be a MarkovState.")
        chain_indices = jnp.arange(state.num_chains, dtype=jnp.uint32)
        proposal_keys = jax.vmap(
            lambda chain: derive_key(key, _PROPOSAL_ADDRESS, chain, state.step_index)
        )(chain_indices)
        acceptance_keys = jax.vmap(
            lambda chain: derive_key(key, _ACCEPTANCE_ADDRESS, chain, state.step_index)
        )(chain_indices)

        def one_step(current, current_log_target, proposal_key, acceptance_key):
            proposed = self.proposal.sample(proposal_key, current)
            proposed_log_target = _real_scalar(log_target(proposed), role="log_target")
            forward = _real_scalar(
                self.proposal.log_prob(proposed, current),
                role="proposal.log_prob",
            )
            reverse = _real_scalar(
                self.proposal.log_prob(current, proposed),
                role="proposal.log_prob",
            )
            position_valid = _tree_all_finite(proposed)
            target_valid = ~jnp.isnan(proposed_log_target) & ~jnp.isposinf(
                proposed_log_target
            )
            proposal_valid = (
                position_valid & jnp.isfinite(forward) & jnp.isfinite(reverse)
            )
            valid = target_valid & proposal_valid
            raw_ratio = proposed_log_target - current_log_target + reverse - forward
            log_ratio = jnp.where(valid, raw_ratio, -jnp.inf)
            log_uniform = jnp.log(jax.random.uniform(acceptance_key))
            accepted = valid & (log_uniform < jnp.minimum(log_ratio, 0.0))
            next_position = _tree_select(accepted, proposed, current)
            next_log_target = jnp.where(accepted, proposed_log_target, current_log_target)
            return (
                next_position,
                next_log_target,
                MarkovTransitionInfo(
                    accepted=accepted,
                    log_acceptance_ratio=log_ratio,
                    proposal_valid=proposal_valid,
                    target_valid=target_valid,
                ),
            )

        positions, values, info = jax.vmap(one_step)(
            state.position,
            state.log_target,
            proposal_keys,
            acceptance_keys,
        )
        next_state = MarkovState(
            positions,
            values,
            valid=jnp.isfinite(values),
            step_index=state.step_index + jnp.asarray(1, dtype=jnp.uint32),
        )
        return next_state, info


def sample_markov(
    log_target: Callable[[PyTree[Any]], Any],
    kernel: MetropolisHastings,
    state: MarkovState,
    /,
    *,
    key: Key[Array, ""],
    num_draws: int,
    steps_per_draw: int = 1,
    warmup_steps: int = 0,
) -> MarkovSampleResult:
    """Advance persistent chains and retain chain-by-draw samples."""
    if not callable(log_target):
        raise TypeError("log_target must be callable.")
    if not isinstance(kernel, MetropolisHastings):
        raise TypeError("kernel must be a MetropolisHastings instance.")
    if not isinstance(state, MarkovState):
        raise TypeError("state must be a MarkovState.")
    draws = int(num_draws)
    transitions = int(steps_per_draw)
    warmup = int(warmup_steps)
    if draws <= 0:
        raise ValueError("num_draws must be positive.")
    if transitions <= 0:
        raise ValueError("steps_per_draw must be positive.")
    if warmup < 0:
        raise ValueError("warmup_steps must be non-negative.")

    def discard_step(carry, _):
        next_state, _info = kernel.step(log_target, carry, key)
        return next_state, None

    warmed, _ = jax.lax.scan(discard_step, state, xs=None, length=warmup)

    def collect_draw(carry, _):
        def transition_step(inner, __):
            next_state, info = kernel.step(log_target, inner, key)
            return next_state, info

        next_state, infos = jax.lax.scan(
            transition_step,
            carry,
            xs=None,
            length=transitions,
        )
        output = (
            next_state.position,
            next_state.log_target,
            infos.accepted,
            infos.log_acceptance_ratio,
            infos.proposal_valid,
            infos.target_valid,
        )
        return next_state, output

    final_state, outputs = jax.lax.scan(
        collect_draw,
        warmed,
        xs=None,
        length=draws,
    )
    samples, values, accepted, ratios, proposal_valid, target_valid = outputs
    return MarkovSampleResult(
        samples=_swap_draw_chain(samples),
        log_target=jnp.swapaxes(values, 0, 1),
        accepted=jnp.transpose(accepted, (2, 0, 1)),
        log_acceptance_ratio=jnp.transpose(ratios, (2, 0, 1)),
        proposal_valid=jnp.transpose(proposal_valid, (2, 0, 1)),
        target_valid=jnp.transpose(target_valid, (2, 0, 1)),
        final_state=final_state,
        root_key=key,
        kernel_id=kernel.kernel_id,
        proposal_id=kernel.proposal.proposal_id,
        warmup_steps=warmup,
        steps_per_draw=transitions,
    )


__all__ = [
    "MarkovSampleResult",
    "MarkovState",
    "MarkovTransitionInfo",
    "MetropolisHastings",
    "sample_markov",
]
