#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from .._execution_array import shard_tree_axis
from .._execution_runtime import ExecutionGroup
from .._iteration import (
    bind_iteration_scope,
    finalize_iteration,
    initialize_iteration,
    IterationCapabilities,
    IterationCoordinates,
    IterationEvidence,
    IterationPhase,
    IterationPlan,
    IterationRecord,
    update_iteration,
)
from .._strict import StrictModule
from ._addressing import derive_key, SampleAddress
from ._chain import AbstractChainSampleResult
from ._proposals import AbstractProposal
from ._targets import FullMarkovTarget, IncrementalMarkovTarget, MarkovTargetState


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

_RAW_TARGET_ID = "raw-callable"


def _resolve_target(target, /):
    if isinstance(target, (FullMarkovTarget, IncrementalMarkovTarget)):
        return target
    if callable(target):
        return FullMarkovTarget(target, target_id=_RAW_TARGET_ID)
    raise TypeError(
        "target must be callable, FullMarkovTarget, or IncrementalMarkovTarget."
    )


def _bool_scalar(value: Any, /, *, role: str) -> Array:
    array = jnp.asarray(value, dtype=bool)
    if array.shape != ():
        raise ValueError(f"{role} must return one scalar; got shape {array.shape}.")
    return array


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


def _swap_draw_chain(tree: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree_util.tree_map(lambda value: jnp.swapaxes(value, 0, 1), tree)


class MarkovState(StrictModule):
    """Persistent chain positions and their current target caches."""

    position: PyTree[Array]
    log_target: Array
    cache: PyTree[Array]
    valid: Array
    step_index: Array
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        position: PyTree[Any],
        log_target: Array,
        /,
        *,
        cache: PyTree[Any] = (),
        valid: Array | None = None,
        step_index: Array | int = 0,
        target_id: str = _RAW_TARGET_ID,
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
        caches = jax.tree_util.tree_map(jnp.asarray, cache)
        cache_leaves = jax.tree_util.tree_leaves(caches)
        if any(leaf.ndim < 1 or int(leaf.shape[0]) != count for leaf in cache_leaves):
            raise ValueError("Every target-cache leaf must share the chain axis.")
        declared_valid = (
            jnp.ones((count,), dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        if declared_valid.shape != (count,):
            raise ValueError(
                f"valid must have shape ({count},); got {declared_valid.shape}."
            )
        cache_finite = jnp.ones((count,), dtype=bool)
        for leaf in cache_leaves:
            axes = tuple(range(1, leaf.ndim))
            finite = jnp.isfinite(leaf)
            cache_finite = cache_finite & (jnp.all(finite, axis=axes) if axes else finite)
        validity = (
            declared_valid
            & jnp.isfinite(values)
            & jax.vmap(_tree_all_finite)(positions)
            & cache_finite
        )
        index = jnp.asarray(step_index, dtype=jnp.uint32)
        if index.shape != ():
            raise ValueError("step_index must be scalar.")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("target_id must be non-empty.")
        self.position = positions
        self.log_target = values
        self.cache = caches
        self.valid = validity
        self.step_index = index
        self.target_id = target_id

    @property
    def num_chains(self) -> int:
        return int(self.log_target.shape[0])


class MarkovTransitionInfo(StrictModule):
    """Per-chain evidence from one Metropolis--Hastings transition."""

    accepted: Array
    log_acceptance_ratio: Array
    proposal_valid: Array
    target_valid: Array


class MarkovIterationMetrics(StrictModule):
    """Per-chain transition evidence for iteration observers."""

    accepted: Array
    log_acceptance_ratio: Array
    proposal_valid: Array
    target_valid: Array
    log_target: Array
    warmup: Array
    draw_index: Array

    def __init__(
        self,
        info: MarkovTransitionInfo,
        log_target,
        /,
        *,
        warmup,
        draw_index,
    ):
        self.accepted = jnp.asarray(info.accepted, dtype=bool)
        self.log_acceptance_ratio = jnp.asarray(info.log_acceptance_ratio)
        self.proposal_valid = jnp.asarray(info.proposal_valid, dtype=bool)
        self.target_valid = jnp.asarray(info.target_valid, dtype=bool)
        self.log_target = jnp.asarray(log_target)
        self.warmup = jnp.asarray(warmup, dtype=bool)
        self.draw_index = jnp.asarray(draw_index, dtype=jnp.int32)


def _markov_iteration_record(
    state: MarkovState,
    info: MarkovTransitionInfo,
    phase,
    /,
    *,
    warmup,
    draw_index,
    active,
    terminal=False,
) -> IterationRecord:
    return IterationRecord(
        IterationCoordinates(
            phase,
            state.step_index.astype(jnp.int32),
            invocation=draw_index,
            attempt=state.step_index.astype(jnp.int32),
            accepted=state.step_index.astype(jnp.int32),
            active=active,
            committed=active,
            terminal=terminal,
        ),
        jnp.where(state.valid, 0, 1).astype(jnp.int32),
        MarkovIterationMetrics(
            info,
            state.log_target,
            warmup=warmup,
            draw_index=draw_index,
        ),
    )


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
    target_id: str = eqx.field(static=True)
    warmup_steps: int = eqx.field(static=True)
    steps_per_draw: int = eqx.field(static=True)
    iteration_evidence: IterationEvidence | None

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
        target_id: str,
        warmup_steps: int,
        steps_per_draw: int,
        iteration_evidence: IterationEvidence | None = None,
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
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("target_id must be non-empty.")
        if iteration_evidence is not None and not isinstance(
            iteration_evidence, IterationEvidence
        ):
            raise TypeError("iteration_evidence must be IterationEvidence or None.")
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
        self.target_id = target_id
        self.warmup_steps = int(warmup_steps)
        self.steps_per_draw = int(steps_per_draw)
        self.iteration_evidence = iteration_evidence

    @property
    def num_chains(self) -> int:
        return int(self.log_target.shape[0])

    @property
    def num_draws(self) -> int:
        return int(self.log_target.shape[1])

    @property
    def chain_provenance(self) -> str:
        provenance = f"markov:{self.kernel_id}:{self.proposal_id}"
        if self.target_id == _RAW_TARGET_ID:
            return provenance
        return f"{provenance}:{self.target_id}"

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
        target,
        initial_positions: PyTree[Any],
        /,
        *,
        execution_group: ExecutionGroup | None = None,
    ) -> MarkovState:
        resolved = _resolve_target(target)
        _chain_count(initial_positions)
        positions = jax.tree_util.tree_map(jnp.asarray, initial_positions)
        if execution_group is not None:
            positions = shard_tree_axis(positions, execution_group)
        target_states = jax.vmap(resolved.initialize)(positions)
        valid = target_states.valid & jax.vmap(_tree_all_finite)(positions)
        values = eqx.error_if(
            target_states.log_target,
            ~jnp.all(valid),
            "Initial Markov positions must have finite positions and log targets.",
        )
        return MarkovState(
            positions,
            values,
            cache=target_states.cache,
            valid=valid,
            target_id=resolved.target_id,
        )

    def refresh(
        self,
        target,
        state: MarkovState,
        /,
    ) -> MarkovState:
        resolved = _resolve_target(target)
        if not isinstance(state, MarkovState):
            raise TypeError("state must be a MarkovState.")
        if state.target_id != resolved.target_id:
            raise ValueError("Target identity does not match the Markov state.")
        current = MarkovTargetState(
            position=state.position,
            log_target=state.log_target,
            cache=state.cache,
            valid=state.valid,
        )
        if isinstance(resolved, IncrementalMarkovTarget):
            refreshed = jax.vmap(resolved.refresh)(current)
            values = refreshed.log_target
        else:
            refreshed = jax.vmap(resolved.initialize)(state.position)
            values = eqx.error_if(
                refreshed.log_target,
                ~jnp.all(refreshed.valid),
                "Refreshed Markov positions must have finite log targets.",
            )
        valid = refreshed.valid & jax.vmap(_tree_all_finite)(state.position)
        return MarkovState(
            state.position,
            values,
            cache=refreshed.cache,
            valid=valid,
            step_index=state.step_index,
            target_id=resolved.target_id,
        )

    def step(
        self,
        target,
        state: MarkovState,
        key: Key[Array, ""],
        /,
    ) -> tuple[MarkovState, MarkovTransitionInfo]:
        resolved = _resolve_target(target)
        if not isinstance(state, MarkovState):
            raise TypeError("state must be a MarkovState.")
        if state.target_id != resolved.target_id:
            raise ValueError("Target identity does not match the Markov state.")
        chain_indices = jnp.arange(state.num_chains, dtype=jnp.uint32)
        proposal_keys = jax.vmap(
            lambda chain: derive_key(key, _PROPOSAL_ADDRESS, chain, state.step_index)
        )(chain_indices)
        acceptance_keys = jax.vmap(
            lambda chain: derive_key(key, _ACCEPTANCE_ADDRESS, chain, state.step_index)
        )(chain_indices)

        def one_step(
            current,
            current_log_target,
            current_cache,
            current_valid,
            proposal_key,
            acceptance_key,
        ):
            current_target = MarkovTargetState(
                position=current,
                log_target=current_log_target,
                cache=current_cache,
                valid=current_valid,
            )
            move = self.proposal.propose(proposal_key, current)
            forward = _real_scalar(move.log_forward, role="proposal.log_forward")
            reverse = _real_scalar(move.log_reverse, role="proposal.log_reverse")
            proposal_valid = (
                _bool_scalar(move.valid, role="proposal validity")
                & _tree_all_finite(move.position)
                & jnp.isfinite(forward)
                & jnp.isfinite(reverse)
            )
            target_proposal = resolved.propose(
                current_target, move.position, move.payload
            )
            target_valid = current_valid & _bool_scalar(
                target_proposal.valid, role="target proposal validity"
            )
            valid = target_valid & proposal_valid
            raw_ratio = target_proposal.log_ratio + reverse - forward
            log_ratio = jnp.where(valid & jnp.isfinite(raw_ratio), raw_ratio, -jnp.inf)
            log_uniform = jnp.log(jax.random.uniform(acceptance_key))
            accepted = valid & (log_uniform < jnp.minimum(log_ratio, 0.0))
            committed = resolved.commit(
                current_target, move.position, target_proposal, accepted
            )
            return committed, MarkovTransitionInfo(
                accepted=accepted,
                log_acceptance_ratio=log_ratio,
                proposal_valid=proposal_valid,
                target_valid=target_valid,
            )

        target_states, info = jax.vmap(one_step)(
            state.position,
            state.log_target,
            state.cache,
            state.valid,
            proposal_keys,
            acceptance_keys,
        )
        next_index = state.step_index + jnp.asarray(1, dtype=jnp.uint32)
        if isinstance(resolved, IncrementalMarkovTarget):
            refresh_due = (next_index % resolved.refresh_cadence) == 0
            target_states = jax.lax.cond(
                refresh_due,
                lambda values: jax.vmap(resolved.refresh)(values),
                lambda values: values,
                target_states,
            )
            info = MarkovTransitionInfo(
                accepted=info.accepted,
                log_acceptance_ratio=info.log_acceptance_ratio,
                proposal_valid=info.proposal_valid,
                target_valid=info.target_valid & target_states.valid,
            )
        next_state = MarkovState(
            target_states.position,
            target_states.log_target,
            cache=target_states.cache,
            valid=target_states.valid,
            step_index=next_index,
            target_id=resolved.target_id,
        )
        return next_state, info


def sample_markov(
    target,
    kernel: MetropolisHastings,
    state: MarkovState,
    /,
    *,
    key: Key[Array, ""],
    num_draws: int,
    steps_per_draw: int = 1,
    warmup_steps: int = 0,
    iteration: IterationPlan | None = None,
) -> MarkovSampleResult:
    """Advance persistent chains with optional pure transition observation."""
    resolved = _resolve_target(target)
    if not isinstance(kernel, MetropolisHastings):
        raise TypeError("kernel must be a MetropolisHastings instance.")
    if not isinstance(state, MarkovState):
        raise TypeError("state must be a MarkovState.")
    if iteration is not None and not isinstance(iteration, IterationPlan):
        raise TypeError("iteration must be IterationPlan or None.")
    draws = int(num_draws)
    transitions = int(steps_per_draw)
    warmup = int(warmup_steps)
    if draws <= 0:
        raise ValueError("num_draws must be positive.")
    if transitions <= 0:
        raise ValueError("steps_per_draw must be positive.")
    if warmup < 0:
        raise ValueError("warmup_steps must be non-negative.")

    iteration_scope = None
    iteration_capabilities = None
    iteration_state = None
    if iteration is not None:
        iteration_capabilities = IterationCapabilities(
            ("terminal", "output", "step"),
            mapped_records=True,
        )
        iteration_scope = bind_iteration_scope(
            iteration,
            iteration_capabilities,
            f"markov:{kernel.kernel_id}",
        )
        initial_info = MarkovTransitionInfo(
            jnp.zeros_like(state.valid),
            jnp.full_like(state.log_target, jnp.nan),
            state.valid,
            state.valid,
        )
        initial_record = _markov_iteration_record(
            state,
            initial_info,
            IterationPhase.START,
            warmup=True,
            draw_index=-1,
            active=state.valid,
        )
        iteration_state = initialize_iteration(iteration, initial_record)

    if iteration is None:

        def discard_step(carry, _):
            next_state, _info = kernel.step(resolved, carry, key)
            return next_state, None

        warmed, _ = jax.lax.scan(discard_step, state, xs=None, length=warmup)

        def collect_draw(carry, _):
            def transition_step(inner, __):
                next_state, info = kernel.step(resolved, inner, key)
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
    else:
        assert iteration_state is not None

        def discard_step(carry, _):
            current, observed = carry
            next_state, info = kernel.step(resolved, current, key)
            record = _markov_iteration_record(
                next_state,
                info,
                IterationPhase.COMMIT,
                warmup=True,
                draw_index=-1,
                active=(
                    next_state.valid
                    if iteration.granularity == "step"
                    else jnp.zeros_like(next_state.valid)
                ),
            )
            observed = update_iteration(iteration, observed, record, allow_stop=False)
            return (next_state, observed), None

        (warmed, iteration_state), _ = jax.lax.scan(
            discard_step,
            (state, iteration_state),
            xs=None,
            length=warmup,
        )

        def collect_draw(carry, draw_index):
            def transition_step(inner, _):
                current, observed = inner
                next_state, info = kernel.step(resolved, current, key)
                record = _markov_iteration_record(
                    next_state,
                    info,
                    IterationPhase.COMMIT,
                    warmup=False,
                    draw_index=draw_index,
                    active=(
                        next_state.valid
                        if iteration.granularity == "step"
                        else jnp.zeros_like(next_state.valid)
                    ),
                )
                observed = update_iteration(iteration, observed, record, allow_stop=False)
                return (next_state, observed), info

            (next_state, observed), infos = jax.lax.scan(
                transition_step,
                carry,
                xs=None,
                length=transitions,
            )
            if iteration.granularity == "output":
                last_info = jax.tree.map(lambda value: value[-1], infos)
                record = _markov_iteration_record(
                    next_state,
                    last_info,
                    IterationPhase.COMMIT,
                    warmup=False,
                    draw_index=draw_index,
                    active=next_state.valid,
                )
                observed = update_iteration(iteration, observed, record, allow_stop=False)
            output = (
                next_state.position,
                next_state.log_target,
                infos.accepted,
                infos.log_acceptance_ratio,
                infos.proposal_valid,
                infos.target_valid,
            )
            return (next_state, observed), output

        (final_state, iteration_state), outputs = jax.lax.scan(
            collect_draw,
            (warmed, iteration_state),
            xs=jnp.arange(draws, dtype=jnp.int32),
        )

    samples, values, accepted, ratios, proposal_valid, target_valid = outputs
    iteration_evidence = None
    if iteration is not None:
        assert iteration_scope is not None
        assert iteration_capabilities is not None
        assert iteration_state is not None
        final_info = MarkovTransitionInfo(
            accepted[-1, -1],
            ratios[-1, -1],
            proposal_valid[-1, -1],
            target_valid[-1, -1],
        )
        terminal = _markov_iteration_record(
            final_state,
            final_info,
            IterationPhase.TERMINAL,
            warmup=False,
            draw_index=draws - 1,
            active=final_state.valid,
            terminal=True,
        )
        iteration_evidence = finalize_iteration(
            iteration,
            iteration_scope,
            iteration_capabilities,
            iteration_state,
            terminal,
        )
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
        target_id=resolved.target_id,
        warmup_steps=warmup,
        steps_per_draw=transitions,
        iteration_evidence=iteration_evidence,
    )


__all__ = [
    "MarkovIterationMetrics",
    "MarkovSampleResult",
    "MarkovState",
    "MarkovTransitionInfo",
    "MetropolisHastings",
    "sample_markov",
]
