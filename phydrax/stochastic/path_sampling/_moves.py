#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-work path proposals with decomposed Metropolis--Hastings evidence."""

from __future__ import annotations

import abc
from collections.abc import Callable
from math import isfinite, log, pi

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ._core import (
    _fixed_step_time_grid_valid,
    FunctionalDynamicsKernel,
    PATH_PROPAGATION_KERNEL_FAILURE,
    PATH_PROPAGATION_NONFINITE,
    PATH_PROPAGATION_OVERFLOW,
    PATH_PROPAGATION_SUCCESS,
    PathBuffer,
    select_path,
)
from ._targets import (
    AbstractPathAction,
    AbstractPathEnsemble,
    FirstPassagePathEnsemble,
    FixedPathEnsemble,
    path_log_target,
)


class ShootingSelection(StrictModule):
    index: Array
    log_probability: Array
    eligible_count: Array
    valid: Array


class AbstractShootingSelector(StrictModule):
    selector_id: AbstractAttribute[str]

    @abc.abstractmethod
    def select(self, key: Key[Array, ""], path: PathBuffer, /) -> ShootingSelection:
        raise NotImplementedError

    @abc.abstractmethod
    def log_probability(self, path: PathBuffer, index: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def eligible_count(self, path: PathBuffer, /) -> Array:
        raise NotImplementedError


class UniformShootingSelector(AbstractShootingSelector, NonTrainableState):
    """Uniform selector over a half-open interior index interval."""

    endpoint_margin: int = eqx.field(static=True)
    selector_id: str = eqx.field(static=True)

    def __init__(self, endpoint_margin: int = 1, *, selector_id: str | None = None):
        margin = int(endpoint_margin)
        if margin < 0:
            raise ValueError("endpoint_margin must be non-negative.")
        self.endpoint_margin = margin
        self.selector_id = selector_id or canonical_fingerprint(
            {"kind": "uniform-path-shooting-selector-v1", "endpoint_margin": margin}
        )

    def eligible_count(self, path: PathBuffer, /) -> Array:
        return jnp.maximum(path.length - 2 * self.endpoint_margin, 0)

    def log_probability(self, path: PathBuffer, index: Array, /) -> Array:
        count = self.eligible_count(path)
        eligible = (
            (index >= self.endpoint_margin)
            & (index < path.length - self.endpoint_margin)
            & (count > 0)
        )
        return jnp.where(eligible, -jnp.log(count.astype(path.positions.dtype)), -jnp.inf)

    def select(self, key: Key[Array, ""], path: PathBuffer, /) -> ShootingSelection:
        count = self.eligible_count(path)
        offset = jax.random.randint(key, (), 0, jnp.maximum(count, 1), dtype=jnp.int32)
        index = offset + self.endpoint_margin
        valid = (count > 0) & path.valid()
        return ShootingSelection(index, self.log_probability(path, index), count, valid)


class WeightedShootingSelector(AbstractShootingSelector, NonTrainableState):
    """Normalized state-biased selector with explicit asymmetric reverse density."""

    log_weight: Callable[[Array], Array] = eqx.field(static=True)
    endpoint_margin: int = eqx.field(static=True)
    selector_id: str = eqx.field(static=True)

    def __init__(
        self,
        log_weight: Callable[[Array], Array],
        /,
        *,
        endpoint_margin: int = 1,
        selector_id: str,
    ):
        if not callable(log_weight):
            raise TypeError("log_weight must be callable.")
        margin = int(endpoint_margin)
        if margin < 0:
            raise ValueError("endpoint_margin must be non-negative.")
        if not isinstance(selector_id, str) or not selector_id:
            raise ValueError("selector_id must be non-empty.")
        self.log_weight = log_weight
        self.endpoint_margin = margin
        self.selector_id = selector_id

    def eligible_count(self, path: PathBuffer, /) -> Array:
        return jnp.maximum(path.length - 2 * self.endpoint_margin, 0)

    def _logits(self, path: PathBuffer, /) -> tuple[Array, Array, Array]:
        values = jnp.asarray(self.log_weight(path.positions))
        if values.shape != (path.capacity,) or jnp.iscomplexobj(values):
            raise ValueError(
                "Shooting log weights must return one real value per path point."
            )
        index = jnp.arange(path.capacity, dtype=jnp.int32)
        geometric = (
            path.mask
            & (index >= self.endpoint_margin)
            & (index < path.length - self.endpoint_margin)
        )
        invalid = geometric & (jnp.isnan(values) | jnp.isposinf(values))
        support = geometric & jnp.isfinite(values)
        valid = ~jnp.any(invalid) & jnp.any(support)
        return jnp.where(support, values, -jnp.inf), support, valid

    def log_probability(self, path: PathBuffer, index: Array, /) -> Array:
        logits, support, valid = self._logits(path)
        maximum = jnp.where(valid, jnp.max(logits), 0.0)
        normalizer = maximum + jnp.log(jnp.sum(jnp.exp(logits - maximum)))
        selected = jnp.clip(index, 0, path.capacity - 1)
        return jnp.where(
            valid & support[selected],
            logits[selected] - normalizer,
            -jnp.inf,
        )

    def select(self, key: Key[Array, ""], path: PathBuffer, /) -> ShootingSelection:
        logits, support, weights_valid = self._logits(path)
        index = jax.random.categorical(key, logits).astype(jnp.int32)
        count = self.eligible_count(path)
        valid = path.valid() & weights_valid & (count > 0) & support[index]
        return ShootingSelection(index, self.log_probability(path, index), count, valid)


class ShootingModification(StrictModule):
    state: Array
    forward_log_density: Array
    reverse_log_density: Array
    valid: Array


class AbstractShootingModifier(StrictModule):
    modifier_id: AbstractAttribute[str]

    @abc.abstractmethod
    def apply(self, key: Key[Array, ""], state: Array, /) -> ShootingModification:
        raise NotImplementedError


class IdentityShootingModifier(AbstractShootingModifier, NonTrainableState):
    modifier_id: str = eqx.field(static=True, default="identity-shooting-modifier")

    def apply(self, key: Key[Array, ""], state: Array, /) -> ShootingModification:
        del key
        return ShootingModification(
            state,
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.all(jnp.isfinite(state)),
        )


class GaussianShootingModifier(AbstractShootingModifier, NonTrainableState):
    """Normalized symmetric Gaussian perturbation of a shooting state."""

    scale: float = eqx.field(static=True)
    modifier_id: str = eqx.field(static=True)

    def __init__(self, scale: float, /, *, modifier_id: str | None = None):
        scale_ = float(scale)
        if not isfinite(scale_) or scale_ <= 0.0:
            raise ValueError("scale must be finite and positive.")
        self.scale = scale_
        self.modifier_id = modifier_id or canonical_fingerprint(
            {"kind": "gaussian-shooting-modifier-v1", "scale": scale_.hex()}
        )

    def apply(self, key: Key[Array, ""], state: Array, /) -> ShootingModification:
        noise = self.scale * jax.random.normal(key, state.shape, dtype=state.dtype)
        proposed = state + noise
        squared = jnp.sum((noise / self.scale) ** 2)
        log_density = -0.5 * squared - state.size * log(self.scale * (2.0 * pi) ** 0.5)
        valid = jnp.all(jnp.isfinite(proposed)) & jnp.isfinite(log_density)
        return ShootingModification(proposed, log_density, log_density, valid)


def _validated_selection(
    selector: AbstractShootingSelector,
    key: Key[Array, ""],
    path: PathBuffer,
    /,
) -> ShootingSelection:
    result = selector.select(key, path)
    if not isinstance(result, ShootingSelection):
        raise TypeError("A shooting selector must return ShootingSelection.")
    raw_index = jnp.asarray(result.index)
    reported_log = jnp.asarray(result.log_probability)
    reported_count = jnp.asarray(result.eligible_count)
    reported_valid = jnp.asarray(result.valid)
    expected_count = jnp.asarray(selector.eligible_count(path))
    if (
        raw_index.shape != ()
        or not jnp.issubdtype(raw_index.dtype, jnp.integer)
        or reported_log.shape != ()
        or jnp.iscomplexobj(reported_log)
        or not jnp.issubdtype(reported_log.dtype, jnp.floating)
        or reported_count.shape != ()
        or not jnp.issubdtype(reported_count.dtype, jnp.integer)
        or reported_valid.shape != ()
        or reported_valid.dtype != jnp.bool_
        or expected_count.shape != ()
        or not jnp.issubdtype(expected_count.dtype, jnp.integer)
    ):
        raise ValueError(
            "ShootingSelection must carry scalar integer index/count, real floating log density, and Boolean validity."
        )
    claimed_log = jnp.asarray(selector.log_probability(path, raw_index))
    if (
        claimed_log.shape != ()
        or jnp.iscomplexobj(claimed_log)
        or not jnp.issubdtype(claimed_log.dtype, jnp.floating)
    ):
        raise ValueError("Selector log_probability must return one real floating scalar.")
    density_dtype = jnp.result_type(reported_log.dtype, claimed_log.dtype, jnp.float32)
    reported_density = reported_log.astype(density_dtype)
    claimed_density = claimed_log.astype(density_dtype)
    tolerance = (
        32.0 * jnp.finfo(density_dtype).eps * jnp.maximum(jnp.abs(claimed_density), 1.0)
    )
    density_matches = (
        jnp.isfinite(reported_density)
        & jnp.isfinite(claimed_density)
        & (jnp.abs(reported_density - claimed_density) <= tolerance)
    )
    in_range = (raw_index >= 0) & (raw_index < path.length)
    count_matches = (reported_count == expected_count) & (reported_count > 0)
    valid = reported_valid & path.valid() & in_range & count_matches & density_matches
    safe_index = jnp.clip(raw_index, 0, path.capacity - 1).astype(jnp.int32)
    return ShootingSelection(
        safe_index,
        claimed_density,
        expected_count.astype(jnp.int32),
        valid,
    )


def _validated_modification(
    modifier: AbstractShootingModifier,
    key: Key[Array, ""],
    state: Array,
    /,
) -> ShootingModification:
    result = modifier.apply(key, state)
    if not isinstance(result, ShootingModification):
        raise TypeError("A shooting modifier must return ShootingModification.")
    proposed = jnp.asarray(result.state)
    forward = jnp.asarray(result.forward_log_density)
    reverse = jnp.asarray(result.reverse_log_density)
    reported_valid = jnp.asarray(result.valid)
    if (
        proposed.shape != state.shape
        or proposed.dtype != state.dtype
        or forward.shape != ()
        or reverse.shape != ()
        or jnp.iscomplexobj(forward)
        or jnp.iscomplexobj(reverse)
        or not jnp.issubdtype(forward.dtype, jnp.floating)
        or not jnp.issubdtype(reverse.dtype, jnp.floating)
        or reported_valid.shape != ()
        or reported_valid.dtype != jnp.bool_
    ):
        raise ValueError(
            "ShootingModification must preserve state shape/dtype and carry real scalar densities and Boolean validity."
        )
    valid = (
        reported_valid
        & jnp.all(jnp.isfinite(proposed))
        & jnp.isfinite(forward)
        & jnp.isfinite(reverse)
    )
    return ShootingModification(proposed, forward, reverse, valid)


class PathProposalEvaluation(StrictModule):
    """Complete additive evidence for one path-space proposal."""

    target_log_ratio: Array
    selector_log_ratio: Array
    modifier_log_ratio: Array
    propagation_log_ratio: Array
    length_log_ratio: Array
    exchange_log_ratio: Array
    log_acceptance_ratio: Array
    target_valid: Array
    selector_valid: Array
    modifier_valid: Array
    propagation_valid: Array
    length_valid: Array
    exchange_valid: Array
    proposal_valid: Array
    propagation_status: Array

    @classmethod
    def rejected(cls, status: int, /) -> PathProposalEvaluation:
        zero = jnp.asarray(0.0)
        false = jnp.asarray(False)
        true = jnp.asarray(True)
        return cls(
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            -jnp.inf,
            false,
            false,
            false,
            false,
            false,
            true,
            false,
            jnp.asarray(status, jnp.int32),
        )


class PathMoveResult(StrictModule):
    current: PathBuffer
    proposed: PathBuffer
    committed: PathBuffer
    evaluation: PathProposalEvaluation
    accepted: Array
    shooting_index: Array
    candidate_shooting_index: Array


class _GrowthResult(StrictModule):
    positions: Array
    log_probability: Array
    count: Array
    terminal_hit: Array
    valid: Array
    status: Array


def _grow_fixed(
    kernel: FunctionalDynamicsKernel,
    ensemble: AbstractPathEnsemble,
    key: Key[Array, ""],
    initial: Array,
    kernel_direction: Array,
    terminal_direction: Array,
    terminal_required: Array,
    requested_steps: Array,
    capacity: int,
    /,
) -> _GrowthResult:
    maximum = capacity - 1
    keys = jax.random.split(key, maximum)
    values = jnp.zeros((maximum,) + initial.shape, dtype=initial.dtype)
    density_dtype = kernel.transition_log_density(
        initial, initial, kernel_direction
    ).dtype

    def body(carry, item):
        state, output, log_probability, count, done, failed, status = carry
        index, step_key = item
        should_step = (~done) & (~failed) & (index < requested_steps)
        transition = kernel.step(step_key, state, kernel_direction)
        finite = (
            transition.valid
            & jnp.all(jnp.isfinite(transition.state))
            & jnp.isfinite(transition.log_transition_density)
        )
        use = should_step & finite
        output = output.at[index].set(jnp.where(use, transition.state, output[index]))
        state = jnp.where(use, transition.state, state)
        log_probability = log_probability + jnp.where(
            use,
            transition.log_transition_density,
            jnp.asarray(0.0, dtype=density_dtype),
        )
        count = count + use.astype(jnp.int32)
        failed_now = should_step & ~finite
        status = jnp.where(failed_now, transition.status, status)
        hit = (
            use
            & terminal_required
            & ensemble.terminal(transition.state, terminal_direction)
        )
        finished_count = use & ~terminal_required & ((index + 1) >= requested_steps)
        return (
            state,
            output,
            log_probability,
            count,
            done | hit | finished_count,
            failed | failed_now,
            status,
        ), None

    initial_carry = (
        initial,
        values,
        jnp.asarray(0.0, dtype=density_dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(PATH_PROPAGATION_SUCCESS, dtype=jnp.int32),
    )
    (_, output, log_probability, count, done, failed, status), _ = jax.lax.scan(
        body,
        initial_carry,
        (jnp.arange(maximum, dtype=jnp.int32), keys),
    )
    overflow = terminal_required & ~done & ~failed
    status = jnp.where(overflow, PATH_PROPAGATION_OVERFLOW, status)
    used = jnp.arange(maximum, dtype=jnp.int32) < count
    finite = jnp.all(
        jnp.isfinite(
            jnp.where(
                used.reshape((maximum,) + (1,) * initial.ndim),
                output,
                0.0,
            )
        )
    ) & jnp.isfinite(log_probability)
    status = jnp.where(~finite, PATH_PROPAGATION_NONFINITE, status)
    valid = ~failed & ~overflow & finite & (status == PATH_PROPAGATION_SUCCESS)
    return _GrowthResult(output, log_probability, count, done, valid, status)


def _segment_log_density(
    kernel: FunctionalDynamicsKernel,
    path: PathBuffer,
    shooting_index: Array,
    relative_direction: Array,
    /,
) -> Array:
    indices = jnp.arange(path.capacity - 1, dtype=jnp.int32)
    forward_values = jax.vmap(kernel.transition_log_density, in_axes=(0, 0, None))(
        path.positions[:-1], path.positions[1:], path.direction
    )
    backward_values = jax.vmap(kernel.transition_log_density, in_axes=(0, 0, None))(
        path.positions[1:], path.positions[:-1], -path.direction
    )
    forward_mask = (indices >= shooting_index) & (indices < path.length - 1)
    backward_mask = (indices < shooting_index) & (indices < path.length - 1)
    values = jnp.where(relative_direction > 0, forward_values, backward_values)
    mask = jnp.where(relative_direction > 0, forward_mask, backward_mask)
    return jnp.sum(jnp.where(mask, values, 0.0))


def _assemble_two_way(
    current: PathBuffer,
    modification: ShootingModification,
    backward: _GrowthResult,
    forward: _GrowthResult,
    shooting_index: Array,
    kernel: FunctionalDynamicsKernel,
    /,
) -> PathBuffer:
    capacity = current.capacity
    index = jnp.arange(capacity, dtype=jnp.int32)
    backward_source = jnp.clip(backward.count - 1 - index, 0, capacity - 2)
    forward_source = jnp.clip(index - backward.count - 1, 0, capacity - 2)
    event_index = index.reshape((capacity,) + (1,) * len(current.event_shape))
    positions = jnp.where(
        event_index < backward.count,
        backward.positions[backward_source],
        jnp.where(
            event_index == backward.count,
            modification.state,
            forward.positions[forward_source],
        ),
    )
    length = backward.count + 1 + forward.count
    active = index < length
    positions = jnp.where(
        active.reshape((capacity,) + (1,) * len(current.event_shape)),
        positions,
        jnp.zeros_like(positions),
    )
    times = jnp.where(
        active,
        index.astype(current.times.dtype) * kernel.time_step,
        0.0,
    )
    ancestry = current.lineage[shooting_index]
    lineage = jnp.where(active, ancestry, -jnp.ones_like(current.lineage))
    return PathBuffer(positions, times, length, active, current.direction, lineage)


def _assemble_one_way(
    current: PathBuffer,
    modification: ShootingModification,
    growth: _GrowthResult,
    shooting_index: Array,
    direction: Array,
    kernel: FunctionalDynamicsKernel,
    /,
) -> tuple[PathBuffer, Array]:
    capacity = current.capacity
    index = jnp.arange(capacity, dtype=jnp.int32)
    ancestry = current.lineage[shooting_index]
    forward = direction > 0
    event_index = index.reshape((capacity,) + (1,) * len(current.event_shape))

    forward_source = jnp.clip(index - shooting_index - 1, 0, capacity - 2)
    forward_positions = jnp.where(
        event_index < shooting_index,
        current.positions,
        jnp.where(
            event_index == shooting_index,
            modification.state,
            growth.positions[forward_source],
        ),
    )
    forward_length = shooting_index + 1 + growth.count
    forward_lineage = jnp.where(index < shooting_index, current.lineage, ancestry)

    suffix_length = current.length - shooting_index - 1
    backward_source = jnp.clip(growth.count - 1 - index, 0, capacity - 2)
    suffix_source = jnp.clip(shooting_index + index - growth.count, 0, capacity - 1)
    backward_positions = jnp.where(
        event_index < growth.count,
        growth.positions[backward_source],
        jnp.where(
            event_index == growth.count,
            modification.state,
            current.positions[suffix_source],
        ),
    )
    backward_length = growth.count + 1 + suffix_length
    backward_lineage = jnp.where(
        index <= growth.count, ancestry, current.lineage[suffix_source]
    )

    positions = jnp.where(forward, forward_positions, backward_positions)
    length = jnp.where(forward, forward_length, backward_length)
    active = index < length
    positions = jnp.where(
        active.reshape((capacity,) + (1,) * len(current.event_shape)),
        positions,
        jnp.zeros_like(positions),
    )
    times = jnp.where(
        active,
        index.astype(current.times.dtype) * kernel.time_step,
        0.0,
    )
    lineage = jnp.where(
        active,
        jnp.where(forward, forward_lineage, backward_lineage),
        -jnp.ones_like(current.lineage),
    )
    candidate_index = jnp.where(forward, shooting_index, growth.count)
    return PathBuffer(
        positions, times, length, active, current.direction, lineage
    ), candidate_index


def _proposal_evaluation(
    ensemble: AbstractPathEnsemble,
    action: AbstractPathAction,
    selector: AbstractShootingSelector,
    current: PathBuffer,
    proposed: PathBuffer,
    current_index: Array,
    proposed_index: Array,
    modification: ShootingModification,
    forward_propagation: Array,
    reverse_propagation: Array,
    propagation_valid: Array,
    propagation_status: Array,
    /,
) -> PathProposalEvaluation:
    current_target = path_log_target(ensemble, action, current)
    proposed_target = path_log_target(ensemble, action, proposed)
    raw_target_ratio = proposed_target - current_target
    raw_forward_selector = selector.log_probability(current, current_index)
    raw_reverse_selector = selector.log_probability(proposed, proposed_index)
    raw_modifier_ratio = (
        modification.reverse_log_density - modification.forward_log_density
    )
    raw_propagation_ratio = reverse_propagation - forward_propagation
    evidence_dtype = jnp.result_type(
        raw_target_ratio.dtype,
        raw_forward_selector.dtype,
        raw_reverse_selector.dtype,
        raw_modifier_ratio.dtype,
        raw_propagation_ratio.dtype,
        jnp.float32,
    )
    target_ratio = jnp.asarray(raw_target_ratio, dtype=evidence_dtype)
    forward_selector = jnp.asarray(raw_forward_selector, dtype=evidence_dtype)
    reverse_selector = jnp.asarray(raw_reverse_selector, dtype=evidence_dtype)
    modifier_ratio = jnp.asarray(raw_modifier_ratio, dtype=evidence_dtype)
    propagation_ratio = jnp.asarray(raw_propagation_ratio, dtype=evidence_dtype)
    current_count = selector.eligible_count(current)
    proposed_count = selector.eligible_count(proposed)
    length_valid = (current_count > 0) & (proposed_count > 0)
    length_ratio = jnp.where(
        length_valid,
        jnp.log(current_count.astype(evidence_dtype))
        - jnp.log(proposed_count.astype(evidence_dtype)),
        -jnp.inf,
    )
    selector_ratio = reverse_selector - forward_selector - length_ratio
    zero = jnp.asarray(0.0, dtype=evidence_dtype)
    values = jnp.stack(
        (
            target_ratio,
            selector_ratio,
            modifier_ratio,
            propagation_ratio,
            length_ratio,
            zero,
        )
    )
    target_valid = jnp.isfinite(current_target) & jnp.isfinite(proposed_target)
    selector_valid = jnp.isfinite(forward_selector) & jnp.isfinite(reverse_selector)
    modifier_valid = modification.valid & jnp.isfinite(modifier_ratio)
    all_valid = (
        target_valid
        & selector_valid
        & modifier_valid
        & propagation_valid
        & length_valid
        & jnp.all(jnp.isfinite(values))
        & proposed.valid()
    )
    ratio = jnp.where(all_valid, jnp.sum(values), -jnp.inf)
    return PathProposalEvaluation(
        target_ratio,
        selector_ratio,
        modifier_ratio,
        propagation_ratio,
        length_ratio,
        zero,
        ratio,
        target_valid,
        selector_valid,
        modifier_valid,
        propagation_valid,
        length_valid,
        jnp.asarray(True),
        all_valid,
        jnp.asarray(propagation_status, jnp.int32),
    )


def _accept(
    current: PathBuffer,
    proposed: PathBuffer,
    evaluation: PathProposalEvaluation,
    key: Key[Array, ""],
    current_index: Array,
    proposed_index: Array,
    /,
) -> PathMoveResult:
    accepted = evaluation.proposal_valid & (
        jnp.log(jax.random.uniform(key))
        < jnp.minimum(evaluation.log_acceptance_ratio, 0.0)
    )
    return PathMoveResult(
        current,
        proposed,
        select_path(current, proposed, accepted),
        evaluation,
        accepted,
        current_index,
        proposed_index,
    )


def propose_one_way_shooting(
    ensemble: AbstractPathEnsemble,
    action: AbstractPathAction,
    kernel: FunctionalDynamicsKernel,
    selector: AbstractShootingSelector,
    modifier: AbstractShootingModifier,
    current: PathBuffer,
    key: Key[Array, ""],
    /,
) -> PathMoveResult:
    """Propose one one-way shooting move; any propagation failure rejects once."""

    if not kernel.capabilities.supports_backward:
        raise ValueError("One-way shooting requires backward-capable dynamics.")
    if not kernel.capabilities.fixed_step:
        raise ValueError("One-way shooting requires fixed-step dynamics.")
    selection_key, modifier_key, direction_key, propagation_key, acceptance_key = (
        jax.random.split(key, 5)
    )
    selection = _validated_selection(selector, selection_key, current)
    modification = _validated_modification(
        modifier, modifier_key, current.positions[selection.index]
    )
    relative_direction = jnp.where(jax.random.bernoulli(direction_key), 1, -1).astype(
        jnp.int8
    )
    kernel_direction = relative_direction * current.direction
    fixed = isinstance(ensemble, FixedPathEnsemble)
    first_passage = isinstance(ensemble, FirstPassagePathEnsemble)
    fixed_requested = jnp.where(
        relative_direction > 0,
        current.length - selection.index - 1,
        selection.index,
    )
    requested = jnp.where(
        fixed,
        fixed_requested,
        jnp.where(
            first_passage & (relative_direction < 0),
            selection.index,
            current.capacity - 1,
        ),
    )
    terminal_required = jnp.asarray(ensemble.requires_terminal_hit) & ~(
        first_passage & (relative_direction < 0)
    )
    growth = _grow_fixed(
        kernel,
        ensemble,
        propagation_key,
        modification.state,
        kernel_direction,
        relative_direction,
        terminal_required,
        requested,
        current.capacity,
    )
    proposed, proposed_index = _assemble_one_way(
        current,
        modification,
        growth,
        selection.index,
        relative_direction,
        kernel,
    )
    reverse_probability = _segment_log_density(
        kernel, current, selection.index, relative_direction
    )
    capacity_valid = proposed.length <= current.capacity
    grid_valid = _fixed_step_time_grid_valid(
        current, kernel.time_step
    ) & _fixed_step_time_grid_valid(proposed, kernel.time_step)
    propagation_valid = selection.valid & growth.valid & capacity_valid & grid_valid
    status = jnp.where(
        growth.status != PATH_PROPAGATION_SUCCESS,
        growth.status,
        jnp.where(
            ~capacity_valid,
            PATH_PROPAGATION_OVERFLOW,
            jnp.where(
                ~grid_valid,
                PATH_PROPAGATION_KERNEL_FAILURE,
                PATH_PROPAGATION_SUCCESS,
            ),
        ),
    )
    evaluation = _proposal_evaluation(
        ensemble,
        action,
        selector,
        current,
        proposed,
        selection.index,
        proposed_index,
        modification,
        growth.log_probability,
        reverse_probability,
        propagation_valid,
        status,
    )
    return _accept(
        current, proposed, evaluation, acceptance_key, selection.index, proposed_index
    )


def propose_two_way_shooting(
    ensemble: AbstractPathEnsemble,
    action: AbstractPathAction,
    kernel: FunctionalDynamicsKernel,
    selector: AbstractShootingSelector,
    modifier: AbstractShootingModifier,
    current: PathBuffer,
    key: Key[Array, ""],
    /,
) -> PathMoveResult:
    """Regrow both temporal directions from one modified shooting state."""

    if not kernel.capabilities.supports_backward:
        raise ValueError("Two-way shooting requires backward-capable dynamics.")
    if not kernel.capabilities.fixed_step:
        raise ValueError("Two-way shooting requires fixed-step dynamics.")
    selection_key, modifier_key, backward_key, forward_key, acceptance_key = (
        jax.random.split(key, 5)
    )
    selection = _validated_selection(selector, selection_key, current)
    modification = _validated_modification(
        modifier, modifier_key, current.positions[selection.index]
    )
    fixed = isinstance(ensemble, FixedPathEnsemble)
    first_passage = isinstance(ensemble, FirstPassagePathEnsemble)
    backward_steps = jnp.where(
        fixed | first_passage, selection.index, current.capacity - 1
    )
    forward_steps = jnp.where(
        fixed,
        current.length - selection.index - 1,
        current.capacity - 1,
    )
    backward_terminal_required = jnp.asarray(
        ensemble.requires_terminal_hit and not first_passage
    )
    forward_terminal_required = jnp.asarray(ensemble.requires_terminal_hit)
    backward = _grow_fixed(
        kernel,
        ensemble,
        backward_key,
        modification.state,
        -current.direction,
        jnp.asarray(-1, jnp.int8),
        backward_terminal_required,
        backward_steps,
        current.capacity,
    )
    forward = _grow_fixed(
        kernel,
        ensemble,
        forward_key,
        modification.state,
        current.direction,
        jnp.asarray(1, jnp.int8),
        forward_terminal_required,
        forward_steps,
        current.capacity,
    )
    proposed = _assemble_two_way(
        current, modification, backward, forward, selection.index, kernel
    )
    proposed_index = backward.count
    reverse_probability = _segment_log_density(
        kernel, current, selection.index, jnp.asarray(-1, jnp.int8)
    ) + _segment_log_density(kernel, current, selection.index, jnp.asarray(1, jnp.int8))
    forward_probability = backward.log_probability + forward.log_probability
    capacity_valid = proposed.length <= current.capacity
    status = jnp.where(
        backward.status != PATH_PROPAGATION_SUCCESS, backward.status, forward.status
    )
    grid_valid = _fixed_step_time_grid_valid(
        current, kernel.time_step
    ) & _fixed_step_time_grid_valid(proposed, kernel.time_step)
    propagation_valid = (
        selection.valid & backward.valid & forward.valid & capacity_valid & grid_valid
    )
    status = jnp.where(
        ~grid_valid,
        PATH_PROPAGATION_KERNEL_FAILURE,
        jnp.where(~capacity_valid, PATH_PROPAGATION_OVERFLOW, status),
    )
    evaluation = _proposal_evaluation(
        ensemble,
        action,
        selector,
        current,
        proposed,
        selection.index,
        proposed_index,
        modification,
        forward_probability,
        reverse_probability,
        propagation_valid,
        status,
    )
    return _accept(
        current, proposed, evaluation, acceptance_key, selection.index, proposed_index
    )


def propose_path_reversal(
    ensemble: AbstractPathEnsemble,
    action: AbstractPathAction,
    kernel: FunctionalDynamicsKernel,
    current: PathBuffer,
    key: Key[Array, ""],
    /,
) -> PathMoveResult:
    """Metropolize exact active-prefix time reversal."""

    if not kernel.capabilities.reversible:
        raise ValueError("Path reversal requires a reversible dynamics kernel.")
    proposed = current.time_reversed()
    current_target = path_log_target(ensemble, action, current)
    proposed_target = path_log_target(ensemble, action, proposed)
    evidence_dtype = jnp.result_type(
        current_target.dtype, proposed_target.dtype, jnp.float32
    )
    ratio = jnp.asarray(proposed_target - current_target, dtype=evidence_dtype)
    valid = (
        jnp.isfinite(current_target) & jnp.isfinite(proposed_target) & proposed.valid()
    )
    ratio = jnp.where(valid, ratio, -jnp.inf)
    zero = jnp.asarray(0.0, dtype=evidence_dtype)
    true = jnp.asarray(True)
    evaluation = PathProposalEvaluation(
        ratio,
        zero,
        zero,
        zero,
        zero,
        zero,
        ratio,
        valid,
        true,
        true,
        true,
        true,
        true,
        valid,
        jnp.asarray(PATH_PROPAGATION_SUCCESS, jnp.int32),
    )
    return _accept(
        current,
        proposed,
        evaluation,
        key,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )


def propose_path_shift(
    ensemble: AbstractPathEnsemble,
    action: AbstractPathAction,
    kernel: FunctionalDynamicsKernel,
    current: PathBuffer,
    key: Key[Array, ""],
    /,
    *,
    maximum_shift: int,
) -> PathMoveResult:
    """Shift a fixed-length path and regrow exactly the discarded endpoint."""

    if not isinstance(ensemble, FixedPathEnsemble):
        raise ValueError("Path shifting requires a fixed-length ensemble.")
    if not kernel.capabilities.supports_backward:
        raise ValueError("Path shifting requires backward-capable dynamics.")
    if not kernel.capabilities.fixed_step:
        raise ValueError("Path shifting requires fixed-step dynamics.")
    maximum = int(maximum_shift)
    if maximum <= 0 or maximum >= ensemble.path_length:
        raise ValueError("maximum_shift must lie in [1, path_length).")
    amount_key, orientation_key, propagation_key, acceptance_key = jax.random.split(
        key, 4
    )
    amount = jax.random.randint(amount_key, (), 1, maximum + 1, dtype=jnp.int32)
    forward = jax.random.bernoulli(orientation_key)
    initial_index = jnp.where(forward, current.length - 1, 0)
    direction = jnp.where(forward, 1, -1).astype(jnp.int8)
    modification = _validated_modification(
        IdentityShootingModifier(),
        propagation_key,
        current.positions[initial_index],
    )
    growth = _grow_fixed(
        kernel,
        ensemble,
        propagation_key,
        modification.state,
        direction * current.direction,
        direction,
        jnp.asarray(False),
        amount,
        current.capacity,
    )

    index = jnp.arange(current.capacity, dtype=jnp.int32)
    retained_count = current.length - amount
    event_index = index.reshape((current.capacity,) + (1,) * len(current.event_shape))
    forward_old_source = jnp.clip(index + amount, 0, current.capacity - 1)
    forward_growth_source = jnp.clip(index - retained_count, 0, current.capacity - 2)
    forward_positions = jnp.where(
        event_index < retained_count,
        current.positions[forward_old_source],
        growth.positions[forward_growth_source],
    )
    forward_lineage = jnp.where(
        index < retained_count,
        current.lineage[forward_old_source],
        current.lineage[current.length - 1],
    )

    backward_growth_source = jnp.clip(amount - 1 - index, 0, current.capacity - 2)
    backward_old_source = jnp.clip(index - amount, 0, current.capacity - 1)
    backward_positions = jnp.where(
        event_index < amount,
        growth.positions[backward_growth_source],
        current.positions[backward_old_source],
    )
    backward_lineage = jnp.where(
        index < amount,
        current.lineage[0],
        current.lineage[backward_old_source],
    )
    active = index < current.length
    event_mask = active.reshape((current.capacity,) + (1,) * len(current.event_shape))
    proposed = PathBuffer(
        jnp.where(
            event_mask,
            jnp.where(forward, forward_positions, backward_positions),
            jnp.zeros_like(current.positions),
        ),
        jnp.where(
            active,
            index.astype(current.times.dtype) * kernel.time_step,
            0.0,
        ),
        current.length,
        active,
        current.direction,
        jnp.where(
            active,
            jnp.where(forward, forward_lineage, backward_lineage),
            -jnp.ones_like(current.lineage),
        ),
    )
    reverse_shooting_index = jnp.where(forward, amount, current.length - amount - 1)
    reverse_direction = -direction
    reverse_probability = _segment_log_density(
        kernel, current, reverse_shooting_index, reverse_direction
    )
    selector = UniformShootingSelector(endpoint_margin=0)
    grid_valid = _fixed_step_time_grid_valid(
        current, kernel.time_step
    ) & _fixed_step_time_grid_valid(proposed, kernel.time_step)
    propagation_valid = growth.valid & grid_valid
    status = jnp.where(grid_valid, growth.status, PATH_PROPAGATION_KERNEL_FAILURE)
    evaluation = _proposal_evaluation(
        ensemble,
        action,
        selector,
        current,
        proposed,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
        modification,
        growth.log_probability,
        reverse_probability,
        propagation_valid,
        status,
    )
    return _accept(
        current,
        proposed,
        evaluation,
        acceptance_key,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )


class ReplicaExchangeResult(StrictModule):
    left: PathBuffer
    right: PathBuffer
    evaluation: PathProposalEvaluation
    accepted: Array


def propose_replica_exchange(
    left_ensemble: AbstractPathEnsemble,
    right_ensemble: AbstractPathEnsemble,
    action: AbstractPathAction,
    left: PathBuffer,
    right: PathBuffer,
    key: Key[Array, ""],
    /,
) -> ReplicaExchangeResult:
    """Swap neighboring interface replicas with symmetric exchange evidence."""
    if not isinstance(left, PathBuffer) or not isinstance(right, PathBuffer):
        raise TypeError("Replica exchange requires PathBuffer values.")
    left_signature = (
        left.positions.shape,
        left.positions.dtype,
        left.times.shape,
        left.times.dtype,
        left.mask.shape,
        left.mask.dtype,
        left.lineage.shape,
        left.lineage.dtype,
    )
    right_signature = (
        right.positions.shape,
        right.positions.dtype,
        right.times.shape,
        right.times.dtype,
        right.mask.shape,
        right.mask.dtype,
        right.lineage.shape,
        right.lineage.dtype,
    )
    if left_signature != right_signature:
        raise ValueError("Replica exchange paths must have identical shapes and dtypes.")

    current = path_log_target(left_ensemble, action, left) + path_log_target(
        right_ensemble, action, right
    )
    crossed = path_log_target(left_ensemble, action, right) + path_log_target(
        right_ensemble, action, left
    )
    evidence_dtype = jnp.result_type(current.dtype, crossed.dtype, jnp.float32)
    ratio = jnp.asarray(crossed - current, dtype=evidence_dtype)
    valid = jnp.isfinite(current) & jnp.isfinite(crossed)
    ratio = jnp.where(valid, ratio, -jnp.inf)
    zero = jnp.asarray(0.0, dtype=evidence_dtype)
    true = jnp.asarray(True)
    evaluation = PathProposalEvaluation(
        zero,
        zero,
        zero,
        zero,
        zero,
        ratio,
        ratio,
        true,
        true,
        true,
        true,
        true,
        valid,
        valid,
        jnp.asarray(PATH_PROPAGATION_SUCCESS, jnp.int32),
    )
    accepted = valid & (jnp.log(jax.random.uniform(key)) < jnp.minimum(ratio, 0.0))
    return ReplicaExchangeResult(
        select_path(left, right, accepted),
        select_path(right, left, accepted),
        evaluation,
        accepted,
    )


__all__ = [
    "AbstractShootingModifier",
    "AbstractShootingSelector",
    "GaussianShootingModifier",
    "IdentityShootingModifier",
    "PathMoveResult",
    "PathProposalEvaluation",
    "propose_one_way_shooting",
    "propose_path_reversal",
    "propose_path_shift",
    "propose_replica_exchange",
    "propose_two_way_shooting",
    "ReplicaExchangeResult",
    "ShootingModification",
    "ShootingSelection",
    "UniformShootingSelector",
    "WeightedShootingSelector",
]
