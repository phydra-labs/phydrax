#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class WalkForwardPlan(StrictModule):
    training_window: int = eqx.field(static=True)
    test_window: int = eqx.field(static=True)
    step: int = eqx.field(static=True)
    embargo: int = eqx.field(static=True)
    anchored: bool = eqx.field(static=True)

    def __init__(
        self,
        training_window: int,
        test_window: int,
        /,
        *,
        step: int | None = None,
        embargo: int = 0,
        anchored: bool = False,
    ):
        raw_step = test_window if step is None else step
        if any(
            isinstance(value, bool)
            for value in (training_window, test_window, raw_step, embargo)
        ):
            raise TypeError("Backtest window sizes must be integers.")
        if not isinstance(anchored, bool):
            raise TypeError("anchored must be a boolean.")
        training = int(training_window)
        test = int(test_window)
        step_ = int(raw_step)
        embargo_ = int(embargo)
        if training <= 0 or test <= 0 or step_ < test or embargo_ < 0:
            raise ValueError(
                "Backtest windows must be positive, step must prevent overlapping tests, "
                "and embargo must be non-negative."
            )
        self.training_window, self.test_window = training, test
        self.step, self.embargo, self.anchored = step_, embargo_, bool(anchored)


class WalkForwardSplit(StrictModule):
    train_start: int = eqx.field(static=True)
    train_stop: int = eqx.field(static=True)
    test_start: int = eqx.field(static=True)
    test_stop: int = eqx.field(static=True)

    def __init__(
        self, train_start: int, train_stop: int, test_start: int, test_stop: int, /
    ):
        values = tuple(
            int(value) for value in (train_start, train_stop, test_start, test_stop)
        )
        if values[0] < 0 or not values[0] < values[1] <= values[2] < values[3]:
            raise ValueError(
                "Split indices require train_start < train_stop <= test_start < test_stop."
            )
        self.train_start, self.train_stop, self.test_start, self.test_stop = values


class BacktestDecisions(StrictModule):
    weights: Array
    information_end_indices: Array

    def __init__(self, weights: ArrayLike, information_end_indices: ArrayLike, /):
        weights_ = jnp.asarray(weights)
        information = jnp.asarray(information_end_indices)
        if weights_.ndim != 2 or 0 in weights_.shape:
            raise ValueError("weights must have shape (split, asset).")
        if information.shape != (weights_.shape[0],) or not jnp.issubdtype(
            information.dtype, jnp.integer
        ):
            raise TypeError("information_end_indices must be one integer per split.")
        weights_ = weights_.astype(jnp.result_type(weights_.dtype, jnp.float32))
        if not np.all(np.isfinite(np.asarray(weights_))):
            raise ValueError("Backtest decision weights must be finite.")
        self.weights = weights_
        self.information_end_indices = information.astype(jnp.int32)


class WalkForwardResult(StrictModule):
    out_of_sample_returns: Array
    wealth: Array
    observation_indices: Array
    split_index: Array
    split_returns: Array
    valid: Array
    splits: tuple[WalkForwardSplit, ...] = eqx.field(static=True)


class NestedBacktestPlan(StrictModule):
    outer: WalkForwardPlan
    inner: WalkForwardPlan

    def __init__(self, outer: WalkForwardPlan, inner: WalkForwardPlan, /):
        if not isinstance(outer, WalkForwardPlan) or not isinstance(
            inner, WalkForwardPlan
        ):
            raise TypeError("outer and inner must be WalkForwardPlan values.")
        self.outer, self.inner = outer, inner


class NestedBacktestSplits(StrictModule):
    outer: tuple[WalkForwardSplit, ...] = eqx.field(static=True)
    inner: tuple[tuple[WalkForwardSplit, ...], ...] = eqx.field(static=True)


class NestedBacktestDecisions(StrictModule):
    """Precomputed inner candidates and outer refits with explicit information cutoffs."""

    inner_weights: Array
    inner_information_end_indices: Array
    outer_candidate_weights: Array
    outer_information_end_indices: Array

    def __init__(
        self,
        inner_weights: ArrayLike,
        inner_information_end_indices: ArrayLike,
        outer_candidate_weights: ArrayLike,
        outer_information_end_indices: ArrayLike,
        /,
    ):
        inner = jnp.asarray(inner_weights)
        inner_information = jnp.asarray(inner_information_end_indices)
        outer = jnp.asarray(outer_candidate_weights, dtype=inner.dtype)
        outer_information = jnp.asarray(outer_information_end_indices)
        if inner.ndim != 4 or 0 in inner.shape:
            raise ValueError(
                "inner_weights must have shape (outer, inner, candidate, asset)."
            )
        if inner_information.shape != inner.shape[:-1] or not jnp.issubdtype(
            inner_information.dtype, jnp.integer
        ):
            raise TypeError("inner information cutoffs must match inner candidates.")
        expected_outer = (inner.shape[0], inner.shape[2], inner.shape[3])
        if outer.shape != expected_outer:
            raise ValueError(
                "outer_candidate_weights must have shape (outer, candidate, asset)."
            )
        if outer_information.shape != expected_outer[:-1] or not jnp.issubdtype(
            outer_information.dtype, jnp.integer
        ):
            raise TypeError("outer information cutoffs must match outer candidates.")
        inner = inner.astype(jnp.result_type(inner.dtype, jnp.float32))
        outer = outer.astype(inner.dtype)
        if not np.all(np.isfinite(np.asarray(inner))) or not np.all(
            np.isfinite(np.asarray(outer))
        ):
            raise ValueError("Nested decision weights must be finite.")
        self.inner_weights, self.outer_candidate_weights = inner, outer
        self.inner_information_end_indices = inner_information.astype(jnp.int32)
        self.outer_information_end_indices = outer_information.astype(jnp.int32)


class NestedBacktestResult(StrictModule):
    out_of_sample_returns: Array
    wealth: Array
    selected_candidate: Array
    inner_scores: Array
    observation_indices: Array
    valid: Array
    splits: NestedBacktestSplits = eqx.field(static=True)


def walk_forward_splits(
    length: int, plan: WalkForwardPlan, /
) -> tuple[WalkForwardSplit, ...]:
    """Construct deterministic, non-overlapping, embargoed out-of-sample splits."""

    if not isinstance(plan, WalkForwardPlan):
        raise TypeError("plan must be a WalkForwardPlan.")
    observations = int(length)
    if isinstance(length, bool) or observations <= 0:
        raise ValueError("length must be a positive integer.")
    splits: list[WalkForwardSplit] = []
    train_stop = plan.training_window
    while train_stop + plan.embargo + plan.test_window <= observations:
        train_start = 0 if plan.anchored else train_stop - plan.training_window
        test_start = train_stop + plan.embargo
        splits.append(
            WalkForwardSplit(
                train_start,
                train_stop,
                test_start,
                test_start + plan.test_window,
            )
        )
        train_stop += plan.step
    if not splits:
        raise ValueError("The series is too short for one requested walk-forward split.")
    return tuple(splits)


def validate_no_lookahead(
    splits: tuple[WalkForwardSplit, ...],
    decisions: BacktestDecisions,
    /,
) -> None:
    """Reject decisions whose information cutoff reaches beyond their train window."""

    if any(not isinstance(split, WalkForwardSplit) for split in splits):
        raise TypeError("splits must contain WalkForwardSplit values.")
    if not isinstance(decisions, BacktestDecisions):
        raise TypeError("decisions must be BacktestDecisions.")
    if decisions.weights.shape[0] != len(splits):
        raise ValueError("There must be one decision per split.")
    information = np.asarray(decisions.information_end_indices)
    for index, split in enumerate(splits):
        if (
            information[index] < split.train_start
            or information[index] >= split.train_stop
        ):
            raise ValueError(
                "Decision information must end inside its training window and before test data."
            )


def evaluate_walk_forward(
    asset_returns: ArrayLike,
    plan: WalkForwardPlan,
    decisions: BacktestDecisions,
    /,
    *,
    initial_wealth: float = 1.0,
) -> WalkForwardResult:
    """Evaluate externally produced decisions strictly on their held-out windows."""

    returns = jnp.asarray(asset_returns)
    if returns.ndim != 2 or 0 in returns.shape:
        raise ValueError("asset_returns must have shape (observation, asset).")
    returns = returns.astype(jnp.result_type(returns.dtype, jnp.float32))
    if not np.all(np.isfinite(np.asarray(returns))):
        raise ValueError("asset_returns must be finite.")
    if decisions.weights.shape[1] != returns.shape[1]:
        raise ValueError("Decision and return asset axes must match.")
    wealth0 = float(initial_wealth)
    if not np.isfinite(wealth0) or wealth0 <= 0.0:
        raise ValueError("initial_wealth must be finite and positive.")
    splits = walk_forward_splits(int(returns.shape[0]), plan)
    validate_no_lookahead(splits, decisions)
    realized: list[Array] = []
    indices: list[Array] = []
    labels: list[Array] = []
    split_returns = []
    for index, split in enumerate(splits):
        segment = returns[split.test_start : split.test_stop] @ decisions.weights[index]
        realized.append(segment)
        indices.append(jnp.arange(split.test_start, split.test_stop, dtype=jnp.int32))
        labels.append(jnp.full((plan.test_window,), index, dtype=jnp.int32))
        split_returns.append(jnp.prod(1.0 + segment) - 1.0)
    out = jnp.concatenate(realized)
    wealth = wealth0 * jnp.cumprod(1.0 + out)
    valid = jnp.all(jnp.isfinite(wealth)) & jnp.all(1.0 + out > 0.0)
    return WalkForwardResult(
        out_of_sample_returns=out,
        wealth=wealth,
        observation_indices=jnp.concatenate(indices),
        split_index=jnp.concatenate(labels),
        split_returns=jnp.stack(split_returns),
        valid=valid,
        splits=splits,
    )


def nested_backtest_splits(
    length: int, plan: NestedBacktestPlan, /
) -> NestedBacktestSplits:
    """Create inner splits contained wholly inside each outer training window."""

    if not isinstance(plan, NestedBacktestPlan):
        raise TypeError("plan must be a NestedBacktestPlan.")
    outer = walk_forward_splits(length, plan.outer)
    all_inner: list[tuple[WalkForwardSplit, ...]] = []
    for split in outer:
        local_length = split.train_stop - split.train_start
        local = walk_forward_splits(local_length, plan.inner)
        translated = tuple(
            WalkForwardSplit(
                item.train_start + split.train_start,
                item.train_stop + split.train_start,
                item.test_start + split.train_start,
                item.test_stop + split.train_start,
            )
            for item in local
        )
        if any(item.test_stop > split.train_stop for item in translated):
            raise ValueError("An inner split escaped its outer training window.")
        all_inner.append(translated)
    return NestedBacktestSplits(outer=outer, inner=tuple(all_inner))


def evaluate_nested_backtest(
    asset_returns: ArrayLike,
    plan: NestedBacktestPlan,
    decisions: NestedBacktestDecisions,
    /,
    *,
    initial_wealth: float = 1.0,
) -> NestedBacktestResult:
    """Select each outer candidate on inner holdouts, then evaluate outer holdouts."""

    if not isinstance(plan, NestedBacktestPlan):
        raise TypeError("plan must be a NestedBacktestPlan.")
    if not isinstance(decisions, NestedBacktestDecisions):
        raise TypeError("decisions must be NestedBacktestDecisions.")
    returns = jnp.asarray(asset_returns)
    if returns.ndim != 2 or 0 in returns.shape:
        raise ValueError("asset_returns must have shape (observation, asset).")
    returns = returns.astype(jnp.result_type(returns.dtype, jnp.float32))
    if not np.all(np.isfinite(np.asarray(returns))):
        raise ValueError("asset_returns must be finite.")
    splits = nested_backtest_splits(int(returns.shape[0]), plan)
    outer_count = len(splits.outer)
    inner_counts = tuple(len(group) for group in splits.inner)
    if len(set(inner_counts)) != 1:
        raise ValueError(
            "Nested decision tensors require equal inner split counts; use rolling outer windows."
        )
    inner_count = inner_counts[0]
    if decisions.inner_weights.shape[:2] != (outer_count, inner_count):
        raise ValueError("Nested decisions do not match the generated split topology.")
    if decisions.inner_weights.shape[-1] != returns.shape[1]:
        raise ValueError("Nested decision and return asset axes must match.")
    if decisions.outer_candidate_weights.shape[0] != outer_count:
        raise ValueError("There must be one outer candidate bank per outer split.")
    inner_scores = np.zeros(
        (outer_count, decisions.inner_weights.shape[2]),
        dtype=np.asarray(returns).dtype,
    )
    selected = np.zeros((outer_count,), dtype=np.int32)
    realized: list[Array] = []
    observation_indices: list[Array] = []
    inner_information = np.asarray(decisions.inner_information_end_indices)
    outer_information = np.asarray(decisions.outer_information_end_indices)
    for outer_index, (outer, inner_group) in enumerate(
        zip(splits.outer, splits.inner, strict=True)
    ):
        for inner_index, inner in enumerate(inner_group):
            for candidate in range(decisions.inner_weights.shape[2]):
                cutoff = int(inner_information[outer_index, inner_index, candidate])
                if cutoff < inner.train_start or cutoff >= inner.train_stop:
                    raise ValueError(
                        "An inner candidate uses information outside its inner training window."
                    )
                path = (
                    returns[inner.test_start : inner.test_stop]
                    @ decisions.inner_weights[outer_index, inner_index, candidate]
                )
                inner_scores[outer_index, candidate] += float(jnp.mean(path))
        inner_scores[outer_index] /= inner_count
        chosen = int(np.argmax(inner_scores[outer_index]))
        selected[outer_index] = chosen
        cutoff = int(outer_information[outer_index, chosen])
        if cutoff < outer.train_start or cutoff >= outer.train_stop:
            raise ValueError(
                "An outer refit uses information outside its outer training window."
            )
        path = (
            returns[outer.test_start : outer.test_stop]
            @ decisions.outer_candidate_weights[outer_index, chosen]
        )
        realized.append(path)
        observation_indices.append(
            jnp.arange(outer.test_start, outer.test_stop, dtype=jnp.int32)
        )
    wealth0 = float(initial_wealth)
    if not np.isfinite(wealth0) or wealth0 <= 0.0:
        raise ValueError("initial_wealth must be finite and positive.")
    out = jnp.concatenate(realized)
    wealth = wealth0 * jnp.cumprod(1.0 + out)
    valid = jnp.all(jnp.isfinite(wealth)) & jnp.all(1.0 + out > 0.0)
    return NestedBacktestResult(
        out_of_sample_returns=out,
        wealth=wealth,
        selected_candidate=jnp.asarray(selected),
        inner_scores=jnp.asarray(inner_scores),
        observation_indices=jnp.concatenate(observation_indices),
        valid=valid,
        splits=splits,
    )


__all__ = [
    "BacktestDecisions",
    "NestedBacktestDecisions",
    "NestedBacktestPlan",
    "NestedBacktestResult",
    "NestedBacktestSplits",
    "WalkForwardPlan",
    "WalkForwardResult",
    "WalkForwardSplit",
    "evaluate_nested_backtest",
    "evaluate_walk_forward",
    "nested_backtest_splits",
    "validate_no_lookahead",
    "walk_forward_splits",
]
