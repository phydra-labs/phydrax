#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TYPE_CHECKING

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax.domain import GridBatch, GridSampling, PointBatch, PointSampling

from ..._doc import DOC_KEY0
from ..._sampling import DesignLike, resolve_design, UnitDesign
from ..._strict import StrictModule
from ._adaptive import (
    _set_batch_rows,
    _single_axis_and_size,
    AbstractCollocationPolicy,
    CollocationPolicy,
    CollocationPopulation,
)
from ._coreset import CoresetCollocationPolicy
from ._separable import HierarchicalAxisPolicy, SeparableCollocationPolicy


if TYPE_CHECKING:
    from phydrax.domain import DomainFunction

    from ._adaptive import PointwiseSamplingTerm


PolicySupportTier = Literal["stable", "conditional"]


@dataclass(frozen=True)
class CollocationDefaults:
    """Recommended robust and adaptive starting points for PINN collocation."""

    sampling_mode: Literal["fixed"] = "fixed"
    sampler: Literal["sobol_scrambled"] = "sobol_scrambled"
    adaptive_policy: Literal["r3"] = "r3"


@dataclass(frozen=True)
class CollocationPolicySupport:
    """Public support status and applicability of one collocation method."""

    name: str
    tier: PolicySupportTier
    applicability: str


RECOMMENDED_COLLOCATION_DEFAULTS = CollocationDefaults()

COLLOCATION_POLICY_SUPPORT: Mapping[str, CollocationPolicySupport] = MappingProxyType(
    {
        "fixed_sobol": CollocationPolicySupport(
            "fixed_sobol", "stable", "Robust unconditional paired-point default."
        ),
        "periodic": CollocationPolicySupport(
            "periodic", "stable", "Unscored periodic replacement at fixed capacity."
        ),
        "r3": CollocationPolicySupport(
            "r3", "stable", "General opt-in adaptive paired-point policy."
        ),
        "rar_d": CollocationPolicySupport(
            "rar_d", "conditional", "Oscillatory or distributed residual structure."
        ),
        "coreset": CollocationPolicySupport(
            "coreset",
            "conditional",
            "Residual importance with kernel-diverse paired support.",
        ),
        "periodic_separable": CollocationPolicySupport(
            "periodic_separable", "stable", "Coordinate-separable models."
        ),
        "hierarchical_axes": CollocationPolicySupport(
            "hierarchical_axes",
            "conditional",
            "Nested coordinate-separable discretizations.",
        ),
    }
)


def collocation_policy_support(
    policy: str | AbstractCollocationPolicy | None,
    /,
) -> CollocationPolicySupport:
    """Return the declared support tier for a retained collocation method."""
    if policy is None:
        name = "fixed_sobol"
    elif isinstance(policy, str):
        name = policy
    elif isinstance(policy, ControlledCollocationPolicy):
        return collocation_policy_support(policy.base_policy)
    elif isinstance(policy, CollocationPolicy):
        name = policy.algorithm
    elif isinstance(policy, CoresetCollocationPolicy):
        name = "coreset"
    elif isinstance(policy, HierarchicalAxisPolicy):
        name = "hierarchical_axes"
    elif isinstance(policy, SeparableCollocationPolicy):
        name = "periodic_separable"
    else:
        raise TypeError(f"No support declaration exists for {type(policy).__name__}.")
    if name not in COLLOCATION_POLICY_SUPPORT:
        raise ValueError(f"Unknown collocation method {name!r}.")
    return COLLOCATION_POLICY_SUPPORT[name]


class RefreshSchedule(StrictModule):
    """Host-side fixed-interval adaptive refresh schedule."""

    every: int
    start_at: int

    def __init__(self, every: int, *, start_at: int = 1):
        if int(every) <= 0:
            raise ValueError("RefreshSchedule.every must be positive.")
        if int(start_at) <= 0:
            raise ValueError("RefreshSchedule.start_at must be positive.")
        self.every = int(every)
        self.start_at = int(start_at)

    def due(self, last_attempt: Array, iter_: int | Array, /) -> Array:
        step = jnp.asarray(iter_, dtype=jnp.int32)
        first = step >= self.start_at
        interval = (step - jnp.asarray(last_attempt, dtype=jnp.int32)) >= self.every
        return first & interval


class ResidualMonitor(StrictModule):
    """Configuration for an independent fixed collocation monitor population."""

    sampler: UnitDesign
    epsilon: Array

    def __init__(
        self,
        *,
        sampler: DesignLike = "sobol_scrambled",
        epsilon: float = 1e-12,
    ):
        if float(epsilon) <= 0.0:
            raise ValueError("ResidualMonitor.epsilon must be positive.")
        self.sampler = resolve_design(sampler)
        self.epsilon = jnp.asarray(epsilon, dtype=float)


class RefreshGuard(StrictModule):
    """Acceptance and suspension rules for proposed population refreshes."""

    max_relative_regression: Array
    absolute_tolerance: Array
    max_consecutive_rejections: int
    suspension_steps: int

    def __init__(
        self,
        *,
        max_relative_regression: float = 0.0,
        absolute_tolerance: float = 0.0,
        max_consecutive_rejections: int = 2,
        suspension_steps: int = 100,
    ):
        if float(max_relative_regression) < 0.0:
            raise ValueError("max_relative_regression must be non-negative.")
        if float(absolute_tolerance) < 0.0:
            raise ValueError("absolute_tolerance must be non-negative.")
        if int(max_consecutive_rejections) <= 0:
            raise ValueError("max_consecutive_rejections must be positive.")
        if int(suspension_steps) < 0:
            raise ValueError("suspension_steps must be non-negative.")
        self.max_relative_regression = jnp.asarray(
            max_relative_regression, dtype=float
        )
        self.absolute_tolerance = jnp.asarray(absolute_tolerance, dtype=float)
        self.max_consecutive_rejections = int(max_consecutive_rejections)
        self.suspension_steps = int(suspension_steps)

    def accepts(self, baseline: Array, monitored: Array, /) -> Array:
        limit = (
            jnp.asarray(baseline, dtype=float)
            * (1.0 + self.max_relative_regression)
            + self.absolute_tolerance
        )
        return jnp.asarray(monitored, dtype=float) <= limit


class AdaptationBudget(StrictModule):
    """Limits that stop further refreshes once an evaluation budget is exhausted."""

    max_refresh_attempts: int | None
    max_candidate_evaluations: int | None
    max_monitor_evaluations: int | None
    max_training_evaluations: int | None

    def __init__(
        self,
        *,
        max_refresh_attempts: int | None = None,
        max_candidate_evaluations: int | None = None,
        max_monitor_evaluations: int | None = None,
        max_training_evaluations: int | None = None,
    ):
        values = (
            max_refresh_attempts,
            max_candidate_evaluations,
            max_monitor_evaluations,
            max_training_evaluations,
        )
        if any(value is not None and int(value) < 0 for value in values):
            raise ValueError("Adaptation budget limits must be non-negative.")
        self.max_refresh_attempts = (
            None if max_refresh_attempts is None else int(max_refresh_attempts)
        )
        self.max_candidate_evaluations = (
            None
            if max_candidate_evaluations is None
            else int(max_candidate_evaluations)
        )
        self.max_monitor_evaluations = (
            None if max_monitor_evaluations is None else int(max_monitor_evaluations)
        )
        self.max_training_evaluations = (
            None
            if max_training_evaluations is None
            else int(max_training_evaluations)
        )


class CoverageAnchors(StrictModule):
    """Persistent low-discrepancy coverage floor for paired point populations."""

    fraction: Array

    def __init__(self, fraction: float = 0.25):
        if not 0.0 <= float(fraction) < 1.0:
            raise ValueError("CoverageAnchors.fraction must lie in [0, 1).")
        self.fraction = jnp.asarray(fraction, dtype=float)


class ControlledCollocationPopulation(StrictModule):
    """Training, monitor, rollback, and accounting state for adaptive control."""

    current: Any
    rollback: Any
    anchor_reference: Any | None
    monitor_batch: PointBatch | GridBatch
    monitor_point_count: int
    last_control_step: Array
    proposal_pending: Array
    baseline_monitor_mean: Array
    monitor_mean: Array
    monitor_rms: Array
    monitor_max: Array
    refresh_attempt_count: Array
    refresh_accept_count: Array
    refresh_reject_count: Array
    consecutive_rejections: Array
    suspended_until: Array
    candidate_evaluations: Array
    monitor_evaluations: Array
    training_evaluations: Array

    def __init__(
        self,
        current: Any,
        rollback: Any,
        monitor_batch: PointBatch | GridBatch,
        monitor_point_count: int,
        *,
        anchor_reference: Any | None = None,
        last_control_step: int | Array = 0,
        proposal_pending: bool | Array = False,
        baseline_monitor_mean: float | Array = 0.0,
        monitor_mean: float | Array = 0.0,
        monitor_rms: float | Array = 0.0,
        monitor_max: float | Array = 0.0,
        refresh_attempt_count: int | Array = 0,
        refresh_accept_count: int | Array = 0,
        refresh_reject_count: int | Array = 0,
        consecutive_rejections: int | Array = 0,
        suspended_until: int | Array = 0,
        candidate_evaluations: int | Array = 0,
        monitor_evaluations: int | Array = 0,
        training_evaluations: int | Array = 0,
    ):
        self.current = current
        self.rollback = rollback
        self.anchor_reference = anchor_reference
        self.monitor_batch = monitor_batch
        self.monitor_point_count = int(monitor_point_count)
        self.last_control_step = jnp.asarray(last_control_step, dtype=jnp.int32)
        self.proposal_pending = jnp.asarray(proposal_pending, dtype=bool)
        self.baseline_monitor_mean = jnp.asarray(
            baseline_monitor_mean, dtype=float
        )
        self.monitor_mean = jnp.asarray(monitor_mean, dtype=float)
        self.monitor_rms = jnp.asarray(monitor_rms, dtype=float)
        self.monitor_max = jnp.asarray(monitor_max, dtype=float)
        self.refresh_attempt_count = jnp.asarray(
            refresh_attempt_count, dtype=jnp.int32
        )
        self.refresh_accept_count = jnp.asarray(
            refresh_accept_count, dtype=jnp.int32
        )
        self.refresh_reject_count = jnp.asarray(
            refresh_reject_count, dtype=jnp.int32
        )
        self.consecutive_rejections = jnp.asarray(
            consecutive_rejections, dtype=jnp.int32
        )
        self.suspended_until = jnp.asarray(suspended_until, dtype=jnp.int32)
        self.candidate_evaluations = jnp.asarray(
            candidate_evaluations, dtype=jnp.int32
        )
        self.monitor_evaluations = jnp.asarray(
            monitor_evaluations, dtype=jnp.int32
        )
        self.training_evaluations = jnp.asarray(
            training_evaluations, dtype=jnp.int32
        )

    @property
    def refresh_count(self) -> Array:
        return self.refresh_attempt_count

    @property
    def last_refresh(self) -> Array:
        return self.last_control_step


class ControlledCollocationPolicy(AbstractCollocationPolicy):
    """Validation-gated controller around any adaptive collocation policy.

    Population proposals remain policy-local. This controller owns scheduling,
    a fixed monitor population, proposal acceptance, suspension, coverage anchors,
    and logical residual-evaluation accounting.
    """

    base_policy: AbstractCollocationPolicy
    schedule: RefreshSchedule
    monitor: ResidualMonitor
    guard: RefreshGuard
    budget: AdaptationBudget
    anchors: CoverageAnchors
    refresh_every: int

    def __init__(
        self,
        base_policy: AbstractCollocationPolicy,
        *,
        schedule: RefreshSchedule | None = None,
        monitor: ResidualMonitor | None = None,
        guard: RefreshGuard | None = None,
        budget: AdaptationBudget | None = None,
        anchors: CoverageAnchors | None = None,
    ):
        if isinstance(base_policy, ControlledCollocationPolicy):
            raise TypeError("ControlledCollocationPolicy cannot wrap another controller.")
        if schedule is None:
            start_at = (
                base_policy.start_at
                if isinstance(base_policy, CoresetCollocationPolicy)
                else 1
            )
            schedule = RefreshSchedule(base_policy.refresh_every, start_at=start_at)
        self.base_policy = base_policy
        self.schedule = schedule
        self.monitor = ResidualMonitor() if monitor is None else monitor
        self.guard = RefreshGuard() if guard is None else guard
        self.budget = AdaptationBudget() if budget is None else budget
        self.anchors = CoverageAnchors(0.0) if anchors is None else anchors
        self.refresh_every = schedule.every

    def initialize(
        self,
        constraint: PointwiseSamplingTerm,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> ControlledCollocationPopulation:
        current = self.base_policy.initialize(constraint, key=jr.fold_in(key, 1))
        monitor_batch = _sample_monitor_batch(
            constraint,
            sampler=self.monitor.sampler,
            key=jr.fold_in(key, 2),
        )
        point_count = _population_logical_count(self.base_policy, current)
        anchor_reference = (
            current if float(self.anchors.fraction) > 0.0 else None
        )
        if anchor_reference is not None:
            _validate_anchor_population(current)
        return ControlledCollocationPopulation(
            current,
            current,
            monitor_batch,
            point_count,
            anchor_reference=anchor_reference,
        )

    def should_refresh(
        self,
        population: ControlledCollocationPopulation,
        iter_: int | Array,
    ) -> Array:
        step = jnp.asarray(iter_, dtype=jnp.int32)
        allowed = bool(step >= population.suspended_until)
        if bool(population.proposal_pending):
            allowed = allowed and _monitor_budget_available(
                self, population, monitor_batches=1
            )
        else:
            allowed = allowed and _within_budget(
                self,
                population,
                self.base_policy.refresh_residual_evaluations(population.current),
                monitor_batches=2,
            )
        return self.schedule.due(population.last_control_step, step) & allowed

    def loss_batch_and_weight(
        self,
        population: ControlledCollocationPopulation,
        /,
    ) -> tuple[Any, cx.Field | None]:
        return self.base_policy.loss_batch_and_weight(population.current)

    def data_metrics(
        self,
        population: ControlledCollocationPopulation,
        /,
    ) -> dict[str, Array]:
        metrics = dict(self.base_policy.data_metrics(population.current))
        metrics.update(
            {
                "control_monitor_mean": population.monitor_mean,
                "control_monitor_rms": population.monitor_rms,
                "control_monitor_max": population.monitor_max,
                "control_refresh_attempt_count": jnp.asarray(
                    population.refresh_attempt_count, dtype=float
                ),
                "control_refresh_accept_count": jnp.asarray(
                    population.refresh_accept_count, dtype=float
                ),
                "control_refresh_reject_count": jnp.asarray(
                    population.refresh_reject_count, dtype=float
                ),
                "control_consecutive_rejections": jnp.asarray(
                    population.consecutive_rejections, dtype=float
                ),
                "control_suspended_until": jnp.asarray(
                    population.suspended_until, dtype=float
                ),
                "control_candidate_evaluations": jnp.asarray(
                    population.candidate_evaluations, dtype=float
                ),
                "control_monitor_evaluations": jnp.asarray(
                    population.monitor_evaluations, dtype=float
                ),
                "control_training_evaluations": jnp.asarray(
                    population.training_evaluations, dtype=float
                ),
                "control_anchor_fraction": jnp.asarray(
                    self.anchors.fraction, dtype=float
                ),
                "control_proposal_pending": jnp.asarray(
                    population.proposal_pending, dtype=float
                ),
            }
        )
        return metrics

    def refresh_residual_evaluations(
        self,
        population: ControlledCollocationPopulation,
        /,
    ) -> int:
        return (
            population.monitor_point_count
            + self.base_policy.refresh_residual_evaluations(population.current)
        )

    def record_training_evaluation(
        self,
        population: ControlledCollocationPopulation,
        /,
        *,
        multiplier: int = 1,
    ) -> ControlledCollocationPopulation:
        if int(multiplier) <= 0:
            raise ValueError("Training evaluation multiplier must be positive.")
        count = (
            _population_logical_count(self.base_policy, population.current)
            * int(multiplier)
        )
        return _replace_controlled(
            population,
            training_evaluations=population.training_evaluations + count,
        )

    def refresh(
        self,
        constraint: PointwiseSamplingTerm,
        functions: Mapping[str, DomainFunction],
        population: ControlledCollocationPopulation,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array,
    ) -> ControlledCollocationPopulation:
        step = jnp.asarray(iter_, dtype=jnp.int32)
        mean, rms, maximum, monitor_count = _monitor_statistics(
            constraint,
            functions,
            population.monitor_batch,
            key=jr.fold_in(key, 1),
            epsilon=self.monitor.epsilon,
        )
        monitor_evaluations = population.monitor_evaluations + monitor_count
        if bool(population.proposal_pending) and not bool(
            self.guard.accepts(population.baseline_monitor_mean, mean)
        ):
            consecutive = population.consecutive_rejections + 1
            suspended_until = population.suspended_until
            if int(consecutive) >= self.guard.max_consecutive_rejections:
                suspended_until = step + self.guard.suspension_steps
            return _replace_controlled(
                population,
                current=population.rollback,
                rollback=population.rollback,
                last_control_step=step,
                proposal_pending=False,
                monitor_mean=mean,
                monitor_rms=rms,
                monitor_max=maximum,
                refresh_reject_count=population.refresh_reject_count + 1,
                consecutive_rejections=consecutive,
                suspended_until=suspended_until,
                monitor_evaluations=monitor_evaluations,
            )

        accepted_count = population.refresh_accept_count
        if bool(population.proposal_pending):
            accepted_count = accepted_count + 1
        candidate_count = self.base_policy.refresh_residual_evaluations(
            population.current
        )
        monitored = _replace_controlled(
            population,
            rollback=population.current,
            last_control_step=step,
            proposal_pending=False,
            baseline_monitor_mean=mean,
            monitor_mean=mean,
            monitor_rms=rms,
            monitor_max=maximum,
            refresh_accept_count=accepted_count,
            consecutive_rejections=0,
            monitor_evaluations=monitor_evaluations,
        )
        if not _within_budget(
            self,
            monitored,
            candidate_count,
            monitor_batches=1,
        ):
            return monitored
        proposal = self.base_policy.refresh(
            constraint,
            functions,
            population.current,
            key=jr.fold_in(key, 2),
            iter_=step,
        )
        if population.anchor_reference is not None:
            proposal = _inject_coverage_anchors(
                proposal,
                population.anchor_reference,
                fraction=float(self.anchors.fraction),
            )
        return _replace_controlled(
            monitored,
            current=proposal,
            rollback=population.current,
            proposal_pending=True,
            refresh_attempt_count=population.refresh_attempt_count + 1,
            candidate_evaluations=(
                population.candidate_evaluations + candidate_count
            ),
        )

    def settle(
        self,
        constraint: PointwiseSamplingTerm,
        functions: Mapping[str, DomainFunction],
        population: ControlledCollocationPopulation,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array,
    ) -> ControlledCollocationPopulation:
        """Validate the terminal proposal without admitting another proposal."""
        if not bool(population.proposal_pending):
            return population
        step = jnp.asarray(iter_, dtype=jnp.int32)
        mean, rms, maximum, monitor_count = _monitor_statistics(
            constraint,
            functions,
            population.monitor_batch,
            key=key,
            epsilon=self.monitor.epsilon,
        )
        monitor_evaluations = population.monitor_evaluations + monitor_count
        if not bool(
            self.guard.accepts(population.baseline_monitor_mean, mean)
        ):
            consecutive = population.consecutive_rejections + 1
            suspended_until = population.suspended_until
            if int(consecutive) >= self.guard.max_consecutive_rejections:
                suspended_until = step + self.guard.suspension_steps
            return _replace_controlled(
                population,
                current=population.rollback,
                rollback=population.rollback,
                last_control_step=step,
                proposal_pending=False,
                monitor_mean=mean,
                monitor_rms=rms,
                monitor_max=maximum,
                refresh_reject_count=population.refresh_reject_count + 1,
                consecutive_rejections=consecutive,
                suspended_until=suspended_until,
                monitor_evaluations=monitor_evaluations,
            )
        return _replace_controlled(
            population,
            rollback=population.current,
            last_control_step=step,
            proposal_pending=False,
            baseline_monitor_mean=mean,
            monitor_mean=mean,
            monitor_rms=rms,
            monitor_max=maximum,
            refresh_accept_count=population.refresh_accept_count + 1,
            consecutive_rejections=0,
            monitor_evaluations=monitor_evaluations,
        )


def controlled_collocation(
    policy: AbstractCollocationPolicy,
    /,
    **kwargs: Any,
) -> ControlledCollocationPolicy:
    """Wrap ``policy`` in validation-gated adaptive control."""
    return ControlledCollocationPolicy(policy, **kwargs)


def _sample_monitor_batch(
    constraint: PointwiseSamplingTerm,
    /,
    *,
    sampler: DesignLike,
    key: Key[Array, ""],
) -> PointBatch | GridBatch:
    sampling = constraint.sampling
    if isinstance(sampling, GridSampling):
        dense = (
            None
            if sampling.dense is None
            else PointSampling(
                sampling.dense.count,
                layout=sampling.dense.layout,
                design=sampler,
            )
        )
        monitor_sampling = GridSampling(
            sampling.axes,
            dense=dense,
            design=sampler,
        )
    elif isinstance(sampling, PointSampling):
        monitor_sampling = PointSampling(
            sampling.count,
            layout=sampling.layout,
            design=sampler,
        )
    else:
        raise TypeError("Adaptive monitors require one sampling plan.")
    batch = constraint.component.sample(monitor_sampling, key=key)
    if isinstance(batch, tuple):
        raise TypeError("Adaptive monitors require one structured batch.")
    return batch


def _monitor_statistics(
    constraint: PointwiseSamplingTerm,
    functions: Mapping[str, DomainFunction],
    batch: PointBatch | GridBatch,
    /,
    *,
    key: Key[Array, ""],
    epsilon: Array,
) -> tuple[Array, Array, Array, int]:
    score = constraint.pointwise_score(functions, batch, key=key)
    values = jax.lax.stop_gradient(jnp.asarray(score.data, dtype=float))
    values = jnp.nan_to_num(
        values,
        nan=jnp.finfo(values.dtype).max,
        posinf=jnp.finfo(values.dtype).max,
        neginf=0.0,
    )
    values = jnp.maximum(values, 0.0)
    mean = jnp.mean(values)
    return mean, jnp.sqrt(jnp.maximum(mean, epsilon)), jnp.max(values), values.size


def _population_logical_count(
    policy: AbstractCollocationPolicy,
    population: Any,
    /,
) -> int:
    metrics = policy.data_metrics(population)
    if "logical_point_count" in metrics:
        return int(metrics["logical_point_count"])
    if "point_count" in metrics:
        return int(metrics["point_count"])
    raise TypeError(
        f"{type(policy).__name__}.data_metrics() must expose point_count or "
        "logical_point_count for controlled accounting."
    )


def _monitor_budget_available(
    policy: ControlledCollocationPolicy,
    population: ControlledCollocationPopulation,
    /,
    *,
    monitor_batches: int,
) -> bool:
    maximum = policy.budget.max_monitor_evaluations
    return maximum is None or (
        int(population.monitor_evaluations)
        + int(monitor_batches) * population.monitor_point_count
        <= maximum
    )


def _within_budget(
    policy: ControlledCollocationPolicy,
    population: ControlledCollocationPopulation,
    candidate_count: int,
    /,
    *,
    monitor_batches: int,
) -> bool:
    budget = policy.budget
    if (
        budget.max_refresh_attempts is not None
        and int(population.refresh_attempt_count) >= budget.max_refresh_attempts
    ):
        return False
    if (
        budget.max_candidate_evaluations is not None
        and int(population.candidate_evaluations) + candidate_count
        > budget.max_candidate_evaluations
    ):
        return False
    if not _monitor_budget_available(
        policy, population, monitor_batches=monitor_batches
    ):
        return False
    if (
        budget.max_training_evaluations is not None
        and int(population.training_evaluations) >= budget.max_training_evaluations
    ):
        return False
    return True


def _validate_anchor_population(population: Any, /) -> None:
    if not isinstance(population, CollocationPopulation):
        raise TypeError("Coverage anchors require a paired CollocationPopulation.")


def _inject_coverage_anchors(
    population: Any,
    reference: Any,
    /,
    *,
    fraction: float,
) -> Any:
    _validate_anchor_population(population)
    if not isinstance(reference, CollocationPopulation):
        raise TypeError("Paired anchor state must use matching populations.")
    return _inject_collocation_anchors(
        population,
        reference,
        fraction=fraction,
    )


def _inject_collocation_anchors(
    population: CollocationPopulation,
    reference: CollocationPopulation,
    /,
    *,
    fraction: float,
) -> CollocationPopulation:
    axis, size = _single_axis_and_size(population.batch)
    reference_axis, reference_size = _single_axis_and_size(reference.batch)
    if axis != reference_axis or size != reference_size:
        raise ValueError("Coverage anchor and proposal populations must have equal shape.")
    count = min(size - 1, max(1, int(round(size * fraction))))
    indices = jnp.arange(count)
    anchored_batch = _set_batch_rows(
        population.batch,
        indices,
        _take_first_rows(reference.batch, count),
    )
    active = population.active
    if active is not None:
        active_data = jnp.asarray(active.data)
        active = cx.Field(active_data.at[:count].set(1.0), dims=active.dims)
    age_data = jnp.asarray(population.age.data)
    reference_age = jnp.asarray(reference.age.data)
    age = cx.Field(
        age_data.at[:count].set(reference_age[:count] + 1),
        dims=population.age.dims,
    )
    anchored = eqx.tree_at(lambda state: state.batch, population, anchored_batch)
    anchored = eqx.tree_at(lambda state: state.age, anchored, age)
    if active is not None:
        anchored = eqx.tree_at(lambda state: state.active, anchored, active)
    return anchored


def _take_first_rows(batch: PointBatch, count: int, /) -> PointBatch:
    axis, _ = _single_axis_and_size(batch)

    def take(value: Any) -> Any:
        if not isinstance(value, cx.Field) or axis not in value.named_dims:
            return value
        position = value.dims.index(axis)
        indices = [slice(None)] * value.data.ndim
        indices[position] = slice(0, count)
        return cx.Field(value.data[tuple(indices)], dims=value.dims)

    points = jax.tree_util.tree_map(
        take,
        batch.points,
        is_leaf=lambda value: isinstance(value, cx.Field),
    )
    metadata = jax.tree_util.tree_map(
        take,
        batch.metadata,
        is_leaf=lambda value: isinstance(value, cx.Field),
    )
    return PointBatch(points, batch.structure, metadata=metadata)


def _replace_controlled(
    population: ControlledCollocationPopulation,
    /,
    **updates: Any,
) -> ControlledCollocationPopulation:
    return ControlledCollocationPopulation(
        updates.get("current", population.current),
        updates.get("rollback", population.rollback),
        updates.get("monitor_batch", population.monitor_batch),
        updates.get("monitor_point_count", population.monitor_point_count),
        anchor_reference=updates.get(
            "anchor_reference", population.anchor_reference
        ),
        last_control_step=updates.get(
            "last_control_step", population.last_control_step
        ),
        proposal_pending=updates.get(
            "proposal_pending", population.proposal_pending
        ),
        baseline_monitor_mean=updates.get(
            "baseline_monitor_mean", population.baseline_monitor_mean
        ),
        monitor_mean=updates.get("monitor_mean", population.monitor_mean),
        monitor_rms=updates.get("monitor_rms", population.monitor_rms),
        monitor_max=updates.get("monitor_max", population.monitor_max),
        refresh_attempt_count=updates.get(
            "refresh_attempt_count", population.refresh_attempt_count
        ),
        refresh_accept_count=updates.get(
            "refresh_accept_count", population.refresh_accept_count
        ),
        refresh_reject_count=updates.get(
            "refresh_reject_count", population.refresh_reject_count
        ),
        consecutive_rejections=updates.get(
            "consecutive_rejections", population.consecutive_rejections
        ),
        suspended_until=updates.get(
            "suspended_until", population.suspended_until
        ),
        candidate_evaluations=updates.get(
            "candidate_evaluations", population.candidate_evaluations
        ),
        monitor_evaluations=updates.get(
            "monitor_evaluations", population.monitor_evaluations
        ),
        training_evaluations=updates.get(
            "training_evaluations", population.training_evaluations
        ),
    )


__all__ = [
    "AdaptationBudget",
    "COLLOCATION_POLICY_SUPPORT",
    "CollocationDefaults",
    "CollocationPolicySupport",
    "ControlledCollocationPolicy",
    "ControlledCollocationPopulation",
    "CoverageAnchors",
    "PolicySupportTier",
    "RECOMMENDED_COLLOCATION_DEFAULTS",
    "RefreshGuard",
    "RefreshSchedule",
    "ResidualMonitor",
    "collocation_policy_support",
    "controlled_collocation",
]
