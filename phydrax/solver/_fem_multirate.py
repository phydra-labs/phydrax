#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class DGMultirateTracePlan(StrictModule, NonTrainableState):
    facet_levels: Array
    maximum_level: int = eqx.field(static=True)
    history_depth: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        facet_levels: ArrayLike,
        /,
        *,
        history_depth: int = 3,
    ):
        levels = jnp.asarray(facet_levels, dtype=jnp.int32)
        depth = int(history_depth)
        if levels.ndim != 2 or levels.shape[1] != 2 or levels.size == 0:
            raise ValueError("DG multirate facet levels require shape (facets, 2).")
        if bool(jnp.any(levels < 0)) or depth < 1:
            raise ValueError("DG multirate levels/history depth are invalid.")
        maximum = int(jnp.max(levels))
        self.facet_levels = levels
        self.maximum_level = maximum
        self.history_depth = depth
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dg-multirate-trace-plan",
                "facet_shape": list(levels.shape),
                "maximum_level": maximum,
                "history_depth": depth,
            }
        )

    @property
    def ticks_per_macro_step(self) -> int:
        return 1 << self.maximum_level

    def active_level(self, tick: int, /) -> int:
        tick_ = int(tick)
        if tick_ < 0 or tick_ >= self.ticks_per_macro_step:
            raise ValueError("DG multirate tick is out of bounds.")
        level = 0
        while level < self.maximum_level and tick_ % (1 << (level + 1)) == 0:
            level += 1
        return level


class DGTraceHistory(StrictModule):
    values: Array
    times: Array
    effective_depth: Array

    def __init__(
        self,
        values: ArrayLike,
        times: ArrayLike,
        effective_depth: ArrayLike,
        /,
    ):
        values_ = jnp.asarray(values)
        times_ = jnp.asarray(times)
        effective = jnp.asarray(effective_depth, dtype=jnp.int32)
        if (
            values_.ndim < 2
            or times_.shape != (values_.shape[0],)
            or effective.shape != ()
        ):
            raise ValueError("DG trace history layouts are incompatible.")
        effective = eqx.error_if(
            effective,
            (effective < 0) | (effective > values_.shape[0]),
            "DG trace history depth is out of bounds.",
        )
        self.values = values_
        self.times = times_
        self.effective_depth = effective

    @classmethod
    def empty(
        cls,
        depth: int,
        trace_shape: tuple[int, ...],
        dtype,
        /,
    ) -> DGTraceHistory:
        return cls(
            jnp.zeros((int(depth),) + tuple(trace_shape), dtype=dtype),
            jnp.full((int(depth),), jnp.nan),
            0,
        )

    def update(
        self,
        value: ArrayLike,
        time: ArrayLike,
        /,
        *,
        accepted: ArrayLike = True,
    ) -> DGTraceHistory:
        value_ = jnp.asarray(value)
        time_ = jnp.asarray(time)
        accepted_ = jnp.asarray(accepted, dtype=bool)
        if (
            value_.shape != self.values.shape[1:]
            or time_.shape != ()
            or accepted_.shape != ()
        ):
            raise ValueError("DG trace update value/time/acceptance are incompatible.")

        def perform(history):
            values = jnp.concatenate((value_[None], history.values[:-1]), axis=0)
            times = jnp.concatenate((time_[None], history.times[:-1]), axis=0)
            return DGTraceHistory(
                values,
                times,
                jnp.minimum(history.effective_depth + 1, history.values.shape[0]),
            )

        return jax.lax.cond(accepted_, perform, lambda history: history, self)

    def predict(self, time: ArrayLike, /) -> Array:
        target = jnp.asarray(time)
        if target.shape != ():
            raise ValueError("DG trace prediction time must be scalar.")
        count = self.values.shape[0]
        valid = jnp.arange(count) < self.effective_depth
        safe_times = jnp.where(valid, self.times, target + jnp.arange(count) + 1.0)
        weights = []
        for index in range(count):
            numerator = jnp.asarray(1.0, dtype=target.dtype)
            denominator = jnp.asarray(1.0, dtype=target.dtype)
            for other in range(count):
                if other != index:
                    numerator = numerator * jnp.where(
                        valid[other], target - safe_times[other], 1.0
                    )
                    denominator = denominator * jnp.where(
                        valid[other],
                        safe_times[index] - safe_times[other],
                        1.0,
                    )
            weights.append(jnp.where(valid[index], numerator / denominator, 0.0))
        weights_ = jnp.stack(tuple(weights))
        predicted = jnp.tensordot(weights_, self.values, axes=((0,), (0,)))
        return jnp.where(self.effective_depth > 0, predicted, jnp.zeros_like(predicted))


class DGInterfaceFluxResult(StrictModule):
    plus: Array
    minus: Array
    conservation_defect: Array


def conservative_multirate_flux(
    plus_trace: ArrayLike,
    minus_trace: ArrayLike,
    normal: ArrayLike,
    flux: Callable,
    /,
) -> DGInterfaceFluxResult:
    if not callable(flux):
        raise TypeError("flux must be callable.")
    numerical = jnp.asarray(flux(plus_trace, minus_trace, normal))
    plus = numerical
    minus = -numerical
    defect = jnp.sqrt(jnp.sum(jnp.abs(plus + minus) ** 2))
    return DGInterfaceFluxResult(plus=plus, minus=minus, conservation_defect=defect)


class TimeSlabFluxLedger(StrictModule):
    integrated_flux: Array
    accumulated_duration: Array
    expected_duration: Array
    complete: Array
    ledger_id: str = eqx.field(static=True)

    def __init__(
        self,
        integrated_flux: ArrayLike,
        accumulated_duration: ArrayLike,
        expected_duration: ArrayLike,
        /,
        *,
        ledger_id: str,
    ):
        flux = jnp.asarray(integrated_flux)
        accumulated = jnp.asarray(accumulated_duration)
        expected = jnp.asarray(expected_duration)
        if (
            flux.ndim < 1
            or accumulated.shape != ()
            or expected.shape != ()
            or not str(ledger_id)
        ):
            raise ValueError("Time-slab flux ledger shapes or ID are invalid.")
        self.integrated_flux = flux
        self.accumulated_duration = accumulated
        self.expected_duration = expected
        self.complete = jnp.isclose(accumulated, expected)
        self.ledger_id = str(ledger_id)

    @classmethod
    def zeros(
        cls,
        route_count: int,
        component_shape: tuple[int, ...],
        duration: ArrayLike,
        /,
        *,
        ledger_id: str,
        dtype=float,
    ) -> "TimeSlabFluxLedger":
        return cls(
            jnp.zeros((int(route_count),) + tuple(component_shape), dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(duration, dtype=dtype),
            ledger_id=ledger_id,
        )

    def add_substep(
        self, flux_rate: ArrayLike, step_size: ArrayLike, /
    ) -> "TimeSlabFluxLedger":
        flux = jnp.asarray(flux_rate)
        step = jnp.asarray(step_size)
        if flux.shape != self.integrated_flux.shape or step.shape != ():
            raise ValueError("Time-slab substep flux or step shape changed.")
        accumulated = self.accumulated_duration + step
        return TimeSlabFluxLedger(
            self.integrated_flux + step * flux,
            accumulated,
            self.expected_duration,
            ledger_id=self.ledger_id,
        )

    def equal_opposite_contributions(self, /) -> DGInterfaceFluxResult:
        defect = jnp.sqrt(
            jnp.sum(jnp.abs(self.integrated_flux - self.integrated_flux) ** 2)
        )
        return DGInterfaceFluxResult(
            self.integrated_flux,
            -self.integrated_flux,
            defect,
        )


class ConservativeLocalTimeStepPlan(StrictModule, NonTrainableState):
    cell_levels: Array
    macro_step_size: float = eqx.field(static=True)
    maximum_level: int = eqx.field(static=True)
    substep_count: int = eqx.field(static=True)
    trace_plan: DGMultirateTracePlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_levels: ArrayLike,
        macro_step_size: float,
        trace_plan: DGMultirateTracePlan,
        /,
    ):
        levels = jnp.asarray(cell_levels, dtype=jnp.int32)
        step = float(macro_step_size)
        if (
            levels.ndim != 1
            or levels.size == 0
            or float(jnp.min(levels)) < 0
            or not 0.0 < step
            or not isinstance(trace_plan, DGMultirateTracePlan)
        ):
            raise ValueError("Local time-step levels, macro step, or trace plan invalid.")
        maximum = int(jnp.max(levels))
        self.cell_levels = levels
        self.macro_step_size = step
        self.maximum_level = maximum
        self.substep_count = 1 << maximum
        self.trace_plan = trace_plan
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-local-time-step-plan",
                "levels": tuple(int(value) for value in levels),
                "macro_step_size": step,
                "trace_plan": trace_plan.plan_id,
            }
        )

    def cell_step_sizes(self, /) -> Array:
        return self.macro_step_size / (2.0**self.cell_levels)

    def active_cells(self, substep: int, /) -> Array:
        index = int(substep)
        if index < 0 or index >= self.substep_count:
            raise ValueError("Local time substep is out of range.")
        stride = 2 ** (self.maximum_level - self.cell_levels)
        return (index % stride) == 0

    def synchronized(self, substep: int, /) -> bool:
        return int(substep) == self.substep_count - 1


__all__ = [
    "DGInterfaceFluxResult",
    "ConservativeLocalTimeStepPlan",
    "DGMultirateTracePlan",
    "DGTraceHistory",
    "conservative_multirate_flux",
    "TimeSlabFluxLedger",
]
