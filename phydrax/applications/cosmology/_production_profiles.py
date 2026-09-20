#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed fixed-step production adapters for bounded dark-matter profiles."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...solver._fixed_step import (
    AbstractFixedStepMethod,
    FixedStepResult,
    RobustRetryPolicy,
)
from ...solver._production_runtime import (
    PreparedProductionRun,
    ProductionRunPlan,
    ProductionRunState,
    ProductionTriggerBinding,
)
from ...solver._runtime_lifecycle import ExactTimeSchedule, StreamingMomentPlan
from ._background import FLRWBackground
from ._particles import CosmologicalParticleState
from ._sidm import CosmologicalSIDMPlan
from ._wave_dark_matter import PreparedPeriodicWaveDarkMatter, WaveDarkMatterState


def _fixed_schedule(
    scale_factors: ArrayLike,
    profile_name: str,
    /,
) -> tuple[Array, float, int, float]:
    values = np.asarray(scale_factors)
    if values.ndim != 1 or values.size < 2 or not np.issubdtype(values.dtype, np.inexact):
        raise ValueError("Production scale-factor schedule must be an inexact vector.")
    differences = np.diff(values.astype("float64"))
    if (
        np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
        or np.any(differences <= 0.0)
    ):
        raise ValueError(
            "Production scale factors must be finite, positive, and increasing."
        )
    step = float(differences[0])
    tolerance = float(
        64.0 * np.finfo(values.dtype).eps * max(abs(float(values[-1])), 1.0)
    )
    if not np.allclose(differences, step, rtol=0.0, atol=tolerance):
        raise ValueError(
            f"{profile_name} production requires a uniform fixed scale-factor schedule."
        )
    return (
        jax.lax.stop_gradient(jnp.asarray(values)),
        step,
        values.size - 1,
        tolerance,
    )


def _schedule_alignment(
    scale_factors: Array,
    fixed_step_size: float,
    schedule_tolerance: float,
    step_index: Array,
    time: Array,
    state_scale_factor: Array,
    step_size: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    count = scale_factors.size - 1
    index = jnp.asarray(step_index, dtype=jnp.int32).reshape(())
    safe_index = jnp.clip(index, 0, count - 1)
    start = scale_factors[safe_index].astype(state_scale_factor.dtype)
    end = scale_factors[safe_index + 1].astype(state_scale_factor.dtype)
    scheduled_time = jnp.asarray(time, dtype=state_scale_factor.dtype).reshape(())
    proposed_step = jnp.asarray(step_size, dtype=state_scale_factor.dtype).reshape(())
    tolerance = jnp.asarray(schedule_tolerance, dtype=state_scale_factor.dtype)
    aligned = (
        (index >= 0)
        & (index < count)
        & (jnp.abs(scheduled_time - start) <= tolerance)
        & (jnp.abs(state_scale_factor - start) <= tolerance)
        & (
            jnp.abs(
                proposed_step
                - jnp.asarray(fixed_step_size, dtype=state_scale_factor.dtype)
            )
            <= tolerance
        )
    )
    mismatch = jnp.maximum(
        jnp.maximum(jnp.abs(scheduled_time - start), jnp.abs(state_scale_factor - start)),
        jnp.abs(proposed_step - (end - start)),
    )
    return start, end, aligned, mismatch


def _transactional_state(successful: Array, proposed: Any, current: Any, /) -> Any:
    return jax.tree.map(
        lambda new, old: jnp.where(successful, new, old), proposed, current
    )


class AbstractScheduledCosmologyProductionMethod(AbstractFixedStepMethod):
    """Common fixed scale-factor surface for typed prepared-plan adapters.

    New profiles subclass this surface and own a typed prepared plan. Physics is
    never supplied as an unidentifiable callback.
    """

    scale_factors: Array
    fixed_step_size: float = eqx.field(static=True)
    interval_count: int = eqx.field(static=True)
    schedule_tolerance: float = eqx.field(static=True)
    profile_name: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def required_step_size(self) -> None:
        # Decimal absolute knots can differ by roundoff after runtime subtraction;
        # this adapter owns the one tolerance-aware schedule admission below.
        return None

    @property
    def schedule_alignment_tolerance(self) -> float:
        return self.schedule_tolerance

    @property
    def allows_step_reduction(self) -> bool:
        return False

    @property
    def start_scale_factor(self) -> float:
        return float(np.asarray(self.scale_factors[0]))

    @property
    def end_scale_factor(self) -> float:
        return float(np.asarray(self.scale_factors[-1]))

    def production_run_plan(
        self,
        /,
        *,
        checkpoint_interval: int = 1,
        segment_steps: int = 32,
        output_schedule: ExactTimeSchedule | None = None,
        moments: Sequence[StreamingMomentPlan] = (),
        trigger_bindings: Sequence[ProductionTriggerBinding] = (),
        device_resident: bool = False,
    ) -> ProductionRunPlan:
        """Bind this exact profile method to the shared production runtime."""

        return ProductionRunPlan(
            self,
            RobustRetryPolicy(maximum_retries=0),
            step_size=self.fixed_step_size,
            end_time=self.end_scale_factor,
            maximum_steps=self.interval_count,
            checkpoint_interval=checkpoint_interval,
            segment_steps=segment_steps,
            output_schedule=output_schedule,
            moments=moments,
            trigger_bindings=trigger_bindings,
            device_resident=device_resident,
        )

    def initial_run_state(
        self,
        runtime: PreparedProductionRun,
        state: Any,
        /,
        *,
        controller_state: Any = (),
        rng_state: Any = (),
    ) -> ProductionRunState:
        """Create a runtime state at the exact first scale-factor knot."""

        if (
            not isinstance(runtime, PreparedProductionRun)
            or runtime.plan.method.method_id != self.method_id
        ):
            raise ValueError("Prepared production runtime does not bind this method.")
        return runtime.initial_state(
            state,
            time=self.start_scale_factor,
            controller_state=controller_state,
            rng_state=rng_state,
        )

    @abc.abstractmethod
    def step(
        self,
        step_index: Array,
        time: Array,
        state: Any,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        raise NotImplementedError


class PeriodicWaveProductionMethod(AbstractScheduledCosmologyProductionMethod):
    """One accepted periodic-wave schedule interval per production transaction."""

    prepared: PreparedPeriodicWaveDarkMatter

    def __init__(self, prepared: PreparedPeriodicWaveDarkMatter, /):
        if not isinstance(prepared, PreparedPeriodicWaveDarkMatter):
            raise TypeError("prepared must be PreparedPeriodicWaveDarkMatter.")
        schedule, step, count, tolerance = _fixed_schedule(
            prepared.scale_factors, "Periodic-wave"
        )
        self.prepared = prepared
        self.scale_factors = schedule
        self.fixed_step_size = step
        self.interval_count = count
        self.schedule_tolerance = tolerance
        self.profile_name = "periodic-wave"
        self.method_id = canonical_fingerprint(
            {
                "kind": "periodic-wave-production-method",
                "prepared": prepared.prepared_id,
                "schedule": array_tree_fingerprint(np.asarray(schedule)),
                "step_size": step,
                "interval_count": count,
                "schedule_tolerance": tolerance,
                "time_coordinate": "scale-factor",
            }
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: WaveDarkMatterState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        if not isinstance(state, WaveDarkMatterState):
            raise TypeError("Periodic-wave production requires WaveDarkMatterState.")
        if args is not None:
            raise ValueError("Periodic-wave production does not accept runtime args.")
        start, end, aligned, mismatch = _schedule_alignment(
            self.scale_factors,
            self.fixed_step_size,
            self.schedule_tolerance,
            step_index,
            time,
            state.scale_factor,
            step_size,
        )
        safe_state = WaveDarkMatterState(state.psi, start)
        interval_prepared = eqx.tree_at(
            lambda value: value.scale_factors,
            self.prepared,
            jnp.stack((start, end)),
        )
        result = interval_prepared.solve(safe_state)
        successful = result.successful & aligned
        proposed = _transactional_state(aligned, result.state, state)
        accepted = _transactional_state(successful, proposed, state)
        diagnostics = result.diagnostics
        physical_residual = jnp.maximum(
            jnp.maximum(
                diagnostics.norm_relative_error[0],
                diagnostics.poisson_relative_residual[-1],
            ),
            diagnostics.potential_zero_mode_absolute[-1],
        )
        residual = jnp.maximum(physical_residual, mismatch)
        return FixedStepResult(
            proposed,
            accepted,
            successful,
            residual,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=state.scale_factor.dtype),
        )


class RareSIDMProductionState(StrictModule):
    """Checkpoint-complete rare-SIDM state with semantic random-event epoch."""

    particles: CosmologicalParticleState
    prng_root: Array
    event_epoch: Array

    def __init__(
        self,
        particles: CosmologicalParticleState,
        prng_root: ArrayLike,
        event_epoch: ArrayLike = 0,
        /,
    ):
        if not isinstance(particles, CosmologicalParticleState):
            raise TypeError("particles must be CosmologicalParticleState.")
        key_data = jnp.asarray(jr.key_data(prng_root), dtype=jnp.uint32)
        epoch = jnp.asarray(event_epoch, dtype=jnp.int64)
        if key_data.shape != (2,) or epoch.shape != ():
            raise ValueError("SIDM PRNG root and event epoch shapes are invalid.")
        epoch = eqx.error_if(
            epoch,
            epoch < 0,
            "SIDM event epoch must be nonnegative.",
        )
        self.particles = particles
        self.prng_root = key_data
        self.event_epoch = epoch


def _semantic_event_key(root_data: Array, event_epoch: Array, /) -> Array:
    key = jr.wrap_key_data(root_data)
    epoch = jnp.asarray(event_epoch, dtype=jnp.int64)
    low = epoch.astype(jnp.uint32)
    high = jnp.right_shift(epoch, 32).astype(jnp.uint32)
    return jr.fold_in(jr.fold_in(key, low), high)


class RareSIDMProductionMethod(AbstractScheduledCosmologyProductionMethod):
    """One rare equal-mass SIDM/PM interval per atomic production transaction."""

    plan: CosmologicalSIDMPlan
    background: FLRWBackground
    collision_half_steps_per_interval: int = eqx.field(static=True, default=2)

    def __init__(
        self,
        plan: CosmologicalSIDMPlan,
        background: FLRWBackground,
        /,
    ):
        if not isinstance(plan, CosmologicalSIDMPlan):
            raise TypeError("plan must be CosmologicalSIDMPlan.")
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        if background.scale.scale_id != plan.particle_mesh.kinematics.scale.scale_id:
            raise ValueError("SIDM production background and particle scale disagree.")
        schedule, step, count, tolerance = _fixed_schedule(
            plan.particle_mesh.scale_factors, "Rare-SIDM"
        )
        self.plan = plan
        self.background = background
        self.scale_factors = schedule
        self.fixed_step_size = step
        self.interval_count = count
        self.schedule_tolerance = tolerance
        self.profile_name = "rare-sidm-equal"
        self.collision_half_steps_per_interval = 2
        self.method_id = canonical_fingerprint(
            {
                "kind": "rare-equal-sidm-production-method",
                "plan": plan.plan_id,
                "cosmology": background.physical_state.content_id(),
                "schedule": array_tree_fingerprint(np.asarray(schedule)),
                "step_size": step,
                "interval_count": count,
                "schedule_tolerance": tolerance,
                "event_epoch_stride": self.collision_half_steps_per_interval,
                "time_coordinate": "scale-factor",
            }
        )

    def initialize(
        self,
        particles: CosmologicalParticleState,
        prng_root: ArrayLike,
        /,
        *,
        event_epoch: ArrayLike = 0,
    ) -> RareSIDMProductionState:
        return RareSIDMProductionState(particles, prng_root, event_epoch)

    @property
    def particle_ids(self) -> Array:
        return self.plan.particle_mesh.kinematics.particles.particle_ids

    @property
    def active_mask(self) -> Array:
        return self.plan.particle_mesh.kinematics.particles.active_mask

    def step(
        self,
        step_index: Array,
        time: Array,
        state: RareSIDMProductionState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        if not isinstance(state, RareSIDMProductionState):
            raise TypeError("Rare-SIDM production requires RareSIDMProductionState.")
        particles = state.particles
        start, end, aligned, mismatch = _schedule_alignment(
            self.scale_factors,
            self.fixed_step_size,
            self.schedule_tolerance,
            step_index,
            time,
            particles.scale_factor,
            step_size,
        )
        safe_particles = CosmologicalParticleState(
            particles.positions,
            particles.canonical_momenta,
            start,
        )
        interval_plan = eqx.tree_at(
            lambda value: value.particle_mesh.scale_factors,
            self.plan,
            jnp.stack((start, end)),
        )
        root = _semantic_event_key(state.prng_root, state.event_epoch)
        result = interval_plan.rollout(
            self.background,
            safe_particles,
            root,
            args,
        )
        advanced_epoch = state.event_epoch + jnp.asarray(
            self.collision_half_steps_per_interval,
            dtype=state.event_epoch.dtype,
        )
        interval_candidate = RareSIDMProductionState(
            result.state,
            state.prng_root,
            advanced_epoch,
        )
        proposed = _transactional_state(aligned, interval_candidate, state)
        successful = result.successful & aligned
        accepted = _transactional_state(successful, proposed, state)
        first = result.diagnostics.first_half_collisions
        second = result.diagnostics.second_half_collisions
        residual = jnp.maximum(
            mismatch,
            jnp.maximum(
                result.diagnostics.particle_mesh.maximum_mass_balance_defect,
                jnp.maximum(
                    jnp.abs(first.total_kinetic_energy_defect),
                    jnp.abs(second.total_kinetic_energy_defect),
                ),
            ),
        )
        work = (
            first.event_count.astype(jnp.int32)
            + second.event_count.astype(jnp.int32)
            + jnp.asarray(1, dtype=jnp.int32)
        )
        return FixedStepResult(
            proposed,
            accepted,
            successful,
            residual,
            jnp.asarray(1, dtype=jnp.int32),
            work,
            jnp.asarray(False),
            jnp.zeros((), dtype=particles.scale_factor.dtype),
        )


__all__ = [
    "AbstractScheduledCosmologyProductionMethod",
    "PeriodicWaveProductionMethod",
    "RareSIDMProductionMethod",
    "RareSIDMProductionState",
]
