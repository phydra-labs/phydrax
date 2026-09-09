#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._numerics._checkpointed_scan import (
    checkpointed_scan,
    CheckpointedScanMode,
    PreparedReplaySchedule,
)
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....series import SampledSeries, SeriesSupport
from ._acquisition import AcousticGrid, SeismicAcquisition
from ._constant_density import (
    _gradient,
    AcousticSimulation,
    AcousticState,
    ConstantDensityAcousticPlan,
)


class VariableDensityAcousticPlan(StrictModule, NonTrainableState):
    """Conservative staggered acoustic pressure/velocity with cell density."""

    baseline: ConstantDensityAcousticPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: AcousticGrid,
        time_step: float,
        step_count: int,
        maximum_wavespeed: float,
        /,
        *,
        time_origin: float = 0.0,
        absorber_cells: int = 0,
        absorber_strength: float = 3.0,
        cfl_limit: float = 0.95,
    ):
        self.baseline = ConstantDensityAcousticPlan(
            grid,
            time_step,
            step_count,
            maximum_wavespeed,
            1.0,
            time_origin=time_origin,
            absorber_cells=absorber_cells,
            absorber_strength=absorber_strength,
            cfl_limit=cfl_limit,
        )
        self.plan_id = canonical_fingerprint(
            {"kind": "variable-density-acoustic-plan", "baseline": self.baseline.plan_id}
        )

    @property
    def grid(self) -> AcousticGrid:
        return self.baseline.grid

    @property
    def time_step(self) -> float:
        return self.baseline.time_step

    @property
    def step_count(self) -> int:
        return self.baseline.step_count

    @property
    def time_origin(self) -> float:
        return self.baseline.time_origin

    def _materials(
        self, wavespeed: ArrayLike, density_kg_m3: ArrayLike
    ) -> tuple[Array, Array]:
        speed = jnp.broadcast_to(jnp.asarray(wavespeed), self.grid.shape)
        density = jnp.broadcast_to(jnp.asarray(density_kg_m3), self.grid.shape)
        speed = eqx.error_if(
            speed,
            jnp.any(~jnp.isfinite(speed))
            | jnp.any(speed <= 0)
            | jnp.any(speed > self.baseline.maximum_wavespeed)
            | jnp.any(~jnp.isfinite(density))
            | jnp.any(density <= 0),
            "Variable-density acoustic material must be finite, positive, and CFL bounded.",
        )
        return speed, density

    def initial_state(self, pressure: ArrayLike = 0.0, /) -> AcousticState:
        state = self.baseline.initial_state(pressure)
        return AcousticState(
            state.split_pressure,
            state.velocity,
            state.step_index,
            self.plan_id,
        )

    def _state(self, state: AcousticState) -> AcousticState:
        if state.plan_id != self.plan_id:
            raise ValueError("Variable-density acoustic state belongs to another plan.")
        if state.split_pressure.shape != (self.grid.dimensions,) + self.grid.shape:
            raise ValueError("Variable-density acoustic pressure state shape is invalid.")
        finite = jnp.all(jnp.isfinite(state.split_pressure))
        for axis, velocity in enumerate(state.velocity):
            shape = list(self.grid.shape)
            shape[axis] += 1
            if velocity.shape != tuple(shape):
                raise ValueError("Variable-density acoustic velocity shape is invalid.")
            finite = finite & jnp.all(jnp.isfinite(velocity))
        split = eqx.error_if(
            state.split_pressure,
            ~finite | (state.step_index < 0),
            "Variable-density acoustic state must be finite with nonnegative clock.",
        )
        return AcousticState(split, state.velocity, state.step_index, state.plan_id)

    def _face_density(self, density: Array, axis: int) -> Array:
        left = jnp.take(density, jnp.arange(density.shape[axis] - 1), axis=axis)
        right = jnp.take(density, jnp.arange(1, density.shape[axis]), axis=axis)
        interior = 0.5 * (left + right)
        first = jnp.take(density, jnp.asarray([0]), axis=axis)
        last = jnp.take(density, jnp.asarray([density.shape[axis] - 1]), axis=axis)
        return jnp.concatenate((first, interior, last), axis=axis)

    def _advance(
        self,
        state: AcousticState,
        speed: Array,
        density: Array,
        source_density: Array,
    ) -> AcousticState:
        dt = self.time_step
        bulk = density * speed**2
        parts = tuple(
            state.split_pressure[axis] * self.baseline.pressure_damping[axis]
            for axis in range(self.grid.dimensions)
        )
        pressure = jnp.sum(jnp.stack(parts), axis=0)
        half_velocity = tuple(
            velocity * damping
            - 0.5
            * dt
            * _gradient(pressure, axis, spacing)
            / self._face_density(density, axis)
            for axis, (velocity, damping, spacing) in enumerate(
                zip(
                    state.velocity,
                    self.baseline.velocity_damping,
                    self.grid.spacing,
                    strict=True,
                )
            )
        )
        parts = tuple(
            part
            - dt * bulk * jnp.diff(velocity, axis=axis) / spacing
            + dt * bulk * source_density / self.grid.dimensions
            for axis, (part, velocity, spacing) in enumerate(
                zip(parts, half_velocity, self.grid.spacing, strict=True)
            )
        )
        pressure = jnp.sum(jnp.stack(parts), axis=0)
        velocity = tuple(
            (
                value
                - 0.5
                * dt
                * _gradient(pressure, axis, spacing)
                / self._face_density(density, axis)
            )
            * damping
            for axis, (value, damping, spacing) in enumerate(
                zip(
                    half_velocity,
                    self.baseline.velocity_damping,
                    self.grid.spacing,
                    strict=True,
                )
            )
        )
        split = jnp.stack(
            tuple(
                part * damping
                for part, damping in zip(
                    parts, self.baseline.pressure_damping, strict=True
                )
            )
        )
        return AcousticState(split, velocity, state.step_index + 1, self.plan_id)

    def simulate(
        self,
        wavespeed: ArrayLike,
        density_kg_m3: ArrayLike,
        acquisition: SeismicAcquisition,
        source_rates: ArrayLike,
        /,
        *,
        initial_state: AcousticState | None = None,
        replay: CheckpointedScanMode = "full",
        block_size: int | None = None,
        schedule: PreparedReplaySchedule | None = None,
        save_wavefield: bool = False,
    ) -> AcousticSimulation:
        if (
            acquisition.sources.grid_id != self.grid.grid_id
            or acquisition.receivers.grid_id != self.grid.grid_id
        ):
            raise ValueError("Variable-density acquisition belongs to another grid.")
        speed, density = self._materials(wavespeed, density_kg_m3)
        rates = jnp.asarray(source_rates, dtype=speed.dtype)
        if rates.shape != (self.step_count, acquisition.sources.count):
            raise ValueError("Variable-density source history has wrong shape.")
        rates = eqx.error_if(
            rates,
            jnp.any(~jnp.isfinite(rates)),
            "Variable-density source rates must be finite.",
        )
        initial = (
            self.initial_state() if initial_state is None else self._state(initial_state)
        )

        def body(state: AcousticState, rate: Array) -> tuple[AcousticState, Any]:
            source = acquisition.sources.transpose(rate) / self.grid.cell_measure
            advanced = self._advance(state, speed, density, source)
            pressure = advanced.pressure
            return advanced, (
                acquisition.receivers.apply(pressure),
                pressure if save_wavefield else None,
            )

        final, (samples, fields) = checkpointed_scan(
            body,
            initial,
            rates,
            length=self.step_count,
            mode=replay,
            block_size=block_size,
            schedule=schedule,
        )
        samples = jnp.concatenate(
            (acquisition.receivers.apply(initial.pressure)[None, :], samples), axis=0
        ).T
        times = self.time_origin + self.time_step * (
            initial.step_index + jnp.arange(self.step_count + 1)
        )
        support = SeriesSupport(
            times,
            series_shape=(acquisition.receivers.count,),
            series_axes=("receiver",),
            coordinate_name="time_s",
            coordinate_id=self.plan_id,
        )
        traces = SampledSeries(support, samples, series_id=acquisition.acquisition_id)
        wavefield = (
            None
            if fields is None
            else jnp.concatenate((initial.pressure[None, ...], fields), axis=0)
        )
        return AcousticSimulation(
            traces,
            self._state(final),
            wavefield,
            jnp.all(jnp.isfinite(samples)),
            self.plan_id,
            acquisition.acquisition_id,
        )


__all__ = ["VariableDensityAcousticPlan"]
