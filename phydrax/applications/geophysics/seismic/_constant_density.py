#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Constant-density scalar acoustics on static 2D/3D staggered Cartesian grids.

The interior equations are p_t = -rho*c**2 div(v) + rho*c**2 Q/V and
v_t = -grad(p)/rho. Sources are volume rates (m³/s in 3D, m²/s per unit
out-of-plane length in 2D). Pressure is in Pa, velocity in m/s, time in s.
Velocity-Verlet and centred staggered differences are second order. The outer
normal velocity is zero: without a layer the box has reflecting rigid walls.

The optional absorber evolves actual axis-split pressure memories with matching
axis velocity damping. Polynomial damping is Strang-split from propagation.
This is a split-field damping layer, NOT a claim of perfectly matched discrete
PML or CPML. Finite thickness, oblique incidence, corners, spatial dispersion,
and material variation in the layer cause reflection; qualify those numerically
for each survey. There is no free-surface, elasticity, attenuation, or topography.
"""

from __future__ import annotations

from math import isfinite, prod, sqrt
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._numerics._checkpointed_scan import (
    checkpointed_scan,
    CheckpointedScanMode,
    PreparedReplaySchedule,
)
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....artifacts import DifferentiationContract
from ....ein import contract
from ....series import SampledSeries, SeriesSupport
from .._evidence import GeophysicalCapabilityEvidence, GeophysicalResourceEstimate
from ._acquisition import AcousticGrid, SeismicAcquisition


class AcousticState(StrictModule):
    """Complete restart state, including every split-layer auxiliary field."""

    split_pressure: Array
    velocity: tuple[Array, ...]
    step_index: Array
    plan_id: str = eqx.field(static=True)

    @property
    def pressure(self) -> Array:
        return jnp.sum(self.split_pressure, axis=0)


class AcousticCheckpoint(StrictModule):
    """Exact state and material for deterministic continuation under one plan."""

    state: AcousticState
    wavespeed: Array
    plan_id: str = eqx.field(static=True)


class AcousticSimulation(StrictModule):
    """Pressure traces include the initial node and every completed time step."""

    traces: SampledSeries
    final_state: AcousticState
    wavefield: Array | None
    successful: Array
    plan_id: str = eqx.field(static=True)
    acquisition_id: str = eqx.field(static=True)
    sample_unit: str = eqx.field(static=True, default="Pa")
    time_unit: str = eqx.field(static=True, default="s")


def _gradient(pressure: Array, axis: int, spacing: float) -> Array:
    """Pressure difference on interior faces, zero normal wall derivative."""
    interior = jnp.diff(pressure, axis=axis) / spacing
    pads = [(0, 0)] * pressure.ndim
    pads[axis] = (1, 1)
    return jnp.pad(interior, pads)


class ConstantDensityAcousticPlan(StrictModule, NonTrainableState):
    """Fixed-step native propagation and full/step/block/scheduled AD replay.

    ``maximum_wavespeed`` is a hard material bound, not an estimated CFL hint.
    Runtime wavespeeds must be finite, strictly positive, and below that bound.
    Gradients describe the executed fixed discrete model, with the prepared grid,
    acquisition, damping, source clock, and checkpoint schedule held fixed.
    """

    grid: AcousticGrid
    pressure_damping: tuple[Array, ...]
    velocity_damping: tuple[Array, ...]
    time_step: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    maximum_wavespeed: float = eqx.field(static=True)
    density: float = eqx.field(static=True)
    time_origin: float = eqx.field(static=True)
    absorber_cells: int = eqx.field(static=True)
    absorber_strength: float = eqx.field(static=True)
    cfl_number: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: AcousticGrid,
        time_step: float,
        step_count: int,
        maximum_wavespeed: float,
        density: float = 1000.0,
        /,
        *,
        time_origin: float = 0.0,
        absorber_cells: int = 0,
        absorber_strength: float = 3.0,
        cfl_limit: float = 0.95,
    ):
        dt, cmax, rho = float(time_step), float(maximum_wavespeed), float(density)
        if not all(isfinite(value) and value > 0 for value in (dt, cmax, rho)):
            raise ValueError(
                "Acoustic time step, maximum wavespeed, and density must be finite and positive."
            )
        if int(step_count) != step_count or step_count <= 0:
            raise ValueError("Acoustic step_count must be a positive integer.")
        width = int(absorber_cells)
        if width != absorber_cells or width < 0 or 2 * width >= min(grid.shape):
            raise ValueError(
                "Absorber width must be nonnegative and leave an interior cell region."
            )
        strength = float(absorber_strength)
        if not isfinite(strength) or strength <= 0:
            raise ValueError("Absorber strength must be finite and positive.")
        if not isfinite(cfl_limit) or not 0 < cfl_limit < 1:
            raise ValueError("Acoustic cfl_limit must lie strictly between zero and one.")
        courant = dt * cmax * sqrt(sum(spacing**-2 for spacing in grid.spacing))
        if courant > cfl_limit:
            raise ValueError(f"Acoustic CFL {courant:g} exceeds limit {cfl_limit:g}.")
        if not isfinite(time_origin):
            raise ValueError("Acoustic time_origin must be finite.")
        p_damping, v_damping = [], []
        for axis, (size, spacing) in enumerate(
            zip(grid.shape, grid.spacing, strict=True)
        ):
            for face, output in ((False, p_damping), (True, v_damping)):
                coordinates = np.arange(size + int(face), dtype=float) + (
                    0.0 if face else 0.5
                )
                if width:
                    depth = (
                        np.maximum(
                            np.maximum(width - coordinates, coordinates - (size - width)),
                            0.0,
                        )
                        / width
                    )
                    sigma = strength * cmax / (width * spacing) * depth**2
                else:
                    sigma = np.zeros_like(coordinates)
                shape = [1] * grid.dimensions
                shape[axis] = coordinates.size
                output.append(jnp.asarray(np.exp(-0.5 * dt * sigma).reshape(shape)))
        self.grid, self.pressure_damping, self.velocity_damping = (
            grid,
            tuple(p_damping),
            tuple(v_damping),
        )
        self.time_step, self.step_count, self.maximum_wavespeed = (
            dt,
            int(step_count),
            cmax,
        )
        self.density, self.time_origin = rho, float(time_origin)
        self.absorber_cells, self.absorber_strength, self.cfl_number = (
            width,
            strength,
            courant,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scalar-acoustic-verlet-split-damping",
                "grid": grid.grid_id,
                "time_step_s": dt,
                "steps": self.step_count,
                "cmax_m_s": cmax,
                "density_kg_m3": rho,
                "time_origin_s": self.time_origin,
                "absorber_cells": width,
                "absorber_strength": strength,
            }
        )

    @property
    def capability_evidence(self) -> GeophysicalCapabilityEvidence:
        dimension = self.grid.dimensions
        return GeophysicalCapabilityEvidence(
            "constant-density-acoustic",
            (dimension,),
            field_equations=("first-order-scalar-acoustic",),
            source_models=(
                "monopole-volume-rate"
                if dimension == 3
                else "monopole-area-rate-per-unit-thickness",
            ),
            receiver_models=("multilinear-pressure",),
            boundary_models=(
                "rigid-wall",
                "finite-polynomial-split-damping",
            ),
            material_models=("positive-scalar-wavespeed", "constant-density"),
            limitations=(
                "fixed-cartesian-grid",
                "no-free-surface",
                "no-elasticity",
                "no-attenuation",
                "damping-is-not-cpml",
            ),
            differentiation=DifferentiationContract(
                upstream_physical_parameters=True,
                stored_values=True,
                query_coordinates=False,
                local_parameters=True,
                stochastic_realization=False,
                higher_order=True,
            ),
        )

    def resource_estimate(
        self,
        *,
        receiver_count: int = 0,
        checkpoint_count: int = 0,
        maximum_bytes: int | None = None,
    ) -> GeophysicalResourceEstimate:
        receivers = int(receiver_count)
        checkpoints = int(checkpoint_count)
        if receivers < 0 or checkpoints < 0:
            raise ValueError("Receiver and checkpoint counts must be nonnegative.")
        cells = int(prod(self.grid.shape))
        scalar_bytes = int(self.pressure_damping[0].dtype.itemsize)
        state_scalars = cells * (1 + 2 * self.grid.dimensions)
        retained = state_scalars * scalar_bytes
        retained += sum(
            value.size * value.dtype.itemsize for value in self.pressure_damping
        )
        retained += sum(
            value.size * value.dtype.itemsize for value in self.velocity_damping
        )
        workspace = 3 * state_scalars * scalar_bytes
        checkpoint = checkpoints * state_scalars * scalar_bytes
        observations = receivers * (self.step_count + 1) * scalar_bytes
        return GeophysicalResourceEstimate(
            retained_bytes=retained,
            workspace_bytes=workspace,
            checkpoint_bytes=checkpoint,
            observation_bytes=observations,
            source_batch_size=1,
            maximum_bytes=maximum_bytes,
        )

    def _wavespeed(self, wavespeed: ArrayLike) -> Array:
        value = jnp.asarray(wavespeed, dtype=float)
        if value.shape not in ((), self.grid.shape):
            raise ValueError("Wavespeed must be scalar or match the acoustic grid.")
        value = eqx.error_if(
            value,
            jnp.any(
                ~jnp.isfinite(value) | (value <= 0) | (value > self.maximum_wavespeed)
            ),
            "Acoustic wavespeed must be finite, positive, and within the prepared CFL bound.",
        )
        return jnp.broadcast_to(value, self.grid.shape)

    def initial_state(self, pressure: ArrayLike = 0.0, /) -> AcousticState:
        values = jnp.asarray(pressure, dtype=float)
        if values.shape not in ((), self.grid.shape):
            raise ValueError(
                "Initial pressure must be scalar or match the acoustic grid."
            )
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Initial acoustic pressure must be finite.",
        )
        split = jnp.broadcast_to(
            values / self.grid.dimensions, (self.grid.dimensions,) + self.grid.shape
        )
        velocity = []
        for axis in range(self.grid.dimensions):
            shape = list(self.grid.shape)
            shape[axis] += 1
            velocity.append(jnp.zeros(tuple(shape), dtype=values.dtype))
        return AcousticState(
            split, tuple(velocity), jnp.asarray(0, dtype=jnp.int32), self.plan_id
        )

    def _state(self, state: AcousticState) -> AcousticState:
        if (
            state.plan_id != self.plan_id
            or state.split_pressure.shape != (self.grid.dimensions,) + self.grid.shape
        ):
            raise ValueError("Acoustic state does not belong to this prepared plan.")
        if len(state.velocity) != self.grid.dimensions or state.step_index.shape != ():
            raise ValueError("Acoustic state velocity or clock shape is invalid.")
        finite = jnp.all(jnp.isfinite(state.split_pressure))
        if not jnp.issubdtype(state.step_index.dtype, jnp.integer):
            raise ValueError("Acoustic state clock must be an integer step index.")
        for axis, velocity in enumerate(state.velocity):
            shape = list(self.grid.shape)
            shape[axis] += 1
            if velocity.shape != tuple(shape):
                raise ValueError("Acoustic staggered velocity shape is invalid.")
            finite = finite & jnp.all(jnp.isfinite(velocity))
            boundary = jnp.take(velocity, jnp.asarray([0, shape[axis] - 1]), axis=axis)
            finite = finite & jnp.all(boundary == 0)
        split = eqx.error_if(
            state.split_pressure,
            ~finite | (state.step_index < 0),
            "Acoustic state must be finite with a nonnegative clock and zero normal wall velocities.",
        )
        return AcousticState(split, state.velocity, state.step_index, state.plan_id)

    def _advance(
        self, state: AcousticState, speed: Array, source_density: Array
    ) -> AcousticState:
        dt, rho = self.time_step, self.density
        bulk = rho * speed**2
        parts = tuple(
            state.split_pressure[axis] * self.pressure_damping[axis]
            for axis in range(self.grid.dimensions)
        )
        pressure = jnp.sum(jnp.stack(parts), axis=0)
        half_velocity = tuple(
            velocity * damping - (0.5 * dt / rho) * _gradient(pressure, axis, spacing)
            for axis, (velocity, damping, spacing) in enumerate(
                zip(state.velocity, self.velocity_damping, self.grid.spacing, strict=True)
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
            (value - (0.5 * dt / rho) * _gradient(pressure, axis, spacing)) * damping
            for axis, (value, damping, spacing) in enumerate(
                zip(half_velocity, self.velocity_damping, self.grid.spacing, strict=True)
            )
        )
        split = jnp.stack(
            tuple(
                part * damping
                for part, damping in zip(parts, self.pressure_damping, strict=True)
            )
        )
        return AcousticState(split, velocity, state.step_index + 1, self.plan_id)

    def step(
        self,
        state: AcousticState,
        wavespeed: ArrayLike,
        acquisition: SeismicAcquisition,
        source_rates: ArrayLike,
        /,
    ) -> AcousticState:
        """Advance once; rates are the interval-mean physical monopole volume rates."""
        self._acquisition(acquisition)
        rates = jnp.asarray(source_rates, dtype=float)
        if rates.shape != (acquisition.sources.count,):
            raise ValueError(
                "A source step requires one physical volume rate per source."
            )
        rates = eqx.error_if(
            rates, jnp.any(~jnp.isfinite(rates)), "Acoustic source rates must be finite."
        )
        return self._state(
            self._advance(
                self._state(state),
                self._wavespeed(wavespeed),
                acquisition.sources.transpose(rates) / self.grid.cell_measure,
            )
        )

    def _acquisition(self, acquisition: SeismicAcquisition) -> None:
        if (
            acquisition.sources.grid_id != self.grid.grid_id
            or acquisition.receivers.grid_id != self.grid.grid_id
        ):
            raise ValueError("Acoustic acquisition does not belong to this grid.")

    def simulate(
        self,
        wavespeed: ArrayLike,
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
        """Run exactly ``step_count`` steps using native checkpointed_scan.

        ``replay='full'`` retains AD intermediates; other modes rematerialize the
        same operator with the complete split pressure and velocity carry. The
        primal and derivatives must agree up to floating-point reduction order.
        No wavefield histories are retained as outputs unless explicitly asked.
        """
        self._acquisition(acquisition)
        speed = self._wavespeed(wavespeed)
        rates = jnp.asarray(source_rates, dtype=speed.dtype)
        if rates.shape != (self.step_count, acquisition.sources.count):
            raise ValueError(
                "Acoustic source history must have shape (step_count, source_count)."
            )
        rates = eqx.error_if(
            rates, jnp.any(~jnp.isfinite(rates)), "Acoustic source rates must be finite."
        )
        initial = (
            self.initial_state() if initial_state is None else self._state(initial_state)
        )

        def body(state: AcousticState, rate: Array) -> tuple[AcousticState, Any]:
            density = acquisition.sources.transpose(rate) / self.grid.cell_measure
            advanced = self._advance(state, speed, density)
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
        final = self._state(final)
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
            final,
            wavefield,
            jnp.all(jnp.isfinite(samples)),
            self.plan_id,
            acquisition.acquisition_id,
        )

    def checkpoint(
        self, state: AcousticState, wavespeed: ArrayLike, /
    ) -> AcousticCheckpoint:
        return AcousticCheckpoint(
            self._state(state), self._wavespeed(wavespeed), self.plan_id
        )

    def restart(
        self,
        checkpoint: AcousticCheckpoint,
        acquisition: SeismicAcquisition,
        source_rates: ArrayLike,
        /,
        **kwargs: Any,
    ) -> AcousticSimulation:
        """Continue a checkpoint without dropping or reinitializing absorber memory."""
        if checkpoint.plan_id != self.plan_id:
            raise ValueError("Acoustic checkpoint belongs to a different prepared plan.")
        return self.simulate(
            checkpoint.wavespeed,
            acquisition,
            source_rates,
            initial_state=checkpoint.state,
            **kwargs,
        )

    def energy(self, state: AcousticState, wavespeed: ArrayLike, /) -> Array:
        """Physical acoustic energy (J in 3D, J/m in 2D), not split-memory norm."""
        state_ = self._state(state)
        bulk = self.density * self._wavespeed(wavespeed) ** 2
        pressure = state_.pressure.reshape((-1,))
        potential = contract("i,i->", pressure, (state_.pressure / bulk).reshape((-1,)))
        kinetic = sum(
            contract("i,i->", value.reshape((-1,)), value.reshape((-1,)))
            for value in state_.velocity
        )
        return 0.5 * self.grid.cell_measure * (potential + self.density * kinetic)


def ricker_wavelet(times: ArrayLike, frequency: float, /, *, delay: float = 0.0) -> Array:
    """Dimensionless Ricker pulse; multiply by a physical source-rate amplitude."""
    if not isfinite(frequency) or frequency <= 0 or not isfinite(delay):
        raise ValueError(
            "Ricker frequency must be finite and positive; delay must be finite."
        )
    phase = (jnp.pi * frequency * (jnp.asarray(times, dtype=float) - delay)) ** 2
    return (1.0 - 2.0 * phase) * jnp.exp(-phase)


__all__ = [
    "AcousticCheckpoint",
    "ConstantDensityAcousticPlan",
    "AcousticSimulation",
    "AcousticState",
    "ricker_wavelet",
]
