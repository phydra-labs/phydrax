#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

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
from ._acquisition import AcousticGrid, PreparedAcousticSampling


def _derivative(values: Array, axis: int, spacing: float) -> Array:
    return (jnp.roll(values, -1, axis=axis) - jnp.roll(values, 1, axis=axis)) / (
        2.0 * spacing
    )


def _voigt_pairs(dimension: int) -> tuple[tuple[int, int], ...]:
    return (
        ((0, 0), (1, 1), (0, 1))
        if dimension == 2
        else ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    )


class ElasticWaveState(StrictModule):
    velocity_half_step_m_s: Array
    stress_Pa: Array
    step_index: Array
    plan_id: str = eqx.field(static=True)


class ElasticWaveSimulation(StrictModule):
    receiver_velocity_m_s: Array
    receiver_stress_Pa: Array
    final_state: ElasticWaveState
    finite: Array
    plan_id: str = eqx.field(static=True)


class ElasticAcquisition(StrictModule, NonTrainableState):
    sources: PreparedAcousticSampling
    receivers: PreparedAcousticSampling
    acquisition_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: AcousticGrid,
        source_positions_m: ArrayLike,
        receiver_positions_m: ArrayLike,
        /,
    ):
        self.sources = PreparedAcousticSampling(grid, source_positions_m)
        self.receivers = PreparedAcousticSampling(grid, receiver_positions_m)
        self.acquisition_id = canonical_fingerprint(
            {
                "kind": "elastic-acquisition",
                "sources": self.sources.sampling_id,
                "receivers": self.receivers.sampling_id,
            }
        )


class PeriodicIsotropicElasticWavePlan(StrictModule, NonTrainableState):
    """Periodic cell-centered stress and half-step velocity elastic dynamics.

    Centered periodic derivatives are an exact negative-transpose pair. The
    symplectic stress/velocity staggering conserves the corresponding discrete
    energy up to timestep error. Free surfaces and CPML are separate plans.
    """

    grid: AcousticGrid
    time_step_s: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    maximum_p_wavespeed_m_s: float = eqx.field(static=True)
    cfl_number: float = eqx.field(static=True)
    voigt_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: AcousticGrid,
        time_step_s: float,
        step_count: int,
        maximum_p_wavespeed_m_s: float,
        /,
        *,
        cfl_limit: float = 0.9,
    ):
        if not isinstance(grid, AcousticGrid):
            raise TypeError("Elastic waves require AcousticGrid.")
        dt, maximum = float(time_step_s), float(maximum_p_wavespeed_m_s)
        steps = int(step_count)
        courant = float(dt * maximum * np.sqrt(sum(value**-2 for value in grid.spacing)))
        if (
            not np.isfinite(dt)
            or dt <= 0
            or steps <= 0
            or not np.isfinite(maximum)
            or maximum <= 0
            or not np.isfinite(cfl_limit)
            or not 0 < cfl_limit < 1
            or courant > cfl_limit
        ):
            raise ValueError("Elastic timestep/count/wavespeed or CFL limit is invalid.")
        self.grid, self.time_step_s, self.step_count = grid, dt, steps
        self.maximum_p_wavespeed_m_s, self.cfl_number = maximum, courant
        self.voigt_pairs = _voigt_pairs(grid.dimensions)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-isotropic-elastic-wave",
                "grid": grid.grid_id,
                "time_step_s": dt,
                "step_count": steps,
                "maximum_p_wavespeed_m_s": maximum,
            }
        )

    def _materials(
        self,
        density_kg_m3: ArrayLike,
        lame_lambda_Pa: ArrayLike,
        shear_modulus_Pa: ArrayLike,
    ) -> tuple[Array, Array, Array]:
        density, lame, shear = (
            jnp.broadcast_to(jnp.asarray(value), self.grid.shape)
            for value in (density_kg_m3, lame_lambda_Pa, shear_modulus_Pa)
        )
        p_speed = jnp.sqrt((lame + 2.0 * shear) / density)
        density = eqx.error_if(
            density,
            jnp.any(~jnp.isfinite(density))
            | jnp.any(density <= 0)
            | jnp.any(~jnp.isfinite(lame))
            | jnp.any(~jnp.isfinite(shear))
            | jnp.any(shear <= 0)
            | jnp.any(lame + 2.0 * shear / self.grid.dimensions <= 0)
            | jnp.any(~jnp.isfinite(p_speed))
            | jnp.any(p_speed > self.maximum_p_wavespeed_m_s),
            "Elastic density and Lamé fields must be stable, finite, and CFL bounded.",
        )
        return density, lame, shear

    def initial_state(self) -> ElasticWaveState:
        velocity = jnp.zeros((self.grid.dimensions,) + self.grid.shape)
        stress = jnp.zeros((len(self.voigt_pairs),) + self.grid.shape)
        return ElasticWaveState(
            velocity, stress, jnp.asarray(0, dtype=jnp.int32), self.plan_id
        )

    def _stress_tensor(self, stress: Array) -> Array:
        tensor = jnp.zeros(self.grid.shape + (self.grid.dimensions, self.grid.dimensions))
        for component, (row, column) in enumerate(self.voigt_pairs):
            tensor = tensor.at[..., row, column].set(stress[component])
            tensor = tensor.at[..., column, row].set(stress[component])
        return tensor

    def _stress_voigt(self, tensor: Array) -> Array:
        return jnp.stack([tensor[..., row, column] for row, column in self.voigt_pairs])

    def _advance(
        self,
        state: ElasticWaveState,
        density: Array,
        lame: Array,
        shear: Array,
        force_density: Array,
        moment_rate: Array,
    ) -> ElasticWaveState:
        tensor = self._stress_tensor(state.stress_Pa)
        divergence = []
        for component in range(self.grid.dimensions):
            value = jnp.zeros(self.grid.shape)
            for axis, spacing in enumerate(self.grid.spacing):
                value = value + _derivative(tensor[..., component, axis], axis, spacing)
            divergence.append(value)
        velocity = (
            state.velocity_half_step_m_s
            + self.time_step_s * (jnp.stack(divergence) + force_density) / density
        )
        gradient = jnp.stack(
            [
                jnp.stack(
                    [
                        _derivative(velocity[component], axis, spacing)
                        for axis, spacing in enumerate(self.grid.spacing)
                    ],
                    axis=-1,
                )
                for component in range(self.grid.dimensions)
            ],
            axis=-2,
        )
        strain_rate = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
        trace = jnp.trace(strain_rate, axis1=-2, axis2=-1)
        stress_rate = 2.0 * shear[..., None, None] * strain_rate
        stress_rate = stress_rate + lame[..., None, None] * trace[
            ..., None, None
        ] * jnp.eye(self.grid.dimensions)
        stress_rate = stress_rate + self._stress_tensor(moment_rate)
        stress = tensor + self.time_step_s * stress_rate
        return ElasticWaveState(
            velocity,
            self._stress_voigt(stress),
            state.step_index + 1,
            self.plan_id,
        )

    def simulate(
        self,
        density_kg_m3: ArrayLike,
        lame_lambda_Pa: ArrayLike,
        shear_modulus_Pa: ArrayLike,
        acquisition: ElasticAcquisition,
        force_history_N_m3: ArrayLike,
        moment_rate_history_Pa_s: ArrayLike,
        /,
        *,
        replay: CheckpointedScanMode = "full",
        block_size: int | None = None,
        schedule: PreparedReplaySchedule | None = None,
    ) -> ElasticWaveSimulation:
        if (
            not isinstance(acquisition, ElasticAcquisition)
            or acquisition.sources.grid_id != self.grid.grid_id
        ):
            raise ValueError("Elastic acquisition belongs to another grid.")
        density, lame, shear = self._materials(
            density_kg_m3, lame_lambda_Pa, shear_modulus_Pa
        )
        forces = jnp.asarray(force_history_N_m3)
        moments = jnp.asarray(moment_rate_history_Pa_s)
        if forces.shape != (
            self.step_count,
            acquisition.sources.count,
            self.grid.dimensions,
        ):
            raise ValueError("Elastic force history has wrong shape.")
        if moments.shape != (
            self.step_count,
            acquisition.sources.count,
            len(self.voigt_pairs),
        ):
            raise ValueError("Elastic moment-rate history has wrong shape.")
        forces = eqx.error_if(
            forces,
            jnp.any(~jnp.isfinite(forces)) | jnp.any(~jnp.isfinite(moments)),
            "Elastic source histories must be finite.",
        )
        initial = self.initial_state()

        def body(state: ElasticWaveState, sources) -> tuple[ElasticWaveState, Any]:
            force, moment = sources
            force_density = (
                jnp.stack(
                    [
                        acquisition.sources.transpose(force[:, component])
                        for component in range(self.grid.dimensions)
                    ]
                )
                / self.grid.cell_measure
            )
            moment_density = (
                jnp.stack(
                    [
                        acquisition.sources.transpose(moment[:, component])
                        for component in range(len(self.voigt_pairs))
                    ]
                )
                / self.grid.cell_measure
            )
            advanced = self._advance(
                state, density, lame, shear, force_density, moment_density
            )
            receiver_velocity = jnp.stack(
                [
                    acquisition.receivers.apply(value)
                    for value in advanced.velocity_half_step_m_s
                ],
                axis=-1,
            )
            receiver_stress = jnp.stack(
                [acquisition.receivers.apply(value) for value in advanced.stress_Pa],
                axis=-1,
            )
            return advanced, (receiver_velocity, receiver_stress)

        final, (velocity, stress) = checkpointed_scan(
            body,
            initial,
            (forces, moments),
            length=self.step_count,
            mode=replay,
            block_size=block_size,
            schedule=schedule,
        )
        return ElasticWaveSimulation(
            velocity,
            stress,
            final,
            jnp.all(jnp.isfinite(velocity)) & jnp.all(jnp.isfinite(stress)),
            self.plan_id,
        )


__all__ = [
    "ElasticAcquisition",
    "ElasticWaveSimulation",
    "ElasticWaveState",
    "PeriodicIsotropicElasticWavePlan",
]
