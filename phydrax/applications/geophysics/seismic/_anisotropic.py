#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._acquisition import AcousticGrid
from ._elastic import _derivative, _voigt_pairs, ElasticAcquisition


class ElasticStiffness(StrictModule):
    matrix_Pa: Array
    dimension: int = eqx.field(static=True)
    voigt_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)

    def __init__(self, matrix_Pa: ArrayLike, dimension: int, /):
        dimension_ = int(dimension)
        pairs = _voigt_pairs(dimension_)
        matrix = jnp.asarray(matrix_Pa)
        if matrix.shape[-2:] != (len(pairs), len(pairs)):
            raise ValueError("Elastic stiffness trailing Voigt dimensions are invalid.")
        symmetric = jnp.max(jnp.abs(matrix - jnp.swapaxes(matrix, -1, -2)))
        eigenvalues = jnp.linalg.eigvalsh(0.5 * (matrix + jnp.swapaxes(matrix, -1, -2)))
        matrix = eqx.error_if(
            matrix,
            jnp.any(~jnp.isfinite(matrix))
            | (symmetric > 1e-10 * jnp.maximum(jnp.max(jnp.abs(matrix)), 1.0))
            | jnp.any(eigenvalues <= 0),
            "Elastic stiffness must be finite symmetric positive definite.",
        )
        self.matrix_Pa = 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))
        self.dimension, self.voigt_pairs = dimension_, pairs

    @classmethod
    def isotropic(
        cls, lame_lambda_Pa: ArrayLike, shear_modulus_Pa: ArrayLike, dimension: int, /
    ) -> ElasticStiffness:
        lame, shear = jnp.broadcast_arrays(
            jnp.asarray(lame_lambda_Pa), jnp.asarray(shear_modulus_Pa)
        )
        pairs = _voigt_pairs(int(dimension))
        size = len(pairs)
        matrix = jnp.zeros(lame.shape + (size, size))
        for row, (i, j) in enumerate(pairs):
            for column, (k, l) in enumerate(pairs):
                value = lame * (i == j) * (k == l) + shear * (
                    (i == k) * (j == l) + (i == l) * (j == k)
                )
                matrix = matrix.at[..., row, column].set(value)
        return cls(matrix, dimension)

    @classmethod
    def vti(
        cls,
        c11_Pa: ArrayLike,
        c33_Pa: ArrayLike,
        c44_Pa: ArrayLike,
        c66_Pa: ArrayLike,
        c13_Pa: ArrayLike,
        /,
    ) -> ElasticStiffness:
        c11, c33, c44, c66, c13 = jnp.broadcast_arrays(
            *(jnp.asarray(value) for value in (c11_Pa, c33_Pa, c44_Pa, c66_Pa, c13_Pa))
        )
        c12 = c11 - 2.0 * c66
        matrix = jnp.zeros(c11.shape + (6, 6))
        matrix = matrix.at[..., 0, 0].set(c11)
        matrix = matrix.at[..., 1, 1].set(c11)
        matrix = matrix.at[..., 2, 2].set(c33)
        matrix = matrix.at[..., 0, 1].set(c12)
        matrix = matrix.at[..., 1, 0].set(c12)
        for row, column in ((0, 2), (2, 0), (1, 2), (2, 1)):
            matrix = matrix.at[..., row, column].set(c13)
        matrix = matrix.at[..., 3, 3].set(2.0 * c44)
        matrix = matrix.at[..., 4, 4].set(2.0 * c44)
        matrix = matrix.at[..., 5, 5].set(2.0 * c66)
        return cls(matrix, 3)

    def stress(self, strain_voigt: ArrayLike, /) -> Array:
        strain = jnp.asarray(strain_voigt)
        if strain.shape[-1] != self.matrix_Pa.shape[-1]:
            raise ValueError("Elastic strain Voigt dimension does not match stiffness.")
        return ein.contract("...ij,...j->...i", self.matrix_Pa, strain)


class StandardLinearSolidSpectrum(StrictModule):
    relaxation_times_s: Array
    modulus_fractions: Array

    def __init__(self, relaxation_times_s: ArrayLike, modulus_fractions: ArrayLike, /):
        times, fractions = jnp.broadcast_arrays(
            jnp.asarray(relaxation_times_s), jnp.asarray(modulus_fractions)
        )
        if times.ndim != 1 or times.size == 0:
            raise ValueError("Viscoelastic spectrum must be a nonempty vector.")
        times = eqx.error_if(
            times,
            jnp.any(~jnp.isfinite(times))
            | jnp.any(times <= 0)
            | jnp.any(~jnp.isfinite(fractions))
            | jnp.any(fractions < 0)
            | (jnp.sum(fractions) >= 1),
            "Viscoelastic relaxation must be causal with nonnegative fractions summing below one.",
        )
        self.relaxation_times_s, self.modulus_fractions = times, fractions

    def update(self, memory: Array, strain: Array, dt_s: ArrayLike, /) -> Array:
        dt = jnp.asarray(dt_s)
        decay = jnp.exp(-dt / self.relaxation_times_s)
        shape = (self.relaxation_times_s.size,) + (1,) * strain.ndim
        return (
            decay.reshape(shape) * memory
            + (1.0 - decay.reshape(shape)) * strain[None, ...]
        )


class AnisotropicViscoelasticState(StrictModule):
    velocity_m_s: Array
    strain_voigt: Array
    memory_strain: Array
    step_index: Array
    plan_id: str = eqx.field(static=True)


class PeriodicAnisotropicViscoelasticPlan(StrictModule, NonTrainableState):
    grid: AcousticGrid
    time_step_s: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    spectrum: StandardLinearSolidSpectrum
    voigt_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: AcousticGrid,
        time_step_s: float,
        step_count: int,
        spectrum: StandardLinearSolidSpectrum,
        /,
    ):
        dt, steps = float(time_step_s), int(step_count)
        if not isinstance(grid, AcousticGrid) or not isinstance(
            spectrum, StandardLinearSolidSpectrum
        ):
            raise TypeError(
                "Anisotropic viscoelastic plan requires grid and SLS spectrum."
            )
        if not np.isfinite(dt) or dt <= 0 or steps <= 0:
            raise ValueError("Anisotropic viscoelastic timestep/count are invalid.")
        self.grid, self.time_step_s, self.step_count = grid, dt, steps
        self.spectrum, self.voigt_pairs = spectrum, _voigt_pairs(grid.dimensions)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-anisotropic-viscoelastic-wave",
                "grid": grid.grid_id,
                "time_step_s": dt,
                "step_count": steps,
                "relaxation_times_s": spectrum.relaxation_times_s,
                "modulus_fractions": spectrum.modulus_fractions,
            }
        )

    def initial_state(self) -> AnisotropicViscoelasticState:
        shape = (len(self.voigt_pairs),) + self.grid.shape
        return AnisotropicViscoelasticState(
            jnp.zeros((self.grid.dimensions,) + self.grid.shape),
            jnp.zeros(shape),
            jnp.zeros((self.spectrum.relaxation_times_s.size,) + shape),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def _stress(
        self, state: AnisotropicViscoelasticState, stiffness: ElasticStiffness
    ) -> Array:
        effective = state.strain_voigt - jnp.sum(
            self.spectrum.modulus_fractions.reshape(
                (-1,) + (1,) * state.strain_voigt.ndim
            )
            * state.memory_strain,
            axis=0,
        )
        field = jnp.moveaxis(effective, 0, -1)
        return jnp.moveaxis(stiffness.stress(field), -1, 0)

    def step(
        self,
        state: AnisotropicViscoelasticState,
        density_kg_m3: ArrayLike,
        stiffness: ElasticStiffness,
        /,
        *,
        force_density_N_m3: ArrayLike = 0.0,
    ) -> AnisotropicViscoelasticState:
        if state.plan_id != self.plan_id:
            raise ValueError("Viscoelastic state belongs to a different wave plan.")
        if (
            not isinstance(stiffness, ElasticStiffness)
            or stiffness.dimension != self.grid.dimensions
        ):
            raise TypeError(
                "Elastic stiffness dimension does not match viscoelastic grid."
            )
        density = jnp.broadcast_to(jnp.asarray(density_kg_m3), self.grid.shape)
        force = jnp.broadcast_to(
            jnp.asarray(force_density_N_m3), (self.grid.dimensions,) + self.grid.shape
        )
        density = eqx.error_if(
            density,
            jnp.any(~jnp.isfinite(density))
            | jnp.any(density <= 0)
            | jnp.any(~jnp.isfinite(force)),
            "Viscoelastic density/force must be finite and density positive.",
        )
        stress_voigt = self._stress(state, stiffness)
        stress_tensor = jnp.zeros(
            self.grid.shape + (self.grid.dimensions, self.grid.dimensions)
        )
        for component, (row, column) in enumerate(self.voigt_pairs):
            stress_tensor = stress_tensor.at[..., row, column].set(
                stress_voigt[component]
            )
            stress_tensor = stress_tensor.at[..., column, row].set(
                stress_voigt[component]
            )
        divergence = []
        for component in range(self.grid.dimensions):
            value = jnp.zeros(self.grid.shape)
            for axis, spacing in enumerate(self.grid.spacing):
                value = value + _derivative(
                    stress_tensor[..., component, axis], axis, spacing
                )
            divergence.append(value)
        velocity = (
            state.velocity_m_s
            + self.time_step_s * (jnp.stack(divergence) + force) / density
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
        strain_rate_voigt = jnp.stack(
            [strain_rate[..., row, column] for row, column in self.voigt_pairs]
        )
        strain = state.strain_voigt + self.time_step_s * strain_rate_voigt
        memory = self.spectrum.update(state.memory_strain, strain, self.time_step_s)
        return AnisotropicViscoelasticState(
            velocity, strain, memory, state.step_index + 1, self.plan_id
        )

    def simulate(
        self,
        density_kg_m3: ArrayLike,
        stiffness: ElasticStiffness,
        acquisition: ElasticAcquisition,
        force_history_N_m3: ArrayLike,
        /,
    ) -> tuple[AnisotropicViscoelasticState, Array]:
        force = jnp.asarray(force_history_N_m3)
        if force.shape != (
            self.step_count,
            acquisition.sources.count,
            self.grid.dimensions,
        ):
            raise ValueError("Viscoelastic force history has wrong shape.")
        state = self.initial_state()
        observations = []
        for source in force:
            field = (
                jnp.stack(
                    [
                        acquisition.sources.transpose(source[:, component])
                        for component in range(self.grid.dimensions)
                    ]
                )
                / self.grid.cell_measure
            )
            state = self.step(state, density_kg_m3, stiffness, force_density_N_m3=field)
            observations.append(
                jnp.stack(
                    [
                        acquisition.receivers.apply(component)
                        for component in state.velocity_m_s
                    ],
                    axis=-1,
                )
            )
        return state, jnp.stack(observations)


__all__ = [
    "AnisotropicViscoelasticState",
    "ElasticStiffness",
    "PeriodicAnisotropicViscoelasticPlan",
    "StandardLinearSolidSpectrum",
]
