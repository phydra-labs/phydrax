#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._acquisition import AcousticGrid, SeismicAcquisition
from ._constant_density import _gradient
from ._variable_density import VariableDensityAcousticPlan


class CartesianCPML(StrictModule, NonTrainableState):
    cell_a: tuple[Array, ...]
    cell_b: tuple[Array, ...]
    cell_kappa: tuple[Array, ...]
    face_a: tuple[Array, ...]
    face_b: tuple[Array, ...]
    face_kappa: tuple[Array, ...]
    widths: tuple[tuple[int, int], ...] = eqx.field(static=True)
    target_reflection: float = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: AcousticGrid,
        time_step_s: float,
        maximum_wavespeed_m_s: float,
        /,
        *,
        widths: tuple[tuple[int, int], ...],
        target_reflection: float = 1e-6,
        polynomial_order: int = 3,
        maximum_kappa: float = 5.0,
        alpha_maximum_Hz: float = 0.0,
    ):
        if len(widths) != grid.dimensions or any(
            len(value) != 2
            or value[0] < 0
            or value[1] < 0
            or value[0] + value[1] >= grid.shape[axis]
            for axis, value in enumerate(widths)
        ):
            raise ValueError("CPML widths must leave a nonempty interior on every axis.")
        reflection = float(target_reflection)
        order, kappa, alpha = (
            int(polynomial_order),
            float(maximum_kappa),
            float(alpha_maximum_Hz),
        )
        if (
            not np.isfinite(reflection)
            or not 0 < reflection < 1
            or order < 1
            or not np.isfinite(kappa)
            or kappa < 1
            or not np.isfinite(alpha)
            or alpha < 0
        ):
            raise ValueError("CPML reflection/order/kappa/alpha parameters are invalid.")
        cell_a, cell_b, cell_kappa = [], [], []
        face_a, face_b, face_kappa = [], [], []
        for axis, (lower, upper) in enumerate(widths):
            for face, a_out, b_out, k_out in (
                (False, cell_a, cell_b, cell_kappa),
                (True, face_a, face_b, face_kappa),
            ):
                size = grid.shape[axis] + int(face)
                coordinate = np.arange(size, dtype=float) + (0.0 if face else 0.5)
                depth = np.zeros(size)
                if lower:
                    depth = np.maximum(depth, (lower - coordinate) / lower)
                if upper:
                    depth = np.maximum(
                        depth, (coordinate - (grid.shape[axis] - upper)) / upper
                    )
                depth = np.clip(depth, 0.0, 1.0)
                thickness = max(lower, upper, 1) * grid.spacing[axis]
                sigma_max = (
                    -(order + 1)
                    * maximum_wavespeed_m_s
                    * np.log(reflection)
                    / (2 * thickness)
                )
                sigma = sigma_max * depth**order
                kappa_profile = 1.0 + (kappa - 1.0) * depth**order
                alpha_profile = alpha * (1.0 - depth) * (depth > 0)
                decay = np.exp(-(sigma / kappa_profile + alpha_profile) * time_step_s)
                denominator = sigma + kappa_profile * alpha_profile
                safe_denominator = np.where(denominator > 0, denominator, 1.0)
                coefficient = sigma * (decay - 1.0) / (kappa_profile * safe_denominator)
                coefficient = np.where(denominator > 0, coefficient, 0.0)
                shape = [1] * grid.dimensions
                shape[axis] = size
                a_out.append(jnp.asarray(coefficient.reshape(shape)))
                b_out.append(jnp.asarray(decay.reshape(shape)))
                k_out.append(jnp.asarray(kappa_profile.reshape(shape)))
        self.cell_a, self.cell_b, self.cell_kappa = (
            tuple(cell_a),
            tuple(cell_b),
            tuple(cell_kappa),
        )
        self.face_a, self.face_b, self.face_kappa = (
            tuple(face_a),
            tuple(face_b),
            tuple(face_kappa),
        )
        self.widths, self.target_reflection = widths, reflection
        self.profile_id = canonical_fingerprint(
            {
                "kind": "cartesian-cpml",
                "grid": grid.grid_id,
                "time_step_s": time_step_s,
                "maximum_wavespeed_m_s": maximum_wavespeed_m_s,
                "widths": widths,
                "target_reflection": reflection,
                "polynomial_order": order,
                "maximum_kappa": kappa,
                "alpha_maximum_Hz": alpha,
            }
        )


class CPMLAcousticState(StrictModule):
    pressure_Pa: Array
    velocity_m_s: tuple[Array, ...]
    gradient_memory: tuple[Array, ...]
    divergence_memory: tuple[Array, ...]
    step_index: Array
    plan_id: str = eqx.field(static=True)


class CPMLAcousticSimulation(StrictModule):
    traces_Pa: Array
    final_state: CPMLAcousticState
    finite: Array


class TractionFreeBoundary(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    normal_axis: int = eqx.field(static=True)
    side: Literal["lower", "upper"] = eqx.field(static=True)
    voigt_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)

    def __init__(
        self,
        dimension: Literal[2, 3],
        normal_axis: int,
        side: Literal["lower", "upper"],
        /,
    ):
        dimension_, axis = int(dimension), int(normal_axis)
        if (
            dimension_ not in (2, 3)
            or not 0 <= axis < dimension_
            or side not in ("lower", "upper")
        ):
            raise ValueError("Traction-free dimension, axis, or side is invalid.")
        self.dimension, self.normal_axis, self.side = dimension_, axis, side
        self.voigt_pairs = (
            ((0, 0), (1, 1), (0, 1))
            if dimension_ == 2
            else ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        )

    def apply(self, stress_voigt: ArrayLike, /) -> Array:
        stress = jnp.asarray(stress_voigt)
        if stress.ndim != self.dimension + 1 or stress.shape[0] != len(self.voigt_pairs):
            raise ValueError("Traction-free stress array has wrong shape.")
        index = 0 if self.side == "lower" else stress.shape[1 + self.normal_axis] - 1
        result = stress
        for component, pair in enumerate(self.voigt_pairs):
            if self.normal_axis in pair:
                selection: list[slice | int] = [slice(None)] * stress.ndim
                selection[0], selection[1 + self.normal_axis] = component, index
                result = result.at[tuple(selection)].set(0.0)
        return result


class CPMLVariableDensityAcousticPlan(StrictModule, NonTrainableState):
    acoustic: VariableDensityAcousticPlan
    cpml: CartesianCPML
    free_surface_axis: int | None = eqx.field(static=True)
    free_surface_side: Literal["lower", "upper"] | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        acoustic: VariableDensityAcousticPlan,
        cpml: CartesianCPML,
        /,
        *,
        free_surface_axis: int | None = None,
        free_surface_side: Literal["lower", "upper"] | None = None,
    ):
        if not isinstance(acoustic, VariableDensityAcousticPlan) or not isinstance(
            cpml, CartesianCPML
        ):
            raise TypeError("CPML acoustic plan requires acoustic and CPML plans.")
        if (free_surface_axis is None) != (free_surface_side is None):
            raise ValueError(
                "Acoustic free surface axis and side must be supplied together."
            )
        if free_surface_axis is not None:
            axis = int(free_surface_axis)
            if not 0 <= axis < acoustic.grid.dimensions or free_surface_side not in (
                "lower",
                "upper",
            ):
                raise ValueError("Acoustic free-surface axis or side is invalid.")
            width = cpml.widths[axis][0 if free_surface_side == "lower" else 1]
            if width != 0:
                raise ValueError("A physical free surface cannot also carry CPML width.")
        self.acoustic, self.cpml = acoustic, cpml
        self.free_surface_axis, self.free_surface_side = (
            free_surface_axis,
            free_surface_side,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cpml-variable-density-acoustic",
                "acoustic": acoustic.plan_id,
                "cpml": cpml.profile_id,
                "free_surface": (free_surface_axis, free_surface_side),
            }
        )

    def initial_state(self) -> CPMLAcousticState:
        grid = self.acoustic.grid
        velocity, gradient, divergence = [], [], []
        for axis in range(grid.dimensions):
            shape = list(grid.shape)
            shape[axis] += 1
            velocity.append(jnp.zeros(tuple(shape)))
            gradient.append(jnp.zeros(tuple(shape)))
            divergence.append(jnp.zeros(grid.shape))
        return CPMLAcousticState(
            jnp.zeros(grid.shape),
            tuple(velocity),
            tuple(gradient),
            tuple(divergence),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def _free_surface(self, pressure: Array) -> Array:
        if self.free_surface_axis is None:
            return pressure
        axis = self.free_surface_axis
        index = 0 if self.free_surface_side == "lower" else pressure.shape[axis] - 1
        selection: list[slice | int] = [slice(None)] * pressure.ndim
        selection[axis] = index
        return pressure.at[tuple(selection)].set(0.0)

    def step(
        self,
        state: CPMLAcousticState,
        wavespeed: ArrayLike,
        density_kg_m3: ArrayLike,
        source_density_s_inverse: ArrayLike,
        /,
    ) -> CPMLAcousticState:
        if state.plan_id != self.plan_id:
            raise ValueError("CPML acoustic state belongs to another plan.")
        speed, density = self.acoustic._materials(wavespeed, density_kg_m3)
        source = jnp.broadcast_to(
            jnp.asarray(source_density_s_inverse), self.acoustic.grid.shape
        )
        dt = self.acoustic.time_step
        velocity, gradient_memory = [], []
        for axis, spacing in enumerate(self.acoustic.grid.spacing):
            derivative = _gradient(state.pressure_Pa, axis, spacing)
            memory = (
                self.cpml.face_b[axis] * state.gradient_memory[axis]
                + self.cpml.face_a[axis] * derivative
            )
            effective = derivative / self.cpml.face_kappa[axis] + memory
            value = state.velocity_m_s[
                axis
            ] - dt * effective / self.acoustic._face_density(density, axis)
            selection_lower: list[slice | int] = [slice(None)] * value.ndim
            selection_upper: list[slice | int] = [slice(None)] * value.ndim
            selection_lower[axis], selection_upper[axis] = 0, value.shape[axis] - 1
            value = value.at[tuple(selection_lower)].set(0.0)
            value = value.at[tuple(selection_upper)].set(0.0)
            velocity.append(value)
            gradient_memory.append(memory)
        pressure_rate = jnp.zeros(self.acoustic.grid.shape)
        divergence_memory = []
        for axis, spacing in enumerate(self.acoustic.grid.spacing):
            derivative = jnp.diff(velocity[axis], axis=axis) / spacing
            memory = (
                self.cpml.cell_b[axis] * state.divergence_memory[axis]
                + self.cpml.cell_a[axis] * derivative
            )
            pressure_rate = (
                pressure_rate + derivative / self.cpml.cell_kappa[axis] + memory
            )
            divergence_memory.append(memory)
        pressure = state.pressure_Pa - dt * density * speed**2 * (pressure_rate - source)
        pressure = self._free_surface(pressure)
        return CPMLAcousticState(
            pressure,
            tuple(velocity),
            tuple(gradient_memory),
            tuple(divergence_memory),
            state.step_index + 1,
            self.plan_id,
        )

    def simulate(
        self,
        wavespeed: ArrayLike,
        density_kg_m3: ArrayLike,
        acquisition: SeismicAcquisition,
        source_rates: ArrayLike,
        /,
    ) -> CPMLAcousticSimulation:
        rates = jnp.asarray(source_rates)
        if rates.shape != (self.acoustic.step_count, acquisition.sources.count):
            raise ValueError("CPML acoustic source history has wrong shape.")
        state = self.initial_state()
        traces = [acquisition.receivers.apply(state.pressure_Pa)]
        for rate in rates:
            source = acquisition.sources.transpose(rate) / self.acoustic.grid.cell_measure
            state = self.step(state, wavespeed, density_kg_m3, source)
            traces.append(acquisition.receivers.apply(state.pressure_Pa))
        values = jnp.stack(traces, axis=1)
        finite = jnp.all(jnp.isfinite(values))
        return CPMLAcousticSimulation(values, state, finite)


__all__ = [
    "CPMLAcousticSimulation",
    "CPMLAcousticState",
    "CPMLVariableDensityAcousticPlan",
    "CartesianCPML",
    "TractionFreeBoundary",
]
