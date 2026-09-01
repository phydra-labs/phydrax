#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._incident_wave import IncidentWavePlan, WaveSample
from ...solver._mac_ale import MACALEStageGeometry
from ._free_surface_ale import FaceTuple, PreparedGraphSurfaceALE


class WaveGaugeDiagnostics(StrictModule):
    incident_variance: Array
    reflected_variance: Array
    reflection_coefficient: Array
    amplitude_error: Array
    phase_error: Array
    finite: Array
    valid: Array


class ActiveAbsorptionState(StrictModule):
    eta_history: Array
    target_history: Array
    time_history: Array
    write_index: Array
    correction: Array


class WaveForcingResult(StrictModule):
    momentum_rate: FaceTuple
    eta_rate_source: Array
    prescribed_velocity: FaceTuple
    surface_pressure_head: Array
    wave_work_rate: Array
    relaxation_work_rate: Array
    sponge_dissipation_rate: Array
    controller_work_rate: Array
    sample: WaveSample
    finite: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


class WaveForcingPlan(StrictModule, NonTrainableState):
    """Incident-wave boundary, relaxation, sponge, and active absorption owner."""

    provider: IncidentWavePlan
    generation_weight: Array
    sponge_weight: Array
    boundary_axis: int = eqx.field(static=True)
    boundary_side: Literal["lower", "upper"] = eqx.field(static=True)
    return_flow: bool = eqx.field(static=True)
    active_gain: float = eqx.field(static=True)
    history_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider: IncidentWavePlan,
        generation_weight: ArrayLike,
        sponge_weight: ArrayLike,
        /,
        *,
        boundary_axis: int = 0,
        boundary_side: Literal["lower", "upper"] = "lower",
        return_flow: bool = True,
        active_gain: float = 0.0,
        history_size: int = 128,
    ):
        if not isinstance(provider, IncidentWavePlan):
            raise TypeError("provider must be IncidentWavePlan.")
        generation = jnp.asarray(generation_weight)
        sponge = jnp.asarray(sponge_weight, dtype=generation.dtype)
        if generation.shape != sponge.shape or generation.ndim != 3:
            raise ValueError("Wave generation/sponge weights need one 3-D cell shape.")
        if bool(
            jnp.any(~jnp.isfinite(generation))
            | jnp.any(~jnp.isfinite(sponge))
            | jnp.any((generation < 0.0) | (generation > 1.0))
            | jnp.any(sponge < 0.0)
        ):
            raise ValueError("Wave generation/sponge weights are invalid.")
        axis = int(boundary_axis)
        gain = float(active_gain)
        size = int(history_size)
        if axis not in (0, 1) or boundary_side not in ("lower", "upper"):
            raise ValueError("Wave boundary must be one horizontal side.")
        if not np.isfinite(gain) or gain < 0.0 or size <= 1:
            raise ValueError("Wave absorption gain/history are invalid.")
        self.provider = provider
        self.generation_weight = generation
        self.sponge_weight = sponge
        self.boundary_axis = axis
        self.boundary_side = boundary_side
        self.return_flow = bool(return_flow)
        self.active_gain = gain
        self.history_size = size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wave-forcing-plan",
                "provider": provider.provider_id,
                "generation": array_tree_fingerprint(np.asarray(generation)),
                "sponge": array_tree_fingerprint(np.asarray(sponge)),
                "axis": axis,
                "side": boundary_side,
                "return_flow": bool(return_flow),
                "active_gain": gain,
                "history_size": size,
            }
        )

    def initial_controller_state(
        self, horizontal_shape: tuple[int, int], dtype
    ) -> ActiveAbsorptionState:
        return ActiveAbsorptionState(
            eta_history=jnp.zeros((self.history_size,) + horizontal_shape, dtype=dtype),
            target_history=jnp.zeros(
                (self.history_size,) + horizontal_shape, dtype=dtype
            ),
            time_history=jnp.zeros((self.history_size,), dtype=dtype),
            write_index=jnp.asarray(0, dtype=jnp.int32),
            correction=jnp.zeros(horizontal_shape, dtype=dtype),
        )

    def update_controller(
        self,
        state: ActiveAbsorptionState,
        time: ArrayLike,
        measured_eta: ArrayLike,
        target_eta: ArrayLike,
        /,
    ) -> ActiveAbsorptionState:
        measured = jnp.asarray(measured_eta)
        target = jnp.asarray(target_eta, dtype=measured.dtype)
        index = state.write_index % self.history_size
        eta_history = state.eta_history.at[index].set(measured)
        target_history = state.target_history.at[index].set(target)
        time_history = state.time_history.at[index].set(jnp.asarray(time))
        correction = -self.active_gain * (measured - target)
        return ActiveAbsorptionState(
            eta_history,
            target_history,
            time_history,
            state.write_index + 1,
            correction,
        )

    def diagnostics(self, state: ActiveAbsorptionState, /) -> WaveGaugeDiagnostics:
        incident = state.target_history
        reflected = state.eta_history - incident
        incident_variance = jnp.mean(incident**2)
        reflected_variance = jnp.mean(reflected**2)
        reflection = jnp.sqrt(
            jnp.where(
                incident_variance > 0.0, reflected_variance / incident_variance, 0.0
            )
        )
        amplitude_error = jnp.sqrt(jnp.mean(state.eta_history**2)) - jnp.sqrt(
            incident_variance
        )
        covariance = jnp.mean(state.eta_history * incident)
        phase_error = jnp.arccos(
            jnp.clip(
                covariance
                / jnp.sqrt(
                    jnp.maximum(
                        jnp.mean(state.eta_history**2) * incident_variance,
                        jnp.finfo(incident.dtype).tiny,
                    )
                ),
                -1.0,
                1.0,
            )
        )
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        incident_variance,
                        reflected_variance,
                        reflection,
                        amplitude_error,
                        phase_error,
                    )
                )
            )
        )
        return WaveGaugeDiagnostics(
            incident_variance,
            reflected_variance,
            reflection,
            amplitude_error,
            phase_error,
            finite,
            finite,
        )

    def evaluate(
        self,
        surface: PreparedGraphSurfaceALE,
        geometry: MACALEStageGeometry,
        time: ArrayLike,
        velocity: FaceTuple,
        eta: Array,
        controller: ActiveAbsorptionState | None = None,
        /,
    ) -> WaveForcingResult:
        centers = geometry.cell_centers
        sample = self.provider.sample(time, centers)
        reconstructed = geometry.reconstruct_cell_velocity(velocity)
        target = sample.velocity
        relaxation_acceleration = self.generation_weight[..., None] * (
            target - reconstructed
        )
        sponge_acceleration = -self.sponge_weight[..., None] * (
            reconstructed
            - jnp.asarray(
                (self.provider.current[0], self.provider.current[1], 0.0),
                dtype=reconstructed.dtype,
            )
        )
        total_cell_acceleration = relaxation_acceleration + sponge_acceleration
        momentum_rate = tuple(
            geometry.face_dual_measures[axis]
            * _cell_to_faces(
                total_cell_acceleration[..., axis],
                axis,
                surface.plan.reference.grid.structured_axes[axis].periodic,
            )
            for axis in range(3)
        )
        prescribed = [jnp.zeros_like(value) for value in geometry.face_measures]
        boundary_index = 0 if self.boundary_side == "lower" else -1
        face_coordinates = jnp.take(
            geometry.face_centers[self.boundary_axis],
            boundary_index,
            axis=self.boundary_axis,
        )
        boundary_sample = self.provider.sample(time, face_coordinates)
        boundary_normal = boundary_sample.velocity[..., self.boundary_axis]
        if self.return_flow:
            boundary_normal = boundary_normal - jnp.mean(boundary_normal)
        if controller is not None:
            correction = jnp.take(
                controller.correction,
                0 if self.boundary_side == "lower" else -1,
                axis=self.boundary_axis,
            )
            boundary_normal = boundary_normal + correction[..., None]
        location = [slice(None)] * prescribed[self.boundary_axis].ndim
        location[self.boundary_axis] = boundary_index
        prescribed[self.boundary_axis] = (
            prescribed[self.boundary_axis].at[tuple(location)].set(boundary_normal)
        )
        target_surface = self.provider.sample(
            time,
            jnp.concatenate(
                (
                    geometry.cell_centers[..., -1, :2],
                    eta[..., None],
                ),
                axis=-1,
            ),
        )
        eta_rate_source = jnp.zeros_like(eta)
        wave_power = jnp.sum(
            geometry.cell_volumes
            * jnp.sum(reconstructed * relaxation_acceleration, axis=-1)
        )
        sponge_dissipation = -jnp.sum(
            geometry.cell_volumes * jnp.sum(reconstructed * sponge_acceleration, axis=-1)
        )
        controller_work = jnp.asarray(0.0, dtype=eta.dtype)
        if controller is not None:
            controller_work = jnp.sum(controller.correction * eta_rate_source)
        finite = (
            sample.finite
            & boundary_sample.finite
            & target_surface.finite
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in momentum_rate))
            )
            & jnp.all(jnp.isfinite(eta_rate_source))
        )
        return WaveForcingResult(
            momentum_rate=momentum_rate,
            eta_rate_source=eta_rate_source,
            prescribed_velocity=tuple(prescribed),
            surface_pressure_head=target_surface.pressure_head,
            wave_work_rate=wave_power,
            relaxation_work_rate=wave_power,
            sponge_dissipation_rate=jnp.maximum(sponge_dissipation, 0.0),
            controller_work_rate=controller_work,
            sample=sample,
            finite=finite,
            valid=finite,
            plan_id=self.plan_id,
        )


def _cell_to_faces(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        faces = 0.5 * (jnp.roll(moved, 1, axis=0) + moved)
    else:
        interior = 0.5 * (moved[:-1] + moved[1:])
        faces = jnp.concatenate((moved[:1], interior, moved[-1:]), axis=0)
    return jnp.moveaxis(faces, 0, axis)


__all__ = [
    "ActiveAbsorptionState",
    "WaveForcingPlan",
    "WaveForcingResult",
    "WaveGaugeDiagnostics",
]
