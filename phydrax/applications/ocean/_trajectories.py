#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import StateLayout, TrajectoryData
from ._hydrostatic import (
    _cell_from_faces,
    HydrostaticOceanState,
    PreparedHydrostaticOcean,
)


OCEAN_POSITION_LAYOUT = StateLayout(
    (3,),
    axes=("position_component",),
    component_names=("horizontal_0", "horizontal_1", "vertical"),
    layout_id="ocean:passive-position",
)


def lower_ocean_trajectories(
    coordinates: ArrayLike,
    positions: ArrayLike,
    /,
    *,
    sample_valid: ArrayLike | None = None,
    reset_mask: ArrayLike | None = None,
    source_id: str,
) -> TrajectoryData:
    return TrajectoryData(
        coordinates,
        positions,
        state_layout=OCEAN_POSITION_LAYOUT,
        sample_valid=sample_valid,
        reset_mask=reset_mask,
        coordinate_id="ocean-time",
        source_id=source_id,
        case_axes=("particle",),
        case_axis_roles=("case",),
    )


class PassiveOceanTrajectoryResult(StrictModule):
    trajectory: TrajectoryData
    sampled_velocity: Array
    exited: Array
    active_steps: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PassiveOceanTrajectoryPlan(StrictModule, NonTrainableState):
    """Fixed-capacity passive advection through one prepared ocean snapshot."""

    ocean: PreparedHydrostaticOcean
    maximum_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, ocean: PreparedHydrostaticOcean, maximum_steps: int, /):
        if not isinstance(ocean, PreparedHydrostaticOcean):
            raise TypeError("ocean must be a PreparedHydrostaticOcean.")
        capacity = int(maximum_steps)
        if capacity <= 0:
            raise ValueError("maximum_steps must be positive.")
        self.ocean = ocean
        self.maximum_steps = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "passive-ocean-trajectory",
                "ocean": ocean.prepared_id,
                "maximum_steps": capacity,
                "interpolation": "fixed-grid-nearest-cell",
            }
        )

    def _sample(
        self, state: HydrostaticOceanState, positions: Array, /
    ) -> tuple[Array, Array]:
        geometry = self.ocean.geometry
        view = self.ocean.view(state)
        u = _cell_from_faces(view.velocity[0], 0, geometry.periodic[0])
        v = _cell_from_faces(view.velocity[1], 1, geometry.periodic[1])
        vertical_velocity_face = view.vertical_flux / geometry.cell_area[..., None]
        w = 0.5 * (vertical_velocity_face[..., :-1] + vertical_velocity_face[..., 1:])
        first_nodes = geometry.longitude[:, 0]
        second_nodes = geometry.latitude[0, :]
        first = positions[:, 0]
        if geometry.periodic[0]:
            period = (
                2.0 * jnp.pi
                if geometry.horizontal_coordinate == "latitude-longitude"
                else (first_nodes[-1] - first_nodes[0])
            )
            delta = jnp.abs(first[:, None] - first_nodes[None, :])
            delta = jnp.minimum(delta, jnp.abs(period - delta))
        else:
            delta = jnp.abs(first[:, None] - first_nodes[None, :])
        first_index = jnp.argmin(delta, axis=1)
        second_index = jnp.argmin(
            jnp.abs(positions[:, 1, None] - second_nodes[None, :]), axis=1
        )
        depth = geometry.rest_depth[first_index, second_index]
        eta = state.eta[first_index, second_index]
        if geometry.vertical_coordinate == "zstar":
            cumulative_fraction = (
                jnp.cumsum(geometry.reference_layer_fraction).at[-1].set(1.0)
            )
            upper_boundaries = (
                -depth[:, None] + (depth + eta)[:, None] * cumulative_fraction[None, :]
            )
        else:
            epoch = geometry.metric_epoch(state.eta)
            layer_thickness = epoch.layer_thickness[first_index, second_index]
            upper_boundaries = -depth[:, None] + jnp.cumsum(layer_thickness, axis=-1)
        layer_index = jnp.sum(positions[:, 2, None] >= upper_boundaries, axis=-1).astype(
            jnp.int32
        )
        safe_layer = jnp.clip(layer_index, 0, geometry.cell_shape[-1] - 1)
        velocity = jnp.stack(
            (
                u[first_index, second_index, safe_layer],
                v[first_index, second_index, safe_layer],
                w[first_index, second_index, safe_layer],
            ),
            axis=-1,
        )
        horizontal_valid = (
            (
                jnp.ones_like(first, dtype=bool)
                if geometry.periodic[0]
                else (first >= first_nodes[0]) & (first <= first_nodes[-1])
            )
            & (positions[:, 1] >= second_nodes[0])
            & (positions[:, 1] <= second_nodes[-1])
        )
        vertical_valid = (positions[:, 2] >= -depth) & (positions[:, 2] <= eta)
        valid = (
            horizontal_valid
            & vertical_valid
            & (layer_index >= 0)
            & (layer_index < geometry.cell_shape[-1])
            & jnp.all(jnp.isfinite(velocity), axis=-1)
            & view.eos_successful
        )
        return velocity, valid

    def advect(
        self,
        initial_positions: ArrayLike,
        state: HydrostaticOceanState,
        step_size: ArrayLike,
        step_count: ArrayLike,
        /,
    ) -> PassiveOceanTrajectoryResult:
        positions = jnp.asarray(initial_positions, dtype=state.eta.dtype)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Passive ocean positions must have shape (particles, 3).")
        snapshot_view = self.ocean.view(state)
        dt = jnp.asarray(step_size, dtype=positions.dtype).reshape(())
        requested = jnp.asarray(step_count, dtype=jnp.int32).reshape(())
        capacity_valid = (requested >= 0) & (requested <= self.maximum_steps)
        particle_count = positions.shape[0]
        initial_active = jnp.all(jnp.isfinite(positions), axis=-1)

        def step(carry, index):
            current, active = carry
            velocity, inside = self._sample(state, current)
            execute = (index < requested) & active & inside & capacity_valid
            if self.ocean.geometry.horizontal_coordinate == "latitude-longitude":
                longitude_rate = velocity[:, 0] / (
                    self.ocean.geometry.radius
                    * jnp.maximum(jnp.cos(current[:, 1]), 1.0e-12)
                )
                latitude_rate = velocity[:, 1] / self.ocean.geometry.radius
                rate = jnp.stack((longitude_rate, latitude_rate, velocity[:, 2]), axis=-1)
            else:
                rate = velocity
            candidate = current + dt * rate
            _, candidate_inside = self._sample(state, candidate)
            next_positions = jnp.where(execute[:, None], candidate, current)
            candidate_valid = execute & candidate_inside
            next_active = active & jnp.where(
                index < requested,
                inside & jnp.where(execute, candidate_inside, True),
                True,
            )
            return (next_positions, next_active), (
                next_positions,
                velocity,
                execute,
                candidate_valid,
            )

        (_, final_active), history = jax.lax.scan(
            step,
            (positions, initial_active),
            jnp.arange(self.maximum_steps),
        )
        (
            position_history,
            velocity_history,
            executed,
            sample_valid_history,
        ) = history
        position_history = jnp.swapaxes(position_history, 0, 1)
        velocity_history = jnp.swapaxes(velocity_history, 0, 1)
        executed = jnp.swapaxes(executed, 0, 1)
        sample_valid_history = jnp.swapaxes(sample_valid_history, 0, 1)
        all_positions = jnp.concatenate((positions[:, None, :], position_history), axis=1)
        sample_valid = jnp.concatenate(
            (initial_active[:, None], sample_valid_history), axis=1
        )
        coordinates = jnp.broadcast_to(
            dt * jnp.arange(self.maximum_steps + 1, dtype=dt.dtype)[None, :],
            (particle_count, self.maximum_steps + 1),
        )
        trajectory = lower_ocean_trajectories(
            coordinates,
            all_positions,
            sample_valid=sample_valid,
            source_id=self.plan_id,
        )
        finite = (
            jnp.all(jnp.isfinite(all_positions))
            & jnp.all(jnp.isfinite(velocity_history))
            & jnp.isfinite(dt)
            & snapshot_view.eos_finite
        )
        successful = (
            finite
            & capacity_valid
            & snapshot_view.eos_valid
            & snapshot_view.eos_successful
            & jnp.all(final_active | ~initial_active)
        )
        return PassiveOceanTrajectoryResult(
            trajectory=trajectory,
            sampled_velocity=velocity_history,
            exited=initial_active & ~final_active,
            active_steps=jnp.sum(executed.astype(jnp.int32), axis=1),
            finite=finite,
            successful=successful,
            plan_id=self.plan_id,
        )


__all__ = [
    "OCEAN_POSITION_LAYOUT",
    "PassiveOceanTrajectoryPlan",
    "PassiveOceanTrajectoryResult",
    "lower_ocean_trajectories",
]
