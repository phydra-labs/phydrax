#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.vortex._filament import VortexFilamentState, VortexFilamentTopology
from ...solver._rotor import BladeElementRotorPlan, RotorResult


class ActuatorFlowResult(StrictModule):
    rotor: RotorResult
    bound_sources: VortexFilamentState
    shed_sources: VortexFilamentState
    sampled_velocity: Array
    load_power_residual: Array
    successful: Array
    actuator_id: str = eqx.field(static=True)


class ActuatorLineFlowPlan(StrictModule, NonTrainableState):
    rotor: BladeElementRotorPlan
    blade_azimuth: Array
    hub_position: Array
    axis: Array
    core_radius: float = eqx.field(static=True)
    actuator_id: str = eqx.field(static=True)

    def __init__(
        self,
        rotor: BladeElementRotorPlan,
        blade_azimuth: ArrayLike,
        hub_position: ArrayLike,
        axis: ArrayLike,
        /,
        *,
        core_radius: float,
    ):
        if not isinstance(rotor, BladeElementRotorPlan) or float(core_radius) <= 0.0:
            raise ValueError("Actuator line requires rotor and positive core radius.")
        azimuth, hub, axis_ = (
            jnp.asarray(blade_azimuth, dtype=float),
            jnp.asarray(hub_position, dtype=float),
            jnp.asarray(axis, dtype=float),
        )
        if (
            azimuth.shape != (rotor.blade_count,)
            or hub.shape != (3,)
            or axis_.shape != (3,)
        ):
            raise ValueError("Actuator blade/hub/axis shapes are invalid.")
        norm = jnp.linalg.norm(axis_)
        axis_ = eqx.error_if(
            axis_, ~jnp.isfinite(norm) | (norm <= 0.0), "Actuator axis is invalid."
        )
        self.rotor, self.blade_azimuth, self.hub_position, self.axis = (
            rotor,
            azimuth,
            hub,
            axis_ / norm,
        )
        self.core_radius = float(core_radius)
        self.actuator_id = canonical_fingerprint(
            {
                "kind": "actuator-line-flow",
                "rotor": rotor.rotor_id,
                "blade_azimuth": tuple(float(value) for value in azimuth),
                "core_radius": self.core_radius,
            }
        )

    def _basis(self):
        reference = jnp.where(
            jnp.abs(self.axis[0]) < 0.9,
            jnp.asarray((1.0, 0.0, 0.0)),
            jnp.asarray((0.0, 1.0, 0.0)),
        )
        first = jnp.cross(self.axis, reference)
        first = first / jnp.linalg.norm(first)
        second = jnp.cross(self.axis, first)
        return first, second

    def solve(
        self,
        sampled_velocity: ArrayLike,
        angular_velocity: ArrayLike,
        collective_pitch: ArrayLike = 0.0,
        /,
        *,
        reynolds: ArrayLike = 1.0e6,
        mach: ArrayLike = 0.0,
    ) -> ActuatorFlowResult:
        velocity = jnp.asarray(sampled_velocity, dtype=self.rotor.radius.dtype)
        if velocity.shape != (self.rotor.blade_count, self.rotor.radius.size, 3):
            raise ValueError("Actuator sampled velocity must have blade/station/3 shape.")
        axial = jnp.mean(jnp.sum(velocity * self.axis, axis=-1))
        rotor_result = self.rotor.solve(
            axial, angular_velocity, collective_pitch, reynolds=reynolds, mach=mach
        )
        first, second = self._basis()
        vertices = []
        strengths = []
        segments = []
        shed_vertices = []
        shed_segments = []
        shed_strengths = []
        for blade in range(self.rotor.blade_count):
            radial = (
                jnp.cos(self.blade_azimuth[blade]) * first
                + jnp.sin(self.blade_azimuth[blade]) * second
            )
            blade_points = self.hub_position + self.rotor.radius[:, None] * radial
            base = len(vertices)
            vertices.extend(tuple(blade_points))
            wake_base = len(shed_vertices)
            shed_vertices.extend(tuple(blade_points))
            shed_vertices.extend(
                tuple(blade_points + self.axis * self.rotor.section_width[:, None])
            )
            for station in range(self.rotor.radius.size - 1):
                segments.append((base + station, base + station + 1))
                strengths.append(rotor_result.bound_circulation[station])
                shed_segments.append(
                    (wake_base + station, wake_base + self.rotor.radius.size + station)
                )
                shed_strengths.append(
                    rotor_result.bound_circulation[station]
                    - rotor_result.bound_circulation[station + 1]
                )
        bound_topology = VortexFilamentTopology.from_segments(
            tuple(segments), vertex_capacity=len(vertices)
        )
        shed_topology = VortexFilamentTopology.from_segments(
            tuple(shed_segments), vertex_capacity=len(shed_vertices)
        )
        bound = VortexFilamentState(
            bound_topology,
            jnp.stack(tuple(vertices)),
            jnp.stack(tuple(strengths)),
            jnp.full((len(segments),), self.core_radius),
        )
        shed = VortexFilamentState(
            shed_topology,
            jnp.stack(tuple(shed_vertices)),
            jnp.stack(tuple(shed_strengths)),
            jnp.full((len(shed_segments),), self.core_radius),
        )
        aerodynamic_power = jnp.sum(
            rotor_result.section_force[..., 1] * self.rotor.radius
        ) * jnp.asarray(angular_velocity)
        residual = rotor_result.power - aerodynamic_power
        successful = rotor_result.successful & jnp.isfinite(residual)
        return ActuatorFlowResult(
            rotor_result, bound, shed, velocity, residual, successful, self.actuator_id
        )


__all__ = ["ActuatorFlowResult", "ActuatorLineFlowPlan"]
