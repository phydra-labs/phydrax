#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._assembly import ParticleInteractionLedger
from ._bipartite_neighborhood import BipartiteNeighborhoodState
from ._core import ParticleDiscretization, ParticleSetPlan
from ._pairwise import ParticleBox
from ._smoothing import AbstractSPHSmoothingKernel


WallSlipPolicy: TypeAlias = Literal["no-slip", "free-slip"]


class WallParticleQualityReport(StrictModule):
    particle_count: int = eqx.field(static=True)
    minimum_volume: Array
    maximum_volume: Array
    minimum_spacing: Array
    normal_defect: Array
    report_id: str = eqx.field(static=True)


class PreparedWallParticles(StrictModule, NonTrainableState):
    positions: Array
    normals: Array
    volumes: Array
    particles: ParticleDiscretization
    quality: WallParticleQualityReport
    prepared_id: str = eqx.field(static=True)


class WallParticleGenerationPlan(StrictModule, NonTrainableState):
    geometry: Any
    kernel: AbstractSPHSmoothingKernel
    spacing: float = eqx.field(static=True)
    smoothing_length: float = eqx.field(static=True)
    layers: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: Any,
        kernel: AbstractSPHSmoothingKernel,
        spacing: float,
        smoothing_length: float,
        /,
        *,
        layers: int = 2,
    ):
        spacing_ = float(spacing)
        smoothing = float(smoothing_length)
        layers_ = int(layers)
        if spacing_ <= 0.0 or smoothing <= 0.0 or layers_ <= 0:
            raise ValueError(
                "Wall spacing, smoothing length, and layers must be positive."
            )
        if kernel.dimension <= 0:
            raise ValueError("Wall kernel dimension must be positive.")
        self.geometry = geometry
        self.kernel = kernel
        self.spacing = spacing_
        self.smoothing_length = smoothing
        self.layers = layers_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wall-particle-generation",
                "kernel": kernel.kernel_id,
                "spacing": spacing_,
                "smoothing_length": smoothing,
                "layers": layers_,
            }
        )

    def prepare(self) -> PreparedWallParticles:
        bounds = np.asarray(self.geometry.bounds, dtype=float)
        if bounds.shape != (2, self.kernel.dimension):
            raise ValueError("Geometry bounds must have shape (2, dimension).")
        axes = tuple(
            np.arange(bounds[0, axis], bounds[1, axis] + 0.5 * self.spacing, self.spacing)
            for axis in range(self.kernel.dimension)
        )
        grid = np.meshgrid(*axes, indexing="ij")
        candidates = np.stack(tuple(axis.reshape(-1) for axis in grid), axis=-1)
        field = np.asarray(self.geometry.signed_distance(jnp.asarray(candidates)))
        selected = np.abs(field) <= 0.55 * self.spacing
        if not np.any(selected):
            raise ValueError("Wall generation found no boundary samples.")
        surface_candidates = candidates[selected]
        distance = field[selected]
        normal = np.asarray(
            self.geometry.boundary_normal(jnp.asarray(surface_candidates)), dtype=float
        )
        surface = surface_candidates - distance[:, None] * normal
        positions = np.concatenate(
            tuple(
                surface + layer * self.spacing * normal for layer in range(self.layers)
            ),
            axis=0,
        )
        normals = np.concatenate((normal,) * self.layers, axis=0)
        quantized = np.round(positions / (0.1 * self.spacing)).astype(np.int64)
        _, unique = np.unique(quantized, axis=0, return_index=True)
        positions = positions[np.sort(unique)]
        normals = normals[np.sort(unique)]
        displacement = positions[:, None, :] - positions[None, :, :]
        distance_matrix = np.sqrt(np.sum(displacement * displacement, axis=-1))
        kernel_sum = np.sum(
            np.asarray(
                self.kernel.value(jnp.asarray(distance_matrix), self.smoothing_length)
            ),
            axis=1,
        )
        if np.any(kernel_sum <= 0.0) or np.any(~np.isfinite(kernel_sum)):
            raise ValueError("Wall volume kernel sum is invalid.")
        volumes = 1.0 / kernel_sum
        pair_distance = np.where(
            np.eye(positions.shape[0], dtype=bool), np.inf, distance_matrix
        )
        minimum_spacing = np.min(pair_distance)
        normal_norm = np.sqrt(np.sum(normals * normals, axis=-1))
        normal_defect = np.max(np.abs(normal_norm - 1.0))
        particles = ParticleSetPlan(
            np.arange(positions.shape[0]),
            volumes,
            ambient_dimension=self.kernel.dimension,
            name="wall-particles",
        ).prepare()
        report_id = canonical_fingerprint(
            {
                "kind": "wall-particle-quality",
                "plan": self.plan_id,
                "particle_count": int(positions.shape[0]),
                "minimum_volume": float(np.min(volumes)),
                "maximum_volume": float(np.max(volumes)),
                "minimum_spacing": float(minimum_spacing),
                "normal_defect": float(normal_defect),
            }
        )
        quality = WallParticleQualityReport(
            int(positions.shape[0]),
            jnp.asarray(np.min(volumes)),
            jnp.asarray(np.max(volumes)),
            jnp.asarray(minimum_spacing),
            jnp.asarray(normal_defect),
            report_id,
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-wall-particles",
                "plan": self.plan_id,
                "particles": particles.prepared_id,
                "quality": report_id,
            }
        )
        return PreparedWallParticles(
            jnp.asarray(positions),
            jnp.asarray(normals),
            jnp.asarray(volumes),
            particles,
            quality,
            prepared_id,
        )


class PrescribedWallMotion(StrictModule, NonTrainableState):
    position: Callable[[Array, Array, Any], Array]
    velocity: Callable[[Array, Array, Any], Array]
    acceleration: Callable[[Array, Array, Any], Array]
    motion_id: str = eqx.field(static=True)

    @classmethod
    def stationary(cls) -> "PrescribedWallMotion":
        def position(time, reference, args):
            del time, args
            return reference

        def zero(time, reference, args):
            del time, args
            return jnp.zeros_like(reference)

        return cls(position, zero, zero, "wall-motion:stationary")


class AdamiWallBoundaryPlan(StrictModule, NonTrainableState):
    material: Any
    slip: WallSlipPolicy = eqx.field(static=True)
    atmospheric_pressure: float = eqx.field(static=True)
    kinematic_viscosity: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        material: Any,
        /,
        *,
        slip: WallSlipPolicy = "no-slip",
        atmospheric_pressure: float = 0.0,
        kinematic_viscosity: float = 0.0,
    ):
        if slip not in ("no-slip", "free-slip"):
            raise ValueError("Wall slip policy must be 'no-slip' or 'free-slip'.")
        if (
            not np.isfinite(atmospheric_pressure)
            or not np.isfinite(kinematic_viscosity)
            or kinematic_viscosity < 0.0
        ):
            raise ValueError("Wall pressure and viscosity must be finite and valid.")
        self.material = material
        self.slip = slip
        self.atmospheric_pressure = float(atmospheric_pressure)
        self.kinematic_viscosity = float(kinematic_viscosity)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "adami-wall-boundary",
                "material": material.material_id,
                "slip": slip,
                "atmospheric_pressure": atmospheric_pressure,
                "kinematic_viscosity": kinematic_viscosity,
            }
        )


class WallInteractionResult(StrictModule):
    fluid_force: Array
    wall_reaction: Array
    wall_pressure: Array
    wall_density: Array
    wall_velocity: Array
    support_denominator: Array
    ledger: ParticleInteractionLedger
    successful: Array


def evaluate_wall_interaction(
    plan: AdamiWallBoundaryPlan,
    wall: PreparedWallParticles,
    relation_state: BipartiteNeighborhoodState,
    fluid_position: ArrayLike,
    fluid_velocity: ArrayLike,
    fluid_density: ArrayLike,
    fluid_pressure: ArrayLike,
    fluid_mass: ArrayLike,
    kernel: AbstractSPHSmoothingKernel,
    smoothing_length: float,
    /,
    *,
    time: ArrayLike = 0.0,
    args: Any = None,
    motion: PrescribedWallMotion | None = None,
    gravity: ArrayLike | None = None,
    box: ParticleBox | None = None,
) -> WallInteractionResult:
    if not relation_state.successful:
        raise ValueError("Bipartite wall relation is unsuccessful.")
    relation = relation_state.relation
    fluid_x = jnp.asarray(fluid_position)
    fluid_v = jnp.asarray(fluid_velocity)
    density = jnp.asarray(fluid_density)
    pressure = jnp.asarray(fluid_pressure)
    mass = jnp.asarray(fluid_mass)
    motion_ = PrescribedWallMotion.stationary() if motion is None else motion
    wall_x = motion_.position(jnp.asarray(time), wall.positions, args)
    wall_v = motion_.velocity(jnp.asarray(time), wall.positions, args)
    wall_a = motion_.acceleration(jnp.asarray(time), wall.positions, args)
    target = relation.target_indices
    source = relation.source_indices
    displacement = fluid_x[target] - wall_x[source]
    if box is not None:
        displacement = box.minimum_image(displacement)
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    valid = relation.valid & (distance < kernel.support_radius(smoothing_length))
    weights = jnp.where(valid, kernel.value(distance, smoothing_length), 0.0)
    fluid_volume = mass / density
    gravity_ = (
        jnp.zeros((fluid_x.shape[-1],)) if gravity is None else jnp.asarray(gravity)
    )
    hydrostatic = jnp.sum(
        (gravity_ - wall_a[source]) * (wall_x[source] - fluid_x[target]), axis=-1
    )
    denominator = (
        jnp.zeros((wall.positions.shape[0],), dtype=fluid_x.dtype)
        .at[source]
        .add(fluid_volume[target] * weights)
    )
    pressure_numerator = (
        jnp.zeros_like(denominator)
        .at[source]
        .add(
            fluid_volume[target]
            * (pressure[target] + density[target] * hydrostatic)
            * weights
        )
    )
    safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
    wall_pressure = jnp.where(
        denominator > 0.0,
        pressure_numerator / safe_denominator,
        plan.atmospheric_pressure,
    )
    wall_density = plan.material.density_from_pressure(wall_pressure)
    velocity_numerator = (
        jnp.zeros_like(wall_v)
        .at[source]
        .add(fluid_volume[target, None] * fluid_v[target] * weights[:, None])
    )
    averaged_velocity = velocity_numerator / safe_denominator[:, None]
    if plan.slip == "no-slip":
        ghost_velocity = 2.0 * wall_v - averaged_velocity
    else:
        relative = averaged_velocity - wall_v
        ghost_velocity = (
            averaged_velocity
            - 2.0 * jnp.sum(relative * wall.normals, axis=-1)[:, None] * wall.normals
        )
    gradient = kernel.gradient(displacement, distance, smoothing_length)
    fluid_pair_volume = fluid_volume[target]
    wall_pair_volume = wall.volumes[source]
    pressure_average = (
        wall_density[source] * pressure[target] + density[target] * wall_pressure[source]
    ) / (density[target] + wall_density[source])
    pair_force = (
        -(fluid_pair_volume**2 + wall_pair_volume**2)[:, None]
        * pressure_average[:, None]
        * gradient
    )
    if plan.kinematic_viscosity > 0.0:
        velocity_difference = fluid_v[target] - ghost_velocity[source]
        radial = jnp.sum(displacement * gradient, axis=-1)
        denominator_viscous = distance**2 + 0.01 * smoothing_length**2
        viscous_scalar = (
            plan.kinematic_viscosity
            * (density[target] + wall_density[source])
            * (fluid_pair_volume**2 + wall_pair_volume**2)
            * radial
            / denominator_viscous
        )
        pair_force = pair_force + viscous_scalar[:, None] * velocity_difference
    pair_force = jnp.where(valid[:, None], pair_force, 0.0)
    fluid_force = jnp.zeros_like(fluid_x).at[target].add(pair_force)
    wall_reaction = jnp.zeros_like(wall_x).at[source].add(-pair_force)
    ledger = ParticleInteractionLedger.from_forces(
        fluid_force,
        wall_reaction,
        fluid_v,
        wall_v,
        jnp.sum(valid),
    )
    return WallInteractionResult(
        fluid_force,
        wall_reaction,
        wall_pressure,
        wall_density,
        ghost_velocity,
        denominator,
        ledger,
        jnp.all(denominator > 0.0),
    )


__all__ = [
    "AdamiWallBoundaryPlan",
    "PrescribedWallMotion",
    "PreparedWallParticles",
    "WallInteractionResult",
    "WallParticleGenerationPlan",
    "WallParticleQualityReport",
    "WallSlipPolicy",
    "evaluate_wall_interaction",
]
