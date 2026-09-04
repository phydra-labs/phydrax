#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import SmallLinearSolvePlan
from ..wave._fields import PlaneFieldSpace, ScalarPlaneField
from ._core import (
    beamlet_curvature,
    BeamletFrame,
    BeamletStatus,
    GaussianBeamletState,
)


class BeamletReconstructionPlan(StrictModule, NonTrainableState):
    """Fixed-support, fixed-tile reconstruction plan for Gaussian beamlets."""

    space: PlaneFieldSpace
    longitudinal_coordinate: Array
    frame_id: str = eqx.field(static=True)
    tile_size: int = eqx.field(static=True)
    solve_plan: SmallLinearSolvePlan
    caustic_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        space: PlaneFieldSpace,
        longitudinal_coordinate: ArrayLike,
        /,
        *,
        tile_size: int = 4096,
        singular_tolerance: float = 1e-12,
        maximum_condition: float = 1e12,
        caustic_tolerance: float = 1e-10,
    ):
        if not isinstance(space, PlaneFieldSpace):
            raise TypeError("space must be a PlaneFieldSpace.")
        coordinate = jnp.asarray(longitudinal_coordinate)
        if coordinate.shape != () or not jnp.issubdtype(coordinate.dtype, jnp.floating):
            raise ValueError("longitudinal_coordinate must be a real scalar.")
        size = int(tile_size)
        if size <= 0:
            raise ValueError("tile_size must be positive.")
        if singular_tolerance <= 0.0 or maximum_condition <= 1.0:
            raise ValueError("Beamlet solve tolerances are invalid.")
        if caustic_tolerance <= 0.0:
            raise ValueError("caustic_tolerance must be positive.")
        self.space = space
        self.longitudinal_coordinate = coordinate
        self.frame_id = BeamletFrame(space.frame).frame_id
        self.tile_size = size
        self.solve_plan = SmallLinearSolvePlan(
            2,
            singular_tolerance=singular_tolerance,
            maximum_condition=maximum_condition,
            refinement_iterations=1,
        )
        self.caustic_tolerance = float(caustic_tolerance)

    def prepare(self, /) -> "PreparedBeamletReconstruction":
        points = self.space.world_points.reshape((-1, 3))
        point_count = int(points.shape[0])
        tile_count = (point_count + self.tile_size - 1) // self.tile_size
        padded_count = tile_count * self.tile_size
        padding = padded_count - point_count
        padded = jnp.pad(points, ((0, padding), (0, 0)))
        active = jnp.arange(padded_count) < point_count
        return PreparedBeamletReconstruction(
            self,
            padded.reshape((tile_count, self.tile_size, 3)),
            active.reshape((tile_count, self.tile_size)),
            point_count=point_count,
            tile_count=tile_count,
        )


class PreparedBeamletReconstruction(StrictModule, NonTrainableState):
    plan: BeamletReconstructionPlan
    world_point_tiles: Array
    active_tiles: Array
    point_count: int = eqx.field(static=True)
    tile_count: int = eqx.field(static=True)

    def __init__(
        self,
        plan: BeamletReconstructionPlan,
        world_point_tiles: ArrayLike,
        active_tiles: ArrayLike,
        /,
        *,
        point_count: int,
        tile_count: int,
    ):
        if not isinstance(plan, BeamletReconstructionPlan):
            raise TypeError("plan must be a BeamletReconstructionPlan.")
        points = jnp.asarray(world_point_tiles)
        active = jnp.asarray(active_tiles, dtype=bool)
        expected_points = (int(tile_count), plan.tile_size, 3)
        expected_active = expected_points[:-1]
        if points.shape != expected_points or active.shape != expected_active:
            raise ValueError("Prepared beamlet tiles do not match the plan.")
        if int(point_count) != prod(plan.space.shape):
            raise ValueError("point_count does not match the target field space.")
        self.plan = plan
        self.world_point_tiles = points
        self.active_tiles = active
        self.point_count = int(point_count)
        self.tile_count = int(tile_count)

    def execute(
        self,
        state: GaussianBeamletState,
        /,
    ) -> "BeamletReconstructionResult":
        return reconstruct_gaussian_beamlets(self, state)


class BeamletReconstructionEvidence(StrictModule, NonTrainableState):
    contributing_beamlets: Array
    total_beamlets: Array
    minimum_caustic_distance: Array
    maximum_condition_estimate: Array
    finite: Array
    frame_consistent: Array
    valid: Array
    status: Array
    tile_count: int = eqx.field(static=True)
    tile_size: int = eqx.field(static=True)


class BeamletReconstructionResult(StrictModule):
    field: ScalarPlaneField
    evidence: BeamletReconstructionEvidence

    @property
    def successful(self) -> Array:
        return self.evidence.valid


def reconstruct_gaussian_beamlets(
    prepared: PreparedBeamletReconstruction,
    state: GaussianBeamletState,
    /,
) -> BeamletReconstructionResult:
    """Coherently reconstruct a scalar field without a point-by-beamlet array."""
    if not isinstance(prepared, PreparedBeamletReconstruction) or not isinstance(
        state, GaussianBeamletState
    ):
        raise TypeError("reconstruction requires a prepared plan and beamlet state.")
    curvature = beamlet_curvature(
        state,
        solve_plan=prepared.plan.solve_plan,
    )
    leading_size = prod(state.beamlet_shape) if state.beamlet_shape else 1
    origins = jnp.asarray(state.chief_ray.origins).reshape((leading_size, 3))
    directions = jnp.asarray(state.chief_ray.directions).reshape((leading_size, 3))
    basis = state.transverse_basis.reshape((leading_size, 3, 2))
    curvature_matrix = curvature.curvature.reshape((leading_size, 2, 2))
    determinant = curvature.determinant.reshape((leading_size,))
    determinant_phase = state.determinant_phase.reshape((leading_size,))
    initial_determinant = state.initial_determinant.reshape((leading_size,))
    wavenumbers = state.medium_wavenumbers.reshape((leading_size,))
    amplitudes = state.amplitudes.reshape((leading_size,))
    refractive_indices = jnp.asarray(state.chief_ray.refractive_indices).reshape(
        (leading_size,)
    )
    vacuum_wavenumbers = wavenumbers / refractive_indices
    optical_path = jnp.asarray(state.chief_ray.optical_path_lengths).reshape(
        (leading_size,)
    )
    frame_consistent = jnp.asarray(state.frame_id == prepared.plan.frame_id)
    beamlet_valid = (
        state.valid.reshape((leading_size,))
        & curvature.successful.reshape((leading_size,))
        & frame_consistent
    )
    h_scale = jnp.sum(jnp.abs(state.h) ** 2, axis=(-2, -1)).reshape((leading_size,))
    caustic_distance = jnp.abs(determinant) / jnp.maximum(
        h_scale, jnp.finfo(determinant.real.dtype).tiny
    )
    beamlet_valid = beamlet_valid & (caustic_distance > prepared.plan.caustic_tolerance)
    initial_square_root = jnp.sqrt(jnp.abs(initial_determinant)) * jnp.exp(
        0.5j * jnp.angle(initial_determinant)
    )
    determinant_square_root = jnp.sqrt(jnp.abs(determinant)) * jnp.exp(
        0.5j * determinant_phase
    )
    safe_square_root = jnp.where(
        jnp.abs(determinant_square_root) > 0.0, determinant_square_root, 1.0
    )
    normalized_amplitude = amplitudes * initial_square_root / safe_square_root

    def evaluate_tile(_: None, inputs: tuple[Array, Array]):
        points, active = inputs
        displacement = points[None, :, :] - origins[:, None, :]
        transverse = contract("bpc,bca->bpa", displacement, basis)
        longitudinal = contract("bpc,bc->bp", displacement, directions)
        quadratic = contract(
            "bpi,bij,bpj->bp",
            transverse,
            curvature_matrix,
            transverse,
        )
        phase = vacuum_wavenumbers[:, None] * optical_path[:, None] + wavenumbers[
            :, None
        ] * (longitudinal + 0.5 * quadratic)
        contributions = normalized_amplitude[:, None] * jnp.exp(1j * phase)
        contributions = jnp.where(beamlet_valid[:, None], contributions, 0.0)
        values = jnp.sum(contributions, axis=0)
        return None, jnp.where(active, values, 0.0)

    _, tiles = jax.lax.scan(
        evaluate_tile,
        None,
        (prepared.world_point_tiles, prepared.active_tiles),
    )
    values = tiles.reshape((-1,))[: prepared.point_count].reshape(
        prepared.plan.space.shape
    )
    contributing = jnp.sum(beamlet_valid.astype(jnp.int32))
    total = jnp.asarray(leading_size, dtype=jnp.int32)
    finite = jnp.all(jnp.isfinite(values))
    all_valid = contributing == total
    any_valid = contributing > 0
    valid = finite & any_valid & all_valid
    status = jnp.where(
        ~finite,
        int(BeamletStatus.NONFINITE_INPUT),
        jnp.where(
            ~any_valid,
            int(BeamletStatus.NO_VALID_BEAMLETS),
            jnp.where(
                ~all_valid,
                int(BeamletStatus.PARTIAL_RECONSTRUCTION),
                int(BeamletStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    status = jnp.where(
        ~frame_consistent,
        int(BeamletStatus.FRAME_MISMATCH),
        status,
    )
    status = jnp.where(
        ~finite,
        int(BeamletStatus.NONFINITE_INPUT),
        status,
    )
    minimum_caustic = jnp.min(caustic_distance)
    maximum_condition = jnp.max(curvature.condition_estimate)
    field = ScalarPlaneField(
        prepared.plan.space,
        values,
        state.angular_frequency,
        prepared.plan.longitudinal_coordinate,
    )
    evidence = BeamletReconstructionEvidence(
        contributing,
        total,
        minimum_caustic,
        maximum_condition,
        finite,
        frame_consistent,
        valid,
        status,
        tile_count=prepared.tile_count,
        tile_size=prepared.plan.tile_size,
    )
    return BeamletReconstructionResult(field, evidence)


__all__ = [
    "BeamletReconstructionEvidence",
    "BeamletReconstructionPlan",
    "BeamletReconstructionResult",
    "PreparedBeamletReconstruction",
    "reconstruct_gaussian_beamlets",
]
