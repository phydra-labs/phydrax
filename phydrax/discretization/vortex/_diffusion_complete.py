#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    solve as solve_linear,
)
from ..particle import ParticleBox, ParticlePairRelation
from ._capabilities import VortexDiffusionCapabilities
from ._interfaces import VortexDiffusionDiagnostics, VortexDiffusionEvaluation
from ._source import VortexSourceState


class CoreSpreadingEvidence(StrictModule):
    minimum_core: Array
    maximum_core: Array
    maximum_overlap_ratio: Array
    stable_step: Array
    compatible: Array


class GaussianCoreSpreadingPlan(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    overlap_limit: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    capabilities: VortexDiffusionCapabilities

    def __init__(self, dimension: int, /, *, overlap_limit: float = 2.0):
        dimension_ = int(dimension)
        overlap = float(overlap_limit)
        if dimension_ not in (2, 3) or not math.isfinite(overlap) or overlap <= 0.0:
            raise ValueError("Core-spreading dimension/overlap limit is invalid.")
        self.dimension = dimension_
        self.overlap_limit = overlap
        self.capabilities = VortexDiffusionCapabilities(
            dimension_,
            required_source_fields=(
                "positions",
                "strength",
                "active_mask",
                "core_radius",
            ),
            derivatives=("source-core-radius",),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gaussian-core-spreading",
                "dimension": dimension_,
                "overlap_limit": overlap,
            }
        )

    def rate(
        self, source: VortexSourceState, viscosity: ArrayLike, /
    ) -> tuple[Array, CoreSpreadingEvidence]:
        if source.dimension != self.dimension or source.core_radius is None:
            raise ValueError("Core spreading requires matching source core radii.")
        viscosity_ = jnp.asarray(viscosity, dtype=source.positions.dtype)
        if viscosity_.shape != ():
            raise ValueError("Core-spreading viscosity must be scalar.")
        core = source.safe_core_radius()
        core_rate = 2.0 * viscosity_ / core
        displacement = (
            source.safe_positions()[:, None, :] - source.safe_positions()[None, :, :]
        )
        distance = jnp.linalg.norm(displacement, axis=-1)
        valid_pair = (
            source.active_mask[:, None]
            & source.active_mask[None, :]
            & ~jnp.eye(source.capacity, dtype=bool)
        )
        minimum_distance = jnp.min(jnp.where(valid_pair, distance, jnp.inf), axis=1)
        overlap = core / jnp.maximum(minimum_distance, jnp.finfo(core.dtype).tiny)
        compatible = (
            jnp.all(jnp.where(source.active_mask, overlap <= self.overlap_limit, True))
            & jnp.isfinite(viscosity_)
            & (viscosity_ >= 0.0)
        )
        stable_step = jnp.where(
            viscosity_ > 0.0, 0.125 * jnp.min(core) ** 2 / viscosity_, jnp.inf
        )
        evidence = CoreSpreadingEvidence(
            jnp.min(core),
            jnp.max(core),
            jnp.max(jnp.where(source.active_mask, overlap, 0.0)),
            stable_step,
            compatible,
        )
        return jnp.where(source.active_mask, core_rate, 0.0), evidence


class NeighborhoodPSEEvidence(StrictModule):
    pair_count: Array
    total_rate_defect: Array
    support_radius: Array
    neighborhood_id: str = eqx.field(static=True)


class GaussianPSENeighborhoodPlan(StrictModule, NonTrainableState):
    relation: ParticlePairRelation
    smoothing_scale: float = eqx.field(static=True)
    cutoff_factor: float = eqx.field(static=True)
    box: ParticleBox | None
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    capabilities: VortexDiffusionCapabilities

    def __init__(
        self,
        relation: ParticlePairRelation,
        dimension: int,
        smoothing_scale: float,
        /,
        *,
        cutoff_factor: float = 4.0,
        box: ParticleBox | None = None,
    ):
        if not isinstance(relation, ParticlePairRelation):
            raise TypeError("relation must be ParticlePairRelation.")
        dimension_, epsilon, cutoff = (
            int(dimension),
            float(smoothing_scale),
            float(cutoff_factor),
        )
        if dimension_ not in (2, 3) or epsilon <= 0.0 or cutoff <= 0.0:
            raise ValueError("Neighborhood PSE controls are invalid.")
        if box is not None and (
            not isinstance(box, ParticleBox) or box.ambient_dimension != dimension_
        ):
            raise ValueError("Neighborhood PSE box is incompatible.")
        self.relation = relation
        self.smoothing_scale = epsilon
        self.cutoff_factor = cutoff
        self.box = box
        self.dimension = dimension_
        self.capabilities = VortexDiffusionCapabilities(
            dimension_,
            required_source_fields=("positions", "strength", "active_mask", "volume"),
            domain="periodic"
            if box is not None and bool(jnp.any(box.periodic))
            else "free-space",
            derivatives=("source-position", "source-strength", "source-volume"),
            acceleration="direct",
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gaussian-pse-neighborhood",
                "relation": relation.relation_schema_id,
                "dimension": dimension_,
                "smoothing_scale": epsilon,
                "cutoff_factor": cutoff,
                "box": None if box is None else box.box_id,
            }
        )

    def evaluate(
        self, source: VortexSourceState, viscosity: ArrayLike, /
    ) -> tuple[VortexDiffusionEvaluation, NeighborhoodPSEEvidence]:
        if source.dimension != self.dimension or source.volume is None:
            raise ValueError("Neighborhood PSE requires matching source volumes.")
        left = self.relation.relation.source_indices
        right = self.relation.relation.target_indices
        if jnp.max(jnp.concatenate((left, right)), initial=0) >= source.capacity:
            raise ValueError("Neighborhood relation exceeds source capacity.")
        position, strength, volume = (
            source.safe_positions(),
            source.safe_strength(),
            source.safe_volume(),
        )
        viscosity_ = jnp.asarray(viscosity, dtype=position.dtype)
        displacement = position[left] - position[right]
        if self.box is not None:
            displacement = self.box.minimum_image(displacement)
        squared = jnp.sum(displacement * displacement, axis=-1)
        epsilon = jnp.asarray(self.smoothing_scale, dtype=position.dtype)
        scaled = squared / epsilon**2
        active = (
            source.active_mask[left]
            & source.active_mask[right]
            & self.relation.relation.valid
            & (scaled < self.cutoff_factor**2)
        )
        kernel = jnp.exp(-scaled) / (
            (jnp.pi ** (0.5 * self.dimension)) * epsilon**self.dimension
        )
        omega_left = strength[left] / (
            volume[left] if self.dimension == 2 else volume[left, None]
        )
        omega_right = strength[right] / (
            volume[right] if self.dimension == 2 else volume[right, None]
        )
        factor = 4.0 * viscosity_ * volume[left] * volume[right] * kernel / epsilon**2
        flux = (
            factor * (omega_right - omega_left)
            if self.dimension == 2
            else factor[:, None] * (omega_right - omega_left)
        )
        flux = jnp.where(active if self.dimension == 2 else active[:, None], flux, 0.0)
        rate = jnp.zeros_like(strength).at[left].add(flux).at[right].add(-flux)
        total = jnp.sum(rate, axis=0)
        scale = jnp.maximum(jnp.max(jnp.sum(jnp.abs(rate), axis=0)), 1.0)
        defect = jnp.max(jnp.abs(total)) / scale
        successful = (
            jnp.isfinite(viscosity_)
            & (viscosity_ >= 0.0)
            & jnp.all(jnp.isfinite(rate))
            & (defect <= 256 * jnp.finfo(rate.dtype).eps)
        )
        diagnostics = VortexDiffusionDiagnostics(
            jnp.asarray(source.capacity, dtype=jnp.int32),
            jnp.sum(active, dtype=jnp.int32),
            total,
            jnp.asarray(True),
            jnp.all(jnp.isfinite(rate)),
            jnp.asarray(True),
            successful,
            successful,
            None,
        )
        evaluation = VortexDiffusionEvaluation(
            rate,
            successful,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "neighborhood-pse-evaluation", "plan": self.plan_id}
            ),
            diagnostics,
        )
        evidence = NeighborhoodPSEEvidence(
            jnp.sum(active, dtype=jnp.int32),
            defect,
            self.cutoff_factor * epsilon,
            self.relation.relation_schema_id,
        )
        return evaluation, evidence


class RBFReinitializationResult(StrictModule):
    strength: Array
    residual_norm: Array
    condition_estimate: Array
    successful: Array
    reinitialization_id: str = eqx.field(static=True)
    linear_result: object


class GaussianRBFReinitializationPlan(StrictModule, NonTrainableState):
    regularization: float = eqx.field(static=True)
    policy: LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        regularization: float = 1.0e-10,
        policy: LinearSolvePolicy | None = None,
    ):
        regularization_ = float(regularization)
        if not math.isfinite(regularization_) or regularization_ < 0.0:
            raise ValueError("RBF regularization must be finite and nonnegative.")
        self.regularization = regularization_
        self.policy = LinearSolvePolicy(DenseSVD()) if policy is None else policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gaussian-rbf-vortex-reinitialization",
                "regularization": regularization_,
            }
        )

    def apply(
        self,
        source: VortexSourceState,
        new_positions: ArrayLike,
        new_core_radius: ArrayLike,
        /,
    ) -> RBFReinitializationResult:
        if source.core_radius is None:
            raise ValueError("RBF reinitialization requires source core radii.")
        target = jnp.asarray(new_positions, dtype=source.positions.dtype)
        target_core = jnp.asarray(new_core_radius, dtype=source.positions.dtype)
        if target.shape != source.positions.shape or target_core.shape != (
            source.capacity,
        ):
            raise ValueError(
                "RBF target positions/core shape must match source capacity."
            )
        old_delta = (
            source.safe_positions()[:, None, :] - source.safe_positions()[None, :, :]
        )
        new_delta = source.safe_positions()[:, None, :] - target[None, :, :]
        old_scale = source.safe_core_radius()[None, :]
        new_scale = target_core[None, :]
        old_matrix = jnp.exp(-jnp.sum(old_delta * old_delta, axis=-1) / old_scale**2)
        matrix = jnp.exp(-jnp.sum(new_delta * new_delta, axis=-1) / new_scale**2)
        matrix = matrix + self.regularization * jnp.eye(
            source.capacity, dtype=matrix.dtype
        )
        rhs = old_matrix @ source.safe_strength()
        linear = solve_linear(
            LeastSquaresProblem(
                DenseLinearOperator(matrix), problem_id=f"{self.plan_id}:rbf"
            ),
            rhs,
            policy=self.policy,
        )
        strength = jnp.asarray(linear.value)
        residual = jnp.linalg.norm(matrix @ strength - rhs)
        condition = jnp.linalg.cond(matrix)
        successful = (
            linear.successful & jnp.all(jnp.isfinite(strength)) & jnp.isfinite(condition)
        )
        return RBFReinitializationResult(
            strength, residual, condition, successful, self.plan_id, linear
        )


class VortexRedistributionResult(StrictModule):
    position: Array
    strength: Array
    volume: Array
    circulation_residual: Array
    first_moment_residual: Array
    second_moment_residual: Array
    successful: Array
    redistribution_id: str = eqx.field(static=True)
    linear_result: object


class VortexRedistributionPlan(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    policy: LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, policy: LinearSolvePolicy | None = None):
        dimension_ = int(dimension)
        if dimension_ not in (2, 3):
            raise ValueError("VRM dimension must be 2 or 3.")
        self.dimension = dimension_
        self.policy = LinearSolvePolicy(DenseSVD()) if policy is None else policy
        self.plan_id = canonical_fingerprint(
            {"kind": "vortex-redistribution", "dimension": dimension_}
        )

    def apply(
        self, source: VortexSourceState, candidate_positions: ArrayLike, /
    ) -> VortexRedistributionResult:
        candidate = jnp.asarray(candidate_positions, dtype=source.positions.dtype)
        if (
            source.dimension != self.dimension
            or candidate.ndim != 2
            or candidate.shape[1] != self.dimension
        ):
            raise ValueError("VRM candidate positions are incompatible.")
        old_position, old_strength = source.safe_positions(), source.safe_strength()
        rows = [jnp.ones((candidate.shape[0],), dtype=candidate.dtype)]
        rows.extend(candidate[:, axis] for axis in range(self.dimension))
        rows.extend(candidate[:, axis] ** 2 for axis in range(self.dimension))
        matrix = jnp.stack(tuple(rows), axis=0)
        old_rows = [jnp.ones((source.capacity,), dtype=candidate.dtype)]
        old_rows.extend(old_position[:, axis] for axis in range(self.dimension))
        old_rows.extend(old_position[:, axis] ** 2 for axis in range(self.dimension))
        old_matrix = jnp.stack(tuple(old_rows), axis=0)
        rhs = old_matrix @ old_strength
        linear = solve_linear(
            LeastSquaresProblem(
                DenseLinearOperator(matrix), problem_id=f"{self.plan_id}:moments"
            ),
            rhs,
            policy=self.policy,
        )
        new_strength = jnp.asarray(linear.value)
        residual = matrix @ new_strength - rhs
        strength_residual = residual[0]
        first_residual = residual[1 : 1 + self.dimension]
        second_residual = residual[1 + self.dimension :]
        volume_total = (
            jnp.sum(source.safe_volume())
            if source.volume is not None
            else jnp.asarray(float(source.capacity), dtype=candidate.dtype)
        )
        new_volume = jnp.full(
            (candidate.shape[0],),
            volume_total / candidate.shape[0],
            dtype=candidate.dtype,
        )
        successful = linear.successful & jnp.all(jnp.isfinite(new_strength))
        return VortexRedistributionResult(
            candidate,
            new_strength,
            new_volume,
            strength_residual,
            first_residual,
            second_residual,
            successful,
            self.plan_id,
            linear,
        )


__all__ = [
    "CoreSpreadingEvidence",
    "GaussianCoreSpreadingPlan",
    "GaussianPSENeighborhoodPlan",
    "GaussianRBFReinitializationPlan",
    "NeighborhoodPSEEvidence",
    "RBFReinitializationResult",
    "VortexRedistributionPlan",
    "VortexRedistributionResult",
]
