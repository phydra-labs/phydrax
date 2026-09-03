#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.spatial import PreparedSparseVoxelGrid, SparseVoxelField
from ._certificate import (
    DistanceSemantics,
    ExactSDFEnclosureCertificate,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from ._contracts import CompiledGeometry


class VoxelGeometrySamplingEvidence(NonTrainableState, StrictModule):
    """Numerical and enclosure evidence for sampled voxel geometry."""

    active_samples: jax.Array
    finite_samples: jax.Array
    certified_sign_samples: jax.Array
    unresolved_samples: jax.Array
    maximum_enclosure_width: jax.Array
    finite: jax.Array
    successful: jax.Array


class PreparedVoxelGeometrySamples(StrictModule):
    """Sampled boundary field with explicit sign-enclosure authority."""

    field: SparseVoxelField
    lower_bounds: jax.Array
    upper_bounds: jax.Array
    sign_certified: jax.Array
    narrow_band: jax.Array
    evidence: VoxelGeometrySamplingEvidence
    certificate: FieldCertificate = eqx.field(static=True)
    sampling_id: str = eqx.field(static=True)


class VoxelGeometrySamplingPlan(StrictModule, NonTrainableState):
    """Sample a compiled boundary field on an existing sparse voxel topology."""

    grid: PreparedSparseVoxelGrid
    enclosure: ExactSDFEnclosureCertificate | None = eqx.field(static=True)
    narrow_band_width: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedSparseVoxelGrid,
        /,
        *,
        enclosure: ExactSDFEnclosureCertificate | None = None,
        narrow_band_width: float | None = None,
    ) -> None:
        if not isinstance(grid, PreparedSparseVoxelGrid):
            raise TypeError("grid must be PreparedSparseVoxelGrid.")
        width = None if narrow_band_width is None else float(narrow_band_width)
        if width is not None and (not np.isfinite(width) or width < 0.0):
            raise ValueError("narrow_band_width must be finite and nonnegative.")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "enclosure", enclosure)
        object.__setattr__(self, "narrow_band_width", width)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "voxel-geometry-sampling-plan",
                    "grid_id": grid.grid_id,
                    "enclosure": None
                    if enclosure is None
                    else {
                        "evaluation_error": enclosure.evaluation_error,
                        "lipschitz_upper_bound": enclosure.lipschitz_upper_bound,
                    },
                    "narrow_band_width": width,
                }
            ),
        )

    def sample(self, geometry: CompiledGeometry, /) -> PreparedVoxelGeometrySamples:
        if not isinstance(geometry, CompiledGeometry):
            raise TypeError("geometry must be CompiledGeometry.")
        if geometry.ambient_dimension != self.grid.dimension:
            raise ValueError("Geometry and voxel dimensions disagree.")
        centers = self.grid.voxel_centers()
        flat_centers = centers.reshape((-1, self.grid.dimension))
        sampled = geometry.boundary_field(flat_centers).reshape(
            (self.grid.brick_capacity, self.grid.voxels_per_brick)
        )
        active = self.grid.voxel_active & self.grid.brick_active[:, None]
        sampled = jnp.where(active, sampled, 0.0)
        finite_samples = active & jnp.isfinite(sampled)
        finite = jnp.all(~active | finite_samples)
        if self.enclosure is None:
            lower = jnp.where(active, -jnp.inf, 0.0)
            upper = jnp.where(active, jnp.inf, 0.0)
            sign_certified = jnp.zeros_like(active)
            enclosure_width = jnp.asarray(jnp.inf, dtype=sampled.dtype)
        else:
            cell_width = (
                jnp.asarray(self.grid.address_plan.upper, dtype=sampled.dtype)
                - jnp.asarray(self.grid.address_plan.lower, dtype=sampled.dtype)
            ) / self.grid.address_plan.resolution
            radius = 0.5 * jnp.sqrt(jnp.sum(cell_width**2))
            error = jnp.asarray(
                self.enclosure.evaluation_error
                + self.enclosure.lipschitz_upper_bound * radius,
                dtype=sampled.dtype,
            )
            lower = jnp.where(active, sampled - error, 0.0)
            upper = jnp.where(active, sampled + error, 0.0)
            sign_certified = active & ((upper < 0.0) | (lower > 0.0))
            enclosure_width = 2.0 * error
        if self.narrow_band_width is None:
            narrow_band = active
        elif self.enclosure is None:
            narrow_band = active & (jnp.abs(sampled) <= self.narrow_band_width)
        else:
            narrow_band = active & (
                (lower <= self.narrow_band_width) & (upper >= -self.narrow_band_width)
            )
        field = SparseVoxelField(self.grid, sampled)
        certificate = FieldCertificate(
            zero_set_accuracy=ZeroSetAccuracy.APPROXIMATE,
            sign_reliability=(
                SignReliability.LOCAL
                if self.enclosure is not None
                else SignReliability.UNRELIABLE
            ),
            distance_semantics=(
                DistanceSemantics.APPROXIMATE
                if geometry.field_certificate.is_signed_distance
                else DistanceSemantics.LEVEL_SET
            ),
            regularity=FieldRegularity.PIECEWISE_SMOOTH,
            safe_step_factor=None,
            validity_region="sparse_voxel_support",
            parameter_differentiable=geometry.field_certificate.parameter_differentiable,
            provenance=(
                *geometry.field_certificate.provenance,
                "sparse_voxel_sampling",
            ),
        )
        active_samples = jnp.sum(active, dtype=jnp.int32)
        certified_samples = jnp.sum(sign_certified, dtype=jnp.int32)
        evidence = VoxelGeometrySamplingEvidence(
            active_samples=active_samples,
            finite_samples=jnp.sum(finite_samples, dtype=jnp.int32),
            certified_sign_samples=certified_samples,
            unresolved_samples=active_samples - certified_samples,
            maximum_enclosure_width=enclosure_width,
            finite=finite,
            successful=finite & self.grid.evidence.successful,
        )
        return PreparedVoxelGeometrySamples(
            field=field,
            lower_bounds=lower,
            upper_bounds=upper,
            sign_certified=sign_certified,
            narrow_band=narrow_band,
            evidence=evidence,
            certificate=certificate,
            sampling_id=canonical_fingerprint(
                {
                    "kind": "prepared-voxel-geometry-samples",
                    "plan": self.plan_id,
                    "geometry_kernel": type(geometry.kernel).__name__,
                }
            ),
        )


__all__ = [
    "PreparedVoxelGeometrySamples",
    "VoxelGeometrySamplingEvidence",
    "VoxelGeometrySamplingPlan",
]
