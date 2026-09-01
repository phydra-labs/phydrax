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
from ._interfaces import VortexDiffusionDiagnostics, VortexDiffusionEvaluation
from ._source import VortexSourceState


class WallCorrectedPSEEvidence(StrictModule):
    wall_pair_count: Array
    mirror_pair_count: Array
    normalization_minimum: Array
    total_rate: Array
    wall_flux: Array
    conservative_with_flux: Array
    policy: str = eqx.field(static=True)


class WallCorrectedPSEPlan(StrictModule, NonTrainableState):
    smoothing_scale: float = eqx.field(static=True)
    cutoff_factor: float = eqx.field(static=True)
    policy: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        smoothing_scale: float,
        /,
        *,
        cutoff_factor: float = 4.0,
        policy: str = "mirror",
    ):
        if (
            float(smoothing_scale) <= 0.0
            or float(cutoff_factor) <= 0.0
            or policy not in ("mirror", "one-sided")
        ):
            raise ValueError("Wall-corrected PSE controls are invalid.")
        self.smoothing_scale, self.cutoff_factor, self.policy = (
            float(smoothing_scale),
            float(cutoff_factor),
            policy,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wall-corrected-pse",
                "smoothing_scale": self.smoothing_scale,
                "cutoff_factor": self.cutoff_factor,
                "policy": policy,
            }
        )

    def evaluate(
        self,
        source: VortexSourceState,
        viscosity: ArrayLike,
        wall_point: ArrayLike,
        wall_normal: ArrayLike,
        /,
        *,
        prescribed_wall_flux: ArrayLike | None = None,
    ) -> tuple[VortexDiffusionEvaluation, WallCorrectedPSEEvidence]:
        if source.volume is None:
            raise ValueError("Wall-corrected PSE requires source volume.")
        point, normal = (
            jnp.asarray(wall_point, dtype=source.positions.dtype),
            jnp.asarray(wall_normal, dtype=source.positions.dtype),
        )
        if (
            point.shape != source.positions.shape
            or normal.shape != source.positions.shape
        ):
            raise ValueError("Wall point/normal must match source positions.")
        normal_norm = jnp.linalg.norm(normal, axis=-1)
        normal = normal / jnp.maximum(normal_norm, jnp.finfo(normal.dtype).tiny)[:, None]
        distance = jnp.sum((source.safe_positions() - point) * normal, axis=-1)
        mirror_position = source.safe_positions() - 2.0 * distance[:, None] * normal
        strength, volume = source.safe_strength(), source.safe_volume()
        epsilon = jnp.asarray(self.smoothing_scale, dtype=source.positions.dtype)
        viscosity_ = jnp.asarray(viscosity, dtype=source.positions.dtype)
        displacement = (
            source.safe_positions()[:, None, :] - source.safe_positions()[None, :, :]
        )
        mirror_displacement = (
            source.safe_positions()[:, None, :] - mirror_position[None, :, :]
        )
        squared = jnp.sum(displacement**2, axis=-1)
        mirror_squared = jnp.sum(mirror_displacement**2, axis=-1)
        active_pair = (
            source.active_mask[:, None]
            & source.active_mask[None, :]
            & ~jnp.eye(source.capacity, dtype=bool)
            & (squared < (self.cutoff_factor * epsilon) ** 2)
        )
        mirror_pair = (
            source.active_mask[:, None]
            & source.active_mask[None, :]
            & (mirror_squared < (self.cutoff_factor * epsilon) ** 2)
        )
        normalization = jnp.pi ** (-0.5 * source.dimension) * epsilon ** (
            -source.dimension
        )
        kernel = normalization * jnp.exp(-squared / epsilon**2)
        mirror_kernel = normalization * jnp.exp(-mirror_squared / epsilon**2)
        omega = strength / (volume if source.dimension == 2 else volume[:, None])
        difference = omega[None, ...] - omega[:, None, ...]
        pair_factor = 4.0 * viscosity_ * volume[:, None] * volume[None, :] / epsilon**2
        pair_flux = (
            pair_factor * kernel * difference
            if source.dimension == 2
            else pair_factor[..., None] * kernel[..., None] * difference
        )
        pair_flux = jnp.where(
            active_pair if source.dimension == 2 else active_pair[..., None],
            pair_flux,
            0.0,
        )
        bulk_rate = jnp.sum(pair_flux, axis=1)
        if self.policy == "mirror":
            mirrored_omega = -omega
            mirror_difference = mirrored_omega[None, ...] - omega[:, None, ...]
            mirror_flux = (
                pair_factor * mirror_kernel * mirror_difference
                if source.dimension == 2
                else pair_factor[..., None] * mirror_kernel[..., None] * mirror_difference
            )
            mirror_flux = jnp.where(
                mirror_pair if source.dimension == 2 else mirror_pair[..., None],
                mirror_flux,
                0.0,
            )
            wall_rate = jnp.sum(mirror_flux, axis=1)
            correction_normalization = jnp.ones((source.capacity,), dtype=epsilon.dtype)
        else:
            support = jnp.sum(
                jnp.where(active_pair, kernel * volume[None, :], 0.0), axis=1
            )
            full_support = jnp.maximum(
                jnp.sum(kernel * volume[None, :], axis=1), jnp.finfo(epsilon.dtype).tiny
            )
            correction_normalization = support / full_support
            wall_rate = bulk_rate * (
                1.0 / jnp.maximum(correction_normalization, 0.25) - 1.0
            ).reshape((-1,) + (1,) * (bulk_rate.ndim - 1))
        prescribed = (
            jnp.zeros_like(bulk_rate)
            if prescribed_wall_flux is None
            else jnp.asarray(prescribed_wall_flux, dtype=bulk_rate.dtype)
        )
        if prescribed.shape != bulk_rate.shape:
            raise ValueError("prescribed_wall_flux must match source strength shape.")
        rate = bulk_rate + wall_rate + prescribed
        total = jnp.sum(rate, axis=0)
        wall_flux = jnp.sum(wall_rate + prescribed, axis=0)
        residual = total - wall_flux
        conservative = jnp.max(jnp.abs(residual)) <= 512 * jnp.finfo(
            rate.dtype
        ).eps * jnp.maximum(jnp.max(jnp.abs(total)), 1.0)
        finite = (
            jnp.all(jnp.isfinite(rate))
            & jnp.all(jnp.isfinite(normal))
            & jnp.isfinite(viscosity_)
            & (viscosity_ >= 0.0)
        )
        successful = finite & conservative
        evidence = WallCorrectedPSEEvidence(
            jnp.sum(active_pair, dtype=jnp.int32),
            jnp.sum(mirror_pair, dtype=jnp.int32),
            jnp.min(correction_normalization),
            total,
            wall_flux,
            conservative,
            self.policy,
        )
        diagnostics = VortexDiffusionDiagnostics(
            jnp.asarray(source.capacity, dtype=jnp.int32),
            jnp.sum(active_pair, dtype=jnp.int32),
            total,
            finite,
            finite,
            jnp.asarray(True),
            conservative,
            successful,
            evidence,
        )
        evaluation = VortexDiffusionEvaluation(
            rate,
            successful,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "wall-corrected-pse-evaluation", "plan": self.plan_id}
            ),
            diagnostics,
        )
        return evaluation, evidence


__all__ = ["WallCorrectedPSEEvidence", "WallCorrectedPSEPlan"]
