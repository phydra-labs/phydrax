# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from ...._strict import StrictModule
from .._binding import PreparedNucleotideBinding


class BaseFrameEvaluation(StrictModule):
    centers: Array
    axes: Array
    valid: Array
    covered: Array
    geometry_margin: Array


def base_frames(
    positions,
    binding: PreparedNucleotideBinding,
    *,
    image_policy: str,
    tolerance: float = 1e-10,
) -> BaseFrameEvaluation:
    """C2-centered x direction and purine/pyrimidine-oriented right-handed axes.

    Axes have shape (nucleotide, Cartesian component, local axis). Coordinates
    must already be unwrapped under the explicitly declared nonperiodic or
    unwrapped policy; individual minimum images cannot define a base frame.
    Missing/degenerate rings retain their construct position and a false mask.
    tolerance is a relative degeneracy threshold, independent of length units.
    """
    if image_policy not in ("nonperiodic", "unwrapped") or (
        binding.periodic and image_policy != "unwrapped"
    ):
        raise ValueError(
            "Base frames require explicit nonperiodic or unwrapped coordinates."
        )
    coordinates = jnp.asarray(
        positions, dtype=jnp.result_type(jnp.asarray(positions).dtype, jnp.float32)
    )
    if coordinates.shape != (binding.support_size, 3):
        raise ValueError("Coordinates must match the bound atom support.")
    points = coordinates[binding.ring_indices]
    finite = jnp.all(jnp.isfinite(points), axis=(-2, -1))
    covered = jnp.all(binding.ring_mask, axis=-1)
    points = jnp.where(jnp.isfinite(points), points, 0.0)
    centers = jnp.mean(points, axis=1)
    first = points[:, 0] - centers
    first_sq = jnp.sum(first**2, axis=-1)
    ring_scale = jnp.mean(jnp.sum((points - centers[:, None, :]) ** 2, axis=-1), axis=-1)
    threshold = jnp.maximum(tolerance**2 * ring_scale, jnp.finfo(points.dtype).tiny)
    x = first / jnp.sqrt(jnp.where(first_sq > threshold, first_sq, 1.0))[:, None]
    normal = jnp.cross(x, points[:, 1] - centers)
    normal_sq = jnp.sum(normal**2, axis=-1)
    z = normal / jnp.sqrt(jnp.where(normal_sq > threshold, normal_sq, 1.0))[:, None]
    y = jnp.cross(z, x)
    valid = covered & finite & (first_sq > threshold) & (normal_sq > threshold)
    axes = jnp.stack((x, y, z), axis=-1)
    margin = jnp.sqrt(jnp.minimum(first_sq, normal_sq))
    return BaseFrameEvaluation(
        jnp.where(valid[:, None], centers, 0.0),
        jnp.where(valid[:, None, None], axes, jnp.eye(3)),
        valid,
        covered,
        margin,
    )


__all__ = ["BaseFrameEvaluation", "base_frames"]
