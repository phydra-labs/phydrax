#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._hypersurface import ProjectiveHypersurface
from ._hypersurface_patch import HypersurfacePatchGeometry


class ProjectiveLineSamples(StrictModule):
    homogeneous_points: Array
    chart_indices: Array
    pivot_indices: Array
    polynomial_residuals: Array
    smoothness_margins: Array
    valid: Array
    line_ids: Array
    root_ids: Array

    def __init__(
        self,
        *,
        homogeneous_points: ArrayLike,
        chart_indices: ArrayLike,
        pivot_indices: ArrayLike,
        polynomial_residuals: ArrayLike,
        smoothness_margins: ArrayLike,
        valid: ArrayLike,
        line_ids: ArrayLike,
        root_ids: ArrayLike,
    ):
        self.homogeneous_points = jnp.asarray(homogeneous_points)
        self.chart_indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        self.pivot_indices = jnp.asarray(pivot_indices, dtype=jnp.int32)
        self.polynomial_residuals = jnp.asarray(polynomial_residuals)
        self.smoothness_margins = jnp.asarray(smoothness_margins)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.line_ids = jnp.asarray(line_ids, dtype=jnp.int32)
        self.root_ids = jnp.asarray(root_ids, dtype=jnp.int32)


def intersect_projective_line(
    hypersurface: ProjectiveHypersurface,
    start: ArrayLike,
    direction: ArrayLike,
    /,
) -> Array:
    """Return host-computed roots of one projective-line intersection."""
    if not isinstance(hypersurface, ProjectiveHypersurface):
        raise TypeError("hypersurface must be a ProjectiveHypersurface.")
    start_ = np.asarray(start, dtype=np.complex128)
    direction_ = np.asarray(direction, dtype=np.complex128)
    expected = (hypersurface.projective_dimension + 1,)
    if start_.shape != expected or direction_.shape != expected:
        raise ValueError(f"Line vectors must have shape {expected}.")
    count = hypersurface.degree + 1
    roots_of_unity = np.exp(2j * np.pi * np.arange(count) / count)
    values = np.asarray(
        [
            hypersurface.polynomial(jnp.asarray(start_ + root * direction_))
            for root in roots_of_unity
        ],
        dtype=np.complex128,
    )
    coefficients = np.fft.fft(values) / count
    threshold = np.finfo(float).eps * max(1.0, np.max(np.abs(coefficients))) * 100
    while coefficients.size > 1 and abs(coefficients[-1]) <= threshold:
        coefficients = coefficients[:-1]
    roots = np.roots(coefficients[::-1])
    return jnp.asarray(start_[None, :] + roots[:, None] * direction_[None, :])


def sample_projective_hypersurface(
    hypersurface: ProjectiveHypersurface,
    key: Array,
    line_count: int,
    /,
    *,
    tolerance: float = 1e-7,
) -> ProjectiveLineSamples:
    if not isinstance(hypersurface, ProjectiveHypersurface):
        raise TypeError("hypersurface must be a ProjectiveHypersurface.")
    count = int(line_count)
    if count < 1:
        raise ValueError("line_count must be positive.")
    dimension = hypersurface.projective_dimension + 1
    start_key, direction_key = jax.random.split(key)
    starts = jax.random.normal(start_key, (count, dimension, 2))
    directions = jax.random.normal(direction_key, (count, dimension, 2))
    starts_complex = starts[..., 0] + 1j * starts[..., 1]
    directions_complex = directions[..., 0] + 1j * directions[..., 1]
    point_blocks = [
        intersect_projective_line(
            hypersurface, starts_complex[index], directions_complex[index]
        )
        for index in range(count)
    ]
    points = jnp.concatenate(point_blocks, axis=0)
    points = points / jnp.linalg.norm(points, axis=-1, keepdims=True)
    geometry = HypersurfacePatchGeometry(hypersurface, tolerance=tolerance)
    evaluations = [geometry.evaluate(points[index]) for index in range(points.shape[0])]
    return ProjectiveLineSamples(
        homogeneous_points=points,
        chart_indices=jnp.asarray([value.chart_index for value in evaluations]),
        pivot_indices=jnp.asarray([value.pivot_index for value in evaluations]),
        polynomial_residuals=jnp.stack(
            [value.polynomial_residual for value in evaluations]
        ),
        smoothness_margins=jnp.stack([value.smoothness_margin for value in evaluations]),
        valid=jnp.stack([value.valid for value in evaluations]),
        line_ids=jnp.repeat(
            jnp.arange(count), jnp.asarray([block.shape[0] for block in point_blocks])
        ),
        root_ids=jnp.concatenate([jnp.arange(block.shape[0]) for block in point_blocks]),
    )


__all__ = [
    "ProjectiveLineSamples",
    "intersect_projective_line",
    "sample_projective_hypersurface",
]
