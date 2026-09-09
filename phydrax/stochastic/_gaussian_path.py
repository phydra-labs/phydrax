#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-grid Brownian paths from ordered independent Gaussian factors."""

from __future__ import annotations

from numbers import Integral
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import HermitianSpectrum
from ..special._dtype import promote_real
from ..special._normal import normal_quantile


GaussianPathConstructionMethod: TypeAlias = Literal["chronological", "bridge", "pca"]


class GaussianPathConstructionPlan(StrictModule, NonTrainableState):
    """A finite Brownian grid and its requested independent-factor ordering.

    ``times`` may start at any finite time. The constructed Brownian path is
    anchored at zero at ``times[0]`` and has covariance
    ``min(times[i] - times[0], times[j] - times[0])`` thereafter. Truncation is
    available only for the PCA construction; chronological and bridge plans use
    every interval factor.
    """

    times: Array
    method: GaussianPathConstructionMethod = eqx.field(static=True)
    factor_rank: int = eqx.field(static=True)
    num_times: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        method: GaussianPathConstructionMethod = "chronological",
        factor_rank: int | None = None,
    ):
        (nodes,) = promote_real("GaussianPathConstructionPlan times", times)
        nodes_host = np.asarray(jax.device_get(nodes))
        if nodes_host.ndim != 1 or nodes_host.size < 2:
            raise ValueError(
                "Gaussian path times must be one-dimensional with at least two nodes."
            )
        if np.any(~np.isfinite(nodes_host)):
            raise ValueError("Gaussian path times must be finite.")
        widths_host = np.diff(nodes_host)
        if np.any(~np.isfinite(widths_host)) or np.any(widths_host <= 0.0):
            raise ValueError(
                "Gaussian path times must have finite, strictly positive intervals."
            )
        if method not in ("chronological", "bridge", "pca"):
            raise ValueError("method must be 'chronological', 'bridge', or 'pca'.")

        full_rank = int(nodes_host.size - 1)
        if factor_rank is None:
            rank = full_rank
        elif isinstance(factor_rank, bool) or not isinstance(factor_rank, Integral):
            raise TypeError("factor_rank must be an integer or None.")
        else:
            rank = int(factor_rank)
        if rank <= 0 or rank > full_rank:
            raise ValueError(
                f"factor_rank must lie in [1, {full_rank}] for this time grid."
            )
        if method != "pca" and rank != full_rank:
            raise ValueError(
                "Only the PCA Gaussian path construction may be rank-truncated."
            )

        self.times = nodes
        self.method = method
        self.factor_rank = rank
        self.num_times = int(nodes_host.size)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gaussian-path-construction-plan",
                "times": array_tree_fingerprint(nodes_host),
                "method": method,
                "factor_rank": rank,
            }
        )


class PreparedGaussianPathConstruction(StrictModule, NonTrainableState):
    """Resolved Brownian path and increment factors on one fixed grid.

    ``ordering`` records grid-node indices for chronological and bridge methods.
    For PCA it records the source indices in the native ascending eigensystem,
    ordered from largest retained eigenvalue to smallest.
    """

    plan: GaussianPathConstructionPlan
    factor: Array
    increment_factor: Array
    covariance: Array
    covariance_residual: Array
    relative_covariance_residual: Array
    ordering: Array
    construction_id: str = eqx.field(static=True)


class GaussianPathConstructionEvidence(StrictModule):
    """Numerical reconstruction and covariance-factor evidence."""

    covariance_residual: Array
    relative_covariance_residual: Array
    factor_residual: Array
    reconstruction_residual: Array
    endpoint_residual: Array
    finite: Array
    valid: Array


class GaussianPathResult(StrictModule):
    """Brownian increments and their zero-anchored fixed-grid path."""

    times: Array
    standard_normals: Array
    increments: Array
    path: Array
    valid: Array
    evidence: GaussianPathConstructionEvidence
    plan_id: str = eqx.field(static=True)
    construction_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    @property
    def covariance_residual(self) -> Array:
        return self.evidence.covariance_residual

    @property
    def factor_residual(self) -> Array:
        return self.evidence.factor_residual

    @property
    def reconstruction_residual(self) -> Array:
        return self.evidence.reconstruction_residual


def _brownian_covariance(times: Array, /) -> Array:
    elapsed = times[1:] - times[0]
    return jnp.minimum(elapsed[:, None], elapsed[None, :])


def _chronological_factor(times: Array, /) -> tuple[Array, Array]:
    widths = jnp.diff(times)
    rank = int(widths.size)
    factor = (
        jnp.tril(jnp.ones((rank, rank), dtype=times.dtype)) * jnp.sqrt(widths)[None, :]
    )
    ordering = jnp.arange(1, rank + 1, dtype=jnp.int32)
    return factor, ordering


def _bridge_factor(times: Array, /) -> tuple[Array, Array]:
    nodes = np.asarray(jax.device_get(times))
    count = int(nodes.size)
    rank = count - 1
    coefficients = np.zeros((count, rank), dtype=nodes.dtype)
    ordering = np.empty((rank,), dtype=np.int32)
    horizon = nodes[-1] - nodes[0]
    coefficients[-1, 0] = np.sqrt(horizon)
    ordering[0] = count - 1
    intervals: list[tuple[int, int]] = [(0, count - 1)]

    for column in range(1, rank):
        splittable = tuple(
            interval for interval in intervals if interval[1] > interval[0] + 1
        )
        left, right = max(
            splittable,
            key=lambda interval: (nodes[interval[1]] - nodes[interval[0]], -interval[0]),
        )
        midpoint_time = 0.5 * (nodes[left] + nodes[right])
        middle = min(
            range(left + 1, right),
            key=lambda index: (abs(nodes[index] - midpoint_time), index),
        )
        intervals.remove((left, right))
        intervals.extend(((left, middle), (middle, right)))

        left_distance = nodes[middle] - nodes[left]
        right_distance = nodes[right] - nodes[middle]
        interval_width = nodes[right] - nodes[left]
        coefficients[middle] = (right_distance / interval_width) * coefficients[left] + (
            left_distance / interval_width
        ) * coefficients[right]
        coefficients[middle, column] = np.sqrt(
            left_distance * right_distance / interval_width
        )
        ordering[column] = middle

    return jnp.asarray(coefficients[1:]), jnp.asarray(ordering)


def _canonicalize_eigenvector_signs(vectors: Array, /) -> Array:
    columns = jnp.arange(vectors.shape[1])
    pivots = jnp.argmax(jnp.abs(vectors), axis=0)
    pivot_values = vectors[pivots, columns]
    signs = jnp.where(pivot_values < 0.0, -1.0, 1.0).astype(vectors.dtype)
    return vectors * signs[None, :]


def _pca_factor(
    covariance: Array,
    rank: int,
    /,
) -> tuple[Array, Array]:
    tolerance = float(
        256.0 * max(covariance.shape[0], 1) * np.finfo(np.dtype(covariance.dtype)).eps
    )
    spectrum = HermitianSpectrum(covariance, tolerance=tolerance)
    eigenvalues = spectrum.eigenvalues[::-1][:rank]
    eigenvectors = spectrum.eigenvectors[:, ::-1][:, :rank]
    eigenvectors = _canonicalize_eigenvector_signs(eigenvectors)
    factor = eigenvectors * jnp.sqrt(jnp.maximum(eigenvalues, 0.0))[None, :]
    full_rank = int(covariance.shape[0])
    ordering = jnp.arange(full_rank - 1, full_rank - rank - 1, -1, dtype=jnp.int32)
    return factor, ordering


def prepare_gaussian_path_construction(
    plan: GaussianPathConstructionPlan,
    /,
) -> PreparedGaussianPathConstruction:
    """Resolve a chronological, Brownian-bridge, or PCA Brownian factor."""
    if not isinstance(plan, GaussianPathConstructionPlan):
        raise TypeError("plan must be a GaussianPathConstructionPlan.")
    covariance = _brownian_covariance(plan.times)
    if plan.method == "chronological":
        factor, ordering = _chronological_factor(plan.times)
    elif plan.method == "bridge":
        factor, ordering = _bridge_factor(plan.times)
    else:
        factor, ordering = _pca_factor(covariance, plan.factor_rank)

    anchored_factor = jnp.concatenate(
        (jnp.zeros((1, plan.factor_rank), dtype=factor.dtype), factor), axis=0
    )
    increment_factor = jnp.diff(anchored_factor, axis=0)
    residual = factor @ factor.T - covariance
    covariance_scale = jnp.sqrt(jnp.sum(jnp.square(covariance)))
    residual_norm = jnp.sqrt(jnp.sum(jnp.square(residual)))
    relative_residual = residual_norm / jnp.maximum(
        covariance_scale, jnp.finfo(covariance.dtype).tiny
    )
    construction_id = canonical_fingerprint(
        {
            "kind": "prepared-gaussian-path-construction",
            "plan_id": plan.plan_id,
            "factor": array_tree_fingerprint(jax.device_get(factor)),
            "increment_factor": array_tree_fingerprint(jax.device_get(increment_factor)),
            "ordering": array_tree_fingerprint(jax.device_get(ordering)),
        }
    )
    return PreparedGaussianPathConstruction(
        plan=plan,
        factor=factor,
        increment_factor=increment_factor,
        covariance=covariance,
        covariance_residual=residual,
        relative_covariance_residual=relative_residual,
        ordering=ordering,
        construction_id=construction_id,
    )


def _prepared_and_factors(
    prepared: PreparedGaussianPathConstruction,
    values: ArrayLike,
    /,
    *,
    name: str,
) -> tuple[Array, Array, Array]:
    if not isinstance(prepared, PreparedGaussianPathConstruction):
        raise TypeError("prepared must be a PreparedGaussianPathConstruction.")
    normals, factor = promote_real(name, values, prepared.factor)
    increment_factor = jnp.asarray(prepared.increment_factor, dtype=factor.dtype)
    if normals.ndim < 1 or normals.shape[-1] != prepared.plan.factor_rank:
        raise ValueError(
            f"{name} must have final factor dimension {prepared.plan.factor_rank}."
        )
    normals_host = np.asarray(jax.device_get(normals))
    if np.any(~np.isfinite(normals_host)):
        raise ValueError(f"{name} active values must be finite.")
    return normals, factor, increment_factor


def brownian_increments_from_normals(
    prepared: PreparedGaussianPathConstruction,
    normals: ArrayLike,
    /,
) -> Array:
    """Map independent standard Normals to fixed-grid Brownian increments."""
    values, _, increment_factor = _prepared_and_factors(
        prepared,
        normals,
        name="normals",
    )
    return values @ increment_factor.T


def gaussian_path_from_unit_design(
    prepared: PreparedGaussianPathConstruction,
    unit_design: ArrayLike,
    /,
) -> GaussianPathResult:
    """Transform open-unit-cube points into Brownian increments and paths.

    Arbitrary leading design axes are preserved. Every final-axis coordinate is
    active, so values must be finite and strictly inside ``(0, 1)``; the closed
    endpoint behavior remains available through :func:`normal_quantile` itself.
    """
    if not isinstance(prepared, PreparedGaussianPathConstruction):
        raise TypeError("prepared must be a PreparedGaussianPathConstruction.")
    (design,) = promote_real("unit_design", unit_design)
    if design.ndim < 1 or design.shape[-1] != prepared.plan.factor_rank:
        raise ValueError(
            f"unit_design must have final factor dimension {prepared.plan.factor_rank}."
        )
    design_host = np.asarray(jax.device_get(design))
    if np.any(~np.isfinite(design_host)) or np.any(
        (design_host <= 0.0) | (design_host >= 1.0)
    ):
        raise ValueError("unit_design active values must be finite and lie in (0, 1).")

    raw_normals = normal_quantile(design)
    normals, factor, increment_factor = _prepared_and_factors(
        prepared,
        raw_normals,
        name="unit_design normal factors",
    )
    increments = normals @ increment_factor.T
    noninitial_path = normals @ factor.T
    initial = jnp.zeros(noninitial_path.shape[:-1] + (1,), dtype=noninitial_path.dtype)
    path = jnp.concatenate((initial, noninitial_path), axis=-1)

    reconstructed = jnp.concatenate((initial, jnp.cumsum(increments, axis=-1)), axis=-1)
    path_increments = jnp.diff(path, axis=-1)
    factor_residual = jnp.max(jnp.abs(increments - path_increments), axis=-1)
    reconstruction_residual = jnp.max(jnp.abs(reconstructed - path), axis=-1)
    if prepared.plan.method == "bridge":
        endpoint_reference = (
            jnp.sqrt(prepared.plan.times[-1] - prepared.plan.times[0]) * normals[..., 0]
        )
        endpoint_residual = jnp.abs(path[..., -1] - endpoint_reference)
    else:
        endpoint_residual = jnp.abs(path[..., -1] - reconstructed[..., -1])

    finite = (
        jnp.all(jnp.isfinite(normals), axis=-1)
        & jnp.all(jnp.isfinite(increments), axis=-1)
        & jnp.all(jnp.isfinite(path), axis=-1)
    )
    scale = jnp.maximum(jnp.max(jnp.abs(path), axis=-1), 1.0)
    tolerance = (
        128.0
        * max(prepared.plan.num_times, prepared.plan.factor_rank)
        * jnp.finfo(path.dtype).eps
        * scale
    )
    valid = (
        finite
        & (factor_residual <= tolerance)
        & (reconstruction_residual <= tolerance)
        & (endpoint_residual <= tolerance)
    )
    evidence = GaussianPathConstructionEvidence(
        covariance_residual=jnp.asarray(prepared.covariance_residual, dtype=path.dtype),
        relative_covariance_residual=jnp.asarray(
            prepared.relative_covariance_residual, dtype=path.dtype
        ),
        factor_residual=factor_residual,
        reconstruction_residual=reconstruction_residual,
        endpoint_residual=endpoint_residual,
        finite=finite,
        valid=valid,
    )
    realization_id = canonical_fingerprint(
        {
            "kind": "gaussian-path-realization",
            "construction_id": prepared.construction_id,
            "unit_design": array_tree_fingerprint(design_host),
        }
    )
    return GaussianPathResult(
        times=jnp.asarray(prepared.plan.times, dtype=path.dtype),
        standard_normals=normals,
        increments=increments,
        path=path,
        valid=valid,
        evidence=evidence,
        plan_id=prepared.plan.plan_id,
        construction_id=prepared.construction_id,
        realization_id=realization_id,
    )


__all__ = [
    "brownian_increments_from_normals",
    "gaussian_path_from_unit_design",
    "GaussianPathConstructionEvidence",
    "GaussianPathConstructionMethod",
    "GaussianPathConstructionPlan",
    "GaussianPathResult",
    "prepare_gaussian_path_construction",
    "PreparedGaussianPathConstruction",
]
