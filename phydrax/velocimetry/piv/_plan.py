#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..imaging import ImageGeometry2D, ImagePair2D
from ._types import PIVPreparationReport, PIVResult, WindowGrid2D
from ._windows import _pair, prepare_window_grid


_CORRELATION_MODES = frozenset(("linear", "circular", "extended"))
_DEFORMATION_MODES = frozenset(("none", "second", "symmetric"))
_SUBPIXEL_METHODS = frozenset(("parabolic", "gaussian"))


class PIVPassPlan(StrictModule, NonTrainableState):
    """Static geometry and deformation policy for one interrogation pass."""

    window_size: tuple[int, int] = eqx.field(static=True)
    overlap: tuple[int, int] = eqx.field(static=True)
    search_margin: tuple[int, int] = eqx.field(static=True)
    deformation: str = eqx.field(static=True)
    pass_id: str = eqx.field(static=True)

    def __init__(
        self,
        window_size: int | Sequence[int],
        overlap: int | Sequence[int],
        search_margin: int | Sequence[int],
        /,
        *,
        deformation: str = "none",
    ):
        window = _pair(window_size, name="window_size", minimum=2)
        overlap_ = _pair(overlap, name="overlap", minimum=0)
        margin = _pair(search_margin, name="search_margin", minimum=0)
        deformation_ = str(deformation)
        if any(overlap_[axis] >= window[axis] for axis in range(2)):
            raise ValueError("overlap must be smaller than window_size.")
        if deformation_ not in _DEFORMATION_MODES:
            raise ValueError("deformation must be 'none', 'second', or 'symmetric'.")
        self.window_size = window
        self.overlap = overlap_
        self.search_margin = margin
        self.deformation = deformation_
        self.pass_id = canonical_fingerprint(
            {
                "kind": "piv-pass-plan",
                "window_size": list(window),
                "overlap": list(overlap_),
                "search_margin": list(margin),
                "deformation": deformation_,
            }
        )


class PIVPlan(StrictModule, NonTrainableState):
    """Resource-bounded classical PIV algorithm and decision policy."""

    passes: tuple[PIVPassPlan, ...]
    correlation_mode: str = eqx.field(static=True)
    normalized_correlation: bool = eqx.field(static=True)
    top_k: int = eqx.field(static=True)
    subpixel_method: str = eqx.field(static=True)
    minimum_valid_fraction: float = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    minimum_peak_ratio: float = eqx.field(static=True)
    minimum_correlation: float = eqx.field(static=True)
    maximum_displacement: tuple[float, float] = eqx.field(static=True)
    median_threshold: float = eqx.field(static=True)
    median_epsilon: float = eqx.field(static=True)
    minimum_neighbors: int = eqx.field(static=True)
    validation_radius: int = eqx.field(static=True)
    replacement_radius: int = eqx.field(static=True)
    replacement_iterations: int = eqx.field(static=True)
    retain_correlation: bool = eqx.field(static=True)
    requested_compute_dtype: str = eqx.field(static=True)
    resource_limit_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        passes: Sequence[PIVPassPlan],
        /,
        *,
        correlation_mode: str = "extended",
        normalized_correlation: bool = True,
        top_k: int = 2,
        subpixel_method: str = "gaussian",
        minimum_valid_fraction: float = 0.5,
        chunk_size: int = 64,
        minimum_peak_ratio: float = 1.05,
        minimum_correlation: float = 0.0,
        maximum_displacement: Sequence[float] = (float("inf"), float("inf")),
        median_threshold: float = 2.0,
        median_epsilon: float = 0.1,
        minimum_neighbors: int = 3,
        validation_radius: int = 1,
        replacement_radius: int = 1,
        replacement_iterations: int = 2,
        retain_correlation: bool = False,
        compute_dtype: str = "float32",
        resource_limit_bytes: int = 512 * 1024 * 1024,
    ):
        passes_ = tuple(passes)
        if not passes_ or any(not isinstance(item, PIVPassPlan) for item in passes_):
            raise ValueError("passes must contain at least one PIVPassPlan.")
        mode = str(correlation_mode)
        if mode not in _CORRELATION_MODES:
            raise ValueError("correlation_mode must be linear, circular, or extended.")
        method = str(subpixel_method)
        if method not in _SUBPIXEL_METHODS:
            raise ValueError("subpixel_method must be parabolic or gaussian.")
        if mode != "extended" and any(
            p.search_margin[axis] >= p.window_size[axis]
            for p in passes_
            for axis in range(2)
        ):
            raise ValueError(
                "Linear/circular search margins must be smaller than windows."
            )
        if mode == "circular" and any(
            2 * p.search_margin[axis] >= p.window_size[axis]
            for p in passes_
            for axis in range(2)
        ):
            raise ValueError("Circular search margins must be below half-window.")
        top_k_ = int(top_k)
        chunk_size_ = int(chunk_size)
        if top_k_ < 2:
            raise ValueError("top_k must be at least two for peak-ratio evidence.")
        if any(
            top_k_ > (2 * item.search_margin[0] + 1) * (2 * item.search_margin[1] + 1)
            for item in passes_
        ):
            raise ValueError("top_k exceeds a pass correlation-surface capacity.")
        if chunk_size_ < 1:
            raise ValueError("chunk_size must be positive.")
        valid_fraction = float(minimum_valid_fraction)
        if not 0.0 < valid_fraction <= 1.0:
            raise ValueError("minimum_valid_fraction must be in (0, 1].")
        maximum = tuple(float(value) for value in maximum_displacement)
        if len(maximum) != 2 or any(
            value <= 0.0 or math.isnan(value) for value in maximum
        ):
            raise ValueError("maximum_displacement must contain two positive values.")
        if (
            not math.isfinite(float(minimum_peak_ratio))
            or not math.isfinite(float(minimum_correlation))
            or minimum_peak_ratio < 0.0
            or minimum_correlation < -1.0
        ):
            raise ValueError("Correlation thresholds are outside their valid range.")
        if (
            not math.isfinite(float(median_threshold))
            or not math.isfinite(float(median_epsilon))
            or median_threshold <= 0.0
            or median_epsilon <= 0.0
        ):
            raise ValueError("Median validation thresholds must be positive.")
        if minimum_neighbors < 0 or validation_radius < 1:
            raise ValueError("Neighborhood validation settings are invalid.")
        if replacement_radius < 1 or replacement_iterations < 0:
            raise ValueError("Replacement settings are invalid.")
        dtype = str(compute_dtype)
        if dtype not in ("float32", "float64"):
            raise ValueError("compute_dtype must be float32 or float64.")
        resource_limit = int(resource_limit_bytes)
        if resource_limit < 1:
            raise ValueError("resource_limit_bytes must be positive.")
        payload = {
            "kind": "piv-plan",
            "passes": [item.pass_id for item in passes_],
            "correlation_mode": mode,
            "normalized_correlation": bool(normalized_correlation),
            "top_k": top_k_,
            "subpixel_method": method,
            "minimum_valid_fraction": valid_fraction,
            "chunk_size": chunk_size_,
            "minimum_peak_ratio": float(minimum_peak_ratio),
            "minimum_correlation": float(minimum_correlation),
            "maximum_displacement": [
                "infinity" if value == float("inf") else value for value in maximum
            ],
            "median_threshold": float(median_threshold),
            "median_epsilon": float(median_epsilon),
            "minimum_neighbors": int(minimum_neighbors),
            "validation_radius": int(validation_radius),
            "replacement_radius": int(replacement_radius),
            "replacement_iterations": int(replacement_iterations),
            "retain_correlation": bool(retain_correlation),
            "compute_dtype": dtype,
            "resource_limit_bytes": resource_limit,
        }
        self.passes = passes_
        self.correlation_mode = mode
        self.normalized_correlation = bool(normalized_correlation)
        self.top_k = top_k_
        self.subpixel_method = method
        self.minimum_valid_fraction = valid_fraction
        self.chunk_size = chunk_size_
        self.minimum_peak_ratio = float(minimum_peak_ratio)
        self.minimum_correlation = float(minimum_correlation)
        self.maximum_displacement = maximum
        self.median_threshold = float(median_threshold)
        self.median_epsilon = float(median_epsilon)
        self.minimum_neighbors = int(minimum_neighbors)
        self.validation_radius = int(validation_radius)
        self.replacement_radius = int(replacement_radius)
        self.replacement_iterations = int(replacement_iterations)
        self.retain_correlation = bool(retain_correlation)
        self.requested_compute_dtype = dtype
        self.resource_limit_bytes = resource_limit
        self.plan_id = canonical_fingerprint(payload)

    def prepare(self, geometry: ImageGeometry2D, /) -> PreparedPIV:
        return prepare_piv(self, geometry)


class PreparedPIV(StrictModule, NonTrainableState):
    """Prepared image-specific grids and resource/numeric evidence."""

    plan: PIVPlan
    grids: tuple[WindowGrid2D, ...]
    report: PIVPreparationReport
    geometry_id: str = eqx.field(static=True)
    image_shape: tuple[int, int] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def run(self, pair: ImagePair2D, /) -> PIVResult:
        from ._deformation import execute_piv

        return execute_piv(self, pair)


def _resolved_dtypes(requested: str) -> tuple[str, str]:
    resolved = requested
    if requested == "float64" and not bool(jax.config.jax_enable_x64):
        resolved = "float32"
    return resolved, "complex128" if resolved == "float64" else "complex64"


def _working_bytes(plan: PIVPlan, grid: WindowGrid2D, real_bytes: int) -> int:
    window = grid.window_size
    margin = grid.search_margin
    second = tuple(
        window[axis] + (2 * margin[axis] if plan.correlation_mode == "extended" else 0)
        for axis in range(2)
    )
    fft = tuple(
        window[axis] + second[axis] - 1
        if plan.correlation_mode != "circular"
        else window[axis]
        for axis in range(2)
    )
    real_values = (
        6 * (window[0] * window[1] + second[0] * second[1]) + 7 * fft[0] * fft[1]
    )
    complex_values = 6 * fft[0] * (fft[1] // 2 + 1)
    return plan.chunk_size * (real_values * real_bytes + complex_values * 2 * real_bytes)


def prepare_piv(plan: PIVPlan, geometry: ImageGeometry2D, /) -> PreparedPIV:
    """Resolve fixed grids, padded capacities, FFT precision, and memory bound."""
    if not isinstance(plan, PIVPlan):
        raise TypeError("plan must be a PIVPlan.")
    if not isinstance(geometry, ImageGeometry2D):
        raise TypeError("geometry must be an ImageGeometry2D.")
    grids = tuple(
        prepare_window_grid(
            geometry.image_shape, item.window_size, item.overlap, item.search_margin
        )
        for item in plan.passes
    )
    counts = tuple(grid.grid_shape[0] * grid.grid_shape[1] for grid in grids)
    padded = tuple(
        ((count + plan.chunk_size - 1) // plan.chunk_size) * plan.chunk_size
        for count in counts
    )
    resolved, fft_complex = _resolved_dtypes(plan.requested_compute_dtype)
    real_bytes = 8 if resolved == "float64" else 4
    working = max(_working_bytes(plan, grid, real_bytes) for grid in grids)
    if working > plan.resource_limit_bytes:
        raise MemoryError(
            f"Prepared PIV requires {working} working bytes, exceeding "
            f"resource_limit_bytes={plan.resource_limit_bytes}."
        )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-piv",
            "plan_id": plan.plan_id,
            "geometry_id": geometry.geometry_id,
            "grids": [grid.grid_id for grid in grids],
            "resolved_compute_dtype": resolved,
            "fft_complex_dtype": fft_complex,
        }
    )
    report = PIVPreparationReport(
        plan.plan_id,
        prepared_id,
        geometry.image_shape,
        tuple(grid.grid_shape for grid in grids),
        counts,
        padded,
        working,
        plan.resource_limit_bytes,
        plan.requested_compute_dtype,
        resolved,
        fft_complex,
        plan.correlation_mode,
        plan.retain_correlation,
    )
    return PreparedPIV(
        plan, grids, report, geometry.geometry_id, geometry.image_shape, prepared_id
    )


__all__ = ["PIVPassPlan", "PIVPlan", "PreparedPIV", "prepare_piv"]
