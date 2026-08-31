#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ..imaging import (
    bilinear_sample,
    DenseDisplacementField2D,
    image_coordinates,
    ImageGeometry2D,
    ImagePair2D,
    ImageSample2D,
)
from ._correlation import correlate_windows
from ._peaks import find_top_peaks
from ._replacement import replace_invalid_vectors
from ._types import (
    PIVQuality2D,
    PIVResult,
    PIVRetention,
    PIVStatus2D,
    PIVUncertainty2D,
    WindowGrid2D,
)
from ._validation import validate_field
from ._windows import extract_windows


class DeformedImagePair2D(StrictModule):
    """Deformed arrays and masks retained separately from the source pair."""

    first: Array
    second: Array
    first_mask: Array
    second_mask: Array
    predictor_rc: Array
    mode: str = eqx.field(static=True)


def interpolate_displacement(
    field: DenseDisplacementField2D,
    coordinates_rc: Array,
    /,
    *,
    extrapolate_nearest: bool = True,
) -> ImageSample2D:
    """Interpolate a rectilinear field, optionally extending its edge vectors."""
    if not isinstance(field, DenseDisplacementField2D):
        raise TypeError("field must be a DenseDisplacementField2D.")
    if field.positions_rc.ndim != 3:
        raise ValueError("field must have a two-dimensional rectilinear grid.")
    coordinates = jnp.asarray(coordinates_rc, dtype=float)
    if coordinates.shape[-1] != 2:
        raise ValueError("coordinates_rc must have shape (..., 2).")
    rows = field.positions_rc[:, 0, 0]
    columns = field.positions_rc[0, :, 1]
    row_coordinate = coordinates[..., 0]
    column_coordinate = coordinates[..., 1]
    if extrapolate_nearest:
        row_coordinate = jnp.clip(row_coordinate, rows[0], rows[-1])
        column_coordinate = jnp.clip(column_coordinate, columns[0], columns[-1])
    row_index = jnp.interp(row_coordinate, rows, jnp.arange(rows.shape[0], dtype=float))
    column_index = jnp.interp(
        column_coordinate, columns, jnp.arange(columns.shape[0], dtype=float)
    )
    index_coordinates = jnp.stack((row_index, column_index), axis=-1)
    sampled = bilinear_sample(
        field.displacement_rc,
        index_coordinates,
        valid_mask=field.valid,
        fill_value=0.0,
    )
    if extrapolate_nearest:
        return sampled
    inside = (
        (coordinates[..., 0] >= rows[0])
        & (coordinates[..., 0] <= rows[-1])
        & (coordinates[..., 1] >= columns[0])
        & (coordinates[..., 1] <= columns[-1])
    )
    valid = sampled.valid & inside
    return ImageSample2D(jnp.where(valid[..., None], sampled.values, 0.0), valid)


def predictor_at_grid(
    field: DenseDisplacementField2D | None,
    grid: WindowGrid2D,
    /,
) -> ImageSample2D:
    """Evaluate the previous-pass predictor on a new fixed interrogation grid."""
    if field is None:
        return ImageSample2D(
            jnp.zeros(grid.grid_shape + (2,), dtype=float),
            jnp.ones(grid.grid_shape, dtype=bool),
        )
    return interpolate_displacement(field, grid.centers_rc, extrapolate_nearest=True)


def deform_image_pair(
    pair: ImagePair2D,
    predictor: DenseDisplacementField2D | None,
    /,
    *,
    mode: str,
) -> DeformedImagePair2D:
    """Apply second-frame or symmetric deformation without changing the source pair."""
    if not isinstance(pair, ImagePair2D):
        raise TypeError("pair must be an ImagePair2D.")
    mode_ = str(mode)
    if mode_ not in ("none", "second", "symmetric"):
        raise ValueError("mode must be none, second, or symmetric.")
    coordinates = image_coordinates(pair.geometry)
    if predictor is None or mode_ == "none":
        predictor_values = jnp.zeros(pair.geometry.image_shape + (2,), dtype=float)
        return DeformedImagePair2D(
            pair.first,
            pair.second,
            pair.first_mask,
            pair.second_mask,
            predictor_values,
            mode_,
        )
    dense = interpolate_displacement(predictor, coordinates, extrapolate_nearest=True)
    predictor_values = jnp.where(dense.valid[..., None], dense.values, 0.0)
    if mode_ == "second":
        first = ImageSample2D(pair.first, pair.first_mask)
        second = bilinear_sample(
            pair.second,
            coordinates + predictor_values,
            valid_mask=pair.second_mask,
            fill_value=0.0,
        )
    else:
        first = bilinear_sample(
            pair.first,
            coordinates - 0.5 * predictor_values,
            valid_mask=pair.first_mask,
            fill_value=0.0,
        )
        second = bilinear_sample(
            pair.second,
            coordinates + 0.5 * predictor_values,
            valid_mask=pair.second_mask,
            fill_value=0.0,
        )
    return DeformedImagePair2D(
        first.values,
        second.values,
        first.valid & dense.valid,
        second.valid & dense.valid,
        predictor_values,
        mode_,
    )


def execute_piv(prepared: object, pair: ImagePair2D, /) -> PIVResult:
    """Execute every prepared pass and preserve raw/validated/replaced stages."""
    from ._plan import PreparedPIV

    if not isinstance(prepared, PreparedPIV):
        raise TypeError("prepared must be a PreparedPIV.")
    if not isinstance(pair, ImagePair2D):
        raise TypeError("pair must be an ImagePair2D.")
    if pair.geometry.geometry_id != prepared.geometry_id:
        raise ValueError("Image-pair geometry does not match the prepared geometry.")
    dtype = (
        jnp.float64
        if prepared.report.resolved_compute_dtype == "float64"
        else jnp.float32
    )
    runtime_pair = ImagePair2D(
        pair.first.astype(dtype),
        pair.second.astype(dtype),
        pair.geometry,
        first_mask=pair.first_mask,
        second_mask=pair.second_mask,
        delta_t=pair.delta_t,
        pair_id=pair.pair_id,
        provenance=pair.provenance,
    )
    predictor: DenseDisplacementField2D | None = None
    final = None
    for pass_index, (pass_plan, grid) in enumerate(
        zip(prepared.plan.passes, prepared.grids, strict=True)
    ):
        deformed = deform_image_pair(runtime_pair, predictor, mode=pass_plan.deformation)
        first_windows = extract_windows(
            deformed.first.astype(dtype),
            deformed.first_mask,
            grid,
        )
        second_windows = extract_windows(
            deformed.second.astype(dtype),
            deformed.second_mask,
            grid,
            extended=prepared.plan.correlation_mode == "extended",
        )
        correlation = correlate_windows(
            first_windows,
            second_windows,
            mode=prepared.plan.correlation_mode,
            search_margin=pass_plan.search_margin,
            chunk_size=prepared.plan.chunk_size,
            minimum_valid_fraction=prepared.plan.minimum_valid_fraction,
            normalized=prepared.plan.normalized_correlation,
        )
        peaks = find_top_peaks(
            correlation,
            top_k=prepared.plan.top_k,
            method=prepared.plan.subpixel_method,
        )
        prediction = predictor_at_grid(
            predictor if pass_plan.deformation != "none" else None,
            grid,
        )
        residual = peaks.offsets_rc[:, 0].reshape(grid.grid_shape + (2,)).astype(dtype)
        displacement = prediction.values.astype(dtype) + residual
        raw_valid = (
            prediction.valid
            & peaks.valid[:, 0].reshape(grid.grid_shape)
            & jnp.all(jnp.isfinite(displacement), axis=-1)
        )
        provenance = (
            pair.pair_id,
            prepared.prepared_id,
            f"pass-{pass_index}",
            pass_plan.deformation,
        )
        raw = DenseDisplacementField2D(
            grid.centers_rc.astype(dtype),
            displacement,
            raw_valid,
            geometry_id=prepared.geometry_id,
            provenance=provenance,
        )
        primary = peaks.values[:, 0].reshape(grid.grid_shape)
        secondary = peaks.values[:, 1].reshape(grid.grid_shape)
        epsilon = jnp.finfo(dtype).eps
        peak_ratio = primary / jnp.maximum(jnp.abs(secondary), epsilon)
        finite_surface = correlation.valid & jnp.isfinite(correlation.values)
        surface_sum = jnp.sum(
            jnp.where(finite_surface, correlation.values * correlation.values, 0.0),
            axis=(-2, -1),
        )
        surface_count = jnp.sum(finite_surface, axis=(-2, -1))
        surface_rms = jnp.sqrt(surface_sum / jnp.maximum(surface_count, 1))
        peak_to_rms = primary / jnp.maximum(surface_rms.reshape(grid.grid_shape), epsilon)
        integer_offset = jnp.rint(peaks.offsets_rc[:, 0]).astype(jnp.int32)
        row_index = jnp.clip(
            integer_offset[:, 0] + pass_plan.search_margin[0],
            0,
            correlation.overlap.shape[-2] - 1,
        )
        column_index = jnp.clip(
            integer_offset[:, 1] + pass_plan.search_margin[1],
            0,
            correlation.overlap.shape[-1] - 1,
        )
        batch_index = jnp.arange(correlation.overlap.shape[0], dtype=jnp.int32)
        selected_overlap = correlation.overlap[
            batch_index, row_index, column_index
        ].reshape(grid.grid_shape)
        overlap_fraction = selected_overlap / float(
            pass_plan.window_size[0] * pass_plan.window_size[1]
        )
        quality = PIVQuality2D(
            primary,
            secondary,
            peak_ratio,
            peak_to_rms,
            overlap_fraction,
        )
        covariance = peaks.covariance_rc[:, 0].reshape(grid.grid_shape + (2, 2))
        uncertainty_valid = raw_valid & jnp.all(jnp.isfinite(covariance), axis=(-2, -1))
        uncertainty = PIVUncertainty2D(
            covariance,
            uncertainty_valid,
            f"{prepared.plan.subpixel_method}-peak-curvature",
        )
        validated, validation_evidence = validate_field(
            raw,
            quality,
            maximum_displacement=prepared.plan.maximum_displacement,
            minimum_correlation=prepared.plan.minimum_correlation,
            minimum_peak_ratio=prepared.plan.minimum_peak_ratio,
            radius=prepared.plan.validation_radius,
            minimum_neighbors=prepared.plan.minimum_neighbors,
            median_threshold=prepared.plan.median_threshold,
            median_epsilon=prepared.plan.median_epsilon,
        )
        replaced, replacement_evidence = replace_invalid_vectors(
            validated,
            radius=prepared.plan.replacement_radius,
            iterations=prepared.plan.replacement_iterations,
            minimum_neighbors=max(1, prepared.plan.minimum_neighbors),
        )
        correlated = jnp.isfinite(primary)
        peak_fitted = raw.valid
        code = (
            (~correlated).astype(jnp.int32)
            | ((~peak_fitted).astype(jnp.int32) << 1)
            | ((~validated.valid).astype(jnp.int32) << 2)
            | (replacement_evidence.unresolved.astype(jnp.int32) << 3)
            | (replacement_evidence.replaced.astype(jnp.int32) << 4)
        )
        status = PIVStatus2D(
            code,
            correlated,
            peak_fitted,
            validated.valid,
            replacement_evidence.replaced,
        )
        if prepared.plan.retain_correlation:
            retained_correlation = correlation.values.reshape(
                grid.grid_shape + correlation.values.shape[-2:]
            )
            retained_overlap = correlation.overlap.reshape(
                grid.grid_shape + correlation.overlap.shape[-2:]
            )
        else:
            retained_correlation = jnp.zeros((0, 0, 0), dtype=dtype)
            retained_overlap = jnp.zeros((0, 0, 0), dtype=dtype)
        retention = PIVRetention(
            retained_correlation,
            retained_overlap,
            correlation.lags_rc,
            prepared.plan.retain_correlation,
            pair.pair_id,
            prepared.prepared_id,
            prepared.plan.requested_compute_dtype,
            prepared.report.resolved_compute_dtype,
            prepared.report.fft_complex_dtype,
        )
        final = PIVResult(
            raw,
            validated,
            replaced,
            quality,
            uncertainty,
            validation_evidence,
            replacement_evidence,
            status,
            retention,
            pair.pair_id,
            prepared.plan.plan_id,
            prepared.prepared_id,
            prepared.plan.requested_compute_dtype,
            prepared.report.resolved_compute_dtype,
            prepared.report.fft_complex_dtype,
        )
        predictor = replaced
    if final is None:
        raise RuntimeError("Prepared PIV contains no passes.")
    return final


def piv(
    first: Array,
    second: Array,
    plan: object,
    /,
    *,
    geometry: ImageGeometry2D | None = None,
    first_mask: Array | None = None,
    second_mask: Array | None = None,
    delta_t: Array | float = 1.0,
    pair_id: str | None = None,
    provenance: tuple[str, ...] = (),
) -> PIVResult:
    """One-shot array-to-result convenience with explicit preparation evidence."""
    from ._plan import PIVPlan

    if not isinstance(plan, PIVPlan):
        raise TypeError("plan must be a PIVPlan.")
    geometry_ = ImageGeometry2D(jnp.shape(first)) if geometry is None else geometry
    pair = ImagePair2D(
        first,
        second,
        geometry_,
        first_mask=first_mask,
        second_mask=second_mask,
        delta_t=delta_t,
        pair_id=pair_id,
        provenance=provenance,
    )
    return plan.prepare(geometry_).run(pair)


__all__ = [
    "DeformedImagePair2D",
    "deform_image_pair",
    "execute_piv",
    "interpolate_displacement",
    "piv",
    "predictor_at_grid",
]
