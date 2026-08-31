#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from ..imaging import (
    bilinear_sample,
    DenseDisplacementField2D,
    image_coordinates,
    ImagePair2D,
)
from ._deformation import interpolate_displacement
from ._types import PIVResult, ResidualDisparityDiagnostics2D


def residual_disparity(
    pair: ImagePair2D,
    source: DenseDisplacementField2D | PIVResult,
    /,
    *,
    stage: str = "replaced",
) -> ResidualDisparityDiagnostics2D:
    """Diagnose brightness disparity after sampling frame two at measured endpoints."""
    if not isinstance(pair, ImagePair2D):
        raise TypeError("pair must be an ImagePair2D.")
    if isinstance(source, PIVResult):
        if stage == "raw":
            field = source.raw
        elif stage == "validated":
            field = source.validated
        elif stage == "replaced":
            field = source.replaced
        else:
            raise ValueError("stage must be raw, validated, or replaced.")
    elif isinstance(source, DenseDisplacementField2D):
        field = source
    else:
        raise TypeError("source must be a DenseDisplacementField2D or PIVResult.")
    if field.geometry_id != pair.geometry.geometry_id:
        raise ValueError("Field and image-pair geometries differ.")
    coordinates = image_coordinates(pair.geometry)
    dense = interpolate_displacement(field, coordinates, extrapolate_nearest=True)
    warped_second = bilinear_sample(
        pair.second,
        coordinates + dense.values,
        valid_mask=pair.second_mask,
        fill_value=0.0,
    )
    valid = pair.first_mask & dense.valid & warped_second.valid
    residual = jnp.where(valid, warped_second.values - pair.first, 0.0)
    absolute = jnp.abs(residual)
    squared = residual * residual
    count = jnp.sum(valid)
    divisor = jnp.maximum(count, 1)
    mean = jnp.sum(residual) / divisor
    rms = jnp.sqrt(jnp.sum(squared) / divisor)
    maximum = jnp.max(jnp.where(valid, absolute, 0.0))
    return ResidualDisparityDiagnostics2D(
        residual,
        valid,
        absolute,
        squared,
        count / float(valid.size),
        mean,
        rms,
        maximum,
        field.field_id,
    )


__all__ = ["residual_disparity"]
