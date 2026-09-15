#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import linear_interpolate
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class CorrectionMap(StrictModule, NonTrainableState):
    coordinates: Array
    values: Array
    uncertainties: Array
    input_name: str = eqx.field(static=True)
    input_unit_id: str = eqx.field(static=True)
    output_name: str = eqx.field(static=True)
    correlation_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    correction_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        values: ArrayLike,
        uncertainties: ArrayLike,
        /,
        *,
        input_name: str,
        input_unit_id: str,
        output_name: str,
        correlation_id: str,
        source_id: str,
    ):
        coordinates_ = np.asarray(coordinates, dtype=float)
        values_ = np.asarray(values, dtype=float)
        uncertainties_ = np.asarray(uncertainties, dtype=float)
        if (
            coordinates_.ndim != 1
            or coordinates_.size < 2
            or values_.shape != coordinates_.shape
            or uncertainties_.shape != coordinates_.shape
            or np.any(~np.isfinite(coordinates_))
            or np.any(np.diff(coordinates_) <= 0.0)
        ):
            raise ValueError(
                "Correction coordinates/values/uncertainties must be aligned on an increasing axis."
            )
        if (
            np.any(~np.isfinite(values_))
            or np.any(~np.isfinite(uncertainties_))
            or np.any(uncertainties_ < 0.0)
        ):
            raise ValueError(
                "Correction values and uncertainties must be finite and uncertainties nonnegative."
            )
        labels = tuple(
            str(value).strip()
            for value in (
                input_name,
                input_unit_id,
                output_name,
                correlation_id,
                source_id,
            )
        )
        if any(not value for value in labels):
            raise ValueError("Correction semantic labels are required.")
        self.coordinates = jnp.asarray(coordinates_)
        self.values = jnp.asarray(values_)
        self.uncertainties = jnp.asarray(uncertainties_)
        (
            self.input_name,
            self.input_unit_id,
            self.output_name,
            self.correlation_id,
            self.source_id,
        ) = labels
        self.correction_id = canonical_fingerprint(
            {
                "kind": "collider-correction-map",
                "arrays": array_tree_fingerprint((coordinates_, values_, uncertainties_)),
                "labels": list(labels),
            }
        )


class CorrectionResult(StrictModule, NonTrainableState):
    factors: Array
    uncertainties: Array
    corrected_values: Array
    in_domain: Array
    finite: Array
    valid: Array
    correction_id: str = eqx.field(static=True)


def apply_correction(
    correction: CorrectionMap,
    coordinates: ArrayLike,
    uncorrected_values: ArrayLike,
    /,
) -> CorrectionResult:
    if not isinstance(correction, CorrectionMap):
        raise TypeError("correction must be CorrectionMap.")
    coordinates_ = jnp.asarray(coordinates, dtype=correction.coordinates.dtype)
    uncorrected = jnp.asarray(uncorrected_values, dtype=coordinates_.dtype)
    if coordinates_.shape != uncorrected.shape:
        raise ValueError("Correction coordinates and uncorrected values must align.")
    factors = linear_interpolate(
        correction.coordinates,
        correction.values,
        coordinates_,
        bounds="fill",
        fill_value=jnp.nan,
    )
    uncertainties = linear_interpolate(
        correction.coordinates,
        correction.uncertainties,
        coordinates_,
        bounds="fill",
        fill_value=jnp.nan,
    )
    corrected = uncorrected * factors.values
    finite = jnp.isfinite(corrected) & jnp.isfinite(uncertainties.values)
    valid = factors.support & uncertainties.support & finite
    return CorrectionResult(
        factors.values,
        uncertainties.values,
        jnp.where(valid, corrected, jnp.nan),
        factors.support & uncertainties.support,
        finite,
        valid,
        correction.correction_id,
    )


__all__ = ["CorrectionMap", "CorrectionResult", "apply_correction"]
