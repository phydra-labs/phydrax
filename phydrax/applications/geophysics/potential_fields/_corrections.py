#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._gravity import FreeSpaceGravityPlan, GravityResult


class TerrainCorrectionPlan(StrictModule, NonTrainableState):
    gravity: FreeSpaceGravityPlan
    density_contrast_kg_m3: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, gravity: FreeSpaceGravityPlan, density_contrast_kg_m3: ArrayLike, /
    ):
        if not isinstance(gravity, FreeSpaceGravityPlan):
            raise TypeError("Terrain correction requires a free-space gravity plan.")
        contrast = jnp.broadcast_to(
            jnp.asarray(density_contrast_kg_m3), (gravity.source.cell_count,)
        )
        contrast = eqx.error_if(
            contrast,
            jnp.any(~jnp.isfinite(contrast)),
            "Terrain density contrast must be finite.",
        )
        self.gravity, self.density_contrast_kg_m3 = gravity, contrast
        self.plan_id = canonical_fingerprint(
            {
                "kind": "terrain-correction-plan",
                "gravity": gravity.plan_id,
                "density_contrast_kg_m3": np.asarray(contrast),
            }
        )

    def correction(self) -> GravityResult:
        return self.gravity.evaluate(self.density_contrast_kg_m3)


class RegionalTrendPlan(StrictModule, NonTrainableState):
    design: Array
    terms: tuple[tuple[int, int], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, x: ArrayLike, y: ArrayLike, /, *, total_degree: int):
        x_, y_ = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        degree = int(total_degree)
        if (
            x_.shape != y_.shape
            or x_.ndim != 1
            or x_.size == 0
            or np.any(~np.isfinite(x_))
            or np.any(~np.isfinite(y_))
            or degree < 0
        ):
            raise ValueError("Regional trend coordinates and degree are invalid.")
        x_scale, y_scale = max(np.ptp(x_), 1.0), max(np.ptp(y_), 1.0)
        x_normalized = (x_ - np.mean(x_)) / x_scale
        y_normalized = (y_ - np.mean(y_)) / y_scale
        terms = tuple(
            (x_power, y_power)
            for total in range(degree + 1)
            for x_power in range(total + 1)
            for y_power in (total - x_power,)
        )
        design = np.stack(
            [x_normalized**x_power * y_normalized**y_power for x_power, y_power in terms],
            axis=1,
        )
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise ValueError(
                "Regional trend coordinates do not identify all polynomial terms."
            )
        self.design, self.terms = jnp.asarray(design), terms
        self.plan_id = canonical_fingerprint(
            {"kind": "regional-trend-plan", "design": design, "terms": terms}
        )

    def evaluate(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape != (self.design.shape[1],):
            raise ValueError("Regional coefficients must match polynomial terms.")
        return self.design @ values


class FourierContinuationPlan(StrictModule, NonTrainableState):
    shape: tuple[int, int] = eqx.field(static=True)
    spacing_m: tuple[float, float] = eqx.field(static=True)
    height_change_m: float = eqx.field(static=True)
    maximum_amplification: float = eqx.field(static=True)
    multiplier: Array
    direction: Literal["upward", "downward"] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        shape: tuple[int, int],
        spacing_m: tuple[float, float],
        height_change_m: float,
        /,
        *,
        maximum_amplification: float = 100.0,
    ):
        shape_ = tuple(int(value) for value in shape)
        spacing = tuple(float(value) for value in spacing_m)
        height = float(height_change_m)
        amplification = float(maximum_amplification)
        if (
            len(shape_) != 2
            or min(shape_) < 2
            or len(spacing) != 2
            or any(not np.isfinite(value) or value <= 0 for value in spacing)
            or not np.isfinite(height)
            or height == 0
            or not np.isfinite(amplification)
            or amplification <= 1
        ):
            raise ValueError(
                "Continuation grid, spacing, height, or amplification is invalid."
            )
        kx = 2 * np.pi * np.fft.fftfreq(shape_[0], d=spacing[0])
        ky = 2 * np.pi * np.fft.rfftfreq(shape_[1], d=spacing[1])
        wavenumber = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
        exponent = -wavenumber * height
        direction = "upward" if height > 0 else "downward"
        if direction == "downward":
            exponent = np.minimum(exponent, np.log(amplification))
        raw = np.exp(exponent)
        self.shape, self.spacing_m = shape_, spacing
        self.height_change_m, self.maximum_amplification = height, amplification
        self.multiplier, self.direction = jnp.asarray(raw), direction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fourier-potential-continuation",
                "shape": shape_,
                "spacing_m": spacing,
                "height_change_m": height,
                "maximum_amplification": amplification,
            }
        )

    def apply(self, field: ArrayLike, /) -> Array:
        values = jnp.asarray(field)
        if values.shape != self.shape or jnp.issubdtype(
            values.dtype, jnp.complexfloating
        ):
            raise ValueError(
                "Potential continuation requires one real grid matching its plan."
            )
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Potential continuation input must be finite.",
        )
        return jnp.fft.irfftn(
            jnp.fft.rfftn(values, norm="ortho") * self.multiplier,
            s=self.shape,
            norm="ortho",
        )

    @property
    def regularized(self) -> bool:
        return self.direction == "downward"


__all__ = ["FourierContinuationPlan", "RegionalTrendPlan", "TerrainCorrectionPlan"]
