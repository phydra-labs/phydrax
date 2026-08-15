#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._chart import CoordinateChart
from ._metric import LorentzianConvention, LorentzianMetric


class _MinkowskiMetricMap(StrictModule):
    dimension: int = eqx.field(static=True)
    convention: LorentzianConvention = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        convention: LorentzianConvention,
        /,
    ):
        self.dimension = int(dimension)
        self.convention = convention

    def __call__(self, coordinates: Array, /) -> Array:
        signs = jnp.ones((self.dimension,), dtype=coordinates.dtype)
        signs = signs.at[0].set(-1.0)
        matrix = jnp.diag(signs)
        return matrix if self.convention == "mostly_plus" else -matrix


class _FLRWMetricMap(StrictModule):
    scale_factor: Callable[[Array], Array]
    spatial_curvature: int = eqx.field(static=True)
    convention: LorentzianConvention = eqx.field(static=True)

    def __init__(
        self,
        scale_factor: Callable[[Array], Array],
        spatial_curvature: int,
        convention: LorentzianConvention,
        /,
    ):
        self.scale_factor = scale_factor
        self.spatial_curvature = int(spatial_curvature)
        self.convention = convention

    def __call__(self, coordinates: Array, /) -> Array:
        time, radius, polar, _ = coordinates
        scale = jnp.asarray(self.scale_factor(time))
        if scale.shape != ():
            raise ValueError("FLRW scale_factor must return one scalar.")
        denominator = 1.0 - self.spatial_curvature * radius**2
        scale = eqx.error_if(
            scale,
            ~jnp.isfinite(scale) | (scale <= 0.0),
            "FLRW scale_factor must be finite and positive.",
        )
        denominator = eqx.error_if(
            denominator,
            denominator <= 0.0,
            "FLRW spherical coordinates require 1 - k r² > 0.",
        )
        scale_squared = scale**2
        diagonal = jnp.stack(
            (
                -jnp.ones_like(scale),
                scale_squared / denominator,
                scale_squared * radius**2,
                scale_squared * radius**2 * jnp.sin(polar) ** 2,
            )
        )
        matrix = jnp.diag(diagonal)
        return matrix if self.convention == "mostly_plus" else -matrix


def _assemble_adm_matrix(
    lapse: Array,
    shift: Array,
    spatial_metric: Array,
    convention: LorentzianConvention,
    /,
) -> Array:
    shift_covector = oe.contract("...ij,...j->...i", spatial_metric, shift)
    time_time = -(lapse**2) + oe.contract(
        "...i,...i->...",
        shift,
        shift_covector,
    )
    first_row = jnp.concatenate((time_time[..., None], shift_covector), axis=-1)
    remaining = jnp.concatenate(
        (shift_covector[..., :, None], spatial_metric),
        axis=-1,
    )
    matrix = jnp.concatenate((first_row[..., None, :], remaining), axis=-2)
    return matrix if convention == "mostly_plus" else -matrix


class _ADMMetricMap(StrictModule):
    lapse: Callable[[Array], Array]
    shift: Callable[[Array], Array]
    spatial_metric: Callable[[Array], Array]
    spatial_dimension: int = eqx.field(static=True)
    convention: LorentzianConvention = eqx.field(static=True)

    def __init__(
        self,
        lapse: Callable[[Array], Array],
        shift: Callable[[Array], Array],
        spatial_metric: Callable[[Array], Array],
        spatial_dimension: int,
        convention: LorentzianConvention,
        /,
    ):
        self.lapse = lapse
        self.shift = shift
        self.spatial_metric = spatial_metric
        self.spatial_dimension = int(spatial_dimension)
        self.convention = convention

    def __call__(self, coordinates: Array, /) -> Array:
        lapse = jnp.asarray(self.lapse(coordinates))
        shift = jnp.asarray(self.shift(coordinates))
        spatial = jnp.asarray(self.spatial_metric(coordinates))
        if lapse.shape != ():
            raise ValueError("ADM lapse must return one scalar.")
        if shift.shape != (self.spatial_dimension,):
            raise ValueError(
                f"ADM shift must have shape {(self.spatial_dimension,)}; "
                f"got {shift.shape}."
            )
        expected = (self.spatial_dimension, self.spatial_dimension)
        if spatial.shape != expected:
            raise ValueError(
                f"ADM spatial_metric must have shape {expected}; got {spatial.shape}."
            )
        lapse = eqx.error_if(
            lapse,
            ~jnp.isfinite(lapse) | (lapse <= 0.0),
            "ADM lapse must be finite and positive.",
        )
        spatial = eqx.error_if(
            spatial,
            jnp.any(~jnp.isfinite(spatial))
            | jnp.any(jnp.abs(spatial - spatial.T) > 1e-10)
            | (jnp.min(jnp.linalg.eigvalsh(spatial)) <= 0.0),
            "ADM spatial_metric must be finite, symmetric, and positive definite.",
        )
        return _assemble_adm_matrix(lapse, shift, spatial, self.convention)


class _SchwarzschildMetricMap(StrictModule):
    mass: Array
    convention: LorentzianConvention = eqx.field(static=True)

    def __init__(self, mass: ArrayLike, convention: LorentzianConvention, /):
        mass_array = jnp.asarray(mass)
        if mass_array.shape != ():
            raise ValueError("Schwarzschild mass must be scalar.")
        self.mass = mass_array
        self.convention = convention

    def __call__(self, coordinates: Array, /) -> Array:
        _, radius, polar, _ = coordinates
        mass = eqx.error_if(
            self.mass,
            ~jnp.isfinite(self.mass) | (self.mass <= 0.0),
            "Schwarzschild mass must be finite and positive.",
        )
        factor = 1.0 - 2.0 * mass / radius
        factor = eqx.error_if(
            factor,
            ~jnp.isfinite(factor) | (factor <= 0.0),
            "Schwarzschild coordinates require radius > 2 * mass.",
        )
        diagonal = jnp.stack(
            (
                -factor,
                1.0 / factor,
                radius**2,
                radius**2 * jnp.sin(polar) ** 2,
            )
        )
        matrix = jnp.diag(diagonal)
        return matrix if self.convention == "mostly_plus" else -matrix


def minkowski_metric(
    chart: CoordinateChart,
    /,
    *,
    convention: LorentzianConvention = "mostly_plus",
) -> LorentzianMetric:
    """Construct flat Minkowski spacetime in Cartesian coordinates."""
    return LorentzianMetric(
        _MinkowskiMetricMap(chart.dimension, convention),
        chart=chart,
        convention=convention,
    )


def flrw_metric(
    scale_factor: Callable[[Array], Array],
    /,
    *,
    chart: CoordinateChart,
    spatial_curvature: int = 0,
    convention: LorentzianConvention = "mostly_plus",
) -> LorentzianMetric:
    """Construct a 3+1 FLRW metric in ``(t, r, θ, φ)`` coordinates."""
    if chart.dimension != 4:
        raise ValueError("flrw_metric requires a four-dimensional chart.")
    if not callable(scale_factor):
        raise TypeError("scale_factor must be callable.")
    if int(spatial_curvature) not in (-1, 0, 1):
        raise ValueError("spatial_curvature must be -1, 0, or 1.")
    return LorentzianMetric(
        _FLRWMetricMap(scale_factor, int(spatial_curvature), convention),
        chart=chart,
        convention=convention,
    )


def adm_metric(
    lapse: Callable[[Array], Array],
    shift: Callable[[Array], Array],
    spatial_metric: Callable[[Array], Array],
    /,
    *,
    chart: CoordinateChart,
    convention: LorentzianConvention = "mostly_plus",
) -> LorentzianMetric:
    """Construct a Lorentzian metric from lapse, shift, and an SPD spatial metric."""
    if chart.dimension < 2:
        raise ValueError("adm_metric requires at least one spatial dimension.")
    if not callable(lapse) or not callable(shift) or not callable(spatial_metric):
        raise TypeError("ADM lapse, shift, and spatial_metric must be callable.")
    return LorentzianMetric(
        _ADMMetricMap(
            lapse,
            shift,
            spatial_metric,
            chart.dimension - 1,
            convention,
        ),
        chart=chart,
        convention=convention,
    )


def schwarzschild_metric(
    mass: ArrayLike,
    /,
    *,
    chart: CoordinateChart,
    convention: LorentzianConvention = "mostly_plus",
) -> LorentzianMetric:
    """Construct the exterior Schwarzschild metric in spherical coordinates."""
    if chart.dimension != 4:
        raise ValueError("schwarzschild_metric requires a four-dimensional chart.")
    return LorentzianMetric(
        _SchwarzschildMetricMap(mass, convention),
        chart=chart,
        convention=convention,
    )


__all__ = [
    "adm_metric",
    "flrw_metric",
    "minkowski_metric",
    "schwarzschild_metric",
]
