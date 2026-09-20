#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix._metric import LorentzianMetric
from ...metrix._spacetime_conventions import RelativityConvention
from ...units import UnitDefinition
from ._gr_bundles import gr_chart_identity, gr_metric_identity


GRTemporalDirection: TypeAlias = Literal["future", "past"]


_LEVI_CIVITA_4 = jnp.asarray(
    [
        [
            [
                [
                    float((i - j) * (i - k) * (i - l) * (j - k) * (j - l) * (k - l) / 12)
                    for l in range(4)
                ]
                for k in range(4)
            ]
            for j in range(4)
        ]
        for i in range(4)
    ]
)


def _inner(metric: Array, left: Array, right: Array, /) -> Array:
    return ein.contract("...i,...ij,...j->...", left, metric, right)


def _normalize_signed(
    vector: Array,
    metric: Array,
    sign: float,
    /,
) -> tuple[Array, Array]:
    norm = sign * _inner(metric, vector, vector)
    finite = jnp.all(jnp.isfinite(vector), axis=-1) & jnp.isfinite(norm)
    valid = finite & (norm > 0.0)
    safe = jnp.sqrt(jnp.maximum(norm, jnp.finfo(vector.dtype).tiny))
    return vector / safe[..., None], valid


def _project(vector: Array, basis: Array, metric: Array, /) -> Array:
    numerator = _inner(metric, vector, basis)
    denominator = _inner(metric, basis, basis)
    safe = jnp.where(
        jnp.abs(denominator) > jnp.finfo(vector.dtype).tiny,
        denominator,
        jnp.ones_like(denominator),
    )
    return vector - (numerator / safe)[..., None] * basis


class GRObserverScreenPlan(StrictModule):
    """Observer event, tetrad seeds, and fixed pixel directions for GR rays."""

    metric: LorentzianMetric
    scale: RelativityScaleContract
    convention: RelativityConvention
    coordinate_unit: UnitDefinition
    affine_parameter_unit: UnitDefinition
    observer_coordinates: Array
    observer_velocity: Array
    line_of_sight: Array
    screen_up: Array
    pixel_coordinates: Array
    pixel_mask: Array
    ray_energy: Array
    temporal_direction: GRTemporalDirection = eqx.field(static=True)
    orthonormal_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    coordinate_unit_id: str = eqx.field(static=True)
    affine_parameter_unit_id: str = eqx.field(static=True)

    def __init__(
        self,
        metric: LorentzianMetric,
        observer_coordinates: ArrayLike,
        observer_velocity: ArrayLike,
        line_of_sight: ArrayLike,
        screen_up: ArrayLike,
        pixel_coordinates: ArrayLike,
        /,
        *,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        coordinate_unit: UnitDefinition,
        affine_parameter_unit: UnitDefinition,
        metric_semantic_id: str | None = None,
        metric_numeric_id: str | None = None,
        pixel_mask: ArrayLike | None = None,
        ray_energy: ArrayLike = 1.0,
        temporal_direction: GRTemporalDirection = "past",
        orthonormal_tolerance: float = 1.0e-6,
        plan_id: str | None = None,
    ):
        if not isinstance(metric, LorentzianMetric) or metric.chart.dimension != 4:
            raise TypeError(
                "GR observer screens require a four-dimensional LorentzianMetric."
            )
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        if not isinstance(convention, RelativityConvention):
            raise TypeError("convention must be a RelativityConvention.")
        if convention.metric_signature != metric.convention:
            raise ValueError(
                "RelativityConvention metric signature must match the metric."
            )
        if not isinstance(coordinate_unit, UnitDefinition):
            raise TypeError("coordinate_unit must be a UnitDefinition.")
        if not isinstance(affine_parameter_unit, UnitDefinition):
            raise TypeError("affine_parameter_unit must be a UnitDefinition.")
        coordinates = jnp.asarray(observer_coordinates)
        if jnp.iscomplexobj(coordinates):
            raise TypeError("Observer coordinates must be real.")
        if not jnp.issubdtype(coordinates.dtype, jnp.floating):
            coordinates = coordinates.astype("float64")
        velocity = jnp.asarray(observer_velocity, dtype=coordinates.dtype)
        sight = jnp.asarray(line_of_sight, dtype=coordinates.dtype)
        up = jnp.asarray(screen_up, dtype=coordinates.dtype)
        pixels = jnp.asarray(pixel_coordinates, dtype=coordinates.dtype)
        if any(value.shape != (4,) for value in (coordinates, velocity, sight, up)):
            raise ValueError("Observer event and tetrad seeds must each have shape (4,).")
        if pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.shape[0] < 1:
            raise ValueError("pixel_coordinates must have shape (num_rays, 2).")
        mask = (
            jnp.ones((pixels.shape[0],), dtype=jnp.bool_)
            if pixel_mask is None
            else jnp.asarray(pixel_mask, dtype=jnp.bool_)
        )
        if mask.shape != (pixels.shape[0],):
            raise ValueError("pixel_mask must have shape (num_rays,).")
        energy = jnp.asarray(ray_energy, dtype=coordinates.dtype)
        if energy.shape not in ((), (pixels.shape[0],)):
            raise ValueError("ray_energy must be scalar or have shape (num_rays,).")
        energy = jnp.broadcast_to(energy, (pixels.shape[0],))
        if temporal_direction not in ("future", "past"):
            raise ValueError("temporal_direction must be 'future' or 'past'.")
        tolerance = float(orthonormal_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("orthonormal_tolerance must be finite and positive.")
        metric_id = gr_metric_identity(
            metric,
            semantic_id=metric_semantic_id,
            numeric_id=metric_numeric_id,
        )
        chart_id = gr_chart_identity(metric)
        plan_id_ = (
            canonical_fingerprint(
                {
                    "kind": "gr-observer-screen-plan",
                    "chart_id": chart_id,
                    "metric_id": metric_id,
                    "convention_id": convention.convention_id,
                    "scale_id": scale.scale_id,
                    "coordinate_unit_id": coordinate_unit.unit_id,
                    "affine_parameter_unit_id": affine_parameter_unit.unit_id,
                    "observer": coordinates,
                    "velocity": velocity,
                    "line_of_sight": sight,
                    "screen_up": up,
                    "pixels": pixels,
                    "pixel_mask": mask,
                    "ray_energy": energy,
                    "temporal_direction": temporal_direction,
                    "orthonormal_tolerance": tolerance,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not plan_id_:
            raise ValueError("plan_id must be non-empty.")
        self.metric = metric
        self.scale = scale
        self.convention = convention
        self.coordinate_unit = coordinate_unit
        self.affine_parameter_unit = affine_parameter_unit
        self.observer_coordinates = coordinates
        self.observer_velocity = velocity
        self.line_of_sight = sight
        self.screen_up = up
        self.pixel_coordinates = pixels
        self.pixel_mask = mask
        self.ray_energy = energy
        self.temporal_direction = temporal_direction
        self.orthonormal_tolerance = tolerance
        self.plan_id = plan_id_
        self.metric_id = metric_id
        self.chart_id = chart_id
        self.convention_id = convention.convention_id
        self.scale_id = scale.scale_id
        self.coordinate_unit_id = coordinate_unit.unit_id
        self.affine_parameter_unit_id = affine_parameter_unit.unit_id

    def initialize(self) -> GRObserverScreenResult:
        return initialize_gr_observer_screen(self)


class GRObserverScreenResult(StrictModule, NonTrainableState):
    """Orthonormal observer tetrad and one null initial state per pixel."""

    observer_coordinates: Array
    tetrad: Array
    pixel_coordinates: Array
    ray_coordinates: Array
    ray_tangents: Array
    screen_basis: Array
    valid: Array
    tetrad_residual: Array
    null_residual: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    coordinate_unit_id: str = eqx.field(static=True)
    affine_parameter_unit_id: str = eqx.field(static=True)

    def __init__(
        self,
        observer_coordinates: ArrayLike,
        tetrad: ArrayLike,
        pixel_coordinates: ArrayLike,
        ray_coordinates: ArrayLike,
        ray_tangents: ArrayLike,
        screen_basis: ArrayLike,
        valid: ArrayLike,
        tetrad_residual: ArrayLike,
        null_residual: ArrayLike,
        /,
        *,
        plan_id: str,
        result_id: str,
        metric_id: str,
        chart_id: str,
        convention_id: str,
        scale_id: str,
        coordinate_unit_id: str,
        affine_parameter_unit_id: str,
    ):
        observer = jnp.asarray(observer_coordinates)
        tetrad_ = jnp.asarray(tetrad, dtype=observer.dtype)
        pixels = jnp.asarray(pixel_coordinates, dtype=observer.dtype)
        coordinates = jnp.asarray(ray_coordinates, dtype=observer.dtype)
        tangents = jnp.asarray(ray_tangents, dtype=observer.dtype)
        basis = jnp.asarray(screen_basis, dtype=observer.dtype)
        valid_ = jnp.asarray(valid, dtype=jnp.bool_)
        tetrad_error = jnp.asarray(tetrad_residual, dtype=observer.dtype)
        null_error = jnp.asarray(null_residual, dtype=observer.dtype)
        count = pixels.shape[0]
        if observer.shape != (4,) or tetrad_.shape != (4, 4):
            raise ValueError(
                "Observer coordinates and tetrad must have shapes (4,) and (4, 4)."
            )
        if pixels.shape != (count, 2):
            raise ValueError("pixel_coordinates must have shape (num_rays, 2).")
        if coordinates.shape != (count, 4) or tangents.shape != (count, 4):
            raise ValueError(
                "Screen ray coordinates and tangents must have shape (num_rays, 4)."
            )
        if basis.shape != (count, 2, 4):
            raise ValueError("Screen bases must have shape (num_rays, 2, 4).")
        if valid_.shape != (count,) or null_error.shape != (count,):
            raise ValueError("Screen validity and null residual must match the ray axis.")
        if tetrad_error.shape != ():
            raise ValueError("tetrad_residual must be scalar.")
        identities = (
            plan_id,
            result_id,
            metric_id,
            chart_id,
            convention_id,
            scale_id,
            coordinate_unit_id,
            affine_parameter_unit_id,
        )
        if any(not isinstance(value, str) or not value for value in identities):
            raise ValueError("GR observer screen identities must be non-empty strings.")
        self.observer_coordinates = observer
        self.tetrad = tetrad_
        self.pixel_coordinates = pixels
        self.ray_coordinates = coordinates
        self.ray_tangents = tangents
        self.screen_basis = basis
        self.valid = valid_
        self.tetrad_residual = tetrad_error
        self.null_residual = null_error
        self.plan_id = plan_id
        self.result_id = result_id
        self.metric_id = metric_id
        self.chart_id = chart_id
        self.convention_id = convention_id
        self.scale_id = scale_id
        self.coordinate_unit_id = coordinate_unit_id
        self.affine_parameter_unit_id = affine_parameter_unit_id


def initialize_gr_observer_screen(
    plan: GRObserverScreenPlan, /
) -> GRObserverScreenResult:
    """Materialize a metric-orthonormal tetrad and null pixel-ray directions."""

    if not isinstance(plan, GRObserverScreenPlan):
        raise TypeError("plan must be a GRObserverScreenPlan.")
    metric = plan.metric(plan.observer_coordinates)
    temporal_sign = float(-1 if plan.metric.convention == "mostly_plus" else 1)
    spatial_sign = -temporal_sign
    observer, observer_valid = _normalize_signed(
        plan.observer_velocity, metric, temporal_sign
    )
    sight_seed = _project(plan.line_of_sight, observer, metric)
    sight, sight_valid = _normalize_signed(sight_seed, metric, spatial_sign)
    up_seed = _project(_project(plan.screen_up, observer, metric), sight, metric)
    up, up_valid = _normalize_signed(up_seed, metric, spatial_sign)
    lowered_observer = ein.contract("ij,j->i", metric, observer)
    lowered_sight = ein.contract("ij,j->i", metric, sight)
    lowered_up = ein.contract("ij,j->i", metric, up)
    right_seed = ein.contract(
        "mabc,a,b,c->m",
        _LEVI_CIVITA_4.astype(metric.dtype),
        lowered_observer,
        lowered_sight,
        lowered_up,
    )
    right, right_valid = _normalize_signed(right_seed, metric, spatial_sign)
    tetrad = jnp.stack((observer, right, up, sight))
    expected = jnp.diag(
        jnp.asarray(
            (temporal_sign, spatial_sign, spatial_sign, spatial_sign),
            dtype=metric.dtype,
        )
    )
    gram = ein.contract("ai,ij,bj->ab", tetrad, metric, tetrad)
    tetrad_residual = jnp.max(jnp.abs(gram - expected))

    horizontal = plan.pixel_coordinates[:, 0]
    vertical = plan.pixel_coordinates[:, 1]
    spatial_seed = (
        sight[None, :]
        + horizontal[:, None] * right[None, :]
        + vertical[:, None] * up[None, :]
    )
    ray_direction, direction_valid = jax.vmap(
        lambda value: _normalize_signed(value, metric, spatial_sign)
    )(spatial_seed)
    temporal_factor = 1.0 if plan.temporal_direction == "future" else -1.0
    tangents = plan.ray_energy[:, None] * (
        temporal_factor * observer[None, :] + ray_direction
    )
    coordinates = jnp.broadcast_to(
        plan.observer_coordinates, (plan.pixel_coordinates.shape[0], 4)
    )

    first_seed = jax.vmap(lambda direction: _project(right, direction, metric))(
        ray_direction
    )
    first, first_valid = jax.vmap(
        lambda value: _normalize_signed(value, metric, spatial_sign)
    )(first_seed)
    second_seed = jax.vmap(
        lambda direction, first_axis: _project(
            _project(up, direction, metric), first_axis, metric
        )
    )(ray_direction, first)
    second, second_valid = jax.vmap(
        lambda value: _normalize_signed(value, metric, spatial_sign)
    )(second_seed)
    basis = jnp.stack((first, second), axis=1)
    null_residual = jnp.abs(
        jax.vmap(lambda tangent: _inner(metric, tangent, tangent))(tangents)
    )
    finite = (
        jnp.all(jnp.isfinite(tetrad))
        & jnp.all(jnp.isfinite(tangents), axis=-1)
        & jnp.all(jnp.isfinite(basis), axis=(-2, -1))
    )
    tetrad_valid = (
        observer_valid
        & sight_valid
        & up_valid
        & right_valid
        & jnp.isfinite(tetrad_residual)
        & (tetrad_residual <= plan.orthonormal_tolerance)
    )
    valid = (
        plan.pixel_mask
        & finite
        & tetrad_valid
        & direction_valid
        & first_valid
        & second_valid
        & (plan.ray_energy > 0.0)
        & (null_residual <= plan.orthonormal_tolerance)
    )
    return GRObserverScreenResult(
        plan.observer_coordinates,
        tetrad,
        plan.pixel_coordinates,
        coordinates,
        tangents,
        basis,
        valid,
        tetrad_residual,
        null_residual,
        plan_id=plan.plan_id,
        result_id=canonical_fingerprint(
            {"kind": "gr-observer-screen-result", "plan": plan.plan_id}
        ),
        metric_id=plan.metric_id,
        chart_id=plan.chart_id,
        convention_id=plan.convention_id,
        scale_id=plan.scale_id,
        coordinate_unit_id=plan.coordinate_unit_id,
        affine_parameter_unit_id=plan.affine_parameter_unit_id,
    )


__all__ = [
    "GRObserverScreenPlan",
    "GRObserverScreenResult",
    "GRTemporalDirection",
    "initialize_gr_observer_screen",
]
