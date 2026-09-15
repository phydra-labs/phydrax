#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import cast, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._numerics import log_normalize, weight_ess
from ..._strict import StrictModule
from ...integration import WeightedSampleBatch


CrossingDirection: TypeAlias = Literal["inward", "outward", "both"]


class CrossingStatus(IntEnum):
    """Numerical disposition of a weighted surface-crossing reduction."""

    SUCCESS = 0
    NO_CROSSINGS = 1
    GRAZING_ONLY = 2
    INVALID_INPUT = 3


class SurfaceCrossingDiagnostics(StrictModule):
    """Fixed-shape evidence for one weighted spherical crossing measure."""

    candidate_count: Array
    crossing_count: Array
    grazing_count: Array
    invalid_count: Array
    log_total_flux_weight: Array
    effective_sample_size: Array
    finite: Array
    successful: Array
    status: Array


class SurfaceCrossingMeasure(StrictModule):
    """Flux and induced density measures for the same surface crossings.

    ``flux`` retains each trajectory's crossing weight. ``density`` applies
    ``v_initial / abs(n dot v_crossing)``, including the obliquity factor.
    Repeated crossings retain ancestry IDs and are not asserted independent.
    """

    flux: WeightedSampleBatch
    density: WeightedSampleBatch
    initial_to_crossing_density_jacobian: Array
    initial_speeds_m_s: Array
    crossing_speeds_m_s: Array
    signed_normal_speeds_m_s: Array
    surface_residuals_m: Array
    diagnostics: SurfaceCrossingDiagnostics
    radius_m: Array
    direction: CrossingDirection = eqx.field(static=True)


class ObservationFlux(StrictModule):
    """Spherical observation-rate estimate with explicit area and exposure."""

    total_crossing_weight: Array
    area_m2: Array
    exposure_s: Array
    flux_m2_s: Array
    effective_sample_size: Array
    valid: Array


def _raw_weights(batch: WeightedSampleBatch, /) -> tuple[Array, Array, Array]:
    if not all(isinstance(axis, int) for axis in batch.sample_axes):
        raise TypeError("Surface crossings currently require raw integer sample axes.")
    axes = cast(tuple[int, ...], batch.sample_axes)
    if axes != (0,):
        raise ValueError("Surface crossings require one leading trajectory sample axis.")
    weights = jnp.asarray(batch.log_weights)
    if weights.ndim != 1:
        raise ValueError("Surface crossing path log_weights must be one-dimensional.")
    mask = (
        jnp.ones(weights.shape, dtype=bool)
        if batch.mask is None
        else jnp.asarray(batch.mask, dtype=bool)
    )
    support = (
        jnp.ones(weights.shape, dtype=bool)
        if batch.support_valid is None
        else jnp.broadcast_to(jnp.asarray(batch.support_valid, dtype=bool), weights.shape)
    )
    return weights, mask, support


def spherical_surface_crossings(
    crossing_states: ArrayLike,
    crossing_valid: ArrayLike,
    source_paths: WeightedSampleBatch,
    radius_m: ArrayLike,
    /,
    *,
    direction: CrossingDirection = "both",
    grazing_tolerance_m_s: float = 1.0e-12,
) -> SurfaceCrossingMeasure:
    """Construct unbiased weighted surface flux and density crossing measures."""
    states = jnp.asarray(crossing_states, dtype=float)
    valid = jnp.asarray(crossing_valid, dtype=bool)
    path_log_weights, path_mask, path_support = _raw_weights(source_paths)
    radius = jnp.asarray(radius_m, dtype=states.dtype).reshape(())
    if states.ndim != 3 or states.shape[-1] != 6:
        raise ValueError("crossing_states must have shape (path, event, 6).")
    if valid.shape != states.shape[:2]:
        raise ValueError("crossing_valid must match the path and event axes.")
    if states.shape[0] != path_log_weights.shape[0]:
        raise ValueError("Crossing states and source weights must share the path axis.")
    source_states = jnp.asarray(source_paths.samples, dtype=states.dtype)
    if source_states.shape != (states.shape[0], 6):
        raise ValueError("Source samples must retain one initial packed state per path.")
    if direction not in ("inward", "outward", "both"):
        raise ValueError("direction must be 'inward', 'outward', or 'both'.")
    tolerance = float(grazing_tolerance_m_s)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("grazing_tolerance_m_s must be finite and positive.")
    radius = eqx.error_if(
        radius,
        ~jnp.isfinite(radius) | (radius <= 0.0),
        "Crossing radius must be finite and positive.",
    )
    positions = states[..., :3]
    velocities = states[..., 3:]
    norms = jnp.sqrt(jnp.sum(positions * positions, axis=-1))
    surface_residual = norms - radius
    surface_tolerance = jnp.sqrt(jnp.finfo(states.dtype).eps) * jnp.maximum(radius, 1.0)
    on_surface = jnp.abs(surface_residual) <= surface_tolerance
    normals = positions / jnp.maximum(norms[..., None], jnp.finfo(states.dtype).tiny)
    signed_normal_speed = ein.contract("pei,pei->pe", velocities, normals)
    normal_speed = jnp.abs(signed_normal_speed)
    crossing_speed = jnp.sqrt(jnp.sum(velocities * velocities, axis=-1))
    initial_speed = jnp.sqrt(jnp.sum(source_states[:, 3:] ** 2, axis=-1))
    repeated_initial_speed = jnp.broadcast_to(initial_speed[:, None], valid.shape)
    direction_mask = (
        signed_normal_speed < -tolerance
        if direction == "inward"
        else signed_normal_speed > tolerance
        if direction == "outward"
        else jnp.ones(valid.shape, dtype=bool)
    )
    admissible_weight = jnp.isfinite(path_log_weights) | jnp.isneginf(path_log_weights)
    path_included = path_mask[:, None] & path_support[:, None]
    finite_state = jnp.all(jnp.isfinite(states), axis=-1)
    speed_valid = (
        jnp.isfinite(crossing_speed)
        & (crossing_speed > 0.0)
        & jnp.isfinite(repeated_initial_speed)
        & (repeated_initial_speed > 0.0)
    )
    invalid = (
        valid
        & path_included
        & (~admissible_weight[:, None] | ~finite_state | ~on_surface | ~speed_valid)
    )
    candidates = (
        valid
        & path_included
        & admissible_weight[:, None]
        & finite_state
        & on_surface
        & speed_valid
    )
    grazing = candidates & (normal_speed <= tolerance)
    selected = candidates & direction_mask & ~grazing
    repeated_log_weights = jnp.broadcast_to(path_log_weights[:, None], valid.shape)
    safe_normal_speed = jnp.maximum(normal_speed, jnp.finfo(states.dtype).tiny)
    jacobian = repeated_initial_speed / safe_normal_speed
    flux_log_weights = repeated_log_weights
    density_log_weights = repeated_log_weights + jnp.log(jacobian)

    def repeat_identifier(value, default=None):
        if value is None:
            if default is None:
                return None
            path_values = default
        else:
            path_values = jnp.asarray(value, dtype=jnp.int32).reshape((states.shape[0],))
        return jnp.broadcast_to(path_values[:, None], valid.shape).reshape((-1,))

    ancestry_ids = repeat_identifier(
        source_paths.ancestry_ids,
        jnp.arange(states.shape[0], dtype=jnp.int32),
    )
    stratum_ids = repeat_identifier(source_paths.stratum_ids)
    pair_ids = repeat_identifier(source_paths.pair_ids)
    replicate_ids = repeat_identifier(source_paths.replicate_ids)
    flattened_states = states.reshape((-1, 6))
    flattened_mask = selected.reshape((-1,))
    flux = WeightedSampleBatch(
        flattened_states,
        flux_log_weights.reshape((-1,)),
        mask=flattened_mask,
        ancestry_ids=ancestry_ids,
        stratum_ids=stratum_ids,
        pair_ids=pair_ids,
        replicate_ids=replicate_ids,
        sample_axes=0,
        provenance=f"{source_paths.provenance}:spherical-flux:{direction}",
        independent=False,
    )
    density = WeightedSampleBatch(
        flattened_states,
        density_log_weights.reshape((-1,)),
        mask=flattened_mask,
        ancestry_ids=ancestry_ids,
        stratum_ids=stratum_ids,
        pair_ids=pair_ids,
        replicate_ids=replicate_ids,
        sample_axes=0,
        provenance=f"{source_paths.provenance}:spherical-density:{direction}",
        independent=False,
    )
    normalized, log_total, weight_valid = log_normalize(
        flux_log_weights.reshape((-1,)),
        axes=0,
        mask=flattened_mask,
    )
    effective = jnp.where(weight_valid, weight_ess(normalized, axis=0), 0.0)
    candidate_count = jnp.sum(candidates, dtype=jnp.int32)
    selected_count = jnp.sum(selected, dtype=jnp.int32)
    grazing_count = jnp.sum(grazing, dtype=jnp.int32)
    invalid_count = jnp.sum(invalid, dtype=jnp.int32) + jnp.sum(
        path_mask & ~path_support, dtype=jnp.int32
    )
    input_finite = (invalid_count == 0) & jnp.all(~selected | jnp.isfinite(normal_speed))
    finite = input_finite & ((selected_count == 0) | weight_valid)
    status = jnp.where(
        ~finite,
        int(CrossingStatus.INVALID_INPUT),
        jnp.where(
            selected_count > 0,
            int(CrossingStatus.SUCCESS),
            jnp.where(
                grazing_count > 0,
                int(CrossingStatus.GRAZING_ONLY),
                int(CrossingStatus.NO_CROSSINGS),
            ),
        ),
    ).astype(jnp.int32)
    diagnostics = SurfaceCrossingDiagnostics(
        candidate_count,
        selected_count,
        grazing_count,
        invalid_count,
        log_total,
        effective,
        finite,
        status == int(CrossingStatus.SUCCESS),
        status,
    )
    return SurfaceCrossingMeasure(
        flux,
        density,
        jacobian.reshape((-1,)),
        repeated_initial_speed.reshape((-1,)),
        crossing_speed.reshape((-1,)),
        signed_normal_speed.reshape((-1,)),
        surface_residual.reshape((-1,)),
        diagnostics,
        radius,
        direction,
    )


def observation_flux(
    crossings: SurfaceCrossingMeasure,
    exposure_s: ArrayLike,
    /,
) -> ObservationFlux:
    """Reduce crossing weights to an all-sky flux at the declared sphere."""
    if not isinstance(crossings, SurfaceCrossingMeasure):
        raise TypeError("crossings must be a SurfaceCrossingMeasure.")
    exposure = jnp.asarray(exposure_s, dtype=crossings.radius_m.dtype).reshape(())
    exposure = eqx.error_if(
        exposure,
        ~jnp.isfinite(exposure) | (exposure <= 0.0),
        "exposure_s must be finite and positive.",
    )
    log_total = crossings.diagnostics.log_total_flux_weight
    total = jnp.where(jnp.isfinite(log_total), jnp.exp(log_total), 0.0)
    area = 4.0 * jnp.pi * crossings.radius_m**2
    flux = total / (area * exposure)
    valid = crossings.diagnostics.successful & jnp.isfinite(flux)
    return ObservationFlux(
        total,
        area,
        exposure,
        flux,
        crossings.diagnostics.effective_sample_size,
        valid,
    )


__all__ = [
    "CrossingStatus",
    "CrossingDirection",
    "ObservationFlux",
    "SurfaceCrossingDiagnostics",
    "SurfaceCrossingMeasure",
    "observation_flux",
    "spherical_surface_crossings",
]
