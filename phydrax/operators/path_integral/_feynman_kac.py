#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ...discretization import TemporalMesh
from ._action import _paths_array, potential_action
from ._diffusion import DiffusionLike, DriftLike, sample_diffusion_paths
from ._estimate import _estimate_positive_log_weights, PathIntegralEstimate
from ._potential import _as_point_time_callable, PotentialLike


class SourceFeynmanKacEstimate(eqx.Module):
    """Terminal/source decomposition on one shared finite path population."""

    estimate: PathIntegralEstimate
    terminal_term: Array
    source_term: Array
    source_quadrature: str = eqx.field(static=True)
    temporal_error: Array
    quadrature_refinement_difference: Array
    boundary_event_error: Array
    claim: str = eqx.field(static=True)


class AdaptiveFeynmanKacEstimate(eqx.Module):
    """Feynman-Kac estimate retaining canonical adaptive path/replay evidence."""

    source_estimate: SourceFeynmanKacEstimate
    path_valid: Array
    path_status: Array
    accepted_steps: Array
    rejected_steps: Array
    temporal_evidence: Any
    event_mask: Any
    realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    valid: Array
    claim: str = eqx.field(static=True)


def _point_values(
    function: PotentialLike,
    positions: Array,
    times: Array,
    /,
    *,
    position_var: str,
    time_var: str,
    key: Key[Array, ""],
    role: str,
) -> Array:
    state_dim = int(positions.shape[-1])
    flat_positions = jnp.reshape(positions, (-1, state_dim))
    flat_times = jnp.reshape(
        jnp.broadcast_to(times, positions.shape[:-1]),
        (-1,),
    )
    function_ = _as_point_time_callable(
        function,
        position_var=position_var,
        time_var=time_var,
        key=key,
        role=role,
    )
    values = jnp.asarray(jax.vmap(function_)(flat_positions, flat_times))
    if values.shape != flat_times.shape:
        raise ValueError(f"{role} must return one scalar per point.")
    if jnp.iscomplexobj(values):
        raise TypeError(f"{role} values must be real.")
    values = eqx.error_if(
        values,
        ~jnp.all(jnp.isfinite(values)),
        f"{role} values must be finite.",
    )
    return jnp.reshape(values, positions.shape[:-1])


def _terminal_values(
    terminal: PotentialLike,
    paths: Array,
    /,
    *,
    slicing: TemporalMesh,
    position_var: str,
    time_var: str,
    key: Key[Array, ""],
) -> Array:
    endpoint = paths[..., -1, :]
    state_dim = int(endpoint.shape[-1])
    flat_endpoint = jnp.reshape(endpoint, (-1, state_dim))
    terminal_fn = _as_point_time_callable(
        terminal,
        position_var=position_var,
        time_var=time_var,
        key=key,
        role="Terminal function",
    )
    values = jnp.asarray(
        jax.vmap(terminal_fn, in_axes=(0, None))(flat_endpoint, slicing.t1)
    )
    expected_shape = (flat_endpoint.shape[0],)
    if values.shape != expected_shape:
        raise ValueError(
            "terminal must return one real scalar per endpoint; "
            f"got {values.shape}, expected {expected_shape}."
        )
    if jnp.iscomplexobj(values):
        raise TypeError("Feynman-Kac terminal values must be real.")
    values = eqx.error_if(
        values,
        ~jnp.all(jnp.isfinite(values)),
        "Feynman-Kac terminal values must be finite.",
    )
    return jnp.reshape(values, endpoint.shape[:-1])


def feynman_kac_from_paths(
    terminal: PotentialLike,
    paths: ArrayLike,
    /,
    *,
    slicing: TemporalMesh,
    killing: PotentialLike | None = None,
    source: PotentialLike | None = None,
    source_quadrature: Literal["left", "trapezoid", "midpoint"] = "midpoint",
    position_var: str = "x",
    time_var: str = "t",
    key: Key[Array, ""] = DOC_KEY0,
) -> PathIntegralEstimate:
    r"""Estimate a terminal Feynman--Kac expectation from supplied paths.

    The estimator is
    $\mathbb E[g(X_{t_1},t_1)\exp(-\int_{t_0}^{t_1}c(X_s,s)\,ds)]$.
    Midpoint quadrature is used for the optional killing rate ``c``.
    """
    if source is not None:
        return source_feynman_kac_from_paths(
            terminal,
            source,
            paths,
            slicing=slicing,
            killing=killing,
            source_quadrature=source_quadrature,
            position_var=position_var,
            time_var=time_var,
            key=key,
        ).estimate
    q = _paths_array(paths, slicing)
    count = int(q.shape[-3])
    if count < 1:
        raise ValueError("paths must contain at least one path.")
    terminal_key, killing_key = jr.split(key)
    terminal_value = _terminal_values(
        terminal,
        q,
        slicing=slicing,
        position_var=position_var,
        time_var=time_var,
        key=terminal_key,
    )
    if killing is None:
        log_weights = jnp.zeros(q.shape[:-2], dtype=q.dtype)
    else:
        log_weights = -potential_action(
            q,
            killing,
            slicing=slicing,
            position_var=position_var,
            time_var=time_var,
            key=killing_key,
        )

    max_log_weight = jnp.max(log_weights, axis=-1, keepdims=True)
    scaled_weight = jnp.exp(log_weights - max_log_weight)
    scaled_samples = terminal_value * scaled_weight
    scale = jnp.exp(jnp.squeeze(max_log_weight, axis=-1))
    value = scale * jnp.mean(scaled_samples, axis=-1)
    if count == 1:
        standard_error = jnp.full_like(value, jnp.nan)
    else:
        centered = scaled_samples - jnp.mean(
            scaled_samples,
            axis=-1,
            keepdims=True,
        )
        sample_variance = jnp.sum(centered * centered, axis=-1) / float(count - 1)
        standard_error = scale * jnp.sqrt(sample_variance / float(count))

    weight_estimate = _estimate_positive_log_weights(
        log_weights,
        scale=jnp.ones_like(value),
    )
    return PathIntegralEstimate(
        value=value,
        standard_error=standard_error,
        effective_sample_size=weight_estimate.effective_sample_size,
        log_mean_weight=weight_estimate.log_mean_weight,
        num_paths=count,
    )


def source_feynman_kac_from_paths(
    terminal: PotentialLike,
    source: PotentialLike,
    paths: ArrayLike,
    /,
    *,
    slicing: TemporalMesh,
    killing: PotentialLike | None = None,
    source_quadrature: Literal["left", "trapezoid", "midpoint"] = "midpoint",
    temporal_error: ArrayLike = jnp.nan,
    boundary_event_error: ArrayLike = jnp.nan,
    position_var: str = "x",
    time_var: str = "t",
    key: Key[Array, ""] = DOC_KEY0,
) -> SourceFeynmanKacEstimate:
    """Estimate terminal plus Duhamel terms on the same fixed path population."""
    if source_quadrature not in ("left", "trapezoid", "midpoint"):
        raise ValueError("source_quadrature must be 'left', 'trapezoid', or 'midpoint'.")
    q = _paths_array(paths, slicing)
    count = int(q.shape[-3])
    if count < 1:
        raise ValueError("paths must contain at least one path.")
    terminal_key, source_key, killing_key, refinement_key = jr.split(key, 4)
    terminal_values = _terminal_values(
        terminal,
        q,
        slicing=slicing,
        position_var=position_var,
        time_var=time_var,
        key=terminal_key,
    )
    midpoint_positions = 0.5 * (q[..., :-1, :] + q[..., 1:, :])
    if killing is None:
        killing_values = jnp.zeros(q.shape[:-2] + (slicing.num_steps,), dtype=q.dtype)
    else:
        killing_values = _point_values(
            killing,
            midpoint_positions,
            slicing.midpoints,
            position_var=position_var,
            time_var=time_var,
            key=killing_key,
            role="Killing rate",
        )
    interval_killing = killing_values * slicing.widths
    cumulative_before = jnp.concatenate(
        (
            jnp.zeros(interval_killing.shape[:-1] + (1,), dtype=interval_killing.dtype),
            jnp.cumsum(interval_killing, axis=-1)[..., :-1],
        ),
        axis=-1,
    )
    cumulative_nodes = jnp.concatenate(
        (
            jnp.zeros(interval_killing.shape[:-1] + (1,), dtype=interval_killing.dtype),
            jnp.cumsum(interval_killing, axis=-1),
        ),
        axis=-1,
    )
    terminal_samples = terminal_values * jnp.exp(-cumulative_nodes[..., -1])

    source_midpoint = _point_values(
        source,
        midpoint_positions,
        slicing.midpoints,
        position_var=position_var,
        time_var=time_var,
        key=source_key,
        role="Feynman-Kac source",
    )
    midpoint_samples = jnp.sum(
        slicing.widths
        * jnp.exp(-(cumulative_before + 0.5 * interval_killing))
        * source_midpoint,
        axis=-1,
    )
    source_nodes = _point_values(
        source,
        q,
        slicing.times,
        position_var=position_var,
        time_var=time_var,
        key=refinement_key,
        role="Feynman-Kac source",
    )
    weighted_nodes = jnp.exp(-cumulative_nodes) * source_nodes
    left_samples = jnp.sum(
        slicing.widths * weighted_nodes[..., :-1],
        axis=-1,
    )
    trapezoid_samples = jnp.sum(
        0.5 * slicing.widths * (weighted_nodes[..., :-1] + weighted_nodes[..., 1:]),
        axis=-1,
    )
    if source_quadrature == "left":
        source_samples = left_samples
        paired_refinement = left_samples - midpoint_samples
    elif source_quadrature == "trapezoid":
        source_samples = trapezoid_samples
        paired_refinement = trapezoid_samples - midpoint_samples
    else:
        source_samples = midpoint_samples
        paired_refinement = midpoint_samples - trapezoid_samples
    samples = terminal_samples + source_samples
    value = jnp.mean(samples, axis=-1)
    if count == 1:
        standard_error = jnp.full_like(value, jnp.nan)
    else:
        centered = samples - jnp.mean(samples, axis=-1, keepdims=True)
        variance = jnp.sum(centered * centered, axis=-1) / float(count - 1)
        standard_error = jnp.sqrt(variance / float(count))
    terminal_log_weight = -cumulative_nodes[..., -1]
    weight_estimate = _estimate_positive_log_weights(
        terminal_log_weight,
        scale=jnp.ones_like(value),
    )
    estimate = PathIntegralEstimate(
        value=value,
        standard_error=standard_error,
        effective_sample_size=weight_estimate.effective_sample_size,
        log_mean_weight=weight_estimate.log_mean_weight,
        num_paths=count,
    )
    return SourceFeynmanKacEstimate(
        estimate=estimate,
        terminal_term=jnp.mean(terminal_samples, axis=-1),
        source_term=jnp.mean(source_samples, axis=-1),
        source_quadrature=source_quadrature,
        temporal_error=jnp.asarray(temporal_error),
        quadrature_refinement_difference=jnp.mean(paired_refinement, axis=-1),
        boundary_event_error=jnp.asarray(boundary_event_error),
        claim="fixed-output-mesh-source-feynman-kac",
    )


def _common_stochastic_time_axis(times: ArrayLike, path_count: int, /) -> np.ndarray:
    values = np.asarray(times)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError(
            "Stochastic path times must have a trailing axis of at least two nodes."
        )
    if values.ndim == 1:
        common = values
    else:
        rows = values.reshape((-1, values.shape[-1]))
        if rows.shape[0] != path_count:
            raise ValueError(
                "Stochastic path time rows must match the declared path count."
            )
        common = rows[0]
        if not np.array_equal(rows, np.broadcast_to(common, rows.shape)):
            raise ValueError("Every stochastic path must use one common time axis.")
    return common


def source_feynman_kac_from_stochastic_paths(
    result,
    terminal: PotentialLike,
    source: PotentialLike,
    /,
    *,
    killing: PotentialLike | None = None,
    source_quadrature: Literal["left", "trapezoid", "midpoint"] = "midpoint",
    boundary_event_error: ArrayLike = jnp.nan,
    position_var: str = "x",
    time_var: str = "t",
    key: Key[Array, ""] = DOC_KEY0,
) -> AdaptiveFeynmanKacEstimate:
    """Consume the canonical SST fixed-output path ensemble without reintegration."""
    from ...stochastic._path_ensemble import StochasticPathEnsembleResult

    if not isinstance(result, StochasticPathEnsembleResult):
        raise TypeError("result must be StochasticPathEnsembleResult.")
    slicing = TemporalMesh(
        _common_stochastic_time_axis(result.times, result.path_count),
        role="internal",
        mesh_id=f"feynman-kac:{result.plan_id}",
    )
    estimate = source_feynman_kac_from_paths(
        terminal,
        source,
        result.states,
        slicing=slicing,
        killing=killing,
        source_quadrature=source_quadrature,
        boundary_event_error=boundary_event_error,
        position_var=position_var,
        time_var=time_var,
        key=key,
    )
    return AdaptiveFeynmanKacEstimate(
        source_estimate=estimate,
        path_valid=result.path_valid,
        path_status=result.status,
        accepted_steps=result.accepted_steps,
        rejected_steps=result.rejected_steps,
        temporal_evidence=result.temporal_evidence,
        event_mask=result.event_mask,
        realization_id=result.realization_id,
        coupling_id=result.coupling_id,
        valid=result.valid & jnp.isfinite(estimate.estimate.value),
        claim="canonical-adaptive-path-ensemble-fixed-output-evidence",
    )


def feynman_kac_expectation(
    terminal: PotentialLike,
    drift: DriftLike,
    diffusion: DiffusionLike,
    x0: ArrayLike,
    /,
    *,
    slicing: TemporalMesh,
    num_paths: int,
    killing: PotentialLike | None = None,
    position_var: str = "x",
    time_var: str = "t",
    key: Key[Array, ""] = DOC_KEY0,
) -> PathIntegralEstimate:
    """Simulate diffusion paths and estimate a terminal Feynman--Kac value."""
    path_key, estimate_key = jr.split(key)
    paths = sample_diffusion_paths(
        drift,
        diffusion,
        x0,
        slicing=slicing,
        num_paths=num_paths,
        position_var=position_var,
        time_var=time_var,
        key=path_key,
    )
    return feynman_kac_from_paths(
        terminal,
        paths,
        slicing=slicing,
        killing=killing,
        position_var=position_var,
        time_var=time_var,
        key=estimate_key,
    )


__all__ = [
    "feynman_kac_expectation",
    "AdaptiveFeynmanKacEstimate",
    "SourceFeynmanKacEstimate",
    "source_feynman_kac_from_paths",
    "source_feynman_kac_from_stochastic_paths",
    "feynman_kac_from_paths",
]
