#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail-closed finite geometry adapters for path measures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class PreparedGeometryPathKernel(StrictModule):
    """Exact interval image construction truncated at a declared image capacity."""

    lower: Array
    upper: Array
    diffusion: Array
    image_capacity: int = eqx.field(static=True)
    behavior: Literal["absorbing", "reflecting"] = eqx.field(static=True)
    geometry_class: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        behavior: Literal["absorbing", "reflecting"],
        diffusion: float = 1.0,
        image_capacity: int,
    ):
        low = jnp.asarray(lower, dtype=float)
        high = jnp.asarray(upper, dtype=float)
        if low.shape != () or high.shape != ():
            raise ValueError("The prepared image route currently supports one interval.")
        if behavior not in ("absorbing", "reflecting"):
            raise ValueError("behavior must be 'absorbing' or 'reflecting'.")
        diffusion_ = float(diffusion)
        capacity = int(image_capacity)
        if (
            not np.isfinite(float(low))
            or not np.isfinite(float(high))
            or not float(high) > float(low)
        ):
            raise ValueError("lower and upper must be finite increasing scalars.")
        if not np.isfinite(diffusion_) or diffusion_ <= 0.0 or capacity < 0:
            raise ValueError(
                "diffusion must be positive and image_capacity non-negative."
            )
        self.lower = low
        self.upper = high
        self.diffusion = jnp.asarray(diffusion_)
        self.image_capacity = capacity
        self.behavior = behavior
        self.geometry_class = "exact-affine-interval"
        self.claim = "finite-image-sum-with-analytic-tail-bound"


class GeometryKernelEstimate(StrictModule):
    value: Array
    omitted_tail_bound: Array
    mass_conserving_boundary: Array
    valid: Array
    image_capacity: int = eqx.field(static=True)
    behavior: str = eqx.field(static=True)


def interval_heat_kernel(
    prepared: PreparedGeometryPathKernel,
    x0: ArrayLike,
    x1: ArrayLike,
    time: ArrayLike,
    /,
) -> GeometryKernelEstimate:
    """Evaluate the finite Dirichlet/Neumann interval image sum and tail bound."""
    if not isinstance(prepared, PreparedGeometryPathKernel):
        raise TypeError("prepared must be PreparedGeometryPathKernel.")
    start = jnp.asarray(x0, dtype=prepared.lower.dtype)
    end = jnp.asarray(x1, dtype=prepared.lower.dtype)
    duration = jnp.asarray(time, dtype=prepared.lower.dtype)
    if start.shape != () or end.shape != () or duration.shape != ():
        raise ValueError("x0, x1, and time must be scalars.")
    length = prepared.upper - prepared.lower
    x = start - prepared.lower
    y = end - prepared.lower
    indices = jnp.arange(
        -prepared.image_capacity,
        prepared.image_capacity + 1,
        dtype=prepared.lower.dtype,
    )
    variance = 4.0 * prepared.diffusion * duration
    normalization = 1.0 / jnp.sqrt(jnp.pi * variance)
    direct = jnp.exp(-((y - x + 2.0 * indices * length) ** 2) / variance)
    mirror = jnp.exp(-((y + x + 2.0 * indices * length) ** 2) / variance)
    sign = -1.0 if prepared.behavior == "absorbing" else 1.0
    value = normalization * jnp.sum(direct + sign * mirror)
    first_omitted_distance = jnp.maximum(
        2.0 * (prepared.image_capacity + 1) * length - 2.0 * length,
        0.0,
    )
    tail = 2.0 / length * jsp.erfc(first_omitted_distance / jnp.sqrt(variance))
    inside = (
        (start >= prepared.lower)
        & (start <= prepared.upper)
        & (end >= prepared.lower)
        & (end <= prepared.upper)
    )
    valid = inside & jnp.isfinite(duration) & (duration > 0.0) & jnp.isfinite(value)
    return GeometryKernelEstimate(
        value=value,
        omitted_tail_bound=tail,
        mass_conserving_boundary=jnp.asarray(prepared.behavior == "reflecting"),
        valid=valid,
        image_capacity=prepared.image_capacity,
        behavior=prepared.behavior,
    )


class SpecularReflectionResult(StrictModule):
    velocity: Array
    normal_norm: Array
    normal_velocity: Array
    grazing: Array
    valid: Array


def specular_reflect(
    velocity: ArrayLike,
    normal: ArrayLike,
    /,
    *,
    grazing_tolerance: float = 1e-10,
    normal_tolerance: float = 1e-12,
) -> SpecularReflectionResult:
    """Reflect a kinetic velocity at a regular localized boundary."""
    v = jnp.asarray(velocity)
    n = jnp.asarray(normal, dtype=v.dtype)
    if v.ndim != 1 or n.shape != v.shape or v.size < 1:
        raise ValueError("velocity and normal must be matching nonempty vectors.")
    norm = jnp.sqrt(jnp.real(jnp.vdot(n, n)))
    unit = n / jnp.where(norm > normal_tolerance, norm, 1.0)
    normal_velocity = jnp.real(jnp.vdot(unit, v))
    grazing = jnp.abs(normal_velocity) <= grazing_tolerance
    reflected = v - 2.0 * normal_velocity * unit
    valid = (
        jnp.all(jnp.isfinite(v))
        & jnp.all(jnp.isfinite(n))
        & (norm > normal_tolerance)
        & ~grazing
    )
    return SpecularReflectionResult(
        velocity=reflected,
        normal_norm=norm,
        normal_velocity=normal_velocity,
        grazing=grazing,
        valid=valid,
    )


def killed_path_mask(boundary_values: ArrayLike, /) -> Array:
    """Keep finite non-positive interior nodes active through the boundary."""
    values = jnp.asarray(boundary_values)
    if values.ndim < 1:
        raise ValueError("boundary_values must have a trailing path-node axis.")
    inside = jnp.isfinite(values) & (values <= 0.0)
    return jnp.cumprod(inside.astype(jnp.int32), axis=-1).astype(bool)


def prepare_path_boundary_schedule(
    geometry,
    behavior: Literal["absorbing", "specular"],
    vector_field: Callable[[Array, Array, Any], Array],
    /,
    *,
    maximum_events: int,
    plan_id: str,
    grazing_tolerance: float = 1e-8,
    event_tolerance: float = 1e-10,
):
    """Build the canonical DCD schedule from one compiled GTA boundary field."""
    from ...geometry import CompiledGeometry, GeometryCapability
    from ...solver._hybrid_event import HybridEventPlan
    from ...solver._hybrid_schedule import HybridSchedulePlan, ScheduledHybridEvent

    if not isinstance(geometry, CompiledGeometry):
        raise TypeError("geometry must be CompiledGeometry.")
    if behavior not in ("absorbing", "specular"):
        raise ValueError("behavior must be 'absorbing' or 'specular'.")
    if not callable(vector_field):
        raise TypeError("vector_field must be callable.")
    geometry.require_valid()
    if behavior == "specular":
        geometry.require(GeometryCapability.BOUNDARY_NORMAL)
    dimension = geometry.ambient_dimension

    def guard(time, state, args):
        del time, args
        return jnp.asarray(geometry.boundary_field(state[:dimension])).reshape(())

    def reset(time, state, args):
        del time, args
        if behavior == "absorbing":
            return state
        if state.shape != (2 * dimension,):
            raise ValueError(
                "Specular path events require concatenated position/velocity state."
            )
        normal = geometry.boundary_normal(state[:dimension])
        reflected = specular_reflect(
            state[dimension:],
            normal,
            grazing_tolerance=grazing_tolerance,
        )
        return state.at[dimension:].set(
            jnp.where(reflected.valid, reflected.velocity, jnp.nan)
        )

    event = HybridEventPlan(
        guard,
        reset,
        vector_field,
        vector_field,
        event_kind=f"path-boundary:{behavior}",
        grazing_tolerance=grazing_tolerance,
        event_tolerance=event_tolerance,
        plan_id=plan_id,
    )
    scheduled = ScheduledHybridEvent(
        event,
        direction=1,
        priority=0,
        terminal=behavior == "absorbing",
    )
    return HybridSchedulePlan(
        (scheduled,),
        maximum_events=int(maximum_events),
    )


__all__ = [
    "GeometryKernelEstimate",
    "PreparedGeometryPathKernel",
    "SpecularReflectionResult",
    "prepare_path_boundary_schedule",
    "interval_heat_kernel",
    "killed_path_mask",
    "specular_reflect",
]
