#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class PolarEvaluation(StrictModule):
    lift: Array
    drag: Array
    moment: Array
    inside_domain: Array
    extrapolated: Array
    finite: Array
    polar_id: str = eqx.field(static=True)


class MultiAxisAirfoilPolar(StrictModule, NonTrainableState):
    angle_axis: Array
    reynolds_axis: Array
    mach_axis: Array
    flap_axis: Array
    lift_table: Array
    drag_table: Array
    moment_table: Array
    endpoint: str = eqx.field(static=True)
    polar_id: str = eqx.field(static=True)

    def __init__(
        self,
        angle_axis: ArrayLike,
        reynolds_axis: ArrayLike,
        mach_axis: ArrayLike,
        flap_axis: ArrayLike,
        lift_table: ArrayLike,
        drag_table: ArrayLike,
        moment_table: ArrayLike | None = None,
        /,
        *,
        endpoint: str = "error",
    ):
        axes = tuple(
            np.asarray(axis, dtype=float)
            for axis in (angle_axis, reynolds_axis, mach_axis, flap_axis)
        )
        if any(
            axis.ndim != 1
            or axis.size < 2
            or np.any(~np.isfinite(axis))
            or np.any(np.diff(axis) <= 0.0)
            for axis in axes
        ):
            raise ValueError("Polar axes must be finite strictly increasing vectors.")
        shape = tuple(axis.size for axis in axes)
        lift, drag = (
            np.asarray(lift_table, dtype=float),
            np.asarray(drag_table, dtype=float),
        )
        moment = (
            np.zeros(shape)
            if moment_table is None
            else np.asarray(moment_table, dtype=float)
        )
        if (
            lift.shape != shape
            or drag.shape != shape
            or moment.shape != shape
            or np.any(~np.isfinite(lift))
            or np.any(~np.isfinite(drag))
            or np.any(~np.isfinite(moment))
        ):
            raise ValueError(
                "Polar coefficient tables must match axis shape and be finite."
            )
        if endpoint not in ("error", "clamp", "linear"):
            raise ValueError("Polar endpoint policy must be error, clamp, or linear.")
        self.angle_axis, self.reynolds_axis, self.mach_axis, self.flap_axis = tuple(
            jnp.asarray(axis) for axis in axes
        )
        self.lift_table, self.drag_table, self.moment_table = (
            jnp.asarray(lift),
            jnp.asarray(drag),
            jnp.asarray(moment),
        )
        self.endpoint = endpoint
        self.polar_id = canonical_fingerprint(
            {
                "kind": "multi-axis-airfoil-polar",
                "axes": [array_tree_fingerprint(axis) for axis in axes],
                "lift": array_tree_fingerprint(lift),
                "drag": array_tree_fingerprint(drag),
                "moment": array_tree_fingerprint(moment),
                "endpoint": endpoint,
            }
        )

    def evaluate(
        self, angle: ArrayLike, reynolds: ArrayLike, mach: ArrayLike, flap: ArrayLike, /
    ) -> PolarEvaluation:
        queries = tuple(
            jnp.asarray(value, dtype=self.angle_axis.dtype)
            for value in (angle, reynolds, mach, flap)
        )
        shape = jnp.broadcast_shapes(*(query.shape for query in queries))
        queries = tuple(jnp.broadcast_to(query, shape) for query in queries)
        axes = (self.angle_axis, self.reynolds_axis, self.mach_axis, self.flap_axis)
        lower_indices, fractions, inside = [], [], jnp.ones(shape, dtype=bool)
        for axis, query in zip(axes, queries, strict=True):
            inside = inside & (query >= axis[0]) & (query <= axis[-1])
            clipped = (
                jnp.clip(query, axis[0], axis[-1]) if self.endpoint != "linear" else query
            )
            lower = jnp.clip(
                jnp.searchsorted(axis, clipped, side="right") - 1, 0, axis.size - 2
            )
            denominator = axis[lower + 1] - axis[lower]
            fraction = (clipped - axis[lower]) / denominator
            lower_indices.append(lower)
            fractions.append(fraction)
        if self.endpoint == "error":
            queries = tuple(
                eqx.error_if(
                    query,
                    jnp.any(~inside),
                    "Polar query lies outside the sampled domain.",
                )
                for query in queries
            )

        def interpolate(table):
            value = jnp.zeros(shape, dtype=table.dtype)
            for corner in itertools.product((0, 1), repeat=4):
                weight = jnp.ones(shape, dtype=table.dtype)
                indices = []
                for axis_index, upper in enumerate(corner):
                    indices.append(lower_indices[axis_index] + upper)
                    weight = weight * (
                        fractions[axis_index] if upper else 1.0 - fractions[axis_index]
                    )
                value = value + weight * table[tuple(indices)]
            return value

        lift, drag, moment = (
            interpolate(self.lift_table),
            interpolate(self.drag_table),
            interpolate(self.moment_table),
        )
        finite = (
            jnp.all(jnp.isfinite(lift))
            & jnp.all(jnp.isfinite(drag))
            & jnp.all(jnp.isfinite(moment))
        )
        return PolarEvaluation(lift, drag, moment, inside, ~inside, finite, self.polar_id)


class DynamicStallState(StrictModule):
    lagged_angle: Array
    lagged_lift: Array
    separation: Array


class DynamicStallResult(StrictModule):
    state: DynamicStallState
    lift: Array
    drag_increment: Array
    moment_increment: Array
    stable_step: Array
    finite: Array
    model_id: str = eqx.field(static=True)


class DynamicStallPlan(StrictModule, NonTrainableState):
    angle_time_scale: float = eqx.field(static=True)
    lift_time_scale: float = eqx.field(static=True)
    stall_angle: float = eqx.field(static=True)
    separation_time_scale: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        angle_time_scale: float,
        lift_time_scale: float,
        stall_angle: float,
        separation_time_scale: float,
        /,
    ):
        values = tuple(
            float(value)
            for value in (angle_time_scale, lift_time_scale, separation_time_scale)
        )
        if any(
            not math.isfinite(value) or value <= 0.0 for value in values
        ) or not math.isfinite(float(stall_angle)):
            raise ValueError("Dynamic-stall controls are invalid.")
        (
            self.angle_time_scale,
            self.lift_time_scale,
            self.stall_angle,
            self.separation_time_scale,
        ) = values[0], values[1], float(stall_angle), values[2]
        self.model_id = canonical_fingerprint(
            {
                "kind": "dynamic-stall-plan",
                "angle_time_scale": values[0],
                "lift_time_scale": values[1],
                "stall_angle": float(stall_angle),
                "separation_time_scale": values[2],
            }
        )

    def initialize(self, angle: ArrayLike, lift: ArrayLike, /) -> DynamicStallState:
        angle_, lift_ = jnp.asarray(angle), jnp.asarray(lift)
        if angle_.shape != lift_.shape:
            raise ValueError("Dynamic-stall angle/lift shapes must match.")
        return DynamicStallState(angle_, lift_, jnp.zeros_like(angle_))

    def step(
        self,
        state: DynamicStallState,
        angle: ArrayLike,
        quasi_steady_lift: ArrayLike,
        time_step: ArrayLike,
        /,
    ) -> DynamicStallResult:
        angle_, lift_, dt = (
            jnp.asarray(angle),
            jnp.asarray(quasi_steady_lift),
            jnp.asarray(time_step),
        )
        if (
            angle_.shape != state.lagged_angle.shape
            or lift_.shape != angle_.shape
            or dt.shape != ()
        ):
            raise ValueError("Dynamic-stall step shapes are invalid.")
        lagged_angle = (
            state.lagged_angle
            + dt * (angle_ - state.lagged_angle) / self.angle_time_scale
        )
        separation_equilibrium = jax_sigmoid(
            8.0 * (jnp.abs(lagged_angle) - abs(self.stall_angle))
        )
        separation = (
            state.separation
            + dt
            * (separation_equilibrium - state.separation)
            / self.separation_time_scale
        )
        target_lift = (1.0 - 0.7 * separation) * lift_
        lagged_lift = (
            state.lagged_lift
            + dt * (target_lift - state.lagged_lift) / self.lift_time_scale
        )
        drag_increment = 1.2 * separation * jnp.sin(lagged_angle) ** 2
        moment_increment = -0.25 * separation * lagged_lift
        stable = 0.25 * min(
            self.angle_time_scale, self.lift_time_scale, self.separation_time_scale
        )
        finite = jnp.all(jnp.isfinite(lagged_lift)) & jnp.isfinite(dt) & (dt > 0.0)
        return DynamicStallResult(
            DynamicStallState(lagged_angle, lagged_lift, separation),
            lagged_lift,
            drag_increment,
            moment_increment,
            jnp.asarray(stable, dtype=dt.dtype),
            finite,
            self.model_id,
        )


def jax_sigmoid(value: Array, /) -> Array:
    return 1.0 / (1.0 + jnp.exp(-value))


class CompressibilityCorrectionPlan(StrictModule, NonTrainableState):
    model: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, model: str = "prandtl-glauert", /):
        if model not in ("prandtl-glauert", "karman-tsien"):
            raise ValueError("Compressibility model is unsupported.")
        self.model = model
        self.plan_id = canonical_fingerprint(
            {"kind": "lifting-compressibility-correction", "model": model}
        )

    def apply(
        self, lift: ArrayLike, pressure: ArrayLike, mach: ArrayLike, /
    ) -> tuple[Array, Array, Array]:
        lift_, pressure_, mach_ = (
            jnp.asarray(lift),
            jnp.asarray(pressure),
            jnp.asarray(mach),
        )
        beta = jnp.sqrt(jnp.maximum(1.0 - mach_**2, jnp.finfo(lift_.dtype).tiny))
        valid = (mach_ >= 0.0) & (mach_ < 0.8)
        if self.model == "prandtl-glauert":
            corrected_lift, corrected_pressure = lift_ / beta, pressure_ / beta
        else:
            denominator = beta + 0.5 * mach_**2 / (1.0 + beta) * pressure_
            corrected_lift, corrected_pressure = (
                lift_ / denominator,
                pressure_ / denominator,
            )
        return corrected_lift, corrected_pressure, valid


__all__ = [
    "CompressibilityCorrectionPlan",
    "DynamicStallPlan",
    "DynamicStallResult",
    "DynamicStallState",
    "MultiAxisAirfoilPolar",
    "PolarEvaluation",
]
