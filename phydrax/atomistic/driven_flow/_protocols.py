#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


HomogeneousFlowKind = Literal[
    "steady-shear",
    "oscillatory-shear",
    "planar-extension",
    "uniaxial-extension",
    "biaxial-extension",
]


class HomogeneousFlowEvaluation(StrictModule):
    velocity_gradient: Array
    rate_of_strain: Array
    vorticity_tensor: Array
    accumulated_strain: Array
    instantaneous_rate: Array
    active: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class HomogeneousFlowProtocolPlan(StrictModule):
    kind: HomogeneousFlowKind = eqx.field(static=True)
    rate: float = eqx.field(static=True)
    strain_amplitude: float = eqx.field(static=True)
    angular_frequency: float = eqx.field(static=True)
    start_time: float = eqx.field(static=True)
    stop_time: float | None = eqx.field(static=True)
    flow_axis: int = eqx.field(static=True)
    gradient_axis: int = eqx.field(static=True)
    extension_axis: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: HomogeneousFlowKind,
        /,
        *,
        rate: float = 0.0,
        strain_amplitude: float = 0.0,
        angular_frequency: float = 0.0,
        start_time: float = 0.0,
        stop_time: float | None = None,
        flow_axis: int = 0,
        gradient_axis: int = 1,
        extension_axis: int = 0,
    ):
        if kind not in (
            "steady-shear",
            "oscillatory-shear",
            "planar-extension",
            "uniaxial-extension",
            "biaxial-extension",
        ):
            raise ValueError("Unknown homogeneous flow protocol.")
        rate_ = float(rate)
        amplitude = float(strain_amplitude)
        frequency = float(angular_frequency)
        start = float(start_time)
        stop = None if stop_time is None else float(stop_time)
        if any(
            not math.isfinite(value) for value in (rate_, amplitude, frequency, start)
        ):
            raise ValueError("Flow protocol controls must be finite.")
        if start < 0.0 or (
            stop is not None and (not math.isfinite(stop) or stop <= start)
        ):
            raise ValueError("Flow protocol time window is invalid.")
        if kind == "oscillatory-shear" and (amplitude <= 0.0 or frequency <= 0.0):
            raise ValueError(
                "Oscillatory shear requires positive amplitude and frequency."
            )
        if kind != "oscillatory-shear" and rate_ == 0.0:
            raise ValueError("Steady shear and extension require a nonzero rate.")
        flow = int(flow_axis)
        gradient = int(gradient_axis)
        extension = int(extension_axis)
        if (
            any(axis not in range(3) for axis in (flow, gradient, extension))
            or flow == gradient
        ):
            raise ValueError(
                "Flow axes must address distinct three-dimensional directions."
            )
        self.kind = kind
        self.rate = rate_
        self.strain_amplitude = amplitude
        self.angular_frequency = frequency
        self.start_time = start
        self.stop_time = stop
        self.flow_axis = flow
        self.gradient_axis = gradient
        self.extension_axis = extension
        self.plan_id = canonical_fingerprint(
            {
                "kind": "homogeneous-flow-protocol",
                "flow_kind": kind,
                "rate": rate_,
                "strain_amplitude": amplitude,
                "angular_frequency": frequency,
                "start_time": start,
                "stop_time": stop,
                "flow_axis": flow,
                "gradient_axis": gradient,
                "extension_axis": extension,
            }
        )

    def evaluate(self, time: ArrayLike, /) -> HomogeneousFlowEvaluation:
        value = jnp.asarray(time).reshape(())
        elapsed = jnp.maximum(value - self.start_time, 0.0)
        active = value >= self.start_time
        if self.stop_time is not None:
            active = active & (value < self.stop_time)
            elapsed = jnp.minimum(elapsed, self.stop_time - self.start_time)
        gradient = jnp.zeros((3, 3), dtype=value.dtype)
        if self.kind == "oscillatory-shear":
            instantaneous = (
                self.strain_amplitude
                * self.angular_frequency
                * jnp.cos(self.angular_frequency * elapsed)
            )
            strain = self.strain_amplitude * jnp.sin(self.angular_frequency * elapsed)
            gradient = gradient.at[self.flow_axis, self.gradient_axis].set(
                jnp.where(active, instantaneous, 0.0)
            )
        elif self.kind == "steady-shear":
            instantaneous = jnp.asarray(self.rate, dtype=value.dtype)
            strain = self.rate * elapsed
            gradient = gradient.at[self.flow_axis, self.gradient_axis].set(
                jnp.where(active, instantaneous, 0.0)
            )
        elif self.kind == "planar-extension":
            instantaneous = jnp.asarray(self.rate, dtype=value.dtype)
            strain = self.rate * elapsed
            axes = [axis for axis in range(3) if axis != self.extension_axis]
            compression_axis = axes[0]
            gradient = gradient.at[self.extension_axis, self.extension_axis].set(
                jnp.where(active, self.rate, 0.0)
            )
            gradient = gradient.at[compression_axis, compression_axis].set(
                jnp.where(active, -self.rate, 0.0)
            )
        elif self.kind == "uniaxial-extension":
            instantaneous = jnp.asarray(self.rate, dtype=value.dtype)
            strain = self.rate * elapsed
            gradient = gradient + jnp.eye(3, dtype=value.dtype) * jnp.where(
                active, -0.5 * self.rate, 0.0
            )
            gradient = gradient.at[self.extension_axis, self.extension_axis].set(
                jnp.where(active, self.rate, 0.0)
            )
        else:
            instantaneous = jnp.asarray(self.rate, dtype=value.dtype)
            strain = self.rate * elapsed
            gradient = gradient + jnp.eye(3, dtype=value.dtype) * jnp.where(
                active, self.rate, 0.0
            )
            gradient = gradient.at[self.extension_axis, self.extension_axis].set(
                jnp.where(active, -2.0 * self.rate, 0.0)
            )
        rate_of_strain = 0.5 * (gradient + gradient.T)
        vorticity = 0.5 * (gradient - gradient.T)
        finite = jnp.all(jnp.isfinite(gradient)) & jnp.isfinite(strain)
        successful = finite & (
            jnp.abs(jnp.trace(gradient)) <= 64.0 * jnp.finfo(value.dtype).eps
        )
        return HomogeneousFlowEvaluation(
            gradient,
            rate_of_strain,
            vorticity,
            strain,
            jnp.where(active, instantaneous, 0.0),
            active,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "HomogeneousFlowEvaluation",
    "HomogeneousFlowKind",
    "HomogeneousFlowProtocolPlan",
]
