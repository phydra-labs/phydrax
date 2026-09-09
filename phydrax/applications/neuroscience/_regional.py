#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Regional rate/oscillator dynamics; all time derivatives use seconds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...ein import contract
from ...solver import DelayHistoryWindow


def _parameter(value: ArrayLike, name: str, /, *, positive: bool = False) -> Array:
    raw = jnp.asarray(value)
    if raw.ndim > 1 or jnp.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real scalar or region vector.")
    array = raw.astype(jnp.result_type(raw.dtype, float))
    bad = ~jnp.all(jnp.isfinite(array))
    if positive:
        bad = bad | jnp.any(array <= 0.0)
    return eqx.error_if(
        array, bad, f"{name} must be finite" + (" and positive." if positive else ".")
    )


class RegionalConnectivity(StrictModule):
    """Host-prepared directed connectivity in the exact declared region order.

    ``weights[target, source]`` are dimensionless. ``delay_unit`` is either
    seconds (``s``) or milliseconds (``ms``); stored delays are seconds.
    ``incoming_abs`` divides each incoming row by its absolute-weight sum at
    preparation, leaving isolated rows zero. No other normalization is implicit.

    Preparation fixes the zero/positive delay pattern, not the weights. Even a
    zero-weight entry with a positive declared delay remains a delayed route,
    so differentiating a weight through zero does not change causal topology.
    Prepare outside JIT; numerical model parameters and weights remain leaves.
    """

    region_ids: tuple[str, ...] = eqx.field(static=True)
    weights: Array
    delays_s: Array
    delayed_targets: Array
    delayed_sources: Array
    positive_delays_s: Array
    minimum_delay_s: float = eqx.field(static=True)
    maximum_delay_s: float = eqx.field(static=True)
    propagation_lags_s: tuple[float, ...] = eqx.field(static=True)
    normalization: Literal["none", "incoming_abs"] = eqx.field(static=True)

    def __init__(
        self,
        region_ids: Sequence[str],
        weights: ArrayLike,
        delays: ArrayLike,
        /,
        *,
        delay_unit: Literal["s", "ms"] = "s",
        normalization: Literal["none", "incoming_abs"] = "none",
    ):
        labels = tuple(region_ids)
        if (
            not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("region_ids must be nonempty, unique string identifiers.")
        if delay_unit not in ("s", "ms"):
            raise ValueError("delay_unit must be 's' or 'ms'.")
        if normalization not in ("none", "incoming_abs"):
            raise ValueError("normalization must be 'none' or 'incoming_abs'.")
        raw_weights = np.asarray(weights)
        raw_delays = np.asarray(delays)
        shape = (len(labels), len(labels))
        if raw_weights.shape != shape or raw_delays.shape != shape:
            raise ValueError("weights and delays must have shape [target, source].")
        if np.iscomplexobj(raw_weights) or np.iscomplexobj(raw_delays):
            raise ValueError("Regional weights and delays must be real.")
        if not np.all(np.isfinite(raw_weights)):
            raise ValueError("Regional weights must be finite.")
        if not np.all(np.isfinite(raw_delays)) or np.any(raw_delays < 0.0):
            raise ValueError("Physical delays must be finite and nonnegative.")
        delays_s = raw_delays.astype(float) * (0.001 if delay_unit == "ms" else 1.0)
        target, source = np.nonzero(delays_s > 0.0)
        positive = delays_s[target, source]
        weight_array = jnp.asarray(raw_weights, dtype=float)
        if normalization == "incoming_abs":
            denominator = jnp.sum(jnp.abs(weight_array), axis=1, keepdims=True)
            weight_array = weight_array / jnp.where(denominator > 0.0, denominator, 1.0)
        self.region_ids = labels
        self.weights = weight_array
        self.delays_s = jnp.asarray(delays_s)
        self.delayed_targets = jnp.asarray(target, dtype=jnp.int32)
        self.delayed_sources = jnp.asarray(source, dtype=jnp.int32)
        self.positive_delays_s = jnp.asarray(positive)
        self.minimum_delay_s = float(positive.min()) if positive.size else 0.0
        self.maximum_delay_s = float(positive.max()) if positive.size else 0.0
        self.propagation_lags_s = tuple(float(value) for value in np.unique(positive))
        self.normalization = normalization

    @property
    def region_count(self) -> int:
        return len(self.region_ids)

    @property
    def delayed_edge_count(self) -> int:
        return int(self.positive_delays_s.size)


def _delayed_coupling(
    connectivity: RegionalConnectivity,
    state: Array,
    history: DelayHistoryWindow,
    /,
) -> Array:
    """Reduce scalar-lag queries without an [edge, region, state] temporary.

    Checkpointing the loop body also avoids retaining every full interpolated
    state in its transpose. The native history is the only history owner.
    """

    def add_edge(index, accumulated):
        target = connectivity.delayed_targets[index]
        source = connectivity.delayed_sources[index]
        past = history.value(connectivity.positive_delays_s[index])
        weighted = connectivity.weights[target, source] * past[source, :2]
        return accumulated.at[target].add(weighted)

    return jax.lax.fori_loop(
        0,
        connectivity.delayed_edge_count,
        jax.checkpoint(add_edge),
        jnp.zeros((connectivity.region_count, 2), dtype=state.dtype),
    )


def regional_coupling(
    connectivity: RegionalConnectivity,
    state: ArrayLike,
    history: DelayHistoryWindow | None = None,
    /,
) -> Array:
    """Incoming two-coordinate sum, with exact zero/current and positive/past routing.

    ``state`` has shape ``[region, 2]`` or the joint neural/BOLD shape
    ``[region, 6]``. Positive physical delays require an explicit native history
    window; they are never treated as instantaneous or rounded to a clock grid.
    """
    values = jnp.asarray(state)
    if (
        values.ndim != 2
        or values.shape[0] != connectivity.region_count
        or values.shape[1] not in (2, 6)
    ):
        raise ValueError("Regional state must have shape [region, 2] or [region, 6].")
    instantaneous = jnp.where(connectivity.delays_s == 0.0, connectivity.weights, 0.0)
    incoming = contract("ij,jc->ic", instantaneous, values[:, :2])
    if connectivity.delayed_edge_count:
        if history is None:
            raise ValueError(
                "Positive physical delays require an explicit history window."
            )
        incoming = incoming + _delayed_coupling(connectivity, values, history)
    return incoming


class WilsonCowan(StrictModule):
    """Excitatory/inhibitory fractions with logistic recruitment and saturation.

    For ``z=(E,I)``, ``tau_E E' = -E + (1-E) sigmoid(a_E h_E)`` and
    ``tau_I I' = -I + (1-I) sigmoid(a_I h_I)``. ``h_E`` receives the delayed
    incoming E sum times ``coupling_gain``; ``h_I`` has local coupling only.
    Biases, thresholds, local weights and external inputs are dimensionless.
    Time constants are seconds, and all parameters may be scalar or [region].
    """

    tau_e_s: Array
    tau_i_s: Array
    c_ee: Array
    c_ei: Array
    c_ie: Array
    c_ii: Array
    slope_e: Array
    slope_i: Array
    threshold_e: Array
    threshold_i: Array
    baseline_e: Array
    baseline_i: Array
    coupling_gain: Array

    def __init__(
        self,
        *,
        tau_e_s: ArrayLike = 0.02,
        tau_i_s: ArrayLike = 0.01,
        c_ee: ArrayLike = 12.0,
        c_ei: ArrayLike = 10.0,
        c_ie: ArrayLike = 10.0,
        c_ii: ArrayLike = 0.0,
        slope_e: ArrayLike = 1.0,
        slope_i: ArrayLike = 1.0,
        threshold_e: ArrayLike = 2.0,
        threshold_i: ArrayLike = 3.0,
        baseline_e: ArrayLike = 0.0,
        baseline_i: ArrayLike = 0.0,
        coupling_gain: ArrayLike = 1.0,
    ):
        self.tau_e_s = _parameter(tau_e_s, "tau_e_s", positive=True)
        self.tau_i_s = _parameter(tau_i_s, "tau_i_s", positive=True)
        self.c_ee = _parameter(c_ee, "c_ee")
        self.c_ei = _parameter(c_ei, "c_ei")
        self.c_ie = _parameter(c_ie, "c_ie")
        self.c_ii = _parameter(c_ii, "c_ii")
        self.slope_e = _parameter(slope_e, "slope_e", positive=True)
        self.slope_i = _parameter(slope_i, "slope_i", positive=True)
        self.threshold_e = _parameter(threshold_e, "threshold_e")
        self.threshold_i = _parameter(threshold_i, "threshold_i")
        self.baseline_e = _parameter(baseline_e, "baseline_e")
        self.baseline_i = _parameter(baseline_i, "baseline_i")
        self.coupling_gain = _parameter(coupling_gain, "coupling_gain")

    def __call__(
        self, state: Array, incoming: Array, external: Array, incoming_weight: Array, /
    ) -> Array:
        del incoming_weight
        excitatory, inhibitory = state[:, 0], state[:, 1]
        h_e = (
            self.c_ee * excitatory
            - self.c_ei * inhibitory
            + self.baseline_e
            - self.threshold_e
            + self.coupling_gain * incoming[:, 0]
            + external[:, 0]
        )
        h_i = (
            self.c_ie * excitatory
            - self.c_ii * inhibitory
            + self.baseline_i
            - self.threshold_i
            + external[:, 1]
        )
        d_e = (
            -excitatory + (1.0 - excitatory) * jax.nn.sigmoid(self.slope_e * h_e)
        ) / self.tau_e_s
        d_i = (
            -inhibitory + (1.0 - inhibitory) * jax.nn.sigmoid(self.slope_i * h_i)
        ) / self.tau_i_s
        return jnp.stack((d_e, d_i), axis=-1)


class Hopf(StrictModule):
    """Supercritical Hopf normal form with directed diffusive coupling.

    ``z' = (a - cubic*|z|²)z + i*(2*pi*frequency_hz)z + G*C + drive``;
    ``C_i = sum_j W[i,j]*(z_j(t-delay[i,j])-z_i(t))``. ``a``, ``cubic`` and
    ``G`` use inverse seconds for dimensionless z. External drive is z/second.
    Negative frequencies reverse rotation; no frequency normalization is hidden.
    """

    a_per_s: Array
    frequency_hz: Array
    coupling_per_s: Array
    cubic_per_s: Array

    def __init__(
        self,
        *,
        a_per_s: ArrayLike = -0.1,
        frequency_hz: ArrayLike = 0.04,
        coupling_per_s: ArrayLike = 0.1,
        cubic_per_s: ArrayLike = 1.0,
    ):
        self.a_per_s = _parameter(a_per_s, "a_per_s")
        self.frequency_hz = _parameter(frequency_hz, "frequency_hz")
        self.coupling_per_s = _parameter(coupling_per_s, "coupling_per_s")
        self.cubic_per_s = _parameter(cubic_per_s, "cubic_per_s", positive=True)

    def __call__(
        self, state: Array, incoming: Array, external: Array, incoming_weight: Array, /
    ) -> Array:
        x, y = state[:, 0], state[:, 1]
        radial_rate = self.a_per_s - self.cubic_per_s * (x * x + y * y)
        omega = 2.0 * jnp.pi * self.frequency_hz
        local = jnp.stack(
            (radial_rate * x - omega * y, radial_rate * y + omega * x), axis=-1
        )
        coupling = incoming - incoming_weight[:, None] * state
        return local + self.coupling_per_s[..., None] * coupling + external


__all__ = ["Hopf", "RegionalConnectivity", "WilsonCowan", "regional_coupling"]
