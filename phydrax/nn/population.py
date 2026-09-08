#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed physical-LIF population codes backed by native weighted SVD.

Construction is host-side and stochastic construction requires an explicit key.
Rate evaluation, decoder application, assessment, and causal filtering are pure
JAX operations. Currents are nA, physical times are ms, and activities are Hz.
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._numerics import solve_weighted_least_squares, WeightedLeastSquaresResult
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..domain import HyperRectangle
from ..ein import contract


if TYPE_CHECKING:
    from ..applications.electrophysiology._neurons import LeakyIntegrateAndFire


Target = Callable[[Array], Array] | ArrayLike | None


def lif_rate_response(
    neuron: LeakyIntegrateAndFire, injected_current_nA: ArrayLike, /
) -> Array:
    """Steady firing rate in Hz for the physical reset/refractory LIF model.

    A suprathreshold neuron charges from reset to threshold for
    ``C/g * log1p(g * (threshold-reset) / (I-I_threshold))`` ms, then
    spends ``refractory_ms`` at reset. The rheobase and subthreshold rates are
    exactly zero. This is not the timestep-dependent artificial LIF response.
    """
    from ..applications.electrophysiology._neurons import LeakyIntegrateAndFire

    if not isinstance(neuron, LeakyIntegrateAndFire):
        raise TypeError("Rate response requires a physical LeakyIntegrateAndFire.")
    current = jnp.asarray(injected_current_nA)
    if jnp.iscomplexobj(current):
        raise TypeError("Physical currents must be real-valued.")
    threshold_current = neuron.leak_conductance_uS * (
        neuron.threshold_mV - neuron.resting_mV
    )
    excess = current - threshold_current
    firing = excess > 0.0
    safe_excess = jnp.where(firing, excess, 1.0)
    span = neuron.threshold_mV - neuron.reset_mV
    ratio = neuron.leak_conductance_uS * span / safe_excess
    small = ratio < jnp.sqrt(jnp.finfo(jnp.result_type(ratio, float)).eps)
    safe_g = jnp.where(neuron.leak_conductance_uS > 0.0, neuron.leak_conductance_uS, 1.0)
    regular = neuron.capacitance_nF / safe_g * jnp.log1p(ratio)
    small_ratio = jnp.where(small, ratio, 0.0)
    limit = neuron.capacitance_nF * span / jnp.where(small, safe_excess, 1.0)
    limit = limit * (1.0 - small_ratio / 2.0 + small_ratio**2 / 3.0)
    charge_ms = jnp.where(small, limit, regular)
    rate = jnp.where(firing, 1000.0 / (neuron.refractory_ms + charge_ms), 0.0)
    return jnp.where(jnp.isnan(current), jnp.nan, rate)


def _host_vector(value: ArrayLike, size: int, name: str, /) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise TypeError(f"{name} must be real-valued.")
    result = np.broadcast_to(raw, (size,)).astype(float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result


def _validate_neuron(neuron: LeakyIntegrateAndFire, count: int, /) -> None:
    # Applications load after nn; importing the physical owner at construction
    # avoids reversing the package initialization dependency.
    from ..applications.electrophysiology._neurons import LeakyIntegrateAndFire

    if not isinstance(neuron, LeakyIntegrateAndFire):
        raise TypeError("Population codes require a physical LeakyIntegrateAndFire.")
    capacitance = _host_vector(neuron.capacitance_nF, count, "capacitance_nF")
    conductance = _host_vector(neuron.leak_conductance_uS, count, "leak_conductance_uS")
    threshold = _host_vector(neuron.threshold_mV, count, "threshold_mV")
    reset = _host_vector(neuron.reset_mV, count, "reset_mV")
    refractory = _host_vector(neuron.refractory_ms, count, "refractory_ms")
    _host_vector(neuron.resting_mV, count, "resting_mV")
    if np.any(capacitance <= 0.0) or np.any(conductance < 0.0):
        raise ValueError(
            "Population LIF needs positive capacitance and nonnegative leak."
        )
    if np.any(reset >= threshold) or np.any(refractory < 0.0):
        raise ValueError("Population LIF requires reset < threshold and refractory >= 0.")


class LIFPopulation(StrictModule, NonTrainableState):
    """Frozen physical neurons and affine current encoding on a native box domain.

    Coordinates are mapped componentwise from the domain to [-1, 1]. Each
    Euclidean-unit encoder's projection is divided by its L1 norm, so its range
    over the entire box is [-1, 1], including corners in multiple dimensions.
    Gain and bias have nA units. Arrays may be constructed explicitly to inspect
    silent/duplicate neurons; fitting diagnoses rather than deletes them.
    """

    domain: HyperRectangle
    neuron: LeakyIntegrateAndFire
    encoders: Array
    gain_nA: Array
    bias_nA: Array
    projection_encoders: Array

    def __init__(
        self,
        domain: HyperRectangle,
        neuron: LeakyIntegrateAndFire,
        encoders: ArrayLike,
        gain_nA: ArrayLike,
        bias_nA: ArrayLike,
        /,
    ):
        if not isinstance(domain, HyperRectangle):
            raise TypeError("domain must be a native HyperRectangle.")
        if not np.all(np.isfinite(np.asarray(domain.bounds))):
            raise ValueError("Population domain bounds must be finite.")
        raw = np.asarray(encoders)
        if np.iscomplexobj(raw):
            raise TypeError("encoders must be real-valued.")
        vectors = np.asarray(raw, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != domain.spatial_dim:
            raise ValueError("encoders must have shape (neurons, domain.spatial_dim).")
        count = int(vectors.shape[0])
        if count < 1 or not np.all(np.isfinite(vectors)):
            raise ValueError("encoders must contain at least one finite neuron.")
        lengths = np.sqrt(np.sum(vectors * vectors, axis=-1))
        if np.any(lengths == 0.0) or not np.all(np.isfinite(lengths)):
            raise ValueError("Each encoder must have a finite nonzero norm.")
        vectors = vectors / lengths[:, None]
        _validate_neuron(neuron, count)
        self.domain = domain
        self.neuron = neuron
        self.encoders = jnp.asarray(vectors)
        self.projection_encoders = jnp.asarray(
            vectors / np.sum(np.abs(vectors), axis=-1, keepdims=True)
        )
        self.gain_nA = jnp.asarray(_host_vector(gain_nA, count, "gain_nA"))
        self.bias_nA = jnp.asarray(_host_vector(bias_nA, count, "bias_nA"))

    @property
    def neuron_count(self) -> int:
        return int(self.encoders.shape[0])

    def currents(self, points: ArrayLike, /) -> Array:
        """Return inward injected current, preserving arbitrary leading axes."""
        values = jnp.asarray(points)
        if values.ndim < 1 or values.shape[-1] != self.domain.spatial_dim:
            raise ValueError("points must have a trailing domain coordinate axis.")
        normalized = (
            2.0 * (values - self.domain.lower) / (self.domain.upper - self.domain.lower)
            - 1.0
        )
        projections = contract("...d,nd->...n", normalized, self.projection_encoders)
        return projections * self.gain_nA + self.bias_nA

    def rates(self, points: ArrayLike, /) -> Array:
        """Return steady physical LIF activities in Hz, without clipping points."""
        return lif_rate_response(self.neuron, self.currents(points))


def prepare_lif_population(
    domain: HyperRectangle,
    neuron: LeakyIntegrateAndFire,
    neuron_count: int,
    /,
    *,
    key: Key,
    encoders: ArrayLike | None = None,
    intercepts: ArrayLike | None = None,
    maximum_rates_hz: ArrayLike | None = None,
    intercept_range: tuple[float, float] = (-0.9, 0.9),
    maximum_rate_range_hz: tuple[float, float] = (50.0, 150.0),
) -> LIFPopulation:
    """Construct encoders and invert physical rates into gains and biases.

    The intercept is the normalized directional projection at rheobase; the
    maximum rate is reached at projection +1. Supplied encoders are normalized.
    Neuron parameters may be scalar or have one value per neuron. Requested
    maximum rates must be strictly below the refractory frequency ceiling.
    """
    count = int(neuron_count)
    if count < 1 or count != neuron_count:
        raise ValueError("neuron_count must be a positive integer.")
    if not isinstance(domain, HyperRectangle):
        raise TypeError("domain must be a native HyperRectangle.")
    _validate_neuron(neuron, count)
    encoder_key, intercept_key, rate_key = jr.split(key, 3)
    if encoders is None:
        encoders = jr.normal(encoder_key, (count, domain.spatial_dim))
    if intercepts is None:
        lower, upper = (float(value) for value in intercept_range)
        if not (isfinite(lower) and isfinite(upper) and -1.0 <= lower < upper < 1.0):
            raise ValueError("intercept_range requires -1 <= lower < upper < 1.")
        intercepts = jr.uniform(intercept_key, (count,), minval=lower, maxval=upper)
    intercept = _host_vector(intercepts, count, "intercepts")
    if np.any(intercept < -1.0) or np.any(intercept >= 1.0):
        raise ValueError("intercepts must lie in [-1, 1).")
    if maximum_rates_hz is None:
        lower, upper = (float(value) for value in maximum_rate_range_hz)
        if not (isfinite(lower) and isfinite(upper) and 0.0 < lower < upper):
            raise ValueError("maximum_rate_range_hz requires 0 < lower < upper.")
        maximum_rates_hz = jr.uniform(rate_key, (count,), minval=lower, maxval=upper)
    maximum = _host_vector(maximum_rates_hz, count, "maximum_rates_hz")
    refractory = np.asarray(neuron.refractory_ms)
    if np.any(maximum <= 0.0) or np.any(maximum * refractory >= 1000.0):
        raise ValueError(
            "Maximum rates must be positive and below the refractory ceiling."
        )
    conductance = np.asarray(neuron.leak_conductance_uS)
    charge_ms = 1000.0 / maximum - refractory
    capacitance = np.asarray(neuron.capacitance_nF)
    scaled_charge = charge_ms * conductance / capacitance
    # z/expm1(z) in a non-overflowing form, with its perfect-integrator limit.
    denominator = np.where(scaled_charge == 0.0, 1.0, -np.expm1(-scaled_charge))
    rate_ratio = np.where(
        scaled_charge == 0.0,
        1.0,
        scaled_charge * np.exp(-scaled_charge) / denominator,
    )
    excess = (
        capacitance
        * (np.asarray(neuron.threshold_mV) - np.asarray(neuron.reset_mV))
        / charge_ms
        * rate_ratio
    )
    if np.any(excess <= 0.0) or not np.all(np.isfinite(excess)):
        raise ValueError(
            "Requested rates cannot be represented by finite physical gains."
        )
    threshold_current = conductance * (
        np.asarray(neuron.threshold_mV) - np.asarray(neuron.resting_mV)
    )
    gain = excess / (1.0 - intercept)
    bias = threshold_current - gain * intercept
    result = LIFPopulation(domain, neuron, encoders, gain, bias)
    if result.neuron_count != count:
        raise ValueError("encoders must contain neuron_count rows.")
    return result


def sample_population_points(
    population: LIFPopulation,
    count: int,
    /,
    *,
    key: Key,
    sampler: str = "latin_hypercube",
) -> Array:
    """Sample evaluation or independent held-out points using the native domain."""
    size = int(count)
    if size < 1 or size != count:
        raise ValueError("count must be a positive integer.")
    return population.domain.sample_interior(size, sampler=sampler, key=key)


def _targets(points: Array, target: Target, /) -> Array:
    values = (
        points
        if target is None
        else jax.vmap(target)(points)
        if callable(target)
        else target
    )
    result = jnp.asarray(values)
    if result.ndim < 1 or result.shape[0] != points.shape[0]:
        raise ValueError("target must have one leading entry per evaluation point.")
    if jnp.iscomplexobj(result):
        raise TypeError("Population targets must be real-valued.")
    if result.size == 0:
        raise ValueError("Population targets must contain at least one output.")
    return result


def _sample_measure(
    count: int,
    mask: ArrayLike | None,
    weights: ArrayLike | None,
    *values: Array,
) -> tuple[Array, Array, Array]:
    requested = (
        jnp.ones((count,), dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
    )
    raw = jnp.ones((count,)) if weights is None else jnp.asarray(weights)
    if requested.shape != (count,) or raw.shape != (count,):
        raise ValueError("mask and weights must have one entry per sample.")
    if jnp.iscomplexobj(raw):
        raise TypeError("Sample weights must be real-valued.")
    raw = raw.astype(jnp.result_type(raw, float))
    valid = requested & jnp.isfinite(raw) & (raw > 0.0)
    for value in values:
        valid = valid & jnp.all(jnp.isfinite(value.reshape((count, -1))), axis=-1)
    effective = jnp.where(valid, raw, 0.0)
    weight_sum = jnp.sum(effective)
    # Relative weights define the measure independently of global scaling,
    # including when the sum is below one and a ridge penalty is requested.
    normalized = effective / jnp.where(weight_sum > 0.0, weight_sum, 1.0)
    return valid, normalized, weight_sum


class PopulationAssessment(StrictModule, NonTrainableState):
    """Weighted output-wise errors; empty effective samples produce NaN errors."""

    prediction: Array
    residual: Array
    valid_rows: Array
    sample_count: Array
    weight_sum: Array
    rmse: Array
    target_rms: Array
    relative_rmse: Array
    maximum_absolute_error: Array
    valid: Array


def _assess(
    prediction: Array,
    target: Array,
    mask: ArrayLike | None,
    weights: ArrayLike | None,
) -> PopulationAssessment:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must agree.")
    count = int(prediction.shape[0])
    valid_rows, normalized, weight_sum = _sample_measure(
        count, mask, weights, prediction, target
    )
    shape = (count,) + (1,) * (prediction.ndim - 1)
    selected = valid_rows.reshape(shape)
    measure = normalized.reshape(shape)
    residual = prediction - target
    safe_residual = jnp.where(selected, residual, 0.0)
    safe_target = jnp.where(selected, target, 0.0)
    valid = (weight_sum > 0.0) & jnp.isfinite(weight_sum)
    rmse = jnp.where(
        valid, jnp.sqrt(jnp.sum(measure * safe_residual**2, axis=0)), jnp.nan
    )
    target_rms = jnp.where(
        valid, jnp.sqrt(jnp.sum(measure * safe_target**2, axis=0)), jnp.nan
    )
    relative = jnp.where(
        target_rms > 0.0,
        rmse / jnp.where(target_rms > 0.0, target_rms, 1.0),
        jnp.where(rmse == 0.0, 0.0, jnp.inf),
    )
    maximum = jnp.where(valid, jnp.max(jnp.abs(safe_residual), axis=0), jnp.nan)
    return PopulationAssessment(
        prediction,
        residual,
        valid_rows,
        jnp.sum(valid_rows),
        weight_sum,
        rmse,
        target_rms,
        jnp.where(valid, relative, jnp.nan),
        maximum,
        valid,
    )


class PopulationCode(StrictModule, NonTrainableState):
    """A deliberately frozen fit, its population, and native SVD diagnostics.

    ``least_squares`` retains rank, singular values, conditioning, status,
    validity, and normal-equation error. Ridge may yield a usable fit even when
    rank is deficient; it does not change the reported unregularized rank.
    ``silent_neurons`` identifies selected zero-activity columns on valid rows.
    Duplicate/dependent columns appear as a rank deficit, not silently pruned
    neurons. ``training_weight_sum`` retains the unnormalized effective measure.
    """

    population: LIFPopulation
    least_squares: WeightedLeastSquaresResult
    neuron_mask: Array
    silent_neurons: Array
    activity_rms_hz: Array
    training_weight_sum: Array

    def decode_rates(self, rates_hz: ArrayLike, /) -> Array:
        """Apply decoders to continuous rates in Hz, not unfiltered spike counts."""
        rates = jnp.asarray(rates_hz)
        if rates.ndim < 1 or rates.shape[-1] != self.population.neuron_count:
            raise ValueError("rates_hz must have a trailing population neuron axis.")
        coefficients = self.least_squares.raw_coefficients.reshape(
            (self.population.neuron_count, -1)
        )
        selected_rates = jnp.where(self.neuron_mask, rates, 0.0)
        decoded = contract("...n,no->...o", selected_rates, coefficients)
        return (
            decoded.reshape(rates.shape[:-1] + self.least_squares.output_shape)
            + self.least_squares.intercept
        )

    def __call__(self, points: ArrayLike, /) -> Array:
        return self.decode_rates(self.population.rates(points))


def fit_population_decoder(
    population: LIFPopulation,
    evaluation_points: ArrayLike,
    target: Target = None,
    /,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    neuron_mask: ArrayLike | None = None,
    ridge: float = 1.0e-6,
    rcond: float | None = None,
) -> PopulationCode:
    """Fit pointwise callable/array/identity targets with the native weighted SVD.

    Weights are relative nonnegative sample masses, normalized after masks and
    nonfinite rows are excluded. Zero, negative, and nonfinite weights do not
    contribute, matching the native least-squares substrate. Multiplying all
    weights by a positive constant leaves the fit and ridge tradeoff unchanged.
    No centering/scaling or hidden intercept feature changes the rate decoder.
    Rank deficiency is reported, including selected silent/duplicate neurons.
    """
    points = jnp.asarray(evaluation_points)
    if points.ndim != 2 or points.shape[0] < 1:
        raise ValueError(
            "evaluation_points must be a nonempty (samples, coordinates) array."
        )
    target_values = _targets(points, target)
    rates = population.rates(points)
    active = (
        jnp.ones((population.neuron_count,), dtype=bool)
        if neuron_mask is None
        else jnp.asarray(neuron_mask, dtype=bool)
    )
    if active.shape != (population.neuron_count,):
        raise ValueError("neuron_mask must have one entry per neuron.")
    rates = jnp.where(active, rates, 0.0)
    valid_rows, normalized, weight_sum = _sample_measure(
        points.shape[0], mask, weights, points, rates, target_values
    )
    result = solve_weighted_least_squares(
        rates,
        target_values,
        mask=valid_rows,
        weights=normalized,
        feature_mask=active,
        ridge=ridge,
        rcond=rcond,
        min_samples=1,
    )
    safe_rates = jnp.where(valid_rows[:, None], rates, 0.0)
    activity_rms = jnp.sqrt(jnp.sum(normalized[:, None] * safe_rates**2, axis=0))
    return PopulationCode(
        population,
        result,
        active,
        active & (activity_rms == 0.0),
        activity_rms,
        weight_sum,
    )


def assess_population_code(
    code: PopulationCode,
    points: ArrayLike,
    target: Target = None,
    /,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> PopulationAssessment:
    """Assess on caller-supplied held-out points, independently of fitting data."""
    values = jnp.asarray(points)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("points must be a nonempty (samples, coordinates) array.")
    return _assess(code(values), _targets(values, target), mask, weights)


class FilteredSpikeRates(StrictModule, NonTrainableState):
    """Explicit Hz activities from causal filtering of binned spike counts.

    Each bin's count/duration is a zero-order-held rate; the first-order filter
    is integrated exactly over that bin and reported at its right endpoint.
    This preserves the DC rate without pretending spikes occurred at bin edges.
    ``final_rate_hz`` can seed the next chunk; ``initial_rate_hz`` records the
    actual prehistory needed for a matched deterministic filtering baseline.
    """

    rates_hz: Array
    initial_rate_hz: Array
    final_rate_hz: Array
    dt_ms: float = eqx.field(static=True)
    time_constant_ms: float = eqx.field(static=True)


def _filter_rates(
    rates: Array, initial: Array, dt_ms: float, tau_ms: float
) -> tuple[Array, Array]:
    decay = jnp.exp(-dt_ms / tau_ms)
    increment = -jnp.expm1(-dt_ms / tau_ms)

    def step(previous, rate):
        current = decay * previous + increment * rate
        return current, current

    return jax.lax.scan(step, initial, rates)


def filter_population_spikes(
    spike_counts: ArrayLike,
    /,
    *,
    dt_ms: float,
    time_constant_ms: float,
    initial_rate_hz: ArrayLike | None = None,
) -> FilteredSpikeRates:
    """Causally filter nonnegative counts shaped (time, ..., neurons) into Hz.

    Counts may exceed one; no spike-generation runtime is introduced. Inputs
    must already share an explicit uniform clock. Fractional counts are allowed
    for expected-count/weighted-event calculations, not rounded to booleans.
    """
    dt, tau = float(dt_ms), float(time_constant_ms)
    if not (isfinite(dt) and isfinite(tau) and dt > 0.0 and tau > 0.0):
        raise ValueError("dt_ms and time_constant_ms must be finite and positive.")
    counts = jnp.asarray(spike_counts)
    if jnp.iscomplexobj(counts):
        raise TypeError("spike_counts must be real-valued.")
    counts = counts.astype(jnp.result_type(counts, float))
    if counts.ndim < 2 or counts.shape[-1] < 1:
        raise ValueError("spike_counts must have shape (time, ..., neurons).")
    counts = eqx.error_if(
        counts,
        jnp.any(~jnp.isfinite(counts) | (counts < 0.0)),
        "spike_counts must be finite and nonnegative.",
    )
    initial = (
        jnp.zeros(counts.shape[1:], dtype=counts.dtype)
        if initial_rate_hz is None
        else jnp.asarray(initial_rate_hz)
    )
    if jnp.iscomplexobj(initial):
        raise TypeError("initial_rate_hz must be real-valued.")
    initial = initial.astype(counts.dtype)
    if initial.shape != counts.shape[1:]:
        raise ValueError("initial_rate_hz must match the non-time spike-count axes.")
    initial = eqx.error_if(
        initial,
        jnp.any(~jnp.isfinite(initial) | (initial < 0.0)),
        "initial_rate_hz must be finite and nonnegative.",
    )
    final, rates = _filter_rates(counts * (1000.0 / dt), initial, dt, tau)
    return FilteredSpikeRates(rates, initial, final, dt, tau)


def decode_filtered_spikes(
    code: PopulationCode, filtered: FilteredSpikeRates, /
) -> Array:
    """Apply the Hz decoder only to an explicitly filtered spike-rate result."""
    if not isinstance(filtered, FilteredSpikeRates):
        raise TypeError("filtered must be produced by filter_population_spikes.")
    return code.decode_rates(filtered.rates_hz)


class PopulationSpikeAssessment(StrictModule, NonTrainableState):
    """Temporal error decomposition under one effective sample measure.

    Signed residuals add: total = approximation + filtering + spike_variability.
    Their RMSEs do not add: the errors are generally correlated. ``filtering``
    measures deterministic lag relative to unfiltered decoded physical rates;
    ``spike_variability`` compares spikes against that matched filtered baseline.
    """

    approximation: PopulationAssessment
    filtering: PopulationAssessment
    spike_variability: PopulationAssessment
    total: PopulationAssessment


def assess_population_spikes(
    code: PopulationCode,
    points: ArrayLike,
    filtered: FilteredSpikeRates,
    target: Target = None,
    /,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> PopulationSpikeAssessment:
    """Separate function approximation, temporal filtering, and spike errors.

    ``points`` is (time, coordinates), aligned to the same uniform bins as the
    spike result. Masks affect assessment only, never causal filter evolution.
    The deterministic baseline uses the recorded filter initial condition.
    """
    if not isinstance(filtered, FilteredSpikeRates):
        raise TypeError("filtered must be produced by filter_population_spikes.")
    values = jnp.asarray(points)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError(
            "points must have shape (time, coordinates), with nonempty time."
        )
    rates = code.population.rates(values)
    if rates.shape != filtered.rates_hz.shape:
        raise ValueError(
            "Filtered spikes and pointwise population rates must have matching shapes."
        )
    _, filtered_rates = _filter_rates(
        rates, filtered.initial_rate_hz, filtered.dt_ms, filtered.time_constant_ms
    )
    target_values = _targets(values, target)
    rate_prediction = code.decode_rates(rates)
    filtered_prediction = code.decode_rates(filtered_rates)
    spike_prediction = decode_filtered_spikes(code, filtered)
    valid_rows, _, _ = _sample_measure(
        values.shape[0],
        mask,
        weights,
        target_values,
        rate_prediction,
        filtered_prediction,
        spike_prediction,
    )
    return PopulationSpikeAssessment(
        _assess(rate_prediction, target_values, valid_rows, weights),
        _assess(filtered_prediction, rate_prediction, valid_rows, weights),
        _assess(spike_prediction, filtered_prediction, valid_rows, weights),
        _assess(spike_prediction, target_values, valid_rows, weights),
    )


__all__ = [
    "FilteredSpikeRates",
    "LIFPopulation",
    "PopulationAssessment",
    "PopulationCode",
    "PopulationSpikeAssessment",
    "assess_population_code",
    "assess_population_spikes",
    "decode_filtered_spikes",
    "filter_population_spikes",
    "fit_population_decoder",
    "lif_rate_response",
    "prepare_lif_population",
    "sample_population_points",
]
