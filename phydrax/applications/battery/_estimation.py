#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exactly linear state estimation for a narrowly qualified battery ECM.

The route in this module is intentionally separate from the general thermal ECM. It
uses the hidden state ``[state_of_charge, polarization_voltage...]`` and is exact only
for constant R/C values, fixed reference capacity, a held passive-sign current, known
(or constant) temperature, and explicitly global affine OCV and entropic laws. The
global mathematical law is required because a nondegenerate Gaussian state cannot be
confined to a finite interpolation interval. Physical SOC bounds remain explicit
result evidence. The native state-space problem delegates the actual filter and
smoother recursions to :mod:`phydrax.uq`.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...operators.interpolation import InterpolationResult
from ...stochastic import (
    GaussianStatePrior,
    LinearGaussianObservationModel,
    LinearGaussianParameterization,
    LinearGaussianTransitionKernel,
    ObservationSequence,
    SampledStateSpaceInput,
    StateSpaceModel,
    StateSpaceProblem,
    StateSpaceStepContext,
)
from ...uq import (
    kalman_filter,
    kalman_innovation_diagnostics,
    KalmanCovarianceForm,
    KalmanExecutionMethod,
    KalmanFilterResult,
    KalmanInnovationDiagnostics,
    KalmanSmootherResult,
    rts_smoother,
)
from ._ecm import ThermalEquivalentCircuitParameters, ThermalEquivalentCircuitPlan
from ._observations import BatteryRecordRole, BatteryTimeSeriesRecord
from ._properties import ConstantPropertyLaw, TabulatedPropertyLaw


TemperatureMode = Literal["isothermal", "known"]


def _real_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return array


def _host_finite(array: Array, name: str, /) -> np.ndarray:
    host = np.asarray(array)
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must be finite.")
    return host


def _positive_scalar(value: ArrayLike, name: str, /) -> Array:
    array = _real_array(value, name)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    host = _host_finite(array, name)
    if not bool(host > 0.0):
        raise ValueError(f"{name} must be positive.")
    return array


def _covariance_matrix(
    value: ArrayLike,
    size: int,
    name: str,
    /,
) -> Array:
    array = _real_array(value, name)
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}).")
    source_host = _host_finite(array, name)
    epsilon = np.finfo(source_host.dtype).eps
    host = source_host.astype(float, copy=False)
    tolerance = 64.0 * epsilon * max(1.0, float(np.max(np.abs(host))))
    if not np.allclose(host, host.T, rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} must be symmetric.")
    symmetric = 0.5 * (host + host.T)
    if np.any(np.linalg.eigvalsh(symmetric) < -tolerance):
        raise ValueError(f"{name} must be positive semidefinite.")
    return 0.5 * (array + array.T)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


class GloballyAffineSOCPropertyLaw(StrictModule):
    """An explicit real-line affine law for an exact Gaussian ECM observation.

    ``intercept`` is the value at zero SOC and ``slope`` is per unit state of
    charge. This distinct contract cannot be confused with a finite property table.
    The general thermal ECM retains its separate physical ``0 <= SOC <= 1`` evidence.
    """

    intercept: Array
    slope: Array
    quantity: str = eqx.field(static=True)
    coordinate: str = eqx.field(static=True)
    value_unit: str = eqx.field(static=True)
    coordinate_unit: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        intercept: ArrayLike,
        slope: ArrayLike,
        /,
        *,
        quantity: str,
        value_unit: str,
        source_id: str,
    ):
        intercept_array = _real_array(intercept, "intercept")
        slope_array = _real_array(slope, "slope")
        if intercept_array.shape != () or slope_array.shape != ():
            raise ValueError("Global affine property intercept and slope must be scalar.")
        dtype = jnp.result_type(intercept_array, slope_array)
        intercept_array = intercept_array.astype(dtype)
        slope_array = slope_array.astype(dtype)
        _host_finite(intercept_array, "intercept")
        _host_finite(slope_array, "slope")
        quantity_id = _identifier(quantity, "Property quantity")
        unit_id = _identifier(value_unit, "Property value unit")
        source = _identifier(source_id, "Property source ID")
        self.intercept = intercept_array
        self.slope = slope_array
        self.quantity = quantity_id
        self.coordinate = "state_of_charge"
        self.value_unit = unit_id
        self.coordinate_unit = "1"
        self.source_id = source
        self.law_id = canonical_fingerprint(
            {
                "kind": "battery-global-affine-soc-property-law",
                "quantity": quantity_id,
                "value_unit": unit_id,
                "source_id": source,
                "mathematical_support": "real-line",
            }
        )

    def evaluate(
        self,
        query: ArrayLike,
        /,
        *,
        derivative_order: int = 0,
    ) -> InterpolationResult:
        if derivative_order not in (0, 1):
            raise ValueError("Global affine laws support derivative orders zero and one.")
        query_array = _real_array(query, "query")
        support = jnp.isfinite(query_array)
        values = (
            jnp.zeros_like(query_array, dtype=self.slope.dtype) + self.slope
            if derivative_order
            else self.intercept + self.slope * query_array
        )
        return InterpolationResult(jnp.where(support, values, 0.0), support)

    def __call__(self, query: ArrayLike, /) -> InterpolationResult:
        return self.evaluate(query)


def _qualify_global_extension(
    physical_law: ConstantPropertyLaw | TabulatedPropertyLaw,
    global_law: GloballyAffineSOCPropertyLaw,
    interval: Array,
    owner: str,
    expected_unit: str,
    /,
) -> Array:
    if not isinstance(global_law, GloballyAffineSOCPropertyLaw):
        raise TypeError(
            f"{owner} must be GloballyAffineSOCPropertyLaw; finite-support "
            "constant and tabulated laws cannot define an exact Gaussian profile."
        )
    if global_law.value_unit != expected_unit:
        raise ValueError(f"{owner} must use value unit {expected_unit!r}.")
    if global_law.quantity != physical_law.quantity:
        raise ValueError(f"{owner} quantity must match the thermal ECM property.")
    bounds = np.asarray(interval, dtype=float)
    support = np.asarray(physical_law.support_bounds, dtype=float)
    if bounds[0] < support[0] or bounds[1] > support[1]:
        raise ValueError(
            f"The thermal ECM {owner} does not support the physical interval."
        )
    if isinstance(physical_law, TabulatedPropertyLaw):
        nodes = np.asarray(physical_law.nodes, dtype=float)
        mask = np.asarray(physical_law.source_mask, dtype=bool)
        overlaps = (nodes[:-1] < bounds[1]) & (nodes[1:] > bounds[0])
        if not np.any(overlaps) or np.any(overlaps & ~(mask[:-1] & mask[1:])):
            raise ValueError(f"The thermal ECM {owner} has a physical support gap.")
        interior = nodes[(nodes > bounds[0]) & (nodes < bounds[1])]
        coordinates = np.concatenate(([bounds[0]], interior, [bounds[1]]))
    else:
        coordinates = bounds
    queries = jnp.asarray(coordinates, dtype=interval.dtype)
    physical = np.asarray(physical_law.evaluate(queries).values, dtype=float)
    extension = np.asarray(global_law.evaluate(queries).values, dtype=float)
    epsilon = np.finfo(np.asarray(physical_law.evaluate(queries).values).dtype).eps
    scale = max(1.0, float(np.max(np.abs(physical))), float(np.max(np.abs(extension))))
    if not np.allclose(
        physical, extension, rtol=64.0 * epsilon, atol=64.0 * epsilon * scale
    ):
        raise ValueError(
            f"The global {owner} does not agree with the thermal ECM on the "
            "declared physical SOC interval."
        )
    return jnp.stack((global_law.intercept, global_law.slope))


def _prepared_args(context: StateSpaceStepContext, /) -> PreparedExactAffineECMEstimation:
    arguments = context.args
    if not isinstance(arguments, PreparedExactAffineECMEstimation):
        raise TypeError(
            "Exact affine ECM state-space context has incompatible arguments."
        )
    return arguments


def _transition_matrix(
    start_time_s: Array,
    end_time_s: Array,
    context: StateSpaceStepContext,
    /,
) -> Array:
    prepared = _prepared_args(context)
    duration = end_time_s - start_time_s
    decay = jnp.exp(
        -duration / (prepared.branch_resistances_ohm * prepared.branch_capacitances_f)
    )
    return jnp.diag(jnp.concatenate((jnp.ones((1,), dtype=decay.dtype), decay)))


def _transition_offset(
    start_time_s: Array,
    end_time_s: Array,
    context: StateSpaceStepContext,
    /,
) -> Array:
    prepared = _prepared_args(context)
    duration = end_time_s - start_time_s
    current_a = context.transition_start_input[0]
    inverse_time_constants = 1.0 / (
        prepared.branch_resistances_ohm * prepared.branch_capacitances_f
    )
    state_of_charge_increment = current_a * duration / prepared.reference_capacity_c
    polarization_increment = (
        current_a
        * prepared.branch_resistances_ohm
        * (-jnp.expm1(-inverse_time_constants * duration))
    )
    return jnp.concatenate(
        (state_of_charge_increment.reshape((1,)), polarization_increment)
    )


def _transition_covariance(
    start_time_s: Array,
    end_time_s: Array,
    context: StateSpaceStepContext,
    /,
) -> Array:
    prepared = _prepared_args(context)
    duration = end_time_s - start_time_s
    decay_rates = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=prepared.branch_resistances_ohm.dtype),
            -1.0 / (prepared.branch_resistances_ohm * prepared.branch_capacitances_f),
        )
    )
    pair_rates = decay_rates[:, None] + decay_rates[None, :]
    safe_rates = jnp.where(pair_rates == 0.0, 1.0, pair_rates)
    integral = jnp.where(
        pair_rates == 0.0,
        duration,
        jnp.expm1(pair_rates * duration) / safe_rates,
    )
    return prepared.process_noise_covariance_rate * integral


def _observation_matrix(time_s: Array, context: StateSpaceStepContext, /) -> Array:
    del time_s
    prepared = _prepared_args(context)
    temperature_k = context.observation_input[1]
    temperature_delta = temperature_k - prepared.reference_temperature_k
    slope = (
        prepared.ocv_coefficients_v[1]
        + temperature_delta * prepared.entropic_coefficients_v_per_k[1]
    )
    return jnp.concatenate(
        (
            slope.reshape((1, 1)),
            jnp.ones((1, prepared.branch_count), dtype=slope.dtype),
        ),
        axis=1,
    )


def _observation_offset(time_s: Array, context: StateSpaceStepContext, /) -> Array:
    del time_s
    prepared = _prepared_args(context)
    current_a = context.observation_input[0]
    temperature_k = context.observation_input[1]
    temperature_delta = temperature_k - prepared.reference_temperature_k
    open_circuit_intercept = (
        prepared.ocv_coefficients_v[0]
        + temperature_delta * prepared.entropic_coefficients_v_per_k[0]
    )
    voltage_v = open_circuit_intercept + current_a * prepared.series_resistance_ohm
    return voltage_v.reshape((1,))


def _observation_covariance(time_s: Array, context: StateSpaceStepContext, /) -> Array:
    del time_s
    prepared = _prepared_args(context)
    return prepared.voltage_variance_v2.reshape((1, 1))


class ExactAffineECMEstimationPlan(StrictModule, NonTrainableState):
    """Static qualification boundary for exact affine ECM state estimation."""

    ecm_plan: ThermalEquivalentCircuitPlan
    state_of_charge_interval: Array
    temperature_mode: TemperatureMode = eqx.field(static=True)
    branch_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ecm_plan: ThermalEquivalentCircuitPlan,
        state_of_charge_interval: ArrayLike,
        /,
        *,
        temperature_mode: TemperatureMode = "known",
        hysteresis: bool = False,
        capacity_fade: bool = False,
        parameter_estimation: bool = False,
    ):
        if not isinstance(ecm_plan, ThermalEquivalentCircuitPlan):
            raise TypeError("ecm_plan must be a ThermalEquivalentCircuitPlan.")
        interval = _real_array(state_of_charge_interval, "state_of_charge_interval")
        if interval.shape != (2,):
            raise ValueError("state_of_charge_interval must contain two bounds.")
        host_interval = _host_finite(interval, "state_of_charge_interval")
        if (
            not host_interval[1] > host_interval[0]
            or host_interval[0] < 0.0
            or host_interval[1] > 1.0
        ):
            raise ValueError(
                "state_of_charge_interval must be an increasing subset of [0, 1]."
            )
        if temperature_mode not in ("isothermal", "known"):
            raise ValueError("temperature_mode must be 'isothermal' or 'known'.")
        for value, name in (
            (hysteresis, "hysteresis"),
            (capacity_fade, "capacity_fade"),
            (parameter_estimation, "parameter_estimation"),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean.")
            if value:
                raise ValueError(
                    f"Exact affine ECM state estimation does not support {name}."
                )
        self.ecm_plan = ecm_plan
        self.state_of_charge_interval = interval
        self.temperature_mode = temperature_mode
        self.branch_count = ecm_plan.branch_count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "battery-exact-affine-ecm-estimation-plan",
                "ecm_plan_id": ecm_plan.plan_id,
                "physical_state_of_charge_interval": host_interval.tolist(),
                "property_law_mathematical_support": "real-line",
                "temperature_mode": temperature_mode,
                "current_sign": "positive-enters-positive-terminal",
                "current_interpolation": "zero-order-hold",
                "resistance_capacitance": "constant",
                "capacity": "fixed",
                "hysteresis": False,
                "capacity_fade": False,
                "parameter_estimation": False,
            }
        )

    def prepare(
        self,
        parameters: ThermalEquivalentCircuitParameters,
        /,
        *,
        open_circuit_voltage: GloballyAffineSOCPropertyLaw,
        entropic_coefficient: GloballyAffineSOCPropertyLaw,
        process_noise_covariance_rate: ArrayLike | None = None,
        voltage_variance_v2: ArrayLike,
    ) -> PreparedExactAffineECMEstimation:
        """Qualify one constant-parameter ECM and retain its dynamic array leaves.

        The required global laws extend, and must agree with, the thermal ECM laws
        throughout the declared physical SOC interval. This makes the Gaussian
        likelihood exactly affine even though a Gaussian has real-line support.
        ``process_noise_covariance_rate`` is the continuous-time covariance rate
        in the declared state coordinates. Its exact integral under the diagonal
        ECM drift is formed separately for every observation interval.
        """

        if not isinstance(parameters, ThermalEquivalentCircuitParameters):
            raise TypeError("parameters must be ThermalEquivalentCircuitParameters.")
        if parameters.branch_resistances_ohm.shape != (self.branch_count,):
            raise ValueError("ECM parameters do not match the estimation branch count.")
        ocv_coefficients = _qualify_global_extension(
            parameters.open_circuit_voltage,
            open_circuit_voltage,
            self.state_of_charge_interval,
            "open_circuit_voltage",
            "V",
        )
        entropic_coefficients = _qualify_global_extension(
            parameters.entropic_coefficient,
            entropic_coefficient,
            self.state_of_charge_interval,
            "entropic_coefficient",
            "V/K",
        )
        state_size = self.branch_count + 1
        process_covariance = (
            jnp.zeros((state_size, state_size), dtype=ocv_coefficients.dtype)
            if process_noise_covariance_rate is None
            else _covariance_matrix(
                process_noise_covariance_rate,
                state_size,
                "process_noise_covariance_rate",
            )
        )
        voltage_variance = _positive_scalar(voltage_variance_v2, "voltage_variance_v2")
        identity = array_tree_fingerprint(
            {
                "series_resistance_ohm": parameters.series_resistance_ohm,
                "branch_resistances_ohm": parameters.branch_resistances_ohm,
                "branch_capacitances_f": parameters.branch_capacitances_f,
                "reference_capacity_c": parameters.reference_capacity_c,
                "reference_temperature_k": parameters.reference_temperature_k,
                "ocv_coefficients_v": ocv_coefficients,
                "entropic_coefficients_v_per_k": entropic_coefficients,
                "process_noise_covariance_rate": process_covariance,
                "voltage_variance_v2": voltage_variance,
            }
        )
        prepared_id = canonical_fingerprint(
            {
                "global_ocv_law_id": open_circuit_voltage.law_id,
                "global_entropic_law_id": entropic_coefficient.law_id,
                "kind": "prepared-battery-exact-affine-ecm-estimation",
                "plan_id": self.plan_id,
                "source_parameter_id": parameters.parameter_id,
                "numeric_sha256": identity["sha256"],
            }
        )
        return PreparedExactAffineECMEstimation(
            plan=self,
            series_resistance_ohm=parameters.series_resistance_ohm,
            branch_resistances_ohm=parameters.branch_resistances_ohm,
            branch_capacitances_f=parameters.branch_capacitances_f,
            reference_capacity_c=parameters.reference_capacity_c,
            reference_temperature_k=parameters.reference_temperature_k,
            ocv_coefficients_v=ocv_coefficients,
            entropic_coefficients_v_per_k=entropic_coefficients,
            process_noise_covariance_rate=process_covariance,
            voltage_variance_v2=voltage_variance,
            prepared_id=prepared_id,
        )


class PreparedExactAffineECMEstimation(StrictModule):
    """Qualified exact coefficients; all physical/statistical values remain leaves."""

    plan: ExactAffineECMEstimationPlan
    series_resistance_ohm: Array
    branch_resistances_ohm: Array
    branch_capacitances_f: Array
    reference_capacity_c: Array
    reference_temperature_k: Array
    ocv_coefficients_v: Array
    entropic_coefficients_v_per_k: Array
    process_noise_covariance_rate: Array
    voltage_variance_v2: Array
    branch_count: int = eqx.field(static=True)
    state_names: tuple[str, ...] = eqx.field(static=True)
    state_units: tuple[str, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        plan: ExactAffineECMEstimationPlan,
        series_resistance_ohm: Array,
        branch_resistances_ohm: Array,
        branch_capacitances_f: Array,
        reference_capacity_c: Array,
        reference_temperature_k: Array,
        ocv_coefficients_v: Array,
        entropic_coefficients_v_per_k: Array,
        process_noise_covariance_rate: Array,
        voltage_variance_v2: Array,
        prepared_id: str,
    ):
        self.plan = plan
        self.series_resistance_ohm = series_resistance_ohm
        self.branch_resistances_ohm = branch_resistances_ohm
        self.branch_capacitances_f = branch_capacitances_f
        self.reference_capacity_c = reference_capacity_c
        self.reference_temperature_k = reference_temperature_k
        self.ocv_coefficients_v = ocv_coefficients_v
        self.entropic_coefficients_v_per_k = entropic_coefficients_v_per_k
        self.process_noise_covariance_rate = process_noise_covariance_rate
        self.voltage_variance_v2 = voltage_variance_v2
        self.branch_count = plan.branch_count
        self.state_names = ("state_of_charge",) + tuple(
            f"polarization_voltage_{index}_v" for index in range(plan.branch_count)
        )
        self.state_units = ("1",) + ("V",) * plan.branch_count
        self.prepared_id = prepared_id

    @property
    def state_of_charge_interval(self) -> Array:
        return self.plan.state_of_charge_interval

    def problem(
        self,
        times_s: ArrayLike,
        current_a: ArrayLike,
        voltage_v: ArrayLike,
        known_temperature_k: ArrayLike,
        /,
        *,
        prior_mean: ArrayLike,
        prior_covariance: ArrayLike,
        voltage_mask: ArrayLike | None = None,
        step_valid: ArrayLike | None = None,
        case_axes: Sequence[str] | None = None,
        case_ids: Sequence[str] | None = None,
        identity_id: str = "battery-exact-affine-ecm-observations",
    ) -> StateSpaceProblem:
        """Build the native masked linear-Gaussian problem for held current samples.

        Current is positive when entering the positive terminal.  The zero-order-hold
        signal is right-continuous: transition ``k`` uses the current at its left
        endpoint, while voltage observation ``k`` uses the current at that sample.
        The first observation time is also the prior time, so its transition is the
        exact identity.
        """

        if not isinstance(identity_id, str) or not identity_id:
            raise ValueError("identity_id must be a non-empty string.")
        times = _real_array(times_s, "times_s")
        current = _real_array(current_a, "current_a")
        voltage = _real_array(voltage_v, "voltage_v")
        if times.ndim < 1:
            raise ValueError("times_s must have a trailing sample axis.")
        if current.shape != times.shape or voltage.shape != times.shape:
            raise ValueError(
                "current_a and voltage_v must have the same shape as times_s."
            )
        _host_finite(times, "times_s")
        _host_finite(current, "current_a")
        _host_finite(voltage, "voltage_v")
        temperature_raw = _real_array(known_temperature_k, "known_temperature_k")
        if self.plan.temperature_mode == "isothermal" and temperature_raw.shape != ():
            raise ValueError(
                "Isothermal estimation requires one scalar known temperature."
            )
        try:
            temperature = jnp.broadcast_to(temperature_raw, times.shape)
        except ValueError as error:
            raise ValueError(
                "known_temperature_k must be scalar or have the same shape as times_s."
            ) from error
        temperature_host = _host_finite(temperature, "known_temperature_k")
        if np.any(temperature_host <= 0.0):
            raise ValueError("known_temperature_k must be positive.")

        case_shape = tuple(int(size) for size in times.shape[:-1])
        axes = (
            tuple(f"case_{axis}" for axis in range(len(case_shape)))
            if case_axes is None
            else tuple(case_axes)
        )
        if len(axes) != len(case_shape):
            raise ValueError(
                "case_axes rank must match the inferred physical case shape."
            )
        count = prod(case_shape) if case_shape else 1
        resolved_case_ids = (
            tuple(f"case:{index}" for index in range(count))
            if case_ids is None
            else tuple(case_ids)
        )
        if len(resolved_case_ids) != count:
            raise ValueError("case_ids must contain one ID per physical case.")

        valid = (
            jnp.ones(times.shape, dtype=bool)
            if step_valid is None
            else jnp.broadcast_to(jnp.asarray(step_valid, dtype=bool), times.shape)
        )
        mask = (
            valid
            if voltage_mask is None
            else jnp.broadcast_to(jnp.asarray(voltage_mask, dtype=bool), times.shape)
        )
        inputs = jnp.stack((current, temperature), axis=-1)
        input_identity = array_tree_fingerprint(
            {"times_s": times, "inputs": inputs, "knot_valid": valid}
        )
        input_id = canonical_fingerprint(
            {
                "kind": "battery-held-current-known-temperature",
                "identity_id": identity_id,
                "sha256": input_identity["sha256"],
            }
        )
        input_signal = SampledStateSpaceInput(
            times,
            inputs,
            knot_valid=valid,
            interpolation="zero-order-hold",
            input_id=input_id,
        )

        observation_identity = array_tree_fingerprint(
            {
                "times_s": times,
                "voltage_v": voltage,
                "step_valid": valid,
                "voltage_mask": mask,
            }
        )
        sequence_id = canonical_fingerprint(
            {
                "kind": "battery-masked-voltage-sequence",
                "identity_id": identity_id,
                "case_ids": list(resolved_case_ids),
                "sha256": observation_identity["sha256"],
            }
        )
        observations = ObservationSequence(
            times,
            voltage[..., None],
            case_axes=axes,
            case_shape=case_shape,
            observation_axes=("terminal_voltage",),
            step_valid=valid,
            observation_mask=mask[..., None],
            case_ids=resolved_case_ids,
            sequence_id=sequence_id,
            sensor_id="battery-terminal-voltage-v",
            approximation_id="exact-affine-ecm",
        )

        state_size = self.branch_count + 1
        mean = _real_array(prior_mean, "prior_mean")
        expected_mean_shape = case_shape + (state_size,)
        if mean.shape != expected_mean_shape:
            raise ValueError(f"prior_mean must have shape {expected_mean_shape}.")
        mean_host = _host_finite(mean, "prior_mean")
        bounds = np.asarray(self.state_of_charge_interval, dtype=float)
        if np.any((mean_host[..., 0] < bounds[0]) | (mean_host[..., 0] > bounds[1])):
            raise ValueError("Initial mean SOC is outside the declared affine interval.")
        covariance = _real_array(prior_covariance, "prior_covariance")
        prior_identity = array_tree_fingerprint({"mean": mean, "covariance": covariance})
        prior_id = canonical_fingerprint(
            {
                "kind": "battery-exact-affine-ecm-prior",
                "identity_id": identity_id,
                "sha256": prior_identity["sha256"],
            }
        )
        prior = GaussianStatePrior(
            mean,
            covariance,
            state_shape=(state_size,),
            prior_id=prior_id,
        )
        parameterization = LinearGaussianParameterization(
            _transition_matrix,
            _transition_covariance,
            state_shape=(state_size,),
            offset=_transition_offset,
            parameterization_id=self.prepared_id,
            resolved_method="analytic-exponential-held-current",
        )
        transition = LinearGaussianTransitionKernel(
            parameterization,
            process_id="battery-exact-affine-ecm-state",
            approximation_id="exact-affine-isothermal-ecm",
            has_log_density=False,
        )
        observation = LinearGaussianObservationModel(
            _observation_matrix,
            _observation_covariance,
            state_shape=(state_size,),
            observation_shape=(1,),
            offset=_observation_offset,
            observation_id="battery-passive-terminal-voltage",
        )
        model = StateSpaceModel(
            prior,
            transition,
            observation,
            model_id=self.prepared_id,
            parameter_id=self.prepared_id,
            discretization_id="analytic-held-current-exponential",
            metadata={
                "profile": "exact-affine-ecm-estimation",
                "temperature_mode": self.plan.temperature_mode,
                "terminal_current_sign": "positive-enters-positive-terminal",
                "terminal_voltage_sign": "positive-terminal-minus-negative-terminal",
                "state_names": self.state_names,
                "state_units": self.state_units,
                "physical_state_of_charge_interval": tuple(bounds.tolist()),
                "property_law_mathematical_support": "real-line",
                "hysteresis": False,
                "capacity_fade": False,
                "parameter_estimation": False,
            },
        )
        problem_id = canonical_fingerprint(
            {
                "kind": "battery-exact-affine-ecm-state-space-problem",
                "prepared_id": self.prepared_id,
                "prior_id": prior_id,
                "sequence_id": sequence_id,
                "input_id": input_id,
            }
        )
        return StateSpaceProblem(
            model,
            observations,
            initial_time=times[..., 0],
            problem_id=problem_id,
            args=self,
            input_signal=input_signal,
        )

    def problem_from_record(
        self,
        record: BatteryTimeSeriesRecord,
        /,
        *,
        prior_mean: ArrayLike,
        prior_covariance: ArrayLike,
    ) -> StateSpaceProblem:
        """Build a problem from one canonical SI protocol record without sign guesses."""

        if not isinstance(record, BatteryTimeSeriesRecord):
            raise TypeError("record must be a BatteryTimeSeriesRecord.")
        if record.role is not BatteryRecordRole.PROTOCOL:
            raise ValueError(
                "Exact ECM estimation requires a protocol time-series record."
            )
        if not bool(jnp.all(record.current_mask)):
            raise ValueError(
                "Exact ECM estimation requires known current at every sample."
            )
        if not bool(jnp.all(record.temperature_mask)):
            raise ValueError(
                "Exact ECM estimation requires known temperature at every sample."
            )
        if len(record.segments) != 1:
            raise ValueError(
                "Exact held-current estimation refuses records with unobserved gap intervals."
            )
        return self.problem(
            record.time_s,
            record.current_a,
            record.voltage_v,
            record.temperature_k,
            prior_mean=prior_mean,
            prior_covariance=prior_covariance,
            voltage_mask=record.voltage_mask,
            case_ids=(record.record_id,),
            identity_id=record.content_fingerprint,
        )


class ExactAffineECMEstimationResult(StrictModule):
    """Native results with distinct exact-likelihood and physical-mean evidence.

    ``likelihood_valid`` refers to the globally affine Gaussian model. ``valid``
    additionally fails closed after a predicted, filtered, or smoothed mean exits
    the declared physical SOC interval; it does not claim bounded Gaussian mass.
    """

    filter_result: KalmanFilterResult
    smoother_result: KalmanSmootherResult
    innovation_diagnostics: KalmanInnovationDiagnostics
    filtered_voltage_residuals_v: Array
    predicted_mean_in_support: Array
    filtered_mean_in_support: Array
    smoothed_mean_in_support: Array
    likelihood_valid: Array
    physical_mean_support_valid: Array
    valid: Array
    step_valid: Array
    state_of_charge_interval: Array
    problem_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    estimation_id: str = eqx.field(static=True)

    @property
    def innovations_v(self) -> Array:
        return self.filter_result.innovations[..., 0]

    @property
    def innovation_covariances_v2(self) -> Array:
        return self.filter_result.innovation_covariances[..., 0, 0]

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.step_valid, axis=-1)

    @property
    def likelihood_successful(self) -> Array:
        return jnp.all(self.likelihood_valid | ~self.step_valid, axis=-1)


def estimate_exact_affine_ecm(
    problem: StateSpaceProblem,
    /,
    *,
    covariance_regularization: float = 0.0,
    filter_method: KalmanExecutionMethod = "auto",
    smoother_method: KalmanExecutionMethod = "auto",
    covariance_form: KalmanCovarianceForm = "covariance",
    raise_on_failure: bool = False,
) -> ExactAffineECMEstimationResult:
    """Invoke native Kalman filtering and RTS smoothing, then add domain evidence."""

    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    prepared = problem.args
    if not isinstance(prepared, PreparedExactAffineECMEstimation):
        raise TypeError("problem is not a prepared exact affine ECM problem.")
    if problem.model.model_id != prepared.prepared_id:
        raise ValueError("Exact affine ECM problem and prepared model IDs do not match.")

    filtered = kalman_filter(
        problem,
        covariance_regularization=covariance_regularization,
        raise_on_failure=raise_on_failure,
        method=filter_method,
        covariance_form=covariance_form,
    )
    smoothed = rts_smoother(
        filtered,
        method=smoother_method,
        covariance_form=covariance_form,
    )
    diagnostics = kalman_innovation_diagnostics(filtered)

    lower, upper = prepared.state_of_charge_interval

    def in_support(means: Array) -> Array:
        state_of_charge = means[..., 0]
        return (
            jnp.isfinite(state_of_charge)
            & (state_of_charge >= lower)
            & (state_of_charge <= upper)
        )

    predicted_support = in_support(filtered.predicted_means)
    filtered_support = in_support(filtered.filtered_means)
    smoothed_support = in_support(smoothed.means)
    local_support = predicted_support & filtered_support & smoothed_support
    step_valid = problem.observations.step_valid
    support_prefix = jnp.cumprod(
        jnp.where(step_valid, local_support, True).astype(jnp.int32),
        axis=-1,
    ).astype(bool)
    likelihood_valid = step_valid & filtered.valid & smoothed.valid
    valid = likelihood_valid & support_prefix

    if not isinstance(problem.input_signal, SampledStateSpaceInput):
        raise TypeError("Exact affine ECM problem requires its sampled held input.")
    current_a = problem.input_signal.values[..., 0]
    temperature_k = problem.input_signal.values[..., 1]
    temperature_delta = temperature_k - prepared.reference_temperature_k
    intercept = (
        prepared.ocv_coefficients_v[0]
        + temperature_delta * prepared.entropic_coefficients_v_per_k[0]
    )
    slope = (
        prepared.ocv_coefficients_v[1]
        + temperature_delta * prepared.entropic_coefficients_v_per_k[1]
    )
    filtered_states = filtered.filtered_means
    fitted_voltage = (
        intercept
        + slope * filtered_states[..., 0]
        + jnp.sum(filtered_states[..., 1:], axis=-1)
        + prepared.series_resistance_ohm * current_a
    )
    observed_voltage = problem.observations.values[..., 0]
    observed = problem.observations.observation_mask[..., 0] & step_valid
    residuals = jnp.where(observed, observed_voltage - fitted_voltage, 0.0)

    estimation_id = canonical_fingerprint(
        {
            "kind": "battery-exact-affine-ecm-estimation-result",
            "problem_id": problem.problem_id,
            "filter_method": filtered.execution_method,
            "smoother_method": smoothed.execution_method,
            "covariance_form": covariance_form,
            "covariance_regularization": float(covariance_regularization),
        }
    )
    result = ExactAffineECMEstimationResult(
        filter_result=filtered,
        smoother_result=smoothed,
        innovation_diagnostics=diagnostics,
        filtered_voltage_residuals_v=residuals,
        predicted_mean_in_support=predicted_support,
        filtered_mean_in_support=filtered_support,
        smoothed_mean_in_support=smoothed_support,
        likelihood_valid=likelihood_valid,
        physical_mean_support_valid=support_prefix,
        valid=valid,
        step_valid=step_valid,
        state_of_charge_interval=prepared.state_of_charge_interval,
        problem_id=problem.problem_id,
        prepared_id=prepared.prepared_id,
        estimation_id=estimation_id,
    )
    if raise_on_failure and not bool(jnp.all(result.successful)):
        raise RuntimeError(
            "Exact affine ECM estimation left its declared SOC support or failed."
        )
    return result


__all__ = [
    "estimate_exact_affine_ecm",
    "ExactAffineECMEstimationPlan",
    "ExactAffineECMEstimationResult",
    "GloballyAffineSOCPropertyLaw",
    "PreparedExactAffineECMEstimation",
    "TemperatureMode",
]
