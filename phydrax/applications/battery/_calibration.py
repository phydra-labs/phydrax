#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...observation import (
    CholeskyCovarianceAction,
    CoordinateLayout,
    PrecisionCovarianceAction,
)
from ...optim import CompositeLeastSquaresProblem, NonlinearLeastSquaresProblem
from ...uq import ParameterSpace, PosteriorProblem
from ._data import BatteryGroupSplit
from ._ecm import ThermalEquivalentCircuitAdapter, ThermalEquivalentCircuitLedger
from ._experiment import PreparedBatteryExperiment
from ._observations import (
    BatteryTimeSeriesRecord,
    CANONICAL_BATTERY_CHANNELS,
    CANONICAL_BATTERY_UNITS,
)
from ._protocol import BatteryProtocolValues
from ._results import BatteryExperimentResult
from ._spm import PrescribedCurrentSpmAdapter, SpmLedger
from ._spme_marquis2019 import Marquis2019SpmeAdapter, Marquis2019SpmeLedger
from ._spme_side_reactions import (
    BrosaPlanellaSpmeSeiAdapter,
    BrosaPlanellaSpmeSeiLedger,
)
from ._tspme_brosa_planella import (
    BrosaPlanellaTspmeAdapter,
    BrosaPlanellaTspmeLedger,
)


_COVARIANCE_ACTIONS = (CholeskyCovarianceAction, PrecisionCovarianceAction)
_BUILTIN_LEDGER_ADAPTERS = (
    ThermalEquivalentCircuitAdapter,
    PrescribedCurrentSpmAdapter,
    Marquis2019SpmeAdapter,
    BrosaPlanellaSpmeSeiAdapter,
    BrosaPlanellaTspmeAdapter,
)
_BUILTIN_LEDGER_TYPES = (
    ThermalEquivalentCircuitLedger,
    SpmLedger,
    Marquis2019SpmeLedger,
    BrosaPlanellaSpmeSeiLedger,
    BrosaPlanellaTspmeLedger,
)
_CHANNEL_INDEX = {name: index for index, name in enumerate(CANONICAL_BATTERY_CHANNELS)}
_CHANNEL_UNIT = dict(
    zip(CANONICAL_BATTERY_CHANNELS, CANONICAL_BATTERY_UNITS, strict=True)
)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _type_id(value: Any, /) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


class BatteryCalibrationParameterBranches(StrictModule):
    """Physical shared, experiment-initial, and experiment-nuisance branches."""

    shared_physical: PyTree[Any]
    initial_conditions: tuple[PyTree[Any], ...]
    nuisance: tuple[PyTree[Any], ...]

    def __init__(
        self,
        shared_physical: PyTree[Any],
        initial_conditions: Sequence[PyTree[Any]],
        nuisance: Sequence[PyTree[Any]],
        /,
    ):
        self.shared_physical = shared_physical
        self.initial_conditions = tuple(initial_conditions)
        self.nuisance = tuple(nuisance)


def _identity_unpack(value: PyTree[Any], /) -> BatteryCalibrationParameterBranches:
    if not isinstance(value, BatteryCalibrationParameterBranches):
        raise TypeError(
            "The default calibration projection requires "
            "BatteryCalibrationParameterBranches physical parameters."
        )
    return value


def _identity_pack(value: BatteryCalibrationParameterBranches, /) -> PyTree[Any]:
    return value


class BatteryCalibrationProjection(StrictModule, NonTrainableState):
    """Named inverse pair between a ParameterSpace tree and calibration branches."""

    unpack_fn: Callable[[PyTree[Any]], BatteryCalibrationParameterBranches] = eqx.field(
        static=True
    )
    pack_fn: Callable[[BatteryCalibrationParameterBranches], PyTree[Any]] = eqx.field(
        static=True
    )
    projection_id: str = eqx.field(static=True)

    def __init__(
        self,
        unpack: Callable[[PyTree[Any]], BatteryCalibrationParameterBranches],
        pack: Callable[[BatteryCalibrationParameterBranches], PyTree[Any]],
        /,
        *,
        projection_id: str,
    ):
        if not callable(unpack) or not callable(pack):
            raise TypeError(
                "Calibration projection pack and unpack values must be callable."
            )
        self.unpack_fn = unpack
        self.pack_fn = pack
        self.projection_id = _identifier(projection_id, "Calibration projection ID")

    def unpack(self, physical: PyTree[Any], /) -> BatteryCalibrationParameterBranches:
        branches = self.unpack_fn(physical)
        if not isinstance(branches, BatteryCalibrationParameterBranches):
            raise TypeError(
                "Calibration projection unpack must return "
                "BatteryCalibrationParameterBranches."
            )
        return branches

    def pack(self, branches: BatteryCalibrationParameterBranches, /) -> PyTree[Any]:
        if not isinstance(branches, BatteryCalibrationParameterBranches):
            raise TypeError("Calibration projection pack requires parameter branches.")
        return self.pack_fn(branches)


_DEFAULT_PROJECTION = BatteryCalibrationProjection(
    _identity_unpack,
    _identity_pack,
    projection_id="battery-calibration:parameter-branches:identity",
)


class BatteryCalibrationFailurePolicy(StrictModule, NonTrainableState):
    """Fail-closed interpretation of declared battery runtime evidence."""

    require_ledger_success: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, *, require_ledger_success: bool = True):
        if not isinstance(require_ledger_success, bool):
            raise TypeError("require_ledger_success must be boolean.")
        self.require_ledger_success = require_ledger_success
        self.policy_id = canonical_fingerprint(
            {
                "kind": "battery-calibration-failure-policy",
                "runtime_failure": "positive-infinite-residual",
                "posterior_failure": "negative-infinite-log-likelihood",
                "require_ledger_success": require_ledger_success,
            }
        )


def _resolve_channel_map(
    channel_map: Mapping[str, str] | Sequence[str] | Sequence[tuple[str, str]],
    /,
) -> tuple[tuple[str, str], ...]:
    if isinstance(channel_map, Mapping):
        entries = tuple(channel_map.items())
    else:
        raw = tuple(channel_map)
        if raw and all(isinstance(value, str) for value in raw):
            entries = tuple((value, value) for value in raw)
        else:
            entries = raw
    if not entries:
        raise ValueError("Battery calibration requires at least one observed channel.")
    resolved: list[tuple[str, str]] = []
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(
                "channel_map must contain observation/model channel pairs or channel names."
            )
        observed, model = entry
        observed_ = _identifier(observed, "Observed battery channel")
        model_ = _identifier(model, "Model battery output")
        if observed_ not in _CHANNEL_INDEX:
            allowed = ", ".join(CANONICAL_BATTERY_CHANNELS)
            raise ValueError(
                f"Unsupported observed battery channel {observed_!r}; expected one of {allowed}."
            )
        resolved.append((observed_, model_))
    observed_names = tuple(value[0] for value in resolved)
    model_names = tuple(value[1] for value in resolved)
    if len(set(observed_names)) != len(observed_names):
        raise ValueError("Observed calibration channels must be unique.")
    if len(set(model_names)) != len(model_names):
        raise ValueError("Mapped model calibration outputs must be unique.")
    return tuple(resolved)


def _protocol_values_from_observation(
    prepared: PreparedBatteryExperiment,
    observation: BatteryTimeSeriesRecord,
    /,
) -> BatteryProtocolValues:
    protocol = prepared.plan.protocol
    if protocol.guard_count:
        raise ValueError(
            "Guard thresholds cannot be inferred from observations; provide protocol_values."
        )
    times = np.asarray(observation.time_s, dtype=float)
    currents = np.asarray(observation.current_a, dtype=float)
    current_mask = np.asarray(observation.current_mask, dtype=bool)
    boundaries = np.asarray(protocol.boundary_times_s, dtype=float)
    side = "left" if protocol.node_side == "left" else "right"
    interval_indices = np.searchsorted(boundaries, times, side=side) - 1
    interval_indices = np.clip(interval_indices, 0, len(protocol.steps) - 1)
    amplitudes: list[float] = []
    for step_index, current_index in enumerate(
        np.asarray(protocol.interval_current_indices, dtype=int)
    ):
        selected = (interval_indices == step_index) & current_mask
        values = currents[selected]
        if current_index >= 0:
            if values.size == 0:
                raise ValueError(
                    "Each prescribed-current step needs a retained current observation "
                    "or explicit protocol_values."
                )
            if np.any(values != values[0]):
                raise ValueError(
                    "A current step is not constant in the observation; provide explicit "
                    "protocol_values rather than inferring an amplitude."
                )
            amplitudes.append(float(values[0]))
        elif values.size and np.any(values != 0.0):
            raise ValueError(
                "Observed passive terminal current must be zero during a rest step."
            )
    return BatteryProtocolValues(protocol, np.asarray(amplitudes, dtype=float))


def _channel_arrays(
    observation: BatteryTimeSeriesRecord,
    names: tuple[str, ...],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    values = (
        np.asarray(observation.current_a),
        np.asarray(observation.voltage_v),
        np.asarray(observation.temperature_k),
    )
    masks = (
        np.asarray(observation.current_mask, dtype=bool),
        np.asarray(observation.voltage_mask, dtype=bool),
        np.asarray(observation.temperature_mask, dtype=bool),
    )
    return (
        np.stack(tuple(values[_CHANNEL_INDEX[name]] for name in names), axis=-1),
        np.stack(tuple(masks[_CHANNEL_INDEX[name]] for name in names), axis=-1),
    )


def battery_calibration_coordinate_layout(
    observation: BatteryTimeSeriesRecord,
    channel_map: Mapping[str, str] | Sequence[str] | Sequence[tuple[str, str]],
    /,
) -> CoordinateLayout:
    """Return the exact retained sample-major covariance coordinate layout."""

    if not isinstance(observation, BatteryTimeSeriesRecord):
        raise TypeError("observation must be a BatteryTimeSeriesRecord.")
    mapping = _resolve_channel_map(channel_map)
    names = tuple(value[0] for value in mapping)
    _, mask = _channel_arrays(observation, names)
    labels = tuple(
        f"{observation.record_id}:sample-{sample_index}:{name}"
        for sample_index in range(mask.shape[0])
        for channel_index, name in enumerate(names)
        if mask[sample_index, channel_index]
    )
    if not labels:
        raise ValueError("Battery calibration experiment has no retained observations.")
    return CoordinateLayout(labels)


def _resolve_scales(
    scales: ArrayLike | Mapping[str, float], names: tuple[str, ...], /
) -> np.ndarray:
    if isinstance(scales, Mapping):
        if set(scales) != set(names):
            raise ValueError(
                "channel_scales mapping must exactly cover observed channels."
            )
        values = np.asarray(tuple(scales[name] for name in names), dtype=float)
    else:
        values = np.asarray(scales, dtype=float)
        if values.shape == ():
            values = np.broadcast_to(values, (len(names),)).copy()
    if values.shape != (len(names),):
        raise ValueError(
            "channel_scales must be scalar or provide one scale per channel."
        )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(
            "Battery calibration channel scales must be finite and positive."
        )
    return values


class BatteryCalibrationExperiment(StrictModule, NonTrainableState):
    """One prepared experiment with immutable masked SI observations and whitening."""

    prepared: PreparedBatteryExperiment
    observation: BatteryTimeSeriesRecord
    protocol_values: BatteryProtocolValues
    observed_values: Array
    retained_indices: Array
    required_sample_mask: Array
    coordinate_layout: CoordinateLayout
    channel_scales: Array | None
    covariance_action: CholeskyCovarianceAction | PrecisionCovarianceAction | None
    precision_cholesky: Array | None
    gaussian_log_normalizer: Array
    channel_names: tuple[str, ...] = eqx.field(static=True)
    model_output_names: tuple[str, ...] = eqx.field(static=True)
    output_indices: tuple[int, ...] = eqx.field(static=True)
    nuisance_model: Callable[[Array, PyTree[Any]], ArrayLike] | None = eqx.field(
        static=True
    )
    nuisance_model_id: str | None = eqx.field(static=True)
    ledger_success: Callable[[Any], ArrayLike] | None = eqx.field(static=True)
    ledger_success_id: str | None = eqx.field(static=True)
    residual_size: int = eqx.field(static=True)
    calibration_experiment_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedBatteryExperiment,
        observation: BatteryTimeSeriesRecord,
        /,
        *,
        channel_map: Mapping[str, str] | Sequence[str] | Sequence[tuple[str, str]],
        channel_scales: ArrayLike | Mapping[str, float] | None = None,
        covariance_action: CholeskyCovarianceAction
        | PrecisionCovarianceAction
        | None = None,
        protocol_values: BatteryProtocolValues | None = None,
        nuisance_model: Callable[[Array, PyTree[Any]], ArrayLike] | None = None,
        nuisance_model_id: str | None = None,
        ledger_success: Callable[[Any], ArrayLike] | None = None,
        ledger_success_id: str | None = None,
    ):
        if not isinstance(prepared, PreparedBatteryExperiment):
            raise TypeError("prepared must be a PreparedBatteryExperiment.")
        if not isinstance(observation, BatteryTimeSeriesRecord):
            raise TypeError("observation must be a BatteryTimeSeriesRecord.")
        save_times = np.asarray(prepared.plan.save_times_s, dtype=float)
        observation_times = np.asarray(observation.time_s, dtype=float)
        if not np.array_equal(save_times, observation_times):
            raise ValueError(
                "Prepared battery save times must exactly equal observation sample times."
            )
        mapping = _resolve_channel_map(channel_map)
        observed_names = tuple(value[0] for value in mapping)
        model_names = tuple(value[1] for value in mapping)
        available = prepared.plan.outputs.names
        if any(name not in available for name in model_names):
            missing = tuple(name for name in model_names if name not in available)
            raise ValueError(
                f"Prepared battery outputs do not contain mapped channels {missing}."
            )
        output_indices = tuple(available.index(name) for name in model_names)
        output_units = tuple(prepared.output_units[index] for index in output_indices)
        expected_units = tuple(_CHANNEL_UNIT[name] for name in observed_names)
        if output_units != expected_units:
            raise ValueError(
                "Mapped battery model outputs and observations must use identical SI units."
            )

        observed_matrix, retained_mask = _channel_arrays(observation, observed_names)
        retained_indices_host = np.flatnonzero(retained_mask.reshape(-1))
        if retained_indices_host.size == 0:
            raise ValueError(
                "Battery calibration experiment has no retained observations."
            )
        retained_observations = observed_matrix.reshape(-1)[retained_indices_host]
        required_samples = np.any(retained_mask, axis=-1)
        coordinate_layout = battery_calibration_coordinate_layout(observation, mapping)

        if (channel_scales is None) == (covariance_action is None):
            raise ValueError(
                "Provide exactly one of channel_scales or covariance_action for whitening."
            )
        scales_array: Array | None
        precision_cholesky: Array | None = None
        if channel_scales is not None:
            channel_scale_values = _resolve_scales(channel_scales, observed_names)
            retained_scales = np.broadcast_to(
                channel_scale_values, retained_mask.shape
            ).reshape(-1)[retained_indices_host]
            scales_array = jax.lax.stop_gradient(jnp.asarray(retained_scales))
            logdet = 2.0 * np.sum(np.log(retained_scales))
            action_id = canonical_fingerprint(
                {
                    "kind": "battery-channel-scale-whitening",
                    "channels": list(observed_names),
                    "scales": array_tree_fingerprint(channel_scale_values),
                }
            )
        else:
            if not isinstance(covariance_action, _COVARIANCE_ACTIONS):
                raise TypeError(
                    "covariance_action must be a CholeskyCovarianceAction or "
                    "PrecisionCovarianceAction."
                )
            if covariance_action.layout.layout_id != coordinate_layout.layout_id:
                raise ValueError(
                    "Covariance action layout must exactly match retained "
                    "sample-channel coordinates."
                )
            scales_array = None
            supplied_logdet = float(np.asarray(covariance_action.logdet_covariance))
            logdet = supplied_logdet
            action_id = covariance_action.action_id
            if isinstance(covariance_action, PrecisionCovarianceAction):
                precision = np.asarray(covariance_action.precision)
                try:
                    factor = np.linalg.cholesky(precision)
                except np.linalg.LinAlgError as error:
                    raise ValueError(
                        "Precision covariance action must be positive definite for whitening."
                    ) from error
                derived_logdet = float(-2.0 * np.sum(np.log(np.diag(factor))))
                tolerance = (
                    64.0 * np.finfo(precision.dtype).eps * max(1.0, abs(derived_logdet))
                )
                if (
                    not np.isfinite(supplied_logdet)
                    or abs(supplied_logdet - derived_logdet) > tolerance
                ):
                    raise ValueError(
                        "Precision covariance logdet must equal the determinant "
                        "implied by its positive-definite precision."
                    )
                logdet = derived_logdet
                precision_cholesky = jax.lax.stop_gradient(jnp.asarray(factor))

        values = (
            _protocol_values_from_observation(prepared, observation)
            if protocol_values is None
            else protocol_values
        )
        prepared.plan.protocol._validate_values(values)
        if nuisance_model is not None and not callable(nuisance_model):
            raise TypeError("nuisance_model must be callable or None.")
        if (nuisance_model is None) != (nuisance_model_id is None):
            raise ValueError(
                "nuisance_model and nuisance_model_id must either both be supplied or both omitted."
            )
        nuisance_id = (
            None
            if nuisance_model_id is None
            else _identifier(nuisance_model_id, "Nuisance model ID")
        )
        if ledger_success is not None and not callable(ledger_success):
            raise TypeError("ledger_success must be callable or None.")
        if (ledger_success is None) != (ledger_success_id is None):
            raise ValueError(
                "ledger_success and ledger_success_id must either both be supplied or both omitted."
            )
        ledger_id = (
            None
            if ledger_success_id is None
            else _identifier(ledger_success_id, "Ledger success ID")
        )
        if ledger_success is None and not isinstance(
            prepared.plan.model, _BUILTIN_LEDGER_ADAPTERS
        ):
            raise TypeError(
                "Unknown battery adapters must declare ledger_success and "
                "ledger_success_id during calibration preparation."
            )
        residual_size = int(retained_indices_host.size)
        self.prepared = prepared
        self.observation = observation
        self.protocol_values = values
        self.observed_values = jax.lax.stop_gradient(jnp.asarray(retained_observations))
        self.retained_indices = jax.lax.stop_gradient(
            jnp.asarray(retained_indices_host, dtype=jnp.int32)
        )
        self.required_sample_mask = jax.lax.stop_gradient(
            jnp.asarray(required_samples, dtype=bool)
        )
        self.coordinate_layout = coordinate_layout
        self.channel_scales = scales_array
        self.covariance_action = covariance_action
        self.precision_cholesky = precision_cholesky
        dtype = jnp.result_type(self.observed_values, float)
        self.gaussian_log_normalizer = jax.lax.stop_gradient(
            jnp.asarray(
                0.5 * (residual_size * np.log(2.0 * np.pi) + logdet),
                dtype=dtype,
            )
        )
        self.channel_names = observed_names
        self.model_output_names = model_names
        self.output_indices = output_indices
        self.nuisance_model = nuisance_model
        self.nuisance_model_id = nuisance_id
        self.ledger_success = ledger_success
        self.ledger_success_id = ledger_id
        self.residual_size = residual_size
        self.calibration_experiment_id = canonical_fingerprint(
            {
                "kind": "battery-calibration-experiment",
                "preparation_id": prepared.preparation_id,
                "record_id": observation.record_id,
                "observation_fingerprint": observation.content_fingerprint,
                "channel_map": [list(value) for value in mapping],
                "whitening_id": action_id,
                "protocol_values": array_tree_fingerprint(values),
                "nuisance_model_id": nuisance_id,
                "ledger_success_id": ledger_id,
            }
        )

    def _prediction(
        self, result: BatteryExperimentResult, nuisance: PyTree[Any], /
    ) -> Array:
        prediction = jnp.take(
            result.outputs.values,
            jnp.asarray(self.output_indices, dtype=jnp.int32),
            axis=-1,
        )
        if self.nuisance_model is not None:
            prediction = jnp.asarray(self.nuisance_model(prediction, nuisance))
        if prediction.shape != (self.observation.time_s.size, len(self.channel_names)):
            raise ValueError(
                "Nuisance-adjusted battery prediction must retain time/channel shape."
            )
        return prediction

    def _whiten(self, difference: Array, /) -> Array:
        if self.channel_scales is not None:
            return difference / self.channel_scales
        if isinstance(self.covariance_action, CholeskyCovarianceAction):
            return self.covariance_action.whiten(difference)
        if isinstance(self.covariance_action, PrecisionCovarianceAction):
            if self.precision_cholesky is None:
                raise RuntimeError(
                    "Prepared precision whitening lost its Cholesky factor."
                )
            return self.precision_cholesky.T @ difference
        raise RuntimeError("Battery calibration experiment has no whitening action.")

    def _ledger_is_successful(self, ledger: Any, /) -> Array:
        if isinstance(ledger, _BUILTIN_LEDGER_TYPES):
            successful = jnp.asarray(ledger.successful, dtype=bool)
            if successful.shape != ():
                raise ValueError("Battery ledger successful evidence must be scalar.")
        elif self.ledger_success is None:
            raise TypeError(
                "Battery adapter returned an unsupported ledger without an "
                "explicit ledger_success contract."
            )
        else:
            successful = jnp.asarray(True)
        if self.ledger_success is not None:
            declared = jnp.asarray(self.ledger_success(ledger), dtype=bool)
            if declared.shape != ():
                raise ValueError("ledger_success must return one scalar boolean.")
            successful = successful & declared
        return successful

    def evaluate(
        self,
        shared_physical: PyTree[Any],
        initial_condition: PyTree[Any],
        nuisance: PyTree[Any],
        /,
        *,
        require_ledger_success: bool,
    ) -> tuple[Array, Array, Array]:
        """Return mapped prediction, finite whitened residual, and success evidence."""

        result = self.prepared.run(
            shared_physical,
            initial_condition,
            self.protocol_values,
        )
        prediction = self._prediction(result, nuisance)
        selected_prediction = jnp.take(prediction.reshape(-1), self.retained_indices)
        whitened = self._whiten(selected_prediction - self.observed_values)
        finite = jnp.all(jnp.isfinite(prediction)) & jnp.all(jnp.isfinite(whitened))
        required_valid = jnp.all(
            (~self.required_sample_mask) | jnp.asarray(result.outputs.valid, dtype=bool)
        )
        successful = jnp.asarray(result.successful, dtype=bool) & required_valid & finite
        if require_ledger_success:
            successful = successful & self._ledger_is_successful(result.ledger)
        safe_residual = jnp.where(jnp.isfinite(whitened), whitened, 0.0)
        return prediction, safe_residual, successful


class BatteryCalibrationPlan(StrictModule):
    """Immutable multi-experiment calibration topology and parameter transformation."""

    experiments: tuple[BatteryCalibrationExperiment, ...]
    parameter_space: ParameterSpace
    projection: BatteryCalibrationProjection
    failure_policy: BatteryCalibrationFailurePolicy
    split_id: str = eqx.field(static=True)
    fit_partition: Literal["train", "calibration"] = eqx.field(static=True)
    parameter_space_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    calibration_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        experiments: Sequence[BatteryCalibrationExperiment],
        parameter_space: ParameterSpace,
        /,
        *,
        group_split: BatteryGroupSplit,
        fit_partition: Literal["train", "calibration"],
        projection: BatteryCalibrationProjection | None = None,
        failure_policy: BatteryCalibrationFailurePolicy | None = None,
        parameter_space_id: str | None = None,
        problem_id: str = "battery-calibration",
    ):
        experiments_ = tuple(experiments)
        if not experiments_ or any(
            not isinstance(value, BatteryCalibrationExperiment) for value in experiments_
        ):
            raise TypeError(
                "experiments must be a non-empty sequence of "
                "BatteryCalibrationExperiment values."
            )
        experiment_ids = tuple(value.calibration_experiment_id for value in experiments_)
        if len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("Battery calibration experiments must be unique.")
        if not isinstance(group_split, BatteryGroupSplit):
            raise TypeError("group_split must be a BatteryGroupSplit.")
        if fit_partition not in ("train", "calibration"):
            raise ValueError(
                "fit_partition must be 'train' or 'calibration'; test data "
                "cannot enter calibration objectives."
            )
        if fit_partition == "train":
            partition_record_ids = group_split.train_record_ids
            partition_cell_ids = group_split.train_cell_ids
        else:
            partition_record_ids = group_split.calibration_record_ids
            partition_cell_ids = group_split.calibration_cell_ids
        bindings = {binding.record_id: binding for binding in group_split.record_bindings}
        for experiment in experiments_:
            record = experiment.observation
            binding = bindings.get(record.record_id)
            if (
                binding is None
                or binding.cell_id != record.cell_id
                or binding.content_fingerprint != record.content_fingerprint
                or binding.raw_digest != record.raw_digest
            ):
                raise ValueError(
                    "Calibration observation does not match the split's "
                    "immutable record content binding."
                )
            if (
                record.record_id in group_split.test_record_ids
                or record.cell_id in group_split.test_cell_ids
            ):
                raise ValueError(
                    "Test-partition battery records and cells cannot enter "
                    "calibration objectives."
                )
            if (
                record.record_id not in partition_record_ids
                or record.cell_id not in partition_cell_ids
            ):
                raise ValueError(
                    "Every calibration record and cell must belong to the "
                    "declared fit_partition."
                )
        if not isinstance(parameter_space, ParameterSpace):
            raise TypeError("parameter_space must be a ParameterSpace.")
        projection_ = _DEFAULT_PROJECTION if projection is None else projection
        if not isinstance(projection_, BatteryCalibrationProjection):
            raise TypeError("projection must be a BatteryCalibrationProjection or None.")
        failure_ = (
            BatteryCalibrationFailurePolicy()
            if failure_policy is None
            else failure_policy
        )
        if not isinstance(failure_, BatteryCalibrationFailurePolicy):
            raise TypeError(
                "failure_policy must be a BatteryCalibrationFailurePolicy or None."
            )
        problem_id_ = _identifier(problem_id, "Battery calibration problem ID")

        initial_physical = parameter_space.constrain(parameter_space.initial)
        branches = projection_.unpack(initial_physical)
        _validate_branches(branches, len(experiments_))
        repacked = projection_.pack(branches)
        _validate_projection_roundtrip(initial_physical, repacked)
        for experiment, nuisance in zip(experiments_, branches.nuisance, strict=True):
            if experiment.nuisance_model is None and jax.tree_util.tree_leaves(nuisance):
                raise ValueError(
                    "A non-empty nuisance branch requires a declared nuisance_model."
                )

        if parameter_space_id is None:
            space_id = canonical_fingerprint(
                {
                    "kind": "battery-calibration-parameter-space",
                    "raw_shapes": [list(value) for value in parameter_space.raw_shapes],
                    "physical_shapes": [
                        list(value) for value in parameter_space.physical_shapes
                    ],
                    "structure": str(
                        jax.tree_util.tree_structure(parameter_space.initial)
                    ),
                    "arrays": array_tree_fingerprint(parameter_space),
                    "custom_log_prior": (
                        None
                        if parameter_space.custom_log_prior is None
                        else _type_id(parameter_space.custom_log_prior)
                    ),
                }
            )
        else:
            space_id = _identifier(parameter_space_id, "Parameter space ID")
        self.experiments = experiments_
        self.parameter_space = parameter_space
        self.projection = projection_
        self.failure_policy = failure_
        self.split_id = group_split.split_id
        self.fit_partition = fit_partition
        self.parameter_space_id = space_id
        self.problem_id = problem_id_
        self.calibration_plan_id = canonical_fingerprint(
            {
                "kind": "battery-calibration-plan",
                "experiments": list(experiment_ids),
                "parameter_space_id": space_id,
                "projection_id": projection_.projection_id,
                "failure_policy_id": failure_.policy_id,
                "split_id": group_split.split_id,
                "fit_partition": fit_partition,
                "problem_id": problem_id_,
            }
        )

    def prepare(self, /) -> PreparedBatteryCalibration:
        return PreparedBatteryCalibration(self)


def _validate_branches(
    branches: BatteryCalibrationParameterBranches, experiment_count: int, /
) -> None:
    if len(branches.initial_conditions) != experiment_count:
        raise ValueError(
            "Calibration initial_conditions must provide one branch per experiment."
        )
    if len(branches.nuisance) != experiment_count:
        raise ValueError("Calibration nuisance must provide one branch per experiment.")


def _validate_projection_roundtrip(
    original: PyTree[Any], repacked: PyTree[Any], /
) -> None:
    if jax.tree_util.tree_structure(original) != jax.tree_util.tree_structure(repacked):
        raise ValueError("Calibration projection pack/unpack changes PyTree structure.")
    original_leaves = jax.tree_util.tree_leaves(original)
    repacked_leaves = jax.tree_util.tree_leaves(repacked)
    for initial, recovered in zip(original_leaves, repacked_leaves, strict=True):
        initial_array = np.asarray(initial)
        recovered_array = np.asarray(recovered)
        if initial_array.shape != recovered_array.shape or not np.array_equal(
            initial_array, recovered_array
        ):
            raise ValueError(
                "Calibration projection pack/unpack must be an exact inverse."
            )


class PreparedBatteryCalibration(StrictModule):
    """Prepared fixed-shape residual and likelihood evaluation over search coordinates."""

    plan: BatteryCalibrationPlan
    residual_size: int = eqx.field(static=True)
    gaussian_log_normalizer: Array
    preparation_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(self, plan: BatteryCalibrationPlan, /):
        if not isinstance(plan, BatteryCalibrationPlan):
            raise TypeError("plan must be a BatteryCalibrationPlan.")
        residual_size = sum(value.residual_size for value in plan.experiments)
        self.plan = plan
        self.residual_size = residual_size
        self.gaussian_log_normalizer = jnp.sum(
            jnp.stack(tuple(value.gaussian_log_normalizer for value in plan.experiments))
        )
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "prepared-battery-calibration",
                "calibration_plan_id": plan.calibration_plan_id,
                "residual_size": residual_size,
            }
        )
        self.problem_id = plan.problem_id

    def _physical_evaluation(
        self, physical: PyTree[Any], /
    ) -> tuple[tuple[Array, ...], Array, Array]:
        branches = self.plan.projection.unpack(physical)
        _validate_branches(branches, len(self.plan.experiments))
        predictions: list[Array] = []
        residuals: list[Array] = []
        successes: list[Array] = []
        for experiment, initial, nuisance in zip(
            self.plan.experiments,
            branches.initial_conditions,
            branches.nuisance,
            strict=True,
        ):
            prediction, residual, successful = experiment.evaluate(
                branches.shared_physical,
                initial,
                nuisance,
                require_ledger_success=(self.plan.failure_policy.require_ledger_success),
            )
            predictions.append(prediction)
            residuals.append(residual)
            successes.append(successful)
        residual = jnp.concatenate(tuple(residuals))
        successful = jnp.all(jnp.stack(tuple(successes)))
        return tuple(predictions), residual, successful

    def physical_residual(self, physical: PyTree[Any], /) -> Array:
        """Return whitened residuals, or positive infinity for declared failures."""

        _, residual, successful = self._physical_evaluation(physical)
        return jnp.where(successful, residual, jnp.full_like(residual, jnp.inf))

    def residual(self, position: PyTree[Any], /) -> Array:
        """Evaluate the whitened residual in ParameterSpace search coordinates."""

        physical = self.plan.parameter_space.constrain(position)
        return self.physical_residual(physical)

    def physical_log_likelihood(self, physical: PyTree[Any], /) -> Array:
        """Evaluate the explicitly normalized Gaussian likelihood."""

        _, residual, successful = self._physical_evaluation(physical)
        value = -0.5 * jnp.sum(residual * residual) - self.gaussian_log_normalizer
        return jnp.where(successful, value, jnp.asarray(-jnp.inf, dtype=value.dtype))

    def physical_prediction(self, physical: PyTree[Any], /) -> tuple[Array, ...]:
        predictions, _, _ = self._physical_evaluation(physical)
        return predictions


class _CalibrationSearchResidual(StrictModule):
    prepared: PreparedBatteryCalibration

    def __call__(self, position: PyTree[Any], args: Any, /) -> Array:
        del args
        return self.prepared.residual(position)


class _CalibrationNegativeLogPrior(StrictModule):
    parameter_space: ParameterSpace

    def __call__(self, position: PyTree[Any], args: Any, /) -> Array:
        del args
        return -self.parameter_space.unconstrained_log_prior(position)


def _prepared(
    value: BatteryCalibrationPlan | PreparedBatteryCalibration, /
) -> PreparedBatteryCalibration:
    if isinstance(value, PreparedBatteryCalibration):
        return value
    if isinstance(value, BatteryCalibrationPlan):
        return value.prepare()
    raise TypeError("Expected a BatteryCalibrationPlan or PreparedBatteryCalibration.")


def prepare_battery_least_squares(
    calibration: BatteryCalibrationPlan | PreparedBatteryCalibration,
    /,
    *,
    include_prior: bool = False,
) -> NonlinearLeastSquaresProblem | CompositeLeastSquaresProblem:
    """Build a native least-squares problem in unconstrained search coordinates."""

    if not isinstance(include_prior, bool):
        raise TypeError("include_prior must be boolean.")
    prepared = _prepared(calibration)
    residual = _CalibrationSearchResidual(prepared)
    if include_prior:
        return CompositeLeastSquaresProblem(
            residual,
            _CalibrationNegativeLogPrior(prepared.plan.parameter_space),
            problem_id=prepared.problem_id,
        )
    return NonlinearLeastSquaresProblem(
        residual,
        problem_id=prepared.problem_id,
    )


def prepare_battery_posterior(
    calibration: BatteryCalibrationPlan | PreparedBatteryCalibration,
    /,
) -> PosteriorProblem:
    """Build a native transformed posterior with explicit Gaussian normalization."""

    prepared = _prepared(calibration)

    def log_likelihood(physical):
        return prepared.physical_log_likelihood(physical)

    def predict(physical):
        return prepared.physical_prediction(physical)

    def gauss_newton_residual(physical):
        return prepared.physical_residual(physical)

    return PosteriorProblem(
        prepared.plan.parameter_space,
        log_likelihood,
        predict=predict,
        gauss_newton_residual=gauss_newton_residual,
    )


__all__ = [
    "BatteryCalibrationExperiment",
    "BatteryCalibrationFailurePolicy",
    "BatteryCalibrationParameterBranches",
    "BatteryCalibrationPlan",
    "BatteryCalibrationProjection",
    "PreparedBatteryCalibration",
    "battery_calibration_coordinate_layout",
    "prepare_battery_least_squares",
    "prepare_battery_posterior",
]
