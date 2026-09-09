#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Integral
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...optim import Bounds, FiniteAxis, FiniteProductSpace, MinimizationProblem
from ...qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    QualificationCriterion,
    QualificationEvidence,
    ReleaseIndex,
    ReleaseTrustPolicy,
    require_profile,
    SupportTuple,
    validate_qualification_causality,
)
from ...uq import experiment_design_objective, ExperimentDesignResult
from ._calibration import PreparedBatteryCalibration
from ._protocol import BatteryProtocolValues


BatteryOEDCriterion = Literal["d_optimal", "a_optimal", "e_optimal"]
_CRITERIA = ("d_optimal", "a_optimal", "e_optimal")


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _finite_ordered_pair(lower: float, upper: float, name: str, /) -> tuple[float, float]:
    lower_ = float(lower)
    upper_ = float(upper)
    if not np.isfinite(lower_) or not np.isfinite(upper_) or lower_ > upper_:
        raise ValueError(f"{name} bounds must be finite and ordered.")
    return lower_, upper_


class BatteryOEDModelContract(StrictModule, NonTrainableState):
    """Trusted release and criterion-bound smooth-derivative evidence."""

    smoothness_criterion: QualificationCriterion
    smoothness_start: CampaignStartRecord
    smoothness_observation: CampaignObservationRecord
    smoothness_evidence: QualificationEvidence
    derivative_criterion: QualificationCriterion
    derivative_start: CampaignStartRecord
    derivative_observation: CampaignObservationRecord
    derivative_evidence: QualificationEvidence
    model_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    release_index_id: str = eqx.field(static=True)
    release_subject_id: str = eqx.field(static=True)
    release_support_tuple_id: str = eqx.field(static=True)
    smoothness_criterion_id: str = eqx.field(static=True)
    derivative_criterion_id: str = eqx.field(static=True)
    smoothness_evidence_id: str = eqx.field(static=True)
    derivative_evidence_id: str = eqx.field(static=True)
    evidence_replay_ids: tuple[str, str] = eqx.field(static=True)
    admitted_at: int = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        model_id: str,
        release_index: ReleaseIndex,
        profile_id: str,
        support_tuple: SupportTuple,
        trust_policy: ReleaseTrustPolicy,
        smoothness_criterion: QualificationCriterion,
        smoothness_start: CampaignStartRecord,
        smoothness_observation: CampaignObservationRecord,
        smoothness_evidence: QualificationEvidence,
        derivative_criterion: QualificationCriterion,
        derivative_start: CampaignStartRecord,
        derivative_observation: CampaignObservationRecord,
        derivative_evidence: QualificationEvidence,
        /,
        *,
        at_time: int,
    ):
        model = _identifier(model_id, "Battery OED model ID")
        if not isinstance(release_index, ReleaseIndex):
            raise TypeError("release_index must be a ReleaseIndex.")
        if not isinstance(support_tuple, SupportTuple):
            raise TypeError("support_tuple must be a SupportTuple.")
        timestamp = int(at_time)
        profile = require_profile(
            release_index,
            profile_id,
            support_tuple,
            trust_policy,
            at_time=timestamp,
        )
        release_subject_id = canonical_fingerprint(
            {
                "kind": "battery-oed-release-subject",
                "model_id": model,
                "profile": {
                    "provider": profile.provider,
                    "name": profile.name,
                    "version": profile.version,
                },
                "support_tuple_id": support_tuple.support_tuple_id,
            }
        )
        specifications = (
            (
                "smoothness",
                smoothness_criterion,
                smoothness_start,
                smoothness_observation,
                smoothness_evidence,
                "battery-model-smoothness-defect",
                "battery-oed-local-smoothness",
            ),
            (
                "derivative",
                derivative_criterion,
                derivative_start,
                derivative_observation,
                derivative_evidence,
                "battery-model-directional-derivative-error",
                "battery-oed-fixed-path-derivative",
            ),
        )
        criteria: list[QualificationCriterion] = []
        starts: list[CampaignStartRecord] = []
        observations: list[CampaignObservationRecord] = []
        evidence_values: list[QualificationEvidence] = []
        required_subjects = {
            model,
            release_subject_id,
            support_tuple.support_tuple_id,
        }
        for (
            role,
            criterion,
            start,
            observation,
            evidence,
            metric,
            applicability,
        ) in specifications:
            if not isinstance(criterion, QualificationCriterion):
                raise TypeError(f"{role}_criterion must be a QualificationCriterion.")
            checked_criterion = QualificationCriterion.from_record(criterion.to_record())
            if (
                checked_criterion.support_tuple_id != support_tuple.support_tuple_id
                or checked_criterion.metric != metric
                or checked_criterion.unit != "1"
                or checked_criterion.aggregation != "maximum"
                or checked_criterion.comparison != "less-than-or-equal"
                or checked_criterion.applicability != applicability
            ):
                raise ValueError(
                    f"Battery OED {role} criterion has incompatible semantics."
                )
            if not checked_criterion.is_valid(timestamp):
                raise ValueError(f"Battery OED {role} criterion is not current.")
            if not isinstance(start, CampaignStartRecord):
                raise TypeError(f"{role}_start must be a CampaignStartRecord.")
            if not isinstance(observation, CampaignObservationRecord):
                raise TypeError(
                    f"{role}_observation must be a CampaignObservationRecord."
                )
            if not isinstance(evidence, QualificationEvidence):
                raise TypeError(f"{role}_evidence must be a QualificationEvidence.")
            checked_evidence = QualificationEvidence.from_record(evidence.to_record())
            if checked_evidence.evidence_kind not in ("scientific", "reference"):
                raise ValueError(
                    f"Battery OED {role} evidence must be scientific or reference."
                )
            if not checked_evidence.passed:
                raise ValueError(f"Battery OED {role} evidence must have passed.")
            if not checked_evidence.is_current(timestamp):
                raise ValueError(f"Battery OED {role} evidence is not current.")
            if checked_evidence.criteria_ids != (checked_criterion.criterion_id,):
                raise ValueError(
                    f"Battery OED {role} evidence must cite only its exact criterion."
                )
            if not required_subjects.issubset(checked_evidence.subject_ids):
                raise ValueError(
                    f"Battery OED {role} evidence must bind exact model, profile, and support."
                )
            if checked_evidence.replay_id != start.resolved_run_spec_id:
                raise ValueError(
                    f"Battery OED {role} evidence replay does not bind its run."
                )
            validate_qualification_causality(
                checked_criterion,
                start,
                observation,
                checked_evidence,
            )
            criteria.append(checked_criterion)
            starts.append(CampaignStartRecord.from_record(start.to_record()))
            observations.append(
                CampaignObservationRecord.from_record(observation.to_record())
            )
            evidence_values.append(checked_evidence)
        smooth_criterion, derivative_criterion = criteria
        smooth_start, derivative_start = starts
        smooth_observation, derivative_observation = observations
        smoothness, derivative = evidence_values
        if smoothness.topology != derivative.topology:
            raise ValueError("Battery OED evidence must bind one exact topology.")
        if smoothness.replay_id != derivative.replay_id:
            raise ValueError("Battery OED evidence must bind one exact replay.")
        if not {"smoothness", "derivative"}.issubset(profile.required_gates):
            raise ValueError(
                "Released battery OED profile must require smoothness and derivative gates."
            )
        gates = {value.gate: value for value in profile.release_evidence}
        for role, criterion, evidence in (
            ("smoothness", smooth_criterion, smoothness),
            ("derivative", derivative_criterion, derivative),
        ):
            gate = gates[role]
            required_citations = {criterion.criterion_id, evidence.evidence_id}
            if set(gate.evidence_ids) != required_citations:
                raise ValueError(
                    f"Battery OED {role} gate must cite its exact criterion and evidence."
                )
        self.smoothness_criterion = smooth_criterion
        self.smoothness_start = smooth_start
        self.smoothness_observation = smooth_observation
        self.smoothness_evidence = smoothness
        self.derivative_criterion = derivative_criterion
        self.derivative_start = derivative_start
        self.derivative_observation = derivative_observation
        self.derivative_evidence = derivative
        self.model_id = model
        self.release_subject_id = release_subject_id
        self.profile_id = profile.profile_id
        self.release_index_id = release_index.index_id
        self.release_support_tuple_id = support_tuple.support_tuple_id
        self.smoothness_criterion_id = smooth_criterion.criterion_id
        self.derivative_criterion_id = derivative_criterion.criterion_id
        self.smoothness_evidence_id = smoothness.evidence_id
        self.derivative_evidence_id = derivative.evidence_id
        self.evidence_replay_ids = (smoothness.replay_id, derivative.replay_id)
        self.admitted_at = timestamp
        self.contract_id = canonical_fingerprint(
            {
                "kind": "battery-oed-released-smooth-model",
                "model_id": model,
                "profile_id": profile.profile_id,
                "release_index_id": release_index.index_id,
                "support_tuple_id": support_tuple.support_tuple_id,
                "smoothness_criterion_id": smooth_criterion.criterion_id,
                "derivative_criterion_id": derivative_criterion.criterion_id,
                "smoothness_evidence_id": smoothness.evidence_id,
                "derivative_evidence_id": derivative.evidence_id,
                "evidence_replay_ids": list(self.evidence_replay_ids),
                "admitted_at": timestamp,
            }
        )


class BatteryOEDSupport(StrictModule, NonTrainableState):
    """Explicit SI support for fixed-topology battery design candidates."""

    current_lower_a: Array
    current_upper_a: Array
    voltage_lower_v: float = eqx.field(static=True)
    voltage_upper_v: float = eqx.field(static=True)
    temperature_lower_k: float = eqx.field(static=True)
    temperature_upper_k: float = eqx.field(static=True)
    duration_lower_s: float = eqx.field(static=True)
    duration_upper_s: float = eqx.field(static=True)
    stoichiometry_bounds: tuple[tuple[str, float, float], ...] = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        current_lower_a: ArrayLike,
        current_upper_a: ArrayLike,
        /,
        *,
        voltage_v: tuple[float, float],
        temperature_k: tuple[float, float],
        stoichiometry: Mapping[str, tuple[float, float]],
        duration_s: tuple[float, float],
    ):
        lower_input = np.asarray(current_lower_a)
        upper_input = np.asarray(current_upper_a)
        if lower_input.ndim > 1 or upper_input.ndim > 1:
            raise ValueError("Battery OED current bounds must be scalar or rank one.")
        if (
            np.iscomplexobj(lower_input)
            or np.iscomplexobj(upper_input)
            or not np.all(np.isfinite(lower_input))
            or not np.all(np.isfinite(upper_input))
        ):
            raise ValueError("Battery OED current bounds must be finite and real.")
        try_shape = np.broadcast_shapes(lower_input.shape, upper_input.shape)
        lower_host = np.broadcast_to(lower_input, try_shape).astype(float)
        upper_host = np.broadcast_to(upper_input, try_shape).astype(float)
        if np.any(lower_host > upper_host):
            raise ValueError("Battery OED current bounds must be ordered.")
        if not isinstance(stoichiometry, Mapping) or not stoichiometry:
            raise TypeError(
                "stoichiometry must be a non-empty component-to-bounds mapping."
            )
        stoichiometry_bounds = tuple(
            sorted(
                (
                    _identifier(component, "Stoichiometry component"),
                    *_finite_ordered_pair(bounds[0], bounds[1], "Stoichiometry"),
                )
                for component, bounds in stoichiometry.items()
            )
        )
        components = tuple(value[0] for value in stoichiometry_bounds)
        if len(set(components)) != len(components):
            raise ValueError("Battery OED stoichiometry components must be unique.")
        voltage = _finite_ordered_pair(*voltage_v, "Voltage")
        temperature = _finite_ordered_pair(*temperature_k, "Temperature")
        duration = _finite_ordered_pair(*duration_s, "Duration")
        if duration[0] <= 0.0:
            raise ValueError("Battery OED duration support must be strictly positive.")

        self.current_lower_a = jax.lax.stop_gradient(jnp.asarray(lower_host))
        self.current_upper_a = jax.lax.stop_gradient(jnp.asarray(upper_host))
        self.voltage_lower_v, self.voltage_upper_v = voltage
        self.temperature_lower_k, self.temperature_upper_k = temperature
        self.duration_lower_s, self.duration_upper_s = duration
        self.stoichiometry_bounds = stoichiometry_bounds
        self.support_id = canonical_fingerprint(
            {
                "kind": "battery-oed-support",
                "current_bounds": array_tree_fingerprint(
                    (self.current_lower_a, self.current_upper_a)
                ),
                "voltage_v": list(voltage),
                "temperature_k": list(temperature),
                "stoichiometry": [list(value) for value in stoichiometry_bounds],
                "duration_s": list(duration),
            }
        )

    def current_bounds(self, count: int, /) -> tuple[Array, Array]:
        """Materialize current support for the fixed number of pulse amplitudes."""

        shape = (int(count),)
        if self.current_lower_a.shape not in (
            (),
            shape,
        ) or self.current_upper_a.shape not in (
            (),
            shape,
        ):
            raise ValueError(
                "Battery OED current bounds must be scalar or match the current-step count."
            )
        return (
            jnp.broadcast_to(self.current_lower_a, shape),
            jnp.broadcast_to(self.current_upper_a, shape),
        )


class BatteryOEDResult(StrictModule):
    """Fixed-shape local information, feasibility, and replay evidence."""

    whitened_prediction: Array
    whitened_jacobian: Array
    fisher_information: Array
    information: Array
    criterion: ExperimentDesignResult
    simulation_valid: Array
    ledger_valid: Array
    current_valid: Array
    voltage_valid: Array
    temperature_valid: Array
    stoichiometry_valid: Array
    duration_valid: Array
    sensitivity_valid: Array
    valid: Array
    oed_plan_id: str = eqx.field(static=True)
    calibration_preparation_id: str = eqx.field(static=True)
    model_contract_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    experiment_plan_id: str = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)
    support_tuple_id: str = eqx.field(static=True)
    oed_support_id: str = eqx.field(static=True)
    criterion_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    battery_run_id: str | None = eqx.field(static=True)
    protocol_values_digest: str | None = eqx.field(static=True)
    initial_state_digest: str | None = eqx.field(static=True)
    parameters_digest: str | None = eqx.field(static=True)


class PreparedBatteryOED(StrictModule):
    """Prepared local sensitivity action over one fixed battery protocol topology."""

    calibration: PreparedBatteryCalibration
    nominal_position: PyTree[Any]
    support: BatteryOEDSupport
    model_contract: BatteryOEDModelContract
    prior_information: Array
    current_lower_a: Array
    current_upper_a: Array
    experiment_index: int = eqx.field(static=True)
    criterion_name: BatteryOEDCriterion = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    parameter_count: int = eqx.field(static=True)
    current_count: int = eqx.field(static=True)
    voltage_index: int = eqx.field(static=True)
    temperature_index: int = eqx.field(static=True)
    stoichiometry_indices: tuple[int, ...] = eqx.field(static=True)
    duration_valid: bool = eqx.field(static=True)
    criterion_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        calibration: PreparedBatteryCalibration,
        support: BatteryOEDSupport,
        model_contract: BatteryOEDModelContract,
        /,
        *,
        experiment_index: int = 0,
        nominal_position: PyTree[Any] | None = None,
        prior_information: ArrayLike | None = None,
        criterion: BatteryOEDCriterion = "d_optimal",
        regularization: float = 0.0,
    ):
        if not isinstance(calibration, PreparedBatteryCalibration):
            raise TypeError("calibration must be a PreparedBatteryCalibration.")
        if not isinstance(support, BatteryOEDSupport):
            raise TypeError("support must be a BatteryOEDSupport.")
        if not isinstance(model_contract, BatteryOEDModelContract):
            raise TypeError("model_contract must be a BatteryOEDModelContract.")
        if isinstance(experiment_index, bool) or not isinstance(
            experiment_index, Integral
        ):
            raise TypeError("experiment_index must be an integer.")
        index = int(experiment_index)
        if index < 0 or index >= len(calibration.plan.experiments):
            raise IndexError("experiment_index is outside the calibration plan.")
        if criterion not in _CRITERIA:
            raise ValueError(
                "criterion must be 'd_optimal', 'a_optimal', or 'e_optimal'."
            )
        regularization_ = float(regularization)
        if not np.isfinite(regularization_) or regularization_ < 0.0:
            raise ValueError("regularization must be finite and non-negative.")

        experiment = calibration.plan.experiments[index]
        protocol = experiment.prepared.plan.protocol
        if protocol.current_step_count < 1:
            raise ValueError("Battery OED requires at least one fixed current step.")
        if protocol.guard_count:
            raise ValueError(
                "Battery OED requires a fixed horizon without moving stop-event durations."
            )
        if experiment.prepared.plan.model.model_id != model_contract.model_id:
            raise ValueError(
                "Released smooth model identity does not match the calibration model."
            )
        evidence_subjects = {
            model_contract.model_id,
            model_contract.release_subject_id,
            model_contract.release_support_tuple_id,
            calibration.preparation_id,
            calibration.plan.calibration_plan_id,
            experiment.calibration_experiment_id,
            experiment.prepared.preparation_id,
            model_contract.evidence_replay_ids[0],
        }
        for role, evidence in (
            ("smoothness", model_contract.smoothness_evidence),
            ("derivative", model_contract.derivative_evidence),
        ):
            if set(evidence.subject_ids) != evidence_subjects:
                raise ValueError(
                    f"Battery OED {role} evidence does not bind the exact calibration run."
                )
            if evidence.topology != experiment.prepared.plan.experiment_plan_id:
                raise ValueError(
                    f"Battery OED {role} evidence does not bind the fixed topology."
                )
        point = (
            calibration.plan.parameter_space.initial
            if nominal_position is None
            else nominal_position
        )
        calibration.plan.parameter_space.constrain(point)
        flat_point, _ = ravel_pytree(point)
        parameter_count = int(flat_point.size)
        if parameter_count == 0:
            raise ValueError(
                "Battery OED requires at least one local parameter coordinate."
            )

        names = experiment.prepared.plan.outputs.names
        required = (
            "voltage_v",
            "temperature_k",
            *(f"stoichiometry:{value[0]}" for value in support.stoichiometry_bounds),
        )
        missing = tuple(name for name in required if name not in names)
        if missing:
            raise ValueError(
                f"Battery OED output selection is missing support channels {missing}."
            )
        units = experiment.prepared.output_units
        expected_units = ("V", "K", *("1" for _ in support.stoichiometry_bounds))
        indices = tuple(names.index(name) for name in required)
        actual_units = tuple(units[value] for value in indices)
        if actual_units != expected_units:
            raise ValueError(
                "Battery OED support channels must retain their canonical SI units."
            )

        lower, upper = support.current_bounds(protocol.current_step_count)
        durations = np.asarray(
            tuple(step.duration_s for step in protocol.steps), dtype=float
        )
        duration_valid = bool(
            np.all(durations >= support.duration_lower_s)
            and np.all(durations <= support.duration_upper_s)
        )
        if prior_information is None:
            prior = np.zeros((parameter_count, parameter_count), dtype=float)
        else:
            prior = np.asarray(prior_information, dtype=float)
        if prior.shape != (parameter_count, parameter_count):
            raise ValueError(
                "prior_information must be square in local parameter coordinates."
            )
        tolerance = 64.0 * np.finfo(prior.dtype).eps * max(1.0, np.linalg.norm(prior))
        if not np.all(np.isfinite(prior)) or not np.allclose(
            prior, prior.T, rtol=0.0, atol=tolerance
        ):
            raise ValueError("prior_information must be finite and symmetric.")
        prior = 0.5 * (prior + prior.T)
        if np.any(np.linalg.eigvalsh(prior) < -tolerance):
            raise ValueError("prior_information must be positive semidefinite.")

        criterion_id = canonical_fingerprint(
            {
                "kind": "battery-oed-criterion",
                "criterion": criterion,
                "regularization": regularization_,
                "prior_information": array_tree_fingerprint(prior),
            }
        )
        self.calibration = calibration
        self.nominal_position = point
        self.support = support
        self.model_contract = model_contract
        self.prior_information = jax.lax.stop_gradient(jnp.asarray(prior))
        self.current_lower_a = lower
        self.current_upper_a = upper
        self.experiment_index = index
        self.criterion_name = criterion
        self.regularization = regularization_
        self.parameter_count = parameter_count
        self.current_count = protocol.current_step_count
        self.voltage_index = indices[0]
        self.temperature_index = indices[1]
        self.stoichiometry_indices = indices[2:]
        self.duration_valid = duration_valid
        self.criterion_id = criterion_id
        self.problem_id = f"{calibration.problem_id}:oed"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "prepared-battery-oed",
                "calibration_preparation_id": calibration.preparation_id,
                "experiment_index": index,
                "experiment_plan_id": experiment.prepared.plan.experiment_plan_id,
                "model_contract_id": model_contract.contract_id,
                "support_id": support.support_id,
                "criterion_id": criterion_id,
                "nominal_position": array_tree_fingerprint(point),
            }
        )

    def _candidate(self, amplitudes: ArrayLike, /) -> tuple[Array, Array]:
        values = jnp.asarray(amplitudes)
        if jnp.iscomplexobj(values):
            raise TypeError("Battery OED amplitudes must be real-valued.")
        if values.shape != (self.current_count,):
            raise ValueError(
                "Battery OED amplitudes must match the fixed current-step count."
            )
        values = values.astype(jnp.result_type(values, float))
        finite = jnp.all(jnp.isfinite(values))
        safe = jnp.where(jnp.isfinite(values), values, 0.0)
        supported = (
            finite
            & jnp.all(safe >= self.current_lower_a)
            & jnp.all(safe <= self.current_upper_a)
        )
        return safe, supported

    def _run(self, amplitudes: Array, position: PyTree[Any], /):
        experiment = self.calibration.plan.experiments[self.experiment_index]
        physical = self.calibration.plan.parameter_space.constrain(position)
        branches = self.calibration.plan.projection.unpack(physical)
        protocol_values = BatteryProtocolValues(
            experiment.prepared.plan.protocol,
            amplitudes,
            experiment.protocol_values.stop_thresholds,
        )
        result = experiment.prepared.run(
            branches.shared_physical,
            branches.initial_conditions[self.experiment_index],
            protocol_values,
        )
        prediction = experiment._prediction(
            result, branches.nuisance[self.experiment_index]
        )
        selected = jnp.take(prediction.reshape(-1), experiment.retained_indices)
        whitened_prediction = experiment._whiten(selected)
        return whitened_prediction, result

    def sensitivity_action(self, amplitudes: ArrayLike, direction: ArrayLike, /) -> Array:
        """Apply the local whitened-prediction Jacobian to one flat direction."""

        candidate, _ = self._candidate(amplitudes)
        flat_point, unravel = ravel_pytree(self.nominal_position)
        vector = jnp.asarray(direction, dtype=flat_point.dtype)
        if vector.shape != (self.parameter_count,):
            raise ValueError(
                "Sensitivity direction must match local parameter coordinates."
            )

        def prediction(flat: Array, /) -> Array:
            traced_candidate = candidate + jnp.zeros_like(candidate) * jnp.sum(flat)
            return self._run(traced_candidate, unravel(flat))[0]

        return jax.jvp(prediction, (flat_point,), (vector,))[1]

    def whitened_jacobian(self, amplitudes: ArrayLike, /) -> Array:
        """Materialize the prepared local sensitivity action in parameter coordinates."""

        candidate, _ = self._candidate(amplitudes)
        flat_point, unravel = ravel_pytree(self.nominal_position)

        def prediction(flat: Array, /) -> Array:
            traced_candidate = candidate + jnp.zeros_like(candidate) * jnp.sum(flat)
            return self._run(traced_candidate, unravel(flat))[0]

        _, action = jax.linearize(prediction, flat_point)
        basis = jnp.eye(self.parameter_count, dtype=flat_point.dtype)
        return jax.vmap(action)(basis).T

    def evaluate(self, amplitudes: ArrayLike, /) -> BatteryOEDResult:
        """Evaluate one amplitude vector without converting failures into finite scores."""

        candidate, current_valid = self._candidate(amplitudes)
        whitened_prediction, run = self._run(candidate, self.nominal_position)
        jacobian = self.whitened_jacobian(candidate)
        sensitivity_valid = jnp.all(jnp.isfinite(whitened_prediction)) & jnp.all(
            jnp.isfinite(jacobian)
        )
        safe_jacobian = jnp.where(sensitivity_valid, jacobian, jnp.zeros_like(jacobian))
        fisher = safe_jacobian.T @ safe_jacobian
        information = fisher + self.prior_information
        criterion = experiment_design_objective(
            information,
            criterion=self.criterion_name,
            regularization=self.regularization,
        )

        outputs = run.outputs
        values = outputs.values
        output_valid = jnp.all(outputs.valid) & jnp.all(jnp.isfinite(values))
        simulation_valid = (
            jnp.asarray(run.successful, dtype=bool)
            & output_valid
            & ~jnp.asarray(run.termination.terminated, dtype=bool)
        )
        experiment = self.calibration.plan.experiments[self.experiment_index]
        ledger_valid = experiment._ledger_is_successful(run.ledger)
        voltage = values[:, self.voltage_index]
        temperature = values[:, self.temperature_index]
        voltage_valid = jnp.all(
            (voltage >= self.support.voltage_lower_v)
            & (voltage <= self.support.voltage_upper_v)
        )
        temperature_valid = jnp.all(
            (temperature >= self.support.temperature_lower_k)
            & (temperature <= self.support.temperature_upper_k)
        )
        stoichiometry_valid = jnp.asarray(True)
        for index, (_, lower, upper) in zip(
            self.stoichiometry_indices,
            self.support.stoichiometry_bounds,
            strict=True,
        ):
            values_ = values[:, index]
            stoichiometry_valid = stoichiometry_valid & jnp.all(
                (values_ >= lower) & (values_ <= upper)
            )
        duration_valid = jnp.asarray(self.duration_valid)
        valid = (
            simulation_valid
            & ledger_valid
            & current_valid
            & voltage_valid
            & temperature_valid
            & stoichiometry_valid
            & duration_valid
            & sensitivity_valid
            & criterion.valid
        )
        return BatteryOEDResult(
            whitened_prediction,
            jacobian,
            fisher,
            information,
            criterion,
            simulation_valid,
            ledger_valid,
            current_valid,
            voltage_valid,
            temperature_valid,
            stoichiometry_valid,
            duration_valid,
            sensitivity_valid,
            valid,
            self.plan_id,
            self.calibration.preparation_id,
            self.model_contract.contract_id,
            run.model_id,
            run.experiment_plan_id,
            run.protocol_id,
            run.support_tuple_id,
            self.support.support_id,
            self.criterion_id,
            self.model_contract.evidence_replay_ids[1],
            run.run_id,
            run.protocol_values_digest,
            run.initial_state_digest,
            run.parameters_digest,
        )


class BatteryOEDCandidateSet(StrictModule, NonTrainableState):
    """One correlated finite set of fixed-count current amplitude vectors."""

    amplitudes_a: Array
    candidate_ids: tuple[str, ...] = eqx.field(static=True)
    candidate_set_id: str = eqx.field(static=True)

    def __init__(self, amplitudes_a: ArrayLike, /):
        host = np.asarray(amplitudes_a)
        if host.ndim != 2 or host.shape[0] == 0 or host.shape[1] == 0:
            raise ValueError(
                "Battery OED candidate amplitudes must be a non-empty matrix."
            )
        if np.iscomplexobj(host) or not np.all(np.isfinite(host)):
            raise ValueError("Battery OED candidate amplitudes must be finite and real.")
        identifiers = tuple(
            canonical_fingerprint(
                {
                    "kind": "battery-oed-amplitude-candidate",
                    "amplitudes_a": array_tree_fingerprint(row),
                }
            )
            for row in host
        )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Battery OED candidate amplitudes must be unique.")
        self.amplitudes_a = jax.lax.stop_gradient(jnp.asarray(host, dtype=float))
        self.candidate_ids = identifiers
        self.candidate_set_id = canonical_fingerprint(
            {
                "kind": "battery-oed-candidate-set",
                "candidate_ids": list(identifiers),
            }
        )


class BatteryOEDEnumerationProblem(StrictModule):
    """Native finite-space/evaluator construction for exact candidate enumeration."""

    space: FiniteProductSpace
    evaluator: Callable[[Array], tuple[Array, Array]] = eqx.field(static=True)
    candidate_ids: tuple[str, ...] = eqx.field(static=True)
    candidate_set_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)


def _raw_continuous_value(prepared: PreparedBatteryOED, amplitudes: Array, /) -> Array:
    result = prepared.evaluate(amplitudes)
    return jnp.where(result.valid, -result.criterion.value, jnp.nan)


@eqx.filter_custom_jvp
def _continuous_value(prepared: PreparedBatteryOED, amplitudes: Array, /) -> Array:
    return _raw_continuous_value(prepared, amplitudes)


@_continuous_value.def_jvp
def _continuous_value_jvp(primals, tangents):
    prepared, amplitudes = primals
    _, amplitude_tangent = tangents
    tangent = (
        jnp.zeros_like(amplitudes) if amplitude_tangent is None else amplitude_tangent
    )
    return jax.jvp(
        lambda value: _raw_continuous_value(prepared, value),
        (amplitudes,),
        (tangent,),
    )


class _BatteryOEDContinuousObjective(StrictModule):
    prepared: PreparedBatteryOED

    def __call__(self, amplitudes: Array, dynamic_args: Any, /):
        del dynamic_args
        result = self.prepared.evaluate(jax.lax.stop_gradient(amplitudes))
        value = _continuous_value(self.prepared, amplitudes)
        return value, result


def prepare_battery_oed(
    calibration: PreparedBatteryCalibration,
    support: BatteryOEDSupport,
    model_contract: BatteryOEDModelContract,
    /,
    **kwargs: Any,
) -> PreparedBatteryOED:
    """Prepare local battery OED directly from a prepared calibration contract."""

    return PreparedBatteryOED(calibration, support, model_contract, **kwargs)


def evaluate_battery_oed(
    prepared: PreparedBatteryOED, amplitudes_a: ArrayLike, /
) -> BatteryOEDResult:
    """Evaluate one fixed-topology battery OED candidate."""

    if not isinstance(prepared, PreparedBatteryOED):
        raise TypeError("prepared must be a PreparedBatteryOED.")
    return prepared.evaluate(amplitudes_a)


def prepare_battery_oed_enumeration(
    prepared: PreparedBatteryOED,
    candidates: BatteryOEDCandidateSet | ArrayLike,
    /,
) -> BatteryOEDEnumerationProblem:
    """Construct inputs accepted directly by the native finite search runtime."""

    if not isinstance(prepared, PreparedBatteryOED):
        raise TypeError("prepared must be a PreparedBatteryOED.")
    candidate_set = (
        candidates
        if isinstance(candidates, BatteryOEDCandidateSet)
        else BatteryOEDCandidateSet(candidates)
    )
    if candidate_set.amplitudes_a.shape[1:] != (prepared.current_count,):
        raise ValueError(
            "Every discrete candidate must match the fixed current-step count."
        )
    space = FiniteProductSpace(FiniteAxis(candidate_set.amplitudes_a))
    problem_id = canonical_fingerprint(
        {
            "kind": "battery-oed-enumeration-problem",
            "oed_plan_id": prepared.plan_id,
            "candidate_set_id": candidate_set.candidate_set_id,
            "space_id": space.space_id,
        }
    )

    def evaluator(amplitudes: Array, /) -> tuple[Array, Array]:
        result = prepared.evaluate(amplitudes)
        score = jnp.where(result.valid, -result.criterion.value, jnp.nan)
        return score, result.valid

    return BatteryOEDEnumerationProblem(
        space,
        evaluator,
        candidate_set.candidate_ids,
        candidate_set.candidate_set_id,
        problem_id,
    )


def prepare_battery_oed_continuous_problem(
    prepared: PreparedBatteryOED, /
) -> MinimizationProblem:
    """Construct a native bounded differentiable amplitude minimization problem."""

    if not isinstance(prepared, PreparedBatteryOED):
        raise TypeError("prepared must be a PreparedBatteryOED.")
    return MinimizationProblem(
        _BatteryOEDContinuousObjective(prepared),
        has_aux=True,
        bounds=Bounds(prepared.current_lower_a, prepared.current_upper_a),
        problem_id=f"{prepared.problem_id}:continuous-amplitudes:{prepared.plan_id}",
    )


__all__ = [
    "BatteryOEDCandidateSet",
    "BatteryOEDCriterion",
    "BatteryOEDEnumerationProblem",
    "BatteryOEDModelContract",
    "BatteryOEDResult",
    "BatteryOEDSupport",
    "PreparedBatteryOED",
    "evaluate_battery_oed",
    "prepare_battery_oed",
    "prepare_battery_oed_continuous_problem",
    "prepare_battery_oed_enumeration",
]
