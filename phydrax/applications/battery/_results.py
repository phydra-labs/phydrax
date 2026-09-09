#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class BatteryRunStatus(IntEnum):
    """Fail-closed application disposition for one battery execution."""

    SUCCESS = 0
    DOMAIN_ERROR = 1
    NATIVE_SOLVE_FAILED = 2
    NONFINITE_OUTPUT = 3
    CAPACITY_EXCEEDED = 4
    MODEL_LEDGER_FAILED = 5


class BatteryModelOutput(StrictModule):
    """Adapter-produced observables and one domain-valid flag per sample."""

    values: Array
    domain_valid: Array

    def __init__(self, values: ArrayLike, domain_valid: ArrayLike, /):
        values_ = jnp.asarray(values)
        valid_ = jnp.asarray(domain_valid, dtype=bool)
        if values_.ndim < 1:
            raise ValueError(
                "Battery model output values require a final observable axis."
            )
        if valid_.shape != values_.shape[:-1]:
            raise ValueError(
                "Battery model output domain_valid must match every non-observable axis."
            )
        self.values = values_
        self.domain_valid = valid_


class BatterySelectedOutputs(StrictModule):
    """Selected SI observations with deterministic fill outside the valid prefix."""

    times_s: Array
    values: Array
    valid: Array
    names: tuple[str, ...] = eqx.field(static=True)
    units: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        times_s: ArrayLike,
        values: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        names: tuple[str, ...],
        units: tuple[str, ...],
    ):
        times = jnp.asarray(times_s)
        values_ = jnp.asarray(values)
        valid_ = jnp.asarray(valid, dtype=bool)
        if times.ndim != 1 or valid_.shape != times.shape:
            raise ValueError(
                "Selected battery times and valid mask must share one rank-one shape."
            )
        if values_.shape != times.shape + (len(names),):
            raise ValueError(
                "Selected battery output values do not match names and save times."
            )
        if len(units) != len(names) or len(set(names)) != len(names):
            raise ValueError(
                "Selected battery output names must be unique and have units."
            )
        self.times_s = times
        self.values = values_
        self.valid = valid_
        self.names = names
        self.units = units


class BatteryTermination(StrictModule):
    """Fixed-shape native event outcome and stable guard-reason topology."""

    terminated: Array
    time_s: Array
    reason_index: Array
    protocol_step_index: Array
    derivative_valid: Array
    reason_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        terminated: ArrayLike,
        time_s: ArrayLike,
        reason_index: ArrayLike,
        protocol_step_index: ArrayLike,
        /,
        *,
        reason_names: tuple[str, ...],
        derivative_valid: ArrayLike = True,
    ):
        terminated_ = jnp.asarray(terminated, dtype=bool)
        time = jnp.asarray(time_s)
        reason = jnp.asarray(reason_index, dtype=jnp.int32)
        step = jnp.asarray(protocol_step_index, dtype=jnp.int32)
        derivative = jnp.asarray(derivative_valid, dtype=bool)
        if any(
            value.shape != () for value in (terminated_, time, reason, step, derivative)
        ):
            raise ValueError("Battery termination fields must be scalar arrays.")
        if len(set(reason_names)) != len(reason_names):
            raise ValueError("Battery termination reason names must be unique.")
        self.terminated = terminated_
        self.time_s = time
        self.reason_index = reason
        self.protocol_step_index = step
        self.derivative_valid = derivative
        self.reason_names = reason_names

    @property
    def reason(self) -> str:
        """Return the host-readable winning reason, or normal completion."""
        index = int(np.asarray(self.reason_index))
        return "completed" if index < 0 else self.reason_names[index]


class BatteryDAERestartEvidence(StrictModule):
    """Both one-sided states and the actual consistency solve at a forcing restart."""

    time_s: Array
    state_before: Array
    state_rate_before: Array
    initialization: Any
    differential_state_unchanged: Array
    active: Array
    segment_index: int = eqx.field(static=True)
    input_policy_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return (~self.active) | (
            self.initialization.valid & self.differential_state_unchanged
        )


class BatteryDAEReplayEvidence(StrictModule):
    """Ordered native replay tapes plus every consistency/history-reset boundary."""

    initialization: Any
    segments: tuple[Any, ...]
    events: tuple[Any, ...]
    restarts: tuple[BatteryDAERestartEvidence, ...]
    segment_active: Array
    native_successful: Array
    replay_id: str = eqx.field(static=True)
    restart_policy: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        valid = self.native_successful & self.initialization.valid
        for restart in self.restarts:
            valid = valid & restart.successful
        return valid

    @property
    def derivative_valid(self) -> Array:
        valid = self.successful
        for active, events in zip(self.segment_active, self.events, strict=True):
            if events is not None:
                valid = valid & (
                    (~active) | jnp.all((~events.replay.active) | events.derivative_valid)
                )
        return valid


class BatteryDAESolution(StrictModule):
    """One-sided samples backed by complete, independently restarted native solves.

    Inactive segment slots are shape-only neutral storage, never numerical evidence.
    Native histories, status, initialization and event tapes remain on ``segments``.
    """

    times: Array
    states: Array
    state_rates: Array
    forcing_currents_a: Array
    valid: Array
    rate_valid: Array
    status: Array
    residual_norm: Array
    residual_threshold: Array
    differential_residual_norm: Array
    constraint_norm: Array
    segments: tuple[Any, ...]
    replay: BatteryDAEReplayEvidence
    termination_status: Array
    problem_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.replay.successful

    @property
    def events(self) -> tuple[Any, ...]:
        return self.replay.events

    @property
    def initialization(self) -> Any:
        return self.replay.initialization


class BatteryExperimentResult(StrictModule):
    """Native solve, selected observations, ledger, status, and exact provenance."""

    native_solution: Any
    outputs: BatterySelectedOutputs
    ledger: Any
    application_status: Array
    termination: BatteryTermination
    run_id: str | None = eqx.field(static=True)
    protocol_values_digest: str | None = eqx.field(static=True)
    initial_state_digest: str | None = eqx.field(static=True)
    parameters_digest: str | None = eqx.field(static=True)
    experiment_plan_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    support_tuple_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    numerical_admission_id: str | None = eqx.field(static=True)
    predictive_admission_id: str | None = eqx.field(static=True)
    distribution_id: str | None = eqx.field(static=True)
    validity_envelope_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        native_solution: Any,
        outputs: BatterySelectedOutputs,
        ledger: Any,
        application_status: ArrayLike,
        termination: BatteryTermination,
        /,
        *,
        run_id: str | None,
        protocol_values_digest: str | None,
        initial_state_digest: str | None,
        parameters_digest: str | None,
        experiment_plan_id: str,
        preparation_id: str,
        protocol_id: str,
        model_id: str,
        profile_id: str,
        support_tuple_id: str,
        problem_id: str,
        numerical_admission_id: str | None = None,
        predictive_admission_id: str | None = None,
        distribution_id: str | None = None,
        validity_envelope_id: str | None = None,
    ):
        status = jnp.asarray(application_status, dtype=jnp.int32)
        if status.shape != ():
            raise ValueError("Battery application status must be one scalar code.")
        if not isinstance(outputs, BatterySelectedOutputs):
            raise TypeError("outputs must be BatterySelectedOutputs.")
        if not isinstance(termination, BatteryTermination):
            raise TypeError("termination must be BatteryTermination.")
        admission_ids = (
            numerical_admission_id,
            predictive_admission_id,
            distribution_id,
            validity_envelope_id,
        )
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in admission_ids
        ):
            raise ValueError(
                "Battery admission identities must be non-empty strings or None."
            )
        if numerical_admission_id is None and predictive_admission_id is not None:
            raise ValueError("Predictive battery admission requires numerical admission.")
        if numerical_admission_id is not None and (
            distribution_id is None or validity_envelope_id is None
        ):
            raise ValueError(
                "Numerical battery admission requires distribution and envelope identities."
            )
        identifiers = (
            experiment_plan_id,
            preparation_id,
            protocol_id,
            model_id,
            profile_id,
            support_tuple_id,
            problem_id,
        )
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError(
                "Battery result provenance identities must be non-empty strings."
            )
        dynamic_identities = (
            run_id,
            protocol_values_digest,
            initial_state_digest,
            parameters_digest,
        )
        present = tuple(value is not None for value in dynamic_identities)
        if any(present) and not all(present):
            raise ValueError(
                "Battery run ID and dynamic value digests must be present together."
            )
        if all(present) and any(
            not isinstance(value, str) or not value for value in dynamic_identities
        ):
            raise ValueError("Concrete battery run identities must be non-empty strings.")
        self.native_solution = native_solution
        self.outputs = outputs
        self.ledger = ledger
        self.application_status = status
        self.numerical_admission_id = numerical_admission_id
        self.predictive_admission_id = predictive_admission_id
        self.distribution_id = distribution_id
        self.validity_envelope_id = validity_envelope_id
        self.termination = termination
        self.run_id = run_id
        self.protocol_values_digest = protocol_values_digest
        self.initial_state_digest = initial_state_digest
        self.parameters_digest = parameters_digest
        (
            self.experiment_plan_id,
            self.preparation_id,
            self.protocol_id,
            self.model_id,
            self.profile_id,
            self.support_tuple_id,
            self.problem_id,
        ) = identifiers

    @property
    def successful(self) -> Array:
        return self.application_status == int(BatteryRunStatus.SUCCESS)

    @property
    def admission_scope(self) -> str:
        """Distinguish development, numerical equation support and product prediction."""
        if self.numerical_admission_id is None:
            return "unreleased-development"
        if self.predictive_admission_id is None:
            return "numerical-equation-support-only"
        return "predictive-parameter-support"

    @property
    def evidence_ready(self) -> bool:
        """Whether concrete identities are available, not a scientific release claim."""
        return self.run_id is not None

    def require_evidence_identity(self, /) -> str:
        """Return the concrete run identity or refuse traced/unsealed execution."""
        if self.run_id is None:
            raise ValueError(
                "Battery evidence requires concrete parameter, protocol-value, and "
                "initial-state digests."
            )
        return self.run_id


__all__ = [
    "BatteryDAEReplayEvidence",
    "BatteryDAERestartEvidence",
    "BatteryDAESolution",
    "BatteryExperimentResult",
    "BatteryModelOutput",
    "BatteryRunStatus",
    "BatterySelectedOutputs",
    "BatteryTermination",
]
