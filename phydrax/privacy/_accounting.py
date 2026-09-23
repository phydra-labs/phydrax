#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.util
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .._fingerprint import canonical_fingerprint, canonical_json
from .._validation import finite_real_scalar
from ._definition import _require_record, NeighboringRelation, PrivacyDefinition


class AccountingMethod(StrEnum):
    """Numerical accountant used to evaluate a mechanism trace."""

    PLD = "pld"
    RDP = "rdp"


def _exact_integer(value: object, name: str, /, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


@dataclass(frozen=True, slots=True)
class PrivacyBudget:
    """Maximum approximate-DP privacy loss permitted for one release scope."""

    epsilon: float
    delta: float
    budget_id: str = field(init=False)

    def __post_init__(self) -> None:
        epsilon = finite_real_scalar(self.epsilon, "epsilon")
        delta = finite_real_scalar(self.delta, "delta")
        if epsilon < 0.0:
            raise ValueError("epsilon must be non-negative.")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must lie in (0, 1).")
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(
            self, "budget_id", canonical_fingerprint(self._content_record())
        )

    def _content_record(self) -> dict[str, object]:
        return {"kind": "privacy-budget", "epsilon": self.epsilon, "delta": self.delta}

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "budget_id": self.budget_id}

    @classmethod
    def from_record(cls, record: Mapping[str, Any], /) -> PrivacyBudget:
        _require_record(
            record,
            "privacy-budget",
            frozenset(("epsilon", "delta")),
            optional=frozenset(("budget_id",)),
        )
        value = cls(float(record["epsilon"]), float(record["delta"]))
        recorded_id = record.get("budget_id")
        if recorded_id is not None and str(recorded_id) != value.budget_id:
            raise ValueError("Serialized privacy budget has an invalid content address.")
        return value


@dataclass(frozen=True, slots=True)
class MechanismTrace:
    """Content-addressed composition of one upstream DP event."""

    event_json: str
    repetitions: int = 1
    trace_id: str = field(init=False)

    def __post_init__(self) -> None:
        record = json.loads(self.event_json)
        if not isinstance(record, dict):
            raise TypeError("event_json must encode one DP event mapping.")
        canonical = canonical_json(record)
        repetitions = _exact_integer(self.repetitions, "repetitions", minimum=1)
        object.__setattr__(self, "event_json", canonical)
        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(
            self, "trace_id", canonical_fingerprint(self._content_record())
        )

    @classmethod
    def from_event(cls, event: object, /, *, repetitions: int = 1) -> MechanismTrace:
        return cls(canonical_json(dp_event_to_record(event)), repetitions)

    @classmethod
    def from_record(cls, record: Mapping[str, Any], /) -> MechanismTrace:
        _require_record(
            record,
            "mechanism-trace",
            frozenset(("event", "repetitions")),
            optional=frozenset(("trace_id",)),
        )
        event = record["event"]
        if not isinstance(event, Mapping):
            raise TypeError("Serialized mechanism event must be a mapping.")
        value = cls(
            canonical_json(dict(event)),
            _exact_integer(record["repetitions"], "repetitions", minimum=1),
        )
        recorded_id = record.get("trace_id")
        if recorded_id is not None and str(recorded_id) != value.trace_id:
            raise ValueError("Serialized mechanism trace has an invalid content address.")
        return value

    @property
    def event_record(self) -> dict[str, object]:
        value = json.loads(self.event_json)
        if not isinstance(value, dict):
            raise TypeError("Canonical event payload is not a mapping.")
        return value

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "mechanism-trace",
            "event": self.event_record,
            "repetitions": self.repetitions,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "trace_id": self.trace_id}


@dataclass(frozen=True, slots=True)
class PrivacyGuarantee:
    """Accounted approximate-DP upper bound for a mechanism trace."""

    epsilon: float
    delta: float
    definition_id: str
    trace_id: str
    accounting_method: AccountingMethod
    accountant_id: str
    guarantee_id: str = field(init=False)

    def __post_init__(self) -> None:
        epsilon = finite_real_scalar(self.epsilon, "accounted epsilon")
        delta = finite_real_scalar(self.delta, "accounted delta")
        if epsilon < 0.0 or not 0.0 < delta < 1.0:
            raise ValueError(
                "Accounted epsilon/delta are outside the approximate-DP domain."
            )
        if not self.definition_id or not self.trace_id or not self.accountant_id:
            raise ValueError("Definition, trace, and accountant identities are required.")
        if not isinstance(self.accounting_method, AccountingMethod):
            raise TypeError("accounting_method must be an AccountingMethod.")
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(
            self, "guarantee_id", canonical_fingerprint(self._content_record())
        )

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "privacy-guarantee",
            "epsilon": self.epsilon,
            "delta": self.delta,
            "definition_id": self.definition_id,
            "trace_id": self.trace_id,
            "accounting_method": self.accounting_method.value,
            "accountant_id": self.accountant_id,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "guarantee_id": self.guarantee_id}

    @classmethod
    def from_record(cls, record: Mapping[str, Any], /) -> PrivacyGuarantee:
        _require_record(
            record,
            "privacy-guarantee",
            frozenset(
                (
                    "epsilon",
                    "delta",
                    "definition_id",
                    "trace_id",
                    "accounting_method",
                    "accountant_id",
                )
            ),
            optional=frozenset(("guarantee_id",)),
        )
        value = cls(
            float(record["epsilon"]),
            float(record["delta"]),
            str(record["definition_id"]),
            str(record["trace_id"]),
            AccountingMethod(str(record["accounting_method"])),
            str(record["accountant_id"]),
        )
        recorded_id = record.get("guarantee_id")
        if recorded_id is not None and str(recorded_id) != value.guarantee_id:
            raise ValueError(
                "Serialized privacy guarantee has an invalid content address."
            )
        return value


def _dp_accounting() -> Any:
    if importlib.util.find_spec("dp_accounting") is None:
        raise ImportError(
            "Differential-privacy accounting requires phydrax[privacy-jax]."
        )
    return importlib.import_module("dp_accounting")


def _upstream_neighboring_relation(
    relation: NeighboringRelation,
    dp: Any,
) -> object:
    if relation is NeighboringRelation.ADD_OR_REMOVE_ONE:
        return dp.NeighboringRelation.ADD_OR_REMOVE_ONE
    if relation is NeighboringRelation.REPLACE_ONE:
        return dp.NeighboringRelation.REPLACE_ONE
    if relation is NeighboringRelation.REPLACE_SPECIAL:
        return dp.NeighboringRelation.REPLACE_SPECIAL
    raise ValueError(f"Unsupported neighboring relation {relation!r}.")


def dp_event_to_record(event: object, /) -> dict[str, object]:
    """Serialize only explicitly supported Google DP Accounting events."""
    dp = _dp_accounting()
    event_module = dp.dp_event
    if isinstance(event, dp.NoOpDpEvent):
        return {"kind": "no-op"}
    if isinstance(event, dp.NonPrivateDpEvent):
        return {"kind": "non-private"}
    if isinstance(event, dp.GaussianDpEvent):
        return {"kind": "gaussian", "noise_multiplier": float(event.noise_multiplier)}
    if isinstance(event, dp.LaplaceDpEvent):
        return {"kind": "laplace", "noise_multiplier": float(event.noise_multiplier)}
    if isinstance(event, event_module.DiscreteLaplaceDpEvent):
        return {
            "kind": "discrete-laplace",
            "noise_parameter": float(event.noise_parameter),
            "sensitivity": int(event.sensitivity),
        }
    if isinstance(event, dp.RandomizedResponseDpEvent):
        return {
            "kind": "randomized-response",
            "noise_parameter": float(event.noise_parameter),
            "num_buckets": int(event.num_buckets),
        }
    if isinstance(event, dp.PoissonSampledDpEvent):
        return {
            "kind": "poisson-sampled",
            "sampling_probability": float(event.sampling_probability),
            "event": dp_event_to_record(event.event),
        }
    if isinstance(event, dp.SampledWithReplacementDpEvent):
        return {
            "kind": "sampled-with-replacement",
            "source_dataset_size": int(event.source_dataset_size),
            "sample_size": int(event.sample_size),
            "event": dp_event_to_record(event.event),
        }
    if isinstance(event, dp.SampledWithoutReplacementDpEvent):
        return {
            "kind": "sampled-without-replacement",
            "source_dataset_size": int(event.source_dataset_size),
            "sample_size": int(event.sample_size),
            "event": dp_event_to_record(event.event),
        }
    if isinstance(event, dp.SelfComposedDpEvent):
        return {
            "kind": "self-composed",
            "count": int(event.count),
            "event": dp_event_to_record(event.event),
        }
    if isinstance(event, dp.ComposedDpEvent):
        return {
            "kind": "composed",
            "events": [dp_event_to_record(value) for value in event.events],
        }
    if isinstance(event, dp.SingleEpochTreeAggregationDpEvent):
        counts = event.step_counts
        if isinstance(counts, Sequence) and not isinstance(counts, (str, bytes)):
            step_counts: int | list[int] = [int(value) for value in counts]
        else:
            step_counts = int(counts)
        return {
            "kind": "single-epoch-tree-aggregation",
            "noise_multiplier": float(event.noise_multiplier),
            "step_counts": step_counts,
        }
    if isinstance(event, event_module.RepeatAndSelectDpEvent):
        return {
            "kind": "repeat-and-select",
            "event": dp_event_to_record(event.event),
            "mean": float(event.mean),
            "shape": float(event.shape),
        }
    if isinstance(event, event_module.ZCDpEvent):
        return {"kind": "zcdp", "rho": float(event.rho), "xi": float(event.xi)}
    if isinstance(event, dp.TruncatedSubsampledGaussianDpEvent):
        return {
            "kind": "truncated-subsampled-gaussian",
            "dataset_size": int(event.dataset_size),
            "sampling_probability": float(event.sampling_probability),
            "truncated_batch_size": int(event.truncated_batch_size),
            "noise_multiplier": float(event.noise_multiplier),
        }
    raise TypeError(
        f"Unsupported DP event type {type(event).__module__}.{type(event).__name__}."
    )


def _nested_event(record: Mapping[str, Any], name: str, /) -> object:
    nested = record[name]
    if not isinstance(nested, Mapping):
        raise TypeError(f"Serialized {name} must be a DP event mapping.")
    return dp_event_from_record(nested)


_EVENT_RECORD_FIELDS = {
    "no-op": frozenset(("kind",)),
    "non-private": frozenset(("kind",)),
    "gaussian": frozenset(("kind", "noise_multiplier")),
    "laplace": frozenset(("kind", "noise_multiplier")),
    "discrete-laplace": frozenset(("kind", "noise_parameter", "sensitivity")),
    "randomized-response": frozenset(("kind", "noise_parameter", "num_buckets")),
    "poisson-sampled": frozenset(("kind", "sampling_probability", "event")),
    "sampled-with-replacement": frozenset(
        ("kind", "source_dataset_size", "sample_size", "event")
    ),
    "sampled-without-replacement": frozenset(
        ("kind", "source_dataset_size", "sample_size", "event")
    ),
    "self-composed": frozenset(("kind", "count", "event")),
    "composed": frozenset(("kind", "events")),
    "single-epoch-tree-aggregation": frozenset(
        ("kind", "noise_multiplier", "step_counts")
    ),
    "repeat-and-select": frozenset(("kind", "event", "mean", "shape")),
    "zcdp": frozenset(("kind", "rho", "xi")),
    "truncated-subsampled-gaussian": frozenset(
        (
            "kind",
            "dataset_size",
            "sampling_probability",
            "truncated_batch_size",
            "noise_multiplier",
        )
    ),
}


def dp_event_from_record(record: Mapping[str, Any], /) -> object:
    """Reconstruct one event without dynamic module or class lookup."""
    if not isinstance(record, Mapping):
        raise TypeError("DP event record must be a mapping.")
    kind_value = record["kind"]
    if type(kind_value) is not str:
        raise TypeError("Serialized DP event kind must be a string.")
    kind = kind_value
    expected_fields = _EVENT_RECORD_FIELDS.get(kind)
    if expected_fields is None:
        raise ValueError(f"Unsupported serialized DP event kind {kind!r}.")
    actual_fields = frozenset(record)
    if actual_fields != expected_fields:
        raise ValueError(
            "Serialized DP event fields differ from the canonical contract; "
            f"missing={sorted(expected_fields - actual_fields)}, "
            f"unknown={sorted(actual_fields - expected_fields)}."
        )
    dp = _dp_accounting()
    event_module = dp.dp_event
    if kind == "no-op":
        return dp.NoOpDpEvent()
    if kind == "non-private":
        return dp.NonPrivateDpEvent()
    if kind == "gaussian":
        return dp.GaussianDpEvent(float(record["noise_multiplier"]))
    if kind == "laplace":
        return dp.LaplaceDpEvent(float(record["noise_multiplier"]))
    if kind == "discrete-laplace":
        return event_module.DiscreteLaplaceDpEvent(
            float(record["noise_parameter"]),
            _exact_integer(record["sensitivity"], "sensitivity", minimum=1),
        )
    if kind == "randomized-response":
        return dp.RandomizedResponseDpEvent(
            float(record["noise_parameter"]),
            _exact_integer(record["num_buckets"], "num_buckets", minimum=2),
        )
    if kind == "poisson-sampled":
        return dp.PoissonSampledDpEvent(
            float(record["sampling_probability"]), _nested_event(record, "event")
        )
    if kind == "sampled-with-replacement":
        return dp.SampledWithReplacementDpEvent(
            _exact_integer(
                record["source_dataset_size"], "source_dataset_size", minimum=1
            ),
            _exact_integer(record["sample_size"], "sample_size", minimum=1),
            _nested_event(record, "event"),
        )
    if kind == "sampled-without-replacement":
        return dp.SampledWithoutReplacementDpEvent(
            _exact_integer(
                record["source_dataset_size"], "source_dataset_size", minimum=1
            ),
            _exact_integer(record["sample_size"], "sample_size", minimum=1),
            _nested_event(record, "event"),
        )
    if kind == "self-composed":
        return dp.SelfComposedDpEvent(
            _nested_event(record, "event"),
            _exact_integer(record["count"], "count", minimum=1),
        )
    if kind == "composed":
        events = record["events"]
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            raise TypeError("Serialized composed events must be a sequence.")
        values = []
        for value in events:
            if not isinstance(value, Mapping):
                raise TypeError("Every serialized composed event must be a mapping.")
            values.append(dp_event_from_record(value))
        return dp.ComposedDpEvent(values)
    if kind == "single-epoch-tree-aggregation":
        counts = record["step_counts"]
        if isinstance(counts, Sequence) and not isinstance(counts, (str, bytes)):
            counts = [
                _exact_integer(value, "step_counts item", minimum=1) for value in counts
            ]
        else:
            counts = _exact_integer(counts, "step_counts", minimum=1)
        return dp.SingleEpochTreeAggregationDpEvent(
            float(record["noise_multiplier"]), counts
        )
    if kind == "repeat-and-select":
        return event_module.RepeatAndSelectDpEvent(
            _nested_event(record, "event"),
            float(record["mean"]),
            float(record["shape"]),
        )
    if kind == "zcdp":
        return dp.ZCDpEvent(float(record["rho"]), float(record["xi"]))
    if kind == "truncated-subsampled-gaussian":
        return dp.TruncatedSubsampledGaussianDpEvent(
            _exact_integer(record["dataset_size"], "dataset_size", minimum=1),
            float(record["sampling_probability"]),
            _exact_integer(
                record["truncated_batch_size"],
                "truncated_batch_size",
                minimum=1,
            ),
            float(record["noise_multiplier"]),
        )
    raise ValueError(f"Unsupported serialized DP event kind {kind!r}.")


def _trace_event(trace: MechanismTrace, dp: Any, /) -> object:
    event = dp_event_from_record(trace.event_record)
    if trace.repetitions == 1:
        return event
    return dp.SelfComposedDpEvent(event, trace.repetitions)


def _accountant(
    method: AccountingMethod,
    definition: PrivacyDefinition,
    dp: Any,
) -> tuple[Any, str]:
    neighboring = _upstream_neighboring_relation(definition.neighboring_relation, dp)
    if method is AccountingMethod.PLD:
        return (
            dp.pld.PLDAccountant(neighboring),
            "google-dp-accounting:0.6.0:pld-discretization-1e-4",
        )
    if method is AccountingMethod.RDP:
        return (
            dp.rdp.RdpAccountant(neighboring_relation=neighboring),
            "google-dp-accounting:0.6.0:rdp-default-orders",
        )
    raise ValueError(f"Unsupported accounting method {method!r}.")


def account_mechanism_traces(
    traces: Sequence[MechanismTrace],
    definition: PrivacyDefinition,
    budget: PrivacyBudget,
    /,
    *,
    method: AccountingMethod = AccountingMethod.PLD,
) -> PrivacyGuarantee:
    """Account an exact sequence of traces and enforce the supplied budget."""
    if not traces:
        raise ValueError("At least one mechanism trace is required.")
    if not isinstance(definition, PrivacyDefinition):
        raise TypeError("definition must be a PrivacyDefinition.")
    if not isinstance(budget, PrivacyBudget):
        raise TypeError("budget must be a PrivacyBudget.")
    dp = _dp_accounting()
    accountant, accountant_id = _accountant(method, definition, dp)
    events = [_trace_event(trace, dp) for trace in traces]
    composed = events[0] if len(events) == 1 else dp.ComposedDpEvent(events)
    if not accountant.supports(composed):
        raise ValueError(
            f"{accountant_id} does not support the supplied event under {definition.neighboring_relation.value}."
        )
    epsilon = float(accountant.compose(composed).get_epsilon(budget.delta))
    if not math.isfinite(epsilon):
        raise ValueError("The composed mechanism does not have a finite privacy bound.")
    tolerance = max(1e-10, abs(budget.epsilon) * 1e-8)
    if epsilon > budget.epsilon + tolerance:
        raise ValueError(
            f"Accounted epsilon {epsilon:.12g} exceeds budget {budget.epsilon:.12g}."
        )
    trace_id = (
        traces[0].trace_id
        if len(traces) == 1
        else canonical_fingerprint(
            {
                "kind": "composed-mechanism-traces",
                "traces": [value.to_record() for value in traces],
            }
        )
    )
    return PrivacyGuarantee(
        epsilon,
        budget.delta,
        definition.definition_id,
        trace_id,
        method,
        accountant_id,
    )


def account_mechanism_trace(
    trace: MechanismTrace,
    definition: PrivacyDefinition,
    budget: PrivacyBudget,
    /,
    *,
    method: AccountingMethod = AccountingMethod.PLD,
) -> PrivacyGuarantee:
    return account_mechanism_traces((trace,), definition, budget, method=method)


__all__ = [
    "AccountingMethod",
    "MechanismTrace",
    "PrivacyBudget",
    "PrivacyGuarantee",
    "account_mechanism_trace",
    "account_mechanism_traces",
    "dp_event_from_record",
    "dp_event_to_record",
]
