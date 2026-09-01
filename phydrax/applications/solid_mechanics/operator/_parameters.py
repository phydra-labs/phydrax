#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._frozendict import frozendict
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


MechanicsParameterRole = Literal[
    "geometry",
    "material",
    "load",
    "boundary",
    "constraint",
    "history",
]
MechanicsParameterKind = Literal[
    "continuous",
    "integer",
    "discrete",
    "categorical",
]
MechanicsParameterWeightKind = Literal["equal", "probability", "importance"]


def _json_value(value: Any, /) -> Any:
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Parameter metadata values must be finite.")
        return value
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Parameter values must be numeric arrays or JSON scalars.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Numeric parameter values must be finite.")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def _condition_value(value: Any, /) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("Conditional parent parameters must be scalar.")
    return array.item()


class MechanicsParameterField(StrictModule, NonTrainableState):
    """One named scalar or tensor parameter with explicit support semantics."""

    name: str = eqx.field(static=True)
    role: MechanicsParameterRole = eqx.field(static=True)
    kind: MechanicsParameterKind = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    lower: Array | None
    upper: Array | None
    support: tuple[Any, ...] = eqx.field(static=True)
    active_when: frozendict[str, tuple[Any, ...]] = eqx.field(static=True)
    unit: str | None = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        role: MechanicsParameterRole,
        kind: MechanicsParameterKind = "continuous",
        shape: Sequence[int] = (),
        lower: Any | None = None,
        upper: Any | None = None,
        support: Sequence[Any] = (),
        active_when: Mapping[str, Sequence[Any]] | None = None,
        unit: str | None = None,
    ):
        resolved_name = str(name)
        if not resolved_name:
            raise ValueError("Mechanics parameter field names must be non-empty.")
        if role not in (
            "geometry",
            "material",
            "load",
            "boundary",
            "constraint",
            "history",
        ):
            raise ValueError("Unknown mechanics parameter field role.")
        if kind not in ("continuous", "integer", "discrete", "categorical"):
            raise ValueError("Unknown mechanics parameter field kind.")
        resolved_shape = tuple(int(size) for size in shape)
        if any(size <= 0 for size in resolved_shape):
            raise ValueError("Mechanics parameter field shape entries must be positive.")
        choices = tuple(support)
        if kind in ("discrete", "categorical"):
            if resolved_shape:
                raise ValueError(
                    "Discrete and categorical parameter fields must be scalar."
                )
            if not choices or len(
                {_json_value(value).__repr__() for value in choices}
            ) != len(choices):
                raise ValueError(
                    "Discrete parameter support must be non-empty and unique."
                )
            if lower is not None or upper is not None:
                raise ValueError(
                    "Discrete parameter fields do not accept interval bounds."
                )
            lower_array = None
            upper_array = None
        else:
            if choices:
                raise ValueError(
                    "Continuous and integer fields do not accept discrete support."
                )
            lower_array = (
                None
                if lower is None
                else jnp.broadcast_to(jnp.asarray(lower), resolved_shape or ())
            )
            upper_array = (
                None
                if upper is None
                else jnp.broadcast_to(jnp.asarray(upper), resolved_shape or ())
            )
            for label, bound in (("lower", lower_array), ("upper", upper_array)):
                if bound is not None and bool(jnp.any(~jnp.isfinite(bound))):
                    raise ValueError(f"Parameter {label} bound must be finite.")
            if (
                lower_array is not None
                and upper_array is not None
                and bool(jnp.any(lower_array > upper_array))
            ):
                raise ValueError("Parameter lower bounds cannot exceed upper bounds.")

        condition_values: dict[str, tuple[Any, ...]] = {}
        for parent, allowed in ({} if active_when is None else active_when).items():
            parent_name = str(parent)
            normalized_allowed = tuple(_condition_value(value) for value in allowed)
            if (
                not parent_name
                or not normalized_allowed
                or len({_json_value(value).__repr__() for value in normalized_allowed})
                != len(normalized_allowed)
            ):
                raise ValueError(
                    "Conditional parameters require named parents and unique allowed values."
                )
            condition_values[parent_name] = normalized_allowed
        conditions = frozendict(condition_values)
        if any(not parent or not allowed for parent, allowed in conditions.items()):
            raise ValueError(
                "Conditional parameters require named parents and allowed values."
            )
        resolved_unit = None if unit is None else str(unit)
        if resolved_unit == "":
            raise ValueError("Parameter units must be non-empty when provided.")
        payload = {
            "kind": "mechanics-parameter-field",
            "role": role,
            "name": resolved_name,
            "parameter_kind": kind,
            "shape": list(resolved_shape),
            "lower": None if lower_array is None else _json_value(lower_array),
            "upper": None if upper_array is None else _json_value(upper_array),
            "support": [_json_value(value) for value in choices],
            "active_when": {
                parent: [_json_value(value) for value in allowed]
                for parent, allowed in conditions.items()
            },
            "unit": resolved_unit,
        }
        self.name = resolved_name
        self.role = role
        self.kind = kind
        self.shape = resolved_shape
        self.lower = lower_array
        self.upper = upper_array
        self.support = choices
        self.active_when = conditions
        self.unit = resolved_unit
        self.field_id = canonical_fingerprint(payload)

    def is_active(self, resolved_values: Mapping[str, Any], /) -> bool:
        for parent, allowed in self.active_when.items():
            if parent not in resolved_values:
                raise ValueError(
                    f"Conditional parent {parent!r} must precede field {self.name!r}."
                )
            parent_value = _condition_value(resolved_values[parent])
            if parent_value not in allowed:
                return False
        return True

    def validate(self, value: Any, /) -> Any:
        """Return one immutable normalized value or reject it outside support."""
        if self.kind in ("discrete", "categorical"):
            scalar = _condition_value(value)
            if scalar not in self.support:
                raise ValueError(
                    f"Parameter {self.name!r} lies outside its discrete support."
                )
            return scalar
        array = jnp.asarray(value)
        if array.shape != self.shape:
            raise ValueError(
                f"Parameter {self.name!r} must have shape {self.shape}; got {array.shape}."
            )
        if bool(jnp.any(~jnp.isfinite(array))):
            raise ValueError(f"Parameter {self.name!r} must be finite.")
        if self.kind == "integer" and bool(jnp.any(array != jnp.round(array))):
            raise ValueError(f"Parameter {self.name!r} must be integer-valued.")
        if self.lower is not None and bool(jnp.any(array < self.lower)):
            raise ValueError(f"Parameter {self.name!r} is below its support.")
        if self.upper is not None and bool(jnp.any(array > self.upper)):
            raise ValueError(f"Parameter {self.name!r} is above its support.")
        return array


class MechanicsParameterSpec(StrictModule, NonTrainableState):
    """Ordered joint parameter law, including hierarchical activation rules."""

    fields: tuple[MechanicsParameterField, ...]
    spec_id: str = eqx.field(static=True)
    spec_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        fields: Sequence[MechanicsParameterField],
        /,
        *,
        spec_id: str | None = None,
    ):
        resolved = tuple(fields)
        if not resolved:
            raise ValueError("MechanicsParameterSpec requires at least one field.")
        if any(not isinstance(field, MechanicsParameterField) for field in resolved):
            raise TypeError(
                "Mechanics parameter specs contain MechanicsParameterField values."
            )
        names = tuple(field.name for field in resolved)
        if len(set(names)) != len(names):
            raise ValueError("Mechanics parameter field names must be unique.")
        preceding: dict[str, MechanicsParameterField] = {}
        for field in resolved:
            unknown = set(field.active_when) - set(preceding)
            if unknown:
                raise ValueError(
                    f"Conditional field {field.name!r} has nonpreceding parents {sorted(unknown)}."
                )
            for parent_name, allowed in field.active_when.items():
                parent = preceding[parent_name]
                if parent.shape:
                    raise ValueError(
                        f"Conditional parent {parent_name!r} must be scalar."
                    )
                for value in allowed:
                    parent.validate(value)
            preceding[field.name] = field
        derived = canonical_fingerprint(
            {
                "kind": "mechanics-parameter-spec",
                "fields": [field.field_id for field in resolved],
            }
        )
        identifier = derived if spec_id is None else str(spec_id)
        if not identifier:
            raise ValueError("Mechanics parameter spec IDs must be non-empty.")
        self.fields = resolved
        self.spec_id = identifier
        self.spec_fingerprint = derived

    @property
    def field_by_name(self) -> frozendict[str, MechanicsParameterField]:
        return frozendict({field.name: field for field in self.fields})

    def normalize(self, values: Mapping[str, Any], /) -> frozendict[str, Any]:
        supplied = {str(name): value for name, value in values.items()}
        if any(not name for name in supplied):
            raise ValueError("Mechanics parameter value names must be non-empty.")
        resolved: dict[str, Any] = {}
        for field in self.fields:
            active = field.is_active(resolved)
            present = field.name in supplied
            if active and not present:
                raise ValueError(f"Active parameter {field.name!r} is missing.")
            if not active and present:
                raise ValueError(
                    f"Inactive conditional parameter {field.name!r} must be absent."
                )
            if active:
                resolved[field.name] = field.validate(supplied[field.name])
        unknown = set(supplied) - set(resolved)
        if unknown:
            raise ValueError(
                f"Unknown or inactive mechanics parameters: {sorted(unknown)}."
            )
        return frozendict(resolved)

    def contains(self, values: Mapping[str, Any], /) -> bool:
        try:
            self.normalize(values)
        except (TypeError, ValueError):
            return False
        return True

    def stratum_id(self, values: Mapping[str, Any], /) -> str:
        normalized = self.normalize(values)
        strata = {
            field.name: (
                _json_value(normalized[field.name])
                if field.kind in ("discrete", "categorical")
                else {"active": True}
            )
            for field in self.fields
            if field.name in normalized
            and (field.kind in ("discrete", "categorical") or field.active_when)
        }
        return canonical_fingerprint(
            {
                "kind": "mechanics-parameter-stratum",
                "spec_fingerprint": self.spec_fingerprint,
                "values": strata,
            }
        )


class MechanicsParameterRealization(StrictModule, NonTrainableState):
    """One supported joint draw with an exact probability or importance mass."""

    spec: MechanicsParameterSpec
    values: frozendict[str, Any]
    probability_weight: float | None = eqx.field(static=True)
    importance_weight: float | None = eqx.field(static=True)
    case_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    stratum_id: str = eqx.field(static=True)
    realization_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        spec: MechanicsParameterSpec,
        values: Mapping[str, Any],
        /,
        *,
        probability_weight: float | None = None,
        importance_weight: float | None = None,
        case_id: str | None = None,
        realization_id: str | None = None,
        stratum_id: str | None = None,
    ):
        if not isinstance(spec, MechanicsParameterSpec):
            raise TypeError("spec must be a MechanicsParameterSpec.")
        if probability_weight is not None and importance_weight is not None:
            raise ValueError(
                "A realization cannot carry both probability and importance weight."
            )
        for label, weight in (
            ("probability", probability_weight),
            ("importance", importance_weight),
        ):
            if weight is not None and (
                not math.isfinite(float(weight)) or float(weight) <= 0.0
            ):
                raise ValueError(
                    f"Mechanics {label} weights must be finite and positive."
                )
        normalized = spec.normalize(values)
        value_payload = {name: _json_value(value) for name, value in normalized.items()}
        derived_realization = canonical_fingerprint(
            {
                "kind": "mechanics-parameter-realization",
                "spec_fingerprint": spec.spec_fingerprint,
                "values": value_payload,
            }
        )
        resolved_realization = (
            derived_realization if realization_id is None else str(realization_id)
        )
        resolved_case = resolved_realization if case_id is None else str(case_id)
        resolved_stratum = (
            spec.stratum_id(normalized) if stratum_id is None else str(stratum_id)
        )
        if not resolved_realization or not resolved_case or not resolved_stratum:
            raise ValueError("Realization, case, and stratum IDs must be non-empty.")
        self.spec = spec
        self.values = normalized
        self.probability_weight = (
            None if probability_weight is None else float(probability_weight)
        )
        self.importance_weight = (
            None if importance_weight is None else float(importance_weight)
        )
        self.case_id = resolved_case
        self.realization_id = resolved_realization
        self.stratum_id = resolved_stratum
        self.realization_fingerprint = derived_realization

    @property
    def weight(self) -> float | None:
        if self.probability_weight is not None:
            return self.probability_weight
        return self.importance_weight


class MechanicsParameterDistribution(StrictModule, NonTrainableState):
    """Finite joint case design preserving correlations and stratum identities."""

    spec: MechanicsParameterSpec
    realizations: tuple[MechanicsParameterRealization, ...]
    normalized_weights: Array
    effective_sample_size: Array
    weight_kind: MechanicsParameterWeightKind = eqx.field(static=True)
    distribution_id: str = eqx.field(static=True)
    distribution_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        spec: MechanicsParameterSpec,
        realizations: Sequence[MechanicsParameterRealization],
        /,
        *,
        distribution_id: str | None = None,
    ):
        if not isinstance(spec, MechanicsParameterSpec):
            raise TypeError("spec must be a MechanicsParameterSpec.")
        resolved = tuple(realizations)
        if not resolved:
            raise ValueError("Mechanics parameter distributions require realizations.")
        if any(not isinstance(item, MechanicsParameterRealization) for item in resolved):
            raise TypeError(
                "Distribution entries must be MechanicsParameterRealization values."
            )
        if any(item.spec.spec_id != spec.spec_id for item in resolved):
            raise ValueError(
                "Every realization must use the distribution parameter spec."
            )
        case_ids = tuple(item.case_id for item in resolved)
        realization_ids = tuple(item.realization_id for item in resolved)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Distribution case IDs must be unique.")
        if len(set(realization_ids)) != len(realization_ids):
            raise ValueError("Distribution realization IDs must be unique.")
        kinds = {
            "probability"
            if item.probability_weight is not None
            else "importance"
            if item.importance_weight is not None
            else "equal"
            for item in resolved
        }
        if len(kinds) != 1:
            raise ValueError(
                "A distribution cannot mix equal, probability, and importance weights."
            )
        weight_kind = next(iter(kinds))
        if weight_kind == "equal":
            raw = jnp.ones((len(resolved),), dtype=float)
        else:
            raw = jnp.asarray([item.weight for item in resolved], dtype=float)
        total = jnp.sum(raw)
        weights = raw / total
        if weight_kind == "probability" and not bool(
            jnp.isclose(total, 1.0, rtol=1.0e-6, atol=1.0e-8)
        ):
            raise ValueError("Exact probability weights must sum to one.")
        ess = 1.0 / jnp.sum(weights * weights)
        derived = canonical_fingerprint(
            {
                "kind": "mechanics-parameter-distribution",
                "spec_fingerprint": spec.spec_fingerprint,
                "weight_kind": weight_kind,
                "realizations": [
                    {
                        "case_id": item.case_id,
                        "realization_id": item.realization_id,
                        "realization_fingerprint": item.realization_fingerprint,
                        "stratum_id": item.stratum_id,
                        "weight": item.weight,
                    }
                    for item in resolved
                ],
            }
        )
        identifier = derived if distribution_id is None else str(distribution_id)
        if not identifier:
            raise ValueError("Mechanics parameter distribution IDs must be non-empty.")
        self.spec = spec
        self.realizations = resolved
        self.normalized_weights = weights
        self.effective_sample_size = ess
        self.weight_kind = weight_kind
        self.distribution_id = identifier
        self.distribution_fingerprint = derived

    def realization(self, case_id: str, /) -> MechanicsParameterRealization:
        matches = tuple(item for item in self.realizations if item.case_id == case_id)
        if not matches:
            raise KeyError(f"Unknown mechanics parameter case {case_id!r}.")
        return matches[0]

    def normalized_weight(
        self,
        case: int | str | MechanicsParameterRealization,
        /,
    ) -> Array:
        if isinstance(case, MechanicsParameterRealization):
            case_id = case.case_id
        elif isinstance(case, str):
            case_id = case
        else:
            index = int(case)
            if index < 0 or index >= len(self.realizations):
                raise IndexError("Mechanics parameter case index is out of range.")
            return self.normalized_weights[index]
        for index, realization in enumerate(self.realizations):
            if realization.case_id == case_id:
                return self.normalized_weights[index]
        raise KeyError(f"Unknown mechanics parameter case {case_id!r}.")

    def contains(self, values: Mapping[str, Any], /) -> bool:
        """Whether values lie in the declared joint parameter support."""
        return self.spec.contains(values)

    def assert_disjoint(
        self,
        other: "MechanicsParameterDistribution",
        /,
        *,
        by: Literal["case", "realization"] = "realization",
    ) -> None:
        """Reject leakage between independently declared parameter designs."""
        if not isinstance(other, MechanicsParameterDistribution):
            raise TypeError("other must be a MechanicsParameterDistribution.")
        if self.spec.spec_fingerprint != other.spec.spec_fingerprint:
            raise ValueError("Parameter distributions use different parameter specs.")
        if by not in ("case", "realization"):
            raise ValueError(
                "Distribution disjointness must compare case or realization IDs."
            )
        if by == "case":
            own = {item.case_id for item in self.realizations}
            foreign = {item.case_id for item in other.realizations}
        else:
            own = {item.realization_fingerprint for item in self.realizations}
            foreign = {item.realization_fingerprint for item in other.realizations}
        overlap = own & foreign
        if overlap:
            raise ValueError(
                "Mechanics parameter designs are not held-out; overlapping "
                f"{by} identities: {sorted(overlap)}."
            )


__all__ = [
    "MechanicsParameterDistribution",
    "MechanicsParameterField",
    "MechanicsParameterKind",
    "MechanicsParameterRealization",
    "MechanicsParameterSpec",
    "MechanicsParameterWeightKind",
    "MechanicsParameterRole",
]
