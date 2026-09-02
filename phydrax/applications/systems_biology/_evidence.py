#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-side biological fact, condition, reference, and plan-field bindings."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._gene_expression import PreparedTelegraphGeneExpression
from ._network import PreparedStoichiometricNetwork
from ._whole_cell import PreparedWholeCellAssembly


BiologicalValue: TypeAlias = bool | int | float | str
EvidenceTarget: TypeAlias = (
    PreparedStoichiometricNetwork
    | PreparedTelegraphGeneExpression
    | PreparedWholeCellAssembly
)


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{owner} must be a non-empty, trimmed string.")
    return value


def _key_segment(value: str, owner: str, /) -> str:
    segment = _identifier(value, owner)
    if ":" in segment:
        raise ValueError(f"{owner} must not contain the reserved ':' delimiter.")
    return segment


def _value(value: BiologicalValue, owner: str, /) -> BiologicalValue:
    if not isinstance(value, (bool, int, float, str)):
        raise TypeError(f"{owner} must be bool, int, float, or str.")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{owner} must be finite.")
    return value


class BiologicalReference(StrictModule, NonTrainableState):
    """Namespaced immutable source reference with an exact locator."""

    namespace: str = eqx.field(static=True)
    identifier: str = eqx.field(static=True)
    locator: str = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(self, namespace: str, identifier: str, locator: str, /):
        namespace_value = _key_segment(namespace, "Reference namespace")
        identifier_value = _key_segment(identifier, "Reference identifier")
        locator_value = _identifier(locator, "Reference locator")
        self.namespace = namespace_value
        self.identifier = identifier_value
        self.locator = locator_value
        self.reference_id = canonical_fingerprint(
            {
                "kind": "biological-reference",
                "namespace": namespace_value,
                "identifier": identifier_value,
                "locator": locator_value,
            }
        )

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.identifier}"


class BiologicalCondition(StrictModule, NonTrainableState):
    """Namespaced experimental or biological context value."""

    namespace: str = eqx.field(static=True)
    name: str = eqx.field(static=True)
    value: BiologicalValue = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)

    def __init__(self, namespace: str, name: str, value: BiologicalValue, /):
        namespace_value = _key_segment(namespace, "Condition namespace")
        name_value = _key_segment(name, "Condition name")
        normalized = _value(value, "Condition value")
        self.namespace = namespace_value
        self.name = name_value
        self.value = normalized
        self.condition_id = canonical_fingerprint(
            {
                "kind": "biological-condition",
                "namespace": namespace_value,
                "name": name_value,
                "value": normalized,
            }
        )

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.name}"


class BiologicalFact(StrictModule, NonTrainableState):
    """Namespaced scalar fact anchored to one exact source reference."""

    namespace: str = eqx.field(static=True)
    name: str = eqx.field(static=True)
    value: BiologicalValue = eqx.field(static=True)
    unit: str = eqx.field(static=True)
    reference: BiologicalReference
    fact_id: str = eqx.field(static=True)

    def __init__(
        self,
        namespace: str,
        name: str,
        value: BiologicalValue,
        unit: str,
        reference: BiologicalReference,
        /,
    ):
        if not isinstance(reference, BiologicalReference):
            raise TypeError("reference must be BiologicalReference.")
        namespace_value = _key_segment(namespace, "Fact namespace")
        name_value = _key_segment(name, "Fact name")
        normalized = _value(value, "Fact value")
        unit_value = _identifier(unit, "Fact unit")
        self.namespace = namespace_value
        self.name = name_value
        self.value = normalized
        self.unit = unit_value
        self.reference = reference
        self.fact_id = canonical_fingerprint(
            {
                "kind": "biological-fact",
                "namespace": namespace_value,
                "name": name_value,
                "value": normalized,
                "unit": unit_value,
                "reference": reference.reference_id,
            }
        )

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.name}"


class PlanFieldAssertion(StrictModule, NonTrainableState):
    """Bind one fact to one closed plan-field path under an optional condition."""

    fact_key: str = eqx.field(static=True)
    field_path: str = eqx.field(static=True)
    condition_key: str | None = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        fact_key: str,
        field_path: str,
        /,
        *,
        condition_key: str | None = None,
        absolute_tolerance: float = 0.0,
    ):
        fact_value = _identifier(fact_key, "Assertion fact key")
        field_value = _identifier(field_path, "Assertion field path")
        condition_value = (
            None
            if condition_key is None
            else _identifier(condition_key, "Assertion condition key")
        )
        tolerance = float(absolute_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("absolute_tolerance must be finite and nonnegative.")
        self.fact_key = fact_value
        self.field_path = field_value
        self.condition_key = condition_value
        self.absolute_tolerance = tolerance


class BiologicalEvidenceBinding(StrictModule, NonTrainableState):
    """Resolved fact-to-field assertions tied to an exact prepared identity."""

    field_paths: tuple[str, ...] = eqx.field(static=True)
    target_units: tuple[str, ...] = eqx.field(static=True)
    fact_ids: tuple[str, ...] = eqx.field(static=True)
    condition_ids: tuple[str | None, ...] = eqx.field(static=True)
    reference_ids: tuple[str, ...] = eqx.field(static=True)
    matched: Array
    valid: Array
    target_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)


def bind_biological_evidence(
    target: EvidenceTarget,
    facts: Sequence[BiologicalFact],
    conditions: Sequence[BiologicalCondition],
    assertions: Sequence[PlanFieldAssertion],
    /,
) -> BiologicalEvidenceBinding:
    """Resolve a closed assertion set and reject ambiguous host evidence."""
    if isinstance(target, PreparedTelegraphGeneExpression):
        fields = target.evidence_fields()
        units = target.evidence_units()
        target_id = target.model_id
    elif isinstance(target, PreparedStoichiometricNetwork):
        fields = target.evidence_fields()
        units = target.evidence_units()
        target_id = target.network_id
    elif isinstance(target, PreparedWholeCellAssembly):
        fields = target.evidence_fields()
        units = target.evidence_units()
        target_id = target.assembly_id
    else:
        raise TypeError("target must be a prepared systems-biology plan.")
    if set(fields) != set(units):
        raise ValueError("Target evidence field and unit schemas must match.")
    fact_values = tuple(facts)
    condition_values = tuple(conditions)
    assertion_values = tuple(assertions)
    if not fact_values or any(
        not isinstance(item, BiologicalFact) for item in fact_values
    ):
        raise TypeError("facts must contain BiologicalFact objects.")
    if any(not isinstance(item, BiologicalCondition) for item in condition_values):
        raise TypeError("conditions must contain BiologicalCondition objects.")
    if not assertion_values or any(
        not isinstance(item, PlanFieldAssertion) for item in assertion_values
    ):
        raise TypeError("assertions must contain PlanFieldAssertion objects.")
    references_by_key: dict[str, BiologicalReference] = {}
    for fact in fact_values:
        reference = fact.reference
        if (
            reference.key in references_by_key
            and references_by_key[reference.key].reference_id != reference.reference_id
        ):
            raise ValueError(f"Conflicting biological references for {reference.key!r}.")
        references_by_key[reference.key] = reference
    facts_by_key: dict[str, BiologicalFact] = {}
    for fact in fact_values:
        if fact.key in facts_by_key and facts_by_key[fact.key].fact_id != fact.fact_id:
            raise ValueError(f"Conflicting biological facts for {fact.key!r}.")
        facts_by_key[fact.key] = fact
    conditions_by_key: dict[str, BiologicalCondition] = {}
    for condition in condition_values:
        if (
            condition.key in conditions_by_key
            and conditions_by_key[condition.key].condition_id != condition.condition_id
        ):
            raise ValueError(f"Conflicting biological conditions for {condition.key!r}.")
        conditions_by_key[condition.key] = condition
    assertion_bindings: dict[tuple[str, str | None], str] = {}
    matched = []
    resolved_facts = []
    resolved_conditions = []
    for assertion in assertion_values:
        if assertion.fact_key not in facts_by_key:
            raise ValueError(f"Assertion references unknown fact {assertion.fact_key!r}.")
        if assertion.field_path not in fields:
            raise ValueError(
                f"Assertion references unknown field {assertion.field_path!r}."
            )
        if (
            assertion.condition_key is not None
            and assertion.condition_key not in conditions_by_key
        ):
            raise ValueError(
                f"Assertion references unknown condition {assertion.condition_key!r}."
            )
        fact = facts_by_key[assertion.fact_key]
        slot = (assertion.field_path, assertion.condition_key)
        if slot in assertion_bindings and assertion_bindings[slot] != fact.fact_id:
            raise ValueError(
                f"Conflicting assertions target field {assertion.field_path!r}."
            )
        assertion_bindings[slot] = fact.fact_id
        actual = fields[assertion.field_path]
        expected = fact.value
        actual_boolean = isinstance(actual, bool)
        expected_boolean = isinstance(expected, bool)
        if actual_boolean or expected_boolean:
            agrees = (
                actual_boolean and expected_boolean and bool(actual) == bool(expected)
            )
        elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            agrees = abs(float(actual) - float(expected)) <= assertion.absolute_tolerance
        else:
            agrees = actual == expected
        matched.append(agrees and fact.unit == units[assertion.field_path])
        resolved_facts.append(fact)
        resolved_conditions.append(
            None
            if assertion.condition_key is None
            else conditions_by_key[assertion.condition_key]
        )
    fact_ids = tuple(item.fact_id for item in resolved_facts)
    condition_ids = tuple(
        None if item is None else item.condition_id for item in resolved_conditions
    )
    reference_ids = tuple(item.reference.reference_id for item in resolved_facts)
    field_paths = tuple(item.field_path for item in assertion_values)
    target_units = tuple(units[path] for path in field_paths)
    binding_id = canonical_fingerprint(
        {
            "kind": "systems-biology-evidence-binding",
            "target": target_id,
            "fields": [
                (
                    assertion.field_path,
                    fields[assertion.field_path],
                    units[assertion.field_path],
                    fact.fact_id,
                    condition_id,
                    assertion.absolute_tolerance,
                )
                for assertion, fact, condition_id in zip(
                    assertion_values,
                    resolved_facts,
                    condition_ids,
                    strict=True,
                )
            ],
        }
    )
    matched_array = jnp.asarray(matched, dtype=bool)
    return BiologicalEvidenceBinding(
        field_paths,
        target_units,
        fact_ids,
        condition_ids,
        reference_ids,
        matched_array,
        jnp.all(matched_array),
        target_id,
        binding_id,
    )


__all__ = [
    "BiologicalCondition",
    "BiologicalEvidenceBinding",
    "BiologicalFact",
    "BiologicalReference",
    "BiologicalValue",
    "EvidenceTarget",
    "PlanFieldAssertion",
    "bind_biological_evidence",
]
