#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mixed and spinning correlator systems with explicit crossing-basis gauges."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


class ExternalPrimaryOperator(StrictModule):
    """One primary with explicit Lorentz/global representation and parity."""

    label: str = eqx.field(static=True)
    scaling_dimension: float = eqx.field(static=True)
    lorentz_highest_weight: tuple[int, ...] = eqx.field(static=True)
    global_representation: str = eqx.field(static=True)
    parity: Literal[-1, 1] = eqx.field(static=True)
    reality: Literal["real", "complex", "pseudoreal"] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        label: str,
        scaling_dimension: float,
        lorentz_highest_weight: Sequence[int],
        global_representation: str,
        /,
        *,
        parity: Literal[-1, 1] = 1,
        reality: Literal["real", "complex", "pseudoreal"] = "real",
    ):
        label_ = _identifier(label, "primary label")
        dimension = float(scaling_dimension)
        weight = tuple(int(value) for value in lorentz_highest_weight)
        representation = _identifier(global_representation, "global representation")
        if not math.isfinite(dimension) or dimension < 0.0:
            raise ValueError(
                "Primary scaling dimensions must be finite and non-negative."
            )
        if not weight or any(value < 0 for value in weight):
            raise ValueError(
                "Lorentz highest weights must be non-empty and non-negative."
            )
        if parity not in (-1, 1):
            raise ValueError("Primary parity must be -1 or +1.")
        if reality not in ("real", "complex", "pseudoreal"):
            raise ValueError("Unknown primary reality convention.")
        content = {
            "kind": "external-primary-operator",
            "label": label_,
            "scaling_dimension": dimension,
            "lorentz_highest_weight": weight,
            "global_representation": representation,
            "parity": parity,
            "reality": reality,
        }
        self.label = label_
        self.scaling_dimension = dimension
        self.lorentz_highest_weight = weight
        self.global_representation = representation
        self.parity = parity
        self.reality = reality
        self.operator_id = canonical_fingerprint(content)

    @property
    def scalar(self) -> bool:
        return all(value == 0 for value in self.lorentz_highest_weight)


class CorrelatorSpecification(StrictModule):
    """One ordered four-primary correlator and its tensor structures."""

    label: str = eqx.field(static=True)
    external_labels: tuple[str, str, str, str] = eqx.field(static=True)
    tensor_structure_labels: tuple[str, ...] = eqx.field(static=True)
    structure_parities: tuple[Literal[-1, 1], ...] = eqx.field(static=True)
    correlator_id: str = eqx.field(static=True)

    def __init__(
        self,
        label: str,
        external_labels: Sequence[str],
        tensor_structure_labels: Sequence[str],
        /,
        *,
        structure_parities: Sequence[Literal[-1, 1]] | None = None,
    ):
        label_ = _identifier(label, "correlator label")
        external = tuple(
            _identifier(value, "external label") for value in external_labels
        )
        structures = tuple(
            _identifier(value, "tensor structure label")
            for value in tensor_structure_labels
        )
        if len(external) != 4:
            raise ValueError("A correlator requires exactly four ordered primaries.")
        if not structures or len(set(structures)) != len(structures):
            raise ValueError("Tensor structure labels must be non-empty and unique.")
        parities = (
            (1,) * len(structures)
            if structure_parities is None
            else tuple(int(value) for value in structure_parities)
        )
        if len(parities) != len(structures) or any(
            value not in (-1, 1) for value in parities
        ):
            raise ValueError("One ±1 parity is required per tensor structure.")
        content = {
            "kind": "correlator-specification",
            "label": label_,
            "external_labels": external,
            "tensor_structure_labels": structures,
            "structure_parities": parities,
        }
        self.label = label_
        self.external_labels = external
        self.tensor_structure_labels = structures
        self.structure_parities = cast(tuple[Literal[-1, 1], ...], parities)
        self.correlator_id = canonical_fingerprint(content)


class CrossingSystemGenerator(StrictModule):
    """One finite crossing generator on all correlator tensor components."""

    label: str = eqx.field(static=True)
    matrix: Array
    order: int = eqx.field(static=True)
    generator_id: str = eqx.field(static=True)

    def __init__(self, label: str, matrix: ArrayLike, order: int, /):
        label_ = _identifier(label, "crossing generator label")
        matrix_ = np.asarray(matrix, dtype=np.complex128)
        order_ = int(order)
        if matrix_.ndim != 2 or matrix_.shape[0] != matrix_.shape[1] or not matrix_.size:
            raise ValueError("Crossing generator matrices must be non-empty and square.")
        if not np.all(np.isfinite(matrix_)) or order_ < 1:
            raise ValueError("Crossing generator values and order must be valid.")
        self.label = label_
        self.matrix = jnp.asarray(matrix_)
        self.order = order_
        self.generator_id = canonical_fingerprint(
            {
                "kind": "crossing-system-generator",
                "label": label_,
                "matrix": array_tree_fingerprint(matrix_),
                "order": order_,
            }
        )


class CorrelatorSystemPlan(StrictModule):
    """Finite mixed/spinning correlator crossing system in one basis gauge."""

    primaries: tuple[ExternalPrimaryOperator, ...]
    correlators: tuple[CorrelatorSpecification, ...]
    generators: tuple[CrossingSystemGenerator, ...]
    component_labels: tuple[str, ...] = eqx.field(static=True)
    basis_gauge_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_group_order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        primaries: Sequence[ExternalPrimaryOperator],
        correlators: Sequence[CorrelatorSpecification],
        generators: Sequence[CrossingSystemGenerator],
        /,
        *,
        basis_gauge_id: str,
        tolerance: float = 1e-10,
        maximum_group_order: int = 48,
    ):
        primaries_ = tuple(primaries)
        correlators_ = tuple(correlators)
        generators_ = tuple(generators)
        if not primaries_ or any(
            not isinstance(value, ExternalPrimaryOperator) for value in primaries_
        ):
            raise TypeError("primaries must contain ExternalPrimaryOperator values.")
        if not correlators_ or any(
            not isinstance(value, CorrelatorSpecification) for value in correlators_
        ):
            raise TypeError("correlators must contain CorrelatorSpecification values.")
        if not generators_ or any(
            not isinstance(value, CrossingSystemGenerator) for value in generators_
        ):
            raise TypeError("generators must contain CrossingSystemGenerator values.")
        labels = {value.label for value in primaries_}
        if len(labels) != len(primaries_):
            raise ValueError("Primary labels must be unique.")
        if len({value.label for value in correlators_}) != len(correlators_):
            raise ValueError("Correlator labels must be unique.")
        if any(not set(value.external_labels) <= labels for value in correlators_):
            raise ValueError("Correlators reference unknown external primaries.")
        components = tuple(
            f"{correlator.label}:{structure}"
            for correlator in correlators_
            for structure in correlator.tensor_structure_labels
        )
        if any(
            value.matrix.shape != (len(components), len(components))
            for value in generators_
        ):
            raise ValueError(
                "Every crossing generator must act on all tensor components."
            )
        if len({value.label for value in generators_}) != len(generators_):
            raise ValueError("Crossing generator labels must be unique.")
        tolerance_ = float(tolerance)
        maximum = int(maximum_group_order)
        gauge = _identifier(basis_gauge_id, "basis gauge ID")
        if not math.isfinite(tolerance_) or tolerance_ <= 0.0 or maximum < 1:
            raise ValueError(
                "Crossing tolerance and group-order capacity must be positive."
            )
        content = {
            "kind": "correlator-system-plan",
            "primaries": [value.operator_id for value in primaries_],
            "correlators": [value.correlator_id for value in correlators_],
            "generators": [value.generator_id for value in generators_],
            "basis_gauge_id": gauge,
            "tolerance": tolerance_,
            "maximum_group_order": maximum,
        }
        self.primaries = primaries_
        self.correlators = correlators_
        self.generators = generators_
        self.component_labels = components
        self.basis_gauge_id = gauge
        self.tolerance = tolerance_
        self.maximum_group_order = maximum
        self.plan_id = canonical_fingerprint(content)


class CorrelatorSystemEvidence(StrictModule):
    generator_order_residuals: Array
    inverse_residuals: Array
    closure_residual: Array
    group_order: int = eqx.field(static=True)
    scalar_only: bool = eqx.field(static=True)
    mixed: bool = eqx.field(static=True)
    spinning: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedCorrelatorSystem(StrictModule):
    plan: CorrelatorSystemPlan
    matrices: Array
    words: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    multiplication_table: Array
    evidence: CorrelatorSystemEvidence
    prepared_id: str = eqx.field(static=True)

    def act(self, group_element: int, components: ArrayLike, /) -> Array:
        index = int(group_element)
        if index < 0 or index >= self.evidence.group_order:
            raise ValueError("Crossing group element is out of range.")
        values = jnp.asarray(components, dtype=self.matrices.dtype)
        if values.shape[-1:] != (len(self.plan.component_labels),):
            raise ValueError("Crossing components have the wrong trailing dimension.")
        return ein.contract("ij,...j->...i", self.matrices[index], values)


def _matching_matrix(
    matrices: Sequence[np.ndarray],
    candidate: np.ndarray,
    tolerance: float,
    /,
) -> int | None:
    for index, matrix in enumerate(matrices):
        if np.allclose(matrix, candidate, rtol=tolerance, atol=tolerance):
            return index
    return None


def prepare_correlator_system(plan: CorrelatorSystemPlan, /) -> PreparedCorrelatorSystem:
    """Close and audit the finite crossing action for mixed or spinning systems."""

    if not isinstance(plan, CorrelatorSystemPlan):
        raise TypeError("plan must be CorrelatorSystemPlan.")
    dimension = len(plan.component_labels)
    identity = np.eye(dimension, dtype=np.complex128)
    generator_matrices = tuple(np.asarray(value.matrix) for value in plan.generators)
    order_residuals = []
    for generator, matrix in zip(plan.generators, generator_matrices, strict=True):
        order_residuals.append(
            float(
                np.linalg.norm(np.linalg.matrix_power(matrix, generator.order) - identity)
            )
        )
    if max(order_residuals) > plan.tolerance:
        raise ValueError("A crossing generator violates its declared order.")
    matrices = [identity]
    words: list[tuple[str, ...]] = [()]
    cursor = 0
    while cursor < len(matrices):
        base = matrices[cursor]
        word = words[cursor]
        for generator, matrix in zip(plan.generators, generator_matrices, strict=True):
            candidate = matrix @ base
            match = _matching_matrix(matrices, candidate, plan.tolerance)
            if match is None:
                if len(matrices) >= plan.maximum_group_order:
                    raise ValueError("Crossing closure exceeds maximum_group_order.")
                matrices.append(candidate)
                words.append(word + (generator.label,))
        cursor += 1
    matrix_table = np.stack(matrices, axis=0)
    group_order = len(matrices)
    multiplication = np.empty((group_order, group_order), dtype=np.int32)
    closure_residual = 0.0
    for left in range(group_order):
        for right in range(group_order):
            product = matrix_table[left] @ matrix_table[right]
            match = _matching_matrix(matrices, product, plan.tolerance)
            if match is None:
                raise ValueError("Crossing action is not closed under multiplication.")
            multiplication[left, right] = match
            closure_residual = max(
                closure_residual,
                float(np.linalg.norm(product - matrix_table[match])),
            )
    inverse_residuals = []
    for index in range(group_order):
        inverse_matches = np.flatnonzero(
            (multiplication[index] == 0) & (multiplication[:, index] == 0)
        )
        if inverse_matches.size != 1:
            raise ValueError("Crossing action does not have unique inverses.")
        inverse_residuals.append(
            float(
                np.linalg.norm(
                    matrix_table[inverse_matches[0]] @ matrix_table[index] - identity
                )
            )
        )
    primary_by_label = {value.label: value for value in plan.primaries}
    used = tuple(
        primary_by_label[label]
        for value in plan.correlators
        for label in value.external_labels
    )
    scalar_only = all(value.scalar for value in used)
    mixed = len({value.label for value in used}) > 1
    spinning = any(not value.scalar for value in used)
    evidence_id = canonical_fingerprint(
        {
            "kind": "correlator-system-evidence",
            "plan": plan.plan_id,
            "group_order": group_order,
            "order_residuals": order_residuals,
            "inverse_residuals": inverse_residuals,
            "closure_residual": closure_residual,
            "scalar_only": scalar_only,
            "mixed": mixed,
            "spinning": spinning,
        }
    )
    evidence = CorrelatorSystemEvidence(
        generator_order_residuals=jnp.asarray(order_residuals),
        inverse_residuals=jnp.asarray(inverse_residuals),
        closure_residual=jnp.asarray(closure_residual),
        group_order=group_order,
        scalar_only=scalar_only,
        mixed=mixed,
        spinning=spinning,
        evidence_id=evidence_id,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-correlator-system",
            "plan": plan.plan_id,
            "matrices": array_tree_fingerprint(matrix_table),
            "multiplication": array_tree_fingerprint(multiplication),
        }
    )
    return PreparedCorrelatorSystem(
        plan=plan,
        matrices=jnp.asarray(matrix_table),
        words=tuple(words),
        multiplication_table=jnp.asarray(multiplication),
        evidence=evidence,
        prepared_id=prepared_id,
    )


__all__ = [
    "CorrelatorSpecification",
    "CorrelatorSystemEvidence",
    "CorrelatorSystemPlan",
    "CrossingSystemGenerator",
    "ExternalPrimaryOperator",
    "PreparedCorrelatorSystem",
    "prepare_correlator_system",
]
