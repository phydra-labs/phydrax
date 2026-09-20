#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evidence-bearing sampled Weil–Petersson and Yukawa integrals."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._calabi_yau import ProjectiveMeasureTarget


ModuliRepresentativeKind: TypeAlias = Literal["algebraic", "harmonic"]


class CalabiYauModuliObservablePlan(StrictModule):
    """Declared representative provenance, batching, and scientific tolerances."""

    modulus_labels: tuple[str, ...] = eqx.field(static=True)
    representative_kind: ModuliRepresentativeKind = eqx.field(static=True)
    representative_source_id: str = eqx.field(static=True)
    batch_count: int = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    positivity_tolerance: float = eqx.field(static=True)
    yukawa_symmetry_tolerance: float = eqx.field(static=True)
    maximum_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        modulus_labels: Sequence[str],
        /,
        *,
        representative_kind: ModuliRepresentativeKind,
        representative_source_id: str,
        batch_count: int = 4,
        hermiticity_tolerance: float = 1e-9,
        positivity_tolerance: float = 1e-10,
        yukawa_symmetry_tolerance: float = 1e-9,
        maximum_elements: int = 10_000_000,
    ):
        labels = tuple(str(value) for value in modulus_labels)
        kind = str(representative_kind)
        source = str(representative_source_id)
        batches = int(batch_count)
        hermiticity = float(hermiticity_tolerance)
        positivity = float(positivity_tolerance)
        symmetry = float(yukawa_symmetry_tolerance)
        maximum = int(maximum_elements)
        if (
            not labels
            or any(not value for value in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("modulus_labels must be unique and non-empty.")
        if kind not in {"algebraic", "harmonic"}:
            raise ValueError("representative_kind must be algebraic or harmonic.")
        if not source or batches < 1 or maximum < 1:
            raise ValueError("Representative source and resource bounds are required.")
        if any(
            not np.isfinite(value) or value < 0.0
            for value in (hermiticity, positivity, symmetry)
        ):
            raise ValueError(
                "Moduli-observable tolerances must be finite and nonnegative."
            )
        self.modulus_labels = labels
        self.representative_kind = kind
        self.representative_source_id = source
        self.batch_count = batches
        self.hermiticity_tolerance = hermiticity
        self.positivity_tolerance = positivity
        self.yukawa_symmetry_tolerance = symmetry
        self.maximum_elements = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "calabi-yau-moduli-observable-plan",
                "modulus_labels": labels,
                "representative_kind": kind,
                "representative_source_id": source,
                "batch_count": batches,
                "hermiticity_tolerance": hermiticity,
                "positivity_tolerance": positivity,
                "yukawa_symmetry_tolerance": symmetry,
                "maximum_elements": maximum,
            }
        )

    @property
    def modulus_count(self) -> int:
        return len(self.modulus_labels)


class PreparedCalabiYauModuliSamples(StrictModule):
    """Fixed sample measure and caller-qualified pointwise moduli integrands."""

    plan: CalabiYauModuliObservablePlan = eqx.field(static=True)
    measure: ProjectiveMeasureTarget
    representatives: Array
    yukawa_density: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: CalabiYauModuliObservablePlan,
        measure: ProjectiveMeasureTarget,
        representatives: ArrayLike,
        yukawa_density: ArrayLike,
        /,
    ):
        if not isinstance(plan, CalabiYauModuliObservablePlan):
            raise TypeError("plan must be CalabiYauModuliObservablePlan.")
        if not isinstance(measure, ProjectiveMeasureTarget):
            raise TypeError("measure must be ProjectiveMeasureTarget.")
        values = jnp.asarray(representatives)
        yukawa = jnp.asarray(yukawa_density, dtype=values.dtype)
        sample_count = measure.normalized_weights.shape[0]
        if values.ndim < 3 or values.shape[:2] != (
            sample_count,
            plan.modulus_count,
        ):
            raise ValueError(
                "representatives must have shape (samples, moduli, components...)."
            )
        expected_yukawa = (
            sample_count,
            plan.modulus_count,
            plan.modulus_count,
            plan.modulus_count,
        )
        if yukawa.shape != expected_yukawa:
            raise ValueError(f"yukawa_density must have shape {expected_yukawa}.")
        if values.size + yukawa.size > plan.maximum_elements:
            raise ValueError("Moduli observable samples exceed maximum_elements.")
        if plan.batch_count > sample_count:
            raise ValueError("batch_count cannot exceed the sample count.")
        self.plan = plan
        self.measure = measure
        self.representatives = values.reshape((sample_count, plan.modulus_count, -1))
        self.yukawa_density = yukawa
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-calabi-yau-moduli-samples",
                "plan": plan.plan_id,
                "measure_precision": measure.precision.policy_id,
                "measure_kind": measure.measure_kind,
                "representatives": array_tree_fingerprint(values),
                "yukawa_density": array_tree_fingerprint(yukawa),
            }
        )


class CalabiYauModuliObservableResult(StrictModule):
    weil_petersson_metric: Array
    physical_weil_petersson_metric: Array
    yukawa_couplings: Array
    physical_yukawa_couplings: Array
    weil_petersson_batch_standard_error: Array
    yukawa_batch_standard_error: Array
    hermiticity_residual: Array
    minimum_metric_eigenvalue: Array
    yukawa_symmetry_residual: Array
    effective_sample_size: Array
    finite: Array
    accepted: Array
    authoritative: Array
    prepared_id: str = eqx.field(static=True)
    representative_kind: ModuliRepresentativeKind = eqx.field(static=True)
    representative_source_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def _weighted_observables(
    representatives: Array,
    yukawa_density: Array,
    weights: Array,
    /,
) -> tuple[Array, Array]:
    metric = ein.contract(
        "s,smi,sni->mn",
        weights,
        jnp.conj(representatives),
        representatives,
    )
    yukawa = ein.contract("s,smnp->mnp", weights, yukawa_density)
    return metric, yukawa


def evaluate_calabi_yau_moduli_observables(
    prepared: PreparedCalabiYauModuliSamples, /
) -> CalabiYauModuliObservableResult:
    if not isinstance(prepared, PreparedCalabiYauModuliSamples):
        raise TypeError("prepared must be PreparedCalabiYauModuliSamples.")
    weights = prepared.measure.normalized_weights
    metric, yukawa = _weighted_observables(
        prepared.representatives, prepared.yukawa_density, weights
    )
    physical_mass = prepared.measure.physical_mass
    physical_metric = physical_mass * metric
    physical_yukawa = physical_mass * yukawa
    batch_indices = np.array_split(np.arange(weights.shape[0]), prepared.plan.batch_count)
    metric_batches = []
    yukawa_batches = []
    for indices in batch_indices:
        index = jnp.asarray(indices)
        batch_weights = weights[index]
        batch_weights = batch_weights / jnp.sum(batch_weights)
        batch_metric, batch_yukawa = _weighted_observables(
            prepared.representatives[index],
            prepared.yukawa_density[index],
            batch_weights,
        )
        metric_batches.append(batch_metric)
        yukawa_batches.append(batch_yukawa)
    metric_batch_array = jnp.stack(metric_batches)
    yukawa_batch_array = jnp.stack(yukawa_batches)
    denominator = jnp.sqrt(float(prepared.plan.batch_count))
    metric_error = jnp.std(metric_batch_array, axis=0, ddof=0) / denominator
    yukawa_error = jnp.std(yukawa_batch_array, axis=0, ddof=0) / denominator
    metric_scale = jnp.maximum(1.0, jnp.linalg.norm(metric))
    hermiticity = jnp.linalg.norm(metric - jnp.conj(metric.T)) / metric_scale
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (metric + jnp.conj(metric.T)))
    minimum = jnp.min(eigenvalues)
    symmetry_residual = jnp.asarray(0.0, dtype=jnp.real(yukawa).dtype)
    for order in permutations((0, 1, 2)):
        symmetry_residual = jnp.maximum(
            symmetry_residual,
            jnp.linalg.norm(yukawa - jnp.transpose(yukawa, order))
            / jnp.maximum(1.0, jnp.linalg.norm(yukawa)),
        )
    finite = (
        prepared.measure.valid
        & jnp.all(jnp.isfinite(metric))
        & jnp.all(jnp.isfinite(yukawa))
        & jnp.all(jnp.isfinite(metric_error))
        & jnp.all(jnp.isfinite(yukawa_error))
    )
    accepted = (
        finite
        & (hermiticity <= prepared.plan.hermiticity_tolerance)
        & (minimum >= -prepared.plan.positivity_tolerance)
        & (symmetry_residual <= prepared.plan.yukawa_symmetry_tolerance)
    )
    authoritative = accepted & (prepared.plan.representative_kind == "harmonic")
    return CalabiYauModuliObservableResult(
        weil_petersson_metric=metric,
        physical_weil_petersson_metric=physical_metric,
        yukawa_couplings=yukawa,
        physical_yukawa_couplings=physical_yukawa,
        weil_petersson_batch_standard_error=metric_error,
        yukawa_batch_standard_error=yukawa_error,
        hermiticity_residual=hermiticity,
        minimum_metric_eigenvalue=minimum,
        yukawa_symmetry_residual=symmetry_residual,
        effective_sample_size=prepared.measure.effective_sample_size,
        finite=finite,
        accepted=accepted,
        authoritative=authoritative,
        prepared_id=prepared.prepared_id,
        representative_kind=prepared.plan.representative_kind,
        representative_source_id=prepared.plan.representative_source_id,
        claim=(
            "sampled-harmonic-representative-weil-petersson-and-yukawa-evidence"
            if prepared.plan.representative_kind == "harmonic"
            else "sampled-algebraic-proxy-not-harmonic-weil-petersson-or-physical-yukawa"
        ),
    )


__all__ = [
    "CalabiYauModuliObservablePlan",
    "CalabiYauModuliObservableResult",
    "ModuliRepresentativeKind",
    "PreparedCalabiYauModuliSamples",
    "evaluate_calabi_yau_moduli_observables",
]
