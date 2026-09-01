#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    FeatureMapping,
    MethodKind,
    OutputKind,
)


class FunctionalProfileStatus(IntEnum):
    SUCCESS = 0
    INVALID_ABUNDANCE = 1
    INVALID_MAPPING = 2
    EMPTY_PROFILE = 3
    COMPOSITION_ERROR = 4


def _functional_contract(tolerance: float) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "mass-conserving functional profile projection",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.ARRAY,
        conditioning_statement=(
            "Each source feature's mass is divided among its supplied valid function mappings "
            "in proportion to nonnegative mapping confidence."
        ),
        truncation_statement=(
            "Unmapped source-feature mass remains explicit and participates in compositional "
            "normalization."
        ),
        capacity_semantics="Source and function capacities are fixed by FeatureMapping dictionaries.",
        assumptions=("Input feature abundances are finite and nonnegative.",),
        nondifferentiable_outputs=("status", "valid"),
        input_dtype="float/int32/bool",
        compute_dtype="float32",
        output_dtype="float32",
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
    )


class FunctionalProfileResult(StrictModule):
    raw_function_mass: Array
    raw_unannotated_mass: Array
    function_abundance: Array
    unannotated_abundance: Array
    total_mass: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    mapping_id: str = eqx.field(static=True)
    function_dictionary_id: str = eqx.field(static=True)


def quantify_functional_profile(
    mapping: FeatureMapping,
    feature_abundance: ArrayLike,
    /,
    *,
    sample_valid: ArrayLike | None = None,
    composition_tolerance: float = 1.0e-6,
) -> FunctionalProfileResult:
    """Project source-feature abundance while retaining all unannotated mass."""
    tolerance = float(composition_tolerance)
    if tolerance < 0.0:
        raise ValueError("composition_tolerance must be nonnegative.")
    abundance = jnp.asarray(feature_abundance)
    if abundance.ndim == 1:
        abundance = abundance[None, :]
    if abundance.ndim != 2 or abundance.shape[1] != mapping.source.capacity:
        raise ValueError(
            "feature_abundance must have shape (sample_capacity, source feature capacity)."
        )
    if not jnp.issubdtype(abundance.dtype, jnp.inexact):
        abundance = abundance.astype(jnp.float32)
    samples = abundance.shape[0]
    cases = (
        jnp.ones((samples,), dtype=bool)
        if sample_valid is None
        else jnp.asarray(sample_valid, dtype=bool)
    )
    if cases.shape != (samples,):
        raise ValueError("sample_valid must match sample capacity.")

    source_active = mapping.source.active
    target_capacity = mapping.target.capacity
    relation = mapping.relation
    route_confidence = jnp.asarray(mapping.confidence, dtype=abundance.dtype)
    mapping_finite = jnp.isfinite(route_confidence) & (route_confidence >= 0.0)
    mapping_ok = jnp.all((~relation.valid) | mapping_finite)
    safe_source = jnp.where(relation.valid, relation.source_indices, 0)
    safe_target = jnp.where(relation.valid, relation.target_indices, 0)

    def source_body(source: int, totals: Array) -> Array:
        terms = jnp.where(
            relation.valid & mapping_finite & (safe_source == source),
            route_confidence,
            0.0,
        )
        return totals.at[source].set(compensated_sum(terms))

    confidence_total = jax.lax.fori_loop(
        0,
        mapping.source.capacity,
        source_body,
        jnp.zeros((mapping.source.capacity,), dtype=abundance.dtype),
    )
    normalized_confidence = jnp.where(
        relation.valid & mapping_finite,
        route_confidence
        / jnp.maximum(confidence_total[safe_source], jnp.finfo(abundance.dtype).tiny),
        0.0,
    )
    route_mass = (
        abundance[:, safe_source]
        * normalized_confidence[None, :]
        * relation.valid[None, :]
    )

    def target_body(target: int, profile: Array) -> Array:
        terms = jnp.where(
            relation.valid[None, :] & (safe_target[None, :] == target),
            route_mass,
            0.0,
        )
        mass = compensated_sum(terms, axis=1)
        return profile.at[:, target].set(mass)

    raw_function = jax.lax.fori_loop(
        0,
        target_capacity,
        target_body,
        jnp.zeros((samples, target_capacity), dtype=abundance.dtype),
    )
    mapped_source = confidence_total > 0.0
    unannotated_terms = jnp.where(
        source_active[None, :] & (~mapped_source[None, :]),
        abundance,
        0.0,
    )
    raw_unannotated = compensated_sum(unannotated_terms, axis=1)
    source_input_valid = jnp.all(
        (~source_active[None, :]) | (jnp.isfinite(abundance) & (abundance >= 0.0)),
        axis=1,
    )
    total_function = compensated_sum(raw_function, axis=1)
    total_mass = total_function + raw_unannotated
    nonempty = total_mass > 0.0
    denominator = jnp.maximum(total_mass, jnp.finfo(abundance.dtype).tiny)
    function_abundance = jnp.where(
        nonempty[:, None], raw_function / denominator[:, None], 0.0
    )
    unannotated_abundance = jnp.where(nonempty, raw_unannotated / denominator, 0.0)
    normalized_total = compensated_sum(function_abundance, axis=1) + unannotated_abundance
    defect = jnp.where(nonempty, jnp.abs(normalized_total - 1.0), 0.0)
    composition_ok = defect <= tolerance
    valid = cases & source_input_valid & mapping_ok & nonempty & composition_ok
    status = jnp.where(
        ~mapping_ok,
        int(FunctionalProfileStatus.INVALID_MAPPING),
        jnp.where(
            ~source_input_valid,
            int(FunctionalProfileStatus.INVALID_ABUNDANCE),
            jnp.where(
                ~nonempty,
                int(FunctionalProfileStatus.EMPTY_PROFILE),
                jnp.where(
                    composition_ok,
                    int(FunctionalProfileStatus.SUCCESS),
                    int(FunctionalProfileStatus.COMPOSITION_ERROR),
                ),
            ),
        ),
    ).astype(jnp.int32)
    status = jnp.where(cases, status, int(FunctionalProfileStatus.EMPTY_PROFILE))
    raw_function = jnp.where(valid[:, None], raw_function, 0.0)
    raw_unannotated = jnp.where(valid, raw_unannotated, 0.0)
    function_abundance = jnp.where(valid[:, None], function_abundance, 0.0)
    unannotated_abundance = jnp.where(valid, unannotated_abundance, 0.0)
    total_mass = jnp.where(valid, total_mass, 0.0)
    contract = _functional_contract(tolerance)
    evidence = jnp.stack(
        (
            jnp.sum(
                source_active[None, :] & (abundance > 0.0),
                axis=1,
                dtype=jnp.int32,
            ).astype(abundance.dtype),
            jnp.sum(
                source_active[None, :] & mapped_source[None, :] & (abundance > 0.0),
                axis=1,
                dtype=jnp.int32,
            ).astype(abundance.dtype),
            total_function,
            raw_unannotated,
            defect,
        ),
        axis=1,
    )
    return FunctionalProfileResult(
        raw_function,
        raw_unannotated,
        function_abundance,
        unannotated_abundance,
        total_mass,
        valid,
        status,
        evidence,
        contract,
        mapping.mapping_id,
        mapping.target.dictionary_id,
    )


__all__ = [
    "FunctionalProfileResult",
    "FunctionalProfileStatus",
    "quantify_functional_profile",
]
