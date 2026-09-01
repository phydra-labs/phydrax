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
    MethodKind,
    OutputKind,
)
from ._assignment import AmbiguousTaxonomicAssignment
from ._taxonomy import TaxonomyTree


class CommunityStatus(IntEnum):
    SUCCESS = 0
    VERSION_MISMATCH = 1
    INVALID_WEIGHT = 2
    EMPTY_COMMUNITY = 3
    COMPOSITION_ERROR = 4


def _community_contract(tolerance: float) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "ambiguity-aware compositional community abundance",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.PROBABILISTIC,
        conditioning_statement=(
            "Valid assignment weights and each read's unclassified mass form one per-read "
            "composition before read weighting."
        ),
        truncation_statement=(
            "All taxonomy bins and unclassified mass are retained; no renormalization discards "
            "unclassified observations."
        ),
        capacity_semantics="Taxon capacity is the validated taxonomy node capacity.",
        assumptions=("Read weights are finite and nonnegative.",),
        nondifferentiable_outputs=("status", "valid"),
        input_dtype="float/int32/bool",
        compute_dtype="float32",
        output_dtype="float32",
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
    )


class CommunityAbundanceResult(StrictModule):
    raw_taxon_mass: Array
    raw_unclassified_mass: Array
    taxon_abundance: Array
    unclassified_abundance: Array
    total_mass: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    taxonomy_id: str = eqx.field(static=True)
    version_id: str = eqx.field(static=True)


def estimate_community_abundance(
    taxonomy: TaxonomyTree,
    assignments: AmbiguousTaxonomicAssignment,
    /,
    *,
    read_weights: ArrayLike | None = None,
    composition_tolerance: float = 1.0e-6,
) -> CommunityAbundanceResult:
    """Aggregate ambiguous assignments while conserving explicit unclassified mass."""
    tolerance = float(composition_tolerance)
    if tolerance < 0.0:
        raise ValueError("composition_tolerance must be nonnegative.")
    record_capacity = assignments.record_ids.shape[0]
    weights = (
        jnp.ones((record_capacity,), dtype=assignments.weights.dtype)
        if read_weights is None
        else jnp.asarray(read_weights, dtype=assignments.weights.dtype)
    )
    if weights.shape != (record_capacity,):
        raise ValueError("read_weights must match record capacity.")
    finite_nonnegative = jnp.isfinite(weights) & (weights >= 0.0)
    malformed = jnp.any(assignments.case_mask & (~finite_nonnegative))
    version_match = jnp.asarray(
        assignments.taxonomy_id == taxonomy.taxonomy_id
        and assignments.version_id == taxonomy.version.version_id,
        dtype=bool,
    )
    active_read = assignments.case_mask & assignments.valid & finite_nonnegative
    route_mass = assignments.weights * weights[:, None]
    route_valid = assignments.assigned_valid & active_read[:, None]
    safe_taxon = jnp.clip(assignments.taxon_indices, 0, max(taxonomy.capacity - 1, 0))

    def body(taxon: int, abundance: Array) -> Array:
        terms = jnp.where(
            route_valid & (safe_taxon == taxon),
            route_mass,
            0.0,
        )
        mass = compensated_sum(terms.reshape((-1,)))
        return abundance.at[taxon].set(mass)

    raw_taxon = jax.lax.fori_loop(
        0,
        taxonomy.capacity,
        body,
        jnp.zeros((taxonomy.capacity,), dtype=assignments.weights.dtype),
    )
    raw_unclassified = compensated_sum(
        jnp.where(
            active_read,
            assignments.unclassified_mass * weights,
            0.0,
        )
    )
    classified_mass = compensated_sum(raw_taxon)
    total_mass = classified_mass + raw_unclassified
    output_available = total_mass > 0.0
    taxon_abundance = jnp.where(
        output_available,
        raw_taxon / jnp.maximum(total_mass, jnp.finfo(raw_taxon.dtype).tiny),
        0.0,
    )
    unclassified_abundance = jnp.where(
        output_available,
        raw_unclassified / jnp.maximum(total_mass, jnp.finfo(raw_taxon.dtype).tiny),
        0.0,
    )
    normalized_total = compensated_sum(taxon_abundance) + unclassified_abundance
    composition_defect = jnp.where(output_available, jnp.abs(normalized_total - 1.0), 0.0)
    composition_ok = composition_defect <= tolerance
    valid = version_match & (~malformed) & output_available & composition_ok
    status = jnp.where(
        ~version_match,
        int(CommunityStatus.VERSION_MISMATCH),
        jnp.where(
            malformed,
            int(CommunityStatus.INVALID_WEIGHT),
            jnp.where(
                ~output_available,
                int(CommunityStatus.EMPTY_COMMUNITY),
                jnp.where(
                    composition_ok,
                    int(CommunityStatus.SUCCESS),
                    int(CommunityStatus.COMPOSITION_ERROR),
                ),
            ),
        ),
    ).astype(jnp.int32)
    raw_taxon = jnp.where(version_match, raw_taxon, 0.0)
    raw_unclassified = jnp.where(version_match, raw_unclassified, 0.0)
    taxon_abundance = jnp.where(version_match, taxon_abundance, 0.0)
    unclassified_abundance = jnp.where(version_match, unclassified_abundance, 0.0)
    total_mass = jnp.where(version_match, total_mass, 0.0)
    classified_mass = jnp.where(version_match, classified_mass, 0.0)
    composition_defect = jnp.where(version_match, composition_defect, 0.0)
    contract = _community_contract(tolerance)
    evidence = jnp.asarray(
        (
            jnp.sum(active_read, dtype=jnp.int32),
            total_mass,
            classified_mass,
            raw_unclassified,
            composition_defect,
        ),
        dtype=assignments.weights.dtype,
    )
    return CommunityAbundanceResult(
        raw_taxon,
        raw_unclassified,
        taxon_abundance,
        unclassified_abundance,
        total_mass,
        valid,
        status,
        evidence,
        contract,
        taxonomy.taxonomy_id,
        taxonomy.version.version_id,
    )


__all__ = [
    "CommunityAbundanceResult",
    "CommunityStatus",
    "estimate_community_abundance",
]
