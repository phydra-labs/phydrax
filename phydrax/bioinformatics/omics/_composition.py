#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    BiologicalGrouping,
    DifferentiationKind,
    ExchangeabilityPlan,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


COMPOSITION_SUCCESS = 0
COMPOSITION_EMPTY_DONOR = 1
COMPOSITION_INVALID_INDEX = 2
COMPOSITION_NONFINITE_DESIGN = 3
COMPOSITION_MISSING_EXCHANGEABILITY = 4
COMPOSITION_EMPTY_GROUP = 5


def composition_status_name(status: int, /) -> str:
    """Return the stable name of a donor-composition status code."""
    names = (
        "success",
        "empty_donor",
        "invalid_cell_index",
        "nonfinite_donor_design",
        "missing_exchangeability_plan",
        "empty_contrast_group",
    )
    code = int(status)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown composition status {code}.")
    return names[code]


def _composition_contract(name: str) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        name,
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Cell-type counts are conditioned on the supplied annotations and are "
            "aggregated before any donor-level contrast."
        ),
        truncation_statement="No cells, donors, or cell types are truncated.",
        capacity_semantics="donor_count and cell_type_count declare complete spaces.",
        assumptions=(
            "Donors are independent biological replicate units.",
            "Each cell has exactly one donor and one cell-type label.",
        ),
        nondifferentiable_outputs=("counts", "status", "valid"),
    )


class DonorCompositionEvidence(StrictModule):
    """Evidence that cell-level rows were reduced to donor replicate units."""

    cell_indices_valid: Array
    donor_observed: Array
    design_finite: Array
    exchangeability_declared: Array
    cell_count: Array
    replicate_unit: str = eqx.field(static=True)


class DonorCompositionInputs(StrictModule):
    """Donor-by-cell-type inputs suitable for compositional count models."""

    counts: Array
    totals: Array
    proportions: Array
    log_offset: Array
    donor_design: Array
    grouping: BiologicalGrouping | None
    exchangeability: ExchangeabilityPlan | None
    valid: Array
    status: Array
    evidence: DonorCompositionEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    claim_kind: str = eqx.field(static=True)


class DonorCompositionContrastEvidence(StrictModule):
    """Group sizes and replicate declaration used for a donor-level contrast."""

    first_group_donors: Array
    second_group_donors: Array
    donor_replicates_used: Array
    exchangeability_declared: Array
    pseudocount: float = eqx.field(static=True)
    replicate_unit: str = eqx.field(static=True)


class DonorCompositionContrast(StrictModule):
    """Equal-donor log-ratio contrast; cells do not inflate replication."""

    log_ratio_effect: Array
    standard_error: Array
    valid: Array
    status: Array
    evidence: DonorCompositionContrastEvidence
    exchangeability: ExchangeabilityPlan | None
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    claim_kind: str = eqx.field(static=True)


def donor_composition_inputs(
    donor_index: ArrayLike,
    cell_type_index: ArrayLike,
    donor_design: ArrayLike,
    /,
    *,
    donor_count: int,
    cell_type_count: int,
    grouping: BiologicalGrouping | None = None,
    exchangeability: ExchangeabilityPlan | None = None,
    method_contract: BioinformaticsMethodContract | None = None,
) -> DonorCompositionInputs:
    """Aggregate annotated cells into donor-level model inputs and offsets."""
    donors = jnp.asarray(donor_index)
    cell_types = jnp.asarray(cell_type_index)
    design = jnp.asarray(donor_design)
    num_donors = int(donor_count)
    num_cell_types = int(cell_type_count)
    if num_donors < 1 or num_cell_types < 2:
        raise ValueError("Composition inputs require donors and at least two cell types.")
    if donors.ndim != 1 or cell_types.ndim != 1 or donors.shape != cell_types.shape:
        raise ValueError("donor_index and cell_type_index must be matching vectors.")
    if not jnp.issubdtype(donors.dtype, jnp.integer) or not jnp.issubdtype(
        cell_types.dtype, jnp.integer
    ):
        raise TypeError("Composition indices must have integer dtypes.")
    if design.ndim != 2 or int(design.shape[0]) != num_donors:
        raise ValueError("donor_design must have shape (donor_count, covariate_count).")

    donors = donors.astype(jnp.int32)
    cell_types = cell_types.astype(jnp.int32)
    in_bounds = (
        (donors >= 0)
        & (donors < num_donors)
        & (cell_types >= 0)
        & (cell_types < num_cell_types)
    )
    safe_donors = jnp.where(in_bounds, donors, 0)
    safe_cell_types = jnp.where(in_bounds, cell_types, 0)
    counts = (
        jnp.zeros((num_donors, num_cell_types), dtype=jnp.int32)
        .at[safe_donors, safe_cell_types]
        .add(in_bounds.astype(jnp.int32))
    )
    totals = jnp.sum(counts, axis=1, dtype=jnp.int32)
    proportions = counts / jnp.maximum(totals[:, None], 1)
    design_finite = jnp.all(jnp.isfinite(design), axis=1)
    indices_valid = jnp.all(in_bounds)
    exchangeability_declared = jnp.asarray(exchangeability is not None)
    donor_observed = totals > 0
    valid = indices_valid & donor_observed & design_finite & exchangeability_declared
    status = jnp.where(
        ~indices_valid,
        COMPOSITION_INVALID_INDEX,
        jnp.where(
            ~donor_observed,
            COMPOSITION_EMPTY_DONOR,
            jnp.where(
                ~design_finite,
                COMPOSITION_NONFINITE_DESIGN,
                jnp.where(
                    exchangeability_declared,
                    COMPOSITION_SUCCESS,
                    COMPOSITION_MISSING_EXCHANGEABILITY,
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = DonorCompositionEvidence(
        indices_valid,
        donor_observed,
        design_finite,
        exchangeability_declared,
        jnp.asarray(donors.size, dtype=jnp.int32),
        "donor",
    )
    return DonorCompositionInputs(
        counts,
        totals,
        proportions,
        jnp.log(jnp.maximum(totals, 1)),
        design,
        grouping,
        exchangeability,
        valid,
        status,
        evidence,
        method_contract
        if method_contract is not None
        else _composition_contract("donor_cell_composition_inputs"),
        "exact_descriptive",
    )


def donor_composition_logratio_contrast(
    inputs: DonorCompositionInputs,
    second_group: ArrayLike,
    /,
    *,
    pseudocount: float = 0.5,
    method_contract: BioinformaticsMethodContract | None = None,
) -> DonorCompositionContrast:
    """Contrast compositions with donors, rather than cells, as replicates."""
    if not isinstance(inputs, DonorCompositionInputs):
        raise TypeError("inputs must be DonorCompositionInputs.")
    group = jnp.asarray(second_group, dtype=bool)
    donor_count = int(inputs.counts.shape[0])
    if group.shape != (donor_count,):
        raise ValueError("second_group must contain one indicator per donor.")
    pseudo = float(pseudocount)
    if not math.isfinite(pseudo) or pseudo <= 0.0:
        raise ValueError("pseudocount must be finite and positive.")

    usable = inputs.valid
    first = usable & ~group
    second = usable & group
    first_count = jnp.sum(first, dtype=jnp.int32)
    second_count = jnp.sum(second, dtype=jnp.int32)
    complete_groups = (first_count > 0) & (second_count > 0)
    denominator = inputs.totals[:, None] + pseudo * inputs.counts.shape[1]
    log_composition = jnp.log((inputs.counts + pseudo) / denominator)
    first_mean = jnp.sum(
        jnp.where(first[:, None], log_composition, 0.0), axis=0
    ) / jnp.maximum(first_count, 1)
    second_mean = jnp.sum(
        jnp.where(second[:, None], log_composition, 0.0), axis=0
    ) / jnp.maximum(second_count, 1)
    first_residual = jnp.where(first[:, None], log_composition - first_mean, 0.0)
    second_residual = jnp.where(second[:, None], log_composition - second_mean, 0.0)
    first_variance = jnp.sum(first_residual**2, axis=0) / jnp.maximum(first_count - 1, 1)
    second_variance = jnp.sum(second_residual**2, axis=0) / jnp.maximum(
        second_count - 1, 1
    )
    standard_error = jnp.sqrt(
        first_variance / jnp.maximum(first_count, 1)
        + second_variance / jnp.maximum(second_count, 1)
    )
    exchangeability_declared = jnp.asarray(inputs.exchangeability is not None)
    valid = complete_groups & exchangeability_declared & jnp.all(inputs.valid)
    status = jnp.where(
        ~exchangeability_declared,
        COMPOSITION_MISSING_EXCHANGEABILITY,
        jnp.where(complete_groups, COMPOSITION_SUCCESS, COMPOSITION_EMPTY_GROUP),
    ).astype(jnp.int32)
    evidence = DonorCompositionContrastEvidence(
        first_count,
        second_count,
        first_count + second_count,
        exchangeability_declared,
        pseudo,
        "donor",
    )
    return DonorCompositionContrast(
        second_mean - first_mean,
        standard_error,
        valid,
        status,
        evidence,
        inputs.exchangeability,
        method_contract
        if method_contract is not None
        else _composition_contract("donor_cell_composition_logratio_contrast"),
        "model_based_estimate",
    )


__all__ = [
    "COMPOSITION_EMPTY_DONOR",
    "COMPOSITION_EMPTY_GROUP",
    "COMPOSITION_INVALID_INDEX",
    "COMPOSITION_MISSING_EXCHANGEABILITY",
    "COMPOSITION_NONFINITE_DESIGN",
    "COMPOSITION_SUCCESS",
    "DonorCompositionContrast",
    "DonorCompositionContrastEvidence",
    "DonorCompositionEvidence",
    "DonorCompositionInputs",
    "composition_status_name",
    "donor_composition_inputs",
    "donor_composition_logratio_contrast",
]
