#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExchangeabilityPlan,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._assay import SpatialAssay
from ._neighbors import SpatialNeighborGraph


SpatialStatistic = Literal["moran", "geary"]


class SpatialStatisticStatus(IntEnum):
    OK = 0
    GRAPH_INVALID = 1
    INSUFFICIENT_DONORS = 2
    ZERO_VARIANCE = 3
    INVALID_EXCHANGEABILITY = 4


class SpatialStatisticEvidence(StrictModule):
    donor_count: Array
    section_count: Array
    exchangeability_group_count: Array
    effective_spot_count: Array
    graph_edge_count: Array
    permutation_count: Array


_MORAN_CONTRACT = BioinformaticsMethodContract(
    "design_aware_moran_permutation_test",
    MethodKind.EXACT_MODEL,
    ExecutionKind.STOCHASTIC_ESTIMATE,
    DifferentiationKind.NONE,
    OutputKind.PROBABILISTIC,
    conditioning_statement=(
        "The reported randomization p-value is conditional on the supplied donor, "
        "section, and exchangeability assignments."
    ),
    truncation_statement=(
        "The observed Moran statistic is evaluated on the full valid graph; the null "
        "tail probability is a finite Monte Carlo estimate with add-one correction."
    ),
    capacity_semantics=(
        "The null distribution has exactly the requested static permutation count."
    ),
    assumptions=(
        "Exchangeability holds only within each declared randomization group.",
        "At least two independent donors are required for inferential validity.",
    ),
    nondifferentiable_outputs=("p_value", "null_distribution", "status", "evidence"),
)

_GEARY_CONTRACT = BioinformaticsMethodContract(
    "design_aware_geary_permutation_test",
    MethodKind.EXACT_MODEL,
    ExecutionKind.STOCHASTIC_ESTIMATE,
    DifferentiationKind.NONE,
    OutputKind.PROBABILISTIC,
    conditioning_statement=(
        "The reported randomization p-value is conditional on the supplied donor, "
        "section, and exchangeability assignments."
    ),
    truncation_statement=(
        "The observed Geary statistic is evaluated on the full valid graph; the null "
        "tail probability is a finite Monte Carlo estimate with add-one correction."
    ),
    capacity_semantics=(
        "The null distribution has exactly the requested static permutation count."
    ),
    assumptions=(
        "Exchangeability holds only within each declared randomization group.",
        "At least two independent donors are required for inferential validity.",
    ),
    nondifferentiable_outputs=("p_value", "null_distribution", "status", "evidence"),
)


class SpatialAutocorrelationResult(StrictModule):
    statistic: Array
    expected: Array
    p_value: Array
    null_distribution: Array
    valid: Array
    status: Array
    evidence: SpatialStatisticEvidence
    method_contract: BioinformaticsMethodContract


def _distinct_count(labels: Array, active: Array, /) -> Array:
    same_prior = (labels[:, None] == labels[None, :]) & active[None, :]
    prior = jnp.tril(same_prior, k=-1)
    first = active & ~jnp.any(prior, axis=1)
    return jnp.sum(first, dtype=jnp.int32)


def _statistic(
    values: Array,
    graph: SpatialNeighborGraph,
    observation_weights: Array,
    kind: SpatialStatistic,
    /,
) -> tuple[Array, Array, Array]:
    active = observation_weights > 0.0
    total = jnp.sum(observation_weights)
    mean = jnp.sum(observation_weights * values) / jnp.where(total > 0.0, total, 1.0)
    centered = jnp.where(active, values - mean, 0.0)
    denominator = jnp.sum(observation_weights * centered * centered)
    neighbor = centered[graph.relation.source_indices]
    route_weight = graph.weight * graph.relation.valid.astype(graph.weight.dtype)
    row_weight = observation_weights[:, None]
    graph_mass = jnp.sum(row_weight * route_weight)
    effective_n = total * total / jnp.maximum(jnp.sum(observation_weights**2), 1.0e-30)
    if kind == "moran":
        numerator = jnp.sum(row_weight * route_weight * centered[:, None] * neighbor)
        statistic = (
            effective_n
            * numerator
            / jnp.where(
                (graph_mass > 0.0) & (denominator > 0.0), graph_mass * denominator, 1.0
            )
        )
        expected = -1.0 / jnp.maximum(effective_n - 1.0, 1.0)
    else:
        numerator = jnp.sum(
            row_weight * route_weight * (centered[:, None] - neighbor) ** 2
        )
        statistic = (
            (effective_n - 1.0)
            * numerator
            / jnp.where(
                (graph_mass > 0.0) & (denominator > 0.0),
                2.0 * graph_mass * denominator,
                1.0,
            )
        )
        expected = jnp.asarray(1.0, dtype=values.dtype)
    return statistic, expected, denominator


def _permutation_indices(
    key: Array,
    blocks: Array,
    permutable: Array,
    permutations: int,
    /,
) -> Array:
    count = int(blocks.shape[0])
    maximum = jnp.max(jnp.where(permutable, blocks, -1), initial=-1)
    effective_blocks = jnp.where(
        permutable,
        blocks,
        maximum + 1 + jnp.arange(count, dtype=blocks.dtype),
    )
    destinations = jnp.argsort(effective_blocks, stable=True).astype(jnp.int32)
    random_score = jax.random.uniform(key, (permutations, count))

    def one(score):
        sources = jnp.lexsort((score, effective_blocks)).astype(jnp.int32)
        return jnp.zeros((count,), dtype=jnp.int32).at[destinations].set(sources)

    return jax.vmap(one)(random_score)


def _plan_assignments(
    plan: ExchangeabilityPlan,
    observation_entity_indices: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    units = plan.experimental_units
    match = (
        observation_entity_indices[:, None] == units.observation_indices[None, :]
    ) & units.included[None, :]
    match_count = jnp.sum(match, axis=1, dtype=jnp.int32)
    route = jnp.argmax(match, axis=1)
    blocks = plan.exchangeability_group_ids[route]
    donor = units.lineage.biological_replicate_group_ids[units.observation_indices[route]]
    permutable = plan.permutation_mask[route]
    valid = jnp.all(match_count == 1)
    return blocks, donor, permutable, valid


def spatial_autocorrelation_test(
    values: Any,
    graph: SpatialNeighborGraph,
    key: Array,
    /,
    *,
    statistic: SpatialStatistic = "moran",
    permutations: int = 999,
    observation_weights: Any | None = None,
    donor_index: Any | None = None,
    section_index: Any | None = None,
    exchangeability_blocks: Any | None = None,
    exchangeability_plan: ExchangeabilityPlan | None = None,
    observation_entity_indices: Any | None = None,
) -> SpatialAutocorrelationResult:
    """Estimate a restricted-randomization tail probability for Moran or Geary."""
    if not isinstance(graph, SpatialNeighborGraph):
        raise TypeError("graph must be a SpatialNeighborGraph.")
    if statistic not in ("moran", "geary"):
        raise ValueError("statistic must be 'moran' or 'geary'.")
    permutation_count = int(permutations)
    if permutation_count <= 0:
        raise ValueError("permutations must be positive.")
    observations = jnp.asarray(values, dtype=float)
    if observations.ndim != 1:
        raise ValueError("Spatial statistic values must be a rank-1 spot vector.")
    count = int(observations.shape[0])
    if graph.relation.source_size != count or graph.relation.target_shape != (count,):
        raise ValueError("Graph rows and source space must align with values.")
    weights = (
        jnp.ones((count,), dtype=observations.dtype)
        if observation_weights is None
        else jnp.asarray(observation_weights, dtype=observations.dtype)
    )
    if weights.shape != (count,):
        raise ValueError("observation_weights must have shape (spot,).")
    host_weights = np.asarray(weights)
    if np.any(~np.isfinite(host_weights)) or np.any(host_weights < 0.0):
        raise ValueError("observation_weights must be finite and non-negative.")
    host_active = host_weights > 0.0
    if np.any(~np.isfinite(np.asarray(observations)[host_active])):
        raise ValueError("Active spatial statistic values must be finite.")
    observations = jnp.where(weights > 0.0, observations, 0.0)
    if donor_index is None:
        donors = jnp.zeros((count,), dtype=jnp.int32)
    else:
        donors = jnp.asarray(donor_index, dtype=jnp.int32)
    sections = (
        jnp.zeros((count,), dtype=jnp.int32)
        if section_index is None
        else jnp.asarray(section_index, dtype=jnp.int32)
    )
    if donors.shape != (count,) or sections.shape != (count,):
        raise ValueError("donor_index and section_index must have shape (spot,).")

    design_valid = jnp.asarray(True)
    permutable = weights > 0.0
    if exchangeability_plan is not None:
        if exchangeability_blocks is not None:
            raise ValueError(
                "Supply exchangeability_plan or exchangeability_blocks, not both."
            )
        if not isinstance(exchangeability_plan, ExchangeabilityPlan):
            raise TypeError("exchangeability_plan must be an ExchangeabilityPlan.")
        if observation_entity_indices is None:
            raise ValueError(
                "observation_entity_indices are required with an ExchangeabilityPlan."
            )
        entity_indices = jnp.asarray(observation_entity_indices, dtype=jnp.int32)
        if entity_indices.shape != (count,):
            raise ValueError("observation_entity_indices must have shape (spot,).")
        blocks, planned_donors, planned_mask, design_valid = _plan_assignments(
            exchangeability_plan, entity_indices
        )
        donors = planned_donors if donor_index is None else donors
        permutable = permutable & planned_mask
    elif exchangeability_blocks is None:
        blocks = sections
    else:
        blocks = jnp.asarray(exchangeability_blocks, dtype=jnp.int32)
        if blocks.shape != (count,):
            raise ValueError("exchangeability_blocks must have shape (spot,).")
    design_valid = (
        design_valid
        & jnp.all(jnp.where(weights > 0.0, donors >= 0, True))
        & jnp.all(jnp.where(weights > 0.0, sections >= 0, True))
        & jnp.all(jnp.where(permutable, blocks >= 0, True))
    )

    active = weights > 0.0
    donor_count = _distinct_count(donors, active)
    section_count = _distinct_count(sections, active)
    group_count = _distinct_count(blocks, active & permutable)
    observed, expected, variance = _statistic(observations, graph, weights, statistic)
    indices = _permutation_indices(key, blocks, permutable, permutation_count)
    null = jax.vmap(
        lambda index: _statistic(observations[index], graph, weights, statistic)[0]
    )(indices)
    tail = jnp.abs(null - expected) >= jnp.abs(observed - expected)
    p_value = (1.0 + jnp.sum(tail)) / float(permutation_count + 1)

    graph_valid = jnp.asarray(graph.valid)
    enough_donors = donor_count >= 2
    nonconstant = variance > 0.0
    valid = graph_valid & enough_donors & nonconstant & design_valid
    status = jnp.where(
        ~graph_valid,
        int(SpatialStatisticStatus.GRAPH_INVALID),
        jnp.where(
            ~design_valid,
            int(SpatialStatisticStatus.INVALID_EXCHANGEABILITY),
            jnp.where(
                ~enough_donors,
                int(SpatialStatisticStatus.INSUFFICIENT_DONORS),
                jnp.where(
                    ~nonconstant,
                    int(SpatialStatisticStatus.ZERO_VARIANCE),
                    int(SpatialStatisticStatus.OK),
                ),
            ),
        ),
    ).astype(jnp.int32)
    p_value = jnp.where(valid, p_value, jnp.nan)
    evidence = SpatialStatisticEvidence(
        donor_count=donor_count,
        section_count=section_count,
        exchangeability_group_count=group_count,
        effective_spot_count=(
            jnp.sum(weights) ** 2 / jnp.maximum(jnp.sum(weights**2), 1.0e-30)
        ),
        graph_edge_count=jnp.sum(graph.relation.valid, dtype=jnp.int32),
        permutation_count=jnp.asarray(permutation_count, dtype=jnp.int32),
    )
    return SpatialAutocorrelationResult(
        statistic=observed,
        expected=expected,
        p_value=p_value,
        null_distribution=null,
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_MORAN_CONTRACT if statistic == "moran" else _GEARY_CONTRACT,
    )


def assay_autocorrelation_test(
    assay: SpatialAssay,
    feature_index: int,
    graph: SpatialNeighborGraph,
    key: Array,
    /,
    **kwargs: Any,
) -> SpatialAutocorrelationResult:
    """Run design-aware inference using donor, section, and density assay metadata."""
    if not isinstance(assay, SpatialAssay):
        raise TypeError("assay must be a SpatialAssay.")
    feature = int(feature_index)
    if feature < 0 or feature >= assay.data.feature_count:
        raise ValueError("feature_index is out of range.")
    forbidden = {"observation_weights", "donor_index", "section_index"}.intersection(
        kwargs
    )
    if forbidden:
        raise ValueError(
            f"Assay-derived design arguments cannot be overridden: {sorted(forbidden)}"
        )
    return spatial_autocorrelation_test(
        assay.data.values[:, feature],
        graph,
        key,
        observation_weights=assay.data.spot_weights,
        donor_index=assay.donor_index(),
        section_index=assay.section_index(),
        **kwargs,
    )


__all__ = [
    "SpatialAutocorrelationResult",
    "SpatialStatisticEvidence",
    "SpatialStatisticStatus",
    "assay_autocorrelation_test",
    "spatial_autocorrelation_test",
]
