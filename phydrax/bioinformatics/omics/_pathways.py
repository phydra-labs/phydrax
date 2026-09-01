#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...sparse import route_reduce
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExchangeabilityPlan,
    ExecutionKind,
    FeatureMapping,
    MethodKind,
    OntologyGraph,
    OutputKind,
)


PATHWAY_SUCCESS = 0
PATHWAY_INVALID_STATISTIC = 1
PATHWAY_MISSING_EXCHANGEABILITY = 2
PATHWAY_CORRECTED_EMBEDDING_FORBIDDEN = 3
PATHWAY_UNMAPPED_SET = 4
PATHWAY_CAPACITY_EXCEEDED = 5
PATHWAY_ONTOLOGY_NONCONVERGED = 6


def pathway_status_name(status: int, /) -> str:
    """Return the stable name of an ontology feature-set test status code."""
    names = (
        "success",
        "invalid_feature_statistic",
        "missing_exchangeability_plan",
        "corrected_embedding_forbidden",
        "unmapped_feature_set",
        "declared_capacity_exceeded",
        "ontology_propagation_nonconverged",
    )
    code = int(status)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown feature-set test status {code}.")
    return names[code]


def _pathway_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "ontology_aware_feature_set_permutation_test",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.STOCHASTIC_ESTIMATE,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "P-values condition on the supplied exchangeable permutation statistics, "
            "feature-set mapping, and ontology parent relations."
        ),
        truncation_statement=(
            "Permutation and ontology capacities are preflighted; exceedance is an "
            "invalid result and never silently discards permutations or relations."
        ),
        capacity_semantics=(
            "maximum_permutations and maximum_ontology_steps are explicit compile "
            "capacities with observable failure status."
        ),
        assumptions=(
            "Permutation rows were generated under the declared exchangeability plan.",
            "Confirmatory statistics were not computed from a corrected embedding.",
        ),
        nondifferentiable_outputs=("p_value", "ontology_adjusted_p_value", "status"),
    )


class OntologyFeatureSetTestPlan(StrictModule):
    """Bounded confirmatory resampling plan with explicit input provenance."""

    maximum_permutations: int = eqx.field(static=True)
    maximum_ontology_steps: int = eqx.field(static=True)
    two_sided: bool = eqx.field(static=True)
    corrected_embedding_used: bool = eqx.field(static=True)
    input_provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_permutations: int,
        maximum_ontology_steps: int,
        input_provenance_id: str,
        two_sided: bool = True,
        corrected_embedding_used: bool = False,
    ):
        permutations = int(maximum_permutations)
        steps = int(maximum_ontology_steps)
        if permutations < 1 or steps < 1:
            raise ValueError("Feature-set test capacities must be positive.")
        if not input_provenance_id:
            raise ValueError("input_provenance_id must be non-empty.")
        self.maximum_permutations = permutations
        self.maximum_ontology_steps = steps
        self.two_sided = bool(two_sided)
        self.corrected_embedding_used = bool(corrected_embedding_used)
        self.input_provenance_id = str(input_provenance_id)


class OntologyFeatureSetEvidence(StrictModule):
    """Mapping, exchangeability, capacity, and hierarchy evidence for a test."""

    feature_set_route_count: Array
    feature_set_weight: Array
    permutation_count: Array
    exchangeability_declared: Array
    corrected_embedding_absent: Array
    capacities_satisfied: Array
    ontology_converged: Array
    raw_p_value: Array
    ontology_adjusted_p_value: Array
    replicate_unit: str = eqx.field(static=True)


class OntologyFeatureSetTestResult(StrictModule):
    """Feature-set scores and ontology-gated permutation p-values."""

    observed_score: Array
    permutation_score: Array
    p_value: Array
    ontology_adjusted_p_value: Array
    valid: Array
    status: Array
    evidence: OntologyFeatureSetEvidence
    exchangeability: ExchangeabilityPlan | None
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    claim_kind: str = eqx.field(static=True)


def ontology_feature_set_test(
    feature_statistic: ArrayLike,
    permutation_statistics: ArrayLike,
    membership: FeatureMapping,
    ontology: OntologyGraph,
    plan: OntologyFeatureSetTestPlan,
    /,
    *,
    exchangeability: ExchangeabilityPlan | None,
    method_contract: BioinformaticsMethodContract | None = None,
) -> OntologyFeatureSetTestResult:
    """Test feature sets and gate child significance through ontology parents."""
    observed = jnp.asarray(feature_statistic)
    permutations = jnp.asarray(permutation_statistics)
    if not isinstance(membership, FeatureMapping):
        raise TypeError("membership must be a FeatureMapping.")
    if not isinstance(ontology, OntologyGraph):
        raise TypeError("ontology must be an OntologyGraph.")
    if not isinstance(plan, OntologyFeatureSetTestPlan):
        raise TypeError("plan must be OntologyFeatureSetTestPlan.")
    if observed.ndim != 1 or permutations.ndim != 2:
        raise ValueError("Feature statistic and permutations must be rank one and two.")
    if permutations.shape[1] != observed.shape[0]:
        raise ValueError("Permutation statistics must share the feature axis.")
    if membership.source.capacity != observed.shape[0]:
        raise ValueError("Membership source dictionary must match feature statistics.")
    if membership.target.dictionary_id != ontology.features.dictionary_id:
        raise ValueError("Membership targets must be the ontology feature dictionary.")

    relation = membership.relation
    target_count = relation.target_size
    route_valid = (
        relation.valid
        & jnp.isfinite(membership.confidence)
        & (membership.confidence > 0.0)
    )
    safe_source = jnp.where(route_valid, relation.source_indices, 0)
    route_weight = jnp.where(route_valid, membership.confidence, 0.0)
    set_weight = route_reduce(relation, route_weight, reduction="sum")
    route_count = jax.ops.segment_sum(
        route_valid.astype(jnp.int32),
        jnp.where(route_valid, relation.target_indices, 0),
        target_count,
    )

    def aggregate(values: Array) -> Array:
        routed = values[safe_source] * route_weight
        total = route_reduce(relation, routed, reduction="sum")
        return jnp.where(
            set_weight > 0.0, total / jnp.where(set_weight > 0.0, set_weight, 1.0), 0.0
        )

    observed_score = aggregate(observed)
    permutation_score = jax.vmap(aggregate)(permutations)
    observed_test = jnp.abs(observed_score) if plan.two_sided else observed_score
    permutation_test = jnp.abs(permutation_score) if plan.two_sided else permutation_score
    exceedances = jnp.sum(
        permutation_test >= observed_test[None, :], axis=0, dtype=jnp.int32
    )
    permutation_count = int(permutations.shape[0])
    p_value = (exceedances + 1.0) / (permutation_count + 1.0)

    ontology_relation = ontology.relation
    if (
        ontology_relation.source_size != target_count
        or ontology_relation.target_size != target_count
    ):
        raise ValueError("Ontology relation must be internal to feature-set nodes.")
    ontology_valid = ontology_relation.valid
    child = jnp.where(ontology_valid, ontology_relation.source_indices, 0)
    parent = jnp.where(ontology_valid, ontology_relation.target_indices, 0)

    def propagate(_: int, values: Array) -> Array:
        parent_p = jnp.where(ontology_valid, values[parent], 0.0)
        inherited = jax.ops.segment_max(parent_p, child, target_count)
        return jnp.maximum(values, inherited)

    adjusted = jax.lax.fori_loop(0, plan.maximum_ontology_steps, propagate, p_value)
    next_adjusted = propagate(0, adjusted)
    ontology_converged = jnp.all(jnp.isclose(next_adjusted, adjusted))
    finite = jnp.all(jnp.isfinite(observed)) & jnp.all(jnp.isfinite(permutations))
    exchangeability_declared = jnp.asarray(exchangeability is not None)
    corrected_embedding_absent = jnp.asarray(not plan.corrected_embedding_used)
    capacities_satisfied = jnp.asarray(
        permutation_count <= plan.maximum_permutations
        and target_count <= plan.maximum_ontology_steps
    )
    all_sets_mapped = jnp.all(route_count > 0)
    valid = (
        finite
        & exchangeability_declared
        & corrected_embedding_absent
        & capacities_satisfied
        & all_sets_mapped
        & ontology_converged
    )
    status = jnp.where(
        ~finite,
        PATHWAY_INVALID_STATISTIC,
        jnp.where(
            ~exchangeability_declared,
            PATHWAY_MISSING_EXCHANGEABILITY,
            jnp.where(
                ~corrected_embedding_absent,
                PATHWAY_CORRECTED_EMBEDDING_FORBIDDEN,
                jnp.where(
                    ~all_sets_mapped,
                    PATHWAY_UNMAPPED_SET,
                    jnp.where(
                        ~capacities_satisfied,
                        PATHWAY_CAPACITY_EXCEEDED,
                        jnp.where(
                            ontology_converged,
                            PATHWAY_SUCCESS,
                            PATHWAY_ONTOLOGY_NONCONVERGED,
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = OntologyFeatureSetEvidence(
        route_count,
        set_weight,
        jnp.asarray(permutation_count, dtype=jnp.int32),
        exchangeability_declared,
        corrected_embedding_absent,
        capacities_satisfied,
        ontology_converged,
        p_value,
        adjusted,
        "experimental_unit",
    )
    return OntologyFeatureSetTestResult(
        observed_score,
        permutation_score,
        p_value,
        adjusted,
        valid,
        status,
        evidence,
        exchangeability,
        method_contract if method_contract is not None else _pathway_contract(),
        "confirmatory_resampling_test",
    )


__all__ = [
    "PATHWAY_CAPACITY_EXCEEDED",
    "PATHWAY_CORRECTED_EMBEDDING_FORBIDDEN",
    "PATHWAY_INVALID_STATISTIC",
    "PATHWAY_MISSING_EXCHANGEABILITY",
    "PATHWAY_ONTOLOGY_NONCONVERGED",
    "PATHWAY_SUCCESS",
    "PATHWAY_UNMAPPED_SET",
    "OntologyFeatureSetEvidence",
    "OntologyFeatureSetTestPlan",
    "OntologyFeatureSetTestResult",
    "ontology_feature_set_test",
    "pathway_status_name",
]
