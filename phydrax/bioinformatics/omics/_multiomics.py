#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...sparse import route_reduce
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    FeatureMapping,
    MethodKind,
    OutputKind,
)


MULTIOMICS_SUCCESS = 0
MULTIOMICS_NO_TRAINING_OVERLAP = 1
MULTIOMICS_NONFINITE = 2
MULTIOMICS_UNMAPPED_REFERENCE_FEATURE = 3
MULTIOMICS_PROVENANCE_MISMATCH = 4
MULTIOMICS_MISSING_ALL_MODALITIES = 5


def multiomics_status_name(status: int, /) -> str:
    """Return the stable name of a multimodal-alignment status code."""
    names = (
        "success",
        "no_joint_training_observations",
        "nonfinite_observed_value",
        "unmapped_reference_feature",
        "fitted_provenance_mismatch",
        "all_modalities_missing",
    )
    code = int(status)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown multimodal-alignment status {code}.")
    return names[code]


def _alignment_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "missing_modality_feature_relation_alignment",
        MethodKind.LEARNED,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.EXACT_AD,
        OutputKind.ARRAY,
        conditioning_statement=(
            "Alignment is conditioned on the supplied learned feature-relation "
            "weights and normalization fitted only on declared training samples."
        ),
        truncation_statement="No observations or relation routes are truncated.",
        capacity_semantics=(
            "Every feature route occupies one explicit EdgeRelation slot; invalid "
            "slots are masked rather than inferred."
        ),
        assumptions=(
            "Reference and query rows identify the same experimental observations.",
            "Missing modalities are structural missingness, not numerical zeros.",
        ),
        nondifferentiable_outputs=("modality_presence", "status", "valid"),
    )


class MultiomicAlignmentPlan(StrictModule):
    """Training provenance and sparse one-to-many query-to-reference relations."""

    training_sample_mask: Array
    feature_mapping: FeatureMapping
    relation_weights: Array
    fit_provenance_id: str = eqx.field(static=True)
    minimum_scale: float = eqx.field(static=True)
    learned: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        training_sample_mask: ArrayLike,
        feature_mapping: FeatureMapping,
        relation_weights: ArrayLike,
        /,
        *,
        fit_provenance_id: str,
        minimum_scale: float = 1e-6,
        learned: bool = True,
    ):
        training = jnp.asarray(training_sample_mask, dtype=bool)
        weights = jnp.asarray(relation_weights)
        if training.ndim != 1:
            raise ValueError("training_sample_mask must be rank one.")
        if not isinstance(feature_mapping, FeatureMapping):
            raise TypeError("feature_mapping must be a FeatureMapping.")
        if weights.shape != feature_mapping.relation.route_shape:
            raise ValueError("relation_weights must contain one value per mapping route.")
        if not fit_provenance_id:
            raise ValueError("fit_provenance_id must be non-empty.")
        scale = float(minimum_scale)
        if scale <= 0.0:
            raise ValueError("minimum_scale must be positive.")
        self.training_sample_mask = training
        self.feature_mapping = feature_mapping
        self.relation_weights = weights
        self.fit_provenance_id = str(fit_provenance_id)
        self.minimum_scale = scale
        self.learned = bool(learned)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multiomic-alignment-plan",
                "feature_mapping_id": feature_mapping.mapping_id,
                "fit_provenance_id": self.fit_provenance_id,
                "minimum_scale": scale,
                "learned": self.learned,
                "arrays": array_tree_fingerprint((training, weights)),
            }
        )


class MultiomicAlignmentEvidence(StrictModule):
    """Observed-modality, training fit, and feature-relation diagnostics."""

    modality_presence: Array
    joint_training_sample_count: Array
    fitted_on_training_only: Array
    fitted_provenance_match: Array
    target_feature_route_count: Array
    target_feature_weight: Array
    one_to_many_source_feature: Array
    observed_values_finite: Array


class MultiomicAlignmentResult(StrictModule):
    """Aligned values that preserve missing modality rather than fabricating it."""

    standardized_reference: Array
    mapped_query: Array
    joint_embedding: Array
    valid: Array
    status: Array
    evidence: MultiomicAlignmentEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    claim_kind: str = eqx.field(static=True)


def align_modalities(
    reference_values: ArrayLike,
    query_values: ArrayLike,
    reference_observed: ArrayLike,
    query_observed: ArrayLike,
    plan: MultiomicAlignmentPlan,
    /,
    *,
    expected_fit_provenance_id: str,
    method_contract: BioinformaticsMethodContract | None = None,
) -> MultiomicAlignmentResult:
    """Apply a training-fitted sparse feature alignment with explicit missingness."""
    reference = jnp.asarray(reference_values)
    query = jnp.asarray(query_values)
    reference_mask = jnp.asarray(reference_observed, dtype=bool)
    query_mask = jnp.asarray(query_observed, dtype=bool)
    if not isinstance(plan, MultiomicAlignmentPlan):
        raise TypeError("plan must be MultiomicAlignmentPlan.")
    if reference.ndim != 2 or query.ndim != 2:
        raise ValueError("Modality values must have shape (sample, feature).")
    if reference.shape[0] != query.shape[0]:
        raise ValueError("Reference and query modalities must share a sample axis.")
    sample_count = int(reference.shape[0])
    if reference_mask.shape != (sample_count,) or query_mask.shape != (sample_count,):
        raise ValueError("Observed masks must contain one value per sample.")
    if plan.training_sample_mask.shape != (sample_count,):
        raise ValueError("The plan training mask must match the sample axis.")
    relation = plan.feature_mapping.relation
    if (
        relation.source_size != query.shape[1]
        or relation.target_size != reference.shape[1]
    ):
        raise ValueError(
            "Feature relation spaces must match query and reference columns."
        )
    if not expected_fit_provenance_id:
        raise ValueError("expected_fit_provenance_id must be non-empty.")

    reference_finite = jnp.all(jnp.isfinite(reference), axis=1)
    query_finite = jnp.all(jnp.isfinite(query), axis=1)
    observed_finite = (~reference_mask | reference_finite) & (~query_mask | query_finite)
    joint_training = (
        plan.training_sample_mask & reference_mask & query_mask & observed_finite
    )
    training_count = jnp.sum(joint_training, dtype=jnp.int32)
    safe_training_count = jnp.maximum(training_count, 1)
    training_weight = joint_training[:, None]
    reference_mean = (
        jnp.sum(jnp.where(training_weight, reference, 0.0), axis=0) / safe_training_count
    )
    query_mean = (
        jnp.sum(jnp.where(training_weight, query, 0.0), axis=0) / safe_training_count
    )
    reference_variance = (
        jnp.sum(
            jnp.where(training_weight, (reference - reference_mean) ** 2, 0.0), axis=0
        )
        / safe_training_count
    )
    query_variance = (
        jnp.sum(jnp.where(training_weight, (query - query_mean) ** 2, 0.0), axis=0)
        / safe_training_count
    )
    reference_scale = jnp.maximum(jnp.sqrt(reference_variance), plan.minimum_scale)
    query_scale = jnp.maximum(jnp.sqrt(query_variance), plan.minimum_scale)
    reference_z = (reference - reference_mean) / reference_scale
    query_z = (query - query_mean) / query_scale
    reference_z = jnp.where(
        reference_mask[:, None] & reference_finite[:, None], reference_z, 0.0
    )
    query_z = jnp.where(query_mask[:, None] & query_finite[:, None], query_z, 0.0)

    effective_weight = plan.relation_weights * plan.feature_mapping.confidence
    route_valid = relation.valid & jnp.isfinite(effective_weight)
    safe_source = jnp.where(route_valid, relation.source_indices, 0)
    route_values = (
        query_z[:, safe_source] * jnp.where(route_valid, effective_weight, 0.0)[None, :]
    )
    mapped_sum = jax.vmap(lambda value: route_reduce(relation, value, reduction="sum"))(
        route_values
    )
    target_weight = route_reduce(
        relation,
        jnp.where(route_valid, jnp.abs(effective_weight), 0.0),
        reduction="sum",
    )
    mapped_query = jnp.where(
        target_weight[None, :] > 0.0,
        mapped_sum / jnp.where(target_weight[None, :] > 0.0, target_weight[None, :], 1.0),
        0.0,
    )
    mapped_query = jnp.where(query_mask[:, None], mapped_query, 0.0)
    modality_presence = jnp.stack((reference_mask, query_mask), axis=1)
    joint_embedding = jnp.concatenate(
        (reference_z, mapped_query, modality_presence.astype(reference_z.dtype)), axis=1
    )

    source_route_count = jax.ops.segment_sum(
        route_valid.astype(jnp.int32),
        jnp.where(route_valid, relation.source_indices, 0),
        relation.source_size,
    )
    target_route_count = jax.ops.segment_sum(
        route_valid.astype(jnp.int32),
        jnp.where(route_valid, relation.target_indices, 0),
        relation.target_size,
    )
    one_to_many = source_route_count > 1
    provenance_match = jnp.asarray(
        plan.fit_provenance_id == str(expected_fit_provenance_id)
    )
    all_features_mapped = jnp.all(target_route_count > 0)
    has_modality = reference_mask | query_mask
    global_status = jnp.where(
        training_count == 0,
        MULTIOMICS_NO_TRAINING_OVERLAP,
        jnp.where(
            ~jnp.all(observed_finite),
            MULTIOMICS_NONFINITE,
            jnp.where(
                ~all_features_mapped,
                MULTIOMICS_UNMAPPED_REFERENCE_FEATURE,
                jnp.where(
                    provenance_match,
                    MULTIOMICS_SUCCESS,
                    MULTIOMICS_PROVENANCE_MISMATCH,
                ),
            ),
        ),
    ).astype(jnp.int32)
    status = jnp.where(
        has_modality,
        global_status,
        MULTIOMICS_MISSING_ALL_MODALITIES,
    ).astype(jnp.int32)
    valid = (
        has_modality
        & observed_finite
        & (training_count > 0)
        & provenance_match
        & all_features_mapped
    )
    evidence = MultiomicAlignmentEvidence(
        modality_presence,
        training_count,
        jnp.asarray(True),
        provenance_match,
        target_route_count,
        target_weight,
        one_to_many,
        observed_finite,
    )
    return MultiomicAlignmentResult(
        reference_z,
        mapped_query,
        joint_embedding,
        valid,
        status,
        evidence,
        method_contract if method_contract is not None else _alignment_contract(),
        "learned_alignment",
    )


__all__ = [
    "MULTIOMICS_MISSING_ALL_MODALITIES",
    "MULTIOMICS_NONFINITE",
    "MULTIOMICS_NO_TRAINING_OVERLAP",
    "MULTIOMICS_PROVENANCE_MISMATCH",
    "MULTIOMICS_SUCCESS",
    "MULTIOMICS_UNMAPPED_REFERENCE_FEATURE",
    "MultiomicAlignmentEvidence",
    "MultiomicAlignmentPlan",
    "MultiomicAlignmentResult",
    "align_modalities",
    "multiomics_status_name",
]
