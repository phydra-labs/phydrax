#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class SourceSampleKind(str, Enum):
    """How source points for one image analysis were obtained."""

    AFFINE_PRNG = "affine_prng"
    EXPLICIT = "explicit"


class PolynomialImageAnalysisStatus(str, Enum):
    """Terminal status of a bounded numerical polynomial-image analysis."""

    DISCOVERED = "discovered"
    INSUFFICIENT_DISCOVERY_SAMPLES = "insufficient_discovery_samples"
    INSUFFICIENT_HELDOUT_SAMPLES = "insufficient_heldout_samples"
    RESOURCE_LIMIT = "resource_limit"
    NONFINITE_INPUT = "nonfinite_input"
    NONFINITE_IMAGE = "nonfinite_image"
    JACOBIAN_RANK_AMBIGUOUS = "jacobian_rank_ambiguous"
    JACOBIAN_RANK_INCONSISTENT = "jacobian_rank_inconsistent"
    INTERPOLATION_RANK_AMBIGUOUS = "interpolation_rank_ambiguous"
    NO_RELATION = "no_relation"
    HELDOUT_REJECTED = "heldout_rejected"


class EvidenceDisposition(str, Enum):
    """Whether one sharply delimited mathematical claim was established."""

    NOT_ASSESSED = "not_assessed"
    SUPPORTED = "supported"
    REJECTED = "rejected"


class PolynomialImageClaimEvidence(StrictModule, NonTrainableState):
    """Separate evidence ledgers for claims that do not imply one another.

    Numerical relation discovery is not exact containment. Exact containment does
    not establish equality of ideals, the real image, or any topological claim.
    """

    numerical_discovery: EvidenceDisposition = eqx.field(static=True)
    exact_containment: EvidenceDisposition = eqx.field(static=True)
    ideal_equality: EvidenceDisposition = eqx.field(static=True)
    real_geometry: EvidenceDisposition = eqx.field(static=True)
    topology: EvidenceDisposition = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        numerical_discovery: EvidenceDisposition = EvidenceDisposition.NOT_ASSESSED,
        exact_containment: EvidenceDisposition = EvidenceDisposition.NOT_ASSESSED,
        ideal_equality: EvidenceDisposition = EvidenceDisposition.NOT_ASSESSED,
        real_geometry: EvidenceDisposition = EvidenceDisposition.NOT_ASSESSED,
        topology: EvidenceDisposition = EvidenceDisposition.NOT_ASSESSED,
    ):
        values = (
            numerical_discovery,
            exact_containment,
            ideal_equality,
            real_geometry,
            topology,
        )
        if any(not isinstance(value, EvidenceDisposition) for value in values):
            raise TypeError(
                "Image claim dispositions must be EvidenceDisposition values."
            )
        (
            self.numerical_discovery,
            self.exact_containment,
            self.ideal_equality,
            self.real_geometry,
            self.topology,
        ) = values
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "polynomial-image-claim-evidence",
                "numerical_discovery": numerical_discovery.value,
                "exact_containment": exact_containment.value,
                "ideal_equality": ideal_equality.value,
                "real_geometry": real_geometry.value,
                "topology": topology.value,
            }
        )


class TargetMonomialSupport(StrictModule, NonTrainableState):
    """Canonical fixed monomial support used to search for target relations."""

    variable_labels: tuple[str, ...] = eqx.field(static=True)
    exponents: Array
    target_dimension: int = eqx.field(static=True)
    monomial_count: int = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        variable_labels: Sequence[str],
        exponents: ArrayLike,
        /,
    ):
        labels = tuple(str(label) for label in variable_labels)
        powers = np.asarray(exponents)
        if not labels or any(not label for label in labels):
            raise ValueError("Target variable labels must be non-empty.")
        if len(set(labels)) != len(labels):
            raise ValueError("Target variable labels must be unique.")
        if powers.ndim != 2 or powers.shape[1] != len(labels):
            raise ValueError(
                "Target exponents must have shape (monomial_count, target_dimension)."
            )
        if powers.shape[0] == 0:
            raise ValueError(
                "Target monomial support must contain at least one monomial."
            )
        if not np.issubdtype(powers.dtype, np.integer):
            raise TypeError("Target monomial exponents must be integers.")
        if np.any(powers < 0) or np.any(powers > np.iinfo(np.int32).max):
            raise ValueError(
                "Target monomial exponents must be non-negative int32 values."
            )
        powers = powers.astype(np.int32, copy=False)
        order = np.asarray(
            sorted(range(powers.shape[0]), key=lambda index: tuple(powers[index])),
            dtype=np.int32,
        )
        powers = powers[order]
        if powers.shape[0] > 1 and np.any(np.all(powers[1:] == powers[:-1], axis=1)):
            raise ValueError("Target monomial support cannot contain duplicate rows.")
        self.variable_labels = labels
        self.exponents = jnp.asarray(powers, dtype=jnp.int32)
        self.target_dimension = len(labels)
        self.monomial_count = powers.shape[0]
        self.support_id = canonical_fingerprint(
            {
                "kind": "target-monomial-support",
                "variable_labels": labels,
                "exponents": array_tree_fingerprint(powers),
            }
        )

    def evaluate(self, points: ArrayLike, /) -> Array:
        """Evaluate every fixed monomial at points with trailing target axis."""

        values = jnp.asarray(points)
        if values.ndim < 1 or values.shape[-1] != self.target_dimension:
            raise ValueError(
                "Target points must have trailing axis equal to target_dimension."
            )
        return jnp.prod(
            values[..., None, :] ** self.exponents,
            axis=-1,
        )


class PolynomialImageResourceEvidence(StrictModule, NonTrainableState):
    """Static host-resource accounting, separate from numerical acceptance."""

    sample_count: int = eqx.field(static=True)
    monomial_count: int = eqx.field(static=True)
    design_entries: int = eqx.field(static=True)
    estimated_svd_bytes: int = eqx.field(static=True)
    within_budget: bool = eqx.field(static=True)
    limiting_resource: str | None = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        sample_count: int,
        monomial_count: int,
        design_entries: int,
        estimated_svd_bytes: int,
        within_budget: bool,
        limiting_resource: str | None,
    ):
        counts = tuple(
            (
                sample_count,
                monomial_count,
                design_entries,
                estimated_svd_bytes,
            )
        )
        if any(value < 0 for value in counts):
            raise ValueError("Image-analysis resource counts must be non-negative.")
        limiting = None if limiting_resource is None else str(limiting_resource)
        if limiting is not None and not limiting:
            raise ValueError("limiting_resource must be non-empty or None.")
        if bool(within_budget) != (limiting is None):
            raise ValueError(
                "within_budget and limiting_resource must describe the same decision."
            )
        (
            self.sample_count,
            self.monomial_count,
            self.design_entries,
            self.estimated_svd_bytes,
        ) = counts
        self.within_budget = bool(within_budget)
        self.limiting_resource = limiting
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "polynomial-image-resource-evidence",
                "sample_count": counts[0],
                "monomial_count": counts[1],
                "design_entries": counts[2],
                "estimated_svd_bytes": counts[3],
                "within_budget": bool(within_budget),
                "limiting_resource": limiting,
            }
        )


class JacobianRankEvidence(StrictModule, NonTrainableState):
    """Two-sided numerical rank evidence over all sampled map Jacobians."""

    singular_values: Array
    lower_ranks: Array
    upper_ranks: Array
    lower_cutoffs: Array
    upper_cutoffs: Array
    rank_cutoff_gaps: Array
    condition_estimates: Array
    dimension_lower_bound: int = eqx.field(static=True)
    dimension_upper_bound: int = eqx.field(static=True)
    cutoff_resolved: bool = eqx.field(static=True)
    sample_consistent: bool = eqx.field(static=True)
    resolved: bool = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    svd_plan_ids: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        singular_values: ArrayLike,
        lower_ranks: ArrayLike,
        upper_ranks: ArrayLike,
        lower_cutoffs: ArrayLike,
        upper_cutoffs: ArrayLike,
        /,
        *,
        provider: str,
        svd_plan_ids: Sequence[str],
    ):
        singular = jnp.asarray(singular_values)
        lower = jnp.asarray(lower_ranks, dtype=jnp.int32)
        upper = jnp.asarray(upper_ranks, dtype=jnp.int32)
        low_cut = jnp.asarray(lower_cutoffs, dtype=singular.dtype)
        high_cut = jnp.asarray(upper_cutoffs, dtype=singular.dtype)
        if singular.ndim != 2:
            raise ValueError("Jacobian singular values must have shape (samples, modes).")
        expected = (singular.shape[0],)
        if any(value.shape != expected for value in (lower, upper, low_cut, high_cut)):
            raise ValueError("Jacobian rank arrays must have one entry per sample.")
        if np.any(np.asarray(lower) > np.asarray(upper)):
            raise ValueError("Jacobian lower ranks cannot exceed upper ranks.")
        central_cutoff = jnp.sqrt(low_cut * high_cut)
        cutoff_gaps = jnp.min(
            jnp.abs(singular - central_cutoff[:, None]),
            axis=-1,
        )
        retained = singular > high_cut[:, None]
        largest = jnp.max(singular, axis=-1)
        infinity = jnp.asarray(jnp.inf, dtype=singular.dtype)
        smallest_retained = jnp.min(
            jnp.where(retained, singular, infinity),
            axis=-1,
        )
        condition_estimates = jnp.where(
            lower > 0,
            largest / smallest_retained,
            infinity,
        )
        lower_dimension = int(np.max(np.asarray(lower), initial=0))
        upper_dimension = int(np.max(np.asarray(upper), initial=0))
        provider_ = str(provider)
        plan_ids = tuple(str(value) for value in svd_plan_ids)
        if not provider_ or any(not value for value in plan_ids):
            raise ValueError("Jacobian SVD provider and plan IDs must be non-empty.")
        if len(plan_ids) not in (0, singular.shape[0]):
            raise ValueError("Jacobian SVD plan IDs must be empty or sample-aligned.")
        self.singular_values = singular
        self.lower_ranks = lower
        self.upper_ranks = upper
        self.lower_cutoffs = low_cut
        self.upper_cutoffs = high_cut
        self.rank_cutoff_gaps = cutoff_gaps
        self.condition_estimates = condition_estimates
        self.dimension_lower_bound = lower_dimension
        self.dimension_upper_bound = upper_dimension
        lower_host = np.asarray(lower)
        upper_host = np.asarray(upper)
        cutoff_resolved = bool(np.all(lower_host == upper_host))
        sample_consistent = bool(
            lower_host.size
            and np.all(lower_host == lower_host[0])
            and np.all(upper_host == upper_host[0])
        )
        self.cutoff_resolved = cutoff_resolved
        self.sample_consistent = sample_consistent
        self.resolved = cutoff_resolved and sample_consistent
        self.provider = provider_
        self.svd_plan_ids = plan_ids
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "polynomial-image-jacobian-rank-evidence",
                "arrays": array_tree_fingerprint(
                    (
                        singular,
                        lower,
                        upper,
                        low_cut,
                        high_cut,
                        cutoff_gaps,
                        condition_estimates,
                    )
                ),
                "dimension_lower_bound": lower_dimension,
                "dimension_upper_bound": upper_dimension,
                "resolved": self.resolved,
                "provider": provider_,
                "cutoff_resolved": cutoff_resolved,
                "sample_consistent": sample_consistent,
                "svd_plan_ids": plan_ids,
            }
        )


class TargetRelationEvidence(StrictModule, NonTrainableState):
    """Fixed-shape nullspace candidates and independent held-out residuals."""

    singular_values: Array
    rank_lower_bound: int = eqx.field(static=True)
    rank_upper_bound: int = eqx.field(static=True)
    rank_lower_cutoff: Array
    rank_upper_cutoff: Array
    candidate_coefficients: Array
    candidate_active: Array
    discovery_residuals: Array
    heldout_residuals: Array
    heldout_tolerances: Array
    heldout_accepted: Array
    relation_count: int = eqx.field(static=True)
    rank_resolved: bool = eqx.field(static=True)
    validation_accepted: bool = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    svd_plan_id: str | None = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        singular_values: ArrayLike,
        *,
        rank_lower_bound: int,
        rank_upper_bound: int,
        rank_lower_cutoff: ArrayLike,
        rank_upper_cutoff: ArrayLike,
        candidate_coefficients: ArrayLike,
        candidate_active: ArrayLike,
        discovery_residuals: ArrayLike,
        heldout_residuals: ArrayLike,
        heldout_tolerances: ArrayLike,
        heldout_accepted: ArrayLike,
        provider: str,
        svd_plan_id: str | None,
    ):
        singular = jnp.asarray(singular_values)
        coefficients = jnp.asarray(candidate_coefficients)
        active = jnp.asarray(candidate_active, dtype=jnp.bool_)
        discovery = jnp.asarray(discovery_residuals, dtype=singular.dtype)
        heldout = jnp.asarray(heldout_residuals, dtype=singular.dtype)
        tolerances = jnp.asarray(heldout_tolerances, dtype=singular.dtype)
        accepted = jnp.asarray(heldout_accepted, dtype=jnp.bool_)
        if singular.ndim != 1 or coefficients.ndim != 2:
            raise ValueError("Relation singular values and candidates have invalid rank.")
        capacity = coefficients.shape[0]
        if coefficients.shape[1] != singular.shape[0]:
            raise ValueError("Candidate width must equal the monomial count.")
        if any(
            value.shape != (capacity,)
            for value in (active, discovery, heldout, tolerances, accepted)
        ):
            raise ValueError("Relation candidate diagnostics must share capacity.")
        lower_rank = int(rank_lower_bound)
        upper_rank = int(rank_upper_bound)
        if not (0 <= lower_rank <= upper_rank <= singular.shape[0]):
            raise ValueError("Relation rank bounds are invalid.")
        provider_ = str(provider)
        plan_id = None if svd_plan_id is None else str(svd_plan_id)
        if not provider_ or (plan_id is not None and not plan_id):
            raise ValueError("Relation SVD provider identity is invalid.")
        relation_count = int(np.count_nonzero(np.asarray(active)))
        validation_accepted = bool(np.all(np.asarray(accepted)[np.asarray(active)]))
        self.singular_values = singular
        self.rank_lower_bound = lower_rank
        self.rank_upper_bound = upper_rank
        self.rank_lower_cutoff = jnp.asarray(
            rank_lower_cutoff, dtype=singular.dtype
        ).reshape(())
        self.rank_upper_cutoff = jnp.asarray(
            rank_upper_cutoff, dtype=singular.dtype
        ).reshape(())
        self.candidate_coefficients = coefficients
        self.candidate_active = active
        self.discovery_residuals = discovery
        self.heldout_residuals = heldout
        self.heldout_tolerances = tolerances
        self.heldout_accepted = accepted
        self.relation_count = relation_count
        self.rank_resolved = lower_rank == upper_rank
        self.validation_accepted = validation_accepted
        self.provider = provider_
        self.svd_plan_id = plan_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "polynomial-image-target-relation-evidence",
                "arrays": array_tree_fingerprint(
                    (
                        singular,
                        coefficients,
                        active,
                        discovery,
                        heldout,
                        tolerances,
                        accepted,
                    )
                ),
                "rank_lower_bound": lower_rank,
                "rank_upper_bound": upper_rank,
                "rank_lower_cutoff": self.rank_lower_cutoff,
                "rank_upper_cutoff": self.rank_upper_cutoff,
                "relation_count": relation_count,
                "rank_resolved": self.rank_resolved,
                "validation_accepted": validation_accepted,
                "provider": provider_,
                "svd_plan_id": plan_id,
            }
        )

    @property
    def relation_coefficients(self) -> Array:
        """Return active normalized candidates without padded inactive rows."""

        return self.candidate_coefficients[: self.relation_count]


class PolynomialImageAnalysisResult(StrictModule, NonTrainableState):
    """Numerical image evidence with no implicit, real, ideal, or topology promotion."""

    status: PolynomialImageAnalysisStatus = eqx.field(static=True)
    source_kind: SourceSampleKind = eqx.field(static=True)
    source_points: Array
    heldout_source_points: Array
    target_points: Array
    heldout_target_points: Array
    jacobian_rank: JacobianRankEvidence
    relations: TargetRelationEvidence
    resources: PolynomialImageResourceEvidence
    claims: PolynomialImageClaimEvidence
    map_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: PolynomialImageAnalysisStatus,
        source_kind: SourceSampleKind,
        source_points: ArrayLike,
        heldout_source_points: ArrayLike,
        target_points: ArrayLike,
        heldout_target_points: ArrayLike,
        jacobian_rank: JacobianRankEvidence,
        relations: TargetRelationEvidence,
        resources: PolynomialImageResourceEvidence,
        claims: PolynomialImageClaimEvidence,
        /,
        *,
        map_id: str,
        plan_id: str,
    ):
        if not isinstance(status, PolynomialImageAnalysisStatus):
            raise TypeError("status must be PolynomialImageAnalysisStatus.")
        if not isinstance(source_kind, SourceSampleKind):
            raise TypeError("source_kind must be SourceSampleKind.")
        if not isinstance(jacobian_rank, JacobianRankEvidence):
            raise TypeError("jacobian_rank must be JacobianRankEvidence.")
        if not isinstance(relations, TargetRelationEvidence):
            raise TypeError("relations must be TargetRelationEvidence.")
        if not isinstance(resources, PolynomialImageResourceEvidence):
            raise TypeError("resources must be PolynomialImageResourceEvidence.")
        if not isinstance(claims, PolynomialImageClaimEvidence):
            raise TypeError("claims must be PolynomialImageClaimEvidence.")
        map_id_ = str(map_id)
        plan_id_ = str(plan_id)
        if not map_id_ or not plan_id_:
            raise ValueError("Image analysis map_id and plan_id must be non-empty.")
        source = jnp.asarray(source_points)
        heldout_source = jnp.asarray(heldout_source_points)
        target = jnp.asarray(target_points)
        heldout_target = jnp.asarray(heldout_target_points)
        if any(
            value.ndim != 2 for value in (source, heldout_source, target, heldout_target)
        ):
            raise ValueError("Image analysis point arrays must be matrices.")
        if (
            source.shape[0] != target.shape[0]
            or heldout_source.shape[0] != heldout_target.shape[0]
        ):
            raise ValueError("Source and target sample counts must agree.")
        self.status = status
        self.source_kind = source_kind
        self.source_points = source
        self.heldout_source_points = heldout_source
        self.target_points = target
        self.heldout_target_points = heldout_target
        self.jacobian_rank = jacobian_rank
        self.relations = relations
        self.resources = resources
        self.claims = claims
        self.map_id = map_id_
        self.plan_id = plan_id_
        self.result_id = canonical_fingerprint(
            {
                "kind": "polynomial-image-analysis-result",
                "status": status.value,
                "source_kind": source_kind.value,
                "points": array_tree_fingerprint(
                    (source, heldout_source, target, heldout_target)
                ),
                "jacobian_rank": jacobian_rank.evidence_id,
                "relations": relations.evidence_id,
                "resources": resources.evidence_id,
                "claims": claims.evidence_id,
                "map": map_id_,
                "plan": plan_id_,
            }
        )

    @property
    def accepted(self) -> bool:
        return self.status is PolynomialImageAnalysisStatus.DISCOVERED


class ExactCompositionRemainder(StrictModule, NonTrainableState):
    """Canonical sparse exact remainder for one composed target relation."""

    equation_label: str = eqx.field(static=True)
    terms: tuple[tuple[tuple[int, ...], str], ...] = eqx.field(static=True)
    remainder_id: str = eqx.field(static=True)

    def __init__(
        self,
        equation_label: str,
        terms: Sequence[tuple[Sequence[int], str]],
        /,
    ):
        label = str(equation_label)
        canonical_terms = tuple(
            (tuple(exponent), str(coefficient)) for exponent, coefficient in terms
        )
        if not label:
            raise ValueError("Exact composition equation labels must be non-empty.")
        if any(
            any(power < 0 for power in exponent) or not coefficient
            for exponent, coefficient in canonical_terms
        ):
            raise ValueError(
                "Exact composition terms must be canonical and non-negative."
            )
        self.equation_label = label
        self.terms = canonical_terms
        self.remainder_id = canonical_fingerprint(
            {
                "kind": "exact-polynomial-composition-remainder",
                "equation_label": label,
                "terms": canonical_terms,
            }
        )

    @property
    def is_zero(self) -> bool:
        return not self.terms


class ExactPolynomialContainmentResult(StrictModule, NonTrainableState):
    """Exact composition evidence for containment, and nothing stronger."""

    remainders: tuple[ExactCompositionRemainder, ...]
    contained: bool = eqx.field(static=True)
    claims: PolynomialImageClaimEvidence
    map_id: str = eqx.field(static=True)
    relation_system_id: str = eqx.field(static=True)
    proof_id: str = eqx.field(static=True)

    def __init__(
        self,
        remainders: Sequence[ExactCompositionRemainder],
        claims: PolynomialImageClaimEvidence,
        /,
        *,
        map_id: str,
        relation_system_id: str,
    ):
        values = tuple(remainders)
        if not values or any(
            not isinstance(value, ExactCompositionRemainder) for value in values
        ):
            raise TypeError("Exact containment needs non-empty composition remainders.")
        if not isinstance(claims, PolynomialImageClaimEvidence):
            raise TypeError("claims must be PolynomialImageClaimEvidence.")
        map_id_ = str(map_id)
        relation_id = str(relation_system_id)
        if not map_id_ or not relation_id:
            raise ValueError("Exact containment identities must be non-empty.")
        contained = all(value.is_zero for value in values)
        expected = (
            EvidenceDisposition.SUPPORTED if contained else EvidenceDisposition.REJECTED
        )
        if claims.exact_containment is not expected:
            raise ValueError("Exact containment claims must agree with composition.")
        self.remainders = values
        self.contained = contained
        self.claims = claims
        self.map_id = map_id_
        self.relation_system_id = relation_id
        self.proof_id = canonical_fingerprint(
            {
                "kind": "exact-polynomial-image-containment",
                "map": map_id_,
                "relations": relation_id,
                "remainders": [value.remainder_id for value in values],
                "claims": claims.evidence_id,
            }
        )


__all__ = [
    "EvidenceDisposition",
    "ExactCompositionRemainder",
    "ExactPolynomialContainmentResult",
    "JacobianRankEvidence",
    "PolynomialImageAnalysisResult",
    "PolynomialImageAnalysisStatus",
    "PolynomialImageClaimEvidence",
    "PolynomialImageResourceEvidence",
    "SourceSampleKind",
    "TargetMonomialSupport",
    "TargetRelationEvidence",
]
