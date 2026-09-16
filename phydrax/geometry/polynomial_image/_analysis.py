#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, MaterializationPolicy, RankPolicy
from ...linalg.svd import svd, SVDProblem, SVDResourcePolicy, SVDSolvePolicy
from ._contracts import (
    EvidenceDisposition,
    JacobianRankEvidence,
    PolynomialImageAnalysisResult,
    PolynomialImageAnalysisStatus,
    PolynomialImageClaimEvidence,
    PolynomialImageResourceEvidence,
    SourceSampleKind,
    TargetMonomialSupport,
    TargetRelationEvidence,
)
from ._map import SparsePolynomialMap


_SVD_PROVIDER = "phydrax.linalg.svd.DenseSVD"


class PolynomialImageAnalysisPolicy(StrictModule, NonTrainableState):
    """Numerical decisions and hard host-resource bounds for image discovery."""

    rank_relative_tolerance: float = eqx.field(static=True)
    rank_ambiguity_factor: float = eqx.field(static=True)
    heldout_absolute_tolerance: float = eqx.field(static=True)
    heldout_relative_tolerance: float = eqx.field(static=True)
    maximum_samples: int = eqx.field(static=True)
    maximum_monomials: int = eqx.field(static=True)
    maximum_design_entries: int = eqx.field(static=True)
    maximum_svd_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        rank_relative_tolerance: float = 1.0e-8,
        rank_ambiguity_factor: float = 8.0,
        heldout_absolute_tolerance: float = 1.0e-9,
        heldout_relative_tolerance: float = 1.0e-7,
        maximum_samples: int = 100_000,
        maximum_monomials: int = 2_048,
        maximum_design_entries: int = 20_000_000,
        maximum_svd_bytes: int = 1_073_741_824,
    ):
        rank_tolerance = float(rank_relative_tolerance)
        ambiguity = float(rank_ambiguity_factor)
        absolute = float(heldout_absolute_tolerance)
        relative = float(heldout_relative_tolerance)
        if not math.isfinite(rank_tolerance) or rank_tolerance <= 0.0:
            raise ValueError("rank_relative_tolerance must be finite and positive.")
        if not math.isfinite(ambiguity) or ambiguity <= 1.0:
            raise ValueError("rank_ambiguity_factor must be finite and greater than one.")
        if any(not math.isfinite(value) or value < 0.0 for value in (absolute, relative)):
            raise ValueError("Held-out tolerances must be finite and non-negative.")
        limits = tuple(
            int(value)
            for value in (
                maximum_samples,
                maximum_monomials,
                maximum_design_entries,
                maximum_svd_bytes,
            )
        )
        if any(value <= 0 for value in limits):
            raise ValueError("Image-analysis resource limits must be positive.")
        self.rank_relative_tolerance = rank_tolerance
        self.rank_ambiguity_factor = ambiguity
        self.heldout_absolute_tolerance = absolute
        self.heldout_relative_tolerance = relative
        (
            self.maximum_samples,
            self.maximum_monomials,
            self.maximum_design_entries,
            self.maximum_svd_bytes,
        ) = limits
        self.policy_id = canonical_fingerprint(
            {
                "kind": "polynomial-image-analysis-policy-v1",
                "rank_relative_tolerance": rank_tolerance,
                "rank_ambiguity_factor": ambiguity,
                "heldout_absolute_tolerance": absolute,
                "heldout_relative_tolerance": relative,
                "maximum_samples": limits[0],
                "maximum_monomials": limits[1],
                "maximum_design_entries": limits[2],
                "maximum_svd_bytes": limits[3],
            }
        )


class PolynomialImageAnalysisPlan(StrictModule, NonTrainableState):
    """Fixed support, samples, tolerances, and resources for image analysis."""

    polynomial_map: SparsePolynomialMap
    target_support: TargetMonomialSupport
    policy: PolynomialImageAnalysisPolicy
    source_kind: SourceSampleKind = eqx.field(static=True)
    discovery_sample_count: int = eqx.field(static=True)
    heldout_sample_count: int = eqx.field(static=True)
    root_key_data: Array
    lower_bounds: Array
    upper_bounds: Array
    explicit_source_points: Array
    explicit_heldout_source_points: Array
    map_support_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        polynomial_map: SparsePolynomialMap,
        target_support: TargetMonomialSupport,
        /,
        *,
        source_samples: ArrayLike | None = None,
        heldout_source_samples: ArrayLike | None = None,
        discovery_sample_count: int = 64,
        heldout_sample_count: int = 16,
        key: ArrayLike | None = None,
        lower_bounds: ArrayLike | float = -1.0,
        upper_bounds: ArrayLike | float = 1.0,
        policy: PolynomialImageAnalysisPolicy | None = None,
    ):
        if not isinstance(polynomial_map, SparsePolynomialMap):
            raise TypeError("polynomial_map must be a SparsePolynomialMap.")
        if not isinstance(target_support, TargetMonomialSupport):
            raise TypeError("target_support must be a TargetMonomialSupport.")
        if target_support.target_dimension != polynomial_map.target_dimension:
            raise ValueError("Target monomials and polynomial map dimensions differ.")
        if target_support.variable_labels != polynomial_map.target_labels:
            raise ValueError("Target monomial labels must match map coordinate labels.")
        policy_ = PolynomialImageAnalysisPolicy() if policy is None else policy
        if not isinstance(policy_, PolynomialImageAnalysisPolicy):
            raise TypeError("policy must be a PolynomialImageAnalysisPolicy.")
        dimension = polynomial_map.source_dimension
        key_data = np.zeros((2,), dtype=np.uint32)
        if source_samples is None:
            if key is None:
                raise ValueError("Affine source sampling requires an explicit PRNG key.")
            key_data = np.asarray(jr.key_data(key), dtype=np.uint32)
            if key_data.shape != (2,):
                raise ValueError("Affine source PRNG key must contain one key.")
            if any(
                group.geometry != "affine"
                for group in polynomial_map.system.support.groups
            ):
                raise ValueError(
                    "PRNG source sampling requires affine variable groups; "
                    "supply explicit source samples for other source geometries."
                )
            if heldout_source_samples is not None:
                raise ValueError(
                    "heldout_source_samples requires explicit source_samples."
                )
            discovery_count = int(discovery_sample_count)
            heldout_count = int(heldout_sample_count)
            if discovery_count < 0 or heldout_count < 0:
                raise ValueError("Affine sample counts must be non-negative.")
            lower = np.broadcast_to(
                np.asarray(lower_bounds, dtype=float), (dimension,)
            ).copy()
            upper = np.broadcast_to(
                np.asarray(upper_bounds, dtype=float), (dimension,)
            ).copy()
            if (
                not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower >= upper)
            ):
                raise ValueError("Affine sample bounds must be finite and increasing.")
            explicit = np.zeros((0, dimension), dtype=float)
            explicit_heldout = np.zeros((0, dimension), dtype=float)
            kind = SourceSampleKind.AFFINE_PRNG
        else:
            if key is not None:
                raise ValueError("A PRNG key is invalid with explicit source samples.")
            if heldout_source_samples is None:
                raise ValueError(
                    "Explicit source samples require explicit held-out samples."
                )
            explicit = np.asarray(source_samples)
            explicit_heldout = np.asarray(heldout_source_samples)
            if (
                explicit.ndim != 2
                or explicit_heldout.ndim != 2
                or explicit.shape[1] != dimension
                or explicit_heldout.shape[1] != dimension
            ):
                raise ValueError(
                    "Explicit source points must have shape (samples, source_dimension)."
                )
            if not np.issubdtype(explicit.dtype, np.inexact):
                explicit = explicit.astype(float)
            if not np.issubdtype(explicit_heldout.dtype, np.inexact):
                explicit_heldout = explicit_heldout.astype(float)
            dtype = np.result_type(explicit.dtype, explicit_heldout.dtype)
            explicit = explicit.astype(dtype, copy=False)
            explicit_heldout = explicit_heldout.astype(dtype, copy=False)
            discovery_count = explicit.shape[0]
            heldout_count = explicit_heldout.shape[0]
            lower = np.zeros((dimension,), dtype=float)
            upper = np.ones((dimension,), dtype=float)
            kind = SourceSampleKind.EXPLICIT
        self.polynomial_map = polynomial_map
        self.target_support = target_support
        self.policy = policy_
        self.source_kind = kind
        self.discovery_sample_count = discovery_count
        self.heldout_sample_count = heldout_count
        self.root_key_data = jnp.asarray(key_data, dtype=jnp.uint32)
        self.lower_bounds = jnp.asarray(lower)
        self.upper_bounds = jnp.asarray(upper)
        self.explicit_source_points = jnp.asarray(explicit)
        self.explicit_heldout_source_points = jnp.asarray(explicit_heldout)
        self.map_support_id = polynomial_map.system.support.support_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polynomial-image-analysis-plan-v1",
                "map_support": self.map_support_id,
                "coefficient_dtype": np.dtype(
                    polynomial_map.system.coefficients.dtype
                ).str,
                "target_support": target_support.support_id,
                "source_kind": kind.value,
                "sample_counts": [discovery_count, heldout_count],
                "sampling_data": array_tree_fingerprint(
                    (
                        self.root_key_data,
                        self.lower_bounds,
                        self.upper_bounds,
                        self.explicit_source_points,
                        self.explicit_heldout_source_points,
                    )
                ),
                "policy": policy_.policy_id,
            }
        )

    def prepare(self) -> PreparedPolynomialImageAnalysis:
        """Sample once and run the initial bounded numerical analysis."""

        return prepare_polynomial_image_analysis(self)


class PreparedPolynomialImageAnalysis(StrictModule, NonTrainableState):
    """Frozen source samples and latest result for coefficient-only refreshes."""

    plan: PolynomialImageAnalysisPlan
    polynomial_map: SparsePolynomialMap
    source_points: Array
    heldout_source_points: Array
    result: PolynomialImageAnalysisResult
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PolynomialImageAnalysisPlan,
        polynomial_map: SparsePolynomialMap,
        source_points: ArrayLike,
        heldout_source_points: ArrayLike,
        result: PolynomialImageAnalysisResult,
        /,
        *,
        numeric_version: ArrayLike = 0,
    ):
        if not isinstance(plan, PolynomialImageAnalysisPlan):
            raise TypeError("plan must be a PolynomialImageAnalysisPlan.")
        if not isinstance(polynomial_map, SparsePolynomialMap):
            raise TypeError("polynomial_map must be a SparsePolynomialMap.")
        if not isinstance(result, PolynomialImageAnalysisResult):
            raise TypeError("result must be a PolynomialImageAnalysisResult.")
        source = jnp.asarray(source_points)
        heldout = jnp.asarray(heldout_source_points)
        if source.shape != (
            plan.discovery_sample_count,
            polynomial_map.source_dimension,
        ) or heldout.shape != (
            plan.heldout_sample_count,
            polynomial_map.source_dimension,
        ):
            raise ValueError("Prepared polynomial-image samples have wrong shapes.")
        version = jnp.asarray(numeric_version, dtype=jnp.int32).reshape(())
        if int(np.asarray(version)) < 0:
            raise ValueError("numeric_version must be non-negative.")
        self.plan = plan
        self.polynomial_map = polynomial_map
        self.source_points = source
        self.heldout_source_points = heldout
        self.result = result
        self.numeric_version = version
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-polynomial-image-analysis-v1",
                "plan": plan.plan_id,
                "samples": array_tree_fingerprint((source, heldout)),
            }
        )

    def refresh(
        self,
        polynomial_map: SparsePolynomialMap,
        /,
    ) -> PreparedPolynomialImageAnalysis:
        """Reanalyze new coefficients while preserving supports and source samples."""

        return refresh_polynomial_image_analysis(self, polynomial_map)


def plan_polynomial_image_analysis(
    polynomial_map: SparsePolynomialMap,
    target_support: TargetMonomialSupport,
    /,
    **kwargs,
) -> PolynomialImageAnalysisPlan:
    """Create a fixed-support image-analysis plan."""

    return PolynomialImageAnalysisPlan(polynomial_map, target_support, **kwargs)


def _sample_affine(plan: PolynomialImageAnalysisPlan) -> tuple[Array, Array]:
    dtype = jnp.real(
        jnp.zeros((), dtype=plan.polynomial_map.system.coefficients.dtype)
    ).dtype
    lower = plan.lower_bounds.astype(dtype)
    upper = plan.upper_bounds.astype(dtype)
    root = plan.root_key_data
    discovery_key = jr.fold_in(root, 0)
    heldout_key = jr.fold_in(root, 1)
    discovery = jr.uniform(
        discovery_key,
        (plan.discovery_sample_count, plan.polynomial_map.source_dimension),
        dtype=dtype,
        minval=lower,
        maxval=upper,
    )
    heldout = jr.uniform(
        heldout_key,
        (plan.heldout_sample_count, plan.polynomial_map.source_dimension),
        dtype=dtype,
        minval=lower,
        maxval=upper,
    )
    return discovery, heldout


def _resource_evidence(
    plan: PolynomialImageAnalysisPlan,
) -> PolynomialImageResourceEvidence:
    total = plan.discovery_sample_count + plan.heldout_sample_count
    monomials = plan.target_support.monomial_count
    design_entries = total * monomials
    itemsize = np.dtype(plan.polynomial_map.system.coefficients.dtype).itemsize
    source_dimension = plan.polynomial_map.source_dimension
    target_dimension = plan.polynomial_map.target_dimension
    discovery_count = plan.discovery_sample_count
    design_storage = (
        discovery_count * monomials
        + monomials * monomials
        + discovery_count * discovery_count
    )
    jacobian_storage = (
        source_dimension * target_dimension
        + source_dimension * source_dimension
        + target_dimension * target_dimension
    )
    estimated_bytes = 4 * itemsize * max(design_storage, jacobian_storage)
    limiting: str | None = None
    policy = plan.policy
    if total > policy.maximum_samples:
        limiting = "maximum_samples"
    elif monomials > policy.maximum_monomials:
        limiting = "maximum_monomials"
    elif design_entries > policy.maximum_design_entries:
        limiting = "maximum_design_entries"
    elif estimated_bytes > policy.maximum_svd_bytes:
        limiting = "maximum_svd_bytes"
    return PolynomialImageResourceEvidence(
        sample_count=total,
        monomial_count=monomials,
        design_entries=design_entries,
        estimated_svd_bytes=estimated_bytes,
        within_budget=limiting is None,
        limiting_resource=limiting,
    )


def _rank_bounds(
    singular_values: np.ndarray,
    relative_tolerance: float,
    ambiguity_factor: float,
) -> tuple[int, int, float, float]:
    real_dtype = np.asarray(singular_values).real.dtype
    tiny = np.finfo(real_dtype).tiny
    largest = float(np.max(singular_values, initial=0.0))
    central = relative_tolerance * max(largest, tiny)
    lower_cutoff = central / ambiguity_factor
    upper_cutoff = central * ambiguity_factor
    lower_rank = int(np.count_nonzero(singular_values > upper_cutoff))
    upper_rank = int(np.count_nonzero(singular_values > lower_cutoff))
    return lower_rank, upper_rank, lower_cutoff, upper_cutoff


def _svd(
    matrix: Array,
    policy: PolynomialImageAnalysisPolicy,
    problem_id: str,
):
    rows, columns = matrix.shape
    count = min(rows, columns)
    problem = SVDProblem(
        DenseLinearOperator(matrix, operator_id=f"{problem_id}:operator"),
        problem_id=problem_id,
    )
    result = svd(
        problem,
        policy=SVDSolvePolicy(
            count=count,
            materialization=MaterializationPolicy(
                max_entries=policy.maximum_design_entries,
                max_bytes=policy.maximum_svd_bytes,
            ),
            rank=RankPolicy(relative_cutoff=policy.rank_relative_tolerance),
            resources=SVDResourcePolicy(
                preparation_bytes=policy.maximum_svd_bytes,
                workspace_bytes=policy.maximum_svd_bytes,
                operator_matvecs=max(rows * columns, 1),
            ),
        ),
    )
    return result


def _empty_jacobian_evidence(
    sample_count: int,
    mode_count: int,
    dtype,
) -> JacobianRankEvidence:
    return JacobianRankEvidence(
        jnp.zeros((sample_count, mode_count), dtype=dtype),
        jnp.zeros((sample_count,), dtype=jnp.int32),
        jnp.zeros((sample_count,), dtype=jnp.int32),
        jnp.zeros((sample_count,), dtype=dtype),
        jnp.zeros((sample_count,), dtype=dtype),
        provider=_SVD_PROVIDER,
        svd_plan_ids=(),
    )


def _empty_relation_evidence(
    monomial_count: int,
    dtype,
) -> TargetRelationEvidence:
    return TargetRelationEvidence(
        jnp.zeros((monomial_count,), dtype=dtype),
        rank_lower_bound=0,
        rank_upper_bound=0,
        rank_lower_cutoff=jnp.asarray(0.0, dtype=dtype),
        rank_upper_cutoff=jnp.asarray(0.0, dtype=dtype),
        candidate_coefficients=jnp.zeros((monomial_count, monomial_count), dtype=dtype),
        candidate_active=jnp.zeros((monomial_count,), dtype=bool),
        discovery_residuals=jnp.zeros((monomial_count,), dtype=dtype),
        heldout_residuals=jnp.zeros((monomial_count,), dtype=dtype),
        heldout_tolerances=jnp.zeros((monomial_count,), dtype=dtype),
        heldout_accepted=jnp.zeros((monomial_count,), dtype=bool),
        provider=_SVD_PROVIDER,
        svd_plan_id=None,
    )


def _early_result(
    plan: PolynomialImageAnalysisPlan,
    polynomial_map: SparsePolynomialMap,
    source: Array,
    heldout: Array,
    resources: PolynomialImageResourceEvidence,
    status: PolynomialImageAnalysisStatus,
) -> PolynomialImageAnalysisResult:
    coefficient_dtype = polynomial_map.system.coefficients.dtype
    target_dtype = jnp.result_type(coefficient_dtype, source.dtype)
    target = jnp.zeros(
        (source.shape[0], polynomial_map.target_dimension), dtype=target_dtype
    )
    heldout_target = jnp.zeros(
        (heldout.shape[0], polynomial_map.target_dimension), dtype=target_dtype
    )
    mode_count = min(polynomial_map.source_dimension, polynomial_map.target_dimension)
    return PolynomialImageAnalysisResult(
        status,
        plan.source_kind,
        source,
        heldout,
        target,
        heldout_target,
        _empty_jacobian_evidence(
            source.shape[0] + heldout.shape[0], mode_count, jnp.real(target).dtype
        ),
        _empty_relation_evidence(
            plan.target_support.monomial_count, jnp.real(target).dtype
        ),
        resources,
        PolynomialImageClaimEvidence(),
        map_id=polynomial_map.map_id,
        plan_id=plan.plan_id,
    )


def _jacobian_rank_evidence(
    polynomial_map: SparsePolynomialMap,
    points: Array,
    policy: PolynomialImageAnalysisPolicy,
    plan_id: str,
) -> JacobianRankEvidence:
    jacobians = polynomial_map.jacobian(points)
    mode_count = min(polynomial_map.source_dimension, polynomial_map.target_dimension)
    values: list[np.ndarray] = []
    lower_ranks: list[int] = []
    upper_ranks: list[int] = []
    lower_cutoffs: list[float] = []
    upper_cutoffs: list[float] = []
    plan_ids: list[str] = []
    for index in range(points.shape[0]):
        decomposition = _svd(
            jacobians[index],
            policy,
            f"{plan_id}:jacobian:{index}",
        )
        singular = np.asarray(decomposition.singular_values)
        if singular.shape != (mode_count,):
            raise RuntimeError("Jacobian SVD returned an unexpected singular spectrum.")
        lower, upper, low_cutoff, high_cutoff = _rank_bounds(
            singular,
            policy.rank_relative_tolerance,
            policy.rank_ambiguity_factor,
        )
        values.append(singular)
        lower_ranks.append(lower)
        upper_ranks.append(upper)
        lower_cutoffs.append(low_cutoff)
        upper_cutoffs.append(high_cutoff)
        plan_ids.append(decomposition.provenance.plan_id)
    dtype = jnp.real(jacobians).dtype
    return JacobianRankEvidence(
        jnp.asarray(np.asarray(values), dtype=dtype),
        jnp.asarray(lower_ranks, dtype=jnp.int32),
        jnp.asarray(upper_ranks, dtype=jnp.int32),
        jnp.asarray(lower_cutoffs, dtype=dtype),
        jnp.asarray(upper_cutoffs, dtype=dtype),
        provider=_SVD_PROVIDER,
        svd_plan_ids=plan_ids,
    )


def _canonical_nullspace_basis(
    right_vectors: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Return an RREF-derived basis independent of SVD signs and basis rotations."""

    basis = np.asarray(right_vectors[:, rank:].T).copy()
    if basis.shape[0] == 0:
        return basis
    tolerance = 128.0 * np.finfo(basis.real.dtype).eps
    pivot_row = 0
    for column in range(basis.shape[1]):
        if pivot_row == basis.shape[0]:
            break
        remaining = np.abs(basis[pivot_row:, column])
        selected_offset = int(np.argmax(remaining))
        if remaining[selected_offset] <= tolerance:
            continue
        selected = pivot_row + selected_offset
        if selected != pivot_row:
            basis[[pivot_row, selected]] = basis[[selected, pivot_row]]
        basis[pivot_row] /= basis[pivot_row, column]
        for row in range(basis.shape[0]):
            if row != pivot_row:
                basis[row] -= basis[row, column] * basis[pivot_row]
        pivot_row += 1
    basis[np.abs(basis) <= tolerance] = 0
    for row in range(basis.shape[0]):
        magnitudes = np.abs(basis[row])
        pivot = int(np.argmax(magnitudes))
        if magnitudes[pivot] > 0.0:
            basis[row] /= basis[row, pivot]
        basis[row, np.abs(basis[row]) <= tolerance] = 0
    nonzero = tuple(
        np.flatnonzero(np.abs(basis[row]) > tolerance) for row in range(basis.shape[0])
    )
    if any(indices.size == 0 for indices in nonzero):
        raise RuntimeError("Nullspace normalization lost an independent basis row.")
    order = sorted(
        range(basis.shape[0]),
        key=lambda row: (
            int(nonzero[row][0]),
            tuple(np.round(basis[row].real, decimals=14)),
            tuple(np.round(basis[row].imag, decimals=14))
            if np.iscomplexobj(basis)
            else (),
        ),
    )
    return basis[np.asarray(order, dtype=np.int32)]


def _relation_evidence(
    plan: PolynomialImageAnalysisPlan,
    discovery_target: Array,
    heldout_target: Array,
) -> TargetRelationEvidence:
    support = plan.target_support
    policy = plan.policy
    design = support.evaluate(discovery_target)
    heldout_design = support.evaluate(heldout_target)
    decomposition = _svd(design, policy, f"{plan.plan_id}:target-monomials")
    singular = np.asarray(decomposition.singular_values)
    lower_rank, upper_rank, low_cutoff, high_cutoff = _rank_bounds(
        singular,
        policy.rank_relative_tolerance,
        policy.rank_ambiguity_factor,
    )
    right = np.asarray(decomposition.right_vectors)
    candidates = _canonical_nullspace_basis(right, upper_rank)
    capacity = support.monomial_count
    padded = np.zeros((capacity, capacity), dtype=right.dtype)
    padded[: candidates.shape[0]] = candidates
    active = np.zeros((capacity,), dtype=bool)
    active[: candidates.shape[0]] = True
    discovery_values = np.asarray(design) @ padded.T
    heldout_values = np.asarray(heldout_design) @ padded.T
    discovery_residuals = np.max(np.abs(discovery_values), axis=0, initial=0.0)
    heldout_residuals = np.max(np.abs(heldout_values), axis=0, initial=0.0)
    heldout_scale = np.max(
        np.abs(np.asarray(heldout_design)) @ np.abs(padded).T,
        axis=0,
        initial=0.0,
    )
    tolerances = (
        policy.heldout_absolute_tolerance
        + policy.heldout_relative_tolerance * heldout_scale
    )
    accepted = active & np.isfinite(heldout_residuals) & (heldout_residuals <= tolerances)
    return TargetRelationEvidence(
        jnp.asarray(singular),
        rank_lower_bound=lower_rank,
        rank_upper_bound=upper_rank,
        rank_lower_cutoff=jnp.asarray(low_cutoff, dtype=singular.dtype),
        rank_upper_cutoff=jnp.asarray(high_cutoff, dtype=singular.dtype),
        candidate_coefficients=jnp.asarray(padded),
        candidate_active=jnp.asarray(active),
        discovery_residuals=jnp.asarray(discovery_residuals),
        heldout_residuals=jnp.asarray(heldout_residuals),
        heldout_tolerances=jnp.asarray(tolerances),
        heldout_accepted=jnp.asarray(accepted),
        provider=_SVD_PROVIDER,
        svd_plan_id=decomposition.provenance.plan_id,
    )


def _analyze(
    plan: PolynomialImageAnalysisPlan,
    polynomial_map: SparsePolynomialMap,
    source: Array,
    heldout: Array,
) -> PolynomialImageAnalysisResult:
    resources = _resource_evidence(plan)
    if not resources.within_budget:
        return _early_result(
            plan,
            polynomial_map,
            source,
            heldout,
            resources,
            PolynomialImageAnalysisStatus.RESOURCE_LIMIT,
        )
    if source.shape[0] < plan.target_support.monomial_count:
        return _early_result(
            plan,
            polynomial_map,
            source,
            heldout,
            resources,
            PolynomialImageAnalysisStatus.INSUFFICIENT_DISCOVERY_SAMPLES,
        )
    if heldout.shape[0] == 0:
        return _early_result(
            plan,
            polynomial_map,
            source,
            heldout,
            resources,
            PolynomialImageAnalysisStatus.INSUFFICIENT_HELDOUT_SAMPLES,
        )
    if not np.all(np.isfinite(np.asarray(source))) or not np.all(
        np.isfinite(np.asarray(heldout))
    ):
        return _early_result(
            plan,
            polynomial_map,
            source,
            heldout,
            resources,
            PolynomialImageAnalysisStatus.NONFINITE_INPUT,
        )
    discovery_target = polynomial_map.evaluate(source)
    heldout_target = polynomial_map.evaluate(heldout)
    all_points = jnp.concatenate((source, heldout), axis=0)
    all_targets = jnp.concatenate((discovery_target, heldout_target), axis=0)
    if not np.all(np.isfinite(np.asarray(all_targets))):
        return _early_result(
            plan,
            polynomial_map,
            source,
            heldout,
            resources,
            PolynomialImageAnalysisStatus.NONFINITE_IMAGE,
        )
    jacobian_rank = _jacobian_rank_evidence(
        polynomial_map,
        all_points,
        plan.policy,
        plan.plan_id,
    )
    relations = _relation_evidence(plan, discovery_target, heldout_target)
    if not jacobian_rank.cutoff_resolved:
        status = PolynomialImageAnalysisStatus.JACOBIAN_RANK_AMBIGUOUS
    elif not jacobian_rank.sample_consistent:
        status = PolynomialImageAnalysisStatus.JACOBIAN_RANK_INCONSISTENT
    elif not relations.rank_resolved:
        status = PolynomialImageAnalysisStatus.INTERPOLATION_RANK_AMBIGUOUS
    elif relations.relation_count == 0:
        status = PolynomialImageAnalysisStatus.NO_RELATION
    elif not relations.validation_accepted:
        status = PolynomialImageAnalysisStatus.HELDOUT_REJECTED
    else:
        status = PolynomialImageAnalysisStatus.DISCOVERED
    if (
        relations.relation_count
        and relations.validation_accepted
        and relations.rank_resolved
    ):
        numerical = EvidenceDisposition.SUPPORTED
    elif relations.relation_count and not relations.validation_accepted:
        numerical = EvidenceDisposition.REJECTED
    else:
        numerical = EvidenceDisposition.NOT_ASSESSED
    claims = PolynomialImageClaimEvidence(numerical_discovery=numerical)
    return PolynomialImageAnalysisResult(
        status,
        plan.source_kind,
        source,
        heldout,
        discovery_target,
        heldout_target,
        jacobian_rank,
        relations,
        resources,
        claims,
        map_id=polynomial_map.map_id,
        plan_id=plan.plan_id,
    )


def prepare_polynomial_image_analysis(
    plan: PolynomialImageAnalysisPlan,
    /,
) -> PreparedPolynomialImageAnalysis:
    """Freeze source samples and execute one bounded numerical discovery pass."""

    if not isinstance(plan, PolynomialImageAnalysisPlan):
        raise TypeError("plan must be a PolynomialImageAnalysisPlan.")
    if plan.source_kind is SourceSampleKind.AFFINE_PRNG:
        source, heldout = _sample_affine(plan)
    else:
        source = plan.explicit_source_points
        heldout = plan.explicit_heldout_source_points
    result = _analyze(plan, plan.polynomial_map, source, heldout)
    return PreparedPolynomialImageAnalysis(
        plan,
        plan.polynomial_map,
        source,
        heldout,
        result,
    )


def refresh_polynomial_image_analysis(
    prepared: PreparedPolynomialImageAnalysis,
    polynomial_map: SparsePolynomialMap,
    /,
) -> PreparedPolynomialImageAnalysis:
    """Refresh numerical evidence while preserving fixed supports and samples."""

    if not isinstance(prepared, PreparedPolynomialImageAnalysis):
        raise TypeError("prepared must be PreparedPolynomialImageAnalysis.")
    if not isinstance(polynomial_map, SparsePolynomialMap):
        raise TypeError("polynomial_map must be SparsePolynomialMap.")
    if (
        polynomial_map.system.coefficients.dtype
        != prepared.polynomial_map.system.coefficients.dtype
    ):
        raise ValueError("Polynomial-image refresh must preserve coefficient dtype.")
    if polynomial_map.system.support.support_id != prepared.plan.map_support_id:
        raise ValueError("Polynomial-image refresh must preserve map support.")
    result = _analyze(
        prepared.plan,
        polynomial_map,
        prepared.source_points,
        prepared.heldout_source_points,
    )
    return PreparedPolynomialImageAnalysis(
        prepared.plan,
        polynomial_map,
        prepared.source_points,
        prepared.heldout_source_points,
        result,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


__all__ = [
    "PolynomialImageAnalysisPlan",
    "PolynomialImageAnalysisPolicy",
    "PreparedPolynomialImageAnalysis",
    "plan_polynomial_image_analysis",
    "prepare_polynomial_image_analysis",
    "refresh_polynomial_image_analysis",
]
