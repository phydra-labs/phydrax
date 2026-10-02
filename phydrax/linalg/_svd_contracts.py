# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import math
from enum import IntEnum
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree

from .._differentiation import DerivativeRoute
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._validation import nonnegative_integer, positive_integer
from ..typing import Bool, Dim, Float, Inexact, Int32, parse, PRNGKey, Scalar, Scope
from ._materialization import MaterializationPolicy
from ._operators import AbstractLinearOperator
from ._policies import FailurePolicy, RankPolicy
from ._randomized import ProbeRefresh
from ._singular_subspaces import SingularSubspaceResponse
from ._spaces import _coordinate_dtype


class SVDSourceDim(Dim):
    """Source-coordinate extent, not a scientific space identity."""


class SVDTargetDim(Dim):
    """Target-coordinate extent, independent of the source extent."""


class SVDModeDim(Dim):
    """Selected singular-mode capacity."""


class SVDSketchDim(Dim):
    """Fixed sampled range capacity, distinct from selected modes."""


SVDTarget: TypeAlias = Literal["largest", "smallest"]
SVDDifferentiationMode: TypeAlias = Literal[
    "none", "singular-values", "projector", "basis"
]
SVDCertificateKind: TypeAlias = Literal[
    "exact-spectrum", "deterministic-frobenius", "independent-gaussian"
]


@final
class SVDProblem(StrictModule):
    operator: AbstractLinearOperator
    problem_id: str = eqx.field(static=True)

    def __init__(
        self, operator: AbstractLinearOperator, /, *, problem_id: str | None = None
    ) -> None:
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape:
            raise ValueError("SVDProblem requires an unbatched operator.")
        source_dtype, target_dtype = (
            _coordinate_dtype(operator.source),
            _coordinate_dtype(operator.target),
        )
        if source_dtype != target_dtype or not np.issubdtype(source_dtype, np.inexact):
            raise TypeError(
                "SVD source and target must share one real or complex coordinate dtype."
            )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "svd-problem",
                    "operator": operator.operator_id,
                    "source": operator.source.space_id,
                    "target": operator.target.space_id,
                }
            )
            if problem_id is None
            else problem_id
        )
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("problem_id must be a non-empty string.")
        self.operator, self.problem_id = operator, identifier

    @property
    def maximum_rank(self) -> int:
        return min(self.operator.source.size, self.operator.target.size)


@final
class DenseSVD(StrictModule):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "dense-svd"


@final
class RandomizedSVD(StrictModule):
    oversampling: int = eqx.field(static=True)
    power_iterations: int = eqx.field(static=True)
    probe_refresh: ProbeRefresh = eqx.field(static=True)

    def __init__(
        self,
        *,
        oversampling: int = 8,
        power_iterations: int = 2,
        probe_refresh: ProbeRefresh = "reuse",
    ) -> None:
        oversampling_ = nonnegative_integer(oversampling, "oversampling")
        iterations = nonnegative_integer(power_iterations, "power_iterations")
        refresh = parse(probe_refresh, ProbeRefresh, "probe_refresh")
        self.oversampling, self.power_iterations, self.probe_refresh = (
            oversampling_,
            iterations,
            refresh,
        )

    @property
    def name(self) -> str:
        return "randomized-svd"


@final
class SVDTolerancePolicy(StrictModule):
    residual: float = eqx.field(static=True)
    orthogonality: float = eqx.field(static=True)

    def __init__(self, *, residual: float = 1e-7, orthogonality: float = 1e-7) -> None:
        values = float(residual), float(orthogonality)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("SVD tolerances must be finite and non-negative.")
        self.residual, self.orthogonality = values


@final
class SVDApproximationPolicy(StrictModule):
    audit_probes: int = eqx.field(static=True)
    failure_probability: float = eqx.field(static=True)
    numerical_allowance: float = eqx.field(static=True)
    require_leading: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        audit_probes: int = 8,
        failure_probability: float = 1e-6,
        numerical_allowance: float = 64.0,
        require_leading: bool = False,
    ) -> None:
        probes = positive_integer(audit_probes, "audit_probes")
        delta, allowance = float(failure_probability), float(numerical_allowance)
        if not math.isfinite(delta) or not 0 < delta < 1:
            raise ValueError(
                "failure_probability must lie strictly between zero and one."
            )
        if not math.isfinite(allowance) or allowance < 0:
            raise ValueError("numerical_allowance must be finite and non-negative.")
        if not isinstance(require_leading, bool):
            raise TypeError("require_leading must be a boolean.")
        self.audit_probes, self.failure_probability = probes, delta
        self.numerical_allowance, self.require_leading = allowance, require_leading


@final
class SVDResourcePolicy(StrictModule):
    preparation_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    operator_matvecs: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        preparation_bytes: int = 512 * 1024 * 1024,
        workspace_bytes: int = 512 * 1024 * 1024,
        operator_matvecs: int = 1_000_000,
    ) -> None:
        preparation = nonnegative_integer(preparation_bytes, "preparation_bytes")
        workspace = nonnegative_integer(workspace_bytes, "workspace_bytes")
        actions = nonnegative_integer(operator_matvecs, "operator_matvecs")
        self.preparation_bytes, self.workspace_bytes, self.operator_matvecs = (
            preparation,
            workspace,
            actions,
        )


@final
class SVDSolvePolicy(StrictModule):
    method: DenseSVD | RandomizedSVD
    count: int = eqx.field(static=True)
    which: SVDTarget = eqx.field(static=True)
    tolerance: SVDTolerancePolicy
    approximation: SVDApproximationPolicy
    rank: RankPolicy
    materialization: MaterializationPolicy
    resources: SVDResourcePolicy
    differentiation: SVDDifferentiationMode = eqx.field(static=True)
    failure: FailurePolicy

    def __init__(
        self,
        method: DenseSVD | RandomizedSVD | None = None,
        /,
        *,
        count: int = 1,
        which: SVDTarget = "largest",
        tolerance: SVDTolerancePolicy | None = None,
        approximation: SVDApproximationPolicy | None = None,
        rank: RankPolicy | None = None,
        materialization: MaterializationPolicy | None = None,
        resources: SVDResourcePolicy | None = None,
        differentiation: SVDDifferentiationMode = "none",
        failure: FailurePolicy | None = None,
    ) -> None:
        method_ = DenseSVD() if method is None else method
        if not isinstance(method_, (DenseSVD, RandomizedSVD)):
            raise TypeError("method must be DenseSVD or RandomizedSVD.")
        count_ = positive_integer(count, "count")
        which_ = parse(which, SVDTarget, "which")
        mode = parse(differentiation, SVDDifferentiationMode, "differentiation")
        values = (
            SVDTolerancePolicy() if tolerance is None else tolerance,
            SVDApproximationPolicy() if approximation is None else approximation,
            RankPolicy() if rank is None else rank,
            MaterializationPolicy() if materialization is None else materialization,
            SVDResourcePolicy() if resources is None else resources,
            FailurePolicy() if failure is None else failure,
        )
        kinds = (
            SVDTolerancePolicy,
            SVDApproximationPolicy,
            RankPolicy,
            MaterializationPolicy,
            SVDResourcePolicy,
            FailurePolicy,
        )
        for value, kind in zip(values, kinds, strict=True):
            if not isinstance(value, kind):
                raise TypeError(f"Expected {kind.__name__} policy.")
        self.method, self.count, self.which, self.differentiation = (
            method_,
            count_,
            which_,
            mode,
        )
        (
            self.tolerance,
            self.approximation,
            self.rank,
            self.materialization,
            self.resources,
            self.failure,
        ) = values


@final
class SVDCostEstimate(StrictModule):
    storage_bytes: int = eqx.field(static=True)
    original_operator_bytes: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    apply_workspace_bytes: int = eqx.field(static=True)
    operator_matvec_count: int = eqx.field(static=True)
    adjoint_matvec_count: int = eqx.field(static=True)
    preparation_forward_block_calls: int = eqx.field(static=True)
    preparation_adjoint_block_calls: int = eqx.field(static=True)
    solve_forward_block_calls: int = eqx.field(static=True)
    solve_adjoint_block_calls: int = eqx.field(static=True)
    deterministic_scan_entries: int = eqx.field(static=True)
    provider_workspace_exact: bool = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)


@final
class SVDSolvePlan(StrictModule):
    problem_id: str = eqx.field(static=True)
    policy: SVDSolvePolicy
    cost: SVDCostEstimate
    sketch_size: int = eqx.field(static=True)
    effective_oversampling: int = eqx.field(static=True)
    certificate_kind: SVDCertificateKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: SVDProblem,
        policy: SVDSolvePolicy,
        cost: SVDCostEstimate,
        sketch_size: int,
        certificate_kind: SVDCertificateKind,
        /,
    ) -> None:
        if (
            not isinstance(problem, SVDProblem)
            or not isinstance(policy, SVDSolvePolicy)
            or not isinstance(cost, SVDCostEstimate)
        ):
            raise TypeError(
                "SVD plan requires native problem, policy, and cost contracts."
            )
        width = positive_integer(sketch_size, "sketch_size")
        if not policy.count <= width <= problem.maximum_rank:
            raise ValueError(
                "SVD sketch_size must cover count and fit both operator dimensions."
            )
        certificate_kind = parse(certificate_kind, SVDCertificateKind, "certificate_kind")
        if not cost.accepted:
            raise ValueError(f"SVD is infeasible: {cost.reason}.")
        method = policy.method
        identity = {
            "kind": "svd-solve-plan",
            "problem": problem.problem_id,
            "operator": problem.operator.operator_id,
            "source": problem.operator.source.space_id,
            "target": problem.operator.target.space_id,
            "method": method.name,
            "oversampling": method.oversampling
            if isinstance(method, RandomizedSVD)
            else 0,
            "power_iterations": method.power_iterations
            if isinstance(method, RandomizedSVD)
            else 0,
            "probe_refresh": method.probe_refresh
            if isinstance(method, RandomizedSVD)
            else "reuse",
            "count": policy.count,
            "which": policy.which,
            "differentiation": policy.differentiation,
            "failure": policy.failure.mode,
            "rank_relative": policy.rank.relative_cutoff,
            "rank_absolute": policy.rank.absolute_cutoff,
            "full_rank": policy.rank.require_full_rank,
            "residual": policy.tolerance.residual,
            "orthogonality": policy.tolerance.orthogonality,
            "audit_probes": policy.approximation.audit_probes,
            "failure_probability": policy.approximation.failure_probability,
            "numerical_allowance": policy.approximation.numerical_allowance,
            "require_leading": policy.approximation.require_leading,
            "materialization_entries": policy.materialization.max_entries,
            "materialization_bytes": policy.materialization.max_bytes,
            "preparation_budget": policy.resources.preparation_bytes,
            "workspace_budget": policy.resources.workspace_bytes,
            "action_budget": policy.resources.operator_matvecs,
            "certificate": certificate_kind,
            "sketch_size": sketch_size,
        }
        identity["sketch_size"] = width
        self.problem_id, self.policy, self.cost = problem.problem_id, policy, cost
        self.sketch_size = width
        self.effective_oversampling = width - policy.count
        self.certificate_kind = certificate_kind
        self.plan_id = canonical_fingerprint(identity)


@final
class DenseSVDState(StrictModule):
    __strict_contract__ = True
    reduced_operator: Inexact[SVDTargetDim, SVDSourceDim]
    source_factor: Inexact[SVDSourceDim] | Inexact[SVDSourceDim, SVDSourceDim]
    target_factor: Inexact[SVDTargetDim] | Inexact[SVDTargetDim, SVDTargetDim]
    preparation_status: Int32[Scalar]
    diagonal: bool = eqx.field(static=True)

    def __init__(
        self,
        reduced_operator: Array,
        source_factor: Array,
        target_factor: Array,
        preparation_status: Array,
        diagonal: bool,
        /,
    ) -> None:
        if not isinstance(diagonal, bool):
            raise TypeError("diagonal must be a boolean.")
        scope = Scope()
        reduced_operator = parse(
            reduced_operator,
            Inexact[SVDTargetDim, SVDSourceDim],
            "reduced_operator",
            scope=scope,
        )
        source_form = (
            Inexact[SVDSourceDim] if diagonal else Inexact[SVDSourceDim, SVDSourceDim]
        )
        target_form = (
            Inexact[SVDTargetDim] if diagonal else Inexact[SVDTargetDim, SVDTargetDim]
        )
        source_factor = parse(source_factor, source_form, "source_factor", scope=scope)
        target_factor = parse(target_factor, target_form, "target_factor", scope=scope)
        preparation_status = parse(
            preparation_status, Int32[Scalar], "preparation_status"
        )
        if (
            source_factor.dtype != reduced_operator.dtype
            or target_factor.dtype != reduced_operator.dtype
        ):
            raise TypeError("Dense SVD state arrays must share one coordinate dtype.")
        self.reduced_operator, self.source_factor, self.target_factor = (
            reduced_operator,
            source_factor,
            target_factor,
        )
        self.preparation_status, self.diagonal = preparation_status, diagonal


@final
class RandomizedSVDState(StrictModule):
    __strict_contract__ = True
    range_basis: Inexact[SVDTargetDim, SVDSketchDim]
    compressed_operator: Inexact[SVDSketchDim, SVDSourceDim]
    source_factor: Inexact[SVDSourceDim]
    target_factor: Inexact[SVDTargetDim]
    preparation_status: Int32[Scalar]
    qr_minimum_margin: Float[Scalar]
    qr_full_rank: Bool[Scalar]
    range_bound: Float[Scalar]
    range_allowance: Float[Scalar]
    audit_failure_probability: Float[Scalar]
    audit_maximum_norm: Float[Scalar]
    total_energy: Float[Scalar] | None
    root_key: PRNGKey
    sketch_version: Int32[Scalar]
    audit_version: Int32[Scalar]

    def __init__(
        self,
        range_basis: Array,
        compressed_operator: Array,
        source_factor: Array,
        target_factor: Array,
        preparation_status: Array,
        qr_minimum_margin: Array,
        qr_full_rank: Array,
        range_bound: Array,
        range_allowance: Array,
        audit_failure_probability: Array,
        audit_maximum_norm: Array,
        total_energy: Array | None,
        root_key: PRNGKey,
        sketch_version: Array,
        audit_version: Array,
        /,
    ) -> None:
        scope = Scope()
        basis = parse(
            range_basis, Inexact[SVDTargetDim, SVDSketchDim], "range_basis", scope=scope
        )
        core = parse(
            compressed_operator,
            Inexact[SVDSketchDim, SVDSourceDim],
            "compressed_operator",
            scope=scope,
        )
        source = parse(source_factor, Inexact[SVDSourceDim], "source_factor", scope=scope)
        target = parse(target_factor, Inexact[SVDTargetDim], "target_factor", scope=scope)
        if basis.shape[1] < 1 or basis.shape[1] > min(basis.shape[0], core.shape[1]):
            raise ValueError(
                "Randomized SVD width must be positive and fit both coordinate dimensions."
            )
        if any(value.dtype != basis.dtype for value in (core, source, target)):
            raise TypeError("Randomized SVD coordinate arrays must share one dtype.")
        status = parse(preparation_status, Int32[Scalar], "preparation_status")
        margin = parse(qr_minimum_margin, Float[Scalar], "qr_minimum_margin")
        rank_valid = parse(qr_full_rank, Bool[Scalar], "qr_full_rank")
        bound = parse(range_bound, Float[Scalar], "range_bound")
        allowance = parse(range_allowance, Float[Scalar], "range_allowance")
        probability = parse(
            audit_failure_probability, Float[Scalar], "audit_failure_probability"
        )
        maximum = parse(audit_maximum_norm, Float[Scalar], "audit_maximum_norm")
        energy = parse(total_energy, Float[Scalar] | None, "total_energy")
        if any(
            value.dtype != basis.real.dtype
            for value in (margin, bound, allowance, probability, maximum)
        ):
            raise TypeError(
                "Randomized SVD norm evidence must use the coordinate real dtype."
            )
        if energy is not None and energy.dtype != basis.real.dtype:
            raise TypeError(
                "Randomized SVD total energy must use the coordinate real dtype."
            )
        root = parse(root_key, PRNGKey, "root_key")
        sketch = parse(sketch_version, Int32[Scalar], "sketch_version")
        audit = parse(audit_version, Int32[Scalar], "audit_version")
        (
            self.range_basis,
            self.compressed_operator,
            self.source_factor,
            self.target_factor,
        ) = basis, core, source, target
        self.preparation_status, self.qr_minimum_margin, self.qr_full_rank = (
            status,
            margin,
            rank_valid,
        )
        (
            self.range_bound,
            self.range_allowance,
            self.audit_failure_probability,
            self.audit_maximum_norm,
        ) = bound, allowance, probability, maximum
        self.total_energy, self.root_key, self.sketch_version, self.audit_version = (
            energy,
            root,
            sketch,
            audit,
        )


@final
class PreparedSVDSolve(StrictModule):
    problem: SVDProblem
    plan: SVDSolvePlan
    state: DenseSVDState | RandomizedSVDState
    numeric_version: Array

    def __init__(
        self,
        problem: SVDProblem,
        plan: SVDSolvePlan,
        state: DenseSVDState | RandomizedSVDState,
        /,
        numeric_version: int | Array = 0,
    ) -> None:
        if not isinstance(problem, SVDProblem) or not isinstance(plan, SVDSolvePlan):
            raise TypeError("Expected SVDProblem and SVDSolvePlan.")
        if not isinstance(state, (DenseSVDState, RandomizedSVDState)):
            raise TypeError("Expected a method-specific SVD state.")
        if problem.problem_id != plan.problem_id:
            raise ValueError("Prepared SVD problem and plan IDs must match.")
        dense = isinstance(plan.policy.method, DenseSVD)
        if dense != isinstance(state, DenseSVDState):
            raise TypeError("Prepared state must match its planned method.")
        version = jnp.asarray(numeric_version)
        if version.shape != () or not jnp.issubdtype(version.dtype, jnp.integer):
            raise ValueError("numeric_version must be an integer scalar.")
        version = eqx.error_if(
            version,
            (version < 0) | (version > jnp.iinfo(jnp.int32).max),
            "numeric_version is outside the admitted range.",
        ).astype(jnp.int32)
        self.problem, self.plan, self.state, self.numeric_version = (
            problem,
            plan,
            state,
            version,
        )


class SVDSolveStatus(IntEnum):
    SUCCESS = 0
    RESIDUAL_TOLERANCE_NOT_MET = 1
    RANK_DEFICIENT = 2
    NONFINITE_OUTPUT = 3
    DIFFERENTIATION_REJECTED = 4
    LEADING_UNCERTIFIED = 5
    GLOBAL_RANK_UNCERTIFIED = 6
    PREPARATION_FAILED = 7


@final
class SVDRankEvidence(StrictModule):
    lower_bound: Array
    upper_bound: Array
    threshold_lower: Array
    threshold_upper: Array
    available: Array
    full_spectrum: bool = eqx.field(static=True)
    deterministic_exact: Array
    certificate_kind: SVDCertificateKind = eqx.field(static=True)
    failure_probability: Array


@final
class SVDRangeEvidence(StrictModule):
    spectral_upper_bound: Array
    numerical_allowance: Array
    spectrum_numerical_allowance: Array
    failure_probability: Array
    audit_maximum_norm: Array
    available: Array
    certificate_kind: SVDCertificateKind = eqx.field(static=True)
    independent_audit_assumption: bool = eqx.field(static=True)
    provider_action_error_known: bool = eqx.field(static=True)
    total_energy: Array | None
    singular_value_lower_bounds: Array
    singular_value_upper_bounds: Array
    omitted_spectrum_upper_bound: Array
    factor_residual_upper_bound: Array


@final
class SVDLeadingEvidence(StrictModule):
    certified: Array
    gap_lower_bound: Array
    tail_upper_bound: Array
    full_spectrum: bool = eqx.field(static=True)


@final
class SVDSolveDiagnostics(StrictModule):
    left_residual_norms: Array
    right_residual_norms: Array
    relative_residuals: Array
    left_orthogonality_error: Array
    right_orthogonality_error: Array
    isolation_gaps: Array
    cutoff_gap: Array
    pivot_magnitudes: Array
    pivot_gaps: Array
    qr_minimum_margin: Array
    projection_energies: Array
    converged: Array
    rank_evidence: SVDRankEvidence
    operator_matvec_count: Array
    adjoint_matvec_count: Array


@final
class SVDSolveProvenance(StrictModule):
    __strict_contract__ = True
    method: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    which: SVDTarget = eqx.field(static=True)
    differentiation: SVDDifferentiationMode = eqx.field(static=True)
    derivative_route: DerivativeRoute = eqx.field(static=True)
    numeric_version: Array
    root_key: PRNGKey | None
    sketch_version: Array
    audit_version: Array


@final
class SVDSolveResult(StrictModule):
    __strict_contract__ = True
    singular_values: Float[SVDModeDim]
    left_vectors: PyTree[Array]
    right_vectors: PyTree[Array]
    left_coordinates: Inexact[SVDTargetDim, SVDModeDim]
    right_coordinates: Inexact[SVDSourceDim, SVDModeDim]
    left_response: SingularSubspaceResponse
    right_response: SingularSubspaceResponse
    status: Array
    primal_status: Array
    derivative_status: Array
    derivative_valid: Array
    converged: Array
    rank_evidence: SVDRankEvidence
    range_evidence: SVDRangeEvidence
    leading_evidence: SVDLeadingEvidence
    diagnostics: SVDSolveDiagnostics
    provenance: SVDSolveProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(SVDSolveStatus.SUCCESS)


def require_exact_svd_rank(evidence: SVDRankEvidence | SVDSolveResult, /) -> Array:
    """Consume available deterministic global rank evidence, never core rank."""
    if isinstance(evidence, SVDSolveResult):
        valid = jnp.asarray(True)
        rank = evidence.rank_evidence
    elif isinstance(evidence, SVDRankEvidence):
        rank, valid = evidence, jnp.asarray(True)
    else:
        raise TypeError("Expected SVDRankEvidence or SVDSolveResult.")
    valid = valid & rank.available & (rank.lower_bound == rank.upper_bound)
    valid = valid & rank.deterministic_exact & rank.full_spectrum
    return eqx.error_if(
        rank.lower_bound,
        ~valid,
        "Deterministic exact global SVD rank evidence is required.",
    )
