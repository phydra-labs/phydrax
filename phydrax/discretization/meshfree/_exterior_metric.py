#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""All-equation graph metric moment problems, without metric clipping.

The exact signed metric minimizes ``0.5 w^T Phi^-1 w`` subject to every
original moment equation ``A w = b``. With ``S = sqrt(Phi)``, ``B = D^-1 A S``
and ``w = S z``, this is the exact minimum Euclidean norm solution of
``B z = D^-1 b``. ``D`` is the declared row scaling. The default
``"multilevel-craig"`` solver runs preconditioned Craig on ``B B^T`` and returns
``z = B^T y``. The preconditioner is a vector-field jet hierarchy
(:mod:`._metric_multilevel`); it changes only the iteration count, never the
minimum-norm solution. ``"lsmr"`` runs LSMR from a zero start on row-equilibrated
``B``. Left scaling changes neither the constraint set nor the minimum-norm
solution. The relaxed signed objective
``0.5 |z|^2 + C/2 |D^-1 (A S z - b)|^2`` keeps exactly the declared ``D`` as the
stacked least-squares problem ``[sqrt(C) B; I] z = [sqrt(C) D^-1 b; 0]``.
No Schur factor, explicit inverse, or row selection is formed for either problem.
"""

from __future__ import annotations

from enum import IntEnum
from itertools import combinations_with_replacement
from math import isfinite
from typing import assert_never, cast, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    certify_rectangular_rank,
    Craig,
    DiagonalLinearOperator,
    DifferentiationPolicy,
    FailurePolicy,
    GalerkinHierarchyBuilder,
    GaussSeidelPreconditionerBuilder,
    GeneralizedLSMR,
    IdentityLinearOperator,
    JacobianLinearOperator,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSolveResult,
    LSMR,
    MinimumNormProblem,
    OperatorProperties,
    PreconditionerProperties,
    PreconditioningPolicy,
    prepare,
    prepare_sparse_row_rank,
    PreparedLinearSolve,
    PropertyEvidence,
    RankPolicy,
    RectangularRankCertificate,
    refresh,
    solve,
    SolveResourcePolicy,
    SparseAssemblyPolicy,
    SparseRowRankEvidence,
    SparseRowRankPolicy,
    StabilityLowerBound,
    StackedLinearOperator,
    TolerancePolicy,
)
from ...linalg.svd import DenseSVD, SVDSolvePolicy
from ...optim import (
    conic_primal_jvp,
    ConicActiveSetEvidence,
    ConicProgram,
    ConicProgramData,
    ConicSensitivityResult,
    ConicSensitivityStatus,
    ConvexProgramExecution,
    ConvexProgramStatus,
    ConvexSolvePolicy,
    ConvexTermination,
    NativeHomogeneousConic,
    NonnegativeCone,
    prepare_conic_sensitivity,
    prepare_convex_program,
    PreparedConvexProgram,
    PreparedMatrixFreeConicSensitivity,
    ProductCone,
    solve_prepared_convex_program,
    ZeroCone,
)
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ...sparse._linear import _SparseStoragePlan
from ...typing import Bool, Dim, Float64, Int32, parse, Scalar
from ._metric_multilevel import metric_jet_transfers


class _MetricEdgeDim(Dim):
    """Compact undirected metric edges."""


class _MomentRowDim(Dim):
    """All local moment equations, including dependent equations."""


class _EquationNodeDim(Dim):
    """Nodes at which polynomial moment equations are requested."""


class _ConicRouteDim(Dim):
    """Moment coefficient routes retained in the conic equality block."""


class _ConicMomentDim(Dim):
    """Original moment rows posed as conic equalities."""


class _SlackRowDim(Dim):
    """Moment rows penalized by an explicit slack variable."""


MetricSign: TypeAlias = Literal["signed", "nonnegative"]
MetricAcceptance: TypeAlias = Literal["exact", "relaxed"]
BoundaryClosure: TypeAlias = Literal["second-moment", "area-vector"]
MetricSolver: TypeAlias = Literal["multilevel-craig", "lsmr"]


class MeshfreeMetricStatus(IntEnum):
    ACCEPTED = 0
    INCOMPATIBLE_MOMENTS = 1
    PROVIDER_FAILURE = 2
    AMPLIFICATION_LIMIT = 3
    NONFINITE = 4
    RELAXED = 5


class MeshfreeMomentRowStatus(IntEnum):
    SATISFIED = 0
    INCOMPATIBLE = 1
    RELAXED = 2


def _moment_exponents(dimension: int, degree: int, /) -> np.ndarray:
    """Deterministic degree-graded multi-indices ``1 <= |alpha| <= degree``.

    Within one degree the order is ``combinations_with_replacement`` of the
    intrinsic axes, so degrees one and two are ``x_k`` and ``x_k x_l`` (k<=l).
    """
    if dimension not in (1, 2, 3) or degree < 1:
        raise ValueError("Moment multi-indices need dimension 1, 2 or 3 and degree >= 1.")
    rows = [
        np.bincount(np.asarray(axes, dtype=np.int64), minlength=dimension)
        for order in range(1, degree + 1)
        for axes in combinations_with_replacement(range(dimension), order)
    ]
    return np.stack(rows).astype(np.int32)


def _laplacian_moment_rhs(exponents: np.ndarray, /) -> np.ndarray:
    """Unit-volume Laplacian functional of ``(x - x_i)^alpha`` at ``x_i``.

    ``Delta x^alpha = sum_k alpha_k (alpha_k - 1) x^(alpha - 2 e_k)`` vanishes at
    the origin unless ``alpha - 2 e_k = 0``, where it equals two.
    """
    values = np.asarray(exponents)
    total = values.sum(axis=1)
    return np.where((total == 2) & (values.max(axis=1) == 2), 2.0, 0.0).astype(np.float64)


def _boundary_moment_indices(dimension: int, closure: BoundaryClosure, /) -> np.ndarray:
    """Degree-graded multi-index positions posed at boundary-closure nodes."""
    second = _moment_exponents(dimension, 2).shape[0]
    match closure:
        case "area-vector":
            return np.arange(second, dtype=np.int32)
        case "second-moment":
            return np.arange(dimension, second, dtype=np.int32)
        case unknown:
            assert_never(unknown)


@final
class MeshfreeMetricPolicy(StrictModule):
    """Exact moment constraints, or an explicitly penalized slack problem.

    ``moment_degree`` declares every polynomial moment ``1 <= |alpha| <= degree``
    that the edge metric must reproduce for the Laplacian functional. Higher
    degrees are never truncated: an infeasible system is refused or, only when
    requested, relaxed. ``rank`` bounds the dense rank-revealing certificate that
    admits signed exact derivatives; beyond its limits rank evidence is reported
    unavailable. ``row_rank_profile`` additionally records the sparse pivot
    profile as a diagnostic; it is never a rank certificate. Nonnegative metrics
    use a native sparse conic program; its exact equality block keeps the pivot
    profile's independent rows (an interior-point presolve) while every original
    row is still audited. The strict-active derivative is bound eagerly by
    :meth:`PreparedMeshfreeMetric.bind_active_set`.

    ``boundary_closure`` selects the rows of boundary-closure nodes (nodes with
    declared boundary area vectors ``s``). ``"second-moment"`` (default) poses
    ``sum_e w_e d_e d_e^T = 2 V I``; the boundary normal measure is then the
    deficit ``S_i = -sum_e w_e d_e``, which satisfies ``sum_i S_i = 0`` and
    ``sum_i x_i S_i^T = sum_i V_i I`` exactly and makes edge transport and
    divergence exact for affine fields; ``|S - s|`` is reported by the exterior
    as evidence. ``"area-vector"`` additionally poses ``sum_e w_e d_e = -s``.
    Both degrees together are overdetermined: compatibility requires the
    discrete Gauss identities ``sum_i x_il x_im s_ik = sum_i V_i (x_il d_km +
    x_im d_kl)`` and more, which lattices satisfy and irregular clouds
    generally do not; such moments are refused with a witness, never relaxed.

    ``solver`` selects the exact signed solve; the objective is the same for both.
    ``"multilevel-craig"`` (the exact signed default) runs Craig on ``B B^T``
    preconditioned by a Galerkin V-cycle. The V-cycle uses symmetric
    Gauss-Seidel smoothing over nested spline vector-field jets
    (:func:`._metric_multilevel.metric_jet_transfers`). Iterations grow slowly
    with refinement. ``B B^T`` has condition number ``O(h^-6)`` (see that
    module), so plain LSMR on equilibrated ``B`` needs ``O(h^-2.5)`` iterations.
    The hierarchy needs moment displacements that are coordinate displacements
    (default or minimum-image charts); surface charts declare ``"lsmr"``. Relaxed
    signed metrics always use ``"lsmr"`` on their stacked least-squares problem.
    Nonnegative metrics take no solver. ``multilevel_resources`` and
    ``multilevel_assembly`` bound the exactly assembled ``B B^T`` and the coarse
    Galerkin products. A hierarchy beyond them is refused at preparation.
    """

    __strict_contract__ = True
    sign: MetricSign = eqx.field(static=True)
    acceptance: MetricAcceptance = eqx.field(static=True)
    boundary_closure: BoundaryClosure = eqx.field(static=True)
    moment_degree: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    amplification_limit: float = eqx.field(static=True)
    slack_penalty: float = eqx.field(static=True)
    maximum_symbolic_entries: int = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    row_rank_profile: bool = eqx.field(static=True)
    solver: MetricSolver | None = eqx.field(static=True)
    multilevel_resources: SolveResourcePolicy
    multilevel_assembly: SparseAssemblyPolicy
    rank: SVDSolvePolicy
    conic: ConvexSolvePolicy

    def __init__(
        self,
        sign: MetricSign = "signed",
        *,
        acceptance: MetricAcceptance = "exact",
        boundary_closure: BoundaryClosure = "second-moment",
        moment_degree: int = 2,
        tolerance: float = 1e-9,
        amplification_limit: float = 1e8,
        slack_penalty: float = 1e4,
        maximum_symbolic_entries: int = 2_000_000,
        maximum_steps: int = 4096,
        row_rank_profile: bool = False,
        rank: SVDSolvePolicy | None = None,
        conic: ConvexSolvePolicy | None = None,
        solver: MetricSolver | None = None,
        multilevel_resources: SolveResourcePolicy | None = None,
        multilevel_assembly: SparseAssemblyPolicy | None = None,
    ) -> None:
        sign_ = parse(sign, MetricSign, "sign")
        acceptance_ = parse(acceptance, MetricAcceptance, "acceptance")
        boundary_closure_ = parse(boundary_closure, BoundaryClosure, "boundary_closure")
        if isinstance(moment_degree, bool) or not isinstance(moment_degree, int):
            raise TypeError("moment_degree must be an integer.")
        if moment_degree < 2:
            raise ValueError(
                "The Laplacian metric needs moment_degree >= 2; second moments define it."
            )
        numbers = (tolerance, amplification_limit, slack_penalty)
        if any(not isfinite(float(v)) or v <= 0 for v in numbers):
            raise ValueError(
                "Metric tolerance, amplification and slack penalty must be positive."
            )
        if maximum_symbolic_entries < 1 or maximum_steps < 1:
            raise ValueError(
                "Metric symbolic capacity and iteration bound must be positive."
            )
        if not isinstance(row_rank_profile, bool):
            raise TypeError("row_rank_profile must be a bool.")
        # Rank certificates use the QR (gesvd) driver: divide-and-conquer can
        # fail to converge on the exactly clustered spectra of lattice clouds.
        rank_ = (
            SVDSolvePolicy(
                DenseSVD(algorithm="qr"), rank=RankPolicy(relative_cutoff=1e-12)
            )
            if rank is None
            else rank
        )
        if not isinstance(rank_, SVDSolvePolicy):
            raise TypeError("rank must be an SVDSolvePolicy.")
        conic_ = (
            ConvexSolvePolicy(
                NativeHomogeneousConic(
                    primal_step=min(0.05, 0.5 / slack_penalty)
                    if acceptance_ == "relaxed"
                    else 0.05,
                    dual_step=0.05,
                ),
                termination=ConvexTermination(
                    absolute=tolerance, maximum_steps=maximum_steps
                ),
                failure=FailurePolicy("status"),
            )
            if conic is None
            else conic
        )
        if not conic_.method.capabilities.sparse or conic_.regularization != 0:
            raise ValueError(
                "Metric conic provider must support sparse data without objective jitter."
            )
        match sign_, acceptance_, solver:
            case "signed", "exact", None:
                solver_: MetricSolver | None = "multilevel-craig"
            case "signed", "relaxed", None | "lsmr":
                solver_ = "lsmr"
            case "signed", "relaxed", "multilevel-craig":
                raise ValueError(
                    "Relaxed signed metrics solve a stacked least-squares problem with 'lsmr'."
                )
            case "signed", "exact", declared:
                solver_ = parse(declared, MetricSolver, "solver")
            case "nonnegative", _, None:
                solver_ = None
            case "nonnegative", _, _:
                raise ValueError(
                    "Nonnegative metrics use the conic policy, not a solver."
                )
            case unknown:
                raise ValueError(f"Unknown metric policy {unknown!r}.")
        resources = (
            SolveResourcePolicy(workspace_bytes=2**32, preconditioner_bytes=2**30)
            if multilevel_resources is None
            else multilevel_resources
        )
        if not isinstance(resources, SolveResourcePolicy):
            raise TypeError("multilevel_resources must be a SolveResourcePolicy.")
        assembly = (
            SparseAssemblyPolicy(
                max_nnz=50_000_000,
                max_bytes=2**31,
                max_contributions=400_000_000,
                max_workspace_bytes=2**32,
            )
            if multilevel_assembly is None
            else multilevel_assembly
        )
        if not isinstance(assembly, SparseAssemblyPolicy):
            raise TypeError("multilevel_assembly must be a SparseAssemblyPolicy.")
        self.sign = sign_
        self.acceptance = acceptance_
        self.boundary_closure = boundary_closure_
        self.moment_degree = moment_degree
        self.tolerance = float(tolerance)
        self.amplification_limit = float(amplification_limit)
        self.slack_penalty = float(slack_penalty)
        self.maximum_symbolic_entries = int(maximum_symbolic_entries)
        self.maximum_steps = int(maximum_steps)
        self.row_rank_profile = row_rank_profile
        self.rank = rank_
        self.conic = conic_
        self.solver = solver_
        self.multilevel_resources = resources
        self.multilevel_assembly = assembly


@final
class MeshfreeMetricResult(StrictModule):
    """Candidate metric with every original moment, sign and solver audit.

    ``rank`` comes from ``rank_certificate`` (``-1`` when the bounded dense
    certificate is unavailable); a sparse ``row_rank_profile`` is diagnostic only.
    Per-node fields follow the metric's moment nodes: interior equation nodes,
    then boundary-closure nodes (``PreparedMeshfreeMetric.row_nodes``).
    ``incompatibility_witness`` is the native left-null witness ``y`` in original
    row coordinates (``A^T y = 0``, ``<y, b> != 0``), zero unless the exact
    signed solve classified the moments incompatible. ``derivative_available``
    follows ``derivative_contract``: signed exact needs a matching fixed-rank
    certificate of maximal rank, relaxed signed a successful stacked solve with
    certified full column rank, and nonnegative a bound regular fixed active set.
    ``kkt_rank_certificate`` is set by ``bind_active_set`` only when no positive
    KKT stability bound was certified: its ``route`` separates a budget-refused
    (``"unavailable"``, raise ``policy.rank`` materialization) from a computed
    singular or not-fixed-rank KKT Jacobian.
    """

    __strict_contract__ = True
    weights: Float64[_MetricEdgeDim]
    moment_residual: Float64[_MomentRowDim]
    normalized_residual: Float64[_MomentRowDim]
    slack: Float64[_MomentRowDim]
    row_status: Int32[_MomentRowDim]
    incompatibility_witness: Float64[_MomentRowDim]
    node_feasible: Bool[_EquationNodeDim]
    first_moment_residual: Float64[_EquationNodeDim]
    second_moment_residual: Float64[_EquationNodeDim]
    higher_moment_residual: Float64[_EquationNodeDim]
    amplification: Float64[_EquationNodeDim]
    negative_count: Int32[Scalar]
    zero_count: Int32[Scalar]
    rank: Int32[Scalar]
    rank_maximal: Bool[Scalar]
    rank_certificate: RectangularRankCertificate
    row_rank_profile: SparseRowRankEvidence | None
    status: Int32[Scalar]
    provider_status: Int32[Scalar]
    accepted: Bool[Scalar]
    exact: Bool[Scalar]
    nonnegative: Bool[Scalar]
    derivative_available: Bool[Scalar]
    linear_result: LinearSolveResult | None
    conic_execution: ConvexProgramExecution | None
    active_set: ConicActiveSetEvidence | None
    conic_sensitivity: PreparedMatrixFreeConicSensitivity | None
    kkt_rank_certificate: RectangularRankCertificate | None
    derivative_contract: str = eqx.field(static=True)

    @property
    def hilbert_admitted(self) -> Array:
        return self.accepted & jnp.all(jnp.isfinite(self.weights) & (self.weights > 0))

    @property
    def redundant_constraints(self) -> Array:
        """Left nullity of the moment design; ``-1`` without rank evidence."""
        rows = self.moment_residual.shape[0]
        return jnp.where(self.rank >= 0, rows - self.rank, -1).astype(jnp.int32)


def _coordinate_operator(
    relation: EdgeRelation,
    values: Array,
    source: ArraySpace,
    target: ArraySpace,
    storage: _SparseStoragePlan,
    *,
    symmetric: bool = False,
    positive: bool = False,
) -> SparseCoordinateOperator:
    evidence: dict[str, PropertyEvidence] = {}
    if symmetric:
        evidence["self_adjoint"] = "construction"
    if positive:
        evidence["positive_semidefinite"] = "construction"
    return SparseCoordinateOperator(
        relation,
        values,
        source=source,
        target=target,
        storage_plan=storage,
        properties=OperatorProperties(
            self_adjoint=symmetric,
            positive_semidefinite=positive,
            evidence=evidence,
        ),
    )


@final
class _KKTStability(StrictModule):
    """Weyl-verified smallest singular value of one projection-KKT Jacobian."""

    policy: SVDSolvePolicy

    def __call__(self, operator: JacobianLinearOperator, /) -> StabilityLowerBound:
        certificate = certify_rectangular_rank(operator, self.policy)
        bound = (
            certificate.smallest_retained_singular_value
            - certificate.singular_value_error_bound
        )
        regular = (
            certificate.fixed_rank
            & (certificate.rank == operator.source.size)
            & jnp.isfinite(bound)
            & (bound > 0)
        )
        return StabilityLowerBound(
            operator, jnp.where(regular, bound, 0.0), evidence="verified"
        )


@final
class MeshfreeMetricGeometry(StrictModule):
    """Moment-node coordinates framing the multilevel metric preconditioner.

    ``points`` has one row per moment node, in the order of
    ``PreparedMeshfreeMetric.row_nodes``. Coordinates are in the frame of the
    moment displacements: every edge displacement is the minimum-image
    coordinate difference of its endpoints. ``lower``/``upper`` bound the
    coordinate box, and ``periodic`` axes wrap with period ``upper - lower``.
    ``spacing`` is a typical nearest-node distance; it sets the finest spline
    cell at about two spacings. Only the preconditioner, and so the iteration
    count, depends on this geometry.
    """

    points: np.ndarray
    lower: tuple[float, ...] = eqx.field(static=True)
    upper: tuple[float, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    spacing: float = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        periodic: tuple[bool, ...],
        spacing: float,
    ) -> None:
        points_ = np.asarray(points, dtype=np.float64)
        lower_ = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper_ = np.asarray(upper, dtype=np.float64).reshape(-1)
        periodic_ = tuple(bool(axis) for axis in periodic)
        if points_.ndim != 2 or points_.shape[1] not in (1, 2, 3):
            raise ValueError("Metric geometry points must have shape (nodes, 1|2|3).")
        dimension = points_.shape[1]
        if lower_.size != dimension or upper_.size != dimension:
            raise ValueError("Metric geometry box must match the point dimension.")
        if len(periodic_) != dimension:
            raise ValueError("Metric geometry needs one periodic flag per axis.")
        if not (
            np.all(np.isfinite(points_))
            and np.all(np.isfinite(lower_) & np.isfinite(upper_))
            and np.all(upper_ > lower_)
        ):
            raise ValueError("Metric geometry needs finite points and a positive box.")
        if not (isfinite(float(spacing)) and float(spacing) > 0):
            raise ValueError("Metric geometry spacing must be finite and positive.")
        self.points = points_
        self.lower = tuple(float(value) for value in lower_)
        self.upper = tuple(float(value) for value in upper_)
        self.periodic = periodic_
        self.spacing = float(spacing)


@final
class PreparedMeshfreeMetric(StrictModule):
    """One immutable sparse moment pattern with differentiable numeric data."""

    __strict_contract__ = True
    constraint: SparseCoordinateOperator
    rhs: Float64[_MomentRowDim]
    row_scaling: Float64[_MomentRowDim]
    prior: Float64[_MetricEdgeDim]
    policy: MeshfreeMetricPolicy
    row_rank_profile: SparseRowRankEvidence | None
    slack_rows: Int32[_SlackRowDim]
    conic_routes: Int32[_ConicRouteDim]
    conic_moments: Int32[_ConicMomentDim]
    linear_prepared: PreparedLinearSolve | None
    geometry: MeshfreeMetricGeometry | None
    preconditioner_levels: tuple[int, ...] = eqx.field(static=True)
    conic_relation: EdgeRelation
    conic_storage: _SparseStoragePlan
    conic_source: ArraySpace
    conic_target: ArraySpace
    objective_relation: EdgeRelation
    objective_storage: _SparseStoragePlan
    conic_prepared: PreparedConvexProgram | None
    row_nodes: Int32[_MomentRowDim]
    row_moments: Int32[_MomentRowDim]
    row_degrees: Int32[_MomentRowDim]
    equation_count: int = eqx.field(static=True)
    boundary_count: int = eqx.field(static=True)
    moment_count: int = eqx.field(static=True)
    boundary_moment_count: int = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    moment_degrees: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        constraint: SparseCoordinateOperator,
        rhs: ArrayLike,
        row_scaling: ArrayLike,
        prior: ArrayLike,
        policy: MeshfreeMetricPolicy,
        *,
        equation_count: int,
        intrinsic_dimension: int,
        boundary_count: int = 0,
        geometry: MeshfreeMetricGeometry | None = None,
    ) -> None:
        """Prepare provider structure over every original moment row.

        Rows are grouped by moment node: ``equation_count`` interior nodes with
        every declared moment, then ``boundary_count`` boundary-closure nodes
        with the moments selected by ``policy.boundary_closure``. The
        boundary-flux right-hand side is the caller's. This host-only assembly
        keeps one row ordering across the transformed linear template, the
        optional pivot diagnostic and conic preparation. ``geometry`` frames the
        multilevel preconditioner of the ``"multilevel-craig"`` exact signed solve
        and is required by it.
        """
        rhs_ = jnp.asarray(rhs, dtype=jnp.float64)
        scaling_ = jnp.asarray(row_scaling, dtype=jnp.float64)
        prior_ = jnp.asarray(prior, dtype=jnp.float64)
        if not isinstance(policy, MeshfreeMetricPolicy):
            raise TypeError("policy must be a MeshfreeMetricPolicy.")
        if (
            equation_count < 0
            or boundary_count < 0
            or equation_count + boundary_count < 1
            or intrinsic_dimension not in (1, 2, 3)
        ):
            raise ValueError(
                "Moment preparation requires moment nodes and intrinsic dimension 1, 2 or 3."
            )
        exponents = _moment_exponents(intrinsic_dimension, policy.moment_degree)
        moment_count = exponents.shape[0]
        boundary_moments = _boundary_moment_indices(
            intrinsic_dimension, policy.boundary_closure
        )
        boundary_moment_count = boundary_moments.size
        row_nodes = np.concatenate(
            (
                np.repeat(np.arange(equation_count, dtype=np.int32), moment_count),
                equation_count
                + np.repeat(
                    np.arange(boundary_count, dtype=np.int32), boundary_moment_count
                ),
            )
        )
        row_moments = np.concatenate(
            (
                np.tile(np.arange(moment_count, dtype=np.int32), equation_count),
                np.tile(boundary_moments, boundary_count),
            )
        )
        relation = constraint.relation
        if not isinstance(relation, EdgeRelation):
            raise TypeError(
                "Moment constraints require native edge-list sparse coordinates."
            )
        if not isinstance(constraint.source, ArraySpace) or not isinstance(
            constraint.target, ArraySpace
        ):
            raise TypeError(
                "Moment constraints require native array-coordinate source and target spaces."
            )
        if len(constraint.source.shape) != 1 or len(constraint.target.shape) != 1:
            raise ValueError("Moment spaces must use one edge and one equation vector.")
        if rhs_.shape != row_nodes.shape or scaling_.shape != rhs_.shape:
            raise ValueError(
                "Moment RHS and scaling must match equation nodes and declared moments."
            )
        if (
            prior_.shape != (constraint.source.size,)
            or constraint.target.size != rhs_.size
        ):
            raise ValueError("Moment design, RHS and edge prior dimensions disagree.")
        if not np.all(np.isfinite(np.asarray(scaling_)) & (np.asarray(scaling_) > 0)):
            raise ValueError("Moment row scaling must be finite and positive.")
        if not np.all(np.isfinite(np.asarray(prior_)) & (np.asarray(prior_) > 0)):
            raise ValueError("Moment edge prior must be finite and positive.")
        rows = np.asarray(relation.target_indices)
        columns = np.asarray(relation.source_indices)
        coefficients = np.asarray(
            constraint.coefficients / scaling_[relation.target_indices]
        )
        if not np.all(np.isfinite(coefficients)) or not np.all(
            np.isfinite(np.asarray(rhs_))
        ):
            raise ValueError("Prepared moment data must be finite.")
        normalized_design = eqx.tree_at(
            lambda op: op.coefficients,
            constraint,
            jnp.asarray(coefficients, dtype=jnp.float64),
        )
        relaxed = policy.acceptance == "relaxed"
        # Relaxed acceptance penalizes every row; exact acceptance none.
        slack_rows = (
            np.arange(row_nodes.size, dtype=np.int32)
            if relaxed
            else np.zeros((0,), dtype=np.int32)
        )
        # An exact nonnegative program poses equalities to an interior-point
        # method, which needs independent rows: the tolerance-defined pivot
        # profile presolves dependent rows. It is not a rank certificate; every
        # original row, including the dropped ones, is audited after the solve.
        exact_conic = policy.sign == "nonnegative" and not relaxed
        row_rank_profile = (
            prepare_sparse_row_rank(
                normalized_design,
                SparseRowRankPolicy(
                    policy.rank.rank,
                    maximum_rows=policy.maximum_symbolic_entries,
                    maximum_input_nonzeros=policy.maximum_symbolic_entries,
                    maximum_factor_nonzeros=policy.maximum_symbolic_entries,
                    maximum_elimination_work=policy.maximum_symbolic_entries,
                ),
            )
            if policy.row_rank_profile or exact_conic
            else None
        )
        edges = constraint.source.size
        moments = constraint.target.size
        linear_prepared: PreparedLinearSolve | None = None
        transfers: tuple[
            tuple[SparseCoordinateOperator, SparseCoordinateOperator], ...
        ] = ()
        setup: SparseCoordinateOperator | None = None
        if policy.solver == "multilevel-craig":
            if geometry is None:
                raise ValueError(
                    "The 'multilevel-craig' metric solver needs MeshfreeMetricGeometry "
                    "with coordinate moment displacements; declare solver='lsmr' for "
                    "intrinsic surface charts."
                )
            if geometry.points.shape != (
                equation_count + boundary_count,
                intrinsic_dimension,
            ):
                raise ValueError(
                    "Metric geometry needs one intrinsic-dimension point per moment node."
                )
            design = sp.csr_matrix(
                (coefficients * np.sqrt(np.asarray(prior_))[columns], (rows, columns)),
                shape=(moments, edges),
            )
            setup, transfers = _multilevel_setup(
                design,
                geometry,
                exponents[row_moments],
                row_nodes,
                np.asarray(scaling_),
                policy,
                constraint.target,
            )
        if policy.sign == "signed":
            transformed = _solved_operator(
                normalized_design,
                jnp.sqrt(prior_),
                _row_equilibration(normalized_design, prior_)
                if policy.solver == "lsmr" and policy.acceptance == "exact"
                else None,
            )
            linear_prepared = prepare(
                _signed_problem(transformed, policy, None),
                _signed_policy(policy, transfers, setup),
            )
        conic_moments = (
            np.asarray(row_rank_profile.selected_rows, dtype=np.int32)
            if exact_conic and row_rank_profile is not None
            else np.arange(moments, dtype=np.int32)
        )
        if conic_moments.size == 0:
            raise ValueError(
                "Moment design has zero rank; no diffusion metric is definable."
            )
        conic_inverse = np.full(moments, -1, dtype=np.int32)
        conic_inverse[conic_moments] = np.arange(conic_moments.size, dtype=np.int32)
        conic_routes = np.flatnonzero(conic_inverse[rows] >= 0).astype(np.int32)
        equalities = conic_moments.size
        conic_slacks = conic_inverse[slack_rows]
        variables = edges + conic_slacks.size
        conic_rows = conic_inverse[rows[conic_routes]]
        conic_columns = columns[conic_routes]
        conic_rows = np.concatenate((conic_rows, conic_slacks))
        conic_columns = np.concatenate(
            (conic_columns, edges + np.arange(conic_slacks.size))
        )
        conic_rows = np.concatenate((conic_rows, equalities + np.arange(edges)))
        conic_columns = np.concatenate((conic_columns, np.arange(edges)))
        conic_relation = EdgeRelation(
            conic_columns,
            conic_rows,
            source_size=variables,
            target_size=equalities + edges,
        )
        conic_storage = _SparseStoragePlan(conic_relation)
        conic_source = ArraySpace(
            (variables,),
            dtype=jnp.float64,
            space_id=f"{constraint.source.space_id}:conic",
        )
        conic_target = ArraySpace(
            (equalities + edges,),
            dtype=jnp.float64,
            space_id=f"{constraint.target.space_id}:conic",
        )
        objective_relation = EdgeRelation(
            np.arange(variables),
            np.arange(variables),
            source_size=variables,
            target_size=variables,
        )
        objective_storage = _SparseStoragePlan(objective_relation)
        conic_prepared: PreparedConvexProgram | None = None
        if policy.sign == "nonnegative":
            initial_values = jnp.asarray(coefficients[conic_routes], dtype=jnp.float64)
            initial_quadratic = 1 / prior_
            initial_values = jnp.concatenate(
                (initial_values, jnp.ones((conic_slacks.size,)))
            )
            initial_quadratic = jnp.concatenate(
                (initial_quadratic, jnp.full((conic_slacks.size,), policy.slack_penalty))
            )
            initial_values = jnp.concatenate((initial_values, -jnp.ones((edges,))))
            matrix = _coordinate_operator(
                conic_relation, initial_values, conic_source, conic_target, conic_storage
            )
            quadratic = _coordinate_operator(
                objective_relation,
                initial_quadratic,
                conic_source,
                conic_source,
                objective_storage,
                symmetric=True,
                positive=True,
            )
            program = ConicProgram(
                quadratic,
                jnp.zeros((variables,), dtype=jnp.float64),
                matrix,
                jnp.concatenate(
                    (
                        (rhs_ / scaling_)[conic_moments],
                        jnp.zeros((edges,), dtype=jnp.float64),
                    )
                ),
                ProductCone((ZeroCone(equalities), NonnegativeCone(edges))),
                problem_id=f"{constraint.operator_id}:nonnegative",
                convexity_evidence="construction",
            )
            conic_prepared = prepare_convex_program(program, policy.conic)
        self.constraint = constraint
        self.rhs = rhs_
        self.row_scaling = scaling_
        self.prior = prior_
        self.policy = policy
        self.equation_count = int(equation_count)
        self.boundary_count = int(boundary_count)
        self.intrinsic_dimension = int(intrinsic_dimension)
        self.moment_count = moment_count
        self.boundary_moment_count = boundary_moment_count
        self.moment_degrees = tuple(int(value) for value in exponents.sum(axis=1))
        self.row_nodes = jnp.asarray(row_nodes)
        self.row_moments = jnp.asarray(row_moments)
        self.row_degrees = jnp.asarray(
            exponents.sum(axis=1)[row_moments], dtype=jnp.int32
        )
        self.row_rank_profile = row_rank_profile
        self.slack_rows = jnp.asarray(slack_rows)
        self.conic_routes = jnp.asarray(conic_routes)
        self.conic_moments = jnp.asarray(conic_moments)
        self.linear_prepared = linear_prepared
        self.geometry = geometry
        self.preconditioner_levels = tuple(
            int(prolongation.source.size) for _, prolongation in transfers
        )
        self.conic_relation = conic_relation
        self.conic_storage = conic_storage
        self.conic_source = conic_source
        self.conic_target = conic_target
        self.objective_relation = objective_relation
        self.objective_storage = objective_storage
        self.conic_prepared = conic_prepared

    def _numeric(
        self,
        prior: ArrayLike | None,
        rhs: ArrayLike | None,
        coefficients: ArrayLike | None,
        /,
    ) -> tuple[Array, Array, Array]:
        phi = self.prior if prior is None else jnp.asarray(prior, dtype=jnp.float64)
        raw_rhs = self.rhs if rhs is None else jnp.asarray(rhs, dtype=jnp.float64)
        raw_values = (
            self.constraint.coefficients
            if coefficients is None
            else jnp.asarray(coefficients, dtype=jnp.float64)
        )
        if (
            phi.shape != self.prior.shape
            or raw_rhs.shape != self.rhs.shape
            or raw_values.shape != self.constraint.coefficients.shape
        ):
            raise ValueError(
                "Metric refresh must preserve the prepared sparse dimensions."
            )
        phi = eqx.error_if(
            phi,
            jnp.any(~jnp.isfinite(phi) | (phi <= 0)),
            "Metric prior must be finite and strictly positive.",
        )
        return phi, raw_rhs, raw_values

    def _normalized_values(self, raw_values: Array, /) -> Array:
        relation = cast(EdgeRelation, self.constraint.relation)
        return raw_values / self.row_scaling[relation.target_indices]

    def _conic_program(
        self, phi: Array, normalized_values: Array, normalized_rhs: Array, /
    ) -> PreparedConvexProgram:
        if self.conic_prepared is None:
            raise RuntimeError(
                "Nonnegative metric has no prepared native conic lifecycle."
            )
        conic_values = normalized_values[self.conic_routes]
        conic_rhs = normalized_rhs[self.conic_moments]
        quadratic_values = 1 / phi
        slacks = self.slack_rows.size
        conic_values = jnp.concatenate((conic_values, jnp.ones((slacks,))))
        quadratic_values = jnp.concatenate(
            (quadratic_values, jnp.full((slacks,), self.policy.slack_penalty))
        )
        conic_values = jnp.concatenate((conic_values, -jnp.ones(phi.shape)))
        quadratic = _coordinate_operator(
            self.objective_relation,
            quadratic_values,
            self.conic_source,
            self.conic_source,
            self.objective_storage,
            symmetric=True,
            positive=True,
        )
        matrix = _coordinate_operator(
            self.conic_relation,
            conic_values,
            self.conic_source,
            self.conic_target,
            self.conic_storage,
        )
        program = eqx.tree_at(
            lambda p: (p.quadratic, p.constraint_matrix, p.constraint_rhs),
            self.conic_prepared.program,
            (quadratic, matrix, jnp.concatenate((conic_rhs, jnp.zeros(phi.shape)))),
        )
        return PreparedConvexProgram(
            program,
            self.conic_prepared.template,
            numeric_version=self.conic_prepared.numeric_version + 1,
            numeric_binding_id=f"{self.conic_prepared.numeric_binding_id}:metric-refresh",
        )

    def solve(
        self,
        *,
        prior: ArrayLike | None = None,
        rhs: ArrayLike | None = None,
        coefficients: ArrayLike | None = None,
    ) -> MeshfreeMetricResult:
        """Solve and retain the complete metric admission evidence.

        Signed and conic branches share the final physical moment assessment.
        Keep their reduction order and failure evidence explicit rather than
        hiding provider-dependent admission in small forwarding helpers.
        """
        phi, raw_rhs, raw_values = self._numeric(prior, rhs, coefficients)
        values = self._normalized_values(raw_values)
        normalized_rhs = raw_rhs / self.row_scaling
        relaxed = self.policy.acceptance == "relaxed"
        signed_exact = self.policy.sign == "signed" and not relaxed
        root = jnp.sqrt(phi)
        design = eqx.tree_at(lambda op: op.coefficients, self.constraint, values)
        equilibration = (
            _row_equilibration(design, phi)
            if signed_exact and self.policy.solver == "lsmr"
            else None
        )
        transformed = _solved_operator(design, root, equilibration)
        # Rank evidence is a property of the numeric operator, not a
        # differentiable quantity; certify a stopped copy of the same revision.
        certificate = certify_rectangular_rank(
            _solved_operator(
                eqx.tree_at(
                    lambda op: op.coefficients,
                    self.constraint,
                    jax.lax.stop_gradient(values),
                ),
                jax.lax.stop_gradient(root),
                equilibration,
            ),
            self.policy.rank,
        )
        rank_maximal = (certificate.rank >= 0) & (
            certificate.rank == min(certificate.rows, certificate.columns)
        )
        witness = jnp.zeros_like(raw_rhs)
        incompatible = jnp.asarray(False)
        linear_result: LinearSolveResult | None = None
        conic_execution: ConvexProgramExecution | None = None
        if self.policy.sign == "signed":
            if self.linear_prepared is None:
                raise RuntimeError("Signed metric requires its prepared native solve.")
            problem = _signed_problem(transformed, self.policy, certificate)
            bound = refresh(self.linear_prepared, problem)
            if relaxed:
                penalty = jnp.sqrt(jnp.asarray(self.policy.slack_penalty))
                linear_result = solve(
                    bound, (penalty * normalized_rhs, jnp.zeros_like(phi))
                )
                derivative = linear_result.successful & linear_result.derivative_valid
            else:
                row_transform = (
                    jnp.ones_like(normalized_rhs)
                    if equilibration is None
                    else equilibration
                )
                linear_result = solve(bound, normalized_rhs * row_transform)
                evidence = linear_result.minimum_norm
                if evidence is None:
                    raise RuntimeError(
                        "Native minimum-norm solves must return constraint evidence."
                    )
                incompatible = evidence.incompatible
                # B = E D^-1 A S (E = I for Craig), so B^T y' = 0 gives
                # A^T (D^-1 E y') = 0.
                witness = jnp.where(
                    incompatible,
                    jnp.asarray(evidence.left_null_witness)
                    * row_transform
                    / self.row_scaling,
                    0.0,
                )
            if not relaxed:
                derivative = (
                    linear_result.successful
                    & linear_result.derivative_valid
                    & rank_maximal
                )
            weights = root * linear_result.value
            provider_status = linear_result.status.astype(jnp.int32)
            provider_success = linear_result.successful
        else:
            conic_execution = solve_prepared_convex_program(
                self._conic_program(phi, values, normalized_rhs)
            )
            conic_result = conic_execution.result
            weights = conic_result.primal[: phi.size]
            provider_status = conic_result.status.astype(jnp.int32)
            provider_success = conic_result.successful
            incompatible = jnp.asarray(not relaxed) & (
                conic_result.status == int(ConvexProgramStatus.PRIMAL_INFEASIBLE)
            )
            # The strict-active derivative is bound eagerly by bind_active_set.
            derivative = jnp.asarray(False)
        raw_constraint = eqx.tree_at(
            lambda op: op.coefficients, self.constraint, raw_values
        )
        residual = raw_constraint.mv(weights) - raw_rhs
        normalized = residual / self.row_scaling
        thresholds = self.policy.tolerance * (1 + jnp.abs(normalized_rhs))
        row_ok = jnp.abs(normalized) <= thresholds
        node_count = self.equation_count + self.boundary_count

        def node_max(values: Array) -> Array:
            return jax.ops.segment_max(
                values, self.row_nodes, num_segments=node_count, indices_are_sorted=True
            )

        node_ok = node_max((~row_ok).astype(jnp.int32)) == 0
        moment_magnitude = jnp.abs(residual)

        def degree_residual(mask: Array) -> Array:
            return node_max(jnp.where(mask, moment_magnitude, 0.0))

        degrees = self.row_degrees
        absolute_moment = eqx.tree_at(
            lambda op: op.coefficients, raw_constraint, jnp.abs(raw_values)
        ).mv(jnp.abs(weights))
        amplification = node_max(absolute_moment / (self.row_scaling + jnp.abs(raw_rhs)))
        finite = jnp.all(jnp.isfinite(weights)) & jnp.all(jnp.isfinite(residual))
        amplification_ok = jnp.all(amplification <= self.policy.amplification_limit)
        exact = jnp.all(row_ok)
        nonnegative = jnp.all(weights >= 0)
        sign_ok = nonnegative if self.policy.sign == "nonnegative" else jnp.asarray(True)
        # Only a validated witness classifies exact signed moments incompatible;
        # an unwitnessed row-audit failure is a provider failure, never a claim.
        unwitnessed = ~exact & ~incompatible if signed_exact else jnp.asarray(False)
        accepted = (
            finite
            & provider_success
            & amplification_ok
            & sign_ok
            & (jnp.asarray(relaxed) | exact)
        )
        status = jnp.where(
            ~finite,
            int(MeshfreeMetricStatus.NONFINITE),
            jnp.where(
                incompatible,
                int(MeshfreeMetricStatus.INCOMPATIBLE_MOMENTS),
                jnp.where(
                    ~provider_success | ~sign_ok | unwitnessed,
                    int(MeshfreeMetricStatus.PROVIDER_FAILURE),
                    jnp.where(
                        ~amplification_ok,
                        int(MeshfreeMetricStatus.AMPLIFICATION_LIMIT),
                        jnp.where(
                            ~exact & ~jnp.asarray(relaxed),
                            int(MeshfreeMetricStatus.INCOMPATIBLE_MOMENTS),
                            int(
                                MeshfreeMetricStatus.RELAXED
                                if relaxed
                                else MeshfreeMetricStatus.ACCEPTED
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return MeshfreeMetricResult(
            weights=weights,
            moment_residual=residual,
            normalized_residual=normalized,
            slack=residual if relaxed else jnp.zeros_like(residual),
            row_status=jnp.where(
                row_ok,
                int(MeshfreeMomentRowStatus.SATISFIED),
                int(
                    MeshfreeMomentRowStatus.RELAXED
                    if relaxed
                    else MeshfreeMomentRowStatus.INCOMPATIBLE
                ),
            ).astype(jnp.int32),
            incompatibility_witness=witness,
            node_feasible=node_ok,
            first_moment_residual=degree_residual(degrees == 1),
            second_moment_residual=degree_residual(degrees == 2),
            higher_moment_residual=degree_residual(degrees > 2),
            amplification=amplification,
            negative_count=jnp.sum(weights < 0, dtype=jnp.int32),
            zero_count=jnp.sum(weights == 0, dtype=jnp.int32),
            rank=certificate.rank,
            rank_maximal=rank_maximal,
            rank_certificate=certificate,
            row_rank_profile=self.row_rank_profile,
            status=status,
            provider_status=provider_status,
            accepted=accepted,
            exact=exact,
            nonnegative=nonnegative,
            derivative_available=derivative & accepted,
            linear_result=linear_result,
            conic_execution=conic_execution,
            active_set=None,
            conic_sensitivity=None,
            kkt_rank_certificate=None,
            derivative_contract=_derivative_contract(self.policy),
        )

    def bind_active_set(
        self,
        result: MeshfreeMetricResult,
        /,
        *,
        prior: ArrayLike | None = None,
        rhs: ArrayLike | None = None,
        coefficients: ArrayLike | None = None,
        fixed_active_set: ConicActiveSetEvidence | None = None,
    ) -> MeshfreeMetricResult:
        """Bind strict-active projection-KKT evidence to a nonnegative result.

        This is an eager boundary. ``prior``, ``rhs`` and ``coefficients`` must
        be the numeric data of the solve that produced ``result``; the native
        conic sensitivity re-audits that execution against the rebuilt program in
        original coordinates, so foreign data fails its KKT audit. Roles of every
        constraint are classified, and a derivative is admitted only for a
        regular fixed active set with Weyl-verified KKT nonsingularity under
        ``policy.rank``. ``fixed_active_set`` freezes roles from an earlier
        binding; any change refuses the derivative.
        """
        if self.policy.sign != "nonnegative":
            raise ValueError("Active-set evidence belongs to nonnegative metrics.")
        execution = result.conic_execution
        if execution is None:
            raise ValueError("Bind active sets to a nonnegative result of this metric.")
        phi, raw_rhs, raw_values = self._numeric(prior, rhs, coefficients)
        program = self._conic_program(
            phi, self._normalized_values(raw_values), raw_rhs / self.row_scaling
        )
        sensitivity = prepare_conic_sensitivity(
            program,
            execution,
            linear=LinearSolvePolicy(
                LSMR(),
                tolerance=TolerancePolicy(
                    relative=1e-13,
                    absolute=1e-14,
                    max_steps=self.policy.maximum_steps,
                ),
            ),
            representation="matrix-free",
            stability=_KKTStability(self.policy.rank),
            fixed_active_set=fixed_active_set,
        )
        if not isinstance(sensitivity, PreparedMatrixFreeConicSensitivity):
            raise TypeError("Sparse metric programs bind matrix-free sensitivities.")
        regular = sensitivity.active_set.status == int(
            ConicSensitivityStatus.REGULAR_FIXED_ACTIVE
        )
        # A nonpositive stability bound conflates a refused dense certificate
        # with a certified singular KKT; classify it explicitly at this eager
        # boundary. A refused certificate costs no SVD; only a computed singular
        # case pays a second one.
        kkt_certificate = (
            None
            if bool(sensitivity.stability.valid)
            else certify_rectangular_rank(sensitivity.operator, self.policy.rank)
        )
        return eqx.tree_at(
            lambda owner: (
                owner.active_set,
                owner.conic_sensitivity,
                owner.derivative_available,
                owner.kkt_rank_certificate,
            ),
            result,
            (
                sensitivity.active_set,
                sensitivity,
                result.accepted & regular,
                kkt_certificate,
            ),
            is_leaf=lambda value: value is None,
        )

    def strict_active_jvp(
        self,
        result: MeshfreeMetricResult,
        /,
        *,
        prior: ArrayLike | None = None,
        rhs: ArrayLike | None = None,
        coefficients: ArrayLike | None = None,
    ) -> tuple[Array, ConicSensitivityResult]:
        """Edge-weight tangent of a bound nonnegative metric on its fixed active set.

        Tangents are in the physical coordinates of ``solve``: prior ``Phi``,
        raw moment RHS, and raw moment coefficients. The weight tangent is NaN
        wherever the native sensitivity is unavailable.
        """
        sensitivity = result.conic_sensitivity
        if sensitivity is None:
            raise ValueError(
                "Strict-active derivatives need a result bound by bind_active_set."
            )
        data = sensitivity.original_data
        bound_quadratic = data.quadratic
        if not isinstance(bound_quadratic, SparseCoordinateOperator):
            raise TypeError("Metric programs bind a sparse diagonal objective.")
        inverse_phi = bound_quadratic.coefficients[: self.prior.size]
        zero_edges = jnp.zeros(self.prior.shape, dtype=jnp.float64)
        dphi = zero_edges if prior is None else jnp.asarray(prior, dtype=jnp.float64)
        drhs = (
            jnp.zeros_like(self.rhs)
            if rhs is None
            else jnp.asarray(rhs, dtype=jnp.float64)
        )
        dvalues = (
            jnp.zeros_like(self.constraint.coefficients)
            if coefficients is None
            else jnp.asarray(coefficients, dtype=jnp.float64)
        )
        if (
            dphi.shape != self.prior.shape
            or drhs.shape != self.rhs.shape
            or dvalues.shape != self.constraint.coefficients.shape
        ):
            raise ValueError("Metric tangents must match the prepared sparse dimensions.")
        # The bound quadratic stores 1/Phi on edge variables: d(1/Phi) = -dPhi/Phi^2.
        quadratic = -dphi * inverse_phi * inverse_phi
        matrix = self._normalized_values(dvalues)[self.conic_routes]
        normalized_rhs = (drhs / self.row_scaling)[self.conic_moments]
        slacks = self.slack_rows.size
        quadratic = jnp.concatenate((quadratic, jnp.zeros((slacks,))))
        matrix = jnp.concatenate((matrix, jnp.zeros((slacks,))))
        matrix = jnp.concatenate((matrix, zero_edges))
        tangent = eqx.tree_at(
            lambda value: (
                value.quadratic,
                value.constraint_matrix,
                value.constraint_rhs,
            ),
            ConicProgramData(
                bound_quadratic,
                jnp.zeros_like(data.linear),
                data.constraint_matrix,
                data.constraint_rhs,
                jnp.zeros_like(data.lower_bounds),
                jnp.zeros_like(data.upper_bounds),
            ),
            (
                eqx.tree_at(lambda op: op.coefficients, bound_quadratic, quadratic),
                eqx.tree_at(
                    lambda op: op.coefficients,
                    cast(SparseCoordinateOperator, data.constraint_matrix),
                    matrix,
                ),
                jnp.concatenate((normalized_rhs, zero_edges)),
            ),
        )
        sensitivity_result = conic_primal_jvp(sensitivity, tangent)
        return jnp.asarray(sensitivity_result.value)[: self.prior.size], (
            sensitivity_result
        )


def _transformed_design(
    design: SparseCoordinateOperator, root: Array, /
) -> AbstractLinearOperator:
    return design @ DiagonalLinearOperator(
        root,
        space=design.source,
        operator_id=f"{design.operator_id}:metric-root",
    )


def _row_equilibration(design: SparseCoordinateOperator, phi: Array, /) -> Array:
    """Reciprocal Euclidean norms of the rows of ``D^-1 A S``; zero rows keep one.

    Left-scaling a consistent system changes neither its solution set nor the
    minimum-norm solution, so it only conditions the exact solve; every original
    row is still audited. It is a stopped preconditioner, not model data.
    """
    squared = eqx.tree_at(
        lambda op: op.coefficients, design, design.coefficients * design.coefficients
    ).mv(phi)
    norm = jnp.sqrt(jax.lax.stop_gradient(squared))
    return jnp.where(norm > 0, 1 / jnp.where(norm > 0, norm, 1.0), 1.0)


def _solved_operator(
    design: SparseCoordinateOperator,
    root: Array,
    equilibration: Array | None,
    /,
) -> AbstractLinearOperator:
    transformed = _transformed_design(design, root)
    if equilibration is None:
        return transformed
    return (
        DiagonalLinearOperator(
            equilibration,
            space=transformed.target,
            operator_id=f"{transformed.operator_id}:row-equilibration",
        )
        @ transformed
    )


def _signed_problem(
    transformed: AbstractLinearOperator,
    policy: MeshfreeMetricPolicy,
    certificate: RectangularRankCertificate | None,
    /,
) -> MinimumNormProblem | LeastSquaresProblem:
    match policy.acceptance:
        case "exact":
            return MinimumNormProblem(
                transformed,
                problem_id=f"{transformed.operator_id}:exact-minimum-norm",
                rank_certificate=certificate,
            )
        case "relaxed":
            penalty = DiagonalLinearOperator(
                jnp.full(
                    (transformed.target.size,),
                    np.sqrt(policy.slack_penalty),
                    dtype=jnp.float64,
                ),
                space=transformed.target,
                operator_id=f"{transformed.operator_id}:slack-penalty",
            )
            stack = StackedLinearOperator(
                (penalty @ transformed, IdentityLinearOperator(transformed.source)),
                operator_id=f"{transformed.operator_id}:relaxed-stack",
            )
            return LeastSquaresProblem(
                stack, problem_id=f"{transformed.operator_id}:relaxed-least-squares"
            )
        case acceptance:
            raise ValueError(f"Unknown metric acceptance {acceptance!r}.")


_SYMMETRIC_CYCLE = PreconditionerProperties(
    linear=True,
    stationary=True,
    self_adjoint=True,
    positive_definite=True,
    evidence={
        name: "construction"
        for name in ("linear", "stationary", "self_adjoint", "positive_definite")
    },
)
"""A V-cycle with symmetric Gauss-Seidel on every level of the positive
semidefinite setup ``B B^T + c Z``. Every level has a positive diagonal: null
jet columns are dropped and empty rows decoupled. The cycle is a fixed,
linear, self-adjoint, positive-definite action (convergent symmetric smoothing)."""


def _multilevel_setup(
    design: sp.csr_matrix,
    geometry: MeshfreeMetricGeometry,
    exponents: np.ndarray,
    row_nodes: np.ndarray,
    row_scaling: np.ndarray,
    policy: MeshfreeMetricPolicy,
    space: ArraySpace,
    /,
) -> tuple[
    SparseCoordinateOperator,
    tuple[tuple[SparseCoordinateOperator, SparseCoordinateOperator], ...],
]:
    """Frozen setup operator and jet transfers of the Craig V-cycle.

    The setup operator is ``B B^T`` of the prepared numerics. Rows of ``B`` that
    vanish at preparation get a typical diagonal entry ``c`` in the setup
    operator only. One example is the ``xy`` moment on axis-aligned lattice
    edges. Without that entry Gauss-Seidel would be undefined there. The jets
    leave such rows out. Craig still iterates on the true ``B B^T``, so the
    minimum-norm solution of every original row is unchanged.
    """
    gram = sp.csr_matrix(design @ design.T)
    if gram.nnz > policy.multilevel_assembly.max_nnz:
        raise ValueError(
            f"Metric row Gram B B^T needs {gram.nnz} entries, exceeding "
            f"multilevel_assembly.max_nnz={policy.multilevel_assembly.max_nnz}."
        )
    diagonal = gram.diagonal()
    empty = diagonal == 0
    if np.all(empty):
        raise ValueError("Metric moment design has no nonzero row.")
    gram = sp.csr_matrix(
        gram + sp.diags(np.where(empty, np.median(diagonal[~empty]), 0.0))
    )
    gram.sum_duplicates()
    gram.sort_indices()
    entries = gram.tocoo()
    setup = SparseCoordinateOperator(
        EdgeRelation(
            entries.col.astype(np.int32),
            entries.row.astype(np.int32),
            source_size=space.size,
            target_size=space.size,
        ),
        jnp.asarray(entries.data, dtype=jnp.float64),
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id=f"{space.space_id}:metric-row-gram-setup",
    )
    transfers = metric_jet_transfers(
        geometry.points,
        exponents,
        row_nodes,
        row_scaling,
        design,
        lower=np.asarray(geometry.lower),
        upper=np.asarray(geometry.upper),
        periodic=geometry.periodic,
        spacing=geometry.spacing,
        moment_degree=policy.moment_degree,
        target=space,
    )
    if not transfers:
        raise RuntimeError("Metric jet hierarchy has no energetic coarse field.")
    return setup, transfers


def _signed_policy(
    policy: MeshfreeMetricPolicy,
    transfers: tuple[tuple[SparseCoordinateOperator, SparseCoordinateOperator], ...],
    setup: SparseCoordinateOperator | None,
    /,
) -> LinearSolvePolicy:
    # An absolute Euclidean residual bound tol implies every row satisfies the
    # per-row admission threshold tol * (1 + |b_i|); relaxed solves converge on
    # their normal equations relative to the data.
    match policy.acceptance:
        case "exact":
            tolerance = TolerancePolicy(
                relative=0.0,
                absolute=policy.tolerance,
                max_steps=policy.maximum_steps,
            )
        case "relaxed":
            tolerance = TolerancePolicy(
                relative=policy.tolerance,
                absolute=policy.tolerance,
                max_steps=policy.maximum_steps,
            )
        case acceptance:
            raise ValueError(f"Unknown metric acceptance {acceptance!r}.")
    match policy.solver:
        case "lsmr":
            return LinearSolvePolicy(
                GeneralizedLSMR(),
                tolerance=tolerance,
                differentiation=DifferentiationPolicy("mathematical"),
                failure=FailurePolicy("status"),
            )
        case "multilevel-craig":
            if setup is None or not transfers:
                raise ValueError(
                    "The multilevel metric solve needs its prepared hierarchy."
                )
            smoother = GaussSeidelPreconditionerBuilder(direction="symmetric")
            hierarchy = GalerkinHierarchyBuilder(
                transfers,
                (smoother,) * len(transfers),
                smoother,
                properties=_SYMMETRIC_CYCLE,
                assembly=policy.multilevel_assembly,
            )
            # The cycle is built once from the prepared setup operator and then
            # frozen. Numeric refreshes, including traced coefficient and prior
            # updates, keep a valid SPD preconditioner, and Craig stays exact.
            # Only the iteration count follows how far the numerics drift.
            return LinearSolvePolicy(
                Craig(),
                tolerance=tolerance,
                preconditioning=PreconditioningPolicy(
                    hierarchy, setup_operator=setup, refresh="frozen"
                ),
                differentiation=DifferentiationPolicy("mathematical"),
                failure=FailurePolicy("status"),
                resources=policy.multilevel_resources,
            )
        case None:
            raise ValueError("Nonnegative metrics have no signed solver.")
        case unknown:
            assert_never(unknown)


def _derivative_contract(policy: MeshfreeMetricPolicy, /) -> str:
    match policy.sign, policy.acceptance:
        case "signed", "exact":
            return (
                "all-equation minimum-norm derivative on a certified maximal fixed rank"
            )
        case "signed", "relaxed":
            return "all-equation stacked least-squares derivative, certified full column rank"
        case "nonnegative", _:
            return (
                "strict-active fixed-set projection-KKT derivative after bind_active_set"
            )
        case sign, acceptance:
            raise ValueError(f"Unknown metric policy {sign!r}/{acceptance!r}.")
