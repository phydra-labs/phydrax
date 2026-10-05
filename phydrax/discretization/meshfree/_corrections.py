# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Learned candidate corrections constrained by complete meshfree contracts.

A learned model proposes a candidate ``w_c = w_base + delta`` for the undirected
metric edge weights of one prepared exterior graph. The executed correction is
the minimum change in the declared weighted norm ``|w - w_c|_{Phi^-1}`` over the
*full* moment affine set ``{w : A w = b}``, every original row included. With
``S = sqrt(Phi)``, ``w = w_c + S z`` and the positive row equilibration
``E = diag(1 / |A_i S|)``, the signed route is the native rectangular
minimum-norm solve of ``E A S z = E (b - A w_c)``; left scaling preserves the
affine set and the minimizer. The map ``w_c -> w`` is affine with the weighted
orthogonal projector as derivative, executed by the same native solve.

Positivity is never repaired by clipping. A signed projection whose minimum
weight falls below the declared margin is refused, or, only when explicitly
selected, replaced by the native conic correction
``min 0.5 |w - w_c|^2_{Phi^-1}`` subject to ``A w = b`` and ``w >= margin``.
Dependent moment rows are presolved by the native sparse row-rank profile for
the interior point; every original row is audited afterwards. Infeasibility is
reported with an independently audited Farkas or left-null witness.

Moments, sign margin, coercivity (native no-shift sparse Cholesky of the
Dirichlet-reduced weighted graph Laplacian) and correction norm are separate
evidence fields. Oriented edge fluxes are a distinct object: a learned flux is
evaluated in both endpoint orders, its parity ``F_ij = -F_ji`` is audited, and
an antisymmetric candidate is projected onto the declared nodal balance
``G F = q`` with the same native minimum-norm route. Undirected positive metric
coefficients are not antisymmetric fluxes and the two never share a type.
"""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import assert_never, final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...linalg import (
    ArraySpace,
    DifferentiationPolicy,
    FailurePolicy,
    GeneralizedLSMR,
    LinearDerivativeSolvePolicy,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSolveStatus,
    LSMR,
    MinimumNormProblem,
    OperatorProperties,
    prepare,
    prepare_sparse_row_rank,
    PreparedLinearSolve,
    RankPolicy,
    solve,
    SparseRowRankEvidence,
    SparseRowRankPolicy,
    TolerancePolicy,
)
from ...linalg.svd import DenseSVD, SVDSolvePolicy
from ...optim import (
    conic_primal_jvp,
    ConicProgram,
    ConicProgramData,
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
    refresh_convex_program,
    solve_prepared_convex_program,
    ZeroCone,
)
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ...typing import checked, Dim, Float64, Int32, parse, Scalar
from ._exterior import MeshfreeStiffnessEvidence, PreparedMeshfreeExteriorCalculus
from ._exterior_metric import _KKTStability


MeshfreeCorrectionRoute: TypeAlias = Literal["signed-minimum-norm", "nonnegative-conic"]
"""``signed-minimum-norm``: native rectangular minimum-norm projection.
``nonnegative-conic``: native conic correction with ``w >= margin``."""

MeshfreeCorrectionSignPolicy: TypeAlias = Literal["refuse", "constrained"]
"""Signed-route response to a positivity conflict: refuse with status, or
solve the explicitly selected constrained conic correction."""

MeshfreeCorrectionProvider: TypeAlias = Literal["none", "minimum-norm", "conic"]
MeshfreeCorrectionWitness: TypeAlias = Literal["none", "left-null", "farkas"]
MeshfreeCorrectionSubject: TypeAlias = Literal[
    "candidate", "projection", "conic-solution"
]
"""Which vector the evidence audits: the unchanged candidate (no provider point
exists), the signed projection, or the conic solution."""

MeshfreeCorrectionDerivative: TypeAlias = Literal[
    "unavailable", "minimum-norm-projector", "conic-fixed-active-set"
]


class MeshfreeCorrectionStatus(IntEnum):
    ADMITTED = 0
    MOMENT_INCOMPATIBLE = 1
    BALANCE_INCOMPATIBLE = 2
    POSITIVITY_CONFLICT = 3
    COERCIVITY_CONFLICT = 4
    PARITY_CONFLICT = 5
    INFEASIBLE = 6
    NONFINITE_CANDIDATE = 7
    PROVIDER_UNRESOLVED = 8
    AUDIT_FAILED = 9


class _CorrectionEdgeDim(Dim):
    """Compact canonical edges of one prepared exterior graph."""


class _CorrectionRowDim(Dim):
    """Every original constraint row: moment rows or nodal balance rows."""


class _CorrectionSelectedRowDim(Dim):
    """Independent moment rows posed as conic equalities."""


@final
class MeshfreeMetricCorrectionPolicy(StrictModule):
    """Declared route, sign response, positive margin and solver bounds.

    ``margin`` is the required lower bound of every corrected weight; it is a
    physical stiffness scale of the graph and has no default. ``tolerance``
    bounds every original moment row relative to ``D_i + |b_i|`` and the sign
    margin absolutely. The nonnegative conic route always enforces the margin,
    so it requires ``sign="constrained"``.
    """

    __strict_contract__ = True
    route: MeshfreeCorrectionRoute = eqx.field(static=True)
    sign: MeshfreeCorrectionSignPolicy = eqx.field(static=True)
    margin: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    rank: SVDSolvePolicy
    conic: ConvexSolvePolicy

    @checked
    def __init__(
        self,
        route: MeshfreeCorrectionRoute = "signed-minimum-norm",
        /,
        *,
        margin: float,
        sign: MeshfreeCorrectionSignPolicy = "refuse",
        tolerance: float = 1e-9,
        maximum_steps: int = 4096,
        rank: SVDSolvePolicy | None = None,
        conic: ConvexSolvePolicy | None = None,
    ) -> None:
        route_ = parse(route, MeshfreeCorrectionRoute, "route")
        sign_ = parse(sign, MeshfreeCorrectionSignPolicy, "sign")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)):
            raise TypeError("margin must be a real number.")
        if not isfinite(margin) or margin <= 0:
            raise ValueError("Corrected metric weights need a finite positive margin.")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise TypeError("tolerance must be a real number.")
        if not isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Correction tolerance must be finite and positive.")
        if isinstance(maximum_steps, bool) or not isinstance(maximum_steps, int):
            raise TypeError("maximum_steps must be an integer.")
        if maximum_steps < 1:
            raise ValueError("maximum_steps must be positive.")
        match route_:
            case "signed-minimum-norm":
                pass
            case "nonnegative-conic":
                if sign_ != "constrained":
                    raise ValueError(
                        "The nonnegative conic route always enforces the margin; "
                        "declare sign='constrained'."
                    )
            case unknown:
                assert_never(unknown)
        rank_ = (
            SVDSolvePolicy(DenseSVD(), rank=RankPolicy(relative_cutoff=1e-12))
            if rank is None
            else rank
        )
        conic_ = (
            ConvexSolvePolicy(
                NativeHomogeneousConic(),
                termination=ConvexTermination(
                    absolute=float(tolerance), maximum_steps=maximum_steps
                ),
                failure=FailurePolicy("status"),
            )
            if conic is None
            else conic
        )
        if not conic_.method.capabilities.sparse or conic_.regularization != 0:
            raise ValueError(
                "Correction conic provider must support sparse data without objective jitter."
            )
        self.route = route_
        self.sign = sign_
        self.margin = float(margin)
        self.tolerance = float(tolerance)
        self.maximum_steps = maximum_steps
        self.rank = rank_
        self.conic = conic_


@final
class MeshfreeMetricProjection(StrictModule):
    """Traceable signed projection of one candidate onto the full moment set.

    ``weights`` is affine in the candidate; its derivative is the weighted
    projector executed by the native minimum-norm solve (``rhs-only``
    derivative route). Consumers gate on ``admissible`` (solve success and
    sign margin); a refused projection must not be stepped on.
    """

    __strict_contract__ = True
    weights: Float64[_CorrectionEdgeDim]
    correction: Float64[_CorrectionEdgeDim]
    sign_margin: Float64[Scalar]
    linear: LinearSolveResult

    @property
    def successful(self) -> Array:
        return self.linear.successful

    @property
    def derivative_valid(self) -> Array:
        return self.linear.derivative_valid

    @property
    def admissible(self) -> Array:
        return self.linear.successful & (self.sign_margin >= 0)


@final
class MeshfreeMetricCorrectionEvidence(StrictModule):
    """Independent audits of one corrected (or refused) metric candidate.

    ``audited_weights`` is the vector named by ``audited_subject``; it is
    evidence, not a published metric. ``moment_residual`` is ``A w - b`` over
    every original row and ``normalized_moment_residual`` divides it by
    ``D_i + |b_i|``. ``sign_margin`` is ``min(w) - margin``. ``stiffness`` is
    the native no-shift sparse Cholesky evidence of the Dirichlet-reduced
    weighted graph Laplacian (``None`` for a nonfinite subject).
    ``correction_norm`` is ``|w - w_c|_{Phi^-1}``. ``witness`` lives in original
    raw rows (``A^T y = 0`` or ``A^T y >= 0``) and is audited independently of
    the provider. ``signed_minimum_weight`` records the signed projection's
    minimum when that projection was computed (NaN otherwise).
    """

    __strict_contract__ = True
    audited_weights: Float64[_CorrectionEdgeDim]
    moment_residual: Float64[_CorrectionRowDim]
    normalized_moment_residual: Float64[_CorrectionRowDim]
    witness: Float64[_CorrectionRowDim]
    stiffness: MeshfreeStiffnessEvidence | None
    status: MeshfreeCorrectionStatus = eqx.field(static=True)
    route: MeshfreeCorrectionRoute = eqx.field(static=True)
    provider: MeshfreeCorrectionProvider = eqx.field(static=True)
    provider_status: int = eqx.field(static=True)
    audited_subject: MeshfreeCorrectionSubject = eqx.field(static=True)
    maximum_moment_residual: float = eqx.field(static=True)
    moments_exact: bool = eqx.field(static=True)
    minimum_weight: float = eqx.field(static=True)
    sign_margin: float = eqx.field(static=True)
    positive: bool = eqx.field(static=True)
    coercive: bool = eqx.field(static=True)
    correction_norm: float = eqx.field(static=True)
    witness_kind: MeshfreeCorrectionWitness = eqx.field(static=True)
    witness_residual: float = eqx.field(static=True)
    witness_margin: float = eqx.field(static=True)
    witness_valid: bool = eqx.field(static=True)
    signed_provider_status: int = eqx.field(static=True)
    signed_minimum_weight: float = eqx.field(static=True)
    derivative: MeshfreeCorrectionDerivative = eqx.field(static=True)
    derivative_available: bool = eqx.field(static=True)
    sensitivity_status: int = eqx.field(static=True)
    margin: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    @property
    def admitted(self) -> bool:
        return self.status is MeshfreeCorrectionStatus.ADMITTED


@final
class MeshfreeMetricCorrection(StrictModule):
    """Audited correction; ``weights`` exists only when admitted."""

    __strict_contract__ = True
    evidence: MeshfreeMetricCorrectionEvidence
    candidate: Float64[_CorrectionEdgeDim]
    weights: Float64[_CorrectionEdgeDim] | None
    linear: LinearSolveResult | None
    conic: ConvexProgramExecution | None
    sensitivity: PreparedMatrixFreeConicSensitivity | None

    @property
    def admitted(self) -> bool:
        return self.evidence.admitted

    @property
    def status(self) -> MeshfreeCorrectionStatus:
        return self.evidence.status

    def require_weights(self) -> Array:
        if self.weights is None:
            raise ValueError(
                "Metric correction was refused with status "
                f"{self.evidence.status.name}; inspect its evidence."
            )
        return self.weights


@final
class MeshfreeEdgeFluxCandidate(StrictModule):
    """Learned oriented edge flux evaluated in both endpoint orders.

    ``forward[e]`` is the model flux from ``pairs[e, 0]`` to ``pairs[e, 1]`` and
    ``reverse[e]`` the same model evaluated with the endpoints swapped. A
    conservative oriented flux satisfies ``reverse == -forward``.
    """

    __strict_contract__ = True
    forward: Float64[_CorrectionEdgeDim]
    reverse: Float64[_CorrectionEdgeDim]

    def __init__(self, forward: ArrayLike, reverse: ArrayLike, /) -> None:
        forward_ = jnp.asarray(forward, dtype=jnp.float64)
        reverse_ = jnp.asarray(reverse, dtype=jnp.float64)
        if forward_.ndim != 1 or forward_.shape != reverse_.shape:
            raise ValueError(
                "Oriented flux candidates need one value per edge and orientation."
            )
        self.forward = forward_
        self.reverse = reverse_


@final
class MeshfreeEdgeFluxProjection(StrictModule):
    """Traceable balance projection of one oriented flux candidate."""

    __strict_contract__ = True
    flux: Float64[_CorrectionEdgeDim]
    correction: Float64[_CorrectionEdgeDim]
    parity_defect: Float64[Scalar]
    linear: LinearSolveResult

    @property
    def successful(self) -> Array:
        return self.linear.successful

    @property
    def derivative_valid(self) -> Array:
        return self.linear.derivative_valid


@final
class MeshfreeEdgeFluxCorrectionEvidence(StrictModule):
    """Parity, balance and witness audits of one oriented flux correction.

    ``parity_defect`` is ``forward + reverse`` of the learned candidate.
    ``balance_residual`` is ``G F - q`` on every equation node of the audited
    flux, normalized by ``sum_e |G_ie F_e| + |q_i|`` in
    ``maximum_balance_residual``.
    """

    __strict_contract__ = True
    audited_flux: Float64[_CorrectionEdgeDim]
    parity_defect: Float64[_CorrectionEdgeDim]
    balance_residual: Float64[_CorrectionRowDim]
    witness: Float64[_CorrectionRowDim]
    status: MeshfreeCorrectionStatus = eqx.field(static=True)
    provider: MeshfreeCorrectionProvider = eqx.field(static=True)
    provider_status: int = eqx.field(static=True)
    audited_subject: MeshfreeCorrectionSubject = eqx.field(static=True)
    maximum_parity_defect: float = eqx.field(static=True)
    antisymmetric: bool = eqx.field(static=True)
    maximum_balance_residual: float = eqx.field(static=True)
    balanced: bool = eqx.field(static=True)
    correction_norm: float = eqx.field(static=True)
    witness_kind: MeshfreeCorrectionWitness = eqx.field(static=True)
    witness_residual: float = eqx.field(static=True)
    witness_margin: float = eqx.field(static=True)
    witness_valid: bool = eqx.field(static=True)
    derivative_available: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    @property
    def admitted(self) -> bool:
        return self.status is MeshfreeCorrectionStatus.ADMITTED


@final
class MeshfreeEdgeFluxCorrection(StrictModule):
    """Audited oriented flux; ``flux`` exists only when admitted."""

    __strict_contract__ = True
    evidence: MeshfreeEdgeFluxCorrectionEvidence
    flux: Float64[_CorrectionEdgeDim] | None
    linear: LinearSolveResult | None

    @property
    def admitted(self) -> bool:
        return self.evidence.admitted

    def require_flux(self) -> Array:
        if self.flux is None:
            raise ValueError(
                "Edge flux correction was refused with status "
                f"{self.evidence.status.name}; inspect its evidence."
            )
        return self.flux


class _Outcome(NamedTuple):
    """Host record of one provider attempt before the shared audits."""

    values: np.ndarray | None
    subject: MeshfreeCorrectionSubject
    provider: MeshfreeCorrectionProvider
    provider_status: int
    refusal: MeshfreeCorrectionStatus | None
    witness: np.ndarray
    witness_kind: MeshfreeCorrectionWitness
    linear: LinearSolveResult | None
    execution: ConvexProgramExecution | None
    program: PreparedConvexProgram | None


class _WitnessAudit(NamedTuple):
    residual: float
    margin: float
    valid: bool


def _refusal(status: MeshfreeCorrectionStatus, rows: int, /) -> _Outcome:
    """A refusal decided before any provider produced a point."""
    return _Outcome(
        None, "candidate", "none", -1, status, np.zeros(rows), "none", None, None, None
    )


def _minimum_norm_outcome(
    result: LinearSolveResult,
    values: Array,
    equilibration: Array,
    incompatible: MeshfreeCorrectionStatus,
    /,
) -> _Outcome:
    """Classify one native minimum-norm projection without reinterpreting failure.

    Only a certified native incompatibility becomes a witness refusal;
    nonconvergence stays an unresolved provider failure.
    """
    status = int(np.asarray(result.status))
    rows = equilibration.size
    if status == int(LinearSolveStatus.SUCCESS):
        return _Outcome(
            np.asarray(values),
            "projection",
            "minimum-norm",
            status,
            None,
            np.zeros(rows),
            "none",
            result,
            None,
            None,
        )
    evidence = result.minimum_norm
    if (
        status == int(LinearSolveStatus.INCOMPATIBLE_RHS)
        and evidence is not None
        and bool(np.asarray(evidence.incompatible))
    ):
        # (E C S)^T y = 0 gives C^T (E y) = 0: E y is the raw-row witness.
        witness = np.asarray(evidence.left_null_witness) * np.asarray(equilibration)
        return _Outcome(
            None,
            "candidate",
            "minimum-norm",
            status,
            incompatible,
            witness,
            "left-null",
            result,
            None,
            None,
        )
    return _Outcome(
        None,
        "candidate",
        "minimum-norm",
        status,
        MeshfreeCorrectionStatus.PROVIDER_UNRESOLVED,
        np.zeros(rows),
        "none",
        result,
        None,
        None,
    )


def _positive(minimum: float, policy: MeshfreeMetricCorrectionPolicy, /) -> bool:
    """Strictly positive weights at or above the declared margin within tolerance."""
    return minimum - policy.margin >= -policy.tolerance and minimum > 0


def _positive_edge_vector(value: ArrayLike, size: int, name: str, /) -> Array:
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have one value per compact canonical edge.")
    host = np.asarray(array)
    if not np.all(np.isfinite(host) & (host > 0)):
        raise ValueError(f"{name} must be finite and strictly positive.")
    return array


def _edge_vector(value: ArrayLike, size: int, name: str, /) -> Array:
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have one value per compact canonical edge.")
    return array


def _route_arrays(
    operator: SparseCoordinateOperator, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valid ``(row, column, coefficient)`` routes of one edge-list operator."""
    relation = operator.relation
    if not isinstance(relation, EdgeRelation):
        raise TypeError("Correction constraints require native edge-list coordinates.")
    valid = np.asarray(relation.valid)
    return (
        np.asarray(relation.target_indices)[valid].astype(np.int32),
        np.asarray(relation.source_indices)[valid].astype(np.int32),
        np.asarray(operator.coefficients, dtype=np.float64)[valid],
    )


@final
class _MinimumNormProjector(StrictModule):
    """Native weighted projection ``c -> c + S B^+ E (t - C c)`` with ``B = E C S``.

    ``E_i = 1 / |C_i S|`` (one on zero rows) is a positive left scaling, so it
    preserves the affine set and the minimum-norm solution. The candidate
    enters only through the right-hand side of a fixed operator, so the native
    ``rhs-only`` implicit derivative is the exact derivative of the affine
    projection: the weighted projector ``I - S B^+ B S^-1``.
    """

    __strict_contract__ = True
    design: SparseCoordinateOperator
    root: Float64[_CorrectionEdgeDim]
    equilibration: Float64[_CorrectionRowDim]
    linear: PreparedLinearSolve

    def apply(
        self, candidate: Array, target: Array, /
    ) -> tuple[Array, LinearSolveResult]:
        rhs = self.equilibration * (target - self.design.mv(candidate))
        result = solve(self.linear, rhs)
        return candidate + self.root * result.value, result


def _prepare_projector(
    design: SparseCoordinateOperator,
    root: np.ndarray,
    identifier: str,
    tolerance: float,
    maximum_steps: int,
    /,
) -> _MinimumNormProjector:
    """Equilibrated ``E C S`` and its prepared native minimum-norm solve."""
    rows, columns, values = _route_arrays(design)
    row_count = design.target.size
    scaled = values * root[columns]
    norms = np.sqrt(np.bincount(rows, weights=scaled * scaled, minlength=row_count))
    equilibration = np.where(norms > 0, 1.0 / np.where(norms > 0, norms, 1.0), 1.0)
    edges = root.size
    operator = SparseCoordinateOperator(
        EdgeRelation(columns, rows, source_size=edges, target_size=row_count),
        jnp.asarray(equilibration[rows] * scaled, dtype=jnp.float64),
        source=ArraySpace((edges,), dtype=np.float64, space_id=f"{identifier}:edges"),
        target=ArraySpace((row_count,), dtype=np.float64, space_id=f"{identifier}:rows"),
        operator_id=f"{identifier}:equilibrated-design",
    )
    # The absolute equilibrated residual bound gives |C_i w - t_i| <= tol' |C_i S|;
    # tol' = tol / 10 covers the ratio of the row norm |C_i S| to the audit
    # scale. Stationarity is verified against the same absolute bound.
    policy = LinearSolvePolicy(
        GeneralizedLSMR(),
        tolerance=TolerancePolicy(
            relative=0.0, absolute=0.1 * tolerance, max_steps=maximum_steps
        ),
        differentiation=DifferentiationPolicy("rhs-only"),
        derivative_solve=LinearDerivativeSolvePolicy(maximum_steps=maximum_steps),
        failure=FailurePolicy("status"),
    )
    prepared = prepare(
        MinimumNormProblem(operator, problem_id=f"{identifier}:minimum-norm"), policy
    )
    return _MinimumNormProjector(
        design, jnp.asarray(root), jnp.asarray(equilibration), prepared
    )


def _witness_audit(
    kind: MeshfreeCorrectionWitness,
    witness: np.ndarray,
    image: np.ndarray,
    rhs_pairing: float,
    lower_pairing: float,
    row_scale: np.ndarray,
    coefficient_scale: float,
    tolerance: float,
    /,
) -> _WitnessAudit:
    """Independent check that a witness excludes every admissible vector.

    In normalized rows ``y' = row_scale * y`` the declared tolerance acts on
    ``|y'|_1``. A left-null witness needs ``A^T y = 0`` and ``|<y, b>|`` above
    that floor. A Farkas witness for ``{A w = b, w >= m}`` needs ``A^T y >= 0``
    and ``<y, b> < m sum(A^T y)``: then every such ``w`` would give
    ``<y, b> = (A^T y)^T w >= m sum(A^T y)``.
    """
    normalized = witness * row_scale
    bound = (
        tolerance
        * coefficient_scale
        * max(float(np.max(np.abs(normalized), initial=0.0)), 1.0)
    )
    floor = tolerance * float(np.sum(np.abs(normalized)))
    match kind:
        case "none":
            return _WitnessAudit(0.0, 0.0, False)
        case "left-null":
            residual = float(np.max(np.abs(image), initial=0.0))
            margin = abs(rhs_pairing)
        case "farkas":
            residual = float(max(-np.min(image, initial=0.0), 0.0))
            margin = lower_pairing - rhs_pairing
        case unknown:
            assert_never(unknown)
    return _WitnessAudit(
        residual, margin, residual <= bound and margin > floor + residual
    )


@final
class MeshfreeMetricCorrectionPlan(StrictModule):
    """Prepared full-moment correction of one exterior metric.

    ``norm_weights`` is the diagonal ``Phi`` of the declared correction norm
    (the exterior metric prior by default). Preparation is host-only: the
    equilibrated design, its native minimum-norm solve, the sparse row-rank
    presolve and the conic template are built once and reused per candidate.
    """

    __strict_contract__ = True
    exterior: PreparedMeshfreeExteriorCalculus
    policy: MeshfreeMetricCorrectionPolicy
    norm_weights: Float64[_CorrectionEdgeDim]
    projector: _MinimumNormProjector
    row_profile: SparseRowRankEvidence
    conic_rows: Int32[_CorrectionSelectedRowDim]
    conic_prepared: PreparedConvexProgram

    @checked
    def __init__(
        self,
        exterior: PreparedMeshfreeExteriorCalculus,
        /,
        *,
        policy: MeshfreeMetricCorrectionPolicy,
        norm_weights: ArrayLike | None = None,
    ) -> None:
        if exterior.coercivity_policy.assessment == "unassessed":
            raise ValueError(
                "Metric corrections audit coercivity natively; prepare the exterior "
                "with MeshfreeCoercivityPolicy('sparse-cholesky')."
            )
        system = exterior.metric_system
        edges = system.prior.size
        phi = _positive_edge_vector(
            system.prior if norm_weights is None else norm_weights, edges, "norm_weights"
        )
        rows, columns, values = _route_arrays(system.constraint)
        row_count = system.rhs.size
        identifier = f"{system.constraint.operator_id}:learned-correction"
        projector = _prepare_projector(
            system.constraint,
            np.sqrt(np.asarray(phi)),
            identifier,
            policy.tolerance,
            policy.maximum_steps,
        )
        scaling = np.asarray(system.row_scaling)
        normalized = values / scaling[rows]
        normalized_design = SparseCoordinateOperator(
            EdgeRelation(columns, rows, source_size=edges, target_size=row_count),
            jnp.asarray(normalized),
            source=system.constraint.source,
            target=system.constraint.target,
            operator_id=f"{identifier}:normalized-design",
        )
        # An interior point needs independent equalities: the tolerance-defined
        # pivot profile presolves dependent rows; every original row is audited.
        profile = prepare_sparse_row_rank(
            normalized_design, SparseRowRankPolicy(policy.rank.rank)
        )
        selected = np.asarray(profile.selected_rows, dtype=np.int32)
        if selected.size == 0:
            raise ValueError(
                "Moment design has zero rank; no metric correction is definable."
            )
        self.exterior = exterior
        self.policy = policy
        self.norm_weights = phi
        self.projector = projector
        self.row_profile = profile
        self.conic_rows = jnp.asarray(selected)
        self.conic_prepared = _conic_template(
            rows,
            columns,
            normalized,
            selected,
            np.asarray(system.rhs) / scaling,
            np.asarray(phi),
            policy,
            identifier,
        )

    def _candidate(self, candidate: ArrayLike, /) -> Array:
        return _edge_vector(candidate, self.norm_weights.size, "candidate")

    def project(self, candidate: ArrayLike, /) -> MeshfreeMetricProjection:
        """Signed weighted projection; traceable, differentiable in the candidate."""
        value = self._candidate(candidate)
        weights, result = self.projector.apply(value, self.exterior.metric_system.rhs)
        return MeshfreeMetricProjection(
            weights, weights - value, jnp.min(weights) - self.policy.margin, result
        )

    def correct(self, candidate: ArrayLike, /) -> MeshfreeMetricCorrection:
        """Eager host boundary: project, dispatch the sign policy, audit, publish."""
        value = self._candidate(candidate)
        host = np.asarray(value)
        if not np.all(np.isfinite(host)):
            refusal = _refusal(
                MeshfreeCorrectionStatus.NONFINITE_CANDIDATE,
                self.projector.equilibration.size,
            )
            return self._assemble(value, refusal, -1, float("nan"))
        signed_status, signed_minimum = -1, float("nan")
        match self.policy.route:
            case "signed-minimum-norm":
                projection = self.project(value)
                outcome = _minimum_norm_outcome(
                    projection.linear,
                    projection.weights,
                    self.projector.equilibration,
                    MeshfreeCorrectionStatus.MOMENT_INCOMPATIBLE,
                )
                signed_status = outcome.provider_status
                if outcome.values is not None:
                    signed_minimum = float(np.min(outcome.values, initial=np.inf))
                    outcome = self._sign_response(value, outcome, signed_minimum)
            case "nonnegative-conic":
                outcome = self._conic(value)
            case unknown:
                assert_never(unknown)
        return self._assemble(value, outcome, signed_status, signed_minimum)

    @checked
    def tangent(
        self, correction: MeshfreeMetricCorrection, candidate_tangent: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Weight tangent of an admitted correction and its availability flag.

        The signed route is the native ``rhs-only`` JVP of :meth:`project`, the
        weighted projector. The conic route applies
        the native strict-active projection-KKT JVP to the linear-term tangent
        ``-v / Phi``; it is NaN wherever that derivative is unavailable.
        """
        direction = self._candidate(candidate_tangent)
        if not correction.admitted:
            raise ValueError(
                f"Correction status {correction.status.name} publishes no derivative."
            )
        match correction.evidence.provider:
            case "minimum-norm":

                def projected(value: Array) -> tuple[Array, Array]:
                    projection = self.project(value)
                    return projection.weights, projection.linear.derivative_valid

                _, tangent, valid = jax.jvp(
                    projected, (correction.candidate,), (direction,), has_aux=True
                )
                return tangent, valid
            case "conic":
                sensitivity = correction.sensitivity
                if sensitivity is None:
                    raise ValueError(
                        "Admitted conic corrections retain their sensitivity."
                    )
                program = self.conic_prepared.program
                if not isinstance(program, ConicProgram):
                    raise TypeError("Conic corrections require a prepared ConicProgram.")
                tangent = eqx.tree_at(
                    lambda item: item.linear,
                    ConicProgramData.zeros_like(program),
                    -direction / self.norm_weights,
                )
                result = conic_primal_jvp(sensitivity, tangent)
                return jnp.asarray(result.value)[: direction.size], result.available
            case "none":
                raise ValueError("A correction without provider publishes no derivative.")
            case unknown:
                assert_never(unknown)

    def _sign_response(
        self, candidate: Array, outcome: _Outcome, minimum: float, /
    ) -> _Outcome:
        if _positive(minimum, self.policy):
            return outcome
        match self.policy.sign:
            case "refuse":
                return outcome
            case "constrained":
                return self._conic(candidate)
            case unknown:
                assert_never(unknown)

    def _conic(self, candidate: Array, /) -> _Outcome:
        program = eqx.tree_at(
            lambda item: item.linear,
            self.conic_prepared.program,
            -candidate / self.norm_weights,
        )
        prepared = refresh_convex_program(self.conic_prepared, program)
        execution = solve_prepared_convex_program(prepared)
        result = execution.result
        status = int(np.asarray(result.status))
        rows = self.projector.equilibration.size
        if bool(np.asarray(result.successful)):
            return _Outcome(
                np.asarray(result.primal),
                "conic-solution",
                "conic",
                status,
                None,
                np.zeros(rows),
                "none",
                None,
                execution,
                prepared,
            )
        if status == int(ConvexProgramStatus.PRIMAL_INFEASIBLE) and bool(
            np.asarray(result.certificate.dual_ray_valid)
        ):
            selected = np.asarray(self.conic_rows)
            scaling = np.asarray(self.exterior.metric_system.row_scaling)
            ray = np.asarray(result.certificate.inequality_dual_ray)[: selected.size]
            witness = np.zeros(rows)
            # Equalities were posed in rows D^-1 A: raw witness is y / D.
            witness[selected] = ray / scaling[selected]
            peak = float(np.max(np.abs(witness * scaling), initial=0.0))
            return _Outcome(
                None,
                "candidate",
                "conic",
                status,
                MeshfreeCorrectionStatus.INFEASIBLE,
                witness / max(peak, np.finfo(np.float64).tiny),
                "farkas",
                None,
                execution,
                prepared,
            )
        return _Outcome(
            None,
            "candidate",
            "conic",
            status,
            MeshfreeCorrectionStatus.PROVIDER_UNRESOLVED,
            np.zeros(rows),
            "none",
            None,
            execution,
            prepared,
        )

    def _assemble(
        self,
        candidate: Array,
        outcome: _Outcome,
        signed_status: int,
        signed_minimum: float,
        /,
    ) -> MeshfreeMetricCorrection:
        system = self.exterior.metric_system
        host_candidate = np.asarray(candidate)
        subject = host_candidate if outcome.values is None else outcome.values
        finite = bool(np.all(np.isfinite(subject)))
        residual = np.asarray(system.constraint.mv(jnp.asarray(subject))) - np.asarray(
            system.rhs
        )
        scale = np.asarray(system.row_scaling) + np.abs(np.asarray(system.rhs))
        normalized = residual / scale
        maximum = float(np.max(np.abs(normalized), initial=0.0))
        moments_exact = finite and maximum <= self.policy.tolerance
        minimum = float(np.min(subject, initial=np.inf))
        sign_margin = minimum - self.policy.margin
        positive = finite and _positive(minimum, self.policy)
        stiffness = (
            self.exterior.stiffness_evidence(jnp.asarray(subject)) if finite else None
        )
        coercive = (
            stiffness is not None
            and bool(np.asarray(stiffness.spd))
            and bool(np.asarray(stiffness.anchored_components))
        )
        delta = subject - host_candidate
        norm = float(np.sqrt(np.sum(delta * delta / np.asarray(self.norm_weights))))
        witness = self._audit_witness(outcome)
        status = _status(
            outcome.refusal,
            outcome.witness_kind,
            witness.valid,
            moments_exact,
            positive,
            coercive,
        )
        derivative, available, sensitivity_status, sensitivity = self._derivative(
            status, outcome
        )
        evidence = MeshfreeMetricCorrectionEvidence(
            audited_weights=jnp.asarray(subject),
            moment_residual=jnp.asarray(residual),
            normalized_moment_residual=jnp.asarray(normalized),
            witness=jnp.asarray(outcome.witness),
            stiffness=stiffness,
            status=status,
            route=self.policy.route,
            provider=outcome.provider,
            provider_status=outcome.provider_status,
            audited_subject=outcome.subject,
            maximum_moment_residual=maximum,
            moments_exact=moments_exact,
            minimum_weight=minimum,
            sign_margin=sign_margin,
            positive=positive,
            coercive=coercive,
            correction_norm=norm,
            witness_kind=outcome.witness_kind,
            witness_residual=witness.residual,
            witness_margin=witness.margin,
            witness_valid=witness.valid,
            signed_provider_status=signed_status,
            signed_minimum_weight=signed_minimum,
            derivative=derivative,
            derivative_available=available,
            sensitivity_status=sensitivity_status,
            margin=self.policy.margin,
            tolerance=self.policy.tolerance,
        )
        admitted = status is MeshfreeCorrectionStatus.ADMITTED
        return MeshfreeMetricCorrection(
            evidence=evidence,
            candidate=candidate,
            weights=jnp.asarray(subject) if admitted else None,
            linear=outcome.linear,
            conic=outcome.execution,
            sensitivity=sensitivity,
        )

    def _audit_witness(self, outcome: _Outcome, /) -> _WitnessAudit:
        system = self.exterior.metric_system
        witness = outcome.witness
        image = np.asarray(system.constraint.transpose_mv(jnp.asarray(witness)))
        scaling = np.asarray(system.row_scaling)
        rhs = np.asarray(system.rhs)
        rows, _, coefficients = _route_arrays(system.constraint)
        return _witness_audit(
            outcome.witness_kind,
            witness,
            image,
            float(rhs @ witness),
            self.policy.margin * float(np.sum(image)),
            scaling,
            float(np.max(np.abs(coefficients) / scaling[rows], initial=0.0)),
            self.policy.tolerance,
        )

    def _derivative(
        self, status: MeshfreeCorrectionStatus, outcome: _Outcome, /
    ) -> tuple[
        MeshfreeCorrectionDerivative, bool, int, PreparedMatrixFreeConicSensitivity | None
    ]:
        if status is not MeshfreeCorrectionStatus.ADMITTED:
            return "unavailable", False, -1, None
        match outcome.provider:
            case "minimum-norm":
                linear = outcome.linear
                if linear is None:
                    raise RuntimeError("Signed corrections retain their native solve.")
                valid = bool(np.asarray(linear.derivative_valid))
                return "minimum-norm-projector", valid, -1, None
            case "conic":
                if outcome.execution is None or outcome.program is None:
                    raise RuntimeError("Conic corrections retain their native execution.")
                sensitivity = prepare_conic_sensitivity(
                    outcome.program,
                    outcome.execution,
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
                )
                if not isinstance(sensitivity, PreparedMatrixFreeConicSensitivity):
                    raise TypeError(
                        "Sparse correction programs bind matrix-free sensitivities."
                    )
                code = int(np.asarray(sensitivity.active_set.status))
                regular = code == int(ConicSensitivityStatus.REGULAR_FIXED_ACTIVE)
                return "conic-fixed-active-set", regular, code, sensitivity
            case "none":
                return "unavailable", False, -1, None
            case unknown:
                assert_never(unknown)


def _status(
    refusal: MeshfreeCorrectionStatus | None,
    witness_kind: MeshfreeCorrectionWitness,
    witness_valid: bool,
    moments_exact: bool,
    positive: bool,
    coercive: bool,
    /,
) -> MeshfreeCorrectionStatus:
    """Provider refusals first; a claimed infeasibility needs an audited witness."""
    if refusal is not None:
        match witness_kind:
            case "none":
                return refusal
            case "left-null" | "farkas":
                return refusal if witness_valid else MeshfreeCorrectionStatus.AUDIT_FAILED
            case unknown:
                assert_never(unknown)
    if not moments_exact:
        return MeshfreeCorrectionStatus.AUDIT_FAILED
    if not positive:
        return MeshfreeCorrectionStatus.POSITIVITY_CONFLICT
    if not coercive:
        return MeshfreeCorrectionStatus.COERCIVITY_CONFLICT
    return MeshfreeCorrectionStatus.ADMITTED


def _conic_template(
    rows: np.ndarray,
    columns: np.ndarray,
    normalized: np.ndarray,
    selected: np.ndarray,
    normalized_rhs: np.ndarray,
    phi: np.ndarray,
    policy: MeshfreeMetricCorrectionPolicy,
    identifier: str,
    /,
) -> PreparedConvexProgram:
    """``min 0.5 w^T Phi^-1 w - (Phi^-1 w_c)^T w`` s.t. selected ``D^-1 A w = D^-1 b``
    and ``-w + s = -margin``, ``s >= 0``; the candidate only refreshes the
    linear term."""
    edges = phi.size
    inverse = np.full(normalized_rhs.size, -1, dtype=np.int32)
    inverse[selected] = np.arange(selected.size, dtype=np.int32)
    keep = inverse[rows] >= 0
    diagonal = np.arange(edges, dtype=np.int32)
    equalities = selected.size
    variables = ArraySpace(
        (edges,), dtype=np.float64, space_id=f"{identifier}:conic-edges"
    )
    quadratic = SparseCoordinateOperator(
        EdgeRelation(diagonal, diagonal, source_size=edges, target_size=edges),
        jnp.asarray(1.0 / phi),
        source=variables,
        target=variables,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    constraint = SparseCoordinateOperator(
        EdgeRelation(
            np.concatenate((columns[keep], diagonal)),
            np.concatenate((inverse[rows[keep]], equalities + diagonal)),
            source_size=edges,
            target_size=equalities + edges,
        ),
        jnp.asarray(np.concatenate((normalized[keep], -np.ones(edges)))),
        source=variables,
        target=ArraySpace(
            (equalities + edges,), dtype=np.float64, space_id=f"{identifier}:conic-rows"
        ),
    )
    program = ConicProgram(
        quadratic,
        jnp.zeros((edges,), dtype=jnp.float64),
        constraint,
        jnp.asarray(
            np.concatenate((normalized_rhs[selected], np.full(edges, -policy.margin)))
        ),
        ProductCone((ZeroCone(equalities), NonnegativeCone(edges))),
        problem_id=f"{identifier}:positive-margin",
        convexity_evidence="construction",
    )
    return prepare_convex_program(program, policy.conic)


@final
class MeshfreeEdgeFluxCorrectionPlan(StrictModule):
    """Prepared parity audit and nodal-balance projection of oriented fluxes.

    ``G`` is the transpose incidence restricted to equation (non-Dirichlet)
    nodes: ``(G F)_i = sum_e sign(i, e) F_e`` with ``-1`` at ``pairs[e, 0]`` and
    ``+1`` at ``pairs[e, 1]``. ``norm_weights`` declares the correction norm
    ``|F - F_c|_{W^-1}`` (Euclidean by default).
    """

    __strict_contract__ = True
    exterior: PreparedMeshfreeExteriorCalculus
    norm_weights: Float64[_CorrectionEdgeDim]
    balance_operator: SparseCoordinateOperator
    projector: _MinimumNormProjector
    tolerance: float = eqx.field(static=True)

    @checked
    def __init__(
        self,
        exterior: PreparedMeshfreeExteriorCalculus,
        /,
        *,
        norm_weights: ArrayLike | None = None,
        tolerance: float = 1e-9,
        maximum_steps: int = 4096,
    ) -> None:
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise TypeError("tolerance must be a real number.")
        if not isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Correction tolerance must be finite and positive.")
        if isinstance(maximum_steps, bool) or not isinstance(maximum_steps, int):
            raise TypeError("maximum_steps must be an integer.")
        if maximum_steps < 1:
            raise ValueError("maximum_steps must be positive.")
        edges = exterior.lengths.size
        weights = _positive_edge_vector(
            np.ones(edges) if norm_weights is None else norm_weights,
            edges,
            "norm_weights",
        )
        node_rows = np.full(exterior.points.shape[0], -1, dtype=np.int32)
        equations = np.asarray(exterior.equation_indices)
        if equations.size == 0:
            raise ValueError("Flux balance needs at least one equation node.")
        node_rows[equations] = np.arange(equations.size, dtype=np.int32)
        edge_index, nodes, signs = _route_arrays(exterior.incidence)
        # Incidence maps nodes to edges; its transpose maps edges to node balances.
        rows = node_rows[nodes]
        keep = rows >= 0
        identifier = f"{exterior.incidence.operator_id}:flux-balance"
        balance = SparseCoordinateOperator(
            EdgeRelation(
                edge_index[keep],
                rows[keep],
                source_size=edges,
                target_size=equations.size,
            ),
            jnp.asarray(signs[keep]),
            source=ArraySpace((edges,), dtype=np.float64, space_id=f"{identifier}:edges"),
            target=ArraySpace(
                (equations.size,), dtype=np.float64, space_id=f"{identifier}:nodes"
            ),
            operator_id=identifier,
        )
        self.exterior = exterior
        self.norm_weights = weights
        self.balance_operator = balance
        self.projector = _prepare_projector(
            balance,
            np.sqrt(np.asarray(weights)),
            identifier,
            float(tolerance),
            maximum_steps,
        )
        self.tolerance = float(tolerance)

    def _balance(self, balance: ArrayLike, /) -> Array:
        value = jnp.asarray(balance, dtype=jnp.float64)
        if value.shape != (self.exterior.points.shape[0],):
            raise ValueError("Flux balance must have one value per compact node.")
        return value[self.exterior.equation_indices]

    @checked
    def project(
        self, candidate: MeshfreeEdgeFluxCandidate, balance: ArrayLike, /
    ) -> MeshfreeEdgeFluxProjection:
        """Balance projection of the forward flux; traceable and differentiable in
        the candidate (the declared balance is constraint data)."""
        forward = _edge_vector(candidate.forward, self.norm_weights.size, "candidate")
        flux, result = self.projector.apply(forward, self._balance(balance))
        return MeshfreeEdgeFluxProjection(
            flux,
            flux - forward,
            jnp.max(jnp.abs(candidate.forward + candidate.reverse)),
            result,
        )

    @checked
    def correct(
        self, candidate: MeshfreeEdgeFluxCandidate, balance: ArrayLike, /
    ) -> MeshfreeEdgeFluxCorrection:
        """Eager host boundary: parity audit, balance projection, independent audits."""
        forward = np.asarray(
            _edge_vector(candidate.forward, self.norm_weights.size, "candidate")
        )
        reverse = np.asarray(candidate.reverse)
        target = np.asarray(self._balance(balance))
        rows = self.projector.equilibration.size
        parity = forward + reverse
        parity_scale = max(float(np.max(np.abs(forward), initial=0.0)), 1.0)
        parity_maximum = float(np.max(np.abs(parity), initial=0.0))
        finite = bool(np.all(np.isfinite(forward) & np.isfinite(reverse)))
        antisymmetric = finite and parity_maximum <= self.tolerance * parity_scale
        if not finite:
            outcome = _refusal(MeshfreeCorrectionStatus.NONFINITE_CANDIDATE, rows)
        elif not antisymmetric:
            # A non-antisymmetric learned flux is not a conservative oriented
            # flux; it is refused, never antisymmetrized.
            outcome = _refusal(MeshfreeCorrectionStatus.PARITY_CONFLICT, rows)
        else:
            projection = self.project(candidate, balance)
            outcome = _minimum_norm_outcome(
                projection.linear,
                projection.flux,
                self.projector.equilibration,
                MeshfreeCorrectionStatus.BALANCE_INCOMPATIBLE,
            )
        return self._assemble(
            forward, parity, parity_maximum, antisymmetric, target, outcome
        )

    def _assemble(
        self,
        forward: np.ndarray,
        parity: np.ndarray,
        parity_maximum: float,
        antisymmetric: bool,
        target: np.ndarray,
        outcome: _Outcome,
        /,
    ) -> MeshfreeEdgeFluxCorrection:
        subject = forward if outcome.values is None else outcome.values
        finite = bool(np.all(np.isfinite(subject)))
        operator = self.balance_operator
        residual = np.asarray(operator.mv(jnp.asarray(subject))) - target
        magnitude = np.asarray(
            eqx.tree_at(
                lambda op: op.coefficients, operator, jnp.abs(operator.coefficients)
            ).mv(jnp.abs(jnp.asarray(subject)))
        ) + np.abs(target)
        normalized = np.where(
            magnitude > 0,
            np.abs(residual) / np.where(magnitude > 0, magnitude, 1.0),
            np.abs(residual),
        )
        maximum = float(np.max(normalized, initial=0.0))
        balanced = finite and maximum <= self.tolerance
        delta = subject - forward
        norm = float(np.sqrt(np.sum(delta * delta / np.asarray(self.norm_weights))))
        image = np.asarray(operator.transpose_mv(jnp.asarray(outcome.witness)))
        witness = _witness_audit(
            outcome.witness_kind,
            outcome.witness,
            image,
            float(target @ outcome.witness),
            0.0,
            np.ones(target.size),
            1.0,
            self.tolerance,
        )
        # Sign margins and coercivity are metric contracts; oriented fluxes are signed.
        status = _status(
            outcome.refusal, outcome.witness_kind, witness.valid, balanced, True, True
        )
        admitted = status is MeshfreeCorrectionStatus.ADMITTED
        available = (
            admitted
            and outcome.linear is not None
            and bool(np.asarray(outcome.linear.derivative_valid))
        )
        evidence = MeshfreeEdgeFluxCorrectionEvidence(
            audited_flux=jnp.asarray(subject),
            parity_defect=jnp.asarray(parity),
            balance_residual=jnp.asarray(residual),
            witness=jnp.asarray(outcome.witness),
            status=status,
            provider=outcome.provider,
            provider_status=outcome.provider_status,
            audited_subject=outcome.subject,
            maximum_parity_defect=parity_maximum,
            antisymmetric=antisymmetric,
            maximum_balance_residual=maximum,
            balanced=balanced,
            correction_norm=norm,
            witness_kind=outcome.witness_kind,
            witness_residual=witness.residual,
            witness_margin=witness.margin,
            witness_valid=witness.valid,
            derivative_available=available,
            tolerance=self.tolerance,
        )
        return MeshfreeEdgeFluxCorrection(
            evidence=evidence,
            flux=jnp.asarray(subject) if admitted else None,
            linear=outcome.linear,
        )


__all__ = [
    "MeshfreeCorrectionDerivative",
    "MeshfreeCorrectionProvider",
    "MeshfreeCorrectionRoute",
    "MeshfreeCorrectionSignPolicy",
    "MeshfreeCorrectionStatus",
    "MeshfreeCorrectionSubject",
    "MeshfreeCorrectionWitness",
    "MeshfreeEdgeFluxCandidate",
    "MeshfreeEdgeFluxCorrection",
    "MeshfreeEdgeFluxCorrectionEvidence",
    "MeshfreeEdgeFluxCorrectionPlan",
    "MeshfreeEdgeFluxProjection",
    "MeshfreeMetricCorrection",
    "MeshfreeMetricCorrectionEvidence",
    "MeshfreeMetricCorrectionPlan",
    "MeshfreeMetricCorrectionPolicy",
    "MeshfreeMetricProjection",
]
