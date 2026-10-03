# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Point-support concentration transfer with declared constraints and audits.

A transfer ``T`` maps source concentrations to target concentrations on a
fixed sparse route set. Every request is conservative,
``w_newᵀ T = w_oldᵀ``; constant reproduction ``T 1 = 1``, polynomial moments
and coefficient nonnegativity are separate declared constraints. The executed
coefficients are the minimum change from the supplied base coefficients
subject to exactly the declared equations: the native rectangular
minimum-norm solve for signed requests and the native sparse conic solve for
nonnegative requests. Every equation, sign and coverage requirement is then
audited on the host, independently of the provider.

If total source and target measure differ, conservation and constant
reproduction are jointly impossible: summing the conservation equations gives
``sum_e m_new[r_e] t_e = sum m_old`` while the target-measure-weighted sum of
the constant equations gives the same left side equal to ``sum m_new``. That
obstruction is refused with its explicit witness before any solve. The same
summation shows that conservation with exact reproduction of ambient
polynomials of degree ``k`` requires both quadratures ``(x, m)`` to integrate
every such polynomial identically. Local support can make an otherwise
globally feasible problem infeasible as well; a certified provider witness
(left-null vector or Farkas ray) is reported separately from an unresolved
provider failure.
"""

from __future__ import annotations

from enum import IntEnum
from typing import assert_never, final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...ein import contract
from ...linalg import (
    ArraySpace,
    DifferentiationPolicy,
    FactorizationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveStatus,
    LSMR,
    MinimumNormProblem,
    OperatorProperties,
    pseudoinverse,
    RankPolicy,
    solve,
    TolerancePolicy,
)
from ...optim import (
    ConicProgram,
    ConvexProgramStatus,
    ConvexSolvePolicy,
    ConvexTermination,
    NativeHomogeneousConic,
    NonnegativeCone,
    ProductCone,
    solve_conic_program,
    ZeroCone,
)
from ...sparse import EdgeRelation, RowRelation, SparseCoordinateOperator
from ...typing import Dim, Float64, Int32, parse
from .._gram import diagonal_gram_space
from .._spaces import DiscreteFieldSpace, TensorDofLayout
from .._topology_epoch import TopologyEpoch, TopologyEpochTransition
from .._transfer import FieldTransfer, TransferProperties
from ._stencils import _basis_exponents, PreparedLocalStencils


PointTransferRoute: TypeAlias = Literal[
    "conservative-signed", "conservative-positive", "joint"
]
"""``conservative-signed``: conservation only, signed coefficients.
``conservative-positive``: conservation with nonnegative coefficients.
``joint``: conservation, constant reproduction and declared moments,
optionally nonnegative."""

PointTransferObjective: TypeAlias = Literal[
    "coefficient-change", "measure-weighted-change"
]
"""Minimized correction ``sum_e d_e (t_e - t0_e)^2``: ``d_e = 1`` or the
target measure of the route's row relative to the mean target measure."""

PointTransferProvider: TypeAlias = Literal["none", "minimum-norm", "conic"]
PointTransferWitness: TypeAlias = Literal[
    "none", "measure-obstruction", "left-null", "farkas"
]


class PointTransferStatus(IntEnum):
    ADMITTED = 0
    UNCOVERED_SOURCE = 1
    UNCOVERED_TARGET = 2
    MEASURE_OBSTRUCTION = 3
    INFEASIBLE = 4
    PROVIDER_UNRESOLVED = 5
    AUDIT_FAILED = 6
    AMPLIFICATION_EXCEEDED = 7


class TransferSourceDim(Dim):
    """Compact source support points."""


class TransferTargetDim(Dim):
    """Compact target support points."""


class TransferRouteDim(Dim):
    """Valid flat transfer routes."""


class TransferIntrinsicDim(Dim):
    """Per-target moment coordinate frame."""


class TransferMomentDim(Dim):
    """Declared nonconstant monomials per target row."""


class TransferEquationDim(Dim):
    """Declared transfer equations."""


@final
class PointTransferRequest(StrictModule):
    """One explicit constraint/objective declaration of a point transfer."""

    route: PointTransferRoute = eqx.field(static=True)
    moment_degree: int = eqx.field(static=True)
    nonnegative: bool = eqx.field(static=True)
    objective: PointTransferObjective = eqx.field(static=True)

    def __init__(
        self,
        route: PointTransferRoute,
        /,
        *,
        moment_degree: int = 0,
        nonnegative: bool = False,
        objective: PointTransferObjective = "coefficient-change",
    ) -> None:
        route_ = parse(route, PointTransferRoute, "route")
        objective_ = parse(objective, PointTransferObjective, "objective")
        if isinstance(moment_degree, bool) or not isinstance(
            moment_degree, (int, np.integer)
        ):
            raise TypeError("moment_degree must be an integer.")
        if not isinstance(nonnegative, bool):
            raise TypeError("nonnegative must be bool.")
        degree = int(moment_degree)
        match route_:
            case "conservative-signed":
                if nonnegative or degree:
                    raise ValueError(
                        "conservative-signed declares conservation only; request "
                        "conservative-positive or joint for further constraints."
                    )
                sign = False
            case "conservative-positive":
                if degree:
                    raise ValueError(
                        "conservative-positive declares no moments; request joint."
                    )
                sign = True
            case "joint":
                if degree < 0:
                    raise ValueError("Joint moment degree must be nonnegative.")
                sign = nonnegative
            case unknown:
                assert_never(unknown)
        self.route = route_
        self.moment_degree = degree
        self.nonnegative = sign
        self.objective = objective_

    @property
    def constant(self) -> bool:
        return self.route == "joint"


@final
class PointTransferEvidence(StrictModule):
    """Host audit of every declared equation, sign, coverage and objective.

    Residuals are measured on the audited coefficients (the base coefficients
    when the request was refused before a solve): conservation relative to each
    source measure, constant reproduction absolute, and moments in each row's
    scaled coordinates. ``witness`` lives in the declared equation rows
    (conservation rows divided by their source measure, then constant rows, then
    moment rows) and is audited against them independently of the provider.

    ``lebesgue_constant`` is ``max_r sum_e |t_e|``, the max-norm amplification
    of the transfer. A degree-``p`` exact transfer errs on a smooth field by at
    most ``(1 + lebesgue_constant) |f|_{p+1} rho^(p+1) / (p+1)!`` with ``rho`` the
    largest route length, so its order needs a bounded Lebesgue constant. The
    minimum-change solve does not bound it: an ill-conditioned joint system on
    sparse routes can force a large correction and amplification.
    """

    __strict_contract__ = True
    request: PointTransferRequest
    coefficients: Float64[TransferRouteDim]
    conservation_residual: Float64[TransferSourceDim]
    constant_residual: Float64[TransferTargetDim]
    moment_residual: Float64[TransferTargetDim, TransferMomentDim]
    witness: Float64[TransferEquationDim]
    status: PointTransferStatus = eqx.field(static=True)
    provider: PointTransferProvider = eqx.field(static=True)
    provider_status: int = eqx.field(static=True)
    witness_kind: PointTransferWitness = eqx.field(static=True)
    witness_residual: float = eqx.field(static=True)
    witness_margin: float = eqx.field(static=True)
    obstruction_defect: float = eqx.field(static=True)
    correction_norm: float = eqx.field(static=True)
    objective_value: float = eqx.field(static=True)
    minimum_coefficient: float = eqx.field(static=True)
    lebesgue_constant: float = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    constant_preserving: bool = eqx.field(static=True)
    nonnegative: bool = eqx.field(static=True)
    moments_exact: bool = eqx.field(static=True)
    uncovered_sources: tuple[int, ...] = eqx.field(static=True)
    uncovered_targets: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    @property
    def admitted(self) -> bool:
        return self.status is PointTransferStatus.ADMITTED


@final
class PreparedPointTransfer(StrictModule):
    """Audited transfer; ``transfer`` exists only when the request was admitted.

    Concentration values are differentiable through the frozen transfer: the
    primal action is linear, its coordinate dual is ``dual_pullback_operator``
    and its measure-weighted Hilbert adjoint is ``hilbert_adjoint_operator``.
    The route selection and geometry are frozen and carry no derivative.
    """

    __strict_contract__ = True
    evidence: PointTransferEvidence
    transfer: FieldTransfer | None
    relation: EdgeRelation
    source_measures: Float64[TransferSourceDim]
    target_measures: Float64[TransferTargetDim]

    @property
    def admitted(self) -> bool:
        return self.evidence.admitted

    def _admitted(self) -> FieldTransfer:
        if self.transfer is None:
            raise ValueError(
                "Point transfer was refused with status "
                f"{self.evidence.status.name}; inspect its evidence."
            )
        return self.transfer

    def apply(self, concentrations: ArrayLike, /) -> Array:
        return self._admitted().primal_operator.mv(jnp.asarray(concentrations))

    def apply_content(self, content: ArrayLike, /) -> Array:
        """Convert extensive content to concentration before applying T."""
        return self.target_measures * self.apply(
            jnp.asarray(content) / self.source_measures
        )

    def epoch_transition(
        self, source: TopologyEpoch, target: TopologyEpoch, /
    ) -> TopologyEpochTransition:
        transfer = self._admitted()
        defect = np.abs(
            np.asarray(self.evidence.conservation_residual)
            * np.asarray(self.source_measures)
        )
        return TopologyEpochTransition(
            source,
            target,
            transfer,
            self.source_measures,
            self.target_measures,
            measure_defect_bound=defect,
        )


class TransferEntryDim(Dim):
    """Nonzero entries of the declared equation matrix."""


@final
class _Routes(StrictModule):
    """Compact valid routes in deterministic relation order."""

    __strict_contract__ = True
    rows: Int32[TransferRouteDim]
    columns: Int32[TransferRouteDim]
    base: Float64[TransferRouteDim]
    coordinates: Float64[TransferRouteDim, TransferIntrinsicDim] | None


@final
class _System(StrictModule):
    """Declared equations ``B t = b`` in normalized rows."""

    __strict_contract__ = True
    route_indices: Int32[TransferEntryDim]
    equation_indices: Int32[TransferEntryDim]
    values: Float64[TransferEntryDim]
    rhs: Float64[TransferEquationDim]
    exponents: tuple[tuple[int, ...], ...] = eqx.field(static=True)


@final
class _Solution(StrictModule):
    __strict_contract__ = True
    values: Float64[TransferRouteDim]
    witness: Float64[TransferEquationDim]
    status: PointTransferStatus = eqx.field(static=True)
    provider: PointTransferProvider = eqx.field(static=True)
    provider_status: int = eqx.field(static=True)
    witness_kind: PointTransferWitness = eqx.field(static=True)


def _outcome(
    values: ArrayLike,
    status: PointTransferStatus,
    provider: PointTransferProvider,
    provider_status: int,
    /,
    *,
    witness: ArrayLike | None = None,
    witness_kind: PointTransferWitness = "none",
) -> _Solution:
    certificate = (
        np.zeros((0,), dtype=np.float64)
        if witness is None
        else np.asarray(witness, dtype=np.float64)
    )
    return _Solution(
        jnp.asarray(np.asarray(values, dtype=np.float64)),
        jnp.asarray(certificate),
        status,
        provider,
        provider_status,
        witness_kind,
    )


def _flat_routes(
    relation: EdgeRelation | RowRelation,
    coefficients: ArrayLike,
    offsets: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int]:
    if not isinstance(relation, (EdgeRelation, RowRelation)):
        raise TypeError("Point transfer needs a native sparse relation.")
    if isinstance(relation, RowRelation):
        if relation.case_shape or len(relation.target_shape) != 1:
            raise ValueError(
                "Point transfers require one unbatched compact target vector."
            )
        target_size = relation.targets_per_case
        edge = relation.as_edge_relation()
    else:
        target_size = relation.target_size
        edge = relation
    base = np.asarray(coefficients, dtype=np.float64)
    if base.shape != relation.route_shape or not np.all(np.isfinite(base)):
        raise ValueError("Transfer coefficients must be finite and match routes.")
    valid = np.asarray(edge.valid).reshape(-1)
    rows = np.asarray(edge.target_indices).reshape(-1)[valid]
    columns = np.asarray(edge.source_indices).reshape(-1)[valid]
    shift: np.ndarray | None = None
    if offsets is not None:
        raw = np.asarray(offsets, dtype=np.float64)
        if (
            raw.ndim != len(relation.route_shape) + 1
            or raw.shape[:-1] != relation.route_shape
            or not np.all(np.isfinite(raw.reshape(-1, raw.shape[-1])[valid]))
        ):
            raise ValueError(
                "Route offsets must be finite (routes..., dimension) source-minus-target."
            )
        shift = raw.reshape(-1, raw.shape[-1])[valid]
    return rows, columns, base.reshape(-1)[valid], shift, target_size


def _measure(value: ArrayLike, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array) & (array > 0)):
        raise ValueError(f"{name} must be positive, finite, and match the relation.")
    return array


@final
class PointTransferPlan(StrictModule):
    """Host preparation of one declared point transfer on fixed routes.

    ``offsets`` are route source-minus-target displacements (minimum image for
    periodic supports) and are required when moments are declared. ``frames``
    optionally project each target row's offsets onto an intrinsic coordinate
    frame ``(targets, intrinsic, ambient)``; bulk supports use ambient
    coordinates. ``lebesgue_bound`` declares the admissible max-norm
    amplification ``max_r sum_e |t_e|``; an audited transfer above it is refused
    with ``AMPLIFICATION_EXCEEDED``, since its accuracy order is then unbounded.
    Preparation is bounded sparse host work and never forms a dense transfer
    matrix.
    """

    __strict_contract__ = True
    routes: _Routes
    source_measures: Float64[TransferSourceDim]
    target_measures: Float64[TransferTargetDim]
    request: PointTransferRequest
    linear_policy: LinearSolvePolicy | None
    conic_policy: ConvexSolvePolicy | None
    tolerance: float = eqx.field(static=True)
    lebesgue_bound: float | None = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation: EdgeRelation | RowRelation,
        coefficients: ArrayLike,
        source_measures: ArrayLike,
        target_measures: ArrayLike,
        /,
        *,
        source_id: str,
        target_id: str,
        request: PointTransferRequest,
        offsets: ArrayLike | None = None,
        frames: ArrayLike | None = None,
        tolerance: float = 1e-10,
        linear_policy: LinearSolvePolicy | None = None,
        conic_policy: ConvexSolvePolicy | None = None,
        lebesgue_bound: float | None = None,
    ) -> None:
        if not isinstance(request, PointTransferRequest):
            raise TypeError("request must be a PointTransferRequest.")
        rows, columns, base, shift, target_size = _flat_routes(
            relation, coefficients, offsets
        )
        old = _measure(source_measures, relation.source_size, "source_measures")
        new = _measure(target_measures, target_size, "target_measures")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Transfer tolerance must be positive and finite.")
        if any(
            not isinstance(value, str) or not value.strip() or value != value.strip()
            for value in (source_id, target_id)
        ):
            raise ValueError(
                "Transfer supports require explicit canonical source_id and target_id."
            )
        if request.moment_degree and shift is None:
            raise ValueError("Declared moments require route offsets.")
        coordinates = shift
        if frames is not None:
            if shift is None:
                raise ValueError("Moment frames require route offsets.")
            frame = np.asarray(frames, dtype=np.float64)
            if (
                frame.ndim != 3
                or frame.shape[0] != target_size
                or frame.shape[2] != shift.shape[1]
                or not np.all(np.isfinite(frame))
            ):
                raise ValueError(
                    "frames must be finite (targets, intrinsic, ambient) row frames."
                )
            # One frame per route row: a bounded per-route matvec.
            coordinates = np.sum(frame[rows] * shift[:, None, :], axis=-1)
        for policy, kind, name in (
            (linear_policy, LinearSolvePolicy, "linear_policy"),
            (conic_policy, ConvexSolvePolicy, "conic_policy"),
        ):
            if policy is not None and not isinstance(policy, kind):
                raise TypeError(f"{name} must be a native {kind.__name__}.")
            if policy is not None and policy.failure.mode != "status":
                raise ValueError(
                    f"{name} must return status so infeasibility evidence is retained."
                )
        if lebesgue_bound is not None and not (
            np.isfinite(lebesgue_bound) and lebesgue_bound >= 1.0
        ):
            raise ValueError(
                "lebesgue_bound must be finite and at least one (constant rows force "
                "row sums of at least one)."
            )
        self.routes = _Routes(
            jnp.asarray(rows, dtype=jnp.int32),
            jnp.asarray(columns, dtype=jnp.int32),
            jnp.asarray(base),
            None if coordinates is None else jnp.asarray(coordinates),
        )
        self.source_measures, self.target_measures = jnp.asarray(old), jnp.asarray(new)
        self.request = request
        self.linear_policy, self.conic_policy = linear_policy, conic_policy
        self.tolerance = float(tolerance)
        self.lebesgue_bound = None if lebesgue_bound is None else float(lebesgue_bound)
        self.source_id, self.target_id = source_id, target_id

    @classmethod
    def from_stencils(
        cls,
        stencils: PreparedLocalStencils,
        source_measures: ArrayLike,
        target_measures: ArrayLike,
        /,
        *,
        request: PointTransferRequest,
        functional_index: int = 0,
        tolerance: float = 1e-10,
        linear_policy: LinearSolvePolicy | None = None,
        conic_policy: ConvexSolvePolicy | None = None,
        lebesgue_bound: float | None = None,
    ) -> PointTransferPlan:
        """Base coefficients and offsets from admitted cross-target stencils."""
        relation, weights, offsets = stencil_routes(stencils, functional_index)
        return cls(
            relation,
            weights,
            source_measures,
            target_measures,
            source_id=f"{stencils.neighborhood.neighborhood_id}:source",
            target_id=f"{stencils.neighborhood.neighborhood_id}:target",
            request=request,
            offsets=offsets,
            tolerance=tolerance,
            linear_policy=linear_policy,
            conic_policy=conic_policy,
            lebesgue_bound=lebesgue_bound,
        )

    def prepare(self) -> PreparedPointTransfer:
        old = np.asarray(self.source_measures)
        new = np.asarray(self.target_measures)
        rows = np.asarray(self.routes.rows)
        columns = np.asarray(self.routes.columns)
        base = np.asarray(self.routes.base)
        system = _system(self.routes, old, new, self.request)
        weights = _objective_weights(rows, new, self.request.objective)
        solution = self._solve(system, base, weights, old, new)
        evidence = _evidence(
            solution, system, base, weights, rows, columns, old, new, self
        )
        relation = EdgeRelation(columns, rows, source_size=old.size, target_size=new.size)
        transfer = (
            _field_transfer(evidence, rows, columns, old, new, self)
            if evidence.admitted
            else None
        )
        return PreparedPointTransfer(
            evidence, transfer, relation, self.source_measures, self.target_measures
        )

    def _solve(
        self,
        system: _System,
        base: np.ndarray,
        weights: np.ndarray,
        old: np.ndarray,
        new: np.ndarray,
    ) -> _Solution:
        rows = np.asarray(self.routes.rows)
        columns = np.asarray(self.routes.columns)
        if np.any(np.bincount(columns, minlength=old.size) == 0):
            return _outcome(base, PointTransferStatus.UNCOVERED_SOURCE, "none", -1)
        if self.request.constant and np.any(np.bincount(rows, minlength=new.size) == 0):
            return _outcome(base, PointTransferStatus.UNCOVERED_TARGET, "none", -1)
        defect = float(np.sum(old) - np.sum(new))
        if self.request.constant and abs(defect) > self.tolerance * float(
            np.sum(old) + np.sum(new)
        ):
            # Conservation rows scaled back by m_old minus the m_new-weighted
            # constant rows annihilate every route; their right sides differ
            # by the total-measure defect.
            witness = np.zeros_like(np.asarray(system.rhs))
            witness[: old.size] = old
            witness[old.size : old.size + new.size] = -new
            return _outcome(
                base,
                PointTransferStatus.MEASURE_OBSTRUCTION,
                "none",
                -1,
                witness=witness,
                witness_kind="measure-obstruction",
            )
        if _feasible(system, base, self.tolerance, self.request.nonnegative):
            # The minimum change of an already feasible base is exactly zero.
            return _outcome(base, PointTransferStatus.ADMITTED, "none", -1)
        if self.request.nonnegative:
            return self._solve_conic(system, base, weights)
        return self._solve_signed(system, base, weights)

    def _identifier(self, kind: str, size: int, /) -> str:
        return canonical_fingerprint(
            {
                "kind": f"point-transfer-{kind}",
                "source": self.source_id,
                "target": self.target_id,
                "request": [
                    self.request.route,
                    self.request.moment_degree,
                    self.request.nonnegative,
                    self.request.objective,
                ],
                "size": size,
            }
        )

    def _solve_signed(
        self, system: _System, base: np.ndarray, weights: np.ndarray
    ) -> _Solution:
        count = base.size
        sources = self.source_measures.shape[0]
        identifier = self._identifier("minimum-norm", count)
        # z = sqrt(d) (t - t0) keeps the declared objective as the Euclidean
        # minimum norm; no column preconditioning changes the norm.
        residual = np.asarray(system.rhs) - _apply(system, base)
        elimination = _eliminate_local_rows(
            system,
            np.asarray(self.routes.rows),
            np.asarray(self.routes.columns),
            weights,
            residual,
            sources,
            self.target_measures.shape[0],
        )
        if elimination.status != int(LinearSolveStatus.SUCCESS):
            return _outcome(
                base,
                PointTransferStatus.PROVIDER_UNRESOLVED,
                "minimum-norm",
                elimination.status,
            )
        if np.max(np.abs(elimination.local_residual), initial=0.0) > self.tolerance:
            # c - C C^+ c lies in null(C^T) and pairs positively with c: a
            # left-null witness of one target's own constant/moment rows.
            return _outcome(
                base,
                PointTransferStatus.INFEASIBLE,
                "minimum-norm",
                int(LinearSolveStatus.INCOMPATIBLE_RHS),
                witness=elimination.local_residual,
                witness_kind="left-null",
            )
        # Rows are equilibrated: scaling equations changes neither the feasible
        # set nor the minimum-norm point. A left-null witness of the scaled rows
        # maps back as ``y = D y'``.
        norms = np.sqrt(
            np.bincount(
                elimination.rows,
                weights=elimination.values * elimination.values,
                minlength=sources,
            )
        )
        row_scale = 1.0 / np.where(norms > 0, norms, 1.0)
        operator = SparseCoordinateOperator(
            EdgeRelation(
                elimination.columns,
                elimination.rows,
                source_size=count,
                target_size=sources,
            ),
            elimination.values * row_scale[elimination.rows],
            source=ArraySpace(
                (count,), dtype=np.float64, space_id=identifier + ":routes"
            ),
            target=ArraySpace(
                (sources,), dtype=np.float64, space_id=identifier + ":conservation"
            ),
            operator_id=identifier,
        )
        policy = (
            LinearSolvePolicy(
                LSMR(),
                tolerance=TolerancePolicy(
                    relative=min(1e-12, 1e-2 * self.tolerance),
                    absolute=1e-15,
                    max_steps=max(200, 4 * sources),
                ),
                failure=FailurePolicy("status"),
                # Host preparation freezes the coefficients (no geometry or
                # selection derivative), so no tangent solve is prepared.
                differentiation=DifferentiationPolicy("none"),
            )
            if self.linear_policy is None
            else self.linear_policy
        )
        result = solve(
            MinimumNormProblem(operator),
            jnp.asarray(row_scale * elimination.rhs),
            policy=policy,
        )
        status = int(np.asarray(result.status))
        evidence = result.minimum_norm
        if status == int(LinearSolveStatus.SUCCESS):
            correction = elimination.particular + np.asarray(result.value)
            return _outcome(
                base + correction / np.sqrt(weights),
                PointTransferStatus.ADMITTED,
                "minimum-norm",
                status,
            )
        if status == int(LinearSolveStatus.INCOMPATIBLE_RHS) and evidence is not None:
            return _outcome(
                base,
                PointTransferStatus.INFEASIBLE,
                "minimum-norm",
                status,
                witness=elimination.witness(
                    row_scale * np.asarray(evidence.left_null_witness)
                ),
                witness_kind="left-null",
            )
        return _outcome(
            base, PointTransferStatus.PROVIDER_UNRESOLVED, "minimum-norm", status
        )

    def _solve_conic(
        self, system: _System, base: np.ndarray, weights: np.ndarray
    ) -> _Solution:
        count, equations = base.size, np.asarray(system.rhs).size
        identifier = self._identifier("conic", count)
        variables = ArraySpace(
            (count,), dtype=np.float64, space_id=identifier + ":routes"
        )
        diagonal = np.arange(count)
        quadratic = SparseCoordinateOperator(
            EdgeRelation(diagonal, diagonal, source_size=count, target_size=count),
            weights,
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
        # Cone rows: b - B t in {0}^m and t >= 0 written as 0 - (-I) t >= 0.
        constraint = SparseCoordinateOperator(
            EdgeRelation(
                np.concatenate((np.asarray(system.route_indices), diagonal)),
                np.concatenate(
                    (np.asarray(system.equation_indices), equations + diagonal)
                ),
                source_size=count,
                target_size=equations + count,
            ),
            np.concatenate((np.asarray(system.values), -np.ones(count))),
            source=variables,
            target=ArraySpace(
                (equations + count,), dtype=np.float64, space_id=identifier + ":cone"
            ),
        )
        program = ConicProgram(
            quadratic,
            jnp.asarray(-weights * base),
            constraint,
            jnp.concatenate((system.rhs, jnp.zeros((count,), dtype=jnp.float64))),
            ProductCone((ZeroCone(equations), NonnegativeCone(count))),
            problem_id=identifier,
            convexity_evidence="construction",
        )
        policy = (
            ConvexSolvePolicy(
                NativeHomogeneousConic(),
                # KKT termination at the declared tolerance; the host audit then
                # checks every declared row and sign independently.
                termination=ConvexTermination(absolute=self.tolerance, maximum_steps=200),
                failure=FailurePolicy("status"),
            )
            if self.conic_policy is None
            else self.conic_policy
        )
        result = solve_conic_program(program, policy=policy)
        status = int(np.asarray(result.status))
        if status == int(ConvexProgramStatus.OPTIMAL):
            return _outcome(result.primal, PointTransferStatus.ADMITTED, "conic", status)
        if status == int(ConvexProgramStatus.PRIMAL_INFEASIBLE) and bool(
            np.asarray(result.certificate.dual_ray_valid)
        ):
            ray = np.asarray(result.certificate.inequality_dual_ray)[:equations]
            return _outcome(
                base,
                PointTransferStatus.INFEASIBLE,
                "conic",
                status,
                witness=ray / max(float(np.max(np.abs(ray))), np.finfo(np.float64).tiny),
                witness_kind="farkas",
            )
        return _outcome(base, PointTransferStatus.PROVIDER_UNRESOLVED, "conic", status)


def stencil_routes(
    stencils: PreparedLocalStencils, functional_index: int, /
) -> tuple[RowRelation, Array, Array]:
    """Relation, base weights and source-minus-target offsets of stencils."""
    if not isinstance(stencils, PreparedLocalStencils):
        raise TypeError("Cross-target preparation requires PreparedLocalStencils.")
    if stencils.report.refused_rows or not 0 <= functional_index < len(stencils.weights):
        raise ValueError(
            "Cross-target functional index or refused stencil rows are invalid."
        )
    return (
        stencils.neighborhood.relation,
        stencils.weights[functional_index],
        stencils.offsets,
    )


def _system(
    routes: _Routes, old: np.ndarray, new: np.ndarray, request: PointTransferRequest
) -> _System:
    """Normalized declared rows: conservation, then constant, then moments."""
    rows = np.asarray(routes.rows)
    columns = np.asarray(routes.columns)
    count = rows.size
    route = [np.arange(count)]
    equation = [columns]
    values = [new[rows] / old[columns]]
    rhs = [np.ones(old.size)]
    exponents: tuple[tuple[int, ...], ...] = ()
    if request.constant:
        route.append(np.arange(count))
        equation.append(old.size + rows)
        values.append(np.ones(count))
        rhs.append(np.ones(new.size))
    if request.moment_degree and routes.coordinates is not None:
        coordinates = np.asarray(routes.coordinates)
        scale = np.zeros(new.size)
        np.maximum.at(scale, rows, np.linalg.norm(coordinates, axis=1))
        scale = np.where(scale > 0, scale, 1.0)
        scaled = coordinates / scale[rows, None]
        exponents = _basis_exponents(coordinates.shape[1], request.moment_degree)[1:]
        offset = old.size + new.size
        for index, exponent in enumerate(exponents):
            route.append(np.arange(count))
            equation.append(offset + rows * len(exponents) + index)
            values.append(np.prod(scaled ** np.asarray(exponent), axis=1))
        rhs.append(np.zeros(new.size * len(exponents)))
    return _System(
        jnp.asarray(np.concatenate(route), dtype=jnp.int32),
        jnp.asarray(np.concatenate(equation), dtype=jnp.int32),
        jnp.asarray(np.concatenate(values)),
        jnp.asarray(np.concatenate(rhs)),
        exponents,
    )


# Native batched SVD pseudoinverse of the per-target local rows; rank
# deficiency is admitted and resolved by the local consistency residual.
_LOCAL_ROWS_PSEUDOINVERSE = FactorizationPolicy(
    "svd",
    rank=RankPolicy(relative_cutoff=1e-12),
    failure=FailurePolicy("status"),
)


class _Elimination(NamedTuple):
    """Conservation rows on the null space of every target's local rows."""

    particular: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray
    rhs: np.ndarray
    local_residual: np.ndarray
    local_inverse: np.ndarray
    local_equations: np.ndarray
    route_targets: np.ndarray
    route_slots: np.ndarray
    route_sources: np.ndarray
    route_conservation: np.ndarray
    status: int

    def witness(self, conservation: np.ndarray, /) -> np.ndarray:
        """Full left-null vector from one of the reduced conservation rows.

        ``(A P)^T y_A = 0`` means ``A^T y_A = C^T mu`` with
        ``mu = (C^+)^T A^T y_A``, so ``y = (y_A, -mu)`` annihilates every row.
        """
        witness = np.zeros_like(self.local_residual)
        witness[: conservation.size] = conservation
        if not self.local_equations.size:
            return witness
        image = np.zeros(self.local_inverse.shape[:2])
        image[self.route_targets, self.route_slots] = (
            self.route_conservation * conservation[self.route_sources]
        )
        multipliers = np.asarray(
            contract("tsl,ts->tl", jnp.asarray(self.local_inverse), jnp.asarray(image))
        )
        witness[self.local_equations.reshape(-1)] = -multipliers.reshape(-1)
        return witness


def _eliminate_local_rows(
    system: _System,
    rows: np.ndarray,
    columns: np.ndarray,
    weights: np.ndarray,
    residual: np.ndarray,
    sources: int,
    targets: int,
) -> _Elimination:
    """Exact elimination of every target's constant and moment rows.

    The local rows ``C_r`` of target ``r`` touch only its own routes, so with
    the native batched pseudoinverse ``z_C = C^+ c`` is the minimum-norm local
    solution and ``P = I - C^+ C`` the block-diagonal orthogonal projector onto
    ``null(C)``. As ``z_C`` lies in ``range(C^T)``, the minimum-norm solution of
    all declared rows is ``z_C + u`` with ``u`` the minimum-norm solution of the
    conservation rows ``A P u = a - A z_C``: the global Krylov solve sees only
    one equation per source.
    """
    route = np.asarray(system.route_indices)
    equation = np.asarray(system.equation_indices)
    values = np.asarray(system.values) / np.sqrt(weights)[route]
    count = rows.size
    conservation = equation < sources
    route_conservation = np.zeros(count)
    route_conservation[route[conservation]] = values[conservation]
    local_rows = (residual.size - sources) // targets
    if not local_rows:
        return _Elimination(
            np.zeros(count),
            columns,
            np.arange(count),
            route_conservation,
            residual[:sources],
            np.zeros_like(residual),
            np.zeros((0, 0, 0)),
            np.zeros((0, 0), dtype=np.int64),
            rows,
            np.zeros(count, dtype=np.int64),
            columns,
            route_conservation,
            int(LinearSolveStatus.SUCCESS),
        )
    order = np.argsort(rows, kind="stable")
    per_target = np.bincount(rows, minlength=targets)
    first = np.concatenate(([0], np.cumsum(per_target)[:-1]))
    slots = np.empty(count, dtype=np.int64)
    slots[order] = np.arange(count) - first[rows[order]]
    width = int(np.max(per_target))
    moments = local_rows - 1
    index = equation[~conservation] - sources
    owner = np.where(index < targets, index, (index - targets) // max(moments, 1))
    local = np.where(index < targets, 0, 1 + (index - targets) % max(moments, 1))
    matrix = np.zeros((targets, local_rows, width))
    matrix[owner, local, slots[route[~conservation]]] = values[~conservation]
    local_equations = np.concatenate(
        (
            sources + np.arange(targets)[:, None],
            sources
            + targets
            + moments * np.arange(targets)[:, None]
            + np.arange(moments)[None, :],
        ),
        axis=1,
    )
    local_rhs = residual[local_equations]
    inverse_result = pseudoinverse(jnp.asarray(matrix), _LOCAL_ROWS_PSEUDOINVERSE)
    status = np.asarray(inverse_result.status)
    inverse = jnp.asarray(inverse_result.value)
    particular_block = contract("tsl,tl->ts", inverse, jnp.asarray(local_rhs))
    local_residual = np.zeros_like(residual)
    local_residual[local_equations.reshape(-1)] = (
        local_rhs
        - np.asarray(contract("tls,ts->tl", jnp.asarray(matrix), particular_block))
    ).reshape(-1)
    projector = jnp.eye(width)[None] - contract(
        "tsl,tlk->tsk", inverse, jnp.asarray(matrix)
    )
    particular = np.asarray(particular_block)[rows, slots]
    # Route of each padded slot; conservation row of route e becomes
    # a_e P_r[slot e, :] over the routes of its own target.
    padded = np.full((targets, width), -1, dtype=np.int64)
    padded[rows, slots] = np.arange(count)
    partners = padded[rows]
    valid = partners >= 0
    coefficients = route_conservation[:, None] * np.asarray(projector)[rows, slots]
    reduced_rows = np.broadcast_to(columns[:, None], partners.shape)[valid]
    rhs = residual[:sources] - np.bincount(
        columns, weights=route_conservation * particular, minlength=sources
    )
    return _Elimination(
        particular,
        reduced_rows,
        partners[valid],
        coefficients[valid],
        rhs,
        local_residual,
        np.asarray(inverse),
        local_equations,
        rows,
        slots,
        columns,
        route_conservation,
        int(LinearSolveStatus.SUCCESS)
        if np.all(status == int(LinearSolveStatus.SUCCESS))
        else int(np.max(status)),
    )


def _apply(system: _System, coefficients: np.ndarray) -> np.ndarray:
    return np.bincount(
        np.asarray(system.equation_indices),
        weights=np.asarray(system.values)
        * coefficients[np.asarray(system.route_indices)],
        minlength=np.asarray(system.rhs).size,
    )


def _transpose(system: _System, witness: np.ndarray, count: int) -> np.ndarray:
    return np.bincount(
        np.asarray(system.route_indices),
        weights=np.asarray(system.values) * witness[np.asarray(system.equation_indices)],
        minlength=count,
    )


def _feasible(
    system: _System, coefficients: np.ndarray, tolerance: float, nonnegative: bool
) -> bool:
    residual = _apply(system, coefficients) - np.asarray(system.rhs)
    signs = not nonnegative or bool(np.all(coefficients >= 0))
    return signs and bool(np.max(np.abs(residual), initial=0.0) <= tolerance)


def _objective_weights(
    rows: np.ndarray, new: np.ndarray, objective: PointTransferObjective
) -> np.ndarray:
    match objective:
        case "coefficient-change":
            return np.ones(rows.size)
        case "measure-weighted-change":
            return new[rows] / np.mean(new)
        case unknown:
            assert_never(unknown)


def _witness_audit(
    solution: _Solution, system: _System, count: int, tolerance: float
) -> tuple[float, float, bool]:
    """Independent check that the witness proves the declared rows infeasible.

    For ``|B t - b| <= tol`` one has ``y^T b = (B^T y)^T t - y^T r``. Equality
    witnesses need ``B^T y = 0`` and Farkas rays ``B^T y >= 0`` within the
    declared tolerance, and the margin ``b^T y`` must exceed the declared
    tolerance acting on the witness (``tol |y|_1``) plus that residual, so no
    coefficient vector meets the declared equations within tolerance.
    """
    witness = np.asarray(solution.witness)
    if solution.witness_kind == "none":
        return 0.0, 0.0, True
    rhs = np.asarray(system.rhs)
    image = _transpose(system, witness, count)
    margin = float(rhs @ witness)
    scale = float(np.max(np.abs(np.asarray(system.values)))) * max(
        float(np.max(np.abs(witness))), 1.0
    )
    bound = tolerance * scale * np.sqrt(count)
    floor = tolerance * float(np.sum(np.abs(witness)))
    match solution.witness_kind:
        case "measure-obstruction" | "left-null":
            residual = float(np.max(np.abs(image), initial=0.0))
            valid = residual <= bound and abs(margin) > floor + residual
        case "farkas":
            # B^T y >= 0 with b^T y < 0 excludes every t >= 0 with B t = b.
            residual = float(max(-np.min(image, initial=0.0), 0.0))
            valid = residual <= bound and -margin > floor + residual
        case unknown:
            assert_never(unknown)
    return residual, margin, valid


def _evidence(
    solution: _Solution,
    system: _System,
    base: np.ndarray,
    weights: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    plan: PointTransferPlan,
) -> PointTransferEvidence:
    values = np.asarray(solution.values)
    tolerance = plan.tolerance
    conservation = (
        np.bincount(columns, weights=new[rows] * values, minlength=old.size) - old
    ) / old
    constant = np.bincount(rows, weights=values, minlength=new.size) - 1.0
    moments = _apply(system, values)[old.size + new.size :] if system.exponents else ()
    moment = np.asarray(moments, dtype=np.float64).reshape(
        new.size, len(system.exponents)
    )
    conservative = bool(np.max(np.abs(conservation), initial=0.0) <= tolerance)
    constant_ok = bool(np.max(np.abs(constant), initial=0.0) <= tolerance)
    moments_ok = bool(np.max(np.abs(moment), initial=0.0) <= tolerance)
    # Signs are audited to the declared tolerance relative to the largest
    # coefficient: provider roundoff at an active bound is not a sign change.
    nonnegative = bool(np.all(values >= -tolerance * np.max(np.abs(values), initial=0.0)))
    request = plan.request
    status = solution.status
    residual, margin, witness_valid = _witness_audit(
        solution, system, values.size, tolerance
    )
    if status is PointTransferStatus.INFEASIBLE and not witness_valid:
        status = PointTransferStatus.PROVIDER_UNRESOLVED
    if status is PointTransferStatus.ADMITTED and not (
        conservative
        and (not request.constant or constant_ok)
        and (not system.exponents or moments_ok)
        and (not request.nonnegative or nonnegative)
    ):
        status = PointTransferStatus.AUDIT_FAILED
    lebesgue = float(
        np.max(np.bincount(rows, weights=np.abs(values), minlength=new.size))
    )
    if (
        status is PointTransferStatus.ADMITTED
        and plan.lebesgue_bound is not None
        and lebesgue > plan.lebesgue_bound
    ):
        status = PointTransferStatus.AMPLIFICATION_EXCEEDED
    change = values - base
    return PointTransferEvidence(
        request,
        jnp.asarray(values),
        jnp.asarray(conservation),
        jnp.asarray(constant),
        jnp.asarray(moment),
        solution.witness,
        status,
        solution.provider,
        solution.provider_status,
        solution.witness_kind,
        residual,
        margin,
        float(np.sum(old) - np.sum(new)),
        float(np.linalg.norm(change)),
        float(0.5 * np.sum(weights * change * change)),
        float(np.min(values, initial=np.inf)) if values.size else 0.0,
        lebesgue,
        conservative,
        constant_ok,
        nonnegative,
        moments_ok,
        tuple(
            int(i) for i in np.flatnonzero(np.bincount(columns, minlength=old.size) == 0)
        ),
        tuple(int(i) for i in np.flatnonzero(np.bincount(rows, minlength=new.size) == 0)),
        tolerance,
    )


def _field_transfer(
    evidence: PointTransferEvidence,
    rows: np.ndarray,
    columns: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    plan: PointTransferPlan,
) -> FieldTransfer:
    values = np.asarray(evidence.coefficients)
    source_space, _ = diagonal_gram_space(
        old,
        dtype=np.float64,
        space_id=canonical_fingerprint({"support": plan.source_id, "measures": old}),
    )
    target_space, _ = diagonal_gram_space(
        new,
        dtype=np.float64,
        space_id=canonical_fingerprint({"support": plan.target_id, "measures": new}),
    )
    forward = EdgeRelation(columns, rows, source_size=old.size, target_size=new.size)
    reverse = EdgeRelation(rows, columns, source_size=new.size, target_size=old.size)
    identifier = canonical_fingerprint(
        {
            "kind": "point-transfer",
            "source": plan.source_id,
            "target": plan.target_id,
            "rows": rows,
            "columns": columns,
            "values": values,
            "source_measures": old,
            "target_measures": new,
        }
    )
    exact: list[str] = []
    if evidence.constant_preserving:
        exact.append("constant")
    if plan.request.moment_degree and evidence.moments_exact:
        exact.append(f"polynomial-degree-{plan.request.moment_degree}")
    return FieldTransfer(
        _concentration_space(plan.source_id, source_space, old.size),
        _concentration_space(plan.target_id, target_space, new.size),
        SparseCoordinateOperator(
            forward,
            values,
            source=source_space,
            target=target_space,
            operator_id=identifier + ":primal",
        ),
        dual_pullback_operator=SparseCoordinateOperator(
            reverse,
            values,
            source=target_space,
            target=source_space,
            operator_id=identifier + ":dual",
        ),
        hilbert_adjoint_operator=SparseCoordinateOperator(
            reverse,
            values * new[rows] / old[columns],
            source=target_space,
            target=source_space,
            operator_id=identifier + ":hilbert",
        ),
        properties=TransferProperties(
            conservative=evidence.conservative,
            constant_preserving=evidence.constant_preserving,
            positivity_preserving=evidence.nonnegative,
            adjoint_paired=True,
            differentiable_geometry=False,
            exact_on=exact,
        ),
    )


def _concentration_space(
    support_id: str, space: ArraySpace, size: int
) -> DiscreteFieldSpace:
    return DiscreteFieldSpace(
        "concentration",
        support_id,
        TensorDofLayout(("point",), (size,)),
        space,
        representation="point_value",
    )


__all__ = [
    "PointTransferEvidence",
    "PointTransferObjective",
    "PointTransferPlan",
    "PointTransferProvider",
    "PointTransferRequest",
    "PointTransferRoute",
    "PointTransferStatus",
    "PointTransferWitness",
    "PreparedPointTransfer",
    "stencil_routes",
]
