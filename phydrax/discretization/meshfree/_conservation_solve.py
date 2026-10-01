# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Learned edge conservation through the native prepared nonlinear lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._admissibility import guard_derivative_validity
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    GMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    prepare as prepare_linear,
    prepare_sparse_factorization,
    PreparedLinearSolve,
    refresh as refresh_linear,
    refresh_sparse_factorization_values,
    solve as solve_linear,
    solve_transpose,
    SparseFactorizationPlan,
    SparseFactorizationPolicy,
    SparseFactorizationStatus,
    TolerancePolicy,
)
from ...nonlinear import (
    implicit_root_result,
    ImplicitRootDerivativePolicy,
    JacobianPolicy,
    NewtonKrylov,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
    solve_prepared_nonlinear,
)
from ...solver.coupling._parameters import ParameterBinding, PreparedParameters
from ...sparse import (
    compile_sparse_jacobian,
    EdgeRelation,
    SparseCoordinateOperator,
    SparsePattern,
)
from ...sparse._linear import _SparseStoragePlan
from ...typing import Bool, Dim, Float64, Int32, Scalar
from ._constitutive import (
    AbstractEdgeConstitutiveLaw,
    EdgeFrameFeatures,
    LipschitzEdgeFlux,
    MonotoneEdgeConductance,
)
from ._coverage import EdgeCoverageAssessment, EdgeFeatureCoverage
from ._exterior import PreparedMeshfreeExteriorCalculus


class _ConservationNodeDim(Dim):
    """Compact physical conservation nodes."""


class _ConservationEdgeDim(Dim):
    """Compact physical constitutive edges."""


class _ConservationFreeDim(Dim):
    """Dirichlet-reduced equation coordinates."""


class _ConservationRouteDim(Dim):
    """Stored routes of the reduced sparse edge stiffness."""


class MeshfreeContractionStatus(IntEnum):
    CERTIFIED = 0
    CONTRACT_UNCERTIFIED = 1
    NOT_APPLICABLE = 2


@final
class MeshfreeConstitutiveEvidence(StrictModule, NonTrainableState):
    """Proof hypotheses and native factor status, not a spectral estimate."""

    __strict_contract__ = True
    positive_metric: Bool[Scalar]
    anchored: Bool[Scalar]
    background_factor_status: Int32[Scalar]
    background_minimum_pivot: Float64[Scalar]
    coercivity_certified: Bool[Scalar]
    strong_monotonicity_lower_bound: Float64[Scalar]
    contraction_status: Int32[Scalar]
    contraction_bound: Float64[Scalar]
    norm: str = eqx.field(static=True)
    conditions: tuple[str, ...] = eqx.field(static=True)


@final
class MeshfreeConservationLedger(StrictModule, NonTrainableState):
    __strict_contract__ = True
    source_integral: Float64[Scalar]
    outward_boundary_flux: Float64[Scalar]
    balance_defect: Float64[Scalar]
    internal_flux_sum: Float64[Scalar]
    maximum_equation_defect: Float64[Scalar]


@final
class MeshfreeConservationResult(StrictModule):
    """Primal status is retained; an unaccepted public state is NaN, never a score.

    Native implicit derivatives check their tangent/adjoint solves. An adjoint
    status is not invented before a cotangent is supplied: ``adjoint`` returns
    an explicit native linear result alongside this primal result.
    The portable-status result uses native derivative-validity guards: JVPs and
    VJPs of failed public state/flux values are undefined NaN, never finite zero.
    ``require_success`` explicitly raises for an unsuccessful forward result.
    """

    __strict_contract__ = True
    state: Float64[_ConservationNodeDim]
    root: NonlinearResult
    edge_flux: Float64[_ConservationEdgeDim]
    ledger: MeshfreeConservationLedger
    coverage: EdgeCoverageAssessment | None
    constitutive_evidence: MeshfreeConstitutiveEvidence
    accepted: Bool[Scalar]
    background: LinearSolveResult | None

    @property
    def successful(self) -> Array:
        return self.accepted

    @property
    def primal_status(self) -> Array:
        return self.root.status

    def require_success(self) -> Array:
        return eqx.error_if(
            self.state,
            ~self.accepted,
            "Meshfree conservation forward solve failed; inspect primal, coverage and constitutive evidence.",
        )


@final
class MeshfreeConservationAdjoint(StrictModule):
    """Native statuses and value; failed primal/adjoint sensitivities are NaN."""

    __strict_contract__ = True
    primal: MeshfreeConservationResult
    linear: LinearSolveResult
    value: Float64[_ConservationFreeDim]
    accepted: Bool[Scalar]

    @property
    def primal_status(self) -> Array:
        return self.primal.primal_status

    @property
    def adjoint_status(self) -> Array:
        return self.linear.status


@final
class _ConservationRuntime(StrictModule):
    __strict_contract__ = True
    law_parameters: tuple[Array, ...]
    conductance: Float64[Scalar]
    source: Float64[_ConservationNodeDim]
    boundary_values: Float64[_ConservationNodeDim]


def _node_array(value: ArrayLike, count: int, name: str, /) -> Array:
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.shape not in ((), (count,)):
        raise ValueError(f"{name} must be scalar or one value per compact node.")
    return jnp.broadcast_to(array, (count,))


@final
class MeshfreeConservationProblem(StrictModule, NonTrainableState):
    """Integrated steady equations ``B.T a law(Bu) = volumes * source``.

    The equation mask names solved rows. Every other row is prescribed by
    ``boundary_values``; no zero-volume padding or guessed natural boundary is
    introduced. Even/odd constitutive context is immutable *external* data,
    independent of the unknown state, so the monotonicity proof is global.
    """

    exterior: PreparedMeshfreeExteriorCalculus
    law: AbstractEdgeConstitutiveLaw
    features: EdgeFrameFeatures
    __strict_contract__ = True
    source: Float64[_ConservationNodeDim]
    boundary_values: Float64[_ConservationNodeDim]
    boundary_mask: Bool[_ConservationNodeDim]
    equation_mask: Bool[_ConservationNodeDim]
    coverage: EdgeFeatureCoverage | None
    parameters: PreparedParameters
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        exterior: PreparedMeshfreeExteriorCalculus,
        law: AbstractEdgeConstitutiveLaw,
        /,
        *,
        features: EdgeFrameFeatures | None = None,
        source: ArrayLike = 0.0,
        boundary_values: ArrayLike = 0.0,
        boundary_mask: ArrayLike | None = None,
        equation_mask: ArrayLike | None = None,
        coverage: EdgeFeatureCoverage | None = None,
        parameter_bindings: tuple[ParameterBinding, ...] = (),
        problem_id: str = "meshfree-conservation",
    ) -> None:
        if not isinstance(exterior, PreparedMeshfreeExteriorCalculus) or not isinstance(
            law, AbstractEdgeConstitutiveLaw
        ):
            raise TypeError(
                "Conservation requires native prepared exterior geometry and an edge constitutive law."
            )
        if not bool(exterior.metric_result.accepted) or not bool(
            exterior.metric_result.hilbert_admitted
        ):
            raise ValueError(
                "Learned dissipative conservation requires an admitted strictly positive metric; signed pairings are refused."
            )
        count = exterior.points.shape[0]
        equations = np.asarray(
            exterior.equation_mask if equation_mask is None else equation_mask
        )
        boundary = ~equations if boundary_mask is None else np.asarray(boundary_mask)
        if (
            equations.dtype != np.bool_
            or boundary.dtype != np.bool_
            or equations.shape != (count,)
            or boundary.shape != (count,)
        ):
            raise ValueError(
                "Equation and boundary masks must be Boolean compact-node vectors."
            )
        if (
            np.any(equations & boundary)
            or not np.all(equations | boundary)
            or not np.any(equations)
        ):
            raise ValueError(
                "Every node is either an equation row or a prescribed boundary row, with at least one equation."
            )
        if np.any(equations & ~np.asarray(exterior.equation_mask)):
            raise ValueError(
                "Conservation equations must be a subset of the exterior's moment-certified rows."
            )
        features_ = (
            EdgeFrameFeatures(
                exterior.points,
                exterior.pairs,
                source_id=exterior.incidence.source.space_id,
            )
            if features is None
            else features
        )
        if (
            not isinstance(features_, EdgeFrameFeatures)
            or features_.source_id != exterior.incidence.source.space_id
            or features_.geometry_id
            != canonical_fingerprint(
                array_tree_fingerprint((exterior.points, exterior.pairs))
            )
        ):
            raise ValueError(
                "Constitutive features must retain the prepared exterior's nominal geometry and edge orientation."
            )
        if (
            features_.even.shape[1] != law.even_size
            or features_.odd.shape[1] != law.odd_size
        ):
            raise ValueError(
                "Typed even/odd feature widths must match the law's declared input schema."
            )
        if coverage is not None and (
            not isinstance(coverage, EdgeFeatureCoverage)
            or coverage.feature_names != features_.even_names
        ):
            raise ValueError(
                "Coverage must name exactly the immutable even-feature layout."
            )
        source_ = _node_array(source, count, "source")
        values_ = _node_array(boundary_values, count, "boundary_values")
        if not np.all(np.isfinite(np.asarray(source_))) or not np.all(
            np.isfinite(np.asarray(values_))
        ):
            raise ValueError("Prepared source and boundary values must be finite.")
        if (
            not isinstance(problem_id, str)
            or not problem_id
            or problem_id.strip() != problem_id
        ):
            raise ValueError(
                "problem_id must be a nonempty stripped scientific identity."
            )
        if not isinstance(parameter_bindings, tuple) or any(
            not isinstance(binding, ParameterBinding) for binding in parameter_bindings
        ):
            raise TypeError(
                "parameter_bindings must be a tuple of native ParameterBinding objects."
            )
        ids = [binding.binding_id for binding in parameter_bindings]
        targets = [
            target.name for binding in parameter_bindings for target in binding.targets
        ]
        if len(set(ids)) != len(ids) or len(set(targets)) != len(targets):
            raise ValueError("Parameter identities and runtime targets must be unique.")
        for binding in parameter_bindings:
            if binding.change != "refresh" or any(
                target.component != "meshfree"
                or target.name not in ("law", "conductance", "source", "boundary_values")
                for target in binding.targets
            ):
                raise ValueError(
                    "Conservation parameters bind refresh-only meshfree law/conductance/source/boundary_values inputs."
                )
            for target in binding.targets:
                required_role = (
                    "coefficient"
                    if target.name in ("law", "conductance")
                    else "source"
                    if target.name == "source"
                    else "boundary"
                )
                if binding.role != required_role:
                    raise ValueError(
                        "Parameter physical role must agree with its runtime input."
                    )
        parameter_owner = PreparedParameters(parameter_bindings, (), ())
        identity = canonical_fingerprint(
            {
                "kind": "meshfree-conservation",
                "owner": problem_id,
                "geometry_source": exterior.incidence.source.space_id,
                "edge_source": exterior.incidence.target.space_id,
                "geometry": array_tree_fingerprint((exterior.points, exterior.pairs)),
                "equations": equations.tolist(),
            }
        )
        self.exterior = exterior
        self.law = law
        self.features = features_
        self.source = source_
        self.boundary_values = values_
        self.boundary_mask = jnp.asarray(boundary)
        self.equation_mask = jnp.asarray(equations)
        self.coverage = coverage
        self.parameters = parameter_owner
        self.problem_id = identity

    def runtime(
        self,
        *,
        parameters: Mapping[str, object] | None = None,
        law: AbstractEdgeConstitutiveLaw | None = None,
    ) -> _ConservationRuntime:
        bound = self.parameters.bind(("meshfree",), None, parameters).arguments[
            "meshfree"
        ]
        inputs = {} if bound is None else bound
        if not isinstance(inputs, Mapping):
            raise TypeError(
                "Native parameter binding must yield a meshfree runtime input mapping."
            )
        if law is not None and "law" in inputs:
            raise ValueError(
                "A bound law parameter and an explicit law refresh cannot both supply the same physical input."
            )
        law_ = inputs.get("law", self.law if law is None else law)
        if (
            not isinstance(law_, AbstractEdgeConstitutiveLaw)
            or type(law_) is not type(self.law)
            or law_.in_size != self.law.in_size
        ):
            raise ValueError(
                "Runtime constitutive refresh must preserve the law family and feature schema."
            )
        if isinstance(law_, MonotoneEdgeConductance):
            law_.potential.input_convex_certificate()
        numeric_law, law_traits = eqx.partition(law_, eqx.is_array)
        _, prepared_traits = eqx.partition(self.law, eqx.is_array)
        if eqx.tree_equal(law_traits, prepared_traits) is not True:
            raise ValueError(
                "Constitutive metadata changed; prepare a new solve for different activation, transforms or static law traits."
            )
        law_parameters = tuple(jax.tree_util.tree_leaves(numeric_law))

        def value(name: str, default: ArrayLike) -> Array:
            item = inputs.get(name, default)
            if isinstance(item, AbstractArrayModel):
                if item.in_size != "scalar":
                    raise ValueError(
                        "A runtime parameter model must declare a scalar constant input."
                    )
                return jnp.asarray(
                    item(jnp.zeros((), dtype=jnp.float64)), dtype=jnp.float64
                )
            return jnp.asarray(item, dtype=jnp.float64)

        conductance = value("conductance", 1.0)
        if conductance.shape != ():
            raise ValueError(
                "The constitutive multiplier must be scalar; spatial context belongs to typed external edge features."
            )
        source = _node_array(value("source", self.source), self.source.size, "source")
        boundary = _node_array(
            value("boundary_values", self.boundary_values),
            self.source.size,
            "boundary_values",
        )
        return _ConservationRuntime(law_parameters, conductance, source, boundary)


@final
class _ConservationResidual(StrictModule, NonTrainableState):
    __strict_contract__ = True
    exterior: PreparedMeshfreeExteriorCalculus
    features: EdgeFrameFeatures
    free_indices: Int32[_ConservationFreeDim]
    validity: _ConservationValidity
    law_reference: AbstractEdgeConstitutiveLaw

    def bound_law(self, args: _ConservationRuntime, /) -> AbstractEdgeConstitutiveLaw:
        """Reconstruct canonical law metadata around the actual numerical leaves."""
        numeric_template, traits = eqx.partition(self.law_reference, eqx.is_array)
        numeric_law = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(numeric_template), args.law_parameters
        )
        return eqx.combine(numeric_law, traits)

    def reconstruct(self, state: Array, args: _ConservationRuntime, /) -> Array:
        return args.boundary_values.at[self.free_indices].set(state)

    def integrated_flux(
        self, state: Array, args: _ConservationRuntime, /
    ) -> tuple[Array, Array]:
        full = self.reconstruct(state, args)
        edge = (
            args.conductance
            * self.exterior.metric_result.weights
            * self.bound_law(args).flux(self.exterior.gradient(full), self.features)
        )
        return self.exterior.incidence.transpose_mv(edge), -edge

    def __call__(self, state: Array, args: _ConservationRuntime, /) -> Array:
        def evaluate(_operand: None) -> Array:
            integrated = self.integrated_flux(state, args)[0]
            return (integrated - self.exterior.node_volumes * args.source)[
                self.free_indices
            ]

        return jax.lax.cond(
            self.validity.admitted(args), evaluate, lambda _: jnp.zeros_like(state), None
        )


@final
class _ConservationValidity(StrictModule, NonTrainableState):
    __strict_contract__ = True
    coverage_admitted: Bool[Scalar]
    background_admitted: Bool[Scalar]

    def __call__(
        self,
        state: Array,
        residual: Array,
        auxiliary: object,
        args: _ConservationRuntime,
        /,
    ) -> Array:
        del state, residual, auxiliary
        return self.admitted(args)

    def admitted(self, args: _ConservationRuntime, /) -> Array:
        return (
            self.coverage_admitted
            & self.background_admitted
            & jnp.isfinite(args.conductance)
            & (args.conductance > 0)
            & jnp.all(jnp.isfinite(args.source))
            & jnp.all(jnp.isfinite(args.boundary_values))
        )

    def trial(self, state: Array, args: _ConservationRuntime, /) -> Array:
        del state
        return self.admitted(args)


@final
class _ReducedStiffness(StrictModule, NonTrainableState):
    __strict_contract__ = True
    exterior: PreparedMeshfreeExteriorCalculus
    relation: EdgeRelation
    routes: Int32[_ConservationRouteDim]
    space: ArraySpace
    storage: _SparseStoragePlan

    def operator(self, conductances: Array, /) -> SparseCoordinateOperator:
        full = self.exterior.stiffness(conductances)
        properties = OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "construction"}
        )
        return SparseCoordinateOperator(
            self.relation,
            full.coefficients[self.routes],
            source=self.space,
            target=self.space,
            properties=properties,
            storage_plan=self.storage,
            operator_id=f"{self.space.space_id}:stiffness",
        )


@final
class PreparedMeshfreeConservationSolve(StrictModule, NonTrainableState):
    """Native nonlinear template, fixed sparse edge pattern and explicit background.

    The certificate's norm is ``||v||_A0`` for the Dirichlet-reduced background
    stiffness, not the unweighted Euclidean norm. Fixed external context and the
    same edge metric give ``||T(u)-T(v)||_A0 <= (L/b)||u-v||_A0`` by Cauchy-Schwarz.
    No claim is made for context that depends on the solved nodal field.
    """

    __strict_contract__ = True

    problem: MeshfreeConservationProblem
    native: PreparedNonlinearSolve
    derivative_policy: ImplicitRootDerivativePolicy
    residual: _ConservationResidual
    reduced: _ReducedStiffness
    background: PreparedLinearSolve
    background_factor_plan: SparseFactorizationPlan
    coverage_assessment: EdgeCoverageAssessment | None
    anchored: Bool[Scalar]
    factor_status: Int32[Scalar]
    factor_minimum_pivot: Float64[Scalar]

    def refresh(
        self,
        *,
        parameters: Mapping[str, object] | None = None,
        law: AbstractEdgeConstitutiveLaw | None = None,
        initial_state: ArrayLike | None = None,
    ) -> PreparedNonlinearSolve:
        args = self.problem.runtime(parameters=parameters, law=law)
        return self._refresh_runtime(args, initial_state)

    def _refresh_runtime(
        self,
        args: _ConservationRuntime,
        initial_state: ArrayLike | None,
        background_successful: Array | None = None,
        /,
    ) -> PreparedNonlinearSolve:
        state = (
            self.native.state
            if initial_state is None
            else jnp.asarray(initial_state, dtype=jnp.float64)
        )
        if state.shape == self.problem.source.shape:
            state = state[self.residual.free_indices]
        problem = self.native.problem
        if background_successful is not None:
            validity = _ConservationValidity(
                jnp.asarray(True)
                if self.coverage_assessment is None
                else self.coverage_assessment.admitted,
                (self.factor_status == int(SparseFactorizationStatus.SUCCESS))
                & self.anchored
                & background_successful,
            )
            problem = eqx.tree_at(
                lambda candidate: candidate.validity_function, problem, validity
            )
        return refresh_nonlinear(self.native, problem, state, args=args)

    def background_solve(
        self, args: _ConservationRuntime, /, *, state: ArrayLike | None = None
    ) -> LinearSolveResult:
        """One native background diffusion solve; optionally one true Picard step."""
        law = self.residual.bound_law(args)
        b = law.background_conductance
        weights = self.problem.exterior.metric_result.weights
        conductances = args.conductance * b * weights
        operator = self.reduced.operator(conductances)
        prepared = refresh_linear(
            self.background,
            LinearSystem(operator, problem_id=self.background.problem.problem_id),
        )
        boundary = args.boundary_values.at[self.residual.free_indices].set(0)
        full_operator = self.problem.exterior.stiffness(conductances)
        rhs = (
            self.problem.exterior.node_volumes * args.source - full_operator.mv(boundary)
        )[self.residual.free_indices]
        if state is not None:
            full = jnp.asarray(state, dtype=jnp.float64)
            if full.shape == self.native.state.shape:
                full = self.residual.reconstruct(full, args)
            jumps = self.problem.exterior.gradient(full)
            nonlinear = law.flux(jumps, self.problem.features) - b * jumps
            rhs = (
                rhs
                - self.problem.exterior.incidence.transpose_mv(
                    args.conductance * weights * nonlinear
                )[self.residual.free_indices]
            )
        return solve_linear(prepared, rhs)

    def constitutive_evidence(
        self, args: _ConservationRuntime, /
    ) -> MeshfreeConstitutiveEvidence:
        factor_good = (
            self.factor_status == int(SparseFactorizationStatus.SUCCESS)
        ) & self.anchored
        positive = (
            jnp.isfinite(args.conductance)
            & (args.conductance > 0)
            & self.problem.exterior.metric_result.hilbert_admitted
        )
        law = self.residual.bound_law(args)
        b = law.background_conductance
        if isinstance(law, MonotoneEdgeConductance):
            coercive = factor_good & positive
            lower = jnp.where(coercive, jnp.asarray(1.0, dtype=jnp.float64), jnp.nan)
            status = jnp.asarray(
                int(MeshfreeContractionStatus.NOT_APPLICABLE), dtype=jnp.int32
            )
            bound = jnp.asarray(jnp.nan, dtype=jnp.float64)
        elif isinstance(law, LipschitzEdgeFlux):
            lipschitz = jnp.asarray(law.certified_lipschitz_bound(), dtype=jnp.float64)
            bound = lipschitz / b
            certified = factor_good & positive & jnp.isfinite(bound) & (bound < 1)
            status = jnp.where(
                certified,
                int(MeshfreeContractionStatus.CERTIFIED),
                int(MeshfreeContractionStatus.CONTRACT_UNCERTIFIED),
            ).astype(jnp.int32)
            lower = jnp.where(factor_good & positive, 1.0 - bound, jnp.nan)
            coercive = certified
        else:
            lower = jnp.asarray(jnp.nan, dtype=jnp.float64)
            coercive = jnp.asarray(False)
            bound = jnp.asarray(jnp.nan, dtype=jnp.float64)
            status = jnp.asarray(
                int(MeshfreeContractionStatus.CONTRACT_UNCERTIFIED), dtype=jnp.int32
            )
        return MeshfreeConstitutiveEvidence(
            positive,
            self.anchored,
            self.factor_status,
            self.factor_minimum_pivot,
            coercive,
            lower,
            status,
            bound,
            "Dirichlet-background-energy",
            (
                "fixed-external-context",
                "same-positive-edge-metric",
                "native-no-shift-sparse-Cholesky",
                "positive-runtime-conductance",
            ),
        )

    def solve(
        self,
        *,
        parameters: Mapping[str, object] | None = None,
        law: AbstractEdgeConstitutiveLaw | None = None,
        initial_state: ArrayLike | None = None,
        implicit: bool = True,
    ) -> MeshfreeConservationResult:
        args = self.problem.runtime(parameters=parameters, law=law)
        background_result = (
            self.background_solve(args)
            if isinstance(self.problem.law, LipschitzEdgeFlux)
            else None
        )
        state = initial_state
        if state is None and background_result is not None:
            state = background_result.value
        native = self._refresh_runtime(
            args,
            state,
            None if background_result is None else jnp.all(background_result.successful),
        )
        root = (
            implicit_root_result(native, derivative_policy=self.derivative_policy)
            if implicit
            else solve_prepared_nonlinear(native)
        )
        evidence = self.constitutive_evidence(args)
        accepted = root.successful
        if background_result is not None:
            accepted = accepted & jnp.all(background_result.successful)
        if self.coverage_assessment is not None:
            accepted = accepted & self.coverage_assessment.admitted
        status = jnp.where(
            root.successful & ~accepted,
            int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
            root.status,
        )
        root = eqx.tree_at(lambda value: value.status, root, status)
        full = self.residual.reconstruct(root.state, args)
        integrated, edge_flux = jax.lax.cond(
            accepted,
            lambda _: self.residual.integrated_flux(root.state, args),
            lambda _: (
                jnp.full_like(args.source, jnp.nan),
                jnp.full_like(self.problem.exterior.lengths, jnp.nan),
            ),
            None,
        )
        source_integral = jnp.sum(
            (self.problem.exterior.node_volumes * args.source)[self.residual.free_indices]
        )
        boundary_flux = -jnp.sum(jnp.where(self.problem.equation_mask, 0, integrated))
        defect = integrated - self.problem.exterior.node_volumes * args.source
        ledger = MeshfreeConservationLedger(
            source_integral,
            boundary_flux,
            source_integral - boundary_flux,
            jnp.sum(integrated),
            jnp.max(jnp.abs(defect[self.residual.free_indices])),
        )
        public_state, public_flux = guard_derivative_validity(
            (jnp.where(accepted, full, jnp.nan), jnp.where(accepted, edge_flux, jnp.nan)),
            accepted,
            dependencies=(args, initial_state),
            failure="status",
            message="Meshfree conservation derivatives require a successful forward result.",
        )
        return MeshfreeConservationResult(
            public_state,
            root,
            public_flux,
            ledger,
            self.coverage_assessment,
            evidence,
            accepted,
            background_result,
        )

    def adjoint(
        self,
        primal: MeshfreeConservationResult,
        cotangent: ArrayLike,
        /,
        *,
        parameters: Mapping[str, object] | None = None,
        law: AbstractEdgeConstitutiveLaw | None = None,
    ) -> MeshfreeConservationAdjoint:
        """Explicit native transpose solve retaining both primal and adjoint status."""
        args = self.problem.runtime(parameters=parameters, law=law)
        rhs = jnp.asarray(cotangent, dtype=jnp.float64)
        if rhs.shape == self.problem.source.shape:
            rhs = rhs[self.residual.free_indices]
        slope = (
            args.conductance
            * self.problem.exterior.metric_result.weights
            * self.residual.bound_law(args).derivative(
                self.problem.exterior.gradient(primal.state), self.problem.features
            )
        )
        operator = self.reduced.operator(slope)
        _, adjoint_policy = self.derivative_policy.resolve(self.native.method)
        linear = solve_transpose(LinearSystem(operator), rhs, policy=adjoint_policy)
        accepted = primal.accepted & jnp.all(linear.successful)
        value = guard_derivative_validity(
            jnp.where(accepted, linear.value, jnp.nan),
            accepted,
            dependencies=(args, cotangent),
            failure="status",
            message="Meshfree conservation adjoint derivatives require successful primal and adjoint results.",
        )
        return MeshfreeConservationAdjoint(primal, linear, value, accepted)


def prepare_meshfree_conservation_solve(
    problem: MeshfreeConservationProblem,
    /,
    *,
    initial_state: ArrayLike | None = None,
    parameters: Mapping[str, object] | None = None,
    linear_policy: LinearSolvePolicy | None = None,
    termination: NonlinearTermination | None = None,
    derivative_policy: ImplicitRootDerivativePolicy | None = None,
) -> PreparedMeshfreeConservationSolve:
    """Prepare native sparse compressed Jacobian and reusable nonlinear/linear plans."""
    if not isinstance(problem, MeshfreeConservationProblem):
        raise TypeError("problem must be a MeshfreeConservationProblem.")
    free = np.flatnonzero(np.asarray(problem.equation_mask)).astype(np.int32)
    inverse = np.full(problem.source.size, -1, dtype=np.int32)
    inverse[free] = np.arange(free.size, dtype=np.int32)
    pairs = np.asarray(problem.exterior.pairs)
    i, j = pairs[:, 0], pairs[:, 1]
    rows = np.stack((i, j, i, j), axis=1).reshape(-1)
    columns = np.stack((i, j, j, i), axis=1).reshape(-1)
    routes = np.flatnonzero((inverse[rows] >= 0) & (inverse[columns] >= 0)).astype(
        np.int32
    )
    relation = EdgeRelation(
        inverse[columns[routes]],
        inverse[rows[routes]],
        source_size=free.size,
        target_size=free.size,
    )
    space = ArraySpace(
        (free.size,), dtype=jnp.float64, space_id=f"{problem.problem_id}:free-state"
    )
    storage = _SparseStoragePlan(relation)
    reduced = _ReducedStiffness(
        problem.exterior, relation, jnp.asarray(routes), space, storage
    )
    args = problem.runtime(parameters=parameters)
    policy = (
        LinearSolvePolicy(
            GMRES(restart=min(32, max(1, free.size))),
            tolerance=TolerancePolicy(
                relative=1e-10, absolute=1e-12, max_steps=max(64, 2 * free.size)
            ),
        )
        if linear_policy is None
        else linear_policy
    )
    derivative = (
        ImplicitRootDerivativePolicy(
            tangent_linear_policy=policy, adjoint_linear_policy=policy
        )
        if derivative_policy is None
        else derivative_policy
    )
    if not isinstance(policy, LinearSolvePolicy) or not isinstance(
        derivative, ImplicitRootDerivativePolicy
    ):
        raise TypeError("Linear and derivative policies must be native policy objects.")
    weights = problem.exterior.metric_result.weights
    # Reachability is through strictly positive physical weights, not geometric adjacency.
    reachable = np.asarray(problem.boundary_mask).copy()
    live_pairs = pairs[np.asarray(weights) > 0]
    for _ in range(problem.source.size):
        before = reachable.copy()
        for left, right in live_pairs:
            if reachable[left] or reachable[right]:
                reachable[left] = reachable[right] = True
        if np.array_equal(before, reachable):
            break
    anchored = bool(np.all(reachable))
    background_operator = reduced.operator(
        args.conductance * problem.law.background_conductance * weights
    )
    factor_plan = prepare_sparse_factorization(
        background_operator, SparseFactorizationPolicy("cholesky")
    )
    factor = refresh_sparse_factorization_values(
        factor_plan, background_operator.sparse_storage().values
    )
    background_good = (
        (factor.status == int(SparseFactorizationStatus.SUCCESS))
        & factor.diagnostics.finite
        & anchored
    )
    coverage_assessment = (
        None
        if problem.coverage is None
        else problem.coverage.assess(problem.features.even)
    )
    coverage_good = (
        jnp.asarray(True) if coverage_assessment is None else coverage_assessment.admitted
    )
    validity = _ConservationValidity(coverage_good, background_good)
    residual = _ConservationResidual(
        problem.exterior, problem.features, jnp.asarray(free), validity, problem.law
    )
    initial = (
        problem.boundary_values[free]
        if initial_state is None
        else jnp.asarray(initial_state, dtype=jnp.float64)
    )
    if initial.shape == problem.source.shape:
        initial = initial[free]
    if initial.shape != (free.size,):
        raise ValueError("Initial state must be full-node or reduced equation-node data.")
    pattern = SparsePattern(relation, symmetric=True)
    sparse_plan = compile_sparse_jacobian(
        residual,
        initial,
        source=space,
        target=space,
        sample_args=args,
        structure=pattern,
        compiler="native",
        symmetric=True,
        mode="fwd",
        plan_id=f"{problem.problem_id}:fixed-edge-Jacobian",
    )
    nonlinear_problem = NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        validity=validity,
        trial_validity=validity.trial,
        trial_validity_id=f"{problem.problem_id}:constitutive-domain",
        problem_id=problem.problem_id,
    )
    method = NewtonKrylov(
        jacobian_policy=JacobianPolicy("sparse", sparse_plan=sparse_plan),
        linear_policy=policy,
    )
    native = prepare_nonlinear(
        nonlinear_problem, initial, method=method, termination=termination, args=args
    )
    background = prepare_linear(
        LinearSystem(background_operator, problem_id=f"{problem.problem_id}:background"),
        policy,
    )
    return PreparedMeshfreeConservationSolve(
        problem,
        native,
        derivative,
        residual,
        reduced,
        background,
        factor_plan,
        coverage_assessment,
        jnp.asarray(anchored),
        factor.status,
        factor.diagnostics.minimum_pivot,
    )


__all__ = [
    "MeshfreeConservationAdjoint",
    "MeshfreeConservationLedger",
    "MeshfreeConservationProblem",
    "MeshfreeConservationResult",
    "MeshfreeConstitutiveEvidence",
    "MeshfreeContractionStatus",
    "PreparedMeshfreeConservationSolve",
    "prepare_meshfree_conservation_solve",
]
