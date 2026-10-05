# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Learned edge conservation through the native prepared nonlinear lifecycle.

Scalar laws solve one nodal field; coupled O(3) laws solve a block state with
one packed component vector per node. Both share one background coercivity
assessment selected by the exterior's ``MeshfreeCoercivityPolicy``: a property
audit may be unassessed while a separately admitted nonlinear root converges.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    PreparedLinearSolve,
    refresh as refresh_linear,
    refresh_sparse_factorization_values,
    solve as solve_linear,
    solve_transpose,
    SparseFactorizationPlan,
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
    route_reduce,
    SparseCoordinateOperator,
    SparsePattern,
)
from ...sparse._linear import _SparseStoragePlan
from ...sparse._streamed import (
    PreparedStreamedRelation,
    StreamedPayloadSpec,
    StreamedRelationPlan,
)
from ...typing import Bool, Dim, Float64, Int32, Scalar
from ._constitutive import (
    AbstractCoupledEdgeConstitutiveLaw,
    AbstractEdgeConstitutiveLaw,
    EdgeFrameFeatures,
    LipschitzCoupledEdgeFlux,
    LipschitzEdgeFlux,
    MonotoneCoupledEdgeFlux,
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


class _ConservationComponentDim(Dim):
    """Packed O(3) components of one coupled nodal state."""


class MeshfreeContractionStatus(IntEnum):
    CERTIFIED = 0
    CONTRACT_UNCERTIFIED = 1
    NOT_APPLICABLE = 2


@final
class MeshfreeConstitutiveEvidence(StrictModule, NonTrainableState):
    """Proof hypotheses and native factor status, not a spectral estimate.

    ``background_assessed`` is false when the coercivity policy selected no
    property assessment; the factor status is then ``-1`` and no coercivity or
    contraction certificate is claimed, although a root may still converge.
    """

    __strict_contract__ = True
    positive_metric: Bool[Scalar]
    anchored: Bool[Scalar]
    background_assessed: Bool[Scalar]
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
class MeshfreeComponentConservationLedger(StrictModule, NonTrainableState):
    """Additive balance of every packed component of a coupled state."""

    __strict_contract__ = True
    source_integral: Float64[_ConservationComponentDim]
    outward_boundary_flux: Float64[_ConservationComponentDim]
    balance_defect: Float64[_ConservationComponentDim]
    internal_flux_sum: Float64[_ConservationComponentDim]
    maximum_equation_defect: Float64[_ConservationComponentDim]


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
class MeshfreeCoupledConservationResult(StrictModule):
    """Block-state forward result; the semantics of ``MeshfreeConservationResult``.

    ``state`` has one packed component vector per node and ``edge_flux`` one per
    edge. The coverage assessment concerns the shared immutable edge features
    that condition every component of the law.
    """

    __strict_contract__ = True
    state: Float64[_ConservationNodeDim, _ConservationComponentDim]
    root: NonlinearResult
    edge_flux: Float64[_ConservationEdgeDim, _ConservationComponentDim]
    ledger: MeshfreeComponentConservationLedger
    coverage: EdgeCoverageAssessment | None
    constitutive_evidence: MeshfreeConstitutiveEvidence
    accepted: Bool[Scalar]

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
            "Coupled meshfree conservation forward solve failed; inspect primal, coverage and constitutive evidence.",
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
class MeshfreeCoupledConservationAdjoint(StrictModule):
    """Block adjoint on free equation nodes; failed sensitivities are NaN."""

    __strict_contract__ = True
    primal: MeshfreeCoupledConservationResult
    linear: LinearSolveResult
    value: Float64[_ConservationFreeDim, _ConservationComponentDim]
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
    metric_weights: Float64[_ConservationEdgeDim]
    source: Float64[_ConservationNodeDim]
    boundary_values: Float64[_ConservationNodeDim]


@final
class _CoupledConservationRuntime(StrictModule):
    __strict_contract__ = True
    law_parameters: tuple[Array, ...]
    conductance: Float64[Scalar]
    metric_weights: Float64[_ConservationEdgeDim]
    source: Float64[_ConservationNodeDim, _ConservationComponentDim]
    boundary_values: Float64[_ConservationNodeDim, _ConservationComponentDim]


def _node_array(value: ArrayLike, count: int, name: str, /) -> Array:
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.shape not in ((), (count,)):
        raise ValueError(f"{name} must be scalar or one value per compact node.")
    return jnp.broadcast_to(array, (count,))


def _component_array(value: ArrayLike, count: int, size: int, name: str, /) -> Array:
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.shape not in ((), (size,), (count, size)):
        raise ValueError(
            f"{name} must be scalar, one packed component vector, or one vector per compact node."
        )
    return jnp.broadcast_to(array, (count, size))


def _validated_masks(
    exterior: PreparedMeshfreeExteriorCalculus,
    equation_mask: ArrayLike | None,
    boundary_mask: ArrayLike | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
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
    return equations, boundary


def _validated_features(
    exterior: PreparedMeshfreeExteriorCalculus,
    features: EdgeFrameFeatures | None,
    even_size: int,
    odd_size: int,
    /,
) -> EdgeFrameFeatures:
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
    if features_.even.shape[1] != even_size or features_.odd.shape[1] != odd_size:
        raise ValueError(
            "Typed even/odd feature widths must match the law's declared input schema."
        )
    return features_


def _validated_coverage(
    coverage: EdgeFeatureCoverage | None, features: EdgeFrameFeatures, /
) -> None:
    if coverage is not None and (
        not isinstance(coverage, EdgeFeatureCoverage)
        or coverage.feature_names != features.even_names
    ):
        raise ValueError("Coverage must name exactly the immutable even-feature layout.")


def _parameter_owner(
    parameter_bindings: tuple[ParameterBinding, ...], /
) -> PreparedParameters:
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
            or target.name
            not in ("law", "conductance", "metric_weights", "source", "boundary_values")
            for target in binding.targets
        ):
            raise ValueError(
                "Conservation parameters bind refresh-only meshfree law/conductance/metric_weights/source/boundary_values inputs."
            )
        for target in binding.targets:
            required_role = (
                "coefficient"
                if target.name in ("law", "conductance", "metric_weights")
                else "source"
                if target.name == "source"
                else "boundary"
            )
            if binding.role != required_role:
                raise ValueError(
                    "Parameter physical role must agree with its runtime input."
                )
    return PreparedParameters(parameter_bindings, (), ())


def _problem_identity(
    kind: str,
    problem_id: str,
    exterior: PreparedMeshfreeExteriorCalculus,
    equations: np.ndarray,
    /,
) -> str:
    if (
        not isinstance(problem_id, str)
        or not problem_id
        or problem_id.strip() != problem_id
    ):
        raise ValueError("problem_id must be a nonempty stripped scientific identity.")
    return canonical_fingerprint(
        {
            "kind": kind,
            "owner": problem_id,
            "geometry_source": exterior.incidence.source.space_id,
            "edge_source": exterior.incidence.target.space_id,
            "geometry": array_tree_fingerprint((exterior.points, exterior.pairs)),
            "equations": equations.tolist(),
        }
    )


def _require_admitted_metric(exterior: PreparedMeshfreeExteriorCalculus, /) -> None:
    if not bool(exterior.metric_result.accepted) or not bool(
        exterior.metric_result.hilbert_admitted
    ):
        raise ValueError(
            "Learned dissipative conservation requires an admitted strictly positive metric; signed pairings are refused."
        )


def _bound_inputs(
    parameters: PreparedParameters,
    values: Mapping[str, object] | None,
    law: object | None,
    /,
) -> Mapping[str, object]:
    bound = parameters.bind(("meshfree",), None, values).arguments["meshfree"]
    inputs = {} if bound is None else bound
    if not isinstance(inputs, Mapping):
        raise TypeError(
            "Native parameter binding must yield a meshfree runtime input mapping."
        )
    if law is not None and "law" in inputs:
        raise ValueError(
            "A bound law parameter and an explicit law refresh cannot both supply the same physical input."
        )
    return inputs


def _law_leaves(
    law: AbstractArrayModel, reference: AbstractArrayModel, /
) -> tuple[Array, ...]:
    """Numeric law leaves after refusing changed activations, transforms or traits."""
    if isinstance(law, MonotoneEdgeConductance | MonotoneCoupledEdgeFlux):
        law.potential.input_convex_certificate()
    numeric_law, law_traits = eqx.partition(law, eqx.is_array)
    _, prepared_traits = eqx.partition(reference, eqx.is_array)
    if eqx.tree_equal(law_traits, prepared_traits) is not True:
        raise ValueError(
            "Constitutive metadata changed; prepare a new solve for different activation, transforms or static law traits."
        )
    return tuple(jax.tree_util.tree_leaves(numeric_law))


def _bind_numeric_law[LawT: AbstractArrayModel](
    reference: LawT, leaves: tuple[Array, ...], /
) -> LawT:
    """Reconstruct canonical law metadata around the actual numerical leaves."""
    numeric_template, traits = eqx.partition(reference, eqx.is_array)
    numeric_law = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(numeric_template), leaves
    )
    return eqx.combine(numeric_law, traits)


def _runtime_value(item: object, /) -> Array:
    if isinstance(item, AbstractArrayModel):
        if item.in_size != "scalar":
            raise ValueError(
                "A runtime parameter model must declare a scalar constant input."
            )
        return jnp.asarray(item(jnp.zeros((), dtype=jnp.float64)), dtype=jnp.float64)
    return jnp.asarray(item, dtype=jnp.float64)


def _runtime_conductance(inputs: Mapping[str, object], /) -> Array:
    conductance = _runtime_value(inputs.get("conductance", 1.0))
    if conductance.shape != ():
        raise ValueError(
            "The constitutive multiplier must be scalar; spatial context belongs to typed external edge features."
        )
    return conductance


def _runtime_metric_weights(
    inputs: Mapping[str, object], exterior: PreparedMeshfreeExteriorCalculus, /
) -> Array:
    """Runtime edge metric in compact edge order; the prepared metric by default."""
    weights = _runtime_value(inputs.get("metric_weights", exterior.metric_result.weights))
    if weights.shape != exterior.lengths.shape:
        raise ValueError(
            "Runtime metric weights must hold one value per compact exterior edge."
        )
    return weights


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
        _require_admitted_metric(exterior)
        count = exterior.points.shape[0]
        equations, boundary = _validated_masks(exterior, equation_mask, boundary_mask)
        features_ = _validated_features(exterior, features, law.even_size, law.odd_size)
        _validated_coverage(coverage, features_)
        source_ = _node_array(source, count, "source")
        values_ = _node_array(boundary_values, count, "boundary_values")
        if not np.all(np.isfinite(np.asarray(source_))) or not np.all(
            np.isfinite(np.asarray(values_))
        ):
            raise ValueError("Prepared source and boundary values must be finite.")
        identity = _problem_identity(
            "meshfree-conservation", problem_id, exterior, equations
        )
        parameter_owner = _parameter_owner(parameter_bindings)
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
        inputs = _bound_inputs(self.parameters, parameters, law)
        law_ = inputs.get("law", self.law if law is None else law)
        if (
            not isinstance(law_, AbstractEdgeConstitutiveLaw)
            or type(law_) is not type(self.law)
            or law_.in_size != self.law.in_size
        ):
            raise ValueError(
                "Runtime constitutive refresh must preserve the law family and feature schema."
            )
        law_parameters = _law_leaves(law_, self.law)
        conductance = _runtime_conductance(inputs)
        source = _node_array(
            _runtime_value(inputs.get("source", self.source)),
            self.source.size,
            "source",
        )
        boundary = _node_array(
            _runtime_value(inputs.get("boundary_values", self.boundary_values)),
            self.source.size,
            "boundary_values",
        )
        return _ConservationRuntime(
            law_parameters,
            conductance,
            _runtime_metric_weights(inputs, self.exterior),
            source,
            boundary,
        )


@final
class MeshfreeCoupledConservationProblem(StrictModule, NonTrainableState):
    """Block equations ``(B.T kron I) a F(B u) = volumes * source`` for a coupled law.

    ``u`` holds one packed O(3) component vector per node and ``F`` is an
    ``AbstractCoupledEdgeConstitutiveLaw`` acting on endpoint differences in the
    oriented 3-D edge frame. Masks, boundary prescription, immutable external
    context and parameter semantics match ``MeshfreeConservationProblem``;
    source and boundary values are scalar, one packed vector, or one vector per
    node.
    """

    exterior: PreparedMeshfreeExteriorCalculus
    law: AbstractCoupledEdgeConstitutiveLaw
    features: EdgeFrameFeatures
    __strict_contract__ = True
    source: Float64[_ConservationNodeDim, _ConservationComponentDim]
    boundary_values: Float64[_ConservationNodeDim, _ConservationComponentDim]
    boundary_mask: Bool[_ConservationNodeDim]
    equation_mask: Bool[_ConservationNodeDim]
    coverage: EdgeFeatureCoverage | None
    parameters: PreparedParameters
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        exterior: PreparedMeshfreeExteriorCalculus,
        law: AbstractCoupledEdgeConstitutiveLaw,
        /,
        *,
        features: EdgeFrameFeatures | None = None,
        source: ArrayLike = 0.0,
        boundary_values: ArrayLike = 0.0,
        boundary_mask: ArrayLike | None = None,
        equation_mask: ArrayLike | None = None,
        coverage: EdgeFeatureCoverage | None = None,
        parameter_bindings: tuple[ParameterBinding, ...] = (),
        problem_id: str = "meshfree-coupled-conservation",
    ) -> None:
        if not isinstance(exterior, PreparedMeshfreeExteriorCalculus) or not isinstance(
            law, AbstractCoupledEdgeConstitutiveLaw
        ):
            raise TypeError(
                "Coupled conservation requires native prepared exterior geometry and a coupled edge law."
            )
        _require_admitted_metric(exterior)
        count = exterior.points.shape[0]
        size = law.representation.packed_size
        equations, boundary = _validated_masks(exterior, equation_mask, boundary_mask)
        features_ = _validated_features(exterior, features, law.even_size, law.odd_size)
        if features_.spatial_dimension != 3:
            raise ValueError(
                "Coupled O(3) conservation requires a three-dimensional Cartesian edge frame."
            )
        _validated_coverage(coverage, features_)
        source_ = _component_array(source, count, size, "source")
        values_ = _component_array(boundary_values, count, size, "boundary_values")
        if not np.all(np.isfinite(np.asarray(source_))) or not np.all(
            np.isfinite(np.asarray(values_))
        ):
            raise ValueError("Prepared source and boundary values must be finite.")
        identity = _problem_identity(
            f"meshfree-coupled-conservation:{size}", problem_id, exterior, equations
        )
        parameter_owner = _parameter_owner(parameter_bindings)
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

    @property
    def component_count(self) -> int:
        return self.source.shape[1]

    def runtime(
        self,
        *,
        parameters: Mapping[str, object] | None = None,
        law: AbstractCoupledEdgeConstitutiveLaw | None = None,
    ) -> _CoupledConservationRuntime:
        inputs = _bound_inputs(self.parameters, parameters, law)
        law_ = inputs.get("law", self.law if law is None else law)
        if (
            not isinstance(law_, AbstractCoupledEdgeConstitutiveLaw)
            or type(law_) is not type(self.law)
            or law_.in_size != self.law.in_size
            or law_.representation.packed_size != self.component_count
        ):
            raise ValueError(
                "Runtime constitutive refresh must preserve the law family, representation and feature schema."
            )
        law_parameters = _law_leaves(law_, self.law)
        count, size = self.source.shape
        source = _component_array(
            _runtime_value(inputs.get("source", self.source)), count, size, "source"
        )
        boundary = _component_array(
            _runtime_value(inputs.get("boundary_values", self.boundary_values)),
            count,
            size,
            "boundary_values",
        )
        return _CoupledConservationRuntime(
            law_parameters,
            _runtime_conductance(inputs),
            _runtime_metric_weights(inputs, self.exterior),
            source,
            boundary,
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
        args: _ConservationRuntime | _CoupledConservationRuntime,
        /,
    ) -> Array:
        del state, residual, auxiliary
        return self.admitted(args)

    def admitted(
        self, args: _ConservationRuntime | _CoupledConservationRuntime, /
    ) -> Array:
        # Runtime metric weights are admitted only strictly positive: a
        # corrected metric is never clipped into a fictitious positive pairing.
        return (
            self.coverage_admitted
            & self.background_admitted
            & jnp.isfinite(args.conductance)
            & (args.conductance > 0)
            & jnp.all(jnp.isfinite(args.metric_weights) & (args.metric_weights > 0))
            & jnp.all(jnp.isfinite(args.source))
            & jnp.all(jnp.isfinite(args.boundary_values))
        )

    def trial(
        self, state: Array, args: _ConservationRuntime | _CoupledConservationRuntime, /
    ) -> Array:
        del state
        return self.admitted(args)


def _scalar_edge_flux(
    parameters: tuple[AbstractEdgeConstitutiveLaw, Array],
    first: Array,
    second: Array,
    edge: tuple[Array, Array, Array],
    /,
) -> tuple[Array, Array]:
    """Metric-weighted law flux of one canonical edge, from its endpoint jump."""
    law, conductance = parameters
    metric, even, odd = edge
    flux = conductance * metric * law.edge_flux(second - first, even, odd)
    return flux, flux


def _coupled_edge_flux(
    parameters: tuple[AbstractCoupledEdgeConstitutiveLaw, Array],
    first: Array,
    second: Array,
    edge: tuple[Array, Array, Array, Array],
    /,
) -> tuple[Array, Array]:
    """Metric-weighted packed coupled flux of one canonical edge."""
    law, conductance = parameters
    metric, tangent, even, odd = edge
    flux = conductance * metric * law.edge_flux(second - first, tangent, even, odd)
    return flux, flux


def _received_flux(parameters: object, node: Array, aggregate: Array, /) -> Array:
    """Second-endpoint epilogue: the complete incoming flux sum of the node."""
    del parameters, node
    return aggregate


def _edge_balance(
    edges: PreparedStreamedRelation,
    reaction: EdgeRelation,
    edge_function: Callable[..., tuple[Array, Array]],
    parameters: object,
    full: Array,
    edge_data: tuple[Array, ...],
    /,
) -> tuple[Array, Array]:
    """Integrated ``B.T f`` and per-edge flux ``f``; each edge law is evaluated once.

    The streamed receiver sum carries ``+f`` to every canonical edge's second
    endpoint; the returned edge flux itself carries the exactly opposite ``-f``
    to its first endpoint, so action and reaction never depend on law parity.
    """
    spec = jax.ShapeDtypeStruct(full.shape[1:], jnp.float64)
    result = edges.evaluate(
        StreamedPayloadSpec(message=spec, output=spec, edge_output=spec),
        edge_function,
        _received_flux,
        parameters,
        full,
        full,
        edge_data,
    )
    flux = result.edge_outputs
    if flux is None:
        raise RuntimeError("The streamed relation omitted its requested edge flux.")
    return result.receiver_outputs - route_reduce(reaction, flux), flux


def _prepared_edges(
    problem: MeshfreeConservationProblem | MeshfreeCoupledConservationProblem,
    execution: StreamedRelationPlan | None,
    /,
) -> tuple[PreparedStreamedRelation, EdgeRelation]:
    """Streamed canonical edge schedule (first -> second endpoint) and its reaction."""
    if execution is None:
        execution = StreamedRelationPlan()
    elif not isinstance(execution, StreamedRelationPlan):
        raise TypeError("execution must be a StreamedRelationPlan or None.")
    pairs = np.asarray(problem.exterior.pairs)
    count = problem.source.shape[0]
    relation = EdgeRelation(
        pairs[:, 0], pairs[:, 1], source_size=count, target_size=count
    )
    prepared = execution.prepare(
        relation,
        owner_id=f"{problem.exterior.incidence.source.space_id}:constitutive-edges",
    )
    return prepared, relation.transpose()


@final
class _ConservationResidual(StrictModule, NonTrainableState):
    __strict_contract__ = True
    exterior: PreparedMeshfreeExteriorCalculus
    features: EdgeFrameFeatures
    free_indices: Int32[_ConservationFreeDim]
    validity: _ConservationValidity
    law_reference: AbstractEdgeConstitutiveLaw
    edges: PreparedStreamedRelation
    reaction: EdgeRelation

    def bound_law(self, args: _ConservationRuntime, /) -> AbstractEdgeConstitutiveLaw:
        return _bind_numeric_law(self.law_reference, args.law_parameters)

    def reconstruct(self, state: Array, args: _ConservationRuntime, /) -> Array:
        return args.boundary_values.at[self.free_indices].set(state)

    def full_balance(
        self, full: Array, args: _ConservationRuntime, /
    ) -> tuple[Array, Array]:
        """Integrated law flux ``B.T f`` of a full nodal field and edge flux ``f``."""
        return _edge_balance(
            self.edges,
            self.reaction,
            _scalar_edge_flux,
            (self.bound_law(args), args.conductance),
            full,
            (args.metric_weights, self.features.even, self.features.odd),
        )

    def integrated_flux(
        self, state: Array, args: _ConservationRuntime, /
    ) -> tuple[Array, Array]:
        integrated, edge = self.full_balance(self.reconstruct(state, args), args)
        return integrated, -edge

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
class _CoupledConservationResidual(StrictModule, NonTrainableState):
    """Flattened block residual over free nodes, component index fastest."""

    __strict_contract__ = True
    exterior: PreparedMeshfreeExteriorCalculus
    features: EdgeFrameFeatures
    free_indices: Int32[_ConservationFreeDim]
    validity: _ConservationValidity
    law_reference: AbstractCoupledEdgeConstitutiveLaw
    edges: PreparedStreamedRelation
    reaction: EdgeRelation

    def bound_law(
        self, args: _CoupledConservationRuntime, /
    ) -> AbstractCoupledEdgeConstitutiveLaw:
        return _bind_numeric_law(self.law_reference, args.law_parameters)

    def reconstruct(self, state: Array, args: _CoupledConservationRuntime, /) -> Array:
        free = state.reshape((self.free_indices.size, args.source.shape[1]))
        return args.boundary_values.at[self.free_indices].set(free)

    def jumps(self, full: Array, /) -> Array:
        return jax.vmap(self.exterior.gradient, in_axes=1, out_axes=1)(full)

    def integrated_flux(
        self, state: Array, args: _CoupledConservationRuntime, /
    ) -> tuple[Array, Array]:
        integrated, edge = _edge_balance(
            self.edges,
            self.reaction,
            _coupled_edge_flux,
            (self.bound_law(args), args.conductance),
            self.reconstruct(state, args),
            (
                args.metric_weights,
                self.features.tangents,
                self.features.even,
                self.features.odd,
            ),
        )
        return integrated, -edge

    def __call__(self, state: Array, args: _CoupledConservationRuntime, /) -> Array:
        def evaluate(_operand: None) -> Array:
            integrated = self.integrated_flux(state, args)[0]
            defect = integrated - self.exterior.node_volumes[:, None] * args.source
            return defect[self.free_indices].reshape(-1)

        return jax.lax.cond(
            self.validity.admitted(args), evaluate, lambda _: jnp.zeros_like(state), None
        )


@final
class _ReducedStiffness(StrictModule, NonTrainableState):
    """Dirichlet-reduced edge stiffness on the routes of ``exterior.stiffness``.

    Values follow the canonical per-edge route order ``(i,i), (j,j), (i,j),
    (j,i)`` with signs ``(+,+,-,-)``; building them directly keeps the operator
    identity static, so traced runtime metrics never fingerprint array content.
    """

    __strict_contract__ = True
    relation: EdgeRelation
    routes: Int32[_ConservationRouteDim]
    space: ArraySpace
    storage: _SparseStoragePlan

    def operator(self, conductances: Array, /) -> SparseCoordinateOperator:
        signed = conductances[:, None] * jnp.asarray((1.0, 1.0, -1.0, -1.0))
        properties = OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "construction"}
        )
        return SparseCoordinateOperator(
            self.relation,
            signed.reshape(-1)[self.routes],
            source=self.space,
            target=self.space,
            properties=properties,
            storage_plan=self.storage,
            operator_id=f"{self.space.space_id}:stiffness",
        )


@final
class _ReducedBlockStiffness(StrictModule, NonTrainableState):
    """``(B.T kron I) diag(a) blockdiag(J_e) (B kron I)`` on free block coordinates."""

    __strict_contract__ = True
    relation: EdgeRelation
    edges: Int32[_ConservationRouteDim]
    signs: Float64[_ConservationRouteDim]
    space: ArraySpace
    storage: _SparseStoragePlan

    def operator(self, blocks: Array, /) -> SparseCoordinateOperator:
        values = (self.signs[:, None, None] * blocks[self.edges]).reshape(-1)
        return SparseCoordinateOperator(
            self.relation,
            values,
            source=self.space,
            target=self.space,
            storage_plan=self.storage,
            operator_id=f"{self.space.space_id}:block-stiffness",
        )


def _reduced_routes(
    pairs: np.ndarray, free: np.ndarray, count: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stiffness routes (in ``exterior.stiffness`` order) between free nodes."""
    inverse = np.full(count, -1, dtype=np.int32)
    inverse[free] = np.arange(free.size, dtype=np.int32)
    i, j = pairs[:, 0], pairs[:, 1]
    rows = np.stack((i, j, i, j), axis=1).reshape(-1)
    columns = np.stack((i, j, j, i), axis=1).reshape(-1)
    routes = np.flatnonzero((inverse[rows] >= 0) & (inverse[columns] >= 0)).astype(
        np.int32
    )
    return routes, inverse[rows[routes]], inverse[columns[routes]]


def _anchored(boundary: np.ndarray, pairs: np.ndarray, weights: np.ndarray, /) -> bool:
    """Reachability from prescribed nodes through strictly positive physical weights."""
    reachable = boundary.copy()
    live = pairs[weights > 0]
    left, right = live[:, 0], live[:, 1]
    for _ in range(boundary.size):
        touched = reachable[left] | reachable[right]
        updated = reachable.copy()
        updated[left[touched]] = True
        updated[right[touched]] = True
        if np.array_equal(updated, reachable):
            break
        reachable = updated
    return bool(np.all(reachable))


@final
class _BackgroundAssessment(StrictModule, NonTrainableState):
    """Policy-selected property assessment of the reduced background stiffness."""

    __strict_contract__ = True
    plan: SparseFactorizationPlan | None
    anchored: Bool[Scalar]
    factor_status: Int32[Scalar]
    minimum_pivot: Float64[Scalar]
    admitted: Bool[Scalar]
    source: str = eqx.field(static=True)

    @property
    def assessed(self) -> Array:
        return jnp.asarray(self.plan is not None)

    @property
    def certified(self) -> Array:
        return (
            self.assessed
            & (self.factor_status == int(SparseFactorizationStatus.SUCCESS))
            & self.anchored
        )

    def refreshed(
        self, operator: SparseCoordinateOperator, weights: Array, /
    ) -> _BackgroundAssessment:
        """Numeric refresh for runtime metric weights; never re-analysed.

        Prepared anchoring is reachability through the admitted positive metric;
        it carries over exactly when every runtime weight is positive.
        """
        anchored = self.anchored & jnp.all(jnp.isfinite(weights) & (weights > 0))
        plan = self.plan
        if plan is None:
            return eqx.tree_at(
                lambda owner: (owner.anchored, owner.admitted),
                self,
                (anchored, anchored),
            )
        factor = refresh_sparse_factorization_values(
            plan, operator.sparse_storage().values
        )
        return _BackgroundAssessment(
            plan=plan,
            anchored=anchored,
            factor_status=factor.status.astype(jnp.int32),
            minimum_pivot=factor.diagnostics.minimum_pivot,
            admitted=(factor.status == int(SparseFactorizationStatus.SUCCESS))
            & factor.diagnostics.finite
            & anchored,
            source=self.source,
        )


def _assess_background(
    exterior: PreparedMeshfreeExteriorCalculus,
    operator: SparseCoordinateOperator,
    free: np.ndarray,
    boundary: np.ndarray,
    /,
) -> _BackgroundAssessment:
    """Charge the selected factor to the exterior's shared coercivity policy.

    An unassessed policy factors nothing: the root is then admitted on anchoring
    alone and every coercivity claim is unavailable.
    """
    pairs = np.asarray(exterior.pairs)
    anchored = _anchored(boundary, pairs, np.asarray(exterior.metric_result.weights))
    policy = exterior.coercivity_policy
    plan = policy.prepare(operator, np.asarray(exterior.points)[free])
    if plan is None:
        return _BackgroundAssessment(
            plan=None,
            anchored=jnp.asarray(anchored),
            factor_status=jnp.asarray(-1, dtype=jnp.int32),
            minimum_pivot=jnp.asarray(jnp.nan, dtype=jnp.float64),
            admitted=jnp.asarray(anchored),
            source=policy.source,
        )
    factor = refresh_sparse_factorization_values(plan, operator.sparse_storage().values)
    admitted = (
        (factor.status == int(SparseFactorizationStatus.SUCCESS))
        & factor.diagnostics.finite
        & anchored
    )
    return _BackgroundAssessment(
        plan=plan,
        anchored=jnp.asarray(anchored),
        factor_status=factor.status.astype(jnp.int32),
        minimum_pivot=factor.diagnostics.minimum_pivot,
        admitted=admitted,
        source=policy.source,
    )


def _evidence(
    background: _BackgroundAssessment,
    positive: Array,
    lower: Array,
    coercive: Array,
    status: Array,
    bound: Array,
    /,
    *,
    norm: str,
) -> MeshfreeConstitutiveEvidence:
    return MeshfreeConstitutiveEvidence(
        positive,
        background.anchored,
        background.assessed,
        background.factor_status,
        background.minimum_pivot,
        coercive,
        lower,
        status,
        bound,
        norm,
        (
            "fixed-external-context",
            "same-positive-edge-metric",
            background.source,
            "positive-runtime-conductance",
        ),
    )


def _default_linear_policy(size: int, /) -> LinearSolvePolicy:
    return LinearSolvePolicy(
        GMRES(restart=min(32, max(1, size))),
        tolerance=TolerancePolicy(
            relative=1e-10, absolute=1e-12, max_steps=max(64, 2 * size)
        ),
    )


def _resolved_policies(
    size: int,
    linear_policy: LinearSolvePolicy | None,
    derivative_policy: ImplicitRootDerivativePolicy | None,
    /,
) -> tuple[LinearSolvePolicy, ImplicitRootDerivativePolicy]:
    policy = _default_linear_policy(size) if linear_policy is None else linear_policy
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
    return policy, derivative


def _coverage_state(
    coverage: EdgeFeatureCoverage | None, features: EdgeFrameFeatures, /
) -> tuple[EdgeCoverageAssessment | None, Array]:
    assessment = None if coverage is None else coverage.assess(features.even)
    return assessment, (jnp.asarray(True) if assessment is None else assessment.admitted)


def _accepted_root(root: NonlinearResult, accepted: Array, /) -> NonlinearResult:
    status = jnp.where(
        root.successful & ~accepted,
        int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
        root.status,
    )
    return eqx.tree_at(lambda value: value.status, root, status)


@final
class PreparedMeshfreeConservationSolve(StrictModule, NonTrainableState):
    """Native nonlinear template, fixed sparse edge pattern and explicit background.

    The certificate's norm is ``||v||_A0`` for the Dirichlet-reduced background
    stiffness, not the unweighted Euclidean norm. Fixed external context and the
    same edge metric give ``||T(u)-T(v)||_A0 <= (L/b)||u-v||_A0`` by Cauchy-Schwarz.
    No claim is made for context that depends on the solved nodal field.
    """

    problem: MeshfreeConservationProblem
    native: PreparedNonlinearSolve
    derivative_policy: ImplicitRootDerivativePolicy
    residual: _ConservationResidual
    reduced: _ReducedStiffness
    background: PreparedLinearSolve
    assessment: _BackgroundAssessment
    coverage_assessment: EdgeCoverageAssessment | None

    def background_assessment(
        self, args: _ConservationRuntime, /
    ) -> _BackgroundAssessment:
        """Prepared coercivity factor refreshed at the runtime metric and law."""
        b = self.residual.bound_law(args).background_conductance
        return self.assessment.refreshed(
            self.reduced.operator(args.conductance * b * args.metric_weights),
            args.metric_weights,
        )

    def refresh(
        self,
        *,
        parameters: Mapping[str, object] | None = None,
        law: AbstractEdgeConstitutiveLaw | None = None,
        initial_state: ArrayLike | None = None,
    ) -> PreparedNonlinearSolve:
        args = self.problem.runtime(parameters=parameters, law=law)
        return self._refresh_runtime(
            args, initial_state, self.background_assessment(args).admitted
        )

    def _refresh_runtime(
        self,
        args: _ConservationRuntime,
        initial_state: ArrayLike | None,
        background_admitted: Array,
        /,
    ) -> PreparedNonlinearSolve:
        state = (
            self.native.state
            if initial_state is None
            else jnp.asarray(initial_state, dtype=jnp.float64)
        )
        if state.shape == self.problem.source.shape:
            state = state[self.residual.free_indices]
        validity = _ConservationValidity(
            jnp.asarray(True)
            if self.coverage_assessment is None
            else self.coverage_assessment.admitted,
            background_admitted,
        )
        problem = eqx.tree_at(
            lambda candidate: candidate.validity_function, self.native.problem, validity
        )
        return refresh_nonlinear(self.native, problem, state, args=args)

    def background_solve(
        self, args: _ConservationRuntime, /, *, state: ArrayLike | None = None
    ) -> LinearSolveResult:
        """One native background diffusion solve; optionally one true Picard step."""
        law = self.residual.bound_law(args)
        b = law.background_conductance
        weights = args.metric_weights
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
            # B.T(c w (F - b D)) = B.T(c w F) - B.T diag(c b w) B u.
            nonlinear = self.residual.full_balance(full, args)[0] - full_operator.mv(full)
            rhs = rhs - nonlinear[self.residual.free_indices]
        return solve_linear(prepared, rhs)

    def constitutive_evidence(
        self,
        args: _ConservationRuntime,
        /,
        background: _BackgroundAssessment | None = None,
    ) -> MeshfreeConstitutiveEvidence:
        background_ = (
            self.background_assessment(args) if background is None else background
        )
        factor_good = background_.certified
        positive = (
            jnp.isfinite(args.conductance)
            & (args.conductance > 0)
            & jnp.all(jnp.isfinite(args.metric_weights) & (args.metric_weights > 0))
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
        return _evidence(
            background_,
            positive,
            lower,
            coercive,
            status,
            bound,
            norm="Dirichlet-background-energy",
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
        background = self.background_assessment(args)
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
            background.admitted
            if background_result is None
            else background.admitted & jnp.all(background_result.successful),
        )
        root = (
            implicit_root_result(native, derivative_policy=self.derivative_policy)
            if implicit
            else solve_prepared_nonlinear(native)
        )
        evidence = self.constitutive_evidence(args, background)
        accepted = root.successful & background.admitted
        if background_result is not None:
            accepted = accepted & jnp.all(background_result.successful)
        if self.coverage_assessment is not None:
            accepted = accepted & self.coverage_assessment.admitted
        root = _accepted_root(root, accepted)
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
            * args.metric_weights
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
    execution: StreamedRelationPlan | None = None,
) -> PreparedMeshfreeConservationSolve:
    """Prepare native sparse compressed Jacobian and reusable nonlinear/linear plans.

    The background stiffness assessment is selected and charged by
    ``problem.exterior.coercivity_policy``. ``execution`` selects the streamed
    canonical edge schedule on which every residual evaluates each edge law once.
    """
    if not isinstance(problem, MeshfreeConservationProblem):
        raise TypeError("problem must be a MeshfreeConservationProblem.")
    free = np.flatnonzero(np.asarray(problem.equation_mask)).astype(np.int32)
    pairs = np.asarray(problem.exterior.pairs)
    routes, rows, columns = _reduced_routes(pairs, free, problem.source.size)
    relation = EdgeRelation(columns, rows, source_size=free.size, target_size=free.size)
    space = ArraySpace(
        (free.size,), dtype=jnp.float64, space_id=f"{problem.problem_id}:free-state"
    )
    storage = _SparseStoragePlan(relation)
    reduced = _ReducedStiffness(relation, jnp.asarray(routes), space, storage)
    args = problem.runtime(parameters=parameters)
    policy, derivative = _resolved_policies(free.size, linear_policy, derivative_policy)
    background_operator = reduced.operator(
        args.conductance * problem.law.background_conductance * args.metric_weights
    )
    assessment = _assess_background(
        problem.exterior, background_operator, free, np.asarray(problem.boundary_mask)
    )
    coverage_assessment, coverage_good = _coverage_state(
        problem.coverage, problem.features
    )
    validity = _ConservationValidity(coverage_good, assessment.admitted)
    residual = _ConservationResidual(
        problem.exterior,
        problem.features,
        jnp.asarray(free),
        validity,
        problem.law,
        *_prepared_edges(problem, execution),
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
    sparse_plan = compile_sparse_jacobian(
        residual,
        initial,
        source=space,
        target=space,
        sample_args=args,
        structure=SparsePattern(relation, symmetric=True),
        compiler="native",
        symmetric=True,
        mode="fwd",
        plan_id=f"{problem.problem_id}:fixed-edge-Jacobian",
    )
    native = _prepared_nonlinear(
        problem.problem_id,
        residual,
        validity,
        space,
        initial,
        args,
        JacobianPolicy("sparse", sparse_plan=sparse_plan),
        policy,
        termination,
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
        assessment,
        coverage_assessment,
    )


def _prepared_nonlinear[RuntimeT: (_ConservationRuntime, _CoupledConservationRuntime)](
    problem_id: str,
    residual: Callable[[Array, RuntimeT], Array],
    validity: _ConservationValidity,
    space: ArraySpace,
    initial: Array,
    args: RuntimeT,
    jacobian: JacobianPolicy,
    policy: LinearSolvePolicy,
    termination: NonlinearTermination | None,
    /,
) -> PreparedNonlinearSolve:
    nonlinear_problem = NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        validity=validity,
        trial_validity=validity.trial,
        trial_validity_id=f"{problem_id}:constitutive-domain",
        problem_id=problem_id,
    )
    method = NewtonKrylov(jacobian_policy=jacobian, linear_policy=policy)
    return prepare_nonlinear(
        nonlinear_problem, initial, method=method, termination=termination, args=args
    )


@final
class PreparedMeshfreeCoupledConservationSolve(StrictModule, NonTrainableState):
    """Native NewtonKrylov template for a coupled law on a fixed block edge pattern.

    Coercivity holds in the component-blocked background energy
    ``||v||_(A0 kron I)``: a law with whole-state strong monotonicity ``m > 0``
    on a positive anchored metric with a certified SPD background satisfies
    ``<R(u)-R(v), u-v> >= m ||u-v||^2_(A0 kron I)``. A Lipschitz coupled law
    additionally reports the Picard contraction bound ``L/b`` of its learned
    perturbation. The block Jacobian is symmetric only for potential-gradient
    laws; Lipschitz laws use the nonsymmetric pattern.
    """

    problem: MeshfreeCoupledConservationProblem
    native: PreparedNonlinearSolve
    derivative_policy: ImplicitRootDerivativePolicy
    residual: _CoupledConservationResidual
    block: _ReducedBlockStiffness
    background_stiffness: _ReducedStiffness
    assessment: _BackgroundAssessment
    coverage_assessment: EdgeCoverageAssessment | None

    def refresh(
        self,
        *,
        parameters: Mapping[str, object] | None = None,
        law: AbstractCoupledEdgeConstitutiveLaw | None = None,
        initial_state: ArrayLike | None = None,
    ) -> PreparedNonlinearSolve:
        args = self.problem.runtime(parameters=parameters, law=law)
        return self._refresh_runtime(
            args, initial_state, self.background_assessment(args).admitted
        )

    def background_assessment(
        self, args: _CoupledConservationRuntime, /
    ) -> _BackgroundAssessment:
        """Prepared coercivity factor refreshed at the runtime metric and law."""
        b = self.residual.bound_law(args).background_conductance
        return self.assessment.refreshed(
            self.background_stiffness.operator(
                args.conductance * b * args.metric_weights
            ),
            args.metric_weights,
        )

    def _refresh_runtime(
        self,
        args: _CoupledConservationRuntime,
        initial_state: ArrayLike | None,
        background_admitted: Array,
        /,
    ) -> PreparedNonlinearSolve:
        validity = _ConservationValidity(
            jnp.asarray(True)
            if self.coverage_assessment is None
            else self.coverage_assessment.admitted,
            background_admitted,
        )
        problem = eqx.tree_at(
            lambda candidate: candidate.validity_function, self.native.problem, validity
        )
        return refresh_nonlinear(
            self.native, problem, self._initial(initial_state), args=args
        )

    def _initial(self, initial_state: ArrayLike | None, /) -> Array:
        if initial_state is None:
            return self.native.state
        state = jnp.asarray(initial_state, dtype=jnp.float64)
        if state.shape == self.problem.source.shape:
            state = state[self.residual.free_indices]
        return state.reshape(self.native.state.shape)

    def constitutive_evidence(
        self,
        args: _CoupledConservationRuntime,
        /,
        background: _BackgroundAssessment | None = None,
    ) -> MeshfreeConstitutiveEvidence:
        background_ = (
            self.background_assessment(args) if background is None else background
        )
        factor_good = background_.certified
        positive = (
            jnp.isfinite(args.conductance)
            & (args.conductance > 0)
            & jnp.all(jnp.isfinite(args.metric_weights) & (args.metric_weights > 0))
        )
        law = self.residual.bound_law(args)
        monotonicity = jnp.asarray(law.monotonicity_lower_bound(), dtype=jnp.float64)
        coercive = (
            factor_good & positive & jnp.isfinite(monotonicity) & (monotonicity > 0)
        )
        lower = jnp.where(factor_good & positive, monotonicity, jnp.nan)
        if isinstance(law, LipschitzCoupledEdgeFlux):
            bound = (
                jnp.asarray(law.certified_lipschitz_bound(), dtype=jnp.float64)
                / law.background_conductance
            )
            status = jnp.where(
                factor_good & positive & jnp.isfinite(bound) & (bound < 1),
                int(MeshfreeContractionStatus.CERTIFIED),
                int(MeshfreeContractionStatus.CONTRACT_UNCERTIFIED),
            ).astype(jnp.int32)
        else:
            bound = jnp.asarray(jnp.nan, dtype=jnp.float64)
            status = jnp.asarray(
                int(MeshfreeContractionStatus.NOT_APPLICABLE), dtype=jnp.int32
            )
        return _evidence(
            background_,
            positive,
            lower,
            coercive,
            status,
            bound,
            norm="component-blocked-Dirichlet-background-energy",
        )

    def solve(
        self,
        *,
        parameters: Mapping[str, object] | None = None,
        law: AbstractCoupledEdgeConstitutiveLaw | None = None,
        initial_state: ArrayLike | None = None,
        implicit: bool = True,
    ) -> MeshfreeCoupledConservationResult:
        args = self.problem.runtime(parameters=parameters, law=law)
        background = self.background_assessment(args)
        native = self._refresh_runtime(args, initial_state, background.admitted)
        root = (
            implicit_root_result(native, derivative_policy=self.derivative_policy)
            if implicit
            else solve_prepared_nonlinear(native)
        )
        evidence = self.constitutive_evidence(args, background)
        accepted = root.successful & background.admitted
        if self.coverage_assessment is not None:
            accepted = accepted & self.coverage_assessment.admitted
        root = _accepted_root(root, accepted)
        full = self.residual.reconstruct(root.state, args)
        integrated, edge_flux = jax.lax.cond(
            accepted,
            lambda _: self.residual.integrated_flux(root.state, args),
            lambda _: (
                jnp.full_like(args.source, jnp.nan),
                jnp.full(
                    (self.problem.exterior.lengths.size, args.source.shape[1]),
                    jnp.nan,
                    dtype=jnp.float64,
                ),
            ),
            None,
        )
        free = self.residual.free_indices
        volumes = self.problem.exterior.node_volumes[:, None]
        source_integral = jnp.sum((volumes * args.source)[free], axis=0)
        boundary_flux = -jnp.sum(
            jnp.where(self.problem.equation_mask[:, None], 0, integrated), axis=0
        )
        defect = integrated - volumes * args.source
        ledger = MeshfreeComponentConservationLedger(
            source_integral,
            boundary_flux,
            source_integral - boundary_flux,
            jnp.sum(integrated, axis=0),
            jnp.max(jnp.abs(defect[free]), axis=0),
        )
        public_state, public_flux = guard_derivative_validity(
            (jnp.where(accepted, full, jnp.nan), jnp.where(accepted, edge_flux, jnp.nan)),
            accepted,
            dependencies=(args, initial_state),
            failure="status",
            message="Coupled meshfree conservation derivatives require a successful forward result.",
        )
        return MeshfreeCoupledConservationResult(
            public_state,
            root,
            public_flux,
            ledger,
            self.coverage_assessment,
            evidence,
            accepted,
        )

    def adjoint(
        self,
        primal: MeshfreeCoupledConservationResult,
        cotangent: ArrayLike,
        /,
        *,
        parameters: Mapping[str, object] | None = None,
        law: AbstractCoupledEdgeConstitutiveLaw | None = None,
    ) -> MeshfreeCoupledConservationAdjoint:
        """Native block transpose solve retaining both primal and adjoint status."""
        args = self.problem.runtime(parameters=parameters, law=law)
        free = self.residual.free_indices
        rhs = jnp.asarray(cotangent, dtype=jnp.float64)
        if rhs.shape == self.problem.source.shape:
            rhs = rhs[free]
        if rhs.shape != (free.size, args.source.shape[1]):
            raise ValueError(
                "Coupled cotangents must be full-node or free-node component blocks."
            )
        blocks = (
            args.conductance
            * args.metric_weights[:, None, None]
            * self.residual.bound_law(args).derivative(
                self.residual.jumps(primal.state), self.problem.features
            )
        )
        operator = self.block.operator(blocks)
        _, adjoint_policy = self.derivative_policy.resolve(self.native.method)
        linear = solve_transpose(
            LinearSystem(operator), rhs.reshape(-1), policy=adjoint_policy
        )
        accepted = primal.accepted & jnp.all(linear.successful)
        value = guard_derivative_validity(
            jnp.where(accepted, linear.value, jnp.nan).reshape(rhs.shape),
            accepted,
            dependencies=(args, cotangent),
            failure="status",
            message="Coupled meshfree conservation adjoint derivatives require successful primal and adjoint results.",
        )
        return MeshfreeCoupledConservationAdjoint(primal, linear, value, accepted)


def prepare_meshfree_coupled_conservation_solve(
    problem: MeshfreeCoupledConservationProblem,
    /,
    *,
    initial_state: ArrayLike | None = None,
    parameters: Mapping[str, object] | None = None,
    linear_policy: LinearSolvePolicy | None = None,
    termination: NonlinearTermination | None = None,
    derivative_policy: ImplicitRootDerivativePolicy | None = None,
    execution: StreamedRelationPlan | None = None,
) -> PreparedMeshfreeCoupledConservationSolve:
    """Prepare the block sparse Jacobian, NewtonKrylov template and background audit.

    ``execution`` selects the streamed canonical edge schedule on which every
    residual evaluates each coupled edge law once.
    """
    if not isinstance(problem, MeshfreeCoupledConservationProblem):
        raise TypeError("problem must be a MeshfreeCoupledConservationProblem.")
    free = np.flatnonzero(np.asarray(problem.equation_mask)).astype(np.int32)
    size = problem.component_count
    pairs = np.asarray(problem.exterior.pairs)
    routes, rows, columns = _reduced_routes(pairs, free, problem.source.shape[0])
    scalar_relation = EdgeRelation(
        columns, rows, source_size=free.size, target_size=free.size
    )
    scalar_space = ArraySpace(
        (free.size,), dtype=jnp.float64, space_id=f"{problem.problem_id}:free-nodes"
    )
    args = problem.runtime(parameters=parameters)
    background_stiffness = _ReducedStiffness(
        scalar_relation,
        jnp.asarray(routes),
        scalar_space,
        _SparseStoragePlan(scalar_relation),
    )
    background_operator = background_stiffness.operator(
        args.conductance * problem.law.background_conductance * args.metric_weights
    )
    assessment = _assess_background(
        problem.exterior, background_operator, free, np.asarray(problem.boundary_mask)
    )
    # Block routes (route, p, q): row r*C+p, column c*C+q, component index fastest.
    component = np.arange(size, dtype=np.int32)
    block_rows = (rows[:, None, None] * size + component[None, :, None]).repeat(
        size, axis=2
    )
    block_columns = (columns[:, None, None] * size + component[None, None, :]).repeat(
        size, axis=1
    )
    block_relation = EdgeRelation(
        block_columns.reshape(-1),
        block_rows.reshape(-1),
        source_size=free.size * size,
        target_size=free.size * size,
    )
    space = ArraySpace(
        (free.size * size,),
        dtype=jnp.float64,
        space_id=f"{problem.problem_id}:free-block-state",
    )
    block = _ReducedBlockStiffness(
        block_relation,
        jnp.asarray(routes // 4, dtype=jnp.int32),
        jnp.asarray(np.asarray((1.0, 1.0, -1.0, -1.0))[routes % 4]),
        space,
        _SparseStoragePlan(block_relation),
    )
    policy, derivative = _resolved_policies(
        free.size * size, linear_policy, derivative_policy
    )
    coverage_assessment, coverage_good = _coverage_state(
        problem.coverage, problem.features
    )
    validity = _ConservationValidity(coverage_good, assessment.admitted)
    residual = _CoupledConservationResidual(
        problem.exterior,
        problem.features,
        jnp.asarray(free),
        validity,
        problem.law,
        *_prepared_edges(problem, execution),
    )
    initial = (
        problem.boundary_values[free]
        if initial_state is None
        else jnp.asarray(initial_state, dtype=jnp.float64)
    )
    if initial.shape == problem.source.shape:
        initial = initial[free]
    if initial.shape != (free.size, size):
        raise ValueError(
            "Initial state must be full-node or free-node packed component blocks."
        )
    initial = initial.reshape(-1)
    # Only potential-gradient laws have a symmetric block Jacobian.
    symmetric = isinstance(problem.law, MonotoneCoupledEdgeFlux)
    sparse_plan = compile_sparse_jacobian(
        residual,
        initial,
        source=space,
        target=space,
        sample_args=args,
        structure=SparsePattern(block_relation, symmetric=symmetric),
        compiler="native",
        symmetric=symmetric,
        mode="fwd",
        plan_id=f"{problem.problem_id}:fixed-block-edge-Jacobian",
    )
    native = _prepared_nonlinear(
        problem.problem_id,
        residual,
        validity,
        space,
        initial,
        args,
        JacobianPolicy("sparse", sparse_plan=sparse_plan),
        policy,
        termination,
    )
    return PreparedMeshfreeCoupledConservationSolve(
        problem,
        native,
        derivative,
        residual,
        block,
        background_stiffness,
        assessment,
        coverage_assessment,
    )


__all__ = [
    "MeshfreeComponentConservationLedger",
    "MeshfreeConservationAdjoint",
    "MeshfreeConservationLedger",
    "MeshfreeConservationProblem",
    "MeshfreeConservationResult",
    "MeshfreeConstitutiveEvidence",
    "MeshfreeContractionStatus",
    "MeshfreeCoupledConservationAdjoint",
    "MeshfreeCoupledConservationProblem",
    "MeshfreeCoupledConservationResult",
    "PreparedMeshfreeConservationSolve",
    "PreparedMeshfreeCoupledConservationSolve",
    "prepare_meshfree_conservation_solve",
    "prepare_meshfree_coupled_conservation_solve",
]
