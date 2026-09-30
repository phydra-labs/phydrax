#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import assert_never, final, Literal, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from .._admissibility import guard_derivative_validity, refuse_derivative_dependencies
from .._differentiation import (
    DerivativeRoute,
    DerivativeSurface,
    OwnerDerivativeCapability,
)
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_complex import TetrahedralConnectivity
from ..discretization.bem._bc_dual import prepare_buffa_christiansen_dual_3d
from ..discretization.bem._rwg import RWGSurfaceCurrentSpace3D
from ..discretization.bem._surface_complex import OrientedTriangleSurfaceComplex3D
from ..linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    BlockLinearOperator,
    BlockSpace,
    coordinate_space,
    DifferentiationMode,
    DifferentiationPolicy,
    DualSpace,
    estimate_operator_action_cost,
    FailurePolicy,
    FGMRES,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    prepare as prepare_linear,
    PreparedLinearSolve,
    solve as solve_linear,
    TransposeLinearOperator,
)
from ..operators.integral.layer_potential._elasticity3d import (
    ElasticitySingleLayerDP0Galerkin3D,
)
from ..operators.integral.layer_potential._maxwell3d import MaxwellEFIEPolicy3D
from ..operators.integral.layer_potential._maxwell_bc3d import (
    prepare_maxwell_bc_efie_3d,
    PreparedMaxwellBCEFIE3D,
)
from ._nonmatching_fem_bem3d import (
    _prepare,
    CoupledFEMBEMResult3D,
    PreparedNonmatchingFEMBEM3D,
)


if TYPE_CHECKING:
    from ..discretization.fem._de_rham import FiniteElementDeRhamComplex


_PHYSICS = "static isotropic three-dimensional linear elasticity"
_FORMULATION = (
    "caller-prepared Costabel symmetric primal-traction block "
    "[A_sym, C^T; C, V], where A_sym already contains the interior elasticity "
    "and hypersingular trace contribution, C contains the exact signed "
    "Calderon/trace map, C^T is its exact algebraic transpose, and V is the "
    "weak DP0 Kelvin single layer"
)
_NORMAL = "outward-from-fem-interior"
_GEOMETRY = (
    "one caller-qualified matching closed oriented three-dimensional interface; "
    "the BEM surface normal points from the FEM interior into the exterior"
)
_NON_GOALS = (
    "automatic volume-to-surface matching or trace construction",
    "nonmatching, mortar, partial, open, curved, moving, or higher-order interfaces",
    "anisotropic, heterogeneous, nonlinear, dynamic, or contact elasticity",
    "continuum certification or discretization-error estimation",
    "Maxwell coupling without an exact H(curl)-to-RWG tangential trace and dual map",
)
_LOAD_ARGUMENTS = ("interior_load", "boundary_load")
_FIXED_STRUCTURE_REFUSALS = {
    "geometry": (
        "the matching interface, the BEM surface mesh and panelization, and the "
        "caller-qualified volume discretization are fixed prepared structure; "
        "shape derivatives are not qualified"
    ),
    "kernel": (
        "the Kelvin fundamental solution and its Lame parameters (shear modulus, "
        "Poisson ratio) are assembled into the fixed prepared weak DP0 single "
        "layer V; kernel and material derivatives are not qualified"
    ),
    "quadrature": (
        "the regular and singular Kelvin quadrature rules and pair classification "
        "are fixed prepared structure"
    ),
    "interior_operator": (
        "the caller-prepared A_sym entries, including the hypersingular trace "
        "contribution, are fixed prepared operator arrays; interior operator and "
        "hypersingular derivatives are not qualified"
    ),
    "trace_maps": (
        "the caller-prepared exact signed Calderon trace map C and its conormal "
        "transpose C^T are fixed prepared operator arrays; trace-map derivatives "
        "are not qualified"
    ),
}
_STOPPED_LOAD_REFUSAL = (
    "differentiation mode 'none' stops the prepared elasticity FEM-BEM solve; "
    "prepare with mode 'rhs-only' to differentiate with respect to the loads"
)
_MATHEMATICAL_REFUSAL = (
    "Elasticity FEM-BEM differentiation mode 'mathematical' is unsupported: the "
    "caller-prepared A_sym (including the hypersingular trace term), the exact "
    "trace/conormal maps C and C^T, and the weak DP0 Kelvin single layer V are "
    "fixed prepared operators without runtime operator arguments; Lame/kernel, "
    "hypersingular, trace-map, and geometry derivatives are not qualified. Use "
    "'rhs-only' for derivatives with respect to interior_load and boundary_load."
)
_ALGORITHMIC_REFUSAL = (
    "Elasticity FEM-BEM differentiation mode 'algorithmic' is unsupported: the "
    "unrolled Krylov derivative of the executed iteration is not the solution-map "
    "derivative and is not qualified. Use 'rhs-only' for derivatives with respect "
    "to interior_load and boundary_load."
)


class VectorFEMBEMSupportReport(StrictModule, NonTrainableState):
    """Explicit implemented and rejected vector FEM--BEM coupling envelope."""

    implemented: tuple[str, ...] = eqx.field(static=True)
    rejected: tuple[str, ...] = eqx.field(static=True)
    continuum_certified: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)
    matching_maxwell_supported: bool = eqx.field(static=True)
    nonmatching_maxwell_supported: bool = eqx.field(static=True)
    automatic_periodic_maxwell_supported: bool = eqx.field(static=True)
    caller_built_periodic_maxwell_supported: bool = eqx.field(static=True)


class ElasticityFEMBEMInterfaceQualification3D(StrictModule, NonTrainableState):
    """Caller evidence binding exact matching-interface elasticity blocks.

    This object records, but does not manufacture, the signed Calderon trace map
    ``C`` and its conormal transpose ``C^T``.  Preparation additionally verifies
    their operator identities, spaces, executable transpose relation, and normal
    convention.  Geometry and discretization accuracy remain the caller's
    reported finite-dimensional evidence; no continuum claim is inferred.
    """

    interface_id: str = eqx.field(static=True)
    interior_space_id: str = eqx.field(static=True)
    boundary_space_id: str = eqx.field(static=True)
    trace_operator_id: str = eqx.field(static=True)
    conormal_operator_id: str = eqx.field(static=True)
    bem_operator_id: str = eqx.field(static=True)
    orientation: str = eqx.field(static=True)
    provider_ids: tuple[str, ...] = eqx.field(static=True)
    precision_evidence: tuple[str, ...] = eqx.field(static=True)
    resource_evidence: tuple[tuple[str, int], ...] = eqx.field(static=True)
    error_evidence: tuple[str, ...] = eqx.field(static=True)
    matching: bool = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    continuum_certified: bool = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)

    def __init__(
        self,
        interface_id: str,
        interior_space_id: str,
        boundary_space_id: str,
        trace_operator_id: str,
        conormal_operator_id: str,
        bem_operator_id: str,
        /,
        *,
        orientation: str = _NORMAL,
        provider_ids: tuple[str, ...],
        precision_evidence: tuple[str, ...],
        resource_evidence: tuple[tuple[str, int], ...],
        error_evidence: tuple[str, ...],
        matching: bool = True,
        spatial_dimension: int = 3,
        continuum_certified: bool = False,
    ) -> None:
        identifiers = tuple(
            str(value)
            for value in (
                interface_id,
                interior_space_id,
                boundary_space_id,
                trace_operator_id,
                conormal_operator_id,
                bem_operator_id,
            )
        )
        providers = tuple(str(value) for value in provider_ids)
        precision = tuple(str(value) for value in precision_evidence)
        errors = tuple(str(value) for value in error_evidence)
        resources = tuple((str(name), int(value)) for name, value in resource_evidence)
        if any(not value for value in identifiers):
            raise ValueError(
                "Interface and operator/space identifiers must be non-empty."
            )
        if not providers or any(not value for value in providers):
            raise ValueError("provider_ids must contain explicit non-empty providers.")
        if not precision or any(not value for value in precision):
            raise ValueError("precision_evidence must be explicit and non-empty.")
        if not errors or any(not value for value in errors):
            raise ValueError("error_evidence must be explicit and non-empty.")
        if not resources or any(not name or value < 0 for name, value in resources):
            raise ValueError(
                "resource_evidence must contain non-empty names and nonnegative values."
            )
        if not str(orientation):
            raise ValueError("orientation must be non-empty.")
        self.interface_id = identifiers[0]
        self.interior_space_id = identifiers[1]
        self.boundary_space_id = identifiers[2]
        self.trace_operator_id = identifiers[3]
        self.conormal_operator_id = identifiers[4]
        self.bem_operator_id = identifiers[5]
        self.orientation = str(orientation)
        self.provider_ids = providers
        self.precision_evidence = precision
        self.resource_evidence = resources
        self.error_evidence = errors
        self.matching = bool(matching)
        self.spatial_dimension = int(spatial_dimension)
        self.continuum_certified = bool(continuum_certified)
        self.qualification_id = canonical_fingerprint(
            {
                "kind": "elasticity-fem-bem-interface-qualification-3d",
                "interface": identifiers[0],
                "spaces": identifiers[1:3],
                "operators": identifiers[3:],
                "orientation": str(orientation),
                "providers": providers,
                "precision": precision,
                "resources": resources,
                "errors": errors,
                "matching": bool(matching),
                "dimension": int(spatial_dimension),
                "continuum_certified": bool(continuum_certified),
            }
        )


class ElasticityFEMBEMResult3D(StrictModule, NonTrainableState):
    """Solved finite-dimensional symmetric static-elasticity coupling state.

    ``valid`` is primal acceptance (successful, finite linear solve with finite
    block residual and symmetry evidence).  ``derivative_valid`` additionally
    requires the prepared ``derivative_capability`` route to hold for this solve:
    it is ``valid`` and the converged implicit linear-solve derivative under
    ``'rhs-only'`` and always false under the stopped ``'none'`` route.
    """

    interior_displacement: Array
    boundary_traction: Array
    interior_load: Array
    boundary_load: Array
    linear_result: LinearSolveResult
    relative_block_residual: Array
    symmetry_defect: Array
    bem_maximum_quadrature_error: Array
    valid: Array
    derivative_valid: Array
    derivative_capability: OwnerDerivativeCapability
    spatial_dimension: int = eqx.field(static=True)
    physics: str = eqx.field(static=True)
    geometry_contract: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider_ids: tuple[str, ...] = eqx.field(static=True)
    precision_evidence: tuple[str, ...] = eqx.field(static=True)
    resource_evidence: tuple[tuple[str, int], ...] = eqx.field(static=True)
    error_evidence: tuple[str, ...] = eqx.field(static=True)
    normal_convention: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    continuum_certified: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PreparedElasticityFEMBEM3D(StrictModule, NonTrainableState):
    """Prepared operator-level Costabel symmetric elasticity FEM--BEM product.

    The volume H1 vector trace substrate does not yet expose a matching vector
    interface constructor.  Consequently this bounded product accepts only
    caller-prepared blocks with explicit qualification.  It does not rebuild
    geometry or operators.  The exact block is ``[A_sym, C^T; C, V]``: callers
    must include the hypersingular trace term in ``A_sym`` and the Calderon sign
    and jump in ``C``.  ``C^T`` must be a PHYDRAX algebraic transpose view, and
    ``V`` is the landed weak DP0 Kelvin single layer.  These checks establish a
    symmetric discrete block, not a continuum transmission certificate.

    ``derivative_capability`` names the runtime arguments whose derivatives the
    product admits.  Under differentiation mode ``'rhs-only'`` the loads
    ``interior_load`` and ``boundary_load`` are admitted as implicit
    solver-argument derivatives of the converged solve; under the default
    ``'none'`` they are refused with ``derivative-unsupported``.  ``A_sym``
    (including the hypersingular term), the trace/conormal maps, the Kelvin
    kernel and Lame parameters, quadrature, and geometry are fixed prepared
    structure whose derivatives are refused in every mode; ``'mathematical'`` and
    ``'algorithmic'`` are refused at preparation.
    """

    interior_operator: AbstractLinearOperator
    trace_operator: AbstractLinearOperator
    conormal_operator: AbstractLinearOperator
    bem: ElasticitySingleLayerDP0Galerkin3D
    interface: ElasticityFEMBEMInterfaceQualification3D
    operator: BlockLinearOperator
    prepared_linear: PreparedLinearSolve
    linear_policy: LinearSolvePolicy
    bem_maximum_quadrature_error: Array
    derivative_capability: OwnerDerivativeCapability
    spatial_dimension: int = eqx.field(static=True)
    physics: str = eqx.field(static=True)
    geometry_contract: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider_ids: tuple[str, ...] = eqx.field(static=True)
    precision_evidence: tuple[str, ...] = eqx.field(static=True)
    resource_evidence: tuple[tuple[str, int], ...] = eqx.field(static=True)
    error_evidence: tuple[str, ...] = eqx.field(static=True)
    normal_convention: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    continuum_certified: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def right_hand_side(
        self,
        interior_load: ArrayLike,
        boundary_load: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        """Validate and return one two-block load."""

        return (
            self.interior_operator.target.validate(interior_load),
            self.bem.weak_operator.target.validate(boundary_load),
        )

    def solve(
        self,
        interior_load: ArrayLike,
        boundary_load: ArrayLike,
        /,
    ) -> ElasticityFEMBEMResult3D:
        """Solve one prepared finite-dimensional symmetric coupling system."""

        return solve_elasticity_fem_bem_3d(self, interior_load, boundary_load)


def vector_fem_bem_support_report() -> VectorFEMBEMSupportReport:
    """Return the exact vector coupling support boundary of this module."""

    implemented = (
        "static isotropic elasticity 3D: caller-prepared Costabel symmetric "
        "[A_sym, C^T; C, V] with landed weak DP0 Kelvin V",
        "static elasticity 3D derivatives: differentiation mode 'rhs-only' admits "
        "implicit derivatives of interior_displacement and boundary_traction with "
        "respect to the runtime loads interior_load and boundary_load at an "
        "accepted converged solve (NaN or error otherwise); the default mode "
        "'none' refuses load derivatives with derivative-unsupported",
        "matching lowest-order tetrahedral Maxwell 3D: automatically constructed "
        "outward tangential trace, exact RWG/BC dual conormal and genuine BC EFIE",
        "nonmatching Maxwell 3D: caller-qualified FE mortar with its magnetic "
        "conormal in the upper coupling and both block residuals",
        "periodic Maxwell 3D: caller-built mortar/block route over bounded "
        "periodic image operators; no automatic periodic interface construction",
    )
    rejected = (
        "Automatic vector-H1 matching-interface trace preparation is unavailable; "
        "exact caller-prepared elasticity maps are required.",
        "Stokes, dynamic elasticity, anisotropic elasticity, and nonmatching elasticity are not implemented.",
        "Static elasticity FEM-BEM derivatives with respect to A_sym (including the "
        "hypersingular term), the trace/conormal maps, the Kelvin kernel and Lame "
        "parameters, quadrature, and geometry are rejected; differentiation modes "
        "'mathematical' and 'algorithmic' are refused at preparation.",
    )
    return VectorFEMBEMSupportReport(
        implemented=implemented,
        rejected=rejected,
        continuum_certified=False,
        report_id=canonical_fingerprint(
            {
                "kind": "vector-fem-bem-support-report",
                "implemented": implemented,
                "rejected": rejected,
                "continuum_certified": False,
            }
        ),
        matching_maxwell_supported=True,
        nonmatching_maxwell_supported=True,
        automatic_periodic_maxwell_supported=False,
        caller_built_periodic_maxwell_supported=True,
    )


def _default_linear_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        FGMRES(restart=30, stagnation_iterations=30),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )


def _qualified_differentiation(
    mode: DifferentiationMode, /
) -> Literal["none", "rhs-only"]:
    match mode:
        case "none" | "rhs-only":
            return mode
        case "mathematical":
            raise ValueError(_MATHEMATICAL_REFUSAL)
        case "algorithmic":
            raise ValueError(_ALGORITHMIC_REFUSAL)
        case _:
            assert_never(mode)


def _derivative_capability(
    prepared_id: str, mode: Literal["none", "rhs-only"], /
) -> OwnerDerivativeCapability:
    match mode:
        case "rhs-only":
            return OwnerDerivativeCapability(
                prepared_id,
                admitted=dict.fromkeys(
                    _LOAD_ARGUMENTS, DerivativeSurface.SOLVER_ARGUMENT
                ),
                refused=_FIXED_STRUCTURE_REFUSALS,
                route=DerivativeRoute.IMPLICIT,
                conditions=(
                    "accepted-result",
                    "prepared-geometry-fixed",
                    "solve-converged",
                ),
            )
        case "none":
            return OwnerDerivativeCapability(
                prepared_id,
                admitted={},
                refused={
                    **_FIXED_STRUCTURE_REFUSALS,
                    **dict.fromkeys(_LOAD_ARGUMENTS, _STOPPED_LOAD_REFUSAL),
                },
                route=DerivativeRoute.STOPPED,
                conditions=("prepared-geometry-fixed",),
            )
        case _:
            assert_never(mode)


def _derivative_envelope(
    prepared: PreparedElasticityFEMBEM3D,
    outputs: tuple[Array, Array, Array, Array, LinearSolveResult],
    loads: tuple[Array, Array],
    valid: Array,
    /,
) -> tuple[Array, Array, Array, Array, LinearSolveResult]:
    route = prepared.derivative_capability.derivative_contract.route
    match route:
        case DerivativeRoute.IMPLICIT:
            return guard_derivative_validity(
                outputs,
                valid,
                dependencies=loads,
                failure=prepared.linear_policy.failure.mode,
                message=(
                    "Elasticity FEM-BEM load derivatives require an accepted "
                    "converged solve."
                ),
            )
        case DerivativeRoute.STOPPED:
            return refuse_derivative_dependencies(
                outputs,
                loads,
                message=(
                    f"prepared elasticity FEM-BEM {prepared.prepared_id} refuses "
                    f"derivatives with respect to interior_load and boundary_load: "
                    f"{_STOPPED_LOAD_REFUSAL}"
                ),
            )
        case _:
            raise ValueError(
                f"Elasticity FEM-BEM does not qualify the {route.value} derivative route."
            )


def _require_same_space(
    actual: AbstractVectorSpace, expected_id: str, role: str, /
) -> None:
    if actual.space_id != expected_id:
        raise ValueError(
            f"{role} space {actual.space_id!r} does not match qualified space {expected_id!r}."
        )


def _validate_exact_transpose_pair(
    trace: AbstractLinearOperator,
    conormal: AbstractLinearOperator,
    /,
) -> None:
    conormal_is_transpose = (
        isinstance(conormal, TransposeLinearOperator)
        and conormal.operator.operator_id == trace.operator_id
    )
    trace_is_transpose = (
        isinstance(trace, TransposeLinearOperator)
        and trace.operator.operator_id == conormal.operator_id
    )
    if not (conormal_is_transpose or trace_is_transpose):
        raise ValueError(
            "The conormal and trace maps must be one exact PHYDRAX algebraic transpose pair."
        )


def prepare_elasticity_fem_bem_3d(
    interior_operator: AbstractLinearOperator,
    trace_operator: AbstractLinearOperator,
    conormal_operator: AbstractLinearOperator,
    bem: ElasticitySingleLayerDP0Galerkin3D,
    interface: ElasticityFEMBEMInterfaceQualification3D,
    /,
    *,
    linear: LinearSolvePolicy | None = None,
) -> PreparedElasticityFEMBEM3D:
    """Prepare a qualified Costabel symmetric static-elasticity block."""

    operators = (interior_operator, trace_operator, conormal_operator)
    if not all(isinstance(operator, AbstractLinearOperator) for operator in operators):
        raise TypeError("interior, trace, and conormal inputs must be linear operators.")
    if not isinstance(bem, ElasticitySingleLayerDP0Galerkin3D):
        raise TypeError("bem must be an ElasticitySingleLayerDP0Galerkin3D.")
    if not isinstance(interface, ElasticityFEMBEMInterfaceQualification3D):
        raise TypeError("interface must be an ElasticityFEMBEMInterfaceQualification3D.")
    if (
        interface.spatial_dimension != 3
        or not interface.matching
        or interface.continuum_certified
    ):
        raise ValueError(
            "Elasticity FEM-BEM requires a matching discrete 3D qualification without a continuum-certification claim."
        )
    if interface.orientation != _NORMAL:
        raise ValueError(
            f"Elasticity FEM-BEM requires interface orientation {_NORMAL!r}."
        )
    if (
        any(operator.batch_shape for operator in operators)
        or bem.weak_operator.batch_shape
    ):
        raise ValueError("Elasticity FEM-BEM blocks must be unbatched operators.")
    if not interior_operator.source.compatible(interior_operator.target):
        raise ValueError(
            "The interior elasticity block must be square on one exact space."
        )
    if not interior_operator.properties.certifies("self_adjoint"):
        raise ValueError(
            "The caller-prepared A_sym block must carry non-unknown self-adjoint evidence."
        )
    if (
        not interior_operator.capabilities.transpose
        or not interior_operator.capabilities.adjoint
    ):
        raise ValueError(
            "The interior elasticity block must provide transpose and adjoint actions."
        )

    boundary = bem.weak_operator.source
    if not boundary.compatible(bem.weak_operator.target):
        raise ValueError("The weak elasticity DP0 single layer must be square.")
    report = bem.assembly_report
    contract = bem.contract
    if (
        contract.ambient_dimension != 3
        or contract.pde != "static isotropic Navier-Cauchy elasticity without body force"
        or report.component_count != 1
        or report.face_count != bem.face_count
        or report.continuum_discretization_error_estimated
        or not bool(report.finite)
        or not bool(report.accuracy_supported)
    ):
        raise ValueError(
            "The elasticity DP0 provider lies outside the supported static coupling envelope."
        )
    if "outward source normal" not in contract.traction_convention:
        raise ValueError(
            "The elasticity DP0 provider has an incompatible traction orientation."
        )
    if (
        not bem.weak_operator.capabilities.transpose
        or not bem.weak_operator.capabilities.adjoint
    ):
        raise ValueError(
            "The elasticity DP0 operator must provide transpose and adjoint actions."
        )

    _require_same_space(interior_operator.source, interface.interior_space_id, "Interior")
    _require_same_space(boundary, interface.boundary_space_id, "Boundary")
    if trace_operator.operator_id != interface.trace_operator_id:
        raise ValueError("The trace map does not match its qualified operator identity.")
    if conormal_operator.operator_id != interface.conormal_operator_id:
        raise ValueError(
            "The conormal map does not match its qualified operator identity."
        )
    if bem.weak_operator.operator_id != interface.bem_operator_id:
        raise ValueError("The elasticity BEM operator does not match its qualification.")
    if not trace_operator.source.compatible(
        interior_operator.source
    ) or not trace_operator.target.compatible(boundary):
        raise ValueError(
            "The exact trace map must route interior displacement to DP0 boundary space."
        )
    if not conormal_operator.source.compatible(
        boundary
    ) or not conormal_operator.target.compatible(interior_operator.source):
        raise ValueError(
            "The exact conormal map must route DP0 traction to the interior dual space."
        )
    if (
        not trace_operator.capabilities.transpose
        or not trace_operator.capabilities.adjoint
    ):
        raise ValueError("The trace map must provide transpose and adjoint actions.")
    if (
        not conormal_operator.capabilities.transpose
        or not conormal_operator.capabilities.adjoint
    ):
        raise ValueError("The conormal map must provide transpose and adjoint actions.")
    _validate_exact_transpose_pair(trace_operator, conormal_operator)

    policy = _default_linear_policy() if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be a LinearSolvePolicy or None.")
    mode = _qualified_differentiation(policy.differentiation.mode)

    block_space = BlockSpace(
        (interior_operator.source, boundary),
        names=("interior_displacement", "boundary_traction"),
    )
    operator = BlockLinearOperator(
        (
            (interior_operator, conormal_operator),
            (trace_operator, bem.weak_operator),
        ),
        source=block_space,
        target=block_space,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
        operator_id=canonical_fingerprint(
            {
                "kind": "elasticity-fem-bem-costabel-symmetric-3d",
                "interior": interior_operator.operator_id,
                "trace": trace_operator.operator_id,
                "conormal": conormal_operator.operator_id,
                "single_layer": bem.weak_operator.operator_id,
                "interface": interface.qualification_id,
                "normal": _NORMAL,
            }
        ),
    )
    problem_id = canonical_fingerprint(
        {
            "kind": "elasticity-fem-bem-linear-system-3d",
            "operator": operator.operator_id,
        }
    )
    prepared_linear = prepare_linear(
        LinearSystem(operator, problem_id=problem_id), policy
    )
    cost = estimate_operator_action_cost(operator)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-elasticity-fem-bem-3d",
            "operator": operator.operator_id,
            "linear_plan": prepared_linear.plan.plan_id,
            "interface": interface.qualification_id,
            "bem_report": report.report_id,
        }
    )
    providers = interface.provider_ids + (
        interface.qualification_id,
        bem.weak_operator.operator_id,
        bem.contract.provider,
        bem.kernel.kernel_id,
        report.report_id,
    )
    precision = interface.precision_evidence + (
        bem.contract.precision,
        str(boundary.structure().dtype),
    )
    resources = interface.resource_evidence + (
        ("block_unknowns", operator.source.size),
        ("bem_preparation_workspace_bytes", report.preparation_workspace_bytes),
        ("bem_resident_bytes", report.resident_bytes),
        ("operator_storage_bytes", cost.storage_bytes),
        ("operator_action_workspace_bytes_per_rhs", cost.apply_workspace_bytes_per_rhs),
    )
    errors = interface.error_evidence + (
        f"elasticity DP0 assembly report {report.report_id}",
        f"operator action cost exact={cost.exact}: {cost.reason}",
        "linear diagnostics and a discrete block residual are returned per solve",
        "continuum discretization error is not estimated",
    )
    return PreparedElasticityFEMBEM3D(
        interior_operator=interior_operator,
        trace_operator=trace_operator,
        conormal_operator=conormal_operator,
        bem=bem,
        interface=interface,
        operator=operator,
        prepared_linear=prepared_linear,
        linear_policy=policy,
        bem_maximum_quadrature_error=report.maximum_quadrature_error,
        derivative_capability=_derivative_capability(prepared_id, mode),
        spatial_dimension=3,
        physics=_PHYSICS,
        geometry_contract=_GEOMETRY,
        formulation=_FORMULATION,
        provider_ids=providers,
        precision_evidence=precision,
        resource_evidence=resources,
        error_evidence=errors,
        normal_convention=_NORMAL,
        non_goals=_NON_GOALS,
        continuum_certified=False,
        prepared_id=prepared_id,
    )


def solve_elasticity_fem_bem_3d(
    prepared: PreparedElasticityFEMBEM3D,
    interior_load: ArrayLike,
    boundary_load: ArrayLike,
    /,
) -> ElasticityFEMBEMResult3D:
    """Solve one qualified symmetric static-elasticity FEM--BEM block.

    Under ``'rhs-only'`` the displacement, traction, and residual/symmetry
    evidence carry implicit load derivatives guarded by primal acceptance: a
    rejected solve returns NaN tangents (``failure='status'``) or raises
    (``failure='error'``).  Under ``'none'`` any derivative request with respect
    to the loads raises ``derivative-unsupported`` instead of returning zero.
    """

    if not isinstance(prepared, PreparedElasticityFEMBEM3D):
        raise TypeError("prepared must be a PreparedElasticityFEMBEM3D.")
    right_hand_side = prepared.right_hand_side(interior_load, boundary_load)
    linear_result = solve_linear(prepared.prepared_linear, right_hand_side)
    interior, traction = prepared.operator.source.validate(linear_result.value)
    image = prepared.operator.mv((interior, traction))
    residual = (image[0] - right_hand_side[0], image[1] - right_hand_side[1])
    residual_squared = jnp.real(prepared.operator.target.inner(residual, residual))
    right_squared = jnp.real(
        prepared.operator.target.inner(right_hand_side, right_hand_side)
    )
    relative_residual = jnp.sqrt(residual_squared) / jnp.maximum(
        jnp.sqrt(right_squared), jnp.asarray(1.0, dtype=residual_squared.dtype)
    )
    transpose_image = prepared.operator.transpose_mv((interior, traction))
    symmetry_difference = (
        transpose_image[0] - image[0],
        transpose_image[1] - image[1],
    )
    symmetry_defect = jnp.sqrt(
        jnp.real(prepared.operator.target.inner(symmetry_difference, symmetry_difference))
    )
    finite = (
        jnp.all(jnp.isfinite(interior))
        & jnp.all(jnp.isfinite(traction))
        & jnp.isfinite(relative_residual)
        & jnp.isfinite(symmetry_defect)
    )
    valid = linear_result.successful & linear_result.diagnostics.finite & finite
    interior, traction, relative_residual, symmetry_defect, linear_result = (
        _derivative_envelope(
            prepared,
            (interior, traction, relative_residual, symmetry_defect, linear_result),
            right_hand_side,
            valid,
        )
    )
    return ElasticityFEMBEMResult3D(
        interior_displacement=interior,
        boundary_traction=traction,
        interior_load=right_hand_side[0],
        boundary_load=right_hand_side[1],
        linear_result=linear_result,
        relative_block_residual=relative_residual,
        symmetry_defect=symmetry_defect,
        bem_maximum_quadrature_error=prepared.bem_maximum_quadrature_error,
        valid=valid,
        derivative_valid=valid & linear_result.derivative_valid,
        derivative_capability=prepared.derivative_capability,
        spatial_dimension=prepared.spatial_dimension,
        physics=prepared.physics,
        geometry_contract=prepared.geometry_contract,
        formulation=prepared.formulation,
        provider_ids=prepared.provider_ids,
        precision_evidence=prepared.precision_evidence,
        resource_evidence=prepared.resource_evidence,
        error_evidence=prepared.error_evidence,
        normal_convention=prepared.normal_convention,
        non_goals=prepared.non_goals,
        continuum_certified=prepared.continuum_certified,
        prepared_id=prepared.prepared_id,
    )


__all__ = [
    "ElasticityFEMBEMInterfaceQualification3D",
    "ElasticityFEMBEMResult3D",
    "PreparedElasticityFEMBEM3D",
    "VectorFEMBEMSupportReport",
    "prepare_elasticity_fem_bem_3d",
    "solve_elasticity_fem_bem_3d",
    "PreparedMatchingMaxwellFEMBEM3D",
    "prepare_matching_maxwell_fem_bem_3d",
    "vector_fem_bem_support_report",
]


@final
class PreparedMatchingMaxwellFEMBEM3D(StrictModule, NonTrainableState):
    """Automatic lowest-order tetrahedral volume-to-oriented-surface coupling."""

    volume_complex: FiniteElementDeRhamComplex
    boundary: PreparedMaxwellBCEFIE3D
    rwg_trace: AbstractLinearOperator
    dual_trace: AbstractLinearOperator
    dual_conormal: AbstractLinearOperator
    magnetic_conormal: AbstractLinearOperator
    coupled: PreparedNonmatchingFEMBEM3D
    prepared_id: str = eqx.field(static=True)

    def solve(
        self, interior_load: ArrayLike, boundary_load: ArrayLike, /
    ) -> CoupledFEMBEMResult3D:
        return self.coupled.solve(interior_load, boundary_load)


def _matching_maxwell_surface(
    complex: FiniteElementDeRhamComplex,
) -> tuple[
    OrientedTriangleSurfaceComplex3D, npt.NDArray[np.int32], npt.NDArray[np.int32]
]:
    connectivity = complex.mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise ValueError("Matching Maxwell requires a tetrahedral volume mesh.")
    from ..discretization._boundary_complex import boundary_subcomplex

    boundary = boundary_subcomplex(
        complex.topology, boundary_mask=connectivity.boundary_faces
    )
    parents = np.asarray(boundary.parent_indices[2], dtype=np.int32)
    points = np.asarray(complex.mesh.coordinates)
    faces = np.asarray(connectivity.faces, dtype=np.int32)[parents].copy()
    # Orient geometrically outward, including negatively parameterized cells.
    cells = np.concatenate(
        tuple(np.asarray(block.vertices) for block in complex.mesh.blocks)
    )
    centroids = np.mean(points[cells], axis=1)
    cell_faces = np.asarray(connectivity.cell_faces)
    owners = np.empty(complex.topology.entities(2).count, dtype=np.int32)
    owners[cell_faces.reshape(-1)] = np.repeat(np.arange(cells.shape[0]), 4)
    corners = points[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    inward = (
        np.sum(normals * (centroids[owners[parents]] - corners.mean(axis=1)), axis=1)
        > 0.0
    )
    faces[inward] = faces[inward][:, (0, 2, 1)]
    vertices = np.unique(faces)
    local_faces = np.searchsorted(vertices, faces).astype(np.int32)
    return (
        OrientedTriangleSurfaceComplex3D(points[vertices], local_faces),
        vertices,
        np.asarray(boundary.parent_indices[1], dtype=np.int32),
    )


def prepare_matching_maxwell_fem_bem_3d(
    complex: FiniteElementDeRhamComplex,
    interior_operator: AbstractLinearOperator,
    /,
    *,
    wavenumber: ArrayLike,
    wave_impedance: ArrayLike = 1.0,
    boundary_policy: MaxwellEFIEPolicy3D | None = None,
    policy: LinearSolvePolicy | None = None,
    residual_tolerance: float = 1e-5,
) -> PreparedMatchingMaxwellFEMBEM3D:
    """Construct the exact matching trace, its dual conormal and solve artifact.

    Only trimmed order-one tetrahedral FE complexes are admitted. Boundary
    current coefficients are BC coefficients on the genuine barycentric space;
    geometry/kernel accuracy retains the bounded refined RWG EFIE evidence.
    The volume curl-curl/material operator is supplied as a native operator.
    """
    from ..discretization.fem._de_rham import FiniteElementDeRhamComplex
    from ..exterior._traces import trace_map

    if not isinstance(complex, FiniteElementDeRhamComplex):
        raise TypeError("complex must be FiniteElementDeRhamComplex.")
    if complex.dimension != 3 or complex.family != "trimmed" or complex.order != 1:
        raise ValueError("Matching Maxwell requires trimmed order-one tetrahedral FE.")
    if complex.primal_twist != "untwisted" or complex.mesh.coordinates.shape[1] != 3:
        raise ValueError("Matching Maxwell requires untwisted electric forms in R3.")
    if not isinstance(interior_operator, AbstractLinearOperator):
        raise TypeError("interior_operator must be an AbstractLinearOperator.")
    volume = complex.hilbert_complex().space(1)
    if not isinstance(volume, ArraySpace):
        raise TypeError("Matching Maxwell requires an array-valued FE edge space.")
    if not (
        interior_operator.source.compatible(volume)
        or interior_operator.source.compatible(coordinate_space(volume))
    ):
        raise ValueError("Interior operator source must be the declared FE edge space.")
    dtype = np.dtype(volume.structure().dtype)
    if not np.issubdtype(dtype, np.complexfloating):
        raise TypeError(
            "Matching time-harmonic Maxwell requires complex FE coefficients."
        )
    surface, vertices, parent_edges = _matching_maxwell_surface(complex)
    rwg = RWGSurfaceCurrentSpace3D(surface, coefficient_dtype=dtype)
    bc = prepare_buffa_christiansen_dual_3d(rwg)
    boundary = prepare_maxwell_bc_efie_3d(
        bc,
        wavenumber,
        wave_impedance=wave_impedance,
        policy=boundary_policy,
    )
    restriction = trace_map(complex).maps[1]
    connectivity = complex.mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise ValueError("Matching Maxwell requires tetrahedral connectivity.")
    edges = np.asarray(connectivity.edges)
    lookup = {
        tuple(edge): index for index, edge in enumerate(edges[parent_edges].tolist())
    }
    surface_edges = vertices[np.asarray(surface.edge_vertices)]
    permutation = jnp.asarray(
        [lookup[tuple(edge)] for edge in surface_edges.tolist()], dtype=jnp.int32
    )
    scale = -1.0 / surface.edge_lengths.astype(dtype)
    trace_id = canonical_fingerprint(
        {
            "kind": "matching-maxwell-rwg-trace",
            "map": restriction.operator_id,
            "surface": surface.complex_id,
        }
    )

    def trace_action(value: Array) -> Array:
        restricted = restriction.target.flatten(
            restriction.mv(restriction.source.unflatten(value))
        )
        return scale * restricted[permutation]

    def trace_transpose(value: Array) -> Array:
        restricted = (
            jnp.zeros((restriction.target.size,), dtype=dtype)
            .at[permutation]
            .add(scale * value)
        )
        return restriction.source.flatten(
            restriction.transpose_mv(restriction.target.unflatten(restricted))
        )

    rwg_trace = FunctionLinearOperator(
        trace_action,
        source=volume,
        target=rwg.vector_space,
        transpose_action=trace_transpose,
        operator_id=trace_id,
    )
    cross = bc.cross_mass.matrix.astype(dtype)
    dual_id = canonical_fingerprint(
        {"kind": "matching-maxwell-dual-trace", "trace": trace_id, "bc": bc.space_id}
    )

    def dual_action(value: Array) -> Array:
        return cross.T @ rwg_trace.mv(value)

    def dual_transpose(value: Array) -> Array:
        return rwg_trace.transpose_mv(cross @ value)

    dual_trace = FunctionLinearOperator(
        dual_action,
        source=volume,
        target=DualSpace(bc.vector_space),
        transpose_action=dual_transpose,
        operator_id=dual_id,
    )

    def conormal_action(value: Array) -> Array:
        return jnp.conj(dual_transpose(jnp.conj(value)))

    def conormal_transpose(value: Array) -> Array:
        return jnp.conj(dual_action(jnp.conj(value)))

    dual_conormal = FunctionLinearOperator(
        conormal_action,
        source=bc.vector_space,
        target=DualSpace(volume),
        transpose_action=conormal_transpose,
        operator_id=f"{dual_id}:conormal",
    )
    magnetic_scale = (
        1j * boundary.refined_efie.wavenumber * boundary.refined_efie.wave_impedance
    )

    def magnetic_load(value: Array) -> Array:
        return magnetic_scale * dual_conormal.mv(boundary.magnetic_trace.mv(value))

    def magnetic_load_transpose(value: Array) -> Array:
        return magnetic_scale * boundary.magnetic_trace.transpose_mv(
            dual_conormal.transpose_mv(value)
        )

    magnetic_conormal = FunctionLinearOperator(
        magnetic_load,
        source=bc.vector_space,
        target=DualSpace(volume),
        transpose_action=magnetic_load_transpose,
        operator_id=f"{dual_id}:magnetic-conormal",
    )

    def forward_conormal(value: Array) -> Array:
        return jnp.conj(magnetic_load_transpose(jnp.conj(value)))

    def forward_conormal_transpose(value: Array) -> Array:
        return jnp.conj(magnetic_load(jnp.conj(value)))

    conormal_test = FunctionLinearOperator(
        forward_conormal,
        source=volume,
        target=DualSpace(bc.vector_space),
        transpose_action=forward_conormal_transpose,
        operator_id=f"{dual_id}:magnetic-test",
    )
    coupled = _prepare(
        interior_operator,
        boundary.operator,
        dual_trace,
        conormal_test,
        "matching-maxwell",
        (
            restriction.operator_id,
            bc.evidence.evidence_id,
            boundary.assembly_report.report_id,
        ),
        policy,
        residual_tolerance,
        boundary,
    )
    return PreparedMatchingMaxwellFEMBEM3D(
        complex,
        boundary,
        rwg_trace,
        dual_trace,
        dual_conormal,
        magnetic_conormal,
        coupled,
        coupled.prepared_id,
    )
