#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import assert_never

import equinox as eqx
import jax.numpy as jnp
import numpy as np
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
from .._trainable import ExplicitFreeze, NonTrainableState
from ..discretization._topology import EntitySelection
from ..discretization.fem._generic import FiniteElementDiscretization
from ..discretization.fem._interface_trace import (
    prepare_matching_scalar_interface_trace_3d,
    PreparedMatchingScalarInterfaceTrace3D,
)
from ..geometry import MeshRegion
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    BlockLinearOperator,
    BlockSpace,
    DiagonalLinearOperator,
    DifferentiationMode,
    DifferentiationPolicy,
    estimate_operator_action_cost,
    FailurePolicy,
    FGMRES,
    IdentityLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    prepare as prepare_linear,
    PreparedLinearSolve,
    refresh as refresh_linear,
    solve as solve_linear,
    transpose,
)
from ..operators.integral.layer_potential._laplace3d import (
    evaluate_laplace_layer_3d,
    LaplaceLayerPotential3D,
)
from ..operators.integral.layer_potential._scalar_calderon3d import (
    ScalarCalderonDP0Galerkin3D,
)
from ..operators.integral.layer_potential._surface3d import SurfaceTargetReport3D
from ..sparse import EdgeRelation, SparseLinearMap


_FORMULATION = (
    "projected Johnson-Nedelec: a_kappa(u,v)-<phi,gamma v>=(f,v)+<g_N,gamma v>; "
    "<psi,(1/2-K)(P0(gamma u)-g_D)+V phi>=0, with exact facet-average P0, "
    "piecewise-constant interior conductivity kappa, g_D=gamma0- u-gamma0+ u, "
    "and g_N=kappa gamma1- u-gamma1+ u"
)
_PDE = (
    "scalar interior -div(kappa grad u)=f with piecewise-constant kappa / "
    "homogeneous decaying unit-conductivity exterior Laplace with DP0 transmission jumps"
)
_GEOMETRY = "complete affine P1 tetrahedral exterior matched bijectively to one closed outward triangular DP0 surface"
_NORMAL = (
    "n points from the FEM interior into the exterior; phi is gamma1+ using this "
    "same n, not the exterior-domain outward normal"
)
_NON_GOALS = (
    "nonmatching or mortar coupling",
    "partial, open, curved, moving, higher-order, or two-dimensional interfaces",
    "vector, elasticity, Stokes, Helmholtz, or acoustic equations",
    "geometry, kernel, quadrature, or exterior-medium derivatives",
    "dense operator fallback",
    "continuum certification or discretization-error estimation",
)
_RUNTIME_DATA = ("volume_source_coefficients", "dirichlet_jump", "conormal_jump")
_CONDUCTIVITY = "conductivity"
_FIXED_REFUSALS = {
    "geometry": (
        "The tetrahedral mesh, closed surface panelization, and bijective interface "
        "matching are prepared once on the host; geometry derivatives are not qualified."
    ),
    "kernel": (
        "The Laplace Calderon kernel has no runtime parameter; kernel derivatives "
        "are not qualified."
    ),
    "quadrature": (
        "Calderon quadrature orders, pair classes, and exact facet-average routes are "
        "fixed preparation decisions."
    ),
    "exterior_conductivity": (
        "The exterior medium has the unit conductivity fixed by the Calderon kernel; "
        "it is not a runtime argument."
    ),
}


class ScalarLaplaceFEMBEMResult3D(StrictModule, ExplicitFreeze):
    """One solved matching-interface scalar 3D Poisson/Laplace transmission state.

    The geometry, side convention, Johnson--Nédélec formulation, concrete FEM
    and scalar Calderón providers, precision and resource evidence, and
    non-goals are carried explicitly.  ``valid`` certifies only the reported
    finite-dimensional solve, positive finite conductivity, and preparation
    checks; it never certifies the continuum solution.  ``derivative_valid``
    separately reports whether the published ``derivative_capability`` holds
    for this evaluation: it requires an admitted route, a valid primal result,
    and a converged solve.  Derivatives of a result that is not valid are NaN
    (status failure mode) or raise (error failure mode).  Exterior layer
    potentials are built from the retained Cauchy data on request, outside the
    traced solve.
    """

    interior_coefficients: Array
    exterior_dirichlet_trace: Array
    exterior_conormal: Array
    interior_element_conormal: Array
    volume_source_coefficients: Array
    volume_load: Array
    dirichlet_jump: Array
    conormal_jump: Array
    conductivity: Array
    calderon: ScalarCalderonDP0Galerkin3D
    linear_result: LinearSolveResult
    relative_block_residual: Array
    interface_equation_defect: Array
    flux_balance_defect: Array
    conormal_mismatch_norm: Array
    bem_quadrature_maximum_errors: Array
    bem_quadrature_evaluations: Array
    valid: Array
    derivative_valid: Array
    derivative_capability: OwnerDerivativeCapability
    spatial_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry_contract: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider_ids: tuple[str, str, str] = eqx.field(static=True)
    precision_evidence: tuple[str, ...] = eqx.field(static=True)
    resource_evidence: tuple[tuple[str, int], ...] = eqx.field(static=True)
    error_evidence: tuple[str, ...] = eqx.field(static=True)
    normal_convention: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def exterior_layer_potentials(
        self,
    ) -> tuple[LaplaceLayerPotential3D, LaplaceLayerPotential3D]:
        """Return the exterior double layer of ``gamma0+ u`` and single layer of ``gamma1+ u``."""

        double_layer = self.calderon.double_layer_potential(self.exterior_dirichlet_trace)
        single_layer = self.calderon.single_layer_potential(self.exterior_conormal)
        # Preparation admits only the Laplace Calderon kernel family.
        if not isinstance(double_layer, LaplaceLayerPotential3D) or not isinstance(
            single_layer, LaplaceLayerPotential3D
        ):
            raise RuntimeError(
                "Scalar Calderon layer potentials must be Laplace potentials."
            )
        return double_layer, single_layer

    def evaluate_exterior(
        self,
        targets: ArrayLike,
        /,
        *,
        accuracy_clearance: float = 0.0,
    ) -> tuple[Array, tuple[SurfaceTargetReport3D, SurfaceTargetReport3D]]:
        """Evaluate ``D gamma0+ - S gamma1+`` at certified exterior targets."""

        double_layer, single_layer = self.exterior_layer_potentials()
        double_values, double_report = evaluate_laplace_layer_3d(
            double_layer,
            targets,
            target_side="exterior",
            accuracy_clearance=accuracy_clearance,
        )
        single_values, single_report = evaluate_laplace_layer_3d(
            single_layer,
            targets,
            target_side="exterior",
            accuracy_clearance=accuracy_clearance,
        )
        return double_values - single_values, (double_report, single_report)


class PreparedScalarLaplaceFEMBEM3D(StrictModule, NonTrainableState):
    """Prepared matching 3D scalar P1 FEM / DP0 BEM transmission product.

    This product is bounded to interior ``-div(kappa grad u)=f`` with a
    positive piecewise-constant conductivity ``kappa`` and a homogeneous,
    decaying unit-conductivity exterior Laplace field on one exactly matching
    closed triangular interface, with optional DP0 transmission jumps
    ``g_D=gamma0- u-gamma0+ u`` and ``g_N=kappa gamma1- u-gamma1+ u``.  It uses
    the nonsymmetric, stable projected Johnson--Nédélec block: the affine FEM
    Dirichlet trace enters the DP0 Calderón equation through its exact
    facet-average projection, while the conormal pairing in the FEM equation is
    integrated exactly.  The normal is the outward-from-interior
    ``gamma0+``/``gamma1+`` convention.

    The interior stiffness is ``G^T diag(kappa |T|) G`` with the exact affine P1
    cell-gradient map ``G`` prepared once; a runtime conductivity changes only
    the diagonal and rebinds the prepared solve numerically.  The existing FEM
    mass provider, scalar DP0 Calderón provider, exact algebraic transpose
    actions, precision, blocked-action resources, quadrature/interface error
    evidence, and the owner derivative capability selected by the linear
    policy are retained.  No continuum certification is claimed.

    Non-goals include public coupling graphs, nonmatching mortar methods,
    partial/open or curved interfaces, 2D, vector/acoustic PDEs, geometry or
    kernel derivatives, hidden dense fallback, and moving/high-order geometry.
    """

    discretization: FiniteElementDiscretization
    surface: MeshRegion
    calderon: ScalarCalderonDP0Galerkin3D
    interface: PreparedMatchingScalarInterfaceTrace3D
    mass_operator: AbstractLinearOperator
    cell_gradient_operator: SparseLinearMap
    cell_volumes: Array
    conductivity_space: ArraySpace
    exterior_trace_relation: AbstractLinearOperator
    operator: BlockLinearOperator
    prepared_linear: PreparedLinearSolve
    linear_policy: LinearSolvePolicy
    bem_quadrature_maximum_errors: Array
    bem_quadrature_evaluations: Array
    derivative_capability: OwnerDerivativeCapability
    spatial_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry_contract: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider_ids: tuple[str, str, str] = eqx.field(static=True)
    precision_evidence: tuple[str, ...] = eqx.field(static=True)
    resource_evidence: tuple[tuple[str, int], ...] = eqx.field(static=True)
    error_evidence: tuple[str, ...] = eqx.field(static=True)
    normal_convention: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def volume_load(self, volume_source_coefficients: ArrayLike, /) -> Array:
        """Assemble ``(f,v)`` from the P1 nodal interpolant of scalar ``f``."""

        values = self.mass_operator.source.validate(volume_source_coefficients)
        return self.mass_operator.mv(values)

    def stiffness(
        self, conductivity: ArrayLike | None = None, /
    ) -> AbstractLinearOperator:
        """Return ``a_kappa`` as ``G^T diag(kappa |T|) G``; ``None`` is unit conductivity.

        The conductivity holds one value per tetrahedron in mesh-cell order.
        """

        kappa = (
            jnp.ones_like(self.cell_volumes)
            if conductivity is None
            else self.conductivity_space.validate(conductivity)
        )
        return _conductivity_stiffness(
            self.cell_gradient_operator, self.cell_volumes, kappa
        )

    def right_hand_side(
        self,
        volume_source_coefficients: ArrayLike,
        /,
        *,
        dirichlet_jump: ArrayLike | None = None,
        conormal_jump: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        """Return ``((f,v)+<g_N,gamma v>, (1/2-K) g_D)`` for the two block rows."""

        load = self.volume_load(volume_source_coefficients)
        space = self.calderon.space
        dirichlet = (
            space.zeros() if dirichlet_jump is None else space.validate(dirichlet_jump)
        )
        conormal = (
            space.zeros() if conormal_jump is None else space.validate(conormal_jump)
        )
        return (
            load + self.interface.boundary_load(conormal),
            self.exterior_trace_relation.mv(dirichlet),
        )

    def solve(
        self,
        volume_source_coefficients: ArrayLike,
        /,
        *,
        dirichlet_jump: ArrayLike | None = None,
        conormal_jump: ArrayLike | None = None,
        conductivity: ArrayLike | None = None,
    ) -> ScalarLaplaceFEMBEMResult3D:
        """Solve the prepared finite-dimensional transmission problem."""

        return solve_scalar_laplace_fem_bem_3d(
            self,
            volume_source_coefficients,
            dirichlet_jump=dirichlet_jump,
            conormal_jump=conormal_jump,
            conductivity=conductivity,
        )


def _default_linear_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        FGMRES(restart=30, stagnation_iterations=30),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )


def _conductivity_stiffness(
    gradient: SparseLinearMap,
    volumes: Array,
    conductivity: Array,
    /,
) -> AbstractLinearOperator:
    # Affine P1 gradients are constant per tetrahedron, so G^T diag(kappa |T|) G
    # is the exact P1 stiffness of a piecewise-constant conductivity.
    weights = jnp.repeat(conductivity * volumes, gradient.target.size // volumes.size)
    weight = DiagonalLinearOperator(
        weights,
        space=gradient.target,
        operator_id=canonical_fingerprint(
            {
                "kind": "scalar-fem-bem-cell-conductivity-weight-3d",
                "gradient": gradient.operator_id,
            }
        ),
    )
    return transpose(gradient) @ weight @ gradient


def _cell_gradient_route(
    discretization: FiniteElementDiscretization,
    field_name: str,
    /,
) -> tuple[SparseLinearMap, Array]:
    """Prepare the exact affine P1 cell-gradient map and cell volumes on the host."""

    field_index = discretization._field_index(field_name)
    geometry = discretization.evaluate_geometry(
        field_name, discretization.default_runtime.coordinates
    )
    if len(geometry) != 1:
        raise ValueError("The supported FEM envelope is one affine tetrahedron block.")
    gradients = np.asarray(geometry[0].physical_gradients, dtype=np.float64)
    volumes = np.asarray(geometry[0].measure, dtype=np.float64)
    cell_dofs = np.asarray(discretization.dof_maps[field_index].cell_dofs[0])
    if gradients.ndim != 4 or gradients.shape[2:] != (4, 3):
        raise ValueError("Scalar FEM-BEM requires affine P1 tetrahedral gradients.")
    variation = np.max(np.abs(gradients - gradients[:, :1]))
    scale = max(float(np.max(np.abs(gradients))), 1.0)
    if variation > 64.0 * np.finfo(np.float64).eps * scale:
        raise ValueError("Scalar FEM-BEM requires cellwise-constant P1 gradients.")
    cells = cell_dofs.shape[0]
    rows = np.arange(3 * cells, dtype=np.int32).reshape(cells, 3)
    relation = EdgeRelation(
        np.broadcast_to(cell_dofs[:, None, :], (cells, 3, 4)).reshape(-1),
        np.broadcast_to(rows[:, :, None], (cells, 3, 4)).reshape(-1),
        source_size=discretization.field_spaces[field_index].vector_space.size,
        target_size=3 * cells,
    )
    gradient = SparseLinearMap(
        relation,
        jnp.asarray(np.swapaxes(gradients[:, 0], 1, 2).reshape(-1)),
        operator_id=canonical_fingerprint(
            {
                "kind": "scalar-fem-bem-p1-cell-gradient-3d",
                "fem": discretization.prepared_id,
                "field": field_name,
            }
        ),
    )
    return gradient, jnp.asarray(volumes)


def _validate_calderon(
    calderon: ScalarCalderonDP0Galerkin3D,
    surface: MeshRegion,
    /,
) -> None:
    if calderon.panelization.atlas.source_id != surface.feature_id:
        raise ValueError("The scalar Calderon provider is not bound to surface.")
    convention = calderon.trace_convention
    if (
        convention.ambient_dimension != 3
        or convention.normal_orientation != "interior-to-exterior"
        or convention.double_layer_dirichlet_jump("exterior") != 0.5
    ):
        raise ValueError(
            "The scalar Calderon provider has an incompatible exterior trace or normal convention."
        )
    if (
        calderon.face_count != calderon.space.size
        or calderon.panelization.panel_count != calderon.face_count
    ):
        raise ValueError("The scalar Calderon DP0 face routes are inconsistent.")
    report = calderon.assembly_report
    if calderon.kernel.family != "laplace" or report.pde != "-Delta(u)=0":
        raise ValueError("Scalar FEM-BEM preparation requires the Laplace kernel.")
    if not bool(report.finite) or not bool(report.accuracy_supported):
        raise ValueError(
            "Scalar Calderon quadrature does not support this prepared coupling."
        )
    if (
        report.ambient_dimension != 3
        or report.materializable
        or report.continuum_discretization_error_estimated
    ):
        raise ValueError("Scalar Calderon evidence is outside the bounded contract.")


def _select_policy(linear: LinearSolvePolicy | None, /) -> LinearSolvePolicy:
    policy = _default_linear_policy() if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be a LinearSolvePolicy or None.")
    match policy.differentiation.mode:
        case "none" | "rhs-only" | "mathematical":
            return policy
        case "algorithmic":
            raise ValueError(
                "The scalar FEM-BEM solve admits differentiation mode 'none', "
                "'rhs-only', or 'mathematical'; unrolled 'algorithmic' Krylov "
                "derivatives are not the solution-map derivative and are not qualified."
            )
        case invalid:
            assert_never(invalid)


def _derivative_capability(
    owner_id: str,
    mode: DifferentiationMode,
    /,
) -> OwnerDerivativeCapability:
    """Owner-qualified derivatives of one prepared scalar FEM-BEM solve map."""

    admitted: dict[str, DerivativeSurface] = {}
    refused = dict(_FIXED_REFUSALS)
    match mode:
        case "none":
            reason = "Prepared with differentiation mode 'none'; no derivative route is admitted."
            refused.update({name: reason for name in (*_RUNTIME_DATA, _CONDUCTIVITY)})
        case "rhs-only":
            admitted.update(
                {name: DerivativeSurface.SOLVER_ARGUMENT for name in _RUNTIME_DATA}
            )
            refused[_CONDUCTIVITY] = (
                "Prepared with differentiation mode 'rhs-only': the interior stiffness "
                "arrays are stopped, so conductivity derivatives require mode "
                "'mathematical'."
            )
        case "mathematical":
            admitted.update(
                {name: DerivativeSurface.SOLVER_ARGUMENT for name in _RUNTIME_DATA}
            )
            admitted[_CONDUCTIVITY] = DerivativeSurface.PHYSICAL_PARAMETER
        case "algorithmic":
            raise ValueError("Algorithmic FEM-BEM derivatives are not qualified.")
        case invalid:
            assert_never(invalid)
    return OwnerDerivativeCapability(
        owner_id,
        admitted=admitted,
        refused=refused,
        route=DerivativeRoute.IMPLICIT if admitted else DerivativeRoute.STOPPED,
        conditions=(
            ("accepted-result", "prepared-geometry-fixed", "solve-converged")
            if admitted
            else ("prepared-geometry-fixed",)
        ),
    )


def prepare_scalar_laplace_fem_bem_3d(
    discretization: FiniteElementDiscretization,
    surface: MeshRegion,
    calderon: ScalarCalderonDP0Galerkin3D,
    /,
    *,
    field_name: str = "u",
    interface_selection: EntitySelection | None = None,
    coordinate_tolerance: float | None = None,
    linear: LinearSolvePolicy | None = None,
) -> PreparedScalarLaplaceFEMBEM3D:
    """Prepare the fixed Johnson--Nédélec P1/DP0 scalar coupling block.

    Geometry, interface routes, Calderón operators, the P1 cell-gradient map,
    and the linear solve plan are prepared once.  ``linear`` selects the
    derivative route: ``'none'`` (default) admits no derivative,
    ``'rhs-only'`` admits the volume source and transmission jumps, and
    ``'mathematical'`` additionally admits the interior conductivity; all are
    implicit solution-map derivatives.  ``'algorithmic'`` is refused.
    """

    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be a FiniteElementDiscretization.")
    if not isinstance(surface, MeshRegion):
        raise TypeError("surface must be a MeshRegion.")
    if not isinstance(calderon, ScalarCalderonDP0Galerkin3D):
        raise TypeError("calderon must be a ScalarCalderonDP0Galerkin3D.")
    _validate_calderon(calderon, surface)
    report = calderon.assembly_report
    policy = _select_policy(linear)

    coupling = prepare_matching_scalar_interface_trace_3d(
        discretization,
        surface,
        calderon.space,
        field_name=field_name,
        interface=interface_selection,
        coordinate_tolerance=coordinate_tolerance,
    )
    mass, _ = discretization.assemble_field_operators(
        field_name, discretization.default_runtime
    )
    gradient, volumes = _cell_gradient_route(discretization, str(field_name))
    stiffness = _conductivity_stiffness(gradient, volumes, jnp.ones_like(volumes))
    identity = IdentityLinearOperator(calderon.space)
    exterior_trace_relation = 0.5 * identity - calderon.double_layer
    bottom_left = exterior_trace_relation @ coupling.trace_operator
    block_space = BlockSpace(
        (mass.source, calderon.space),
        names=("interior_field", "exterior_conormal"),
    )
    operator = BlockLinearOperator(
        (
            (stiffness, -coupling.boundary_load_operator),
            (bottom_left, calderon.single_layer),
        ),
        source=block_space,
        target=block_space,
        operator_id=canonical_fingerprint(
            {
                "kind": "scalar-laplace-fem-bem-johnson-nedelec-3d",
                "fem": discretization.prepared_id,
                "interior": stiffness.operator_id,
                "interface": coupling.prepared_id,
                "single_layer": calderon.single_layer.operator_id,
                "double_layer": calderon.double_layer.operator_id,
                "normal": _NORMAL,
            }
        ),
    )
    problem_id = canonical_fingerprint(
        {
            "kind": "scalar-laplace-fem-bem-linear-system-3d",
            "operator": operator.operator_id,
        }
    )
    prepared_linear = prepare_linear(
        LinearSystem(operator, problem_id=problem_id), policy
    )
    cost = estimate_operator_action_cost(operator)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-scalar-laplace-fem-bem-3d",
            "operator": operator.operator_id,
            "fem": discretization.prepared_id,
            "surface": surface.feature_id,
            "calderon": calderon.panelization.panelization_id,
            "linear_plan": prepared_linear.plan.plan_id,
        }
    )
    resources = coupling.resource_evidence + (
        ("block_unknowns", operator.source.size),
        ("bem_preparation_workspace_bytes", report.preparation_workspace_bytes),
        ("bem_resident_bytes", report.resident_bytes),
        (
            "bem_action_workspace_bytes_per_rhs",
            report.action_workspace_bytes_per_rhs,
        ),
        ("operator_storage_bytes", cost.storage_bytes),
        ("operator_action_workspace_bytes_per_rhs", cost.apply_workspace_bytes_per_rhs),
    )
    errors = coupling.error_evidence + (
        f"scalar Calderon assembly report {report.report_id}",
        f"operator action cost exact={cost.exact}: {cost.reason}",
        "linear diagnostics are returned per solve",
        "continuum discretization error is not estimated",
    )
    return PreparedScalarLaplaceFEMBEM3D(
        discretization=discretization,
        surface=surface,
        calderon=calderon,
        interface=coupling,
        mass_operator=mass,
        cell_gradient_operator=gradient,
        cell_volumes=volumes,
        conductivity_space=ArraySpace(volumes.shape, dtype=volumes.dtype),
        exterior_trace_relation=exterior_trace_relation,
        operator=operator,
        prepared_linear=prepared_linear,
        linear_policy=policy,
        bem_quadrature_maximum_errors=report.quadrature_maximum_errors,
        bem_quadrature_evaluations=report.quadrature_evaluations,
        derivative_capability=_derivative_capability(
            prepared_id, policy.differentiation.mode
        ),
        spatial_dimension=3,
        pde=_PDE,
        geometry_contract=_GEOMETRY,
        formulation=_FORMULATION,
        provider_ids=(
            discretization.prepared_id,
            calderon.panelization.panelization_id,
            coupling.prepared_id,
        ),
        precision_evidence=(
            discretization.precision_policy.policy_id,
            report.precision_policy_id,
            str(calderon.space.dtype),
        ),
        resource_evidence=resources,
        error_evidence=errors,
        normal_convention=_NORMAL,
        non_goals=_NON_GOALS,
        field_name=str(field_name),
        prepared_id=prepared_id,
    )


def _bound_linear(
    prepared: PreparedScalarLaplaceFEMBEM3D,
    conductivity: Array | None,
    /,
) -> tuple[BlockLinearOperator, PreparedLinearSolve]:
    """Rebind the prepared solve numerically for a runtime conductivity."""

    if conductivity is None:
        return prepared.operator, prepared.prepared_linear
    blocks = prepared.operator.blocks
    operator = BlockLinearOperator(
        (
            (
                _conductivity_stiffness(
                    prepared.cell_gradient_operator,
                    prepared.cell_volumes,
                    conductivity,
                ),
                blocks[0][1],
            ),
            blocks[1],
        ),
        source=prepared.operator.source,
        target=prepared.operator.target,
        operator_id=prepared.operator.operator_id,
    )
    problem = LinearSystem(
        operator, problem_id=prepared.prepared_linear.problem.problem_id
    )
    return operator, refresh_linear(prepared.prepared_linear, problem)


def _refuse_unadmitted(
    result: ScalarLaplaceFEMBEMResult3D,
    capability: OwnerDerivativeCapability,
    arguments: dict[str, Array],
    /,
) -> ScalarLaplaceFEMBEMResult3D:
    """Raise at transformation time for derivatives through unadmitted arguments."""

    refused = tuple(name for name in arguments if not capability.admits(name))
    if not refused:
        return result
    reasons = dict(capability.refused)
    return refuse_derivative_dependencies(
        result,
        tuple(arguments[name] for name in refused),
        message=(
            f"prepared scalar FEM-BEM solve {capability.owner_id} refuses derivatives "
            "with respect to "
            + "; ".join(f"{name!r} ({reasons[name]})" for name in refused)
        ),
    )


def _block_evidence(
    operator: BlockLinearOperator,
    state: tuple[Array, Array],
    right_hand_side: tuple[Array, Array],
    /,
) -> tuple[Array, Array]:
    """Relative residual of both original block rows and the interface-row defect."""

    image = operator.mv(state)
    first_residual = image[0] - right_hand_side[0]
    second_residual = image[1] - right_hand_side[1]
    residual_squared = (
        jnp.vdot(first_residual, first_residual).real
        + jnp.vdot(second_residual, second_residual).real
    )
    right_squared = (
        jnp.vdot(right_hand_side[0], right_hand_side[0]).real
        + jnp.vdot(right_hand_side[1], right_hand_side[1]).real
    )
    relative_residual = jnp.sqrt(residual_squared) / jnp.maximum(
        jnp.sqrt(right_squared), jnp.asarray(1.0, dtype=right_squared.dtype)
    )
    interface_defect = jnp.sqrt(jnp.vdot(second_residual, second_residual).real)
    return relative_residual, interface_defect


def solve_scalar_laplace_fem_bem_3d(
    prepared: PreparedScalarLaplaceFEMBEM3D,
    volume_source_coefficients: ArrayLike,
    /,
    *,
    dirichlet_jump: ArrayLike | None = None,
    conormal_jump: ArrayLike | None = None,
    conductivity: ArrayLike | None = None,
) -> ScalarLaplaceFEMBEMResult3D:
    """Solve a prepared interior-source / decaying-exterior transmission case.

    ``dirichlet_jump`` and ``conormal_jump`` are DP0 facet data (default zero);
    ``conductivity`` holds one positive value per tetrahedron (default one).  A
    runtime conductivity rebinds the prepared solve numerically without
    replanning.  Derivatives follow ``prepared.derivative_capability``: a
    derivative request through an argument it does not admit raises
    ``ValueError`` at transformation time instead of returning a silent zero.
    """

    if not isinstance(prepared, PreparedScalarLaplaceFEMBEM3D):
        raise TypeError("prepared must be a PreparedScalarLaplaceFEMBEM3D.")
    source = prepared.mass_operator.source.validate(volume_source_coefficients)
    space = prepared.calderon.space
    dirichlet = (
        space.zeros() if dirichlet_jump is None else space.validate(dirichlet_jump)
    )
    jump = space.zeros() if conormal_jump is None else space.validate(conormal_jump)
    kappa = (
        None
        if conductivity is None
        else prepared.conductivity_space.validate(conductivity)
    )
    kappa_values = jnp.ones_like(prepared.cell_volumes) if kappa is None else kappa
    volume_load = prepared.mass_operator.mv(source)
    right_hand_side = (
        volume_load + prepared.interface.boundary_load(jump),
        prepared.exterior_trace_relation.mv(dirichlet),
    )
    operator, linear = _bound_linear(prepared, kappa)
    linear_result = solve_linear(linear, right_hand_side)
    interior, conormal = operator.source.validate(linear_result.value)
    relative_residual, interface_defect = _block_evidence(
        operator, (interior, conormal), right_hand_side
    )
    flux_balance = jnp.sum(volume_load) + prepared.interface.integrated_flux(
        conormal + jump
    )
    interior_conormal = kappa_values[
        prepared.interface.owner_cells
    ] * prepared.interface.conormal(interior)
    mismatch = interior_conormal - conormal - jump
    conormal_mismatch = jnp.sqrt(jnp.vdot(mismatch, mismatch).real)
    finite = (
        jnp.all(jnp.isfinite(interior))
        & jnp.all(jnp.isfinite(conormal))
        & jnp.isfinite(relative_residual)
        & jnp.isfinite(interface_defect)
        & jnp.isfinite(flux_balance)
        & jnp.isfinite(conormal_mismatch)
    )
    admissible = jnp.all(jnp.isfinite(kappa_values) & (kappa_values > 0.0))
    valid = (
        linear_result.successful & linear_result.diagnostics.finite & finite & admissible
    )
    capability = prepared.derivative_capability
    admitted = capability.derivative_contract.route is not DerivativeRoute.STOPPED
    if admitted:
        # Every returned derivative-bearing path of the solve (including the
        # raw linear result and the residual evidence) shares one validity guard.
        (
            interior,
            conormal,
            interior_conormal,
            linear_result,
            relative_residual,
            interface_defect,
            flux_balance,
            conormal_mismatch,
        ) = guard_derivative_validity(
            (
                interior,
                conormal,
                interior_conormal,
                linear_result,
                relative_residual,
                interface_defect,
                flux_balance,
                conormal_mismatch,
            ),
            valid,
            failure=prepared.linear_policy.failure.mode,
            message=(
                "The scalar FEM-BEM result is not valid; its solution-map derivative "
                "is refused."
            ),
        )
    trace = prepared.interface.trace(interior) - dirichlet
    result = ScalarLaplaceFEMBEMResult3D(
        interior_coefficients=interior,
        exterior_dirichlet_trace=trace,
        exterior_conormal=conormal,
        interior_element_conormal=interior_conormal,
        volume_source_coefficients=source,
        volume_load=volume_load,
        dirichlet_jump=dirichlet,
        conormal_jump=jump,
        conductivity=kappa_values,
        calderon=prepared.calderon,
        linear_result=linear_result,
        relative_block_residual=relative_residual,
        interface_equation_defect=interface_defect,
        flux_balance_defect=flux_balance,
        conormal_mismatch_norm=conormal_mismatch,
        bem_quadrature_maximum_errors=prepared.bem_quadrature_maximum_errors,
        bem_quadrature_evaluations=prepared.bem_quadrature_evaluations,
        valid=valid,
        derivative_valid=(
            valid & jnp.all(linear_result.derivative_valid)
            if admitted
            else jnp.asarray(False)
        ),
        derivative_capability=capability,
        spatial_dimension=prepared.spatial_dimension,
        pde=prepared.pde,
        geometry_contract=prepared.geometry_contract,
        formulation=prepared.formulation,
        provider_ids=prepared.provider_ids,
        precision_evidence=prepared.precision_evidence,
        resource_evidence=prepared.resource_evidence,
        error_evidence=prepared.error_evidence,
        normal_convention=prepared.normal_convention,
        non_goals=prepared.non_goals,
        prepared_id=prepared.prepared_id,
    )
    return _refuse_unadmitted(
        result,
        capability,
        {
            "volume_source_coefficients": source,
            "dirichlet_jump": dirichlet,
            "conormal_jump": jump,
            _CONDUCTIVITY: kappa_values,
        },
    )


__all__ = [
    "PreparedScalarLaplaceFEMBEM3D",
    "ScalarLaplaceFEMBEMResult3D",
    "prepare_scalar_laplace_fem_bem_3d",
    "solve_scalar_laplace_fem_bem_3d",
]
