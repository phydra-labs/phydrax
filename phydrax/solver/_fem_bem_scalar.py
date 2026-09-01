#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._topology import EntitySelection
from ..discretization.fem._generic import FiniteElementDiscretization
from ..discretization.fem._interface_trace import (
    prepare_matching_scalar_interface_trace_3d,
    PreparedMatchingScalarInterfaceTrace3D,
)
from ..geometry import MeshRegion
from ..linalg import (
    AbstractLinearOperator,
    BlockLinearOperator,
    BlockSpace,
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
    solve as solve_linear,
)
from ..operators.integral.layer_potential._laplace3d import (
    evaluate_laplace_layer_3d,
    LaplaceLayerPotential3D,
)
from ..operators.integral.layer_potential._scalar_calderon3d import (
    ScalarCalderonDP0Galerkin3D,
)


_FORMULATION = (
    "projected Johnson-Nedelec: a(u,v)-<phi,gamma v>=(f,v); "
    "<psi,(1/2-K)P0(gamma u)+V phi>=0, with exact facet-average P0"
)
_PDE = "scalar interior Poisson / homogeneous decaying exterior Laplace"
_GEOMETRY = (
    "complete affine P1 tetrahedral exterior matched bijectively to one closed "
    "outward triangular DP0 surface"
)
_NORMAL = (
    "n points from the FEM interior into the exterior; phi is gamma1+ using this "
    "same n, not the exterior-domain outward normal"
)
_NON_GOALS = (
    "nonmatching or mortar coupling",
    "partial, open, curved, moving, higher-order, or two-dimensional interfaces",
    "vector, elasticity, Stokes, Helmholtz, or acoustic equations",
    "dense operator fallback",
    "continuum certification or discretization-error estimation",
)


class ScalarLaplaceFEMBEMResult3D(StrictModule, NonTrainableState):
    """One solved matching-interface scalar 3D Poisson/Laplace transmission state.

    The geometry, side convention, Johnson--Nédélec formulation, concrete FEM
    and scalar Calderón providers, precision and resource evidence, and
    non-goals are carried explicitly.  ``valid`` certifies only the reported
    finite-dimensional solve and preparation checks; it never certifies the
    continuum solution.
    """

    interior_coefficients: Array
    exterior_dirichlet_trace: Array
    exterior_conormal: Array
    interior_element_conormal: Array
    volume_source_coefficients: Array
    volume_load: Array
    exterior_double_layer: LaplaceLayerPotential3D
    exterior_single_layer: LaplaceLayerPotential3D
    linear_result: LinearSolveResult
    relative_block_residual: Array
    interface_equation_defect: Array
    flux_balance_defect: Array
    conormal_mismatch_norm: Array
    bem_quadrature_maximum_errors: Array
    bem_quadrature_evaluations: Array
    valid: Array
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

    def evaluate_exterior(
        self,
        targets: ArrayLike,
        /,
        *,
        accuracy_clearance: float = 0.0,
    ):
        """Evaluate ``D gamma0+ - S gamma1+`` at certified exterior targets."""

        double_values, double_report = evaluate_laplace_layer_3d(
            self.exterior_double_layer,
            targets,
            target_side="exterior",
            accuracy_clearance=accuracy_clearance,
        )
        single_values, single_report = evaluate_laplace_layer_3d(
            self.exterior_single_layer,
            targets,
            target_side="exterior",
            accuracy_clearance=accuracy_clearance,
        )
        return double_values - single_values, (double_report, single_report)


class PreparedScalarLaplaceFEMBEM3D(StrictModule, NonTrainableState):
    """Prepared matching 3D scalar P1 FEM / DP0 BEM transmission product.

    This product is bounded to interior ``-Delta u=f`` and a homogeneous,
    decaying exterior Laplace field on one exactly matching closed triangular
    interface.  It uses the nonsymmetric, stable projected Johnson--Nédélec
    block: the affine FEM Dirichlet trace enters the DP0 Calderón equation
    through its exact facet-average projection, while the conormal pairing in
    the FEM equation is integrated exactly.  The normal is the
    outward-from-interior ``gamma0+``/``gamma1+`` convention.  The existing FEM
    mass/stiffness provider, scalar DP0 Calderón provider, exact algebraic
    transpose actions, precision, blocked-action resources, and
    quadrature/interface error evidence are retained.  No continuum
    certification is claimed.

    Non-goals include public coupling graphs, nonmatching mortar methods,
    partial/open or curved interfaces, 2D, vector/acoustic PDEs, hidden dense
    fallback, and moving/high-order geometry.
    """

    discretization: FiniteElementDiscretization
    surface: MeshRegion
    calderon: ScalarCalderonDP0Galerkin3D
    interface: PreparedMatchingScalarInterfaceTrace3D
    mass_operator: AbstractLinearOperator
    stiffness_operator: AbstractLinearOperator
    exterior_trace_relation: AbstractLinearOperator
    operator: BlockLinearOperator
    prepared_linear: PreparedLinearSolve
    linear_policy: LinearSolvePolicy
    bem_quadrature_maximum_errors: Array
    bem_quadrature_evaluations: Array
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

    def right_hand_side(self, volume_source_coefficients: ArrayLike, /):
        """Return the interior volume load and homogeneous exterior equation."""

        load = self.volume_load(volume_source_coefficients)
        return load, self.calderon.space.zeros()

    def solve(
        self,
        volume_source_coefficients: ArrayLike,
        /,
    ) -> ScalarLaplaceFEMBEMResult3D:
        """Solve the prepared finite-dimensional transmission problem."""

        return solve_scalar_laplace_fem_bem_3d(self, volume_source_coefficients)


def _default_linear_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        FGMRES(restart=30, stagnation_iterations=30),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
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
    """Prepare the fixed Johnson--Nédélec P1/DP0 scalar coupling block."""

    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be a FiniteElementDiscretization.")
    if not isinstance(surface, MeshRegion):
        raise TypeError("surface must be a MeshRegion.")
    if not isinstance(calderon, ScalarCalderonDP0Galerkin3D):
        raise TypeError("calderon must be a ScalarCalderonDP0Galerkin3D.")
    if calderon.panelization.atlas.source_id != surface.feature_id:
        raise ValueError("The scalar Calderon provider is not bound to surface.")
    convention = calderon.trace_convention
    if (
        convention.ambient_dimension != 3
        or convention.normal_orientation != "interior-to-exterior"
        or convention.double_layer_dirichlet_jump("exterior") != 0.5
    ):
        raise ValueError(
            "The scalar Calderon provider has an incompatible exterior trace "
            "or normal convention."
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
    policy = _default_linear_policy() if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be a LinearSolvePolicy or None.")
    if policy.differentiation.mode != "none":
        raise ValueError(
            "The bounded scalar FEM-BEM solve requires differentiation mode 'none'."
        )

    coupling = prepare_matching_scalar_interface_trace_3d(
        discretization,
        surface,
        calderon.space,
        field_name=field_name,
        interface=interface_selection,
        coordinate_tolerance=coordinate_tolerance,
    )
    mass, stiffness = discretization.assemble_field_operators(
        field_name, discretization.default_runtime
    )
    identity = IdentityLinearOperator(calderon.space)
    exterior_trace_relation = 0.5 * identity - calderon.double_layer
    bottom_left = exterior_trace_relation @ coupling.trace_operator
    block_space = BlockSpace(
        (stiffness.source, calderon.space),
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
        stiffness_operator=stiffness,
        exterior_trace_relation=exterior_trace_relation,
        operator=operator,
        prepared_linear=prepared_linear,
        linear_policy=policy,
        bem_quadrature_maximum_errors=report.quadrature_maximum_errors,
        bem_quadrature_evaluations=report.quadrature_evaluations,
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


def solve_scalar_laplace_fem_bem_3d(
    prepared: PreparedScalarLaplaceFEMBEM3D,
    volume_source_coefficients: ArrayLike,
    /,
) -> ScalarLaplaceFEMBEMResult3D:
    """Solve a prepared interior-source / homogeneous-exterior transmission case."""

    if not isinstance(prepared, PreparedScalarLaplaceFEMBEM3D):
        raise TypeError("prepared must be a PreparedScalarLaplaceFEMBEM3D.")
    source = prepared.mass_operator.source.validate(volume_source_coefficients)
    volume_load = prepared.mass_operator.mv(source)
    right_hand_side = (volume_load, prepared.calderon.space.zeros())
    linear_result = solve_linear(prepared.prepared_linear, right_hand_side)
    interior, conormal = prepared.operator.source.validate(linear_result.value)
    trace = prepared.interface.trace(interior)
    image = prepared.operator.mv((interior, conormal))
    first_residual = image[0] - volume_load
    second_residual = image[1]
    residual_squared = (
        jnp.vdot(first_residual, first_residual).real
        + jnp.vdot(second_residual, second_residual).real
    )
    right_squared = jnp.vdot(volume_load, volume_load).real
    relative_residual = jnp.sqrt(residual_squared) / jnp.maximum(
        jnp.sqrt(right_squared), jnp.asarray(1.0, dtype=volume_load.dtype)
    )
    interface_defect = jnp.sqrt(jnp.vdot(second_residual, second_residual).real)
    flux_balance = jnp.sum(volume_load) + prepared.interface.integrated_flux(conormal)
    interior_conormal = prepared.interface.conormal(interior)
    conormal_mismatch = jnp.sqrt(
        jnp.vdot(interior_conormal - conormal, interior_conormal - conormal).real
    )
    double_layer = prepared.calderon.double_layer_potential(trace)
    single_layer = prepared.calderon.single_layer_potential(conormal)
    finite = (
        jnp.all(jnp.isfinite(interior))
        & jnp.all(jnp.isfinite(conormal))
        & jnp.isfinite(relative_residual)
        & jnp.isfinite(interface_defect)
        & jnp.isfinite(flux_balance)
        & jnp.isfinite(conormal_mismatch)
    )
    valid = linear_result.successful & linear_result.diagnostics.finite & finite
    return ScalarLaplaceFEMBEMResult3D(
        interior_coefficients=interior,
        exterior_dirichlet_trace=trace,
        exterior_conormal=conormal,
        interior_element_conormal=interior_conormal,
        volume_source_coefficients=source,
        volume_load=volume_load,
        exterior_double_layer=double_layer,
        exterior_single_layer=single_layer,
        linear_result=linear_result,
        relative_block_residual=relative_residual,
        interface_equation_defect=interface_defect,
        flux_balance_defect=flux_balance,
        conormal_mismatch_norm=conormal_mismatch,
        bem_quadrature_maximum_errors=prepared.bem_quadrature_maximum_errors,
        bem_quadrature_evaluations=prepared.bem_quadrature_evaluations,
        valid=valid,
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


__all__ = [
    "PreparedScalarLaplaceFEMBEM3D",
    "ScalarLaplaceFEMBEMResult3D",
    "prepare_scalar_laplace_fem_bem_3d",
    "solve_scalar_laplace_fem_bem_3d",
]
