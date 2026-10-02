#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DiagonalLinearOperator,
    FailurePolicy,
    GMRES,
    ILUPreconditionerBuilder,
    LinearSolveControl,
    LinearSolveDiagnostics,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    plan_sparse_assembly,
    PreconditioningPolicy,
    prepare,
    prepare_sparse_assembly,
    PreparedLinearSolve,
    PreparedSparseAssembly,
    refresh,
    refresh_sparse_assembly,
    solve,
    SparseAssemblyPolicy,
    TolerancePolicy,
    transpose,
)
from ..sparse import SparseCoordinateOperator
from ..typing import checked, parse
from ._point_cloud import PreparedPointCloudDiscretization


if TYPE_CHECKING:
    from .meshfree._multilevel import PreparedMeshfreeHierarchy

PointBoundaryKind: TypeAlias = Literal["dirichlet", "neumann", "robin"]
PointDiffusionForm: TypeAlias = Literal["collocated", "dissipative"]
PointPoissonPreconditioner: TypeAlias = Literal["multilevel", "ilu", "none"]
PointNeumannCompatibility: TypeAlias = Literal["refuse", "project"]


@final
class PointBoundaryPlan(StrictModule):
    kind: PointBoundaryKind = eqx.field(static=True)
    values: Array
    robin_coefficient: Array | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: PointBoundaryKind,
        values: ArrayLike,
        /,
        *,
        robin_coefficient: ArrayLike | None = None,
    ) -> None:
        kind = parse(kind, PointBoundaryKind, "kind")
        values_ = jnp.asarray(values)
        if values_.ndim != 1:
            raise ValueError("Point boundary values must be a vector.")
        values_ = eqx.error_if(
            values_,
            jnp.any(~jnp.isfinite(values_)),
            "Point boundary values must be finite.",
        )
        if kind == "robin":
            if robin_coefficient is None:
                raise ValueError("Robin boundaries require robin_coefficient.")
            coefficient = jnp.broadcast_to(jnp.asarray(robin_coefficient), values_.shape)
            coefficient = eqx.error_if(
                coefficient,
                jnp.any(~jnp.isfinite(coefficient)) | jnp.any(coefficient < 0.0),
                "Robin coefficients must be finite and nonnegative.",
            )
        else:
            if robin_coefficient is not None:
                raise ValueError("Only Robin boundaries accept robin_coefficient.")
            coefficient = None
        self.kind = kind
        self.values = values_
        self.robin_coefficient = coefficient
        self.plan_id = canonical_fingerprint(
            {
                "kind": "point-boundary-plan",
                "boundary_kind": kind,
                "values": array_tree_fingerprint(values_),
                "coefficient": None
                if coefficient is None
                else array_tree_fingerprint(coefficient),
            }
        )


@final
class PointSBPReport(StrictModule, NonTrainableState):
    maximum_green_residual: float = eqx.field(static=True)
    maximum_conservation_residual: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)
    evidence_scope: Literal["full-sparse-coefficient-identity"] = eqx.field(
        static=True, default="full-sparse-coefficient-identity"
    )


def point_sbp_report(
    discretization: PreparedPointCloudDiscretization,
    /,
    *,
    tolerance: float = 1e-8,
    maximum_coefficients: int = 1_000_000,
) -> PointSBPReport:
    """Check every coefficient of MD + DᵀM = Bn, at host preparation.

    The conservation check is 1ᵀMD = 1ᵀBn, not a zero boundary flux claim.
    Resource refusal never substitutes a finite set of polynomial probes.
    """
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("SBP tolerance must be finite and nonnegative.")
    budget = int(maximum_coefficients)
    count, width = discretization.relation.source_indices.shape
    if budget < 1 or 2 * count * width + count > budget:
        raise ValueError("Full sparse SBP identity exceeds maximum_coefficients.")
    boundary_weights = discretization.plan.boundary_quadrature_weights
    if boundary_weights is None:
        raise ValueError("Point SBP evidence requires boundary_quadrature_weights.")
    mass = np.asarray(discretization.quadrature_weights)
    boundary = np.asarray(boundary_weights)
    indices = np.asarray(discretization.relation.source_indices)
    valid = np.asarray(discretization.relation.valid)
    normals = np.asarray(discretization.plan.boundary_normals)
    maximum_green = maximum_conservation = 0.0
    for axis, (first, _) in enumerate(discretization.derivative_weights):
        entries: dict[tuple[int, int], float] = {}
        conservation = -boundary * normals[:, axis]
        weights = np.asarray(first)
        for row in range(count):
            for route in range(width):
                if not valid[row, route]:
                    continue
                column = int(indices[row, route])
                value = float(mass[row] * weights[row, route])
                entries[row, column] = entries.get((row, column), 0.0) + value
                entries[column, row] = entries.get((column, row), 0.0) + value
                conservation[column] += value
            entries[row, row] = entries.get((row, row), 0.0) - float(
                boundary[row] * normals[row, axis]
            )
        maximum_green = max(
            maximum_green, max((abs(v) for v in entries.values()), default=0.0)
        )
        maximum_conservation = max(
            maximum_conservation, float(np.max(np.abs(conservation)))
        )
    identifier = canonical_fingerprint(
        {
            "kind": "point-sbp-report",
            "discretization": discretization.prepared_id,
            "green": maximum_green,
            "conservation": maximum_conservation,
            "tolerance": threshold,
        }
    )
    return PointSBPReport(
        maximum_green,
        maximum_conservation,
        maximum_green <= threshold and maximum_conservation <= threshold,
        identifier,
    )


@final
class PointDiffusionOperator(StrictModule, NonTrainableState):
    """Collocated +div(k grad), or the quadrature-adjoint action -M⁻¹ ΣDᵀMkD.

    Collocation applies the variable-coefficient product rule. Dissipative
    energy is nonpositive in the quadrature pairing, not necessarily in ℓ².
    Energy stability alone does not imply continuum consistency for arbitrary
    quadrature/non-SBP derivatives; no approximation order is certified here.
    """

    discretization: PreparedPointCloudDiscretization
    diffusivity: Array
    form: PointDiffusionForm = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: PreparedPointCloudDiscretization,
        diffusivity: ArrayLike = 1.0,
        /,
        *,
        form: PointDiffusionForm = "dissipative",
    ) -> None:
        form = parse(form, PointDiffusionForm, "form")
        coefficient = jnp.broadcast_to(
            jnp.asarray(diffusivity, dtype=jnp.float64), discretization.state_shape
        )
        coefficient = eqx.error_if(
            coefficient,
            jnp.any(~jnp.isfinite(coefficient)) | jnp.any(coefficient <= 0.0),
            "Point diffusivity must be finite and positive.",
        )
        self.discretization = discretization
        self.diffusivity = coefficient
        self.form = form
        self.operator_id = canonical_fingerprint(
            {
                "kind": "point-diffusion",
                "discretization": discretization.prepared_id,
                "form": form,
                "diffusivity": array_tree_fingerprint(coefficient),
            }
        )

    def mv(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.ndim < 1 or value.shape[0] != self.discretization.state_shape[0]:
            raise ValueError("Point diffusion values must begin with the point count.")
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        mass = self.discretization.quadrature_weights.reshape(shape)
        coefficient = self.diffusivity.reshape(shape)
        output = jnp.zeros_like(value)
        for axis in range(self.discretization.spatial_dimension):
            derivative = self.discretization.partial_derivative(value, axis=axis)
            if self.form == "dissipative":
                output = (
                    output
                    - self.discretization.transpose_partial_derivative(
                        mass * coefficient * derivative, axis=axis
                    )
                    / mass
                )
            else:
                coefficient_gradient = self.discretization.partial_derivative(
                    self.diffusivity, axis=axis
                ).reshape(shape)
                output = (
                    output
                    + coefficient
                    * self.discretization.partial_derivative(value, axis=axis, order=2)
                    + coefficient_gradient * derivative
                )
        return output

    def energy_rate(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        return jnp.real(
            jnp.vdot(
                value,
                self.discretization.quadrature_weights.reshape(shape) * self.mv(value),
            )
        )

    def stiffness(self) -> AbstractLinearOperator:
        """Native sparse-composable positive-sign elliptic operator.

        Dissipative returns -M L; collocated returns -L.
        """
        d = self.discretization
        space = ArraySpace(
            d.state_shape,
            dtype=self.diffusivity.dtype,
            space_id=f"{d.prepared_id}:point-scalar-coordinates",
        )
        terms: list[AbstractLinearOperator] = []
        for axis, (first, second) in enumerate(d.derivative_weights):
            gradient = SparseCoordinateOperator(
                d.relation,
                first,
                source=space,
                target=space,
                operator_id=f"{d.prepared_id}:point-first-derivative:{axis}",
            )
            if self.form == "dissipative":
                weighted = DiagonalLinearOperator(
                    d.quadrature_weights * self.diffusivity, space=space
                )
                terms.append(transpose(gradient) @ weighted @ gradient)
            else:
                second_map = SparseCoordinateOperator(
                    d.relation,
                    second,
                    source=space,
                    target=space,
                    operator_id=f"{d.prepared_id}:point-second-derivative:{axis}",
                )
                terms.append(
                    -(DiagonalLinearOperator(self.diffusivity, space=space) @ second_map)
                    - DiagonalLinearOperator(
                        d.partial_derivative(self.diffusivity, axis=axis), space=space
                    )
                    @ gradient
                )
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result


@final
class PointCloudPoissonResult(StrictModule):
    values: Array
    residual_norm: Array
    compatible: Array
    source_correction: Array
    compatibility_residual: Array
    gauge_residual: Array
    boundary_residual_norm: Array
    linear_result: LinearSolveResult
    correction_linear_result: LinearSolveResult | None
    residual_tolerance: Array

    @property
    def successful(self) -> Array:
        correction_ok = (
            jnp.asarray(True)
            if self.correction_linear_result is None
            else self.correction_linear_result.successful
        )
        return (
            self.linear_result.successful
            & correction_ok
            & jnp.isfinite(self.residual_norm)
            & (self.residual_norm <= self.residual_tolerance)
            & (self.boundary_residual_norm <= self.residual_tolerance)
            & (self.gauge_residual <= self.residual_tolerance)
        )

    @property
    def status(self) -> Array:
        return self.linear_result.status

    @property
    def diagnostics(self) -> LinearSolveDiagnostics:
        return self.linear_result.diagnostics


@final
class PointCloudPoissonPlan(StrictModule):
    discretization: PreparedPointCloudDiscretization
    boundary: PointBoundaryPlan
    form: PointDiffusionForm = eqx.field(static=True)
    preconditioner: PointPoissonPreconditioner = eqx.field(static=True)
    compatibility: PointNeumannCompatibility = eqx.field(static=True)
    tolerance: TolerancePolicy
    assembly_policy: SparseAssemblyPolicy
    gauge_index: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: PreparedPointCloudDiscretization,
        boundary: PointBoundaryPlan,
        /,
        *,
        form: PointDiffusionForm = "collocated",
        preconditioner: PointPoissonPreconditioner = "multilevel",
        compatibility: PointNeumannCompatibility = "refuse",
        tolerance: TolerancePolicy | None = None,
        assembly_policy: SparseAssemblyPolicy | None = None,
        gauge_index: int | None = None,
    ) -> None:
        if not isinstance(
            discretization, PreparedPointCloudDiscretization
        ) or not isinstance(boundary, PointBoundaryPlan):
            raise TypeError(
                "Poisson requires a prepared point cloud and PointBoundaryPlan."
            )
        if boundary.values.shape != discretization.state_shape:
            raise ValueError("Boundary values must match point count.")
        mask = np.asarray(discretization.plan.boundary_mask)
        if not np.any(mask):
            raise ValueError("Poisson requires explicit boundary points.")
        if boundary.kind == "robin" and not np.any(
            np.asarray(boundary.robin_coefficient)[mask] > 0
        ):
            raise ValueError(
                "Zero-coefficient Robin is Neumann; choose neumann with an explicit compatibility policy."
            )
        interior = np.flatnonzero(~mask)
        if interior.size == 0:
            raise ValueError("Poisson requires at least one interior point.")
        gauge = int(interior[0]) if gauge_index is None else int(gauge_index)
        if gauge < 0 or gauge >= mask.size or mask[gauge]:
            raise ValueError("Gauge must select an interior point.")
        form_ = parse(form, PointDiffusionForm, "form")
        preconditioner_ = parse(
            preconditioner, PointPoissonPreconditioner, "preconditioner"
        )
        compatibility_ = parse(compatibility, PointNeumannCompatibility, "compatibility")
        tolerance_ = (
            TolerancePolicy(relative=1e-9, absolute=1e-10, max_steps=1000)
            if tolerance is None
            else tolerance
        )
        assembly_policy_ = (
            SparseAssemblyPolicy() if assembly_policy is None else assembly_policy
        )
        if not isinstance(tolerance_, TolerancePolicy) or not isinstance(
            assembly_policy_, SparseAssemblyPolicy
        ):
            raise TypeError("Invalid Poisson tolerance or assembly policy.")
        identifier = canonical_fingerprint(
            {
                "kind": "point-poisson-plan",
                "discretization": discretization.prepared_id,
                "boundary": boundary.plan_id,
                "form": form_,
                "preconditioner": preconditioner_,
                "compatibility": compatibility_,
                "gauge": gauge,
            }
        )
        self.discretization = discretization
        self.boundary = boundary
        self.form = form_
        self.preconditioner = preconditioner_
        self.compatibility = compatibility_
        self.tolerance = tolerance_
        self.assembly_policy = assembly_policy_
        self.gauge_index = gauge
        self.plan_id = identifier

    def prepare(self, diffusivity: ArrayLike = 1.0, /) -> PreparedPointCloudPoisson:
        return PreparedPointCloudPoisson(self, diffusivity)


def _poisson_operators(
    plan: PointCloudPoissonPlan, diffusion: PointDiffusionOperator
) -> tuple[AbstractLinearOperator, AbstractLinearOperator]:
    d = plan.discretization
    stiffness = diffusion.stiffness()
    space = stiffness.source
    mask = d.plan.boundary_mask
    fixed = DiagonalLinearOperator(mask.astype(diffusion.diffusivity.dtype), space=space)
    free = DiagonalLinearOperator(
        (~mask).astype(diffusion.diffusivity.dtype), space=space
    )
    if plan.boundary.kind == "dirichlet":
        return stiffness, free @ stiffness @ free + fixed
    conormal_weights = jnp.zeros_like(d.derivative_weights[0][0])
    for axis, (first, _) in enumerate(d.derivative_weights):
        conormal_weights = (
            conormal_weights
            + diffusion.diffusivity[:, None]
            * d.plan.boundary_normals[:, axis, None]
            * first
        )
    conormal = SparseCoordinateOperator(
        d.relation, conormal_weights, source=space, target=space
    )
    if plan.boundary.kind == "robin":
        robin_coefficient = plan.boundary.robin_coefficient
        if robin_coefficient is None:
            raise ValueError("Robin boundaries require robin_coefficient.")
        conormal = conormal + DiagonalLinearOperator(robin_coefficient, space=space)
    physical = free @ stiffness + fixed @ conormal
    if plan.boundary.kind != "neumann":
        return physical, physical
    gauge = (
        jnp.zeros(d.state_shape, dtype=diffusion.diffusivity.dtype)
        .at[plan.gauge_index]
        .set(1.0)
    )
    keep = DiagonalLinearOperator(1.0 - gauge, space=space)
    return physical, keep @ physical + DiagonalLinearOperator(gauge, space=space)


@final
class PreparedPointCloudPoisson(StrictModule, NonTrainableState):
    """Reusable sparse assembly and native solve; preparation is host-side.

    Neumann uses one explicit point gauge. Compatibility is checked against
    the complete original algebraic equations. Projection changes the interior
    source by a reported constant, determined from the removed equation—not a
    quadrature integral. Additional nullspaces cause native solve refusal.

    Collocated form targets the continuum product-rule equation. Dissipative
    form solves the selected quadrature-adjoint equation; independent accuracy
    and full sparse SBP evidence must be assessed separately.
    """

    plan: PointCloudPoissonPlan
    diffusion: PointDiffusionOperator
    physical_assembly: PreparedSparseAssembly
    assembly: PreparedSparseAssembly
    linear_solve: PreparedLinearSolve
    hierarchy: PreparedMeshfreeHierarchy | None

    @checked
    def __init__(
        self, plan: PointCloudPoissonPlan, diffusivity: ArrayLike = 1.0, /
    ) -> None:
        diffusion = PointDiffusionOperator(
            plan.discretization, diffusivity, form=plan.form
        )
        physical, gauged = _poisson_operators(plan, diffusion)
        physical_assembly = prepare_sparse_assembly(
            plan_sparse_assembly(physical, plan.assembly_policy), physical
        )
        assembly = prepare_sparse_assembly(
            plan_sparse_assembly(gauged, plan.assembly_policy), gauged
        )
        preconditioning = None
        hierarchy = None
        if plan.preconditioner == "ilu":
            preconditioning = PreconditioningPolicy(
                ILUPreconditionerBuilder(), refresh="numeric"
            )
        elif plan.preconditioner == "multilevel":
            from .meshfree._multilevel import (
                meshfree_multigrid_builder,
                MeshfreeHierarchyPlan,
            )

            boundary = plan.discretization.plan.boundary_mask
            if plan.boundary.kind == "neumann":
                boundary = boundary.at[plan.gauge_index].set(True)
            fine_space = assembly.operator.source
            if not isinstance(fine_space, ArraySpace):
                raise TypeError(
                    "Point Poisson multilevel requires scalar ArraySpace coordinates."
                )
            hierarchy = MeshfreeHierarchyPlan(
                plan.discretization.plan.points, boundary=boundary
            ).prepare(fine_space)
            preconditioning = PreconditioningPolicy(
                meshfree_multigrid_builder(
                    hierarchy, coarse_solver=ILUPreconditionerBuilder()
                )
            )
        policy = LinearSolvePolicy(
            GMRES(restart=min(40, plan.discretization.state_shape[0])),
            tolerance=plan.tolerance,
            preconditioning=preconditioning,
            failure=FailurePolicy("error"),
        )
        linear_solve = prepare(
            LinearSystem(assembly.operator, problem_id=plan.plan_id), policy
        )
        self.plan = plan
        self.diffusion = diffusion
        self.physical_assembly = physical_assembly
        self.assembly = assembly
        self.linear_solve = linear_solve
        self.hierarchy = hierarchy

    def refresh(self, diffusivity: ArrayLike, /) -> PreparedPointCloudPoisson:
        diffusion = PointDiffusionOperator(
            self.plan.discretization, diffusivity, form=self.plan.form
        )
        physical, gauged = _poisson_operators(self.plan, diffusion)
        physical_assembly = refresh_sparse_assembly(self.physical_assembly, physical)
        assembly = refresh_sparse_assembly(self.assembly, gauged)
        linear_solve = refresh(
            self.linear_solve,
            LinearSystem(assembly.operator, problem_id=self.plan.plan_id),
        )
        return eqx.tree_at(
            lambda p: (p.diffusion, p.physical_assembly, p.assembly, p.linear_solve),
            self,
            (diffusion, physical_assembly, assembly, linear_solve),
        )

    def solve(
        self, source: ArrayLike, /, *, boundary_values: ArrayLike | None = None
    ) -> PointCloudPoissonResult:
        d = self.plan.discretization
        dtype = self.diffusion.diffusivity.dtype
        source_ = jnp.asarray(source, dtype=dtype)
        values = (
            self.plan.boundary.values
            if boundary_values is None
            else jnp.asarray(boundary_values, dtype=dtype)
        )
        if source_.shape != d.state_shape or values.shape != d.state_shape:
            raise ValueError("Poisson source/boundary values must match point count.")
        source_ = eqx.error_if(
            source_,
            jnp.any(~jnp.isfinite(source_)) | jnp.any(~jnp.isfinite(values)),
            "Poisson source and boundary values must be finite.",
        )
        mask = d.plan.boundary_mask
        scale = (
            d.quadrature_weights
            if self.plan.form == "dissipative"
            else jnp.ones_like(source_)
        )
        physical_rhs = jnp.where(mask, values, scale * source_)
        lift = jnp.where(mask, values, 0.0)
        rhs = physical_rhs
        control = None
        if self.plan.boundary.kind == "dirichlet":
            rhs = jnp.where(
                mask, values, scale * source_ - self.physical_assembly.operator.mv(lift)
            )
            # Lifting can amplify the algebraic RHS. Stop against the requested
            # original-equation tolerance, not the artificially enlarged norm.
            physical_tolerance = (
                self.plan.tolerance.absolute
                + self.plan.tolerance.relative * jnp.linalg.norm(physical_rhs)
            )
            control = LinearSolveControl(
                relative_tolerance=0.0, absolute_tolerance=physical_tolerance
            )
        elif self.plan.boundary.kind == "neumann":
            rhs = rhs.at[self.plan.gauge_index].set(0.0)
        linear_result = solve(self.linear_solve, rhs, control=control)
        solution = linear_result.value
        correction = jnp.zeros_like(source_)
        correction_result = None
        residual_before = self.physical_assembly.operator.mv(solution) - physical_rhs
        if self.plan.boundary.kind == "dirichlet":
            residual_before = jnp.where(mask, solution - values, residual_before)
        compatibility_residual = jnp.linalg.norm(residual_before)
        threshold = (
            self.plan.tolerance.absolute
            + self.plan.tolerance.relative * jnp.linalg.norm(physical_rhs)
        )
        compatible = compatibility_residual <= threshold
        if self.plan.boundary.kind == "neumann" and self.plan.compatibility == "project":
            direction = jnp.where(mask, 0.0, scale)
            direction_rhs = direction.at[self.plan.gauge_index].set(0.0)
            correction_result = solve(self.linear_solve, direction_rhs)
            response = correction_result.value
            denominator = (self.physical_assembly.operator.mv(response) - direction)[
                self.plan.gauge_index
            ]
            denominator = eqx.error_if(
                denominator,
                ~jnp.isfinite(denominator)
                | (
                    jnp.abs(denominator)
                    <= jnp.finfo(dtype).eps * jnp.linalg.norm(direction)
                ),
                "Neumann source projection direction is algebraically incompatible with the left nullspace.",
            )
            shift = -residual_before[self.plan.gauge_index] / denominator
            correction = jnp.where(mask, 0.0, shift)
            solution = solution + shift * response
        physical_rhs = physical_rhs + scale * correction
        residual = self.physical_assembly.operator.mv(solution) - physical_rhs
        if self.plan.boundary.kind == "dirichlet":
            residual = jnp.where(mask, solution - values, residual)
        norm = jnp.linalg.norm(residual)
        residual_tolerance = (
            self.plan.tolerance.absolute
            + self.plan.tolerance.relative * jnp.linalg.norm(physical_rhs)
        )
        solution = eqx.error_if(
            solution,
            ~jnp.isfinite(norm) | (norm > residual_tolerance),
            "Point Poisson refused: incompatible Neumann data or unresolved physical/boundary equations.",
        )
        gauge = (
            jnp.abs(solution[self.plan.gauge_index])
            if self.plan.boundary.kind == "neumann"
            else jnp.asarray(0.0, dtype=dtype)
        )
        return PointCloudPoissonResult(
            solution,
            norm,
            compatible,
            correction,
            compatibility_residual,
            gauge,
            jnp.linalg.norm(jnp.where(mask, residual, 0.0)),
            linear_result,
            correction_result,
            residual_tolerance,
        )


__all__ = [
    "PointBoundaryKind",
    "PointBoundaryPlan",
    "PointDiffusionForm",
    "PointDiffusionOperator",
    "PointPoissonPreconditioner",
    "PointNeumannCompatibility",
    "PointCloudPoissonPlan",
    "PreparedPointCloudPoisson",
    "PointCloudPoissonResult",
    "PointSBPReport",
    "point_sbp_report",
]
