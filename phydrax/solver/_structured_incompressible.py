#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_difference import (
    diagonalize_fd_laplacian,
    FDLaplacianSolvePlan,
)
from ..discretization.finite_volume import FaceVelocity, PreparedMACOperators
from ..discretization.finite_volume._mac_boundary import (
    MACBoundaryPlan,
    MACBoundaryStageData,
    MACPressureClosureKind,
    PreparedMACBoundaryPlan,
)
from ..linalg import (
    DiagonalPreconditioner,
    DifferentiationPolicy,
    FGMRES,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    PCG,
    PreconditioningPolicy,
    prepare,
    PreparedLinearSolve,
    refresh,
    solve,
    TolerancePolicy,
    TransformDiagonalSolveResult,
)
from ..linalg._transform_line import (
    PreparedTransformLineSolve,
    TransformLineNullspacePolicy,
    TransformLineRepresentation,
    TransformLineSolvePlan,
    TransformLineSolveResult,
)
from ._mac_separable import (
    certify_separable_action,
    diagonal_resource_counts,
    iterative_workspace_bytes,
    modal_sum,
    pressure_cell_axis_transform,
    pressure_cell_line_coefficients,
)


MACPressureSolveMethod: TypeAlias = Literal[
    "auto", "direct", "transform", "hybrid", "iterative"
]
MACPressureGaugeKind: TypeAlias = Literal["zero-mean", "none"]
MACPressureCompatibilityKind: TypeAlias = Literal["projected", "unprojected"]


class _WeightedMACPressureAction(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    face_inverse_momentum: FaceVelocity

    def __init__(
        self,
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        face_inverse_momentum: FaceVelocity,
        /,
    ):
        self.operators = operators
        self.boundaries = boundaries
        self.face_inverse_momentum = operators.validate_velocity(face_inverse_momentum)

    def __call__(self, pressure: Array, /) -> Array:
        value = self.operators.validate_pressure(pressure)
        if self.boundaries.closure_kind == "neumann":
            volumes = self.operators.discretization.cell_volumes.astype(value.dtype)
            mean = jnp.sum(volumes * value) / jnp.sum(volumes)
            value = value - mean
        else:
            mean = jnp.asarray(0.0, dtype=value.dtype)
        gradient = self.boundaries.pressure_gradient(value, None, homogeneous=True)
        weighted = tuple(
            coefficient * derivative
            for coefficient, derivative in zip(
                self.face_inverse_momentum, gradient, strict=True
            )
        )
        return -self.operators.divergence(weighted) + mean


class MACPressureClosureReport(StrictModule, NonTrainableState):
    """Closure-dependent gauge, compatibility, and integrated mass evidence."""

    kind: MACPressureClosureKind = eqx.field(static=True)
    gauge: MACPressureGaugeKind = eqx.field(static=True)
    compatibility: MACPressureCompatibilityKind = eqx.field(static=True)
    integrated_mass_flux: Array
    mass_defect: Array
    gauge_defect: Array
    finite: Array
    successful: Array
    closure_id: str = eqx.field(static=True)


class MACPressureProjectionResult(StrictModule):
    velocity: FaceVelocity
    pressure: Array
    pressure_increment: Array
    divergence_before: Array
    divergence_after: Array
    pressure_residual: Array
    compatible_rhs: Array
    face_inverse_momentum: FaceVelocity
    gauge_defect: Array
    closure: MACPressureClosureReport
    solve_method: str = eqx.field(static=True)
    route_reason: str = eqx.field(static=True)
    linear: LinearSolveResult | None
    transform: TransformDiagonalSolveResult | None
    hybrid: TransformLineSolveResult | None
    hybrid_action_defect: Array
    hybrid_line_axis: int | None = eqx.field(static=True)
    maximum_resource_bytes: int = eqx.field(static=True)
    finite: Array
    converged: Array
    projection_id: str = eqx.field(static=True)


class MACRateProjectionResult(StrictModule):
    """Projected momentum rate and its pressure Lagrange multiplier."""

    rate: FaceVelocity
    pressure: Array
    divergence_before: Array
    divergence_after: Array
    pressure_residual: Array
    compatible_rhs: Array
    face_inverse_density: FaceVelocity
    gauge_defect: Array
    closure: MACPressureClosureReport
    solve_method: str = eqx.field(static=True)
    route_reason: str = eqx.field(static=True)
    linear: LinearSolveResult | None
    transform: TransformDiagonalSolveResult | None
    hybrid: TransformLineSolveResult | None
    hybrid_action_defect: Array
    hybrid_line_axis: int | None = eqx.field(static=True)
    maximum_resource_bytes: int = eqx.field(static=True)
    finite: Array
    converged: Array
    projection_id: str = eqx.field(static=True)


class MACPressureProjectionPlan(StrictModule, NonTrainableState):
    """Prepared compatible MAC projection with closure-aware pressure solves."""

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    density: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    solve_method: MACPressureSolveMethod = eqx.field(static=True)
    closure_kind: MACPressureClosureKind = eqx.field(static=True)
    gauge_kind: MACPressureGaugeKind = eqx.field(static=True)
    compatibility_kind: MACPressureCompatibilityKind = eqx.field(static=True)
    nonsymmetric_traction: bool = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    linear_problem: LinearSystem
    prepared_linear: PreparedLinearSolve
    transform_plan: FDLaplacianSolvePlan | None
    hybrid_plan: PreparedTransformLineSolve | None
    hybrid_action_defect: Array
    hybrid_line_axis: int | None = eqx.field(static=True)
    maximum_resource_bytes: int = eqx.field(static=True)
    constant_route: str = eqx.field(static=True)
    route_reason: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    pressure_problem_id: str = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        /,
        *,
        boundaries: PreparedMACBoundaryPlan | MACBoundaryPlan | None = None,
        density: float = 1.0,
        tolerance: float = 1e-9,
        maximum_iterations: int = 500,
        solve_method: MACPressureSolveMethod = "auto",
        linear_policy: LinearSolvePolicy | None = None,
        hybrid_line_axis: int | None = None,
        maximum_resource_bytes: int = 512 * 1024**2,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        boundaries_ = (
            MACBoundaryPlan(operators).prepare()
            if boundaries is None
            else boundaries.prepare()
            if isinstance(boundaries, MACBoundaryPlan)
            else boundaries
        )
        if not isinstance(boundaries_, PreparedMACBoundaryPlan):
            raise TypeError(
                "boundaries must be PreparedMACBoundaryPlan, MACBoundaryPlan, or None."
            )
        if boundaries_.operators.prepared_id != operators.prepared_id:
            raise ValueError("MAC projection boundaries must use the same operators.")
        density_ = float(density)
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        budget = int(maximum_resource_bytes)
        if (
            not np.isfinite(density_)
            or density_ <= 0.0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or iterations <= 0
            or budget <= 0
        ):
            raise ValueError(
                "Projection density, tolerance, iterations, and resources are invalid."
            )
        if solve_method not in ("auto", "direct", "transform", "hybrid", "iterative"):
            raise ValueError(
                "solve_method must be 'auto', 'direct', 'transform', 'hybrid', or "
                "'iterative'."
            )
        line_axis = None if hybrid_line_axis is None else int(hybrid_line_axis)
        dimension = len(operators.discretization.cell_shape)
        if line_axis is not None and (line_axis < 0 or line_axis >= dimension):
            raise ValueError("hybrid_line_axis is outside the MAC tensor rank.")
        if solve_method == "hybrid" and line_axis is None:
            raise ValueError(
                "Hybrid MAC projection requires an explicit hybrid_line_axis."
            )
        iterative_workspace_bytes(
            operators.discretization.cell_shape,
            operators.pressure_space.dtype,
            budget,
            "MAC pressure iterative",
        )
        closure_kind = boundaries_.closure_kind
        gauge_kind: MACPressureGaugeKind = (
            "zero-mean" if closure_kind == "neumann" else "none"
        )
        compatibility_kind: MACPressureCompatibilityKind = (
            "projected" if closure_kind == "neumann" else "unprojected"
        )
        closure_id = canonical_fingerprint(
            {
                "kind": "mac-pressure-closure",
                "boundaries": boundaries_.prepared_id,
                "closure": closure_kind,
                "gauge": gauge_kind,
                "compatibility": compatibility_kind,
            }
        )
        nonsymmetric_traction = any(
            boundary.kind == "traction-open" and boundary.backflow_coefficient > 0.0
            for boundary in boundaries_.sides
        )
        unit_face = tuple(
            jnp.ones(layout.shape, dtype=operators.pressure_space.dtype)
            for layout in operators.discretization.face_layouts
        )
        operator_id = canonical_fingerprint(
            {
                "kind": "mac-closure-pressure-operator",
                "operators": operators.prepared_id,
                "closure": closure_id,
                "nonsymmetric_traction": nonsymmetric_traction,
            }
        )
        pressure_operator = FunctionLinearOperator(
            _WeightedMACPressureAction(operators, boundaries_, unit_face),
            source=operators.pressure_space,
            target=operators.pressure_space,
            properties=OperatorProperties(
                self_adjoint=not nonsymmetric_traction,
                positive_definite=not nonsymmetric_traction,
                evidence=(
                    {}
                    if nonsymmetric_traction
                    else {
                        "self_adjoint": "construction",
                        "positive_definite": "construction",
                    }
                ),
            ),
            operator_id=operator_id,
        )
        pressure_problem_id = canonical_fingerprint(
            {"kind": "mac-pressure-system", "operator": operator_id}
        )
        problem = LinearSystem(pressure_operator, problem_id=pressure_problem_id)
        pressure_preconditioning = (
            None
            if nonsymmetric_traction
            else PreconditioningPolicy(
                DiagonalPreconditioner(
                    jnp.ones(
                        (operators.pressure_space.size,),
                        dtype=operators.pressure_space.dtype,
                    ),
                    space=operators.pressure_space,
                    positive_definite=True,
                    preconditioner_id=canonical_fingerprint(
                        {
                            "kind": "mac-pressure-constant-preconditioner",
                            "operators": operators.prepared_id,
                            "closure": closure_id,
                        }
                    ),
                ),
                side="left",
                refresh="frozen",
            )
        )
        policy = (
            LinearSolvePolicy(
                FGMRES(restart=min(30, iterations)) if nonsymmetric_traction else PCG(),
                tolerance=TolerancePolicy(
                    relative=tolerance_,
                    absolute=tolerance_,
                    max_steps=iterations,
                ),
                preconditioning=pressure_preconditioning,
                differentiation=DifferentiationPolicy("mathematical"),
            )
            if linear_policy is None
            else linear_policy
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        prepared_linear = prepare(problem, policy)
        transform_plan = (
            self._prepare_transform(operators, boundaries_, tolerance_, budget)
            if solve_method in ("auto", "direct", "transform")
            and not nonsymmetric_traction
            else None
        )
        if solve_method == "transform" and transform_plan is None:
            raise ValueError(
                "Transform MAC projection requires a uniform certified separable closure."
            )
        if solve_method == "direct" and transform_plan is None and line_axis is None:
            raise ValueError(
                "Explicit direct MAC projection has no certified transform route."
            )
        hybrid_plan = None
        hybrid_action_defect = jnp.asarray(jnp.inf, dtype=operators.pressure_space.dtype)
        if (
            not nonsymmetric_traction
            and line_axis is not None
            and (
                solve_method in ("direct", "hybrid")
                or (solve_method == "auto" and transform_plan is None)
            )
        ):
            hybrid_plan, hybrid_action_defect = self._prepare_hybrid(
                operators,
                boundaries_,
                line_axis,
                tolerance_,
                policy,
                budget,
            )
        if solve_method == "hybrid" and hybrid_plan is None:
            raise ValueError(
                "Hybrid MAC projection requires a certified symmetric line action."
            )
        if solve_method == "hybrid":
            constant_route = "hybrid"
            route_reason = "explicit certified transform-line pressure action"
        elif solve_method == "transform":
            constant_route = "transform"
            route_reason = "explicit certified tensor-transform pressure action"
        elif solve_method == "direct":
            if transform_plan is not None:
                constant_route = "transform"
                route_reason = "explicit direct request accepted by tensor action"
            elif hybrid_plan is not None:
                constant_route = "hybrid"
                route_reason = "explicit direct request accepted by transform-line action"
            else:
                raise ValueError(
                    "Explicit direct MAC projection has no certified exact representation."
                )
        elif solve_method == "iterative":
            constant_route = "iterative"
            route_reason = "explicit iterative pressure route"
        elif transform_plan is not None:
            constant_route = "transform"
            route_reason = "auto selected exact constant-coefficient tensor action"
        elif hybrid_plan is not None:
            constant_route = "hybrid"
            route_reason = "auto selected exact retained-line action"
        else:
            constant_route = "iterative"
            route_reason = (
                "auto selected FGMRES for stabilized nonsymmetric traction"
                if nonsymmetric_traction
                else "auto selected PCG because no exact action certified"
            )
        identifier = canonical_fingerprint(
            {
                "kind": "mac-pressure-projection-plan",
                "operators": operators.prepared_id,
                "boundaries": boundaries_.prepared_id,
                "density": density_,
                "tolerance": tolerance_,
                "solve_method": solve_method,
                "closure": closure_id,
                "linear_plan": prepared_linear.plan.plan_id,
                "transform_plan": (
                    None if transform_plan is None else transform_plan.plan_id
                ),
                "hybrid_plan": (
                    None if hybrid_plan is None else hybrid_plan.plan.plan_id
                ),
                "hybrid_line_axis": line_axis,
                "maximum_resource_bytes": budget,
                "constant_route": constant_route,
                "route_reason": route_reason,
                "nonsymmetric_traction": nonsymmetric_traction,
            }
        )
        self.operators = operators
        self.boundaries = boundaries_
        self.density = density_
        self.tolerance = tolerance_
        self.solve_method = solve_method
        self.closure_kind = closure_kind
        self.gauge_kind = gauge_kind
        self.compatibility_kind = compatibility_kind
        self.linear_policy = policy
        self.linear_problem = problem
        self.prepared_linear = prepared_linear
        self.nonsymmetric_traction = nonsymmetric_traction
        self.transform_plan = transform_plan
        self.hybrid_plan = hybrid_plan
        self.hybrid_action_defect = hybrid_action_defect
        self.hybrid_line_axis = line_axis
        self.maximum_resource_bytes = budget
        self.constant_route = constant_route
        self.route_reason = route_reason
        self.operator_id = operator_id
        self.pressure_problem_id = pressure_problem_id
        self.closure_id = closure_id
        self.plan_id = identifier

    @staticmethod
    def _prepare_transform(
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        tolerance: float,
        maximum_resource_bytes: int,
        /,
    ) -> FDLaplacianSolvePlan | None:
        if not operators.report.transform_eligible:
            return None
        grid = operators.discretization.grid
        resource_dtype = (
            np.result_type(operators.pressure_space.dtype, np.complex64)
            if any(axis.periodic for axis in grid.structured_axes)
            else operators.pressure_space.dtype
        )
        diagonal_resource_counts(
            operators.discretization.cell_shape,
            resource_dtype,
            maximum_resource_bytes,
            "MAC pressure transform",
        )
        boundary_kinds = {}
        for axis_index, (name, axis) in enumerate(
            zip(grid.axis_names, grid.structured_axes, strict=True)
        ):
            if axis.periodic:
                boundary_kinds[name] = ("periodic", "periodic")
            else:
                lower = (
                    "dirichlet"
                    if boundaries.side_kind(axis_index, "lower")
                    in ("pressure-outlet", "traction-open")
                    else "neumann"
                )
                upper = (
                    "dirichlet"
                    if boundaries.side_kind(axis_index, "upper")
                    in ("pressure-outlet", "traction-open")
                    else "neumann"
                )
                boundary_kinds[name] = (lower, upper)
        diagonalization = diagonalize_fd_laplacian(grid, boundary_kinds)
        probe = jnp.arange(
            int(np.prod(operators.discretization.cell_shape)),
            dtype=operators.pressure_space.dtype,
        ).reshape(operators.discretization.cell_shape)
        direct_action = -diagonalization.apply(probe)
        unit_face = tuple(
            jnp.ones(layout.shape, dtype=operators.pressure_space.dtype)
            for layout in operators.discretization.face_layouts
        )
        homogeneous_gradient = boundaries.pressure_gradient(probe, None, homogeneous=True)
        mac_action = -operators.divergence(
            tuple(
                coefficient * derivative
                for coefficient, derivative in zip(
                    unit_face, homogeneous_gradient, strict=True
                )
            )
        )
        defect = float(jnp.max(jnp.abs(direct_action - mac_action)))
        if defect > max(100.0 * tolerance, 5e-10):
            raise RuntimeError(
                "FD transform and MAC pressure operators failed exact-action identity."
            )
        return FDLaplacianSolvePlan(
            diagonalization,
            operator_scale=-1.0,
            compatibility=(
                "project_rhs" if boundaries.closure_kind == "neumann" else "error"
            ),
            gauge=(
                "zero_mean" if boundaries.closure_kind == "neumann" else "minimum_norm"
            ),
            zero_tolerance=tolerance,
        )

    @staticmethod
    def _prepare_hybrid(
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        line_axis: int,
        tolerance: float,
        policy: LinearSolvePolicy,
        maximum_resource_bytes: int,
        /,
        *,
        cell_coefficient: Array | None = None,
        face_coefficient: FaceVelocity | None = None,
    ) -> tuple[PreparedTransformLineSolve, Array]:
        shape = operators.discretization.cell_shape
        if len(shape) != 3:
            raise ValueError(
                "Hybrid MAC projection requires a three-dimensional tensor product."
            )
        if boundaries.closure_kind != "neumann":
            raise ValueError(
                "Hybrid MAC projection requires an all-Neumann pressure closure."
            )
        grid = operators.discretization.grid
        physical_line = grid.structured_axes[line_axis]
        if physical_line.periodic:
            raise ValueError(
                "Hybrid MAC projection requires an explicitly nonperiodic line axis."
            )
        dtype = operators.pressure_space.dtype
        if cell_coefficient is None:
            cell = jnp.ones(shape, dtype=dtype)
            face = tuple(
                jnp.ones(layout.shape, dtype=dtype)
                for layout in operators.discretization.face_layouts
            )
        else:
            cell = operators.validate_pressure(cell_coefficient)
            face = (
                operators.interpolate_inverse_momentum(cell)
                if face_coefficient is None
                else operators.validate_velocity(face_coefficient)
            )
            moved_cell = np.moveaxis(np.asarray(cell), line_axis, 0).reshape(
                (shape[line_axis], -1)
            )
            scale = max(1.0, float(np.max(np.abs(moved_cell[:, 0]))))
            if np.max(np.abs(moved_cell - moved_cell[:, :1])) > tolerance * scale:
                raise ValueError(
                    "Exact hybrid beta must vary only along the retained physical line."
                )
        transverse_data = []
        for axis_index, axis in enumerate(grid.structured_axes):
            if axis_index == line_axis:
                continue
            axis_data = pressure_cell_axis_transform(axis, dtype)
            if axis_data is None:
                raise ValueError(
                    "Hybrid MAC projection requires uniform transform-compatible "
                    "transverse axes."
                )
            transverse_data.append(axis_data)
        base_lower, _, base_upper = pressure_cell_line_coefficients(physical_line, dtype)
        line_values = jnp.moveaxis(cell, line_axis, 0).reshape((shape[line_axis], -1))[
            :, 0
        ]
        line_faces = jnp.moveaxis(face[line_axis], line_axis, 0).reshape(
            (shape[line_axis] + 1, -1)
        )[:, 0]
        lower = base_lower * line_faces[1:-1]
        upper = base_upper * line_faces[1:-1]
        diagonal = jnp.zeros((shape[line_axis],), dtype=dtype)
        diagonal = diagonal.at[1:].add(-lower)
        diagonal = diagonal.at[:-1].add(-upper)
        transverse_modal = modal_sum(
            tuple(item[1] for item in transverse_data), dtype=dtype
        )
        representation = TransformLineRepresentation(
            tuple(item[0] for item in transverse_data),
            line_axis,
            lower,
            diagonal,
            upper,
            transverse_modal,
            transverse_line_scale=line_values,
            certification_tolerance=tolerance,
            representation_id=canonical_fingerprint(
                {
                    "kind": "mac-pressure-transform-line",
                    "operators": operators.prepared_id,
                    "boundaries": boundaries.prepared_id,
                    "line_axis": line_axis,
                    "coefficient": canonical_fingerprint(
                        {"cell": np.asarray(cell).tolist()}
                    ),
                }
            ),
        )
        probe = jnp.sin(
            0.31 * jnp.arange(int(np.prod(shape)), dtype=dtype).reshape(shape) + 0.17
        )
        action_defect, action_certified = certify_separable_action(
            representation.apply(probe),
            -operators.weighted_laplacian(probe, face),
            tolerance,
        )
        if not action_certified:
            raise RuntimeError(
                "Hybrid MAC pressure action failed exact physical-action evidence."
            )
        nullspace = TransformLineNullspacePolicy(
            jnp.asarray(physical_line.interval_widths, dtype=dtype),
            zero_mode_index=0,
            pin_row=0,
        )
        prepared = TransformLineSolvePlan(
            representation,
            operator_scale=1.0,
            diagonal_shift=0.0,
            nullspace=nullspace,
            tolerance=tolerance,
            differentiation=policy.differentiation,
            maximum_resource_bytes=maximum_resource_bytes,
        ).prepare()
        return prepared, action_defect

    def _stage(self, stage: MACBoundaryStageData | None, /) -> MACBoundaryStageData:
        return (
            self.boundaries.evaluate(jnp.asarray(0.0), None)
            if stage is None
            else self.boundaries.validate_stage(stage)
        )

    def project(
        self,
        velocity: FaceVelocity,
        step_size: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
        inverse_momentum_diagonal: ArrayLike | None = None,
        boundary_stage: MACBoundaryStageData | None = None,
    ) -> MACPressureProjectionResult:
        stage = self._stage(boundary_stage)
        values = self.operators.validate_velocity(velocity)
        dtype = self.operators.pressure_space.dtype
        step = jnp.asarray(step_size, dtype=dtype).reshape(())
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Pressure projection step_size must be positive and finite.",
        )
        incoming_pressure = (
            jnp.zeros(self.operators.discretization.cell_shape, dtype=dtype)
            if pressure is None
            else self.operators.validate_pressure(pressure)
        )
        if self.closure_kind == "neumann":
            incoming_pressure = self.operators.gauge_project(incoming_pressure)
        inverse = (
            jnp.full(
                self.operators.discretization.cell_shape,
                step / self.density,
                dtype=dtype,
            )
            if inverse_momentum_diagonal is None
            else self.operators.validate_pressure(inverse_momentum_diagonal)
        )
        inverse = eqx.error_if(
            inverse,
            jnp.any(~jnp.isfinite(inverse) | (inverse <= 0.0)),
            "Inverse momentum diagonal must be positive and finite.",
        )
        face_inverse = self.operators.interpolate_inverse_momentum(inverse)
        divergence_before = self.operators.divergence(values)
        zero_pressure = jnp.zeros_like(divergence_before)
        boundary_gradient = self.boundaries.pressure_gradient(
            zero_pressure, stage, homogeneous=False
        )
        boundary_divergence = self.operators.divergence(
            tuple(
                coefficient * derivative
                for coefficient, derivative in zip(
                    face_inverse, boundary_gradient, strict=True
                )
            )
        )
        raw_rhs = -divergence_before + boundary_divergence
        rhs = (
            self.operators.compatibility_project(raw_rhs)
            if self.closure_kind == "neumann"
            else raw_rhs
        )
        route = self.constant_route
        route_reason = self.route_reason
        active_hybrid_plan = self.hybrid_plan
        active_hybrid_defect = self.hybrid_action_defect
        direct_scale = step / self.density
        if inverse_momentum_diagonal is not None:
            coefficient_host = np.asarray(inverse)
            coefficient_scale = max(1.0, float(np.max(np.abs(coefficient_host))))
            coefficient_constant = bool(
                np.max(coefficient_host) - np.min(coefficient_host)
                <= self.tolerance * coefficient_scale
            )
            coefficient_line = False
            if self.hybrid_line_axis is not None:
                moved = np.moveaxis(coefficient_host, self.hybrid_line_axis, 0).reshape(
                    (
                        self.operators.discretization.cell_shape[self.hybrid_line_axis],
                        -1,
                    )
                )
                coefficient_line = bool(
                    np.max(np.abs(moved - moved[:, :1]))
                    <= self.tolerance * coefficient_scale
                )
            if (
                coefficient_constant
                and self.transform_plan is not None
                and self.solve_method in ("auto", "direct", "transform")
            ):
                route = "transform"
                direct_scale = jnp.reshape(inverse, (-1,))[0]
                route_reason = (
                    "exact constant runtime coefficient retained the tensor transform"
                )
            elif (
                not self.nonsymmetric_traction
                and coefficient_line
                and self.hybrid_line_axis is not None
                and self.solve_method in ("auto", "direct", "hybrid")
            ):
                active_hybrid_plan, active_hybrid_defect = self._prepare_hybrid(
                    self.operators,
                    self.boundaries,
                    self.hybrid_line_axis,
                    self.tolerance,
                    self.linear_policy,
                    self.maximum_resource_bytes,
                    cell_coefficient=inverse,
                    face_coefficient=face_inverse,
                )
                route = "hybrid"
                direct_scale = jnp.asarray(1.0, dtype=dtype)
                route_reason = "exact positive beta along the retained line certified physical action"
            elif self.solve_method in ("direct", "transform", "hybrid"):
                raise ValueError(
                    "Explicit direct MAC pressure request is unsupported by the runtime "
                    "coefficient; no iterative fallback was taken."
                )
            else:
                route = "iterative"
                route_reason = (
                    "FGMRES selected for stabilized nonsymmetric traction"
                    if self.nonsymmetric_traction
                    else "PCG selected for general positive beta"
                )
        if route == "transform":
            transform = self.transform_plan.solve(rhs / direct_scale)
            solution_candidate = (
                self.operators.gauge_project(transform.value)
                if self.closure_kind == "neumann"
                else transform.value
            )
            solve_success = transform.converged
            linear = None
            hybrid = None
        elif route == "hybrid":
            hybrid = active_hybrid_plan.solve(rhs / direct_scale)
            solution_candidate = self.operators.gauge_project(hybrid.candidate)
            solve_success = hybrid.converged
            linear = None
            transform = None
        else:
            pressure_operator = FunctionLinearOperator(
                _WeightedMACPressureAction(self.operators, self.boundaries, face_inverse),
                source=self.operators.pressure_space,
                target=self.operators.pressure_space,
                properties=OperatorProperties(
                    self_adjoint=not self.nonsymmetric_traction,
                    positive_definite=not self.nonsymmetric_traction,
                    evidence=(
                        {}
                        if self.nonsymmetric_traction
                        else {
                            "self_adjoint": "construction",
                            "positive_definite": "construction",
                        }
                    ),
                ),
                operator_id=self.operator_id,
            )
            problem = LinearSystem(pressure_operator, problem_id=self.pressure_problem_id)
            prepared = refresh(self.prepared_linear, problem)
            linear = solve(prepared, rhs, initial_guess=incoming_pressure)
            solution_candidate = linear.value
            if self.closure_kind == "neumann":
                solution_candidate = self.operators.gauge_project(solution_candidate)
            solve_success = linear.successful
            transform = None
            hybrid = None
        if self.closure_kind == "neumann":
            increment_candidate = solution_candidate
            pressure_candidate = self.operators.gauge_project(
                incoming_pressure + increment_candidate
            )
        else:
            pressure_candidate = solution_candidate
            increment_candidate = pressure_candidate - incoming_pressure
        correction_gradient = self.boundaries.pressure_gradient(
            solution_candidate,
            stage,
            homogeneous=self.closure_kind == "neumann",
        )
        corrected_candidate = tuple(
            component - coefficient * gradient
            for component, coefficient, gradient in zip(
                values,
                face_inverse,
                correction_gradient,
                strict=True,
            )
        )
        action = _WeightedMACPressureAction(self.operators, self.boundaries, face_inverse)
        residual = action(solution_candidate) - rhs
        volumes = self.operators.discretization.cell_volumes.astype(dtype)
        residual_norm = jnp.sqrt(jnp.sum(volumes * residual**2))
        rhs_norm = jnp.sqrt(jnp.sum(volumes * rhs**2))
        gauge_defect = (
            jnp.abs(jnp.sum(volumes * pressure_candidate))
            if self.closure_kind == "neumann"
            else jnp.asarray(0.0, dtype=dtype)
        )
        divergence_candidate = self.operators.divergence(corrected_candidate)
        divergence_norm = jnp.sqrt(jnp.sum(volumes * divergence_candidate**2))
        integrated_mass_flux = jnp.sum(volumes * divergence_candidate)
        mass_defect = jnp.abs(integrated_mass_flux)
        finite = (
            stage.finite
            & jnp.all(jnp.isfinite(solution_candidate))
            & jnp.all(jnp.isfinite(divergence_candidate))
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(gauge_defect)
            & jnp.isfinite(mass_defect)
        )
        converged = (
            solve_success
            & stage.successful
            & finite
            & (residual_norm <= self.tolerance * jnp.maximum(rhs_norm, 1.0))
            & (divergence_norm <= self.tolerance * jnp.maximum(rhs_norm, 1.0))
            & (gauge_defect <= self.tolerance)
        )
        corrected = tuple(
            jnp.where(converged, candidate, original)
            for candidate, original in zip(corrected_candidate, values, strict=True)
        )
        pressure_value = jnp.where(converged, pressure_candidate, incoming_pressure)
        increment = jnp.where(
            converged, increment_candidate, jnp.zeros_like(increment_candidate)
        )
        divergence_after = self.operators.divergence(corrected)
        reported_mass_flux = jnp.sum(volumes * divergence_after)
        closure = MACPressureClosureReport(
            kind=self.closure_kind,
            gauge=self.gauge_kind,
            compatibility=self.compatibility_kind,
            integrated_mass_flux=reported_mass_flux,
            mass_defect=jnp.abs(reported_mass_flux),
            gauge_defect=gauge_defect,
            finite=finite,
            successful=converged,
            closure_id=self.closure_id,
        )
        return MACPressureProjectionResult(
            velocity=corrected,
            pressure=pressure_value,
            pressure_increment=increment,
            divergence_before=divergence_before,
            divergence_after=divergence_after,
            pressure_residual=residual,
            compatible_rhs=rhs,
            face_inverse_momentum=face_inverse,
            gauge_defect=gauge_defect,
            closure=closure,
            solve_method=route,
            route_reason=route_reason,
            linear=linear,
            transform=transform,
            hybrid_action_defect=active_hybrid_defect,
            hybrid=hybrid,
            hybrid_line_axis=self.hybrid_line_axis,
            maximum_resource_bytes=self.maximum_resource_bytes,
            finite=finite,
            converged=converged,
            projection_id=self.plan_id,
        )

    def project_rate(
        self,
        rate: FaceVelocity,
        /,
        *,
        boundary_stage: MACBoundaryStageData | None = None,
    ) -> MACRateProjectionResult:
        """Project one face-velocity rate without changing essential normal faces."""
        projected = self.project(rate, 1.0, boundary_stage=boundary_stage)
        return MACRateProjectionResult(
            rate=projected.velocity,
            pressure=projected.pressure_increment,
            divergence_before=projected.divergence_before,
            divergence_after=projected.divergence_after,
            pressure_residual=projected.pressure_residual,
            compatible_rhs=projected.compatible_rhs,
            face_inverse_density=projected.face_inverse_momentum,
            gauge_defect=projected.gauge_defect,
            closure=projected.closure,
            solve_method=projected.solve_method,
            route_reason=projected.route_reason,
            linear=projected.linear,
            transform=projected.transform,
            hybrid_action_defect=projected.hybrid_action_defect,
            hybrid=projected.hybrid,
            hybrid_line_axis=projected.hybrid_line_axis,
            maximum_resource_bytes=projected.maximum_resource_bytes,
            finite=projected.finite,
            converged=projected.converged,
            projection_id=self.plan_id,
        )


__all__ = [
    "MACPressureClosureReport",
    "MACPressureCompatibilityKind",
    "MACPressureGaugeKind",
    "MACRateProjectionResult",
    "MACPressureProjectionPlan",
    "MACPressureProjectionResult",
    "MACPressureSolveMethod",
]
