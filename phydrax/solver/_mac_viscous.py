#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_boundary import MACBoundaryStageData
from ..discretization.finite_volume._mac_momentum import PreparedMACMomentumOperators
from ..equations._mac_incompressible import CompiledMACIncompressibleDynamics
from ..equations._mac_les import MACLESStageResult, PreparedMACAlgebraicLES
from ..linalg import (
    ArraySpace,
    ConjugateGradient,
    DifferentiationPolicy,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    refresh,
    solve,
    TensorLinearTransform,
    TolerancePolicy,
)
from ..linalg._linear_transform import AbstractLinearTransform
from ..linalg._transform_line import (
    PreparedTransformLineSolve,
    TransformLineFactors,
    TransformLineRepresentation,
    TransformLineSolvePlan,
    TransformLineSolveResult,
)
from ._mac_composite_projection import (
    CompositeMACProjectionPlan,
    CompositeMACProjectionResult,
)
from ._mac_separable import (
    certify_separable_action,
    diagonal_resource_counts,
    iterative_workspace_bytes,
    modal_sum,
    normal_velocity_is_essential,
    velocity_face_axis_transform,
    velocity_face_line_coefficients,
)
from ._structured_incompressible import MACPressureProjectionResult
from ._temporal_method import TemporalMethodCapabilities


if TYPE_CHECKING:
    from ._mac_stage_inverse_general import MACOperatorStageSolveResult


MACHelmholtzSolveMethod: TypeAlias = Literal["auto", "transform", "hybrid", "iterative"]
MAC_VISCOUS_SUCCESS = 0
MAC_VISCOUS_HELMHOLTZ_FAILURE = -1
MAC_VISCOUS_PROJECTION_FAILURE = -2
MAC_VISCOUS_BOUNDARY_FAILURE = -3
MAC_VISCOUS_HISTORY_INVALID = -4
MAC_VISCOUS_CLOSURE_FAILURE = -5


def _zeros(momentum: PreparedMACMomentumOperators, /) -> FaceVelocity:
    dtype = momentum.operators.pressure_space.dtype
    return tuple(
        jnp.zeros(layout.shape, dtype=dtype)
        for layout in momentum.operators.discretization.face_layouts
    )


def _unknown_shape(
    momentum: PreparedMACMomentumOperators, component: int, /
) -> tuple[int, ...]:
    shape = list(momentum.operators.discretization.face_layouts[component].shape)
    if not momentum.operators.discretization.grid.structured_axes[component].periodic:
        shape[component] -= 2
    return tuple(shape)


def _extract_unknown(
    momentum: PreparedMACMomentumOperators, component: int, value: Array, /
) -> Array:
    if momentum.operators.discretization.grid.structured_axes[component].periodic:
        return value
    location = [slice(None)] * value.ndim
    location[component] = slice(1, -1)
    return value[tuple(location)]


def _inject_unknown(
    momentum: PreparedMACMomentumOperators, component: int, value: Array, /
) -> Array:
    shape = momentum.operators.discretization.face_layouts[component].shape
    if momentum.operators.discretization.grid.structured_axes[component].periodic:
        if value.shape != shape:
            raise ValueError("Periodic MAC component unknown has the wrong shape.")
        return value
    expected = _unknown_shape(momentum, component)
    if value.shape != expected:
        raise ValueError(f"MAC component unknown must have shape {expected}.")
    location = [slice(None)] * value.ndim
    location[component] = slice(1, -1)
    return jnp.zeros(shape, dtype=value.dtype).at[tuple(location)].set(value)


def _single_component(
    momentum: PreparedMACMomentumOperators, component: int, value: Array, /
) -> FaceVelocity:
    result = list(_zeros(momentum))
    result[component] = value
    return tuple(result)


class _HomogeneousHelmholtzAction(StrictModule, NonTrainableState):
    momentum: PreparedMACMomentumOperators
    component: int = eqx.field(static=True)
    mass: Array
    diffusion: Array

    def __call__(self, value: Array, /) -> Array:
        value_ = self.momentum.operators.velocity_space.spaces[self.component].validate(
            value
        )
        laplacian = self.momentum.homogeneous_laplacian(
            _single_component(self.momentum, self.component, value_)
        )[self.component]
        return self.mass * value_ - self.diffusion * laplacian


class _PositiveUnknownLaplacian(StrictModule, NonTrainableState):
    momentum: PreparedMACMomentumOperators
    component: int = eqx.field(static=True)

    def __call__(self, value: Array, /) -> Array:
        full = _inject_unknown(self.momentum, self.component, value)
        laplacian = self.momentum.homogeneous_laplacian(
            _single_component(self.momentum, self.component, full)
        )[self.component]
        return -_extract_unknown(self.momentum, self.component, laplacian)


class MACHelmholtzResourceEstimate(StrictModule, NonTrainableState):
    component: int = eqx.field(static=True)
    route: str = eqx.field(static=True)
    factor_count: int = eqx.field(static=True)
    factor_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    total_bytes: int = eqx.field(static=True)


class _PreparedComponent(StrictModule, NonTrainableState):
    component: int = eqx.field(static=True)
    space: ArraySpace
    policy: LinearSolvePolicy
    problem: LinearSystem
    prepared: PreparedLinearSolve
    operator_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    transform: TensorLinearTransform | None
    modal_values: Array | None
    hybrid: TransformLineRepresentation | None
    prepared_hybrid: PreparedTransformLineSolve | None
    route: str = eqx.field(static=True)
    action_defect: Array
    trace_defect: Array
    resources: MACHelmholtzResourceEstimate
    component_id: str = eqx.field(static=True)


class MACHelmholtzResult(StrictModule):
    """Component Helmholtz solution without a pressure gauge."""

    value: FaceVelocity
    candidate: FaceVelocity
    residual: FaceVelocity
    effective_rhs: FaceVelocity
    affine_laplacian: FaceVelocity
    component_residual_norms: Array
    residual_norm: Array
    relative_residual: Array
    action_defect: Array
    trace_defect: Array
    boundary_defect: Array
    linear: tuple[LinearSolveResult | None, ...]
    hybrid: tuple[TransformLineSolveResult | None, ...]
    factors: tuple[TransformLineFactors | None, ...]
    resources: tuple[MACHelmholtzResourceEstimate, ...]
    component_converged: Array
    finite: Array
    converged: Array
    routes: tuple[str, ...] = eqx.field(static=True)
    differentiation_policy: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.converged


class MACHelmholtzSolvePlan(StrictModule, NonTrainableState):
    """Prepared mass-minus-diffusion solve with affine boundary separation."""

    momentum: PreparedMACMomentumOperators
    components: tuple[_PreparedComponent, ...]
    solve_method: MACHelmholtzSolveMethod = eqx.field(static=True)
    hybrid_line_axis: int | None = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    fixed_mass_coefficient: float | None = eqx.field(static=True)
    fixed_diffusion_coefficient: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        momentum: PreparedMACMomentumOperators,
        /,
        *,
        solve_method: MACHelmholtzSolveMethod = "auto",
        hybrid_line_axis: int | None = None,
        tolerance: float = 1e-9,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
        fixed_mass_coefficient: float | None = None,
        fixed_diffusion_coefficient: float | None = None,
        maximum_resource_bytes: int = 512 * 1024**2,
    ):
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        if solve_method not in ("auto", "transform", "hybrid", "iterative"):
            raise ValueError("Unknown MAC Helmholtz solve_method.")
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        budget = int(maximum_resource_bytes)
        if (
            tolerance_ <= 0.0
            or not np.isfinite(tolerance_)
            or iterations <= 0
            or budget <= 0
        ):
            raise ValueError(
                "MAC Helmholtz tolerance, iterations, or resources are invalid."
            )
        if (fixed_mass_coefficient is None) != (fixed_diffusion_coefficient is None):
            raise ValueError(
                "Fixed mass and diffusion coefficients must be supplied together."
            )
        fixed: tuple[float, float] | None
        if fixed_mass_coefficient is None:
            fixed = None
        else:
            mass = float(fixed_mass_coefficient)
            diffusion = float(fixed_diffusion_coefficient)
            if mass <= 0.0 or diffusion < 0.0 or not np.isfinite(mass + diffusion):
                raise ValueError("Fixed Helmholtz coefficients are invalid.")
            fixed = (mass, diffusion)
        if hybrid_line_axis is not None:
            hybrid_line_axis = int(hybrid_line_axis)
            if hybrid_line_axis < 0 or hybrid_line_axis >= momentum.dimension:
                raise ValueError("hybrid_line_axis is outside the MAC tensor rank.")
        components = tuple(
            self._prepare_component(
                momentum,
                component,
                solve_method,
                hybrid_line_axis,
                tolerance_,
                iterations,
                linear_policy,
                fixed,
                budget,
            )
            for component in range(momentum.dimension)
        )
        if len({value.policy.differentiation.mode for value in components}) != 1:
            raise ValueError(
                "MAC Helmholtz components require one differentiation policy."
            )
        self.momentum = momentum
        self.components = components
        self.solve_method = solve_method
        self.hybrid_line_axis = hybrid_line_axis
        self.tolerance = tolerance_
        self.fixed_mass_coefficient = None if fixed is None else fixed[0]
        self.fixed_diffusion_coefficient = None if fixed is None else fixed[1]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-helmholtz-solve-plan",
                "momentum": momentum.prepared_id,
                "components": [value.component_id for value in components],
                "method": solve_method,
                "hybrid_line_axis": hybrid_line_axis,
                "fixed_coefficients": fixed,
                "tolerance": tolerance_,
                "differentiation": components[0].policy.differentiation.mode,
            }
        )

    @staticmethod
    def _prepare_component(
        momentum: PreparedMACMomentumOperators,
        component: int,
        method: MACHelmholtzSolveMethod,
        hybrid_line_axis: int | None,
        tolerance: float,
        iterations: int,
        supplied_policy: LinearSolvePolicy | None,
        fixed: tuple[float, float] | None,
        budget: int,
        /,
    ) -> _PreparedComponent:
        space = momentum.operators.velocity_space.spaces[component]
        if not isinstance(space, ArraySpace):
            raise TypeError("MAC component spaces must be ArraySpace values.")
        operator_id = canonical_fingerprint(
            {
                "kind": "mac-homogeneous-helmholtz",
                "momentum": momentum.prepared_id,
                "component": component,
            }
        )
        operator = FunctionLinearOperator(
            _HomogeneousHelmholtzAction(
                momentum,
                component,
                jnp.asarray(1.0, dtype=space.dtype),
                jnp.asarray(1.0, dtype=space.dtype),
            ),
            source=space,
            target=space,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=operator_id,
        )
        problem_id = canonical_fingerprint(
            {"kind": "mac-helmholtz-system", "operator": operator_id}
        )
        problem = LinearSystem(operator, problem_id=problem_id)
        policy = supplied_policy or LinearSolvePolicy(
            ConjugateGradient(),
            tolerance=TolerancePolicy(
                relative=tolerance, absolute=tolerance, max_steps=iterations
            ),
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        prepared = prepare(problem, policy)

        transform = None
        modal_values = None
        action_defect = jnp.asarray(jnp.inf, dtype=space.dtype)
        trace_defect = jnp.asarray(jnp.inf, dtype=space.dtype)
        direct_eligible = method in (
            "auto",
            "transform",
        ) and normal_velocity_is_essential(momentum, component)
        data: list[tuple[AbstractLinearTransform, Array, float]] = []
        direct_resource_data = None
        if direct_eligible:
            for derivative_axis in range(momentum.dimension):
                axis_data = velocity_face_axis_transform(
                    momentum, component, derivative_axis
                )
                if axis_data is None:
                    direct_eligible = False
                    break
                data.append(axis_data)
        if direct_eligible:
            shape = _unknown_shape(momentum, component)
            direct_dtype = jnp.result_type(*[item[0].modal_space.dtype for item in data])
            direct_resource_data = diagonal_resource_counts(
                shape, direct_dtype, budget, "MAC transform"
            )
            transform = TensorLinearTransform(tuple(item[0] for item in data))
            modal_values = modal_sum(tuple(item[1] for item in data))
            probe = jnp.sin(
                0.37 * jnp.arange(int(np.prod(shape)), dtype=space.dtype).reshape(shape)
                + component
            )
            exact = _PositiveUnknownLaplacian(momentum, component)(probe)
            represented = jnp.real(
                transform.synthesize(modal_values * transform.analyze(probe))
            ).astype(exact.dtype)
            action_defect, action_certified = certify_separable_action(
                represented, exact, tolerance
            )
            expected_trace = sum(
                int(np.prod(shape)) // shape[axis] * item[2]
                for axis, item in enumerate(data)
            )
            trace_defect = jnp.abs(jnp.sum(modal_values) - expected_trace)
            epsilon = jnp.finfo(space.dtype).eps
            direct_eligible = action_certified and bool(
                np.asarray(
                    jnp.isfinite(trace_defect)
                    & (
                        trace_defect
                        <= jnp.maximum(
                            100.0 * tolerance,
                            4096.0 * epsilon * jnp.maximum(1.0, expected_trace),
                        )
                    )
                )
            )
            if not direct_eligible:
                transform = None
                modal_values = None

        hybrid = None
        prepared_hybrid = None
        hybrid_eligible = (
            method in ("auto", "hybrid")
            and (method == "hybrid" or not direct_eligible)
            and hybrid_line_axis is not None
            and normal_velocity_is_essential(momentum, component)
        )
        transverse_data: list[tuple[AbstractLinearTransform, Array, float]] = []
        if hybrid_eligible:
            for derivative_axis in range(momentum.dimension):
                if derivative_axis == hybrid_line_axis:
                    continue
                axis_data = velocity_face_axis_transform(
                    momentum, component, derivative_axis
                )
                if axis_data is None:
                    hybrid_eligible = False
                    break
                transverse_data.append(axis_data)
        if hybrid_eligible:
            grid_axis = momentum.operators.discretization.grid.structured_axes[
                hybrid_line_axis
            ]
            count = int(grid_axis.interval_widths.size)
            if hybrid_line_axis == component and not grid_axis.periodic:
                count -= 1
            hybrid_eligible = count >= 1 and (not grid_axis.periodic or count >= 3)
        if hybrid_eligible:
            lower, diagonal, upper, corners = velocity_face_line_coefficients(
                momentum, component, hybrid_line_axis
            )
            transverse_modal = (
                modal_sum(tuple(item[1] for item in transverse_data))
                if transverse_data
                else jnp.asarray(0.0, dtype=space.dtype)
            )
            hybrid = TransformLineRepresentation(
                tuple(item[0] for item in transverse_data),
                hybrid_line_axis,
                lower,
                diagonal,
                upper,
                transverse_modal,
                periodic_corners=corners,
                certification_tolerance=tolerance,
                representation_id=canonical_fingerprint(
                    {
                        "kind": "mac-transform-line",
                        "momentum": momentum.prepared_id,
                        "component": component,
                        "line_axis": hybrid_line_axis,
                    }
                ),
            )
            shape = _unknown_shape(momentum, component)
            probe = jnp.cos(
                0.29 * jnp.arange(int(np.prod(shape)), dtype=space.dtype).reshape(shape)
                + component
            )
            exact = _PositiveUnknownLaplacian(momentum, component)(probe)
            action_defect, hybrid_eligible = certify_separable_action(
                hybrid.apply(probe), exact, tolerance
            )
            trace_defect = hybrid.report.trace_defect
            if not hybrid_eligible:
                hybrid = None

        if method == "transform":
            if not direct_eligible:
                raise ValueError(
                    "Transform Helmholtz requires a certified uniform action."
                )
            route = "transform"
        elif method == "hybrid":
            if not hybrid_eligible or fixed is None:
                raise ValueError(
                    "Hybrid Helmholtz requires a certified line action and fixed coefficients."
                )
            route = "hybrid"
        elif method == "iterative":
            route = "iterative"
        elif direct_eligible:
            route = "transform"
        elif hybrid_eligible and fixed is not None:
            route = "hybrid"
        else:
            route = "iterative"

        if route == "hybrid":
            prepared_hybrid = TransformLineSolvePlan(
                hybrid,
                diagonal_shift=fixed[0],
                operator_scale=fixed[1],
                tolerance=tolerance,
                differentiation=policy.differentiation,
                maximum_resource_bytes=budget,
            ).prepare()
            estimate = prepared_hybrid.resources
            resources = MACHelmholtzResourceEstimate(
                component,
                route,
                estimate.factor_count,
                estimate.factor_bytes,
                estimate.workspace_bytes,
                estimate.total_bytes,
            )
        elif route == "transform":
            count, factor_bytes, workspace_bytes, total_bytes = direct_resource_data
            resources = MACHelmholtzResourceEstimate(
                component,
                route,
                count,
                factor_bytes,
                workspace_bytes,
                total_bytes,
            )
        else:
            workspace_bytes = iterative_workspace_bytes(
                space.shape, space.dtype, budget, "MAC iterative"
            )
            resources = MACHelmholtzResourceEstimate(
                component, route, 0, 0, workspace_bytes, workspace_bytes
            )
        component_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-helmholtz-component",
                "momentum": momentum.prepared_id,
                "component": component,
                "route": route,
                "linear_plan": prepared.plan.plan_id,
                "transform": None if transform is None else transform.transform_id,
                "hybrid": None if hybrid is None else hybrid.representation_id,
            }
        )
        return _PreparedComponent(
            component,
            space,
            policy,
            problem,
            prepared,
            operator_id,
            problem_id,
            transform,
            modal_values,
            hybrid,
            prepared_hybrid,
            route,
            action_defect,
            trace_defect,
            resources,
            component_id,
        )

    @property
    def differentiation_policy(self) -> str:
        return self.components[0].policy.differentiation.mode

    def _coefficients(
        self,
        mass_coefficient: ArrayLike | None,
        diffusion_coefficient: ArrayLike | None,
        /,
    ) -> tuple[Array, Array]:
        dtype = self.momentum.operators.pressure_space.dtype
        if self.fixed_mass_coefficient is None:
            if mass_coefficient is None or diffusion_coefficient is None:
                raise ValueError("Dynamic Helmholtz plans require both coefficients.")
            mass = jnp.asarray(mass_coefficient, dtype=dtype).reshape(())
            diffusion = jnp.asarray(diffusion_coefficient, dtype=dtype).reshape(())
        else:
            mass = jnp.asarray(self.fixed_mass_coefficient, dtype=dtype)
            diffusion = jnp.asarray(self.fixed_diffusion_coefficient, dtype=dtype)
            if mass_coefficient is not None:
                supplied = jnp.asarray(mass_coefficient, dtype=dtype).reshape(())
                mass = eqx.error_if(
                    mass, supplied != mass, "Fixed Helmholtz mass cannot change."
                )
            if diffusion_coefficient is not None:
                supplied = jnp.asarray(diffusion_coefficient, dtype=dtype).reshape(())
                diffusion = eqx.error_if(
                    diffusion,
                    supplied != diffusion,
                    "Fixed Helmholtz diffusion cannot change.",
                )
        mass = eqx.error_if(
            mass,
            ~jnp.isfinite(mass) | (mass <= 0.0),
            "Helmholtz mass must be positive and finite.",
        )
        diffusion = eqx.error_if(
            diffusion,
            ~jnp.isfinite(diffusion) | (diffusion < 0.0),
            "Helmholtz diffusion must be nonnegative and finite.",
        )
        return mass, diffusion

    def solve(
        self,
        right_hand_side: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        /,
        *,
        mass_coefficient: ArrayLike | None = None,
        diffusion_coefficient: ArrayLike | None = None,
        initial_guess: FaceVelocity | None = None,
    ) -> MACHelmholtzResult:
        rhs = self.momentum.operators.validate_velocity(right_hand_side)
        stage = self.momentum.boundaries.validate_stage(boundary_stage)
        mass, diffusion = self._coefficients(mass_coefficient, diffusion_coefficient)
        zero = _zeros(self.momentum)
        affine = tuple(
            direct - homogeneous
            for direct, homogeneous in zip(
                self.momentum.laplacian(zero, stage),
                self.momentum.homogeneous_laplacian(zero),
                strict=True,
            )
        )
        target = self.momentum.boundaries.enforce(zero, stage)
        effective_rhs = tuple(
            value + diffusion * forcing + mass * boundary
            for value, forcing, boundary in zip(
                self.momentum.boundaries.homogeneous_rate(rhs),
                affine,
                target,
                strict=True,
            )
        )
        fallback = (
            target
            if initial_guess is None
            else self.momentum.boundaries.enforce(
                self.momentum.operators.validate_velocity(initial_guess), stage
            )
        )
        candidates: list[Array] = []
        linears: list[LinearSolveResult | None] = []
        hybrids: list[TransformLineSolveResult | None] = []
        factors: list[TransformLineFactors | None] = []
        route_success: list[Array] = []
        for prepared in self.components:
            component = prepared.component
            component_rhs = effective_rhs[component]
            if prepared.route == "transform":
                unknown_rhs = _extract_unknown(self.momentum, component, component_rhs)
                modal_values = prepared.modal_values
                mass_ = mass
                diffusion_ = diffusion
                if self.differentiation_policy in ("rhs-only", "none"):
                    modal_values = jax.lax.stop_gradient(modal_values)
                    mass_ = jax.lax.stop_gradient(mass_)
                    diffusion_ = jax.lax.stop_gradient(diffusion_)
                denominator = mass_ + diffusion_ * modal_values
                unknown = jnp.real(
                    prepared.transform.synthesize(
                        prepared.transform.analyze(unknown_rhs) / denominator
                    )
                ).astype(component_rhs.dtype)
                candidates.append(
                    _inject_unknown(self.momentum, component, unknown) + target[component]
                )
                linears.append(None)
                hybrids.append(None)
                factors.append(None)
                route_success.append(
                    jnp.all(jnp.isfinite(denominator)) & jnp.all(denominator > 0.0)
                )
            elif prepared.route == "hybrid":
                result = prepared.prepared_hybrid.solve(
                    _extract_unknown(self.momentum, component, component_rhs)
                )
                candidates.append(
                    _inject_unknown(self.momentum, component, result.candidate)
                    + target[component]
                )
                linears.append(None)
                hybrids.append(result)
                factors.append(prepared.prepared_hybrid.factors)
                route_success.append(result.converged)
            else:
                operator = FunctionLinearOperator(
                    _HomogeneousHelmholtzAction(
                        self.momentum, component, mass, diffusion
                    ),
                    source=prepared.space,
                    target=prepared.space,
                    properties=OperatorProperties(
                        self_adjoint=True,
                        positive_definite=True,
                        evidence={
                            "self_adjoint": "construction",
                            "positive_definite": "construction",
                        },
                    ),
                    operator_id=prepared.operator_id,
                )
                result = solve(
                    refresh(
                        prepared.prepared,
                        LinearSystem(operator, problem_id=prepared.problem_id),
                    ),
                    component_rhs,
                    initial_guess=fallback[component],
                )
                candidates.append(result.value)
                linears.append(result)
                hybrids.append(None)
                factors.append(None)
                route_success.append(result.successful)
        candidate = self.momentum.boundaries.enforce(tuple(candidates), stage)
        residual = tuple(
            mass * value - diffusion * laplacian - effective
            for value, laplacian, effective in zip(
                candidate,
                self.momentum.homogeneous_laplacian(candidate),
                effective_rhs,
                strict=True,
            )
        )
        component_norms = jnp.stack(
            tuple(
                jnp.sqrt(jnp.real(space.inner(value, value)))
                for space, value in zip(
                    self.momentum.operators.velocity_space.spaces, residual, strict=True
                )
            )
        )
        rhs_norms = jnp.stack(
            tuple(
                jnp.sqrt(jnp.real(space.inner(value, value)))
                for space, value in zip(
                    self.momentum.operators.velocity_space.spaces,
                    effective_rhs,
                    strict=True,
                )
            )
        )
        residual_norm = jnp.linalg.norm(component_norms)
        relative_residual = residual_norm / jnp.maximum(1.0, jnp.linalg.norm(rhs_norms))
        boundary_defect = self.momentum.boundaries.defect(candidate, stage)
        component_converged = (
            jnp.stack(tuple(route_success))
            & jnp.isfinite(component_norms)
            & (component_norms <= self.tolerance * jnp.maximum(1.0, rhs_norms))
        )
        finite = (
            stage.successful
            & jnp.all(
                jnp.stack(
                    tuple(jnp.all(jnp.isfinite(value)) for value in candidate + residual)
                )
            )
            & jnp.isfinite(relative_residual)
        )
        converged = (
            finite
            & jnp.all(component_converged)
            & (relative_residual <= self.tolerance)
            & (boundary_defect <= self.tolerance)
        )
        value = tuple(
            jnp.where(converged, accepted, previous)
            for accepted, previous in zip(candidate, fallback, strict=True)
        )
        if self.differentiation_policy == "none":
            value = tuple(jax.lax.stop_gradient(item) for item in value)
        return MACHelmholtzResult(
            value=value,
            candidate=candidate,
            residual=residual,
            effective_rhs=effective_rhs,
            affine_laplacian=affine,
            component_residual_norms=component_norms,
            residual_norm=residual_norm,
            relative_residual=relative_residual,
            action_defect=jnp.max(
                jnp.stack(tuple(item.action_defect for item in self.components))
            ),
            trace_defect=jnp.max(
                jnp.stack(tuple(item.trace_defect for item in self.components))
            ),
            boundary_defect=boundary_defect,
            linear=tuple(linears),
            hybrid=tuple(hybrids),
            factors=tuple(factors),
            resources=tuple(item.resources for item in self.components),
            component_converged=component_converged,
            finite=finite,
            converged=converged,
            routes=tuple(item.route for item in self.components),
            differentiation_policy=self.differentiation_policy,
            plan_id=self.plan_id,
        )


def _explicit_rate(
    dynamics: CompiledMACIncompressibleDynamics,
    time: ArrayLike,
    state: ArrayLike,
    args: Any,
    /,
    *,
    boundary_stage: MACBoundaryStageData | None = None,
) -> FaceVelocity:
    """Evaluate only the declared explicit partition; SGS is never inferred here."""
    stage = (
        dynamics.boundary_stage(time, args)
        if boundary_stage is None
        else dynamics.momentum.boundaries.validate_stage(boundary_stage)
    )
    velocity = dynamics.momentum.boundaries.enforce(
        dynamics.unpack_velocity(state), stage
    )
    convection = dynamics.momentum.convection(velocity, stage=stage)
    forcing = dynamics._forcing(jnp.asarray(time), velocity, args)
    return tuple(
        -advective + source for advective, source in zip(convection, forcing, strict=True)
    )


def _prepared_algebraic_les(
    dynamics: CompiledMACIncompressibleDynamics, /
) -> PreparedMACAlgebraicLES | None:
    prepared = dynamics.algebraic_les
    if prepared is None:
        return None
    if not isinstance(prepared, PreparedMACAlgebraicLES):
        raise TypeError(
            "Frozen MAC implicit methods require stateless PreparedMACAlgebraicLES; "
            "dynamic and continuation-bearing closures are unsupported."
        )
    if prepared.momentum.prepared_id != dynamics.momentum.prepared_id:
        raise ValueError("Prepared MAC LES and compiled momentum IDs differ.")
    if prepared.viscosity_action.momentum.prepared_id != dynamics.momentum.prepared_id:
        raise ValueError(
            "Prepared MAC LES viscosity action and compiled momentum IDs differ."
        )
    return prepared


def _evaluate_frozen_les(
    prepared: PreparedMACAlgebraicLES,
    velocity: FaceVelocity,
    boundary_stage: MACBoundaryStageData,
    /,
) -> tuple[MACLESStageResult, Array]:
    stage = prepared.evaluate(velocity, boundary_stage)
    if stage.prepared_id != prepared.prepared_id:
        raise ValueError("MAC LES stage belongs to a different preparation.")
    if stage.viscosity_result.action_id != prepared.viscosity_action.action_id:
        raise ValueError("MAC LES stage viscosity action ID does not match.")
    if stage.boundary_stage_id != boundary_stage.stage_id:
        raise ValueError("MAC LES stage boundary ID does not match.")
    viscosity = jnp.where(
        stage.successful,
        stage.model_result.kinematic_viscosity,
        jnp.zeros_like(stage.model_result.kinematic_viscosity),
    )
    return stage, viscosity


def _variable_viscosity_policy(
    tolerance: float,
    maximum_iterations: int,
    supplied: LinearSolvePolicy | None,
    /,
) -> LinearSolvePolicy:
    tolerance_ = float(tolerance)
    iterations = int(maximum_iterations)
    if tolerance_ <= 0.0 or not np.isfinite(tolerance_) or iterations <= 0:
        raise ValueError(
            "Frozen MAC LES tolerance and maximum_iterations must be positive."
        )
    if supplied is not None:
        if not isinstance(supplied, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        return supplied
    return LinearSolvePolicy(
        GMRES(restart=min(40, iterations)),
        tolerance=TolerancePolicy(
            relative=tolerance_,
            absolute=tolerance_,
            max_steps=iterations,
        ),
        differentiation=DifferentiationPolicy("mathematical"),
    )


def _constraint_operators(
    dynamics: CompiledMACIncompressibleDynamics, /
) -> tuple[FunctionLinearOperator, FunctionLinearOperator]:
    operators = dynamics.momentum.operators
    boundaries = dynamics.momentum.boundaries

    def divergence(velocity):
        return operators.divergence(velocity)

    def negative_gradient(pressure):
        return tuple(
            -value
            for value in boundaries.pressure_gradient(pressure, None, homogeneous=True)
        )

    def gradient(pressure):
        return boundaries.pressure_gradient(pressure, None, homogeneous=True)

    def negative_divergence(velocity):
        return -operators.divergence(velocity)

    divergence_operator = FunctionLinearOperator(
        divergence,
        source=operators.velocity_space,
        target=operators.pressure_space,
        transpose_action=negative_gradient,
        operator_id=canonical_fingerprint(
            {
                "kind": "mac-frozen-stage-divergence",
                "operators": operators.prepared_id,
                "boundaries": boundaries.prepared_id,
            }
        ),
    )
    gradient_operator = FunctionLinearOperator(
        gradient,
        source=operators.pressure_space,
        target=operators.velocity_space,
        transpose_action=negative_divergence,
        operator_id=canonical_fingerprint(
            {
                "kind": "mac-frozen-stage-gradient",
                "operators": operators.prepared_id,
                "boundaries": boundaries.prepared_id,
            }
        ),
    )
    return divergence_operator, gradient_operator


def _incoming_pressure(
    dynamics: CompiledMACIncompressibleDynamics,
    pressure: ArrayLike | None,
    dtype: jnp.dtype,
    /,
) -> Array:
    return (
        jnp.zeros(
            dynamics.momentum.operators.discretization.cell_shape,
            dtype=dtype,
        )
        if pressure is None
        else dynamics.momentum.operators.gauge_project(pressure)
    )


def _frozen_status(
    history_valid: Array,
    boundary_successful: Array,
    coefficient_successful: Array,
    predictor_successful: Array,
    projection_successful: Array,
    /,
) -> Array:
    return jnp.where(
        ~history_valid,
        MAC_VISCOUS_HISTORY_INVALID,
        jnp.where(
            ~boundary_successful,
            MAC_VISCOUS_BOUNDARY_FAILURE,
            jnp.where(
                ~coefficient_successful,
                MAC_VISCOUS_CLOSURE_FAILURE,
                jnp.where(
                    ~predictor_successful,
                    MAC_VISCOUS_HELMHOLTZ_FAILURE,
                    jnp.where(
                        ~projection_successful,
                        MAC_VISCOUS_PROJECTION_FAILURE,
                        MAC_VISCOUS_SUCCESS,
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)


class MACIMEXEulerResult(StrictModule):
    time: Array
    attempted_time: Array
    step_size: Array
    pressure_correction_coefficient: Array
    previous_state: Array
    state: Array
    velocity: FaceVelocity
    pressure: Array
    explicit_rate: FaceVelocity
    helmholtz: MACHelmholtzResult | None
    projection: MACPressureProjectionResult | CompositeMACProjectionResult
    les_stage: MACLESStageResult | None
    frozen_kinematic_viscosity: Array | None
    coefficient_state: Array | None
    coefficient_time: Array | None
    stage_inverse: MACOperatorStageSolveResult | None
    finite: Array
    accepted: Array
    status: Array
    stage_plan_id: str | None = eqx.field(static=True)
    predictor_inverse_id: str | None = eqx.field(static=True)
    projection_inverse_id: str | None = eqx.field(static=True)
    temporal_profile: str = eqx.field(static=True)
    coefficient_refresh: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class MACIMEXEulerMethod(StrictModule, NonTrainableState):
    """IMEX Euler with a distinct accepted-state frozen algebraic-LES profile."""

    dynamics: CompiledMACIncompressibleDynamics
    helmholtz: MACHelmholtzSolvePlan | None
    variable_linear_policy: LinearSolvePolicy | None
    pressure_linear_policy: LinearSolvePolicy | None
    face_density: FaceVelocity | None
    divergence_operator: FunctionLinearOperator | None
    gradient_operator: FunctionLinearOperator | None
    fixed_step_size: float | None = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    implicit_les: bool = eqx.field(static=True)
    temporal_profile: str = eqx.field(static=True)
    coefficient_refresh: str = eqx.field(static=True)
    capabilities: TemporalMethodCapabilities
    prepared_les_id: str | None = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        /,
        *,
        fixed_step_size: float | None = None,
        solve_method: MACHelmholtzSolveMethod = "auto",
        hybrid_line_axis: int | None = None,
        tolerance: float = 1e-9,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
        maximum_resource_bytes: int = 512 * 1024**2,
    ):
        if not isinstance(dynamics, CompiledMACIncompressibleDynamics):
            raise TypeError("dynamics must be CompiledMACIncompressibleDynamics.")
        prepared_les = _prepared_algebraic_les(dynamics)
        implicit_les = prepared_les is not None and prepared_les.model.coefficient > 0.0
        fixed = None if fixed_step_size is None else float(fixed_step_size)
        if fixed is not None and (fixed <= 0.0 or not np.isfinite(fixed)):
            raise ValueError("fixed_step_size must be positive and finite.")
        tolerance_ = float(tolerance)
        if implicit_les:
            if solve_method not in ("auto", "iterative") or hybrid_line_axis is not None:
                raise ValueError(
                    "Frozen MAC algebraic LES supports only the iterative momentum "
                    "and composite-pressure routes."
                )
            policy = _variable_viscosity_policy(
                tolerance_, maximum_iterations, linear_policy
            )
            pressure_policy = _variable_viscosity_policy(
                tolerance_, maximum_iterations, None
            )
            density = tuple(
                jnp.ones(
                    layout.shape,
                    dtype=dynamics.momentum.operators.pressure_space.dtype,
                )
                for layout in dynamics.momentum.operators.discretization.face_layouts
            )
            divergence, gradient = _constraint_operators(dynamics)
            helmholtz = None
            profile = "mac-frozen-algebraic-les-imex-euler"
            refresh_kind = "accepted-state-once-per-attempt"
        else:
            viscosity = float(np.asarray(dynamics.problem.viscosity))
            helmholtz = MACHelmholtzSolvePlan(
                dynamics.momentum,
                solve_method=solve_method,
                hybrid_line_axis=hybrid_line_axis,
                tolerance=tolerance_,
                maximum_iterations=maximum_iterations,
                linear_policy=linear_policy,
                fixed_mass_coefficient=None if fixed is None else 1.0,
                fixed_diffusion_coefficient=(
                    None if fixed is None else fixed * viscosity
                ),
                maximum_resource_bytes=maximum_resource_bytes,
            )
            policy = None
            pressure_policy = None
            density = None
            divergence = None
            gradient = None
            profile = "mac-constant-laplacian-imex-euler"
            refresh_kind = "none"
        self.dynamics = dynamics
        self.helmholtz = helmholtz
        self.variable_linear_policy = policy
        self.pressure_linear_policy = pressure_policy
        self.face_density = density
        self.divergence_operator = divergence
        self.gradient_operator = gradient
        self.fixed_step_size = fixed
        self.tolerance = tolerance_
        self.implicit_les = implicit_les
        self.temporal_profile = profile
        self.coefficient_refresh = refresh_kind
        self.prepared_les_id = None if prepared_les is None else prepared_les.prepared_id
        identifier = canonical_fingerprint(
            {
                "kind": profile,
                "dynamics": dynamics.compilation_id,
                "implicit_les": (None if not implicit_les else prepared_les.prepared_id),
                "viscosity_action": (
                    None if not implicit_les else prepared_les.viscosity_action.action_id
                ),
                "helmholtz": None if helmholtz is None else helmholtz.plan_id,
                "fixed_step_size": fixed,
                "pressure_coefficient": "dt",
                "coefficient_refresh": refresh_kind,
            }
        )
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("additive-ode",),
            method_class="ark",
            order=1,
            adaptive=fixed is None,
            history_depth=1,
            stage_abscissae=(0.0, 1.0),
            causal_stage_extent=1.0,
            noise_requirement="none",
            method_id=identifier,
        )
        self.method_id = identifier

    def _step_size(self, value: ArrayLike | None, /) -> Array:
        dtype = self.dynamics.momentum.operators.pressure_space.dtype
        if self.fixed_step_size is None:
            if value is None:
                raise ValueError("Dynamic MAC IMEX Euler requires step_size.")
            step = jnp.asarray(value, dtype=dtype).reshape(())
        else:
            step = jnp.asarray(self.fixed_step_size, dtype=dtype)
            if value is not None:
                supplied = jnp.asarray(value, dtype=dtype).reshape(())
                step = eqx.error_if(
                    step, supplied != step, "Fixed MAC IMEX Euler step cannot change."
                )
        return eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "MAC IMEX Euler step must be positive and finite.",
        )

    def _constant_step(
        self,
        time: Array,
        current_state: Array,
        step: Array,
        pressure: ArrayLike | None,
        args: Any,
        /,
    ) -> MACIMEXEulerResult:
        current_velocity = self.dynamics.unpack_velocity(current_state)
        attempted_time = time + step
        boundary_stage = self.dynamics.boundary_stage(attempted_time, args)
        explicit = _explicit_rate(self.dynamics, time, current_state, args)
        rhs = tuple(
            value + step * rate
            for value, rate in zip(current_velocity, explicit, strict=True)
        )
        viscosity = self.dynamics.problem.viscosity.astype(step.dtype)
        helmholtz = self.helmholtz.solve(
            rhs,
            boundary_stage,
            mass_coefficient=None if self.fixed_step_size is not None else 1.0,
            diffusion_coefficient=(
                None if self.fixed_step_size is not None else step * viscosity
            ),
            initial_guess=current_velocity,
        )
        incoming_pressure = _incoming_pressure(self.dynamics, pressure, step.dtype)
        projection = self.dynamics.projection.project(
            helmholtz.value,
            step,
            pressure=incoming_pressure,
            boundary_stage=boundary_stage,
        )
        candidate_velocity = self.dynamics.momentum.boundaries.enforce(
            projection.velocity, boundary_stage
        )
        candidate_state = self.dynamics.momentum.operators.velocity_space.flatten(
            candidate_velocity
        )
        finite = (
            boundary_stage.successful
            & helmholtz.finite
            & jnp.all(jnp.isfinite(candidate_state))
            & jnp.all(jnp.isfinite(projection.pressure))
        )
        accepted = finite & helmholtz.converged & projection.converged
        status = _frozen_status(
            jnp.asarray(True),
            boundary_stage.successful,
            jnp.asarray(True),
            helmholtz.converged,
            projection.converged,
        )
        return MACIMEXEulerResult(
            time=jnp.where(accepted, attempted_time, time),
            attempted_time=attempted_time,
            step_size=step,
            pressure_correction_coefficient=step,
            previous_state=current_state,
            state=jnp.where(accepted, candidate_state, current_state),
            velocity=tuple(
                jnp.where(accepted, candidate, current)
                for candidate, current in zip(
                    candidate_velocity, current_velocity, strict=True
                )
            ),
            pressure=jnp.where(accepted, projection.pressure, incoming_pressure),
            explicit_rate=explicit,
            helmholtz=helmholtz,
            projection=projection,
            les_stage=None,
            frozen_kinematic_viscosity=None,
            coefficient_state=None,
            coefficient_time=None,
            stage_inverse=None,
            finite=finite,
            accepted=accepted,
            status=status,
            stage_plan_id=None,
            predictor_inverse_id=None,
            projection_inverse_id=None,
            temporal_profile=self.temporal_profile,
            coefficient_refresh=self.coefficient_refresh,
            method_id=self.method_id,
        )

    def _frozen_les_step(
        self,
        time: Array,
        current_state: Array,
        step: Array,
        pressure: ArrayLike | None,
        args: Any,
        /,
    ) -> MACIMEXEulerResult:
        from ._mac_stage_inverse_general import MACVariableViscosityStagePlan

        prepared_les = self.dynamics.algebraic_les
        current_boundary = self.dynamics.boundary_stage(time, args)
        current_velocity = self.dynamics.momentum.boundaries.enforce(
            self.dynamics.unpack_velocity(current_state), current_boundary
        )
        les_stage, frozen_viscosity = _evaluate_frozen_les(
            prepared_les, current_velocity, current_boundary
        )
        explicit = _explicit_rate(
            self.dynamics,
            time,
            current_state,
            args,
            boundary_stage=current_boundary,
        )
        attempted_time = time + step
        boundary_stage = self.dynamics.boundary_stage(attempted_time, args)
        total_viscosity = frozen_viscosity + self.dynamics.problem.viscosity.astype(
            frozen_viscosity.dtype
        )
        stage_plan = MACVariableViscosityStagePlan(
            self.dynamics.momentum,
            self.face_density,
            total_viscosity,
            step,
            rhs_scale=step,
            viscosity_action=prepared_les.viscosity_action,
            stage_id=f"{self.method_id}/accepted-state",
        )
        inverse = stage_plan.inverse(
            boundary_stage, linear_policy=self.variable_linear_policy
        )
        physical_rhs = tuple(
            value / step + rate
            for value, rate in zip(current_velocity, explicit, strict=True)
        )
        predictor = inverse.solve_affine(physical_rhs)
        inverse_operator = inverse.operator()
        projection_plan = CompositeMACProjectionPlan(
            self.divergence_operator,
            self.gradient_operator,
            inverse_operator,
            self.dynamics.momentum.operators.gauge_project,
            linear_policy=self.pressure_linear_policy,
            tolerance=self.tolerance,
        )
        incoming_pressure = _incoming_pressure(self.dynamics, pressure, step.dtype)
        projection = projection_plan.project(predictor.value, pressure=incoming_pressure)
        candidate_velocity = self.dynamics.momentum.boundaries.enforce(
            projection.velocity, boundary_stage
        )
        candidate_state = self.dynamics.momentum.operators.velocity_space.flatten(
            candidate_velocity
        )
        finite = (
            current_boundary.finite
            & boundary_stage.finite
            & les_stage.finite
            & predictor.finite
            & projection.finite
            & jnp.all(jnp.isfinite(candidate_state))
            & jnp.all(jnp.isfinite(projection.pressure))
        )
        accepted = (
            current_boundary.successful
            & boundary_stage.successful
            & les_stage.successful
            & predictor.converged
            & projection.accepted
            & finite
        )
        status = _frozen_status(
            jnp.asarray(True),
            current_boundary.successful & boundary_stage.successful,
            les_stage.successful,
            predictor.converged,
            projection.accepted,
        )
        return MACIMEXEulerResult(
            time=jnp.where(accepted, attempted_time, time),
            attempted_time=attempted_time,
            step_size=step,
            pressure_correction_coefficient=step,
            previous_state=current_state,
            state=jnp.where(accepted, candidate_state, current_state),
            velocity=tuple(
                jnp.where(accepted, candidate, current)
                for candidate, current in zip(
                    candidate_velocity, current_velocity, strict=True
                )
            ),
            pressure=jnp.where(accepted, projection.pressure, incoming_pressure),
            explicit_rate=explicit,
            helmholtz=None,
            projection=projection,
            les_stage=les_stage,
            frozen_kinematic_viscosity=frozen_viscosity,
            coefficient_state=current_state,
            coefficient_time=time,
            stage_inverse=predictor,
            finite=finite,
            accepted=accepted,
            status=status,
            stage_plan_id=stage_plan.plan_id,
            predictor_inverse_id=inverse.operator_id,
            projection_inverse_id=projection_plan.inverse_momentum.operator_id,
            temporal_profile=self.temporal_profile,
            coefficient_refresh=self.coefficient_refresh,
            method_id=self.method_id,
        )

    def step(
        self,
        time: ArrayLike,
        state: ArrayLike,
        /,
        *,
        step_size: ArrayLike | None = None,
        pressure: ArrayLike | None = None,
        args: Any = None,
    ) -> MACIMEXEulerResult:
        step = self._step_size(step_size)
        current_state = self.dynamics.validate_state(state)
        time_ = jnp.asarray(time, dtype=step.dtype).reshape(())
        if self.implicit_les:
            return self._frozen_les_step(time_, current_state, step, pressure, args)
        return self._constant_step(time_, current_state, step, pressure, args)


class MACSBDF2State(StrictModule):
    """Complete fixed-step restart state; it alone determines the next refresh."""

    time: Array
    previous_state: Array
    state: Array
    previous_explicit_rate: FaceVelocity
    explicit_rate: FaceVelocity
    pressure: Array
    accepted_steps: Array
    valid: Array
    status: Array
    method_id: str = eqx.field(static=True)


class MACSBDF2GStabilityLedger(StrictModule):
    """Exact BDF2 G-identity on the weighted MAC velocity space."""

    g_energy_before: Array
    g_energy_after: Array
    temporal_dissipation: Array
    bdf_work: Array
    identity_defect: Array
    finite: Array
    successful: Array
    identity: str = eqx.field(static=True)


def _velocity_norm_squared(
    dynamics: CompiledMACIncompressibleDynamics,
    velocity: FaceVelocity,
    /,
) -> Array:
    return jnp.real(dynamics.momentum.operators.velocity_space.inner(velocity, velocity))


def _g_stability_ledger(
    dynamics: CompiledMACIncompressibleDynamics,
    previous: FaceVelocity,
    current: FaceVelocity,
    candidate: FaceVelocity,
    accepted: Array,
    /,
) -> MACSBDF2GStabilityLedger:
    two_current_minus_previous = tuple(
        2.0 * value - old for value, old in zip(current, previous, strict=True)
    )
    two_candidate_minus_current = tuple(
        2.0 * value - old for value, old in zip(candidate, current, strict=True)
    )
    second_difference = tuple(
        value - 2.0 * middle + old
        for value, middle, old in zip(candidate, current, previous, strict=True)
    )
    bdf_difference = tuple(
        1.5 * value - 2.0 * middle + 0.5 * old
        for value, middle, old in zip(candidate, current, previous, strict=True)
    )
    space = dynamics.momentum.operators.velocity_space
    before = 0.25 * (
        _velocity_norm_squared(dynamics, current)
        + _velocity_norm_squared(dynamics, two_current_minus_previous)
    )
    after = 0.25 * (
        _velocity_norm_squared(dynamics, candidate)
        + _velocity_norm_squared(dynamics, two_candidate_minus_current)
    )
    temporal = 0.25 * _velocity_norm_squared(dynamics, second_difference)
    work = jnp.real(space.inner(bdf_difference, candidate))
    defect = jnp.abs(work - (after - before + temporal))
    values = jnp.stack((before, after, temporal, work, defect))
    finite = jnp.all(jnp.isfinite(values))
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(values)))
    tolerance = 4096.0 * jnp.finfo(values.dtype).eps * scale
    return MACSBDF2GStabilityLedger(
        before,
        after,
        temporal,
        work,
        defect,
        finite,
        accepted & finite & (defect <= tolerance),
        "inner(BDF2(u[n+1],u[n],u[n-1]),u[n+1])="
        "G(u[n+1],u[n])-G(u[n],u[n-1])+one-quarter-norm(second-difference)^2",
    )


class MACSBDF2StepResult(StrictModule):
    history: MACSBDF2State
    attempted_time: Array
    step_size: Array
    pressure_correction_coefficient: Array
    velocity: FaceVelocity
    pressure: Array
    helmholtz: MACHelmholtzResult | None
    projection: MACPressureProjectionResult | CompositeMACProjectionResult
    coefficient_projection: MACPressureProjectionResult | None
    les_stage: MACLESStageResult | None
    frozen_kinematic_viscosity: Array | None
    coefficient_state: Array | None
    coefficient_time: Array | None
    stage_inverse: MACOperatorStageSolveResult | None
    g_stability: MACSBDF2GStabilityLedger | None
    finite: Array
    accepted: Array
    status: Array
    startup: bool = eqx.field(static=True)
    stage_plan_id: str | None = eqx.field(static=True)
    predictor_inverse_id: str | None = eqx.field(static=True)
    projection_inverse_id: str | None = eqx.field(static=True)
    temporal_profile: str = eqx.field(static=True)
    coefficient_extrapolation: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class MACSBDF2Method(StrictModule, NonTrainableState):
    """Fixed-step SBDF2 with projected 2u[n]-u[n-1] coefficient refresh."""

    dynamics: CompiledMACIncompressibleDynamics
    step_size: float = eqx.field(static=True)
    startup_method: MACIMEXEulerMethod
    helmholtz: MACHelmholtzSolvePlan | None
    implicit_les: bool = eqx.field(static=True)
    temporal_profile: str = eqx.field(static=True)
    coefficient_extrapolation: str = eqx.field(static=True)
    allows_adaptive_step: bool = eqx.field(static=True)
    capabilities: TemporalMethodCapabilities
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        step_size: float,
        /,
        *,
        solve_method: MACHelmholtzSolveMethod = "auto",
        hybrid_line_axis: int | None = None,
        tolerance: float = 1e-9,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
        maximum_resource_bytes: int = 512 * 1024**2,
    ):
        if not isinstance(dynamics, CompiledMACIncompressibleDynamics):
            raise TypeError("dynamics must be CompiledMACIncompressibleDynamics.")
        step = float(step_size)
        if step <= 0.0 or not np.isfinite(step):
            raise ValueError("MACSBDF2Method requires a positive finite fixed step.")
        startup = MACIMEXEulerMethod(
            dynamics,
            fixed_step_size=step,
            solve_method=solve_method,
            hybrid_line_axis=hybrid_line_axis,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            linear_policy=linear_policy,
            maximum_resource_bytes=maximum_resource_bytes,
        )
        if startup.implicit_les:
            helmholtz = None
            profile = "mac-frozen-algebraic-les-sbdf2"
            extrapolation = "projected-2*u[n]-u[n-1]-at-t[n+1]"
        else:
            viscosity = float(np.asarray(dynamics.problem.viscosity))
            helmholtz = MACHelmholtzSolvePlan(
                dynamics.momentum,
                solve_method=solve_method,
                hybrid_line_axis=hybrid_line_axis,
                tolerance=tolerance,
                maximum_iterations=maximum_iterations,
                linear_policy=linear_policy,
                fixed_mass_coefficient=1.5,
                fixed_diffusion_coefficient=step * viscosity,
                maximum_resource_bytes=maximum_resource_bytes,
            )
            profile = "mac-constant-laplacian-sbdf2"
            extrapolation = "none"
        identifier = canonical_fingerprint(
            {
                "kind": profile,
                "dynamics": dynamics.compilation_id,
                "step_size": step,
                "startup": startup.method_id,
                "helmholtz": None if helmholtz is None else helmholtz.plan_id,
                "pressure_coefficient": "2*dt/3",
                "coefficient_extrapolation": extrapolation,
                "failure": "atomic-retain-complete-history",
                "g_stability": "bdf2-weighted-velocity-identity",
            }
        )
        self.dynamics = dynamics
        self.step_size = step
        self.startup_method = startup
        self.helmholtz = helmholtz
        self.implicit_les = startup.implicit_les
        self.temporal_profile = profile
        self.coefficient_extrapolation = extrapolation
        self.allows_adaptive_step = False
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("additive-ode",),
            method_class="bdf",
            order=2,
            adaptive=False,
            history_depth=2,
            stage_abscissae=(1.0,),
            causal_stage_extent=1.0,
            noise_requirement="none",
            method_id=identifier,
        )
        self.method_id = identifier

    def initialize(
        self,
        time: ArrayLike,
        state: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
        args: Any = None,
    ) -> MACSBDF2StepResult:
        initial_state = self.dynamics.validate_state(state)
        dtype = self.dynamics.momentum.operators.pressure_space.dtype
        time_ = jnp.asarray(time, dtype=dtype).reshape(())
        startup = self.startup_method.step(
            time_, initial_state, pressure=pressure, args=args
        )
        initial_explicit = startup.explicit_rate
        following_explicit = jax.lax.cond(
            startup.accepted,
            lambda _: _explicit_rate(self.dynamics, startup.time, startup.state, args),
            lambda _: initial_explicit,
            operand=None,
        )
        history = MACSBDF2State(
            time=startup.time,
            previous_state=initial_state,
            state=startup.state,
            previous_explicit_rate=initial_explicit,
            explicit_rate=following_explicit,
            pressure=startup.pressure,
            accepted_steps=startup.accepted.astype(jnp.int32),
            valid=startup.accepted,
            status=startup.status,
            method_id=self.method_id,
        )
        return MACSBDF2StepResult(
            history=history,
            attempted_time=startup.attempted_time,
            step_size=startup.step_size,
            pressure_correction_coefficient=startup.pressure_correction_coefficient,
            velocity=startup.velocity,
            pressure=startup.pressure,
            helmholtz=startup.helmholtz,
            projection=startup.projection,
            coefficient_projection=None,
            les_stage=startup.les_stage,
            frozen_kinematic_viscosity=startup.frozen_kinematic_viscosity,
            coefficient_state=startup.coefficient_state,
            coefficient_time=startup.coefficient_time,
            stage_inverse=startup.stage_inverse,
            g_stability=None,
            finite=startup.finite,
            accepted=startup.accepted,
            status=startup.status,
            startup=True,
            stage_plan_id=startup.stage_plan_id,
            predictor_inverse_id=startup.predictor_inverse_id,
            projection_inverse_id=startup.projection_inverse_id,
            temporal_profile=self.temporal_profile,
            coefficient_extrapolation=self.coefficient_extrapolation,
            method_id=self.method_id,
        )

    def _constant_step(
        self,
        history: MACSBDF2State,
        current_state: Array,
        previous_state: Array,
        current_velocity: FaceVelocity,
        previous_velocity: FaceVelocity,
        step: Array,
        args: Any,
        /,
    ) -> MACSBDF2StepResult:
        attempted_time = history.time + step
        boundary_stage = self.dynamics.boundary_stage(attempted_time, args)
        rhs = tuple(
            2.0 * current - 0.5 * previous + step * (2.0 * current_rate - previous_rate)
            for current, previous, current_rate, previous_rate in zip(
                current_velocity,
                previous_velocity,
                history.explicit_rate,
                history.previous_explicit_rate,
                strict=True,
            )
        )
        helmholtz = self.helmholtz.solve(
            rhs, boundary_stage, initial_guess=current_velocity
        )
        pressure_coefficient = (2.0 / 3.0) * step
        projection = self.dynamics.projection.project(
            helmholtz.value,
            pressure_coefficient,
            pressure=history.pressure,
            boundary_stage=boundary_stage,
        )
        candidate_velocity = self.dynamics.momentum.boundaries.enforce(
            projection.velocity, boundary_stage
        )
        candidate_state = self.dynamics.momentum.operators.velocity_space.flatten(
            candidate_velocity
        )
        finite = (
            boundary_stage.successful
            & helmholtz.finite
            & jnp.all(jnp.isfinite(candidate_state))
            & jnp.all(jnp.isfinite(projection.pressure))
        )
        accepted = history.valid & finite & helmholtz.converged & projection.converged
        local_status = _frozen_status(
            history.valid,
            boundary_stage.successful,
            jnp.asarray(True),
            helmholtz.converged,
            projection.converged,
        )
        status = jnp.where(history.valid, local_status, history.status)
        next_explicit = jax.lax.cond(
            accepted,
            lambda _: _explicit_rate(
                self.dynamics, attempted_time, candidate_state, args
            ),
            lambda _: history.explicit_rate,
            operand=None,
        )
        next_history = MACSBDF2State(
            time=jnp.where(accepted, attempted_time, history.time),
            previous_state=jnp.where(accepted, current_state, previous_state),
            state=jnp.where(accepted, candidate_state, current_state),
            previous_explicit_rate=tuple(
                jnp.where(accepted, current, previous)
                for current, previous in zip(
                    history.explicit_rate,
                    history.previous_explicit_rate,
                    strict=True,
                )
            ),
            explicit_rate=next_explicit,
            pressure=jnp.where(accepted, projection.pressure, history.pressure),
            accepted_steps=history.accepted_steps + accepted.astype(jnp.int32),
            valid=accepted,
            status=status,
            method_id=self.method_id,
        )
        return MACSBDF2StepResult(
            history=next_history,
            attempted_time=attempted_time,
            step_size=step,
            pressure_correction_coefficient=pressure_coefficient,
            velocity=tuple(
                jnp.where(accepted, candidate, current)
                for candidate, current in zip(
                    candidate_velocity, current_velocity, strict=True
                )
            ),
            pressure=next_history.pressure,
            helmholtz=helmholtz,
            projection=projection,
            coefficient_projection=None,
            les_stage=None,
            frozen_kinematic_viscosity=None,
            coefficient_state=None,
            coefficient_time=None,
            stage_inverse=None,
            g_stability=_g_stability_ledger(
                self.dynamics,
                previous_velocity,
                current_velocity,
                candidate_velocity,
                accepted,
            ),
            finite=finite,
            accepted=accepted,
            status=status,
            startup=False,
            stage_plan_id=None,
            predictor_inverse_id=None,
            projection_inverse_id=None,
            temporal_profile=self.temporal_profile,
            coefficient_extrapolation=self.coefficient_extrapolation,
            method_id=self.method_id,
        )

    def _frozen_les_step(
        self,
        history: MACSBDF2State,
        current_state: Array,
        previous_state: Array,
        current_velocity: FaceVelocity,
        previous_velocity: FaceVelocity,
        step: Array,
        args: Any,
        /,
    ) -> MACSBDF2StepResult:
        from ._mac_stage_inverse_general import MACVariableViscosityStagePlan

        prepared_les = self.dynamics.algebraic_les
        attempted_time = history.time + step
        boundary_stage = self.dynamics.boundary_stage(attempted_time, args)
        extrapolated = tuple(
            2.0 * current - previous
            for current, previous in zip(current_velocity, previous_velocity, strict=True)
        )
        extrapolated = self.dynamics.momentum.boundaries.enforce(
            extrapolated, boundary_stage
        )
        coefficient_projection = self.dynamics.projection.project(
            extrapolated,
            1.0,
            boundary_stage=boundary_stage,
        )
        coefficient_velocity = self.dynamics.momentum.boundaries.enforce(
            coefficient_projection.velocity, boundary_stage
        )
        coefficient_state = self.dynamics.momentum.operators.velocity_space.flatten(
            coefficient_velocity
        )
        les_stage, frozen_viscosity = _evaluate_frozen_les(
            prepared_les, coefficient_velocity, boundary_stage
        )
        total_viscosity = frozen_viscosity + self.dynamics.problem.viscosity.astype(
            frozen_viscosity.dtype
        )
        stage_plan = MACVariableViscosityStagePlan(
            self.dynamics.momentum,
            self.startup_method.face_density,
            total_viscosity,
            step,
            rhs_scale=step,
            viscosity_action=prepared_les.viscosity_action,
            stage_id=f"{self.method_id}/projected-extrapolated-state",
        )
        inverse = stage_plan.inverse(
            boundary_stage,
            linear_policy=self.startup_method.variable_linear_policy,
        )
        physical_rhs = tuple(
            (2.0 * current - 0.5 * previous) / step + 2.0 * current_rate - previous_rate
            for current, previous, current_rate, previous_rate in zip(
                current_velocity,
                previous_velocity,
                history.explicit_rate,
                history.previous_explicit_rate,
                strict=True,
            )
        )
        predictor = inverse.solve_affine(physical_rhs)
        inverse_operator = inverse.operator()
        projection_plan = CompositeMACProjectionPlan(
            self.startup_method.divergence_operator,
            self.startup_method.gradient_operator,
            inverse_operator,
            self.dynamics.momentum.operators.gauge_project,
            linear_policy=self.startup_method.pressure_linear_policy,
            tolerance=self.startup_method.tolerance,
        )
        projection = projection_plan.project(predictor.value, pressure=history.pressure)
        candidate_velocity = self.dynamics.momentum.boundaries.enforce(
            projection.velocity, boundary_stage
        )
        candidate_state = self.dynamics.momentum.operators.velocity_space.flatten(
            candidate_velocity
        )
        coefficient_successful = coefficient_projection.converged & les_stage.successful
        finite = (
            boundary_stage.finite
            & coefficient_projection.finite
            & les_stage.finite
            & predictor.finite
            & projection.finite
            & jnp.all(jnp.isfinite(candidate_state))
            & jnp.all(jnp.isfinite(projection.pressure))
        )
        accepted = (
            history.valid
            & boundary_stage.successful
            & coefficient_successful
            & predictor.converged
            & projection.accepted
            & finite
        )
        status = _frozen_status(
            history.valid,
            boundary_stage.successful,
            coefficient_successful,
            predictor.converged,
            projection.accepted,
        )
        next_explicit = jax.lax.cond(
            accepted,
            lambda _: _explicit_rate(
                self.dynamics, attempted_time, candidate_state, args
            ),
            lambda _: history.explicit_rate,
            operand=None,
        )
        next_history = MACSBDF2State(
            time=jnp.where(accepted, attempted_time, history.time),
            previous_state=jnp.where(accepted, current_state, previous_state),
            state=jnp.where(accepted, candidate_state, current_state),
            previous_explicit_rate=tuple(
                jnp.where(accepted, current, previous)
                for current, previous in zip(
                    history.explicit_rate,
                    history.previous_explicit_rate,
                    strict=True,
                )
            ),
            explicit_rate=next_explicit,
            pressure=jnp.where(accepted, projection.pressure, history.pressure),
            accepted_steps=history.accepted_steps + accepted.astype(jnp.int32),
            valid=history.valid,
            status=jnp.where(accepted, status, history.status),
            method_id=self.method_id,
        )
        ledger = _g_stability_ledger(
            self.dynamics,
            previous_velocity,
            current_velocity,
            candidate_velocity,
            accepted,
        )
        return MACSBDF2StepResult(
            history=next_history,
            attempted_time=attempted_time,
            step_size=step,
            pressure_correction_coefficient=(2.0 / 3.0) * step,
            velocity=tuple(
                jnp.where(accepted, candidate, current)
                for candidate, current in zip(
                    candidate_velocity, current_velocity, strict=True
                )
            ),
            pressure=next_history.pressure,
            helmholtz=None,
            projection=projection,
            coefficient_projection=coefficient_projection,
            les_stage=les_stage,
            frozen_kinematic_viscosity=frozen_viscosity,
            coefficient_state=coefficient_state,
            coefficient_time=attempted_time,
            stage_inverse=predictor,
            g_stability=ledger,
            finite=finite,
            accepted=accepted,
            status=status,
            startup=False,
            stage_plan_id=stage_plan.plan_id,
            predictor_inverse_id=inverse.operator_id,
            projection_inverse_id=projection_plan.inverse_momentum.operator_id,
            temporal_profile=self.temporal_profile,
            coefficient_extrapolation=self.coefficient_extrapolation,
            method_id=self.method_id,
        )

    def step(self, history: MACSBDF2State, /, *, args: Any = None) -> MACSBDF2StepResult:
        if not isinstance(history, MACSBDF2State):
            raise TypeError("history must be MACSBDF2State.")
        if history.method_id != self.method_id:
            raise ValueError("MAC SBDF2 history belongs to a different method.")
        current_state = self.dynamics.validate_state(history.state)
        previous_state = self.dynamics.validate_state(history.previous_state)
        current_velocity = self.dynamics.unpack_velocity(current_state)
        previous_velocity = self.dynamics.unpack_velocity(previous_state)
        self.dynamics.momentum.operators.validate_velocity(history.previous_explicit_rate)
        self.dynamics.momentum.operators.validate_velocity(history.explicit_rate)
        step = jnp.asarray(
            self.step_size,
            dtype=self.dynamics.momentum.operators.pressure_space.dtype,
        )
        if self.implicit_les:
            return self._frozen_les_step(
                history,
                current_state,
                previous_state,
                current_velocity,
                previous_velocity,
                step,
                args,
            )
        return self._constant_step(
            history,
            current_state,
            previous_state,
            current_velocity,
            previous_velocity,
            step,
            args,
        )


__all__ = [
    "MACHelmholtzResourceEstimate",
    "MACHelmholtzResult",
    "MACHelmholtzSolveMethod",
    "MACHelmholtzSolvePlan",
    "MACIMEXEulerMethod",
    "MACIMEXEulerResult",
    "MACSBDF2Method",
    "MACSBDF2GStabilityLedger",
    "MACSBDF2State",
    "MACSBDF2StepResult",
    "MAC_VISCOUS_BOUNDARY_FAILURE",
    "MAC_VISCOUS_CLOSURE_FAILURE",
    "MAC_VISCOUS_HELMHOLTZ_FAILURE",
    "MAC_VISCOUS_HISTORY_INVALID",
    "MAC_VISCOUS_PROJECTION_FAILURE",
    "MAC_VISCOUS_SUCCESS",
]
