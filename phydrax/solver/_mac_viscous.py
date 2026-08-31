#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

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
from ..linalg import (
    ArraySpace,
    ConjugateGradient,
    FFTLinearTransform,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    RealTrigonometricTransform,
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
from ._structured_incompressible import MACPressureProjectionResult


MACHelmholtzSolveMethod: TypeAlias = Literal["auto", "transform", "hybrid", "iterative"]
MAC_VISCOUS_SUCCESS = 0
MAC_VISCOUS_HELMHOLTZ_FAILURE = -1
MAC_VISCOUS_PROJECTION_FAILURE = -2
MAC_VISCOUS_BOUNDARY_FAILURE = -3
MAC_VISCOUS_HISTORY_INVALID = -4
_ESSENTIAL_NORMAL_KINDS = (
    "no-slip",
    "free-slip",
    "symmetry",
    "velocity-inflow",
    "normal-flux-inflow",
)


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


def _normal_is_essential(
    momentum: PreparedMACMomentumOperators, component: int, /
) -> bool:
    axis = momentum.operators.discretization.grid.structured_axes[component]
    return axis.periodic or all(
        momentum.boundaries.side_kind(component, side) in _ESSENTIAL_NORMAL_KINDS
        for side in ("lower", "upper")
    )


def _uniform_spacing(axis, /) -> float | None:
    widths = np.asarray(axis.interval_widths, dtype=float)
    if widths.size == 0:
        return None
    spacing = float(widths[0])
    if (
        not np.isfinite(spacing)
        or spacing <= 0.0
        or not np.allclose(widths, spacing, rtol=1e-10, atol=1e-12)
    ):
        return None
    return spacing


def _axis_transform(
    momentum: PreparedMACMomentumOperators,
    component: int,
    derivative_axis: int,
    /,
) -> tuple[AbstractLinearTransform, Array, float] | None:
    axis = momentum.operators.discretization.grid.structured_axes[derivative_axis]
    spacing = _uniform_spacing(axis)
    if spacing is None:
        return None
    dtype = momentum.operators.pressure_space.dtype
    if axis.periodic:
        count = int(axis.interval_widths.size)
        transform: AbstractLinearTransform = FFTLinearTransform(
            count, dtype=np.result_type(dtype, np.complex64)
        )
        angles = 2.0 * np.pi * np.arange(count) / count
        trace = 0.0 if count == 1 else 2.0 * count / spacing**2
    elif derivative_axis == component:
        if not _normal_is_essential(momentum, component):
            return None
        count = int(axis.interval_widths.size) - 1
        if count < 1:
            return None
        transform = RealTrigonometricTransform("dst", 1, count, dtype=dtype)
        angles = (np.arange(count) + 1.0) * np.pi / (count + 1.0)
        trace = 2.0 * count / spacing**2
    else:
        count = int(axis.interval_widths.size)
        lower_d = momentum.boundaries.tangential_dirichlet(derivative_axis, "lower")
        upper_d = momentum.boundaries.tangential_dirichlet(derivative_axis, "upper")
        if lower_d and upper_d:
            transform = RealTrigonometricTransform("dst", 2, count, dtype=dtype)
            angles = (np.arange(count) + 1.0) * np.pi / count
            trace = (2.0 * count + 2.0) / spacing**2
        elif not lower_d and not upper_d:
            transform = RealTrigonometricTransform("dct", 2, count, dtype=dtype)
            angles = np.arange(count) * np.pi / count
            trace = max(2.0 * count - 2.0, 0.0) / spacing**2
        elif lower_d:
            transform = RealTrigonometricTransform("dst", 4, count, dtype=dtype)
            angles = (np.arange(count) + 0.5) * np.pi / count
            trace = 2.0 * count / spacing**2
        else:
            transform = RealTrigonometricTransform("dct", 4, count, dtype=dtype)
            angles = (np.arange(count) + 0.5) * np.pi / count
            trace = 2.0 * count / spacing**2
    spectrum = 4.0 * np.sin(0.5 * angles) ** 2 / spacing**2
    return transform, jnp.asarray(spectrum, dtype=dtype), trace


def _modal_sum(spectra: tuple[Array, ...], /) -> Array:
    shape = tuple(int(value.size) for value in spectra)
    result = jnp.zeros(shape, dtype=jnp.result_type(*[value.dtype for value in spectra]))
    for axis, spectrum in enumerate(spectra):
        reshape = [1] * len(shape)
        reshape[axis] = int(spectrum.size)
        result = result + spectrum.reshape(tuple(reshape))
    return result


def _line_coefficients(
    momentum: PreparedMACMomentumOperators,
    component: int,
    line_axis: int,
    /,
) -> tuple[Array, Array, Array, tuple[Array, Array] | None]:
    axis = momentum.operators.discretization.grid.structured_axes[line_axis]
    dtype = momentum.operators.pressure_space.dtype
    widths = jnp.asarray(axis.interval_widths, dtype=dtype)
    centers = jnp.asarray(axis.interval_centers, dtype=dtype)
    if axis.periodic:
        period = jnp.asarray(axis.bounds[1] - axis.bounds[0], dtype=dtype)
        previous = jnp.roll(centers, 1).at[0].add(-period)
        distances = centers - previous
        if line_axis == component:
            dual = momentum.face_dual_widths[component]
            lower_full = -1.0 / (dual * jnp.roll(widths, 1))
            upper_full = -1.0 / (dual * widths)
        else:
            lower_full = -1.0 / (widths * distances)
            upper_full = -1.0 / (widths * jnp.roll(distances, -1))
        return (
            lower_full[1:],
            -(lower_full + upper_full),
            upper_full[:-1],
            (lower_full[0], upper_full[-1]),
        )
    if line_axis == component:
        dual = momentum.face_dual_widths[component][1:-1]
        diagonal = 1.0 / (dual * widths[:-1]) + 1.0 / (dual * widths[1:])
        if dual.size == 1:
            empty = jnp.zeros((0,), dtype=dtype)
            return empty, diagonal, empty, None
        return (
            -1.0 / (dual[1:] * widths[1:-1]),
            diagonal,
            -1.0 / (dual[:-1] * widths[1:-1]),
            None,
        )
    count = int(widths.size)
    distances = centers[1:] - centers[:-1]
    lower = -1.0 / (widths[1:] * distances)
    upper = -1.0 / (widths[:-1] * distances)
    diagonal = jnp.zeros((count,), dtype=dtype)
    if count > 1:
        diagonal = diagonal.at[1:].add(1.0 / (widths[1:] * distances))
        diagonal = diagonal.at[:-1].add(1.0 / (widths[:-1] * distances))
    if momentum.boundaries.tangential_dirichlet(line_axis, "lower"):
        diagonal = diagonal.at[0].add(1.0 / (widths[0] * (centers[0] - axis.bounds[0])))
    if momentum.boundaries.tangential_dirichlet(line_axis, "upper"):
        diagonal = diagonal.at[-1].add(
            1.0 / (widths[-1] * (axis.bounds[1] - centers[-1]))
        )
    return lower, diagonal, upper, None


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
        direct_eligible = _normal_is_essential(momentum, component)
        data: list[tuple[AbstractLinearTransform, Array, float]] = []
        if direct_eligible:
            for derivative_axis in range(momentum.dimension):
                axis_data = _axis_transform(momentum, component, derivative_axis)
                if axis_data is None:
                    direct_eligible = False
                    break
                data.append(axis_data)
        if direct_eligible:
            transform = TensorLinearTransform(tuple(item[0] for item in data))
            modal_values = _modal_sum(tuple(item[1] for item in data))
            shape = _unknown_shape(momentum, component)
            probe = jnp.sin(
                0.37 * jnp.arange(int(np.prod(shape)), dtype=space.dtype).reshape(shape)
                + component
            )
            exact = _PositiveUnknownLaplacian(momentum, component)(probe)
            represented = jnp.real(
                transform.synthesize(modal_values * transform.analyze(probe))
            ).astype(exact.dtype)
            action_defect = jnp.linalg.norm((represented - exact).reshape((-1,)))
            expected_trace = sum(
                int(np.prod(shape)) // shape[axis] * item[2]
                for axis, item in enumerate(data)
            )
            trace_defect = jnp.abs(jnp.sum(modal_values) - expected_trace)
            epsilon = jnp.finfo(space.dtype).eps
            scale = jnp.maximum(1.0, jnp.linalg.norm(exact.reshape((-1,))))
            direct_eligible = bool(
                np.asarray(
                    jnp.isfinite(action_defect)
                    & jnp.isfinite(trace_defect)
                    & (
                        action_defect
                        <= jnp.maximum(100.0 * tolerance, 4096.0 * epsilon * scale)
                    )
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
        hybrid_eligible = hybrid_line_axis is not None and _normal_is_essential(
            momentum, component
        )
        transverse_data: list[tuple[AbstractLinearTransform, Array, float]] = []
        if hybrid_eligible:
            for derivative_axis in range(momentum.dimension):
                if derivative_axis == hybrid_line_axis:
                    continue
                axis_data = _axis_transform(momentum, component, derivative_axis)
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
            lower, diagonal, upper, corners = _line_coefficients(
                momentum, component, hybrid_line_axis
            )
            transverse_modal = (
                _modal_sum(tuple(item[1] for item in transverse_data))
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
            action_defect = jnp.linalg.norm((hybrid.apply(probe) - exact).reshape((-1,)))
            epsilon = jnp.finfo(space.dtype).eps
            scale = jnp.maximum(1.0, jnp.linalg.norm(exact.reshape((-1,))))
            hybrid_eligible = bool(
                np.asarray(
                    jnp.isfinite(action_defect)
                    & (
                        action_defect
                        <= jnp.maximum(100.0 * tolerance, 4096.0 * epsilon * scale)
                    )
                )
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
            count = int(np.prod(transform.modal_space.shape))
            factor_bytes = count * np.dtype(transform.modal_space.dtype).itemsize
            workspace_bytes = 3 * factor_bytes
            if factor_bytes + workspace_bytes > budget:
                raise ValueError("MAC transform resources exceed the configured budget.")
            resources = MACHelmholtzResourceEstimate(
                component,
                route,
                count,
                factor_bytes,
                workspace_bytes,
                factor_bytes + workspace_bytes,
            )
        else:
            workspace_bytes = 6 * space.size * np.dtype(space.dtype).itemsize
            if workspace_bytes > budget:
                raise ValueError("MAC iterative resources exceed the configured budget.")
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
) -> FaceVelocity:
    _, convection, _, forcing = dynamics.rate_components(time, state, args)
    return tuple(
        -advective + source for advective, source in zip(convection, forcing, strict=True)
    )


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
    helmholtz: MACHelmholtzResult
    projection: MACPressureProjectionResult
    finite: Array
    accepted: Array
    status: Array
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class MACIMEXEulerMethod(StrictModule, NonTrainableState):
    """Backward-Euler diffusion with explicit transport and pressure correction."""

    dynamics: CompiledMACIncompressibleDynamics
    helmholtz: MACHelmholtzSolvePlan
    fixed_step_size: float | None = eqx.field(static=True)
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
        fixed = None if fixed_step_size is None else float(fixed_step_size)
        if fixed is not None and (fixed <= 0.0 or not np.isfinite(fixed)):
            raise ValueError("fixed_step_size must be positive and finite.")
        viscosity = float(np.asarray(dynamics.problem.viscosity))
        helmholtz = MACHelmholtzSolvePlan(
            dynamics.momentum,
            solve_method=solve_method,
            hybrid_line_axis=hybrid_line_axis,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            linear_policy=linear_policy,
            fixed_mass_coefficient=None if fixed is None else 1.0,
            fixed_diffusion_coefficient=None if fixed is None else fixed * viscosity,
            maximum_resource_bytes=maximum_resource_bytes,
        )
        self.dynamics = dynamics
        self.helmholtz = helmholtz
        self.fixed_step_size = fixed
        self.method_id = canonical_fingerprint(
            {
                "kind": "mac-imex-euler",
                "dynamics": dynamics.compilation_id,
                "helmholtz": helmholtz.plan_id,
                "fixed_step_size": fixed,
                "pressure_coefficient": "dt",
            }
        )

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
        current_velocity = self.dynamics.unpack_velocity(current_state)
        time_ = jnp.asarray(time, dtype=step.dtype).reshape(())
        attempted_time = time_ + step
        boundary_stage = self.dynamics.momentum.boundaries.evaluate(attempted_time, args)
        explicit = _explicit_rate(self.dynamics, time_, current_state, args)
        rhs = tuple(
            value + step * rate
            for value, rate in zip(current_velocity, explicit, strict=True)
        )
        viscosity = self.dynamics.problem.viscosity.astype(step.dtype)
        helmholtz = self.helmholtz.solve(
            rhs,
            boundary_stage,
            mass_coefficient=None if self.fixed_step_size is not None else 1.0,
            diffusion_coefficient=None
            if self.fixed_step_size is not None
            else step * viscosity,
            initial_guess=current_velocity,
        )
        incoming_pressure = (
            jnp.zeros(
                self.dynamics.momentum.operators.discretization.cell_shape,
                dtype=step.dtype,
            )
            if pressure is None
            else self.dynamics.momentum.operators.gauge_project(pressure)
        )
        projection = self.dynamics.projection.project(
            helmholtz.value, step, pressure=incoming_pressure
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
        status = jnp.where(
            ~boundary_stage.successful,
            MAC_VISCOUS_BOUNDARY_FAILURE,
            jnp.where(
                ~helmholtz.converged,
                MAC_VISCOUS_HELMHOLTZ_FAILURE,
                jnp.where(
                    ~projection.converged,
                    MAC_VISCOUS_PROJECTION_FAILURE,
                    MAC_VISCOUS_SUCCESS,
                ),
            ),
        ).astype(jnp.int32)
        return MACIMEXEulerResult(
            time=jnp.where(accepted, attempted_time, time_),
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
            finite=finite,
            accepted=accepted,
            status=status,
            method_id=self.method_id,
        )


class MACSBDF2State(StrictModule):
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


class MACSBDF2StepResult(StrictModule):
    history: MACSBDF2State
    attempted_time: Array
    step_size: Array
    pressure_correction_coefficient: Array
    velocity: FaceVelocity
    pressure: Array
    helmholtz: MACHelmholtzResult
    projection: MACPressureProjectionResult
    finite: Array
    accepted: Array
    status: Array
    startup: bool = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class MACSBDF2Method(StrictModule, NonTrainableState):
    """Fixed-step SBDF2 with IMEX-Euler startup and fail-closed history."""

    dynamics: CompiledMACIncompressibleDynamics
    step_size: float = eqx.field(static=True)
    startup_method: MACIMEXEulerMethod
    helmholtz: MACHelmholtzSolvePlan
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
        identifier = canonical_fingerprint(
            {
                "kind": "mac-sbdf2",
                "dynamics": dynamics.compilation_id,
                "step_size": step,
                "startup": startup.method_id,
                "helmholtz": helmholtz.plan_id,
                "pressure_coefficient": "2*dt/3",
                "failure": "retain-complete-history",
            }
        )
        self.dynamics = dynamics
        self.step_size = step
        self.startup_method = startup
        self.helmholtz = helmholtz
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
        initial_explicit = _explicit_rate(self.dynamics, time_, initial_state, args)
        startup = self.startup_method.step(
            time_, initial_state, pressure=pressure, args=args
        )
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
            finite=startup.finite,
            accepted=startup.accepted,
            status=startup.status,
            startup=True,
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
        step = jnp.asarray(
            self.step_size,
            dtype=self.dynamics.momentum.operators.pressure_space.dtype,
        )
        attempted_time = history.time + step
        boundary_stage = self.dynamics.momentum.boundaries.evaluate(attempted_time, args)
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
            helmholtz.value, pressure_coefficient, pressure=history.pressure
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
        local_status = jnp.where(
            ~boundary_stage.successful,
            MAC_VISCOUS_BOUNDARY_FAILURE,
            jnp.where(
                ~helmholtz.converged,
                MAC_VISCOUS_HELMHOLTZ_FAILURE,
                jnp.where(
                    ~projection.converged,
                    MAC_VISCOUS_PROJECTION_FAILURE,
                    MAC_VISCOUS_SUCCESS,
                ),
            ),
        ).astype(jnp.int32)
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
                    history.explicit_rate, history.previous_explicit_rate, strict=True
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
            finite=finite,
            accepted=accepted,
            status=status,
            startup=False,
            method_id=self.method_id,
        )


__all__ = [
    "MACHelmholtzResourceEstimate",
    "MACHelmholtzResult",
    "MACHelmholtzSolveMethod",
    "MACHelmholtzSolvePlan",
    "MACIMEXEulerMethod",
    "MACIMEXEulerResult",
    "MACSBDF2Method",
    "MACSBDF2State",
    "MACSBDF2StepResult",
    "MAC_VISCOUS_BOUNDARY_FAILURE",
    "MAC_VISCOUS_HELMHOLTZ_FAILURE",
    "MAC_VISCOUS_HISTORY_INVALID",
    "MAC_VISCOUS_PROJECTION_FAILURE",
    "MAC_VISCOUS_SUCCESS",
]
