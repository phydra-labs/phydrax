#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import (
    FaceVelocity,
    PreparedMACOperators,
)
from ..linalg import (
    DiagonalLinearOperator,
    FunctionLinearOperator,
    JacobiPreconditionerBuilder,
    LinearSolveControl,
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
    solve_many,
    TolerancePolicy,
)
from ..typing import checked


def _maximum_abs(values: tuple[Array, ...], dtype: jnp.dtype, /) -> Array:
    if not values:
        return jnp.asarray(0.0, dtype=dtype)
    return jnp.max(jnp.stack(tuple(jnp.max(jnp.abs(value)) for value in values)))


def _flux_scale(operators: PreparedMACOperators, velocity: FaceVelocity, /) -> Array:
    """Volume-norm flux scale ``sqrt(sum_f (u_f A_f)^2 / (A_f d_f))``.

    Up to a stencil constant it bounds the volume-weighted MAC divergence norm
    of ``velocity``: it is the gross flux magnitude whose net cancellation the
    projection enforces and the rounding scale of each divergence it forms.
    """
    total = jnp.asarray(0.0, dtype=operators.pressure_space.dtype)
    for component, measure, dual in zip(
        velocity,
        operators.discretization.face_measures,
        operators.face_dual_measures,
        strict=True,
    ):
        total = total + jnp.sum((component * measure) ** 2 / dual)
    return jnp.sqrt(total)


class _VariableCoefficientMACPressureAction(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    face_coefficient: FaceVelocity

    def __init__(
        self,
        operators: PreparedMACOperators,
        face_coefficient: FaceVelocity,
        /,
    ) -> None:
        self.operators = operators
        self.face_coefficient = operators.validate_velocity(face_coefficient)

    def __call__(self, pressure: Array, /) -> Array:
        return self.operators.positive_gauged_weighted_laplacian(
            pressure, self.face_coefficient
        )


def _pressure_system(
    operators: PreparedMACOperators,
    face_coefficient: FaceVelocity,
    operator_id: str,
    problem_id: str,
    diagonal_id: str,
    /,
    *,
    jacobi_preconditioning: bool,
) -> tuple[LinearSystem, DiagonalLinearOperator | None]:
    """Gauged SPD pressure system and optional exact diagonal.

    The Jacobi diagonal is prepared only when requested. Schur-complement
    basis solves can be better conditioned without diagonal scaling because
    their localized right-hand sides already live in the mean-free pressure
    subspace.
    """
    pressure_operator = FunctionLinearOperator(
        _VariableCoefficientMACPressureAction(operators, face_coefficient),
        source=operators.pressure_space,
        target=operators.pressure_space,
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
    diagonal = (
        DiagonalLinearOperator(
            operators.positive_gauged_weighted_laplacian_diagonal(
                face_coefficient
            ).reshape(-1),
            space=operators.pressure_space,
            properties=OperatorProperties(
                diagonal=True,
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "diagonal": "construction",
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=diagonal_id,
        )
        if jacobi_preconditioning
        else None
    )
    return LinearSystem(pressure_operator, problem_id=problem_id), diagonal


class MACVariableDensityProjectionResult(StrictModule):
    """Fail-closed pressure impulse applied to staggered face momentum."""

    momentum: FaceVelocity
    velocity: FaceVelocity
    pressure: Array
    pressure_increment: Array
    pressure_impulse: FaceVelocity
    divergence_before: Array
    divergence_after: Array
    divergence_target: Array
    divergence_defect: Array
    pressure_residual: Array
    compatible_rhs: Array
    face_inverse_density: FaceVelocity
    gauge_defect: Array
    coefficient_contrast: Array
    preparation_id: str = eqx.field(static=True)
    residual_norm: Array
    divergence_norm: Array
    divergence_scale: Array
    compatibility_defect: Array
    divergence_identity_residual: Array
    momentum_impulse_residual: Array
    velocity_identity_residual: Array
    minimum_face_density: Array
    positive: Array
    finite: Array
    linear: LinearSolveResult
    converged: Array
    successful: Array
    solve_method: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)


class MACVariableDensityRateProjectionResult(StrictModule):
    """Projected velocity rate and its pressure force per unit face volume."""

    velocity_rate: FaceVelocity
    momentum_pressure_rate: FaceVelocity
    pressure: Array
    divergence_before: Array
    divergence_after: Array
    divergence_target: Array
    divergence_defect: Array
    pressure_residual: Array
    compatible_rhs: Array
    face_inverse_density: FaceVelocity
    gauge_defect: Array
    coefficient_contrast: Array
    preparation_id: str = eqx.field(static=True)
    residual_norm: Array
    divergence_norm: Array
    divergence_scale: Array
    compatibility_defect: Array
    positive: Array
    finite: Array
    linear: LinearSolveResult
    converged: Array
    successful: Array
    solve_method: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)


class _VariableDensityProjectionPreparation(StrictModule):
    """Validated lane state and the physical right-hand side of its pressure solve."""

    momentum: FaceVelocity
    face_inverse_density: FaceVelocity
    step: Array
    incoming_pressure: Array
    target: Array
    velocity_before: FaceVelocity
    divergence_before: Array
    rhs: Array
    incompatible_defect: Array
    compatibility_defect: Array
    coefficient: FaceVelocity
    divergence_scale: Array
    absolute_tolerance: Array


class MACVariableDensityProjectionPlan(StrictModule, NonTrainableState):
    """Prepared iterative variable-coefficient MAC pressure projection.

    The increment ``phi`` solves the gauged SPD system
    ``A phi = b`` with ``A = -div(c grad .) + mean(.)``, ``c = dt / rho_f`` and
    ``b = -P(div u* - target)`` (``P`` removes the volume mean). The solve is
    native PCG, optionally with a Jacobi preconditioner refreshed from the
    exact runtime diagonal. It has one threshold,
    ``||A phi - b||_V <= tolerance * (sigma + ||b||_V)``. The native PCG only
    stops when the true residual meets it: the recurrence residual nominates a
    stop, the true residual is evaluated, and on a miss the recurrence
    restarts from the true residual. Recurrence drift at high density contrast
    therefore costs iterations, never a stop that the final certification
    rejects. ``sigma`` (`divergence_scale`) is the volume-norm flux scale of
    ``u*``, of the incoming-pressure correction and of the target divergence.
    The absolute part therefore scales with the flow rather than carrying
    fixed units.

    A projection is successful exactly when that native solve succeeds, the
    target is compatible with the net boundary flux to ``tolerance * sigma``
    (the mean defect is not removable by any pressure), and the applied impulse,
    velocity, gauge and divergence identities hold to dtype rounding. No other
    tolerance test is applied.
    """

    operators: PreparedMACOperators
    tolerance: float = eqx.field(static=True)
    solve_method: str = eqx.field(static=True)
    jacobi_preconditioning: bool = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    linear_problem: LinearSystem
    prepared_linear: PreparedLinearSolve
    operator_id: str = eqx.field(static=True)
    pressure_problem_id: str = eqx.field(static=True)
    preconditioner_setup_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        operators: PreparedMACOperators,
        /,
        *,
        tolerance: float = 1e-9,
        maximum_iterations: int = 500,
        solve_method: str = "auto",
        jacobi_preconditioning: bool = True,
    ) -> None:
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0 or iterations <= 0:
            raise ValueError("Projection tolerance and maximum_iterations are invalid.")
        if solve_method not in ("auto", "iterative", "direct"):
            raise ValueError("solve_method must be 'auto', 'iterative', or 'direct'.")
        if solve_method == "direct":
            raise ValueError(
                "Variable-density direct pressure solve is unsupported here; no iterative fallback was taken."
            )
        if not isinstance(jacobi_preconditioning, bool):
            raise TypeError("jacobi_preconditioning must be a bool.")
        unit_face = tuple(
            jnp.ones(layout.shape, dtype=operators.pressure_space.dtype)
            for layout in operators.discretization.face_layouts
        )
        operator_id = canonical_fingerprint(
            {
                "kind": "mac-variable-coefficient-gauged-pressure-operator",
                "operators": operators.prepared_id,
            }
        )
        problem_id = canonical_fingerprint(
            {
                "kind": "mac-variable-density-pressure-system",
                "operator": operator_id,
            }
        )
        setup_id = canonical_fingerprint(
            {
                "kind": "mac-variable-density-pressure-diagonal",
                "operator": operator_id,
            }
        )
        problem, diagonal = _pressure_system(
            operators,
            unit_face,
            operator_id,
            problem_id,
            setup_id,
            jacobi_preconditioning=jacobi_preconditioning,
        )
        # The static relative tolerance and the per-call absolute tolerance
        # ``tolerance * sigma`` are the only acceptance threshold of the solve.
        preconditioning = (
            None
            if diagonal is None
            else PreconditioningPolicy(
                JacobiPreconditionerBuilder(),
                setup_operator=diagonal,
                side="left",
                refresh="numeric",
            )
        )
        policy = LinearSolvePolicy(
            PCG(),
            tolerance=TolerancePolicy(
                relative=tolerance_,
                absolute=0.0,
                max_steps=iterations,
            ),
            preconditioning=preconditioning,
        )
        prepared = prepare(problem, policy)
        identifier = canonical_fingerprint(
            {
                "kind": "mac-variable-density-projection-plan",
                "operators": operators.prepared_id,
                "tolerance": tolerance_,
                "linear_plan": prepared.plan.plan_id,
                "route": "pcg",
                "jacobi_preconditioning": jacobi_preconditioning,
            }
        )
        self.operators = operators
        self.tolerance = tolerance_
        self.solve_method = "pcg"
        self.jacobi_preconditioning = jacobi_preconditioning
        self.linear_policy = policy
        self.linear_problem = problem
        self.prepared_linear = prepared
        self.operator_id = operator_id
        self.pressure_problem_id = problem_id
        self.preconditioner_setup_id = setup_id
        self.plan_id = identifier

    def validate_face_inverse_density(
        self, face_inverse_density: FaceVelocity, /
    ) -> FaceVelocity:
        values = self.operators.validate_velocity(face_inverse_density)
        valid = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value) & (value > 0.0)) for value in values)
            )
        )
        return tuple(
            eqx.error_if(
                value,
                ~valid,
                "Projection face inverse density must be positive and finite.",
            )
            for value in values
        )

    def _validated_inputs(
        self,
        momentum: FaceVelocity,
        face_inverse_density: FaceVelocity,
        step_size: ArrayLike,
        pressure: ArrayLike | None,
        target_divergence: ArrayLike | None,
        /,
    ) -> tuple[FaceVelocity, FaceVelocity, Array, Array, Array]:
        values = self.operators.validate_velocity(momentum)
        inverse = self.validate_face_inverse_density(face_inverse_density)
        dtype = self.operators.pressure_space.dtype
        cell_shape = self.operators.discretization.cell_shape
        finite_momentum = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in values))
        )
        values = tuple(
            eqx.error_if(
                value,
                ~finite_momentum,
                "Projected MAC face momentum must be finite.",
            )
            for value in values
        )
        step = jnp.asarray(step_size, dtype=dtype).reshape(())
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Variable-density projection step_size must be positive and finite.",
        )
        incoming_pressure = (
            jnp.zeros(cell_shape, dtype=dtype)
            if pressure is None
            else self.operators.gauge_project(pressure)
        )
        target = (
            jnp.zeros(cell_shape, dtype=dtype)
            if target_divergence is None
            else jnp.asarray(target_divergence, dtype=dtype)
        )
        if target.shape != tuple(cell_shape):
            raise ValueError("target_divergence must match the MAC cell-pressure shape.")
        target = eqx.error_if(
            target,
            jnp.any(~jnp.isfinite(target)),
            "Target divergence must be finite.",
        )
        return values, inverse, step, incoming_pressure, target

    def _validated_many_inputs(
        self,
        momentum: FaceVelocity,
        face_inverse_density: FaceVelocity,
        step_size: ArrayLike,
        pressure: ArrayLike | None,
        target_divergence: ArrayLike | None,
        /,
    ) -> tuple[FaceVelocity, FaceVelocity, Array, Array, Array]:
        dtype = self.operators.pressure_space.dtype
        values = tuple(jnp.asarray(component, dtype=dtype) for component in momentum)
        layouts = self.operators.discretization.face_layouts
        if len(values) != len(layouts):
            raise ValueError("MAC momentum requires one normal component per axis.")
        if any(
            value.ndim != len(layout.shape) + 1 or value.shape[1:] != tuple(layout.shape)
            for value, layout in zip(values, layouts, strict=True)
        ):
            raise ValueError(
                "Batched MAC momentum must have a leading lane axis and then the face layout."
            )
        lane_count = values[0].shape[0]
        if lane_count < 1 or any(value.shape[0] != lane_count for value in values):
            raise ValueError("Batched MAC momentum lanes must be nonempty and aligned.")
        finite_momentum = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in values))
        )
        values = tuple(
            eqx.error_if(
                value,
                ~finite_momentum,
                "Projected MAC face momentum must be finite.",
            )
            for value in values
        )
        inverse = self.validate_face_inverse_density(face_inverse_density)
        step = jnp.asarray(step_size, dtype=dtype).reshape(())
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Variable-density projection step_size must be positive and finite.",
        )
        cell_shape = tuple(self.operators.discretization.cell_shape)
        lane_shape = (lane_count,) + cell_shape
        incoming_pressure = (
            jnp.zeros(lane_shape, dtype=dtype)
            if pressure is None
            else jnp.asarray(pressure, dtype=dtype)
        )
        if incoming_pressure.shape != lane_shape:
            raise ValueError(
                "Batched pressure must have a leading lane axis and then the MAC cell shape."
            )
        incoming_pressure = jax.vmap(self.operators.gauge_project)(incoming_pressure)
        target = (
            jnp.zeros(lane_shape, dtype=dtype)
            if target_divergence is None
            else jnp.asarray(target_divergence, dtype=dtype)
        )
        if target.shape != lane_shape:
            raise ValueError(
                "Batched target_divergence must have a leading lane axis and then the MAC cell shape."
            )
        target = eqx.error_if(
            target,
            jnp.any(~jnp.isfinite(target)),
            "Target divergence must be finite.",
        )
        return values, inverse, step, incoming_pressure, target

    def _refreshed_pressure_solve(
        self, face_coefficient: FaceVelocity, /
    ) -> PreparedLinearSolve:
        problem, diagonal = _pressure_system(
            self.operators,
            face_coefficient,
            self.operator_id,
            self.pressure_problem_id,
            self.preconditioner_setup_id,
            jacobi_preconditioning=self.jacobi_preconditioning,
        )
        return (
            refresh(self.prepared_linear, problem)
            if diagonal is None
            else refresh(self.prepared_linear, problem, setup_operator=diagonal)
        )

    def _solve_increment(
        self,
        face_coefficient: FaceVelocity,
        rhs: Array,
        incoming_pressure: Array,
        absolute_tolerance: Array,
        /,
    ) -> LinearSolveResult:
        return solve(
            self._refreshed_pressure_solve(face_coefficient),
            rhs,
            initial_guess=incoming_pressure,
            control=LinearSolveControl(absolute_tolerance=absolute_tolerance),
        )

    def _solve_increment_many(
        self,
        face_coefficient: FaceVelocity,
        rhs: Array,
        incoming_pressure: Array,
        divergence_scale: Array,
        /,
    ) -> LinearSolveResult:
        # Normalize every lane by its own physical flux scale. One shared
        # absolute tolerance then represents exactly
        # ``tolerance * (scale + ||rhs||)`` in every original lane.
        scale = jax.lax.stop_gradient(
            jnp.where(divergence_scale > 0.0, divergence_scale, 1.0)
        )
        leading_scale = scale.reshape((scale.shape[0],) + (1,) * (rhs.ndim - 1))
        trailing_scale = scale.reshape((1,) * (rhs.ndim - 1) + (scale.shape[0],))
        canonical = solve_many(
            self._refreshed_pressure_solve(face_coefficient),
            jnp.moveaxis(rhs / leading_scale, 0, -1),
            initial_guess=jnp.moveaxis(incoming_pressure / leading_scale, 0, -1),
            control=LinearSolveControl(
                absolute_tolerance=jnp.asarray(self.tolerance, dtype=rhs.dtype)
            ),
        )
        return eqx.tree_at(
            lambda result: (
                result.value,
                result.diagnostics.residual_norm,
                result.diagnostics.normal_residual_norm,
                result.diagnostics.compatibility_residual,
                result.diagnostics.gauge_residual,
            ),
            canonical,
            (
                canonical.value * trailing_scale,
                canonical.diagnostics.residual_norm * scale,
                canonical.diagnostics.normal_residual_norm * scale,
                canonical.diagnostics.compatibility_residual * scale,
                canonical.diagnostics.gauge_residual * scale,
            ),
        )

    def _prepare_projection(
        self,
        values: FaceVelocity,
        inverse: FaceVelocity,
        step: Array,
        incoming_pressure: Array,
        target: Array,
        /,
    ) -> _VariableDensityProjectionPreparation:
        dtype = self.operators.pressure_space.dtype
        velocity_before = tuple(
            coefficient * component
            for coefficient, component in zip(inverse, values, strict=True)
        )
        divergence_before = self.operators.divergence(velocity_before)
        volumes = self.operators.discretization.cell_volumes.astype(dtype)
        divergence_defect_before = divergence_before - target
        rhs = -self.operators.compatibility_project(divergence_defect_before)
        incompatible_defect = divergence_defect_before + rhs
        compatibility_defect = jnp.sqrt(jnp.sum(volumes * incompatible_defect**2))
        coefficient = tuple(step * value for value in inverse)
        incoming_correction = tuple(
            value * derivative
            for value, derivative in zip(
                coefficient,
                self.operators.gradient(incoming_pressure),
                strict=True,
            )
        )
        divergence_scale = (
            _flux_scale(self.operators, velocity_before)
            + _flux_scale(self.operators, incoming_correction)
            + jnp.sqrt(jnp.sum(volumes * target**2))
        )
        return _VariableDensityProjectionPreparation(
            momentum=values,
            face_inverse_density=inverse,
            step=step,
            incoming_pressure=incoming_pressure,
            target=target,
            velocity_before=velocity_before,
            divergence_before=divergence_before,
            rhs=rhs,
            incompatible_defect=incompatible_defect,
            compatibility_defect=compatibility_defect,
            coefficient=coefficient,
            divergence_scale=divergence_scale,
            absolute_tolerance=self.tolerance * divergence_scale,
        )

    def _finish_projection(
        self,
        prepared: _VariableDensityProjectionPreparation,
        linear: LinearSolveResult,
        /,
    ) -> MACVariableDensityProjectionResult:
        dtype = self.operators.pressure_space.dtype
        volumes = self.operators.discretization.cell_volumes.astype(dtype)
        values = prepared.momentum
        inverse = prepared.face_inverse_density
        step = prepared.step
        incoming_pressure = prepared.incoming_pressure
        target = prepared.target
        increment_candidate = self.operators.gauge_project(linear.value)
        gradient = self.operators.gradient(increment_candidate)
        impulse_candidate = tuple(-step * value for value in gradient)
        momentum_candidate = tuple(
            component + impulse
            for component, impulse in zip(values, impulse_candidate, strict=True)
        )
        velocity_candidate = tuple(
            inverse_value * component
            for inverse_value, component in zip(inverse, momentum_candidate, strict=True)
        )
        pressure_candidate = self.operators.gauge_project(
            incoming_pressure + increment_candidate
        )
        # The applied increment is mean-free, so its residual omits the gauge
        # row; its volume norm never exceeds the native residual.
        residual = (
            -self.operators.weighted_laplacian(increment_candidate, prepared.coefficient)
            - prepared.rhs
        )
        residual_norm = jnp.sqrt(jnp.sum(volumes * residual**2))
        divergence_candidate = self.operators.divergence(velocity_candidate)
        divergence_defect_candidate = divergence_candidate - target
        divergence_norm = jnp.sqrt(jnp.sum(volumes * divergence_defect_candidate**2))
        divergence_identity = (
            divergence_defect_candidate - prepared.incompatible_defect - residual
        )
        divergence_identity_residual = jnp.sqrt(jnp.sum(volumes * divergence_identity**2))
        gauge_defect = jnp.abs(jnp.sum(volumes * pressure_candidate))
        coefficient_minimum = jnp.min(
            jnp.stack(tuple(jnp.min(value) for value in inverse))
        )
        coefficient_maximum = jnp.max(
            jnp.stack(tuple(jnp.max(value) for value in inverse))
        )
        coefficient_contrast = coefficient_maximum / coefficient_minimum
        impulse_residual = _maximum_abs(
            tuple(
                candidate - original - impulse
                for candidate, original, impulse in zip(
                    momentum_candidate, values, impulse_candidate, strict=True
                )
            ),
            dtype,
        )
        velocity_residual = _maximum_abs(
            tuple(
                velocity_value - inverse_value * momentum_value
                for velocity_value, inverse_value, momentum_value in zip(
                    velocity_candidate, inverse, momentum_candidate, strict=True
                )
            ),
            dtype,
        )
        positive = jnp.all(jnp.stack(tuple(jnp.all(value > 0.0) for value in inverse)))
        finite = (
            jnp.all(jnp.isfinite(increment_candidate))
            & jnp.all(jnp.isfinite(divergence_candidate))
            & jnp.all(jnp.isfinite(divergence_defect_candidate))
            & jnp.all(
                jnp.stack(
                    tuple(jnp.all(jnp.isfinite(value)) for value in momentum_candidate)
                )
            )
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(divergence_identity_residual)
            & jnp.isfinite(gauge_defect)
        )
        # Each identity is exact in real arithmetic; the bound is worst-case
        # rounding of a reduction over every pressure cell.
        rounding = 10.0 * jnp.finfo(dtype).eps * self.operators.pressure_space.size
        correction_scale = _flux_scale(
            self.operators,
            tuple(
                inverse_value * impulse
                for inverse_value, impulse in zip(inverse, impulse_candidate, strict=True)
            ),
        )
        pressure_scale = jnp.sum(
            volumes * (jnp.abs(incoming_pressure) + jnp.abs(increment_candidate))
        )
        momentum_scale = _maximum_abs(values, dtype) + _maximum_abs(
            impulse_candidate, dtype
        )
        converged = (
            linear.successful
            & positive
            & finite
            & (
                prepared.compatibility_defect
                <= prepared.absolute_tolerance + rounding * prepared.divergence_scale
            )
            & (
                divergence_identity_residual
                <= rounding * (prepared.divergence_scale + correction_scale)
            )
            & (gauge_defect <= rounding * pressure_scale)
            & (impulse_residual <= rounding * momentum_scale)
            & (velocity_residual <= rounding * _maximum_abs(velocity_candidate, dtype))
        )
        momentum_value = tuple(
            jnp.where(converged, candidate, original)
            for candidate, original in zip(momentum_candidate, values, strict=True)
        )
        velocity_value = tuple(
            inverse_value * component
            for inverse_value, component in zip(inverse, momentum_value, strict=True)
        )
        pressure_value = jnp.where(converged, pressure_candidate, incoming_pressure)
        increment = jnp.where(
            converged, increment_candidate, jnp.zeros_like(increment_candidate)
        )
        impulse = tuple(
            jnp.where(converged, candidate, jnp.zeros_like(candidate))
            for candidate in impulse_candidate
        )
        divergence_after = self.operators.divergence(velocity_value)
        divergence_defect = divergence_after - target
        return MACVariableDensityProjectionResult(
            momentum=momentum_value,
            velocity=velocity_value,
            pressure=pressure_value,
            pressure_increment=increment,
            pressure_impulse=impulse,
            divergence_before=prepared.divergence_before,
            divergence_after=divergence_after,
            divergence_target=target,
            divergence_defect=divergence_defect,
            pressure_residual=residual,
            compatible_rhs=prepared.rhs,
            face_inverse_density=inverse,
            gauge_defect=gauge_defect,
            coefficient_contrast=coefficient_contrast,
            preparation_id=self.plan_id,
            residual_norm=residual_norm,
            divergence_norm=divergence_norm,
            divergence_scale=prepared.divergence_scale,
            compatibility_defect=prepared.compatibility_defect,
            divergence_identity_residual=divergence_identity_residual,
            momentum_impulse_residual=impulse_residual,
            velocity_identity_residual=velocity_residual,
            minimum_face_density=1.0 / coefficient_maximum,
            positive=positive,
            finite=finite,
            linear=linear,
            converged=converged,
            successful=converged,
            solve_method=self.solve_method,
            projection_id=self.plan_id,
        )

    def _project_many(
        self,
        momentum: FaceVelocity,
        face_inverse_density: FaceVelocity,
        step_size: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
        target_divergence: ArrayLike | None = None,
    ) -> MACVariableDensityProjectionResult:
        """Project leading lanes through one shared native multi-RHS pressure solve."""
        values, inverse, step, incoming_pressure, target = self._validated_many_inputs(
            momentum,
            face_inverse_density,
            step_size,
            pressure,
            target_divergence,
        )
        prepared = jax.vmap(
            self._prepare_projection,
            in_axes=(0, None, None, 0, 0),
        )(values, inverse, step, incoming_pressure, target)
        linear = self._solve_increment_many(
            tuple(step * value for value in inverse),
            prepared.rhs,
            incoming_pressure,
            prepared.divergence_scale,
        )
        lane_count = incoming_pressure.shape[0]
        linear = eqx.tree_at(
            lambda result: result.value,
            linear,
            jnp.moveaxis(linear.value, -1, 0),
        )

        def lane_evidence(value: Array) -> Array:
            if value.ndim == 0:
                return jnp.broadcast_to(value, (lane_count,))
            return value

        linear = jax.tree.map(lane_evidence, linear)
        return jax.vmap(self._finish_projection)(prepared, linear)

    def project(
        self,
        momentum: FaceVelocity,
        face_inverse_density: FaceVelocity,
        step_size: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
        target_divergence: ArrayLike | None = None,
    ) -> MACVariableDensityProjectionResult:
        values, inverse, step, incoming_pressure, target = self._validated_inputs(
            momentum, face_inverse_density, step_size, pressure, target_divergence
        )
        prepared = self._prepare_projection(
            values, inverse, step, incoming_pressure, target
        )
        linear = self._solve_increment(
            prepared.coefficient,
            prepared.rhs,
            incoming_pressure,
            prepared.absolute_tolerance,
        )
        return self._finish_projection(prepared, linear)

    def project_velocity_rate(
        self,
        velocity_rate: FaceVelocity,
        face_inverse_density: FaceVelocity,
        /,
        *,
        pressure: ArrayLike | None = None,
        target_divergence: ArrayLike | None = None,
    ) -> MACVariableDensityRateProjectionResult:
        inverse = self.validate_face_inverse_density(face_inverse_density)
        rate = self.operators.validate_velocity(velocity_rate)
        equivalent_momentum_rate = tuple(
            value / coefficient for value, coefficient in zip(rate, inverse, strict=True)
        )
        projected = self.project(
            equivalent_momentum_rate,
            inverse,
            1.0,
            pressure=pressure,
            target_divergence=target_divergence,
        )
        return MACVariableDensityRateProjectionResult(
            velocity_rate=projected.velocity,
            momentum_pressure_rate=projected.pressure_impulse,
            pressure=projected.pressure_increment,
            divergence_before=projected.divergence_before,
            divergence_after=projected.divergence_after,
            divergence_target=projected.divergence_target,
            divergence_defect=projected.divergence_defect,
            pressure_residual=projected.pressure_residual,
            compatible_rhs=projected.compatible_rhs,
            face_inverse_density=projected.face_inverse_density,
            gauge_defect=projected.gauge_defect,
            coefficient_contrast=projected.coefficient_contrast,
            preparation_id=projected.preparation_id,
            residual_norm=projected.residual_norm,
            divergence_norm=projected.divergence_norm,
            divergence_scale=projected.divergence_scale,
            compatibility_defect=projected.compatibility_defect,
            positive=projected.positive,
            finite=projected.finite,
            linear=projected.linear,
            converged=projected.converged,
            successful=projected.successful,
            solve_method=self.solve_method,
            projection_id=self.plan_id,
        )


__all__ = [
    "MACVariableDensityProjectionPlan",
    "MACVariableDensityProjectionResult",
    "MACVariableDensityRateProjectionResult",
]
