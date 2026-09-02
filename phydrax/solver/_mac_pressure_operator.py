#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import (
    FaceVelocity,
    PreparedMACOperators,
)
from ..discretization.finite_volume._mac_boundary import (
    MACBoundaryPlan,
    MACBoundaryStageData,
    PreparedMACBoundaryPlan,
)
from ..linalg import (
    AbstractPreconditioner,
    DifferentiationPolicy,
    FGMRES,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    PCG,
    PreconditionerProperties,
    PreconditioningPolicy,
    prepare,
    PreparedLinearSolve,
    solve,
    TolerancePolicy,
)


MACPressureRouteRequest: TypeAlias = Literal[
    "auto", "direct", "transform", "hybrid", "iterative"
]
MACPressureRouteKind: TypeAlias = Literal["transform", "hybrid", "pcg", "fgmres"]
MACPressureCoefficientKind: TypeAlias = Literal["constant", "line", "general"]
MACPressurePreconditionerKind: TypeAlias = Literal["constant", "line", "none"]
MACPressureSideName: TypeAlias = Literal["lower", "upper"]


def _set_axis_boundary(values: Array, axis: int, index: int, data: Array, /) -> Array:
    moved = jnp.moveaxis(values, axis, 0)
    moved = moved.at[index].set(data)
    return jnp.moveaxis(moved, 0, axis)


def _face_extrema(values: FaceVelocity, /) -> tuple[Array, Array]:
    minima = jnp.stack(tuple(jnp.min(value) for value in values))
    maxima = jnp.stack(tuple(jnp.max(value) for value in values))
    return jnp.min(minima), jnp.max(maxima)


def _line_structure(values: Array, line_axis: int, tolerance: float, /) -> bool:
    moved = np.moveaxis(np.asarray(values), line_axis, 0).reshape(
        (values.shape[line_axis], -1)
    )
    reference = moved[:, :1]
    scale = max(1.0, float(np.max(np.abs(reference))))
    return bool(np.max(np.abs(moved - reference)) <= tolerance * scale)


class MACPressureRobinSide(StrictModule, NonTrainableState):
    """One static separable Robin trace alpha p + beta dp/dn = value."""

    axis: int = eqx.field(static=True)
    side: MACPressureSideName = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    value: Array
    side_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: int,
        side: MACPressureSideName,
        alpha: float,
        beta: float,
        value: ArrayLike = 0.0,
        /,
    ):
        axis_ = int(axis)
        alpha_ = float(alpha)
        beta_ = float(beta)
        if axis_ < 0 or side not in ("lower", "upper"):
            raise ValueError("Robin axis and side are invalid.")
        if (
            not np.isfinite(alpha_)
            or not np.isfinite(beta_)
            or alpha_ < 0.0
            or beta_ <= 0.0
        ):
            raise ValueError(
                "Symmetric pressure Robin data require finite alpha >= 0 and beta > 0."
            )
        data = jnp.asarray(value)
        if not bool(np.all(np.isfinite(np.asarray(data)))):
            raise ValueError("Robin boundary values must be finite.")
        self.axis = axis_
        self.side = side
        self.alpha = alpha_
        self.beta = beta_
        self.value = data
        self.side_id = canonical_fingerprint(
            {
                "kind": "mac-pressure-robin-side",
                "axis": axis_,
                "side": side,
                "alpha": alpha_,
                "beta": beta_,
                "value": array_tree_fingerprint(data),
            }
        )


class MACPressureCoefficientReport(StrictModule, NonTrainableState):
    minimum: Array
    maximum: Array
    contrast: Array
    positive: Array
    finite: Array
    structure: MACPressureCoefficientKind = eqx.field(static=True)
    line_axis: int | None = eqx.field(static=True)
    coefficient_id: str = eqx.field(static=True)


class MACPressurePreparationEvidence(StrictModule, NonTrainableState):
    coefficient: MACPressureCoefficientReport
    action_defect: Array
    lift_defect: Array
    symmetry_defect: Array
    boundary_power: Array
    jvp_defect: Array
    vjp_defect: Array
    resource_bytes: int = eqx.field(static=True)
    resource_limit_bytes: int = eqx.field(static=True)
    gauge_removed: bool = eqx.field(static=True)
    direct_eligible: bool = eqx.field(static=True)
    robin_eligible: bool = eqx.field(static=True)
    frozen_preparation: bool = eqx.field(static=True)
    certified: Array
    evidence_id: str = eqx.field(static=True)


class MACPressureExecutionEvidence(StrictModule, NonTrainableState):
    residual_norm: Array
    relative_residual: Array
    gauge_defect: Array
    boundary_power: Array
    finite: Array
    converged: Array
    preparation_id: str = eqx.field(static=True)
    coefficient_id: str = eqx.field(static=True)
    geometry_epoch: int = eqx.field(static=True)


class MACPressureSolveResult(StrictModule):
    value: Array
    candidate: Array
    residual: Array
    compatible_rhs: Array
    linear: LinearSolveResult | None
    evidence: MACPressureExecutionEvidence
    route: MACPressureRouteKind = eqx.field(static=True)
    route_reason: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.converged


class _ScaledIdentityPressurePreconditioner(AbstractPreconditioner):
    inverse_scale: Array

    def __init__(self, operators: PreparedMACOperators, scale: ArrayLike, /):
        scale_ = jnp.asarray(scale, dtype=operators.pressure_space.dtype).reshape(())
        if not bool(np.isfinite(np.asarray(scale_))) or bool(np.asarray(scale_) <= 0.0):
            raise ValueError("Pressure preconditioner scale must be finite and positive.")
        self.space = operators.pressure_space
        self.properties = PreconditionerProperties(
            linear=True,
            stationary=True,
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "linear": "construction",
                "stationary": "construction",
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        )
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "mac-constant-pressure-preconditioner",
                "operators": operators.prepared_id,
                "scale": repr(np.asarray(scale_).item()),
            }
        )
        self.inverse_scale = 1.0 / scale_

    def apply(self, residual, /, *, iteration=None):
        del iteration
        return self.space.validate(residual) * self.inverse_scale


class MACWeightedPressureAction(StrictModule, NonTrainableState):
    """Closure-aware matrix-free action A p = -D(beta G_h p)."""

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    face_coefficient: FaceVelocity
    robin_sides: tuple[MACPressureRobinSide, ...]
    gauge: bool = eqx.field(static=True)
    nonsymmetric_traction: bool = eqx.field(static=True)
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        face_coefficient: FaceVelocity,
        /,
        *,
        robin_sides: Sequence[MACPressureRobinSide] = (),
        gauge: bool,
        nonsymmetric_traction: bool = False,
        action_id: str,
    ):
        sides = tuple(robin_sides)
        dimension = len(operators.discretization.cell_shape)
        if not all(
            isinstance(side, MACPressureRobinSide) and side.axis < dimension
            for side in sides
        ):
            raise ValueError("Robin sides must address axes in the MAC tensor rank.")
        if len({(side.axis, side.side) for side in sides}) != len(sides):
            raise ValueError("At most one Robin pressure condition may own each side.")
        self.operators = operators
        self.boundaries = boundaries
        self.face_coefficient = operators.validate_velocity(face_coefficient)
        self.robin_sides = sides
        self.gauge = bool(gauge)
        self.nonsymmetric_traction = bool(nonsymmetric_traction)
        self.action_id = str(action_id)

    def _robin_gradient(
        self, pressure: Array, gradient: FaceVelocity, /, *, homogeneous: bool
    ):
        output = list(gradient)
        for condition in self.robin_sides:
            axis = condition.axis
            side_index = 0 if condition.side == "lower" else -1
            moved = jnp.moveaxis(pressure, axis, 0)
            trace = moved[side_index]
            datum = jnp.asarray(condition.value, dtype=pressure.dtype)
            if datum.shape == ():
                datum = jnp.broadcast_to(datum, trace.shape)
            if datum.shape != trace.shape:
                raise ValueError(
                    "Robin pressure data must be scalar or match its side shape."
                )
            forcing = jnp.zeros_like(datum) if homogeneous else datum
            outward = (forcing - condition.alpha * trace) / condition.beta
            coordinate_derivative = -outward if condition.side == "lower" else outward
            output[axis] = _set_axis_boundary(
                output[axis], axis, side_index, coordinate_derivative
            )
        return tuple(output)

    def gradient(
        self,
        pressure: ArrayLike,
        stage: MACBoundaryStageData | None,
        /,
        *,
        homogeneous: bool,
    ) -> FaceVelocity:
        value = self.operators.validate_pressure(pressure)
        gradient = self.boundaries.pressure_gradient(
            value, stage, homogeneous=homogeneous
        )
        return self._robin_gradient(value, gradient, homogeneous=homogeneous)

    def __call__(self, pressure: Array, /) -> Array:
        value = self.operators.validate_pressure(pressure)
        volumes = self.operators.discretization.cell_volumes.astype(value.dtype)
        mean = (
            jnp.sum(volumes * value) / jnp.sum(volumes)
            if self.gauge
            else jnp.asarray(0.0, dtype=value.dtype)
        )
        projected = value - mean
        gradient = self.gradient(projected, None, homogeneous=True)
        weighted = tuple(
            coefficient * derivative
            for coefficient, derivative in zip(
                self.face_coefficient, gradient, strict=True
            )
        )
        return -self.operators.divergence(weighted) + mean

    def lift(self, stage: MACBoundaryStageData, /) -> Array:
        zero = jnp.zeros(
            self.operators.discretization.cell_shape,
            dtype=self.operators.pressure_space.dtype,
        )
        gradient = self.gradient(zero, stage, homogeneous=False)
        homogeneous = self.gradient(zero, stage, homogeneous=True)
        affine = tuple(
            coefficient * (full - base)
            for coefficient, full, base in zip(
                self.face_coefficient, gradient, homogeneous, strict=True
            )
        )
        return self.operators.divergence(affine)


class MACPressureOperatorSpec(StrictModule, NonTrainableState):
    """Immutable coefficient, closure, route, and resource declaration."""

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    cell_coefficient: Array
    face_coefficient: FaceVelocity
    robin_sides: tuple[MACPressureRobinSide, ...]
    requested_route: MACPressureRouteRequest = eqx.field(static=True)
    route: MACPressureRouteKind = eqx.field(static=True)
    route_reason: str = eqx.field(static=True)
    coefficient_kind: MACPressureCoefficientKind = eqx.field(static=True)
    line_axis: int | None = eqx.field(static=True)
    gauge: bool = eqx.field(static=True)
    nonsymmetric_traction: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    maximum_resource_bytes: int = eqx.field(static=True)
    geometry_epoch: int = eqx.field(static=True)
    coefficient_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        coefficient: ArrayLike = 1.0,
        /,
        *,
        boundaries: PreparedMACBoundaryPlan | MACBoundaryPlan | None = None,
        solve_method: MACPressureRouteRequest = "auto",
        line_axis: int | None = None,
        robin_sides: Sequence[MACPressureRobinSide] = (),
        nonsymmetric_traction: bool = False,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
        maximum_resource_bytes: int = 512 * 1024**2,
        geometry_epoch: int = 0,
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
                "boundaries must be a prepared or unprepared MAC boundary plan."
            )
        if boundaries_.operators.prepared_id != operators.prepared_id:
            raise ValueError(
                "Pressure boundaries and operators must share one preparation."
            )
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        resource_limit = int(maximum_resource_bytes)
        epoch = int(geometry_epoch)
        if (
            not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or iterations <= 0
            or resource_limit <= 0
            or epoch < 0
        ):
            raise ValueError(
                "Pressure tolerance, iterations, resources, or epoch are invalid."
            )
        if solve_method not in ("auto", "direct", "transform", "hybrid", "iterative"):
            raise ValueError("Unknown MAC pressure route request.")
        axis = None if line_axis is None else int(line_axis)
        dimension = len(operators.discretization.cell_shape)
        if axis is not None and (axis < 0 or axis >= dimension):
            raise ValueError("line_axis is outside the MAC pressure tensor rank.")
        value = jnp.asarray(coefficient, dtype=operators.pressure_space.dtype)
        if value.shape == ():
            cell = jnp.full(operators.discretization.cell_shape, value, dtype=value.dtype)
        else:
            cell = operators.validate_pressure(value)
        if not bool(np.all(np.isfinite(np.asarray(cell)))) or bool(
            np.any(np.asarray(cell) <= 0.0)
        ):
            raise ValueError("MAC pressure coefficient beta must be finite and positive.")
        face = operators.interpolate_inverse_momentum(cell)
        minimum, maximum = _face_extrema(face)
        scale = max(1.0, float(np.asarray(maximum)))
        constant = bool(float(np.asarray(maximum - minimum)) <= tolerance_ * scale)
        line = axis is not None and _line_structure(cell, axis, tolerance_)
        structure: MACPressureCoefficientKind = (
            "constant" if constant else "line" if line else "general"
        )
        sides = tuple(robin_sides)
        if not all(isinstance(side, MACPressureRobinSide) for side in sides):
            raise TypeError("robin_sides must contain MACPressureRobinSide values.")
        has_robin_anchor = any(side.alpha > 0.0 for side in sides)
        gauge = boundaries_.closure_kind == "neumann" and not has_robin_anchor
        stabilized_traction = bool(nonsymmetric_traction) or any(
            boundary.kind == "traction-open" and boundary.backflow_coefficient > 0.0
            for boundary in boundaries_.sides
        )
        symmetric = not stabilized_traction
        transform_eligible = (
            structure == "constant"
            and operators.report.transform_eligible
            and not sides
            and symmetric
        )
        hybrid_eligible = (
            structure in ("constant", "line")
            and axis is not None
            and dimension == 3
            and boundaries_.closure_kind == "neumann"
            and not sides
            and symmetric
            and not operators.discretization.grid.structured_axes[axis].periodic
        )
        if solve_method == "transform":
            if not transform_eligible:
                raise ValueError(
                    "Explicit transform pressure solve is unsupported for this coefficient/action."
                )
            route: MACPressureRouteKind = "transform"
            reason = "explicit certified constant-coefficient tensor transform"
        elif solve_method == "hybrid":
            if not hybrid_eligible:
                raise ValueError(
                    "Explicit hybrid pressure solve is unsupported for this coefficient/action."
                )
            route = "hybrid"
            reason = "explicit certified transform-line coefficient action"
        elif solve_method == "direct":
            if transform_eligible:
                route = "transform"
                reason = "explicit direct request accepted by exact tensor action"
            elif hybrid_eligible:
                route = "hybrid"
                reason = "explicit direct request accepted by exact transform-line action"
            else:
                raise ValueError(
                    "Explicit direct pressure solve has no certified exact representation."
                )
        elif solve_method == "iterative":
            route = "fgmres" if not symmetric else "pcg"
            reason = (
                "explicit flexible iteration for stabilized nonsymmetric traction"
                if not symmetric
                else "explicit PCG with certified constant preconditioner"
            )
        elif transform_eligible:
            route = "transform"
            reason = "auto selected exact constant-coefficient tensor action"
        elif hybrid_eligible:
            route = "hybrid"
            reason = "auto selected exact coefficient-along-line action"
        else:
            route = "fgmres" if not symmetric else "pcg"
            reason = (
                "auto selected FGMRES because stabilized traction is nonsymmetric"
                if not symmetric
                else "auto selected PCG because beta is not exactly separable"
            )
        coefficient_id = array_tree_fingerprint(face)
        operator_id = canonical_fingerprint(
            {
                "kind": "mac-weighted-pressure-operator",
                "operators": operators.prepared_id,
                "boundaries": boundaries_.prepared_id,
                "coefficient": coefficient_id,
                "robin": [side.side_id for side in sides],
                "gauge": gauge,
                "nonsymmetric_traction": stabilized_traction,
            }
        )
        self.operators = operators
        self.boundaries = boundaries_
        self.cell_coefficient = cell
        self.face_coefficient = face
        self.robin_sides = sides
        self.requested_route = solve_method
        self.route = route
        self.route_reason = reason
        self.coefficient_kind = structure
        self.line_axis = axis
        self.gauge = gauge
        self.nonsymmetric_traction = stabilized_traction
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.maximum_resource_bytes = resource_limit
        self.geometry_epoch = epoch
        self.coefficient_id = coefficient_id
        self.operator_id = operator_id
        self.spec_id = canonical_fingerprint(
            {
                "kind": "mac-pressure-operator-spec",
                "operator": operator_id,
                "requested_route": solve_method,
                "route": route,
                "route_reason": reason,
                "line_axis": axis,
                "tolerance": tolerance_,
                "maximum_iterations": iterations,
                "maximum_resource_bytes": resource_limit,
                "geometry_epoch": epoch,
            }
        )

    def prepare(self, /) -> "PreparedMACPressureOperator":
        return PreparedMACPressureOperator(self)


class PreparedMACPressureOperator(StrictModule, NonTrainableState):
    """Frozen pressure action and selected exact or Krylov execution route."""

    spec: MACPressureOperatorSpec
    action: MACWeightedPressureAction
    linear_problem: LinearSystem | None
    prepared_linear: PreparedLinearSolve | None
    preparation: MACPressurePreparationEvidence
    preparation_id: str = eqx.field(static=True)

    def __init__(self, spec: MACPressureOperatorSpec, /):
        if not isinstance(spec, MACPressureOperatorSpec):
            raise TypeError("spec must be MACPressureOperatorSpec.")
        action = MACWeightedPressureAction(
            spec.operators,
            spec.boundaries,
            spec.face_coefficient,
            robin_sides=spec.robin_sides,
            gauge=spec.gauge,
            nonsymmetric_traction=spec.nonsymmetric_traction,
            action_id=spec.operator_id,
        )
        shape = spec.operators.discretization.cell_shape
        dtype = spec.operators.pressure_space.dtype
        count = int(np.prod(shape))
        workspace_vectors = spec.maximum_iterations + 8 if spec.route == "fgmres" else 9
        resource_bytes = workspace_vectors * count * np.dtype(dtype).itemsize
        if resource_bytes > spec.maximum_resource_bytes:
            raise ValueError(
                "MAC pressure route workspace exceeds maximum_resource_bytes."
            )
        probe = jnp.sin(0.37 * jnp.arange(count, dtype=dtype).reshape(shape) + 0.2)
        second = jnp.cos(0.19 * jnp.arange(count, dtype=dtype).reshape(shape) + 0.4)
        image = action(probe)
        action_defect = jnp.max(jnp.abs(action(probe + second) - image - action(second)))
        volumes = spec.operators.discretization.cell_volumes.astype(dtype)
        symmetry_left = jnp.sum(volumes * probe * action(second))
        symmetry_right = jnp.sum(volumes * image * second)
        symmetry_defect = jnp.abs(symmetry_left - symmetry_right)
        tangent = jnp.cos(0.11 * jnp.arange(count, dtype=dtype).reshape(shape) + 0.7)
        _, tangent_image = jax.jvp(action, (probe,), (tangent,))
        jvp_defect = jnp.max(jnp.abs(tangent_image - action(tangent)))
        cotangent = jnp.sin(0.23 * jnp.arange(count, dtype=dtype).reshape(shape) + 0.1)
        _, pullback = jax.vjp(action, probe)
        transpose_image = pullback(cotangent)[0]
        vjp_defect = jnp.abs(
            jnp.vdot(cotangent, image) - jnp.vdot(transpose_image, probe)
        )
        stage = spec.boundaries.evaluate(jnp.asarray(0.0, dtype=dtype), None)
        lift = action.lift(stage)
        lift_defect = jnp.max(jnp.abs(lift - action.lift(stage)))
        boundary_power = jnp.real(jnp.sum(volumes * probe * image))
        minimum, maximum = _face_extrema(spec.face_coefficient)
        coefficient = MACPressureCoefficientReport(
            minimum=minimum,
            maximum=maximum,
            contrast=maximum / minimum,
            positive=minimum > 0.0,
            finite=jnp.isfinite(minimum) & jnp.isfinite(maximum),
            structure=spec.coefficient_kind,
            line_axis=spec.line_axis,
            coefficient_id=spec.coefficient_id,
        )
        epsilon = jnp.finfo(jnp.real(probe).dtype).eps
        scale = jnp.maximum(1.0, jnp.linalg.norm(image.reshape((-1,))))
        certified = (
            coefficient.positive
            & coefficient.finite
            & jnp.all(jnp.isfinite(image))
            & jnp.isfinite(action_defect)
            & jnp.isfinite(symmetry_defect)
            & jnp.isfinite(jvp_defect)
            & jnp.isfinite(vjp_defect)
            & (
                action_defect
                <= jnp.maximum(100.0 * spec.tolerance, 4096.0 * epsilon * scale)
            )
            & (
                jvp_defect
                <= jnp.maximum(100.0 * spec.tolerance, 4096.0 * epsilon * scale)
            )
            & (
                jnp.asarray(True)
                if spec.nonsymmetric_traction
                else symmetry_defect
                <= jnp.maximum(100.0 * spec.tolerance, 4096.0 * epsilon * scale)
            )
        )
        if not bool(np.asarray(certified)):
            raise RuntimeError(
                "Prepared MAC pressure operator failed action certification."
            )
        preparation_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-pressure-operator",
                "spec": spec.spec_id,
                "coefficient": spec.coefficient_id,
                "geometry_epoch": spec.geometry_epoch,
                "route": spec.route,
            }
        )
        direct = spec.route in ("transform", "hybrid")
        evidence = MACPressurePreparationEvidence(
            coefficient=coefficient,
            action_defect=action_defect,
            lift_defect=lift_defect,
            symmetry_defect=symmetry_defect,
            boundary_power=boundary_power,
            jvp_defect=jvp_defect,
            vjp_defect=vjp_defect,
            resource_bytes=resource_bytes,
            resource_limit_bytes=spec.maximum_resource_bytes,
            gauge_removed=not spec.gauge,
            direct_eligible=direct,
            robin_eligible=all(
                side.alpha >= 0.0 and side.beta > 0.0 for side in spec.robin_sides
            ),
            frozen_preparation=True,
            certified=certified,
            evidence_id=canonical_fingerprint(
                {
                    "kind": "mac-pressure-preparation-evidence",
                    "preparation": preparation_id,
                }
            ),
        )
        if direct:
            problem = None
            prepared = None
        else:
            properties = OperatorProperties(
                self_adjoint=not spec.nonsymmetric_traction,
                positive_definite=not spec.nonsymmetric_traction,
                evidence=(
                    {}
                    if spec.nonsymmetric_traction
                    else {
                        "self_adjoint": "verified",
                        "positive_definite": "construction",
                    }
                ),
            )
            operator = FunctionLinearOperator(
                action,
                source=spec.operators.pressure_space,
                target=spec.operators.pressure_space,
                transpose_action=None if spec.nonsymmetric_traction else action,
                properties=properties,
                operator_id=spec.operator_id,
            )
            problem = LinearSystem(
                operator,
                problem_id=canonical_fingerprint(
                    {"kind": "mac-pressure-linear-system", "operator": spec.operator_id}
                ),
            )
            preconditioning = None
            if not spec.nonsymmetric_traction:
                reference = jnp.sqrt(minimum * maximum)
                preconditioning = PreconditioningPolicy(
                    _ScaledIdentityPressurePreconditioner(spec.operators, reference),
                    side="left",
                    refresh="frozen",
                )
            policy = LinearSolvePolicy(
                FGMRES(restart=min(30, spec.maximum_iterations))
                if spec.nonsymmetric_traction
                else PCG(),
                tolerance=TolerancePolicy(
                    relative=spec.tolerance,
                    absolute=spec.tolerance,
                    max_steps=spec.maximum_iterations,
                ),
                preconditioning=preconditioning,
                differentiation=DifferentiationPolicy("mathematical"),
            )
            prepared = prepare(problem, policy)
        self.spec = spec
        self.action = action
        self.linear_problem = problem
        self.prepared_linear = prepared
        self.preparation = evidence
        self.preparation_id = preparation_id

    def validate_frozen(
        self,
        coefficient: ArrayLike,
        /,
        *,
        geometry_epoch: int | None = None,
    ) -> Array:
        value = jnp.asarray(coefficient, dtype=self.spec.operators.pressure_space.dtype)
        cell = (
            jnp.full(
                self.spec.operators.discretization.cell_shape, value, dtype=value.dtype
            )
            if value.shape == ()
            else self.spec.operators.validate_pressure(value)
        )
        face = self.spec.operators.interpolate_inverse_momentum(cell)
        if array_tree_fingerprint(face) != self.spec.coefficient_id:
            raise ValueError(
                "Pressure coefficient differs from the frozen prepared coefficient; prepare again."
            )
        epoch = (
            self.spec.geometry_epoch if geometry_epoch is None else int(geometry_epoch)
        )
        if epoch != self.spec.geometry_epoch:
            raise ValueError(
                "Pressure geometry epoch differs from the frozen preparation; refresh it."
            )
        return cell

    def lifted_rhs(
        self,
        right_hand_side: ArrayLike,
        /,
        *,
        boundary_stage: MACBoundaryStageData | None = None,
    ) -> Array:
        rhs = self.spec.operators.validate_pressure(right_hand_side)
        stage = (
            self.spec.boundaries.evaluate(jnp.asarray(0.0, dtype=rhs.dtype), None)
            if boundary_stage is None
            else self.spec.boundaries.validate_stage(boundary_stage)
        )
        lifted = rhs + self.action.lift(stage)
        return (
            self.spec.operators.compatibility_project(lifted)
            if self.spec.gauge
            else lifted
        )

    def solve(
        self,
        right_hand_side: ArrayLike,
        /,
        *,
        initial_guess: ArrayLike | None = None,
        boundary_stage: MACBoundaryStageData | None = None,
        direct_solve: Callable[[Array], Array] | None = None,
    ) -> MACPressureSolveResult:
        rhs = self.lifted_rhs(right_hand_side, boundary_stage=boundary_stage)
        initial = (
            jnp.zeros_like(rhs)
            if initial_guess is None
            else self.spec.operators.validate_pressure(initial_guess)
        )
        if self.spec.gauge:
            initial = self.spec.operators.gauge_project(initial)
        if self.spec.route in ("transform", "hybrid"):
            if direct_solve is None:
                raise ValueError(
                    "The certified direct route requires its prepared transform solve action."
                )
            candidate = self.spec.operators.validate_pressure(direct_solve(rhs))
            linear = None
            solver_success = jnp.asarray(True)
        else:
            if self.prepared_linear is None:
                raise RuntimeError(
                    "Iterative pressure route has no prepared linear solve."
                )
            linear = solve(self.prepared_linear, rhs, initial_guess=initial)
            candidate = linear.value
            solver_success = linear.successful
        value = (
            self.spec.operators.gauge_project(candidate) if self.spec.gauge else candidate
        )
        residual = self.action(value) - rhs
        volumes = self.spec.operators.discretization.cell_volumes.astype(value.dtype)
        residual_norm = jnp.sqrt(jnp.sum(volumes * residual**2))
        rhs_norm = jnp.sqrt(jnp.sum(volumes * rhs**2))
        relative = residual_norm / jnp.maximum(rhs_norm, 1.0)
        gauge_defect = (
            jnp.abs(jnp.sum(volumes * value))
            if self.spec.gauge
            else jnp.asarray(0.0, dtype=value.dtype)
        )
        boundary_power = jnp.real(jnp.sum(volumes * value * self.action(value)))
        finite = (
            jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(boundary_power)
        )
        converged = (
            solver_success
            & finite
            & (relative <= self.spec.tolerance)
            & (gauge_defect <= self.spec.tolerance)
        )
        evidence = MACPressureExecutionEvidence(
            residual_norm=residual_norm,
            relative_residual=relative,
            gauge_defect=gauge_defect,
            boundary_power=boundary_power,
            finite=finite,
            converged=converged,
            preparation_id=self.preparation_id,
            coefficient_id=self.spec.coefficient_id,
            geometry_epoch=self.spec.geometry_epoch,
        )
        return MACPressureSolveResult(
            value=jnp.where(converged, value, initial),
            candidate=value,
            residual=residual,
            compatible_rhs=rhs,
            linear=linear,
            evidence=evidence,
            route=self.spec.route,
            route_reason=self.spec.route_reason,
            preparation_id=self.preparation_id,
        )


class MACWeightedPressureIterationResult(StrictModule):
    pressure: Array
    residual: Array
    compatible_rhs: Array
    residual_norm: Array
    relative_residual: Array
    coefficient_contrast: Array
    action_defect: Array
    symmetry_defect: Array
    boundary_power: Array
    gauge_defect: Array
    gcl_residual: Array
    metric_residual: Array
    finite: Array
    converged: Array
    route: str = eqx.field(static=True)
    route_reason: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    geometry_epoch: int = eqx.field(static=True)
    prepared_geometry_epoch: int = eqx.field(static=True)
    preconditioner_refreshed: bool = eqx.field(static=True)


def execute_weighted_pressure_iteration(
    geometry,
    right_hand_side: ArrayLike,
    face_coefficient: FaceVelocity,
    initial_guess: ArrayLike,
    tolerance: float,
    maximum_iterations: int,
    /,
    *,
    geometry_id: str,
    geometry_epoch: int,
    prepared_geometry_epoch: int,
    gcl_residual: ArrayLike = 0.0,
    metric_residual: ArrayLike = 0.0,
) -> MACWeightedPressureIterationResult:
    """Shared mapped/ALE matrix-free PCG with epoch-bound preconditioning."""

    tolerance_ = float(tolerance)
    steps = int(maximum_iterations)
    epoch = int(geometry_epoch)
    prepared_epoch = int(prepared_geometry_epoch)
    identity = str(geometry_id)
    if (
        not identity
        or not np.isfinite(tolerance_)
        or tolerance_ <= 0.0
        or steps <= 0
        or epoch < 0
        or prepared_epoch < 0
    ):
        raise ValueError(
            "Mapped pressure iteration policy or geometry identity is invalid."
        )
    if epoch != prepared_epoch:
        raise ValueError(
            "Mapped pressure preconditioner belongs to another geometry epoch; refresh it."
        )
    coefficient = geometry.validate_velocity(face_coefficient)
    minima = jnp.stack(tuple(jnp.min(value) for value in coefficient))
    maxima = jnp.stack(tuple(jnp.max(value) for value in coefficient))
    minimum = jnp.min(minima)
    maximum = jnp.max(maxima)
    coefficient_valid = jnp.isfinite(minimum) & jnp.isfinite(maximum) & (minimum > 0.0)
    coefficient = tuple(
        eqx.error_if(
            value,
            ~coefficient_valid,
            "Mapped pressure beta must be finite and positive.",
        )
        for value in coefficient
    )
    rhs = geometry.compatibility_project(right_hand_side)
    pressure = geometry.gauge_project(initial_guess)
    volumes = geometry.cell_volumes.astype(pressure.dtype)

    def action(value):
        return geometry.pressure_action(value, coefficient)

    residual = rhs - action(pressure)
    preconditioner_scale = jnp.sqrt(minimum * maximum)
    preconditioned = residual / preconditioner_scale
    direction = preconditioned
    pairing = jnp.sum(volumes * residual * preconditioned)
    rhs_norm_squared = jnp.sum(volumes * rhs * rhs)
    threshold = tolerance_**2 * jnp.maximum(rhs_norm_squared, 1.0)
    active = jnp.sum(volumes * residual * residual) > threshold
    failed = jnp.asarray(False)

    def body(_, state):
        value, residual_, direction_, pairing_, active_, failed_ = state
        image = action(direction_)
        denominator = jnp.sum(volumes * direction_ * image)
        valid = active_ & jnp.isfinite(denominator) & (denominator > 0.0)
        alpha = jnp.where(valid, pairing_ / denominator, 0.0)
        next_value = value + alpha * direction_
        next_residual = residual_ - alpha * image
        next_preconditioned = next_residual / preconditioner_scale
        next_pairing = jnp.sum(volumes * next_residual * next_preconditioned)
        residual_norm_squared = jnp.sum(volumes * next_residual * next_residual)
        running = valid & (residual_norm_squared > threshold)
        beta = jnp.where(running & (pairing_ > 0.0), next_pairing / pairing_, 0.0)
        return (
            next_value,
            next_residual,
            next_preconditioned + beta * direction_,
            next_pairing,
            running,
            failed_ | (active_ & ~valid),
        )

    pressure, _, _, _, active, failed = jax.lax.fori_loop(
        0,
        steps,
        body,
        (pressure, residual, direction, pairing, active, failed),
    )
    pressure = geometry.gauge_project(pressure)
    residual = action(pressure) - rhs
    residual_norm = jnp.sqrt(jnp.sum(volumes * residual * residual))
    rhs_norm = jnp.sqrt(rhs_norm_squared)
    relative = residual_norm / jnp.maximum(rhs_norm, 1.0)
    gauge_defect = jnp.abs(jnp.sum(volumes * pressure))
    count = pressure.size
    probe = jnp.sin(
        0.37 * jnp.arange(count, dtype=pressure.dtype).reshape(pressure.shape) + 0.2
    )
    second = jnp.cos(
        0.19 * jnp.arange(count, dtype=pressure.dtype).reshape(pressure.shape) + 0.4
    )
    action_defect = jnp.max(
        jnp.abs(action(probe + second) - action(probe) - action(second))
    )
    symmetry_defect = jnp.abs(
        jnp.sum(volumes * probe * action(second))
        - jnp.sum(volumes * action(probe) * second)
    )
    boundary_power = jnp.real(jnp.sum(volumes * pressure * action(pressure)))
    gcl = jnp.asarray(gcl_residual, dtype=pressure.dtype).reshape(())
    metric = jnp.asarray(metric_residual, dtype=pressure.dtype).reshape(())
    finite = (
        coefficient_valid
        & jnp.all(jnp.isfinite(pressure))
        & jnp.all(jnp.isfinite(residual))
        & jnp.isfinite(action_defect)
        & jnp.isfinite(symmetry_defect)
        & jnp.isfinite(boundary_power)
        & jnp.isfinite(gcl)
        & jnp.isfinite(metric)
    )
    converged = (
        ~active
        & ~failed
        & finite
        & (relative <= tolerance_)
        & (gauge_defect <= tolerance_)
    )
    return MACWeightedPressureIterationResult(
        pressure=pressure,
        residual=residual,
        compatible_rhs=rhs,
        residual_norm=residual_norm,
        relative_residual=relative,
        coefficient_contrast=maximum / minimum,
        action_defect=action_defect,
        symmetry_defect=symmetry_defect,
        boundary_power=boundary_power,
        gauge_defect=gauge_defect,
        gcl_residual=gcl,
        metric_residual=metric,
        finite=finite,
        converged=converged,
        route="pcg",
        route_reason=(
            "mapped/ALE matrix-free weighted action with epoch-refreshed "
            "constant preconditioner"
        ),
        geometry_id=identity,
        geometry_epoch=epoch,
        prepared_geometry_epoch=prepared_epoch,
        preconditioner_refreshed=True,
    )


__all__ = [
    "MACPressureCoefficientKind",
    "MACPressureCoefficientReport",
    "MACPressureExecutionEvidence",
    "MACPressureOperatorSpec",
    "MACPressurePreparationEvidence",
    "MACPressurePreconditionerKind",
    "MACPressureRobinSide",
    "MACPressureRouteKind",
    "MACPressureRouteRequest",
    "MACPressureSolveResult",
    "MACWeightedPressureIterationResult",
    "execute_weighted_pressure_iteration",
    "MACWeightedPressureAction",
    "PreparedMACPressureOperator",
]
