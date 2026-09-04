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
from ..discretization.finite_volume._incompressible import (
    FaceVelocity,
    PreparedMACOperators,
)
from ..discretization.finite_volume._mac_boundary import (
    MACBoundaryStageData,
    PreparedMACBoundaryPlan,
)
from ..discretization.finite_volume._mac_momentum import (
    PreparedMACMomentumOperators,
)
from ..discretization.finite_volume._mac_variational_viscosity import (
    FrozenMACVariationalViscosityAction,
    PreparedMACVariationalViscosityAction,
)
from ..linalg import (
    AbstractLinearOperator,
    DifferentiationPolicy,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    solve,
    TolerancePolicy,
)
from ._mac_stage_inverse_momentum import (
    MACDiagonalStageInverseMomentum,
    MACStageInverseMomentumDiagnostics,
)


class MACVariableDensityStageInverseMomentum(StrictModule, NonTrainableState):
    """Facewise dt/(a0 rho) inverse momentum for explicit-viscosity stages."""

    diagonal: MACDiagonalStageInverseMomentum
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        boundary_stage: MACBoundaryStageData,
        face_density: FaceVelocity,
        stage_coefficient: ArrayLike,
        /,
        *,
        stage_id: str,
    ):
        density = operators.validate_velocity(face_density)
        coefficient = jnp.asarray(stage_coefficient)
        inverse = tuple(
            coefficient
            / eqx.error_if(
                value,
                jnp.any(~jnp.isfinite(value) | (value <= 0.0)),
                "Face density must be positive and finite.",
            )
            for value in density
        )
        self.diagonal = MACDiagonalStageInverseMomentum(
            operators,
            boundaries,
            boundary_stage,
            inverse,
            stage_id=stage_id,
        )
        self.stage_id = str(stage_id)

    @property
    def operators(self) -> PreparedMACOperators:
        return self.diagonal.operators

    def apply_inverse(self, rhs: FaceVelocity, /) -> FaceVelocity:
        return self.diagonal.apply_inverse(rhs)

    def operator(self, /) -> FunctionLinearOperator:
        return self.diagonal.operator()

    def diagnostics(
        self, rhs: FaceVelocity, value: FaceVelocity, /
    ) -> MACStageInverseMomentumDiagnostics:
        return self.diagonal.diagnostics(rhs, value)


class MACOperatorStageSolveResult(StrictModule):
    """Affine predictor solve with its homogeneous inverse evidence."""

    value: FaceVelocity
    homogeneous_value: FaceVelocity
    effective_rhs: FaceVelocity
    boundary_affine_action: FaceVelocity
    linear: LinearSolveResult
    boundary_defect: Array
    finite: Array
    converged: Array
    stage_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)


class MACOperatorStageInverseMomentum(StrictModule, NonTrainableState):
    """Frozen SPD inverse on corrections plus one affine boundary load."""

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    boundary_stage: MACBoundaryStageData
    momentum_operator: AbstractLinearOperator
    rhs_scale: object
    boundary_affine_action: FaceVelocity
    stage_coefficient: object
    linear_policy: LinearSolvePolicy
    prepared: PreparedLinearSolve
    stage_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        boundary_stage: MACBoundaryStageData,
        momentum_operator: AbstractLinearOperator,
        /,
        *,
        rhs_scale: ArrayLike = 1.0,
        boundary_affine_action: FaceVelocity | None = None,
        stage_coefficient: ArrayLike = 0.0,
        linear_policy: LinearSolvePolicy | None = None,
        stage_id: str,
    ):
        if not momentum_operator.source.compatible(operators.velocity_space) or not (
            momentum_operator.target.compatible(operators.velocity_space)
        ):
            raise ValueError("Momentum operator must act on the MAC velocity space.")
        if not momentum_operator.properties.certifies("self_adjoint") or not (
            momentum_operator.properties.certifies("positive_definite")
        ):
            raise ValueError("Momentum operator must certify self-adjoint positivity.")
        stage = boundaries.validate_stage(boundary_stage)
        scale = jnp.asarray(rhs_scale)
        coefficient = jnp.asarray(stage_coefficient)
        if scale.shape != () or coefficient.shape != ():
            raise ValueError("Stage inverse coefficients must be scalar.")
        if boundary_affine_action is None:
            affine = tuple(
                jnp.zeros(layout.shape, dtype=operators.pressure_space.dtype)
                for layout in operators.discretization.face_layouts
            )
        else:
            affine = operators.validate_velocity(boundary_affine_action)
        policy = (
            LinearSolvePolicy(
                GMRES(restart=40),
                tolerance=TolerancePolicy(
                    relative=1.0e-9,
                    absolute=1.0e-9,
                    max_steps=500,
                ),
                differentiation=DifferentiationPolicy("mathematical"),
            )
            if linear_policy is None
            else linear_policy
        )
        identifier = str(stage_id)
        if not identifier:
            raise ValueError("stage_id must be nonempty.")
        self.operators = operators
        self.boundaries = boundaries
        self.boundary_stage = stage
        self.momentum_operator = momentum_operator
        self.rhs_scale = scale
        self.boundary_affine_action = affine
        self.stage_coefficient = coefficient
        self.prepared = prepare(
            LinearSystem(
                momentum_operator,
                problem_id=f"mac-stage-momentum/{identifier}",
            ),
            policy,
        )
        self.linear_policy = policy
        self.stage_id = identifier
        self.operator_id = canonical_fingerprint(
            {
                "kind": "mac-operator-stage-inverse",
                "stage": identifier,
                "momentum": momentum_operator.operator_id,
                "boundary": stage.stage_id,
            }
        )

    def _solve_effective(self, effective_rhs: FaceVelocity, /) -> LinearSolveResult:
        return solve(self.prepared, effective_rhs)

    def solve_homogeneous(self, rhs: FaceVelocity, /) -> LinearSolveResult:
        """Solve for a homogeneous correction, as required by pressure projection."""
        values = self.operators.validate_velocity(rhs)
        effective = self.boundaries.homogeneous_rate(
            tuple(self.rhs_scale * value for value in values)
        )
        return self._solve_effective(effective)

    def solve_affine(self, rhs: FaceVelocity, /) -> MACOperatorStageSolveResult:
        """Solve the predictor while retaining the evaluated boundary affine load."""
        values = self.operators.validate_velocity(rhs)
        effective = self.boundaries.homogeneous_rate(
            tuple(
                self.rhs_scale * value - self.stage_coefficient * affine
                for value, affine in zip(values, self.boundary_affine_action, strict=True)
            )
        )
        linear = self._solve_effective(effective)
        homogeneous = self.boundaries.homogeneous_rate(linear.value)
        value = self.boundaries.enforce(homogeneous, self.boundary_stage)
        boundary_defect = self.boundaries.defect(value, self.boundary_stage)
        finite = (
            self.boundary_stage.finite
            & jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(component))
                        for block in (value, homogeneous, effective)
                        for component in block
                    )
                )
            )
            & jnp.isfinite(boundary_defect)
        )
        converged = (
            self.boundary_stage.successful
            & linear.successful
            & finite
            & (boundary_defect <= 1.0e-8)
        )
        return MACOperatorStageSolveResult(
            value,
            homogeneous,
            effective,
            self.boundary_affine_action,
            linear,
            boundary_defect,
            finite,
            converged,
            self.stage_id,
            self.operator_id,
        )

    def apply_inverse(self, rhs: FaceVelocity, /) -> FaceVelocity:
        result = self.solve_homogeneous(rhs)
        return self.boundaries.homogeneous_rate(result.value)

    def operator(self, /) -> FunctionLinearOperator:
        return FunctionLinearOperator(
            self.apply_inverse,
            source=self.operators.velocity_space,
            target=self.operators.velocity_space,
            transpose_action=self.apply_inverse,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=self.operator_id,
        )

    def diagnostics(
        self, rhs: FaceVelocity, value: FaceVelocity, /
    ) -> MACStageInverseMomentumDiagnostics:
        rhs_ = self.operators.validate_velocity(rhs)
        value_ = self.boundaries.homogeneous_rate(self.operators.validate_velocity(value))
        effective = self.boundaries.homogeneous_rate(
            tuple(self.rhs_scale * component for component in rhs_)
        )
        reconstructed = self.momentum_operator.mv(value_)
        residual = tuple(
            left - right for left, right in zip(reconstructed, effective, strict=True)
        )
        norm = jnp.sqrt(jnp.real(self.operators.velocity_space.inner(residual, residual)))
        rhs_norm = jnp.sqrt(
            jnp.real(self.operators.velocity_space.inner(effective, effective))
        )
        finite = jnp.isfinite(norm) & jnp.isfinite(rhs_norm)
        return MACStageInverseMomentumDiagnostics(
            norm,
            norm / jnp.maximum(rhs_norm, 1.0),
            finite,
            finite & (norm <= 1.0e-8 * jnp.maximum(rhs_norm, 1.0)),
            self.stage_id,
        )


class MACVariableViscosityStagePlan(StrictModule, NonTrainableState):
    """Implicit MAC stage backed by the prepared variational viscosity action."""

    momentum: PreparedMACMomentumOperators
    viscosity_action: PreparedMACVariationalViscosityAction
    frozen_viscosity_action: FrozenMACVariationalViscosityAction
    face_density: FaceVelocity
    cell_viscosity: object
    stage_coefficient: object
    rhs_scale: object
    momentum_operator: FunctionLinearOperator
    stage_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        momentum: PreparedMACMomentumOperators,
        face_density: FaceVelocity,
        cell_viscosity: ArrayLike,
        stage_coefficient: ArrayLike,
        /,
        *,
        rhs_scale: ArrayLike = 1.0,
        viscosity_action: PreparedMACVariationalViscosityAction | None = None,
        stage_id: str,
    ):
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        operators = momentum.operators
        density = operators.validate_velocity(face_density)
        coefficient = jnp.asarray(stage_coefficient)
        rhs_scale_ = jnp.asarray(rhs_scale)
        if coefficient.shape != () or rhs_scale_.shape != ():
            raise ValueError("Variable-viscosity stage coefficients must be scalar.")
        density = tuple(
            eqx.error_if(
                value,
                jnp.any(~jnp.isfinite(value) | (value <= 0.0)),
                "Variable-viscosity face density must be positive and finite.",
            )
            for value in density
        )
        coefficient = eqx.error_if(
            coefficient,
            ~jnp.isfinite(coefficient) | (coefficient < 0.0),
            "Variable-viscosity stage coefficient must be finite and nonnegative.",
        )
        if viscosity_action is None:
            action_ = PreparedMACVariationalViscosityAction(momentum)
        else:
            if not isinstance(viscosity_action, PreparedMACVariationalViscosityAction):
                raise TypeError(
                    "viscosity_action must be PreparedMACVariationalViscosityAction "
                    "or None."
                )
            if viscosity_action.momentum.prepared_id != momentum.prepared_id:
                raise ValueError(
                    "Prepared variational viscosity action and momentum IDs differ."
                )
            action_ = viscosity_action
        frozen_viscosity = action_.freeze(
            cell_viscosity, action_.homogeneous_boundary_stage
        )
        viscosity = frozen_viscosity.cell_viscosity

        def action(values):
            values_ = operators.validate_velocity(values)
            bounded = momentum.boundaries.homogeneous_rate(values_)
            essential = tuple(
                value - homogeneous
                for value, homogeneous in zip(values_, bounded, strict=True)
            )
            diffusion = frozen_viscosity.positive_operator_action(bounded)
            homogeneous_result = momentum.boundaries.homogeneous_rate(
                tuple(
                    mass * value + coefficient * viscous
                    for mass, value, viscous in zip(
                        density, bounded, diffusion, strict=True
                    )
                )
            )
            return tuple(
                homogeneous + mass * trace
                for homogeneous, mass, trace in zip(
                    homogeneous_result, density, essential, strict=True
                )
            )

        identifier = str(stage_id)
        if not identifier:
            raise ValueError("stage_id must be nonempty.")
        operator_id = canonical_fingerprint(
            {
                "kind": "mac-variable-viscosity-momentum",
                "variational_action": action_.action_id,
                "momentum": momentum.prepared_id,
                "stage": identifier,
            }
        )
        operator = FunctionLinearOperator(
            action,
            source=operators.velocity_space,
            target=operators.velocity_space,
            transpose_action=action,
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
        self.momentum = momentum
        self.viscosity_action = action_
        self.frozen_viscosity_action = frozen_viscosity
        self.face_density = density
        self.cell_viscosity = viscosity
        self.stage_coefficient = coefficient
        self.rhs_scale = rhs_scale_
        self.momentum_operator = operator
        self.stage_id = identifier
        self.plan_id = operator_id

    def inverse(
        self,
        boundary_stage: MACBoundaryStageData,
        /,
        *,
        linear_policy: LinearSolvePolicy | None = None,
    ) -> MACOperatorStageInverseMomentum:
        stage = self.momentum.boundaries.validate_stage(boundary_stage)
        frozen_stage = self.viscosity_action.freeze(self.cell_viscosity, stage)
        return MACOperatorStageInverseMomentum(
            self.momentum.operators,
            self.momentum.boundaries,
            stage,
            self.momentum_operator,
            rhs_scale=self.rhs_scale,
            boundary_affine_action=frozen_stage.boundary_affine_action(),
            stage_coefficient=self.stage_coefficient,
            linear_policy=linear_policy,
            stage_id=self.stage_id,
        )


__all__ = [
    "MACOperatorStageInverseMomentum",
    "MACOperatorStageSolveResult",
    "MACVariableViscosityStagePlan",
    "MACVariableDensityStageInverseMomentum",
]
