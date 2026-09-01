#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

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
from ..linalg import (
    AbstractLinearOperator,
    DifferentiationPolicy,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
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


class MACOperatorStageInverseMomentum(StrictModule, NonTrainableState):
    """Implicit inverse of a supplied SPD momentum operator on MAC velocity space."""

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    boundary_stage: MACBoundaryStageData
    momentum_operator: AbstractLinearOperator
    rhs_scale: object
    linear_policy: LinearSolvePolicy
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        boundary_stage: MACBoundaryStageData,
        momentum_operator: AbstractLinearOperator,
        /,
        *,
        rhs_scale: ArrayLike = 1.0,
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
        scale = jnp.asarray(rhs_scale)
        if scale.shape != ():
            raise ValueError("rhs_scale must be scalar.")
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
        self.operators = operators
        self.boundaries = boundaries
        self.boundary_stage = boundary_stage
        self.momentum_operator = momentum_operator
        self.rhs_scale = scale
        self.linear_policy = policy
        self.stage_id = str(stage_id)

    def apply_inverse(self, rhs: FaceVelocity, /) -> FaceVelocity:
        values = self.operators.validate_velocity(rhs)
        scaled = tuple(self.rhs_scale * value for value in values)
        result = solve(
            LinearSystem(
                self.momentum_operator,
                problem_id=f"mac-stage-momentum/{self.stage_id}",
            ),
            scaled,
            policy=self.linear_policy,
            initial_guess=tuple(jnp.zeros_like(value) for value in scaled),
        )
        flattened = self.operators.velocity_space.flatten(result.value)
        checked = eqx.error_if(
            flattened,
            ~result.successful,
            "Implicit MAC momentum inverse failed.",
        )
        return tuple(self.operators.velocity_space.unflatten(checked))

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
            operator_id=canonical_fingerprint(
                {
                    "kind": "mac-operator-stage-inverse",
                    "stage": self.stage_id,
                    "momentum": self.momentum_operator.operator_id,
                }
            ),
        )

    def diagnostics(
        self, rhs: FaceVelocity, value: FaceVelocity, /
    ) -> MACStageInverseMomentumDiagnostics:
        rhs_ = self.operators.validate_velocity(rhs)
        value_ = self.operators.validate_velocity(value)
        reconstructed = self.momentum_operator.mv(value_)
        residual = tuple(
            left - self.rhs_scale * right
            for left, right in zip(reconstructed, rhs_, strict=True)
        )
        norm = jnp.sqrt(jnp.real(self.operators.velocity_space.inner(residual, residual)))
        rhs_norm = jnp.sqrt(jnp.real(self.operators.velocity_space.inner(rhs_, rhs_)))
        finite = jnp.isfinite(norm) & jnp.isfinite(rhs_norm)
        return MACStageInverseMomentumDiagnostics(
            norm,
            norm / jnp.maximum(rhs_norm, 1.0),
            finite,
            finite & (norm <= 1.0e-8 * jnp.maximum(rhs_norm, 1.0)),
            self.stage_id,
        )


class MACVariableViscosityStagePlan(StrictModule, NonTrainableState):
    """SPD variable-viscosity MAC momentum action from a discrete strain energy."""

    momentum: PreparedMACMomentumOperators
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
        stage_id: str,
    ):
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        operators = momentum.operators
        density = operators.validate_velocity(face_density)
        viscosity = operators.validate_pressure(cell_viscosity)
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
        viscosity = eqx.error_if(
            viscosity,
            jnp.any(~jnp.isfinite(viscosity) | (viscosity <= 0.0)),
            "Cell viscosity must be positive and finite.",
        )
        coefficient = eqx.error_if(
            coefficient,
            ~jnp.isfinite(coefficient) | (coefficient < 0.0),
            "Variable-viscosity stage coefficient must be finite and nonnegative.",
        )
        grid = operators.discretization.grid
        volumes = operators.discretization.cell_volumes.astype(viscosity.dtype)

        def cell_velocity(values):
            components = []
            for component, axis in enumerate(grid.structured_axes):
                moved = jnp.moveaxis(values[component], component, 0)
                if axis.periodic:
                    averaged = 0.5 * (moved + jnp.roll(moved, -1, axis=0))
                else:
                    averaged = 0.5 * (moved[:-1] + moved[1:])
                components.append(jnp.moveaxis(averaged, 0, component))
            return jnp.stack(tuple(components), axis=-1)

        def derivative(value, axis_index):
            axis = grid.structured_axes[axis_index]
            centers = axis.interval_centers.astype(value.dtype)
            moved = jnp.moveaxis(value, axis_index, 0)
            if moved.shape[0] == 1:
                result = jnp.zeros_like(moved)
            elif axis.periodic:
                period = axis.bounds[1] - axis.bounds[0]
                forward = jnp.roll(centers, -1).at[-1].add(period)
                backward = jnp.roll(centers, 1).at[0].add(-period)
                denominator = (forward - backward).reshape(
                    (-1,) + (1,) * (moved.ndim - 1)
                )
                result = (
                    jnp.roll(moved, -1, axis=0) - jnp.roll(moved, 1, axis=0)
                ) / denominator
            else:
                lower = (moved[1] - moved[0]) / (centers[1] - centers[0])
                upper = (moved[-1] - moved[-2]) / (centers[-1] - centers[-2])
                if moved.shape[0] == 2:
                    result = jnp.stack((lower, upper), axis=0)
                else:
                    denominator = (centers[2:] - centers[:-2]).reshape(
                        (-1,) + (1,) * (moved.ndim - 1)
                    )
                    interior = (moved[2:] - moved[:-2]) / denominator
                    result = jnp.concatenate((lower[None], interior, upper[None]), axis=0)
            return jnp.moveaxis(result, 0, axis_index)

        def viscous_energy(values):
            bounded = momentum.boundaries.homogeneous_rate(values)
            cells = cell_velocity(bounded)
            gradient = jnp.stack(
                tuple(
                    derivative(cells, axis) for axis in range(len(grid.structured_axes))
                ),
                axis=-1,
            )
            strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
            return 0.5 * jnp.sum(volumes * viscosity * jnp.sum(strain**2, axis=(-2, -1)))

        energy_gradient = jax.grad(viscous_energy)

        def action(values):
            bounded = momentum.boundaries.homogeneous_rate(
                operators.validate_velocity(values)
            )
            viscous_covector = energy_gradient(bounded)
            viscous_vector = operators.velocity_space.inverse_riesz(viscous_covector)
            result = tuple(
                mass * value + coefficient * diffusion
                for mass, value, diffusion in zip(
                    density, bounded, viscous_vector, strict=True
                )
            )
            return momentum.boundaries.homogeneous_rate(result)

        identifier = str(stage_id)
        if not identifier:
            raise ValueError("stage_id must be nonempty.")
        operator_id = canonical_fingerprint(
            {
                "kind": "mac-variable-viscosity-momentum",
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
        return MACOperatorStageInverseMomentum(
            self.momentum.operators,
            self.momentum.boundaries,
            boundary_stage,
            self.momentum_operator,
            rhs_scale=self.rhs_scale,
            linear_policy=linear_policy,
            stage_id=self.stage_id,
        )


__all__ = [
    "MACOperatorStageInverseMomentum",
    "MACVariableViscosityStagePlan",
    "MACVariableDensityStageInverseMomentum",
]
