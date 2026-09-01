#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TypeAlias

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
from ..linalg import FunctionLinearOperator, OperatorProperties
from ._mac_viscous import MACHelmholtzResult, MACHelmholtzSolvePlan


class MACStageInverseMomentumDiagnostics(StrictModule):
    residual_norm: Array
    relative_residual: Array
    finite: Array
    converged: Array
    stage_id: str = eqx.field(static=True)


class MACDiagonalStageInverseMomentum(StrictModule, NonTrainableState):
    """Positive face-diagonal inverse momentum on the homogeneous correction space."""

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    boundary_stage: MACBoundaryStageData
    inverse_diagonal: FaceVelocity
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        boundary_stage: MACBoundaryStageData,
        inverse_diagonal: FaceVelocity | ArrayLike,
        /,
        *,
        stage_id: str | None = None,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        if isinstance(inverse_diagonal, (tuple, list)):
            values = operators.validate_velocity(tuple(inverse_diagonal))
        else:
            array = jnp.asarray(inverse_diagonal)
            values = (
                tuple(
                    jnp.full(
                        layout.shape,
                        array,
                        dtype=operators.pressure_space.dtype,
                    )
                    for layout in operators.discretization.face_layouts
                )
                if array.shape == ()
                else operators.validate_velocity(inverse_diagonal)
            )
        values = tuple(
            eqx.error_if(
                value,
                jnp.any(~jnp.isfinite(value) | (value <= 0.0)),
                "Inverse momentum diagonal must be positive and finite.",
            )
            for value in values
        )
        self.operators = operators
        self.boundaries = boundaries
        self.boundary_stage = boundary_stage
        self.inverse_diagonal = values
        self.stage_id = (
            canonical_fingerprint(
                {
                    "kind": "mac-diagonal-stage-inverse-momentum",
                    "operators": operators.prepared_id,
                    "boundary_stage": boundary_stage.stage_id,
                }
            )
            if stage_id is None
            else str(stage_id)
        )

    def apply_inverse(self, rhs: FaceVelocity, /) -> FaceVelocity:
        values = self.operators.validate_velocity(rhs)
        return self.boundaries.homogeneous_rate(
            tuple(
                inverse * value
                for inverse, value in zip(self.inverse_diagonal, values, strict=True)
            )
        )

    def operator(self, /) -> FunctionLinearOperator:
        return FunctionLinearOperator(
            self.apply_inverse,
            source=self.operators.velocity_space,
            target=self.operators.velocity_space,
            transpose_action=self.apply_inverse,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                diagonal=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                    "diagonal": "construction",
                },
            ),
            operator_id=f"mac-stage-inverse/{self.stage_id}",
        )

    def diagnostics(
        self, rhs: FaceVelocity, value: FaceVelocity, /
    ) -> MACStageInverseMomentumDiagnostics:
        rhs_ = self.operators.validate_velocity(rhs)
        value_ = self.operators.validate_velocity(value)
        reconstructed = tuple(
            item / inverse
            for item, inverse in zip(value_, self.inverse_diagonal, strict=True)
        )
        residual = tuple(
            left - right for left, right in zip(reconstructed, rhs_, strict=True)
        )
        norm = jnp.sqrt(jnp.real(self.operators.velocity_space.inner(residual, residual)))
        rhs_norm = jnp.sqrt(jnp.real(self.operators.velocity_space.inner(rhs_, rhs_)))
        finite = jnp.isfinite(norm) & jnp.isfinite(rhs_norm)
        return MACStageInverseMomentumDiagnostics(
            norm,
            norm / jnp.maximum(rhs_norm, 1.0),
            finite,
            finite,
            self.stage_id,
        )


class MACHelmholtzStageInverseMomentum(StrictModule, NonTrainableState):
    """Repeated homogeneous inverse of the actual MAC Helmholtz stage operator."""

    plan: MACHelmholtzSolvePlan
    boundary_stage: MACBoundaryStageData
    mass_coefficient: Array | None
    diffusion_coefficient: Array | None
    rhs_scale: Array
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MACHelmholtzSolvePlan,
        boundary_stage: MACBoundaryStageData,
        /,
        *,
        mass_coefficient: ArrayLike | None = None,
        diffusion_coefficient: ArrayLike | None = None,
        rhs_scale: ArrayLike = 1.0,
        stage_id: str,
    ):
        if not isinstance(plan, MACHelmholtzSolvePlan):
            raise TypeError("plan must be MACHelmholtzSolvePlan.")
        identifier = str(stage_id)
        if not identifier:
            raise ValueError("stage_id must be nonempty.")
        scale = jnp.asarray(rhs_scale)
        scale = eqx.error_if(
            scale,
            ~jnp.isfinite(scale) | (scale <= 0.0),
            "Stage inverse RHS scale must be positive and finite.",
        )
        self.plan = plan
        self.boundary_stage = boundary_stage
        self.mass_coefficient = (
            None if mass_coefficient is None else jnp.asarray(mass_coefficient)
        )
        self.diffusion_coefficient = (
            None if diffusion_coefficient is None else jnp.asarray(diffusion_coefficient)
        )
        self.rhs_scale = scale
        self.stage_id = identifier

    @property
    def operators(self) -> PreparedMACOperators:
        return self.plan.momentum.operators

    def solve(self, rhs: FaceVelocity, /) -> MACHelmholtzResult:
        values = self.operators.validate_velocity(rhs)
        scaled = tuple(self.rhs_scale * value for value in values)
        zero = tuple(jnp.zeros_like(value) for value in values)
        return self.plan.solve(
            scaled,
            self.boundary_stage,
            mass_coefficient=self.mass_coefficient,
            diffusion_coefficient=self.diffusion_coefficient,
            initial_guess=zero,
        )

    def apply_inverse(self, rhs: FaceVelocity, /) -> FaceVelocity:
        result = self.solve(rhs)
        flattened = self.operators.velocity_space.flatten(result.value)
        checked = eqx.error_if(
            flattened,
            ~result.converged,
            "MAC stage inverse momentum solve failed.",
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
            operator_id=f"mac-stage-inverse/{self.stage_id}",
        )

    def diagnostics(
        self, rhs: FaceVelocity, value: FaceVelocity, /
    ) -> MACStageInverseMomentumDiagnostics:
        del value
        result = self.solve(rhs)
        return MACStageInverseMomentumDiagnostics(
            result.residual_norm,
            result.relative_residual,
            result.finite,
            result.converged,
            self.stage_id,
        )


MACStageInverseMomentum: TypeAlias = (
    MACDiagonalStageInverseMomentum | MACHelmholtzStageInverseMomentum
)


__all__ = [
    "MACDiagonalStageInverseMomentum",
    "MACHelmholtzStageInverseMomentum",
    "MACStageInverseMomentum",
    "MACStageInverseMomentumDiagnostics",
]
