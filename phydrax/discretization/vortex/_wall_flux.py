#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSolveResult,
    solve as solve_linear,
)
from ...operators.integral.vortex._panels2d import (
    FlowPanelGeometry2D,
    panel_influence_matrix_2d,
)
from ._population import VortexPopulationState
from ._wall import (
    BoundarySheetParticleTransferPlan2D,
    BoundarySheetParticleTransferResult,
    WallVortexPoolState,
)


class WallVorticityFluxEvidence(StrictModule):
    normal_residual: Array
    tangential_residual: Array
    circulation_flux: Array
    boundary_work: Array
    slip_norm: Array
    finite: Array


class WallVorticityFluxResult(StrictModule):
    source_strength: Array
    vortex_sheet_strength: Array
    emitted_circulation: Array
    evidence: WallVorticityFluxEvidence
    linear_result: LinearSolveResult
    successful: Array
    solver_id: str = eqx.field(static=True)


class BoundaryIntegralVorticityFluxPlan2D(StrictModule, NonTrainableState):
    geometry: FlowPanelGeometry2D
    policy: LinearSolvePolicy
    solver_id: str = eqx.field(static=True)

    def __init__(
        self, geometry: FlowPanelGeometry2D, /, *, policy: LinearSolvePolicy | None = None
    ):
        if not isinstance(geometry, FlowPanelGeometry2D):
            raise TypeError("geometry must be FlowPanelGeometry2D.")
        self.geometry = geometry
        self.policy = LinearSolvePolicy(DenseSVD()) if policy is None else policy
        self.solver_id = canonical_fingerprint(
            {
                "kind": "boundary-integral-vorticity-flux-2d",
                "geometry": geometry.geometry_id,
            }
        )

    def solve(
        self,
        incident_velocity: ArrayLike,
        body_velocity: ArrayLike,
        time_step: ArrayLike,
        /,
    ) -> WallVorticityFluxResult:
        incident = jnp.asarray(incident_velocity, dtype=self.geometry.control.dtype)
        body = jnp.asarray(body_velocity, dtype=incident.dtype)
        dt = jnp.asarray(time_step, dtype=incident.dtype)
        if (
            incident.shape != self.geometry.control.shape
            or body.shape != incident.shape
            or dt.shape != ()
        ):
            raise ValueError("Wall flux velocity/time arrays are incompatible.")
        source_normal, source_tangent = panel_influence_matrix_2d(
            self.geometry, kind="source"
        )
        vortex_normal, vortex_tangent = panel_influence_matrix_2d(
            self.geometry, kind="vortex"
        )
        relative = incident - body
        normal_rhs = -jnp.sum(relative * self.geometry.normal, axis=-1)
        tangent_rhs = -jnp.sum(relative * self.geometry.tangent, axis=-1)
        block = jnp.block(
            [
                [source_normal, vortex_normal],
                [source_tangent, vortex_tangent],
            ]
        )
        rhs = jnp.concatenate((normal_rhs, tangent_rhs))
        linear = solve_linear(
            LeastSquaresProblem(
                DenseLinearOperator(block), problem_id=f"{self.solver_id}:no-slip"
            ),
            rhs,
            policy=self.policy,
        )
        solution = jnp.asarray(linear.value)
        count = int(self.geometry.length.size)
        source_strength, sheet = solution[:count], solution[count:]
        residual = block @ solution - rhs
        normal_residual, tangential_residual = residual[:count], residual[count:]
        emitted = sheet * self.geometry.length
        circulation_flux = jnp.sum(emitted) / dt
        boundary_traction = (
            sheet[:, None] * self.geometry.tangent
            + source_strength[:, None] * self.geometry.normal
        )
        work = jnp.sum(boundary_traction * body * self.geometry.length[:, None])
        slip = jnp.sqrt(jnp.sum(normal_residual**2 + tangential_residual**2))
        finite = jnp.all(jnp.isfinite(solution)) & jnp.isfinite(dt) & (dt > 0.0)
        tolerance = 1.0e-8 * jnp.maximum(jnp.linalg.norm(rhs), 1.0)
        successful = linear.successful & finite & (slip <= tolerance)
        evidence = WallVorticityFluxEvidence(
            normal_residual, tangential_residual, circulation_flux, work, slip, finite
        )
        return WallVorticityFluxResult(
            source_strength, sheet, emitted, evidence, linear, successful, self.solver_id
        )

    def transfer(
        self,
        flux: WallVorticityFluxResult,
        pool: WallVortexPoolState,
        transfer: BoundarySheetParticleTransferPlan2D,
        /,
    ) -> BoundarySheetParticleTransferResult:
        if not isinstance(flux, WallVorticityFluxResult) or not isinstance(
            transfer, BoundarySheetParticleTransferPlan2D
        ):
            raise TypeError("Wall transfer requires flux result and transfer plan.")
        result = transfer.transfer(pool, self.geometry, flux.vortex_sheet_strength)
        return eqx.tree_at(
            lambda value: value.successful, result, result.successful & flux.successful
        )


class WallCrossingResult(StrictModule):
    state: VortexPopulationState
    crossing_count: Array
    reflected_count: Array
    absorbed_circulation: Array
    successful: Array
    policy_id: str = eqx.field(static=True)


class WallCrossingPlan(StrictModule, NonTrainableState):
    signed_distance: Callable[[Array], Array]
    normal: Callable[[Array], Array]
    policy: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, signed_distance, normal, /, *, policy: str, policy_id: str):
        if (
            not callable(signed_distance)
            or not callable(normal)
            or policy not in ("reflect", "absorb")
            or not str(policy_id)
        ):
            raise ValueError("Wall crossing plan inputs are invalid.")
        self.signed_distance, self.normal, self.policy, self.policy_id = (
            signed_distance,
            normal,
            policy,
            str(policy_id),
        )

    def apply(self, state: VortexPopulationState, /) -> WallCrossingResult:
        distance = jax_vmap(self.signed_distance, state.positions)
        crossing = state.active_mask & (distance < 0.0)
        normal = jax_vmap(self.normal, state.positions)
        if self.policy == "reflect":
            position = (
                state.positions - 2.0 * jnp.minimum(distance, 0.0)[:, None] * normal
            )
            active = state.active_mask
            strength = state.strength
            reflected = jnp.sum(crossing, dtype=jnp.int32)
            absorbed = jnp.zeros_like(jnp.sum(state.strength, axis=0))
        else:
            position = state.positions
            active = state.active_mask & ~crossing
            strength_mask = active if state.strength.ndim == 1 else active[:, None]
            absorbed = jnp.sum(
                jnp.where(
                    crossing if state.strength.ndim == 1 else crossing[:, None],
                    state.strength,
                    0.0,
                ),
                axis=0,
            )
            strength = jnp.where(strength_mask, state.strength, 0.0)
            reflected = jnp.asarray(0, dtype=jnp.int32)
        candidate = VortexPopulationState(
            position,
            strength,
            state.core_radius,
            state.volume,
            active,
            jnp.where(active, state.stable_ids, -1),
            state.parent_ids,
            state.source_codes,
            state.age,
            state.next_stable_id,
        )
        finite = jnp.all(jnp.isfinite(position))
        return WallCrossingResult(
            candidate,
            jnp.sum(crossing, dtype=jnp.int32),
            reflected,
            absorbed,
            finite,
            self.policy_id,
        )


def jax_vmap(function, values):
    import jax

    return jax.vmap(function)(values)


class SeparationModelResult(StrictModule):
    separation: Array
    confidence: Array
    resolved: bool = eqx.field(static=True)
    model_id: str = eqx.field(static=True)


class ReducedSeparationModel(StrictModule, NonTrainableState):
    critical_pressure_gradient: float = eqx.field(static=True)
    critical_shear: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(self, critical_pressure_gradient: float, critical_shear: float, /):
        self.critical_pressure_gradient, self.critical_shear = (
            float(critical_pressure_gradient),
            float(critical_shear),
        )
        self.model_id = canonical_fingerprint(
            {
                "kind": "reduced-separation-model",
                "critical_pressure_gradient": self.critical_pressure_gradient,
                "critical_shear": self.critical_shear,
            }
        )

    def evaluate(
        self, adverse_pressure_gradient: ArrayLike, wall_shear: ArrayLike, /
    ) -> SeparationModelResult:
        gradient, shear = jnp.asarray(adverse_pressure_gradient), jnp.asarray(wall_shear)
        separation = (gradient >= self.critical_pressure_gradient) & (
            shear <= self.critical_shear
        )
        confidence = jnp.minimum(
            jnp.abs(gradient - self.critical_pressure_gradient)
            / jnp.maximum(abs(self.critical_pressure_gradient), 1.0e-12),
            jnp.abs(shear - self.critical_shear)
            / jnp.maximum(abs(self.critical_shear), 1.0e-12),
        )
        return SeparationModelResult(separation, confidence, False, self.model_id)


__all__ = [
    "BoundaryIntegralVorticityFluxPlan2D",
    "ReducedSeparationModel",
    "SeparationModelResult",
    "WallCrossingPlan",
    "WallCrossingResult",
    "WallVorticityFluxEvidence",
    "WallVorticityFluxResult",
]
