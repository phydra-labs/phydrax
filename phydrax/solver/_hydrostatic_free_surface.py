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
from ..discretization.finite_volume._hydrostatic_grid import (
    HydrostaticMetricEpoch,
    PreparedHydrostaticGrid,
)
from ..linalg import (
    ArraySpace,
    ConjugateGradient,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    solve,
    TolerancePolicy,
)


class HydrostaticFreeSurfaceResult(StrictModule):
    """Implicit free-surface solution and corrected layer transports."""

    eta: Array
    transports: tuple[Array, Array]
    rhs: Array
    residual_norm: Array
    continuity_residual: Array
    iterations: Array
    converged: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class LinearImplicitFreeSurfacePlan(StrictModule, NonTrainableState):
    """Matrix-free linear implicit free-surface Helmholtz solve."""

    geometry: PreparedHydrostaticGrid
    gravity: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: PreparedHydrostaticGrid,
        /,
        *,
        gravity: float = 9.81,
        tolerance: float = 1.0e-10,
        maximum_iterations: int = 500,
    ):
        if not isinstance(geometry, PreparedHydrostaticGrid):
            raise TypeError("geometry must be PreparedHydrostaticGrid.")
        gravity_ = float(gravity)
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if gravity_ <= 0.0 or tolerance_ <= 0.0 or iterations <= 0:
            raise ValueError("Invalid free-surface gravity, tolerance, or iterations.")
        self.geometry = geometry
        self.gravity = gravity_
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "linear-implicit-hydrostatic-free-surface",
                "geometry": geometry.geometry_id,
                "gravity": gravity_,
                "tolerance": tolerance_,
                "maximum_iterations": iterations,
            }
        )

    def _surface_flux_from_eta(
        self,
        eta: Array,
        epoch: HydrostaticMetricEpoch,
        /,
        *,
        boundary_values=None,
    ) -> tuple[Array, Array]:
        gx, gy = self.geometry.surface_gradient(
            eta, boundary_values=boundary_values
        )
        return (
            jnp.sum(epoch.x_face_area, axis=-1) * gx,
            jnp.sum(epoch.y_face_area, axis=-1) * gy,
        )

    def solve(
        self,
        eta: ArrayLike,
        predictor_transports: tuple[ArrayLike, ArrayLike],
        epoch: HydrostaticMetricEpoch,
        step_size: ArrayLike,
        freshwater_rate: ArrayLike | None = None,
        /,
        *,
        boundary_values=None,
    ) -> HydrostaticFreeSurfaceResult:
        eta_old = jnp.asarray(eta, dtype=self.geometry.cell_area.dtype)
        dt = jnp.asarray(step_size, dtype=eta_old.dtype).reshape(())
        if eta_old.shape != self.geometry.horizontal_shape:
            raise ValueError("Free-surface eta shape is invalid.")
        x = jnp.asarray(predictor_transports[0], dtype=eta_old.dtype)
        y = jnp.asarray(predictor_transports[1], dtype=eta_old.dtype)
        if x.shape != self.geometry.x_face_shape or y.shape != self.geometry.y_face_shape:
            raise ValueError("Predictor layer-transport shapes are invalid.")
        source = (
            jnp.zeros_like(eta_old)
            if freshwater_rate is None
            else jnp.asarray(freshwater_rate, dtype=eta_old.dtype)
        )
        if source.shape != eta_old.shape:
            raise ValueError("Freshwater free-surface source shape is invalid.")
        predictor_net = self.geometry.surface_net_transport((x, y))
        rhs = eta_old - dt * predictor_net / self.geometry.cell_area + dt * source
        if boundary_values is not None:
            boundary_flux = self._surface_flux_from_eta(
                jnp.zeros_like(eta_old),
                epoch,
                boundary_values=boundary_values,
            )
            boundary_laplacian = _surface_net_flux(
                self.geometry, boundary_flux[0], boundary_flux[1]
            )
            rhs = rhs + (
                self.gravity * dt**2 / self.geometry.cell_area
            ) * boundary_laplacian
        space = ArraySpace(self.geometry.horizontal_shape, dtype=eta_old.dtype)

        def action(eta_value):
            fx, fy = self._surface_flux_from_eta(eta_value, epoch)
            laplacian = _surface_net_flux(self.geometry, fx, fy)
            return (
                eta_value - (self.gravity * dt**2 / self.geometry.cell_area) * laplacian
            )

        operator = FunctionLinearOperator(
            action,
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
            operator_id=canonical_fingerprint(
                {
                    "kind": "hydrostatic-free-surface-action",
                    "plan": self.plan_id,
                    "epoch": epoch.epoch_id,
                }
            ),
        )
        prepared = prepare(
            LinearSystem(operator),
            LinearSolvePolicy(
                ConjugateGradient(),
                tolerance=TolerancePolicy(
                    relative=self.tolerance,
                    absolute=self.tolerance,
                    max_steps=self.maximum_iterations,
                ),
            ),
        )
        linear = solve(
            prepared,
            rhs,
            initial_guess=eta_old,
        )
        eta_new = linear.value
        pressure_x, pressure_y = self.geometry.layer_pressure_transport_force(
            eta_new, epoch, boundary_values=boundary_values
        )
        corrected = (
            x + self.gravity * dt * pressure_x,
            y + self.gravity * dt * pressure_y,
        )
        continuity = (
            eta_new
            - eta_old
            + dt
            * self.geometry.surface_net_transport(corrected)
            / self.geometry.cell_area
            - dt * source
        )
        residual_norm = jnp.sqrt(jnp.real(jnp.vdot(continuity, continuity)))
        finite = (
            jnp.all(jnp.isfinite(eta_new))
            & jnp.all(jnp.isfinite(corrected[0]))
            & jnp.all(jnp.isfinite(corrected[1]))
            & jnp.isfinite(residual_norm)
        )
        successful = linear.successful & finite & epoch.valid
        return HydrostaticFreeSurfaceResult(
            eta=eta_new,
            transports=corrected,
            rhs=rhs,
            residual_norm=residual_norm,
            continuity_residual=continuity,
            iterations=linear.diagnostics.iterations,
            converged=linear.successful,
            finite=finite,
            successful=successful,
            plan_id=self.plan_id,
        )


def _surface_net_flux(
    geometry: PreparedHydrostaticGrid, x_flux: Array, y_flux: Array, /
) -> Array:
    return geometry.surface_net_flux((x_flux, y_flux))


__all__ = ["HydrostaticFreeSurfaceResult", "LinearImplicitFreeSurfacePlan"]
