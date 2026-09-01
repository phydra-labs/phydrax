#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._free_surface_ale import PreparedGraphSurfaceALE


class GraphCapillarityResult(StrictModule):
    surface_area: Array
    surface_energy: Array
    generalized_force: Array
    pressure_head: Array
    dual_residual: Array
    dual_residual_norm: Array
    timestep_limit: Array
    iterations: Array
    finite: Array
    converged: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class GraphCapillarityPlan(StrictModule, NonTrainableState):
    """Variational fitted-graph surface tension through the kinematic dual."""

    surface: PreparedGraphSurfaceALE
    surface_tension: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: PreparedGraphSurfaceALE,
        surface_tension: float,
        /,
        *,
        tolerance: float = 1.0e-10,
        maximum_iterations: int = 200,
    ):
        if not isinstance(surface, PreparedGraphSurfaceALE):
            raise TypeError("surface must be PreparedGraphSurfaceALE.")
        sigma = float(surface_tension)
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if (
            not np.isfinite(sigma)
            or sigma < 0.0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or iterations <= 0
        ):
            raise ValueError("Invalid graph surface-tension plan.")
        self.surface = surface
        self.surface_tension = sigma
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "graph-capillarity-plan",
                "surface": surface.surface_id,
                "surface_tension": sigma,
                "tolerance": tolerance_,
                "maximum_iterations": iterations,
                "triangulation": "fixed-lower-left-to-upper-right",
            }
        )

    def surface_area(self, eta: ArrayLike, /) -> Array:
        eta_ = jnp.asarray(eta)
        geometry = self.surface.geometry(jnp.asarray(0.0), eta_, jnp.zeros_like(eta_))
        top = geometry.mapped_vertices[:, :, -1, :]
        p00 = top[:-1, :-1]
        p10 = top[1:, :-1]
        p01 = top[:-1, 1:]
        p11 = top[1:, 1:]
        first = 0.5 * jnp.linalg.norm(jnp.cross(p10 - p00, p11 - p00), axis=-1)
        second = 0.5 * jnp.linalg.norm(jnp.cross(p11 - p00, p01 - p00), axis=-1)
        return jnp.sum(first + second)

    def _surface_jacobian_actions(self, eta: Array):
        def volume_map(value):
            return self.surface._column_volumes(value)

        def jacobian(rate):
            return jax.jvp(volume_map, (eta,), (rate,))[1]

        def transpose(head):
            return jax.linear_transpose(jacobian, jnp.zeros_like(eta))(head)[0]

        return jacobian, transpose

    def _solve_head(self, eta: Array, covector: Array, /):
        jacobian, transpose = self._surface_jacobian_actions(eta)

        def normal_action(head):
            return jacobian(transpose(head))

        rhs = jacobian(covector)
        head = jnp.zeros_like(eta)
        residual = rhs - normal_action(head)
        direction = residual
        norm = jnp.real(jnp.vdot(residual, residual))
        threshold = self.tolerance**2 * jnp.maximum(norm, 1.0)
        active = norm > threshold
        failed = jnp.asarray(False)

        def body(_, state):
            value, residual_, direction_, norm_, active_, failed_ = state
            image = normal_action(direction_)
            denominator = jnp.real(jnp.vdot(direction_, image))
            valid = active_ & jnp.isfinite(denominator) & (denominator > 0.0)
            alpha = jnp.where(valid, norm_ / denominator, 0.0)
            next_value = value + alpha * direction_
            next_residual = residual_ - alpha * image
            next_norm = jnp.real(jnp.vdot(next_residual, next_residual))
            running = valid & (next_norm > threshold)
            beta = jnp.where(running & (norm_ > 0.0), next_norm / norm_, 0.0)
            return (
                next_value,
                next_residual,
                next_residual + beta * direction_,
                next_norm,
                running,
                failed_ | (active_ & ~valid),
            )

        head, _, _, _, active, failed = jax.lax.fori_loop(
            0,
            self.maximum_iterations,
            body,
            (head, residual, direction, norm, active, failed),
        )
        dual_residual = transpose(head) - covector
        residual_norm = jnp.sqrt(jnp.real(jnp.vdot(dual_residual, dual_residual)))
        finite = jnp.all(jnp.isfinite(head)) & jnp.isfinite(residual_norm)
        return head, dual_residual, residual_norm, ~active & ~failed & finite

    def evaluate(
        self,
        eta: ArrayLike,
        density: float,
        /,
    ) -> GraphCapillarityResult:
        eta_ = jnp.asarray(eta)
        density_ = float(density)
        if density_ <= 0.0 or not np.isfinite(density_):
            raise ValueError("Capillary density must be finite and positive.")
        area = self.surface_area(eta_)
        energy = self.surface_tension * area
        generalized_force = jax.grad(
            lambda value: self.surface_tension * self.surface_area(value)
        )(eta_)
        covector = generalized_force / density_
        if self.surface_tension == 0.0:
            head = jnp.zeros_like(eta_)
            dual_residual = jnp.zeros_like(eta_)
            residual_norm = jnp.asarray(0.0, dtype=eta_.dtype)
            converged = jnp.asarray(True)
        else:
            head, dual_residual, residual_norm, converged = self._solve_head(
                eta_, covector
            )
        geometry = self.surface.geometry(jnp.asarray(0.0), eta_, jnp.zeros_like(eta_))
        top_measure = jnp.take(geometry.face_measures[2], -1, axis=2)
        minimum_length = jnp.sqrt(jnp.min(top_measure))
        timestep = jnp.where(
            self.surface_tension > 0.0,
            0.5
            * jnp.sqrt(density_ * minimum_length**3 / (math.pi * self.surface_tension)),
            jnp.inf,
        )
        timestep_valid = jnp.isfinite(timestep) | (
            (self.surface_tension == 0.0) & jnp.isinf(timestep)
        )
        finite = (
            jnp.isfinite(area)
            & jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(generalized_force))
            & jnp.all(jnp.isfinite(head))
            & jnp.isfinite(residual_norm)
            & timestep_valid
        )
        return GraphCapillarityResult(
            surface_area=area,
            surface_energy=energy,
            generalized_force=generalized_force,
            pressure_head=head,
            dual_residual=dual_residual,
            dual_residual_norm=residual_norm,
            timestep_limit=timestep,
            iterations=jnp.asarray(self.maximum_iterations, dtype=jnp.int32),
            finite=finite,
            converged=converged,
            successful=finite & converged,
            plan_id=self.plan_id,
        )


__all__ = ["GraphCapillarityPlan", "GraphCapillarityResult"]
