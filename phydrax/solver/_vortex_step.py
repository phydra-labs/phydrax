#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..nonlinear import (
    AbstractNonlinearMethod,
    implicit_root_result,
    ImplicitRootDerivativePolicy,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._vortex_lattice import SteadyVortexLatticePlan


class SampledAirfoilPolar(StrictModule, NonTrainableState):
    angle: Array
    lift: Array
    drag: Array
    moment: Array
    endpoint: Literal["clamp", "error"] = eqx.field(static=True)
    polar_id: str = eqx.field(static=True)

    def __init__(
        self,
        angle: ArrayLike,
        lift: ArrayLike,
        drag: ArrayLike,
        moment: ArrayLike | None = None,
        /,
        *,
        endpoint: Literal["clamp", "error"] = "clamp",
    ):
        alpha = jnp.asarray(angle, dtype=float)
        cl = jnp.asarray(lift, dtype=float)
        cd = jnp.asarray(drag, dtype=float)
        cm = jnp.zeros_like(alpha) if moment is None else jnp.asarray(moment, dtype=float)
        if (
            alpha.ndim != 1
            or alpha.size < 2
            or cl.shape != alpha.shape
            or cd.shape != alpha.shape
            or cm.shape != alpha.shape
        ):
            raise ValueError("Polar arrays must be matching nonempty vectors.")
        if (
            not bool(jnp.all(jnp.diff(alpha) > 0.0))
            or not bool(jnp.all(jnp.isfinite(alpha)))
            or not bool(jnp.all(jnp.isfinite(cl)))
            or not bool(jnp.all(jnp.isfinite(cd)))
            or not bool(jnp.all(jnp.isfinite(cm)))
        ):
            raise ValueError(
                "Polar samples must be finite with strictly increasing angle."
            )
        if endpoint not in ("clamp", "error"):
            raise ValueError("Polar endpoint policy must be 'clamp' or 'error'.")
        self.angle, self.lift, self.drag, self.moment = alpha, cl, cd, cm
        self.endpoint = endpoint
        self.polar_id = canonical_fingerprint(
            {
                "kind": "sampled-airfoil-polar",
                "angle": array_tree_fingerprint(alpha),
                "lift": array_tree_fingerprint(cl),
                "drag": array_tree_fingerprint(cd),
                "moment": array_tree_fingerprint(cm),
                "endpoint": endpoint,
            }
        )

    def evaluate(self, angle: ArrayLike, /) -> tuple[Array, Array, Array]:
        alpha = jnp.asarray(angle, dtype=self.angle.dtype)
        if self.endpoint == "error":
            alpha = eqx.error_if(
                alpha,
                jnp.any((alpha < self.angle[0]) | (alpha > self.angle[-1])),
                "Airfoil polar query is outside its sample range.",
            )
        return (
            jnp.interp(alpha, self.angle, self.lift),
            jnp.interp(alpha, self.angle, self.drag),
            jnp.interp(alpha, self.angle, self.moment),
        )


class VortexStepResult(StrictModule):
    circulation: Array
    effective_angle: Array
    lift_coefficient: Array
    drag_coefficient: Array
    panel_force: Array
    total_force: Array
    residual_norm: Array
    nonlinear_result: NonlinearResult
    successful: Array
    solver_id: str = eqx.field(static=True)


class VortexStepPlan(StrictModule, NonTrainableState):
    """Nonlinear polar-coupled circulation root over a prepared lifting surface."""

    lattice: SteadyVortexLatticePlan
    polar: SampledAirfoilPolar
    nonlinear_method: AbstractNonlinearMethod | None
    termination: NonlinearTermination | None
    derivative_policy: ImplicitRootDerivativePolicy | None
    solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        lattice: SteadyVortexLatticePlan,
        polar: SampledAirfoilPolar,
        /,
        *,
        nonlinear_method: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
        derivative_policy: ImplicitRootDerivativePolicy | None = None,
    ):
        if not isinstance(lattice, SteadyVortexLatticePlan) or not isinstance(
            polar, SampledAirfoilPolar
        ):
            raise TypeError("VortexStepPlan requires lattice and polar objects.")
        self.lattice = lattice
        self.polar = polar
        self.nonlinear_method = nonlinear_method
        self.termination = termination
        self.derivative_policy = derivative_policy
        self.solver_id = canonical_fingerprint(
            {
                "kind": "vortex-step-plan",
                "lattice": lattice.solver_id,
                "polar": polar.polar_id,
            }
        )

    def solve(
        self,
        freestream_velocity: ArrayLike,
        initial_circulation: ArrayLike | None = None,
        /,
    ) -> VortexStepResult:
        surface = self.lattice.surface
        freestream = jnp.asarray(freestream_velocity, dtype=surface.control_point.dtype)
        if freestream.shape == (3,):
            freestream = jnp.broadcast_to(freestream, (surface.panel_count, 3))
        if freestream.shape != (surface.panel_count, 3):
            raise ValueError("Vortex-step freestream shape is invalid.")
        influence = self.lattice.influence_velocity()
        chord_vector = 0.5 * (
            (surface.trailing_edge[:-1] - surface.leading_edge[:-1])
            + (surface.trailing_edge[1:] - surface.leading_edge[1:])
        )
        chord_direction = chord_vector / jnp.linalg.norm(chord_vector, axis=-1)[:, None]
        span_direction = (surface.bound_end - surface.bound_start) / surface.span_width[
            :, None
        ]

        def fields(gamma):
            velocity = freestream + contract("tjc,j->tc", influence, gamma)
            chord_speed = jnp.sum(velocity * chord_direction, axis=-1)
            normal_speed = jnp.sum(velocity * surface.normal, axis=-1)
            angle = jnp.arctan2(normal_speed, chord_speed)
            cl, cd, cm = self.polar.evaluate(angle)
            crossflow = jnp.linalg.norm(jnp.cross(velocity, span_direction), axis=-1)
            target = 0.5 * crossflow * surface.chord * cl
            return velocity, angle, cl, cd, cm, target

        def residual(gamma, args):
            del args
            return fields(gamma)[-1] - gamma

        initial = (
            jnp.zeros((surface.panel_count,), dtype=freestream.dtype)
            if initial_circulation is None
            else jnp.asarray(initial_circulation, dtype=freestream.dtype)
        )
        if initial.shape != (surface.panel_count,):
            raise ValueError("initial_circulation must have panel_count shape.")
        problem = NonlinearSystemProblem(
            residual, problem_id=f"{self.solver_id}:circulation"
        )
        nonlinear = implicit_root_result(
            problem,
            initial,
            method=self.nonlinear_method,
            termination=self.termination,
            derivative_policy=self.derivative_policy,
        )
        gamma = jnp.asarray(nonlinear.state)
        velocity, angle, cl, cd, _, target = fields(gamma)
        dynamic_pressure = (
            0.5 * self.lattice.density * jnp.sum(velocity * velocity, axis=-1)
        )
        lift_direction = jnp.cross(span_direction, velocity)
        lift_direction = (
            lift_direction
            / jnp.maximum(
                jnp.linalg.norm(lift_direction, axis=-1), jnp.finfo(velocity.dtype).tiny
            )[:, None]
        )
        drag_direction = (
            -velocity
            / jnp.maximum(
                jnp.linalg.norm(velocity, axis=-1), jnp.finfo(velocity.dtype).tiny
            )[:, None]
        )
        area = surface.chord * surface.span_width
        panel_force = (
            dynamic_pressure[:, None]
            * area[:, None]
            * (cl[:, None] * lift_direction + cd[:, None] * drag_direction)
        )
        residual_norm = jnp.linalg.norm(target - gamma)
        successful = nonlinear.successful & jnp.all(jnp.isfinite(panel_force))
        return VortexStepResult(
            gamma,
            angle,
            cl,
            cd,
            panel_force,
            jnp.sum(panel_force, axis=0),
            residual_norm,
            nonlinear,
            successful,
            self.solver_id,
        )


__all__ = ["SampledAirfoilPolar", "VortexStepPlan", "VortexStepResult"]
