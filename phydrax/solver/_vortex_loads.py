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
from ..discretization.vortex._lifting_complete import PreparedMultiLiftingSurface


class VortexLoadResult(StrictModule):
    panel_force: Array
    panel_moment: Array
    total_force: Array
    total_moment: Array
    method: str = eqx.field(static=True)
    reference_point: Array
    finite: Array
    load_id: str = eqx.field(static=True)
    evidence: object


class KuttaJoukowskiLoadPlan(StrictModule, NonTrainableState):
    density: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, density: float = 1.0, /):
        if float(density) <= 0.0:
            raise ValueError("Kutta-Joukowski density must be positive.")
        self.density = float(density)
        self.plan_id = canonical_fingerprint(
            {"kind": "kutta-joukowski-load-plan", "density": self.density}
        )

    def evaluate(
        self,
        surface: PreparedMultiLiftingSurface,
        circulation: ArrayLike,
        velocity: ArrayLike,
        /,
        *,
        reference_point: ArrayLike = (0.0, 0.0, 0.0),
    ) -> VortexLoadResult:
        gamma = jnp.asarray(circulation, dtype=surface.control_point.dtype)
        flow = jnp.asarray(velocity, dtype=surface.control_point.dtype)
        reference = jnp.asarray(reference_point, dtype=surface.control_point.dtype)
        if (
            gamma.shape != (surface.panel_count,)
            or flow.shape != (surface.panel_count, 3)
            or reference.shape != (3,)
        ):
            raise ValueError("Kutta-Joukowski load arrays are incompatible.")
        bound = surface.bound_end - surface.bound_start
        force = self.density * gamma[:, None] * jnp.cross(flow, bound)
        moment = jnp.cross(surface.control_point - reference, force)
        finite = jnp.all(jnp.isfinite(force)) & jnp.all(jnp.isfinite(moment))
        return VortexLoadResult(
            force,
            moment,
            jnp.sum(force, axis=0),
            jnp.sum(moment, axis=0),
            "kutta-joukowski",
            reference,
            finite,
            self.plan_id,
            None,
        )


class UnsteadyBernoulliLoadPlan(StrictModule, NonTrainableState):
    density: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, density: float = 1.0, /):
        if float(density) <= 0.0:
            raise ValueError("Bernoulli density must be positive.")
        self.density = float(density)
        self.plan_id = canonical_fingerprint(
            {"kind": "unsteady-bernoulli-load-plan", "density": self.density}
        )

    def evaluate(
        self,
        surface: PreparedMultiLiftingSurface,
        tangential_speed: ArrayLike,
        potential_rate: ArrayLike,
        reference_speed: ArrayLike,
        /,
        *,
        reference_point: ArrayLike = (0.0, 0.0, 0.0),
    ) -> VortexLoadResult:
        speed = jnp.asarray(tangential_speed, dtype=surface.control_point.dtype)
        rate = jnp.asarray(potential_rate, dtype=speed.dtype)
        reference_speed_ = jnp.asarray(reference_speed, dtype=speed.dtype)
        reference = jnp.asarray(reference_point, dtype=speed.dtype)
        if (
            speed.shape != (surface.panel_count,)
            or rate.shape != speed.shape
            or reference_speed_.shape != ()
        ):
            raise ValueError("Unsteady Bernoulli load arrays are incompatible.")
        pressure = -self.density * (rate + 0.5 * (speed**2 - reference_speed_**2))
        area = surface.chord * surface.span_width
        force = -pressure[:, None] * area[:, None] * surface.normal
        moment = jnp.cross(surface.control_point - reference, force)
        finite = jnp.all(jnp.isfinite(force)) & jnp.all(jnp.isfinite(pressure))
        return VortexLoadResult(
            force,
            moment,
            jnp.sum(force, axis=0),
            jnp.sum(moment, axis=0),
            "unsteady-bernoulli",
            reference,
            finite,
            self.plan_id,
            pressure,
        )


class ImpulseLoadPlan(StrictModule, NonTrainableState):
    density: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, density: float = 1.0, /):
        if float(density) <= 0.0:
            raise ValueError("Impulse density must be positive.")
        self.density = float(density)
        self.plan_id = canonical_fingerprint(
            {"kind": "impulse-load-plan", "density": self.density}
        )

    def evaluate(
        self,
        previous_impulse: ArrayLike,
        current_impulse: ArrayLike,
        time_step: ArrayLike,
        /,
        *,
        reference_point: ArrayLike = (0.0, 0.0, 0.0),
    ) -> VortexLoadResult:
        previous = jnp.asarray(previous_impulse)
        current = jnp.asarray(current_impulse, dtype=previous.dtype)
        dt = jnp.asarray(time_step, dtype=previous.dtype)
        reference = jnp.asarray(reference_point, dtype=previous.dtype)
        if (
            previous.shape != (3,)
            or current.shape != (3,)
            or dt.shape != ()
            or reference.shape != (3,)
        ):
            raise ValueError("Impulse load arrays are incompatible.")
        force = -self.density * (current - previous) / dt
        finite = jnp.all(jnp.isfinite(force)) & jnp.isfinite(dt) & (dt > 0.0)
        return VortexLoadResult(
            force[None, :],
            jnp.zeros((1, 3), dtype=force.dtype),
            force,
            jnp.zeros((3,), dtype=force.dtype),
            "impulse",
            reference,
            finite,
            self.plan_id,
            None,
        )


class TrefftzInducedDragPlan(StrictModule, NonTrainableState):
    density: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, density: float = 1.0, /):
        self.density = float(density)
        if self.density <= 0.0:
            raise ValueError("Trefftz density must be positive.")
        self.plan_id = canonical_fingerprint(
            {"kind": "trefftz-induced-drag-plan", "density": self.density}
        )

    def evaluate(
        self, circulation: ArrayLike, downwash: ArrayLike, span_width: ArrayLike, /
    ) -> tuple[Array, Array]:
        gamma, velocity, width = (
            jnp.asarray(circulation),
            jnp.asarray(downwash),
            jnp.asarray(span_width),
        )
        if gamma.shape != velocity.shape or width.shape != gamma.shape:
            raise ValueError("Trefftz circulation/downwash/span arrays must match.")
        panel_drag = -self.density * gamma * velocity * width
        return panel_drag, jnp.sum(panel_drag)


__all__ = [
    "ImpulseLoadPlan",
    "KuttaJoukowskiLoadPlan",
    "TrefftzInducedDragPlan",
    "UnsteadyBernoulliLoadPlan",
    "VortexLoadResult",
]
