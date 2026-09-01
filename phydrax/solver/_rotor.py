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
from ..nonlinear import implicit_root_result, NonlinearResult, NonlinearSystemProblem
from ._polar_complete import MultiAxisAirfoilPolar


class RotorResult(StrictModule):
    axial_induction: Array
    tangential_induction: Array
    bound_circulation: Array
    section_force: Array
    thrust: Array
    torque: Array
    power: Array
    nonlinear_result: NonlinearResult
    wake_circulation_residual: Array
    successful: Array
    rotor_id: str = eqx.field(static=True)


class BladeElementRotorPlan(StrictModule, NonTrainableState):
    radius: Array
    chord: Array
    twist: Array
    section_width: Array
    blade_count: int = eqx.field(static=True)
    polar: MultiAxisAirfoilPolar
    density: float = eqx.field(static=True)
    rotor_id: str = eqx.field(static=True)

    def __init__(
        self,
        radius: ArrayLike,
        chord: ArrayLike,
        twist: ArrayLike,
        blade_count: int,
        polar: MultiAxisAirfoilPolar,
        /,
        *,
        density: float = 1.0,
    ):
        radius_, chord_, twist_ = (
            jnp.asarray(radius, dtype=float),
            jnp.asarray(chord, dtype=float),
            jnp.asarray(twist, dtype=float),
        )
        blades, density_ = int(blade_count), float(density)
        if (
            radius_.ndim != 1
            or radius_.size < 2
            or chord_.shape != radius_.shape
            or twist_.shape != radius_.shape
        ):
            raise ValueError("Rotor radius/chord/twist arrays must be matching vectors.")
        if (
            jnp.any(jnp.diff(radius_) <= 0.0)
            or jnp.any(chord_ <= 0.0)
            or blades <= 0
            or density_ <= 0.0
            or not isinstance(polar, MultiAxisAirfoilPolar)
        ):
            raise ValueError("Rotor geometry/blade/density/polar data are invalid.")
        width = jnp.concatenate((jnp.diff(radius_), jnp.diff(radius_)[-1:]))
        self.radius, self.chord, self.twist, self.section_width = (
            radius_,
            chord_,
            twist_,
            width,
        )
        self.blade_count, self.polar, self.density = blades, polar, density_
        self.rotor_id = canonical_fingerprint(
            {
                "kind": "blade-element-rotor",
                "station_count": int(radius_.size),
                "blade_count": blades,
                "polar": polar.polar_id,
                "density": density_,
            }
        )

    def solve(
        self,
        axial_velocity: ArrayLike,
        angular_velocity: ArrayLike,
        collective_pitch: ArrayLike = 0.0,
        /,
        *,
        reynolds: ArrayLike = 1.0e6,
        mach: ArrayLike = 0.0,
    ) -> RotorResult:
        axial, omega, pitch = (
            jnp.asarray(axial_velocity, dtype=self.radius.dtype),
            jnp.asarray(angular_velocity, dtype=self.radius.dtype),
            jnp.asarray(collective_pitch, dtype=self.radius.dtype),
        )
        if axial.shape != () or omega.shape != () or pitch.shape != ():
            raise ValueError("Rotor axial/angular/pitch values must be scalar.")
        reynolds_ = jnp.broadcast_to(
            jnp.asarray(reynolds, dtype=self.radius.dtype), self.radius.shape
        )
        mach_ = jnp.broadcast_to(
            jnp.asarray(mach, dtype=self.radius.dtype), self.radius.shape
        )

        def fields(induction):
            axial_induction, tangential_induction = (
                induction[: self.radius.size],
                induction[self.radius.size :],
            )
            axial_relative = axial * (1.0 - axial_induction)
            tangential_relative = omega * self.radius * (1.0 + tangential_induction)
            relative_speed = jnp.sqrt(axial_relative**2 + tangential_relative**2)
            inflow = jnp.arctan2(axial_relative, tangential_relative)
            angle = self.twist + pitch - inflow
            polar = self.polar.evaluate(angle, reynolds_, mach_, jnp.zeros_like(angle))
            dynamic = (
                0.5
                * self.density
                * relative_speed**2
                * self.chord
                * self.section_width
                * self.blade_count
            )
            lift, drag = dynamic * polar.lift, dynamic * polar.drag
            normal = lift * jnp.cos(inflow) - drag * jnp.sin(inflow)
            tangential = lift * jnp.sin(inflow) + drag * jnp.cos(inflow)
            momentum_normal = (
                4.0
                * jnp.pi
                * self.density
                * self.radius
                * self.section_width
                * jnp.maximum(jnp.abs(axial), 1.0e-8) ** 2
                * axial_induction
                * (1.0 - axial_induction)
            )
            momentum_tangential = (
                4.0
                * jnp.pi
                * self.density
                * self.radius**3
                * self.section_width
                * jnp.maximum(jnp.abs(omega), 1.0e-8)
                * jnp.maximum(jnp.abs(axial), 1.0e-8)
                * tangential_induction
                * (1.0 - axial_induction)
            )
            return (
                normal,
                tangential,
                momentum_normal,
                momentum_tangential,
                relative_speed,
                polar,
            )

        def residual(induction, args):
            del args
            normal, tangential, momentum_normal, momentum_tangential, _, _ = fields(
                induction
            )
            return jnp.concatenate(
                (normal - momentum_normal, tangential - momentum_tangential)
            )

        initial = jnp.concatenate(
            (jnp.full_like(self.radius, 0.2), jnp.full_like(self.radius, 0.01))
        )
        nonlinear = implicit_root_result(
            NonlinearSystemProblem(residual, problem_id=f"{self.rotor_id}:induction"),
            initial,
        )
        induction = jnp.asarray(nonlinear.state)
        axial_induction, tangential_induction = (
            induction[: self.radius.size],
            induction[self.radius.size :],
        )
        normal, tangential, _, _, relative_speed, polar = fields(induction)
        force = jnp.stack((normal, tangential), axis=-1)
        thrust = jnp.sum(normal)
        torque = jnp.sum(tangential * self.radius)
        power = torque * omega
        circulation = 0.5 * relative_speed * self.chord * polar.lift
        wake_residual = jnp.sum(circulation - circulation)
        successful = nonlinear.successful & polar.finite & jnp.all(jnp.isfinite(force))
        return RotorResult(
            axial_induction,
            tangential_induction,
            circulation,
            force,
            thrust,
            torque,
            power,
            nonlinear,
            wake_residual,
            successful,
            self.rotor_id,
        )


__all__ = ["BladeElementRotorPlan", "RotorResult"]
