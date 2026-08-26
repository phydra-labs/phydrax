#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_NORMAL_ALE_CONTRACT = "physical-normal-flux-minus-grid-transport-relative-waves-v1"


def _normal_ale_inputs(
    left: Array,
    right: Array,
    normal: Array,
    grid_normal_velocity: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    left_ = jnp.asarray(left)
    right_ = jnp.asarray(right)
    normal_ = jnp.asarray(normal)
    grid_velocity = jnp.asarray(grid_normal_velocity)
    if grid_velocity.shape != left_.shape[:-1]:
        raise ValueError(
            "grid_normal_velocity must exactly match the face batch shape; "
            "scalar broadcasting is not permitted."
        )
    if not jnp.issubdtype(grid_velocity.dtype, jnp.floating):
        raise TypeError("grid_normal_velocity must have a real floating dtype.")
    grid_velocity = eqx.error_if(
        grid_velocity,
        jnp.any(~jnp.isfinite(grid_velocity)),
        "grid_normal_velocity must be finite.",
    )
    return left_, right_, normal_, grid_velocity


class NumericalFluxResult(StrictModule):
    """One canonical normal flux density and its signal-speed bound."""

    normal_flux: Array
    max_speed: Array

    def __init__(self, normal_flux: Array, max_speed: Array, /):
        flux = jnp.asarray(normal_flux)
        speed = jnp.asarray(max_speed)
        if speed.shape != flux.shape[:-1]:
            raise ValueError("Numerical flux speed must match the face batch shape.")
        self.normal_flux = flux
        self.max_speed = speed


class AbstractNumericalFluxPlan(StrictModule, NonTrainableState):
    """Interface solver returning a conservative normal flux density."""

    flux_id: str = eqx.field(static=True)
    differentiability: str = eqx.field(static=True)

    @abc.abstractmethod
    def face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        raise NotImplementedError


class AbstractSymmetricTwoPointFluxPlan(AbstractNumericalFluxPlan):
    """Symmetric consistent flux reusable in interface and volume methods."""

    symmetric: bool = eqx.field(static=True)
    consistent: bool = eqx.field(static=True)

    @abc.abstractmethod
    def two_point_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        raise NotImplementedError



class RusanovFluxPlan(AbstractNumericalFluxPlan):
    """Local Lax–Friedrichs flux with optional smooth wave-speed magnitude."""

    smooth_epsilon: float = eqx.field(static=True)

    def __init__(self, *, smooth_epsilon: float = 0.0):
        epsilon = float(smooth_epsilon)
        if not np.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("smooth_epsilon must be finite and non-negative.")
        self.smooth_epsilon = epsilon
        self.differentiability = (
            "almost_everywhere" if epsilon == 0.0 else "smooth_surrogate"
        )
        self.flux_id = canonical_fingerprint(
            {
                "kind": "rusanov-flux",
                "smooth_epsilon": epsilon,
                "normal_ale_contract": _NORMAL_ALE_CONTRACT,
            }
        )

    def face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        speed = jnp.asarray(system.max_wave_speed(left_, right_, int(axis), args))
        if self.smooth_epsilon > 0.0:
            speed = jnp.sqrt(speed**2 + self.smooth_epsilon**2)
        flux = 0.5 * (
            system.physical_flux(left_, int(axis), args)
            + system.physical_flux(right_, int(axis), args)
        ) - 0.5 * speed[..., None] * (right_ - left_)
        return NumericalFluxResult(flux, speed)

    def normal_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        normal_ = jnp.asarray(normal)
        speed = system.max_normal_wave_speed(left_, right_, normal_, args)
        if self.smooth_epsilon > 0.0:
            speed = jnp.sqrt(speed**2 + self.smooth_epsilon**2)
        flux = 0.5 * (
            system.physical_normal_flux(left_, normal_, args)
            + system.physical_normal_flux(right_, normal_, args)
        ) - 0.5 * speed[..., None] * (right_ - left_)
        return NumericalFluxResult(flux, speed)

    def normal_ale_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        grid_normal_velocity: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        left_, right_, normal_, grid_velocity = _normal_ale_inputs(
            left, right, normal, grid_normal_velocity
        )
        lower, upper = system.normal_signal_bounds(left_, right_, normal_, args)
        relative_lower = jnp.asarray(lower) - grid_velocity
        relative_upper = jnp.asarray(upper) - grid_velocity
        speed = jnp.maximum(jnp.abs(relative_lower), jnp.abs(relative_upper))
        if self.smooth_epsilon > 0.0:
            speed = jnp.sqrt(speed**2 + self.smooth_epsilon**2)
        transport_state = 0.5 * (left_ + right_)
        flux = (
            0.5
            * (
                system.physical_normal_flux(left_, normal_, args)
                + system.physical_normal_flux(right_, normal_, args)
            )
            - grid_velocity[..., None] * transport_state
            - 0.5 * speed[..., None] * (right_ - left_)
        )
        return NumericalFluxResult(flux, speed)


class HLLFluxPlan(AbstractNumericalFluxPlan):
    """Two-wave Harten–Lax–van Leer numerical flux."""

    def __init__(self):
        self.differentiability = "almost_everywhere"
        self.flux_id = canonical_fingerprint(
            {"kind": "hll-flux", "normal_ale_contract": _NORMAL_ALE_CONTRACT}
        )

    def face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        lower, upper = system.signal_bounds(left_, right_, int(axis), args)
        lower = jnp.minimum(jnp.asarray(lower), 0.0)
        upper = jnp.maximum(jnp.asarray(upper), 0.0)
        left_flux = system.physical_flux(left_, int(axis), args)
        right_flux = system.physical_flux(right_, int(axis), args)
        denominator = upper - lower
        middle = (
            upper[..., None] * left_flux
            - lower[..., None] * right_flux
            + (lower * upper)[..., None] * (right_ - left_)
        ) / jnp.where(denominator == 0.0, 1.0, denominator)[..., None]
        flux = jnp.where(
            (lower >= 0.0)[..., None],
            left_flux,
            jnp.where((upper <= 0.0)[..., None], right_flux, middle),
        )
        speed = jnp.maximum(jnp.abs(lower), jnp.abs(upper))
        return NumericalFluxResult(flux, speed)

    def normal_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        normal_ = jnp.asarray(normal)
        lower, upper = system.normal_signal_bounds(left_, right_, normal_, args)
        lower = jnp.minimum(jnp.asarray(lower), 0.0)
        upper = jnp.maximum(jnp.asarray(upper), 0.0)
        left_flux = system.physical_normal_flux(left_, normal_, args)
        right_flux = system.physical_normal_flux(right_, normal_, args)
        denominator = upper - lower
        middle = (
            upper[..., None] * left_flux
            - lower[..., None] * right_flux
            + (lower * upper)[..., None] * (right_ - left_)
        ) / jnp.where(denominator == 0.0, 1.0, denominator)[..., None]
        flux = jnp.where(
            (lower >= 0.0)[..., None],
            left_flux,
            jnp.where((upper <= 0.0)[..., None], right_flux, middle),
        )
        return NumericalFluxResult(flux, jnp.maximum(jnp.abs(lower), jnp.abs(upper)))

    def normal_ale_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        grid_normal_velocity: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        left_, right_, normal_, grid_velocity = _normal_ale_inputs(
            left, right, normal, grid_normal_velocity
        )
        lower, upper = system.normal_signal_bounds(left_, right_, normal_, args)
        lower = jnp.minimum(jnp.asarray(lower) - grid_velocity, 0.0)
        upper = jnp.maximum(jnp.asarray(upper) - grid_velocity, 0.0)
        left_flux = (
            system.physical_normal_flux(left_, normal_, args)
            - grid_velocity[..., None] * left_
        )
        right_flux = (
            system.physical_normal_flux(right_, normal_, args)
            - grid_velocity[..., None] * right_
        )
        denominator = upper - lower
        middle = (
            upper[..., None] * left_flux
            - lower[..., None] * right_flux
            + (lower * upper)[..., None] * (right_ - left_)
        ) / jnp.where(denominator == 0.0, 1.0, denominator)[..., None]
        flux = jnp.where(
            (lower >= 0.0)[..., None],
            left_flux,
            jnp.where((upper <= 0.0)[..., None], right_flux, middle),
        )
        return NumericalFluxResult(flux, jnp.maximum(jnp.abs(lower), jnp.abs(upper)))


class HLLCFluxPlan(AbstractNumericalFluxPlan):
    """Contact-resolving HLLC flux for Euler-compatible state layouts."""

    def __init__(self):
        self.differentiability = "almost_everywhere"
        self.flux_id = canonical_fingerprint(
            {"kind": "hllc-euler-flux", "normal_ale_contract": _NORMAL_ALE_CONTRACT}
        )

    def face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        axis_ = int(axis)
        if system.component_count != system.dimension + 2:
            raise TypeError("HLLC requires an Euler-compatible state layout.")
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        primitive_left = system.conserved_to_primitive(left_)
        primitive_right = system.conserved_to_primitive(right_)
        density_left = primitive_left[..., 0]
        density_right = primitive_right[..., 0]
        velocity_left = primitive_left[..., 1:-1]
        velocity_right = primitive_right[..., 1:-1]
        pressure_left = primitive_left[..., -1]
        pressure_right = primitive_right[..., -1]
        normal_left = velocity_left[..., axis_]
        normal_right = velocity_right[..., axis_]
        lower, upper = system.signal_bounds(left_, right_, axis_, args)
        denominator = density_left * (lower - normal_left) - density_right * (
            upper - normal_right
        )
        contact = (
            pressure_right
            - pressure_left
            + density_left * normal_left * (lower - normal_left)
            - density_right * normal_right * (upper - normal_right)
        ) / jnp.where(denominator == 0.0, 1.0, denominator)

        def star_state(
            state: Array,
            density: Array,
            velocity: Array,
            pressure: Array,
            signal: Array,
            normal_velocity: Array,
        ) -> Array:
            star_density = density * (signal - normal_velocity) / (signal - contact)
            star_velocity = velocity.at[..., axis_].set(contact)
            specific_energy = state[..., -1] / density
            star_energy = star_density * (
                specific_energy
                + (contact - normal_velocity)
                * (contact + pressure / (density * (signal - normal_velocity)))
            )
            return jnp.concatenate(
                (
                    star_density[..., None],
                    star_density[..., None] * star_velocity,
                    star_energy[..., None],
                ),
                axis=-1,
            )

        left_star = star_state(
            left_, density_left, velocity_left, pressure_left, lower, normal_left
        )
        right_star = star_state(
            right_, density_right, velocity_right, pressure_right, upper, normal_right
        )
        left_flux = system.physical_flux(left_, axis_, args)
        right_flux = system.physical_flux(right_, axis_, args)
        left_star_flux = left_flux + lower[..., None] * (left_star - left_)
        right_star_flux = right_flux + upper[..., None] * (right_star - right_)
        flux = jnp.where(
            (lower >= 0.0)[..., None],
            left_flux,
            jnp.where(
                (contact >= 0.0)[..., None],
                left_star_flux,
                jnp.where((upper > 0.0)[..., None], right_star_flux, right_flux),
            ),
        )
        speed = jnp.maximum(jnp.abs(lower), jnp.abs(upper))
        return NumericalFluxResult(flux, speed)

    def normal_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        if system.component_count != system.dimension + 2:
            raise TypeError("HLLC requires an Euler-compatible state layout.")
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        normal_ = jnp.asarray(normal)
        primitive_left = system.conserved_to_primitive(left_)
        primitive_right = system.conserved_to_primitive(right_)
        density_left = primitive_left[..., 0]
        density_right = primitive_right[..., 0]
        velocity_left = primitive_left[..., 1:-1]
        velocity_right = primitive_right[..., 1:-1]
        pressure_left = primitive_left[..., -1]
        pressure_right = primitive_right[..., -1]
        normal_left = jnp.sum(velocity_left * normal_, axis=-1)
        normal_right = jnp.sum(velocity_right * normal_, axis=-1)
        lower, upper = system.normal_signal_bounds(left_, right_, normal_, args)
        denominator = density_left * (lower - normal_left) - density_right * (
            upper - normal_right
        )
        contact = (
            pressure_right
            - pressure_left
            + density_left * normal_left * (lower - normal_left)
            - density_right * normal_right * (upper - normal_right)
        ) / jnp.where(denominator == 0.0, 1.0, denominator)

        def star_state(
            state: Array,
            density: Array,
            velocity: Array,
            pressure: Array,
            signal: Array,
            normal_velocity: Array,
        ) -> Array:
            star_density = density * (signal - normal_velocity) / (signal - contact)
            star_velocity = velocity + (contact - normal_velocity)[..., None] * normal_
            specific_energy = state[..., -1] / density
            star_energy = star_density * (
                specific_energy
                + (contact - normal_velocity)
                * (contact + pressure / (density * (signal - normal_velocity)))
            )
            return jnp.concatenate(
                (
                    star_density[..., None],
                    star_density[..., None] * star_velocity,
                    star_energy[..., None],
                ),
                axis=-1,
            )

        left_star = star_state(
            left_,
            density_left,
            velocity_left,
            pressure_left,
            lower,
            normal_left,
        )
        right_star = star_state(
            right_,
            density_right,
            velocity_right,
            pressure_right,
            upper,
            normal_right,
        )
        left_flux = system.physical_normal_flux(left_, normal_, args)
        right_flux = system.physical_normal_flux(right_, normal_, args)
        left_star_flux = left_flux + lower[..., None] * (left_star - left_)
        right_star_flux = right_flux + upper[..., None] * (right_star - right_)
        flux = jnp.where(
            (lower >= 0.0)[..., None],
            left_flux,
            jnp.where(
                (contact >= 0.0)[..., None],
                left_star_flux,
                jnp.where(
                    (upper > 0.0)[..., None],
                    right_star_flux,
                    right_flux,
                ),
            ),
        )
        return NumericalFluxResult(flux, jnp.maximum(jnp.abs(lower), jnp.abs(upper)))

    def normal_ale_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        grid_normal_velocity: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        if system.component_count != system.dimension + 2:
            raise TypeError("HLLC requires an Euler-compatible state layout.")
        left_, right_, normal_, grid_velocity = _normal_ale_inputs(
            left, right, normal, grid_normal_velocity
        )
        primitive_left = system.conserved_to_primitive(left_)
        primitive_right = system.conserved_to_primitive(right_)
        density_left = primitive_left[..., 0]
        density_right = primitive_right[..., 0]
        velocity_left = primitive_left[..., 1:-1]
        velocity_right = primitive_right[..., 1:-1]
        pressure_left = primitive_left[..., -1]
        pressure_right = primitive_right[..., -1]
        normal_left = jnp.sum(velocity_left * normal_, axis=-1)
        normal_right = jnp.sum(velocity_right * normal_, axis=-1)
        lower, upper = system.normal_signal_bounds(left_, right_, normal_, args)
        denominator = density_left * (lower - normal_left) - density_right * (
            upper - normal_right
        )
        contact = (
            pressure_right
            - pressure_left
            + density_left * normal_left * (lower - normal_left)
            - density_right * normal_right * (upper - normal_right)
        ) / jnp.where(denominator == 0.0, 1.0, denominator)

        def star_state(
            state: Array,
            density: Array,
            velocity: Array,
            pressure: Array,
            signal: Array,
            normal_velocity: Array,
        ) -> Array:
            star_density = density * (signal - normal_velocity) / (signal - contact)
            star_velocity = velocity + (contact - normal_velocity)[..., None] * normal_
            specific_energy = state[..., -1] / density
            star_energy = star_density * (
                specific_energy
                + (contact - normal_velocity)
                * (contact + pressure / (density * (signal - normal_velocity)))
            )
            return jnp.concatenate(
                (
                    star_density[..., None],
                    star_density[..., None] * star_velocity,
                    star_energy[..., None],
                ),
                axis=-1,
            )

        left_star = star_state(
            left_,
            density_left,
            velocity_left,
            pressure_left,
            lower,
            normal_left,
        )
        right_star = star_state(
            right_,
            density_right,
            velocity_right,
            pressure_right,
            upper,
            normal_right,
        )
        relative_lower = lower - grid_velocity
        relative_upper = upper - grid_velocity
        relative_contact = contact - grid_velocity
        left_flux = (
            system.physical_normal_flux(left_, normal_, args)
            - grid_velocity[..., None] * left_
        )
        right_flux = (
            system.physical_normal_flux(right_, normal_, args)
            - grid_velocity[..., None] * right_
        )
        left_star_flux = left_flux + relative_lower[..., None] * (left_star - left_)
        right_star_flux = right_flux + relative_upper[..., None] * (right_star - right_)
        flux = jnp.where(
            (relative_lower >= 0.0)[..., None],
            left_flux,
            jnp.where(
                (relative_contact >= 0.0)[..., None],
                left_star_flux,
                jnp.where(
                    (relative_upper > 0.0)[..., None],
                    right_star_flux,
                    right_flux,
                ),
            ),
        )
        speed = jnp.maximum(jnp.abs(relative_lower), jnp.abs(relative_upper))
        return NumericalFluxResult(flux, speed)


class RoeFluxPlan(AbstractNumericalFluxPlan):
    """Roe characteristic flux with a quadratic entropy fix."""

    entropy_fix: float = eqx.field(static=True)

    def __init__(self, *, entropy_fix: float = 0.1):
        fix = float(entropy_fix)
        if not np.isfinite(fix) or fix <= 0.0:
            raise ValueError("entropy_fix must be finite and positive.")
        self.entropy_fix = fix
        self.differentiability = "almost_everywhere"
        self.flux_id = canonical_fingerprint({"kind": "roe-flux", "entropy_fix": fix})

    def face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        left_matrix, right_matrix, eigenvalues = system.eigensystem(
            left_, right_, int(axis), args
        )
        absolute = jnp.abs(eigenvalues)
        delta = self.entropy_fix * jnp.maximum(
            jnp.max(absolute, axis=-1, keepdims=True), 1.0
        )
        fixed = jnp.where(
            absolute < delta,
            0.5 * (absolute**2 / delta + delta),
            absolute,
        )
        jump = right_ - left_
        characteristic = oe.contract("...ij,...j->...i", left_matrix, jump)
        dissipation = oe.contract(
            "...ij,...j->...i", right_matrix, fixed * characteristic
        )
        flux = 0.5 * (
            system.physical_flux(left_, int(axis), args)
            + system.physical_flux(right_, int(axis), args)
            - dissipation
        )
        return NumericalFluxResult(flux, jnp.max(absolute, axis=-1))


def _logarithmic_mean(left: Array, right: Array, /) -> Array:
    average = 0.5 * (left + right)
    difference = right - left
    log_difference = jnp.log(right) - jnp.log(left)
    near = jnp.abs(difference) <= 1e-7 * jnp.abs(average)
    safe_difference = jnp.where(near, jnp.ones_like(difference), difference)
    safe_log_difference = jnp.where(
        near, jnp.ones_like(log_difference), log_difference
    )
    ratio = safe_difference / safe_log_difference
    return jnp.where(near, average, ratio)


class EntropyConservativeEulerFluxPlan(AbstractSymmetricTwoPointFluxPlan):
    """Chandrashekar-type symmetric entropy-conservative Euler flux."""

    def __init__(self):
        self.symmetric = True
        self.consistent = True
        self.differentiability = "smooth_discrete"
        self.flux_id = canonical_fingerprint(
            {
                "kind": "entropy-conservative-euler-flux-v2",
                "symmetric": True,
                "consistent": True,
            }
        )

    def two_point_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        del args
        axis_ = int(axis)
        if system.component_count != system.dimension + 2:
            raise TypeError("Entropy-conservative Euler flux requires Euler layout.")
        primitive_left = system.conserved_to_primitive(left)
        primitive_right = system.conserved_to_primitive(right)
        rho_left, rho_right = primitive_left[..., 0], primitive_right[..., 0]
        velocity_left, velocity_right = (
            primitive_left[..., 1:-1],
            primitive_right[..., 1:-1],
        )
        pressure_left, pressure_right = primitive_left[..., -1], primitive_right[..., -1]
        beta_left = rho_left / (2.0 * pressure_left)
        beta_right = rho_right / (2.0 * pressure_right)
        rho_log = _logarithmic_mean(rho_left, rho_right)
        beta_log = _logarithmic_mean(beta_left, beta_right)
        rho_average = 0.5 * (rho_left + rho_right)
        beta_average = 0.5 * (beta_left + beta_right)
        velocity_average = 0.5 * (velocity_left + velocity_right)
        velocity_square_average = 0.5 * (velocity_left**2 + velocity_right**2)
        pressure_average = rho_average / (2.0 * beta_average)
        mass_flux = rho_log * velocity_average[..., axis_]
        momentum_flux = mass_flux[..., None] * velocity_average
        momentum_flux = momentum_flux.at[..., axis_].add(pressure_average)
        internal_energy = 1.0 / (2.0 * (system.gamma - 1.0) * beta_log)
        energy_flux = mass_flux * (
            internal_energy - 0.5 * jnp.sum(velocity_square_average, axis=-1)
        ) + jnp.sum(velocity_average * momentum_flux, axis=-1)
        return jnp.concatenate(
            (mass_flux[..., None], momentum_flux, energy_flux[..., None]), axis=-1
        )

    def face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        flux = self.two_point_flux(system, left, right, axis, args)
        speed = system.max_wave_speed(left, right, int(axis), args)
        return NumericalFluxResult(flux, speed)


class EntropyStableEulerFluxPlan(AbstractNumericalFluxPlan):
    """Entropy-conservative central flux with Rusanov state dissipation."""

    central: EntropyConservativeEulerFluxPlan
    dissipation: float = eqx.field(static=True)

    def __init__(self, *, dissipation: float = 1.0):
        coefficient = float(dissipation)
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("dissipation must be finite and non-negative.")
        self.central = EntropyConservativeEulerFluxPlan()
        self.dissipation = coefficient
        self.differentiability = "almost_everywhere"
        self.flux_id = canonical_fingerprint(
            {"kind": "entropy-stable-euler-flux", "dissipation": coefficient}
        )

    def face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        central = self.central.face_flux(system, left, right, axis, args)
        flux = central.normal_flux - 0.5 * self.dissipation * central.max_speed[
            ..., None
        ] * (right - left)
        return NumericalFluxResult(flux, central.max_speed)

    def entropy_dissipation(self, system: Any, left: Array, right: Array, /) -> Array:
        speed = system.max_wave_speed(left, right, 0, None)
        entropy_jump = system.entropy_variables(right) - system.entropy_variables(left)
        return (
            -0.5
            * self.dissipation
            * speed
            * jnp.sum(entropy_jump * (right - left), axis=-1)
        )


__all__ = [
    "AbstractNumericalFluxPlan",
    "AbstractSymmetricTwoPointFluxPlan",
    "EntropyConservativeEulerFluxPlan",
    "EntropyStableEulerFluxPlan",
    "HLLCFluxPlan",
    "HLLFluxPlan",
    "NumericalFluxResult",
    "RoeFluxPlan",
    "RusanovFluxPlan",
]
