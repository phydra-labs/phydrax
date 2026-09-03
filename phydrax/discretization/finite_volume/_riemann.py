#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


if TYPE_CHECKING:
    from ...equations._entropy_pair import ConvexEntropyPair


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
    """One canonical normal flux density, signal bound, and fallback evidence."""

    normal_flux: Array
    max_speed: Array
    fallback_activated: Array

    def __init__(
        self,
        normal_flux: Array,
        max_speed: Array,
        /,
        *,
        fallback_activated: Array | None = None,
    ):
        flux = jnp.asarray(normal_flux)
        speed = jnp.asarray(max_speed)
        fallback = (
            jnp.zeros(speed.shape, dtype=bool)
            if fallback_activated is None
            else jnp.asarray(fallback_activated, dtype=bool)
        )
        if speed.shape != flux.shape[:-1] or fallback.shape != speed.shape:
            raise ValueError(
                "Numerical flux speed/fallback must match the face batch shape."
            )
        self.normal_flux = flux
        self.max_speed = speed
        self.fallback_activated = fallback


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


class AbstractArbitraryNormalNumericalFluxPlan(AbstractNumericalFluxPlan):
    """Typed capability for conservative fluxes on arbitrary unit normals."""

    @abc.abstractmethod
    def normal_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        raise NotImplementedError


class AbstractSymmetricTwoPointFluxPlan(AbstractArbitraryNormalNumericalFluxPlan):
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


class RusanovFluxPlan(AbstractArbitraryNormalNumericalFluxPlan):
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


class HLLFluxPlan(AbstractArbitraryNormalNumericalFluxPlan):
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


class HLLCFluxPlan(AbstractArbitraryNormalNumericalFluxPlan):
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


class HLLDFluxPlan(AbstractNumericalFluxPlan):
    """Five-wave HLLD flux for the canonical eight-component ideal-MHD layout."""

    denominator_epsilon: float = eqx.field(static=True)
    normal_field_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        denominator_epsilon: float = 1e-10,
        normal_field_tolerance: float = 1e-10,
    ):
        epsilon = float(denominator_epsilon)
        tolerance = float(normal_field_tolerance)
        if (
            not np.isfinite(epsilon)
            or epsilon <= 0.0
            or not np.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("HLLD tolerances are invalid.")
        self.denominator_epsilon = epsilon
        self.normal_field_tolerance = tolerance
        self.differentiability = "branchwise"
        self.flux_id = canonical_fingerprint(
            {
                "kind": "hlld-ideal-mhd-flux",
                "denominator_epsilon": epsilon,
                "normal_field_tolerance": tolerance,
            }
        )

    @staticmethod
    def _layout(system: Any, axis: int, /) -> tuple[int, int]:
        expected = (
            "density",
            "momentum_x",
            "momentum_y",
            "momentum_z",
            "total_energy",
            "magnetic_x",
            "magnetic_y",
            "magnetic_z",
        )
        if tuple(system.component_names) != expected:
            raise TypeError("HLLD requires the canonical ideal-MHD state layout.")
        axis_ = int(axis)
        if axis_ < 0 or axis_ >= system.dimension:
            raise ValueError("HLLD flux axis is out of range.")
        tangential = tuple(value for value in range(3) if value != axis_)
        return tangential[0], tangential[1]

    def _safe_denominator(self, value: Array, /) -> tuple[Array, Array]:
        small = jnp.abs(value) <= self.denominator_epsilon
        sign = jnp.where(value >= 0.0, 1.0, -1.0)
        return jnp.where(small, sign * self.denominator_epsilon, value), small

    @staticmethod
    def _fast_speed(primitive: Array, axis: int, gamma: float, /) -> Array:
        density = primitive[..., 0]
        pressure = primitive[..., 4]
        magnetic = primitive[..., 5:8]
        sound_squared = gamma * pressure / density
        magnetic_squared = jnp.sum(magnetic**2, axis=-1) / density
        normal_squared = magnetic[..., axis] ** 2 / density
        discriminant = jnp.maximum(
            (sound_squared + magnetic_squared) ** 2
            - 4.0 * sound_squared * normal_squared,
            0.0,
        )
        return jnp.sqrt(0.5 * (sound_squared + magnetic_squared + jnp.sqrt(discriminant)))

    @staticmethod
    def _state(
        density: Array,
        normal_velocity: Array,
        tangential_velocity: tuple[Array, Array],
        energy: Array,
        normal_magnetic: Array,
        tangential_magnetic: tuple[Array, Array],
        axis: int,
        tangential_axes: tuple[int, int],
        /,
    ) -> Array:
        velocity = jnp.zeros(density.shape + (3,), dtype=density.dtype)
        magnetic = jnp.zeros_like(velocity)
        velocity = velocity.at[..., axis].set(normal_velocity)
        velocity = velocity.at[..., tangential_axes[0]].set(tangential_velocity[0])
        velocity = velocity.at[..., tangential_axes[1]].set(tangential_velocity[1])
        magnetic = magnetic.at[..., axis].set(normal_magnetic)
        magnetic = magnetic.at[..., tangential_axes[0]].set(tangential_magnetic[0])
        magnetic = magnetic.at[..., tangential_axes[1]].set(tangential_magnetic[1])
        return jnp.concatenate(
            (
                density[..., None],
                density[..., None] * velocity,
                energy[..., None],
                magnetic,
            ),
            axis=-1,
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
        tangential_axes = self._layout(system, axis_)
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        primitive_left = system.conserved_to_primitive(left_)
        primitive_right = system.conserved_to_primitive(right_)
        rho_l = primitive_left[..., 0]
        rho_r = primitive_right[..., 0]
        velocity_l = primitive_left[..., 1:4]
        velocity_r = primitive_right[..., 1:4]
        pressure_l = primitive_left[..., 4]
        pressure_r = primitive_right[..., 4]
        magnetic_l = primitive_left[..., 5:8]
        magnetic_r = primitive_right[..., 5:8]
        vn_l = velocity_l[..., axis_]
        vn_r = velocity_r[..., axis_]
        bn_l = magnetic_l[..., axis_]
        bn_r = magnetic_r[..., axis_]
        bn = 0.5 * (bn_l + bn_r)
        normal_scale = 1.0 + jnp.maximum(jnp.abs(bn_l), jnp.abs(bn_r))
        inconsistent_bn = (
            jnp.abs(bn_l - bn_r) > self.normal_field_tolerance * normal_scale
        )
        cf_l = self._fast_speed(primitive_left, axis_, system.gamma)
        cf_r = self._fast_speed(primitive_right, axis_, system.gamma)
        s_l = jnp.minimum(vn_l - cf_l, vn_r - cf_r)
        s_r = jnp.maximum(vn_l + cf_l, vn_r + cf_r)
        flux_l = system.physical_flux(left_, axis_, args)
        flux_r = system.physical_flux(right_, axis_, args)
        total_pressure_l = pressure_l + 0.5 * jnp.sum(magnetic_l**2, axis=-1)
        total_pressure_r = pressure_r + 0.5 * jnp.sum(magnetic_r**2, axis=-1)
        contact_denominator_raw = rho_r * (s_r - vn_r) - rho_l * (s_l - vn_l)
        contact_denominator, bad_contact = self._safe_denominator(contact_denominator_raw)
        s_m = (
            rho_r * vn_r * (s_r - vn_r)
            - rho_l * vn_l * (s_l - vn_l)
            + total_pressure_l
            - total_pressure_r
        ) / contact_denominator
        total_pressure_star_l = total_pressure_l + rho_l * (s_l - vn_l) * (s_m - vn_l)
        total_pressure_star_r = total_pressure_r + rho_r * (s_r - vn_r) * (s_m - vn_r)
        total_pressure_star = 0.5 * (total_pressure_star_l + total_pressure_star_r)

        def star_state(
            state: Array,
            density: Array,
            velocity: Array,
            magnetic: Array,
            total_pressure: Array,
            signal: Array,
            normal_velocity: Array,
        ):
            density_denominator, bad_density = self._safe_denominator(signal - s_m)
            density_star = density * (signal - normal_velocity) / density_denominator
            transverse_denominator_raw = (
                density * (signal - normal_velocity) * (signal - s_m) - bn**2
            )
            transverse_denominator, bad_transverse = self._safe_denominator(
                transverse_denominator_raw
            )
            transverse_velocity = []
            transverse_magnetic = []
            for transverse_axis in tangential_axes:
                velocity_t = velocity[..., transverse_axis]
                magnetic_t = magnetic[..., transverse_axis]
                velocity_star_t = velocity_t - (
                    bn * magnetic_t * (s_m - normal_velocity) / transverse_denominator
                )
                magnetic_star_t = (
                    magnetic_t
                    * (density * (signal - normal_velocity) ** 2 - bn**2)
                    / transverse_denominator
                )
                transverse_velocity.append(velocity_star_t)
                transverse_magnetic.append(magnetic_star_t)
            velocity_star = jnp.zeros_like(velocity)
            magnetic_star = jnp.zeros_like(magnetic)
            velocity_star = velocity_star.at[..., axis_].set(s_m)
            magnetic_star = magnetic_star.at[..., axis_].set(bn)
            for index, transverse_axis in enumerate(tangential_axes):
                velocity_star = velocity_star.at[..., transverse_axis].set(
                    transverse_velocity[index]
                )
                magnetic_star = magnetic_star.at[..., transverse_axis].set(
                    transverse_magnetic[index]
                )
            velocity_dot_magnetic = jnp.sum(velocity * magnetic, axis=-1)
            star_dot = jnp.sum(velocity_star * magnetic_star, axis=-1)
            energy_numerator = (
                (signal - normal_velocity) * state[..., 4]
                - total_pressure * normal_velocity
                + total_pressure_star * s_m
                + bn * (velocity_dot_magnetic - star_dot)
            )
            energy_denominator, bad_energy = self._safe_denominator(signal - s_m)
            energy_star = energy_numerator / energy_denominator
            star = self._state(
                density_star,
                s_m,
                (transverse_velocity[0], transverse_velocity[1]),
                energy_star,
                bn,
                (transverse_magnetic[0], transverse_magnetic[1]),
                axis_,
                tangential_axes,
            )
            return (
                star,
                density_star,
                velocity_star,
                magnetic_star,
                bad_density | bad_transverse | bad_energy,
            )

        left_star, rho_l_star, velocity_l_star, magnetic_l_star, bad_l = star_state(
            left_,
            rho_l,
            velocity_l,
            magnetic_l,
            total_pressure_l,
            s_l,
            vn_l,
        )
        right_star, rho_r_star, velocity_r_star, magnetic_r_star, bad_r = star_state(
            right_,
            rho_r,
            velocity_r,
            magnetic_r,
            total_pressure_r,
            s_r,
            vn_r,
        )
        sqrt_l = jnp.sqrt(jnp.maximum(rho_l_star, self.denominator_epsilon))
        sqrt_r = jnp.sqrt(jnp.maximum(rho_r_star, self.denominator_epsilon))
        root_sum, bad_roots = self._safe_denominator(sqrt_l + sqrt_r)
        sign_bn = jnp.where(bn >= 0.0, 1.0, -1.0)
        velocity_ss = jnp.zeros_like(velocity_l_star)
        magnetic_ss = jnp.zeros_like(magnetic_l_star)
        velocity_ss = velocity_ss.at[..., axis_].set(s_m)
        magnetic_ss = magnetic_ss.at[..., axis_].set(bn)
        for transverse_axis in tangential_axes:
            velocity_value = (
                sqrt_l * velocity_l_star[..., transverse_axis]
                + sqrt_r * velocity_r_star[..., transverse_axis]
                + sign_bn
                * (
                    magnetic_r_star[..., transverse_axis]
                    - magnetic_l_star[..., transverse_axis]
                )
            ) / root_sum
            magnetic_value = (
                sqrt_l * magnetic_r_star[..., transverse_axis]
                + sqrt_r * magnetic_l_star[..., transverse_axis]
                + sign_bn
                * sqrt_l
                * sqrt_r
                * (
                    velocity_r_star[..., transverse_axis]
                    - velocity_l_star[..., transverse_axis]
                )
            ) / root_sum
            velocity_ss = velocity_ss.at[..., transverse_axis].set(velocity_value)
            magnetic_ss = magnetic_ss.at[..., transverse_axis].set(magnetic_value)
        dot_l_star = jnp.sum(velocity_l_star * magnetic_l_star, axis=-1)
        dot_r_star = jnp.sum(velocity_r_star * magnetic_r_star, axis=-1)
        dot_ss = jnp.sum(velocity_ss * magnetic_ss, axis=-1)
        energy_l_ss = left_star[..., 4] - sqrt_l * sign_bn * (dot_l_star - dot_ss)
        energy_r_ss = right_star[..., 4] + sqrt_r * sign_bn * (dot_r_star - dot_ss)
        left_ss = self._state(
            rho_l_star,
            s_m,
            (
                velocity_ss[..., tangential_axes[0]],
                velocity_ss[..., tangential_axes[1]],
            ),
            energy_l_ss,
            bn,
            (
                magnetic_ss[..., tangential_axes[0]],
                magnetic_ss[..., tangential_axes[1]],
            ),
            axis_,
            tangential_axes,
        )
        right_ss = self._state(
            rho_r_star,
            s_m,
            (
                velocity_ss[..., tangential_axes[0]],
                velocity_ss[..., tangential_axes[1]],
            ),
            energy_r_ss,
            bn,
            (
                magnetic_ss[..., tangential_axes[0]],
                magnetic_ss[..., tangential_axes[1]],
            ),
            axis_,
            tangential_axes,
        )
        s_l_star = s_m - jnp.abs(bn) / sqrt_l
        s_r_star = s_m + jnp.abs(bn) / sqrt_r
        flux_l_star = flux_l + s_l[..., None] * (left_star - left_)
        flux_r_star = flux_r + s_r[..., None] * (right_star - right_)
        flux_l_ss = flux_l_star + s_l_star[..., None] * (left_ss - left_star)
        flux_r_ss = flux_r_star + s_r_star[..., None] * (right_ss - right_star)
        hlld_flux = jnp.where(
            (s_l >= 0.0)[..., None],
            flux_l,
            jnp.where(
                (s_l_star >= 0.0)[..., None],
                flux_l_star,
                jnp.where(
                    (s_m >= 0.0)[..., None],
                    flux_l_ss,
                    jnp.where(
                        (s_r_star > 0.0)[..., None],
                        flux_r_ss,
                        jnp.where((s_r > 0.0)[..., None], flux_r_star, flux_r),
                    ),
                ),
            ),
        )
        intermediate_valid = (
            system.admissible(left_star)
            & system.admissible(right_star)
            & system.admissible(left_ss)
            & system.admissible(right_ss)
        )
        finite = (
            jnp.all(jnp.isfinite(left_star), axis=-1)
            & jnp.all(jnp.isfinite(right_star), axis=-1)
            & jnp.all(jnp.isfinite(left_ss), axis=-1)
            & jnp.all(jnp.isfinite(right_ss), axis=-1)
            & jnp.all(jnp.isfinite(hlld_flux), axis=-1)
        )
        weak_normal = jnp.abs(bn) <= self.denominator_epsilon
        fallback = (
            inconsistent_bn
            | bad_contact
            | bad_l
            | bad_r
            | bad_roots
            | weak_normal
            | ~intermediate_valid
            | ~finite
        )
        hll = HLLFluxPlan().face_flux(system, left_, right_, axis_, args)
        flux = jnp.where(fallback[..., None], hll.normal_flux, hlld_flux)
        return NumericalFluxResult(
            flux,
            hll.max_speed,
            fallback_activated=fallback,
        )


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
        characteristic = ein.contract("...ij,...j->...i", left_matrix, jump)
        dissipation = ein.contract(
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
    safe_log_difference = jnp.where(near, jnp.ones_like(log_difference), log_difference)
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

    def normal_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        normal_ = jnp.asarray(normal)
        flux = jnp.stack(
            tuple(
                self.two_point_flux(system, left, right, axis, args)
                for axis in range(system.dimension)
            ),
            axis=-1,
        )
        contracted = ein.contract("...id,...d->...i", flux, normal_, backend="jax")
        speed = system.max_normal_wave_speed(left, right, normal_, args)
        return NumericalFluxResult(contracted, speed)


class EntropyStableFluxPlan(AbstractArbitraryNormalNumericalFluxPlan):
    """Pair-bound entropy-conservative flux with scalar entropy dissipation."""

    central: AbstractSymmetricTwoPointFluxPlan
    entropy_pair: ConvexEntropyPair
    dissipation: float = eqx.field(static=True)

    def __init__(
        self,
        central: AbstractSymmetricTwoPointFluxPlan,
        entropy_pair: ConvexEntropyPair,
        /,
        *,
        dissipation: float = 1.0,
    ):
        from ...equations._entropy_pair import ConvexEntropyPair

        if not isinstance(central, AbstractSymmetricTwoPointFluxPlan):
            raise TypeError("central must be a symmetric two-point flux plan.")
        if not central.symmetric or not central.consistent:
            raise ValueError(
                "Entropy-stable central flux must be symmetric and consistent."
            )
        if not isinstance(entropy_pair, ConvexEntropyPair):
            raise TypeError("entropy_pair must be ConvexEntropyPair.")
        coefficient = float(dissipation)
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("dissipation must be finite and non-negative.")
        self.central = central
        self.entropy_pair = entropy_pair
        self.dissipation = coefficient
        self.differentiability = "almost_everywhere"
        self.flux_id = canonical_fingerprint(
            {
                "kind": "pair-bound-entropy-stable-flux",
                "central": central.flux_id,
                "entropy_pair": entropy_pair.pair_id,
                "dissipation": coefficient,
            }
        )

    def _validate_system(self, system: Any, /) -> None:
        if system.system_id != self.entropy_pair.system.system_id:
            raise ValueError("Entropy-stable flux system differs from its entropy pair.")

    def face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        self._validate_system(system)
        central = self.central.face_flux(system, left, right, axis, args)
        flux = central.normal_flux - 0.5 * self.dissipation * central.max_speed[
            ..., None
        ] * (right - left)
        return NumericalFluxResult(flux, central.max_speed)

    def normal_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        self._validate_system(system)
        normal_ = jnp.asarray(normal)
        if normal_.shape[-1:] != (system.dimension,):
            raise ValueError("Entropy-stable flux normal has the wrong dimension.")
        fluxes = jnp.stack(
            tuple(
                self.central.two_point_flux(system, left, right, axis, args)
                for axis in range(system.dimension)
            ),
            axis=-1,
        )
        central = ein.contract("...id,...d->...i", fluxes, normal_, backend="jax")
        speed = system.max_normal_wave_speed(left, right, normal_, args)
        flux = central - 0.5 * self.dissipation * speed[..., None] * (right - left)
        return NumericalFluxResult(flux, speed)

    def entropy_dissipation(
        self,
        left: Array,
        right: Array,
        speed: Array,
        /,
    ) -> Array:
        jump = self.entropy_pair.entropy_variables(right) - (
            self.entropy_pair.entropy_variables(left)
        )
        return (
            -0.5
            * self.dissipation
            * jnp.asarray(speed)
            * ein.contract("...i,...i->...", jump, right - left, backend="jax")
        )


class EntropyStableEulerFluxPlan(AbstractArbitraryNormalNumericalFluxPlan):
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

    def normal_face_flux(
        self,
        system: Any,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> NumericalFluxResult:
        central = self.central.normal_face_flux(system, left, right, normal, args)
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
    "AbstractArbitraryNormalNumericalFluxPlan",
    "AbstractNumericalFluxPlan",
    "AbstractSymmetricTwoPointFluxPlan",
    "EntropyConservativeEulerFluxPlan",
    "EntropyStableFluxPlan",
    "EntropyStableEulerFluxPlan",
    "HLLCFluxPlan",
    "HLLFluxPlan",
    "NumericalFluxResult",
    "RoeFluxPlan",
    "RusanovFluxPlan",
]
