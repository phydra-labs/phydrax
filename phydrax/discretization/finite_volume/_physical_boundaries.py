#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._boundary import _boundary_value, AbstractFiniteVolumeBoundary


PrimitiveBoundaryTarget = Callable[[Array, Array, Array, Array, Any], ArrayLike]
ScalarBoundaryTarget = Callable[[Array, Array, Array, Array, Any], ArrayLike]


def _normal_velocity(velocity: Array, normal: Array, /) -> Array:
    return jnp.sum(velocity * normal, axis=-1)


def _replace_normal_velocity(
    velocity: Array,
    normal: Array,
    value: Array,
    /,
) -> Array:
    current = _normal_velocity(velocity, normal)
    return velocity + (value - current)[..., None] * normal


class SlipWallBoundary(AbstractFiniteVolumeBoundary):
    """Impermeable inviscid wall for Cartesian or mapped normals."""

    def __init__(self):
        self.boundary_id = canonical_fingerprint({"kind": "fv-slip-wall"})

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del time, coordinates, axis, args
        primitive = system.conserved_to_primitive(interior)
        velocity = primitive[..., 1:-1]
        reflected = (
            velocity
            - 2.0 * _normal_velocity(velocity, outward_normal)[..., None] * outward_normal
        )
        return system.primitive_to_conserved(primitive.at[..., 1:-1].set(reflected))


class NoSlipAdiabaticWallBoundary(AbstractFiniteVolumeBoundary):
    """No-slip wall with zero normal temperature gradient."""

    wall_velocity: Array

    def __init__(self, wall_velocity: ArrayLike, /):
        velocity = jnp.asarray(wall_velocity)
        if velocity.ndim != 1 or velocity.size == 0:
            raise ValueError("wall_velocity must be a non-empty component vector.")
        self.wall_velocity = velocity
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "fv-no-slip-adiabatic-wall",
                "wall_velocity": array_tree_fingerprint(velocity),
            }
        )

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del time, coordinates, outward_normal, axis, args
        primitive = system.conserved_to_primitive(interior)
        if self.wall_velocity.shape != (system.dimension,):
            raise ValueError("Wall velocity must match the system dimension.")
        exterior_velocity = 2.0 * self.wall_velocity - primitive[..., 1:-1]
        return system.primitive_to_conserved(
            primitive.at[..., 1:-1].set(exterior_velocity)
        )

    def heat_flux(self, interior: Array, /) -> Array:
        return jnp.zeros(interior.shape[:-1], dtype=interior.dtype)


class NoSlipIsothermalWallBoundary(AbstractFiniteVolumeBoundary):
    """No-slip wall with a prescribed face temperature."""

    wall_velocity: Array
    wall_temperature: Array

    def __init__(
        self,
        wall_velocity: ArrayLike,
        wall_temperature: ArrayLike,
        /,
    ):
        velocity = jnp.asarray(wall_velocity)
        temperature = jnp.asarray(wall_temperature).reshape(())
        if velocity.ndim != 1 or velocity.size == 0:
            raise ValueError("wall_velocity must be a non-empty component vector.")
        temperature = eqx.error_if(
            temperature,
            ~jnp.isfinite(temperature) | (temperature <= 0.0),
            "Wall temperature must be finite and positive.",
        )
        self.wall_velocity = velocity
        self.wall_temperature = temperature
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "fv-no-slip-isothermal-wall",
                "wall_velocity": array_tree_fingerprint(velocity),
                "wall_temperature": array_tree_fingerprint(temperature),
            }
        )

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del time, coordinates, outward_normal, axis, args
        primitive = system.conserved_to_primitive(interior)
        if self.wall_velocity.shape != (system.dimension,):
            raise ValueError("Wall velocity must match the system dimension.")
        interior_temperature = system.temperature(interior)
        exterior_temperature = 2.0 * self.wall_temperature - interior_temperature
        exterior_temperature = eqx.error_if(
            exterior_temperature,
            jnp.any(exterior_temperature <= 0.0),
            "Isothermal ghost temperature became non-positive.",
        )
        density = primitive[..., 0]
        pressure = density * system.material.gas_constant * exterior_temperature
        exterior_velocity = 2.0 * self.wall_velocity - primitive[..., 1:-1]
        exterior_primitive = primitive.at[..., 1:-1].set(exterior_velocity)
        exterior_primitive = exterior_primitive.at[..., -1].set(pressure)
        return system.primitive_to_conserved(exterior_primitive)


class PrescribedHeatFluxWallBoundary(AbstractFiniteVolumeBoundary):
    """No-slip wall with prescribed outward thermal heat flux."""

    wall_velocity: Array
    heat_flux_target: ScalarBoundaryTarget = eqx.field(static=True)

    def __init__(
        self,
        wall_velocity: ArrayLike,
        heat_flux_target: ScalarBoundaryTarget,
        /,
        *,
        boundary_id: str,
    ):
        velocity = jnp.asarray(wall_velocity)
        if (
            velocity.ndim != 1
            or velocity.size == 0
            or not callable(heat_flux_target)
            or not str(boundary_id)
        ):
            raise ValueError("Heat-flux wall requires velocity, target, and boundary_id.")
        self.wall_velocity = velocity
        self.heat_flux_target = heat_flux_target
        self.boundary_id = str(boundary_id)

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del time, coordinates, outward_normal, axis, args
        primitive = system.conserved_to_primitive(interior)
        exterior_velocity = 2.0 * self.wall_velocity - primitive[..., 1:-1]
        return system.primitive_to_conserved(
            primitive.at[..., 1:-1].set(exterior_velocity)
        )

    def normal_heat_flux(
        self,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        args: Any,
        /,
    ) -> Array:
        value = self.heat_flux_target(
            time,
            interior,
            coordinates,
            outward_normal,
            args,
        )
        return jnp.broadcast_to(jnp.asarray(value), interior.shape[:-1])


class SupersonicInflowBoundary(AbstractFiniteVolumeBoundary):
    """Prescribed full primitive inflow state."""

    target: PrimitiveBoundaryTarget = eqx.field(static=True)

    def __init__(self, target: PrimitiveBoundaryTarget, /, *, boundary_id: str):
        if not callable(target) or not str(boundary_id):
            raise ValueError("Supersonic inflow requires a target and boundary_id.")
        self.target = target
        self.boundary_id = str(boundary_id)

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        primitive = self.target(
            time,
            system.conserved_to_primitive(interior),
            coordinates,
            outward_normal,
            args,
        )
        del axis
        return system.primitive_to_conserved(_boundary_value(primitive, interior.shape))


class SupersonicOutflowBoundary(AbstractFiniteVolumeBoundary):
    """Pure extrapolation when every characteristic leaves the domain."""

    def __init__(self):
        self.boundary_id = canonical_fingerprint({"kind": "fv-supersonic-outflow"})

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del system, time, coordinates, outward_normal, axis, args
        return jnp.asarray(interior)


class CharacteristicInflowBoundary(AbstractFiniteVolumeBoundary):
    """Linearized characteristic projection of a target primitive state."""

    target: PrimitiveBoundaryTarget = eqx.field(static=True)

    def __init__(self, target: PrimitiveBoundaryTarget, /, *, boundary_id: str):
        if not callable(target) or not str(boundary_id):
            raise ValueError("Characteristic inflow requires target and boundary_id.")
        self.target = target
        self.boundary_id = str(boundary_id)

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        primitive = self.target(
            time,
            system.conserved_to_primitive(interior),
            coordinates,
            outward_normal,
            args,
        )
        target = system.primitive_to_conserved(_boundary_value(primitive, interior.shape))
        left_matrix, right_matrix, speeds = system.eigensystem(
            interior, target, axis, args
        )
        amplitudes = jnp.einsum("...ij,...j->...i", left_matrix, target - interior)
        outward_sign = jnp.sign(outward_normal[..., axis])
        incoming = speeds * outward_sign[..., None] < 0.0
        correction = jnp.einsum(
            "...ij,...j->...i",
            right_matrix,
            jnp.where(incoming, amplitudes, 0.0),
        )
        return interior + correction


class CharacteristicOutflowBoundary(AbstractFiniteVolumeBoundary):
    """Pressure-controlled linearized characteristic outflow."""

    pressure_target: ScalarBoundaryTarget = eqx.field(static=True)

    def __init__(
        self,
        pressure_target: ScalarBoundaryTarget,
        /,
        *,
        boundary_id: str,
    ):
        if not callable(pressure_target) or not str(boundary_id):
            raise ValueError("Characteristic outflow requires pressure and boundary_id.")
        self.pressure_target = pressure_target
        self.boundary_id = str(boundary_id)

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        primitive = system.conserved_to_primitive(interior)
        pressure = self.pressure_target(
            time, primitive, coordinates, outward_normal, args
        )
        target_primitive = primitive.at[..., -1].set(
            jnp.broadcast_to(jnp.asarray(pressure), primitive.shape[:-1])
        )
        target = system.primitive_to_conserved(target_primitive)
        left_matrix, right_matrix, speeds = system.eigensystem(
            interior, target, axis, args
        )
        amplitudes = jnp.einsum("...ij,...j->...i", left_matrix, target - interior)
        outward_sign = jnp.sign(outward_normal[..., axis])
        incoming = speeds * outward_sign[..., None] < 0.0
        correction = jnp.einsum(
            "...ij,...j->...i",
            right_matrix,
            jnp.where(incoming, amplitudes, 0.0),
        )
        return interior + correction


class FarFieldBoundary(AbstractFiniteVolumeBoundary):
    """Characteristic far-field state with incoming-wave replacement."""

    projector: CharacteristicInflowBoundary

    def __init__(self, target: PrimitiveBoundaryTarget, /, *, boundary_id: str):
        projector = CharacteristicInflowBoundary(target, boundary_id=boundary_id)
        self.projector = projector
        self.boundary_id = projector.boundary_id

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        return self.projector.exterior_state(
            system,
            time,
            interior,
            coordinates,
            outward_normal,
            axis,
            args,
        )


__all__ = [
    "CharacteristicInflowBoundary",
    "CharacteristicOutflowBoundary",
    "FarFieldBoundary",
    "NoSlipAdiabaticWallBoundary",
    "NoSlipIsothermalWallBoundary",
    "PrescribedHeatFluxWallBoundary",
    "SlipWallBoundary",
    "SupersonicInflowBoundary",
    "SupersonicOutflowBoundary",
]
