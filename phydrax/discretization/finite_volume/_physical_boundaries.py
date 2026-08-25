#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._boundary import (
    _boundary_value,
    _require_axis_aligned_ale_normal,
    _static_ale_exterior_state,
    _validate_ale_boundary_context,
    AbstractFiniteVolumeBoundary,
    ALEBoundaryContext,
)


PrimitiveBoundaryTarget = Callable[[Array, Array, Array, Array, Any], ArrayLike]
ScalarBoundaryTarget = Callable[[Array, Array, Array, Array, Any], ArrayLike]
WallVelocityProvider = Callable[[Array, Array, Array, Any], ArrayLike]


def _normal_velocity(velocity: Array, normal: Array, /) -> Array:
    return oe.contract("...i,...i->...", velocity, normal)


def _primitive_velocity(system: Any, primitive: Array, /) -> Array:
    from ...equations._multiphase import TwoMaterialVOFSystem

    if isinstance(system, TwoMaterialVOFSystem):
        return system.primitive_velocity(primitive)
    return primitive[..., 1:-1]


def _with_primitive_velocity(system: Any, primitive: Array, velocity: Array, /) -> Array:
    from ...equations._multiphase import TwoMaterialVOFSystem

    if isinstance(system, TwoMaterialVOFSystem):
        return system.with_primitive_velocity(primitive, velocity)
    return primitive.at[..., 1:-1].set(velocity)


def _replace_normal_velocity(
    velocity: Array,
    normal: Array,
    value: Array,
    /,
) -> Array:
    current = _normal_velocity(velocity, normal)
    return velocity + (value - current)[..., None] * normal


def _wall_velocity_value(
    value: ArrayLike,
    shape: tuple[int, ...],
    /,
) -> Array:
    velocity = jnp.asarray(value)
    if velocity.shape == (shape[-1],):
        return jnp.broadcast_to(velocity, shape)
    if velocity.shape != shape:
        raise ValueError(
            "Wall velocity provider must return one vector or one vector per "
            "boundary quadrature point."
        )
    return velocity


def _reject_unsupported_ale_wall(
    system: Any,
    interior: Array,
    context: ALEBoundaryContext,
    axis: int,
    /,
) -> Array:
    _validate_ale_boundary_context(system, interior, context, axis)
    raise ValueError(
        "ALE no-slip and thermal wall combinations are unsupported; "
        "use MovingSlipWallBoundary with an Euler-compatible system."
    )


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
        velocity = _primitive_velocity(system, primitive)
        reflected = (
            velocity
            - 2.0 * _normal_velocity(velocity, outward_normal)[..., None] * outward_normal
        )
        return system.primitive_to_conserved(
            _with_primitive_velocity(system, primitive, reflected)
        )

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        return _static_ale_exterior_state(self, system, interior, context, axis)


class MovingSlipWallBoundary(AbstractFiniteVolumeBoundary):
    """Inviscid moving wall with explicit wall/grid kinematic certification."""

    wall_velocity_provider: WallVelocityProvider = eqx.field(static=True)
    wall_velocity_provider_id: str = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        wall_velocity_provider: WallVelocityProvider,
        /,
        *,
        wall_velocity_provider_id: str,
        absolute_tolerance: float = 1.0e-12,
        relative_tolerance: float = 1.0e-10,
    ):
        provider_id = str(wall_velocity_provider_id)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if not callable(wall_velocity_provider):
            raise TypeError("wall_velocity_provider must be callable.")
        if not provider_id:
            raise ValueError("wall_velocity_provider_id must be non-empty.")
        if (
            not math.isfinite(absolute)
            or not math.isfinite(relative)
            or absolute < 0.0
            or relative < 0.0
        ):
            raise ValueError("Moving-wall tolerances must be finite and nonnegative.")
        self.wall_velocity_provider = wall_velocity_provider
        self.wall_velocity_provider_id = provider_id
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "fv-moving-slip-wall",
                "wall_velocity_provider_id": provider_id,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
            }
        )

    def make_context(
        self,
        time: ArrayLike,
        face_point: ArrayLike,
        outward_normal: ArrayLike,
        quadrature_grid_velocity: ArrayLike,
        args: Any,
        /,
        *,
        topology_epoch_id: str,
        geometry_layout_id: str,
        geometry_version: ArrayLike,
        face_block_id: str,
        motion_plan_id: str,
    ) -> ALEBoundaryContext:
        point = jnp.asarray(face_point)
        wall_velocity = _wall_velocity_value(
            self.wall_velocity_provider(
                jnp.asarray(time),
                point,
                jnp.asarray(outward_normal),
                args,
            ),
            point.shape,
        )
        return ALEBoundaryContext(
            face_point=point,
            outward_normal=outward_normal,
            quadrature_grid_velocity=quadrature_grid_velocity,
            wall_velocity=wall_velocity,
            time=time,
            args=args,
            topology_epoch_id=topology_epoch_id,
            geometry_layout_id=geometry_layout_id,
            geometry_version=geometry_version,
            face_block_id=face_block_id,
            motion_plan_id=motion_plan_id,
            absolute_tolerance=self.absolute_tolerance,
            relative_tolerance=self.relative_tolerance,
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
        del system, time, interior, coordinates, outward_normal, axis, args
        raise ValueError(
            "MovingSlipWallBoundary requires ale_exterior_state and explicit "
            "wall/grid kinematics."
        )

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        _validate_ale_boundary_context(system, interior, context, axis)
        dimension = int(system.dimension)
        expected_components = (
            "density",
            *(f"momentum_{index}" for index in range(dimension)),
            "total_energy",
        )
        if (
            tuple(system.component_names) != expected_components
            or jnp.asarray(interior).shape[-1] != dimension + 2
        ):
            raise TypeError(
                "MovingSlipWallBoundary requires an Euler-compatible conservative system."
            )
        if (
            context.absolute_tolerance != self.absolute_tolerance
            or context.relative_tolerance != self.relative_tolerance
        ):
            raise ValueError(
                "ALE boundary context tolerances do not match the moving wall."
            )
        expected_wall_velocity = _wall_velocity_value(
            self.wall_velocity_provider(
                context.time,
                context.face_point,
                context.outward_normal,
                context.args,
            ),
            context.wall_velocity.shape,
        )
        wall_velocity = eqx.error_if(
            context.wall_velocity,
            jnp.any(context.wall_velocity != expected_wall_velocity),
            "ALE boundary context wall velocity does not match its provider.",
        )
        wall_velocity = eqx.error_if(
            wall_velocity,
            jnp.any(context.kinematic_defect > context.kinematic_tolerance),
            "Moving wall normal velocity does not match grid normal velocity.",
        )
        primitive = system.conserved_to_primitive(interior)
        if primitive.shape != jnp.asarray(interior).shape:
            raise TypeError(
                "Euler-compatible primitive and conservative states must align."
            )
        velocity = _primitive_velocity(system, primitive)
        relative_velocity = velocity - wall_velocity
        reflected_velocity = (
            velocity
            - 2.0
            * _normal_velocity(
                relative_velocity,
                context.outward_normal,
            )[..., None]
            * context.outward_normal
        )
        exterior_primitive = _with_primitive_velocity(
            system, primitive, reflected_velocity
        )
        return system.primitive_to_conserved(exterior_primitive)


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

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        return _reject_unsupported_ale_wall(system, interior, context, axis)


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

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        return _reject_unsupported_ale_wall(system, interior, context, axis)


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

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        return _reject_unsupported_ale_wall(system, interior, context, axis)


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

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        return _static_ale_exterior_state(self, system, interior, context, axis)


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

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        return _static_ale_exterior_state(self, system, interior, context, axis)


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
        amplitudes = oe.contract("...ij,...j->...i", left_matrix, target - interior)
        outward_sign = jnp.sign(outward_normal[..., axis])
        incoming = speeds * outward_sign[..., None] < 0.0
        correction = oe.contract(
            "...ij,...j->...i",
            right_matrix,
            jnp.where(incoming, amplitudes, 0.0),
        )
        return interior + correction

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        aligned_interior = _require_axis_aligned_ale_normal(context, axis, interior)
        return _static_ale_exterior_state(self, system, aligned_interior, context, axis)


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
        amplitudes = oe.contract("...ij,...j->...i", left_matrix, target - interior)
        outward_sign = jnp.sign(outward_normal[..., axis])
        incoming = speeds * outward_sign[..., None] < 0.0
        correction = oe.contract(
            "...ij,...j->...i",
            right_matrix,
            jnp.where(incoming, amplitudes, 0.0),
        )
        return interior + correction

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        aligned_interior = _require_axis_aligned_ale_normal(context, axis, interior)
        return _static_ale_exterior_state(self, system, aligned_interior, context, axis)


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

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        aligned_interior = _require_axis_aligned_ale_normal(context, axis, interior)
        return _static_ale_exterior_state(self, system, aligned_interior, context, axis)


__all__ = [
    "CharacteristicInflowBoundary",
    "CharacteristicOutflowBoundary",
    "FarFieldBoundary",
    "NoSlipAdiabaticWallBoundary",
    "MovingSlipWallBoundary",
    "NoSlipIsothermalWallBoundary",
    "PrescribedHeatFluxWallBoundary",
    "SlipWallBoundary",
    "SupersonicInflowBoundary",
    "SupersonicOutflowBoundary",
]
