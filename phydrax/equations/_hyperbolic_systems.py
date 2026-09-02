#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import inverse
from ._materials import IdealGasMaterial
from ._transport_closures import AbstractTransportClosure


ScalarFlux = Callable[[Array, int, Any], ArrayLike]
ScalarWaveSpeed = Callable[[Array, Array, int, Any], ArrayLike]


class AbstractConservationSystem(StrictModule, NonTrainableState):
    """Physical conservation system independent of a numerical method."""

    dimension: int = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        raise NotImplementedError

    def physical_normal_flux(
        self,
        state: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> Array:
        normal_ = jnp.asarray(normal)
        if normal_.ndim == 0 or normal_.shape[-1] != self.dimension:
            raise ValueError(
                "Normal vectors must have a trailing dimension matching "
                f"system dimension {self.dimension}."
            )
        flux = jnp.zeros_like(state)
        for axis in range(self.dimension):
            flux = flux + normal_[..., axis, None] * self.physical_flux(state, axis, args)
        return flux

    def max_normal_wave_speed(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> Array:
        lower, upper = self.normal_signal_bounds(left, right, normal, args)
        return jnp.maximum(jnp.abs(lower), jnp.abs(upper))

    @abc.abstractmethod
    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def conserved_to_primitive(self, state: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def reflect_state(self, state: Array, axis: int, /) -> Array:
        raise NotImplementedError

    @property
    def component_count(self) -> int:
        return len(self.component_names)

    @abc.abstractmethod
    def admissible(self, state: Array, /) -> Array:
        """Unconstrained systems admit every finite state by default."""
        return jnp.all(jnp.isfinite(state), axis=-1)


class AbstractCharacteristicSystem(AbstractConservationSystem):
    """Conservation system with a Roe-like directional eigensystem."""

    @abc.abstractmethod
    def eigensystem(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array]:
        raise NotImplementedError


class AbstractNormalReflectionSystem(abc.ABC):
    """Optional capability for reflection across an arbitrary unit normal."""

    @abc.abstractmethod
    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        raise NotImplementedError


class AbstractNormalCharacteristicSystem(abc.ABC):
    """Optional capability for characteristic decomposition along a normal."""

    @abc.abstractmethod
    def normal_eigensystem(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array]:
        raise NotImplementedError


class AbstractEntropyDiffusionSystem(abc.ABC):
    """Optional capability for entropy-dissipative viscous fluxes."""

    @abc.abstractmethod
    def viscous_flux(
        self,
        state: Array,
        conserved_gradient: Array,
        args: Any = None,
        /,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def maximum_diffusivity(self, state: Array, args: Any = None, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def entropy_viscous_production(
        self,
        state: Array,
        conserved_gradient: Array,
        args: Any = None,
        /,
    ) -> Array:
        raise NotImplementedError


def _orthonormal_normal_frame(normal: Array, dimension: int, /) -> Array:
    value = jnp.asarray(normal)
    if value.shape[-1] != dimension:
        raise ValueError("Normal frame dimension is incompatible with the system.")
    norm = jnp.sqrt(oe.contract("...d,...d->...", value, value, backend="jax"))
    unit = value / norm[..., None]
    if dimension == 1:
        return unit[..., None, :]
    if dimension == 2:
        tangent = jnp.stack((-unit[..., 1], unit[..., 0]), axis=-1)
        return jnp.stack((unit, tangent), axis=-2)
    if dimension == 3:
        seed = jax.nn.one_hot(jnp.argmin(jnp.abs(unit), axis=-1), 3, dtype=unit.dtype)
        first = jnp.cross(seed, unit)
        first = (
            first
            / jnp.sqrt(oe.contract("...d,...d->...", first, first, backend="jax"))[
                ..., None
            ]
        )
        second = jnp.cross(unit, first)
        return jnp.stack((unit, first, second), axis=-2)
    raise ValueError("Normal characteristic frames require dimension one, two, or three.")


def _conserved_normal_rotation(frame: Array, /) -> Array:
    dimension = frame.shape[-1]
    component_count = dimension + 2
    matrix = jnp.broadcast_to(
        jnp.eye(component_count, dtype=frame.dtype),
        frame.shape[:-2] + (component_count, component_count),
    )
    return matrix.at[..., 1 : 1 + dimension, 1 : 1 + dimension].set(frame)


class AbstractAdmissibleSystem(AbstractConservationSystem):
    """Conservation system with a convex admissible-state predicate."""

    @abc.abstractmethod
    def admissible(self, state: Array, /) -> Array:
        raise NotImplementedError


class AbstractEntropySystem(AbstractConservationSystem):
    """Conservation system exposing entropy variables."""

    @abc.abstractmethod
    def entropy_variables(self, state: Array, /) -> Array:
        raise NotImplementedError


class ScalarConservationSystem(
    AbstractConservationSystem, AbstractNormalReflectionSystem
):
    flux: ScalarFlux = eqx.field(static=True)
    wave_speed: ScalarWaveSpeed = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        flux: ScalarFlux,
        wave_speed: ScalarWaveSpeed,
        /,
        *,
        system_id: str,
        component_name: str = "value",
    ):
        dimension_ = int(dimension)
        identifier = str(system_id)
        component = str(component_name)
        if dimension_ not in (1, 2, 3):
            raise ValueError("Scalar conservation dimension must be one, two, or three.")
        if not callable(flux) or not callable(wave_speed):
            raise TypeError("flux and wave_speed must be callable.")
        if not identifier or not component:
            raise ValueError("system_id and component_name must be non-empty.")
        self.dimension = dimension_
        self.component_names = (component,)
        self.system_id = identifier
        self.flux = flux
        self.wave_speed = wave_speed

    def admissible(self, state: Array, /) -> Array:
        return jnp.all(jnp.isfinite(jnp.asarray(state)), axis=-1)

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        return jnp.asarray(self.flux(jnp.asarray(state), int(axis), args))

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        return jnp.asarray(self.wave_speed(left, right, int(axis), args))

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        speed = self.max_wave_speed(left, right, axis, args)
        return -speed, speed

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        normal_ = jnp.asarray(normal)
        speed = jnp.zeros(left.shape[:-1], dtype=left.dtype)
        for axis in range(self.dimension):
            speed = speed + jnp.abs(normal_[..., axis]) * self.max_wave_speed(
                left, right, axis, args
            )
        return -speed, speed

    def conserved_to_primitive(self, state: Array, /) -> Array:
        return jnp.asarray(state)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        return jnp.asarray(primitive)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        del axis
        return jnp.asarray(state)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        del normal
        return jnp.asarray(state)


class EulerSystem(
    AbstractCharacteristicSystem,
    AbstractAdmissibleSystem,
    AbstractEntropySystem,
    AbstractNormalReflectionSystem,
    AbstractNormalCharacteristicSystem,
):
    """Ideal-gas Euler equations in one, two, or three dimensions."""

    material: IdealGasMaterial

    def __init__(
        self,
        dimension: int = 1,
        /,
        *,
        material: IdealGasMaterial | None = None,
    ):
        dimension_ = int(dimension)
        material_ = IdealGasMaterial() if material is None else material
        if dimension_ not in (1, 2, 3):
            raise ValueError("Euler dimension must be one, two, or three.")
        if not isinstance(material_, IdealGasMaterial):
            raise TypeError("EulerSystem currently requires IdealGasMaterial.")
        self.dimension = dimension_
        self.component_names = (
            "density",
            *(f"momentum_{axis}" for axis in range(dimension_)),
            "total_energy",
        )
        self.material = material_
        self.system_id = canonical_fingerprint(
            {
                "kind": "euler-system",
                "dimension": dimension_,
                "material": material_.material_id,
            }
        )

    @property
    def gamma(self) -> float:
        return self.material.gamma

    @property
    def density_floor(self) -> float:
        return self.material.density_floor

    @property
    def pressure_floor(self) -> float:
        return self.material.pressure_floor

    def pressure(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum = value[..., 1 : 1 + self.dimension]
        specific_internal_energy = (
            value[..., -1] / density - 0.5 * jnp.sum(momentum**2, axis=-1) / density**2
        )
        return self.material.pressure(density, specific_internal_energy)

    def temperature(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        return self.material.temperature(value[..., 0], self.pressure(value))

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        velocity = value[..., 1 : 1 + self.dimension] / density[..., None]
        return jnp.concatenate(
            (density[..., None], velocity, self.pressure(value)[..., None]), axis=-1
        )

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = jnp.asarray(primitive)
        density = value[..., 0]
        velocity = value[..., 1 : 1 + self.dimension]
        pressure = value[..., -1]
        energy = density * self.material.specific_internal_energy(
            density, pressure
        ) + 0.5 * density * jnp.sum(velocity**2, axis=-1)
        return jnp.concatenate(
            (density[..., None], density[..., None] * velocity, energy[..., None]),
            axis=-1,
        )

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        axis_ = int(axis)
        if not 0 <= axis_ < self.dimension:
            raise ValueError("Euler flux axis is out of range.")
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum = value[..., 1 : 1 + self.dimension]
        energy = value[..., -1]
        velocity = momentum / density[..., None]
        normal_velocity = velocity[..., axis_]
        pressure = self.pressure(value)
        momentum_flux = momentum * normal_velocity[..., None]
        momentum_flux = momentum_flux.at[..., axis_].add(pressure)
        return jnp.concatenate(
            (
                momentum[..., axis_ : axis_ + 1],
                momentum_flux,
                ((energy + pressure) * normal_velocity)[..., None],
            ),
            axis=-1,
        )

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        del args
        axis_ = int(axis)

        def speed(state: Array) -> Array:
            primitive = self.conserved_to_primitive(state)
            sound = self.material.sound_speed(primitive[..., 0], primitive[..., -1])
            return jnp.abs(primitive[..., 1 + axis_]) + sound

        return jnp.maximum(speed(left), speed(right))

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        axis_ = int(axis)
        left_primitive = self.conserved_to_primitive(left)
        right_primitive = self.conserved_to_primitive(right)
        left_sound = self.material.sound_speed(
            left_primitive[..., 0], left_primitive[..., -1]
        )
        right_sound = self.material.sound_speed(
            right_primitive[..., 0], right_primitive[..., -1]
        )
        lower = jnp.minimum(
            left_primitive[..., 1 + axis_] - left_sound,
            right_primitive[..., 1 + axis_] - right_sound,
        )
        upper = jnp.maximum(
            left_primitive[..., 1 + axis_] + left_sound,
            right_primitive[..., 1 + axis_] + right_sound,
        )
        return lower, upper

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        normal_ = jnp.asarray(normal)
        left_primitive = self.conserved_to_primitive(left)
        right_primitive = self.conserved_to_primitive(right)
        left_sound = self.material.sound_speed(
            left_primitive[..., 0], left_primitive[..., -1]
        )
        right_sound = self.material.sound_speed(
            right_primitive[..., 0], right_primitive[..., -1]
        )
        left_velocity = jnp.sum(left_primitive[..., 1:-1] * normal_, axis=-1)
        right_velocity = jnp.sum(right_primitive[..., 1:-1] * normal_, axis=-1)
        return (
            jnp.minimum(left_velocity - left_sound, right_velocity - right_sound),
            jnp.maximum(left_velocity + left_sound, right_velocity + right_sound),
        )

    def admissible(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return self.material.admissible(value[..., 0], self.pressure(value))

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        value = jnp.asarray(state)
        return value.at[..., 1 + int(axis)].multiply(-1.0)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        value = jnp.asarray(state)
        normal_ = jnp.asarray(normal)
        norm = jnp.sqrt(oe.contract("...d,...d->...", normal_, normal_, backend="jax"))
        unit = normal_ / norm[..., None]
        momentum = value[..., 1 : 1 + self.dimension]
        normal_momentum = oe.contract("...d,...d->...", momentum, unit, backend="jax")
        reflected = momentum - 2.0 * normal_momentum[..., None] * unit
        return value.at[..., 1 : 1 + self.dimension].set(reflected)

    def eigensystem(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array]:
        del args
        axis_ = int(axis)
        left_primitive = self.conserved_to_primitive(left)
        right_primitive = self.conserved_to_primitive(right)
        left_root = jnp.sqrt(left_primitive[..., 0])
        right_root = jnp.sqrt(right_primitive[..., 0])
        denominator = left_root + right_root
        velocity = (
            left_root[..., None] * left_primitive[..., 1:-1]
            + right_root[..., None] * right_primitive[..., 1:-1]
        ) / denominator[..., None]
        left_enthalpy = (left[..., -1] + left_primitive[..., -1]) / left_primitive[..., 0]
        right_enthalpy = (right[..., -1] + right_primitive[..., -1]) / right_primitive[
            ..., 0
        ]
        enthalpy = (left_root * left_enthalpy + right_root * right_enthalpy) / denominator
        speed_squared = jnp.sum(velocity**2, axis=-1)
        sound = jnp.sqrt(
            jnp.maximum(
                (self.material.gamma - 1.0) * (enthalpy - 0.5 * speed_squared),
                jnp.finfo(left.dtype).tiny,
            )
        )
        basis = jnp.eye(self.dimension, dtype=left.dtype)
        normal = basis[axis_]
        acoustic_minus = jnp.concatenate(
            (
                jnp.ones_like(sound)[..., None],
                velocity - sound[..., None] * normal,
                (enthalpy - velocity[..., axis_] * sound)[..., None],
            ),
            axis=-1,
        )
        contact = jnp.concatenate(
            (
                jnp.ones_like(sound)[..., None],
                velocity,
                (0.5 * speed_squared)[..., None],
            ),
            axis=-1,
        )
        shear_columns = []
        for transverse in range(self.dimension):
            if transverse == axis_:
                continue
            momentum = jnp.broadcast_to(basis[transverse], velocity.shape)
            shear_columns.append(
                jnp.concatenate(
                    (
                        jnp.zeros_like(sound)[..., None],
                        momentum,
                        velocity[..., transverse : transverse + 1],
                    ),
                    axis=-1,
                )
            )
        acoustic_plus = jnp.concatenate(
            (
                jnp.ones_like(sound)[..., None],
                velocity + sound[..., None] * normal,
                (enthalpy + velocity[..., axis_] * sound)[..., None],
            ),
            axis=-1,
        )
        columns = (acoustic_minus, contact, *shear_columns, acoustic_plus)
        right_matrix = jnp.stack(columns, axis=-1)
        inverse_result = inverse(right_matrix)
        left_matrix = eqx.error_if(
            inverse_result.value,
            jnp.any(~inverse_result.successful),
            "Euler characteristic basis is singular.",
        )
        eigenvalues = jnp.stack(
            (
                velocity[..., axis_] - sound,
                velocity[..., axis_],
                *(velocity[..., axis_] for _ in shear_columns),
                velocity[..., axis_] + sound,
            ),
            axis=-1,
        )
        return left_matrix, right_matrix, eigenvalues

    def normal_eigensystem(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array]:
        frame = _orthonormal_normal_frame(normal, self.dimension)
        rotation = _conserved_normal_rotation(frame)
        left_local = oe.contract("...ij,...j->...i", rotation, left, backend="jax")
        right_local = oe.contract("...ij,...j->...i", rotation, right, backend="jax")
        left_matrix, right_matrix, speeds = self.eigensystem(
            left_local, right_local, 0, args
        )
        right_global = oe.contract(
            "...ji,...jk->...ik", rotation, right_matrix, backend="jax"
        )
        left_global = oe.contract(
            "...ij,...jk->...ik", left_matrix, rotation, backend="jax"
        )
        return left_global, right_global, speeds

    def entropy_variables(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        primitive = self.conserved_to_primitive(value)
        density = primitive[..., 0]
        velocity = primitive[..., 1:-1]
        pressure = primitive[..., -1]
        entropy = jnp.log(pressure) - self.material.gamma * jnp.log(density)
        beta = density / (2.0 * pressure)
        first = (self.material.gamma - entropy) / (
            self.material.gamma - 1.0
        ) - beta * jnp.sum(velocity**2, axis=-1)
        return jnp.concatenate(
            (
                first[..., None],
                2.0 * beta[..., None] * velocity,
                (-2.0 * beta)[..., None],
            ),
            axis=-1,
        )


class CompressibleNavierStokesSystem(
    AbstractCharacteristicSystem,
    AbstractAdmissibleSystem,
    AbstractEntropySystem,
    AbstractNormalReflectionSystem,
    AbstractNormalCharacteristicSystem,
    AbstractEntropyDiffusionSystem,
):
    """Ideal-gas compressible Navier–Stokes physical system."""

    inviscid: EulerSystem
    material: IdealGasMaterial
    transport: AbstractTransportClosure

    def __init__(
        self,
        transport: AbstractTransportClosure,
        dimension: int = 1,
        /,
        *,
        material: IdealGasMaterial | None = None,
    ):
        if not isinstance(transport, AbstractTransportClosure):
            raise TypeError("transport must be an AbstractTransportClosure.")
        inviscid = EulerSystem(dimension, material=material)
        self.dimension = inviscid.dimension
        self.component_names = inviscid.component_names
        self.inviscid = inviscid
        self.material = inviscid.material
        self.transport = transport
        self.system_id = canonical_fingerprint(
            {
                "kind": "compressible-navier-stokes-system",
                "inviscid": inviscid.system_id,
                "transport": transport.closure_id,
            }
        )

    @property
    def gamma(self) -> float:
        return self.material.gamma

    @property
    def density_floor(self) -> float:
        return self.material.density_floor

    @property
    def pressure_floor(self) -> float:
        return self.material.pressure_floor

    def pressure(self, state: ArrayLike, /) -> Array:
        return self.inviscid.pressure(state)

    def temperature(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        return self.material.temperature(value[..., 0], self.pressure(value))

    def conserved_to_primitive(self, state: Array, /) -> Array:
        return self.inviscid.conserved_to_primitive(state)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        return self.inviscid.primitive_to_conserved(primitive)

    def primitive_gradients(
        self,
        state: ArrayLike,
        conserved_gradient: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        """Return velocity and temperature gradients from conserved gradients."""
        value = jnp.asarray(state)
        gradient = jnp.asarray(conserved_gradient)
        expected = value.shape + (self.dimension,)
        if gradient.shape != expected:
            raise ValueError(
                "Conserved gradients must append one physical derivative axis."
            )
        density = value[..., 0]
        momentum = value[..., 1 : 1 + self.dimension]
        velocity = momentum / density[..., None]
        density_gradient = gradient[..., 0, :]
        momentum_gradient = gradient[..., 1 : 1 + self.dimension, :]
        velocity_gradient = (
            momentum_gradient - velocity[..., :, None] * density_gradient[..., None, :]
        ) / density[..., None, None]
        energy_gradient = gradient[..., -1, :]
        speed_squared = oe.contract("...i,...i->...", velocity, velocity, backend="jax")
        kinetic_gradient = (
            oe.contract(
                "...i,...ij->...j",
                velocity,
                momentum_gradient,
                backend="jax",
            )
            - 0.5 * speed_squared[..., None] * density_gradient
        )
        pressure = self.pressure(value)
        pressure_gradient = (self.gamma - 1.0) * (energy_gradient - kinetic_gradient)
        gas_constant = self.material.gas_constant
        temperature_gradient = pressure_gradient / (
            density[..., None] * gas_constant
        ) - pressure[..., None] * density_gradient / (
            density[..., None] ** 2 * gas_constant
        )
        return velocity_gradient, temperature_gradient

    def entropy_viscous_production(
        self,
        state: Array,
        conserved_gradient: Array,
        args: Any = None,
        /,
    ) -> Array:
        value = jnp.asarray(state)
        gradient = jnp.asarray(conserved_gradient)
        flat_state = value.reshape((-1, value.shape[-1]))
        flat_gradient = gradient.reshape((-1, gradient.shape[-2], gradient.shape[-1]))
        hessian = jax.vmap(jax.jacfwd(lambda point: self.entropy_variables(point)))(
            flat_state
        )
        entropy_gradient = oe.contract(
            "nij,njd->nid", hessian, flat_gradient, backend="jax"
        )
        viscous_flux = self.viscous_flux(value, gradient, args).reshape(
            entropy_gradient.shape
        )
        production = oe.contract(
            "nid,nid->n", entropy_gradient, viscous_flux, backend="jax"
        )
        return production.reshape(value.shape[:-1])

    def viscous_flux(
        self,
        state: ArrayLike,
        conserved_gradient: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        """Return the positive diffusive flux tensor for conserved variables."""
        value = jnp.asarray(state)
        velocity = value[..., 1 : 1 + self.dimension] / value[..., :1]
        velocity_gradient, temperature_gradient = self.primitive_gradients(
            value, conserved_gradient
        )
        temperature = self.temperature(value)
        properties = self.transport.properties(temperature, value, args)
        return self.viscous_flux_from_primitive_gradients(
            velocity,
            velocity_gradient,
            temperature_gradient,
            properties.dynamic_viscosity,
            properties.bulk_viscosity,
            properties.thermal_conductivity,
        )

    def viscous_flux_from_primitive_gradients(
        self,
        velocity: ArrayLike,
        velocity_gradient: ArrayLike,
        temperature_gradient: ArrayLike,
        dynamic_viscosity: ArrayLike,
        bulk_viscosity: ArrayLike,
        thermal_conductivity: ArrayLike,
        /,
    ) -> Array:
        """Return diffusive flux from primitive gradients and transport coefficients."""
        velocity_ = jnp.asarray(velocity)
        gradient = jnp.asarray(velocity_gradient)
        temperature_gradient_ = jnp.asarray(temperature_gradient)
        viscosity = jnp.asarray(dynamic_viscosity)
        bulk = jnp.asarray(bulk_viscosity)
        conductivity = jnp.asarray(thermal_conductivity)
        coefficient_shape = velocity_.shape[:-1]
        if (
            velocity_.shape[-1] != self.dimension
            or gradient.shape != velocity_.shape + (self.dimension,)
            or temperature_gradient_.shape != velocity_.shape
            or viscosity.shape != coefficient_shape
            or bulk.shape != coefficient_shape
            or conductivity.shape != coefficient_shape
        ):
            raise ValueError(
                "Primitive gradients and transport coefficients are incompatible."
            )
        divergence = jnp.trace(gradient, axis1=-2, axis2=-1)
        identity = jnp.eye(self.dimension, dtype=velocity_.dtype)
        deviatoric = (
            gradient
            + jnp.swapaxes(gradient, -1, -2)
            - (2.0 / 3.0) * divergence[..., None, None] * identity
        )
        stress = (
            viscosity[..., None, None] * deviatoric
            + bulk[..., None, None] * divergence[..., None, None] * identity
        )
        energy_flux = (
            oe.contract("...i,...ij->...j", velocity_, stress, backend="jax")
            + conductivity[..., None] * temperature_gradient_
        )
        mass_flux = jnp.zeros(
            velocity_.shape[:-1] + (1, self.dimension), dtype=velocity_.dtype
        )
        return jnp.concatenate((mass_flux, stress, energy_flux[..., None, :]), axis=-2)

    def maximum_diffusivity(self, state: Array, args: Any = None, /) -> Array:
        value = jnp.asarray(state)
        temperature = self.temperature(value)
        properties = self.transport.properties(temperature, value, args)
        density = value[..., 0]
        heat_capacity = self.material.gas_constant / (self.gamma - 1.0)
        return jnp.maximum(
            properties.dynamic_viscosity / density,
            properties.thermal_conductivity / (density * heat_capacity),
        )

    def viscous_normal_flux(
        self,
        state: ArrayLike,
        conserved_gradient: ArrayLike,
        normal: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        normal_ = jnp.asarray(normal)
        flux = self.viscous_flux(state, conserved_gradient, args)
        if normal_.shape != flux.shape[:-2] + (self.dimension,):
            raise ValueError("Viscous normal shape is incompatible with the state.")
        return oe.contract("...ij,...j->...i", flux, normal_, backend="jax")

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        return self.inviscid.physical_flux(state, axis, args)

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        return self.inviscid.max_wave_speed(left, right, axis, args)

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        return self.inviscid.signal_bounds(left, right, axis, args)

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        return self.inviscid.normal_signal_bounds(left, right, normal, args)

    def admissible(self, state: Array, /) -> Array:
        return self.inviscid.admissible(state)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return self.inviscid.reflect_state(state, axis)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        return self.inviscid.reflect_normal_state(state, normal)

    def eigensystem(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array]:
        return self.inviscid.eigensystem(left, right, axis, args)

    def normal_eigensystem(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array]:
        return self.inviscid.normal_eigensystem(left, right, normal, args)

    def entropy_variables(self, state: Array, /) -> Array:
        return self.inviscid.entropy_variables(state)


class MultispeciesEulerSystem(AbstractAdmissibleSystem):
    """Calorically perfect conservative multispecies Euler system."""

    species_gammas: tuple[float, ...] = eqx.field(static=True)
    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)

    def __init__(
        self,
        species_gammas: Sequence[float],
        dimension: int = 1,
        /,
        *,
        density_floor: float = 1e-12,
        pressure_floor: float = 1e-12,
    ):
        gammas = tuple(float(value) for value in species_gammas)
        dimension_ = int(dimension)
        if not gammas or any(not np.isfinite(value) or value <= 1.0 for value in gammas):
            raise ValueError("Every species gamma must be finite and greater than one.")
        if dimension_ not in (1, 2, 3):
            raise ValueError("Multispecies Euler dimension must be one, two, or three.")
        if density_floor <= 0.0 or pressure_floor <= 0.0:
            raise ValueError("Multispecies floors must be positive.")
        self.dimension = dimension_
        self.species_gammas = gammas
        self.density_floor = float(density_floor)
        self.pressure_floor = float(pressure_floor)
        self.component_names = (
            *(f"species_density_{index}" for index in range(len(gammas))),
            *(f"momentum_{axis}" for axis in range(dimension_)),
            "total_energy",
        )
        self.system_id = canonical_fingerprint(
            {
                "kind": "multispecies-euler-system",
                "dimension": dimension_,
                "species_gammas": list(gammas),
                "density_floor": float(density_floor),
                "pressure_floor": float(pressure_floor),
            }
        )

    @property
    def species_count(self) -> int:
        return len(self.species_gammas)

    def density(self, state: Array, /) -> Array:
        return jnp.sum(state[..., : self.species_count], axis=-1)

    def mixture_gamma(self, state: Array, /) -> Array:
        species = state[..., : self.species_count]
        density = jnp.sum(species, axis=-1)
        fractions = species / density[..., None]
        heat_capacity = jnp.sum(
            fractions / (jnp.asarray(self.species_gammas, dtype=state.dtype) - 1.0),
            axis=-1,
        )
        return 1.0 + 1.0 / heat_capacity

    def pressure(self, state: Array, /) -> Array:
        density = self.density(state)
        momentum = state[..., self.species_count : -1]
        return (self.mixture_gamma(state) - 1.0) * (
            state[..., -1] - 0.5 * jnp.sum(momentum**2, axis=-1) / density
        )

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        density = self.density(value)
        velocity = value[..., self.species_count : -1] / density[..., None]
        return jnp.concatenate(
            (value[..., : self.species_count], velocity, self.pressure(value)[..., None]),
            axis=-1,
        )

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = jnp.asarray(primitive)
        species = value[..., : self.species_count]
        density = jnp.sum(species, axis=-1)
        velocity = value[..., self.species_count : -1]
        pressure = value[..., -1]
        provisional = jnp.concatenate(
            (
                species,
                density[..., None] * velocity,
                jnp.ones_like(pressure)[..., None],
            ),
            axis=-1,
        )
        energy = pressure / (
            self.mixture_gamma(provisional) - 1.0
        ) + 0.5 * density * jnp.sum(velocity**2, axis=-1)
        return provisional.at[..., -1].set(energy)

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        value = jnp.asarray(state)
        density = self.density(value)
        momentum = value[..., self.species_count : -1]
        velocity = momentum / density[..., None]
        normal_velocity = velocity[..., int(axis)]
        pressure = self.pressure(value)
        species_flux = value[..., : self.species_count] * normal_velocity[..., None]
        momentum_flux = momentum * normal_velocity[..., None]
        momentum_flux = momentum_flux.at[..., int(axis)].add(pressure)
        energy_flux = (value[..., -1] + pressure) * normal_velocity
        return jnp.concatenate(
            (species_flux, momentum_flux, energy_flux[..., None]), axis=-1
        )

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        del args
        axis_ = int(axis)

        def speed(state: Array) -> Array:
            density = self.density(state)
            velocity = state[..., self.species_count + axis_] / density
            sound = jnp.sqrt(self.mixture_gamma(state) * self.pressure(state) / density)
            return jnp.abs(velocity) + sound

        return jnp.maximum(speed(left), speed(right))

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        speed = self.max_wave_speed(left, right, axis, args)
        return -speed, speed

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        normal_ = jnp.asarray(normal)
        speed = jnp.zeros(left.shape[:-1], dtype=left.dtype)
        for axis in range(self.dimension):
            speed = speed + jnp.abs(normal_[..., axis]) * self.max_wave_speed(
                left, right, axis, args
            )
        return -speed, speed

    def admissible(self, state: Array, /) -> Array:
        return jnp.all(
            state[..., : self.species_count] >= self.density_floor, axis=-1
        ) & (self.pressure(state) >= self.pressure_floor)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return jnp.asarray(state).at[..., self.species_count + int(axis)].multiply(-1.0)


class IdealMHDSystem(AbstractAdmissibleSystem, AbstractNormalReflectionSystem):
    """Ideal MHD with three-vector momentum and magnetic field."""

    material: IdealGasMaterial

    def __init__(
        self,
        dimension: int = 1,
        /,
        *,
        material: IdealGasMaterial | None = None,
    ):
        dimension_ = int(dimension)
        material_ = IdealGasMaterial() if material is None else material
        if dimension_ not in (1, 2, 3):
            raise ValueError("MHD dimension must be one, two, or three.")
        if not isinstance(material_, IdealGasMaterial):
            raise TypeError("IdealMHDSystem requires IdealGasMaterial.")
        self.dimension = dimension_
        self.material = material_
        self.component_names = (
            "density",
            "momentum_x",
            "momentum_y",
            "momentum_z",
            "total_energy",
            "magnetic_x",
            "magnetic_y",
            "magnetic_z",
        )
        self.system_id = canonical_fingerprint(
            {
                "kind": "ideal-mhd-system",
                "dimension": dimension_,
                "material": material_.material_id,
            }
        )

    @property
    def gamma(self) -> float:
        return self.material.gamma

    @property
    def density_floor(self) -> float:
        return self.material.density_floor

    @property
    def pressure_floor(self) -> float:
        return self.material.pressure_floor

    def pressure(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum_squared = jnp.sum(value[..., 1:4] ** 2, axis=-1)
        magnetic_squared = jnp.sum(value[..., 5:8] ** 2, axis=-1)
        internal = (
            value[..., 4] - 0.5 * momentum_squared / density - 0.5 * magnetic_squared
        )
        return self.material.pressure(density, internal / density)

    def temperature(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return self.material.temperature(value[..., 0], self.pressure(value))

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        return jnp.concatenate(
            (
                density[..., None],
                value[..., 1:4] / density[..., None],
                self.pressure(value)[..., None],
                value[..., 5:8],
            ),
            axis=-1,
        )

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = jnp.asarray(primitive)
        density = value[..., 0]
        velocity = value[..., 1:4]
        pressure = value[..., 4]
        magnetic = value[..., 5:8]
        energy = (
            density * self.material.specific_internal_energy(density, pressure)
            + 0.5 * density * jnp.sum(velocity**2, axis=-1)
            + 0.5 * jnp.sum(magnetic**2, axis=-1)
        )
        return jnp.concatenate(
            (
                density[..., None],
                density[..., None] * velocity,
                energy[..., None],
                magnetic,
            ),
            axis=-1,
        )

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        axis_ = int(axis)
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum = value[..., 1:4]
        energy = value[..., 4]
        magnetic = value[..., 5:8]
        velocity = momentum / density[..., None]
        pressure = self.pressure(value)
        total_pressure = pressure + 0.5 * jnp.sum(magnetic**2, axis=-1)
        normal_velocity = velocity[..., axis_]
        normal_magnetic = magnetic[..., axis_]
        momentum_flux = (
            momentum * normal_velocity[..., None] - normal_magnetic[..., None] * magnetic
        )
        momentum_flux = momentum_flux.at[..., axis_].add(total_pressure)
        energy_flux = (
            energy + total_pressure
        ) * normal_velocity - normal_magnetic * jnp.sum(velocity * magnetic, axis=-1)
        magnetic_flux = (
            normal_velocity[..., None] * magnetic - normal_magnetic[..., None] * velocity
        )
        magnetic_flux = magnetic_flux.at[..., axis_].set(0.0)
        return jnp.concatenate(
            (
                momentum[..., axis_ : axis_ + 1],
                momentum_flux,
                energy_flux[..., None],
                magnetic_flux,
            ),
            axis=-1,
        )

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        del args
        axis_ = int(axis)

        def speed(state: Array) -> Array:
            density = state[..., 0]
            velocity = state[..., 1 + axis_] / density
            magnetic = state[..., 5:8]
            sound_squared = self.material.sound_speed(density, self.pressure(state)) ** 2
            magnetic_squared = jnp.sum(magnetic**2, axis=-1) / density
            normal_squared = magnetic[..., axis_] ** 2 / density
            discriminant = jnp.maximum(
                (sound_squared + magnetic_squared) ** 2
                - 4.0 * sound_squared * normal_squared,
                0.0,
            )
            fast = jnp.sqrt(
                0.5 * (sound_squared + magnetic_squared + jnp.sqrt(discriminant))
            )
            return jnp.abs(velocity) + fast

        return jnp.maximum(speed(left), speed(right))

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        speed = self.max_wave_speed(left, right, axis, args)
        return -speed, speed

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        normal_ = jnp.asarray(normal)
        speed = jnp.zeros(left.shape[:-1], dtype=left.dtype)
        for axis in range(self.dimension):
            speed = speed + jnp.abs(normal_[..., axis]) * self.max_wave_speed(
                left, right, axis, args
            )
        return -speed, speed

    def admissible(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return self.material.admissible(value[..., 0], self.pressure(value))

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return jnp.asarray(state).at[..., 1 + int(axis)].multiply(-1.0)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        value = jnp.asarray(state)
        normal_ = jnp.asarray(normal)
        unit = (
            normal_
            / jnp.sqrt(oe.contract("...d,...d->...", normal_, normal_, backend="jax"))[
                ..., None
            ]
        )
        momentum = value[..., 1:4]
        magnetic = value[..., 5:8]
        reflected_momentum = (
            momentum
            - 2.0
            * oe.contract("...d,...d->...", momentum, unit, backend="jax")[..., None]
            * unit
        )
        reflected_magnetic = (
            2.0
            * oe.contract("...d,...d->...", magnetic, unit, backend="jax")[..., None]
            * unit
            - magnetic
        )
        result = value.at[..., 1:4].set(reflected_momentum)
        return result.at[..., 5:8].set(reflected_magnetic)


class ShallowWaterSystem(AbstractAdmissibleSystem):
    """One- or two-dimensional shallow-water conservation system."""

    gravity: float = eqx.field(static=True)

    def __init__(
        self,
        dimension: int = 1,
        /,
        *,
        gravity: float = 9.81,
    ):
        dimension_ = int(dimension)
        gravity_ = float(gravity)
        if dimension_ not in (1, 2):
            raise ValueError("Shallow-water dimension must be one or two.")
        if not np.isfinite(gravity_) or gravity_ <= 0.0:
            raise ValueError("Shallow-water gravity must be finite and positive.")
        self.dimension = dimension_
        self.gravity = gravity_
        self.component_names = (
            "depth",
            *(f"discharge_{axis}" for axis in range(dimension_)),
        )
        self.system_id = canonical_fingerprint(
            {
                "kind": "shallow-water-system",
                "dimension": dimension_,
                "gravity": gravity_,
                "dry_state": "zero-depth-zero-discharge",
            }
        )

    def velocity(self, state: Array, /) -> Array:
        """Return velocity with the exact dry-state value defined as zero."""
        value = jnp.asarray(state)
        depth = value[..., :1]
        safe_depth = jnp.where(depth > 0.0, depth, 1.0)
        velocity = value[..., 1:] / safe_depth
        return jnp.where(depth > 0.0, velocity, 0.0)

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return jnp.concatenate((value[..., :1], self.velocity(value)), axis=-1)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = jnp.asarray(primitive)
        return jnp.concatenate((value[..., :1], value[..., :1] * value[..., 1:]), axis=-1)

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        value = jnp.asarray(state)
        axis_ = int(axis)
        depth = value[..., 0]
        discharge = value[..., 1:]
        normal_velocity = self.velocity(value)[..., axis_]
        momentum_flux = discharge * normal_velocity[..., None]
        momentum_flux = momentum_flux.at[..., axis_].add(0.5 * self.gravity * depth**2)
        return jnp.concatenate(
            (discharge[..., axis_ : axis_ + 1], momentum_flux), axis=-1
        )

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        axis_ = int(axis)
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        left_velocity = self.velocity(left_)[..., axis_]
        right_velocity = self.velocity(right_)[..., axis_]
        left_celerity = jnp.sqrt(self.gravity * jnp.maximum(left_[..., 0], 0.0))
        right_celerity = jnp.sqrt(self.gravity * jnp.maximum(right_[..., 0], 0.0))
        lower = jnp.minimum(
            left_velocity - left_celerity,
            right_velocity - right_celerity,
        )
        upper = jnp.maximum(
            left_velocity + left_celerity,
            right_velocity + right_celerity,
        )
        return lower, upper

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        lower, upper = self.signal_bounds(left, right, axis, args)
        return jnp.maximum(jnp.abs(lower), jnp.abs(upper))

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        normal_ = jnp.asarray(normal)
        if normal_.ndim == 0 or normal_.shape[-1] != self.dimension:
            raise ValueError(
                "Normal vectors must have a trailing dimension matching "
                f"system dimension {self.dimension}."
            )
        left_velocity = jnp.sum(self.velocity(left_) * normal_, axis=-1)
        right_velocity = jnp.sum(self.velocity(right_) * normal_, axis=-1)
        left_celerity = jnp.sqrt(self.gravity * jnp.maximum(left_[..., 0], 0.0))
        right_celerity = jnp.sqrt(self.gravity * jnp.maximum(right_[..., 0], 0.0))
        lower = jnp.minimum(
            left_velocity - left_celerity,
            right_velocity - right_celerity,
        )
        upper = jnp.maximum(
            left_velocity + left_celerity,
            right_velocity + right_celerity,
        )
        return lower, upper

    def admissible(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        depth = value[..., 0]
        discharge = value[..., 1:]
        finite = jnp.all(jnp.isfinite(value), axis=-1)
        dry_momentum = jnp.all(discharge == 0.0, axis=-1)
        return finite & (depth >= 0.0) & ((depth > 0.0) | dry_momentum)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return jnp.asarray(state).at[..., 1 + int(axis)].multiply(-1.0)


__all__ = [
    "AbstractAdmissibleSystem",
    "AbstractCharacteristicSystem",
    "AbstractConservationSystem",
    "AbstractEntropyDiffusionSystem",
    "AbstractEntropySystem",
    "AbstractNormalCharacteristicSystem",
    "AbstractNormalReflectionSystem",
    "CompressibleNavierStokesSystem",
    "EulerSystem",
    "IdealMHDSystem",
    "MultispeciesEulerSystem",
    "ScalarConservationSystem",
    "ShallowWaterSystem",
]
