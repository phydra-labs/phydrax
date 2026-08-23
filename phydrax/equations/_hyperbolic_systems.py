#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
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
        flux = jnp.zeros_like(state)
        for axis in range(self.dimension):
            flux = flux + normal_[..., axis, None] * self.physical_flux(
                state, axis, args
            )
        return flux

    def max_normal_wave_speed(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> Array:
        normal_ = jnp.asarray(normal)
        speed = jnp.zeros(left.shape[:-1], dtype=left.dtype)
        for axis in range(self.dimension):
            speed = speed + jnp.abs(normal_[..., axis]) * self.max_wave_speed(
                left, right, axis, args
            )
        return speed

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        speed = self.max_normal_wave_speed(left, right, normal, args)
        return -speed, speed



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


class ScalarConservationSystem(AbstractConservationSystem):
    """One-component conservation system from stable-ID callables."""

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


    def conserved_to_primitive(self, state: Array, /) -> Array:
        return jnp.asarray(state)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        return jnp.asarray(primitive)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        del axis
        return jnp.asarray(state)


class EulerSystem(
    AbstractCharacteristicSystem,
    AbstractAdmissibleSystem,
    AbstractEntropySystem,
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
            value[..., -1] / density
            - 0.5 * jnp.sum(momentum**2, axis=-1) / density**2
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
            sound = self.material.sound_speed(
                primitive[..., 0], primitive[..., -1]
            )
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


    def admissible(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return self.material.admissible(value[..., 0], self.pressure(value))

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        value = jnp.asarray(state)
        return value.at[..., 1 + int(axis)].multiply(-1.0)

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
        left_enthalpy = (left[..., -1] + left_primitive[..., -1]) / left_primitive[
            ..., 0
        ]
        right_enthalpy = (right[..., -1] + right_primitive[..., -1]) / right_primitive[
            ..., 0
        ]
        enthalpy = (
            left_root * left_enthalpy + right_root * right_enthalpy
        ) / denominator
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
            momentum = jnp.broadcast_to(
                basis[transverse], velocity.shape
            )
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
        left_matrix = jnp.linalg.inv(right_matrix)
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

    def entropy_variables(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        primitive = self.conserved_to_primitive(value)
        density = primitive[..., 0]
        velocity = primitive[..., 1:-1]
        pressure = primitive[..., -1]
        entropy = jnp.log(pressure) - self.material.gamma * jnp.log(density)
        beta = density / (2.0 * pressure)
        first = (
            (self.material.gamma - entropy) / (self.material.gamma - 1.0)
            - beta * jnp.sum(velocity**2, axis=-1)
        )
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

    def physical_flux(
        self, state: Array, axis: int, args: Any = None, /
    ) -> Array:
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

    def admissible(self, state: Array, /) -> Array:
        return self.inviscid.admissible(state)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return self.inviscid.reflect_state(state, axis)

    def eigensystem(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array]:
        return self.inviscid.eigensystem(left, right, axis, args)

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
        energy = pressure / (self.mixture_gamma(provisional) - 1.0) + 0.5 * density * jnp.sum(
            velocity**2, axis=-1
        )
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


    def admissible(self, state: Array, /) -> Array:
        return jnp.all(
            state[..., : self.species_count] >= self.density_floor, axis=-1
        ) & (self.pressure(state) >= self.pressure_floor)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return jnp.asarray(state).at[..., self.species_count + int(axis)].multiply(-1.0)


class IdealMHDSystem(AbstractAdmissibleSystem):
    """Ideal MHD with three-vector momentum and magnetic field."""

    gamma: float = eqx.field(static=True)
    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)

    def __init__(
        self,
        dimension: int = 1,
        /,
        *,
        gamma: float = 1.4,
        density_floor: float = 1e-12,
        pressure_floor: float = 1e-12,
    ):
        dimension_ = int(dimension)
        gamma_ = float(gamma)
        if dimension_ not in (1, 2, 3):
            raise ValueError("MHD dimension must be one, two, or three.")
        if not np.isfinite(gamma_) or gamma_ <= 1.0:
            raise ValueError("MHD gamma must be finite and greater than one.")
        if density_floor <= 0.0 or pressure_floor <= 0.0:
            raise ValueError("MHD floors must be positive.")
        self.dimension = dimension_
        self.gamma = gamma_
        self.density_floor = float(density_floor)
        self.pressure_floor = float(pressure_floor)
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
                "gamma": gamma_,
                "density_floor": float(density_floor),
                "pressure_floor": float(pressure_floor),
            }
        )

    def pressure(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum_squared = jnp.sum(value[..., 1:4] ** 2, axis=-1)
        magnetic_squared = jnp.sum(value[..., 5:8] ** 2, axis=-1)
        return (self.gamma - 1.0) * (
            value[..., 4] - 0.5 * momentum_squared / density - 0.5 * magnetic_squared
        )

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
            pressure / (self.gamma - 1.0)
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
        momentum_flux = momentum * normal_velocity[..., None] - normal_magnetic[..., None] * magnetic
        momentum_flux = momentum_flux.at[..., axis_].add(total_pressure)
        energy_flux = (energy + total_pressure) * normal_velocity - normal_magnetic * jnp.sum(
            velocity * magnetic, axis=-1
        )
        magnetic_flux = normal_velocity[..., None] * magnetic - normal_magnetic[..., None] * velocity
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
            sound_squared = self.gamma * self.pressure(state) / density
            magnetic_squared = jnp.sum(magnetic**2, axis=-1) / density
            normal_squared = magnetic[..., axis_] ** 2 / density
            discriminant = jnp.maximum(
                (sound_squared + magnetic_squared) ** 2
                - 4.0 * sound_squared * normal_squared,
                0.0,
            )
            fast = jnp.sqrt(
                0.5
                * (
                    sound_squared
                    + magnetic_squared
                    + jnp.sqrt(discriminant)
                )
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


    def admissible(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return (value[..., 0] >= self.density_floor) & (
            self.pressure(value) >= self.pressure_floor
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return jnp.asarray(state).at[..., 1 + int(axis)].multiply(-1.0)


class ShallowWaterSystem(AbstractAdmissibleSystem):
    """Flat or bathymetric shallow-water conservation system."""

    gravity: float = eqx.field(static=True)
    depth_floor: float = eqx.field(static=True)

    def __init__(
        self,
        dimension: int = 1,
        /,
        *,
        gravity: float = 9.81,
        depth_floor: float = 1e-10,
    ):
        dimension_ = int(dimension)
        gravity_ = float(gravity)
        floor_ = float(depth_floor)
        if dimension_ not in (1, 2):
            raise ValueError("Shallow-water dimension must be one or two.")
        if not np.isfinite(gravity_) or gravity_ <= 0.0 or floor_ <= 0.0:
            raise ValueError("Shallow-water gravity and depth floor must be positive.")
        self.dimension = dimension_
        self.gravity = gravity_
        self.depth_floor = floor_
        self.component_names = (
            "depth",
            *(f"discharge_{axis}" for axis in range(dimension_)),
        )
        self.system_id = canonical_fingerprint(
            {
                "kind": "shallow-water-system",
                "dimension": dimension_,
                "gravity": gravity_,
                "depth_floor": floor_,
            }
        )

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return jnp.concatenate(
            (value[..., :1], value[..., 1:] / value[..., :1]), axis=-1
        )

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = jnp.asarray(primitive)
        return jnp.concatenate(
            (value[..., :1], value[..., :1] * value[..., 1:]), axis=-1
        )

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        value = jnp.asarray(state)
        depth = value[..., 0]
        discharge = value[..., 1:]
        velocity = discharge / depth[..., None]
        normal_velocity = velocity[..., int(axis)]
        momentum_flux = discharge * normal_velocity[..., None]
        momentum_flux = momentum_flux.at[..., int(axis)].add(0.5 * self.gravity * depth**2)
        return jnp.concatenate(
            (discharge[..., int(axis) : int(axis) + 1], momentum_flux), axis=-1
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
            return jnp.abs(state[..., 1 + axis_] / state[..., 0]) + jnp.sqrt(
                self.gravity * state[..., 0]
            )

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


    def admissible(self, state: Array, /) -> Array:
        return jnp.asarray(state)[..., 0] >= self.depth_floor

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return jnp.asarray(state).at[..., 1 + int(axis)].multiply(-1.0)


__all__ = [
    "AbstractAdmissibleSystem",
    "AbstractCharacteristicSystem",
    "AbstractConservationSystem",
    "AbstractEntropySystem",
    "CompressibleNavierStokesSystem",
    "EulerSystem",
    "IdealMHDSystem",
    "MultispeciesEulerSystem",
    "ScalarConservationSystem",
    "ShallowWaterSystem",
]
