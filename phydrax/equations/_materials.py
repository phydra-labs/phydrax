#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class AbstractThermodynamicMaterial(StrictModule, NonTrainableState):
    """Caloric and thermal closure independent of a conservation layout."""

    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def pressure(self, density: Array, specific_internal_energy: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def specific_internal_energy(self, density: Array, pressure: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def temperature(self, density: Array, pressure: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def sound_speed(self, density: Array, pressure: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def specific_enthalpy(self, density: Array, pressure: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def specific_heat_cp(self, density: Array, pressure: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def admissible(self, density: Array, pressure: Array, /) -> Array:
        raise NotImplementedError


class IdealGasMaterial(AbstractThermodynamicMaterial):
    """Calorically perfect ideal gas."""

    gamma: float = eqx.field(static=True)
    gas_constant: float = eqx.field(static=True)

    def __init__(
        self,
        gamma: float = 1.4,
        gas_constant: float = 1.0,
        /,
        *,
        density_floor: float = 1e-12,
        pressure_floor: float = 1e-12,
    ):
        gamma_ = float(gamma)
        gas_constant_ = float(gas_constant)
        density_floor_ = float(density_floor)
        pressure_floor_ = float(pressure_floor)
        if (
            not np.isfinite(gamma_)
            or gamma_ <= 1.0
            or not np.isfinite(gas_constant_)
            or gas_constant_ <= 0.0
            or density_floor_ <= 0.0
            or pressure_floor_ <= 0.0
        ):
            raise ValueError(
                "Ideal-gas parameters and floors must be finite and positive."
            )
        self.gamma = gamma_
        self.gas_constant = gas_constant_
        self.density_floor = density_floor_
        self.pressure_floor = pressure_floor_
        self.material_id = canonical_fingerprint(
            {
                "kind": "ideal-gas-material",
                "gamma": gamma_,
                "gas_constant": gas_constant_,
                "density_floor": density_floor_,
                "pressure_floor": pressure_floor_,
            }
        )

    def pressure(self, density: Array, specific_internal_energy: Array, /) -> Array:
        return (self.gamma - 1.0) * density * specific_internal_energy

    def specific_internal_energy(self, density: Array, pressure: Array, /) -> Array:
        return pressure / ((self.gamma - 1.0) * density)

    def temperature(self, density: Array, pressure: Array, /) -> Array:
        return pressure / (density * self.gas_constant)

    def sound_speed(self, density: Array, pressure: Array, /) -> Array:
        return jnp.sqrt(self.gamma * pressure / density)

    def specific_enthalpy(self, density: Array, pressure: Array, /) -> Array:
        return self.gamma * pressure / ((self.gamma - 1.0) * density)

    def specific_heat_cp(self, density: Array, pressure: Array, /) -> Array:
        del pressure
        return jnp.full_like(
            density,
            self.gamma * self.gas_constant / (self.gamma - 1.0),
        )

    def admissible(self, density: Array, pressure: Array, /) -> Array:
        return (density >= self.density_floor) & (pressure >= self.pressure_floor)


class StiffenedGasMaterial(AbstractThermodynamicMaterial):
    """Calorically perfect stiffened-gas closure."""

    gamma: float = eqx.field(static=True)
    pressure_offset: float = eqx.field(static=True)
    reference_energy: float = eqx.field(static=True)
    heat_capacity: float = eqx.field(static=True)

    def __init__(
        self,
        gamma: float,
        pressure_offset: float,
        heat_capacity: float,
        /,
        *,
        reference_energy: float = 0.0,
        density_floor: float = 1e-12,
        pressure_floor: float = 1e-12,
    ):
        values = tuple(
            float(value)
            for value in (
                gamma,
                pressure_offset,
                heat_capacity,
                reference_energy,
                density_floor,
                pressure_floor,
            )
        )
        gamma_, offset, capacity, reference, density_floor_, pressure_floor_ = values
        if (
            any(not np.isfinite(value) for value in values)
            or gamma_ <= 1.0
            or capacity <= 0.0
            or density_floor_ <= 0.0
            or pressure_floor_ + offset <= 0.0
        ):
            raise ValueError("Stiffened-gas parameters are invalid.")
        self.gamma = gamma_
        self.pressure_offset = offset
        self.heat_capacity = capacity
        self.reference_energy = reference
        self.density_floor = density_floor_
        self.pressure_floor = pressure_floor_
        self.material_id = canonical_fingerprint(
            {
                "kind": "stiffened-gas-material",
                "gamma": gamma_,
                "pressure_offset": offset,
                "heat_capacity": capacity,
                "reference_energy": reference,
                "density_floor": density_floor_,
                "pressure_floor": pressure_floor_,
            }
        )

    def pressure(self, density: Array, specific_internal_energy: Array, /) -> Array:
        return (self.gamma - 1.0) * density * (
            specific_internal_energy - self.reference_energy
        ) - self.gamma * self.pressure_offset

    def specific_internal_energy(self, density: Array, pressure: Array, /) -> Array:
        return self.reference_energy + (pressure + self.gamma * self.pressure_offset) / (
            (self.gamma - 1.0) * density
        )

    def temperature(self, density: Array, pressure: Array, /) -> Array:
        return (
            self.specific_internal_energy(density, pressure) - self.reference_energy
        ) / self.heat_capacity

    def sound_speed(self, density: Array, pressure: Array, /) -> Array:
        return jnp.sqrt(self.gamma * (pressure + self.pressure_offset) / density)

    def specific_enthalpy(self, density: Array, pressure: Array, /) -> Array:
        return self.specific_internal_energy(density, pressure) + pressure / density

    def specific_heat_cp(self, density: Array, pressure: Array, /) -> Array:
        del pressure
        return jnp.full_like(density, self.gamma * self.heat_capacity)

    def admissible(self, density: Array, pressure: Array, /) -> Array:
        return (density >= self.density_floor) & (pressure + self.pressure_offset > 0.0)


@eqx.filter_jit
def _two_material_velocity_squared(velocity: Array, /) -> Array:
    """Return a contraction that remains differentiable under JAX transforms."""

    return ein.contract("...d,...d->...", velocity, velocity)


class TwoMaterialPrimitiveState(StrictModule, NonTrainableState):
    """Primitive state for the five-equation two-material model.

    The two phase densities are *material* densities (rather than partial
    densities).  The conserved partial masses are therefore
    ``alpha_0 * density_0`` and ``(1 - alpha_0) * density_1``.
    """

    density_0: Array
    density_1: Array
    velocity: Array
    pressure: Array
    alpha_0: Array

    @property
    def alpha_1(self) -> Array:
        return 1.0 - self.alpha_0

    @property
    def alpha(self) -> Array:
        """Alias for the material-zero volume fraction."""

        return self.alpha_0

    @property
    def rho_0(self) -> Array:
        return self.density_0

    @property
    def rho_1(self) -> Array:
        return self.density_1

    @property
    def dimension(self) -> int:
        return int(jnp.asarray(self.velocity).shape[-1])

    def as_array(self, /) -> Array:
        """Pack the state as ``[rho0, rho1, velocity..., p, alpha0]``."""

        density_0 = jnp.asarray(self.density_0)
        density_1 = jnp.asarray(self.density_1, dtype=density_0.dtype)
        velocity = jnp.asarray(self.velocity, dtype=density_0.dtype)
        pressure = jnp.asarray(self.pressure, dtype=density_0.dtype)
        alpha_0 = jnp.asarray(self.alpha_0, dtype=density_0.dtype)
        return jnp.concatenate(
            (
                density_0[..., None],
                density_1[..., None],
                velocity,
                pressure[..., None],
                alpha_0[..., None],
            ),
            axis=-1,
        )


class TwoMaterialEOSReport(StrictModule, NonTrainableState):
    """Immutable diagnostics for an affine two-material pressure solve."""

    alpha_0: Array
    alpha_1: Array
    pressure_coefficient: Array
    pressure_intercept: Array
    internal_energy_coefficient: Array
    internal_energy_offset: Array
    pressure: Array
    specific_internal_energy: Array
    temperature: Array
    sound_speed: Array
    admissible: Array

    @property
    def mixture_pressure_coefficient(self) -> Array:
        """Coefficient in ``e_int = A p + C``."""

        return self.internal_energy_coefficient

    @property
    def mixture_internal_energy_coefficient(self) -> Array:
        return self.internal_energy_coefficient

    @property
    def mixture_internal_energy_offset(self) -> Array:
        return self.internal_energy_offset

    @property
    def pressure_offset(self) -> Array:
        """Intercept in ``p = pressure_coefficient * e_int + pressure_offset``."""

        return self.pressure_intercept

    @property
    def energy_coefficient(self) -> Array:
        return self.internal_energy_coefficient

    @property
    def energy_offset(self) -> Array:
        return self.internal_energy_offset

    @property
    def mixture_pressure_intercept(self) -> Array:
        return self.pressure_intercept

    @property
    def internal_energy_intercept(self) -> Array:
        return self.internal_energy_offset


class TwoMaterialEOSClosure(StrictModule, NonTrainableState):
    """Common-pressure closure for two affine caloric materials.

    The supported state layout is
    ``[partial_mass_0, partial_mass_1, momentum..., total_energy, alpha_0]``
    in conserved variables and
    ``[density_0, density_1, velocity..., pressure, alpha_0]`` in primitive
    variables.  No nonlinear or iterative pressure solve is needed: both
    supported materials have an affine relation between pressure and
    specific internal energy.
    """

    material_0: AbstractThermodynamicMaterial
    material_1: AbstractThermodynamicMaterial
    alpha_floor: float = eqx.field(static=True)
    density_floor: float = eqx.field(static=True)
    mass_floor: float = eqx.field(static=True)
    energy_floor: float = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)
    model_variant: str = eqx.field(static=True)

    def __init__(
        self,
        material_0: AbstractThermodynamicMaterial,
        material_1: AbstractThermodynamicMaterial,
        /,
        *,
        alpha_floor: float = 1.0e-12,
        density_floor: float | None = None,
        mass_floor: float = 1.0e-12,
        energy_floor: float = 1.0e-12,
        identity: str | None = None,
    ):
        if not isinstance(material_0, (IdealGasMaterial, StiffenedGasMaterial)):
            raise TypeError(
                "material_0 must be an IdealGasMaterial or StiffenedGasMaterial."
            )
        if not isinstance(material_1, (IdealGasMaterial, StiffenedGasMaterial)):
            raise TypeError(
                "material_1 must be an IdealGasMaterial or StiffenedGasMaterial."
            )
        alpha_floor_ = float(alpha_floor)
        mass_floor_ = float(mass_floor)
        energy_floor_ = float(energy_floor)
        if density_floor is None:
            density_floor_ = max(
                float(material_0.density_floor), float(material_1.density_floor)
            )
        else:
            density_floor_ = float(density_floor)
        if (
            not np.isfinite(alpha_floor_)
            or alpha_floor_ < 0.0
            or alpha_floor_ >= 0.5
            or not np.isfinite(density_floor_)
            or density_floor_ <= 0.0
            or not np.isfinite(mass_floor_)
            or mass_floor_ <= 0.0
            or not np.isfinite(energy_floor_)
            or energy_floor_ <= 0.0
        ):
            raise ValueError(
                "Two-material floors must be finite, positive, and alpha_floor < 0.5."
            )
        self.material_0 = material_0
        self.material_1 = material_1
        self.alpha_floor = alpha_floor_
        self.density_floor = density_floor_
        self.mass_floor = mass_floor_
        self.energy_floor = energy_floor_
        self.model_variant = "kapila-five-equation-v1"
        generated_id = canonical_fingerprint(
            {
                "kind": "two-material-eos-closure",
                "model_variant": self.model_variant,
                "material_0": material_0.material_id,
                "material_1": material_1.material_id,
                "alpha_floor": alpha_floor_,
                "density_floor": density_floor_,
                "mass_floor": mass_floor_,
                "energy_floor": energy_floor_,
                "identity": identity,
            }
        )
        self.closure_id = generated_id

    @property
    def eos_id(self) -> str:
        return self.closure_id

    @property
    def identity(self) -> str:
        return self.closure_id

    @property
    def materials(self) -> tuple[AbstractThermodynamicMaterial, ...]:
        return (self.material_0, self.material_1)

    @staticmethod
    def _array_dtype(value: Array) -> jnp.dtype:
        dtype = jnp.asarray(value).dtype
        if not jnp.issubdtype(dtype, jnp.inexact):
            return jnp.dtype(jnp.float32)
        return dtype

    def _phase_activity(self, alpha_0: Array, /) -> tuple[Array, Array]:
        """Return phase activity while preserving exact pure-phase masks."""

        alpha_0 = jnp.asarray(alpha_0)
        dtype = self._array_dtype(alpha_0)
        alpha_0 = alpha_0.astype(dtype)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        floor = jnp.asarray(self.alpha_floor, dtype=dtype)
        finite = jnp.isfinite(alpha_0)
        active_0 = finite & (alpha_0 > zero) & (alpha_0 >= floor)
        active_1 = finite & (alpha_0 < one) & (alpha_0 <= one - floor)
        return active_0, active_1

    @staticmethod
    def _phase_affine(
        material: AbstractThermodynamicMaterial,
    ) -> tuple[float, float, float]:
        """Return ``(a, reference, q)`` for ``e=reference+(p+q)/(a*rho)``."""

        if isinstance(material, IdealGasMaterial):
            return (
                float(material.gamma - 1.0),
                0.0,
                0.0,
            )
        if isinstance(material, StiffenedGasMaterial):
            return (
                float(material.gamma - 1.0),
                float(material.reference_energy),
                float(material.gamma * material.pressure_offset),
            )
        raise TypeError("Unsupported material in affine two-material closure.")

    @staticmethod
    def _phase_pressure_floor(material: AbstractThermodynamicMaterial) -> float:
        if isinstance(material, IdealGasMaterial):
            return float(material.pressure_floor)
        if isinstance(material, StiffenedGasMaterial):
            return float(-material.pressure_offset + material.pressure_floor)
        raise TypeError("Unsupported material in affine two-material closure.")

    def _phase_coefficients(
        self, alpha_0: Array, mass_0: Array, mass_1: Array, /
    ) -> tuple[Array, Array, Array]:
        """Return ``A``, ``C`` and ``p`` coefficients for ``e_int=A*p+C``."""

        alpha = jnp.asarray(alpha_0)
        dtype = self._array_dtype(alpha)
        alpha = alpha.astype(dtype)
        mass_0 = jnp.asarray(mass_0, dtype=dtype)
        mass_1 = jnp.asarray(mass_1, dtype=dtype)
        alpha_1 = jnp.asarray(1.0, dtype=dtype) - alpha
        a_0, reference_0, q_0 = self._phase_affine(self.material_0)
        a_1, reference_1, q_1 = self._phase_affine(self.material_1)
        coefficient = alpha / jnp.asarray(a_0, dtype=dtype) + alpha_1 / jnp.asarray(
            a_1, dtype=dtype
        )
        offset = (
            mass_0 * jnp.asarray(reference_0, dtype=dtype)
            + mass_1 * jnp.asarray(reference_1, dtype=dtype)
            + alpha * jnp.asarray(q_0 / a_0, dtype=dtype)
            + alpha_1 * jnp.asarray(q_1 / a_1, dtype=dtype)
        )
        return coefficient, offset, jnp.asarray(1.0, dtype=dtype) / coefficient

    @staticmethod
    def _unpack_primitive(value: Array | TwoMaterialPrimitiveState):
        if isinstance(value, TwoMaterialPrimitiveState):
            density_0 = jnp.asarray(value.density_0)
            dtype = TwoMaterialEOSClosure._array_dtype(density_0)
            return (
                density_0.astype(dtype),
                jnp.asarray(value.density_1, dtype=dtype),
                jnp.asarray(value.velocity, dtype=dtype),
                jnp.asarray(value.pressure, dtype=dtype),
                jnp.asarray(value.alpha_0, dtype=dtype),
                True,
            )
        array = jnp.asarray(value)
        if array.ndim == 0 or array.shape[-1] < 5:
            raise ValueError(
                "Primitive state must have trailing layout [rho0,rho1,velocity...,p,alpha0]."
            )
        dtype = TwoMaterialEOSClosure._array_dtype(array)
        array = array.astype(dtype)
        return (
            array[..., 0],
            array[..., 1],
            array[..., 2:-2],
            array[..., -2],
            array[..., -1],
            False,
        )

    @staticmethod
    def _unpack_conserved(value: Array):
        array = jnp.asarray(value)
        if array.ndim == 0 or array.shape[-1] < 5:
            raise ValueError(
                "Conserved state must have trailing layout [m0,m1,momentum...,E,alpha0]."
            )
        dtype = TwoMaterialEOSClosure._array_dtype(array)
        array = array.astype(dtype)
        return (
            array[..., 0],
            array[..., 1],
            array[..., 2:-2],
            array[..., -2],
            array[..., -1],
            dtype,
        )

    def primitive_state(self, value: Array, /) -> TwoMaterialPrimitiveState:
        """Convert a packed primitive array to an immutable state object."""

        density_0, density_1, velocity, pressure, alpha_0, _ = self._unpack_primitive(
            value
        )
        return TwoMaterialPrimitiveState(
            density_0=density_0,
            density_1=density_1,
            velocity=velocity,
            pressure=pressure,
            alpha_0=alpha_0,
        )

    def primitive_to_conserved(
        self, primitive: Array | TwoMaterialPrimitiveState, /
    ) -> Array:
        """Pack a primitive state into five-equation conserved variables."""

        density_0, density_1, velocity, pressure, alpha_0, _ = self._unpack_primitive(
            primitive
        )
        dtype = self._array_dtype(density_0)
        density_1 = jnp.asarray(density_1, dtype=dtype)
        velocity = jnp.asarray(velocity, dtype=dtype)
        pressure = jnp.asarray(pressure, dtype=dtype)
        alpha_0 = jnp.asarray(alpha_0, dtype=dtype)
        alpha_1 = jnp.asarray(1.0, dtype=dtype) - alpha_0
        active_0, active_1 = self._phase_activity(alpha_0)
        safe_alpha_0 = jnp.where(active_0, alpha_0, jnp.asarray(0.0, dtype=dtype))
        safe_alpha_1 = jnp.where(active_1, alpha_1, jnp.asarray(0.0, dtype=dtype))
        mass_0 = safe_alpha_0 * jnp.where(
            jnp.isfinite(density_0) & (density_0 > 0.0),
            density_0,
            jnp.asarray(self.density_floor, dtype=dtype),
        )
        mass_1 = safe_alpha_1 * jnp.where(
            jnp.isfinite(density_1) & (density_1 > 0.0),
            density_1,
            jnp.asarray(self.density_floor, dtype=dtype),
        )
        density = mass_0 + mass_1
        safe_density_0 = jnp.where(
            jnp.isfinite(density_0) & (density_0 > 0.0),
            density_0,
            jnp.asarray(self.density_floor, dtype=dtype),
        )
        safe_density_1 = jnp.where(
            jnp.isfinite(density_1) & (density_1 > 0.0),
            density_1,
            jnp.asarray(self.density_floor, dtype=dtype),
        )
        phase_pressure_0_floor = self._phase_pressure_floor(self.material_0)
        phase_pressure_1_floor = self._phase_pressure_floor(self.material_1)
        safe_pressure_0 = jnp.maximum(
            jnp.where(jnp.isfinite(pressure), pressure, phase_pressure_0_floor),
            jnp.asarray(phase_pressure_0_floor, dtype=dtype),
        )
        safe_pressure_1 = jnp.maximum(
            jnp.where(jnp.isfinite(pressure), pressure, phase_pressure_1_floor),
            jnp.asarray(phase_pressure_1_floor, dtype=dtype),
        )
        internal_energy = (
            safe_alpha_0
            * safe_density_0
            * self.material_0.specific_internal_energy(safe_density_0, safe_pressure_0)
            + safe_alpha_1
            * safe_density_1
            * self.material_1.specific_internal_energy(safe_density_1, safe_pressure_1)
        )
        kinetic = (
            jnp.asarray(0.5, dtype=dtype)
            * _two_material_velocity_squared(velocity)
            * density
        )
        total_energy = internal_energy + kinetic
        momentum = density[..., None] * velocity
        return jnp.concatenate(
            (
                mass_0[..., None],
                mass_1[..., None],
                momentum,
                total_energy[..., None],
                alpha_0[..., None],
            ),
            axis=-1,
        ).astype(dtype)

    def conserved_to_primitive(self, conserved: Array, /) -> Array:
        """Solve the affine common pressure and unpack primitive variables."""

        mass_0, mass_1, momentum, total_energy, alpha_0, dtype = self._unpack_conserved(
            conserved
        )
        alpha_0 = alpha_0.astype(dtype)
        alpha_1 = jnp.asarray(1.0, dtype=dtype) - alpha_0
        density = mass_0 + mass_1
        safe_density = jnp.where(
            jnp.isfinite(density) & (density > 0.0),
            density,
            jnp.asarray(self.density_floor, dtype=dtype),
        )
        velocity = momentum / safe_density[..., None]
        kinetic = (
            jnp.asarray(0.5, dtype=dtype)
            * _two_material_velocity_squared(momentum)
            / safe_density
        )
        internal_energy = total_energy - kinetic
        coefficient, offset, _ = self._phase_coefficients(alpha_0, mass_0, mass_1)
        pressure = (internal_energy - offset) / coefficient
        active_0, active_1 = self._phase_activity(alpha_0)
        safe_alpha_0 = jnp.where(active_0, alpha_0, jnp.asarray(1.0, dtype=dtype))
        safe_alpha_1 = jnp.where(active_1, alpha_1, jnp.asarray(1.0, dtype=dtype))
        density_0 = mass_0 / safe_alpha_0
        density_1 = mass_1 / safe_alpha_1
        return jnp.concatenate(
            (
                density_0[..., None],
                density_1[..., None],
                velocity,
                pressure[..., None],
                alpha_0[..., None],
            ),
            axis=-1,
        ).astype(dtype)

    def report(self, conserved: Array, /) -> TwoMaterialEOSReport:
        """Return the immutable pressure/internal-energy mixture report."""

        mass_0, mass_1, momentum, total_energy, alpha_0, dtype = self._unpack_conserved(
            conserved
        )
        alpha_0 = alpha_0.astype(dtype)
        alpha_1 = jnp.asarray(1.0, dtype=dtype) - alpha_0
        density = mass_0 + mass_1
        safe_density = jnp.where(
            jnp.isfinite(density) & (density > 0.0),
            density,
            jnp.asarray(self.density_floor, dtype=dtype),
        )
        velocity_squared = _two_material_velocity_squared(momentum)
        kinetic = jnp.asarray(0.5, dtype=dtype) * velocity_squared / safe_density
        internal_energy_density = total_energy - kinetic
        coefficient, offset, inverse = self._phase_coefficients(alpha_0, mass_0, mass_1)
        pressure = inverse * internal_energy_density - inverse * offset
        pressure_intercept = -inverse * offset
        primitive = self.conserved_to_primitive(conserved)
        density_0 = primitive[..., 0]
        density_1 = primitive[..., 1]
        phase_pressure = jnp.where(
            jnp.isfinite(pressure),
            pressure,
            jnp.asarray(0.0, dtype=dtype),
        )
        phase_pressure_0_floor = self._phase_pressure_floor(self.material_0)
        phase_pressure_1_floor = self._phase_pressure_floor(self.material_1)
        safe_pressure_0 = jnp.maximum(
            phase_pressure, jnp.asarray(phase_pressure_0_floor, dtype=dtype)
        )
        safe_pressure_1 = jnp.maximum(
            phase_pressure, jnp.asarray(phase_pressure_1_floor, dtype=dtype)
        )
        safe_density_0 = jnp.where(
            jnp.isfinite(density_0) & (density_0 > 0.0),
            density_0,
            jnp.asarray(self.density_floor, dtype=dtype),
        )
        safe_density_1 = jnp.where(
            jnp.isfinite(density_1) & (density_1 > 0.0),
            density_1,
            jnp.asarray(self.density_floor, dtype=dtype),
        )
        temperature_0 = self.material_0.temperature(safe_density_0, safe_pressure_0)
        temperature_1 = self.material_1.temperature(safe_density_1, safe_pressure_1)
        temperature = alpha_0 * temperature_0 + alpha_1 * temperature_1
        sound_squared_0 = (
            self.material_0.sound_speed(
                safe_density_0,
                safe_pressure_0,
            )
            ** 2
        )
        sound_squared_1 = (
            self.material_1.sound_speed(
                safe_density_1,
                safe_pressure_1,
            )
            ** 2
        )
        safe_sound_squared_0 = jnp.where(
            jnp.isfinite(sound_squared_0) & (sound_squared_0 > 0.0),
            sound_squared_0,
            jnp.asarray(1.0, dtype=dtype),
        )
        safe_sound_squared_1 = jnp.where(
            jnp.isfinite(sound_squared_1) & (sound_squared_1 > 0.0),
            sound_squared_1,
            jnp.asarray(1.0, dtype=dtype),
        )
        compressibility = alpha_0 / (
            jnp.maximum(density_0, jnp.asarray(self.density_floor, dtype=dtype))
            * safe_sound_squared_0
        ) + alpha_1 / (
            jnp.maximum(density_1, jnp.asarray(self.density_floor, dtype=dtype))
            * safe_sound_squared_1
        )
        safe_compressibility = jnp.where(
            jnp.isfinite(compressibility) & (compressibility > 0.0),
            compressibility,
            jnp.asarray(1.0, dtype=dtype),
        )
        sound_speed = jnp.sqrt(
            jnp.maximum(
                jnp.asarray(1.0, dtype=dtype) / (safe_density * safe_compressibility),
                jnp.asarray(0.0, dtype=dtype),
            )
        )
        admissible = self._admissible_components(
            mass_0,
            mass_1,
            density,
            total_energy,
            internal_energy_density,
            alpha_0,
            density_0,
            density_1,
            pressure,
        )
        return TwoMaterialEOSReport(
            alpha_0=alpha_0,
            alpha_1=alpha_1,
            pressure_coefficient=inverse,
            pressure_intercept=pressure_intercept,
            internal_energy_coefficient=coefficient,
            internal_energy_offset=offset,
            pressure=pressure,
            specific_internal_energy=internal_energy_density / safe_density,
            temperature=temperature,
            sound_speed=sound_speed,
            admissible=admissible,
        )

    eos_report = report

    def pressure(self, conserved: Array, /) -> Array:
        return self.report(conserved).pressure

    def temperature(self, conserved: Array, /) -> Array:
        return self.report(conserved).temperature

    def sound_speed(self, conserved: Array, /) -> Array:
        return self.report(conserved).sound_speed

    def phase_densities(self, conserved: Array, /) -> tuple[Array, Array]:
        """Return material densities at the common-pressure state."""

        primitive = self.conserved_to_primitive(conserved)
        return primitive[..., 0], primitive[..., 1]

    def _phase_sound_speeds_from_primitive(
        self,
        density_0: Array,
        density_1: Array,
        pressure: Array,
        /,
    ) -> tuple[Array, Array]:
        dtype = self._array_dtype(density_0)
        density_0 = jnp.asarray(density_0, dtype=dtype)
        density_1 = jnp.asarray(density_1, dtype=dtype)
        pressure = jnp.asarray(pressure, dtype=dtype)
        safe_density_0 = jnp.where(
            jnp.isfinite(density_0) & (density_0 > 0.0),
            density_0,
            jnp.asarray(self.density_floor, dtype=dtype),
        )
        safe_density_1 = jnp.where(
            jnp.isfinite(density_1) & (density_1 > 0.0),
            density_1,
            jnp.asarray(self.density_floor, dtype=dtype),
        )
        pressure_floor_0 = jnp.asarray(
            self._phase_pressure_floor(self.material_0), dtype=dtype
        )
        pressure_floor_1 = jnp.asarray(
            self._phase_pressure_floor(self.material_1), dtype=dtype
        )
        finite_pressure = jnp.isfinite(pressure)
        safe_pressure_0 = jnp.maximum(
            jnp.where(finite_pressure, pressure, pressure_floor_0),
            pressure_floor_0,
        )
        safe_pressure_1 = jnp.maximum(
            jnp.where(finite_pressure, pressure, pressure_floor_1),
            pressure_floor_1,
        )
        return (
            self.material_0.sound_speed(safe_density_0, safe_pressure_0),
            self.material_1.sound_speed(safe_density_1, safe_pressure_1),
        )

    def phase_sound_speeds(self, conserved: Array, /) -> tuple[Array, Array]:
        """Return phase sound speeds evaluated at the common mixture pressure."""

        primitive = self.conserved_to_primitive(conserved)
        sound_0, sound_1 = self._phase_sound_speeds_from_primitive(
            primitive[..., 0],
            primitive[..., 1],
            primitive[..., -2],
        )
        valid = self.admissible(conserved)
        nan_0 = jnp.full_like(sound_0, jnp.nan)
        nan_1 = jnp.full_like(sound_1, jnp.nan)
        return jnp.where(valid, sound_0, nan_0), jnp.where(valid, sound_1, nan_1)

    def dilatation_coefficient(self, conserved: Array, /) -> Array:
        r"""Return the Kapila mechanical-equilibrium coefficient ``K``.

        For material zero with volume fraction ``alpha``,

        ``K = alpha*(1-alpha)*(rho1*c1**2-rho0*c0**2)
        / (alpha*rho1*c1**2 + (1-alpha)*rho0*c0**2)``.

        Pure phases return exactly zero. Invalid states return NaN rather than
        contributing a plausible but thermodynamically unsupported source.
        """

        value = jnp.asarray(conserved)
        primitive = self.conserved_to_primitive(value)
        density_0 = primitive[..., 0]
        density_1 = primitive[..., 1]
        pressure = primitive[..., -2]
        alpha_0 = primitive[..., -1]
        dtype = self._array_dtype(density_0)
        one = jnp.asarray(1.0, dtype=dtype)
        zero = jnp.asarray(0.0, dtype=dtype)
        alpha_1 = one - alpha_0
        sound_0, sound_1 = self._phase_sound_speeds_from_primitive(
            density_0,
            density_1,
            pressure,
        )
        stiffness_0 = density_0 * sound_0**2
        stiffness_1 = density_1 * sound_1**2
        denominator = alpha_0 * stiffness_1 + alpha_1 * stiffness_0
        denominator_valid = jnp.isfinite(denominator) & (denominator > 0.0)
        safe_denominator = jnp.where(denominator_valid, denominator, one)
        coefficient = alpha_0 * alpha_1 * (stiffness_1 - stiffness_0) / safe_denominator
        pure = (alpha_0 == zero) | (alpha_0 == one)
        identical_materials = jnp.asarray(
            self.material_0.material_id == self.material_1.material_id
        )
        active_0, active_1 = self._phase_activity(alpha_0)
        phase_0_valid = (
            jnp.isfinite(density_0)
            & (density_0 > 0.0)
            & jnp.isfinite(sound_0)
            & (sound_0 > 0.0)
        )
        phase_1_valid = (
            jnp.isfinite(density_1)
            & (density_1 > 0.0)
            & jnp.isfinite(sound_1)
            & (sound_1 > 0.0)
        )
        acoustics_valid = (
            jnp.where(active_0, phase_0_valid, True)
            & jnp.where(active_1, phase_1_valid, True)
            & jnp.where(pure, True, denominator_valid)
        )
        valid = self.admissible(value) & acoustics_valid & jnp.isfinite(coefficient)
        result = jnp.where(pure | identical_materials, zero, coefficient)
        return jnp.where(valid, result, jnp.full_like(result, jnp.nan))

    def specific_internal_energy(self, conserved: Array, /) -> Array:
        return self.report(conserved).specific_internal_energy

    def _admissible_components(
        self,
        mass_0: Array,
        mass_1: Array,
        density: Array,
        total_energy: Array,
        internal_energy_density: Array,
        alpha_0: Array,
        density_0: Array,
        density_1: Array,
        pressure: Array,
    ) -> Array:
        finite = (
            jnp.isfinite(mass_0)
            & jnp.isfinite(mass_1)
            & jnp.isfinite(density)
            & jnp.isfinite(total_energy)
            & jnp.isfinite(internal_energy_density)
            & jnp.isfinite(alpha_0)
            & jnp.isfinite(density_0)
            & jnp.isfinite(density_1)
            & jnp.isfinite(pressure)
        )
        alpha_tolerance = (
            jnp.asarray(64.0, dtype=alpha_0.dtype) * jnp.finfo(alpha_0.dtype).eps
        )
        alpha_bounded = (alpha_0 >= -alpha_tolerance) & (alpha_0 <= 1.0 + alpha_tolerance)
        alpha_checked = jnp.clip(alpha_0, 0.0, 1.0)
        alpha_valid = alpha_bounded & (
            (alpha_checked == 0.0)
            | (alpha_checked == 1.0)
            | (alpha_checked >= self.alpha_floor)
            & (alpha_checked <= 1.0 - self.alpha_floor)
        )
        active_0, active_1 = self._phase_activity(alpha_checked)
        material_0_valid = self.material_0.admissible(density_0, pressure)
        material_1_valid = self.material_1.admissible(density_1, pressure)
        return (
            finite
            & alpha_valid
            & (mass_0 >= 0.0)
            & (mass_1 >= 0.0)
            & (density >= self.density_floor)
            & (internal_energy_density >= self.energy_floor)
            & jnp.where(active_0, mass_0 >= self.mass_floor, mass_0 == 0.0)
            & jnp.where(active_1, mass_1 >= self.mass_floor, mass_1 == 0.0)
            & jnp.where(active_0, material_0_valid, True)
            & jnp.where(active_1, material_1_valid, True)
        )

    def admissible(self, state: Array | TwoMaterialPrimitiveState, /) -> Array:
        """Return a fail-closed admissibility mask for conserved or primitive data."""

        if isinstance(state, TwoMaterialPrimitiveState):
            density_0, density_1, velocity, pressure, alpha_0, _ = self._unpack_primitive(
                state
            )
            dtype = self._array_dtype(density_0)
            alpha_1 = jnp.asarray(1.0, dtype=dtype) - alpha_0
            mass_0 = alpha_0 * density_0
            mass_1 = alpha_1 * density_1
            density = mass_0 + mass_1
            kinetic = (
                jnp.asarray(0.5, dtype=dtype)
                * _two_material_velocity_squared(velocity)
                * density
            )
            _, _, _ = self._phase_coefficients(alpha_0, mass_0, mass_1)
            safe_density_0 = jnp.where(
                jnp.isfinite(density_0) & (density_0 > 0.0),
                density_0,
                jnp.asarray(self.density_floor, dtype=dtype),
            )
            safe_density_1 = jnp.where(
                jnp.isfinite(density_1) & (density_1 > 0.0),
                density_1,
                jnp.asarray(self.density_floor, dtype=dtype),
            )
            internal = (
                alpha_0
                * safe_density_0
                * self.material_0.specific_internal_energy(safe_density_0, pressure)
                + alpha_1
                * safe_density_1
                * self.material_1.specific_internal_energy(safe_density_1, pressure)
            )
            return self._admissible_components(
                mass_0,
                mass_1,
                density,
                internal + kinetic,
                internal,
                alpha_0,
                density_0,
                density_1,
                pressure,
            )
        mass_0, mass_1, momentum, total_energy, alpha_0, dtype = self._unpack_conserved(
            state
        )
        del momentum
        primitive = self.conserved_to_primitive(state)
        density_0 = primitive[..., 0]
        density_1 = primitive[..., 1]
        pressure = primitive[..., -2]
        density = mass_0 + mass_1
        safe_density = jnp.where(
            jnp.isfinite(density) & (density > 0.0),
            density,
            jnp.asarray(self.density_floor, dtype=dtype),
        )
        velocity = primitive[..., 2:-2]
        kinetic = (
            jnp.asarray(0.5, dtype=dtype)
            * _two_material_velocity_squared(velocity)
            * safe_density
        )
        internal = total_energy - kinetic
        return self._admissible_components(
            mass_0,
            mass_1,
            density,
            total_energy,
            internal,
            alpha_0,
            density_0,
            density_1,
            pressure,
        )

    def admissible_primitive(
        self, primitive: Array | TwoMaterialPrimitiveState, /
    ) -> Array:
        if not isinstance(primitive, TwoMaterialPrimitiveState):
            primitive = self.primitive_state(primitive)
        return self.admissible(primitive)

    def pressure_from_primitive(
        self, primitive: Array | TwoMaterialPrimitiveState, /
    ) -> Array:
        if isinstance(primitive, TwoMaterialPrimitiveState):
            return jnp.asarray(primitive.pressure)
        return self._unpack_primitive(primitive)[3]

    def temperature_from_primitive(
        self, primitive: Array | TwoMaterialPrimitiveState, /
    ) -> Array:
        density_0, density_1, _, pressure, alpha_0, _ = self._unpack_primitive(primitive)
        dtype = self._array_dtype(density_0)
        alpha_0 = jnp.asarray(alpha_0, dtype=dtype)
        alpha_1 = jnp.asarray(1.0, dtype=dtype) - alpha_0
        safe_density_0 = jnp.where(
            jnp.isfinite(density_0) & (density_0 > 0.0),
            density_0,
            jnp.asarray(self.density_floor, dtype=dtype),
        )
        safe_density_1 = jnp.where(
            jnp.isfinite(density_1) & (density_1 > 0.0),
            density_1,
            jnp.asarray(self.density_floor, dtype=dtype),
        )
        safe_pressure = jnp.where(
            jnp.isfinite(pressure),
            pressure,
            jnp.asarray(0.0, dtype=dtype),
        )
        return alpha_0 * self.material_0.temperature(
            safe_density_0, safe_pressure
        ) + alpha_1 * self.material_1.temperature(safe_density_1, safe_pressure)


__all__ = [
    "AbstractThermodynamicMaterial",
    "IdealGasMaterial",
    "StiffenedGasMaterial",
    "TwoMaterialEOSClosure",
    "TwoMaterialEOSReport",
    "TwoMaterialPrimitiveState",
]
