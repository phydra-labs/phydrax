#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._geometry_precision import GeometryPrecisionPolicy
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._space import TensorSpectralDiscretization


class PeriodicLerayProjector(StrictModule, NonTrainableState):
    """Exact modal incompressibility projector on an all-Fourier tensor space."""

    discretization: TensorSpectralDiscretization
    wavenumbers: tuple[Array, ...]
    wavenumber_squared: Array
    inverse_wavenumber_squared: Array
    admissibility_mask: Array
    spatial_dimension: int = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)

    def __init__(self, discretization: TensorSpectralDiscretization, /):
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        dimension = len(discretization.axes)
        if dimension not in (2, 3) or any(
            axis.family != "fourier" for axis in discretization.axes
        ):
            raise ValueError(
                "Periodic incompressibility requires two or three Fourier axes."
            )
        coefficient_dtype = jnp.dtype(discretization.plan.precision.coefficient_dtype)
        real_dtype = jnp.empty((), dtype=coefficient_dtype).real.dtype
        wave_values = []
        masks = []
        for axis_index, axis in enumerate(discretization.axes):
            values = (
                2.0
                * jnp.asarray(jnp.pi, dtype=real_dtype)
                * axis.modes.mode_numbers.astype(real_dtype)
                / axis.length.astype(real_dtype)
            )
            shape = [1] * dimension
            shape[axis_index] = axis.mode_count
            wave_values.append(
                jnp.broadcast_to(values.reshape(tuple(shape)), discretization.modal_shape)
            )
            masks.append(
                jnp.broadcast_to(
                    (~axis.modes.nyquist_mask).reshape(tuple(shape)),
                    discretization.modal_shape,
                )
            )
        squared = jnp.zeros(discretization.modal_shape, dtype=real_dtype)
        admissible = jnp.ones(discretization.modal_shape, dtype=bool)
        for values, mask in zip(wave_values, masks, strict=True):
            squared = squared + values**2
            admissible = admissible & mask
        safe = jnp.where(squared > 0.0, squared, jnp.ones_like(squared))
        inverse = jnp.where(squared > 0.0, 1.0 / safe, jnp.zeros_like(squared))
        state_shape = discretization.modal_shape + (dimension,)
        identifier = canonical_fingerprint(
            {
                "kind": "periodic-leray-projector-v1",
                "discretization": discretization.prepared_id,
                "dimension": dimension,
                "state_shape": list(state_shape),
                "nyquist_policy": "zero-self-conjugate",
            }
        )
        self.discretization = discretization
        self.wavenumbers = tuple(wave_values)
        self.wavenumber_squared = squared
        self.inverse_wavenumber_squared = inverse
        self.admissibility_mask = admissible
        self.spatial_dimension = dimension
        self.state_shape = state_shape
        self.projector_id = identifier

    def validate_state(self, state: ArrayLike, /, *, owner: str = "Velocity") -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"{owner} must have modal velocity shape {self.state_shape}; "
                f"got {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError(f"{owner} must use complex modal coefficients.")
        return value

    def zero_forbidden_modes(self, state: ArrayLike, /) -> Array:
        """Remove self-conjugate Nyquist modes incompatible with odd derivatives."""
        value = self.validate_state(state)
        return value * self.admissibility_mask[..., None]

    def divergence(self, state: ArrayLike, /) -> Array:
        """Return modal divergence with the scalar modal shape."""
        value = self.zero_forbidden_modes(state)
        result = jnp.zeros(self.discretization.modal_shape, dtype=value.dtype)
        for component, wave in enumerate(self.wavenumbers):
            result = result + 1j * wave.astype(value.dtype) * value[..., component]
        return result

    def project(self, state: ArrayLike, /) -> Array:
        """Apply the modewise Leray projector and the real-field Nyquist policy."""
        value = self.zero_forbidden_modes(state)
        longitudinal = jnp.zeros(self.discretization.modal_shape, dtype=value.dtype)
        for component, wave in enumerate(self.wavenumbers):
            longitudinal = longitudinal + wave.astype(value.dtype) * value[..., component]
        components = []
        inverse = self.inverse_wavenumber_squared.astype(value.real.dtype)
        for component, wave in enumerate(self.wavenumbers):
            components.append(
                value[..., component] - wave.astype(value.dtype) * inverse * longitudinal
            )
        return self.zero_forbidden_modes(jnp.stack(tuple(components), axis=-1))

    def pressure_from_unconstrained_rhs(self, rhs: ArrayLike, /) -> Array:
        """Recover mean-zero pressure from an unconstrained modal momentum rate."""
        value = self.zero_forbidden_modes(rhs)
        longitudinal = jnp.zeros(self.discretization.modal_shape, dtype=value.dtype)
        for component, wave in enumerate(self.wavenumbers):
            longitudinal = longitudinal + wave.astype(value.dtype) * value[..., component]
        pressure = (
            -1j * self.inverse_wavenumber_squared.astype(value.dtype) * longitudinal
        )
        return pressure * self.admissibility_mask

    def divergence_norm(self, state: ArrayLike, /) -> Array:
        return GeometryPrecisionPolicy().norm(self.divergence(state).reshape((-1,)))

    @property
    def state_size(self) -> int:
        return prod(self.state_shape)


class IncompressibleSpectralDiagnostics(StrictModule):
    """Physical, modal, and energy-balance evidence for one spectral state."""

    kinetic_energy: Array
    nonlinear_energy_rate: Array
    forcing_power: Array
    viscous_energy_rate: Array
    dissipation: Array
    semidiscrete_energy_rate: Array
    energy_balance_defect: Array
    divergence_norm: Array
    imaginary_leakage: Array
    forbidden_mode_norm: Array
    pressure_gauge_residual: Array
    projector_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        kinetic_energy: ArrayLike,
        nonlinear_energy_rate: ArrayLike,
        forcing_power: ArrayLike,
        viscous_energy_rate: ArrayLike,
        dissipation: ArrayLike,
        semidiscrete_energy_rate: ArrayLike,
        energy_balance_defect: ArrayLike,
        divergence_norm: ArrayLike,
        imaginary_leakage: ArrayLike,
        forbidden_mode_norm: ArrayLike,
        pressure_gauge_residual: ArrayLike,
        projector_id: str,
    ):
        identifier = str(projector_id)
        if not identifier:
            raise ValueError("projector_id must be non-empty.")
        self.kinetic_energy = jnp.asarray(kinetic_energy)
        self.nonlinear_energy_rate = jnp.asarray(nonlinear_energy_rate)
        self.forcing_power = jnp.asarray(forcing_power)
        self.viscous_energy_rate = jnp.asarray(viscous_energy_rate)
        self.dissipation = jnp.asarray(dissipation)
        self.semidiscrete_energy_rate = jnp.asarray(semidiscrete_energy_rate)
        self.energy_balance_defect = jnp.asarray(energy_balance_defect)
        self.divergence_norm = jnp.asarray(divergence_norm)
        self.imaginary_leakage = jnp.asarray(imaginary_leakage)
        self.forbidden_mode_norm = jnp.asarray(forbidden_mode_norm)
        self.pressure_gauge_residual = jnp.asarray(pressure_gauge_residual)
        self.projector_id = identifier


__all__ = ["IncompressibleSpectralDiagnostics", "PeriodicLerayProjector"]
