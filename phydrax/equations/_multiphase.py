#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compressible two-material volume-of-fluid equations.

The conservative state is, explicitly,
``[m0, m1, momentum[dimension], total_energy, alpha0]``.  Consequently a
``dimension``-dimensional layout has ``dimension + 4`` components (the
``d + 4`` count is intentional: both partial masses and the volume fraction
are retained in addition to the Euler variables).
"""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._hyperbolic_systems import AbstractAdmissibleSystem
from ._materials import TwoMaterialEOSClosure, TwoMaterialPrimitiveState


class TwoMaterialVOFStateLayout(StrictModule, NonTrainableState):
    """Static component layout for the compressible two-material VOF state."""

    dimension: int = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    momentum_start: int = eqx.field(static=True)
    momentum_stop: int = eqx.field(static=True)
    energy_index: int = eqx.field(static=True)
    alpha_index: int = eqx.field(static=True)

    def __init__(self, dimension: int = 1, /):
        dimension_ = int(dimension)
        if dimension_ not in (1, 2, 3):
            raise ValueError("Two-material VOF dimension must be one, two, or three.")
        momentum_start = 2
        momentum_stop = momentum_start + dimension_
        self.dimension = dimension_
        self.component_names = (
            "partial_mass_0",
            "partial_mass_1",
            *(f"momentum_{axis}" for axis in range(dimension_)),
            "total_energy",
            "alpha_0",
        )
        self.momentum_start = momentum_start
        self.momentum_stop = momentum_stop
        self.energy_index = momentum_stop
        self.alpha_index = momentum_stop + 1

    @property
    def component_count(self) -> int:
        """Number of conserved components (``dimension + 4``)."""

        return len(self.component_names)

    @property
    def momentum_slice(self) -> slice:
        return slice(self.momentum_start, self.momentum_stop)


class TwoMaterialVOFDiagnostics(StrictModule, NonTrainableState):
    """Thermodynamic diagnostics shared by a two-material VOF system."""

    eos: TwoMaterialEOSClosure
    layout: TwoMaterialVOFStateLayout
    diagnostics_id: str = eqx.field(static=True)

    def __init__(self, eos: TwoMaterialEOSClosure, dimension: int = 1, /):
        if not isinstance(eos, TwoMaterialEOSClosure):
            raise TypeError("eos must be a TwoMaterialEOSClosure.")
        self.eos = eos
        self.layout = TwoMaterialVOFStateLayout(dimension)
        self.diagnostics_id = canonical_fingerprint(
            {
                "kind": "two-material-vof-diagnostics",
                "dimension": self.layout.dimension,
                "model_variant": eos.model_variant,
                "eos": eos.closure_id,
            }
        )

    def _state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.ndim == 0 or int(value.shape[-1]) != self.layout.component_count:
            raise ValueError(
                "Two-material VOF state must end in "
                f"{self.layout.component_count} components; got {value.shape}."
            )
        return value

    def pressure(self, state: ArrayLike, /) -> Array:
        value = self._state(state)
        return jnp.asarray(self.eos.pressure(value))

    def sound_speed(self, state: ArrayLike, /) -> Array:
        value = self._state(state)
        return jnp.asarray(self.eos.sound_speed(value))

    def phase_densities(self, state: ArrayLike, /) -> tuple[Array, Array]:
        value = self._state(state)
        return self.eos.phase_densities(value)

    def phase_sound_speeds(self, state: ArrayLike, /) -> tuple[Array, Array]:
        value = self._state(state)
        return self.eos.phase_sound_speeds(value)

    def dilatation_coefficient(self, state: ArrayLike, /) -> Array:
        value = self._state(state)
        return self.eos.dilatation_coefficient(value)

    def admissible(self, state: ArrayLike, /) -> Array:
        value = self._state(state)
        finite = jnp.all(jnp.isfinite(value), axis=-1)
        alpha = value[..., self.layout.alpha_index]
        bounded_alpha = (alpha >= 0.0) & (alpha <= 1.0)
        closure_valid = jnp.asarray(self.eos.admissible(value), dtype=bool)
        return finite & bounded_alpha & closure_valid


def _primitive_array(primitive: Any, dimension: int, /) -> Array:
    """Normalize a closure primitive record to its packed array form."""
    if isinstance(primitive, TwoMaterialPrimitiveState):
        primitive = primitive.as_array()
    value = jnp.asarray(primitive)
    expected = dimension + 4
    if value.ndim == 0 or int(value.shape[-1]) != expected:
        raise ValueError(
            "Two-material primitive state must end in "
            f"{expected} components; got {value.shape}."
        )
    return value


class TwoMaterialVOFSystem(AbstractAdmissibleSystem):
    """Compressible two-material VOF conservation system.

    The extensive variables use conservative Euler fluxes.  The alpha
    component of the ordinary physical Riemann flux is exactly zero until a
    PLIC/aperture stage supplies :meth:`phase_transport_flux`; capillary terms
    are deliberately not part of this physical flux.
    """

    eos: TwoMaterialEOSClosure
    layout: TwoMaterialVOFStateLayout
    diagnostics: TwoMaterialVOFDiagnostics

    def __init__(self, dimension: int = 1, /, *, eos: TwoMaterialEOSClosure):
        if not isinstance(eos, TwoMaterialEOSClosure):
            raise TypeError("eos must be a TwoMaterialEOSClosure.")
        self.layout = TwoMaterialVOFStateLayout(dimension)
        self.dimension = self.layout.dimension
        self.component_names = self.layout.component_names
        self.eos = eos
        self.diagnostics = TwoMaterialVOFDiagnostics(eos, self.dimension)
        self.system_id = canonical_fingerprint(
            {
                "kind": "two-material-vof-system",
                "dimension": self.dimension,
                "model_variant": eos.model_variant,
                "eos": eos.closure_id,
                "components": self.component_names,
            }
        )

    @property
    def component_count(self) -> int:
        return self.layout.component_count

    @property
    def alpha_index(self) -> int:
        return self.layout.alpha_index

    def primitive_velocity(self, primitive: Array, /) -> Array:
        return primitive[..., 2 : 2 + self.dimension]

    def with_primitive_velocity(self, primitive: Array, velocity: Array, /) -> Array:
        return primitive.at[..., 2 : 2 + self.dimension].set(velocity)

    def _state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.ndim == 0 or int(value.shape[-1]) != self.component_count:
            raise ValueError(
                "Two-material VOF state must end in "
                f"{self.component_count} components; got {value.shape}."
            )
        return value

    def _axis(self, axis: int, /) -> int:
        axis_ = int(axis)
        if axis_ < 0 or axis_ >= self.dimension:
            raise ValueError("Two-material VOF flux axis is out of range.")
        return axis_

    def _normal(self, normal: ArrayLike, /) -> Array:
        value = jnp.asarray(normal)
        if value.ndim == 0 or int(value.shape[-1]) != self.dimension:
            raise ValueError(
                "Two-material VOF normals must have a trailing dimension of "
                f"{self.dimension}; got {value.shape}."
            )
        return value

    def pressure(self, state: ArrayLike, /) -> Array:
        return self.diagnostics.pressure(state)

    def sound_speed(self, state: ArrayLike, /) -> Array:
        return self.diagnostics.sound_speed(state)

    def phase_densities(self, state: ArrayLike, /) -> tuple[Array, Array]:
        return self.diagnostics.phase_densities(state)

    def phase_sound_speeds(self, state: ArrayLike, /) -> tuple[Array, Array]:
        return self.diagnostics.phase_sound_speeds(state)

    def dilatation_coefficient(self, state: ArrayLike, /) -> Array:
        return self.diagnostics.dilatation_coefficient(state)

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = self._state(state)
        return _primitive_array(self.eos.conserved_to_primitive(value), self.dimension)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = jnp.asarray(primitive)
        if value.ndim == 0 or int(value.shape[-1]) != self.component_count:
            raise ValueError(
                "Two-material primitive state must end in "
                f"{self.component_count} components; got {value.shape}."
            )
        return jnp.asarray(self.eos.primitive_to_conserved(value))

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        axis_ = self._axis(axis)
        value = self._state(state)
        primitive = self.conserved_to_primitive(value)
        velocity = primitive[..., 2 : 2 + self.dimension]
        normal_velocity = velocity[..., axis_]
        pressure = self.pressure(value)
        momentum = value[..., self.layout.momentum_slice]
        energy = value[..., self.layout.energy_index]
        flux = jnp.zeros_like(value)
        flux = flux.at[..., 0].set(value[..., 0] * normal_velocity)
        flux = flux.at[..., 1].set(value[..., 1] * normal_velocity)
        flux = flux.at[..., self.layout.momentum_slice].set(
            momentum * normal_velocity[..., None]
        )
        flux = flux.at[..., self.layout.momentum_start + axis_].add(pressure)
        flux = flux.at[..., self.layout.energy_index].set(
            (energy + pressure) * normal_velocity
        )
        return flux

    def phase_transport_flux(
        self, alpha_face: ArrayLike, volume_flux: ArrayLike, /
    ) -> Array:
        """Return the PLIC/aperture phase flux ``alpha_face * volume_flux``.

        The physical Riemann flux intentionally leaves the volume-fraction
        component at zero.  A finite-volume VOF update must supply the
        aperture (or PLIC face fraction) selected by its interface transport
        stage; using the cell alpha here would silently lose conservative
        dilatation coupling.
        """

        alpha = jnp.asarray(alpha_face)
        flux = jnp.asarray(volume_flux)
        if not jnp.issubdtype(alpha.dtype, jnp.inexact):
            alpha = alpha.astype(jnp.result_type(alpha, flux, float))
        if not jnp.issubdtype(flux.dtype, jnp.inexact):
            flux = flux.astype(jnp.result_type(alpha, flux, float))
        if alpha.shape != flux.shape:
            raise ValueError(
                "alpha_face and volume_flux must have matching shapes; "
                f"got {alpha.shape} and {flux.shape}."
            )
        valid = jnp.isfinite(alpha) & jnp.isfinite(flux) & (alpha >= 0.0) & (alpha <= 1.0)
        return jnp.where(
            valid,
            alpha * flux,
            jnp.full_like(alpha * flux, jnp.nan),
        )

    def volume_fraction_source(
        self,
        alpha: ArrayLike,
        divergence: ArrayLike,
        state: ArrayLike,
        /,
    ) -> Array:
        """Return the conservative Kapila source ``(alpha + K) div(u)``."""

        value = self._state(state)
        coefficient = self.dilatation_coefficient(value)
        dtype = coefficient.dtype
        alpha_value = jnp.asarray(alpha, dtype=dtype)
        divergence_value = jnp.asarray(divergence, dtype=dtype)
        alpha_value, divergence_value, coefficient = jnp.broadcast_arrays(
            alpha_value,
            divergence_value,
            coefficient,
        )
        source = (alpha_value + coefficient) * divergence_value
        valid = (
            jnp.isfinite(alpha_value)
            & (alpha_value >= 0.0)
            & (alpha_value <= 1.0)
            & jnp.isfinite(divergence_value)
            & jnp.isfinite(coefficient)
            & jnp.isfinite(source)
        )
        return jnp.where(valid, source, jnp.full_like(source, jnp.nan))

    def normal_volume_flux(
        self, left: ArrayLike, right: ArrayLike, normal: ArrayLike, /
    ) -> Array:
        left_ = self._state(left)
        right_ = self._state(right)
        normal_ = self._normal(normal)
        left_velocity = self.conserved_to_primitive(left_)[..., 2 : 2 + self.dimension]
        right_velocity = self.conserved_to_primitive(right_)[..., 2 : 2 + self.dimension]
        return 0.5 * jnp.sum((left_velocity + right_velocity) * normal_, axis=-1)

    def phase_density_fluxes(
        self,
        left: ArrayLike,
        right: ArrayLike,
        phase0_volume_flux: ArrayLike,
        phase1_volume_flux: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        left_ = self._state(left)
        right_ = self._state(right)
        primitive_left = self.conserved_to_primitive(left_)
        primitive_right = self.conserved_to_primitive(right_)
        density0 = 0.5 * (primitive_left[..., 0] + primitive_right[..., 0])
        density1 = 0.5 * (primitive_left[..., 1] + primitive_right[..., 1])
        return (
            density0 * jnp.asarray(phase0_volume_flux),
            density1 * jnp.asarray(phase1_volume_flux),
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
        axis_ = self._axis(axis)
        basis = jnp.eye(self.dimension, dtype=jnp.result_type(left, right, float))[axis_]
        return self.normal_signal_bounds(left, right, basis)

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
        left_ = self._state(left)
        right_ = self._state(right)
        normal_ = self._normal(normal)
        left_primitive = self.conserved_to_primitive(left_)
        right_primitive = self.conserved_to_primitive(right_)
        left_velocity = left_primitive[..., 2 : 2 + self.dimension]
        right_velocity = right_primitive[..., 2 : 2 + self.dimension]
        left_normal_velocity = oe.contract("...i,...i->...", left_velocity, normal_)
        right_normal_velocity = oe.contract("...i,...i->...", right_velocity, normal_)
        left_sound = self.sound_speed(left_)
        right_sound = self.sound_speed(right_)
        return (
            jnp.minimum(
                left_normal_velocity - left_sound, right_normal_velocity - right_sound
            ),
            jnp.maximum(
                left_normal_velocity + left_sound, right_normal_velocity + right_sound
            ),
        )

    def admissible(self, state: Array, /) -> Array:
        return self.diagnostics.admissible(state)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        value = self._state(state)
        axis_ = self._axis(axis)
        return value.at[..., self.layout.momentum_start + axis_].multiply(-1.0)


__all__ = [
    "TwoMaterialVOFDiagnostics",
    "TwoMaterialVOFStateLayout",
    "TwoMaterialVOFSystem",
]
