#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ...discretization import PreparedTensorGrid
from ...geometry import RigidFrame
from ._fields import (
    _angular_frequency,
    _complex_field_values,
    _longitudinal_coordinate,
    PlaneFieldSpace,
)


AnalyticPulsePolarization = Literal["scalar", "tangential"]

# CODATA 2018. Susceptibilities below are electric SI susceptibilities, so the
# physical nonlinear polarization is epsilon_0 times the contracted response.
_VACUUM_PERMITTIVITY = 8.854_187_812_8e-12


def _real_finite_array(name: str, value: ArrayLike, shape: tuple[int, ...], /) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if jnp.iscomplexobj(array) or not jnp.issubdtype(array.dtype, jnp.number):
        raise TypeError(f"{name} must be a real numeric array.")
    result = array.astype(jnp.result_type(array.dtype, jnp.float32))
    return eqx.error_if(
        result,
        jnp.any(~jnp.isfinite(result)),
        f"{name} must contain only finite values.",
    )


class InstantaneousScalarSusceptibility(StrictModule):
    """Lossless instantaneous scalar electric susceptibility through third order.

    ``second_order`` and ``third_order`` are the SI electric susceptibilities
    chi(2) and chi(3). For real electric field ``E`` the returned physical
    polarization is ``epsilon_0 * (chi2 * E**2 + chi3 * E**3)``. There is no
    delayed response, ionization, plasma state, or implicit linear material law.
    """

    second_order: Array
    third_order: Array

    def __init__(
        self,
        second_order: ArrayLike = 0.0,
        third_order: ArrayLike = 0.0,
        /,
    ):
        self.second_order = _real_finite_array("second_order", second_order, ())
        self.third_order = _real_finite_array("third_order", third_order, ())

    def physical_polarization(self, electric_field: ArrayLike, /) -> Array:
        """Evaluate the real instantaneous nonlinear polarization in SI units."""
        field = jnp.asarray(electric_field)
        if jnp.iscomplexobj(field) or not jnp.issubdtype(field.dtype, jnp.number):
            raise TypeError("physical scalar electric_field must be real numeric.")
        field = field.astype(jnp.result_type(field.dtype, jnp.float32))
        return jnp.asarray(_VACUUM_PERMITTIVITY, dtype=field.dtype) * (
            self.second_order * field**2 + self.third_order * field**3
        )


class OrientedTensorSusceptibility(StrictModule):
    """Crystal-frame instantaneous chi(2)/chi(3) rotated into the lab frame.

    ``crystal_frame.rotation`` maps crystal components to lab components. The
    frame translation is intentionally irrelevant to this homogeneous local
    response. Tensors use output-first crystal indices ``chi2[i,j,k]`` and
    ``chi3[i,j,k,l]``; no symmetry is inserted or inferred.
    """

    second_order: Array
    third_order: Array
    crystal_frame: RigidFrame

    def __init__(
        self,
        second_order: ArrayLike,
        third_order: ArrayLike,
        crystal_frame: RigidFrame,
        /,
    ):
        if not isinstance(crystal_frame, RigidFrame) or crystal_frame.dimension != 3:
            raise ValueError("crystal_frame must be a three-dimensional RigidFrame.")
        self.second_order = _real_finite_array("second_order", second_order, (3, 3, 3))
        self.third_order = _real_finite_array("third_order", third_order, (3, 3, 3, 3))
        self.crystal_frame = crystal_frame

    def physical_polarization(self, electric_field: ArrayLike, /) -> Array:
        """Evaluate physical polarization for lab-frame real three-vectors."""
        field = jnp.asarray(electric_field)
        if field.ndim < 1 or field.shape[-1] != 3:
            raise ValueError("electric_field must have trailing shape (3,).")
        if jnp.iscomplexobj(field) or not jnp.issubdtype(field.dtype, jnp.number):
            raise TypeError("physical vector electric_field must be real numeric.")
        field = field.astype(jnp.result_type(field.dtype, jnp.float32))
        rotation = self.crystal_frame.rotation.astype(field.dtype)
        crystal_field = contract("ij,...i->...j", rotation, field)
        quadratic = contract(
            "ijk,...j,...k->...i",
            self.second_order,
            crystal_field,
            crystal_field,
        )
        cubic = contract(
            "ijkl,...j,...k,...l->...i",
            self.third_order,
            crystal_field,
            crystal_field,
            crystal_field,
        )
        crystal_polarization = jnp.asarray(_VACUUM_PERMITTIVITY, dtype=field.dtype) * (
            quadratic + cubic
        )
        return contract("ij,...j->...i", rotation, crystal_polarization)


class AnalyticPulseField(StrictModule):
    """Carrier-resolved positive-frequency electric field on a plane and time grid.

    The one-dimensional temporal grid is periodic and Fourier sampled. Values are
    the analytic electric field itself, not a slowly varying envelope. With the
    package phasor convention, a positive-frequency component varies as
    ``exp(-1j * omega * t)`` and the real field is ``real(values)``. The unitary
    forward temporal transform is therefore ``ifft(values, norm='ortho')``.
    DC and the even-grid Nyquist bin are outside the represented analytic band.

    A scalar pulse has shape ``space.shape + temporal_grid.shape``. A tangential
    pulse appends a two-component ``(u, v)`` axis. Independent cases use ``vmap``;
    this value deliberately has no leading batch axes and owns no duplicate grid
    coordinates.
    """

    space: PlaneFieldSpace
    temporal_grid: PreparedTensorGrid
    values: Array
    angular_frequency: Array
    longitudinal_coordinate: Array
    polarization: AnalyticPulsePolarization = eqx.field(static=True)

    def __init__(
        self,
        space: PlaneFieldSpace,
        temporal_grid: PreparedTensorGrid,
        values: ArrayLike,
        angular_frequency: ArrayLike,
        longitudinal_coordinate: ArrayLike,
        /,
        *,
        polarization: AnalyticPulsePolarization = "scalar",
    ):
        if not isinstance(space, PlaneFieldSpace):
            raise TypeError("space must be a PlaneFieldSpace.")
        if not isinstance(temporal_grid, PreparedTensorGrid):
            raise TypeError("temporal_grid must be a PreparedTensorGrid.")
        if len(temporal_grid.shape) != 1:
            raise ValueError(
                "An analytic pulse requires an exactly one-dimensional temporal grid."
            )
        axis = temporal_grid.axes[0]
        if axis.basis != "fourier" or not axis.periodic or axis.primary_entity != "point":
            raise ValueError(
                "The temporal grid must be a periodic point-primary Fourier grid."
            )
        if polarization not in ("scalar", "tangential"):
            raise ValueError("polarization must be 'scalar' or 'tangential'.")
        temporal_shape = temporal_grid.shape
        expected_shape = (
            space.shape + temporal_shape
            if polarization == "scalar"
            else space.shape + temporal_shape + (2,)
        )
        self.space = space
        self.temporal_grid = temporal_grid
        self.values = _complex_field_values("values", values, expected_shape)
        self.angular_frequency = _angular_frequency(angular_frequency)
        self.longitudinal_coordinate = _longitudinal_coordinate(longitudinal_coordinate)
        self.polarization = polarization

    @property
    def temporal_coordinates(self) -> Array:
        return self.temporal_grid.axes[0].nodes

    @property
    def temporal_weights(self) -> Array:
        return self.temporal_grid.quadrature_weights

    @property
    def temporal_size(self) -> int:
        return self.temporal_grid.shape[0]

    @property
    def polarization_components(self) -> int:
        return 1 if self.polarization == "scalar" else 2


def _project_physical_to_analytic(
    physical_polarization: Array,
    positive_frequency_mask: Array,
    temporal_axis: int,
    /,
) -> Array:
    axis = int(temporal_axis) % physical_polarization.ndim
    mask = jnp.asarray(positive_frequency_mask, dtype=bool)
    if mask.shape != (physical_polarization.shape[axis],):
        raise ValueError("positive_frequency_mask must match the temporal-axis length.")
    mask_shape = [1] * physical_polarization.ndim
    mask_shape[axis] = mask.size
    resolved_mask = mask.reshape(tuple(mask_shape))
    physical_spectrum = jnp.fft.ifft(physical_polarization, axis=axis, norm="ortho")
    analytic_spectrum = jnp.where(resolved_mask, 2.0 * physical_spectrum, 0.0)
    return jnp.fft.fft(analytic_spectrum, axis=axis, norm="ortho")


def instantaneous_nonlinear_polarization(
    susceptibility: InstantaneousScalarSusceptibility | OrientedTensorSusceptibility,
    analytic_electric_field: ArrayLike,
    positive_frequency_mask: ArrayLike,
    /,
    *,
    temporal_axis: int = -1,
) -> Array:
    """Return the exact represented-band analytic part of an instantaneous response.

    The real field is reconstructed as ``real(E_plus)``. Its physical nonlinear
    polarization is evaluated before a unitary temporal transform projects onto
    strictly positive ``exp(-1j*omega*t)`` bins. Positive bins are doubled; DC,
    non-positive bins, and the even-grid Nyquist bin are zero. This construction
    retains both sum- and difference-frequency mixing and fixes the chi factors
    without a rotating-wave or envelope approximation.
    """
    analytic = jnp.asarray(analytic_electric_field)
    if not jnp.iscomplexobj(analytic):
        raise TypeError("analytic_electric_field must be complex.")
    axis = int(temporal_axis) % analytic.ndim
    if isinstance(susceptibility, InstantaneousScalarSusceptibility):
        physical = susceptibility.physical_polarization(jnp.real(analytic))
    elif isinstance(susceptibility, OrientedTensorSusceptibility):
        if analytic.shape[-1] != 3 or axis == analytic.ndim - 1:
            raise ValueError(
                "Tensor response requires lab-vector fields with trailing shape (3,) "
                "and a distinct temporal axis."
            )
        physical = susceptibility.physical_polarization(jnp.real(analytic))
    else:
        raise TypeError(
            "susceptibility must be InstantaneousScalarSusceptibility or "
            "OrientedTensorSusceptibility."
        )
    return _project_physical_to_analytic(
        physical,
        jnp.asarray(positive_frequency_mask, dtype=bool),
        axis,
    )


__all__ = [
    "AnalyticPulseField",
    "InstantaneousScalarSusceptibility",
    "OrientedTensorSusceptibility",
    "instantaneous_nonlinear_polarization",
]
