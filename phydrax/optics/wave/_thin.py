#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._fields import PlaneFieldSpace, ScalarPlaneField, TangentialPlaneField


PlaneField = ScalarPlaneField | TangentialPlaneField


def _complex_transmission(
    name: str,
    value: ArrayLike,
    shape: tuple[int, ...],
    /,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if not jnp.issubdtype(array.dtype, jnp.number) or jnp.issubdtype(
        array.dtype, jnp.bool_
    ):
        raise TypeError(f"{name} must be numeric.")
    transmission = array.astype(jnp.result_type(array.dtype, jnp.complex64))
    return eqx.error_if(
        transmission,
        jnp.any(~jnp.isfinite(jnp.real(transmission)))
        | jnp.any(~jnp.isfinite(jnp.imag(transmission))),
        f"{name} must be finite.",
    )


def _require_field_space(field: PlaneField, space: PlaneFieldSpace, /) -> None:
    if not isinstance(field, (ScalarPlaneField, TangentialPlaneField)):
        raise TypeError("field must be a ScalarPlaneField or TangentialPlaneField.")
    if field.space.space_id != space.space_id:
        raise ValueError("field and thin transmission must use the same plane space.")


class ScalarThinTransmission(StrictModule):
    """A sampled scalar thin mask acting identically on both Jones components."""

    space: PlaneFieldSpace
    transmission: Array
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: PlaneFieldSpace,
        transmission: ArrayLike,
        /,
        *,
        operator_id: str | None = None,
    ):
        if not isinstance(space, PlaneFieldSpace):
            raise TypeError("space must be a PlaneFieldSpace.")
        values = _complex_transmission("transmission", transmission, space.shape)
        generated = canonical_fingerprint(
            {
                "kind": "scalar-thin-transmission",
                "space": space.space_id,
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            }
        )
        identifier = generated if operator_id is None else str(operator_id)
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.space = space
        self.transmission = values
        self.operator_id = identifier

    @property
    def coefficient_count(self) -> int:
        return prod(self.transmission.shape)

    def apply(self, field: PlaneField, /) -> PlaneField:
        _require_field_space(field, self.space)
        if isinstance(field, ScalarPlaneField):
            return ScalarPlaneField(
                self.space,
                self.transmission * field.values,
                field.angular_frequency,
                field.longitudinal_coordinate,
            )
        return TangentialPlaneField(
            self.space,
            self.transmission[..., None] * field.values,
            field.angular_frequency,
            field.longitudinal_coordinate,
        )

    def __call__(self, field: PlaneField, /) -> PlaneField:
        return self.apply(field)


class JonesThinTransmission(StrictModule):
    """A sampled 2-by-2 Jones action in the plane's tangential basis."""

    space: PlaneFieldSpace
    transmission: Array
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: PlaneFieldSpace,
        transmission: ArrayLike,
        /,
        *,
        operator_id: str | None = None,
    ):
        if not isinstance(space, PlaneFieldSpace):
            raise TypeError("space must be a PlaneFieldSpace.")
        shape = space.shape + (2, 2)
        values = _complex_transmission("transmission", transmission, shape)
        generated = canonical_fingerprint(
            {
                "kind": "jones-thin-transmission",
                "space": space.space_id,
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            }
        )
        identifier = generated if operator_id is None else str(operator_id)
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.space = space
        self.transmission = values
        self.operator_id = identifier

    @property
    def coefficient_count(self) -> int:
        return prod(self.transmission.shape)

    def apply(self, field: TangentialPlaneField, /) -> TangentialPlaneField:
        if not isinstance(field, TangentialPlaneField):
            raise TypeError("A Jones thin transmission requires a TangentialPlaneField.")
        if field.space.space_id != self.space.space_id:
            raise ValueError("field and thin transmission must use the same plane space.")
        values = ein.contract("...ij,...j->...i", self.transmission, field.values)
        return TangentialPlaneField(
            self.space,
            values,
            field.angular_frequency,
            field.longitudinal_coordinate,
        )

    def __call__(self, field: TangentialPlaneField, /) -> TangentialPlaneField:
        return self.apply(field)


def thin_lens(
    space: PlaneFieldSpace,
    focal_length: ArrayLike,
    medium_wavenumber: ArrayLike,
    /,
    *,
    operator_id: str | None = None,
) -> ScalarThinTransmission:
    """Construct an ideal paraxial thin lens from explicit physical parameters.

    The returned sampled action is ``exp(-i k (u² + v²) / (2 f))``. A negative
    focal length is valid for a diverging lens. The ideal lens wavenumber is real;
    absorption belongs in propagation or in an explicitly supplied thin mask.
    """
    if not isinstance(space, PlaneFieldSpace):
        raise TypeError("space must be a PlaneFieldSpace.")
    supplied_focal = jnp.asarray(focal_length)
    if supplied_focal.shape != ():
        raise ValueError("focal_length must be a scalar.")
    if (
        jnp.iscomplexobj(supplied_focal)
        or not jnp.issubdtype(supplied_focal.dtype, jnp.number)
        or jnp.issubdtype(supplied_focal.dtype, jnp.bool_)
    ):
        raise TypeError("focal_length must be real numeric data.")
    focal = supplied_focal.astype(jnp.result_type(supplied_focal.dtype, jnp.float32))
    focal = eqx.error_if(
        focal,
        (~jnp.isfinite(focal)) | (focal == 0.0),
        "focal_length must be finite and nonzero.",
    )
    supplied = jnp.asarray(medium_wavenumber)
    if supplied.shape != ():
        raise ValueError("medium_wavenumber must be a scalar.")
    if not jnp.issubdtype(supplied.dtype, jnp.number) or jnp.issubdtype(
        supplied.dtype, jnp.bool_
    ):
        raise TypeError("medium_wavenumber must be numeric.")
    wavenumber = supplied.astype(jnp.result_type(supplied.dtype, jnp.complex64))
    wavenumber = eqx.error_if(
        wavenumber,
        (~jnp.isfinite(jnp.real(wavenumber)))
        | (~jnp.isfinite(jnp.imag(wavenumber)))
        | (jnp.real(wavenumber) <= 0.0)
        | (jnp.imag(wavenumber) != 0.0),
        "An ideal thin lens requires a finite positive real medium_wavenumber.",
    )
    coordinates = space.transverse_coordinates
    radius_squared = jnp.sum(coordinates * coordinates, axis=-1)
    transmission = jnp.exp(-0.5j * jnp.real(wavenumber) * radius_squared / focal)
    identifier = (
        canonical_fingerprint(
            {
                "kind": "ideal-thin-lens",
                "space": space.space_id,
                "convention": "exp(-i*k*r2/(2*f))",
            }
        )
        if operator_id is None
        else str(operator_id)
    )
    return ScalarThinTransmission(space, transmission, operator_id=identifier)


__all__ = [
    "JonesThinTransmission",
    "ScalarThinTransmission",
    "thin_lens",
]
