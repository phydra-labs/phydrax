#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ._boundary_cascade import prepare_layer_boundary
from ._factorization import _dense_solve
from ._layer import recover_longitudinal_fields
from ._runtime import (
    FourierModalSolveResult,
    PreparedFourierModalLayer,
    PreparedFourierModalMaxwell,
)


class FourierModalFieldResult(StrictModule):
    electric_harmonics: Array
    magnetic_harmonics: Array
    electric_field: Array
    magnetic_field: Array
    longitudinal_offset: Array
    layer_id: str = eqx.field(static=True)


class DiffractionOrderFarField(StrictModule):
    wavevectors: Array
    directions: Array
    polar_angle: Array
    azimuthal_angle: Array
    power: Array
    propagating: Array
    side: str = eqx.field(static=True)


def _prepared_layer_location(
    prepared: PreparedFourierModalMaxwell,
    layer_index: int,
    /,
) -> tuple[int, PreparedFourierModalLayer]:
    requested = int(layer_index)
    current = 0
    for element_index, element in enumerate(prepared.elements):
        if not isinstance(element, PreparedFourierModalLayer):
            continue
        if current == requested:
            return element_index, element
        current += 1
    raise IndexError(f"Layer index {requested} is out of range.")


def fields_in_layer(
    prepared: PreparedFourierModalMaxwell,
    result: FourierModalSolveResult,
    layer_index: int,
    longitudinal_offset: ArrayLike,
    /,
    *,
    coordinates: ArrayLike | None = None,
) -> FourierModalFieldResult:
    if not result.boundary_electric_fields:
        raise ValueError("The prepared solve did not retain boundary fields.")
    element_index, layer = _prepared_layer_location(prepared, layer_index)
    offset = jnp.asarray(longitudinal_offset, dtype=layer.operator.matrix.real.dtype)
    if offset.ndim > 0:
        raise ValueError("longitudinal_offset must be scalar.")
    offset = eqx.error_if(
        offset,
        (offset < 0.0) | (offset > jnp.real(layer.layer.thickness)),
        "longitudinal_offset must lie within the selected layer.",
    )
    left_electric = result.boundary_electric_fields[element_index]
    left_magnetic = result.boundary_magnetic_fields[element_index]
    partial = prepare_layer_boundary(
        layer.operator, offset, prepared.plan.policy.boundary
    )
    magnetic = _dense_solve(
        partial.d,
        left_magnetic - partial.c @ left_electric,
    )
    electric = partial.a @ left_electric + partial.b @ magnetic
    tangential = jnp.concatenate((electric, magnetic), axis=0)
    electric_z, magnetic_z = recover_longitudinal_fields(layer.operator, tangential)
    count = prepared.problem.harmonics.harmonic_count
    electric_harmonics = jnp.stack(
        (electric[:count], electric[count:], electric_z),
        axis=1,
    )
    magnetic_harmonics = jnp.stack(
        (magnetic[:count], magnetic[count:], magnetic_z),
        axis=1,
    )
    lattice = prepared.problem.harmonics
    if coordinates is None:
        electric_field = lattice.synthesis(electric_harmonics)
        magnetic_field = lattice.synthesis(magnetic_harmonics)
    else:
        electric_field = lattice.evaluate(electric_harmonics, coordinates)
        magnetic_field = lattice.evaluate(magnetic_harmonics, coordinates)
    return FourierModalFieldResult(
        electric_harmonics,
        magnetic_harmonics,
        electric_field,
        magnetic_field,
        offset,
        layer_id=layer.layer.layer_id,
    )


def poynting_flux(
    electric_field: ArrayLike,
    magnetic_field: ArrayLike,
    /,
) -> Array:
    electric = jnp.asarray(electric_field)
    magnetic = jnp.asarray(magnetic_field, dtype=electric.dtype)
    if electric.shape != magnetic.shape or electric.shape[-2] != 3:
        raise ValueError(
            "Fields must have equal shape with the Cartesian component axis second-last."
        )
    return 0.5 * jnp.real(
        electric[..., 0, :] * jnp.conj(magnetic[..., 1, :])
        - electric[..., 1, :] * jnp.conj(magnetic[..., 0, :])
    )


def cell_integrated_poynting_flux(
    prepared: PreparedFourierModalMaxwell,
    field: FourierModalFieldResult,
    /,
) -> Array:
    flux = poynting_flux(field.electric_field, field.magnetic_field)
    physical_axes = tuple(range(prepared.problem.harmonics.periodic_dimension))
    return jnp.mean(flux, axis=physical_axes) * prepared.problem.harmonics.cell_measure


def diffraction_order_far_field(
    prepared: PreparedFourierModalMaxwell,
    result: FourierModalSolveResult,
    /,
    *,
    side: str = "right",
) -> DiffractionOrderFarField:
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")
    modes = prepared.right_modes if side == "right" else prepared.left_modes
    amplitudes = result.right_outgoing if side == "right" else result.left_outgoing
    transverse = prepared.problem.harmonics.in_plane_wavevectors(
        prepared.problem.bloch_wavevector
    )
    kz = (
        modes.longitudinal_wavevector
        if side == "right"
        else -modes.longitudinal_wavevector
    )
    wavevectors = jnp.concatenate((transverse, kz[:, None]), axis=-1)
    magnitude = jnp.sqrt(jnp.sum(jnp.abs(wavevectors) ** 2, axis=-1))
    safe_magnitude = jnp.where(magnitude > 0.0, magnitude, 1.0)
    directions = jnp.real(wavevectors / safe_magnitude[:, None])
    polar = jnp.arccos(jnp.clip(directions[:, 2], -1.0, 1.0))
    azimuth = jnp.arctan2(directions[:, 1], directions[:, 0])
    count = prepared.problem.harmonics.harmonic_count
    rhs_count = amplitudes.shape[1]
    power = (jnp.abs(modes.flux_weights)[:, None] * jnp.abs(amplitudes) ** 2).reshape(
        (count, 2, rhs_count)
    )
    power = jnp.where(modes.propagating.reshape((count, 2, 1)), power, 0.0)
    return DiffractionOrderFarField(
        wavevectors,
        directions,
        polar,
        azimuth,
        power,
        modes.propagating.reshape((count, 2)),
        side=side,
    )


__all__ = [
    "DiffractionOrderFarField",
    "FourierModalFieldResult",
    "cell_integrated_poynting_flux",
    "diffraction_order_far_field",
    "fields_in_layer",
    "poynting_flux",
]
