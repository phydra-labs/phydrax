#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...discretization.spectral import LatticeHarmonicDiscretization
from ...solver.maxwell.fourier_modal import FourierModalFieldResult
from ._fields import PlaneFieldSpace, TangentialPlaneField


class TangentialElectromagneticPlane(StrictModule):
    """Electric and magnetic tangential fields on one shared plane space."""

    electric: TangentialPlaneField
    magnetic: TangentialPlaneField
    source_kind: Literal["fourier-modal", "periodic-window"] = eqx.field(static=True)

    def __init__(
        self,
        electric: TangentialPlaneField,
        magnetic: TangentialPlaneField,
        /,
        *,
        source_kind: Literal["fourier-modal", "periodic-window"],
    ):
        if not isinstance(electric, TangentialPlaneField) or not isinstance(
            magnetic, TangentialPlaneField
        ):
            raise TypeError("electric and magnetic must be TangentialPlaneField values.")
        if electric.space.space_id != magnetic.space.space_id:
            raise ValueError("Electric and magnetic fields must share one plane space.")
        frequency = eqx.error_if(
            magnetic.angular_frequency,
            magnetic.angular_frequency != electric.angular_frequency,
            "Electric and magnetic fields must share angular frequency.",
        )
        coordinate = eqx.error_if(
            magnetic.longitudinal_coordinate,
            magnetic.longitudinal_coordinate != electric.longitudinal_coordinate,
            "Electric and magnetic fields must share longitudinal coordinate.",
        )
        magnetic = eqx.tree_at(
            lambda field: (
                field.angular_frequency,
                field.longitudinal_coordinate,
            ),
            magnetic,
            (frequency, coordinate),
        )
        if source_kind not in ("fourier-modal", "periodic-window"):
            raise ValueError("source_kind is not a supported electromagnetic adapter.")
        self.electric = electric
        self.magnetic = magnetic
        self.source_kind = source_kind


class FourierModalPlaneEvidence(StrictModule):
    """Qualification evidence for one periodic Fourier-modal field adapter.

    Status is 0 for accepted conversion, 1 for a coordinate mismatch, and 2
    when the source field carries non-finite residual evidence or a non-success
    continuous-layer status.
    """

    maximum_coordinate_residual: Array
    coordinate_tolerance: Array
    boundary_solve_residual: Array
    local_constitutive_residual: Array
    continuous_segment_defect: Array
    continuous_segment_index: Array
    continuous_status: Array
    source_qualified: Array
    accepted: Array
    status: Array
    primitive_vectors: Array
    sample_shape: tuple[int, int] = eqx.field(static=True)
    excitation_index: int = eqx.field(static=True)
    source_layer_id: str = eqx.field(static=True)
    source_topology: Literal["periodic-cell"] = eqx.field(static=True)
    target_topology: Literal["periodic-cell"] = eqx.field(static=True)


class FourierModalPlaneAdapterResult(StrictModule):
    """Periodic electromagnetic plane and the evidence qualifying its coordinates."""

    plane: TangentialElectromagneticPlane
    evidence: FourierModalPlaneEvidence


class PeriodicWindowConversionEvidence(StrictModule):
    """Evidence for explicit periodic tiling followed by finite windowing.

    Status is 0 for accepted conversion, 1 for a coordinate mismatch, and 2
    for a non-finite or out-of-range window.
    """

    maximum_coordinate_residual: Array
    coordinate_tolerance: Array
    window_minimum: Array
    window_maximum: Array
    accepted: Array
    status: Array
    tile_counts: tuple[int, int] = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    target_space_id: str = eqx.field(static=True)
    source_topology: Literal["periodic-cell"] = eqx.field(static=True)
    target_topology: Literal["finite-window"] = eqx.field(static=True)


class PeriodicWindowAdapterResult(StrictModule):
    """Finite-support electromagnetic plane and its conversion evidence."""

    plane: TangentialElectromagneticPlane
    evidence: PeriodicWindowConversionEvidence


def _positive_scalar(value: ArrayLike, name: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.ndim != 0 or not jnp.issubdtype(scalar.dtype, jnp.floating):
        raise TypeError(f"{name} must be a real scalar.")
    return eqx.error_if(
        scalar,
        (~jnp.isfinite(scalar)) | (scalar < 0),
        f"{name} must be finite and nonnegative.",
    )


def _select_vector_field(
    values: Array,
    sample_shape: tuple[int, int],
    excitation_index: int,
    name: str,
    /,
) -> Array:
    if values.shape == sample_shape + (3,):
        if excitation_index != 0:
            raise IndexError(f"{name} has one excitation; excitation_index must be zero.")
        return values
    if values.ndim != 4 or values.shape[:3] != sample_shape + (3,):
        raise ValueError(
            f"{name} must have shape {sample_shape + (3,)} or "
            f"{sample_shape + (3, 'R')}; got {values.shape}."
        )
    if excitation_index < 0 or excitation_index >= values.shape[3]:
        raise IndexError(f"excitation_index is out of range for {name}.")
    return values[..., excitation_index]


def fourier_modal_field_to_tangential_plane(
    field: FourierModalFieldResult,
    lattice: LatticeHarmonicDiscretization,
    space: PlaneFieldSpace,
    angular_frequency: ArrayLike,
    longitudinal_coordinate: ArrayLike,
    /,
    *,
    excitation_index: int = 0,
    coordinate_tolerance: ArrayLike = 1.0e-7,
) -> FourierModalPlaneAdapterResult:
    """Qualify and adapt one Fourier-modal excitation to a periodic EM plane.

    Fourier-modal vector components are interpreted in the destination frame's
    local basis. The adapter therefore requires its sampled cell coordinates to
    equal the plane space's sole transverse coordinates; it never regrids or
    changes topology.
    """
    if not isinstance(field, FourierModalFieldResult):
        raise TypeError("field must be a FourierModalFieldResult.")
    if not isinstance(lattice, LatticeHarmonicDiscretization):
        raise TypeError("lattice must be a LatticeHarmonicDiscretization.")
    if not isinstance(space, PlaneFieldSpace):
        raise TypeError("space must be a PlaneFieldSpace.")
    if lattice.periodic_dimension != 2:
        raise ValueError("The plane adapter requires a two-dimensional lattice.")
    if space.topology != "periodic-cell":
        raise ValueError("Fourier-modal fields can only enter a periodic-cell space.")
    sample_shape = lattice.sample_shape
    if space.shape != sample_shape:
        raise ValueError(
            f"Plane shape {space.shape} must equal lattice sample shape {sample_shape}."
        )
    index = int(excitation_index)
    electric_vector = _select_vector_field(
        field.electric_field, sample_shape, index, "electric_field"
    )
    magnetic_vector = _select_vector_field(
        field.magnetic_field, sample_shape, index, "magnetic_field"
    )
    tolerance = _positive_scalar(coordinate_tolerance, "coordinate_tolerance")
    residual = jnp.max(
        jnp.abs(space.transverse_coordinates - lattice.physical_coordinates)
    )
    coordinates_match = jnp.isfinite(residual) & (residual <= tolerance)
    source_qualified = (
        jnp.all(jnp.isfinite(field.boundary_solve_residual))
        & jnp.all(jnp.isfinite(field.local_constitutive_residual))
        & jnp.all(jnp.isfinite(field.continuous_segment_defect))
        & jnp.all((field.continuous_status == -1) | (field.continuous_status == 0))
    )
    accepted = coordinates_match & source_qualified
    status = jnp.where(~source_qualified, 2, jnp.where(coordinates_match, 0, 1)).astype(
        jnp.int32
    )
    complex_dtype = jnp.result_type(
        electric_vector.dtype, magnetic_vector.dtype, jnp.complex64
    )
    rejected = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=complex_dtype)
    electric_values = jnp.where(accepted, electric_vector[..., :2], rejected)
    magnetic_values = jnp.where(accepted, magnetic_vector[..., :2], rejected)
    electric = TangentialPlaneField(
        space,
        electric_values,
        angular_frequency,
        longitudinal_coordinate,
    )
    magnetic = TangentialPlaneField(
        space,
        magnetic_values,
        angular_frequency,
        longitudinal_coordinate,
    )
    plane = TangentialElectromagneticPlane(
        electric, magnetic, source_kind="fourier-modal"
    )
    evidence = FourierModalPlaneEvidence(
        maximum_coordinate_residual=residual,
        coordinate_tolerance=tolerance,
        boundary_solve_residual=field.boundary_solve_residual,
        local_constitutive_residual=field.local_constitutive_residual,
        continuous_segment_defect=field.continuous_segment_defect,
        continuous_segment_index=field.continuous_segment_index,
        continuous_status=field.continuous_status,
        source_qualified=source_qualified,
        accepted=accepted,
        status=status,
        primitive_vectors=lattice.primitive_vectors,
        sample_shape=sample_shape,
        excitation_index=index,
        source_layer_id=field.layer_id,
        source_topology="periodic-cell",
        target_topology="periodic-cell",
    )
    return FourierModalPlaneAdapterResult(plane=plane, evidence=evidence)


def tile_periodic_plane_to_finite_window(
    source: TangentialElectromagneticPlane,
    finite_space: PlaneFieldSpace,
    primitive_vectors: ArrayLike,
    window: ArrayLike,
    /,
    *,
    tile_counts: tuple[int, int],
    coordinate_tolerance: ArrayLike = 1.0e-7,
) -> PeriodicWindowAdapterResult:
    """Explicitly tile a periodic EM cell, then apply a finite-support window."""
    if not isinstance(source, TangentialElectromagneticPlane):
        raise TypeError("source must be a TangentialElectromagneticPlane.")
    if not isinstance(finite_space, PlaneFieldSpace):
        raise TypeError("finite_space must be a PlaneFieldSpace.")
    if source.electric.space.topology != "periodic-cell":
        raise ValueError("source must live on a periodic-cell space.")
    if finite_space.topology != "finite-window":
        raise ValueError("finite_space must have finite-window topology.")
    counts = tuple(int(value) for value in tile_counts)
    if len(counts) != 2 or any(value < 1 for value in counts):
        raise ValueError("tile_counts must contain two positive integers.")
    source_shape = source.electric.space.shape
    target_shape = (source_shape[0] * counts[0], source_shape[1] * counts[1])
    if finite_space.shape != target_shape:
        raise ValueError(
            f"finite_space shape must be {target_shape} for tile_counts {counts}."
        )
    vectors = jnp.asarray(primitive_vectors)
    if vectors.shape != (2, 2) or not jnp.issubdtype(vectors.dtype, jnp.floating):
        raise TypeError("primitive_vectors must be a real array with shape (2, 2).")
    window_values = jnp.asarray(window)
    if window_values.shape != target_shape or not jnp.issubdtype(
        window_values.dtype, jnp.floating
    ):
        raise TypeError(f"window must be a real array with shape {target_shape}.")
    window_valid = (
        jnp.all(jnp.isfinite(window_values))
        & jnp.all(window_values >= 0)
        & jnp.all(window_values <= 1)
    )
    source_coordinates = source.electric.space.transverse_coordinates
    tile_0 = jnp.arange(counts[0], dtype=vectors.dtype)[:, None, None, None, None]
    tile_1 = jnp.arange(counts[1], dtype=vectors.dtype)[None, None, :, None, None]
    expected_coordinates = (
        source_coordinates[None, :, None, :, :]
        + tile_0 * vectors[0]
        + tile_1 * vectors[1]
    ).reshape(target_shape + (2,))
    expected_local_points = jnp.concatenate(
        (
            expected_coordinates,
            jnp.zeros(target_shape + (1,), dtype=expected_coordinates.dtype),
        ),
        axis=-1,
    )
    expected_world_points = source.electric.space.frame.apply(expected_local_points)
    tolerance = _positive_scalar(coordinate_tolerance, "coordinate_tolerance")
    residual = jnp.max(jnp.abs(finite_space.world_points - expected_world_points))
    coordinates_match = jnp.isfinite(residual) & (residual <= tolerance)
    accepted = coordinates_match & window_valid
    status = jnp.where(~window_valid, 2, jnp.where(coordinates_match, 0, 1)).astype(
        jnp.int32
    )
    tiled_electric = jnp.tile(source.electric.values, counts + (1,))
    tiled_magnetic = jnp.tile(source.magnetic.values, counts + (1,))
    rejected = jnp.asarray(jnp.nan + 1j * jnp.nan, tiled_electric.dtype)
    electric_values = jnp.where(
        accepted, tiled_electric * window_values[..., None], rejected
    )
    magnetic_values = jnp.where(
        accepted, tiled_magnetic * window_values[..., None], rejected
    )
    electric = TangentialPlaneField(
        finite_space,
        electric_values,
        source.electric.angular_frequency,
        source.electric.longitudinal_coordinate,
    )
    magnetic = TangentialPlaneField(
        finite_space,
        magnetic_values,
        source.magnetic.angular_frequency,
        source.magnetic.longitudinal_coordinate,
    )
    plane = TangentialElectromagneticPlane(
        electric, magnetic, source_kind="periodic-window"
    )
    evidence = PeriodicWindowConversionEvidence(
        maximum_coordinate_residual=residual,
        coordinate_tolerance=tolerance,
        window_minimum=jnp.min(window_values),
        window_maximum=jnp.max(window_values),
        accepted=accepted,
        status=status,
        tile_counts=counts,
        source_space_id=source.electric.space.space_id,
        target_space_id=finite_space.space_id,
        source_topology="periodic-cell",
        target_topology="finite-window",
    )
    return PeriodicWindowAdapterResult(plane=plane, evidence=evidence)


__all__ = [
    "FourierModalPlaneAdapterResult",
    "FourierModalPlaneEvidence",
    "PeriodicWindowAdapterResult",
    "PeriodicWindowConversionEvidence",
    "TangentialElectromagneticPlane",
    "fourier_modal_field_to_tangential_plane",
    "tile_periodic_plane_to_finite_window",
]
