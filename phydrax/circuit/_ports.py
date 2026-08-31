#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


PortId: TypeAlias = str
WaveConvention: TypeAlias = Literal["kurokawa-power"]


class ElectricalWaveReference(StrictModule):
    """Positive-real electrical reference for RMS Kurokawa power waves."""

    z0: Array
    convention: WaveConvention = eqx.field(static=True)

    def __init__(
        self, z0: ArrayLike, /, *, convention: WaveConvention = "kurokawa-power"
    ):
        if convention != "kurokawa-power":
            raise ValueError("Only the 'kurokawa-power' convention is supported.")
        impedance = jnp.asarray(z0)
        if not jnp.issubdtype(impedance.dtype, jnp.number) or jnp.issubdtype(
            impedance.dtype, jnp.bool_
        ):
            raise TypeError("z0 must have a real or complex numeric dtype.")
        impedance = impedance.astype(jnp.result_type(impedance, jnp.complex128))
        impedance = eqx.error_if(
            impedance,
            ~jnp.all(jnp.isfinite(impedance)) | jnp.any(jnp.real(impedance) <= 0.0),
            "Electrical reference impedances must be finite with Re(z0) > 0.",
        )
        self.z0 = impedance
        self.convention = convention


class ModalWaveReference(StrictModule):
    """Exact identity of one unit-flux modal power-wave coordinate."""

    reference_plane: Array
    basis_id: str = eqx.field(static=True)
    mode_id: str = eqx.field(static=True)
    polarization: str = eqx.field(static=True)
    normalization: str = eqx.field(static=True)
    orientation: str = eqx.field(static=True)

    def __init__(
        self,
        basis_id: str,
        mode_id: str,
        /,
        *,
        polarization: str,
        normalization: str = "unit-flux",
        orientation: str = "into-component",
        reference_plane: ArrayLike = 0.0,
    ):
        identities = {
            "basis_id": str(basis_id),
            "mode_id": str(mode_id),
            "polarization": str(polarization),
            "normalization": str(normalization),
            "orientation": str(orientation),
        }
        if any(not value for value in identities.values()):
            raise ValueError("Modal reference identities must be non-empty.")
        plane = jnp.asarray(reference_plane)
        if plane.ndim != 0 or not jnp.issubdtype(plane.dtype, jnp.number):
            raise ValueError("reference_plane must be one numeric scalar.")
        plane = eqx.error_if(
            plane,
            ~jnp.isfinite(plane),
            "reference_plane must be finite.",
        )
        self.reference_plane = plane
        self.basis_id = identities["basis_id"]
        self.mode_id = identities["mode_id"]
        self.polarization = identities["polarization"]
        self.normalization = identities["normalization"]
        self.orientation = identities["orientation"]


WaveReference: TypeAlias = ElectricalWaveReference | ModalWaveReference


class WaveChannelAddress(StrictModule):
    """Stable coordinate address inside one block-valued wave port."""

    port_id: PortId = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)

    def __init__(self, port_id: PortId, coordinate_id: str, /):
        port, coordinate = str(port_id), str(coordinate_id)
        if not port or not coordinate:
            raise ValueError("Wave channel address IDs must be non-empty.")
        self.port_id, self.coordinate_id = port, coordinate


class WavePort(StrictModule):
    """One block-valued bidirectional power-wave port."""

    references: tuple[WaveReference, ...]
    port_id: PortId = eqx.field(static=True)
    coordinate_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        port_id: PortId,
        references: WaveReference | Sequence[WaveReference],
        /,
        *,
        coordinate_ids: Sequence[str] | None = None,
    ):
        identifier = str(port_id)
        if not identifier:
            raise ValueError("port_id must be non-empty.")
        reference_tuple = (
            (references,)
            if isinstance(references, (ElectricalWaveReference, ModalWaveReference))
            else tuple(references)
        )
        if not reference_tuple or any(
            not isinstance(value, (ElectricalWaveReference, ModalWaveReference))
            for value in reference_tuple
        ):
            raise TypeError("references must contain typed wave references.")
        if coordinate_ids is None:
            coordinates = tuple(
                value.mode_id
                if isinstance(value, ModalWaveReference)
                else (identifier if len(reference_tuple) == 1 else str(index))
                for index, value in enumerate(reference_tuple)
            )
        else:
            coordinates = tuple(str(value) for value in coordinate_ids)
        if (
            len(coordinates) != len(reference_tuple)
            or any(not value for value in coordinates)
            or len(set(coordinates)) != len(coordinates)
        ):
            raise ValueError(
                "coordinate_ids must be unique, non-empty, and match references."
            )
        for coordinate, reference in zip(coordinates, reference_tuple, strict=True):
            if (
                isinstance(reference, ModalWaveReference)
                and coordinate != reference.mode_id
            ):
                raise ValueError(
                    "Modal wave coordinates must equal their reference mode IDs."
                )
        self.references = reference_tuple
        self.port_id = identifier
        self.coordinate_ids = coordinates

    @property
    def size(self) -> int:
        return len(self.references)

    @property
    def channels(self) -> tuple[WaveChannelAddress, ...]:
        return tuple(
            WaveChannelAddress(self.port_id, coordinate)
            for coordinate in self.coordinate_ids
        )


def references_compatible(
    first: WaveReference,
    second: WaveReference,
    /,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Array:
    """Return exact-static and tolerance-qualified dynamic link compatibility."""
    if type(first) is not type(second):
        return jnp.asarray(False)
    if isinstance(first, ElectricalWaveReference):
        return jnp.asarray(first.convention == second.convention) & jnp.allclose(
            first.z0, second.z0, rtol=rtol, atol=atol
        )
    assert isinstance(first, ModalWaveReference) and isinstance(
        second, ModalWaveReference
    )
    static_equal = (
        first.basis_id == second.basis_id
        and first.mode_id == second.mode_id
        and first.polarization == second.polarization
        and first.normalization == second.normalization
        and first.orientation == second.orientation
    )
    return jnp.asarray(static_equal) & jnp.allclose(
        first.reference_plane, second.reference_plane, rtol=rtol, atol=atol
    )


def transformed_references_compatible(
    first: WaveReference,
    second: WaveReference,
    /,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Array:
    """Check physical chart compatibility when an explicit lossless map is supplied."""
    if type(first) is not type(second):
        return jnp.asarray(False)
    if isinstance(first, ElectricalWaveReference):
        return jnp.asarray(first.convention == second.convention) & jnp.allclose(
            first.z0, second.z0, rtol=rtol, atol=atol
        )
    assert isinstance(first, ModalWaveReference) and isinstance(
        second, ModalWaveReference
    )
    static_equal = (
        first.normalization == second.normalization
        and first.orientation == second.orientation
    )
    return jnp.asarray(static_equal) & jnp.allclose(
        first.reference_plane, second.reference_plane, rtol=rtol, atol=atol
    )


def voltage_current_to_power_waves(
    voltage: ArrayLike,
    current: ArrayLike,
    reference: ElectricalWaveReference,
    /,
) -> tuple[Array, Array]:
    """Map RMS phasor voltage/current into incident/outgoing Kurokawa waves."""
    if not isinstance(reference, ElectricalWaveReference):
        raise TypeError("reference must be ElectricalWaveReference.")
    voltage_ = jnp.asarray(voltage)
    current_ = jnp.asarray(current)
    root = jnp.sqrt(jnp.real(reference.z0))
    incident = (voltage_ + reference.z0 * current_) / (2.0 * root)
    outgoing = (voltage_ - jnp.conj(reference.z0) * current_) / (2.0 * root)
    return incident, outgoing


def power_waves_to_voltage_current(
    incident: ArrayLike,
    outgoing: ArrayLike,
    reference: ElectricalWaveReference,
    /,
) -> tuple[Array, Array]:
    """Invert the Kurokawa map without assuming a real reference impedance."""
    if not isinstance(reference, ElectricalWaveReference):
        raise TypeError("reference must be ElectricalWaveReference.")
    incident_ = jnp.asarray(incident)
    outgoing_ = jnp.asarray(outgoing)
    root = jnp.sqrt(jnp.real(reference.z0))
    current = (incident_ - outgoing_) / root
    voltage = root * (incident_ + outgoing_) - 1j * jnp.imag(reference.z0) * current
    return voltage, current


__all__ = [
    "ElectricalWaveReference",
    "ModalWaveReference",
    "PortId",
    "WaveConvention",
    "WaveChannelAddress",
    "WavePort",
    "WaveReference",
    "power_waves_to_voltage_current",
    "references_compatible",
    "transformed_references_compatible",
    "voltage_current_to_power_waves",
]
