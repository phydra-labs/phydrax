#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Optical pulse envelopes lowered to solver-native Maxwell plane antennas.

The optics owner maps a `PulseEnvelopeField` (sampled directly, from
`sample_focused_gaussian_pulse_envelope`, or from an openPMD LaserEnvelope
import) onto `phydrax.solver.maxwell.SampledPlaneCurrentAntennaPlan`; the solver
never imports optics. The plane frame must be aligned with the structured grid:
its normal is a signed grid axis (the emission direction) and each tangential
basis vector a signed tangential grid axis. The tangential Jones envelope is
rotated into grid components and the sample axes are reordered to increasing
grid coordinates; the world coordinates of the plane are the grid coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..._physical import ElectromagneticScaleContract
from ...discretization import StructuredCochainBridge
from ...solver._maxwell_antenna import SampledPlaneCurrentAntennaPlan
from ...solver._maxwell_far_field import HomogeneousMaxwellExterior
from ._envelope import PulseEnvelopeField


if TYPE_CHECKING:
    from ...interchange._openpmd_laser import OpenPMDLaserEnvelopeImportResult


def _signed_axis(vector: np.ndarray, name: str, /) -> tuple[int, float]:
    axis = int(np.argmax(np.abs(vector)))
    sign = float(np.sign(vector[axis]))
    expected = np.zeros(3)
    expected[axis] = sign
    if not np.allclose(vector, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"The plane {name} must be aligned with a grid axis.")
    return axis, sign


def pulse_envelope_antenna(
    field: PulseEnvelopeField,
    bridge: StructuredCochainBridge,
    /,
    *,
    medium: HomogeneousMaxwellExterior | None = None,
    beta: float = 0.0,
    scale: ElectromagneticScaleContract | None = None,
    provenance_id: str | None = None,
) -> SampledPlaneCurrentAntennaPlan:
    """Build a one-way plane antenna launching ``field`` along its plane normal.

    ``field`` must carry a tangential polarization; the rest-frame sample times
    are its pulse-time coordinates and outside them the antenna is silent. The
    magnetic envelope follows the plane-wave relation of the antenna medium
    (paraxial launch). ``medium``, ``beta``, and ``scale`` are forwarded to the
    antenna plan.
    """
    if not isinstance(field, PulseEnvelopeField):
        raise TypeError("field must be a PulseEnvelopeField.")
    if field.polarization != "tangential":
        raise ValueError(
            "Scalar envelopes carry no electric polarization; sample a tangential "
            "envelope with a Jones vector."
        )
    rotation = np.asarray(field.plane_space.frame.rotation, dtype=np.float64)
    translation = np.asarray(field.plane_space.frame.translation, dtype=np.float64)
    normal_axis, normal_sign = _signed_axis(rotation[:, 2], "normal")
    first_axis, first_sign = _signed_axis(rotation[:, 0], "first tangent")
    second_axis, second_sign = _signed_axis(rotation[:, 1], "second tangent")
    tangential = tuple(axis for axis in range(3) if axis != normal_axis)
    local_first, local_second = (
        np.asarray(axis, dtype=np.float64) for axis in field.plane_space.coordinate_axes
    )
    # values[u, v, t, (E_u, E_v)] -> grid components (E_b, E_c) on grid axes.
    values = np.asarray(field.values, dtype=np.complex128)
    world = values @ rotation[:, :2].T
    grid_values = world[..., list(tangential)]
    if (first_axis, second_axis) == tangential:
        coordinates = [
            translation[first_axis] + first_sign * local_first,
            translation[second_axis] + second_sign * local_second,
        ]
        signs = (first_sign, second_sign)
    else:
        grid_values = np.swapaxes(grid_values, 0, 1)
        coordinates = [
            translation[second_axis] + second_sign * local_second,
            translation[first_axis] + first_sign * local_first,
        ]
        signs = (second_sign, first_sign)
    for position, sign in enumerate(signs):
        if sign < 0.0:
            coordinates[position] = coordinates[position][::-1]
            grid_values = np.flip(grid_values, axis=position)
    return SampledPlaneCurrentAntennaPlan(
        bridge,
        normal_axis,
        float(translation[normal_axis]),
        coordinates[0],
        coordinates[1],
        np.asarray(field.time_space.coordinates, dtype=np.float64),
        grid_values,
        carrier_angular_frequency=float(field.carrier_angular_frequency),
        direction="positive" if normal_sign > 0.0 else "negative",
        medium=medium,
        beta=beta,
        scale=scale,
        provenance_id=provenance_id,
    )


def openpmd_laser_envelope_antenna(
    imported: OpenPMDLaserEnvelopeImportResult,
    bridge: StructuredCochainBridge,
    /,
    *,
    medium: HomogeneousMaxwellExterior | None = None,
    beta: float = 0.0,
    scale: ElectromagneticScaleContract | None = None,
) -> SampledPlaneCurrentAntennaPlan:
    """Build an antenna from an imported openPMD LaserEnvelope.

    The antenna provenance is the import report's target identity, so the
    antenna identity binds the exact decoded LaserEnvelope record.
    """
    # Interchange imports optics lazily; resolve the import-result type the same way.
    from ...interchange._openpmd_laser import OpenPMDLaserEnvelopeImportResult

    if not isinstance(imported, OpenPMDLaserEnvelopeImportResult):
        raise TypeError("imported must be an OpenPMDLaserEnvelopeImportResult.")
    return pulse_envelope_antenna(
        imported.field,
        bridge,
        medium=medium,
        beta=beta,
        scale=scale,
        provenance_id=imported.report.target_id,
    )


__all__ = ["openpmd_laser_envelope_antenna", "pulse_envelope_antenna"]
