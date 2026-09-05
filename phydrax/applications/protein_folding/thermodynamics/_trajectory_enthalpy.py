# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....atomistic import AtomisticTrajectory
from ....series import SampledSeries, SeriesSupport
from ....units import conversion_factor, derived_unit


def native_enthalpy_series(
    trajectory: AtomisticTrajectory,
    *,
    pressure,
    pressure_unit,
    volumes,
    volume_unit,
    source_id: str,
) -> SampledSeries:
    """Form total H=U+K+pV from a native retained trajectory and explicit volume.

    Pressure and volume are scalar or sample-aligned in exact declared units;
    they are never inferred from coordinates or a protein bounding box. For a
    finite isolated model the caller may explicitly choose zero pressure. The
    output is single-system energy in ``trajectory.units.scale.energy_unit``;
    any conversion to molar energy is a separate semantic transformation.
    Existing sample/reset masks and native physical times remain authoritative.
    """
    if not isinstance(trajectory, AtomisticTrajectory):
        raise TypeError("Supply a native AtomisticTrajectory, not an optimizer trace.")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("An executed trajectory source artifact is required.")
    energy = trajectory.units.scale.energy_unit
    reference_volume = derived_unit(
        "native-volume", ((trajectory.units.scale.length_unit, 3),)
    )
    pressure_factor = float(
        conversion_factor(pressure_unit, trajectory.units.pressure_unit)
    )
    volume_factor = float(conversion_factor(volume_unit, reference_volume))
    pressure_values = np.asarray(pressure, dtype=float)
    volume_values = np.asarray(volumes, dtype=float)
    shape = tuple(trajectory.times.shape)
    if pressure_values.shape not in ((), shape) or volume_values.shape not in ((), shape):
        raise ValueError(
            "Pressure and volume must be scalar or aligned to retained trajectory samples."
        )
    if (
        not np.all(np.isfinite(pressure_values))
        or not np.all(np.isfinite(volume_values))
        or np.any(volume_values <= 0)
    ):
        raise ValueError(
            "Pressure must be finite and material box volumes finite and positive."
        )
    valid = trajectory.valid & trajectory.sample_mask
    values = (
        trajectory.energies[:, 0]
        + trajectory.energies[:, 1]
        + jnp.asarray(pressure_values * pressure_factor)
        * jnp.asarray(volume_values * volume_factor)
    )
    support = SeriesSupport(
        trajectory.times,
        node_valid=valid,
        edge_valid=valid[:-1] & valid[1:],
        coordinate_name="physical-time",
        coordinate_id=trajectory.units.time_unit.unit_id,
    )
    identity = canonical_fingerprint(
        {
            "kind": "native-total-enthalpy-series",
            "source": source_id,
            "trajectory": trajectory.trajectory_id,
            "pressure": array_tree_fingerprint(pressure_values),
            "volume": array_tree_fingerprint(volume_values),
            "pressure_unit": pressure_unit.unit_id,
            "volume_unit": volume_unit.unit_id,
            "energy_unit": energy.unit_id,
        }
    )
    return SampledSeries(support, jnp.where(valid, values, 0.0), series_id=identity)


__all__ = ["native_enthalpy_series"]
