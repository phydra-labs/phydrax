#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from ...units import convert_value, derived_unit, KILOMETER, SECOND, UnitDefinition
from ._bodies import CelestialBodyCatalog
from ._context import AstrodynamicsContext
from ._data import AstrodynamicsDataProvenance
from ._ephemeris import TabulatedEphemeris
from ._state import CartesianOrbitState, CartesianOrbitTrajectory
from ._status import AstrodynamicsStatus


_KILOMETER_PER_SECOND = derived_unit("km/s", ((KILOMETER, 1), (SECOND, -1)))


def cartesian_state_from_scaled_arrays(
    position: ArrayLike,
    velocity: ArrayLike,
    context: AstrodynamicsContext,
    /,
    *,
    position_unit: UnitDefinition,
    velocity_unit: UnitDefinition,
) -> CartesianOrbitState:
    """Convert explicitly typed provider arrays into one unchanged context."""

    if not isinstance(context, AstrodynamicsContext):
        raise TypeError("context must be an AstrodynamicsContext.")
    if context.scale.length_coordinate_kind != "physical":
        raise ValueError("Coordinate adapters require physical length coordinates.")
    if not isinstance(position_unit, UnitDefinition) or not isinstance(
        velocity_unit, UnitDefinition
    ):
        raise TypeError("Provider position and velocity units must be UnitDefinition.")
    return CartesianOrbitState(
        convert_value(
            position,
            source=position_unit,
            target=context.scale.length_unit,
        ),
        convert_value(
            velocity,
            source=velocity_unit,
            target=context.scale.velocity_unit,
        ),
        context,
    )


def cartesian_state_from_coordinate_provider(
    coordinate: Any,
    extractor: Callable[
        [Any],
        tuple[ArrayLike, ArrayLike, UnitDefinition, UnitDefinition],
    ],
    context: AstrodynamicsContext,
    /,
) -> CartesianOrbitState:
    """Adapt a coordinate object through an explicit host extractor."""

    if not callable(extractor):
        raise TypeError("extractor must be callable.")
    position, velocity, position_unit, velocity_unit = extractor(coordinate)
    return cartesian_state_from_scaled_arrays(
        position,
        velocity,
        context,
        position_unit=position_unit,
        velocity_unit=velocity_unit,
    )


def tabulated_ephemeris_from_spice(
    spice: Any,
    targets: Sequence[str],
    et_seconds: ArrayLike,
    catalog: CelestialBodyCatalog,
    provenance: AstrodynamicsDataProvenance,
    /,
    *,
    observer: str,
    frame: str,
    aberration: str = "NONE",
) -> TabulatedEphemeris:
    """Sample a supplied SpiceyPy-compatible module outside traced execution."""

    if not isinstance(catalog, CelestialBodyCatalog):
        raise TypeError("catalog must be a CelestialBodyCatalog.")
    if not isinstance(provenance, AstrodynamicsDataProvenance):
        raise TypeError("provenance must be AstrodynamicsDataProvenance.")
    if catalog.context.scale.length_coordinate_kind != "physical":
        raise ValueError("SPICE adapters require physical length coordinates.")
    if (
        catalog.context.frame.origin_id != observer
        or catalog.context.frame.orientation_id != frame
    ):
        raise ValueError("SPICE observer and frame must match the catalog context.")
    if catalog.context.epoch.time_scale != "TDB" or not catalog.context.epoch.continuous:
        raise ValueError("SPICE ET conversion requires a continuous TDB reference epoch.")
    target_names = tuple(str(value) for value in targets)
    if len(target_names) != catalog.capacity:
        raise ValueError("SPICE targets must match catalog capacity.")
    source_nodes = np.asarray(et_seconds, dtype=float)
    if (
        source_nodes.ndim != 1
        or source_nodes.size < 2
        or np.any(~np.isfinite(source_nodes))
        or np.any(np.diff(source_nodes) <= 0.0)
    ):
        raise ValueError("SPICE ET nodes must be a strictly increasing vector.")
    states = np.empty((source_nodes.size, catalog.capacity, 6), dtype=float)
    for time_index, et in enumerate(source_nodes):
        for body_index, target in enumerate(target_names):
            state_km, _ = spice.spkezr(target, float(et), frame, aberration, observer)
            states[time_index, body_index] = np.asarray(state_km, dtype=float)
    velocity_unit = _KILOMETER_PER_SECOND
    epoch = catalog.context.epoch.instant.julian_date
    epoch_et_seconds = ((epoch.high - 2451545.0) + epoch.low) * 86400.0
    nodes = convert_value(
        source_nodes - epoch_et_seconds,
        source=SECOND,
        target=catalog.context.scale.time_unit,
    )
    positions = convert_value(
        states[..., :3],
        source=KILOMETER,
        target=catalog.context.scale.length_unit,
    )
    velocities = convert_value(
        states[..., 3:],
        source=velocity_unit,
        target=catalog.context.scale.velocity_unit,
    )
    converted_states = jnp.concatenate((positions, velocities), axis=-1)
    return TabulatedEphemeris(
        nodes, converted_states, catalog, provenance, bounds_policy="error"
    )


def trajectory_from_sgp4(
    satrec: Any,
    julian_day: ArrayLike,
    julian_fraction: ArrayLike,
    context: AstrodynamicsContext,
    /,
) -> CartesianOrbitTrajectory:
    """Adapt a python-sgp4/Astroz-compatible scalar Satrec API to TEME states."""

    if not isinstance(context, AstrodynamicsContext):
        raise TypeError("context must be an AstrodynamicsContext.")
    if context.scale.length_coordinate_kind != "physical":
        raise ValueError("SGP4 adapters require physical length coordinates.")
    if (
        context.frame.origin_id != "earth"
        or context.frame.orientation_id != "TEME"
        or not context.frame.pseudo_inertial
        or context.epoch.time_scale != "UTC"
        or context.epoch.continuous
    ):
        raise ValueError("SGP4 context must retain its Earth-TEME UTC epoch semantics.")
    jd, fraction = np.broadcast_arrays(
        np.asarray(julian_day, dtype=float), np.asarray(julian_fraction, dtype=float)
    )
    if jd.ndim != 1 or jd.size < 1:
        raise ValueError("SGP4 Julian dates must form a nonempty rank-one schedule.")
    states = np.zeros((jd.size, 6), dtype=float)
    status = np.zeros((jd.size,), dtype=np.int32)
    valid = np.ones((jd.size,), dtype=bool)
    for index, (day, part) in enumerate(zip(jd, fraction, strict=True)):
        error, position_km, velocity_km_s = satrec.sgp4(float(day), float(part))
        status[index] = int(error)
        valid[index] = int(error) == 0
        states[index, :3] = np.asarray(position_km, dtype=float)
        states[index, 3:] = np.asarray(velocity_km_s, dtype=float)
    velocity_unit = _KILOMETER_PER_SECOND
    positions = convert_value(
        states[:, :3],
        source=KILOMETER,
        target=context.scale.length_unit,
    )
    velocities = convert_value(
        states[:, 3:],
        source=velocity_unit,
        target=context.scale.velocity_unit,
    )
    states = jnp.concatenate((positions, velocities), axis=-1)
    epoch = context.epoch.instant.julian_date
    relative_seconds = ((jd - epoch.high) + (fraction - epoch.low)) * 86400.0
    relative_times = convert_value(
        relative_seconds,
        source=SECOND,
        target=context.scale.time_unit,
    )
    native_status = np.where(
        valid,
        int(AstrodynamicsStatus.SUCCESS),
        int(AstrodynamicsStatus.NO_SOLUTION),
    ).astype(np.int32)
    return CartesianOrbitTrajectory(
        relative_times,
        states,
        jnp.asarray(valid),
        jnp.asarray(native_status),
        context,
        trajectory_id="external:sgp4-compatible",
    )


__all__ = [
    "cartesian_state_from_coordinate_provider",
    "cartesian_state_from_scaled_arrays",
    "tabulated_ephemeris_from_spice",
    "trajectory_from_sgp4",
]
