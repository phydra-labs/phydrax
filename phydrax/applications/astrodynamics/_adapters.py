#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from ._bodies import CelestialBodyCatalog
from ._context import AstrodynamicsContext
from ._data import AstrodynamicsDataProvenance
from ._ephemeris import TabulatedEphemeris
from ._state import CartesianOrbitState, CartesianOrbitTrajectory
from ._status import AstrodynamicsStatus


def cartesian_state_from_scaled_arrays(
    position: ArrayLike,
    velocity: ArrayLike,
    context: AstrodynamicsContext,
    /,
    *,
    position_to_reference: float,
    velocity_to_reference: float,
) -> CartesianOrbitState:
    """Convert provider arrays into one canonical context without quantity leaves."""

    if not isinstance(context, AstrodynamicsContext):
        raise TypeError("context must be an AstrodynamicsContext.")
    position_factor = float(position_to_reference) / context.scale.length_to_reference
    velocity_factor = (
        float(velocity_to_reference)
        * context.scale.time_to_reference
        / context.scale.length_to_reference
    )
    return CartesianOrbitState(
        jnp.asarray(position) * position_factor,
        jnp.asarray(velocity) * velocity_factor,
        context,
    )


def cartesian_state_from_coordinate_provider(
    coordinate: Any,
    extractor: Callable[[Any], tuple[ArrayLike, ArrayLike, float, float]],
    context: AstrodynamicsContext,
    /,
) -> CartesianOrbitState:
    """Adapt a Coordinax/Astropy-like value through an explicit host extractor."""

    if not callable(extractor):
        raise TypeError("extractor must be callable.")
    position, velocity, position_to_reference, velocity_to_reference = extractor(
        coordinate
    )
    return cartesian_state_from_scaled_arrays(
        position,
        velocity,
        context,
        position_to_reference=position_to_reference,
        velocity_to_reference=velocity_to_reference,
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

    target_names = tuple(str(value) for value in targets)
    if len(target_names) != catalog.capacity:
        raise ValueError("SPICE targets must match catalog capacity.")
    nodes = np.asarray(et_seconds, dtype=float)
    if nodes.ndim != 1 or nodes.size < 2 or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("SPICE ET nodes must be a strictly increasing vector.")
    states = np.empty((nodes.size, catalog.capacity, 6), dtype=float)
    for time_index, et in enumerate(nodes):
        for body_index, target in enumerate(target_names):
            state_km, _ = spice.spkezr(target, float(et), frame, aberration, observer)
            states[time_index, body_index] = np.asarray(state_km, dtype=float)
    length_factor = 1000.0 / catalog.context.scale.length_to_reference
    velocity_factor = (
        1000.0
        * catalog.context.scale.time_to_reference
        / catalog.context.scale.length_to_reference
    )
    states[..., :3] *= length_factor
    states[..., 3:] *= velocity_factor
    return TabulatedEphemeris(nodes, states, catalog, provenance, bounds_policy="error")


def trajectory_from_sgp4(
    satrec: Any,
    julian_day: ArrayLike,
    julian_fraction: ArrayLike,
    context: AstrodynamicsContext,
    /,
) -> CartesianOrbitTrajectory:
    """Adapt a python-sgp4/Astroz-compatible scalar Satrec API to TEME states."""

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
    states[:, :3] *= 1000.0 / context.scale.length_to_reference
    states[:, 3:] *= (
        1000.0 * context.scale.time_to_reference / context.scale.length_to_reference
    )
    relative_seconds = ((jd + fraction) - (jd[0] + fraction[0])) * 86400.0
    native_status = np.where(
        valid,
        int(AstrodynamicsStatus.SUCCESS),
        int(AstrodynamicsStatus.NO_SOLUTION),
    ).astype(np.int32)
    return CartesianOrbitTrajectory(
        jnp.asarray(relative_seconds),
        jnp.asarray(states),
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
