#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._array_archive import write_array_archive
from ..._strict import StrictModule
from ._free_surface_ale import FreeSurfaceALEState
from ._free_surface_step import (
    FreeSurfaceALEContinuationState,
    FreeSurfaceALELedger,
    PreparedOnePhaseFreeSurfaceALE,
)


class FreeSurfaceALEDiagnosticView(StrictModule):
    eta: Array
    velocity: tuple[Array, ...]
    scalars: dict[str, Array]
    mapped_vertices: Array
    cell_volumes: Array
    pressure_head: Array
    kinetic_energy: Array
    gravitational_energy: Array
    volume: Array
    ledger: FreeSurfaceALELedger | None
    successful: Array
    hydrodynamics_id: str = eqx.field(static=True)


def free_surface_diagnostic_view(
    hydrodynamics: PreparedOnePhaseFreeSurfaceALE,
    state: FreeSurfaceALEContinuationState | FreeSurfaceALEState,
    /,
) -> FreeSurfaceALEDiagnosticView:
    continuation = state if isinstance(state, FreeSurfaceALEContinuationState) else None
    physical = continuation.state if continuation is not None else state
    eta_rate = (
        continuation.eta_rate
        if continuation is not None
        else jnp.zeros_like(physical.eta)
    )
    pressure = (
        continuation.pressure_head
        if continuation is not None
        else jnp.zeros(hydrodynamics.reference.cell_shape)
    )
    view = hydrodynamics.view(physical, eta_rate)
    gravitational = (
        0.5
        * hydrodynamics.plan.density
        * hydrodynamics.plan.gravity
        * jnp.sum(hydrodynamics.surface.horizontal_area * physical.eta**2)
    )
    evidence = hydrodynamics.surface.geometry_evidence(physical.eta, eta_rate)
    return FreeSurfaceALEDiagnosticView(
        eta=physical.eta,
        velocity=view.velocity,
        scalars=view.scalars,
        mapped_vertices=view.geometry.mapped_vertices,
        cell_volumes=view.geometry.cell_volumes,
        pressure_head=pressure,
        kinetic_energy=view.kinetic_energy,
        gravitational_energy=gravitational,
        volume=view.volume,
        ledger=None if continuation is None else continuation.ledger,
        successful=evidence.valid,
        hydrodynamics_id=hydrodynamics.prepared_id,
    )


def write_free_surface_output(
    path: str | Path,
    hydrodynamics: PreparedOnePhaseFreeSurfaceALE,
    state: FreeSurfaceALEContinuationState | FreeSurfaceALEState,
    /,
) -> Path:
    view = free_surface_diagnostic_view(hydrodynamics, state)
    arrays: dict[str, object] = {
        "eta": view.eta,
        "mapped_vertices": view.mapped_vertices,
        "cell_volumes": view.cell_volumes,
        "pressure_head": view.pressure_head,
        "kinetic_energy": view.kinetic_energy,
        "gravitational_energy": view.gravitational_energy,
        "volume": view.volume,
        "successful": view.successful,
    }
    for axis, velocity in enumerate(view.velocity):
        arrays[f"velocity/{axis}"] = velocity
    for name, scalar in view.scalars.items():
        arrays[f"scalar/{name}"] = scalar
    return write_array_archive(
        path,
        manifest={
            "kind": "one-phase-free-surface-ale-output",
            "hydrodynamics_id": hydrodynamics.prepared_id,
            "scalar_names": sorted(view.scalars),
        },
        arrays=arrays,
    )


__all__ = [
    "FreeSurfaceALEDiagnosticView",
    "free_surface_diagnostic_view",
    "write_free_surface_output",
]
