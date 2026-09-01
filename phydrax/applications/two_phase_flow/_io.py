#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._strict import StrictModule
from ._step import IncompressibleTwoPhaseVOFMethod, TwoPhaseContinuationState
from ._vof import PreparedIncompressibleTwoPhaseVOF


class TwoPhaseDiagnosticView(StrictModule):
    alpha: Array
    density: Array
    viscosity: Array
    velocity: tuple[Array, ...]
    pressure: Array
    plic_normal: Array
    plic_offset: Array
    level_set: Array
    liquid_volume: Array
    gas_volume: Array
    interface_measure: Array
    topology_event_count: Array
    kinetic_energy: Array
    successful: Array
    two_phase_id: str = eqx.field(static=True)


def two_phase_diagnostic_view(
    two_phase: PreparedIncompressibleTwoPhaseVOF,
    state: TwoPhaseContinuationState,
    /,
) -> TwoPhaseDiagnosticView:
    view = two_phase.view(state.state, state.pressure)
    kinetic = 0.5 * sum(
        jnp.sum(rho * measure * component**2)
        for rho, measure, component in zip(
            two_phase.face_density(view.density),
            two_phase.operators.face_dual_measures,
            view.velocity,
            strict=True,
        )
    )
    return TwoPhaseDiagnosticView(
        alpha=view.alpha,
        density=view.density,
        viscosity=view.viscosity,
        velocity=view.velocity,
        pressure=view.pressure,
        plic_normal=view.plic.normal,
        plic_offset=view.plic.plane_offset,
        level_set=state.state.level_set,
        liquid_volume=view.topology.liquid_volume,
        gas_volume=view.topology.gas_volume,
        interface_measure=view.topology.interface_measure,
        topology_event_count=view.topology.component_proxy,
        kinetic_energy=kinetic,
        successful=view.plic.valid & view.topology.valid,
        two_phase_id=two_phase.prepared_id,
    )


def write_two_phase_checkpoint(
    path: str | Path,
    two_phase: PreparedIncompressibleTwoPhaseVOF,
    method: IncompressibleTwoPhaseVOFMethod,
    time: ArrayLike,
    accepted_step: ArrayLike,
    state: TwoPhaseContinuationState,
    /,
) -> Path:
    arrays: dict[str, object] = {
        "time": jnp.asarray(time),
        "accepted_step": jnp.asarray(accepted_step),
    }
    specification = pack_array_tree("state", state, arrays)
    return write_array_archive(
        path,
        manifest={
            "kind": "incompressible-two-phase-vof-checkpoint",
            "schema": "alpha-momentum-clsvof-topology-ledger",
            "two_phase_id": two_phase.prepared_id,
            "method_id": method.method_id,
            "state": specification,
        },
        arrays=arrays,
    )


def read_two_phase_checkpoint(
    path: str | Path,
    two_phase: PreparedIncompressibleTwoPhaseVOF,
    method: IncompressibleTwoPhaseVOFMethod,
    template: TwoPhaseContinuationState,
    /,
) -> tuple[Array, Array, TwoPhaseContinuationState]:
    manifest, arrays = read_array_archive(path)
    if manifest.get("kind") != "incompressible-two-phase-vof-checkpoint":
        raise ValueError("Archive is not a two-phase VOF checkpoint.")
    if manifest.get("two_phase_id") != two_phase.prepared_id:
        raise ValueError("Two-phase checkpoint model identity mismatch.")
    if manifest.get("method_id") != method.method_id:
        raise ValueError("Two-phase checkpoint method identity mismatch.")
    restored = unpack_array_tree(manifest["state"], arrays, template)
    return jnp.asarray(arrays["time"]), jnp.asarray(arrays["accepted_step"]), restored


def write_two_phase_output(
    path: str | Path,
    two_phase: PreparedIncompressibleTwoPhaseVOF,
    state: TwoPhaseContinuationState,
    /,
) -> Path:
    view = two_phase_diagnostic_view(two_phase, state)
    arrays: dict[str, object] = {
        "alpha": view.alpha,
        "density": view.density,
        "viscosity": view.viscosity,
        "pressure": view.pressure,
        "plic_normal": view.plic_normal,
        "plic_offset": view.plic_offset,
        "level_set": view.level_set,
        "liquid_volume": view.liquid_volume,
        "gas_volume": view.gas_volume,
        "interface_measure": view.interface_measure,
        "topology_event_count": view.topology_event_count,
        "kinetic_energy": view.kinetic_energy,
        "successful": view.successful,
    }
    for axis, component in enumerate(view.velocity):
        arrays[f"velocity/{axis}"] = component
    return write_array_archive(
        path,
        manifest={
            "kind": "incompressible-two-phase-vof-output",
            "two_phase_id": two_phase.prepared_id,
        },
        arrays=arrays,
    )


__all__ = [
    "TwoPhaseDiagnosticView",
    "read_two_phase_checkpoint",
    "two_phase_diagnostic_view",
    "write_two_phase_checkpoint",
    "write_two_phase_output",
]
