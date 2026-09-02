#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from operator import index
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._strict import StrictModule
from ...interchange import (
    AdapterReport,
    AdapterStatus,
    HostInspectionConversion,
    HostInspectionField,
    HostInspectionFrame,
)
from ...solver import FixedStepResult
from ._step import IncompressibleTwoPhaseVOFMethod, TwoPhaseContinuationState
from ._vof import PreparedIncompressibleTwoPhaseVOF


class TwoPhaseDiagnosticView(StrictModule):
    alpha: Array
    density: Array
    viscosity: Array
    velocity: tuple[Array, ...]
    pressure: Array
    geometry_epoch: Array
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
    geometry_id: str = eqx.field(static=True)


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
        geometry_epoch=state.state.geometry_epoch,
        successful=view.plic.valid & view.topology.valid,
        geometry_id=view.geometry_id,
        two_phase_id=two_phase.prepared_id,
    )


def two_phase_inspection_frame(
    two_phase: PreparedIncompressibleTwoPhaseVOF,
    state: TwoPhaseContinuationState,
    /,
    *,
    time: ArrayLike,
    step: int,
    state_kind: str,
    successful: ArrayLike,
    status: int,
    result_id: str,
) -> HostInspectionConversion:
    """Copy one native VOF diagnostic view into an immutable host frame."""
    if not isinstance(two_phase, PreparedIncompressibleTwoPhaseVOF):
        raise TypeError("two_phase must be PreparedIncompressibleTwoPhaseVOF.")
    if not isinstance(state, TwoPhaseContinuationState):
        raise TypeError("state must be TwoPhaseContinuationState.")
    if state_kind not in ("candidate", "accepted"):
        raise ValueError("state_kind must be candidate or accepted.")
    identifier = str(result_id).strip()
    if not identifier:
        raise ValueError("result_id must be a non-empty string.")

    view = two_phase_diagnostic_view(two_phase, state)
    discretization = two_phase.plan.discretization
    support_id = discretization.support.support_id
    cell_layout_id = discretization.cell_layout.layout_id
    global_layout_id = f"{discretization.prepared_id}:global-diagnostics"
    geometry_epoch = int(np.asarray(view.geometry_epoch))
    expected_geometry_epoch = (
        -1 if two_phase.geometry is None else int(np.asarray(two_phase.geometry.epoch))
    )
    if geometry_epoch != expected_geometry_epoch:
        raise ValueError("VOF state geometry epoch does not match the prepared geometry.")
    provenance_id = (
        f"{identifier}:diagnostic-view:{view.two_phase_id}:"
        f"geometry={view.geometry_id!r}:epoch={geometry_epoch}"
    )
    cell_valid = view.successful
    fields = [
        HostInspectionField(
            "alpha",
            view.alpha,
            cell_valid,
            "cell",
            support_id,
            cell_layout_id,
            "cell_average",
            provenance_id=provenance_id,
        ),
        HostInspectionField(
            "density",
            view.density,
            cell_valid,
            "cell",
            support_id,
            cell_layout_id,
            "derived_cell_value",
            provenance_id=provenance_id,
        ),
        HostInspectionField(
            "viscosity",
            view.viscosity,
            cell_valid,
            "cell",
            support_id,
            cell_layout_id,
            "derived_cell_value",
            provenance_id=provenance_id,
        ),
        HostInspectionField(
            "pressure",
            view.pressure,
            cell_valid,
            "cell",
            support_id,
            cell_layout_id,
            "cell_value",
            provenance_id=provenance_id,
        ),
        HostInspectionField(
            "plic_normal",
            view.plic_normal,
            cell_valid,
            "cell",
            support_id,
            cell_layout_id,
            "plic_normal",
            component_labels=discretization.grid.axis_names,
            provenance_id=provenance_id,
        ),
        HostInspectionField(
            "plic_offset",
            view.plic_offset,
            cell_valid,
            "cell",
            support_id,
            cell_layout_id,
            "plic_plane_offset",
            provenance_id=provenance_id,
        ),
        HostInspectionField(
            "level_set",
            view.level_set,
            cell_valid,
            "cell",
            support_id,
            cell_layout_id,
            "cell_value",
            provenance_id=provenance_id,
        ),
    ]
    for axis_name, component, layout in zip(
        discretization.grid.axis_names,
        view.velocity,
        discretization.face_layouts,
        strict=True,
    ):
        fields.append(
            HostInspectionField(
                f"velocity:{axis_name}",
                component,
                cell_valid,
                "face",
                support_id,
                layout.layout_id,
                "face_value",
                provenance_id=provenance_id,
            )
        )
    for name, values, representation in (
        ("liquid_volume", view.liquid_volume, "integral"),
        ("gas_volume", view.gas_volume, "integral"),
        ("interface_measure", view.interface_measure, "integral"),
        ("topology_event_count", view.topology_event_count, "diagnostic_count"),
        ("kinetic_energy", view.kinetic_energy, "integral"),
        ("geometry_epoch", view.geometry_epoch, "geometry_epoch"),
        ("successful", view.successful, "diagnostic_flag"),
    ):
        fields.append(
            HostInspectionField(
                name,
                values,
                True,
                "global",
                support_id,
                global_layout_id,
                representation,
                provenance_id=provenance_id,
            )
        )

    frame = HostInspectionFrame(
        time,
        step,
        state_kind,
        successful,
        status,
        tuple(fields),
        two_phase.prepared_id,
        identifier,
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "phydrax-two-phase-diagnostic-view",
        "phydrax-host-inspection-frame",
        source_id=provenance_id,
        target_id=identifier,
        coordinate_mapping=("native-time", "native-step"),
        preserved_fields=tuple(field.name for field in fields),
    )
    return HostInspectionConversion(frame, report)


def two_phase_inspection_frames(
    two_phase: PreparedIncompressibleTwoPhaseVOF,
    result: FixedStepResult,
    /,
    *,
    time: ArrayLike,
    step: int,
    step_size: ArrayLike,
    result_id: str,
) -> tuple[HostInspectionConversion, HostInspectionConversion]:
    """Inspect candidate and rollback-aware accepted VOF states without merging them."""
    if not isinstance(result, FixedStepResult):
        raise TypeError("result must be a FixedStepResult.")
    if not isinstance(
        result.candidate_state, TwoPhaseContinuationState
    ) or not isinstance(result.accepted_state, TwoPhaseContinuationState):
        raise TypeError("Fixed-step result states must be TwoPhaseContinuationState.")
    source_time_array = np.asarray(time)
    step_size_array = np.asarray(step_size)
    if source_time_array.shape != () or step_size_array.shape != ():
        raise ValueError("time and step_size must be scalars.")
    source_time = float(source_time_array)
    dt = float(step_size_array)
    if not np.isfinite(source_time) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time must be finite and step_size must be finite and positive.")
    source_step = index(step)
    if source_step < 0:
        raise ValueError("step must be nonnegative.")
    successful_array = np.asarray(result.successful, dtype=bool)
    if successful_array.shape != ():
        raise ValueError("Fixed-step result successful must be scalar.")
    successful = bool(successful_array)
    candidate_time = source_time + dt
    candidate_step = source_step + 1
    accepted_time = candidate_time if successful else source_time
    accepted_step = candidate_step if successful else source_step
    status = 0 if successful else 1
    base_id = str(result_id).strip()
    if not base_id:
        raise ValueError("result_id must be a non-empty string.")
    candidate = two_phase_inspection_frame(
        two_phase,
        result.candidate_state,
        time=candidate_time,
        step=candidate_step,
        state_kind="candidate",
        successful=successful,
        status=status,
        result_id=f"{base_id}:candidate",
    )
    accepted = two_phase_inspection_frame(
        two_phase,
        result.accepted_state,
        time=accepted_time,
        step=accepted_step,
        state_kind="accepted",
        successful=successful,
        status=status,
        result_id=f"{base_id}:accepted",
    )
    return candidate, accepted


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
    "two_phase_inspection_frame",
    "two_phase_inspection_frames",
    "write_two_phase_checkpoint",
    "write_two_phase_output",
]
