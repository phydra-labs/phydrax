#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

from ..discretization.flip import FLIPRuntimeState, FLIPStepResult
from ..interchange import (
    AdapterReport,
    AdapterStatus,
    HostInspectionConversion,
    HostInspectionField,
    HostInspectionFrame,
)
from ._flip import CompiledFLIPProblem


def _flip_inspection_frame(
    compiled: CompiledFLIPProblem,
    state: FLIPRuntimeState,
    result: FLIPStepResult,
    /,
    *,
    state_kind: str,
    attempt_id: str,
    result_id: str,
) -> HostInspectionConversion:
    particles = compiled.transfer.particles
    discretization = compiled.projection.operators.discretization
    particle_support_id = particles.support.support_id
    grid_support_id = discretization.support.support_id
    particle_labels = discretization.grid.axis_names
    state_provenance = f"{result_id}:state:{compiled.compilation_id}"
    attempt_provenance = f"{attempt_id}:step-evidence:{compiled.compilation_id}"
    active = particles.active_mask
    position_space, velocity_space = particles.field_spaces

    fields = [
        HostInspectionField(
            "position",
            state.particles.position,
            active,
            "particle",
            particle_support_id,
            position_space.layout.layout_id,
            position_space.representation,
            component_labels=particle_labels,
            provenance_id=state_provenance,
        ),
        HostInspectionField(
            "particle_velocity",
            state.particles.velocity,
            active,
            "particle",
            particle_support_id,
            velocity_space.layout.layout_id,
            velocity_space.representation,
            component_labels=particle_labels,
            provenance_id=state_provenance,
        ),
        HostInspectionField(
            "pressure",
            state.pressure,
            True,
            "cell",
            grid_support_id,
            discretization.cell_layout.layout_id,
            "cell_value",
            provenance_id=state_provenance,
        ),
        HostInspectionField(
            "attempt_liquid_fraction",
            result.liquid_fraction,
            True,
            "cell",
            grid_support_id,
            discretization.cell_layout.layout_id,
            "derived_cell_value",
            provenance_id=attempt_provenance,
        ),
    ]
    for axis_name, pre, post, layout in zip(
        discretization.grid.axis_names,
        result.pre_grid_velocity,
        result.post_grid_velocity,
        discretization.face_layouts,
        strict=True,
    ):
        fields.extend(
            (
                HostInspectionField(
                    f"attempt_pre_grid_velocity:{axis_name}",
                    pre,
                    True,
                    "face",
                    grid_support_id,
                    layout.layout_id,
                    "face_value",
                    provenance_id=attempt_provenance,
                ),
                HostInspectionField(
                    f"attempt_post_grid_velocity:{axis_name}",
                    post,
                    True,
                    "face",
                    grid_support_id,
                    layout.layout_id,
                    "face_value",
                    provenance_id=attempt_provenance,
                ),
            )
        )

    projection = result.diagnostics.details
    for name, values, representation in (
        ("attempt_liquid_mask", projection.liquid_mask, "support_mask"),
        ("attempt_pressure_increment", projection.pressure_increment, "cell_value"),
        ("attempt_divergence_before", projection.divergence_before, "cell_value"),
        ("attempt_divergence_after", projection.divergence_after, "cell_value"),
    ):
        fields.append(
            HostInspectionField(
                name,
                values,
                True,
                "cell",
                grid_support_id,
                discretization.cell_layout.layout_id,
                representation,
                provenance_id=attempt_provenance,
            )
        )

    global_layout_id = f"{compiled.compilation_id}:step-diagnostics"
    diagnostics = result.diagnostics
    for name, values, representation in (
        ("liquid_count", diagnostics.liquid_count, "diagnostic_count"),
        ("air_count", diagnostics.air_count, "diagnostic_count"),
        ("classification_margin", diagnostics.classification_margin, "diagnostic"),
        ("mass_balance_defect", diagnostics.mass_balance_defect, "diagnostic"),
        ("momentum_balance_defect", diagnostics.momentum_balance_defect, "diagnostic"),
        ("extrapolation_holes", diagnostics.extrapolation_holes, "diagnostic_count"),
        ("projection_residual", diagnostics.projection_residual, "diagnostic"),
        ("divergence_norm", diagnostics.divergence_norm, "diagnostic"),
        (
            "maximum_displacement_fraction",
            diagnostics.maximum_displacement_fraction,
            "diagnostic",
        ),
        ("energy_before", diagnostics.energy_before, "diagnostic"),
        ("energy_after", diagnostics.energy_after, "diagnostic"),
        ("step_successful", diagnostics.successful, "diagnostic_flag"),
        ("rejection_reason", diagnostics.rejection_reason, "diagnostic_status"),
    ):
        fields.append(
            HostInspectionField(
                name,
                values,
                True,
                "global",
                grid_support_id,
                global_layout_id,
                representation,
                provenance_id=attempt_provenance,
            )
        )

    frame = HostInspectionFrame(
        state.time,
        state.accepted_step,
        state_kind,
        result.successful,
        state.status,
        tuple(fields),
        compiled.compilation_id,
        result_id,
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "phydrax-flip-state-and-step-evidence",
        "phydrax-host-inspection-frame",
        source_id=f"{state_provenance}+{attempt_provenance}",
        target_id=result_id,
        coordinate_mapping=("native-time", "native-accepted-step"),
        preserved_fields=tuple(field.name for field in fields),
    )
    return HostInspectionConversion(frame, report)


def flip_inspection_frames(
    compiled: CompiledFLIPProblem,
    result: FLIPStepResult,
    /,
    *,
    result_id: str,
) -> tuple[HostInspectionConversion, HostInspectionConversion]:
    """Return distinct candidate and rollback-aware accepted FLIP host frames."""
    if not isinstance(compiled, CompiledFLIPProblem):
        raise TypeError("compiled must be a CompiledFLIPProblem.")
    if not isinstance(result, FLIPStepResult):
        raise TypeError("result must be a FLIPStepResult.")
    if not isinstance(result.candidate_state, FLIPRuntimeState) or not isinstance(
        result.accepted_state, FLIPRuntimeState
    ):
        raise TypeError("FLIP result states must be FLIPRuntimeState.")
    identifier = str(result_id).strip()
    if not identifier:
        raise ValueError("result_id must be a non-empty string.")
    if np.asarray(result.successful).shape != ():
        raise ValueError("FLIP result successful must be scalar.")
    candidate = _flip_inspection_frame(
        compiled,
        result.candidate_state,
        result,
        state_kind="candidate",
        attempt_id=identifier,
        result_id=f"{identifier}:candidate",
    )
    accepted = _flip_inspection_frame(
        compiled,
        result.accepted_state,
        result,
        state_kind="accepted",
        attempt_id=identifier,
        result_id=f"{identifier}:accepted",
    )
    return candidate, accepted


__all__ = ["flip_inspection_frames"]
