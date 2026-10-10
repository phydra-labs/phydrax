import pytest

from phydrax.meshing import (
    MeshingDiagnostic,
    MeshingDiagnosticSeverity,
    MeshingEvidenceBinding,
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)


def test_failed_stage_terminates_trace_and_prevents_success() -> None:
    start = MeshingStageReport(
        MeshingStageKind.SOURCE_INSPECTION, MeshingStageStatus.PASSED
    )
    failed = MeshingStageReport(
        MeshingStageKind.SURFACE_MESHING,
        MeshingStageStatus.FAILED,
        diagnostics=(
            MeshingDiagnostic(MeshingDiagnosticSeverity.ERROR, "Meshing failed."),
        ),
    )
    assert not MeshingTrace((start, failed)).successful
    with pytest.raises(ValueError, match="terminate"):
        MeshingTrace((failed, start))


def test_unresolved_mandatory_check_is_terminal_and_carries_quantities() -> None:
    diagnostic = MeshingDiagnostic(
        MeshingDiagnosticSeverity.ERROR,
        "global_embedding unresolved: exterior_degree_capacity",
        entity_ids=(7, 9),
        quantities=(("mesh_to_source_deviation", 0.1, 0.25),),
    )
    unresolved = MeshingStageReport(
        MeshingStageKind.CERTIFICATION,
        MeshingStageStatus.UNRESOLVED,
        diagnostics=(diagnostic,),
    )
    start = MeshingStageReport(MeshingStageKind.VOLUME_FILL, MeshingStageStatus.PASSED)

    assert diagnostic.quantities == (("mesh_to_source_deviation", 0.1, 0.25),)
    assert not MeshingTrace((start, unresolved)).successful
    with pytest.raises(ValueError, match="terminate"):
        MeshingTrace((unresolved, start))
    with pytest.raises(ValueError, match="error diagnostic"):
        MeshingStageReport(MeshingStageKind.CERTIFICATION, MeshingStageStatus.UNRESOLVED)


def test_evidence_binding_is_part_of_trace_identity() -> None:
    stage = MeshingStageReport(MeshingStageKind.VOLUME_FILL, MeshingStageStatus.PASSED)

    def binding(revision: str) -> MeshingEvidenceBinding:
        return MeshingEvidenceBinding(
            source_id="domain",
            source_revision=revision,
            topology_id="topology",
            geometry_id="geometry",
            geometry_layout_id="layout",
            policy_ids=("audit-policy", "limits"),
            runtime_id="runtime",
        )

    first = MeshingTrace((stage,), binding=binding("r1"))
    second = MeshingTrace((stage,), binding=binding("r2"))

    assert first.binding is not None
    assert first.binding.policy_ids == ("audit-policy", "limits")
    assert first.trace_id != second.trace_id
    assert first.trace_id != MeshingTrace((stage,)).trace_id
