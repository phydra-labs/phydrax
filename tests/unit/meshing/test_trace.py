import pytest

from phydrax.meshing import (
    MeshingDiagnostic,
    MeshingDiagnosticSeverity,
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)


def test_failed_stage_terminates_trace_and_prevents_success():
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
