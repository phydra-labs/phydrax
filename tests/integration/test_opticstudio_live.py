#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import pytest

from phydrax.interchange.opticstudio import (
    opticstudio_availability,
    OpticStudioAnalysisRequest,
    OpticStudioBackend,
    run_opticstudio_analysis,
)


@pytest.mark.opticstudio_live
def test_live_opticstudio_session_and_system_data_analysis():
    availability = opticstudio_availability()
    if not availability.available:
        pytest.skip(f"OpticStudio unavailable: {availability.reason}")
    with OpticStudioBackend().open_session() as session:
        result = run_opticstudio_analysis(
            session, OpticStudioAnalysisRequest("system-data")
        )
    assert result.report.valid
    assert result.artifact.status == "complete"
    assert result.payload_json
