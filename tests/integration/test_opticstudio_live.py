#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os

import pytest

from phydrax.interchange.opticstudio import (
    opticstudio_availability,
    OpticStudioAnalysisRequest,
    OpticStudioBackend,
    run_opticstudio_analysis,
)


@pytest.mark.opticstudio_live
@pytest.mark.skipif(
    os.environ.get("PHYDRAX_RUN_OPTICSTUDIO_LIVE") != "1",
    reason="set PHYDRAX_RUN_OPTICSTUDIO_LIVE=1 to run the OpticStudio live guard",
)
def test_live_opticstudio_session_and_system_data_analysis():
    availability = opticstudio_availability()
    if not availability.available:
        pytest.skip(f"OpticStudio unavailable: {availability.reason}")
    try:
        session = OpticStudioBackend().open_session()
    except Exception as error:
        pytest.skip(
            "OpticStudio package is present but a licensed live session is unavailable: "
            f"{type(error).__name__}"
        )
    with session:
        result = run_opticstudio_analysis(
            session, OpticStudioAnalysisRequest("system-data")
        )
    assert result.report.valid
    assert result.artifact.status == "complete"
    assert result.payload_json
