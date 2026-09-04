#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Optional host-only OpticStudio interoperability."""

from ._adapter import (
    export_sequential_to_opticstudio,
    opticstudio_availability,
    OPTICSTUDIO_CAPABILITIES,
    OpticStudioAnalysisRequest,
    OpticStudioBackend,
    OpticStudioRunResult,
    OpticStudioSession,
    run_opticstudio_analysis,
)


__all__ = [
    "OPTICSTUDIO_CAPABILITIES",
    "OpticStudioAnalysisRequest",
    "OpticStudioBackend",
    "OpticStudioRunResult",
    "OpticStudioSession",
    "export_sequential_to_opticstudio",
    "opticstudio_availability",
    "run_opticstudio_analysis",
]
