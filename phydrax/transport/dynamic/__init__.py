#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact finite-state Schrödinger bridges and dynamic transport."""

from ._adapters import (
    bridge_path_law_diagnostics,
    BridgeInferenceAdapter,
    BridgePathLawDiagnostics,
    TerminalDistributionControlAdapter,
)
from ._kernel import (
    bridge_path_log_prob,
    BridgePathSample,
    ControlledTransitionKernel,
    reference_path_log_prob,
    sample_bridge,
    sample_bridge_paths,
    sample_bridge_state_indices,
)
from ._problem import (
    BridgeProblemProvenance,
    FiniteBridgeTarget,
    SchrodingerBridgeProblem,
)
from ._solver import (
    BridgeProvenance,
    require_converged_bridge,
    SchrodingerBridgeDiagnostics,
    SchrodingerBridgeResult,
    SchrodingerBridgeSolver,
    solve_schrodinger_bridge,
)


__all__ = [
    "BridgeInferenceAdapter",
    "BridgePathLawDiagnostics",
    "BridgePathSample",
    "BridgeProblemProvenance",
    "BridgeProvenance",
    "ControlledTransitionKernel",
    "FiniteBridgeTarget",
    "SchrodingerBridgeDiagnostics",
    "SchrodingerBridgeProblem",
    "SchrodingerBridgeResult",
    "SchrodingerBridgeSolver",
    "TerminalDistributionControlAdapter",
    "bridge_path_law_diagnostics",
    "bridge_path_log_prob",
    "reference_path_log_prob",
    "require_converged_bridge",
    "sample_bridge",
    "sample_bridge_paths",
    "sample_bridge_state_indices",
    "solve_schrodinger_bridge",
]
