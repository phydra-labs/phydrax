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
from ._diffusion_problem import (
    DiffusionBridgePlan,
    DiffusionBridgeProblem,
    PreparedDiffusionBridge,
)
from ._diffusion_solver import (
    DiffusionBridgeDiagnostics,
    DiffusionBridgeResult,
    prepare_diffusion_bridge,
    sample_diffusion_bridge,
    solve_diffusion_bridge,
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
    "DiffusionBridgeDiagnostics",
    "DiffusionBridgePlan",
    "DiffusionBridgeProblem",
    "DiffusionBridgeResult",
    "FiniteBridgeTarget",
    "SchrodingerBridgeDiagnostics",
    "SchrodingerBridgeProblem",
    "SchrodingerBridgeResult",
    "SchrodingerBridgeSolver",
    "PreparedDiffusionBridge",
    "TerminalDistributionControlAdapter",
    "bridge_path_law_diagnostics",
    "bridge_path_log_prob",
    "reference_path_log_prob",
    "prepare_diffusion_bridge",
    "require_converged_bridge",
    "sample_bridge",
    "sample_bridge_paths",
    "sample_bridge_state_indices",
    "sample_diffusion_bridge",
    "solve_schrodinger_bridge",
    "solve_diffusion_bridge",
]
