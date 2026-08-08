#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact condition transforms and their typed staged compiler."""

from ._ansatz import (
    enforce_blend,
    enforce_dirichlet,
    enforce_initial,
    enforce_neumann,
    enforce_robin,
    enforce_sommerfeld,
    enforce_traction,
)
from ._api import compile, EnforcementOptions
from ._compile import EnforcementProgram, InteriorAnchors
from ._graph import enforce_cochain_values, enforce_graph_values
from ._spec import (
    DerivativeRequirement,
    EnforcementKind,
    EnforcementSpec,
    EnforcementStage,
)
from ._trajectory import (
    enforce_ragged_time_series,
    RaggedTimeSeriesHardGate,
    RaggedTimeSeriesHardInterpolation,
)


__all__ = [
    "DerivativeRequirement",
    "EnforcementKind",
    "EnforcementOptions",
    "EnforcementProgram",
    "EnforcementSpec",
    "EnforcementStage",
    "InteriorAnchors",
    "RaggedTimeSeriesHardGate",
    "RaggedTimeSeriesHardInterpolation",
    "compile",
    "enforce_blend",
    "enforce_cochain_values",
    "enforce_dirichlet",
    "enforce_graph_values",
    "enforce_initial",
    "enforce_neumann",
    "enforce_ragged_time_series",
    "enforce_robin",
    "enforce_sommerfeld",
    "enforce_traction",
]
