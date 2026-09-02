#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from ._availability import probe_backend
from ._types import BackendAvailability, BackendCapabilities


CVXPY_CAPABILITIES = BackendCapabilities(
    backend="cvxpy",
    problem_kinds=("optimization.canonical-convex-model",),
    execution="host",
    host_only=True,
    supports_matrix_free=False,
    supports_assembled=True,
    coordinate_dtypes=("float64",),
    supports_plan_prepare_solve_refresh=False,
)


def cvxpy_availability() -> BackendAvailability:
    """Probe optional CVXPY canonicalization without importing it eagerly."""
    return probe_backend(
        CVXPY_CAPABILITIES,
        module="cvxpy",
        requirement="install a compatible cvxpy distribution",
        distributions=("cvxpy",),
    )


__all__ = ["CVXPY_CAPABILITIES", "cvxpy_availability"]
