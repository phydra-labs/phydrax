#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from types import ModuleType

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._availability import import_backend_module, probe_backend
from ._types import AbstractExternalBackend, BackendAvailability, BackendCapabilities


CLARABEL_CAPABILITIES = BackendCapabilities(
    backend="clarabel",
    problem_kinds=(
        "optimization.linear-program",
        "optimization.quadratic-program",
        "optimization.conic-program",
    ),
    execution="host",
    host_only=True,
    supports_matrix_free=False,
    supports_assembled=True,
    coordinate_dtypes=("float64",),
    supports_plan_prepare_solve_refresh=True,
)


def clarabel_availability() -> BackendAvailability:
    """Probe Clarabel without importing its native extension at package import."""

    return probe_backend(
        CLARABEL_CAPABILITIES,
        module="clarabel",
        requirement="install phydrax[clarabel] (clarabel==0.11.1)",
        distributions=("clarabel",),
    )


class ClarabelBackend(AbstractExternalBackend):
    """Lazy host conic-backend inspection boundary."""

    @property
    def name(self) -> str:
        return "clarabel"

    @property
    def capabilities(self) -> BackendCapabilities:
        return CLARABEL_CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return clarabel_availability()


class ClarabelPlan(StrictModule):
    """Immutable Clarabel tolerance and iteration configuration."""

    max_iterations: int = eqx.field(static=True)
    tolerance_gap_abs: float = eqx.field(static=True)
    tolerance_gap_rel: float = eqx.field(static=True)
    tolerance_feasibility: float = eqx.field(static=True)
    presolve: bool = eqx.field(static=True)
    verbose: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_iterations: int = 200,
        tolerance_gap_abs: float = 1e-8,
        tolerance_gap_rel: float = 1e-8,
        tolerance_feasibility: float = 1e-8,
        presolve: bool = True,
        verbose: bool = False,
    ):
        steps = int(max_iterations)
        tolerances = tuple(
            float(value)
            for value in (
                tolerance_gap_abs,
                tolerance_gap_rel,
                tolerance_feasibility,
            )
        )
        if steps < 1:
            raise ValueError("max_iterations must be positive.")
        if any(not isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("Clarabel tolerances must be finite and positive.")
        self.max_iterations = steps
        (
            self.tolerance_gap_abs,
            self.tolerance_gap_rel,
            self.tolerance_feasibility,
        ) = tolerances
        self.presolve = bool(presolve)
        self.verbose = bool(verbose)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "clarabel-plan",
                "max_iterations": steps,
                "tolerances": list(tolerances),
                "presolve": bool(presolve),
                "verbose": bool(verbose),
            }
        )


class PreparedClarabel(StrictModule):
    """Imported Clarabel extension paired with immutable settings."""

    plan: ClarabelPlan
    module: ModuleType
    settings: object
    backend_version: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ClarabelPlan,
        module: ModuleType,
        settings: object,
        /,
        *,
        backend_version: str,
    ):
        if not isinstance(plan, ClarabelPlan):
            raise TypeError("plan must be a ClarabelPlan.")
        version = str(backend_version)
        if not version:
            raise ValueError("backend_version must be non-empty.")
        self.plan = plan
        self.module = module
        self.settings = settings
        self.backend_version = version


def prepare_clarabel(plan: ClarabelPlan | None = None, /) -> PreparedClarabel:
    """Import Clarabel and create settings without binding numeric program data."""

    selected = ClarabelPlan() if plan is None else plan
    if not isinstance(selected, ClarabelPlan):
        raise TypeError("plan must be a ClarabelPlan or None.")
    availability = clarabel_availability()
    module = import_backend_module(
        availability,
        "optimization.conic-program",
        "clarabel",
    )
    settings = module.DefaultSettings()
    settings.max_iter = selected.max_iterations
    settings.tol_gap_abs = selected.tolerance_gap_abs
    settings.tol_gap_rel = selected.tolerance_gap_rel
    settings.tol_feas = selected.tolerance_feasibility
    settings.presolve_enable = selected.presolve
    settings.verbose = selected.verbose
    version = dict(availability.versions).get("clarabel", "unknown")
    return PreparedClarabel(
        selected,
        module,
        settings,
        backend_version=version,
    )


__all__ = [
    "CLARABEL_CAPABILITIES",
    "ClarabelBackend",
    "ClarabelPlan",
    "PreparedClarabel",
    "clarabel_availability",
    "prepare_clarabel",
]
