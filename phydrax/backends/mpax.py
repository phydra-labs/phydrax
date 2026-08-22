#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._availability import import_backend_module, probe_backend
from ._types import AbstractExternalBackend, BackendAvailability, BackendCapabilities


MPAXAlgorithm: TypeAlias = Literal["rapdhg", "r2hpdhg"]

MPAX_CAPABILITIES = BackendCapabilities(
    backend="mpax",
    problem_kinds=("optimization.linear-program", "optimization.quadratic-program"),
    execution="device",
    host_only=False,
    supports_matrix_free=False,
    supports_assembled=True,
    coordinate_dtypes=("float32", "float64"),
    supports_plan_prepare_solve_refresh=True,
)


def mpax_availability() -> BackendAvailability:
    """Probe MPAX without importing it during package import."""

    return probe_backend(
        MPAX_CAPABILITIES,
        module="mpax",
        requirement="install a compatible mpax==0.2.4 distribution",
        distributions=("mpax",),
    )


class MPAXBackend(AbstractExternalBackend):
    """Lazy optional MPAX backend inspection boundary."""

    @property
    def name(self) -> str:
        return "mpax"

    @property
    def capabilities(self) -> BackendCapabilities:
        return MPAX_CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return mpax_availability()


class MPAXPlan(StrictModule):
    """Static first-order algorithm and termination configuration."""

    algorithm: MPAXAlgorithm = eqx.field(static=True)
    eps_abs: float = eqx.field(static=True)
    eps_rel: float = eqx.field(static=True)
    eps_primal_infeasible: float = eqx.field(static=True)
    eps_dual_infeasible: float = eqx.field(static=True)
    iteration_limit: int = eqx.field(static=True)
    warm_start: bool = eqx.field(static=True)
    feasibility_polishing: bool = eqx.field(static=True)
    unroll: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        algorithm: MPAXAlgorithm = "rapdhg",
        /,
        *,
        eps_abs: float = 1e-6,
        eps_rel: float = 1e-6,
        eps_primal_infeasible: float = 1e-8,
        eps_dual_infeasible: float = 1e-8,
        iteration_limit: int = 10_000,
        warm_start: bool = False,
        feasibility_polishing: bool = False,
        unroll: bool = False,
    ):
        if algorithm not in ("rapdhg", "r2hpdhg"):
            raise ValueError("algorithm must be 'rapdhg' or 'r2hpdhg'.")
        values = tuple(
            float(value)
            for value in (
                eps_abs,
                eps_rel,
                eps_primal_infeasible,
                eps_dual_infeasible,
            )
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("MPAX tolerances must be positive.")
        steps = int(iteration_limit)
        if steps < 1:
            raise ValueError("iteration_limit must be positive.")
        if bool(unroll) and steps >= 100_000:
            raise ValueError("Unrolled MPAX execution requires a bounded finite budget.")
        self.algorithm = algorithm
        (
            self.eps_abs,
            self.eps_rel,
            self.eps_primal_infeasible,
            self.eps_dual_infeasible,
        ) = values
        self.iteration_limit = steps
        self.warm_start = bool(warm_start)
        self.feasibility_polishing = bool(feasibility_polishing)
        self.unroll = bool(unroll)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpax-plan",
                "algorithm": algorithm,
                "eps_abs": values[0],
                "eps_rel": values[1],
                "eps_primal_infeasible": values[2],
                "eps_dual_infeasible": values[3],
                "iteration_limit": steps,
                "warm_start": bool(warm_start),
                "feasibility_polishing": bool(feasibility_polishing),
                "unroll": bool(unroll),
            }
        )


class PreparedMPAX(StrictModule):
    """Instantiated MPAX solver paired with one immutable plan."""

    plan: MPAXPlan
    solver: Any
    backend_version: str = eqx.field(static=True)

    def __init__(self, plan: MPAXPlan, solver: Any, /, *, backend_version: str):
        if not isinstance(plan, MPAXPlan):
            raise TypeError("plan must be an MPAXPlan.")
        version = str(backend_version)
        if not version:
            raise ValueError("backend_version must be non-empty.")
        self.plan = plan
        self.solver = solver
        self.backend_version = version


def prepare_mpax(plan: MPAXPlan | None = None, /) -> PreparedMPAX:
    """Import and instantiate the explicitly selected MPAX algorithm."""

    selected = MPAXPlan() if plan is None else plan
    if not isinstance(selected, MPAXPlan):
        raise TypeError("plan must be an MPAXPlan or None.")
    availability = mpax_availability()
    module = import_backend_module(availability, "optimization.linear-program", "mpax")
    common = dict(
        eps_abs=selected.eps_abs,
        eps_rel=selected.eps_rel,
        eps_primal_infeasible=selected.eps_primal_infeasible,
        eps_dual_infeasible=selected.eps_dual_infeasible,
        iteration_limit=selected.iteration_limit,
        warm_start=selected.warm_start,
        feasibility_polishing=selected.feasibility_polishing,
        unroll=selected.unroll,
        verbose=False,
    )
    solver = (
        module.raPDHG(**common)
        if selected.algorithm == "rapdhg"
        else module.r2HPDHG(**common)
    )
    version = dict(availability.versions).get("mpax", "unknown")
    return PreparedMPAX(selected, solver, backend_version=version)


def refresh_mpax(prepared: PreparedMPAX, /) -> PreparedMPAX:
    """Return unchanged stateless algorithm preparation for new numeric program data."""

    if not isinstance(prepared, PreparedMPAX):
        raise TypeError("prepared must be a PreparedMPAX.")
    return prepared


def solve_mpax(
    prepared: PreparedMPAX,
    problem: Any,
    /,
    *,
    initial_primal_solution: Any = None,
    initial_dual_solution: Any = None,
):
    """Execute MPAX on one already converted LP/QP model."""

    if not isinstance(prepared, PreparedMPAX):
        raise TypeError("prepared must be a PreparedMPAX.")
    if (
        initial_primal_solution is not None or initial_dual_solution is not None
    ) and not (prepared.plan.warm_start):
        raise ValueError("MPAX warm-start arrays require plan.warm_start=True.")
    return prepared.solver.optimize(
        problem,
        initial_primal_solution=initial_primal_solution,
        initial_dual_solution=initial_dual_solution,
    )


__all__ = [
    "MPAXAlgorithm",
    "MPAXBackend",
    "MPAXPlan",
    "MPAX_CAPABILITIES",
    "PreparedMPAX",
    "mpax_availability",
    "prepare_mpax",
    "refresh_mpax",
    "solve_mpax",
]
