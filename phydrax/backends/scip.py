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


SCIP_CAPABILITIES = BackendCapabilities(
    backend="scip",
    problem_kinds=("optimization.mixed-integer-linear-program",),
    execution="host",
    host_only=True,
    supports_matrix_free=False,
    supports_assembled=True,
    coordinate_dtypes=("float64",),
    supports_plan_prepare_solve_refresh=False,
    requires_explicit_release=True,
)


def scip_availability() -> BackendAvailability:
    """Probe PySCIPOpt without importing it at package import time."""
    return probe_backend(
        SCIP_CAPABILITIES,
        module="pyscipopt",
        requirement="install phydrax[scip] (pyscipopt==6.2.1)",
        distributions=("pyscipopt",),
    )


class SCIPBackend(AbstractExternalBackend):
    """Lazy host mixed-integer backend inspection boundary."""

    @property
    def name(self) -> str:
        return "scip"

    @property
    def capabilities(self) -> BackendCapabilities:
        return SCIP_CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return scip_availability()


class SCIPPlan(StrictModule):
    """Typed deterministic SCIP resource and optimality settings."""

    maximum_nodes: int = eqx.field(static=True)
    time_limit: float | None = eqx.field(static=True)
    absolute_gap: float = eqx.field(static=True)
    relative_gap: float = eqx.field(static=True)
    presolve: bool = eqx.field(static=True)
    threads: int = eqx.field(static=True)
    random_seed: int = eqx.field(static=True)
    verbose: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_nodes: int = 100_000,
        time_limit: float | None = None,
        absolute_gap: float = 0.0,
        relative_gap: float = 0.0,
        presolve: bool = True,
        threads: int = 1,
        random_seed: int = 0,
        verbose: bool = False,
    ):
        nodes = int(maximum_nodes)
        threads_ = int(threads)
        seed = int(random_seed)
        time = None if time_limit is None else float(time_limit)
        gaps = float(absolute_gap), float(relative_gap)
        if nodes < 1 or threads_ < 1 or seed < 0:
            raise ValueError(
                "SCIP node/thread limits must be positive and seed nonnegative."
            )
        if time is not None and (not isfinite(time) or time <= 0.0):
            raise ValueError("time_limit must be positive finite or None.")
        if any(not isfinite(value) or value < 0.0 for value in gaps):
            raise ValueError("SCIP optimality gaps must be finite and nonnegative.")
        self.maximum_nodes = nodes
        self.time_limit = time
        self.absolute_gap, self.relative_gap = gaps
        self.presolve = bool(presolve)
        self.threads = threads_
        self.random_seed = seed
        self.verbose = bool(verbose)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scip-plan",
                "maximum_nodes": nodes,
                "time_limit": time,
                "absolute_gap": gaps[0],
                "relative_gap": gaps[1],
                "presolve": bool(presolve),
                "threads": threads_,
                "random_seed": seed,
                "verbose": bool(verbose),
            }
        )


class PreparedSCIP(StrictModule):
    """Imported PySCIPOpt module paired with immutable settings."""

    plan: SCIPPlan
    module: ModuleType
    backend_version: str = eqx.field(static=True)


def prepare_scip(plan: SCIPPlan | None = None, /) -> PreparedSCIP:
    selected = SCIPPlan() if plan is None else plan
    if not isinstance(selected, SCIPPlan):
        raise TypeError("plan must be a SCIPPlan or None.")
    availability = scip_availability()
    module = import_backend_module(
        availability,
        "optimization.mixed-integer-linear-program",
        "pyscipopt",
    )
    version = dict(availability.versions).get("pyscipopt", "unknown")
    return PreparedSCIP(selected, module, version)


__all__ = [
    "SCIP_CAPABILITIES",
    "PreparedSCIP",
    "SCIPBackend",
    "SCIPPlan",
    "prepare_scip",
    "scip_availability",
]
