# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Bound rational reference preparation with certificate domain limits."""

from collections.abc import Iterator
from contextlib import contextmanager

from phydrax._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.meshing import MeshingLimits


@contextmanager
def coordinate_reference_budget(
    limits: MeshCertificateLimits | MeshingLimits | None = None,
    *,
    queries_upper: int = 0,
) -> Iterator[CoordinateEnclosureBudget]:
    """Yield a reference ledger and settle arithmetic work before host reservation.

    Work and managed-storage bounds come from the supplied resource limits.
    Callers sum the structural query bounds of each source preparation in the
    request. Arithmetic-only references require no source-query allowance.
    """
    limits = MeshCertificateLimits() if limits is None else limits
    ticket = HostBatchTicket(
        limits.maximum_work_units,
        queries_upper=queries_upper,
        bytes_upper=limits.maximum_scratch_bytes,
    )
    budget = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=limits.maximum_work_units,
        maximum_memory_bytes=limits.maximum_scratch_bytes,
    )
    with ticket, budget.activate():
        yield budget


@contextmanager
def native_meshing_request(
    limits: MeshingLimits, /
) -> Iterator[
    tuple[NativeExecutionScope, CoordinateEnclosureBudget, NativeHostStorageWorkspace]
]:
    """Yield the supplied native allowances, an idle coordinate ledger, and workspace.

    The work bootstrap is settled before yielding, so a product phase can bind
    its own ticket. Retained coordinate caches close before the workspace and
    native scope, including on scientific refusal.
    """
    with NativeExecutionScope(limits.native_limits()) as scope:
        bootstrap = scope.batch(0)
        bootstrap.settle()
        budget = CoordinateEnclosureBudget(
            bootstrap,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with (
            scope.host_workspace() as workspace,
            budget.retained_storage(
                lambda required: scope.batch(0, bytes_upper=required)
            ),
        ):
            yield scope, budget, workspace
