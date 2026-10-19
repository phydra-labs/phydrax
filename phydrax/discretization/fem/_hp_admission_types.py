# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Resolve nominal hp admission records after the meshing import cycle closes."""

from __future__ import annotations

from functools import cache
from typing import NamedTuple, TYPE_CHECKING


if TYPE_CHECKING:
    from ...meshing._decision import (
        AdaptationAction,
        PhysicalErrorEvidence,
        RouteFeasibility,
        SolverAwareDecision,
    )


class HPAdmissionTypes(NamedTuple):
    action: type[AdaptationAction]
    physical_evidence: type[PhysicalErrorEvidence]
    solver_decision: type[SolverAwareDecision]
    feasibility: type[RouteFeasibility]


@cache
def hp_admission_types() -> HPAdmissionTypes:
    """Return the four nominal classes used by hp admission boundaries."""
    # HP preparation -> meshing decisions -> providers -> lineage -> HP preparation.
    from ...meshing._decision import (
        AdaptationAction,
        PhysicalErrorEvidence,
        RouteFeasibility,
        SolverAwareDecision,
    )

    return HPAdmissionTypes(
        AdaptationAction, PhysicalErrorEvidence, SolverAwareDecision, RouteFeasibility
    )
