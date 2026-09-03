#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    AbstractPreconditioner,
    AbstractPreconditionerBuilder,
    BlockFactorizationPreconditionerBuilder,
    JacobiPreconditionerBuilder,
    PreconditioningPolicy,
)


class MACImmersedPressureBlockPreconditionerEvidence(StrictModule, NonTrainableState):
    pressure_component_id: str = eqx.field(static=True)
    marker_component_id: str = eqx.field(static=True)
    factorization: str = eqx.field(static=True)
    owns_kkt_operator: bool = eqx.field(static=True)
    pressure_action_reused: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class MACImmersedPressureBlockPreconditionerPlan(StrictModule, NonTrainableState):
    """Compose a pressure approximate inverse into the existing immersed KKT solve.

    The composition supplies only the pressure and marker diagonal actions.  Assembly,
    signs, off-diagonal transfer blocks, and KKT ownership remain with the immersed
    projection plan.
    """

    pressure_solver: AbstractPreconditioner | AbstractPreconditionerBuilder
    marker_solver: AbstractPreconditioner | AbstractPreconditionerBuilder
    factorization: str = eqx.field(static=True)
    side: str = eqx.field(static=True)
    refresh: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    evidence: MACImmersedPressureBlockPreconditionerEvidence

    def __init__(
        self,
        pressure_solver: AbstractPreconditioner
        | AbstractPreconditionerBuilder
        | None = None,
        marker_solver: AbstractPreconditioner
        | AbstractPreconditionerBuilder
        | None = None,
        /,
        *,
        factorization: str = "diagonal",
        side: str = "right",
        refresh: str = "numeric",
    ):
        pressure = (
            JacobiPreconditionerBuilder() if pressure_solver is None else pressure_solver
        )
        marker = JacobiPreconditionerBuilder() if marker_solver is None else marker_solver
        allowed = (AbstractPreconditioner, AbstractPreconditionerBuilder)
        if not isinstance(pressure, allowed) or not isinstance(marker, allowed):
            raise TypeError(
                "Immersed pressure and marker solvers must be prepared preconditioners "
                "or preconditioner builders."
            )
        if factorization not in ("diagonal", "lower", "upper", "ldu"):
            raise ValueError("Unknown immersed block factorization form.")
        if side not in ("left", "right") or refresh not in (
            "frozen",
            "numeric",
            "rebuild",
        ):
            raise ValueError(
                "Immersed preconditioner side or refresh contract is invalid."
            )
        pressure_id = (
            pressure.preconditioner_id
            if isinstance(pressure, AbstractPreconditioner)
            else pressure.builder_id
        )
        marker_id = (
            marker.preconditioner_id
            if isinstance(marker, AbstractPreconditioner)
            else marker.builder_id
        )
        identifier = canonical_fingerprint(
            {
                "kind": "mac-immersed-pressure-block-preconditioner",
                "pressure": pressure_id,
                "marker": marker_id,
                "factorization": factorization,
                "side": side,
                "refresh": refresh,
                "owns_kkt_operator": False,
            }
        )
        self.pressure_solver = pressure
        self.marker_solver = marker
        self.factorization = factorization
        self.side = side
        self.refresh = refresh
        self.plan_id = identifier
        self.evidence = MACImmersedPressureBlockPreconditionerEvidence(
            pressure_component_id=pressure_id,
            marker_component_id=marker_id,
            factorization=factorization,
            owns_kkt_operator=False,
            pressure_action_reused=True,
            evidence_id=canonical_fingerprint(
                {
                    "kind": "mac-immersed-pressure-block-preconditioner-evidence",
                    "plan": identifier,
                }
            ),
        )

    def policy(self, /) -> PreconditioningPolicy:
        builder = BlockFactorizationPreconditionerBuilder(
            self.pressure_solver,
            self.marker_solver,
            self.factorization,
        )
        return PreconditioningPolicy(
            builder,
            side=self.side,
            refresh=self.refresh,
        )


__all__ = [
    "MACImmersedPressureBlockPreconditionerEvidence",
    "MACImmersedPressureBlockPreconditionerPlan",
]
