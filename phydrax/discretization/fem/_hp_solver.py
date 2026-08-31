#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import LocalEliminationPlan, LocalEliminationResult
from ._hp import FiniteElementHPTransferPlan
from ._hp_runtime import FiniteElementHPEpoch, FiniteElementHPTraceConstraintPlan


class FiniteElementHPCondensationPlan(StrictModule, NonTrainableState):
    """Degree-bucket local elimination with retained trace coordinates."""

    bucket_degrees: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    eliminations: tuple[LocalEliminationPlan, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bucket_degrees: Sequence[tuple[int, ...]],
        eliminations: Sequence[LocalEliminationPlan],
        /,
    ):
        degrees = tuple(
            tuple(int(value) for value in degree) for degree in bucket_degrees
        )
        plans = tuple(eliminations)
        if (
            not degrees
            or len(degrees) != len(plans)
            or len(set(degrees)) != len(degrees)
            or any(
                not isinstance(plan, LocalEliminationPlan) for plan in plans
            )
        ):
            raise ValueError("hp condensation degrees or elimination plans are invalid.")
        self.bucket_degrees = degrees
        self.eliminations = plans
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-condensation",
                "degrees": [list(value) for value in degrees],
                "eliminations": [value.plan_id for value in plans],
            }
        )

    def condense(
        self,
        degree: tuple[int, ...],
        local_matrix: ArrayLike,
        local_rhs: ArrayLike,
        /,
    ) -> LocalEliminationResult:
        key = tuple(int(value) for value in degree)
        if key not in self.bucket_degrees:
            raise KeyError(f"No hp condensation bucket for degree {key!r}.")
        return self.eliminations[self.bucket_degrees.index(key)].condense(
            local_matrix,
            local_rhs,
        )

    def reconstruct(
        self,
        degree: tuple[int, ...],
        retained_solution: ArrayLike,
        result: LocalEliminationResult,
        /,
    ) -> Array:
        key = tuple(int(value) for value in degree)
        if key not in self.bucket_degrees:
            raise KeyError(f"No hp condensation bucket for degree {key!r}.")
        return self.eliminations[self.bucket_degrees.index(key)].reconstruct(
            retained_solution,
            result,
        )


def finite_element_hp_condensation_plan(
    epoch: FiniteElementHPEpoch,
    field_name: str,
    /,
) -> FiniteElementHPCondensationPlan:
    if epoch.discretization is None:
        raise ValueError("hp condensation requires a prepared discretization.")
    field_index = epoch.discretization._field_index(field_name)
    degrees = []
    plans = []
    for element in epoch.discretization.elements[field_index]:
        nodes = np.asarray(element.reference_nodes)
        boundary = np.any(
            np.isclose(nodes, 0.0) | np.isclose(nodes, 1.0),
            axis=1,
        )
        retained = np.flatnonzero(boundary).astype(np.int32)
        if retained.size == 0 or retained.size == element.local_dof_count:
            continue
        axis_degrees = tuple(
            np.unique(nodes[:, axis]).size - 1 for axis in range(nodes.shape[1])
        )
        degrees.append(axis_degrees)
        plans.append(LocalEliminationPlan(element.local_dof_count, retained))
    if not plans:
        raise ValueError(
            "hp condensation requires at least one element with interior DOFs."
        )
    return FiniteElementHPCondensationPlan(degrees, plans)


class FiniteElementHPSkeletonPlan(StrictModule, NonTrainableState):
    """Constraint and mortar identities defining one hp trace skeleton."""

    trace_constraint_ids: tuple[str, ...] = eqx.field(static=True)
    interface_ids: tuple[str, ...] = eqx.field(static=True)
    retained_dofs_by_degree: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        epoch: FiniteElementHPEpoch,
        condensation: FiniteElementHPCondensationPlan,
        /,
    ):
        constraints = tuple(
            value.plan_id
            for _, value in epoch.constraints
            if isinstance(value, FiniteElementHPTraceConstraintPlan)
        )
        interfaces = tuple(
            value
            for value, valid in zip(
                epoch.interfaces.interface_ids,
                np.asarray(epoch.interfaces.valid),
                strict=True,
            )
            if valid
        )
        retained = tuple(
            tuple(int(value) for value in np.asarray(plan.retained_dofs))
            for plan in condensation.eliminations
        )
        self.trace_constraint_ids = constraints
        self.interface_ids = interfaces
        self.retained_dofs_by_degree = retained
        self.epoch_id = epoch.epoch_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-skeleton",
                "epoch": epoch.epoch_id,
                "constraints": list(constraints),
                "interfaces": list(interfaces),
                "retained": [list(value) for value in retained],
            }
        )


class FiniteElementHPMultigridPlan(StrictModule, NonTrainableState):
    """Combined h/p epoch hierarchy with explicit adjacent transfer roles."""

    level_epoch_ids: tuple[str, ...] = eqx.field(static=True)
    transfers: tuple[FiniteElementHPTransferPlan, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        epochs: Sequence[FiniteElementHPEpoch],
        transfers: Sequence[FiniteElementHPTransferPlan],
        /,
    ):
        levels = tuple(epochs)
        transfers_ = tuple(transfers)
        if len(levels) < 2 or len(transfers_) != len(levels) - 1:
            raise ValueError(
                "hp multigrid requires one transfer between adjacent levels."
            )
        for fine, coarse, transfer in zip(
            levels[:-1], levels[1:], transfers_, strict=True
        ):
            if (
                not isinstance(fine, FiniteElementHPEpoch)
                or not isinstance(coarse, FiniteElementHPEpoch)
                or not isinstance(transfer, FiniteElementHPTransferPlan)
                or transfer.source_topology_id != fine.topology.topology_id
                or transfer.target_topology_id != coarse.topology.topology_id
            ):
                raise ValueError("hp multigrid epoch and transfer identities disagree.")
        self.level_epoch_ids = tuple(value.epoch_id for value in levels)
        self.transfers = transfers_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-multigrid",
                "levels": list(self.level_epoch_ids),
                "transfers": [value.transfer_id for value in transfers_],
            }
        )

    def restrict(self, level: int, values: ArrayLike, /) -> Array:
        index = int(level)
        if index < 0 or index >= len(self.transfers):
            raise ValueError("hp multigrid restriction level is out of range.")
        return self.transfers[index].apply_mass_projection(values)

    def pullback(self, level: int, coarse_dual: ArrayLike, /) -> Array:
        index = int(level)
        if index < 0 or index >= len(self.transfers):
            raise ValueError("hp multigrid pullback level is out of range.")
        return self.transfers[index].pullback_raw(coarse_dual)


class FiniteElementHPSolverRefreshPlan(StrictModule, NonTrainableState):
    """Inspectable route, metric, skeleton, and signature-cache invalidation."""

    reused_signatures: tuple[str, ...] = eqx.field(static=True)
    new_signatures: tuple[str, ...] = eqx.field(static=True)
    routes_changed: bool = eqx.field(static=True)
    metrics_changed: bool = eqx.field(static=True)
    skeleton_changed: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        accepted: FiniteElementHPEpoch,
        candidate: FiniteElementHPEpoch,
        /,
    ):
        accepted_signatures = {
            canonical_fingerprint(
                {
                    "kind": "finite-element-hp-kernel-signature",
                    "degree": [int(value) for value in degree],
                    "cell_kind": accepted.topology.cell_kind,
                }
            )
            for degree, valid in zip(
                np.asarray(accepted.worksets.bucket_degrees),
                np.asarray(accepted.worksets.bucket_valid),
                strict=True,
            )
            if valid
        }
        candidate_signatures = {
            canonical_fingerprint(
                {
                    "kind": "finite-element-hp-kernel-signature",
                    "degree": [int(value) for value in degree],
                    "cell_kind": candidate.topology.cell_kind,
                }
            )
            for degree, valid in zip(
                np.asarray(candidate.worksets.bucket_degrees),
                np.asarray(candidate.worksets.bucket_valid),
                strict=True,
            )
            if valid
        }
        reused = tuple(sorted(accepted_signatures & candidate_signatures))
        new = tuple(sorted(candidate_signatures - accepted_signatures))
        self.reused_signatures = reused
        self.new_signatures = new
        self.routes_changed = accepted.topology.plan_id != candidate.topology.plan_id
        self.metrics_changed = (
            accepted.geometry.geometry_id != candidate.geometry.geometry_id
        )
        self.skeleton_changed = (
            accepted.interfaces.plan_id != candidate.interfaces.plan_id
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-solver-refresh",
                "accepted": accepted.epoch_id,
                "candidate": candidate.epoch_id,
                "reused": list(reused),
                "new": list(new),
                "routes_changed": self.routes_changed,
                "metrics_changed": self.metrics_changed,
                "skeleton_changed": self.skeleton_changed,
            }
        )


__all__ = [
    "FiniteElementHPCondensationPlan",
    "FiniteElementHPMultigridPlan",
    "FiniteElementHPSkeletonPlan",
    "FiniteElementHPSolverRefreshPlan",
    "finite_element_hp_condensation_plan",
]
