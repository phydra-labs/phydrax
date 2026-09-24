#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import (
    AbstractFixedStepMethod,
    FixedStepResult,
    ProductionCaseManifest,
    ProductionRunPlan,
    RobustRetryPolicy,
)
from ._multiphysics import (
    CoupledMultiphysicsPlan,
    CoupledMultiphysicsState,
    CoupledMultiphysicsStepInputs,
)


class CoupledMultiphysicsEpochIdentity(StrictModule, NonTrainableState):
    phase_epoch_id: str = eqx.field(static=True)
    thermal_epoch_id: str = eqx.field(static=True)
    mechanics_topology_id: str = eqx.field(static=True)
    flow_topology_id: str = eqx.field(static=True)
    electrostatic_topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    event_realization_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        phase_epoch_id: str,
        thermal_epoch_id: str,
        mechanics_topology_id: str,
        flow_topology_id: str,
        electrostatic_topology_id: str,
        partition_id: str,
        event_realization_id: str,
    ):
        values = tuple(
            str(value)
            for value in (
                phase_epoch_id,
                thermal_epoch_id,
                mechanics_topology_id,
                flow_topology_id,
                electrostatic_topology_id,
                partition_id,
                event_realization_id,
            )
        )
        if any(not value for value in values):
            raise ValueError("Coupled epoch identity components must be nonempty.")
        (
            self.phase_epoch_id,
            self.thermal_epoch_id,
            self.mechanics_topology_id,
            self.flow_topology_id,
            self.electrostatic_topology_id,
            self.partition_id,
            self.event_realization_id,
        ) = values
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "coupled-multiphysics-epoch",
                "phase": values[0],
                "thermal": values[1],
                "mechanics": values[2],
                "flow": values[3],
                "electrostatic": values[4],
                "partition": values[5],
                "events": values[6],
            }
        )


class CoupledMultiphysicsProductionCase(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    initial_state: CoupledMultiphysicsState
    manifest: ProductionCaseManifest
    case_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        initial_state: CoupledMultiphysicsState,
        /,
        *,
        method_id: str,
        precision_id: str,
        topology_id: str,
        geometry_layout_id: str,
        dtype: str,
    ):
        identifier = str(name)
        if not identifier or not isinstance(initial_state, CoupledMultiphysicsState):
            raise ValueError("Coupled production case is invalid.")
        case_id = canonical_fingerprint(
            {
                "kind": "coupled-multiphysics-production-case",
                "name": identifier,
                "method": method_id,
                "precision": precision_id,
                "topology": topology_id,
                "geometry": geometry_layout_id,
                "dtype": dtype,
                "state": array_tree_fingerprint(initial_state),
            }
        )
        self.name = identifier
        self.initial_state = initial_state
        self.manifest = ProductionCaseManifest(
            problem_id=case_id,
            method_id=method_id,
            precision_id=precision_id,
            topology_id=topology_id,
            geometry_layout_id=geometry_layout_id,
            dtype=dtype,
        )
        self.case_id = case_id


class CoupledMultiphysicsFixedStepMethod(AbstractFixedStepMethod, NonTrainableState):
    plan: CoupledMultiphysicsPlan
    method_id: str = eqx.field(static=True)

    def __init__(self, plan: CoupledMultiphysicsPlan, /):
        if not isinstance(plan, CoupledMultiphysicsPlan):
            raise TypeError("plan must be CoupledMultiphysicsPlan.")
        self.plan = plan
        self.method_id = canonical_fingerprint(
            {
                "kind": "coupled-multiphysics-fixed-step-method",
                "plan": plan.plan_id,
            }
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: CoupledMultiphysicsState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index
        if not isinstance(args, CoupledMultiphysicsStepInputs):
            raise TypeError(
                "Coupled multiphysics fixed step requires CoupledMultiphysicsStepInputs."
            )
        result = self.plan.step(state, args, time, step_size)
        ledger = result.evidence.ledger
        residual = jnp.max(
            jnp.stack(
                (
                    jnp.abs(ledger.energy_residual),
                    ledger.exchange_defect,
                    ledger.maximum_conservation_defect,
                    jnp.abs(ledger.entropy.residual),
                )
            )
        )
        return FixedStepResult(
            result.candidate_state,
            result.accepted_state,
            result.successful,
            residual,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=step_size.dtype),
        )

    def production_case(
        self,
        name: str,
        initial_state: CoupledMultiphysicsState,
        /,
        *,
        precision_id: str,
        topology_id: str,
        geometry_layout_id: str,
        dtype: str,
    ) -> CoupledMultiphysicsProductionCase:
        return CoupledMultiphysicsProductionCase(
            name,
            initial_state,
            method_id=self.method_id,
            precision_id=precision_id,
            topology_id=topology_id,
            geometry_layout_id=geometry_layout_id,
            dtype=dtype,
        )

    def production_run_plan(
        self,
        /,
        *,
        step_size: float,
        end_time: float,
        maximum_steps: int,
        checkpoint_interval: int,
        segment_steps: int = 8,
        retry_policy: RobustRetryPolicy | None = None,
    ) -> ProductionRunPlan:
        retry = (
            RobustRetryPolicy(maximum_retries=2) if retry_policy is None else retry_policy
        )
        return ProductionRunPlan(
            self,
            retry,
            step_size=step_size,
            end_time=end_time,
            maximum_steps=maximum_steps,
            checkpoint_interval=checkpoint_interval,
            segment_steps=segment_steps,
        )


__all__ = [
    "CoupledMultiphysicsEpochIdentity",
    "CoupledMultiphysicsFixedStepMethod",
    "CoupledMultiphysicsProductionCase",
]
