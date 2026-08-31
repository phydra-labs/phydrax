#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._distributed import (
    LatticeBoltzmannHaloSchedule,
    ShardedLatticeBoltzmannExecutionPlan,
)
from ._dynamics import PreparedLatticeBoltzmannDynamics
from ._execution import (
    LatticeBoltzmannRealizationResult,
    ReferenceLatticeBoltzmannExecutionPlan,
)


class PreparedDistributedLatticeBoltzmannDynamics(StrictModule, NonTrainableState):
    """Actual prepared hydrodynamics executed over one fixed spatial sharding.

    The population tensor retains global JAX semantics under ``NamedSharding``.
    Direction-selective halo routes certify the fixed decomposition and XLA lowers
    the global pull operations to the required collectives; Q is never partitioned.
    """

    dynamics: PreparedLatticeBoltzmannDynamics
    halo: LatticeBoltzmannHaloSchedule
    reference: ReferenceLatticeBoltzmannExecutionPlan
    execution: ShardedLatticeBoltzmannExecutionPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedLatticeBoltzmannDynamics,
        halo: LatticeBoltzmannHaloSchedule,
        /,
    ):
        if not isinstance(dynamics, PreparedLatticeBoltzmannDynamics):
            raise TypeError("dynamics must be PreparedLatticeBoltzmannDynamics.")
        if not isinstance(halo, LatticeBoltzmannHaloSchedule):
            raise TypeError("halo must be LatticeBoltzmannHaloSchedule.")
        if (
            halo.velocity_set.lattice_id
            != dynamics.discretization.velocity_set.lattice_id
        ):
            raise ValueError("Distributed halo and dynamics velocity sets do not match.")
        if halo.schedule.global_shape != dynamics.discretization.grid.shape:
            raise ValueError("Distributed halo global shape must match the LBM grid.")
        expected_routes = dynamics.discretization.velocity_set.population_count
        if len(halo.routes) != expected_routes:
            raise ValueError("Distributed halo must certify every population direction.")
        if any(route.direction_index != index for index, route in enumerate(halo.routes)):
            raise ValueError(
                "Distributed halo directions must be canonical and complete."
            )
        reference = ReferenceLatticeBoltzmannExecutionPlan.from_dynamics(dynamics)
        execution = ShardedLatticeBoltzmannExecutionPlan(reference, halo)
        self.dynamics = dynamics
        self.halo = halo
        self.reference = reference
        self.execution = execution
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-distributed-lattice-boltzmann-dynamics",
                "dynamics": dynamics.prepared_id,
                "halo": halo.schedule_id,
                "execution": execution.plan_id,
                "program_manifest": dynamics.program_manifest.manifest_id,
            }
        )

    def shard(self, populations: Array, /) -> Array:
        return self.execution.shard(populations)

    def realize(
        self,
        initial_populations: Array,
        /,
        *,
        step_count: int,
        args: Any,
        t0: Any = 0.0,
        verify_equivalence: bool = True,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> LatticeBoltzmannRealizationResult:
        return self.execution.realize(
            initial_populations,
            step_count=step_count,
            step_size=self.dynamics.scaling.time_step,
            args=args,
            t0=t0,
            rtol=rtol,
            atol=atol,
            verify_equivalence=verify_equivalence,
        )


__all__ = ["PreparedDistributedLatticeBoltzmannDynamics"]
