#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._compressible_contracts import (
    CompressibleKineticPopulationState,
    CompressibleKineticStepResult,
)
from ._compressible_execution import (
    IntegerLatticeTransportEvidence,
    IntegerLatticeTransportPlan,
)


if TYPE_CHECKING:
    from ..lattice_boltzmann._checkpoint import KineticCheckpointPlan
from ._positive_kinetic import PositiveCompressibleKineticPlan
from ._quasi_equilibrium import FullRangeQuasiEquilibriumPlan, QuasiEquilibriumEvidence


class CompressibleKineticRuntimeState(StrictModule):
    kinetic: CompressibleKineticPopulationState
    time: Array
    step_index: Array
    parity: Array


class CompressibleKineticRuntimeEvidence(StrictModule):
    collision: CompressibleKineticStepResult
    quasi_equilibrium: QuasiEquilibriumEvidence | None
    transport: IntegerLatticeTransportEvidence
    successful: Array
    runtime_id: str = eqx.field(static=True)


class CompressibleKineticRuntimeResult(StrictModule):
    candidate: CompressibleKineticRuntimeState
    accepted: CompressibleKineticRuntimeState
    evidence: CompressibleKineticRuntimeEvidence
    successful: Array


class CompressibleKineticRuntimePlan(StrictModule, NonTrainableState):
    model: PositiveCompressibleKineticPlan
    transport: IntegerLatticeTransportPlan
    quasi_equilibrium: FullRangeQuasiEquilibriumPlan | None
    time_step: float = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: PositiveCompressibleKineticPlan,
        transport: IntegerLatticeTransportPlan,
        /,
        *,
        time_step: float = 1.0,
        quasi_equilibrium: FullRangeQuasiEquilibriumPlan | None = None,
        precision_policy_id: str = "homogeneous-float64",
    ):
        if not isinstance(model, PositiveCompressibleKineticPlan):
            raise TypeError("model must be a PositiveCompressibleKineticPlan.")
        if not isinstance(transport, IntegerLatticeTransportPlan):
            raise TypeError("transport must be an IntegerLatticeTransportPlan.")
        if model.rule.rule_id != transport.rule.rule_id:
            raise ValueError("Runtime model and transport velocity rules differ.")
        if quasi_equilibrium is not None and (
            not isinstance(quasi_equilibrium, FullRangeQuasiEquilibriumPlan)
            or quasi_equilibrium.model.model_id != model.model_id
        ):
            raise ValueError("quasi_equilibrium must be prepared for this model.")
        step = float(time_step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("time_step must be finite and positive.")
        if not precision_policy_id:
            raise ValueError("precision_policy_id must be non-empty.")
        self.model = model
        self.transport = transport
        self.quasi_equilibrium = quasi_equilibrium
        self.time_step = step
        self.precision_policy_id = str(precision_policy_id)
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "compressible-kinetic-runtime",
                "model": model.model_id,
                "transport": transport.transport_id,
                "quasi_equilibrium": None
                if quasi_equilibrium is None
                else quasi_equilibrium.plan_id,
                "time_step": step,
                "precision": precision_policy_id,
            }
        )

    def initialize(
        self,
        density: ArrayLike,
        velocity: ArrayLike,
        temperature: ArrayLike,
        /,
    ) -> CompressibleKineticRuntimeState:
        kinetic = self.model.initialize(density, velocity, temperature)
        return CompressibleKineticRuntimeState(
            kinetic,
            jnp.asarray(0.0, dtype=kinetic.populations[0].dtype),
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def advance(
        self,
        state: CompressibleKineticRuntimeState,
        relaxation_rate: ArrayLike,
        /,
    ) -> CompressibleKineticRuntimeResult:
        if self.quasi_equilibrium is None:
            collision = self.model.collide(state.kinetic, relaxation_rate)
            quasi_evidence = None
        else:
            collision, quasi_evidence = self.quasi_equilibrium.collide(
                state.kinetic, relaxation_rate
            )
        streamed, transport_evidence = self.transport.stream(
            collision.accepted, parity=state.parity
        )
        successful = jnp.all(collision.successful) & transport_evidence.successful
        candidate = CompressibleKineticRuntimeState(
            streamed,
            state.time + self.time_step,
            state.step_index + 1,
            transport_evidence.parity,
        )
        accepted = CompressibleKineticRuntimeState(
            CompressibleKineticPopulationState(
                tuple(
                    jnp.where(successful, new, old)
                    for new, old in zip(
                        streamed.populations, state.kinetic.populations, strict=True
                    )
                ),
                jnp.where(
                    successful, streamed.equilibrium_dual, state.kinetic.equilibrium_dual
                ),
                jnp.where(successful, streamed.stabilizer, state.kinetic.stabilizer),
                jnp.where(
                    successful, streamed.frame_velocity, state.kinetic.frame_velocity
                ),
                jnp.where(
                    successful,
                    streamed.frame_temperature_scale,
                    state.kinetic.frame_temperature_scale,
                ),
                state.kinetic.layout,
            ),
            jnp.where(successful, candidate.time, state.time),
            jnp.where(successful, candidate.step_index, state.step_index),
            jnp.where(successful, candidate.parity, state.parity),
        )
        evidence = CompressibleKineticRuntimeEvidence(
            collision=collision,
            quasi_equilibrium=quasi_evidence,
            transport=transport_evidence,
            successful=successful,
            runtime_id=self.runtime_id,
        )
        return CompressibleKineticRuntimeResult(candidate, accepted, evidence, successful)

    def checkpoint_plan(self) -> KineticCheckpointPlan:
        from ..lattice_boltzmann._checkpoint import KineticCheckpointPlan
        from ..lattice_boltzmann._program import (
            coupled_population_manifest,
            KineticFieldRole,
            KineticFieldSpec,
            KineticProgramManifest,
        )

        names = tuple(field.name for field in self.model.layout.fields)
        channels = tuple(
            ("mass", "momentum", "energy") if field.role == "particle" else ("energy",)
            for field in self.model.layout.fields
        )
        base = coupled_population_manifest(
            "compressible-kinetic",
            self.model.rule.rule_id,
            self.precision_policy_id,
            self.model.rule.population_count,
            self.model.rule.dimension,
            names,
            channels,
        )
        extras = (
            KineticFieldSpec(
                "equilibrium_dual",
                KineticFieldRole.COUPLED_STATE,
                (self.model.rule.dual_dimension,),
                initialized=True,
                checkpoint_required=True,
            ),
            KineticFieldSpec(
                "stabilizer",
                KineticFieldRole.COUPLED_STATE,
                initialized=True,
                checkpoint_required=True,
            ),
            KineticFieldSpec(
                "frame_velocity",
                KineticFieldRole.COUPLED_STATE,
                (self.model.rule.dimension,),
                initialized=True,
                checkpoint_required=True,
            ),
            KineticFieldSpec(
                "frame_temperature_scale",
                KineticFieldRole.COUPLED_STATE,
                initialized=True,
                checkpoint_required=True,
            ),
        )
        manifest = KineticProgramManifest(
            base.program_kind,
            base.lattice_id,
            base.precision_policy_id,
            (*base.fields, *extras),
            base.stages,
        )
        return KineticCheckpointPlan(
            self.runtime_id,
            manifest,
            topology_id=self.transport.transport_id,
            execution_id=self.transport.storage.plan_id,
        )


__all__ = [
    "CompressibleKineticRuntimeEvidence",
    "CompressibleKineticRuntimePlan",
    "CompressibleKineticRuntimeResult",
    "CompressibleKineticRuntimeState",
]
