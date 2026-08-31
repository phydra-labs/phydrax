#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import DiscretizationBundle, DiscretizationRecord
from ..discretization._core import DiscretizationKey, DiscretizationRole
from ..discretization.particle import (
    conversion_state_admissible,
    ParticleConversionState,
    ParticleConversionStateGeometry,
    ParticleInternalBatchPlan,
    PreparedParticleInternalBatch,
)
from ._particle_reaction import (
    EvaporationPhaseChangePlan,
    ParticlePhaseChangeEvaluation,
    ParticleReactionEvaluation,
    ParticleReactionNetworkPlan,
)
from ._particle_thermochemistry import (
    evaluate_particle_transport,
    ParticleThermochemicalMaterialBundle,
    ParticleTransportEvaluation,
)


class ParticleConversionRejectionReason(IntFlag):
    NONE = 0
    TRANSPORT = 1 << 0
    BALANCE = 1 << 6
    REACTION = 1 << 1
    PHASE_CHANGE = 1 << 2
    NONFINITE = 1 << 3
    ADMISSIBILITY = 1 << 4
    SOLVER = 1 << 5


class ParticleConversionProblemIR(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    materials: tuple[ParticleThermochemicalMaterialBundle, ...]
    reactions: tuple[ParticleReactionNetworkPlan | None, ...]
    phase_changes: tuple[EvaporationPhaseChangePlan | None, ...]
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        materials,
        /,
        *,
        reactions=None,
        phase_changes=None,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        material_values = tuple(materials)
        if not name_:
            raise ValueError("Particle conversion problem name must be nonempty.")
        if not material_values or any(
            not isinstance(value, ParticleThermochemicalMaterialBundle)
            for value in material_values
        ):
            raise TypeError("materials must contain thermochemical material bundles.")
        reaction_values = (
            (None,) * len(material_values) if reactions is None else tuple(reactions)
        )
        phase_values = (
            (None,) * len(material_values)
            if phase_changes is None
            else tuple(phase_changes)
        )
        if len(reaction_values) != len(material_values) or len(phase_values) != len(
            material_values
        ):
            raise ValueError(
                "Material, reaction, and phase-change batch counts must match."
            )
        for material, reaction, phase in zip(
            material_values, reaction_values, phase_values, strict=True
        ):
            schema_id = material.thermodynamics.schema.schema_id
            if reaction is not None and (
                not isinstance(reaction, ParticleReactionNetworkPlan)
                or reaction.schema.schema_id != schema_id
            ):
                raise ValueError("Reaction network does not match material schema.")
            if phase is not None and (
                not isinstance(phase, EvaporationPhaseChangePlan)
                or phase.schema.schema_id != schema_id
            ):
                raise ValueError("Phase-change plan does not match material schema.")
        generated = canonical_fingerprint(
            {
                "kind": "particle-conversion-problem-ir",
                "name": name_,
                "materials": [value.bundle_id for value in material_values],
                "reactions": [
                    None if value is None else value.network_id
                    for value in reaction_values
                ],
                "phase_changes": [
                    None if value is None else value.plan_id for value in phase_values
                ],
            }
        )
        self.name = name_
        self.materials = material_values
        self.reactions = reaction_values
        self.phase_changes = phase_values
        self.problem_id = generated if problem_id is None else str(problem_id)
        if not self.problem_id:
            raise ValueError("problem_id must be nonempty.")


class ParticleConversionBatchEvaluation(StrictModule):
    transport: ParticleTransportEvaluation
    reaction: ParticleReactionEvaluation | None
    phase_change: ParticlePhaseChangeEvaluation | None
    internal_energy_rate: Array
    species_amount_rate: Array
    explicit_step_restriction: Array
    successful: Array
    batch_id: str = eqx.field(static=True)


class ParticleConversionEvaluation(StrictModule):
    batches: tuple[ParticleConversionBatchEvaluation, ...]
    rejection_reasons: Array
    successful: Array
    dynamics_id: str = eqx.field(static=True)


class PreparedParticleConversionDynamics(StrictModule, NonTrainableState):
    problem: ParticleConversionProblemIR
    batches: tuple[PreparedParticleInternalBatch, ...]
    dynamics_id: str = eqx.field(static=True)

    def __init__(self, problem, batches, /):
        if not isinstance(problem, ParticleConversionProblemIR):
            raise TypeError("problem must be a ParticleConversionProblemIR.")
        batch_values = tuple(batches)
        if len(batch_values) != len(problem.materials) or any(
            not isinstance(value, PreparedParticleInternalBatch) for value in batch_values
        ):
            raise ValueError(
                "Prepared conversion batches do not match problem materials."
            )
        for batch, material in zip(batch_values, problem.materials, strict=True):
            if batch.species_count != material.thermodynamics.schema.species_count:
                raise ValueError(
                    "Prepared batch species count does not match material schema."
                )
        self.problem = problem
        self.batches = batch_values
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "prepared-particle-conversion-dynamics",
                "problem": problem.problem_id,
                "batches": [value.prepared_id for value in batch_values],
            }
        )

    @property
    def state_geometry(self):
        return ParticleConversionStateGeometry(self.dynamics_id)

    def evaluate(
        self,
        state: ParticleConversionState,
        boundaries,
        /,
    ) -> ParticleConversionEvaluation:
        boundary_values = tuple(boundaries)
        if len(state.batches) != len(self.batches) or len(boundary_values) != len(
            self.batches
        ):
            raise ValueError(
                "Conversion state/boundary batch counts do not match dynamics."
            )
        evaluations = []
        reasons = jnp.zeros((), dtype=jnp.int32)
        for prepared, batch_state, material, reaction, phase, boundary in zip(
            self.batches,
            state.batches,
            self.problem.materials,
            self.problem.reactions,
            self.problem.phase_changes,
            boundary_values,
            strict=True,
        ):
            transport = evaluate_particle_transport(
                prepared, batch_state, material, boundary
            )
            reaction_evaluation = (
                None
                if reaction is None
                else reaction.evaluate(prepared, batch_state, material.thermodynamics)
            )
            metrics = prepared.mesh.metrics(batch_state.outer_scale)
            phase_evaluation = (
                None
                if phase is None
                else phase.evaluate(
                    prepared,
                    batch_state,
                    transport.thermodynamic_state,
                    metrics,
                )
            )
            energy_rate = transport.internal_energy_rate
            species_rate = transport.species_amount_rate
            restriction = transport.explicit_step_restriction
            successful = transport.successful
            if reaction_evaluation is not None:
                energy_rate = energy_rate + reaction_evaluation.internal_energy_rate
                species_rate = species_rate + reaction_evaluation.species_amount_rate
                restriction = jnp.minimum(
                    restriction, reaction_evaluation.explicit_step_restriction
                )
                successful = successful & reaction_evaluation.successful
                reasons = reasons | jnp.where(
                    ~reaction_evaluation.successful,
                    int(ParticleConversionRejectionReason.REACTION),
                    0,
                ).astype(jnp.int32)
            if phase_evaluation is not None:
                energy_rate = energy_rate + phase_evaluation.internal_energy_rate
                species_rate = species_rate + phase_evaluation.species_amount_rate
                restriction = jnp.minimum(
                    restriction, phase_evaluation.explicit_step_restriction
                )
                successful = successful & phase_evaluation.successful
                reasons = reasons | jnp.where(
                    ~phase_evaluation.successful,
                    int(ParticleConversionRejectionReason.PHASE_CHANGE),
                    0,
                ).astype(jnp.int32)
            reasons = reasons | jnp.where(
                ~transport.successful,
                int(ParticleConversionRejectionReason.TRANSPORT),
                0,
            ).astype(jnp.int32)
            successful = (
                successful
                & jnp.all(jnp.isfinite(energy_rate))
                & jnp.all(jnp.isfinite(species_rate))
            )
            evaluations.append(
                ParticleConversionBatchEvaluation(
                    transport,
                    reaction_evaluation,
                    phase_evaluation,
                    energy_rate,
                    species_rate,
                    restriction,
                    successful,
                    prepared.prepared_id,
                )
            )
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value.internal_energy_rate))
                    & jnp.all(jnp.isfinite(value.species_amount_rate))
                    & ~jnp.isnan(value.explicit_step_restriction)
                    for value in evaluations
                )
            )
        )
        admissible = conversion_state_admissible(state)
        reasons = reasons | jnp.where(
            ~finite,
            int(ParticleConversionRejectionReason.NONFINITE),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            ~admissible,
            int(ParticleConversionRejectionReason.ADMISSIBILITY),
            0,
        ).astype(jnp.int32)
        successful = (reasons == 0) & jnp.all(
            jnp.stack(tuple(value.successful for value in evaluations))
        )
        return ParticleConversionEvaluation(
            tuple(evaluations), reasons, successful, self.dynamics_id
        )


class CompiledParticleConversionProblem(StrictModule, NonTrainableState):
    problem: ParticleConversionProblemIR
    dynamics: PreparedParticleConversionDynamics
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(self, problem, dynamics, bundle, /):
        self.problem = problem
        self.dynamics = dynamics
        self.discretization_bundle = bundle
        self.compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-particle-conversion-problem",
                "problem": problem.problem_id,
                "dynamics": dynamics.dynamics_id,
                "bundle": bundle.bundle_id,
            }
        )

    def initialize_state(self, batches, /):
        from ..discretization.particle import initialize_particle_conversion_state

        state = initialize_particle_conversion_state(
            batches, state_id=self.dynamics.dynamics_id
        )
        if len(state.batches) != len(self.dynamics.batches):
            raise ValueError("Initial conversion state batch count is invalid.")
        for value, prepared in zip(state.batches, self.dynamics.batches, strict=True):
            if value.batch_id != prepared.prepared_id:
                raise ValueError("Initial conversion state batch ID is invalid.")
        return state


def compile_particle_conversion_problem(
    problem: ParticleConversionProblemIR,
    particles,
    batch_plans,
    /,
) -> CompiledParticleConversionProblem:
    if not isinstance(problem, ParticleConversionProblemIR):
        raise TypeError("problem must be a ParticleConversionProblemIR.")
    plans = tuple(batch_plans)
    if len(plans) != len(problem.materials) or any(
        not isinstance(value, ParticleInternalBatchPlan) for value in plans
    ):
        raise ValueError("Internal batch plans do not match conversion problem.")
    batches = tuple(value.prepare(particles) for value in plans)
    dynamics = PreparedParticleConversionDynamics(problem, batches)
    particle_record = DiscretizationRecord(
        particles.key,
        "particle-support",
        particles.prepared_id,
        numeric_version=particles.numeric_version,
        resource_evidence_id=particles.resource_evidence_id,
    )
    batch_records = tuple(
        DiscretizationRecord(
            DiscretizationKey(
                f"particle-internal-batch-{index}",
                DiscretizationRole.AUXILIARY,
                domain_labels=("material_point", "particle_internal"),
            ),
            "particle-internal-shells",
            value.prepared_id,
            dependency_key_ids=(particles.key.key_id,),
        )
        for index, value in enumerate(batches)
    )
    bundle = DiscretizationBundle((particle_record,) + batch_records)
    return CompiledParticleConversionProblem(problem, dynamics, bundle)


__all__ = [
    "CompiledParticleConversionProblem",
    "ParticleConversionBatchEvaluation",
    "ParticleConversionEvaluation",
    "ParticleConversionProblemIR",
    "ParticleConversionRejectionReason",
    "PreparedParticleConversionDynamics",
    "compile_particle_conversion_problem",
]
