#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..lifecycle import (
    AnalysisPlan,
    ExecutionPlan,
    ModelManifest,
    NumericRevision,
    ResultManifest,
    RunRecord,
)
from ._committee import (
    AcquisitionPlan,
    AcquisitionRecord,
    AtomisticUncertaintyEvidence,
    CommitteeAtomisticPotential,
    CommitteeReductionPolicy,
)
from ._frame import AtomisticFrame, AtomisticSiteDomain
from ._graph import AtomisticGraphExecutionPlan
from ._hybrid import AbstractExternalAtomisticProvider, ExternalAtomisticEvaluation
from ._potential import AbstractAtomisticPotential
from ._potential_program import AtomisticPotentialProgram, LearnedGraphPotentialTerm
from ._qualification import AtomisticDynamicsQualificationResult
from ._system import PreparedAtomisticSystem
from ._training import (
    AtomisticTrainingPolicy,
    AtomisticTrainingProblem,
    AtomisticTrainingResult,
    fit_atomistic_potential,
)
from ._types import AtomisticBatch


class AtomisticLabelRecord(StrictModule, NonTrainableState):
    frame: AtomisticFrame
    energy: Array
    forces: Array
    stress: Array | None
    successful: Array
    split: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    acquisition_id: str = eqx.field(static=True)
    configuration_id: str = eqx.field(static=True)
    label_id: str = eqx.field(static=True)


class AtomisticLabelSet(StrictModule, NonTrainableState):
    records: tuple[AtomisticLabelRecord, ...]
    revision: NumericRevision
    system_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    label_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        records: Sequence[AtomisticLabelRecord],
        /,
        *,
        parent: "AtomisticLabelSet | None" = None,
    ):
        values = tuple(records)
        if not values or any(
            not isinstance(value, AtomisticLabelRecord) for value in values
        ):
            raise TypeError("AtomisticLabelSet requires non-empty label records.")
        if any(not bool(value.successful) for value in values):
            raise ValueError("Only successful labels may enter a label set.")
        first = values[0].frame
        if any(
            value.frame.system_id != first.system_id
            or value.frame.topology_id != first.topology_id
            or value.frame.unit_system_id != first.unit_system_id
            for value in values[1:]
        ):
            raise ValueError("Label records must share system, topology, and units.")
        keys = tuple((value.configuration_id, value.provider_id) for value in values)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "Label records contain duplicate configuration/provider pairs."
            )
        digest = canonical_fingerprint(
            {
                "kind": "atomistic-label-content",
                "labels": [value.label_id for value in values],
            }
        )
        parent_digest = None if parent is None else parent.revision.content_digest
        self.records = values
        self.revision = NumericRevision(
            digest,
            label=f"atomistic-labels-{len(values)}",
            parent_digest=parent_digest,
            metadata={"system_id": first.system_id, "record_count": str(len(values))},
        )
        self.system_id = first.system_id
        self.topology_id = first.topology_id
        self.unit_system_id = first.unit_system_id
        self.label_set_id = canonical_fingerprint(
            {
                "kind": "atomistic-label-set",
                "revision": self.revision.revision_id,
                "system": first.system_id,
                "topology": first.topology_id,
                "units": first.unit_system_id,
            }
        )

    def append(self, records: Sequence[AtomisticLabelRecord], /) -> "AtomisticLabelSet":
        additions = tuple(records)
        if not additions:
            return self
        return AtomisticLabelSet(self.records + additions, parent=self)

    def training_problem(
        self,
        system: PreparedAtomisticSystem,
        graph_execution: AtomisticGraphExecutionPlan,
        /,
    ) -> AtomisticTrainingProblem:
        if system.prepared_id != self.system_id:
            raise ValueError("Label set belongs to another prepared atomistic system.")
        training = tuple(value for value in self.records if value.split == "train")
        validation = tuple(value for value in self.records if value.split == "validation")
        if not training:
            raise ValueError("A label set requires at least one training record.")

        def batch(records: tuple[AtomisticLabelRecord, ...]) -> AtomisticBatch:
            count = len(records)
            plan = system.plan
            cells = None
            periodic = None
            if plan.cell is not None:
                cells = jnp.stack(
                    tuple(
                        plan.cell.vectors
                        if record.frame.cell_vectors is None
                        else record.frame.cell_vectors
                        for record in records
                    )
                )
                periodic = jnp.broadcast_to(plan.cell.periodic_mask, (count, 3))
            return AtomisticBatch(
                jnp.broadcast_to(plan.atomic_numbers, (count, plan.particle_ids.size)),
                jnp.stack(tuple(record.frame.positions for record in records)),
                jnp.broadcast_to(plan.masses, (count, plan.particle_ids.size)),
                plan.units.scale,
                particle_ids=jnp.broadcast_to(
                    plan.particle_ids, (count, plan.particle_ids.size)
                ),
                atom_mask=jnp.broadcast_to(
                    plan.active_mask, (count, plan.particle_ids.size)
                ),
                cells=cells,
                periodic_axes=periodic,
                structure_ids=tuple(record.configuration_id for record in records),
            )

        training_batch = batch(training)
        validation_batch = None if not validation else batch(validation)
        return AtomisticTrainingProblem(
            training_batch,
            graph_execution,
            training_energy=jnp.stack(tuple(value.energy for value in training)),
            training_forces=jnp.stack(tuple(value.forces for value in training)),
            validation_batch=validation_batch,
            validation_energy=(
                None
                if not validation
                else jnp.stack(tuple(value.energy for value in validation))
            ),
            validation_forces=(
                None
                if not validation
                else jnp.stack(tuple(value.forces for value in validation))
            ),
        )


class AtomisticLearningCampaignPlan(StrictModule, NonTrainableState):
    system: PreparedAtomisticSystem
    provider: AbstractExternalAtomisticProvider
    acquisition: AcquisitionPlan
    graph_execution: AtomisticGraphExecutionPlan
    runtime_graph_execution: AtomisticGraphExecutionPlan
    training: AtomisticTrainingPolicy
    committee_reduction: CommitteeReductionPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: PreparedAtomisticSystem,
        provider: AbstractExternalAtomisticProvider,
        acquisition: AcquisitionPlan,
        graph_execution: AtomisticGraphExecutionPlan,
        runtime_graph_execution: AtomisticGraphExecutionPlan,
        training: AtomisticTrainingPolicy,
        committee_reduction: CommitteeReductionPolicy,
        /,
    ):
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("system must be PreparedAtomisticSystem.")
        if not isinstance(provider, AbstractExternalAtomisticProvider):
            raise TypeError("provider must implement AbstractExternalAtomisticProvider.")
        if not isinstance(acquisition, AcquisitionPlan):
            raise TypeError("acquisition must be AcquisitionPlan.")
        if (
            not isinstance(graph_execution, AtomisticGraphExecutionPlan)
            or graph_execution.backend != "dense"
        ):
            raise TypeError("graph_execution must be a dense training graph plan.")
        if (
            not isinstance(runtime_graph_execution, AtomisticGraphExecutionPlan)
            or runtime_graph_execution.backend != "particle"
        ):
            raise TypeError("runtime_graph_execution must be a particle graph plan.")
        if not isinstance(training, AtomisticTrainingPolicy):
            raise TypeError("training must be AtomisticTrainingPolicy.")
        if not isinstance(committee_reduction, CommitteeReductionPolicy):
            raise TypeError("committee_reduction must be CommitteeReductionPolicy.")
        self.system = system
        self.provider = provider
        self.acquisition = acquisition
        self.graph_execution = graph_execution
        self.runtime_graph_execution = runtime_graph_execution
        self.training = training
        self.committee_reduction = committee_reduction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-learning-campaign",
                "system": system.prepared_id,
                "provider": provider.provider_id,
                "acquisition": acquisition.plan_id,
                "graph_execution": graph_execution.plan_id,
                "runtime_graph_execution": runtime_graph_execution.plan_id,
                "training": training.policy_id,
                "committee": committee_reduction.policy_id,
            }
        )


class AtomisticLearningCampaignState(StrictModule, NonTrainableState):
    labels: AtomisticLabelSet
    committee: CommitteeAtomisticPotential | None
    round_index: int = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        labels: AtomisticLabelSet,
        /,
        *,
        committee: CommitteeAtomisticPotential | None = None,
        round_index: int = 0,
    ):
        if not isinstance(labels, AtomisticLabelSet):
            raise TypeError("labels must be AtomisticLabelSet.")
        if committee is not None and not isinstance(
            committee, CommitteeAtomisticPotential
        ):
            raise TypeError("committee must be CommitteeAtomisticPotential or None.")
        index = int(round_index)
        if index < 0:
            raise ValueError("round_index must be nonnegative.")
        self.labels = labels
        self.committee = committee
        self.round_index = index
        self.state_id = canonical_fingerprint(
            {
                "kind": "atomistic-learning-campaign-state",
                "labels": labels.label_set_id,
                "committee": None if committee is None else committee.committee_id,
                "round": index,
            }
        )


class AtomisticCampaignLifecycle(StrictModule, NonTrainableState):
    analysis: AnalysisPlan
    execution: ExecutionPlan
    run: RunRecord
    models: tuple[ModelManifest, ...]
    result: ResultManifest


class AtomisticCampaignRoundResult(StrictModule, NonTrainableState):
    state: AtomisticLearningCampaignState
    acquisitions: tuple[AcquisitionRecord, ...]
    labels: tuple[AtomisticLabelRecord, ...]
    training_results: tuple[AtomisticTrainingResult, ...]
    qualification: AtomisticDynamicsQualificationResult | None
    lifecycle: AtomisticCampaignLifecycle
    promoted: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def label_atomistic_acquisitions(
    system: PreparedAtomisticSystem,
    provider: AbstractExternalAtomisticProvider,
    acquisitions: Sequence[AcquisitionRecord],
    /,
    *,
    split: str = "train",
) -> tuple[AtomisticLabelRecord, ...]:
    if split not in ("train", "validation"):
        raise ValueError("split must be 'train' or 'validation'.")
    records = []
    for acquisition in acquisitions:
        frame = acquisition.frame
        if frame.system_id != system.prepared_id:
            raise ValueError("Acquisition frame belongs to another system.")
        if frame.coordinate_domain is not AtomisticSiteDomain.DOF_ATOMS:
            raise ValueError("Label providers require physical degree-of-freedom atoms.")
        evaluation: ExternalAtomisticEvaluation = provider.evaluate(
            system, frame.positions, frame.cell_vectors
        )
        configuration_id = canonical_fingerprint(
            {
                "kind": "atomistic-label-configuration",
                "system": frame.system_id,
                "topology": frame.topology_id,
                "units": frame.unit_system_id,
                "arrays": array_tree_fingerprint(
                    {
                        "stable_ids": frame.stable_ids,
                        "positions": frame.positions,
                        "cell": frame.cell_vectors,
                    }
                ),
            }
        )
        label_id = canonical_fingerprint(
            {
                "kind": "atomistic-label",
                "configuration": configuration_id,
                "provider": evaluation.provider_id,
                "acquisition": acquisition.plan_id,
                "split": split,
                "arrays": array_tree_fingerprint(
                    {
                        "energy": evaluation.energy,
                        "forces": evaluation.forces,
                        "stress": evaluation.stress,
                    }
                ),
            }
        )
        records.append(
            AtomisticLabelRecord(
                frame=frame,
                energy=evaluation.energy,
                forces=evaluation.forces,
                stress=evaluation.stress,
                successful=evaluation.successful,
                split=split,
                provider_id=evaluation.provider_id,
                acquisition_id=acquisition.plan_id,
                configuration_id=configuration_id,
                label_id=label_id,
            )
        )
    return tuple(records)


def _campaign_lifecycle(
    plan: AtomisticLearningCampaignPlan,
    state: AtomisticLearningCampaignState,
    training_results: tuple[AtomisticTrainingResult, ...],
    qualification: AtomisticDynamicsQualificationResult | None,
    result_id: str,
    successful: bool,
    /,
) -> AtomisticCampaignLifecycle:
    analysis = AnalysisPlan(
        plan.plan_id,
        plan.provider.provider_id,
        plan.graph_execution.plan_id,
        (plan.system.prepared_id,),
        material_plan_id=plan.system.plan.system_id,
        constraint_ids=(plan.system.topology.topology_id,),
        capability_ids=(
            plan.acquisition.scoring.policy_id,
            plan.committee_reduction.policy_id,
        ),
    )
    execution_id = canonical_fingerprint(
        {
            "kind": "atomistic-campaign-execution",
            "plan": plan.plan_id,
            "round": state.round_index,
        }
    )
    execution = ExecutionPlan(
        execution_id,
        "jax-host",
        plan.system.plan.coordinate_dtype,
        plan.training.policy_id,
        reduction_policy_id=plan.committee_reduction.policy_id,
    )
    manifests = tuple(
        ModelManifest(
            result.best_potential.potential_id,
            analysis.analysis_plan_id,
            state.labels.revision.revision_id,
            {"parameter_state": result.best_potential.parameter_state_id},
            unit_contract_id=plan.system.plan.units.unit_system_id,
            association_ids=(result.problem_id, result.policy_id),
        )
        for result in training_results
    )
    run_id = canonical_fingerprint(
        {
            "kind": "atomistic-campaign-run",
            "plan": plan.plan_id,
            "state": state.state_id,
            "result": result_id,
        }
    )
    evidence_ids = tuple(result.result_id for result in training_results) + (
        () if qualification is None else (qualification.result_id,)
    )
    result_manifest = ResultManifest(
        result_id,
        run_id,
        {"campaign": "1"},
        {"campaign": result_id},
        evidence_ids=evidence_ids,
    )
    run = RunRecord(
        run_id,
        analysis.analysis_plan_id,
        state.labels.revision.revision_id,
        execution.execution_plan_id,
        "completed" if successful else "failed",
        result_ids=(result_manifest.manifest_id,),
        diagnostic_ids=evidence_ids,
    )
    return AtomisticCampaignLifecycle(
        analysis=analysis,
        execution=execution,
        run=run,
        models=manifests,
        result=result_manifest,
    )


def run_atomistic_campaign_round(
    plan: AtomisticLearningCampaignPlan,
    state: AtomisticLearningCampaignState,
    frames: Sequence[AtomisticFrame],
    uncertainty: Sequence[AtomisticUncertaintyEvidence],
    initial_potentials: Sequence[AbstractAtomisticPotential],
    keys: Sequence[Key[Array, ""]],
    qualify: Callable[
        [CommitteeAtomisticPotential], AtomisticDynamicsQualificationResult
    ],
    /,
    *,
    descriptors: Array | None = None,
) -> AtomisticCampaignRoundResult:
    """Execute one bounded label, retrain, qualify, and promotion transaction."""

    if not isinstance(plan, AtomisticLearningCampaignPlan) or not isinstance(
        state, AtomisticLearningCampaignState
    ):
        raise TypeError("plan and state must satisfy campaign contracts.")
    if state.labels.system_id != plan.system.prepared_id:
        raise ValueError("Campaign state belongs to another plan system.")
    potentials = tuple(initial_potentials)
    key_values = tuple(keys)
    if len(potentials) < 2 or len(potentials) != len(key_values):
        raise ValueError(
            "Campaign retraining requires aligned potential and key committees."
        )
    acquisitions = plan.acquisition.select(frames, uncertainty, descriptors=descriptors)
    if not acquisitions:
        result_id = canonical_fingerprint(
            {
                "kind": "atomistic-campaign-round",
                "plan": plan.plan_id,
                "state": state.state_id,
                "acquisitions": [],
            }
        )
        lifecycle = _campaign_lifecycle(plan, state, (), None, result_id, True)
        return AtomisticCampaignRoundResult(
            state=state,
            acquisitions=(),
            labels=(),
            training_results=(),
            qualification=None,
            lifecycle=lifecycle,
            promoted=jnp.asarray(False),
            successful=jnp.asarray(True),
            plan_id=plan.plan_id,
            result_id=result_id,
        )
    labels = label_atomistic_acquisitions(
        plan.system, plan.provider, acquisitions, split="train"
    )
    successful_labels = tuple(value for value in labels if bool(value.successful))
    if len(successful_labels) != len(labels):
        result_id = canonical_fingerprint(
            {
                "kind": "atomistic-campaign-round",
                "plan": plan.plan_id,
                "state": state.state_id,
                "acquisitions": [value.plan_id for value in acquisitions],
                "provider_failed": True,
            }
        )
        lifecycle = _campaign_lifecycle(plan, state, (), None, result_id, False)
        return AtomisticCampaignRoundResult(
            state=state,
            acquisitions=acquisitions,
            labels=labels,
            training_results=(),
            qualification=None,
            lifecycle=lifecycle,
            promoted=jnp.asarray(False),
            successful=jnp.asarray(False),
            plan_id=plan.plan_id,
            result_id=result_id,
        )
    updated_labels = state.labels.append(successful_labels)
    problem = updated_labels.training_problem(plan.system, plan.graph_execution)
    results = tuple(
        fit_atomistic_potential(potential, problem, plan.training, key=key)
        for potential, key in zip(potentials, key_values, strict=True)
    )
    training_successful = jnp.all(
        jnp.stack(tuple(result.successful for result in results))
    )
    if not bool(training_successful):
        next_state = AtomisticLearningCampaignState(
            updated_labels,
            committee=state.committee,
            round_index=state.round_index + 1,
        )
        result_id = canonical_fingerprint(
            {
                "kind": "atomistic-campaign-round",
                "plan": plan.plan_id,
                "state": state.state_id,
                "next_state": next_state.state_id,
                "labels": [value.label_id for value in labels],
                "training": [value.result_id for value in results],
                "training_failed": True,
            }
        )
        lifecycle = _campaign_lifecycle(plan, next_state, results, None, result_id, False)
        return AtomisticCampaignRoundResult(
            state=next_state,
            acquisitions=acquisitions,
            labels=labels,
            training_results=results,
            qualification=None,
            lifecycle=lifecycle,
            promoted=jnp.asarray(False),
            successful=jnp.asarray(False),
            plan_id=plan.plan_id,
            result_id=result_id,
        )
    programs = tuple(
        AtomisticPotentialProgram(
            [LearnedGraphPotentialTerm(result.best_potential)]
        ).prepare(plan.system, graph_execution=plan.runtime_graph_execution)
        for result in results
    )
    candidate = CommitteeAtomisticPotential(programs, plan.committee_reduction)
    qualification = qualify(candidate)
    if not isinstance(qualification, AtomisticDynamicsQualificationResult):
        raise TypeError("qualify must return AtomisticDynamicsQualificationResult.")
    promoted = (
        training_successful
        & qualification.execution_successful
        & qualification.claims_satisfied
    )
    next_state = (
        AtomisticLearningCampaignState(
            updated_labels,
            committee=candidate,
            round_index=state.round_index + 1,
        )
        if bool(promoted)
        else AtomisticLearningCampaignState(
            updated_labels,
            committee=state.committee,
            round_index=state.round_index + 1,
        )
    )
    result_id = canonical_fingerprint(
        {
            "kind": "atomistic-campaign-round",
            "plan": plan.plan_id,
            "state": state.state_id,
            "next_state": next_state.state_id,
            "acquisitions": [value.plan_id for value in acquisitions],
            "labels": [value.label_id for value in labels],
            "training": [value.result_id for value in results],
            "qualification": qualification.result_id,
        }
    )
    round_successful = training_successful & qualification.execution_successful
    lifecycle = _campaign_lifecycle(
        plan,
        next_state,
        results,
        qualification,
        result_id,
        bool(round_successful),
    )
    return AtomisticCampaignRoundResult(
        state=next_state,
        acquisitions=acquisitions,
        labels=labels,
        training_results=results,
        qualification=qualification,
        lifecycle=lifecycle,
        promoted=promoted,
        successful=round_successful,
        plan_id=plan.plan_id,
        result_id=result_id,
    )


__all__ = [
    "AtomisticCampaignLifecycle",
    "AtomisticCampaignRoundResult",
    "AtomisticLabelRecord",
    "AtomisticLabelSet",
    "AtomisticLearningCampaignPlan",
    "AtomisticLearningCampaignState",
    "label_atomistic_acquisitions",
    "run_atomistic_campaign_round",
]
