#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Acquire, label, retrain, qualify and promote transactions of a committee.

`AtomisticLabelSet` is the append-only, revisioned set of authoritative
provider labels. It exposes its records as an immutable canonical
`AtomisticStructureSource` and lowers its train/validation records to the
streaming `AtomisticTrainingProblem`; campaign members are fitted by
`fit_atomistic_potential`. External provider labels are honest interop: they
do not satisfy a native-teacher requirement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._execution_plan import ExecutionPlan
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import NumericRevision, SemanticProvenance
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..lifecycle import (
    AnalysisPlan,
    ModelManifest,
    ResultManifest,
    RevisionLineage,
    RunRecord,
)
from ..typing import checked, parse, PRNGKey
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
from ._neighborhood_bundle import AtomisticNeighborhoodBundlePlan
from ._potential import (
    AbstractAtomisticModel,
    AbstractAtomisticPotential,
    atomistic_model_revision,
    AtomisticStressConvention,
)
from ._potential_program import AtomisticPotentialProgram, LearnedGraphPotentialTerm
from ._qualification import AtomisticDynamicsQualificationResult
from ._structure_store import AtomisticStructureSource
from ._system import PreparedAtomisticSystem
from ._training import (
    atomistic_training_loader,
    atomistic_training_source,
    AtomisticObjectiveWeighting,
    AtomisticTrainingFields,
    AtomisticTrainingPolicy,
    AtomisticTrainingProblem,
    AtomisticTrainingResult,
    fit_atomistic_potential,
)
from ._training_objectives import AbstractAtomisticObjective, AtomisticTrainingPrecision
from ._types import AtomisticBatch, AtomisticSpeciesKind
from ._units import AtomisticUnitSystem


AtomisticLabelSplit = Literal["train", "validation"]
"""Membership of one label record: training or lineage-disjoint validation."""

_LABEL_SLOTS = ("energy", "forces")
_PERIODIC_LABEL_SLOTS = ("energy", "forces", "stress")


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


@final
class AtomisticLabelSet(StrictModule, NonTrainableState):
    """Append-only label records with a canonical revision and parent lineage."""

    records: tuple[AtomisticLabelRecord, ...]
    revision: NumericRevision
    lineage: RevisionLineage
    system_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    units: AtomisticUnitSystem
    label_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        records: Sequence[AtomisticLabelRecord],
        /,
        *,
        parent: "AtomisticLabelSet | None" = None,
    ) -> None:
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
            or value.frame.units.unit_system_id != first.units.unit_system_id
            or not np.array_equal(
                np.asarray(value.frame.stable_ids), np.asarray(first.stable_ids)
            )
            for value in values[1:]
        ):
            raise ValueError(
                "Label records must share system, topology, units, and stable-ID order."
            )
        keys = tuple((value.configuration_id, value.provider_id) for value in values)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "Label records contain duplicate configuration/provider pairs."
            )
        if parent is not None:
            if not isinstance(parent, AtomisticLabelSet):
                raise TypeError("parent must be an AtomisticLabelSet or None.")
            parent_count = len(parent.records)
            if (
                parent.system_id != first.system_id
                or parent.topology_id != first.topology_id
                or parent.units.unit_system_id != first.units.unit_system_id
                or len(values) <= parent_count
                or tuple(value.label_id for value in values[:parent_count])
                != tuple(value.label_id for value in parent.records)
            ):
                raise ValueError(
                    "Label-set revisions must extend the exact current parent record."
                )
        revision = NumericRevision(
            SemanticProvenance(
                {
                    "kind": "atomistic-label-set",
                    "system_id": first.system_id,
                    "topology_id": first.topology_id,
                    "unit_system_id": first.units.unit_system_id,
                }
            ),
            {"labels": tuple(value.label_id for value in values)},
        )
        lineage = RevisionLineage(
            revision,
            label=f"atomistic-labels-{len(values)}",
            parent_revision_id=None if parent is None else parent.revision.revision_id,
            parent_lineage_id=None if parent is None else parent.lineage.lineage_id,
        )
        self.records = values
        self.revision = revision
        self.lineage = lineage
        self.system_id = first.system_id
        self.topology_id = first.topology_id
        self.units = first.units
        self.label_set_id = canonical_fingerprint(
            {
                "kind": "atomistic-label-set",
                "revision": revision.revision_id,
                "lineage": lineage.lineage_id,
            }
        )

    def append(self, records: Sequence[AtomisticLabelRecord], /) -> "AtomisticLabelSet":
        additions = tuple(records)
        if not additions:
            return self
        return AtomisticLabelSet(self.records + additions, parent=self)

    def _admit_system(self, system: PreparedAtomisticSystem, /) -> None:
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("system must be a PreparedAtomisticSystem.")
        if system.prepared_id != self.system_id:
            raise ValueError("Label set belongs to another prepared atomistic system.")
        if self.units.unit_system_id != system.plan.units.unit_system_id:
            raise ValueError("Label units differ from the prepared system unit system.")
        expected_ids = np.asarray(system.plan.particle_ids)
        if any(
            not np.array_equal(np.asarray(value.frame.stable_ids), expected_ids)
            for value in self.records
        ):
            raise ValueError(
                "Label frame stable-ID order must match the prepared atomistic system."
            )
        if system.plan.cell is None:
            if any(value.frame.cell_vectors is not None for value in self.records):
                raise ValueError("Label frames of a finite system cannot carry a cell.")
            if any(value.stress is not None for value in self.records):
                raise ValueError("Provider stress labels require a periodic system.")

    def source(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        split: AtomisticLabelSplit | None = None,
    ) -> AtomisticStructureSource:
        """Immutable canonical structure source of the label records.

        One record per label (in label-set order, optionally restricted to one
        ``split``): its record ID is the label ID and its independent-unit
        lineage the exact configuration identity, so a configuration labeled
        by several providers is one unit. Geometry keeps the frame positions
        and, for a periodic system, the frame cell vectors (the system cell
        when a frame carries none); species, masses (in the system mass unit)
        and stable IDs come from the prepared system. Energy and conservative
        forces are always available. A ``stress`` field exists when any record
        of the set carries provider stress: the native tensile Cauchy stress in
        the system pressure unit (``CAUCHY_TENSION_POSITIVE``), unavailable on
        records without it. Every split therefore shares one schema.
        """
        self._admit_system(system)
        if split is None:
            records = self.records
        else:
            split = parse(split, AtomisticLabelSplit, "split")
            records = tuple(value for value in self.records if value.split == split)
            if not records:
                raise ValueError(f"The label set holds no {split!r} records.")
        return _label_source(
            system,
            records,
            stress=any(value.stress is not None for value in self.records),
        )

    def training_problem(
        self,
        system: PreparedAtomisticSystem,
        graph_execution: AtomisticGraphExecutionPlan,
        objectives: Sequence[AbstractAtomisticObjective],
        /,
        *,
        cutoff: float,
        skin: float = 0.0,
        batch_size: int | None = None,
        seed: int = 0,
        weighting: AtomisticObjectiveWeighting = "literal",
        precision: AtomisticTrainingPrecision | None = None,
    ) -> AtomisticTrainingProblem:
        """Lower the train/validation records into a streaming training problem.

        The ``train`` source feeds an epoch loader of ``batch_size`` draws
        (``None``: full batch) shuffled from ``seed``; the ``validation``
        source, when present, a full-batch validation loader whose lineage,
        record identities and contents must be disjoint from training.
        ``objectives`` bind the canonical ``energy``, ``forces`` and, for a
        periodic system with provider stress, ``stress`` slots; graph
        topologies are prepared for ``cutoff + skin``.
        """
        if not any(value.split == "train" for value in self.records):
            raise ValueError("A label set requires at least one training record.")
        training = self.source(system, split="train")
        validation = (
            self.source(system, split="validation")
            if any(value.split == "validation" for value in self.records)
            else None
        )
        kind = AtomisticSpeciesKind(training.schema.field("species").convention)
        fields = (
            AtomisticTrainingFields(species_kind=kind)
            if system.plan.cell is None
            else AtomisticTrainingFields(
                cell="cell", periodic="periodic", species_kind=kind
            )
        )
        return AtomisticTrainingProblem(
            atomistic_training_loader(
                training, batch_size=batch_size, stream="training", seed=seed
            ),
            graph_execution,
            objectives,
            scale=system.plan.units.scale,
            cutoff=cutoff,
            skin=skin,
            fields=fields,
            validation=(
                None
                if validation is None
                else atomistic_training_loader(validation, stream="validation")
            ),
            weighting=weighting,
            precision=precision,
        )


def _label_batch(
    system: PreparedAtomisticSystem, records: tuple[AtomisticLabelRecord, ...], /
) -> AtomisticBatch:
    count = len(records)
    plan = system.plan
    capacity = plan.particle_ids.size
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
        jnp.broadcast_to(plan.atomic_numbers, (count, capacity)),
        jnp.stack(tuple(record.frame.positions for record in records)),
        jnp.broadcast_to(plan.masses, (count, capacity)),
        plan.units.scale,
        particle_ids=jnp.broadcast_to(plan.particle_ids, (count, capacity)),
        atom_type_ids=jnp.broadcast_to(plan.atom_type_ids, (count, capacity)),
        element_mask=jnp.broadcast_to(plan.element_mask, (count, capacity)),
        atom_mask=jnp.broadcast_to(plan.active_mask, (count, capacity)),
        cells=cells,
        periodic_axes=periodic,
        structure_ids=tuple(record.configuration_id for record in records),
    )


def _label_source(
    system: PreparedAtomisticSystem,
    records: tuple[AtomisticLabelRecord, ...],
    /,
    *,
    stress: bool,
) -> AtomisticStructureSource:
    """Ingest label records at float64; records without stress are unavailable."""
    stress_values = None
    stress_available = None
    if stress:
        stress_values = np.stack(
            tuple(
                np.zeros((3, 3), dtype=np.float64)
                if record.stress is None
                else np.asarray(record.stress, dtype=np.float64)
                for record in records
            )
        )
        stress_available = np.asarray(
            tuple(record.stress is not None for record in records), dtype=np.bool_
        )
    return atomistic_training_source(
        _label_batch(system, records),
        mass_unit=system.plan.units.mass_unit,
        energy=np.stack(
            tuple(np.asarray(record.energy, dtype=np.float64) for record in records)
        ),
        forces=np.stack(
            tuple(np.asarray(record.forces, dtype=np.float64) for record in records)
        ),
        stress=stress_values,
        stress_available=stress_available,
        record_ids=tuple(record.label_id for record in records),
        lineage_ids=tuple(record.configuration_id for record in records),
        stress_convention=AtomisticStressConvention.CAUCHY_TENSION_POSITIVE,
    )


@final
class AtomisticLearningCampaignPlan(StrictModule, NonTrainableState):
    """Provider, acquisition, member training and runtime resources of a campaign.

    Members are fitted on ``graph_execution`` against ``objectives`` over the
    label set's canonical sources (``skin``, ``batch_size``, ``seed``,
    ``weighting`` and ``precision`` as in `AtomisticLabelSet.training_problem`;
    the problem cutoff is the largest member cutoff). Objectives may bind only
    the provider label slots ``energy``, ``forces`` and, for a periodic system,
    ``stress``. A trained member enters the candidate committee as a learned
    graph program on ``runtime_graph_execution`` with the one
    ``runtime_resources`` bundle plan whose requirements equal the member's.
    """

    system: PreparedAtomisticSystem
    provider: AbstractExternalAtomisticProvider
    acquisition: AcquisitionPlan
    graph_execution: AtomisticGraphExecutionPlan
    runtime_graph_execution: AtomisticGraphExecutionPlan
    runtime_resources: tuple[AtomisticNeighborhoodBundlePlan, ...]
    training: AtomisticTrainingPolicy
    objectives: tuple[AbstractAtomisticObjective, ...]
    precision: AtomisticTrainingPrecision
    committee_reduction: CommitteeReductionPolicy
    skin: float = eqx.field(static=True)
    batch_size: int | None = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    weighting: AtomisticObjectiveWeighting = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
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
        *,
        objectives: Sequence[AbstractAtomisticObjective],
        runtime_resources: Sequence[AtomisticNeighborhoodBundlePlan],
        skin: float = 0.0,
        batch_size: int | None = None,
        seed: int = 0,
        weighting: AtomisticObjectiveWeighting = "literal",
        precision: AtomisticTrainingPrecision | None = None,
    ) -> None:
        if (
            not isinstance(runtime_graph_execution, AtomisticGraphExecutionPlan)
            or runtime_graph_execution.backend != "particle"
        ):
            raise TypeError("runtime_graph_execution must be a particle graph plan.")
        objectives_ = tuple(objectives)
        if not objectives_ or any(
            not isinstance(value, AbstractAtomisticObjective) for value in objectives_
        ):
            raise TypeError(
                "objectives must be nonempty AbstractAtomisticObjective values."
            )
        names = [value.name for value in objectives_]
        if len(set(names)) != len(names):
            raise ValueError("Objective names must be unique within a campaign.")
        admitted = _LABEL_SLOTS if system.plan.cell is None else _PERIODIC_LABEL_SLOTS
        unknown = sorted(
            {slot for value in objectives_ for slot in value.label_slots}.difference(
                admitted
            )
        )
        if unknown:
            raise ValueError(
                f"Campaign objectives bind label slots {unknown!r}; provider labels "
                f"supply only {list(admitted)!r}."
            )
        resources = tuple(runtime_resources)
        if not resources or any(
            not isinstance(value, AtomisticNeighborhoodBundlePlan) for value in resources
        ):
            raise TypeError(
                "runtime_resources must be nonempty AtomisticNeighborhoodBundlePlan values."
            )
        coverage = [
            frozenset(requirement.requirement_id for requirement in value.requirements)
            for value in resources
        ]
        if len(set(coverage)) != len(coverage):
            raise ValueError(
                "runtime_resources must cover pairwise distinct requirement sets."
            )
        skin_ = float(skin)
        if not np.isfinite(skin_) or skin_ < 0.0:
            raise ValueError("skin must be finite and nonnegative.")
        if batch_size is not None and (isinstance(batch_size, bool) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer or None.")
        if isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        weighting_ = parse(weighting, AtomisticObjectiveWeighting, "weighting")
        precision_ = AtomisticTrainingPrecision() if precision is None else precision
        self.system = system
        self.provider = provider
        self.acquisition = acquisition
        self.graph_execution = graph_execution
        self.runtime_graph_execution = runtime_graph_execution
        self.runtime_resources = resources
        self.training = training
        self.objectives = objectives_
        self.precision = precision_
        self.committee_reduction = committee_reduction
        self.skin = skin_
        self.batch_size = None if batch_size is None else int(batch_size)
        self.seed = int(seed)
        self.weighting = weighting_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-learning-campaign",
                "system": system.prepared_id,
                "provider": provider.provider_id,
                "acquisition": acquisition.plan_id,
                "graph_execution": graph_execution.plan_id,
                "runtime_graph_execution": runtime_graph_execution.plan_id,
                "runtime_resources": [value.plan_id for value in resources],
                "training": training.policy_id,
                "objectives": [value.objective_id for value in objectives_],
                "precision": precision_.precision_id,
                "skin": skin_,
                "batch_size": self.batch_size,
                "seed": self.seed,
                "weighting": weighting_,
                "committee": committee_reduction.policy_id,
            }
        )


def _runtime_member(
    plan: AtomisticLearningCampaignPlan, model: AbstractAtomisticModel, /
) -> tuple[LearnedGraphPotentialTerm, AtomisticNeighborhoodBundlePlan]:
    """Learned runtime term of one member and the resource plan covering it."""
    # A periodic campaign system requests periodic execution; the learned term
    # admits it only when the member declares that capability.
    term = LearnedGraphPotentialTerm(model, allow_periodic=plan.system.cell is not None)
    required = frozenset(
        requirement.requirement_id for requirement in term.requirements.neighborhoods
    )
    for resources in plan.runtime_resources:
        if (
            frozenset(
                requirement.requirement_id for requirement in resources.requirements
            )
            == required
        ):
            return term, resources
    raise ValueError(
        "No campaign runtime resource plan covers exactly the learned member's "
        "neighborhood requirements."
    )


class AtomisticLearningCampaignState(StrictModule, NonTrainableState):
    labels: AtomisticLabelSet
    committee: CommitteeAtomisticPotential | None
    round_index: int = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        labels: AtomisticLabelSet,
        /,
        *,
        committee: CommitteeAtomisticPotential | None = None,
        round_index: int = 0,
    ) -> None:
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
    split: AtomisticLabelSplit = "train",
) -> tuple[AtomisticLabelRecord, ...]:
    split = parse(split, AtomisticLabelSplit, "split")
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
                "units": frame.units.unit_system_id,
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
    revisions = tuple(
        atomistic_model_revision(result.best_potential) for result in training_results
    )
    manifests = tuple(
        ModelManifest(
            revision.semantic_id,
            analysis.analysis_plan_id,
            revision.revision_id,
            {"parameters": revision.content_id},
            unit_contract_id=plan.system.plan.units.unit_system_id,
            association_ids=(
                state.labels.label_set_id,
                result.problem_id,
                result.policy_id,
                result.capabilities_id,
            ),
        )
        for result, revision in zip(training_results, revisions, strict=True)
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


# Keep one bounded promotion transaction visible: labeling and fitted candidates
# cannot replace the promoted committee before the same round's qualification.
# Native training owns numerical iteration; this boundary orders evidence and commit.
def run_atomistic_campaign_round(
    plan: AtomisticLearningCampaignPlan,
    state: AtomisticLearningCampaignState,
    frames: Sequence[AtomisticFrame],
    uncertainty: Sequence[AtomisticUncertaintyEvidence],
    initial_potentials: Sequence[AbstractAtomisticPotential],
    keys: Sequence[PRNGKey],
    qualify: Callable[
        [CommitteeAtomisticPotential], AtomisticDynamicsQualificationResult
    ],
    /,
    *,
    descriptors: Array | None = None,
) -> AtomisticCampaignRoundResult:
    """Execute one bounded label, retrain, qualify, and promotion transaction.

    Every initial member must admit a learned runtime term covered by one
    campaign resource plan; this is refused before any provider evaluation.
    Acquired frames are labeled into the ``train`` split, the extended label
    set is lowered to one shared training problem, and every member is fitted
    with the campaign policy. A provider or training failure, or an
    unsatisfied qualification, keeps the previously promoted committee.
    """

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
    for potential in potentials:
        _runtime_member(plan, potential)
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
    problem = updated_labels.training_problem(
        plan.system,
        plan.graph_execution,
        plan.objectives,
        cutoff=max(float(potential.configuration.cutoff) for potential in potentials),
        skin=plan.skin,
        batch_size=plan.batch_size,
        seed=plan.seed,
        weighting=plan.weighting,
        precision=plan.precision,
    )
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
    members = tuple(_runtime_member(plan, result.best_potential) for result in results)
    programs = tuple(
        AtomisticPotentialProgram([term]).prepare(
            plan.system,
            neighborhood_bundle=resources,
            graph_execution=plan.runtime_graph_execution,
        )
        for term, resources in members
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
    "AtomisticLabelSplit",
    "AtomisticLearningCampaignPlan",
    "AtomisticLearningCampaignState",
    "label_atomistic_acquisitions",
    "run_atomistic_campaign_round",
]
