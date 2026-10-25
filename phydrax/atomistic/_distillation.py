#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native offline and on-policy atomistic distillation.

A native teacher (`AtomisticTeacher`) is a frozen parameter-role holder of a
native model: its parameters never receive gradients, while coordinate and
strain derivatives (forces, stress, Hessian-vector products) stay available.
`label_atomistic_structures` evaluates the teacher natively, jitted over
batches prepared by `AtomisticBatchPreparation`, and streams the labeled
records in bounded chunks through the canonical `AtomisticStructureStoreWriter`,
returning the committed `AtomisticStoredStructureSource` (resumable after an
interruption). Its schema extends the canonical training schema with typed
teacher fields. Label identity binds the teacher (architecture, numeric
revision, name), the label request (including the stored Hessian probe
definition) and every source record identity and digest, so labels of
different teacher revisions or definitions stay distinguishable.

Distillation objectives plug into `AtomisticTrainingProblem` and
`fit_atomistic_potential`: E/F/S/atom-energy matching reuses the P7 objectives
on the teacher slots; `AtomisticEmbeddingObjective` matches representations
through a trainable equivariant `AtomisticEmbeddingProjector`;
`AtomisticHessianVectorObjective` matches stored teacher HVPs with the
student's mixed-third-order admitted Hessian action; and
`AtomisticBoltzmannObjective` is a self-normalized finite-sample KL estimator
over complete groups of one phase-space measure, one chemistry and one
authenticated generating Hamiltonian (`AtomisticGeneratorState`, a fixed row
of a canonically qualified multistate segment) or an MBAR-normalized
`AtomisticGeneratorMixture` that keeps its analysis's retained selection.

The on-policy transaction (`run_atomistic_distillation_segment`) is the
committed phase machine ROLLOUT -> LABEL -> TRAIN -> QUALIFY -> PROMOTE over a
native campaign rollout publishing into a bounded FIFO
`AtomisticExperienceBuffer` (a campaign sink, not `AtomisticReplayPolicy`).
Walkers carry their complete dynamics continuation between cadence stages and
segments and are rebound to a new driving student by the owner's Hamiltonian
transition. Every committed transition is persisted by
`write_atomistic_distillation_restart`; a failed phase never replaces the
promoted student nor consumes an uncommitted cursor.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from .._array_archive import (
    ArrayArchiveLimits,
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from .._differentiation import (
    ComponentAuthority,
    DERIVATIVE_UNSUPPORTED,
    DerivativeSurface,
    DifferentiationRequest,
    RegularityPolicy,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._iteration import IterationSession
from .._jax_config import default_device, x64_enabled
from .._model._ports import ValuePort
from .._publication import publish_resource_set
from .._resource_set import open_bounded_resource_set, ResourceSetLimits
from .._sampling._addressing import derive_key, SampleAddress
from .._strict import StrictModule
from .._trainable import (
    ExplicitFreeze,
    NonTrainableState,
    ParameterOwner,
    partition_parameters,
)
from ..axes import Axis, AxisKey
from ..nn.operator.layers._o3 import O3IrrepLinear
from ..nn.operator.representations._irreps import O3IrrepLayout
from ..typing import parse
from ..units import ONE, UnitDefinition
from ..uq import (
    FreeEnergySelectionPlan,
    FreeEnergyStatus,
    multistate_bennett_acceptance_ratio,
)
from ..uq._atomistic_free_energy import reduced_potential_dataset_from_multistate
from ._campaign import (
    AbstractAtomisticCampaignSink,
    AtomisticCampaignOutput,
    AtomisticCampaignPlan,
    AtomisticCampaignStore,
    AtomisticCampaignTask,
    AtomisticDynamicsContinuation,
    AtomisticDynamicsStageOperation,
    AtomisticOutputReceipt,
    AtomisticStageOutcome,
    AtomisticStagePlan,
    AtomisticStageRoute,
    AtomisticStageRouting,
    AtomisticTaskPacket,
)
from ._checkpoint import _restore_training_restart_members, _training_restart_members
from ._conditioning import (
    AtomisticConditionLevel,
    AtomisticObservable,
    AtomisticObservableRequest,
)
from ._dynamics import (
    AtomisticDynamicsPlan,
    AtomisticIntegratorPlan,
    PreparedAtomisticDynamics,
)
from ._graph import AtomisticGraphExecutionPlan
from ._hessian import ATOMISTIC_HESSIAN_REQUEST
from ._model_artifact import read_atomistic_model_artifact, write_atomistic_model_artifact
from ._neighborhood_bundle import AtomisticNeighborhoodBundlePlan
from ._potential import (
    AbstractAtomisticModel,
    AbstractAtomisticPotential,
    atomistic_model_revision,
    AtomisticStressConvention,
)
from ._potential_program import AtomisticPotentialProgram, LearnedGraphPotentialTerm
from ._structure_store import (
    AbstractAtomisticStructureSource,
    AtomisticStoreCommit,
    AtomisticStoredStructureSource,
    AtomisticStoreField,
    AtomisticStructureBatch,
    AtomisticStructureRecord,
    AtomisticStructureSource,
    AtomisticStructureStoreSchema,
    AtomisticStructureStoreWriter,
    open_atomistic_structure_store,
)
from ._system import PreparedAtomisticSystem
from ._thermodynamic import (
    AtomisticThermodynamicStatePlan,
    PreparedThermodynamicStateTable,
)
from ._training import (
    ATOMISTIC_CARTESIAN_AXIS,
    atomistic_training_loader,
    atomistic_training_schema,
    AtomisticBatchPreparation,
    AtomisticTrainingFields,
    AtomisticTrainingPolicy,
    AtomisticTrainingProblem,
    AtomisticTrainingResult,
    evaluate_atomistic_objectives,
    fit_atomistic_potential,
)
from ._training_data import (
    AtomisticFieldBinding,
    AtomisticLoaderCursor,
    AtomisticLoaderStaging,
    AtomisticStructureLoader,
)
from ._training_objectives import (
    _binding,
    _finite_positive,
    _loss,
    _masked,
    _moments,
    _objective_identity,
    _require_binding,
    _require_energy_model,
    _require_model_scale,
    _schedule,
    _term,
    AbstractAtomisticObjective,
    atomistic_stress_unit,
    atomistic_training_derivatives,
    atomistic_training_hessian,
    atomistic_training_outputs,
    AtomisticHostLabels,
    AtomisticObjectiveSchedule,
    AtomisticObjectiveTerm,
    AtomisticResidualLoss,
    AtomisticTrainingBatch,
)
from ._types import AtomisticScaleContract, AtomisticSpeciesKind, AtomisticStatus
from .sampling._multistate import (
    AtomisticCanonicalSamplingQualification,
    AtomisticMultistateEquilibriumSamples,
    AtomisticMultistateSegmentResult,
)


AtomisticHessianProbeKind = Literal["rademacher", "gaussian"]
AtomisticDistillationPhase = Literal["rollout", "label", "train", "qualify", "promote"]
AtomisticDistillationDrive = Literal["promoted", "candidate"]

ATOMISTIC_DISTILLATION_PHASES: tuple[AtomisticDistillationPhase, ...] = (
    "rollout",
    "label",
    "train",
    "qualify",
    "promote",
)
"""Committed phase order of one on-policy distillation segment."""

TEACHER_ENERGY_SLOT = "teacher_energy"
TEACHER_FORCES_SLOT = "teacher_forces"
TEACHER_STRESS_SLOT = "teacher_stress"
TEACHER_ATOM_ENERGY_SLOT = "teacher_atom_energy"
TEACHER_EMBEDDING_SLOT = "teacher_embedding"
HESSIAN_PROBE_SLOT = "hessian_probe"
TEACHER_HVP_SLOT = "teacher_hvp"
BOLTZMANN_GROUP_SLOT = "boltzmann_group"
BOLTZMANN_GENERATOR_SLOT = "boltzmann_generator"
BOLTZMANN_PV_SLOT = "boltzmann_pv"

ATOMISTIC_HVP_TRAINING_REQUEST = DifferentiationRequest(
    (DerivativeSurface.INPUT, DerivativeSurface.MODEL_PARAMETER),
    order=3,
    authority=ComponentAuthority.MODEL,
)
"""Mixed third order: student parameter gradients of a coordinate HVP loss."""

_TEACHER_HESSIAN_OBSERVABLES = AtomisticObservableRequest(
    (AtomisticObservable.HESSIAN_ACTION,), derivatives=(ATOMISTIC_HESSIAN_REQUEST,)
)
_STUDENT_HESSIAN_OBSERVABLES = AtomisticObservableRequest(
    (AtomisticObservable.HESSIAN_ACTION,),
    derivatives=(ATOMISTIC_HESSIAN_REQUEST, ATOMISTIC_HVP_TRAINING_REQUEST),
)
_PROBE_ADDRESS = SampleAddress(
    "phydrax.atomistic.distillation", "hessian-probe", role="probe"
)
_CAMPAIGN_ADDRESS = SampleAddress(
    "phydrax.atomistic.distillation", "rollout-campaign", role="segment"
)
_TRAINING_ADDRESS = SampleAddress(
    "phydrax.atomistic.distillation", "student-training", role="segment"
)
_PROBE_AXIS_SCOPE = "atomistic-distillation"
_WALKER_FAMILY = "walker"
_BUFFER_FILE = "buffer.archive"
_BUFFER_FORMAT = "phydrax-atomistic-experience-buffer"
_RESTART_FORMAT = "phydrax-atomistic-distillation-restart"
_RESTART_RECEIPT = "restart.json"
_RESTART_STATE = "state.archive"
_RESTART_TRAINING_PREFIX = "student/"
_RESTART_MODELS = ("promoted", "driving", "candidate")
_BUFFER_LIMITS = ArrayArchiveLimits(
    max_container_bytes=1 << 32,
    max_aggregate_bytes=1 << 32,
    max_member_bytes=1 << 28,
    max_manifest_bytes=1 << 26,
    max_members=1 << 18,
)
_STATE_LIMITS = ArrayArchiveLimits(
    max_container_bytes=1 << 31,
    max_aggregate_bytes=1 << 31,
    max_member_bytes=1 << 28,
    max_manifest_bytes=1 << 24,
    max_members=1 << 16,
)
_RESTART_LIMITS = ResourceSetLimits(
    max_total_bytes=1 << 35,
    max_member_bytes=1 << 33,
    max_members=24,
    max_depth=3,
)
_RECEIPT_BYTES = 1 << 20


# ------------------------------------------------------------------- validators


def _name(value: str, label: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty canonical string.")
    return value


def _positive(value: int, label: str, /, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer.")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return resolved


def _finite(value: float, label: str, /) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{label} must be a real number.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be finite.")
    return resolved


def _require_x64(owner: str, /) -> None:
    if not x64_enabled():
        raise ValueError(f"{owner} stores float64 teacher labels; enable JAX x64.")


def _digest_words(digest: str, /) -> tuple[int, ...]:
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("A semantic probe address needs a SHA-256 record digest.")
    return tuple(int(digest[start : start + 8], 16) for start in range(0, 64, 8))


def _model_revision(model: AbstractAtomisticModel, /) -> str:
    return atomistic_model_revision(model).revision_id


def _component_digest(components: Sequence[Any], /) -> dict[str, Any]:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tuple(components), eqx.is_array))
    return array_tree_fingerprint([np.asarray(leaf) for leaf in jax.device_get(leaves)])


# --------------------------------------------------------------------- teacher


@final
class AtomisticTeacher(StrictModule, ExplicitFreeze):
    """A native model frozen on purpose as the source of distillation labels.

    The holder is an explicit freeze: no teacher leaf is ever a training
    parameter. ``frozen`` is the in-graph view: every inexact leaf of the
    teacher (parameters, fixed tables and model state) passes through
    ``stop_gradient``, so enclosing derivatives never reach the teacher while
    coordinate and strain derivatives (forces, stress, Hessian actions) are
    untouched. ``teacher_id`` binds the architecture, force method, numeric
    revision, the content of the fixed and model-state lanes and the name.
    """

    model: AbstractAtomisticModel
    name: str = eqx.field(static=True)
    architecture_id: str = eqx.field(static=True)
    revision_id: str = eqx.field(static=True)
    teacher_id: str = eqx.field(static=True)

    def __init__(self, model: AbstractAtomisticModel, /, *, name: str) -> None:
        if not isinstance(model, AbstractAtomisticModel):
            raise TypeError("model must implement AbstractAtomisticModel.")
        name_ = _name(name, "name")
        revision = _model_revision(model)
        _, state, fixed = partition_parameters(model)
        self.model = model
        self.name = name_
        self.architecture_id = model.architecture_id
        self.revision_id = revision
        self.teacher_id = canonical_fingerprint(
            {
                "kind": "atomistic-distillation-teacher",
                "name": name_,
                "architecture": model.architecture_id,
                "method": model.method_id,
                "revision": revision,
                "fixed": _component_digest((state, fixed)),
            }
        )

    @property
    def frozen(self) -> AbstractAtomisticModel:
        """The teacher with stopped gradients on every leaf and live coordinates."""
        return jax.tree.map(
            lambda leaf: (
                jax.lax.stop_gradient(leaf) if eqx.is_inexact_array(leaf) else leaf
            ),
            self.model,
        )


# ---------------------------------------------------------------------- probes


@final
class AtomisticHessianProbe(StrictModule, NonTrainableState):
    """Definition of the stored Hessian-vector probes of one HVP label.

    Probe ``k`` of a record is drawn from the key
    ``derive_key(key(seed), address, *digest_words, k)`` of the record's
    content digest, so it follows the record, never its batch position, draw or
    lane. ``rademacher`` probes have unit-magnitude components, ``gaussian``
    standard normal ones; probes are coordinate tangents in the length unit and
    are stored with the label they produced.
    """

    kind: AtomisticHessianProbeKind = eqx.field(static=True)
    count: int = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    probe_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: AtomisticHessianProbeKind = "rademacher",
        /,
        *,
        count: int,
        seed: int,
    ) -> None:
        kind_ = parse(kind, AtomisticHessianProbeKind, "kind")
        count_ = _positive(count, "count")
        seed_ = _positive(seed, "seed", minimum=0)
        if seed_ >= 1 << 32:
            raise ValueError("seed must lie in [0, 2**32).")
        self.kind = kind_
        self.count = count_
        self.seed = seed_
        self.probe_id = canonical_fingerprint(
            {
                "kind": "atomistic-hessian-probe",
                "distribution": kind_,
                "count": count_,
                "seed": seed_,
                "address": [
                    _PROBE_ADDRESS.namespace,
                    _PROBE_ADDRESS.operation,
                    _PROBE_ADDRESS.role,
                ],
                "key": "record-digest-words-then-probe-index",
            }
        )

    @property
    def axis(self) -> Axis:
        return Axis(AxisKey(_PROBE_AXIS_SCOPE, f"hessian-probe-{self.count}"), self.count)

    def draw(self, record_digest: str, atoms: int, /) -> np.ndarray:
        """Host probes ``(atoms, count, 3)`` of one record, in float64."""
        _require_x64("AtomisticHessianProbe.draw")
        words = _digest_words(record_digest)
        rows = _positive(atoms, "atoms")
        with default_device(jax.devices("cpu")[0]):
            base = derive_key(jr.key(self.seed), _PROBE_ADDRESS, *words)
            keys = jax.vmap(lambda index: jr.fold_in(base, index))(
                jnp.arange(self.count, dtype=jnp.uint32)
            )
            if self.kind == "rademacher":
                values = jax.vmap(
                    lambda key: jr.rademacher(key, (rows, 3), dtype=jnp.float64)
                )(keys)
            else:
                values = jax.vmap(
                    lambda key: jr.normal(key, (rows, 3), dtype=jnp.float64)
                )(keys)
        return np.moveaxis(np.asarray(values, dtype=np.float64), 0, 1)


# ------------------------------------------------------------- label requests


def _embedding_level(observable: AtomisticObservable, /) -> AtomisticConditionLevel:
    if observable is AtomisticObservable.NODE_EMBEDDINGS:
        return AtomisticConditionLevel.ATOM
    if observable is AtomisticObservable.GRAPH_EMBEDDINGS:
        return AtomisticConditionLevel.CASE
    raise ValueError("Embedding labels use NODE_EMBEDDINGS or GRAPH_EMBEDDINGS.")


def _embedding_spec(
    value: tuple[AtomisticObservable, ValuePort, O3IrrepLayout], label: str, /
) -> tuple[AtomisticObservable, ValuePort, O3IrrepLayout]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{label} must be an (observable, port, O3IrrepLayout) tuple.")
    observable = parse(value[0], AtomisticObservable, f"{label} observable")
    port, layout = value[1], value[2]
    _embedding_level(observable)
    if not isinstance(port, ValuePort):
        raise TypeError(f"{label} port must be a ValuePort.")
    if not isinstance(layout, O3IrrepLayout):
        raise TypeError(f"{label} layout must be an O3IrrepLayout.")
    if port.event_shape != (layout.packed_size,):
        raise ValueError(
            f"{label} port carries {port.event_shape}; the declared irrep layout packs "
            f"{layout.packed_size} components."
        )
    return observable, port, layout


def _embedding_convention(port: ValuePort, layout: O3IrrepLayout, /) -> str:
    return f"teacher-embedding;port={port.port_id};layout={layout.layout_id}"


def _probe_convention(probe: AtomisticHessianProbe, /) -> str:
    return f"hessian-probe;probe={probe.probe_id}"


def _atom_energy_convention(teacher: AtomisticTeacher, complete: bool, /) -> str:
    return f"teacher-atom-energy;teacher={teacher.teacher_id};complete={str(complete).lower()}"


@final
class AtomisticLabelRequest(StrictModule, NonTrainableState):
    """The teacher quantities of one label definition.

    ``embeddings`` binds one representation ``(observable, port, layout)``:
    ``NODE_EMBEDDINGS`` (atom level) or ``GRAPH_EMBEDDINGS`` (structure level)
    with the exact port (layer and representation identity) and the
    ``O3IrrepLayout`` of its packed components (axis identity). ``hvp`` stores
    coordinate Hessian-vector products of the declared probes together with
    the probes; ``hessian_policy`` is the regularity permission of the teacher
    Hessian (almost-everywhere models need it explicitly). Stress labels exist
    only on cases with a periodic axis and a finite nonsingular cell.
    """

    energy: bool = eqx.field(static=True)
    forces: bool = eqx.field(static=True)
    stress: bool = eqx.field(static=True)
    atom_energy: bool = eqx.field(static=True)
    embedding_observable: AtomisticObservable | None = eqx.field(static=True)
    embedding_port: ValuePort | None
    embedding_layout: O3IrrepLayout | None
    hvp: AtomisticHessianProbe | None
    hessian_policy: RegularityPolicy | None
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        energy: bool = True,
        forces: bool = True,
        stress: bool = False,
        atom_energy: bool = False,
        embeddings: tuple[AtomisticObservable, ValuePort, O3IrrepLayout] | None = None,
        hvp: AtomisticHessianProbe | None = None,
        hessian_policy: RegularityPolicy | None = None,
    ) -> None:
        flags = (energy, forces, stress, atom_energy)
        if any(not isinstance(value, bool) for value in flags):
            raise TypeError("energy, forces, stress and atom_energy must be booleans.")
        spec = None if embeddings is None else _embedding_spec(embeddings, "embeddings")
        if hvp is not None and not isinstance(hvp, AtomisticHessianProbe):
            raise TypeError("hvp must be an AtomisticHessianProbe or None.")
        if hessian_policy is not None and (
            hvp is None or not isinstance(hessian_policy, RegularityPolicy)
        ):
            raise ValueError("hessian_policy is a RegularityPolicy of a requested HVP.")
        if not any(flags) and spec is None and hvp is None:
            raise ValueError("A label request needs at least one teacher quantity.")
        self.energy = energy
        self.forces = forces
        self.stress = stress
        self.atom_energy = atom_energy
        self.embedding_observable = None if spec is None else spec[0]
        self.embedding_port = None if spec is None else spec[1]
        self.embedding_layout = None if spec is None else spec[2]
        self.hvp = hvp
        self.hessian_policy = hessian_policy
        self.request_id = canonical_fingerprint(
            {
                "kind": "atomistic-label-request",
                "energy": energy,
                "forces": forces,
                "stress": stress,
                "atom_energy": atom_energy,
                "embedding": None
                if spec is None
                else [spec[0].value, spec[1].port_id, spec[2].layout_id],
                "hvp": None if hvp is None else hvp.probe_id,
                "hessian_policy": None
                if hessian_policy is None
                else [
                    hessian_policy.allow_almost_everywhere,
                    hessian_policy.allow_undeclared,
                ],
            }
        )

    @property
    def conservative(self) -> bool:
        """Whether a scalar-energy teacher is required."""
        return (
            self.energy
            or self.forces
            or self.stress
            or self.atom_energy
            or self.hvp is not None
        )

    @property
    def slots(self) -> tuple[str, ...]:
        """Teacher fields stored by this request, in canonical order."""
        slots: list[str] = []
        if self.energy:
            slots.append(TEACHER_ENERGY_SLOT)
        if self.forces:
            slots.append(TEACHER_FORCES_SLOT)
        if self.stress:
            slots.append(TEACHER_STRESS_SLOT)
        if self.atom_energy:
            slots.append(TEACHER_ATOM_ENERGY_SLOT)
        if self.embedding_observable is not None:
            slots.append(TEACHER_EMBEDDING_SLOT)
        if self.hvp is not None:
            slots.extend((HESSIAN_PROBE_SLOT, TEACHER_HVP_SLOT))
        return tuple(slots)


def atomistic_label_id(
    teacher: AtomisticTeacher, request: AtomisticLabelRequest, /
) -> str:
    """Identity of one teacher label definition (teacher + request + probes)."""
    if not isinstance(teacher, AtomisticTeacher):
        raise TypeError("teacher must be an AtomisticTeacher.")
    if not isinstance(request, AtomisticLabelRequest):
        raise TypeError("request must be an AtomisticLabelRequest.")
    return canonical_fingerprint(
        {
            "kind": "atomistic-teacher-label",
            "teacher": teacher.teacher_id,
            "request": request.request_id,
        }
    )


def _label_record_id(label_id: str, record_id: str, record_digest: str, /) -> str:
    return "label-" + canonical_fingerprint(
        {
            "kind": "atomistic-teacher-label-record",
            "label": label_id,
            "record": record_id,
            "digest": record_digest,
        }
    )


def _require_teacher(
    teacher: AtomisticTeacher, request: AtomisticLabelRequest, /
) -> None:
    atomistic_label_id(teacher, request)
    if request.conservative and not (
        isinstance(teacher.model, AbstractAtomisticPotential)
        and teacher.model.capabilities.conservative_energy
    ):
        raise ValueError(
            "Energy, force, stress, atom-energy and HVP labels need a conservative "
            "scalar-energy teacher; property-only teachers label embeddings only."
        )
    if request.hvp is not None:
        contract = teacher.model.derivative_contract
        if contract is None:
            raise ValueError(
                f"{DERIVATIVE_UNSUPPORTED}: the teacher declares no derivative contract; "
                "its Hessian is never inferred."
            )
        contract.require(ATOMISTIC_HESSIAN_REQUEST, policy=request.hessian_policy)


# ---------------------------------------------------------------------- schema


def _canonical_levels(schema: AtomisticStructureStoreSchema, /) -> None:
    if len(schema.levels) != 2:
        raise ValueError(
            "Distillation sources use the canonical structure/atom schema; other "
            "levels have no teacher-label mapping."
        )


def atomistic_distillation_schema(
    base: AtomisticStructureStoreSchema,
    teacher: AtomisticTeacher,
    request: AtomisticLabelRequest,
    /,
    *,
    stress_convention: AtomisticStressConvention = (
        AtomisticStressConvention.CAUCHY_TENSION_POSITIVE
    ),
    atom_energy_complete: bool | None = None,
) -> AtomisticStructureStoreSchema:
    """The canonical structure/atom schema extended with typed teacher fields.

    Energies, forces and stress keep the canonical physical conventions of the
    training objectives; the teacher-specific atom-energy decomposition
    ``teacher-atom-energy;teacher=<id>;complete=<bool>``, the embedding
    ``port``/``layout`` and the probe definition enter their field
    conventions. ``atom_energy_complete`` is the teacher decomposition's
    completeness and is required exactly when atom energies are requested.
    """
    if not isinstance(base, AtomisticStructureStoreSchema):
        raise TypeError("base must be an AtomisticStructureStoreSchema.")
    _canonical_levels(base)
    _require_teacher(teacher, request)
    if request.atom_energy != (atom_energy_complete is not None):
        raise ValueError(
            "atom_energy_complete is declared exactly for atom-energy labels."
        )
    clash = set(base.field_names) & set(request.slots)
    if clash:
        raise ValueError(f"The source already holds teacher fields {sorted(clash)!r}.")
    convention = parse(stress_convention, AtomisticStressConvention, "stress_convention")
    scale = teacher.model.scale
    structure, atom = base.structure_level, base.atom_level
    fields: list[AtomisticStoreField] = list(base.fields)
    if request.energy:
        fields.append(
            AtomisticStoreField(
                TEACHER_ENERGY_SLOT,
                structure,
                "float64",
                kind="continuous",
                role="label",
                convention="total-energy",
                unit=scale.energy_unit,
                extensivity="extensive",
            )
        )
    if request.forces:
        fields.append(
            AtomisticStoreField(
                TEACHER_FORCES_SLOT,
                atom,
                "float64",
                kind="continuous",
                role="label",
                convention="conservative-force",
                axes=(ATOMISTIC_CARTESIAN_AXIS,),
                unit=scale.force_unit,
                extensivity="intensive",
            )
        )
    if request.stress:
        fields.append(
            AtomisticStoreField(
                TEACHER_STRESS_SLOT,
                structure,
                "float64",
                kind="continuous",
                role="label",
                convention=convention.value,
                axes=(ATOMISTIC_CARTESIAN_AXIS, ATOMISTIC_CARTESIAN_AXIS),
                unit=atomistic_stress_unit(scale),
                extensivity="intensive",
                availability="optional",
            )
        )
    if request.atom_energy and atom_energy_complete is not None:
        fields.append(
            AtomisticStoreField(
                TEACHER_ATOM_ENERGY_SLOT,
                atom,
                "float64",
                kind="continuous",
                role="label",
                convention=_atom_energy_convention(teacher, atom_energy_complete),
                unit=scale.energy_unit,
                extensivity="extensive",
            )
        )
    if (
        request.embedding_observable is not None
        and request.embedding_port is not None
        and request.embedding_layout is not None
    ):
        level = _embedding_level(request.embedding_observable)
        layout = request.embedding_layout
        port = request.embedding_port
        fields.append(
            AtomisticStoreField(
                TEACHER_EMBEDDING_SLOT,
                atom if level is AtomisticConditionLevel.ATOM else structure,
                "float64",
                kind="continuous",
                role="label",
                convention=_embedding_convention(port, layout),
                axes=(
                    Axis(
                        AxisKey(_PROBE_AXIS_SCOPE, f"embedding-{layout.layout_id[:32]}"),
                        layout.packed_size,
                    ),
                ),
                unit=ONE,
                extensivity="intensive",
            )
        )
    if request.hvp is not None:
        axes = (request.hvp.axis, ATOMISTIC_CARTESIAN_AXIS)
        fields.append(
            AtomisticStoreField(
                HESSIAN_PROBE_SLOT,
                atom,
                "float64",
                kind="continuous",
                role="derived",
                convention=_probe_convention(request.hvp),
                axes=axes,
                unit=scale.length_unit,
                extensivity="intensive",
            )
        )
        fields.append(
            AtomisticStoreField(
                TEACHER_HVP_SLOT,
                atom,
                "float64",
                kind="continuous",
                role="label",
                convention="coordinate-hessian-vector-product",
                axes=axes,
                unit=scale.force_unit,
                extensivity="intensive",
            )
        )
    return AtomisticStructureStoreSchema(base.levels, tuple(fields))


# ------------------------------------------------------------- teacher labels


@final
class _TeacherLabels(StrictModule):
    energy: Array | None
    forces: Array | None
    stress: Array | None
    stress_available: Array | None
    atom_energy: Array | None
    embedding: Array | None
    hvp: Array | None
    successful: Array
    overflow: Array
    atom_energy_complete: bool | None = eqx.field(static=True)


def _finite_rows(values: Array, mask: Array | None, /) -> Array:
    cases = values.shape[0]
    finite = jnp.isfinite(values)
    if mask is not None:
        finite = finite | ~mask.reshape(mask.shape + (1,) * (values.ndim - mask.ndim))
    return jnp.all(finite.reshape((cases, -1)), axis=1)


@final
class _TeacherEvaluator(StrictModule, ExplicitFreeze):
    """Pure native teacher evaluation over one prepared training batch."""

    teacher: AtomisticTeacher
    request: AtomisticLabelRequest

    def __call__(
        self, batch: AtomisticTrainingBatch, probes: Array | None, /
    ) -> _TeacherLabels:
        model = self.teacher.frozen
        request = self.request
        cases = batch.case_count
        mask = batch.atom_mask
        successful = jnp.ones((cases,), dtype=jnp.bool_)
        overflow = jnp.zeros((cases,), dtype=jnp.bool_)
        energy = forces = stress = stress_available = None
        atom_energy = embedding = hvp = None
        complete: bool | None = None
        if request.energy or request.forces or request.stress:
            derivatives = atomistic_training_derivatives(
                model,
                batch,
                forces=request.forces,
                stress_cases=batch.stress_cells if request.stress else None,
            )
            successful = successful & derivatives.successful
            overflow = overflow | derivatives.overflow
            if request.energy:
                energy = derivatives.energy.astype(jnp.float64)
                successful = successful & jnp.isfinite(energy)
            if request.forces:
                if derivatives.forces is None:
                    raise RuntimeError("Requested teacher forces were not returned.")
                forces = derivatives.forces.astype(jnp.float64)
                successful = successful & _finite_rows(forces, mask)
            if request.stress:
                if derivatives.stress is None:
                    raise RuntimeError("Requested teacher stress was not returned.")
                stress_available = derivatives.stress_available & batch.stress_cells
                stress = jnp.where(
                    stress_available[:, None, None], derivatives.stress, 0.0
                ).astype(jnp.float64)
                successful = successful & _finite_rows(stress, None)
        observables: list[AtomisticObservable] = []
        if request.atom_energy:
            observables.append(AtomisticObservable.ATOM_ENERGY)
        if request.embedding_observable is not None:
            observables.append(request.embedding_observable)
        if observables:
            outputs, evaluated, exceeded = atomistic_training_outputs(
                model, batch, AtomisticObservableRequest(tuple(observables))
            )
            successful = successful & evaluated
            overflow = overflow | exceeded
            if request.atom_energy:
                atom_energy = outputs.atom_energy.astype(jnp.float64)
                complete = outputs.atom_energy_complete
                successful = successful & _finite_rows(atom_energy, mask)
            if request.embedding_observable is not None:
                value = outputs.observable(
                    request.embedding_observable, port=request.embedding_port
                )
                level = _embedding_level(request.embedding_observable)
                if value.level is not level:
                    raise ValueError(
                        "The teacher embedding has another observable level."
                    )
                values = value.values.astype(jnp.float64)
                if level is AtomisticConditionLevel.ATOM:
                    embedding = values.reshape((cases, batch.atom_capacity, -1))
                    successful = successful & _finite_rows(embedding, mask)
                else:
                    embedding = values
                    successful = successful & _finite_rows(embedding, None)
                successful = successful & value.available
        if request.hvp is not None:
            if probes is None:
                raise RuntimeError("HVP labels need their stored probes.")
            operator = atomistic_training_hessian(
                model,
                batch,
                request=_TEACHER_HESSIAN_OBSERVABLES,
                policy=request.hessian_policy,
            )
            dtype = model.precision.coordinate_dtype
            hvp = jnp.stack(
                [
                    operator.mv(probes[:, :, index, :].astype(dtype)).astype(jnp.float64)
                    for index in range(request.hvp.count)
                ],
                axis=2,
            )
            successful = successful & operator.action_valid & _finite_rows(hvp, mask)
            overflow = overflow | operator.overflow
        return _TeacherLabels(
            energy=energy,
            forces=forces,
            stress=stress,
            stress_available=stress_available,
            atom_energy=atom_energy,
            embedding=embedding,
            hvp=hvp,
            successful=successful & ~overflow,
            overflow=overflow,
            atom_energy_complete=complete,
        )


@eqx.filter_jit
def _evaluate_teacher(
    evaluator: _TeacherEvaluator, batch: AtomisticTrainingBatch, probes: Array | None, /
) -> _TeacherLabels:
    return evaluator(batch, probes)


class AtomisticTeacherLabelError(ValueError):
    """A teacher evaluation failed on a record; no label of it is published."""


@dataclass(frozen=True, slots=True)
class _HostLabel:
    values: Mapping[str, np.ndarray]
    available: Mapping[str, np.ndarray]


def _teacher_label_batches(
    teacher: AtomisticTeacher,
    request: AtomisticLabelRequest,
    preparation: AtomisticBatchPreparation,
    cursor: AtomisticLoaderCursor,
    /,
    *,
    staging: AtomisticLoaderStaging | None = None,
) -> Iterator[tuple[dict[int, _HostLabel], bool | None]]:
    """Label the remaining draws of ``cursor.epoch``, one global batch at a time.

    Each yield holds the labels of every valid draw of one closed global batch
    keyed by its epoch draw position, and the teacher's atom-energy
    decomposition completeness (``None`` without atom energies). Only one
    batch of labels and probes is resident.
    """
    _require_x64("Teacher labeling")
    preparation.admit(teacher.model)
    loader = preparation.loader
    evaluator = _TeacherEvaluator(teacher, request)
    coordinate = teacher.model.precision.coordinate_dtype
    labels: dict[int, _HostLabel] = {}
    complete: bool | None = None
    with loader.epoch(cursor, staging=staging) as epoch:
        for item in epoch:
            provenance = item.provenance
            for shard in range(item.payload.layout.local_shards):
                batch = preparation.prepare_batch(
                    item.payload,
                    provenance,
                    coordinate_dtype=coordinate,
                    shard=shard,
                )
                cases = batch.case_count
                rows = range(shard * cases, (shard + 1) * cases)
                mask = np.asarray(batch.atom_mask, dtype=np.bool_)
                probes = None
                if request.hvp is not None:
                    host = np.zeros(
                        (cases, batch.atom_capacity, request.hvp.count, 3), np.float64
                    )
                    for case, row in enumerate(rows):
                        if provenance.valid[row]:
                            host[case, mask[case]] = request.hvp.draw(
                                provenance.record_digests[row],
                                int(np.count_nonzero(mask[case])),
                            )
                    probes = jnp.asarray(host)
                result = jax.device_get(_evaluate_teacher(evaluator, batch, probes))
                if result.atom_energy_complete is not None:
                    complete = result.atom_energy_complete
                for case, row in enumerate(rows):
                    if not provenance.valid[row]:
                        continue
                    if not bool(result.successful[case]):
                        cause = (
                            "graph capacity"
                            if bool(result.overflow[case])
                            else "scientific"
                        )
                        raise AtomisticTeacherLabelError(
                            f"Teacher {teacher.name!r} failed ({cause}) on record "
                            f"{provenance.record_ids[row]!r}; its label is not published."
                        )
                    labels[provenance.draw_indices[row]] = _host_label(
                        request, result, case, mask[case], probes
                    )
            if item.closes_batch:
                yield labels, complete
                labels = {}


def _host_label(
    request: AtomisticLabelRequest,
    result: _TeacherLabels,
    case: int,
    mask: np.ndarray,
    probes: Array | None,
    /,
) -> _HostLabel:
    values: dict[str, np.ndarray] = {}
    available: dict[str, np.ndarray] = {}

    def host(value: Any, /) -> np.ndarray:
        return np.asarray(value, dtype=np.float64)

    if result.energy is not None:
        values[TEACHER_ENERGY_SLOT] = host(result.energy[case]).reshape((1,))
    if result.forces is not None:
        values[TEACHER_FORCES_SLOT] = host(result.forces[case])[mask]
    if result.stress is not None and result.stress_available is not None:
        flag = bool(result.stress_available[case])
        values[TEACHER_STRESS_SLOT] = host(result.stress[case])[None]
        available[TEACHER_STRESS_SLOT] = np.asarray([flag], dtype=np.bool_)
    if result.atom_energy is not None:
        values[TEACHER_ATOM_ENERGY_SLOT] = host(result.atom_energy[case])[mask]
    if result.embedding is not None:
        embedding = host(result.embedding[case])
        values[TEACHER_EMBEDDING_SLOT] = (
            embedding[mask]
            if request.embedding_observable is AtomisticObservable.NODE_EMBEDDINGS
            else embedding[None]
        )
    if result.hvp is not None and probes is not None:
        values[HESSIAN_PROBE_SLOT] = host(probes[case])[mask]
        values[TEACHER_HVP_SLOT] = host(result.hvp[case])[mask]
    return _HostLabel(MappingProxyType(values), MappingProxyType(available))


def _record_columns(
    batch: AtomisticStructureBatch, index: int, /
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, int]:
    schema = batch.schema
    offsets = batch.offsets(schema.atom_level)
    start, stop = int(offsets[index]), int(offsets[index + 1])
    values: dict[str, np.ndarray] = {}
    available: dict[str, np.ndarray] = {}
    for name in batch.fields:
        field = schema.field(name)
        rows = (
            slice(index, index + 1)
            if field.level == schema.structure_level
            else slice(start, stop)
        )
        values[name] = np.asarray(batch.values[name][rows])
        if field.availability == "optional":
            available[name] = np.asarray(batch.available[name][rows], dtype=np.bool_)
    return values, available, np.asarray(batch.stable_ids[start:stop]), stop - start


def _extended_record(
    schema: AtomisticStructureStoreSchema,
    base: AtomisticStructureBatch,
    index: int,
    /,
    *,
    record_id: str,
    lineage_id: str,
    extra: Mapping[str, np.ndarray],
    extra_available: Mapping[str, np.ndarray],
) -> AtomisticStructureRecord:
    values, available, stable, atoms = _record_columns(base, index)
    values.update(extra)
    available.update(extra_available)
    return AtomisticStructureRecord(
        schema,
        record_id=record_id,
        lineage_id=lineage_id,
        counts={schema.atom_level: [atoms]},
        stable_ids=stable,
        values=values,
        available=available,
    )


def _resumable_head(
    store: AtomisticStructureStoreWriter,
    name: str,
    schemas: Sequence[AtomisticStructureStoreSchema],
    expected: Callable[[int], tuple[str, str]],
    total: int,
    /,
) -> AtomisticStoredStructureSource | None:
    """The committed prefix of one streamed output, or ``None`` before creation.

    An existing store must be this output (``name``, one of ``schemas``) with
    no retraction and exactly the expected ``(record ID, lineage)`` prefix;
    anything else is refused and never overwritten.
    """
    if not (Path(store.root) / "commits").exists():
        return None
    head = open_atomistic_structure_store(store.root, limits=store.limits)
    if head.commit.store_name != name or head.schema.schema_id not in {
        schema.schema_id for schema in schemas
    }:
        raise ValueError(
            f"The store at {store.root!r} holds another output (name or schema); it is "
            "never overwritten."
        )
    if head.commit.retractions or head.record_count > total:
        raise ValueError(f"The store at {store.root!r} is not a prefix of this output.")
    for index in range(head.record_count):
        if (head.record_ids[index], head.lineage_ids[index]) != expected(index):
            raise ValueError(
                f"The store at {store.root!r} is not a prefix of this output: record "
                f"{index} differs."
            )
    return head


def _append_records(
    store: AtomisticStructureStoreWriter,
    parent: AtomisticStoreCommit | None,
    schema: AtomisticStructureStoreSchema,
    name: str,
    records: Sequence[AtomisticStructureRecord],
    /,
) -> AtomisticStoreCommit:
    """Publish one bounded chunk (creating the store at its first chunk)."""
    head = store.create(schema, name=name) if parent is None else parent
    if not records:
        return head
    return store.append(head, AtomisticStructureSource(schema, tuple(records)))


def label_atomistic_structures(
    teacher: AtomisticTeacher,
    loader: AtomisticStructureLoader,
    fields: AtomisticTrainingFields,
    request: AtomisticLabelRequest,
    /,
    *,
    graph_execution: AtomisticGraphExecutionPlan,
    cutoff: float,
    store: AtomisticStructureStoreWriter,
    skin: float = 0.0,
    stress_convention: AtomisticStressConvention = (
        AtomisticStressConvention.CAUCHY_TENSION_POSITIVE
    ),
    staging: AtomisticLoaderStaging | None = None,
) -> AtomisticStoredStructureSource:
    """Label every record of ``loader``'s sources into a committed structure store.

    The loader's epoch 0 must draw every record exactly once in canonical
    component-then-record order on this process (``draw="epoch"``,
    ``shuffle=False``, unit repeat counts, every shard local). Batches are
    prepared by `AtomisticBatchPreparation` and the teacher is jitted over
    them. Memory is bounded: only one global loader batch of labels, probes and
    re-read source rows is resident, and labeled records are appended through
    the canonical ``store`` writer whenever ``store.layout.records_per_shard``
    are pending (the writer's limits bound every shard, commit and read).
    Each labeled record has ID ``label-<digest>`` of (label definition, source
    record ID, source record digest), the source lineage, every source field
    and the teacher fields of `atomistic_distillation_schema`. The result is
    the lazily read committed head; its ``source_id`` is the content identity,
    so labeling the same records with the same teacher and request reproduces
    it. The store is named ``teacher-labels-<label_id>`` and lives as long as
    its directory: an interrupted run resumes after its committed prefix (the
    exact expected records under the same schema), a complete store is
    returned unchanged, and a store of another definition, source, order or
    with a retraction is refused. A failed teacher evaluation raises
    `AtomisticTeacherLabelError` and publishes nothing of its batch.
    """
    if not isinstance(loader, AtomisticStructureLoader):
        raise TypeError("loader must be an AtomisticStructureLoader.")
    if not isinstance(store, AtomisticStructureStoreWriter):
        raise TypeError("store must be an AtomisticStructureStoreWriter.")
    _require_teacher(teacher, request)
    mixture = loader.mixture
    schema = mixture.schema
    _canonical_levels(schema)
    if mixture.draw != "epoch" or mixture.shuffle:
        raise ValueError(
            "Labeling streams one unshuffled complete epoch (draw='epoch', "
            "shuffle=False) so records arrive in canonical order."
        )
    if any(component.weight != 1 for component in mixture.components):
        raise ValueError("A labeling epoch draws every record exactly once (weight 1).")
    if len(loader.placement.local_shards) != loader.placement.shard_count:
        raise ValueError("Labeling publishes from one process holding every shard.")
    preparation = AtomisticBatchPreparation(
        loader,
        graph_execution,
        scale=teacher.model.scale,
        cutoff=cutoff,
        skin=skin,
        fields=fields,
    )
    candidates = {
        complete: atomistic_distillation_schema(
            schema,
            teacher,
            request,
            stress_convention=stress_convention,
            atom_energy_complete=complete,
        )
        for complete in ((True, False) if request.atom_energy else (None,))
    }
    label_id = atomistic_label_id(teacher, request)
    sources = tuple(component.source for component in mixture.components)
    offsets = np.cumsum([0, *mixture.record_counts], dtype=np.int64)
    total = int(offsets[-1])

    def locate(index: int, /) -> tuple[int, int]:
        component = int(np.searchsorted(offsets, index, side="right")) - 1
        return component, index - int(offsets[component])

    def expected(index: int, /) -> tuple[str, str]:
        component, ordinal = locate(index)
        source = sources[component]
        return (
            _label_record_id(
                label_id, source.record_ids[ordinal], source.record_digests[ordinal]
            ),
            source.lineage_ids[ordinal],
        )

    name = f"teacher-labels-{label_id}"
    head = _resumable_head(store, name, tuple(candidates.values()), expected, total)
    committed = 0 if head is None else head.record_count
    if head is not None and committed == total:
        return head
    labeled = None if head is None else head.schema
    parent = None if head is None else head.commit
    component, ordinal = locate(committed)
    position = mixture.block_starts[component] + ordinal
    batch_index = position // loader.batch_size
    if position not in loader.epoch_plan(0).batch(batch_index):
        raise RuntimeError("The unshuffled labeling epoch is not in draw order.")
    cursor = AtomisticLoaderCursor(
        loader_id=loader.loader_id,
        semantics_id=loader.semantics_id,
        structure_id=loader.structure_id,
        stream=loader.stream,
        source_ids=mixture.source_ids,
        epoch=0,
        batch_index=batch_index,
        payload_index=0,
    )
    pending: list[AtomisticStructureRecord] = []
    following = committed
    for labels, complete in _teacher_label_batches(
        teacher, request, preparation, cursor, staging=staging
    ):
        resolved = candidates[complete]
        if labeled is None:
            labeled = resolved
        elif resolved.schema_id != labeled.schema_id:
            raise AtomisticTeacherLabelError(
                "The teacher atom-energy decomposition completeness differs from the "
                "committed labels."
            )
        draws = np.asarray(sorted(labels), dtype=np.int64)
        owners, ordinals = mixture.resolve(0, draws)
        rows: dict[int, tuple[AtomisticStructureBatch, dict[int, int]]] = {}
        for owner in np.unique(owners).tolist():
            local = np.unique(ordinals[owners == owner])
            rows[owner] = (
                sources[owner].read(local),
                {int(value): place for place, value in enumerate(local.tolist())},
            )
        for draw, owner, record in zip(
            draws.tolist(), owners.tolist(), ordinals.tolist(), strict=True
        ):
            index = int(offsets[owner]) + record
            if index < following:
                continue
            if index != following:
                raise ValueError(
                    f"Record {expected(following)[0]!r} was not drawn in order; the "
                    "labeling loader must draw every record."
                )
            base, place = rows[owner]
            label = labels[draw]
            record_id, lineage_id = expected(index)
            pending.append(
                _extended_record(
                    resolved,
                    base,
                    place[record],
                    record_id=record_id,
                    lineage_id=lineage_id,
                    extra=label.values,
                    extra_available=label.available,
                )
            )
            following += 1
        if len(pending) >= store.layout.records_per_shard:
            parent = _append_records(store, parent, resolved, name, pending)
            pending = []
    if following != total or labeled is None:
        raise ValueError(
            f"Epoch 0 drew {following} of {total} records; the labeling loader must "
            "draw every record."
        )
    _append_records(store, parent, labeled, name, pending)
    return open_atomistic_structure_store(store.root, limits=store.limits)


def require_atomistic_labels(
    labeled: AbstractAtomisticStructureSource,
    teacher: AtomisticTeacher,
    request: AtomisticLabelRequest,
    source: AbstractAtomisticStructureSource,
    /,
) -> None:
    """Refuse labels that are not exactly this teacher's labels of ``source``.

    Every labeled record must be the label record of the corresponding source
    record (same order, identity, content digest and lineage); a stale,
    reordered, truncated or foreign-teacher label source is refused. Only
    record metadata is compared; no record rows are read.
    """
    if not isinstance(labeled, AbstractAtomisticStructureSource) or not isinstance(
        source, AbstractAtomisticStructureSource
    ):
        raise TypeError("labeled and source must be atomistic structure sources.")
    label_id = atomistic_label_id(teacher, request)
    expected = tuple(
        _label_record_id(label_id, record, digest)
        for record, digest in zip(source.record_ids, source.record_digests, strict=True)
    )
    if labeled.record_ids != expected:
        if set(labeled.record_ids) == set(expected):
            raise ValueError("The teacher labels are reordered relative to the source.")
        raise ValueError(
            "The teacher labels are stale: they belong to other source records, "
            "another teacher revision or another label definition."
        )
    if labeled.lineage_ids != source.lineage_ids:
        raise ValueError("The teacher labels carry another independent-unit lineage.")
    if set(source.schema.field_names) - set(labeled.schema.field_names):
        raise ValueError("The teacher labels dropped source fields.")
    if set(request.slots) - set(labeled.schema.field_names):
        raise ValueError("The teacher labels lack requested teacher fields.")


# ------------------------------------------------------------ embedding match


def _require_irrep_map(student: O3IrrepLayout, teacher: O3IrrepLayout, /) -> None:
    available = {(block.degree, block.parity) for block in student.blocks}
    for block in teacher.blocks:
        if (block.degree, block.parity) not in available:
            raise ValueError(
                f"Teacher block {block.name!r} (l={block.degree}, parity={block.parity}) "
                "has no student block of equal irrep; filling it would mix unequal "
                "irreps, which no equivariant multiplicity map does."
            )


@final
class AtomisticEmbeddingProjector(StrictModule, ParameterOwner):
    """Trainable equivariant map from student to teacher representations.

    An `O3IrrepLinear` multiplicity-space map: each teacher block mixes only
    the student blocks of identical degree and parity, so scalars receive a
    learned dense projector and nontrivial irreps an equivariant one. Its
    weights are its own parameter space; ``projector_id`` is the artifact
    identity of both layouts and the map family.
    """

    linear: O3IrrepLinear
    projector_id: str = eqx.field(static=True)

    def __init__(
        self,
        student_layout: O3IrrepLayout,
        teacher_layout: O3IrrepLayout,
        /,
        *,
        key: Array,
        dtype: Any = jnp.float64,
    ) -> None:
        if not isinstance(student_layout, O3IrrepLayout) or not isinstance(
            teacher_layout, O3IrrepLayout
        ):
            raise TypeError("Projector layouts must be O3IrrepLayout values.")
        _require_irrep_map(student_layout, teacher_layout)
        self.linear = O3IrrepLinear(student_layout, teacher_layout, dtype=dtype, key=key)
        self.projector_id = canonical_fingerprint(
            {
                "kind": "atomistic-embedding-projector",
                "map": "o3-irrep-linear-multiplicity",
                "student": student_layout.layout_id,
                "teacher": teacher_layout.layout_id,
            }
        )

    @property
    def student_layout(self) -> O3IrrepLayout:
        return self.linear.in_layout

    @property
    def teacher_layout(self) -> O3IrrepLayout:
        return self.linear.out_layout

    def __call__(self, values: Array, /) -> Array:
        return self.linear(values)


@final
class AtomisticEmbeddingObjective(AbstractAtomisticObjective):
    """Representation matching through a trainable equivariant projector.

    ``student`` and ``teacher`` bind ``(observable, port, layout)``: the layer
    and representation (port) and the packed irrep axis (layout). Both use one
    level (node or graph embeddings). The student value is mapped by the
    `AtomisticEmbeddingProjector` component supplied to the trainer and
    compared componentwise with the stored teacher label; every available
    component has unit support.
    """

    name: str = eqx.field(static=True)
    slot: str = eqx.field(static=True)
    observable: AtomisticObservable = eqx.field(static=True)
    port: ValuePort
    student_layout: O3IrrepLayout
    teacher_port: ValuePort
    teacher_layout: O3IrrepLayout
    level: AtomisticConditionLevel = eqx.field(static=True)
    loss: AtomisticResidualLoss
    scale: float | None = eqx.field(static=True)
    schedule: AtomisticObjectiveSchedule
    objective_id: str = eqx.field(static=True)

    def __init__(
        self,
        student: tuple[AtomisticObservable, ValuePort, O3IrrepLayout],
        teacher: tuple[AtomisticObservable, ValuePort, O3IrrepLayout],
        /,
        *,
        slot: str = TEACHER_EMBEDDING_SLOT,
        name: str = "embedding",
        loss: AtomisticResidualLoss | None = None,
        scale: float | None = None,
        schedule: AtomisticObjectiveSchedule | None = None,
    ) -> None:
        observable, port, student_layout = _embedding_spec(student, "student")
        teacher_observable, teacher_port, teacher_layout = _embedding_spec(
            teacher, "teacher"
        )
        level = _embedding_level(observable)
        if _embedding_level(teacher_observable) is not level:
            raise ValueError("Student and teacher embeddings must share one level.")
        _require_irrep_map(student_layout, teacher_layout)
        loss_ = _loss(loss)
        if loss_.vector_only:
            raise ValueError("Embedding components use componentwise residual forms.")
        self.name = _name(name, "name")
        self.slot = _name(slot, "slot")
        self.observable = observable
        self.port = port
        self.student_layout = student_layout
        self.teacher_port = teacher_port
        self.teacher_layout = teacher_layout
        self.level = level
        self.loss = loss_
        self.scale = _finite_positive(scale, "scale")
        self.schedule = _schedule(schedule)
        self.objective_id = _objective_identity(
            "atomistic-embedding-objective",
            name=self.name,
            slot=self.slot,
            observable=observable.value,
            port=port.port_id,
            student_layout=student_layout.layout_id,
            teacher_port=teacher_port.port_id,
            teacher_layout=teacher_layout.layout_id,
            loss=loss_.loss_id,
            scale=self.scale,
            schedule=self.schedule.schedule_id,
        )

    @property
    def requires_component(self) -> bool:
        return True

    def admit_component(self, component: Any, /) -> None:
        if not isinstance(component, AtomisticEmbeddingProjector):
            raise TypeError(
                f"Objective {self.name!r} needs an AtomisticEmbeddingProjector component."
            )
        if (
            component.student_layout.layout_id != self.student_layout.layout_id
            or component.teacher_layout.layout_id != self.teacher_layout.layout_id
        ):
            raise ValueError(f"Objective {self.name!r} projector maps other layouts.")

    @property
    def label_slots(self) -> tuple[str, ...]:
        return (self.slot,)

    def admit(
        self,
        model: AbstractAtomisticModel,
        bindings: Mapping[str, AtomisticFieldBinding],
        scale: AtomisticScaleContract,
        /,
    ) -> None:
        owner = f"Embedding objective {self.name!r}"
        _require_model_scale(model, scale, owner)
        _require_binding(
            _binding(bindings, self.slot, owner),
            owner,
            level="atom" if self.level is AtomisticConditionLevel.ATOM else "structure",
            shape=(self.teacher_layout.packed_size,),
            unit=ONE,
            conventions=(_embedding_convention(self.teacher_port, self.teacher_layout),),
        )

    def label_moments(self, labels: AtomisticHostLabels, /) -> np.ndarray:
        values, available = labels.label(self.slot)
        if self.level is AtomisticConditionLevel.CASE:
            active = labels.case_valid & available
        else:
            active = labels.case_valid[:, None] & available & labels.atom_mask
        return _moments(np.asarray(values)[active])

    def term(
        self,
        model: AbstractAtomisticModel,
        component: Any,
        batch: AtomisticTrainingBatch,
        /,
        *,
        scale: float,
        dtype: str,
    ) -> AtomisticObjectiveTerm:
        if not isinstance(component, AtomisticEmbeddingProjector):
            raise TypeError(f"Objective {self.name!r} needs its projector component.")
        outputs, successful, overflow = atomistic_training_outputs(
            model, batch, AtomisticObservableRequest((self.observable,))
        )
        value = outputs.observable(self.observable, port=self.port)
        if value.level is not self.level:
            raise ValueError(f"Embedding {self.name!r} has another observable level.")
        weight_dtype = component.linear.weights[0].dtype
        prediction = component(value.values.astype(weight_dtype)).astype(dtype)
        labels, available = batch.label(self.slot)
        size = float(self.teacher_layout.packed_size)
        if self.level is AtomisticConditionLevel.ATOM:
            prediction = prediction.reshape((batch.case_count, batch.atom_capacity, -1))
            rows = (
                (batch.case_valid & value.available)[:, None]
                & available
                & batch.atom_mask
            )
            weight = jnp.where(rows, batch.case_weight.astype(dtype)[:, None], 0.0)
        else:
            rows = batch.case_valid & value.available & available
            weight = jnp.where(rows, batch.case_weight.astype(dtype), 0.0)
        residual = _masked((prediction - labels.astype(dtype)) / scale, rows[..., None])
        penalty = jnp.sum(self.loss.pointwise(residual), axis=-1)
        numerator = jnp.sum(weight * penalty, dtype=dtype)
        support = size * jnp.sum(weight, dtype=dtype)
        valid = batch.case_valid
        return _term(
            numerator,
            support,
            valid & overflow,
            valid & ~(successful & value.successful),
        )


# ------------------------------------------------------------------- HVP match


@final
class AtomisticHessianVectorObjective(AbstractAtomisticObjective):
    """Coordinate Hessian-vector-product matching against stored teacher HVPs.

    The student action ``H_s v`` of every stored probe ``v`` uses the native
    Hessian binding with the admitted mixed-third-order request
    ``(INPUT, MODEL_PARAMETER)`` of order three (``ATOMISTIC_HVP_TRAINING_REQUEST``)
    under ``policy``; a model whose derivative contract does not admit it is
    refused at admission, never silently differentiated. Probes and labels come
    from the stored slots of the label definition ``probe``. Each available
    atom component of every probe has unit support. Strain-HVP variants have
    their own mixed request and are not served by this coordinate objective.
    """

    name: str = eqx.field(static=True)
    probe: AtomisticHessianProbe
    probe_slot: str = eqx.field(static=True)
    slot: str = eqx.field(static=True)
    policy: RegularityPolicy | None
    loss: AtomisticResidualLoss
    scale: float | None = eqx.field(static=True)
    schedule: AtomisticObjectiveSchedule
    objective_id: str = eqx.field(static=True)

    def __init__(
        self,
        probe: AtomisticHessianProbe,
        /,
        *,
        policy: RegularityPolicy | None = None,
        probe_slot: str = HESSIAN_PROBE_SLOT,
        slot: str = TEACHER_HVP_SLOT,
        name: str = "hvp",
        loss: AtomisticResidualLoss | None = None,
        scale: float | None = None,
        schedule: AtomisticObjectiveSchedule | None = None,
    ) -> None:
        if not isinstance(probe, AtomisticHessianProbe):
            raise TypeError("probe must be an AtomisticHessianProbe.")
        if policy is not None and not isinstance(policy, RegularityPolicy):
            raise TypeError("policy must be a RegularityPolicy or None.")
        loss_ = _loss(loss)
        if loss_.vector_only:
            raise ValueError("HVP components use componentwise residual forms.")
        self.name = _name(name, "name")
        self.probe = probe
        self.probe_slot = _name(probe_slot, "probe_slot")
        self.slot = _name(slot, "slot")
        self.policy = policy
        self.loss = loss_
        self.scale = _finite_positive(scale, "scale")
        self.schedule = _schedule(schedule)
        self.objective_id = _objective_identity(
            "atomistic-hessian-vector-objective",
            name=self.name,
            probe=probe.probe_id,
            probe_slot=self.probe_slot,
            slot=self.slot,
            policy=None
            if policy is None
            else [policy.allow_almost_everywhere, policy.allow_undeclared],
            loss=loss_.loss_id,
            scale=self.scale,
            schedule=self.schedule.schedule_id,
        )

    @property
    def label_slots(self) -> tuple[str, ...]:
        return (self.probe_slot, self.slot)

    def admit(
        self,
        model: AbstractAtomisticModel,
        bindings: Mapping[str, AtomisticFieldBinding],
        scale: AtomisticScaleContract,
        /,
    ) -> None:
        owner = f"HVP objective {self.name!r}"
        _require_energy_model(model, owner)
        _require_model_scale(model, scale, owner)
        contract = model.derivative_contract
        if contract is None:
            raise ValueError(
                f"{DERIVATIVE_UNSUPPORTED}: {owner} needs a declared derivative contract "
                "admitting the mixed third-order (INPUT, MODEL_PARAMETER) request."
            )
        contract.require(ATOMISTIC_HESSIAN_REQUEST, policy=self.policy)
        contract.require(ATOMISTIC_HVP_TRAINING_REQUEST, policy=self.policy)
        shape = (self.probe.count, 3)
        probe = _binding(bindings, self.probe_slot, owner)
        if (
            probe.level_kind != "atom"
            or probe.value_kind != "continuous"
            or probe.role != "derived"
            or tuple(axis.size for axis in probe.axes) != shape
            or probe.unit is None
            or probe.unit.unit_id != scale.length_unit.unit_id
            or probe.convention != _probe_convention(self.probe)
        ):
            raise ValueError(
                f"{owner}: slot {self.probe_slot!r} must bind this probe definition's "
                "stored atom-level probes in the length unit."
            )
        _require_binding(
            _binding(bindings, self.slot, owner),
            owner,
            level="atom",
            shape=shape,
            unit=scale.force_unit,
            conventions=("coordinate-hessian-vector-product",),
        )

    def label_moments(self, labels: AtomisticHostLabels, /) -> np.ndarray:
        values, available = labels.label(self.slot)
        active = labels.case_valid[:, None] & available & labels.atom_mask
        return _moments(np.asarray(values)[active])

    def term(
        self,
        model: AbstractAtomisticModel,
        component: Any,
        batch: AtomisticTrainingBatch,
        /,
        *,
        scale: float,
        dtype: str,
    ) -> AtomisticObjectiveTerm:
        del component
        operator = atomistic_training_hessian(
            model, batch, request=_STUDENT_HESSIAN_OBSERVABLES, policy=self.policy
        )
        probes, probe_available = batch.label(self.probe_slot)
        labels, available = batch.label(self.slot)
        coordinate = model.precision.coordinate_dtype
        actions = jnp.stack(
            [
                operator.mv(probes[:, :, index, :].astype(coordinate))
                for index in range(self.probe.count)
            ],
            axis=2,
        )
        atoms = batch.case_valid[:, None] & available & probe_available & batch.atom_mask
        weight = jnp.where(atoms, batch.case_weight.astype(dtype)[:, None], 0.0)
        residual = _masked(
            (actions.astype(dtype) - labels.astype(dtype)) / scale,
            atoms[:, :, None, None],
        )
        penalty = jnp.sum(self.loss.pointwise(residual), axis=(2, 3))
        numerator = jnp.sum(weight * penalty, dtype=dtype)
        support = float(3 * self.probe.count) * jnp.sum(weight, dtype=dtype)
        valid = batch.case_valid
        return _term(
            numerator,
            support,
            valid & operator.overflow,
            valid & ~operator.response_valid,
        )


# ------------------------------------------------------------- Boltzmann match


_MIXTURE_REFUSALS = (
    FreeEnergyStatus.POOR_OVERLAP
    | FreeEnergyStatus.DISCONNECTED
    | FreeEnergyStatus.INSUFFICIENT_SAMPLES
    | FreeEnergyStatus.RANK_DEFICIENT
)
_MIXTURE_NUMERICAL_REFUSALS = FreeEnergyStatus.NONCONVERGED | FreeEnergyStatus.NONFINITE


def _logsumexp(values: np.ndarray, axis: int, /) -> np.ndarray:
    peak = np.max(values, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(values - peak), axis=axis))


def _equilibrium_samples(
    result: AtomisticMultistateSegmentResult,
    qualification: AtomisticCanonicalSamplingQualification,
    dynamics: PreparedAtomisticDynamics,
    owner: str,
    /,
) -> AtomisticMultistateEquilibriumSamples:
    """Canonical admission of a segment's draws as configuration-bound samples.

    The multistate owner refuses unsuccessful segments, adapting (SAMS or
    bias-history) draws, unauthenticated frozen histories and wrong-owner bias
    evidence; the samples must have been produced under ``qualification`` (of
    ``dynamics``, with fixed row Hamiltonians) by a segment that captured
    every draw's configuration.
    """
    if not isinstance(result, AtomisticMultistateSegmentResult):
        raise TypeError(f"{owner}: result must be an AtomisticMultistateSegmentResult.")
    if not isinstance(qualification, AtomisticCanonicalSamplingQualification):
        raise TypeError(
            f"{owner}: qualification must be an AtomisticCanonicalSamplingQualification."
        )
    if not isinstance(dynamics, PreparedAtomisticDynamics):
        raise TypeError(f"{owner}: dynamics must be a PreparedAtomisticDynamics.")
    if qualification.dynamics_id != dynamics.prepared_id:
        raise ValueError(f"{owner}: the qualification binds another dynamics runtime.")
    samples = AtomisticMultistateEquilibriumSamples(result)
    if samples.qualification_id != qualification.qualification_id:
        raise ValueError(
            f"{owner}: the samples were not produced under this canonical sampling "
            "qualification."
        )
    if samples.measure_id != qualification.measure_id:
        raise ValueError(f"{owner}: the samples belong to another phase-space measure.")
    if samples.unit_system_id != dynamics.system.plan.units.unit_system_id:
        raise ValueError(f"{owner}: the samples use another unit system.")
    qualification.require_fixed_row_hamiltonians(owner)
    if samples.draw_positions is None or samples.draw_image_counts is None:
        raise ValueError(
            f"{owner}: distribution distillation needs the segment's captured draw "
            "configurations (AtomisticMultistateSegmentPlan with "
            "AtomisticDrawConfigurationCapture); reduced potentials alone do not "
            "establish which configuration a record is."
        )
    return samples


def _generator_evidence(
    samples: AtomisticMultistateEquilibriumSamples, /
) -> dict[str, Any]:
    return {
        "run": samples.run_id,
        "segment": samples.segment_id,
        "producer": samples.producer_id,
        "qualification": samples.qualification_id,
        "sampling_exact": samples.sampling_exact,
        "sampling_bias_bound": float(samples.sampling_bias_bound).hex(),
        "measure": samples.measure_id,
        "convention": samples.reduced_convention_id,
        "states": list(samples.state_ids),
        "potentials": list(samples.potential_ids),
        "biases": list(samples.bias_ids),
        "histories": list(samples.bias_history_digests),
    }


def _frozen_array(value: Any, dtype: Any, /) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@final
class AtomisticGeneratorSamples(StrictModule, NonTrainableState):
    """Captured configurations of a generator's admitted draws on its sampled system.

    For every admitted column (``columns``, draw-major ``n = draw * R +
    replica``) the multistate owner's capture records the exact configuration
    whose reduced potentials were evaluated. ``positions`` are its unwrapped
    Cartesian coordinates (wrapped coordinates plus image counts times the
    drawn cell, the dynamics' own unwrapping) of the system's active atoms in
    system order, ``species`` the lane species codes and ``cells`` the drawn
    cell vectors of a periodic system. Captured values of any coordinate dtype
    (float32 or float64) are first promoted exactly to float64, the canonical
    record dtype, and the unwrapping is evaluated once here in float64;
    canonical records carry these values, so the proof compares float64 with
    float64 bitwise for every supported coordinate dtype.
    The sampled system's chemistry (atomic
    numbers, type IDs, masses, particle IDs, periodic axes) and units come from
    the qualified dynamics. A record is a sample exactly when its geometry and
    identity equal one column (`require_record`).
    """

    positions: np.ndarray
    species: np.ndarray
    cells: np.ndarray | None
    atomic_numbers: np.ndarray
    atom_type_ids: np.ndarray
    masses: np.ndarray
    particle_ids: np.ndarray
    scale: AtomisticScaleContract
    mass_unit: UnitDefinition
    periodic_axes: tuple[bool, ...] | None = eqx.field(static=True)
    columns: tuple[int, ...] = eqx.field(static=True)
    chains: tuple[int, ...] = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    samples_id: str = eqx.field(static=True)

    def __init__(
        self,
        samples: AtomisticMultistateEquilibriumSamples,
        dynamics: PreparedAtomisticDynamics,
        columns: Sequence[int],
        /,
    ) -> None:
        if samples.draw_positions is None or samples.draw_image_counts is None:
            raise ValueError("The segment captured no draw configurations.")
        system = dynamics.system
        plan = system.plan
        chosen = np.asarray(columns, dtype=np.int64)
        active = np.asarray(system.active_mask, dtype=np.bool_)
        wrapped = np.asarray(samples.draw_positions, dtype=np.float64)[chosen]
        images = np.asarray(samples.draw_image_counts, dtype=np.float64)[chosen]
        if samples.draw_cell_vectors is None:
            if system.cell is not None or np.any(images != 0.0):
                raise ValueError("A periodic draw was captured without its cell.")
            cells = None
            unwrapped = wrapped
            periodic = None
        else:
            if system.cell is None:
                raise ValueError("A nonperiodic system draw carries a cell.")
            cells = np.asarray(samples.draw_cell_vectors, dtype=np.float64)[
                chosen
            ].reshape((-1, 3, 3))
            unwrapped = wrapped + images @ cells
            periodic = tuple(bool(value) for value in system.cell.periodic_axes)
        species = np.asarray(samples.draw_species)[chosen][:, active]
        if not np.all(np.isfinite(unwrapped[:, active])) or np.any(species < 0):
            raise ValueError("An admitted draw configuration is not a committed draw.")
        self.positions = _frozen_array(unwrapped[:, active], np.float64)
        self.species = _frozen_array(species, np.int32)
        self.cells = None if cells is None else _frozen_array(cells, np.float64)
        self.atomic_numbers = _frozen_array(
            np.asarray(plan.atomic_numbers)[active], np.int32
        )
        self.atom_type_ids = _frozen_array(
            np.asarray(plan.atom_type_ids)[active], np.int32
        )
        self.masses = _frozen_array(np.asarray(plan.masses)[active], np.float64)
        self.particle_ids = _frozen_array(np.asarray(plan.particle_ids)[active], np.int64)
        self.scale = plan.units.scale
        self.mass_unit = plan.units.mass_unit
        self.periodic_axes = periodic
        self.columns = tuple(int(value) for value in chosen.tolist())
        self.chains = tuple(
            int(value) for value in np.asarray(samples.chain_index)[chosen].tolist()
        )
        self.run_id = samples.run_id
        self.system_id = system.prepared_id
        self.samples_id = canonical_fingerprint(
            {
                "kind": "atomistic-generator-samples",
                "segment": samples.segment_id,
                "system": self.system_id,
                "columns": list(self.columns),
                "chains": list(self.chains),
                "configurations": array_tree_fingerprint(
                    [self.positions, self.species]
                    + ([] if self.cells is None else [self.cells])
                ),
            }
        )

    def row(self, column: int, /) -> int:
        """Position of an admitted column in ``columns``."""
        index = int(np.searchsorted(np.asarray(self.columns), column))
        if index >= len(self.columns) or self.columns[index] != column:
            raise ValueError(f"Column {column} is not an admitted sample.")
        return index

    def species_codes(self, row: int, kind: AtomisticSpeciesKind, /) -> np.ndarray:
        """The species codes of one sample in ``kind`` (atomic numbers or types)."""
        if kind is AtomisticSpeciesKind.ATOMIC_NUMBER:
            if not np.array_equal(self.species[row], self.atom_type_ids):
                raise ValueError(
                    "A draw carries other species codes than its sampled system; its "
                    "atomic numbers are not established."
                )
            return self.atomic_numbers
        return self.species[row]

    def require_record(
        self,
        row: int,
        values: Mapping[str, np.ndarray],
        stable_ids: np.ndarray,
        kind: AtomisticSpeciesKind,
        /,
        *,
        record_id: str,
    ) -> None:
        """Refuse a record that is not exactly the captured sample ``row``.

        Positions (unwrapped, active atoms in system order), species codes,
        masses, particle IDs and, for a periodic system, the cell and periodic
        axes must equal the capture bitwise.
        """
        column = self.columns[row]

        def refuse(what: str, /) -> None:
            raise ValueError(
                f"Record {record_id!r} is not the captured configuration of generator "
                f"column {column} ({what} differ)."
            )

        if not np.array_equal(
            np.asarray(values["positions"], dtype=np.float64), self.positions[row]
        ):
            refuse("positions")
        if not np.array_equal(
            np.asarray(values["species"]), self.species_codes(row, kind)
        ):
            refuse("species")
        if not np.array_equal(
            np.asarray(values["masses"], dtype=np.float64), self.masses
        ):
            refuse("masses")
        if not np.array_equal(np.asarray(stable_ids), self.particle_ids):
            refuse("particle IDs")
        if self.cells is None or self.periodic_axes is None:
            if "cell" in values:
                refuse("cells")
            return
        if "cell" not in values or not np.array_equal(
            np.asarray(values["cell"], dtype=np.float64).reshape((3, 3)), self.cells[row]
        ):
            refuse("cells")
        if not np.array_equal(
            np.asarray(values["periodic"], dtype=np.bool_).reshape(-1),
            np.asarray(self.periodic_axes, dtype=np.bool_),
        ):
            refuse("periodic axes")


def _sample_record_id(generator_id: str, column: int, /) -> str:
    return "sample-" + canonical_fingerprint(
        {
            "kind": "atomistic-generator-sample",
            "generator": generator_id,
            "column": column,
        }
    )


@final
class AtomisticGeneratorState(StrictModule, NonTrainableState):
    """One generating Hamiltonian: a fixed row of an authenticated segment.

    The admitted samples are the canonical equilibrium draws
    (`AtomisticMultistateEquilibriumSamples`) generated at row ``state_index``
    of ``result`` under ``qualification`` of ``dynamics``; with ``selection``
    (rows ``K >= 2``) only the draws retained by its evidence on the segment's
    `ReducedPotentialDataset`. The segment must have captured its draw
    configurations: ``samples`` holds the exact configuration of every
    admitted column. The generator reduced potential of draw ``n`` is the
    owner's ``u_k(x_n)``, including the row's frozen bias at its declared
    owner; it is never supplied by the caller. ``reduced_potentials`` holds it
    for every admitted column and ``None`` otherwise.
    """

    samples: AtomisticGeneratorSamples
    run_id: str = eqx.field(static=True)
    segment_id: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    sampling_exact: bool = eqx.field(static=True)
    sampling_bias_bound: float = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    reduced_convention_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)
    potential_id: str = eqx.field(static=True)
    bias_id: str | None = eqx.field(static=True)
    selection_id: str | None = eqx.field(static=True)
    columns: tuple[int, ...] = eqx.field(static=True)
    reduced_potentials: tuple[float | None, ...] = eqx.field(static=True)
    generator_id: str = eqx.field(static=True)

    def __init__(
        self,
        result: AtomisticMultistateSegmentResult,
        state_index: int,
        /,
        *,
        qualification: AtomisticCanonicalSamplingQualification,
        dynamics: PreparedAtomisticDynamics,
        selection: FreeEnergySelectionPlan | None = None,
    ) -> None:
        owner = "AtomisticGeneratorState"
        samples = _equilibrium_samples(result, qualification, dynamics, owner)
        index = _positive(state_index, "state_index", minimum=0)
        if index >= len(samples.state_ids):
            raise ValueError("state_index names no row of the segment.")
        values = np.asarray(samples.reduced_potentials, dtype=np.float64)
        admitted = (
            np.asarray(samples.sample_active, dtype=np.bool_)
            & (np.asarray(samples.origin_state) == index)
            & np.asarray(samples.coverage, dtype=np.bool_)[index]
        )
        selection_id = None
        if selection is not None:
            if not isinstance(selection, FreeEnergySelectionPlan):
                raise TypeError("selection must be a FreeEnergySelectionPlan.")
            if len(samples.state_ids) < 2:
                raise ValueError(
                    "A selection plan acts on the segment's ReducedPotentialDataset, "
                    "which needs at least two rows; a one-row segment admits every "
                    "equilibrium draw (exclude burn-in by the committed segments used)."
                )
            evidence = selection.select(reduced_potential_dataset_from_multistate(result))
            admitted &= np.asarray(evidence.retained, dtype=np.bool_).reshape(-1)
            selection_id = evidence.selection_id
        columns = tuple(int(value) for value in np.flatnonzero(admitted))
        if not columns:
            raise ValueError("The generating row has no admitted equilibrium sample.")
        if not np.all(np.isfinite(values[index, admitted])):
            raise ValueError("An admitted generator reduced potential is not finite.")
        configurations = AtomisticGeneratorSamples(samples, dynamics, columns)
        self.samples = configurations
        self.run_id = samples.run_id
        self.segment_id = samples.segment_id
        self.qualification_id = samples.qualification_id
        self.sampling_exact = bool(samples.sampling_exact)
        self.sampling_bias_bound = float(samples.sampling_bias_bound)
        self.measure_id = samples.measure_id
        self.reduced_convention_id = samples.reduced_convention_id
        self.state_id = samples.state_ids[index]
        self.potential_id = samples.potential_ids[index]
        self.bias_id = samples.bias_ids[index]
        self.selection_id = selection_id
        self.columns = columns
        self.reduced_potentials = tuple(
            float(values[index, column]) if flag else None
            for column, flag in enumerate(admitted.tolist())
        )
        self.generator_id = canonical_fingerprint(
            {
                "kind": "atomistic-generator-state",
                **_generator_evidence(samples),
                "row": index,
                "selection": selection_id,
                "samples": configurations.samples_id,
                "density": "row-reduced-potential",
            }
        )

    @property
    def energy_unit(self) -> UnitDefinition:
        return self.samples.scale.energy_unit


@final
class AtomisticGeneratorMixture(StrictModule, NonTrainableState):
    """MBAR-normalized mixture of the generating rows of an authenticated segment.

    Samples of different generators share no partition function, so their
    importance ratios cannot be combined directly. The canonical equilibrium
    draws of ``result`` (admitted under ``qualification`` of ``dynamics``, with
    captured configurations) are converted to the segment's
    `ReducedPotentialDataset`; ``selection`` (default: every draw) is resolved
    once to `FreeEnergySelectionEvidence`, the same evidence drives
    `phydrax.uq.multistate_bennett_acceptance_ratio`, and the mixture reduced
    potential ``u_mix(x_n) = -log sum_k (N_k / N) exp(f_k - u_k(x_n))`` uses its
    RETAINED counts ``N_k``. Only retained columns carry a potential and a
    configuration in ``samples``; excluded (burn-in, thinned) columns are
    ``None`` and refused by `atomistic_boltzmann_source`. Poor overlap,
    disconnected states, insufficient samples, rank deficiency,
    nonconvergence or nonfinite estimates are refused; remaining statistical
    bits are retained as evidence.
    """

    samples: AtomisticGeneratorSamples
    dataset_id: str = eqx.field(static=True)
    analysis_id: str = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    reduced_convention_id: str = eqx.field(static=True)
    state_ids: tuple[str, ...] = eqx.field(static=True)
    free_energies: tuple[float, ...] = eqx.field(static=True)
    retained_counts: tuple[int, ...] = eqx.field(static=True)
    mixture_probabilities: tuple[float, ...] = eqx.field(static=True)
    columns: tuple[int, ...] = eqx.field(static=True)
    reduced_potentials: tuple[float | None, ...] = eqx.field(static=True)
    numerical_status: int = eqx.field(static=True)
    statistical_status: int = eqx.field(static=True)
    generator_id: str = eqx.field(static=True)

    def __init__(
        self,
        result: AtomisticMultistateSegmentResult,
        /,
        *,
        qualification: AtomisticCanonicalSamplingQualification,
        dynamics: PreparedAtomisticDynamics,
        selection: FreeEnergySelectionPlan | None = None,
        reference_state: int = 0,
        maximum_iterations: int = 10_000,
        tolerance: float = 1.0e-10,
    ) -> None:
        samples = _equilibrium_samples(
            result, qualification, dynamics, "AtomisticGeneratorMixture"
        )
        if selection is not None and not isinstance(selection, FreeEnergySelectionPlan):
            raise TypeError("selection must be a FreeEnergySelectionPlan or None.")
        dataset = reduced_potential_dataset_from_multistate(result)
        evidence = (FreeEnergySelectionPlan() if selection is None else selection).select(
            dataset
        )
        analysis = multistate_bennett_acceptance_ratio(
            dataset,
            evidence,
            reference_state=reference_state,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )
        if (
            analysis.dataset_id != dataset.dataset_id
            or analysis.selection_id != evidence.selection_id
        ):
            raise RuntimeError(
                "The MBAR analysis is not bound to its selection evidence."
            )
        numerical = FreeEnergyStatus(int(np.asarray(analysis.numerical_status)))
        statistical = FreeEnergyStatus(int(np.asarray(analysis.statistical_status)))
        refused = (numerical & _MIXTURE_NUMERICAL_REFUSALS) | (
            statistical & _MIXTURE_REFUSALS
        )
        if refused:
            raise ValueError(
                f"The generator mixture normalization is not admitted ({refused!r}); "
                "mixed-generator distribution objectives need converged, connected "
                "MBAR evidence with sufficient overlap."
            )
        values = np.asarray(dataset.values, dtype=np.float64)
        retained = np.asarray(evidence.retained, dtype=np.bool_).reshape(-1) & np.asarray(
            dataset.sample_active, dtype=np.bool_
        )
        origin = np.asarray(dataset.origin_state, dtype=np.int64)
        counts = np.bincount(origin[retained], minlength=values.shape[0])
        free = np.asarray(analysis.free_energies, dtype=np.float64)
        total = int(np.sum(counts))
        with np.errstate(divide="ignore"):
            log_mass = np.where(counts > 0, np.log(counts / total), -np.inf)
        mixture = -_logsumexp(log_mass[:, None] + free[:, None] - values, 0)
        columns = tuple(int(value) for value in np.flatnonzero(retained))
        configurations = AtomisticGeneratorSamples(samples, dynamics, columns)
        self.samples = configurations
        self.dataset_id = dataset.dataset_id
        self.analysis_id = analysis.analysis_id
        self.selection_id = evidence.selection_id
        self.qualification_id = samples.qualification_id
        self.measure_id = dataset.measure_id
        self.reduced_convention_id = dataset.reduced_convention_id
        self.state_ids = dataset.state_ids
        self.free_energies = tuple(float(value) for value in free)
        self.retained_counts = tuple(int(value) for value in counts)
        self.mixture_probabilities = tuple(float(value) / total for value in counts)
        self.columns = columns
        self.reduced_potentials = tuple(
            float(value) if flag else None
            for value, flag in zip(mixture, retained.tolist(), strict=True)
        )
        self.numerical_status = int(numerical)
        self.statistical_status = int(statistical)
        self.generator_id = canonical_fingerprint(
            {
                "kind": "atomistic-generator-mixture",
                **_generator_evidence(samples),
                "dataset": dataset.dataset_id,
                "analysis": analysis.analysis_id,
                "selection": evidence.selection_id,
                "retained_counts": list(self.retained_counts),
                "samples": configurations.samples_id,
                "density": "mbar-mixture:-log(sum_k N_k/N exp(f_k-u_k))",
            }
        )

    @property
    def energy_unit(self) -> UnitDefinition:
        return self.samples.scale.energy_unit


def atomistic_generator_source(
    generator: AtomisticGeneratorState | AtomisticGeneratorMixture,
    /,
    *,
    species_kind: AtomisticSpeciesKind = AtomisticSpeciesKind.ATOMIC_NUMBER,
) -> AtomisticStructureSource:
    """The canonical unlabeled records of a generator's admitted samples.

    One record per admitted column, in column order, under
    `atomistic_training_schema` of the sampled system's units: the captured
    unwrapped positions, ``species_kind`` codes, masses and particle IDs of
    the active atoms (and the drawn cell of a periodic system). Record
    ``sample-<digest>`` binds the generator and column; samples of one chain
    share its lineage. Label these records (for example with
    `label_atomistic_structures`) for `atomistic_boltzmann_source`, which
    accepts a record only when it is exactly a captured configuration.
    """
    if not isinstance(generator, (AtomisticGeneratorState, AtomisticGeneratorMixture)):
        raise TypeError(
            "generator must be an AtomisticGeneratorState or an AtomisticGeneratorMixture."
        )
    kind = parse(species_kind, AtomisticSpeciesKind, "species_kind")
    samples = generator.samples
    periodic = samples.cells is not None
    schema = atomistic_training_schema(
        samples.scale,
        mass_unit=samples.mass_unit,
        species_kind=kind,
        periodic=periodic,
        energy=False,
        forces=False,
        stress=False,
    )
    atoms = int(samples.positions.shape[1])
    records = []
    for row, column in enumerate(samples.columns):
        values: dict[str, np.ndarray] = {
            "positions": samples.positions[row],
            "species": samples.species_codes(row, kind),
            "masses": samples.masses,
        }
        if samples.cells is not None and samples.periodic_axes is not None:
            values["cell"] = samples.cells[row][None]
            values["periodic"] = np.asarray(samples.periodic_axes, dtype=np.bool_)[None]
        records.append(
            AtomisticStructureRecord(
                schema,
                record_id=_sample_record_id(generator.generator_id, column),
                lineage_id="chain-"
                + canonical_fingerprint(
                    {"run": samples.run_id, "chain": samples.chains[row]}
                )[:32],
                counts={schema.atom_level: [atoms]},
                stable_ids=samples.particle_ids,
                values=values,
            )
        )
    return AtomisticStructureSource(schema, tuple(records))


@final
class AtomisticBoltzmannGroup(StrictModule, NonTrainableState):
    """One estimation group: target state, chemistry and authenticated generator.

    The target is the unbiased teacher Boltzmann distribution
    ``exp(-beta (E + P V))`` on the generator's exact phase-space measure and
    reduced-potential convention, at inverse temperature ``beta`` and optional
    pressure, for one chemistry (species composition, never just an atom
    count). The generator is ONE authenticated generating row
    (`AtomisticGeneratorState`) or an MBAR `AtomisticGeneratorMixture`; the
    group holds exactly its admitted samples (``size``).
    """

    measure_id: str = eqx.field(static=True)
    reduced_convention_id: str = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    pressure: float | None = eqx.field(static=True)
    composition: tuple[tuple[int, int], ...] = eqx.field(static=True)
    size: int = eqx.field(static=True)
    generator: AtomisticGeneratorState | AtomisticGeneratorMixture
    group_id: str = eqx.field(static=True)

    def __init__(
        self,
        generator: AtomisticGeneratorState | AtomisticGeneratorMixture,
        /,
        *,
        beta: float,
        composition: Mapping[int, int],
        pressure: float | None = None,
    ) -> None:
        if not isinstance(
            generator, (AtomisticGeneratorState, AtomisticGeneratorMixture)
        ):
            raise TypeError(
                "generator must be an AtomisticGeneratorState or an "
                "AtomisticGeneratorMixture."
            )
        beta_ = _finite(beta, "beta")
        if beta_ <= 0.0:
            raise ValueError("beta must be positive.")
        pressure_ = None if pressure is None else _finite(pressure, "pressure")
        if not isinstance(composition, Mapping) or not composition:
            raise ValueError("composition maps species codes to positive atom counts.")
        resolved = tuple(
            sorted(
                (_positive(code, "species", minimum=0), _positive(count, "count"))
                for code, count in composition.items()
            )
        )
        self.measure_id = generator.measure_id
        self.reduced_convention_id = generator.reduced_convention_id
        self.beta = beta_
        self.pressure = pressure_
        self.composition = resolved
        self.size = len(generator.columns)
        self.generator = generator
        self.group_id = canonical_fingerprint(
            {
                "kind": "atomistic-boltzmann-group",
                "measure": generator.measure_id,
                "convention": generator.reduced_convention_id,
                "beta": float(beta_).hex(),
                "pressure": None if pressure_ is None else float(pressure_).hex(),
                "composition": [list(item) for item in resolved],
                "size": self.size,
                "generator": generator.generator_id,
            }
        )

    @classmethod
    def from_table(
        cls,
        table: PreparedThermodynamicStateTable,
        state_index: int,
        /,
        *,
        composition: Mapping[int, int],
        generator: AtomisticGeneratorState | AtomisticGeneratorMixture,
    ) -> AtomisticBoltzmannGroup:
        """The target of one unbiased row of the owning prepared state table.

        A biased row is a generator, not a target: the objective's teacher and
        student reduced potentials are ``beta (E + P V)`` without a bias, so a
        biased target row is refused. The row's measure and reduced convention
        must be the generator's.
        """
        if not isinstance(table, PreparedThermodynamicStateTable):
            raise TypeError("table must be a PreparedThermodynamicStateTable.")
        index = _positive(state_index, "state_index", minimum=0)
        if index >= table.state_count:
            raise ValueError("state_index names no table row.")
        if table.bias_ids[index] is not None:
            raise ValueError(
                "A biased table row cannot be a Boltzmann target: the target is the "
                "unbiased teacher distribution beta (E + P V)."
            )
        if not bool(np.asarray(table.temperature_mask)[index]):
            raise ValueError("A Boltzmann group needs a state with a temperature.")
        if not isinstance(
            generator, (AtomisticGeneratorState, AtomisticGeneratorMixture)
        ):
            raise TypeError(
                "generator must be an AtomisticGeneratorState or an "
                "AtomisticGeneratorMixture."
            )
        if (
            table.phase_space_measure_id != generator.measure_id
            or table.reduced_convention_id != generator.reduced_convention_id
        ):
            raise ValueError(
                "The generator was sampled on another measure or reduced-potential "
                "convention than the target row."
            )
        pressure = (
            float(np.asarray(table.pressure)[index])
            if bool(np.asarray(table.pressure_mask)[index])
            else None
        )
        return cls(
            generator,
            beta=float(np.asarray(table.beta)[index]),
            pressure=pressure,
            composition=composition,
        )


def _groups_id(groups: Sequence[AtomisticBoltzmannGroup], /) -> str:
    return canonical_fingerprint(
        {
            "kind": "atomistic-boltzmann-groups",
            "groups": [group.group_id for group in groups],
        }
    )


def _boltzmann_groups(
    groups: Sequence[AtomisticBoltzmannGroup], /
) -> tuple[AtomisticBoltzmannGroup, ...]:
    resolved = tuple(groups)
    if not resolved or any(
        not isinstance(group, AtomisticBoltzmannGroup) for group in resolved
    ):
        raise TypeError("groups must be nonempty AtomisticBoltzmannGroup values.")
    if len({group.group_id for group in resolved}) != len(resolved):
        raise ValueError("Boltzmann groups must be distinct.")
    return resolved


def _require_boltzmann_units(
    schema: AtomisticStructureStoreSchema,
    groups: Sequence[AtomisticBoltzmannGroup],
    energy_slot: str,
    /,
) -> UnitDefinition:
    """Admit the new slots and every group's energy, length and mass units.

    Returns the energy slot's unit, which is also the ``P V`` unit.
    """
    energy_field = schema.field(energy_slot)
    for name in (BOLTZMANN_GROUP_SLOT, BOLTZMANN_GENERATOR_SLOT, BOLTZMANN_PV_SLOT):
        if name in schema.field_names:
            raise ValueError(f"The source already holds {name!r}.")
    if energy_field.unit is None:
        raise ValueError("The Boltzmann energy slot needs an energy unit.")
    for position, group in enumerate(groups):
        if energy_field.unit.unit_id != group.generator.energy_unit.unit_id:
            raise ValueError(
                f"Group {position}'s generator reduced potentials use another energy "
                "unit than the energy slot."
            )
        samples = group.generator.samples
        if group.pressure is not None and samples.cells is None:
            raise ValueError("A pressure group needs periodic samples for P V.")
        for field, unit in (
            ("positions", samples.scale.length_unit),
            ("masses", samples.mass_unit),
        ):
            declared = schema.field(field).unit
            if declared is None or declared.unit_id != unit.unit_id:
                raise ValueError(
                    f"Records store {field!r} in another unit than group {position}'s "
                    "sampled system."
                )
    return energy_field.unit


def _require_complete_groups(
    labeled: AbstractAtomisticStructureSource,
    groups: Sequence[AtomisticBoltzmannGroup],
    members: Sequence[int],
    chosen: Sequence[int],
    /,
) -> None:
    """Admit every claim as a distinct admitted column; every group is complete."""
    used: list[set[int]] = [set() for _ in groups]
    for record, (position, column) in enumerate(zip(members, chosen, strict=True)):
        if not 0 <= position < len(groups):
            raise ValueError(f"Record {labeled.record_ids[record]!r} names no group.")
        potentials = groups[position].generator.reduced_potentials
        if not 0 <= column < len(potentials) or potentials[column] is None:
            raise ValueError(
                f"Record {labeled.record_ids[record]!r} names column {column}, which is "
                "not an admitted sample of its group's generator (inactive, adapting, "
                "another row or excluded by the selection evidence)."
            )
        if column in used[position]:
            raise ValueError("A generator column is used twice.")
        used[position].add(column)
    for position, group in enumerate(groups):
        if len(used[position]) != group.size:
            raise ValueError(
                f"Group {position} holds {len(used[position])} of the "
                f"{group.size} admitted samples of its generator; a group is the "
                "complete admitted sample set."
            )


def _proven_pressure_volumes(
    labeled: AbstractAtomisticStructureSource,
    groups: Sequence[AtomisticBoltzmannGroup],
    members: Sequence[int],
    chosen: Sequence[int],
    kind: AtomisticSpeciesKind,
    chunk: int,
    /,
) -> np.ndarray:
    """Prove every claim against its captured configuration; per-record ``P V``.

    Bounded: only the canonical geometry and identity rows are read, ``chunk``
    records at a time, before anything is published.
    """
    count = labeled.record_count
    projection = ("positions", "species", "masses") + (
        ("cell", "periodic") if "cell" in labeled.schema.field_names else ()
    )
    pv = np.zeros((count,), dtype=np.float64)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        batch = labeled.read(np.arange(start, stop, dtype=np.int64), fields=projection)
        for local, record in enumerate(range(start, stop)):
            group = groups[members[record]]
            samples = group.generator.samples
            values, _, stable_ids, _ = _record_columns(batch, local)
            samples.require_record(
                samples.row(chosen[record]),
                values,
                stable_ids,
                kind,
                record_id=labeled.record_ids[record],
            )
            composition = tuple(
                sorted(
                    Counter(int(code) for code in values["species"].reshape(-1)).items()
                )
            )
            if composition != group.composition:
                raise ValueError(
                    f"Record {labeled.record_ids[record]!r} has composition "
                    f"{composition} but group {members[record]} declares "
                    f"{group.composition}; equal atom counts of another chemistry are "
                    "another system."
                )
            if group.pressure is not None:
                cell = np.asarray(values["cell"], dtype=np.float64).reshape((3, 3))
                pv[record] = group.pressure * abs(float(np.linalg.det(cell)))
    return pv


def _boltzmann_schema(
    schema: AtomisticStructureStoreSchema,
    groups_id: str,
    energy_unit: UnitDefinition,
    /,
) -> AtomisticStructureStoreSchema:
    """The labeled schema extended with group, generator density and ``P V``."""
    structure = schema.structure_level
    return AtomisticStructureStoreSchema(
        schema.levels,
        (
            *schema.fields,
            AtomisticStoreField(
                BOLTZMANN_GROUP_SLOT,
                structure,
                "int32",
                kind="categorical",
                role="label",
                convention=f"boltzmann-group;groups={groups_id}",
            ),
            AtomisticStoreField(
                BOLTZMANN_GENERATOR_SLOT,
                structure,
                "float64",
                kind="continuous",
                role="label",
                convention="generator-reduced-potential",
                unit=ONE,
                extensivity="intensive",
            ),
            AtomisticStoreField(
                BOLTZMANN_PV_SLOT,
                structure,
                "float64",
                kind="continuous",
                role="label",
                convention="pressure-volume",
                unit=energy_unit,
                extensivity="extensive",
            ),
        ),
    )


def _store_boltzmann_records(
    store: AtomisticStructureStoreWriter,
    name: str,
    extended: AtomisticStructureStoreSchema,
    labeled: AbstractAtomisticStructureSource,
    groups: Sequence[AtomisticBoltzmannGroup],
    members: Sequence[int],
    chosen: Sequence[int],
    pv: np.ndarray,
    /,
) -> None:
    """Append the proven records with their Boltzmann columns in bounded chunks.

    The store named ``name`` resumes from its committed prefix; a complete
    store is left unchanged.
    """
    count = labeled.record_count
    chunk = store.layout.records_per_shard
    head = _resumable_head(
        store,
        name,
        (extended,),
        lambda index: (labeled.record_ids[index], labeled.lineage_ids[index]),
        count,
    )
    parent = None if head is None else head.commit
    for start in range(0 if head is None else head.record_count, count, chunk):
        stop = min(start + chunk, count)
        base = labeled.read(np.arange(start, stop, dtype=np.int64))
        records = []
        for local, record in enumerate(range(start, stop)):
            position = members[record]
            reduced = groups[position].generator.reduced_potentials[chosen[record]]
            if reduced is None:
                raise RuntimeError("An admitted generator column lost its potential.")
            records.append(
                _extended_record(
                    extended,
                    base,
                    local,
                    record_id=labeled.record_ids[record],
                    lineage_id=labeled.lineage_ids[record],
                    extra={
                        BOLTZMANN_GROUP_SLOT: np.asarray([position], dtype=np.int32),
                        BOLTZMANN_GENERATOR_SLOT: np.asarray([reduced], dtype=np.float64),
                        BOLTZMANN_PV_SLOT: np.asarray([pv[record]], dtype=np.float64),
                    },
                    extra_available={},
                )
            )
        parent = _append_records(store, parent, extended, name, records)
    if parent is None:
        _append_records(store, parent, extended, name, ())


def atomistic_boltzmann_source(
    labeled: AbstractAtomisticStructureSource,
    groups: Sequence[AtomisticBoltzmannGroup],
    /,
    *,
    membership: Sequence[int],
    columns: Sequence[int],
    store: AtomisticStructureStoreWriter,
    energy_slot: str = TEACHER_ENERGY_SLOT,
) -> AtomisticStoredStructureSource:
    """Attach group membership, generator density and ``P V`` into a store.

    Record ``i`` claims generator column ``columns[i]`` of
    ``groups[membership[i]]``, and the claim is proven, not declared: its
    canonical geometry/identity fields (``positions``, ``species``,
    ``masses``, stable particle IDs and, for a periodic system, ``cell`` and
    ``periodic``) must equal that column's captured configuration bitwise
    (`AtomisticGeneratorSamples.require_record`), in the sampled system's
    length and mass units. A relaxation frame, a configuration of another
    column or any perturbed geometry is refused. Records built by
    `atomistic_generator_source` and labeled by the teacher satisfy it. Every
    group must hold exactly its generator's admitted columns, each once: an
    excluded (inactive, adapting, other-row or not retained by the selection)
    column or an omitted admitted column is refused, so no caller-supplied
    density or subset enters. The generator reduced potential is the
    authenticated owner value of that column. Every record's species
    composition must equal its group's; ``P V`` uses the group pressure and
    the record (= sampled) cell volume, zero without a pressure; the energy
    unit must be the generator's.

    Records are proven in a bounded first pass (only the canonical geometry
    and identity rows, ``store.layout.records_per_shard`` records at a time)
    and then appended to ``store`` in chunks of that size, so a refused source
    publishes nothing. The store is named by the output identity; an
    interrupted run resumes from its committed prefix and a complete store is
    returned unchanged.
    """
    if not isinstance(labeled, AbstractAtomisticStructureSource):
        raise TypeError("labeled must be an atomistic structure source.")
    if not isinstance(store, AtomisticStructureStoreWriter):
        raise TypeError("store must be an AtomisticStructureStoreWriter.")
    schema = labeled.schema
    _canonical_levels(schema)
    resolved = _boltzmann_groups(groups)
    count = labeled.record_count
    members = tuple(int(value) for value in membership)
    chosen = tuple(int(value) for value in columns)
    if not len(members) == len(chosen) == count:
        raise ValueError("membership and columns name every record.")
    energy_unit = _require_boltzmann_units(schema, resolved, energy_slot)
    kind = AtomisticSpeciesKind(schema.field("species").convention)
    _require_complete_groups(labeled, resolved, members, chosen)
    pv = _proven_pressure_volumes(
        labeled, resolved, members, chosen, kind, store.layout.records_per_shard
    )
    extended = _boltzmann_schema(schema, _groups_id(resolved), energy_unit)
    name = "boltzmann-" + canonical_fingerprint(
        {
            "kind": "atomistic-boltzmann-source",
            "labeled": labeled.source_id,
            "schema": extended.schema_id,
            "membership": list(members),
            "columns": list(chosen),
            "energy_slot": energy_slot,
        }
    )
    _store_boltzmann_records(
        store, name, extended, labeled, resolved, members, chosen, pv
    )
    return open_atomistic_structure_store(store.root, limits=store.limits)


def _masked_logsumexp(values: Array, member: Array, present: Array, /) -> Array:
    """Per-group logsumexp over member cases; absent groups read finite zeros."""
    filler = jnp.where(present[None, :], -jnp.inf, 0.0).astype(values.dtype)
    return jax.nn.logsumexp(jnp.where(member, values, filler), axis=0)


@final
class AtomisticBoltzmannObjective(AbstractAtomisticObjective):
    """Self-normalized finite-sample KL estimator over complete sample groups.

    For each group (one measure, chemistry, state and generating density
    ``q``) with reduced potentials ``u_p = beta (E_teacher + P V)``,
    ``u_s = beta (E_student + P V)`` and generator ``u_q``, the forward
    estimator is ``sum_i wbar_i (u_s - u_p)_i + log sum_i w_i e^{-(u_s-u_q)_i}
    - log sum_i w_i e^{-(u_p-u_q)_i}`` with ``wbar`` the self-normalized weights
    ``∝ w_i e^{-(u_p - u_q)_i}``, and the reverse estimator the same with
    teacher and student exchanged. The objective is ``lambda F + (1 - lambda) R``
    with ``lambda = forward_weight``. This is a consistent but biased
    finite-sample estimator of the population KL, not the population KL. The
    sampled coordinates are fixed data: gradients flow through the student
    reduced potentials and every self-normalized weight, never through
    sampling. The estimator is not additive over cases: its support counts
    groups and its numerator sums per-group estimates, so every group must be
    complete in one payload (one draw per sample); a partial group, an
    undeclared chemistry or a failed evaluation poisons the update.
    ``w_i`` is the loader case weight.
    """

    name: str = eqx.field(static=True)
    groups: tuple[AtomisticBoltzmannGroup, ...]
    forward_weight: float = eqx.field(static=True)
    energy_slot: str = eqx.field(static=True)
    group_slot: str = eqx.field(static=True)
    generator_slot: str = eqx.field(static=True)
    pv_slot: str = eqx.field(static=True)
    scale: float | None = eqx.field(static=True)
    schedule: AtomisticObjectiveSchedule
    objective_id: str = eqx.field(static=True)

    def __init__(
        self,
        groups: Sequence[AtomisticBoltzmannGroup],
        /,
        *,
        forward_weight: float,
        energy_slot: str = TEACHER_ENERGY_SLOT,
        group_slot: str = BOLTZMANN_GROUP_SLOT,
        generator_slot: str = BOLTZMANN_GENERATOR_SLOT,
        pv_slot: str = BOLTZMANN_PV_SLOT,
        name: str = "boltzmann",
        scale: float = 1.0,
        schedule: AtomisticObjectiveSchedule | None = None,
    ) -> None:
        resolved = _boltzmann_groups(groups)
        weight = _finite(forward_weight, "forward_weight")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("forward_weight must lie in [0, 1].")
        self.name = _name(name, "name")
        self.groups = resolved
        self.forward_weight = weight
        self.energy_slot = _name(energy_slot, "energy_slot")
        self.group_slot = _name(group_slot, "group_slot")
        self.generator_slot = _name(generator_slot, "generator_slot")
        self.pv_slot = _name(pv_slot, "pv_slot")
        self.scale = _finite_positive(scale, "scale")
        self.schedule = _schedule(schedule)
        self.objective_id = _objective_identity(
            "atomistic-boltzmann-objective",
            name=self.name,
            groups=_groups_id(resolved),
            forward_weight=float(weight).hex(),
            energy_slot=self.energy_slot,
            group_slot=self.group_slot,
            generator_slot=self.generator_slot,
            pv_slot=self.pv_slot,
            scale=self.scale,
            schedule=self.schedule.schedule_id,
            estimator="self-normalized-finite-sample-kl;fixed-sampled-coordinates",
        )

    @property
    def label_slots(self) -> tuple[str, ...]:
        return (self.energy_slot, self.group_slot, self.generator_slot, self.pv_slot)

    def admit(
        self,
        model: AbstractAtomisticModel,
        bindings: Mapping[str, AtomisticFieldBinding],
        scale: AtomisticScaleContract,
        /,
    ) -> None:
        owner = f"Boltzmann objective {self.name!r}"
        _require_energy_model(model, owner)
        _require_model_scale(model, scale, owner)
        _require_binding(
            _binding(bindings, self.energy_slot, owner),
            owner,
            level="structure",
            shape=(),
            unit=scale.energy_unit,
        )
        _require_binding(
            _binding(bindings, self.generator_slot, owner),
            owner,
            level="structure",
            shape=(),
            unit=ONE,
            conventions=("generator-reduced-potential",),
        )
        _require_binding(
            _binding(bindings, self.pv_slot, owner),
            owner,
            level="structure",
            shape=(),
            unit=scale.energy_unit,
            conventions=("pressure-volume",),
        )
        group = _binding(bindings, self.group_slot, owner)
        if (
            group.level_kind != "structure"
            or group.value_kind != "categorical"
            or group.axes
            or group.convention != f"boltzmann-group;groups={_groups_id(self.groups)}"
        ):
            raise ValueError(
                f"{owner}: slot {self.group_slot!r} must bind the membership of exactly "
                "these groups."
            )

    def label_moments(self, labels: AtomisticHostLabels, /) -> np.ndarray:
        del labels
        return np.zeros((3,), dtype=np.float64)

    def term(
        self,
        model: AbstractAtomisticModel,
        component: Any,
        batch: AtomisticTrainingBatch,
        /,
        *,
        scale: float,
        dtype: str,
    ) -> AtomisticObjectiveTerm:
        del component
        derivatives = atomistic_training_derivatives(model, batch, forces=False)
        teacher, teacher_available = batch.label(self.energy_slot)
        membership, group_available = batch.label(self.group_slot)
        generator, generator_available = batch.label(self.generator_slot)
        pv, pv_available = batch.label(self.pv_slot)
        count = len(self.groups)
        alphabet = sorted(
            {code for group in self.groups for code, _ in group.composition}
        )
        table = np.zeros((count, len(alphabet)), dtype=np.int32)
        for row, group in enumerate(self.groups):
            for code, atoms in group.composition:
                table[row, alphabet.index(code)] = atoms
        sizes = jnp.asarray([group.size for group in self.groups], dtype=jnp.int32)
        betas = jnp.asarray([group.beta for group in self.groups], dtype=dtype)
        valid = (
            batch.case_valid
            & teacher_available
            & group_available
            & generator_available
            & pv_available
        )
        index = jnp.clip(membership.astype(jnp.int32), 0, count - 1)
        declared = (membership >= 0) & (membership < count)
        species = batch.species
        counts = jnp.stack(
            [
                jnp.sum(batch.atom_mask & (species == code), axis=1, dtype=jnp.int32)
                for code in alphabet
            ],
            axis=1,
        )
        chemistry = (
            declared
            & jnp.all(counts == jnp.asarray(table)[index], axis=1)
            & (
                jnp.sum(batch.atom_mask, axis=1, dtype=jnp.int32)
                == jnp.sum(jnp.asarray(table)[index], axis=1)
            )
        )
        member = (valid & declared)[:, None] & (
            index[:, None] == jnp.arange(count, dtype=jnp.int32)[None, :]
        )
        drawn = jnp.sum(member, axis=0, dtype=jnp.int32)
        present = drawn > 0
        complete = drawn == sizes
        beta = betas[index]
        reduced_teacher = beta * (teacher.astype(dtype) + pv.astype(dtype))
        reduced_student = beta * (derivatives.energy.astype(dtype) + pv.astype(dtype))
        reduced_generator = generator.astype(dtype)
        log_weight = jnp.log(jnp.where(valid, batch.case_weight.astype(dtype), 1.0))
        teacher_log = (-(reduced_teacher - reduced_generator) + log_weight)[:, None]
        student_log = (-(reduced_student - reduced_generator) + log_weight)[:, None]
        teacher_norm = _masked_logsumexp(
            jnp.broadcast_to(teacher_log, member.shape), member, present
        )
        student_norm = _masked_logsumexp(
            jnp.broadcast_to(student_log, member.shape), member, present
        )
        safe_teacher = jnp.where(present, teacher_norm, 0.0)
        safe_student = jnp.where(present, student_norm, 0.0)
        forward_weights = jnp.where(
            member, jnp.exp(teacher_log - safe_teacher[None, :]), 0.0
        )
        reverse_weights = jnp.where(
            member, jnp.exp(student_log - safe_student[None, :]), 0.0
        )
        gap = (reduced_student - reduced_teacher)[:, None]
        forward = jnp.sum(forward_weights * gap, axis=0) + (safe_student - safe_teacher)
        reverse = jnp.sum(-reverse_weights * gap, axis=0) + (safe_teacher - safe_student)
        estimate = self.forward_weight * forward + (1.0 - self.forward_weight) * reverse
        usable = present & complete
        numerator = jnp.sum(jnp.where(usable, estimate, 0.0), dtype=dtype) / scale
        support = jnp.sum(present, dtype=dtype)
        incomplete = jnp.any(member & (present & ~complete)[None, :], axis=1)
        case_valid = batch.case_valid
        return _term(
            numerator,
            support,
            case_valid & derivatives.overflow,
            case_valid & (~derivatives.successful | ~valid | ~chemistry | incomplete),
        )


# ----------------------------------------------------------- experience buffer


@final
class AtomisticExperienceAdmission(StrictModule, NonTrainableState):
    """Admission rule of on-policy frames into the experience buffer.

    A frame is admitted when its stage outcome is one of ``outcomes``, it
    carries positions, an atom mask and species, and (with
    ``require_evaluation``) finite generator energy and forces whose largest
    atomic force norm does not exceed ``maximum_force``. Refused frames are
    recorded as durable evidence so their publication stays idempotent.
    """

    outcomes: tuple[int, ...] = eqx.field(static=True)
    require_evaluation: bool = eqx.field(static=True)
    maximum_force: float | None = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        outcomes: Sequence[AtomisticStageOutcome] = (AtomisticStageOutcome.COMPLETED,),
        require_evaluation: bool = True,
        maximum_force: float | None = None,
    ) -> None:
        resolved = tuple(
            sorted({int(AtomisticStageOutcome(value)) for value in outcomes})
        )
        if not resolved:
            raise ValueError("At least one admitted outcome is required.")
        if not isinstance(require_evaluation, bool):
            raise TypeError("require_evaluation must be a boolean.")
        force = _finite_positive(maximum_force, "maximum_force")
        if force is not None and not require_evaluation:
            raise ValueError("maximum_force needs the generator evaluation.")
        self.outcomes = resolved
        self.require_evaluation = require_evaluation
        self.maximum_force = force
        self.admission_id = canonical_fingerprint(
            {
                "kind": "atomistic-experience-admission",
                "outcomes": list(resolved),
                "require_evaluation": require_evaluation,
                "maximum_force": None if force is None else float(force).hex(),
            }
        )

    def refusal(self, record: AtomisticCampaignOutput, /) -> str | None:
        """The refusal reason of one output, or ``None`` when admitted."""
        ports = record.ports
        if int(record.outcome) not in self.outcomes:
            return f"outcome-{AtomisticStageOutcome(record.outcome).name.lower()}"
        if "positions" not in ports or "atom_mask" not in ports:
            return "missing-geometry"
        if "species" not in ports and "atom_type_ids" not in ports:
            return "missing-species"
        if self.require_evaluation:
            if "energy" not in ports or "forces" not in ports:
                return "missing-evaluation"
            energy = np.asarray(ports["energy"], dtype=np.float64)
            forces = np.asarray(ports["forces"], dtype=np.float64)
            if not (np.all(np.isfinite(energy)) and np.all(np.isfinite(forces))):
                return "nonfinite-evaluation"
            if self.maximum_force is not None:
                mask = np.asarray(ports["atom_mask"], dtype=np.bool_)
                norms = np.linalg.norm(forces.reshape((-1, 3))[mask], axis=1)
                if norms.size and float(np.max(norms)) > self.maximum_force:
                    return "force-above-maximum"
        return None


@final
class AtomisticReplayDraw(StrictModule, NonTrainableState):
    """Explicit replay draw probabilities by policy age.

    A frame generated by the policy revision ``a`` revisions before the current
    one is repeated ``age_repeats[min(a, len - 1)]`` times per replay epoch
    (``0`` excludes it); its exact per-draw probability is that repeat count
    over the epoch draw count. Records of one age class are drawn uniformly.
    """

    age_repeats: tuple[int, ...] = eqx.field(static=True)
    draw_id: str = eqx.field(static=True)

    def __init__(self, age_repeats: Sequence[int] = (1,), /) -> None:
        resolved = tuple(
            _positive(value, "age repeat", minimum=0) for value in age_repeats
        )
        if not resolved or max(resolved) == 0:
            raise ValueError("age_repeats needs a positive repeat count.")
        self.age_repeats = resolved
        self.draw_id = canonical_fingerprint(
            {"kind": "atomistic-replay-draw", "age_repeats": list(resolved)}
        )

    def repeats(self, age: int, /) -> int:
        return self.age_repeats[
            min(_positive(age, "age", minimum=0), len(self.age_repeats) - 1)
        ]


@final
@dataclass(frozen=True, slots=True, eq=False)
class AtomisticFrameLabel:
    """Teacher label arrays of one frame under one label definition."""

    label_id: str
    values: Mapping[str, np.ndarray]
    available: Mapping[str, np.ndarray]


@final
@dataclass(frozen=True, slots=True, eq=False)
class AtomisticExperienceFrame:
    """One admitted on-policy frame: campaign identity, ports and labels."""

    semantic_key: str
    payload_id: str
    campaign_id: str
    task_id: str
    material_id: str
    stage: str
    stage_entry: int
    outcome: int
    policy_id: str
    sequence: int
    ports: Mapping[str, np.ndarray]
    labels: tuple[AtomisticFrameLabel, ...]

    def label(self, label_id: str, /) -> AtomisticFrameLabel | None:
        for value in self.labels:
            if value.label_id == label_id:
                return value
        return None


@final
@dataclass(frozen=True, slots=True)
class AtomisticEvictionRecord:
    """FIFO eviction evidence: the frame and the revision that evicted it."""

    semantic_key: str
    payload_id: str
    sequence: int
    revision_index: int


@final
@dataclass(frozen=True, slots=True)
class AtomisticRefusalRecord:
    """Admission refusal evidence of one published output."""

    semantic_key: str
    payload_id: str
    reason: str


@final
@dataclass(frozen=True, slots=True, eq=False)
class AtomisticReplayRevision:
    """One immutable buffer revision; ``lineage`` lists every revision ID.

    ``change`` is the canonical transition that produced this revision from
    ``lineage[-2]`` (``None`` for the genesis); ``revision_id`` is the
    fingerprint of that parent and change, so the latest transition is
    authenticated by its parent.
    """

    index: int
    revision_id: str
    lineage: tuple[str, ...]
    frames: tuple[AtomisticExperienceFrame, ...]
    evicted: tuple[AtomisticEvictionRecord, ...]
    refused: tuple[AtomisticRefusalRecord, ...]
    change: Mapping[str, Any] | None
    next_sequence: int


def _policy_id(revision_ids: Sequence[str], /) -> str:
    return canonical_fingerprint(
        {"kind": "atomistic-generating-policy", "revisions": list(revision_ids)}
    )


def _genesis_id(
    sink_id: str, capacity: int, admission: AtomisticExperienceAdmission, /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "atomistic-experience-buffer-genesis",
            "sink": sink_id,
            "capacity": capacity,
            "admission": admission.admission_id,
        }
    )


def _revision_id(parent: str, change: Mapping[str, Any], /) -> str:
    return canonical_fingerprint(
        {
            "kind": "atomistic-experience-buffer-revision",
            "parent": parent,
            "change": dict(change),
        }
    )


def _label_digest(label: AtomisticFrameLabel, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "atomistic-frame-label",
            "label": label.label_id,
            "values": array_tree_fingerprint(dict(label.values)),
            "available": array_tree_fingerprint(dict(label.available)),
        }
    )


def _frozen_ports(ports: Mapping[str, Any], /) -> Mapping[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for name in sorted(ports):
        array = np.array(ports[name], copy=True)
        array.setflags(write=False)
        values[name] = array
    return MappingProxyType(values)


@final
class AtomisticExperienceBuffer(AbstractAtomisticCampaignSink):
    """Bounded FIFO experience replay of on-policy frames; a campaign sink.

    ``publish`` is durable before it returns: every accepted, refused or
    evicting transition atomically replaces the buffer archive under ``root``
    and only then updates the in-memory revision. Publication is idempotent by
    ``record.semantic_key`` (retained, evicted or refused); a different payload
    under a known key raises. When more than ``capacity`` frames are retained
    the oldest admitted frame is evicted with recorded evidence. Teacher
    labels attach to retained frames per label definition and are never
    duplicated; a label attachment records the content digest of every
    attached label, so `label_successor` authenticates an attachment that was
    durably committed before its consumer's own checkpoint. Walker dynamics
    continuation is owned by the campaign, not by this sink. This is
    experience replay for training, unrelated to `AtomisticReplayPolicy`.
    """

    __slots__ = ("_admission", "_capacity", "_path", "_revision", "sink_id")

    def __init__(
        self,
        root: str | Path,
        /,
        *,
        capacity: int,
        admission: AtomisticExperienceAdmission | None = None,
        sink_id: str = "distillation-replay",
    ) -> None:
        admission_ = AtomisticExperienceAdmission() if admission is None else admission
        if not isinstance(admission_, AtomisticExperienceAdmission):
            raise TypeError("admission must be an AtomisticExperienceAdmission.")
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)
        self._path = directory / _BUFFER_FILE
        self._capacity = _positive(capacity, "capacity")
        self._admission = admission_
        self.sink_id = _name(sink_id, "sink_id")
        genesis = _genesis_id(self.sink_id, self._capacity, admission_)
        if self._path.exists():
            self._revision = self._read(genesis)
        else:
            self._revision = AtomisticReplayRevision(
                0, genesis, (genesis,), (), (), (), None, 0
            )

    # -- properties ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def admission(self) -> AtomisticExperienceAdmission:
        return self._admission

    @property
    def revision(self) -> AtomisticReplayRevision:
        return self._revision

    @property
    def location(self) -> Path:
        return self._path

    def require_revision(self, revision_id: str, /, *, exact: bool) -> None:
        """Refuse a stale or foreign committed revision.

        ``exact`` requires the current revision; otherwise the committed
        revision must be an ancestor of the current one (frames published after
        the commit by an interrupted rollout).
        """
        current = self._revision
        if exact and current.revision_id != revision_id:
            raise ValueError(
                "The experience buffer changed after the committed revision; the "
                "committed replay revision is stale."
            )
        if revision_id not in current.lineage:
            raise ValueError(
                "The committed replay revision is not in this buffer's lineage."
            )

    def label_successor(
        self, parent: str, label_id: str, /
    ) -> AtomisticReplayRevision | None:
        """The current revision when it is the label attachment committed on ``parent``.

        Authenticated exactly: the current revision is the direct successor of
        ``parent``, its recorded change is a label attachment of ``label_id``
        only whose fingerprint with ``parent`` reproduces the revision ID, every
        attached label is held by its retained frame with the recorded content
        digest, and every retained frame holds ``label_id``. Any other
        revision (a publication, another definition, a later change) returns
        ``None``.
        """
        current = self._revision
        change = current.change
        if (
            len(current.lineage) < 2
            or current.lineage[-2] != parent
            or change is None
            or set(change) != {"labels"}
            or current.revision_id != _revision_id(parent, change)
        ):
            return None
        frames = {frame.semantic_key: frame for frame in current.frames}
        for key, attached, digest in change["labels"]:
            frame = frames.get(key)
            label = None if frame is None else frame.label(attached)
            if attached != label_id or label is None or _label_digest(label) != digest:
                return None
        if any(frame.label(label_id) is None for frame in current.frames):
            return None
        return current

    # -- sink protocol ---------------------------------------------------------------

    def receipt(self, semantic_key: str, /) -> AtomisticOutputReceipt | None:
        known = self._known(semantic_key)
        if known is None:
            return None
        payload, state = known
        return AtomisticOutputReceipt(
            semantic_key, self.sink_id, payload, f"{self._path}#{state}"
        )

    def publish(
        self, record: AtomisticCampaignOutput, /, *, round_index: int
    ) -> AtomisticOutputReceipt | None:
        del round_index
        if not isinstance(record, AtomisticCampaignOutput):
            raise TypeError("record must be an AtomisticCampaignOutput.")
        known = self._known(record.semantic_key)
        if known is not None:
            if known[0] != record.payload_id:
                raise ValueError(
                    "A different payload is already published under this semantic key."
                )
            return self.receipt(record.semantic_key)
        current = self._revision
        reason = self._admission.refusal(record)
        frames = list(current.frames)
        evicted = list(current.evicted)
        refused = list(current.refused)
        sequence = current.next_sequence
        index = current.index + 1
        if reason is None:
            frames.append(
                AtomisticExperienceFrame(
                    semantic_key=record.semantic_key,
                    payload_id=record.payload_id,
                    campaign_id=record.campaign_id,
                    task_id=record.task_id,
                    material_id=record.material_id,
                    stage=record.stage,
                    stage_entry=int(record.stage_entry),
                    outcome=int(record.outcome),
                    policy_id=_policy_id(record.numeric_revision_ids),
                    sequence=sequence,
                    ports=_frozen_ports(record.ports),
                    labels=(),
                )
            )
            sequence += 1
            while len(frames) > self._capacity:
                oldest = frames.pop(0)
                evicted.append(
                    AtomisticEvictionRecord(
                        oldest.semantic_key, oldest.payload_id, oldest.sequence, index
                    )
                )
        else:
            refused.append(
                AtomisticRefusalRecord(record.semantic_key, record.payload_id, reason)
            )
        change = {
            "publish": record.semantic_key,
            "payload": record.payload_id,
            "refused": reason,
        }
        self._commit(current, change, frames, evicted, refused, sequence)
        return self.receipt(record.semantic_key)

    # -- labels ----------------------------------------------------------------------

    def attach_labels(
        self, labels: Mapping[str, AtomisticFrameLabel], /, *, parent: str
    ) -> AtomisticReplayRevision:
        """Durably attach teacher labels to retained frames of revision ``parent``.

        Keys are frame semantic keys; a frame already labeled under the same
        label definition is refused, so labels are never duplicated. The
        committed change records every attached label's content digest.
        """
        current = self._revision
        if current.revision_id != parent:
            raise ValueError("Labels attach to a stale buffer revision.")
        if not labels:
            return current
        frames = []
        attached = set()
        for frame in current.frames:
            label = labels.get(frame.semantic_key)
            if label is None:
                frames.append(frame)
                continue
            if not isinstance(label, AtomisticFrameLabel):
                raise TypeError("labels must map semantic keys to AtomisticFrameLabel.")
            if frame.label(label.label_id) is not None:
                raise ValueError(
                    f"Frame {frame.semantic_key!r} already holds label {label.label_id!r}."
                )
            attached.add(frame.semantic_key)
            frames.append(
                AtomisticExperienceFrame(
                    **{
                        name: getattr(frame, name)
                        for name in AtomisticExperienceFrame.__dataclass_fields__
                        if name != "labels"
                    },
                    labels=(*frame.labels, label),
                )
            )
        missing = set(labels) - attached
        if missing:
            raise ValueError(
                f"Labels name frames that are not retained: {sorted(missing)!r}."
            )
        change = {
            "labels": [
                [key, label.label_id, _label_digest(label)]
                for key, label in sorted(labels.items())
            ],
        }
        self._commit(
            current,
            change,
            frames,
            list(current.evicted),
            list(current.refused),
            current.next_sequence,
        )
        return self._revision

    # -- persistence -----------------------------------------------------------------

    def _known(self, semantic_key: str, /) -> tuple[str, str] | None:
        current = self._revision
        for frame in current.frames:
            if frame.semantic_key == semantic_key:
                return frame.payload_id, "retained"
        for item in current.evicted:
            if item.semantic_key == semantic_key:
                return item.payload_id, "evicted"
        for item in current.refused:
            if item.semantic_key == semantic_key:
                return item.payload_id, "refused"
        return None

    def _commit(
        self,
        current: AtomisticReplayRevision,
        change: Mapping[str, Any],
        frames: list[AtomisticExperienceFrame],
        evicted: list[AtomisticEvictionRecord],
        refused: list[AtomisticRefusalRecord],
        sequence: int,
        /,
    ) -> None:
        revision_id = _revision_id(current.revision_id, change)
        candidate = AtomisticReplayRevision(
            index=current.index + 1,
            revision_id=revision_id,
            lineage=(*current.lineage, revision_id),
            frames=tuple(frames),
            evicted=tuple(evicted),
            refused=tuple(refused),
            change=MappingProxyType(dict(change)),
            next_sequence=sequence,
        )
        manifest, arrays = self._encode(candidate)
        write_array_archive(
            self._path, manifest=manifest, arrays=arrays, limits=_BUFFER_LIMITS
        )
        self._revision = candidate

    def _header(self) -> dict[str, Any]:
        return {
            "format": _BUFFER_FORMAT,
            "sink_id": self.sink_id,
            "capacity": self._capacity,
            "admission_id": self._admission.admission_id,
        }

    def _encode(
        self, revision: AtomisticReplayRevision, /
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        arrays: dict[str, np.ndarray] = {}
        frames = []
        for frame in revision.frames:
            prefix = f"frame/{frame.sequence}"
            for name, value in frame.ports.items():
                arrays[f"{prefix}/port/{name}"] = np.asarray(value)
            labels = []
            for position, label in enumerate(frame.labels):
                for name, value in label.values.items():
                    arrays[f"{prefix}/label/{position}/value/{name}"] = np.asarray(value)
                for name, value in label.available.items():
                    arrays[f"{prefix}/label/{position}/available/{name}"] = np.asarray(
                        value
                    )
                labels.append(
                    {
                        "label_id": label.label_id,
                        "values": sorted(label.values),
                        "available": sorted(label.available),
                    }
                )
            frames.append(
                {
                    "semantic_key": frame.semantic_key,
                    "payload_id": frame.payload_id,
                    "campaign_id": frame.campaign_id,
                    "task_id": frame.task_id,
                    "material_id": frame.material_id,
                    "stage": frame.stage,
                    "stage_entry": frame.stage_entry,
                    "outcome": frame.outcome,
                    "policy_id": frame.policy_id,
                    "sequence": frame.sequence,
                    "ports": sorted(frame.ports),
                    "labels": labels,
                }
            )
        manifest = {
            **self._header(),
            "index": revision.index,
            "revision_id": revision.revision_id,
            "lineage": list(revision.lineage),
            "change": None if revision.change is None else dict(revision.change),
            "next_sequence": revision.next_sequence,
            "frames": frames,
            "evicted": [
                [item.semantic_key, item.payload_id, item.sequence, item.revision_index]
                for item in revision.evicted
            ],
            "refused": [
                [item.semantic_key, item.payload_id, item.reason]
                for item in revision.refused
            ],
        }
        return manifest, arrays

    def _read(self, genesis: str, /) -> AtomisticReplayRevision:
        manifest, arrays = read_array_archive(self._path, limits=_BUFFER_LIMITS)
        header = self._header()
        if any(manifest.get(name) != value for name, value in header.items()):
            raise ValueError(
                "The experience buffer archive belongs to another sink, capacity or "
                "admission rule."
            )
        lineage = tuple(str(value) for value in manifest["lineage"])
        change = manifest["change"]
        if (
            not lineage
            or lineage[0] != genesis
            or lineage[-1] != manifest["revision_id"]
            or (change is None) != (len(lineage) == 1)
            or (change is not None and _revision_id(lineage[-2], change) != lineage[-1])
        ):
            raise ValueError("The experience buffer lineage is not canonical.")
        frames = []
        for entry in manifest["frames"]:
            prefix = f"frame/{entry['sequence']}"
            labels = tuple(
                AtomisticFrameLabel(
                    str(label["label_id"]),
                    _frozen_ports(
                        {
                            name: arrays[f"{prefix}/label/{position}/value/{name}"]
                            for name in label["values"]
                        }
                    ),
                    _frozen_ports(
                        {
                            name: arrays[f"{prefix}/label/{position}/available/{name}"]
                            for name in label["available"]
                        }
                    ),
                )
                for position, label in enumerate(entry["labels"])
            )
            frames.append(
                AtomisticExperienceFrame(
                    semantic_key=str(entry["semantic_key"]),
                    payload_id=str(entry["payload_id"]),
                    campaign_id=str(entry["campaign_id"]),
                    task_id=str(entry["task_id"]),
                    material_id=str(entry["material_id"]),
                    stage=str(entry["stage"]),
                    stage_entry=int(entry["stage_entry"]),
                    outcome=int(entry["outcome"]),
                    policy_id=str(entry["policy_id"]),
                    sequence=int(entry["sequence"]),
                    ports=_frozen_ports(
                        {name: arrays[f"{prefix}/port/{name}"] for name in entry["ports"]}
                    ),
                    labels=labels,
                )
            )
        revision = AtomisticReplayRevision(
            index=int(manifest["index"]),
            revision_id=str(manifest["revision_id"]),
            lineage=lineage,
            frames=tuple(frames),
            evicted=tuple(
                AtomisticEvictionRecord(str(a), str(b), int(c), int(d))
                for a, b, c, d in manifest["evicted"]
            ),
            refused=tuple(
                AtomisticRefusalRecord(str(a), str(b), str(c))
                for a, b, c in manifest["refused"]
            ),
            change=None if change is None else MappingProxyType(dict(change)),
            next_sequence=int(manifest["next_sequence"]),
        )
        if len(revision.frames) > self._capacity or revision.index != len(lineage) - 1:
            raise ValueError("The experience buffer archive is inconsistent.")
        encoded, encoded_arrays = self._encode(revision)
        # The archive owner authenticates its reserved array inventory; this
        # layer compares only its own manifest fields and reconstructed values.
        application = {
            name: value for name, value in manifest.items() if name != "arrays"
        }
        if (
            set(manifest) != set(encoded) | {"arrays"}
            or encoded != application
            or array_tree_fingerprint(encoded_arrays)
            != array_tree_fingerprint(dict(arrays))
        ):
            raise ValueError("The experience buffer archive is not canonical.")
        return revision


# ---------------------------------------------------------------------- rollout


@final
class AtomisticDistillationRollout(StrictModule, ExplicitFreeze):
    """Native campaign rollout of walkers driven by one student revision.

    ``program`` is the potential program template whose learned term named
    ``term`` is the student; the student is rebound by rebuilding that term
    (same name, force group, periodic permission, conditioning, key and
    request) over a new model revision, re-preparing the program on ``system``
    with ``neighborhood_bundle``/``graph_execution`` and the dynamics with
    ``integrator``. No hidden hook mutates a prepared runtime. Each segment is
    one campaign over the walkers: ``frames`` dynamics stages of ``cadence``
    accepted steps, every stage publishing its frame (the last is the final
    frame). Observation cadence never changes the dynamics: the campaign
    carries each walker's complete `AtomisticDynamicsContinuation`
    (kinematics, controllers such as thermostat chains, step, time, RNG,
    images, term states, force cache and neighborhoods) from stage to stage,
    and `PreparedAtomisticCampaign.terminal_packets` returns it after the
    segment. The next segment's campaign adopts that continuation bitwise. When
    the driving student changes, `rebind` moves every continuation to the new
    student's operation through the dynamics owner's Hamiltonian transition
    while the old operation is still alive; no per-walker model copy is kept.
    """

    system: PreparedAtomisticSystem
    program: AtomisticPotentialProgram
    neighborhood_bundle: AtomisticNeighborhoodBundlePlan
    graph_execution: AtomisticGraphExecutionPlan | None
    integrator: AtomisticIntegratorPlan
    states: tuple[AtomisticThermodynamicStatePlan, ...]
    walkers: tuple[AtomisticTaskPacket, ...]
    term: str = eqx.field(static=True)
    state_index: int = eqx.field(static=True)
    cadence: int = eqx.field(static=True)
    frames: int = eqx.field(static=True)
    steps_per_round: int = eqx.field(static=True)
    lane_capacity: int = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    rollout_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: PreparedAtomisticSystem,
        program: AtomisticPotentialProgram,
        /,
        *,
        term: str,
        neighborhood_bundle: AtomisticNeighborhoodBundlePlan,
        integrator: AtomisticIntegratorPlan,
        states: Sequence[AtomisticThermodynamicStatePlan],
        walkers: Sequence[AtomisticTaskPacket],
        cadence: int,
        frames: int,
        graph_execution: AtomisticGraphExecutionPlan | None = None,
        state_index: int = 0,
        steps_per_round: int = 1,
        lane_capacity: int | None = None,
        material_id: str = "material",
    ) -> None:
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("system must be a PreparedAtomisticSystem.")
        if not isinstance(program, AtomisticPotentialProgram):
            raise TypeError("program must be an AtomisticPotentialProgram.")
        if not isinstance(neighborhood_bundle, AtomisticNeighborhoodBundlePlan):
            raise TypeError(
                "neighborhood_bundle must be an AtomisticNeighborhoodBundlePlan."
            )
        if not isinstance(integrator, AtomisticIntegratorPlan):
            raise TypeError("integrator must be a native atomistic integrator plan.")
        name = _name(term, "term")
        matches = [value for value in program.terms if value.name == name]
        if len(matches) != 1 or not isinstance(matches[0], LearnedGraphPotentialTerm):
            raise ValueError(
                f"The program needs exactly one learned term named {name!r}."
            )
        rows = tuple(states)
        if not rows or any(
            not isinstance(row, AtomisticThermodynamicStatePlan) for row in rows
        ):
            raise TypeError("states must be AtomisticThermodynamicStatePlan values.")
        packets = tuple(walkers)
        if not packets or any(
            not isinstance(packet, AtomisticTaskPacket)
            or packet.positions is None
            or packet.continuation is not None
            for packet in packets
        ):
            raise ValueError(
                "walkers must be initial AtomisticTaskPacket values with positions."
            )
        index = _positive(state_index, "state_index", minimum=0)
        if index >= len(rows):
            raise ValueError("state_index names no thermodynamic state.")
        self.system = system
        self.program = program
        self.neighborhood_bundle = neighborhood_bundle
        self.graph_execution = graph_execution
        self.integrator = integrator
        self.states = rows
        self.walkers = packets
        self.term = name
        self.state_index = index
        self.cadence = _positive(cadence, "cadence")
        self.frames = _positive(frames, "frames")
        self.steps_per_round = _positive(steps_per_round, "steps_per_round")
        self.lane_capacity = (
            len(packets)
            if lane_capacity is None
            else _positive(lane_capacity, "lane_capacity")
        )
        self.material_id = _name(material_id, "material_id")
        self.rollout_id = canonical_fingerprint(
            {
                "kind": "atomistic-distillation-rollout",
                "system": system.prepared_id,
                "program": [
                    value.term_id for value in program.terms if value.name != name
                ],
                "coefficients": np.asarray(program.coefficients).tolist(),
                "term": name,
                "bundle": neighborhood_bundle.plan_id,
                "graph_execution": None
                if graph_execution is None
                else graph_execution.plan_id,
                "integrator": integrator.plan_id,
                "states": [row.state_id for row in rows],
                "state_index": index,
                "walkers": [array_tree_fingerprint(packet.ports()) for packet in packets],
                "walker_times": [float(packet.time) for packet in packets],
                "cadence": self.cadence,
                "frames": self.frames,
                "steps_per_round": self.steps_per_round,
                "lane_capacity": self.lane_capacity,
                "material": self.material_id,
            }
        )

    def learned_term(self) -> LearnedGraphPotentialTerm:
        term = next(value for value in self.program.terms if value.name == self.term)
        if not isinstance(term, LearnedGraphPotentialTerm):
            raise TypeError("The rollout student term is not a learned graph term.")
        return term

    def program_for(
        self, student: AbstractAtomisticModel, /
    ) -> AtomisticPotentialProgram:
        """The program with the student term rebuilt over ``student``."""
        template = self.learned_term()
        source = template.potential
        architecture = getattr(source, "architecture_id", None)
        if architecture is not None and student.architecture_id != architecture:
            raise ValueError("The rollout student must keep the template architecture.")
        replacement = LearnedGraphPotentialTerm(
            student,
            name=template.name,
            force_group=template.force_group,
            allow_periodic=template.allow_periodic,
            conditioning=template.conditioning,
            key=template.key,
            request=template.request,
        )
        terms = tuple(
            replacement if value.name == self.term else value
            for value in self.program.terms
        )
        return AtomisticPotentialProgram(
            terms, coefficients=np.asarray(self.program.coefficients)
        )

    def operation(
        self, student: AbstractAtomisticModel, /
    ) -> AtomisticDynamicsStageOperation:
        """The native dynamics stage operation of one student revision."""
        prepared = self.program_for(student).prepare(
            self.system,
            neighborhood_bundle=self.neighborhood_bundle,
            graph_execution=self.graph_execution,
        )
        dynamics = AtomisticDynamicsPlan(self.system, prepared, self.integrator).prepare()
        return AtomisticDynamicsStageOperation(
            dynamics,
            self.states,
            state_index=self.state_index,
            steps_per_round=self.steps_per_round,
        )

    def campaign(
        self,
        operation: AtomisticDynamicsStageOperation,
        walkers: Sequence[AtomisticTaskPacket],
        campaign_id: str,
        /,
    ) -> AtomisticCampaignPlan:
        """One segment campaign: ``frames`` publishing stages per walker."""
        packets = tuple(walkers)
        stages = []
        for frame in range(self.frames):
            following = f"frame-{frame + 1}" if frame + 1 < self.frames else None
            stages.append(
                AtomisticStagePlan(
                    f"frame-{frame}",
                    {_WALKER_FAMILY: operation},
                    budget=self.cadence,
                    routing=AtomisticStageRouting(
                        completed=AtomisticStageRoute(following, publish=True)
                    ),
                    lane_capacity=self.lane_capacity,
                    queue_capacity=max(len(packets), 1),
                )
            )
        tasks = tuple(
            AtomisticCampaignTask(
                f"walker-{index}",
                _WALKER_FAMILY,
                packet,
                material_id=self.material_id,
                ordinal=index,
            )
            for index, packet in enumerate(packets)
        )
        return AtomisticCampaignPlan(
            campaign_id,
            tuple(stages),
            tasks,
            max_pending_publications=max(len(packets) * self.frames, 1),
        )

    def rebind(
        self,
        walkers: Sequence[AtomisticTaskPacket],
        previous: AbstractAtomisticModel,
        student: AbstractAtomisticModel,
        /,
    ) -> tuple[AtomisticTaskPacket, ...]:
        """Walkers of ``previous``'s operation rebound to ``student``'s operation.

        A walker that still carries no dynamics continuation (before its first
        campaign) is unchanged. Every continuation must be bound to the
        operation of ``previous``; it is moved by the campaign owner's
        ``transition_packet`` (the dynamics owner's ``transition_hamiltonian``:
        configuration, momenta, time, step, RNG, controllers and counters are
        kept, forces are re-evaluated and the energy jump is booked as external
        work, the transition lineage is recorded). An owner refusal raises; an
        unchanged operation binding returns the walkers unchanged.
        """
        packets = tuple(walkers)
        if all(packet.continuation is None for packet in packets):
            return packets
        old = self.operation(previous)
        new = self.operation(student)
        if old.binding_id == new.binding_id:
            return packets
        rebound = []
        for packet in packets:
            continuation = packet.continuation
            if continuation is None:
                rebound.append(packet)
                continue
            if (
                not isinstance(continuation, AtomisticDynamicsContinuation)
                or continuation.source_binding_id != old.binding_id
            ):
                raise ValueError(
                    "A walker continuation is not bound to the previous driving "
                    "student's operation."
                )
            # The canonical packet of the transitioned continuation, exactly as a
            # restart rebuilds it.
            transitioned = old.transition_packet(packet, new).continuation
            rebound.append(new.continuation_packet(transitioned))
        return tuple(rebound)


# ----------------------------------------------------------------- qualification


@final
class AtomisticDistillationQualification(StrictModule, NonTrainableState):
    """Native qualification of a trained candidate before promotion.

    ``labels`` are teacher labels of the unlabeled ``source`` (for example the
    store returned by `label_atomistic_structures`); the plan admits them only
    when `require_atomistic_labels` proves they are exactly ITS teacher and
    request's labels of ``source`` (every record ID binds the label
    definition, so another teacher revision or request with the same schema is
    refused). The candidate's selected view is evaluated with the training
    problem's objectives and normalization over ``labels`` (lineage-disjoint
    from the replay frames). It qualifies when the evaluation succeeds and the
    summed normalized objective value is finite, at most ``maximum_loss`` and,
    with ``require_improvement``, below the promoted student's qualified value.
    """

    labels: AbstractAtomisticStructureSource
    source: AbstractAtomisticStructureSource
    maximum_loss: float = eqx.field(static=True)
    require_improvement: bool = eqx.field(static=True)
    batch_size: int | None = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)

    def __init__(
        self,
        labels: AbstractAtomisticStructureSource,
        source: AbstractAtomisticStructureSource,
        /,
        *,
        maximum_loss: float,
        require_improvement: bool = False,
        batch_size: int | None = None,
    ) -> None:
        if not isinstance(labels, AbstractAtomisticStructureSource) or not isinstance(
            source, AbstractAtomisticStructureSource
        ):
            raise TypeError("labels and source must be atomistic structure sources.")
        bound = _finite(maximum_loss, "maximum_loss")
        if not isinstance(require_improvement, bool):
            raise TypeError("require_improvement must be a boolean.")
        self.labels = labels
        self.source = source
        self.maximum_loss = bound
        self.require_improvement = require_improvement
        self.batch_size = (
            None if batch_size is None else _positive(batch_size, "batch_size")
        )
        self.qualification_id = canonical_fingerprint(
            {
                "kind": "atomistic-distillation-qualification",
                "labels": labels.source_id,
                "source": source.source_id,
                "maximum_loss": float(bound).hex(),
                "require_improvement": require_improvement,
                "batch_size": self.batch_size,
            }
        )


@final
class AtomisticQualificationOutcome(StrictModule, NonTrainableState):
    """Committed qualification evidence of one candidate."""

    candidate_revision: str = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    values: tuple[float | None, ...] = eqx.field(static=True)
    value: float | None = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)


# ------------------------------------------------------------------------- plan


@final
class AtomisticOnPolicyDistillationPlan(StrictModule, ExplicitFreeze):
    """Complete declaration of one on-policy distillation campaign.

    ``student`` is the initial student and the restore template of every
    student revision; ``components`` are initial trainable objective
    components by objective name. ``drive`` declares which student drives the
    next segment's rollout: ``"promoted"`` waits for promotion, ``"candidate"``
    drives with the latest successfully trained candidate even before it is
    promoted. Training always continues from the latest trained candidate.
    """

    teacher: AtomisticTeacher
    request: AtomisticLabelRequest
    student: AbstractAtomisticModel
    components: tuple[Any, ...]
    rollout: AtomisticDistillationRollout
    objectives: tuple[AbstractAtomisticObjective, ...]
    policy: AtomisticTrainingPolicy
    graph_execution: AtomisticGraphExecutionPlan
    fields: AtomisticTrainingFields
    replay: AtomisticReplayDraw
    admission: AtomisticExperienceAdmission
    qualification: AtomisticDistillationQualification
    schema: AtomisticStructureStoreSchema
    base_schema: AtomisticStructureStoreSchema
    component_names: tuple[str, ...] = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    skin: float = eqx.field(static=True)
    batch_size: int | None = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    drive: AtomisticDistillationDrive = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    maximum_rounds: int = eqx.field(static=True)
    stress_convention: AtomisticStressConvention = eqx.field(static=True)
    label_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        teacher: AtomisticTeacher,
        request: AtomisticLabelRequest,
        student: AbstractAtomisticModel,
        rollout: AtomisticDistillationRollout,
        objectives: Sequence[AbstractAtomisticObjective],
        policy: AtomisticTrainingPolicy,
        graph_execution: AtomisticGraphExecutionPlan,
        qualification: AtomisticDistillationQualification,
        cutoff: float,
        capacity: int,
        fields: AtomisticTrainingFields | None = None,
        components: Mapping[str, Any] | None = None,
        skin: float = 0.0,
        batch_size: int | None = None,
        replay: AtomisticReplayDraw | None = None,
        admission: AtomisticExperienceAdmission | None = None,
        drive: AtomisticDistillationDrive = "promoted",
        seed: int = 0,
        maximum_rounds: int = 10_000,
        stress_convention: AtomisticStressConvention = (
            AtomisticStressConvention.CAUCHY_TENSION_POSITIVE
        ),
        atom_energy_complete: bool | None = None,
    ) -> None:
        _require_teacher(teacher, request)
        if not isinstance(student, AbstractAtomisticModel):
            raise TypeError("student must implement AbstractAtomisticModel.")
        if not isinstance(rollout, AtomisticDistillationRollout):
            raise TypeError("rollout must be an AtomisticDistillationRollout.")
        if not isinstance(policy, AtomisticTrainingPolicy):
            raise TypeError("policy must be an AtomisticTrainingPolicy.")
        if not isinstance(qualification, AtomisticDistillationQualification):
            raise TypeError(
                "qualification must be an AtomisticDistillationQualification."
            )
        objectives_ = tuple(objectives)
        if not objectives_ or any(
            not isinstance(value, AbstractAtomisticObjective) for value in objectives_
        ):
            raise TypeError(
                "objectives must be nonempty AbstractAtomisticObjective values."
            )
        supplied = {} if components is None else dict(components)
        names = tuple(sorted(supplied))
        if set(names) - {value.name for value in objectives_}:
            raise ValueError("components name unknown objectives.")
        if student.scale.scale_id != teacher.model.scale.scale_id:
            raise ValueError("Student and teacher scales differ.")
        system = rollout.system
        units = system.plan.units
        if units.scale.scale_id != student.scale.scale_id:
            raise ValueError("The rollout system and the student scales differ.")
        fields_ = (
            AtomisticTrainingFields(
                species_kind=student.capabilities.species_kind,
                **(
                    {"cell": "cell", "periodic": "periodic"}
                    if system.cell is not None
                    else {}
                ),
            )
            if fields is None
            else fields
        )
        base = atomistic_training_schema(
            units.scale,
            mass_unit=units.mass_unit,
            species_kind=fields_.species_kind,
            periodic=system.cell is not None,
            energy=False,
            forces=False,
            stress=False,
        )
        schema = atomistic_distillation_schema(
            base,
            teacher,
            request,
            stress_convention=stress_convention,
            atom_energy_complete=atom_energy_complete,
        )
        require_atomistic_labels(
            qualification.labels, teacher, request, qualification.source
        )
        if qualification.labels.schema.schema_id != schema.schema_id:
            raise ValueError(
                "The qualification labels must use this plan's canonical replay schema."
            )
        drive_ = parse(drive, AtomisticDistillationDrive, "drive")
        self.teacher = teacher
        self.request = request
        self.student = student
        self.components = tuple(supplied[name] for name in names)
        self.component_names = names
        self.rollout = rollout
        self.objectives = objectives_
        self.policy = policy
        self.graph_execution = graph_execution
        self.fields = fields_
        self.replay = AtomisticReplayDraw() if replay is None else replay
        self.admission = (
            AtomisticExperienceAdmission() if admission is None else admission
        )
        self.qualification = qualification
        self.schema = schema
        self.base_schema = base
        self.cutoff = _finite(cutoff, "cutoff")
        self.skin = _finite(skin, "skin")
        self.batch_size = (
            None if batch_size is None else _positive(batch_size, "batch_size")
        )
        self.capacity = _positive(capacity, "capacity")
        self.drive = drive_
        self.seed = _positive(seed, "seed", minimum=0)
        self.maximum_rounds = _positive(maximum_rounds, "maximum_rounds")
        self.stress_convention = parse(
            stress_convention, AtomisticStressConvention, "stress_convention"
        )
        self.label_id = atomistic_label_id(teacher, request)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-on-policy-distillation-plan",
                "label": self.label_id,
                "student": {
                    "architecture": student.architecture_id,
                    "method": student.method_id,
                    "revision": _model_revision(student),
                },
                "components": {
                    "names": list(names),
                    "content": _component_digest(self.components),
                },
                "rollout": rollout.rollout_id,
                "objectives": [value.objective_id for value in objectives_],
                "policy": policy.policy_id,
                "graph_execution": graph_execution.plan_id,
                "fields": fields_.fields_id,
                "schema": schema.schema_id,
                "replay": self.replay.draw_id,
                "admission": self.admission.admission_id,
                "qualification": qualification.qualification_id,
                "cutoff": self.cutoff,
                "skin": self.skin,
                "batch_size": self.batch_size,
                "capacity": self.capacity,
                "drive": drive_,
                "seed": self.seed,
                "maximum_rounds": self.maximum_rounds,
            }
        )

    @property
    def sink_id(self) -> str:
        return f"distillation-{self.plan_id[:24]}"

    def buffer(self, directory: str | Path, /) -> AtomisticExperienceBuffer:
        """The plan's experience buffer under ``directory``."""
        return AtomisticExperienceBuffer(
            Path(directory) / "replay",
            capacity=self.capacity,
            admission=self.admission,
            sink_id=self.sink_id,
        )

    def component_map(self, components: Sequence[Any], /) -> dict[str, Any]:
        return dict(zip(self.component_names, tuple(components), strict=True))


# ------------------------------------------------------------------------ state


def _walker_digest(packet: AtomisticTaskPacket, /) -> dict[str, Any]:
    """Identity of one walker: phase-point ports and complete dynamics continuation."""
    continuation = packet.continuation
    return {
        "ports": array_tree_fingerprint(packet.ports()),
        "time": float(packet.time),
        "continuation": None
        if continuation is None
        else {
            "binding": continuation.source_binding_id,
            "lineage": list(continuation.lineage),
            "state": _component_digest((continuation.state,)),
        },
    }


@final
class AtomisticOnPolicyDistillationState(StrictModule, ExplicitFreeze):
    """Committed continuation of an on-policy distillation campaign.

    ``phase`` is the next phase to execute in segment ``segment``. The state
    retains every walker's packet with its complete dynamics continuation
    (bound to the driving student's operation; none before the first
    rollout), the generating policy history (``policy_ids``, oldest first,
    ages counted from the last), the committed replay revision, the
    labeled-frame cursor, the training result and the replay revision it
    trained on, the candidate, promoted and driving students with their
    components, the qualification outcome and the pending publication.
    """

    driving: AbstractAtomisticModel
    promoted: AbstractAtomisticModel
    candidate: AbstractAtomisticModel | None
    components: tuple[Any, ...]
    walkers: tuple[AtomisticTaskPacket, ...]
    training: AtomisticTrainingResult | None
    qualification: AtomisticQualificationOutcome | None
    plan_id: str = eqx.field(static=True)
    segment: int = eqx.field(static=True)
    phase: AtomisticDistillationPhase = eqx.field(static=True)
    policy_ids: tuple[str, ...] = eqx.field(static=True)
    replay_revision: str = eqx.field(static=True)
    labeled_frames: int = eqx.field(static=True)
    training_revision: str | None = eqx.field(static=True)
    promoted_value: float | None = eqx.field(static=True)
    pending_publication: str | None = eqx.field(static=True)
    publications: tuple[str, ...] = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        plan_id: str,
        segment: int,
        phase: AtomisticDistillationPhase,
        driving: AbstractAtomisticModel,
        promoted: AbstractAtomisticModel,
        candidate: AbstractAtomisticModel | None,
        components: Sequence[Any],
        walkers: Sequence[AtomisticTaskPacket],
        policy_ids: Sequence[str],
        replay_revision: str,
        labeled_frames: int,
        training: AtomisticTrainingResult | None,
        training_revision: str | None,
        qualification: AtomisticQualificationOutcome | None,
        promoted_value: float | None,
        pending_publication: str | None,
        publications: Sequence[str],
    ) -> None:
        phase_ = parse(phase, AtomisticDistillationPhase, "phase")
        if (training is None) != (training_revision is None):
            raise ValueError("A training result and its replay revision come together.")
        if phase_ in ("qualify", "promote") and (training is None or candidate is None):
            raise ValueError(
                "Qualification and promotion need a committed training result."
            )
        if phase_ == "promote" and qualification is None:
            raise ValueError("Promotion needs a committed qualification outcome.")
        if pending_publication is not None and phase_ != "promote":
            raise ValueError("A pending publication belongs to the promotion phase.")
        self.plan_id = _name(plan_id, "plan_id")
        self.segment = _positive(segment, "segment", minimum=0)
        self.phase = phase_
        self.driving = driving
        self.promoted = promoted
        self.candidate = candidate
        self.components = tuple(components)
        packets = tuple(walkers)
        if any(
            not isinstance(packet, AtomisticTaskPacket)
            or not (
                packet.continuation is None
                or isinstance(packet.continuation, AtomisticDynamicsContinuation)
            )
            for packet in packets
        ):
            raise TypeError(
                "walkers must be AtomisticTaskPacket values carrying no or a dynamics "
                "continuation."
            )
        self.walkers = packets
        self.policy_ids = tuple(policy_ids)
        self.replay_revision = _name(replay_revision, "replay_revision")
        self.labeled_frames = _positive(labeled_frames, "labeled_frames", minimum=0)
        self.training = training
        self.training_revision = training_revision
        self.qualification = qualification
        self.promoted_value = promoted_value
        self.pending_publication = pending_publication
        self.publications = tuple(publications)
        self.state_id = canonical_fingerprint(
            {
                "kind": "atomistic-on-policy-distillation-state",
                "plan": self.plan_id,
                "segment": self.segment,
                "phase": phase_,
                "driving": _model_revision(driving),
                "promoted": _model_revision(promoted),
                "candidate": None if candidate is None else _model_revision(candidate),
                "components": _component_digest(self.components),
                "walkers": [_walker_digest(packet) for packet in self.walkers],
                "policies": list(self.policy_ids),
                "replay": self.replay_revision,
                "labeled": self.labeled_frames,
                "training": None if training is None else training.result_id,
                "training_revision": training_revision,
                "qualification": None
                if qualification is None
                else [qualification.candidate_revision, qualification.accepted],
                "promoted_value": None
                if promoted_value is None
                else float(promoted_value).hex(),
                "pending": pending_publication,
                "publications": list(self.publications),
            }
        )

    @classmethod
    def initial(
        cls, plan: AtomisticOnPolicyDistillationPlan, /
    ) -> AtomisticOnPolicyDistillationState:
        """Segment 0 before its rollout: the initial student drives and is promoted."""
        if not isinstance(plan, AtomisticOnPolicyDistillationPlan):
            raise TypeError("plan must be an AtomisticOnPolicyDistillationPlan.")
        return cls(
            plan_id=plan.plan_id,
            segment=0,
            phase="rollout",
            driving=plan.student,
            promoted=plan.student,
            candidate=None,
            components=plan.components,
            walkers=plan.rollout.walkers,
            policy_ids=(),
            replay_revision=_genesis_id(plan.sink_id, plan.capacity, plan.admission),
            labeled_frames=0,
            training=None,
            training_revision=None,
            qualification=None,
            promoted_value=None,
            pending_publication=None,
            publications=(),
        )

    def replace(self, **changes: Any) -> AtomisticOnPolicyDistillationState:
        """A new committed state with ``changes`` (identity recomputed)."""
        names = (
            "plan_id",
            "segment",
            "phase",
            "driving",
            "promoted",
            "candidate",
            "components",
            "walkers",
            "policy_ids",
            "replay_revision",
            "labeled_frames",
            "training",
            "training_revision",
            "qualification",
            "promoted_value",
            "pending_publication",
            "publications",
        )
        unknown = set(changes) - set(names)
        if unknown:
            raise ValueError(f"Unknown state fields {sorted(unknown)!r}.")
        values = {name: getattr(self, name) for name in names}
        values.update(changes)
        return AtomisticOnPolicyDistillationState(**values)


@final
class AtomisticDistillationReport(StrictModule, NonTrainableState):
    """Host evidence of one segment call: phases, costs, counts and outcome."""

    segment: int = eqx.field(static=True)
    phases: tuple[AtomisticDistillationPhase, ...] = eqx.field(static=True)
    phase_seconds: tuple[float, ...] = eqx.field(static=True)
    frames_published: int = eqx.field(static=True)
    frames_labeled: int = eqx.field(static=True)
    replay_records: int = eqx.field(static=True)
    replay_probabilities: tuple[float, ...] = eqx.field(static=True)
    training_status: int | None = eqx.field(static=True)
    promoted: bool = eqx.field(static=True)
    failure: str | None = eqx.field(static=True)


@final
class AtomisticDistillationSegmentResult(StrictModule, ExplicitFreeze):
    state: AtomisticOnPolicyDistillationState
    report: AtomisticDistillationReport

    @property
    def successful(self) -> bool:
        return self.report.failure is None


# ---------------------------------------------------------------- transactions


def _require_state(
    plan: AtomisticOnPolicyDistillationPlan, state: AtomisticOnPolicyDistillationState, /
) -> None:
    if not isinstance(plan, AtomisticOnPolicyDistillationPlan):
        raise TypeError("plan must be an AtomisticOnPolicyDistillationPlan.")
    if not isinstance(state, AtomisticOnPolicyDistillationState):
        raise TypeError("state must be an AtomisticOnPolicyDistillationState.")
    if state.plan_id != plan.plan_id:
        raise ValueError("The distillation state belongs to another plan.")


def _segment_key(
    plan: AtomisticOnPolicyDistillationPlan, address: SampleAddress, segment: int, /
) -> Array:
    return derive_key(jr.key(plan.seed), address, segment)


def _campaign_id(
    plan: AtomisticOnPolicyDistillationPlan, segment: int, policy: str, /
) -> str:
    return (
        "distill-"
        + canonical_fingerprint(
            {
                "kind": "atomistic-distillation-segment-campaign",
                "plan": plan.plan_id,
                "segment": segment,
                "policy": policy,
            }
        )[:40]
    )


def _frame_record(
    plan: AtomisticOnPolicyDistillationPlan, frame: AtomisticExperienceFrame, /
) -> AtomisticStructureRecord:
    """The canonical unlabeled record of one frame (on the rollout system)."""
    system = plan.rollout.system
    ports = frame.ports
    mask = np.asarray(ports["atom_mask"], dtype=np.bool_)
    if mask.shape != (system.capacity,):
        raise ValueError("A frame does not belong to the rollout system capacity.")
    kind = plan.fields.species_kind
    species = (
        ports["species"]
        if kind is AtomisticSpeciesKind.ATOMIC_NUMBER
        else ports.get("atom_type_ids", ports.get("species"))
    )
    values: dict[str, np.ndarray] = {
        "positions": np.asarray(ports["positions"], dtype=np.float64)[mask],
        "species": np.asarray(species, dtype=np.int32)[mask],
        "masses": np.asarray(system.plan.masses, dtype=np.float64)[mask],
    }
    if system.cell is not None:
        values["cell"] = np.asarray(ports["cell_vectors"], dtype=np.float64).reshape(
            (1, 3, 3)
        )
        values["periodic"] = np.asarray(system.cell.periodic_axes, dtype=np.bool_)[None]
    return AtomisticStructureRecord(
        plan.base_schema,
        record_id=f"frame-{frame.semantic_key}",
        lineage_id=f"walker-{plan.plan_id[:16]}-{frame.task_id}",
        counts={plan.base_schema.atom_level: [int(np.count_nonzero(mask))]},
        stable_ids=np.asarray(system.plan.particle_ids, dtype=np.int64)[mask],
        values=values,
    )


def _label_frames(
    plan: AtomisticOnPolicyDistillationPlan,
    frames: Sequence[AtomisticExperienceFrame],
    /,
) -> dict[str, AtomisticFrameLabel]:
    """Teacher labels of retained frames (at most the buffer capacity)."""
    source = AtomisticStructureSource(
        plan.base_schema, tuple(_frame_record(plan, frame) for frame in frames)
    )
    loader = atomistic_training_loader(
        source, batch_size=plan.batch_size, stream="labeling", shuffle=False
    )
    preparation = AtomisticBatchPreparation(
        loader,
        plan.graph_execution,
        scale=plan.teacher.model.scale,
        cutoff=plan.cutoff,
        skin=plan.skin,
        fields=plan.fields,
    )
    labels: dict[str, AtomisticFrameLabel] = {}
    for batch, complete in _teacher_label_batches(
        plan.teacher, plan.request, preparation, loader.start(0)
    ):
        if plan.request.atom_energy:
            expected = _atom_energy_convention(plan.teacher, bool(complete))
            if plan.schema.field(TEACHER_ATOM_ENERGY_SLOT).convention != expected:
                raise AtomisticTeacherLabelError(
                    "The teacher atom-energy decomposition completeness differs from "
                    "the plan."
                )
        for draw, label in batch.items():
            labels[frames[draw].semantic_key] = AtomisticFrameLabel(
                plan.label_id, label.values, label.available
            )
    if len(labels) != len(frames):
        raise RuntimeError("The frame labeling epoch did not draw every frame.")
    return labels


@final
@dataclass(frozen=True, slots=True, eq=False)
class _Replay:
    problem: AtomisticTrainingProblem
    records: int
    probabilities: tuple[float, ...]


def _replay_problem(
    plan: AtomisticOnPolicyDistillationPlan,
    state: AtomisticOnPolicyDistillationState,
    revision: AtomisticReplayRevision,
    /,
) -> _Replay:
    """The training problem of one immutable replay revision."""
    frames = [
        frame for frame in revision.frames if frame.label(plan.label_id) is not None
    ]
    if not frames:
        raise ValueError("The replay revision holds no labeled experience.")
    latest = len(state.policy_ids) - 1
    classes: dict[int, list[AtomisticStructureRecord]] = {}
    for frame in frames:
        if frame.policy_id not in state.policy_ids:
            raise ValueError(
                f"Frame {frame.semantic_key!r} was generated by a policy outside this "
                "run's history; the replay source is stale or foreign."
            )
        age = latest - state.policy_ids.index(frame.policy_id)
        repeats = plan.replay.repeats(age)
        if repeats == 0:
            continue
        label = frame.label(plan.label_id)
        base = _frame_record(plan, frame)
        values = dict(base.values)
        available = dict(base.available)
        if label is not None:
            values.update(label.values)
            available.update(label.available)
        classes.setdefault(repeats, []).append(
            AtomisticStructureRecord(
                plan.schema,
                record_id=base.record_id,
                lineage_id=base.lineage_id,
                counts=base.counts,
                stable_ids=base.stable_ids,
                values=values,
                available=available,
            )
        )
    if not classes:
        raise ValueError("Every labeled frame has zero replay draw probability.")
    repeats = sorted(classes, reverse=True)
    sources = tuple(
        AtomisticStructureSource(plan.schema, tuple(classes[value])) for value in repeats
    )
    loader = atomistic_training_loader(
        *sources,
        batch_size=plan.batch_size,
        stream="replay",
        seed=(plan.seed + state.segment) % (1 << 32),
        weights=repeats,
    )
    problem = AtomisticTrainingProblem(
        loader,
        plan.graph_execution,
        plan.objectives,
        scale=plan.student.scale,
        cutoff=plan.cutoff,
        skin=plan.skin,
        fields=plan.fields,
    )
    probabilities = tuple(
        float(value)
        for value, source in zip(
            loader.mixture.record_probabilities, sources, strict=True
        )
        for _ in range(source.record_count)
    )
    return _Replay(problem, sum(source.record_count for source in sources), probabilities)


def _qualify(
    plan: AtomisticOnPolicyDistillationPlan,
    state: AtomisticOnPolicyDistillationState,
    problem: AtomisticTrainingProblem,
    /,
) -> AtomisticQualificationOutcome:
    result = state.training
    candidate = state.candidate
    if result is None or candidate is None:
        raise RuntimeError("Qualification needs a committed training result.")
    qualification = plan.qualification
    loader = atomistic_training_loader(
        qualification.labels,
        batch_size=qualification.batch_size,
        stream="qualification",
        shuffle=False,
    )
    preparation = AtomisticBatchPreparation(
        loader,
        plan.graph_execution,
        scale=plan.student.scale,
        cutoff=plan.cutoff,
        skin=plan.skin,
        fields=plan.fields,
        label_slots=problem.preparation.label_slots,
        cell_slots=problem.preparation.cell_slots,
    )
    evaluation = evaluate_atomistic_objectives(
        result.best_tree, problem, result.normalization, preparation
    )
    values = tuple(float(value) for value in evaluation.values)
    total = math.fsum(values)
    finite = evaluation.successful and math.isfinite(total)
    if not finite:
        reason = "failed-evaluation"
    elif total > qualification.maximum_loss:
        reason = "above-maximum-loss"
    elif (
        qualification.require_improvement
        and state.promoted_value is not None
        and not total < state.promoted_value
    ):
        reason = "no-improvement"
    else:
        reason = "qualified"
    return AtomisticQualificationOutcome(
        candidate_revision=_model_revision(candidate),
        names=evaluation.names,
        values=tuple(value if math.isfinite(value) else None for value in values),
        value=total if math.isfinite(total) else None,
        successful=finite,
        accepted=reason == "qualified",
        reason=reason,
    )


def _publish_model(directory: Path, model: AbstractAtomisticModel, /) -> str:
    revision = _model_revision(model)
    target = directory / "publications" / f"{revision}.phydrax"
    if target.exists():
        read_atomistic_model_artifact(target, numeric_revision_id=revision)
        return revision
    target.parent.mkdir(parents=True, exist_ok=True)
    write_atomistic_model_artifact(target, model)
    return revision


def _require_committed_replay(
    plan: AtomisticOnPolicyDistillationPlan,
    phase: AtomisticDistillationPhase,
    replay_revision: str,
    buffer: AtomisticExperienceBuffer,
    /,
) -> None:
    """Admit the buffer revision a committed phase continues from.

    ROLLOUT admits later frames of its interrupted campaign; LABEL admits the
    exact revision or its authenticated label successor (the attachment was
    committed, the TRAIN restart was not); every other phase needs the exact
    revision.
    """
    if phase == "rollout":
        buffer.require_revision(replay_revision, exact=False)
    elif not (
        phase == "label"
        and buffer.label_successor(replay_revision, plan.label_id) is not None
    ):
        buffer.require_revision(replay_revision, exact=True)


@final
@dataclass(slots=True, eq=False)
class _SegmentEvidence:
    """Report evidence accumulated by the phases of one segment call."""

    segment: int
    phases: list[AtomisticDistillationPhase]
    seconds: list[float]
    published: int = 0
    labeled: int = 0
    records: int = 0
    probabilities: tuple[float, ...] = ()
    training_status: int | None = None
    promoted: bool = False

    def result(
        self, state: AtomisticOnPolicyDistillationState, failure: str | None, /
    ) -> AtomisticDistillationSegmentResult:
        return AtomisticDistillationSegmentResult(
            state,
            AtomisticDistillationReport(
                segment=self.segment,
                phases=tuple(self.phases),
                phase_seconds=tuple(self.seconds),
                frames_published=self.published,
                frames_labeled=self.labeled,
                replay_records=self.records,
                replay_probabilities=self.probabilities,
                training_status=self.training_status,
                promoted=self.promoted,
                failure=failure,
            ),
        )


def _rollout_transition(
    plan: AtomisticOnPolicyDistillationPlan,
    state: AtomisticOnPolicyDistillationState,
    root: Path,
    buffer: AtomisticExperienceBuffer,
    session: IterationSession | None,
    evidence: _SegmentEvidence,
    /,
) -> tuple[AtomisticOnPolicyDistillationState, str | None]:
    """ROLLOUT: run or resume the segment campaign; commit terminal continuations."""
    operation = plan.rollout.operation(state.driving)
    policy = _policy_id(operation.numeric_revision_ids)
    campaign_id = _campaign_id(plan, state.segment, policy)
    prepared = plan.rollout.campaign(operation, state.walkers, campaign_id).prepare()
    store = AtomisticCampaignStore(root / "campaign" / f"{campaign_id}.campaign")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    before = buffer.revision.index
    campaign = (
        prepared.resume(store)
        if store.path.exists()
        else prepared.start(_segment_key(plan, _CAMPAIGN_ADDRESS, state.segment))
    )
    campaign = prepared.run(
        campaign,
        buffer,
        store,
        maximum_rounds=plan.maximum_rounds,
        session=session,
    )
    evidence.published = buffer.revision.index - before
    if not campaign.complete:
        return state, "rollout campaign incomplete within maximum_rounds"
    tasks = tuple(f"walker-{index}" for index in range(len(state.walkers)))
    terminal = prepared.terminal_packets(campaign)
    ended = [
        task
        for task in tasks
        if task not in terminal
        or int(campaign.task(task).outcome) != int(AtomisticStageOutcome.COMPLETED)
    ]
    if ended:
        return state, f"walkers {ended} ended without a completed dynamics continuation"
    history = (
        state.policy_ids
        if state.policy_ids and state.policy_ids[-1] == policy
        else (*state.policy_ids, policy)
    )
    committed = state.replace(
        phase="label",
        walkers=tuple(
            operation.continuation_packet(terminal[task].continuation) for task in tasks
        ),
        policy_ids=history,
        replay_revision=buffer.revision.revision_id,
    )
    return committed, None


def _label_transition(
    plan: AtomisticOnPolicyDistillationPlan,
    state: AtomisticOnPolicyDistillationState,
    buffer: AtomisticExperienceBuffer,
    evidence: _SegmentEvidence,
    /,
) -> tuple[AtomisticOnPolicyDistillationState, str | None]:
    """LABEL: attach teacher labels, or recover the committed label successor."""
    if buffer.revision.revision_id != state.replay_revision:
        # The attachment was committed; the TRAIN restart was not.
        revision = buffer.label_successor(state.replay_revision, plan.label_id)
        if revision is None or revision.change is None:
            raise ValueError("The committed replay revision is stale.")
        evidence.labeled = len(revision.change["labels"])
    else:
        pending = [
            frame
            for frame in buffer.revision.frames
            if frame.label(plan.label_id) is None
        ]
        try:
            labels = _label_frames(plan, pending) if pending else {}
        except AtomisticTeacherLabelError as error:
            return state, f"labeling failed: {error}"
        revision = buffer.attach_labels(labels, parent=state.replay_revision)
        evidence.labeled = len(labels)
    committed = state.replace(
        phase="train",
        replay_revision=revision.revision_id,
        labeled_frames=sum(
            frame.label(plan.label_id) is not None for frame in revision.frames
        ),
    )
    return committed, None


def _train_transition(
    plan: AtomisticOnPolicyDistillationPlan,
    state: AtomisticOnPolicyDistillationState,
    buffer: AtomisticExperienceBuffer,
    evidence: _SegmentEvidence,
    /,
) -> tuple[AtomisticOnPolicyDistillationState, str | None]:
    """TRAIN: fit the latest candidate on the committed replay revision."""
    replay = _replay_problem(plan, state, buffer.revision)
    evidence.records = replay.records
    evidence.probabilities = replay.probabilities
    start = state.candidate if state.candidate is not None else state.promoted
    result = fit_atomistic_potential(
        start,
        replay.problem,
        plan.policy,
        key=_segment_key(plan, _TRAINING_ADDRESS, state.segment),
        components=plan.component_map(state.components) or None,
    )
    evidence.training_status = int(np.asarray(result.status))
    if AtomisticStatus(evidence.training_status) is not AtomisticStatus.SUCCESS:
        return state, f"training failed: {result.termination}"
    best = result.best_tree.components
    committed = state.replace(
        phase="qualify",
        candidate=result.best_potential,
        components=tuple(
            best[[objective.name for objective in plan.objectives].index(name)]
            for name in plan.component_names
        ),
        training=result,
        training_revision=buffer.revision.revision_id,
    )
    return committed, None


def _promote_transition(
    plan: AtomisticOnPolicyDistillationPlan,
    state: AtomisticOnPolicyDistillationState,
    root: Path,
    evidence: _SegmentEvidence,
    /,
) -> tuple[AtomisticOnPolicyDistillationState, str | None]:
    """PROMOTE: rebind walkers to the next driver, then publish a qualified candidate.

    A refused rebind fails before anything is published, so the promoted
    student and the publications stay unchanged.
    """
    outcome = state.qualification
    candidate = state.candidate
    if outcome is None or candidate is None:
        raise RuntimeError("Promotion needs a committed qualification.")
    accepted = outcome.accepted and state.pending_publication is not None
    new_promoted = candidate if accepted else state.promoted
    driving = new_promoted if plan.drive == "promoted" else candidate
    try:
        walkers = plan.rollout.rebind(state.walkers, state.driving, driving)
    except ValueError as error:
        return state, f"walker rebind refused: {error}"
    value = state.promoted_value
    publications = state.publications
    if accepted:
        if _publish_model(root, candidate) != state.pending_publication:
            raise ValueError("The pending publication names another candidate.")
        value = outcome.value
        publications = (*publications, state.pending_publication)
        evidence.promoted = True
    committed = state.replace(
        segment=state.segment + 1,
        phase="rollout",
        driving=driving,
        promoted=new_promoted,
        walkers=walkers,
        promoted_value=value,
        publications=publications,
        training=None,
        training_revision=None,
        qualification=None,
        pending_publication=None,
    )
    return committed, None


def run_atomistic_distillation_segment(
    plan: AtomisticOnPolicyDistillationPlan,
    state: AtomisticOnPolicyDistillationState,
    /,
    *,
    directory: str | Path,
    stop_after: AtomisticDistillationPhase | None = None,
    session: IterationSession | None = None,
) -> AtomisticDistillationSegmentResult:
    """Execute the committed phases of one segment from ``state.phase``.

    ROLLOUT runs (or resumes) the segment campaign of the driving student into
    the experience buffer and commits every walker's complete final dynamics
    continuation (a walker whose final outcome is not COMPLETED fails the
    phase); LABEL attaches teacher labels to every retained unlabeled frame;
    TRAIN fits the latest candidate on the replay revision; QUALIFY evaluates
    it on the qualification labels; PROMOTE publishes and promotes a qualified
    candidate (a rejected one leaves the promoted student unchanged), selects
    the next driving student by ``plan.drive`` and, when it differs, rebinds
    every walker continuation to it through the owner's Hamiltonian transition
    before the previous driving student is released. After every committed
    transition the restart is written under ``directory``; ``stop_after``
    returns after that phase's commit. The buffer's label attachment is
    committed before the TRAIN restart: an interruption between the two
    resumes LABEL from the authenticated label successor of the committed
    replay revision (`AtomisticExperienceBuffer.label_successor`), while any
    other buffer change is refused as stale. A failed phase returns the last
    committed state with ``report.failure`` set: no promoted student is
    replaced and no uncommitted cursor is consumed.
    """
    _require_state(plan, state)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    stop = (
        None
        if stop_after is None
        else parse(stop_after, AtomisticDistillationPhase, "stop_after")
    )
    buffer = plan.buffer(root)
    _require_committed_replay(plan, state.phase, state.replay_revision, buffer)
    evidence = _SegmentEvidence(state.segment, [], [])
    while True:
        phase = state.phase
        started = time.perf_counter()
        failure: str | None = None
        if phase == "rollout":
            state, failure = _rollout_transition(
                plan, state, root, buffer, session, evidence
            )
        elif phase == "label":
            state, failure = _label_transition(plan, state, buffer, evidence)
        elif phase == "train":
            state, failure = _train_transition(plan, state, buffer, evidence)
        elif phase == "qualify":
            replay = _replay_problem(plan, state, buffer.revision)
            outcome = _qualify(plan, state, replay.problem)
            state = state.replace(
                phase="promote",
                qualification=outcome,
                pending_publication=outcome.candidate_revision
                if outcome.accepted
                else None,
            )
        else:
            state, failure = _promote_transition(plan, state, root, evidence)
        evidence.phases.append(phase)
        evidence.seconds.append(time.perf_counter() - started)
        if failure is not None:
            return evidence.result(state, failure)
        write_atomistic_distillation_restart(root, plan, state)
        if stop == phase or phase == "promote":
            return evidence.result(state, None)


# ---------------------------------------------------------------------- restart


def _archive_bytes(manifest: Mapping[str, Any], arrays: Mapping[str, Any], /) -> bytes:
    with TemporaryDirectory(prefix="phydrax-distillation-") as scratch:
        path = Path(scratch) / _RESTART_STATE
        write_array_archive(path, manifest=manifest, arrays=arrays, limits=_STATE_LIMITS)
        return path.read_bytes()


def _model_bytes(model: AbstractAtomisticModel, /) -> bytes:
    with TemporaryDirectory(prefix="phydrax-distillation-") as scratch:
        path = Path(scratch) / "model.phydrax"
        write_atomistic_model_artifact(path, model)
        return path.read_bytes()


def _walker_payload(
    packets: Sequence[AtomisticTaskPacket], arrays: dict[str, object], /
) -> list[dict[str, Any] | None]:
    """Continuation of every walker; ``None`` for a walker before its first rollout."""
    payload: list[dict[str, Any] | None] = []
    for index, packet in enumerate(packets):
        continuation = packet.continuation
        if continuation is None:
            payload.append(None)
            continue
        payload.append(
            {
                "binding": continuation.source_binding_id,
                "lineage": list(continuation.lineage),
                "state": pack_array_tree(
                    f"walker/{index}", jax.device_get(continuation.state), arrays
                ),
            }
        )
    return payload


def _restored_walkers(
    plan: AtomisticOnPolicyDistillationPlan,
    driving: AbstractAtomisticModel,
    entries: Sequence[Mapping[str, Any] | None],
    arrays: Mapping[str, np.ndarray],
    /,
) -> tuple[AtomisticTaskPacket, ...]:
    """Rebuild every committed walker through the driving student's operation.

    The operation authenticates each continuation (binding, prepared dynamics
    and thermodynamic table) before it becomes a packet; a walker before its
    first rollout is the plan's initial walker.
    """
    initial = plan.rollout.walkers
    if len(entries) != len(initial):
        raise ValueError("The restart walkers differ from the plan's walkers.")
    operation = None
    walkers = []
    for index, entry in enumerate(entries):
        if entry is None:
            walkers.append(initial[index])
            continue
        if operation is None:
            operation = plan.rollout.operation(driving)
        if entry["binding"] != operation.binding_id:
            raise ValueError(
                "A restart walker continuation is not bound to the driving student's "
                "operation."
            )
        state = unpack_array_tree(entry["state"], arrays, operation.state_template())
        walkers.append(
            operation.continuation_packet(
                operation.continuation(state, lineage=tuple(entry["lineage"]))
            )
        )
    return tuple(walkers)


def write_atomistic_distillation_restart(
    directory: str | Path,
    plan: AtomisticOnPolicyDistillationPlan,
    state: AtomisticOnPolicyDistillationState,
    /,
) -> Path:
    """Atomically publish the committed distillation state under ``directory``.

    ``directory/restart`` is one resource set: the state archive (phase,
    segment, cursors, revisions, every walker's complete dynamics
    continuation with its operation binding and transition lineage, policy
    history, qualification, publications and component leaves), native
    artifacts of the promoted, driving and candidate students, and, while a
    training result is committed, the training owner's restart members (under
    ``student/``) rebuilt against the replay revision it trained on. A
    ``restart.json`` receipt binds every member digest. The experience buffer,
    the campaign store and the publications are their own durable owners
    beside it.
    """
    _require_state(plan, state)
    root = Path(directory)
    arrays: dict[str, object] = {}
    components = pack_array_tree(
        "components", eqx.filter(state.components, eqx.is_array), arrays
    )
    outcome = state.qualification
    manifest = {
        "format": _RESTART_FORMAT,
        "plan_id": plan.plan_id,
        "state_id": state.state_id,
        "segment": state.segment,
        "phase": state.phase,
        "policy_ids": list(state.policy_ids),
        "replay_revision": state.replay_revision,
        "labeled_frames": state.labeled_frames,
        "training_revision": state.training_revision,
        "promoted_value": None
        if state.promoted_value is None
        else float(state.promoted_value),
        "pending_publication": state.pending_publication,
        "publications": list(state.publications),
        "walkers": _walker_payload(state.walkers, arrays),
        "components": components,
        "models": {
            "promoted": _model_revision(state.promoted),
            "driving": _model_revision(state.driving),
            "candidate": None
            if state.candidate is None
            else _model_revision(state.candidate),
        },
        "qualification": None
        if outcome is None
        else {
            "candidate_revision": outcome.candidate_revision,
            "names": list(outcome.names),
            "values": list(outcome.values),
            "value": outcome.value,
            "successful": outcome.successful,
            "accepted": outcome.accepted,
            "reason": outcome.reason,
        },
    }
    members: dict[str, bytes] = {_RESTART_STATE: _archive_bytes(manifest, arrays)}
    for name, model in (
        ("promoted", state.promoted),
        ("driving", state.driving),
        ("candidate", state.candidate),
    ):
        if model is not None:
            members[f"{name}.phydrax"] = _model_bytes(model)
    if state.training is not None:
        buffer = plan.buffer(root)
        buffer.require_revision(state.replay_revision, exact=True)
        if state.training_revision != buffer.revision.revision_id:
            raise ValueError("The training result belongs to another replay revision.")
        problem = _replay_problem(plan, state, buffer.revision).problem
        for name, data in _training_restart_members(
            state.training, problem, plan.policy
        ).items():
            members[_RESTART_TRAINING_PREFIX + name] = data
    receipt = {
        "format": _RESTART_FORMAT,
        "plan_id": plan.plan_id,
        "state_id": state.state_id,
        "members": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(members.items())
        },
    }
    members[_RESTART_RECEIPT] = (
        json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    target = root / "restart"
    publish_resource_set(target, members, limits=_RESTART_LIMITS, mode="atomic_replace")
    return target


def _read_model(data: bytes, revision: str, /) -> AbstractAtomisticModel:
    with TemporaryDirectory(prefix="phydrax-distillation-") as scratch:
        path = Path(scratch) / "model.phydrax"
        path.write_bytes(data)
        model = read_atomistic_model_artifact(path, numeric_revision_id=revision).model
    if not isinstance(model, AbstractAtomisticModel):
        raise TypeError("A restart model artifact holds no native atomistic model.")
    return model


def read_atomistic_distillation_restart(
    directory: str | Path, plan: AtomisticOnPolicyDistillationPlan, /
) -> AtomisticOnPolicyDistillationState:
    """Restore the committed distillation state in a fresh process.

    ``plan`` is reconstructed by the caller (provider-free: the student,
    teacher and component templates come from the plan). Every member digest
    must equal the receipt; models are restored from their artifacts pinned to
    the recorded numeric revisions; walker continuations are rebuilt and
    authenticated by the driving student's operation; the training result is
    restored by the training owner against the rebuilt replay problem; the
    experience buffer must hold the committed replay revision (exactly, except
    during a rollout where later frames of the interrupted campaign may follow
    it, and during LABEL where the authenticated label successor of an
    attachment committed before the TRAIN restart is admitted); the
    reconstructed state must reproduce the recorded state identity.
    """
    if not isinstance(plan, AtomisticOnPolicyDistillationPlan):
        raise TypeError("plan must be an AtomisticOnPolicyDistillationPlan.")
    root = Path(directory).expanduser().absolute()
    source = root / "restart"
    with open_bounded_resource_set(
        source.name, trusted_root=source.parent, limits=_RESTART_LIMITS
    ) as bundle:
        receipt = json.loads(
            bundle.read_member(_RESTART_RECEIPT, maximum_bytes=_RECEIPT_BYTES).decode(
                "utf-8"
            )
        )
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"format", "plan_id", "state_id", "members"}
            or receipt["format"] != _RESTART_FORMAT
            or receipt["plan_id"] != plan.plan_id
        ):
            raise ValueError(
                "The distillation restart receipt is not canonical for this plan."
            )
        inventory = {
            member.relative_path
            for member in bundle.manifest.members
            if member.relative_path != _RESTART_RECEIPT
        }
        if inventory != set(receipt["members"]):
            raise ValueError(
                "The distillation restart members differ from their receipt."
            )
        members = {name: bundle.read_member(name) for name in receipt["members"]}
    for name, data in members.items():
        if hashlib.sha256(data).hexdigest() != receipt["members"][name]:
            raise ValueError(f"Distillation restart member {name!r} is corrupt.")
    with TemporaryDirectory(prefix="phydrax-distillation-") as scratch:
        path = Path(scratch) / _RESTART_STATE
        path.write_bytes(members[_RESTART_STATE])
        manifest, arrays = read_array_archive(path, limits=_STATE_LIMITS)
    if (
        manifest.get("format") != _RESTART_FORMAT
        or manifest.get("plan_id") != plan.plan_id
    ):
        raise ValueError("The distillation state archive belongs to another plan.")
    revisions = manifest["models"]
    models: dict[str, AbstractAtomisticModel | None] = {}
    for name in _RESTART_MODELS:
        revision = revisions[name]
        key = f"{name}.phydrax"
        if (revision is None) != (key not in members):
            raise ValueError(f"The restart {name} model inventory is inconsistent.")
        models[name] = None if revision is None else _read_model(members[key], revision)
    template = eqx.filter(plan.components, eqx.is_array)
    restored = unpack_array_tree(manifest["components"], arrays, template)
    components = eqx.combine(
        restored, eqx.filter(plan.components, eqx.is_array, inverse=True)
    )
    record = manifest["qualification"]
    qualification = (
        None
        if record is None
        else AtomisticQualificationOutcome(
            candidate_revision=str(record["candidate_revision"]),
            names=tuple(record["names"]),
            values=tuple(record["values"]),
            value=record["value"],
            successful=bool(record["successful"]),
            accepted=bool(record["accepted"]),
            reason=str(record["reason"]),
        )
    )
    promoted, driving = models["promoted"], models["driving"]
    if promoted is None or driving is None:
        raise ValueError(
            "A distillation restart always holds promoted and driving students."
        )
    walkers = _restored_walkers(plan, driving, manifest["walkers"], arrays)
    trained = manifest["training_revision"] is not None
    training_members = {
        name.removeprefix(_RESTART_TRAINING_PREFIX): data
        for name, data in members.items()
        if name.startswith(_RESTART_TRAINING_PREFIX)
    }
    if trained != bool(training_members):
        raise ValueError(
            "The distillation restart training members differ from its committed "
            "training result."
        )
    # A committed training result is restored on the state of its TRAIN commit.
    state = AtomisticOnPolicyDistillationState(
        plan_id=plan.plan_id,
        segment=int(manifest["segment"]),
        phase="train" if trained else manifest["phase"],
        driving=driving,
        promoted=promoted,
        candidate=models["candidate"],
        components=tuple(components),
        walkers=walkers,
        policy_ids=tuple(manifest["policy_ids"]),
        replay_revision=str(manifest["replay_revision"]),
        labeled_frames=int(manifest["labeled_frames"]),
        training=None,
        training_revision=None,
        qualification=None if trained else qualification,
        promoted_value=manifest["promoted_value"],
        pending_publication=None if trained else manifest["pending_publication"],
        publications=tuple(manifest["publications"]),
    )
    buffer = plan.buffer(root)
    _require_committed_replay(plan, manifest["phase"], state.replay_revision, buffer)
    if trained:
        if manifest["training_revision"] != buffer.revision.revision_id:
            raise ValueError(
                "The committed training result belongs to another replay revision."
            )
        problem = _replay_problem(plan, state, buffer.revision).problem
        _, training = _restore_training_restart_members(
            training_members,
            problem,
            plan.policy,
            components=plan.component_map(plan.components) or None,
        )
        state = state.replace(
            phase=manifest["phase"],
            training=training,
            training_revision=str(manifest["training_revision"]),
            qualification=qualification,
            pending_publication=manifest["pending_publication"],
        )
    if state.state_id != manifest["state_id"] or state.state_id != receipt["state_id"]:
        raise ValueError(
            "The restored distillation state does not reproduce its identity."
        )
    return state


__all__ = [
    "ATOMISTIC_DISTILLATION_PHASES",
    "ATOMISTIC_HVP_TRAINING_REQUEST",
    "BOLTZMANN_GENERATOR_SLOT",
    "BOLTZMANN_GROUP_SLOT",
    "BOLTZMANN_PV_SLOT",
    "HESSIAN_PROBE_SLOT",
    "TEACHER_ATOM_ENERGY_SLOT",
    "TEACHER_EMBEDDING_SLOT",
    "TEACHER_ENERGY_SLOT",
    "TEACHER_FORCES_SLOT",
    "TEACHER_HVP_SLOT",
    "TEACHER_STRESS_SLOT",
    "AtomisticBoltzmannGroup",
    "AtomisticBoltzmannObjective",
    "AtomisticDistillationDrive",
    "AtomisticDistillationPhase",
    "AtomisticDistillationQualification",
    "AtomisticDistillationReport",
    "AtomisticDistillationRollout",
    "AtomisticDistillationSegmentResult",
    "AtomisticEmbeddingObjective",
    "AtomisticEmbeddingProjector",
    "AtomisticEvictionRecord",
    "AtomisticExperienceAdmission",
    "AtomisticExperienceBuffer",
    "AtomisticExperienceFrame",
    "AtomisticFrameLabel",
    "AtomisticGeneratorMixture",
    "AtomisticGeneratorSamples",
    "AtomisticGeneratorState",
    "AtomisticHessianProbe",
    "AtomisticHessianProbeKind",
    "AtomisticHessianVectorObjective",
    "AtomisticLabelRequest",
    "AtomisticOnPolicyDistillationPlan",
    "AtomisticOnPolicyDistillationState",
    "AtomisticQualificationOutcome",
    "AtomisticRefusalRecord",
    "AtomisticReplayDraw",
    "AtomisticReplayRevision",
    "AtomisticTeacher",
    "AtomisticTeacherLabelError",
    "atomistic_boltzmann_source",
    "atomistic_distillation_schema",
    "atomistic_generator_source",
    "atomistic_label_id",
    "label_atomistic_structures",
    "read_atomistic_distillation_restart",
    "require_atomistic_labels",
    "run_atomistic_distillation_segment",
    "write_atomistic_distillation_restart",
]
