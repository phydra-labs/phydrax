#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Streaming atomistic training over immutable structure sources.

`AtomisticTrainingProblem` binds an immutable training loader (source mixture,
field bindings, workload buckets), an optional lineage-separated validation
loader, the graph execution plan, the typed field slots and the declared
objectives. `fit_atomistic_potential` drives the shared role-aware training
kernel with one kernel objective per genuine objective: every loader payload of
a global batch accumulates into one window and the closing payload attempts one
update, so numerators and supports merge before normalization. Full-batch
training is an ordinary one-batch loader. Optimizer transforms, clipping,
parameter selection, low-rank irrep adaptation, EMA targets and the
accepted-update recurrence are the shared kernel's; validation-driven plateau
scaling is a committed rule-state transition. A failed update (nonfinite or
capacity) never advances the committed state or the loader cursor.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from jaxtyping import PyTree

from phydrax.optim._adam import adam

from .._differentiation import ComponentAuthority
from .._doc import DOC_KEY0
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._iteration import IterationSession
from .._jax_config import x64_enabled
from .._strict import StrictModule
from .._trainable import (
    combine_parameters,
    ExplicitFreeze,
    NonTrainableState,
    parameter_field,
)
from .._training import (
    ExponentialMovingAverageTargetPolicy,
    TrainingController,
    TrainingIterationKind,
    TrainingProgress,
)
from .._training_checkpoint import (
    load_training_checkpoint,
    read_training_checkpoint_metadata,
    save_training_checkpoint,
)
from .._training_kernel import (
    AbstractKernelUpdateRule,
    build_training_checkpoint,
    KernelObjective,
    KernelUpdateContext,
    prepare_training_kernel,
    PreparedTrainingKernel,
    restore_training_checkpoint,
    SubspaceTrainingTree,
    TrainingAttemptOutcome,
    TrainingKernelSpec,
    TrainingKernelState,
    TrainingKeys,
)
from .._training_objective import _ObjectiveContribution
from ..axes import Axis, AxisKey
from ..discretization._periodic_cell import lattice_measure
from ..nn.operator.layers._o3 import O3IrrepLinear
from ..nn.parameters import ParameterSubspace
from ..typing import ConvertibleToArray, parse, PRNGKey
from ..units import MASS, UnitDefinition
from ._conditioning import (
    AtomisticConditioning,
    AtomisticConditioningSpec,
    AtomisticConditionLevel,
)
from ._graph import AtomisticGraphExecutionPlan, AtomisticGraphLayout
from ._potential import (
    AbstractAtomisticModel,
    atomistic_model_revision,
    AtomisticStressConvention,
)
from ._structure_store import (
    AbstractAtomisticStructureSource,
    AtomisticStoreField,
    AtomisticStoreLevel,
    AtomisticStructureRecord,
    AtomisticStructureSource,
    AtomisticStructureStoreSchema,
)
from ._training_data import (
    AtomisticFieldBinding,
    AtomisticLevelCapacity,
    AtomisticLoaderCursor,
    AtomisticLoaderItem,
    AtomisticLoaderPayload,
    AtomisticLoaderStaging,
    AtomisticMixtureComponent,
    AtomisticPayloadProvenance,
    AtomisticSourceMixture,
    AtomisticSplitEvidence,
    AtomisticStructureLoader,
    AtomisticWorkloadBucket,
    AtomisticWorkloadBuckets,
)
from ._training_objectives import (
    AbstractAtomisticObjective,
    AbstractAtomisticObjectiveReduction,
    atomistic_stress_unit,
    AtomisticHostLabels,
    AtomisticLocalObjectiveReduction,
    AtomisticObjectiveTerm,
    AtomisticTrainingBatch,
    AtomisticTrainingPrecision,
)
from ._types import (
    AtomisticBatch,
    AtomisticScaleContract,
    AtomisticSpeciesKind,
    AtomisticStatus,
)


AtomisticObjectiveWeighting = Literal["literal", "normalized"]
AtomisticEvaluationView = Literal["current", "target"]

ATOMISTIC_CARTESIAN_AXIS = Axis(AxisKey("atomistic", "cartesian"), 3, ("x", "y", "z"))
"""Canonical Cartesian component axis of atomistic training fields."""

ATOMISTIC_CELL_VECTOR_AXIS = Axis(AxisKey("atomistic", "cell-vector"), 3, ("a", "b", "c"))
"""Canonical row-lattice-vector axis of atomistic cell fields."""


def _name(value: str, label: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty canonical string.")
    return value


def _positive_int(value: int, label: str, /, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer.")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return resolved


def _finite(value: float, label: str, /, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{label} must be a real number.")
    resolved = float(value)
    if not math.isfinite(resolved) or (resolved <= 0.0 if positive else resolved < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be finite and {qualifier}.")
    return resolved


# ------------------------------------------------------------------ fields


@final
class AtomisticTrainingFields(StrictModule, NonTrainableState):
    """Loader slots carrying the bound geometry, identities and conditions.

    ``positions`` are atom-level Cartesian coordinates in the model length
    unit; ``species`` atom-level codes of ``species_kind`` (atomic numbers or
    atom-type IDs); ``masses`` atom-level masses in an explicit MASS-dimension
    unit. Periodic data bind a structure-level ``(3, 3)`` row ``cell`` in the
    length unit and a structure-level ``(3,)`` boolean ``periodic`` mask, which
    are declared together. ``conditions`` maps each field of
    ``conditioning`` to the slot carrying its values (case- or atom-level).
    """

    positions: str = eqx.field(static=True)
    species: str = eqx.field(static=True)
    masses: str = eqx.field(static=True)
    cell: str | None = eqx.field(static=True)
    periodic: str | None = eqx.field(static=True)
    species_kind: AtomisticSpeciesKind = eqx.field(static=True)
    conditioning: AtomisticConditioningSpec | None
    conditions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    fields_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        positions: str = "positions",
        species: str = "species",
        masses: str = "masses",
        cell: str | None = None,
        periodic: str | None = None,
        species_kind: AtomisticSpeciesKind = AtomisticSpeciesKind.ATOMIC_NUMBER,
        conditioning: AtomisticConditioningSpec | None = None,
        conditions: Mapping[str, str] | None = None,
    ) -> None:
        if (cell is None) != (periodic is None):
            raise ValueError("cell and periodic slots are declared together.")
        kind = parse(species_kind, AtomisticSpeciesKind, "species_kind")
        mapping = {} if conditions is None else dict(conditions)
        if conditioning is None:
            if mapping:
                raise ValueError("Condition slots require a conditioning specification.")
        else:
            if not isinstance(conditioning, AtomisticConditioningSpec):
                raise TypeError("conditioning must be an AtomisticConditioningSpec.")
            unknown = set(mapping).difference(conditioning.names)
            if unknown:
                raise ValueError(f"Undeclared condition slots: {sorted(unknown)!r}.")
            missing = [
                field.name
                for field in conditioning.fields
                if field.required and field.name not in mapping
            ]
            if missing:
                raise ValueError(f"Required conditions lack slots: {missing!r}.")
        slots = (positions, species, masses) + tuple(
            value for value in (cell, periodic) if value is not None
        )
        for slot in slots:
            _name(slot, "slot")
        if len(set(slots)) != len(slots):
            raise ValueError("Training field slots must be distinct.")
        self.positions = positions
        self.species = species
        self.masses = masses
        self.cell = cell
        self.periodic = periodic
        self.species_kind = kind
        self.conditioning = conditioning
        self.conditions = tuple(sorted(mapping.items()))
        self.fields_id = canonical_fingerprint(
            {
                "kind": "atomistic-training-fields",
                "positions": positions,
                "species": species,
                "masses": masses,
                "cell": cell,
                "periodic": periodic,
                "species_kind": kind.value,
                "conditioning": None if conditioning is None else conditioning.spec_id,
                "conditions": self.conditions,
            }
        )

    @property
    def slots(self) -> tuple[str, ...]:
        geometry = (self.positions, self.species, self.masses)
        cell = (
            ()
            if self.cell is None or self.periodic is None
            else (self.cell, self.periodic)
        )
        return geometry + cell + tuple(slot for _, slot in self.conditions)


def _bindings(loader: AtomisticStructureLoader, /) -> dict[str, AtomisticFieldBinding]:
    return {binding.slot: binding for binding in loader.bindings}


def _require_field(
    bindings: Mapping[str, AtomisticFieldBinding],
    slot: str,
    /,
    *,
    level: str,
    value_kinds: tuple[str, ...],
    role: str,
    shape: tuple[int, ...],
    unit: UnitDefinition | None = None,
    conventions: tuple[str, ...] | None = None,
) -> AtomisticFieldBinding:
    if slot not in bindings:
        raise ValueError(f"The training loader binds no slot {slot!r}.")
    binding = bindings[slot]
    if binding.level_kind != level or binding.value_kind not in value_kinds:
        raise ValueError(
            f"Slot {slot!r} must be a {level}-level {'/'.join(value_kinds)} field."
        )
    if binding.role != role:
        raise ValueError(f"Slot {slot!r} must have the {role!r} role.")
    if tuple(axis.size for axis in binding.axes) != shape:
        raise ValueError(f"Slot {slot!r} must have trailing shape {shape}.")
    if unit is not None and (
        binding.unit is None or binding.unit.unit_id != unit.unit_id
    ):
        raise ValueError(
            f"Slot {slot!r} must be bound in {unit.symbol}; declare convert_from."
        )
    if conventions is not None and binding.convention not in conventions:
        raise ValueError(
            f"Slot {slot!r} convention {binding.convention!r} is not one of {conventions!r}."
        )
    return binding


def _admit_fields(
    fields: AtomisticTrainingFields,
    bindings: Mapping[str, AtomisticFieldBinding],
    scale: AtomisticScaleContract,
    /,
) -> None:
    _require_field(
        bindings,
        fields.positions,
        level="atom",
        value_kinds=("continuous",),
        role="geometry",
        shape=(3,),
        unit=scale.length_unit,
    )
    _require_field(
        bindings,
        fields.species,
        level="atom",
        value_kinds=("categorical",),
        role="identity",
        shape=(),
        conventions=(fields.species_kind.value,),
    )
    masses = _require_field(
        bindings,
        fields.masses,
        level="atom",
        value_kinds=("continuous",),
        role="identity",
        shape=(),
    )
    if masses.unit is None or masses.unit.dimension != MASS:
        raise ValueError(f"Slot {fields.masses!r} must carry a MASS-dimension unit.")
    if fields.cell is not None and fields.periodic is not None:
        _require_field(
            bindings,
            fields.cell,
            level="structure",
            value_kinds=("continuous",),
            role="geometry",
            shape=(3, 3),
            unit=scale.length_unit,
            conventions=("row-lattice-vectors",),
        )
        _require_field(
            bindings,
            fields.periodic,
            level="structure",
            value_kinds=("mask",),
            role="geometry",
            shape=(3,),
            conventions=("periodic-axes",),
        )
    if fields.conditioning is not None:
        for name, slot in fields.conditions:
            field = fields.conditioning.fields[fields.conditioning.index(name)]
            level = "structure" if field.level is AtomisticConditionLevel.CASE else "atom"
            binding = _require_field(
                bindings,
                slot,
                level=level,
                value_kinds=("continuous", "categorical", "mask", "integer"),
                role="condition",
                shape=field.port.event_shape,
                unit=field.unit,
            )
            if (binding.unit is None) != (field.unit is None):
                raise ValueError(f"Condition slot {slot!r} unit declaration differs.")


# ------------------------------------------------------------- preparation


@final
class AtomisticBatchPreparation(StrictModule, NonTrainableState):
    """Host preparation of one loader's payloads into graph-bound batches.

    Binds the loader, the graph execution plan, the scale, ``cutoff + skin``,
    the field slots and the label slots carried into every batch. It needs no
    objective: training problems compose one per loader, and native teacher
    labeling prepares unlabeled payloads with ``label_slots=()``. Labels in
    ``cell_slots`` are only defined on cases with a periodic axis and a finite
    nonsingular cell and are refused elsewhere. ``preparation_id`` binds the
    loader semantics and every scientific preparation parameter;
    ``execution_id`` the graph execution plan and loader layouts.
    """

    loader: AtomisticStructureLoader
    graph_execution: AtomisticGraphExecutionPlan
    fields: AtomisticTrainingFields
    scale: AtomisticScaleContract
    cutoff: float = eqx.field(static=True)
    skin: float = eqx.field(static=True)
    label_slots: tuple[str, ...] = eqx.field(static=True)
    label_levels: tuple[Literal["structure", "atom"], ...] = eqx.field(static=True)
    cell_slots: tuple[str, ...] = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)

    def __init__(
        self,
        loader: AtomisticStructureLoader,
        graph_execution: AtomisticGraphExecutionPlan,
        /,
        *,
        scale: AtomisticScaleContract,
        cutoff: float,
        skin: float = 0.0,
        fields: AtomisticTrainingFields | None = None,
        label_slots: Sequence[str] = (),
        cell_slots: Sequence[str] = (),
    ) -> None:
        if not isinstance(loader, AtomisticStructureLoader):
            raise TypeError("loader must be an AtomisticStructureLoader.")
        if not isinstance(graph_execution, AtomisticGraphExecutionPlan):
            raise TypeError("graph_execution must be an AtomisticGraphExecutionPlan.")
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        fields_ = AtomisticTrainingFields() if fields is None else fields
        if not isinstance(fields_, AtomisticTrainingFields):
            raise TypeError("fields must be AtomisticTrainingFields or None.")
        cutoff_ = _finite(cutoff, "cutoff", positive=True)
        skin_ = _finite(skin, "skin")
        slots = tuple(_name(slot, "label slot") for slot in label_slots)
        cells = tuple(_name(slot, "cell slot") for slot in cell_slots)
        if len(set(slots)) != len(slots) or not set(cells) <= set(slots):
            raise ValueError("Label slots are distinct and include every cell slot.")
        bindings = _bindings(loader)
        _admit_fields(fields_, bindings, scale)
        if set(slots) & set(fields_.slots):
            raise ValueError("Label slots must differ from geometry and condition slots.")
        schema = loader.mixture.schema
        for slot in slots:
            if slot not in bindings:
                raise ValueError(f"The loader binds no label slot {slot!r}.")
            binding = bindings[slot]
            if binding.level_kind not in ("structure", "atom"):
                raise ValueError(f"Label slot {slot!r} must be structure- or atom-level.")
            if (
                binding.value_kind == "continuous"
                and np.dtype(binding.dtype or schema.field(binding.field).dtype).itemsize
                == 8
                and not x64_enabled()
            ):
                raise ValueError(
                    f"Label slot {slot!r} is float64; enable JAX x64 so label precision "
                    "is preserved through ingest."
                )
        for slot in cells:
            if bindings[slot].level_kind != "structure":
                raise ValueError(
                    f"Cell-bound label slot {slot!r} must be structure-level."
                )
        levels: tuple[Literal["structure", "atom"], ...] = tuple(
            "structure" if bindings[slot].level_kind == "structure" else "atom"
            for slot in slots
        )
        self.loader = loader
        self.graph_execution = graph_execution
        self.fields = fields_
        self.scale = scale
        self.cutoff = cutoff_
        self.skin = skin_
        self.label_slots = slots
        self.label_levels = levels
        self.cell_slots = cells
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "atomistic-batch-preparation",
                "loader": loader.semantics_id,
                "scale": scale.scale_id,
                "fields": fields_.fields_id,
                "cutoff": cutoff_,
                "skin": skin_,
                "label_slots": slots,
                "cell_slots": cells,
            }
        )
        self.execution_id = canonical_fingerprint(
            {
                "kind": "atomistic-batch-preparation-execution",
                "graph_execution": graph_execution.plan_id,
                "loader": loader.structure_id,
            }
        )

    def admit(self, model: AbstractAtomisticModel, /) -> None:
        """Refuse a model whose scale, species kind or cutoff the data cannot serve."""
        if not isinstance(model, AbstractAtomisticModel):
            raise TypeError("model must implement AbstractAtomisticModel.")
        if model.scale.scale_id != self.scale.scale_id:
            raise ValueError("The model and the prepared data scales differ.")
        if model.capabilities.species_kind is not self.fields.species_kind:
            raise ValueError(
                "The model and the training fields declare other species kinds."
            )
        if float(model.configuration.cutoff) > self.cutoff:
            raise ValueError("The preparation cutoff is smaller than the model cutoff.")

    def _geometry(
        self, payload: AtomisticLoaderPayload, /
    ) -> tuple[int, int, np.ndarray]:
        schema = self.loader.mixture.schema
        layout = payload.layout
        cases = layout.case_capacity * layout.local_shards
        atom = payload.level(schema.atom_level)
        rows = np.asarray(atom.row_mask, dtype=np.bool_)
        atoms = rows.size // cases
        return cases, atoms, rows.reshape((cases, atoms))

    @staticmethod
    def _label(
        payload: AtomisticLoaderPayload, slot: str, level: str, cases: int, atoms: int, /
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(payload.value(slot))
        available = np.asarray(payload.availability(slot), dtype=np.bool_)
        if level == "structure":
            return values.reshape((cases, *values.shape[1:])), available.reshape((cases,))
        return (
            values.reshape((cases, atoms, *values.shape[1:])),
            available.reshape((cases, atoms)),
        )

    def host_labels(self, payload: AtomisticLoaderPayload, /) -> AtomisticHostLabels:
        """Host label view of one whole payload (for fitted normalization moments)."""
        cases, atoms, mask = self._geometry(payload)
        species = np.asarray(payload.value(self.fields.species)).astype(np.int64)
        values: dict[str, np.ndarray] = {}
        available: dict[str, np.ndarray] = {}
        for slot, level in zip(self.label_slots, self.label_levels, strict=True):
            values[slot], available[slot] = self._label(
                payload, slot, level, cases, atoms
            )
        return AtomisticHostLabels(
            case_valid=np.asarray(payload.case_valid, dtype=np.bool_),
            atom_mask=mask,
            species=species.reshape((cases, atoms)),
            values=values,
            available=available,
        )

    def prepare_batch(
        self,
        payload: AtomisticLoaderPayload,
        provenance: AtomisticPayloadProvenance,
        /,
        *,
        coordinate_dtype: Any,
        shard: int = 0,
    ) -> AtomisticTrainingBatch:
        """Host-prepare one local data shard of a payload.

        Local shard ``shard`` owns cases ``[shard * C, (shard + 1) * C)`` of the
        shard-major payload (``C`` the bucket case capacity) and the matching
        rows of the per-case atom level; only structure and atom levels are
        bound, so no index rebasing occurs. Stable particle identities come from
        the payload; padding cases keep their real geometry and zero weight.

        The graph payload is bound to the ``AtomisticGraphLayout`` of the
        bucket's case/atom capacities, coordinate dtype and periodicity class
        (cells declared: every axis may be periodic). The layout is a pure
        function of that configuration, so every payload and shard of a bucket
        has one static tree definition and reuses one compiled route search,
        evaluation and step; record and draw identities stay host provenance
        and dynamic case address words.
        """
        total, atoms, full_mask = self._geometry(payload)
        layout = payload.layout
        if not 0 <= shard < layout.local_shards:
            raise ValueError(f"shard must lie in [0, {layout.local_shards}).")
        cases = layout.case_capacity
        rows = slice(shard * cases, (shard + 1) * cases)
        fields = self.fields

        def per_case(values: np.ndarray, /) -> np.ndarray:
            return values.reshape((total, *values.shape[1:]))[rows]

        def per_atom(values: np.ndarray, /) -> np.ndarray:
            return values.reshape((total, atoms, *values.shape[1:]))[rows]

        mask = full_mask[rows]
        positions = per_atom(np.asarray(payload.value(fields.positions)))
        codes = per_atom(np.asarray(payload.value(fields.species)).astype(np.int64))
        masses = per_atom(np.asarray(payload.value(fields.masses)))
        words = per_atom(np.asarray(payload.stable_id_words, dtype=np.uint64))
        stable = ((words[..., 0] << np.uint64(32)) | words[..., 1]).astype(np.int64)
        spare = np.max(np.where(mask, stable, 0), axis=1, keepdims=True) + 1
        identities = np.where(
            mask, stable, spare + np.arange(atoms, dtype=np.int64)[None, :]
        )
        codes = np.where(mask, codes, 0)
        if fields.species_kind is AtomisticSpeciesKind.ATOMIC_NUMBER:
            numbers, types, elements = codes, codes, mask
        else:
            numbers = np.zeros_like(codes)
            types = codes
            elements = np.zeros_like(mask)
        cells = periodic = None
        stress_cells = np.zeros((cases,), dtype=np.bool_)
        if fields.cell is not None and fields.periodic is not None:
            cells = per_case(np.asarray(payload.value(fields.cell), dtype=np.float64))
            periodic = per_case(
                np.asarray(payload.value(fields.periodic), dtype=np.bool_)
            )
            _, embedded = lattice_measure(cells)
            stress_cells = np.any(periodic, axis=1) & np.asarray(embedded, dtype=np.bool_)
        valid = per_case(np.asarray(payload.case_valid, dtype=np.bool_))
        record_ids = provenance.record_ids[rows]
        structure_ids = tuple(
            (
                f"{source[:16]}/{record}/draw-{draw}"
                if is_valid
                else f"padding-{shard}-{index}/{record}"
            )
            for index, (source, record, draw, is_valid) in enumerate(
                zip(
                    provenance.source_ids[rows],
                    record_ids,
                    provenance.draw_indices[rows],
                    provenance.valid[rows],
                    strict=True,
                )
            )
        )
        structures = AtomisticBatch(
            numbers,
            positions,
            np.where(mask, masses, 1.0),
            self.scale,
            particle_ids=identities,
            atom_type_ids=types,
            element_mask=elements,
            atom_mask=mask,
            cells=cells,
            periodic_axes=periodic,
            structure_ids=structure_ids,
            coordinate_dtype=coordinate_dtype,
        )
        labels: list[Array] = []
        available: list[Array] = []
        for slot, level in zip(self.label_slots, self.label_levels, strict=True):
            values, label_mask = self._label(payload, slot, level, total, atoms)
            values, label_mask = values[rows], label_mask[rows]
            if level == "atom":
                label_mask = label_mask & mask
            if slot in self.cell_slots:
                refused = valid & label_mask & ~stress_cells
                if np.any(refused):
                    record = record_ids[int(np.flatnonzero(refused)[0])]
                    raise ValueError(
                        f"Record {record!r} carries a {slot!r} label without a "
                        "periodic axis and a finite nonsingular cell; "
                        "cell-volume-normalized stress needs that declared volume."
                    )
            labels.append(jnp.asarray(values))
            available.append(jnp.asarray(label_mask))
        conditioning = None
        if fields.conditioning is not None:
            spec = fields.conditioning
            condition_values: dict[str, Any] = {}
            for name, slot in fields.conditions:
                field = spec.fields[spec.index(name)]
                value = np.asarray(payload.value(slot))
                value = (
                    per_case(value)
                    if field.level is AtomisticConditionLevel.CASE
                    else per_atom(value).reshape((cases * atoms, *value.shape[1:]))
                )
                condition_values[name] = value.reshape(
                    (value.shape[0], *field.port.event_shape)
                )
            conditioning = AtomisticConditioning(
                spec, condition_values, case_count=cases, atom_capacity=atoms
            )
        graph_layout = AtomisticGraphLayout(
            self.scale,
            self.graph_execution,
            case_capacity=cases,
            atom_capacity=atoms,
            coordinate_dtype=structures.positions.dtype,
            cutoff=self.cutoff,
            skin=self.skin,
            periodic_axes=(cells is not None,) * 3,
            cells=cells is not None,
        )
        return AtomisticTrainingBatch(
            graph=graph_layout.prepare(structures),
            conditioning=conditioning,
            case_valid=jnp.asarray(valid),
            case_weight=jnp.asarray(per_case(np.asarray(payload.case_weight))),
            draw_index=jnp.asarray(per_case(np.asarray(payload.draw_index))),
            case_epoch=jnp.asarray(per_case(np.asarray(payload.case_epoch))),
            stress_cells=jnp.asarray(stress_cells),
            labels=tuple(labels),
            available=tuple(available),
            label_slots=self.label_slots,
            label_levels=self.label_levels,
        )

    def prepare_step(
        self,
        payload: AtomisticLoaderPayload,
        provenance: AtomisticPayloadProvenance,
        reduction: AbstractAtomisticObjectiveReduction,
        /,
        *,
        coordinate_dtype: Any,
    ) -> Any:
        """Prepare every local data shard and place them with ``reduction``."""
        shards = payload.layout.local_shards
        if reduction.local_shards != shards:
            raise ValueError(
                f"The reduction places {reduction.local_shards} local shards; the "
                f"payload carries {shards}."
            )
        return reduction.assemble(
            tuple(
                self.prepare_batch(
                    payload, provenance, coordinate_dtype=coordinate_dtype, shard=shard
                )
                for shard in range(shards)
            )
        )


# ------------------------------------------------------------------ problem


@final
class AtomisticTrainingProblem(StrictModule, NonTrainableState):
    """Immutable training data, graph preparation and declared objectives.

    ``training`` is the training loader (its source mixture, field bindings and
    workload buckets); ``validation`` an optional loader whose independent-unit
    lineage, record identities and content digests are disjoint from training
    (``AtomisticSplitEvidence``). Both bind identical slot contracts and are
    prepared by their own ``AtomisticBatchPreparation`` (``preparation`` and
    ``validation_preparation``). ``weighting="normalized"`` divides every
    objective's schedule weight by their sum at the same committed update;
    ``"literal"`` uses the weights as declared. ``problem_id`` binds the
    scientific data, objectives and weighting; ``execution_id`` binds the graph
    execution plan and loader layouts, which may change across a continuation
    (for example to grow a capacity) without changing the problem.
    """

    training: AtomisticStructureLoader
    validation: AtomisticStructureLoader | None
    preparation: AtomisticBatchPreparation
    validation_preparation: AtomisticBatchPreparation | None
    objectives: tuple[AbstractAtomisticObjective, ...]
    precision: AtomisticTrainingPrecision
    split_evidence: AtomisticSplitEvidence | None
    weighting: AtomisticObjectiveWeighting = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)

    def __init__(
        self,
        training: AtomisticStructureLoader,
        graph_execution: AtomisticGraphExecutionPlan,
        objectives: Sequence[AbstractAtomisticObjective],
        /,
        *,
        scale: AtomisticScaleContract,
        cutoff: float,
        skin: float = 0.0,
        fields: AtomisticTrainingFields | None = None,
        validation: AtomisticStructureLoader | None = None,
        weighting: AtomisticObjectiveWeighting = "literal",
        precision: AtomisticTrainingPrecision | None = None,
    ) -> None:
        precision_ = AtomisticTrainingPrecision() if precision is None else precision
        if not isinstance(precision_, AtomisticTrainingPrecision):
            raise TypeError("precision must be an AtomisticTrainingPrecision or None.")
        objectives_ = tuple(objectives)
        if not objectives_ or any(
            not isinstance(value, AbstractAtomisticObjective) for value in objectives_
        ):
            raise TypeError(
                "objectives must be nonempty AbstractAtomisticObjective values."
            )
        names = [value.name for value in objectives_]
        if len(set(names)) != len(names):
            raise ValueError("Objective names must be unique within a problem.")
        weighting_ = parse(weighting, AtomisticObjectiveWeighting, "weighting")
        if (
            weighting_ == "normalized"
            and max(value.schedule.minimum for value in objectives_) <= 0.0
        ):
            raise ValueError(
                "Normalized weighting needs an objective whose schedule is positive "
                "at every committed update."
            )
        slots: list[str] = []
        cell_slots: list[str] = []
        for objective in objectives_:
            for slot in objective.label_slots:
                if slot not in slots:
                    slots.append(slot)
            for slot in objective.admissible_cell_slots:
                if slot not in cell_slots:
                    cell_slots.append(slot)
        preparation = AtomisticBatchPreparation(
            training,
            graph_execution,
            scale=scale,
            cutoff=cutoff,
            skin=skin,
            fields=fields,
            label_slots=tuple(slots),
            cell_slots=tuple(cell_slots),
        )
        evidence = None
        validation_preparation = None
        if validation is not None:
            validation_preparation = AtomisticBatchPreparation(
                validation,
                graph_execution,
                scale=scale,
                cutoff=cutoff,
                skin=skin,
                fields=preparation.fields,
                label_slots=tuple(slots),
                cell_slots=tuple(cell_slots),
            )
            training_bindings = _bindings(training)
            other = _bindings(validation)
            for slot in (*preparation.fields.slots, *slots):
                if other[slot].binding_id != training_bindings[slot].binding_id:
                    raise ValueError(
                        f"Validation slot {slot!r} differs from the training binding."
                    )
            evidence = AtomisticSplitEvidence(
                {"training": training.mixture, "validation": validation.mixture}
            )
        self.training = training
        self.validation = validation
        self.preparation = preparation
        self.validation_preparation = validation_preparation
        self.objectives = objectives_
        self.precision = precision_
        self.split_evidence = evidence
        self.weighting = weighting_
        self.problem_id = canonical_fingerprint(
            {
                "kind": "atomistic-training-problem",
                "preparation": preparation.preparation_id,
                "validation": None if validation is None else validation.semantics_id,
                "split": None if evidence is None else "lineage-disjoint",
                "objectives": [value.objective_id for value in objectives_],
                "weighting": weighting_,
                "precision": precision_.precision_id,
            }
        )
        self.execution_id = canonical_fingerprint(
            {
                "kind": "atomistic-training-execution",
                "training": preparation.execution_id,
                "validation": (
                    None
                    if validation_preparation is None
                    else validation_preparation.execution_id
                ),
            }
        )

    @property
    def scale(self) -> AtomisticScaleContract:
        return self.preparation.scale

    @property
    def fields(self) -> AtomisticTrainingFields:
        return self.preparation.fields

    @property
    def graph_execution(self) -> AtomisticGraphExecutionPlan:
        return self.preparation.graph_execution

    @property
    def cutoff(self) -> float:
        return self.preparation.cutoff

    @property
    def label_slots(self) -> tuple[str, ...]:
        return self.preparation.label_slots

    def admit(self, model: AbstractAtomisticModel, /) -> None:
        """Refuse a model, field or objective mismatch before any optimizer state."""
        self.preparation.admit(model)
        bindings = _bindings(self.training)
        for objective in self.objectives:
            objective.admit(model, bindings, self.scale)


# ------------------------------------------------------------ normalization


@final
class AtomisticTrainingNormalization(StrictModule, NonTrainableState):
    """Physical residual scale of every objective, fitted or declared.

    A ``None`` objective scale is fitted from one full epoch of the training
    loader only (root-mean-square label magnitude, or the standard deviation
    of the scaled energy for energy objectives); a fit without labels, with
    nonfinite labels or below ``floor`` is refused rather than clamped.
    ``counts`` are the label entries each fit used.
    """

    names: tuple[str, ...] = eqx.field(static=True)
    scales: tuple[float, ...] = eqx.field(static=True)
    fitted: tuple[bool, ...] = eqx.field(static=True)
    counts: tuple[int, ...] = eqx.field(static=True)
    floor: float = eqx.field(static=True)
    fitted_from: str = eqx.field(static=True)
    normalization_id: str = eqx.field(static=True)

    def scale(self, name: str, /) -> float:
        return self.scales[self.names.index(name)]


def fit_atomistic_normalization(
    problem: AtomisticTrainingProblem, /, *, floor: float = 1e-12
) -> AtomisticTrainingNormalization:
    """Resolve every objective scale; fitted scales read one training epoch."""
    if not isinstance(problem, AtomisticTrainingProblem):
        raise TypeError("problem must be an AtomisticTrainingProblem.")
    floor_ = _finite(floor, "floor", positive=True)
    objectives = problem.objectives
    moments = [np.zeros((3,), dtype=np.float64) for _ in objectives]
    if any(objective.scale is None for objective in objectives):
        loader = problem.training
        with loader.epoch(loader.start(0)) as epoch:
            for item in epoch:
                labels = problem.preparation.host_labels(item.payload)
                for index, objective in enumerate(objectives):
                    if objective.scale is None:
                        moments[index] += objective.label_moments(labels)
    scales = tuple(
        objective.scale
        if objective.scale is not None
        else objective.fitted_scale(moment, floor_)
        for objective, moment in zip(objectives, moments, strict=True)
    )
    names = tuple(objective.name for objective in objectives)
    fitted = tuple(objective.scale is None for objective in objectives)
    counts = tuple(int(moment[0]) for moment in moments)
    return AtomisticTrainingNormalization(
        names=names,
        scales=scales,
        fitted=fitted,
        counts=counts,
        floor=floor_,
        fitted_from=problem.training.semantics_id,
        normalization_id=canonical_fingerprint(
            {
                "kind": "atomistic-training-normalization",
                "problem": problem.problem_id,
                "names": names,
                "scales": [float(value).hex() for value in scales],
                "fitted": fitted,
                "counts": counts,
                "floor": floor_,
            }
        ),
    )


# ------------------------------------------------------------- adaptation


@final
class AtomisticParameterSelection(StrictModule, NonTrainableState):
    """Exact trained-parameter selection; every other leaf stays frozen.

    Paths are resolved by the shared ``ParameterSubspace`` against the model
    passed to the trainer (``leaf_paths`` exact leaves, ``subtree_paths`` all
    selectable leaves below disjoint subtrees).
    """

    leaf_paths: tuple[str, ...] = eqx.field(static=True)
    subtree_paths: tuple[str, ...] = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)

    def __init__(
        self, *, leaf_paths: Sequence[str] = (), subtree_paths: Sequence[str] = ()
    ) -> None:
        leaves = tuple(str(path) for path in leaf_paths)
        subtrees = tuple(str(path) for path in subtree_paths)
        if bool(leaves) == bool(subtrees):
            raise ValueError("Declare exactly one of leaf_paths or subtree_paths.")
        self.leaf_paths = leaves
        self.subtree_paths = subtrees
        self.selection_id = canonical_fingerprint(
            {
                "kind": "atomistic-parameter-selection",
                "leaves": leaves,
                "subtrees": subtrees,
            }
        )

    def subspace(self, model: Any, /) -> ParameterSubspace:
        if self.leaf_paths:
            return ParameterSubspace.from_leaf_paths(model, self.leaf_paths)
        return ParameterSubspace.from_subtree_paths(model, self.subtree_paths)


@final
class _FrozenModel(StrictModule, ExplicitFreeze):
    """The adapted base model, frozen on purpose."""

    model: Any


def _path_node(tree: Any, path: tuple[Any, ...], /) -> Any:
    node = tree
    for entry in path:
        if isinstance(entry, jax.tree_util.GetAttrKey):
            node = getattr(node, entry.name)
        elif isinstance(entry, jax.tree_util.SequenceKey):
            node = node[entry.idx]
        elif isinstance(entry, jax.tree_util.DictKey):
            node = node[entry.key]
        else:
            raise ValueError(f"Unsupported adaptation path entry {entry!r}.")
    return node


def _irrep_site_paths(model: Any, /) -> dict[str, tuple[Any, ...]]:
    flattened = jax.tree_util.tree_flatten_with_path(
        model, is_leaf=lambda value: isinstance(value, O3IrrepLinear)
    )[0]
    return {
        jax.tree_util.keystr(path): path
        for path, leaf in flattened
        if isinstance(leaf, O3IrrepLinear)
    }


def atomistic_irrep_linear_sites(model: Any, /) -> tuple[str, ...]:
    """Paths of every ``O3IrrepLinear`` multiplicity map in ``model``."""
    return tuple(_irrep_site_paths(model))


@final
class AtomisticIrrepLowRankAdaptation(StrictModule, NonTrainableState):
    """Low-rank adaptation inside the multiplicity spaces of equivariant maps.

    Each selected ``O3IrrepLinear`` site maps channels of identical degree and
    parity per output block, ``W_k`` of shape ``(out multiplicity, fan-in)``.
    The adaptation trains ``W_k + (alpha / r) A_k B_k`` with ``A_k`` zero at
    initialization and ``B_k`` Gaussian, ``r = min(rank, out, fan-in)``. The
    update acts on multiplicity indices only, so every block stays an O(3)
    equivariant map of its own irrep; unequal irreps are never mixed. A dense
    adapter over packed features is not admitted.
    """

    sites: tuple[str, ...] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    stddev: float = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    adaptation_id: str = eqx.field(static=True)

    def __init__(
        self,
        sites: Sequence[str],
        /,
        *,
        rank: int,
        alpha: float | None = None,
        stddev: float = 0.01,
        seed: int = 0,
    ) -> None:
        sites_ = tuple(str(site) for site in sites)
        if not sites_ or len(set(sites_)) != len(sites_):
            raise ValueError("sites must name distinct O3IrrepLinear paths.")
        rank_ = _positive_int(rank, "rank")
        alpha_ = float(rank_) if alpha is None else _finite(alpha, "alpha", positive=True)
        self.sites = sites_
        self.rank = rank_
        self.alpha = alpha_
        self.stddev = _finite(stddev, "stddev", positive=True)
        self.seed = _positive_int(seed, "seed", minimum=0)
        self.adaptation_id = canonical_fingerprint(
            {
                "kind": "atomistic-irrep-low-rank-adaptation",
                "sites": sites_,
                "rank": rank_,
                "alpha": alpha_,
                "stddev": self.stddev,
                "seed": self.seed,
            }
        )

    def tree(self, model: Any, /) -> AtomisticIrrepLowRankTree:
        available = _irrep_site_paths(model)
        missing = [site for site in self.sites if site not in available]
        if missing:
            raise ValueError(f"Unknown O3IrrepLinear adaptation sites: {missing!r}.")
        key = jax.random.key(self.seed)
        left: list[tuple[Array, ...]] = []
        right: list[tuple[Array, ...]] = []
        for site in self.sites:
            linear = _path_node(model, available[site])
            site_left: list[Array] = []
            site_right: list[Array] = []
            for weight in linear.weights:
                rows, columns = weight.shape
                rank = min(self.rank, rows, columns)
                key, subkey = jax.random.split(key)
                site_left.append(jnp.zeros((rows, rank), dtype=weight.dtype))
                site_right.append(
                    jax.random.normal(subkey, (rank, columns), dtype=weight.dtype)
                    * self.stddev
                )
            left.append(tuple(site_left))
            right.append(tuple(site_right))
        return AtomisticIrrepLowRankTree(
            _FrozenModel(model),
            tuple(left),
            tuple(right),
            sites=self.sites,
            scales=tuple(
                tuple(self.alpha / factor.shape[1] for factor in site_left)
                for site_left in left
            ),
        )


@final
class AtomisticIrrepLowRankTree(StrictModule):
    """Kernel tree: frozen base model plus trainable multiplicity factors.

    ``sites`` are ``O3IrrepLinear`` paths (``jax.tree_util.keystr``) of the base
    model; ``scales[s][k]`` is ``alpha / r`` of block ``k`` of site ``s``.
    """

    base: _FrozenModel
    left: tuple[tuple[Array, ...], ...] = parameter_field()
    right: tuple[tuple[Array, ...], ...] = parameter_field()
    sites: tuple[str, ...] = eqx.field(static=True)
    scales: tuple[tuple[float, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        base: _FrozenModel,
        left: tuple[tuple[Array, ...], ...],
        right: tuple[tuple[Array, ...], ...],
        /,
        *,
        sites: tuple[str, ...],
        scales: tuple[tuple[float, ...], ...],
    ) -> None:
        if not len(sites) == len(left) == len(right) == len(scales):
            raise ValueError("Every adapted site needs its factors and scales.")
        self.base = base
        self.left = left
        self.right = right
        self.sites = sites
        self.scales = scales

    def model(self) -> Any:
        """Base model with every adapted block merged as ``W + s A B``."""
        merged = self.base.model
        paths = _irrep_site_paths(merged)
        for site, lefts, rights, scales in zip(
            self.sites, self.left, self.right, self.scales, strict=True
        ):
            path = paths[site]
            linear = _path_node(merged, path)
            weights = tuple(
                weight + scale * (a @ b)
                for weight, a, b, scale in zip(
                    linear.weights, lefts, rights, scales, strict=True
                )
            )
            merged = eqx.tree_at(
                lambda tree, path=path: _path_node(tree, path).weights, merged, weights
            )
        return merged


AtomisticAdaptation = AtomisticParameterSelection | AtomisticIrrepLowRankAdaptation


@final
class AtomisticTrainedTree(StrictModule):
    """The kernel's trained tree: the (adapted) model and auxiliary components.

    ``components`` aligns with the problem objectives; an objective that needs a
    trainable component (for example an embedding projector) finds it at its
    own position, every other position is ``None``.
    """

    model: Any
    components: tuple[Any, ...]

    def potential(self) -> AbstractAtomisticModel:
        model = self.model
        if isinstance(model, SubspaceTrainingTree):
            return model.model()
        if isinstance(model, AtomisticIrrepLowRankTree):
            return model.model()
        return model

    def base_potential(self) -> AbstractAtomisticModel:
        """The adapted base model: the restore template of a training checkpoint.

        Under a parameter selection it is the selection's initial model; under
        low-rank adaptation the frozen base; otherwise the trained model.
        """
        model = self.model
        if isinstance(model, SubspaceTrainingTree):
            subspace = model.complement.subspace
            return subspace.reconstruct(subspace.initial)
        if isinstance(model, AtomisticIrrepLowRankTree):
            return model.base.model
        return model


def _trained_tree(
    model: AbstractAtomisticModel,
    problem: AtomisticTrainingProblem,
    adaptation: AtomisticAdaptation | None,
    components: Mapping[str, Any] | None,
    /,
) -> AtomisticTrainedTree:
    supplied = {} if components is None else dict(components)
    names = {objective.name for objective in problem.objectives}
    unknown = set(supplied).difference(names)
    if unknown:
        raise ValueError(f"Components name unknown objectives: {sorted(unknown)!r}.")
    aligned: list[Any] = []
    for objective in problem.objectives:
        component = supplied.get(objective.name)
        if objective.requires_component and component is None:
            raise ValueError(f"Objective {objective.name!r} needs a trainable component.")
        objective.admit_component(component)
        aligned.append(component)
    if adaptation is None:
        tree_model: Any = model
    elif isinstance(adaptation, AtomisticParameterSelection):
        tree_model = SubspaceTrainingTree.from_subspace(adaptation.subspace(model))
    elif isinstance(adaptation, AtomisticIrrepLowRankAdaptation):
        tree_model = adaptation.tree(model)
    else:
        raise TypeError("adaptation must be an AtomisticAdaptation or None.")
    return AtomisticTrainedTree(tree_model, tuple(aligned))


# ------------------------------------------------------------- update rule


@final
class AtomisticPlateauPolicy(StrictModule, NonTrainableState):
    """Validation-driven learning-rate scaling as a committed rule transition.

    A validation that is not below ``best - threshold`` increments the stale
    count (ignored during ``cooldown`` validations after a reduction); more than
    ``patience`` stale validations multiply the update scale by ``factor``
    (never below ``minimum_scale``) and start the cooldown.
    """

    factor: float = eqx.field(static=True)
    patience: int = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    cooldown: int = eqx.field(static=True)
    minimum_scale: float = eqx.field(static=True)
    plateau_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        factor: float = 0.5,
        patience: int = 0,
        threshold: float = 0.0,
        cooldown: int = 0,
        minimum_scale: float = 0.0,
    ) -> None:
        factor_ = _finite(factor, "factor", positive=True)
        if factor_ >= 1.0:
            raise ValueError("factor must lie in (0, 1).")
        self.factor = factor_
        self.patience = _positive_int(patience, "patience", minimum=0)
        self.threshold = _finite(threshold, "threshold")
        self.cooldown = _positive_int(cooldown, "cooldown", minimum=0)
        self.minimum_scale = _finite(minimum_scale, "minimum_scale")
        self.plateau_id = canonical_fingerprint(
            {
                "kind": "atomistic-plateau-policy",
                "factor": factor_,
                "patience": self.patience,
                "threshold": self.threshold,
                "cooldown": self.cooldown,
                "minimum_scale": self.minimum_scale,
            }
        )


@final
class AtomisticOptimizerState(StrictModule):
    """Committed optimizer and plateau rule state of one atomistic run."""

    inner: Any
    update_scale: Array
    best_validation: Array
    stale_validations: Array
    cooldown: Array


@final
class _AtomisticUpdateRule(AbstractKernelUpdateRule):
    """Optax transformation with a committed validation-plateau update scale."""

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ()
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self, optimizer: optax.GradientTransformation, /, *, rule_id: str
    ) -> None:
        self.optimizer = optimizer
        self.rule_id = rule_id

    @property
    def supports_accumulation(self) -> bool:
        return True

    def init(self, parameters: PyTree[Any], /) -> AtomisticOptimizerState:
        dtype = jnp.result_type(
            *(jnp.real(leaf).dtype for leaf in jax.tree_util.tree_leaves(parameters))
        )
        return AtomisticOptimizerState(
            inner=self.optimizer.init(parameters),
            update_scale=jnp.ones((), dtype=dtype),
            best_validation=jnp.asarray(jnp.inf, dtype=dtype),
            stale_validations=jnp.zeros((), dtype=jnp.int32),
            cooldown=jnp.zeros((), dtype=jnp.int32),
        )

    def propose(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        value: Array,
        rule_state: AtomisticOptimizerState,
        context: KernelUpdateContext,
        /,
    ) -> tuple[PyTree[Any], AtomisticOptimizerState, AtomisticOptimizerState, Array]:
        del context
        if isinstance(self.optimizer, optax.GradientTransformationExtraArgs):
            updates, inner = self.optimizer.update(
                gradients, rule_state.inner, parameters, value=value, grad=gradients
            )
        else:
            updates, inner = self.optimizer.update(
                gradients, rule_state.inner, parameters
            )
        scaled = jax.tree.map(
            lambda update: update * rule_state.update_scale.astype(update.dtype), updates
        )
        candidate = optax.apply_updates(parameters, scaled)
        return (
            candidate,
            dataclasses.replace(rule_state, inner=inner),
            rule_state,
            jnp.asarray(True),
        )

    def rule_state_finite(self, rule_state: AtomisticOptimizerState, /) -> Array:
        finite = jnp.isfinite(rule_state.update_scale)
        for leaf in jax.tree_util.tree_leaves(rule_state.inner):
            if eqx.is_inexact_array(leaf):
                finite = finite & jnp.all(jnp.isfinite(leaf))
        return finite


def _plateau_transition(
    policy: AtomisticPlateauPolicy, state: AtomisticOptimizerState, value: float, /
) -> AtomisticOptimizerState:
    best = float(np.asarray(state.best_validation))
    stale = int(np.asarray(state.stale_validations))
    cooldown = int(np.asarray(state.cooldown))
    scale = float(np.asarray(state.update_scale))
    if value < best - policy.threshold:
        best, stale = value, 0
    else:
        stale += 1
    if cooldown > 0:
        cooldown -= 1
        stale = 0
    if stale > policy.patience:
        scale = max(scale * policy.factor, policy.minimum_scale)
        cooldown, stale = policy.cooldown, 0
    dtype = state.update_scale.dtype
    return dataclasses.replace(
        state,
        update_scale=jnp.asarray(scale, dtype=dtype),
        best_validation=jnp.asarray(best, dtype=dtype),
        stale_validations=jnp.asarray(stale, dtype=jnp.int32),
        cooldown=jnp.asarray(cooldown, dtype=jnp.int32),
    )


# ------------------------------------------------------------------ policy


@final
class AtomisticTrainingPolicy(StrictModule, NonTrainableState):
    """Optimizer, adaptation, evaluation view, selection and stopping policy.

    ``maximum_steps`` counts committed (accepted) updates; one global loader
    batch is one accumulation window. The default optimizer is the native Adam
    at ``learning_rate``; a custom Optax transformation needs a stable
    ``optimizer_id``. ``clip_global_norm`` clips the merged window gradient.
    ``target`` is the kernel's EMA recurrence of the PARAMETER lane;
    ``evaluation`` selects the parameter view used for validation, selection
    and the returned evaluation model. The selection metric is the sum of every
    objective's normalized validation value times ``selection_weights`` (by
    objective name; default: the schedule's final weight). ``plateau`` scales
    updates after stale validations. ``staging`` is execution-only.
    """

    maximum_steps: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    optimizer: optax.GradientTransformation | None = eqx.field(static=True)
    optimizer_id: str = eqx.field(static=True)
    clip_global_norm: float | None = eqx.field(static=True)
    adaptation: AtomisticParameterSelection | AtomisticIrrepLowRankAdaptation | None
    target: ExponentialMovingAverageTargetPolicy | None = eqx.field(static=True)
    evaluation: AtomisticEvaluationView = eqx.field(static=True)
    plateau: AtomisticPlateauPolicy | None
    selection_weights: tuple[tuple[str, float], ...] = eqx.field(static=True)
    validation_every: int = eqx.field(static=True)
    patience: int | None = eqx.field(static=True)
    min_delta: float = eqx.field(static=True)
    select_best: bool = eqx.field(static=True)
    normalization_floor: float = eqx.field(static=True)
    staging: AtomisticLoaderStaging | None
    policy_id: str = eqx.field(static=True)
    continuation_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_steps: int,
        learning_rate: float = 1e-3,
        optimizer: optax.GradientTransformation | None = None,
        optimizer_id: str | None = None,
        clip_global_norm: float | None = None,
        adaptation: AtomisticParameterSelection
        | AtomisticIrrepLowRankAdaptation
        | None = None,
        target: ExponentialMovingAverageTargetPolicy | None = None,
        evaluation: AtomisticEvaluationView = "current",
        plateau: AtomisticPlateauPolicy | None = None,
        selection_weights: Mapping[str, float] | None = None,
        validation_every: int = 1,
        patience: int | None = None,
        min_delta: float = 0.0,
        select_best: bool = True,
        normalization_floor: float = 1e-12,
        staging: AtomisticLoaderStaging | None = None,
    ) -> None:
        steps = _positive_int(maximum_steps, "maximum_steps", minimum=0)
        rate = _finite(learning_rate, "learning_rate", positive=True)
        if optimizer is None:
            if optimizer_id is not None:
                raise ValueError("optimizer_id names a custom optimizer only.")
            resolved_id = f"phydrax.optim.adam:{rate!r}"
        else:
            if not isinstance(optimizer, optax.GradientTransformation):
                raise TypeError("optimizer must be an Optax GradientTransformation.")
            if not optimizer_id:
                raise ValueError("Custom optimizers require a stable optimizer_id.")
            resolved_id = _name(optimizer_id, "optimizer_id")
        clip = (
            None
            if clip_global_norm is None
            else _finite(clip_global_norm, "clip_global_norm", positive=True)
        )
        if adaptation is not None and not isinstance(
            adaptation, (AtomisticParameterSelection, AtomisticIrrepLowRankAdaptation)
        ):
            raise TypeError("adaptation must be an AtomisticAdaptation or None.")
        if target is not None and not isinstance(
            target, ExponentialMovingAverageTargetPolicy
        ):
            raise TypeError("target must be an ExponentialMovingAverageTargetPolicy.")
        view = parse(evaluation, AtomisticEvaluationView, "evaluation")
        if view == "target" and target is None:
            raise ValueError("The target evaluation view requires an EMA target policy.")
        if plateau is not None and not isinstance(plateau, AtomisticPlateauPolicy):
            raise TypeError("plateau must be an AtomisticPlateauPolicy or None.")
        weights = tuple(
            sorted(
                (_name(name, "selection weight name"), _finite(value, "selection weight"))
                for name, value in (
                    {} if selection_weights is None else selection_weights
                ).items()
            )
        )
        every = _positive_int(validation_every, "validation_every")
        patience_ = None if patience is None else _positive_int(patience, "patience")
        delta = _finite(min_delta, "min_delta")
        floor = _finite(normalization_floor, "normalization_floor", positive=True)
        if staging is not None and not isinstance(staging, AtomisticLoaderStaging):
            raise TypeError("staging must be an AtomisticLoaderStaging or None.")
        continuation = {
            "kind": "atomistic-training-continuation-policy",
            "optimizer": resolved_id,
            "clip_global_norm": clip,
            "adaptation": (
                None
                if adaptation is None
                else getattr(adaptation, "selection_id", None)
                or getattr(adaptation, "adaptation_id", None)
            ),
            "target": None if target is None else dataclasses.asdict(target),
            "evaluation": view,
            "plateau": None if plateau is None else plateau.plateau_id,
            "selection_weights": weights,
            "validation_every": every,
            "patience": patience_,
            "min_delta": delta,
            "select_best": bool(select_best),
            "normalization_floor": floor,
        }
        self.maximum_steps = steps
        self.learning_rate = rate
        self.optimizer = optimizer
        self.optimizer_id = resolved_id
        self.clip_global_norm = clip
        self.adaptation = adaptation
        self.target = target
        self.evaluation = view
        self.plateau = plateau
        self.selection_weights = weights
        self.validation_every = every
        self.patience = patience_
        self.min_delta = delta
        self.select_best = bool(select_best)
        self.normalization_floor = floor
        self.staging = staging
        self.continuation_id = canonical_fingerprint(continuation)
        self.policy_id = canonical_fingerprint({**continuation, "maximum_steps": steps})

    def transformation(self) -> optax.GradientTransformation:
        base = adam(self.learning_rate) if self.optimizer is None else self.optimizer
        if self.clip_global_norm is None:
            return base
        return optax.chain(optax.clip_by_global_norm(self.clip_global_norm), base)

    def selection_weight(self, objective: AbstractAtomisticObjective, /) -> float:
        for name, value in self.selection_weights:
            if name == objective.name:
                return value
        return objective.schedule.end


# ---------------------------------------------------------- kernel objectives


def _objective_weight(
    schedules: tuple[Any, ...],
    index: int,
    weighting: AtomisticObjectiveWeighting,
    step: Array,
    dtype: Any,
    /,
) -> Array:
    weight = schedules[index](step, dtype=dtype)
    if weighting == "literal":
        return weight
    total = sum(schedule(step, dtype=dtype) for schedule in schedules)
    return weight / total


@final
class _KernelObjective(StrictModule, ExplicitFreeze):
    """One problem objective as a shared-kernel objective callable."""

    objective: AbstractAtomisticObjective
    reduction: AbstractAtomisticObjectiveReduction
    schedules: tuple[Any, ...]
    index: int = eqx.field(static=True)
    weighting: AtomisticObjectiveWeighting = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    def __call__(
        self,
        parameters: Any,
        model_state: Any,
        fixed: Any,
        batch: AtomisticTrainingBatch,
        keys: TrainingKeys,
    ) -> tuple[_ObjectiveContribution, Any, AtomisticObjectiveTerm]:
        bound = self.reduction.bind_parameters(parameters)
        tree = combine_parameters(bound, model_state, fixed)
        term = self.objective.term(
            tree.potential(),
            tree.components[self.index],
            batch,
            scale=self.scale,
            dtype=self.dtype,
        )
        weight = _objective_weight(
            self.schedules, self.index, self.weighting, keys.accepted_cursor, self.dtype
        )
        term = self.reduction.reduce(
            dataclasses.replace(term, numerator=term.numerator * weight)
        )
        return _ObjectiveContribution(term.numerator, term.support), model_state, term


@final
class _ObjectiveEvaluator(StrictModule, ExplicitFreeze):
    """Unweighted reduced terms of every objective at one parameter view."""

    objectives: tuple[AbstractAtomisticObjective, ...]
    reduction: AbstractAtomisticObjectiveReduction
    scales: tuple[float, ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    def __call__(
        self, tree: AtomisticTrainedTree, batch: AtomisticTrainingBatch, /
    ) -> tuple[AtomisticObjectiveTerm, ...]:
        model = tree.potential()
        return tuple(
            self.reduction.reduce(
                objective.term(
                    model, tree.components[index], batch, scale=scale, dtype=self.dtype
                )
            )
            for index, (objective, scale) in enumerate(
                zip(self.objectives, self.scales, strict=True)
            )
        )


def _kernel(
    tree: AtomisticTrainedTree,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    normalization: AtomisticTrainingNormalization,
    reduction: AbstractAtomisticObjectiveReduction,
    /,
) -> PreparedTrainingKernel:
    schedules = tuple(objective.schedule for objective in problem.objectives)
    objectives = tuple(
        KernelObjective(
            objective_id=f"atomistic-{objective.name}",
            kind=objective.kind,
            route=objective.route,
            fn=_KernelObjective(
                objective,
                reduction,
                schedules,
                index=index,
                weighting=problem.weighting,
                scale=normalization.scales[index],
                dtype=problem.precision.loss_dtype,
            ),
        )
        for index, objective in enumerate(problem.objectives)
    )
    rule_id = canonical_fingerprint(
        {
            "kind": "atomistic-update-rule",
            "continuation": policy.continuation_id,
            "reduction": reduction.reduction_id,
        }
    )
    return prepare_training_kernel(
        tree,
        objectives,
        TrainingKernelSpec(
            _AtomisticUpdateRule(policy.transformation(), rule_id=rule_id),
            context="fit_atomistic_potential",
            rejection_budget=0,
            target_policy=policy.target,
            accumulation_dtype=problem.precision.accumulation_dtype,
        ),
        root_authority=ComponentAuthority.MODEL,
    )


@eqx.filter_jit
def _attempt(
    kernel: PreparedTrainingKernel,
    reduction: AbstractAtomisticObjectiveReduction,
    state: TrainingKernelState,
    batch: AtomisticTrainingBatch,
) -> Any:
    return reduction.map_step(kernel.attempt)(state, batch)


@eqx.filter_jit
def _accumulate(
    kernel: PreparedTrainingKernel,
    reduction: AbstractAtomisticObjectiveReduction,
    state: TrainingKernelState,
    batch: AtomisticTrainingBatch,
) -> Any:
    return reduction.map_step(kernel.accumulate_with_diagnostics)(state, batch)


@eqx.filter_jit
def _evaluate(
    evaluator: _ObjectiveEvaluator,
    tree: AtomisticTrainedTree,
    batch: AtomisticTrainingBatch,
) -> tuple[AtomisticObjectiveTerm, ...]:
    return evaluator.reduction.map_step(evaluator)(tree, batch)


# ------------------------------------------------------------------ evaluation


@final
class AtomisticObjectiveEvaluation(StrictModule, NonTrainableState):
    """Merged numerators/supports of every objective over one full loader epoch.

    ``values[k]`` is objective ``k``'s numerator sum over its support sum (zero
    for zero support); failures count valid cases over the whole pass.
    """

    names: tuple[str, ...] = eqx.field(static=True)
    numerators: tuple[float, ...] = eqx.field(static=True)
    supports: tuple[float, ...] = eqx.field(static=True)
    capacity_failures: int = eqx.field(static=True)
    scientific_failures: int = eqx.field(static=True)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(
            numerator / support if support > 0.0 else 0.0
            for numerator, support in zip(self.numerators, self.supports, strict=True)
        )

    @property
    def successful(self) -> bool:
        return (
            self.capacity_failures == 0
            and self.scientific_failures == 0
            and all(math.isfinite(value) for value in self.values)
        )


def evaluate_atomistic_objectives(
    tree: AtomisticTrainedTree,
    problem: AtomisticTrainingProblem,
    normalization: AtomisticTrainingNormalization,
    preparation: AtomisticBatchPreparation,
    /,
    *,
    reduction: AbstractAtomisticObjectiveReduction | None = None,
    staging: AtomisticLoaderStaging | None = None,
) -> AtomisticObjectiveEvaluation:
    """Evaluate every objective over epoch 0 of ``loader`` without differentiation."""
    reduction_ = AtomisticLocalObjectiveReduction() if reduction is None else reduction
    evaluator = _ObjectiveEvaluator(
        problem.objectives,
        reduction_,
        scales=normalization.scales,
        dtype=problem.precision.loss_dtype,
    )
    coordinate = tree.potential().precision.coordinate_dtype
    count = len(problem.objectives)
    numerators = np.zeros((count,), dtype=np.float64)
    supports = np.zeros((count,), dtype=np.float64)
    capacity = scientific = 0
    loader = preparation.loader
    with loader.epoch(loader.start(0), staging=staging) as epoch:
        for item in epoch:
            batch = preparation.prepare_step(
                item.payload, item.provenance, reduction_, coordinate_dtype=coordinate
            )
            terms = jax.device_get(_evaluate(evaluator, tree, batch))
            for index, term in enumerate(terms):
                numerators[index] += float(term.numerator)
                supports[index] += float(term.support)
                capacity += int(term.capacity_failures)
                scientific += int(term.scientific_failures)
    return AtomisticObjectiveEvaluation(
        names=tuple(objective.name for objective in problem.objectives),
        numerators=tuple(float(value) for value in numerators),
        supports=tuple(float(value) for value in supports),
        capacity_failures=capacity,
        scientific_failures=scientific,
    )


# ------------------------------------------------------------------ result


_UNSUPPORTED_EPOCH = "unsupported_training_epoch"


def _unsupported_epoch(
    loader: AtomisticStructureLoader, cursor: AtomisticLoaderCursor, unsupported: int, /
) -> bool:
    """Whether the consecutive zero-support windows ending at ``cursor`` span an epoch.

    Every loader global batch is one window and windows are contiguous in loader
    order, so ``unsupported`` consecutive windows closing at an epoch boundary
    cover the complete preceding epoch once they reach its batch count.
    """
    return (
        cursor.batch_index == 0
        and cursor.payload_index == 0
        and unsupported >= loader.batch_count
    )


@final
class AtomisticTrainingResult(StrictModule, ExplicitFreeze):
    """Complete continuation, final/evaluation/best models, histories and status.

    ``trained_tree`` is the committed kernel tree (adapted model and auxiliary
    components); ``potential`` its merged model; ``evaluation_potential`` the
    declared evaluation view (EMA target or current); ``best_potential`` the
    selected evaluation view. ``training_state`` is the committed kernel state
    and ``cursor`` the loader position after the last accumulated payload; on
    a failed update both remain at the last accepted boundary, so the failed
    window is replayed exactly by a continuation (for example after growing a
    capacity in the execution plan). ``training_values[k]`` is the weighted
    window objective before accepted update ``k + 1`` and
    ``training_objectives[k]`` its weighted per-objective values.
    ``skipped_windows`` counts every committed zero-support window and
    ``unsupported_windows`` the consecutive ones since the last accepted update
    (or the start of the run). Termination ``"unsupported_training_epoch"``
    (status ``STOPPED_EARLY``) means those consecutive windows cover one
    complete loader epoch: the training stream offered no supported window, so
    the run stopped at that epoch boundary with optimizer, schedules and EMA
    target untouched by the skips. Such a result is not ``successful``.
    """

    trained_tree: AtomisticTrainedTree
    best_tree: AtomisticTrainedTree
    training_state: TrainingKernelState
    normalization: AtomisticTrainingNormalization
    cursor: AtomisticLoaderCursor
    training_values: Array
    training_objectives: Array
    validation_values: Array
    validation_objectives: Array
    validation_steps: Array
    final_loss: Array
    best_loss: Array
    status: Array
    progress: TrainingProgress = eqx.field(static=True)
    termination: str = eqx.field(static=True)
    skipped_windows: int = eqx.field(static=True)
    unsupported_windows: int = eqx.field(static=True)
    capacity_failures: int = eqx.field(static=True)
    scientific_failures: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    continuation_id: str = eqx.field(static=True)
    reduction_id: str = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)
    training_checkpoint_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def potential(self) -> AbstractAtomisticModel:
        return self.trained_tree.potential()

    @property
    def best_potential(self) -> AbstractAtomisticModel:
        return self.best_tree.potential()

    @property
    def components(self) -> tuple[Any, ...]:
        return self.trained_tree.components

    @property
    def successful(self) -> Array:
        """Whether the run ended without failure and with a supported stream."""
        if self.termination == _UNSUPPORTED_EPOCH:
            return jnp.asarray(False)
        return (self.status == int(AtomisticStatus.SUCCESS)) | (
            self.status == int(AtomisticStatus.STOPPED_EARLY)
        )


@dataclass(slots=True)
class _Histories:
    training: list[float]
    objectives: list[list[float]]
    validation: list[float]
    validation_objectives: list[list[float]]
    validation_steps: list[int]

    @classmethod
    def resumed(cls, continuation: AtomisticTrainingResult, /) -> _Histories:
        return cls(
            np.asarray(continuation.training_values).tolist(),
            np.asarray(continuation.training_objectives).tolist(),
            np.asarray(continuation.validation_values).tolist(),
            np.asarray(continuation.validation_objectives).tolist(),
            np.asarray(continuation.validation_steps).tolist(),
        )


def _fingerprint(values: Sequence[float], /) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def _tree_digest(tree: Any, /) -> dict[str, Any]:
    """Content digest of every array leaf of a (possibly ``None``-holed) tree."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    return array_tree_fingerprint([np.asarray(leaf) for leaf in jax.device_get(leaves)])


def _fixed_digest(kernel: PreparedTrainingKernel, /) -> dict[str, Any]:
    return _tree_digest(kernel.fixed)


def _evaluation_tree(
    kernel: PreparedTrainingKernel,
    state: TrainingKernelState,
    policy: AtomisticTrainingPolicy,
    /,
) -> AtomisticTrainedTree:
    return (
        kernel.target_tree(state) if policy.evaluation == "target" else kernel.tree(state)
    )


def _assemble_result(
    kernel: PreparedTrainingKernel,
    state: TrainingKernelState,
    best_tree: AtomisticTrainedTree,
    normalization: AtomisticTrainingNormalization,
    cursor: AtomisticLoaderCursor,
    histories: _Histories,
    progress: TrainingProgress,
    /,
    *,
    problem: AtomisticTrainingProblem,
    policy_id: str,
    continuation_id: str,
    reduction_id: str,
    status: AtomisticStatus,
    termination: str,
    skipped: int,
    unsupported: int,
    capacity: int,
    scientific: int,
) -> AtomisticTrainingResult:
    tree = kernel.tree(state)
    model = tree.potential()
    best = best_tree.potential()
    count = len(problem.objectives)
    if histories.training:
        final = histories.training[-1]
    elif histories.validation:
        final = histories.validation[-1]
    else:
        final = float("nan")
    best_value = float("nan") if progress.best_value is None else progress.best_value
    dtype = problem.precision.accumulation_dtype
    capabilities_id = model.capabilities.capabilities_id
    result_id = canonical_fingerprint(
        {
            "kind": "atomistic-training-result",
            "problem": problem.problem_id,
            "policy": policy_id,
            "reduction": reduction_id,
            "capabilities": capabilities_id,
            "model": atomistic_model_revision(model).revision_id,
            "best_model": atomistic_model_revision(best).revision_id,
            "components": _tree_digest(tree.components),
            "best_components": _tree_digest(best_tree.components),
            "normalization": normalization.normalization_id,
            "training_checkpoint": kernel.checkpoint_id,
            "cursor": cursor.cursor_id,
            "updates": progress.update_step,
            "iteration_session": progress.iteration_session_id,
            "iteration_control": progress.iteration_control_id,
            "status": int(status),
            "termination": termination,
            "skipped": skipped,
            "unsupported": unsupported,
            "capacity_failures": capacity,
            "scientific_failures": scientific,
            "training": _fingerprint(histories.training),
            "objectives": [_fingerprint(row) for row in histories.objectives],
            "validation": _fingerprint(histories.validation),
            "validation_objectives": [
                _fingerprint(row) for row in histories.validation_objectives
            ],
            "validation_steps": list(histories.validation_steps),
        }
    )
    return AtomisticTrainingResult(
        trained_tree=tree,
        best_tree=best_tree,
        training_state=state,
        normalization=normalization,
        cursor=cursor,
        training_values=jnp.asarray(histories.training, dtype=dtype),
        training_objectives=jnp.asarray(histories.objectives, dtype=dtype).reshape(
            (len(histories.objectives), count)
        ),
        validation_values=jnp.asarray(histories.validation, dtype=dtype),
        validation_objectives=jnp.asarray(
            histories.validation_objectives, dtype=dtype
        ).reshape((len(histories.validation_objectives), count)),
        validation_steps=jnp.asarray(histories.validation_steps, dtype=jnp.int32),
        final_loss=jnp.asarray(final, dtype=dtype),
        best_loss=jnp.asarray(best_value, dtype=dtype),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        progress=progress,
        termination=termination,
        skipped_windows=skipped,
        unsupported_windows=unsupported,
        capacity_failures=capacity,
        scientific_failures=scientific,
        problem_id=problem.problem_id,
        execution_id=problem.execution_id,
        policy_id=policy_id,
        continuation_id=continuation_id,
        reduction_id=reduction_id,
        capabilities_id=capabilities_id,
        training_checkpoint_id=kernel.checkpoint_id,
        result_id=result_id,
    )


# ------------------------------------------------------------------ fit


def _selection_value(
    evaluation: AtomisticObjectiveEvaluation,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
) -> float:
    return float(
        sum(
            policy.selection_weight(objective) * value
            for objective, value in zip(
                problem.objectives, evaluation.values, strict=True
            )
        )
    )


def _resume(
    potential: AbstractAtomisticModel,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    normalization: AtomisticTrainingNormalization,
    reduction: AbstractAtomisticObjectiveReduction,
    continuation: AtomisticTrainingResult,
    /,
) -> tuple[PreparedTrainingKernel, TrainingKernelState]:
    if not isinstance(continuation, AtomisticTrainingResult):
        raise TypeError("continuation must be an AtomisticTrainingResult or None.")
    saved = continuation.potential
    if (
        type(potential) is not type(saved)
        or potential.architecture_id != saved.architecture_id
        or potential.capabilities.capabilities_id != continuation.capabilities_id
    ):
        raise ValueError(
            "Continuation potential must have the same concrete family and "
            "configuration as the supplied potential."
        )
    if continuation.problem_id != problem.problem_id:
        raise ValueError("Continuation result belongs to a different training problem.")
    if continuation.continuation_id != policy.continuation_id:
        raise ValueError(
            "Continuation policy changed optimizer, adaptation, loss or selection semantics."
        )
    if continuation.reduction_id != reduction.reduction_id:
        raise ValueError("Continuation changed the objective reduction.")
    if continuation.normalization.normalization_id != normalization.normalization_id:
        raise ValueError(
            "Continuation normalization no longer matches the training data."
        )
    if continuation.progress.update_step > policy.maximum_steps:
        raise ValueError("Continuation step exceeds the requested training ceiling.")
    problem.training.require_cursor(continuation.cursor)
    kernel = _kernel(continuation.trained_tree, problem, policy, normalization, reduction)
    if continuation.training_checkpoint_id != kernel.checkpoint_id:
        raise ValueError("Continuation training state belongs to a different kernel.")
    state = restore_training_checkpoint(
        kernel,
        build_training_checkpoint(
            kernel, continuation.training_state, allow_intermediate=True
        ),
    ).state
    return kernel, state


def _failure_status(capacity: int, /) -> AtomisticStatus:
    return AtomisticStatus.NEIGHBOR_OVERFLOW if capacity else AtomisticStatus.NONFINITE


@dataclass(slots=True)
class _FitSegment:
    """Host lifecycle of one ``fit_atomistic_potential`` segment.

    Owns the committed kernel state and loader cursor, the last committed
    boundary a failed window rolls back to, the open window's accumulation
    diagnostics, the histories, skip counters and terminal status. Each phase
    keeps the run's event, commit and termination order: ``open`` (initial
    validation and pre-update stops), ``epoch`` (payload dispatch into
    ``accumulate``/``accept``/``skip``/``fail``) and ``finish``.
    """

    problem: AtomisticTrainingProblem
    policy: AtomisticTrainingPolicy
    reduction: AbstractAtomisticObjectiveReduction
    normalization: AtomisticTrainingNormalization
    kernel: PreparedTrainingKernel
    coordinate: Any
    evaluation_preparation: AtomisticBatchPreparation
    control: TrainingController
    state: TrainingKernelState
    cursor: AtomisticLoaderCursor
    histories: _Histories
    skipped: int
    unsupported: int
    boundary_state: TrainingKernelState = dataclasses.field(init=False)
    boundary_cursor: AtomisticLoaderCursor = dataclasses.field(init=False)
    window: list[tuple[AtomisticObjectiveTerm, ...]] = dataclasses.field(
        default_factory=list
    )
    status: AtomisticStatus = AtomisticStatus.SUCCESS
    termination: str = "maximum_steps"
    capacity: int = 0
    scientific: int = 0

    def stop_early(self, termination: str, /) -> Literal[False]:
        self.status = AtomisticStatus.STOPPED_EARLY
        self.termination = termination
        return False

    def locate(self, microstep: int, /) -> None:
        self.control.progress = dataclasses.replace(
            self.control.progress,
            epoch=self.cursor.epoch,
            next_batch_index=self.cursor.batch_index,
            microstep=microstep,
        )

    def validate(self, step: int, /) -> bool:
        """Evaluate the declared view; drive selection and the plateau transition."""
        view = _evaluation_tree(self.kernel, self.state, self.policy)
        evaluation = evaluate_atomistic_objectives(
            view,
            self.problem,
            self.normalization,
            self.evaluation_preparation,
            reduction=self.reduction,
            staging=self.policy.staging,
        )
        value = _selection_value(evaluation, self.problem, self.policy)
        self.histories.validation.append(value)
        self.histories.validation_objectives.append(list(evaluation.values))
        self.histories.validation_steps.append(step)
        if not evaluation.successful or not math.isfinite(value):
            self.capacity, self.scientific = (
                evaluation.capacity_failures,
                evaluation.scientific_failures,
            )
            self.status = _failure_status(self.capacity)
            self.termination = "failed_validation_evaluation"
            return False
        policy = self.policy
        self.control.select(
            value, view, step=step, min_delta=policy.min_delta, patience=policy.patience
        )
        if policy.plateau is not None:
            self.state = dataclasses.replace(
                self.state,
                rule_state=_plateau_transition(
                    policy.plateau, self.state.rule_state, value
                ),
            )
        self.control.emit(
            TrainingIterationKind.VALIDATION,
            metrics={
                "loss": jnp.asarray(value),
                **{
                    f"{name}_loss": jnp.asarray(objective)
                    for name, objective in zip(
                        evaluation.names, evaluation.values, strict=True
                    )
                },
            },
        )
        return True

    def open(self, *, fresh: bool) -> bool:
        """Initial validation and pre-update stops; whether updates may run."""
        if fresh:
            self.validate(0)
        control, maximum = self.control, self.policy.maximum_steps
        if (
            self.status is AtomisticStatus.SUCCESS
            and control.progress.update_step < maximum
            and _unsupported_epoch(self.problem.training, self.cursor, self.unsupported)
        ):
            self.stop_early(_UNSUPPORTED_EPOCH)
        if self.status is AtomisticStatus.SUCCESS and control.stop_requested:
            self.stop_early("host_control_stop_before_first_update")
        self.boundary_state, self.boundary_cursor = self.state, self.cursor
        return (
            self.status is AtomisticStatus.SUCCESS
            and control.progress.update_step < maximum
            and not control.stop_requested
        )

    def epoch(self) -> bool:
        """Stream the rest of the cursor's epoch; whether the run continues."""
        cursor = self.cursor
        with self.problem.training.epoch(cursor, staging=self.policy.staging) as epoch:
            if cursor.batch_index == 0 and cursor.payload_index == 0:
                self.control.emit(
                    TrainingIterationKind.EPOCH_START,
                    metrics={"epoch": jnp.asarray(cursor.epoch)},
                )
                if self.control.stop_requested:
                    return self.stop_early("host_control_stop_at_epoch_start")
            for item in epoch:
                if not self.payload(item):
                    return False
        return True

    def payload(self, item: AtomisticLoaderItem, /) -> bool:
        """Accumulate one payload or attempt the update its window closes."""
        batch = self.problem.preparation.prepare_step(
            item.payload,
            item.provenance,
            self.reduction,
            coordinate_dtype=self.coordinate,
        )
        if not item.closes_batch:
            return self.accumulate(batch, item.cursor)
        candidate, evidence = _attempt(self.kernel, self.reduction, self.state, batch)
        outcome, supported = jax.device_get((evidence.outcome, evidence.supported))
        outcome = TrainingAttemptOutcome(int(outcome))
        if outcome is TrainingAttemptOutcome.ACCEPTED:
            return self.accept(candidate, evidence, item.cursor)
        if not bool(supported) and outcome is TrainingAttemptOutcome.REJECTED_FINITE:
            return self.skip(candidate, item.cursor)
        self.fail(evidence)
        return False

    def accumulate(
        self, batch: AtomisticTrainingBatch, cursor: AtomisticLoaderCursor, /
    ) -> bool:
        self.state, diagnostics = _accumulate(
            self.kernel, self.reduction, self.state, batch
        )
        self.window.append(diagnostics)
        self.cursor = cursor
        self.locate(len(self.window))
        self.control.deliver(
            TrainingIterationKind.MICROSTEP,
            metrics={
                "payload_index": jnp.asarray(cursor.payload_index),
                "microstep": jnp.asarray(len(self.window)),
            },
        )
        if self.control.stop_requested:
            return self.stop_early("host_control_stop_inside_accumulation_window")
        return True

    def accept(
        self,
        candidate: TrainingKernelState,
        evidence: Any,
        cursor: AtomisticLoaderCursor,
        /,
    ) -> bool:
        """Commit an accepted update, then validate on schedule and stop if due."""
        self.state, self.cursor = candidate, cursor
        self.unsupported = 0
        step = self.control.progress.update_step + 1
        values = [float(value) for value in jax.device_get(evidence.objective_values)]
        self.histories.training.append(float(jax.device_get(evidence.value)))
        self.histories.objectives.append(values)
        self.control.complete_update(step)
        self.locate(0)
        self.control.emit(TrainingIterationKind.UPDATE, metrics={"loss": evidence.value})
        self.window = []
        maximum = self.policy.maximum_steps
        if (
            step % self.policy.validation_every == 0 or step == maximum
        ) and not self.validate(step):
            return False
        self.boundary_state, self.boundary_cursor = self.state, self.cursor
        if self.control.stop_requested:
            return self.stop_early("selection_or_host_control_stop")
        return step < maximum

    def skip(
        self, candidate: TrainingKernelState, cursor: AtomisticLoaderCursor, /
    ) -> bool:
        """Commit a zero-support window; stop once skips cover a complete epoch."""
        self.state, self.cursor = candidate, cursor
        self.boundary_state, self.boundary_cursor = self.state, self.cursor
        self.skipped += 1
        self.unsupported += 1
        self.window = []
        self.locate(0)
        self.control.emit(TrainingIterationKind.SKIP)
        if _unsupported_epoch(self.problem.training, self.cursor, self.unsupported):
            return self.stop_early(_UNSUPPORTED_EPOCH)
        if self.control.stop_requested:
            return self.stop_early("host_control_stop_after_skipped_window")
        return True

    def fail(self, evidence: Any, /) -> None:
        """Count the failed window's failures and roll back to the last boundary."""
        for terms in jax.device_get([*self.window, evidence.diagnostics]):
            for term in terms:
                self.capacity += int(term.capacity_failures)
                self.scientific += int(term.scientific_failures)
        self.status = _failure_status(self.capacity)
        self.termination = (
            "capacity_failure_in_update_window"
            if self.capacity
            else "nonfinite_update_window"
        )
        self.control.emit(TrainingIterationKind.FAILURE)
        # A failed window never advances: restore the last committed
        # boundary (an accepted update, a skip, or the resumed state).
        self.state, self.cursor = self.boundary_state, self.boundary_cursor

    def finish(self) -> AtomisticTrainingResult:
        """Validate a never-validated run, select the best view and assemble."""
        if not self.histories.validation and self.status is AtomisticStatus.SUCCESS:
            self.validate(self.control.progress.update_step)
        view = _evaluation_tree(self.kernel, self.state, self.policy)
        best_tree = self.control.selected(view) if self.policy.select_best else view
        result = _assemble_result(
            self.kernel,
            self.state,
            best_tree,
            self.normalization,
            self.cursor,
            self.histories,
            self.control.progress,
            problem=self.problem,
            policy_id=self.policy.policy_id,
            continuation_id=self.policy.continuation_id,
            reduction_id=self.reduction.reduction_id,
            status=self.status,
            termination=self.termination,
            skipped=self.skipped,
            unsupported=self.unsupported,
            capacity=self.capacity,
            scientific=self.scientific,
        )
        self.control.emit(
            TrainingIterationKind.RUN_TERMINAL,
            metrics={"final_loss": result.final_loss, "best_loss": result.best_loss},
        )
        return result


def fit_atomistic_potential(
    potential: AbstractAtomisticModel,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
    *,
    key: PRNGKey = DOC_KEY0,
    session: IterationSession | None = None,
    continuation: AtomisticTrainingResult | None = None,
    components: Mapping[str, Any] | None = None,
    reduction: AbstractAtomisticObjectiveReduction | None = None,
) -> AtomisticTrainingResult:
    """Fit an atomistic model over streamed minibatches with the shared kernel.

    Every loader payload is host-prepared (structures, frozen graph topology,
    labels) and accumulated; the payload that closes a global batch attempts
    one update. An accepted update commits parameters, optimizer, EMA target and
    the loader cursor; a window with zero support is a committed skip that
    advances only the loader cursor. Once consecutive skips cover one complete
    loader epoch, the run stops at that epoch boundary with ``STOPPED_EARLY``
    and termination ``"unsupported_training_epoch"`` (not ``successful``); a
    continuation of such a result over the same problem stops again without
    reading data. A nonfinite or capacity-failing window rolls the kernel back
    and terminates with ``NONFINITE`` or ``NEIGHBOR_OVERFLOW``, leaving the
    result at the last accepted boundary. Validation (over the validation
    loader, or a training epoch without one) evaluates the declared view every
    ``validation_every`` accepted updates, drives best-model selection, early
    stopping and the plateau transition. Host stop requests are honoured at
    payload boundaries, including inside an accumulation window, whose open
    state the result keeps. A continuation resumes the committed kernel state,
    loader cursor and consecutive-skip count; it equals the uninterrupted run
    bitwise only under ordered streamed accumulation and a fixed
    device/runtime/compiler profile.
    """
    if not isinstance(problem, AtomisticTrainingProblem):
        raise TypeError("problem must be an AtomisticTrainingProblem.")
    if not isinstance(policy, AtomisticTrainingPolicy):
        raise TypeError("policy must be an AtomisticTrainingPolicy.")
    reduction_ = AtomisticLocalObjectiveReduction() if reduction is None else reduction
    if not isinstance(reduction_, AbstractAtomisticObjectiveReduction):
        raise TypeError(
            "reduction must be an AbstractAtomisticObjectiveReduction or None."
        )
    problem.admit(potential)
    normalization = fit_atomistic_normalization(problem, floor=policy.normalization_floor)
    if continuation is None:
        tree = _trained_tree(potential, problem, policy.adaptation, components)
        kernel = _kernel(tree, problem, policy, normalization, reduction_)
        state = kernel.init(tree, key)
        progress = TrainingProgress()
        histories = _Histories([], [], [], [], [])
        cursor = problem.training.start(0)
        skipped = unsupported = 0
    else:
        if components is not None:
            raise ValueError("A continuation keeps its own trained components.")
        kernel, state = _resume(
            potential, problem, policy, normalization, reduction_, continuation
        )
        # Host iteration sessions are segment-local: a continuation binds the
        # caller's (fresh or absent) session; selection state is retained.
        progress = dataclasses.replace(
            continuation.progress,
            iteration_session_id=None,
            iteration_control_id=None,
            iteration_session_cursor=0,
            iteration_stop_requested=False,
        )
        histories = _Histories.resumed(continuation)
        cursor = continuation.cursor
        skipped = continuation.skipped_windows
        unsupported = continuation.unsupported_windows
    coordinate = kernel.tree(state).potential().precision.coordinate_dtype
    evaluation_preparation = (
        problem.preparation
        if problem.validation_preparation is None
        else problem.validation_preparation
    )
    control = TrainingController(
        total_steps=policy.maximum_steps,
        algorithm_id="atomistic-training",
        progress=progress,
        session=session,
    )
    if continuation is not None:
        control.best_payload = continuation.best_tree
    control.emit(TrainingIterationKind.RUN_START)
    segment = _FitSegment(
        problem=problem,
        policy=policy,
        reduction=reduction_,
        normalization=normalization,
        kernel=kernel,
        coordinate=coordinate,
        evaluation_preparation=evaluation_preparation,
        control=control,
        state=state,
        cursor=cursor,
        histories=histories,
        skipped=skipped,
        unsupported=unsupported,
    )
    running = segment.open(fresh=continuation is None)
    while running:
        running = segment.epoch()
    return segment.finish()


# ------------------------------------------------------------------ checkpoint


_CHECKPOINT_FORMAT = "atomistic-training-checkpoint"
_CHECKPOINT_FIELDS = frozenset(
    {
        "problem_id",
        "execution_id",
        "policy_id",
        "continuation_id",
        "reduction_id",
        "capabilities_id",
        "training_checkpoint_id",
        "normalization_id",
        "fixed_digest",
        "cursor",
        "status",
        "termination",
        "skipped_windows",
        "unsupported_windows",
        "capacity_failures",
        "scientific_failures",
        "update_count",
        "validation_count",
        "objective_count",
        "result_id",
    }
)


def write_atomistic_training_checkpoint(
    path: str | Path,
    result: AtomisticTrainingResult,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
    *,
    reduction: AbstractAtomisticObjectiveReduction | None = None,
) -> Path:
    """Atomically write a pickle-free fresh-process continuation boundary.

    The directory holds the committed kernel payload (parameters, optimizer and
    plateau state, EMA target, open accumulation window, root key, cursors and
    selection progress), the selected evaluation tree, the loader cursor and
    the histories. Normalization is refitted from the training source on
    restore and must reproduce its identity; the FIXED lane (frozen base model,
    complement) is bound by content digest.
    """
    if not isinstance(result, AtomisticTrainingResult):
        raise TypeError("result must be an AtomisticTrainingResult.")
    reduction_ = AtomisticLocalObjectiveReduction() if reduction is None else reduction
    if result.continuation_id != policy.continuation_id:
        raise ValueError("policy does not match the result's continuation semantics.")
    if result.problem_id != problem.problem_id:
        raise ValueError("problem does not match the result's training problem.")
    kernel = _kernel(
        result.trained_tree, problem, policy, result.normalization, reduction_
    )
    if kernel.checkpoint_id != result.training_checkpoint_id:
        raise ValueError("Result training state belongs to a different training kernel.")
    destination = Path(path)
    save_training_checkpoint(
        destination,
        build_training_checkpoint(
            kernel,
            result.training_state,
            selection=result.progress,
            allow_intermediate=True,
        ),
        (
            result.best_tree,
            result.training_values,
            result.training_objectives,
            result.validation_values,
            result.validation_objectives,
            result.validation_steps,
        ),
        format=_CHECKPOINT_FORMAT,
        metadata={
            "problem_id": result.problem_id,
            "execution_id": result.execution_id,
            "policy_id": result.policy_id,
            "continuation_id": result.continuation_id,
            "reduction_id": result.reduction_id,
            "capabilities_id": result.capabilities_id,
            "training_checkpoint_id": result.training_checkpoint_id,
            "normalization_id": result.normalization.normalization_id,
            "fixed_digest": _fixed_digest(kernel),
            "cursor": result.cursor.to_payload(),
            "status": int(np.asarray(result.status)),
            "termination": result.termination,
            "skipped_windows": result.skipped_windows,
            "unsupported_windows": result.unsupported_windows,
            "capacity_failures": result.capacity_failures,
            "scientific_failures": result.scientific_failures,
            "update_count": int(result.training_values.shape[0]),
            "validation_count": int(result.validation_values.shape[0]),
            "objective_count": len(problem.objectives),
            "result_id": result.result_id,
        },
    )
    return destination


def _count(metadata: Mapping[str, Any], name: str, /) -> int:
    value = metadata[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Atomistic training checkpoint field {name!r} is invalid.")
    return value


def read_atomistic_training_checkpoint(
    path: str | Path,
    potential: AbstractAtomisticModel,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
    *,
    components: Mapping[str, Any] | None = None,
    reduction: AbstractAtomisticObjectiveReduction | None = None,
) -> AtomisticTrainingResult:
    """Restore a continuation result against fresh templates in a new process.

    ``potential`` (and ``components``) are structural templates; under a
    parameter selection or low-rank adaptation ``potential`` must be the exact
    adapted base, whose FIXED content digest is verified. The problem, policy
    continuation semantics, refitted normalization, reduction, capabilities,
    kernel, loader cursor and the restored content identity must all match;
    serialized identities are verified, never trusted. ``policy`` may raise
    ``maximum_steps`` for a continued run.
    """
    reduction_ = AtomisticLocalObjectiveReduction() if reduction is None else reduction
    problem.admit(potential)
    metadata = read_training_checkpoint_metadata(path, format=_CHECKPOINT_FORMAT)
    if set(metadata) != _CHECKPOINT_FIELDS:
        raise ValueError("Atomistic training checkpoint manifest is not canonical.")
    normalization = fit_atomistic_normalization(problem, floor=policy.normalization_floor)
    tree = _trained_tree(potential, problem, policy.adaptation, components)
    kernel = _kernel(tree, problem, policy, normalization, reduction_)
    expected = {
        "problem_id": problem.problem_id,
        "continuation_id": policy.continuation_id,
        "reduction_id": reduction_.reduction_id,
        "capabilities_id": potential.capabilities.capabilities_id,
        "training_checkpoint_id": kernel.checkpoint_id,
        "normalization_id": normalization.normalization_id,
        "fixed_digest": _fixed_digest(kernel),
    }
    mismatched = [name for name, value in expected.items() if metadata[name] != value]
    if mismatched:
        raise ValueError(
            f"Atomistic training checkpoint identity mismatch: {mismatched!r}."
        )
    for name in ("policy_id", "termination", "execution_id"):
        if not isinstance(metadata[name], str):
            raise ValueError("Atomistic training checkpoint identities are invalid.")
    cursor = AtomisticLoaderCursor.from_payload(metadata["cursor"])
    problem.training.require_cursor(cursor)
    status = AtomisticStatus(_count(metadata, "status"))
    updates = _count(metadata, "update_count")
    validations = _count(metadata, "validation_count")
    objectives = _count(metadata, "objective_count")
    if objectives != len(problem.objectives):
        raise ValueError("Atomistic training checkpoint objective count differs.")
    dtype = problem.precision.accumulation_dtype
    loaded = load_training_checkpoint(
        path,
        kernel,
        kernel.init(tree, DOC_KEY0),
        (
            tree,
            jnp.zeros((updates,), dtype=dtype),
            jnp.zeros((updates, objectives), dtype=dtype),
            jnp.zeros((validations,), dtype=dtype),
            jnp.zeros((validations, objectives), dtype=dtype),
            jnp.zeros((validations,), dtype=jnp.int32),
        ),
        format=_CHECKPOINT_FORMAT,
    )
    progress = loaded.restored.selection
    if progress is None:
        raise ValueError("Atomistic training checkpoint lacks selection progress.")
    (
        best_tree,
        training_values,
        training_objectives,
        validation_values,
        validation_objectives,
        validation_steps,
    ) = loaded.extra
    result = _assemble_result(
        kernel,
        loaded.restored.state,
        best_tree,
        normalization,
        cursor,
        _Histories(
            np.asarray(training_values).tolist(),
            np.asarray(training_objectives).tolist(),
            np.asarray(validation_values).tolist(),
            np.asarray(validation_objectives).tolist(),
            np.asarray(validation_steps).tolist(),
        ),
        progress,
        problem=problem,
        policy_id=metadata["policy_id"],
        continuation_id=policy.continuation_id,
        reduction_id=reduction_.reduction_id,
        status=status,
        termination=metadata["termination"],
        skipped=_count(metadata, "skipped_windows"),
        unsupported=_count(metadata, "unsupported_windows"),
        capacity=_count(metadata, "capacity_failures"),
        scientific=_count(metadata, "scientific_failures"),
    )
    if result.result_id != metadata["result_id"]:
        raise ValueError("Atomistic training checkpoint content identity is corrupt.")
    return result


# ---------------------------------------------------- canonical training sources


def atomistic_training_schema(
    scale: AtomisticScaleContract,
    /,
    *,
    mass_unit: UnitDefinition,
    species_kind: AtomisticSpeciesKind = AtomisticSpeciesKind.ATOMIC_NUMBER,
    periodic: bool = False,
    energy: bool = True,
    forces: bool = True,
    stress: bool = False,
    stress_convention: AtomisticStressConvention = AtomisticStressConvention.CAUCHY_TENSION_POSITIVE,
) -> AtomisticStructureStoreSchema:
    """Canonical structure/atom schema of energy, force and stress training data.

    Geometry is ``positions`` (Cartesian, length unit), ``species``
    (``species_kind`` codes) and ``masses``; periodic data add a row ``cell``
    and a ``periodic`` axis mask. Labels are optional per row: total
    ``energy``, conservative ``forces`` and tensile-positive cell-volume
    ``stress`` in ``stress_convention``.
    """
    if not isinstance(mass_unit, UnitDefinition) or mass_unit.dimension != MASS:
        raise ValueError("mass_unit must be a MASS-dimension UnitDefinition.")
    if stress and not periodic:
        raise ValueError("Stress labels require periodic cell fields.")
    kind = parse(species_kind, AtomisticSpeciesKind, "species_kind")
    convention = parse(stress_convention, AtomisticStressConvention, "stress_convention")
    length = scale.length_unit
    levels = (
        AtomisticStoreLevel("structure", "structure"),
        AtomisticStoreLevel("atom", "atom", parent="structure"),
    )
    fields: list[AtomisticStoreField] = [
        AtomisticStoreField(
            "positions",
            "atom",
            "float64",
            kind="continuous",
            role="geometry",
            convention="cartesian",
            axes=(ATOMISTIC_CARTESIAN_AXIS,),
            unit=length,
            extensivity="intensive",
        ),
        AtomisticStoreField(
            "species",
            "atom",
            "int32",
            kind="categorical",
            role="identity",
            convention=kind.value,
        ),
        AtomisticStoreField(
            "masses",
            "atom",
            "float64",
            kind="continuous",
            role="identity",
            convention="particle-mass",
            unit=mass_unit,
            extensivity="extensive",
        ),
    ]
    if periodic:
        fields.extend(
            (
                AtomisticStoreField(
                    "cell",
                    "structure",
                    "float64",
                    kind="continuous",
                    role="geometry",
                    convention="row-lattice-vectors",
                    axes=(ATOMISTIC_CELL_VECTOR_AXIS, ATOMISTIC_CARTESIAN_AXIS),
                    unit=length,
                    extensivity="intensive",
                ),
                AtomisticStoreField(
                    "periodic",
                    "structure",
                    "bool",
                    kind="mask",
                    role="geometry",
                    convention="periodic-axes",
                    axes=(ATOMISTIC_CELL_VECTOR_AXIS,),
                ),
            )
        )
    if energy:
        fields.append(
            AtomisticStoreField(
                "energy",
                "structure",
                "float64",
                kind="continuous",
                role="label",
                convention="total-energy",
                unit=scale.energy_unit,
                extensivity="extensive",
                availability="optional",
            )
        )
    if forces:
        fields.append(
            AtomisticStoreField(
                "forces",
                "atom",
                "float64",
                kind="continuous",
                role="label",
                convention="conservative-force",
                axes=(ATOMISTIC_CARTESIAN_AXIS,),
                unit=scale.force_unit,
                extensivity="intensive",
                availability="optional",
            )
        )
    if stress:
        fields.append(
            AtomisticStoreField(
                "stress",
                "structure",
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
    return AtomisticStructureStoreSchema(levels, tuple(fields))


def _optional_labels(
    values: ConvertibleToArray | None,
    available: ConvertibleToArray | None,
    shape: tuple[int, ...],
    default: np.ndarray,
    label: str,
    /,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if values is None:
        if available is not None:
            raise ValueError(f"{label} availability requires {label} labels.")
        return None, None
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{label} labels must have shape {shape}.")
    mask = default if available is None else np.asarray(available, dtype=np.bool_)
    if mask.shape != default.shape:
        raise ValueError(f"{label} availability must have shape {default.shape}.")
    return array, mask


def atomistic_training_source(
    batch: AtomisticBatch,
    /,
    *,
    mass_unit: UnitDefinition,
    energy: ConvertibleToArray | None = None,
    forces: ConvertibleToArray | None = None,
    stress: ConvertibleToArray | None = None,
    energy_available: ConvertibleToArray | None = None,
    force_available: ConvertibleToArray | None = None,
    stress_available: ConvertibleToArray | None = None,
    record_ids: Sequence[str] | None = None,
    lineage_ids: Sequence[str] | None = None,
    stress_convention: AtomisticStressConvention = AtomisticStressConvention.CAUCHY_TENSION_POSITIVE,
) -> AtomisticStructureSource:
    """Ingest a fixed-capacity batch and its labels as one immutable source.

    Every case becomes one record holding its active atoms in slot order, their
    stable particle IDs, the batch species codes (atomic numbers, or atom-type
    IDs for non-element particles) and masses in ``mass_unit``. ``energy`` is
    ``(case,)``, ``forces`` ``(case, atom, 3)`` and ``stress`` ``(case, 3, 3)``;
    availability masks are per case, per atom and per case. A case without an
    available label stores the canonical zero. ``lineage_ids`` default to the
    record IDs (each case its own independent unit).
    """
    if not isinstance(batch, AtomisticBatch):
        raise TypeError("batch must be an AtomisticBatch.")
    cases, atoms = batch.case_count, batch.atom_capacity
    mask = np.asarray(batch.atom_mask, dtype=np.bool_)
    elements = np.asarray(batch.element_mask, dtype=np.bool_)
    if np.any(elements != mask) and np.any(elements):
        raise ValueError(
            "A training source holds either element or non-element particles."
        )
    kind = (
        AtomisticSpeciesKind.ATOMIC_NUMBER
        if np.all(elements == mask)
        else AtomisticSpeciesKind.ATOM_TYPE_ID
    )
    codes = np.asarray(
        batch.atomic_numbers
        if kind is AtomisticSpeciesKind.ATOMIC_NUMBER
        else batch.atom_type_ids
    )
    periodic = batch.cells is not None and batch.periodic_axes is not None
    energies, energy_mask = _optional_labels(
        energy, energy_available, (cases,), np.ones((cases,), np.bool_), "energy"
    )
    force_values, force_mask = _optional_labels(
        forces, force_available, (cases, atoms, 3), mask.copy(), "force"
    )
    stress_values, stress_mask = _optional_labels(
        stress, stress_available, (cases, 3, 3), np.ones((cases,), np.bool_), "stress"
    )
    if stress_values is not None and not periodic:
        raise ValueError("Stress labels require batch cells and periodic axes.")
    schema = atomistic_training_schema(
        batch.scale,
        mass_unit=mass_unit,
        species_kind=kind,
        periodic=periodic,
        energy=energies is not None,
        forces=force_values is not None,
        stress=stress_values is not None,
        stress_convention=stress_convention,
    )
    ids = tuple(batch.structure_ids) if record_ids is None else tuple(record_ids)
    lineages = ids if lineage_ids is None else tuple(lineage_ids)
    if len(ids) != cases or len(lineages) != cases:
        raise ValueError("record_ids and lineage_ids name every case.")
    positions = np.asarray(batch.positions, dtype=np.float64)
    masses = np.asarray(batch.masses, dtype=np.float64)
    particles = np.asarray(batch.particle_ids, dtype=np.int64)
    cells = None if not periodic else np.asarray(batch.cells, dtype=np.float64)
    axes = None if not periodic else np.asarray(batch.periodic_axes, dtype=np.bool_)
    records = []
    for case in range(cases):
        active = mask[case]
        count = int(np.count_nonzero(active))
        values: dict[str, Any] = {
            "positions": positions[case][active],
            "species": codes[case][active].astype(np.int32),
            "masses": masses[case][active],
        }
        available: dict[str, Any] = {}
        if periodic and cells is not None and axes is not None:
            values["cell"] = cells[case][None]
            values["periodic"] = axes[case][None]
        if energies is not None and energy_mask is not None:
            values["energy"] = np.where(energy_mask[case], energies[case], 0.0)[None]
            available["energy"] = energy_mask[case][None]
        if force_values is not None and force_mask is not None:
            selected = force_mask[case][active]
            values["forces"] = np.where(
                selected[:, None], force_values[case][active], 0.0
            )
            available["forces"] = selected
        if stress_values is not None and stress_mask is not None:
            values["stress"] = np.where(stress_mask[case], stress_values[case], 0.0)[None]
            available["stress"] = stress_mask[case][None]
        records.append(
            AtomisticStructureRecord(
                schema,
                record_id=ids[case],
                lineage_id=lineages[case],
                counts={"atom": [count]},
                stable_ids=particles[case][active],
                values=values,
                available=available,
            )
        )
    return AtomisticStructureSource(schema, tuple(records))


def atomistic_training_loader(
    *sources: AbstractAtomisticStructureSource,
    batch_size: int | None = None,
    stream: str = "training",
    seed: int = 0,
    shuffle: bool | None = None,
    weights: Sequence[int] | None = None,
    max_bytes: int = 1 << 34,
) -> AtomisticStructureLoader:
    """Epoch loader over canonical training sources with one bucket per extent.

    Binds every canonical field present in the (shared) schema to the slot of
    the same name. ``batch_size=None`` is the full-batch policy: one global
    batch of every draw. The single bucket admits the largest atom count of
    every source; ``max_bytes`` bounds one payload.
    """
    if not sources:
        raise ValueError("At least one training source is required.")
    schema = sources[0].schema
    if any(source.schema.schema_id != schema.schema_id for source in sources):
        raise ValueError("Training sources must share one canonical schema.")
    counts = [int(source.record_count) for source in sources]
    repeats = [1] * len(sources) if weights is None else [int(value) for value in weights]
    draws = sum(count * repeat for count, repeat in zip(counts, repeats, strict=True))
    size = draws if batch_size is None else _positive_int(batch_size, "batch_size")
    atoms = max(int(np.max(source.level_counts(schema.atom_level))) for source in sources)
    bindings = []
    for field in schema.fields:
        bindings.append(
            AtomisticFieldBinding(
                field.name,
                field.name,
                level_kind=schema.level(field.level).kind,
                value_kind=field.kind,
                role=field.role,
                convention=field.convention,
                axes=field.axes,
                unit=field.unit,
            )
        )
    mixture = AtomisticSourceMixture(
        tuple(
            AtomisticMixtureComponent(source, weight=repeat)
            for source, repeat in zip(sources, repeats, strict=True)
        ),
        draw="epoch",
        objective="sampled",
        seed=seed,
        shuffle=shuffle,
    )
    bucket = AtomisticWorkloadBucket(
        "atoms",
        cases=min(size, draws),
        levels=(
            AtomisticLevelCapacity(schema.atom_level, layout="per-case", rows=atoms),
        ),
        max_bytes=max_bytes,
    )
    return AtomisticStructureLoader(
        mixture,
        tuple(bindings),
        AtomisticWorkloadBuckets((bucket,)),
        batch_size=size,
        stream=stream,
    )


__all__ = [
    "ATOMISTIC_CARTESIAN_AXIS",
    "ATOMISTIC_CELL_VECTOR_AXIS",
    "AtomisticAdaptation",
    "AtomisticBatchPreparation",
    "AtomisticEvaluationView",
    "AtomisticIrrepLowRankAdaptation",
    "AtomisticIrrepLowRankTree",
    "AtomisticObjectiveEvaluation",
    "AtomisticObjectiveWeighting",
    "AtomisticOptimizerState",
    "AtomisticParameterSelection",
    "AtomisticPlateauPolicy",
    "AtomisticTrainedTree",
    "AtomisticTrainingFields",
    "AtomisticTrainingNormalization",
    "AtomisticTrainingPolicy",
    "AtomisticTrainingProblem",
    "AtomisticTrainingResult",
    "atomistic_irrep_linear_sites",
    "atomistic_training_loader",
    "atomistic_training_schema",
    "atomistic_training_source",
    "evaluate_atomistic_objectives",
    "fit_atomistic_normalization",
    "fit_atomistic_potential",
    "read_atomistic_training_checkpoint",
    "write_atomistic_training_checkpoint",
]
