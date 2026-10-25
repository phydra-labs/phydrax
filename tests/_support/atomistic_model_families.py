"""Native model-family scenarios shared by integration tests, benchmark and smoke.

One owner for the invariants that the family workflows, the family benchmark
and the direct smoke all depend on:

* the production construction of each native family: `MACEPotential`,
  `MACELESPotential.initialize`, AIMNet2 and UMA rebuilt by the production
  reconstruction APIs from the admitted random source-layout fixtures of
  ``tests/_support/aimnet2_source.py`` and ``tests/_support/uma_source.py``;
* each family's own condition semantics (AIMNet2 continuous total charge and
  NSE spin multiplicity; UMA categorical total charge, null-or-multiplicity spin
  index and task code; MACE/MACELES unconditioned), never one shared mapping;
* water geometries and graph execution resources;
* the canonical structure-store columns that carry those conditions through
  the training loader: `atomistic_training_source` ingests geometry and labels
  only, so condition fields extend its schema with explicit ``condition``
  roles, kinds and units.

Every model here has random or random source-layout parameters: nothing in this
module supports a trained-accuracy, pretrained-weight or source-provider
equivalence claim.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.random as jr
import numpy as np
import numpy.typing as npt

import phydrax as phx
from phydrax.atomistic import (
    atomistic_energy_derivatives,
    atomistic_training_source,
    AtomisticBatch,
    AtomisticConditioning,
    AtomisticConditioningSpec,
    AtomisticGraphExecutionPlan,
    AtomisticGraphLayout,
    AtomisticStoreDType,
    AtomisticStoreExtensivity,
    AtomisticStoreField,
    AtomisticStoreValueKind,
    AtomisticStructureRecord,
    AtomisticStructureSource,
    AtomisticStructureStoreSchema,
    AtomisticTrainingFields,
    AtomisticUnitSystem,
    energy_and_forces,
)
from phydrax.atomistic._latent_electrostatics import (
    LatentEwaldDefinition,
    LatentEwaldResources,
)
from phydrax.discretization import ParticleImageCapacity
from phydrax.nn.atomistic import (
    aimnet2_potential_from_source,
    externalize_source_long_range,
    MACEArchitecture,
    MACELESPotential,
    MACEPotential,
    parse_uma_source_config,
    uma_initial_source_tensors,
    uma_model_from_source,
    UMAMoLEBinding,
)
from phydrax.sparse import StreamedRelationPlan
from phydrax.units import DALTON, ELEMENTARY_CHARGE, UnitDefinition
from tests._support.aimnet2_source import RC_S, source_inventory
from tests._support.uma_source import uma_config


UNITS = AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
SCALE = UNITS.scale
WATER = np.asarray([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2400, 0.9266, 0.0]])
WATER_NUMBERS = (8, 1, 1)
WATER_MASSES = (15.999, 1.008, 1.008)
# Liquid-water number density: one molecule per 29.9 cubic angstrom.
WATER_VOLUME = 29.9
SHEAR = np.asarray([[1.0, 0.0, 0.0], [0.08, 1.0, 0.0], [0.05, -0.06, 1.0]])
MACE_CUTOFF = 3.0
FAMILIES = ("mace", "maceles", "aimnet2", "aimnet2-nse", "uma")
"""Native families with a conservative scalar potential in these scenarios."""

# (total charge, spin multiplicity) of a closed-shell water cluster and its
# doublet ions: the multiplicity follows the electron-count parity.
NEUTRAL = (0, 1)
CHARGE_STATES = ((0, 1), (1, 2), (-1, 2))


# ------------------------------------------------------------------ geometry


def _rotation(rng: np.random.Generator, /) -> np.ndarray:
    matrix, upper = np.linalg.qr(rng.normal(size=(3, 3)))
    matrix = matrix * np.sign(np.diag(upper))
    return matrix * np.sign(np.linalg.det(matrix))


def water_molecules(
    count: int, /, *, seed: int, spacing: float = 2.9, jitter: float = 0.03
) -> np.ndarray:
    """``(3 count, 3)`` positions of rotated waters on a cubic grid."""
    if count < 1:
        raise ValueError("count must be positive.")
    rng = np.random.default_rng(seed)
    side = math.ceil(count ** (1.0 / 3.0) - 1e-12)
    grid = np.stack(np.meshgrid(*(np.arange(side),) * 3, indexing="ij"), axis=-1)
    centers = spacing * grid.reshape((-1, 3))[:count].astype(np.float64)
    molecules = [
        WATER @ _rotation(rng).T + center + jitter * rng.normal(size=(3, 3))
        for center in centers
    ]
    return np.concatenate(molecules, axis=0)


def water_box(
    molecules: int, /, *, seed: int, jitter: float = 0.03
) -> tuple[np.ndarray, np.ndarray]:
    """Sheared periodic cell at liquid-water density and its ``(3 m, 3)`` positions."""
    if molecules < 1:
        raise ValueError("molecules must be positive.")
    rng = np.random.default_rng(seed)
    cell = (molecules * WATER_VOLUME) ** (1.0 / 3.0) * SHEAR
    side = math.ceil(molecules ** (1.0 / 3.0) - 1e-12)
    grid = np.stack(np.meshgrid(*(np.arange(side),) * 3, indexing="ij"), axis=-1)
    fractional = (grid.reshape((-1, 3))[:molecules] + 0.5) / side
    centers = fractional @ cell
    positions = [
        WATER @ _rotation(rng).T + center + jitter * rng.normal(size=(3, 3))
        for center in centers
    ]
    return cell, np.concatenate(positions, axis=0)


def water_batch(
    positions: np.ndarray, /, *, cells: np.ndarray | None = None, prefix: str = "water"
) -> AtomisticBatch:
    """Batch of ``(case, 3 m, 3)`` water clusters, periodic when ``cells`` is given."""
    coordinates = np.asarray(positions, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1] % 3 or coordinates.shape[2] != 3:
        raise ValueError("positions must have shape (case, 3 * molecules, 3).")
    cases, atoms = coordinates.shape[:2]
    molecules = atoms // 3
    numbers = np.tile(np.asarray(WATER_NUMBERS * molecules, dtype=np.int32), (cases, 1))
    masses = np.tile(np.asarray(WATER_MASSES * molecules, dtype=np.float64), (cases, 1))
    periodic = None
    if cells is not None:
        cells = np.asarray(cells, dtype=np.float64)
        if cells.shape != (cases, 3, 3):
            raise ValueError("cells must have shape (case, 3, 3).")
        periodic = np.ones((cases, 3), dtype=np.bool_)
    return AtomisticBatch(
        numbers,
        coordinates,
        masses,
        SCALE,
        cells=cells,
        periodic_axes=periodic,
        structure_ids=tuple(f"{prefix}-{index}" for index in range(cases)),
    )


def composition(batch: AtomisticBatch, case: int = 0, /) -> dict[int, int]:
    """Element counts of one case (the UMA merged-MoLE binding composition)."""
    mask = np.asarray(batch.atom_mask, dtype=np.bool_)[case]
    numbers, counts = np.unique(
        np.asarray(batch.atomic_numbers)[case][mask], return_counts=True
    )
    return {
        int(number): int(count) for number, count in zip(numbers, counts, strict=True)
    }


def finite_execution(
    atoms: int, /, *, neighbors: int | None = None
) -> AtomisticGraphExecutionPlan:
    """Dense finite graph execution bounded by the per-case atom capacity."""
    return AtomisticGraphExecutionPlan(
        max(atoms - 1, 1) if neighbors is None else neighbors, maximum_dense_atoms=atoms
    )


def periodic_execution(
    *,
    degree: int = 128,
    edges: int = 4096,
    particles_per_cell: int = 16,
    search_cell: np.ndarray | None = None,
    search_margin: float = 0.0,
    streamed: StreamedRelationPlan | None = None,
) -> AtomisticGraphExecutionPlan:
    """Particle image-graph execution with explicit route resources.

    ``search_cell``/``search_margin`` declare the reusable layout's route-search
    envelope; training layouts over strained cells need it.
    """
    return AtomisticGraphExecutionPlan(
        degree,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=particles_per_cell,
            maximum_edges=edges,
            maximum_degree=degree,
            maximum_images=343,
        ),
        streamed=streamed,
        search_cell=search_cell,
        search_margin=search_margin,
    )


# --------------------------------------------------------------- the families


def mace_model(
    seed: int = 3,
    /,
    *,
    channels: int = 3,
    hidden_degree: int = 1,
    cutoff: float = MACE_CUTOFF,
    streaming: StreamedRelationPlan | None = None,
) -> MACEPotential:
    """Scale-shift MACE (the local component MACELES admits) on H/O."""
    return MACEPotential(
        SCALE,
        MACEArchitecture(
            species=(1, 8),
            cutoff=cutoff,
            radial_basis_count=4,
            cutoff_power=5,
            channel_count=channels,
            hidden_degree=hidden_degree,
            edge_degree=2,
            interactions=("real-agnostic-density", "real-agnostic-residual"),
            correlations=(2, 2),
            radial_widths=(6,),
            readout_width=4,
            average_neighbor_count=6.0,
            energy_scaling="scale-shift",
        ),
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        energy_scale=np.asarray([1.3], dtype=np.float64),
        energy_shift=np.asarray([0.2], dtype=np.float64),
        streaming=streaming,
        key=jr.key(seed),
    )


def maceles_model(
    mace: MACEPotential,
    /,
    *,
    atom_capacity: int,
    cell: np.ndarray | None = None,
    arguments: dict[str, Any] | None = None,
    seed: int = 11,
) -> MACELESPotential:
    """MACELES with source-default latent Ewald; periodic modes cover ``cell``."""
    definition = LatentEwaldDefinition.from_source_arguments(arguments or {})
    resources = LatentEwaldResources(
        mode_extents=None
        if cell is None
        else definition.covering_extents(float(np.max(np.linalg.norm(cell, axis=1))))
    )
    return MACELESPotential.initialize(
        mace,
        definition,
        atom_capacity=atom_capacity,
        resources=resources,
        key=jr.key(seed),
    )


def aimnet2_model(
    seed: int = 3, /, *, channels: int = 1, aev_cutoff: float = RC_S
) -> Any:
    """AIMNet2 rebuilt from a random source-layout inventory (no source runtime).

    ``aev_cutoff`` is the declared source ``aev.rc_s`` (also the source metadata
    neighbor radius); the short-range Coulomb radius ``RC_SR`` is a distinct
    declared radius that native admission requires to lie within it.
    """
    radius = float(aev_cutoff)
    declaration, tensors = source_inventory(seed, channels=channels, aev_cutoff=radius)
    core, state, _ = externalize_source_long_range(declaration, tensors)
    reconstruction = aimnet2_potential_from_source(
        SCALE, core, state, neighbor_cutoff=radius
    )
    if reconstruction.architecture.charge_channels != channels:
        raise RuntimeError("The AIMNet2 fixture did not declare the requested channels.")
    return reconstruction.potential


def uma_model(
    seed: int = 3,
    /,
    *,
    streaming: StreamedRelationPlan | None = None,
    node_block: int = 2,
    **config: Any,
) -> Any:
    """UMA rebuilt by the production source reconstruction from random tensors."""
    model_config, tasks = uma_config(**config)
    declaration = parse_uma_source_config(
        model_config, tasks, roll_gauge="two-chart-nlerp-zero-roll"
    )
    model, residual = uma_model_from_source(
        SCALE,
        declaration,
        uma_initial_source_tensors(declaration, key=jr.key(seed)),
        streaming=StreamedRelationPlan(receiver_tile=2, edge_tile=4, channel_capacity=512)
        if streaming is None
        else streaming,
        node_block=node_block,
    )
    if residual >= 1e-6:
        raise RuntimeError(
            f"UMA stored-basis residual {residual} exceeds the fixture bound."
        )
    return model


def uma_property_model(seed: int = 7, /) -> Any:
    """UMA whose source energy heads are mean-reduced: typed reports, no energy."""
    model_config, tasks = uma_config(direct=True, ff_type="spectral")
    for name in ("omat_energy", "omol_energy"):
        model_config["heads"][name]["reduce"] = "mean"
    declaration = parse_uma_source_config(model_config, tasks)
    model, _ = uma_model_from_source(
        SCALE, declaration, uma_initial_source_tensors(declaration, key=jr.key(seed))
    )
    return model


def uma_task(*, periodic: bool) -> str:
    """Materials task for periodic cells, molecular task for finite clusters."""
    return "omat" if periodic else "omol"


def uma_spin_index(task: str, multiplicity: int, /) -> int:
    """Source spin index: multiplicity for ``omol``; the null token 0 otherwise."""
    return int(multiplicity) if task == "omol" else 0


def uma_binding(
    model: Any,
    batch: AtomisticBatch,
    *,
    periodic: bool,
    charge: int = 0,
    multiplicity: int = 1,
) -> UMAMoLEBinding:
    """Merged-MoLE binding of the batch's first case composition and conditions."""
    task = uma_task(periodic=periodic)
    return UMAMoLEBinding(
        composition(batch),
        total_charge=charge,
        spin_index=uma_spin_index(task, multiplicity),
        task=model.schema.code(task),
        composition_policy=model.configuration.composition_policy,
    )


# --------------------------------------------------------------- conditions


@dataclass(frozen=True, slots=True)
class ConditionColumn:
    """One structure-level stored condition and its per-case values."""

    field: AtomisticStoreField
    values: np.ndarray


def _column(
    name: str,
    values: npt.ArrayLike,
    dtype: AtomisticStoreDType,
    *,
    kind: AtomisticStoreValueKind,
    convention: str,
    unit: UnitDefinition | None = None,
    extensivity: AtomisticStoreExtensivity = "not-applicable",
) -> ConditionColumn:
    field = AtomisticStoreField(
        name,
        "structure",
        dtype,
        kind=kind,
        role="condition",
        convention=convention,
        unit=unit,
        extensivity=extensivity,
    )
    return ConditionColumn(field, np.asarray(values, dtype=field.numpy_dtype))


@dataclass(frozen=True)
class FamilyScenario:
    """A native family model with its exact condition mapping.

    ``conditioning(batch, states)`` and ``columns(states)`` map one physical
    ``(total charge, multiplicity)`` per case to the family's own condition
    fields; a family without a condition port refuses any non-neutral state.
    """

    name: str
    model: Any
    periodic: bool
    conditioning: Callable[
        [AtomisticBatch, Sequence[tuple[int, int]]], AtomisticConditioning | None
    ]
    columns: Callable[[Sequence[tuple[int, int]]], tuple[ConditionColumn, ...]]

    @property
    def spec(self) -> AtomisticConditioningSpec | None:
        if self.name.startswith("aimnet2"):
            return self.model.conditioning_spec
        if self.name == "uma":
            return self.model.schema.spec
        return None


def _unconditioned(name: str, /) -> tuple[Callable[..., Any], Callable[..., Any]]:
    def refuse(states: Sequence[tuple[int, int]]) -> None:
        if any(tuple(state) != NEUTRAL for state in states):
            raise ValueError(f"{name} has no charge or spin condition port.")

    def conditioning(batch: AtomisticBatch, states: Sequence[tuple[int, int]]) -> None:
        del batch
        refuse(states)

    def columns(states: Sequence[tuple[int, int]]) -> tuple[ConditionColumn, ...]:
        refuse(states)
        return ()

    return conditioning, columns


def _aimnet2_conditions(model: Any, /) -> tuple[Callable[..., Any], Callable[..., Any]]:
    nse = model.configuration.charge_channels == 2

    def values(states: Sequence[tuple[int, int]]) -> dict[str, np.ndarray]:
        charges = np.asarray([state[0] for state in states], dtype=np.float64)
        result = {"charge": charges}
        if nse:
            result["multiplicity"] = np.asarray(
                [state[1] for state in states], dtype=np.float64
            )
        elif any(int(state[1]) != 1 for state in states):
            raise ValueError("Single-channel AIMNet2 has no spin multiplicity condition.")
        return result

    def conditioning(
        batch: AtomisticBatch, states: Sequence[tuple[int, int]]
    ) -> AtomisticConditioning:
        return AtomisticConditioning(
            model.conditioning_spec,
            values(states),
            case_count=batch.case_count,
            atom_capacity=batch.atom_capacity,
        )

    def columns(states: Sequence[tuple[int, int]]) -> tuple[ConditionColumn, ...]:
        stored = values(states)
        # Continuous total charge in e; the multiplicity 2S + 1 is an exact
        # unitless integer that the continuous condition field reads exactly.
        result = [
            _column(
                "charge",
                stored["charge"],
                "float64",
                kind="continuous",
                convention="total-charge",
                unit=ELEMENTARY_CHARGE,
                extensivity="extensive",
            )
        ]
        if nse:
            result.append(
                _column(
                    "multiplicity",
                    [int(state[1]) for state in states],
                    "int64",
                    kind="integer",
                    convention="spin-multiplicity",
                )
            )
        return tuple(result)

    return conditioning, columns


def _uma_conditions(
    model: Any, /, *, periodic: bool
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    task = uma_task(periodic=periodic)

    def conditioning(
        batch: AtomisticBatch, states: Sequence[tuple[int, int]]
    ) -> AtomisticConditioning:
        return model.schema.conditioning(
            [task] * len(states),
            atom_capacity=batch.atom_capacity,
            charges=[int(state[0]) for state in states],
            spins=[uma_spin_index(task, state[1]) for state in states],
        )

    def columns(states: Sequence[tuple[int, int]]) -> tuple[ConditionColumn, ...]:
        return (
            # Categorical in the model, but an elementary-charge count.
            _column(
                "total_charge",
                [int(state[0]) for state in states],
                "int64",
                kind="integer",
                convention="total-charge",
                unit=ELEMENTARY_CHARGE,
                extensivity="extensive",
            ),
            _column(
                "uma_spin_index",
                [uma_spin_index(task, state[1]) for state in states],
                "int64",
                kind="categorical",
                convention="uma-source-spin-index-null-zero-else-multiplicity",
            ),
            _column(
                "uma_task",
                [model.schema.code(task)] * len(states),
                "int64",
                kind="categorical",
                convention="uma-sorted-dataset-mapping-key-code",
            ),
        )

    return conditioning, columns


def family_scenario(
    name: str,
    /,
    *,
    periodic: bool,
    atom_capacity: int,
    cell: np.ndarray | None = None,
    seed: int = 3,
    model: Any = None,
) -> FamilyScenario:
    """The named family (random/source-layout parameters) and its conditions.

    ``model`` overrides the default construction (a teacher, a restored
    artifact or a re-seeded student of the same family).
    """
    match name:
        case "mace":
            resolved = mace_model(seed) if model is None else model
            conditioning, columns = _unconditioned(name)
        case "maceles":
            if periodic and cell is None:
                raise ValueError("Periodic MACELES needs the cell its modes must cover.")
            resolved = (
                maceles_model(
                    mace_model(seed),
                    atom_capacity=atom_capacity,
                    cell=cell if periodic else None,
                )
                if model is None
                else model
            )
            conditioning, columns = _unconditioned(name)
        case "aimnet2" | "aimnet2-nse":
            channels = 2 if name == "aimnet2-nse" else 1
            resolved = aimnet2_model(seed, channels=channels) if model is None else model
            conditioning, columns = _aimnet2_conditions(resolved)
        case "uma":
            resolved = uma_model(seed, ff_type="spectral") if model is None else model
            conditioning, columns = _uma_conditions(resolved, periodic=periodic)
        case _:
            raise ValueError(f"Unknown native family {name!r}.")
    return FamilyScenario(name, resolved, periodic, conditioning, columns)


# ------------------------------------------------------------ training data


def conditioned_training_source(
    batch: AtomisticBatch,
    columns: Sequence[ConditionColumn],
    /,
    *,
    energy: Any,
    forces: Any,
    stress: Any = None,
    record_ids: Sequence[str],
) -> AtomisticStructureSource:
    """Canonical labeled source whose schema adds structure-level conditions."""
    base = atomistic_training_source(
        batch,
        mass_unit=DALTON,
        energy=energy,
        forces=forces,
        stress=stress,
        record_ids=record_ids,
    )
    if not columns:
        return base
    for column in columns:
        if column.values.shape != (base.record_count,):
            raise ValueError(f"Condition {column.field.name!r} needs one value per case.")
    schema = AtomisticStructureStoreSchema(
        base.schema.levels, (*base.schema.fields, *(column.field for column in columns))
    )
    rows = base.read(np.arange(base.record_count, dtype=np.int64))
    offsets = rows.offsets(base.schema.atom_level)
    records = []
    for index in range(base.record_count):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        values: dict[str, np.ndarray] = {}
        available: dict[str, np.ndarray] = {}
        for name in rows.fields:
            field = base.schema.field(name)
            span = (
                slice(index, index + 1)
                if field.level == "structure"
                else slice(start, stop)
            )
            values[name] = np.asarray(rows.values[name][span])
            if field.availability == "optional":
                available[name] = np.asarray(rows.available[name][span])
        for column in columns:
            values[column.field.name] = column.values[index : index + 1]
        records.append(
            AtomisticStructureRecord(
                schema,
                record_id=base.record_ids[index],
                lineage_id=base.lineage_ids[index],
                counts={base.schema.atom_level: [stop - start]},
                stable_ids=np.asarray(rows.stable_ids[start:stop]),
                values=values,
                available=available,
            )
        )
    return AtomisticStructureSource(schema, tuple(records))


def training_fields(
    scenario: FamilyScenario, columns: Sequence[ConditionColumn], /
) -> AtomisticTrainingFields:
    """Loader slots of geometry, periodic cell and every stored condition."""
    periodic = {"cell": "cell", "periodic": "periodic"} if scenario.periodic else {}
    if not columns:
        return AtomisticTrainingFields(**periodic)
    return AtomisticTrainingFields(
        **periodic,
        conditioning=scenario.spec,
        conditions={column.field.name: column.field.name for column in columns},
    )


def teacher_source(
    teacher: FamilyScenario,
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    states: Sequence[tuple[int, int]],
    /,
    *,
    prefix: str,
) -> AtomisticStructureSource:
    """E/F (and periodic S) labels of a fixed native teacher, with conditions.

    The teacher is another random model of the same family: the labels define
    a consistent conservative target, not reference physics.
    """
    labels = energy_and_forces(
        teacher.model,
        batch,
        execution,
        compute_stress=teacher.periodic,
        conditioning=teacher.conditioning(batch, states),
    )
    if not bool(np.all(np.asarray(labels.valid))):
        raise RuntimeError(
            f"Teacher labels of {teacher.name} failed: {np.asarray(labels.status)}."
        )
    return conditioned_training_source(
        batch,
        teacher.columns(states),
        energy=labels.energy,
        forces=labels.forces,
        stress=labels.stress if teacher.periodic else None,
        record_ids=tuple(f"{prefix}-{index}" for index in range(batch.case_count)),
    )


def training_problem(
    teacher: FamilyScenario,
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    states: Sequence[tuple[int, int]],
    /,
    *,
    prefix: str = "family",
) -> Any:
    """Full-batch E/F (+S when periodic) problem through the canonical loader."""
    source = teacher_source(teacher, batch, execution, states, prefix=prefix)
    objectives = [
        phx.atomistic.AtomisticEnergyObjective(),
        phx.atomistic.AtomisticForceObjective(),
    ]
    if teacher.periodic:
        objectives.append(phx.atomistic.AtomisticStressObjective())
    return phx.atomistic.AtomisticTrainingProblem(
        phx.atomistic.atomistic_training_loader(source, shuffle=False),
        execution,
        tuple(objectives),
        scale=SCALE,
        cutoff=float(teacher.model.configuration.cutoff),
        fields=training_fields(teacher, teacher.columns(states)),
    )


__all__ = [
    "CHARGE_STATES",
    "ConditionColumn",
    "FAMILIES",
    "FamilyScenario",
    "MACE_CUTOFF",
    "NEUTRAL",
    "SCALE",
    "SHEAR",
    "UNITS",
    "WATER",
    "aimnet2_model",
    "composition",
    "conditioned_training_source",
    "deformed_batch",
    "energy_oracle",
    "family_scenario",
    "finite_execution",
    "mace_model",
    "maceles_model",
    "periodic_execution",
    "teacher_source",
    "training_fields",
    "training_problem",
    "uma_binding",
    "uma_model",
    "uma_property_model",
    "uma_spin_index",
    "uma_task",
    "water_batch",
    "water_box",
    "water_molecules",
]


def deformed_batch(
    batch: AtomisticBatch, positions: Any, deformation: np.ndarray
) -> AtomisticBatch:
    """Preserve material identity at ``x @ F.T`` in the cell ``H @ F.T``."""
    return AtomisticBatch(
        np.asarray(batch.atomic_numbers),
        np.asarray(positions, dtype=np.float64) @ deformation.T,
        np.asarray(batch.masses),
        batch.scale,
        particle_ids=np.asarray(batch.particle_ids),
        atom_type_ids=np.asarray(batch.atom_type_ids),
        element_mask=np.asarray(batch.element_mask),
        atom_mask=np.asarray(batch.atom_mask),
        cells=None if batch.cells is None else np.asarray(batch.cells) @ deformation.T,
        periodic_axes=(
            None if batch.periodic_axes is None else np.asarray(batch.periodic_axes)
        ),
        structure_ids=batch.structure_ids,
        coordinate_dtype=batch.positions.dtype,
    )


@eqx.filter_jit
def _payload_energy(model: Any, payload: Any, conditioning: Any) -> tuple[Any, Any]:
    derivative = atomistic_energy_derivatives(
        model,
        payload,
        payload.positions,
        compute_forces=False,
        conditioning=conditioning,
    )
    return derivative.energy[0], derivative.successful[0]


def energy_oracle(
    model: Any,
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    conditioning: AtomisticConditioning | None = None,
) -> Callable[[Any, Any], float]:
    """Case-0 energy at positions and homogeneous strain, with fresh graph evidence.

    Every probe prepares its native payload at the probed geometry. Routes
    certified only at the baseline must not enter a finite difference as NaN.
    """
    layout = AtomisticGraphLayout.for_batch(
        batch, execution, cutoff=float(model.configuration.cutoff)
    )

    def energy(positions: Any, strain: Any) -> float:
        deformation = np.eye(3, dtype=np.float64) + np.asarray(strain, dtype=np.float64)
        value, successful = _payload_energy(
            model,
            layout.prepare(deformed_batch(batch, positions, deformation)),
            conditioning,
        )
        assert bool(successful), "A finite-difference energy probe was not successful."
        return float(value)

    return energy
