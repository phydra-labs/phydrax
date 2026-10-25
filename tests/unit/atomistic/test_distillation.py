#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native distillation: teachers, labels, objectives, replay and the transaction.

Every control is independent of the code under test: teacher forces, stress
and Hessians are compared with finite differences of natively labeled
energies/forces, embedding matching with an explicit block-wise projector
reference, HVP parameter gradients with finite parameter perturbations, and
the Boltzmann estimator with closed-form discrete normalizations over the
authenticated generator densities of real multistate segments. Labels are
streamed into real bounded structure stores, including interrupted runs.
Phase interruption tests abandon the in-memory state at every committed
boundary (and between the label attachment and its phase checkpoint) and
resume from the restart with a freshly built plan.
"""

from __future__ import annotations

import itertools
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax._array_archive import read_array_archive, write_array_archive
from phydrax._differentiation import (
    DERIVATIVE_UNSUPPORTED,
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    GradientLevel,
    RegularityPolicy,
    SurfaceDerivative,
)
from phydrax._strict import StrictModule
from phydrax._trainable import combine_parameters, NonTrainableState, partition_parameters
from phydrax.atomistic import (
    AtomisticBatch,
    AtomisticCellListImageSearch,
    AtomisticGraphExecutionPlan,
    AtomisticModelOutputs,
    AtomisticObservable,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
)
from phydrax.atomistic._campaign import (
    AtomisticCampaignOutput,
    AtomisticCampaignStore,
    AtomisticStageOutcome,
    AtomisticTaskPacket,
)
from phydrax.atomistic._conditioning import AtomisticConditionLevel
from phydrax.atomistic._distillation import (
    atomistic_boltzmann_source,
    atomistic_generator_source,
    atomistic_label_id,
    AtomisticBoltzmannGroup,
    AtomisticBoltzmannObjective,
    AtomisticDistillationDrive,
    AtomisticDistillationQualification,
    AtomisticDistillationRollout,
    AtomisticEmbeddingObjective,
    AtomisticEmbeddingProjector,
    AtomisticExperienceAdmission,
    AtomisticExperienceBuffer,
    AtomisticFrameLabel,
    AtomisticGeneratorMixture,
    AtomisticGeneratorState,
    AtomisticHessianProbe,
    AtomisticHessianVectorObjective,
    AtomisticLabelRequest,
    AtomisticOnPolicyDistillationPlan,
    AtomisticOnPolicyDistillationState,
    AtomisticReplayDraw,
    AtomisticTeacher,
    BOLTZMANN_GENERATOR_SLOT,
    HESSIAN_PROBE_SLOT,
    label_atomistic_structures,
    read_atomistic_distillation_restart,
    require_atomistic_labels,
    run_atomistic_distillation_segment,
    TEACHER_ATOM_ENERGY_SLOT,
    TEACHER_EMBEDDING_SLOT,
    TEACHER_ENERGY_SLOT,
    TEACHER_FORCES_SLOT,
    TEACHER_HVP_SLOT,
    TEACHER_STRESS_SLOT,
)
from phydrax.atomistic._dynamics import AtomisticDynamicsPlan
from phydrax.atomistic._model_inputs import AtomisticGraphInputs
from phydrax.atomistic._potential import (
    AbstractAtomisticPotential,
    atomistic_model_revision,
)
from phydrax.atomistic._potential_program import (
    AtomisticPotentialProgram,
    LearnedGraphPotentialTerm,
)
from phydrax.atomistic._structure_store import (
    AtomisticStoreError,
    AtomisticStoreLayout,
    AtomisticStoreLimits,
    AtomisticStructureRecord,
    AtomisticStructureSource,
    AtomisticStructureStoreWriter,
    open_atomistic_structure_store,
)
from phydrax.atomistic._system import AtomisticSystemPlan
from phydrax.atomistic._thermal import (
    BAOABLangevinPlan,
    NoseHooverChainParameters,
    NoseHooverChainPlan,
)
from phydrax.atomistic._thermodynamic import (
    AtomisticPhaseSpaceMeasurePlan,
    AtomisticThermodynamicStatePlan,
)
from phydrax.atomistic._training import (
    atomistic_training_loader,
    atomistic_training_source,
    AtomisticBatchPreparation,
    AtomisticTrainedTree,
    AtomisticTrainingFields,
    AtomisticTrainingPolicy,
    AtomisticTrainingProblem,
    evaluate_atomistic_objectives,
    fit_atomistic_normalization,
    fit_atomistic_potential,
)
from phydrax.atomistic._training_data import AtomisticLoaderError
from phydrax.atomistic._training_objectives import (
    atomistic_training_derivatives,
    AtomisticAtomEnergyObjective,
    AtomisticEnergyObjective,
    AtomisticForceObjective,
    AtomisticStressObjective,
)
from phydrax.atomistic._units import AtomisticUnitSystem
from phydrax.atomistic.sampling import (
    AtomisticBiasRowBinding,
    AtomisticBiasTerm,
    AtomisticCanonicalSamplingQualification,
    AtomisticDrawConfigurationCapture,
    AtomisticMultistatePlan,
    AtomisticMultistateSegmentPlan,
    AtomisticSAMSPlan,
    CollectiveVariableKind,
    HarmonicBiasForm,
    PreparedThermodynamicStateTable,
)
from phydrax.discretization import ParticleImageCapacity, PeriodicCell
from phydrax.nn.atomistic import MACEArchitecture, MACEPotential
from phydrax.nn.atomistic._mace import mace_node_feature_port
from phydrax.nn.operator.representations._irreps import O3IrrepBlock, O3IrrepLayout
from phydrax.units import ANGSTROM, DALTON, ELECTRONVOLT
from phydrax.uq import FreeEnergySelectionPlan
from tests._support.atomistic_bundles import (
    neighborhood_bundle_plan,
    role_capacity,
    search_capacity,
)
from tests._support.atomistic_dynamics_fixtures import (
    atomistic_system,
    ExternalFieldTerm,
    measure,
    prepare_potential,
)
from tests._support.atomistic_sampling_fixtures import cv_program


pytest.importorskip("zarr")


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
CUTOFF = 3.0
EXECUTION = AtomisticGraphExecutionPlan(4, maximum_dense_atoms=3)
FIELDS = AtomisticTrainingFields()
AE = RegularityPolicy(allow_almost_everywhere=True)
CELL = np.asarray([[2.8, 0.0, 0.0], [0.3, 2.9, 0.0], [0.2, 0.1, 3.0]])
REDUCED = AtomisticUnitSystem.reduced()
_STORE_ROOTS: list[Path] = []
_STORE_ORDINAL = itertools.count()


# ---------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _label_store_root(tmp_path: Path) -> Iterator[None]:
    """Every labeling of a test publishes into its own store under ``tmp_path``."""
    _STORE_ROOTS.append(tmp_path / "label-stores")
    yield
    _STORE_ROOTS.pop()


def _store_limits(*, max_records: int = 100_000, max_read_bytes: int = 1 << 26) -> Any:
    return AtomisticStoreLimits(
        max_commit_bytes=1 << 22,
        max_member_bytes=1 << 24,
        max_shard_bytes=1 << 26,
        max_shard_members=1 << 14,
        max_chunk_bytes=1 << 20,
        max_index_bytes=1 << 24,
        max_decoded_bytes=1 << 26,
        max_read_bytes=max_read_bytes,
        max_records=max_records,
    )


def _writer(root: Path | None = None, *, max_records: int = 100_000) -> Any:
    """A bounded canonical store writer (two records per shard and commit)."""
    target = _STORE_ROOTS[-1] / f"store-{next(_STORE_ORDINAL)}" if root is None else root
    return AtomisticStructureStoreWriter(
        target,
        layout=AtomisticStoreLayout(records_per_shard=2, target_chunk_bytes=4096),
        limits=_store_limits(max_records=max_records),
    )


def _mace(seed: int, *, channels: int = 3, cutoff: float = CUTOFF) -> MACEPotential:
    return MACEPotential(
        SCALE,
        MACEArchitecture(
            species=(1, 8),
            cutoff=cutoff,
            radial_basis_count=4,
            cutoff_power=5,
            channel_count=channels,
            hidden_degree=1,
            edge_degree=2,
            interactions=("real-agnostic-density", "real-agnostic-residual"),
            correlations=(2, 2),
            radial_widths=(6,),
            readout_width=4,
            average_neighbor_count=2.0,
            energy_scaling="scale-shift",
        ),
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        energy_scale=np.asarray([1.3], dtype=np.float64),
        energy_shift=np.asarray([0.2], dtype=np.float64),
        key=jr.key(seed),
    )


_WATER = np.asarray([[0.0, 0.0, 0.0], [0.95, 0.1, 0.0], [-0.25, 0.9, 0.15]])


def _molecules(positions: np.ndarray, *, prefix: str = "m") -> AtomisticBatch:
    count = positions.shape[0]
    return AtomisticBatch(
        np.tile(np.asarray([8, 1, 1], dtype=np.int32), (count, 1)),
        positions,
        np.tile(np.asarray([15.999, 1.008, 1.008]), (count, 1)),
        SCALE,
        structure_ids=tuple(f"{prefix}-{index}" for index in range(count)),
    )


def _water_source(
    count: int = 3, *, seed: int = 7, prefix: str = "m"
) -> AtomisticStructureSource:
    rng = np.random.default_rng(seed)
    positions = _WATER[None] + 0.05 * rng.normal(size=(count, 3, 3))
    return atomistic_training_source(
        _molecules(positions, prefix=prefix), mass_unit=DALTON
    )


def _loader(*sources: AtomisticStructureSource, batch_size: int | None = None) -> Any:
    return atomistic_training_loader(*sources, batch_size=batch_size, shuffle=False)


def _label(
    teacher: AtomisticTeacher,
    source: Any,
    request: AtomisticLabelRequest,
    *,
    execution: AtomisticGraphExecutionPlan = EXECUTION,
    fields: AtomisticTrainingFields = FIELDS,
    store: Any = None,
    batch_size: int | None = None,
) -> Any:
    return label_atomistic_structures(
        teacher,
        _loader(source, batch_size=batch_size),
        fields,
        request,
        graph_execution=execution,
        cutoff=CUTOFF,
        store=_writer() if store is None else store,
    )


def _column(source: AtomisticStructureSource, name: str) -> tuple[np.ndarray, Any]:
    batch = source.read(np.arange(source.record_count, dtype=np.int64))
    return np.asarray(batch.values[name]), batch


def _first_batch(loader: Any, slots: tuple[str, ...]) -> Any:
    preparation = AtomisticBatchPreparation(
        loader, EXECUTION, scale=SCALE, cutoff=CUTOFF, fields=FIELDS, label_slots=slots
    )
    with loader.epoch(loader.start(0)) as epoch:
        item = next(iter(epoch))
    return preparation.prepare_batch(
        item.payload, item.provenance, coordinate_dtype="float64"
    )


def _node_layout(model: MACEPotential) -> O3IrrepLayout:
    architecture = model.configuration
    return O3IrrepLayout(
        tuple(
            O3IrrepBlock(
                f"layer{index}_{block.name}",
                block.degree,
                block.parity,
                multiplicity=block.multiplicity,
            )
            for index in range(architecture.interaction_count)
            for block in architecture.output_layout(index).blocks
        )
    )


def _embedding(model: MACEPotential) -> tuple[Any, Any, O3IrrepLayout]:
    return (
        AtomisticObservable.NODE_EMBEDDINGS,
        mace_node_feature_port(model.configuration, AtomisticConditionLevel.ATOM),
        _node_layout(model),
    )


def _leaves(tree: Any) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


# -------------------------------------------------- analytic harmonic potential


class _Configuration(StrictModule, NonTrainableState):
    cutoff: float = eqx.field(static=True)


class _Harmonic(AbstractAtomisticPotential):
    """``E = stiffness |x|^2 / 2`` summed over atoms; contract set per instance."""

    stiffness: jax.Array
    configuration: _Configuration
    scale: AtomisticScaleContract
    precision: AtomisticPrecisionPolicy
    parameter_derivatives: bool = eqx.field(static=True)
    architecture_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        stiffness: float,
        *,
        parameter_derivatives: bool = True,
        scale: AtomisticScaleContract = SCALE,
    ) -> None:
        self.stiffness = jnp.asarray(stiffness, dtype=jnp.float64)
        self.configuration = _Configuration(cutoff=1.0)
        self.scale = scale
        self.precision = AtomisticPrecisionPolicy()
        self.parameter_derivatives = parameter_derivatives
        self.architecture_id = "test-harmonic-well"
        self.method_id = "negative-position-gradient"

    @property
    def derivative_contract(self) -> DerivativeContract:
        surfaces = (SurfaceDerivative(DerivativeSurface.INPUT, GradientLevel.SMOOTH),)
        if self.parameter_derivatives:
            surfaces += (
                SurfaceDerivative(
                    DerivativeSurface.MODEL_PARAMETER, GradientLevel.SMOOTH
                ),
            )
        return DerivativeContract(
            surfaces,
            route=DerivativeRoute.DIRECT,
            regularity=DerivativeRegularity.smooth(degree_bound=2),
        )

    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        del batch

    def evaluate_graph(
        self, inputs: AtomisticGraphInputs, request: Any = None, /
    ) -> AtomisticModelOutputs:
        cases, atoms = inputs.case_count, inputs.atom_capacity
        positions = inputs.positions.reshape((cases, atoms, 3))
        mask = inputs.atom_mask.reshape((cases, atoms))
        atom_energy = jnp.where(
            mask, 0.5 * self.stiffness * jnp.sum(positions * positions, axis=-1), 0.0
        )
        return AtomisticModelOutputs(
            atom_energy.sum(axis=1), atom_energy, inputs.successful
        )


_TWO_ATOMS = AtomisticGraphExecutionPlan(2, maximum_dense_atoms=2)


def _sample_labels(source: Any, teacher: float) -> Any:
    """Reduced-unit harmonic teacher labels of (canonical or forged) sample records."""
    return label_atomistic_structures(
        AtomisticTeacher(
            _Harmonic(teacher, scale=REDUCED.scale), name="harmonic-teacher"
        ),
        _loader(source),
        FIELDS,
        AtomisticLabelRequest(energy=True, forces=False),
        graph_execution=_TWO_ATOMS,
        cutoff=1.0,
        store=_writer(),
    )


def _boltzmann_estimate(
    objective: AtomisticBoltzmannObjective, source: Any, student: float
) -> tuple[float, float]:
    loader = _loader(source)
    preparation = AtomisticBatchPreparation(
        loader,
        _TWO_ATOMS,
        scale=REDUCED.scale,
        cutoff=1.0,
        fields=FIELDS,
        label_slots=objective.label_slots,
    )
    with loader.epoch(loader.start(0)) as epoch:
        item = next(iter(epoch))
    batch = preparation.prepare_batch(
        item.payload, item.provenance, coordinate_dtype="float64"
    )
    term = objective.term(
        _Harmonic(student, scale=REDUCED.scale), None, batch, scale=1.0, dtype="float64"
    )
    return float(term.numerator), float(term.support)


# --------------------------------------------- authenticated multistate samples


TETHER = 1.0
UMBRELLA = 2.0


def _tethered_segment(
    temperatures: tuple[float, ...],
    *,
    capacity: int,
    windows: tuple[float, ...] | None = None,
    sams: bool = False,
    capture: bool = True,
    seed: int = 0,
    evidence: str = "tethered-radial-baoab",
) -> tuple[Any, Any, Any, Any]:
    """A real committed multistate segment of the reduced tethered radial system.

    Atom 0 is immobile at the origin and atom 1 moves under ``(k/2)|x1|^2``;
    with ``windows`` every row adds the umbrella ``(kappa/2)(r - c_k)^2`` of
    the distance ``r = |x1|`` (a frozen, stateless bias row). With
    ``capture`` the segment records every draw's configuration under an
    explicit byte budget. Returns ``(table, qualification, result, dynamics)``.
    """
    system = atomistic_system([0, 1], [1.0, 1.0], mobile_mask=[False, True])
    tether = ExternalFieldTerm("tether", "harmonic", [0.0, 0.0, 0.0], stiffness=TETHER)
    terms: tuple[Any, ...] = (tether,)
    if windows is not None:
        variables = cv_program(system, CollectiveVariableKind.DISTANCE, [0, 1])
        terms = (
            tether,
            AtomisticBiasTerm(
                variables,
                HarmonicBiasForm(np.asarray([1.0]), UMBRELLA),
                name="umbrella",
                force_group=1,
            ),
        )
    dynamics = AtomisticDynamicsPlan(
        system, prepare_potential(system, terms), BAOABLangevinPlan(0.1, 2.0)
    ).prepare()
    phase_space = measure(system)
    rows = tuple(
        AtomisticThermodynamicStatePlan(
            phase_space,
            ensemble="nvt",
            temperature=temperature,
            state_id=f"row-{index}",
            bias=None
            if windows is None
            else AtomisticBiasRowBinding({"umbrella": np.asarray([windows[index]])}),
        )
        for index, temperature in enumerate(temperatures)
    )
    table = PreparedThermodynamicStateTable(dynamics, rows)
    qualification = AtomisticCanonicalSamplingQualification(
        dynamics, table, evidence, sampling_exact=False, sampling_bias_bound=0.03
    )
    count = len(rows)
    runtime = AtomisticMultistatePlan(
        table,
        np.arange(count) + 100,
        qualification=qualification,
        sams=AtomisticSAMSPlan(np.full(count, 1.0 / count), adaptation_steps=8)
        if sams
        else None,
        run_id=f"distillation-generators-{seed}",
    ).prepare(dynamics)
    lanes = tuple(
        dynamics.initialize_state(
            jnp.asarray([[0.0, 0.0, 0.0], [1.0 + 0.1 * index, 0.0, 0.0]]),
            table,
            state_index=index,
            velocity=jnp.zeros((2, 3)),
            key=jr.key(seed + index),
        )
        for index in range(count)
    )
    initial = runtime.initialize(
        lanes, jnp.arange(count, dtype=jnp.int32), jr.key(seed + 50)
    )
    result = AtomisticMultistateSegmentPlan(
        runtime,
        capacity,
        0,
        0,
        runtime.initial_continuation_id,
        configurations=AtomisticDrawConfigurationCapture(max_bytes=1 << 22)
        if capture
        else None,
    ).run(initial)
    assert bool(result.successful) and int(result.count) == capacity
    return table, qualification, result, dynamics


def _draw_configurations(result: Any) -> np.ndarray:
    """Independent draw-major view ``(capacity * R, N, 3)`` of the raw capture."""
    positions = np.asarray(result.draw_positions, dtype=np.float64)
    return positions.reshape((-1, *positions.shape[2:]))


def _replaced(source: Any, row: int, positions: np.ndarray) -> Any:
    """The same records with record ``row`` moved to ``positions`` (a forged frame)."""
    records = list(_records(source))
    original = records[row]
    values = dict(original.values)
    values["positions"] = np.asarray(positions, dtype=np.float64)
    records[row] = AtomisticStructureRecord(
        source.schema,
        record_id=original.record_id,
        lineage_id=original.lineage_id,
        counts={"atom": [positions.shape[0]]},
        stable_ids=original.stable_ids,
        values=values,
        available=dict(original.available),
    )
    return AtomisticStructureSource(source.schema, tuple(records))


def _discrete_kl(
    reduced_p: np.ndarray, reduced_s: np.ndarray, reduced_q: np.ndarray, weight: float
) -> float:
    """Exact mixed KL between the sample-supported p ~ e^-(u_p-u_q), s ~ e^-(u_s-u_q)."""
    p = np.exp(-(reduced_p - reduced_q) - np.max(-(reduced_p - reduced_q)))
    s = np.exp(-(reduced_s - reduced_q) - np.max(-(reduced_s - reduced_q)))
    p, s = p / p.sum(), s / s.sum()
    return weight * float(np.sum(p * np.log(p / s))) + (1.0 - weight) * float(
        np.sum(s * np.log(s / p))
    )


# ---------------------------------------------------------------------- teacher


def test_teacher_frozen_view_stops_parameter_gradients_and_keeps_forces() -> None:
    teacher = AtomisticTeacher(_mace(1), name="teacher")
    batch = _first_batch(_loader(_water_source()), ())

    def total(model: Any) -> jax.Array:
        forces = atomistic_training_derivatives(model, batch, forces=True).forces
        assert forces is not None
        return jnp.sum(forces**2)

    frozen_gradient = eqx.filter_grad(lambda holder: total(holder.frozen))(teacher)
    assert all(np.all(leaf == 0.0) for leaf in _leaves(frozen_gradient))
    live_gradient = eqx.filter_grad(total)(teacher.model)
    assert any(np.any(leaf != 0.0) for leaf in _leaves(live_gradient))
    forces = atomistic_training_derivatives(teacher.frozen, batch, forces=True).forces
    assert forces is not None and bool(jnp.any(forces != 0.0))
    reference = atomistic_training_derivatives(teacher.model, batch, forces=True).forces
    np.testing.assert_array_equal(np.asarray(forces), np.asarray(reference))


def test_teacher_forces_equal_energy_finite_differences() -> None:
    step = 1.0e-5
    displaced = [_WATER]
    for atom in range(3):
        for axis in range(3):
            for sign in (1.0, -1.0):
                moved = _WATER.copy()
                moved[atom, axis] += sign * step
                displaced.append(moved)
    source = atomistic_training_source(
        _molecules(np.asarray(displaced), prefix="fd"), mass_unit=DALTON
    )
    labeled = _label(
        AtomisticTeacher(_mace(1), name="teacher"), source, AtomisticLabelRequest()
    )
    energies, _ = _column(labeled, TEACHER_ENERGY_SLOT)
    forces, _ = _column(labeled, TEACHER_FORCES_SLOT)
    expected = np.zeros((3, 3))
    for index in range(9):
        plus, minus = energies[1 + 2 * index], energies[2 + 2 * index]
        expected[index // 3, index % 3] = -(plus - minus) / (2.0 * step)
    np.testing.assert_allclose(forces[:3], expected, rtol=1e-6, atol=1e-8)


def test_teacher_stress_equals_strain_finite_differences() -> None:
    execution = AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=8,
            maximum_edges=1024,
            maximum_degree=64,
            maximum_images=125,
        ),
        search_cell=CELL,
        search_margin=0.1,
    )
    positions = np.asarray([[0.0, 0.0, 0.0], [1.1, 0.3, 0.2]])
    step = 1.0e-6
    geometry = [(positions, CELL)]
    for row in range(3):
        for column in range(3):
            for sign in (1.0, -1.0):
                strain = np.eye(3)
                strain[row, column] += 0.5 * sign * step
                strain[column, row] += 0.5 * sign * step
                geometry.append((positions @ strain, CELL @ strain))
    count = len(geometry)
    batch = AtomisticBatch(
        np.tile(np.asarray([1, 8], dtype=np.int32), (count, 1)),
        np.stack([value[0] for value in geometry]),
        np.tile(np.asarray([1.008, 15.999]), (count, 1)),
        SCALE,
        cells=np.stack([value[1] for value in geometry]),
        periodic_axes=np.ones((count, 3), dtype=np.bool_),
        structure_ids=tuple(f"strain-{index}" for index in range(count)),
    )
    fields = AtomisticTrainingFields(cell="cell", periodic="periodic")
    labeled = _label(
        AtomisticTeacher(_mace(1), name="teacher"),
        atomistic_training_source(batch, mass_unit=DALTON),
        AtomisticLabelRequest(stress=True),
        execution=execution,
        fields=fields,
    )
    energies, columns = _column(labeled, TEACHER_ENERGY_SLOT)
    stress = np.asarray(columns.values[TEACHER_STRESS_SLOT])
    assert bool(np.all(columns.available[TEACHER_STRESS_SLOT]))
    volume = abs(float(np.linalg.det(CELL)))
    expected = np.zeros((3, 3))
    for index in range(9):
        plus, minus = energies[1 + 2 * index], energies[2 + 2 * index]
        expected[index // 3, index % 3] = (plus - minus) / (2.0 * step) / volume
    observed = 0.5 * (stress[0] + stress[0].T)
    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-8)
    problem = AtomisticTrainingProblem(
        _loader(labeled),
        execution,
        (AtomisticStressObjective(TEACHER_STRESS_SLOT, name="stress", scale=1.0),),
        scale=SCALE,
        cutoff=CUTOFF,
        fields=fields,
    )
    teacher_view = AtomisticTrainedTree(_mace(1), (None,))
    evaluation = evaluate_atomistic_objectives(
        teacher_view, problem, fit_atomistic_normalization(problem), problem.preparation
    )
    assert evaluation.successful
    assert evaluation.values[0] == pytest.approx(0.0, abs=1e-20)


# ----------------------------------------------------------------- label identity


def test_offline_labels_are_idempotent_and_distinguish_teacher_definitions() -> None:
    source = _water_source()
    request = AtomisticLabelRequest()
    first_teacher = AtomisticTeacher(_mace(1), name="teacher")
    first = _label(first_teacher, source, request)
    again = _label(first_teacher, source, request)
    assert first.source_id == again.source_id
    assert first.record_ids == again.record_ids
    assert first.lineage_ids == source.lineage_ids
    require_atomistic_labels(first, first_teacher, request, source)

    other_revision = AtomisticTeacher(_mace(2), name="teacher")
    other = _label(other_revision, source, request)
    assert set(other.record_ids).isdisjoint(first.record_ids)
    assert atomistic_label_id(other_revision, request) != atomistic_label_id(
        first_teacher, request
    )
    with pytest.raises(ValueError, match="stale"):
        require_atomistic_labels(other, first_teacher, request, source)

    other_definition = AtomisticLabelRequest(
        hvp=AtomisticHessianProbe("gaussian", count=1, seed=3)
    )
    assert other_definition.request_id != request.request_id
    with pytest.raises(ValueError, match="stale"):
        require_atomistic_labels(first, first_teacher, other_definition, source)

    # A source whose records changed or were reordered refuses the old labels.
    changed = _water_source(seed=8)
    with pytest.raises(ValueError, match="stale"):
        require_atomistic_labels(first, first_teacher, request, changed)
    reversed_source = AtomisticStructureSource(
        source.schema, tuple(reversed(_records(source)))
    )
    with pytest.raises(ValueError, match="reordered"):
        require_atomistic_labels(first, first_teacher, request, reversed_source)

    # A training continuation refuses a reordered replay source (cursor drift).
    problem = AtomisticTrainingProblem(
        _loader(first),
        EXECUTION,
        (AtomisticEnergyObjective(TEACHER_ENERGY_SLOT, scale=1.0),),
        scale=SCALE,
        cutoff=CUTOFF,
    )
    result = fit_atomistic_potential(
        _mace(3), problem, AtomisticTrainingPolicy(maximum_steps=1), key=jr.key(0)
    )
    swapped = AtomisticStructureSource(first.schema, tuple(reversed(_records(first))))
    with pytest.raises(AtomisticLoaderError):
        _loader(swapped).require_cursor(result.cursor)


def _records(source: AtomisticStructureSource) -> tuple[AtomisticStructureRecord, ...]:
    batch = source.read(np.arange(source.record_count, dtype=np.int64))
    offsets = batch.offsets("atom")
    records = []
    for index in range(source.record_count):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        values = {}
        available = {}
        for name in batch.fields:
            field = source.schema.field(name)
            rows = (
                slice(index, index + 1)
                if field.level == "structure"
                else slice(start, stop)
            )
            values[name] = np.asarray(batch.values[name][rows])
            if field.availability == "optional":
                available[name] = np.asarray(batch.available[name][rows])
        records.append(
            AtomisticStructureRecord(
                source.schema,
                record_id=source.record_ids[index],
                lineage_id=source.lineage_ids[index],
                counts={"atom": [stop - start]},
                stable_ids=np.asarray(batch.stable_ids[start:stop]),
                values=values,
                available=available,
            )
        )
    return tuple(records)


def _read_limits_admitting(root: Path, records: int) -> Any:
    """The smallest power-of-two read bound admitting ``records`` stored records."""
    for exponent in range(6, 30):
        limits = _store_limits(max_read_bytes=1 << exponent)
        try:
            open_atomistic_structure_store(root, limits=limits).read(
                np.arange(records, dtype=np.int64)
            )
        except AtomisticStoreError:
            continue
        return limits
    raise AssertionError("No read bound admits the records.")


def test_offline_labels_stream_through_a_bounded_resumable_store(
    tmp_path: Path,
) -> None:
    raw = _water_source(6, seed=12, prefix="raw")
    raw_root = tmp_path / "raw"
    raw_writer = _writer(raw_root)
    raw_writer.append(raw_writer.create(raw.schema, name="raw"), raw)
    # Reads are admitted for one loader batch (two records), never the source.
    stored = open_atomistic_structure_store(
        raw_root, limits=_read_limits_admitting(raw_root, 2)
    )
    with pytest.raises(AtomisticStoreError, match="limit"):
        stored.read(np.arange(6, dtype=np.int64))
    teacher = AtomisticTeacher(_mace(1), name="teacher")
    request = AtomisticLabelRequest()
    reference = _label(
        teacher, stored, request, batch_size=2, store=_writer(tmp_path / "reference")
    )
    require_atomistic_labels(reference, teacher, request, stored)
    assert (
        reference.commit.store_name
        == f"teacher-labels-{atomistic_label_id(teacher, request)}"
    )
    # One create commit and one append per two-record chunk.
    assert reference.commit.sequence == 3
    in_memory, _ = _column(reference, TEACHER_ENERGY_SLOT)

    # Interrupted run: the third chunk is never committed (its append is refused).
    interrupted = tmp_path / "interrupted"
    with pytest.raises(AtomisticStoreError, match="limit"):
        _label(
            teacher,
            stored,
            request,
            batch_size=2,
            store=_writer(interrupted, max_records=4),
        )
    partial = open_atomistic_structure_store(interrupted, limits=_store_limits())
    assert partial.record_ids == reference.record_ids[:4]
    resumed = _label(teacher, stored, request, batch_size=2, store=_writer(interrupted))
    assert resumed.source_id == reference.source_id
    assert resumed.record_ids == reference.record_ids
    assert resumed.commit.sequence == 3
    energies, _ = _column(resumed, TEACHER_ENERGY_SLOT)
    np.testing.assert_array_equal(energies, in_memory)
    # A complete store is returned unchanged.
    again = _label(teacher, stored, request, batch_size=2, store=_writer(interrupted))
    assert again.commit.commit_id == resumed.commit.commit_id
    # Another label definition never overwrites or extends the store.
    with pytest.raises(ValueError, match="another output"):
        _label(
            AtomisticTeacher(_mace(2), name="teacher"),
            stored,
            request,
            batch_size=2,
            store=_writer(interrupted),
        )
    # A shuffled or repeated epoch has no canonical streaming order.
    with pytest.raises(ValueError, match="unshuffled"):
        label_atomistic_structures(
            teacher,
            atomistic_training_loader(stored, batch_size=2),
            FIELDS,
            request,
            graph_execution=EXECUTION,
            cutoff=CUTOFF,
            store=_writer(),
        )
    with pytest.raises(ValueError, match="exactly once"):
        label_atomistic_structures(
            teacher,
            atomistic_training_loader(stored, batch_size=2, shuffle=False, weights=[2]),
            FIELDS,
            request,
            graph_execution=EXECUTION,
            cutoff=CUTOFF,
            store=_writer(),
        )


# ------------------------------------------------------------- student updates


def test_real_student_update_keeps_teacher_leaves_unchanged() -> None:
    teacher = AtomisticTeacher(_mace(1), name="teacher")
    before = _leaves(teacher)
    labeled = _label(teacher, _water_source(4), AtomisticLabelRequest())
    problem = AtomisticTrainingProblem(
        _loader(labeled, batch_size=2),
        EXECUTION,
        (
            AtomisticEnergyObjective(TEACHER_ENERGY_SLOT, name="energy", scale=1.0),
            AtomisticForceObjective(TEACHER_FORCES_SLOT, name="forces", scale=1.0),
        ),
        scale=SCALE,
        cutoff=CUTOFF,
    )
    student = _mace(3)
    result = fit_atomistic_potential(
        student,
        problem,
        AtomisticTrainingPolicy(maximum_steps=3, learning_rate=1e-3),
        key=jr.key(4),
    )
    assert bool(result.successful)
    assert result.training_values.shape == (3,)
    assert atomistic_model_revision(result.potential).revision_id != (
        atomistic_model_revision(student).revision_id
    )
    for observed, expected in zip(_leaves(teacher), before, strict=True):
        np.testing.assert_array_equal(observed, expected)


def test_energy_force_atom_energy_objectives_equal_independent_reference() -> None:
    source = _water_source(3)
    teacher = AtomisticTeacher(_mace(1), name="teacher")
    student = _mace(3)
    request = AtomisticLabelRequest(atom_energy=True)
    labeled = _label(teacher, source, request)
    student_view = _label(AtomisticTeacher(student, name="student-view"), source, request)
    convention = labeled.schema.field(TEACHER_ATOM_ENERGY_SLOT).convention
    complete = convention.endswith("complete=true")
    objectives = (
        AtomisticEnergyObjective(
            TEACHER_ENERGY_SLOT, name="energy", reduction="total-structure", scale=1.0
        ),
        AtomisticForceObjective(TEACHER_FORCES_SLOT, name="forces", scale=1.0),
        AtomisticAtomEnergyObjective(
            TEACHER_ATOM_ENERGY_SLOT,
            name="atom-energy",
            decomposition=convention,
            complete=complete,
            scale=1.0,
        ),
    )
    problem = AtomisticTrainingProblem(
        _loader(labeled), EXECUTION, objectives, scale=SCALE, cutoff=CUTOFF
    )
    evaluation = evaluate_atomistic_objectives(
        AtomisticTrainedTree(student, (None, None, None)),
        problem,
        fit_atomistic_normalization(problem),
        problem.preparation,
    )
    assert evaluation.successful
    for index, slot in enumerate(
        (TEACHER_ENERGY_SLOT, TEACHER_FORCES_SLOT, TEACHER_ATOM_ENERGY_SLOT)
    ):
        target, _ = _column(labeled, slot)
        prediction, _ = _column(student_view, slot)
        reference = np.mean((prediction - target) ** 2)
        assert evaluation.values[index] == pytest.approx(reference, rel=1e-10)
    # The same objective against the teacher itself vanishes.
    zero = evaluate_atomistic_objectives(
        AtomisticTrainedTree(teacher.model, (None, None, None)),
        problem,
        fit_atomistic_normalization(problem),
        problem.preparation,
    )
    assert max(zero.values) == pytest.approx(0.0, abs=1e-20)
    # Another teacher's decomposition is another gauge.
    with pytest.raises(ValueError):
        AtomisticTrainingProblem(
            _loader(labeled),
            EXECUTION,
            (
                AtomisticAtomEnergyObjective(
                    TEACHER_ATOM_ENERGY_SLOT,
                    decomposition=convention.replace(teacher.teacher_id, "0" * 64),
                    complete=complete,
                    scale=1.0,
                ),
            ),
            scale=SCALE,
            cutoff=CUTOFF,
        ).admit(student)


# --------------------------------------------------------------------- embeddings


def _projected_reference(
    projector: AtomisticEmbeddingProjector, values: np.ndarray
) -> np.ndarray:
    """Block-wise ``W_k x / sqrt(f_k)`` written out over the declared layouts."""
    student, teacher = projector.student_layout, projector.teacher_layout
    blocks = []
    start = 0
    for block in student.blocks:
        stop = start + block.size
        blocks.append(
            values[:, start:stop].reshape((-1, block.multiplicity, block.dimension))
        )
        start = stop
    outputs = []
    for weight, target in zip(projector.linear.weights, teacher.blocks, strict=True):
        stacked = np.concatenate(
            [
                blocks[index]
                for index, block in enumerate(student.blocks)
                if (block.degree, block.parity) == (target.degree, target.parity)
            ],
            axis=1,
        )
        mapped = np.einsum("of,afd->aod", np.asarray(weight), stacked) / np.sqrt(
            weight.shape[1]
        )
        outputs.append(mapped.reshape((mapped.shape[0], -1)))
    return np.concatenate(outputs, axis=1)


def test_embedding_objective_uses_equivariant_projector_and_trains_it() -> None:
    source = _water_source(3)
    teacher_model = _mace(1, channels=4)
    student = _mace(3, channels=3)
    teacher_spec = _embedding(teacher_model)
    student_spec = _embedding(student)
    labeled = _label(
        AtomisticTeacher(teacher_model, name="teacher"),
        source,
        AtomisticLabelRequest(energy=True, forces=False, embeddings=teacher_spec),
    )
    student_view = _label(
        AtomisticTeacher(student, name="student-view"),
        source,
        AtomisticLabelRequest(energy=True, forces=False, embeddings=student_spec),
    )
    projector = AtomisticEmbeddingProjector(
        student_spec[2], teacher_spec[2], key=jr.key(9)
    )
    objective = AtomisticEmbeddingObjective(student_spec, teacher_spec, scale=1.0)
    problem = AtomisticTrainingProblem(
        _loader(labeled), EXECUTION, (objective,), scale=SCALE, cutoff=CUTOFF
    )
    evaluation = evaluate_atomistic_objectives(
        AtomisticTrainedTree(student, (projector,)),
        problem,
        fit_atomistic_normalization(problem),
        problem.preparation,
    )
    target, _ = _column(labeled, TEACHER_EMBEDDING_SLOT)
    features, _ = _column(student_view, TEACHER_EMBEDDING_SLOT)
    reference = np.mean((_projected_reference(projector, features) - target) ** 2)
    assert evaluation.successful
    assert evaluation.values[0] == pytest.approx(reference, rel=1e-10)

    before = _leaves(projector)
    result = fit_atomistic_potential(
        student,
        problem,
        AtomisticTrainingPolicy(maximum_steps=2, learning_rate=1e-2),
        key=jr.key(5),
        components={objective.name: projector},
    )
    assert bool(result.successful)
    trained = result.components[0]
    assert isinstance(trained, AtomisticEmbeddingProjector)
    assert trained.projector_id == projector.projector_id
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(_leaves(trained), before, strict=True)
    )
    with pytest.raises(ValueError, match="needs a trainable component"):
        fit_atomistic_potential(
            student, problem, AtomisticTrainingPolicy(maximum_steps=1), key=jr.key(5)
        )


def test_embedding_projector_refuses_unequal_irreps() -> None:
    student = O3IrrepLayout((O3IrrepBlock("scalars", 0, 1, multiplicity=3),))
    teacher = O3IrrepLayout(
        (
            O3IrrepBlock("scalars", 0, 1, multiplicity=2),
            O3IrrepBlock("vectors", 1, -1, multiplicity=2),
        )
    )
    with pytest.raises(ValueError, match="unequal irreps"):
        AtomisticEmbeddingProjector(student, teacher, key=jr.key(0))
    pseudo = O3IrrepLayout((O3IrrepBlock("pseudo", 0, -1, multiplicity=2),))
    with pytest.raises(ValueError, match="unequal irreps"):
        AtomisticEmbeddingProjector(student, pseudo, key=jr.key(0))
    scalar = AtomisticEmbeddingProjector(
        student, O3IrrepLayout((O3IrrepBlock("t", 0, 1, multiplicity=2),)), key=jr.key(0)
    )
    assert scalar.linear.weights[0].shape == (2, 3)


# ---------------------------------------------------------------------------- HVP


def test_hvp_labels_match_dense_hessian_and_student_gradient_matches_finite_differences() -> (
    None
):
    probe = AtomisticHessianProbe("rademacher", count=2, seed=5)
    source = _water_source(1)
    teacher = AtomisticTeacher(_mace(1), name="teacher")
    labeled = _label(teacher, source, AtomisticLabelRequest(hvp=probe))
    probes, _ = _column(labeled, HESSIAN_PROBE_SLOT)
    actions, _ = _column(labeled, TEACHER_HVP_SLOT)
    assert set(np.unique(probes)) <= {-1.0, 1.0}
    base = np.asarray(_column(source, "positions")[0])
    step = 1.0e-5
    displaced = []
    for coordinate in range(9):
        for sign in (1.0, -1.0):
            moved = base.copy()
            moved[coordinate // 3, coordinate % 3] += sign * step
            displaced.append(moved)
    forces, _ = _column(
        _label(
            teacher,
            atomistic_training_source(
                _molecules(np.asarray(displaced), prefix="h"), mass_unit=DALTON
            ),
            AtomisticLabelRequest(energy=False),
        ),
        TEACHER_FORCES_SLOT,
    )
    forces = forces.reshape((9, 2, 3, 3))
    hessian = -(forces[:, 0] - forces[:, 1]).reshape((9, 9)).T / (2.0 * step)
    for index in range(probe.count):
        expected = (hessian @ probes[:, index, :].reshape(-1)).reshape((3, 3))
        np.testing.assert_allclose(actions[:, index, :], expected, rtol=1e-5, atol=1e-7)

    objective = AtomisticHessianVectorObjective(probe, policy=AE, scale=1.0)
    batch = _first_batch(_loader(labeled), objective.label_slots)
    parameters, state, fixed = partition_parameters(_mace(3))

    def numerator(values: Any) -> jax.Array:
        term = objective.term(
            combine_parameters(values, state, fixed),
            None,
            batch,
            scale=1.0,
            dtype="float64",
        )
        return term.numerator

    gradient = jax.jit(jax.grad(numerator))(parameters)
    leaves, treedef = jax.tree.flatten(parameters)
    rng = np.random.default_rng(11)
    direction = jax.tree.unflatten(
        treedef, [jnp.asarray(rng.normal(size=leaf.shape)) for leaf in leaves]
    )
    directional = sum(
        jnp.vdot(left, right)
        for left, right in zip(
            jax.tree.leaves(gradient), jax.tree.leaves(direction), strict=True
        )
    )
    delta = 1.0e-6
    evaluate = jax.jit(numerator)
    shifted = [
        evaluate(
            jax.tree.map(lambda value, d: value + sign * delta * d, parameters, direction)
        )
        for sign in (1.0, -1.0)
    ]
    np.testing.assert_allclose(
        directional, (shifted[0] - shifted[1]) / (2.0 * delta), rtol=1e-5
    )
    student_view = _label(
        AtomisticTeacher(_mace(3), name="student-view"),
        source,
        AtomisticLabelRequest(energy=False, forces=False, hvp=probe),
    )
    student_actions, _ = _column(student_view, TEACHER_HVP_SLOT)
    term = objective.term(_mace(3), None, batch, scale=1.0, dtype="float64")
    np.testing.assert_allclose(
        float(term.numerator), np.sum((student_actions - actions) ** 2), rtol=1e-9
    )
    assert float(term.support) == 3 * probe.count * 3


def test_mixed_third_order_is_refused_without_admission() -> None:
    probe = AtomisticHessianProbe(count=1, seed=2)
    labeled = _label(
        AtomisticTeacher(_mace(1), name="teacher"),
        _water_source(1),
        AtomisticLabelRequest(energy=False, forces=False, hvp=probe),
    )
    loader = _loader(labeled)
    bindings = {binding.slot: binding for binding in loader.bindings}
    with pytest.raises(ValueError, match="almost-everywhere"):
        AtomisticHessianVectorObjective(probe, scale=1.0).admit(_mace(3), bindings, SCALE)
    AtomisticHessianVectorObjective(probe, policy=AE, scale=1.0).admit(
        _mace(3), bindings, SCALE
    )
    coordinate_only = _Harmonic(1.0, parameter_derivatives=False)
    with pytest.raises(ValueError, match=DERIVATIVE_UNSUPPORTED):
        AtomisticHessianVectorObjective(probe, scale=1.0).admit(
            coordinate_only, bindings, SCALE
        )
    other_probe = AtomisticHessianProbe(count=1, seed=3)
    with pytest.raises(ValueError, match="probe definition"):
        AtomisticHessianVectorObjective(other_probe, policy=AE, scale=1.0).admit(
            _mace(3), bindings, SCALE
        )


# ---------------------------------------------------------------------- Boltzmann


def test_boltzmann_estimator_uses_the_authenticated_biased_generator_density() -> None:
    windows = (1.0, 1.3)
    table, qualification, result, dynamics = _tethered_segment(
        (1.0, 1.0), capacity=24, windows=windows, seed=3
    )
    generator = AtomisticGeneratorState(
        result, 0, qualification=qualification, dynamics=dynamics
    )
    reduced = np.asarray(result.reduced_potentials)
    origin = np.asarray(result.origin_state)
    capacity, _, replicas = reduced.shape
    # Independent draw-major column mapping n = draw * R + replica of row 0.
    expected_columns = tuple(
        draw * replicas + replica
        for draw in range(capacity)
        for replica in range(replicas)
        if origin[draw, replica] == 0
    )
    assert generator.columns == expected_columns
    draws = _draw_configurations(result)
    beta = float(np.asarray(table.beta)[0])
    for column in expected_columns:
        draw, replica = divmod(column, replicas)
        assert generator.reduced_potentials[column] == reduced[draw, 0, replica]
        # The owner's density is u_0 including the frozen umbrella of row 0,
        # recomputed from the column's natively captured configuration.
        radius = float(np.linalg.norm(draws[column, 1] - draws[column, 0]))
        biased = beta * (
            0.5 * TETHER * float(np.sum(draws[column] ** 2))
            + 0.5 * UMBRELLA * (radius - windows[0]) ** 2
        )
        assert generator.reduced_potentials[column] == pytest.approx(
            biased, rel=1e-12, abs=1e-12
        )
    # A biased row is a generator, never a target.
    with pytest.raises(ValueError, match="biased table row"):
        AtomisticBoltzmannGroup.from_table(
            table, 0, composition={1: 2}, generator=generator
        )
    group = AtomisticBoltzmannGroup(generator, beta=beta, composition={1: 2})
    assert group.size == len(expected_columns)
    canonical = atomistic_generator_source(generator)
    assert canonical.record_count == group.size
    sampled, _ = _column(canonical, "positions")
    np.testing.assert_array_equal(
        sampled.reshape((group.size, -1, 3)), draws[list(expected_columns)]
    )
    labeled = _sample_labels(canonical, 1.0)
    source = atomistic_boltzmann_source(
        labeled,
        (group,),
        membership=[0] * group.size,
        columns=list(generator.columns),
        store=_writer(),
    )
    reduced_q = np.asarray(
        [reduced[column // replicas, 0, column % replicas] for column in expected_columns]
    )
    stored = source.read(np.arange(group.size, dtype=np.int64))
    np.testing.assert_array_equal(
        np.asarray(stored.values[BOLTZMANN_GENERATOR_SLOT]).reshape(-1), reduced_q
    )
    squared = np.sum(draws[list(expected_columns)] ** 2, axis=(1, 2))
    for weight in (1.0, 0.0, 0.25):
        objective = AtomisticBoltzmannObjective((group,), forward_weight=weight)
        numerator, support = _boltzmann_estimate(objective, source, 2.0)
        assert support == 1.0
        expected = _discrete_kl(
            beta * 0.5 * squared, beta * 0.5 * 2.0 * squared, reduced_q, weight
        )
        assert numerator == pytest.approx(expected, rel=1e-10)
    # A payload holding only part of a group poisons the update.
    objective = AtomisticBoltzmannObjective((group,), forward_weight=0.5)
    loader = atomistic_training_loader(source, batch_size=group.size // 2, shuffle=False)
    preparation = AtomisticBatchPreparation(
        loader,
        _TWO_ATOMS,
        scale=REDUCED.scale,
        cutoff=1.0,
        fields=FIELDS,
        label_slots=objective.label_slots,
    )
    with loader.epoch(loader.start(0)) as epoch:
        item = next(iter(epoch))
    batch = preparation.prepare_batch(
        item.payload, item.provenance, coordinate_dtype="float64"
    )
    term = objective.term(
        _Harmonic(2.0, scale=REDUCED.scale), None, batch, scale=1.0, dtype="float64"
    )
    assert not np.isfinite(float(term.numerator))
    assert int(term.scientific_failures) > 0


def test_boltzmann_sources_refuse_unproven_or_incomplete_samples() -> None:
    table, qualification, result, dynamics = _tethered_segment(
        (1.0, 1.5), capacity=12, seed=4
    )
    _, foreign, _, foreign_dynamics = _tethered_segment(
        (1.0, 1.5), capacity=2, seed=4, evidence="another-kernel-qualification"
    )
    with pytest.raises(ValueError, match="not produced under this canonical sampling"):
        AtomisticGeneratorState(
            result, 0, qualification=foreign, dynamics=foreign_dynamics
        )
    _, _, _, other_dynamics = _tethered_segment(
        (1.0,), capacity=2, windows=(1.0,), seed=5
    )
    with pytest.raises(ValueError, match="another dynamics runtime"):
        AtomisticGeneratorState(
            result, 0, qualification=qualification, dynamics=other_dynamics
        )
    # Reduced potentials alone never establish which configuration a record is.
    _, plain_qualification, plain, plain_dynamics = _tethered_segment(
        (1.0, 1.5), capacity=4, capture=False, seed=4
    )
    with pytest.raises(ValueError, match="captured draw configurations"):
        AtomisticGeneratorState(
            plain, 0, qualification=plain_qualification, dynamics=plain_dynamics
        )
    generator = AtomisticGeneratorState(
        result, 0, qualification=qualification, dynamics=dynamics
    )
    other_row = AtomisticGeneratorState(
        result, 1, qualification=qualification, dynamics=dynamics
    )
    group = AtomisticBoltzmannGroup.from_table(
        table, 0, composition={1: 2}, generator=generator
    )
    count = group.size
    columns = list(generator.columns)
    canonical = atomistic_generator_source(generator)
    labeled = _sample_labels(canonical, 1.0)

    def attach(source: Any, groups: Any, members: list[int], chosen: list[int]) -> Any:
        return atomistic_boltzmann_source(
            source, groups, membership=members, columns=chosen, store=_writer()
        )

    # A draw of another generating row is not an admitted sample of this one.
    with pytest.raises(ValueError, match="not an admitted sample"):
        attach(labeled, (group,), [0] * count, [other_row.columns[0], *columns[1:]])
    with pytest.raises(ValueError, match="used twice"):
        attach(labeled, (group,), [0] * count, [columns[0], *columns[:-1]])
    # Genuine labels claiming the wrong columns are not those columns' samples.
    with pytest.raises(ValueError, match="captured configuration"):
        attach(labeled, (group,), [0] * count, columns[::-1])
    # A relaxed or otherwise moved frame is not the captured draw it claims.
    moved = np.asarray(canonical.read(np.asarray([0])).values["positions"])
    moved = moved + np.asarray([[0.0, 0.0, 0.0], [1.0e-6, 0.0, 0.0]])
    with pytest.raises(ValueError, match="captured configuration"):
        attach(
            _sample_labels(_replaced(canonical, 0, moved), 1.0),
            (group,),
            [0] * count,
            columns,
        )
    # A group is the complete admitted sample set: no caller-chosen subset.
    subset = AtomisticStructureSource(canonical.schema, _records(canonical)[:-1])
    with pytest.raises(ValueError, match="complete admitted sample set"):
        attach(_sample_labels(subset, 1.0), (group,), [0] * (count - 1), columns[:-1])
    other_chemistry = AtomisticBoltzmannGroup(
        generator, beta=float(np.asarray(table.beta)[0]), composition={8: 2}
    )
    with pytest.raises(ValueError, match="another chemistry"):
        attach(labeled, (other_chemistry,), [0] * count, columns)
    # Adapting (SAMS) draws are never admitted equilibrium samples.
    _, sams_qualification, adaptive, sams_dynamics = _tethered_segment(
        (1.0, 1.5), capacity=12, sams=True, seed=7
    )
    active = np.asarray(adaptive.sample_active).reshape(-1)
    adapting = np.asarray(adaptive.sams_adapting).reshape(-1)
    assert bool(np.any(adapting)) and not bool(np.any(active & adapting))
    row = int(np.asarray(adaptive.origin_state).reshape(-1)[np.flatnonzero(active)[0]])
    sampled = AtomisticGeneratorState(
        adaptive, row, qualification=sams_qualification, dynamics=sams_dynamics
    )
    assert not any(adapting[column] for column in sampled.columns)
    first_adapting = int(np.flatnonzero(adapting)[0])
    assert sampled.reduced_potentials[first_adapting] is None
    sampled_group = AtomisticBoltzmannGroup(sampled, beta=1.0, composition={1: 2})
    with pytest.raises(ValueError, match="not an admitted sample"):
        attach(
            _sample_labels(atomistic_generator_source(sampled), 1.0),
            (sampled_group,),
            [0] * sampled_group.size,
            [first_adapting, *sampled.columns[1:]],
        )
    # Selection evidence lives on the >= 2-row dataset; one row admits every draw.
    _, single_qualification, single, single_dynamics = _tethered_segment(
        (1.0,), capacity=4, seed=9
    )
    with pytest.raises(ValueError, match="at least two rows"):
        AtomisticGeneratorState(
            single,
            0,
            qualification=single_qualification,
            dynamics=single_dynamics,
            selection=FreeEnergySelectionPlan(burn_in=1),
        )
    one_row = AtomisticGeneratorState(
        single, 0, qualification=single_qualification, dynamics=single_dynamics
    )
    assert len(one_row.columns) == 4


def test_generator_mixture_keeps_the_mbar_retained_selection() -> None:
    table, qualification, result, dynamics = _tethered_segment(
        (1.0, 1.4), capacity=160, seed=9
    )
    burn_in = 40
    mixture = AtomisticGeneratorMixture(
        result,
        qualification=qualification,
        dynamics=dynamics,
        selection=FreeEnergySelectionPlan(burn_in=burn_in),
    )
    reduced = np.asarray(result.reduced_potentials)
    origin = np.asarray(result.origin_state).reshape(-1)
    capacity, states, replicas = reduced.shape
    # Without exchange every replica chain stays on its row: the selection
    # drops each chain's first ``burn_in`` draws.
    retained = [
        draw * replicas + replica
        for draw in range(burn_in, capacity)
        for replica in range(replicas)
    ]
    assert mixture.columns == tuple(retained)
    counts = np.bincount(origin[retained], minlength=states)
    assert mixture.retained_counts == tuple(int(value) for value in counts)
    np.testing.assert_allclose(
        mixture.mixture_probabilities, counts / counts.sum(), rtol=0.0, atol=0.0
    )
    values = reduced.transpose((1, 0, 2)).reshape((states, -1))
    log_terms = (
        np.log(counts / counts.sum())[:, None]
        + np.asarray(mixture.free_energies)[:, None]
        - values
    )
    peak = np.max(log_terms, axis=0)
    expected = -(peak + np.log(np.sum(np.exp(log_terms - peak), axis=0)))
    for column in retained:
        assert mixture.reduced_potentials[column] == pytest.approx(
            expected[column], rel=1e-12, abs=1e-12
        )
    excluded = [
        draw * replicas + replica
        for draw in range(burn_in)
        for replica in range(replicas)
    ]
    assert all(mixture.reduced_potentials[column] is None for column in excluded)
    unselected = AtomisticGeneratorMixture(
        result, qualification=qualification, dynamics=dynamics
    )
    assert len(unselected.columns) == capacity * replicas
    assert unselected.selection_id != mixture.selection_id
    assert unselected.generator_id != mixture.generator_id
    group = AtomisticBoltzmannGroup.from_table(
        table, 0, composition={1: 2}, generator=mixture
    )
    assert group.size == len(retained)
    labeled = _sample_labels(atomistic_generator_source(mixture), 1.0)
    # A burn-in column excluded by the analysis cannot enter the objective.
    with pytest.raises(ValueError, match="excluded by the selection"):
        atomistic_boltzmann_source(
            labeled,
            (group,),
            membership=[0] * group.size,
            columns=[excluded[0], *retained[1:]],
            store=_writer(),
        )
    source = atomistic_boltzmann_source(
        labeled,
        (group,),
        membership=[0] * group.size,
        columns=retained,
        store=_writer(),
    )
    stored = source.read(np.arange(group.size, dtype=np.int64))
    np.testing.assert_allclose(
        np.asarray(stored.values[BOLTZMANN_GENERATOR_SLOT]).reshape(-1),
        expected[retained],
        rtol=1e-12,
        atol=1e-12,
    )


# ---------------------------------------------------------------- replay buffer


def _output(
    key: str,
    *,
    outcome: AtomisticStageOutcome = AtomisticStageOutcome.COMPLETED,
    task: str = "walker-0",
    time: float = 1.0,
    energy: float = -1.0,
    revision: str = "policy-0",
    campaign: str = "campaign-0",
) -> AtomisticCampaignOutput:
    ports = {
        "positions": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) + time,
        "momenta": np.zeros((2, 3)),
        "species": np.asarray([1, 1], dtype=np.int32),
        "atom_mask": np.ones((2,), dtype=np.bool_),
        "energy": np.asarray(energy),
        "forces": np.zeros((2, 3)),
        "time": np.asarray(time),
    }
    return AtomisticCampaignOutput(
        semantic_key=key,
        payload_id=f"payload-{key}-{energy}",
        campaign_id=campaign,
        task_id=task,
        material_id="material",
        run_id="run-0",
        replica_id="replica-0",
        group_id=None,
        parent_id=None,
        stage="frame-0",
        stage_entry=1,
        outcome=outcome,
        ports=ports,
        evaluation_key=None,
        numeric_revision_ids=(revision,),
        rng_words=np.zeros((2,), dtype=np.uint32),
        counters={},
        reasons=0,
        campaign_reasons=0,
    )


def test_experience_buffer_fifo_admission_eviction_and_idempotence(
    tmp_path: Path,
) -> None:
    admission = AtomisticExperienceAdmission(maximum_force=10.0)
    buffer = AtomisticExperienceBuffer(tmp_path, capacity=2, admission=admission)
    genesis = buffer.revision.revision_id
    receipts = [
        buffer.publish(_output(f"k{index}", time=float(index)), round_index=0)
        for index in range(3)
    ]
    assert all(receipt is not None for receipt in receipts)
    assert [frame.semantic_key for frame in buffer.revision.frames] == ["k1", "k2"]
    assert [item.semantic_key for item in buffer.revision.evicted] == ["k0"]
    evicted = buffer.receipt("k0")
    assert evicted is not None and evicted.location.endswith("#evicted")
    # Idempotent replay: no duplicate admission, eviction or revision.
    revision = buffer.revision.revision_id
    again = buffer.publish(_output("k0", time=0.0), round_index=1)
    assert again == evicted
    assert buffer.revision.revision_id == revision
    with pytest.raises(ValueError, match="different payload"):
        buffer.publish(_output("k1", time=1.0, energy=-2.0), round_index=1)
    # Admission refusals are durable evidence with receipts.
    failed = buffer.publish(
        _output("k3", outcome=AtomisticStageOutcome.FAILED, time=3.0), round_index=1
    )
    assert failed is not None and failed.location.endswith("#refused")
    nonfinite = buffer.publish(
        _output("k4", energy=float("nan"), time=4.0), round_index=1
    )
    assert nonfinite is not None and nonfinite.location.endswith("#refused")
    assert [item.reason for item in buffer.revision.refused] == [
        "outcome-failed",
        "nonfinite-evaluation",
    ]
    # Durable before return: a reopened buffer is the same revision.
    reopened = AtomisticExperienceBuffer(tmp_path, capacity=2, admission=admission)
    assert reopened.revision.revision_id == buffer.revision.revision_id
    assert reopened.revision.lineage[0] == genesis
    reopened.require_revision(revision, exact=False)
    with pytest.raises(ValueError, match="stale"):
        reopened.require_revision(revision, exact=True)
    with pytest.raises(ValueError, match="another sink"):
        AtomisticExperienceBuffer(tmp_path, capacity=3, admission=admission)
    # Labels attach once per definition.
    label = AtomisticFrameLabel("label-a", {"teacher_energy": np.asarray([1.0])}, {})
    before = reopened.revision.revision_id
    labeled = reopened.attach_labels({"k1": label}, parent=before)
    frame = next(frame for frame in labeled.frames if frame.semantic_key == "k1")
    assert frame.label("label-a") is not None
    with pytest.raises(ValueError, match="already holds"):
        reopened.attach_labels({"k1": label}, parent=labeled.revision_id)
    with pytest.raises(ValueError, match="stale"):
        reopened.attach_labels({"k2": label}, parent=revision)
    # A partial attachment (k2 still unlabeled) is not a complete label successor.
    assert reopened.label_successor(before, "label-a") is None
    other = AtomisticFrameLabel("label-a", {"teacher_energy": np.asarray([2.0])}, {})
    complete = reopened.attach_labels({"k2": other}, parent=labeled.revision_id)
    # The committed attachment authenticates against its parent, durably.
    durable = AtomisticExperienceBuffer(tmp_path, capacity=2, admission=admission)
    successor = durable.label_successor(labeled.revision_id, "label-a")
    assert successor is not None and successor.revision_id == complete.revision_id
    assert durable.label_successor(labeled.revision_id, "label-b") is None
    assert durable.label_successor(before, "label-a") is None
    # Any later change (a publication) is no longer the label successor.
    durable.publish(_output("k5", time=5.0), round_index=2)
    assert durable.label_successor(labeled.revision_id, "label-a") is None


def test_replay_scalar_labels_restore_with_authenticated_array_inventory(
    tmp_path: Path,
) -> None:
    buffer = AtomisticExperienceBuffer(tmp_path, capacity=2)
    buffer.publish(_output("scalar"), round_index=0)
    label = AtomisticFrameLabel(
        "scalar-label",
        {"teacher_energy": np.asarray(-1.25, dtype=np.float64)},
        {"teacher_energy": np.asarray(True, dtype=np.bool_)},
    )
    parent = buffer.revision.revision_id
    committed = buffer.attach_labels({"scalar": label}, parent=parent)
    restored = AtomisticExperienceBuffer(tmp_path, capacity=2)
    assert restored.revision.revision_id == committed.revision_id
    successor = restored.label_successor(parent, label.label_id)
    assert successor is not None and successor.revision_id == committed.revision_id
    recovered = restored.revision.frames[0].label(label.label_id)
    assert recovered is not None
    np.testing.assert_array_equal(
        recovered.values["teacher_energy"], label.values["teacher_energy"]
    )
    np.testing.assert_array_equal(
        recovered.available["teacher_energy"], label.available["teacher_energy"]
    )
    assert recovered.values["teacher_energy"].shape == ()
    assert recovered.available["teacher_energy"].shape == ()


def test_replay_archive_refuses_foreign_application_metadata(tmp_path: Path) -> None:
    buffer = AtomisticExperienceBuffer(tmp_path, capacity=2)
    buffer.publish(_output("canonical"), round_index=0)
    manifest, arrays = read_array_archive(buffer._path)
    application = {name: value for name, value in manifest.items() if name != "arrays"}
    application["foreign_application_field"] = True
    write_array_archive(buffer._path, manifest=application, arrays=arrays)
    with pytest.raises(ValueError):
        AtomisticExperienceBuffer(tmp_path, capacity=2)


def test_replay_draw_probabilities_follow_policy_age() -> None:
    draw = AtomisticReplayDraw((3, 1, 0))
    assert [draw.repeats(age) for age in range(5)] == [3, 1, 0, 0, 0]
    with pytest.raises(ValueError):
        AtomisticReplayDraw((0, 0))


# ------------------------------------------------------------ on-policy controller


def _units() -> AtomisticUnitSystem:
    return AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()


def _system() -> Any:
    return AtomisticSystemPlan(
        np.asarray([0, 1, 2]),
        np.asarray([8, 1, 1]),
        np.asarray([15.999, 1.008, 1.008]),
        _units(),
    ).prepare()


def _bundle(model: Any) -> Any:
    return neighborhood_bundle_plan(
        LearnedGraphPotentialTerm(model, name="student").requirements.neighborhoods,
        search=AtomisticCellListImageSearch(maximum_candidate_slots=4096),
        skin=0.5,
        search_capacities=(search_capacity(12, particles_per_cell=3, images=27),),
        role_capacities=(role_capacity(8, 4),),
        frame=PeriodicCell(
            20.0 * np.eye(3), origin=np.full((3,), -10.0), periodic_axes=(False,) * 3
        ),
    )


def _rollout(
    student: Any,
    *,
    frames: int = 2,
    cadence: int = 2,
    integrator: Any = None,
) -> AtomisticDistillationRollout:
    system = _system()
    return AtomisticDistillationRollout(
        system,
        AtomisticPotentialProgram([LearnedGraphPotentialTerm(student, name="student")]),
        term="student",
        neighborhood_bundle=_bundle(student),
        graph_execution=AtomisticGraphExecutionPlan(4, backend="particle"),
        integrator=BAOABLangevinPlan(0.25, 0.01) if integrator is None else integrator,
        states=(
            AtomisticThermodynamicStatePlan(
                AtomisticPhaseSpaceMeasurePlan(system), ensemble="nvt", temperature=300.0
            ),
        ),
        walkers=tuple(
            AtomisticTaskPacket(
                positions=_WATER + 0.02 * index,
                momenta=np.zeros((3, 3)),
                species=np.asarray([8, 1, 1], dtype=np.int32),
                atom_mask=np.ones((3,), dtype=np.bool_),
            )
            for index in range(2)
        ),
        cadence=cadence,
        frames=frames,
        steps_per_round=2,
    )


def _plan(
    *,
    maximum_loss: float = 1.0e6,
    drive: AtomisticDistillationDrive = "promoted",
    capacity: int = 8,
    frames: int = 2,
    qualification_teacher: Any = None,
) -> AtomisticOnPolicyDistillationPlan:
    student = _mace(3)
    teacher = AtomisticTeacher(_mace(1), name="teacher")
    request = AtomisticLabelRequest()
    rollout = _rollout(student, frames=frames)
    labeler = teacher if qualification_teacher is None else qualification_teacher
    held_out = _water_source(2, seed=21, prefix="qualify")
    qualification = AtomisticDistillationQualification(
        _label(labeler, held_out, request),
        held_out,
        maximum_loss=maximum_loss,
    )
    return AtomisticOnPolicyDistillationPlan(
        teacher=teacher,
        request=request,
        student=student,
        rollout=rollout,
        objectives=(
            AtomisticEnergyObjective(TEACHER_ENERGY_SLOT, name="energy", scale=1.0),
            AtomisticForceObjective(TEACHER_FORCES_SLOT, name="forces", scale=1.0),
        ),
        policy=AtomisticTrainingPolicy(maximum_steps=2, learning_rate=1e-3),
        graph_execution=EXECUTION,
        qualification=qualification,
        cutoff=CUTOFF,
        capacity=capacity,
        drive=drive,
        replay=AtomisticReplayDraw((2, 1)),
        seed=17,
    )


def _signature(state: AtomisticOnPolicyDistillationState) -> tuple[Any, ...]:
    return (
        state.segment,
        state.phase,
        state.replay_revision,
        state.labeled_frames,
        atomistic_model_revision(state.promoted).revision_id,
        atomistic_model_revision(state.driving).revision_id,
        state.publications,
        state.state_id,
    )


def test_every_phase_boundary_resumes_to_the_uninterrupted_segment(
    tmp_path: Path,
) -> None:
    plan = _plan()
    reference = run_atomistic_distillation_segment(
        plan,
        AtomisticOnPolicyDistillationState.initial(plan),
        directory=tmp_path / "reference",
    )
    assert reference.successful
    assert reference.report.phases == ("rollout", "label", "train", "qualify", "promote")
    assert reference.state.segment == 1 and reference.state.phase == "rollout"
    for boundary in ("rollout", "label", "train", "qualify"):
        directory = tmp_path / boundary
        interrupted = run_atomistic_distillation_segment(
            plan,
            AtomisticOnPolicyDistillationState.initial(plan),
            directory=directory,
            stop_after=boundary,
        )
        assert interrupted.successful and interrupted.report.phases[-1] == boundary
        # Fresh plan objects stand in for a fresh process.
        restored = read_atomistic_distillation_restart(directory, _plan())
        assert restored.state_id == interrupted.state.state_id
        resumed = run_atomistic_distillation_segment(
            _plan(), restored, directory=directory
        )
        assert resumed.successful
        assert _signature(resumed.state) == _signature(reference.state)


def test_labels_are_never_duplicated_and_replay_is_bounded(tmp_path: Path) -> None:
    plan = _plan(capacity=3)
    state = AtomisticOnPolicyDistillationState.initial(plan)
    for _ in range(2):
        result = run_atomistic_distillation_segment(plan, state, directory=tmp_path)
        assert result.successful
        state = result.state
    buffer = plan.buffer(tmp_path)
    frames = buffer.revision.frames
    assert len(frames) == 3
    assert len(buffer.revision.evicted) == 2 * 2 * 2 - 3
    for frame in frames:
        assert sum(label.label_id == plan.label_id for label in frame.labels) == 1
    assert state.labeled_frames == 3
    assert len(state.policy_ids) >= 1
    # A replayed LABEL transition over an already labeled revision adds nothing.
    relabel = run_atomistic_distillation_segment(
        plan, state.replace(phase="label"), directory=tmp_path, stop_after="label"
    )
    assert relabel.report.frames_labeled == 0


def test_rejected_promotion_keeps_the_promoted_student(tmp_path: Path) -> None:
    plan = _plan(maximum_loss=-1.0)
    initial = AtomisticOnPolicyDistillationState.initial(plan)
    result = run_atomistic_distillation_segment(plan, initial, directory=tmp_path)
    assert result.successful and not result.report.promoted
    state = result.state
    assert atomistic_model_revision(state.promoted).revision_id == (
        atomistic_model_revision(plan.student).revision_id
    )
    assert atomistic_model_revision(state.driving).revision_id == (
        atomistic_model_revision(plan.student).revision_id
    )
    assert state.publications == ()
    assert state.candidate is not None
    assert not (tmp_path / "publications").exists()
    # Declared drive policy: a candidate drives before promotion.
    candidate_plan = _plan(maximum_loss=-1.0, drive="candidate")
    driven = run_atomistic_distillation_segment(
        candidate_plan,
        AtomisticOnPolicyDistillationState.initial(candidate_plan),
        directory=tmp_path / "candidate",
    )
    assert driven.state.candidate is not None
    assert atomistic_model_revision(driven.state.driving).revision_id == (
        atomistic_model_revision(driven.state.candidate).revision_id
    )


def test_stale_replay_revision_refuses_restart(tmp_path: Path) -> None:
    plan = _plan()
    result = run_atomistic_distillation_segment(
        plan,
        AtomisticOnPolicyDistillationState.initial(plan),
        directory=tmp_path,
        stop_after="label",
    )
    assert result.successful
    buffer = plan.buffer(tmp_path)
    buffer.publish(_output("foreign-frame", revision="foreign-policy"), round_index=0)
    with pytest.raises(ValueError, match="stale"):
        read_atomistic_distillation_restart(tmp_path, _plan())
    with pytest.raises(ValueError, match="stale"):
        run_atomistic_distillation_segment(plan, result.state, directory=tmp_path)
    other = _plan(capacity=5)
    with pytest.raises(ValueError, match="another plan|not canonical"):
        read_atomistic_distillation_restart(tmp_path, other)


def test_label_attachment_committed_before_its_phase_checkpoint_resumes(
    tmp_path: Path,
) -> None:
    plan = _plan()
    initial = AtomisticOnPolicyDistillationState.initial(plan)
    reference = run_atomistic_distillation_segment(
        plan, initial, directory=tmp_path / "reference"
    )
    assert reference.successful
    directory = tmp_path / "crash"
    rolled = run_atomistic_distillation_segment(
        plan, initial, directory=directory, stop_after="rollout"
    )
    assert rolled.successful and rolled.state.phase == "label"
    saved = tmp_path / "label-phase-restart"
    shutil.copytree(directory / "restart", saved)
    labeled = run_atomistic_distillation_segment(
        plan, rolled.state, directory=directory, stop_after="label"
    )
    assert labeled.successful and labeled.report.frames_labeled > 0
    # Crash after the durable label attachment but before the TRAIN restart:
    # the LABEL restart (committed replay revision R0) is the last checkpoint
    # while the buffer already holds its label successor R1.
    shutil.rmtree(directory / "restart")
    shutil.copytree(saved, directory / "restart")
    restored = read_atomistic_distillation_restart(directory, _plan())
    assert restored.phase == "label"
    assert restored.state_id == rolled.state.state_id
    buffer = plan.buffer(directory)
    assert buffer.revision.revision_id != restored.replay_revision
    assert buffer.label_successor(restored.replay_revision, plan.label_id) is not None
    resumed = run_atomistic_distillation_segment(_plan(), restored, directory=directory)
    assert resumed.successful
    assert resumed.report.frames_labeled == labeled.report.frames_labeled
    assert _signature(resumed.state) == _signature(reference.state)

    # Any other replay change after the attachment is stale, never a successor.
    other = tmp_path / "foreign"
    rolled = run_atomistic_distillation_segment(
        plan, initial, directory=other, stop_after="rollout"
    )
    shutil.copytree(other / "restart", tmp_path / "foreign-restart")
    run_atomistic_distillation_segment(
        plan, rolled.state, directory=other, stop_after="label"
    )
    shutil.rmtree(other / "restart")
    shutil.copytree(tmp_path / "foreign-restart", other / "restart")
    plan.buffer(other).publish(
        _output("foreign-frame", revision="foreign-policy"), round_index=0
    )
    with pytest.raises(ValueError, match="stale"):
        read_atomistic_distillation_restart(other, _plan())
    with pytest.raises(ValueError, match="stale"):
        run_atomistic_distillation_segment(plan, rolled.state, directory=other)


def test_qualification_labels_must_be_the_plans_teacher_labels() -> None:
    # Same schema, another numerical teacher revision: refused, not admitted.
    with pytest.raises(ValueError, match="stale"):
        _plan(qualification_teacher=AtomisticTeacher(_mace(2), name="teacher"))
    # Same weights under another teacher name is another label definition.
    with pytest.raises(ValueError, match="stale"):
        _plan(qualification_teacher=AtomisticTeacher(_mace(1), name="other-teacher"))


def _final_walkers(rollout: AtomisticDistillationRollout, root: Path) -> Any:
    model = rollout.learned_term().potential
    assert isinstance(model, AbstractAtomisticPotential)
    operation = rollout.operation(model)
    prepared = rollout.campaign(operation, rollout.walkers, "cadence-control").prepare()
    root.mkdir(parents=True, exist_ok=True)
    store = AtomisticCampaignStore(root / "control.campaign")
    sink = AtomisticExperienceBuffer(root / "sink", capacity=16)
    state = prepared.run(prepared.start(jr.key(4)), sink, store, maximum_rounds=64)
    assert state.complete
    return prepared.terminal_packets(state)


def test_observation_cadence_does_not_change_walker_dynamics(tmp_path: Path) -> None:
    # A Nose-Hoover chain carries controller state: two observed frames of two
    # steps must continue exactly like one unobserved stage of four steps.
    student = _mace(3)
    integrator = NoseHooverChainPlan(0.25, NoseHooverChainParameters(20.0))
    observed = _final_walkers(
        _rollout(student, frames=2, cadence=2, integrator=integrator), tmp_path / "two"
    )
    single = _final_walkers(
        _rollout(student, frames=1, cadence=4, integrator=integrator), tmp_path / "one"
    )
    assert set(observed) == set(single) == {"walker-0", "walker-1"}
    for task in observed:
        left = observed[task].continuation.state
        right = single[task].continuation.state
        assert int(np.asarray(left.step_index)) == int(np.asarray(right.step_index)) == 4
        assert float(np.asarray(left.time)) == float(np.asarray(right.time))
        for name in ("positions", "momenta"):
            np.testing.assert_allclose(
                np.asarray(getattr(left.kinematics, name)),
                np.asarray(getattr(right.kinematics, name)),
                rtol=1e-12,
                atol=1e-12,
            )
        chains = _leaves(left.controllers)
        assert any(np.any(leaf != 0) for leaf in chains)
        for a, b in zip(chains, _leaves(right.controllers), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        assert bool(eqx.tree_equal(left.random_key, right.random_key))


def test_promotion_rebinds_complete_walker_continuations(tmp_path: Path) -> None:
    plan = _plan()
    qualified = run_atomistic_distillation_segment(
        plan,
        AtomisticOnPolicyDistillationState.initial(plan),
        directory=tmp_path,
        stop_after="qualify",
    )
    assert qualified.successful
    before = qualified.state.walkers
    previous = plan.rollout.operation(qualified.state.driving)
    assert all(
        walker.continuation.source_binding_id == previous.binding_id for walker in before
    )
    promoted = run_atomistic_distillation_segment(
        plan, qualified.state, directory=tmp_path
    )
    assert promoted.successful and promoted.report.promoted
    state = promoted.state
    operation = plan.rollout.operation(state.driving)
    assert operation.binding_id != previous.binding_id
    for old, new in zip(before, state.walkers, strict=True):
        assert new.continuation.source_binding_id == operation.binding_id
        assert len(new.continuation.lineage) == len(old.continuation.lineage) + 1
        source, target = old.continuation.state, new.continuation.state
        # The owner transition keeps the physical and stochastic continuation.
        for name in ("positions", "momenta"):
            np.testing.assert_array_equal(
                np.asarray(getattr(target.kinematics, name)),
                np.asarray(getattr(source.kinematics, name)),
            )
        assert float(np.asarray(target.time)) == float(np.asarray(source.time))
        assert int(np.asarray(target.step_index)) == int(np.asarray(source.step_index))
        assert bool(eqx.tree_equal(target.random_key, source.random_key))
        for a, b in zip(
            _leaves(target.controllers), _leaves(source.controllers), strict=True
        ):
            np.testing.assert_array_equal(a, b)
    # The rebound continuation is the committed restart, rebuilt in a fresh plan.
    restored = read_atomistic_distillation_restart(tmp_path, _plan())
    assert restored.state_id == state.state_id
    # The next segment continues from it: its walkers advance from that time.
    following = run_atomistic_distillation_segment(
        plan, state, directory=tmp_path, stop_after="rollout"
    )
    assert following.successful
    for old, new in zip(state.walkers, following.state.walkers, strict=True):
        assert (
            int(np.asarray(new.continuation.state.step_index))
            == int(np.asarray(old.continuation.state.step_index)) + 2 * 2
        )
