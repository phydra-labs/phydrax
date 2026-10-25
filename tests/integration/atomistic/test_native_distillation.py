#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""End-to-end native distillation.

Offline: an immutable structure store is labeled by a native teacher in
bounded chunks published through the canonical store writer, a student is
fitted on the committed labels and its training continuation is restored in a
fresh process. On-policy: native dynamics of the student publish frames into
the bounded experience buffer, the teacher labels them, the student trains,
qualifies and is promoted (its walkers' complete dynamics continuations are
rebound by the owner's Hamiltonian transition), and the walker, replay and
student state resume in a fresh process. The random teacher demonstrates
mechanics only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax.random as jr
import numpy as np
import pytest

from phydrax.atomistic import (
    AtomisticBatch,
    AtomisticCellListImageSearch,
    AtomisticGraphExecutionPlan,
    AtomisticScaleContract,
)
from phydrax.atomistic._campaign import AtomisticTaskPacket
from phydrax.atomistic._checkpoint import (
    read_atomistic_training_restart,
    write_atomistic_training_restart,
)
from phydrax.atomistic._distillation import (
    AtomisticDistillationQualification,
    AtomisticDistillationRollout,
    AtomisticLabelRequest,
    AtomisticOnPolicyDistillationPlan,
    AtomisticOnPolicyDistillationState,
    AtomisticReplayDraw,
    AtomisticTeacher,
    label_atomistic_structures,
    read_atomistic_distillation_restart,
    require_atomistic_labels,
    run_atomistic_distillation_segment,
    TEACHER_ENERGY_SLOT,
    TEACHER_FORCES_SLOT,
)
from phydrax.atomistic._model_artifact import read_atomistic_model_artifact
from phydrax.atomistic._potential import atomistic_model_revision
from phydrax.atomistic._potential_program import (
    AtomisticPotentialProgram,
    LearnedGraphPotentialTerm,
)
from phydrax.atomistic._structure_store import (
    AtomisticStoreLayout,
    AtomisticStoreLimits,
    AtomisticStructureStoreWriter,
    open_atomistic_structure_store,
)
from phydrax.atomistic._system import AtomisticSystemPlan
from phydrax.atomistic._thermal import BAOABLangevinPlan
from phydrax.atomistic._thermodynamic import (
    AtomisticPhaseSpaceMeasurePlan,
    AtomisticThermodynamicStatePlan,
)
from phydrax.atomistic._training import (
    atomistic_training_loader,
    atomistic_training_source,
    AtomisticTrainingFields,
    AtomisticTrainingPolicy,
    AtomisticTrainingProblem,
    fit_atomistic_potential,
)
from phydrax.atomistic._training_objectives import (
    AtomisticEnergyObjective,
    AtomisticForceObjective,
)
from phydrax.atomistic._units import AtomisticUnitSystem
from phydrax.discretization import PeriodicCell
from phydrax.nn.atomistic import MACEArchitecture, MACEPotential
from phydrax.units import ANGSTROM, DALTON, ELECTRONVOLT
from tests._support.atomistic_bundles import (
    neighborhood_bundle_plan,
    role_capacity,
    search_capacity,
)


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
CUTOFF = 3.0
EXECUTION = AtomisticGraphExecutionPlan(4, maximum_dense_atoms=3)
WATER = np.asarray([[0.0, 0.0, 0.0], [0.95, 0.1, 0.0], [-0.25, 0.9, 0.15]])
REPOSITORY = Path(__file__).resolve().parents[3]


def _mace(seed: int) -> MACEPotential:
    return MACEPotential(
        SCALE,
        MACEArchitecture(
            species=(1, 8),
            cutoff=CUTOFF,
            radial_basis_count=4,
            cutoff_power=5,
            channel_count=3,
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


def _source(count: int, seed: int, prefix: str) -> Any:
    rng = np.random.default_rng(seed)
    positions = WATER[None] + 0.05 * rng.normal(size=(count, 3, 3))
    batch = AtomisticBatch(
        np.tile(np.asarray([8, 1, 1], dtype=np.int32), (count, 1)),
        positions,
        np.tile(np.asarray([15.999, 1.008, 1.008]), (count, 1)),
        SCALE,
        structure_ids=tuple(f"{prefix}-{index}" for index in range(count)),
    )
    return atomistic_training_source(batch, mass_unit=DALTON)


def _store_limits() -> AtomisticStoreLimits:
    return AtomisticStoreLimits(
        max_commit_bytes=1 << 20,
        max_member_bytes=1 << 20,
        max_shard_bytes=1 << 24,
        max_shard_members=8192,
        max_chunk_bytes=1 << 16,
        max_index_bytes=1 << 20,
        max_decoded_bytes=1 << 22,
        max_read_bytes=1 << 22,
        max_records=1000,
    )


def _writer(root: Path) -> AtomisticStructureStoreWriter:
    return AtomisticStructureStoreWriter(
        root,
        layout=AtomisticStoreLayout(records_per_shard=2, target_chunk_bytes=4096),
        limits=_store_limits(),
    )


def _teacher() -> AtomisticTeacher:
    return AtomisticTeacher(_mace(1), name="random-teacher")


def _offline_problem(labeled: Any) -> AtomisticTrainingProblem:
    return AtomisticTrainingProblem(
        atomistic_training_loader(labeled, batch_size=2, shuffle=False),
        EXECUTION,
        (
            AtomisticEnergyObjective(TEACHER_ENERGY_SLOT, name="energy", scale=1.0),
            AtomisticForceObjective(TEACHER_FORCES_SLOT, name="forces", scale=1.0),
        ),
        scale=SCALE,
        cutoff=CUTOFF,
    )


def _offline_policy(steps: int) -> AtomisticTrainingPolicy:
    return AtomisticTrainingPolicy(maximum_steps=steps, learning_rate=1e-3)


def resume_offline_in_fresh_process(store: str, bundle: str) -> dict[str, Any]:
    """Fresh-process continuation: rebuild the problem from the store and resume."""
    labeled = open_atomistic_structure_store(Path(store), limits=_store_limits())
    problem = _offline_problem(labeled)
    artifact, result = read_atomistic_training_restart(
        Path(bundle), problem, _offline_policy(4)
    )
    resumed = fit_atomistic_potential(
        artifact.model, problem, _offline_policy(4), continuation=result
    )
    return {
        "result_id": result.result_id,
        "updates": int(resumed.training_values.shape[0]),
        "revision": atomistic_model_revision(resumed.potential).revision_id,
    }


def _fresh(function: str, *arguments: str) -> dict[str, Any]:
    script = (
        "import json, sys\n"
        "from tests.integration.atomistic.test_native_distillation import "
        f"{function}\n"
        f"print(json.dumps({function}(*sys.argv[1:])))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, *arguments],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY)},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_offline_store_labeling_student_fit_and_fresh_process_restart(
    tmp_path: Path,
) -> None:
    pytest.importorskip("zarr")
    raw_root = tmp_path / "raw-store"
    writer = _writer(raw_root)
    raw = _source(6, 3, "raw")
    writer.append(writer.create(raw.schema, name="raw"), raw)
    stored = open_atomistic_structure_store(raw_root, limits=_store_limits())
    teacher = _teacher()
    request = AtomisticLabelRequest()
    label_root = tmp_path / "label-store"
    published = label_atomistic_structures(
        teacher,
        atomistic_training_loader(stored, batch_size=4, shuffle=False),
        AtomisticTrainingFields(),
        request,
        graph_execution=EXECUTION,
        cutoff=CUTOFF,
        store=_writer(label_root),
    )
    require_atomistic_labels(published, teacher, request, raw)
    reopened = open_atomistic_structure_store(label_root, limits=_store_limits())
    assert reopened.source_id == published.source_id
    assert reopened.commit.commit_id == published.commit.commit_id

    problem = _offline_problem(published)
    student = _mace(3)
    teacher_model = teacher.model
    assert isinstance(teacher_model, MACEPotential)
    teacher_before = np.array(teacher_model.embedding, copy=True)
    partial = fit_atomistic_potential(student, problem, _offline_policy(2), key=jr.key(2))
    assert bool(partial.successful)
    np.testing.assert_array_equal(np.asarray(teacher_model.embedding), teacher_before)
    bundle = write_atomistic_training_restart(
        tmp_path / "student", partial, problem, _offline_policy(2)
    )
    fresh = _fresh("resume_offline_in_fresh_process", str(label_root), str(bundle))
    assert fresh["result_id"] == partial.result_id
    assert fresh["updates"] == 4
    direct = fit_atomistic_potential(
        partial.potential, problem, _offline_policy(4), continuation=partial
    )
    assert fresh["revision"] == atomistic_model_revision(direct.potential).revision_id


# ------------------------------------------------------------------- on-policy


def _system() -> Any:
    return AtomisticSystemPlan(
        np.asarray([0, 1, 2]),
        np.asarray([8, 1, 1]),
        np.asarray([15.999, 1.008, 1.008]),
        AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
    ).prepare()


def on_policy_plan(root: str | Path) -> AtomisticOnPolicyDistillationPlan:
    """The on-policy plan, rebuilt identically in every process.

    The qualification labels live in one committed store under ``root``; a
    rebuilt plan reopens the complete store unchanged.
    """
    system = _system()
    student = _mace(3)
    teacher = _teacher()
    request = AtomisticLabelRequest()
    bundle = neighborhood_bundle_plan(
        LearnedGraphPotentialTerm(student, name="student").requirements.neighborhoods,
        search=AtomisticCellListImageSearch(maximum_candidate_slots=4096),
        skin=0.5,
        search_capacities=(search_capacity(12, particles_per_cell=3, images=27),),
        role_capacities=(role_capacity(8, 4),),
        frame=PeriodicCell(
            20.0 * np.eye(3), origin=np.full((3,), -10.0), periodic_axes=(False,) * 3
        ),
    )
    rollout = AtomisticDistillationRollout(
        system,
        AtomisticPotentialProgram([LearnedGraphPotentialTerm(student, name="student")]),
        term="student",
        neighborhood_bundle=bundle,
        graph_execution=AtomisticGraphExecutionPlan(4, backend="particle"),
        integrator=BAOABLangevinPlan(0.25, 0.01),
        states=(
            AtomisticThermodynamicStatePlan(
                AtomisticPhaseSpaceMeasurePlan(system), ensemble="nvt", temperature=300.0
            ),
        ),
        walkers=tuple(
            AtomisticTaskPacket(
                positions=WATER + 0.03 * index,
                momenta=np.zeros((3, 3)),
                species=np.asarray([8, 1, 1], dtype=np.int32),
                atom_mask=np.ones((3,), dtype=np.bool_),
            )
            for index in range(2)
        ),
        cadence=3,
        frames=2,
        steps_per_round=3,
    )
    held_out = _source(2, 31, "qualify")
    qualification = AtomisticDistillationQualification(
        label_atomistic_structures(
            teacher,
            atomistic_training_loader(held_out, shuffle=False),
            AtomisticTrainingFields(),
            request,
            graph_execution=EXECUTION,
            cutoff=CUTOFF,
            store=_writer(Path(root) / "qualification-labels"),
        ),
        held_out,
        maximum_loss=1.0e6,
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
        capacity=6,
        batch_size=4,
        replay=AtomisticReplayDraw((2, 1)),
        seed=23,
    )


def resume_on_policy_in_fresh_process(root: str, directory: str) -> dict[str, Any]:
    plan = on_policy_plan(root)
    state = read_atomistic_distillation_restart(Path(directory), plan)
    result = run_atomistic_distillation_segment(plan, state, directory=Path(directory))
    return {
        "restored": state.state_id,
        "state": result.state.state_id,
        "segment": result.state.segment,
        "promoted": atomistic_model_revision(result.state.promoted).revision_id,
    }


def test_on_policy_dynamics_teacher_buffer_train_qualify_promote_and_resume(
    tmp_path: Path,
) -> None:
    pytest.importorskip("zarr")
    plan = on_policy_plan(tmp_path)
    initial = AtomisticOnPolicyDistillationState.initial(plan)
    first = run_atomistic_distillation_segment(plan, initial, directory=tmp_path / "run")
    assert first.successful and first.report.promoted
    state = first.state
    assert state.segment == 1
    assert len(state.publications) == 1
    artifact = read_atomistic_model_artifact(
        tmp_path / "run" / "publications" / f"{state.publications[0]}.phydrax",
        numeric_revision_id=state.publications[0],
    )
    assert atomistic_model_revision(artifact.model).revision_id == (
        atomistic_model_revision(state.promoted).revision_id
    )
    # Walkers continue from their complete final dynamics state (two frames of
    # three steps), rebound to the promoted student's operation.
    operation = plan.rollout.operation(state.driving)
    for walker, initial_walker in zip(state.walkers, initial.walkers, strict=True):
        assert float(walker.time) == pytest.approx(float(initial_walker.time) + 6 * 0.25)
        assert walker.momenta is not None and bool(np.any(walker.momenta != 0.0))
        assert walker.continuation is not None
        assert walker.continuation.source_binding_id == operation.binding_id
        assert int(np.asarray(walker.continuation.state.step_index)) == 6
        assert len(walker.continuation.lineage) == 1
    buffer = plan.buffer(tmp_path / "run")
    assert buffer.revision.revision_id == state.replay_revision
    assert len(buffer.revision.frames) == 4
    assert all(frame.label(plan.label_id) is not None for frame in buffer.revision.frames)
    assert first.report.replay_records == 4
    assert first.report.phase_seconds and all(
        value >= 0.0 for value in first.report.phase_seconds
    )

    # Interrupt the next segment after its training commit, then resume in a
    # fresh process from the restart, the replay buffer and the campaign store.
    interrupted = run_atomistic_distillation_segment(
        plan, state, directory=tmp_path / "run", stop_after="train"
    )
    assert interrupted.successful and interrupted.state.phase == "qualify"
    assert len(plan.buffer(tmp_path / "run").revision.evicted) == 2
    fresh = _fresh(
        "resume_on_policy_in_fresh_process", str(tmp_path), str(tmp_path / "run")
    )
    assert fresh["restored"] == interrupted.state.state_id
    assert fresh["segment"] == 2

    # The same transitions without interruption reach the same committed state.
    reference = tmp_path / "reference"
    uninterrupted = run_atomistic_distillation_segment(plan, initial, directory=reference)
    uninterrupted = run_atomistic_distillation_segment(
        plan, uninterrupted.state, directory=reference
    )
    assert fresh["state"] == uninterrupted.state.state_id
    assert (
        fresh["promoted"]
        == atomistic_model_revision(uninterrupted.state.promoted).revision_id
    )
