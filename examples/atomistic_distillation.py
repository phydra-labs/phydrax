"""Native offline and on-policy distillation of a small MACE student.

The teacher is a randomly initialized native MACE model: the example shows the
mechanics (frozen teacher, typed teacher labels streamed into a bounded
structure store, replay, the committed ROLLOUT -> LABEL -> TRAIN -> QUALIFY ->
PROMOTE transaction with complete walker continuation, and restart), never
pretrained or chemical accuracy. Labeling needs the ``atomistic-data`` extra.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


scale = phx.atomistic.AtomisticScaleContract(phx.units.ANGSTROM, phx.units.ELECTRONVOLT)
cutoff = 3.0
water = np.asarray([[0.0, 0.0, 0.0], [0.95, 0.1, 0.0], [-0.25, 0.9, 0.15]])


def mace(seed: int) -> phx.nn.atomistic.MACEPotential:
    return phx.nn.atomistic.MACEPotential(
        scale,
        phx.nn.atomistic.MACEArchitecture(
            species=(1, 8),
            cutoff=cutoff,
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
        atomic_energies=np.asarray([[-1.0, -2.0]]),
        energy_scale=np.asarray([1.3]),
        energy_shift=np.asarray([0.2]),
        key=jax.random.key(seed),
    )


def structures(
    count: int, seed: int, prefix: str
) -> phx.atomistic.AtomisticStructureSource:
    positions = water[None] + 0.05 * np.random.default_rng(seed).normal(
        size=(count, 3, 3)
    )
    batch = phx.atomistic.AtomisticBatch(
        np.tile(np.asarray([8, 1, 1], dtype=np.int32), (count, 1)),
        positions,
        np.tile(np.asarray([15.999, 1.008, 1.008]), (count, 1)),
        scale,
        structure_ids=tuple(f"{prefix}-{index}" for index in range(count)),
    )
    return phx.atomistic.atomistic_training_source(batch, mass_unit=phx.units.DALTON)


graph = phx.atomistic.AtomisticGraphExecutionPlan(4, maximum_dense_atoms=3)
workspace = tempfile.TemporaryDirectory()
root = Path(workspace.name)


def label_store(name: str) -> phx.atomistic.AtomisticStructureStoreWriter:
    """Bounded canonical store of one labeled source (two records per chunk)."""
    return phx.atomistic.AtomisticStructureStoreWriter(
        root / name,
        layout=phx.atomistic.AtomisticStoreLayout(
            records_per_shard=2, target_chunk_bytes=1 << 16
        ),
        limits=phx.atomistic.AtomisticStoreLimits(
            max_commit_bytes=1 << 20,
            max_member_bytes=1 << 24,
            max_shard_bytes=1 << 26,
            max_shard_members=1 << 14,
            max_chunk_bytes=1 << 20,
            max_index_bytes=1 << 22,
            max_decoded_bytes=1 << 26,
            max_read_bytes=1 << 26,
            max_records=1 << 16,
        ),
    )


fields = phx.atomistic.AtomisticTrainingFields()
teacher = phx.atomistic.AtomisticTeacher(mace(1), name="random-teacher")
probe = phx.atomistic.AtomisticHessianProbe("rademacher", count=1, seed=7)
request = phx.atomistic.AtomisticLabelRequest(hvp=probe)
objectives = (
    phx.atomistic.AtomisticEnergyObjective("teacher_energy", name="energy", scale=1.0),
    phx.atomistic.AtomisticForceObjective("teacher_forces", name="forces", scale=1.0),
    phx.atomistic.AtomisticHessianVectorObjective(
        probe,
        policy=phx.RegularityPolicy(allow_almost_everywhere=True),
        scale=1.0,
        schedule=phx.atomistic.AtomisticObjectiveSchedule.constant(0.1),
    ),
)

# Offline: native teacher labels (energies, forces, stored HVP probes/actions),
# streamed batch by batch into a committed, resumable structure store.
labeled = phx.atomistic.label_atomistic_structures(
    teacher,
    phx.atomistic.atomistic_training_loader(
        structures(6, 3, "offline"), batch_size=2, shuffle=False
    ),
    fields,
    request,
    graph_execution=graph,
    cutoff=cutoff,
    store=label_store("offline-labels"),
)
problem = phx.atomistic.AtomisticTrainingProblem(
    phx.atomistic.atomistic_training_loader(labeled, batch_size=3),
    graph,
    objectives,
    scale=scale,
    cutoff=cutoff,
)
offline = phx.atomistic.fit_atomistic_potential(
    mace(3),
    problem,
    phx.atomistic.AtomisticTrainingPolicy(maximum_steps=4, learning_rate=1e-3),
    key=jax.random.key(0),
)
if not bool(offline.successful):
    raise RuntimeError("Offline distillation did not complete.")
print("offline", offline.result_id, float(offline.final_loss))

# On-policy: the student's native dynamics generate frames for the teacher.
units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
system = phx.atomistic.AtomisticSystemPlan(
    np.asarray([0, 1, 2]),
    np.asarray([8, 1, 1]),
    np.asarray([15.999, 1.008, 1.008]),
    units,
).prepare()
student = offline.best_potential
requirements = phx.atomistic.LearnedGraphPotentialTerm(
    student, name="student"
).requirements.neighborhoods
resources = phx.atomistic.AtomisticNeighborhoodBundlePlan(
    requirements,
    (
        phx.atomistic.AtomisticNeighborhoodSource(
            "learned-graph",
            phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS,
            requirements,
            phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
            phx.discretization.ParticleImageCapacityLadder(
                (
                    phx.discretization.ParticleImageCapacity(
                        maximum_particles_per_cell=3,
                        maximum_edges=12,
                        maximum_degree=4,
                        maximum_images=27,
                        maximum_case_edges=12,
                    ),
                )
            ),
            skin=0.5,
        ),
    ),
    tuple(
        phx.atomistic.AtomisticRoleCapacityLadder(
            requirement,
            (phx.atomistic.AtomisticRoleCapacity(maximum_routes=8, maximum_degree=4),),
        )
        for requirement in requirements
    ),
    frame=phx.discretization.PeriodicCell(
        20.0 * jnp.eye(3), origin=jnp.full((3,), -10.0), periodic_axes=(False,) * 3
    ),
)
rollout = phx.atomistic.AtomisticDistillationRollout(
    system,
    phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LearnedGraphPotentialTerm(student, name="student")]
    ),
    term="student",
    neighborhood_bundle=resources,
    graph_execution=phx.atomistic.AtomisticGraphExecutionPlan(4, backend="particle"),
    integrator=phx.atomistic.BAOABLangevinPlan(0.25, 0.01),
    states=(
        phx.atomistic.AtomisticThermodynamicStatePlan(
            phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
            ensemble="nvt",
            temperature=300.0,
        ),
    ),
    walkers=tuple(
        phx.atomistic.AtomisticTaskPacket(
            positions=water + 0.03 * index,
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
held_out = structures(2, 11, "qualify")
plan = phx.atomistic.AtomisticOnPolicyDistillationPlan(
    teacher=teacher,
    request=request,
    student=student,
    rollout=rollout,
    objectives=objectives,
    policy=phx.atomistic.AtomisticTrainingPolicy(maximum_steps=2, learning_rate=1e-3),
    graph_execution=graph,
    qualification=phx.atomistic.AtomisticDistillationQualification(
        phx.atomistic.label_atomistic_structures(
            teacher,
            phx.atomistic.atomistic_training_loader(held_out, shuffle=False),
            fields,
            request,
            graph_execution=graph,
            cutoff=cutoff,
            store=label_store("qualification-labels"),
        ),
        held_out,
        maximum_loss=1.0e6,
    ),
    cutoff=cutoff,
    capacity=8,
    replay=phx.atomistic.AtomisticReplayDraw((2, 1)),
    seed=5,
)
directory = root / "on-policy"
state = phx.atomistic.AtomisticOnPolicyDistillationState.initial(plan)
first = phx.atomistic.run_atomistic_distillation_segment(
    plan, state, directory=directory, stop_after="label"
)
if not first.successful:
    raise RuntimeError(f"On-policy rollout/label failed: {first.report.failure}")
# A fresh process would rebuild ``plan`` and read the committed restart.
restored = phx.atomistic.read_atomistic_distillation_restart(directory, plan)
qualified = phx.atomistic.run_atomistic_distillation_segment(
    plan, restored, directory=directory, stop_after="qualify"
)
if not qualified.successful:
    raise RuntimeError(f"On-policy qualification failed: {qualified.report.failure}")
qualification = qualified.state.qualification
if qualification is None:
    raise RuntimeError("The completed qualification phase returned no decision.")
segment = phx.atomistic.run_atomistic_distillation_segment(
    plan, qualified.state, directory=directory
)
if not segment.successful:
    raise RuntimeError(f"On-policy distillation failed: {segment.report.failure}")
print(
    "on-policy",
    segment.state.segment,
    segment.report.promoted,
    {
        phase: seconds
        for result in (first, qualified, segment)
        for phase, seconds in zip(
            result.report.phases, result.report.phase_seconds, strict=True
        )
    },
)
workspace.cleanup()
