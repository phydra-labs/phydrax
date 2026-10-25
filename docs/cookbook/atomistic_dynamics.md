# Run a bounded periodic atomistic trajectory

These recipes follow `examples/atomistic_dynamics.py`,
`examples/periodic_mace_dynamics.py`, and `examples/atomistic_relaxation_campaign.py`.
Every neighborhood, capacity, and graph resource is
declared explicitly; nothing falls back to a dense or default search.

Prepare one explicit reduced unit system, fixed periodic cell, atomistic support, and
short-range potential program:

```python
import jax
import jax.numpy as jnp
import numpy as np
import phydrax as phx

ATOMS = phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS
units = phx.atomistic.AtomisticUnitSystem.reduced()
cell = phx.discretization.PeriodicCell(4.0 * jnp.eye(3))
system = phx.atomistic.AtomisticSystemPlan(
    [100, 101, 102, 103],
    [1, 1, 1, 1],
    [1.0, 1.0, 1.0, 1.0],
    units,
    atom_type_ids=[0, 0, 0, 0],
    cell=cell,
).prepare()
program = phx.atomistic.AtomisticPotentialProgram(
    [phx.atomistic.LennardJonesPotential([0.2], [0.8], 1.5, switch_distance=1.3)]
)
```

The program lists every neighborhood it needs in `program.requirements.neighborhoods`.
Each entry names an owner, a role, and a cutoff, and carries no storage. One
`AtomisticNeighborhoodBundlePlan` binds them to explicit resources:

- a certified image source, made of a search policy, a skin, and a search capacity
  ladder;
- one role capacity ladder per bounded requirement.

There is no second neighborhood argument anywhere downstream:

```python
def bundle_plan(program, search_capacities, role_capacities, *, skin, deformation_margin):
    (pairs,) = program.requirements.neighborhoods
    return phx.atomistic.AtomisticNeighborhoodBundlePlan(
        (pairs,),
        (
            phx.atomistic.AtomisticNeighborhoodSource(
                "lennard-jones-pairs",
                ATOMS,
                (pairs,),
                phx.atomistic.AtomisticCellListImageSearch(
                    maximum_candidate_slots=10_000_000,
                    deformation_margin=deformation_margin,
                ),
                phx.discretization.ParticleImageCapacityLadder(search_capacities),
                skin=skin,
            ),
        ),
        (phx.atomistic.AtomisticRoleCapacityLadder(pairs, role_capacities),),
    )


# Search radius 1.5 + 0.2 skin; four atoms per cell, six unordered pairs.
small_search = phx.discretization.ParticleImageCapacity(
    maximum_particles_per_cell=4,
    maximum_edges=12,
    maximum_degree=3,
    maximum_images=27,
    maximum_case_edges=12,
)
small_role = phx.atomistic.AtomisticRoleCapacity(maximum_routes=6, maximum_degree=3)
potential = program.prepare(
    system,
    neighborhood_bundle=bundle_plan(
        program, (small_search,), (small_role,), skin=0.2, deformation_margin=0.0
    ),
)
dynamics = phx.atomistic.AtomisticDynamicsPlan(
    system, potential, phx.atomistic.BAOABLangevinPlan(2.0e-4, 0.2)
).prepare()
nvt_state = phx.atomistic.AtomisticThermodynamicStatePlan(
    phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
    ensemble="nvt",
    temperature=1.0,
)
thermodynamic = nvt_state.prepare(dynamics)
```

The source searches `cutoff + skin = 1.7`. It searches again only after its
displacement and cell-deformation certificate expires. `dynamics.bundle` is the
prepared bundle the Hamiltonian is bound to, and the accepted state carries exactly one
bundle state in `state.neighborhoods`.

Initialize with exactly one of velocity or momentum, then run a fixed-capacity trajectory:

```python
positions = cell.cartesian(
    jnp.asarray(
        [[0.10, 0.10, 0.10], [0.35, 0.10, 0.10],
         [0.10, 0.35, 0.10], [0.35, 0.35, 0.10]]
    )
)
initial = dynamics.initialize_state(
    positions,
    thermodynamic,
    velocity=jnp.zeros_like(positions),
    key=jax.random.key(0),
)
result = phx.atomistic.AtomisticRolloutPlan(
    dynamics,
    thermodynamic,
    phx.atomistic.AtomisticTrajectoryPlan(20, sample_stride=5),
    replay=phx.atomistic.AtomisticReplayPolicy("step"),
).rollout(initial)
if not bool(result.successful):
    raise RuntimeError("atomistic rollout rejected a step")
diagnostics = dynamics.diagnostics(result.final_state, thermodynamic)
```

`initialize_state` raises on an inadmissible initial state.
`initialize_state_detailed` returns the same state with its `successful` flag and
`rejection_reasons` instead. The trajectory capacity is resolved before execution.
`result.replay` records accepted and rejected counts plus route, image, and stochastic
digests. `diagnostics.neighborhood_rebuild_count` counts source searches. Persist exact
continuation state with `AtomisticCheckpointPlan`, `write_atomistic_checkpoint`, and
`read_atomistic_checkpoint`.

## Grow capacity only when a step needs it

Capacity growth is declared up front, in the bundle plan. Give the source a
`ParticleImageCapacityLadder`, and give each bounded requirement an
`AtomisticRoleCapacityLadder`, each with more than one strictly growing entry. Only the
first entries are used until `retry_atomistic_step_with_capacity` observes a rejection
whose only causes are `SOURCE_CAPACITY` or `ROUTE_CAPACITY`. Overflow can also set the
potential and nonfinite flags, and those are allowed.

The function then does the following:

1. It selects the next covering declared entries.
2. It rebinds the Hamiltonian with `potential.with_neighborhood_bundle` through
   `rebind_hamiltonian`.
3. It re-primes the force cache with `prime_state`.
4. It retries the same physical attempt at the same step, revision, RNG address, and
   controller state.

Scientific, domain, stale, thermodynamic, and constraint failures are returned
unchanged, and an exhausted declared ladder raises:

```python
replay_potential = program.prepare(
    system,
    neighborhood_bundle=bundle_plan(
        program,
        (
            small_search,
            phx.discretization.ParticleImageCapacity(
                maximum_particles_per_cell=8,
                maximum_edges=24,
                maximum_degree=6,
                maximum_images=27,
                maximum_case_edges=24,
            ),
        ),
        (
            small_role,
            phx.atomistic.AtomisticRoleCapacity(maximum_routes=12, maximum_degree=6),
        ),
        skin=0.2,
        deformation_margin=0.0,
    ),
)
replay_dynamics = phx.atomistic.AtomisticDynamicsPlan(
    system, replay_potential, phx.atomistic.BAOABLangevinPlan(2.0e-4, 0.2)
).prepare()
replay_table = nvt_state.prepare(replay_dynamics)
replay_state = replay_dynamics.initialize_state(
    positions,
    replay_table,
    velocity=jnp.zeros_like(positions),
    key=jax.random.key(0),
)
for _ in range(5):
    replay_dynamics, replay_table, evaluation = (
        phx.atomistic.retry_atomistic_step_with_capacity(
            replay_dynamics, replay_state, replay_table, (nvt_state,)
        )
    )
    if not bool(evaluation.successful):
        raise RuntimeError(f"step rejected: {int(evaluation.rejection_reasons)}")
    replay_state = evaluation.accepted_state
(pair_requirement,) = program.requirements.neighborhoods
slot = replay_dynamics.bundle.slot(pair_requirement)
routes = replay_dynamics.bundle.capacity(slot).maximum_routes
```

The returned runtime and table replace the old ones, because a grown bundle is a new
resource binding of the same scientific Hamiltonian. A checkpoint written afterwards
records the grown capacity levels. A fresh process rebinds its preparation with
`read_atomistic_checkpoint_resources(path).bind(potential)` before building the
dynamics plan.

## Thermostats and cell methods on a small crystal

A 108-atom FCC Lennard-Jones crystal is bound to one bundle. The 4.8 box exceeds twice
the 2.0 classical cutoff, with margin for cell fluctuations, so no pair has two minimum
images inside the cutoff. `deformation_margin` lets the source certificate absorb cell
deformation. Integrator-owned MTK cells scale every atom, so the measure declares
atomic volume scaling:

```python
lattice = 1.6
fractional = np.asarray(
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
)
offsets = np.stack(
    np.meshgrid(np.arange(3), np.arange(3), np.arange(3), indexing="ij"), axis=-1
).reshape((-1, 1, 3))
crystal_positions = jnp.asarray(
    (lattice * (offsets + fractional[None] + 0.05)).reshape((-1, 3))
)
count = crystal_positions.shape[0]
crystal = phx.atomistic.AtomisticSystemPlan(
    np.arange(count),
    np.ones((count,), dtype=np.int32),
    np.ones((count,)),
    units,
    atom_type_ids=np.zeros((count,), dtype=np.int32),
    cell=phx.discretization.PeriodicCell(3.0 * lattice * np.eye(3)),
).prepare()
crystal_program = phx.atomistic.AtomisticPotentialProgram(
    [phx.atomistic.LennardJonesPotential([1.0], [1.0], 2.0, switch_distance=1.8)]
)
crystal_potential = crystal_program.prepare(
    crystal,
    neighborhood_bundle=bundle_plan(
        crystal_program,
        (
            phx.discretization.ParticleImageCapacity(
                maximum_particles_per_cell=16,
                maximum_edges=6400,
                maximum_degree=80,
                maximum_images=27,
                maximum_case_edges=6400,
            ),
        ),
        (phx.atomistic.AtomisticRoleCapacity(maximum_routes=3200, maximum_degree=80),),
        skin=0.3,
        deformation_margin=0.2,
    ),
)
measure = phx.atomistic.AtomisticPhaseSpaceMeasurePlan(
    crystal, volume_coordinate_convention="atomic"
)
temperature, pressure = 0.5, 1.0
velocity = jnp.sqrt(temperature) * jax.random.normal(
    jax.random.key(7), crystal_positions.shape
)
velocity = velocity - jnp.mean(velocity, axis=0)


def crystal_run(integrator, ensemble, *, steps=40, barostat=None):
    runtime = phx.atomistic.AtomisticDynamicsPlan(
        crystal, crystal_potential, integrator
    ).prepare()
    table = phx.atomistic.AtomisticThermodynamicStatePlan(
        measure,
        ensemble=ensemble,
        temperature=None if ensemble == "nph" else temperature,
        pressure=None if ensemble == "nvt" else pressure,
    ).prepare(runtime)
    start = runtime.initialize_state(
        crystal_positions, table, velocity=velocity, key=jax.random.key(3)
    )
    rollout = phx.atomistic.AtomisticRolloutPlan(
        runtime,
        table,
        phx.atomistic.AtomisticTrajectoryPlan(steps, retention="final"),
        replay=phx.atomistic.AtomisticReplayPolicy("step"),
        barostat=barostat,
        barostat_interval=None if barostat is None else 5,
    ).rollout(start)
    if not bool(rollout.successful):
        raise RuntimeError(f"{type(integrator).__name__} rollout failed")
    before = runtime.diagnostics(start, table)
    after = runtime.diagnostics(rollout.final_state, table)
    return {
        "extended_energy_drift": float(after.extended_energy - before.extended_energy),
        "temperature": float(after.temperature),
        "pressure": float(after.pressure),
        "volume": float(after.volume),
        "accepted_volume_moves": int(rollout.final_state.controllers.accepted_moves),
    }


step = 1.0e-3
nhc = crystal_run(
    phx.atomistic.NoseHooverChainPlan(
        step,
        phx.atomistic.NoseHooverChainParameters(
            0.1, chain_length=3, yoshida_suzuki_order=3
        ),
    ),
    "nvt",
)
csvr = crystal_run(phx.atomistic.StochasticVelocityRescalingPlan(step, 0.1), "nvt")
mtk_npt = crystal_run(
    phx.atomistic.MTKNPTPlan(
        step,
        phx.atomistic.MTKBarostatPlan(1.0, chart="triclinic-symmetric"),
        phx.atomistic.NoseHooverChainParameters(0.1),
        phx.atomistic.NoseHooverChainParameters(1.0),
    ),
    "npt",
)
mtk_nph = crystal_run(
    phx.atomistic.MTKNPHPlan(
        step, phx.atomistic.MTKBarostatPlan(1.0, inertia_temperature=temperature)
    ),
    "nph",
)
mc_npt = crystal_run(
    phx.atomistic.BAOABLangevinPlan(step, 1.0),
    "npt",
    barostat=phx.atomistic.IsotropicMonteCarloBarostatPlan(0.01),
)
```

The five runs show the following:

- **Nose–Hoover chain.** `NoseHooverChainPlan` is a canonical chain with fourth-order
  Suzuki–Yoshida weights (`yoshida_suzuki_order=3`), and its chain energy enters
  `extended_energy`.
- **CSVR.** `StochasticVelocityRescalingPlan` is Bussi–Donadio–Parrinello stochastic
  velocity rescaling.
- **MTK NPT.** `MTKNPTPlan` owns the cell on the six-dimensional, rotation-free
  `triclinic-symmetric` chart, with particle and cell chains.
- **MTK NPH.** `MTKNPHPlan` has no thermostat of any kind. Its `inertia_temperature`
  only sets the cell inertia `W = (N_f + 3) k T_W tau_p^2` and is not a target.
- **Monte Carlo NPT.** The scheduled `IsotropicMonteCarloBarostatPlan` owns the cell
  over a canonical particle method that does not. An NPT rollout needs exactly one cell
  owner.

For the MTK runs, `extended_energy` is `K + U + K_cell + P_ext V + E_chains`, and its
drift is a realization diagnostic. `diagnostics.pressure` is `trace(P) / 3` of the
instantaneous internal pressure `P = (sum_i p_i p_i^T / m_i - sym(dE/dstrain)) / V`.
That is the kinetic momentum flux minus the tension-positive strain stress cached by
every force evaluation. It is not a position-moment virial.

## Periodic native MACE dynamics

This recipe follows `examples/periodic_mace_dynamics.py`. Two water-like molecules sit in
a triclinic cell smaller than twice the 3 Å cutoff. The learned term declares one
directed image-graph requirement, and an image source enumerates every translation
within `cutoff + skin`, including repeated images of one pair when the geometry places
them inside the cutoff. The model is a random native initialization, not a fitted or
released potential.

```python
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import phydrax as phx

UNITS = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
CELL = np.array([[4.6, 0.0, 0.0], [0.9, 4.4, 0.0], [0.5, -0.6, 4.8]])
POSITIONS = np.array(
    [
        [0.2, 0.3, 0.1],
        [1.16, 0.3, 0.1],
        [-0.04, 1.23, 0.1],
        [2.5, 2.4, 2.6],
        [3.46, 2.4, 2.6],
        [2.26, 3.33, 2.6],
    ]
)
NUMBERS = np.array([8, 1, 1, 8, 1, 1])
MASSES = np.array([15.999, 1.008, 1.008, 15.999, 1.008, 1.008])

architecture = phx.nn.atomistic.MACEArchitecture(
    species=(1, 8),
    cutoff=3.0,
    radial_basis_count=4,
    cutoff_power=5,
    channel_count=4,
    hidden_degree=1,
    edge_degree=2,
    interactions=("real-agnostic", "real-agnostic-residual"),
    correlations=(2, 2),
    radial_widths=(8,),
    readout_width=4,
    average_neighbor_count=6.0,
)
model = phx.nn.atomistic.MACEPotential(
    UNITS.scale,
    architecture,
    atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
    key=jax.random.key(0),
)
```

A cell-list image source of radius `cutoff + skin` serves the graph requirement.
`ParticleImageCapacity` charges cell occupancy, stored edges, receiver degree, and image
translations separately. The role ladder bounds the graph's stored routes and receiver
degree. Deterministic streamed accumulation fixes the order of edge sums. The same
prepared program drives fixed-cell NVE dynamics:

```python
def prepare(model):
    structure = phx.atomistic.AtomicStructure(
        jnp.asarray(NUMBERS),
        jnp.asarray(POSITIONS),
        jnp.asarray(MASSES),
        UNITS.scale,
        cell=jnp.asarray(CELL),
        periodic_axes=jnp.asarray([True, True, True]),
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(structure, UNITS).prepare()
    execution = phx.atomistic.AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        streamed=phx.sparse.StreamedRelationPlan(accumulation="deterministic"),
    )
    program = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LearnedGraphPotentialTerm(model, allow_periodic=True)]
    )
    (graph,) = program.requirements.neighborhoods
    bundle = phx.atomistic.AtomisticNeighborhoodBundlePlan(
        (graph,),
        (
            phx.atomistic.AtomisticNeighborhoodSource(
                "mace-graph",
                phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS,
                (graph,),
                phx.atomistic.AtomisticCellListImageSearch(
                    maximum_candidate_slots=10_000_000
                ),
                phx.discretization.ParticleImageCapacityLadder(
                    (
                        phx.discretization.ParticleImageCapacity(
                            maximum_particles_per_cell=16,
                            maximum_edges=2048,
                            maximum_degree=96,
                            maximum_images=343,
                            maximum_case_edges=2048,
                        ),
                    )
                ),
                skin=0.4,
            ),
        ),
        (
            phx.atomistic.AtomisticRoleCapacityLadder(
                graph,
                (
                    phx.atomistic.AtomisticRoleCapacity(
                        maximum_routes=2048, maximum_degree=96
                    ),
                ),
            ),
        ),
    )
    potential = program.prepare(
        system, graph_execution=execution, neighborhood_bundle=bundle
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system, potential, phx.atomistic.VelocityVerletPlan(0.25)
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
    ).prepare(dynamics)
    return dynamics, thermodynamic


def run(dynamics, thermodynamic, state, steps):
    for _ in range(steps):
        state = dynamics.step(state, thermodynamic)
    return state


dynamics, thermodynamic = prepare(model)
velocity = 0.002 * jax.random.normal(jax.random.key(1), POSITIONS.shape)
state = dynamics.initialize_state(
    jnp.asarray(POSITIONS), thermodynamic, velocity=velocity, key=jax.random.key(2)
)
energy = state.force.potential_energy  # eV
stress = state.force.stress  # eV / Å^3, tension positive, H' = H @ F.T
state = run(dynamics, thermodynamic, state, 20)
reference = run(dynamics, thermodynamic, state, 10)
residual = reference.energy.cumulative_balance_residual  # eV
```

`initialize_state` primes the force cache. The cell is fully periodic 3x3 and the
learned term owns a cell derivative, so `dynamics.strain_response` defaults to true and
the cache holds `strain_derivative = dE/dstrain` and the tension-positive `stress`.
Self-image routes contribute to the strain derivative even though their net positional
force cancels. `state.energy.cumulative_balance_residual` tracks the NVE energy
balance.

`BAOABLangevinPlan`, `NoseHooverChainPlan`, or `StochasticVelocityRescalingPlan` with an
`ensemble="nvt"` state supply fixed-cell NVT. Because the program owns its cell
derivative, MTK NPT/NPH and scheduled Monte Carlo NPT are admitted by the same rules as
for classical programs; see the [dynamics guide](../guides_atomistic_dynamics.md).
Admission is not evidence that a model is stable under cell fluctuations.

A portable restart bundles the following:

- the pickle-free native artifact of every executed learned term's model;
- the term bindings;
- the complete dynamics state, with the bundle recipe and capacity levels.

`read_atomistic_restart_models` returns one binding per learned term, keyed by term
name. A Hamiltonian with two learned terms restores two bindings, and its fresh
preparation must execute both. The rebuilt preparation must match the one that wrote
the restart, neighborhood bundle plan and capacity levels included. A missing,
additional, or altered term, model, system, integrator, thermodynamic table, or bundle
is refused:

```python
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "restart"
    plan = phx.atomistic.AtomisticCheckpointPlan(dynamics, thermodynamic)
    phx.atomistic.write_atomistic_restart(path, plan, state)

    (binding,) = phx.atomistic.read_atomistic_restart_models(path).values()
    restored_model = binding.artifact.model
    restored_dynamics, restored_thermodynamic = prepare(restored_model)
    template = restored_dynamics.initialize_state(
        jnp.asarray(POSITIONS),
        restored_thermodynamic,
        velocity=jnp.zeros_like(velocity),
        key=jax.random.key(2),
    )
    resumed = phx.atomistic.read_atomistic_restart(
        path,
        phx.atomistic.AtomisticCheckpointPlan(
            restored_dynamics, restored_thermodynamic
        ),
        template,
    ).state
    continued = run(restored_dynamics, restored_thermodynamic, resumed, 10)
identical = bool(
    jnp.array_equal(continued.kinematics.positions, reference.kinematics.positions)
)
```

The restored state is exact. The continuation equals the uninterrupted trajectory
bitwise only under the following conditions:

- the graph plan selects deterministic streamed accumulation, as here;
- the resuming process uses the same device type and runtime versions;
- the compiler profile is fixed.

If the saved run grew its capacities through `retry_atomistic_step_with_capacity`,
rebind the fresh potential with
`phx.atomistic.read_atomistic_checkpoint_resources(path).bind(potential)` before
building `AtomisticDynamicsPlan`.

Native MACE support envelopes are unreleased candidates. A random-initialized model
demonstrates the runtime contract, not material accuracy or MD stability.

## Staged heterogeneous campaigns

This recipe follows `examples/atomistic_relaxation_campaign.py`. Two Lennard-Jones
families, dimers and tetramers in a periodic box, flow through four native stages:
FIRE relaxation, BAOAB NVT, MTK NPT, and E/F/S labeling. The example is written so that
one task fails scientifically (two coincident atoms), one task is capacity limited, and
the output sink accepts only one record per round, so finished lanes queue behind
bounded publication.

An `AtomisticCampaignPlan` composes existing native owners. It does not replace them
with another scheduler, integrator, or optimizer:

- each stage's scientific transition belongs to its operation (`PreparedAtomisticDynamics`,
  `PreparedAtomisticRelaxation`, `PreparedAtomisticNudgedElasticBand`, a prepared
  sampler, or a prepared Hamiltonian);
- `ExecutionWorksetPlan` supplies semantic RNG keys and signature grouping;
- the bounded frontier and refill helpers of the execution pool own admission;
- lifecycle array archives make outputs and commits durable;
- `IterationSession` receives one observation record per round.

This is the ordinary lane campaign. All of its lanes run in one process, and each lane is
a whole system or a whole path band. Distributed domain-decomposed dynamics across stages
has a separate owner, `AtomisticDomainCampaignPlan`, built on
`AtomisticStageTransportPlan`. Each of its stages owns a child `ExecutionGroup`, and
checkpoint, restore, and migration follow their own topology contract. See
[stage x domain campaigns](../guides_atomistic_distributed_execution.md#stage-x-domain-campaigns).
The two owners publish through the same `AbstractAtomisticCampaignSink` boundary, but
neither owner runs the other.

### Stages, families, and lane banks

The system builders and stage operations (`_system`, `_potential`, `_relax`, `_nvt`, and
`_npt`) are defined in the example. Each operation wraps one prepared native runtime:

- `AtomisticRelaxationStageOperation(prepared_relaxation, rounds_per_segment=50)` wraps a
  `FIRE` relaxation prepared with `group_count=LANES`;
- `AtomisticDynamicsStageOperation(prepared_dynamics, thermodynamic_rows,
  steps_per_round=10)` wraps a BAOAB or MTK runtime together with its thermodynamic table;
- `AtomisticLabelStageOperation(prepared_program, stress=True)` labels with a prepared
  Hamiltonian.

The excerpt below abridges the task list to the failing task. The example builds three
dimers and two tetramers, then appends this coincident dimer:

```python
def build_plan() -> AtomisticCampaignPlan:
    dimer, tetramer = _system(2), _system(4)
    stages = (
        AtomisticStagePlan(
            "relax",
            {"dimer": _relax(dimer), "tetramer": _relax(tetramer)},
            routing=AtomisticStageRouting(
                exhausted=AtomisticStageRoute("nvt", publish=True)
            ),
            lane_capacity=LANES,
        ),
        AtomisticStagePlan(
            "nvt",
            # Two directed-half routes hold a dimer but not a relaxed tetramer:
            # tetramers are rebucketed to the next declared role capacity.
            {"dimer": _nvt(dimer, (2, 64)), "tetramer": _nvt(tetramer, (2, 64))},
            budget=40,
            lane_capacity=LANES,
            queue_capacity=2,
        ),
        AtomisticStagePlan(
            "npt",
            {"dimer": _npt(dimer), "tetramer": _npt(tetramer)},
            budget=20,
            lane_capacity=LANES,
            queue_capacity=2,
            # Another integrator and ensemble: an explicit new run, not a continuation.
            continuation=AtomisticContinuationPolicy.RESTART,
        ),
        AtomisticStagePlan(
            "label",
            {
                "dimer": AtomisticLabelStageOperation(_potential(dimer), stress=True),
                "tetramer": AtomisticLabelStageOperation(
                    _potential(tetramer), stress=True
                ),
            },
            lane_capacity=LANES,
            queue_capacity=1,
            entry=AtomisticEntryPolicy.SHARE,
        ),
    )
    tasks = [
        AtomisticCampaignTask(
            "dimer-coincident",
            "dimer",
            AtomisticTaskPacket(positions=np.asarray([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])),
            material_id="dimer",
        ),
        # The example also builds dimer-0..2 and tetramer-0..1 the same way.
    ]
    return AtomisticCampaignPlan(
        "lj-relaxation-campaign", stages, tuple(tasks), max_pending_publications=2
    )
```

A task is `AtomisticCampaignTask(task_id, family, packet, material_id=...)`, with
optional `run_id`, `replica_id`, `group_id`, `entry_stage`, and `ordinal`. Its initial
`AtomisticTaskPacket` carries unwrapped positions and, where needed, a cell, momenta,
species, a path, or an earlier evaluation. The only method continuation an initial
packet may carry is a walker's `AtomisticDynamicsContinuation` from an earlier campaign,
which the entry stage continues or refuses.

Mixed sizes are expressed as **families**. Each family maps to an operation over a
system of its own capacity, so dimers and tetramers occupy different fixed-capacity lane
banks. Nothing is padded to the largest task. Each bank's `AtomisticStageSignature` holds
only layout identities: the system, resource-bundle, and Hamiltonian layouts, the
derivative request, the precision, the method-state layout, and the placement.
Scientific identity, numeric revisions, tasks, and budgets are dynamic `IdentityWords`
or host records. Every material and revision of one layout therefore shares one
executable per entrypoint.

`PreparedAtomisticCampaign` validates the plan before any numerical work:

- Stage names, task IDs, and routes must be unique and resolvable.
- Every family must have an operation in every stage it can be routed to.
- `budget` is required for dynamics and generation stages. It counts accepted dynamics
  steps or sampler steps per visit. Relaxation and reaction-path stages use their owner's
  convergence and evaluation budget instead.
- `lane_capacity` bounds the resident lanes of each bank. A natively grouped operation
  (relaxation, a reaction path, or a fixed-lane sampler) must have a native group count
  equal to `lane_capacity`.
- Stages that use the same operation (equal `binding_id`) share one bank. The bank
  advances all of their lanes in one batched native call, so it needs one lane capacity
  in every stage.
- `queue_capacity` bounds how many tasks may wait to enter a stage.

### Outcomes, routing, and status on labels

Each lane ends a stage visit with one `AtomisticStageOutcome`:

| Outcome | Produced by | Default route |
| --- | --- | --- |
| `CONVERGED` | relaxation or reaction path met its convergence target | next stage without publishing; publish if this is the last stage |
| `COMPLETED` | a fixed budget (dynamics steps, sampler steps) or a successful label | same as `CONVERGED` |
| `EXHAUSTED` | relaxation or path reached its evaluation limit before convergence | terminate and publish |
| `FAILED` | scientific, numerical, or cell failure; owner-refused initial state; a non-finite label; failed re-prime; refused admission | terminate and publish |
| `CAPACITY_REFUSED` | the declared capacity ladder is exhausted | terminate and publish |

`CAPACITY_PENDING` is not terminal. The attempt is not consumed, and the capacity
lifecycle replays it (see below). `AtomisticStageRouting` overrides a default for each
outcome with `AtomisticStageRoute(next_stage, publish=...)`.

In the example, the relax stage sends `EXHAUSTED` tasks on to NVT and publishes the
unconverged relaxation, so that record stays visible. A `FAILED` relaxation keeps the
default route: it terminates, and its failure record is published. A failed task
therefore never reaches the label stage, and no label is fabricated for it.

Every published `AtomisticCampaignOutput` carries the information needed to judge the
record:

- `outcome`;
- the method-owned `reasons` code;
- the `AtomisticCampaignReason` bits in `campaign_reasons` (for example
  `ADMISSION_REFUSED`, `INVALID_PACKET`, `CELL_MISMATCH`, `LADDER_EXHAUSTED`,
  `REPRIME_FAILED`, `UNROUTABLE`, `SHARED_EVALUATION`, `IDENTITY_MISMATCH`,
  `CONTINUATION_REFUSED`, `DYNAMICS_RESTARTED`, and the capacity kinds);
- `counters` (`consumed`, `stage_entry`), `numeric_revision_ids`,
  `hamiltonian_lineage`, `rng_words`, and stable task, material, run, replica, group, and
  parent identities.

Energy, forces, and stress ports appear only when the packet carries a finite accepted
`AtomisticEvaluationRecord`. A label lane whose evaluation is unsuccessful or non-finite
ends `FAILED` and publishes no energy ports. Filter training data on `outcome` and
`campaign_reasons` rather than on the presence of a file.

### Groups and whole reaction paths

Tasks with the same `group_id` must share a family and an entry stage, and a group may
not exceed the lane capacity of any stage of its family. The group moves as one unit:

- refill admits it atomically, or not at all;
- it leaves a stage only after every member still in that stage is terminal. Until
  then, finished members wait in their lanes, counted by
  `AtomisticRoundEvidence.group_waiting`;
- the next stage queue and the pending-publication bank must both admit the whole unit
  before any member moves. Publication itself proceeds record by record, so
  backpressure can leave some of a group's records pending for a later round.

Bank admission has prefix semantics: a group that does not fit blocks the later units
for that bank.

A reaction path is a stronger kind of group. With `AtomisticReactionPathStageOperation`,
one lane is one complete band from `phydrax.chemistry.reaction.AtomisticNudgedElasticBandPlan`
(`group_count` bands per bank). The packet's `path` has shape `(M, N, 3)`, and its first
and last images are the endpoints. The whole band is admitted, advanced, checkpointed,
exported, and published together. Individual image slots are never refilled. A band
that needs larger capacities at admission is transferred to the bank at the grown
binding. In flight, the owner's bank-wide `grow` replays capacity-pending bands. The
exported packet carries `path` and `path_energies`, with `positions` set to the
native result's interior climbing image. Fixed endpoints need not be minima;
an endpoint with greater energy does not replace the reported climbing image.

### Walker continuation, force caches, and sharing

Between stages, a task carries its last **accepted** state as a host packet:

- unwrapped positions;
- the current cell, which for an NPT stage is its last accepted cell;
- momenta and time;
- the accepted evaluation record;
- for a dynamics walker, its complete `AtomisticDynamicsContinuation`: thermostat
  chains, barostat, GLE memory and Monte Carlo counters, step, time, RNG key, images,
  term states, force cache, and neighborhood resources, bound to the exporting
  operation.

A stage boundary is an observation cadence, not a new run. The next dynamics stage
adopts a state of the same runtime bitwise, so a walker observed every 5 steps follows
exactly the trajectory of one 20-step visit, including stateful thermostats and
barostats. Another row of the same thermodynamic table is entered by the owner's warm
rebase. A changed Hamiltonian of the same method is entered by the source runtime's
`transition_hamiltonian`, which records lineage and books the energy jump as external
work. A changed integrator, table, constraints, or system is refused with
`CONTINUATION_REFUSED` unless the stage declares
`continuation=AtomisticContinuationPolicy.RESTART`. In the example, NVT to MTK NPT is
such a change. A restart is initialized through `initialize_state_detailed` from the
packet's positions, momenta, cell, and time, with the campaign's semantic key, and the
visit records `DYNAMICS_RESTARTED`. A packet without a continuation (after relaxation,
for example) is initialized the same way.

Labeling evaluates the walker's configuration without moving it, so the walker's
continuation passes through a label stage: the label output also reports its momenta
and time, and a following dynamics stage continues the run. Relaxation and path stages
move the configuration and end the run. A rejected trial never becomes a stage output,
and a refused walker keeps its continuation as evidence.
`PreparedAtomisticCampaign.terminal_packets(state)` returns every terminal walker's
final continuation packet. That packet can enter a later campaign. A changed model is
entered first through `old_operation.transition_packet(packet, new_operation)`, while
the old runtime is still available.

A force evaluation is shared only when it is the same evaluation. With
`entry=AtomisticEntryPolicy.SHARE`, a label operation built with `reuse=True` adopts the
incoming evaluation when the operation's `AtomisticEvaluationKey` is served by the
packet's key. That requires the same Hamiltonian, numeric revisions, conditions, and bias,
and a subset of the available quantities. Such reuse sets `SHARED_EVALUATION` and is
counted in `shared_evaluations`. Dynamics, relaxation, path, and generation operations
have no shared-entry transition, so preparation refuses `SHARE` for them.

Re-priming a force cache never consumes a physical step, an RNG draw, or stage budget.
It happens in two places:

- **Capacity.** `AtomisticCapacityMode.REBUCKET` operations (dynamics, label, and
  generation) move only the offending task to a bank at the next declared capacities.
  For dynamics, the owner calls `rebind_hamiltonian` and then `prime_state`, and the
  task carries its consumed budget into the new bank. If the re-prime fails, the task
  ends `FAILED` with `REPRIME_FAILED`. If the ladder is exhausted, it ends
  `CAPACITY_REFUSED` with `LADDER_EXHAUSTED`. `GROW` operations (relaxation and reaction
  paths) grow their whole bank in place, which preserves every optimizer history and
  replays the pending proposal. The example declares the NVT role ladder `(2, 64)` so
  that the first level holds a dimer but not a relaxed tetramer, which forces the
  tetramers to be rebucketed.
- **Revision.** `PreparedAtomisticCampaign.rebind_operation(state, stage, family,
  operation)` binds another Hamiltonian with the same operation kind and static
  signature. It is accepted only in the refilled phase.
  - Under `AtomisticRevisionPolicy.DRAIN`, admitted lanes finish under the old binding
    and new admissions use the new binding.
  - Under `REPRIME`, every admitted lane passes through the owner's explicit
    `transition_hamiltonian`, and the lineage is recorded. Dynamics keeps
    configuration, momenta, time, step, RNG, controllers, and counters, re-primes forces,
    and books the energy jump as external work. Relaxation and path banks keep the
    accepted geometry, re-prime the accepted iterate, and reset optimizer history
    explicitly. Label lanes drop a reused evaluation and are evaluated by the new
    teacher, and lanes that already hold a label are refused. `REPRIME` is refused while
    the bank holds outputs that await publication under the old Hamiltonian, while
    capacity transfers are queued, or while the bank is shared with other stage families.

  The rebind becomes durable at the next commit. `resume(store, operations=...)` then
  needs the rebound operations that the plan does not contain.

### Round transaction, backpressure, and exactly-once publication

`run_round(state, sink, store, session=None)` composes one transaction:

1. `evaluate` advances each bank's runnable lanes by one native segment and runs the
   capacity lifecycle.
2. `stage_outputs` classifies terminal lanes and moves each whole unit to its next stage
   queue and/or the bounded pending-publication bank. A unit moves only when both the
   queue (`queue_capacity`) and the pending bank (`max_pending_publications`) admit it.
   Otherwise it parks in its lanes with its accepted state, and no physics advances.
3. `publish` asks the sink for receipts in order and stops at the first `None`, which
   signals downstream backpressure.
4. `commit` atomically replaces the `AtomisticCampaignStore` envelope. The envelope holds
   every bank, queue, transfer, task record, pending output, and receipt.
5. `refill` admits queued transfers first, then stage queues in arrival order, into free
   lanes.

Each phase checks the phase of the state it receives, so a phase cannot be skipped or
repeated. `run(state, sink, store, maximum_rounds=...)` loops until
`state.complete`, which requires every task to be terminal and every output durable.

An output's `semantic_key` depends only on the campaign, task, stage, and stage entry.
Its `payload_id` fingerprints the complete content. A sink is idempotent by semantic key:

- `receipt` finds an output that is already durable;
- publishing the same payload again returns the existing receipt;
- a different payload under the same key is refused.

`AtomisticDirectorySink(root, accept_per_round=...)` writes one archive per output.
`accept_per_round` sets an explicit throughput limit, which is how the example produces
backpressure. Admission RNG keys come from `PreparedExecutionWorksets.semantic_keys` over
`stage/task`, with the task's stage entry count as the counter. They do not depend on
lane, round, or completion order.

### Fresh-process continuation

The example's `_process` (abridged; the full version also prints a JSON summary):

```python
def _process(work: Path, phase: str, rounds: int) -> None:
    prepared = build_plan().prepare()
    store = AtomisticCampaignStore(work / "campaign.ckpt")
    sink = AtomisticDirectorySink(work / "outputs", accept_per_round=1)
    if phase == "interrupt":
        state = prepared.start(jax.random.key(2026))
        for _ in range(rounds):
            state = prepared.run_round(state, sink, store)
        # Lose the process after publishing, before the commit is durable.
        prepared.publish(prepared.stage_outputs(prepared.evaluate(state)), sink)
        return
    state = prepared.resume(store)
    for _ in range(2000):
        if state.complete:
            break
        state = prepared.run_round(state, sink, store)
    if not state.complete:
        raise RuntimeError("The campaign did not finish within its round bound.")
```

Run without arguments, the example starts this function twice as separate subprocesses:

1. The `interrupt` process commits several rounds. It then evaluates, stages, and
   publishes one more round, and **exits normally before that round's commit**.
2. The `resume` process rebuilds the plan, calls `resume(store)` to restore the last
   durable commit and refill deterministically, and runs to completion.

The records published in the uncommitted round are still pending in the restored
envelope. When they are published again, `sink.receipt` returns the existing receipt,
so nothing is duplicated. The driver then reads `AtomisticDirectorySink.outputs()` and
checks that every semantic key is unique.

This is a controlled exit at the publish/commit boundary, not a `SIGKILL` crash test.
The killed-rank and exact-restart gates of the distributed stage x domain owner are a
separate harness, described in the
[distributed guide](../guides_atomistic_distributed_execution.md#qualification-and-gates).
`write_atomistic_campaign_checkpoint` and `read_atomistic_campaign_checkpoint` write and
verify a standalone envelope from a committed or refilled state.

Passing `session=` to `run_round` emits one commit-phase `IterationRecord` per round
under the scope `atomistic-campaign:<campaign_id>`. Its metrics come from
`AtomisticRoundEvidence`:

- lane capacity, occupied, advanced, and admitted lanes;
- parked lanes, group waits, pending outputs, published outputs, and backpressure;
- shared evaluations;
- capacity transfers, growths, and refusals;
- terminal and spawned tasks.

These are runtime observations of one run on the device and runtime that executed it.
They are not throughput qualification.
