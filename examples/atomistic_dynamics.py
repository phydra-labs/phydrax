#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic Lennard-Jones dynamics: thermostats, MTK cells, MC volume moves, replay.

Every Hamiltonian is prepared against one explicit neighborhood bundle plan
built from ``program.requirements.neighborhoods``: a certified image source
(cell-list search, skin, search capacity ladder) and one role capacity ladder
per bounded requirement.  Dynamics read their resources from
``potential.neighborhood_bundle``; there is no separate neighborhood argument.
"""

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


ATOMS = phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS
units = phx.atomistic.AtomisticUnitSystem.reduced()


def bundle_plan(
    program: phx.atomistic.AtomisticPotentialProgram,
    search_capacities: tuple[phx.discretization.ParticleImageCapacity, ...],
    role_capacities: tuple[phx.atomistic.AtomisticRoleCapacity, ...],
    *,
    skin: float,
    deformation_margin: float,
) -> phx.atomistic.AtomisticNeighborhoodBundlePlan:
    """One cell-list image source serving the single Lennard-Jones pair view."""
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


# --------------------------------------------------------------------------
# 1. Four-atom BAOAB NVT rollout and a capacity-only replay.
# --------------------------------------------------------------------------
cell = phx.discretization.PeriodicCell(4.0 * jnp.eye(3))
system = phx.atomistic.AtomisticSystemPlan(
    # ty: ignore[invalid-argument-type]
    [100, 101, 102, 103],
    # ty: ignore[invalid-argument-type]
    [1, 1, 1, 1],
    # ty: ignore[invalid-argument-type]
    [1.0, 1.0, 1.0, 1.0],
    units,
    # ty: ignore[invalid-argument-type]
    atom_type_ids=[0, 0, 0, 0],
    cell=cell,
).prepare()
program = phx.atomistic.AtomisticPotentialProgram(
    # ty: ignore[invalid-argument-type]
    [phx.atomistic.LennardJonesPotential([0.2], [0.8], 1.5, switch_distance=1.3)]
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
positions = cell.cartesian(
    jnp.asarray(
        [
            [0.10, 0.10, 0.10],
            [0.35, 0.10, 0.10],
            [0.10, 0.35, 0.10],
            [0.35, 0.35, 0.10],
        ]
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
    raise RuntimeError("Atomistic dynamics rollout failed")
diagnostics = dynamics.diagnostics(result.final_state, thermodynamic)
print(
    {
        "section": "baoab-nvt",
        "samples": int(result.trajectory.count),
        "accepted_steps": int(result.replay.accepted_steps),
        "total_energy": float(diagnostics.total_energy),
        "temperature": float(diagnostics.temperature),
        "neighbor_rebuilds": int(diagnostics.neighborhood_rebuild_count),
    }
)

# Capacity-only replay: the role and search ladders declare two entries; the
# first reproduces the resources above, the second is selected only when a
# step is rejected for source/route capacity alone.  The replay keeps the
# accepted state, step, revision, RNG and controllers.
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
        raise RuntimeError(f"Replayed step rejected: {int(evaluation.rejection_reasons)}")
    replay_state = evaluation.accepted_state
(pair_requirement,) = program.requirements.neighborhoods
slot = replay_dynamics.bundle.slot(pair_requirement)
print(
    {
        "section": "capacity-replay",
        "step_index": int(replay_state.step_index),
        "role_capacity_routes": replay_dynamics.bundle.capacity(slot).maximum_routes,
    }
)

# --------------------------------------------------------------------------
# 2. Small FCC Lennard-Jones crystal: NHC, CSVR, MTK NPT, MTK NPH, MC NPT.
# --------------------------------------------------------------------------
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
crystal_count = crystal_positions.shape[0]
crystal_cell = phx.discretization.PeriodicCell(3.0 * lattice * np.eye(3))
crystal = phx.atomistic.AtomisticSystemPlan(
    np.arange(crystal_count),
    np.ones((crystal_count,), dtype=np.int32),
    np.ones((crystal_count,)),
    units,
    atom_type_ids=np.zeros((crystal_count,), dtype=np.int32),
    cell=crystal_cell,
).prepare()
crystal_program = phx.atomistic.AtomisticPotentialProgram(
    # ty: ignore[invalid-argument-type]
    [phx.atomistic.LennardJonesPotential([1.0], [1.0], 2.0, switch_distance=1.8)]
)
# A 4.8 box exceeds twice the 2.0 classical cutoff with margin for cell
# fluctuations, so no pair has two minimum images inside the cutoff.
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
# Integrator-owned cell methods scale every atom (atomic volume convention).
measure = phx.atomistic.AtomisticPhaseSpaceMeasurePlan(
    crystal, volume_coordinate_convention="atomic"
)
temperature = 0.5
pressure = 1.0
velocity = jnp.sqrt(temperature) * jax.random.normal(
    jax.random.key(7), crystal_positions.shape
)
velocity = velocity - jnp.mean(velocity, axis=0)


def crystal_run(
    integrator: phx.atomistic.AtomisticIntegratorPlan,
    ensemble: phx.atomistic.AtomisticEnsembleKind,
    *,
    steps: int = 40,
    barostat: phx.atomistic.IsotropicMonteCarloBarostatPlan | None = None,
) -> dict:
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
        "method": type(integrator).__name__,
        "ensemble": ensemble,
        "extended_energy_drift": float(after.extended_energy - before.extended_energy),
        "temperature": float(after.temperature),
        "pressure": float(after.pressure),
        "volume": float(after.volume),
        "monte_carlo_moves": int(rollout.final_state.controllers.move_counter),
        "accepted_volume_moves": int(rollout.final_state.controllers.accepted_moves),
    }


step = 1.0e-3
for summary in (
    # Canonical Nose-Hoover chain with fourth-order Suzuki-Yoshida weights.
    crystal_run(
        phx.atomistic.NoseHooverChainPlan(
            step,
            phx.atomistic.NoseHooverChainParameters(
                0.1, chain_length=3, yoshida_suzuki_order=3
            ),
        ),
        "nvt",
    ),
    # Canonical stochastic velocity rescaling (Bussi-Donadio-Parrinello).
    crystal_run(phx.atomistic.StochasticVelocityRescalingPlan(step, 0.1), "nvt"),
    # MTK NPT on the six-dimensional rotation-free symmetric strain chart.
    crystal_run(
        phx.atomistic.MTKNPTPlan(
            step,
            phx.atomistic.MTKBarostatPlan(1.0, chart="triclinic-symmetric"),
            phx.atomistic.NoseHooverChainParameters(0.1),
            phx.atomistic.NoseHooverChainParameters(1.0),
        ),
        "npt",
    ),
    # MTK NPH: no thermostat; inertia_temperature only sets the cell mass.
    crystal_run(
        phx.atomistic.MTKNPHPlan(
            step,
            phx.atomistic.MTKBarostatPlan(1.0, inertia_temperature=temperature),
        ),
        "nph",
    ),
    # Scheduled isotropic Monte Carlo volume moves over a canonical thermostat.
    crystal_run(
        phx.atomistic.BAOABLangevinPlan(step, 1.0),
        "npt",
        barostat=phx.atomistic.IsotropicMonteCarloBarostatPlan(0.01),
    ),
):
    print(summary)
