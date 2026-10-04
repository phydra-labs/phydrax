# Distributed atomistic execution

PhydraX distributed atomistics provides two routes that share the same fixed-capacity
plan/prepare/state model:

- The slab runtime, `DistributedAtomisticPlan`, handles classical programs. It owns
  transactional migration, phase reductions, distributed PME, and polarization. The
  same prepared contract supports a single-device local reference and explicitly
  supplied JAX collectives. The local reference is the numerical oracle for collective
  implementations, not a communication stub.
- [Owner-local learned execution](#owner-local-learned-execution), through
  `OwnerLocalAtomisticPlan`, evaluates layered learned potentials (native MACE)
  partition-locally on an image-aware graph. Feature halos are exchanged per layer and
  reverse cotangents are returned exactly once.

Every distributed MACE support tuple is an unreleased candidate
(`atomistic.mace-inference.distributed.profile`, `released=False`). No multi-host,
GPU, or scaling claim follows from the evidence described on this page.

## Plan and prepare

A `DistributedAtomisticPlan` binds a `PreparedAtomisticSystem` to a `ParticleDomainDecompositionPlan`. The decomposition currently uses stable slabs along the first Cartesian axis. Canonical particle identity never depends on shard placement.

The plan fixes every compiled capacity:

- `partition_capacity`: owned particles per partition;
- `halo_capacity`: particles per directed source/destination route;
- `migration_capacity`: owner changes in one candidate transition;
- `thermostat_capacity`, `barostat_capacity`, and `bias_capacity`: extended-state vectors;
- a `DistributedOutputMask` and `DistributedReductionPolicy`;
- optional `DistributedPMEPlan` and `DistributedPolarizationPlan` contracts.

```python
box = phx.discretization.ParticleBox(
    [0.0, 0.0, 0.0],
    [8.0, 8.0, 8.0],
    periodic_axes=(True, True, True),
)
decomposition = phx.discretization.ParticleDomainDecompositionPlan(
    4, 1.2, box
)
plan = phx.atomistic.DistributedAtomisticPlan(
    prepared_system,
    decomposition,
    partition_capacity=1024,
    halo_capacity=256,
    migration_capacity=128,
    output_mask=phx.atomistic.DistributedOutputMask(atom_energy=False),
    reduction=phx.atomistic.DistributedReductionPolicy("deterministic"),
    pme=phx.atomistic.DistributedPMEPlan((96, 96, 96)),
    polarization=phx.atomistic.DistributedPolarizationPlan(
        maximum_iterations=100,
        tolerance=1.0e-7,
    ),
)
runtime = plan.prepare_runtime()
state = runtime.initialize(
    positions,
    momenta=momenta,
    rng_key=rng_key,
    run_id="production-42",
    replica_id="replica-0",
    epoch_id="epoch-7",
)
```

Preparation produces stable `owner`, `permutation`, `inverse_permutation`, and `block_bounds` arrays. Owned slots, directed halo routes, receive routes, and local layouts are padded to their planned shapes. Padding indices are `-1` and always accompanied by masks.

## Ownership, halo exchange, and force return

Ownership and routing are functions of positions, active masks, box policy, and the immutable decomposition plan. Periodic positions are wrapped for ownership. An active nonperiodic coordinate outside the domain fails the state.

`exchange_distributed_halos(runtime, state, values)` gathers a canonical particle payload into padded send routes. The local-reference runtime transposes the source/destination route axes to realize the receive layout. `reverse_halo_force_return` performs deterministic local-reference accumulation. `reverse_distributed_halo_force_return(runtime, state, forces)` invokes the explicit reverse collective before accumulating contributions on owner ranks. Masked padding never contributes, and the sum of returned force equals the sum of valid received force.

Halo and ownership overflow are evidence, not truncation success. Padded arrays remain valid, while `state.status.successful` is false.

## Migration is transactional

A discrete ownership change is a candidate/evaluation/commit transition:

```python
candidate = phx.atomistic.propose_distributed_migration(
    plan, state, proposed_positions
)
next_state = phx.atomistic.commit_distributed_migration(
    plan, state, candidate
)
```

The candidate records a fixed-capacity migration list, count, finite evidence, rebuilt routes, and its exact complete source state. Commit applies positions and ownership only when every source array and run/replica/epoch identity still matches and all ownership, halo, migration, and finite checks pass. Otherwise it returns the target state's physical/decomposition arrays and marks it unsuccessful. A successful commit increments `decomposition_epoch` exactly once. Collective migration is rejected because this runtime does not define continuation-payload migration collectives; it never changes collective ownership without communicating all continuation fields.

Differentiation through a trajectory is valid only while the discrete ownership, route topology, and event schedule are fixed. Migration decisions are not silently differentiated.

## Evaluation phases and outputs

`evaluate_distributed_atomistic` accepts an `AtomisticPotentialEvaluation` for the direct phase, an optional sparse correction, and optional state-bound `DistributedReciprocalEvidence`. It partitions atom energy and force by canonical ownership, accounts for any global energy residual once, and performs the declared reduction. The returned `DistributedPhaseEvidence` reports energies and globally reduced success for direct, sparse-correction, reciprocal, and final reduction work.

A reciprocal evaluation requires a configured `DistributedPMEPlan` and must first pass through `certify_distributed_reciprocal(runtime.pme_runtime(), state, evaluation)`. The resulting evidence is consumed by `evaluate_distributed_atomistic` or `distributed_particle_mesh_electrostatics(runtime, state, evidence)`. A polarization plan prepares the warm-start capacity; `certify_distributed_polarization(runtime.polarization_runtime(), state, dipoles, residual, iterations)` requires a nonnegative finite residual and nonnegative integral iteration count. Both evidence objects bind positions, cell, step/decomposition epochs, and run/replica/epoch identities. Reusing evidence for another state fails closed.

The output request is static. Unrequested energy, force, virial, atom-energy, or partition-energy arrays retain their documented fixed shapes and are filled with zeros. `result.available` records the five requested outputs in that order. No `None`-dependent compiled branch is introduced.

`DistributedReductionPolicy("deterministic")` accumulates partitions in increasing index order. `"compensated"` uses the same fixed order with compensated accumulation. `"fast"` selects the backend reduction. Collective runtimes additionally invoke the supplied global sum callable.

`halo_short_range_evaluate(plan, state, potential, neighborhood)` is a labeled
global-evaluate-then-mask reference for short-range classical programs. It evaluates
the complete prepared program on the global local-reference state, then uses owner
masks to attribute the outputs to slabs. It is not owner-local execution. It refuses
collective states, reciprocal terms, and a halo radius smaller than the program cutoff.
Layered learned potentials use
[owner-local learned execution](#owner-local-learned-execution).

## Domain and load evidence

`distributed_domain_evidence` combines owned and halo counts with optional per-partition pair and iterative work. It reports weighted work, imbalance, finite/domain checks, and each capacity check. A nonfinite input, outside-domain particle, or capacity failure makes `successful` false.

## Collective execution

Multi-device execution must be requested explicitly:

```python
plan = phx.atomistic.DistributedAtomisticPlan(
    prepared_system,
    decomposition,
    execution_mode="collective",
)
operations = phx.atomistic.DistributedCollectiveOperations(
    exchange_routes,
    reverse_exchange_routes,
    global_sum,
    partition_index=local_partition,
    collective_id="mesh-axis-dp",
)
runtime = plan.prepare_runtime(operations)
```

`exchange_routes(send, mask)` must be a JAX-traceable callable mapping rank-local `(source, destination, slot, ...)` sends to `(destination, source, slot, ...)` receives. `reverse_exchange_routes(receive, mask)` performs the inverse owner-directed communication. `global_sum(value)` must sum one rank-local contribution across the mesh, and `partition_index` identifies that contribution. Evaluation all-reduces each particle, energy, virial, phase-ledger, and failure contribution exactly once. PhydraX deliberately supplies none of these operations and rejects collective preparation without all of them. APIs without a prepared runtime, including the short-range convenience evaluator, reject collective states rather than executing a local fallback.

## Checkpoints

`DistributedAtomisticState` contains all continuation state:

- positions, momenta, and the physical cell;
- ownership, permutation, fixed local/halo routes, and decomposition epoch;
- partition momentum and energy;
- thermostat, barostat, polarization warm-start, and bias state;
- the canonical RNG key and step index;
- plan, prepared-runtime, run, replica, and epoch identities.

`checkpoint_distributed_atomistic` creates an in-memory checkpoint whose identity digests every continuation-relevant array plus all static identities. `restore_distributed_atomistic_checkpoint` rejects another runtime or a mismatched payload identity. Checkpoint identity is a host-side provenance operation; numerical state evolution remains JAX compatible.

## Owner-local learned execution

`OwnerLocalAtomisticPlan` executes a layered learned potential on owned receivers.
Currently that means native `phx.nn.atomistic.MACEPotential`, the only implementation
of the layered interaction contract. The plan covers one fixed periodic cell. Its
static contract binds:

- a `phx.discretization.spatial.DistributedOwnershipPlan` whose `MortonAddressPlan`
  addresses the unit fractional box `[0, 1)^3` with the cell's periodic axes;
- a `phx.discretization.FractionalOwnerPartition` with one fractional region per
  owner. Triclinic cells are partitioned in lattice coordinates, not as diagonal
  Cartesian slabs. On nonperiodic axes of a partially periodic cell, the outer regions
  extend to infinity;
- a complete `PeriodicImageStencil` from `cell.image_stencil(cutoff + skin, ...)`.
  Its radius must cover `cutoff + skin`, and its `fractional_excursion` must be at
  least 1 so wrapped endpoints are admitted;
- a `phx.sparse.StreamedRelationPlan`. Each owner's receiver relation is prepared once
  per topology epoch and reused by every interaction layer;
- the capacities `alias_capacity`, `edge_capacity`, `halo_capacity`,
  `migration_capacity`, and `message_capacity_bytes` (one interaction's halo message
  per owner). Exceeding any of them refuses the topology, evaluation, or migration;
  nothing is truncated.

The plan also requires a three-dimensional cell, and the model cutoff must not exceed
the plan `cutoff`.

### Execution semantics

Each owner holds its receiver rows and builds its edges from periodic image aliases
exchanged under the fractional partition. An edge is `(receiver row, source column, n)`
with `d = x_receiver - x_source + n @ H`. A periodic alias reuses the physical atom's
deduplicated halo column. Repeated images of one source therefore stay distinct edges
with their full multiplicity, and nonzero self images are included.

`evaluate_owner_local_atomistic` runs one interaction after another:

1. Before each interaction, the source payload of owned rows crosses the canonical
   `DistributedHaloPlan` gather once.
2. The reverse sweep replays the interactions in reverse order from retained
   inter-layer node states and applies the model's ordinary JAX local VJP.
3. Each layer's source-column cotangents return to their owners exactly once through
   the halo transpose.

Geometry cotangents reach receiver rows directly and source rows through the same
transpose. Each owner's edge partial of the shared cell derivative is reduced in
owner order into one `strain_gradient` (row cell `H' = H @ F.T`).
`stress = sym(strain_gradient) / volume` is available only for a fully periodic 3D
cell; otherwise `stress_available` is false.

Because the route is ordinary differentiable JAX, `owner_local_loss_gradient`
differentiates the owner-local forces themselves. It returns the exact
parameter-lane gradient of `w_E (E - E_ref)^2 + w_F sum |F - F_ref|^2 +
w_S |stress - stress_ref|^2`, with mixed coordinate/parameter derivatives through every
halo exchange. Stress supervision requires a stress reference and a fully periodic 3D
cell.

Evaluation fails closed. A stale owner epoch, a refused topology, a failed streamed
relation, non-finite output, or an expired displacement certificate
(`2 max|x - x_ref| > skin`) sets `successful` false and poisons every numerical output
with NaN. `OwnerLocalExecutionStatus` records which check failed.

```python
import jax
import numpy as np
import phydrax as phx
from phydrax.execution import ExecutionRuntime

jax.config.update("jax_enable_x64", True)

units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
cutoff, skin = 2.4, 0.3
vectors = np.array([[2.2, 0.0, 0.0], [0.6, 2.5, 0.0], [0.3, 0.4, 2.8]])
architecture = phx.nn.atomistic.MACEArchitecture(
    species=(1, 8),
    cutoff=cutoff,
    radial_basis_count=4,
    cutoff_power=5,
    channel_count=4,
    hidden_degree=1,
    edge_degree=2,
    interactions=("real-agnostic", "real-agnostic-residual"),
    correlations=(2, 2),
    radial_widths=(8,),
    readout_width=4,
    average_neighbor_count=8.0,
)
model = phx.nn.atomistic.MACEPotential(
    units.scale,
    architecture,
    atomic_energies=np.array([-0.3, -1.1]),
    key=jax.random.key(0),
)

owners = 2
cell = phx.discretization.PeriodicCell(vectors)
address = phx.discretization.spatial.MortonAddressPlan(
    (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 8, periodic_axes=(True, True, True)
)
group = ExecutionRuntime.current().child_groups(len(jax.devices()))[0]
ownership = phx.discretization.spatial.DistributedOwnershipPlan(
    address, group, 8, owner_lanes=owners
)
plan = phx.atomistic.OwnerLocalAtomisticPlan(
    ownership,
    phx.discretization.FractionalOwnerPartition(cell, (owners, 1, 1)),
    cell.image_stencil(cutoff + skin, maximum_image_count=4096),
    streaming=phx.sparse.StreamedRelationPlan(
        receiver_tile=4,
        edge_tile=32,
        channel_capacity=model.configuration.message_payload_width(),
    ),
    cutoff=cutoff,
    skin=skin,
    alias_capacity=512,
    edge_capacity=512,
    halo_capacity=8,
    migration_capacity=8,
    message_capacity_bytes=1 << 20,
)

rng = np.random.default_rng(4)
positions = rng.uniform(0.0, 1.0, (6, 3)) @ vectors
atomic_numbers = np.array([1, 8, 1, 8, 1, 8], dtype=np.int32)
species = np.asarray(
    model.species_indices(atomic_numbers, np.ones(atomic_numbers.shape, dtype=bool))
)
state = phx.atomistic.prepare_owner_local_atomistic(
    plan,
    model,
    positions,
    species,
    rng_key=jax.random.key(7),
    velocities=np.zeros_like(positions),
    thermostat_state=np.zeros((2,)),
)
evaluation = phx.atomistic.evaluate_owner_local_atomistic(plan, model, state)
if not bool(evaluation.successful):
    raise RuntimeError(f"owner-local evaluation failed: {evaluation.status}")
forces = state.layout.collect(evaluation.forces)  # logical atom order
stress = evaluation.stress  # fully periodic cell: stress_available is True

fit = phx.atomistic.owner_local_loss_gradient(
    plan,
    model,
    state,
    -2.0,
    np.zeros_like(positions),
    energy_weight=0.5,
    force_weight=2.0,
)
```

`prepare_owner_local_atomistic` is the host ingress. It wraps stored coordinates into
the cell and records the integer image counts that recover unwrapped trajectories. It
assigns owners from wrapped fractional coordinates, distributes every per-atom input in
logical atom order, and builds the first topology epoch. A refused first topology
raises with its evidence instead of returning an unusable state. `species` are the
model's native species indices.

With `owner_lanes`, the owner regions execute as named `vmap` lanes of one device. They
use the same collectives, packets, capacities, and epochs as device ownership. This
lane reference is a numerical oracle: for two- and three-layer native MACE it is
compared with an independent single-device global MACE on brute-force image routes. It
is not hardware evidence. Omitting `owner_lanes` maps one owner per device of the
`ExecutionGroup` through `shard_map`. On forced host CPU devices
(`XLA_FLAGS=--xla_force_host_platform_device_count=2`) that route demonstrates
functional collective parity only. Multi-host, GPU, and performance tuples are
unqualified.

### Dynamics, migration, and model updates

The owner-local state is a complete accepted continuation state. Its owner-blocked rows
hold:

- stored positions and integer image counts;
- velocities, masses, and species;
- the force cache and the step it belongs to;
- declared per-atom payload (for example constraint membership or per-atom thermostat
  and bias histories).

Replicated state holds the typed RNG key, thermostat, bias, and constraint state, and the
step index. `state.with_dynamics(positions, velocities, step_index=..., forces=...)`
advances the dynamics within the current owner and topology epoch. Inactive padding
rows stay exactly zero.

`rebuild_owner_local_atomistic(plan, model, state)` is one transaction:

1. Every atom moves, with its complete continuation payload, to the owner of its
   wrapped fractional coordinate. Stable IDs are preserved.
2. Stored positions wrap by whole lattice translations that are added to the image
   counts.
3. The new topology epoch is discovered.

Migration overflow, epoch inconsistency, alias/edge/halo overflow, or non-finite
positions on any owner return the accepted state unchanged. The failed attempt's
evidence is kept in `transition.migration`
(`phx.discretization.spatial.DistributedMigrationEvidence`) and
`transition.topology` (`OwnerLocalTopologyEvidence`). `transition.committed` reports
the outcome.

`rebind_owner_local_model` binds an updated numeric model revision and marks the force
cache stale. The revision is checked at ingress, at restore, and at rebind. It is not
checked by hashing parameters on every evaluation.

`checkpoint_owner_local_atomistic` is the explicit host egress of every owner-blocked
and replicated continuation array, including the accepted topology routes and
reference positions. `restore_owner_local_atomistic` requires the same plan, the same
model revision, and an intact content identity. It rebuilds the layout, halo plan, and
prepared relations through their constructors, so a restored run reproduces the
uninterrupted accepted evolution.

```python
transition = phx.atomistic.rebuild_owner_local_atomistic(plan, model, state)
if not bool(transition.committed):
    raise RuntimeError("owner-local migration or topology was refused")
state = transition.state
checkpoint = phx.atomistic.checkpoint_owner_local_atomistic(plan, state)
restored = phx.atomistic.restore_owner_local_atomistic(plan, model, checkpoint)
```

Integer route discovery, image enumeration, owner assignment, and topology epochs are
not differentiated. Derivatives hold within one accepted topology epoch. Changing the
ownership partition, capacities, or streamed plan changes `plan_id`. States and
checkpoints from another plan are refused.
