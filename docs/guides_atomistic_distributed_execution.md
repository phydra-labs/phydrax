# Distributed atomistic execution

PhydraX distributed atomistics uses a fixed-capacity, plan/prepare/state model. The same prepared contract supports a single-device local reference and explicitly supplied JAX collectives. The local reference is the numerical oracle for collective implementations; it is not a communication stub.

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
