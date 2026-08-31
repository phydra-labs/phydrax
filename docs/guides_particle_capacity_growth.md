# Particle capacity growth

Particle capacity can grow automatically between accepted execution epochs. Array shapes remain fixed inside every compiled epoch; no JAX program attempts dynamic-shape allocation.

## Execution epochs

`ParticleExecutionEpoch` binds prepared DEM dynamics, accepted state, occupancy history, retired slots, and one structural identity. `ParticleCapacityRequest` reports the additional slots required by insertion or fragmentation. `ParticleCapacityGrowthPolicy` chooses the next capacity using a geometric factor, minimum increment, and hard maximum.

`grow_particle_execution_epoch` performs one transaction:

1. reserve new monotone particle IDs;
2. prepare a larger particle support, body set, neighborhood, pair-identity space, contact model, and caches;
3. transfer kinematics, body properties, contact history, wall history, and ledgers by stable identity;
4. initialize appended slots to finite inactive state;
5. reevaluate loads against the new prepared structure;
6. accept only when migration, balance, and geometry checks pass.

Existing particle-pair identities are structured from physical endpoint IDs and therefore do not depend on capacity or slot rank.

## Occupancy and retirement

Prepared slots and runtime occupancy are separate. `ParticleExecutionEpoch.ever_occupied` records whether a reserved identity has been used. `retired` identities cannot be reused, preventing stale contact history from being attached to a new physical particle.

`insert_reactive_particles_with_growth` and `fragment_particle_with_growth` grow first when necessary, then retry the original event without partially committing it.

## Segmented execution

`advance_particle_epoch_segments` runs fixed-shape segments separated by explicit transitions. Replay records contain every epoch and transition identity. `pullback_particle_epoch_transition` transfers cotangents for persistent state; new padding receives zero sensitivity. Growth decisions remain discrete stopped-gradient events.

## Failure semantics

A hard maximum produces `CAPACITY_LIMIT_REACHED`. The old epoch remains accepted and the request remains inspectable. Unsupported neighborhood growth, failed history migration, nonfinite state, or balance loss similarly rejects the whole transition.

Run `examples/growing_reactive_particle_pool.py` and `tools/particle_capacity_growth_qualification.py` for the complete insertion path.
