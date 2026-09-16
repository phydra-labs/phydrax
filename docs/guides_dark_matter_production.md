# Dark-matter production profiles

Dark-matter production is a set of separately qualified profiles, not one universal
state or solver. Each profile fixes its physics, support, execution topology, precision,
capacities, differentiation contract, output products, and scientific claim.

## Claim profiles

`phydrax.applications.cosmology` exposes independent `ScientificClaimProfile` factories
for periodic wave DM, rare equal-weight SIDM, mixed wave/particle and
wave/particle/gas evolution, periodic wave AMR, differential, weighted, frequent and
spherical gravothermal SIDM, and inelastic SIDM. A signed promotion for one profile
never promotes another.

Promotion proceeds through experimental, numerically qualified, scientifically
qualified, and production levels. Calibration/model-selection cases determine frozen
criteria before locked evaluation. Every profile records supported and refused claims,
required metrics, differentiation semantics, and evidence artifacts.

## Production runtime

`PeriodicWaveProductionMethod` and `RareSIDMProductionMethod` adapt prepared physics to
the existing fixed-step production runtime. They retain one increasing scale-factor
schedule, accepted-step transaction, resource policy, trigger cursor, and bounded
streaming evidence. The SIDM adapter additionally retains the canonical random root and
semantic event epoch.

Analysis output and restart are different products:

- `WaveSimulationSnapshot`, `ParticleSimulationSnapshot`, `GasSimulationSnapshot`, and
  `CommonGravitySimulationSnapshot` are observation products.
- `CosmologyOutputBundle` binds those products at one physical time level.
- `DarkMatterRestartSnapshot` binds exact state, schedule/output cursors, stable IDs,
  masks, incarnations and lineage, random/event epoch, topology/partition, physics,
  support and artifact identities.
- `DarkMatterCheckpointContract` uses the existing runtime checkpoint envelope and
  durable stores. Restore rejects changed physics or support.

Checkpoints are published only at accepted synchronization boundaries. Incomplete shard
or manifest publication is not readable.

## Correlated component initial conditions

`PrimordialModeRealization` and `ComponentTransferMatrixProduct` apply one stable latent
mode realization to component density/current transfers, preserving all component
auto- and cross-covariances, gauge, units, provider and artifact identity.

`MixedInitialConditionPlan` reuses the admitted LPT particle owner. Wave seeds remain
distinct:

- `WavePhaseSeedPlan` reconstructs a mean-zero phase only from a compatible irrotational
  mass current and reports curl, node, gauge and de Broglie evidence.
- `SolitonSeedPlan` creates a normalized localized profile.
- `VortexSeedPlan` creates integer winding and explicit node/core evidence.
- `ImportedComplexFieldValidationPlan` admits a supplied complex field without
  discarding source rights or conventions.

Density-only phase invention, circulation through nodes, and gauge/unit mismatch fail
closed.

## Shared fixed-grid gravity

`MixedDensityAssembler` creates named wave, particle and gas comoving-density sources on
one prepared grid. `SharedPeriodicGravityPlan` sums them, subtracts the mean once, runs
one periodic Poisson solve, and returns the one potential plus matched particle/gas
forces and source/force evidence.

`WaveParticleCosmologyPlan` and `WaveParticleGasCosmologyPlan` apply endpoint-consistent
wave potential half-kicks, wave kinetic drift, particle KDK, and the admitted gas
homogeneous/source operation. Each interval records time levels, component mass/work,
wave norm/phase, particle transfer, gas positivity, Poisson/gauge/net-force residuals,
and whole-state rollback.

`DistributedMixedExecutionPlan` prepares real `NamedSharding` execution over an
`ExecutionPlan`. `PreparedDistributedMixedExecution` owns fixed-capacity stable-ID
particle migration/ghost exchange, unique particle deposit/gather, sharded common
density and periodic Poisson, sharded wave and gas updates, global atomic acceptance,
and exact-coverage distributed checkpoints with changed-sharding restore. Resource,
collective, route and receive-capacity failures roll back without host gathering.

## Output and restart invariants

- One component cannot commit while another rolls back.
- The common scale factor and potential time level are explicit.
- External products remain stop-gradient.
- Topology, event selection, particle allocation, reactions and lineage remain
  nondifferentiable.
- A snapshot is never a restart payload.
- Runtime identities include all static physics and support choices.
