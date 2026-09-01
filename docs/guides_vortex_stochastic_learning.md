# Stochastic and learned vortex methods

## Random vortices

`RandomVortexSolverPlan` advances full vortex ensembles with named Brownian
realizations, real deterministic velocity backends, free/periodic/reflecting/
absorbing boundaries, forcing-weight recurrence, and optional antithetic noise.
Evidence includes absorbed/reflected counts, weight balance, displacement moments,
and antithetic weak-moment residual. Core and volume travel with every ensemble;
no unit-core substitute is inserted.

## Native vorticity learning

`NativeVorticityLearningPlan` optimizes a real Phydrax model through the native
minimization API. The weighted objective combines sample mismatch, circulation,
and optional dissipation constraints. The result retains the native optimizer
status and separate scientific evidence.

`PeriodicVorticityReconstructionPlan` recovers incompressible velocity and
optional gradients through Fourier inversion, including mean-vorticity and
divergence defects. Bounded learned reconstructions should use the MAC/immersed
hybrid rather than an unqualified Python relaxation loop.

`ConstrainedLearnedClosure` removes mean strength production, rejects positive
energy transfer unless permitted by its model, and reports a normalized
out-of-distribution boundary. It never invents a classical fallback.

## Assimilation

`VortexObservationSet` represents velocity, vorticity, pressure/load, or particle
track data with explicit uncertainty and a fixed linear observation operator.
`VortexDataAssimilationPlan` combines observations and prior precision through a
native least-squares solve and retains innovation, weighted residual, prior
residual, and solve evidence.

## Persistence

`VortexCheckpointPlan` writes checksum-validated pickle-free array archives,
including source state, event journal, RNG realization, accepted times, topology,
and backend IDs. `VortexReplayPlan` joins fixed-topology epochs through exact
transition callbacks and optional pullbacks. Particle/ring export payloads retain
stable IDs, lineage, connectivity, circulation, core, volume, and age.
