# Particle distributed execution and replay

`ParticleBackendPolicy` distinguishes pure-JAX, Pallas, and Triton execution and
fast, deterministic, or compensated reduction semantics.
`ParticleKernelRequestPlan` declares which pair features are required and whether
they are materialized. `certify_particle_precision` evaluates reduced-precision
results in certification precision.

`ParticleDomainDecompositionPlan` provides the reference partition, owned/halo
masks, halo update/sum, migration, and load-balance evidence. The reference path
is the semantic authority for accelerator and multi-device lowerings.

`ParticleBenchmarkRegistry` stores evidence-qualified benchmark records.
`ParticleQualificationArtifact` writes a versioned commercial artifact.
`ParticleReplayPacket` stores failure state, last accepted state, method/problem
identity, time, step, and status; replay runs from the accepted state rather than
the partial candidate. `ParticleSupportMatrix` records the qualified method,
dimension, backend, precision, maturity, and evidence combination.
