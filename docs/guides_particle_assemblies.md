# Particle assemblies and fixed-step programs

`ParticleAssemblyPlan` composes explicitly named particle populations. Stable
identity is `(population ID, particle ID)`. Dynamic population states are packed
by `ParticleAssemblyStateLayout`; static and prescribed boundary populations do
not occupy temporal state blocks.

`DenseBipartiteParticleNeighborhoodPlan` and
`CellListBipartiteParticleNeighborhoodPlan` produce target/source interaction
routes with qualified endpoint identities and fail-closed pair capacity.
`ParticleInteractionLedger` records reciprocal force, power, pair count, and
action--reaction defect.

Integrator-bound methods use `FixedStepProblem` and `solve_fixed_step` rather
than pretending to be autonomous ODE drifts. Each `FixedStepResult` separates
candidate and accepted state, convergence, residual, iterations, and work.
Accepted-step transforms are explicit and composable; density renormalization is
one such transform.

Fixed-step derivatives are derivatives of the fixed scan, iteration counts, and
active sets. Adaptive retries and topology reallocation are not part of this
substrate.
