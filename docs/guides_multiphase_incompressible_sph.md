# Multiphase, IISPH, and DFSPH

`PreparedMultiphaseWCSPHDynamics` composes two continuity-density WCSPH phases
through a bipartite relation. Interface pressure uses density-weighted pressure
and particle-volume pair forces. Dynamic viscosity is harmonically averaged.
Optional color-interface surface tension is a reciprocal pair term. Phase and
global ledgers expose interface force, power, pair count, and action--reaction
defect.

`PreparedIISPH` retains pressure for warm start. A fixed step predicts velocity
and density, applies the matrix-free pressure action, performs projected pressure
iterations with atmospheric free-surface rows, corrects velocity, and advects.
`IISPHStepResult` separates candidate and accepted state, predicted/corrected
density, residual, work, convergence, and success.

`PreparedDFSPH` prepares the pressure factor, applies a divergence projection,
then a constant-density projection. The two residuals, iteration counts, and
convergence flags remain independent. `DFSPHFixedStepMethod` accepts a step only
when both constraints pass and no active particle has a deficient factor.

Both methods initially differentiate fixed iteration counts with frozen surface
and pressure active sets. They do not claim implicit active-set sensitivities.
