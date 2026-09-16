# Thermodynamic equilibrium

Phydrax equilibrium solvers operate on one immutable homogeneous Helmholtz model and return explicit solver, phase, root, stability, and derivative evidence.

## Compiled pure-component saturation curves

`AntoineSaturationPressurePlan` is the bounded scalar saturation law used by
compiled evaporation and phase-transfer kernels. Construction requires an
explicit temperature interval that remains above the Antoine pole. Forward
pressure and analytic inverse-temperature evaluations report domain margins,
finite evidence, and success; extrapolation is never certified.

This local fixed-shape law is distinct from multicomponent flash. A host-level
flash may prepare parameters or tables, but `FixedTwoPhaseTPFlashPlan` is not
called from a cell residual or JIT-compiled phase-transfer source.

## Ideal-gas chemical equilibrium

`ChemicalEquilibriumPlan` solves ideal-phase TP, TV, HP, UV, SP, or SV equilibrium subject to elemental and charge balances. Gas activities use the same explicit standard pressure as thermodynamic reverse rates; liquid/solid phases use ideal within-phase activities. Results retain active phases, conserved-property residual, optimization evidence, and fixed-active-set derivative validity.

## Peng–Robinson phase roots

`PengRobinsonResidualHelmholtzTerm` implements the PR78 alpha convention with quadratic attractive-parameter and linear covolume mixing. `peng_robinson_roots` returns three fixed root slots, validity and mechanical-stability masks, pressure residuals, Gibbs values, and root-separation evidence. Invalid or unstable roots are never silently selected.

## Tangent-plane stability

`TPDSearchPlan` searches deterministic feed, Wilson, inverse-Wilson, component-enriched, dense-root, and dilute-root starts. A negative result is an instability witness. A nonnegative result means no instability was found under the declared finite search; it is not a proof of global stability.

## Fixed two-phase TP flash

`FixedTwoPhaseTPFlashPlan` uses ordered dense and dilute phase slots. It first runs TPD, then solves Rachford–Rice and fugacity consistency through damped fixed-point refinement. The result always has fixed phase/component shapes and reports material and fugacity residuals.

The initial scope is nonreactive, non-electrolyte, two-phase phi–phi equilibrium. Pure-component coexistence, critical coalescence, bubble/dew endpoints, root switching, phase appearance, and active-component changes do not have an ordinary two-sided derivative.

Equilibrium orchestration is a host-level operation. It must not be called inside a CFD flux or compiled DAE residual.
