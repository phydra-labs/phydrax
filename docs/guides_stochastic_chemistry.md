# Stochastic chemistry

`ChemicalJumpProcess` lowers a compatible prepared chemical mechanism to the existing
finite-activity jump-process protocol. It reuses exact SSA, next-reaction, Poisson-clock
replay, and finite-state generator infrastructure rather than introducing a chemistry-
specific stochastic solver.

Discrete compatibility requires integral stoichiometry, mass-action reaction orders,
nonnegative integer counts, a positive system measure, and rate laws with an explicit
jump interpretation. Reversible reactions become separate forward and reverse
channels. Propensities use falling-factorial combinatorics and preserve the mechanism's
element and charge invariants.

## Relaxed trajectories

`RelaxedChemicalJumpPlan` is an explicitly biased continuous relaxation. Smooth
selector weights replace discrete channel selection and Gaussian channel weights
replace integer jumps. It exists for gradient-based design—not as an exact SSA mode.

Every relaxed result reports selector and jump controls through its plan identity,
nonintegral-state residual, minimum state, accepted event count, event-capacity
exhaustion, and finite-state status. `relaxed_exact_moment_discrepancy` compares relaxed
and exact sample moments. Rare-event and distributional claims require exact SSA
qualification at the selected sharpness.
