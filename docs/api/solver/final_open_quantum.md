# Open-system fail-closed closure

Production promotion requires quantified approximation evidence and frozen
array archives. Missing or unknown physicality never becomes success.

## Correctness cutover

- dense Lindblad accepts positive-semidefinite trace-one initial states;
- pseudomode embeddings preserve caller initial states exactly;
- memory/TCL execution validity is separate from map physicality;
- process physicality includes the initial density;
- fixed-step jump probabilities above the declared limit fail;
- Fock evidence includes state normalization and explicit thresholds;
- Gaussian state/channel `hbar` conventions must match.

## Generic quantum jumps

Quantum state-vector dynamics adapt to `JumpDifferentialProblem` and
`solve_jump_differential`; channel intensities and reset maps remain matrix-free.

::: phydrax.solver.solve_quantum_jump_generic

## Local Lindblad channels and LPDO

Finite local Lindblad terms are exponentiated, reshuffled into a Choi matrix,
and converted into Kraus operators only after CP/TP certification.

::: phydrax.tensor_network.prepare_local_lindblad_channel

::: phydrax.solver.diagnose_purified_stationarity

## HEOM

Capacity-checked scaled topology, BDF1–5 implicit solves, tier-block
preconditioning evidence, and independent bath/depth continuation are exposed
separately.

::: phydrax.solver.solve_heom_bdf

::: phydrax.solver.solve_heom_continuation_grid

## Causal processes

Sequential system-memory channels are the physical process representation.
Tomography reports support and identifiability; memory compression is accepted
only when the compressed process remains CPTP.

## Artifacts

::: phydrax.solver.write_open_system_artifact

::: phydrax.solver.read_open_system_artifact

## Permanent claim boundaries

- Representation closure is not integration exactness.
- `valid` is not convergence, physicality, or promotion.
- Saved-state positivity is not complete positivity between saved times.
- Top-level Fock population, ADO norm, discarded singular weight, root
  residual, ESS, likelihood, and small-reference agreement are diagnostics,
  not universal error bounds.
- MPS and sampled neural trajectories remain approximate unravelings.
- A finite stationarity window does not establish uniqueness or global mixing.
- Process fitting does not identify latent memory gauge without a quotient
  analysis and complete intervention design.
- Deterministic replay establishes reproducibility, not statistical accuracy.
