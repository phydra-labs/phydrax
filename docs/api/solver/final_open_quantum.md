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

Fixed-step BDF1–5 and adaptive matrix-free BDF1 are separate APIs. Adaptive
HEOM uses a full-step/two-half-step error estimator, accepted/rejected step
history, bounded attempts, and an actual tier-diagonal right preconditioner.
Bath/depth continuation remains an independent approximation axis.

::: phydrax.solver.solve_heom_bdf

::: phydrax.solver.solve_heom_adaptive_bdf

::: phydrax.solver.solve_heom_continuation_grid

## Causal processes and direct memory

Sequential system-memory channels are the physical process representation.
Tomography uses a tangent-space quotient rank and canonical selected-map
fingerprints. Memory refit requires disjoint held-out interventions, nonzero
pre-fit error, and measured improvement. Direct-memory evolution is promoted
only after operator-basis map reconstruction and Choi CP/TP certification.

## Qualification artifacts

Campaign artifact creation, verification, and graduation live under
`tools.open_system_campaigns`, not the solver API. The tooling writes one
complete unverified record, verifies precision linkage and exact independent
reproduction, and only then passes a verified campaign into graduation.

```text
python -m tools.open_system_campaign_matrix --output-directory <directory>
```

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
