# Exact targeted free-energy maps

A targeted map transports configurations between two normalized reduced-potential
measures while retaining the exact Jacobian contribution. Estimation remains in the
existing FEP and BAR functions.

`AbstractBijector` is the single invertible-array contract. `TargetedMapPlan`
binds one shape-preserving bijector to an event shape and content-addressed
architecture and parameter identities. `AffineFlowBijector` supplies a native
exact affine chart with forward, inverse, and log-determinant actions.

For source configuration x and map M, `evaluate_targeted_work` computes

    W_forward = u_target(M(x)) - u_source(x) - log|det J_M(x)|.

For target samples it also computes the inverse-map work. The result retains mapped
samples, both log determinants, round-trip residuals, per-sample validity, and the
problem identity. Convert the oriented observations to `ReducedWorkDataset` before
calling `free_energy_perturbation` or `bennett_acceptance_ratio`. The dataset must
identify the source/destination potentials and measures, sample lineage, exact sampling
qualification or finite bias bound, map, producer, and run; estimators do not accept
anonymous arrays.

## Training

`fit_targeted_free_energy_map` minimizes declared forward and optional reverse mean
work plus optional displacement regularization. Model selection uses separate
validation samples. The result reports exact-map validity and forward/reverse
importance effective sample sizes. Each update is one training-kernel attempt: an
invalid training evaluation carries no support and a nonfinite one rolls back, so
neither is committed. Training then stops with `valid=False` and keeps the last
accepted map.

Hutchinson traces and approximate continuous-flow densities do not satisfy this
contract. A nonfinite map, potential, log determinant, or inverse round trip fails the
work evaluation.

## Atomistic coordinates

`CenterOfMassPreservingBijector` applies an exact internal bijector in a mass-weighted
translation-free basis while passing center-of-mass coordinates unchanged. The mass
scalings cancel between chart and inverse, so the only nonconstant Jacobian is the
internal bijector's exact Jacobian.

`ControlledHamiltonianReducedPotential` adapts one state of a prepared controlled
Hamiltonian and one matching neighborhood to the reduced-potential contract. The
Cartesian adapter rejects periodic tori and controls that decouple a region from its
environment; those cases do not define the required common normalized Cartesian
density. State-dependent masses, constraints, changing event dimension, and unsupported
virtual geometry are rejected during controlled-Hamiltonian preparation.
