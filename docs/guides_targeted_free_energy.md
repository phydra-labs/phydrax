# Exact targeted free-energy maps

A targeted map transports configurations between two normalized reduced-potential
measures while retaining the exact Jacobian contribution. Estimation remains in the
existing FEP and BAR functions.

`AbstractBijector` is the single invertible-array contract. `TargetedMapPlan` binds one
shape-preserving bijector to an event shape and content-addressed architecture and
parameter identities. `FlowJAXBijectionAdapter` accepts only unconditional FlowJAX
bijections exposing exact forward and inverse log determinants.

For source configuration x and map M, `evaluate_targeted_work` computes

    W_forward = u_target(M(x)) - u_source(x) - log|det J_M(x)|.

For target samples it also computes the inverse-map work. The result retains mapped
samples, both log determinants, round-trip residuals, per-sample validity, and the
problem identity. Pass the resulting work arrays to `free_energy_perturbation` or
`bennett_acceptance_ratio`; there is no duplicate estimator.

## Training

`fit_targeted_free_energy_map` minimizes declared forward and optional reverse mean
work plus optional displacement regularization. Model selection uses separate
validation samples. The result reports exact-map validity and forward/reverse
importance effective sample sizes.

Hutchinson traces and approximate continuous-flow densities do not satisfy this
contract. A nonfinite map, potential, log determinant, or inverse round trip fails the
work evaluation.

## Atomistic coordinates

`CenterOfMassPreservingBijector` applies an exact internal bijector in a mass-weighted
translation-free basis while passing center-of-mass coordinates unchanged. The mass
scalings cancel between chart and inverse, so the only nonconstant Jacobian is the
internal bijector's exact Jacobian.

`AlchemicalEndpointReducedPotential` adapts one prepared alchemical endpoint to the
reduced-potential contract. The initial implementation requires identical active
support with no dummy particles. Constrained manifolds, periodic tori, changing event
dimension, and unrestrained dummy coordinates are rejected rather than treated as
Cartesian densities.
