# Monolithic fluid-particle Newton coupling

`ReactiveMonolithicCouplingPlan` solves fluid storage, particle momentum, and intraparticle conversion as one matrix-free nonlinear root.

## Scope

The monolithic unknown contains:

- fluid velocity, temperature, and species concentration;
- particle velocity;
- every particle-batch internal energy and species amount.

Particle positions, collision topology, wall-feature ownership, AMR masks, insertion, and fragmentation remain frozen during one Newton solve. Crossing one of those event surfaces rejects or splits the macro window rather than differentiating through a topology change.

## Fluid contract

`CellwiseReactiveFluidImplicitPlan` supplies positive momentum, heat, and species storage coefficients. It is the first concrete implicit fluid contract. Prepared finite-volume adapters can expose richer internal fluid residuals through the same state/residual-space structure without changing exchange semantics.

## Coupled residual

The backward-Euler residual includes:

```text
fluid content change − step × deposited source
particle momentum change − step × hydrodynamic force
particle extensive-state change − step × transport/reaction/exchange source
```

`ParticleContinuumExchangePlan` is evaluated once at the candidate state. Its particle and fluid sources are exact opposites. Contact heat and radiation enter through explicit candidate source arrays.

## Newton lifecycle

`prepare_reactive_monolithic_step` constructs a canonical `NonlinearSystemProblem` over `PyTreeSpace`, then calls the native prepared Newton stack. Numerical stages may be refreshed only while capacity, mesh topology, transfer routes, and active sets remain unchanged.

`ReactiveMonolithicSolverPlan` supports Newton-Krylov or trust-region globalization and three preconditioner policies:

- `LOCAL_BLOCK`;
- `BLOCK_FACTORIZATION`;
- `SCHUR_COMPLEMENT`.

All paths remain matrix-free. Particle blocks use local mass and conversion scaling; coupled modes add the particle-to-fluid exchange feedback.

## Acceptance and differentiation

A candidate commits only when nonlinear convergence, physical admissibility, route validity, finiteness, and momentum/energy/species closure all pass. Otherwise fluid, particle conversion, velocity, and ledgers roll back together.

`reactive_monolithic_vjp` differentiates a converged fixed-route root. Species exhaustion, temperature-bound proximity, contact changes, mesh adaptation, and capacity growth invalidate that derivative and require event splitting.

Run `examples/monolithic_reactive_cfd_dem.py` and `tools/reactive_monolithic_qualification.py`.
