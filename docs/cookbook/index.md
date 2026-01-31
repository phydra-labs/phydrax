# Cookbook

This section contains end-to-end “recipes” that all follow the same core pattern:

1) choose a labeled **domain** \(\Omega\) and one or more **components** \(\Omega_i\subseteq\Omega\),  
2) define **fields** as `DomainFunction`s on \(\Omega\),  
3) build **residual operators** \(r=\mathcal{N}(u,\dots)\),  
4) turn residuals into **constraint terms** by sampling + reduction,  
5) assemble a `FunctionalSolver` and optimize.

The goal is to show how Phydrax unifies “SciML modes” (PINNs, inverse problems, hybrid physics–data, operator learning)
via the same mathematical contract: minimize functionals over domains.

!!! info
    The cookbook examples are meant to demonstrate **basic workflows/recipes structurally**. Real workloads typically
    need larger numbers of collocation points and iterations, and often benefit from architecture and hyperparameter
    tuning for optimal accuracy and stability.

## How to choose a workflow

### Point batches vs coord-separable grids

- Use **paired point sampling** (`PointsBatch`) for most PINN-style collocation and scattered data.
- Use **coord-separable sampling** (`CoordSeparableBatch`) for spectral/basis operators and neural operators
  (DeepONet/FNO-style), where you want explicit axis semantics and grid evaluation.

See [Guides → Domains and sampling](../guides_domain.md).

### Soft constraints vs enforced constraints

For boundary/initial conditions you can either:

- add penalty terms (soft constraints), or
- build an enforced ansatz \(\tilde u=\mathcal{H}(u)\) and train only on the remaining terms.

See [API → Constraints → Enforced constraint ansätze](../api/constraints/enforced.md) and
[API → Solver → Enforced constraint pipelines](../api/solver/enforced_constraints.md).

### Differentiation backends

Differential operators support multiple backends (`backend="ad"|"jet"|"fd"|"basis"`) and autodiff modes
(`mode="reverse"|"forward"`). For deep math notes, see
[Appendix → Differentiation modes](../appendix/differentiation_modes.md).

## Recipes

- [Poisson (field learning, soft vs enforced BC)](poisson.md)
- [Heat equation (space–time, initial conditions, optional sensors)](heat.md)
- [Inverse problems + hybrid physics–data](inverse_and_data.md)
- [Operator learning (DatasetDomain × coordinates)](operator_learning.md)
