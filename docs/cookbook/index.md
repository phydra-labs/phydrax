# Cookbook

This section contains end-to-end recipes built from Phydrax's public APIs.
Field-learning recipes share a core pattern: choose labeled domains and
components, define `DomainFunction` fields, build residual operators, reduce
sampled constraints, and optimize the resulting functional with
`FunctionalSolver`. Inference, differential-equation, and control recipes instead
use their typed problem, signal, trajectory, and result contracts directly.

The examples keep physical case and time axes, schedule or sample masks, stable
identifiers, validity/status, and method/backend provenance visible where those
contracts provide them.

!!! info
    The cookbook examples are meant to demonstrate **basic workflows/recipes structurally**. Real workloads typically
    need larger numbers of collocation points and iterations, and often benefit from architecture and hyperparameter
    tuning for optimal accuracy and stability.

## How to choose a workflow

### Point batches vs axis-based grids

- Use `PointSampling` → `PointBatch` for paired collocation and scattered data.
- Use `GridSampling` → `GridBatch` for spectral/basis operators and axis-native
  neural operators such as DeepONet and FNO.

See [Guides → Domains and sampling](../guides_domain.md).

### Soft constraints vs enforced constraints

For boundary/initial conditions you can either:

- add penalty terms (soft constraints), or
- build an enforced ansatz \(\tilde u=\mathcal{H}(u)\) and train only on the remaining terms.

See [API → Enforcement](../api/enforcement.md) and
[API → Solver → Exact enforcement](../api/solver/enforcement.md).

### Differentiation backends

Differential operators support multiple backends (`backend="ad"|"jet"|"fd"|"basis"`) and autodiff modes
(`mode="reverse"|"forward"`). For deep math notes, see
[Appendix → Differentiation modes](../appendix/differentiation_modes.md).

## Recipes

- [Poisson (field learning, soft vs enforced BC)](poisson.md)
- [Heat equation (space–time, initial conditions, optional sensors)](heat.md)
- [Stochastic dynamics (PINNs, SDEs, and semidiscrete SPDEs)](stochastic_dynamics.md)
- [Filtering and smoothing stochastic state](filtering.md)
- [Controlled dynamics (driving paths, CDE integration, and Neural CDEs)](controlled_dynamics.md)
- [Control workflows (finite-horizon control, QPs, and MPC)](control.md)
- [Backward stochastic equations and semilinear PDEs](bsde.md)
- [Physics-informed graph residuals](graph_physics.md)
- [Inverse problems + hybrid physics–data](inverse_and_data.md)
- [Operator learning (DatasetDomain × coordinates)](operator_learning.md)
- [Uncertainty quantification](uncertainty_quantification.md)
- [Neural-operator uncertainty quantification](operator_uncertainty.md)
- [Mechanics and Deep Ritz objectives](mechanics.md)
- [Two-level closed-system quantum dynamics](quantum_two_level.md)
- [Composite systems and a Bell state](quantum_composite.md)
- [Open-system amplitude damping](quantum_open_system.md)
