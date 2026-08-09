# Nonlinear dynamics, identification, and chaos

This recipe follows the shared contract from an explicit system to numerical evolution,
masked trajectory data, sparse identification, nonlinear analysis, and uncertainty. The
important boundary is not a class name: **system law, numerical evolution, estimator data,
and diagnostic evidence are separate objects**.

```python
import jax.numpy as jnp

import phydrax as phx
```

## 1. Declare and evolve a nonlinear flow

A system owns the local vector field and state semantics. The evolution owns the numerical
method. The grid owns the physical coordinates.

```python
sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
layout = phx.dynamics.StateLayout(
    (3,), component_names=("x", "y", "z")
)


def lorenz(time, state, args):
    del time, args
    x, y, z = state
    return jnp.asarray(
        [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    )


system = phx.dynamics.ContinuousSystem(
    lorenz, state_layout=layout, system_id="lorenz-63"
)
evolution = phx.solver.DiffraxEvolution(
    system, rtol=1e-8, atol=1e-10
)
grid = phx.dynamics.TimeGrid(
    jnp.linspace(0.0, 40.0, 4001), time_id="lorenz-observation-grid"
)
trajectory = phx.dynamics.evolve(
    evolution, jnp.asarray([1.0, 1.0, 1.0]), grid
)
assert trajectory.successful
```

`trajectory.states` is numerical output on exactly this grid. Convert it before using an
estimator. The adapter carries node and transition masks and does not reinterpret failed
or padded nodes as data.

```python
data = phx.dynamics.identification.trajectory_data_from_evolution(
    trajectory
)
```

If derivatives were not generated analytically, attach one declared estimate. Local
polynomials operate on physical coordinates and never cross an invalid or reset
transition.

```python
derivative = phx.dynamics.identification.local_polynomial_derivative(
    data, degree=3, window_radius=4
)
data_with_derivative = derivative.attach(data)
```

Inspect `derivative.valid`, `derivative.condition_number`, and `derivative.order`; an
attached derivative is not automatically valid at every sample.

## 2. Discover sparse equations without changing coefficient coordinates

Choose a feature library, formulation, and sparse solver independently.

```python
library = phx.dynamics.identification.PolynomialFeatureLibrary(
    layout, degree=2
)
problem = phx.dynamics.identification.SINDyProblem(
    data=data_with_derivative,
    library=library,
    formulation=phx.dynamics.identification.StrongSINDyFormulation(),
)
regressor = phx.dynamics.identification.SequentialThresholdedLeastSquares(
    0.05,
    ridge=1e-10,
    threshold_space="physical",
    scale_features=True,
    unbiased_refit=True,
)
identified = phx.dynamics.identification.fit_sindy(problem, regressor)

print("\n".join(identified.render_equations()))
identified_system = identified.to_system(system_id="identified-lorenz")
```

Feature scaling conditions the solve; `identified.coefficients` remains in the physical
feature coordinates named by `identified.design.feature_names`. Check all of:

```python
assert identified.valid
print(identified.status)
print(identified.regression.rank)
print(identified.regression.condition_number)
print(identified.regression.history.residual_norm)
```

Do not select a formulation by convenience:

- use `DiscreteSINDyFormulation` for an observed map;
- use `IntegralSINDyFormulation` when endpoint differences are more trustworthy than
  pointwise derivatives;
- use `WeakSINDyFormulation` for noisy trajectories and declare its compact windows,
  boundary policy, and quadrature;
- use `fit_implicit_sindy` for homogeneous implicit or rational equations;
- use `fit_pde_find` for named structured time--space data.

For model selection, `select_sindy_model` fits candidates on training rows and scores
validation rows after a temporal embargo. `fit_ensemble_sindy` reports coefficient and
support samples plus valid-member inclusion probabilities. A failed member remains in the
member mask; it is not replaced.

## 3. DMD and EDMD for map-level evolution

DMD and EDMD consume valid adjacent pairs from the same `TrajectoryData` contract.
Controlled data includes its source-aligned input matrix in the same weighted solve.

```python
dmd = phx.dynamics.identification.fit_dmd(
    data,
    rank=3,
)
print(dmd.diagnostics.weighted_residual_norm)
print(dmd.diagnostics.condition_number)

edmd = phx.dynamics.identification.fit_edmd(
    data,
    phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=2),
    ridge=1e-10,
    decoder_ridge=1e-10,
)
```

DMD fits the physical-state map. EDMD fits feature evolution and a separate physical-state
decoder. `to_system()` is available for ambient Euclidean states. For a manifold-valued
state, supply a geometry-aware identification method rather than treating chart
coordinates as a global physical map.

## 4. Sections and return maps

A section crossing is an event extracted from a trajectory, not merely a sign change in
an unmasked array.

```python
section = phx.dynamics.analysis.AffineSection(
    jnp.asarray([1.0, 0.0, 0.0]),
    0.0,
    state_layout=layout,
    section_id="x-zero",
)
crossings = phx.dynamics.analysis.find_section_crossings(
    data,
    section,
    direction="positive",
    refinement="interpolation",
    max_crossings=128,
)
return_map = phx.dynamics.analysis.section_return_map(crossings)
```

Inspect `crossings.count`, `overflow`, `valid`, `status`, `bracket_start`, `bracket_end`,
and `section_values`. Interpolation refinement is bounded by saved data. To refine by
re-integration, pass the declared evolution and use `refinement="evolution"`.

## 5. Periodic orbits, Floquet multipliers, and continuation

For an autonomous flow, use multiple shooting and a phase condition. The orbit state is a
fixed number of shooting nodes plus its period. Dense Newton is appropriate only below
the declared guard; choose matrix-free Newton--Krylov for larger tangent systems.

```python
phase = phx.dynamics.analysis.ComponentPhaseCondition(
    0,
    0.0,
    state_layout=layout,
)
periodic_problem = phx.dynamics.analysis.PeriodicOrbitProblem(
    evolution,
    kind="flow",
    num_segments=8,
    phase_condition=phase,
    problem_id="lorenz-periodic-orbit",
)
initial_nodes = phx.dynamics.analysis.periodic_nodes_from_state(
    periodic_problem,
    crossings.states[0],
    period=1.5,
)
orbit = phx.dynamics.analysis.solve_periodic_orbit(
    periodic_problem,
    initial_nodes,
    initial_period=1.5,
    linear_method="matrix_free",
    max_iterations=20,
)

floquet = phx.dynamics.analysis.floquet_spectrum(
    orbit,
    method="leading",
    leading_k=3,
)
```

An initial guess is not a periodic orbit. Require `orbit.valid`, inspect its residual and
history, then inspect `floquet.valid`, neutral multiplier evidence, stability, and Krylov
status.

Continuation is a generic square residual with one scalar parameter. For an equilibrium
branch, the residual is the vector field evaluated at an equilibrium state.

```python
continuation_layout = phx.dynamics.StateLayout(
    (3,), component_names=("x", "y", "z")
)


def equilibrium_residual(state, parameter, args):
    del args
    x, y, z = state
    return jnp.asarray(
        [sigma * (y - x), x * (parameter - z) - y, x * y - beta * z]
    )


continuation_problem = phx.dynamics.analysis.ContinuationProblem(
    equilibrium_residual,
    state_layout=continuation_layout,
    parameter_id="rho",
    spectrum_kind="flow",
    problem_id="lorenz-equilibria",
)
branch = phx.dynamics.analysis.continue_branch(
    continuation_problem,
    jnp.zeros((3,)),
    jnp.asarray(0.0),
    method="pseudo_arclength",
    direction=1,
    initial_step=0.02,
    max_points=256,
    parameter_bounds=(-1.0, 30.0),
)
```

`branch.bifurcations.fold` and `hopf` are finite-resolution candidates from stored
spectra. They are not certified normal forms. Use `CallableNormalFormHook` to record a
fixed-width problem-specific diagnostic. Use `CallableBranchSwitchHook` plus
`branch_switch_seed` to propose a new seed explicitly; the hook does not claim that a new
branch was found.

## 6. Lyapunov spectra and covariant directions

A finite-time Lyapunov spectrum belongs to an evolution, grid, tangent method, and
schedule.

```python
spectrum = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
    evolution,
    jnp.asarray([1.0, 1.0, 1.0]),
    grid,
    leading_k=3,
    qr_interval=10,
    burn_in=1000,
    accumulation_interval=100,
)
print(spectrum.exponents)
print(spectrum.kaplan_yorke_dimension)
```

Resume an exact schedule with `checkpoint=spectrum.checkpoint` and `initial_state=None`.
Do not concatenate independently normalized finite-time estimates.

Covariant vectors require a backward triangular sweep:

```python
clv = phx.dynamics.analysis.covariant_directions(
    evolution,
    jnp.asarray([1.0, 1.0, 1.0]),
    grid,
    kind="clv",
    memory_mode="store",
    qr_interval=10,
    save_every=10,
    backward_discard=20,
)
```

`memory_mode="store"` retains forward QR frames. `"recompute"` retains no QR history but
rebuilds prefixes during the backward pass and can require quadratic tangent work.
Returned saved directions consume output memory in both modes. Inspect
`direction_valid`, `covariance_error`, and `backward_convergence_drift`; a small covariance
residual does not prove that the finite backward window has forgotten its terminal
condition.

For finite-amplitude instability at a declared scale, use a separate estimator:

```python
finite_size = phx.dynamics.analysis.finite_size_growth(
    evolution,
    jnp.asarray([1.0, 1.0, 1.0]),
    grid,
    num_directions=16,
    seed=3,
    perturbation_distance=1e-3,
    rescale_interval=10,
)
```

This is not the zero-amplitude Lyapunov limit.

## 7. Recurrence, 0--1, correlation dimension, and surrogates

Dense recurrence and pair-count estimators are quadratic in saved samples. Subsample
explicitly or respect `max_samples`; do not bypass the guard accidentally.

```python
rqa = phx.dynamics.analysis.recurrence_quantification(
    data,
    2.0,
    theiler_window=50,
    minimum_diagonal_length=2,
    minimum_vertical_length=2,
)

zero_one = phx.dynamics.analysis.zero_one_test(
    data,
    component=0,
    observable_id="lorenz-x",
    burn_in=1000,
    num_frequencies=100,
    seed=7,
    fit_lags=(10, 300),
)

correlation = phx.dynamics.analysis.correlation_dimension(
    data,
    jnp.logspace(-2.0, 1.0, 40),
    theiler_window=50,
    fit_indices=(10, 28),
)
```

RQA stores `eligible` separately from `recurrence`; the Theiler window is therefore
auditable. The 0--1 result stores its seeded frequencies, per-frequency statistics,
displacements, contiguous segment mask, fit lags, and spread. Correlation dimension
stores pair counts, correlation sums, local slopes, fit mask, and `r_squared`. Choose the
fit window from an observed scaling regime; the API does not discover or certify one.

Test a scalar statistic against a declared null:

```python
x = data.states[:, 0]
significance = phx.dynamics.analysis.surrogate_significance(
    x,
    lambda values: jnp.mean(values[:-1] * values[1:]),
    statistic_id="lag-one-product",
    method="aaft",
    alternative="greater",
    num_surrogates=999,
    seed=11,
)
```

The result keeps every surrogate statistic and uses a plus-one p-value. Phase-randomized
surrogates preserve the Fourier amplitude spectrum. AAFT additionally approximates the
observed marginal distribution. Neither null is an exchangeability claim for arbitrary
nonstationary data.

## 8. Uncertainty over initial conditions, parameters, noise, and numerics

Generate the cases with the responsible subsystem, compute one diagnostic per case, then
aggregate. Do not label a solver tolerance sweep as process noise or a stochastic path
ensemble as posterior parameter uncertainty.

```python
# Shape: (initial_condition, parameter, noise_realization, tolerance, metric)
diagnostic_samples = jnp.asarray(...)
diagnostic_valid = jnp.asarray(...)

uncertainty = phx.dynamics.analysis.summarize_chaos_uncertainty(
    diagnostic_samples,
    metric_names=("largest-exponent", "zero-one-statistic"),
    case_axes=("initial-condition", "rho", "path", "tolerance"),
    source_kinds=(
        "initial_condition",
        "parameter",
        "noise",
        "numerics",
    ),
    sample_valid=diagnostic_valid,
    confidence=0.95,
    bootstrap_samples=1999,
    seed=13,
)
```

The bootstrap resamples the declared weighted empirical cases. `source_variance` is a
variance of source-level conditional means, not a general Sobol decomposition and not a
causal attribution.

## 9. Shadowing solver boundary

Phydrax evaluates a shadowing candidate but does not silently choose one shadowing
algorithm, segmentation, regularizer, or preconditioner.

```python
shadowing_problem = phx.dynamics.analysis.ShadowingSensitivityProblem(
    evolution,
    # Integrated endpoint parameter-tangent contribution on one segment:
    lambda state, source, target, args: parameter_forcing(
        state, source, target, args
    ),
    lambda coordinate, state, args: state[2],
    parameter_id="rho",
    observable_id="mean-z",
    problem_id="lorenz-rho-shadowing",
    neutral_direction=lambda coordinate, state, args: lorenz(
        coordinate, state, args
    ),
    time_dilation="flow",
)

candidate = phx.dynamics.analysis.evaluate_shadowing_candidate(
    shadowing_problem,
    trajectory,
    tangent_path,
    time_dilation=dilation_path,
    boundary="free",
)
residual = candidate.least_squares_residual()
```

An external least-squares shadowing or NILSS solver can optimize this residual. Require
small dynamic defects, the intended boundary residual, controlled neutral inner products,
and convergence under segment/window refinement before interpreting
`mean_directional_response`.

## 10. Interoperability recipes

### Delay, memory, rough, and structured differential solvers

The differential adapter accepts `DifferentialSolution`, `MemoryEquationSolution`,
`RoughDifferentialSolution`, and `ControlledDifferentialSolution`.

```python
delay_solution = phx.solver.solve_diffrax_delay(
    delay_problem,
    save_times=save_times,
)
delay_data = (
    phx.dynamics.identification.trajectory_data_from_differential_solution(
        delay_solution,
        state_layout=delay_state_layout,
        source_id="retarded-population-study",
    )
)
```

The same path covers deterministic semidiscrete SPDE output and memory-equation output.
For non-Euclidean differential or rough output, pass the original `StateLayout`; the
adapter checks its geometry ID. The adapter records saved states and masks. It does not
reconstruct a delay history, rough control, jump event, or spatial discretization from a
flat state.

A `ControlledDifferentialSolution` is unwrapped without pretending that path levels are
ordinary source-aligned controls. Its path ID stays in `source_id`.

### Finite-horizon controlled dynamics

```python
control_data = phx.dynamics.identification.trajectory_data_from_control(
    control_result.trajectory,
    state_layout=state_layout,
)
```

Controls align with transitions. Controlled DMD or SINDy receives them through the
feature library's matching `InputLayout`; an autonomous library rejects required inputs
rather than dropping them.

### Stochastic paths

```python
stochastic_data = phx.dynamics.identification.trajectory_data_from_stochastic(
    stochastic_trajectory,
    state_layout=state_layout,
)
```

Case and realization axes survive. A deterministic flow Lyapunov calculation on one
realization is not automatically a stochastic Lyapunov exponent. Fit or diagnose each
path under a declared pooling policy, then pass the resulting scalar samples to
`summarize_chaos_uncertainty` with source kind `"noise"` or `"process"`.

### Geometric states

Put the geometry on `StateLayout` before constructing the system. `evolve`, tangent
actions, section evolution refinement, and finite-size separation retain it. Polynomial
coordinate libraries and ambient DMD can still be useful local chart models, but their
`to_system()` conversion rejects non-Euclidean states. A global manifold model needs a
declared geometry-aware library or retraction model.

### Operator and PDE workflows

A neural operator can generate or denoise saved trajectories, but construct
`TrajectoryData` with the original masks and case axes before identification. Do not use
operator confidence as a sample-valid mask unless that policy is declared separately.

For PDE-FIND, preserve time and spatial axes instead of flattening a field into a generic
trajectory:

```python
pde_data = phx.dynamics.identification.StructuredPDEData(
    (time, space),
    field_values,
    state_layout=phx.dynamics.StateLayout(
        (1,), component_names=("u",)
    ),
    coordinate_names=("t", "x"),
    source_id="reaction-diffusion-snapshots",
)
library = phx.dynamics.identification.PolynomialPDELibrary(
    pde_data.state_layout,
    ("t", "x"),
    polynomial_degree=2,
    spatial_derivative_order=2,
    include_interactions=True,
)
pde_result = phx.dynamics.identification.fit_pde_find(
    phx.dynamics.identification.PDEIdentificationProblem(
        data=pde_data, library=library
    ),
    phx.dynamics.identification.SequentialThresholdedLeastSquares(
        0.02, threshold_space="physical"
    ),
)
```

For a semidiscrete PDE, the state is already finite dimensional: construct a
`ContinuousSystem` around the semidiscrete right-hand side and use the ordinary evolution,
Lyapunov, section, periodic-orbit, and continuation substrate. PDE-FIND and semidiscrete
analysis answer different questions and should not be conflated.

## Failure checklist

Before interpreting a nonlinear-dynamics result:

1. verify every relevant `valid`, `status`, and overflow or termination field;
2. inspect rank, conditioning, residuals, and iteration history for identification;
3. confirm no derivative window, weak window, delay embedding, or rollout crosses a reset;
4. confirm physical-time versus iteration normalization;
5. inspect Lyapunov convergence drift and CLV terminal-condition drift;
6. declare Theiler, fit, burn-in, and surrogate windows before looking at the statistic;
7. preserve RNG seeds and all failed ensemble members;
8. separate initial-condition, parameter, path-noise, model, and numerical uncertainty;
9. treat bifurcation flags as candidates until refined and classified;
10. treat a shadowing candidate as solved only after its residual and refinement evidence
    support that claim.
