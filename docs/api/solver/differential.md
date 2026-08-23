# Differential equation integration

The differential backend integrates finite-dimensional initial-value problems through
[Diffrax](https://docs.kidger.site/diffrax/). It is separate from `FunctionalSolver`:
`FunctionalSolver` minimizes a physics/data functional, while `solve_diffrax` numerically
integrates a supplied drift and optional named stochastic terms.

Implicit residuals `F(t, y, ydot, args) = 0` use the separate native
[differential-algebraic solver](differential_algebraic.md). That path owns consistent
initialization, BDF1--BDF5, and endpoint theta rather than encoding a singular mass
matrix as an explicit Diffrax vector field.

## Problem, Wiener terms, and realization contract

`DifferentialProblem` represents

$$
dY_t = f(t,Y_t,a)\,dt + \sum_k g_k(t,Y_t,a)\,dW_t^{(k)}.
$$

Omit `wiener_terms` for an ODE. Each named `WienerTerm` declares one independent
Wiener source: its coefficient, native noise shape, optional basis identity, and
mathematical structure (`additive`, `commutative`, or `general`). For a state of shape
`state_shape`, the coefficient must return `state_shape + noise_shape`.
`coefficient_matrix(time, state, args)` validates that contract and returns the
canonical `(state_size, noise_size)` view without transposing physical axes. The
backend flattens and concatenates terms in declared order; `DifferentialSolution`
retains the corresponding named column slices.

A `WienerRealization` defines one global Brownian path or coupled path batch. Its
`support` is independent of any one solve interval, so solving adjacent subintervals
with the same realization queries one continuous path. It stores the root key, global
support, combined noise shape, sample shape, Brownian-tree tolerance, Lévy-area level,
and optional noise-basis identity. Path keys use `jax.random.fold_in`: increasing the
ensemble size preserves every existing path prefix.

`realization_id` is a computed fingerprint of all path-defining inputs. `label` is
human-readable metadata. `coupling_id` identifies paths intended for common-random-
number comparisons. `WienerRealization.antithetic` constructs explicit `(+W,-W)`
pairs. Reuse one realization across models, parameters, or coarse/fine grids when
common randomness is required.

::: phydrax.solver.DifferentialProblem
    options:
        members:
            - __init__
            - stochastic
            - additive_noise

---

::: phydrax.solver.WienerTerm
    options:
        members:
            - __init__
            - noise_size
            - coefficient_matrix

---

::: phydrax.stochastic.WienerRealization
    options:
        members:
            - __init__
            - independent
            - antithetic
            - num_paths
            - path_keys

## ODE solve

The default deterministic solver is `diffrax.Tsit5` with a PID step-size controller.
The result remains differentiable with respect to array-valued initial states,
parameters, and vector-field leaves.

```python
import jax.numpy as jnp
import phydrax as phx

problem = phx.solver.DifferentialProblem(
    lambda t, y, rate: -rate * y,
    jnp.asarray([1.0]),
    t0=0.0,
    t1=2.0,
    args=jnp.asarray(0.4),
)
solution = phx.solver.solve_diffrax(
    problem,
    save_times=jnp.linspace(0.0, 2.0, 21),
)
```

::: phydrax.solver.solve_diffrax

## Additive IMEX solve

`SplitDifferentialProblem` preserves `y' = f_explicit + f_implicit` as two terms.
`solve_diffrax` defaults this form to `diffrax.KenCarp4`; Sil3 and
KenCarp3/4/5 may be selected explicitly. Passing an ordinary explicit or implicit
solver rejects rather than silently summing the terms.

::: phydrax.solver.SplitDifferentialProblem

---

::: phydrax.solver.split_differential_problem

`DifferentialSolution.temporal_evidence` records method capabilities and the complete
solver/controller/adjoint/event configuration identity. `valid` remains per-output
finite evidence; `successful` additionally requires an acceptable backend result.

## Markov cubature weak solve

`solve_markov_cubature` consumes the same stochastic `DifferentialProblem`, but
returns a deterministic positive weighted law rather than one sampled
`DifferentialSolution`. A `MarkovCubaturePlan` owns the complete static
discretization: temporal mesh, multivariate standard-normal increment rule,
polynomial recombination policy, optional certified Wiener controls, expansion
capacity, path-flow substeps, history policy, and throw policy.

`weak-euler` supports Itô coefficients and additive Stratonovich coefficients.
`stratonovich-flow` requires an additive or commutative Stratonovich declaration
and signature degree at least three. Both methods currently require a real
Euclidean state and a Gaussian rule whose dimension matches the concatenated
named Wiener columns. These gates are mathematical contracts, not backend
fallback choices.

`MarkovCubatureSolution.points`, `log_weights`, and `mask` have fixed retained
capacity at every saved mesh node. `measure(index)` exposes one saved law through
the ordinary dependent weighted-measure integration path. Diagnostics retain
per-step expansion/retention counts, numerical rank, mass and moment residuals,
minimum positive weight, statuses, capacities, weak order, and content identities.
Discrete support selection is frozen under differentiation; the
selected continuous moment system supplies the weight tangent.

::: phydrax.solver.PolynomialRecombination

---

::: phydrax.solver.MarkovCubaturePlan

---

::: phydrax.solver.MarkovCubatureStatus

---

::: phydrax.solver.MarkovCubatureDiagnostics

---

::: phydrax.solver.MarkovCubatureSolution

---

::: phydrax.solver.solve_markov_cubature

## Controlled differential equations

### Differentiable driving paths

`AbstractDifferentiableDrivingPath` is Phydrax's public first-level control
contract. A path has closed `support`, a stable `path_id`, a fixed
`value_shape`, and `evaluate`, `increment`, and `derivative` operations.
`breakpoints` has fixed JAX capacity; the aligned Boolean `breakpoint_mask`
identifies the active interior times at which the first derivative may jump.
The `side="left"` or `"right"` argument gives exact one-sided semantics at
those times.

The concrete paths make interpolation provenance explicit:

- `CallableDrivingPath` wraps declared value and derivative callbacks together
  with support, shape, ID, and a complete derivative-breakpoint schedule.
- `PiecewiseLinearDrivingPath` interpolates a valid sampled prefix exactly.
  Every active interior sample time is a derivative breakpoint.
- `CausalBackwardHermiteDrivingPath` uses only already-observed backward
  slopes. The first interval is linear and its derivative is continuous at
  knots; this is a causal interpolation, not an offline smoothing fit.
- `OfflineCubicDrivingPath` is a global natural cubic interpolant. It needs at
  least four valid samples and uses a JAX tridiagonal solve.
- `FixedBSplineDrivingPath` evaluates a validated `BSplineGrid`. The grid is
  non-trainable configuration, while its finite, inexact `coefficients` are
  ordinary differentiable JAX leaves. Repeated knots have exact one-sided
  derivative semantics; the grid must still declare a continuous path.

Sampled paths preserve the physical leading sample axis. `time_mask` must be a
prefix, and `value_mask` must make each sample wholly valid or wholly invalid;
partial payload repair is rejected. Valid sample times must be finite and
strictly increasing and valid values finite. Times are promoted to real inexact
arrays; numeric values are promoted to an inexact dtype without changing their
physical axes, while masks remain Boolean. No path silently fills, sorts,
extrapolates, or repairs observations.

The `fit` methods on the three sampled path classes return
`DrivingPathFitDiagnostics`. It records support, residual norm and maximum,
minimum and maximum sample spacing, validity/status, sample count and capacity,
value shape, regularization, and exact `method_id`, `approximation_id`, and
`backend`. These interpolants have `regularization == 0`; diagnostics never
relabel interpolation residual as statistical uncertainty.

Gradients follow the numeric leaves. In particular, differentiating through a
`FixedBSplineDrivingPath` returns a coefficient gradient with the same physical
basis and payload axes as `coefficients`. Grid topology, masks, IDs, and
breakpoint schedules are discrete provenance, not differentiable parameters.
Declared callbacks and `RoughDifferentialProblem` fields must return arrays of
the documented shape; Phydrax does not transpose or infer coefficient axes.
Use an inexact dtype for coefficients that participate in gradients.

::: phydrax.solver.AbstractDifferentiableDrivingPath

---

::: phydrax.solver.CallableDrivingPath

---

::: phydrax.solver.PiecewiseLinearDrivingPath

---

::: phydrax.solver.CausalBackwardHermiteDrivingPath

---

::: phydrax.solver.OfflineCubicDrivingPath

---

::: phydrax.solver.FixedBSplineDrivingPath

---

::: phydrax.solver.DrivingPathFitDiagnostics

### CDE solve and the CDE/RDE boundary

`solve_diffrax_cde` solves the differentiable first-level equation
`dY = V0(t, Y, args) dt + V(t, Y, args) dX`. Construct its dynamics with
`RoughDifferentialProblem`: `vector_fields(time, state, args)` returns
`state_shape + (driver_dimension,)`, and an optional
`drift(time, state, args)` returns `state_shape`. The context is always the
last callback argument. The path must have exact shape
`(driver_dimension,)`; matrix- or scalar-shaped controls are not reshaped.

This is deliberately not the rough path solver. A differentiable path supplies
only first-level values and derivatives. An `AbstractRoughControl` with
nontrivial second-level information must go to `solve_rough_differential`;
`solve_diffrax_cde` rejects it rather than discard its lift. Conversely, do not
replace the Phydrax adapter with an ordinary `diffrax.ControlTerm`.
`solve_diffrax_cde` lowers the declared CDE to a Phydrax
`DifferentialProblem`, retains the path and problem IDs, and returns a
`ControlledDifferentialSolution` with the underlying
`DifferentialSolution`, interpolation class, control dimension, derivative
schedule, and lowering metadata.

Derivative breakpoints are exact integration boundaries. Active interior
breakpoints are passed to `diffrax.ClipStepSizeController` as both `step_ts`
and `jump_ts`, so a step lands on the knot and restarts with the selected
one-sided derivative. A path with breakpoint capacity therefore requires an
adaptive Diffrax controller, normally `diffrax.PIDController`; a fixed-step
controller is rejected even if the requested grid happens to contain the
knots. The schedule is stop-gradient discrete data. There is no hidden
fallback that ignores or perturbs a breakpoint.

All ordinary `solve_diffrax` controls remain explicit: solver, step-size
controller, adjoint, initial step, event, tolerances, dense output, step
capacity, and throw policy. `ControlledDifferentialSolution.valid`,
`successful`, solver and geometry provenance, event information, and dense
`evaluate` are delegated without changing axes or masks.

::: phydrax.solver.ControlledDifferentialSolution

---

::: phydrax.solver.solve_diffrax_cde

### Neural CDE loss, training, and exact resume

`NeuralCDEVectorField` adapts a callable such as `phydrax.nn.models.MLP` or
`phydrax.nn.models.KAN`. The model consumes a flattened physical state and must return
exactly `prod(state_shape) * control_dimension` coefficients; the adapter
restores `state_shape + (control_dimension,)`.

`NeuralCDETrainingData` keeps one differentiable path per physical case,
case-leading initial states, rank-two `(case, observation)` times and validity,
and observations shaped `(case, observation) + state_shape`. Every path must
share the same vector control dimension. Valid times are finite, strictly
increasing within a case, and inside its path support. `time_channel` must
evaluate to the physical observation time; this prevents an implicit index
axis from replacing physical time. Invalid observations remain masked rather
than imputed. Stable `case_ids` and `data_id` retain data provenance.

`neural_cde_loss` solves each selected case through `solve_diffrax_cde` at its
own valid physical observation times and returns mean squared error over valid
state scalars. `solve_options` cannot override those save times.
`train_neural_cde` applies deterministic Optax mini-batches and returns
`NeuralCDETrainingState`, including the vector field, optimizer state, last
loss, exact epoch/batch/update position, ordering algorithm, and data,
optimizer, solver-configuration, and dynamics IDs.

Start training with `vector_field=...` and no `state`. Resume with
`state=previous_state` and no `vector_field`; changing batch size, seed,
shuffle choice, data, optimizer ID, solver configuration ID, or dynamics ID
changes the training fingerprint and is rejected. Resume occurs at the exact
private data-plane batch boundary. There is no implicit optimizer reset,
reshuffle, skipped batch, or compatibility repair.

::: phydrax.solver.NeuralCDEVectorField

---

::: phydrax.solver.NeuralCDETrainingData

---

::: phydrax.solver.NeuralCDETrainingState

---

::: phydrax.solver.neural_cde_loss

---

::: phydrax.solver.train_neural_cde

## Probabilistic ODE filtering

`solve_probabilistic_ode` applies a native-JAX Gaussian ODE filter to a real,
deterministic, Euclidean `DifferentialProblem`; it does not dispatch to
Diffrax. `ProbabilisticODEMethod` chooses integrated-Wiener order one through
four, EK0 or EK1 residual linearization, fixed capacity, optional smoothing,
factorization and covariance representation, diffusion calibration,
tolerances, explicit covariance regularization, stiffness threshold, and a
stable `method_id`.

The public literal contracts are `ProbabilisticODEUpdate` (`"ek0"` or
`"ek1"`), `ProbabilisticODEFactorization` (`"dense"` or
`"block_diagonal"`), `ProbabilisticODECovarianceOutput` (`"dense"` or
`"matrix_free"`), `ProbabilisticODECalibration` (`"none"` or
`"quasi_mle"`), and `ProbabilisticODEStatus` (`"success"`, `"stiff"`,
`"nonfinite"`, or `"step_limit_reached"`).

The posterior keeps five uncertainty sources separate:

- `"numerical"` is integrated-Wiener discretization uncertainty;
- `"process"` is declared model-discrepancy covariance;
- `"observation"` is declared residual-observation covariance;
- `"initial_condition"` is declared initial-state covariance;
- `"parameter"` propagates the declared covariance of flattened
  `DifferentialProblem.args`.

None is silently converted into SDE process noise or merged with another
source. A stochastic `DifferentialProblem` is rejected. Parameter uncertainty
requires inexact problem arguments. `source_covariances`,
`covariance_factor`, `covariance_matvec`, and guarded `dense_covariance`
preserve this provenance at each physical save time.

With `adaptive=True`, a uniform pilot pass computes dimensionless residuals and
redistributes exactly `num_steps` across the interval. This is fixed-work
residual adaptation, not an unbounded accept/reject controller, and it cannot
be combined with `step_size`. `"quasi_mle"` calibration rescales only the
numerical covariance component from accumulated normalized residuals;
`"none"` keeps `base_diffusion`. Optional smoothing is Rauch--Tung--Striebel
smoothing over the fixed grid.

Dense filtering guards the augmented dimension
`(order + 1) * state_size` with `max_dense_dimension`.
`dense_covariance` separately guards physical-state materialization. Exceeding
either guard raises; Phydrax never falls back to block diagonal.
`factorization="block_diagonal"` is an explicit diagonal-across-state
approximation and rejects non-diagonal input covariances. Covariance
regularization is exactly the requested nonnegative value; there is no hidden
jitter or posterior repair.

`ProbabilisticODESolution` exposes means, standard deviations, optional dense
covariances, factor and source covariances, residuals, normalized residuals,
step sizes, calibrated diffusion scale, quasi likelihood, validity, stats,
checkpoint, and exact method/approximation/discretization/backend IDs. Status
codes are `PROBABILISTIC_ODE_SUCCESS`,
`PROBABILISTIC_ODE_STIFF`, `PROBABILISTIC_ODE_NONFINITE`, and
`PROBABILISTIC_ODE_STEP_LIMIT_REACHED`; use
`probabilistic_ode_status_name` for their stable names. `successful` requires
success status and every saved marginal valid.

Resume only from `solution.checkpoint`, with the same method ID,
factorization, state shape, and explicit nominal `step_size`; the resumed
problem must start at the checkpoint time. Adaptive solves do not resume
through this fixed-step checkpoint route. Provenance mismatches fail rather
than restart or reinterpret a checkpoint.

::: phydrax.solver.ProbabilisticODEMethod

---

::: phydrax.solver.ProbabilisticODESolution

---

::: phydrax.solver.probabilistic_ode_status_name

---

::: phydrax.solver.solve_probabilistic_ode

## Tangent and Lyapunov analysis

Finite-time Lyapunov spectra consume the shared `phydrax.dynamics` flow/map
evolution contract rather than a solver-specific problem. The implementation,
checkpoint/result contracts, periodic-QR cadence, fixed-step flow
discretization, and Kaplan--Yorke validity rules are documented under
[API → Dynamical systems, identification, and chaos](../dynamics.md#lyapunov-covariant-finite-size-and-recurrence-diagnostics).

## Geometric ODE and Stratonovich solve

Set `DifferentialProblem.state_geometry` when the state is constrained to an
array manifold. Construction validates initial membership. Nontrivial geometry
rejects ordinary solvers. `GeometricEuler` is first order; `RKMK` requires an
exact local pullback. `CommutatorFreeSolver` additionally requires a declared
shared trivialization, supplied by the built-in Euclidean and SO(n) geometries
but not SPD(n) or generic embedded adapters. Stochastic solves require explicit
Stratonovich semantics and `SRKMK`. All geometric methods are fixed-step and
require `dt0`.

The solver's geometry ID must equal the problem's. The solution retains
`state_geometry_id`, stable `solver_id`, and `resolved_method`. Geometric local
interpolation is also used by dense queries and Diffrax root-finding events.

::: phydrax.solver.GeometricEuler

---

::: phydrax.solver.RKMK

---

::: phydrax.solver.CommutatorFreeTableau

---

::: phydrax.solver.CommutatorFreeSolver

---

::: phydrax.solver.SRKMK

---

::: phydrax.solver.solver_state_geometry

## Dense vector interpolation

Pass `dense=True` to retain Diffrax's local interpolants and enable
`DifferentialSolution.evaluate`. Query times may be a scalar or an arbitrarily shaped
array:

```python
solution = phx.solver.solve_diffrax(
    problem,
    save_times=jnp.asarray([0.0, 2.0]),
    dense=True,
)
query_times = jnp.asarray([[0.1, 0.4], [1.2, 1.8]])
interpolated = solution.evaluate(query_times)
assert interpolated.shape == (2, 2, 1)
```

For one trajectory, the output shape is `query_times.shape + state_shape`. For an
ensemble it is `sample_shape + query_times.shape + state_shape`: every realization is
evaluated on the same query array without flattening either the process or query axes.
Scalar query times omit the query axis. Dense evaluation remains JAX-transformable and
differentiable through the solve.

Dense output is opt-in because it retains per-step interpolation data. Query times must
be non-empty, finite, and inside the interval available to every realization; this is
the common interval when event termination differs across an ensemble. The `left`
argument selects the left or right limit at a jump. `has_dense_interpolation` reports
whether evaluation is available. The vectorization is implemented internally by
Phydrax and requires no interpolation package beyond Diffrax.

## SDE solve and process ensemble

The default Itô solver is fixed-step Euler--Maruyama (`diffrax.Euler`); the default
Stratonovich solver is `diffrax.EulerHeun`. SDE calls require a
`WienerRealization` and explicit `dt0`. Before entering Diffrax, Phydrax validates:

- the problem interval lies inside the realization's global support,
- combined noise shape and basis identity agree,
- a fixed-step solve uses a Brownian-tree tolerance strictly smaller than `dt0`,
- general Itô/Stratonovich problems use a solver with the matching interpretation,
- the realization supplies the solver's minimum Lévy-area level.

Explicitly additive noise may use either interpretation-specific solver because the
Itô and Stratonovich equations coincide in that case.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

problem = phx.solver.DifferentialProblem(
    lambda t, y, args: -0.2 * y,
    jnp.zeros((2,)),
    t0=0.0,
    t1=1.0,
    wiener_terms=(
        phx.solver.WienerTerm(
            "state-noise",
            lambda t, y, args: 0.3 * jnp.eye(2),
            (2,),
            structure="additive",
            basis_id="state-space",
        ),
    ),
    interpretation="ito",
)
realization = phx.stochastic.WienerRealization(
    jr.key(0),
    (2,),
    support=(0.0, 1.0),
    sample_shape=(128,),
    tolerance=1e-3,
    noise_id="state-space",
    label="run-0",
)
ensemble = phx.solver.solve_diffrax_ensemble(
    problem,
    save_times=jnp.linspace(0.0, 1.0, 11),
    realization=realization,
    dt0=1e-2,
)
predictive = ensemble.to_predictive(
    sample_dim="path",
    time_dim="time",
    state_dims=("state",),
)
```

The realization owns the sample shape; `solve_diffrax_ensemble` does not accept a
separate path count. Results have shape
`sample_shape + (num_times,) + state_shape`. `to_predictive` currently requires one
sample axis and labels it as `process` uncertainty by default. It does not reinterpret
discretization or solver error as process uncertainty.

::: phydrax.solver.solve_diffrax_ensemble

## Finite-activity jump and hybrid solves

Pure jump models implement `AbstractJumpProcess`. `JumpProcess` supplies
callable intensities, jump maps, and optional marked jumps;
`MassActionJumpProcess` supplies combinatorial reaction propensities and
stoichiometric updates. A matching `PoissonClockRealization` owns unit-rate
thresholds and mark keys for every channel and path.

Use `solve_next_reaction` to advance channel-specific internal clocks or
`solve_direct_ssa` to sample from the total hazard. Both return `JumpSolution`
with a canonical `JumpEventBatch`. The event batch carries a valid-prefix mask
and one status per path: success, event-capacity exhaustion, invalid intensity,
or solver failure. Increase capacity with
`PoissonClockRealization.extend`; extending preserves all existing thresholds
and marks.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

process = phx.stochastic.JumpProcess(
    lambda t, state, args: jnp.asarray([2.0]),
    lambda state, channel, mark, args: state + jnp.asarray([1.0]),
    state_shape=(1,),
    num_channels=1,
    process_id="counting-process",
)
clock = phx.stochastic.PoissonClockRealization(
    jr.key(1),
    1,
    support=(0.0, 1.0),
    max_events_per_channel=16,
    sample_shape=(256,),
    process_id=process.process_id,
)
solution = phx.solver.solve_next_reaction(
    process,
    clock,
    jnp.asarray([0.0]),
    t0=0.0,
    t1=1.0,
    save_times=jnp.linspace(0.0, 1.0, 11),
)
```

`JumpDifferentialProblem` combines the jump process with a
`DifferentialProblem`. `solve_jump_differential` integrates each
state-dependent cumulative hazard and localizes threshold crossings with a
Diffrax event root. Between crossings, the continuous component may be an ODE
or SDE. For an SDE, pass one global `WienerRealization`; the result retains a
`CompositeStochasticRealization` containing the Wiener and Poisson drivers.
After event localization, the stochastic segment is evaluated at the exact
located endpoint so restarting at a jump does not replace the global Brownian
path with a dense-interpolation approximation.

Hybrid jump integration currently rejects nontrivial `state_geometry`; jump
updates and event restarts are not yet geometry-aware.

`finite_state_generator` constructs an explicit continuous-time Markov-chain
generator only for declared finite state sets. `boundary_policy="raise"`
rejects transitions leaving the set; `"leak"` records the escaped rate.

::: phydrax.solver.JumpSolution

---

::: phydrax.solver.JumpDifferentialProblem

---

::: phydrax.solver.JumpDifferentialSolution

---

::: phydrax.solver.solve_next_reaction

---

::: phydrax.solver.solve_direct_ssa

---

::: phydrax.solver.solve_jump_differential

---

::: phydrax.solver.FiniteStateGenerator

---

::: phydrax.solver.finite_state_generator

## Semidiscrete SPDEs

Phydrax's native SPDE path is finite-dimensional method of lines:

\[
dU_t=F_h(t,U_t,a)\,dt+G_h(t,U_t,a)B\,dW_t.
\]

The spatial discretization defines the leading state axes and a matrix-free
Laplacian. A finite-rank `SpatialNoiseBasis` defines

\[
B=\Phi\operatorname{diag}(\sqrt{q_1},\ldots,\sqrt{q_r}),
\qquad
\Phi^\mathsf TM\Phi=I,
\]

where \(M\) is the spatial quadrature mass.
`SemidiscreteSPDE.wiener_realization` derives the combined noise shape and propagates
the basis fingerprint as `noise_id`. A mismatched realization and retained noise basis
therefore fail before integration.

### Spatial discretizations

`SeparableSpectralDiscretization` consumes materialized Fourier, sine, or cosine
`AxisDiscretization` objects:

| Axis basis | Laplacian | Boundary semantics |
| --- | --- | --- |
| `fourier` | FFT spectral derivative | periodic |
| `sine` | odd-extension spectral derivative | homogeneous Dirichlet |
| `cosine` | even-extension spectral derivative | homogeneous Neumann |

Uniform axes use `PreparedTensorGrid` plus `periodic_finite_difference` or an explicit
`FiniteDifferencePlan`; polynomial axes use a collocation method. The methods are not
silently reinterpreted as spectral bases.

Tensor-grid states begin with the declared spatial shape; trailing channel axes are
preserved by `laplacian`, `flatten`, and `unflatten`. `eigenpairs(rank=...)` selects
the lowest requested real modes without assembling the full tensor Laplacian.
`laplacian_matrix()` remains an explicit diagnostic for small systems.

`SpectralDiscretization` wraps a canonical
`phydrax.discretization.SpectralDecomposition`. It reuses a `ModalTransform` and the
selected `OperatorSpectrum`; transform and operator identities remain separate.

::: phydrax.discretization.AbstractStrongFormDiscretization

---

::: phydrax.discretization.SeparableSpectralDiscretization

---

::: phydrax.discretization.SpectralDiscretization

### Finite-rank spatial noise

Construct a basis from explicit weighted-orthonormal modes, a covariance
spectrum evaluated on low Laplacian modes, a nodal covariance matrix, a
continuous kernel, or a covariance matvec:

- `from_discrete_covariance` performs a dense weighted eigendecomposition and is
  intended for small matrices.
- `from_kernel_covariance` uses Matfree pivoted Cholesky. It evaluates scalar
  kernel entries on demand and stores \(O(nr)\) values rather than an
  \(n\times n\) covariance.
- `from_covariance_operator` accepts a state-shaped covariance matvec and uses a
  seeded Matfree randomized Nyström sketch. `oversampling` controls sketch
  width.

Approximate constructors attach `SpatialNoiseApproximation` at
`basis.approximation`. It records the method, requested and retained ranks,
residual kind and estimate, tolerance, convergence flag, and—when randomized—
the key data and sketch size. A non-converged factor remains inspectable rather
than silently being presented as exact. Negative covariance eigenvalues,
non-orthonormal modes, shape mismatches, and ranks larger than the discrete
state are rejected before integration.

`basis_id` hashes state shape, modes, covariance eigenvalues, quadrature,
mode IDs, and spatial discretization provenance. It changes when the grid,
rank, spectrum, modes, or randomized seed changes.

::: phydrax.stochastic.SpatialNoiseBasis

---

::: phydrax.stochastic.SpatialNoiseApproximation

### Composition and integration

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

axis = phx.discretization.FourierAxisSpec(32).materialize(0.0, 1.0)
space = phx.discretization.SeparableSpectralDiscretization((axis,))
noise = phx.stochastic.SpatialNoiseBasis.from_spectrum(
    space,
    lambda eigenvalue: 0.02 * jnp.exp(-0.05 * eigenvalue),
    rank=6,
)
initial = jnp.sin(2.0 * jnp.pi * axis.nodes)

spde = phx.solver.semidiscretize_reaction_diffusion(
    initial,
    space,
    t0=0.0,
    t1=0.2,
    kappa=0.01,
    reaction=lambda t, state, args: state - state**3,
    noise_basis=noise,
    interpretation="ito",
)
realization = spde.wiener_realization(
    jr.key(0),
    sample_shape=(128,),
    tolerance=1e-4,
    label="allen-cahn-0",
)
ensemble = phx.solver.solve_diffrax_ensemble(
    spde.problem,
    save_times=jnp.linspace(0.0, 0.2, 21),
    realization=realization,
    dt0=1e-3,
)
```

`semidiscretize_spde` accepts a general drift and either a
`SpatialNoiseBasis` or an explicit diffusion plus `noise_shape`.
`semidiscretize_reaction_diffusion` supplies
\(\kappa\Delta_hU+R(t,U,a)\) and optionally scales a basis with a scalar,
pointwise, or full diffusion amplitude. Initial state, drift, diffusion, basis,
and noise shapes are checked eagerly.

Both Itô and Stratonovich interpretations pass into `DifferentialProblem`.
Phydrax validates interpretation and Lévy-area compatibility before calling Diffrax.
These APIs solve a finite-rank semidiscrete system. They do not
claim direct infinite-dimensional white-noise integration or automatic
discretization-error uncertainty.

::: phydrax.solver.SemidiscreteSPDE

---

::: phydrax.solver.semidiscretize_spde

---

::: phydrax.solver.semidiscretize_reaction_diffusion

### Semilinear exponential integration

`semidiscretize_semilinear_spde` preserves an explicit linear/nonlinear split.
`solve_semilinear_spde` specializes to fixed-step exponential Euler with exact
compatible modal stochastic convolution for additive finite-rank Itô noise.
`phydrax.linalg.MatrixFunctionPolicy` selects exact spectral, Chebyshev, Lanczos,
or Arnoldi actions without requiring a global operator matrix. Unsupported
specializations use the validated Diffrax backend unless `fallback="error"` is
requested.

::: phydrax.solver.SemilinearDrift

---

::: phydrax.linalg.MatrixFunctionPolicy

---

::: phydrax.linalg.TransformDiagonalRepresentation

---

::: phydrax.solver.semidiscretize_semilinear_spde

---

::: phydrax.solver.solve_semilinear_spde

### Higher-order mild schemes and stochastic collocation

`solve_semilinear_spde(..., scheme="exponential_milstein")` adds the commutative
Milstein factor-JVP correction before applying the linear semigroup. It is available
only for explicitly commutative finite-rank Itô noise; unsupported structure follows
the declared `fallback` policy. `"auto"` preserves exact modal convolution for
compatible additive noise and otherwise selects exponential Euler.

`StochasticCollocationPlan` is the non-sampling alternative for a finite collection of
independent random inputs. It materializes tensor or sparse Smolyak nodes in reference
coordinates, maps them through probability-domain transforms, evaluates one supplied
deterministic solver per node, and returns normalized quadrature weights plus
node-by-node status. It does not reinterpret collocation error as process sampling
uncertainty.


::: phydrax.solver.StochasticCollocationPlan

---

::: phydrax.solver.run_stochastic_collocation

---

::: phydrax.solver.StochasticCollocationResult

### Declared SPDE solution concepts

`SPDESolutionSpec` distinguishes strong, weak, and mild formulations and records
whether the forcing is smooth, spatially truncated, or space-time white.
Pointwise strong stochastic constraints reject unregularized rough forcing
rather than evaluating a mathematically undefined residual.

::: phydrax.stochastic.SPDESolutionSpec

### Convergence and noise truncation

`SPDEConvergenceStudy` refines one axis at a time: time, space, retained noise
rank, or ensemble size. Strong/pathwise levels require one common `coupling_id`.
Weak observables carry sampling error and confidence intervals.
`NoiseTruncationStudy` keeps raw covariance truncation separate from
finite-horizon and stationary solution-aware truncation.

::: phydrax.solver.SPDEConvergenceLevel

---

::: phydrax.solver.SPDEConvergenceStudy

---

::: phydrax.solver.WeakObservableEstimate

---

::: phydrax.solver.weak_observable_estimate

---

::: phydrax.solver.NoiseTruncationLevel

---

::: phydrax.solver.NoiseTruncationStudy

## Coupled, Lévy, rough, memory, and particle dynamics

### Coupled hierarchy execution

`solve_coupled_hierarchy` runs one validated `StochasticCouplingPlan` through a
level-specific solver callback. Every adjacent result carries its shared realization,
pair IDs, coarse/fine validity, observables, and cost. This is the solver-side bridge
for strong convergence studies and multilevel estimators.

`CoupledLevelSolver` is the callback contract
`(level, realization, coarse_result, state_transfer) -> result`.


::: phydrax.solver.CoupledHierarchyResult

---

::: phydrax.solver.solve_coupled_hierarchy

### Infinite-activity Lévy equations

`LevySDEProblem` binds an explicit Lévy process to drift and jump-vector fields.
`solve_levy_sde` supports Euler or jump-adapted Euler on a fixed output grid. Small
jumps are either truncated or replaced by their declared Gaussian covariance; the
result records the cutoff, represented-jump completeness, closure, and realization
identities. No finite-variance assumption is made for the full stable process.

::: phydrax.solver.LevySDEProblem

---

::: phydrax.solver.LevySDESolution

---

::: phydrax.solver.solve_levy_sde

### Rough differential equations

`RoughDifferentialProblem` expects vector fields with shape
`state_shape + (driver_dimension,)`. The Davie scheme consumes both signature levels
and obtains directional derivatives by JAX JVP; Euler intentionally ignores level two.
Save times must be nodes of the lifted partition.

::: phydrax.solver.RoughDifferentialProblem

---

::: phydrax.solver.RoughDifferentialSolution

---

::: phydrax.solver.solve_rough_differential

### Volterra equations

The Volterra solver applies explicit left-point deterministic and stochastic
convolutions. Kernels may be scalar or state-shaped; stochastic coefficients retain a
separate declared noise shape and consume a global `WienerRealization`.

Translation-invariant convolution kernels and Caputo power-law memory have dedicated
contracts on the [delay and functional equations](delay.md) page.

Delay equations now use the unified declared-memory API documented in
[Delay and functional differential equations](delay.md). The old fixed-grid delay
problem and solver are not separate public contracts.

Run the reproducible delay benchmark with:

```console
uv run python tools/delay_benchmarks.py --repeats 10
```

::: phydrax.solver.StochasticVolterraProblem

---

::: phydrax.solver.solve_stochastic_volterra

---

::: phydrax.solver.MemoryEquationSolution

### Interacting McKean--Vlasov particles

`InteractingParticleProblem` passes the current weighted `MeanFieldSnapshot` to each
particle drift and diffusion. Idiosyncratic noise has one component per particle;
optional common noise is represented by a second global Wiener realization.
`InteractingParticleSolution` retains particle validity, empirical means and
covariances, and can expose either one selected empirical measure flow or the complete
coupled population trajectory.

::: phydrax.solver.InteractingParticleProblem

---

::: phydrax.solver.InteractingParticleSolution

---

::: phydrax.solver.solve_interacting_particles

## Result contract

`DifferentialSolution.valid` marks finite saved states. `successful` reduces that mask
across saved times for each realization. `backend_result`, `stats`, `event_mask`,
`solver_name`, stable `solver_id`, resolved-method provenance, `interpretation`,
the `WienerRealization`, geometry ID, and named Wiener-term slices preserve the
integration and stochastic provenance needed to reproduce or couple a path
ensemble.

Finite-step differential, memory, rough, Lévy, and jump result types validate one
explicit sample/time/state axis contract when they are constructed. Shared-grid
results use `times.shape == (num_times,)`, while `DifferentialSolution` retains a
time array per realization with shape `sample_shape + (num_times,)`. In both
layouts, `states` begins with `sample_shape + (num_times,)`, ends with
`state_shape`, and `valid` has shape `sample_shape + (num_times,)`. Shape or rank
mismatches are rejected before a partially valid result can be stored.

Every direct result adapter constructs `StochasticTrajectory` through the same
axis-explicit conversion contract. Physical-case and realization axes are
preserved rather than flattened, while family-specific process, solver,
geometry, discretization, and approximation provenance is retained.
Transition views can therefore be adapted directly to leakage-aware operator
datasets without reconstructing axis meaning from array shapes.

::: phydrax.solver.DifferentialSolution
    options:
        members:
            - __init__
            - num_times
            - successful
            - has_dense_interpolation
            - evaluate
            - to_predictive
            - to_stochastic_trajectory

---

::: phydrax.stochastic.StochasticTrajectory
    options:
        members:
            - __init__
            - from_solution
            - stack_cases
            - adjacent_transitions
            - transitions

---

::: phydrax.stochastic.StochasticTransitionView
    options:
        members:
            - operator_dataset
