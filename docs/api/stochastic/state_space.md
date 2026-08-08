# State-space models and transition adapters

## Contract

A `StateSpaceProblem` combines an explicit prior, transition kernel, observation model,
and `ObservationSequence`. The observation sequence owns physical case axes, timestamps,
missingness masks, schedule validity, stable case IDs, and a sequence ID. A transition
sample carries values, validity, status, and a process ID. Filtering algorithms therefore
consume one common contract without guessing whether an array axis denotes a case,
particle, ensemble member, or state component.

`state_space_key` derives semantic subkeys from the root key, operation, case ID, step,
and optional member. Batch and streaming execution consequently replay the same draws.
Changing an unrelated case, observation mask, or execution order does not silently
renumber an existing random stream.

## Canonical step context

`StateSpaceStepContext` is the single context object passed through a physical
schedule step. `case_index` is the flattened physical-case index, not a particle
or ensemble-member index. `step_index` selects the transition ending at
`observations.times[..., step_index]`: step zero starts at the case's
`initial_time`, and later steps start at the preceding scheduled time. Therefore
`transition_start_input` is the input at that physical interval's left endpoint,
while `transition_end_input` and `observation_input` are the same value at its
right endpoint. `args` is the problem payload passed through unchanged.

When an input is present, `input_breakpoints` is fixed-capacity storage and
`input_breakpoint_valid` is its explicit mask. Only masked entries are interior
breakpoints of the current transition; padding values must never be interpreted
as events. `input_signal` is the typed signal used to construct these values,
and `input_valid` records endpoint support for the step. For a solver stage at
any internal time, call `context.evaluate_input(time)`: it evaluates the same
physical case and returns an `InputEvaluation`, whose `valid` flag must be
checked before consuming `value`. Without an input signal, endpoint and
breakpoint arrays are empty, `input_breakpoint_valid` is empty,
`input_valid` is true, and `evaluate_input` raises `ValueError`.

::: phydrax.stochastic.StateSpaceStepContext

---

## Typed exogenous inputs

Every state-space input has static `case_shape`, `input_shape`, and a nonempty
`input_id`. `StateSpaceProblem(input_signal=...)` requires the input
`case_shape` to equal the observation schedule's physical `case_shape`. At
construction it evaluates both endpoints of every active transition and raises
`ValueError` if any endpoint or observation time is outside support. Inactive
schedule-padding slots impose no support requirement and are marked valid
during prevalidation so that capacity padding cannot reject a problem; the
schedule mask still excludes them from inference. The signal object and its
static `input_id` remain attached to the problem, and filtering results,
exports, and checkpoints preserve that ID as provenance.

`evaluate(time, case_index)` always returns an `InputEvaluation` rather than
silently extrapolating. Outside support, a built-in input can still return a
shape-stable boundary value, but sets `valid` to false; that value must not be
consumed. The built-in sampled and B-spline supports are closed, so both
endpoints are valid. A sampled signal's valid knots must be a nonempty
prefix for every physical case: zero-order hold needs at least one valid knot,
while linear interpolation needs at least two. Valid times must be finite and
strictly increasing, and valid values must be finite. Masked suffix entries are
capacity padding. The supported modes are exactly `"zero-order-hold"` and
`"linear"`; zero-order hold is right-continuous at an interior knot, and linear
interpolation includes both support endpoints.

`BSplineStateSpaceInput` reuses `phydrax.nn.BSplineGrid` for its fixed knot grid
and stores the case-indexed coefficients separately. Its support is the grid's
closed active interval. For both sampled and B-spline inputs,
`breakpoints(t0, t1, case_index)` returns fixed-capacity times plus a Boolean
mask selecting only strict interior breakpoints.

::: phydrax.stochastic.AbstractStateSpaceInput

---

::: phydrax.stochastic.InputEvaluation

---


::: phydrax.stochastic.SampledStateSpaceInput

---

::: phydrax.stochastic.BSplineStateSpaceInput

---

## Callback signatures

Context is explicit, positional, and never inferred. Transition methods are
`sample(key, state, t0, t1, context)` and
`log_prob(next_state, state, t0, t1, context)`. The corresponding
`CallableTransitionKernel` callbacks have those same argument orders.
Observation-model methods are `location(state, time, context)`,
`log_prob(value, state, time, mask, context)`, and
`sample(key, state, time, context, sample_shape=())`.
`CallableObservationModel` callbacks are respectively
`(state, time, context)`, `(value, state, time, mask, context)`, and
`(key, state, time, sample_shape, context)`, so callback context is always the
final positional argument. Parameter callables on Gaussian transition and
observation models likewise receive context in their documented final
position. Phydrax does not probe callback arity or guess a compatibility
signature; callbacks using an older arity must be updated explicitly.

## Priors

::: phydrax.stochastic.AbstractStatePrior

---

::: phydrax.stochastic.GaussianStatePrior

---

::: phydrax.stochastic.CategoricalStatePrior

---

::: phydrax.stochastic.DistributionStatePrior

## Observation models and schedules

::: phydrax.stochastic.ObservationSequence

---

::: phydrax.stochastic.AbstractObservationModel

---

::: phydrax.stochastic.CallableObservationModel

---

::: phydrax.stochastic.GaussianObservationModel

---

::: phydrax.stochastic.LinearGaussianObservationModel

## Transition kernels

`MarginalTransitionKernel` wraps a finite-interval marginal law. Use a pathwise adapter
when a transition must preserve one driver realization, event stream, or cocycle across
the interval.
`LinearGaussianDynamics` represents constant affine Itô dynamics
`dX = (A X + b) dt + L dW`. Its interval parameterization uses a matrix
exponential for the transition, an augmented exponential for the affine offset,
and the Van Loan construction for the covariance. Zero-duration and singular
dispersion intervals remain exact; no jitter is inserted. Pass the dynamics
object directly to `LinearGaussianTransitionKernel`. The legacy transition,
covariance, and offset constructor remains supported through the same
`LinearGaussianParameterization` contract.


::: phydrax.stochastic.AbstractTransitionKernel

---

::: phydrax.stochastic.TransitionSample

---

::: phydrax.stochastic.CallableTransitionKernel

---

::: phydrax.stochastic.MarginalTransitionKernel

---

::: phydrax.stochastic.LinearGaussianDynamics

---

::: phydrax.stochastic.LinearGaussianParameterization

---

::: phydrax.stochastic.LinearGaussianTransitionKernel

---

::: phydrax.stochastic.DifferentialTransitionKernel

---

::: phydrax.stochastic.JumpTransitionKernel

---

::: phydrax.stochastic.JumpDifferentialTransitionKernel

---

::: phydrax.stochastic.FiniteStateTransitionKernel

---

::: phydrax.stochastic.PathwiseTransitionKernel

## Structural state-space components

`compile_structural_state_space` turns named additive scalar-observation components
into the existing linear-Gaussian `StateSpaceModel` contract. Component declaration
order is the physical state order. The compiler records
`structural_component_order`, exact `structural_component_slices`, and one
`StructuralComponentProvenance` record per block in model metadata; each record keeps
the component, transition, and process-noise IDs. It does not collapse a case, time,
or state axis, infer interchangeable components, or repair an invalid specification.
The dense compiler rejects a total state size above `max_state_size` (256 by default).

`LocalLevelComponent`, `TrendComponent`, and `DampedTrendComponent` use elapsed
physical time. `SeasonalComponent` uses trigonometric rotations with an explicit
physical period and harmonic count. `RegressionComponent` accepts a fixed loading or
`design(time, context)`. `AutoregressiveComponent` is a discrete companion block.
`DeterministicTransitionComponent` accepts
`transition(t0, t1, context)` and `observation(time, context)` and has exactly zero
process covariance. Conversely, `ProcessNoiseComponent` has a zero transition matrix:
its state is independent white noise at each endpoint, not a persistent state and not
a missing-data fallback. Zero covariance, zero transition, and zero transition
probability therefore retain their exact meanings; no jitter, hidden fallback, or
repair is inserted.
Its variance and `observation_variance` enter the same endpoint marginal through
their sum. They may coexist when at least one is fixed, but estimating both without
additional structure is unidentifiable; hold one fixed or verify the parameterization
with the rank-revealing state-space identifiability report.

Structural callbacks keep `StateSpaceStepContext` in the final positional slot. The
compiled model may be used in a `StateSpaceProblem` with a typed input, so callbacks
see the same physical case, masked breakpoints, padding rules, and endpoint validity
described above. `case_shape` is preserved in the Gaussian prior. The input belongs to
the problem rather than the compiler; its nonempty `input_id` is consequently retained
by filtering, smoothing, checkpoint, and export results instead of being folded into
component metadata.

::: phydrax.stochastic.AbstractStructuralComponent

---

::: phydrax.stochastic.StructuralComponentProvenance

---

::: phydrax.stochastic.LocalLevelComponent

---

::: phydrax.stochastic.TrendComponent

---

::: phydrax.stochastic.DampedTrendComponent

---

::: phydrax.stochastic.SeasonalComponent

---

::: phydrax.stochastic.RegressionComponent

---

::: phydrax.stochastic.AutoregressiveComponent

---

::: phydrax.stochastic.DeterministicTransitionComponent

---

::: phydrax.stochastic.ProcessNoiseComponent

---

::: phydrax.stochastic.compile_structural_state_space

---

## Model and problem

::: phydrax.stochastic.StateSpaceModel

---

::: phydrax.stochastic.StateSpaceProblem

---

::: phydrax.stochastic.state_space_key
