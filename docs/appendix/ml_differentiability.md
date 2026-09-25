# Derivative contracts

This appendix defines what a derivative claim means anywhere in Phydrax and how
fitted `phydrax.ml` models state theirs. A claim is a mathematical contract, not
merely a statement that JAX can trace an implementation. Every owner — native
solvers, learned components, fitted ML models, and artifacts — uses the same
root vocabulary; the rendered reference is
[API → Derivative contracts and ports](../api/differentiation.md).

```python
from phydrax import (
    DerivativeContract,
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    GradientLevel,
    RegularityPolicy,
    SurfaceDerivative,
)
```

## Surfaces

A `DerivativeSurface` names the quantity a derivative is taken with respect to.

| Surface | Kind | Meaning |
| --- | --- | --- |
| `INPUT` | capability, value | query or call input of the map |
| `PRIMAL_STATE` | capability, value | solver primal state |
| `PHYSICAL_PARAMETER` | capability | physical coefficient of a problem |
| `SOLVER_ARGUMENT` | capability | continuous solver argument |
| `STORED_VALUES` | capability | values stored in an artifact |
| `FIT_FEATURES` | capability | training features of a fit |
| `FIT_TARGETS` | capability | training targets of a fit |
| `FIT_WEIGHTS` | capability | statistical or measure weights of a fit |
| `FIT_HYPERPARAMETERS` | capability | continuous fit hyperparameters |
| `MODEL_PARAMETER` | owned | parameters held by a component |
| `MODEL_STATE` | owned | internal state held by a component |
| `EVENT` | owned | event times or event locations |
| `STOCHASTIC_REALIZATION` | owned | a realized random draw |

`is_owned_surface(surface)` separates the two kinds:

- a **capability** surface must propagate through every participant of a
  combined map. A participant that does not declare it does not support it, so
  an undeclared capability has level `NONE`. `DerivativeContract` therefore
  drops explicit `NONE` capability entries: they carry no information;
- an **owned** surface belongs to the components that hold it. A participant
  that does not declare it does not own it. An explicit `NONE` owned entry is
  kept because it records ownership without a derivative: a hard k-means fit owns
  its centers but supplies no derivative for them.

`INPUT` and `PRIMAL_STATE` are the *value* surfaces. Declared regularity
describes the map in these arguments.

## Levels and condition resolution

`GradientLevel` orders claims from strongest to weakest:

| Level | Meaning |
| --- | --- |
| `SMOOTH` | continuously differentiable under the listed conditions |
| `ALMOST_EVERYWHERE` | differentiable except on a declared null set such as kinks or thresholds |
| `CONDITIONAL` | valid only while listed regularity, active-set, or topology conditions hold |
| `NONE` | no mathematical derivative is supplied |

`weakest_level(levels)` returns the minimum of this order. Levels are never
compared blindly: a claim carrying conditions that are not known to hold resolves
to at most `CONDITIONAL` before any comparison.

```python
import phydrax as phx

phx.resolve_gradient_level(GradientLevel.SMOOTH, conditions=("positive-gap",))
# GradientLevel.CONDITIONAL
phx.gradient_level_at_least(
    GradientLevel.SMOOTH,
    GradientLevel.SMOOTH,
    conditions=("positive-gap",),
    satisfied=("positive-gap",),
)
# True
```

A contract, its surface entries, and each admission are canonical: conditions and
nondifferentiable outputs are sorted, de-duplicated tuples and surfaces are stored
in `DerivativeSurface` order. Compare them as sets.

## Routes

`DerivativeRoute` names the mechanism that forms the declared derivatives.

- `DIRECT`: closed-form or fixed array operations differentiated by JAX.
- `IMPLICIT`: a root or KKT system differentiated through its linearization.
- `UNROLLED`: the finite iterative program actually executed is differentiated.
- `SPECTRAL`: eigenspace, singular-subspace, or projector derivatives under gap
  conditions.
- `RELAXED`: an explicitly smooth replacement for a discrete algorithm is
  differentiated.
- `EXTERNAL_ADJOINT`: an adjoint supplied by an external or dedicated solver.
- `STOPPED`: no derivative mechanism is claimed for the combined map.

The route never changes declared surface levels, but a `STOPPED` contract admits
no differentiation request: every requested level is `NONE` with the reason
`"route-stopped"`. A `STOPPED` ML fit may still declare prediction surfaces — a
RANSAC fit is stopped yet its fitted linear prediction is smooth in its input and
coefficients — and those are admitted by the fitted executable's
`model_execution_contract()`, which differentiates through the `DIRECT` route.
`authority_admits` never admits `STOPPED` as a training route.

### Direct differentiation

Closed-form and fixed array programs use ordinary JAX differentiation. Examples
include strictly regularized ridge regression, affine scaling, Gaussian moment
estimation, and smooth prediction kernels.

For weighted ridge with design `X`, target `Y`, nonnegative diagonal weights `W`,
and regularization `R`, the fitted coefficients solve

```text
(Xᴴ W X + R) B = Xᴴ W Y.
```

The direct derivative is valid when the augmented system has the reported full
rank. A zero-weight sample is excluded before arithmetic, so a nonfinite value in
that sample cannot contaminate the solve. Negative or nonfinite weights make the
fit invalid rather than being silently clipped.

Ordinary unregularized least squares is differentiable only while rank remains
constant. At a rank change, the pseudoinverse map is not continuously
differentiable; the result records rank deficiency instead of hiding it behind a
fallback. The least-squares contract therefore declares its `FIT_*` surfaces
`CONDITIONAL` with the conditions that masks and sparse index structure are fixed,
the retained singular subspace is locally constant, and every fitted augmented
design has full column rank.

### Implicit differentiation

A fitted state `z*` defined by a residual equation

```text
F(z*, data, hyperparameters) = 0
```

can be differentiated by solving the linearized system

```text
(∂F/∂z) dz = -(∂F/∂input) dinput.
```

This is used only when the relevant Jacobian or KKT system is nonsingular and the
active constraints are stable. At an active-set change, a hinge, quantile,
constrained sparse, or QP fit is generally only piecewise-smooth. An implicit
custom derivative must preserve the primal solver's tolerance and regularization
semantics; it may not substitute a different backward objective. Admission
requires classical `C¹` regularity on an implicit route (see
[Admission](#admission)).

### Unrolled differentiation

Fixed-iteration algorithms execute a shape-stable `jax.lax.scan`. The gradient is
that of the finite program actually executed, including its initialization and
number of rounds. Typical uses include proximal sparse regression, EM, soft
clustering, iterative reweighting, and differentiable boosting.

Convergence masking freezes a converged state while retaining a fixed output
structure. Diagnostics report both the fixed capacity and the iteration at which
the tolerance was first met. Nonconvergence does not silently increase the number
of iterations under `jit`. Unrolled gradients can differ numerically from the
derivative of an ideal infinite-iteration fixed point; the `UNROLLED` route makes
this distinction visible.

### Spectral differentiation

PCA, POD, CCA, spectral clustering, and related methods involve eigenspaces or
singular subspaces.

A subspace projector is invariant to sign, complex phase, and rotations inside the
subspace. Its derivative is well-defined while the retained subspace is separated
from the discarded subspace by a nonzero spectral gap. Projector-valued operations
are therefore the strongest default differentiation surface.

An individual basis is not unique:

- every real singular vector admits a sign change;
- every complex singular vector admits a unit phase;
- a repeated singular block admits an arbitrary unitary rotation.

Phydrax applies a deterministic sign/phase convention away from ties and reports
the minimum retained gap. This removes incidental sign or phase flips but cannot
make a repeated eigenspace basis uniquely differentiable. Basis-gradient claims
are conditional on the reported gap and canonical pivot remaining nondegenerate.

Selecting a rank from an energy threshold is discrete. A fixed-rank fit can be
spectrally differentiated under its gap conditions; the selected integer rank
cannot. Exact rank and retained-energy diagnostics remain available as terminal
outputs.

## Regularity algebra

`DerivativeRegularity` declares the smoothness of a map in its value surfaces:

- `continuity`: `-1` for a discontinuous map, `k >= 0` for `Cᵏ`, or `"smooth"`;
- `pieces`: the structure between non-smooth loci — `"polynomial"` pieces with a
  `degree_bound`, `"smooth"` non-polynomial pieces, or `"none"` for no declared
  decomposition;
- `conditions` and an optional `support` region (`None` is the whole domain).

The constructors `DerivativeRegularity.smooth(degree_bound=None)`,
`piecewise_polynomial(continuity=, degree_bound=)`,
`piecewise_smooth(continuity=)`, and `discontinuous()` cover the common cases.
Declarations are canonical: a `Cᵏ` piecewise polynomial of degree at most `k` is
one global polynomial and is stored as smooth, and a smooth map always has smooth
or polynomial pieces.

`admits_order(order)` returns the level and conditions of value derivatives of
that order:

| Case | Level |
| --- | --- |
| `continuity == "smooth"` or `order <= continuity` | `SMOOTH` |
| polynomial pieces with `degree_bound < order` | `NONE` (proven degenerate) |
| `continuity == -1` with `pieces == "none"` | `NONE` (proven degenerate) |
| any other order | `ALMOST_EVERYWHERE` |

From `order >= continuity + 2` the distributional derivative has a singular part
on the non-smooth locus; the almost-everywhere claim then carries the condition
`"singular-part-ignored"`. For example:

```python
from phydrax import DerivativeRegularity

relu = DerivativeRegularity.piecewise_polynomial(continuity=0, degree_bound=1)
relu.admits_order(1)  # (ALMOST_EVERYWHERE, ())
relu.admits_order(2)  # (NONE, ()) — the a.e. second derivative vanishes identically

spline = DerivativeRegularity.piecewise_polynomial(continuity=2, degree_bound=3)
spline.admits_order(2)  # (SMOOTH, ())
spline.admits_order(3)  # (ALMOST_EVERYWHERE, ())
spline.admits_order(4)  # (NONE, ())

step_like = DerivativeRegularity.piecewise_smooth(continuity=-1)
step_like.admits_order(1)  # (ALMOST_EVERYWHERE, ("singular-part-ignored",))
```

Regularity combines through three operations. Each takes the minimum continuity,
the most general piece structure (`polynomial` < `smooth` < `none`), the union of
conditions, and a common support (two different explicit supports are rejected):

| Operation | Map | Polynomial degree bound |
| --- | --- | --- |
| `a.add(b)` | sum or juxtaposition | `max(a, b)` |
| `a.multiply(b)` | product | `a + b` |
| `inner.compose(outer)` | `outer` applied after `inner` | `inner * outer` |

A degree bound survives only while both operands have polynomial pieces. So the
sum of a ReLU and a cubic spline is `C⁰` with degree bound 3, the product of two
ReLUs is `C⁰` with degree bound 2, and a ReLU of a ReLU is `C⁰` with degree
bound 1.

## Contracts

A `DerivativeContract` is the canonical static declaration of one component:

```python
contract = DerivativeContract(
    (
        SurfaceDerivative(DerivativeSurface.INPUT, GradientLevel.SMOOTH),
        SurfaceDerivative(DerivativeSurface.MODEL_PARAMETER, GradientLevel.SMOOTH),
        SurfaceDerivative(
            DerivativeSurface.FIT_FEATURES,
            GradientLevel.CONDITIONAL,
            conditions=("positive-gap",),
        ),
    ),
    route=DerivativeRoute.SPECTRAL,
    regularity=DerivativeRegularity.smooth(),
    nondifferentiable_outputs=("selected_rank",),
)
contract.level(DerivativeSurface.FIT_TARGETS)  # GradientLevel.NONE — undeclared
contract.supported_surfaces  # surfaces whose level is not NONE
```

`DerivativeContract.smooth(surfaces, route=...)` declares every listed surface
`SMOOTH` with smooth regularity. `contract_id` content-addresses the canonical
declaration, so two equal declarations share one identity. `regularity=None`
means undeclared.

## Admission

A consumer never infers support from whether `jax.grad` returns an array. It asks:

```python
request = DifferentiationRequest(
    (DerivativeSurface.INPUT, DerivativeSurface.MODEL_PARAMETER),
    order=1,
    authority=None,
)
admission = contract.admit(request, policy=RegularityPolicy())
admission.supported  # bool
admission.status  # DERIVATIVE_SUPPORTED or DERIVATIVE_UNSUPPORTED
admission.level(DerivativeSurface.INPUT)
admission.route, admission.conditions, admission.reasons
```

`DifferentiationRequest(surfaces, *, order=1, authority=None)` stores its surfaces
as a set in canonical order; `admission.levels` align with `request.surfaces`, so
read them with `admission.level(surface)`. One request may mix prediction and fit
surfaces. `authority` is the `ComponentAuthority` of the requesting owner; `None`
denotes direct eager differentiation outside scientific preparation.

Each admitted level is the weakest of the declared level and the regularity bound
computed by `admit_regularity` under the owner's `RegularityPolicy`:

1. Regularity is consulted only when the request touches a value surface
   (`INPUT`, `PRIMAL_STATE`) or the route is `IMPLICIT`; otherwise the bound is
   `SMOOTH`.
2. Declared regularity is evaluated with `admits_order(request.order)`. A proven
   degeneracy is rejected with the reason `"regularity-degenerate"`; an
   almost-everywhere level needs `RegularityPolicy(allow_almost_everywhere=True)`
   and is otherwise rejected with `"almost-everywhere-not-allowed"`; an implicit
   route needs classical `C¹` (or a `"branch-margin"` regularity condition) and is
   otherwise rejected with `"implicit-requires-c1"`.
3. Undeclared regularity depends on the authority. It is admitted for
   `authority=None` and then carries the condition `"regularity-undeclared"`. It
   is rejected with the reason `"regularity-undeclared"` for `MODEL` and
   `DISCRETIZATION` authorities. `SURROGATE`, `ACCELERATOR`, and `DECISION`
   authorities admit it only with `RegularityPolicy(allow_undeclared=True)` on a
   non-implicit route.
4. The bound applies to the value surfaces, or to every surface on an implicit
   route.

A requested surface the contract does not declare (or declares `NONE`) is rejected
with the reason `"surface-unsupported:<surface>"`, for example
`"surface-unsupported:fit-features"`, and every surface of a `STOPPED` contract is
rejected with the reason `"route-stopped"`. The admission is supported exactly
when no requested level is `NONE`; an unsupported admission names its reasons and
a supported one carries none. `admission.conditions` collect the contract,
requested-surface, and regularity conditions and qualify every admitted level —
resolve them with `gradient_level_at_least` before relying on the level.

`contract.require(request, policy=...)` returns the same admission or raises
`ValueError` whose message starts with `derivative-unsupported` and names the
unsupported surfaces and reasons:

```text
derivative-unsupported: order-1 derivatives with respect to ('fit-features',) are
unsupported (reasons: surface-unsupported:fit-features); inspect
DerivativeContract.admit before transforming.
```

### Authority

`ComponentAuthority` states what a component is trusted to decide inside its
owner, and `authority_admits(authority, route, objective_kind)` states which
(route, `ObjectiveKind`) pairs may train it:

| Authority | Admitted (route, objective) pairs |
| --- | --- |
| `ACCELERATOR` | (unrolled, algorithmic-work), (direct, supervised-proxy) |
| `DISCRETIZATION` | (unrolled, rollout), (direct, rollout), (implicit, solution-map), (direct, supervised-proxy), (direct, physical-residual) |
| `MODEL` | (direct, data-fit), (direct, physical-residual), (direct, supervised-proxy), (implicit, solution-map), (unrolled, rollout), (external-adjoint, solution-map), (external-adjoint, rollout) |
| `SURROGATE` | (direct, physical-residual), (direct, data-fit), (unrolled, rollout) |
| `DECISION` | (direct, rollout), (unrolled, rollout), (relaxed, rollout), (direct, supervised-proxy) |

An accelerator therefore never trains through an implicit solution map, and no
authority trains through a `STOPPED` route.

### Owner admission

Owners admit derivatives while planning, before any batch is traced.

**Field regularity.** The regularity of a domain field is composed over its
evaluation tree: each bound model contributes
`model_execution_contract().regularity`, constants are degree-zero polynomials,
sums and differences take the maximum degree bound, products add degree bounds,
division by a constant keeps the numerator, and known pointwise maps compose. Any
node whose regularity is undeclared, including an opaque callable such as a
`domain.Function` closure, makes the whole field undeclared. A declared field
regularity is therefore an upper bound that never hides a cancellation. For
example, `2 * u + 1` keeps the vanishing Laplacian of a ReLU network `u` with a
linear output, while `u * u` has quadratic pieces.

**Planning.** `trace_derivative_requests(residual, fields, *, authority=None,
policy=None)` admits every recorded request at its accumulated order, so a nested
Laplacian is admitted as an order-4 derivative, and attaches the
`DerivativeAdmission` to `DerivativeRequest.admission`. A derivative along a
variable the field does not depend on vanishes by independence and carries no
admission. A rejection raises `ValueError` naming the field, the order, and the
reasons.

| Owner | Authority | Policy |
| --- | --- | --- |
| `FunctionalSolver` training terms | `SURROGATE` | `regularity_policy`, default `RegularityPolicy(allow_undeclared=True)` |
| direct `partial_n` and term evaluation | `None` | almost-everywhere admitted; undeclared recorded |
| `implicit_root_result` residual components | component binding, else `MODEL` | `RegularityPolicy()` on the `IMPLICIT` route |

`FunctionalSolver` admits its training residuals at construction. Its default
policy is the frontend's declared exploratory policy: fields with undeclared
regularity train with the recorded `"regularity-undeclared"` condition, an
almost-everywhere derivative needs `RegularityPolicy(allow_almost_everywhere=True)`,
and proven degeneracy is always rejected. `FunctionalSolver.derivative_requests`
lists the admitted requests with their conditions. A ReLU network with a linear
output is rejected for a Laplacian residual even when almost-everywhere
derivatives are acknowledged. A ReLU network with a `tanh` output is admitted
only with that acknowledgment, and the admission records `"singular-part-ignored"`.

Direct eager `partial_n` rejects only proven degeneracy. It executes
almost-everywhere derivatives and records an undeclared field regularity as a
`derivative.regularity.undeclared` logging event.

Implicit root differentiation admits the model components of a structured
residual callable and of its `args`. Each component needs classical `C¹`
regularity near the root, or a `"branch-margin"` condition, and its randomness
must be deterministic or bound to one `FrozenRealization`. Newton preparation
(`NewtonKrylov`, `NewtonTrustRegion`, `prepare_nonlinear`) admits the same
randomness for the certified primal map. An opaque residual closure hides its
components. It is recorded as `"residual:determinism-undeclared"` (and, on the
implicit route, `"residual:regularity-undeclared"`) in
`NonlinearResult.component_evidence`.

## Meet and compose

Two operations combine contracts. Both combine routes the same way: equal routes
are kept, and differing routes become `STOPPED` with the condition
`"mixed-derivative-routes:<routes joined by ,>"`, so the combination admits no
request, unless an explicit `composition_route=` is supplied. Conditions and
nondifferentiable outputs are unions.

`a.meet(b, ...)` combines **parallel** parts of one map:

- a capability surface takes the weakest level over every participant, counting
  an undeclared capability as `NONE`;
- an owned surface takes the weakest level over the participants that own it and
  stays absent when none does;
- regularity is the `add` of all participants and is undeclared if any is.

`upstream.compose(downstream)` is the **sequential** map
`downstream(upstream(...))`. The downstream `INPUT` level is the passage through
which every upstream derivative reaches the output:

- `INPUT` is the weakest of both stages' `INPUT` levels;
- a capability surface is the weakest of the upstream level, the downstream
  `INPUT` level, and the downstream level (undeclared is `NONE`);
- an owned surface is the weakest of the upstream level together with the
  downstream `INPUT` level (when upstream owns it) and the downstream level (when
  downstream owns it), and stays absent when neither owns it;
- regularity is `upstream.regularity.compose(downstream.regularity)` and is
  undeclared if either is.

```python
smooth = DerivativeContract.smooth(
    (DerivativeSurface.INPUT, DerivativeSurface.MODEL_PARAMETER)
)
kinked = DerivativeContract(
    (
        SurfaceDerivative(DerivativeSurface.INPUT, GradientLevel.ALMOST_EVERYWHERE),
        SurfaceDerivative(DerivativeSurface.MODEL_PARAMETER, GradientLevel.SMOOTH),
    ),
    route=DerivativeRoute.UNROLLED,
)
smooth.meet(kinked).conditions  # ("mixed-derivative-routes:direct,unrolled",)
smooth.compose(kinked).level(DerivativeSurface.MODEL_PARAMETER)
# ALMOST_EVERYWHERE: the upstream parameters pass through the kinked input
```

ML aggregators use exactly these operations on their child fit contracts:

- `Pipeline` composes its stage contracts in order;
- `FeatureUnion` and `ColumnTransformer` meet their child contracts;
- `cross_validate` (and the outer-fold record of nested cross-validation) meets
  the fold contracts, then adds fixed-fold and differentiable-scorer conditions
  and reports `fold_indices`, `valid`, and `status` as nondifferentiable outputs;
  exact searches and the nested selection itself are `STOPPED`;
- `TransformedTargetRegressor` is `regressor.compose(view)`, where `view` is the
  fitted target transform seen as the stage applied after the regressor: its
  `FIT_TARGETS` level is the weakest of the transform's `FIT_FEATURES` and `INPUT`
  levels. The composite `FIT_TARGETS` is therefore the weakest of the transform
  `FIT_FEATURES`, the transform `INPUT`, and the regressor `FIT_TARGETS`, and
  every other composite surface is additionally weakened by the transform `INPUT`
  level.

Multiclass compositions, calibrated classifiers, and ensembles declare their own
family contracts (see the [family table](#family-contracts)) rather than meeting
child contracts.

## Branch policies

A discrete branch decision — a threshold, an event, a selected regime — has one of
six canonical differentiation semantics. `branch_policy_contract(policy,
surfaces=...)` returns the canonical contract of a `BranchDifferentiationPolicy`
on the given surfaces:

| Policy | Level | Route | Regularity | Condition |
| --- | --- | --- | --- | --- |
| `SMOOTH` | smooth | direct | smooth | none |
| `BRANCHWISE` | almost-everywhere | direct | piecewise smooth, `C⁻¹` | `executed-branch` |
| `FROZEN_DECISION` | smooth | direct | smooth | `decisions-frozen` |
| `SMOOTH_SURROGATE` | smooth | relaxed | smooth | `smooth-surrogate` |
| `EVENT_AWARE` | almost-everywhere | direct | piecewise smooth, `C⁻¹` | `transversal-events` |
| `UNSUPPORTED` | none | stopped | undeclared | none |

The conditions name what is differentiated: the executed branch, the map with
decisions held fixed, a smooth surrogate of the sharp map, or the event-aware flow
away from grazing events. Because `BRANCHWISE` and `EVENT_AWARE` declare
discontinuous piecewise-smooth regularity, their value derivatives are admitted
only under `RegularityPolicy(allow_almost_everywhere=True)` and carry
`"singular-part-ignored"`.

## Fitted ML models

A fitted predictor creates two different maps:

```text
prediction: (fitted parameters, query) -> output
fit:        (training data, weights, hyperparameters) -> fitted parameters
```

`FitResult.derivative_contract` declares both in one contract. `INPUT` and
`MODEL_PARAMETER` describe prediction; `FIT_FEATURES`, `FIT_TARGETS`,
`FIT_WEIGHTS`, and `FIT_HYPERPARAMETERS` describe the fit; the contract route is
the fit route. `MODEL_PARAMETER` is owned and always declared by ML results. A
model may have smooth prediction while its fit is nondifferentiable: hard trees
are the simplest example, because leaf values are parameters but split feature
and threshold selection are discrete. Their fit route is `STOPPED`, so
`FitResult.require_derivative` refuses every request, while
`model_execution_contract().derivative` admits leaf-value (`MODEL_PARAMETER`)
derivatives of the fitted executable. Conversely, a fitted spectral basis may be
callable through smooth matrix multiplication while the basis returned by fitting
is not uniquely differentiable at repeated singular values.

ML families do not declare `regularity`. Admitting a request that touches `INPUT`
with `authority=None` therefore adds the condition `"regularity-undeclared"`,
while a `MODEL` or `DISCRETIZATION` authority is refused with that reason until
the family declares its regularity. An implicit-route fit such as dense-QP
quantile regression consults regularity for every requested surface.

```python
import jax.numpy as jnp
import phydrax as phx

features = jnp.array([[-1.0, 0.2], [-0.4, -0.7], [0.1, 0.3], [0.8, -0.2], [1.0, 0.9]])
targets = 0.7 * features[:, 0] - 0.25 * features[:, 1] + 1.2
result = phx.ml.fit(phx.ml.linear.RidgeRecipe(alpha=1e-3), features, targets)

contract = result.derivative_contract
contract.route  # DerivativeRoute.DIRECT
contract.level(DerivativeSurface.FIT_HYPERPARAMETERS)  # GradientLevel.CONDITIONAL

prediction = DifferentiationRequest(
    (DerivativeSurface.INPUT, DerivativeSurface.MODEL_PARAMETER)
)
admission = result.derivative_admission(prediction)
admission.status  # "derivative-supported"
"regularity-undeclared" in admission.conditions  # True
```

`result.derivative_admission(request, policy=None)` and
`result.require_derivative(request, policy=None)` forward to the contract.

### Fit-time derivative preflight

`phydrax.ml.fit(..., derivative_request=request)` requires the request against
the returned contract before the result reaches the caller. An unsupported
request raises the `derivative-unsupported` `ValueError` instead of returning a
result that a later transformation would silently misuse:

```python
tree_fit = DifferentiationRequest((DerivativeSurface.FIT_FEATURES,))
phx.ml.fit(
    phx.ml.tree.DecisionTreeRegressor(max_depth=2),
    features,
    targets,
    derivative_request=tree_fit,
)
# ValueError: derivative-unsupported: ... (reasons: route-stopped, surface-unsupported:fit-features) ...
```

The contract is static, so the check runs identically inside `jax.jit` or
`jax.grad` tracing. `fit` also binds the batch feature schema, and the target
schema of a supervised fit, into the fitted executable. Calling a recipe's
low-level `fit_batch` directly neither checks a request nor binds schemas; it
remains an expert surface that JAX does not intercept.

## Discrete algorithms

The following operations have no continuous derivative through their exact choice:

- sorting and hard quantiles;
- `argmax`, hard class labels, and hard cluster assignments;
- nearest-neighbor top-k indices;
- selected feature indices;
- category discovery and vocabulary construction;
- graph connectivity and connected components;
- tree split feature, threshold, and topology;
- bootstrap or subset samples;
- early stopping that changes program length.

JAX may return a zero or branch-local derivative for part of such a program. That
does not make the discrete choice differentiable. These outputs appear in
`nondifferentiable_outputs`, and an exact fit whose learned structure is chosen
this way uses `DerivativeRoute.STOPPED`.

## Relaxed alternatives

A relaxed model is a separate mathematical model:

- softmax probabilities instead of labels;
- kernel attention instead of top-k neighbors;
- responsibilities instead of cluster indices;
- continuous feature gates instead of selected indices;
- sigmoid or sparse-continuous tree gates instead of threshold routing;
- smooth sorting or quantile approximations instead of exact order statistics.

`phydrax.transport.fast_soft_sort` and `fast_soft_rank` use an L2
permutahedron projection. Their custom JVP differentiates block averages for the
current pool-adjacent-violators partition. The map is continuous and piecewise
smooth; derivatives are conditional on that active partition, and a sufficiently
hard rank relaxation can have an exactly discrete local value with zero derivative.

Temperature and regularization remain array-valued continuous hyperparameters when
possible. `ContinuousSparseGateRecipe.temperature` and `.sparsity` are differentiable
array leaves, but their `FIT_HYPERPARAMETERS` level is `CONDITIONAL`: score
functions must be differentiable, masks and positive effective mass must remain
fixed, score normalization needs a nonzero range with stable extrema, and the
default absolute correlation must stay away from zero covariance and variance. A
temperature approaching zero can make derivatives singular or ill-conditioned.

Relaxed quantile interiors inherit their solver's regularity. Exact minimum/maximum
endpoints remain only almost-everywhere differentiable, and an absolute-discrepancy
objective has an additional kink at zero residual.

Hardening produces a new exact model and terminates the gradient path; Phydrax does
not install a straight-through gradient by default.

## Weights and masks

The fit contract distinguishes:

- statistical sample weight, which changes empirical importance;
- measure weight, which represents quadrature or a physical empirical measure;
- their explicit product;
- structural masks, which remove observations or entries before arithmetic.

`FIT_WEIGHTS` derivatives with respect to a positive weight are meaningful for
smooth weighted objectives. At zero weight, inclusion can change and the
derivative is generally one-sided or conditional, which is why nearly every family
declares `FIT_WEIGHTS` at most `CONDITIONAL`. Mask booleans and group identifiers
are structural and nondifferentiable.

## Complex values

Linear, spectral, covariance, kernel, and readout families preserve complex values
where their mathematics is defined. Conjugate products are used for Hermitian
inner products. Algorithms based on ordering, categories, tree thresholds, or
real-valued class probabilities reject complex inputs explicitly.

JAX's complex differentiation convention applies to the executable array program.
A family must still state whether its objective is real-valued and whether the
reported gradient is with respect to real parameters, complex parameters, or both.

## Diagnosing invalid derivatives

Before using a fitting gradient, inspect:

1. `result.valid` and `result.status`;
2. rank, condition number, convergence, and capacity diagnostics;
3. `result.derivative_contract.route`;
4. `result.derivative_admission(request)`: its per-surface levels, `reasons`, and
   `conditions`;
5. every listed nondifferentiable output;
6. family-specific eigengap, active-set, temperature, or topology evidence.

`DERIVATIVE_UNSUPPORTED` belongs to `DerivativeAdmission.status`, not the primal
fit status. It means at least one requested surface admits level `NONE`, and
`reasons` say why. Changing the solver, adding regularization, fixing
rank/capacity, or using an explicit relaxed model may create a valid contract;
suppressing the admission does not.

## Family contracts

Each `FitResult` stores the exact `DerivativeContract` of its fit, including the
conditions and nondifferentiable outputs; the table records the declared route and
levels. Constructor options select between the rows where noted. Legend: `S`
smooth, `AE` almost-everywhere, `C` conditional, `N` declared `NONE` (owned
surface only), `–` undeclared capability (level `NONE`). Columns: `IN` = `INPUT`,
`MP` = `MODEL_PARAMETER`, `FF`/`FT`/`FW`/`FH` = `FIT_FEATURES`/`FIT_TARGETS`/
`FIT_WEIGHTS`/`FIT_HYPERPARAMETERS`. A `stopped` fit contract admits no request;
its `IN` and `MP` levels are admitted through the fitted executable's
`model_execution_contract()`.

| Family | Route | IN | MP | FF | FT | FW | FH | Nondifferentiable outputs; principal conditions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `StandardScaler` | direct | S | S | C | – | C | – | feature masks and positive-weight support fixed |
| `MinMaxScaler` | direct | S (AE with `clip`) | S | AE | – | – | – | extremum identities and positive-weight support fixed |
| `MaxAbsScaler` | direct | S | S | AE | – | – | – | maximum-absolute-value identities fixed |
| `RobustScaler` | stopped | S | S | – | – | – | – | median, interquartile range; fitted order statistics fixed during apply |
| `NormScaler` | direct | AE | N | – | – | – | – | the zero vector maps to itself |
| `SimpleImputer` (`strategy="mean"`) | direct | C | S | C | – | C | – | missingness, masks, and positive-weight support fixed |
| `SimpleImputer` (other strategies) | stopped | C | S | – | – | – | – | imputation choice |
| `OrdinalEncoder`, `OneHotEncoder` | stopped | – | N | – | – | – | – | codes, unknown indicators |
| `TargetEncoder` | direct | – | S | – | C | C | C | category membership, unknown indicators; category membership and target masks fixed |
| `PolynomialFeatures` | direct | S | N | – | – | – | – | none |
| `SplineTransformer` | stopped | AE | C | – | – | – | – | knot spans; knot order and active spans fixed |
| `FourierFeatures` | direct | S | S | AE when `period` or `origin` is fitted, else – | – | – | – | extremum identities and positive-weight support fixed |
| `RandomFourierFeatures` (preprocessing) | direct | S | S | – | – | – | S | random frequencies and phases; explicit key fixed |
| `GaussianRandomProjection`, `SparseRandomProjection` | direct | S | S | – | – | – | – | projection draw, sparse support and signs; explicit key fixed |
| `FeatureHasher` | direct | S | N | – | – | – | – | hash routes and signs |
| `PowerTransformer` | stopped | AE | S | – | – | – | – | selected lambda |
| `QuantileTransformer` | stopped | AE | C | – | – | – | – | weighted order statistics; quantile order and interpolation intervals fixed |
| `Pipeline` | stage contracts composed in order | | | | | | | see [Meet and compose](#meet-and-compose) |
| `FeatureUnion`, `ColumnTransformer` | child contracts met | | | | | | | see [Meet and compose](#meet-and-compose) |
| `TransformedTargetRegressor` | regressor composed with the inverse-target view | | | | | | | see [Meet and compose](#meet-and-compose) |
| `OLSRecipe`, `RidgeRecipe`, `TikhonovRecipe` | direct | S | S | C | C | C | C | masks and sparse structure fixed, retained singular subspace locally constant, full-column-rank augmented design |
| `LassoRecipe`, `ElasticNetRecipe`, `GroupLassoRecipe`, `SparseGroupLassoRecipe`, `HuberRegressorRecipe`, `QuantileRegressorRecipe(solver="fixed-subgradient")`, `SGDRegressorRecipe`, `PassiveAggressiveRegressorRecipe` | unrolled | S | AE | AE | AE | C | AE | masks, sparse structure, and iteration count fixed |
| `QuantileRegressorRecipe(solver="dense-qp")` | implicit | S | S | C | C | C | C | dense-QP active set locally constant; masks and active inequality identities fixed |
| `RANSACRegressorRecipe`, `TheilSenRegressorRecipe` | stopped | S | S | – | – | – | – | selected subset, inlier mask, subset scores |
| `LogisticRegressionRecipe`, `MultinomialLogisticRegressionRecipe`, `SGDClassifierRecipe(loss="logistic")` | unrolled | S | S | S | – | C | S | `predict`, `predict_indices`; masks, sparse structure, and iteration count fixed |
| `SGDClassifierRecipe(loss="hinge")`, `PerceptronRecipe`, `PassiveAggressiveClassifierRecipe` | unrolled | S | AE | AE | – | C | AE | `predict`, `predict_indices`, mistake or margin updates |
| `PoissonRegressorRecipe`, `GammaRegressorRecipe`, `TweedieRegressorRecipe` | unrolled | S | S | S | S | C | S | masks, sparse structure, and iteration count fixed |
| `LinearDiscriminantRecipe`, `QuadraticDiscriminantRecipe`, `ShrinkageDiscriminantRecipe`, `RegularizedDiscriminantRecipe` | direct | S | S | C | – | C | C | `predict`, `predict_indices`; fixed vocabulary, positive class mass, nonsingular regularized covariance |
| `GaussianNaiveBayesRecipe`, `MultinomialNaiveBayesRecipe`, `ComplementNaiveBayesRecipe` | direct | S | S | C | – | C | C | `predict`, `predict_indices`; fixed vocabulary, positive class mass, valid feature domain |
| `BernoulliNaiveBayesRecipe`, `CategoricalNaiveBayesRecipe` | direct | AE | S | C | – | C | C | as above |
| `OneVsRestRecipe`, `OneVsOneRecipe`, `OutputCodeRecipe`, `MultilabelRecipe`, `SmoothClassifierChainRecipe` | direct | S | S | C | – | C | C | `predict`, `predict_indices`; every binary component fit valid, fixed vocabulary |
| `ClassifierChainRecipe` | direct | – | C | C | – | C | C | as above |
| `PlattCalibrationRecipe`, `TemperatureCalibrationRecipe`, `VectorCalibrationRecipe`, `MatrixCalibrationRecipe`, `MulticlassCalibrationRecipe`, `CalibratedClassifierRecipe` | unrolled | S | S | C | – | C | C | `predict`, `predict_indices`; fixed vocabulary, positive class support, finite calibration scores |
| `IsotonicCalibrationRecipe` | stopped | – | AE | – | – | – | – | `predict`, `predict_indices`; pool-adjacent-violator blocks fixed |
| `SmoothIsotonicCalibrationRecipe` | stopped | S | S | – | – | – | – | `predict`, `predict_indices` |
| `PCA`, `TruncatedSVD`, `POD` (`differentiate="projector"`, default) | spectral | S | S | C | – | C | – | retained and discarded spectra separated |
| `PCA`, `TruncatedSVD`, `POD` (`differentiate="basis"`) | spectral | S | S | C | – | C | – | non-repeated retained spectrum; unique nonzero canonicalization pivots |
| `PCA`, `TruncatedSVD`, `POD` (`differentiate="none"`) | stopped | S | S | – | – | – | – | none |
| `IncrementalPCA` | spectral | S | S | C | – | C | – | spectral separation at every merge |
| `FactorAnalysis`, `ICA` | unrolled | S | S | C | – | C | – | separated retained eigenspaces; ICA key fixed, FastICA converges without component collisions |
| `CCA`, `PLS` | spectral | S | S | C | C | C | – | separated singular subspaces or cross-covariance spectrum |
| `NMF` | unrolled | AE | S | AE | – | C | – | multiplicative iterates strictly positive |
| `SparseCoding` | unrolled | AE | AE | AE | – | – | – | active set; away from soft-threshold knots |
| `DictionaryLearning` | unrolled | AE | AE | AE | – | C | – | active set; away from soft-threshold knots, nonzero atoms |
| `KernelRidgeRecipe` | direct | S | S | C | C | C | C | support capacity and rank branch fixed |
| `LeastSquaresSVMRecipe` | direct | S | S | C | – | C | C | `predict`; binary labels fixed |
| `SupportVectorClassifierRecipe`, `OneClassSVMRecipe` (kernel methods and outliers) | unrolled | S | S | C | – | C | C | `predict`, support partition; active set and optimization path fixed |
| `SupportVectorRegressorRecipe` | unrolled | S | S | C | AE | AE | AE | away from epsilon-tube and active-mask boundaries |
| `KernelPCARecipe` | spectral | S | S | C | – | C | C | selected eigenspace separated, support mask fixed |
| `NystromRecipe` | spectral | S | S | C (`selection="even"`), – (`"random"`) | – | – | C | landmark indices; landmark selection and eigenspace rank fixed |
| `RandomFourierFeaturesRecipe` (kernel methods) | stopped | S | S | – | – | – | C | sampled frequencies; explicit key fixed |
| `GaussianProcessClassifierRecipe`, `BernoulliGaussianProcessClassifierRecipe`, `CategoricalGaussianProcessClassifierRecipe` | unrolled | S | S | C | – | C | C | `predict`; class labels and Newton iteration count fixed |
| k-neighbors and radius-neighbors regressors and classifiers | stopped | AE | AE | – | – | – | – | neighbor indices or radius membership, `predict`; tie-free top-k, no distance on the radius |
| `KernelNeighborsRegressorRecipe` | relaxed | S | S | S | S | S | S | at least one positive support weight |
| `KernelNeighborsClassifierRecipe` | relaxed | S | S | S | – | S | S | `predict`; fixed labels, a positive support weight |
| `NearestCentroidRecipe` | direct | S | S | S | – | C | – | `predict`; class membership and nonempty classes fixed |
| `KernelDensityRecipe` | direct | S | S | S | – | S | S | none |
| `LocalOutlierFactorRecipe` | stopped | AE | AE | – | – | – | – | neighbor indices, `predict`; tie-free neighbor ordering |
| `NeighborhoodComponentsAnalysisRecipe` | unrolled | S | S | S | – | C | S | labels and active sample mask fixed |
| `MahalanobisMetricRecipe` | spectral | S | S | C | – | C | C | selected eigenspace separated, labels fixed |
| `EmpiricalCovariance`, `WeightedCovariance`, `DiagonalCovariance`, `FactorCovariance`, `LedoitWolfCovariance`, `OASCovariance` | direct | S | S | C | – | C | – | active mask fixed, positive regularized covariance |
| `RobustCovariance`, `GraphicalLasso` | unrolled | S | S | C | – | C | – | active mask fixed, positive regularized covariance |
| `GaussianMixture`, `BayesianGaussianMixture` | unrolled | S | S | C | – | C | C | `predict`; initialization and active mask fixed, nondegenerate covariance or positive prior |
| `KMeans`, `KMedoids`, `MiniBatchKMeans`, `AgglomerativeClustering` | stopped | – | N | – | – | – | – | labels, assignments, medoids, sampled mini-batches, merge tree |
| `SoftKMeans`, `MeanShift`, `AffinityPropagation` | unrolled | S | S | C | – | C | C | hard labels, merged modes, exemplars; positive temperature or bandwidth, fixed initialization and iterations |
| `SpectralClustering`, `SpectralBiclustering`, `SpectralCoclustering` | spectral | S | S | C | – | C | C | hard labels, eigenvector ordering; separated eigenspace, fixed graph support and partitions |
| `DBSCAN`, `ConnectivityClustering` | stopped | C | C | – | – | – | – | labels, core mask, connected components |
| `LocallyLinearEmbeddingRecipe` | spectral | C (– for Hessian LLE and LTSA) | C (N for Hessian LLE and LTSA) | C | – | C | C | neighbor graph; simple retained eigenspaces |
| `SpectralEmbeddingRecipe`, `IsomapRecipe` | spectral | C | C | C | – | C | C | neighbor graph and shortest paths fixed; simple retained eigenvalues |
| `MultidimensionalScalingRecipe` | spectral (classical), unrolled (transductive SMACOF) | S (– for SMACOF) | S (N for SMACOF) | C | – | C | C | simple retained eigenspaces |
| `TSNERecipe` | unrolled | – | N | C | – | C | C | transductive; initialization key, iterations, and perplexity bisection fixed |
| `FuzzyGraphEmbeddingRecipe` | unrolled | C | C | C | – | C | C | k-NN topology, initialization key, and iterations fixed |
| `LabelPropagationRecipe`, `LabelSpreadingRecipe`, `SoftSelfTrainingRecipe`, `SoftOneClassCompositionRecipe` | unrolled | S | S | C | C | C | C | vocabulary and labeled mask fixed |
| `HardLabelPropagationRecipe` | stopped | – | N | – | – | – | – | class index |
| `HardSelfTrainingRecipe` | stopped | C | C | – | – | – | – | pseudo labels and their acceptance |
| `HardOneClassCompositionRecipe` | stopped | – | C | – | – | – | – | one-class acceptance |
| Hard trees, forests, and boosted trees | stopped | – | AE | – | – | – | – | split structure, leaf indices, decision paths, class labels |
| `SoftDecisionTreeRecipe`, `SoftRandomForestRecipe`, `SoftGradientBoostedTreesRecipe` | unrolled | S | S | C | C | C | C | hardened structure and feature choices; positive finite temperatures |
| `BaggingRecipe`, `RandomSubspaceRecipe`, `SoftVotingRecipe`, `StackingRecipe` | stopped | S | S | – | – | – | – | bootstrap indices, feature subspaces, fold assignment |
| `HardVotingRecipe` | stopped | – | N | – | – | – | – | majority vote |
| `MixtureOfExpertsRecipe` | unrolled | S | S | C | C | C | – | expert and gate recipes expose the corresponding fit gradients |
| Exact feature selection (variance, score, mutual-information, recursive, sequential, model-based) | stopped | S | N | – | – | – | – | selected indices and mask |
| `ContinuousSparseGateRecipe` | relaxed | S | S | C | C | C | C | see [Relaxed alternatives](#relaxed-alternatives) |
| `CovarianceOutlierRecipe` | direct | S | S | C | – | C | C | `predict`, threshold, rank; score ordering at the threshold fixed |
| `EllipticEnvelopeRecipe`, `RobustNoveltyRecipe` | unrolled | S | S | C | – | C | C | `predict`, threshold; IRLS iterations and threshold ordering fixed |
| `KernelDensityOutlierRecipe` | direct | S | S | C | – | C | S | `predict`, threshold |
| `IsolationForestRecipe` | stopped | – | N | – | – | – | – | tree topology, split features, hard paths, `predict`; `relaxed()` returns a distinct smooth model |
| `CircuitFeatureTransformRecipe` | direct | C | C | – | – | – | – | dense program and local observables valid |
| `VariationalCircuitClassifierRecipe` | stopped | C | C | – | – | – | – | `predict`; parameter shift certifies first-order Pauli-angle derivatives only |
| Split plans; `GridSearch`, `RandomSearch`, `SuccessiveHalvingSearch`, nested cross-validation results | stopped | – | N | – | – | – | – | split membership, candidate indices, surviving candidates |
| `cross_validate` | fold contracts met | | | | | | | fold indices, validity, status; differentiable scorer |
| `DifferentiableSearchAdapter` | stopped (`derivative_contract`) | – | N | – | – | – | – | `objective_derivative_contract` is the audited cross-validation contract at a fixed vector and fixed folds |

Artifacts, converters, and export are terminal serialization or copy boundaries:
they carry the recorded contract but never a derivative through the source format.

## Status and derivative precedence

`FitResult.status` is evidence about the primal fit. It is not inferred from
whether `jax.grad` returns an array. Implementations apply this conceptual
primal precedence:

1. incompatible static configuration or unsupported storage/dtype raises;
2. empty or underfull effective data reports `ML_INSUFFICIENT_DATA`;
3. active nonfinite values report `ML_NONFINITE`;
4. primal infeasibility reports `ML_INFEASIBLE`;
5. numerical rank failure reports `ML_RANK_DEFICIENT`;
6. an unfinished finite iteration reports `ML_NONCONVERGED`;
7. exhausted structural storage reports `ML_CAPACITY_EXHAUSTED`;
8. otherwise the fit reports `ML_SUCCESS`.

Derivative admission is separate. A request outside the contract yields a
`DerivativeAdmission` with status `DERIVATIVE_UNSUPPORTED` without rewriting
successful primal fit evidence.

A family can refine precedence when one condition makes another undefined, but its
diagnostics must retain the underlying evidence. Regularization can resolve a
rank-deficient raw problem while the diagnostics still report the raw singularity.

## JAX transformation boundary

The fitted array model is the unit passed to `jax.jit`, `jax.vmap`,
`jax.jacfwd`, `jax.jacrev`, or solver composition. Python estimator mutation,
source-package callbacks, data-dependent allocation, and exception-driven
algorithm replacement are outside the contract. Fixed-capacity loops keep one
shape and dtype in every carry. Case axes remain structural batch axes; they are
never inferred from a trailing output or feature dimension.
