# Functional training runtime

`FunctionalSolver` keeps the authored scientific objective separate from any
optimizer-only training strategy.

A stateful run has three planes:

1. **Physical objective** — the authored terms and attached model losses.
2. **Optimizer surrogate** — the same prepared realizations after optional
   pseudo-transient, causal, and balancing transforms.
3. **Selection objective** — independent fixed `evaluation_terms` used for best
   model selection and early stopping.

`solver.loss(...)` always evaluates the first plane. Training diagnostics label
physical and surrogate values separately.

## One immutable update

Every outer update performs the following lifecycle:

1. refresh solver-owned adaptive populations;
2. materialize every selected integration source once;
3. update pseudo-time inverse steps when scheduled;
4. compute and detach causal slab gates;
5. update and detach residual-block multipliers;
6. freeze the optimizer surrogate;
7. reuse it for the gradient, KFAC/GGN curvature, and every line-search
   candidate;
8. run one attempt of Phydrax's internal training kernel, which decides one of
   three outcomes on device: **accepted** (commit parameters, optimizer state,
   target parameters, and the accepted-update counter), **finite rejection**
   (commit only the optimizer state the method declares, such as
   Levenberg-Marquardt damping, KFAC curvature, or a failed Riemannian line
   search; parameters stay unchanged), or **nonfinite** (roll everything back);
9. run fixed model selection when scheduled (accepted updates only);
10. publish a checkpoint only at this accepted-update boundary.

This prevents line searches from comparing candidates evaluated on different
samples, weights, or causal gates. The solver lowers its ordered total as one
kernel objective, so the floating-point sum of the authored terms is unchanged.
Parameters without an owning component slot train with surrogate authority on
physical-residual objectives. Pseudo-transient history, term multipliers, and
diagnostic gradients advance only with accepted updates. A run tolerates 64
consecutive rejected attempts; the next raises
`TrainingRejectionBudgetError`. `training_state.progress.epoch` counts attempts
and `training_state.progress.update_step` counts accepted updates.

Randomness is addressed semantically: refresh, term selection, sampling,
evaluation, NTK probes, selection, reporting, and settling each draw from their
own named site of the run's root key, folded with the attempt cursor and
microstep. A retry after a rejection therefore sees a fresh realization, and a
resumed run replays exactly the keys an uninterrupted run would use. Each
term's realization is addressed by its original term index, independent of
which terms a step selects.

## Parameter lanes

Every backend updates only the PARAMETER lane of `solver.functions`, as returned by
`solver.partition_functions()` (see
[array roles](api/phydrax.md#array-roles-and-lanes)). FIXED leaves are never
trained, and MODEL_STATE leaves are carried unchanged because no functional
objective returns a next model state. Optimizer state, delayed and EMA target
state (`target_policy`), and best-iterate selection all hold parameter-lane trees;
`training_state.kernel_state` holds the committed parameters, optimizer state
(`rule_state`), target parameters (`targets`), root key, and attempt/accepted
cursors.
Undeclared inexact leaves fail before the first update; an explicit
`parameter_subspace` declares its own selection. KFAC and
`solve_linear_trial_space(...)` reject a non-empty model-state lane.

## Optimizer choice

The training plan is optimizer-neutral. Standard Optax transformations,
`phx.optim.soap(...)`, native KFAC, and supported least-squares/GGN methods all
consume the same frozen surrogate. SOAP is Phydrax-owned and needs no optional
package. Its first gradient call initializes bounded per-axis covariance bases
and returns a zero update; subsequent updates and checkpoints retain its Adam
moments, covariances, and orthogonal bases exactly.

## Gradient accumulation

`FunctionalSolver.solve(..., gradient_accumulation=K)` is available for standard
Optax transformations. One logical update refreshes and prepares `K`
independently keyed scalar objective realizations while holding functions,
optimizer state, target state, and the one-based `iter_` schedule fixed. Each
realization has unit support; PhydraX averages their numerator gradients and
calls Optax once. `num_iter` counts optimizer updates, while
`training_state.progress.microstep` counts prepared objective realizations.

```python
trained = solver.solve(
    num_iter=1_000,
    optim=optax.adam(1e-3),
    train_term_sample_size=2,
    gradient_accumulation=4,
    keep_best=False,
)
```

Accumulation enlarges a stochastic objective sample when collocation sources or
`train_term_sample_size` vary between preparations. Repeating a completely
deterministic objective produces the same gradient and provides no memory
benefit. Term reporting averages each term only over microsteps in which that
term was selected.

Values greater than one fail before objective sampling for line searches,
least-squares/GGN, KFAC, mirror or Riemannian optimizers, distribution evolution,
pseudo-transient/causal/balancing policies, and gradient/NTK diagnostics. Those
methods require aggregate candidate-value, residual/Jacobian, curvature, or
population statistics; raw-gradient averaging is not a substitute.
Optimizer-side delayed updates are likewise not lifecycle-equivalent.

## Residual blocks

A vector residual may declare `ResidualBlockLayout` without changing its
physical Frobenius loss:

```python
blocks = phx.terms.ResidualBlockLayout(
    ("momentum_x", "momentum_y", "continuity")
)
term = phx.terms.ResidualPenalty(condition, source, blocks=blocks)
```

Training policies refer to blocks through `ResidualBlockRef(term_index,
block_name)`. Term indices and declared block names are stable; Python mapping
order is never used as a physical pairing convention.

## Pseudo-transient training

For residual block `r_j`, pseudo-transient training uses

`r_tilde_j = r_j + w_j M_j(q_current - q_previous)`.

`ResidualRelaxationMap` explicitly declares the state-to-residual map `M`.
This is required for constrained, mixed, overdetermined, and gauge systems where
a residual cannot be inferred from a similarly named field.
The map identity is formed with the canonical callable payload: StrictModule
operators and plain module-level functions are identified by content, while
opaque operators (lambdas, closures, methods, partials) must declare
`operator_semantic_id` and `operator_numeric_id`; omitting them raises
`TypeError`.

```python
pseudo = phx.solver.PseudoTransientPolicy(
    0,
    phx.solver.ResidualRelaxationMap(
        "u",
        lambda u: u,
        operator_semantic_id="identity-map",
        operator_numeric_id="identity-map",
    ),
    adaptation=phx.solver.PseudoTransientAdaptation(
        start=2,
        every=1000,
    ),
)
training = phx.solver.FunctionalTrainingPlan(
    pseudo_transient=(pseudo,),
)
```

The adaptive inverse step is the measure-weighted directional quotient
`||delta residual|| / ||delta state||`. It uses the same realization for the
current and previous fields, keeps the old value on degenerate/nonfinite
updates, applies explicit bounds, and detaches the accepted coefficient.

Fresh collocation support is part of the method contract. Per-update sources
are accepted. Slower adaptive refresh requires an explicit periodic freshness
policy. Fixed support is rejected unless the caller marks it experimental.

## Causal time slabs

`CausalResidualPolicy` uses explicit physical time boundaries and the detached
loss of every preceding slab. Initial support requires non-overlapping slabs
that partition every collocation point with positive measure.

```python
schedule = phx.sampling.collocation.CausalTimeSlabSchedule(
    (0.0, 0.25, 0.5, 0.75, 1.0),
    causal_strength=1.0,
)
causal = phx.solver.CausalResidualPolicy(0, "t", schedule)
```

The default gate signal is the unchanged physical residual. A surrogate signal
is explicit. Empty slabs, uncovered points, and invalid support fail rather
than silently changing temporal semantics.

## Gradient and NTK balancing

```python
balance = phx.solver.FunctionalTermBalancePolicy(
    (
        phx.terms.ResidualBlockRef(0, "momentum_x"),
        phx.terms.ResidualBlockRef(0, "momentum_y"),
        phx.terms.ResidualBlockRef(0, "continuity"),
    ),
    method="gradient_norm",  # or "ntk_trace"
    every=1000,
)
```

Candidate multipliers are smoothed, bounded, normalized to arithmetic mean one,
and detached. Zero/nonfinite gradient norms or statistically unresolved NTK
traces retain their previous multiplier. KFAC and GGN receive the square-root
multiplier in their residual roots, preserving consistency between scalar loss
and curvature.

Signed energies, likelihoods, posterior terms, unbiased signed estimators, and
model regularizers are not automatically balanceable.

## Gradient alignment

`FunctionalDiagnosticsPolicy(gradient_alignment=True, ...)` reports:

- intra-step alignment among selected residual-block gradients;
- inter-step cosine alignment of the complete residual gradient;
- zero or unavailable gradients as explicit nonfinite diagnostics.

No epsilon is inserted into the normalization of a zero gradient.

With `ntk=True`, the same policy periodically reports matrix-free or
resource-bounded dense NTK trace, trace uncertainty, squared trace, leading
eigenvalue, stable/effective ranks, active condition estimate, and
finite/convergence evidence.

## Selection and checkpointing

Stateful or causal training with `keep_best=True` requires
`FunctionalSelectionPolicy` and fixed `evaluation_terms`.

```python
training = phx.solver.FunctionalTrainingPlan(
    selection=phx.solver.FunctionalSelectionPolicy(every=100),
    checkpoint=phx.solver.FunctionalCheckpointPolicy(
        "checkpoints/run",
        every=1000,
    ),
)
trained = solver.solve(
    num_iter=10_000,
    optim=optax.adam(1e-3),
    training=training,
)
continued = trained.solve(
    num_iter=20_000,
    optim=optax.adam(1e-3),
    training=training,
    resume=True,
)
```

A checkpoint retains the training kernel's committed state (parameters,
optimizer state, target parameters, root key, attempt and accepted cursors,
role schema, objective and update-rule identities), current and best functions
separately, previous pseudo-time fields, adaptive coefficients, collocation
populations, update/microstep progress, gradient-accumulation identity, and run
identities. Restore rejects mismatched accumulation, training plans, target
policies, discretization bundles, roles, objectives, update rules, and array
structures; checkpoints written before the training kernel fail closed.
Periodic checkpoints are published only at accepted-update boundaries; the
final checkpoint of a run may follow a rejected attempt and records that.

## Named sharding

`FunctionalShardingPolicy` maps native sample-axis names onto a caller-owned
JAX mesh. Placement follows array roles: parameters and model state are
replicated, fixed data shards its named sample axes and replicates its other
arrays, and prepared sample fields are sharded.
Ordinary global-array reductions therefore compute one global weighted
numerator divided by one global support. Phydrax never averages already
normalized local means.

## Time windows

`FunctionalTimeWindowPlan` owns physical window boundaries and delegates the
equation-specific initial/terminal conversion to a `FunctionalWindowAdapter`.
Parameter transfer and optimizer-state transfer remain independent.
Optimizer-state transfer requires a `FunctionalTrainingPlan` in every window.
When the next window's parameter and model-state lanes match the previous
window's structure, it continues the previous kernel state: optimizer state,
target parameters, and enforcement state carry over, the adapter's functions
supply parameters and model state, and the root key is re-addressed by window
index with fresh cursors. Otherwise the window starts a fresh run. Results retain every window solver,
terminal field, seam metric, and explicit half-open interior endpoint routing;
extrapolation is rejected.

## Defect and fidelity correction

`prepare_functional_correction(...)` freezes a base field inside an `ExplicitFreeze`
holder, so its model parameters are FIXED, and trains only the
supplied correction fields against the exact nonlinear scaled residual
`R(u_base + epsilon * delta_u) / epsilon`. Optional `replacement_functions`
remain independently trainable for target-owned coefficients or fields. The returned
`FunctionalCorrectionProblem.finalize(...)` rebinds trained composed fields to the
unscaled physical objective.

Correction training requires a pure `ResidualPenalty` objective; nonquadratic scalar
terms are rejected rather than assigned an invented scaled-residual meaning. Frozen
fields retain derivative rules, so target PDE operators differentiate through the
parent with respect to physical coordinates while excluding its parameters.

`prepare_fidelity_pinn_stage(...)` binds this correction to an adjacent
`FidelityPath` relation, target data, target physics, and target-only validation. The
stage identity is part of the functional discretization bundle and therefore of
checkpoint compatibility. See the
[multi-fidelity PINN cookbook](cookbook/multifidelity_pinn.md).

## Conflict-free objective gradients

`ConflictFreeGradientPolicy` composes gradients of the scalar objective
components from one prepared realization. The implementation forms only the
small objective-space Gram matrix, uses Phydrax rank-aware pseudoinverse
factors, and returns the final projection of every objective gradient onto the
composed direction.

Set it with
`FunctionalTrainingPlan(gradient_composition=...)`. Initial support is a
standard Optax update with all terms active, one microstep, no attached model
losses, and no simultaneous term balancing. Infeasible opposite gradients are a
finite rejection that commits nothing, never an unlabeled weighted sum. Stationary and
zero-support objectives remain separately identified.

Componentized terms, including `PhysicsFlowMatchingTerm`, share one sampled
payload and expose multiple gradients without duplicating stochastic
realizations.

## Conflict-free optimizer updates

This experimental update-space contract follows the gradient--update mismatch
analysis of [Xiao et al. (2026)](https://arxiv.org/abs/2609.01558), implemented
independently against Phydrax's functional optimizer and evidence boundaries.

`ConflictFreeUpdatePolicy` constrains the direction that the optimizer actually
applies. For objective gradients `g_i`, a positive descent direction is feasible
when every real pairing `Re <g_i, d>` is nonnegative. A standard Optax transform
returns additive updates, so the runtime projects their negation and applies the
negative aligned direction.

Configure alignment independently or after gradient composition:

```python
training = phx.solver.FunctionalTrainingPlan(
    gradient_composition=phx.optim.ConflictFreeGradientPolicy(),
    update_alignment=phx.optim.ConflictFreeUpdatePolicy(),
)
```

The projector returns the original additive Optax update unchanged when it is
already feasible. Otherwise it solves a loss-dimensional dual problem and
records gradient, constructed-direction, raw-proposal, and applied-direction
conflicts; correction norms; active constraints; and KKT residuals. Cumulative
statistics live in the update rule's state
(`training_state.kernel_state.rule_state.statistics`), so checkpoint resume
preserves the complete mismatch history. An unsuccessful alignment is a finite
rejection that commits nothing.

The cone is built from authored physical objective components on the same
prepared stochastic realization. Term balancing, causal transforms, and
pseudo-transient continuation may change the optimizer proposal without changing
that physical cone. Positive component rescaling cannot change a halfspace sign,
so balancing and update alignment are complementary rather than substitutes.

Initial training integration requires standard Optax, all terms active, one
microstep, and no attached model losses. KFAC, distribution evolution,
line-search, iterative, mirror, and Riemannian backends fail before training
rather than ignoring the policy.

The guarantee is first order and realization-local. Curvature can increase a
component after a finite step, noisy batch gradients need not represent the
population objective, and splitting or grouping objective components changes
the cone. Decoupled weight decay is part of the emitted proposal and may be
partly removed when it conflicts with declared objectives.
