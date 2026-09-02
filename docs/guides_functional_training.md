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
8. commit one accepted iterate;
9. run fixed model selection when scheduled;
10. publish a checkpoint only at this accepted-update boundary.

This prevents line searches from comparing candidates evaluated on different
samples, weights, or causal gates.

## Optimizer choice

The training plan is optimizer-neutral. Standard Optax transformations,
`phx.optim.soap(...)`, native KFAC, and supported least-squares/GGN methods all
consume the same frozen surrogate. SOAP is Phydrax-owned and needs no optional
package. Its first gradient call initializes bounded per-axis covariance bases
and returns a zero update; subsequent updates and checkpoints retain its Adam
moments, covariances, and orthogonal bases exactly.

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

```python
pseudo = phx.solver.PseudoTransientPolicy(
    0,
    phx.solver.ResidualRelaxationMap("u", lambda u: u),
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

A checkpoint retains current and best functions separately, optimizer state,
previous pseudo-time fields, adaptive coefficients, collocation populations,
PRNG state, progress, and run identities. Restore rejects mismatched training
plans and discretization bundles.

## Named sharding

`FunctionalShardingPolicy` maps coordax sample-axis names onto a caller-owned
JAX mesh. Parameters are replicated; prepared sample fields are sharded.
Ordinary global-array reductions therefore compute one global weighted
numerator divided by one global support. Phydrax never averages already
normalized local means.

## Time windows

`FunctionalTimeWindowPlan` owns physical window boundaries and delegates the
equation-specific initial/terminal conversion to a `FunctionalWindowAdapter`.
Parameter transfer and optimizer-state transfer remain independent.
Optimizer-state transfer requires a `FunctionalTrainingPlan` in every window
and identical parameter PyTree structure. Results retain every window solver,
terminal field, seam metric, and explicit half-open interior endpoint routing;
extrapolation is rejected.

## Defect correction

`prepare_functional_correction(...)` freezes a base field and trains only the
supplied correction fields against the exact nonlinear scaled residual
`R(u_base + epsilon * delta_u) / epsilon`. The returned
`FunctionalCorrectionProblem.finalize(...)` rebinds the trained composed fields
to the unscaled physical objective.
Correction training currently requires a pure `ResidualPenalty` objective;
nonquadratic scalar terms are rejected rather than assigned an invented scaled
residual meaning.
