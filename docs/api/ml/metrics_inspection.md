# Metrics and inspection

## Metrics and scorers

Every metric returns explicit value/status evidence through its documented result
contract. Exact label, order, rank, and cluster metrics are distinct from the
`smooth_*` probability, soft-order, and soft-assignment metrics. Output reduction,
averaging, gains, calibration norms, and empty/undefined policies are explicit.

::: phydrax.ml.metrics
    options:
        filters: ["!^_"]

## Selective reliability

`selective_risk_curve` accepts real `loss` and `rejection_score` arrays with
shape `case_shape + (sample,)`. A larger rejection score always means “reject
earlier”; there is no orientation switch. Equal scores are one indivisible
threshold block, so the fixed-capacity curve marks only attainable complete-block
endpoints as valid. Coverage is retained empirical mass, and the same nonnegative
`sample_weight` defines retained risk. AURC is the right-endpoint sum of risk over
successive retained-mass increments, not a trapezoidal integral or an unweighted
average over stored points.

The oracle is deliberately an explicit second evaluation:

```python
curve = phx.ml.metrics.selective_risk_curve(loss, rejection_score)
oracle = phx.ml.metrics.selective_risk_curve(loss, loss)
excess_aurc = curve.aurc - oracle.aurc
```

This keeps the second hard sort and its meaning visible. `spearman_rank_correlation`
uses weighted empirical-CDF midranks: a tied value receives the mass strictly
below it plus half its own tie-block mass. Integer weights therefore agree with
literal frequency replication.

Sorting, tie membership, threshold endpoints, masks, and validity are hard-order
boundaries. AURC and rank correlation do not supply a smooth derivative through
rejection-score or rank order: fixed-order score magnitudes do not change those
statistics, and order or tie-block changes are discrete. `score_threshold`
still records the cutoff in the supplied score units. Retained risks may
differentiate with respect to losses and weights conditional on fixed order, tie
partition, and support. Use a separately named smooth objective when that is the
required mathematical object.

`compare_paired_losses` is narrower than the curve metrics. It bootstraps aligned
one-dimensional additive per-case losses from two already-frozen methods. It
does not accept raw uncertainty scores, AURC, or Spearman correlation. The
reported effect is candidate loss minus reference loss, so lower is better.
Supplying `groups` forces whole-group resampling with one shared draw for both
methods. The central percentile interval is descriptive; the distinct one-sided
upper bound controls the decision:
`noninferior = noninferiority_upper_bound <= noninferiority_margin`.

::: phydrax.ml.metrics.selective_risk_curve

---

::: phydrax.ml.metrics.SelectiveRiskCurveResult

---

::: phydrax.ml.metrics.spearman_rank_correlation

---

::: phydrax.ml.metrics.PairedLossComparisonPlan

---

::: phydrax.ml.metrics.compare_paired_losses

---

::: phydrax.ml.metrics.PairedLossComparisonResult

## Model inspection

Gradient/Jacobian/Hessian sensitivity use the callable model's actual JAX program.
Partial dependence and permutation importance preserve case/sample geometry and
weights. Influence functions require the listed regularity of the fitted objective;
linear leverage and Cook's distance use exact model structure.

`inspect_spectral_neuron` reports the selected spectrum through invariant
cluster projectors. With eigenvalues `λ`, the selected numerical cluster uses
`τ = absolute_tolerance + relative_tolerance × max(1, maxⱼ |λⱼ|)` and includes
exactly the modes within `τ` of the model's selected eigenvalue. Exterior gaps
are measured from the cluster boundary; a missing endpoint neighbour is
reported as `+∞`.

A singleton tolerance cluster with exterior gaps greater than `τ` is reported
as numerically simple. This is conservative numerical evidence, not a claim
that two distinct eigenvalues inside `τ` are mathematically nondifferentiable.
Signed local sensitivities are returned only for a numerically simple selected
mode. Repeated or unresolved clusters instead return the basis-independent
bound `‖PAᵢP‖₂`, where `P` is the full cluster projector.

Global feature bounds `‖Aᵢ‖₂` and their perturbation enclosure are expressed in
the layer's current input units. The report never exposes a solver-selected
eigenvector or basis-dependent tie subgradient.

::: phydrax.ml.inspection.inspect_spectral_neuron

::: phydrax.ml.inspection.SpectralNeuronInspection

::: phydrax.ml.inspection
    options:
        filters: ["!^_"]
