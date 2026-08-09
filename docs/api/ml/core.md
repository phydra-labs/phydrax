# Core machine-learning contracts

The top-level namespace owns the common lifecycle. `MLBatch` fixes axis, mask,
weight, group, and schema semantics. `FitResult` carries the frozen executable,
diagnostics, validity/status, resolved method, and `GradientContract`.
`SparseFeatures` is a fixed-width sparse-row value; dense-only recipes reject it
rather than materializing it implicitly.

Status values distinguish success, insufficient data, nonfinite input,
nonconvergence, infeasibility, rank deficiency, capacity exhaustion, and an
unsupported requested derivative. The temperature and soft-discrete functions are
explicit relaxations, not straight-through versions of exact discrete operations.

## Rank and top-k semantics

The similarly named APIs solve different mathematical problems. They are intentionally
not routed through one shared helper.

| API | Construction | Rank convention | Weights and axes | Conservation claim |
| --- | --- | --- | --- | --- |
| `phydrax.ml.soft_ranks` | pairwise logistic comparisons | one-based, ascending by default; optional descending | unweighted; integer axis | ranks sum to `n * (n + 1) / 2` |
| `phydrax.ml.soft_topk_weights` | logistic gate over descending pairwise ranks | membership only | unweighted; integer axis | values lie in `[0, 1]`; no general sum-to-`k` claim |
| `phydrax.ml.metrics.smooth_*` ranking metrics | metric-specific pairwise ranks and masks | metric-specific | metric-specific sample weights and masks | only the documented metric invariant |
| `phydrax.transport.soft_rank` | entropic monotone coupling | zero-based ascending barycentric rank | weighted; integer array axis or named field dimension | coupling-preserving weighted rank mass |
| `phydrax.transport.fast_soft_rank` | PAV permutahedron projection | zero-based ascending relaxed rank | unweighted; integer array axis or named field dimension | ranks sum to `n * (n - 1) / 2` |
| `phydrax.transport.soft_topk_mask` | barycentric top-bin membership | not a rank output | weighted; integer array axis or named field dimension | uniform memberships sum to `k`; weighted mean is `k / n` |

Use pairwise ML ranks for lightweight small-cardinality losses, fast PAV ranks for
unweighted larger-cardinality arrays, and transport ordering when empirical-measure
weights or a reusable monotone coupling are semantic. Named dimensions alone do not
require Sinkhorn: the fast PAV API also accepts `coordax.Field`. None of these families
hardens its forward value or installs a straight-through gradient.

::: phydrax.ml
    options:
        members:
            - AbstractRecipe
            - FeatureKind
            - FeatureSchema
            - FitDiagnostics
            - FitResult
            - GradientContract
            - MLBatch
            - ML_CAPACITY_EXHAUSTED
            - ML_INFEASIBLE
            - ML_INSUFFICIENT_DATA
            - ML_NONCONVERGED
            - ML_NONFINITE
            - ML_RANK_DEFICIENT
            - ML_SUCCESS
            - ML_UNSUPPORTED_GRADIENT
            - SparseFeatures
            - TargetKind
            - TargetSchema
            - WeightPolicy
            - fit
            - gumbel_softmax
            - masked_softmax
            - soft_ranks
            - soft_topk_weights
            - temperature_sigmoid
            - temperature_softmax
