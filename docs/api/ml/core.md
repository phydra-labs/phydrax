# Core machine-learning contracts

The top-level namespace owns the common lifecycle. `MLBatch` fixes axis, mask,
weight, group, and schema semantics. `FitResult` carries the frozen executable,
diagnostics, validity/status, resolved method, and the canonical
`phydrax.DerivativeContract` of the fit (`result.derivative_contract`).
`result.derivative_admission(request)` and `result.require_derivative(request)`
admit a `phydrax.DifferentiationRequest` against that contract, and
`phydrax.ml.fit(..., derivative_request=request)` requires it before returning;
see [Derivative contracts](../../appendix/ml_differentiability.md) and
[API → Derivative contracts and ports](../differentiation.md).
`SparseFeatures` is a fixed-width sparse-row value; dense-only recipes reject it
rather than materializing it implicitly.

`FeatureSchema(names, *, kinds=None, layout_id="", dimensions=None)` and
`TargetSchema(kind, *, names=(), class_labels=(), dimensions=None)` optionally
declare one `phydrax.units.DimensionSignature` per feature or per named target
component; target dimensions require target names. `phydrax.ml.fit` binds the
batch feature schema, and the target schema of a supervised fit, into the fitted
executable while keeping any schema the fit itself recorded, such as a learned
class vocabulary. Calling a recipe's `fit_batch` directly binds nothing. A bound
executable, and its `FitResult`, report derived ports through `model_ports()`:
the input port is the feature schema, and target-valued families (linear,
kernel-linear, naive Bayes, tree, and transformed-target models) add the target
port; `FeatureUnion` and `ColumnTransformer` output their joined feature schema,
and `Pipeline` outputs the ports of its final stage.

Status values distinguish success, insufficient data, nonfinite input,
nonconvergence, infeasibility, rank deficiency, and capacity exhaustion. An
unsupported derivative request is not a fit status: it is reported by
`DerivativeAdmission.status == phydrax.DERIVATIVE_UNSUPPORTED`. The temperature
and soft-discrete functions are explicit relaxations, not straight-through
versions of exact discrete operations.

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
require Sinkhorn: the fast PAV API also accepts `phydrax.axes.AxisArray`. None of these families
hardens its forward value or installs a straight-through gradient.

::: phydrax.ml
    options:
        members:
            - AbstractRecipe
            - FeatureKind
            - FeatureSchema
            - FitDiagnostics
            - FitResult
            - MLBatch
            - ML_CAPACITY_EXHAUSTED
            - ML_INFEASIBLE
            - ML_INSUFFICIENT_DATA
            - ML_NONCONVERGED
            - ML_NONFINITE
            - ML_RANK_DEFICIENT
            - ML_SUCCESS
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
