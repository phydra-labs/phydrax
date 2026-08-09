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
