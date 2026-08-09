# Trees, ensembles, and selection

## Trees and boosting

The hard-tree surface covers CART, extra/random trees, forests, AdaBoost,
gradient boosting, histogram boosting, and XGBoost-style first/second-order
objectives. Its array representation has fixed tree/node/category capacities and
explicit missing routing. Soft trees, soft forests, and soft boosted trees are
separate smooth models; hardening is terminal.

::: phydrax.ml.tree
    options:
        filters: ["!^_"]

## General ensembles

Bagging, random subspaces, hard/soft voting, stacking, heterogeneous/homogeneous
containers, and mixture-of-experts compositions preserve child fit results and
schema contracts. Bootstrap selection, hard voting, and expert selection remain
discrete where mathematically required.

::: phydrax.ml.ensemble
    options:
        filters: ["!^_"]

## Feature selection

Variance, score, mutual-information, recursive, sequential, and model-based
selectors return exact selected indices. Continuous sparse gates are a separately
named relaxation.

`ContinuousSparseGateRecipe` keeps `temperature` and `sparsity` as differentiable
scalar array leaves. Its fit gradients with respect to features, targets, sample
weights, and those hyperparameters are conditional, not globally smooth. The scorer
must be differentiable at the supplied batch; masks and positive effective mass must
remain fixed; normalized scores need a nonzero range with stable minimum and maximum
identities. The default absolute-correlation scorer also requires nonzero covariance
and variance. Equal-score and zero-variance inputs remain finite, but do not strengthen
that derivative claim. A later hard threshold is a disconnected exact model.

::: phydrax.ml.feature_selection
    options:
        filters: ["!^_"]
