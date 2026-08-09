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

::: phydrax.ml.feature_selection
    options:
        filters: ["!^_"]
