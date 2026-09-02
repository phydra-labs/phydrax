# Linear and probabilistic supervision

## Linear, generalized-linear, robust, sparse, and online models

This namespace includes OLS, ridge/Tikhonov, lasso, elastic-net, group and sparse
group lasso, logistic and multinomial logistic models, Poisson/Gamma/Tweedie
regression, Huber, quantile, RANSAC, Theil--Sen, SGD, perceptron, and
passive-aggressive families. Each recipe states whether fitting is direct,
implicit, unrolled, or structurally stopped.

::: phydrax.ml.linear
    options:
        filters: ["!^_"]

## Exact circuit learning

`phydrax.ml.quantum` provides dense circuit state and local-observable feature
models, exact autodiff and Pauli parameter-shift modes, schema-bound feature
transforms, a binary variational classifier recipe, and standard IQP and
data-reuploading constructions. The models lower through the canonical quantum
program lifecycle and do not call an external provider.

::: phydrax.ml.quantum
    options:
        filters: ["!^_"]

## Discriminant analysis

::: phydrax.ml.discriminant
    options:
        filters: ["!^_"]

## Naive Bayes

::: phydrax.ml.naive_bayes
    options:
        filters: ["!^_"]

## Multiclass, multilabel, and classifier-chain composition

::: phydrax.ml.multiclass
    options:
        filters: ["!^_"]

## Probability calibration

Platt, temperature, vector, matrix, isotonic, and smooth-isotonic calibration are
distinct models. Hard isotonic block construction is not presented as a smooth
fit; its relaxed counterpart is separately named.

::: phydrax.ml.calibration
    options:
        filters: ["!^_"]
