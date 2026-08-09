# Decomposition, covariance, clustering, manifolds, and outliers

## Decomposition and cross-decomposition

PCA, incremental PCA, POD, truncated SVD, factor analysis, ICA, NMF,
dictionary/sparse coding, PLS, and CCA expose fixed-rank array models and
family-specific subspace diagnostics. Spectral fits separate invariant projector
derivatives from basis derivatives that require a nonzero eigengap and stable
canonical phase.

::: phydrax.ml.decomposition
    options:
        filters: ["!^_"]

## Covariance estimation

::: phydrax.ml.covariance
    options:
        filters: ["!^_"]

## Mixture models

Gaussian and Bayesian Gaussian mixtures preserve explicit covariance type,
initialization, empty-component policy, fixed iteration capacity, and convergence
diagnostics. Responsibilities are smooth outputs; component identity and hard
assignments are discrete.

::: phydrax.ml.mixture
    options:
        filters: ["!^_"]

## Clustering and biclustering

The namespace separates centroid/medoid, density, graph, hierarchical, spectral,
streaming, biclustering, and soft-clustering objects. Hard assignments do not claim
a derivative; `SoftKMeans` returns a genuinely relaxed fit.

::: phydrax.ml.clustering
    options:
        filters: ["!^_"]

## Manifold learning

::: phydrax.ml.manifold
    options:
        filters: ["!^_"]

## Outlier and novelty detection

::: phydrax.ml.outliers
    options:
        filters: ["!^_"]

## Semi-supervised learning

Hard and soft label propagation, self-training, and one-class compositions have
separate types so thresholding or pseudo-label selection cannot be mistaken for a
smooth map.

::: phydrax.ml.semi_supervised
    options:
        filters: ["!^_"]
