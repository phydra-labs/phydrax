# Decomposition, covariance, clustering, manifolds, and outliers

## Decomposition and cross-decomposition

PCA, incremental PCA, POD, truncated SVD, factor analysis, ICA, NMF,
dictionary/sparse coding, PLS, and CCA expose fixed-rank array models and
family-specific subspace diagnostics. Spectral fits separate invariant projector
derivatives from basis derivatives that require a nonzero eigengap and stable
canonical phase.

`PCA`, `TruncatedSVD`, and array `POD` accept the native `method`, `tolerance`,
`resources`, and `failure` policies. Exact `DenseSVD()` remains the key-free
default. `RandomizedSVD(oversampling=8, power_iterations=2)` requires an explicit
typed key through `fit_batch(..., key=key)` and requests certified leading
ordering. Approximation tolerances are explicit: selecting a randomized method
does not relax them or fall back to dense. Cases execute in bounded blocks and
keep their original ordering and case-key identity.

Sample weights are normalized by their total mass, not by sample count minus
one. Masks retain the existing observed-feature weighted means and zero
extension; zero physical metric weights mean unsupported coordinates, not a
singular native Hilbert metric. Component capture is measured as the norm of
the original weighted operator acting on each retained right vector. Randomized
capture fractions therefore describe the actual affine encoder/decoder
projection, not compressed singular values squared. Global rank is exposed as
`rank_evidence`, including coverage, bounds, and exact-rank availability.

The callable model is `transform`, an encoder. In default `differentiate="projector"`
mode, raw basis and individual value outputs are stopped; `project` and
`projector` have genuine invariant fit derivatives on admitted fixed support.
Use `result.derivative_admission(request, operation="project")` (or
`require_derivative`) to inspect/require that operation. Encoder/decoder fit-basis
derivatives need `"basis"` mode with isolated retained values and unique
nonzero canonical pivots. The affine mean remains differentiable in projector
mode; this is not a full encoder fit derivative. `"none"` stops fit-derived
outputs, not independent prediction-input/current-model parameter derivatives.

One current weighted basis determines physical components and every primal
projection. Fixed zero-primal compact response/core arrays carry direct-fit
projector/covariance tangents without storing a dense projector or the samples.
The direct fit certificate does not cover fit-through-optimizer basis updates.
Prediction-only/sklearn reconstruction carries no fitted provenance.
Incremental PCA remains a dense, rank-truncated merge algorithm: its existing
paired pseudo rows now carry the smooth retained covariance-factor tangent,
including earlier-chunk means, masses, and repeated positive retained clusters.
Its finite-difference reference is that merge algorithm, not full batch PCA.

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
