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

`DBSCAN` and `ConnectivityClustering` accept a positive static integer
`workspace_bytes`, defaulting to 4 MiB (4,194,304 bytes), divided across fitted
cases. Inclusive-radius adjacency is reused when its pair-reduction working set
is admitted; otherwise budget-derived tiles stream with deterministic
minimum-root propagation. The budget controls pair scratch, excluding fitted
state and required outputs; the minimum tile extent is one. These recipe budgets
are static configuration and change recipe identity.

`SpectralClustering(eigen_method=None)` selects `RestartedLanczos()` on exact
streamed normalized RBF-Laplacian actions. The partial route requires
`cluster_count < sample_count` and retains `cluster_count + 1` eigenvalue/residual
entries. Native convergence, residual-separated modes, and residual intervals
that resolve `eigenvalue_tolerance` certify the retained eigenspace and component
threshold. When only one eigenvalue is omitted, its value is reconstructed from
the trace; its residual bound is the sum of retained residuals. Unresolved or
repeated modes yield `valid=False`, `converged=False`, and degeneracy evidence,
not successful labels. A scientifically otherwise admissible uncertified fit
reports `ML_NONCONVERGED`; insufficient support or empty clusters retain
`ML_INSUFFICIENT_DATA`.

Select `eigen_method=DenseEigh()` explicitly for a full eigensystem. Dense
pair/eigensolver workspace is admitted only within positive
`dense_workspace_bytes` (64 MiB / 67,108,864 bytes by default), with a `ValueError`
on refusal. Spectral `ClusterDiagnostics.eigenvalues` and `eigen_residuals` have
shape `case_shape + (cluster_count + 1,)` for the partial route and
`case_shape + (sample_count,)` for dense execution. `eigen_method` is static
metadata (`"restarted-lanczos"` or `"dense-eigh"`); the numerical evidence remains
dynamic array leaves. All three fields are `None` for non-spectral diagnostics.
Diagnostic PyTree/archive templates change with the additional evidence leaves;
restoration requires a template with the selected route's evidence shape.

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
