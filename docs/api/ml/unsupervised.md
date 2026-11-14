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

Covariance estimators share one native prepared Cholesky factor between the
requested precision matrix and covariance log-determinant. A precision override
uses its factor for covariance and the negated precision log-determinant. Failed
positive-definite preparation produces nonfinite geometry and the existing
nonfinite fit status rather than a valid determinant.
Robust fitting applies the prepared precision solve to centered samples rather
than materializing an inverse for intermediate Mahalanobis distances. Dense
precision remains a requested field of the final fitted covariance model.

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

`KMeans` assigns samples with an exact blocked nearest-center search and sums
every cluster on the `phydrax.sparse` route substrate. `accumulation` is the
`RelationAccumulation` order of those sums: `"fast"` (the default) uses
unordered segment sums, so reruns on parallel backends may differ in the last
bits; `"deterministic"` and `"compensated"` fix the summation order and make
reruns bitwise identical at serial cost. `workspace_bytes` (64 MiB / 67,108,864
bytes by default, as for `MiniBatchKMeans`) bounds the transient
sample-by-center tile; smaller budgets trade loop overhead for memory. Both are
static and change recipe
identity. `StreamingKMeans.update(..., accumulation=...)` sums hard updates the
same way. `"random"` initialization draws distinct rows without replacement in
proportion to their weights; `"k-means++"` keeps a running nearest-seed distance
and repeats a seed only when every weighted row already coincides with one.

`HardClusterModel` predictions use the same exact search: the lowest index wins
exact ties and inactive clusters are never selected. Euclidean distances are
evaluated in Gram form after translating queries and centers by the masked
center mean snapped to a power-of-two grid below the center spread, so offset
coordinates keep their precision and grid-aligned (for example integer)
coordinates keep exact ties. `HardClusterModel.distances` returns exact
difference-form values, streamed over sample rows.

`MiniBatchKMeans` is EMA mini-batch k-means
([Bessa et al., 2026](https://arxiv.org/abs/2611.09501)).
Every cluster keeps exponential moving averages of its batch mass `N_j` and
batch sum; with step `α` a centroid moves toward its batch mean by
`α n_j / N_j`, so it stays in the convex hull of its points, and the first batch
a cluster receives sets it to that batch's mean. Rows are centered on their
weighted mean before fitting. `epochs` passes draw fresh permutations of the
active rows; a fractional pass covers that fraction of them and the final pass
is always whole (`epochs=2.5` runs passes of 1, 0.5 and 1). When omitted,
`batch_size` follows the fitted batch law
`B = c k^0.134749 n^0.876473 T^0.813344`, rounded to the nearest power of two
(`c = 0.00439175` for `batch_law="quality"`, capped at `n // 2`;
`c = 0.035134` for `"throughput"`, capped at the power of two nearest `n / 2`)
with `n` the static sample capacity, and `initial_step` follows
`α0 = 0.8244 k^-0.4617 B^0.9003 n^-0.4498 T^-0.1057`, clipped to `[1e-4, 1]`,
with `n` the active row count. The step decays along a cosine from `α0` to
`α0 / 10`. Both laws were fitted on 96-dimensional embeddings with
`1e5 <= n <= 1e8`, `256 <= k <= 65536`, `n / k >= 8` and `1 <= T <= 25`;
`in_fitted_domain` reports whether a fit lies inside that domain, and the
intrinsic rather than the ambient dimension decides how far the laws carry.
On the synthetic Gaussian-mixture qualification regimes, the law-chosen batch
size and step were within 6% (median 2.4%) of the best point of a local grid,
and halving the starting step often helped at five passes; pass `batch_size`
and `initial_step` explicitly to tune a known workload.
Every fifth update, `revival="split"` revives clusters whose moving removal
cost, the weighted rise of the objective if their rows moved to their
second-nearest centroid, falls below `revival_threshold` (0.1 by default) times
the mean. Donors are drawn without replacement in proportion to their moving
distortion; a child is offset from its donor by `2^-7` of the donor's RMS
radius along a random direction, so the split does not depend on the
coordinate origin, and the two halves share the donor's moving statistics.
Against an independent reference implementation, this removal-cost criterion
had the smallest worst-case and mean excess held-out error of the mass
threshold of the paper, a distortion threshold, and no revival across the
qualification regimes (`benchmarks/clustering_minibatch_qualification.py`).
`revival="off"` disables revival.
The final pass also accumulates exact per-cluster sums, and every cluster that
received rows in it becomes the exact mean of those rows; clusters without rows
are inactive in the model. When an earlier pass exists, revival stops for the
final pass, so its partition covers every active row and the closing means
reproduce the weighted data mean and second moment exactly. When the final pass
is the only one (`epochs <= 1`), revival continues through it and a slot
rewritten by a split closes over the rows it received afterwards. `objective` and
`cluster_mass` describe that last-pass partition at those means; evaluate the
fitted model for a fresh assignment. The closing step lowers the last-pass
objective by the relative `closing_gain`, and a fit converges when that gain is
at most `tolerance`; otherwise it reports `ML_NONCONVERGED`. The default 0.1 is
at least twice the 99th percentile of two-pass closing gains in qualification;
single-pass fits with few rows per cluster can exceed it.
`ClusterDiagnostics.minibatch` (`MiniBatchClusterEvidence`)
carries the static `batch_size`, `batch_law` (`None` for an explicit batch
size), `epochs` and `epoch_fractions`, and per-case `initial_step`, `updates`,
`epoch_objectives` (each pass's objective at assignment time over its rows),
`revivals`, `closing_gain`, `imbalance` (`k` times the sum of squared mass
shares) and `in_fitted_domain`; it is `None` for every other recipe. A pass
costs about one Lloyd iteration. A typed `jax.random.key` is required; the
same key replays the same permutations and revivals, and the sums follow
`accumulation`. `workspace_bytes` is divided across fitted cases.

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
