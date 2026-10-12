# Kernels and neighbors

## Kernel methods

Kernel ridge, least-squares SVM, support-vector classification/regression,
one-class SVM, kernel PCA, Nyström approximation, random Fourier features, and
Bernoulli/categorical Gaussian-process classification share the public Phydrax
kernel protocol. Exact support selection and retained spectral rank are discrete;
solves, fixed supports, and regular spectral subspaces expose the derivative levels
recorded by each fit.

::: phydrax.ml.kernel_methods
    options:
        filters: ["!^_"]

## Neighbors, density, and metric learning

Exact k/radius neighbors expose hard top-k or radius membership. Kernel neighbors
are separately named smooth weighting models. The namespace also includes kernel
density, nearest centroids, local outlier factor, Mahalanobis fitting, and
neighborhood-components analysis.

`ExactNeighborRegressorModel.neighbor_indices` and `__call__`,
`ExactNeighborClassifierModel.predict_proba`, `predict`, and `__call__`, the
corresponding radius-neighbor prediction methods, `KernelDensityModel.log_density`,
`score_samples`, and `__call__`, and `LocalOutlierFactorModel.score_samples`,
`predict`, and `__call__` accept a positive static integer `workspace_bytes`,
defaulting to 4 MiB (4,194,304 bytes). This selects pair-reduction scratch, not a
total-memory cap on inputs, fitted state, or required outputs. The budget is
divided across fitted cases. Complete query/support tiles execute directly when
admitted; otherwise both axes stream, with a minimum tile extent of one.
`predict_chunked` retains its separate query `chunk_size` parameter and uses the
default pair-workspace budget inside each chunk.

The tile extent comes from a conservative full-row byte estimate. LOF fitting
uses the same 4 MiB budget across cases, with recipe `chunk_size` as an additional
tile upper bound and self-neighbors excluded. Top-k ties use original support
index. Gaussian density uses streaming log-sum-exp; inactive support contributes
zero, including entirely inactive cases and their query gradients.

::: phydrax.ml.neighbors
    options:
        filters: ["!^_"]
