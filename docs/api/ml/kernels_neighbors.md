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

::: phydrax.ml.neighbors
    options:
        filters: ["!^_"]
