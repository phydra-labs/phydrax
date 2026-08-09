"""Shared numerical kernels for native Phydrax ML families."""

from ._histogram import (
    assign_bins,
    histogram_gradient_statistics,
    quantile_bin_edges,
    xgboost_leaf_weight,
    xgboost_split_gain,
)
from ._iterative import (
    group_soft_threshold,
    IterationResult,
    project_simplex,
    run_fixed_iterations,
    soft_threshold,
)
from ._least_squares import LeastSquaresResult, solve_weighted_least_squares
from ._pairwise import (
    chunked_pairwise_apply,
    hard_assignments,
    MetricName,
    pairwise_distances,
    soft_assignments,
    squared_euclidean_distances,
)
from ._spectral import fit_weighted_subspace, SpectralFitResult
from ._weighted import (
    class_weighted_moments,
    effective_sample_size,
    safe_weighted_values,
    segmented_weighted_mean,
    segmented_weighted_sum,
    weighted_covariance,
    weighted_mean,
    weighted_sum,
)


__all__ = [
    "IterationResult",
    "LeastSquaresResult",
    "MetricName",
    "SpectralFitResult",
    "assign_bins",
    "chunked_pairwise_apply",
    "class_weighted_moments",
    "effective_sample_size",
    "fit_weighted_subspace",
    "group_soft_threshold",
    "hard_assignments",
    "histogram_gradient_statistics",
    "pairwise_distances",
    "project_simplex",
    "quantile_bin_edges",
    "run_fixed_iterations",
    "segmented_weighted_mean",
    "segmented_weighted_sum",
    "safe_weighted_values",
    "soft_assignments",
    "soft_threshold",
    "solve_weighted_least_squares",
    "squared_euclidean_distances",
    "weighted_covariance",
    "weighted_mean",
    "weighted_sum",
    "xgboost_leaf_weight",
    "xgboost_split_gain",
]
