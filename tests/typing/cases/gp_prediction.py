"""Prepared scalar GP means keep dynamic arrays and numerical result types."""

from jax import Array
from typing_extensions import assert_type

from phydrax.uq import (
    ExactGaussianProcessFactor,
    FiniteFeatureGaussianProcessFactor,
    GaussianProcessMeanPredictor,
    prepare_gaussian_process_mean,
    SparseGaussianProcessFactor,
)


def changing_query_mean(
    factor: ExactGaussianProcessFactor
    | FiniteFeatureGaussianProcessFactor
    | SparseGaussianProcessFactor,
    residual: Array,
    query: Array,
) -> None:
    predictor = prepare_gaussian_process_mean(factor, residual, max_workspace_bytes=4096)
    assert_type(predictor, GaussianProcessMeanPredictor)
    assert_type(predictor(query), Array)
    assert_type(predictor.coefficients, Array)
    assert_type(predictor.solve_status, Array)
    assert_type(predictor.retained_storage_elements, int)
    assert_type(predictor.retained_storage_bytes, int)
    prepare_gaussian_process_mean(query, residual)  # ty: ignore[invalid-argument-type]
