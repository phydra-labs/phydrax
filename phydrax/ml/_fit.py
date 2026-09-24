#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

from jaxtyping import ArrayLike

from .._differentiation import DifferentiationRequest
from ._batch import MLBatch
from ._contracts import AbstractRecipe, FitResult
from ._schema import FeatureSchema, TargetSchema
from ._sparse_features import SparseFeatures


def fit(
    recipe: AbstractRecipe,
    features: MLBatch | ArrayLike | SparseFeatures,
    targets: ArrayLike | None = None,
    /,
    *,
    feature_mask: ArrayLike | None = None,
    target_mask: ArrayLike | None = None,
    sample_mask: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    measure_weight: ArrayLike | None = None,
    groups: ArrayLike | None = None,
    feature_schema: FeatureSchema | None = None,
    target_schema: TargetSchema | None = None,
    key: Any = None,
    derivative_request: DifferentiationRequest | None = None,
) -> FitResult:
    """Fit one immutable recipe to a canonical batch or raw feature arrays.

    The fitted executable is bound to the batch feature schema, and to the batch
    target schema when the fit is supervised. A `derivative_request` is required
    against the fit's derivative contract before the result is returned.
    """
    if not isinstance(recipe, AbstractRecipe):
        raise TypeError("recipe must be an AbstractRecipe.")
    if isinstance(features, MLBatch):
        extras = (
            targets,
            feature_mask,
            target_mask,
            sample_mask,
            sample_weight,
            measure_weight,
            groups,
            feature_schema,
            target_schema,
        )
        if any(value is not None for value in extras):
            raise ValueError("Batch metadata cannot accompany an existing MLBatch.")
        batch = features
    else:
        batch = MLBatch(
            features,
            targets,
            feature_mask=feature_mask,
            target_mask=target_mask,
            sample_mask=sample_mask,
            sample_weight=sample_weight,
            measure_weight=measure_weight,
            groups=groups,
            feature_schema=feature_schema,
            target_schema=target_schema,
        )
    result = recipe.fit_batch(batch, key=key)
    if not isinstance(result, FitResult):
        raise TypeError("Recipe.fit_batch must return a FitResult.")
    result = result.bind_schemas(
        batch.feature_schema,
        None if batch.targets is None else batch.target_schema,
    )
    if derivative_request is not None:
        result.require_derivative(derivative_request)
    return result


__all__ = ["fit"]
