#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import log

import jax.numpy as jnp

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..linear._base import (
    AbstractGeneralizedLinearModel,
    AbstractLinearModel,
    LogisticClassifierModel,
)
from ._types import (
    PredictorConstraintCompilation,
    PredictorInputBinding,
    PredictorOutputConstraint,
)


def _model_id(model: AbstractLinearModel, /) -> str:
    return canonical_fingerprint(
        {
            "kind": type(model).__name__,
            "case_shape": list(model.case_shape),
            "target_shape": list(model.target_shape),
            "parameters": array_tree_fingerprint((model.coefficients, model.intercept)),
        }
    )


def _raw_threshold(
    model: AbstractLinearModel,
    constraint: PredictorOutputConstraint,
    /,
) -> float:
    threshold = constraint.threshold
    if constraint.semantic == "raw":
        return threshold
    if isinstance(model, LogisticClassifierModel):
        if not 0.0 < threshold < 1.0:
            raise ValueError("Logistic prediction thresholds must lie in (0, 1).")
        return log(threshold / (1.0 - threshold))
    if isinstance(model, AbstractGeneralizedLinearModel):
        if model.inverse_link == "identity":
            return threshold
        if threshold <= 0.0:
            raise ValueError("Exponential-link thresholds must be positive.")
        return log(threshold)
    return threshold


def compile_linear_predictor_constraint(
    model: AbstractLinearModel,
    binding: PredictorInputBinding,
    constraint: PredictorOutputConstraint,
    /,
) -> PredictorConstraintCompilation:
    """Compile one affine or monotone-link scalar relation exactly."""
    if not isinstance(model, AbstractLinearModel):
        raise TypeError("model must be an AbstractLinearModel.")
    if not isinstance(binding, PredictorInputBinding):
        raise TypeError("binding must be a PredictorInputBinding.")
    if not isinstance(constraint, PredictorOutputConstraint):
        raise TypeError("constraint must be a PredictorOutputConstraint.")
    if model.case_shape:
        raise ValueError("Predictor compilation requires an unbatched fitted model.")
    if model.in_size != binding.feature_count:
        raise ValueError("Binding feature count does not match the fitted model.")
    coefficients = jnp.asarray(model.coefficients).reshape((model.in_size, -1))
    intercept = jnp.asarray(model.intercept).reshape((-1,))
    if constraint.output_index >= coefficients.shape[1]:
        raise ValueError("output_index exceeds the fitted model output dimension.")
    coefficient = coefficients[:, constraint.output_index]
    bias = intercept[constraint.output_index] + ein.contract(
        "i,i->",
        coefficient,
        binding.offset,
    )
    row = ein.contract("i,ij->j", coefficient, binding.matrix)
    positive = jnp.maximum(row, 0.0)
    negative = jnp.minimum(row, 0.0)
    output_lower = (
        bias
        + ein.contract("i,i->", positive, binding.decision_lower)
        + ein.contract("i,i->", negative, binding.decision_upper)
    )
    output_upper = (
        bias
        + ein.contract("i,i->", positive, binding.decision_upper)
        + ein.contract("i,i->", negative, binding.decision_lower)
    )
    base = binding.base_variable_count
    total = base + 1
    equality_row = jnp.zeros((total,), dtype=row.dtype)
    equality_row = equality_row.at[:base].set(-row)
    equality_row = equality_row.at[base].set(1.0)
    equality = [equality_row]
    equality_rhs = [bias]
    inequality = []
    inequality_rhs = []
    threshold = _raw_threshold(model, constraint)
    output_row = jnp.zeros((total,), dtype=row.dtype).at[base].set(1.0)
    if constraint.sense == "upper":
        inequality.append(output_row)
        inequality_rhs.append(jnp.asarray(threshold, dtype=row.dtype))
    elif constraint.sense == "lower":
        inequality.append(-output_row)
        inequality_rhs.append(jnp.asarray(-threshold, dtype=row.dtype))
    else:
        equality.append(output_row)
        equality_rhs.append(jnp.asarray(threshold, dtype=row.dtype))
    equality_matrix = jnp.stack(equality)
    equality_vector = jnp.stack(equality_rhs)
    inequality_matrix = (
        jnp.stack(inequality) if inequality else jnp.empty((0, total), dtype=row.dtype)
    )
    inequality_vector = (
        jnp.stack(inequality_rhs) if inequality_rhs else jnp.empty((0,), dtype=row.dtype)
    )
    return PredictorConstraintCompilation(
        jnp.asarray([output_lower]),
        jnp.asarray([output_upper]),
        equality_matrix,
        equality_vector,
        inequality_matrix,
        inequality_vector,
        binary_indices=(),
        output_variable=base,
        base_variable_count=base,
        model_id=_model_id(model),
        binding_id=binding.binding_id,
        constraint_id=constraint.constraint_id,
        guarantee="exact",
    )


__all__ = ["compile_linear_predictor_constraint"]
