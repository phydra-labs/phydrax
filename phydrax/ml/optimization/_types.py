#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...optim._bounds import Bounds
from ...optim._programming._mixed_integer import MixedIntegerProgram
from ...optim._programming._problem import LinearProgram


PredictorConstraintSense: TypeAlias = Literal["lower", "upper", "equal"]
PredictorOutputSemantic: TypeAlias = Literal["raw", "prediction"]
PredictorCompilationGuarantee: TypeAlias = Literal[
    "exact",
    "tolerance-qualified",
]


class PredictorInputBinding(StrictModule, NonTrainableState):
    """Affine map from canonical decisions to one frozen model input vector."""

    matrix: Array
    offset: Array
    decision_lower: Array
    decision_upper: Array
    discrete_decisions: Array
    input_lower: Array
    input_upper: Array
    feature_layout_id: str = eqx.field(static=True)
    base_structure_id: str = eqx.field(static=True)
    base_variable_count: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        offset: ArrayLike,
        decision_lower: ArrayLike,
        decision_upper: ArrayLike,
        /,
        *,
        discrete_decisions: ArrayLike | None = None,
        feature_layout_id: str,
        base_structure_id: str,
    ):
        matrix_ = jnp.asarray(matrix)
        offset_ = jnp.asarray(offset)
        lower = jnp.asarray(decision_lower)
        upper = jnp.asarray(decision_upper)
        if matrix_.ndim != 2:
            raise ValueError("matrix must have shape (features, variables).")
        features, variables = (int(size) for size in matrix_.shape)
        if features < 1 or variables < 1:
            raise ValueError("Predictor bindings require features and variables.")
        if offset_.shape != (features,):
            raise ValueError(f"offset must have shape ({features},).")
        if lower.shape != (variables,) or upper.shape != (variables,):
            raise ValueError(f"decision bounds must have shape ({variables},).")
        dtype = jnp.result_type(matrix_, offset_, lower, upper, jnp.float32)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError("Predictor bindings must use real floating arrays.")
        matrix_ = matrix_.astype(dtype)
        offset_ = offset_.astype(dtype)
        lower = lower.astype(dtype)
        upper = upper.astype(dtype)
        arrays = tuple(np.asarray(value) for value in (matrix_, offset_, lower, upper))
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("Predictor binding coefficients and bounds must be finite.")
        if np.any(arrays[2] > arrays[3]):
            raise ValueError("decision_lower cannot exceed decision_upper.")
        discrete = (
            jnp.zeros((variables,), dtype=bool)
            if discrete_decisions is None
            else jnp.asarray(discrete_decisions, dtype=bool)
        )
        if discrete.shape != (variables,):
            raise ValueError(f"discrete_decisions must have shape ({variables},).")
        positive = jnp.maximum(matrix_, 0.0)
        negative = jnp.minimum(matrix_, 0.0)
        input_lower = (
            offset_
            + ein.contract("ij,j->i", positive, lower)
            + ein.contract("ij,j->i", negative, upper)
        )
        input_upper = (
            offset_
            + ein.contract("ij,j->i", positive, upper)
            + ein.contract("ij,j->i", negative, lower)
        )
        layout = str(feature_layout_id)
        structure = str(base_structure_id)
        if not layout or not structure:
            raise ValueError("feature_layout_id and base_structure_id must be nonempty.")
        self.matrix = matrix_
        self.offset = offset_
        self.decision_lower = lower
        self.decision_upper = upper
        self.discrete_decisions = discrete
        self.input_lower = input_lower
        self.input_upper = input_upper
        self.feature_layout_id = layout
        self.base_structure_id = structure
        self.base_variable_count = variables
        self.feature_count = features
        self.binding_id = canonical_fingerprint(
            {
                "kind": "predictor-input-binding",
                "layout": layout,
                "base_structure": structure,
                "arrays": array_tree_fingerprint(
                    (matrix_, offset_, lower, upper, discrete)
                ),
            }
        )


class PredictorOutputConstraint(StrictModule, NonTrainableState):
    """One scalar frozen-predictor output relation requested by a compiler."""

    threshold: float = eqx.field(static=True)
    output_index: int = eqx.field(static=True)
    sense: PredictorConstraintSense = eqx.field(static=True)
    semantic: PredictorOutputSemantic = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        threshold: float,
        /,
        *,
        output_index: int = 0,
        sense: PredictorConstraintSense = "upper",
        semantic: PredictorOutputSemantic = "prediction",
        constraint_id: str = "learned-output-constraint",
    ):
        threshold_ = float(threshold)
        output = int(output_index)
        identifier = str(constraint_id)
        if not isfinite(threshold_):
            raise ValueError("Predictor output threshold must be finite.")
        if output < 0:
            raise ValueError("output_index must be nonnegative.")
        if sense not in ("lower", "upper", "equal"):
            raise ValueError("Unknown predictor output constraint sense.")
        if semantic not in ("raw", "prediction"):
            raise ValueError("Unknown predictor output semantic.")
        if not identifier:
            raise ValueError("constraint_id must be nonempty.")
        self.threshold = threshold_
        self.output_index = output
        self.sense = sense
        self.semantic = semantic
        self.constraint_id = identifier


class PredictorConstraintCompilation(StrictModule, NonTrainableState):
    """Canonical affine rows and auxiliaries for one frozen predictor relation."""

    auxiliary_lower: Array
    auxiliary_upper: Array
    equality_matrix: Array
    equality_rhs: Array
    inequality_matrix: Array
    inequality_rhs: Array
    binary_indices: tuple[int, ...] = eqx.field(static=True)
    output_variable: int = eqx.field(static=True)
    base_variable_count: int = eqx.field(static=True)
    auxiliary_count: int = eqx.field(static=True)
    total_variable_count: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)
    guarantee: PredictorCompilationGuarantee = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        auxiliary_lower: ArrayLike,
        auxiliary_upper: ArrayLike,
        equality_matrix: ArrayLike,
        equality_rhs: ArrayLike,
        inequality_matrix: ArrayLike,
        inequality_rhs: ArrayLike,
        /,
        *,
        binary_indices: tuple[int, ...],
        output_variable: int,
        base_variable_count: int,
        model_id: str,
        binding_id: str,
        constraint_id: str,
        guarantee: PredictorCompilationGuarantee,
    ):
        base = int(base_variable_count)
        lower = jnp.asarray(auxiliary_lower)
        upper = jnp.asarray(auxiliary_upper)
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise ValueError("Auxiliary bounds must be equal-length vectors.")
        auxiliary = int(lower.shape[0])
        total = base + auxiliary
        equality = jnp.asarray(equality_matrix)
        equality_rhs_ = jnp.asarray(equality_rhs)
        inequality = jnp.asarray(inequality_matrix)
        inequality_rhs_ = jnp.asarray(inequality_rhs)
        if equality.ndim != 2 or equality.shape[1] != total:
            raise ValueError("equality_matrix has the wrong variable dimension.")
        if inequality.ndim != 2 or inequality.shape[1] != total:
            raise ValueError("inequality_matrix has the wrong variable dimension.")
        if equality_rhs_.shape != (equality.shape[0],):
            raise ValueError("equality_rhs has the wrong shape.")
        if inequality_rhs_.shape != (inequality.shape[0],):
            raise ValueError("inequality_rhs has the wrong shape.")
        dtype = jnp.result_type(
            lower,
            upper,
            equality,
            equality_rhs_,
            inequality,
            inequality_rhs_,
            jnp.float32,
        )
        arrays = tuple(
            jnp.asarray(value, dtype=dtype)
            for value in (
                lower,
                upper,
                equality,
                equality_rhs_,
                inequality,
                inequality_rhs_,
            )
        )
        if any(not np.all(np.isfinite(np.asarray(value))) for value in arrays):
            raise ValueError("Compiled predictor fragments must be finite.")
        if np.any(np.asarray(arrays[0]) > np.asarray(arrays[1])):
            raise ValueError("Auxiliary lower bounds cannot exceed upper bounds.")
        binary = tuple(sorted(int(index) for index in binary_indices))
        if len(set(binary)) != len(binary) or any(
            index < base or index >= total for index in binary
        ):
            raise ValueError("binary_indices must be unique appended coordinates.")
        output = int(output_variable)
        if output < base or output >= total:
            raise ValueError("output_variable must identify an auxiliary coordinate.")
        identifiers = tuple(str(value) for value in (model_id, binding_id, constraint_id))
        if any(not value for value in identifiers):
            raise ValueError("Compilation identifiers must be nonempty.")
        if guarantee not in ("exact", "tolerance-qualified"):
            raise ValueError("Unknown predictor compilation guarantee.")
        (
            self.auxiliary_lower,
            self.auxiliary_upper,
            self.equality_matrix,
            self.equality_rhs,
            self.inequality_matrix,
            self.inequality_rhs,
        ) = arrays
        self.binary_indices = binary
        self.output_variable = output
        self.base_variable_count = base
        self.auxiliary_count = auxiliary
        self.total_variable_count = total
        self.model_id, self.binding_id, self.constraint_id = identifiers
        self.guarantee = guarantee
        self.compilation_id = canonical_fingerprint(
            {
                "kind": "predictor-constraint-compilation",
                "model": identifiers[0],
                "binding": identifiers[1],
                "constraint": identifiers[2],
                "guarantee": guarantee,
                "binary": list(binary),
                "output": output,
                "arrays": array_tree_fingerprint(arrays),
            }
        )


def augment_linear_program(
    base: LinearProgram | MixedIntegerProgram,
    compilation: PredictorConstraintCompilation,
    /,
    *,
    program_id: str | None = None,
) -> LinearProgram | MixedIntegerProgram:
    """Compose one predictor fragment with an existing canonical LP or MILP."""
    if not isinstance(compilation, PredictorConstraintCompilation):
        raise TypeError("compilation must be PredictorConstraintCompilation.")
    if isinstance(base, MixedIntegerProgram):
        if not isinstance(base.relaxation, LinearProgram):
            raise TypeError(
                "Predictor fragments currently augment linear relaxations only."
            )
        relaxation = base.relaxation
        integer = base.integer_indices
        binary = base.binary_indices
        base_id = base.program_id
    elif isinstance(base, LinearProgram):
        relaxation = base
        integer = ()
        binary = ()
        base_id = base.problem_id
    else:
        raise TypeError("base must be LinearProgram or a linear MixedIntegerProgram.")
    if relaxation.num_variables != compilation.base_variable_count:
        raise ValueError("Compilation base variable count does not match the program.")
    if compilation.binding_id == "":
        raise ValueError("Compilation binding identity is empty.")
    identifier = str(program_id or f"{base_id}:learned-constraint")
    auxiliary = compilation.auxiliary_count
    zero_objective = jnp.zeros((auxiliary,), dtype=relaxation.linear.dtype)
    objective = jnp.concatenate((relaxation.linear, zero_objective))
    equality_base = jnp.pad(
        relaxation.equality_matrix,
        ((0, 0), (0, auxiliary)),
    )
    inequality_base = jnp.pad(
        relaxation.inequality_matrix,
        ((0, 0), (0, auxiliary)),
    )
    equality = jnp.concatenate(
        (equality_base, compilation.equality_matrix),
        axis=0,
    )
    equality_rhs = jnp.concatenate(
        (relaxation.equality_rhs, compilation.equality_rhs),
        axis=0,
    )
    inequality = jnp.concatenate(
        (inequality_base, compilation.inequality_matrix),
        axis=0,
    )
    inequality_rhs = jnp.concatenate(
        (relaxation.inequality_rhs, compilation.inequality_rhs),
        axis=0,
    )
    lower = jnp.concatenate((relaxation.lower_bounds, compilation.auxiliary_lower))
    upper = jnp.concatenate((relaxation.upper_bounds, compilation.auxiliary_upper))
    augmented = LinearProgram(
        objective,
        equality_matrix=equality,
        equality_rhs=equality_rhs,
        inequality_matrix=inequality,
        inequality_rhs=inequality_rhs,
        bounds=Bounds(lower, upper),
        problem_id=identifier,
    )
    combined_binary = tuple(sorted((*binary, *compilation.binary_indices)))
    if not integer and not combined_binary:
        return augmented
    return MixedIntegerProgram(
        augmented,
        integer_indices=integer,
        binary_indices=combined_binary,
        program_id=identifier,
    )


__all__ = [
    "PredictorCompilationGuarantee",
    "PredictorConstraintCompilation",
    "PredictorConstraintSense",
    "PredictorInputBinding",
    "PredictorOutputConstraint",
    "PredictorOutputSemantic",
    "augment_linear_program",
]
