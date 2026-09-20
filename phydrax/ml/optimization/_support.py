#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...optim._bounds import Bounds
from ...optim._programming._mixed_integer import MixedIntegerProgram
from ...optim._programming._problem import LinearProgram
from ._types import PredictorInputBinding


class PredictorSupportCompilation(StrictModule, NonTrainableState):
    """Exact joint convex-hull support rows with explicit witness coordinates."""

    support_points: Array
    equality_matrix: Array
    equality_rhs: Array
    auxiliary_lower: Array
    auxiliary_upper: Array
    base_variable_count: int = eqx.field(static=True)
    witness_start: int = eqx.field(static=True)
    witness_stop: int = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)


def compile_convex_hull_support(
    binding: PredictorInputBinding,
    support_points: ArrayLike,
    /,
    *,
    support_id: str,
) -> PredictorSupportCompilation:
    """Compile model inputs into the convex hull of observed feature rows."""
    if not isinstance(binding, PredictorInputBinding):
        raise TypeError("binding must be a PredictorInputBinding.")
    points = jnp.asarray(support_points, dtype=binding.matrix.dtype)
    if points.ndim != 2 or points.shape[1] != binding.feature_count:
        raise ValueError(
            "support_points must have shape (samples, binding.feature_count)."
        )
    samples = points.shape[0]
    if samples < 1 or not np.all(np.isfinite(np.asarray(points))):
        raise ValueError("support_points must be nonempty and finite.")
    identifier = str(support_id)
    if not identifier:
        raise ValueError("support_id must be nonempty.")
    base = binding.base_variable_count
    total = base + samples
    feature_rows = jnp.concatenate((binding.matrix, -points.T), axis=1)
    feature_rhs = -binding.offset
    simplex_row = jnp.concatenate(
        (
            jnp.zeros((base,), dtype=points.dtype),
            jnp.ones((samples,), dtype=points.dtype),
        )
    )
    equality = jnp.concatenate((feature_rows, simplex_row[None, :]), axis=0)
    rhs = jnp.concatenate((feature_rhs, jnp.asarray([1.0], dtype=points.dtype)))
    resolved_id = canonical_fingerprint(
        {
            "kind": "predictor-convex-hull-support",
            "binding": binding.binding_id,
            "support": identifier,
            "points": array_tree_fingerprint(points),
        }
    )
    return PredictorSupportCompilation(
        points,
        equality,
        rhs,
        jnp.zeros((samples,), dtype=points.dtype),
        jnp.ones((samples,), dtype=points.dtype),
        base,
        base,
        total,
        binding.binding_id,
        resolved_id,
    )


def augment_with_support(
    base: LinearProgram | MixedIntegerProgram,
    compilation: PredictorSupportCompilation,
    /,
    *,
    program_id: str | None = None,
) -> LinearProgram | MixedIntegerProgram:
    if not isinstance(compilation, PredictorSupportCompilation):
        raise TypeError("compilation must be PredictorSupportCompilation.")
    if isinstance(base, MixedIntegerProgram):
        relaxation = base.relaxation
        if not isinstance(relaxation, LinearProgram):
            raise TypeError(
                "Support compilation currently augments linear programs only."
            )
        integer = base.integer_indices
        binary = base.binary_indices
        base_id = base.program_id
    elif isinstance(base, LinearProgram):
        relaxation = base
        integer = ()
        binary = ()
        base_id = base.problem_id
    else:
        raise TypeError("base must be LinearProgram or linear MixedIntegerProgram.")
    if relaxation.num_variables != compilation.base_variable_count:
        raise ValueError("Support compilation variable count does not match base.")
    samples = compilation.witness_stop - compilation.witness_start
    objective = jnp.concatenate(
        (relaxation.linear, jnp.zeros((samples,), dtype=relaxation.linear.dtype))
    )
    equality = jnp.concatenate(
        (
            jnp.pad(relaxation.equality_matrix, ((0, 0), (0, samples))),
            compilation.equality_matrix,
        ),
        axis=0,
    )
    equality_rhs = jnp.concatenate((relaxation.equality_rhs, compilation.equality_rhs))
    inequality = jnp.pad(
        relaxation.inequality_matrix,
        ((0, 0), (0, samples)),
    )
    lower = jnp.concatenate((relaxation.lower_bounds, compilation.auxiliary_lower))
    upper = jnp.concatenate((relaxation.upper_bounds, compilation.auxiliary_upper))
    augmented = LinearProgram(
        objective,
        equality_matrix=equality,
        equality_rhs=equality_rhs,
        inequality_matrix=inequality,
        inequality_rhs=relaxation.inequality_rhs,
        bounds=Bounds(lower, upper),
        problem_id=str(program_id or f"{base_id}:convex-support"),
    )
    if not integer and not binary:
        return augmented
    return MixedIntegerProgram(
        augmented,
        integer_indices=integer,
        binary_indices=binary,
        program_id=str(program_id or f"{base_id}:convex-support"),
    )


__all__ = [
    "PredictorSupportCompilation",
    "augment_with_support",
    "compile_convex_hull_support",
]
