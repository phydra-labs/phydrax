#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ...optim._programming._mixed_integer import MixedIntegerProgram
from ...optim._programming._problem import LinearProgram
from ._types import PredictorInputBinding


def bind_predictor_inputs(
    base: LinearProgram | MixedIntegerProgram,
    matrix: ArrayLike,
    offset: ArrayLike,
    /,
    *,
    feature_layout_id: str,
) -> PredictorInputBinding:
    """Bind a frozen model input vector to existing canonical variables."""
    if isinstance(base, MixedIntegerProgram):
        relaxation = base.relaxation
        if not isinstance(relaxation, LinearProgram):
            raise TypeError("Predictor input binding currently requires a linear base.")
        discrete = jnp.zeros((relaxation.num_variables,), dtype=bool)
        discrete = discrete.at[jnp.asarray(base.discrete_indices, dtype=jnp.int32)].set(
            True
        )
        structure_id = base.structure_id
    elif isinstance(base, LinearProgram):
        relaxation = base
        discrete = jnp.zeros((relaxation.num_variables,), dtype=bool)
        structure_id = base.structure_id
    else:
        raise TypeError("base must be LinearProgram or linear MixedIntegerProgram.")
    return PredictorInputBinding(
        matrix,
        offset,
        relaxation.lower_bounds,
        relaxation.upper_bounds,
        discrete_decisions=discrete,
        feature_layout_id=feature_layout_id,
        base_structure_id=structure_id,
    )


__all__ = ["bind_predictor_inputs"]
