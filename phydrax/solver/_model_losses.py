#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._model import model_objective_labels, model_objective_values


def function_model_loss_labels(functions: Any, /) -> tuple[str, ...]:
    """Return static labels for all model objective terms in a function tree."""
    return model_objective_labels(functions)


def function_model_loss_values(
    functions: Any,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    iter_: int | Array | None = None,
) -> tuple[Array, ...]:
    """Evaluate all model objective terms in a function tree."""
    iteration = None if iter_ is None else jnp.asarray(iter_)
    return model_objective_values(functions, key=key, iter_=iteration)


__all__ = [
    "function_model_loss_labels",
    "function_model_loss_values",
]
