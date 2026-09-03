#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import ArraySpace, ConstraintMap


class AbstractDiscreteDirichletConstraint(StrictModule, NonTrainableState):
    """Strong essential constraint resolved onto one prepared discrete field."""

    field_name: str = eqx.field(static=True)
    constraint_map: ConstraintMap
    constrained_dofs: Array
    free_dofs: Array
    dof_coordinates: Array
    constraint_id: str = eqx.field(static=True)

    def lift(
        self,
        values: ArrayLike | Callable[[Array], ArrayLike],
        /,
    ) -> Array:
        if callable(values):
            evaluator = cast(Callable[[Array], ArrayLike], values)
            evaluated = evaluator(self.dof_coordinates)
        else:
            evaluated = values
        raw = jnp.asarray(evaluated)
        full_space = self.constraint_map.full_space
        if not isinstance(full_space, ArraySpace):
            raise TypeError("Discrete Dirichlet lifts require an ArraySpace.")
        full_shape = full_space.shape
        full_size = int(np.prod(full_shape, dtype=int))
        if raw.shape == ():
            full = jnp.broadcast_to(raw, full_shape).reshape((full_size,))
        elif raw.shape == full_shape:
            full = raw.reshape((full_size,))
        elif raw.shape == (int(self.constrained_dofs.size),):
            full = (
                jnp.zeros((full_size,), dtype=raw.dtype)
                .at[self.constrained_dofs]
                .set(raw)
            )
        else:
            raise ValueError(
                "Dirichlet values must be scalar, full-space shaped, or contain "
                "one value per constrained coordinate."
            )
        zeros = jnp.zeros((full_size,), dtype=full.dtype)
        return (
            zeros.at[self.constrained_dofs]
            .set(full[self.constrained_dofs])
            .reshape(full_shape)
        )


__all__ = ["AbstractDiscreteDirichletConstraint"]
