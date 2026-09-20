#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike


def structural_incidence(equation_variables: ArrayLike, /):
    matrix = np.asarray(equation_variables, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError("Structural incidence must be a matrix.")
    return jnp.asarray(matrix), int(np.linalg.matrix_rank(matrix.astype(float)))


def maximum_structural_matching(incidence: ArrayLike, /):
    graph = np.asarray(incidence, dtype=bool)
    matched = [-1] * graph.shape[1]

    def augment(eq, seen):
        for var in np.flatnonzero(graph[eq]):
            if seen[var]:
                continue
            seen[var] = True
            if matched[var] < 0 or augment(matched[var], seen):
                matched[var] = eq
                return True
        return False

    count = sum(augment(eq, [False] * graph.shape[1]) for eq in range(graph.shape[0]))
    return tuple(matched), count


__all__ = ["maximum_structural_matching", "structural_incidence"]
