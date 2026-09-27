from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree

import phydrax.linalg as la


type ArrayTree = PyTree[Array]


def assert_coordinate_transpose_duality(
    operator: la.AbstractLinearOperator,
    primal: ArrayTree,
    covector: ArrayTree,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Check the algebraic bilinear transpose identity in canonical coordinates."""
    image = operator.mv(primal)
    transposed = operator.transpose_mv(covector)
    left = jnp.sum(operator.target.flatten(image) * operator.target.flatten(covector))
    right = jnp.sum(operator.source.flatten(primal) * operator.source.flatten(transposed))
    np.testing.assert_allclose(left, right, rtol=rtol, atol=atol)


def assert_hilbert_adjoint_duality(
    operator: la.AbstractLinearOperator,
    primal: ArrayTree,
    target_vector: ArrayTree,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Check the conjugate Hilbert-adjoint identity under declared pairings."""
    left = operator.target.inner(operator.mv(primal), target_vector)
    right = operator.source.inner(primal, operator.adjoint_mv(target_vector))
    np.testing.assert_allclose(left, right, rtol=rtol, atol=atol)
