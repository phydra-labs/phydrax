#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def transfer_surface_content(transfer_matrix: ArrayLike, source_content: ArrayLike, /):
    matrix = jnp.asarray(transfer_matrix)
    source = jnp.asarray(source_content)
    if matrix.shape[-1] != source.shape[0]:
        raise ValueError("Surface transfer does not match source.")
    return jnp.tensordot(matrix, source, axes=((1,), (0,)))


def surface_transfer_balance(
    transfer_matrix: ArrayLike, source_measure: ArrayLike, target_measure: ArrayLike, /
):
    return jnp.linalg.norm(
        jnp.asarray(transfer_matrix).T @ jnp.asarray(target_measure)
        - jnp.asarray(source_measure)
    )


__all__ = ["surface_transfer_balance", "transfer_surface_content"]
