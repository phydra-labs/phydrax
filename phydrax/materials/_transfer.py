#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def transfer_material_field(transfer_matrix: ArrayLike, source_field: ArrayLike, /):
    matrix = jnp.asarray(transfer_matrix)
    field = jnp.asarray(source_field)
    if matrix.shape[-1] != field.shape[0]:
        raise ValueError("Transfer matrix does not match source field.")
    return jnp.tensordot(matrix, field, axes=((1,), (0,)))


def conservative_transfer_error(
    transfer_matrix: ArrayLike, source_weights: ArrayLike, target_weights: ArrayLike, /
):
    matrix = jnp.asarray(transfer_matrix)
    source = jnp.asarray(source_weights)
    target = jnp.asarray(target_weights)
    return jnp.linalg.norm(matrix.T @ target - source)


__all__ = ["conservative_transfer_error", "transfer_material_field"]
