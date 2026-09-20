#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def transfer_path_contributions(transfer_functions: ArrayLike, path_forces: ArrayLike, /):
    return jnp.asarray(transfer_functions) * jnp.asarray(path_forces)


__all__ = ["transfer_path_contributions"]
