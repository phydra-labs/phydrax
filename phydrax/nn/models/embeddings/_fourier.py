#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence
from typing import Literal

import jax.lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ...._doc import DOC_KEY0
from ..._utils import _canonical_size, _get_size, _get_value_shape, SizeLike
from .._utils import _tuple
from ..core._base import _AbstractBaseModel


class RandomFourierFeatureEmbeddings(_AbstractBaseModel):
    r"""Random Fourier feature embedding.

    Samples a (possibly multi-block) Gaussian matrix $B$ and returns

    $$
    \phi(x)=\big[\cos(Bx),\ \sin(Bx)\big].
    $$
    """

    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int

    embedding_matrix: Array
    trainable: bool

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: int = 32,
        mu: ArrayLike | Sequence[ArrayLike] = 0.0,
        sigma: ArrayLike | Sequence[ArrayLike] = 1.0,
        trainable: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        r"""**Arguments:**

        - `in_size`: Input value size. The input is flattened to a vector.
        - `out_size`: Output feature size (must be even; includes cos and sin parts).
        - `mu`: Mean for Gaussian $B$ blocks (scalar or sequence for multiscale).
        - `sigma`: Standard deviation for Gaussian $B$ blocks (scalar or sequence for multiscale).
        - `trainable`: If `True`, learns $B$; otherwise stops gradients through $B$.
        - `key`: PRNG key.
        """
        in_size_c = _canonical_size(in_size)
        in_shape = _get_value_shape(in_size_c)
        in_dim = _get_size(in_size_c)

        mu_in = _tuple(mu)
        if mu_in is None:
            raise ValueError("`mu` must not be None.")
        sigma_in = _tuple(sigma)
        if sigma_in is None:
            raise ValueError("`sigma` must not be None.")
        num_mu = len(mu_in)
        num_sigma = len(sigma_in)
        mu, sigma = mu_in * num_sigma, sigma_in * num_mu
        assert len(mu) == len(sigma)

        num_embedding_matrices = num_mu * num_sigma
        out_size = int(out_size)
        if out_size <= 0:
            raise ValueError(f"`out_size` must be positive, got {out_size}.")
        if out_size % 2 != 0:
            raise ValueError(f"`out_size` must be even, got {out_size}.")
        if out_size % (2 * num_embedding_matrices) != 0:
            divisor = 2 * num_embedding_matrices
            raise ValueError(
                "`out_size` must be divisible by "
                f"`2 * (len(mu) * len(sigma)) = {divisor}`, got {out_size}."
            )
        embedding_rows = out_size // (2 * num_embedding_matrices)

        keys = jr.split(key, num_embedding_matrices)
        embedding_matrices = [
            jr.normal(key, (embedding_rows, in_dim)) * sigma_ + mu_
            for key, mu_, sigma_ in zip(keys, mu, sigma, strict=True)
        ]

        self.in_size = in_size_c
        self.out_size = out_size
        self.embedding_matrix = jnp.concatenate(embedding_matrices, axis=0)
        self.trainable = trainable

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        x_arr = jnp.asarray(x)
        in_shape = _get_value_shape(self.in_size)
        if self.in_size == "scalar":
            if x_arr.shape == ():
                x_vec = x_arr.reshape((1,))
            elif x_arr.shape == (1,):
                x_vec = x_arr
            else:
                raise ValueError(
                    f"`x` must have scalar shape () or (1,) for in_size='scalar', got {x_arr.shape}."
                )
        else:
            if x_arr.shape != in_shape:
                raise ValueError(f"`x` must have shape {in_shape}, got {x_arr.shape}.")
            x_vec = x_arr.reshape((_get_size(self.in_size),))

        if self.trainable:
            embedding_matrix = self.embedding_matrix
        else:
            embedding_matrix = jax.lax.stop_gradient(self.embedding_matrix)

        x_embedded = embedding_matrix @ x_vec

        cos, sin = jnp.cos(x_embedded), jnp.sin(x_embedded)

        fourier_embeddings = jnp.concatenate((cos, sin))
        return fourier_embeddings
