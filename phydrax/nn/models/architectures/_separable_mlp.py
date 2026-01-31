#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..._utils import _get_size
from ..core._base import _AbstractStructuredInputModel
from ..wrappers._separable_wrappers import Separable
from ._mlp import MLP


class SeparableMLP(_AbstractStructuredInputModel):
    r"""Separable MLP over coordinate-wise scalar submodels.

    This builds one scalar-input `MLP` per coordinate (and per `split_input`
    clone), then wraps them in `phydrax.nn.Separable` to form a low-rank separable
    approximation. With latent size $L$ and output size $m$, the resulting model
    has the form

    $$
    u_o(x)=\sum_{\ell=1}^{L}\prod_{i=1}^{d} g_{i,\ell,o}(x_i),
    $$

    where each coordinate model $g_i$ maps a scalar $x_i$ to $L\cdot m$ features
    (reshaped to $(L,m)$).
    """

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    model: _AbstractStructuredInputModel

    def __init__(
        self,
        *,
        # Separable parameters
        in_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
        latent_size: int = 32,
        output_activation: Callable | None = None,
        keep_outputs_complex: bool = False,
        split_input: int | None = None,
        # MLP parameters
        width_size: int | None = 20,
        depth: int | None = 6,
        hidden_sizes: Sequence[int] | None = None,
        activation: Callable = jax.nn.tanh,
        final_activation: Callable | None = None,
        skip_connection: bool = False,
        rwf: bool | tuple[float, float] = False,
        use_bias: bool = True,
        use_final_bias: bool = True,
        initializer: str = "glorot_normal",
        key: Key[Array, ""] = DOC_KEY0,
    ):
        r"""Create a separable MLP.

        `SeparableMLP` forwards MLP hyperparameters to each internal scalar
        coordinate model. The coordinate models output `latent_size * out_size`
        features so the wrapper can reshape them to $(L,m)$ and contract.
        """
        in_dim = _get_size(in_size)
        clones = (
            int(split_input) if (split_input is not None and int(split_input) > 1) else 1
        )
        if in_dim == 1 and clones == 1:
            raise ValueError(
                "SeparableMLP requires in_size >= 2, or split_input>1 to replicate "
                "a scalar input across multiple coordinate models."
            )

        out_dim = latent_size * _get_size(out_size)
        n_models = in_dim * clones
        keys = jr.split(key, n_models)
        models = tuple(
            MLP(
                in_size="scalar",
                out_size=out_dim,
                width_size=width_size,
                depth=depth,
                hidden_sizes=hidden_sizes,
                activation=activation,
                final_activation=final_activation,
                skip_connection=skip_connection,
                rwf=rwf,
                use_bias=use_bias,
                use_final_bias=use_final_bias,
                initializer=initializer,
                key=subkey,
            )
            for subkey in keys
        )

        self.model = Separable(
            in_size=in_size,
            out_size=out_size,
            latent_size=latent_size,
            models=models,
            output_activation=output_activation,
            keep_outputs_complex=keep_outputs_complex,
            split_input=split_input,
        )
        self.in_size = self.model.in_size
        self.out_size = self.model.out_size

    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        r"""Evaluate the separable MLP.

        Accepts either a vector input `(d,)` or a separable tuple `(x_1,...,x_d)`
        of 1D coordinate arrays (see `phydrax.nn.Separable`).
        """
        return self.model(x, key=key)
