#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence
from typing import ClassVar, Literal

import jax
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..._utils import _get_size
from ..core._base import _AbstractStructuredInputModel, DomainInputMode
from ..core._keys import EvalKey
from ..wrappers._separable_wrappers import Separable
from ._modified_mlp import ModifiedMLP


class SeparableModifiedMLP(_AbstractStructuredInputModel):
    r"""Low-rank separable model with coordinate-wise modified MLPs.

    One scalar-input `ModifiedMLP` is created per coordinate and optional
    `split_input` clone. Each coordinate model emits `latent_size * out_size`
    features, which are contracted by `Separable`.
    """

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    model: _AbstractStructuredInputModel
    _domain_input_mode: ClassVar[DomainInputMode] = "flat"
    _supports_blockwise_input: ClassVar[bool] = True

    def __init__(
        self,
        *,
        in_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
        latent_size: int = 32,
        output_activation: Callable | None = None,
        keep_outputs_complex: bool = False,
        split_input: int | None = None,
        width_size: int = 128,
        depth: int = 6,
        dropout: float | Sequence[float] = 0.0,
        dropout_mode: Literal["elementwise", "feature"] = "feature",
        activation: Callable = jax.nn.tanh,
        final_activation: Callable | None = None,
        rwf: bool | tuple[float, float] = False,
        use_bias: bool = True,
        bias_init_lim: float = 1.0,
        use_final_bias: bool = True,
        initializer: str = "glorot_normal",
        scan: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        """Construct a separable modified MLP."""
        in_dim = _get_size(in_size)
        clones = (
            int(split_input) if split_input is not None and int(split_input) > 1 else 1
        )
        if in_dim == 1 and clones == 1:
            raise ValueError(
                "SeparableModifiedMLP requires in_size >= 2, or split_input > 1."
            )

        out_dim = int(latent_size) * _get_size(out_size)
        keys = jr.split(key, in_dim * clones)
        models = tuple(
            ModifiedMLP(
                in_size="scalar",
                out_size=out_dim,
                width_size=width_size,
                depth=depth,
                dropout=dropout,
                dropout_mode=dropout_mode,
                activation=activation,
                final_activation=final_activation,
                rwf=rwf,
                use_bias=use_bias,
                bias_init_lim=bias_init_lim,
                use_final_bias=use_final_bias,
                initializer=initializer,
                scan=scan,
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
            scan=scan,
        )
        self.in_size = self.model.in_size
        self.out_size = self.model.out_size

    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Evaluate pointwise or coordinate-separable input."""
        return self.model(x, key=key)
