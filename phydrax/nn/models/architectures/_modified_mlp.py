#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..._utils import _canonical_size, _identity, SizeLike
from ..core._base import _AbstractBaseModel
from ..core._keys import EvalKey, fold_in_eval_key
from ..core._scan_utils import (
    pack_scan_modules,
    scan_apply_with_data,
    stack_scan_dynamics,
)
from ..layers._dropout import _dropout_probabilities, Dropout
from ..layers._linear import Linear


class ModifiedMLP(_AbstractBaseModel):
    r"""Modified MLP with two persistent input encoders.

    Two encodings of the original input are reused by every hidden layer:

    $$
    U=\sigma(W_Ux+b_U),\qquad V=\sigma(W_Vx+b_V),
    $$

    $$
    z_k=\sigma(W_kh_{k-1}+b_k),\qquad
    h_k=(1-z_k)\odot U+z_k\odot V.
    $$

    All hidden layers have one common width, as required for the persistent
    encodings to be shared without introducing noncanonical per-layer encoders.
    """

    encoder_u: Linear
    encoder_v: Linear
    layers: tuple[Linear, ...]
    dropouts: tuple[Dropout, ...]
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    final_activation: Callable
    scan: bool
    _scan_enabled: bool
    _scan_static: object | None

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: SizeLike,
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
        r"""Construct a modified MLP.

        **Arguments:**

        - `in_size`: Input value size.
        - `out_size`: Output value size.
        - `width_size`: Shared hidden and encoder width.
        - `depth`: Number of modified hidden layers; must be positive.
        - `dropout`: Drop probability shared by hidden states, or one per hidden state.
        - `dropout_mode`: Elementwise masks or feature masks broadcast over leading axes.
        - `activation`: Encoder and hidden-layer activation.
        - `final_activation`: Optional output activation.
        - `rwf`: Random Weight Factorization configuration for linear layers.
        - `use_bias`: Whether encoders and hidden layers use biases.
        - `bias_init_lim`: Uniform bias initialization bound.
        - `use_final_bias`: Whether the output layer uses a bias.
        - `initializer`: Linear-layer initializer.
        - `scan`: Use `jax.lax.scan` for compatible repeated hidden layers.
        - `key`: PRNG key.
        """
        width = int(width_size)
        hidden_depth = int(depth)
        if width <= 0:
            raise ValueError(f"width_size must be positive, got {width}.")
        if hidden_depth <= 0:
            raise ValueError(f"depth must be positive, got {hidden_depth}.")

        in_size_c = _canonical_size(in_size)
        out_size_c = _canonical_size(out_size)
        keys = jr.split(key, hidden_depth + 3)
        self.encoder_u = Linear(
            in_size=in_size_c,
            out_size=width,
            activation=activation,
            initializer=initializer,
            rwf=rwf,
            use_bias=use_bias,
            bias_init_lim=bias_init_lim,
            key=keys[0],
        )
        self.encoder_v = Linear(
            in_size=in_size_c,
            out_size=width,
            activation=activation,
            initializer=initializer,
            rwf=rwf,
            use_bias=use_bias,
            bias_init_lim=bias_init_lim,
            key=keys[1],
        )

        hidden: list[Linear] = [
            Linear(
                in_size=in_size_c,
                out_size=width,
                activation=activation,
                initializer=initializer,
                rwf=rwf,
                use_bias=use_bias,
                bias_init_lim=bias_init_lim,
                key=keys[2],
            )
        ]
        hidden.extend(
            Linear(
                in_size=width,
                out_size=width,
                activation=activation,
                initializer=initializer,
                rwf=rwf,
                use_bias=use_bias,
                bias_init_lim=bias_init_lim,
                key=keys[index + 2],
            )
            for index in range(1, hidden_depth)
        )
        output = Linear(
            in_size=width,
            out_size=out_size_c,
            activation=None,
            initializer=initializer,
            rwf=rwf,
            use_bias=use_final_bias,
            bias_init_lim=bias_init_lim,
            key=keys[-1],
        )
        self.layers = (*hidden, output)
        dropout_probabilities = _dropout_probabilities(dropout, hidden_depth)
        self.dropouts = tuple(
            Dropout(width, p=probability, mode=dropout_mode)
            for probability in dropout_probabilities
        )
        self.in_size = in_size_c
        self.out_size = out_size_c
        self.final_activation = (
            _identity if final_activation is None else final_activation
        )
        self.scan = bool(scan)
        self._scan_enabled = False
        self._scan_static = None

        if self.scan and hidden_depth > 1:
            repeated_blocks = tuple(
                zip(self.layers[1:-1], self.dropouts[1:], strict=True)
            )
            _, static, enabled = pack_scan_modules(repeated_blocks)
            self._scan_enabled = enabled
            if enabled:
                self._scan_static = static

    @staticmethod
    def _mix(gate: Array, encoder_u: Array, encoder_v: Array, /) -> Array:
        return (1.0 - gate) * encoder_u + gate * encoder_v

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Evaluate the modified MLP at `x` with one key per hidden state."""
        encoder_u = self.encoder_u(x, key=fold_in_eval_key(key, 0))
        encoder_v = self.encoder_v(x, key=fold_in_eval_key(key, 1))
        first_key = fold_in_eval_key(key, 2)
        hidden = self._mix(self.layers[0](x, key=first_key), encoder_u, encoder_v)
        hidden = self.dropouts[0](hidden, key=first_key)

        scanned = False
        if self._scan_enabled and self._scan_static is not None:
            repeated_blocks = tuple(
                zip(self.layers[1:-1], self.dropouts[1:], strict=True)
            )
            dynamic = stack_scan_dynamics(repeated_blocks)
            if dynamic is not None:
                sites = jnp.arange(3, len(self.dropouts) + 2, dtype=jnp.uint32)
                hidden = scan_apply_with_data(
                    dynamic,
                    self._scan_static,
                    hidden,
                    sites,
                    lambda carry, block, site: block[1](
                        self._mix(
                            block[0](carry, key=fold_in_eval_key(key, site)),
                            encoder_u,
                            encoder_v,
                        ),
                        key=fold_in_eval_key(key, site),
                    ),
                )
                scanned = True
        if not scanned:
            for site, (layer, dropout_layer) in enumerate(
                zip(self.layers[1:-1], self.dropouts[1:], strict=True),
                start=3,
            ):
                site_key = fold_in_eval_key(key, site)
                hidden = self._mix(layer(hidden, key=site_key), encoder_u, encoder_v)
                hidden = dropout_layer(hidden, key=site_key)

        output = self.layers[-1](
            hidden, key=fold_in_eval_key(key, len(self.dropouts) + 2)
        )
        return self.final_activation(output)
