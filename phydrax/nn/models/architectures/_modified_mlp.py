#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable
from typing import Literal

import jax
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..._utils import _canonical_size, _identity, SizeLike
from ..core._base import _AbstractBaseModel
from ..core._keys import EvalKey
from ..core._scan_utils import pack_scan_modules, scan_apply, stack_scan_dynamics
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
        self.in_size = in_size_c
        self.out_size = out_size_c
        self.final_activation = (
            _identity if final_activation is None else final_activation
        )
        self.scan = bool(scan)
        self._scan_enabled = False
        self._scan_static = None

        if self.scan and hidden_depth > 1:
            _, static, enabled = pack_scan_modules(self.layers[1:-1])
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
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        """Evaluate the modified MLP at `x`."""
        encoder_u = self.encoder_u(x, key=key)
        encoder_v = self.encoder_v(x, key=key)
        hidden = self._mix(self.layers[0](x, key=key), encoder_u, encoder_v)

        scanned = False
        if self._scan_enabled and self._scan_static is not None:
            dynamic = stack_scan_dynamics(self.layers[1:-1])
            if dynamic is not None:
                hidden = scan_apply(
                    dynamic,
                    self._scan_static,
                    hidden,
                    lambda carry, layer: self._mix(
                        layer(carry, key=key), encoder_u, encoder_v
                    ),
                )
                scanned = True
        if not scanned:
            for layer in self.layers[1:-1]:
                hidden = self._mix(
                    layer(hidden, key=key), encoder_u, encoder_v
                )

        output = self.layers[-1](hidden, key=key)
        return self.final_activation(output)
