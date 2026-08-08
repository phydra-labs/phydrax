#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._model import KFACAffineBlock, KFACLayoutProvider
from .._base import _AbstractBaseModel
from .._keys import EvalKey, fold_in_eval_key
from .._scan import (
    pack_scan_modules,
    scan_apply_with_data,
    stack_scan_dynamics,
)
from .._utils import _canonical_size, _get_value_shape, _identity, SizeLike
from ..layers._dropout import _dropout_probabilities, Dropout
from ..layers._linear import Linear


class MLP(_AbstractBaseModel, KFACLayoutProvider):
    r"""Multi-Layer Perceptron (MLP).

    For input $x\in\mathbb{R}^{d_\text{in}}$ this model applies a sequence of
    affine maps and nonlinearities. Writing $h^{(0)}=x$, a depth-$L$ network is

    $$
    h^{(k)}=\sigma_k\!\left(W_k h^{(k-1)}+b_k\right),\qquad k=1,\dots,L,
    $$

    where hidden layers use `activation`, the final `Linear` layer uses the
    identity nonlinearity, and the output activation $\phi$ (`final_activation`)
    is applied *outside* the last layer:

    $$
    y=\phi\!\left(h^{(L)}\right).
    $$

    If `skip_connection=True` then a residual term is added before $\phi$:

    $$
    h^{(L)}\leftarrow h^{(L)} + P x,
    $$

    where $P$ is the identity when $d_\text{in}=d_\text{out}$ and otherwise a
    learned linear projection.
    """

    layers: tuple[Linear, ...]
    dropouts: tuple[Dropout, ...]
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    final_activation: Callable
    skip_connection: bool
    scan: bool
    _scan_enabled: bool
    _scan_static: object | None
    _residual_proj: Linear | None

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: SizeLike,
        width_size: int | None = None,
        depth: int | None = None,
        hidden_sizes: Sequence[int] | None = None,
        dropout: float | Sequence[float] = 0.0,
        dropout_mode: Literal["elementwise", "feature"] = "feature",
        activation: Callable = jax.nn.tanh,
        final_activation: Callable | None = None,
        skip_connection: bool = False,
        rwf: bool | tuple[float, float] = False,
        use_bias: bool = True,
        use_final_bias: bool = True,
        initializer: str = "glorot_normal",
        scan: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        r"""Construct an MLP.

        You may specify the hidden layout either with (`width_size`, `depth`) or with
        an explicit `hidden_sizes` sequence.

        **Arguments:**

        - `in_size`: Input value size: `"scalar"`, `d` (vector), or `(..., ...)` (tensor).
        - `out_size`: Output value size: `"scalar"`, `m` (vector), or `(..., ...)` (tensor).
        - `width_size`: Uniform hidden width (mutually exclusive with `hidden_sizes`).
        - `depth`: Number of hidden layers (mutually exclusive with `hidden_sizes`).
        - `hidden_sizes`: Explicit hidden layer sizes.
        - `dropout`: Drop probability shared by hidden layers, or one value per hidden layer.
        - `dropout_mode`: Elementwise masks or feature masks broadcast over leading axes.
        - `activation`: Hidden-layer activation (callable).
        - `final_activation`: Output activation (default: identity).
        - `skip_connection`: If `True`, adds a residual connection to the pre-activation output.
        - `rwf`: Random Weight Factorization for `Linear` layers; if `(\mu,\sigma)`,
          initializes $s\sim\mathcal{N}(\mu,\sigma^2)$.
        - `use_bias`: Whether to use biases in hidden `Linear` layers.
        - `use_final_bias`: Whether to use a bias in the final `Linear` layer.
        - `initializer`: Weight initializer name for `Linear` layers.
        - `scan`: If `True`, uses `jax.lax.scan` over repeated hidden layers when
          their topology is compatible. If not compatible, falls back to the
          standard loop path.
        - `key`: PRNG key.
        """
        if (width_size is None) ^ (depth is None):
            raise ValueError("width_size and depth must be provided together.")

        use_width_depth = width_size is not None and depth is not None
        use_hidden_sizes = hidden_sizes is not None
        if use_width_depth == use_hidden_sizes:
            raise ValueError(
                "Must provide either `width_size` and `depth` together, or `hidden_sizes` only."
            )

        if use_width_depth:
            if width_size is None or depth is None:
                raise ValueError("width_size and depth must be provided together.")
            hidden_sizes_list = [int(width_size)] * int(depth)
        else:
            hidden_sizes_list = list(hidden_sizes or ())

        in_size_c = _canonical_size(in_size)
        out_size_c = _canonical_size(out_size)
        in_shape = _get_value_shape(in_size_c)
        out_shape = _get_value_shape(out_size_c)

        final_act_fn = _identity if final_activation is None else final_activation
        need_proj = bool(skip_connection)

        num_layers = 1 if not hidden_sizes_list else len(hidden_sizes_list) + 1
        key_count = num_layers + (1 if need_proj else 0)
        keys = jr.split(key, key_count)
        layer_keys = keys[:num_layers]
        proj_key = keys[-1] if need_proj else None

        rwf_val = rwf
        layers: list[Linear] = []

        if hidden_sizes_list:
            sizes = [int(s) for s in hidden_sizes_list]
            layers.append(
                Linear(
                    in_size=in_size_c,
                    out_size=sizes[0],
                    activation=activation,
                    initializer=initializer,
                    rwf=rwf_val,
                    use_bias=use_bias,
                    key=layer_keys[0],
                )
            )
            for idx, (prev, curr) in enumerate(
                zip(sizes[:-1], sizes[1:], strict=True), start=1
            ):
                layers.append(
                    Linear(
                        in_size=int(prev),
                        out_size=int(curr),
                        activation=activation,
                        initializer=initializer,
                        rwf=rwf_val,
                        use_bias=use_bias,
                        key=layer_keys[idx],
                    )
                )
            layers.append(
                Linear(
                    in_size=int(sizes[-1]),
                    out_size=out_size_c,
                    activation=None,
                    initializer=initializer,
                    rwf=rwf_val,
                    use_bias=use_final_bias,
                    key=layer_keys[-1],
                )
            )
        else:
            layers.append(
                Linear(
                    in_size=in_size_c,
                    out_size=out_size_c,
                    activation=None,
                    initializer=initializer,
                    rwf=rwf_val,
                    use_bias=use_final_bias,
                    key=layer_keys[0],
                )
            )

        self.layers = tuple(layers)
        hidden_output_sizes = tuple(int(size) for size in hidden_sizes_list)
        dropout_probabilities = _dropout_probabilities(dropout, len(hidden_output_sizes))
        self.dropouts = tuple(
            Dropout(size, p=probability, mode=dropout_mode)
            for size, probability in zip(
                hidden_output_sizes, dropout_probabilities, strict=True
            )
        )
        self.in_size = in_size_c
        self.out_size = out_size_c
        self.final_activation = final_act_fn
        self.skip_connection = bool(skip_connection)
        self.scan = bool(scan)
        self._scan_enabled = False
        self._scan_static = None

        if self.scan and len(self.layers) > 2:
            repeated_blocks = tuple(
                zip(self.layers[1:-1], self.dropouts[1:], strict=True)
            )
            _, static, enabled = pack_scan_modules(repeated_blocks)
            self._scan_enabled = enabled
            if enabled:
                self._scan_static = static

        if need_proj and proj_key is not None:
            self._residual_proj = Linear(
                in_size=in_size_c,
                out_size=out_size_c,
                activation=None,
                initializer=initializer,
                rwf=rwf_val,
                use_bias=False,
                key=proj_key,
            )
        else:
            self._residual_proj = None

    def kfac_affine_blocks(self) -> tuple[KFACAffineBlock, ...]:
        blocks = tuple(
            KFACAffineBlock(
                name=f"layers/{index}",
                weight=layer.weight,
                bias=layer.bias,
                random_weight_factorization=layer.random_weight_factorization,
                enforce_positive_weights=layer.enforce_positive_weights,
            )
            for index, layer in enumerate(self.layers)
        )
        if self._residual_proj is None:
            return blocks
        residual = self._residual_proj
        return blocks + (
            KFACAffineBlock(
                name="residual_projection",
                weight=residual.weight,
                bias=residual.bias,
                random_weight_factorization=residual.random_weight_factorization,
                enforce_positive_weights=residual.enforce_positive_weights,
            ),
        )

    def kfac_validation_errors(self) -> tuple[str, ...]:
        return tuple(
            f"active dropout at site {site}"
            for site, dropout in enumerate(self.dropouts)
            if dropout.p > 0.0 and not dropout.inference
        )

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        r"""Evaluate the MLP at `x`.

        Active dropout requires an explicit key. Each hidden layer receives a
        stable, distinct subkey derived from that root key.
        """
        x0 = x
        y = None
        hidden_count = len(self.dropouts)
        if hidden_count and self._scan_enabled and self._scan_static is not None:
            repeated_blocks = tuple(
                zip(self.layers[1:-1], self.dropouts[1:], strict=True)
            )
            dynamic = stack_scan_dynamics(repeated_blocks)
            if dynamic is not None:
                site_key = fold_in_eval_key(key, 0)
                x = self.layers[0](x, key=site_key)
                x = self.dropouts[0](x, key=site_key)
                sites = jnp.arange(1, hidden_count, dtype=jnp.uint32)
                x = scan_apply_with_data(
                    dynamic,
                    self._scan_static,
                    x,
                    sites,
                    lambda carry, block, site: block[1](
                        block[0](carry, key=fold_in_eval_key(key, site)),
                        key=fold_in_eval_key(key, site),
                    ),
                )
                y = self.layers[-1](x, key=fold_in_eval_key(key, hidden_count))
        if y is None:
            for site, (layer, dropout_layer) in enumerate(
                zip(self.layers[:-1], self.dropouts, strict=True)
            ):
                site_key = fold_in_eval_key(key, site)
                x = dropout_layer(layer(x, key=site_key), key=site_key)
            y = self.layers[-1](x, key=fold_in_eval_key(key, hidden_count))

        if self.skip_connection:
            if self._residual_proj is None:
                res = x0
            else:
                res = self._residual_proj(x0, key=fold_in_eval_key(key, hidden_count + 1))
            y = y + res
        return self.final_activation(y)
