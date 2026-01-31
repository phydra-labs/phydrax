#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..._utils import _canonical_size, _get_value_shape, _identity, SizeLike
from ..core._base import _AbstractBaseModel
from ..layers._linear import Linear


class MLP(_AbstractBaseModel):
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
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    final_activation: Callable
    skip_connection: bool
    _residual_proj: Linear | None

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: SizeLike,
        width_size: int | None = None,
        depth: int | None = None,
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
        r"""Construct an MLP.

        You may specify the hidden layout either with (`width_size`, `depth`) or with
        an explicit `hidden_sizes` sequence.

        **Arguments:**

        - `in_size`: Input value size: `"scalar"`, `d` (vector), or `(..., ...)` (tensor).
        - `out_size`: Output value size: `"scalar"`, `m` (vector), or `(..., ...)` (tensor).
        - `width_size`: Uniform hidden width (mutually exclusive with `hidden_sizes`).
        - `depth`: Number of hidden layers (mutually exclusive with `hidden_sizes`).
        - `hidden_sizes`: Explicit hidden layer sizes.
        - `activation`: Hidden-layer activation (callable).
        - `final_activation`: Output activation (default: identity).
        - `skip_connection`: If `True`, adds a residual connection to the pre-activation output.
        - `rwf`: Random Weight Factorization for `Linear` layers; if `(\mu,\sigma)`,
          initializes $s\sim\mathcal{N}(\mu,\sigma^2)$.
        - `use_bias`: Whether to use biases in hidden `Linear` layers.
        - `use_final_bias`: Whether to use a bias in the final `Linear` layer.
        - `initializer`: Weight initializer name for `Linear` layers.
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
        self.in_size = in_size_c
        self.out_size = out_size_c
        self.final_activation = final_act_fn
        self.skip_connection = bool(skip_connection)

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

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        r"""Evaluate the MLP at `x`.

        **Arguments:**

        - `x`: Input with trailing value shape implied by `in_size`. Leading axes are free.
        - `key`: PRNG key forwarded to layers (most layers are deterministic and
          ignore it; it is present for API consistency).

        **Returns:**

        - Output with trailing value shape implied by `out_size`. If `out_size == "scalar"`,
          returns a scalar per leading index (no trailing value axis).
        """
        # Standard MLP path: apply hidden layers then output layer
        x0 = x
        for layer in self.layers[:-1]:
            x = layer(x, key=key)
        # Final layer pre-activation (final layer uses identity activation)
        y = self.layers[-1](x, key=key)
        # Residual addition before final activation
        if self.skip_connection:
            if self._residual_proj is None:
                res = x0
            else:
                res = self._residual_proj(x0, key=key)
            y = y + res
        # Apply final activation if provided
        y = self.final_activation(y)
        return y
