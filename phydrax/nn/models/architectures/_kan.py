#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

# orthax provides Clenshaw evaluators for orthogonal polynomial families
from orthax import (
    chebyshev as _o_cheb,
    hermite as _o_herm,
    hermite_e as _o_herme,
    laguerre as _o_lag,
    legendre as _o_leg,
)

from ...._doc import DOC_KEY0
from ...._strict import StrictModule
from ..._utils import _canonical_size, _get_size, _get_value_shape, _identity, SizeLike
from ..core._base import _AbstractBaseModel
from ..layers._linear import Linear


def _orthax_val(module, name: str, t, coeffs):
    fn = getattr(module, name, None)
    if fn is None:
        raise ValueError(f"orthax module missing {name} evaluator.")
    return fn(t, coeffs)


def _poly_eval(poly: str, coeffs, t, params: dict | None = None):
    """Evaluate an orthogonal polynomial series at scalar `t`.

    Uses orthax's family-specific `*val` evaluators:
    - chebyshev -> chebval
    - legendre  -> legval
    - hermite   -> hermval
    - hermite_e -> hermeval
    - laguerre  -> lagval
    """
    fam = poly.lower()
    if fam in ("chebyshev", "cheb", "t"):
        return _orthax_val(_o_cheb, "chebval", t, coeffs)
    if fam in ("legendre", "p"):
        return _orthax_val(_o_leg, "legval", t, coeffs)
    if fam == "hermite":
        return _orthax_val(_o_herm, "hermval", t, coeffs)
    if fam in ("hermite_e", "herme"):
        return _orthax_val(_o_herme, "hermeval", t, coeffs)
    if fam == "laguerre":
        return _orthax_val(_o_lag, "lagval", t, coeffs)
    raise ValueError(f"Unknown or unsupported orthogonal polynomial family: {poly}")


class KANLayer(StrictModule):
    r"""A single KAN layer with orthogonal polynomial edge functions.

    For input $x\in\mathbb{R}^{d_\text{in}}$ and output $y\in\mathbb{R}^{d_\text{out}}$
    this layer computes, componentwise,

    $$
    y_o(x)=\sum_{i=1}^{d_\text{in}} f_{o,i}(s_{o,i}\,x_i) + b_o,
    $$

    where each edge function $f_{o,i}$ is represented by a degree-$K$ orthogonal
    polynomial expansion

    $$
    f_{o,i}(t)=\sum_{k=0}^{K} c_{o,i,k}\,P_k(\psi(t)).
    $$

    Here $P_k$ is from the chosen family (`poly`, e.g. Chebyshev/Legendre) and
    $\psi(t)$ maps inputs to the canonical interval: $\psi(t)=\tanh(t)$ if
    `use_tanh=True`, otherwise $\psi(t)=\operatorname{clip}(t,-1,1)$.
    """

    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    degree: int
    use_tanh: bool
    scale_mode: Literal["edge", "input", "none"]
    init: Literal["default", "identity"]
    autoscale: bool
    poly: str

    coeffs: Array  # (out, in, degree+1)
    scales: Array  # (out, in) if edge, (in,) if input, () if none
    bias: Array  # (out,)
    # Optional learned per-input affine when scale_mode="none"
    ascale: Array | None
    abias: Array | None
    poly_params: dict | None

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: SizeLike,
        degree: int = 5,
        use_tanh: bool = False,
        scale_mode: Literal["edge", "input", "none"] = "edge",
        init: Literal["default", "identity"] = "default",
        autoscale: bool = False,
        poly: str = "chebyshev",
        poly_params: dict | None = None,
        use_bias: bool = True,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        in_size_c = _canonical_size(in_size)
        out_size_c = _canonical_size(out_size)
        in_shape = _get_value_shape(in_size_c)
        out_shape = _get_value_shape(out_size_c)
        if len(in_shape) > 1:
            raise ValueError(
                "KANLayer expects scalar or 1D inputs; got "
                f"in_size={in_size_c!r} (shape={in_shape!r})."
            )
        if len(out_shape) > 1:
            raise ValueError(
                "KANLayer expects scalar or 1D outputs; got "
                f"out_size={out_size_c!r} (shape={out_shape!r})."
            )

        in_ = _get_size(in_size_c)
        out_ = _get_size(out_size_c)
        ckey, skey, bkey, akey = jr.split(key, 4)
        coeffs = jnp.zeros((out_, in_, degree + 1))
        if degree >= 1:
            coeffs = coeffs.at[..., 1].set(0.05 * jr.normal(ckey, (out_, in_)))
        if scale_mode == "edge":
            scales = 1.0 + 0.01 * jr.normal(skey, (out_, in_))
        elif scale_mode == "input":
            scales = 1.0 + 0.01 * jr.normal(skey, (in_,))
        else:
            scales = jnp.array(1.0)
        bias = jnp.zeros((out_,)) if use_bias else jnp.zeros((out_,))
        self.in_size = in_size_c
        self.out_size = out_size_c

        self.degree = int(degree)
        self.use_tanh = bool(use_tanh)
        self.scale_mode = scale_mode
        self.init = init
        self.autoscale = bool(autoscale)
        self.coeffs = coeffs
        self.scales = scales
        self.bias = bias
        self.poly = poly
        self.poly_params = poly_params or {}
        if init == "identity" and degree >= 1:
            diag = jnp.eye(out_, in_)
            self.coeffs = self.coeffs.at[..., 1].set(diag)
        if self.scale_mode == "none" and self.autoscale:
            self.ascale = jnp.ones((in_,)) + 0.01 * jr.normal(akey, (in_,))
            self.abias = jnp.zeros((in_,))
        else:
            self.ascale = None
            self.abias = None

    def __call__(self, x: Array) -> Array:
        in_ = _get_size(self.in_size)
        x_arr = jnp.asarray(x)
        if self.in_size == "scalar":
            if x_arr.shape == ():
                x_vec = x_arr.reshape((1,))
            elif x_arr.shape == (1,):
                x_vec = x_arr
            else:
                raise ValueError(
                    f"KANLayer expected scalar input shape () or (1,), got {x_arr.shape}."
                )
        else:
            if x_arr.ndim != 1 or int(x_arr.shape[0]) != in_:
                raise ValueError(
                    f"KANLayer expected input shape ({in_},); got {x_arr.shape}."
                )
            x_vec = x_arr

        if self.scale_mode == "edge":
            z = self.scales * x_vec[None, :]  # (out, in)
            z = jnp.tanh(z) if self.use_tanh else jnp.clip(z, -1.0, 1.0)
        else:
            if self.scale_mode == "input":
                z_row = self.scales * x_vec
            else:  # none
                if (
                    self.autoscale
                    and (self.ascale is not None)
                    and (self.abias is not None)
                ):
                    z_row = self.ascale * x_vec + self.abias
                else:
                    z_row = x_vec
            z_row = jnp.tanh(z_row) if self.use_tanh else jnp.clip(z_row, -1.0, 1.0)

        def eval_out(c_row, zr):
            f_i = jax.vmap(lambda c, t: _poly_eval(self.poly, c, t, self.poly_params))(
                c_row, zr
            )
            return jnp.sum(f_i)

        if self.scale_mode == "edge":
            y = (
                jax.vmap(lambda c_row, zr: eval_out(c_row, zr))(self.coeffs, z)
                + self.bias
            )
        else:
            y = jax.vmap(lambda c_row: eval_out(c_row, z_row))(self.coeffs) + self.bias
        if self.out_size == "scalar":
            return y.reshape(())
        return y


class KAN(_AbstractBaseModel):
    r"""Kolmogorov-Arnold Network (KAN) with orthogonal polynomial edge functions.

    Stacks `KANLayer` blocks; each edge uses a degree-`degree` orthogonal polynomial
    expansion (Chebyshev by default).

    Enable `use_tanh=True` to map pre-activations into $[-1,1]$ before evaluating the basis.
    """

    layers: tuple[KANLayer, ...]
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
        degree: int | Sequence[int] = 5,
        use_tanh: bool = False,
        scale_mode: Literal["edge", "input", "none"] = "edge",
        init: Literal["default", "identity"] = "default",
        autoscale: bool = False,
        final_activation: Callable | None = None,
        skip_connection: bool = True,
        use_bias: bool = True,
        poly: str = "chebyshev",
        poly_params: dict | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        in_size_c = _canonical_size(in_size)
        out_size_c = _canonical_size(out_size)
        in_shape = _get_value_shape(in_size_c)
        out_shape = _get_value_shape(out_size_c)
        if len(in_shape) > 1:
            raise ValueError(
                f"KAN expects scalar or 1D inputs; got in_size={in_size_c!r}."
            )
        if len(out_shape) > 1:
            raise ValueError(
                f"KAN expects scalar or 1D outputs; got out_size={out_size_c!r}."
            )
        width_and_depth_defined = width_size is not None and depth is not None
        hidden_sizes_defined = hidden_sizes is not None
        if not (width_and_depth_defined ^ hidden_sizes_defined):
            raise ValueError(
                "Must provide either `width_size` and `depth` together, or `hidden_sizes` only."
            )
        if width_and_depth_defined:
            if width_size is None or depth is None:
                raise ValueError("width_size and depth must be provided together.")
            hidden_sizes_list = [int(width_size)] * int(depth)
        else:
            if hidden_sizes is None:
                raise ValueError(
                    "hidden_sizes must be provided when width_size/depth are absent."
                )
            hidden_sizes_list = list(hidden_sizes)

        has_hidden = len(hidden_sizes_list) > 0
        num_layers = (len(hidden_sizes_list) + 1) if has_hidden else 1
        keys = jr.split(key, num_layers)
        if isinstance(degree, int):
            degrees = [degree] * num_layers
        else:
            degrees = list(degree)
            expected = num_layers
            if len(degrees) != expected:
                raise ValueError(
                    f"degree must have {expected} entries for this architecture; got {len(degrees)}."
                )
        layers: list[KANLayer] = []

        if not has_hidden:
            # Single layer in->out
            layers.append(
                KANLayer(
                    in_size=in_size_c,
                    out_size=out_size_c,
                    degree=degrees[0],
                    use_tanh=use_tanh,
                    scale_mode=scale_mode,
                    init=init,
                    autoscale=autoscale,
                    poly=poly,
                    poly_params=poly_params,
                    use_bias=use_bias,
                    key=keys[0],
                )
            )
        else:
            # First: in -> width
            layers.append(
                KANLayer(
                    in_size=in_size_c,
                    out_size=int(hidden_sizes_list[0]),
                    degree=degrees[0],
                    use_tanh=use_tanh,
                    scale_mode=scale_mode,
                    init=init,
                    autoscale=autoscale,
                    poly=poly,
                    poly_params=poly_params,
                    use_bias=use_bias,
                    key=keys[0],
                )
            )
            # Hidden: width -> width (depth-1 times)
            for i in range(1, len(hidden_sizes_list)):
                layers.append(
                    KANLayer(
                        in_size=int(hidden_sizes_list[i - 1]),
                        out_size=int(hidden_sizes_list[i]),
                        degree=degrees[i],
                        use_tanh=use_tanh,
                        scale_mode=scale_mode,
                        init=init,
                        autoscale=autoscale,
                        poly=poly,
                        poly_params=poly_params,
                        use_bias=use_bias,
                        key=keys[i],
                    )
                )
            # Final: width -> out
            layers.append(
                KANLayer(
                    in_size=int(hidden_sizes_list[-1]),
                    out_size=out_size_c,
                    degree=degrees[-1],
                    use_tanh=use_tanh,
                    scale_mode=scale_mode,
                    init=init,
                    autoscale=autoscale,
                    poly=poly,
                    poly_params=poly_params,
                    use_bias=use_bias,
                    key=keys[-1],
                )
            )

        self.layers = tuple(layers)
        self.in_size = in_size_c
        self.out_size = out_size_c
        self.final_activation = (
            _identity if final_activation is None else final_activation
        )
        self.skip_connection = bool(skip_connection)
        need_proj = self.skip_connection and in_shape != out_shape
        self._residual_proj = (
            Linear(
                in_size=in_size_c,
                out_size=out_size_c,
                activation=None,
                initializer="glorot_normal",
                rwf=False,
                use_bias=False,
                key=DOC_KEY0,
            )
            if need_proj
            else None
        )

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        r"""Evaluate the KAN at `x`.

        Applies the stacked `KANLayer`s and an optional residual connection, then
        applies `final_activation` to the result.
        """
        y = x
        for layer in self.layers:
            y = layer(y)
        if self.skip_connection:
            if self._residual_proj is not None:
                res = self._residual_proj(x)
            else:
                res = x
            y = y + res
        return self.final_activation(y)

    def regularization_loss(
        self,
        *,
        alpha: float = 1e-4,
        order_power: float = 2.0,
        penalize_from: int = 2,
    ) -> Array:
        """Weighted L2 penalty on higher-order Chebyshev coefficients.

        - penalize_from=2 starts at T2, ignoring constant (T0) and linear (T1).
        - order_power controls growth of the weight with k.
        """
        reg = jnp.array(0.0)
        for layer in self.layers:
            c = layer.coeffs  # (out, in, K+1)
            Kp1 = c.shape[-1]
            if Kp1 <= penalize_from:
                continue
            k = jnp.arange(Kp1)
            w = (k**order_power).astype(c.dtype)
            w = jnp.where(k >= penalize_from, w, 0.0)
            reg = reg + jnp.sum((c * w[None, None, :]) ** 2)
        alpha_arr = jnp.asarray(alpha, dtype=reg.dtype)
        return alpha_arr * reg


def chebkan_degree_schedule(
    num_layers: int,
    *,
    start: int,
    end: int,
    strategy: Literal["linear", "geometric", "constant"] = "linear",
) -> list[int]:
    if num_layers <= 0:
        return []
    if strategy == "constant":
        return [int(start)] * num_layers
    if num_layers == 1:
        return [int(end)]
    if strategy == "linear":
        vals = jnp.linspace(start, end, num_layers)
        return [int(jnp.round(v)) for v in vals]
    if strategy == "geometric":
        vals = jnp.geomspace(max(1, start), max(1, end), num_layers)
        return [int(jnp.round(v)) for v in vals]
    raise ValueError(f"Unknown strategy: {strategy}")
