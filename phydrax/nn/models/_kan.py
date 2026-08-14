#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from .._base import _AbstractBaseModel
from .._keys import EvalKey
from .._scan import pack_scan_modules, scan_apply, stack_scan_dynamics
from .._utils import _canonical_size, _get_size, _get_value_shape, _identity, SizeLike
from ..layers._linear import Linear
from ._kan_basis import AbstractEdgeBasis, OrthogonalPolynomialEdgeBasis


def _canonicalize_edge_inputs(inputs: Array, use_tanh: bool) -> Array:
    if use_tanh:
        return jnp.tanh(inputs)
    return jnp.where(
        inputs < -1.0,
        -1.0,
        jnp.where(inputs > 1.0, 1.0, inputs),
    )


class KANEdgeBlock(StrictModule):
    """Shape-homogeneous sparse collection of KAN edges sharing one basis."""

    output_indices: tuple[int, ...] = eqx.field(static=True)
    input_indices: tuple[int, ...] = eqx.field(static=True)
    edge_basis: AbstractEdgeBasis
    coeffs: Any

    def __init__(
        self,
        *,
        output_indices: Sequence[int],
        input_indices: Sequence[int],
        edge_basis: AbstractEdgeBasis,
        coeffs: Any,
    ):
        outputs = tuple(int(index) for index in output_indices)
        inputs = tuple(int(index) for index in input_indices)
        if not outputs or len(outputs) != len(inputs):
            raise ValueError(
                "KAN edge blocks require equally sized nonempty index sequences."
            )
        if any(index < 0 for index in (*outputs, *inputs)):
            raise ValueError("KAN edge-block indices must be nonnegative.")
        if not isinstance(edge_basis, AbstractEdgeBasis):
            raise TypeError("edge_basis must implement AbstractEdgeBasis.")
        for leaf in jax.tree.leaves(coeffs):
            if eqx.is_array(leaf) and (
                leaf.ndim < 2 or leaf.shape[:2] != (len(outputs), 1)
            ):
                raise ValueError(
                    "Every KAN edge-block parameter array must begin with "
                    "(edge_count, 1)."
                )
        self.output_indices = outputs
        self.input_indices = inputs
        self.edge_basis = edge_basis
        self.coeffs = coeffs

    @property
    def edge_count(self) -> int:
        return len(self.output_indices)

    @property
    def degree(self) -> int:
        return self.edge_basis.degree

    def evaluate(self, edge_inputs: Array, /) -> Array:
        selected = edge_inputs[
            jnp.asarray(self.output_indices, dtype=jnp.int32),
            jnp.asarray(self.input_indices, dtype=jnp.int32),
        ]
        return self.edge_basis.evaluate(self.coeffs, selected[:, None])[:, 0]

    def regularization(self) -> Array:
        return self.edge_basis.regularization(self.coeffs)


class KANLayer(StrictModule):
    """A single KAN layer whose scalar edge functions share one typed basis."""

    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    edge_basis: AbstractEdgeBasis | None
    use_tanh: bool
    scale_mode: Literal["edge", "input", "none"]
    init: Literal["default", "identity"]
    autoscale: bool

    coeffs: Any | None  # dense edge-parameter PyTree; None for sparse blocks
    scales: Array | None  # (out, in) if edge, (in,) if input
    bias: Array | None  # (out,)
    ascale: Array | None
    abias: Array | None
    edge_blocks: tuple[KANEdgeBlock, ...]

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: SizeLike,
        edge_basis: AbstractEdgeBasis | None = None,
        use_tanh: bool = False,
        scale_mode: Literal["edge", "input", "none"] = "edge",
        init: Literal["default", "identity"] = "default",
        autoscale: bool = False,
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
        if scale_mode not in ("edge", "input", "none"):
            raise ValueError(f"Unknown KAN scale_mode: {scale_mode!r}.")

        basis = OrthogonalPolynomialEdgeBasis() if edge_basis is None else edge_basis
        if not isinstance(basis, AbstractEdgeBasis):
            raise TypeError("edge_basis must implement AbstractEdgeBasis.")

        in_ = _get_size(in_size_c)
        out_ = _get_size(out_size_c)
        basis = basis.for_layer(in_, out_)
        ckey, skey, _unused_key, akey = jr.split(key, 4)
        coefficients = basis.initialize_coefficients(out_, in_, init, ckey)
        if scale_mode == "edge":
            scales = 1.0 + 0.01 * jr.normal(skey, (out_, in_))
        elif scale_mode == "input":
            scales = 1.0 + 0.01 * jr.normal(skey, (in_,))
        else:
            scales = None

        self.in_size = in_size_c
        self.out_size = out_size_c
        self.edge_basis = basis
        self.use_tanh = bool(use_tanh)
        self.scale_mode = scale_mode
        self.init = init
        self.autoscale = bool(autoscale)
        self.coeffs = coefficients
        self.scales = scales
        self.bias = jnp.zeros((out_,)) if use_bias else None
        if scale_mode == "none" and autoscale:
            self.ascale = jnp.ones((in_,)) + 0.01 * jr.normal(akey, (in_,))
            self.abias = jnp.zeros((in_,))
        else:
            self.ascale = None
            self.abias = None

        self.edge_blocks = ()

    @property
    def degree(self) -> int:
        if self.edge_basis is not None:
            return self.edge_basis.degree
        return max(block.degree for block in self.edge_blocks)

    def _input_vector(self, x: Array, /) -> Array:
        in_ = _get_size(self.in_size)
        x_arr = jnp.asarray(x)
        if self.in_size == "scalar":
            if x_arr.shape == ():
                return x_arr.reshape((1,))
            if x_arr.shape == (1,):
                return x_arr
            raise ValueError(
                f"KANLayer expected scalar input shape () or (1,), got {x_arr.shape}."
            )
        if x_arr.ndim != 1 or int(x_arr.shape[0]) != in_:
            raise ValueError(
                f"KANLayer expected input shape ({in_},); got {x_arr.shape}."
            )
        return x_arr

    def _normalized_edge_inputs(self, x: Array, /) -> Array:
        x_vec = self._input_vector(x)
        if self.scale_mode == "edge":
            if self.scales is None:
                raise RuntimeError("KAN edge scales are missing.")
            edge_inputs = self.scales * x_vec[None, :]
        else:
            if self.scale_mode == "input":
                if self.scales is None:
                    raise RuntimeError("KAN input scales are missing.")
                input_values = self.scales * x_vec
            elif self.autoscale and self.ascale is not None and self.abias is not None:
                input_values = self.ascale * x_vec + self.abias
            else:
                input_values = x_vec
            edge_inputs = jnp.broadcast_to(
                input_values,
                (_get_size(self.out_size), _get_size(self.in_size)),
            )
        return _canonicalize_edge_inputs(edge_inputs, self.use_tanh)

    def __call__(self, x: Array) -> Array:
        edge_inputs = self._normalized_edge_inputs(x)
        if self.edge_blocks:
            output = None
            output_indices = None
            for block in self.edge_blocks:
                block_values = block.evaluate(edge_inputs)
                if output is None:
                    output = jnp.zeros(
                        (_get_size(self.out_size),),
                        dtype=block_values.dtype,
                    )
                output_indices = jnp.asarray(
                    block.output_indices,
                    dtype=jnp.int32,
                )
                output = output.at[output_indices].add(block_values)
            if output is None or output_indices is None:
                raise RuntimeError("KAN edge-block execution produced no output.")
        else:
            if self.edge_basis is None or self.coeffs is None:
                raise RuntimeError("Dense KAN layer edge parameters are missing.")
            edge_values = self.edge_basis.evaluate(self.coeffs, edge_inputs)
            output = jnp.sum(edge_values, axis=-1)
        if self.bias is not None:
            output = output + self.bias
        if self.out_size == "scalar":
            return output.reshape(())
        return output

    def regularization(self) -> Array:
        if self.edge_blocks:
            penalty = jnp.array(0.0)
            for block in self.edge_blocks:
                penalty = penalty + block.regularization()
            return penalty
        if self.edge_basis is None or self.coeffs is None:
            raise RuntimeError("Dense KAN layer edge parameters are missing.")
        return self.edge_basis.regularization(self.coeffs)


class KAN(_AbstractBaseModel):
    """Kolmogorov-Arnold Network with typed scalar edge-function bases.

    ``edge_basis`` may be one basis shared by every layer or one basis per layer.
    The default remains a degree-five Chebyshev basis. B-spline and rational
    B-spline edge bases supply compact support without changing the model's
    residual, scaling, or scan semantics.
    """

    layers: tuple[KANLayer, ...]
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
        edge_basis: AbstractEdgeBasis | Sequence[AbstractEdgeBasis] | None = None,
        use_tanh: bool = False,
        scale_mode: Literal["edge", "input", "none"] = "edge",
        init: Literal["default", "identity"] = "default",
        autoscale: bool = False,
        final_activation: Callable | None = None,
        skip_connection: bool = True,
        use_bias: bool = True,
        scan: bool = False,
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
                "Must provide either `width_size` and `depth` together, or "
                "`hidden_sizes` only."
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
            hidden_sizes_list = [int(size) for size in hidden_sizes]

        num_layers = len(hidden_sizes_list) + 1
        if edge_basis is None:
            default_basis = OrthogonalPolynomialEdgeBasis()
            edge_bases = [default_basis] * num_layers
        elif isinstance(edge_basis, AbstractEdgeBasis):
            edge_bases = [edge_basis] * num_layers
        else:
            edge_bases = list(edge_basis)
            if len(edge_bases) != num_layers:
                raise ValueError(
                    f"edge_basis must have {num_layers} entries for this architecture; "
                    f"got {len(edge_bases)}."
                )
            if not all(
                isinstance(layer_basis, AbstractEdgeBasis) for layer_basis in edge_bases
            ):
                raise TypeError(
                    "Every edge_basis entry must implement AbstractEdgeBasis."
                )

        keys = jr.split(key, num_layers)
        sizes: list[SizeLike] = [in_size_c, *hidden_sizes_list, out_size_c]
        self.layers = tuple(
            KANLayer(
                in_size=sizes[index],
                out_size=sizes[index + 1],
                edge_basis=edge_bases[index],
                use_tanh=use_tanh,
                scale_mode=scale_mode,
                init=init,
                autoscale=autoscale,
                use_bias=use_bias,
                key=keys[index],
            )
            for index in range(num_layers)
        )
        self.in_size = in_size_c
        self.out_size = out_size_c
        self.final_activation = (
            _identity if final_activation is None else final_activation
        )
        self.skip_connection = bool(skip_connection)
        self.scan = bool(scan)
        self._scan_enabled = False
        self._scan_static = None

        if self.scan and len(self.layers) > 2:
            repeated_layers = self.layers[1:-1]
            _, static, enabled = pack_scan_modules(repeated_layers)
            self._scan_enabled = enabled
            if enabled:
                self._scan_static = static

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

    def _replace_layers(self, layers: tuple[KANLayer, ...], /) -> "KAN":
        if len(layers) != len(self.layers):
            raise ValueError("Replacement KAN layers must preserve the layer count.")
        scan_static = None
        scan_enabled = False
        if self.scan and len(layers) > 2:
            _, scan_static_candidate, scan_enabled = pack_scan_modules(layers[1:-1])
            if scan_enabled:
                scan_static = scan_static_candidate
        return eqx.tree_at(
            lambda model: (
                model.layers,
                model._scan_enabled,
                model._scan_static,
            ),
            self,
            (layers, scan_enabled, scan_static),
            is_leaf=lambda value: value is None,
        )

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        """Evaluate the KAN at ``x``."""
        y = x
        out = None
        if self._scan_enabled and self._scan_static is not None:
            repeated_layers = self.layers[1:-1]
            dynamic = stack_scan_dynamics(repeated_layers)
            if dynamic is not None:
                y = self.layers[0](y)
                y = scan_apply(
                    dynamic,
                    self._scan_static,
                    y,
                    lambda carry, layer: layer(carry),
                )
                out = self.layers[-1](y)
        if out is None:
            for layer in self.layers:
                y = layer(y)
            out = y
        y = out
        if self.skip_connection:
            residual = self._residual_proj(x) if self._residual_proj is not None else x
            y = y + residual
        return self.final_activation(y)

    def regularization_loss(self, *, alpha: float = 1e-4) -> Array:
        """Return the sum of basis-specific edge regularizers."""
        regularization = jnp.array(0.0)
        for layer in self.layers:
            regularization = regularization + layer.regularization()
        return jnp.asarray(alpha, dtype=regularization.dtype) * regularization
