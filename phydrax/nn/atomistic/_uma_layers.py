#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""UMA/eSCN-MD layer formulas in the pinned fairchem source parameterization.

Every trainable array keeps its source layout (torch ``Linear`` weights are
``(out, in)``), so a source parameter maps to exactly one native leaf. Linear
maps are either plain (`UMALinear`) or mixtures of linear experts
(`UMAMixtureLinear`, source ``MOLE``: weights ``(E, out, in)`` and one shared
bias). A mixture is never applied as an ensemble of predictions: per
evaluation it is *resolved* against per-case expert coefficients into the
mixed linear operator ``W_b = sum_e c_be W_e`` (`UMACaseLinear`), or *merged*
for one fixed binding into a plain `UMALinear`.

Edge layers act on one directed edge at a time (the streamed relation maps
them over bounded edge tiles); node layers act on node rows. Edge-frame
coefficients are reduced m-major rows (`UMACoefficientLayout`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._model import register_artifact_value
from ..._strict import StrictModule
from ..._trainable import NonTrainableState, ParameterOwner
from ..._validation import nonnegative_integer, positive_finite_float, positive_integer
from ...ein import contract
from ...typing import Dim, parse, Size
from ._uma_so3 import UMACoefficientLayout, UMASphericalGrid


UMANormalizationKind: TypeAlias = Literal["layer_norm", "layer_norm_sh", "rms_norm_sh"]
UMAEdgeActivation: TypeAlias = Literal["gate", "s2"]
UMAFeedForward: TypeAlias = Literal["grid", "spectral"]

_FLOATS = (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64))
# Source epsilon of every equivariant normalization and radial LayerNorm.
UMA_NORM_EPSILON = 1.0e-5


def _float_array(value: ArrayLike, name: str, /, *, rank: int) -> Array:
    array = jnp.asarray(value)
    if array.dtype not in _FLOATS:
        raise TypeError(f"{name} must be float32 or float64.")
    if array.ndim != rank or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty rank-{rank} array.")
    return array


@final
class UMALinear(StrictModule, ParameterOwner):
    """Source ``torch.nn.Linear``: ``x @ weight.T + bias`` with ``weight (out, in)``."""

    weight: Array
    bias: Array | None

    def __init__(self, weight: ArrayLike, bias: ArrayLike | None = None, /) -> None:
        weight_ = _float_array(weight, "Linear weight", rank=2)
        bias_ = None if bias is None else _float_array(bias, "Linear bias", rank=1)
        if bias_ is not None and (
            bias_.shape[0] != weight_.shape[0] or bias_.dtype != weight_.dtype
        ):
            raise ValueError("Linear bias must match the output width and dtype.")
        self.weight = weight_
        self.bias = bias_

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])

    def __call__(self, values: Array, /) -> Array:
        output = values @ self.weight.T
        return output if self.bias is None else output + self.bias


@final
class UMACaseLinear(StrictModule):
    """Mixed linear operators ``W_b`` of one evaluation, selected by case.

    ``weights`` is ``(B, out, in)``. Application forms every case's product and
    selects the row's case, so working memory is ``B x out`` per row instead
    of a gathered ``out x in`` matrix per row.
    """

    weights: Array
    bias: Array | None

    def __init__(self, weights: Array, bias: Array | None, /) -> None:
        self.weights = weights
        self.bias = bias

    @property
    def in_features(self) -> int:
        return int(self.weights.shape[2])

    @property
    def out_features(self) -> int:
        return int(self.weights.shape[1])

    def __call__(self, values: Array, case: Array, /) -> Array:
        """``case`` is one index or broadcasts to the leading shape of ``values``."""
        products = contract("boi,...i->b...o", self.weights, values)
        if jnp.ndim(case) == 0:
            output = products[case]
        else:
            index = jnp.broadcast_to(case, values.shape[:-1])[None, ..., None]
            output = jnp.take_along_axis(products, index, axis=0)[0]
        return output if self.bias is None else output + self.bias


@final
class UMAMixtureLinear(StrictModule, ParameterOwner):
    """Source ``MOLE``: expert weights ``(E, out, in)`` and one shared bias."""

    weights: Array
    bias: Array | None

    def __init__(self, weights: ArrayLike, bias: ArrayLike | None = None, /) -> None:
        weights_ = _float_array(weights, "MoLE weights", rank=3)
        bias_ = None if bias is None else _float_array(bias, "MoLE bias", rank=1)
        if bias_ is not None and (
            bias_.shape[0] != weights_.shape[1] or bias_.dtype != weights_.dtype
        ):
            raise ValueError("MoLE bias must match the output width and dtype.")
        self.weights = weights_
        self.bias = bias_

    @property
    def expert_count(self) -> int:
        return int(self.weights.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.weights.shape[2])

    @property
    def out_features(self) -> int:
        return int(self.weights.shape[1])

    def resolve(self, coefficients: Array, /) -> UMACaseLinear:
        """Per-case mixed operators from expert coefficients ``(B, E)``."""
        mixed = contract(
            "eoi,be->boi", self.weights, coefficients.astype(self.weights.dtype)
        )
        return UMACaseLinear(mixed, self.bias)

    def merge(self, coefficients: Array, /) -> UMALinear:
        """The single mixed operator of one fixed expert coefficient vector ``(E,)``."""
        mixed = contract(
            "eoi,e->oi", self.weights, coefficients.astype(self.weights.dtype)
        )
        return UMALinear(mixed, self.bias)


class UMAInputFeatureDim(Dim):
    pass


class UMAOutputFeatureDim(Dim):
    pass


class UMAExpertDim(Dim):
    pass


@final
class UMASharedMixture(StrictModule, NonTrainableState):
    """Reference to one parameter-shared bias-free mixture (source ``moe_single``).

    The source caches MOLE layers by ``(in, out, bias)``; bias-free layers of
    equal shape are one module (one parameter) across every mixed block. The
    owning potential stores the shared mixtures once and binds references
    before evaluation, so gradients accumulate on the single shared leaf.
    """

    __strict_contract__ = True

    index: int = eqx.field(static=True)
    in_features: Size[UMAInputFeatureDim] = eqx.field(static=True)
    out_features: Size[UMAOutputFeatureDim] = eqx.field(static=True)
    expert_count: Size[UMAExpertDim] = eqx.field(static=True)

    def __init__(
        self, index: int, in_features: int, out_features: int, expert_count: int, /
    ) -> None:
        self.index = nonnegative_integer(index, "index")
        self.in_features = positive_integer(in_features, "in_features")
        self.out_features = positive_integer(out_features, "out_features")
        self.expert_count = positive_integer(expert_count, "expert_count")

    @property
    def bias(self) -> None:
        return None


# A linear-map slot across realization stages: stored parameters are plain,
# mixture or shared-mixture references; `bind_shared_mixtures` replaces
# references by mixtures, `resolve_mixtures` mixtures by per-case operators and
# `merge_mixtures` mixtures by plain linears. Only plain and per-case maps apply.
UMALinearMap: TypeAlias = UMALinear | UMAMixtureLinear | UMASharedMixture | UMACaseLinear
UMAResolvedLinear: TypeAlias = UMALinear | UMACaseLinear


def apply_linear(linear: UMALinearMap, values: Array, case: Array, /) -> Array:
    """Apply a plain or resolved linear map; an unresolved mixture is refused."""
    if isinstance(linear, UMALinear):
        return linear(values)
    if isinstance(linear, UMACaseLinear):
        return linear(values, case)
    raise TypeError("A UMA mixture linear must be bound, resolved or merged before use.")


def bind_shared_mixtures[Tree](tree: Tree, shared: Sequence[UMAMixtureLinear], /) -> Tree:
    """Replace every `UMASharedMixture` reference by its shared mixture."""
    return jax.tree_util.tree_map(
        lambda node: shared[node.index] if isinstance(node, UMASharedMixture) else node,
        tree,
        is_leaf=lambda node: isinstance(node, UMASharedMixture),
    )


def resolve_mixtures[Tree](tree: Tree, coefficients: Array, /) -> Tree:
    """Replace every `UMAMixtureLinear` in ``tree`` by its per-case resolution."""
    return jax.tree_util.tree_map(
        lambda node: (
            node.resolve(coefficients) if isinstance(node, UMAMixtureLinear) else node
        ),
        tree,
        is_leaf=lambda node: isinstance(node, UMAMixtureLinear),
    )


def merge_mixtures[Tree](tree: Tree, coefficients: Array, /) -> Tree:
    """Replace every `UMAMixtureLinear` in ``tree`` by its merged plain linear."""
    return jax.tree_util.tree_map(
        lambda node: (
            node.merge(coefficients) if isinstance(node, UMAMixtureLinear) else node
        ),
        tree,
        is_leaf=lambda node: isinstance(node, UMAMixtureLinear),
    )


def mixture_linears(tree: object, /) -> tuple[UMAMixtureLinear, ...]:
    leaves = jax.tree_util.tree_leaves(
        tree, is_leaf=lambda node: isinstance(node, UMAMixtureLinear)
    )
    return tuple(leaf for leaf in leaves if isinstance(leaf, UMAMixtureLinear))


@final
class UMALayerNorm(StrictModule, ParameterOwner):
    """Source ``torch.nn.LayerNorm`` over the last axis (biased variance)."""

    weight: Array
    bias: Array
    epsilon: float = eqx.field(static=True)

    def __init__(
        self, weight: ArrayLike, bias: ArrayLike, /, *, epsilon: float = UMA_NORM_EPSILON
    ) -> None:
        weight_ = _float_array(weight, "LayerNorm weight", rank=1)
        bias_ = _float_array(bias, "LayerNorm bias", rank=1)
        if bias_.shape != weight_.shape or bias_.dtype != weight_.dtype:
            raise ValueError("LayerNorm weight and bias must share shape and dtype.")
        self.weight = weight_
        self.bias = bias_
        self.epsilon = positive_finite_float(epsilon, "epsilon")

    def __call__(self, values: Array, /) -> Array:
        mean = jnp.mean(values, axis=-1, keepdims=True)
        centered = values - mean
        variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
        return centered * jax.lax.rsqrt(variance + self.epsilon) * self.weight + self.bias


@final
class UMARadialMLP(StrictModule, ParameterOwner):
    """Source ``RadialMLP``: ``(Linear, LayerNorm, SiLU)*`` then a final ``Linear``."""

    linears: tuple[UMALinearMap, ...]
    norms: tuple[UMALayerNorm, ...]

    def __init__(
        self, linears: Sequence[UMALinearMap], norms: Sequence[UMALayerNorm], /
    ) -> None:
        linears_ = tuple(linears)
        norms_ = tuple(norms)
        if not linears_ or len(norms_) != len(linears_) - 1:
            raise ValueError("A radial MLP has one LayerNorm per hidden Linear.")
        for index, linear in enumerate(linears_):
            if (
                not isinstance(linear, (UMALinear, UMAMixtureLinear))
                or linear.bias is None
            ):
                raise TypeError("Radial MLP layers are biased linear maps.")
            if index and linear.in_features != linears_[index - 1].out_features:
                raise ValueError("Radial MLP widths are inconsistent.")
        for norm, linear in zip(norms_, linears_[:-1], strict=True):
            if norm.weight.shape[0] != linear.out_features:
                raise ValueError("Radial MLP LayerNorm widths are inconsistent.")
        self.linears = linears_
        self.norms = norms_

    @property
    def input_width(self) -> int:
        return self.linears[0].in_features

    @property
    def output_width(self) -> int:
        return self.linears[-1].out_features

    def __call__(self, values: Array, case: Array, /) -> Array:
        hidden = values
        for linear, norm in zip(self.linears[:-1], self.norms, strict=True):
            hidden = jax.nn.silu(norm(apply_linear(linear, hidden, case)))
        return apply_linear(self.linears[-1], hidden, case)


@final
class UMASO3Linear(StrictModule, ParameterOwner):
    """Source ``SO3_Linear``: per-degree channel maps ``(L+1, out, in)``, scalar bias."""

    weight: Array
    bias: Array

    def __init__(self, weight: ArrayLike, bias: ArrayLike, /) -> None:
        weight_ = _float_array(weight, "SO3_Linear weight", rank=3)
        bias_ = _float_array(bias, "SO3_Linear bias", rank=1)
        if bias_.shape[0] != weight_.shape[1] or bias_.dtype != weight_.dtype:
            raise ValueError("SO3_Linear bias must match its output width and dtype.")
        self.weight = weight_
        self.bias = bias_

    @property
    def lmax(self) -> int:
        return int(self.weight.shape[0]) - 1

    def __call__(self, values: Array, /) -> Array:
        """``(..., (L+1)**2, in) -> (..., (L+1)**2, out)``; bias on the ``l = 0`` row."""
        expand = np.repeat(np.arange(self.lmax + 1), 2 * np.arange(self.lmax + 1) + 1)
        weight = self.weight[jnp.asarray(expand, dtype=jnp.int32)]
        output = contract("...mi,moi->...mo", values, weight)
        return output.at[..., 0, :].add(self.bias)


def _degree_weights(lmax: int, /, *, include_scalar: bool) -> np.ndarray:
    """Source ``balance_degree_weight``: ``1 / ((2l+1) * degrees)`` per component."""
    first = 0 if include_scalar else 1
    count = lmax + 1 - first
    weights = np.concatenate(
        [
            np.full(2 * degree + 1, 1.0 / (2 * degree + 1))
            for degree in range(first, lmax + 1)
        ]
    )
    return weights / count


@final
class UMANormalization(StrictModule, ParameterOwner):
    """Source equivariant normalizations (``normalization="component"``, affine).

    - ``"layer_norm"`` (``EquivariantLayerNormArray``): per degree, the scalar
      degree is channel-centred; the inverse RMS over components and channels
      scales ``weight[l]``; ``bias`` is added to ``l = 0``.
    - ``"layer_norm_sh"`` (``EquivariantLayerNormArraySphericalHarmonics``):
      ``l = 0`` is a channel LayerNorm (``scalar_weight``, ``scalar_bias``);
      ``l > 0`` share one inverse RMS with degree-balanced component weights
      ``1 / ((2l+1) L)`` and per-degree ``weight[l - 1]``.
    - ``"rms_norm_sh"`` (``EquivariantRMSNormArraySphericalHarmonicsV2``): the
      ``l = 0`` row is channel-centred, one degree-balanced inverse RMS over all
      degrees (weights ``1 / ((2l+1)(L+1))``) scales ``weight[l]`` and ``bias``
      is added to ``l = 0``.
    """

    weight: Array
    bias: Array | None
    scalar_weight: Array | None
    scalar_bias: Array | None
    kind: UMANormalizationKind = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        kind: UMANormalizationKind,
        lmax: int,
        /,
        *,
        weight: ArrayLike,
        bias: ArrayLike | None = None,
        scalar_weight: ArrayLike | None = None,
        scalar_bias: ArrayLike | None = None,
        epsilon: float = UMA_NORM_EPSILON,
    ) -> None:
        kind_ = parse(kind, UMANormalizationKind, "normalization kind")
        lmax_ = nonnegative_integer(lmax, "lmax")
        weight_ = _float_array(weight, "normalization weight", rank=2)
        channels = int(weight_.shape[1])
        rows = lmax_ if kind_ == "layer_norm_sh" else lmax_ + 1
        if kind_ == "layer_norm_sh" and lmax_ == 0:
            raise ValueError("layer_norm_sh needs lmax >= 1 for its degree weights.")
        if weight_.shape[0] != rows:
            raise ValueError(f"{kind_} weight must have {rows} degree rows.")
        expects_bias = kind_ != "layer_norm_sh"
        expects_scalar = kind_ == "layer_norm_sh"
        if (bias is not None) != expects_bias or (
            (scalar_weight is not None, scalar_bias is not None)
            != (expects_scalar, expects_scalar)
        ):
            raise ValueError(f"{kind_} affine parameters do not match its definition.")

        def vector(value: ArrayLike | None, name: str, /) -> Array | None:
            if value is None:
                return None
            array = _float_array(value, name, rank=1)
            if array.shape != (channels,) or array.dtype != weight_.dtype:
                raise ValueError(f"{name} must have shape ({channels},).")
            return array

        self.weight = weight_
        self.bias = vector(bias, "normalization bias")
        self.scalar_weight = vector(scalar_weight, "scalar LayerNorm weight")
        self.scalar_bias = vector(scalar_bias, "scalar LayerNorm bias")
        self.kind = kind_
        self.lmax = lmax_
        self.epsilon = positive_finite_float(epsilon, "epsilon")

    @property
    def channels(self) -> int:
        return int(self.weight.shape[1])

    def __call__(self, values: Array, /) -> Array:
        """Normalize node features ``(..., (L+1)**2, C)``."""
        lmax = self.lmax
        expand = jnp.asarray(
            np.repeat(np.arange(lmax + 1), 2 * np.arange(lmax + 1) + 1), dtype=jnp.int32
        )
        match self.kind:
            case "layer_norm":
                outputs = []
                for degree in range(lmax + 1):
                    feature = values[..., degree * degree : (degree + 1) ** 2, :]
                    if degree == 0:
                        feature = feature - jnp.mean(feature, axis=-1, keepdims=True)
                    norm = jnp.mean(feature * feature, axis=(-2, -1), keepdims=True)
                    scale = jax.lax.rsqrt(norm + self.epsilon) * self.weight[degree]
                    feature = feature * scale
                    if degree == 0:
                        assert self.bias is not None
                        feature = feature + self.bias
                    outputs.append(feature)
                return jnp.concatenate(outputs, axis=-2)
            case "layer_norm_sh":
                assert self.scalar_weight is not None and self.scalar_bias is not None
                scalar = values[..., :1, :]
                mean = jnp.mean(scalar, axis=-1, keepdims=True)
                centered = scalar - mean
                variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
                scalar = (
                    centered * jax.lax.rsqrt(variance + self.epsilon) * self.scalar_weight
                    + self.scalar_bias
                )
                vectors = values[..., 1:, :]
                weights = jnp.asarray(
                    _degree_weights(lmax, include_scalar=False), dtype=values.dtype
                )
                norm = contract("...ic,i->...c", vectors * vectors, weights)
                norm = jnp.mean(norm, axis=-1)[..., None, None]
                scale = jax.lax.rsqrt(norm + self.epsilon) * self.weight[expand[1:] - 1]
                return jnp.concatenate([scalar, vectors * scale], axis=-2)
            case "rms_norm_sh":
                assert self.bias is not None
                scalar = values[..., :1, :]
                scalar = scalar - jnp.mean(scalar, axis=-1, keepdims=True)
                feature = jnp.concatenate([scalar, values[..., 1:, :]], axis=-2)
                weights = jnp.asarray(
                    _degree_weights(lmax, include_scalar=True), dtype=values.dtype
                )
                norm = contract("...ic,i->...c", feature * feature, weights)
                norm = jnp.mean(norm, axis=-1)[..., None, None]
                output = feature * (
                    jax.lax.rsqrt(norm + self.epsilon) * self.weight[expand]
                )
                return output.at[..., 0, :].add(self.bias)


def gate_activation(
    gating: Array, values: Array, row_degrees: Sequence[int], lmax: int, /
) -> Array:
    """Source ``GateActivation``: SiLU on row 0, ``sigmoid`` gates of degree ``l``.

    ``gating`` is ``(..., lmax * C)`` (gate ``l - 1`` occupies channels
    ``(l - 1) C .. l C``); ``row_degrees`` gives each row's degree in the
    ordering of ``values`` (m-major edge rows or l-major node rows).
    """
    channels = values.shape[-1]
    gates = jax.nn.sigmoid(gating).reshape(gating.shape[:-1] + (lmax, channels))
    index = jnp.asarray(np.asarray(row_degrees[1:], dtype=np.int32) - 1)
    scalars = jax.nn.silu(values[..., :1, :])
    vectors = values[..., 1:, :] * jnp.take(gates, index, axis=-2)
    return jnp.concatenate([scalars, vectors], axis=-2)


def separable_s2_activation(
    scalars: Array,
    values: Array,
    grid: UMASphericalGrid,
    layout: UMACoefficientLayout,
    /,
) -> Array:
    """Source ``SeparableS2Activation_M`` on m-major edge rows.

    Row 0 becomes ``silu(scalars)``; the other rows are the ``lmax_mmax`` grid
    projection of ``silu`` applied to the grid values of every row.
    """
    tensors = grid.integrate_m_major(
        jax.nn.silu(grid.project_m_major(values, layout)), layout
    )
    return jnp.concatenate(
        [jax.nn.silu(scalars)[..., None, :], tensors[..., 1:, :]], axis=-2
    )


@final
class UMASO2Convolution(StrictModule, ParameterOwner):
    """Source ``SO2_Convolution`` on one edge's m-major rows ``(R, C_in)``.

    ``m = 0`` rows are flattened degree-major, scaled by their radial slice and
    mapped by the biased ``m0`` linear (its first ``extra`` outputs are the
    gating scalars); each ``m > 0`` order maps ``[x_+m, x_-m]`` by ``W = [W1;
    W2]`` as the complex product ``(W1 + i W2)(x_+m + i x_-m)``, which commutes
    with rolls about the edge axis.
    """

    m0: UMALinearMap
    orders: tuple[UMALinearMap, ...]
    radial: UMARadialMLP | None
    layout: UMACoefficientLayout
    channels_in: int = eqx.field(static=True)
    channels_out: int = eqx.field(static=True)
    extra: int = eqx.field(static=True)

    def __init__(
        self,
        layout: UMACoefficientLayout,
        m0: UMALinearMap,
        orders: Sequence[UMALinearMap],
        /,
        *,
        channels_in: int,
        channels_out: int,
        extra: int = 0,
        radial: UMARadialMLP | None = None,
    ) -> None:
        cin = positive_integer(channels_in, "channels_in")
        cout = positive_integer(channels_out, "channels_out")
        extra_ = nonnegative_integer(extra, "extra")
        orders_ = tuple(orders)
        if len(orders_) != layout.mmax:
            raise ValueError("An SO(2) convolution has one linear per order m >= 1.")
        degrees = layout.lmax + 1
        if (
            m0.in_features != degrees * cin
            or m0.out_features != degrees * cout + extra_
            or m0.bias is None
        ):
            raise ValueError("The m = 0 linear has inconsistent widths.")
        widths = [m0.in_features]
        for order, linear in enumerate(orders_, start=1):
            count = layout.lmax - order + 1
            if (
                linear.in_features != count * cin
                or linear.out_features != 2 * count * cout
                or linear.bias is not None
            ):
                raise ValueError(f"The m = {order} linear has inconsistent widths.")
            widths.append(linear.in_features)
        if radial is not None and radial.output_width != sum(widths):
            raise ValueError("The radial MLP width must cover every order's inputs.")
        self.m0 = m0
        self.orders = orders_
        self.radial = radial
        self.layout = layout
        self.channels_in = cin
        self.channels_out = cout
        self.extra = extra_

    def __call__(
        self, values: Array, edge_features: Array | None, case: Array, /
    ) -> tuple[Array, Array]:
        """``(R, C_in) -> ((R, C_out), (extra,))``; radial weights from ``edge_features``."""
        sizes = self.layout.m_split_sizes
        offsets = np.cumsum((0, *sizes))
        radial = None
        if self.radial is not None:
            if edge_features is None:
                raise ValueError("This SO(2) convolution needs edge features.")
            weights = self.radial(edge_features, case)
            widths = [
                self.m0.in_features,
                *(linear.in_features for linear in self.orders),
            ]
            cuts = np.cumsum((0, *widths))
            radial = [weights[cuts[i] : cuts[i + 1]] for i in range(len(widths))]
        zero = values[offsets[0] : offsets[1]].reshape(-1)
        if radial is not None:
            zero = zero * radial[0]
        zero = apply_linear(self.m0, zero, case)
        gating, zero = zero[: self.extra], zero[self.extra :]
        outputs = [zero.reshape(-1, self.channels_out)]
        for order, linear in enumerate(self.orders, start=1):
            pair = values[offsets[order] : offsets[order + 1]].reshape(2, -1)
            if radial is not None:
                pair = pair * radial[order][None, :]
            mapped = apply_linear(linear, pair, case)
            half = mapped.shape[-1] // 2
            real_first, imaginary_first = mapped[0, :half], mapped[0, half:]
            real_second, imaginary_second = mapped[1, :half], mapped[1, half:]
            outputs.append((real_first - imaginary_second).reshape(-1, self.channels_out))
            outputs.append((real_second + imaginary_first).reshape(-1, self.channels_out))
        return jnp.concatenate(outputs, axis=0), gating


@final
class UMAEdgewise(StrictModule, ParameterOwner):
    """Source ``Edgewise``: rotate, ``SO2_Convolution``, activation, ``SO2_Convolution``, unrotate.

    ``wigner_inverse`` arrives with the polynomial envelope already applied
    (the source's ``wigner_inv_envelope``).
    """

    first: UMASO2Convolution
    second: UMASO2Convolution
    grid: UMASphericalGrid | None
    activation: UMAEdgeActivation = eqx.field(static=True)

    def __init__(
        self,
        first: UMASO2Convolution,
        second: UMASO2Convolution,
        /,
        *,
        activation: UMAEdgeActivation,
        grid: UMASphericalGrid | None = None,
    ) -> None:
        activation_ = parse(activation, UMAEdgeActivation, "activation")
        hidden = first.channels_out
        lmax = first.layout.lmax
        expected = lmax * hidden if activation_ == "gate" else hidden
        if first.extra != expected or first.radial is None:
            raise ValueError("The first SO(2) convolution has the wrong gating width.")
        if second.radial is not None or second.extra or second.channels_in != hidden:
            raise ValueError("The second SO(2) convolution is internal and gate-free.")
        if (activation_ == "s2") != (grid is not None):
            raise ValueError("Only the S2 activation uses the edge grid.")
        if grid is not None and (grid.lmax, grid.mmax) != (lmax, first.layout.mmax):
            raise ValueError("The S2 activation grid must be the lmax_mmax grid.")
        self.first = first
        self.second = second
        self.grid = grid
        self.activation = activation_

    def message(
        self,
        source: Array,
        target: Array,
        wigner: Array,
        wigner_inverse: Array,
        edge_features: Array,
        case: Array,
        /,
    ) -> Array:
        """Node-frame message ``(F, C)`` of one edge from ``(F, C)`` node rows."""
        rotated = wigner @ jnp.concatenate([source, target], axis=-1)
        hidden, gating = self.first(rotated, edge_features, case)
        layout = self.first.layout
        if self.activation == "gate":
            hidden = gate_activation(gating, hidden, layout.m_major_degrees, layout.lmax)
        else:
            assert self.grid is not None
            hidden = separable_s2_activation(gating, hidden, self.grid, layout)
        output, _ = self.second(hidden, None, case)
        return wigner_inverse @ output


@final
class UMAGridAtomwise(StrictModule, ParameterOwner):
    """Source ``GridAtomwise``: bias-free ``Linear, SiLU, Linear, SiLU, Linear`` on the grid."""

    linears: tuple[UMALinearMap, UMALinearMap, UMALinearMap]
    grid: UMASphericalGrid

    def __init__(
        self, linears: Sequence[UMALinearMap], grid: UMASphericalGrid, /
    ) -> None:
        linears_ = tuple(linears)
        if len(linears_) != 3 or any(linear.bias is not None for linear in linears_):
            raise ValueError("The grid feed-forward has three bias-free linears.")
        if grid.lmax != grid.mmax:
            raise ValueError("The grid feed-forward uses the lmax_lmax grid.")
        first, second, third = linears_
        self.linears = (first, second, third)
        self.grid = grid

    def __call__(self, values: Array, case: Array, /) -> Array:
        hidden = self.grid.project(values)
        first, second, third = self.linears
        hidden = jax.nn.silu(apply_linear(first, hidden, case[..., None, None]))
        hidden = jax.nn.silu(apply_linear(second, hidden, case[..., None, None]))
        hidden = apply_linear(third, hidden, case[..., None, None])
        return self.grid.integrate(hidden)


@final
class UMASpectralAtomwise(StrictModule, ParameterOwner):
    """Source ``SpectralAtomwise``: scalar-gated ``SO3_Linear`` pair."""

    scalar: UMALinearMap
    first: UMASO3Linear
    second: UMASO3Linear

    def __init__(
        self, scalar: UMALinearMap, first: UMASO3Linear, second: UMASO3Linear, /
    ) -> None:
        hidden = int(first.weight.shape[1])
        if scalar.bias is None or scalar.out_features != first.lmax * hidden:
            raise ValueError("The spectral gate linear must emit lmax * hidden gates.")
        if second.weight.shape[2] != hidden or first.lmax != second.lmax:
            raise ValueError("The spectral SO3 linears are inconsistent.")
        self.scalar = scalar
        self.first = first
        self.second = second

    def __call__(self, values: Array, case: Array, /) -> Array:
        lmax = self.first.lmax
        gating = jax.nn.silu(apply_linear(self.scalar, values[..., 0, :], case))
        hidden = self.first(values)
        degrees = tuple(
            int(d) for d in np.repeat(np.arange(lmax + 1), 2 * np.arange(lmax + 1) + 1)
        )
        hidden = gate_activation(gating, hidden, degrees, lmax)
        return self.second(hidden)


UMAAtomwise: TypeAlias = UMAGridAtomwise | UMASpectralAtomwise


@final
class UMABlock(StrictModule, ParameterOwner):
    """Source ``eSCNMD_Block``: pre-norm edgewise and atom-wise residual updates.

    ``x <- x + edgewise(norm_1(x) + system embedding on l=0)``, then
    ``x <- x + atomwise(norm_2(x))``.
    """

    norm_1: UMANormalization
    edgewise: UMAEdgewise
    norm_2: UMANormalization
    atomwise: UMAAtomwise

    def __init__(
        self,
        norm_1: UMANormalization,
        edgewise: UMAEdgewise,
        norm_2: UMANormalization,
        atomwise: UMAAtomwise,
        /,
    ) -> None:
        self.norm_1 = norm_1
        self.edgewise = edgewise
        self.norm_2 = norm_2
        self.atomwise = atomwise


@final
class UMAEdgeDegreeEmbedding(StrictModule, ParameterOwner):
    """Source ``EdgeDegreeEmbedding``: rotate radial ``m = 0`` features back and sum.

    The message of one edge is ``wigner_inverse[:, :m0] @ radial(x_edge).reshape(m0,
    C) / rescale`` with ``m0 = lmax + 1``.
    """

    radial: UMARadialMLP
    rescale: float = eqx.field(static=True)

    def __init__(self, radial: UMARadialMLP, /, *, rescale: float) -> None:
        self.radial = radial
        self.rescale = positive_finite_float(rescale, "rescale")

    def message(
        self, wigner_inverse: Array, edge_features: Array, case: Array, m0: int, /
    ) -> Array:
        radial = self.radial(edge_features, case).reshape(m0, -1)
        return (wigner_inverse[:, :m0] @ radial) / self.rescale


@final
class UMAGaussianSmearing(StrictModule, NonTrainableState):
    """Source ``GaussianSmearing(0, cutoff, n, 2)``: ``exp(c (d - mu_k)^2)``.

    Offsets are ``linspace(0, cutoff, n)`` and ``c = -1 / (2 (2 delta)^2)``.
    """

    offsets: Array
    coefficient: float = eqx.field(static=True)

    def __init__(self, cutoff: float, count: int, /, *, dtype: DTypeLike) -> None:
        cutoff_ = positive_finite_float(cutoff, "cutoff")
        count_ = positive_integer(count, "count")
        if count_ < 2:
            raise ValueError("Gaussian smearing needs at least two offsets.")
        offsets = np.linspace(0.0, cutoff_, count_)
        delta = float(offsets[1] - offsets[0])
        self.offsets = jnp.asarray(offsets, dtype=dtype)
        self.coefficient = -0.5 / (2.0 * delta) ** 2

    def __call__(self, distance: Array, /) -> Array:
        shifted = distance[..., None] - self.offsets
        return jnp.exp(self.coefficient * shifted * shifted)


def polynomial_envelope(scaled: Array, /, *, exponent: int = 5) -> Array:
    """Source ``PolynomialEnvelope(p)`` of ``d / cutoff``; zero at and beyond one."""
    p = float(exponent)
    a = -(p + 1.0) * (p + 2.0) / 2.0
    b = p * (p + 2.0)
    c = -p * (p + 1.0) / 2.0
    inside = scaled < 1.0
    safe = jnp.where(inside, scaled, 0.0)
    value = 1.0 + safe**exponent * (a + safe * (b + c * safe))
    return jnp.where(inside, value, 0.0)


for _type in (
    UMALinear,
    UMACaseLinear,
    UMAMixtureLinear,
    UMASharedMixture,
    UMALayerNorm,
    UMARadialMLP,
    UMASO3Linear,
    UMANormalization,
    UMASO2Convolution,
    UMAEdgewise,
    UMAGridAtomwise,
    UMASpectralAtomwise,
    UMABlock,
    UMAEdgeDegreeEmbedding,
    UMAGaussianSmearing,
):
    register_artifact_value(f"phydrax.nn.atomistic:{_type.__name__}", _type)


__all__ = [
    "apply_linear",
    "bind_shared_mixtures",
    "UMASharedMixture",
    "gate_activation",
    "merge_mixtures",
    "mixture_linears",
    "polynomial_envelope",
    "resolve_mixtures",
    "separable_s2_activation",
    "UMA_NORM_EPSILON",
    "UMAAtomwise",
    "UMABlock",
    "UMACaseLinear",
    "UMAEdgeActivation",
    "UMAEdgeDegreeEmbedding",
    "UMAEdgewise",
    "UMAFeedForward",
    "UMAGaussianSmearing",
    "UMAGridAtomwise",
    "UMALayerNorm",
    "UMALinear",
    "UMALinearMap",
    "UMAMixtureLinear",
    "UMANormalization",
    "UMARadialMLP",
    "UMAResolvedLinear",
    "UMASO2Convolution",
    "UMASO3Linear",
    "UMASpectralAtomwise",
]
