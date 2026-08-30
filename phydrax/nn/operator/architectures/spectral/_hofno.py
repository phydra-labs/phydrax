#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from string import ascii_lowercase
from typing import cast, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._spectral._fourier import fourier_resample as _fourier_resample
from phydrax._strict import StrictModule
from phydrax.nn._keys import EvalKey, fold_in_eval_key
from phydrax.nn._scan import (
    pack_scan_modules,
    scan_apply_with_data,
    stack_scan_dynamics,
)
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._dropout import _dropout_probabilities, Dropout
from phydrax.nn.layers._linear import Linear
from phydrax.nn.operator.architectures.spectral._fno import (
    _activation,
    _mode_tuple,
    Activation,
    SpectralConvND,
)
from phydrax.nn.operator.data import OperatorAxis, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


AliasingPolicy = Literal["collocation", "dealiased"]
SpectralChannelMixing = Literal["depthwise", "dense"]


def _dealiased_spectral_resample(
    values: Array,
    output_shape: Sequence[int],
    /,
) -> Array:
    """Band-limited periodic resampling that preserves real Nyquist modes."""
    return _fourier_resample(values, output_shape)


def _complex_normal(key: Key[Array, ""], shape: tuple[int, ...], scale: float) -> Array:
    real_key, imaginary_key = jr.split(key)
    return scale * (
        jr.normal(real_key, shape=shape) + 1j * jr.normal(imaginary_key, shape=shape)
    )


class _DepthwiseSpectralConvND(StrictModule):
    """N-dimensional Fourier convolution with one multiplier per channel and mode."""

    channels: int
    n_modes: tuple[int, ...]
    active_modes: tuple[int, ...]
    weight: Array

    def __init__(
        self,
        *,
        channels: int,
        n_modes: int | Sequence[int],
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.channels = int(channels)
        self.n_modes = _mode_tuple(n_modes)
        self.active_modes = self.n_modes
        if self.channels <= 0:
            raise ValueError("channels must be positive.")
        corners = 2 ** max(0, len(self.n_modes) - 1)
        self.weight = _complex_normal(
            key,
            (corners, self.channels, *self.n_modes),
            1.0 / float(self.channels),
        )

    def with_active_modes(
        self,
        modes: int | Sequence[int],
        /,
    ) -> _DepthwiseSpectralConvND:
        active = _mode_tuple(modes, len(self.n_modes))
        if any(now > maximum for now, maximum in zip(active, self.n_modes, strict=True)):
            raise ValueError("active modes cannot exceed initialized modes.")
        return eqx.tree_at(lambda layer: layer.active_modes, self, active)

    def __call__(self, x: Array, /) -> Array:
        values = jnp.asarray(x)
        ndim = len(self.n_modes)
        if values.ndim < ndim + 1:
            raise ValueError(
                f"Depthwise Fourier convolution expects at least {ndim + 1} axes; "
                f"got {values.ndim}."
            )
        if int(values.shape[-1]) != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {values.shape[-1]}."
            )

        spatial_axes = tuple(range(values.ndim - ndim - 1, values.ndim - 1))
        spatial_shape = tuple(int(values.shape[axis]) for axis in spatial_axes)
        transformed = jnp.fft.rfftn(values, axes=spatial_axes, norm="ortho")
        output_ft = jnp.zeros_like(transformed)
        modes = tuple(
            min(
                int(requested),
                size // 2 + 1 if index == ndim - 1 else max(1, size // 2),
            )
            for index, (requested, size) in enumerate(
                zip(self.active_modes, spatial_shape, strict=True)
            )
        )
        letters = "".join(letter for letter in ascii_lowercase if letter != "c")[:ndim]
        mode_slices = tuple(slice(0, mode) for mode in modes)

        signs = tuple(product((0, 1), repeat=max(0, ndim - 1)))
        for corner, corner_signs in enumerate(signs):
            slices = tuple(
                slice(0, mode) if sign == 0 else slice(-mode, None)
                for sign, mode in zip(corner_signs, modes[:-1], strict=True)
            ) + (slice(0, modes[-1]),)
            block = transformed[(..., *slices, slice(None))]
            weight = self.weight[(corner, slice(None), *mode_slices)]
            result = oe.contract(
                f"...{letters}c,c{letters}->...{letters}c",
                block,
                weight,
            )
            output_ft = output_ft.at[(..., *slices, slice(None))].set(result)

        return jnp.fft.irfftn(
            output_ft,
            s=spatial_shape,
            axes=spatial_axes,
            norm="ortho",
        )


class _ProjectedProductFourierMixer(StrictModule):
    """Projected polynomial interaction followed by a learned Fourier multiplier."""

    projection: Linear
    spectral: _DepthwiseSpectralConvND | SpectralConvND
    interaction_order: int
    channels: int
    n_modes: tuple[int, ...]
    aliasing: AliasingPolicy
    factor_bias: bool
    spectral_channel_mixing: SpectralChannelMixing

    def __init__(
        self,
        *,
        channels: int,
        n_modes: int | Sequence[int],
        interaction_order: int,
        factor_bias: bool,
        spectral_channel_mixing: SpectralChannelMixing,
        aliasing: AliasingPolicy,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.channels = int(channels)
        self.n_modes = _mode_tuple(n_modes)
        self.interaction_order = int(interaction_order)
        self.factor_bias = bool(factor_bias)
        self.spectral_channel_mixing = spectral_channel_mixing
        self.aliasing = aliasing
        if self.channels <= 0 or self.interaction_order <= 0:
            raise ValueError("channels and interaction_order must be positive.")
        if spectral_channel_mixing not in ("depthwise", "dense"):
            raise ValueError("spectral_channel_mixing must be 'depthwise' or 'dense'.")
        if aliasing not in ("collocation", "dealiased"):
            raise ValueError("aliasing must be 'collocation' or 'dealiased'.")

        projection_key, spectral_key = jr.split(key)
        self.projection = Linear(
            in_size=self.channels,
            out_size=(self.interaction_order, self.channels),
            activation=None,
            rwf=False,
            use_bias=self.factor_bias,
            bias_init_lim=0.0,
            key=projection_key,
        )
        if spectral_channel_mixing == "depthwise":
            self.spectral = _DepthwiseSpectralConvND(
                channels=self.channels,
                n_modes=self.n_modes,
                key=spectral_key,
            )
        else:
            self.spectral = SpectralConvND(
                in_channels=self.channels,
                out_channels=self.channels,
                n_modes=self.n_modes,
                factorization="dense",
                key=spectral_key,
            )

    @property
    def active_modes(self) -> tuple[int, ...]:
        return self.spectral.active_modes

    def with_active_modes(
        self,
        modes: int | Sequence[int],
        /,
    ) -> _ProjectedProductFourierMixer:
        replacement = self.spectral.with_active_modes(modes)
        return eqx.tree_at(lambda layer: layer.spectral, self, replacement)

    def _dealiased_shape(self, spatial_shape: tuple[int, ...], /) -> tuple[int, ...]:
        if self.interaction_order == 1:
            return spatial_shape
        return tuple(
            max(
                size,
                self.interaction_order * (size // 2) + int(mode) + 1,
            )
            for size, mode in zip(spatial_shape, self.active_modes, strict=True)
        )

    def __call__(self, values: Array, /) -> Array:
        array = jnp.asarray(values)
        ndim = len(self.n_modes)
        if array.ndim < ndim + 1 or int(array.shape[-1]) != self.channels:
            raise ValueError(
                "Higher-order Fourier mixer expects trailing spatial axes and "
                f"{self.channels} channels; got {array.shape}."
            )
        spatial_shape = tuple(int(size) for size in array.shape[-ndim - 1 : -1])
        factors = self.projection(array)
        working_shape = spatial_shape
        if self.aliasing == "dealiased":
            working_shape = self._dealiased_shape(spatial_shape)
            if working_shape != spatial_shape:
                leading_shape = tuple(int(size) for size in factors.shape[: -ndim - 2])
                flattened = factors.reshape(
                    leading_shape
                    + spatial_shape
                    + (self.interaction_order * self.channels,)
                )
                flattened = _dealiased_spectral_resample(flattened, working_shape)
                factors = flattened.reshape(
                    leading_shape
                    + working_shape
                    + (self.interaction_order, self.channels)
                )
        product_field = jnp.prod(factors, axis=-2)
        output = self.spectral(product_field)
        if working_shape != spatial_shape:
            output = _dealiased_spectral_resample(output, spatial_shape)
        return output


class _RMSNorm(StrictModule):
    scale: Array
    eps: float

    def __init__(self, channels: int, /, *, eps: float):
        self.scale = jnp.ones((int(channels),), dtype=float)
        self.eps = float(eps)
        if self.eps <= 0.0:
            raise ValueError("norm_epsilon must be positive.")

    def __call__(self, values: Array, /) -> Array:
        array = jnp.asarray(values)
        inverse_rms = jax.lax.rsqrt(
            jnp.mean(array * array, axis=-1, keepdims=True) + self.eps
        )
        return array * inverse_rms * self.scale


class _HigherOrderFeedForward(StrictModule):
    expand: Linear
    project: Linear
    hidden_dropout: Dropout
    output_dropout: Dropout
    activation: Activation

    def __init__(
        self,
        channels: int,
        /,
        *,
        expansion: int,
        activation: Activation,
        dropout: float,
        key: Key[Array, ""],
    ):
        hidden_channels = int(channels) * int(expansion)
        if hidden_channels <= 0:
            raise ValueError("ffn_expansion must be positive.")
        expand_key, project_key = jr.split(key)
        self.expand = Linear(
            in_size=channels,
            out_size=hidden_channels,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=expand_key,
        )
        self.project = Linear(
            in_size=hidden_channels,
            out_size=channels,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=project_key,
        )
        self.hidden_dropout = Dropout(hidden_channels, p=dropout, mode="feature")
        self.output_dropout = Dropout(channels, p=dropout, mode="feature")
        self.activation = activation

    def __call__(self, values: Array, /, *, key: EvalKey) -> Array:
        hidden = _activation(self.activation, self.expand(values))
        hidden = self.hidden_dropout(hidden, key=fold_in_eval_key(key, 0))
        output = self.project(hidden)
        return self.output_dropout(output, key=fold_in_eval_key(key, 1))


class _HigherOrderFNOBlock(StrictModule):
    spectral: _ProjectedProductFourierMixer
    mixer_norm: _RMSNorm
    feedforward_norm: _RMSNorm
    feedforward: _HigherOrderFeedForward
    mixer_dropout: Dropout
    residual: bool

    def __init__(
        self,
        *,
        channels: int,
        n_modes: tuple[int, ...],
        interaction_order: int,
        factor_bias: bool,
        spectral_channel_mixing: SpectralChannelMixing,
        aliasing: AliasingPolicy,
        activation: Activation,
        ffn_expansion: int,
        norm_epsilon: float,
        dropout: float,
        residual: bool,
        key: Key[Array, ""],
    ):
        mixer_key, feedforward_key = jr.split(key)
        self.spectral = _ProjectedProductFourierMixer(
            channels=channels,
            n_modes=n_modes,
            interaction_order=interaction_order,
            factor_bias=factor_bias,
            spectral_channel_mixing=spectral_channel_mixing,
            aliasing=aliasing,
            key=mixer_key,
        )
        self.mixer_norm = _RMSNorm(channels, eps=norm_epsilon)
        self.feedforward_norm = _RMSNorm(channels, eps=norm_epsilon)
        self.feedforward = _HigherOrderFeedForward(
            channels,
            expansion=ffn_expansion,
            activation=activation,
            dropout=dropout,
            key=feedforward_key,
        )
        self.mixer_dropout = Dropout(channels, p=dropout, mode="feature")
        self.residual = bool(residual)

    def __call__(self, values: Array, /, *, key: EvalKey = None) -> Array:
        mixer_update = self.spectral(self.mixer_norm(values))
        mixer_update = self.mixer_dropout(
            mixer_update,
            key=fold_in_eval_key(key, 0),
        )
        hidden = values + mixer_update if self.residual else mixer_update
        feedforward_update = self.feedforward(
            self.feedforward_norm(hidden),
            key=fold_in_eval_key(key, 1),
        )
        return hidden + feedforward_update if self.residual else feedforward_update


def _hofno_contract_configuration(model):
    return (
        ("n_modes", model.n_modes),
        ("width", model.width),
        ("depth", len(model.blocks)),
        ("interaction_order", model.interaction_order),
        ("factor_bias", model.factor_bias),
        ("spectral_channel_mixing", model.spectral_channel_mixing),
        ("aliasing", model.aliasing),
        ("ffn_expansion", model.ffn_expansion),
        ("norm_epsilon", model.norm_epsilon),
        ("coordinate_embedding", model.coordinate_embedding),
        ("domain_padding", model.domain_padding),
        ("source_key", model.source_key),
        ("scan", model.scan),
    )


class HOFNO(AbstractOperatorModel):
    """Higher-order Fourier neural operator with explicit polynomial mode mixing.

    ``interaction_order=1`` is the controlled first-order backbone. Orders greater
    than one project the hidden field into distinct factors, multiply those factors
    pointwise, and apply a learned Fourier multiplier. ``aliasing='dealiased'``
    spectrally oversamples the product so unresolved frequencies cannot fold into
    the retained output band.
    """

    operator_architecture = "HOFNO"
    _operator_contract_configuration = staticmethod(_hofno_contract_configuration)

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    n_modes: tuple[int, ...]
    width: int
    lift: Linear
    blocks: tuple[_HigherOrderFNOBlock, ...]
    projection_hidden: Linear
    projection: Linear
    coordinate_embedding: bool
    domain_padding: tuple[float, ...]
    activation: Activation
    source_key: str | None
    scan: bool
    _scan_enabled: bool
    _scan_static: object | None
    interaction_order: int
    factor_bias: bool
    spectral_channel_mixing: SpectralChannelMixing
    aliasing: AliasingPolicy
    ffn_expansion: int
    norm_epsilon: float

    def __init__(
        self,
        *,
        n_modes: Sequence[int],
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        width: int = 32,
        depth: int = 4,
        projection_width: int | None = None,
        interaction_order: int = 2,
        factor_bias: bool = False,
        spectral_channel_mixing: SpectralChannelMixing = "depthwise",
        aliasing: AliasingPolicy = "dealiased",
        ffn_expansion: int = 4,
        norm_epsilon: float = 1e-6,
        coordinate_embedding: bool = False,
        domain_padding: float | Sequence[float] = 0.0,
        activation: Activation = "gelu",
        residual: bool = True,
        dropout: float | Sequence[float] = 0.0,
        source_key: str | None = None,
        scan: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_size = in_channels
        self.out_size = out_channels
        self.n_modes = _mode_tuple(n_modes)
        self.width = int(width)
        self.interaction_order = int(interaction_order)
        self.factor_bias = bool(factor_bias)
        self.spectral_channel_mixing = spectral_channel_mixing
        self.aliasing = aliasing
        self.ffn_expansion = int(ffn_expansion)
        self.norm_epsilon = float(norm_epsilon)
        self.coordinate_embedding = bool(coordinate_embedding)
        self.activation = activation
        self.source_key = source_key
        self.scan = bool(scan)
        self._scan_enabled = False
        self._scan_static = None

        if isinstance(domain_padding, Sequence):
            padding = tuple(
                float(value) for value in cast(Sequence[float], domain_padding)
            )
        else:
            padding = (float(domain_padding),) * len(self.n_modes)
        if len(padding) != len(self.n_modes) or any(value < 0.0 for value in padding):
            raise ValueError(
                "domain_padding must give one non-negative fraction per spatial axis."
            )
        if any(value != 0.0 for value in padding):
            raise ValueError(
                "HOFNO's periodic contract requires domain_padding=0; "
                "spectral de-aliasing is configured separately through aliasing."
            )
        self.domain_padding = padding
        if self.width <= 0 or int(depth) <= 0:
            raise ValueError("width and depth must be positive.")
        if self.interaction_order <= 0:
            raise ValueError("interaction_order must be positive.")
        if self.ffn_expansion <= 0:
            raise ValueError("ffn_expansion must be positive.")
        if self.norm_epsilon <= 0.0:
            raise ValueError("norm_epsilon must be positive.")
        if activation not in ("gelu", "silu", "tanh"):
            raise ValueError("activation must be 'gelu', 'silu', or 'tanh'.")
        if spectral_channel_mixing not in ("depthwise", "dense"):
            raise ValueError("spectral_channel_mixing must be 'depthwise' or 'dense'.")
        if aliasing not in ("collocation", "dealiased"):
            raise ValueError("aliasing must be 'collocation' or 'dealiased'.")

        in_count = _get_size(in_channels)
        out_count = _get_size(out_channels)
        lifted_count = in_count + (len(self.n_modes) if coordinate_embedding else 0)
        keys = jr.split(key, int(depth) + 3)
        self.lift = Linear(
            in_size=lifted_count,
            out_size=self.width,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=keys[0],
        )
        probabilities = _dropout_probabilities(dropout, int(depth))
        self.blocks = tuple(
            _HigherOrderFNOBlock(
                channels=self.width,
                n_modes=self.n_modes,
                interaction_order=self.interaction_order,
                factor_bias=self.factor_bias,
                spectral_channel_mixing=self.spectral_channel_mixing,
                aliasing=self.aliasing,
                activation=self.activation,
                ffn_expansion=self.ffn_expansion,
                norm_epsilon=self.norm_epsilon,
                dropout=probabilities[index],
                residual=residual,
                key=keys[1 + index],
            )
            for index in range(int(depth))
        )
        hidden_width = self.width if projection_width is None else int(projection_width)
        if hidden_width <= 0:
            raise ValueError("projection_width must be positive.")
        self.projection_hidden = Linear(
            in_size=self.width,
            out_size=hidden_width,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=keys[-2],
        )
        self.projection = Linear(
            in_size=hidden_width,
            out_size=out_count,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=keys[-1],
        )
        if self.scan:
            _, static, enabled = pack_scan_modules(self.blocks)
            self._scan_enabled = enabled
            if enabled:
                self._scan_static = static

    def with_active_modes(self, modes: int | Sequence[int], /) -> HOFNO:
        active = _mode_tuple(modes, len(self.n_modes))
        if any(now > maximum for now, maximum in zip(active, self.n_modes, strict=True)):
            raise ValueError("active modes cannot exceed initialized modes.")
        replacements = tuple(
            block.spectral.with_active_modes(active) for block in self.blocks
        )
        return eqx.tree_at(
            lambda model: tuple(block.spectral for block in model.blocks),
            self,
            replacements,
        )

    def _validate_axes(self, axes: tuple[OperatorAxis, ...], /) -> None:
        if len(axes) != len(self.n_modes):
            raise ValueError(
                f"HOFNO expects {len(self.n_modes)} spatial axes, got {len(axes)}."
            )
        for axis in axes:
            if axis.size <= 1:
                raise ValueError(
                    "HOFNO expects coord-separable grid evaluation with at least two "
                    "nodes per spatial axis."
                )
            if not axis.periodic:
                raise ValueError("HOFNO requires periodic tensor-grid axes.")

    def _prepare_values(
        self,
        values: Array,
        axes: tuple[OperatorAxis, ...],
        /,
    ) -> tuple[Array, tuple[int, ...]]:
        array = jnp.asarray(values)
        shape = tuple(axis.size for axis in axes)
        ndim = len(shape)
        explicit_channels = (
            array.ndim > ndim and tuple(array.shape[-ndim - 1 : -1]) == shape
        )
        implicit_scalar = array.ndim >= ndim and tuple(array.shape[-ndim:]) == shape
        expected_channels = _get_size(self.in_size)
        if explicit_channels and int(array.shape[-1]) == expected_channels:
            case_shape = tuple(int(size) for size in array.shape[: -ndim - 1])
        elif implicit_scalar:
            array = array[..., None]
            case_shape = tuple(int(size) for size in array.shape[: -ndim - 1])
        elif explicit_channels:
            case_shape = tuple(int(size) for size in array.shape[: -ndim - 1])
        else:
            raise ValueError(
                "HOFNO values must end in the spatial sample shape, optionally followed "
                f"by channels; got {array.shape} for spatial shape {shape}."
            )
        if int(array.shape[-1]) != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels, got {array.shape[-1]}."
            )
        return array, case_shape

    def _coordinate_features(
        self,
        axes: tuple[OperatorAxis, ...],
        case_shape: tuple[int, ...],
        /,
    ) -> Array:
        normalized = []
        for axis in axes:
            nodes = jnp.asarray(axis.nodes)
            span = nodes[-1] - nodes[0]
            normalized.append(2.0 * (nodes - nodes[0]) / span - 1.0)
        grids = jnp.meshgrid(*normalized, indexing="ij")
        coordinates = jnp.stack(grids, axis=-1)
        return jnp.broadcast_to(coordinates, case_shape + coordinates.shape)

    def _pad(self, values: Array, /) -> tuple[Array, tuple[int, ...]]:
        ndim = len(self.n_modes)
        spatial_shape = tuple(int(size) for size in values.shape[-ndim - 1 : -1])
        pad_counts = tuple(
            int(round(fraction * size))
            for fraction, size in zip(self.domain_padding, spatial_shape, strict=True)
        )
        pad_width = [(0, 0)] * values.ndim
        start = values.ndim - ndim - 1
        for index, count in enumerate(pad_counts):
            pad_width[start + index] = (0, count)
        return jnp.pad(values, pad_width), pad_counts

    def _unpad(self, values: Array, pad_counts: tuple[int, ...], /) -> Array:
        slices: list[slice] = [slice(None)] * values.ndim
        start = values.ndim - len(pad_counts) - 1
        for index, count in enumerate(pad_counts):
            if count:
                slices[start + index] = slice(0, -count)
        return values[tuple(slices)]

    def _prepare_hidden(
        self,
        values: Array,
        axes: tuple[OperatorAxis, ...],
        /,
    ) -> tuple[Array, tuple[int, ...]]:
        self._validate_axes(axes)
        array, case_shape = self._prepare_values(values, axes)
        for axis in axes:
            spacing = jnp.diff(axis.nodes)
            array = eqx.error_if(
                array,
                jnp.logical_not(
                    jnp.allclose(
                        spacing,
                        jnp.mean(spacing),
                        rtol=1e-5,
                        atol=1e-8,
                    )
                ),
                f"HOFNO requires uniformly spaced nodes; axis {axis.name!r} is nonuniform.",
            )
        if self.coordinate_embedding:
            array = jnp.concatenate(
                (array, self._coordinate_features(axes, case_shape)), axis=-1
            )
        hidden = _activation(self.activation, self.lift(array))
        return self._pad(hidden)

    def _execute_hidden(
        self,
        hidden: Array,
        /,
        *,
        key: EvalKey,
    ) -> Array:
        if self._scan_enabled and self._scan_static is not None:
            dynamic = stack_scan_dynamics(self.blocks, self._scan_static)
            if dynamic is not None:
                sites = jnp.arange(len(self.blocks), dtype=jnp.uint32)
                return scan_apply_with_data(
                    dynamic,
                    self._scan_static,
                    hidden,
                    sites,
                    lambda carry, layer, site: layer(
                        carry, key=fold_in_eval_key(key, site)
                    ),
                )
        for site, block in enumerate(self.blocks):
            hidden = block(hidden, key=fold_in_eval_key(key, site))
        return hidden

    def _project_hidden(
        self,
        hidden: Array,
        pad_counts: tuple[int, ...],
        /,
    ) -> Array:
        hidden = self._unpad(hidden, pad_counts)
        output = self.projection(
            _activation(self.activation, self.projection_hidden(hidden))
        )
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def _evaluate(
        self,
        values: Array,
        axes: tuple[OperatorAxis, ...],
        /,
        *,
        key: EvalKey,
    ) -> Array:
        hidden, pad_counts = self._prepare_hidden(values, axes)
        hidden = self._execute_hidden(hidden, key=key)
        return self._project_hidden(hidden, pad_counts)

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if isinstance(x, OperatorBatch):
            return self.__call_operator_batch__(x, key=key)
        if not isinstance(x, tuple) or len(x) != len(self.n_modes) + 1:
            raise ValueError(
                "HOFNO requires (values, axis_0, ..., axis_d) structured input."
            )
        axes = tuple(
            OperatorAxis(f"axis_{index}", jnp.asarray(nodes), periodic=True)
            for index, nodes in enumerate(x[1:])
        )
        return self._evaluate(jnp.asarray(x[0]), axes, key=key)

    def _prepare_operator_batch(
        self,
        batch: OperatorBatch,
        /,
    ) -> tuple[Array, tuple[OperatorAxis, ...], Array | None]:
        if self.source_key is None:
            if len(batch.inputs) != 1:
                raise ValueError(
                    "HOFNO requires source_key when OperatorBatch has multiple sources."
                )
            source = next(iter(batch.inputs.values()))
        else:
            source = batch.input(self.source_key)
        query = batch.require_single_query()
        if not query.axes:
            raise ValueError("HOFNO requires tensor-product query axes.")
        axes = source.axes or query.axes
        if source.axes and source.sample_shape != query.sample_shape:
            raise ValueError(
                "HOFNO currently requires coincident source and query discretizations."
            )
        if source.values is None:
            raise ValueError("HOFNO source values cannot be None.")
        values, case_shape = self._prepare_values(jnp.asarray(source.values), axes)
        if source.mask is not None:
            values = eqx.error_if(
                values,
                jnp.logical_not(jnp.all(source.mask_array(case_shape=case_shape))),
                "HOFNO requires all-valid source masks.",
            )
        query_mask = (
            None if query.mask is None else query.mask_array(case_shape=case_shape)
        )
        if query_mask is not None:
            values = eqx.error_if(
                values,
                jnp.logical_not(jnp.all(query_mask)),
                "HOFNO requires all-valid query masks.",
            )
        return values, axes, query_mask

    def _mask_operator_output(
        self,
        output: Array,
        query_mask: Array | None,
        /,
    ) -> Array:
        if query_mask is None:
            return output
        if self.out_size == "scalar":
            return jnp.where(query_mask, output, 0)
        return jnp.where(query_mask[..., None], output, 0)

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        values, axes, query_mask = self._prepare_operator_batch(batch)
        output = self._evaluate(values, axes, key=key)
        return self._mask_operator_output(output, query_mask)


__all__ = ["HOFNO"]
