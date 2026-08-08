#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from itertools import product
from string import ascii_lowercase
from typing import cast, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, Key

from phydrax._doc import DOC_KEY0
from phydrax._interpolation import fourier_resample as _fourier_resample
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
from phydrax.nn.operator.data import OperatorAxis, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


Factorization = Literal["dense", "cp", "tucker"]
Activation = Literal["gelu", "silu", "tanh"]


def _activation(name: Activation, value: Array, /) -> Array:
    if name == "gelu":
        return jax.nn.gelu(value)
    if name == "silu":
        return jax.nn.silu(value)
    if name == "tanh":
        return jnp.tanh(value)
    raise ValueError("activation must be 'gelu', 'silu', or 'tanh'.")


def _mode_tuple(modes: int | Sequence[int], ndim: int | None = None) -> tuple[int, ...]:
    if isinstance(modes, int):
        if ndim is None:
            return (int(modes),)
        result = (int(modes),) * int(ndim)
    else:
        result = tuple(int(mode) for mode in modes)
        if ndim is not None and len(result) != int(ndim):
            raise ValueError(f"Expected {ndim} mode counts, got {len(result)}.")
    if not result or any(mode <= 0 for mode in result):
        raise ValueError("Every spectral mode count must be positive.")
    return result


def _complex_normal(key: Key[Array, ""], shape: tuple[int, ...], scale: float) -> Array:
    real_key, imag_key = jr.split(key)
    return scale * (
        jr.normal(real_key, shape=shape) + 1j * jr.normal(imag_key, shape=shape)
    )


def _factor_rank(
    rank: int | float,
    in_channels: int,
    out_channels: int,
    modes: tuple[int, ...],
) -> int:
    if isinstance(rank, float):
        if not 0.0 < rank <= 1.0:
            raise ValueError("A floating factorization rank must lie in (0, 1].")
        value = max(1, round(rank * min(in_channels, out_channels, *modes)))
    else:
        value = int(rank)
    if value <= 0:
        raise ValueError("factorization rank must be positive.")
    return max(1, value)


class SpectralConvND(StrictModule):
    """Resolution-independent N-dimensional real Fourier convolution.

    The last array axis is the channel axis. Every spatial axis except the final
    real-FFT axis has independently learned positive and negative low-frequency
    blocks, so two-dimensional and higher-dimensional kernels do not silently omit
    signed modes.
    """

    in_channels: int
    out_channels: int
    n_modes: tuple[int, ...]
    active_modes: tuple[int, ...]
    factorization: Factorization
    rank: int
    weight: Array | None
    factor_in: Array | None
    factor_out: Array | None
    factor_modes: tuple[Array, ...]
    core: Array | None

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        n_modes: int | Sequence[int],
        factorization: Factorization = "dense",
        rank: int | float = 0.5,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = _mode_tuple(n_modes)
        self.active_modes = self.n_modes
        self.factorization = factorization
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if factorization not in ("dense", "cp", "tucker"):
            raise ValueError("factorization must be 'dense', 'cp', or 'tucker'.")

        corners = 2 ** max(0, len(self.n_modes) - 1)
        self.rank = _factor_rank(rank, self.in_channels, self.out_channels, self.n_modes)
        self.weight = None
        self.factor_in = None
        self.factor_out = None
        self.factor_modes = ()
        self.core = None
        scale = 1.0 / float(self.in_channels * self.out_channels)

        if factorization == "dense":
            self.weight = _complex_normal(
                key,
                (
                    corners,
                    self.in_channels,
                    self.out_channels,
                    *self.n_modes,
                ),
                scale,
            )
        elif factorization == "cp":
            keys = jr.split(key, len(self.n_modes) + 2)
            factor_scale = scale ** (1.0 / float(len(self.n_modes) + 2))
            self.factor_in = _complex_normal(
                keys[0], (corners, self.in_channels, self.rank), factor_scale
            )
            self.factor_out = _complex_normal(
                keys[1], (corners, self.out_channels, self.rank), factor_scale
            )
            self.factor_modes = tuple(
                _complex_normal(
                    mode_key,
                    (corners, mode, self.rank),
                    factor_scale,
                )
                for mode_key, mode in zip(keys[2:], self.n_modes, strict=True)
            )
        else:
            keys = jr.split(key, 3)
            self.factor_in = _complex_normal(keys[0], (self.in_channels, self.rank), 1.0)
            self.factor_out = _complex_normal(
                keys[1], (self.out_channels, self.rank), 1.0
            )
            self.core = _complex_normal(
                keys[2],
                (corners, self.rank, self.rank, *self.n_modes),
                scale,
            )

    def with_active_modes(self, modes: int | Sequence[int], /) -> SpectralConvND:
        """Return a copy using a prefix of the initialized spectral modes."""
        active = _mode_tuple(modes, len(self.n_modes))
        if any(now > maximum for now, maximum in zip(active, self.n_modes, strict=True)):
            raise ValueError("active modes cannot exceed initialized modes.")
        return eqx.tree_at(lambda layer: layer.active_modes, self, active)

    def _dense_weight(self, corner: int, modes: tuple[int, ...], /) -> Array:
        mode_slices = tuple(slice(0, mode) for mode in modes)
        if self.factorization == "dense":
            assert self.weight is not None
            return self.weight[(corner, slice(None), slice(None), *mode_slices)]

        if self.factorization == "cp":
            assert self.factor_in is not None and self.factor_out is not None
            in_factor = self.factor_in[corner]
            out_factor = self.factor_out[corner]
            weight = in_factor.reshape(
                (self.in_channels, 1, *(1,) * len(modes), self.rank)
            )
            weight = weight * out_factor.reshape(
                (1, self.out_channels, *(1,) * len(modes), self.rank)
            )
            for index, (factor, mode) in enumerate(
                zip(self.factor_modes, modes, strict=True)
            ):
                factor_shape = (
                    1,
                    1,
                    *(1,) * index,
                    mode,
                    *(1,) * (len(modes) - index - 1),
                    self.rank,
                )
                weight = weight * factor[corner, :mode].reshape(factor_shape)
            return jnp.sum(weight, axis=-1)

        assert self.factor_in is not None
        assert self.factor_out is not None
        assert self.core is not None
        core = self.core[(corner, slice(None), slice(None), *mode_slices)]
        letters = ascii_lowercase[: len(modes)]
        return oe.contract(
            f"ir,os,rs{letters}->io{letters}",
            self.factor_in,
            self.factor_out,
            core,
        )

    def __call__(self, x: Array, /) -> Array:
        values = jnp.asarray(x)
        ndim = len(self.n_modes)
        if values.ndim < ndim + 1:
            raise ValueError(
                f"SpectralConvND expects at least {ndim + 1} axes; got {values.ndim}."
            )
        if int(values.shape[-1]) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {values.shape[-1]}."
            )

        spatial_axes = tuple(range(values.ndim - ndim - 1, values.ndim - 1))
        spatial_shape = tuple(int(values.shape[axis]) for axis in spatial_axes)
        transformed = jnp.fft.rfftn(values, axes=spatial_axes, norm="ortho")
        output_ft = jnp.zeros(
            (*transformed.shape[:-1], self.out_channels), dtype=transformed.dtype
        )

        usable_modes = []
        for index, (requested, size) in enumerate(
            zip(self.active_modes, spatial_shape, strict=True)
        ):
            available = size // 2 + 1 if index == ndim - 1 else max(1, size // 2)
            usable_modes.append(min(int(requested), available))
        modes = tuple(usable_modes)
        letters = ascii_lowercase[:ndim]

        signs = tuple(product((0, 1), repeat=max(0, ndim - 1)))
        for corner, corner_signs in enumerate(signs):
            slices: list[slice] = []
            for sign, mode in zip(corner_signs, modes[:-1], strict=True):
                slices.append(slice(0, mode) if sign == 0 else slice(-mode, None))
            slices.append(slice(0, modes[-1]))
            spatial_slices = tuple(slices)
            block = transformed[(..., *spatial_slices, slice(None))]
            weight = self._dense_weight(corner, modes)
            result = oe.contract(
                f"...{letters}i,io{letters}->...{letters}o",
                block,
                weight,
            )
            output_ft = output_ft.at[(..., *spatial_slices, slice(None))].set(result)

        return jnp.fft.irfftn(
            output_ft,
            s=spatial_shape,
            axes=spatial_axes,
            norm="ortho",
        )


def spectral_resample(
    values: Array,
    output_shape: Sequence[int],
    /,
    *,
    phase_offsets: Sequence[ArrayLike] | None = None,
) -> Array:
    """Band-limited resampling over spatial axes, with optional period shifts."""
    return _fourier_resample(
        values,
        output_shape,
        phase_offsets=phase_offsets,
    )


class MultiScaleSpectralConvND(StrictModule):
    """Parallel Fourier kernels evaluated at multiple band-limited resolutions."""

    in_channels: int
    out_channels: int
    n_modes: tuple[int, ...]
    scales: tuple[float, ...]
    branches: tuple[SpectralConvND, ...]
    branch_gain: Array

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        n_modes: int | Sequence[int],
        scales: Sequence[float] = (1.0, 0.5),
        factorization: Factorization = "dense",
        rank: int | float = 0.5,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = _mode_tuple(n_modes)
        self.scales = tuple(float(scale) for scale in scales)
        if not self.scales or any(scale <= 0.0 or scale > 1.0 for scale in self.scales):
            raise ValueError("scales must contain values in (0, 1].")
        keys = jr.split(key, len(self.scales))
        self.branches = tuple(
            SpectralConvND(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                n_modes=self.n_modes,
                factorization=factorization,
                rank=rank,
                key=branch_key,
            )
            for branch_key in keys
        )
        self.branch_gain = jnp.full(
            (len(self.scales),),
            1.0 / jnp.sqrt(float(len(self.scales))),
            dtype=float,
        )

    def __call__(self, values: Array, /) -> Array:
        array = jnp.asarray(values)
        ndim = len(self.n_modes)
        if array.ndim < ndim + 1:
            raise ValueError("MultiScaleSpectralConvND input rank is too small.")
        source_shape = tuple(int(size) for size in array.shape[-ndim - 1 : -1])
        output = jnp.zeros(
            (*array.shape[:-1], self.out_channels),
            dtype=jnp.result_type(array.dtype, float),
        )
        for gain, scale, branch in zip(
            self.branch_gain, self.scales, self.branches, strict=True
        ):
            branch_shape = tuple(max(2, round(scale * size)) for size in source_shape)
            branch_input = (
                array
                if branch_shape == source_shape
                else spectral_resample(array, branch_shape)
            )
            branch_output = branch(branch_input)
            if branch_shape != source_shape:
                branch_output = spectral_resample(branch_output, source_shape)
            output = output + gain * branch_output
        return output


class _ChannelNorm(StrictModule):
    scale: Array
    bias: Array
    eps: float

    def __init__(self, channels: int, *, eps: float = 1e-5):
        self.scale = jnp.ones((int(channels),), dtype=float)
        self.bias = jnp.zeros((int(channels),), dtype=float)
        self.eps = float(eps)

    def __call__(self, values: Array, /) -> Array:
        mean = jnp.mean(values, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.abs(values - mean) ** 2, axis=-1, keepdims=True)
        normalized = (values - mean) * jax.lax.rsqrt(variance + self.eps)
        return normalized * self.scale + self.bias


class _AxialSpectralConvND(StrictModule):
    """A separable Fourier convolution composed along one axis at a time."""

    in_channels: int
    out_channels: int
    n_modes: tuple[int, ...]
    active_modes: tuple[int, ...]
    factorization: Factorization
    rank: int
    axis_layers: tuple[SpectralConvND, ...]

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        n_modes: int | Sequence[int],
        factorization: Factorization = "dense",
        rank: int | float = 0.5,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = _mode_tuple(n_modes)
        self.active_modes = self.n_modes
        self.factorization = factorization
        self.rank = _factor_rank(
            rank,
            self.in_channels,
            self.out_channels,
            self.n_modes,
        )
        keys = jr.split(key, len(self.n_modes))
        layers = []
        current_channels = self.in_channels
        for mode, axis_key in zip(self.n_modes, keys, strict=True):
            layers.append(
                SpectralConvND(
                    in_channels=current_channels,
                    out_channels=self.out_channels,
                    n_modes=(mode,),
                    factorization=factorization,
                    rank=rank,
                    key=axis_key,
                )
            )
            current_channels = self.out_channels
        self.axis_layers = tuple(layers)

    def with_active_modes(
        self,
        modes: int | Sequence[int],
        /,
    ) -> _AxialSpectralConvND:
        active = _mode_tuple(modes, len(self.n_modes))
        if any(now > maximum for now, maximum in zip(active, self.n_modes, strict=True)):
            raise ValueError("active modes cannot exceed initialized modes.")
        updated = eqx.tree_at(lambda layer: layer.active_modes, self, active)
        replacements = tuple(
            layer.with_active_modes((mode,))
            for layer, mode in zip(updated.axis_layers, active, strict=True)
        )
        return eqx.tree_at(
            lambda layer: layer.axis_layers,
            updated,
            replacements,
        )

    def __call__(self, values: Array, /) -> Array:
        hidden = jnp.asarray(values)
        ndim = len(self.n_modes)
        if hidden.ndim < ndim + 1:
            raise ValueError(
                f"Axial Fourier convolution expects at least {ndim + 1} axes; "
                f"got {hidden.ndim}."
            )
        if int(hidden.shape[-1]) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {hidden.shape[-1]}."
            )
        spatial_start = hidden.ndim - ndim - 1
        for axis, layer in enumerate(self.axis_layers):
            target = spatial_start + axis
            hidden = jnp.moveaxis(hidden, target, -2)
            hidden = layer(hidden)
            hidden = jnp.moveaxis(hidden, -2, target)
        return hidden


class _FNOResidualStep(StrictModule):
    spectral: SpectralConvND | _AxialSpectralConvND
    pointwise: Linear
    normalization: _ChannelNorm
    dropout: Dropout
    activation: Activation
    residual: bool

    def __init__(
        self,
        spectral: SpectralConvND | _AxialSpectralConvND,
        pointwise: Linear,
        normalization: _ChannelNorm,
        dropout: Dropout,
        *,
        activation: Activation,
        residual: bool,
    ):
        self.spectral = spectral
        self.pointwise = pointwise
        self.normalization = normalization
        self.dropout = dropout
        self.activation = activation
        self.residual = bool(residual)

    def __call__(self, x: Array, /, *, key: EvalKey = None) -> Array:
        hidden = self.normalization(self.spectral(x) + self.pointwise(x))
        hidden = self.dropout(_activation(self.activation, hidden), key=key)
        return x + hidden if self.residual else hidden


def _fno_contract_configuration(model):
    return (
        ("n_modes", model.n_modes),
        ("width", model.width),
        ("factorization", model.factorization),
        ("rank", model.factorization_rank),
        ("axial", model.axial_factorization),
        ("coordinate_embedding", model.coordinate_embedding),
        ("domain_padding", model.domain_padding),
        ("source_key", model.source_key),
        ("scan", model.scan),
    )


class _AbstractFNO(AbstractOperatorModel):
    _operator_contract_configuration = staticmethod(_fno_contract_configuration)
    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    n_modes: tuple[int, ...]
    width: int
    lift: Linear
    blocks: tuple[_FNOResidualStep, ...]
    projection_hidden: Linear
    projection: Linear
    coordinate_embedding: bool
    domain_padding: tuple[float, ...]
    activation: Activation
    source_key: str | None
    scan: bool
    _scan_enabled: bool
    _scan_static: object | None
    factorization: Factorization
    factorization_rank: int | float
    axial_factorization: bool

    def _init_fno(
        self,
        *,
        in_channels: int | Literal["scalar"],
        out_channels: int | Literal["scalar"],
        n_modes: int | Sequence[int],
        width: int,
        depth: int,
        projection_width: int | None,
        coordinate_embedding: bool,
        domain_padding: float | Sequence[float],
        activation: Activation,
        residual: bool,
        factorization: Factorization,
        rank: int | float,
        dropout: float | Sequence[float],
        source_key: str | None,
        scan: bool,
        key: Key[Array, ""],
        axial: bool = False,
    ) -> None:
        self.in_size = in_channels
        self.out_size = out_channels
        self.n_modes = _mode_tuple(n_modes)
        self.width = int(width)
        self.coordinate_embedding = bool(coordinate_embedding)
        self.activation = activation
        self.source_key = source_key
        self.scan = bool(scan)
        self._scan_enabled = False
        self._scan_static = None
        self.factorization = factorization
        self.factorization_rank = rank
        self.axial_factorization = bool(axial)

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
        self.domain_padding = padding
        if self.width <= 0 or int(depth) <= 0:
            raise ValueError("width and depth must be positive.")
        if activation not in ("gelu", "silu", "tanh"):
            raise ValueError("activation must be 'gelu', 'silu', or 'tanh'.")

        in_count = _get_size(in_channels)
        out_count = _get_size(out_channels)
        lifted_count = in_count + (len(self.n_modes) if coordinate_embedding else 0)
        keys = jr.split(key, int(depth) * 2 + 4)
        self.lift = Linear(
            in_size=lifted_count,
            out_size=self.width,
            activation=None,
            key=keys[0],
        )
        probabilities = _dropout_probabilities(dropout, int(depth))
        blocks = []
        for index in range(int(depth)):
            spectral_type = _AxialSpectralConvND if axial else SpectralConvND
            spectral = spectral_type(
                in_channels=self.width,
                out_channels=self.width,
                n_modes=self.n_modes,
                factorization=factorization,
                rank=rank,
                key=keys[1 + 2 * index],
            )
            pointwise = Linear(
                in_size=self.width,
                out_size=self.width,
                activation=None,
                key=keys[2 + 2 * index],
            )
            blocks.append(
                _FNOResidualStep(
                    spectral,
                    pointwise,
                    _ChannelNorm(self.width),
                    Dropout(self.width, p=probabilities[index], mode="feature"),
                    activation=activation,
                    residual=residual,
                )
            )
        self.blocks = tuple(blocks)
        hidden_width = self.width if projection_width is None else int(projection_width)
        if hidden_width <= 0:
            raise ValueError("projection_width must be positive.")
        self.projection_hidden = Linear(
            in_size=self.width,
            out_size=hidden_width,
            activation=None,
            key=keys[-2],
        )
        self.projection = Linear(
            in_size=hidden_width,
            out_size=out_count,
            activation=None,
            key=keys[-1],
        )

        if self.scan:
            _, static, enabled = pack_scan_modules(self.blocks)
            self._scan_enabled = enabled
            if enabled:
                self._scan_static = static

    def with_active_modes(self, modes: int | Sequence[int], /) -> _AbstractFNO:
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
                f"FNO expects {len(self.n_modes)} spatial axes, got {len(axes)}."
            )
        for axis in axes:
            if axis.size <= 1:
                raise ValueError(
                    "FNO expects coord-separable grid evaluation with at least two "
                    "nodes per spatial axis."
                )

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
                "FNO values must end in the spatial sample shape, optionally followed "
                f"by channels; got {array.shape} for spatial shape {shape}."
            )
        if int(array.shape[-1]) != _get_size(self.in_size):
            raise ValueError(
                f"Expected {_get_size(self.in_size)} input channels, got {array.shape[-1]}."
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
                f"FNO requires uniformly spaced nodes; axis {axis.name!r} is nonuniform.",
            )
        if self.coordinate_embedding:
            array = jnp.concatenate(
                (array, self._coordinate_features(axes, case_shape)), axis=-1
            )
        hidden = _activation(self.activation, self.lift(array))
        return self._pad(hidden)

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

    @abstractmethod
    def _execute_hidden(
        self,
        hidden: Array,
        /,
        *,
        key: EvalKey,
    ) -> Array:
        raise NotImplementedError

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
                "FNO requires (values, axis_0, ..., axis_d) structured input."
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
                    "FNO requires source_key when OperatorBatch has multiple sources."
                )
            source = next(iter(batch.inputs.values()))
        else:
            source = batch.input(self.source_key)
        if not batch.require_single_query().axes:
            raise ValueError("FNO requires tensor-product query axes.")
        axes = source.axes or batch.require_single_query().axes
        if (
            source.axes
            and source.sample_shape != batch.require_single_query().sample_shape
        ):
            raise ValueError(
                "FNO currently requires coincident source and query discretizations."
            )
        if source.values is None:
            raise ValueError("FNO source values cannot be None.")
        values, case_shape = self._prepare_values(jnp.asarray(source.values), axes)
        if source.mask is not None:
            source_mask = source.mask_array(case_shape=case_shape)
            values = jnp.where(source_mask[..., None], values, 0)
        query_mask = (
            None
            if batch.require_single_query().mask is None
            else batch.require_single_query().mask_array(case_shape=case_shape)
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


def _execute_explicit_fno(
    model: _AbstractFNO,
    hidden: Array,
    /,
    *,
    key: EvalKey,
) -> Array:
    if model._scan_enabled and model._scan_static is not None:
        dynamic = stack_scan_dynamics(model.blocks)
        if dynamic is not None:
            sites = jnp.arange(len(model.blocks), dtype=jnp.uint32)
            return scan_apply_with_data(
                dynamic,
                model._scan_static,
                hidden,
                sites,
                lambda carry, layer, site: layer(carry, key=fold_in_eval_key(key, site)),
            )
    for site, block in enumerate(model.blocks):
        hidden = block(hidden, key=fold_in_eval_key(key, site))
    return hidden


class FNO(_AbstractFNO):
    """Configurable N-dimensional Fourier neural operator."""

    operator_architecture = "FNO"

    def __init__(
        self,
        *,
        n_modes: Sequence[int],
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        width: int = 32,
        depth: int = 4,
        projection_width: int | None = None,
        coordinate_embedding: bool = True,
        domain_padding: float | Sequence[float] = 0.0,
        activation: Activation = "gelu",
        residual: bool = True,
        factorization: Factorization = "dense",
        rank: int | float = 0.5,
        dropout: float | Sequence[float] = 0.0,
        source_key: str | None = None,
        scan: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self._init_fno(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            width=width,
            depth=depth,
            projection_width=projection_width,
            coordinate_embedding=coordinate_embedding,
            domain_padding=domain_padding,
            activation=activation,
            residual=residual,
            factorization=factorization,
            rank=rank,
            dropout=dropout,
            source_key=source_key,
            scan=scan,
            key=key,
        )

    def _execute_hidden(
        self,
        hidden: Array,
        /,
        *,
        key: EvalKey,
    ) -> Array:
        return _execute_explicit_fno(self, hidden, key=key)


class IFNOConvergence(StrictModule):
    """Per-case convergence information from a statically bounded IFNO solve."""

    absolute_residual: Array
    relative_residual: Array
    converged: Array
    iterations: int


class IFNO(_AbstractFNO):
    """Implicit FNO with one shared Fourier update iterated to a fixed point.

    The iteration count is static, so compiled execution always has a fixed
    control-flow shape. :meth:`evaluate_with_diagnostics` reports the RMS change
    from the final iteration, normalized by the RMS magnitude of its result.
    """

    operator_architecture = "IFNO"

    iterations: int
    tolerance: float

    def __init__(
        self,
        *,
        n_modes: int | Sequence[int],
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        width: int = 32,
        iterations: int = 8,
        tolerance: float = 1e-5,
        projection_width: int | None = None,
        coordinate_embedding: bool = True,
        domain_padding: float | Sequence[float] = 0.0,
        activation: Activation = "gelu",
        residual: bool = True,
        factorization: Factorization = "dense",
        rank: int | float = 0.5,
        dropout: float = 0.0,
        source_key: str | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.iterations = int(iterations)
        self.tolerance = float(tolerance)
        if self.iterations <= 0:
            raise ValueError("iterations must be positive.")
        if not self.tolerance > 0.0:
            raise ValueError("tolerance must be positive.")
        self._init_fno(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            width=width,
            depth=1,
            projection_width=projection_width,
            coordinate_embedding=coordinate_embedding,
            domain_padding=domain_padding,
            activation=activation,
            residual=residual,
            factorization=factorization,
            rank=rank,
            dropout=dropout,
            source_key=source_key,
            scan=False,
            key=key,
        )

    def _fixed_point_iteration(
        self,
        hidden: Array,
        /,
        *,
        key: EvalKey,
    ) -> tuple[Array, Array, Array]:
        spatial_ndim = len(self.n_modes)
        reduction_axes = tuple(range(hidden.ndim - spatial_ndim - 1, hidden.ndim))

        def step(
            carry: tuple[Array, Array, Array],
            site: Array,
        ) -> tuple[tuple[Array, Array, Array], None]:
            state, _, _ = carry
            updated = self.blocks[0](
                state,
                key=fold_in_eval_key(key, site),
            )
            difference = updated - state
            absolute = jnp.sqrt(
                jnp.mean(jnp.square(jnp.abs(difference)), axis=reduction_axes)
            )
            magnitude = jnp.sqrt(
                jnp.mean(jnp.square(jnp.abs(updated)), axis=reduction_axes)
            )
            epsilon = jnp.finfo(updated.dtype).eps
            relative = absolute / jnp.maximum(magnitude, epsilon)
            return (updated, absolute, relative), None

        diagnostic_shape = hidden.shape[: hidden.ndim - spatial_ndim - 1]
        initial_residual = jnp.zeros(diagnostic_shape, dtype=hidden.dtype)
        sites = jnp.arange(self.iterations, dtype=jnp.uint32)
        (hidden, absolute, relative), _ = jax.lax.scan(
            step,
            (hidden, initial_residual, initial_residual),
            sites,
        )
        return hidden, absolute, relative

    def _execute_hidden(
        self,
        hidden: Array,
        /,
        *,
        key: EvalKey,
    ) -> Array:
        hidden, _, _ = self._fixed_point_iteration(hidden, key=key)
        return hidden

    def _evaluate_with_diagnostics(
        self,
        values: Array,
        axes: tuple[OperatorAxis, ...],
        /,
        *,
        key: EvalKey,
    ) -> tuple[Array, IFNOConvergence]:
        hidden, pad_counts = self._prepare_hidden(values, axes)
        hidden, absolute, relative = self._fixed_point_iteration(hidden, key=key)
        output = self._project_hidden(hidden, pad_counts)
        diagnostic = IFNOConvergence(
            absolute_residual=absolute,
            relative_residual=relative,
            converged=relative <= self.tolerance,
            iterations=self.iterations,
        )
        return output, diagnostic

    def evaluate_with_diagnostics(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> tuple[Array, IFNOConvergence]:
        """Evaluate and return final per-case fixed-point residual diagnostics."""
        if isinstance(x, OperatorBatch):
            values, axes, query_mask = self._prepare_operator_batch(x)
            output, diagnostic = self._evaluate_with_diagnostics(values, axes, key=key)
            return self._mask_operator_output(output, query_mask), diagnostic
        if not isinstance(x, tuple) or len(x) != len(self.n_modes) + 1:
            raise ValueError(
                "IFNO requires (values, axis_0, ..., axis_d) structured input."
            )
        axes = tuple(
            OperatorAxis(f"axis_{index}", jnp.asarray(nodes), periodic=True)
            for index, nodes in enumerate(x[1:])
        )
        return self._evaluate_with_diagnostics(jnp.asarray(x[0]), axes, key=key)


class AxialFactorizedFNO(_AbstractFNO):
    """FNO whose spectral blocks apply learned one-axis transforms sequentially."""

    operator_architecture = "AxialFactorizedFNO"

    def __init__(
        self,
        *,
        n_modes: Sequence[int],
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        width: int = 32,
        depth: int = 4,
        projection_width: int | None = None,
        coordinate_embedding: bool = True,
        domain_padding: float | Sequence[float] = 0.0,
        activation: Activation = "gelu",
        residual: bool = True,
        factorization: Factorization = "dense",
        rank: int | float = 0.5,
        dropout: float | Sequence[float] = 0.0,
        source_key: str | None = None,
        scan: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self._init_fno(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            width=width,
            depth=depth,
            projection_width=projection_width,
            coordinate_embedding=coordinate_embedding,
            domain_padding=domain_padding,
            activation=activation,
            residual=residual,
            factorization=factorization,
            rank=rank,
            dropout=dropout,
            source_key=source_key,
            scan=scan,
            key=key,
            axial=True,
        )

    def _execute_hidden(
        self,
        hidden: Array,
        /,
        *,
        key: EvalKey,
    ) -> Array:
        return _execute_explicit_fno(self, hidden, key=key)


def SpectralConv1d(
    *,
    in_channels: int,
    out_channels: int,
    modes: int,
    key: Key[Array, ""] = DOC_KEY0,
) -> SpectralConvND:
    return SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=(modes,),
        key=key,
    )


def SpectralConv2d(
    *,
    in_channels: int,
    out_channels: int,
    modes_x: int,
    modes_y: int,
    key: Key[Array, ""] = DOC_KEY0,
) -> SpectralConvND:
    return SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=(modes_x, modes_y),
        key=key,
    )


__all__ = [
    "AxialFactorizedFNO",
    "FNO",
    "IFNO",
    "IFNOConvergence",
    "MultiScaleSpectralConvND",
    "SpectralConv1d",
    "SpectralConv2d",
    "SpectralConvND",
    "spectral_resample",
]
