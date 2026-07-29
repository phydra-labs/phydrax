#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod, sqrt
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ...._strict import StrictModule
from ..._utils import _get_size
from ..core._base import _AbstractOperatorModel
from ..core._keys import EvalKey, split_eval_key
from ..core._operator import FunctionSamples, OperatorAxis, OperatorBatch
from ..layers._linear import Linear
from ..layers._probabilistic_warp import ProbabilisticMultiheadWarp
from ..layers._warp import (
    _boundary_modes,
    MultiheadWarp,
    WarpBoundaryMode,
)
from ..layers._warp_geometry import (
    normalized_lattice_from_nodes,
    RectilinearWarpDiagnostics,
    sample_rectilinear_grid,
    WarpMaskMode,
)


FlowerTransitionMode = Literal["learned", "resolution_consistent"]
FlowerQueryMode = Literal["coincident", "interpolate"]


class _ChannelLastGroupNorm(StrictModule):
    scale: Array
    bias: Array
    channels: int
    groups: int
    spatial_ndim: int
    eps: float

    def __init__(
        self,
        channels: int,
        groups: int,
        spatial_ndim: int,
        /,
        *,
        eps: float = 1e-5,
    ):
        self.channels = int(channels)
        self.groups = int(groups)
        self.spatial_ndim = int(spatial_ndim)
        self.eps = float(eps)
        if self.channels <= 0 or self.groups <= 0:
            raise ValueError("GroupNorm channels and groups must be positive.")
        if self.channels % self.groups != 0:
            raise ValueError("GroupNorm channels must be divisible by groups.")
        self.scale = jnp.ones((self.channels,), dtype=float)
        self.bias = jnp.zeros((self.channels,), dtype=float)

    def __call__(
        self,
        values: Array,
        /,
        *,
        modulation: tuple[Array, Array] | None = None,
        mask: Array | None = None,
    ) -> Array:
        array = jnp.asarray(values)
        if array.ndim < self.spatial_ndim + 1 or int(array.shape[-1]) != self.channels:
            raise ValueError(
                "GroupNorm input must end in the configured spatial dimensions "
                f"and {self.channels} channels; got {array.shape}."
            )
        case_ndim = array.ndim - self.spatial_ndim - 1
        case_shape = tuple(int(size) for size in array.shape[:case_ndim])
        spatial_shape = tuple(
            int(size)
            for size in array.shape[case_ndim : case_ndim + self.spatial_ndim]
        )
        grouped = array.reshape(
            array.shape[:-1] + (self.groups, self.channels // self.groups)
        )
        reduction_axes = tuple(
            range(case_ndim, case_ndim + self.spatial_ndim)
        ) + (grouped.ndim - 1,)
        if mask is None:
            mean = jnp.mean(grouped, axis=reduction_axes, keepdims=True)
            variance = jnp.var(grouped, axis=reduction_axes, keepdims=True)
            valid_mask = None
        else:
            valid_mask = jnp.asarray(mask, dtype=bool)
            expected_mask = case_shape + spatial_shape
            if valid_mask.shape != expected_mask:
                raise ValueError(
                    f"GroupNorm mask must have shape {expected_mask}; "
                    f"got {valid_mask.shape}."
                )
            grouped_mask = valid_mask[..., None, None]
            count = (
                jnp.sum(grouped_mask, axis=reduction_axes, keepdims=True)
                * (self.channels // self.groups)
            )
            array = eqx.error_if(
                array,
                jnp.any(count <= 0),
                "Every masked GroupNorm case must contain a valid sample.",
            )
            count = jnp.maximum(count, 1)
            mean = jnp.sum(
                grouped * grouped_mask,
                axis=reduction_axes,
                keepdims=True,
            ) / count
            variance = jnp.sum(
                (grouped - mean) ** 2 * grouped_mask,
                axis=reduction_axes,
                keepdims=True,
            ) / count
        normalized = ((grouped - mean) * jax.lax.rsqrt(variance + self.eps)).reshape(
            array.shape
        )
        affine_shape = (1,) * (array.ndim - 1) + (self.channels,)
        output = normalized * self.scale.reshape(affine_shape)
        output = output + self.bias.reshape(affine_shape)
        if modulation is not None:
            modulation_scale, modulation_shift = modulation
            expected = case_shape + (self.channels,)
            if (
                modulation_scale.shape != expected
                or modulation_shift.shape != expected
            ):
                raise ValueError(
                    f"FiLM modulation must have shape {expected}; got "
                    f"{modulation_scale.shape} and {modulation_shift.shape}."
                )
            broadcast = case_shape + (1,) * self.spatial_ndim + (self.channels,)
            output = output * (1.0 + modulation_scale.reshape(broadcast)) + (
                modulation_shift.reshape(broadcast)
            )
        if valid_mask is not None:
            output = output * valid_mask[..., None].astype(output.dtype)
        return output


class _FlowerBlock(StrictModule):
    warp: MultiheadWarp | ProbabilisticMultiheadWarp
    identity_projection: Linear
    normalization: _ChannelLastGroupNorm
    modulation: Linear | None

    def __init__(
        self,
        *,
        spatial_ndim: int,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        groups: int,
        boundary: tuple[WarpBoundaryMode, ...],
        fill_value: float,
        conditioning_width: int,
        mask_mode: WarpMaskMode,
        probabilistic_routing: bool,
        minimum_route_scale: float,
        route_scale_factor: float,
        key: Key[Array, ""],
    ):
        warp_key, identity_key, modulation_key = jr.split(key, 3)
        warp_kwargs = dict(
            spatial_ndim=spatial_ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            boundary=boundary,
            conditioning_size=conditioning_width,
            mask_mode=mask_mode,
            displacement_width=out_channels,
            fill_value=fill_value,
            key=warp_key,
        )
        self.warp = (
            ProbabilisticMultiheadWarp(
                **warp_kwargs,
                minimum_scale=minimum_route_scale,
                scale_factor=route_scale_factor,
            )
            if probabilistic_routing
            else MultiheadWarp(**warp_kwargs)
        )
        self.identity_projection = Linear(
            in_size=in_channels,
            out_size=out_channels,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=identity_key,
        )
        self.normalization = _ChannelLastGroupNorm(
            out_channels,
            groups,
            spatial_ndim,
        )
        self.modulation = (
            None
            if conditioning_width == 0
            else Linear(
                in_size=conditioning_width,
                out_size=2 * out_channels,
                activation=None,
                rwf=False,
                bias_init_lim=0.0,
                key=modulation_key,
            )
        )

    def diagnostics(
        self,
        values: Array,
        condition: Array | None,
        /,
        *,
        axis_nodes: Sequence[Array] | None,
        source_mask: Array | None,
        key: EvalKey = None,
    ) -> RectilinearWarpDiagnostics:
        return self.warp.diagnostics(
            values,
            condition=condition,
            axis_nodes=axis_nodes,
            source_mask=source_mask,
            key=key,
        )

    def __call__(
        self,
        values: Array,
        condition: Array | None,
        /,
        *,
        axis_nodes: Sequence[Array] | None = None,
        source_mask: Array | None = None,
        key: EvalKey = None,
    ) -> Array:
        modulation = None
        if self.modulation is not None:
            if condition is None:
                raise ValueError("This Flower block requires conditioning values.")
            scale_shift = self.modulation(condition)
            modulation_scale, modulation_shift = jnp.split(scale_shift, 2, axis=-1)
            modulation = (modulation_scale, modulation_shift)
        hidden = self.warp(
            values,
            condition=condition,
            axis_nodes=axis_nodes,
            source_mask=source_mask,
            key=key,
        ) + self.identity_projection(values)
        return jax.nn.gelu(
            self.normalization(
                hidden,
                modulation=modulation,
                mask=source_mask,
            )
        )


class _StrideTwoConvND(StrictModule):
    weight: Array
    bias: Array
    in_channels: int
    out_channels: int
    spatial_ndim: int
    transpose: bool

    def __init__(
        self,
        *,
        spatial_ndim: int,
        in_channels: int,
        out_channels: int,
        transpose: bool,
        key: Key[Array, ""],
    ):
        self.spatial_ndim = int(spatial_ndim)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.transpose = bool(transpose)
        kernel_shape = (2,) * self.spatial_ndim
        scale = 1.0 / sqrt(float(prod(kernel_shape) * self.in_channels))
        self.weight = scale * jr.normal(
            key,
            kernel_shape + (self.in_channels, self.out_channels),
        )
        self.bias = jnp.zeros((self.out_channels,), dtype=float)

    def __call__(self, values: Array, /) -> Array:
        array = jnp.asarray(values)
        if array.ndim < self.spatial_ndim + 1 or int(array.shape[-1]) != self.in_channels:
            raise ValueError(
                "Stride-two convolution input does not match its configured rank "
                f"and channels; got {array.shape}."
            )
        spatial_shape = tuple(
            int(size) for size in array.shape[-self.spatial_ndim - 1 : -1]
        )
        case_shape = tuple(
            int(size) for size in array.shape[: -self.spatial_ndim - 1]
        )
        batch_count = prod(case_shape) if case_shape else 1
        batched = array.reshape((batch_count,) + spatial_shape + (self.in_channels,))
        spatial = {1: "W", 2: "HW", 3: "DHW"}[self.spatial_ndim]
        dimension_numbers = (
            f"N{spatial}C",
            f"{spatial}IO",
            f"N{spatial}C",
        )
        if self.transpose:
            output = jax.lax.conv_transpose(
                batched,
                self.weight,
                strides=(2,) * self.spatial_ndim,
                padding="VALID",
                dimension_numbers=dimension_numbers,
            )
        else:
            output = jax.lax.conv_general_dilated(
                batched,
                self.weight,
                window_strides=(2,) * self.spatial_ndim,
                padding="VALID",
                dimension_numbers=dimension_numbers,
            )
        output = output + self.bias
        return output.reshape(case_shape + output.shape[1:])


class _ResolutionConsistentTransitionND(StrictModule):
    """Measure-aware restriction or interpolation followed by channel mixing."""

    projection: Linear
    spatial_ndim: int
    in_channels: int
    out_channels: int
    transpose: bool
    boundary: tuple[WarpBoundaryMode, ...]
    mask_mode: WarpMaskMode

    def __init__(
        self,
        *,
        spatial_ndim: int,
        in_channels: int,
        out_channels: int,
        transpose: bool,
        boundary: tuple[WarpBoundaryMode, ...],
        mask_mode: WarpMaskMode,
        key: Key[Array, ""],
    ):
        self.spatial_ndim = int(spatial_ndim)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.transpose = bool(transpose)
        self.boundary = boundary
        self.mask_mode = mask_mode
        self.projection = Linear(
            in_size=self.in_channels,
            out_size=self.out_channels,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=key,
        )

    def _downsample(
        self,
        values: Array,
        axis_nodes: tuple[Array, ...],
        axis_weights: tuple[Array, ...],
        source_mask: Array,
        /,
    ) -> tuple[Array, tuple[Array, ...], tuple[Array, ...], Array]:
        output = jnp.asarray(values)
        mask = jnp.asarray(source_mask, dtype=bool)
        case_ndim = output.ndim - self.spatial_ndim - 1
        coarse_nodes = list(axis_nodes)
        coarse_weights = list(axis_weights)
        if self.mask_mode == "reject":
            output = eqx.error_if(
                output,
                jnp.logical_not(jnp.all(mask)),
                "Resolution-consistent reject mode does not permit source holes.",
            )
        for local_axis in range(self.spatial_ndim):
            absolute_axis = case_ndim + local_axis
            size = int(output.shape[absolute_axis])
            if size % 2:
                raise ValueError("Resolution-consistent restriction requires even axes.")
            left_index = jnp.arange(0, size, 2)
            right_index = left_index + 1
            left = jnp.take(output, left_index, axis=absolute_axis)
            right = jnp.take(output, right_index, axis=absolute_axis)
            left_valid = jnp.take(mask, left_index, axis=absolute_axis)
            right_valid = jnp.take(mask, right_index, axis=absolute_axis)
            weights = coarse_weights[local_axis]
            left_weight = weights[0::2]
            right_weight = weights[1::2]
            value_weight_shape = [1] * output.ndim
            value_weight_shape[absolute_axis] = size // 2
            mask_weight_shape = [1] * mask.ndim
            mask_weight_shape[absolute_axis] = size // 2
            left_value_weight = left_weight.reshape(value_weight_shape)
            right_value_weight = right_weight.reshape(value_weight_shape)
            left_mask_weight = left_weight.reshape(mask_weight_shape)
            right_mask_weight = right_weight.reshape(mask_weight_shape)
            denominator = (
                left_mask_weight * left_valid
                + right_mask_weight * right_valid
            )
            output = (
                left * left_value_weight * left_valid[..., None]
                + right * right_value_weight * right_valid[..., None]
            ) / jnp.maximum(denominator[..., None], jnp.finfo(output.dtype).eps)
            if self.mask_mode == "strict":
                mask = left_valid & right_valid
            else:
                mask = left_valid | right_valid
            output = output * mask[..., None].astype(output.dtype)
            node_denominator = left_weight + right_weight
            coarse_nodes[local_axis] = (
                coarse_nodes[local_axis][0::2] * left_weight
                + coarse_nodes[local_axis][1::2] * right_weight
            ) / node_denominator
            coarse_weights[local_axis] = node_denominator
        return (
            self.projection(output),
            tuple(coarse_nodes),
            tuple(coarse_weights),
            mask,
        )

    def _upsample(
        self,
        values: Array,
        axis_nodes: tuple[Array, ...],
        source_mask: Array,
        target_nodes: tuple[Array, ...],
        target_weights: tuple[Array, ...],
        target_mask: Array,
        /,
    ) -> tuple[Array, tuple[Array, ...], tuple[Array, ...], Array]:
        case_shape = tuple(
            int(size) for size in values.shape[: -self.spatial_ndim - 1]
        )
        query = normalized_lattice_from_nodes(target_nodes)
        query = jnp.broadcast_to(query, case_shape + query.shape)
        sampling_result = sample_rectilinear_grid(
            self.projection(values),
            query,
            spatial_ndim=self.spatial_ndim,
            boundary=self.boundary,
            axis_nodes=axis_nodes,
            source_mask=source_mask,
            mask_mode=self.mask_mode,
            return_support=False,
        )
        if isinstance(sampling_result, tuple):
            raise RuntimeError("Flower prolongation unexpectedly returned support.")
        sampled = sampling_result
        sampled = sampled * target_mask[..., None].astype(sampled.dtype)
        return sampled, target_nodes, target_weights, target_mask

    def __call__(
        self,
        values: Array,
        axis_nodes: tuple[Array, ...],
        axis_weights: tuple[Array, ...],
        source_mask: Array,
        /,
        *,
        target_nodes: tuple[Array, ...] | None = None,
        target_weights: tuple[Array, ...] | None = None,
        target_mask: Array | None = None,
    ) -> tuple[Array, tuple[Array, ...], tuple[Array, ...], Array]:
        if not self.transpose:
            if (
                target_nodes is not None
                or target_weights is not None
                or target_mask is not None
            ):
                raise ValueError("Restriction does not accept target geometry.")
            return self._downsample(
                values,
                axis_nodes,
                axis_weights,
                source_mask,
            )
        if target_nodes is None or target_weights is None or target_mask is None:
            raise ValueError("Prolongation requires target nodes, weights, and mask.")
        return self._upsample(
            values,
            axis_nodes,
            source_mask,
            target_nodes,
            target_weights,
            target_mask,
        )


class FlowerDiagnostics(StrictModule):
    """Per-block learned-route diagnostics for one Flower evaluation."""

    blocks: tuple[RectilinearWarpDiagnostics, ...]
    level_shapes: tuple[tuple[int, ...], ...]
    transition_mode: FlowerTransitionMode

    def __init__(
        self,
        *,
        blocks: Sequence[RectilinearWarpDiagnostics],
        level_shapes: Sequence[Sequence[int]],
        transition_mode: FlowerTransitionMode,
    ):
        self.blocks = tuple(blocks)
        self.level_shapes = tuple(
            tuple(int(size) for size in shape) for shape in level_shapes
        )
        self.transition_mode = transition_mode


class Flower(_AbstractOperatorModel):
    """Multiscale multihead-warp neural operator on rectilinear grids.

    The default follows Muser et al., *Flowers: A Warp Drive for Neural PDE
    Solvers*: learned stride-two transitions, coincident aligned queries,
    deterministic routes, source-hole rejection, and no conservation
    projection. Resolution-consistent transitions, independent query
    interpolation, source masks, probabilistic routes, and channelwise mass
    projection are explicit opt-ins. Learned displacements always use
    domain-normalized coordinates; physical nonuniform and periodic grids
    therefore require unambiguous axis measure metadata.
    """

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    spatial_ndim: int
    width: int
    levels: int
    num_heads: int
    groups: int
    boundary: tuple[WarpBoundaryMode, ...]
    coordinate_embedding: bool
    source_key: str | None
    conditioning_channels: tuple[tuple[str, int], ...]
    fill_value: float
    transition_mode: FlowerTransitionMode
    query_mode: FlowerQueryMode
    source_mask_mode: WarpMaskMode
    probabilistic_routing: bool
    minimum_route_scale: float
    route_scale_factor: float
    conserve_mass: bool
    lift: Linear
    encoder_blocks: tuple[_FlowerBlock, ...]
    down_convolutions: tuple[
        _StrideTwoConvND | _ResolutionConsistentTransitionND, ...
    ]
    bottleneck: _FlowerBlock
    decoder_blocks: tuple[_FlowerBlock, ...]
    up_convolutions: tuple[
        _StrideTwoConvND | _ResolutionConsistentTransitionND, ...
    ]
    projection_hidden: Linear
    projection: Linear

    def __init__(
        self,
        *,
        in_channels: int | Literal["scalar"],
        out_channels: int | Literal["scalar"],
        spatial_ndim: int,
        boundary: WarpBoundaryMode | Sequence[WarpBoundaryMode],
        width: int = 160,
        levels: int = 4,
        num_heads: int = 40,
        groups: int = 40,
        coordinate_embedding: bool = True,
        source_key: str | None = None,
        conditioning_channels: Mapping[
            str, int | Literal["scalar"]
        ] | None = None,
        fill_value: float = 0.0,
        transition_mode: FlowerTransitionMode = "learned",
        query_mode: FlowerQueryMode = "coincident",
        source_mask_mode: WarpMaskMode = "reject",
        probabilistic_routing: bool = False,
        minimum_route_scale: float = 1e-6,
        route_scale_factor: float = 1e-3,
        conserve_mass: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_size = in_channels
        self.out_size = out_channels
        self.spatial_ndim = int(spatial_ndim)
        self.width = int(width)
        self.levels = int(levels)
        self.num_heads = int(num_heads)
        self.groups = int(groups)
        self.coordinate_embedding = bool(coordinate_embedding)
        self.source_key = None if source_key is None else str(source_key)
        self.fill_value = float(fill_value)
        self.transition_mode = transition_mode
        self.query_mode = query_mode
        self.source_mask_mode = source_mask_mode
        self.probabilistic_routing = bool(probabilistic_routing)
        self.minimum_route_scale = float(minimum_route_scale)
        self.route_scale_factor = float(route_scale_factor)
        self.conserve_mass = bool(conserve_mass)
        if self.spatial_ndim not in (1, 2, 3):
            raise ValueError("Flower supports one, two, or three spatial dimensions.")
        if self.width <= 0 or self.levels <= 0:
            raise ValueError("width and levels must be positive.")
        if self.num_heads <= 0 or self.groups <= 0:
            raise ValueError("num_heads and groups must be positive.")
        if self.transition_mode not in ("learned", "resolution_consistent"):
            raise ValueError(
                "transition_mode must be 'learned' or 'resolution_consistent'."
            )
        if self.query_mode not in ("coincident", "interpolate"):
            raise ValueError("query_mode must be 'coincident' or 'interpolate'.")
        if self.source_mask_mode not in ("reject", "renormalize", "strict"):
            raise ValueError(
                "source_mask_mode must be 'reject', 'renormalize', or 'strict'."
            )
        if (
            self.levels > 1
            and self.source_mask_mode != "reject"
            and self.transition_mode != "resolution_consistent"
        ):
            raise ValueError(
                "Masked multilevel Flower requires resolution_consistent transitions."
            )
        if self.minimum_route_scale <= 0.0 or self.route_scale_factor <= 0.0:
            raise ValueError("Probabilistic route scales must be positive.")
        self.boundary = _boundary_modes(boundary, self.spatial_ndim)

        conditions = ()
        if conditioning_channels is not None:
            conditions = tuple(
                sorted(
                    (str(name), _get_size(channels))
                    for name, channels in conditioning_channels.items()
                )
            )
        if len({name for name, _ in conditions}) != len(conditions):
            raise ValueError("conditioning_channels names must be unique.")
        if any(channels <= 0 for _, channels in conditions):
            raise ValueError("Conditioning channel counts must be positive.")
        if self.source_key is not None and self.source_key in {
            name for name, _ in conditions
        }:
            raise ValueError("source_key cannot also name a conditioning input.")
        self.conditioning_channels = conditions
        conditioning_width = sum(channels for _, channels in conditions)

        widths = tuple(self.width * (2**level) for level in range(self.levels))
        if any(
            channels % self.num_heads != 0 or channels % self.groups != 0
            for channels in widths
        ):
            raise ValueError(
                "Every Flower level width must be divisible by num_heads and groups."
            )

        key_count = 4 * self.levels
        keys = iter(jr.split(key, key_count))
        input_count = _get_size(in_channels)
        output_count = _get_size(out_channels)
        if self.conserve_mass and input_count != output_count:
            raise ValueError(
                "conserve_mass requires equal input and output channel counts."
            )
        lifted_input = input_count + (
            self.spatial_ndim if self.coordinate_embedding else 0
        )
        self.lift = Linear(
            in_size=lifted_input,
            out_size=self.width,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=next(keys),
        )

        encoder_blocks = []
        down_convolutions = []
        for level in range(self.levels - 1):
            encoder_blocks.append(
                _FlowerBlock(
                    spatial_ndim=self.spatial_ndim,
                    in_channels=widths[level],
                    out_channels=widths[level],
                    num_heads=self.num_heads,
                    groups=self.groups,
                    boundary=self.boundary,
                    fill_value=self.fill_value,
                    conditioning_width=conditioning_width,
                    mask_mode=self.source_mask_mode,
                    probabilistic_routing=self.probabilistic_routing,
                    minimum_route_scale=self.minimum_route_scale,
                    route_scale_factor=self.route_scale_factor,
                    key=next(keys),
                )
            )
            transition_key = next(keys)
            if self.transition_mode == "learned":
                down_convolutions.append(
                    _StrideTwoConvND(
                        spatial_ndim=self.spatial_ndim,
                        in_channels=widths[level],
                        out_channels=widths[level + 1],
                        transpose=False,
                        key=transition_key,
                    )
                )
            else:
                down_convolutions.append(
                    _ResolutionConsistentTransitionND(
                        spatial_ndim=self.spatial_ndim,
                        in_channels=widths[level],
                        out_channels=widths[level + 1],
                        transpose=False,
                        boundary=self.boundary,
                        mask_mode=self.source_mask_mode,
                        key=transition_key,
                    )
                )
        self.encoder_blocks = tuple(encoder_blocks)
        self.down_convolutions = tuple(down_convolutions)
        self.bottleneck = _FlowerBlock(
            spatial_ndim=self.spatial_ndim,
            in_channels=widths[-1],
            out_channels=widths[-1],
            num_heads=self.num_heads,
            groups=self.groups,
            boundary=self.boundary,
            fill_value=self.fill_value,
            conditioning_width=conditioning_width,
            mask_mode=self.source_mask_mode,
            probabilistic_routing=self.probabilistic_routing,
            minimum_route_scale=self.minimum_route_scale,
            route_scale_factor=self.route_scale_factor,
            key=next(keys),
        )

        decoder_blocks = []
        up_convolutions = []
        for level in range(self.levels - 1, 0, -1):
            decoder_in = widths[level] if level == self.levels - 1 else 2 * widths[level]
            decoder_out = widths[level - 1]
            decoder_blocks.append(
                _FlowerBlock(
                    spatial_ndim=self.spatial_ndim,
                    in_channels=decoder_in,
                    out_channels=decoder_out,
                    num_heads=self.num_heads,
                    groups=self.groups,
                    boundary=self.boundary,
                    fill_value=self.fill_value,
                    conditioning_width=conditioning_width,
                    mask_mode=self.source_mask_mode,
                    probabilistic_routing=self.probabilistic_routing,
                    minimum_route_scale=self.minimum_route_scale,
                    route_scale_factor=self.route_scale_factor,
                    key=next(keys),
                )
            )
            transition_key = next(keys)
            if self.transition_mode == "learned":
                up_convolutions.append(
                    _StrideTwoConvND(
                        spatial_ndim=self.spatial_ndim,
                        in_channels=decoder_out,
                        out_channels=decoder_out,
                        transpose=True,
                        key=transition_key,
                    )
                )
            else:
                up_convolutions.append(
                    _ResolutionConsistentTransitionND(
                        spatial_ndim=self.spatial_ndim,
                        in_channels=decoder_out,
                        out_channels=decoder_out,
                        transpose=True,
                        boundary=self.boundary,
                        mask_mode=self.source_mask_mode,
                        key=transition_key,
                    )
                )
        self.decoder_blocks = tuple(decoder_blocks)
        self.up_convolutions = tuple(up_convolutions)
        self.projection_hidden = Linear(
            in_size=self.width if self.levels == 1 else 2 * self.width,
            out_size=self.width,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=next(keys),
        )
        self.projection = Linear(
            in_size=self.width,
            out_size=output_count,
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=next(keys),
        )

    def _validate_axes(
        self,
        axes: tuple[OperatorAxis, ...],
        values: Array,
        /,
    ) -> Array:
        if len(axes) != self.spatial_ndim:
            raise ValueError(
                f"Flower expects {self.spatial_ndim} spatial axes, got {len(axes)}."
            )
        minimum = 2**self.levels
        checked = values
        for axis, mode in zip(axes, self.boundary, strict=True):
            if axis.size < minimum or axis.size % (2 ** (self.levels - 1)) != 0:
                raise ValueError(
                    f"Flower axis {axis.name!r} must contain a multiple of "
                    f"{2 ** (self.levels - 1)} nodes and leave at least two nodes "
                    "at the coarsest level."
                )
            if (mode == "periodic") != axis.periodic:
                raise ValueError(
                    f"Flower boundary mode {mode!r} disagrees with periodic={axis.periodic} "
                    f"for axis {axis.name!r}."
                )
            nodes = jnp.asarray(axis.nodes)
            spacing = jnp.diff(nodes)
            checked = eqx.error_if(
                checked,
                jnp.logical_not(jnp.all(jnp.isfinite(nodes))),
                f"Flower axis {axis.name!r} contains non-finite nodes.",
            )
            checked = eqx.error_if(
                checked,
                jnp.logical_not(jnp.all(spacing > 0.0)),
                f"Flower axis {axis.name!r} must be strictly increasing.",
            )
            uniform = jnp.allclose(
                spacing,
                jnp.mean(spacing),
                rtol=1e-5,
                atol=1e-8,
            )
            if self.transition_mode == "learned":
                checked = eqx.error_if(
                    checked,
                    jnp.logical_not(uniform),
                    f"Paper-faithful Flower requires uniformly spaced nodes on "
                    f"axis {axis.name!r}; use transition_mode='resolution_consistent' "
                    "for nonuniform grids.",
                )
            elif axis.periodic and axis.quadrature_weights is None:
                checked = eqx.error_if(
                    checked,
                    jnp.logical_not(uniform),
                    f"Nonuniform periodic Flower axis {axis.name!r} requires "
                    "quadrature_weights whose sum is the physical period.",
                )
            if axis.quadrature_weights is not None:
                weights = jnp.asarray(axis.quadrature_weights)
                checked = eqx.error_if(
                    checked,
                    jnp.logical_not(
                        jnp.all(jnp.isfinite(weights) & (weights > 0.0))
                    ),
                    f"Flower axis {axis.name!r} quadrature weights must be "
                    "finite and positive.",
                )
        return checked

    def _axis_geometry(
        self,
        axes: tuple[OperatorAxis, ...],
        values: Array,
        /,
    ) -> tuple[tuple[Array, ...], tuple[Array, ...], Array]:
        normalized_nodes = []
        measure_weights = []
        checked = values
        for axis in axes:
            nodes = jnp.asarray(axis.nodes, dtype=float)
            spacing = jnp.diff(nodes)
            if axis.periodic:
                if axis.quadrature_weights is None:
                    period = nodes[-1] - nodes[0] + jnp.mean(spacing)
                    weights = jnp.full(
                        nodes.shape,
                        period / axis.size,
                        dtype=nodes.dtype,
                    )
                else:
                    weights = jnp.asarray(axis.quadrature_weights, dtype=nodes.dtype)
                    period = jnp.sum(weights)
                checked = eqx.error_if(
                    checked,
                    period <= nodes[-1] - nodes[0],
                    f"Periodic Flower axis {axis.name!r} quadrature weights must "
                    "sum to a period longer than the sampled span.",
                )
                normalized = -1.0 + 2.0 * (nodes - nodes[0]) / period
            else:
                extent = nodes[-1] - nodes[0]
                normalized = -1.0 + 2.0 * (nodes - nodes[0]) / extent
                if axis.quadrature_weights is None:
                    weights = jnp.concatenate(
                        (
                            spacing[:1] / 2.0,
                            (spacing[:-1] + spacing[1:]) / 2.0,
                            spacing[-1:] / 2.0,
                        )
                    )
                else:
                    weights = jnp.asarray(axis.quadrature_weights, dtype=nodes.dtype)
            normalized_nodes.append(normalized)
            measure_weights.append(weights)
        return tuple(normalized_nodes), tuple(measure_weights), checked

    def _prepare_values(
        self,
        values: Array,
        axes: tuple[OperatorAxis, ...],
        /,
    ) -> tuple[Array, tuple[int, ...]]:
        array = jnp.asarray(values)
        if jnp.issubdtype(array.dtype, jnp.complexfloating):
            raise TypeError("Flower currently supports real-valued source fields only.")
        spatial_shape = tuple(axis.size for axis in axes)
        expected_channels = _get_size(self.in_size)
        explicit_spatial = (
            array.ndim >= self.spatial_ndim + 1
            and tuple(array.shape[-self.spatial_ndim - 1 : -1]) == spatial_shape
        )
        explicit = explicit_spatial and int(array.shape[-1]) == expected_channels
        implicit_scalar = (
            expected_channels == 1
            and array.ndim >= self.spatial_ndim
            and tuple(array.shape[-self.spatial_ndim :]) == spatial_shape
        )
        if explicit:
            case_shape = tuple(
                int(size) for size in array.shape[: -self.spatial_ndim - 1]
            )
        elif implicit_scalar:
            case_shape = tuple(int(size) for size in array.shape[: -self.spatial_ndim])
            array = array[..., None]
        elif explicit_spatial:
            raise ValueError(
                f"Expected {expected_channels} source channels, got {array.shape[-1]}."
            )
        else:
            raise ValueError(
                "Flower source values must end in the tensor-grid sample shape, "
                f"optionally followed by {expected_channels} channels; got {array.shape}."
            )
        return self._validate_axes(axes, array), case_shape

    def _coordinate_features(
        self,
        axis_nodes: tuple[Array, ...],
        case_shape: tuple[int, ...],
        dtype: jnp.dtype,
        /,
    ) -> Array:
        coordinates = normalized_lattice_from_nodes(axis_nodes).astype(
            jnp.result_type(dtype, float)
        )
        return jnp.broadcast_to(coordinates, case_shape + coordinates.shape)

    def _source_mask(
        self,
        source_mask: Array | None,
        case_shape: tuple[int, ...],
        spatial_shape: tuple[int, ...],
        values: Array,
        /,
    ) -> tuple[Array, Array]:
        expected = case_shape + spatial_shape
        if source_mask is None:
            mask = jnp.ones(expected, dtype=bool)
        else:
            mask = jnp.asarray(source_mask, dtype=bool)
            if mask.shape == spatial_shape:
                mask = jnp.broadcast_to(mask, expected)
            elif mask.shape != expected:
                raise ValueError(
                    f"Flower source mask must have shape {spatial_shape} or "
                    f"{expected}; got {mask.shape}."
                )
        if self.source_mask_mode == "reject":
            values = eqx.error_if(
                values,
                jnp.logical_not(jnp.all(mask)),
                "Flower source_mask_mode='reject' does not permit source holes.",
            )
        else:
            values = eqx.error_if(
                values,
                jnp.any(
                    jnp.sum(
                        mask,
                        axis=tuple(
                            range(
                                len(case_shape),
                                len(case_shape) + self.spatial_ndim,
                            )
                        ),
                    )
                    == 0
                ),
                "Every Flower case must contain at least one valid source sample.",
            )
        return mask, values

    def _tensor_product_weights(
        self,
        axis_weights: tuple[Array, ...],
        case_shape: tuple[int, ...],
        /,
    ) -> Array:
        factors = jnp.meshgrid(*axis_weights, indexing="ij")
        weights = jnp.ones_like(factors[0])
        for factor in factors:
            weights = weights * factor
        return jnp.broadcast_to(weights, case_shape + weights.shape)

    def _project_conservation(
        self,
        source_values: Array,
        output: Array,
        source_weights: Array,
        target_weights: Array,
        source_mask: Array,
        target_mask: Array,
        /,
    ) -> Array:
        case_ndim = source_values.ndim - self.spatial_ndim - 1
        source_axes = tuple(range(case_ndim, source_values.ndim - 1))
        target_axes = tuple(range(case_ndim, output.ndim - 1))
        source_mass = jnp.sum(
            source_values
            * source_weights[..., None]
            * source_mask[..., None].astype(source_values.dtype),
            axis=source_axes,
        )
        valid_target_weights = target_weights * target_mask.astype(
            target_weights.dtype
        )
        target_mass = jnp.sum(
            output * valid_target_weights[..., None],
            axis=target_axes,
        )
        target_measure = jnp.sum(valid_target_weights, axis=target_axes)
        output = eqx.error_if(
            output,
            jnp.any(target_measure <= 0.0),
            "Every conservative Flower query case must contain positive measure.",
        )
        correction = (source_mass - target_mass) / target_measure[..., None]
        broadcast = source_mass.shape[:-1] + (1,) * len(target_axes) + (
            source_mass.shape[-1],
        )
        corrected = output + correction.reshape(broadcast)
        return corrected * target_mask[..., None].astype(corrected.dtype)

    def _evaluate(
        self,
        values: Array,
        axes: tuple[OperatorAxis, ...],
        condition: Array | None,
        /,
        *,
        source_mask: Array | None = None,
        query_coordinates: Array | None = None,
        query_mask: Array | None = None,
        query_weights: Array | None = None,
        key: EvalKey = None,
        return_diagnostics: bool = False,
    ) -> tuple[Array, FlowerDiagnostics | None]:
        array, case_shape = self._prepare_values(values, axes)
        if condition is not None and tuple(condition.shape[:-1]) != case_shape:
            raise ValueError(
                f"Flower condition case shape must be {case_shape}; "
                f"got {condition.shape[:-1]}."
            )
        spatial_shape = tuple(axis.size for axis in axes)
        normalized_nodes, axis_weights, array = self._axis_geometry(axes, array)
        mask, array = self._source_mask(
            source_mask,
            case_shape,
            spatial_shape,
            array,
        )
        source_values = array
        source_weights = self._tensor_product_weights(axis_weights, case_shape)
        if self.coordinate_embedding:
            array = jnp.concatenate(
                (
                    array,
                    self._coordinate_features(
                        normalized_nodes,
                        case_shape,
                        array.dtype,
                    ),
                ),
                axis=-1,
            )
        hidden = self.lift(array)
        block_keys = iter(split_eval_key(key, 2 * self.levels - 1))
        diagnostics = []
        level_shapes = []
        use_geometry = (
            self.transition_mode == "resolution_consistent"
            or self.source_mask_mode != "reject"
        )

        if self.transition_mode == "learned":
            skips = []
            for block, down in zip(
                self.encoder_blocks,
                self.down_convolutions,
                strict=True,
            ):
                block_key = next(block_keys)
                level_shapes.append(tuple(int(size) for size in hidden.shape[-self.spatial_ndim - 1 : -1]))
                if return_diagnostics:
                    diagnostics.append(
                        block.diagnostics(
                            hidden,
                            condition,
                            axis_nodes=normalized_nodes if use_geometry else None,
                            source_mask=mask if use_geometry else None,
                            key=block_key,
                        )
                    )
                skips.append(hidden)
                hidden = block(
                    hidden,
                    condition,
                    axis_nodes=normalized_nodes if use_geometry else None,
                    source_mask=mask if use_geometry else None,
                    key=block_key,
                )
                if not isinstance(down, _StrideTwoConvND):
                    raise RuntimeError("Flower learned encoder transition mismatch.")
                hidden = jax.nn.relu(down(hidden))
            bottleneck_key = next(block_keys)
            level_shapes.append(
                tuple(
                    int(size)
                    for size in hidden.shape[-self.spatial_ndim - 1 : -1]
                )
            )
            if return_diagnostics:
                diagnostics.append(
                    self.bottleneck.diagnostics(
                        hidden,
                        condition,
                        axis_nodes=None,
                        source_mask=None,
                        key=bottleneck_key,
                    )
                )
            hidden = self.bottleneck(hidden, condition, key=bottleneck_key)
            for block, up, skip in zip(
                self.decoder_blocks,
                self.up_convolutions,
                reversed(skips),
                strict=True,
            ):
                block_key = next(block_keys)
                if return_diagnostics:
                    diagnostics.append(
                        block.diagnostics(
                            hidden,
                            condition,
                            axis_nodes=None,
                            source_mask=None,
                            key=block_key,
                        )
                    )
                if not isinstance(up, _StrideTwoConvND):
                    raise RuntimeError("Flower learned decoder transition mismatch.")
                hidden = jax.nn.relu(
                    up(block(hidden, condition, key=block_key))
                )
                if hidden.shape[:-1] != skip.shape[:-1]:
                    raise ValueError(
                        "Flower decoder and skip shapes disagree: "
                        f"{hidden.shape} versus {skip.shape}."
                    )
                hidden = jnp.concatenate((hidden, skip), axis=-1)
        else:
            skips = []
            current_nodes = normalized_nodes
            current_weights = axis_weights
            current_mask = mask
            for block, down in zip(
                self.encoder_blocks,
                self.down_convolutions,
                strict=True,
            ):
                block_key = next(block_keys)
                level_shapes.append(tuple(int(size) for size in current_mask.shape[-self.spatial_ndim :]))
                if return_diagnostics:
                    diagnostics.append(
                        block.diagnostics(
                            hidden,
                            condition,
                            axis_nodes=current_nodes,
                            source_mask=current_mask,
                            key=block_key,
                        )
                    )
                hidden = block(
                    hidden,
                    condition,
                    axis_nodes=current_nodes,
                    source_mask=current_mask,
                    key=block_key,
                )
                skips.append(
                    (hidden, current_nodes, current_weights, current_mask)
                )
                if not isinstance(down, _ResolutionConsistentTransitionND):
                    raise RuntimeError(
                        "Flower resolution-consistent encoder transition mismatch."
                    )
                hidden, current_nodes, current_weights, current_mask = down(
                    hidden,
                    current_nodes,
                    current_weights,
                    current_mask,
                )
                hidden = jax.nn.relu(hidden)
            bottleneck_key = next(block_keys)
            level_shapes.append(tuple(int(size) for size in current_mask.shape[-self.spatial_ndim :]))
            if return_diagnostics:
                diagnostics.append(
                    self.bottleneck.diagnostics(
                        hidden,
                        condition,
                        axis_nodes=current_nodes,
                        source_mask=current_mask,
                        key=bottleneck_key,
                    )
                )
            hidden = self.bottleneck(
                hidden,
                condition,
                axis_nodes=current_nodes,
                source_mask=current_mask,
                key=bottleneck_key,
            )
            for block, up, skip_data in zip(
                self.decoder_blocks,
                self.up_convolutions,
                reversed(skips),
                strict=True,
            ):
                skip, target_nodes, target_weights, target_mask = skip_data
                block_key = next(block_keys)
                if return_diagnostics:
                    diagnostics.append(
                        block.diagnostics(
                            hidden,
                            condition,
                            axis_nodes=current_nodes,
                            source_mask=current_mask,
                            key=block_key,
                        )
                    )
                hidden = block(
                    hidden,
                    condition,
                    axis_nodes=current_nodes,
                    source_mask=current_mask,
                    key=block_key,
                )
                if not isinstance(up, _ResolutionConsistentTransitionND):
                    raise RuntimeError(
                        "Flower resolution-consistent decoder transition mismatch."
                    )
                hidden, current_nodes, current_weights, current_mask = up(
                    hidden,
                    current_nodes,
                    current_weights,
                    current_mask,
                    target_nodes=target_nodes,
                    target_weights=target_weights,
                    target_mask=target_mask,
                )
                hidden = jax.nn.relu(hidden)
                if hidden.shape[:-1] != skip.shape[:-1]:
                    raise ValueError(
                        "Flower decoder and skip shapes disagree: "
                        f"{hidden.shape} versus {skip.shape}."
                    )
                hidden = jnp.concatenate((hidden, skip), axis=-1)

        output = self.projection(jax.nn.relu(self.projection_hidden(hidden)))
        if query_coordinates is None:
            final_mask = (
                mask
                if query_mask is None
                else mask & jnp.asarray(query_mask, dtype=bool)
            )
            target_weights = (
                source_weights
                if query_weights is None
                else jnp.asarray(query_weights, dtype=source_weights.dtype)
            )
            output = output * final_mask[..., None].astype(output.dtype)
        else:
            sampling_result = sample_rectilinear_grid(
                output,
                query_coordinates,
                spatial_ndim=self.spatial_ndim,
                boundary=self.boundary,
                axis_nodes=normalized_nodes,
                source_mask=mask,
                mask_mode=self.source_mask_mode,
                fill_value=self.fill_value,
                return_support=True,
            )
            if not isinstance(sampling_result, tuple):
                raise RuntimeError("Flower interpolation support was not returned.")
            sampled, support = sampling_result
            query_shape = sampled.shape[:-1]
            final_mask = (
                jnp.ones(query_shape, dtype=bool)
                if query_mask is None
                else jnp.asarray(query_mask, dtype=bool)
            )
            if final_mask.shape != query_shape:
                raise ValueError(
                    f"Flower query mask must have shape {query_shape}; "
                    f"got {final_mask.shape}."
                )
            final_mask = final_mask & support
            output = sampled * final_mask[..., None].astype(sampled.dtype)
            if query_weights is None:
                target_weights = jnp.ones(query_shape, dtype=source_weights.dtype)
            else:
                target_weights = jnp.asarray(
                    query_weights,
                    dtype=source_weights.dtype,
                )
                if target_weights.shape != query_shape:
                    raise ValueError(
                        f"Flower query weights must have shape {query_shape}; "
                        f"got {target_weights.shape}."
                    )
        if self.conserve_mass:
            output = self._project_conservation(
                source_values,
                output,
                source_weights,
                target_weights,
                mask,
                final_mask,
            )
        result = output[..., 0] if self.out_size == "scalar" else output
        diagnostic_result = (
            FlowerDiagnostics(
                blocks=diagnostics,
                level_shapes=level_shapes,
                transition_mode=self.transition_mode,
            )
            if return_diagnostics
            else None
        )
        return result, diagnostic_result

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        condition_names = {name for name, _ in self.conditioning_channels}
        candidates = tuple(
            value for name, value in batch.inputs.items() if name not in condition_names
        )
        if len(candidates) != 1:
            raise ValueError(
                "Flower requires source_key when OperatorBatch does not contain "
                "exactly one nonconditioning source."
            )
        return candidates[0]

    def _conditions(self, batch: OperatorBatch, /) -> Array | None:
        if not self.conditioning_channels:
            return None
        vectors = []
        case_ndim = len(batch.case_shape)
        for name, channels in self.conditioning_channels:
            samples = batch.input(name)
            if samples.values is None:
                raise ValueError(f"Flower condition {name!r} has no values.")
            array = samples.values
            if tuple(int(size) for size in array.shape[:case_ndim]) != batch.case_shape:
                raise ValueError(
                    f"Flower condition {name!r} must begin with case shape "
                    f"{batch.case_shape}; got {array.shape}."
                )
            trailing = tuple(int(size) for size in array.shape[case_ndim:])
            feature_count = prod(trailing) if trailing else 1
            if feature_count != channels:
                raise ValueError(
                    f"Flower condition {name!r} must contain {channels} features "
                    f"per case; got trailing shape {trailing}."
                )
            if samples.mask is not None:
                array = eqx.error_if(
                    array,
                    jnp.logical_not(
                        jnp.all(samples.mask_array(case_shape=batch.case_shape))
                    ),
                    f"Flower condition {name!r} cannot contain masked features.",
                )
            vectors.append(array.reshape(batch.case_shape + (channels,)))
        return jnp.concatenate(vectors, axis=-1)

    def _validate_query_axes(
        self,
        source_axes: tuple[OperatorAxis, ...],
        query_axes: tuple[OperatorAxis, ...],
        values: Array,
        /,
    ) -> Array:
        if len(query_axes) != len(source_axes):
            raise ValueError("Flower requires coincident source and query tensor grids.")
        checked = values
        for source_axis, query_axis in zip(source_axes, query_axes, strict=True):
            if (
                source_axis.name != query_axis.name
                or source_axis.size != query_axis.size
                or source_axis.periodic != query_axis.periodic
            ):
                raise ValueError(
                    "Flower requires matching source/query axis names, sizes, and topology."
                )
            checked = eqx.error_if(
                checked,
                jnp.logical_not(
                    jnp.allclose(
                        source_axis.nodes,
                        query_axis.nodes,
                        rtol=1e-7,
                        atol=1e-10,
                    )
                ),
                f"Flower source and query nodes differ on axis {source_axis.name!r}.",
            )
        return checked

    def _normalize_query_coordinates(
        self,
        source_axes: tuple[OperatorAxis, ...],
        query: FunctionSamples,
        case_shape: tuple[int, ...],
        values: Array,
        /,
    ) -> tuple[Array, Array]:
        if query.axes:
            if len(query.axes) != self.spatial_ndim:
                raise ValueError(
                    f"Flower query grid must have {self.spatial_ndim} axes."
                )
            for source_axis, query_axis in zip(
                source_axes,
                query.axes,
                strict=True,
            ):
                if (
                    source_axis.name != query_axis.name
                    or source_axis.periodic != query_axis.periodic
                ):
                    raise ValueError(
                        "Interpolated Flower query axes must preserve source "
                        "axis names and topology."
                    )
            physical = jnp.stack(
                jnp.meshgrid(
                    *(axis.nodes for axis in query.axes),
                    indexing="ij",
                ),
                axis=-1,
            )
            physical = jnp.broadcast_to(
                physical,
                case_shape + physical.shape,
            )
        elif query.coordinates is not None:
            if int(query.coordinates.shape[-1]) != self.spatial_ndim:
                raise ValueError(
                    f"Flower query coordinates must have trailing dimension "
                    f"{self.spatial_ndim}; got {query.coordinates.shape}."
                )
            physical = query.coordinates_array(case_shape=case_shape)
        else:
            raise ValueError("Flower query requires tensor axes or point coordinates.")

        components = []
        checked = values
        for index, source_axis in enumerate(source_axes):
            nodes = jnp.asarray(source_axis.nodes, dtype=physical.dtype)
            spacing = jnp.diff(nodes)
            if source_axis.periodic:
                period = (
                    nodes[-1] - nodes[0] + jnp.mean(spacing)
                    if source_axis.quadrature_weights is None
                    else jnp.sum(source_axis.quadrature_weights)
                )
                checked = eqx.error_if(
                    checked,
                    period <= nodes[-1] - nodes[0],
                    f"Periodic Flower axis {source_axis.name!r} has invalid period.",
                )
                scale = period
            else:
                scale = nodes[-1] - nodes[0]
            components.append(
                -1.0 + 2.0 * (physical[..., index] - nodes[0]) / scale
            )
        return jnp.stack(components, axis=-1), checked

    def _call_operator_batch(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey,
        return_diagnostics: bool,
    ) -> tuple[Array, FlowerDiagnostics | None]:
        source = self._source(batch)
        if source.values is None:
            raise ValueError("Flower source values cannot be None.")
        if not source.axes:
            raise ValueError("Flower requires a tensor-product source grid.")
        source_values = jnp.asarray(source.values)
        query_coordinates = None
        if self.query_mode == "coincident":
            if not batch.require_single_query().axes:
                raise ValueError(
                    "Paper-faithful Flower requires a tensor-product query grid."
                )
            source_values = self._validate_query_axes(
                source.axes,
                batch.require_single_query().axes,
                source_values,
            )
        else:
            query_coordinates, source_values = self._normalize_query_coordinates(
                source.axes,
                batch.require_single_query(),
                batch.case_shape,
                source_values,
            )
        source_mask = source.mask_array(case_shape=batch.case_shape)
        query_mask = batch.require_single_query().mask_array(case_shape=batch.case_shape)
        if not self.conserve_mass:
            query_weights = batch.require_single_query().quadrature(case_shape=batch.case_shape)
        elif self.query_mode == "coincident":
            query_weights = None
        elif batch.require_single_query().quadrature_weights is not None:
            query_weights = batch.require_single_query().quadrature(case_shape=batch.case_shape)
        elif batch.require_single_query().axes:
            for axis in batch.require_single_query().axes:
                if axis.periodic and axis.quadrature_weights is None:
                    spacing = jnp.diff(axis.nodes)
                    source_values = eqx.error_if(
                        source_values,
                        jnp.logical_not(
                            jnp.allclose(
                                spacing,
                                jnp.mean(spacing),
                                rtol=1e-5,
                                atol=1e-8,
                            )
                        ),
                        "Conservative nonuniform periodic query axes require "
                        "quadrature weights.",
                    )
            _, query_axis_weights, source_values = self._axis_geometry(
                batch.require_single_query().axes,
                source_values,
            )
            query_weights = self._tensor_product_weights(
                query_axis_weights,
                batch.case_shape,
            )
        else:
            raise ValueError(
                "Conservative arbitrary-point Flower queries require explicit "
                "quadrature_weights."
            )
        condition = self._conditions(batch)
        output, diagnostics = self._evaluate(
            source_values,
            source.axes,
            condition,
            source_mask=source_mask,
            query_coordinates=query_coordinates,
            query_mask=query_mask,
            query_weights=query_weights,
            key=key,
            return_diagnostics=return_diagnostics,
        )
        return self.operator_output_specs["output"].validate(output, batch), diagnostics

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        output, _ = self._call_operator_batch(
            batch,
            key=key,
            return_diagnostics=False,
        )
        return output

    def evaluate_with_diagnostics(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> tuple[Array, FlowerDiagnostics]:
        if isinstance(x, OperatorBatch):
            output, diagnostics = self._call_operator_batch(
                x,
                key=key,
                return_diagnostics=True,
            )
        else:
            if self.conditioning_channels:
                raise ValueError(
                    "Conditioned Flower evaluation requires an OperatorBatch."
                )
            if not isinstance(x, tuple) or len(x) != self.spatial_ndim + 1:
                raise ValueError(
                    "Flower requires (values, axis_0, ..., axis_d) structured input."
                )
            axes = tuple(
                OperatorAxis(
                    f"axis_{index}",
                    jnp.asarray(nodes),
                    periodic=self.boundary[index] == "periodic",
                )
                for index, nodes in enumerate(x[1:])
            )
            output, diagnostics = self._evaluate(
                jnp.asarray(x[0]),
                axes,
                None,
                key=key,
                return_diagnostics=True,
            )
        if diagnostics is None:
            raise RuntimeError("Flower diagnostics were not produced.")
        return output, diagnostics

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if isinstance(x, OperatorBatch):
            return self.__call_operator_batch__(x, key=key)
        if self.conditioning_channels:
            raise ValueError(
                "Conditioned Flower evaluation requires an OperatorBatch."
            )
        if not isinstance(x, tuple) or len(x) != self.spatial_ndim + 1:
            raise ValueError(
                "Flower requires (values, axis_0, ..., axis_d) structured input."
            )
        axes = tuple(
            OperatorAxis(
                f"axis_{index}",
                jnp.asarray(nodes),
                periodic=self.boundary[index] == "periodic",
            )
            for index, nodes in enumerate(x[1:])
        )
        output, _ = self._evaluate(
            jnp.asarray(x[0]),
            axes,
            None,
            key=key,
        )
        return output


__all__ = [
    "Flower",
    "FlowerDiagnostics",
    "FlowerQueryMode",
    "FlowerTransitionMode",
]
