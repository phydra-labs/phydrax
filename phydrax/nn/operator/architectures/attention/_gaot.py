#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast, Literal

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax.geometry.operator import BoundsPolicy, TensorGridLatentGeometry
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.operator.architectures.geometric._geometry_operator import (
    _GeometryOperatorCore,
)
from phydrax.nn.operator.data import OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel
from phydrax.nn.operator.layers._graph_transfer import (
    GraphAttentionTransfer,
    MultiscaleGraphTransfer,
)
from phydrax.nn.operator.layers._transformer import OperatorTransformerProcessor


def _magno_transfer(
    *,
    in_channels: int,
    target_channels: int,
    out_channels: int,
    coord_dim: int,
    neighbors: int,
    radii: tuple[float, ...],
    node_width: int,
    heads: int,
    depth: int,
    coordinate_scale: float,
    reference_measure: float,
    fusion: Literal["concat", "gated"],
    target_chunk_size: int | None,
    key: Key[Array, ""],
) -> MultiscaleGraphTransfer:
    keys = jr.split(key, len(radii) + 1)
    transfers = tuple(
        GraphAttentionTransfer(
            in_channels=int(in_channels),
            target_channels=int(target_channels),
            out_channels=int(out_channels),
            coord_dim=int(coord_dim),
            neighbors=int(neighbors),
            radius=radius,
            node_width=int(node_width),
            heads=int(heads),
            depth=int(depth),
            coordinate_scale=float(coordinate_scale),
            require_measure=True,
            target_chunk_size=target_chunk_size,
            key=scale_key,
        )
        for radius, scale_key in zip(radii, keys[:-1], strict=True)
    )
    return MultiscaleGraphTransfer(
        transfers,
        fusion=fusion,
        reference_measure=float(reference_measure),
        width=int(node_width),
        depth=int(depth),
        key=keys[-1],
    )


class GAOT(AbstractOperatorModel):
    """Geometry-aware operator transformer with multiscale graph transfer.

    A MAGNO-style multiscale attention encoder maps irregular physical fields
    onto a structured 2D or 3D latent grid. A patchwise U-shaped transformer
    processes that grid before a multiscale attention decoder evaluates the
    result at arbitrary query coordinates.
    """

    operator_architecture = "GAOT"

    core: _GeometryOperatorCore
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | Literal["scalar"]
    coord_dim: int = eqx.field(static=True)
    latent_shape: tuple[int, ...] = eqx.field(static=True)
    patch_shape: tuple[int, ...] = eqx.field(static=True)
    transfer_radii: tuple[float, ...] = eqx.field(static=True)
    source_keys: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_channels: int
        | Literal["scalar"]
        | Mapping[str, int | Literal["scalar"]] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        coord_dim: int,
        latent_shape: Sequence[int],
        patch_shape: int | Sequence[int] = 2,
        transfer_radius: float,
        transfer_scales: Sequence[float] = (1.0, 2.0, 4.0),
        latent_bounds: Any | None = None,
        bounds_policy: BoundsPolicy = "case_bbox",
        latent_margin: float = 0.0,
        latent_channels: int = 32,
        transformer_width: int = 128,
        transformer_depth: int = 3,
        transformer_heads: int = 8,
        feed_forward_multiplier: float = 4.0,
        attention_dropout: float = 0.0,
        feed_forward_dropout: float = 0.0,
        long_range_skip: bool = True,
        transfer_neighbors: int = 32,
        transfer_width: int = 64,
        transfer_heads: int = 4,
        transfer_depth: int = 2,
        transfer_fusion: Literal["concat", "gated"] = "gated",
        coordinate_scale: float = 1.0,
        reference_measure: float = 1.0,
        source_key: str | None = None,
        query_channels: int = 0,
        query_chunk_size: int | None = 256,
        assume_uniform_measure: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if int(coord_dim) not in (2, 3):
            raise ValueError("GAOT supports coord_dim 2 or 3.")
        shape = tuple(int(size) for size in latent_shape)
        if len(shape) != int(coord_dim) or any(size <= 1 for size in shape):
            raise ValueError(
                "latent_shape must match coord_dim and contain dimensions greater than one."
            )
        patches = (
            (int(patch_shape),) * int(coord_dim)
            if isinstance(patch_shape, int)
            else tuple(int(size) for size in patch_shape)
        )
        if len(patches) != int(coord_dim) or any(size <= 0 for size in patches):
            raise ValueError("patch_shape must give one positive size per dimension.")
        if any(size % patch != 0 for size, patch in zip(shape, patches, strict=True)):
            raise ValueError("Every latent dimension must be divisible by patch_shape.")
        if float(transfer_radius) <= 0.0:
            raise ValueError(
                "transfer_radius must be positive physical-coordinate scale."
            )
        scales = tuple(float(scale) for scale in transfer_scales)
        if not scales or any(scale <= 0.0 for scale in scales):
            raise ValueError("transfer_scales must contain positive values.")
        if len(set(scales)) != len(scales):
            raise ValueError("transfer_scales must be unique.")
        radii = tuple(float(transfer_radius) * scale for scale in scales)
        if int(latent_channels) <= 0 or int(query_channels) < 0:
            raise ValueError(
                "latent_channels must be positive and query_channels non-negative."
            )
        if int(transfer_neighbors) <= 0:
            raise ValueError("transfer_neighbors must be positive.")
        if transfer_fusion not in ("concat", "gated"):
            raise ValueError("transfer_fusion must be 'concat' or 'gated'.")

        if isinstance(in_channels, Mapping):
            if source_key is not None:
                raise ValueError(
                    "source_key cannot be combined with mapping-valued in_channels."
                )
            source_items = tuple(
                (str(name), _get_size(channels))
                for name, channels in cast(
                    Mapping[str, int | Literal["scalar"]], in_channels
                ).items()
            )
            if not source_items:
                raise ValueError("Mapping-valued in_channels cannot be empty.")
            source_keys = tuple(name for name, _ in source_items)
            source_channels = tuple(channels for _, channels in source_items)
            in_size: int | tuple[int, ...] | Literal["scalar"] = tuple(source_channels)
        else:
            source_keys = () if source_key is None else (str(source_key),)
            source_channels = (_get_size(in_channels),)
            in_size = in_channels
        output_channels = _get_size(out_channels)

        geometry = TensorGridLatentGeometry(
            shape,
            bounds=latent_bounds,
            bounds_policy=bounds_policy,
            margin=float(latent_margin),
        )
        keys = jr.split(key, len(source_channels) + 3)
        encoders = tuple(
            _magno_transfer(
                in_channels=channels,
                target_channels=0,
                out_channels=int(latent_channels),
                coord_dim=int(coord_dim),
                neighbors=int(transfer_neighbors),
                radii=radii,
                node_width=int(transfer_width),
                heads=int(transfer_heads),
                depth=int(transfer_depth),
                coordinate_scale=float(coordinate_scale),
                reference_measure=float(reference_measure),
                fusion=transfer_fusion,
                target_chunk_size=query_chunk_size,
                key=keys[index],
            )
            for index, channels in enumerate(source_channels)
        )
        processor = OperatorTransformerProcessor(
            shape,
            int(latent_channels),
            patch_shape=patches,
            model_width=int(transformer_width),
            depth=int(transformer_depth),
            heads=int(transformer_heads),
            feed_forward_multiplier=float(feed_forward_multiplier),
            attention_dropout=float(attention_dropout),
            feed_forward_dropout=float(feed_forward_dropout),
            long_range_skip=bool(long_range_skip),
            key=keys[-3],
        )
        decoder = _magno_transfer(
            in_channels=int(latent_channels),
            target_channels=int(query_channels),
            out_channels=output_channels,
            coord_dim=int(coord_dim),
            neighbors=int(transfer_neighbors),
            radii=radii,
            node_width=int(transfer_width),
            heads=int(transfer_heads),
            depth=int(transfer_depth),
            coordinate_scale=float(coordinate_scale),
            reference_measure=float(reference_measure),
            fusion=transfer_fusion,
            target_chunk_size=query_chunk_size,
            key=keys[-2],
        )
        latent_mixer = (
            None
            if len(encoders) == 1
            else Linear(
                in_size=len(encoders) * int(latent_channels),
                out_size=int(latent_channels),
                key=keys[-1],
            )
        )
        self.core = _GeometryOperatorCore(
            encoders=encoders,
            processor=processor,
            decoder=decoder,
            latent_geometry=geometry,
            source_channels=source_channels,
            source_keys=source_keys,
            latent_channels=int(latent_channels),
            query_channels=int(query_channels),
            latent_mixer=latent_mixer,
            assume_uniform_measure=assume_uniform_measure,
        )
        self.in_size = in_size
        self.out_size = out_channels
        self.coord_dim = int(coord_dim)
        self.latent_shape = shape
        self.patch_shape = patches
        self.transfer_radii = radii
        self.source_keys = source_keys

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        output = self.core(batch, key=key)
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("GAOT requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["GAOT"]
