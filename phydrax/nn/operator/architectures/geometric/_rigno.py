#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax.geometry.operator import RegionalGeometryMode, RegionalPointLatentGeometry
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.operator.architectures.geometric._geometry_operator import (
    _GeometryOperatorCore,
)
from phydrax.nn.operator.data import OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel
from phydrax.nn.operator.layers._graph_transfer import (
    GraphKernelTransfer,
    TransferReduction,
)
from phydrax.nn.operator.layers._regional_processor import RegionalGraphProcessor


class RIGNO(AbstractOperatorModel):
    """Regional interaction graph neural operator for irregular geometries.

    Physical source fields are integrated onto a deterministic fixed-size set
    of regional nodes, processed by a measure-aware latent graph network, and
    decoded onto arbitrary query coordinates. Every graph is constructed per
    case inside JAX; no physical samples or messages can cross case boundaries.
    """

    operator_architecture = "RIGNO"

    core: _GeometryOperatorCore
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | Literal["scalar"]
    coord_dim: int = eqx.field(static=True)
    regional_count: int = eqx.field(static=True)
    source_keys: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_channels: int
        | Literal["scalar"]
        | Mapping[str, int | Literal["scalar"]] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        coord_dim: int,
        regional_count: int,
        regional_mode: RegionalGeometryMode = "farthest_point",
        fixed_regional_points: Any | None = None,
        latent_channels: int = 32,
        processor_neighbors: int = 8,
        processor_radius: float | None = None,
        processor_depth: int = 4,
        processor_width: int = 64,
        processor_mlp_depth: int = 2,
        processor_shared: bool = True,
        processor_edge_dropout: float = 0.0,
        processor_residual_scale: float = 1.0,
        processor_activation: Callable[[Array], Array] = jnp.tanh,
        encoder_neighbors: int = 16,
        decoder_neighbors: int = 16,
        encoder_radius: float | None = None,
        decoder_radius: float | None = None,
        transfer_reduction: TransferReduction = "integral",
        transfer_width: int = 32,
        transfer_depth: int = 2,
        coordinate_scale: float = 1.0,
        source_key: str | None = None,
        query_channels: int = 0,
        query_chunk_size: int | None = 256,
        assume_uniform_measure: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if int(coord_dim) <= 0 or int(regional_count) <= 0:
            raise ValueError("coord_dim and regional_count must be positive.")
        if int(latent_channels) <= 0:
            raise ValueError("latent_channels must be positive.")
        if int(query_channels) < 0:
            raise ValueError("query_channels must be non-negative.")
        if int(processor_neighbors) > int(regional_count):
            raise ValueError("processor_neighbors cannot exceed regional_count.")
        if int(decoder_neighbors) > int(regional_count):
            raise ValueError("decoder_neighbors cannot exceed regional_count.")

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

        geometry = RegionalPointLatentGeometry(
            int(regional_count),
            int(coord_dim),
            mode=regional_mode,
            fixed_points=fixed_regional_points,
        )
        keys = jr.split(key, len(source_channels) + 3)
        encoders = tuple(
            GraphKernelTransfer(
                in_channels=channels,
                out_channels=int(latent_channels),
                coord_dim=int(coord_dim),
                neighbors=int(encoder_neighbors),
                radius=encoder_radius,
                reduction=transfer_reduction,
                width=int(transfer_width),
                depth=int(transfer_depth),
                coordinate_scale=float(coordinate_scale),
                target_chunk_size=query_chunk_size,
                key=keys[index],
            )
            for index, channels in enumerate(source_channels)
        )
        processor = RegionalGraphProcessor(
            int(latent_channels),
            int(coord_dim),
            neighbors=int(processor_neighbors),
            radius=processor_radius,
            depth=int(processor_depth),
            width=int(processor_width),
            mlp_depth=int(processor_mlp_depth),
            target_chunk_size=query_chunk_size,
            shared=bool(processor_shared),
            edge_dropout=float(processor_edge_dropout),
            residual_scale=float(processor_residual_scale),
            activation=processor_activation,
            key=keys[-3],
        )
        decoder = GraphKernelTransfer(
            in_channels=int(latent_channels),
            target_channels=int(query_channels),
            out_channels=output_channels,
            coord_dim=int(coord_dim),
            neighbors=int(decoder_neighbors),
            radius=decoder_radius,
            reduction=transfer_reduction,
            width=int(transfer_width),
            depth=int(transfer_depth),
            coordinate_scale=float(coordinate_scale),
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
        self.regional_count = int(regional_count)
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
            raise TypeError("RIGNO requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["RIGNO"]
