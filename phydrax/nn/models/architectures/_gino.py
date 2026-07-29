#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..._utils import _get_size
from ..core._base import _AbstractOperatorModel
from ..core._keys import EvalKey
from ..core._operator import OperatorBatch
from ..core._operator_geometry import BoundsPolicy, TensorGridLatentGeometry
from ..layers._graph_transfer import GraphKernelTransfer, TransferReduction
from ..layers._linear import Linear
from ._fno import Factorization, FNO
from ._geometry_operator import _GeometryOperatorCore, TensorGridProcessor


class GINO(_AbstractOperatorModel):
    """Geometry-informed neural operator over irregular sources and queries.

    Source fields are transferred to a structured latent grid by a learned
    measure-aware graph integral, processed by an N-dimensional FNO, and
    decoded onto arbitrary query coordinates by a second graph integral.
    """

    core: _GeometryOperatorCore
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | Literal["scalar"]
    coord_dim: int = eqx.field(static=True)
    latent_shape: tuple[int, ...] = eqx.field(static=True)
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
        latent_bounds: Any | None = None,
        bounds_policy: BoundsPolicy = "global",
        latent_margin: float = 0.0,
        latent_channels: int = 32,
        modes: Sequence[int] | None = None,
        fno_width: int = 32,
        fno_depth: int = 4,
        fno_factorization: Factorization = "dense",
        fno_rank: int | float = 0.5,
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
        shape = tuple(int(size) for size in latent_shape)
        if len(shape) != int(coord_dim):
            raise ValueError("latent_shape rank must match coord_dim.")
        if int(latent_channels) <= 0:
            raise ValueError("latent_channels must be positive.")
        if int(query_channels) < 0:
            raise ValueError("query_channels must be non-negative.")
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
        if decoder_neighbors > int(jnp.prod(jnp.asarray(shape))):
            raise ValueError("decoder_neighbors cannot exceed the latent point count.")
        active_modes = (
            tuple(max(1, min(8, size // 2)) for size in shape)
            if modes is None
            else tuple(int(value) for value in modes)
        )
        if len(active_modes) != len(shape) or any(value <= 0 for value in active_modes):
            raise ValueError("modes must give one positive count per latent dimension.")

        geometry = TensorGridLatentGeometry(
            shape,
            bounds=latent_bounds,
            bounds_policy=bounds_policy,
            margin=latent_margin,
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
        fno = FNO(
            n_modes=active_modes,
            in_channels=int(latent_channels),
            out_channels=int(latent_channels),
            width=int(fno_width),
            depth=int(fno_depth),
            coordinate_embedding=True,
            factorization=fno_factorization,
            rank=fno_rank,
            key=keys[-3],
        )
        processor = TensorGridProcessor(fno, geometry, int(latent_channels))
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
        self.latent_shape = shape
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
            raise TypeError("GINO requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["GINO"]
