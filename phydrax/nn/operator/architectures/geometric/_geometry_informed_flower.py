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

from phydrax._doc import DOC_KEY0
from phydrax.geometry.operator import BoundsPolicy, TensorGridLatentGeometry
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.layers._warp import _boundary_modes, WarpBoundaryMode
from phydrax.nn.layers._warp_geometry import WarpMaskMode
from phydrax.nn.operator.architectures.dynamics._flower import (
    Flower,
    FlowerTransitionMode,
)
from phydrax.nn.operator.architectures.geometric._geometry_operator import (
    _GeometryOperatorCore,
    GeometryOperatorDiagnostics,
    LatentSupportKind,
    TensorGridProcessor,
)
from phydrax.nn.operator.data import OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel
from phydrax.nn.operator.layers._graph_transfer import (
    GraphKernelTransfer,
    TransferReduction,
)


class GeometryInformedFlower(AbstractOperatorModel):
    """Geometry encoder/decoder wrapped around a latent-grid :class:`Flower`.

    Irregular source samples are encoded onto a tensor latent grid by learned,
    measure-aware graph integrals. Flower processes that grid, after which a
    second graph integral decodes at arbitrary query coordinates. A named
    occupancy or signed-distance input can additionally define a hard latent
    support mask. Mesh connectivity is not consumed: sources and queries are
    interpreted as weighted point sets in ambient coordinates.
    """

    operator_architecture = "GeometryInformedFlower"

    core: _GeometryOperatorCore
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | Literal["scalar"]
    coord_dim: int = eqx.field(static=True)
    latent_shape: tuple[int, ...] = eqx.field(static=True)
    source_keys: tuple[str, ...] = eqx.field(static=True)
    conditioning_channels: tuple[tuple[str, int], ...] = eqx.field(static=True)
    latent_support_key: str | None = eqx.field(static=True)
    conserve_mass: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_channels: int
        | Literal["scalar"]
        | Mapping[str, int | Literal["scalar"]] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        coord_dim: int,
        latent_shape: Sequence[int],
        boundary: WarpBoundaryMode | Sequence[WarpBoundaryMode] = "clamp",
        latent_bounds: Any | None = None,
        bounds_policy: BoundsPolicy = "global",
        latent_margin: float = 0.0,
        latent_channels: int = 32,
        flower_width: int = 32,
        flower_levels: int = 2,
        flower_num_heads: int = 4,
        flower_groups: int = 4,
        coordinate_embedding: bool = True,
        transition_mode: FlowerTransitionMode = "resolution_consistent",
        source_mask_mode: WarpMaskMode = "strict",
        fill_value: float = 0.0,
        probabilistic_routing: bool = False,
        minimum_route_scale: float = 1e-6,
        route_scale_factor: float = 1e-3,
        conserve_latent_mass: bool = False,
        encoder_neighbors: int = 16,
        decoder_neighbors: int = 16,
        encoder_radius: float | None = None,
        decoder_radius: float | None = None,
        transfer_reduction: TransferReduction = "integral",
        transfer_width: int = 32,
        transfer_depth: int = 2,
        coordinate_scale: float = 1.0,
        source_key: str | None = None,
        conditioning_channels: Mapping[str, int | Literal["scalar"]] | None = None,
        query_channels: int = 0,
        query_chunk_size: int | None = 256,
        assume_uniform_measure: bool = False,
        latent_support_key: str | None = None,
        latent_support_kind: LatentSupportKind = "occupancy",
        latent_support_threshold: float | None = None,
        latent_support_neighbors: int = 4,
        latent_support_radius: float | None = None,
        conserve_mass: bool = False,
        conservation_source_key: str | None = None,
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

        conditions = ()
        if conditioning_channels is not None:
            conditions = tuple(
                sorted(
                    (str(name), _get_size(channels))
                    for name, channels in conditioning_channels.items()
                )
            )
        if len({name for name, _ in conditions}) != len(conditions):
            raise ValueError("Conditioning channel names must be unique.")
        condition_names = {name for name, _ in conditions}
        if condition_names.intersection(source_keys):
            raise ValueError("Conditioning inputs cannot also be encoded sources.")
        if source_key is not None and str(source_key) in condition_names:
            raise ValueError("source_key cannot also name a conditioning input.")
        support_key = None if latent_support_key is None else str(latent_support_key)
        if support_key is not None and support_key in condition_names:
            raise ValueError("A latent support input cannot be a case condition.")
        if support_key is not None and source_mask_mode == "reject":
            raise ValueError(
                "Hard latent support requires source_mask_mode='strict' or 'renormalize'."
            )
        threshold = (
            (0.5 if latent_support_kind == "occupancy" else 0.0)
            if latent_support_threshold is None
            else float(latent_support_threshold)
        )
        if latent_support_kind == "occupancy" and not 0.0 <= threshold <= 1.0:
            raise ValueError("Occupancy support thresholds must lie in [0, 1].")

        boundary_modes = _boundary_modes(boundary, int(coord_dim))
        geometry = TensorGridLatentGeometry(
            shape,
            bounds=latent_bounds,
            bounds_policy=bounds_policy,
            periodic=tuple(mode == "periodic" for mode in boundary_modes),
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
        flower = Flower(
            in_channels=int(latent_channels),
            out_channels=int(latent_channels),
            spatial_ndim=int(coord_dim),
            boundary=boundary_modes,
            width=int(flower_width),
            levels=int(flower_levels),
            num_heads=int(flower_num_heads),
            groups=int(flower_groups),
            coordinate_embedding=coordinate_embedding,
            source_key="latent",
            conditioning_channels=dict(conditions),
            fill_value=float(fill_value),
            transition_mode=transition_mode,
            query_mode="coincident",
            source_mask_mode=source_mask_mode,
            probabilistic_routing=probabilistic_routing,
            minimum_route_scale=float(minimum_route_scale),
            route_scale_factor=float(route_scale_factor),
            conserve_mass=conserve_latent_mass,
            key=keys[-3],
        )
        processor = TensorGridProcessor(
            flower,
            geometry,
            int(latent_channels),
            execution="operator_batch",
            source_key="latent",
            conditioning_channels=conditions,
            supports_diagnostics=True,
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
            conditioning_channels=conditions,
            latent_support_key=support_key,
            latent_support_kind=latent_support_kind,
            latent_support_threshold=threshold,
            latent_support_neighbors=int(latent_support_neighbors),
            latent_support_radius=latent_support_radius,
            conserve_mass=conserve_mass,
            conservation_source_key=conservation_source_key,
        )
        self.in_size = in_size
        self.out_size = out_channels
        self.coord_dim = int(coord_dim)
        self.latent_shape = shape
        self.source_keys = source_keys
        self.conditioning_channels = conditions
        self.latent_support_key = support_key
        self.conserve_mass = bool(conserve_mass)

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

    def evaluate_with_diagnostics(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> tuple[Array, GeometryOperatorDiagnostics]:
        output, diagnostics = self.core.evaluate_with_diagnostics(batch, key=key)
        if self.out_size == "scalar":
            output = output[..., 0]
        return output, diagnostics

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("GeometryInformedFlower requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["GeometryInformedFlower"]
