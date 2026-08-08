#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.nn as jnn
import jax.random as jr
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from .._keys import EvalKey
from ._linear import Linear
from ._warp import MultiheadWarp, WarpBoundaryMode
from ._warp_geometry import (
    GaussianWarpRoute,
    RectilinearWarpDiagnostics,
    WarpMaskMode,
)


class ProbabilisticMultiheadWarp(StrictModule):
    """Multihead warp with a coherent Gaussian displacement-field route.

    Keyless evaluation follows the mean route. Supplying an evaluation key draws
    one coherent displacement at every case, grid point, and head.
    """

    base: MultiheadWarp
    scale_projection: Linear
    minimum_scale: float
    scale_factor: float

    def __init__(
        self,
        *,
        spatial_ndim: int,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        boundary: WarpBoundaryMode | Sequence[WarpBoundaryMode],
        conditioning_size: int = 0,
        mask_mode: WarpMaskMode = "reject",
        displacement_width: int | None = None,
        fill_value: float = 0.0,
        minimum_scale: float = 1e-6,
        scale_factor: float = 1e-3,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if float(minimum_scale) <= 0.0 or float(scale_factor) <= 0.0:
            raise ValueError("Probabilistic warp scales must be positive.")
        base_key, scale_key = jr.split(key)
        self.base = MultiheadWarp(
            spatial_ndim=spatial_ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            boundary=boundary,
            conditioning_size=conditioning_size,
            mask_mode=mask_mode,
            displacement_width=displacement_width,
            fill_value=fill_value,
            key=base_key,
        )
        hidden_width = int(self.base.displacement_hidden.weight.shape[0])
        self.scale_projection = Linear(
            in_size=hidden_width,
            out_size=int(num_heads) * int(spatial_ndim),
            activation=None,
            rwf=False,
            bias_init_lim=0.0,
            key=scale_key,
        )
        self.minimum_scale = float(minimum_scale)
        self.scale_factor = float(scale_factor)

    def distribution(
        self,
        values: Array,
        /,
        *,
        condition: Array | None = None,
    ) -> GaussianWarpRoute:
        mean = self.base.displacement(values, condition=condition)
        hidden = self.base.displacement_features(values, condition=condition)
        raw_scale = self.scale_projection(jax.nn.gelu(hidden)).reshape(mean.shape)
        scale = self.minimum_scale + self.scale_factor * jnn.softplus(raw_scale)
        return GaussianWarpRoute(mean, scale)

    def diagnostics(
        self,
        values: Array,
        /,
        *,
        condition: Array | None = None,
        axis_nodes: Sequence[Array] | None = None,
        source_mask: Array | None = None,
        key: EvalKey = None,
    ) -> RectilinearWarpDiagnostics:
        """Report mean-route or exact keyed sampled-route geometry."""

        distribution = self.distribution(values, condition=condition)
        displacement = distribution.mean if key is None else distribution.sample(key)
        return self.base.diagnostics_from_displacement(
            values,
            displacement,
            axis_nodes=axis_nodes,
            source_mask=source_mask,
            route_scale=distribution.scale,
        )

    def __call__(
        self,
        values: Array,
        /,
        *,
        condition: Array | None = None,
        axis_nodes: Sequence[Array] | None = None,
        source_mask: Array | None = None,
        key: EvalKey = None,
    ) -> Array:
        distribution = self.distribution(values, condition=condition)
        displacement = distribution.mean if key is None else distribution.sample(key)
        return self.base.transport(
            values,
            displacement,
            axis_nodes=axis_nodes,
            source_mask=source_mask,
        )


__all__ = ["ProbabilisticMultiheadWarp"]
