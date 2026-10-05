# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Surface preparation of declared point transfers in target tangent frames.

The generic transfer (``_transfer.py``) owns constraints, solves and audits.
This module owns only what is specific to a hypersurface support: moment
coordinates are the projections of route offsets onto each target point's
tangent plane, built from the authoritative target normals, and the support's
departure from those tangent planes is reported.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...linalg import LinearSolvePolicy
from ...optim import ConvexSolvePolicy
from ...sparse import EdgeRelation, RowRelation
from ...typing import checked, Dim, Float64
from ._stencils import PreparedLocalStencils
from ._transfer import (
    PointTransferPlan,
    PointTransferRequest,
    PreparedPointTransfer,
    stencil_routes,
)


class SurfaceTransferTargetDim(Dim):
    """Target surface points."""


class SurfaceTransferIntrinsicDim(Dim):
    """Tangent directions of a target point."""


class SurfaceTransferAmbientDim(Dim):
    """Ambient coordinates."""


def _tangent_frames(normals: np.ndarray, /) -> np.ndarray:
    """Orthonormal tangent bases ``(points, D - 1, D)`` of unit hypersurface normals.

    The Householder reflection mapping ``n`` to a signed coordinate axis is
    orthogonal and symmetric; its remaining columns span the tangent plane
    deterministically without a branch on the normal direction.
    """
    unit = np.asarray(normals, dtype=np.float64)
    count, dimension = unit.shape
    axis = np.argmax(np.abs(unit), axis=1)
    sign = np.where(unit[np.arange(count), axis] >= 0, 1.0, -1.0)
    reflector = unit.copy()
    reflector[np.arange(count), axis] += sign
    householder = (
        np.eye(dimension)[None]
        - 2.0
        * (reflector[:, :, None] * reflector[:, None, :])
        / np.sum(reflector * reflector, axis=1)[:, None, None]
    )
    keep = np.arange(dimension)[None, :] != axis[:, None]
    return householder[keep].reshape(count, dimension - 1, dimension)


@final
class SurfaceTransferPlan(StrictModule):
    """Declared point transfer between two samplings of one hypersurface.

    ``target_normals`` are the authoritative unit normals of the target points
    (curves in 2-D, sheets in 3-D). Declared moments are imposed in the target
    tangent coordinates; ``normal_departure`` is the largest ratio
    ``|offset . n| / |offset|`` of the support, i.e. how far the routes leave
    the tangent planes where the moment coordinates are exact.
    """

    __strict_contract__ = True
    plan: PointTransferPlan
    frames: Float64[
        SurfaceTransferTargetDim, SurfaceTransferIntrinsicDim, SurfaceTransferAmbientDim
    ]
    normal_departure: float = eqx.field(static=True)

    def __init__(
        self,
        relation: EdgeRelation | RowRelation,
        coefficients: ArrayLike,
        offsets: ArrayLike,
        target_normals: ArrayLike,
        source_measures: ArrayLike,
        target_measures: ArrayLike,
        /,
        *,
        source_id: str,
        target_id: str,
        request: PointTransferRequest,
        tolerance: float = 1e-10,
        linear_policy: LinearSolvePolicy | None = None,
        conic_policy: ConvexSolvePolicy | None = None,
        lebesgue_bound: float | None = None,
    ) -> None:
        normals = np.asarray(target_normals, dtype=np.float64)
        shift = np.asarray(offsets, dtype=np.float64)
        if (
            normals.ndim != 2
            or normals.shape[1] < 2
            or not np.all(np.isfinite(normals))
            or not np.allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=0, atol=1e-10)
        ):
            raise ValueError(
                "target_normals must be finite unit (points, ambient) normals."
            )
        if shift.ndim < 2 or shift.shape[-1] != normals.shape[1]:
            raise ValueError("Route offsets must share the normals' ambient dimension.")
        frames = _tangent_frames(normals)
        plan = PointTransferPlan(
            relation,
            coefficients,
            source_measures,
            target_measures,
            source_id=source_id,
            target_id=target_id,
            request=request,
            offsets=shift,
            frames=frames,
            tolerance=tolerance,
            linear_policy=linear_policy,
            conic_policy=conic_policy,
            lebesgue_bound=lebesgue_bound,
        )
        rows = np.asarray(plan.routes.rows)
        edge = (
            relation
            if isinstance(relation, EdgeRelation)
            else relation.as_edge_relation()
        )
        valid = np.asarray(edge.valid).reshape(-1)
        routes = shift.reshape(-1, shift.shape[-1])[valid]
        lengths = np.linalg.norm(routes, axis=1)
        normal = np.abs(np.sum(routes * normals[rows], axis=1))
        ratio = np.divide(normal, lengths, out=np.zeros_like(normal), where=lengths > 0)
        self.plan = plan
        self.frames = jnp.asarray(frames)
        self.normal_departure = float(np.max(ratio, initial=0.0))

    @classmethod
    @checked
    def from_stencils(
        cls,
        stencils: PreparedLocalStencils,
        source_measures: ArrayLike,
        target_measures: ArrayLike,
        /,
        *,
        target_normals: ArrayLike,
        request: PointTransferRequest,
        functional_index: int = 0,
        tolerance: float = 1e-10,
        linear_policy: LinearSolvePolicy | None = None,
        conic_policy: ConvexSolvePolicy | None = None,
        lebesgue_bound: float | None = None,
    ) -> SurfaceTransferPlan:
        """Base coefficients and offsets from admitted cross-target stencils."""
        relation, weights, offsets = stencil_routes(stencils, functional_index)
        return cls(
            relation,
            weights,
            offsets,
            target_normals,
            source_measures,
            target_measures,
            source_id=f"{stencils.neighborhood.neighborhood_id}:source",
            target_id=f"{stencils.neighborhood.neighborhood_id}:target",
            request=request,
            tolerance=tolerance,
            linear_policy=linear_policy,
            conic_policy=conic_policy,
            lebesgue_bound=lebesgue_bound,
        )

    def prepare(self) -> PreparedPointTransfer:
        return self.plan.prepare()


__all__ = ["SurfaceTransferPlan"]
