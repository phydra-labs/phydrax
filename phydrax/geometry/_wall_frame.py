#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class PlanarWallCoordinates(StrictModule):
    normal_coordinate: Array
    tangential_coordinates: Array
    inside_gap: Array
    frame_id: str = eqx.field(static=True)


class PlanarWallFramePlan(StrictModule, NonTrainableState):
    """Static parallel-wall frame with exact normal and tangential coordinates."""

    origin: Array
    inward_normal: Array
    tangential_basis: Array
    gap: float = eqx.field(static=True)
    cross_section_area: float = eqx.field(static=True)
    length_unit_id: str = eqx.field(static=True)
    lower_wall_id: str = eqx.field(static=True)
    upper_wall_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        origin: ArrayLike,
        inward_normal: ArrayLike,
        tangential_basis: ArrayLike,
        /,
        *,
        gap: float,
        cross_section_area: float,
        length_unit_id: str,
        lower_wall_id: str,
        upper_wall_id: str,
    ) -> None:
        origin_ = np.asarray(origin, dtype=np.float64)
        normal = np.asarray(inward_normal, dtype=np.float64)
        tangent = np.asarray(tangential_basis, dtype=np.float64)
        gap_ = float(gap)
        area = float(cross_section_area)
        identities = tuple(
            str(value) for value in (length_unit_id, lower_wall_id, upper_wall_id)
        )
        dimension = origin_.size
        if (
            origin_.ndim != 1
            or dimension < 2
            or normal.shape != origin_.shape
            or tangent.shape != (dimension - 1, dimension)
            or np.any(~np.isfinite(origin_))
            or np.any(~np.isfinite(normal))
            or np.any(~np.isfinite(tangent))
            or not np.isfinite(gap_)
            or gap_ <= 0.0
            or not np.isfinite(area)
            or area <= 0.0
            or any(not value for value in identities)
            or identities[1] == identities[2]
        ):
            raise ValueError("Planar wall frame inputs are invalid.")
        basis = np.concatenate((normal[None, :], tangent), axis=0)
        if not np.allclose(basis @ basis.T, np.eye(dimension), atol=1.0e-10):
            raise ValueError("Wall normal and tangential basis must be orthonormal.")
        self.origin = jnp.asarray(origin_)
        self.inward_normal = jnp.asarray(normal)
        self.tangential_basis = jnp.asarray(tangent)
        self.gap = gap_
        self.cross_section_area = area
        self.length_unit_id = identities[0]
        self.lower_wall_id = identities[1]
        self.upper_wall_id = identities[2]
        self.frame_id = canonical_fingerprint(
            {
                "kind": "planar-wall-frame",
                "origin": array_tree_fingerprint(origin_),
                "normal": array_tree_fingerprint(normal),
                "tangential_basis": array_tree_fingerprint(tangent),
                "gap": gap_,
                "cross_section_area": area,
                "length_unit": identities[0],
                "lower_wall": identities[1],
                "upper_wall": identities[2],
            }
        )

    @property
    def dimension(self) -> int:
        return self.origin.shape[0]

    def coordinates(self, positions: ArrayLike, /) -> PlanarWallCoordinates:
        position = jnp.asarray(positions)
        if position.shape[-1] != self.dimension:
            raise ValueError("Positions must match the planar wall dimension.")
        relative = position - self.origin.astype(position.dtype)
        normal = contract("...i,i->...", relative, self.inward_normal, backend="jax")
        tangent = contract(
            "...i,ji->...j", relative, self.tangential_basis, backend="jax"
        )
        return PlanarWallCoordinates(
            normal,
            tangent,
            (normal >= 0.0) & (normal <= self.gap),
            self.frame_id,
        )


__all__ = ["PlanarWallCoordinates", "PlanarWallFramePlan"]
