#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import CellMesh
from ....interchange import GeospatialContract


GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67430e-11
_TETRA_BARYCENTRIC = np.asarray(
    (
        (0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105),
        (0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105),
        (0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105),
        (0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685),
    )
)


class GravityQuadratureSource(StrictModule, NonTrainableState):
    points_m: Array
    volume_weights_m3: Array
    cell_indices: Array
    cell_count: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        points_m: ArrayLike,
        volume_weights_m3: ArrayLike,
        cell_indices: ArrayLike,
        cell_count: int,
        /,
    ):
        points = np.asarray(points_m, dtype=float)
        weights = np.asarray(volume_weights_m3, dtype=float)
        indices = np.asarray(cell_indices)
        count = int(cell_count)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or weights.shape != (points.shape[0],)
            or indices.shape != weights.shape
            or not np.issubdtype(indices.dtype, np.integer)
            or count <= 0
            or np.any(indices < 0)
            or np.any(indices >= count)
            or np.any(~np.isfinite(points))
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError(
                "Gravity quadrature points, weights, and cell routes are invalid."
            )
        self.points_m, self.volume_weights_m3 = jnp.asarray(points), jnp.asarray(weights)
        self.cell_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.cell_count = count
        self.source_id = canonical_fingerprint(
            {
                "kind": "gravity-quadrature-source",
                "points_m": points,
                "weights_m3": weights,
                "cell_indices": indices,
                "cell_count": count,
            }
        )

    @classmethod
    def from_tetrahedra(cls, mesh: CellMesh, /) -> GravityQuadratureSource:
        if not isinstance(mesh, CellMesh) or any(
            block.cell_kind != "tetrahedron" for block in mesh.blocks
        ):
            raise TypeError(
                "Tetrahedral gravity quadrature requires a tetrahedral CellMesh."
            )
        cells = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks]
        )
        coordinates = np.asarray(mesh.coordinates, dtype=float)
        tetrahedra = coordinates[cells]
        determinant = np.linalg.det(
            np.stack(
                (
                    tetrahedra[:, 1] - tetrahedra[:, 0],
                    tetrahedra[:, 2] - tetrahedra[:, 0],
                    tetrahedra[:, 3] - tetrahedra[:, 0],
                ),
                axis=-1,
            )
        )
        volumes = np.abs(determinant) / 6.0
        points = np.einsum("qv,cvi->cqi", _TETRA_BARYCENTRIC, tetrahedra)
        weights = np.repeat((volumes / 4.0)[:, None], 4, axis=1)
        indices = np.repeat(np.arange(cells.shape[0])[:, None], 4, axis=1)
        return cls(
            points.reshape((-1, 3)),
            weights.reshape(-1),
            indices.reshape(-1),
            cells.shape[0],
        )


class GravityResult(StrictModule):
    potential_m2_s2: Array
    acceleration_m_s2: Array
    gradient_s2_inverse: Array
    finite: Array


class FreeSpaceGravityPlan(StrictModule, NonTrainableState):
    """Direct free-space Newton kernel with fixed high-order volume quadrature."""

    source: GravityQuadratureSource
    observations_m: Array
    minimum_separation_m: float = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    coordinates: GeospatialContract
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: GravityQuadratureSource,
        observations_m: ArrayLike,
        coordinates: GeospatialContract,
        /,
        *,
        minimum_separation_m: float,
        block_size: int = 4096,
    ):
        if not isinstance(source, GravityQuadratureSource):
            raise TypeError("Free-space gravity requires GravityQuadratureSource.")
        if not isinstance(coordinates, GeospatialContract):
            raise TypeError("Gravity observations require GeospatialContract.")
        coordinates.require_cartesian(dimensions=3)
        observations = np.asarray(observations_m, dtype=float)
        separation, block = float(minimum_separation_m), int(block_size)
        if (
            observations.ndim != 2
            or observations.shape[1] != 3
            or observations.shape[0] == 0
            or np.any(~np.isfinite(observations))
            or not np.isfinite(separation)
            or separation <= 0
            or block <= 0
        ):
            raise ValueError(
                "Gravity observations, separation, or block size are invalid."
            )
        minimum = np.min(
            np.sqrt(
                np.sum(
                    (observations[:, None, :] - np.asarray(source.points_m)[None, :, :])
                    ** 2,
                    axis=-1,
                )
            )
        )
        if minimum < separation:
            raise ValueError(
                "Gravity observation is inside the declared near-singular exclusion."
            )
        self.source = source
        self.observations_m = jnp.asarray(observations)
        self.minimum_separation_m, self.block_size = separation, block
        self.coordinates = coordinates
        self.plan_id = canonical_fingerprint(
            {
                "kind": "free-space-gravity-plan",
                "source": source.source_id,
                "observations_m": observations,
                "minimum_separation_m": separation,
                "block_size": block,
                "coordinates": coordinates.coordinate_id,
            }
        )

    def evaluate(self, density_kg_m3: ArrayLike, /) -> GravityResult:
        density = jnp.broadcast_to(jnp.asarray(density_kg_m3), (self.source.cell_count,))
        density = eqx.error_if(
            density,
            jnp.any(~jnp.isfinite(density)),
            "Gravity density must be finite.",
        )
        masses = density[self.source.cell_indices] * self.source.volume_weights_m3
        potential_rows, acceleration_rows, gradient_rows = [], [], []
        identity = jnp.eye(3, dtype=masses.dtype)
        for start in range(0, self.observations_m.shape[0], self.block_size):
            observation = self.observations_m[start : start + self.block_size]
            displacement = self.source.points_m[None, :, :] - observation[:, None, :]
            radius_squared = jnp.sum(displacement**2, axis=-1)
            radius = jnp.sqrt(radius_squared)
            radius = eqx.error_if(
                radius,
                jnp.any(radius < self.minimum_separation_m),
                "Gravity evaluation entered its near-singular exclusion.",
            )
            inverse = 1.0 / radius
            potential_rows.append(
                GRAVITATIONAL_CONSTANT_M3_KG_S2 * ein.contract("q,oq->o", masses, inverse)
            )
            acceleration_rows.append(
                GRAVITATIONAL_CONSTANT_M3_KG_S2
                * ein.contract("q,oq,oqi->oi", masses, inverse**3, displacement)
            )
            dyad = ein.contract("oqi,oqj->oqij", displacement, displacement)
            kernel = (
                3.0 * dyad * inverse[..., None, None] ** 5
                - identity * inverse[..., None, None] ** 3
            )
            gradient_rows.append(
                GRAVITATIONAL_CONSTANT_M3_KG_S2
                * ein.contract("q,oqij->oij", masses, kernel)
            )
        potential = jnp.concatenate(potential_rows)
        acceleration = jnp.concatenate(acceleration_rows)
        gradient = jnp.concatenate(gradient_rows)
        finite = (
            jnp.all(jnp.isfinite(potential))
            & jnp.all(jnp.isfinite(acceleration))
            & jnp.all(jnp.isfinite(gradient))
        )
        return GravityResult(potential, acceleration, gradient, finite)

    def observe(
        self,
        density_kg_m3: ArrayLike,
        /,
        *,
        component: Literal[
            "potential", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz"
        ],
    ) -> Array:
        result = self.evaluate(density_kg_m3)
        if component == "potential":
            return result.potential_m2_s2
        if component in ("x", "y", "z"):
            return result.acceleration_m_s2[:, {"x": 0, "y": 1, "z": 2}[component]]
        row, column = {
            "xx": (0, 0),
            "xy": (0, 1),
            "xz": (0, 2),
            "yy": (1, 1),
            "yz": (1, 2),
            "zz": (2, 2),
        }[component]
        return result.gradient_s2_inverse[:, row, column]


__all__ = [
    "FreeSpaceGravityPlan",
    "GRAVITATIONAL_CONSTANT_M3_KG_S2",
    "GravityQuadratureSource",
    "GravityResult",
]
