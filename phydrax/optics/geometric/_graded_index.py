#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Refractive-index media and the isotropic graded-index ray specialization."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._bvh import BVHBuildPolicy, PackedBVH, point_select_leaf_items, prepare_bvh
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry.simplicial import AffineSimplexMap
from ...typing import checked
from ._dispersion_rays import (
    AbstractSeparableDispersionHamiltonian,
    DispersionRayPlan,
    PreparedDispersionRay,
)


class AbstractRefractiveIndexField(StrictModule):
    field_id: str = eqx.field(static=True)
    coordinate_contract: SpatialCoordinateContract = eqx.field(static=True)

    @abstractmethod
    def sample(self, points: Array, /) -> tuple[Array, Array, Array, Array]:
        """Return index, gradient, Hessian, and support validity."""
        raise NotImplementedError


class AnalyticRefractiveIndexField(AbstractRefractiveIndexField):
    """Smooth analytic refractive index differentiated by JAX."""

    index_function: Callable[[Array], Array] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        index_function: Callable[[Array], Array],
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        field_id: str,
    ) -> None:
        if not callable(index_function):
            raise TypeError("index_function must be callable.")
        if not isinstance(field_id, str) or not field_id:
            raise ValueError("field_id must be nonempty.")
        self.index_function = index_function
        self.coordinate_contract = coordinate_contract
        self.field_id = canonical_fingerprint(
            {
                "kind": "analytic-refractive-index",
                "name": field_id,
                "coordinates": coordinate_contract.spatial_id,
            }
        )

    def sample(self, points: Array, /) -> tuple[Array, Array, Array, Array]:
        points_ = jnp.asarray(points)
        flat = points_.reshape((-1, 3))
        values = jax.vmap(self.index_function)(flat)
        gradients = jax.vmap(jax.grad(self.index_function))(flat)
        hessians = jax.vmap(jax.hessian(self.index_function))(flat)
        valid = (
            jnp.isfinite(values)
            & (values > 0.0)
            & jnp.all(jnp.isfinite(gradients), axis=-1)
            & jnp.all(jnp.isfinite(hessians), axis=(-2, -1))
        )
        shape = points_.shape[:-1]
        return (
            values.reshape(shape),
            gradients.reshape(shape + (3,)),
            hessians.reshape(shape + (3, 3)),
            valid.reshape(shape),
        )


class StructuredRefractiveIndexField(AbstractRefractiveIndexField, NonTrainableState):
    values: Array
    origin: Array
    spacing: Array

    @checked
    def __init__(
        self,
        values: ArrayLike,
        origin: ArrayLike,
        spacing: ArrayLike,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        field_id: str,
    ) -> None:
        values_host = np.asarray(values)
        origin_host = np.asarray(origin, dtype=np.float64)
        spacing_host = np.asarray(spacing, dtype=np.float64)
        if values_host.ndim != 3 or any(size < 2 for size in values_host.shape):
            raise ValueError(
                "values must be a three-dimensional grid with at least two nodes per axis."
            )
        if (
            not np.issubdtype(values_host.dtype, np.floating)
            or not np.all(np.isfinite(values_host))
            or np.any(values_host <= 0.0)
        ):
            raise ValueError(
                "Refractive indices must be finite positive floating-point values."
            )
        if (
            origin_host.shape != (3,)
            or spacing_host.shape != (3,)
            or not np.all(np.isfinite(origin_host))
            or not np.all(np.isfinite(spacing_host))
            or np.any(spacing_host <= 0.0)
        ):
            raise ValueError(
                "origin and spacing must be finite three-vectors with positive spacing."
            )
        if not isinstance(field_id, str) or not field_id:
            raise ValueError("field_id must be nonempty.")
        self.values = jnp.asarray(values_host)
        self.origin = jnp.asarray(origin_host, dtype=self.values.dtype)
        self.spacing = jnp.asarray(spacing_host, dtype=self.values.dtype)
        self.coordinate_contract = coordinate_contract
        self.field_id = canonical_fingerprint(
            {
                "kind": "structured-refractive-index",
                "name": field_id,
                "values": array_tree_fingerprint(values_host),
                "origin": origin_host.tolist(),
                "spacing": spacing_host.tolist(),
                "coordinates": coordinate_contract.spatial_id,
            }
        )

    def _value_one(self, point: Array) -> Array:
        coordinate = (point - self.origin) / self.spacing
        lower = jnp.floor(coordinate).astype(jnp.int32)
        maximum = jnp.asarray(self.values.shape) - 2
        safe = jnp.clip(lower, 0, maximum)
        fraction = coordinate - safe
        result = jnp.asarray(0.0, dtype=self.values.dtype)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    weight = (
                        (fraction[0] if di else 1.0 - fraction[0])
                        * (fraction[1] if dj else 1.0 - fraction[1])
                        * (fraction[2] if dk else 1.0 - fraction[2])
                    )
                    result = (
                        result
                        + weight * self.values[safe[0] + di, safe[1] + dj, safe[2] + dk]
                    )
        return result

    def sample(self, points: Array, /) -> tuple[Array, Array, Array, Array]:
        points_ = jnp.asarray(points, dtype=self.values.dtype)
        flat = points_.reshape((-1, 3))
        coordinate = (flat - self.origin) / self.spacing
        valid = jnp.all(
            (coordinate >= 0.0) & (coordinate <= jnp.asarray(self.values.shape) - 1),
            axis=-1,
        )
        values = jax.vmap(self._value_one)(flat)
        gradients = jax.vmap(jax.grad(self._value_one))(flat)
        hessians = jax.vmap(jax.hessian(self._value_one))(flat)
        shape = points_.shape[:-1]
        return (
            values.reshape(shape),
            gradients.reshape(shape + (3,)),
            hessians.reshape(shape + (3, 3)),
            valid.reshape(shape),
        )


class TetrahedralRefractiveIndexField(AbstractRefractiveIndexField, NonTrainableState):
    vertices: Array
    tetrahedra: Array
    vertex_values: Array
    origins: Array
    inverse_edges: Array
    value_gradients: Array
    simplex: AffineSimplexMap
    bvh: PackedBVH
    maximum_candidates: int = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        tetrahedra: ArrayLike,
        vertex_values: ArrayLike,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        field_id: str,
        maximum_candidates: int = 64,
    ) -> None:
        vertices_host = np.asarray(vertices, dtype=np.float64)
        tetrahedra_host = np.asarray(tetrahedra, dtype=np.int32)
        values_host = np.asarray(vertex_values, dtype=np.float64)
        if (
            vertices_host.ndim != 2
            or vertices_host.shape[1] != 3
            or not np.all(np.isfinite(vertices_host))
        ):
            raise ValueError("vertices must have shape (vertex_count, 3) and be finite.")
        if (
            tetrahedra_host.ndim != 2
            or tetrahedra_host.shape[1] != 4
            or np.any(tetrahedra_host < 0)
            or np.any(tetrahedra_host >= vertices_host.shape[0])
        ):
            raise ValueError(
                "tetrahedra must have shape (cell_count, 4) with valid indices."
            )
        if (
            values_host.shape != (vertices_host.shape[0],)
            or not np.all(np.isfinite(values_host))
            or np.any(values_host <= 0.0)
        ):
            raise ValueError(
                "vertex_values must be finite positive values for every vertex."
            )
        cells = jnp.asarray(vertices_host[tetrahedra_host])
        simplex = AffineSimplexMap(cells)
        if not bool(jnp.all(simplex.evidence.successful)):
            raise ValueError("Tetrahedral refractive field contains degenerate cells.")
        candidate_capacity = int(maximum_candidates)
        if candidate_capacity <= 0:
            raise ValueError("maximum_candidates must be positive.")
        cell_bounds_min = np.min(cells, axis=1)
        cell_bounds_max = np.max(cells, axis=1)
        bvh = prepare_bvh(
            cell_bounds_min,
            cell_bounds_max,
            policy=BVHBuildPolicy(leaf_size=min(16, cells.shape[0])),
            dtype=simplex.vertices.dtype,
        )
        nodal_values = jnp.asarray(values_host[tetrahedra_host])
        gradients = simplex.physical_gradient(nodal_values)
        self.vertices = jnp.asarray(vertices_host)
        self.tetrahedra = jnp.asarray(tetrahedra_host)
        self.vertex_values = jnp.asarray(values_host)
        self.origins = simplex.origin
        self.inverse_edges = simplex.dual
        self.value_gradients = gradients
        self.simplex = simplex
        self.bvh = bvh
        self.maximum_candidates = candidate_capacity
        self.coordinate_contract = coordinate_contract
        self.field_id = canonical_fingerprint(
            {
                "kind": "tetrahedral-refractive-index",
                "name": field_id,
                "vertices": array_tree_fingerprint(vertices_host),
                "tetrahedra": array_tree_fingerprint(tetrahedra_host),
                "values": array_tree_fingerprint(values_host),
                "maximum_candidates": candidate_capacity,
                "coordinates": coordinate_contract.spatial_id,
            }
        )

    def sample(self, points: Array, /) -> tuple[Array, Array, Array, Array]:
        points_ = jnp.asarray(points, dtype=self.vertices.dtype)
        flat = points_.reshape((-1, 3))
        candidates, candidate_valid, complete = point_select_leaf_items(
            flat,
            bvh=self.bvh,
            maximum_candidates=self.maximum_candidates,
        )
        candidate_points = jnp.broadcast_to(
            flat[:, None, :],
            candidates.shape + (3,),
        )
        tolerance = 64.0 * jnp.finfo(points_.dtype).eps
        contained = (
            self.simplex.contains_at(
                candidate_points,
                candidates,
                tolerance=tolerance,
            )
            & candidate_valid
        )
        safe_cells = jnp.where(contained, candidates, self.simplex.simplex_count)
        cell = jnp.min(safe_cells, axis=-1)
        located = cell < self.simplex.simplex_count
        cell = jnp.minimum(cell, self.simplex.simplex_count - 1)
        barycentric = self.simplex.barycentric_at(flat, cell)
        values = jnp.sum(
            barycentric * self.vertex_values[self.tetrahedra[cell]],
            axis=-1,
        )
        gradients = self.value_gradients[cell]
        hessians = jnp.zeros((flat.shape[0], 3, 3), dtype=points_.dtype)
        valid = located & complete
        shape = points_.shape[:-1]
        return (
            values.reshape(shape),
            gradients.reshape(shape + (3,)),
            hessians.reshape(shape + (3, 3)),
            valid.reshape(shape),
        )


class RefractiveIndexHamiltonian(AbstractSeparableDispersionHamiltonian):
    """Isotropic graded-index Hamiltonian ``H = ½(|p|² − n(x)²)``.

    The canonical momentum is ``p = n t`` for the unit ray tangent ``t``; the ray
    parameter satisfies ``ds = n dτ``, so the optical length is ``∫ n² dτ``.
    """

    field: AbstractRefractiveIndexField

    @checked
    def __init__(self, field: AbstractRefractiveIndexField, /) -> None:
        self.field = field
        self.coordinate_contract = field.coordinate_contract
        self.hamiltonian_id = canonical_fingerprint(
            {"kind": "refractive-index-ray-hamiltonian", "field": field.field_id}
        )

    def potential(self, position: Array, /) -> tuple[Array, Array, Array]:
        n, gradient, _, valid = self.field.sample(position[None, :])
        return -0.5 * n[0] * n[0], -(n[0] * gradient[0]), valid[0] & (n[0] > 0.0)

    def supported(self, position: Array, /) -> Array:
        n, _, _, valid = self.field.sample(position[None, :])
        return valid[0] & (n[0] > 0.0)

    def launch(self, position: Array, direction: Array, /) -> tuple[Array, Array]:
        n, _, _, valid = self.field.sample(position[None, :])
        return n[0] * direction, valid[0] & (n[0] > 0.0)


class GradedIndexRayPlan(StrictModule, NonTrainableState):
    """Graded-index specialization of `DispersionRayPlan`.

    Binds `RefractiveIndexHamiltonian` to the explicit kick–drift–kick schedule
    with roundoff Hamiltonian-drift evidence.
    """

    rays: DispersionRayPlan

    def __init__(
        self, field: AbstractRefractiveIndexField, step_size: float, step_count: int, /
    ) -> None:
        self.rays = DispersionRayPlan(
            RefractiveIndexHamiltonian(field),
            step_size,
            step_count,
            method="kick-drift-kick",
        )

    @property
    def plan_id(self) -> str:
        return self.rays.plan_id

    def prepare(self) -> PreparedDispersionRay:
        return self.rays.prepare()


__all__ = [
    "AbstractRefractiveIndexField",
    "AnalyticRefractiveIndexField",
    "GradedIndexRayPlan",
    "RefractiveIndexHamiltonian",
    "StructuredRefractiveIndexField",
    "TetrahedralRefractiveIndexField",
]
