#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable Hamiltonian rays in continuously varying refractive media."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._bvh import build_packed_bvh, PackedBVH, point_select_leaf_items
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...geometry.simplicial import AffineSimplexMap


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

    def __init__(
        self,
        index_function: Callable[[Array], Array],
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        field_id: str,
    ):
        if not callable(index_function):
            raise TypeError("index_function must be callable.")
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
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

    def __init__(
        self,
        values: ArrayLike,
        origin: ArrayLike,
        spacing: ArrayLike,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        field_id: str,
    ):
        values_host = np.asarray(values)
        origin_host = np.asarray(origin, dtype=float)
        spacing_host = np.asarray(spacing, dtype=float)
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
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
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
    ):
        vertices_host = np.asarray(vertices, dtype=float)
        tetrahedra_host = np.asarray(tetrahedra, dtype=np.int32)
        values_host = np.asarray(vertex_values, dtype=float)
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
        bvh = build_packed_bvh(
            cell_bounds_min,
            cell_bounds_max,
            np.mean(cells, axis=1),
            leaf_size=min(16, cells.shape[0]),
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


class GradedIndexRayState(StrictModule):
    positions: Array
    momenta: Array
    tangent_maps: Array
    geometric_lengths: Array
    optical_lengths: Array
    valid: Array


class GradedIndexRayEvidence(StrictModule, NonTrainableState):
    finite: Array
    field_covered: Array
    maximum_hamiltonian_drift: Array
    maximum_symplectic_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class GradedIndexRayResult(StrictModule):
    state: GradedIndexRayState
    position_history: Array
    momentum_history: Array
    tangent_history: Array
    initial_directions: Array
    final_directions: Array
    evidence: GradedIndexRayEvidence


@dataclass(frozen=True, slots=True)
class GradedIndexRayPlan:
    field: AbstractRefractiveIndexField
    step_size: float
    step_count: int
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.field, AbstractRefractiveIndexField):
            raise TypeError("field must be AbstractRefractiveIndexField.")
        step_size = float(self.step_size)
        step_count = int(self.step_count)
        if not np.isfinite(step_size) or step_size <= 0.0 or step_count < 1:
            raise ValueError("step_size and step_count must be positive.")
        object.__setattr__(self, "step_size", step_size)
        object.__setattr__(self, "step_count", step_count)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "graded-index-ray-plan",
                    "field": self.field.field_id,
                    "step_size": step_size,
                    "step_count": step_count,
                }
            ),
        )

    def prepare(self) -> PreparedGradedIndexRay:
        return PreparedGradedIndexRay(
            self.field, self.step_size, self.step_count, self.plan_id
        )


class PreparedGradedIndexRay(StrictModule):
    field: AbstractRefractiveIndexField
    step_size: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def integrate(
        self, positions: ArrayLike, directions: ArrayLike, /
    ) -> GradedIndexRayResult:
        positions_ = jnp.asarray(positions)
        directions_ = jnp.asarray(directions, dtype=positions_.dtype)
        if (
            positions_.ndim != 2
            or positions_.shape[1] != 3
            or directions_.shape != positions_.shape
        ):
            raise ValueError("positions and directions must share shape (ray_count, 3).")
        direction_norm = jnp.sqrt(jnp.sum(directions_ * directions_, axis=-1))
        direction_valid = direction_norm > jnp.finfo(positions_.dtype).tiny
        unit = directions_ / jnp.where(
            direction_valid[:, None], direction_norm[:, None], 1.0
        )
        n0, _, _, field_valid = self.field.sample(positions_)
        momenta = n0[:, None] * unit
        tangent = jnp.broadcast_to(
            jnp.eye(6, dtype=positions_.dtype), (positions_.shape[0], 6, 6)
        )
        h0 = 0.5 * (jnp.sum(momenta * momenta, axis=-1) - n0 * n0)
        geometric = jnp.zeros_like(n0)
        optical = jnp.zeros_like(n0)
        valid = direction_valid & field_valid & (n0 > 0.0)
        step = jnp.asarray(self.step_size, dtype=positions_.dtype)

        def one_step(vector):
            x, p = vector[:3], vector[3:]
            n, gradient, _, _ = self.field.sample(x[None, :])
            p_half = p + 0.5 * step * n[0] * gradient[0]
            x_new = x + step * p_half
            n_new, gradient_new, _, _ = self.field.sample(x_new[None, :])
            p_new = p_half + 0.5 * step * n_new[0] * gradient_new[0]
            return jnp.concatenate((x_new, p_new))

        def advance(carry, _):
            x, p, mapping, length, optical_path, active, maximum_h = carry
            vector = jnp.concatenate((x, p), axis=-1)
            next_vector = jax.vmap(one_step)(vector)
            jacobian = jax.vmap(jax.jacfwd(one_step))(vector)
            next_mapping = contract("rij,rjk->rik", jacobian, mapping)
            x_new, p_new = next_vector[:, :3], next_vector[:, 3:]
            n_old, _, _, valid_old = self.field.sample(x)
            n_new, _, _, valid_new = self.field.sample(x_new)
            next_length = length + 0.5 * step * (n_old + n_new)
            next_optical = optical_path + 0.5 * step * (n_old * n_old + n_new * n_new)
            hamiltonian = 0.5 * (jnp.sum(p_new * p_new, axis=-1) - n_new * n_new)
            next_active = active & valid_old & valid_new & (n_new > 0.0)
            maximum_h = jnp.maximum(maximum_h, jnp.abs(hamiltonian - h0))
            return (
                x_new,
                p_new,
                next_mapping,
                next_length,
                next_optical,
                next_active,
                maximum_h,
            ), (x_new, p_new, next_mapping)

        initial = (
            positions_,
            momenta,
            tangent,
            geometric,
            optical,
            valid,
            jnp.zeros_like(n0),
        )
        final, history = jax.lax.scan(advance, initial, xs=None, length=self.step_count)
        x_final, p_final, map_final, geometric, optical, valid, max_h = final
        positions_history = jnp.concatenate((positions_[None], history[0]), axis=0)
        momenta_history = jnp.concatenate((momenta[None], history[1]), axis=0)
        tangent_history = jnp.concatenate((tangent[None], history[2]), axis=0)
        n_final, _, _, final_covered = self.field.sample(x_final)
        final_direction = p_final / jnp.where(
            n_final[:, None] > 0.0, n_final[:, None], 1.0
        )
        symplectic = jnp.block(
            [[jnp.zeros((3, 3)), jnp.eye(3)], [-jnp.eye(3), jnp.zeros((3, 3))]]
        ).astype(positions_.dtype)
        residual = (
            contract("rji,jk,rkl->ril", map_final, symplectic, map_final) - symplectic
        )
        maximum_symplectic = jnp.max(jnp.abs(residual))
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value))
                    for value in (x_final, p_final, map_final, geometric, optical)
                )
            )
        )
        covered = jnp.all(valid & final_covered)
        tolerance = 2.0e3 * jnp.finfo(positions_.dtype).eps * self.step_count
        successful = (
            finite
            & covered
            & (jnp.max(max_h) <= tolerance)
            & (maximum_symplectic <= 10.0 * tolerance)
        )
        evidence = GradedIndexRayEvidence(
            finite, covered, jnp.max(max_h), maximum_symplectic, successful, self.plan_id
        )
        state = GradedIndexRayState(
            x_final, p_final, map_final, geometric, optical, valid & final_covered
        )
        return GradedIndexRayResult(
            state,
            positions_history,
            momenta_history,
            tangent_history,
            unit,
            final_direction,
            evidence,
        )


class RayFanResult(StrictModule, NonTrainableState):
    determinant_history: Array
    minimum_singular_value: Array
    caustic_crossings: Array
    caustic_detected: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class RayFanPlan:
    transverse_basis: np.ndarray
    determinant_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        basis = np.array(self.transverse_basis, dtype=float, copy=True)
        tolerance = float(self.determinant_tolerance)
        if basis.shape != (2, 3) or not np.allclose(
            basis @ basis.T, np.eye(2), atol=1.0e-10, rtol=0.0
        ):
            raise ValueError("transverse_basis must be an orthonormal (2, 3) basis.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("determinant_tolerance must be positive.")
        basis.setflags(write=False)
        object.__setattr__(self, "transverse_basis", basis)
        object.__setattr__(self, "determinant_tolerance", tolerance)

    def evaluate(self, rays: GradedIndexRayResult, /) -> RayFanResult:
        basis = jnp.asarray(self.transverse_basis, dtype=rays.tangent_history.dtype)
        position_momentum = rays.tangent_history[..., :3, 3:]
        transverse = contract("ai,srij,bj->srab", basis, position_momentum, basis)
        determinant = (
            transverse[..., 0, 0] * transverse[..., 1, 1]
            - transverse[..., 0, 1] * transverse[..., 1, 0]
        )
        frobenius_squared = jnp.sum(transverse * transverse, axis=(-2, -1))
        discriminant = jnp.maximum(
            frobenius_squared * frobenius_squared - 4.0 * determinant * determinant,
            0.0,
        )
        minimum_eigenvalue = 0.5 * (frobenius_squared - jnp.sqrt(discriminant))
        singular = jnp.sqrt(jnp.maximum(minimum_eigenvalue, 0.0))
        near = jnp.abs(determinant) <= self.determinant_tolerance
        crossings = jnp.sum((determinant[1:] * determinant[:-1] < 0.0) | near[1:], axis=0)
        detected = crossings > 0
        finite = jnp.all(jnp.isfinite(determinant)) & jnp.all(jnp.isfinite(singular))
        return RayFanResult(
            determinant,
            jnp.min(singular, axis=0),
            crossings,
            detected,
            finite & rays.evidence.finite,
        )


__all__ = [
    "AnalyticRefractiveIndexField",
    "AbstractRefractiveIndexField",
    "GradedIndexRayEvidence",
    "GradedIndexRayPlan",
    "GradedIndexRayResult",
    "GradedIndexRayState",
    "PreparedGradedIndexRay",
    "RayFanPlan",
    "RayFanResult",
    "StructuredRefractiveIndexField",
    "TetrahedralRefractiveIndexField",
]
