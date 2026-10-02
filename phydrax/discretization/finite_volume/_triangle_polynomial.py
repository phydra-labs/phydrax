#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import deque
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked
from ._triangle_fv import TriangleFiniteVolumeDiscretization


def evaluate_triangle_second_moments(
    vertices: ArrayLike, triangles: ArrayLike, /
) -> Array:
    points = jnp.asarray(vertices)[jnp.asarray(triangles, dtype=jnp.int32)]
    center = jnp.mean(points, axis=1)
    displacement = points - center[:, None, :]
    x = displacement[..., 0]
    y = displacement[..., 1]
    x_pair = x[:, 0] * x[:, 1] + x[:, 0] * x[:, 2] + x[:, 1] * x[:, 2]
    y_pair = y[:, 0] * y[:, 1] + y[:, 0] * y[:, 2] + y[:, 1] * y[:, 2]
    second_xx = (jnp.sum(x**2, axis=1) + x_pair) / 6.0
    second_yy = (jnp.sum(y**2, axis=1) + y_pair) / 6.0
    off_xy = (
        x[:, 0] * y[:, 1]
        + x[:, 1] * y[:, 0]
        + x[:, 0] * y[:, 2]
        + x[:, 2] * y[:, 0]
        + x[:, 1] * y[:, 2]
        + x[:, 2] * y[:, 1]
    )
    second_xy = (2.0 * jnp.sum(x * y, axis=1) + off_xy) / 12.0
    return jnp.stack((second_xx, second_xy, second_yy), axis=-1)


def _design_rows(
    delta: np.ndarray,
    neighbor_moments: np.ndarray,
    target_moment: np.ndarray,
    scale: float,
) -> np.ndarray:
    return np.stack(
        (
            delta[:, 0] / scale,
            delta[:, 1] / scale,
            (neighbor_moments[:, 0] + delta[:, 0] ** 2 - target_moment[0]) / scale**2,
            (neighbor_moments[:, 1] + delta[:, 0] * delta[:, 1] - target_moment[1])
            / scale**2,
            (neighbor_moments[:, 2] + delta[:, 1] ** 2 - target_moment[2]) / scale**2,
        ),
        axis=-1,
    )


def _quadratic_stencils(
    discretization: TriangleFiniteVolumeDiscretization,
    moments: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = discretization.cell_count
    owner = np.asarray(discretization.owner_cells, dtype=np.int32)
    neighbor = np.asarray(discretization.neighbor_cells, dtype=np.int32)
    centers = np.asarray(discretization.cell_centers)
    adjacency = [set() for _ in range(count)]
    for left, right in zip(owner, neighbor, strict=True):
        if right >= 0:
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
    stencils = []
    designs = []
    for cell in range(count):
        visited = {cell}
        queue = deque(sorted(adjacency[cell]))
        selected = []
        design = np.empty((0, 5))
        while queue:
            candidate = queue.popleft()
            if candidate in visited:
                continue
            visited.add(candidate)
            selected.append(candidate)
            delta = centers[selected] - centers[cell]
            design = _design_rows(
                delta,
                moments[selected],
                moments[cell],
                scales[cell],
            )
            if len(selected) >= 10 and np.linalg.matrix_rank(design) == 5:
                break
            for next_cell in sorted(adjacency[candidate]):
                if next_cell not in visited:
                    queue.append(next_cell)
        if design.shape[0] < 5 or np.linalg.matrix_rank(design) < 5:
            raise ValueError(
                f"Quadratic triangle stencil for cell {cell} is rank deficient."
            )
        stencils.append(tuple(selected))
        designs.append(design)
    capacity = max(len(stencil) for stencil in stencils)
    indices = np.zeros((count, capacity), dtype=np.int32)
    valid = np.zeros((count, capacity), dtype=np.bool_)
    matrix = np.zeros((count, capacity, 5))
    for cell, (stencil, design) in enumerate(zip(stencils, designs, strict=True)):
        indices[cell, : len(stencil)] = stencil
        valid[cell, : len(stencil)] = True
        matrix[cell, : len(stencil)] = design
    return indices, valid, matrix


class TriangleQuadraticReport(StrictModule):
    maximum_condition_number: Array
    minimum_singular_value: Array
    worst_cell: Array
    stencil_capacity: int = eqx.field(static=True)


class PreparedTriangleQuadratic(StrictModule, NonTrainableState):
    discretization: TriangleFiniteVolumeDiscretization
    moments: Array
    characteristic_lengths: Array
    neighbor_cells: Array
    valid: Array
    factors: Array
    report: TriangleQuadraticReport
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: TriangleFiniteVolumeDiscretization,
        /,
        *,
        weight_power: float = 2.0,
    ) -> None:
        moments = np.asarray(
            evaluate_triangle_second_moments(
                discretization.vertices, discretization.triangles
            )
        )
        scales = np.sqrt(np.asarray(discretization.cell_volumes))
        indices, valid, design = _quadratic_stencils(discretization, moments, scales)
        centers = np.asarray(discretization.cell_centers)
        distance = np.linalg.norm(centers[indices] - centers[:, None], axis=-1)
        normalized_distance = distance / scales[:, None]
        weights = np.where(
            valid,
            1.0 / np.maximum(normalized_distance, 1e-14) ** weight_power,
            0.0,
        )
        root_weight = np.sqrt(weights)
        weighted_design = root_weight[..., None] * design
        left, singular, right_t = np.linalg.svd(weighted_design, full_matrices=False)
        minimum = singular[:, -1]
        condition = singular[:, 0] / minimum
        if (
            np.any(~np.isfinite(condition))
            or np.any(minimum <= 1e-12)
            or np.any(condition > 1e8)
        ):
            raise ValueError(
                "Triangle quadratic reconstruction is singular or ill-conditioned."
            )
        right = np.swapaxes(right_t, -1, -2)
        pseudoinverse = ein.contract("cij,cnj->cin", right / singular[:, None, :], left)
        factors = pseudoinverse * root_weight[:, None, :]
        self.discretization = discretization
        self.moments = jnp.asarray(moments)
        self.characteristic_lengths = jnp.asarray(scales)
        self.neighbor_cells = jnp.asarray(indices)
        self.valid = jnp.asarray(valid)
        self.factors = jnp.asarray(factors)
        self.report = TriangleQuadraticReport(
            maximum_condition_number=jnp.asarray(np.max(condition)),
            minimum_singular_value=jnp.asarray(np.min(minimum)),
            worst_cell=jnp.asarray(np.argmax(condition), dtype=jnp.int32),
            stencil_capacity=indices.shape[1],
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-triangle-quadratic",
                "discretization": discretization.prepared_id,
                "weight_power": float(weight_power),
                "capacity": indices.shape[1],
                "scales": array_tree_fingerprint(scales),
            }
        )

    def coefficients(self, values: Array, /) -> Array:
        value = jnp.asarray(values)
        if value.shape[0] != self.discretization.cell_count:
            raise ValueError("Quadratic values must begin with triangle cell count.")
        return _stencil_coefficients(
            value, value, self.neighbor_cells, self.valid, self.factors
        )

    def cell_coefficients(self, values: Array, cell_routes: Array, /) -> Array:
        """Modal coefficients of the selected cells only, `(routes, ..., 5)`."""
        value = jnp.asarray(values)
        if value.shape[0] != self.discretization.cell_count:
            raise ValueError("Quadratic values must begin with triangle cell count.")
        routes = jnp.asarray(cell_routes, dtype=jnp.int32)
        return _stencil_coefficients(
            value,
            value[routes],
            self.neighbor_cells[routes],
            self.valid[routes],
            self.factors[routes],
        )

    def basis_derivative(
        self, cell_routes: Array, points: Array, derivative: tuple[int, int], /
    ) -> Array:
        """Exact coordinate derivative of the mean-free quadratic cell basis.

        `points` has shape `(routes, points, 2)`; the basis is
        `(x - c)^e / h^|e|` minus the cell moment of each quadratic monomial, so
        the zero multi-index returns the basis and higher derivatives drop the
        constant moments. The result has shape `(routes, points, 5)`.
        """
        orders = tuple(derivative)
        if len(orders) != 2 or any(
            isinstance(order, bool) or not isinstance(order, Integral) or order < 0
            for order in orders
        ):
            raise ValueError("derivative must give one non-negative order per axis.")
        routes = jnp.asarray(cell_routes, dtype=jnp.int32)
        evaluation = jnp.asarray(points)
        dtype = evaluation.dtype
        offset = (
            evaluation - self.discretization.cell_centers.astype(dtype)[routes, None, :]
        )
        scale = self.characteristic_lengths.astype(dtype)[routes, None, None]
        exponents = np.asarray(_QUADRATIC_EXPONENTS, dtype=np.int64)
        coefficient = np.ones((exponents.shape[0],), dtype=np.float64)
        for axis, order in enumerate(orders):
            for count in range(int(order)):
                coefficient = coefficient * (exponents[:, axis] - count)
        reduced = np.maximum(exponents - np.asarray(orders, dtype=np.int64), 0)
        monomials = jnp.prod(
            offset[..., None, :] ** jnp.asarray(reduced, dtype=dtype), axis=-1
        )
        values = (
            jnp.asarray(coefficient, dtype=dtype)
            * monomials
            / scale ** jnp.asarray(np.sum(exponents, axis=1), dtype=dtype)
        )
        if any(orders):
            return values
        moments = self.moments.astype(dtype)[routes, None, :]
        return (
            values
            - jnp.concatenate(
                (jnp.zeros(moments.shape[:-1] + (2,), dtype=dtype), moments), axis=-1
            )
            / scale**2
        )


_QUADRATIC_EXPONENTS = ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2))


def _stencil_coefficients(
    value: Array, base: Array, stencils: Array, valid: Array, factors: Array, /
) -> Array:
    difference = value[stencils] - base[:, None, ...]
    mask = valid.reshape(valid.shape + (1,) * (difference.ndim - 2))
    return ein.contract(
        "cin,cn...->c...i",
        factors.astype(value.dtype),
        jnp.where(mask, difference, 0.0),
    )


class TriangleKExactReconstructionPlan(StrictModule, NonTrainableState):
    prepared: PreparedTriangleQuadratic
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(self, prepared: PreparedTriangleQuadratic, /) -> None:
        self.prepared = prepared
        self.plan_id = canonical_fingerprint(
            {"kind": "triangle-k-exact-degree-2", "prepared": prepared.prepared_id}
        )

    def reconstruct_at(self, state: Array, face_points: Array, /) -> tuple[Array, Array]:
        discretization = self.prepared.discretization
        value = jnp.asarray(state)
        points = jnp.asarray(face_points, dtype=value.dtype)
        if points.ndim != 3 or points.shape[0] != discretization.face_measures.size:
            raise ValueError(
                "Triangle face quadrature points must have shape (faces, q, 2)."
            )
        coefficients = self.prepared.coefficients(value)
        owner = discretization.owner_cells
        neighbor = discretization.neighbor_cells
        safe_neighbor = jnp.maximum(neighbor, 0)

        def basis(cell_indices: Array) -> Array:
            return self.prepared.basis_derivative(cell_indices, points, (0, 0))

        left_delta = ein.contract("f...i,fqi->fq...", coefficients[owner], basis(owner))
        right_delta = ein.contract(
            "f...i,fqi->fq...",
            coefficients[safe_neighbor],
            basis(safe_neighbor),
        )
        return (
            value[owner, None, ...] + left_delta,
            value[safe_neighbor, None, ...] + right_delta,
        )

    def reconstruct(self, state: Array, /) -> tuple[Array, Array]:
        points = self.prepared.discretization.face_centers[:, None, :]
        left, right = self.reconstruct_at(state, points)
        return left[:, 0], right[:, 0]


__all__ = [
    "PreparedTriangleQuadratic",
    "TriangleKExactReconstructionPlan",
    "TriangleQuadraticReport",
    "evaluate_triangle_second_moments",
]
