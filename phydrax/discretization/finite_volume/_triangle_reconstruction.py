#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import deque
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import parse
from ._triangle_fv import TriangleFiniteVolumeDiscretization


TriangleLimiterKind: TypeAlias = Literal[
    "unlimited", "barth_jespersen", "venkatakrishnan"
]


def _cell_neighbor_stencils(
    owner: np.ndarray,
    neighbor: np.ndarray,
    cell_count: int,
    centers: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    adjacency = [set() for _ in range(cell_count)]
    for left, right in zip(owner, neighbor, strict=True):
        if right >= 0:
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
    stencils = []
    for cell in range(cell_count):
        visited = {cell}
        queue = deque(sorted(adjacency[cell]))
        selected = []
        while queue:
            candidate = queue.popleft()
            if candidate in visited:
                continue
            visited.add(candidate)
            selected.append(candidate)
            offsets = centers[selected] - centers[cell]
            if len(selected) >= 2 and np.linalg.matrix_rank(offsets) == 2:
                break
            for next_cell in sorted(adjacency[candidate]):
                if next_cell not in visited:
                    queue.append(next_cell)
        offsets = centers[selected] - centers[cell]
        if len(selected) < 2 or np.linalg.matrix_rank(offsets) < 2:
            raise ValueError(f"Triangle WLSQ stencil for cell {cell} is rank deficient.")
        stencils.append(tuple(selected))
    capacity = max(len(stencil) for stencil in stencils)
    indices = np.zeros((cell_count, capacity), dtype=np.int32)
    valid = np.zeros((cell_count, capacity), dtype=np.bool_)
    for cell, stencil in enumerate(stencils):
        indices[cell, : len(stencil)] = stencil
        valid[cell, : len(stencil)] = True
    return indices, valid


class TriangleWLSQReport(StrictModule):
    maximum_condition_number: Array
    worst_cell: Array
    stencil_capacity: int = eqx.field(static=True)


class PreparedTriangleWLSQ(StrictModule, NonTrainableState):
    discretization: TriangleFiniteVolumeDiscretization
    neighbor_cells: Array
    valid: Array
    offsets: Array
    factors: Array
    report: TriangleWLSQReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: TriangleFiniteVolumeDiscretization,
        /,
        *,
        weight_power: float = 2.0,
    ) -> None:
        if not isinstance(discretization, TriangleFiniteVolumeDiscretization):
            raise TypeError("WLSQ requires triangular finite-volume geometry.")
        centers = np.asarray(discretization.cell_centers)
        indices, valid = _cell_neighbor_stencils(
            np.asarray(discretization.owner_cells),
            np.asarray(discretization.neighbor_cells),
            discretization.cell_count,
            centers,
        )
        offsets = centers[indices] - centers[:, None, :]
        distance = np.linalg.norm(offsets, axis=-1)
        weights = np.where(
            valid,
            1.0 / np.maximum(distance, 1e-14) ** weight_power,
            0.0,
        )
        square_root_weights = np.sqrt(weights)
        weighted_offsets = square_root_weights[..., None] * offsets
        left_vectors, singular_values, right_vectors_t = np.linalg.svd(
            weighted_offsets,
            full_matrices=False,
        )
        minimum_singular = singular_values[:, -1]
        condition = singular_values[:, 0] / minimum_singular
        if (
            np.any(~np.isfinite(condition))
            or np.any(minimum_singular <= 1e-12)
            or np.any(condition > 1e6)
        ):
            raise ValueError("Triangle WLSQ geometry is singular or ill-conditioned.")
        right_vectors = np.swapaxes(right_vectors_t, -1, -2)
        scaled_right_vectors = right_vectors / singular_values[:, None, :]
        pseudoinverse = ein.contract("cij,cnj->cin", scaled_right_vectors, left_vectors)
        factors = pseudoinverse * square_root_weights[:, None, :]
        report = TriangleWLSQReport(
            maximum_condition_number=jnp.asarray(np.max(condition)),
            worst_cell=jnp.asarray(np.argmax(condition), dtype=jnp.int32),
            stencil_capacity=indices.shape[1],
        )
        self.discretization = discretization
        self.neighbor_cells = jnp.asarray(indices)
        self.valid = jnp.asarray(valid)
        self.offsets = jnp.asarray(offsets)
        self.factors = jnp.asarray(factors)
        self.report = report
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-triangle-wlsq",
                "discretization": discretization.prepared_id,
                "weight_power": float(weight_power),
                "capacity": indices.shape[1],
            }
        )

    def gradient(self, values: Array, /) -> Array:
        value = jnp.asarray(values)
        if value.shape[0] != self.discretization.cell_count:
            raise ValueError("WLSQ values must begin with triangle cell count.")
        return _stencil_gradient(
            value, value, self.neighbor_cells, self.valid, self.factors
        )

    def cell_gradients(self, values: Array, cell_routes: Array, /) -> Array:
        """WLSQ gradients of the selected cells only, `(routes, ..., 2)`."""
        value = jnp.asarray(values)
        if value.shape[0] != self.discretization.cell_count:
            raise ValueError("WLSQ values must begin with triangle cell count.")
        routes = jnp.asarray(cell_routes, dtype=jnp.int32)
        return _stencil_gradient(
            value,
            value[routes],
            self.neighbor_cells[routes],
            self.valid[routes],
            self.factors[routes],
        )


def _stencil_gradient(
    value: Array, base: Array, stencils: Array, valid: Array, factors: Array, /
) -> Array:
    difference = value[stencils] - base[:, None, ...]
    mask = valid.reshape(valid.shape + (1,) * (difference.ndim - 2))
    return ein.contract(
        "cin,cn...->c...i",
        factors.astype(value.dtype),
        jnp.where(mask, difference, 0.0),
    )


def _limiter_factor(
    limiter: TriangleLimiterKind,
    epsilon: float,
    delta: Array,
    upper: Array,
    lower: Array,
    /,
) -> Array:
    """Barth--Jespersen or Venkatakrishnan factor of one reconstructed increment."""
    allowed = jnp.where(delta >= 0.0, upper, lower)
    ratio = allowed / jnp.where(jnp.abs(delta) > epsilon, delta, 1.0)
    match limiter:
        case "barth_jespersen":
            return jnp.clip(ratio, 0.0, 1.0)
        case "venkatakrishnan":
            numerator = ratio**2 + 2.0 * ratio + epsilon
            denominator = ratio**2 + ratio + 2.0 + epsilon
            return jnp.clip(numerator / denominator, 0.0, 1.0)
        case "unlimited":
            return jnp.ones_like(delta)
        case _:
            assert_never(limiter)


class TriangleMUSCLReconstructionPlan(StrictModule, NonTrainableState):
    gradient: PreparedTriangleWLSQ
    limiter: TriangleLimiterKind = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gradient: PreparedTriangleWLSQ,
        /,
        *,
        limiter: TriangleLimiterKind = "venkatakrishnan",
        epsilon: float = 1e-12,
    ) -> None:
        if not isinstance(gradient, PreparedTriangleWLSQ):
            raise TypeError("gradient must be PreparedTriangleWLSQ.")
        limiter = parse(limiter, TriangleLimiterKind, "limiter")
        self.gradient = gradient
        self.limiter = limiter
        self.epsilon = float(epsilon)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "triangle-muscl",
                "gradient": gradient.prepared_id,
                "limiter": limiter,
                "epsilon": float(epsilon),
            }
        )

    def reconstruct(self, state: Array, /) -> tuple[Array, Array]:
        discretization = self.gradient.discretization
        value = jnp.asarray(state)
        gradient = self.gradient.gradient(value)
        owner = discretization.owner_cells
        neighbor = discretization.neighbor_cells
        safe_neighbor = jnp.maximum(neighbor, 0)
        centers = discretization.cell_centers.astype(value.dtype)
        face_centers = discretization.face_centers.astype(value.dtype)
        owner_offset = face_centers - centers[owner]
        neighbor_offset = face_centers - centers[safe_neighbor]
        owner_delta = ein.contract("f...i,fi->f...", gradient[owner], owner_offset)
        neighbor_delta = ein.contract(
            "f...i,fi->f...",
            gradient[safe_neighbor],
            neighbor_offset,
        )
        if self.limiter == "unlimited":
            owner_factor = jnp.ones(
                (discretization.cell_count,) + value.shape[1:], dtype=value.dtype
            )
        else:
            gathered = value[self.gradient.neighbor_cells]
            mask = self.gradient.valid.reshape(
                self.gradient.valid.shape + (1,) * (gathered.ndim - 2)
            )
            minimum = jnp.min(jnp.where(mask, gathered, value[:, None, ...]), axis=1)
            maximum = jnp.max(jnp.where(mask, gathered, value[:, None, ...]), axis=1)

            def factors(cell_values: Array, delta: Array, cell_indices: Array) -> Array:
                return _limiter_factor(
                    self.limiter,
                    self.epsilon,
                    delta,
                    maximum[cell_indices] - cell_values[cell_indices],
                    minimum[cell_indices] - cell_values[cell_indices],
                )

            owner_face_factor = factors(value, owner_delta, owner)
            neighbor_face_factor = factors(value, neighbor_delta, safe_neighbor)
            owner_factor = jnp.ones(
                (discretization.cell_count,) + value.shape[1:], dtype=value.dtype
            )
            owner_factor = owner_factor.at[owner].min(owner_face_factor)
            neighbor_mask = (neighbor >= 0).reshape((-1,) + (1,) * (value.ndim - 1))
            owner_factor = owner_factor.at[safe_neighbor].min(
                jnp.where(neighbor_mask, neighbor_face_factor, 1.0)
            )
        left = value[owner] + owner_factor[owner] * owner_delta
        right = value[safe_neighbor] + owner_factor[safe_neighbor] * neighbor_delta
        return left, right

    def cell_limiter_factors(self, state: Array, cell_routes: Array, /) -> Array:
        """Limiter factors of the selected cells, `(routes, ...)`.

        A cell's factor is the minimum over its three edge-center increments,
        the same factor `reconstruct` applies on every face of the cell.
        """
        discretization = self.gradient.discretization
        value = jnp.asarray(state)
        routes = jnp.asarray(cell_routes, dtype=jnp.int32)
        base = value[routes]
        if self.limiter == "unlimited":
            return jnp.ones_like(base)
        gradient = self.gradient.cell_gradients(value, routes)
        centers = discretization.cell_centers.astype(value.dtype)[routes]
        edges = jnp.asarray(discretization.connectivity.cell_edges[:, :3])[routes]
        offsets = (
            discretization.face_centers.astype(value.dtype)[edges] - centers[:, None]
        )
        delta = ein.contract("r...i,rei->re...", gradient, offsets)
        gathered = value[self.gradient.neighbor_cells[routes]]
        valid = self.gradient.valid[routes]
        mask = valid.reshape(valid.shape + (1,) * (gathered.ndim - 2))
        minimum = jnp.min(jnp.where(mask, gathered, base[:, None, ...]), axis=1)
        maximum = jnp.max(jnp.where(mask, gathered, base[:, None, ...]), axis=1)
        factor = _limiter_factor(
            self.limiter,
            self.epsilon,
            delta,
            (maximum - base)[:, None, ...],
            (minimum - base)[:, None, ...],
        )
        return jnp.min(factor, axis=1)

    def evaluate_cells(self, state: Array, cell_routes: Array, points: Array, /) -> Array:
        """Limited linear reconstruction of the selected cells at their own points.

        `points` has shape `(routes, points, 2)`; the result has shape
        `(routes, points, ...)` and equals `reconstruct` at face centers.
        """
        discretization = self.gradient.discretization
        value = jnp.asarray(state)
        routes = jnp.asarray(cell_routes, dtype=jnp.int32)
        centers = discretization.cell_centers.astype(value.dtype)[routes]
        offsets = jnp.asarray(points, dtype=value.dtype) - centers[:, None, :]
        delta = ein.contract(
            "r...i,rqi->rq...", self.gradient.cell_gradients(value, routes), offsets
        )
        factor = self.cell_limiter_factors(value, routes)
        return value[routes, None, ...] + factor[:, None, ...] * delta


__all__ = [
    "PreparedTriangleWLSQ",
    "TriangleLimiterKind",
    "TriangleMUSCLReconstructionPlan",
    "TriangleWLSQReport",
]
