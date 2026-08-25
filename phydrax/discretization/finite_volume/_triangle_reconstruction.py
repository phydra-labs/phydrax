#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import deque
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._triangle_fv import TriangleFiniteVolumeDiscretization


TriangleLimiterKind: TypeAlias = Literal[
    "unlimited", "barth_jespersen", "venkatakrishnan"
]


def _cell_neighbour_stencils(
    owner: np.ndarray,
    neighbour: np.ndarray,
    cell_count: int,
    centers: np.ndarray,
    /,
):
    adjacency = [set() for _ in range(cell_count)]
    for left, right in zip(owner, neighbour, strict=True):
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
    valid = np.zeros((cell_count, capacity), dtype=bool)
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
    neighbour_cells: Array
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
    ):
        if not isinstance(discretization, TriangleFiniteVolumeDiscretization):
            raise TypeError("WLSQ requires triangular finite-volume geometry.")
        centers = np.asarray(discretization.cell_centers)
        indices, valid = _cell_neighbour_stencils(
            np.asarray(discretization.owner_cells),
            np.asarray(discretization.neighbour_cells),
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
        pseudoinverse = np.einsum("cij,cnj->cin", scaled_right_vectors, left_vectors)
        factors = pseudoinverse * square_root_weights[:, None, :]
        report = TriangleWLSQReport(
            maximum_condition_number=jnp.asarray(np.max(condition)),
            worst_cell=jnp.asarray(np.argmax(condition), dtype=jnp.int32),
            stencil_capacity=indices.shape[1],
        )
        self.discretization = discretization
        self.neighbour_cells = jnp.asarray(indices)
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
        neighbours = value[self.neighbour_cells]
        difference = neighbours - value[:, None, ...]
        mask = self.valid.reshape(self.valid.shape + (1,) * (difference.ndim - 2))
        difference = jnp.where(mask, difference, 0.0)
        return oe.contract(
            "cin,cn...->c...i",
            self.factors.astype(value.dtype),
            difference,
        )


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
    ):
        if not isinstance(gradient, PreparedTriangleWLSQ):
            raise TypeError("gradient must be PreparedTriangleWLSQ.")
        if limiter not in ("unlimited", "barth_jespersen", "venkatakrishnan"):
            raise ValueError("Unknown triangle MUSCL limiter.")
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
        neighbour = discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        centers = discretization.cell_centers.astype(value.dtype)
        face_centers = discretization.face_centers.astype(value.dtype)
        owner_offset = face_centers - centers[owner]
        neighbour_offset = face_centers - centers[safe_neighbour]
        owner_delta = oe.contract("f...i,fi->f...", gradient[owner], owner_offset)
        neighbour_delta = oe.contract(
            "f...i,fi->f...",
            gradient[safe_neighbour],
            neighbour_offset,
        )
        if self.limiter == "unlimited":
            owner_factor = jnp.ones(
                (discretization.cell_count,) + value.shape[1:], dtype=value.dtype
            )
        else:
            gathered = value[self.gradient.neighbour_cells]
            mask = self.gradient.valid.reshape(
                self.gradient.valid.shape + (1,) * (gathered.ndim - 2)
            )
            minimum = jnp.min(jnp.where(mask, gathered, value[:, None, ...]), axis=1)
            maximum = jnp.max(jnp.where(mask, gathered, value[:, None, ...]), axis=1)

            def factors(cell_values, delta, cell_indices):
                upper = maximum[cell_indices] - cell_values[cell_indices]
                lower = minimum[cell_indices] - cell_values[cell_indices]
                allowed = jnp.where(delta >= 0.0, upper, lower)
                ratio = allowed / jnp.where(jnp.abs(delta) > self.epsilon, delta, 1.0)
                if self.limiter == "barth_jespersen":
                    return jnp.clip(ratio, 0.0, 1.0)
                numerator = ratio**2 + 2.0 * ratio + self.epsilon
                denominator = ratio**2 + ratio + 2.0 + self.epsilon
                return jnp.clip(numerator / denominator, 0.0, 1.0)

            owner_face_factor = factors(value, owner_delta, owner)
            neighbour_face_factor = factors(value, neighbour_delta, safe_neighbour)
            owner_factor = jnp.ones(
                (discretization.cell_count,) + value.shape[1:], dtype=value.dtype
            )
            owner_factor = owner_factor.at[owner].min(owner_face_factor)
            neighbour_mask = (neighbour >= 0).reshape((-1,) + (1,) * (value.ndim - 1))
            owner_factor = owner_factor.at[safe_neighbour].min(
                jnp.where(neighbour_mask, neighbour_face_factor, 1.0)
            )
        left = value[owner] + owner_factor[owner] * owner_delta
        right = value[safe_neighbour] + owner_factor[safe_neighbour] * neighbour_delta
        return left, right


__all__ = [
    "PreparedTriangleWLSQ",
    "TriangleLimiterKind",
    "TriangleMUSCLReconstructionPlan",
    "TriangleWLSQReport",
]
