#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .fem import PreparedFiniteElementCellMap


class CellLocationStatus(IntEnum):
    LOCATED = 0
    OUTSIDE = 1
    DEGENERATE_CELL = 2
    NONFINITE = 3
    RESOURCE_EXCEEDED = 4
    INVERSE_MAP_EXHAUSTED = 5


class SimplicialLocationPolicy(StrictModule, NonTrainableState):
    maximum_candidates: int = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    maximum_seeds: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    reference_tolerance: float = eqx.field(static=True)
    trust_radius: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_candidates: int,
        maximum_iterations: int,
        maximum_seeds: int,
        /,
        *,
        residual_tolerance: float = 1.0e-10,
        reference_tolerance: float = 1.0e-10,
        trust_radius: float = 0.5,
    ):
        capacities = (
            int(maximum_candidates),
            int(maximum_iterations),
            int(maximum_seeds),
        )
        if any(value < 1 for value in capacities):
            raise ValueError("Simplicial location capacities must be positive.")
        if residual_tolerance <= 0.0 or reference_tolerance < 0.0 or trust_radius <= 0.0:
            raise ValueError("Simplicial location tolerances are invalid.")
        self.maximum_candidates, self.maximum_iterations, self.maximum_seeds = capacities
        self.residual_tolerance = float(residual_tolerance)
        self.reference_tolerance = float(reference_tolerance)
        self.trust_radius = float(trust_radius)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "simplicial-location-policy",
                "maximum_candidates": capacities[0],
                "maximum_iterations": capacities[1],
                "maximum_seeds": capacities[2],
                "residual_tolerance": self.residual_tolerance,
                "reference_tolerance": self.reference_tolerance,
                "trust_radius": self.trust_radius,
            }
        )


class CellLocationResult(StrictModule):
    cell_ids: Array
    reference_coordinates: Array
    barycentric: Array
    geometry_residual: Array
    iterations: Array
    jacobian_condition: Array
    inside: Array
    used_fallback: Array
    candidate_count: Array
    status: Array
    successful: Array
    locator_id: str = eqx.field(static=True)


class SegmentLocationResult(StrictModule):
    start: CellLocationResult
    end: CellLocationResult
    crossed: Array
    exited: Array
    successful: Array
    locator_id: str = eqx.field(static=True)


class PreparedSimplicialCellLocator(StrictModule, NonTrainableState):
    """Bounded damped-Newton locator over a canonical prepared FE cell map."""

    cell_map: PreparedFiniteElementCellMap
    coordinates: Array
    cells: Array
    centroids: Array
    policy: SimplicialLocationPolicy
    locator_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_map: PreparedFiniteElementCellMap,
        coordinates: ArrayLike,
        policy: SimplicialLocationPolicy,
        /,
    ):
        if not isinstance(cell_map, PreparedFiniteElementCellMap):
            raise TypeError("cell_map must be PreparedFiniteElementCellMap.")
        if cell_map.coordinate_element.cell_kind not in ("triangle", "tetrahedron"):
            raise ValueError("Simplicial locator requires a triangle/tetrahedron map.")
        if not isinstance(policy, SimplicialLocationPolicy):
            raise TypeError("policy must be SimplicialLocationPolicy.")
        values = jnp.asarray(coordinates)
        if values.shape != (cell_map.coordinate_count, cell_map.ambient_dimension):
            raise ValueError("Locator coordinates do not match the prepared cell map.")
        cells = cell_map.coordinate_dofs
        centroids = jnp.mean(values[cells], axis=1)
        self.cell_map = cell_map
        self.coordinates = values
        self.cells = cells
        self.centroids = centroids
        self.policy = policy
        self.locator_id = canonical_fingerprint(
            {
                "kind": "prepared-simplicial-cell-locator",
                "cell_map": cell_map.cell_map_id,
                "coordinates": array_tree_fingerprint(values),
                "policy": policy.policy_id,
            }
        )

    @property
    def dimension(self) -> int:
        return self.cell_map.reference_dimension

    @property
    def cell_count(self) -> int:
        return self.cell_map.cell_count

    @property
    def coordinate_count(self) -> int:
        return self.cell_map.coordinate_count

    def locate(self, points: ArrayLike, /) -> CellLocationResult:
        values = jnp.asarray(points, dtype=self.coordinates.dtype)
        if values.ndim != 2 or values.shape[1] != self.cell_map.ambient_dimension:
            raise ValueError("Locator points have incompatible ambient dimension.")
        point_count = values.shape[0]
        candidate_capacity = min(self.policy.maximum_candidates, self.cell_count)
        distances = jnp.sum(
            (values[:, None, :] - self.centroids[None, :, :]) ** 2, axis=-1
        )
        candidates = jnp.argsort(distances, axis=1)[:, :candidate_capacity]
        reference_dimension = self.dimension
        centroid_seed = jnp.full(
            (reference_dimension,), 1.0 / (reference_dimension + 1), dtype=values.dtype
        )
        reference_nodes = self.cell_map.coordinate_element.reference_nodes.astype(
            values.dtype
        )
        seed_pool = jnp.concatenate((centroid_seed[None], reference_nodes), axis=0)
        seed_count = min(self.policy.maximum_seeds, int(seed_pool.shape[0]))
        seeds = seed_pool[:seed_count]
        reference = jnp.broadcast_to(
            seeds[None, None, :, :],
            (point_count, candidate_capacity, seed_count, reference_dimension),
        )
        flat_cells = jnp.broadcast_to(
            candidates[:, :, None], (point_count, candidate_capacity, seed_count)
        ).reshape((-1,))
        targets = jnp.broadcast_to(
            values[:, None, None, :],
            (point_count, candidate_capacity, seed_count, values.shape[1]),
        ).reshape((-1, values.shape[1]))
        converged = jnp.zeros((flat_cells.size,), dtype=bool)
        first_iteration = jnp.zeros((flat_cells.size,), dtype=jnp.int32)
        residual_norm = jnp.full((flat_cells.size,), jnp.inf, dtype=values.dtype)
        condition = jnp.full((flat_cells.size,), jnp.inf, dtype=values.dtype)
        reference_flat = reference.reshape((-1, reference_dimension))
        ever_valid_geometry = jnp.zeros_like(converged)
        for iteration in range(self.policy.maximum_iterations):
            evaluation = self.cell_map.evaluate(
                self.coordinates, flat_cells, reference_flat
            )
            residual = evaluation.physical_points - targets
            residual_norm = jnp.sqrt(jnp.sum(residual**2, axis=-1))
            delta = contract("qrd,qd->qr", evaluation.inverse_jacobian, residual)
            delta_norm = jnp.sqrt(jnp.sum(delta**2, axis=-1))
            scale = jnp.minimum(
                1.0,
                self.policy.trust_radius / jnp.maximum(delta_norm, 1.0e-30),
            )
            candidate_reference = reference_flat - scale[:, None] * delta
            newly = (
                (~converged)
                & evaluation.valid
                & (residual_norm <= self.policy.residual_tolerance)
            )
            first_iteration = jnp.where(newly, iteration + 1, first_iteration)
            converged = converged | newly
            reference_flat = jnp.where(
                converged[:, None], reference_flat, candidate_reference
            )
            ever_valid_geometry = ever_valid_geometry | evaluation.valid
        evaluation = self.cell_map.evaluate(self.coordinates, flat_cells, reference_flat)
        residual_norm = jnp.sqrt(
            jnp.sum((evaluation.physical_points - targets) ** 2, axis=-1)
        )
        final_converged = evaluation.valid & (
            residual_norm <= self.policy.residual_tolerance
        )
        first_iteration = jnp.where(
            (~converged) & final_converged,
            self.policy.maximum_iterations,
            first_iteration,
        )
        converged = converged | final_converged
        ever_valid_geometry = ever_valid_geometry | evaluation.valid
        jacobian_norm = jnp.sqrt(jnp.sum(evaluation.jacobian**2, axis=(-2, -1)))
        inverse_norm = jnp.sqrt(jnp.sum(evaluation.inverse_jacobian**2, axis=(-2, -1)))
        condition = jacobian_norm * inverse_norm
        inside_reference = jnp.all(
            reference_flat >= -self.policy.reference_tolerance, axis=-1
        ) & (jnp.sum(reference_flat, axis=-1) <= 1.0 + self.policy.reference_tolerance)
        accepted = converged & evaluation.valid & inside_reference
        accepted = accepted.reshape((point_count, candidate_capacity, seed_count))
        reference_all = reference_flat.reshape(
            (point_count, candidate_capacity, seed_count, reference_dimension)
        )
        residual_all = residual_norm.reshape(
            (point_count, candidate_capacity, seed_count)
        )
        condition_all = condition.reshape((point_count, candidate_capacity, seed_count))
        iteration_all = first_iteration.reshape(
            (point_count, candidate_capacity, seed_count)
        )
        stable_cells = jnp.where(accepted, candidates[:, :, None], self.cell_count)
        flat_choice = jnp.argmin(stable_cells.reshape((point_count, -1)), axis=1)
        candidate_choice = flat_choice // seed_count
        seed_choice = flat_choice % seed_count
        rows = jnp.arange(point_count)
        inside = jnp.any(accepted, axis=(1, 2))
        cell_ids = jnp.where(inside, candidates[rows, candidate_choice], -1)
        reference_result = reference_all[rows, candidate_choice, seed_choice]
        residual_result = residual_all[rows, candidate_choice, seed_choice]
        condition_result = condition_all[rows, candidate_choice, seed_choice]
        iteration_result = iteration_all[rows, candidate_choice, seed_choice]
        barycentric = jnp.concatenate(
            ((1.0 - jnp.sum(reference_result, axis=-1))[:, None], reference_result),
            axis=-1,
        )
        converged_valid = converged.reshape(
            (point_count, candidate_capacity, seed_count)
        ) & evaluation.valid.reshape((point_count, candidate_capacity, seed_count))
        any_valid_geometry = jnp.any(
            ever_valid_geometry.reshape((point_count, candidate_capacity, seed_count)),
            axis=(1, 2),
        )
        outside_domain = (~inside) & jnp.any(converged_valid, axis=(1, 2))
        finite = jnp.all(jnp.isfinite(values), axis=-1)
        candidate_exhausted = (
            (~inside)
            & ~outside_domain
            & any_valid_geometry
            & (candidate_capacity < self.cell_count)
        )
        degenerate = (~inside) & ~any_valid_geometry & finite
        status = jnp.where(
            ~finite,
            int(CellLocationStatus.NONFINITE),
            jnp.where(
                inside,
                int(CellLocationStatus.LOCATED),
                jnp.where(
                    outside_domain,
                    int(CellLocationStatus.OUTSIDE),
                    jnp.where(
                        degenerate,
                        int(CellLocationStatus.DEGENERATE_CELL),
                        jnp.where(
                            candidate_exhausted,
                            int(CellLocationStatus.RESOURCE_EXCEEDED),
                            int(CellLocationStatus.INVERSE_MAP_EXHAUSTED),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return CellLocationResult(
            cell_ids,
            jnp.where(inside[:, None], reference_result, 0.0),
            jnp.where(inside[:, None], barycentric, 0.0),
            jnp.where(inside, residual_result, jnp.inf),
            jnp.where(inside, iteration_result, self.policy.maximum_iterations),
            jnp.where(inside, condition_result, jnp.inf),
            inside,
            seed_choice != 0,
            jnp.sum(jnp.any(accepted, axis=2), axis=1, dtype=jnp.int32),
            status,
            inside & finite,
            self.locator_id,
        )

    def locate_segment(
        self, start: ArrayLike, end: ArrayLike, /
    ) -> SegmentLocationResult:
        left = self.locate(start)
        right = self.locate(end)
        crossed = left.inside & right.inside & (left.cell_ids != right.cell_ids)
        exited = left.inside & ~right.inside
        successful = left.successful & (right.successful | exited)
        return SegmentLocationResult(
            left, right, crossed, exited, successful, self.locator_id
        )


__all__ = [
    "CellLocationResult",
    "CellLocationStatus",
    "PreparedSimplicialCellLocator",
    "SegmentLocationResult",
    "SimplicialLocationPolicy",
]
