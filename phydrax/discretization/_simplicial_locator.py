#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import IntEnum
from typing import final, Literal, Protocol, runtime_checkable, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._bvh import BVHBuildPolicy, PackedBVH, point_select_leaf_items, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import Bool, checked, Dim, Float, Int32, Integer
from ._cell_geometry import CellGeometryElement, RestrictedCellGeometryElement
from ._coordinate_enclosure import coordinate_polynomials, polynomial_bounds
from ._reference_cell import reference_cell_topology
from .fem._cell_map import FiniteElementCellMapEvaluation, PreparedFiniteElementCellMap
from .fem._reference import FiniteElementSpec


# (reference points, converged, first converged iteration, ever-valid geometry)
_NewtonCarry: TypeAlias = tuple[Array, Array, Array, Array]


class CellLocationStatus(IntEnum):
    LOCATED = 0
    OUTSIDE = 1
    DEGENERATE_CELL = 2
    NONFINITE = 3
    RESOURCE_EXCEEDED = 4
    INVERSE_MAP_EXHAUSTED = 5


@final
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
    ) -> None:
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


@final
class CellLocationResult(StrictModule):
    """Located cell, reference coordinates, and every containing candidate.

    `cell_ids` is the lowest-index containing cell (`-1` when none).
    `candidate_cells` lists each bounded candidate cell that contains the point
    (`-1` otherwise) with its `candidate_reference` coordinates, so consumers
    resolving non-smooth loci see every containing cell rather than one.
    `candidate_count` is the number of containing candidates.
    `candidates_complete` is false when BVH capacity or any candidate inverse
    was unresolved. A partial containing-candidate list is never certified.
    """

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
    candidate_cells: Array
    candidate_reference: Array
    candidates_complete: Array
    locator_id: str = eqx.field(static=True)


class SegmentPointDim(Dim):
    """Independently traversed trajectories."""


class SegmentSlotDim(Dim):
    """Fixed facet interval capacity."""


class LocatorCellDim(Dim):
    """Cells of the admitted coordinate map."""


class LocatorStarDim(Dim):
    """Prepared neighboring-cell tie capacity."""


@final
class SegmentLocationResult(StrictModule):
    __strict_contract__ = True

    start: CellLocationResult
    end: CellLocationResult
    crossed: Bool[SegmentPointDim]
    exited: Bool[SegmentPointDim]
    successful: Bool[SegmentPointDim]
    cell_ids: Integer[SegmentPointDim, SegmentSlotDim]
    intervals: Float[SegmentPointDim, SegmentSlotDim, Literal[2]]
    valid: Bool[SegmentPointDim, SegmentSlotDim]
    counts: Int32[SegmentPointDim]
    overflow: Bool[SegmentPointDim]
    tied: Bool[SegmentPointDim, SegmentSlotDim]
    locator_id: str = eqx.field(static=True)


@runtime_checkable
class LocatedCellMap(Protocol):
    """Common metadata and paired chart evaluation of a located cell block."""

    cell_count: int
    ambient_dimension: int
    reference_dimension: int
    coordinate_count: int
    topology_id: str
    geometry_layout_id: str
    cell_map_id: str
    block_name: str

    def evaluate(
        self,
        coordinates: ArrayLike,
        cell_indices: ArrayLike,
        reference_points: ArrayLike,
        /,
    ) -> FiniteElementCellMapEvaluation: ...


class AbstractCellLocator(StrictModule):
    """Inverse chart of one prepared cell block.

    `locate(points, cell_mask=None)` returns a `CellLocationResult` whose
    candidates are restricted to cells where `cell_mask` is true. Canonical FE
    maps and physical Cartesian polyhedral charts share the result contract,
    not a fabricated common finite-element descriptor.
    """

    cell_map: eqx.AbstractVar[LocatedCellMap]
    coordinates: eqx.AbstractVar[Array]
    locator_id: eqx.AbstractVar[str]

    @property
    def canonical_locator(self) -> AbstractCellLocator:
        """Return the whole-source locator independently of query-side selection."""
        return self

    @abc.abstractmethod
    def locate(
        self, points: ArrayLike, /, *, cell_mask: ArrayLike | None = None
    ) -> CellLocationResult:
        raise NotImplementedError


def _certified_cell_bounds(
    cell_map: PreparedFiniteElementCellMap, coordinates: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose each whole mapped cell from its exact coordinate source expression."""
    element = cell_map.coordinate_element
    source_coordinates = cell_map.source_coordinates(coordinates)
    cell_coordinates = tuple(
        tuple(source_coordinates[index] for index in route)
        for route in np.asarray(cell_map.coordinate_dofs)
    )
    lower = np.empty((cell_map.cell_count, cell_map.ambient_dimension), dtype=np.float64)
    upper = np.empty_like(lower)
    for cell, local in enumerate(cell_coordinates):
        polynomials = coordinate_polynomials(element, local)
        if polynomials is None:
            raise ValueError(
                "Simplicial locator requires a canonical coordinate source enclosure."
            )
        for axis, polynomial in enumerate(polynomials):
            lower[cell, axis], upper[cell, axis] = polynomial_bounds(
                polynomial, "simplex", cell_map.reference_dimension
            )
    return lower, upper


def _cell_map_vertices(cell_map: PreparedFiniteElementCellMap, /) -> np.ndarray:
    blocks = (
        cell_map.mesh.blocks
        if cell_map.block_index is None
        else (cell_map.mesh.blocks[cell_map.block_index],)
    )
    return np.concatenate(tuple(np.asarray(block.vertices) for block in blocks), axis=0)


def _is_affine_simplex_element(element: CellGeometryElement, /) -> bool:
    kind = element.cell_kind
    if kind not in ("interval", "triangle", "tetrahedron") and not kind.startswith(
        "simplex:"
    ):
        return False
    while isinstance(element, RestrictedCellGeometryElement):
        element = element.source_element
    return (
        isinstance(element, FiniteElementSpec)
        and element.degree == 1
        and (
            element.cell_kind in ("interval", "triangle", "tetrahedron")
            or element.cell_kind.startswith("simplex:")
        )
    )


class _AbstractPreparedNewtonCellLocator(AbstractCellLocator, NonTrainableState):
    """Shared bounded damped-Newton inverse for prepared coordinate maps."""

    __strict_contract__ = True

    cell_map: PreparedFiniteElementCellMap
    coordinates: Array
    cells: Array
    vertex_star_cells: Int32[LocatorCellDim, LocatorStarDim]
    vertex_star_valid: Bool[LocatorCellDim, LocatorStarDim]
    centroids: Array
    bvh: PackedBVH
    policy: SimplicialLocationPolicy
    locator_id: str = eqx.field(static=True)

    @property
    def dimension(self) -> int:
        return self.cell_map.reference_dimension

    @property
    def cell_count(self) -> int:
        return self.cell_map.cell_count

    @property
    def coordinate_count(self) -> int:
        return self.cell_map.coordinate_count

    def _reference_seeds(self, dtype: jnp.dtype, /) -> Array:
        center = jnp.full((self.dimension,), 1.0 / (self.dimension + 1), dtype=dtype)
        vertices = jnp.asarray(
            reference_cell_topology(self.cell_map.coordinate_element.cell_kind).vertices,
            dtype=dtype,
        )
        return jnp.concatenate((center[None], vertices), axis=0)

    def _inside_reference(self, reference: Array, /) -> Array:
        tolerance = self.policy.reference_tolerance
        return jnp.all(reference >= -tolerance, axis=-1) & (
            jnp.sum(reference, axis=-1) <= 1.0 + tolerance
        )

    def _reference_weights(self, reference: Array, /) -> Array:
        return jnp.concatenate(
            ((1.0 - jnp.sum(reference, axis=-1))[:, None], reference), axis=-1
        )

    @eqx.filter_jit
    def locate(
        self, points: ArrayLike, /, *, cell_mask: ArrayLike | None = None
    ) -> CellLocationResult:
        values = jnp.asarray(points, dtype=self.coordinates.dtype)
        if values.ndim != 2 or values.shape[1] != self.cell_map.ambient_dimension:
            raise ValueError("Locator points have incompatible ambient dimension.")
        if cell_mask is not None:
            mask = jnp.asarray(cell_mask)
            if mask.shape != (self.cell_count,) or mask.dtype != jnp.bool_:
                raise ValueError(
                    "cell_mask must be a boolean array with one entry per cell."
                )
        point_count = values.shape[0]
        candidate_capacity = min(self.policy.maximum_candidates, self.cell_count)
        candidates, candidate_valid, search_complete = point_select_leaf_items(
            values,
            bvh=self.bvh,
            maximum_candidates=candidate_capacity,
            tolerance=self.policy.reference_tolerance,
        )
        if cell_mask is not None:
            candidate_valid = (
                candidate_valid & mask[jnp.where(candidate_valid, candidates, 0)]
            )
        reference_dimension = self.dimension
        seed_pool = self._reference_seeds(values.dtype)
        seed_count = min(self.policy.maximum_seeds, seed_pool.shape[0])
        seeds = seed_pool[:seed_count]
        reference = jnp.broadcast_to(
            seeds[None, None, :, :],
            (point_count, candidate_capacity, seed_count, reference_dimension),
        )
        flat_cells = jnp.broadcast_to(
            candidates[:, :, None], (point_count, candidate_capacity, seed_count)
        ).reshape((-1,))
        flat_candidate_valid = jnp.broadcast_to(
            candidate_valid[:, :, None],
            (point_count, candidate_capacity, seed_count),
        ).reshape((-1,))
        targets = jnp.broadcast_to(
            values[:, None, None, :],
            (point_count, candidate_capacity, seed_count, values.shape[1]),
        ).reshape((-1, values.shape[1]))
        converged = jnp.zeros((flat_cells.size,), dtype=jnp.bool_)
        first_iteration = jnp.zeros((flat_cells.size,), dtype=jnp.int32)
        reference_flat = reference.reshape((-1, reference_dimension))
        ever_valid_geometry = jnp.zeros_like(converged)

        def newton_step(iteration: Array, carry: _NewtonCarry) -> _NewtonCarry:
            current_reference, current_converged, first, ever_valid = carry
            evaluation = self.cell_map.evaluate(
                self.coordinates,
                flat_cells,
                current_reference,
            )
            geometry_valid = evaluation.valid & flat_candidate_valid
            residual = evaluation.physical_points - targets
            residual_norm = jnp.sqrt(jnp.sum(residual**2, axis=-1))
            delta = contract("qrd,qd->qr", evaluation.inverse_jacobian, residual)
            delta_norm = jnp.sqrt(jnp.sum(delta**2, axis=-1))
            scale = jnp.minimum(
                1.0,
                self.policy.trust_radius / jnp.maximum(delta_norm, 1.0e-30),
            )
            candidate_reference = current_reference - scale[:, None] * delta
            newly = (
                (~current_converged)
                & geometry_valid
                & (residual_norm <= self.policy.residual_tolerance)
            )
            first = jnp.where(newly, iteration + 1, first)
            current_converged = current_converged | newly
            current_reference = jnp.where(
                current_converged[:, None],
                current_reference,
                candidate_reference,
            )
            return (
                current_reference,
                current_converged,
                first,
                ever_valid | geometry_valid,
            )

        (
            reference_flat,
            converged,
            first_iteration,
            ever_valid_geometry,
        ) = jax.lax.fori_loop(
            0,
            self.policy.maximum_iterations,
            newton_step,
            (
                reference_flat,
                converged,
                first_iteration,
                ever_valid_geometry,
            ),
        )
        evaluation = self.cell_map.evaluate(self.coordinates, flat_cells, reference_flat)
        residual_norm = jnp.sqrt(
            jnp.sum((evaluation.physical_points - targets) ** 2, axis=-1)
        )
        final_converged = (
            evaluation.valid
            & flat_candidate_valid
            & (residual_norm <= self.policy.residual_tolerance)
        )
        first_iteration = jnp.where(
            (~converged) & final_converged,
            self.policy.maximum_iterations,
            first_iteration,
        )
        converged = converged | final_converged
        ever_valid_geometry = ever_valid_geometry | (
            evaluation.valid & flat_candidate_valid
        )
        jacobian_norm = jnp.sqrt(jnp.sum(evaluation.jacobian**2, axis=(-2, -1)))
        inverse_norm = jnp.sqrt(jnp.sum(evaluation.inverse_jacobian**2, axis=(-2, -1)))
        condition = jacobian_norm * inverse_norm
        inside_reference = self._inside_reference(reference_flat)
        accepted = converged & evaluation.valid & flat_candidate_valid & inside_reference
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
        converged_valid = converged.reshape(
            (point_count, candidate_capacity, seed_count)
        ) & (evaluation.valid & flat_candidate_valid).reshape(
            (point_count, candidate_capacity, seed_count)
        )
        candidate_resolved = jnp.any(converged_valid, axis=2) | ~candidate_valid
        if (
            self.cell_map.ambient_dimension > self.cell_map.reference_dimension
            and _is_affine_simplex_element(self.cell_map.coordinate_element)
        ):
            # An affine source has a constant tangent space. Its orthogonal
            # residual excludes a point from the entire source image, including
            # when bounded Newton cannot converge to an off-manifold point.
            residual = evaluation.physical_points - targets
            tangent_delta = contract("qrd,qd->qr", evaluation.inverse_jacobian, residual)
            orthogonal = residual - contract(
                "qdr,qr->qd", evaluation.jacobian, tangent_delta
            )
            excluded = (
                evaluation.valid
                & flat_candidate_valid
                & (jnp.linalg.norm(orthogonal, axis=-1) > self.policy.residual_tolerance)
            ).reshape((point_count, candidate_capacity, seed_count))
            candidate_resolved = candidate_resolved | jnp.any(excluded, axis=2)
        inverses_complete = jnp.all(candidate_resolved, axis=1)
        stable_cells = jnp.where(accepted, candidates[:, :, None], self.cell_count)
        found = jnp.any(accepted, axis=(1, 2))
        accepted_choice = jnp.argmin(stable_cells.reshape((point_count, -1)), axis=1)
        diagnostic_choice = jnp.argmin(
            jnp.where(
                flat_candidate_valid.reshape(
                    (point_count, candidate_capacity, seed_count)
                ),
                residual_all,
                jnp.inf,
            ).reshape((point_count, -1)),
            axis=1,
        )
        flat_choice = jnp.where(found, accepted_choice, diagnostic_choice)
        candidate_choice = flat_choice // seed_count
        seed_choice = flat_choice % seed_count
        rows = jnp.arange(point_count)
        inside = found & search_complete & inverses_complete
        cell_ids = jnp.where(inside, candidates[rows, candidate_choice], -1)
        reference_result = reference_all[rows, candidate_choice, seed_choice]
        residual_result = residual_all[rows, candidate_choice, seed_choice]
        condition_result = condition_all[rows, candidate_choice, seed_choice]
        iteration_result = iteration_all[rows, candidate_choice, seed_choice]
        barycentric = self._reference_weights(reference_result)
        candidate_accepted = jnp.any(accepted, axis=2)
        candidate_seed = jnp.argmax(accepted, axis=2)
        candidate_reference = jnp.take_along_axis(
            reference_all, candidate_seed[:, :, None, None], axis=2
        )[:, :, 0, :]
        candidate_cells = jnp.where(candidate_accepted, candidates, -1)
        any_valid_geometry = jnp.any(
            ever_valid_geometry.reshape((point_count, candidate_capacity, seed_count)),
            axis=(1, 2),
        )
        has_candidates = jnp.any(candidate_valid, axis=1)
        outside_domain = (
            (~inside) & search_complete & ((~has_candidates) | inverses_complete)
        )
        finite = jnp.all(jnp.isfinite(values), axis=-1)
        candidate_exhausted = (~inside) & ~search_complete
        degenerate = (
            (~inside) & has_candidates & ~any_valid_geometry & finite & search_complete
        )
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
            jnp.where(has_candidates, residual_result, jnp.inf),
            jnp.where(
                iteration_result > 0, iteration_result, self.policy.maximum_iterations
            ),
            jnp.where(has_candidates, condition_result, jnp.inf),
            inside,
            seed_choice != 0,
            jnp.sum(candidate_accepted, axis=1, dtype=jnp.int32),
            status,
            inside & finite,
            candidate_cells,
            jnp.where(candidate_accepted[:, :, None], candidate_reference, 0.0),
            finite & search_complete & inverses_complete,
            self.locator_id,
        )

    def _affine_geometry(self) -> tuple[Array, Array, Array, Array]:
        if not _is_affine_simplex_element(self.cell_map.coordinate_element):
            raise ValueError("Exact facet traversal requires an affine simplex cell map.")
        reference = jnp.zeros(
            (self.cell_count, self.dimension), dtype=self.coordinates.dtype
        )
        geometry = self.cell_map.evaluate(
            self.coordinates, jnp.arange(self.cell_count, dtype=jnp.int32), reference
        )
        gradients = jnp.concatenate(
            (
                -jnp.sum(geometry.inverse_jacobian, axis=1, keepdims=True),
                geometry.inverse_jacobian,
            ),
            axis=1,
        )
        return geometry.physical_points, gradients, geometry.valid, geometry.jacobian

    def affine_gradients(self) -> Array:
        """Return the single owning barycentric-gradient table, cell/local/ambient."""
        return self._affine_geometry()[1]

    def affine_barycentric(self, points: ArrayLike, /) -> Array:
        """Evaluate all affine cells, with shape (point, cell, local vertex)."""
        values = jnp.asarray(points, dtype=self.coordinates.dtype)
        if values.ndim != 2 or values.shape[1] != self.cell_map.ambient_dimension:
            raise ValueError("Affine points have incompatible ambient dimension.")
        origins, gradients, _, _ = self._affine_geometry()
        coordinates = contract("pcd,cvd->pcv", values[:, None] - origins[None], gradients)
        return coordinates.at[:, :, 0].add(1.0)

    def locate_segment(
        self,
        start: ArrayLike,
        end: ArrayLike,
        /,
        *,
        maximum_segments: int | None = None,
    ) -> SegmentLocationResult:
        """Walk exact affine facet intervals, choosing the lowest cell on ties.

        Prepared vertex stars admit face, edge, and vertex crossings without
        allocating point-by-mesh search tables. Gaps terminate at the first
        domain exit rather than joining disconnected pieces. No uniform
        temporal subdivision or endpoint-only admission is used.
        """
        capacity = self.cell_count if maximum_segments is None else maximum_segments
        if capacity < 1:
            raise ValueError("maximum_segments must be positive.")
        first = jnp.asarray(start, dtype=self.coordinates.dtype)
        last = jnp.asarray(end, dtype=first.dtype)
        if first.shape != last.shape or first.ndim != 2:
            raise ValueError("Segment endpoints must be matching rank-two arrays.")
        if first.shape[1] != self.cell_map.ambient_dimension:
            raise ValueError("Segment endpoints have incompatible ambient dimension.")
        origins, gradients, geometry_valid, jacobians = self._affine_geometry()
        left = self.locate(first)
        right = self.locate(last)
        cell_ids, intervals, valid, tied, cursor, current = _walk_facet_intervals(
            first,
            last,
            origins,
            gradients,
            jacobians,
            geometry_valid,
            self.vertex_star_cells,
            self.vertex_star_valid,
            left.cell_ids,
            capacity,
        )
        choice, _, _ = _outgoing_affine_cell(
            first,
            last,
            origins,
            gradients,
            jacobians,
            geometry_valid,
            self.vertex_star_cells,
            self.vertex_star_valid,
            current,
            cursor,
        )
        counts = jnp.sum(valid, axis=1, dtype=jnp.int32)
        started = counts > 0
        overflow = started & (cursor < 1) & (choice >= 0)
        exited = started & (cursor < 1) & ~overflow
        finite = jnp.all(jnp.isfinite(first) & jnp.isfinite(last), axis=1)
        successful = started & finite & ~overflow
        return SegmentLocationResult(
            left,
            right,
            counts > 1,
            exited,
            successful,
            cell_ids,
            intervals,
            valid,
            counts,
            overflow,
            tied,
            self.locator_id,
        )


def _vertex_star_routes(cells: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Prepare sorted local tie candidates; any facet crossing stays in this star."""
    incident: dict[int, set[int]] = {}
    for cell_index, vertices in enumerate(cells.tolist()):
        for vertex in vertices:
            incident.setdefault(vertex, set()).add(cell_index)
    stars = [
        sorted(set().union(*(incident[vertex] for vertex in vertices)))
        for vertices in cells.tolist()
    ]
    capacity = max(map(len, stars))
    routes = np.zeros((cells.shape[0], capacity), dtype=np.int32)
    valid = np.zeros(routes.shape, dtype=np.bool_)
    for cell_index, star in enumerate(stars):
        routes[cell_index, : len(star)] = star
        valid[cell_index, : len(star)] = True
    return routes, valid


def _affine_segment_intervals(
    barycentric: Array, slopes: Array, geometry_valid: Array, /
) -> tuple[Array, Array, Array]:
    """Intersect a line with affine barycentric half-spaces without perturbing t."""
    nonzero = slopes != 0
    roots = -barycentric / jnp.where(nonzero, slopes, 1)
    enter = jnp.maximum(0, jnp.max(jnp.where(slopes > 0, roots, -jnp.inf), axis=2))
    leave = jnp.minimum(1, jnp.min(jnp.where(slopes < 0, roots, jnp.inf), axis=2))
    tolerance = 32 * jnp.finfo(barycentric.dtype).eps
    parallel_valid = jnp.all(nonzero | (barycentric >= -tolerance), axis=2)
    finite = jnp.all(jnp.isfinite(barycentric) & jnp.isfinite(slopes), axis=2)
    valid = geometry_valid & parallel_valid & finite & (leave > enter)
    return enter, leave, valid


def _outgoing_affine_cell(
    start: Array,
    end: Array,
    origins: Array,
    gradients: Array,
    jacobians: Array,
    geometry_valid: Array,
    star_cells: Array,
    star_valid: Array,
    current: Array,
    time: Array,
    /,
) -> tuple[Array, Array, Array]:
    safe = jnp.clip(current, 0, origins.shape[0] - 1)
    candidates = star_cells[safe]
    candidate_valid = star_valid[safe] & (current[:, None] >= 0)
    local_gradients = gradients[candidates]
    barycentric = (
        contract("psd,psvd->psv", start[:, None] - origins[candidates], local_gradients)
        .at[:, :, 0]
        .add(1.0)
    )
    slopes = contract("pd,psvd->psv", end - start, local_gradients)
    if start.shape[1] != gradients.shape[1] - 1:
        local_jacobians = jacobians[candidates]
        projected_start = origins[candidates] + contract(
            "psdn,psn->psd", local_jacobians, barycentric[:, :, 1:]
        )
        projected_direction = contract("psdn,psn->psd", local_jacobians, slopes[:, :, 1:])
        tolerance = 32 * jnp.finfo(start.dtype).eps
        scale = jnp.maximum(1, jnp.max(jnp.abs(start), axis=1))
        candidate_valid = (
            candidate_valid
            & (
                jnp.max(jnp.abs(projected_start - start[:, None]), axis=2)
                <= tolerance * scale[:, None]
            )
            & (
                jnp.max(jnp.abs(projected_direction - (end - start)[:, None]), axis=2)
                <= tolerance * scale[:, None]
            )
        )
    enter, leave, intersects = _affine_segment_intervals(
        barycentric, slopes, geometry_valid[candidates] & candidate_valid
    )
    tolerance = 32 * jnp.finfo(start.dtype).eps
    outgoing = intersects & (enter <= time[:, None] + tolerance) & (leave > time[:, None])
    slot = jnp.argmin(jnp.where(outgoing, candidates, origins.shape[0]), axis=1)
    rows = jnp.arange(start.shape[0])
    found = jnp.any(outgoing, axis=1) & (time < 1)
    choice = jnp.where(found, candidates[rows, slot], -1)
    stop = jnp.where(found, leave[rows, slot], time)
    endpoint = barycentric[rows, slot] + stop[:, None] * slopes[rows, slot]
    facet_tie = jnp.sum(jnp.abs(endpoint) <= tolerance, axis=1) > 1
    tied = found & (facet_tie | (jnp.sum(outgoing, axis=1) > 1))
    return choice, stop, tied


def _walk_facet_intervals(
    start: Array,
    end: Array,
    origins: Array,
    gradients: Array,
    jacobians: Array,
    geometry_valid: Array,
    star_cells: Array,
    star_valid: Array,
    initial_cells: Array,
    capacity: int,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Bounded exact walk with deterministic lowest-index outgoing-cell ties."""
    point_count = start.shape[0]
    ids = jnp.full((point_count, capacity), -1, dtype=jnp.int32)
    intervals = jnp.zeros((point_count, capacity, 2), dtype=start.dtype)
    valid = jnp.zeros((point_count, capacity), dtype=jnp.bool_)
    tied = jnp.zeros_like(valid)
    cursor = jnp.zeros((point_count,), dtype=start.dtype)

    def advance(
        slot: int, carry: tuple[Array, Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        routes, spans, active, ties, time, current = carry
        choice, stop, facet_tie = _outgoing_affine_cell(
            start,
            end,
            origins,
            gradients,
            jacobians,
            geometry_valid,
            star_cells,
            star_valid,
            current,
            time,
        )
        admitted = choice >= 0
        routes = routes.at[:, slot].set(choice)
        spans = spans.at[:, slot].set(
            jnp.stack(
                (jnp.where(admitted, time, 0), jnp.where(admitted, stop, 0)), axis=1
            )
        )
        active = active.at[:, slot].set(admitted)
        ties = ties.at[:, slot].set(facet_tie)
        return routes, spans, active, ties, stop, jnp.where(admitted, choice, current)

    return jax.lax.fori_loop(
        0, capacity, advance, (ids, intervals, valid, tied, cursor, initial_cells)
    )


@final
class PreparedSimplicialCellLocator(_AbstractPreparedNewtonCellLocator):
    """Arbitrary-dimensional simplex inverse with exact whole-source BVH enclosures."""

    @checked
    def __init__(
        self,
        cell_map: PreparedFiniteElementCellMap,
        coordinates: ArrayLike,
        policy: SimplicialLocationPolicy,
        /,
    ) -> None:
        kind = cell_map.coordinate_element.cell_kind
        if kind not in ("interval", "triangle", "tetrahedron") and not kind.startswith(
            "simplex:"
        ):
            raise ValueError("Simplicial locator requires a simplex coordinate map.")
        values = jnp.asarray(coordinates)
        if values.shape != (cell_map.coordinate_count, cell_map.ambient_dimension):
            raise ValueError("Locator coordinates do not match the prepared cell map.")
        cells = cell_map.coordinate_dofs
        lower, upper = _certified_cell_bounds(cell_map, np.asarray(values))
        lower = np.asarray(
            jnp.nextafter(jnp.asarray(lower, dtype=values.dtype), -jnp.inf)
        )
        upper = np.asarray(jnp.nextafter(jnp.asarray(upper, dtype=values.dtype), jnp.inf))
        bvh = prepare_bvh(
            lower,
            upper,
            policy=BVHBuildPolicy(leaf_size=min(16, cell_map.cell_count)),
            dtype=values.dtype,
        )
        center = jnp.full(
            (cell_map.cell_count, cell_map.reference_dimension),
            1.0 / (cell_map.reference_dimension + 1),
            dtype=values.dtype,
        )
        centroids = cell_map.evaluate(
            values, jnp.arange(cell_map.cell_count), center
        ).physical_points
        if _is_affine_simplex_element(cell_map.coordinate_element):
            star_cells, star_valid = _vertex_star_routes(_cell_map_vertices(cell_map))
        else:
            star_cells = np.zeros((cell_map.cell_count, 1), dtype=np.int32)
            star_valid = np.zeros(star_cells.shape, dtype=np.bool_)
        self.cell_map = cell_map
        self.coordinates = values
        self.cells = cells
        self.vertex_star_cells = jnp.asarray(star_cells)
        self.vertex_star_valid = jnp.asarray(star_valid)
        self.centroids = centroids
        self.bvh = bvh
        self.policy = policy
        self.locator_id = canonical_fingerprint(
            {
                "kind": "prepared-simplicial-cell-locator",
                "cell_map": cell_map.cell_map_id,
                "coordinates": array_tree_fingerprint(values),
                "policy": policy.policy_id,
            }
        )


__all__ = [
    "AbstractCellLocator",
    "CellLocationResult",
    "CellLocationStatus",
    "LocatedCellMap",
    "PreparedSimplicialCellLocator",
    "SegmentLocationResult",
    "SimplicialLocationPolicy",
]
