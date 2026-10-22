#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import IntEnum
from typing import final, Literal, Protocol, runtime_checkable, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike

from phydrax.discretization._reference_cell_types import CellGeometryElement
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from phydrax.ein import contract

from .._bvh import BVHBuildPolicy, PackedBVH, point_select_leaf_items, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import Bool, checked, Dim, Float, Int32, Integer
from ._cell_geometry import (
    cell_geometry_preparation_bounds,
    cell_geometry_preparation_query_upper,
    owned_coordinate_preparation,
    RestrictedCellGeometryElement,
)
from ._coordinate_budget import CoordinateEnclosureBudget
from ._coordinate_enclosure import (
    coordinate_bernstein_estimate,
    coordinate_map_expression_estimates,
    coordinate_polynomials,
    coordinate_polynomials_estimate,
    polynomial_bounds,
)
from ._reference_cell import reference_cell_topology
from .fem._cell_map import (
    FiniteElementCellMapEvaluation,
    NumericalFiniteElementCellMap,
    PreparedFiniteElementCellMap,
)


# (reference points, converged, first converged iteration, ever-valid geometry)
_NewtonCarry: TypeAlias = tuple[Array, Array, Array, Array]


class CellLocationStatus(IntEnum):
    LOCATED = 0
    OUTSIDE = 1
    DEGENERATE_CELL = 2
    NONFINITE = 3
    RESOURCE_EXCEEDED = 4
    INVERSE_MAP_EXHAUSTED = 5


class CellLocationResourceError(ValueError):
    """Fixed-site candidate storage exceeds its explicitly admitted pair count."""

    def __init__(self, requested: int, maximum: int, /) -> None:
        self.requested = requested
        self.maximum = maximum
        super().__init__(
            f"Cell location requires {requested} candidate pairs; admitted {maximum}."
        )


def _point_candidate_counts(
    bvh: PackedBVH, points: np.ndarray, tolerance: float, /
) -> np.ndarray:
    """Count exact own-box hits with a depth-bounded stack, without pair storage."""
    lower, upper, left, right, leaves, items, item_lower, item_upper = (
        np.asarray(value)
        for value in (
            bvh.bbox_min,
            bvh.bbox_max,
            bvh.left,
            bvh.right,
            bvh.leaf_id,
            bvh.leaf_items,
            bvh.item_bbox_min,
            bvh.item_bbox_max,
        )
    )
    counts = np.zeros(points.shape[0], dtype=np.int64)
    for row, point in enumerate(points):
        stack = [0]
        while stack:
            node = stack.pop()
            if not np.all(
                (point >= lower[node] - tolerance) & (point <= upper[node] + tolerance)
            ):
                continue
            leaf = int(leaves[node])
            if leaf < 0:
                stack.extend((int(right[node]), int(left[node])))
                continue
            for item in items[leaf]:
                if item >= 0 and np.all(
                    (point >= item_lower[item] - tolerance)
                    & (point <= item_upper[item] + tolerance)
                ):
                    counts[row] += 1
    return counts


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
            raise ValueError(
                f"SimplicialLocationPolicy.capacities must be positive for candidates, iterations, and seeds; got {capacities}."
            )
        if residual_tolerance <= 0.0 or reference_tolerance < 0.0 or trust_radius <= 0.0:
            raise ValueError(
                f"SimplicialLocationPolicy.tolerances must have positive residual tolerance and trust radius with nonnegative reference tolerance; got {(residual_tolerance, reference_tolerance, trust_radius)}."
            )
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
    candidates are restricted to cells where `cell_mask` is true. Finite-element
    maps and physical Cartesian polyhedral charts share location-result metadata.
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

    def support_cells(self, points: Array, /) -> Array:
        """Return candidate cell IDs with negative padding for unsupported points."""
        return self.locate(points).candidate_cells


def cell_locator_preparation_bounds(
    cell_map: PreparedFiniteElementCellMap,
    coordinates: np.ndarray,
    domain: str,
    /,
) -> tuple[int, int, int]:
    """Bound source rebinding, map preparation and whole-cell Bernstein enclosures."""
    geometry = cell_map.coordinate_spec
    work, memory = cell_geometry_preparation_bounds(geometry)
    queries = cell_geometry_preparation_query_upper(geometry)
    source = geometry.exact_source
    periodic = geometry.periodic_source
    coefficient_bits = (
        periodic.preparation_bounds()[2]
        if periodic is not None
        else max(source.maximum_bits, 1075)
        if source is not None
        else None
    )
    routes = np.asarray(cell_map.coordinate_dofs)
    if routes.shape[0]:
        local = coordinates[routes[0]]
        element = cell_map.coordinate_element
        preparation = coordinate_polynomials_estimate(
            element, local, coefficient_bits_upper=coefficient_bits
        )
        expressions = coordinate_map_expression_estimates(
            element, local, coefficient_bits_upper=coefficient_bits
        )
        if preparation is None or expressions is None:
            raise ValueError(
                f"Locator.coordinate_source must provide source-expression enclosures; got {element.cell_kind}."
            )
        # Every cell has this same declared source tree, coefficient shape and
        # source-bit bound; no sampled carrier value defines the estimate.
        work += routes.shape[0] * preparation.work_upper
        memory += routes.shape[0] * preparation.bytes_upper
        for value in expressions:
            enclosure = coordinate_bernstein_estimate(
                value, domain, cell_map.reference_dimension
            )
            work += routes.shape[0] * enclosure.work_upper
            memory += routes.shape[0] * enclosure.bytes_upper
    return work, memory, queries


@contextmanager
def owned_locator_coordinate_preparation(
    cell_map: PreparedFiniteElementCellMap,
    coordinates: np.ndarray,
    domain: str,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> Iterator[CoordinateEnclosureBudget]:
    """Reuse the request ledger or admit its host enclosure preparation."""
    if budget is not None:
        yield budget
        return
    work, memory, queries = cell_locator_preparation_bounds(cell_map, coordinates, domain)
    with owned_coordinate_preparation(
        work, memory, maximum_geometry_queries=queries
    ) as budget:
        yield budget


def _certified_cell_bounds(
    cell_map: PreparedFiniteElementCellMap,
    coordinates: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose each whole mapped cell from its coordinate source expression."""
    element = cell_map.coordinate_element
    source_coordinates = cell_map.source_coordinates(coordinates, budget=budget)
    cell_coordinates = tuple(
        tuple(source_coordinates[index] for index in route)
        for route in np.asarray(cell_map.coordinate_dofs)
    )
    lower = np.empty((cell_map.cell_count, cell_map.ambient_dimension), dtype=np.float64)
    upper = np.empty_like(lower)
    for cell, local in enumerate(cell_coordinates):
        polynomials = coordinate_polynomials(element, local, budget=budget)
        if polynomials is None:
            raise ValueError(
                f"PreparedSimplicialCellLocator.coordinate_source must provide a source-expression enclosure; got {element.cell_kind}."
            )
        for axis, polynomial in enumerate(polynomials):
            lower[cell, axis], upper[cell, axis] = polynomial_bounds(
                polynomial, "simplex", cell_map.reference_dimension, budget=budget
            )
    return lower, upper


def cell_map_vertices(cell_map: PreparedFiniteElementCellMap, /) -> np.ndarray:
    blocks = (
        cell_map.mesh.blocks
        if cell_map.block_index is None
        else (cell_map.mesh.blocks[cell_map.block_index],)
    )
    return np.concatenate(tuple(np.asarray(block.vertices) for block in blocks), axis=0)


def is_affine_simplex_element(element: CellGeometryElement, /) -> bool:
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


class _LocationCandidates(StrictModule):
    """Bounded BVH candidates expanded over the admitted reference seeds."""

    values: Array
    candidates: Array
    candidate_valid: Array
    search_complete: Array
    flat_cells: Array
    flat_candidate_valid: Array
    targets: Array
    reference_flat: Array
    seed_count: int = eqx.field(static=True)


class _LocationInverses(StrictModule):
    """Newton iterates with convergence and geometry evidence for every seed."""

    reference_flat: Array
    converged: Array
    first_iteration: Array
    ever_valid_geometry: Array


class _ResolvedLocationCandidates(StrictModule):
    """Containing candidates and diagnostic arrays after source-image exclusion."""

    accepted: Array
    reference_all: Array
    residual_all: Array
    condition_all: Array
    iteration_all: Array
    inverses_complete: Array
    ever_valid_geometry: Array


@final
class CellLocationArrays(StrictModule):
    """Identity-free numerical location; `identified` binds its locator identity."""

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

    def identified(self, locator_id: str, /) -> CellLocationResult:
        """Attach the owning locator's identity outside compiled location."""
        return CellLocationResult(
            cell_ids=self.cell_ids,
            reference_coordinates=self.reference_coordinates,
            barycentric=self.barycentric,
            geometry_residual=self.geometry_residual,
            iterations=self.iterations,
            jacobian_condition=self.jacobian_condition,
            inside=self.inside,
            used_fallback=self.used_fallback,
            candidate_count=self.candidate_count,
            status=self.status,
            successful=self.successful,
            candidate_cells=self.candidate_cells,
            candidate_reference=self.candidate_reference,
            candidates_complete=self.candidates_complete,
            locator_id=locator_id,
        )


def simplex_reference_seeds(cell_kind: str, dimension: int, dtype: jnp.dtype, /) -> Array:
    """Return the simplex reference center followed by its vertices in cell order."""
    center = jnp.full((dimension,), 1.0 / (dimension + 1), dtype=dtype)
    vertices = jnp.asarray(reference_cell_topology(cell_kind).vertices, dtype=dtype)
    return jnp.concatenate((center[None], vertices), axis=0)


class AbstractNewtonInverse(StrictModule, NonTrainableState):
    """Identity-free bounded damped-Newton inverse of one prepared coordinate map.

    Fields are exactly the numerics read by location: the coordinate map without
    topology, layout or source identities, its coordinates, query hierarchy and
    policy. Compiled location is shared by maps of equal numerical structure;
    each locator attaches its own identity outside compilation.
    """

    cell_map: NumericalFiniteElementCellMap
    coordinates: Array
    bvh: PackedBVH
    policy: SimplicialLocationPolicy
    cell_kind: str = eqx.field(static=True)
    affine_simplex: bool = eqx.field(static=True)

    @property
    def dimension(self) -> int:
        return self.cell_map.reference_dimension

    @property
    def cell_count(self) -> int:
        return self.cell_map.cell_count

    def reference_seeds(self, dtype: jnp.dtype, /) -> Array:
        """Return the reference center followed by reference vertices in cell order."""
        return simplex_reference_seeds(self.cell_kind, self.dimension, dtype)

    def contains_reference(self, reference: Array, /) -> Array:
        """Test simplex containment with the policy's reference-coordinate tolerance."""
        tolerance = self.policy.reference_tolerance
        return jnp.all(reference >= -tolerance, axis=-1) & (
            jnp.sum(reference, axis=-1) <= 1.0 + tolerance
        )

    def reference_weights(self, reference: Array, /) -> Array:
        """Return simplex barycentric weights in reference-vertex order."""
        return jnp.concatenate(
            ((1.0 - jnp.sum(reference, axis=-1))[:, None], reference), axis=-1
        )

    def _prepare_location_candidates(
        self,
        points: ArrayLike,
        cell_mask: ArrayLike | None,
    ) -> _LocationCandidates:
        values = jnp.asarray(points, dtype=self.coordinates.dtype)
        if values.ndim != 2 or values.shape[1] != self.cell_map.ambient_dimension:
            raise ValueError(
                f"PreparedSimplicialCellLocator.points must be rank two with the prepared ambient dimension; got {values.shape}."
            )
        if cell_mask is not None:
            mask = jnp.asarray(cell_mask)
            if mask.shape != (self.cell_count,) or mask.dtype != jnp.bool_:
                raise ValueError(
                    f"PreparedSimplicialCellLocator.cell_mask must be boolean with one entry per cell; got {(mask.shape, mask.dtype)}."
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
        seed_pool = self.reference_seeds(values.dtype)
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
        reference_flat = reference.reshape((-1, reference_dimension))
        return _LocationCandidates(
            values,
            candidates,
            candidate_valid,
            search_complete,
            flat_cells,
            flat_candidate_valid,
            targets,
            reference_flat,
            seed_count,
        )

    def _invert_location_candidates(
        self, prepared: _LocationCandidates
    ) -> _LocationInverses:
        flat_cells = prepared.flat_cells
        flat_candidate_valid = prepared.flat_candidate_valid
        targets = prepared.targets
        reference_flat = prepared.reference_flat
        converged = jnp.zeros((flat_cells.size,), dtype=jnp.bool_)
        first_iteration = jnp.zeros((flat_cells.size,), dtype=jnp.int32)
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

        def active_step(iteration: Array, carry: _NewtonCarry) -> _NewtonCarry:
            # Retain the fixed-trip reverse-mode program, but stop evaluating
            # charts after every admissible candidate has converged.
            active = jnp.any(flat_candidate_valid & ~carry[1])
            return jax.lax.cond(
                active,
                lambda current: newton_step(iteration, current),
                lambda current: current,
                carry,
            )

        (
            reference_flat,
            converged,
            first_iteration,
            ever_valid_geometry,
        ) = jax.lax.fori_loop(
            0,
            self.policy.maximum_iterations,
            active_step,
            (
                reference_flat,
                converged,
                first_iteration,
                ever_valid_geometry,
            ),
        )
        return _LocationInverses(
            reference_flat, converged, first_iteration, ever_valid_geometry
        )

    def _resolve_location_candidates(
        self,
        prepared: _LocationCandidates,
        inverse: _LocationInverses,
    ) -> _ResolvedLocationCandidates:
        candidates = prepared.candidates
        candidate_valid = prepared.candidate_valid
        flat_cells, flat_candidate_valid = (
            prepared.flat_cells,
            prepared.flat_candidate_valid,
        )
        targets = prepared.targets
        point_count, candidate_capacity = candidates.shape
        seed_count, reference_dimension = prepared.seed_count, self.dimension
        reference_flat, converged = inverse.reference_flat, inverse.converged
        first_iteration, ever_valid_geometry = (
            inverse.first_iteration,
            inverse.ever_valid_geometry,
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
        inside_reference = self.contains_reference(reference_flat)
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
            and self.affine_simplex
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
        return _ResolvedLocationCandidates(
            accepted,
            reference_all,
            residual_all,
            condition_all,
            iteration_all,
            inverses_complete,
            ever_valid_geometry,
        )

    def _location_result(
        self,
        prepared: _LocationCandidates,
        resolved: _ResolvedLocationCandidates,
    ) -> CellLocationArrays:
        values, candidates = prepared.values, prepared.candidates
        candidate_valid, search_complete = (
            prepared.candidate_valid,
            prepared.search_complete,
        )
        flat_candidate_valid = prepared.flat_candidate_valid
        point_count, candidate_capacity = candidates.shape
        seed_count = prepared.seed_count
        accepted, reference_all = resolved.accepted, resolved.reference_all
        residual_all, condition_all = resolved.residual_all, resolved.condition_all
        iteration_all, inverses_complete = (
            resolved.iteration_all,
            resolved.inverses_complete,
        )
        ever_valid_geometry = resolved.ever_valid_geometry
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
        barycentric = self.reference_weights(reference_result)
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
        return CellLocationArrays(
            cell_ids=cell_ids,
            reference_coordinates=jnp.where(inside[:, None], reference_result, 0.0),
            barycentric=jnp.where(inside[:, None], barycentric, 0.0),
            geometry_residual=jnp.where(has_candidates, residual_result, jnp.inf),
            iterations=jnp.where(
                iteration_result > 0, iteration_result, self.policy.maximum_iterations
            ),
            jacobian_condition=jnp.where(has_candidates, condition_result, jnp.inf),
            inside=inside,
            used_fallback=seed_choice != 0,
            candidate_count=jnp.sum(candidate_accepted, axis=1, dtype=jnp.int32),
            status=status,
            successful=inside & finite,
            candidate_cells=candidate_cells,
            candidate_reference=jnp.where(
                candidate_accepted[:, :, None], candidate_reference, 0.0
            ),
            candidates_complete=finite & search_complete & inverses_complete,
        )

    def locate_arrays(
        self, points: ArrayLike, cell_mask: ArrayLike | None, /
    ) -> CellLocationArrays:
        """Bounded candidates, fixed-trip Newton inverses and their resolution."""
        prepared = self._prepare_location_candidates(points, cell_mask)
        inverse = self._invert_location_candidates(prepared)
        resolved = self._resolve_location_candidates(prepared, inverse)
        return self._location_result(prepared, resolved)


@eqx.filter_jit
def locate_cells(
    inverse: AbstractNewtonInverse, points: ArrayLike, cell_mask: ArrayLike | None, /
) -> CellLocationArrays:
    """Compile location once per numerical structure, never per source identity."""
    return inverse.locate_arrays(points, cell_mask)


# Rows are located independently: every query count shares one compiled
# program per numerical structure. Trailing rows of the final block repeat its
# last query, so they neither add candidates nor extend shared Newton trips.
_LOCATION_ROWS = 64


def locate_rows(
    inverse: AbstractNewtonInverse, points: ArrayLike, cell_mask: ArrayLike | None, /
) -> CellLocationArrays:
    """Locate fixed-size row blocks and return exactly the requested rows."""
    traced = any(
        isinstance(leaf, Tracer)
        for leaf in jax.tree_util.tree_leaves((inverse, points, cell_mask))
    )
    values = (
        jnp.asarray(points, dtype=inverse.coordinates.dtype)
        if traced
        else np.asarray(points, dtype=np.dtype(inverse.coordinates.dtype))
    )
    count = values.shape[0] if values.ndim == 2 else 0
    if count == 0:
        # Rank and dimension refusals stay with the canonical location program.
        return locate_cells(inverse, values, cell_mask)
    padded = -(-count // _LOCATION_ROWS) * _LOCATION_ROWS
    if traced:
        rows = jnp.concatenate(
            (values, jnp.broadcast_to(values[-1:], (padded - count, values.shape[1])))
        )
    else:
        rows = np.concatenate(
            (values, np.broadcast_to(values[-1:], (padded - count, values.shape[1])))
        )
    blocks = [
        locate_cells(inverse, rows[start : start + _LOCATION_ROWS], cell_mask)
        for start in range(0, padded, _LOCATION_ROWS)
    ]
    if traced:
        return jax.tree_util.tree_map(
            lambda *parts: jnp.concatenate(parts)[:count], *blocks
        )
    host = jax.device_get(blocks)
    return jax.tree_util.tree_map(
        lambda *parts: jax.device_put(np.concatenate(parts)[:count]), *host
    )


def row_blocked[Result](
    function: Callable[..., Result],
    arguments: tuple[object, ...],
    rows: tuple[int, ...],
    /,
) -> Result:
    """Evaluate a row-independent program on location-block padded rows.

    ``arguments[i]`` for ``i`` in ``rows`` share one leading query axis, and every
    output leaf is row-leading. Padding repeats the final query row and is
    discarded, so each requested row is computed exactly as unpadded while query
    counts within one block share one compiled program. Traced callers keep their
    exact shapes: their enclosing program already owns the compilation.
    """
    if any(isinstance(leaf, Tracer) for leaf in jax.tree_util.tree_leaves(arguments)):
        return function(*arguments)
    count = int(np.asarray(arguments[rows[0]]).shape[0])
    padded = -(-count // _LOCATION_ROWS) * _LOCATION_ROWS
    if count == 0 or padded == count:
        return function(*arguments)
    values = list(arguments)
    for index in rows:
        row = np.asarray(values[index])
        if row.shape[0] != count:
            raise ValueError("row_blocked.rows must share one leading query axis.")
        values[index] = np.concatenate(
            (row, np.broadcast_to(row[-1:], (padded - count, *row.shape[1:])))
        )
    result = jax.device_get(function(*values))
    # Host transfer, not ``jnp.asarray``: no conversion program per row count.
    return jax.tree_util.tree_map(lambda leaf: jax.device_put(leaf[:count]), result)


class AbstractPreparedNewtonCellLocator(AbstractCellLocator, NonTrainableState):
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

    @abc.abstractmethod
    def numerical_inverse(self) -> AbstractNewtonInverse:
        """Return the identity-free inverse evaluated by `locate`."""
        raise NotImplementedError

    def reference_seeds(self, dtype: jnp.dtype, /) -> Array:
        """Return the reference center followed by reference vertices in cell order."""
        return self.numerical_inverse().reference_seeds(dtype)

    def contains_reference(self, reference: Array, /) -> Array:
        """Test reference-domain containment with the policy tolerance."""
        return self.numerical_inverse().contains_reference(reference)

    def reference_weights(self, reference: Array, /) -> Array:
        """Return reference-vertex weights of the reference domain."""
        return self.numerical_inverse().reference_weights(reference)

    @checked
    def locate(
        self, points: ArrayLike, /, *, cell_mask: ArrayLike | None = None
    ) -> CellLocationResult:
        # An absent mask admits every cell; one compiled program serves both.
        mask = (
            np.ones(self.cell_count, dtype=np.bool_) if cell_mask is None else cell_mask
        )
        return locate_rows(self.numerical_inverse(), points, mask).identified(
            self.locator_id
        )

    def prepare_location(
        self,
        points: ArrayLike,
        maximum_location_pairs: int,
        /,
        *,
        cell_mask: ArrayLike | None = None,
    ) -> CellLocationResult:
        """Count, admit, then invert the complete fixed-site broad phase."""
        if (
            isinstance(maximum_location_pairs, bool)
            or not isinstance(maximum_location_pairs, int)
            or maximum_location_pairs < 1
        ):
            raise ValueError("maximum_location_pairs must be a positive integer.")
        values = np.asarray(points, dtype=np.dtype(self.coordinates.dtype))
        if values.ndim != 2 or values.shape[1] != self.cell_map.ambient_dimension:
            raise ValueError("Prepared location sites must be (points, dimension).")
        counts = _point_candidate_counts(
            self.bvh, values, self.policy.reference_tolerance
        )
        capacity = max(1, int(np.max(counts, initial=0)))
        requested = values.shape[0] * capacity
        if requested > maximum_location_pairs:
            raise CellLocationResourceError(requested, maximum_location_pairs)
        policy = SimplicialLocationPolicy(
            capacity,
            self.policy.maximum_iterations,
            self.policy.maximum_seeds,
            residual_tolerance=self.policy.residual_tolerance,
            reference_tolerance=self.policy.reference_tolerance,
            trust_radius=self.policy.trust_radius,
        )
        inverse = eqx.tree_at(
            lambda inverse: inverse.policy, self.numerical_inverse(), policy
        )
        return locate_cells(inverse, values, cell_mask).identified(self.locator_id)

    def _affine_geometry(self) -> tuple[Array, Array, Array, Array]:
        if not is_affine_simplex_element(self.cell_map.coordinate_element):
            raise ValueError(
                f"PreparedSimplicialCellLocator.cell_map must be an affine simplex map for facet traversal; got {self.cell_map.coordinate_element.cell_kind}."
            )
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
        """Return barycentric gradients indexed by cell, local vertex, and ambient axis."""
        return self._affine_geometry()[1]

    def affine_barycentric(self, points: ArrayLike, /) -> Array:
        """Evaluate all affine cells, with shape (point, cell, local vertex)."""
        values = jnp.asarray(points, dtype=self.coordinates.dtype)
        if values.ndim != 2 or values.shape[1] != self.cell_map.ambient_dimension:
            raise ValueError(
                f"PreparedSimplicialCellLocator.affine_barycentric.points must be rank two with the prepared ambient dimension; got {values.shape}."
            )
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
        """Walk affine facet intervals, choosing the lowest cell on ties.

        Prepared vertex stars admit face, edge, and vertex crossings without
        allocating point-by-mesh search tables. Gaps terminate at the first
        domain exit rather than joining disconnected pieces. No uniform
        temporal subdivision or endpoint-only admission is used.
        """
        capacity = self.cell_count if maximum_segments is None else maximum_segments
        if capacity < 1:
            raise ValueError(
                f"PreparedSimplicialCellLocator.maximum_segments must be positive; got {capacity}."
            )
        first = jnp.asarray(start, dtype=self.coordinates.dtype)
        last = jnp.asarray(end, dtype=first.dtype)
        if first.shape != last.shape or first.ndim != 2:
            raise ValueError(
                f"PreparedSimplicialCellLocator.segment_endpoints must be matching rank-two arrays; got {(first.shape, last.shape)}."
            )
        if first.shape[1] != self.cell_map.ambient_dimension:
            raise ValueError(
                f"PreparedSimplicialCellLocator.segment_endpoints must match the prepared ambient dimension; got {first.shape}."
            )
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


def vertex_star_routes(cells: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
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
    """Walk affine intervals within bounds with lowest-index outgoing-cell ties."""
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


@eqx.filter_jit
def mapped_physical_points(
    cell_map: NumericalFiniteElementCellMap,
    coordinates: Array,
    cell_indices: Array,
    reference_points: Array,
) -> Array:
    """Project the canonical map evaluation so unused metric work can be eliminated."""
    return cell_map.evaluate(coordinates, cell_indices, reference_points).physical_points


@final
class SimplicialNewtonInverse(AbstractNewtonInverse):
    """Identity-free inverse with simplex reference-domain operations."""


@final
class PreparedSimplicialCellLocator(AbstractPreparedNewtonCellLocator):
    """Invert arbitrary-dimensional simplex maps with whole-source BVH enclosures."""

    @checked
    def __init__(
        self,
        cell_map: PreparedFiniteElementCellMap,
        coordinates: ArrayLike,
        policy: SimplicialLocationPolicy,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        kind = cell_map.coordinate_element.cell_kind
        if kind not in ("interval", "triangle", "tetrahedron") and not kind.startswith(
            "simplex:"
        ):
            raise ValueError(
                f"PreparedSimplicialCellLocator.cell_map must use a simplex coordinate element; got {kind!r}."
            )
        values = jnp.asarray(coordinates)
        if values.shape != (cell_map.coordinate_count, cell_map.ambient_dimension):
            raise ValueError(
                f"PreparedSimplicialCellLocator.coordinates must match the prepared coordinate count and ambient dimension; got {values.shape}."
            )
        cells = cell_map.coordinate_dofs
        host = np.asarray(jax.device_get(values))
        with owned_locator_coordinate_preparation(
            cell_map, host, "simplex", budget=budget
        ) as budget:
            lower, upper = _certified_cell_bounds(cell_map, host, budget=budget)
        lower = np.nextafter(
            np.asarray(lower, dtype=values.dtype),
            np.asarray(-np.inf, dtype=values.dtype),
        )
        upper = np.nextafter(
            np.asarray(upper, dtype=values.dtype),
            np.asarray(np.inf, dtype=values.dtype),
        )
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
        centroids = mapped_physical_points(
            cell_map.numerical, values, jnp.arange(cell_map.cell_count), center
        )
        if is_affine_simplex_element(cell_map.coordinate_element):
            star_cells, star_valid = vertex_star_routes(cell_map_vertices(cell_map))
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

    def numerical_inverse(self) -> SimplicialNewtonInverse:
        return SimplicialNewtonInverse(
            self.cell_map.numerical,
            self.coordinates,
            self.bvh,
            self.policy,
            self.cell_map.coordinate_element.cell_kind,
            is_affine_simplex_element(self.cell_map.coordinate_element),
        )


__all__ = [
    "AbstractCellLocator",
    "CellLocationResult",
    "CellLocationResourceError",
    "CellLocationStatus",
    "LocatedCellMap",
    "PreparedSimplicialCellLocator",
    "SegmentLocationResult",
    "SimplicialLocationPolicy",
]
