#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Screened Poisson surface reconstruction: shared finite-element substrate.

Following Kazhdan and Hoppe (2013), the indicator ``chi`` minimizes

    E(chi) = int |grad chi - V|^2 dx + (alpha / h) sum_k a_k chi(p_k)^2,

where ``V = sum_k a_k n_k B(x - p_k)`` splats the oriented samples with the
trilinear (degree-one) B-spline of the finest cells, ``a_k`` is the
density-estimated surface area represented by sample ``k``, and ``h`` is the
finest cell width. With continuous trilinear finite elements and natural
boundary conditions the normal equations are the sparse symmetric
positive-definite system ``(K + alpha S) chi = D``; screening makes it definite
whenever one sample exists. The system is solved by native Jacobi-preconditioned
conjugate gradients whose status reaches the consumer. The surface is the level
set of ``chi`` at the area-weighted sample average, extracted by marching
tetrahedra on a conforming tetrahedral split of the cells: a piecewise-linear
level set that avoids tetrahedron vertices and the domain boundary is a closed
oriented 2-manifold.

This module owns the regular-grid discretization (Freudenthal split of every
grid cube) and the shared element tensors, native solve, and extraction; the
adaptive octree discretization lives in ``_poisson_octree``. Outward sample
normals make ``chi`` increase outward with a unit jump across the surface, so
the reconstructed region is ``chi - iso < 0``. Screened Poisson smoothing
approximates the sampled surface at the finest resolution; it is not a
feature-preserving reconstruction.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal, TypeAlias

import jax.numpy as jnp
import numpy as np

from ... import ein
from ..._bvh import bvh_nearest_items, BVHBuildPolicy, prepare_bvh
from ...linalg import (
    ArraySpace,
    DifferentiationPolicy,
    FailurePolicy,
    JacobiPreconditionerBuilder,
    linear_status_message,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    PCG,
    PreconditioningPolicy,
    prepare,
    solve,
    TolerancePolicy,
)
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ..simplicial._bvh import TriangleBVH
from ..simplicial._mesh import TriangleMesh


PoissonDiscretization: TypeAlias = Literal["octree", "regular"]
"""``octree`` solves on a 2:1-balanced adaptive octree refined to the finest
width only around the samples, with hanging-node constraints; ``regular``
solves on the full regular grid of the finest width."""

_CORNERS = np.asarray(tuple(itertools.product((0, 1), repeat=3)), dtype=np.int64)
_CORNER_BITS = _CORNERS @ np.asarray((4, 2, 1), dtype=np.int64)
# Two-point Gauss-Legendre quadrature is exact for the per-axis quadratic
# products of trilinear shape functions and their derivatives.
_GAUSS = 0.5 + np.asarray((-0.5, 0.5), dtype=np.float64) / np.sqrt(3.0)
_SOLVER_RELATIVE_TOLERANCE = 1.0e-8
_LEVEL_SET_CLEARANCE = 1.0e-4
_SOLVER_STEPS_PER_GRID_EXTENT = 16
_POINT_POLICY = BVHBuildPolicy(leaf_size=16)


def _shape_values(local: np.ndarray, /) -> np.ndarray:
    """Trilinear shape values ``(..., 8)`` at unit-cube coordinates ``(..., 3)``."""

    factors = np.where(_CORNERS == 1, local[..., None, :], 1.0 - local[..., None, :])
    return np.prod(factors, axis=-1)


def _shape_gradients(local: np.ndarray, /) -> np.ndarray:
    """Unit-cube shape gradients ``(..., 8, 3)`` at coordinates ``(..., 3)``."""

    factors = np.where(_CORNERS == 1, local[..., None, :], 1.0 - local[..., None, :])
    signs = np.where(_CORNERS == 1, 1.0, -1.0)
    gradients = []
    for axis in range(3):
        others = [other for other in range(3) if other != axis]
        gradients.append(signs[:, axis] * np.prod(factors[..., others], axis=-1))
    return np.stack(gradients, axis=-1)


def _reference_element() -> tuple[np.ndarray, np.ndarray]:
    """Unit-cube stiffness ``K[a, b]`` and splat-divergence ``G[a, b, e]`` tensors.

    ``K[a, b] = int grad phi_a . grad phi_b`` and
    ``G[a, b, e] = int phi_b d_e phi_a`` over the unit cube.
    """

    nodes = np.asarray(tuple(itertools.product(_GAUSS, repeat=3)), dtype=np.float64)
    weight = 1.0 / nodes.shape[0]
    values = _shape_values(nodes)
    gradients = _shape_gradients(nodes)
    stiffness = weight * np.asarray(
        ein.contract("qae,qbe->ab", gradients, gradients), dtype=np.float64
    )
    splat = weight * np.asarray(
        ein.contract("qae,qb->abe", gradients, values), dtype=np.float64
    )
    return stiffness, splat


_STIFFNESS, _SPLAT_DIVERGENCE = _reference_element()


def _freudenthal_tetrahedra() -> np.ndarray:
    """Six cube tetrahedra along the main diagonal, as corner indices ``(6, 4)``."""

    tetrahedra = []
    for permutation in itertools.permutations(range(3)):
        offset = np.zeros((3,), dtype=np.int64)
        path = [0]
        for axis in permutation:
            offset[axis] = 1
            path.append(int(offset @ np.asarray((4, 2, 1), dtype=np.int64)))
        tetrahedra.append([int(np.flatnonzero(_CORNER_BITS == bit)[0]) for bit in path])
    return np.asarray(tetrahedra, dtype=np.int64)


def _tetrahedron_cases() -> tuple[np.ndarray, np.ndarray]:
    """Crossing-edge triangles ``(16, 2, 3, 2)`` and counts per inside mask.

    Each triangle lists its three level-set vertices as (inside, outside) local
    corner pairs; two inside corners emit the quad ``ac, ad, bd, bc`` as two
    triangles.
    """

    table = np.zeros((16, 2, 3, 2), dtype=np.int64)
    counts = np.zeros((16,), dtype=np.int64)
    for case in range(16):
        inside = [corner for corner in range(4) if case >> corner & 1]
        outside = [corner for corner in range(4) if not case >> corner & 1]
        match len(inside):
            case 1:
                table[case, 0] = [(inside[0], corner) for corner in outside]
                counts[case] = 1
            case 3:
                table[case, 0] = [(corner, outside[0]) for corner in inside]
                counts[case] = 1
            case 2:
                (a, b), (c, d) = inside, outside
                table[case, 0] = [(a, c), (a, d), (b, d)]
                table[case, 1] = [(a, c), (b, d), (b, c)]
                counts[case] = 2
    return table, counts


_TETRAHEDRA = _freudenthal_tetrahedra()
_CASE_TRIANGLES, _CASE_COUNTS = _tetrahedron_cases()


@dataclass(frozen=True, slots=True)
class PoissonSolveEvidence:
    """Discretization, native solve, and level-set evidence of a Poisson indicator.

    ``grid_shape``, ``grid_origin`` and ``grid_spacing`` describe the finest
    lattice: the full regular grid, or the lattice the octree of depth
    ``octree_depth`` (``None`` on the regular grid) adapts. ``unknowns`` counts
    the solved degrees of freedom, ``leaf_cells`` the finite-element cells, and
    ``hanging_nodes`` the octree nodes constrained to their coarser neighbors.
    ``solver_*`` fields are the native preconditioned conjugate-gradient status
    and diagnostics. ``represented_area`` sums the density-estimated sample
    areas. ``iso_value`` is the area-weighted sample average of the indicator
    and ``iso_value_spread`` the weighted standard deviation of the sampled
    indicator around it; the indicator jumps by one across the surface, so a
    spread comparable to one means the samples are not fit by one level set.
    """

    discretization: PoissonDiscretization
    grid_shape: tuple[int, int, int]
    grid_origin: tuple[float, float, float]
    grid_spacing: float
    octree_depth: int | None
    unknowns: int
    leaf_cells: int
    hanging_nodes: int
    screening: float
    represented_area: float
    solver_status: int
    solver_message: str
    solver_converged: bool
    solver_iterations: int
    solver_relative_residual: float
    iso_value: float
    iso_value_spread: float


@dataclass(frozen=True, slots=True)
class SampledSurfaceDeviation:
    """Sampled two-sided deviation between samples and an extracted surface.

    ``sample_to_surface_*`` are exact Euclidean distances from the retained
    samples to the extracted triangles; ``surface_to_sample_max`` is the largest
    distance from an extracted vertex to its nearest sample. This is sampled
    evidence, not a continuous Hausdorff certificate.
    """

    sample_to_surface_max: float
    sample_to_surface_mean: float
    surface_to_sample_max: float


@dataclass(frozen=True, slots=True)
class TetrahedralField:
    """Piecewise-linear indicator minus its iso-value on conforming tetrahedra.

    Only tetrahedra of cells whose values change sign are retained; vertex
    values are exact functions of the finite-element nodal values, so shared
    faces carry identical vertices and values.
    """

    points: np.ndarray
    values: np.ndarray
    tetrahedra: np.ndarray


@dataclass(frozen=True, slots=True)
class PoissonIndicator:
    """Solved indicator: sampled values minus the iso-value and the extraction field."""

    sample_values: np.ndarray
    field: TetrahedralField
    evidence: PoissonSolveEvidence


@dataclass(frozen=True, slots=True)
class LinearSolveRecord:
    values: np.ndarray
    status: int
    converged: bool
    iterations: int
    relative_residual: float


@dataclass(frozen=True, slots=True)
class _Grid:
    origin: np.ndarray
    spacing: float
    shape: tuple[int, int, int]

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def node_index(self, lattice: np.ndarray, /) -> np.ndarray:
        return (lattice[..., 0] * self.shape[1] + lattice[..., 1]) * self.shape[
            2
        ] + lattice[..., 2]

    def cells(self, /) -> np.ndarray:
        """Lower lattice corner of every cell in row-major order ``(cells, 3)``."""

        ranges = [np.arange(extent - 1, dtype=np.int64) for extent in self.shape]
        return np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape((-1, 3))

    def locate(self, points: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        """Containing cell corners ``(n, 8)`` and trilinear weights ``(n, 8)``."""

        scaled = (points - self.origin) / self.spacing
        upper = np.asarray(self.shape, dtype=np.int64) - 2
        lower = np.clip(np.floor(scaled).astype(np.int64), 0, upper)
        corners = self.node_index(lower[:, None, :] + _CORNERS[None, :, :])
        return corners, _shape_values(scaled - lower)


def sample_areas(neighbor_distances: np.ndarray, /) -> np.ndarray:
    """Surface area represented by each sample from its ``k``-neighbor disk.

    A disk through the ``k``-th nearest neighbor holds ``k + 1`` samples, so the
    local sampling density gives ``a = pi r_k^2 / (k + 1)``.
    """

    radius = neighbor_distances[:, -1]
    if np.any(radius <= 0.0):
        raise ValueError(
            "Sample density is undefined where more than neighborhood_size samples coincide."
        )
    return np.pi * radius**2 / (neighbor_distances.shape[1] + 1)


def _grid(
    points: np.ndarray, spacing: float, padding_cells: int, maximum_nodes: int, /
) -> _Grid:
    lower = np.min(points, axis=0) - padding_cells * spacing
    extent = np.max(points, axis=0) + padding_cells * spacing - lower
    shape = np.ceil(extent / spacing).astype(np.int64) + 1
    nodes = int(np.prod(shape))
    if nodes > maximum_nodes:
        raise ValueError(
            f"Poisson grid needs {nodes} nodes, above maximum_grid_nodes={maximum_nodes}; "
            "increase sample_spacing or the node budget."
        )
    return _Grid(lower, spacing, (int(shape[0]), int(shape[1]), int(shape[2])))


def _operator_entries(
    grid: _Grid,
    corners: np.ndarray,
    weights: np.ndarray,
    screening_weights: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coalesced ``K + alpha S`` entries on the 27-point stencil, row-major.

    Row ``i`` stores offset ``o`` of the stencil at slot ``27 i + o``; slots
    whose neighbor lies outside the grid are dropped.
    """

    size = grid.size
    cells = grid.node_index(grid.cells()[:, None, :] + _CORNERS[None, :, :])
    stencil = np.zeros((size, 27), dtype=np.float64)
    difference = _CORNERS[None, :, :] - _CORNERS[:, None, :] + 1
    slot = (difference[..., 0] * 3 + difference[..., 1]) * 3 + difference[..., 2]
    for first in range(8):
        for second in range(8):
            column = slot[first, second]
            stencil[:, column] += np.bincount(
                cells[:, first],
                weights=np.full(
                    (cells.shape[0],), grid.spacing * _STIFFNESS[first, second]
                ),
                minlength=size,
            )
            stencil[:, column] += np.bincount(
                corners[:, first],
                weights=screening_weights * weights[:, first] * weights[:, second],
                minlength=size,
            )
    lattice = np.stack(
        np.unravel_index(np.arange(size, dtype=np.int64), grid.shape), axis=-1
    )
    offsets = np.asarray(tuple(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)
    neighbor = lattice[:, None, :] + offsets[None, :, :]
    inside = np.all(
        (neighbor >= 0) & (neighbor < np.asarray(grid.shape, dtype=np.int64)), axis=-1
    )
    rows = np.broadcast_to(np.arange(size, dtype=np.int64)[:, None], (size, 27))
    columns = grid.node_index(np.where(inside[..., None], neighbor, 0))
    return rows[inside], columns[inside], stencil[inside]


def cell_divergence(
    cell_nodes: np.ndarray, cell_widths: np.ndarray, field: np.ndarray, size: int, /
) -> np.ndarray:
    """``D_i = int V . grad phi_i`` of the cellwise-trilinear nodal field ``V``.

    ``cell_nodes`` lists the eight corner nodes of each cell in ``_CORNERS``
    order and ``cell_widths`` their physical widths; cells on which ``V``
    vanishes contribute nothing and are skipped.
    """

    active = np.any(np.any(field[cell_nodes] != 0.0, axis=-1), axis=-1)
    nodes = cell_nodes[active]
    local = cell_widths[active, None] ** 2 * np.asarray(
        ein.contract("abe,cbe->ca", _SPLAT_DIVERGENCE, field[nodes]), dtype=np.float64
    )
    return np.bincount(nodes.reshape((-1,)), weights=local.reshape((-1,)), minlength=size)


def splat_field(
    nodes: np.ndarray, weights: np.ndarray, splat: np.ndarray, size: int, /
) -> np.ndarray:
    """Nodal values ``(size, 3)`` of the trilinear splat of per-sample vectors."""

    return np.stack(
        [
            np.bincount(
                nodes.reshape((-1,)),
                weights=(weights[:, :, None] * splat[:, None, :])[..., axis].reshape(
                    (-1,)
                ),
                minlength=size,
            )
            for axis in range(3)
        ],
        axis=1,
    )


def solve_indicator(
    size: int,
    rows: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    rhs: np.ndarray,
    max_steps: int,
    /,
) -> LinearSolveRecord:
    """Native Jacobi-PCG solve of the assembled symmetric positive-definite system."""

    space = ArraySpace((size,), dtype=np.float64, space_id="screened-poisson:indicator")
    operator = SparseCoordinateOperator(
        EdgeRelation(columns, rows, source_size=size, target_size=size),
        jnp.asarray(values, dtype=jnp.float64),
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id="screened-poisson:operator",
        accumulation_dtype=np.float64,
    )
    prepared = prepare(
        LinearSystem(operator, problem_id="screened-poisson:system"),
        LinearSolvePolicy(
            PCG(),
            tolerance=TolerancePolicy(
                relative=_SOLVER_RELATIVE_TOLERANCE,
                absolute=0.0,
                max_steps=max_steps,
            ),
            preconditioning=PreconditioningPolicy(JacobiPreconditionerBuilder()),
            differentiation=DifferentiationPolicy("none"),
            failure=FailurePolicy("status"),
        ),
    )
    result = solve(prepared, jnp.asarray(rhs, dtype=jnp.float64))
    return LinearSolveRecord(
        values=np.asarray(result.value, dtype=np.float64),
        status=int(result.status),
        converged=bool(result.diagnostics.converged),
        iterations=int(result.diagnostics.iterations),
        relative_residual=float(result.diagnostics.relative_residual),
    )


def iso_statistics(sampled: np.ndarray, areas: np.ndarray, /) -> tuple[float, float]:
    """Area-weighted mean of the sampled indicator and its weighted spread."""

    total = float(np.sum(areas))
    iso = float(np.sum(areas * sampled) / total)
    return iso, float(np.sqrt(np.sum(areas * (sampled - iso) ** 2) / total))


def _regular_field(grid: _Grid, field: np.ndarray, /) -> TetrahedralField:
    """Freudenthal tetrahedra of the grid cells whose corner values change sign."""

    inside = field.reshape(grid.shape) < 0.0
    cells = grid.cells()
    corner_inside = inside[tuple((cells[:, None, :] + _CORNERS[None, :, :]).T)].T
    mixed = np.any(corner_inside, axis=1) & ~np.all(corner_inside, axis=1)
    nodes = grid.node_index(cells[mixed][:, None, :] + _CORNERS[None, :, :])
    used, tetrahedra = np.unique(
        nodes[:, _TETRAHEDRA].reshape((-1,)), return_inverse=True
    )
    points = grid.origin + grid.spacing * np.stack(
        np.unravel_index(used, grid.shape), axis=-1
    ).astype(np.float64)
    return TetrahedralField(points, field[used], tetrahedra.reshape((-1, 4)))


def solve_regular_poisson(
    points: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    /,
    *,
    spacing: float,
    screening: float,
    padding_cells: int,
    maximum_grid_nodes: int,
) -> PoissonIndicator:
    """Assemble and solve the screened Poisson indicator on a regular grid."""

    grid = _grid(points, spacing, padding_cells, maximum_grid_nodes)
    corners, weights = grid.locate(points)
    rows, columns, values = _operator_entries(
        grid, corners, weights, screening / spacing * areas
    )
    cell_nodes = grid.node_index(grid.cells()[:, None, :] + _CORNERS[None, :, :])
    field = splat_field(corners, weights, areas[:, None] * normals, grid.size)
    rhs = cell_divergence(
        cell_nodes,
        np.full((cell_nodes.shape[0],), spacing, dtype=np.float64),
        field / spacing**3,
        grid.size,
    )
    record = solve_indicator(
        grid.size,
        rows,
        columns,
        values,
        rhs,
        _SOLVER_STEPS_PER_GRID_EXTENT * max(grid.shape),
    )
    sampled = np.sum(record.values[corners] * weights, axis=1)
    iso, spread = iso_statistics(sampled, areas)
    evidence = PoissonSolveEvidence(
        discretization="regular",
        grid_shape=grid.shape,
        grid_origin=(
            float(grid.origin[0]),
            float(grid.origin[1]),
            float(grid.origin[2]),
        ),
        grid_spacing=grid.spacing,
        octree_depth=None,
        unknowns=grid.size,
        leaf_cells=cell_nodes.shape[0],
        hanging_nodes=0,
        screening=screening,
        represented_area=float(np.sum(areas)),
        solver_status=record.status,
        solver_message=linear_status_message(record.status),
        solver_converged=record.converged,
        solver_iterations=record.iterations,
        solver_relative_residual=record.relative_residual,
        iso_value=iso,
        iso_value_spread=spread,
    )
    return PoissonIndicator(
        sampled - iso, _regular_field(grid, record.values - iso), evidence
    )


def extract_indicator_surface(
    indicator: PoissonIndicator, /
) -> tuple[np.ndarray, np.ndarray]:
    """Oriented level set ``field = 0`` of the piecewise-linear tetrahedral field.

    Vertices with ``field < 0`` are inside. Vertex values are held at least
    ``_LEVEL_SET_CLEARANCE`` (a fraction of the unit indicator jump) away from
    zero on their own side, so crossings stay a resolvable distance from
    tetrahedron vertices instead of collapsing into slivers; this moves the
    level set by at most that fraction of a cell. Level-set vertices are numbered by
    their crossing tetrahedron edge (inside, outside vertex pair), so shared
    tetrahedron faces share vertices exactly; each triangle is oriented toward
    the outside vertex of largest value in its tetrahedron.
    """

    field = indicator.field
    count = field.values.shape[0]
    values = np.where(
        field.values < 0.0,
        np.minimum(field.values, -_LEVEL_SET_CLEARANCE),
        np.maximum(field.values, _LEVEL_SET_CLEARANCE),
    )
    tetra_values = values[field.tetrahedra]
    case = np.sum((tetra_values < 0.0) << np.arange(4, dtype=np.int64), axis=1)
    counts = _CASE_COUNTS[case]
    owner = np.repeat(np.arange(field.tetrahedra.shape[0], dtype=np.int64), counts)
    slot = np.arange(owner.size, dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    local = _CASE_TRIANGLES[case[owner], slot]
    inner = np.take_along_axis(field.tetrahedra[owner], local[..., 0], axis=1)
    outer = np.take_along_axis(field.tetrahedra[owner], local[..., 1], axis=1)
    keys, vertex_of = np.unique(
        (inner * count + outer).reshape((-1,)), return_inverse=True
    )
    inner_nodes, outer_nodes = keys // count, keys % count
    parameter = values[inner_nodes] / (values[inner_nodes] - values[outer_nodes])
    vertices = field.points[inner_nodes] + parameter[:, None] * (
        field.points[outer_nodes] - field.points[inner_nodes]
    )
    faces = vertex_of.reshape((-1, 3)).astype(np.int64)
    triangles = vertices[faces]
    normal = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    reference = field.points[
        field.tetrahedra[owner, np.argmax(tetra_values[owner], axis=1)]
    ]
    flip = np.sum(normal * (reference - triangles[:, 0]), axis=1) < 0.0
    faces = np.where(flip[:, None], faces[:, (0, 2, 1)], faces)
    return vertices, faces.astype(np.int32)


def sampled_surface_deviation(
    points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, /
) -> SampledSurfaceDeviation:
    """Exact sample-to-triangle and vertex-to-nearest-sample distances."""

    triangles = TriangleBVH(TriangleMesh(vertices, faces))
    sample_distance = np.asarray(triangles.query(points).distance, dtype=np.float64)
    return SampledSurfaceDeviation(
        sample_to_surface_max=float(np.max(sample_distance)),
        sample_to_surface_mean=float(np.mean(sample_distance)),
        surface_to_sample_max=float(np.max(nearest_sample_distance(points, vertices))),
    )


def nearest_sample_distance(points: np.ndarray, queries: np.ndarray, /) -> np.ndarray:
    """Exact Euclidean distance from every query to its nearest sample."""

    samples = prepare_bvh(points, points, policy=_POINT_POLICY, dtype=jnp.float64)
    nearest = bvh_nearest_items(samples, queries, k=1, query_batch_capacity=256)
    return np.sqrt(np.asarray(nearest.distance_squared, dtype=np.float64)[:, 0])
