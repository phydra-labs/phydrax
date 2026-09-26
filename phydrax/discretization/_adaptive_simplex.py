#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Device-resident fixed-capacity simplex topology and compiled bisection.

`MaskedSimplexMesh` is the capacity-bucketed layout consumed by masked solver
routes: vertex and cell slots of a static capacity with activity masks and
array-based half-facet adjacency. Inactive lanes are padding that never
contributes; compiled consumers are keyed by the layout signature (cell kind,
ambient dimension, capacities, coordinate precision), never by active counts.

`AdaptiveSimplexState` extends that layout with the Maubach labels and the
bisection forest. `refine_adaptive_simplex` and `coarsen_adaptive_simplex` are
module-level compiled entry points that replay the host bisection of
`phydrax.meshing` exactly: the same labels (canonical tagged tuples), the same
conformity closure (every cell with a split edge is bisected, iteration by
iteration), the same vertex and cell ID issue order, and the same coarsening
families. Slot order equals global-ID order and slots are never reused inside
one prepared epoch, so ranks by slot are ranks by ID and every issue order is a
prefix sum. Every call records its `AdaptiveSimplexStatus` flags in the state,
where they accumulate over the prepared epoch. A failed call (capacity, closure
limit, protected conflict, invalid geometry) rolls back every mesh, topology,
and numerical array but still records its terminal flags; every later call on
that state is refused on device until a new epoch is prepared.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum, IntFlag
from functools import cache
from itertools import combinations, permutations
from typing import final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._geometry_predicates import orient2d, orient3d, PredicateMode, PredicateSign
from .._strict import StrictModule
from .._trainable import NonTrainableState


_CELL_KINDS = {2: "triangle", 3: "tetrahedron"}
_INDEX_LIMIT = np.iinfo(np.int32).max
_CODE_SENTINEL = np.iinfo(np.int64).max
_MINIMUM_BUCKET = 64


class AdaptiveSimplexCounter(IntEnum):
    """Entries of `AdaptiveSimplexState.counters` (totals since preparation)."""

    REQUESTED_REFINEMENTS = 0
    ACCEPTED_REFINEMENTS = 1
    BISECTIONS = 2
    CLOSURE_ITERATIONS = 3
    CREATED_VERTICES = 4
    REQUESTED_COARSENINGS = 5
    COARSENED_VERTICES = 6
    COARSENING_PASSES = 7
    RESTORED_CELLS = 8
    ADMISSIBILITY_TESTS = 9


_COUNTERS = len(AdaptiveSimplexCounter)


@cache
def maubach_bisection_tables(dimension: int, /) -> tuple[np.ndarray, ...]:
    """Static Maubach templates ``(first, second, reversal, pairs, orders, odd)``.

    Row ``k`` of ``first``/``second`` gathers the children of a simplex tagged
    ``k`` from its tuple extended by the midpoint (column ``d + 1``); row ``k``
    of ``reversal`` reverses the first ``k + 1`` entries (the same tagged
    simplex); ``pairs`` are the local edges; ``orders`` enumerate the vertex
    permutations with their parity ``odd``.
    """

    if dimension not in (2, 3):
        raise ValueError("Maubach bisection tables exist for dimensions 2 and 3.")
    width = dimension + 1
    first = np.tile(np.arange(width, dtype=np.int64), (width, 1))
    second = first.copy()
    reversal = first.copy()
    for tag in range(1, width):
        first[tag, tag] = width
        second[tag, :tag] = np.arange(1, tag + 1)
        second[tag, tag] = width
        reversal[tag, : tag + 1] = np.arange(tag, -1, -1)
    pairs = np.asarray(tuple(combinations(range(width), 2)), dtype=np.int64)
    orders = np.asarray(tuple(permutations(range(width))), dtype=np.int64)
    odd_orders = np.sum(orders[:, pairs[:, 0]] > orders[:, pairs[:, 1]], axis=1) % 2
    return first, second, reversal, pairs, orders, odd_orders == 1


def _count(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def masked_simplex_signature(
    cell_kind: str,
    ambient_dimension: int,
    vertex_capacity: int,
    cell_capacity: int,
    coordinate_dtype: np.dtype,
    /,
) -> str:
    """Compile identity of one capacity bucket (never of an active count)."""

    return canonical_fingerprint(
        {
            "kind": "masked-simplex-mesh",
            "cell_kind": cell_kind,
            "ambient_dimension": int(ambient_dimension),
            "vertex_capacity": int(vertex_capacity),
            "cell_capacity": int(cell_capacity),
            "coordinate_dtype": np.dtype(coordinate_dtype).name,
        }
    )


@final
class MaskedSimplexMesh(StrictModule, NonTrainableState):
    """Capacity-bucketed simplex topology with activity masks.

    ``cells[c]`` holds the positively oriented vertex slots of cell slot ``c``;
    rows of inactive cells are in-range padding that never contributes.
    ``facet_neighbors[c, i]`` is the packed half-facet ``slot * (d + 1) + local``
    of the other active cell sharing the facet opposite local vertex ``i``, or
    ``-1`` on the boundary and on inactive cells. ``vertex_ids``/``cell_ids`` are
    the global IDs of allocated slots (``-1`` otherwise); vertex and cell slot
    order equals global-ID order. Static fields form the compile identity.
    """

    cell_kind: str = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    vertex_capacity: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)
    coordinates: Array
    vertex_ids: Array
    vertex_active: Array
    cells: Array
    cell_ids: Array
    cell_active: Array
    facet_neighbors: Array

    def __init__(
        self,
        coordinates: ArrayLike,
        vertex_ids: ArrayLike,
        vertex_active: ArrayLike,
        cells: ArrayLike,
        cell_ids: ArrayLike,
        cell_active: ArrayLike,
        facet_neighbors: ArrayLike,
        /,
    ):
        points = jnp.asarray(coordinates)
        rows = jnp.asarray(cells)
        if points.ndim != 2 or not jnp.issubdtype(points.dtype, jnp.floating):
            raise TypeError("coordinates must be a floating (vertices, ambient) array.")
        if rows.ndim != 2 or rows.dtype != jnp.int32:
            raise TypeError("cells must be an int32 (cells, d + 1) array.")
        dimension = rows.shape[1] - 1
        kind = _CELL_KINDS.get(dimension)
        if kind is None:
            raise ValueError("Masked simplex meshes hold triangles or tetrahedra.")
        ambient = points.shape[1]
        if ambient < dimension or ambient > 3:
            raise ValueError("The ambient dimension must lie in [d, 3].")
        vertex_count = _count(points.shape[0], "vertex capacity")
        cell_count = _count(rows.shape[0], "cell capacity")
        if vertex_count > _INDEX_LIMIT or cell_count * (dimension + 1) > _INDEX_LIMIT:
            raise ValueError("Masked simplex capacities must address int32 slots.")
        arrays = {
            "vertex_ids": (jnp.asarray(vertex_ids), (vertex_count,), jnp.int64),
            "vertex_active": (jnp.asarray(vertex_active), (vertex_count,), jnp.bool_),
            "cell_ids": (jnp.asarray(cell_ids), (cell_count,), jnp.int64),
            "cell_active": (jnp.asarray(cell_active), (cell_count,), jnp.bool_),
            "facet_neighbors": (
                jnp.asarray(facet_neighbors),
                (cell_count, dimension + 1),
                jnp.int32,
            ),
        }
        for name, (value, shape, dtype) in arrays.items():
            if value.shape != shape or value.dtype != dtype:
                raise TypeError(f"{name} must be {jnp.dtype(dtype).name} {shape}.")
        self.cell_kind = kind
        self.ambient_dimension = ambient
        self.vertex_capacity = vertex_count
        self.cell_capacity = cell_count
        self.signature_id = masked_simplex_signature(
            kind, ambient, vertex_count, cell_count, np.dtype(points.dtype)
        )
        self.coordinates = points
        self.vertex_ids = arrays["vertex_ids"][0]
        self.vertex_active = arrays["vertex_active"][0]
        self.cells = rows
        self.cell_ids = arrays["cell_ids"][0]
        self.cell_active = arrays["cell_active"][0]
        self.facet_neighbors = arrays["facet_neighbors"][0]

    @property
    def dimension(self) -> int:
        return self.cells.shape[1] - 1

    @property
    def boundary_facets(self) -> Array:
        """Active half-facets without a neighbor, shaped ``(cells, d + 1)``."""
        return self.cell_active[:, None] & (self.facet_neighbors < 0)


def _facet_columns(width: int, /) -> np.ndarray:
    return np.asarray(
        [[j for j in range(width) if j != i] for i in range(width)], dtype=np.int32
    )


def masked_simplex_facet_neighbors(cells: Array, cell_active: Array, /) -> Array:
    """Sibling half-facets of the active cells by sort-based facet matching.

    Every active half-facet key (its sorted vertex slots) is sorted once; equal
    adjacent keys pair the two cells sharing a facet. A key shared by three or
    more active cells (never produced by conforming bisection) stays unmatched.
    """

    count, width = cells.shape
    keys = jnp.sort(cells[:, _facet_columns(width)], axis=2)
    keys = keys.reshape((count * width, width - 1))
    sentinel = jnp.iinfo(jnp.int32).max
    keys = jnp.where(jnp.repeat(cell_active, width)[:, None], keys, sentinel)
    half = jnp.arange(count * width, dtype=jnp.int32)
    operands = tuple(keys[:, column] for column in range(width - 1)) + (half,)
    ordered = jax.lax.sort(operands, num_keys=width - 1, is_stable=True)
    ordered_keys = jnp.stack(ordered[: width - 1], axis=1)
    ordered_half = ordered[-1]
    same_next = jnp.all(ordered_keys[1:] == ordered_keys[:-1], axis=1) & (
        ordered_keys[1:, 0] != sentinel
    )
    pad = jnp.zeros((1,), dtype=jnp.bool_)
    with_next = jnp.concatenate((same_next, pad))
    with_previous = jnp.concatenate((pad, same_next))
    next_half = jnp.roll(ordered_half, -1)
    previous_half = jnp.roll(ordered_half, 1)
    partner = jnp.where(
        with_next & ~with_previous,
        next_half,
        jnp.where(with_previous & ~with_next, previous_half, -1),
    )
    neighbors = jnp.full((count * width,), -1, dtype=jnp.int32)
    return neighbors.at[ordered_half].set(partner).reshape((count, width))


def _vertex_half_facets(
    cells: Array, cell_active: Array, facet_neighbors: Array, vertex_capacity: int, /
) -> Array:
    """Representative half-facet of every active vertex, boundary facets first."""

    count, width = cells.shape
    halves = count * width
    columns = _facet_columns(width)
    half = jnp.arange(halves, dtype=jnp.int64).reshape((count, width))
    key = half + jnp.where(facet_neighbors < 0, 0, halves).astype(jnp.int64)
    vertices = cells[:, columns]
    vertices = jnp.where(cell_active[:, None, None], vertices, vertex_capacity)
    keys = jnp.broadcast_to(key[:, :, None], vertices.shape)
    best = jnp.full((vertex_capacity,), 2 * halves, dtype=jnp.int64)
    best = best.at[vertices.reshape((-1,))].min(keys.reshape((-1,)), mode="drop")
    return jnp.where(best < 2 * halves, best % halves, -1).astype(jnp.int32)


class AdaptiveSimplexStatus(IntFlag):
    """Outcome flags of one device adaptation call, accumulated per epoch.

    ``CAPACITY_EXCEEDED``, ``CLOSURE_LIMIT``, ``PROTECTED_CONFLICT`` and
    ``INVALID_GEOMETRY`` are terminal failures: the returned state keeps the
    input topology and arrays but records the flags in
    `AdaptiveSimplexState.status_flags`, and every later call on it is refused
    (``report.failed``) until a new epoch is prepared (for resources, with
    larger capacities). ``PASS_LIMIT`` (coarsening stopped at its pass bound)
    and ``NEEDS_HOST_RESOLUTION`` (some filtered device predicate is uncertain)
    are applied outcomes: the commit reports a pass-limited epoch as
    ``PASS_LIMIT`` and resolves uncertain orientation with exact host
    predicates or rejects the epoch.
    """

    COMPLETE = 0
    CAPACITY_EXCEEDED = 1
    CLOSURE_LIMIT = 2
    PROTECTED_CONFLICT = 4
    PASS_LIMIT = 8
    NEEDS_HOST_RESOLUTION = 16
    INVALID_GEOMETRY = 32


_FAILURES = int(
    AdaptiveSimplexStatus.CAPACITY_EXCEEDED
    | AdaptiveSimplexStatus.CLOSURE_LIMIT
    | AdaptiveSimplexStatus.PROTECTED_CONFLICT
    | AdaptiveSimplexStatus.INVALID_GEOMETRY
)


def adaptive_simplex_bucket(count: int, growth_factor: float, /) -> int:
    """Capacity bucket: the next power of two holding ``growth_factor * count``."""

    required = max(int(math.ceil(growth_factor * max(int(count), 1))), _MINIMUM_BUCKET)
    return 1 << (required - 1).bit_length()


@final
class AdaptiveSimplexPolicy(StrictModule, NonTrainableState):
    """Capacity bucket and coarsening pass bound of one device simplex layout.

    Explicit ``vertex_capacity``/``cell_capacity`` pin the bucket; otherwise a
    capacity is the next power of two holding ``growth_factor`` times the
    prepared slot count (at least 64). Repeated adaptation inside one bucket
    reuses every compiled entry point.
    """

    vertex_capacity: int | None = eqx.field(static=True)
    cell_capacity: int | None = eqx.field(static=True)
    growth_factor: float = eqx.field(static=True)
    maximum_coarsening_passes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        vertex_capacity: int | None = None,
        cell_capacity: int | None = None,
        growth_factor: float = 4.0,
        maximum_coarsening_passes: int = 64,
    ):
        vertices = (
            None
            if vertex_capacity is None
            else _count(vertex_capacity, "vertex_capacity")
        )
        cells = None if cell_capacity is None else _count(cell_capacity, "cell_capacity")
        if isinstance(growth_factor, bool) or not isinstance(growth_factor, (int, float)):
            raise TypeError("growth_factor must be a real number.")
        growth = float(growth_factor)
        if not math.isfinite(growth) or growth < 1.0:
            raise ValueError("growth_factor must be finite and at least one.")
        passes = _count(maximum_coarsening_passes, "maximum_coarsening_passes")
        self.vertex_capacity = vertices
        self.cell_capacity = cells
        self.growth_factor = growth
        self.maximum_coarsening_passes = passes
        self.policy_id = canonical_fingerprint(
            {
                "kind": "adaptive-simplex-policy",
                "vertex_capacity": vertices,
                "cell_capacity": cells,
                "growth_factor": growth,
                "maximum_coarsening_passes": passes,
            }
        )

    def capacities(self, vertex_count: int, cell_count: int, /) -> tuple[int, int]:
        """Vertex and cell capacities of a layout preparing the given slot counts."""

        vertices = (
            adaptive_simplex_bucket(vertex_count, self.growth_factor)
            if self.vertex_capacity is None
            else self.vertex_capacity
        )
        cells = (
            adaptive_simplex_bucket(cell_count, self.growth_factor)
            if self.cell_capacity is None
            else self.cell_capacity
        )
        if vertices < vertex_count or cells < cell_count:
            raise ValueError(
                f"The prepared mesh ({vertex_count} vertices, {cell_count} cell slots) "
                f"exceeds the adaptive simplex capacity ({vertices}, {cells})."
            )
        return vertices, cells


@final
class AdaptiveSimplexLayout(StrictModule, NonTrainableState):
    """Static compile identity of one adaptive simplex capacity bucket.

    Every field is static: compiled entry points are cached per layout, never
    per source topology, active count, or adaptation cycle.
    """

    dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    vertex_capacity: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    protected_edge_capacity: int = eqx.field(static=True)
    maximum_closure_iterations: int = eqx.field(static=True)
    maximum_coarsening_passes: int = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)
    mesh_signature_id: str = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        ambient_dimension: int,
        /,
        *,
        vertex_capacity: int,
        cell_capacity: int,
        protected_edge_capacity: int,
        maximum_closure_iterations: int,
        maximum_coarsening_passes: int,
        coordinate_dtype: np.dtype = np.dtype(np.float64),
    ):
        kind = _CELL_KINDS.get(dimension)
        if kind is None or isinstance(dimension, bool):
            raise ValueError("Adaptive simplex layouts are two- or three-dimensional.")
        if ambient_dimension not in range(dimension, 4):
            raise ValueError("The ambient dimension must lie in [d, 3].")
        vertices = _count(vertex_capacity, "vertex_capacity")
        cells = _count(cell_capacity, "cell_capacity")
        if vertices > _INDEX_LIMIT or 2 * cells * (dimension + 1) > _INDEX_LIMIT:
            raise ValueError("Adaptive simplex capacities must address int32 slots.")
        protected = _count(protected_edge_capacity, "protected_edge_capacity")
        closure = _count(maximum_closure_iterations, "maximum_closure_iterations")
        passes = _count(maximum_coarsening_passes, "maximum_coarsening_passes")
        dtype = np.dtype(coordinate_dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("Adaptive simplex coordinates are float32 or float64.")
        mesh_signature = masked_simplex_signature(
            kind, ambient_dimension, vertices, cells, dtype
        )
        self.dimension = int(dimension)
        self.ambient_dimension = int(ambient_dimension)
        self.vertex_capacity = vertices
        self.cell_capacity = cells
        self.protected_edge_capacity = protected
        self.maximum_closure_iterations = closure
        self.maximum_coarsening_passes = passes
        self.coordinate_dtype = dtype.name
        self.mesh_signature_id = mesh_signature
        self.signature_id = canonical_fingerprint(
            {
                "kind": "adaptive-simplex-layout",
                "mesh": mesh_signature,
                "protected_edge_capacity": protected,
                "maximum_closure_iterations": closure,
                "maximum_coarsening_passes": passes,
            }
        )

    @property
    def cell_kind(self) -> str:
        return _CELL_KINDS[self.dimension]


@final
class AdaptiveSimplexState(StrictModule, NonTrainableState):
    """Dynamic arrays of one fixed-capacity adaptive simplex epoch.

    ``mesh`` is the solver-visible layout. Per cell slot: canonical Maubach
    ``tuples`` and ``tags``, source ``blocks``, ``generations``, packed
    ``parents`` (``parent * 2 + ordinal``, ``-1`` for forest roots), the
    ``children`` and ``bisection_vertices`` of the latest bisection, ``retired``
    (coarsened away; never reused), organization ``cell_classes`` and
    ``facet_classes`` aligned with ``mesh.cells`` (class 0: facets created by
    bisection). Per vertex slot: the split edge ``vertex_parents``, the closure
    ``vertex_levels``, the coarsening pass of removal (``-1`` while alive), and
    protection. ``protected_codes`` are sorted edge codes ``low * V + high`` of
    vertex slots. ``cursors`` hold the next vertex/cell slot and global ID,
    ``clocks`` the cumulative closure level, coarsening pass, and status flags
    (`status_flags`), ``counters`` the evidence totals since preparation, and
    ``refine_rejected`` / ``coarsen_marked`` the cumulative mark evidence.
    """

    mesh: MaskedSimplexMesh
    vertex_half_facets: Array
    tuples: Array
    tags: Array
    blocks: Array
    generations: Array
    parents: Array
    children: Array
    bisection_vertices: Array
    retired: Array
    cell_classes: Array
    facet_classes: Array
    vertex_parents: Array
    vertex_levels: Array
    vertex_removal: Array
    vertex_protected: Array
    protected_codes: Array
    refine_rejected: Array
    coarsen_marked: Array
    cursors: Array
    clocks: Array
    counters: Array

    def __init__(
        self,
        mesh: MaskedSimplexMesh,
        /,
        *,
        vertex_half_facets: ArrayLike,
        tuples: ArrayLike,
        tags: ArrayLike,
        blocks: ArrayLike,
        generations: ArrayLike,
        parents: ArrayLike,
        children: ArrayLike,
        bisection_vertices: ArrayLike,
        retired: ArrayLike,
        cell_classes: ArrayLike,
        facet_classes: ArrayLike,
        vertex_parents: ArrayLike,
        vertex_levels: ArrayLike,
        vertex_removal: ArrayLike,
        vertex_protected: ArrayLike,
        protected_codes: ArrayLike,
        refine_rejected: ArrayLike,
        coarsen_marked: ArrayLike,
        cursors: ArrayLike,
        clocks: ArrayLike,
        counters: ArrayLike,
    ):
        if not isinstance(mesh, MaskedSimplexMesh):
            raise TypeError("mesh must be MaskedSimplexMesh.")
        cells, vertices = mesh.cell_capacity, mesh.vertex_capacity
        width = mesh.dimension + 1
        specification = {
            "vertex_half_facets": (vertex_half_facets, (vertices,), jnp.int32),
            "tuples": (tuples, (cells, width), jnp.int32),
            "tags": (tags, (cells,), jnp.int32),
            "blocks": (blocks, (cells,), jnp.int32),
            "generations": (generations, (cells,), jnp.int32),
            "parents": (parents, (cells,), jnp.int32),
            "children": (children, (cells, 2), jnp.int32),
            "bisection_vertices": (bisection_vertices, (cells,), jnp.int32),
            "retired": (retired, (cells,), jnp.bool_),
            "cell_classes": (cell_classes, (cells,), jnp.int32),
            "facet_classes": (facet_classes, (cells, width), jnp.int32),
            "vertex_parents": (vertex_parents, (vertices, 2), jnp.int32),
            "vertex_levels": (vertex_levels, (vertices,), jnp.int32),
            "vertex_removal": (vertex_removal, (vertices,), jnp.int32),
            "vertex_protected": (vertex_protected, (vertices,), jnp.bool_),
            "refine_rejected": (refine_rejected, (cells,), jnp.bool_),
            "coarsen_marked": (coarsen_marked, (cells,), jnp.bool_),
            "cursors": (cursors, (4,), jnp.int64),
            "clocks": (clocks, (3,), jnp.int32),
            "counters": (counters, (_COUNTERS,), jnp.int64),
        }
        arrays = {}
        for name, (value, shape, dtype) in specification.items():
            array = jnp.asarray(value)
            if array.shape != shape or array.dtype != dtype:
                raise TypeError(f"{name} must be {jnp.dtype(dtype).name} {shape}.")
            arrays[name] = array
        codes = jnp.asarray(protected_codes)
        if codes.ndim != 1 or codes.dtype != jnp.int64 or codes.shape[0] < 1:
            raise TypeError("protected_codes must be a non-empty int64 vector.")
        self.mesh = mesh
        self.vertex_half_facets = arrays["vertex_half_facets"]
        self.tuples = arrays["tuples"]
        self.tags = arrays["tags"]
        self.blocks = arrays["blocks"]
        self.generations = arrays["generations"]
        self.parents = arrays["parents"]
        self.children = arrays["children"]
        self.bisection_vertices = arrays["bisection_vertices"]
        self.retired = arrays["retired"]
        self.cell_classes = arrays["cell_classes"]
        self.facet_classes = arrays["facet_classes"]
        self.vertex_parents = arrays["vertex_parents"]
        self.vertex_levels = arrays["vertex_levels"]
        self.vertex_removal = arrays["vertex_removal"]
        self.vertex_protected = arrays["vertex_protected"]
        self.protected_codes = codes
        self.refine_rejected = arrays["refine_rejected"]
        self.coarsen_marked = arrays["coarsen_marked"]
        self.cursors = arrays["cursors"]
        self.clocks = arrays["clocks"]
        self.counters = arrays["counters"]

    @property
    def status_flags(self) -> Array:
        """Cumulative `AdaptiveSimplexStatus` flags of the epoch (traceable).

        Applied calls add ``PASS_LIMIT`` and ``NEEDS_HOST_RESOLUTION``; a failed
        call rolls back every mesh, topology, and numerical array but still
        records its flags, after which every call on the state is refused.
        Stacked part states hold one entry per part.
        """

        return self.clocks[..., 2]


@final
class AdaptiveSimplexReport(StrictModule):
    """Traceable evidence of one device refinement or coarsening call.

    ``status`` holds `AdaptiveSimplexStatus` flags. ``requested``/``accepted``
    count the marks on active cells; ``rejected`` marks the refinement marks
    whose own closure would split a protected edge, or the coarsening marks
    that stay active. ``operations`` counts bisections or restored parents,
    ``iterations`` closure iterations or coarsening passes, ``vertices``
    created or removed vertices. Geometry evidence covers the attempted
    candidate: on success that is the returned state's active cells; on failure
    it records why the candidate was rejected while the state arrays roll back.
    A call on a state whose `status_flags` hold a terminal failure is refused:
    the state is returned unchanged, ``status`` holds those terminal flags, and
    the accepted, operation, iteration, and vertex counts are zero.
    """

    status: Array
    requested: Array
    accepted: Array
    rejected: Array
    operations: Array
    iterations: Array
    vertices: Array
    uncertain_cells: Array
    invalid_cells: Array
    minimum_quality: Array

    @property
    def failed(self) -> Array:
        return (self.status & _FAILURES) != 0


@final
class AdaptiveSimplexUpdate(StrictModule):
    """State and report of one compiled device adaptation call."""

    state: AdaptiveSimplexState
    report: AdaptiveSimplexReport


class _Work(NamedTuple):
    """Array view of one state inside a compiled kernel."""

    coordinates: Array
    vertex_ids: Array
    vertex_active: Array
    cells: Array
    cell_ids: Array
    cell_active: Array
    tuples: Array
    tags: Array
    blocks: Array
    generations: Array
    parents: Array
    children: Array
    bisection_vertices: Array
    retired: Array
    cell_classes: Array
    facet_classes: Array
    vertex_parents: Array
    vertex_levels: Array
    vertex_removal: Array
    vertex_protected: Array
    protected_codes: Array
    refine_rejected: Array
    coarsen_marked: Array
    cursors: Array
    clocks: Array
    counters: Array


def _work(state: AdaptiveSimplexState, /) -> _Work:
    mesh = state.mesh
    return _Work(
        mesh.coordinates,
        mesh.vertex_ids,
        mesh.vertex_active,
        mesh.cells,
        mesh.cell_ids,
        mesh.cell_active,
        state.tuples,
        state.tags,
        state.blocks,
        state.generations,
        state.parents,
        state.children,
        state.bisection_vertices,
        state.retired,
        state.cell_classes,
        state.facet_classes,
        state.vertex_parents,
        state.vertex_levels,
        state.vertex_removal,
        state.vertex_protected,
        state.protected_codes,
        state.refine_rejected,
        state.coarsen_marked,
        state.cursors,
        state.clocks,
        state.counters,
    )


def _state(work: _Work, /) -> AdaptiveSimplexState:
    """Rebuild the public state; derived adjacency is recomputed from the cells."""

    neighbors = masked_simplex_facet_neighbors(work.cells, work.cell_active)
    mesh = MaskedSimplexMesh(
        work.coordinates,
        work.vertex_ids,
        work.vertex_active,
        work.cells,
        work.cell_ids,
        work.cell_active,
        neighbors,
    )
    return AdaptiveSimplexState(
        mesh,
        vertex_half_facets=_vertex_half_facets(
            work.cells, work.cell_active, neighbors, work.vertex_ids.shape[0]
        ),
        tuples=work.tuples,
        tags=work.tags,
        blocks=work.blocks,
        generations=work.generations,
        parents=work.parents,
        children=work.children,
        bisection_vertices=work.bisection_vertices,
        retired=work.retired,
        cell_classes=work.cell_classes,
        facet_classes=work.facet_classes,
        vertex_parents=work.vertex_parents,
        vertex_levels=work.vertex_levels,
        vertex_removal=work.vertex_removal,
        vertex_protected=work.vertex_protected,
        protected_codes=work.protected_codes,
        refine_rejected=work.refine_rejected,
        coarsen_marked=work.coarsen_marked,
        cursors=work.cursors,
        clocks=work.clocks,
        counters=work.counters,
    )


def _select(condition: Array, first, second, /):
    return jax.tree_util.tree_map(
        lambda left, right: jnp.where(condition, left, right), first, second
    )


def _take_rows(values: Array, columns: Array, /) -> Array:
    return jnp.take_along_axis(values, columns, axis=1)


def _edge_codes(first: Array, second: Array, vertex_capacity: int, /) -> Array:
    low = jnp.minimum(first, second).astype(jnp.int64)
    high = jnp.maximum(first, second).astype(jnp.int64)
    return low * vertex_capacity + high


def _members(table: Array, queries: Array, /) -> tuple[Array, Array]:
    """Membership of queries in one ascending sentinel-padded table and positions."""

    position = jnp.clip(jnp.searchsorted(table, queries), 0, table.shape[0] - 1)
    return (table[position] == queries) & (queries != _CODE_SENTINEL), position


def _refinement_edges(work: _Work, /) -> tuple[Array, Array]:
    first = work.tuples[:, 0]
    second = _take_rows(work.tuples, work.tags[:, None])[:, 0]
    return first, second


def _all_edge_codes(tuples: Array, dimension: int, vertex_capacity: int, /) -> Array:
    pairs = maubach_bisection_tables(dimension)[3]
    return _edge_codes(tuples[:, pairs[:, 0]], tuples[:, pairs[:, 1]], vertex_capacity)


def _odd_relative(rows: Array, tuples: Array, dimension: int, /) -> Array:
    """Whether each tuple is an odd permutation of its oriented row."""

    pairs = maubach_bisection_tables(dimension)[3]
    position = jnp.argmax(tuples[:, :, None] == rows[:, None, :], axis=2)
    inversions = jnp.sum(position[:, pairs[:, 0]] > position[:, pairs[:, 1]], axis=1)
    return inversions % 2 == 1


def _canonical(tuples: Array, tags: Array, odd: Array, dimension: int, /):
    """Canonical tagged tuple (smaller by slot = by global ID) and its parity."""

    reversal = jnp.asarray(maubach_bisection_tables(dimension)[2], dtype=jnp.int32)
    reversed_ = _take_rows(tuples, reversal[tags])
    differs = reversed_ != tuples
    first = jnp.argmax(differs, axis=1)[:, None]
    smaller = jnp.any(differs, axis=1) & (
        _take_rows(reversed_, first)[:, 0] < _take_rows(tuples, first)[:, 0]
    )
    flips = ((tags * (tags + 1)) // 2) % 2 == 1
    return jnp.where(smaller[:, None], reversed_, tuples), odd ^ (smaller & flips)


def _oriented_rows(tuples: Array, odd: Array, /) -> Array:
    last = jnp.where(odd, tuples[:, -2], tuples[:, -1])
    before = jnp.where(odd, tuples[:, -1], tuples[:, -2])
    return tuples.at[:, -2].set(before).at[:, -1].set(last)


def _class_opposite(rows: Array, classes: Array, vertices: Array, /) -> Array:
    """Facet class opposite each queried vertex (``(n, k)`` queries per row)."""

    match = vertices[:, :, None] == rows[:, None, :]
    return jnp.sum(jnp.where(match, classes[:, None, :], 0), axis=2, dtype=classes.dtype)


class _Closure(NamedTuple):
    work: _Work
    selected: Array
    pending: Array
    keys: Array
    values: Array
    stored: Array
    status: Array
    iterations: Array
    bisections: Array
    created: Array


class _Issue(NamedTuple):
    """Slots and global IDs one bisection round issues on one part."""

    midpoint: Array
    vertex_slots: Array
    vertex_ends: Array
    vertex_ids: Array
    table_keys: Array
    table_slots: Array
    first_id: Array
    created_local: Array
    created_global: Array
    count_local: Array
    count_global: Array
    conflict: Array


def _global_any(value: Array, axis: str | None, /) -> Array:
    if axis is None:
        return value
    return jax.lax.psum(value.astype(jnp.int32), axis) > 0


def _global_flags(status: Array, axis: str | None, /) -> Array:
    """Union of status flags over all parts."""

    if axis is None:
        return status
    bits = (status[None] >> jnp.arange(8, dtype=jnp.int32)) & 1
    union = jax.lax.psum(bits, axis) > 0
    weights = union.astype(jnp.int32) << jnp.arange(8, dtype=jnp.int32)
    return jnp.sum(weights, dtype=jnp.int32)


def _local_issue(closure: _Closure, known, position, codes, guard: bool, /) -> _Issue:
    """Single-part issue: new edges by slot code (slot order is ID order)."""

    work, selected = closure.work, closure.selected
    vertices, cells = work.vertex_ids.shape[0], work.cell_ids.shape[0]
    ordered = jnp.sort(jnp.where(selected & ~known, codes, _CODE_SENTINEL))
    fresh = (ordered != _CODE_SENTINEL) & jnp.concatenate(
        (jnp.ones((1,), dtype=jnp.bool_), ordered[1:] != ordered[:-1])
    )
    rank = jnp.cumsum(fresh, dtype=jnp.int32) - 1
    created = jnp.sum(fresh, dtype=jnp.int32)
    vertex_slot = work.cursors[0].astype(jnp.int32)
    cell_position = jnp.clip(jnp.searchsorted(ordered, codes), 0, cells - 1)
    count = jnp.sum(selected, dtype=jnp.int32)
    order = jnp.cumsum(selected, dtype=jnp.int32) - 1
    conflict = jnp.zeros((), dtype=jnp.bool_)
    if guard:
        protected, _ = _members(work.protected_codes, ordered)
        conflict = jnp.any(fresh & protected)
    return _Issue(
        jnp.where(known, closure.values[position], vertex_slot + rank[cell_position]),
        jnp.where(fresh, vertex_slot + rank, vertices),
        jnp.stack(
            (
                jnp.where(fresh, ordered // vertices, 0).astype(jnp.int32),
                jnp.where(fresh, ordered % vertices, 0).astype(jnp.int32),
            ),
            axis=1,
        ),
        work.cursors[2] + rank.astype(jnp.int64),
        jnp.where(fresh, ordered, _CODE_SENTINEL),
        vertex_slot + rank,
        work.cursors[3] + 2 * order.astype(jnp.int64),
        created,
        created,
        count,
        count,
        conflict,
    )


def _parts_issue(
    closure: _Closure, known, position, codes, guard: bool, axis: str, /
) -> _Issue:
    """Part-independent issue: new vertex and cell IDs are global ranks.

    Every part gathers the candidate refinement edges (as global vertex-ID
    pairs) and the selected cell IDs of all parts; ranks in the gathered sorted
    order are the single-part issue order, so IDs do not depend on ownership.
    Each part materializes the midpoints of the new edges it holds.
    """

    work, selected = closure.work, closure.selected
    vertices = work.vertex_ids.shape[0]
    ids = work.vertex_ids
    first_end, second_end = _refinement_edges(work)
    low, high = jnp.minimum(first_end, second_end), jnp.maximum(first_end, second_end)
    candidate = selected & ~known
    gathered = (
        jax.lax.all_gather(jnp.where(candidate, ids[low], _CODE_SENTINEL), axis),
        jax.lax.all_gather(jnp.where(candidate, ids[high], _CODE_SENTINEL), axis),
    )
    pair_low, pair_high = jax.lax.sort(
        tuple(value.reshape((-1,)) for value in gathered), num_keys=2
    )
    fresh = (pair_low != _CODE_SENTINEL) & jnp.concatenate(
        (
            jnp.ones((1,), dtype=jnp.bool_),
            (pair_low[1:] != pair_low[:-1]) | (pair_high[1:] != pair_high[:-1]),
        )
    )
    global_rank = jnp.cumsum(fresh, dtype=jnp.int64) - 1
    slots = jnp.arange(vertices, dtype=jnp.int64)
    lookup = jnp.where(slots < work.cursors[0], ids, _CODE_SENTINEL)
    low_slot, low_found = _slot_of(lookup, pair_low)
    high_slot, high_found = _slot_of(lookup, pair_high)
    relevant = fresh & low_found & high_found
    local_rank = jnp.cumsum(relevant, dtype=jnp.int32) - 1
    vertex_slot = work.cursors[0].astype(jnp.int32)
    local_codes = jnp.where(
        relevant, _edge_codes(low_slot, high_slot, vertices), _CODE_SENTINEL
    )
    # Relevant codes ascend in gathered order (slot order is ID order).
    compact = jnp.sort(local_codes)
    midpoint_rank = jnp.searchsorted(compact, codes).astype(jnp.int32)
    selected_ids = jax.lax.all_gather(
        jnp.where(selected, work.cell_ids, _CODE_SENTINEL), axis
    ).reshape((-1,))
    ordered_ids = jnp.sort(selected_ids)
    cell_rank = jnp.searchsorted(ordered_ids, work.cell_ids).astype(jnp.int64)
    return _Issue(
        jnp.where(known, closure.values[position], vertex_slot + midpoint_rank),
        jnp.where(relevant, vertex_slot + local_rank, vertices),
        jnp.stack((low_slot, high_slot), axis=1),
        work.cursors[2] + global_rank,
        local_codes,
        vertex_slot + local_rank,
        work.cursors[3] + 2 * cell_rank,
        jnp.sum(relevant, dtype=jnp.int32),
        jnp.sum(fresh, dtype=jnp.int32),
        jnp.sum(selected, dtype=jnp.int32),
        jnp.sum(ordered_ids != _CODE_SENTINEL, dtype=jnp.int32),
        jnp.any(_members(work.protected_codes, local_codes)[0]) if guard else False,
    )


def _slot_of(lookup: Array, identifiers: Array, /) -> tuple[Array, Array]:
    """Local slot of each global vertex ID in an ascending sentinel-padded table."""

    position = jnp.clip(jnp.searchsorted(lookup, identifiers), 0, lookup.shape[0] - 1)
    found = (lookup[position] == identifiers) & (identifiers != _CODE_SENTINEL)
    return position.astype(jnp.int32), found


def _issued_vertices(work: _Work, issue: _Issue, level: Array, /) -> dict[str, Array]:
    """Midpoints of the new edges at their issued slots and global IDs."""

    ends, target = issue.vertex_ends, issue.vertex_slots
    points = work.coordinates
    midpoints = 0.5 * points[ends[:, 0]] + 0.5 * points[ends[:, 1]]
    return {
        "coordinates": points.at[target].set(midpoints, mode="drop"),
        "vertex_ids": work.vertex_ids.at[target].set(issue.vertex_ids, mode="drop"),
        "vertex_active": work.vertex_active.at[target].set(True, mode="drop"),
        "vertex_parents": work.vertex_parents.at[target].set(ends, mode="drop"),
        "vertex_levels": work.vertex_levels.at[target].set(level, mode="drop"),
    }


def _children(work: _Work, midpoint: Array, dimension: int, /):
    """Canonical tuples, oriented rows, tags, and facet classes of both children.

    Children of every lane are formed from the static templates; halves inherit
    the class of the parent facet they lie in, the facet opposite the midpoint
    is the parent facet opposite the dropped endpoint, and the new interior
    facet has class 0.
    """

    first_table, second_table = maubach_bisection_tables(dimension)[:2]
    first_end, second_end = _refinement_edges(work)
    extended = jnp.concatenate((work.tuples, midpoint[:, None]), axis=1)
    first_tuple = _take_rows(extended, jnp.asarray(first_table, jnp.int32)[work.tags])
    second_tuple = _take_rows(extended, jnp.asarray(second_table, jnp.int32)[work.tags])
    odd = _odd_relative(work.cells, work.tuples, dimension)
    tags = jnp.where(work.tags > 1, work.tags - 1, dimension)
    first_tuple, first_odd = _canonical(first_tuple, tags, odd, dimension)
    second_tuple, second_odd = _canonical(
        second_tuple, tags, odd ^ (work.tags % 2 == 1), dimension
    )
    first_rows = _oriented_rows(first_tuple, first_odd)
    second_rows = _oriented_rows(second_tuple, second_odd)
    dropped = _class_opposite(
        work.cells, work.facet_classes, jnp.stack((second_end, first_end), axis=1)
    )

    def classes(rows, partner, opposite):
        inherited = _class_opposite(work.cells, work.facet_classes, rows)
        return jnp.where(
            rows == midpoint[:, None],
            opposite,
            jnp.where(rows == partner[:, None], 0, inherited),
        )

    return (
        (first_tuple, first_rows, classes(first_rows, first_end, dropped[:, :1])),
        (second_tuple, second_rows, classes(second_rows, second_end, dropped[:, 1:])),
        tags,
    )


def _bisected(work: _Work, selected: Array, issue: _Issue, dimension: int, /) -> _Work:
    """Scatter the children of the selected cells into their issued slots."""

    cells = work.cell_ids.shape[0]
    cell_slot = work.cursors[1].astype(jnp.int32)
    (first_tuple, first_rows, first_classes), second, tags = _children(
        work, issue.midpoint, dimension
    )
    second_tuple, second_rows, second_classes = second
    order = jnp.cumsum(selected, dtype=jnp.int32) - 1
    first_slot = jnp.where(selected, cell_slot + 2 * order, cells)
    second_slot = jnp.where(selected, cell_slot + 2 * order + 1, cells)
    slots = jnp.concatenate((first_slot, second_slot))
    lanes = jnp.arange(cells, dtype=jnp.int32)

    def children(values_first, values_second, target):
        target = target.at[first_slot].set(values_first, mode="drop")
        return target.at[second_slot].set(values_second, mode="drop")

    reset = jnp.full((2 * cells, 2), -1, dtype=jnp.int32)
    return work._replace(
        cells=children(first_rows, second_rows, work.cells),
        tuples=children(first_tuple, second_tuple, work.tuples),
        tags=children(tags, tags, work.tags),
        blocks=children(work.blocks, work.blocks, work.blocks),
        generations=children(
            work.generations + 1, work.generations + 1, work.generations
        ),
        parents=children(2 * lanes, 2 * lanes + 1, work.parents),
        cell_ids=children(issue.first_id, issue.first_id + 1, work.cell_ids),
        cell_classes=children(work.cell_classes, work.cell_classes, work.cell_classes),
        facet_classes=children(first_classes, second_classes, work.facet_classes),
        cell_active=(work.cell_active & ~selected).at[slots].set(True, mode="drop"),
        children=jnp.where(
            selected[:, None],
            jnp.stack((first_slot, second_slot), axis=1),
            work.children.at[slots].set(reset, mode="drop"),
        ),
        bisection_vertices=jnp.where(
            selected,
            issue.midpoint,
            work.bisection_vertices.at[slots].set(-1, mode="drop"),
        ),
        retired=work.retired.at[slots].set(False, mode="drop"),
        refine_rejected=work.refine_rejected.at[slots].set(False, mode="drop"),
        coarsen_marked=work.coarsen_marked.at[slots].set(False, mode="drop"),
    )


def _bisection_round(
    closure: _Closure, dimension: int, guard: bool, axis: str | None, /
) -> _Closure:
    """Bisect the selected cells once; select every cell holding a split edge.

    New vertices are issued in ascending edge order and children in (parent,
    first, second) order, which are the host global-ID orders. The round is
    applied only when every part's capacities hold (and, when guarded, no
    protected edge would be split); otherwise the status records the refusal.
    """

    work, selected = closure.work, closure.selected
    vertices, cells = work.vertex_ids.shape[0], work.cell_ids.shape[0]
    first_end, second_end = _refinement_edges(work)
    codes = _edge_codes(first_end, second_end, vertices)
    known, position = _members(closure.keys, codes)
    issue = (
        _local_issue(closure, known, position, codes, guard)
        if axis is None
        else _parts_issue(closure, known, position, codes, guard, axis)
    )
    fits = (work.cursors[0] + issue.created_local <= vertices) & (
        work.cursors[1] + 2 * issue.count_local.astype(jnp.int64) <= cells
    )
    status = jnp.where(fits, 0, int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)) | jnp.where(
        issue.conflict, int(AdaptiveSimplexStatus.PROTECTED_CONFLICT), 0
    )
    status = _global_flags(status.astype(jnp.int32), axis)

    def apply(_):
        level = work.clocks[0] + closure.iterations + 1
        updated = _bisected(work, selected, issue, dimension)._replace(
            **_issued_vertices(work, issue, level),
            cursors=work.cursors
            + jnp.stack(
                (
                    issue.created_local.astype(jnp.int64),
                    2 * issue.count_local.astype(jnp.int64),
                    issue.created_global.astype(jnp.int64),
                    2 * issue.count_global.astype(jnp.int64),
                )
            ),
        )
        inserted = issue.table_keys != _CODE_SENTINEL
        entry = closure.stored + jnp.cumsum(inserted, dtype=jnp.int32) - 1
        entry = jnp.where(inserted, entry, vertices)
        keys = closure.keys.at[entry].set(issue.table_keys, mode="drop")
        values = closure.values.at[entry].set(issue.table_slots, mode="drop")
        keys, values = jax.lax.sort((keys, values), num_keys=1)
        split, _ = _members(keys, _all_edge_codes(updated.tuples, dimension, vertices))
        following = updated.cell_active & jnp.any(split, axis=1)
        return _Closure(
            updated,
            following,
            _global_any(jnp.any(following), axis),
            keys,
            values,
            closure.stored + issue.created_local,
            closure.status,
            closure.iterations + 1,
            closure.bisections + issue.count_global,
            closure.created + issue.created_global,
        )

    def refuse(_):
        return closure._replace(status=(closure.status | status).astype(jnp.int32))

    return jax.lax.cond(status == 0, apply, refuse, None)


def _closure(
    work: _Work,
    marks: Array,
    layout: AdaptiveSimplexLayout,
    guard: bool,
    axis: str | None = None,
    /,
) -> _Closure:
    """Conformity closure of the marked cells in one compiled while loop.

    With ``axis`` the loop runs on every part of a shard_map; split edges and
    termination are exchanged by collectives until the global fixed point.
    """

    vertices = layout.vertex_capacity
    start = _Closure(
        work,
        marks,
        _global_any(jnp.any(marks), axis),
        jnp.full((vertices,), _CODE_SENTINEL, dtype=jnp.int64),
        jnp.full((vertices,), -1, dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
    )
    limit = layout.maximum_closure_iterations

    def proceed(closure: _Closure):
        return closure.pending & (closure.status == 0) & (closure.iterations < limit)

    def step(closure: _Closure):
        return _bisection_round(closure, layout.dimension, guard, axis)

    result = jax.lax.while_loop(proceed, step, start)
    unfinished = result.pending & (result.status == 0)
    status = result.status | jnp.where(
        unfinished, int(AdaptiveSimplexStatus.CLOSURE_LIMIT), 0
    )
    return result._replace(status=status.astype(jnp.int32))


def _protected_marks(
    work: _Work, marks: Array, layout: AdaptiveSimplexLayout, /
) -> tuple[Array, Array]:
    """Marks whose own conformity closure would split a protected edge.

    Closures of a union are unions of closures (Stevenson), so the closure of
    one mark reaches exactly the cells and edges forward-reachable from it in
    the forcing graph of the union closure: splitting edge ``e`` bisects every
    cell holding ``e``, and bisecting cell ``c`` splits its refinement edge.
    Reverse reachability from the protected edges is one monotone fixed point.
    """

    dimension, vertices = layout.dimension, layout.vertex_capacity
    cells = layout.cell_capacity
    analysis = _closure(work, marks, layout, False)
    union = analysis.work
    slots = jnp.arange(cells, dtype=jnp.int64)
    universe = (union.cell_ids >= 0) & (work.cell_active | (slots >= work.cursors[1]))
    edges = jnp.where(
        universe[:, None],
        _all_edge_codes(union.tuples, dimension, vertices),
        _CODE_SENTINEL,
    )
    flat = edges.reshape((-1,))
    table = jnp.sort(flat)
    edge_index = jnp.searchsorted(table, flat)
    first_end, second_end = _refinement_edges(union)
    refinement = jnp.searchsorted(table, _edge_codes(first_end, second_end, vertices))
    protected, _ = _members(work.protected_codes, table)
    span = edges.shape[1]

    def spread(state):
        bad_edges, _ = state
        bad_cells = jnp.repeat(universe & bad_edges[refinement], span)
        target = jnp.where(bad_cells & (flat != _CODE_SENTINEL), edge_index, flat.size)
        grown = bad_edges.at[target].set(True, mode="drop")
        return grown, jnp.any(grown != bad_edges)

    bad_edges, _ = jax.lax.while_loop(
        lambda state: state[1], spread, (protected, jnp.ones((), dtype=jnp.bool_))
    )
    rejected = marks & bad_edges[refinement]
    failure = int(
        AdaptiveSimplexStatus.CAPACITY_EXCEEDED | AdaptiveSimplexStatus.CLOSURE_LIMIT
    )
    return rejected, (analysis.status & failure).astype(jnp.int32)


def _shape_quality(points: Array, dimension: int, /) -> Array:
    """Mean-ratio shape quality in (0, 1] (1 for equilateral simplices)."""

    pairs = maubach_bisection_tables(dimension)[3]
    edges = points[:, pairs[:, 1]] - points[:, pairs[:, 0]]
    squared = jnp.sum(jnp.sum(edges * edges, axis=2), axis=1)
    spans = points[:, 1:] - points[:, :1]
    if dimension == 2:
        if points.shape[2] == 2:
            area = 0.5 * jnp.abs(
                spans[:, 0, 0] * spans[:, 1, 1] - spans[:, 0, 1] * spans[:, 1, 0]
            )
        else:
            area = 0.5 * jnp.linalg.norm(jnp.cross(spans[:, 0], spans[:, 1]), axis=1)
        return 4.0 * math.sqrt(3.0) * area / squared
    volume = jnp.abs(jnp.sum(spans[:, 0] * jnp.cross(spans[:, 1], spans[:, 2]), axis=1))
    return 12.0 * (0.5 * volume) ** (2.0 / 3.0) / squared


def _geometry_evidence(work: _Work, dimension: int, axis: str | None, /):
    """FILTERED_DEVICE orientation and shape quality of the active cells (all parts)."""

    points = work.coordinates[work.cells]
    active = work.cell_active
    if points.shape[2] == dimension:
        corners = tuple(points[:, index] for index in range(dimension + 1))
        predicate = orient2d if dimension == 2 else orient3d
        result = predicate(*corners, mode=PredicateMode.FILTERED_DEVICE)
        uncertain = active & ~result.certain
        invalid = active & result.certain & (result.signs != int(PredicateSign.POSITIVE))
    else:
        spans = points[:, 1:] - points[:, :1]
        normal = jnp.cross(spans[:, 0], spans[:, 1])
        uncertain = jnp.zeros(active.shape, dtype=jnp.bool_)
        invalid = active & ~jnp.any(normal != 0.0, axis=1)
    quality = jnp.where(active, _shape_quality(points, dimension), jnp.inf)
    uncertain = jnp.sum(uncertain, dtype=jnp.int32)
    invalid = jnp.sum(invalid, dtype=jnp.int32)
    quality = jnp.min(quality)
    if axis is None:
        return uncertain, invalid, quality
    return (
        jax.lax.psum(uncertain, axis),
        jax.lax.psum(invalid, axis),
        jax.lax.pmin(quality, axis),
    )


def _finished(
    source: _Work,
    work: _Work,
    status: Array,
    dimension: int,
    /,
    *,
    requested: Array,
    accepted: Array,
    rejected: Array,
    operations: Array,
    iterations: Array,
    vertices: Array,
    axis: str | None = None,
) -> AdaptiveSimplexUpdate:
    uncertain, invalid, quality = _geometry_evidence(work, dimension, axis)
    status = (
        status
        | jnp.where(invalid > 0, int(AdaptiveSimplexStatus.INVALID_GEOMETRY), 0)
        | jnp.where(uncertain > 0, int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION), 0)
    ).astype(jnp.int32)
    failed = (status & _FAILURES) != 0
    # A failed call rolls back to the source arrays but keeps its status evidence.
    result = _select(failed, source, work)
    result = result._replace(clocks=result.clocks.at[2].set(source.clocks[2] | status))
    zero = jnp.zeros((), dtype=jnp.int32)
    report = AdaptiveSimplexReport(
        status=status,
        requested=requested,
        accepted=jnp.where(failed, zero, accepted),
        rejected=rejected,
        operations=jnp.where(failed, zero, operations),
        iterations=iterations,
        vertices=jnp.where(failed, zero, vertices),
        uncertain_cells=uncertain,
        invalid_cells=invalid,
        minimum_quality=quality,
    )
    return AdaptiveSimplexUpdate(_state(result), report)


def _terminal(work: _Work, axis: str | None, /) -> Array:
    """Terminal failure flags already recorded by the epoch (on any part)."""

    return (_global_flags(work.clocks[2], axis) & _FAILURES).astype(jnp.int32)


def _refused(
    state: AdaptiveSimplexState,
    work: _Work,
    requested: Array,
    rejected: Array,
    dimension: int,
    axis: str | None = None,
    /,
) -> AdaptiveSimplexUpdate:
    """A call on an epoch that already failed: nothing runs, the state is kept."""

    uncertain, invalid, quality = _geometry_evidence(work, dimension, axis)
    zero = jnp.zeros((), dtype=jnp.int32)
    report = AdaptiveSimplexReport(
        status=_terminal(work, axis),
        requested=requested,
        accepted=zero,
        rejected=rejected,
        operations=zero,
        iterations=zero,
        vertices=zero,
        uncertain_cells=uncertain,
        invalid_cells=invalid,
        minimum_quality=quality,
    )
    return AdaptiveSimplexUpdate(state, report)


def _refine(
    layout: AdaptiveSimplexLayout, state: AdaptiveSimplexState, marks: Array, /
) -> AdaptiveSimplexUpdate:
    work = _work(state)
    marks = marks & work.cell_active
    return jax.lax.cond(
        _terminal(work, None) != 0,
        lambda: _refused(
            state,
            work,
            jnp.sum(marks, dtype=jnp.int32),
            jnp.zeros_like(marks),
            layout.dimension,
        ),
        lambda: _refined(layout, work, marks),
    )


def _refined(
    layout: AdaptiveSimplexLayout, work: _Work, marks: Array, /
) -> AdaptiveSimplexUpdate:
    """Protected admissibility, closure, and evidence of one applied refinement."""

    analysed = jnp.any(work.protected_codes != _CODE_SENTINEL) & jnp.any(marks)
    rejected, analysis_status = jax.lax.cond(
        analysed,
        lambda _: _protected_marks(work, marks, layout),
        lambda _: (jnp.zeros_like(marks), jnp.zeros((), dtype=jnp.int32)),
        None,
    )
    accepted = marks & ~rejected
    closure = _closure(work, accepted, layout, True)
    result = closure.work
    requested = jnp.sum(marks, dtype=jnp.int32)
    accepted_count = jnp.sum(accepted, dtype=jnp.int32)
    increments = jnp.zeros((_COUNTERS,), dtype=jnp.int64)
    increments = increments.at[
        jnp.asarray(
            (
                AdaptiveSimplexCounter.REQUESTED_REFINEMENTS,
                AdaptiveSimplexCounter.ACCEPTED_REFINEMENTS,
                AdaptiveSimplexCounter.BISECTIONS,
                AdaptiveSimplexCounter.CLOSURE_ITERATIONS,
                AdaptiveSimplexCounter.CREATED_VERTICES,
                AdaptiveSimplexCounter.ADMISSIBILITY_TESTS,
            )
        )
    ].set(
        jnp.stack(
            (
                requested,
                accepted_count,
                closure.bisections,
                closure.iterations,
                closure.created,
                analysed.astype(jnp.int32),
            )
        ).astype(jnp.int64)
    )
    result = result._replace(
        refine_rejected=result.refine_rejected | rejected,
        clocks=result.clocks.at[0].add(closure.iterations),
        counters=result.counters + increments,
    )
    return _finished(
        work,
        result,
        closure.status | analysis_status,
        layout.dimension,
        requested=requested,
        accepted=accepted_count,
        rejected=rejected,
        operations=closure.bisections,
        iterations=closure.iterations,
        vertices=closure.created,
    )


def _restored_facet_classes(work: _Work, /) -> Array:
    """Facet classes of every slot as the parent of its current two children.

    The facet opposite ``x_k`` (``x_0``) is the first (second) child's facet
    opposite the bisection vertex; split facets take the first child's half.
    """

    first, second = work.children[:, 0], work.children[:, 1]
    safe_first, safe_second = jnp.maximum(first, 0), jnp.maximum(second, 0)
    midpoint = work.bisection_vertices
    rows = work.cells
    x0 = work.tuples[:, 0]
    xk = _take_rows(work.tuples, work.tags[:, None])[:, 0]
    first_rows, first_classes = rows[safe_first], work.facet_classes[safe_first]
    second_rows, second_classes = rows[safe_second], work.facet_classes[safe_second]
    halves = _class_opposite(first_rows, first_classes, rows)
    towards = _class_opposite(first_rows, first_classes, midpoint[:, None])
    backwards = _class_opposite(second_rows, second_classes, midpoint[:, None])
    return jnp.where(
        rows == xk[:, None],
        towards,
        jnp.where(rows == x0[:, None], backwards, halves),
    )


def _record_agreement(work: _Work, /) -> Array:
    """Whether the two halves of every split facet of each record share a class."""

    first, second = (
        jnp.maximum(work.children[:, 0], 0),
        jnp.maximum(work.children[:, 1], 0),
    )
    rows = work.cells
    x0 = work.tuples[:, 0]
    xk = _take_rows(work.tuples, work.tags[:, None])[:, 0]
    first_half = _class_opposite(rows[first], work.facet_classes[first], rows)
    second_half = _class_opposite(rows[second], work.facet_classes[second], rows)
    split = (rows != x0[:, None]) & (rows != xk[:, None])
    return jnp.all(~split | (first_half == second_half), axis=1)


class _Coarsening(NamedTuple):
    work: _Work
    marked: Array
    passes: Array
    removed: Array
    restored: Array
    progressed: Array


def _coarsening_pass(coarsening: _Coarsening, /) -> _Coarsening:
    """Remove every good vertex: unprotected, star = marked families it created."""

    work, marked = coarsening.work, coarsening.marked
    vertices, cells = work.vertex_ids.shape[0], work.cell_ids.shape[0]
    lanes = jnp.arange(cells, dtype=jnp.int32)
    packed = work.parents
    has_parent = packed >= 0
    parent = jnp.where(has_parent, packed // 2, 0)
    ordinal = packed % 2
    sibling = _take_rows(work.children[parent], (1 - ordinal)[:, None])[:, 0]
    current = _take_rows(work.children[parent], ordinal[:, None])[:, 0] == lanes
    leaf = (
        work.cell_active
        & has_parent
        & current
        & work.cell_active[jnp.maximum(sibling, 0)]
        & (sibling >= 0)
    )
    creator = jnp.where(leaf, work.bisection_vertices[parent], vertices)
    corners = jnp.where(work.cell_active[:, None], work.cells, vertices)
    star = (
        jnp.zeros((vertices,), dtype=jnp.int32)
        .at[corners.reshape((-1,))]
        .add(1, mode="drop")
    )
    created = jnp.zeros((vertices,), dtype=jnp.int32).at[creator].add(1, mode="drop")
    candidate = (star > 0) & (star == created) & ~work.vertex_protected
    member = leaf & candidate[jnp.minimum(creator, vertices - 1)]
    eligible = (
        marked
        & (work.cell_classes == work.cell_classes[jnp.maximum(sibling, 0)])
        & _record_agreement(work)[parent]
    )
    spoiled = (
        jnp.zeros((vertices,), dtype=jnp.bool_)
        .at[jnp.where(member & ~eligible, creator, vertices)]
        .set(True, mode="drop")
    )
    good = candidate & ~spoiled
    removed = member & good[jnp.minimum(creator, vertices - 1)]
    undo = (
        jnp.zeros((cells,), dtype=jnp.bool_)
        .at[jnp.where(removed, parent, cells)]
        .set(True, mode="drop")
    )
    first_child = jnp.maximum(work.children[:, 0], 0)
    pass_index = work.clocks[1] + coarsening.passes
    updated = work._replace(
        cell_active=(work.cell_active & ~removed) | undo,
        retired=work.retired | removed,
        cell_classes=jnp.where(undo, work.cell_classes[first_child], work.cell_classes),
        facet_classes=jnp.where(
            undo[:, None], _restored_facet_classes(work), work.facet_classes
        ),
        vertex_active=work.vertex_active & ~good,
        vertex_removal=jnp.where(good, pass_index, work.vertex_removal),
    )
    progressed = jnp.any(removed)
    return _Coarsening(
        updated,
        (marked & ~removed) | undo,
        coarsening.passes + jnp.where(progressed, 1, 0),
        coarsening.removed + jnp.sum(good, dtype=jnp.int32),
        coarsening.restored + jnp.sum(undo, dtype=jnp.int32),
        progressed,
    )


def _coarsen(
    layout: AdaptiveSimplexLayout, state: AdaptiveSimplexState, marks: Array, /
) -> AdaptiveSimplexUpdate:
    work = _work(state)
    raw = marks & (work.cell_ids >= 0)
    marks = raw & work.cell_active
    # Every mark of a refused call stays active, hence rejected.
    return jax.lax.cond(
        _terminal(work, None) != 0,
        lambda: _refused(
            state, work, jnp.sum(marks, dtype=jnp.int32), raw, layout.dimension
        ),
        lambda: _coarsened(layout, work, raw, marks),
    )


def _coarsened(
    layout: AdaptiveSimplexLayout, work: _Work, raw: Array, marks: Array, /
) -> AdaptiveSimplexUpdate:
    """Bounded coarsening passes and evidence of one applied coarsening."""

    zero = jnp.zeros((), dtype=jnp.int32)
    start = _Coarsening(work, marks, zero, zero, zero, jnp.any(marks))
    limit = layout.maximum_coarsening_passes
    result = jax.lax.while_loop(
        lambda value: value.progressed & (value.passes < limit),
        _coarsening_pass,
        start,
    )
    stopped = result.progressed & (result.passes >= limit)
    status = jnp.where(stopped, int(AdaptiveSimplexStatus.PASS_LIMIT), 0)
    # Marks stay rejected unless coarsened away (bisected marks included).
    rejected = raw & ~result.work.retired
    requested = jnp.sum(marks, dtype=jnp.int32)
    increments = jnp.zeros((_COUNTERS,), dtype=jnp.int64)
    increments = increments.at[
        jnp.asarray(
            (
                AdaptiveSimplexCounter.REQUESTED_COARSENINGS,
                AdaptiveSimplexCounter.COARSENED_VERTICES,
                AdaptiveSimplexCounter.COARSENING_PASSES,
                AdaptiveSimplexCounter.RESTORED_CELLS,
            )
        )
    ].set(
        jnp.stack((requested, result.removed, result.passes, result.restored)).astype(
            jnp.int64
        )
    )
    final = result.work._replace(
        coarsen_marked=result.work.coarsen_marked | raw,
        clocks=result.work.clocks.at[1].add(result.passes),
        counters=result.work.counters + increments,
    )
    return _finished(
        work,
        final,
        status,
        layout.dimension,
        requested=requested,
        accepted=requested - jnp.sum(rejected, dtype=jnp.int32),
        rejected=rejected,
        operations=result.restored,
        iterations=result.passes,
        vertices=result.removed,
    )


_compiled_refine = eqx.filter_jit(_refine)
_compiled_coarsen = eqx.filter_jit(_coarsen)


def _checked(
    layout: AdaptiveSimplexLayout, state: AdaptiveSimplexState, marks: ArrayLike, /
) -> Array:
    if not isinstance(layout, AdaptiveSimplexLayout):
        raise TypeError("layout must be AdaptiveSimplexLayout.")
    if not isinstance(state, AdaptiveSimplexState):
        raise TypeError("state must be AdaptiveSimplexState.")
    mesh = state.mesh
    if mesh.signature_id != layout.mesh_signature_id or state.protected_codes.shape != (
        layout.protected_edge_capacity,
    ):
        raise ValueError("The state does not belong to this adaptive simplex layout.")
    mask = jnp.asarray(marks)
    if mask.shape != (layout.cell_capacity,) or mask.dtype != jnp.bool_:
        raise TypeError(f"marks must be a bool ({layout.cell_capacity},) cell-slot mask.")
    return mask


def refine_adaptive_simplex(
    layout: AdaptiveSimplexLayout, state: AdaptiveSimplexState, marks: ArrayLike, /
) -> AdaptiveSimplexUpdate:
    """Bisect the marked active cells and close to a conforming mesh on device.

    Marks whose own closure would split a protected edge are rejected (reported
    in ``report.rejected``); the others are bisected once and closed. Capacity
    overflow, the closure bound, and invalid children roll the state back and
    record their terminal flags in ``state.status_flags``; a state holding a
    terminal flag refuses every later call.
    """

    return _compiled_refine(layout, state, _checked(layout, state, marks))


def coarsen_adaptive_simplex(
    layout: AdaptiveSimplexLayout, state: AdaptiveSimplexState, marks: ArrayLike, /
) -> AdaptiveSimplexUpdate:
    """Undo bisections whose complete, marked, class-uniform families allow it.

    Pass by pass, every unprotected bisection vertex whose star is exactly the
    marked children of its bisections is removed and their parents restored
    under their original IDs; restored parents join the marks of later passes.
    Coarsened slots are retired and never reused inside the prepared epoch. A
    state holding a terminal flag refuses the call (every mark stays rejected).
    """

    return _compiled_coarsen(layout, state, _checked(layout, state, marks))


@final
class AdaptiveSimplexParts(StrictModule, NonTrainableState):
    """Device mesh of one part-sharded adaptive simplex epoch (static identity).

    Part ``p`` of a stacked state runs on ``devices[p]``; the compiled
    refinement is cached per (layout, parts), never per ownership.
    """

    mesh: Mesh = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)

    def __init__(self, devices: Sequence[jax.Device], /, *, axis_name: str = "parts"):
        chosen = tuple(devices)
        if not chosen or not all(isinstance(device, jax.Device) for device in chosen):
            raise TypeError("devices must be a non-empty sequence of jax.Device.")
        if len(set(chosen)) != len(chosen):
            raise ValueError("Every part needs its own device.")
        if not isinstance(axis_name, str) or not axis_name:
            raise ValueError("axis_name must be a non-empty string.")
        self.mesh = Mesh(np.asarray(chosen, dtype=object), (axis_name,))
        self.axis_name = axis_name

    @property
    def part_count(self) -> int:
        return self.mesh.devices.size


def _refined_part(
    layout: AdaptiveSimplexLayout,
    work: _Work,
    mask: Array,
    requested: Array,
    axis: str,
    /,
) -> AdaptiveSimplexUpdate:
    """Collective closure and evidence of one applied part refinement."""

    closure = _closure(work, mask, layout, True, axis)
    increments = jnp.zeros((_COUNTERS,), dtype=jnp.int64)
    increments = increments.at[
        jnp.asarray(
            (
                AdaptiveSimplexCounter.REQUESTED_REFINEMENTS,
                AdaptiveSimplexCounter.ACCEPTED_REFINEMENTS,
                AdaptiveSimplexCounter.BISECTIONS,
                AdaptiveSimplexCounter.CLOSURE_ITERATIONS,
                AdaptiveSimplexCounter.CREATED_VERTICES,
            )
        )
    ].set(
        jnp.stack(
            (
                requested,
                requested,
                closure.bisections,
                closure.iterations,
                closure.created,
            )
        ).astype(jnp.int64)
    )
    result = closure.work._replace(
        clocks=closure.work.clocks.at[0].add(closure.iterations),
        counters=closure.work.counters + increments,
    )
    return _finished(
        work,
        result,
        closure.status,
        layout.dimension,
        requested=requested,
        accepted=requested,
        rejected=jnp.zeros_like(mask),
        operations=closure.bisections,
        iterations=closure.iterations,
        vertices=closure.created,
        axis=axis,
    )


def _refine_parts(
    layout: AdaptiveSimplexLayout,
    parts: AdaptiveSimplexParts,
    states: AdaptiveSimplexState,
    marks: Array,
    /,
) -> AdaptiveSimplexUpdate:
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def local(state_block: AdaptiveSimplexState, marks_block: Array):
        state = jax.tree_util.tree_map(lambda value: value[0], state_block)
        work = _work(state)
        mask = marks_block[0] & work.cell_active
        requested = jax.lax.psum(jnp.sum(mask, dtype=jnp.int32), axis)
        # The refusal is collective: every part sees the union of recorded flags.
        update = jax.lax.cond(
            _terminal(work, axis) != 0,
            lambda: _refused(
                state, work, requested, jnp.zeros_like(mask), layout.dimension, axis
            ),
            lambda: _refined_part(layout, work, mask, requested, axis),
        )
        return jax.tree_util.tree_map(lambda value: value[None], update)

    mapped = jax.shard_map(
        local, mesh=parts.mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
    )
    return mapped(states, marks)


_compiled_refine_parts = eqx.filter_jit(_refine_parts)


def refine_adaptive_simplex_parts(
    layout: AdaptiveSimplexLayout,
    parts: AdaptiveSimplexParts,
    states: AdaptiveSimplexState,
    marks: ArrayLike,
    /,
) -> AdaptiveSimplexUpdate:
    """Refine a part-sharded epoch: owned cells per part, closure across parts.

    ``states`` stacks one local state per part on a leading axis (each part
    holds its owned cells and their vertices). Every closure round gathers the
    candidate refinement edges and selected cell IDs of all parts, so new vertex
    and cell IDs are global ranks, identical to the single-part issue whatever
    the ownership; split edges on shared part boundaries select the neighbor
    part's cells until the global fixed point. A round that would exceed any
    part's capacity or split a protected edge fails for all parts (arrays
    rolled back, terminal flags recorded on every part); there is no per-mark
    protected admissibility on parts. A terminal flag recorded on any part
    refuses every later call on all parts.
    """

    if not isinstance(layout, AdaptiveSimplexLayout):
        raise TypeError("layout must be AdaptiveSimplexLayout.")
    if not isinstance(parts, AdaptiveSimplexParts):
        raise TypeError("parts must be AdaptiveSimplexParts.")
    if not isinstance(states, AdaptiveSimplexState):
        raise TypeError("states must be a stacked AdaptiveSimplexState.")
    count = parts.part_count
    if states.cursors.shape != (count, 4) or states.mesh.cells.shape[:2] != (
        count,
        layout.cell_capacity,
    ):
        raise ValueError("states must stack one local state per part of this layout.")
    mask = jnp.asarray(marks)
    if mask.shape != (count, layout.cell_capacity) or mask.dtype != jnp.bool_:
        raise TypeError(
            f"marks must be a bool ({count}, {layout.cell_capacity}) per-part mask."
        )
    return _compiled_refine_parts(layout, parts, states, mask)


def adaptive_simplex_state(
    layout: AdaptiveSimplexLayout,
    /,
    *,
    coordinates: np.ndarray,
    vertex_ids: np.ndarray,
    vertex_active: np.ndarray,
    vertex_parents: np.ndarray,
    vertex_protected: np.ndarray,
    cells: np.ndarray,
    tuples: np.ndarray,
    tags: np.ndarray,
    blocks: np.ndarray,
    generations: np.ndarray,
    parents: np.ndarray,
    children: np.ndarray,
    bisection_vertices: np.ndarray,
    cell_ids: np.ndarray,
    cell_active: np.ndarray,
    cell_classes: np.ndarray,
    facet_classes: np.ndarray,
    protected_edges: np.ndarray,
    next_vertex_id: int,
    next_cell_id: int,
) -> AdaptiveSimplexState:
    """Pad one host-prepared epoch into its capacity bucket.

    Vertex and cell rows must already be in ascending global-ID order (slot
    order is ID order); ``protected_edges`` are vertex-slot pairs.
    """

    if not isinstance(layout, AdaptiveSimplexLayout):
        raise TypeError("layout must be AdaptiveSimplexLayout.")
    vertex_count, cell_count = vertex_ids.shape[0], cell_ids.shape[0]
    capacity_v, capacity_c = layout.vertex_capacity, layout.cell_capacity
    width = layout.dimension + 1
    if vertex_count > capacity_v or cell_count > capacity_c:
        raise ValueError("The prepared epoch exceeds its capacity bucket.")
    if np.any(np.diff(vertex_ids) <= 0) or np.any(np.diff(cell_ids) <= 0):
        raise ValueError("Slots must follow strictly increasing global IDs.")
    if protected_edges.shape[0] > layout.protected_edge_capacity:
        raise ValueError("The protected edges exceed their capacity bucket.")
    dtype = np.dtype(layout.coordinate_dtype)

    def padded(values, capacity, fill, dtype_):
        array = np.asarray(values, dtype=dtype_)
        result = np.full((capacity,) + array.shape[1:], fill, dtype=dtype_)
        result[: array.shape[0]] = array
        return result

    codes = np.sort(
        np.minimum(protected_edges[:, 0], protected_edges[:, 1]).astype(np.int64)
        * capacity_v
        + np.maximum(protected_edges[:, 0], protected_edges[:, 1]).astype(np.int64)
    )
    work = _Work(
        jnp.asarray(padded(coordinates, capacity_v, 0.0, dtype)),
        jnp.asarray(padded(vertex_ids, capacity_v, -1, np.int64)),
        jnp.asarray(padded(vertex_active, capacity_v, False, np.bool_)),
        jnp.asarray(padded(cells, capacity_c, 0, np.int32)),
        jnp.asarray(padded(cell_ids, capacity_c, -1, np.int64)),
        jnp.asarray(padded(cell_active, capacity_c, False, np.bool_)),
        jnp.asarray(padded(tuples, capacity_c, 0, np.int32)),
        jnp.asarray(padded(tags, capacity_c, layout.dimension, np.int32)),
        jnp.asarray(padded(blocks, capacity_c, 0, np.int32)),
        jnp.asarray(padded(generations, capacity_c, 0, np.int32)),
        jnp.asarray(padded(parents, capacity_c, -1, np.int32)),
        jnp.asarray(padded(children, capacity_c, -1, np.int32)),
        jnp.asarray(padded(bisection_vertices, capacity_c, -1, np.int32)),
        jnp.zeros((capacity_c,), dtype=jnp.bool_),
        jnp.asarray(padded(cell_classes, capacity_c, 0, np.int32)),
        jnp.asarray(padded(facet_classes, capacity_c, 0, np.int32)),
        jnp.asarray(padded(vertex_parents, capacity_v, -1, np.int32)),
        jnp.zeros((capacity_v,), dtype=jnp.int32),
        jnp.full((capacity_v,), -1, dtype=jnp.int32),
        jnp.asarray(padded(vertex_protected, capacity_v, False, np.bool_)),
        jnp.asarray(
            padded(codes, layout.protected_edge_capacity, _CODE_SENTINEL, np.int64)
        ),
        jnp.zeros((capacity_c,), dtype=jnp.bool_),
        jnp.zeros((capacity_c,), dtype=jnp.bool_),
        jnp.asarray(
            np.asarray(
                (vertex_count, cell_count, int(next_vertex_id), int(next_cell_id)),
                dtype=np.int64,
            )
        ),
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.zeros((_COUNTERS,), dtype=jnp.int64),
    )
    if work.cells.shape[1] != width:
        raise ValueError("cells do not match the layout dimension.")
    return _state(work)


__all__ = [
    "AdaptiveSimplexCounter",
    "AdaptiveSimplexLayout",
    "AdaptiveSimplexParts",
    "AdaptiveSimplexPolicy",
    "AdaptiveSimplexReport",
    "AdaptiveSimplexState",
    "AdaptiveSimplexStatus",
    "AdaptiveSimplexUpdate",
    "MaskedSimplexMesh",
    "adaptive_simplex_bucket",
    "adaptive_simplex_state",
    "coarsen_adaptive_simplex",
    "masked_simplex_facet_neighbors",
    "masked_simplex_signature",
    "maubach_bisection_tables",
    "refine_adaptive_simplex",
    "refine_adaptive_simplex_parts",
]
