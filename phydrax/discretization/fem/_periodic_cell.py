#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Continuous P1 coordinates on the vertex orbits of a periodic tetrahedral quotient.

A periodic mesh is a lifted, nonperiodic fundamental-cover carrier plus the
canonical `PeriodicMeshTopology` tables: every lifted vertex names its orbit
representative and the integer lattice shift that maps the representative onto
it. One continuous P1 coordinate per vertex orbit is therefore a field that is
periodic by construction: the lifted copies of an orbit share the coordinate
exactly, with no constraint equation and no residual.

Cell geometry is rebuilt from the representative coordinates and the lattice
vectors, ``x = x_rep + shift @ vectors``. That keeps a moved quotient mesh
periodic exactly and differentiable in both the representative positions and
the lattice.

A periodic fluctuation is defined only up to the constant fields of each
connected quotient component. `PeriodicSimplexOperatorPattern` removes that
null space explicitly: every coordinate of the lowest orbit index of each
component is pinned to zero (the *gauge DOF*). Gradients, fluxes and every
average built from them do not depend on the gauge. Physics owners that admit
further null modes (rigid rotations of an elastic component winding along fewer
than two lattice directions) check ``component_winding_ranks`` and refuse.
"""

from __future__ import annotations

from fractions import Fraction
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import exact_orient3d
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    OperatorProperties,
    prepare_sparse_factorization,
    PreparedSparseFactorization,
    refresh_sparse_factorization,
    SparseFactorizationPlan,
    SparseFactorizationPolicy,
)
from ...sparse import EdgeRelation, SparseLinearMap
from .._cell_mesh import CellBlock, CellMesh
from .._periodic_cell import PeriodicCell
from .._reference_cell import reference_cell_topology


_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_FACTORIZATION = SparseFactorizationPolicy("cholesky", ordering="reverse-cuthill-mckee")
_PROPERTIES = OperatorProperties(
    self_adjoint=True,
    positive_definite=True,
    positive_semidefinite=True,
    evidence={
        "self_adjoint": "construction",
        "positive_definite": "construction",
        "positive_semidefinite": "construction",
    },
)


def _integer_rank(rows: np.ndarray, /) -> int:
    """Exact rank of a small integer matrix by fraction-valued elimination."""
    matrix = [[Fraction(int(value)) for value in row] for row in rows.tolist()]
    rank = 0
    columns = rows.shape[1] if rows.ndim == 2 else 0
    for column in range(columns):
        pivot = next(
            (index for index in range(rank, len(matrix)) if matrix[index][column] != 0),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        lead = matrix[rank][column]
        for index in range(len(matrix)):
            if index != rank and matrix[index][column] != 0:
                factor = matrix[index][column] / lead
                matrix[index] = [
                    value - factor * base
                    for value, base in zip(matrix[index], matrix[rank], strict=True)
                ]
        rank += 1
    return rank


def _components_and_windings(
    cell_dofs: np.ndarray, cell_shifts: np.ndarray, dof_count: int, /
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Connected quotient components and the rank of their realized windings.

    A lifted edge ``a -> b`` is the quotient edge ``rep(a) -> rep(b)`` carrying
    the lattice translation ``shift(b) - shift(a)``. A breadth-first spanning
    tree assigns integer potentials; every edge then closes a cycle whose winding
    is ``p(rep(a)) + t - p(rep(b))``. The exact rank of the windings of one
    component is the number of independent lattice directions it wraps.
    """
    first = np.concatenate([cell_dofs[:, a] for a, _ in _EDGES])
    second = np.concatenate([cell_dofs[:, b] for _, b in _EDGES])
    translation = np.concatenate(
        [cell_shifts[:, b] - cell_shifts[:, a] for a, b in _EDGES]
    ).astype(np.int64)
    sources = np.concatenate((first, second))
    targets = np.concatenate((second, first))
    steps = np.concatenate((translation, -translation))
    order = np.argsort(sources, kind="stable")
    sources, targets, steps = sources[order], targets[order], steps[order]
    offsets = np.searchsorted(sources, np.arange(dof_count + 1))
    component = np.full((dof_count,), -1, dtype=np.int64)
    potential = np.zeros((dof_count, 3), dtype=np.int64)
    count = 0
    for root in range(dof_count):
        if component[root] >= 0:
            continue
        component[root] = count
        frontier = np.asarray((root,), dtype=np.int64)
        while frontier.size:
            starts = offsets[frontier]
            stops = offsets[frontier + 1]
            lengths = stops - starts
            routes = np.repeat(
                starts - np.cumsum(lengths) + lengths, lengths
            ) + np.arange(int(np.sum(lengths)))
            parents = np.repeat(frontier, lengths)
            reached = targets[routes]
            fresh = component[reached] < 0
            reached, parents, routes = reached[fresh], parents[fresh], routes[fresh]
            reached, first_hit = np.unique(reached, return_index=True)
            component[reached] = count
            potential[reached] = potential[parents[first_hit]] + steps[routes[first_hit]]
            frontier = reached
        count += 1
    windings = potential[sources] + steps - potential[targets]
    ranks = []
    for index in range(count):
        selected = windings[(component[sources] == index) & np.any(windings != 0, axis=1)]
        unique = (
            np.unique(selected, axis=0) if selected.size else selected.reshape((0, 3))
        )
        ranks.append(_integer_rank(unique))
    return component.astype(np.int32), tuple(ranks)


@final
class PeriodicSimplexSpace(StrictModule, NonTrainableState):
    """Continuous P1 coordinates on the vertex orbits of a periodic tetrahedral mesh.

    ``mesh`` must carry a `PeriodicMeshTopology` over a fully periodic
    three-dimensional `PeriodicCell` and only nondegenerate tetrahedral blocks;
    cells are stored positively oriented by the exact ``orient3d`` sign of their
    lifted corners (mixed-orientation input is reordered, not refused). Quotient
    DOF ``i`` is the orbit whose
    representative lifted row is ``representative_rows[i]`` (ascending).
    ``cell_dofs``/``cell_shifts`` give, per lifted cell vertex, its orbit and
    lattice shift; ``cell_blocks`` the block index of each cell in
    ``block_names`` order. ``dof_components`` labels the connected quotient
    components, ``component_winding_ranks`` the number of independent lattice
    directions each wraps, and ``gauge_dofs`` the lowest DOF of each component.
    """

    mesh: CellMesh
    cell: PeriodicCell
    cell_dofs: Array
    cell_shifts: Array
    cell_blocks: Array
    representative_rows: Array
    dof_components: Array
    gauge_dofs: Array
    component_winding_ranks: tuple[int, ...] = eqx.field(static=True)
    block_names: tuple[str, ...] = eqx.field(static=True)
    dof_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    def __init__(self, mesh: CellMesh, /) -> None:
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be a CellMesh.")
        topology = mesh.periodic_topology
        if topology is None:
            raise ValueError(
                "A periodic simplex space needs a mesh with periodic topology."
            )
        cell = topology.cell
        if (
            not isinstance(cell, PeriodicCell)
            or cell.ambient_dimension != 3
            or not cell.fully_periodic
            or mesh.topological_dimension != 3
        ):
            raise ValueError(
                "A periodic simplex space needs a fully periodic three-dimensional "
                "PeriodicCell and a volume mesh."
            )
        if not mesh.blocks or any(
            not isinstance(block, CellBlock)
            or reference_cell_topology(block.cell_kind).family != "tetrahedron"
            for block in mesh.blocks
        ):
            raise ValueError("A periodic simplex space needs tetrahedral blocks only.")
        representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
        shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
        vertices = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
        )
        blocks = np.concatenate(
            [
                np.full((np.asarray(block.vertices).shape[0],), index, dtype=np.int32)
                for index, block in enumerate(mesh.blocks)
            ]
        )
        # Only orbits referenced by a cell carry a coordinate.
        rows = np.unique(representatives[vertices])
        orbit = np.searchsorted(rows, representatives)
        corners = np.asarray(mesh.coordinates, dtype=np.float64)[vertices]
        orientation = np.asarray(
            exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3])
        )
        if np.any(orientation == 0):
            raise ValueError("A periodic simplex space refuses degenerate tetrahedra.")
        # P1 coordinates do not depend on vertex order: store every cell positively
        # oriented (exact sign) by swapping its local vertices 1 and 2.
        vertices = np.where(
            (orientation < 0)[:, None], vertices[:, (0, 2, 1, 3)], vertices
        )
        cell_dofs = orbit[vertices]
        cell_shifts = shifts[vertices]
        components, ranks = _components_and_windings(cell_dofs, cell_shifts, rows.size)
        gauge = np.asarray(
            [int(np.argmax(components == index)) for index in range(len(ranks))],
            dtype=np.int32,
        )
        self.mesh = mesh
        self.cell = cell
        self.cell_dofs = jnp.asarray(cell_dofs, dtype=jnp.int32)
        self.cell_shifts = jnp.asarray(cell_shifts, dtype=jnp.int32)
        self.cell_blocks = jnp.asarray(blocks, dtype=jnp.int32)
        self.representative_rows = jnp.asarray(rows, dtype=jnp.int32)
        self.dof_components = jnp.asarray(components, dtype=jnp.int32)
        self.gauge_dofs = jnp.asarray(gauge, dtype=jnp.int32)
        self.component_winding_ranks = ranks
        self.block_names = tuple(block.name for block in mesh.blocks)
        self.dof_count = int(rows.size)
        self.cell_count = int(vertices.shape[0])
        self.space_id = canonical_fingerprint(
            {
                "kind": "periodic-simplex-p1-space",
                "lifted_topology": topology.lifted_topology_id,
                "periodic_topology": topology.periodic_topology_id,
            }
        )

    @property
    def lattice_vectors(self) -> Array:
        """Row lattice vectors of the mesh's periodic cell."""
        return jnp.asarray(self.cell.vectors, dtype=jnp.float64)

    @property
    def representative_coordinates(self) -> Array:
        """Cartesian coordinates of every orbit representative, in DOF order."""
        return jnp.asarray(self.mesh.coordinates, dtype=jnp.float64)[
            self.representative_rows
        ]

    def block_mask(self, names: tuple[str, ...], /) -> Array:
        """Boolean cell mask of the named blocks."""
        unknown = set(names) - set(self.block_names)
        if unknown:
            raise ValueError(f"Unknown block names {sorted(unknown)}.")
        selected = np.asarray([name in names for name in self.block_names], dtype=bool)
        return jnp.asarray(selected)[self.cell_blocks]

    def cell_coordinates(
        self,
        representative_coordinates: ArrayLike | None = None,
        lattice_vectors: ArrayLike | None = None,
        /,
    ) -> Array:
        """Lifted vertex coordinates ``(cells, 4, 3)``: ``x_rep + shift @ vectors``."""
        points = (
            self.representative_coordinates
            if representative_coordinates is None
            else jnp.asarray(representative_coordinates, dtype=jnp.float64)
        )
        vectors = (
            self.lattice_vectors
            if lattice_vectors is None
            else jnp.asarray(lattice_vectors, dtype=jnp.float64)
        )
        if points.shape != (self.dof_count, 3) or vectors.shape != (3, 3):
            raise ValueError(
                "Representative coordinates must be (dofs, 3) and lattice vectors (3, 3)."
            )
        return points[self.cell_dofs] + self.cell_shifts.astype(jnp.float64) @ vectors

    def cell_measure(self, lattice_vectors: ArrayLike | None = None, /) -> Array:
        """Volume of the periodic cell, ``|det(vectors)|``."""
        vectors = (
            self.lattice_vectors
            if lattice_vectors is None
            else jnp.asarray(lattice_vectors, dtype=jnp.float64)
        )
        return jnp.abs(jnp.linalg.det(vectors))


def simplex_gradients(corners: Array, /) -> tuple[Array, Array]:
    """Signed volumes ``(cells,)`` and barycentric gradients ``(cells, 4, 3)``."""
    edges = jnp.swapaxes(corners[:, 1:] - corners[:, :1], -1, -2)
    determinant = jnp.linalg.det(edges)
    inverse = jnp.linalg.inv(edges)
    gradients = jnp.concatenate(
        (-jnp.sum(inverse, axis=1, keepdims=True), inverse), axis=1
    )
    return determinant / 6.0, gradients


@final
class PeriodicSimplexOperatorPattern(StrictModule, NonTrainableState):
    """Fixed quotient sparsity of a ``components``-vector P1 field with its gauge removed.

    Coordinate ``dof * components + component`` is free unless ``dof`` is a
    gauge DOF of `PeriodicSimplexSpace`; every pinned coordinate is zero. Local
    element coordinates are ordered ``node * components + component``.
    ``plan`` is the symbolic supernodal Cholesky analysis of the free pattern,
    computed once and refreshed for every numeric value set.
    """

    space: PeriodicSimplexSpace
    free_index: Array
    entry_positions: Array
    entry_kept: Array
    relation: EdgeRelation
    plan: SparseFactorizationPlan
    components: int = eqx.field(static=True)
    free_count: int = eqx.field(static=True)
    entry_count: int = eqx.field(static=True)
    pattern_id: str = eqx.field(static=True)

    def __init__(self, space: PeriodicSimplexSpace, /, *, components: int) -> None:
        if not isinstance(space, PeriodicSimplexSpace):
            raise TypeError("space must be a PeriodicSimplexSpace.")
        width = int(components)
        if width < 1:
            raise ValueError("components must be positive.")
        full = space.dof_count * width
        pinned = np.zeros((full,), dtype=bool)
        gauge = np.asarray(space.gauge_dofs, dtype=np.int64)
        for component in range(width):
            pinned[gauge * width + component] = True
        free_index = np.full((full,), -1, dtype=np.int64)
        free_index[~pinned] = np.arange(int(np.sum(~pinned)))
        free_count = int(np.sum(~pinned))
        if free_count == 0:
            raise ValueError("A periodic operator pattern needs at least one free DOF.")
        dofs = np.asarray(space.cell_dofs, dtype=np.int64)
        local = (dofs[:, :, None] * width + np.arange(width)[None, None, :]).reshape(
            (dofs.shape[0], 4 * width)
        )
        rows = free_index[np.repeat(local, 4 * width, axis=1)]
        columns = free_index[np.tile(local, (1, 4 * width))]
        kept = (rows >= 0) & (columns >= 0)
        keys = rows * free_count + columns
        unique = np.unique(keys[kept])
        positions = np.where(kept, np.searchsorted(unique, keys), 0)
        targets = (unique // free_count).astype(np.int32)
        sources = (unique % free_count).astype(np.int32)
        relation = EdgeRelation(
            sources, targets, source_size=free_count, target_size=free_count
        )
        pattern_id = canonical_fingerprint(
            {
                "kind": "periodic-simplex-operator-pattern",
                "space": space.space_id,
                "components": width,
            }
        )
        template = SparseLinearMap(
            relation,
            jnp.ones((unique.size,), dtype=jnp.float64),
            properties=_PROPERTIES,
            operator_id=pattern_id,
        )
        self.space = space
        self.free_index = jnp.asarray(free_index, dtype=jnp.int32)
        self.entry_positions = jnp.asarray(positions, dtype=jnp.int32)
        self.entry_kept = jnp.asarray(kept)
        self.relation = relation
        self.plan = prepare_sparse_factorization(template, _FACTORIZATION)
        self.components = width
        self.free_count = free_count
        self.entry_count = int(unique.size)
        self.pattern_id = pattern_id

    def operator(self, element_matrices: ArrayLike, /) -> SparseLinearMap:
        """Assemble ``(cells, 4c, 4c)`` element matrices on the free coordinates."""
        matrices = jnp.asarray(element_matrices, dtype=jnp.float64)
        width = 4 * self.components
        if matrices.shape != (self.space.cell_count, width, width):
            raise ValueError(
                f"Element matrices must have shape ({self.space.cell_count}, {width}, {width})."
            )
        values = jax.ops.segment_sum(
            jnp.where(
                self.entry_kept, matrices.reshape((matrices.shape[0], -1)), 0.0
            ).reshape(-1),
            self.entry_positions.reshape(-1),
            num_segments=self.entry_count,
        )
        return SparseLinearMap(
            self.relation, values, properties=_PROPERTIES, operator_id=self.pattern_id
        )

    def factor(self, operator: SparseLinearMap, /) -> PreparedSparseFactorization:
        """Numeric Cholesky refresh of an assembled operator on the fixed plan."""
        return refresh_sparse_factorization(self.plan, operator)

    def load(self, element_vectors: ArrayLike, /) -> Array:
        """Assemble ``(cases, cells, 4c)`` element vectors into ``(free, cases)``."""
        vectors = jnp.asarray(element_vectors, dtype=jnp.float64)
        width = 4 * self.components
        if vectors.ndim != 3 or vectors.shape[1:] != (self.space.cell_count, width):
            raise ValueError(
                f"Element vectors must have shape (cases, {self.space.cell_count}, {width})."
            )
        dofs = self.space.cell_dofs
        local = (
            dofs[:, :, None] * self.components
            + jnp.arange(self.components)[None, None, :]
        ).reshape((dofs.shape[0], width))
        targets = self.free_index[local]
        kept = targets >= 0
        values = jnp.where(kept[None], vectors, 0.0)
        assembled = jax.vmap(
            lambda case: jax.ops.segment_sum(
                case.reshape(-1),
                jnp.where(kept, targets, 0).reshape(-1),
                num_segments=self.free_count,
            )
        )(values)
        return assembled.T

    def expand(self, free_values: ArrayLike, /) -> Array:
        """Free coordinates ``(free, cases)`` to ``(cases, dofs, components)``."""
        values = jnp.asarray(free_values, dtype=jnp.float64)
        if values.ndim != 2 or values.shape[0] != self.free_count:
            raise ValueError(f"Free values must have shape ({self.free_count}, cases).")
        padded = jnp.concatenate(
            (values, jnp.zeros((1, values.shape[1]), dtype=values.dtype)), axis=0
        )
        index = jnp.where(self.free_index >= 0, self.free_index, self.free_count)
        return padded[index].T.reshape(
            (values.shape[1], self.space.dof_count, self.components)
        )

    def gather(self, field: ArrayLike, /) -> Array:
        """Quotient field ``(cases, dofs, components)`` to ``(cases, cells, 4c)``."""
        values = jnp.asarray(field, dtype=jnp.float64)
        if values.ndim != 3 or values.shape[1:] != (
            self.space.dof_count,
            self.components,
        ):
            raise ValueError(
                f"Fields must have shape (cases, {self.space.dof_count}, {self.components})."
            )
        return values[:, self.space.cell_dofs].reshape(
            (values.shape[0], self.space.cell_count, 4 * self.components)
        )


__all__ = [
    "PeriodicSimplexOperatorPattern",
    "PeriodicSimplexSpace",
    "simplex_gradients",
]
