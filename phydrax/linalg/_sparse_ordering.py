#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host fill-reducing orderings for square sparse patterns.

This module owns symmetric ordering preparation: pattern validation and
identity, validated permutations and inverses, ordering identities, metered
ordering work, and deterministic tie breaking. Every method orders the graph of
``A + Aᵀ`` without its diagonal; the permutation is applied symmetrically.

- ``natural`` is the identity and performs no graph work.
- ``reverse-cuthill-mckee`` is SciPy's bandwidth-reducing ordering.
- ``approximate-minimum-degree`` eliminates on a quotient graph with
  approximate external degrees, element absorption, and indistinguishable
  supervariable detection (Amestoy, Davis and Duff, SIAM J. Matrix Anal. Appl.
  17, 1996). The same engine orders the columns of ``AᵀA`` from the rows of
  ``A`` for pivoted sparse LU (the COLAMD formulation).
- ``nested-dissection`` recursively removes vertex separators and orders each
  part before its separator (George, SIAM J. Numer. Anal. 10, 1973). A part is
  split along a BFS level structure rooted at a pseudo-peripheral vertex
  (George and Liu, ACM TOMS 5, 1979) or, when coordinates are supplied, at the
  median of its widest coordinate axis. The separator is a minimum vertex
  cover of the cut edges (König's theorem on a maximum bipartite matching).
  Parts at or below the leaf capacity are ordered by approximate minimum
  degree; a separator above the separator capacity refuses.

Ties always break by the smallest original index, so an ordering is a pure
function of the pattern, the policy, and the coordinates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from heapq import heapify, heappop, heappush
from math import isqrt, sqrt
from numbers import Integral
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import numpy as np

from .._strict import StrictModule
from ..typing import (
    as_host_array,
    ConvertibleToArray,
    Dim,
    HostFloat64,
    HostInt64,
    parse,
    Scope,
    Size,
)
from ._properties import LinearCapabilityError
from ._sparse_contract import AbstractSparseLinearOperator, SparseStorage


SparseOrdering: TypeAlias = Literal[
    "natural",
    "reverse-cuthill-mckee",
    "approximate-minimum-degree",
    "nested-dissection",
]
NestedDissectionRoute: TypeAlias = Literal["graph", "geometric"]

# Dense rows and columns follow the COLAMD/AMD default threshold ``10 √n``.
_DENSE_FACTOR = 10.0
_DENSE_MINIMUM = 16
_DEFAULT_LEAF_CAPACITY = 64
_DEFAULT_SEPARATOR_CAPACITY = 2048


class _OrderedNodeDim(Dim):
    """Rows and columns of one symmetrically ordered square pattern."""


class _SeparatorDim(Dim):
    """Vertex separators removed by one nested dissection, in discovery order."""


class _LeafDim(Dim):
    """Parts ordered by minimum degree inside one nested dissection."""


class _CoordinateAxisDim(Dim):
    """Coordinate axes of the vertices of a geometrically dissected pattern."""


def _host_capacity(value: int | None, name: str, /) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a host integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


@final
class SparseOrderingPolicy(StrictModule):
    """Ordering method, nested-dissection capacities, and ordering work cap.

    ``leaf_capacity`` and ``separator_capacity`` exist only for
    ``nested-dissection`` (defaults 64 and 2048); other methods refuse them.
    Parts with at most ``leaf_capacity`` vertices are ordered by approximate
    minimum degree. A separator with more than ``separator_capacity`` vertices
    refuses: its clique alone would add ``s(s+1)/2`` factor entries.
    ``max_ordering_work`` bounds standalone preparation; inside sparse
    factorization the work is charged to ``max_symbolic_work`` instead.
    """

    method: SparseOrdering = eqx.field(static=True)
    leaf_capacity: int | None = eqx.field(static=True)
    separator_capacity: int | None = eqx.field(static=True)
    max_ordering_work: int = eqx.field(static=True)

    def __init__(
        self,
        method: SparseOrdering,
        /,
        *,
        leaf_capacity: int | None = None,
        separator_capacity: int | None = None,
        max_ordering_work: int = 512_000_000,
    ) -> None:
        method = parse(method, SparseOrdering, "method")
        leaf = _host_capacity(leaf_capacity, "leaf_capacity")
        separator = _host_capacity(separator_capacity, "separator_capacity")
        work = _host_capacity(max_ordering_work, "max_ordering_work")
        if work is None:
            raise TypeError("max_ordering_work must be a host integer.")
        if method == "nested-dissection":
            leaf = _DEFAULT_LEAF_CAPACITY if leaf is None else leaf
            separator = _DEFAULT_SEPARATOR_CAPACITY if separator is None else separator
        elif leaf is not None or separator is not None:
            raise ValueError(
                "leaf_capacity and separator_capacity apply only to nested-dissection."
            )
        self.method = method
        self.leaf_capacity = leaf
        self.separator_capacity = separator
        self.max_ordering_work = work


@final
class NestedDissectionEvidence(StrictModule):
    """Separators and leaves of one nested dissection.

    ``separator_sizes`` lists separators in top-down discovery order and
    ``leaf_sizes`` the minimum-degree parts in elimination order. A leaf larger
    than ``leaf_capacity`` is a connected part without a separator leaving
    both sides nonempty (for example a clique); it is still ordered by
    approximate minimum degree and remains visible here.
    """

    __strict_contract__ = True
    separator_sizes: HostInt64[_SeparatorDim]
    leaf_sizes: HostInt64[_LeafDim]
    route: NestedDissectionRoute = eqx.field(static=True)
    leaf_capacity: int = eqx.field(static=True)
    separator_capacity: int = eqx.field(static=True)


def _validated_inverse(permutation: np.ndarray, /) -> np.ndarray:
    size = permutation.size
    if np.any(permutation < 0) or np.any(permutation >= size):
        raise ValueError("Sparse ordering permutation entries must lie in [0, n).")
    inverse = np.full(size, -1, dtype=np.int64)
    inverse[permutation] = np.arange(size, dtype=np.int64)
    if np.any(inverse < 0):
        raise ValueError("Sparse ordering permutation must be a bijection of [0, n).")
    return inverse


@final
class PreparedSparseOrdering(StrictModule):
    """Validated symmetric elimination order of one square sparse pattern.

    ``permutation[k]`` is the original row and column eliminated ``k``-th, so a
    consumer factors ``A[permutation][:, permutation]``;
    ``inverse_permutation[i]`` is the elimination position of original index
    ``i``. ``pattern_id`` identifies the ordered CSR pattern, ``work`` is the
    metered ordering work, and ``ordering_id`` identifies the method, route,
    coordinates, and permutation.
    """

    __strict_contract__ = True
    permutation: HostInt64[_OrderedNodeDim]
    inverse_permutation: HostInt64[_OrderedNodeDim]
    dissection: NestedDissectionEvidence | None
    policy: SparseOrderingPolicy
    size: Size[_OrderedNodeDim] = eqx.field(static=True)
    pattern_id: str = eqx.field(static=True)
    coordinates_id: str | None = eqx.field(static=True)
    work: int = eqx.field(static=True)
    ordering_id: str = eqx.field(static=True)

    def __init__(
        self,
        permutation: ConvertibleToArray,
        /,
        *,
        policy: SparseOrderingPolicy,
        pattern_id: str,
        work: int,
        dissection: NestedDissectionEvidence | None = None,
        coordinates_id: str | None = None,
    ) -> None:
        if not isinstance(policy, SparseOrderingPolicy):
            raise TypeError("policy must be a SparseOrderingPolicy.")
        if isinstance(work, bool) or not isinstance(work, Integral) or work < 0:
            raise ValueError("work must be a non-negative host integer.")
        order = as_host_array(
            permutation, HostInt64[_OrderedNodeDim], "permutation", casting="safe"
        )
        dissected = policy.method == "nested-dissection"
        if dissected != (dissection is not None):
            raise ValueError(
                "Nested-dissection evidence is required exactly for nested-dissection."
            )
        geometric = dissection is not None and dissection.route == "geometric"
        if geometric != (coordinates_id is not None):
            raise ValueError(
                "A coordinates identity is required exactly for geometric dissection."
            )
        inverse = _validated_inverse(order)
        payload = b"|".join(
            (
                pattern_id.encode(),
                policy.method.encode(),
                ("" if dissection is None else dissection.route).encode(),
                str(coordinates_id).encode(),
                order.tobytes(),
            )
        )
        self.permutation = order
        self.inverse_permutation = inverse
        self.dissection = dissection
        self.policy = policy
        self.size = order.size
        self.pattern_id = pattern_id
        self.coordinates_id = coordinates_id
        self.work = int(work)
        self.ordering_id = sha256(payload).hexdigest()

    @property
    def method(self) -> SparseOrdering:
        return self.policy.method


@dataclass
class _WorkMeter:
    """Running ordering work, refused above ``limit`` or forwarded to ``forward``.

    ``forward`` charges an enclosing symbolic budget first, so a factorization
    reports its own unified ``symbolic_work`` refusal.
    """

    limit: int | None
    forward: Callable[[int], None] | None = None
    total: int = 0

    def add(self, count: int, /) -> None:
        if self.forward is not None:
            self.forward(count)
        projected = self.total + count
        if self.limit is not None and projected > self.limit:
            raise LinearCapabilityError(
                "Sparse ordering refused before completion: ordering_work requires "
                f"{projected}, exceeding limit {self.limit}."
            )
        self.total = projected


def _validated_pattern(
    operator: AbstractSparseLinearOperator,
    /,
) -> tuple[SparseStorage, np.ndarray, np.ndarray]:
    """Canonical sorted square CSR pattern of a native sparse operator."""
    if not isinstance(operator, AbstractSparseLinearOperator):
        raise TypeError("operator must be an AbstractSparseLinearOperator.")
    storage = operator.sparse_storage()
    if storage.shape[0] != storage.shape[1]:
        raise ValueError("Sparse factorization requires a square operator.")
    indices = np.asarray(storage.indices, dtype=np.int64)
    indptr = np.asarray(storage.indptr, dtype=np.int64)
    if indptr[0] != 0 or indptr[-1] != indices.size:
        raise ValueError("CSR indptr endpoints are inconsistent with its indices.")
    if np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("CSR indptr must be nondecreasing.")
    if np.any(indices < 0) or np.any(indices >= storage.shape[1]):
        raise ValueError("CSR column index is out of range.")
    for row in range(storage.shape[0]):
        columns = indices[indptr[row] : indptr[row + 1]]
        if columns.size > 1 and np.any(columns[1:] <= columns[:-1]):
            raise ValueError("Sparse factorization requires canonical sorted CSR rows.")
    return storage, indices, indptr


def _pattern_identifier(
    shape: tuple[int, int], indices: np.ndarray, indptr: np.ndarray, /
) -> str:
    payload = b"|".join(
        (
            np.asarray(shape, dtype=np.int64).tobytes(),
            indices.tobytes(),
            indptr.tobytes(),
        )
    )
    return sha256(payload).hexdigest()


@dataclass
class _QuotientGraph:
    """Elements, principal variables, and supervariable weights of a graph."""

    element_vars: dict[int, set[int]]
    element_weight: dict[int, int]
    var_elements: list[set[int]]
    var_adjacent: list[set[int]]
    weight: list[int]
    members: list[list[int]]
    alive: list[bool]
    meter: _WorkMeter

    @classmethod
    def _empty(
        cls, size: int, postponed: list[int], meter: _WorkMeter, /
    ) -> _QuotientGraph:
        graph = cls(
            {},
            {},
            [set() for _ in range(size)],
            [set() for _ in range(size)],
            [1] * size,
            [[variable] for variable in range(size)],
            [True] * size,
            meter,
        )
        for variable in postponed:
            graph.alive[variable] = False
        return graph

    @classmethod
    def from_rows(
        cls,
        rows: list[list[int]],
        columns: int,
        postponed: list[int],
        meter: _WorkMeter,
        /,
    ) -> _QuotientGraph:
        """Each row of ``A`` is an initial element: a clique of ``AᵀA``."""
        graph = cls._empty(columns, postponed, meter)
        for element, row in enumerate(rows):
            meter.add(len(row))
            if len(row) < 2:
                continue
            graph.element_vars[element] = set(row)
            graph.element_weight[element] = len(graph.element_vars[element])
            for column in row:
                graph.var_elements[column].add(element)
        return graph

    @classmethod
    def from_adjacency(
        cls,
        adjacency: list[list[int]],
        postponed: list[int],
        meter: _WorkMeter,
        /,
    ) -> _QuotientGraph:
        """Variables of a symmetric graph with no initial elements."""
        graph = cls._empty(len(adjacency), postponed, meter)
        removed = set(postponed)
        for variable, neighbors in enumerate(adjacency):
            meter.add(len(neighbors))
            if variable not in removed:
                graph.var_adjacent[variable] = set(neighbors) - removed
        return graph

    def eliminate(self, pivot: int, new_element: int, /) -> set[int]:
        """Form ``L_p`` from ``p``'s elements and neighbors; prune covered edges."""
        absorbed = self.var_elements[pivot]
        pattern = set(self.var_adjacent[pivot])
        work = len(pattern)
        for element in absorbed:
            variables = self.element_vars.pop(element)
            work += len(variables)
            pattern |= variables
            del self.element_weight[element]
        pattern.discard(pivot)
        self.alive[pivot] = False
        self.var_elements[pivot] = set()
        self.var_adjacent[pivot] = set()
        self.element_vars[new_element] = pattern
        for variable in pattern:
            work += len(self.var_elements[variable]) + len(self.var_adjacent[variable])
            elements = self.var_elements[variable] - absorbed
            elements.add(new_element)
            self.var_elements[variable] = elements
            adjacent = self.var_adjacent[variable] - pattern
            adjacent.discard(pivot)
            self.var_adjacent[variable] = adjacent
        self.meter.add(work)
        return pattern

    def external_weights(self, pattern: set[int], new_element: int, /) -> dict[int, int]:
        """``|L_e \\ L_p|`` for every element touching ``L_p``; empty ones are absorbed."""
        external: dict[int, int] = {}
        work = 0
        for variable in pattern:
            variable_weight = self.weight[variable]
            work += len(self.var_elements[variable])
            for element in self.var_elements[variable]:
                if element != new_element:
                    external[element] = (
                        external.get(element, self.element_weight[element])
                        - variable_weight
                    )
        for element, value in external.items():
            if value == 0:
                variables = self.element_vars.pop(element)
                work += len(variables)
                for variable in variables:
                    self.var_elements[variable].discard(element)
                del self.element_weight[element]
        self.meter.add(work)
        return external

    def merge_indistinguishable(self, pattern: set[int], /) -> None:
        """Merge variables of ``L_p`` with equal elements and neighbors."""
        buckets: dict[tuple[frozenset[int], frozenset[int]], list[int]] = {}
        work = 0
        for variable in sorted(pattern):
            work += len(self.var_elements[variable]) + len(self.var_adjacent[variable])
            key = (
                frozenset(self.var_elements[variable]),
                frozenset(self.var_adjacent[variable]),
            )
            buckets.setdefault(key, []).append(variable)
        for principal, *merged in buckets.values():
            for variable in merged:
                self.weight[principal] += self.weight[variable]
                self.members[principal].extend(self.members[variable])
                self.alive[variable] = False
                for element in self.var_elements[variable]:
                    self.element_vars[element].discard(variable)
                for neighbor in self.var_adjacent[variable]:
                    self.var_adjacent[neighbor].discard(variable)
                self.var_elements[variable] = set()
                self.var_adjacent[variable] = set()
        self.meter.add(work)

    def external_degree(
        self,
        variable: int,
        grown: int,
        new_element: int,
        external: dict[int, int],
        /,
    ) -> int:
        """``|A_i| + |L_p \\ i| + Σ_e |L_e \\ L_p|`` over the remaining elements."""
        self.meter.add(
            len(self.var_adjacent[variable]) + len(self.var_elements[variable])
        )
        return (
            sum(self.weight[neighbor] for neighbor in self.var_adjacent[variable])
            + grown
            + sum(
                external[element]
                for element in self.var_elements[variable]
                if element != new_element
            )
        )


def _minimum_degree_order(
    graph: _QuotientGraph,
    degree: list[int],
    postponed: list[int],
    element_offset: int,
    /,
) -> list[int]:
    """Eliminate principal variables by approximate degree; postponed ones last.

    Eliminating the principal variable ``p`` merges its elements and variable
    neighbors into the new element ``L_p`` (identifier ``element_offset + p``)
    and prunes edges it covers. Degrees are the AMD upper bounds ``min(d_i +
    |L_p \\ i|, |A_i| + |L_p \\ i| + Σ_e |L_e \\ L_p|, n_left − |i|)`` with
    weighted supervariables; elements inside ``L_p`` are absorbed and
    indistinguishable variables of ``L_p`` merge. Ties break by the smallest
    variable index.
    """
    size = len(degree)
    heap = [
        (degree[variable], variable) for variable in range(size) if graph.alive[variable]
    ]
    heapify(heap)
    remaining = size - len(postponed)
    order: list[int] = []
    while heap:
        current, pivot = heappop(heap)
        if not graph.alive[pivot] or current != degree[pivot]:
            continue
        order.extend(graph.members[pivot])
        remaining -= graph.weight[pivot]
        new_element = element_offset + pivot
        pattern = graph.eliminate(pivot, new_element)
        external = graph.external_weights(pattern, new_element)
        graph.merge_indistinguishable(pattern)
        pattern_weight = sum(graph.weight[variable] for variable in pattern)
        graph.element_weight[new_element] = pattern_weight
        for variable in pattern:
            grown = pattern_weight - graph.weight[variable]
            degree[variable] = max(
                0,
                min(
                    degree[variable] + grown,
                    graph.external_degree(variable, grown, new_element, external),
                    remaining - graph.weight[variable],
                ),
            )
            heappush(heap, (degree[variable], variable))
    order.extend(postponed)
    return order


def _dense_threshold(size: int, /) -> int:
    return max(_DENSE_MINIMUM, int(_DENSE_FACTOR * sqrt(size)))


def _initial_column_degrees(
    retained: list[list[int]], columns: int, meter: _WorkMeter, /
) -> list[int]:
    """Exact external degrees in the graph of ``AᵀA`` over the retained rows."""
    import scipy.sparse as sp

    lengths = [len(row) for row in retained]
    pattern = sp.csr_matrix(
        (
            np.ones(sum(lengths), dtype=np.int8),
            np.fromiter(
                (column for row in retained for column in row),
                dtype=np.int64,
                count=sum(lengths),
            ),
            np.concatenate(([0], np.cumsum(lengths, dtype=np.int64))),
        ),
        shape=(len(retained), columns),
    )
    gram = (pattern.T @ pattern).tocsr()
    meter.add(gram.nnz)
    counts = np.diff(gram.indptr)
    return (counts - (gram.diagonal() != 0)).tolist()


def _retained_rows(
    row_lists: list[list[int]], columns: int, /
) -> tuple[list[list[int]], list[int]]:
    """Drop rows and postpone columns denser than ``max(16, 10 √n)`` (COLAMD)."""
    dense = _dense_threshold(columns)
    retained = [row for row in row_lists if len(row) <= dense]
    column_counts = [0] * columns
    for row in retained:
        for column in row:
            column_counts[column] += 1
    postponed = [column for column in range(columns) if column_counts[column] > dense]
    postponed_set = set(postponed)
    return [
        [column for column in row if column not in postponed_set] for row in retained
    ], postponed


def _column_minimum_degree(
    row_lists: list[list[int]], columns: int, meter: _WorkMeter, /
) -> list[int]:
    """Approximate minimum-degree order of the columns of ``AᵀA`` (COLAMD).

    The rows of ``A`` are the initial elements; dense rows are ignored and dense
    columns ordered last.
    """
    retained, postponed = _retained_rows(row_lists, columns)
    graph = _QuotientGraph.from_rows(retained, columns, postponed, meter)
    degree = _initial_column_degrees(retained, columns, meter)
    return _minimum_degree_order(graph, degree, postponed, len(retained))


def _symmetric_minimum_degree(
    adjacency: list[list[int]], meter: _WorkMeter, /
) -> list[int]:
    """Approximate minimum-degree order of a symmetric graph (AMD).

    Vertices of degree above ``max(16, 10 √n)`` are removed and ordered last.
    """
    dense = _dense_threshold(len(adjacency))
    postponed = [
        variable for variable, neighbors in enumerate(adjacency) if len(neighbors) > dense
    ]
    graph = _QuotientGraph.from_adjacency(adjacency, postponed, meter)
    degree = [len(neighbors) for neighbors in graph.var_adjacent]
    return _minimum_degree_order(graph, degree, postponed, 0)


def _symmetric_adjacency(
    size: int, indices: np.ndarray, indptr: np.ndarray, meter: _WorkMeter, /
) -> list[list[int]]:
    """Sorted off-diagonal neighbor lists of ``A + Aᵀ``."""
    meter.add(2 * indices.size + size)
    rows = np.repeat(np.arange(size, dtype=np.int64), np.diff(indptr))
    off_diagonal = rows != indices
    keys = np.unique(
        np.concatenate(
            (
                rows[off_diagonal] * size + indices[off_diagonal],
                indices[off_diagonal] * size + rows[off_diagonal],
            )
        )
    )
    sources, targets = np.divmod(keys, size)
    bounds = np.concatenate(
        ([0], np.cumsum(np.bincount(sources, minlength=size)))
    ).tolist()
    neighbors = targets.tolist()
    return [neighbors[bounds[vertex] : bounds[vertex + 1]] for vertex in range(size)]


def _reverse_cuthill_mckee(
    size: int, indices: np.ndarray, indptr: np.ndarray, meter: _WorkMeter, /
) -> np.ndarray:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    # One symmetrization plus one level traversal of at most 2 nnz entries.
    meter.add(2 * indices.size + size)
    graph = sp.csr_matrix((np.ones(indices.size), indices, indptr), shape=(size, size))
    symmetric = graph + graph.T
    return np.asarray(
        reverse_cuthill_mckee(symmetric, symmetric_mode=True), dtype=np.int64
    )


def _induced_minimum_degree(
    vertices: list[int], adjacency: list[list[int]], meter: _WorkMeter, /
) -> list[int]:
    """Minimum-degree order of the subgraph induced by ascending ``vertices``."""
    local = {vertex: position for position, vertex in enumerate(vertices)}
    induced: list[list[int]] = []
    for vertex in vertices:
        meter.add(len(adjacency[vertex]))
        induced.append(
            [local[neighbor] for neighbor in adjacency[vertex] if neighbor in local]
        )
    return [vertices[position] for position in _symmetric_minimum_degree(induced, meter)]


def _levels(
    root: int, members: set[int], adjacency: list[list[int]], meter: _WorkMeter, /
) -> list[list[int]]:
    """Breadth-first level structure of the subgraph induced by ``members``."""
    seen = {root}
    frontier = [root]
    levels: list[list[int]] = []
    while frontier:
        levels.append(frontier)
        following: list[int] = []
        for vertex in frontier:
            meter.add(len(adjacency[vertex]))
            for neighbor in adjacency[vertex]:
                if neighbor in members and neighbor not in seen:
                    seen.add(neighbor)
                    following.append(neighbor)
        frontier = following
    return levels


def _components(
    vertices: list[int], adjacency: list[list[int]], meter: _WorkMeter, /
) -> list[list[int]]:
    """Connected components of the induced subgraph, by smallest vertex."""
    members = set(vertices)
    seen: set[int] = set()
    components: list[list[int]] = []
    for vertex in vertices:
        if vertex in seen:
            continue
        component = sorted(
            member
            for level in _levels(vertex, members, adjacency, meter)
            for member in level
        )
        seen.update(component)
        components.append(component)
    return components


def _pseudo_peripheral_levels(
    vertices: list[int],
    members: set[int],
    adjacency: list[list[int]],
    meter: _WorkMeter,
    /,
) -> list[list[int]]:
    """Level structure from a George–Liu pseudo-peripheral vertex."""

    def induced_degree(vertex: int, /) -> tuple[int, int]:
        meter.add(len(adjacency[vertex]))
        return sum(neighbor in members for neighbor in adjacency[vertex]), vertex

    levels = _levels(min(vertices, key=induced_degree), members, adjacency, meter)
    while True:
        candidate = min(levels[-1], key=induced_degree)
        candidate_levels = _levels(candidate, members, adjacency, meter)
        if len(candidate_levels) <= len(levels):
            return levels
        levels = candidate_levels


def _cut_cover(edges: list[tuple[int, int]], meter: _WorkMeter, /) -> set[int]:
    """Minimum vertex cover of bipartite cut edges ``(first, second)`` (König)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import maximum_bipartite_matching

    first = sorted({edge[0] for edge in edges})
    second = sorted({edge[1] for edge in edges})
    # Hopcroft–Karp performs O(E √V) work.
    meter.add(
        len(edges) * (isqrt(len(first) + len(second)) + 1) + len(first) + len(second)
    )
    first_index = {vertex: position for position, vertex in enumerate(first)}
    second_index = {vertex: position for position, vertex in enumerate(second)}
    rows = np.asarray([first_index[edge[0]] for edge in edges], dtype=np.int64)
    columns = np.asarray([second_index[edge[1]] for edge in edges], dtype=np.int64)
    bipartite = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, columns)),
        shape=(len(first), len(second)),
    )
    bipartite.sort_indices()
    matched_column = maximum_bipartite_matching(bipartite, perm_type="column").tolist()
    matched_row = [-1] * len(second)
    for row, column in enumerate(matched_column):
        if column >= 0:
            matched_row[column] = row
    # Z: vertices reachable from unmatched first-side vertices by alternating
    # paths. The cover is (first \ Z) ∪ (second ∩ Z).
    reached_rows = [column < 0 for column in matched_column]
    reached_columns = [False] * len(second)
    queue = [row for row, reached in enumerate(reached_rows) if reached]
    bounds = bipartite.indptr.tolist()
    targets = bipartite.indices.tolist()
    while queue:
        row = queue.pop()
        for column in targets[bounds[row] : bounds[row + 1]]:
            if reached_columns[column] or matched_column[row] == column:
                continue
            reached_columns[column] = True
            partner = matched_row[column]
            if partner >= 0 and not reached_rows[partner]:
                reached_rows[partner] = True
                queue.append(partner)
    return {first[row] for row, reached in enumerate(reached_rows) if not reached} | {
        second[column] for column, reached in enumerate(reached_columns) if reached
    }


type _Split = tuple[list[int], list[int], list[int]]


def _covered_split(
    first: set[int], second: set[int], adjacency: list[list[int]], meter: _WorkMeter, /
) -> _Split | None:
    """Separate a bipartition by covering its cut; refuse an emptied side."""
    edges: list[tuple[int, int]] = []
    for vertex in sorted(first):
        meter.add(len(adjacency[vertex]))
        edges.extend(
            (vertex, neighbor) for neighbor in adjacency[vertex] if neighbor in second
        )
    separator = _cut_cover(edges, meter)
    kept_first = sorted(first - separator)
    kept_second = sorted(second - separator)
    if not kept_first or not kept_second:
        return None
    return kept_first, kept_second, sorted(separator)


def _level_split(
    vertices: list[int], adjacency: list[list[int]], meter: _WorkMeter, /
) -> _Split | None:
    """Best covered cut between consecutive BFS levels of a connected part.

    Cuts leaving at least a quarter on each side are candidates (else the most
    balanced one). The separator-to-smaller-part ratio is minimized; ties keep
    the shallower cut.
    """
    members = set(vertices)
    levels = _pseudo_peripheral_levels(vertices, members, adjacency, meter)
    prefix = np.cumsum([len(level) for level in levels]).tolist()
    total = len(vertices)
    cuts = list(range(len(levels) - 1))
    balanced = [
        cut for cut in cuts if min(prefix[cut], total - prefix[cut]) >= total // 4
    ]
    if not balanced and cuts:
        balanced = [
            max(cuts, key=lambda cut: (min(prefix[cut], total - prefix[cut]), -cut))
        ]
    best: tuple[Fraction, _Split] | None = None
    for cut in balanced:
        below = {vertex for level in levels[: cut + 1] for vertex in level}
        split = _covered_split(below, members - below, adjacency, meter)
        if split is None:
            continue
        score = Fraction(len(split[2]), min(len(split[0]), len(split[1])))
        if best is None or score < best[0]:
            best = (score, split)
    return None if best is None else best[1]


def _geometric_split(
    vertices: list[int],
    coordinates: np.ndarray,
    adjacency: list[list[int]],
    meter: _WorkMeter,
    /,
) -> _Split | None:
    """Covered cut at the median of the widest coordinate axis of a part."""
    meter.add(len(vertices) * (coordinates.shape[1] + isqrt(len(vertices)) + 1))
    indices = np.asarray(vertices, dtype=np.int64)
    points = coordinates[indices]
    axis = np.argmax(points.max(axis=0) - points.min(axis=0))
    # Stable order: coordinate first, original index second.
    ranked = indices[np.lexsort((indices, points[:, axis]))].tolist()
    half = len(ranked) // 2
    return _covered_split(set(ranked[:half]), set(ranked[half:]), adjacency, meter)


@dataclass
class _Dissection:
    order: list[int]
    separator_sizes: list[int]
    leaf_sizes: list[int]


def _nested_dissection(
    adjacency: list[list[int]],
    coordinates: np.ndarray | None,
    leaf_capacity: int,
    separator_capacity: int,
    meter: _WorkMeter,
    /,
) -> _Dissection:
    """Order parts before their separators; ascending order inside separators."""
    result = _Dissection([], [], [])
    # (emit, vertices): emitted separators are appended; parts are dissected.
    stack: list[tuple[bool, list[int]]] = (
        [(False, list(range(len(adjacency))))] if adjacency else []
    )
    while stack:
        emit, vertices = stack.pop()
        if emit:
            result.order.extend(vertices)
            continue
        if len(vertices) > leaf_capacity:
            components = _components(vertices, adjacency, meter)
            if len(components) > 1:
                stack.extend((False, component) for component in reversed(components))
                continue
            split = (
                _level_split(vertices, adjacency, meter)
                if coordinates is None
                else _geometric_split(vertices, coordinates, adjacency, meter)
            )
            if split is not None:
                first, second, separator = split
                if len(separator) > separator_capacity:
                    raise LinearCapabilityError(
                        "Sparse ordering refused before completion: nested-dissection "
                        f"separator of {len(separator)} vertices exceeds "
                        f"separator_capacity {separator_capacity}."
                    )
                result.separator_sizes.append(len(separator))
                stack.extend(((True, separator), (False, second), (False, first)))
                continue
        result.leaf_sizes.append(len(vertices))
        result.order.extend(_induced_minimum_degree(vertices, adjacency, meter))
    return result


def _host_coordinates(
    coordinates: ConvertibleToArray, size: int, /
) -> tuple[np.ndarray, str]:
    scope = Scope()
    parse(size, Size[_OrderedNodeDim], "size", scope=scope)
    points = as_host_array(
        coordinates,
        HostFloat64[_OrderedNodeDim, _CoordinateAxisDim],
        "coordinates",
        scope=scope,
    )
    if not np.all(np.isfinite(points)):
        raise ValueError("Nested-dissection coordinates must be finite.")
    payload = np.asarray(points.shape, dtype=np.int64).tobytes() + points.tobytes()
    return points, sha256(payload).hexdigest()


def _order_pattern(
    size: int,
    indices: np.ndarray,
    indptr: np.ndarray,
    policy: SparseOrderingPolicy,
    coordinates: ConvertibleToArray | None,
    meter: _WorkMeter,
    /,
) -> PreparedSparseOrdering:
    """Order one validated square CSR pattern, charging ``meter``."""
    if coordinates is not None and policy.method != "nested-dissection":
        raise ValueError("coordinates apply only to nested-dissection orderings.")
    pattern_id = _pattern_identifier((size, size), indices, indptr)
    dissection: NestedDissectionEvidence | None = None
    coordinates_id: str | None = None
    match policy.method:
        case "natural":
            permutation = np.arange(size, dtype=np.int64)
        case "reverse-cuthill-mckee":
            permutation = _reverse_cuthill_mckee(size, indices, indptr, meter)
        case "approximate-minimum-degree":
            adjacency = _symmetric_adjacency(size, indices, indptr, meter)
            permutation = np.asarray(
                _symmetric_minimum_degree(adjacency, meter), dtype=np.int64
            )
        case "nested-dissection":
            leaf_capacity = policy.leaf_capacity
            separator_capacity = policy.separator_capacity
            if leaf_capacity is None or separator_capacity is None:
                raise ValueError(
                    "Nested dissection requires leaf and separator capacities."
                )
            points: np.ndarray | None = None
            if coordinates is not None:
                points, coordinates_id = _host_coordinates(coordinates, size)
            adjacency = _symmetric_adjacency(size, indices, indptr, meter)
            dissected = _nested_dissection(
                adjacency, points, leaf_capacity, separator_capacity, meter
            )
            permutation = np.asarray(dissected.order, dtype=np.int64)
            dissection = NestedDissectionEvidence(
                separator_sizes=np.asarray(dissected.separator_sizes, dtype=np.int64),
                leaf_sizes=np.asarray(dissected.leaf_sizes, dtype=np.int64),
                route="graph" if points is None else "geometric",
                leaf_capacity=leaf_capacity,
                separator_capacity=separator_capacity,
            )
        case _:
            assert_never(policy.method)
    return PreparedSparseOrdering(
        permutation,
        policy=policy,
        pattern_id=pattern_id,
        work=meter.total,
        dissection=dissection,
        coordinates_id=coordinates_id,
    )


# Ordering reads the concrete host pattern even when a caller prepares inside
# an ambient trace.
@jax.ensure_compile_time_eval()
def prepare_sparse_ordering(
    operator: AbstractSparseLinearOperator,
    policy: SparseOrderingPolicy,
    /,
    *,
    coordinates: ConvertibleToArray | None = None,
) -> PreparedSparseOrdering:
    """Prepare a reusable symmetric fill-reducing ordering of ``operator``'s pattern.

    Values are never read. ``coordinates`` (shape ``(n, d)``, finite) select the
    geometric nested-dissection route. Work above ``policy.max_ordering_work``
    and nested-dissection separators above ``policy.separator_capacity`` raise
    `LinearCapabilityError` before an ordering is returned. Pass the result to
    `prepare_sparse_factorization` to reuse it across symbolic plans.
    """
    if not isinstance(policy, SparseOrderingPolicy):
        raise TypeError("policy must be a SparseOrderingPolicy.")
    storage, indices, indptr = _validated_pattern(operator)
    return _order_pattern(
        storage.shape[0],
        indices,
        indptr,
        policy,
        coordinates,
        _WorkMeter(policy.max_ordering_work),
    )


__all__ = [
    "NestedDissectionEvidence",
    "NestedDissectionRoute",
    "PreparedSparseOrdering",
    "SparseOrdering",
    "SparseOrderingPolicy",
    "prepare_sparse_ordering",
]
