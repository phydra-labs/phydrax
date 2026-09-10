#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import enum
import itertools
from collections import deque
from collections.abc import Iterable, Sequence
from typing import TypeAlias

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._core import CausalSchema, VariableObservability


DirectedEdge: TypeAlias = tuple[str, str]
SymmetricEdge: TypeAlias = tuple[str, str]


class EndpointMark(enum.StrEnum):
    TAIL = "tail"
    ARROW = "arrow"
    CIRCLE = "circle"


class GraphValidationStatus(enum.StrEnum):
    VERIFIED = "verified"
    LOCALLY_VALID = "locally_valid"
    RESOURCE_EXHAUSTED = "resource_exhausted"


class SeparationStatus(enum.StrEnum):
    SEPARATED = "separated"
    CONNECTED = "connected"
    RESOURCE_EXHAUSTED = "resource_exhausted"


class SeparationResult(StrictModule, NonTrainableState):
    status: SeparationStatus = eqx.field(static=True)
    left: tuple[str, ...] = eqx.field(static=True)
    right: tuple[str, ...] = eqx.field(static=True)
    conditioned: tuple[str, ...] = eqx.field(static=True)
    active_path: tuple[str, ...] = eqx.field(static=True)
    explored_paths: int = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def separated(self) -> bool:
        return self.status is SeparationStatus.SEPARATED


class CausalDAG(StrictModule, NonTrainableState):
    """Canonical directed acyclic causal structure."""

    schema: CausalSchema = eqx.field(static=True)
    directed_edges: tuple[DirectedEdge, ...] = eqx.field(static=True)
    topological_order: tuple[str, ...] = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        schema: CausalSchema,
        directed_edges: Iterable[DirectedEdge],
    ) -> None:
        canonical = _canonical_directed(schema, directed_edges)
        order = _topological_order(schema.names, canonical)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "directed_edges", canonical)
        object.__setattr__(self, "topological_order", order)
        object.__setattr__(
            self,
            "graph_id",
            canonical_fingerprint(
                {
                    "kind": "dag",
                    "schema_id": schema.schema_id,
                    "directed_edges": canonical,
                }
            ),
        )

    def parents(self, node: str, /) -> tuple[str, ...]:
        self.schema.index(node)
        return tuple(source for source, target in self.directed_edges if target == node)

    def children(self, node: str, /) -> tuple[str, ...]:
        self.schema.index(node)
        return tuple(target for source, target in self.directed_edges if source == node)


class CausalADMG(StrictModule, NonTrainableState):
    """Observed-variable acyclic directed mixed graph."""

    schema: CausalSchema = eqx.field(static=True)
    directed_edges: tuple[DirectedEdge, ...] = eqx.field(static=True)
    bidirected_edges: tuple[SymmetricEdge, ...] = eqx.field(static=True)
    topological_order: tuple[str, ...] = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        schema: CausalSchema,
        directed_edges: Iterable[DirectedEdge],
        bidirected_edges: Iterable[SymmetricEdge] = (),
    ) -> None:
        latent = [
            variable.name
            for variable in schema.variables
            if variable.observability is VariableObservability.LATENT
        ]
        if latent:
            raise ValueError(
                "CausalADMG is a latent projection over observed variables; "
                f"explicit latent nodes are not allowed: {latent}."
            )
        canonical_directed = _canonical_directed(schema, directed_edges)
        canonical_bidirected = _canonical_symmetric(schema, bidirected_edges)
        order = _topological_order(schema.names, canonical_directed)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "directed_edges", canonical_directed)
        object.__setattr__(self, "bidirected_edges", canonical_bidirected)
        object.__setattr__(self, "topological_order", order)
        object.__setattr__(
            self,
            "graph_id",
            canonical_fingerprint(
                {
                    "kind": "admg",
                    "schema_id": schema.schema_id,
                    "directed_edges": canonical_directed,
                    "bidirected_edges": canonical_bidirected,
                }
            ),
        )

    def parents(self, node: str, /) -> tuple[str, ...]:
        self.schema.index(node)
        return tuple(source for source, target in self.directed_edges if target == node)


class CausalMAG(StrictModule, NonTrainableState):
    """Locally validated ancestral mixed graph for latent/selection semantics."""

    schema: CausalSchema = eqx.field(static=True)
    directed_edges: tuple[DirectedEdge, ...] = eqx.field(static=True)
    bidirected_edges: tuple[SymmetricEdge, ...] = eqx.field(static=True)
    undirected_edges: tuple[SymmetricEdge, ...] = eqx.field(static=True)
    validation_status: GraphValidationStatus = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        schema: CausalSchema,
        directed_edges: Iterable[DirectedEdge],
        bidirected_edges: Iterable[SymmetricEdge] = (),
        undirected_edges: Iterable[SymmetricEdge] = (),
        verify_maximality: bool = True,
        maximum_conditioning_subsets: int = 65_536,
    ) -> None:
        directed = _canonical_directed(schema, directed_edges)
        bidirected = _canonical_symmetric(schema, bidirected_edges)
        undirected = _canonical_symmetric(schema, undirected_edges)
        _topological_order(schema.names, directed)
        _validate_single_mixed_edge(schema.names, directed, bidirected, undirected)
        ancestors = _ancestor_map(schema.names, directed)
        for left, right in bidirected:
            if left in ancestors[right] or right in ancestors[left]:
                raise ValueError(
                    "MAG bidirected endpoints cannot be ancestors of each other."
                )
        undirected_nodes = {node for edge in undirected for node in edge}
        for node in undirected_nodes:
            if any(target == node for _, target in directed):
                raise ValueError(
                    "A MAG node incident to an undirected edge cannot have parents."
                )
            if any(node in edge for edge in bidirected):
                raise ValueError(
                    "A MAG node incident to an undirected edge cannot have spouses."
                )
        status = GraphValidationStatus.LOCALLY_VALID
        if verify_maximality:
            graph = _MixedGraphView(
                names=schema.names,
                directed=directed,
                bidirected=bidirected,
                undirected=undirected,
            )
            complete, maximal = _verify_maximality(
                graph,
                schema=schema,
                maximum_subsets=int(maximum_conditioning_subsets),
            )
            if not complete:
                status = GraphValidationStatus.RESOURCE_EXHAUSTED
            elif not maximal:
                raise ValueError("The supplied ancestral graph is not maximal.")
            else:
                status = GraphValidationStatus.VERIFIED
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "directed_edges", directed)
        object.__setattr__(self, "bidirected_edges", bidirected)
        object.__setattr__(self, "undirected_edges", undirected)
        object.__setattr__(self, "validation_status", status)
        object.__setattr__(
            self,
            "graph_id",
            canonical_fingerprint(
                {
                    "kind": "mag",
                    "schema_id": schema.schema_id,
                    "directed_edges": directed,
                    "bidirected_edges": bidirected,
                    "undirected_edges": undirected,
                }
            ),
        )


class CausalPDAG(StrictModule, NonTrainableState):
    """Partially directed graph without a completed-equivalence-class claim."""

    schema: CausalSchema = eqx.field(static=True)
    directed_edges: tuple[DirectedEdge, ...] = eqx.field(static=True)
    undirected_edges: tuple[SymmetricEdge, ...] = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        schema: CausalSchema,
        directed_edges: Iterable[DirectedEdge] = (),
        undirected_edges: Iterable[SymmetricEdge] = (),
    ) -> None:
        directed = _canonical_directed(schema, directed_edges)
        undirected = _canonical_symmetric(schema, undirected_edges)
        _topological_order(schema.names, directed)
        _validate_single_mixed_edge(schema.names, directed, (), undirected)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "directed_edges", directed)
        object.__setattr__(self, "undirected_edges", undirected)
        object.__setattr__(
            self,
            "graph_id",
            canonical_fingerprint(
                {
                    "kind": "pdag",
                    "schema_id": schema.schema_id,
                    "directed_edges": directed,
                    "undirected_edges": undirected,
                }
            ),
        )


class CausalCPDAG(StrictModule, NonTrainableState):
    """Completed partially directed graph representing a DAG equivalence class."""

    schema: CausalSchema = eqx.field(static=True)
    directed_edges: tuple[DirectedEdge, ...] = eqx.field(static=True)
    undirected_edges: tuple[SymmetricEdge, ...] = eqx.field(static=True)
    extensions: tuple[CausalDAG, ...] = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        schema: CausalSchema,
        directed_edges: Iterable[DirectedEdge] = (),
        undirected_edges: Iterable[SymmetricEdge] = (),
        maximum_extensions: int = 65_536,
    ) -> None:
        pdag = CausalPDAG(
            schema=schema,
            directed_edges=directed_edges,
            undirected_edges=undirected_edges,
        )
        extensions, complete = _consistent_extensions(
            pdag,
            maximum_extensions=int(maximum_extensions),
        )
        if not complete:
            raise ValueError("CPDAG completion exceeded maximum_extensions.")
        if not extensions:
            raise ValueError("The supplied PDAG has no consistent DAG extension.")
        completed_directed, completed_undirected = _complete_from_extensions(extensions)
        if (
            completed_directed != pdag.directed_edges
            or completed_undirected != pdag.undirected_edges
        ):
            raise ValueError("The supplied graph is a PDAG but not a completed PDAG.")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "directed_edges", pdag.directed_edges)
        object.__setattr__(self, "undirected_edges", pdag.undirected_edges)
        object.__setattr__(self, "extensions", extensions)
        object.__setattr__(
            self,
            "graph_id",
            canonical_fingerprint(
                {
                    "kind": "cpdag",
                    "schema_id": schema.schema_id,
                    "directed_edges": pdag.directed_edges,
                    "undirected_edges": pdag.undirected_edges,
                }
            ),
        )


class CausalPAG(StrictModule, NonTrainableState):
    """Locally valid partial ancestral graph with explicit endpoint marks."""

    schema: CausalSchema = eqx.field(static=True)
    endpoint_edges: tuple[tuple[str, EndpointMark, str, EndpointMark], ...] = eqx.field(
        static=True
    )
    provenance_id: str = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        schema: CausalSchema,
        endpoint_edges: Iterable[tuple[str, EndpointMark | str, str, EndpointMark | str]],
        provenance_id: str,
    ) -> None:
        provenance = str(provenance_id).strip()
        if not provenance:
            raise ValueError("PAG construction requires validation/discovery provenance.")
        index = {name: position for position, name in enumerate(schema.names)}
        canonical: list[tuple[str, EndpointMark, str, EndpointMark]] = []
        seen: set[tuple[str, str]] = set()
        for raw_left, raw_left_mark, raw_right, raw_right_mark in endpoint_edges:
            if raw_left not in index or raw_right not in index:
                raise ValueError("PAG endpoint references an unknown schema variable.")
            if raw_left == raw_right:
                raise ValueError("PAG self-edges are invalid.")
            left_mark = EndpointMark(raw_left_mark)
            right_mark = EndpointMark(raw_right_mark)
            if index[raw_left] < index[raw_right]:
                edge = (raw_left, left_mark, raw_right, right_mark)
            else:
                edge = (raw_right, right_mark, raw_left, left_mark)
            pair = (edge[0], edge[2])
            if pair in seen:
                raise ValueError("PAG has more than one endpoint edge for a node pair.")
            seen.add(pair)
            canonical.append(edge)
        canonical_tuple = tuple(
            sorted(canonical, key=lambda edge: (index[edge[0]], index[edge[2]]))
        )
        definite_directed = tuple(
            (left, right)
            if left_mark is EndpointMark.TAIL and right_mark is EndpointMark.ARROW
            else (right, left)
            for left, left_mark, right, right_mark in canonical_tuple
            if {left_mark, right_mark} == {EndpointMark.TAIL, EndpointMark.ARROW}
        )
        _topological_order(schema.names, definite_directed)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "endpoint_edges", canonical_tuple)
        object.__setattr__(self, "provenance_id", provenance)
        object.__setattr__(
            self,
            "graph_id",
            canonical_fingerprint(
                {
                    "kind": "pag",
                    "schema_id": schema.schema_id,
                    "endpoint_edges": [
                        (left, left_mark.value, right, right_mark.value)
                        for left, left_mark, right, right_mark in canonical_tuple
                    ],
                }
            ),
        )


class _MixedGraphView:
    def __init__(
        self,
        *,
        names: tuple[str, ...],
        directed: tuple[DirectedEdge, ...],
        bidirected: tuple[SymmetricEdge, ...],
        undirected: tuple[SymmetricEdge, ...],
    ) -> None:
        self.names = names
        self.directed = directed
        self.bidirected = bidirected
        self.undirected = undirected


def ancestors(
    graph: CausalDAG | CausalADMG | CausalMAG,
    nodes: Iterable[str],
    /,
) -> tuple[str, ...]:
    requested = _canonical_nodes(graph.schema, nodes)
    mapping = _ancestor_map(graph.schema.names, graph.directed_edges)
    result = set(requested)
    for node in requested:
        result.update(mapping[node])
    return tuple(name for name in graph.schema.names if name in result)


def descendants(
    graph: CausalDAG | CausalADMG | CausalMAG,
    nodes: Iterable[str],
    /,
) -> tuple[str, ...]:
    requested = _canonical_nodes(graph.schema, nodes)
    result = set(requested)
    queue = deque(requested)
    children = _children_map(graph.schema.names, graph.directed_edges)
    while queue:
        node = queue.popleft()
        for child in children[node]:
            if child not in result:
                result.add(child)
                queue.append(child)
    return tuple(name for name in graph.schema.names if name in result)


def d_separated(
    graph: CausalDAG,
    left: Iterable[str],
    right: Iterable[str],
    conditioned: Iterable[str] = (),
    *,
    maximum_paths: int = 100_000,
) -> SeparationResult:
    return _separation(
        _MixedGraphView(
            names=graph.schema.names,
            directed=graph.directed_edges,
            bidirected=(),
            undirected=(),
        ),
        schema=graph.schema,
        left=left,
        right=right,
        conditioned=conditioned,
        maximum_paths=maximum_paths,
    )


def m_separated(
    graph: CausalADMG | CausalMAG,
    left: Iterable[str],
    right: Iterable[str],
    conditioned: Iterable[str] = (),
    *,
    maximum_paths: int = 100_000,
) -> SeparationResult:
    return _separation(
        _MixedGraphView(
            names=graph.schema.names,
            directed=graph.directed_edges,
            bidirected=graph.bidirected_edges,
            undirected=graph.undirected_edges if isinstance(graph, CausalMAG) else (),
        ),
        schema=graph.schema,
        left=left,
        right=right,
        conditioned=conditioned,
        maximum_paths=maximum_paths,
    )


def latent_project(
    graph: CausalDAG,
    observed: Iterable[str] | None = None,
) -> CausalADMG:
    """Project explicit latent vertices from a DAG into an observed ADMG."""
    observed_names = (
        tuple(
            variable.name
            for variable in graph.schema.variables
            if variable.observability is not VariableObservability.LATENT
        )
        if observed is None
        else _canonical_nodes(graph.schema, observed)
    )
    if not observed_names:
        raise ValueError("Latent projection requires observed vertices.")
    observed_set = set(observed_names)
    variables = tuple(graph.schema.variable(name) for name in observed_names)
    projected_schema = CausalSchema(variables)
    children = _children_map(graph.schema.names, graph.directed_edges)
    directed: set[DirectedEdge] = set()
    for source in observed_names:
        queue = deque(children[source])
        visited: set[str] = set()
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if node in observed_set:
                if node != source:
                    directed.add((source, node))
                continue
            queue.extend(children[node])
    bidirected: set[SymmetricEdge] = set()
    latent_names = set(graph.schema.names) - observed_set
    index = {name: position for position, name in enumerate(observed_names)}
    for latent in latent_names:
        reachable: set[str] = set()
        queue = deque(children[latent])
        visited: set[str] = set()
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if node in observed_set:
                reachable.add(node)
                continue
            queue.extend(children[node])
        for left, right in itertools.combinations(
            sorted(reachable, key=lambda name: index[name]), 2
        ):
            bidirected.add((left, right))
    return CausalADMG(
        schema=projected_schema,
        directed_edges=directed,
        bidirected_edges=bidirected,
    )


def intervene_graph(
    graph: CausalDAG | CausalADMG,
    targets: Iterable[str],
) -> CausalDAG | CausalADMG:
    """Return perfect-intervention graph surgery for atomic/external regimes."""
    canonical_targets = set(_canonical_nodes(graph.schema, targets))
    directed = tuple(
        edge for edge in graph.directed_edges if edge[1] not in canonical_targets
    )
    if isinstance(graph, CausalDAG):
        return CausalDAG(schema=graph.schema, directed_edges=directed)
    bidirected = tuple(
        edge
        for edge in graph.bidirected_edges
        if edge[0] not in canonical_targets and edge[1] not in canonical_targets
    )
    return CausalADMG(
        schema=graph.schema,
        directed_edges=directed,
        bidirected_edges=bidirected,
    )


def verified_cpdag(
    graph: CausalPDAG,
    *,
    maximum_extensions: int = 65_536,
) -> CausalCPDAG | None:
    """Return a completed equivalence-class graph or ``None`` without repair."""
    extensions, complete = _consistent_extensions(
        graph,
        maximum_extensions=maximum_extensions,
    )
    if not complete or not extensions:
        return None
    directed, undirected = _complete_from_extensions(extensions)
    if directed != graph.directed_edges or undirected != graph.undirected_edges:
        return None
    return CausalCPDAG(
        schema=graph.schema,
        directed_edges=directed,
        undirected_edges=undirected,
        maximum_extensions=maximum_extensions,
    )


def complete_dag_equivalence_class(
    graph: CausalDAG,
    *,
    maximum_extensions: int = 65_536,
) -> CausalCPDAG:
    skeleton = _skeleton(graph.directed_edges, (), ())
    v_structures = _v_structures(graph.schema.names, graph.directed_edges)
    directed: set[DirectedEdge] = set()
    for left, collider, right in v_structures:
        directed.add((left, collider))
        directed.add((right, collider))
    undirected = {
        edge
        for edge in skeleton
        if edge not in {_canonical_pair(*item, graph.schema.names) for item in directed}
    }
    # Constructor computes all consistent extensions and exact completion.
    candidate = CausalPDAG(
        schema=graph.schema,
        directed_edges=directed,
        undirected_edges=undirected,
    )
    extensions, complete = _consistent_extensions(
        candidate,
        maximum_extensions=maximum_extensions,
    )
    if not complete:
        raise ValueError("DAG equivalence completion exceeded maximum_extensions.")
    completed_directed, completed_undirected = _complete_from_extensions(extensions)
    return CausalCPDAG(
        schema=graph.schema,
        directed_edges=completed_directed,
        undirected_edges=completed_undirected,
        maximum_extensions=maximum_extensions,
    )


def _canonical_directed(
    schema: CausalSchema,
    edges: Iterable[DirectedEdge],
) -> tuple[DirectedEdge, ...]:
    index = {name: position for position, name in enumerate(schema.names)}
    canonical: set[DirectedEdge] = set()
    for source, target in edges:
        if source not in index or target not in index:
            raise ValueError("Directed edge references an unknown schema variable.")
        if source == target:
            raise ValueError("Causal self-edges are invalid.")
        if (target, source) in canonical:
            raise ValueError("Opposing directed edges are invalid in an acyclic graph.")
        canonical.add((source, target))
    return tuple(sorted(canonical, key=lambda edge: (index[edge[0]], index[edge[1]])))


def _canonical_pair(left: str, right: str, names: Sequence[str]) -> SymmetricEdge:
    index = {name: position for position, name in enumerate(names)}
    return (left, right) if index[left] < index[right] else (right, left)


def _canonical_symmetric(
    schema: CausalSchema,
    edges: Iterable[SymmetricEdge],
) -> tuple[SymmetricEdge, ...]:
    index = {name: position for position, name in enumerate(schema.names)}
    canonical: set[SymmetricEdge] = set()
    for left, right in edges:
        if left not in index or right not in index:
            raise ValueError("Mixed edge references an unknown schema variable.")
        if left == right:
            raise ValueError("Causal self-edges are invalid.")
        canonical.add((left, right) if index[left] < index[right] else (right, left))
    return tuple(sorted(canonical, key=lambda edge: (index[edge[0]], index[edge[1]])))


def _canonical_nodes(schema: CausalSchema, nodes: Iterable[str]) -> tuple[str, ...]:
    requested = set(nodes)
    unknown = requested - set(schema.names)
    if unknown:
        raise ValueError(f"Unknown causal variables: {sorted(unknown)}.")
    return tuple(name for name in schema.names if name in requested)


def _validate_single_mixed_edge(
    names: Sequence[str],
    directed: Sequence[DirectedEdge],
    bidirected: Sequence[SymmetricEdge],
    undirected: Sequence[SymmetricEdge],
) -> None:
    directed_pairs = {_canonical_pair(*edge, names) for edge in directed}
    bidirected_set = set(bidirected)
    undirected_set = set(undirected)
    if (
        directed_pairs & bidirected_set
        or directed_pairs & undirected_set
        or bidirected_set & undirected_set
    ):
        raise ValueError("This graph kind permits only one edge type per node pair.")


def _topological_order(
    names: Sequence[str],
    directed: Sequence[DirectedEdge],
) -> tuple[str, ...]:
    index = {name: position for position, name in enumerate(names)}
    indegree = {name: 0 for name in names}
    children = {name: [] for name in names}
    for source, target in directed:
        indegree[target] += 1
        children[source].append(target)
    queue = [name for name in names if indegree[name] == 0]
    queue.sort(key=lambda name: index[name])
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in sorted(children[node], key=lambda name: index[name]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
                queue.sort(key=lambda name: index[name])
    if len(order) != len(names):
        raise ValueError("Directed causal edges must be acyclic.")
    return tuple(order)


def _children_map(
    names: Sequence[str],
    directed: Sequence[DirectedEdge],
) -> dict[str, tuple[str, ...]]:
    index = {name: position for position, name in enumerate(names)}
    result: dict[str, list[str]] = {name: [] for name in names}
    for source, target in directed:
        result[source].append(target)
    return {
        name: tuple(sorted(children, key=lambda child: index[child]))
        for name, children in result.items()
    }


def _ancestor_map(
    names: Sequence[str],
    directed: Sequence[DirectedEdge],
) -> dict[str, set[str]]:
    parents: dict[str, set[str]] = {name: set() for name in names}
    for source, target in directed:
        parents[target].add(source)
    result: dict[str, set[str]] = {name: set() for name in names}
    for name in names:
        queue = deque(parents[name])
        while queue:
            parent = queue.popleft()
            if parent in result[name]:
                continue
            result[name].add(parent)
            queue.extend(parents[parent])
    return result


def _skeleton(
    directed: Sequence[DirectedEdge],
    bidirected: Sequence[SymmetricEdge],
    undirected: Sequence[SymmetricEdge],
    names: Sequence[str] | None = None,
) -> set[SymmetricEdge]:
    if names is None:
        names = tuple(dict.fromkeys(node for edge in directed for node in edge))
        names += tuple(node for edge in bidirected for node in edge if node not in names)
        names += tuple(node for edge in undirected for node in edge if node not in names)
    result = {_canonical_pair(*edge, names) for edge in directed}
    result.update(_canonical_pair(*edge, names) for edge in bidirected)
    result.update(_canonical_pair(*edge, names) for edge in undirected)
    return result


def _edge_mark_at(graph: _MixedGraphView, previous: str, node: str) -> EndpointMark:
    if (previous, node) in graph.directed:
        return EndpointMark.ARROW
    if (node, previous) in graph.directed:
        return EndpointMark.TAIL
    pair = _canonical_pair(previous, node, graph.names)
    if pair in graph.bidirected:
        return EndpointMark.ARROW
    if pair in graph.undirected:
        return EndpointMark.TAIL
    raise ValueError("Path contains a nonexistent edge.")


def _separation(
    graph: _MixedGraphView,
    *,
    schema: CausalSchema,
    left: Iterable[str],
    right: Iterable[str],
    conditioned: Iterable[str],
    maximum_paths: int,
) -> SeparationResult:
    canonical_left = _canonical_nodes(schema, left)
    canonical_right = _canonical_nodes(schema, right)
    canonical_conditioned = _canonical_nodes(schema, conditioned)
    if not canonical_left or not canonical_right:
        raise ValueError("Separation queries need non-empty left and right sets.")
    if set(canonical_left) & set(canonical_right):
        raise ValueError("Separation query endpoints must be disjoint.")
    if (set(canonical_left) | set(canonical_right)) & set(canonical_conditioned):
        raise ValueError("Conditioned variables cannot be separation endpoints.")
    if maximum_paths < 1:
        raise ValueError("maximum_paths must be positive.")
    adjacency = {name: set() for name in graph.names}
    for first, second in _skeleton(
        graph.directed,
        graph.bidirected,
        graph.undirected,
        graph.names,
    ):
        adjacency[first].add(second)
        adjacency[second].add(first)
    ancestors_of_conditioned = set(canonical_conditioned)
    ancestor_map = _ancestor_map(graph.names, graph.directed)
    for node in canonical_conditioned:
        ancestors_of_conditioned.update(ancestor_map[node])
    explored = 0
    for source in canonical_left:
        stack: list[tuple[str, tuple[str, ...]]] = [(source, (source,))]
        while stack:
            node, path = stack.pop()
            if node in canonical_right:
                payload = {
                    "status": SeparationStatus.CONNECTED.value,
                    "left": canonical_left,
                    "right": canonical_right,
                    "conditioned": canonical_conditioned,
                    "active_path": path,
                    "explored_paths": explored,
                }
                return SeparationResult(
                    status=SeparationStatus.CONNECTED,
                    left=canonical_left,
                    right=canonical_right,
                    conditioned=canonical_conditioned,
                    active_path=path,
                    explored_paths=explored,
                    result_id=canonical_fingerprint(payload),
                )
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor in path:
                    continue
                candidate = path + (neighbor,)
                explored += 1
                if explored > maximum_paths:
                    payload = {
                        "status": SeparationStatus.RESOURCE_EXHAUSTED.value,
                        "left": canonical_left,
                        "right": canonical_right,
                        "conditioned": canonical_conditioned,
                        "active_path": (),
                        "explored_paths": explored,
                    }
                    return SeparationResult(
                        status=SeparationStatus.RESOURCE_EXHAUSTED,
                        left=canonical_left,
                        right=canonical_right,
                        conditioned=canonical_conditioned,
                        active_path=(),
                        explored_paths=explored,
                        result_id=canonical_fingerprint(payload),
                    )
                if len(candidate) >= 3:
                    previous, middle, following = candidate[-3:]
                    collider = (
                        _edge_mark_at(graph, previous, middle) is EndpointMark.ARROW
                        and _edge_mark_at(graph, following, middle) is EndpointMark.ARROW
                    )
                    if collider and middle not in ancestors_of_conditioned:
                        continue
                    if not collider and middle in canonical_conditioned:
                        continue
                stack.append((neighbor, candidate))
    payload = {
        "status": SeparationStatus.SEPARATED.value,
        "left": canonical_left,
        "right": canonical_right,
        "conditioned": canonical_conditioned,
        "active_path": (),
        "explored_paths": explored,
    }
    return SeparationResult(
        status=SeparationStatus.SEPARATED,
        left=canonical_left,
        right=canonical_right,
        conditioned=canonical_conditioned,
        active_path=(),
        explored_paths=explored,
        result_id=canonical_fingerprint(payload),
    )


def _verify_maximality(
    graph: _MixedGraphView,
    *,
    schema: CausalSchema,
    maximum_subsets: int,
) -> tuple[bool, bool]:
    skeleton = _skeleton(
        graph.directed,
        graph.bidirected,
        graph.undirected,
        graph.names,
    )
    tested = 0
    for left_index, left in enumerate(graph.names):
        for right in graph.names[left_index + 1 :]:
            if _canonical_pair(left, right, graph.names) in skeleton:
                continue
            candidates = [node for node in graph.names if node not in {left, right}]
            found = False
            for size in range(len(candidates) + 1):
                for conditioned in itertools.combinations(candidates, size):
                    tested += 1
                    if tested > maximum_subsets:
                        return False, False
                    result = _separation(
                        graph,
                        schema=schema,
                        left=(left,),
                        right=(right,),
                        conditioned=conditioned,
                        maximum_paths=maximum_subsets,
                    )
                    if result.status is SeparationStatus.RESOURCE_EXHAUSTED:
                        return False, False
                    if result.status is SeparationStatus.SEPARATED:
                        found = True
                        break
                if found:
                    break
            if not found:
                return True, False
    return True, True


def _v_structures(
    names: Sequence[str],
    directed: Sequence[DirectedEdge],
) -> set[tuple[str, str, str]]:
    parents: dict[str, list[str]] = {name: [] for name in names}
    skeleton = _skeleton(directed, (), (), names)
    for source, target in directed:
        parents[target].append(source)
    index = {name: position for position, name in enumerate(names)}
    result: set[tuple[str, str, str]] = set()
    for collider, incoming in parents.items():
        for left, right in itertools.combinations(
            sorted(incoming, key=lambda name: index[name]), 2
        ):
            if _canonical_pair(left, right, names) not in skeleton:
                result.add((left, collider, right))
    return result


def _is_acyclic(names: Sequence[str], directed: Sequence[DirectedEdge]) -> bool:
    indegree = {name: 0 for name in names}
    children = {name: [] for name in names}
    for source, target in directed:
        if source == target or source not in indegree or target not in indegree:
            return False
        indegree[target] += 1
        children[source].append(target)
    queue = deque(name for name in names if indegree[name] == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return visited == len(names)


def _consistent_extensions(
    graph: CausalPDAG,
    *,
    maximum_extensions: int,
) -> tuple[tuple[CausalDAG, ...], bool]:
    if maximum_extensions < 1:
        raise ValueError("maximum_extensions must be positive.")
    reference_colliders = _v_structures(graph.schema.names, graph.directed_edges)
    extensions: list[CausalDAG] = []
    attempted = 0
    for orientations in itertools.product((0, 1), repeat=len(graph.undirected_edges)):
        attempted += 1
        if attempted > maximum_extensions:
            return tuple(extensions), False
        directed = list(graph.directed_edges)
        for orientation, (left, right) in zip(
            orientations,
            graph.undirected_edges,
            strict=True,
        ):
            directed.append((left, right) if orientation == 0 else (right, left))
        canonical_directed = tuple(directed)
        if not _is_acyclic(graph.schema.names, canonical_directed):
            continue
        try_graph = CausalDAG(schema=graph.schema, directed_edges=canonical_directed)
        if (
            _v_structures(graph.schema.names, try_graph.directed_edges)
            == reference_colliders
        ):
            extensions.append(try_graph)
    return tuple(extensions), True


def _complete_from_extensions(
    extensions: Sequence[CausalDAG],
) -> tuple[tuple[DirectedEdge, ...], tuple[SymmetricEdge, ...]]:
    if not extensions:
        raise ValueError("At least one DAG extension is required.")
    schema = extensions[0].schema
    skeleton = _skeleton(extensions[0].directed_edges, (), (), schema.names)
    directed: set[DirectedEdge] = set()
    undirected: set[SymmetricEdge] = set()
    for left, right in skeleton:
        orientations = {
            (left, right) if (left, right) in graph.directed_edges else (right, left)
            for graph in extensions
        }
        if len(orientations) == 1:
            directed.add(next(iter(orientations)))
        else:
            undirected.add((left, right))
    index = {name: position for position, name in enumerate(schema.names)}
    return (
        tuple(sorted(directed, key=lambda edge: (index[edge[0]], index[edge[1]]))),
        tuple(sorted(undirected, key=lambda edge: (index[edge[0]], index[edge[1]]))),
    )


__all__ = [
    "CausalADMG",
    "CausalCPDAG",
    "CausalDAG",
    "CausalMAG",
    "CausalPAG",
    "CausalPDAG",
    "EndpointMark",
    "GraphValidationStatus",
    "SeparationResult",
    "SeparationStatus",
    "ancestors",
    "complete_dag_equivalence_class",
    "d_separated",
    "descendants",
    "intervene_graph",
    "latent_project",
    "m_separated",
    "verified_cpdag",
]
