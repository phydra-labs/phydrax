#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Iterator, Mapping

import coordax as cx
from jaxtyping import PyTree

from ..._frozendict import frozendict
from ..._strict import StrictModule
from ...graph import GraphIR
from .._structure import _validate_reserved_axes, Points, ProductStructure
from ._components import GraphComponentKind


GRAPH_ENTITY_INDEX_KEY = "__phydrax_graph_entity_index__"
GRAPH_GRAPH_INDEX_KEY = "__phydrax_graph_index__"


class GraphBatch(StrictModule, Mapping[str, PyTree[cx.Field]]):
    """A sampled graph-domain batch with topology and named graph axes.

    `GraphBatch` mirrors `PointsBatch` for finite graph supports while carrying the
    `GraphIR` topology required by graph operators and message-passing models. The
    selected graph entity axis is still represented as a normal Phydrax sampling
    axis in `structure`, so ordinary `DomainFunction` evaluation, broadcasting,
    and reductions can operate on node/edge/global fields.
    """

    points: Points
    structure: ProductStructure
    graph: GraphIR
    graph_label: str
    component_kind: GraphComponentKind

    def __init__(
        self,
        *,
        points: Points | Mapping[str, PyTree[cx.Field]],
        structure: ProductStructure,
        graph: GraphIR,
        graph_label: str,
        component_kind: GraphComponentKind,
    ):
        if structure.axis_names is None:
            raise ValueError(
                "GraphBatch requires a canonicalized ProductStructure (axis_names set)."
            )
        _validate_reserved_axes(
            frozendict(points), allowed_axes=frozenset(structure.axis_names)
        )
        self.points = frozendict(points)
        self.structure = structure
        self.graph = graph
        self.graph_label = str(graph_label)
        self.component_kind = component_kind

    def __getitem__(self, key: str) -> PyTree[cx.Field]:
        return self.points[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.points)

    def __len__(self) -> int:
        return len(self.points)


__all__ = [
    "GRAPH_ENTITY_INDEX_KEY",
    "GRAPH_GRAPH_INDEX_KEY",
    "GraphBatch",
]
