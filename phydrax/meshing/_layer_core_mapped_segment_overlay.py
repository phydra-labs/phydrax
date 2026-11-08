#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact original-reference charts for mapped source-edge face support.

These charts are deliberately independent of the physical coordinate map: a
straight edge in the original reference domain need not have a straight image.
"""

from __future__ import annotations

from fractions import Fraction
from typing import cast, TYPE_CHECKING

import numpy as np

from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_transition_types import (
    _require_exact_reference_group,
)
from ..discretization._coordinate_enclosure import (
    add,
    axes,
    compose_expression,
    constant,
    evaluate_expression,
    Expression,
    expression_equal,
    scale,
)
from ..discretization._nested_reference import (
    _chart_from_corners,
    nested_cell_reference_maps,
)
from ..discretization._nested_reference_types import exact_reference_cell_vertices
from ._layer_core_topology import prepare_identity
from ._layer_source_association_replay import _maps, _witness
from .providers._native_sources import NativeLayerCoreSource


if TYPE_CHECKING:
    from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry

type _Point = tuple[Fraction, ...]
type _ReferenceMaps = dict[int, tuple[Expression, ...]]
type _LocalMaps = dict[int, tuple[int, tuple[Expression, ...]]]


def mapped_local_reference_maps(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    root: LayerCoreRootGeometry | None = None,
    /,
    *,
    computed_root: ComputedLayerCoreRoot | None = None,
) -> _LocalMaps:
    """Prove complete current physical maps and return original root-local maps."""
    from ._layer_core_mapped import _canonical_expressions

    domain = source.mapped_domain
    if domain is None:
        raise ValueError("Mapped root-local charts require the original mapped domain.")
    root_proved = False
    if computed_root is not None:
        from ._layer_core_root_geometry import ComputedLayerCoreRoot

        if type(computed_root) is not ComputedLayerCoreRoot or root is None:
            raise ValueError("Mapped chart replay requires its actual computed root.")
        computed_root.require(source, root.specification, root.mesh, root.geometry, root)
        root_proved = (
            computed_root.mapped_chart_bank_equal
            and mesh is root.mesh
            and geometry is root.geometry
        )
    elif root is not None:
        root.require_current(source, root.specification)
    if mesh.mesh_id == domain.reference_mesh.mesh_id:
        from ..geometry._mesh_certificates import _same_tree_content

        if not _same_tree_content(mesh, domain.reference_mesh) or not _same_tree_content(
            geometry, domain.source_geometry
        ):
            raise ValueError(
                "The mapped root mesh or complete source coordinate map changed."
            )
        return {
            int(identifier): (int(identifier), tuple(axes(3)))
            for block in mesh.blocks
            for identifier in block.global_ids
        }
    if root is not None and not root_proved:
        from ._layer_source_association_replay import _conforming_current_rows

        original = (
            mapped_local_reference_maps(
                source, root.mesh, root.geometry, root, computed_root=computed_root
            )
            if computed_root is not None and computed_root.mapped_chart_bank_equal
            else mapped_local_reference_maps(source, root.mesh, root.geometry)
        )
        if mesh is root.mesh and geometry is root.geometry:
            return original
        composed: _LocalMaps = {}
        for cell, _, row in _conforming_current_rows(root, mesh, geometry):
            if row.arguments is None:
                raise ValueError(
                    "Mapped descendant chart lacks exact reference arguments."
                )
            arguments = _canonical_expressions(row.arguments)
            composed[cell] = (
                original[row.source_cell_id][0],
                tuple(
                    compose_expression(value, arguments)
                    for value in original[row.source_cell_id][1]
                ),
            )
        return composed
    restriction = geometry.restriction_source
    if restriction is None or (
        restriction.source_topology_id != domain.reference_mesh.topology_id
        or restriction.source_geometry_id != domain.reference_geometry_id
        or set(restriction.block_parent_reference_maps)
        != {block.name for block in mesh.blocks}
    ):
        raise ValueError(
            "Mapped root-local charts require complete original-root restrictions."
        )
    if not root_proved:
        root_maps, current_maps = (
            _maps(domain.reference_mesh, domain.source_geometry),
            _maps(mesh, geometry),
        )
    root_kinds = {
        int(identifier): block.cell_kind
        for block in domain.reference_mesh.blocks
        for identifier in block.global_ids
    }
    result: _LocalMaps = {}
    for block in mesh.blocks:
        bank = restriction.block_parent_reference_maps[block.name]
        if bank is None:
            raise ValueError("Mapped root-local charts omit a target block.")
        rows = nested_cell_reference_maps(bank)
        if len(rows) != block.cell_count:
            raise ValueError("Mapped root-local charts do not cover a target block.")
        for index, row in enumerate(rows):
            cell = int(block.global_ids[index])
            if (
                row.target_cell_id != index
                or row.target_cell_kind != block.cell_kind
                or row.arguments is None
                or root_kinds.get(row.source_cell_id) != row.source_cell_kind
                or row.source_cell_id not in domain.root_cell_ids
                or cell in result
            ):
                raise ValueError(
                    "Mapped root-local chart changes its source or target identity."
                )
            group = _witness(
                cell,
                row.source_cell_id,
                block.cell_kind,
                row.source_cell_kind,
                row.source_reference_vertices,
                row.arguments,
            )
            if not root_proved:
                _require_exact_reference_group(
                    root_maps, current_maps, group, exact_reference_cell_vertices(group)
                )
            result[cell] = (row.source_cell_id, _canonical_expressions(row.arguments))
    return result


def mapped_local_source_edges(
    source: NativeLayerCoreSource, /
) -> dict[int, dict[int, tuple[_Point, _Point]]]:
    """Resolve authored PLC edge endpoints in each incident original root cell."""
    from ..discretization._reference_cell import reference_cell_topology
    from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource

    domain, authored, query = (
        source.mapped_domain,
        source.reference_source,
        source.fidelity_source,
    )
    if (
        domain is None
        or authored is None
        or not isinstance(query, MappedReferenceBoundarySource)
    ):
        raise ValueError(
            "Mapped root-local edges require their full original source authority."
        )
    relation, root_ids = (
        query.authority.root_relation,
        domain.reference_domain_vertex_root_ids,
    )
    if relation is None or root_ids is None:
        raise ValueError("Mapped root-local edges require declared vertex-root ancestry.")
    mapping, _, _, _ = prepare_identity(
        authored.layers,
        authored.complex,
        authored.vertex_layer_ids,
        authored.cap_polygon_ids,
    )
    allowed = frozenset(int(value) for value in relation.vertex_root_global_ids)
    root_paths = dict(
        zip(domain.root_vertex_ids, domain.root_vertex_occurrence_paths, strict=True)
    )
    reference_vertices = np.asarray(domain.reference_domain.vertices)
    bindings: dict[int, list[tuple[int, int]]] = {}
    for edge, endpoints in enumerate(authored.complex.plc_edge_vertices):
        indices = tuple(int(mapping[int(value)]) for value in endpoints)
        if any(index < 0 or index >= len(root_ids) for index in indices):
            raise ValueError(
                "An original core edge lacks reference-domain root vertices."
            )
        pair = tuple(int(root_ids[index]) for index in indices)
        if pair[0] == pair[1] or not set(pair) <= allowed:
            raise ValueError(
                "An original core edge changes its declared full-source roots."
            )
        for vertex, index, root_id in zip(endpoints, indices, pair, strict=True):
            if not np.array_equal(
                np.asarray(authored.complex.vertices[int(vertex)]),
                reference_vertices[index],
            ) or tuple(authored.complex.vertex_occurrence_paths[int(vertex)]) != tuple(
                root_paths[root_id]
            ):
                raise ValueError(
                    "An original core edge endpoint changed its reference root or occurrence."
                )
        bindings.setdefault(pair[0], []).append((edge, pair[1]))
    global_vertices = np.asarray(domain.reference_mesh.vertex_global_ids, dtype=np.int64)
    result: dict[int, dict[int, tuple[_Point, _Point]]] = {}
    for block in domain.reference_mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        reference = tuple(
            tuple(Fraction.from_float(float(value)) for value in point)
            for point in topology.vertices
        )
        for identifier, rows in zip(block.global_ids, block.vertices, strict=True):
            vertices = tuple(int(value) for value in global_vertices[np.asarray(rows)])
            positions = {value: index for index, value in enumerate(vertices)}
            result[int(identifier)] = {
                edge: (reference[positions[first]], reference[positions[last]])
                for first in vertices
                for edge, last in bindings.get(first, ())
                if last in positions
            }
    return result


def affine_mapped_reference_triangle(
    reference_maps: _ReferenceMaps,
    cell: int,
    chart: tuple[_Point, ...],
    /,
) -> tuple[_Point, ...]:
    """Prove a target chart affine in its original root-cell coordinates."""
    if len(chart) != 3 or cell not in reference_maps:
        raise ValueError(
            "A mapped source facet requires an existing triangular target chart."
        )
    arguments = _chart_from_corners("triangle", chart)
    expressions = tuple(
        compose_expression(value, arguments) for value in reference_maps[cell]
    )
    corners = tuple(
        tuple(evaluate_expression(value, point) for value in expressions)
        for point in (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        )
    )
    if any(not isinstance(value, Fraction) for point in corners for value in point):
        raise ValueError("Mapped source facet coordinates must be exact rationals.")
    u, v = axes(2)
    for axis, expression in enumerate(expressions):
        first, second, third = (point[axis] for point in corners)
        expected = add(
            add(constant(first, 2), scale(u, second - first)), scale(v, third - first)
        )
        if not expression_equal(expression, expected):
            raise ValueError(
                "Nonaffine mapped root-local facet inversion is not supported."
            )
    return cast(tuple[_Point, ...], corners)
