#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reconstruct layer provenance from authored strata and exact cell charts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from fractions import Fraction
from typing import NamedTuple, NoReturn, SupportsIndex

import numpy as np

from .._live_proofs import live_content_fingerprint, LiveProofLedger
from ..discretization import CellMesh
from ..discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
)
from ..discretization._cell_geometry_transition_types import (
    _ExactCellMaps,
    _geometry_map_expressions,
    _require_exact_reference_group,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_enclosure import (
    axes,
    compose_expression,
    Expression,
    expression_equal,
    ExpressionInput,
)
from ..discretization._nested_reference import (
    _chart_from_corners,
    _linear_expression,
    _root_affine_inverse,
    exact_reference_chart_controls,
    nested_cell_reference_maps,
    NestedCellReferenceMap,
)
from ..discretization._nested_reference_types import (
    _reference_cell_contains,
    evaluate_reference_chart_arguments,
    exact_reference_cell_vertices,
    freeze_reference_chart_arguments,
    ReferenceCellMapWitnesses,
    ReferenceCellPartition,
    validate_reference_cell_partition,
)
from ..discretization._reference_cell import reference_cell_topology
from ._association import GeometryAssociation
from ._layer_core_association import layer_source_associations
from ._layer_core_mapped import _block_order_regions, validate_layer_reference_columns
from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry
from ._mixed_adaptation import _affine_chart, _exact_child_to_parent_affine_map
from ._volume_entity_containment import ContainedEntity, reference_entity_containment
from .providers._native_sources import NativeLayerCoreSource


class LayerAssociationReplay(NamedTuple):
    """Fresh source rows and their exact endpoint-to-authored containment."""

    associations: tuple[GeometryAssociation, ...]
    containment: tuple[ContainedEntity, ...]
    layer_cell_ids: tuple[int, ...]
    core_cell_ids: tuple[int, ...]
    reference_maps: tuple[ReferenceCellMapWitnesses, ...]


class _ComputedLayerAssociationReplay:
    """One operation's fully proved source-to-endpoint layer replay."""

    __slots__ = ("source", "mesh", "geometry", "root", "replay")

    def __init__(
        self,
        source: NativeLayerCoreSource,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        root: LayerCoreRootGeometry | None,
        /,
        *,
        computed_root: ComputedLayerCoreRoot | None,
    ) -> None:
        self.source = source
        self.mesh = mesh
        self.geometry = geometry
        self.root = root
        self.replay = replay_layer_source_associations(
            source, mesh, geometry, root, computed_root=computed_root
        )

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("A computed layer association replay is immutable.")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("A computed layer association replay is immutable.")

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> NoReturn:
        raise TypeError("Computed layer association replay cannot be serialized.")

    def require(
        self,
        source: NativeLayerCoreSource,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        root: LayerCoreRootGeometry | None,
        /,
    ) -> LayerAssociationReplay:
        if (
            self.source is not source
            or self.mesh is not mesh
            or self.geometry is not geometry
            or self.root is not root
        ):
            raise ValueError("Computed layer replay binds different exact operands.")
        return self.replay


class _AuthoredRoot(NamedTuple):
    cell_id: int
    cell_kind: str
    offset: tuple[Fraction, ...]
    inverse: tuple[tuple[Fraction, ...], ...]
    bounds: tuple[tuple[Fraction, Fraction], ...]


class _MappedLayerRoots(NamedTuple):
    charts: dict[int, list[_AuthoredRoot]]
    physical_maps: _ExactCellMaps
    cells: dict[int, tuple[str, str]]


def _cell_vertices(mesh: CellMesh, /) -> dict[int, tuple[str, tuple[int, ...]]]:
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    return {
        int(identifier): (block.cell_kind, tuple(int(v) for v in vertex_ids[vertices]))
        for block in mesh.blocks
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True)
    }


def _maps(mesh: CellMesh, geometry: CellGeometrySpec, /) -> _ExactCellMaps:
    elements, routes, _ = geometry.resolve(mesh)
    return _geometry_map_expressions(mesh, geometry, (elements, routes))


def _witness(
    target_id: int,
    source_id: int,
    target_kind: str,
    source_kind: str,
    corners: tuple[tuple[Fraction, ...], ...],
    arguments: tuple[ExpressionInput, ...],
    /,
) -> ReferenceCellMapWitnesses:
    return ReferenceCellMapWitnesses(
        np.asarray([target_id], dtype=np.int64),
        np.asarray([source_id], dtype=np.int64),
        target_kind,
        source_kind,
        np.asarray(
            [[tuple(float(v) for v in point) for point in corners]], dtype=np.float64
        ),
        (corners,),
        freeze_reference_chart_arguments((arguments,)),
    )


def _identity_chart(
    kind: str, ordered: tuple[int, ...], authored: tuple[int, ...], /
) -> tuple[Expression, ...]:
    topology = reference_cell_topology(kind)
    if len(ordered) != len(authored) or set(ordered) != set(authored):
        raise ValueError("Layer root vertex identities do not match the authored cell.")
    points = tuple(
        tuple(Fraction(float(v)) for v in topology.vertices[authored.index(identifier)])
        for identifier in ordered
    )
    return _chart_from_corners(kind, points)


def _prepare_mapped_layer_roots(
    source: NativeLayerCoreSource,
    authority: NativeLayerCoreSource,
    restriction: CellGeometryRestrictionSource | None,
    /,
) -> _MappedLayerRoots | None:
    domain = source.mapped_domain
    if domain is None:
        if source.reference_source is not None:
            raise ValueError("A reference layer source requires its mapped domain.")
        return None
    if source.reference_source is None or restriction is None:
        raise ValueError(
            "Mapped layer replay requires its reference source and root maps."
        )
    validate_layer_reference_columns(source, authority, domain)
    if (
        restriction.source_geometry_id != domain.reference_geometry_id
        or restriction.source_topology_id != domain.reference_mesh.topology_id
    ):
        raise ValueError("Layer root ancestry does not bind the mapped domain.")
    authored_mesh = authority.layers.mesh
    regions = _block_order_regions(authored_mesh, authority.layer_regions)
    controls = exact_reference_chart_controls(
        domain, authored_mesh, CellGeometrySpec.affine(authored_mesh), regions
    )
    atlas: dict[int, list[_AuthoredRoot]] = {}
    for control in controls:
        offset, inverse = _root_affine_inverse(
            control.target_cell_kind, control.source_reference_vertices
        )
        matrix, translation = _exact_child_to_parent_affine_map(
            control.target_cell_kind, control.source_reference_vertices
        )
        if not all(
            expression_equal(dict(left) if isinstance(left, Mapping) else left, right)
            for left, right in zip(
                control.arguments, _affine_chart(matrix, translation), strict=True
            )
        ):
            raise ValueError("Authored layer root chart must be exactly affine.")
        bounds = tuple(
            (
                min(point[axis] for point in control.source_reference_vertices),
                max(point[axis] for point in control.source_reference_vertices),
            )
            for axis in range(3)
        )
        atlas.setdefault(control.source_cell_id, []).append(
            _AuthoredRoot(
                control.target_cell_id,
                control.target_cell_kind,
                offset,
                inverse,
                bounds,
            )
        )
    root_maps = _maps(domain.reference_mesh, domain.source_geometry)
    root_cells = {
        int(identifier): (block.cell_kind, block.name)
        for block in domain.reference_mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    }
    return _MappedLayerRoots(atlas, root_maps, root_cells)


def _mapped_layer_candidates(
    roots: _MappedLayerRoots,
    endpoint_maps: _ExactCellMaps,
    reference: NestedCellReferenceMap,
    target_id: int,
    target_kind: str,
    block_name: str,
    source_blocks: Mapping[str, str] | None,
    /,
) -> Iterator[ReferenceCellMapWitnesses]:
    if reference.arguments is None:
        raise ValueError("Layer root maps must use ordered local target rows.")
    root = roots.cells.get(reference.source_cell_id)
    if (
        root is None
        or root[0] != reference.source_cell_kind
        or (source_blocks is not None and source_blocks[block_name] != root[1])
    ):
        raise ValueError("Layer chart identifies the wrong mapped root cell.")
    root_witness = _witness(
        target_id,
        reference.source_cell_id,
        target_kind,
        reference.source_cell_kind,
        reference.source_reference_vertices,
        reference.arguments,
    )
    _require_exact_reference_group(
        roots.physical_maps,
        endpoint_maps,
        root_witness,
        exact_reference_cell_vertices(root_witness),
    )
    root_arguments = tuple(
        dict(value) if isinstance(value, Mapping) else value
        for value in reference.arguments
    )
    for authored_id, authored_kind, offset, inverse, bounds in roots.charts.get(
        reference.source_cell_id, ()
    ):
        if any(
            point[axis] < bounds[axis][0] or point[axis] > bounds[axis][1]
            for point in reference.source_reference_vertices
            for axis in range(3)
        ):
            continue
        arguments = tuple(
            _linear_expression(
                root_arguments,
                row,
                -sum(
                    (a * b for a, b in zip(row, offset, strict=True)),
                    Fraction(0),
                ),
                3,
            )
            for row in inverse
        )
        corners = _chart_corners(target_kind, arguments)
        if all(_reference_cell_contains(authored_kind, point) for point in corners):
            yield _affine_witness(
                target_id,
                authored_id,
                target_kind,
                authored_kind,
                corners,
                arguments,
            )


def _certify_layer_cover(
    groups: tuple[ReferenceCellMapWitnesses, ...],
    authored_cell_ids: set[int],
    original_maps: _ExactCellMaps | None,
    endpoint_maps: _ExactCellMaps | None,
    /,
) -> set[int]:
    seen: set[int] = set()
    covered: set[int] = set()
    for witness in groups:
        target_id = int(witness.target_cell_ids[0])
        if target_id in seen:
            raise ValueError("An endpoint cell has multiple authored layer owners.")
        seen.add(target_id)
        covered.add(int(witness.source_cell_ids[0]))
        if original_maps is not None:
            if endpoint_maps is None:
                raise ValueError("Direct layer map comparison requires endpoint maps.")
            _require_exact_reference_group(
                original_maps,
                endpoint_maps,
                witness,
                exact_reference_cell_vertices(witness),
            )
    if covered != authored_cell_ids:
        raise ValueError("Endpoint cells do not cover every authored layer cell.")
    validate_reference_cell_partition(ReferenceCellPartition(groups))
    return seen


def _replayed_layer_rows(
    authority: NativeLayerCoreSource,
    mesh: CellMesh,
    containment: tuple[ContainedEntity, ...],
    /,
) -> tuple[GeometryAssociation, ...]:
    originals = layer_source_associations(
        authority.layers,
        authority.layers.mesh,
        authority.layer_regions,
        authority.region_ids,
    )
    by_dimension = {dimension: record for dimension, record in enumerate(originals)}
    original_rows = {
        dimension: {
            int(identifier): row
            for row, identifier in enumerate(record.target_global_ids)
        }
        for dimension, record in by_dimension.items()
    }
    choices: dict[tuple[int, int], tuple[int, int]] = {}
    for relation in containment:
        if relation.fine_dimension > 2 or relation.coarse_dimension > 2:
            continue
        key = (relation.fine_dimension, relation.fine_id)
        value = (relation.coarse_dimension, relation.coarse_id)
        old = choices.get(key)
        if old is None or value[0] < old[0]:
            choices[key] = value
        elif value[0] == old[0] and value[1] != old[1]:
            raise ValueError(
                "An endpoint stratum has conflicting authored layer identities."
            )
    records: list[GeometryAssociation] = []
    for dimension in range(3):
        rows = sorted(
            (target_id, value)
            for (dim, target_id), value in choices.items()
            if dim == dimension
        )
        if not rows:
            continue
        ids = np.asarray([identifier for identifier, _ in rows], dtype=np.int64)
        selected = [
            (by_dimension[coarse_dim], original_rows[coarse_dim][coarse_id])
            for _, (coarse_dim, coarse_id) in rows
        ]
        first = selected[0][0]
        records.append(
            GeometryAssociation(
                first.association_kind,
                first.source_id,
                first.source_revision,
                mesh.entity_set(dimension).entity_set_id,
                ids,
                tuple(record.source_entity_ids[row] for record, row in selected),
                np.zeros(len(rows), dtype=np.float64),
                exact=True,
                source_roles=tuple(record.source_roles[row] for record, row in selected),
                source_occurrence_paths=tuple(
                    record.source_occurrence_paths[row] for record, row in selected
                ),
                source_region_sides=tuple(
                    record.source_region_sides[row] for record, row in selected
                ),
                orientations=np.zeros(len(rows), dtype=np.int8),
            )
        )
    return tuple(records)


def _conforming_root_rows(
    root: LayerCoreRootGeometry, /
) -> dict[int, NestedCellReferenceMap]:
    """Index the independently certified root-prism to authored-prism charts."""
    rows: dict[int, NestedCellReferenceMap] = {}
    for group in root.layer_root_partition.maps:
        for reference in nested_cell_reference_maps(group):
            if reference.arguments is None or reference.target_cell_id in rows:
                raise ValueError(
                    "Root layer partition has ambiguous or missing exact charts."
                )
            rows[reference.target_cell_id] = reference
    return rows


def _conforming_current_rows(
    root: LayerCoreRootGeometry, mesh: CellMesh, geometry: CellGeometrySpec, /
) -> Iterator[tuple[int, str, NestedCellReferenceMap]]:
    """Reprove every current cell against its original bounded root chart."""
    if mesh.mesh_id == root.mesh.mesh_id:
        from ..geometry._mesh_certificates import _same_tree_content

        if not _same_tree_content(mesh, root.mesh) or not _same_tree_content(
            geometry, root.geometry
        ):
            raise ValueError(
                "Conforming root endpoint changed its full mesh or coordinate map."
            )
        for block in mesh.blocks:
            vertices = tuple(
                tuple(Fraction(float(value)) for value in point)
                for point in reference_cell_topology(block.cell_kind).vertices
            )
            arguments = freeze_reference_chart_arguments(
                (tuple(axes(block.topological_dimension)),)
            )[0]
            for identifier in np.asarray(block.global_ids, dtype=np.int64):
                cell = int(identifier)
                yield (
                    cell,
                    block.cell_kind,
                    NestedCellReferenceMap(
                        cell, cell, block.cell_kind, block.cell_kind, vertices, arguments
                    ),
                )
        return
    restriction = geometry.restriction_source
    if restriction is None or (
        restriction.source_topology_id != root.mesh.topology_id
        or restriction.source_geometry_id != cell_geometry_id(root.geometry)
    ):
        raise ValueError(
            "Conforming layer geometry lacks its original-root restrictions."
        )
    block_maps = restriction.block_parent_reference_maps
    if set(block_maps) != {block.name for block in mesh.blocks}:
        raise ValueError("Conforming root maps must cover every endpoint block.")
    root_cells = {
        int(identifier): block.cell_kind
        for block in root.mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    }
    root_maps, endpoint_maps = _maps(root.mesh, root.geometry), _maps(mesh, geometry)
    for block in mesh.blocks:
        witness = block_maps[block.name]
        if witness is None:
            raise ValueError("Conforming root maps require exact cell charts.")
        nested = nested_cell_reference_maps(witness)
        if len(nested) != block.cell_count:
            raise ValueError("Conforming root charts do not cover an endpoint block.")
        for row, reference in enumerate(nested):
            if (
                reference.target_cell_id != row
                or reference.arguments is None
                or reference.target_cell_kind != block.cell_kind
                or root_cells.get(reference.source_cell_id) != reference.source_cell_kind
            ):
                raise ValueError(
                    "Conforming root chart identifies different endpoint cells."
                )
        # Keep the complete per-block exact chart bank and prove it once,
        # replacing only its local target row IDs with the actual global cell
        # IDs. This shares exact basis comparisons across all cells without
        # changing any chart, coefficient, or full-map premise.
        proof = witness._replace(
            target_cell_ids=np.asarray(block.global_ids, dtype=np.int64)
        )
        _require_exact_reference_group(
            root_maps, endpoint_maps, proof, exact_reference_cell_vertices(witness)
        )
        for row, reference in enumerate(nested):
            yield int(block.global_ids[row]), block.cell_kind, reference


def _conforming_layer_replay(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    root: LayerCoreRootGeometry,
    /,
) -> LayerAssociationReplay:
    """Compose current→root and root→authored exact prism maps."""
    authority = source.reference_source if source.mapped_domain is not None else source
    if authority is None:
        raise ValueError(
            "Mapped conforming replay requires the original reference source."
        )
    original_mesh = authority.layers.mesh
    original_cells = _cell_vertices(original_mesh)
    target_cells = _cell_vertices(mesh)
    root_rows = _conforming_root_rows(root)
    candidates: list[ReferenceCellMapWitnesses] = []
    if mesh.mesh_id == root.mesh.mesh_id:
        if cell_geometry_id(geometry) != cell_geometry_id(root.geometry):
            raise ValueError("Conforming root mesh carries a different coordinate map.")
        for reference in root_rows.values():
            assert reference.arguments is not None
            candidates.append(
                _witness(
                    reference.target_cell_id,
                    reference.source_cell_id,
                    reference.target_cell_kind,
                    reference.source_cell_kind,
                    reference.source_reference_vertices,
                    reference.arguments,
                )
            )
    else:
        for target_id, target_kind, current in _conforming_current_rows(
            root, mesh, geometry
        ):
            authored = root_rows.get(current.source_cell_id)
            if authored is None:
                continue
            if authored.arguments is None or current.arguments is None:
                raise ValueError(
                    "Conforming layer ancestry lacks exact reference expressions."
                )
            arguments = tuple(
                compose_expression(value, current.arguments)
                for value in authored.arguments
            )
            candidates.append(
                _affine_witness(
                    target_id,
                    authored.source_cell_id,
                    target_kind,
                    authored.source_cell_kind,
                    _chart_corners(target_kind, arguments),
                    arguments,
                )
            )
    groups = tuple(candidates)
    seen = _certify_layer_cover(
        groups,
        set(original_cells),
        # Root reproof established every root→authored full map. For a
        # descendant, _conforming_current_rows proved every complete
        # current→root map and the groups above are their exact composition.
        # Thus full endpoint→authored equality follows by composition; the
        # complete reference partition is still independently checked here.
        None,
        None,
    )
    containment = reference_entity_containment(mesh, original_mesh, groups)
    return LayerAssociationReplay(
        _replayed_layer_rows(authority, mesh, containment),
        containment,
        tuple(sorted(seen)),
        tuple(sorted(set(target_cells) - seen)),
        groups,
    )


# Layer replays bound to the exact live source, endpoint mesh and map, and
# root (see replay_layer_source_associations).
_REPLAYED_LAYERS: LiveProofLedger[LayerAssociationReplay] = LiveProofLedger()


def replay_layer_source_associations(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    root: LayerCoreRootGeometry | None,
    /,
    *,
    computed_root: ComputedLayerCoreRoot | None = None,
) -> LayerAssociationReplay:
    """Prove exact authored-cell coverage and rebuild incidence-only layer rows.

    Restriction charts are checked against complete physical coordinate maps.  The
    authored reference atlas and its inverse are independent of endpoint labels.

    The root and its operation-local proof are checked on every call. The
    replay itself is made once per exact live ``(source, mesh, geometry,
    root)``; any copy, rebuild or changed content replays in full.
    """
    if source.conforming_execution:
        if root is None:
            raise ValueError(
                "Conforming layer replay requires its original bounded root."
            )
        if computed_root is None:
            root.require_current(source, root.specification)
        elif type(computed_root) is ComputedLayerCoreRoot:
            computed_root.require(source, root.specification, root.mesh, root.geometry, root)
        else:
            raise TypeError("computed_root must be ComputedLayerCoreRoot or None.")
    else:
        if root is not None:
            raise ValueError("Fixed layer replay cannot borrow a conforming root authority.")
        if computed_root is not None:
            raise ValueError("Fixed layer replay cannot use a conforming root proof.")
    operands = (source, mesh, geometry, root)
    replayed = _REPLAYED_LAYERS.find(operands)
    if replayed is not None:
        return replayed
    fingerprint = live_content_fingerprint(operands)
    replay = (
        _conforming_layer_replay(source, mesh, geometry, root)
        if root is not None
        else _fixed_layer_replay(source, mesh, geometry)
    )
    _REPLAYED_LAYERS.record(operands, fingerprint, replay)
    return replay


def _fixed_layer_replay(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> LayerAssociationReplay:
    authority = source.reference_source if source.reference_source is not None else source
    authored_mesh = authority.layers.mesh
    authored = _cell_vertices(authored_mesh)
    target = _cell_vertices(mesh)
    authored_by_vertices = {
        (kind, frozenset(vertices)): cell for cell, (kind, vertices) in authored.items()
    }
    if len(authored_by_vertices) != len(authored):
        raise ValueError("Authored layer cells must have unique vertex incidence.")
    endpoint_maps = _maps(mesh, geometry)
    restriction = geometry.restriction_source
    domain = source.mapped_domain
    mapped_roots = _prepare_mapped_layer_roots(source, authority, restriction)

    original_geometry = CellGeometrySpec.affine(authored_mesh)
    original_maps = _maps(authored_mesh, original_geometry)
    candidates: list[ReferenceCellMapWitnesses] = []
    if restriction is not None:
        block_maps = restriction.block_parent_reference_maps
        block_vertices = dict(
            zip(restriction.block_names, restriction.parent_vertex_ids, strict=True)
        )
        source_blocks = restriction.block_source_blocks
        if set(block_maps) != {block.name for block in mesh.blocks}:
            raise ValueError("Layer root maps must cover every endpoint block.")
        for block in mesh.blocks:
            witness = block_maps[block.name]
            if witness is None:
                raise ValueError("Layer replay requires exact parent reference maps.")
            nested = nested_cell_reference_maps(witness)
            if len(nested) != len(block.global_ids):
                raise ValueError("Layer root maps must cover every endpoint cell.")
            for local_row, reference in enumerate(nested):
                if reference.target_cell_id != local_row or reference.arguments is None:
                    raise ValueError(
                        "Layer root maps must use ordered local target rows."
                    )
                target_id = int(block.global_ids[local_row])
                if reference.target_cell_kind != block.cell_kind:
                    raise ValueError("Layer root chart has the wrong target family.")
                if mapped_roots is not None:
                    candidates.extend(
                        _mapped_layer_candidates(
                            mapped_roots,
                            endpoint_maps,
                            reference,
                            target_id,
                            block.cell_kind,
                            block.name,
                            source_blocks,
                        )
                    )
                else:
                    parent_vertices = tuple(
                        int(v)
                        for v in np.asarray(block_vertices[block.name])[local_row]
                        if v >= 0
                    )
                    key = (reference.source_cell_kind, frozenset(parent_vertices))
                    authored_id = authored_by_vertices.get(key)
                    if authored_id is None:
                        continue
                    root_chart = _identity_chart(
                        reference.source_cell_kind,
                        parent_vertices,
                        authored[authored_id][1],
                    )
                    arguments = tuple(
                        compose_expression(expr, reference.arguments)
                        for expr in root_chart
                    )
                    corners = _chart_corners(block.cell_kind, arguments)
                    candidates.append(
                        _affine_witness(
                            target_id,
                            authored_id,
                            block.cell_kind,
                            reference.source_cell_kind,
                            corners,
                            arguments,
                        )
                    )
    else:
        if domain is not None:
            raise ValueError("Mapped replay requires retained reference ancestry.")
        for target_id, (kind, vertices) in target.items():
            authored_id = authored_by_vertices.get((kind, frozenset(vertices)))
            if authored_id is None:
                continue
            arguments = _identity_chart(kind, vertices, authored[authored_id][1])
            corners = _chart_corners(kind, arguments)
            candidates.append(
                _affine_witness(target_id, authored_id, kind, kind, corners, arguments)
            )

    groups = tuple(candidates)
    seen = _certify_layer_cover(
        groups,
        set(authored),
        original_maps if domain is None else None,
        endpoint_maps,
    )
    containment = reference_entity_containment(mesh, authored_mesh, groups)
    records = _replayed_layer_rows(authority, mesh, containment)
    return LayerAssociationReplay(
        records,
        containment,
        tuple(sorted(seen)),
        tuple(sorted(set(target) - seen)),
        groups,
    )


def _chart_corners(
    kind: str, arguments: tuple[ExpressionInput, ...], /
) -> tuple[tuple[Fraction, ...], ...]:
    points = tuple(
        tuple(Fraction(float(v)) for v in point)
        for point in reference_cell_topology(kind).vertices
    )
    values = tuple(
        evaluate_reference_chart_arguments(kind, arguments, point) for point in points
    )
    if any(not isinstance(value, Fraction) for point in values for value in point):
        raise ValueError("Layer reference corners must be exact rationals.")
    return tuple(
        tuple(value for value in point if isinstance(value, Fraction)) for point in values
    )


def _affine_witness(
    target_id: int,
    source_id: int,
    target_kind: str,
    source_kind: str,
    corners: tuple[tuple[Fraction, ...], ...],
    arguments: tuple[ExpressionInput, ...],
    /,
) -> ReferenceCellMapWitnesses:
    matrix, offset = _exact_child_to_parent_affine_map(target_kind, corners)
    affine = _affine_chart(matrix, offset)
    if len(arguments) != len(affine) or not all(
        expression_equal(dict(left) if isinstance(left, Mapping) else left, right)
        for left, right in zip(arguments, affine, strict=True)
    ):
        raise ValueError("Layer replay requires exact affine reference charts.")
    return _witness(target_id, source_id, target_kind, source_kind, corners, arguments)


__all__ = ["LayerAssociationReplay", "replay_layer_source_associations"]
