#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Replay core boundary facet classes from a validated source chart cover."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import cast, TYPE_CHECKING

import numpy as np

from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._nested_reference_types import exact_reference_cell_vertices
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryFaceImageWitness,
)
from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
from ..geometry._source_fidelity_certificate import SourceFidelityWitness
from ..geometry._source_identity import source_entity_id
from ._association import (
    _source_boundary_image_edge_intervals,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
    source_boundary_edge_intervals,
    source_boundary_image_association_piece_rows,
    source_boundary_image_face_orientations,
    SourceBoundaryTargetPieceRow,
)
from ._association_routes import incidence_pairs, require_transfer_lineage
from ._layer_core_association import _core_source_identity, composite_region_identity
from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry
from ._layer_core_topology import combined_domain, faces, prepare_identity
from ._layer_source_association_replay import (
    _ComputedLayerAssociationReplay,
    LayerAssociationReplay,
    replay_layer_source_associations,
)
from ._lineage import MeshLineage
from ._result import CellMeshingResult
from ._topology_edit import entity_keys
from .providers._native_sources import NativeLayerCoreSource


if TYPE_CHECKING:
    from ._association import _LayerImageSupportPreparation
    from ._certification_inputs import MeshCertificationInputs


def _validated_layer_replay(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    root: LayerCoreRootGeometry | None,
    computed_root: ComputedLayerCoreRoot | None,
    computed_replay: _ComputedLayerAssociationReplay | None,
    /,
) -> LayerAssociationReplay:
    if computed_replay is None:
        return replay_layer_source_associations(
            source, mesh, geometry, root, computed_root=computed_root
        )
    if type(computed_replay) is not _ComputedLayerAssociationReplay:
        raise TypeError("computed_replay must be an operation-local layer replay.")
    return computed_replay.require(source, mesh, geometry, root)


def _image_support_for_audit(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    core_cell_ids: tuple[int, ...],
    root: LayerCoreRootGeometry | None,
    computed_root: ComputedLayerCoreRoot | None,
    preparations: list[_LayerImageSupportPreparation] | None,
    /,
) -> _LayerImageSupportPreparation:
    """Reuse only exact call-local image preparations for this audit."""
    from ._association import _prepare_layer_image_support

    if preparations is not None:
        for preparation in preparations:
            try:
                preparation.require(
                    source,
                    mesh,
                    geometry,
                    core_cell_ids,
                    root=root,
                    computed_root=computed_root,
                )
            except ValueError:
                continue
            return preparation
    preparation = _prepare_layer_image_support(
        source, mesh, geometry, core_cell_ids, root=root, computed_root=computed_root
    )
    preparation.require(
        source, mesh, geometry, core_cell_ids, root=root, computed_root=computed_root
    )
    if preparations is not None:
        preparations.append(preparation)
    return preparation


def _core_facet_sides(
    source: NativeLayerCoreSource, /
) -> tuple[tuple[str | None, str | None], ...]:
    """Bind each authored core facet to its actual composite material sides."""
    authored = source if source.reference_source is None else source.reference_source
    mapping, caps, polygons, _ = prepare_identity(
        authored.layers,
        authored.complex,
        authored.vertex_layer_ids,
        authored.cap_polygon_ids,
    )
    layer_faces = faces(
        authored.layers.mesh, np.asarray(authored.layer_regions, dtype=np.int64)
    )
    cap_regions: dict[int, int] = {}
    for polygon in np.asarray(caps, dtype=np.int64):
        facet = int(authored.complex.polygon_facets[int(polygon)])
        key = tuple(sorted(int(value) for value in mapping[polygons[int(polygon)]]))
        incidents = layer_faces.get(key, ())
        if len(incidents) != 1:
            raise ValueError("A core cap requires one exact authored layer side.")
        region = int(incidents[0].region)
        previous = cap_regions.setdefault(facet, region)
        if previous != region:
            raise ValueError("A core facet has conflicting layer-side materials.")
    core_map = np.asarray(authored.core_region_map, dtype=np.int64)
    identities = tuple(
        composite_region_identity(authored, index)
        for index in range(len(authored.region_ids))
    )
    output = []
    for facet in range(authored.complex.facet_regions.shape[0]):
        positive = int(authored.complex.facet_regions[facet, 0])
        negative = int(authored.complex.facet_regions[facet, 1])
        positive_region = -1 if positive < 0 else int(core_map[positive])
        negative_region = (
            cap_regions[facet]
            if facet in cap_regions
            else -1
            if negative < 0
            else int(core_map[negative])
        )
        output.append(
            (
                None if negative_region < 0 else identities[negative_region],
                None if positive_region < 0 else identities[positive_region],
            )
        )
    return tuple(output)


def core_facet_source_key(
    source: NativeLayerCoreSource, row: int, /
) -> tuple[str, tuple[str, ...]]:
    """Resolve an original core facet without inventing an exterior identity."""
    authored = source if source.reference_source is None else source.reference_source
    complex_ = authored.complex
    identifier = str(complex_.facet_ids[row])
    path = tuple(complex_.facet_occurrence_paths[row])
    if (
        authored.source_boundary_scope is not None
        and authored.core_facet_source_ids[row] is None
    ):
        positive, negative = (int(value) for value in complex_.facet_regions[row])
        if positive < 0 or positive != negative:
            raise ValueError(
                "An unbound core facet is not an authored same-material carrier."
            )
        return source_entity_id(
            authored.source_id,
            authored.source_revision,
            "face",
            identifier,
            occurrence_path=path,
        ), path
    return _core_source_identity(authored, 2, identifier, path)


def core_source_facet_keys(
    source: NativeLayerCoreSource, /
) -> frozenset[tuple[str, tuple[str, ...]]]:
    """Return occurrence-qualified source-bound original core facets."""
    authored = source if source.reference_source is None else source.reference_source
    return frozenset(
        core_facet_source_key(authored, row)
        for row in range(authored.complex.facet_regions.shape[0])
        if authored.source_boundary_scope is None
        or authored.core_facet_source_ids[row] is not None
        or (
            int(authored.complex.facet_regions[row, 0]) >= 0
            and int(authored.complex.facet_regions[row, 0])
            == int(authored.complex.facet_regions[row, 1])
        )
    )


def core_cap_source_keys(
    source: NativeLayerCoreSource, /
) -> frozenset[tuple[str, tuple[str, ...]]]:
    """Return occurrence-qualified authored source facets on the fixed layer cap."""
    authored = source if source.reference_source is None else source.reference_source
    result: set[tuple[str, tuple[str, ...]]] = set()
    for polygon in np.asarray(authored.cap_polygon_ids, dtype=np.int64):
        facet = int(authored.complex.polygon_facets[int(polygon)])
        if (
            authored.source_boundary_scope is not None
            and authored.core_facet_source_ids[facet] is None
        ):
            continue
        result.add(
            _core_source_identity(
                authored,
                2,
                str(authored.complex.facet_ids[facet]),
                tuple(authored.complex.facet_occurrence_paths[facet]),
            )
        )
    return frozenset(result)


def core_adjacent_face_ids(mesh: CellMesh, core_cells: set[int], /) -> frozenset[int]:
    """Identify target faces by actual core-cell incidence, not source label."""
    faces = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    cells = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    return frozenset(
        int(faces[face])
        for face, cell in incidence_pairs(mesh, 2, 3)
        if int(cells[cell]) in core_cells
    )


def replay_core_boundary_faces(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    eligible_face_ids: frozenset[int],
    /,
) -> tuple[GeometryAssociation, ...]:
    """Reconstruct eligible core exterior faces from the independent chart cover.

    Eligibility must be established separately from exact layer/core incidence;
    an exterior chart alone does not distinguish a layer wall from a core facet.
    """
    if isinstance(token.witness, SourceBoundaryFaceImageWitness):
        return replay_core_boundary_image_faces(
            source, mesh, geometry, token, eligible_face_ids
        )
    witness = _require_layer_chart_token(mesh, geometry, token, source)
    if witness.chart_dimension != 2 or witness.reference_dimension != 3:
        raise ValueError("Core boundary replay requires volume boundary face charts.")
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_keys = entity_keys(mesh, 2)
    face_by_key = {
        tuple(int(value) for value in key if value >= 0): int(identifier)
        for key, identifier in zip(face_keys, face_ids, strict=True)
    }
    if not eligible_face_ids <= set(face_by_key.values()):
        raise ValueError("Core boundary eligibility names an absent mesh face.")
    vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells: dict[int, tuple[str, tuple[int, ...]]] = {}
    for block in mesh.blocks:
        for identifier, rows in zip(
            np.asarray(block.global_ids, dtype=np.int64),
            np.asarray(block.vertices, dtype=np.int64),
            strict=True,
        ):
            cells[int(identifier)] = (
                block.cell_kind,
                tuple(int(value) for value in vertices[rows]),
            )
    authored = source if source.reference_source is None else source.reference_source
    complex_ = authored.complex
    facet_sides = _core_facet_sides(source)
    facets: dict[tuple[str, tuple[str, ...]], tuple[str | None, str | None]] = {}
    for row, (identifier, path) in enumerate(
        zip(complex_.facet_ids, complex_.facet_occurrence_paths, strict=True)
    ):
        if (
            authored.source_boundary_scope is not None
            and authored.core_facet_source_ids[row] is None
        ):
            continue
        identity, occurrence = _core_source_identity(
            authored, 2, str(identifier), tuple(path)
        )
        sides = facet_sides[row]
        key = (identity, occurrence)
        previous = facets.setdefault(key, sides)
        if previous != sides:
            raise ValueError(
                "Source facets conflict on occurrence-qualified region sides."
            )
    rows: dict[
        tuple[int, str, tuple[str, ...]],
        tuple[tuple[str | None, str | None], int],
    ] = {}
    piece_rows: dict[
        tuple[int, str, tuple[str, ...]],
        list[tuple[int, int, tuple[int, ...], tuple[tuple[tuple[int, int], ...], ...]]],
    ] = {}
    for index, (cell_id, face_id, local_face) in enumerate(
        zip(
            witness.cell_global_ids,
            witness.entity_global_ids,
            witness.local_entity_indices,
            strict=True,
        )
    ):
        face_id = int(face_id)
        if face_id not in eligible_face_ids:
            continue
        key = (witness.source_entity_ids[index], witness.occurrence_paths[index])
        if key not in facets:
            raise ValueError("An eligible core face is not an authored source facet.")
        cell = cells.get(int(cell_id))
        if cell is None:
            raise ValueError("A source chart names an absent owner cell.")
        kind, global_vertices = cell
        local_faces = reference_cell_topology(kind).entities[2]
        local_face = int(local_face)
        if not 0 <= local_face < len(local_faces):
            raise ValueError("A source chart names an absent local face.")
        face_vertices = tuple(global_vertices[value] for value in local_faces[local_face])
        if len(face_vertices) != 3:
            raise ValueError(
                "Core source boundary replay requires triangular core facets."
            )
        if face_by_key.get(tuple(sorted(face_vertices))) != face_id:
            raise ValueError("A source chart face does not match its owner incidence.")
        inversions = sum(
            first > second
            for position, first in enumerate(face_vertices)
            for second in face_vertices[position + 1 :]
        )
        orientation = int(witness.orientations[index]) * (-1 if inversions % 2 else 1)
        row_key = (face_id, key[0], key[1])
        value = (facets[key], orientation)
        previous = rows.setdefault(row_key, value)
        if previous != value:
            raise ValueError(
                "Core chart pieces disagree on a source facet's side or orientation."
            )
        piece_rows.setdefault(row_key, []).append(
            (
                int(cell_id),
                local_face,
                (),
                witness.target_reference_rational_tokens[index],
            )
        )
    if {key[0] for key in rows} != set(eligible_face_ids):
        raise ValueError("Source charts do not cover every eligible core boundary face.")
    grouped: dict[
        tuple[str, tuple[str, ...], tuple[str | None, str | None]],
        list[
            tuple[
                int,
                int,
                tuple[
                    tuple[
                        int, int, tuple[int, ...], tuple[tuple[tuple[int, int], ...], ...]
                    ],
                    ...,
                ],
            ]
        ],
    ] = {}
    for (face_id, identity, occurrence), (sides, orientation) in rows.items():
        pieces = tuple(sorted(piece_rows[(face_id, identity, occurrence)]))
        grouped.setdefault((identity, occurrence, sides), []).append(
            (face_id, orientation, pieces)
        )
    output = []
    for (identity, occurrence, sides), values in sorted(grouped.items(), key=repr):
        ordered = sorted(values)
        identifiers = np.asarray([value[0] for value in ordered], dtype=np.int64)
        output.append(
            GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                source.source_id,
                source.source_revision,
                mesh.entity_set(2).entity_set_id,
                identifiers,
                (identity,) * identifiers.size,
                np.zeros(identifiers.shape, dtype=np.float64),
                exact=True,
                source_roles=(GeometrySourceEntityRole.FACET,) * identifiers.size,
                source_occurrence_paths=(occurrence,) * identifiers.size,
                source_region_sides=(sides,) * identifiers.size,
                orientations=np.asarray([value[1] for value in ordered], dtype=np.int8),
                target_reference_pieces=tuple(value[2] for value in ordered),
                source_boundary_witness_id=witness.witness_id,
                provenance=GeometryAssociationProvenance.CLASSIFICATION,
            )
        )
    return tuple(output)


def replay_core_boundary_image_faces(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    eligible_face_ids: frozenset[int],
    /,
) -> tuple[GeometryAssociation, ...]:
    """Bind exact source facets to implicit pullbacks of their physical images."""
    witness = _require_layer_image_token(mesh, geometry, token, source)
    if not eligible_face_ids <= set(
        np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64).tolist()
    ):
        raise ValueError("Image source eligibility names an absent target face.")
    authored = source if source.reference_source is None else source.reference_source
    sides_by_facet = _core_facet_sides(source)
    facets: dict[tuple[str, tuple[str, ...]], tuple[str | None, str | None]] = {}
    for row, (identifier, path) in enumerate(
        zip(
            authored.complex.facet_ids,
            authored.complex.facet_occurrence_paths,
            strict=True,
        )
    ):
        if (
            authored.source_boundary_scope is not None
            and authored.core_facet_source_ids[row] is None
        ):
            continue
        key = _core_source_identity(authored, 2, str(identifier), tuple(path))
        previous = facets.setdefault(key, sides_by_facet[row])
        if previous != sides_by_facet[row]:
            raise ValueError("Original image source facets conflict on material sides.")
    pieces = source_boundary_image_association_piece_rows(witness)
    orientations = source_boundary_image_face_orientations(mesh, geometry, witness)
    rows: dict[
        tuple[int, str, tuple[str, ...]], tuple[tuple[str | None, str | None], int]
    ] = {}
    for face, identity, path, sides in zip(
        witness.entity_global_ids,
        witness.source_entity_ids,
        witness.occurrence_paths,
        witness.source_region_sides,
        strict=True,
    ):
        face_id = int(face)
        if face_id not in eligible_face_ids:
            continue
        key = (identity, tuple(path))
        if key not in facets or facets[key] != tuple(sides):
            raise ValueError(
                "Image source facet changes its authored occurrence or material sides."
            )
        row_key = (face_id, identity, tuple(path))
        value = (tuple(sides), orientations[row_key])
        previous = rows.setdefault(row_key, value)
        if previous != value:
            raise ValueError(
                "Image source pieces disagree on material side or orientation."
            )
    grouped: dict[
        tuple[str, tuple[str, ...], tuple[str | None, str | None]],
        list[tuple[int, int, SourceBoundaryTargetPieceRow]],
    ] = {}
    for (face_id, identity, path), (sides, orientation) in rows.items():
        grouped.setdefault((identity, path, sides), []).append(
            (face_id, orientation, pieces[(face_id, identity, path)])
        )
    output: list[GeometryAssociation] = []
    for (identity, path, sides), values in sorted(grouped.items(), key=repr):
        ordered = sorted(values, key=lambda value: value[0])
        identifiers = np.asarray([value[0] for value in ordered], dtype=np.int64)
        output.append(
            GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                source.source_id,
                source.source_revision,
                mesh.entity_set(2).entity_set_id,
                identifiers,
                (identity,) * len(ordered),
                np.zeros(identifiers.shape, dtype=np.float64),
                exact=True,
                source_roles=(GeometrySourceEntityRole.FACET,) * len(ordered),
                source_occurrence_paths=(path,) * len(ordered),
                source_region_sides=(sides,) * len(ordered),
                orientations=np.asarray([value[1] for value in ordered], dtype=np.int8),
                target_reference_pieces=tuple(value[2] for value in ordered),
                source_boundary_witness_id=witness.witness_id,
                provenance=GeometryAssociationProvenance.CLASSIFICATION,
            )
        )
    return tuple(output)


def replay_composite_regions(
    authority: NativeLayerCoreSource,
    source: CellMeshingResult,
    target: CellMesh,
    lineage: MeshLineage,
    /,
) -> tuple[
    GeometryAssociation, GeometryAssociation, tuple[ValidatedSourceFidelityWitness, ...]
]:
    """Regenerate source and target cell classes from certified material coverage.

    The retained coverage domain is independently rebuilt from the authored fixed
    layer/core source before its per-cell material assignment can be consumed.
    """
    from ._adaptation import _transition_regions

    certification = source.certification
    if certification is None or certification.coverage is None:
        raise ValueError("Composite region replay requires accepted material coverage.")
    validated_source = certification.require_current(source.mesh, source.geometry)
    require_transfer_lineage(source.mesh, lineage, target, dimensions=(3,))
    authored = (
        authority if authority.reference_source is None else authority.reference_source
    )
    mapping, caps, polygons, points = prepare_identity(
        authored.layers,
        authored.complex,
        authored.vertex_layer_ids,
        authored.cap_polygon_ids,
    )
    domain, _ = combined_domain(
        authored.layers,
        authored.complex,
        np.asarray(authored.layer_regions, dtype=np.int64),
        mapping,
        caps,
        polygons,
        points,
        authored.source_id,
        authored.region_ids,
        np.asarray(authored.core_region_map, dtype=np.int64),
        authored.region_occurrence_paths,
    )
    inputs = certification.inputs
    mapped = authority.mapped_domain
    expected_domain_id = domain.domain_id if mapped is None else mapped.domain_id
    if mapped is not None and mapped.reference_domain.domain_id != domain.domain_id:
        raise ValueError(
            "Mapped roots do not bind the authored reference layer/core domain."
        )
    if inputs.domain is None or inputs.domain.domain_id != expected_domain_id:
        raise ValueError(
            "Material coverage does not bind the authored layer/core domain."
        )
    if tuple(domain.region_ids) != tuple(authored.region_ids):
        raise ValueError("Material coverage changed the composite region namespace.")
    if inputs.cell_regions is None:
        raise ValueError("Material coverage lacks certified source cell assignments.")
    target_regions = _transition_regions(inputs, target, lineage)
    if target_regions is None:
        raise ValueError("Material ancestry did not produce target region assignments.")

    def table(
        mesh: CellMesh, ids: np.ndarray, regions: np.ndarray
    ) -> GeometryAssociation:
        if (
            ids.shape != regions.shape
            or np.unique(ids).size != ids.size
            or np.any(regions < 0)
            or np.any(regions >= len(authored.region_ids))
            or set(ids.tolist())
            != set(np.asarray(mesh.entity_set(3).entity_ids).tolist())
        ):
            raise ValueError("Certified composite region rows do not cover the endpoint.")
        order = np.argsort(ids, kind="stable")
        ids, regions = ids[order], regions[order]
        return GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            authored.source_id,
            authored.source_revision,
            mesh.entity_set(3).entity_set_id,
            ids,
            tuple(composite_region_identity(authored, int(value)) for value in regions),
            np.zeros(ids.shape, dtype=np.float64),
            exact=True,
            source_roles=(GeometrySourceEntityRole.REGION,) * ids.size,
            source_occurrence_paths=tuple(
                authored.region_occurrence_paths[int(value)] for value in regions
            ),
            orientations=np.zeros(ids.shape, dtype=np.int8),
            provenance=GeometryAssociationProvenance.CLASSIFICATION,
        )

    source_ids = np.asarray(inputs.cell_global_ids, dtype=np.int64)
    target_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in target.blocks)
    )
    return (
        table(source.mesh, source_ids, np.asarray(inputs.cell_regions, dtype=np.int64)),
        table(target, target_ids, np.asarray(target_regions, dtype=np.int64)),
        validated_source,
    )


def replay_core_interface_faces(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    replay: LayerAssociationReplay,
    /,
    *,
    require_coverage: bool,
) -> GeometryAssociation | None:
    """Reconstruct declared core-cap classes from exact authored layer incidence."""
    from ._adaptation import _simplex_chart_weights

    authored = source if source.reference_source is None else source.reference_source
    layer_mesh = authored.layers.mesh
    mapping, caps, polygons, _ = prepare_identity(
        authored.layers,
        authored.complex,
        authored.vertex_layer_ids,
        authored.cap_polygon_ids,
    )
    layer_vertices = np.asarray(layer_mesh.vertex_global_ids, dtype=np.int64)
    layer_face_keys = entity_keys(layer_mesh, 2)
    layer_face_ids = np.asarray(layer_mesh.entity_set(2).entity_ids, dtype=np.int64)
    layer_face_by_key = {
        tuple(int(value) for value in key if value >= 0): int(identifier)
        for key, identifier in zip(layer_face_keys, layer_face_ids, strict=True)
    }
    facet_sides = _core_facet_sides(source)
    cap_faces: dict[
        int, tuple[str, tuple[str, ...], tuple[str | None, str | None], int]
    ] = {}
    for polygon_id in np.asarray(caps, dtype=np.int64):
        facet = int(authored.complex.polygon_facets[int(polygon_id)])
        if (
            authored.source_boundary_scope is not None
            and authored.core_facet_source_ids[facet] is None
        ):
            continue
        vertex_rows = np.asarray(mapping[polygons[int(polygon_id)]], dtype=np.int64)
        if (
            vertex_rows.shape != (3,)
            or np.any(vertex_rows < 0)
            or np.any(vertex_rows >= layer_vertices.size)
        ):
            raise ValueError(
                "A source-bound core cap must be an authored layer triangle."
            )
        oriented = tuple(int(value) for value in layer_vertices[vertex_rows])
        coarse_id = layer_face_by_key.get(tuple(sorted(oriented)))
        if coarse_id is None:
            raise ValueError(
                "A source-bound core cap is absent from the layer incidence."
            )
        key = _core_source_identity(
            authored,
            2,
            str(authored.complex.facet_ids[facet]),
            tuple(authored.complex.facet_occurrence_paths[facet]),
        )
        sides = facet_sides[facet]
        inversions = sum(
            first > second
            for index, first in enumerate(oriented)
            for second in oriented[index + 1 :]
        )
        record = (key[0], key[1], sides, -1 if inversions % 2 else 1)
        previous = cap_faces.setdefault(coarse_id, record)
        if previous != record:
            raise ValueError("A layer cap has conflicting source facet ancestry.")
    if not cap_faces:
        return None
    fine_vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    fine_cells = {
        int(identifier): (
            block.cell_kind,
            tuple(int(value) for value in fine_vertices[rows]),
        )
        for block in mesh.blocks
        for identifier, rows in zip(block.global_ids, block.vertices, strict=True)
    }
    coarse_cells = {
        int(identifier): (
            block.cell_kind,
            tuple(int(value) for value in layer_vertices[rows]),
        )
        for block in layer_mesh.blocks
        for identifier, rows in zip(block.global_ids, block.vertices, strict=True)
    }
    fine_face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    fine_face_keys = entity_keys(mesh, 2)
    fine_keys = {
        int(identifier): tuple(int(value) for value in key if value >= 0)
        for key, identifier in zip(fine_face_keys, fine_face_ids, strict=True)
    }
    coarse_keys = {
        int(identifier): tuple(int(value) for value in key if value >= 0)
        for key, identifier in zip(layer_face_keys, layer_face_ids, strict=True)
    }
    maps = {
        (int(item.target_cell_ids[0]), int(item.source_cell_ids[0])): item
        for item in replay.reference_maps
    }
    rows: dict[int, tuple[str, tuple[str, ...], tuple[str | None, str | None], int]] = {}
    coverage: dict[tuple[int, int], Fraction] = {}
    for item in replay.containment:
        if item.fine_dimension != 2 or item.coarse_dimension != 2:
            continue
        if item.coarse_id not in cap_faces:
            continue
        _, fine_cell_vertices = fine_cells[item.fine_cell_id]
        coarse_kind, coarse_cell_vertices = coarse_cells[item.coarse_cell_id]
        fine_key, coarse_key = fine_keys[item.fine_id], coarse_keys[item.coarse_id]
        if len(fine_key) != 3 or len(coarse_key) != 3:
            raise ValueError("Core cap replay requires triangular face restrictions.")
        witness = maps[(item.fine_cell_id, item.coarse_cell_id)]
        exact = exact_reference_cell_vertices(witness)[0]
        coarse_topology = reference_cell_topology(coarse_kind)
        coarse_corners = tuple(
            tuple(
                Fraction(float(value))
                for value in coarse_topology.vertices[coarse_cell_vertices.index(vertex)]
            )
            for vertex in coarse_key
        )
        weights = tuple(
            _simplex_chart_weights(
                exact[fine_cell_vertices.index(vertex)], coarse_corners, 2
            )
            for vertex in fine_key
        )
        if any(value is None for value in weights):
            raise ValueError("A cap face does not lie in its exact authored face.")
        first, second, third = weights
        assert first is not None and second is not None and third is not None
        determinant = (second[1] - first[1]) * (third[2] - first[2]) - (
            second[2] - first[2]
        ) * (third[1] - first[1])
        if not determinant:
            raise ValueError("A cap face has a degenerate exact reference chart.")
        coverage_key = (item.coarse_id, item.fine_id)
        area = abs(determinant)
        previous_area = coverage.setdefault(coverage_key, area)
        if previous_area != area:
            raise ValueError("A cap restriction has conflicting exact chart areas.")
        identity, path, sides, orientation = cap_faces[item.coarse_id]
        value = (identity, path, sides, orientation * (1 if determinant > 0 else -1))
        previous = rows.setdefault(item.fine_id, value)
        if previous != value:
            raise ValueError("A core cap has conflicting exact source orientations.")
    if require_coverage:
        for coarse_id in cap_faces:
            area = sum(
                (value for (owner, _), value in coverage.items() if owner == coarse_id),
                Fraction(0),
            )
            if area != 1:
                raise ValueError(
                    "Target facets do not exactly cover an authored cap facet."
                )
    if not rows:
        return None
    identifiers = np.asarray(sorted(rows), dtype=np.int64)
    values = tuple(rows[int(identifier)] for identifier in identifiers)
    return GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        authored.source_id,
        authored.source_revision,
        mesh.entity_set(2).entity_set_id,
        identifiers,
        tuple(value[0] for value in values),
        np.zeros(identifiers.shape, dtype=np.float64),
        exact=True,
        source_roles=(GeometrySourceEntityRole.FACET,) * len(values),
        source_occurrence_paths=tuple(value[1] for value in values),
        source_region_sides=tuple(value[2] for value in values),
        orientations=np.asarray([value[3] for value in values], dtype=np.int8),
        provenance=GeometryAssociationProvenance.CLASSIFICATION,
    )


def replay_layer_region_vertices(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    replay: LayerAssociationReplay,
    region_table: GeometryAssociation,
    core_records: tuple[GeometryAssociation, ...],
    /,
) -> GeometryAssociation | None:
    """Reconstruct generated layer vertices' unique composite material classes."""
    authored = source if source.reference_source is None else source.reference_source
    layer = authored.layers
    layer_vertex_ids = np.asarray(layer.mesh.vertex_global_ids, dtype=np.int64)
    wall_vertex_ids = np.asarray(layer.source_wall.vertex_global_ids, dtype=np.int64)
    wall_map = np.asarray(layer.wall_vertices, dtype=np.int64)
    if wall_map.shape != wall_vertex_ids.shape:
        raise ValueError("Authored wall vertex ancestry has incompatible extents.")
    wall_to_layer = {
        int(wall_id): int(layer_vertex_ids[int(row)])
        for wall_id, row in zip(wall_vertex_ids, wall_map, strict=True)
        if row >= 0
    }
    wall_entities: dict[int, set[int]] = {}
    for dimension in range(3):
        wall_keys = (
            entity_keys(layer.source_wall, dimension)
            if dimension < 2
            else tuple(
                wall_vertex_ids[np.asarray(rows, dtype=np.int64)]
                for block in layer.source_wall.blocks
                for rows in block.vertices
            )
        )
        wanted = {
            tuple(sorted(wall_to_layer[int(value)] for value in key if value >= 0))
            for key in wall_keys
            if all(int(value) in wall_to_layer for value in key if value >= 0)
        }
        wall_entities[dimension] = {
            int(identifier)
            for key, identifier in zip(
                entity_keys(layer.mesh, dimension),
                layer.mesh.entity_set(dimension).entity_ids,
                strict=True,
            )
            if tuple(int(value) for value in key if value >= 0) in wanted
        }
    layer_vertices = {
        item.fine_id for item in replay.containment if item.fine_dimension == 0
    }
    wall_vertices = {
        item.fine_id
        for item in replay.containment
        if item.fine_dimension == 0
        and item.coarse_dimension < 3
        and item.coarse_id in wall_entities[item.coarse_dimension]
    }
    covered = {
        int(identifier)
        for record in core_records
        if record.target_entity_set_id == mesh.entity_set(0).entity_set_id
        and (record.source_id, record.source_revision)
        == (authored.source_id, authored.source_revision)
        for identifier in record.target_global_ids
    }
    vertex_ids = np.asarray(mesh.entity_set(0).entity_ids, dtype=np.int64)
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    regions = {
        int(identifier): (identity, path)
        for identifier, identity, path in zip(
            region_table.target_global_ids,
            region_table.source_entity_ids,
            region_table.source_occurrence_paths,
            strict=True,
        )
    }
    if set(regions) != set(cell_ids.tolist()):
        raise ValueError("Composite material classes do not cover endpoint cells.")
    adjacent: dict[int, set[tuple[str, tuple[str, ...]]]] = {}
    for vertex_row, cell_row in incidence_pairs(mesh, 0, 3):
        vertex = int(vertex_ids[vertex_row])
        if (
            vertex in layer_vertices
            and vertex not in wall_vertices
            and vertex not in covered
        ):
            adjacent.setdefault(vertex, set()).add(regions[int(cell_ids[cell_row])])
    selected = tuple(
        (identifier, next(iter(values)))
        for identifier, values in sorted(adjacent.items())
        if len(values) == 1
    )
    if not selected:
        return None
    identifiers = np.asarray([item[0] for item in selected], dtype=np.int64)
    return GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        authored.source_id,
        authored.source_revision,
        mesh.entity_set(0).entity_set_id,
        identifiers,
        tuple(item[1][0] for item in selected),
        np.zeros(identifiers.shape, dtype=np.float64),
        exact=True,
        source_roles=(GeometrySourceEntityRole.REGION,) * identifiers.size,
        source_occurrence_paths=tuple(item[1][1] for item in selected),
        orientations=np.zeros(identifiers.shape, dtype=np.int8),
        provenance=GeometryAssociationProvenance.CLASSIFICATION,
    )


def replay_core_facet_edges(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    witness: SourceBoundaryChartWitness | SourceBoundaryFaceImageWitness,
    replay: LayerAssociationReplay,
    face_tables: tuple[GeometryAssociation, ...],
    /,
    *,
    preparation: _LayerImageSupportPreparation,
    include_cap: bool,
    cap_coverage_proved: bool,
    cap_owner: SourceBoundaryChartWitness | None = None,
) -> tuple[GeometryAssociation, ...]:
    """Classify core edges on exact source-facet chart support."""
    preparation.require(
        source,
        mesh,
        geometry,
        replay.core_cell_ids,
        root=preparation.root,
        computed_root=preparation.computed_root,
    )
    if source.mapped_domain is not None and cap_coverage_proved and cap_owner is None:
        raise ValueError("Mapped facet-edge replay requires its proved cap owner.")
    cap_keys = (
        core_cap_source_keys(source)
        if source.mapped_domain is not None and cap_coverage_proved
        else frozenset()
    )
    authored = source if source.reference_source is None else source.reference_source
    facet_sides: dict[tuple[str, tuple[str, ...]], tuple[str | None, str | None]] = {}
    for table in face_tables:
        for identity, path, sides in zip(
            table.source_entity_ids,
            table.source_occurrence_paths,
            table.source_region_sides,
            strict=True,
        ):
            key = (identity, path)
            previous = facet_sides.setdefault(key, sides)
            if previous != sides:
                raise ValueError("One source facet has conflicting material sides.")
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    core_cells = set(replay.core_cell_ids)
    core_edges = {
        int(edge_ids[edge])
        for edge, cell in incidence_pairs(mesh, 1, 3)
        if int(cell_ids[cell]) in core_cells
    }
    edge_by_key = {
        tuple(int(value) for value in key if value >= 0): int(identifier)
        for key, identifier in zip(entity_keys(mesh, 1), edge_ids, strict=True)
    }
    face_by_id = {
        int(identifier): tuple(int(value) for value in key if value >= 0)
        for key, identifier in zip(
            entity_keys(mesh, 2), mesh.entity_set(2).entity_ids, strict=True
        )
    }
    cap_records: dict[
        tuple[int, str, tuple[str, ...]], tuple[str | None, str | None]
    ] = {}
    cap = (
        replay_core_interface_faces(
            source, mesh, replay, require_coverage=not cap_coverage_proved
        )
        if include_cap
        else None
    )
    if cap is not None:
        for face, identity, path, sides in zip(
            cap.target_global_ids,
            cap.source_entity_ids,
            cap.source_occurrence_paths,
            cap.source_region_sides,
            strict=True,
        ):
            vertices = face_by_id[int(face)]
            for edge_key in combinations(vertices, 2):
                edge_id = edge_by_key.get(edge_key)
                if edge_id is None or edge_id not in core_edges:
                    raise ValueError(
                        "A proved core-cap facet lacks its core edge incidence."
                    )
                key = (edge_id, identity, path)
                previous = cap_records.setdefault(key, sides)
                if previous != sides:
                    raise ValueError("A core-cap edge has conflicting source sides.")
    supports = (
        _source_boundary_image_edge_intervals(mesh, geometry, witness, preparation)
        if isinstance(witness, SourceBoundaryFaceImageWitness)
        else source_boundary_edge_intervals(mesh, witness)
    )
    groups: dict[
        tuple[str, tuple[str, ...], tuple[str | None, str | None], int, str | None],
        list[tuple[int, tuple[Fraction, Fraction]]],
    ] = {}
    for (edge_id, identity, path), sides in cap_records.items():
        groups.setdefault((identity, path, sides, 0, None), []).append(
            (edge_id, (Fraction(0), Fraction(1)))
        )
    for (edge_id, identity, path), intervals in supports.items():
        sides = facet_sides.get((identity, path))
        if (identity, path) in cap_keys and witness is not cap_owner:
            continue
        if (
            edge_id not in core_edges
            or sides is None
            or (edge_id, identity, path) in cap_records
        ):
            continue
        for ordinal, interval in enumerate(intervals):
            groups.setdefault(
                (identity, path, sides, ordinal, witness.witness_id), []
            ).append((edge_id, interval))
    output = []
    for (identity, path, sides, _, witness_id), values in sorted(
        groups.items(), key=repr
    ):
        ordered = sorted(values)
        ids = np.asarray([value[0] for value in ordered], dtype=np.int64)
        output.append(
            GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                authored.source_id,
                authored.source_revision,
                mesh.entity_set(1).entity_set_id,
                ids,
                (identity,) * ids.size,
                np.zeros(ids.shape, dtype=np.float64),
                exact=True,
                source_roles=(GeometrySourceEntityRole.FACET,) * ids.size,
                source_occurrence_paths=(path,) * ids.size,
                source_region_sides=(sides,) * ids.size,
                target_parameter_intervals=tuple(value[1] for value in ordered),
                source_boundary_witness_id=witness_id,
                orientations=np.zeros(ids.shape, dtype=np.int8),
                provenance=GeometryAssociationProvenance.CLASSIFICATION,
            )
        )
    return tuple(output)


def is_internal_core_chart(token: ValidatedSourceFidelityWitness, /) -> bool:
    """Distinguish source PLC carrier charts from physical boundary charts."""
    from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
    from ._layer_core_facet_authority import MappedCoreFacetAuthority

    return isinstance(token.source, MappedReferenceBoundarySource) and isinstance(
        token.source.authority, MappedCoreFacetAuthority
    )


def _require_layer_fidelity_token(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    source: NativeLayerCoreSource,
    /,
) -> SourceFidelityWitness:
    if not isinstance(token, ValidatedSourceFidelityWitness):
        raise TypeError(
            "Partial layer/core association audit requires a validated source token."
        )
    if type(source) is not NativeLayerCoreSource:
        raise TypeError(
            "Partial layer/core association audit requires its original source."
        )
    source.validate_restored()
    from ..geometry._domain_boundary import MeshingDomainBoundarySource
    from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
    from ..geometry._meshing_domain import MeshingDomain

    query = token.source
    if not token.binds(mesh, geometry, query):
        raise ValueError("Partial association proof binds different source operands.")
    if (query.source_id, query.source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        raise ValueError("Partial association query binds another source revision.")
    if query is not source.fidelity_source:
        if isinstance(query, MeshingDomainBoundarySource):
            if not isinstance(source.source_domain, MeshingDomain) or (
                query.domain.domain_id != source.source_domain.domain_id
            ):
                raise ValueError(
                    "Scoped chart query binds a different authored source domain."
                )
        elif isinstance(query, MappedReferenceBoundarySource):
            from ._layer_core_facet_authority import MappedCoreFacetAuthority

            if isinstance(query.authority, MappedCoreFacetAuthority):
                if query.authority.source.binding_id != source.binding_id:
                    raise ValueError("Scoped core chart binds another original source.")
            else:
                original = source.fidelity_source
                if (
                    not isinstance(original, MappedReferenceBoundarySource)
                    or query.authority.binding_id != original.authority.binding_id
                    or query.domain.domain_id != original.domain.domain_id
                ):
                    raise ValueError(
                        "Scoped mapped query binds a different full source authority."
                    )
        else:
            raise TypeError(
                "No exact scoped-query binding supports this source authority."
            )
    return token.witness


def _require_layer_chart_token(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    source: NativeLayerCoreSource,
    /,
) -> SourceBoundaryChartWitness:
    witness = _require_layer_fidelity_token(mesh, geometry, token, source)
    if not isinstance(witness, SourceBoundaryChartWitness):
        raise ValueError(
            "Partial associations require an exact source boundary chart witness."
        )
    return witness


def _require_layer_image_token(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    source: NativeLayerCoreSource,
    /,
) -> SourceBoundaryFaceImageWitness:
    witness = _require_layer_fidelity_token(mesh, geometry, token, source)
    if not isinstance(witness, SourceBoundaryFaceImageWitness):
        raise ValueError("Image associations require an exact source face image witness.")
    return witness


def _require_mapped_cap_chart_cover(
    query: object, witness: SourceBoundaryChartWitness, /
) -> None:
    """Prove each selected original mapped cap chart is covered once, exactly."""
    from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
    from ..geometry._source_fidelity import _screened_intersection_areas

    if not isinstance(query, MappedReferenceBoundarySource):
        raise TypeError("Mapped cap coverage requires its original mapped query.")
    charts = {query.face_chart_ids[row] for row in query._selected_rows(2)}
    pieces = {chart: [] for chart in charts}
    for row, chart in enumerate(witness.source_chart_ids):
        if chart not in pieces:
            raise ValueError(
                "Mapped cap witness names a chart outside its original scope."
            )
        pieces[chart].append(witness.source_parameter_chart(row))
    for triangles in pieces.values():
        if not triangles:
            raise ValueError("Mapped cap source chart coverage is incomplete.")
        total = Fraction()
        for triangle in triangles:
            if len(triangle) != 3 or any(
                len(point) != 2 or point[0] < 0 or point[1] < 0 or point[0] + point[1] > 1
                for point in triangle
            ):
                raise ValueError(
                    "Mapped cap source chart piece leaves its original triangle."
                )
            a, b, c = triangle
            area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            if not area:
                raise ValueError("Mapped cap source chart piece is degenerate.")
            total += area
        if total != 1:
            raise ValueError(
                "Mapped cap source chart coverage is incomplete or repeated."
            )
        if any(
            area > 0
            for _, _, area in _screened_intersection_areas(
                tuple(triangles), tuple(triangles), self_pairs=True
            )
        ):
            raise ValueError("Mapped cap source chart pieces overlap.")


def require_scoped_cap_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validated_tokens: tuple[ValidatedSourceFidelityWitness, ...],
    source: NativeLayerCoreSource,
    /,
) -> None:
    """Require dedicated current scoped charts for every authored cap source facet."""
    required = core_cap_source_keys(source)
    if not required:
        raise ValueError(
            "Conforming layer/core source has no independently bound cap facets."
        )
    covered: set[tuple[str, tuple[str, ...]]] = set()
    mapped_cap_proved = source.mapped_domain is None
    for token in validated_tokens:
        if not isinstance(token, ValidatedSourceFidelityWitness):
            raise TypeError("Cap coverage requires validated source witness tokens.")
        if (
            token.source is source.fidelity_source
            or not isinstance(
                token.witness,
                (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness),
            )
            or is_internal_core_chart(token)
        ):
            continue
        candidates = set(
            zip(
                token.witness.source_entity_ids,
                token.witness.occurrence_paths,
                strict=True,
            )
        )
        if not candidates & required:
            continue
        witness = _require_layer_fidelity_token(mesh, geometry, token, source)
        assert isinstance(
            witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
        )
        if source.mapped_domain is not None:
            from ..geometry._mapped_reference_boundary import (
                MappedReferenceBoundarySource,
            )

            query = token.source
            if (
                isinstance(query, MappedReferenceBoundarySource)
                and isinstance(witness, SourceBoundaryChartWitness)
                and set(query.face_scope_keys) == required
            ):
                _require_mapped_cap_chart_cover(query, witness)
                mapped_cap_proved = True
        covered.update(
            zip(witness.source_entity_ids, witness.occurrence_paths, strict=True)
        )
    if not required <= covered:
        raise ValueError("Scoped source charts do not cover every authored cap facet.")
    if not mapped_cap_proved:
        raise ValueError(
            "Scoped mapped cap charts do not prove complete original coverage."
        )


def mapped_cap_edge_owner(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    replay: LayerAssociationReplay,
    validated_tokens: tuple[ValidatedSourceFidelityWitness, ...],
    /,
) -> SourceBoundaryChartWitness | None:
    """Select a fully proved original mapped cap chart as its edge-support owner."""
    if source.mapped_domain is None or source.cap_refinement != "conforming":
        return None
    from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource

    require_scoped_cap_coverage(mesh, geometry, validated_tokens, source)
    required = core_cap_source_keys(source)
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    layer_cells, core_cells = set(replay.layer_cell_ids), set(replay.core_cell_ids)
    incidents: dict[int, set[int]] = {}
    for face, cell in incidence_pairs(mesh, 2, 3):
        incidents.setdefault(int(face_ids[face]), set()).add(int(cell_ids[cell]))
    interface = {
        face
        for face, cells in incidents.items()
        if len(cells) == 2
        and len(cells & layer_cells) == 1
        and len(cells & core_cells) == 1
    }
    if not interface:
        raise ValueError(
            "Mapped cap edge ownership requires the actual layer/core interface."
        )
    candidates: list[SourceBoundaryChartWitness] = []
    for token in validated_tokens:
        query, witness = token.source, token.witness
        if (
            query is source.fidelity_source
            or not isinstance(query, MappedReferenceBoundarySource)
            or not isinstance(witness, SourceBoundaryChartWitness)
            or is_internal_core_chart(token)
            or set(query.face_scope_keys) != required
        ):
            continue
        _require_layer_fidelity_token(mesh, geometry, token, source)
        _require_mapped_cap_chart_cover(query, witness)
        actual_faces = {int(value) for value in witness.entity_global_ids}
        if actual_faces != interface:
            raise ValueError(
                "Mapped cap source charts do not cover exactly the actual interface."
            )
        candidates.append(witness)
    if not candidates:
        raise ValueError(
            "Mapped cap edge ownership lacks an independently proved scoped witness."
        )
    return min(candidates, key=lambda witness: witness.witness_id)


def validate_partial_image_association_rows(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    associations: tuple[GeometryAssociation, ...],
    source: NativeLayerCoreSource,
    root: LayerCoreRootGeometry | None,
    inputs: MeshCertificationInputs,
    /,
) -> None:
    """Independently reprove image facet claims and their material sides."""
    _validate_partial_image_association_rows(
        mesh, geometry, token, associations, source, root, inputs
    )


def _validate_partial_image_association_rows(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    associations: tuple[GeometryAssociation, ...],
    source: NativeLayerCoreSource,
    root: LayerCoreRootGeometry | None,
    inputs: MeshCertificationInputs,
    /,
    *,
    computed_root: ComputedLayerCoreRoot | None = None,
    computed_replay: _ComputedLayerAssociationReplay | None = None,
) -> None:
    """Reprove image facet claims and their actual incident material sides."""
    from ..discretization._cell_geometry_transition_types import _geometry_map_expressions
    from ..discretization._coordinate_enclosure import evaluate_expression
    from ._certification_inputs import MeshCertificationInputs
    from ._layer_core_association_transfer import _semantic_rows

    witness = _require_layer_image_token(mesh, geometry, token, source)
    if (
        type(inputs) is not MeshCertificationInputs
        or inputs.mesh is not mesh
        or inputs.geometry is not geometry
        or inputs.feature_source is not source
        or inputs.cell_regions is None
    ):
        raise ValueError(
            "Image side proof requires actual current certified material operands."
        )
    authored = source if source.reference_source is None else source.reference_source
    mapping, caps, polygons, points = prepare_identity(
        authored.layers,
        authored.complex,
        authored.vertex_layer_ids,
        authored.cap_polygon_ids,
    )
    actual_domain, _ = combined_domain(
        authored.layers,
        authored.complex,
        np.asarray(authored.layer_regions, dtype=np.int64),
        mapping,
        caps,
        polygons,
        points,
        authored.source_id,
        authored.region_ids,
        np.asarray(authored.core_region_map, dtype=np.int64),
        authored.region_occurrence_paths,
    )
    mapped = source.mapped_domain
    expected_domain_id = actual_domain.domain_id if mapped is None else mapped.domain_id
    if (
        inputs.domain is None
        or inputs.domain.domain_id != expected_domain_id
        or (
            mapped is not None
            and mapped.reference_domain.domain_id != actual_domain.domain_id
        )
        or tuple(actual_domain.region_ids) != tuple(authored.region_ids)
        or actual_domain.region_occurrence_paths != authored.region_occurrence_paths
        or (inputs.source_id, inputs.source_revision)
        != (authored.source_id, authored.source_revision)
    ):
        raise ValueError(
            "Image material operands do not bind the exact original region namespace."
        )
    regions = np.asarray(inputs.cell_regions, dtype=np.int64)
    cell_ids = np.asarray(inputs.cell_global_ids, dtype=np.int64)
    if (
        regions.shape != cell_ids.shape
        or np.unique(cell_ids).size != cell_ids.size
        or np.any(regions < 0)
        or np.any(regions >= len(authored.region_ids))
    ):
        raise ValueError("Image side proof has invalid material rows.")
    material = {
        int(cell): composite_region_identity(authored, int(region))
        for cell, region in zip(cell_ids, regions, strict=True)
    }
    cells = {
        int(identifier): block.cell_kind
        for block in mesh.blocks
        for identifier in block.global_ids
    }
    if set(material) != set(cells):
        raise ValueError("Image material operands do not cover all actual target cells.")
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    canonical_cells = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    owners: dict[int, list[int]] = {}
    for face_row, cell_row in incidence_pairs(mesh, 2, 3):
        owners.setdefault(int(face_ids[face_row]), []).append(
            int(canonical_cells[cell_row])
        )
    resolved = geometry.resolve(mesh)
    maps = _geometry_map_expressions(mesh, geometry, (resolved[0], resolved[1]))
    interiors: dict[int, tuple[Fraction, ...]] = {}
    for row, (face, orientation, sides) in enumerate(
        zip(
            witness.entity_global_ids,
            witness.orientations,
            witness.source_region_sides,
            strict=True,
        )
    ):
        triangle = witness.image_triangle(row)
        first, second, third = triangle
        u = tuple(b - a for a, b in zip(first, second, strict=True))
        v = tuple(b - a for a, b in zip(first, third, strict=True))
        normal = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        sign = int(orientation)
        if sign not in (-1, 1) or not any(normal):
            raise ValueError("Image source orientation is degenerate.")
        incident = owners.get(int(face), [])
        if not 1 <= len(incident) <= 2:
            raise ValueError("Image target face has invalid cell incidence.")
        present: set[int] = set()
        for cell in incident:
            point = interiors.get(cell)
            if point is None:
                topology = reference_cell_topology(cells[cell])
                count = len(topology.vertices)
                reference = tuple(
                    sum(
                        (
                            Fraction.from_float(float(vertex[axis]))
                            for vertex in topology.vertices
                        ),
                        Fraction(0),
                    )
                    / count
                    for axis in range(mesh.topological_dimension)
                )
                values = tuple(
                    evaluate_expression(value, reference)
                    for value in maps.expressions(cell)
                )
                if any(not isinstance(value, Fraction) for value in values):
                    raise ValueError(
                        "Image target material side needs exact rational cell interiors."
                    )
                point = cast(tuple[Fraction, ...], values)
                interiors[cell] = point
            signed = sign * sum(
                normal[axis] * (point[axis] - first[axis]) for axis in range(3)
            )
            if not signed:
                raise ValueError(
                    "Image target cell interior lies on its source facet plane."
                )
            slot = 0 if signed < 0 else 1
            if sides[slot] is None or sides[slot] != material[cell]:
                raise ValueError(
                    "Image source side conflicts with its incident target material."
                )
            present.add(slot)
        if {index for index, value in enumerate(sides) if value is not None} != present:
            raise ValueError(
                "Image source sides do not match complete target cell incidence."
            )
    layer = _validated_layer_replay(
        source, mesh, geometry, root, computed_root, computed_replay
    )
    core_faces = core_adjacent_face_ids(mesh, set(layer.core_cell_ids))
    physical_keys = {
        _core_source_identity(authored, 2, str(identifier), tuple(path))
        for row, (identifier, path) in enumerate(
            zip(
                authored.complex.facet_ids,
                authored.complex.facet_occurrence_paths,
                strict=True,
            )
        )
        if authored.source_boundary_scope is None
        or authored.core_facet_source_ids[row] is not None
    }
    eligible = frozenset(
        int(face)
        for face, identity, path in zip(
            witness.entity_global_ids,
            witness.source_entity_ids,
            witness.occurrence_paths,
            strict=True,
        )
        if int(face) in core_faces and (identity, tuple(path)) in physical_keys
    )
    expected = replay_core_boundary_image_faces(source, mesh, geometry, token, eligible)
    face_set = mesh.entity_set(2).entity_set_id
    actual = _semantic_rows(
        mesh,
        tuple(
            table
            for table in associations
            if table.target_entity_set_id == face_set
            and table.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
            and table.source_boundary_witness_id == witness.witness_id
            and (table.source_id, table.source_revision)
            == (authored.source_id, authored.source_revision)
        ),
    )
    actual_facets = {
        key: tuple(
            value for value in values if value[1] is GeometrySourceEntityRole.FACET
        )
        for key, values in actual.items()
        if any(value[1] is GeometrySourceEntityRole.FACET for value in values)
    }
    if actual_facets != _semantic_rows(mesh, expected):
        raise ValueError(
            "Image source facet occurrence, side or pullback is forged or omitted."
        )


def validate_partial_association_rows(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    associations: tuple[GeometryAssociation, ...],
    source: NativeLayerCoreSource,
    /,
) -> None:
    """Reprove occurrence, material sides and orientations of supported core faces."""
    from ._layer_core_association_transfer import _semantic_rows

    witness = _require_layer_chart_token(mesh, geometry, token, source)
    authored = source if source.reference_source is None else source.reference_source
    face_set = mesh.entity_set(2).entity_set_id
    actual = tuple(
        association
        for association in associations
        if association.target_entity_set_id == face_set
        and association.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
        and (association.source_id, association.source_revision)
        == (authored.source_id, authored.source_revision)
        and association.source_boundary_witness_id == witness.witness_id
    )
    if not actual:
        return
    eligible = frozenset(
        int(identifier)
        for association in actual
        for identifier, role in zip(
            association.target_global_ids, association.source_roles, strict=True
        )
        if role is GeometrySourceEntityRole.FACET
    )
    if not eligible:
        return
    expected = replay_core_boundary_faces(source, mesh, geometry, token, eligible)
    actual_rows = {
        key: tuple(row for row in rows if row[1] is GeometrySourceEntityRole.FACET)
        for key, rows in _semantic_rows(mesh, actual).items()
        if any(row[1] is GeometrySourceEntityRole.FACET for row in rows)
    }
    if actual_rows != _semantic_rows(mesh, expected):
        raise ValueError(
            "Partial core-face source occurrence, side, orientation or support is forged."
        )


def validate_partial_edge_association_rows(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    associations: tuple[GeometryAssociation, ...],
    source: NativeLayerCoreSource,
    validated_tokens: tuple[ValidatedSourceFidelityWitness, ...],
    root: LayerCoreRootGeometry | None,
    /,
) -> None:
    """Independently reprove partial core-edge source and target charts."""
    _validate_partial_edge_association_rows(
        mesh, geometry, token, associations, source, validated_tokens, root
    )


def _validate_partial_edge_association_rows(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    associations: tuple[GeometryAssociation, ...],
    source: NativeLayerCoreSource,
    validated_tokens: tuple[ValidatedSourceFidelityWitness, ...],
    root: LayerCoreRootGeometry | None,
    /,
    *,
    computed_root: ComputedLayerCoreRoot | None = None,
    computed_replay: _ComputedLayerAssociationReplay | None = None,
    image_preparations: list[_LayerImageSupportPreparation] | None = None,
) -> None:
    """Prove every partial core-edge row against exact source and target charts."""
    from ._layer_core_association_transfer import _semantic_rows
    from ._layer_core_strata_replay import (
        _source_face_support,
        incident_core_source_facets,
        mapped_internal_facet_carriers,
        match_core_source_edges,
    )

    witness = _require_layer_fidelity_token(mesh, geometry, token, source)
    if not isinstance(
        witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
    ):
        raise ValueError("Partial core edges require an exact source face witness.")
    authored = source if source.reference_source is None else source.reference_source
    edge_set = mesh.entity_set(1).entity_set_id
    source_edges = tuple(
        association
        for association in associations
        if association.target_entity_set_id == edge_set
        and association.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
        and (association.source_id, association.source_revision)
        == (authored.source_id, authored.source_revision)
    )
    primary_query = token.source is source.fidelity_source
    cap_coverage_proved = source.cap_refinement == "conforming"
    facet_edges = tuple(
        association
        for association in source_edges
        if any(
            role is GeometrySourceEntityRole.FACET for role in association.source_roles
        )
        and (
            association.source_boundary_witness_id == witness.witness_id
            or (
                association.source_boundary_witness_id is None
                and primary_query
                and not cap_coverage_proved
            )
        )
    )
    facet_targets = {
        int(identifier)
        for association in facet_edges
        for identifier in association.target_global_ids
    }
    relevant = any(
        (primary_query and interval is not None) or int(identifier) in facet_targets
        for association in source_edges
        for identifier, role, interval in zip(
            association.target_global_ids,
            association.source_roles,
            association.target_parameter_intervals,
            strict=True,
        )
        if role is GeometrySourceEntityRole.EDGE
    )
    if not facet_edges and not relevant:
        return
    if cap_coverage_proved:
        require_scoped_cap_coverage(mesh, geometry, validated_tokens, source)
    layer = _validated_layer_replay(
        source, mesh, geometry, root, computed_root, computed_replay
    )
    preparation = _image_support_for_audit(
        source,
        mesh,
        geometry,
        layer.core_cell_ids,
        root,
        computed_root,
        image_preparations,
    )
    core_ids = set(layer.core_cell_ids)
    source_keys = core_source_facet_keys(source)
    core_faces = core_adjacent_face_ids(mesh, core_ids)
    eligible = frozenset(
        int(face_id)
        for cell_id, face_id, identity, occurrence in zip(
            witness.cell_global_ids,
            witness.entity_global_ids,
            witness.source_entity_ids,
            witness.occurrence_paths,
            strict=True,
        )
        if int(cell_id) in core_ids
        or (
            not primary_query
            and int(face_id) in core_faces
            and (identity, occurrence) in source_keys
        )
    )
    face_tables = replay_core_boundary_faces(source, mesh, geometry, token, eligible)
    cap = replay_core_interface_faces(
        source, mesh, layer, require_coverage=not cap_coverage_proved
    )
    if cap is not None:
        face_tables = (*face_tables, cap)
    cap_owner = mapped_cap_edge_owner(source, mesh, geometry, layer, validated_tokens)
    if facet_edges:
        expected_facet_edges = replay_core_facet_edges(
            source,
            mesh,
            geometry,
            witness,
            layer,
            face_tables,
            include_cap=primary_query and not cap_coverage_proved,
            preparation=preparation,
            cap_coverage_proved=cap_coverage_proved,
            cap_owner=cap_owner,
        )
        if _semantic_rows(mesh, facet_edges) != _semantic_rows(
            mesh, expected_facet_edges
        ):
            raise ValueError(
                "Source-facet edge occurrence, side or exact support is forged."
            )
    support = _source_face_support(mesh, geometry, witness, preparation)
    for other in validated_tokens:
        if other is token or not isinstance(
            other.witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
        ):
            continue
        if (other.source.source_id, other.source.source_revision) != (
            source.source_id,
            source.source_revision,
        ):
            continue
        chart = other.witness
        keys = set(zip(chart.source_entity_ids, chart.occurrence_paths, strict=True))
        if other.source is not source.fidelity_source and not keys & source_keys:
            continue
        chart = _require_layer_fidelity_token(mesh, geometry, other, source)
        assert isinstance(
            chart, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
        )
        other_eligible = frozenset(
            int(face_id)
            for cell_id, face_id, identity, occurrence in zip(
                chart.cell_global_ids,
                chart.entity_global_ids,
                chart.source_entity_ids,
                chart.occurrence_paths,
                strict=True,
            )
            if int(cell_id) in core_ids
            or (
                other.source is not source.fidelity_source
                and int(face_id) in core_faces
                and (identity, occurrence) in source_keys
            )
        )
        if not is_internal_core_chart(other):
            face_tables = (
                *face_tables,
                *replay_core_boundary_faces(
                    source, mesh, geometry, other, other_eligible
                ),
            )
        for key, (vertices, edges) in _source_face_support(
            mesh, geometry, chart, preparation
        ).items():
            current = support.setdefault(key, (set(), set()))
            current[0].update(vertices)
            current[1].update(edges)
    carriers = incident_core_source_facets(source, mesh, layer, face_tables, support)
    for edge_id, facets in mapped_internal_facet_carriers(
        source, mesh, geometry, layer, root, validated_tokens, computed_root=computed_root
    ).items():
        carriers[edge_id] = carriers.get(edge_id, frozenset()) | facets
    images = preparation.images
    matches = match_core_source_edges(source, images, carriers)
    edge_identities = tuple(
        _core_source_identity(authored, 1, str(identifier), tuple(path))
        for identifier, path in zip(
            authored.complex.plc_edge_ids,
            authored.complex.plc_edge_occurrence_paths,
            strict=True,
        )
    )
    for association in source_edges:
        for row, identifier in enumerate(association.target_global_ids):
            if association.source_roles[row] is not GeometrySourceEntityRole.EDGE:
                continue
            support_interval = association.target_parameter_intervals[row]
            if int(identifier) not in facet_targets and (
                not primary_query or support_interval is None
            ):
                continue
            if support_interval is None:
                support_interval = (Fraction(0), Fraction(1))
            expected = (
                association.source_entity_ids[row],
                association.source_occurrence_paths[row],
                association.source_parameter_intervals[row],
                support_interval,
                int(np.asarray(association.orientations)[row]),
            )
            valid = {
                (
                    edge_identities[source_row][0],
                    edge_identities[source_row][1],
                    (min(interval), max(interval)),
                    target_interval,
                    1 if interval[1] > interval[0] else -1,
                )
                for source_row, interval, target_interval in matches.get(
                    int(identifier), ()
                )
            }
            if expected not in valid:
                raise ValueError(
                    "Partial core-edge identity, parameters or orientation are forged."
                )


def validate_source_edge_face_overlays(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    associations: tuple[GeometryAssociation, ...],
    source: NativeLayerCoreSource,
    validated_tokens: tuple[ValidatedSourceFidelityWitness, ...],
    root: LayerCoreRootGeometry | None,
    /,
) -> None:
    """Independently reprove exact source-edge support inside target faces."""
    _validate_source_edge_face_overlays(
        mesh, geometry, token, associations, source, validated_tokens, root
    )


def _validate_source_edge_face_overlays(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    associations: tuple[GeometryAssociation, ...],
    source: NativeLayerCoreSource,
    validated_tokens: tuple[ValidatedSourceFidelityWitness, ...],
    root: LayerCoreRootGeometry | None,
    /,
    *,
    computed_root: ComputedLayerCoreRoot | None = None,
    computed_replay: _ComputedLayerAssociationReplay | None = None,
    image_preparations: list[_LayerImageSupportPreparation] | None = None,
) -> None:
    """Reprove exact source-edge support through target face interiors."""
    from ._layer_core_association_transfer import (
        _semantic_rows,
        _SemanticKey,
        _SemanticRow,
    )
    from ._layer_core_segment_overlay import replay_source_edge_face_overlays
    from ._layer_core_strata_replay import (
        _source_face_support,
        core_edge_association_tables,
        incident_core_source_facets,
        mapped_internal_facet_carriers,
        match_core_source_edges,
    )

    _require_layer_fidelity_token(mesh, geometry, token, source)
    cap_coverage_proved = source.cap_refinement == "conforming"
    if cap_coverage_proved:
        require_scoped_cap_coverage(mesh, geometry, validated_tokens, source)
    layer = _validated_layer_replay(
        source, mesh, geometry, root, computed_root, computed_replay
    )
    preparation = _image_support_for_audit(
        source,
        mesh,
        geometry,
        layer.core_cell_ids,
        root,
        computed_root,
        image_preparations,
    )
    source_keys = core_source_facet_keys(source)
    core_ids = set(layer.core_cell_ids)
    core_faces = core_adjacent_face_ids(mesh, core_ids)
    primary_token = next(
        (
            value
            for value in validated_tokens
            if value.source is source.fidelity_source
            and isinstance(
                value.witness,
                (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness),
            )
        ),
        None,
    )
    if primary_token is None:
        raise ValueError("Source-edge overlay lacks its validated primary source chart.")
    _require_layer_fidelity_token(mesh, geometry, primary_token, source)
    contexts = [primary_token]
    for candidate in validated_tokens:
        if candidate is primary_token or not isinstance(
            candidate.witness,
            (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness),
        ):
            continue
        if (candidate.source.source_id, candidate.source.source_revision) != (
            source.source_id,
            source.source_revision,
        ):
            continue
        keys = set(
            zip(
                candidate.witness.source_entity_ids,
                candidate.witness.occurrence_paths,
                strict=True,
            )
        )
        if not keys & source_keys:
            continue
        _require_layer_fidelity_token(mesh, geometry, candidate, source)
        if not any(
            value.witness.witness_id == candidate.witness.witness_id for value in contexts
        ):
            contexts.append(candidate)
    face_tables: list[GeometryAssociation] = []
    chart_support: dict[
        tuple[int, str, tuple[str, ...]], tuple[set[int], set[tuple[int, int]]]
    ] = {}
    for candidate in contexts:
        chart = candidate.witness
        assert isinstance(
            chart, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
        )
        primary = candidate.source is source.fidelity_source
        eligible = frozenset(
            int(face)
            for cell, face, identity, path in zip(
                chart.cell_global_ids,
                chart.entity_global_ids,
                chart.source_entity_ids,
                chart.occurrence_paths,
                strict=True,
            )
            if int(cell) in core_ids
            or (
                not primary
                and int(face) in core_faces
                and (identity, path) in source_keys
            )
        )
        if not is_internal_core_chart(candidate):
            face_tables.extend(
                replay_core_boundary_faces(source, mesh, geometry, candidate, eligible)
            )
        for key, (vertices, edges) in _source_face_support(
            mesh, geometry, chart, preparation
        ).items():
            current = chart_support.setdefault(key, (set(), set()))
            current[0].update(vertices)
            current[1].update(edges)
    cap = replay_core_interface_faces(
        source, mesh, layer, require_coverage=not cap_coverage_proved
    )
    if cap is not None:
        face_tables.append(cap)
    carriers = incident_core_source_facets(
        source, mesh, layer, face_tables, chart_support
    )
    for edge, facets in mapped_internal_facet_carriers(
        source, mesh, geometry, layer, root, validated_tokens, computed_root=computed_root
    ).items():
        carriers[edge] = carriers.get(edge, frozenset()) | facets
    matches = match_core_source_edges(source, preparation.images, carriers)
    authored = source if source.reference_source is None else source.reference_source
    original_rows = tuple(range(len(authored.complex.plc_edge_ids)))
    edge_tables = core_edge_association_tables(
        source, mesh, matches, original_rows, required_source_rows=()
    )
    expected = replay_source_edge_face_overlays(
        source,
        mesh,
        geometry,
        tuple(contexts),
        edge_tables,
        original_rows,
        root,
        computed_root=computed_root,
    )
    face_set = mesh.entity_set(2).entity_set_id

    def selected_rows(
        records: tuple[GeometryAssociation, ...],
    ) -> dict[_SemanticKey, tuple[_SemanticRow, ...]]:
        rows = _semantic_rows(
            mesh,
            tuple(
                record
                for record in records
                if record.target_entity_set_id == face_set
                and (record.source_id, record.source_revision)
                == (authored.source_id, authored.source_revision)
            ),
        )
        return {
            key: tuple(
                value for value in values if value[1] is GeometrySourceEntityRole.EDGE
            )
            for key, values in rows.items()
            if any(value[1] is GeometrySourceEntityRole.EDGE for value in values)
        }

    if selected_rows(associations) != selected_rows(expected):
        raise ValueError(
            "Source-edge face overlay identity, side, parameters or support is forged."
        )


__all__ = [
    "core_source_facet_keys",
    "replay_composite_regions",
    "replay_core_boundary_faces",
    "replay_core_interface_faces",
    "replay_core_facet_edges",
    "replay_layer_region_vertices",
    "require_scoped_cap_coverage",
    "validate_partial_association_rows",
    "validate_partial_image_association_rows",
    "validate_partial_edge_association_rows",
    "validate_source_edge_face_overlays",
]
