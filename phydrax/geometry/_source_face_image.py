#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent exact source and full target-face image partition proof."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..discretization._cell_geometry import CellGeometrySpec, ScalarCoordinateElement
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellBlock, CellMesh
from ..discretization._coordinate_enclosure import (
    coordinate_map_expressions,
    evaluate_expression,
)
from ..discretization._reference_cell import reference_cell_topology
from ._domain_boundary import MeshingDomainBoundarySource
from ._domain_planar_boundary import _face_image_is_corner_polygon
from ._mesh_certificates import _same_tree_content
from ._meshing_domain import SourcePreparation
from ._planar_primitives import plane_key, project_polygon, projection_axes, signed_area2
from ._source_boundary_records import SourceBoundaryFaceImageWitness


type _Point2 = tuple[Fraction, Fraction]
type _Triangle2 = tuple[_Point2, _Point2, _Point2]


def validate_source_face_image(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: MeshingDomainBoundarySource,
    witness: SourceBoundaryFaceImageWitness,
    /,
    *,
    preparation: SourcePreparation | None = None,
) -> tuple[int, ...]:
    """Reprove original source strata and exact implicit full-face pullback pieces.

    Every represented target face is proved to be its exact planar corner
    polygon; each image triangle is independently partitioned both on that
    face and on its occurrence-qualified source trim. No affine inverse of a
    bilinear face is constructed or asserted.
    """
    from ._source_fidelity import (
        _curved_trim_lens_caps,
        _exact_affine_target_charts,
        _require_reference_chart_partition,
        _volume_boundary_facet_lookup,
    )

    def require_source_partition(
        target: tuple[_Triangle2, ...], declared: tuple[_Triangle2, ...], /
    ) -> None:
        from ._source_fidelity import (
            _area2,
            _same_triangle_chain_boundary,
            _screened_intersection_areas,
        )

        target_areas = tuple(abs(_area2(triangle)) for triangle in target)
        source_areas = tuple(abs(_area2(triangle)) for triangle in declared)
        if (
            not target
            or not declared
            or any(not value for value in (*target_areas, *source_areas))
        ):
            raise ValueError(
                "Source face image partition contains a degenerate or empty chart."
            )
        if any(
            area > 0
            for _, _, area in _screened_intersection_areas(
                declared, declared, self_pairs=True
            )
        ) or any(
            area > 0
            for _, _, area in _screened_intersection_areas(
                target, target, self_pairs=True
            )
        ):
            raise ValueError("Source face image partition has positive-area overlap.")
        if sum(target_areas, Fraction(0)) != sum(source_areas, Fraction(0)):
            raise ValueError("Source face image partition omits or adds source area.")
        if not _same_triangle_chain_boundary(declared, target):
            covered = [Fraction(0)] * len(target)
            for target_row, _, area in _screened_intersection_areas(
                target, declared, self_pairs=False
            ):
                covered[target_row] += area
            if any(
                value != area for value, area in zip(covered, target_areas, strict=True)
            ):
                raise ValueError("A physical image piece leaves its source chart.")

    if (
        not isinstance(mesh, CellMesh)
        or mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
    ):
        raise TypeError("Face-image fidelity requires a 3D volume CellMesh.")
    if not isinstance(geometry, CellGeometrySpec) or not isinstance(
        source, MeshingDomainBoundarySource
    ):
        raise TypeError(
            "Face-image fidelity requires its exact domain and coordinate owners."
        )
    if type(witness) is not SourceBoundaryFaceImageWitness:
        raise TypeError("Face-image fidelity requires its canonical witness.")
    source.require_current()
    domain = source.domain
    preparation = SourcePreparation.of(domain, preparation)
    if (
        witness.mesh_id != mesh.mesh_id
        or witness.topology_id != mesh.topology_id
        or witness.geometry_id != cell_geometry_id(geometry)
        or witness.source_id != source.source_id
        or witness.source_revision != source.source_revision
        or witness.source_query_id != source.source_scope_id
    ):
        raise ValueError("Face-image witness changes its mesh, geometry or source query.")
    renewed = SourceBoundaryFaceImageWitness(
        mesh_id=witness.mesh_id,
        geometry_id=witness.geometry_id,
        topology_id=witness.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        cell_global_ids=witness.cell_global_ids,
        entity_global_ids=witness.entity_global_ids,
        local_entity_indices=witness.local_entity_indices,
        source_chart_ids=witness.source_chart_ids,
        source_entity_ids=witness.source_entity_ids,
        occurrence_paths=witness.occurrence_paths,
        source_region_sides=witness.source_region_sides,
        orientations=witness.orientations,
        image_triangle_rational_tokens=witness.image_triangle_rational_tokens,
    )
    if not _same_tree_content(witness, renewed):
        raise ValueError(
            "Face-image witness does not reproduce its actual immutable fields."
        )

    count = len(witness.source_chart_ids)
    sides: list[list[str | None]] = [[None, None] for _ in domain.patches]
    for region_row, region in enumerate(domain.regions):
        for patch, sign in region.boundary:
            slot = 0 if sign > 0 else 1
            identity = domain.entity_id(3, region_row)
            if sides[patch][slot] is not None and sides[patch][slot] != identity:
                raise ValueError(
                    "Source patch has ambiguous oriented material incidence."
                )
            sides[patch][slot] = identity
    patches: list[int] = []
    for row, (chart, entity, occurrence, claimed_sides) in enumerate(
        zip(
            witness.source_chart_ids,
            witness.source_entity_ids,
            witness.occurrence_paths,
            witness.source_region_sides,
            strict=True,
        )
    ):
        patch = domain.resolve_entity_id(2, entity, occurrence_path=occurrence)
        if (
            patch not in source.patches
            or chart != entity
            or domain.entity_id(2, patch) != entity
            or domain.source_occurrences[2][patch] != occurrence
            or claimed_sides != tuple(sides[patch])
        ):
            raise ValueError(
                "Face image changes its source chart, occurrence or material sides."
            )
        patches.append(patch)
    if set(patches) != set(source.patches):
        raise ValueError("Face-image witness omits a selected source patch.")

    cells = {
        int(identifier): (block_index, row, block)
        for block_index, block in enumerate(mesh.blocks)
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    facets = _volume_boundary_facet_lookup(mesh)
    if set(source.patches) == set(range(len(domain.patches))):
        exterior = {identifier for identifier, boundary in facets.values() if boundary}
        if not exterior <= {int(value) for value in witness.entity_global_ids}:
            raise ValueError(
                "Full-source face images must cover every target exterior face."
            )
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
    faces: dict[tuple[int, int], tuple[int, tuple[tuple[Fraction, ...], ...]]] = {}
    face_owners: dict[int, tuple[int, int]] = {}
    groups: dict[tuple[int, int], list[int]] = {}
    for row, (cell_value, face_value, local_value) in enumerate(
        zip(
            witness.cell_global_ids,
            witness.entity_global_ids,
            witness.local_entity_indices,
            strict=True,
        )
    ):
        cell_id, face_id, local = int(cell_value), int(face_value), int(local_value)
        key = (cell_id, local)
        actual = facets.get(key)
        found = cells.get(cell_id)
        if found is None or actual is None or actual[0] != face_id:
            raise ValueError("Face image changes its actual target cell/face incidence.")
        previous = face_owners.setdefault(face_id, key)
        if previous != key:
            raise ValueError("A target face image must use one incident-cell side.")
        if key not in faces:
            block_index, cell_row, block = found
            element = elements[block_index]
            if not isinstance(block, CellBlock) or not isinstance(
                element, ScalarCoordinateElement
            ):
                raise TypeError(
                    "Face-image proof requires an exact standard-cell coordinate map."
                )
            topology = reference_cell_topology(block.cell_kind)
            if not 0 <= local < len(topology.entities[2]):
                raise ValueError("Face image names an absent local target face.")
            reference = tuple(
                tuple(Fraction(float(value)) for value in topology.vertices[slot])
                for slot in topology.entities[2][local]
            )
            if len(reference) not in (3, 4):
                raise ValueError(
                    "Face-image proof requires a triangular or quadrilateral target face."
                )
            local_coordinates = tuple(
                coefficients[int(index)]
                for index in np.asarray(routes[block_index][cell_row])
            )
            expressions = coordinate_map_expressions(
                element, local_coordinates, cell_row=cell_row
            )
            corners = tuple(
                tuple(
                    evaluate_expression(expression, point) for expression in expressions
                )
                for point in reference
            )
            if any(
                len(point) != 3
                or any(not isinstance(value, (Fraction, int)) for value in point)
                for point in corners
            ):
                raise ValueError(
                    "Face-image proof requires exact rational target corners."
                )

            def rational(value: object) -> Fraction:
                if not isinstance(value, (Fraction, int)):
                    raise ValueError(
                        "Face-image proof requires exact rational target corners."
                    )
                return Fraction(value)

            exact = tuple(tuple(rational(value) for value in point) for point in corners)
            if not _face_image_is_corner_polygon((expressions, reference), exact):
                raise ValueError(
                    "Target face is not proved to be its regular planar corner polygon."
                )
            faces[key] = (face_id, exact)
        groups.setdefault(key, []).append(row)

    images = tuple(witness.image_triangle(row) for row in range(count))
    standard = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    inverse = _exact_affine_target_charts(
        source,
        tuple(range(count)),
        tuple(patches),
        (standard,) * count,
        {row: images[row] for row in range(count)},
        set(range(count)),
    )
    if inverse is None:
        raise ValueError("Face image does not lie exactly on its affine source chart.")

    by_patch: dict[int, list[int]] = {}
    for row, patch in enumerate(patches):
        by_patch.setdefault(patch, []).append(row)
        expected_orientation = (
            1 if signed_area2(inverse[row]) > 0 else -1
        ) * domain.patches[patch].orientation
        if int(witness.orientations[row]) != expected_orientation:
            raise ValueError("Face image changes its oriented source occurrence.")
    for patch, rows in by_patch.items():
        target = tuple(inverse[row] for row in rows)
        if source.collapsed_pole_slivers(patch, target):
            raise ValueError(
                "Face-image proof cannot omit a collapsed source-pole region."
            )
        declared, lenses = source._trim_partition_reference_unchecked(
            patch, target, preparation
        )
        if lenses:
            surface = domain.patches[patch].surface
            caps = _curved_trim_lens_caps(
                surface, tuple((row, inverse[row]) for row in rows), lenses, preparation
            )
            if any(value != 0 for value in caps.values()):
                raise ValueError(
                    "Face-image proof requires an exact polygonal source trim."
                )
        require_source_partition(target, declared)

    for key, rows in groups.items():
        _, corners = faces[key]
        plane = plane_key(corners[:3])
        axes = projection_axes(plane)
        polygon = project_polygon(corners, axes)
        projected = tuple(project_polygon(images[row], axes) for row in rows)
        target = tuple((piece[0], piece[1], piece[2]) for piece in projected)
        if any(plane_key(images[row]) != plane for row in rows):
            raise ValueError("Face image changes its exact target supporting plane.")
        face_sign = 1 if signed_area2(polygon) > 0 else -1
        if any((1 if signed_area2(piece) > 0 else -1) != face_sign for piece in target):
            raise ValueError("Face image changes its actual target-face orientation.")
        whole = ((polygon[0], polygon[1], polygon[2]),)
        if len(polygon) == 4:
            whole += ((polygon[0], polygon[2], polygon[3]),)
        _require_reference_chart_partition(target, whole)
    return tuple(patches)
