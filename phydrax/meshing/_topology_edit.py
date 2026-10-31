#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host topology-edit records shared by the native adaptation routes.

A route describes its target mesh relative to the source in global identities only:
vertices and cells by global ID, intermediate entities (edges, faces) by their sorted
vertex-global-ID keys. `assemble_topology_edit` owns the one conversion into a
canonical `CellMesh` with deterministic intermediate entity IDs, a complete
`MeshLineage`, and the sparse vertex interpolation stencil.
"""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

import numpy as np

from ..discretization import CellBlock, CellMesh, PolyhedralConnectivity
from ..discretization._cell_complex import polyhedral_connectivity
from ..discretization._cell_mesh import PolyhedralBlock
from ..discretization._nested_reference_types import (
    exact_reference_vertices,
    NestedReferenceWitnesses,
    ReferenceCellMapWitnesses,
    validate_nested_reference_witnesses,
    validate_reference_cell_map_witnesses,
)
from ..discretization._periodic_topology import (
    ExactPeriodicCoordinateSource,
    PeriodicMeshTopology,
)
from ..discretization._reference_cell import (
    facet_orientation_between,
    reference_cell_topology,
)
from ._lineage import (
    EntityLineage,
    EntityLineageKind,
    interpolation_rows_preserve_constants,
    MeshLineage,
    VertexInterpolationStencil,
)


class EntityRelations(NamedTuple):
    """Explicit lineage relations of one entity dimension.

    Vertex and cell keys are ``(r, 1)`` global IDs; edge and face keys are
    ``(r, dimension + 1)`` ascending vertex global IDs. A key present in both
    meshes also receives the identity PRESERVED relation unless an explicit
    relation already maps that source key onto the same target key.
    """

    dimension: int
    source_keys: np.ndarray
    target_keys: np.ndarray
    kinds: np.ndarray


class PrescribedEntityIds(NamedTuple):
    """Global IDs a route restores for new target edges or faces.

    Coarsening restores retired entities under their recorded IDs; ``keys`` are
    sorted vertex-global-ID rows that must not name source entities.
    """

    dimension: int
    keys: np.ndarray
    ids: np.ndarray


class TopologyEditBlock(NamedTuple):
    """Target cells of one fixed-family block, in reference-cell vertex order."""

    name: str
    cell_kind: str
    source_kind: str | None
    cells: np.ndarray
    cell_ids: np.ndarray


class PolyhedralTopologyEditBlock(NamedTuple):
    """Target polyhedra with their outward-oriented packed cell-face incidence."""

    name: str
    source_kind: str | None
    connectivity: PolyhedralConnectivity
    cell_ids: np.ndarray

    @property
    def cell_kind(self) -> str:
        return "polyhedron"


class SharedFaceWitnesses(NamedTuple):
    """Oriented target-cell facet pairs that must form a conforming shared face.

    Cell selectors are stable target global IDs. Facet selectors are local
    reference facets (or local packed faces for polyhedra). Each permutation maps
    the first facet's ordered vertex list to the second's.
    """

    first_cell_ids: np.ndarray
    first_facets: np.ndarray
    second_cell_ids: np.ndarray
    second_facets: np.ndarray
    permutations: np.ndarray


class CellTopologyEdit(NamedTuple):
    """General target cell complex of one topology edit, relative to its source.

    Fixed-family and polyhedral blocks can coexist. Cell and vertex identity is
    supplied in global IDs, while connectivity arrays use coordinate rows.
    Vertex stencils and nested reference witnesses retain exact source ancestry;
    optional periodic orbit keys retain explicit discrete ancestry when a source
    interpolation stencil cannot encode the quotient identity.
    """

    coordinates: np.ndarray
    vertex_global_ids: np.ndarray
    blocks: tuple[TopologyEditBlock | PolyhedralTopologyEditBlock, ...]
    stencil_sources: np.ndarray
    stencil_weights: np.ndarray
    stencil_valid: np.ndarray
    relations: tuple[EntityRelations, ...]
    prescribed_entity_ids: tuple[PrescribedEntityIds, ...] = ()
    shared_faces: SharedFaceWitnesses | None = None
    refinement: NestedReferenceWitnesses | None = None
    coarsening: NestedReferenceWitnesses | None = None
    vertex_orbit_keys: np.ndarray | None = None
    vertex_windings: np.ndarray | None = None
    source_entity_degree: int | None = None
    stencil_exact_weights: tuple[tuple[Fraction, ...], ...] | None = None


def _exact_stencil_row(
    edit: CellTopologyEdit, row: int, /
) -> tuple[tuple[int, Fraction], ...]:
    sources = np.asarray(edit.stencil_sources, dtype=np.int64)
    valid = np.asarray(edit.stencil_valid, dtype=np.bool_)
    weights = np.asarray(edit.stencil_weights, dtype=np.float64)
    if (
        sources.ndim != 2
        or sources.shape != valid.shape
        or sources.shape != weights.shape
        or not 0 <= row < sources.shape[0]
    ):
        raise ValueError("Exact source stencil row is unavailable.")
    if edit.stencil_exact_weights is None:
        exact = tuple(Fraction.from_float(float(value)) for value in weights[row])
    else:
        exact_rows = edit.stencil_exact_weights
        if len(exact_rows) != sources.shape[0] or any(
            len(values) != sources.shape[1] for values in exact_rows
        ):
            raise ValueError("Exact source stencil weights must align with rows.")
        exact = exact_rows[row]
        if any(not isinstance(value, Fraction) for value in exact):
            raise TypeError("Exact source stencil coefficients must be Fractions.")
    return tuple(
        (int(source), coefficient)
        for source, coefficient, is_valid in zip(
            sources[row], exact, valid[row], strict=True
        )
        if is_valid
    )


def _edit_stencil_preserves_constants(edit: CellTopologyEdit, /) -> bool:
    sources = np.asarray(edit.stencil_sources, dtype=np.int64)
    valid = np.asarray(edit.stencil_valid, dtype=np.bool_)
    weights = np.asarray(edit.stencil_weights, dtype=np.float64)
    if (
        sources.ndim != 2
        or sources.shape != valid.shape
        or sources.shape != weights.shape
    ):
        return False
    if edit.stencil_exact_weights is None:
        return interpolation_rows_preserve_constants(weights, valid)
    exact = edit.stencil_exact_weights
    valid_rows = tuple(tuple(bool(value) for value in row) for row in valid.tolist())
    return (
        len(exact) == sources.shape[0]
        and all(len(row) == sources.shape[1] for row in exact)
        and all(
            sum(
                (
                    value
                    for value, is_valid in zip(row, row_valid, strict=True)
                    if is_valid
                ),
                Fraction(0),
            )
            == 1
            and all(
                not is_valid or isinstance(value, Fraction)
                for value, is_valid in zip(row, row_valid, strict=True)
            )
            and all(
                is_valid or value == 0
                for value, is_valid in zip(row, row_valid, strict=True)
            )
            for row, row_valid in zip(exact, valid_rows, strict=True)
        )
    )


def source_family_blocks(
    families: tuple[tuple[str, str], ...],
    block_cells: tuple[np.ndarray, ...],
    block_cell_ids: tuple[np.ndarray, ...],
    /,
) -> tuple[TopologyEditBlock, ...]:
    """Build nonempty target blocks aligned with the source family table."""

    if len(block_cells) != len(families) or len(block_cell_ids) != len(families):
        raise ValueError("Target block arrays must align with source families.")
    return tuple(
        TopologyEditBlock(name, kind, kind, cells, identifiers)
        for (name, kind), cells, identifiers in zip(
            families, block_cells, block_cell_ids, strict=True
        )
        if np.asarray(identifiers).size
    )


def _face_rows(connectivity: object, dimension: int) -> tuple[np.ndarray, ...]:
    """Return each intermediate entity's ordered vertex row."""
    if dimension == 1:
        edges = getattr(connectivity, "edges", None)
        if edges is None:
            raise ValueError("Edge connectivity is unavailable.")
        return tuple(np.asarray(row, dtype=np.int64) for row in np.asarray(edges))
    if dimension != 2:
        raise ValueError(f"Vertex rows of dimension {dimension} are unavailable.")
    offsets = getattr(connectivity, "face_vertex_offsets", None)
    values = getattr(connectivity, "face_vertex_values", None)
    if offsets is not None and values is not None:
        offsets_ = np.asarray(offsets, dtype=np.int64)
        values_ = np.asarray(values, dtype=np.int64)
        return tuple(
            values_[offsets_[index] : offsets_[index + 1]]
            for index in range(offsets_.size - 1)
        )
    faces = getattr(connectivity, "faces", None)
    if faces is None:
        raise ValueError("Face connectivity is unavailable.")
    return tuple(np.asarray(row, dtype=np.int64) for row in np.asarray(faces))


def _padded_keys(
    rows: tuple[np.ndarray, ...], vertex_ids: np.ndarray, width: int
) -> np.ndarray:
    result = np.full((len(rows), width), -1, dtype=np.int64)
    for index, row in enumerate(rows):
        values = np.sort(vertex_ids[np.asarray(row, dtype=np.int64)])
        if values.size > width:
            raise ValueError("Entity key width is smaller than a face.")
        result[index, : values.size] = values
    return result


def _pad_key_table(keys: np.ndarray, width: int) -> np.ndarray:
    value = np.asarray(keys, dtype=np.int64)
    if value.ndim != 2 or value.shape[1] > width:
        raise ValueError("Entity relation keys must be padded rank-two arrays.")
    if value.shape[1] == width:
        return value
    result = np.full((value.shape[0], width), -1, dtype=np.int64)
    result[:, : value.shape[1]] = value
    return result


def _cell_facets(mesh: CellMesh) -> tuple[tuple[np.ndarray, ...], ...]:
    """Ordered facets of all target cells in concatenated block order.

    Fixed-family cells use their local reference facets; only polyhedral cells
    use their packed outward face loops, matching ``SharedFaceWitnesses``.
    """
    topology = mesh.topological_dimension
    connectivity = mesh.connectivity
    facets: list[tuple[np.ndarray, ...]] = []
    row = 0
    for block in mesh.blocks:
        cells = np.asarray(block.vertices, dtype=np.int64)
        if isinstance(block, PolyhedralBlock):
            if not isinstance(connectivity, PolyhedralConnectivity):
                raise ValueError("Polyhedral cells require packed face connectivity.")
            cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
            cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
            signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int8)
            face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
            face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
            for cell in range(row, row + block.cell_count):
                loops = []
                for position in range(cell_offsets[cell], cell_offsets[cell + 1]):
                    face = int(cell_faces[position])
                    loop = face_values[face_offsets[face] : face_offsets[face + 1]]
                    loops.append(loop if signs[position] > 0 else loop[::-1])
                facets.append(tuple(np.asarray(loop, dtype=np.int64) for loop in loops))
        elif block.cell_kind == "polygon":
            facets.extend(
                tuple(
                    np.asarray(
                        (loop[index], loop[(index + 1) % loop.size]), dtype=np.int64
                    )
                    for index in range(loop.size)
                )
                for loop in cells
            )
        else:
            local = reference_cell_topology(block.cell_kind).entities[topology - 1]
            facets.extend(
                tuple(cells[cell, np.asarray(face, dtype=np.int64)] for face in local)
                for cell in range(cells.shape[0])
            )
        row += block.cell_count
    return tuple(facets)


def _polygon_cycle_action(
    first: tuple[int, ...], second: tuple[int, ...], /
) -> tuple[int, ...]:
    """Position map of one polygon loop in another; it must be a dihedral action."""
    positions = {vertex: index for index, vertex in enumerate(second)}
    action = tuple(positions[vertex] for vertex in first)
    size = len(action)
    step = (action[1] - action[0]) % size
    if step not in (1, size - 1) or any(
        (action[(index + 1) % size] - action[index]) % size != step
        for index in range(size)
    ):
        raise ValueError("Shared polygon faces must be the same cycle up to orientation.")
    return action


def _require_shared_faces(target: CellMesh, witnesses: SharedFaceWitnesses, /) -> None:
    first_cells = np.asarray(witnesses.first_cell_ids, dtype=np.int64)
    second_cells = np.asarray(witnesses.second_cell_ids, dtype=np.int64)
    first_facets = np.asarray(witnesses.first_facets, dtype=np.int64)
    second_facets = np.asarray(witnesses.second_facets, dtype=np.int64)
    permutations = np.asarray(witnesses.permutations, dtype=np.int64)
    count = first_cells.size
    if (
        any(
            value.shape != (count,)
            for value in (second_cells, first_facets, second_facets)
        )
        or permutations.ndim != 2
        or permutations.shape[0] != count
    ):
        raise ValueError(
            "Shared-face witnesses must be aligned rank-one/rank-two arrays."
        )
    all_facets = _cell_facets(target)
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target.blocks]
    )
    order = np.argsort(cell_ids, kind="stable")
    sorted_ids = cell_ids[order]

    def rows(ids: np.ndarray) -> np.ndarray:
        position = np.searchsorted(sorted_ids, ids)
        valid = position < sorted_ids.size
        if not np.all(valid) or not np.array_equal(sorted_ids[position], ids):
            raise ValueError("Shared-face witnesses name absent target cell IDs.")
        return order[position]

    first_rows, second_rows = rows(first_cells), rows(second_cells)
    if count == 0:
        return
    vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    keys: dict[tuple[int, ...], int] = {}
    for cell in all_facets:
        for face in cell:
            key = tuple(sorted(vertex_ids[face].tolist()))
            keys[key] = keys.get(key, 0) + 1
    for index in range(count):
        left_cell, right_cell = int(first_rows[index]), int(second_rows[index])
        left_facet, right_facet = int(first_facets[index]), int(second_facets[index])
        if not (
            0 <= left_facet < len(all_facets[left_cell])
            and 0 <= right_facet < len(all_facets[right_cell])
        ):
            raise ValueError("Shared-face witness names an absent local facet.")
        left = vertex_ids[all_facets[left_cell][left_facet]]
        right = vertex_ids[all_facets[right_cell][right_facet]]
        if set(left.tolist()) != set(right.tolist()):
            raise ValueError("Shared-face witness facets have different vertices.")
        left_loop = tuple(int(value) for value in left)
        right_loop = tuple(int(value) for value in right)
        expected = (
            facet_orientation_between(left_loop, right_loop).permutation
            if len(left_loop) <= 4
            else _polygon_cycle_action(left_loop, right_loop)
        )
        declared_row = permutations[index]
        if np.any(declared_row < -1) or np.any(
            (declared_row[:-1] == -1) & (declared_row[1:] >= 0)
        ):
            raise ValueError("Shared-face permutation padding must be a -1 suffix.")
        declared = tuple(int(value) for value in declared_row if value >= 0)
        if expected != declared:
            raise ValueError("Shared-face witness has a wrong orientation permutation.")
        if keys[tuple(sorted(left.tolist()))] != 2:
            raise ValueError(
                "Witnessed shared face must be incident to exactly two cells."
            )


def _cell_records(mesh: CellMesh) -> dict[int, tuple[str, np.ndarray]]:
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    result: dict[int, tuple[str, np.ndarray]] = {}
    for block in mesh.blocks:
        if block.cell_kind == "polyhedron":
            continue
        cells = np.asarray(block.vertices, dtype=np.int64)
        ids = np.asarray(block.global_ids, dtype=np.int64)
        for identifier, row in zip(ids.tolist(), cells, strict=True):
            result[int(identifier)] = (block.cell_kind, vertex_ids[row])
    return result


def _validate_nested_witnesses(
    fine: CellMesh,
    coarse: CellMesh,
    witnesses: NestedReferenceWitnesses,
    /,
) -> None:
    dimension = coarse.topological_dimension
    validate_nested_reference_witnesses(witnesses, dimension=dimension)
    fine_records, coarse_records = _cell_records(fine), _cell_records(coarse)
    fine_ids = np.asarray(witnesses.fine_cell_ids, dtype=np.int64)
    coarse_ids = np.asarray(witnesses.coarse_cell_ids, dtype=np.int64)
    points = np.asarray(witnesses.fine_reference_vertices, dtype=np.float64)
    for fine_id, coarse_id, reference in zip(fine_ids, coarse_ids, points, strict=True):
        fine_record = fine_records.get(int(fine_id))
        coarse_record = coarse_records.get(int(coarse_id))
        if fine_record is None or coarse_record is None:
            raise ValueError(
                "Nested reference witnesses must name present fixed-family cells."
            )
        fine_kind, fine_vertices = fine_record
        coarse_kind, coarse_vertices = coarse_record
        if (
            len(fine_vertices) != dimension + 1
            or len(coarse_vertices) != dimension + 1
            or fine_kind
            not in ("interval", "triangle", "tetrahedron", f"simplex:{dimension}")
            or coarse_kind
            not in ("interval", "triangle", "tetrahedron", f"simplex:{dimension}")
        ):
            raise ValueError(
                "Nested reference witnesses currently require simplex cells."
            )
        barycentric = np.column_stack((1.0 - np.sum(reference, axis=1), reference))
        if np.any(barycentric < 0.0):
            raise ValueError("Nested fine-cell corners must lie in the coarse simplex.")


def _validate_nested_stencil(
    target: CellMesh,
    source: CellMesh,
    edit: CellTopologyEdit,
    witnesses: NestedReferenceWitnesses,
    /,
) -> None:
    fine_records = _cell_records(target)
    coarse_records = _cell_records(source)
    exact = exact_reference_vertices(witnesses)
    target_vertices = np.asarray(edit.vertex_global_ids, dtype=np.int64)
    order = np.argsort(target_vertices, kind="stable")
    ordered = target_vertices[order]
    sources = np.asarray(edit.stencil_sources, dtype=np.int64)
    weights = np.asarray(edit.stencil_weights, dtype=np.float64)
    valid = np.asarray(edit.stencil_valid, dtype=np.bool_)
    if (
        sources.shape != weights.shape
        or sources.shape != valid.shape
        or sources.shape[0] != target_vertices.size
    ):
        raise ValueError(
            "Topology edit vertex stencils must align with target vertex IDs."
        )
    for row, (fine_id, coarse_id) in enumerate(
        zip(witnesses.fine_cell_ids, witnesses.coarse_cell_ids, strict=True)
    ):
        fine = fine_records[int(fine_id)][1]
        coarse = coarse_records[int(coarse_id)][1]
        expected = exact[row]
        if len(fine) != len(coarse) or len(expected) != len(fine):
            raise ValueError("Nested refinement witnesses must be simplex corner maps.")
        position = np.searchsorted(ordered, fine)
        if np.any(position >= ordered.size) or not np.array_equal(
            ordered[position], fine
        ):
            raise ValueError("Nested target corners must have vertex stencils.")
        rows = order[position]
        for local, stencil_row in enumerate(rows):
            active = _exact_stencil_row(edit, int(stencil_row))
            if not active or any(weight < 0 for _, weight in active):
                raise ValueError(
                    "Nested target corners need nonnegative source stencils."
                )
            rational = [Fraction(0) for _ in range(len(coarse))]
            for source_id, coefficient in active:
                locations = np.flatnonzero(coarse == source_id)
                if locations.size != 1:
                    raise ValueError(
                        "A nested stencil source lies outside its parent cell."
                    )
                rational[int(locations[0])] += coefficient
            total = sum(rational, Fraction(0))
            if total <= 0:
                raise ValueError("Nested stencil barycentrics need a positive exact sum.")
            reference = tuple(value / total for value in rational[1:])
            if reference != expected[local]:
                raise ValueError(
                    "Nested reference map must equal the exact vertex stencil."
                )


def nested_reference_witnesses_from_edit(
    source: CellMesh,
    target: CellMesh,
    edit: CellTopologyEdit,
    /,
) -> NestedReferenceWitnesses | None:
    """Derive exact parent-reference maps from refined cell lineage and stencils.

    The returned records are source-cell-ID keyed and retain exact Fraction
    coordinates plus their binary64 execution view. Non-nested cell relations are
    not converted into parent maps.
    """
    if source.topological_dimension != target.topological_dimension:
        raise ValueError(
            "Nested mesh endpoints must have the same topological dimension."
        )
    relation = edit.relations[source.topological_dimension]
    kinds = np.asarray(relation.kinds, dtype=np.int32)
    selected = kinds == int(EntityLineageKind.REFINED_FROM)
    if not np.any(selected):
        return None
    source_keys = np.asarray(relation.source_keys, dtype=np.int64)[selected]
    target_keys = np.asarray(relation.target_keys, dtype=np.int64)[selected]
    if (
        source_keys.ndim != 2
        or source_keys.shape[1] != 1
        or target_keys.shape != source_keys.shape
    ):
        raise ValueError("Refined cell lineage must use stable one-column cell IDs.")
    fine_ids, coarse_ids = target_keys[:, 0], source_keys[:, 0]
    if np.unique(fine_ids).size != fine_ids.size:
        raise ValueError("A refined target cell must have exactly one parent cell.")
    fine_records = _cell_records(target)
    coarse_records = _cell_records(source)
    target_vertices = np.asarray(edit.vertex_global_ids, dtype=np.int64)
    vertex_order = np.argsort(target_vertices, kind="stable")
    ordered_vertices = target_vertices[vertex_order]
    stencil_sources = np.asarray(edit.stencil_sources, dtype=np.int64)
    stencil_weights = np.asarray(edit.stencil_weights, dtype=np.float64)
    stencil_valid = np.asarray(edit.stencil_valid, dtype=np.bool_)
    if (
        stencil_sources.shape != stencil_weights.shape
        or stencil_sources.shape != stencil_valid.shape
        or stencil_sources.shape[0] != target_vertices.size
    ):
        raise ValueError("Topology edit stencils must align with target vertex IDs.")
    dimension = source.topological_dimension
    exact_cells: list[tuple[tuple[Fraction, ...], ...]] = []
    rounded = np.empty((fine_ids.size, dimension + 1, dimension), dtype=np.float64)
    for row, (fine_id, coarse_id) in enumerate(zip(fine_ids, coarse_ids, strict=True)):
        fine_record = fine_records.get(int(fine_id))
        coarse_record = coarse_records.get(int(coarse_id))
        if fine_record is None or coarse_record is None:
            raise ValueError("Refined lineage must name present simplex cells.")
        fine_kind, fine_vertices = fine_record
        coarse_kind, coarse_vertices = coarse_record
        if (
            len(fine_vertices) != dimension + 1
            or len(coarse_vertices) != dimension + 1
            or fine_kind
            not in ("interval", "triangle", "tetrahedron", f"simplex:{dimension}")
            or coarse_kind
            not in ("interval", "triangle", "tetrahedron", f"simplex:{dimension}")
        ):
            raise ValueError(
                "Exact nested coordinate maps currently require simplex cells."
            )
        target_positions = np.searchsorted(ordered_vertices, fine_vertices)
        if np.any(target_positions >= ordered_vertices.size) or not np.array_equal(
            ordered_vertices[target_positions], fine_vertices
        ):
            raise ValueError("Every refined target corner needs a vertex stencil.")
        stencil_rows = vertex_order[target_positions]
        vertices: list[tuple[Fraction, ...]] = []
        for local, stencil_row in enumerate(stencil_rows):
            active = _exact_stencil_row(edit, int(stencil_row))
            if not active or any(weight < 0 for _, weight in active):
                raise ValueError(
                    "Nested target corners need nonnegative finite stencils."
                )
            accumulated = [Fraction(0) for _ in range(dimension + 1)]
            for source_id, weight in active:
                corner = np.flatnonzero(coarse_vertices == source_id)
                if corner.size != 1:
                    raise ValueError(
                        "A refined vertex stencil escapes its source parent cell."
                    )
                accumulated[int(corner[0])] += weight
            total = sum(accumulated, Fraction(0))
            if total != 1:
                raise ValueError(
                    "Exact nested coordinate maps require vertex stencils that sum exactly to one."
                )
            reference = tuple(accumulated[1:])
            if any(value < 0 for value in (*reference, Fraction(1) - sum(reference))):
                raise ValueError(
                    "A refined vertex lies outside its parent reference simplex."
                )
            vertices.append(reference)
            rounded[row, local] = [float(value) for value in reference]
        exact_cells.append(tuple(vertices))
    witnesses = NestedReferenceWitnesses(
        fine_ids,
        coarse_ids,
        rounded,
        fine_reference_exact_vertices=tuple(exact_cells),
    )
    validate_nested_reference_witnesses(witnesses, dimension=dimension)
    return witnesses


def reference_cell_maps_from_edit(
    source: CellMesh,
    target: CellMesh,
    edit: CellTopologyEdit,
    /,
) -> tuple[ReferenceCellMapWitnesses, ...]:
    """Build exact target-reference→source-reference maps from vertex stencils.

    This only accepts preserved or one-parent nested same-dimensional fixed
    cells. Stencil weights are the exact binary64 rationals they represent and
    must sum exactly to one; no rounded physical coordinates are inverted to
    invent ancestry.
    """
    if not isinstance(source, CellMesh) or not isinstance(target, CellMesh):
        raise TypeError("Reference-cell ancestry requires canonical CellMesh endpoints.")
    if not isinstance(edit, CellTopologyEdit):
        raise TypeError("edit must be CellTopologyEdit.")
    if source.topological_dimension != target.topological_dimension:
        raise ValueError(
            "Same-domain reference maps require equal topological dimensions."
        )
    if any(
        not isinstance(block, CellBlock) for block in (*source.blocks, *target.blocks)
    ):
        raise ValueError("Exact reference-cell maps require fixed-family cell blocks.")

    source_records: dict[
        int, tuple[str, tuple[int, ...], tuple[tuple[Fraction, ...], ...]]
    ] = {}
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    for block in source.blocks:
        topology = reference_cell_topology(block.cell_kind)
        connectivity = np.asarray(block.vertices, dtype=np.int64)
        local_vertices = tuple(
            tuple(Fraction(float(value)) for value in vertex)
            for vertex in topology.vertices
        )
        if connectivity.shape[1] != len(local_vertices):
            raise ValueError(
                "Source cell connectivity does not match its reference corners."
            )
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            vertex_ids = tuple(
                int(value) for value in source_vertex_ids[connectivity[row]]
            )
            if int(identifier) in source_records:
                raise ValueError(
                    "Source global cell IDs must be unique across family blocks."
                )
            source_records[int(identifier)] = (
                block.cell_kind,
                vertex_ids,
                local_vertices,
            )

    target_records: dict[int, tuple[str, tuple[int, ...]]] = {}
    target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    for block in target.blocks:
        topology = reference_cell_topology(block.cell_kind)
        connectivity = np.asarray(block.vertices, dtype=np.int64)
        if connectivity.shape[1] != len(topology.vertices):
            raise ValueError(
                "Target cell connectivity does not match its reference corners."
            )
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            cell_vertices = tuple(
                int(value) for value in target_vertex_ids[connectivity[row]]
            )
            if int(identifier) in target_records:
                raise ValueError(
                    "Target global cell IDs must be unique across family blocks."
                )
            target_records[int(identifier)] = (block.cell_kind, cell_vertices)

    dimension = source.topological_dimension
    relation = next(
        (value for value in edit.relations if value.dimension == dimension), None
    )
    if relation is None:
        raise ValueError("Topology edit has no top-cell lineage for reference maps.")
    source_keys = np.asarray(relation.source_keys, dtype=np.int64)
    target_keys = np.asarray(relation.target_keys, dtype=np.int64)
    kinds = np.asarray(relation.kinds, dtype=np.int32)
    if (
        source_keys.ndim != 2
        or source_keys.shape[1] != 1
        or target_keys.shape != source_keys.shape
        or kinds.shape != (source_keys.shape[0],)
    ):
        raise ValueError("Reference-map cell lineage must use aligned stable global IDs.")
    parents: dict[int, int] = {}
    for source_id, target_id, kind in zip(
        source_keys[:, 0], target_keys[:, 0], kinds, strict=True
    ):
        if int(kind) not in (
            int(EntityLineageKind.PRESERVED),
            int(EntityLineageKind.REFINED_FROM),
            int(EntityLineageKind.RELOCATED),
        ):
            continue
        target_id_, source_id_ = int(target_id), int(source_id)
        if target_id_ in parents:
            raise ValueError("A nested target cell cannot have multiple source parents.")
        parents[target_id_] = source_id_
    for target_id, target_record in target_records.items():
        if target_id in parents:
            continue
        source_record = source_records.get(target_id)
        if (
            source_record is not None
            and (source_record[0], source_record[1]) == target_record
        ):
            parents[target_id] = target_id
    if set(parents) != set(target_records):
        raise ValueError(
            "Every target cell needs one preserved or nested source-cell parent."
        )

    target_vertices = np.asarray(edit.vertex_global_ids, dtype=np.int64)
    stencil_sources = np.asarray(edit.stencil_sources, dtype=np.int64)
    stencil_weights = np.asarray(edit.stencil_weights, dtype=np.float64)
    stencil_valid = np.asarray(edit.stencil_valid, dtype=np.bool_)
    if (
        stencil_sources.shape != stencil_weights.shape
        or stencil_sources.shape != stencil_valid.shape
        or stencil_sources.shape[0] != target_vertices.size
        or np.unique(target_vertices).size != target_vertices.size
        or not np.array_equal(np.sort(target_vertices), np.sort(target_vertex_ids))
    ):
        raise ValueError(
            "Topology-edit vertex stencils must cover target global vertices."
        )
    vertex_rows = {int(identifier): row for row, identifier in enumerate(target_vertices)}
    grouped: dict[
        tuple[str, str],
        list[tuple[int, int, tuple[tuple[Fraction, ...], ...]]],
    ] = {}
    for target_id, (target_kind, child_vertices) in target_records.items():
        source_id = parents[target_id]
        parent = source_records.get(source_id)
        if parent is None:
            raise ValueError("Nested target cell references an absent source parent.")
        source_kind, parent_vertices, parent_reference_vertices = parent
        parent_slots = {identifier: row for row, identifier in enumerate(parent_vertices)}
        exact_vertices = []
        for target_vertex in child_vertices:
            stencil_row = vertex_rows[target_vertex]
            active = _exact_stencil_row(edit, stencil_row)
            if not active:
                raise ValueError("Nested target corner has no exact source stencil.")
            accumulated = [Fraction(0) for _ in parent_vertices]
            for source_vertex, weight in active:
                slot = parent_slots.get(source_vertex)
                if slot is None:
                    raise ValueError(
                        "Nested vertex stencil escapes its declared source cell."
                    )
                accumulated[slot] += weight
            if sum(accumulated, Fraction(0)) != 1:
                raise ValueError(
                    "Exact reference ancestry requires source rows whose exact weights sum to one."
                )
            exact_vertices.append(
                tuple(
                    sum(
                        (
                            accumulated[corner] * parent_reference_vertices[corner][axis]
                            for corner in range(len(parent_vertices))
                        ),
                        Fraction(0),
                    )
                    for axis in range(dimension)
                )
            )
        grouped.setdefault((target_kind, source_kind), []).append(
            (target_id, source_id, tuple(exact_vertices))
        )

    witnesses = []
    for (target_kind, source_kind), rows in sorted(grouped.items()):
        rows.sort(key=lambda value: value[0])
        target_ids = np.asarray([value[0] for value in rows], dtype=np.int64)
        source_ids = np.asarray([value[1] for value in rows], dtype=np.int64)
        exact = tuple(value[2] for value in rows)
        execution = np.asarray(
            [[[float(value) for value in point] for point in cell] for cell in exact],
            dtype=np.float64,
        )
        witness = ReferenceCellMapWitnesses(
            target_ids,
            source_ids,
            target_kind,
            source_kind,
            execution,
            exact,
        )
        validate_reference_cell_map_witnesses(witness)
        witnesses.append(witness)
    return tuple(witnesses)


def entity_keys(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    """Stable identity keys aligned with one mesh entity set.

    Intermediate entities are represented by sorted vertex global IDs. Mixed
    faces are padded with ``-1`` after their active sorted IDs; this padding is
    structural and never an entity identity.
    """

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    target = int(dimension)
    if target < 0 or target > mesh.topological_dimension:
        raise ValueError(f"Entity dimension {target} is outside the mesh.")
    if target == 0:
        return np.asarray(mesh.vertex_global_ids, dtype=np.int64)[:, None]
    if target == mesh.topological_dimension:
        return np.asarray(mesh.entity_set(target).entity_ids, dtype=np.int64)[:, None]
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    rows = _face_rows(mesh.connectivity, target)
    width = max((np.asarray(row).size for row in rows), default=target + 1)
    return _padded_keys(rows, vertex_ids, width)


def key_rows(table: np.ndarray, queries: np.ndarray, /) -> np.ndarray:
    """Row of each query key in ``table`` (unique rows), or -1 when absent."""

    table_ = np.asarray(table, dtype=np.int64)
    queries_ = np.asarray(queries, dtype=np.int64)
    if table_.ndim != 2 or queries_.ndim != 2:
        raise ValueError("Entity keys must be rank-two arrays.")
    width = max(table_.shape[1], queries_.shape[1])
    table_ = _pad_key_table(table_, width)
    queries_ = _pad_key_table(queries_, width)
    if queries_.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if table_.shape[0] == 0:
        return np.full((queries_.shape[0],), -1, dtype=np.int64)
    combined = np.concatenate((table_, queries_), axis=0)
    _, inverse = np.unique(combined, axis=0, return_inverse=True)
    inverse = inverse.reshape((-1,))
    owner = np.full((int(np.max(inverse)) + 1,), -1, dtype=np.int64)
    owner[inverse[: table_.shape[0]]] = np.arange(table_.shape[0], dtype=np.int64)
    return owner[inverse[table_.shape[0] :]]


def _target_entity_ids(
    source: CellMesh,
    probe: CellMesh,
    dimension: int,
    prescribed: tuple[PrescribedEntityIds, ...],
    /,
) -> np.ndarray:
    """Preserve IDs by identity key and allocate only genuinely new entities."""

    source_keys = entity_keys(source, dimension)
    source_ids = np.asarray(source.entity_set(dimension).entity_ids, dtype=np.int64)
    target_keys = entity_keys(probe, dimension)
    width = max(source_keys.shape[1], target_keys.shape[1])
    source_keys = _pad_key_table(source_keys, width)
    target_keys = _pad_key_table(target_keys, width)
    rows = key_rows(source_keys, target_keys)
    identifiers = np.where(rows >= 0, source_ids[np.maximum(rows, 0)], -1)
    records = tuple(value for value in prescribed if value.dimension == dimension)
    keys = np.concatenate(
        [
            _pad_key_table(np.asarray(value.keys, dtype=np.int64), width)
            for value in records
        ]
        or [np.zeros((0, width), dtype=np.int64)]
    )
    restored = np.concatenate(
        [np.asarray(value.ids, dtype=np.int64) for value in records]
        or [np.zeros((0,), dtype=np.int64)]
    )
    if (
        keys.shape != (restored.size, width)
        or np.unique(restored).size != restored.size
        or np.any(key_rows(source_keys, keys) >= 0)
        or np.intersect1d(restored, source_ids).size
    ):
        raise ValueError("Prescribed entity IDs must name new, unique entities.")
    matched = key_rows(keys, target_keys)
    restoring = np.flatnonzero(matched >= 0)
    identifiers[restoring] = restored[matched[restoring]]
    new = np.flatnonzero(identifiers < 0)
    order = np.lexsort(target_keys[new].T[::-1])
    start = int(max(np.max(source_ids, initial=-1), np.max(restored, initial=-1))) + 1
    identifiers[new[order]] = np.arange(start, start + new.size, dtype=np.int64)
    return identifiers


def _build_mesh(
    source: CellMesh,
    edit: CellTopologyEdit,
    numeric_version: str,
    entity_ids: dict[int, np.ndarray] | None,
    /,
) -> CellMesh:
    blocks: list[CellBlock | PolyhedralBlock] = []
    entries: list[tuple[str, np.ndarray] | tuple[np.ndarray, ...]] = []
    cell_ids: list[np.ndarray] = []
    has_polyhedra = False
    for block in edit.blocks:
        ids = np.asarray(block.cell_ids, dtype=np.int64)
        if (
            ids.ndim != 1
            or not ids.size
            or np.any(ids < 0)
            or np.unique(ids).size != ids.size
        ):
            raise ValueError(
                "Each topology edit block needs unique nonnegative cell IDs."
            )
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        if isinstance(block, PolyhedralTopologyEditBlock):
            connectivity = block.connectivity
            if not isinstance(connectivity, PolyhedralConnectivity):
                raise TypeError("Polyhedral topology blocks need PolyhedralConnectivity.")
            co = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
            cf = np.asarray(connectivity.cell_face_values, dtype=np.int64)
            signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int8)
            fo = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
            fv = np.asarray(connectivity.face_vertex_values, dtype=np.int32)
            if co.shape != (ids.size + 1,):
                raise ValueError("Polyhedral connectivity must align with its cell IDs.")
            loops_by_cell: list[tuple[np.ndarray, ...]] = []
            vertex_rows: list[np.ndarray] = []
            for row in order.tolist():
                loops = tuple(
                    fv[fo[int(cf[position])] : fo[int(cf[position]) + 1]][
                        :: int(signs[position])
                    ]
                    for position in range(int(co[row]), int(co[row + 1]))
                )
                if len(loops) < 4:
                    raise ValueError(
                        "A polyhedral cell needs at least four oriented faces."
                    )
                loops_by_cell.append(loops)
                vertex_rows.append(np.unique(np.concatenate(loops)))
            if len({row.size for row in vertex_rows}) != 1:
                raise ValueError("Split polyhedral edit blocks by vertex arity.")
            blocks.append(
                PolyhedralBlock(block.name, np.stack(vertex_rows), global_ids=ids)
            )
            entries.extend(loops_by_cell)
            has_polyhedra = True
        else:
            cells = np.asarray(block.cells, dtype=np.int32)
            if cells.ndim != 2 or cells.shape[0] != ids.size:
                raise ValueError("Fixed-family block cells must align with cell IDs.")
            cells = cells[order]
            blocks.append(CellBlock(block.name, block.cell_kind, cells, global_ids=ids))
            entries.append((block.cell_kind, cells))
        cell_ids.append(ids)
    if not blocks:
        raise ValueError("A topology edit must retain at least one target cell block.")
    if source.topological_dimension == 2:
        # Polygonal meshes order blocks by increasing arity; the stable sort keeps
        # the edit's block order among equal arities.
        blocks.sort(key=lambda block: block.arity)
    packed = (
        polyhedral_connectivity(
            entries,
            np.asarray(edit.coordinates).shape[0],
            vertex_global_ids=edit.vertex_global_ids,
            cell_global_ids=np.concatenate(cell_ids),
            edge_global_ids=None if entity_ids is None else entity_ids.get(1),
            face_global_ids=None if entity_ids is None else entity_ids.get(2),
        )
        if has_polyhedra
        else None
    )
    return CellMesh(
        edit.coordinates,
        tuple(blocks),
        vertex_global_ids=edit.vertex_global_ids,
        entity_global_ids=entity_ids,
        numeric_version=numeric_version,
        polyhedral_connectivity=packed,
    )


def _entity_lineage(
    source: CellMesh,
    target: CellMesh,
    relations: EntityRelations,
    /,
) -> EntityLineage:
    dimension = relations.dimension
    source_keys = entity_keys(source, dimension)
    target_keys = entity_keys(target, dimension)
    width = max(
        source_keys.shape[1],
        target_keys.shape[1],
        np.shape(relations.source_keys)[1],
        np.shape(relations.target_keys)[1],
    )
    source_keys = _pad_key_table(source_keys, width)
    target_keys = _pad_key_table(target_keys, width)
    explicit_source_keys = _pad_key_table(relations.source_keys, width)
    explicit_target_keys = _pad_key_table(relations.target_keys, width)
    source_ids = np.asarray(source.entity_set(dimension).entity_ids, dtype=np.int64)
    target_ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
    explicit_sources = key_rows(source_keys, explicit_source_keys)
    explicit_targets = key_rows(target_keys, explicit_target_keys)
    if np.any(explicit_sources < 0) or np.any(explicit_targets < 0):
        raise ValueError(
            f"Dimension-{dimension} relations reference entities outside the meshes."
        )
    related_targets = np.zeros((target_ids.size,), dtype=np.bool_)
    related_targets[explicit_targets] = True
    related_sources = np.zeros((source_ids.size,), dtype=np.bool_)
    related_sources[explicit_sources] = True
    implicit = key_rows(source_keys, target_keys)
    explicit_identity = np.zeros((target_ids.size,), dtype=np.bool_)
    explicit_identity[
        explicit_targets[implicit[explicit_targets] == explicit_sources]
    ] = True
    preserved = np.flatnonzero((implicit >= 0) & ~explicit_identity)
    related_sources[implicit[preserved]] = True
    related_targets[preserved] = True
    sources = np.concatenate(
        (source_ids[explicit_sources], source_ids[implicit[preserved]])
    )
    targets = np.concatenate((target_ids[explicit_targets], target_ids[preserved]))
    kinds = np.concatenate(
        (
            np.asarray(relations.kinds, dtype=np.int32),
            np.full(preserved.shape, int(EntityLineageKind.PRESERVED), dtype=np.int32),
        )
    )
    order = np.lexsort((kinds, targets, sources))
    return EntityLineage(
        dimension,
        source.entity_set(dimension).entity_set_id,
        target.entity_set(dimension).entity_set_id,
        sources[order],
        targets[order],
        kinds[order],
        created_target_ids=np.sort(target_ids[~related_targets]),
        deleted_source_ids=np.sort(source_ids[~related_sources]),
    )


def _rebind_periodic_topology(
    mesh: CellMesh,
    periodic_topology: PeriodicMeshTopology,
    /,
    *,
    exact_coordinate_source: ExactPeriodicCoordinateSource | None = None,
) -> CellMesh:
    """Rebuild the quotient against this exact lifted mesh and vertex order.

    The descriptor stores only its exact coordinate witness identity, so a
    witness not carried by its bound geometry is supplied again here.
    """
    if (
        exact_coordinate_source is not None
        and exact_coordinate_source.source_map_id
        != periodic_topology.coordinate_source_map_id
    ):
        raise ValueError("Periodic rebinding received another exact coordinate witness.")
    entity_ids = {
        dimension: np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        for dimension in range(1, mesh.topological_dimension + 1)
    }
    polyhedral = (
        mesh.connectivity
        if isinstance(mesh.connectivity, PolyhedralConnectivity)
        else None
    )
    lifted = CellMesh(
        mesh.coordinates,
        mesh.blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        entity_global_ids=entity_ids,
        polyhedral_connectivity=polyhedral,
        numeric_version=mesh.numeric_version,
    )
    candidate_vertex_ids = np.asarray(
        periodic_topology.lifted_entity_ids[0], dtype=np.int64
    )
    target_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    order = np.argsort(candidate_vertex_ids, kind="stable")
    positions = np.searchsorted(candidate_vertex_ids[order], target_vertex_ids)
    if (
        candidate_vertex_ids.shape != target_vertex_ids.shape
        or np.unique(candidate_vertex_ids).size != candidate_vertex_ids.size
        or np.unique(target_vertex_ids).size != target_vertex_ids.size
        or np.any(positions >= candidate_vertex_ids.size)
        or not np.array_equal(
            candidate_vertex_ids[order][
                np.minimum(positions, candidate_vertex_ids.size - 1)
            ],
            target_vertex_ids,
        )
    ):
        raise ValueError(
            "Periodic quotient rebinding requires the same stable lifted vertex IDs."
        )
    candidate_rows = order[positions]
    target_row_by_id = {
        int(identifier): row for row, identifier in enumerate(target_vertex_ids.tolist())
    }
    candidate_representatives = np.asarray(
        periodic_topology.vertex_representatives, dtype=np.int64
    )
    root_ids = candidate_vertex_ids[candidate_representatives[candidate_rows]]
    target_representatives = np.asarray(
        [target_row_by_id[int(identifier)] for identifier in root_ids],
        dtype=np.int32,
    )
    target_shifts = np.asarray(periodic_topology.vertex_shifts, dtype=np.int32)[
        candidate_rows
    ]
    quotient_ids = {
        degree: np.asarray(entity.entity_ids, dtype=np.int64)
        for degree, entity in enumerate(periodic_topology.quotient.entity_sets)
    }
    allocator = {
        degree: int(next_id)
        for degree, next_id in enumerate(periodic_topology.allocator_next_ids)
        if next_id >= 0
    }
    rebound_topology = PeriodicMeshTopology(
        lifted,
        periodic_topology.cell,
        target_representatives,
        target_shifts,
        entity_global_ids=quotient_ids,
        entity_allocator_next_ids=allocator or None,
        actual_geometry=periodic_topology._actual_geometry,
        exact_coordinate_source=exact_coordinate_source,
    )
    return CellMesh(
        lifted.coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        entity_global_ids=entity_ids,
        polyhedral_connectivity=polyhedral,
        numeric_version=lifted.numeric_version,
        periodic_topology=rebound_topology,
    )


def bind_periodic_topology_edit(
    source: CellMesh,
    target: CellMesh,
    stencil: VertexInterpolationStencil,
    edit: CellTopologyEdit,
    periodic_topology: object,
    /,
    *,
    exact_coordinate_source: ExactPeriodicCoordinateSource | None = None,
) -> tuple[CellMesh, MeshLineage, VertexInterpolationStencil]:
    """Bind a fully constructed exact-coordinate target and rebuild its lineage."""
    if source.periodic_topology is None or target.periodic_topology is not None:
        raise ValueError(
            "Periodic rebinding requires a periodic source and an unbound target."
        )
    if not isinstance(stencil, VertexInterpolationStencil) or not isinstance(
        edit, CellTopologyEdit
    ):
        raise TypeError(
            "Periodic topology edit requires its exact stencil and topology edit."
        )
    if not isinstance(periodic_topology, PeriodicMeshTopology):
        raise TypeError("periodic_topology must be PeriodicMeshTopology.")
    bound = _rebind_periodic_topology(
        target, periodic_topology, exact_coordinate_source=exact_coordinate_source
    )
    rebound_stencil = VertexInterpolationStencil(
        source.entity_set(0).entity_set_id,
        bound.entity_set(0).entity_set_id,
        np.asarray(stencil.target_global_ids),
        np.asarray(stencil.source_global_ids),
        np.asarray(stencil.weights),
        np.asarray(stencil.valid),
        preserves_constants=stencil.preserves_constants,
        exact_weights=stencil.exact_weights,
    )
    lineage = MeshLineage(
        source.topology_id,
        bound.topology_id,
        tuple(_entity_lineage(source, bound, value) for value in edit.relations),
    )
    return bound, lineage, rebound_stencil


def assemble_topology_edit(
    source: CellMesh,
    edit: CellTopologyEdit,
    /,
    *,
    numeric_version: str,
    bind_periodic: bool = True,
) -> tuple[CellMesh, MeshLineage, VertexInterpolationStencil]:
    """Build a generalized target mesh, complete lineage, and vertex stencil."""

    if not isinstance(source, CellMesh):
        raise TypeError("source must be CellMesh.")
    if not isinstance(bind_periodic, bool):
        raise TypeError("bind_periodic must be bool.")
    if not isinstance(edit, CellTopologyEdit):
        raise TypeError("edit must be CellTopologyEdit.")
    dimension = source.topological_dimension
    if tuple(value.dimension for value in edit.relations) != tuple(range(dimension + 1)):
        raise ValueError("A topology edit needs one relation record per dimension.")
    source_kinds = {block.name: block.cell_kind for block in source.blocks}
    for block in edit.blocks:
        if (
            block.source_kind is not None
            and source_kinds.get(block.name) != block.source_kind
        ):
            raise ValueError(
                f"Topology edit block {block.name!r} has a mismatched source family."
            )
    probe = _build_mesh(source, edit, numeric_version, None)
    entity_ids = {
        degree: _target_entity_ids(source, probe, degree, edit.prescribed_entity_ids)
        for degree in range(1, dimension)
    }
    target = _build_mesh(source, edit, numeric_version, entity_ids or None)
    if edit.refinement is not None:
        _validate_nested_witnesses(target, source, edit.refinement)
        _validate_nested_stencil(target, source, edit, edit.refinement)
    if edit.coarsening is not None:
        _validate_nested_witnesses(source, target, edit.coarsening)
    if edit.refinement is not None and edit.coarsening is not None:
        if np.intersect1d(
            np.asarray(edit.refinement.fine_cell_ids, dtype=np.int64),
            np.asarray(edit.coarsening.coarse_cell_ids, dtype=np.int64),
        ).size:
            raise ValueError("A target cell cannot be both refined and coarsened.")
    if edit.shared_faces is not None:
        _require_shared_faces(target, edit.shared_faces)
    preserves_constants = _edit_stencil_preserves_constants(edit)
    stencil = VertexInterpolationStencil(
        source.entity_set(0).entity_set_id,
        target.entity_set(0).entity_set_id,
        edit.vertex_global_ids,
        edit.stencil_sources,
        edit.stencil_weights,
        edit.stencil_valid,
        preserves_constants=preserves_constants,
        exact_weights=edit.stencil_exact_weights,
    )
    if source.periodic_topology is not None and bind_periodic:
        from ._periodic import periodic_topology_after_edit

        periodic_topology = periodic_topology_after_edit(
            source,
            target,
            stencil,
            vertex_orbit_keys=edit.vertex_orbit_keys,
            vertex_windings=edit.vertex_windings,
            source_entity_degree=edit.source_entity_degree,
        )
        if periodic_topology is None:
            raise ValueError(
                "Periodic topology edits must rebuild their quotient descriptor."
            )
        return bind_periodic_topology_edit(
            source, target, stencil, edit, periodic_topology
        )
    lineage = MeshLineage(
        source.topology_id,
        target.topology_id,
        tuple(_entity_lineage(source, target, value) for value in edit.relations),
    )
    return target, lineage, stencil


__all__ = [
    "CellTopologyEdit",
    "EntityRelations",
    "PolyhedralTopologyEditBlock",
    "PrescribedEntityIds",
    "SharedFaceWitnesses",
    "TopologyEditBlock",
    "bind_periodic_topology_edit",
    "assemble_topology_edit",
    "entity_keys",
    "key_rows",
    "nested_reference_witnesses_from_edit",
    "reference_cell_maps_from_edit",
    "source_family_blocks",
]
