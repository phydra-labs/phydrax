#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Private preparation of scientific nested reference integration pairs."""

from __future__ import annotations

import hashlib
import weakref
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, InitVar
from fractions import Fraction
from typing import final, NamedTuple, NoReturn

import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

import phydrax.discretization._cell_geometry_transition_types as transition_types_owner
import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
from phydrax.discretization._reference_cell_types import CellGeometryElement
from phydrax.linalg._exact import determinant_exact, solve_exact

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..geometry._rational import rational_vector
from ..linalg import SmallLinearSolvePlan, solve_small_linear
from ..typing import checked
from . import (
    _cell_geometry as imported_cell_geometry,
    _coordinate_enclosure as imported_coordinate_enclosure,
    _nested_reference_types as reference_pair_types,
)
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_validity import cell_geometry_id
from ._cell_mesh import CellMesh
from ._coordinate_budget import CoordinateEnclosureBudget
from ._coordinate_enclosure import (
    affine_arguments,
    Polynomial,
    require_rational_reference_scalar,
)
from ._exact_algebraic_geometry import RadicalElement
from ._reference_cell import reference_cell_topology


type _PartitionKey = tuple[str, frozenset[tuple[tuple[object, ...], int]]]
type NestedReferencePieces = tuple[
    reference_pair_types.NestedReferencePair
    | reference_pair_types.PolynomialReferencePair,
    ...,
]


class _ReferenceMap(NamedTuple):
    kind: str
    arguments: tuple[Polynomial, ...]


def _fraction_vertices(kind: str) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(value) for value in row)
        for row in reference_cell_topology(kind).vertices
    )


def _dot(first: Sequence[Fraction], second: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _normal(
    points: Sequence[Sequence[Fraction]],
    center: Sequence[Fraction],
) -> tuple[Fraction, ...]:
    dimension = len(points[0])
    if dimension == 1:
        return (Fraction(1) if points[0][0] > center[0] else Fraction(-1),)
    first = tuple(b - a for a, b in zip(points[0], points[1], strict=True))
    if dimension == 2:
        normal: tuple[Fraction, ...] = (first[1], -first[0])
    else:
        second = tuple(b - a for a, b in zip(points[0], points[2], strict=True))
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
    outward = tuple(a - b for a, b in zip(points[0], center, strict=True))
    return normal if _dot(normal, outward) > 0 else tuple(-value for value in normal)


def _reference_measure(kind: str) -> Fraction:
    return {
        "interval": Fraction(1),
        "triangle": Fraction(1, 2),
        "quadrilateral": Fraction(1),
        "tetrahedron": Fraction(1, 6),
        "prism": Fraction(1, 2),
        "hexahedron": Fraction(1),
        "pyramid": Fraction(1, 3),
    }[kind]


def _reference_piece_measure(
    kind: str,
    arguments: tuple[Polynomial, ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    affine = all(sum(index) <= 1 for value in arguments for index in value)
    if affine:
        coefficients = tuple(
            tuple(
                value.get(
                    tuple(int(axis == column) for axis in range(dimension)),
                    Fraction(0),
                )
                for column in range(dimension)
            )
            for value in arguments
        )
        determinant = determinant_exact(coefficients)
        piece_measure = determinant * _reference_measure(kind)
    else:
        if (
            dimension != 2
            or kind != "quadrilateral"
            or any(
                any(exponent > 1 for exponent in index)
                for value in arguments
                for index in value
            )
        ):
            raise ValueError(
                "nested_reference.arguments must define bilinear quadrilateral charts for nonlinear partitions; got unsupported polynomial action"
            )
        jacobian = imported_coordinate_enclosure.determinant(
            tuple(
                tuple(
                    imported_coordinate_enclosure.derivative(value, axis, budget=budget)
                    for axis in range(2)
                )
                for value in arguments
            ),
            budget=budget,
        )
        determinant = min(
            imported_coordinate_enclosure.bernstein_coefficients(
                jacobian, "box", 2, budget=budget
            )
        )
        piece_measure = mapped_measure_owner.mapped_polynomial_integral_fraction(
            jacobian, kind, budget=budget
        )
    if determinant <= 0:
        raise ValueError(
            "nested_reference.jacobian must be positive on every fine cell; got nonpositive bound"
        )

    return require_rational_reference_scalar(piece_measure)


def _require_partition_facets(
    faces: dict[tuple[tuple[Fraction, ...], ...], list[tuple[Fraction, ...]]],
    planes: Sequence[tuple[tuple[Fraction, ...], Fraction]],
) -> None:
    for points, normals in faces.items():
        if len(normals) == 2:
            if _dot(normals[0], normals[1]) >= 0:
                raise ValueError(
                    "nested_reference.interior_facets must have opposite orientations; got inconsistent normals"
                )
        elif len(normals) == 1:
            if not any(
                all(_dot(normal, point) == bound for point in points)
                and _dot(normal, normals[0]) > 0
                for normal, bound in planes
            ):
                raise ValueError(
                    "nested_reference.boundary_facets must lie on the outward parent boundary; got unpaired interior facet"
                )
        else:
            raise ValueError(
                "nested_reference.facets must have manifold multiplicity; got overlapping facets"
            )


def _require_reference_partition(
    coarse_kind: str,
    fine_maps: Sequence[_ReferenceMap],
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Prove an oriented algebraic reference tiling, rather than sampling it.

    Positive fine cells are contained in the convex parent. Every interior
    polygon is paired with its opposite orientation, and unpaired polygons
    lie on the parent boundary. The closed outward boundary chain therefore
    has constant nonnegative integer multiplicity on that boundary; the algebraic
    reference-volume equality fixes it to one. Its winding number is one in
    the parent and zero outside, so positive cells cannot leave gaps/overlap.
    """
    vertices = _fraction_vertices(coarse_kind)
    dimension = len(vertices[0])
    center = tuple(
        sum((row[axis] for row in vertices), Fraction(0)) / len(vertices)
        for axis in range(dimension)
    )
    planes: list[tuple[tuple[Fraction, ...], Fraction]] = []
    for face in reference_cell_topology(coarse_kind).entities[dimension - 1]:
        points = tuple(vertices[index] for index in face)
        normal = _normal(points, center)
        planes.append((normal, _dot(normal, points[0])))
    faces: dict[tuple[tuple[Fraction, ...], ...], list[tuple[Fraction, ...]]] = {}
    measure = Fraction(0)

    for kind, arguments in fine_maps:
        if len(arguments) != dimension:
            raise ValueError(
                "nested_reference.dimension must agree for every fine map; got incompatible argument count"
            )
        piece_measure = _reference_piece_measure(
            kind, arguments, dimension, budget=budget
        )
        fine = tuple(
            tuple(
                require_rational_reference_scalar(
                    imported_coordinate_enclosure.evaluate(value, vertex, budget=budget)
                )
                for value in arguments
            )
            for vertex in _fraction_vertices(kind)
        )
        if any(_dot(normal, point) > bound for point in fine for normal, bound in planes):
            raise ValueError(
                "nested_reference.corners must lie inside the parent reference cell; got exterior corner"
            )
        fine_center = tuple(
            sum((row[axis] for row in fine), Fraction(0)) / len(fine)
            for axis in range(dimension)
        )
        for face in reference_cell_topology(kind).entities[dimension - 1]:
            points = tuple(fine[index] for index in face)
            faces.setdefault(tuple(sorted(points)), []).append(
                _normal(points, fine_center)
            )
        measure += piece_measure
    if measure != _reference_measure(coarse_kind):
        raise ValueError(
            "nested_reference.partition_measure must equal the parent measure; got unequal rational measures"
        )
    _require_partition_facets(faces, planes)


def _pattern_bytes(
    piece: reference_pair_types.NestedReferencePair | tuple[Polynomial, ...], /
) -> int:
    """Upper bytes of one child-law key row, known before the row is built."""
    if isinstance(piece, reference_pair_types.NestedReferencePair):
        # Row tuple, two binary64 byte strings and their contiguous staging copies.
        return 320 + 16 * (int(np.size(piece.matrix)) + int(np.size(piece.offset)))
    return 256 + 128 * sum(len(value) for value in piece)


def _pattern_row(
    kind: str,
    piece: reference_pair_types.NestedReferencePair | tuple[Polynomial, ...],
    /,
) -> tuple[object, ...]:
    """Exact identity of one fine reference law.

    A dyadic affine law is its binary64 arrays; retained rational rows complete
    a nondyadic one, so equal rows are equal rational maps with equal rounding
    validation. Polynomial laws are their exponent-ordered exact coefficients.
    """
    if isinstance(piece, reference_pair_types.NestedReferencePair):
        return (
            kind,
            np.ascontiguousarray(piece.matrix, dtype=np.float64).tobytes(),
            np.ascontiguousarray(piece.offset, dtype=np.float64).tobytes(),
            piece.exact_matrix,
            piece.exact_offset,
        )
    return kind, tuple(tuple(sorted(value.items())) for value in piece)


def _new_partition_key(
    coarse_kind: str,
    pieces: Sequence[
        tuple[str, reference_pair_types.NestedReferencePair | tuple[Polynomial, ...]]
    ],
    proven: set[_PartitionKey],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PartitionKey | None:
    """Key one parent tiling by its exact child-law multiset; ``None`` if proved.

    `_require_reference_partition` depends only on the coarse reference kind
    and the multiset of exact fine laws (measure sum, corner containment and
    facet pairing), never on physical cells or witness order. Forming a key is
    admitted and charged before any row exists; a new key stays live until the
    caller's memo scope ends, and an already proved key releases its bytes.
    """
    storage = 256 + sum(_pattern_bytes(piece) for _, piece in pieces)
    budget.reserve(len(pieces), storage)
    key = (
        coarse_kind,
        frozenset(Counter(_pattern_row(kind, piece) for kind, piece in pieces).items()),
    )
    if key in proven:
        budget.release_temporary_storage(storage)
        return None
    return key


class _ReferenceRecord(NamedTuple):
    source: int
    target: int
    fine_is_source: bool
    corners: NDArray[np.float64] | None
    homogeneous: NDArray[np.float64] | None = None


class _ReferenceRecords(StrictModule):
    records: tuple[_ReferenceRecord, ...]
    source_kinds: tuple[str, ...]
    target_kinds: tuple[str, ...]


def _transition_reference_records(
    transition: transition_types_owner.CellGeometryTransition,
    source_index: dict[int, int],
    target_index: dict[int, int],
    *,
    nonlinear: bool,
    budget: CoordinateEnclosureBudget,
) -> tuple[_ReferenceRecord, ...]:
    records: list[_ReferenceRecord] = []
    parent_bank = transition_types_owner.require_reference_homogeneous(
        np.asarray(transition.parent_reference_vertices),
        transition.parent_reference_homogeneous,
        "nested_reference.geometry_transition.parent",
        budget=budget,
    )
    coarse_bank = transition_types_owner.require_reference_homogeneous(
        np.asarray(transition.coarsened_reference_vertices),
        transition.coarsened_reference_homogeneous,
        "nested_reference.geometry_transition.coarsened",
        budget=budget,
    )
    for row, (target_id, source_id, corners) in enumerate(
        zip(
            np.asarray(transition.target_cell_ids),
            np.asarray(transition.parent_cell_ids),
            np.asarray(transition.parent_reference_vertices),
            strict=True,
        )
    ):
        include = int(source_id) >= 0 if nonlinear else source_id >= 0
        if (not nonlinear or include) and int(target_id) not in target_index:
            raise ValueError(
                "nested_reference.target_cell must name a present scientific cell; got absent ID"
            )
        if include:
            if int(source_id) not in source_index:
                raise ValueError(
                    "nested_reference.source_cell must name a present scientific cell; got absent ID"
                )
            records.append(
                _ReferenceRecord(
                    source_index[int(source_id)],
                    target_index[int(target_id)],
                    False,
                    corners,
                    None if parent_bank is None else parent_bank[row],
                )
            )
    for row, (source_id, target_id, corners) in enumerate(
        zip(
            np.asarray(transition.coarsened_cell_ids),
            np.asarray(transition.coarsened_into_ids),
            np.asarray(transition.coarsened_reference_vertices),
            strict=True,
        )
    ):
        if int(source_id) not in source_index or int(target_id) not in target_index:
            raise ValueError(
                "nested_reference.coarsening must name present scientific cells; got absent ID"
            )
        records.append(
            _ReferenceRecord(
                source_index[int(source_id)],
                target_index[int(target_id)],
                True,
                corners,
                None if coarse_bank is None else coarse_bank[row],
            )
        )
    return tuple(records)


def _coarsening_reference_records(
    witnesses: transition_types_owner.NestedReferenceWitnesses,
    source_index: dict[int, int],
    target_index: dict[int, int],
    *,
    nonlinear: bool,
    budget: CoordinateEnclosureBudget,
) -> tuple[_ReferenceRecord, ...]:
    fine_ids = transition_types_owner.require_int64_witness(
        witnesses.fine_cell_ids, "nested_reference.coarsening.fine_cell_ids"
    )
    if not nonlinear and np.unique(fine_ids).size != fine_ids.size:
        raise ValueError(
            "nested_reference.coarsening must name each source cell once; got repeated ID"
        )
    vertices = transition_types_owner.require_binary64_witness(
        witnesses.fine_reference_vertices,
        "nested_reference.coarsening.fine_reference_vertices",
    )
    bank = transition_types_owner.require_reference_homogeneous(
        vertices,
        witnesses.fine_reference_homogeneous,
        "nested_reference.coarsening",
        budget=budget,
    )
    coarse_ids = transition_types_owner.require_int64_witness(
        witnesses.coarse_cell_ids, "nested_reference.coarsening.coarse_cell_ids"
    )
    records: list[_ReferenceRecord] = []
    for row, (source_id, target_id, corners) in enumerate(
        zip(
            fine_ids,
            coarse_ids,
            vertices,
            strict=True,
        )
    ):
        if int(source_id) not in source_index or int(target_id) not in target_index:
            raise ValueError(
                "nested_reference.coarsening must name present scientific cells; got absent ID"
            )
        records.append(
            _ReferenceRecord(
                source_index[int(source_id)],
                target_index[int(target_id)],
                True,
                np.asarray(corners),
                None if bank is None else bank[row],
            )
        )
    return tuple(records)


def _refinement_reference_records(
    parents_: ArrayLike | None,
    corners_: ArrayLike | None,
    homogeneous_: ArrayLike | None,
    source_count: int,
    target_count: int,
    dimension: int,
    *,
    nonlinear: bool,
    budget: CoordinateEnclosureBudget,
) -> tuple[_ReferenceRecord, ...]:
    if parents_ is None or (not nonlinear and corners_ is None):
        raise ValueError(
            "nested_reference.refinement must provide parent cells and affine reference corners; got missing input"
        )
    parents = np.asarray(parents_)
    if parents.shape != (target_count,) or not np.issubdtype(parents.dtype, np.integer):
        raise ValueError(
            "nested_reference.parent_cells must contain one integer source parent per target cell; got incompatible array"
        )
    if np.any(parents < 0) or np.any(parents >= source_count):
        raise ValueError(
            "nested_reference.parent_cells must index present source cells; got out-of-range index"
        )
    corners = (
        None
        if corners_ is None
        else transition_types_owner.require_binary64_witness(
            corners_, "nested_reference.parent_reference_vertices"
        )
    )
    if corners is not None and (
        corners.ndim != 3
        or corners.shape[0] != target_count
        or corners.shape[2] != dimension
    ):
        raise ValueError(
            "nested_reference.parent_reference_vertices must align with target cells and reference dimension; got incompatible array"
        )
    if corners is None and homogeneous_ is not None:
        raise ValueError(
            "nested_reference.parent_reference_homogeneous must accompany parent_reference_vertices; got homogeneous rows without vertices"
        )
    bank = (
        None
        if corners is None
        else transition_types_owner.require_reference_homogeneous(
            corners, homogeneous_, "nested_reference.parent", budget=budget
        )
    )
    return tuple(
        _ReferenceRecord(
            int(parent),
            index,
            False,
            None if corners is None else corners[index],
            None if bank is None else bank[index],
        )
        for index, parent in enumerate(parents)
    )


def _reference_records(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    parent_cells: ArrayLike | None,
    parent_reference_vertices: ArrayLike | None,
    parent_reference_homogeneous: ArrayLike | None,
    geometry_transition: transition_types_owner.CellGeometryTransition | None,
    coarsening_witnesses: transition_types_owner.NestedReferenceWitnesses | None,
    *,
    nonlinear: bool,
    budget: CoordinateEnclosureBudget,
) -> _ReferenceRecords:
    source_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in source_mesh.blocks]
    )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
    )
    source_index = {int(value): index for index, value in enumerate(source_ids)}
    target_index = {int(value): index for index, value in enumerate(target_ids)}
    source_kinds = tuple(
        block.cell_kind for block in source_mesh.blocks for _ in range(block.cell_count)
    )
    target_kinds = tuple(
        block.cell_kind for block in target_mesh.blocks for _ in range(block.cell_count)
    )
    if source_mesh.topological_dimension != target_mesh.topological_dimension:
        raise ValueError(
            "nested_reference.dimension must agree at both endpoints; got different dimensions"
        )
    if geometry_transition is not None:
        if (
            geometry_transition.source_topology_id != source_mesh.topology_id
            or geometry_transition.target_topology_id != target_mesh.topology_id
            or geometry_transition.source_geometry_id != cell_geometry_id(source_geometry)
            or geometry_transition.target_geometry_id != cell_geometry_id(target_geometry)
        ):
            raise ValueError(
                "nested_reference.geometry_transition must bind both endpoint maps; got mismatched geometry or topology ID"
            )
        records = _transition_reference_records(
            geometry_transition,
            source_index,
            target_index,
            nonlinear=nonlinear,
            budget=budget,
        )
    elif coarsening_witnesses is not None:
        records = _coarsening_reference_records(
            coarsening_witnesses,
            source_index,
            target_index,
            nonlinear=nonlinear,
            budget=budget,
        )
    else:
        records = _refinement_reference_records(
            parent_cells,
            parent_reference_vertices,
            parent_reference_homogeneous,
            source_ids.size,
            target_ids.size,
            source_mesh.topological_dimension,
            nonlinear=nonlinear,
            budget=budget,
        )
    if not records and not nonlinear:
        raise ValueError(
            "nested_reference.records must contain integration pieces; got empty reference records"
        )
    if {record.source for record in records} != set(range(source_ids.size)) or {
        record.target for record in records
    } != set(range(target_ids.size)):
        raise ValueError(
            "nested_reference.records must cover source and target cells; got uncovered cells"
        )
    return _ReferenceRecords(records, source_kinds, target_kinds)


class _AffineReferenceGroup(StrictModule):
    indices: tuple[int, ...]
    pairs: tuple[reference_pair_types.NestedReferencePair, ...]


def _solve_affine_reference_group(
    kind: str,
    entries: list[tuple[int, NDArray[np.float64]]],
    records: tuple[_ReferenceRecord, ...],
    frame_indices: dict[str, tuple[int, ...]],
    dimension: int,
) -> _AffineReferenceGroup:
    pairs: list[reference_pair_types.NestedReferencePair] = []
    vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    corners = np.stack([row for _, row in entries])
    selected = list(frame_indices[kind])
    frame = (vertices[selected] - vertices[0]).T
    images = np.swapaxes(corners[:, selected] - corners[:, :1], -1, -2)
    solved = solve_small_linear(
        SmallLinearSolvePlan(dimension),
        np.broadcast_to(frame.T, (len(entries), dimension, dimension)),
        np.swapaxes(images, -1, -2),
    )
    if not np.all(np.asarray(solved.successful)):
        raise ValueError(
            "nested_reference.affine_frame must be nonsingular; got failed linear solve"
        )
    matrices = np.swapaxes(np.asarray(solved.value, dtype=np.float64), -1, -2)
    offsets = corners[:, 0] - matrices @ vertices[0]
    reconstructed = vertices[None] @ np.swapaxes(matrices, -1, -2) + offsets[:, None]
    if not np.array_equal(reconstructed, corners):
        raise ValueError(
            "nested_reference.corners must equal the represented affine map; got unequal reconstructed corners"
        )
    for row, (index, _) in enumerate(entries):
        source, target, fine_is_source = records[index][:3]
        pairs.append(
            reference_pair_types.NestedReferencePair(
                source, target, fine_is_source, matrices[row], offsets[row]
            )
        )

    return _AffineReferenceGroup(tuple(index for index, _ in entries), tuple(pairs))


_SIMPLEX_KINDS = frozenset(("interval", "triangle", "tetrahedron"))


def _exact_affine_reference_pair(
    kind: str,
    record: _ReferenceRecord,
    corners: tuple[tuple[Fraction, ...], ...],
    frame_indices: dict[str, tuple[int, ...]],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> reference_pair_types.NestedReferencePair:
    """Solve one source-owned rational child map; its floats are execution views."""
    vertices = tuple(
        tuple(Fraction(float(value)) for value in vertex)
        for vertex in reference_cell_topology(kind).vertices
    )
    bits = max(
        abs(value.numerator).bit_length() + value.denominator.bit_length()
        for corner in corners
        for value in corner
    )
    # Gauss-Jordan on a d x 2d system plus the reconstruction of every corner.
    budget.reserve(
        4 * dimension**3 + 2 * len(vertices) * dimension**2,
        (dimension + 1) * dimension * (128 + 8 * ((2 * dimension * bits + 7) // 8)),
    )
    selected = frame_indices[kind]
    frame_transposed = [
        [vertices[column][axis] - vertices[0][axis] for axis in range(dimension)]
        for column in selected
    ]
    images_transposed = [
        [corners[column][axis] - corners[0][axis] for axis in range(dimension)]
        for column in selected
    ]
    transposed = solve_exact(
        frame_transposed,
        images_transposed,
        singular_message="nested_reference.affine_frame must be nonsingular; got singular reference frame",
    )
    matrix = tuple(
        tuple(transposed[column][axis] for column in range(dimension))
        for axis in range(dimension)
    )
    offset = tuple(
        corners[0][axis] - _dot(matrix[axis], vertices[0]) for axis in range(dimension)
    )
    if any(
        tuple(offset[axis] + _dot(matrix[axis], vertex) for axis in range(dimension))
        != corner
        for vertex, corner in zip(vertices, corners, strict=True)
    ):
        raise ValueError(
            "nested_reference.corners must equal the represented affine map; got unequal rational corners"
        )
    return reference_pair_types.exact_reference_pair(
        record.source, record.target, record.fine_is_source, matrix, offset
    )


def _affine_reference_pairs(
    prepared: _ReferenceRecords,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[reference_pair_types.NestedReferencePair, ...]:
    records, source_kinds, target_kinds = (
        prepared.records,
        prepared.source_kinds,
        prepared.target_kinds,
    )
    frame_indices = {
        "interval": (1,),
        "triangle": (1, 2),
        "quadrilateral": (1, 3),
        "tetrahedron": (1, 2, 3),
        "prism": (1, 2, 3),
        "hexahedron": (1, 3, 4),
        "pyramid": (1, 3, 4),
    }
    built: list[reference_pair_types.NestedReferencePair | None] = [None] * len(records)
    groups: dict[str, list[tuple[int, NDArray[np.float64]]]] = {}
    for index, record in enumerate(records):
        source, target, fine_is_source, corners, homogeneous = record
        kind = source_kinds[source] if fine_is_source else target_kinds[target]
        if corners is None:
            raise ValueError(
                "nested_reference.corners must provide an affine action; got missing corners"
            )
        if kind not in frame_indices or not np.all(np.isfinite(corners)):
            raise ValueError(
                "nested_reference.corners must define a supported finite map; got unsupported kind or nonfinite values"
            )
        arity = len(reference_cell_topology(kind).vertices)
        if corners.shape[0] < arity or corners.shape[1] != dimension:
            raise ValueError(
                "nested_reference.corners must contain every fine-cell corner; got incompatible array"
            )
        exact_bank = None if homogeneous is None else homogeneous[:arity]
        if exact_bank is not None and transition_types_owner.has_exact_homogeneous_rows(
            exact_bank
        ):
            coarse_kind = target_kinds[target] if fine_is_source else source_kinds[source]
            if coarse_kind not in _SIMPLEX_KINDS:
                raise ValueError(
                    "nested_reference.reference_homogeneous must describe simplex coarse corners; got nonsimplex coarse cell"
                )
            built[index] = _exact_affine_reference_pair(
                kind,
                record,
                transition_types_owner.exact_reference_corners(
                    corners[:arity], exact_bank
                ),
                frame_indices,
                dimension,
                budget=budget,
            )
            continue
        groups.setdefault(kind, []).append((index, np.asarray(corners[:arity])))
    for kind, entries in groups.items():
        solved = _solve_affine_reference_group(
            kind, entries, records, frame_indices, dimension
        )
        for index, pair in zip(solved.indices, solved.pairs, strict=True):
            built[index] = pair
    output: list[reference_pair_types.NestedReferencePair] = []
    for pair in built:
        if pair is None:
            raise ValueError(
                "nested_reference.affine_frame must produce every integration piece; got missing map"
            )
        output.append(pair)
    # Group numeric witnesses first; only one parent's algebraic proof bank is live.
    partitions: dict[
        tuple[bool, int], list[tuple[str, reference_pair_types.NestedReferencePair]]
    ] = {}
    for pair in output:
        coarse = (
            pair.fine_is_source,
            pair.target_cell if pair.fine_is_source else pair.source_cell,
        )
        fine_kind = (
            source_kinds[pair.source_cell]
            if pair.fine_is_source
            else target_kinds[pair.target_cell]
        )
        partitions.setdefault(coarse, []).append((fine_kind, pair))
    # Each exact child-law pattern is proved once; repeated parents only key it.
    proven: set[_PartitionKey] = set()
    with budget.temporary_scope():
        for (fine_is_source, index), witnesses in partitions.items():
            coarse_kind = target_kinds[index] if fine_is_source else source_kinds[index]
            key = _new_partition_key(coarse_kind, witnesses, proven, budget=budget)
            if key is None:
                continue
            with budget.temporary_scope():
                maps = []
                for kind, pair in witnesses:
                    matrix, offset = reference_pair_types.rational_pair_arrays(pair)
                    maps.append(
                        _ReferenceMap(
                            kind, affine_arguments(offset, matrix, budget=budget)
                        )
                    )
                _require_reference_partition(coarse_kind, maps, budget=budget)
            proven.add(key)
    return tuple(output)


@checked
def nested_reference_pairs(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
    *,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    parent_reference_homogeneous: ArrayLike | None = None,
    geometry_transition: transition_types_owner.CellGeometryTransition | None = None,
    coarsening_witnesses: transition_types_owner.NestedReferenceWitnesses | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[reference_pair_types.NestedReferencePair, ...]:
    """Bind immediate refinement/coarsening witnesses to geometry IDs.

    Each matrix/offset maps the finer reference cell into the coarser one.
    The field/volume owner additionally proves equality of the represented
    maps under that composition; this preparation never treats root-source
    coefficients as target nodal coordinates. Source-owned rational homogeneous
    corners (``parent_reference_homogeneous``, or the banks of the transition
    or coarsening witnesses) produce exact rational pairs whose binary64
    matrices are execution views only.
    """
    source_geometry.resolve(source_mesh, budget=budget)
    target_geometry.resolve(target_mesh, budget=budget)
    prepared = _reference_records(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        parent_cells,
        parent_reference_vertices,
        parent_reference_homogeneous,
        geometry_transition,
        coarsening_witnesses,
        nonlinear=False,
        budget=budget,
    )
    return _affine_reference_pairs(
        prepared, source_mesh.topological_dimension, budget=budget
    )


def _require_polynomial_reference_action(
    coarse_arguments: tuple[Polynomial, ...],
    fine_arguments: tuple[Polynomial, ...],
    arguments: tuple[Polynomial, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Prove equality of the reference laws coefficient by coefficient.

    Equal retained-root identities and coefficient banks then imply equal
    physical maps under this action, including for nonlinear coarse charts.
    The certified physical sup-norm difference is therefore zero without
    sampling or expanding the physical coordinate map.
    """
    if any(
        imported_coordinate_enclosure.compose(value, arguments, budget=budget) != expected
        for value, expected in zip(coarse_arguments, fine_arguments, strict=True)
    ):
        raise ValueError(
            "nested_reference.arguments must compose to the fine reference law; got unequal polynomial coefficients"
        )


def _polynomial_reference_pair(
    record: _ReferenceRecord,
    source_cells: tuple[
        tuple[CellGeometryElement, imported_coordinate_enclosure.CoordinateSourceBank],
        ...,
    ],
    target_cells: tuple[
        tuple[CellGeometryElement, imported_coordinate_enclosure.CoordinateSourceBank],
        ...,
    ],
    source_roots: tuple[tuple[str, str, int, tuple[int, ...]], ...],
    target_roots: tuple[tuple[str, str, int, tuple[int, ...]], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> reference_pair_types.PolynomialReferencePair:
    source_index, target_index, fine_is_source, witness, homogeneous = record
    source_element, source_bank = source_cells[source_index]
    target_element, target_bank = target_cells[target_index]
    if (
        source_roots[source_index] != target_roots[target_index]
        or source_bank != target_bank
    ):
        raise ValueError(
            "nested_reference.source_root must retain scientific identity and source coefficients; got stale roots or changed coefficient bank"
        )
    coarse, fine = (
        (target_element, source_element)
        if fine_is_source
        else (source_element, target_element)
    )
    arguments = _polynomial_reference_arguments(coarse, fine, budget=budget)
    vertices = reference_cell_topology(fine.cell_kind).vertices
    if witness is not None:
        points = tuple(
            tuple(
                imported_coordinate_enclosure.evaluate(
                    value,
                    rational_vector(vertex),
                    budget=budget,
                )
                for value in arguments
            )
            for vertex in vertices
        )
        exact = np.asarray(
            [imported_coordinate_enclosure.rounded_point(point) for point in points],
            dtype=np.float64,
        )
        if (
            witness.ndim != 2
            or witness.shape[0] < len(vertices)
            or witness.shape[1] != 2
            or not np.array_equal(exact, witness[: len(vertices)])
        ):
            raise ValueError(
                "nested_reference.corners must equal the rounded polynomial action; got incompatible reference record"
            )
        if homogeneous is not None and any(
            homogeneous[index, :].any()
            and transition_types_owner.homogeneous_reference_point(homogeneous[index, :])
            != point
            for index, point in enumerate(points)
        ):
            raise ValueError(
                "nested_reference.reference_homogeneous must equal the rational polynomial action; got unequal source-owned corner"
            )
    jacobian = imported_coordinate_enclosure.determinant(
        tuple(
            tuple(
                imported_coordinate_enclosure.derivative(value, axis, budget=budget)
                for axis in range(2)
            )
            for value in arguments
        ),
        budget=budget,
    )
    controls = imported_coordinate_enclosure.bernstein_coefficients(
        jacobian,
        "box" if fine.cell_kind == "quadrilateral" else "simplex",
        2,
        budget=budget,
    )
    return reference_pair_types.PolynomialReferencePair(
        source_index,
        target_index,
        fine_is_source,
        arguments,
        source_element.cell_kind,
        target_element.cell_kind,
        (
            require_rational_reference_scalar(min(controls)),
            require_rational_reference_scalar(max(controls)),
        ),
    )


def _polynomial_reference_arguments(
    coarse: CellGeometryElement,
    fine: CellGeometryElement,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...]:

    coarse_root, coarse_expressions = (
        imported_coordinate_enclosure.coordinate_reference_chain(coarse, budget=budget)
    )
    fine_root, fine_expressions = (
        imported_coordinate_enclosure.coordinate_reference_chain(fine, budget=budget)
    )
    if any(
        isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
        for value in (*coarse_expressions, *fine_expressions)
    ):
        raise ValueError(
            "nested_reference.arguments must be polynomial; got rational quotient"
        )
    coarse_arguments = tuple(
        value
        for value in coarse_expressions
        if not isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
    )
    fine_arguments = tuple(
        value
        for value in fine_expressions
        if not isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
    )
    if (
        coarse_root.element_id != fine_root.element_id
        or coarse_root.topological_dimension != 2
        or imported_coordinate_enclosure.coordinate_source_signature(coarse_root)
        != imported_coordinate_enclosure.coordinate_source_signature(fine_root)
    ):
        raise ValueError(
            "nested_reference.source_root must bind one two-dimensional coordinate source; got different root or source signature"
        )
    if any(sum(index) > 1 for value in coarse_arguments for index in value):
        _, expressions = imported_coordinate_enclosure.coordinate_reference_chain(
            fine, ancestor=coarse, budget=budget
        )
        if any(
            isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
            for value in expressions
        ):
            raise ValueError(
                "nested_reference.arguments must be polynomial; got rational quotient"
            )
        arguments = tuple(
            value
            for value in expressions
            if not isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
        )
        _require_polynomial_reference_action(
            coarse_arguments, fine_arguments, arguments, budget=budget
        )
        return arguments
    matrix = [
        [
            value.get(tuple(int(axis == column) for axis in range(2)), Fraction(0))
            for column in range(2)
        ]
        for value in coarse_arguments
    ]
    differences = tuple(
        imported_coordinate_enclosure.add(
            value,
            imported_coordinate_enclosure.constant(
                -old.get((0, 0), Fraction(0)), 2, budget=budget
            ),
            budget=budget,
        )
        for value, old in zip(fine_arguments, coarse_arguments, strict=True)
    )
    indices = sorted({index for value in differences for index in value})
    solved = solve_exact(
        matrix,
        [[value.get(index, Fraction(0)) for index in indices] for value in differences],
    )
    arguments: tuple[Polynomial, ...] = tuple(
        {
            index: coefficient
            for index, coefficient in zip(indices, row, strict=True)
            if coefficient
        }
        for row in solved
    )
    _require_polynomial_reference_action(
        coarse_arguments, fine_arguments, arguments, budget=budget
    )
    return arguments


def _has_nonlinear_reference_action(
    element: CellGeometryElement, *, budget: CoordinateEnclosureBudget
) -> bool:
    """Classify the full authored reference law, not its Python owner kind."""

    if imported_coordinate_enclosure.has_cartesian_reference_chain(element):
        # A chain of physical affine restrictions remains an affine reference
        # action. In particular, the collapsed-cube chart makes an
        # affine pyramid restriction look polynomial; that chart is the root
        # parameterization, not a nonlinear authored action.
        current = element
        while isinstance(current, imported_cell_geometry.RestrictedCellGeometryElement):
            current = current.source_element
        if not isinstance(
            current,
            (
                imported_cell_geometry.PolynomialComposedCellGeometryElement,
                imported_cell_geometry.RationalComposedCellGeometryElement,
            ),
        ):
            return False
        _, arguments = imported_coordinate_enclosure.coordinate_reference_chain(
            element, budget=budget
        )
    else:
        # Full coefficient actions are not Cartesian ancestry. Their existing
        # nested route is still checked against the physical maps.
        arguments = imported_coordinate_enclosure.source_expressions(
            element, budget=budget
        )
        if arguments is None:
            raise ValueError(
                "nested_reference.source_expressions must provide the scalar reference law; got absent source expressions"
            )
    for value in arguments:
        numerator, denominator = imported_coordinate_enclosure.expression_parts(
            value,
            reference_cell_topology(element.cell_kind).dimension,
            budget=budget,
        )
        if any(sum(index) != 0 for index in denominator) or not denominator:
            return True
        if any(sum(index) > 1 for index in numerator):
            return True
    return False


def _rooted_reference_pairs(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    target: CellMesh,
    target_geometry: CellGeometrySpec,
    *,
    parent_cells: ArrayLike | None,
    parent_reference_vertices: ArrayLike | None,
    parent_reference_homogeneous: ArrayLike | None,
    geometry_transition: transition_types_owner.CellGeometryTransition | None,
    coarsening_witnesses: transition_types_owner.NestedReferenceWitnesses | None,
    budget: CoordinateEnclosureBudget,
) -> tuple[NestedReferencePieces, float, bool]:
    """Bind polynomial fine actions and their oriented reference tiling.

    Returns the pieces, their certified physical-map bound, and whether they
    are the affine witness route.
    """
    # Reference semantics belong to each block element; physical cell
    # coefficient banks are authenticated by the whole-map proof below.

    source_cells, target_cells = (
        mapped_measure_owner.mapped_geometry_cells(
            source, source_geometry, budget=budget
        ),
        mapped_measure_owner.mapped_geometry_cells(
            target, target_geometry, budget=budget
        ),
    )
    nonlinear = any(
        _has_nonlinear_reference_action(element, budget=budget)
        for element in target_geometry.elements
    ) or (
        (coarsening_witnesses is not None or geometry_transition is not None)
        and any(
            _has_nonlinear_reference_action(element, budget=budget)
            for element in source_geometry.elements
        )
    )
    if not nonlinear or (
        source.topology_id == target.topology_id
        and cell_geometry_id(source_geometry) == cell_geometry_id(target_geometry)
    ):
        pairs = nested_reference_pairs(
            source,
            source_geometry,
            target,
            target_geometry,
            parent_cells=parent_cells,
            parent_reference_vertices=parent_reference_vertices,
            parent_reference_homogeneous=parent_reference_homogeneous,
            geometry_transition=geometry_transition,
            coarsening_witnesses=coarsening_witnesses,
            budget=budget,
        )
        bound = mapped_measure_owner.certify_nested_geometry_pairs(
            source,
            source_geometry,
            target,
            target_geometry,
            pairs,
            budget=budget,
        )
        return pairs, bound, True
    if (
        source.topological_dimension != 2
        or target.topological_dimension != 2
        or source.ambient_dimension != target.ambient_dimension
    ):
        raise ValueError(
            "nested_reference.dimension must define one common surface coordinate space; got incompatible dimensions"
        )
    prepared = _reference_records(
        source,
        source_geometry,
        target,
        target_geometry,
        parent_cells,
        parent_reference_vertices,
        parent_reference_homogeneous,
        geometry_transition,
        coarsening_witnesses,
        nonlinear=True,
        budget=budget,
    )
    records = prepared.records
    source_roots, target_roots = (
        mapped_measure_owner.mapped_root_identities(source, source_geometry),
        mapped_measure_owner.mapped_root_identities(target, target_geometry),
    )
    built: list[reference_pair_types.PolynomialReferencePair] = []
    partitions: dict[tuple[bool, int], list[_ReferenceMap]] = {}
    for record in records:
        source_index, target_index, fine_is_source = record[:3]
        pair = _polynomial_reference_pair(
            record,
            source_cells,
            target_cells,
            source_roots,
            target_roots,
            budget=budget,
        )
        built.append(pair)
        partitions.setdefault(
            (fine_is_source, target_index if fine_is_source else source_index), []
        ).append(
            _ReferenceMap(
                pair.source_kind if fine_is_source else pair.target_kind, pair.arguments
            )
        )
    proven: set[_PartitionKey] = set()
    with budget.temporary_scope():
        for (fine_is_source, parent), maps in partitions.items():
            coarse_kind = (target_cells if fine_is_source else source_cells)[parent][
                0
            ].cell_kind
            key = _new_partition_key(coarse_kind, maps, proven, budget=budget)
            if key is None:
                continue
            _require_reference_partition(coarse_kind, maps, budget=budget)
            proven.add(key)
    # Every pair proves coarse_reference(action) == fine_reference over the
    # entire cell, with identical physical source coefficients and root IDs.
    # This algebraic equality certifies zero physical-map deviation.
    return tuple(built), 0.0, False


class NestedReferenceBinding(NamedTuple):
    """Endpoint maps and witness data proved by one nested preparation."""

    source_topology_id: str
    target_topology_id: str
    source_mesh_id: str
    target_mesh_id: str
    source_geometry_id: str
    target_geometry_id: str
    witness_id: str


def _witness_array(value: ArrayLike | None, /, *, real: bool = False) -> object:
    """Fingerprint one reference input by the values its preparation reads.

    Nothing is rounded or truncated for the identity. Real banks that binary64
    holds without rounding and integer-dtype identifiers that fit int64 take
    that form, so equal values share one identity across dtypes; the
    preparation reads them through the same validators and refuses any other.
    Every other input keeps its own dtype and content identity.
    """
    if value is None:
        return None
    array = np.asarray(value)
    values = (
        transition_types_owner.exact_binary64_values(array)
        if real
        else transition_types_owner.exact_int64_values(array)
    )
    if values is not None:
        return array_tree_fingerprint(values)
    if not array.dtype.hasobject:
        return array_tree_fingerprint(array)
    return {
        "dtype": "object",
        "shape": list(array.shape),
        "values": [_object_witness_token(item) for item in array.flat],
    }


def _object_witness_token(value: object, /) -> list[str]:
    """Lossless token of one object array entry; integers never pass through decimal."""
    match value:
        case bool():
            return ["bool", str(value)]
        case int():
            return ["int", hex(value)]
        case Fraction():
            return ["fraction", hex(value.numerator), hex(value.denominator)]
        case float():
            return ["float", value.hex()]
        case _:
            return [f"{type(value).__module__}.{type(value).__qualname__}", repr(value)]


@checked
def nested_reference_binding(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    target: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
    *,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    parent_reference_homogeneous: ArrayLike | None = None,
    geometry_transition: transition_types_owner.CellGeometryTransition | None = None,
    coarsening_witnesses: transition_types_owner.NestedReferenceWitnesses | None = None,
) -> NestedReferenceBinding:
    """Identity of every input a nested preparation reads, never a route label."""
    transition = geometry_transition
    coarsening = coarsening_witnesses
    witness_id = canonical_fingerprint(
        {
            "kind": "nested-reference-witnesses",
            "parent_cells": _witness_array(parent_cells),
            "parent_reference_vertices": _witness_array(
                parent_reference_vertices, real=True
            ),
            "parent_reference_homogeneous": _witness_array(
                parent_reference_homogeneous, real=True
            ),
            "geometry_transition": None
            if transition is None
            else {
                "transition": transition.transition_id,
                "policy": transition.policy_id,
                "evidence": transition.evidence.evidence_id,
                "endpoints": [
                    transition.source_topology_id,
                    transition.target_topology_id,
                    transition.source_geometry_id,
                    transition.target_geometry_id,
                ],
                "target_cells": _witness_array(transition.target_cell_ids),
                "parent_cells": _witness_array(transition.parent_cell_ids),
                "parent_reference_vertices": _witness_array(
                    transition.parent_reference_vertices, real=True
                ),
                "parent_reference_homogeneous": _witness_array(
                    transition.parent_reference_homogeneous, real=True
                ),
                "coarsened_cells": _witness_array(transition.coarsened_cell_ids),
                "coarsened_into": _witness_array(transition.coarsened_into_ids),
                "coarsened_reference_vertices": _witness_array(
                    transition.coarsened_reference_vertices, real=True
                ),
                "coarsened_reference_homogeneous": _witness_array(
                    transition.coarsened_reference_homogeneous, real=True
                ),
            },
            "coarsening_witnesses": None
            if coarsening is None
            else {
                "fine_cells": _witness_array(coarsening.fine_cell_ids),
                "coarse_cells": _witness_array(coarsening.coarse_cell_ids),
                "fine_reference_vertices": _witness_array(
                    coarsening.fine_reference_vertices, real=True
                ),
                "fine_reference_homogeneous": _witness_array(
                    coarsening.fine_reference_homogeneous, real=True
                ),
            },
        }
    )
    return NestedReferenceBinding(
        source.topology_id,
        target.topology_id,
        source.mesh_id,
        target.mesh_id,
        cell_geometry_id(source_geometry),
        cell_geometry_id(target_geometry),
        witness_id,
    )


@final
@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class PreparedNestedReference:
    """Certified nested integration pieces bound to their endpoints and witnesses.

    Issued only by `prepare_nested_reference`, passed explicitly to its
    consumers and admitted by `require_nested_reference`; never a PyTree leaf,
    static field, archived record or global cache. Direct construction,
    `dataclasses.replace`, copies and pickling are refused, and admission takes
    only a record this module issued, never a subclass or lookalike, so no
    caller-built record can carry other maps, bounds or charges under an issued
    binding. Affine execution views are read-only arrays over immutable bytes;
    polynomial coefficient dicts are re-verified against their issued digest on
    every admission.
    ``affine`` pieces follow the affine route of `nested_reference_pairs`;
    ``rooted`` pieces followed the rooted classification, so polynomial pieces
    appear exactly for nonlinear reference actions. ``geometry_bound`` is the
    certified physical-map deviation, ``None`` when it was not certified.
    ``work_units``, ``geometry_queries`` and ``storage_bytes`` are the completed
    charges of the preparing ledger; every other ledger admitting the record
    pays them once, so sharing removes repeated work but never hides its cost.
    """

    binding: NestedReferenceBinding
    affine: bool
    rooted: bool
    pairs: NestedReferencePieces
    geometry_bound: float | None
    work_units: int
    geometry_queries: int
    storage_bytes: int
    issuer: InitVar[object]

    def __post_init__(self, issuer: object) -> None:
        if issuer is not _NESTED_REFERENCE_ISSUER:
            raise TypeError(
                "PreparedNestedReference.issuer must be prepare_nested_reference; got direct construction"
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "PreparedNestedReference.record must stay in its issuing process uncopied; got copy or serialization"
        )

    def serves(
        self, binding: NestedReferenceBinding, /, *, rooted: bool, certified: bool
    ) -> bool:
        """Whether these pieces cover ``binding`` on the requested route."""
        return (
            self.binding == binding
            and (self.rooted if rooted else self.affine)
            and (not certified or self.geometry_bound is not None)
        )

    def affine_pieces(self) -> tuple[reference_pair_types.NestedReferencePair, ...]:
        """The affine witness pieces; polynomial preparations are refused."""
        pieces = tuple(
            pair
            for pair in self.pairs
            if isinstance(pair, reference_pair_types.NestedReferencePair)
        )
        if not self.affine or len(pieces) != len(self.pairs):
            raise ValueError(
                "PreparedNestedReference.pairs must be affine witness pieces; got a rooted polynomial preparation"
            )
        return pieces

    def certified_bound(self) -> float:
        """The certified physical-map deviation; uncertified pieces are refused."""
        if self.geometry_bound is None:
            raise ValueError(
                "PreparedNestedReference.geometry_bound must be certified; got an uncertified preparation"
            )
        return self.geometry_bound


@dataclass(slots=True, eq=False)
class _NestedReferenceSeal:
    """Issue-time provenance of one record, held only by this module.

    ``content_id`` digests the polynomial coefficient dicts, the one part of the
    pieces Python cannot freeze; ``ledgers`` weakly names every ledger already
    charged for the completed preparation, starting with the preparing one.
    """

    content_id: str
    ledgers: weakref.WeakSet[CoordinateEnclosureBudget]


_NESTED_REFERENCE_ISSUER = object()
# Weak keys: provenance lives exactly as long as its issued record and retains
# neither pieces nor ledgers.
_issued_nested_references: weakref.WeakKeyDictionary[
    PreparedNestedReference, _NestedReferenceSeal
] = weakref.WeakKeyDictionary()


def _integer_record(*values: object) -> bytearray:
    """Prefix-free integer bytes; a non-integer contributes only its identity."""
    record = bytearray()
    for value in values:
        if isinstance(value, int):
            payload = value.to_bytes(value.bit_length() // 8 + 1, "little", signed=True)
            record += len(payload).to_bytes(8, "little") + payload
        else:
            # No integer payload is 2**64 - 1 bytes long.
            record += b"\xff" * 8 + id(value).to_bytes(8, "little")
    return record


def _polynomial_content_id(pairs: NestedReferencePieces, /) -> str:
    """Digest the polynomial coefficient dicts, the pieces' only mutable content.

    Every position is tagged and length-prefixed, so only equal exponent and
    coefficient rows, in equal order, share a digest.
    """
    digest = hashlib.sha256()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, reference_pair_types.PolynomialReferencePair):
            continue
        digest.update(_integer_record(index, len(pair.arguments)))
        for argument in pair.arguments:
            digest.update(_integer_record(len(argument)))
            for exponents, coefficient in argument.items():
                if type(exponents) is tuple:
                    digest.update(b"T" + _integer_record(len(exponents), *exponents))
                else:
                    digest.update(b"X" + _integer_record(id(exponents)))
                match coefficient:
                    case Fraction():
                        digest.update(
                            b"Q"
                            + _integer_record(
                                coefficient.numerator, coefficient.denominator
                            )
                        )
                    case RadicalElement():
                        digest.update(
                            b"R"
                            + _integer_record(
                                len(coefficient.terms),
                                *(
                                    value
                                    for radicand, rational in coefficient.terms
                                    for value in (
                                        radicand,
                                        rational.numerator,
                                        rational.denominator,
                                    )
                                ),
                            )
                        )
                    case _:
                        digest.update(b"O" + _integer_record(id(coefficient)))
    return digest.hexdigest()


def _immutable_bank(values: list[NDArray[np.float64]], /) -> NDArray[np.float64]:
    """One float64 stack whose only owner is an immutable ``bytes`` buffer."""
    bank = np.stack(values).astype(np.float64, copy=False)
    return np.frombuffer(bank.tobytes(), dtype=np.float64).reshape(bank.shape)


def _sealed_pieces(pairs: NestedReferencePieces, /) -> NestedReferencePieces:
    """Rebind affine execution views to two byte-backed read-only banks.

    A view of an immutable buffer cannot be made writeable again, so no alias of
    a working array can rewrite a proved map after issue.
    """
    affine = [
        (index, pair)
        for index, pair in enumerate(pairs)
        if isinstance(pair, reference_pair_types.NestedReferencePair)
    ]
    if not affine:
        return pairs
    matrices = _immutable_bank([pair.matrix for _, pair in affine])
    offsets = _immutable_bank([pair.offset for _, pair in affine])
    sealed = list(pairs)
    for row, (index, pair) in enumerate(affine):
        sealed[index] = pair._replace(matrix=matrices[row], offset=offsets[row])
    return tuple(sealed)


@checked
def require_nested_reference(
    prepared: PreparedNestedReference,
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    target: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
    *,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    parent_reference_homogeneous: ArrayLike | None = None,
    geometry_transition: transition_types_owner.CellGeometryTransition | None = None,
    coarsening_witnesses: transition_types_owner.NestedReferenceWitnesses | None = None,
    rooted: bool,
    certified: bool,
    budget: CoordinateEnclosureBudget,
) -> PreparedNestedReference:
    """Admit an issued preparation for exactly these endpoints, witnesses and route.

    Refuses any record `prepare_nested_reference` did not issue, any other
    endpoints, witnesses or route, and changed polynomial coefficients. A ledger
    not yet charged for the record pays its completed work, geometry queries
    and storage once; the preparing ledger already paid while proving it.
    """
    seal = (
        _issued_nested_references.get(prepared)
        if type(prepared) is PreparedNestedReference
        else None
    )
    if seal is None:
        raise ValueError(
            "PreparedNestedReference.provenance must be a record issued by prepare_nested_reference; "
            f"got an unissued {type(prepared).__qualname__}"
        )
    binding = nested_reference_binding(
        source,
        source_geometry,
        target,
        target_geometry,
        parent_cells=parent_cells,
        parent_reference_vertices=parent_reference_vertices,
        parent_reference_homogeneous=parent_reference_homogeneous,
        geometry_transition=geometry_transition,
        coarsening_witnesses=coarsening_witnesses,
    )
    if not prepared.serves(binding, rooted=rooted, certified=certified):
        raise ValueError(
            "PreparedNestedReference.binding must match the consumer endpoint maps, witnesses and route; got "
            f"mismatched={tuple(name for name, actual, expected in zip(binding._fields, binding, prepared.binding, strict=True) if actual != expected)!r}, "
            f"route={'rooted' if rooted else 'affine'}, certified={certified}, "
            f"prepared=(affine={prepared.affine}, rooted={prepared.rooted}, "
            f"certified={prepared.geometry_bound is not None})"
        )
    if _polynomial_content_id(prepared.pairs) != seal.content_id:
        raise ValueError(
            "PreparedNestedReference.pairs must keep their issued polynomial coefficients; got modified coefficient dicts"
        )
    if budget not in seal.ledgers:
        budget.reserve(prepared.work_units, prepared.storage_bytes)
        budget.reserve_geometry_queries(prepared.geometry_queries)
        seal.ledgers.add(budget)
    return prepared


@checked
def prepare_nested_reference(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    target: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
    *,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    parent_reference_homogeneous: ArrayLike | None = None,
    geometry_transition: transition_types_owner.CellGeometryTransition | None = None,
    coarsening_witnesses: transition_types_owner.NestedReferenceWitnesses | None = None,
    rooted: bool,
    certify: bool = True,
    budget: CoordinateEnclosureBudget,
) -> PreparedNestedReference:
    """Prepare one transition's nested pieces for every consuming field owner.

    ``rooted=False`` is the affine witness route of `nested_reference_pairs`,
    certified by `certify_nested_geometry_pairs` when ``certify``. ``rooted``
    pieces classify each reference action: affine actions take that same route,
    nonlinear surface actions bind polynomial pieces with an algebraic zero
    deviation. Charges stay on ``budget``; the record retains their totals and
    is the only one `require_nested_reference` admits for these pieces.
    """
    if rooted and not certify:
        raise ValueError(
            "prepare_nested_reference.certify must be true for rooted pieces; got False"
        )
    binding = nested_reference_binding(
        source,
        source_geometry,
        target,
        target_geometry,
        parent_cells=parent_cells,
        parent_reference_vertices=parent_reference_vertices,
        parent_reference_homogeneous=parent_reference_homogeneous,
        geometry_transition=geometry_transition,
        coarsening_witnesses=coarsening_witnesses,
    )
    work = budget.work_units
    storage = budget.retained_basis_bytes + budget.temporary_bytes_upper
    queries = budget.ticket.geometry_queries
    pairs: NestedReferencePieces
    geometry_bound: float | None
    if rooted:
        pairs, bound, affine = _rooted_reference_pairs(
            source,
            source_geometry,
            target,
            target_geometry,
            parent_cells=parent_cells,
            parent_reference_vertices=parent_reference_vertices,
            parent_reference_homogeneous=parent_reference_homogeneous,
            geometry_transition=geometry_transition,
            coarsening_witnesses=coarsening_witnesses,
            budget=budget,
        )
        geometry_bound = bound
    else:
        affine_pairs = nested_reference_pairs(
            source,
            source_geometry,
            target,
            target_geometry,
            parent_cells=parent_cells,
            parent_reference_vertices=parent_reference_vertices,
            parent_reference_homogeneous=parent_reference_homogeneous,
            geometry_transition=geometry_transition,
            coarsening_witnesses=coarsening_witnesses,
            budget=budget,
        )
        geometry_bound = (
            mapped_measure_owner.certify_nested_geometry_pairs(
                source,
                source_geometry,
                target,
                target_geometry,
                affine_pairs,
                budget=budget,
            )
            if certify
            else None
        )
        pairs, affine = affine_pairs, True
    # Consumers share these execution views; none may rewrite a proved map.
    pairs = _sealed_pieces(pairs)
    prepared = PreparedNestedReference(
        binding,
        affine,
        rooted,
        pairs,
        geometry_bound,
        budget.work_units - work,
        max(budget.ticket.geometry_queries - queries, 0),
        max(budget.retained_basis_bytes + budget.temporary_bytes_upper - storage, 0),
        _NESTED_REFERENCE_ISSUER,
    )
    _issued_nested_references[prepared] = _NestedReferenceSeal(
        _polynomial_content_id(pairs), weakref.WeakSet((budget,))
    )
    return prepared
