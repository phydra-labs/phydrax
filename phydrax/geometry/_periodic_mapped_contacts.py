# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact translation-image contacts for mapped periodic cell meshes."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import add, constant
from ..discretization._exact_algebraic_geometry import coefficient_endpoint, ExactScalar
from ..discretization._periodic_cell import PeriodicCell
from ..discretization._periodic_exact import (
    exact_periodic_element,
    exact_periodic_generators,
    identification_orders,
    PeriodicIsometryGroup,
    reduce_periodic_exponents,
)
from ..discretization._periodic_topology import PeriodicMeshTopology
from ._embedding_cells import EmbeddingCell
from ._periodic_image_frame import (
    periodic_overlap_image_exponents,
    prepare_periodic_image_frame,
)


if TYPE_CHECKING:
    from ..discretization._coordinate_enclosure import Expression
    from ._periodic_embedding import PeriodicContactResult, PeriodicVertexKey


def certify_periodic_mapped_contacts(
    mesh: CellMesh,
    cells: tuple[EmbeddingCell, ...],
    topology: PeriodicMeshTopology,
    /,
) -> PeriodicContactResult:
    """Prove mapped translation-image contact by exact stratum separators.

    Outward map bounds enumerate every potentially intersecting image. Each
    image is processed separately and each candidate pair must be separated or
    proven to meet only at its declared lifted stratum.
    """
    from ._global_embedding import (
        _cell_simplices,
        _contact_is_declared,
        _piece_contacts,
        _plane_side_shared_contacts,
        _shared_curved_facet_separation,
        _shared_edge_cone_separation,
        _shared_edge_separation,
        _shared_facet_separation,
        _shared_vertex_separation,
        _stratum_residuals,
        _strict_image_separation,
        _VolumePieceFilter,
    )
    from ._periodic_embedding import (
        _image_winding,
        _lifted_vertex_keys,
        _quotient_faces_pair_oriented,
        PeriodicContactResult,
    )
    from ._restricted_embedding import restriction_trace_equal
    from ._stratum_subdivision import (
        glued_face_monotone,
        glued_facet_monotone,
        StratumSubdivisionCache,
        subdivided_stratum_separation,
    )

    period = topology.cell
    if not isinstance(period, (PeriodicCell, PeriodicIsometryGroup)) or not cells:
        return PeriodicContactResult(
            "unresolved",
            (
                (
                    "periodic_mapped_contact_unresolved",
                    (),
                    "Mapped contact requires a translation group and nonempty cells.",
                ),
            ),
        )
    boundary = _quotient_faces_pair_oriented(mesh, topology)
    if boundary is None:
        return PeriodicContactResult(
            "unresolved",
            (
                (
                    "periodic_quotient_incidence_unresolved",
                    (),
                    "Periodic quotient incidence is unresolved.",
                ),
            ),
        )
    dimension = mesh.ambient_dimension
    if mesh.topological_dimension != dimension:
        return PeriodicContactResult(
            "unresolved",
            (
                (
                    "periodic_mapped_contact_unresolved",
                    (),
                    "Mapped contact requires full-dimensional cells.",
                ),
            ),
            boundary,
        )
    if isinstance(period, PeriodicIsometryGroup):
        generators, _ = exact_periodic_generators(period)
        linear_identity = tuple(
            tuple(Fraction(int(row == column)) for column in range(dimension))
            for row in range(dimension)
        )
        if any(
            tuple(tuple(row[:dimension]) for row in generator[:dimension])
            != linear_identity
            for generator in generators
        ):
            return PeriodicContactResult(
                "unresolved",
                (
                    (
                        "periodic_mapped_contact_unresolved",
                        (),
                        "Mapped isometry contacts require exact translational generators.",
                    ),
                ),
                boundary,
            )
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    fixed = np.asarray(topology.vertex_fixed_generators, dtype=np.bool_)
    ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    orders = tuple(int(value) for value in identification_orders(period))
    key_shifts = reduce_periodic_exponents(shifts, orders)
    key_shifts[fixed] = 0
    fixed_by_root = {
        int(ids[int(representatives[index])]): tuple(
            bool(value) for value in fixed[index]
        )
        for index in range(len(ids))
    }
    keys = tuple(
        _lifted_vertex_keys(cell, ids, representatives, key_shifts) for cell in cells
    )
    if any(len(set(row)) != len(row) for row in keys):
        return PeriodicContactResult(
            "violated",
            (
                (
                    "periodic_cell_lift_collision",
                    (),
                    "A cell has repeated lifted vertex identities.",
                ),
            ),
            boundary,
        )
    lower = np.asarray([cell.image_lower for cell in cells], dtype=np.float64)
    upper = np.asarray([cell.image_upper for cell in cells], dtype=np.float64)
    try:
        if isinstance(period, PeriodicCell):
            bounds = np.asarray(
                tuple(
                    corner
                    for cell in cells
                    for corner in product(
                        *zip(cell.image_lower, cell.image_upper, strict=True)
                    )
                ),
                dtype=np.float64,
            )
            frame = prepare_periodic_image_frame(
                bounds,
                np.arange(len(bounds), dtype=np.int64),
                np.zeros((len(bounds), period.rank), dtype=np.int64),
                period,
            )
            actions = frame.images()
        else:
            actions = (
                tuple(int(value) for value in row)
                for row in periodic_overlap_image_exponents(
                    period, ((np.min(lower, axis=0), np.max(upper, axis=0)),)
                )
            )
    except (ArithmeticError, OverflowError, TypeError, ValueError) as error:
        return PeriodicContactResult(
            "unresolved",
            (("periodic_image_frame_unresolved", (), str(error)),),
            boundary,
        )
    tree = prepare_bvh(lower, upper)
    key_ids: dict[PeriodicVertexKey, int] = {}

    def identity(key: PeriodicVertexKey, /) -> int:
        return key_ids.setdefault(key, len(key_ids))

    base = tuple(
        replace(
            cell,
            global_id=index,
            vertex_ids=tuple(identity(key) for key in row),
            owning_mesh_id=None,
            owning_geometry_id=None,
            coordinate_expressions=None if cell.affine else cell.coordinate_expressions,
        )
        for index, (cell, row) in enumerate(zip(cells, keys, strict=True))
    )
    # Canonical exact-enclosure filtering applies to affine lifted tetrahedra;
    # curved maps are deliberately left for their full-map proofs below.
    base_simplices = []
    for cell in base:
        if not cell.affine:
            base_simplices.append(None)
            continue
        try:
            base_simplices.append(_cell_simplices(cell))
        except (ArithmeticError, ValueError):
            base_simplices.append(None)
    piece_filter = _VolumePieceFilter.prepare(mesh, base, base_simplices)
    base_key_ids = key_ids.copy()
    findings: list[tuple[str, tuple[int, ...], str]] = []
    for image in actions:
        if not isinstance(image, tuple):
            raise TypeError("Lattice image must have integer exponents.")
        exponent = tuple(int(value) for value in image)
        if any(exponent) and next(value for value in exponent if value) < 0:
            continue
        matrix = exact_periodic_element(period, np.asarray(exponent))
        translation = tuple(matrix[axis][dimension] for axis in range(dimension))
        zero = not any(exponent)
        if zero:
            images = base
            shifted_tree = tree
        else:
            key_ids = base_key_ids.copy()
            shifted = []
            for index, (cell, row) in enumerate(zip(cells, keys, strict=True)):
                image_keys = tuple(
                    (root, _image_winding(winding, exponent, orders, fixed_by_root[root]))
                    for root, winding in row
                )
                expressions = (
                    None
                    if cell.affine
                    else tuple(
                        add(expression, constant(translation[axis], cell.dimension))
                        for axis, expression in enumerate(cell.coordinates)
                    )
                )
                normalizations = cell.pyramid_normalizations
                if normalizations is not None:
                    normalizations = tuple(
                        item._replace(
                            expression=add(
                                item.expression,
                                constant(translation[axis], cell.dimension),
                            ),
                            apex_trace=(
                                None
                                if item.apex_trace is None
                                else add(
                                    item.apex_trace,
                                    constant(translation[axis], cell.dimension - 1),
                                )
                            ),
                            apex_value=(
                                None
                                if item.apex_value is None
                                else item.apex_value + translation[axis]
                            ),
                        )
                        for axis, item in enumerate(normalizations)
                    )
                shifted.append(
                    replace(
                        cell,
                        global_id=len(cells) + index,
                        vertex_ids=tuple(identity(key) for key in image_keys),
                        owning_mesh_id=None,
                        owning_geometry_id=None,
                        coordinate_expressions=expressions,
                        corner_images=tuple(
                            tuple(
                                value + translation[axis]
                                for axis, value in enumerate(point)
                            )
                            for point in cell.corner_images
                        ),
                        affine_offset=(
                            None
                            if cell.affine_offset is None
                            else tuple(
                                value + translation[axis]
                                for axis, value in enumerate(cell.affine_offset)
                            )
                        ),
                        image_lower=tuple(
                            coefficient_endpoint(
                                Fraction(value) + translation[axis], upper=False
                            )
                            for axis, value in enumerate(cell.image_lower)
                        ),
                        image_upper=tuple(
                            coefficient_endpoint(
                                Fraction(value) + translation[axis], upper=True
                            )
                            for axis, value in enumerate(cell.image_upper)
                        ),
                        pyramid_normalizations=normalizations,
                    )
                )
            images = tuple(shifted)
            shifted_tree = prepare_bvh(
                np.asarray([cell.image_lower for cell in images]),
                np.asarray([cell.image_upper for cell in images]),
            )
        pieces: dict[int, tuple[tuple[tuple[ExactScalar, ...], ...], ...]] = (
            {
                index: value
                for index, value in enumerate(base_simplices)
                if value is not None
            }
            if zero
            else {}
        )
        for left, right in bvh_overlap_pair_blocks(
            tree, shifted_tree, include_touching=True
        ):
            piece_certificates = {}
            if zero:
                ordered = left < right
                left, right = left[ordered], right[ordered]
                if piece_filter is not None:
                    cleared, piece_certificates = piece_filter.certify(left, right)
                    left, right = left[~cleared], right[~cleared]
            shadows = {}
            cache = StratumSubdivisionCache()
            residual_cache = {}
            trace_cache: dict[
                tuple[int, tuple[int, ...]], tuple[Expression, ...] | None
            ] = {}
            for pair_offset, (left_row, right_row) in enumerate(
                zip(left.tolist(), right.tolist(), strict=True)
            ):
                if pair_offset and pair_offset % 64 == 0:
                    shadows = {}
                    cache = StratumSubdivisionCache()
                    residual_cache = {}
                    trace_cache.clear()
                i, j = int(left_row), int(right_row)
                if zero and i >= j:
                    continue
                first, second = base[i], images[j]
                shared = tuple(sorted(set(first.vertex_ids) & set(second.vertex_ids)))
                trace = restriction_trace_equal(first, second, shared, cache=trace_cache)
                entity_ids = (cells[i].global_id, cells[j].global_id)
                if trace is False:
                    findings.append(
                        (
                            "periodic_source_trace_mismatch",
                            entity_ids,
                            "Mapped quotient stratum traces disagree.",
                        )
                    )
                    continue
                if first.affine and second.affine and trace is True:
                    try:
                        if first.global_id not in pieces:
                            pieces[first.global_id] = _cell_simplices(first)
                        if second.global_id not in pieces:
                            pieces[second.global_id] = _cell_simplices(second)
                        affine_result: bool | None = True
                        certified_pieces = piece_certificates.get((i, j))
                        for first_index, first_piece in enumerate(
                            pieces[first.global_id]
                        ):
                            for second_index, second_piece in enumerate(
                                pieces[second.global_id]
                            ):
                                if (
                                    certified_pieces is not None
                                    and certified_pieces[first_index, second_index]
                                ):
                                    common = tuple(
                                        point
                                        for point in first_piece
                                        if point in second_piece
                                    )
                                    if (
                                        _contact_is_declared(
                                            first,
                                            second,
                                            common,
                                            first.vertex_ids,
                                            second.vertex_ids,
                                        )
                                        is True
                                    ):
                                        continue
                                contacts = _plane_side_shared_contacts(
                                    first_piece, second_piece, dimension, dimension
                                )
                                if (
                                    contacts is not None
                                    and _contact_is_declared(
                                        first,
                                        second,
                                        contacts,
                                        first.vertex_ids,
                                        second.vertex_ids,
                                    )
                                    is True
                                ):
                                    continue
                                contacts = _piece_contacts(
                                    first_piece, second_piece, dimension, dimension
                                )
                                declared = _contact_is_declared(
                                    first,
                                    second,
                                    contacts,
                                    first.vertex_ids,
                                    second.vertex_ids,
                                )
                                if declared is not True:
                                    affine_result = declared
                                    break
                            if affine_result is not True:
                                break
                    except (ArithmeticError, ValueError):
                        affine_result = None
                    if affine_result is True:
                        continue
                    if affine_result is False:
                        findings.append(
                            (
                                "periodic_cell_contact",
                                entity_ids,
                                f"Exact affine pieces overlap outside the declared stratum at winding {exponent}.",
                            )
                        )
                        continue
                proved = not shared and _strict_image_separation(first, second, shadows)
                if trace is True:
                    if not proved and len(shared) >= dimension:
                        proved = (
                            glued_face_monotone(first, second, shared)
                            or _shared_facet_separation(first, second, shared, shadows)
                            or glued_facet_monotone(first, second, shared, cache)
                            or (
                                len(shared) in (3, 4)
                                and _shared_curved_facet_separation(first, second, shared)
                                is not None
                            )
                        )
                    if not proved and len(shared) <= 2:
                        if shared not in residual_cache:
                            residual_cache[shared] = _stratum_residuals(first, shared)
                        residuals = residual_cache[shared]
                        proved = residuals is not None and (
                            subdivided_stratum_separation(
                                first, second, shared, residuals, cache
                            )
                            is not None
                        )
                    if not proved and len(shared) == 2:
                        proved = (
                            _shared_edge_separation(first, second, shared, shadows)
                            is not None
                            or _shared_edge_cone_separation(first, second, shared)
                            is not None
                        )
                    if not proved and len(shared) == 1:
                        proved = (
                            _shared_vertex_separation(first, second, shared, {}, shadows)
                            is not None
                        )
                if not proved:
                    findings.append(
                        (
                            "periodic_mapped_contact_unresolved",
                            entity_ids,
                            f"Exact mapped image contact at winding {exponent} and "
                            f"shared stratum {shared} has no completed separation.",
                        )
                    )
    status = (
        "violated"
        if any(
            check in ("periodic_source_trace_mismatch", "periodic_cell_contact")
            for check, _, _ in findings
        )
        else "unresolved"
        if findings
        else "certified"
    )
    return PeriodicContactResult(status, tuple(findings), boundary)
