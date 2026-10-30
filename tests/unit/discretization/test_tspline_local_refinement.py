#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import replace
from fractions import Fraction

import pytest

from phydrax._interpolation._tspline import (
    _cell_extraction,
    _cell_support_rows,
    _ExtractionPiece,
    _ordered_anchor_supports,
    ExtractedBernstein,
    TMesh2D,
)


def _exact_geometry_banks(
    mesh: TMesh2D,
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    controls = []
    weights = []
    for anchor in mesh.anchors:
        u, v = anchor.local_knots.greville_parameter
        controls.append((u, v, u * u + u * v + v * v))
        weights.append(Fraction(3, 4) + v / 5)
    return tuple(controls), tuple(weights)


def _exact_piece(
    extraction: ExtractedBernstein,
    homogeneous: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(
            sum(
                (
                    extraction.exact_extraction_operator[local_row][control]
                    * homogeneous[coefficient_row][component]
                    for local_row, coefficient_row in enumerate(
                        extraction.coefficient_row_ids
                    )
                ),
                Fraction(0),
            )
            for component in range(len(homogeneous[0]))
        )
        for control in range(16)
    )


def test_local_refinement_preserves_distant_cells_and_non_tensor_topology() -> None:
    breaks = tuple(Fraction(index, 16) for index in range(17))
    mesh = TMesh2D.tensor_product(breaks, breaks)
    marked = next(
        cell.cell_id
        for cell in mesh.cells
        if cell.parameter_bounds
        == (Fraction(7, 16), Fraction(8, 16), Fraction(7, 16), Fraction(8, 16))
    )
    distant = mesh.cells[0]

    result = mesh.refine((marked,))
    target = result.refined_mesh
    assert target.certify().passed
    assert {
        extension.missing_direction for extension in target.t_junction_extensions
    } >= {"left", "right", "down", "up"}
    assert target.cell(distant.cell_id).parameter_bounds == distant.parameter_bounds
    assert target.cell_count < (len(target.axis_breaks[0]) - 1) * (
        len(target.axis_breaks[1]) - 1
    )
    assert any(
        anchor.anchor_id == mesh.anchors[0].anchor_id
        and anchor.parameter == mesh.anchors[0].parameter
        and anchor.local_knots == mesh.anchors[0].local_knots
        for anchor in target.anchors
    )


def test_exact_rational_geometry_pieces_survive_nonuniform_refinement() -> None:
    mesh = TMesh2D.tensor_product(
        (Fraction(0), Fraction(2, 3), Fraction(1)),
        (Fraction(0), Fraction(1, 2), Fraction(1)),
    )
    marked = next(
        cell.cell_id
        for cell in mesh.cells
        if cell.parameter_bounds
        == (Fraction(0), Fraction(2, 3), Fraction(0), Fraction(1, 2))
    )
    control_points, weights = _exact_geometry_banks(mesh)
    source_homogeneous = tuple(
        tuple(weight * coordinate for coordinate in point) + (weight,)
        for point, weight in zip(control_points, weights, strict=True)
    )
    field_weights = tuple(
        Fraction(index + 2, index + 5) for index in range(mesh.coefficient_count)
    )
    field_coefficients = tuple(
        (Fraction((index % 7) - 3, 11),) for index in range(mesh.coefficient_count)
    )
    relation = mesh.refine((marked,))
    target_homogeneous = relation.transfer_plan.apply_exact_homogeneous_geometry(
        control_points, weights
    )
    target_field_values, target_field_weights = (
        relation.transfer_plan.apply_exact_rational_field(
            field_coefficients, field_weights
        )
    )
    source_field_homogeneous = tuple(
        (field_weights[index] * row[0], field_weights[index])
        for index, row in enumerate(field_coefficients)
    )
    target_field_homogeneous = tuple(
        (target_field_weights[index] * row[0], target_field_weights[index])
        for index, row in enumerate(target_field_values)
    )
    assert Fraction(1, 3) in relation.transfer_plan.exact_coefficients
    assert any(weight != weights[0] for weight in weights)
    assert any(weight != field_weights[0] for weight in field_weights)

    target_realization = relation.refined_mesh.realize()
    ordered_source_supports = _ordered_anchor_supports(mesh)
    source_cells = mesh.cells
    for target_piece in target_realization.extractions:
        u0, u1, v0, v1 = target_piece.parameter_bounds
        parent = next(
            cell
            for cell in source_cells
            if cell.parameter_bounds[0] <= u0
            and u1 <= cell.parameter_bounds[1]
            and cell.parameter_bounds[2] <= v0
            and v1 <= cell.parameter_bounds[3]
        )
        source_piece = _ExtractionPiece(
            target_piece.piece_id,
            parent.cell_id,
            target_piece.parameter_bounds,
        )
        source_rows = _cell_support_rows(
            mesh, target_piece.parameter_bounds, ordered_source_supports
        )
        source_extraction = _cell_extraction(mesh, source_piece, source_rows)
        source_geometry_piece = _exact_piece(source_extraction, source_homogeneous)
        target_geometry_piece = target_realization.exact_homogeneous_piece(
            target_piece.piece_id, target_homogeneous
        )
        assert target_geometry_piece == source_geometry_piece
        source_field_piece = _exact_piece(source_extraction, source_field_homogeneous)
        target_field_piece = target_realization.exact_homogeneous_piece(
            target_piece.piece_id, target_field_homogeneous
        )
        assert target_field_piece == source_field_piece


def test_canonical_lineage_ids_ignore_mark_iteration_order() -> None:
    mesh = TMesh2D.tensor_product((0, 1), (0, 1))
    marked = (mesh.cells[0].cell_id, mesh.cells[-1].cell_id)
    forward = mesh.refine(marked)
    reverse = mesh.refine(tuple(reversed(marked)))

    assert forward.refined_mesh.mesh_id == reverse.refined_mesh.mesh_id
    assert forward.cell_lineage == reverse.cell_lineage
    assert forward.transfer_plan.plan_id == reverse.transfer_plan.plan_id


def test_exact_transfer_recomputes_retained_basis_certificates() -> None:
    mesh = TMesh2D.tensor_product((0, 1), (0, 1))
    relation = mesh.refine((mesh.cells[0].cell_id,))
    points = tuple(
        tuple(Fraction(value) for value in anchor.local_knots.greville_parameter)
        + (Fraction(0),)
        for anchor in mesh.anchors
    )
    weights = tuple(Fraction(1) for _ in mesh.anchors)
    target = relation.transfer_plan.apply_exact_homogeneous_geometry(points, weights)
    certificate = relation.certificate
    assert certificate.minimum_basis_value is not None
    forged = replace(
        certificate,
        minimum_basis_value=certificate.minimum_basis_value + 1.0,
    )

    with pytest.raises(ValueError, match="recomputed evidence"):
        relation.transfer_plan.verify_exact_homogeneous_geometry_preservation(
            mesh,
            relation.refined_mesh,
            points,
            weights,
            target,
            source_certificate=mesh.certify(),
            target_certificate=forged,
        )
