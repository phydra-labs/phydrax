from fractions import Fraction
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    coordinate_effective_basis,
    FiniteElementSpec,
    PerCellRestrictedCellGeometryElement,
    RationalComposedCellGeometryElement,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._coordinate_enclosure import (
    compose_expression,
    coordinate_map_expressions,
    evaluate_expression,
    ExpressionInput,
    normalize_pyramid_reference_derivative,
    Polynomial,
    RationalPolynomial,
)
from phydrax.discretization._nested_reference import (
    _root_affine_inverse,
    coarsen_reference_cell_maps,
    compose_reference_cell_maps,
    exact_reference_chart_controls,
    nested_cell_reference_maps,
    reference_chart_controls_by_target_id,
)
from phydrax.discretization._nested_reference_types import (
    evaluate_reference_chart_arguments,
    exact_reference_cell_vertices,
    freeze_reference_chart_arguments,
    NestedReferenceWitnesses,
    reference_chart_arguments_contain_source_cell,
    reference_chart_arguments_identity,
    ReferenceCellMapWitnesses,
    validate_reference_cell_map_witnesses,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem._reference import lagrange_element
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain


def test_grouped_composed_tabulation_preserves_rows_and_jacobians() -> None:
    from phydrax.discretization._coordinate_enclosure import axes, scale

    denominator: Polynomial = {(0, 0): Fraction(1), (1, 0): Fraction(1, 2)}
    source = RationalComposedCellGeometryElement(
        lagrange_element("triangle", 2),
        "triangle",
        tuple(RationalPolynomial(axis, denominator) for axis in axes(2)),
    )
    identity = axes(2)
    half = tuple(scale(axis, Fraction(1, 2)) for axis in identity)
    element = PerCellRestrictedCellGeometryElement(
        source, "triangle", (10, 11), (5, 5), (identity, half)
    )
    rows = (1, 0, 1)
    points = jnp.asarray(((0.2, 0.3), (0.1, 0.15)))
    values, gradients = jax.jit(lambda x: element.tabulate_cells(rows, x))(points)
    for position, row in enumerate(rows):
        reference = element.tabulate_cell(row, points)
        np.testing.assert_allclose(values[position], reference[0], rtol=0.0, atol=2e-14)
        np.testing.assert_allclose(
            gradients[position], reference[1], rtol=0.0, atol=2e-14
        )
    for point_index, point in enumerate(points):
        derivative = jax.jacfwd(lambda x: element.tabulate_cells(rows, x[None])[0][:, 0])(
            point
        )
        np.testing.assert_allclose(
            gradients[:, point_index], derivative, rtol=2e-13, atol=2e-14
        )
    empty_values, empty_gradients = element.tabulate_cells((), points)
    assert empty_values.shape == (0, 2, source.local_dof_count)
    assert empty_gradients.shape == (0, 2, source.local_dof_count, 2)


_BASE_VERTICES = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    dtype=np.float64,
)
_BOUNDARY_FACETS = np.asarray(
    ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64
)


def _mapped_domain(*, split: bool = False) -> MappedReferenceDomain:
    if split:
        root_points = np.concatenate(
            (_BASE_VERTICES, np.asarray(((0.5, 0.0, 0.0),))), axis=0
        )
        root_cells = np.asarray(((0, 4, 2, 3), (4, 1, 2, 3)), dtype=np.int32)
        root_ids = np.asarray((71, 72), dtype=np.int64)
        occurrences = (("assembly", "left-root"), ("assembly", "right-root"))
    else:
        root_points = _BASE_VERTICES.copy()
        root_cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
        root_ids = np.asarray((71,), dtype=np.int64)
        occurrences = (("assembly", "curved-root"),)
    root_mesh = CellMesh(
        root_points,
        (CellBlock("roots", "tetrahedron", root_cells, global_ids=root_ids),),
        vertex_global_ids=np.arange(100, 100 + root_points.shape[0], dtype=np.int64),
    )

    element = lagrange_element("tetrahedron", 2)
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    geometry_coordinates = []
    for cell in root_cells:
        vertices = root_points[cell]
        origin = vertices[0]
        matrix = (vertices[1:] - origin).T
        reference_points = origin[None, :] + reference_nodes @ matrix.T
        physical_points = np.column_stack(
            (
                reference_points[:, 0],
                reference_points[:, 1],
                reference_points[:, 2]
                + 0.2 * reference_points[:, 0] * reference_points[:, 1],
            )
        )
        geometry_coordinates.extend(physical_points)
    geometry_coordinates = np.asarray(geometry_coordinates, dtype=np.float64)
    geometry_routes = np.arange(geometry_coordinates.shape[0], dtype=np.int32).reshape(
        (len(root_cells), element.local_dof_count)
    )
    source_geometry = CellGeometrySpec(
        {"roots": element}, {"roots": geometry_routes}, geometry_coordinates
    )
    reference_domain = PiecewiseLinearDomain(
        _BASE_VERTICES,
        _BOUNDARY_FACETS,
        np.tile(np.asarray(((0, -1),), dtype=np.int64), (4, 1)),
        ("solid",),
        source_id="reference-tetrahedron",
    )
    return MappedReferenceDomain(
        reference_domain,
        root_mesh,
        source_geometry,
        np.zeros((len(root_cells),), dtype=np.int64),
        source_id="native-source",
        source_revision="r7",
        coordinate_contract=SpatialCoordinateContract.si(),
        root_cell_occurrence_paths=occurrences,
    )


def _target(
    points: np.ndarray,
    *,
    cell_id: int = 901,
    cell_kind: str = "tetrahedron",
    geometry: CellGeometrySpec | None = None,
) -> tuple[CellMesh, CellGeometrySpec]:
    coordinates = np.asarray(points, dtype=np.float64)
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "target",
                cell_kind,
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=np.asarray((cell_id,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(500, 504, dtype=np.int64),
    )
    return mesh, CellGeometrySpec.affine(mesh) if geometry is None else geometry


def test_exact_reference_subcell_composes_through_curved_physical_root() -> None:
    domain = _mapped_domain()
    assert domain.exact_region_measures() == (Fraction(1, 6),)
    child_points = np.asarray(
        (
            (1 / 8, 1 / 8, 1 / 8),
            (3 / 8, 1 / 8, 1 / 8),
            (1 / 8, 3 / 8, 1 / 8),
            (1 / 8, 1 / 8, 3 / 8),
        ),
        dtype=np.float64,
    )
    mesh, geometry = _target(child_points, cell_id=901)

    (control,) = exact_reference_chart_controls(
        domain, mesh, geometry, np.asarray((0,), dtype=np.int64)
    )

    assert reference_chart_controls_by_target_id((control,))[901] is control
    assert control.target_cell_id == 901
    assert control.source_cell_id == 71
    assert control.source_occurrence_path == ("assembly", "curved-root")
    assert control.source_id == "native-source"
    assert control.source_revision == "r7"
    assert control.mapped_domain_id == domain.domain_id
    assert control.source_geometry_id == domain.reference_geometry_id
    assert control.source_region == 0
    assert control.source_region_id == "solid"
    assert control.target_cell_row == 0
    validate_reference_cell_map_witnesses(control.witness)
    assert reference_chart_arguments_identity(
        control.witness.source_reference_chart_arguments
    ) == reference_chart_arguments_identity((control.arguments,))
    assert control.source_block_name == "roots"
    assert control.source_block_index == 0
    assert control.source_cell_row == 0
    assert exact_reference_cell_vertices(control.witness) == (
        tuple(tuple(Fraction(float(value)) for value in point) for point in child_points),
    )
    assert control.restriction_witness.target_cell_ids.tolist() == [0]

    root_elements, root_routes, _ = domain.source_geometry.resolve(domain.reference_mesh)
    root_element = root_elements[0]
    assert isinstance(root_element, FiniteElementSpec)
    root_coefficients = domain.source_geometry.source_coordinates(
        np.asarray(root_routes[0][0], dtype=np.int64)
    )
    root_expressions = coordinate_map_expressions(
        root_element, root_coefficients, cell_row=0
    )
    physical = tuple(
        compose_expression(expression, control.arguments)
        for expression in root_expressions
    )
    target_point = (Fraction(1, 4),) * 3
    reference_point = tuple(
        evaluate_expression(expression, target_point) for expression in control.arguments
    )
    expected = domain.source_geometry.evaluate_exact(
        domain.reference_mesh,
        np.asarray((control.source_cell_id,), dtype=np.int64),
        ((reference_point,),),
    )[0][0]
    assert (
        tuple(evaluate_expression(expression, target_point) for expression in physical)
        == expected
    )
    assert root_element.degree == 2


def test_simplex_three_alias_uses_canonical_tetra_reference_topology() -> None:
    domain = _mapped_domain()
    child_points = np.asarray(
        (
            (1 / 8, 1 / 8, 1 / 8),
            (3 / 8, 1 / 8, 1 / 8),
            (1 / 8, 3 / 8, 1 / 8),
            (1 / 8, 1 / 8, 3 / 8),
        ),
        dtype=np.float64,
    )
    mesh, geometry = _target(child_points, cell_id=907, cell_kind="simplex:3")

    (control,) = exact_reference_chart_controls(
        domain, mesh, geometry, np.asarray((0,), dtype=np.int64)
    )

    assert control.target_cell_kind == "simplex:3"
    assert control.source_cell_kind == "tetrahedron"
    assert control.source_cell_id == 71
    validate_reference_cell_map_witnesses(control.witness)
    assert exact_reference_cell_vertices(control.witness) == (
        tuple(tuple(Fraction(float(value)) for value in point) for point in child_points),
    )


def test_exact_pyramid_target_chart_uses_pyramid_domain_inclusion_proof() -> None:
    domain = _mapped_domain()
    target_points = np.asarray(
        (
            (1 / 8, 1 / 8, 1 / 8),
            (3 / 8, 1 / 8, 1 / 8),
            (3 / 8, 3 / 8, 1 / 8),
            (1 / 8, 3 / 8, 1 / 8),
            (1 / 4, 1 / 4, 3 / 8),
        ),
        dtype=np.float64,
    )
    source_element = lagrange_element("tetrahedron", 1)
    target_chart: tuple[Polynomial, ...] = (
        {(0, 0, 0): Fraction(1, 8), (1, 0, 0): Fraction(1, 4)},
        {(0, 0, 0): Fraction(1, 8), (0, 1, 0): Fraction(1, 4)},
        {(0, 0, 0): Fraction(1, 8), (0, 0, 1): Fraction(1, 4)},
    )
    target_element = PerCellRestrictedCellGeometryElement(
        source_element, "pyramid", (908,), (71,), (target_chart,)
    )
    target_mesh = CellMesh(
        target_points,
        (
            CellBlock(
                "pyramids",
                "pyramid",
                np.arange(5, dtype=np.int32)[None, :],
                global_ids=np.asarray((908,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(610, 615, dtype=np.int64),
    )
    target_geometry = CellGeometrySpec(
        {"pyramids": target_element},
        {"pyramids": np.arange(4, dtype=np.int32)[None, :]},
        _BASE_VERTICES,
    )
    (control,) = exact_reference_chart_controls(
        domain,
        target_mesh,
        target_geometry,
        np.asarray((0,), dtype=np.int64),
    )

    assert control.target_cell_kind == "pyramid"
    assert control.source_cell_kind == "tetrahedron"
    assert control.source_cell_id == 71
    validate_reference_cell_map_witnesses(control.witness)
    assert nested_cell_reference_maps(control.witness)[0].evaluate(
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 5))
    ) == (Fraction(1, 4), Fraction(1, 4), Fraction(7, 40))

    source_witness = control.witness
    corner_only_witness = ReferenceCellMapWitnesses(
        source_witness.target_cell_ids,
        source_witness.source_cell_ids,
        source_witness.target_cell_kind,
        source_witness.source_cell_kind,
        source_witness.source_reference_vertices,
        source_witness.source_reference_exact_vertices,
    )
    assert (
        nested_cell_reference_maps(corner_only_witness)[0].evaluate(
            (Fraction(1, 2), Fraction(1, 2), Fraction(1))
        )
        == control.source_reference_vertices[-1]
    )

    pyramid_identity: tuple[Polynomial, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
    )
    assert reference_chart_arguments_contain_source_cell(
        "pyramid", "pyramid", pyramid_identity
    )
    outside_pyramid: tuple[Polynomial, ...] = (
        {(0, 0, 0): Fraction(2), (1, 0, 0): Fraction(1)},
        pyramid_identity[1],
        pyramid_identity[2],
    )
    assert not reference_chart_arguments_contain_source_cell(
        "pyramid", "pyramid", outside_pyramid
    )


def test_pyramid_chart_requires_only_a_unique_c0_apex_image() -> None:
    denominator: Polynomial = {
        (0, 0, 0): Fraction(1),
        (0, 0, 1): Fraction(-1),
    }
    numerator: Polynomial = {
        (0, 0, 0): Fraction(3, 4),
        (1, 0, 0): Fraction(-1),
        (2, 0, 0): Fraction(1),
        (0, 0, 1): Fraction(-1, 2),
    }
    chart: tuple[Polynomial | RationalPolynomial, ...] = (
        RationalPolynomial(numerator, denominator),
        {(0, 0, 0): Fraction(1, 2)},
        {(0, 0, 1): Fraction(1)},
    )
    apex = (Fraction(1, 2), Fraction(1, 2), Fraction(1))
    assert evaluate_reference_chart_arguments("pyramid", chart, apex) == apex
    assert normalize_pyramid_reference_derivative(chart[0], 2).apex_value is None

    target_corners = tuple(
        tuple(Fraction(value) for value in point)
        for point in reference_cell_topology("pyramid").vertices
    )
    evaluated_corners = tuple(
        evaluate_reference_chart_arguments("pyramid", chart, point)
        for point in target_corners
    )
    assert all(
        isinstance(value, Fraction) for point in evaluated_corners for value in point
    )
    source_corners = cast(tuple[tuple[Fraction, ...], ...], evaluated_corners)
    witness = ReferenceCellMapWitnesses(
        np.asarray((19,), dtype=np.int64),
        np.asarray((29,), dtype=np.int64),
        "pyramid",
        "pyramid",
        np.asarray(
            [[[float(value) for value in point] for point in source_corners]],
            dtype=np.float64,
        ),
        (source_corners,),
        freeze_reference_chart_arguments((chart,)),
    )
    validate_reference_cell_map_witnesses(witness)
    assert nested_cell_reference_maps(witness)[0].evaluate(apex) == source_corners[-1]

    nonunique_apex = RationalPolynomial(
        {
            (0, 0, 0): Fraction(-1, 2),
            (1, 0, 0): Fraction(1),
        },
        denominator,
    )
    with pytest.raises(ValueError, match="one unique C0 apex image"):
        evaluate_reference_chart_arguments("pyramid", (nonunique_apex,), apex)

    interior_pole = RationalPolynomial(
        {(0, 0, 0): Fraction(1)},
        {
            (0, 0, 0): Fraction(-1, 4),
            (0, 0, 1): Fraction(1),
        },
    )
    with pytest.raises(ValueError, match="denominator is not proved nonzero"):
        evaluate_reference_chart_arguments(
            "pyramid",
            (interior_pole,),
            (Fraction(1, 2), Fraction(1, 2), Fraction(1, 4)),
        )


def test_pyramid_root_inverse_requires_an_exact_affine_chart() -> None:
    reference_vertices = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_cell_topology("pyramid").vertices
    )
    origin = (Fraction(2), Fraction(-1), Fraction(3, 4))
    corners = tuple(
        (
            origin[0] + 2 * point[0],
            origin[1] + 3 * point[1],
            origin[2] + 4 * point[2],
        )
        for point in reference_vertices
    )

    offset, inverse = _root_affine_inverse("pyramid", corners)

    assert offset == origin
    assert inverse == (
        (Fraction(1, 2), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1, 3), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1, 4)),
    )

    nonaffine = list(corners)
    nonaffine[2] = (
        nonaffine[2][0],
        nonaffine[2][1],
        nonaffine[2][2] + Fraction(1, 16),
    )
    with pytest.raises(ValueError, match="exact affine chart"):
        _root_affine_inverse("pyramid", tuple(nonaffine))


def test_rational_degree_one_reference_chart_retains_exact_controls() -> None:
    domain = _mapped_domain()
    x_numerator: Polynomial = {
        (1, 0, 0): Fraction(1),
        (2, 0, 0): Fraction(1),
    }
    x_denominator: Polynomial = {
        (0, 0, 0): Fraction(1),
        (1, 0, 0): Fraction(1),
    }
    y_numerator: Polynomial = {
        (0, 1, 0): Fraction(1),
        (0, 2, 0): Fraction(1),
    }
    y_denominator: Polynomial = {
        (0, 0, 0): Fraction(1),
        (0, 1, 0): Fraction(1),
    }
    z_numerator: Polynomial = {
        (0, 0, 1): Fraction(1),
        (0, 0, 2): Fraction(1),
    }
    z_denominator: Polynomial = {
        (0, 0, 0): Fraction(1),
        (0, 0, 1): Fraction(1),
    }
    rational_arguments = (
        RationalPolynomial(x_numerator, x_denominator),
        RationalPolynomial(y_numerator, y_denominator),
        RationalPolynomial(z_numerator, z_denominator),
    )
    source_element = lagrange_element("tetrahedron", 1)
    rational_element = RationalComposedCellGeometryElement(
        source_element, "tetrahedron", rational_arguments
    )
    child_points = np.asarray(
        (
            (1 / 8, 1 / 8, 1 / 8),
            (3 / 8, 1 / 8, 1 / 8),
            (1 / 8, 3 / 8, 1 / 8),
            (1 / 8, 1 / 8, 3 / 8),
        ),
        dtype=np.float64,
    )
    mesh, _ = _target(child_points, cell_id=902)
    geometry = CellGeometrySpec(
        {"target": rational_element},
        {"target": np.arange(4, dtype=np.int32)[None, :]},
        child_points,
    )

    (control,) = exact_reference_chart_controls(
        domain, mesh, geometry, np.asarray((0,), dtype=np.int64)
    )

    assert control.source_cell_id == 71
    assert control.source_reference_vertices == tuple(
        tuple(Fraction(float(value)) for value in point) for point in child_points
    )
    validate_reference_cell_map_witnesses(control.witness)


def test_rational_reference_witness_roundtrips_and_reproves_whole_cell() -> None:
    corners = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    corner_view = np.asarray(
        [[[float(value) for value in point] for point in corners]],
        dtype=np.float64,
    )
    denominator: Polynomial = {
        (0, 0, 0): Fraction(1),
        (1, 0, 0): Fraction(1),
    }

    def make_witness(multiplier: Fraction) -> ReferenceCellMapWitnesses:
        rational_x_numerator: Polynomial = {
            (1, 0, 0): Fraction(1) + multiplier,
            (2, 0, 0): Fraction(1) - multiplier,
            (1, 1, 0): -multiplier,
            (1, 0, 1): -multiplier,
        }
        rational_x = RationalPolynomial(rational_x_numerator, denominator)
        mutable_y: Polynomial = {(0, 1, 0): Fraction(1)}
        mutable_z: Polynomial = {(0, 0, 1): Fraction(1)}
        chart_arguments = freeze_reference_chart_arguments(
            ((rational_x, mutable_y, mutable_z),)
        )
        mutable_y[(0, 1, 0)] = Fraction(2)
        return ReferenceCellMapWitnesses(
            np.asarray((0,), dtype=np.int64),
            np.asarray((71,), dtype=np.int64),
            "tetrahedron",
            "tetrahedron",
            corner_view.copy(),
            (corners,),
            chart_arguments,
        )

    witness = make_witness(Fraction(1, 4))
    validate_reference_cell_map_witnesses(witness)
    chart = nested_cell_reference_maps(witness)[0]
    point = (Fraction(1, 4),) * 3
    assert chart.arguments is not None
    assert chart.evaluate(point) == (Fraction(21, 80), Fraction(1, 4), Fraction(1, 4))
    target_corners = (
        (Fraction(1, 8), Fraction(1, 8), Fraction(1, 8)),
        (Fraction(3, 8), Fraction(1, 8), Fraction(1, 8)),
        (Fraction(1, 8), Fraction(3, 8), Fraction(1, 8)),
        (Fraction(1, 8), Fraction(1, 8), Fraction(3, 8)),
    )
    target_to_intermediate = ReferenceCellMapWitnesses(
        np.asarray((90,), dtype=np.int64),
        np.asarray((100,), dtype=np.int64),
        "tetrahedron",
        "tetrahedron",
        np.asarray(
            [[[float(value) for value in point] for point in target_corners]],
            dtype=np.float64,
        ),
        (target_corners,),
    )
    composed = compose_reference_cell_maps(
        target_to_intermediate,
        witness,
        intermediate_cell_global_ids=(100,),
    )
    target_map = nested_cell_reference_maps(target_to_intermediate)[0]
    root_point = chart.evaluate(target_map.evaluate(point))
    composed_map = nested_cell_reference_maps(composed)[0]
    assert composed_map.target_cell_id == 90
    assert composed_map.source_cell_id == 71
    assert composed_map.arguments is not None
    assert composed_map.evaluate(point) == root_point

    restriction = CellGeometryRestrictionSource(
        "root-geometry",
        "root-topology",
        {"mapped": np.asarray((71,), dtype=np.int64)},
        {"mapped": np.full((1, 4), -1, dtype=np.int64)},
        block_parent_reference_maps={"mapped": witness},
    )
    cold = CellGeometryRestrictionSource.from_stored_fields(restriction)
    cold_map = cold.block_parent_reference_maps["mapped"]
    assert cold_map is not None
    validate_reference_cell_map_witnesses(cold_map)
    assert reference_chart_arguments_identity(
        cold_map.source_reference_chart_arguments
    ) == reference_chart_arguments_identity(witness.source_reference_chart_arguments)
    assert nested_cell_reference_maps(cold_map)[0].evaluate(point) == chart.evaluate(
        point
    )

    alternative = CellGeometryRestrictionSource(
        "root-geometry",
        "root-topology",
        {"mapped": np.asarray((71,), dtype=np.int64)},
        {"mapped": np.full((1, 4), -1, dtype=np.int64)},
        block_parent_reference_maps={"mapped": make_witness(Fraction(1, 8))},
    )
    assert alternative.restriction_source_id != restriction.restriction_source_id

    escaped = make_witness(Fraction(4))
    with pytest.raises(ValueError, match="whole-cell inclusion"):
        validate_reference_cell_map_witnesses(escaped)

    denominator_with_interior_zero: Polynomial = {
        (0, 0, 0): Fraction(1, 4),
        (1, 0, 0): Fraction(-1),
        (2, 0, 0): Fraction(1),
    }
    matching_corner_numerator: Polynomial = {
        (1, 0, 0): Fraction(1, 4),
        (2, 0, 0): Fraction(-1),
        (3, 0, 0): Fraction(1),
    }
    singular_y: Polynomial = {(0, 1, 0): Fraction(1)}
    singular_z: Polynomial = {(0, 0, 1): Fraction(1)}
    singular_chart = ReferenceCellMapWitnesses(
        witness.target_cell_ids,
        witness.source_cell_ids,
        witness.target_cell_kind,
        witness.source_cell_kind,
        witness.source_reference_vertices,
        witness.source_reference_exact_vertices,
        freeze_reference_chart_arguments(
            (
                (
                    RationalPolynomial(
                        matching_corner_numerator,
                        denominator_with_interior_zero,
                    ),
                    singular_y,
                    singular_z,
                ),
            )
        ),
    )
    with pytest.raises(ValueError, match="denominator is not proved nonzero"):
        validate_reference_cell_map_witnesses(singular_chart)


def test_per_cell_restriction_retains_and_evaluates_rational_chart() -> None:
    source_element = lagrange_element("tetrahedron", 1)
    denominator: Polynomial = {
        (0, 0, 0): Fraction(1),
        (1, 0, 0): Fraction(1),
    }
    rational_x_numerator: Polynomial = {
        (1, 0, 0): Fraction(5, 4),
        (2, 0, 0): Fraction(3, 4),
        (1, 1, 0): Fraction(-1, 4),
        (1, 0, 1): Fraction(-1, 4),
    }
    rational_x = RationalPolynomial(rational_x_numerator, denominator)
    rational_y: Polynomial = {(0, 1, 0): Fraction(1)}
    rational_z: Polynomial = {(0, 0, 1): Fraction(1)}
    rational_chart_row: tuple[ExpressionInput, ...] = (
        rational_x,
        rational_y,
        rational_z,
    )
    rational_chart_bank = (rational_chart_row,)
    element = PerCellRestrictedCellGeometryElement(
        source_element,
        "tetrahedron",
        np.asarray((90,), dtype=np.int64),
        np.asarray((71,), dtype=np.int64),
        rational_chart_bank,
    )

    with pytest.raises(TypeError, match="target_cell_ids"):
        PerCellRestrictedCellGeometryElement(
            source_element,
            "tetrahedron",
            np.asarray((90.0,), dtype=np.float64),
            np.asarray((71,), dtype=np.int64),
            rational_chart_bank,
        )

    retained = element.chart_arguments(0)
    assert isinstance(retained[0], RationalPolynomial)
    assert retained[0].numerator == rational_x.numerator
    assert retained[0].denominator == rational_x.denominator
    point = (Fraction(1, 4),) * 3
    expected = tuple(
        evaluate_expression(expression, point)
        for expression in coordinate_effective_basis(element, cell_row=0)
    )
    values, gradients = element.tabulate_cell(
        0, np.asarray([[float(value) for value in point]], dtype=np.float64)
    )
    np.testing.assert_allclose(
        np.asarray(values)[0],
        np.asarray([float(value) for value in expected]),
        rtol=0.0,
        atol=1e-14,
    )
    assert np.all(np.isfinite(np.asarray(gradients)))

    extrapolated_x: Polynomial = {
        (0, 0, 0): Fraction(2),
        (1, 0, 0): Fraction(1),
    }
    extrapolated_y: Polynomial = {(0, 1, 0): Fraction(1)}
    extrapolated_z: Polynomial = {(0, 0, 1): Fraction(1)}
    extrapolated_chart: tuple[ExpressionInput, ...] = (
        extrapolated_x,
        extrapolated_y,
        extrapolated_z,
    )
    extrapolated = PerCellRestrictedCellGeometryElement(
        source_element,
        "tetrahedron",
        (92,),
        (71,),
        (extrapolated_chart,),
    )
    assert isinstance(extrapolated.chart_arguments(0)[0], dict)
    assert evaluate_expression(extrapolated.chart_arguments(0)[0], point) == Fraction(
        9, 4
    )

    singular_denominator: Polynomial = {
        (0, 0, 0): Fraction(1, 4),
        (1, 0, 0): Fraction(-1),
        (2, 0, 0): Fraction(1),
    }
    matching_numerator: Polynomial = {
        (1, 0, 0): Fraction(1, 4),
        (2, 0, 0): Fraction(-1),
        (3, 0, 0): Fraction(1),
    }
    singular_x = RationalPolynomial(matching_numerator, singular_denominator)
    singular_y: Polynomial = {(0, 1, 0): Fraction(1)}
    singular_z: Polynomial = {(0, 0, 1): Fraction(1)}
    singular_chart: tuple[ExpressionInput, ...] = (
        singular_x,
        singular_y,
        singular_z,
    )
    with pytest.raises(ValueError, match="denominator is not proved nonzero"):
        PerCellRestrictedCellGeometryElement(
            source_element,
            "tetrahedron",
            (91,),
            (71,),
            (singular_chart,),
        )


def test_coarsening_composes_exact_child_root_charts_to_one_parent_chart() -> None:
    parent_corners = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    child_corners = (
        (
            parent_corners[0],
            (Fraction(1, 2), Fraction(0), Fraction(0)),
            parent_corners[2],
            parent_corners[3],
        ),
        (
            (Fraction(1, 2), Fraction(0), Fraction(0)),
            parent_corners[1],
            parent_corners[2],
            parent_corners[3],
        ),
    )
    child_execution = np.asarray(
        [[[float(value) for value in point] for point in cell] for cell in child_corners],
        dtype=np.float64,
    )
    fine_mesh = CellMesh(
        np.concatenate((_BASE_VERTICES, np.asarray(((0.5, 0.0, 0.0),))), axis=0),
        (
            CellBlock(
                "fine",
                "tetrahedron",
                np.asarray(((0, 4, 2, 3), (4, 1, 2, 3)), dtype=np.int32),
                global_ids=np.asarray((10, 11), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(700, 705, dtype=np.int64),
    )
    coarse_mesh = CellMesh(
        _BASE_VERTICES,
        (
            CellBlock(
                "coarse",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=np.asarray((5,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(800, 804, dtype=np.int64),
    )
    fine_to_root = ReferenceCellMapWitnesses(
        np.asarray((0, 1), dtype=np.int64),
        np.asarray((71, 71), dtype=np.int64),
        "tetrahedron",
        "tetrahedron",
        child_execution,
        child_corners,
    )
    coarsening = NestedReferenceWitnesses(
        np.asarray((10, 11), dtype=np.int64),
        np.asarray((5, 5), dtype=np.int64),
        child_execution,
        None,
        child_corners,
    )

    (parent_map,) = coarsen_reference_cell_maps(
        coarsening,
        {"fine": fine_to_root},
        fine_mesh,
        coarse_mesh,
    ).values()

    assert parent_map.target_cell_ids.tolist() == [0]
    assert parent_map.source_cell_ids.tolist() == [71]
    assert exact_reference_cell_vertices(parent_map) == (parent_corners,)
    assert (
        nested_cell_reference_maps(parent_map)[0].evaluate((Fraction(1, 4),) * 3)
        == (Fraction(1, 4),) * 3
    )

    split_roots = ReferenceCellMapWitnesses(
        fine_to_root.target_cell_ids,
        np.asarray((71, 72), dtype=np.int64),
        fine_to_root.target_cell_kind,
        fine_to_root.source_cell_kind,
        fine_to_root.source_reference_vertices,
        fine_to_root.source_reference_exact_vertices,
    )
    with pytest.raises(ValueError, match="spans multiple root cells"):
        coarsen_reference_cell_maps(
            coarsening,
            {"fine": split_roots},
            fine_mesh,
            coarse_mesh,
        )


@pytest.mark.parametrize("cell_kind", ("prism", "hexahedron"))
def test_exact_reference_controls_cover_prism_and_hexahedron_targets(
    cell_kind: str,
) -> None:
    domain = _mapped_domain()
    topology = reference_cell_topology(cell_kind)
    points = np.asarray(
        tuple(
            tuple(1 / 8 + 1 / 8 * float(value) for value in corner)
            for corner in topology.vertices
        ),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "target",
                cell_kind,
                np.arange(len(topology.vertices), dtype=np.int32)[None, :],
                global_ids=np.asarray((906,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(600, 600 + len(topology.vertices), dtype=np.int64),
    )

    (control,) = exact_reference_chart_controls(
        domain,
        mesh,
        CellGeometrySpec.affine(mesh),
        np.asarray((0,), dtype=np.int64),
    )

    assert control.target_cell_kind == cell_kind
    assert control.source_cell_id == 71
    assert len(control.source_reference_vertices) == len(topology.vertices)
    validate_reference_cell_map_witnesses(control.witness)


def test_shared_boundary_has_ambiguous_boundary_owner_but_unique_cell_interior() -> None:
    domain = _mapped_domain(split=True)
    child, geometry = _target(
        np.asarray(
            (
                (0.5, 0.0, 0.0),
                (0.25, 0.25, 0.0),
                (0.25, 0.0, 0.0),
                (0.25, 0.0, 0.25),
            ),
            dtype=np.float64,
        ),
        cell_id=903,
    )

    (control,) = exact_reference_chart_controls(
        domain, child, geometry, np.asarray((0,), dtype=np.int64)
    )

    assert control.source_cell_id == 71
    assert control.source_occurrence_path == ("assembly", "left-root")

    shared_face_cell, shared_face_geometry = _target(
        np.asarray(
            (
                (0.5, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.25, 0.5, 0.0),
            ),
            dtype=np.float64,
        ),
        cell_id=904,
    )
    with pytest.raises(ValueError, match="ambiguous shared-boundary"):
        exact_reference_chart_controls(
            domain,
            shared_face_cell,
            shared_face_geometry,
            np.asarray((0,), dtype=np.int64),
        )


@pytest.mark.parametrize(
    "points",
    (
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, 1.0, 0.0), (2.0, 0.0, 1.0)),
    ),
    ids=("crossing-roots", "outside-roots"),
)
def test_crossing_or_outside_target_cell_is_refused(
    points: tuple[tuple[float, ...], ...],
) -> None:
    domain = _mapped_domain(split=True)
    mesh, geometry = _target(np.asarray(points, dtype=np.float64), cell_id=905)

    with pytest.raises(
        ValueError, match="outside or crosses every exact mapped reference root"
    ):
        exact_reference_chart_controls(
            domain, mesh, geometry, np.asarray((0,), dtype=np.int64)
        )
