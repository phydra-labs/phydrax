from fractions import Fraction

import equinox as eqx
import numpy as np
import pytest

from phydrax.discretization import CellGeometrySpec, CellMesh, ExactPlcCellGeometrySource
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._cell_mesh import CellBlock
from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureResourceError
from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometryConvexSource


type ExactPoint = tuple[Fraction, ...]
type FixtureArguments = tuple[
    CellMesh,
    CellGeometrySpec,
    np.ndarray,
    np.ndarray,
    tuple[Fraction, ...],
    CellMesh,
    np.ndarray,
]


def _fixture(
    maximum_work: int = 100000,
) -> tuple[FixtureArguments, tuple[ExactPoint, ...]]:
    t = Fraction(1.0 / 3.0)
    exact = (
        (Fraction(0),) * 3,
        (3 * t, t, Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    coordinates = np.asarray([[float(x) for x in row] for row in exact])
    parent = CellMesh.from_tetrahedra(coordinates, np.asarray([[0, 1, 2, 3]]))
    root = ExactPlcCellGeometrySource(
        np.asarray([[0.0, 0.0, 0.0], [3.0, 1.0, 0.0]]),
        np.empty((0, 3), dtype=np.int64),
        np.asarray([[0, 1]]),
        np.asarray([0, 1, 0, 0]),
        np.asarray([-1, 0, -1, -1]),
        np.asarray([[0.0, 0.0], [float(t), 0.0], [0.0, 0.0], [0.0, 0.0]]),
        domain_source_id="convex-test",
        domain_source_revision="authored",
        source_triangle_ids=np.empty(0, dtype=np.int64),
        source_triangle_bounds=np.empty(0),
        source_segment_ids=np.asarray([7]),
        source_segment_bounds=np.asarray([1e-15]),
        maximum_work=maximum_work,
    )
    name = parent.blocks[0].name
    geometry = CellGeometrySpec(
        {name: coordinate_lagrange_element("tetrahedron", 1)},
        {name: np.asarray([[0, 1, 2, 3]])},
        coordinates,
        exact_source=root,
    )
    # Nonbinary convex weights, independently evaluated from the original source.
    weights = (Fraction(4, 5), Fraction(1, 5)) * 4
    vertices = np.asarray([0, 1, 1, 2, 2, 3, 3, 0])
    expected = tuple(
        tuple(
            weights[2 * i] * exact[vertices[2 * i]][a]
            + weights[2 * i + 1] * exact[vertices[2 * i + 1]][a]
            for a in range(3)
        )
        for i in range(4)
    )
    target = CellMesh.from_tetrahedra(
        np.asarray([[float(x) for x in row] for row in expected]),
        np.asarray([[0, 1, 2, 3]]),
    )
    args = (
        parent,
        geometry,
        np.arange(0, 9, 2),
        vertices,
        weights,
        target,
        np.asarray(parent.blocks[0].global_ids),
    )
    return args, expected


def test_exact_original_nonbinary_convex_images() -> None:
    args, expected = _fixture()
    source = ExactPlcCellGeometryConvexSource(*args)
    prepared = source.prepare(args[5].coordinates)
    assert prepared.vertices == expected
    assert prepared.vertices[0][0] != Fraction(float(prepared.vertices[0][0]))
    references = source.cell_parent_reference_corners[0]
    assert all(min(row) >= 0 and sum(row) <= 1 for row in references)
    assert source.parent_geometry is args[1]
    assert source.parent_mesh is args[0]


@pytest.mark.parametrize("change", ["coefficients", "sci", "support", "rne", "parent"])
def test_constructor_rejects_inauthentic_bindings(change: str) -> None:
    (
        (
            parent_mesh,
            parent_geometry,
            offsets,
            vertices,
            coefficients,
            target_mesh,
            parents,
        ),
        _,
    ) = _fixture()
    geometry_owner: object = parent_geometry
    if change == "coefficients":
        coefficients = (Fraction(-1), Fraction(2)) + coefficients[2:]
    elif change == "sci":
        parents = np.asarray([999999])
    elif change == "support":
        vertices = np.asarray([999] + vertices[1:].tolist())
    elif change == "rne":
        coords = np.asarray(target_mesh.coordinates).copy()
        coords[0, 0] = np.nextafter(coords[0, 0], np.inf)
        target_mesh = CellMesh.from_tetrahedra(coords, np.asarray([[0, 1, 2, 3]]))
    else:
        geometry_owner = object()
    with pytest.raises((ValueError, TypeError)):
        ExactPlcCellGeometryConvexSource(
            parent_mesh,
            geometry_owner,  # ty: ignore[invalid-argument-type]
            offsets,
            vertices,
            coefficients,
            target_mesh,
            parents,
        )


def test_original_work_allowance_is_not_reset() -> None:
    args, _ = _fixture(maximum_work=100)
    with pytest.raises(CoordinateEnclosureResourceError):
        ExactPlcCellGeometryConvexSource(*args)


def test_changed_support_cannot_reuse_original_binding() -> None:
    args, _ = _fixture()
    source = ExactPlcCellGeometryConvexSource(*args)
    changed = eqx.tree_at(
        lambda value: value.cell_parent_ids, source, np.asarray([999999])
    )
    with pytest.raises(ValueError, match="binding changed"):
        changed.prepare(args[5].coordinates)


def test_constructor_codec_restores_real_original_owners() -> None:
    from phydrax._model._structure import (
        model_from_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts

    register_meshing_source_artifacts()
    args, expected = _fixture()
    source = ExactPlcCellGeometryConvexSource(*args)
    recipe = model_structure_recipe(source)
    restored = model_from_array_recipe(
        recipe,
        model_recipe_array_values(source, recipe, prefix="convex-plc"),
        prefix="convex-plc",
    )
    assert isinstance(restored, ExactPlcCellGeometryConvexSource)
    assert restored.source_id == source.source_id
    assert restored.prepare(restored.target_mesh.coordinates).vertices == expected
    assert isinstance(restored.parent_geometry.exact_source, ExactPlcCellGeometrySource)


def test_q1_whole_law_retains_nonbinary_original_volume() -> None:
    from itertools import product

    from phydrax.discretization._reference_cell import reference_cell_topology

    args, _ = _fixture()
    parent, geometry = args[:2]
    t = Fraction(1.0 / 3.0)
    original = (
        (Fraction(0),) * 3,
        (3 * t, t, Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    corners = reference_cell_topology("hexahedron").vertices
    refs = tuple(
        tuple(Fraction(1, 10) + Fraction(float(x)) / 10 for x in row) for row in corners
    )
    coefficients = tuple(
        value for row in refs for value in (Fraction(1) - sum(row), *row)
    )
    points = tuple(
        tuple(
            sum(
                (coefficients[4 * i + j] * original[j][axis] for j in range(4)),
                Fraction(0),
            )
            for axis in range(3)
        )
        for i in range(8)
    )
    target = CellMesh(
        np.asarray([[float(x) for x in row] for row in points]),
        (CellBlock("hexes", "hexahedron", np.arange(8)[None, :]),),
    )
    source = ExactPlcCellGeometryConvexSource(
        parent,
        geometry,
        np.arange(0, 33, 4),
        np.tile(np.arange(4), 8),
        coefficients,
        target,
        np.asarray(parent.blocks[0].global_ids),
    )
    assert source.prepare(target.coordinates).vertices == points
    target_geometry = CellGeometrySpec.plc(target, source)
    from phydrax.discretization._cell_geometry_transfer import _certified_cell_measures
    from phydrax.discretization._cell_geometry_validity import (
        certify_cell_geometry_validity,
    )
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_expressions,
        expression_evaluate,
    )

    assert certify_cell_geometry_validity(target_geometry, mesh=target).all_certified
    expressions = coordinate_expressions(
        target_geometry.elements[0], target_geometry.source_coordinates()
    )
    assert expressions is not None
    volumes, errors, exact_measure = _certified_cell_measures(target, target_geometry)
    assert exact_measure
    expected_volume = (3 * t) / Fraction(10**3)
    assert abs(Fraction(float(volumes[0])) - expected_volume) <= Fraction(
        float(errors[0])
    )
    assert source.cell_parent_reference_corners == (refs,)
    # Independently evaluate the entire Q1 law at rational interior points:
    # affine reference coordinates commute with the original P1 source law.
    for query in product((Fraction(1, 5), Fraction(2, 3)), repeat=3):
        shape = tuple(
            np.prod([query[a] if row[a] else 1 - query[a] for a in range(3)])
            for row in corners
        )
        image = tuple(
            sum((shape[i] * points[i][axis] for i in range(8)), Fraction(0))
            for axis in range(3)
        )
        reference = tuple(Fraction(1, 10) + query[a] / 10 for a in range(3))
        expected = tuple(
            reference[0] * original[1][a]
            + reference[1] * original[2][a]
            + reference[2] * original[3][a]
            for a in range(3)
        )
        assert image == expected
        assert (
            tuple(expression_evaluate(value, query) for value in expressions) == expected
        )
    assert (3 * t) / Fraction(10**3) != Fraction(float((3 * t) / Fraction(10**3)))


def test_original_parent_inverse_has_real_canonical_receipt_and_refusal() -> None:
    from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureBudget
    from phydrax.discretization._exact_plc_geometry import (
        _convex_parent_inverse,
        _ExactPlcBudget,
    )

    matrix = (
        (Fraction(1), Fraction(1, 5), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(1, 7)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    budget = _ExactPlcBudget(10000, 16384)
    with CoordinateEnclosureBudget(10000, 1_000_000).activate() as ledger:
        before = ledger.work_units
        inverse = _convex_parent_inverse(matrix, budget)
        assert budget.work == ledger.work_units - before > 0
    product = tuple(
        tuple(
            sum(
                (matrix[row][index] * inverse[index][column] for index in range(3)),
                Fraction(0),
            )
            for column in range(3)
        )
        for row in range(3)
    )
    assert product == tuple(
        tuple(Fraction(int(row == column)) for column in range(3)) for row in range(3)
    )
    with (
        CoordinateEnclosureBudget(1, 1_000_000).activate(),
        pytest.raises(CoordinateEnclosureResourceError),
    ):
        _convex_parent_inverse(matrix, _ExactPlcBudget(10000, 16384))
