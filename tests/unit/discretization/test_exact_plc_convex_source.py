from fractions import Fraction

import equinox as eqx
import numpy as np
import pytest

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import CellGeometrySpec, CellMesh, ExactPlcCellGeometrySource
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometryConvexSource
from phydrax.discretization._coordinate_enclosure import CoordinateSourceBank
from phydrax.discretization._reference_cell_types import CellBlock
from tests._support.coordinate_references import coordinate_reference_budget


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
    maximum_work: int = 100000, *, budget: CoordinateEnclosureBudget
) -> tuple[FixtureArguments, tuple[ExactPoint, ...]]:
    t = Fraction(1, 3)
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
        vertex_parameter_ratios=(
            ((0, 1), (0, 1)),
            ((1, 3), (0, 1)),
            ((0, 1), (0, 1)),
            ((0, 1), (0, 1)),
        ),
        maximum_work=maximum_work,
        budget=budget,
    )
    name = parent.blocks[0].name
    geometry = CellGeometrySpec(
        {name: coordinate_lagrange_element("tetrahedron", 1)},
        {name: np.asarray([[0, 1, 2, 3]])},
        coordinates,
        exact_source=root,
    )
    # Nonbinary convex weights, independently evaluated from the source.
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
    with coordinate_reference_budget(queries_upper=7 * (4 + 4)) as ledger:
        args, expected = _fixture(budget=ledger)
        source = ExactPlcCellGeometryConvexSource(*args, budget=ledger)
        prepared = source.prepare(args[5].coordinates, budget=ledger)
        assert prepared.vertices == expected
        assert prepared.vertices[0][0] != Fraction(float(prepared.vertices[0][0]))
        references = source.cell_parent_reference_corners(budget=ledger)[0]
        assert all(min(row) >= 0 and sum(row) <= 1 for row in references)
        assert source.parent_geometry is args[1]
        assert source.parent_mesh is args[0]

        parent, geometry, target = source.owners()
        assert parent is args[0]
        assert geometry is args[1]
        assert target is args[5]


def test_stored_fields_restore_parent_bindings_and_source_identity() -> None:
    with coordinate_reference_budget(queries_upper=5 * (4 + 4)) as ledger:
        args, expected = _fixture(budget=ledger)
        source = ExactPlcCellGeometryConvexSource(*args, budget=ledger)
        root = source.parent_geometry.exact_source
        assert isinstance(root, ExactPlcCellGeometrySource)
        restored_root = ExactPlcCellGeometrySource.from_stored_fields(root)
        assert restored_root.source_id == root.source_id
        assert restored_root.exact_vertex_parameters() == root.exact_vertex_parameters()
        restored = ExactPlcCellGeometryConvexSource.from_stored_fields(source)
        assert restored.source_id == source.source_id
        parent, geometry, target = restored.owners()
        assert parent is source.parent_mesh
        assert geometry is source.parent_geometry
        assert target is source.target_mesh
        assert restored.preparation_query_upper() == source.preparation_query_upper()
        assert restored.prepare(args[5].coordinates, budget=ledger).vertices == expected


def test_stored_plc_rows_are_revalidated() -> None:
    with coordinate_reference_budget(queries_upper=4 + 4) as ledger:
        args, _ = _fixture(budget=ledger)
        root = args[1].exact_source
        assert isinstance(root, ExactPlcCellGeometrySource)
        changed = eqx.tree_at(
            lambda value: value.vertex_rows,
            root,
            root.vertex_rows.at[1].set(999999),
        )
        with pytest.raises(ValueError, match=r"ExactPlcCellGeometrySource\.vertex_rows"):
            ExactPlcCellGeometrySource.from_stored_fields(changed)


def test_stored_convex_parent_ids_are_revalidated() -> None:
    with coordinate_reference_budget(queries_upper=4 + 4) as ledger:
        args, _ = _fixture(budget=ledger)
        source = ExactPlcCellGeometryConvexSource(*args, budget=ledger)
        changed = eqx.tree_at(
            lambda value: value.cell_parent_ids,
            source,
            source.cell_parent_ids.at[0].set(999999),
        )
        with pytest.raises(
            ValueError, match=r"ExactPlcCellGeometryConvexSource\.cell_parent_ids"
        ):
            ExactPlcCellGeometryConvexSource.from_stored_fields(changed)


@pytest.mark.parametrize("change", ["coefficients", "sci", "support", "rne", "parent"])
def test_constructor_rejects_inauthentic_bindings(change: str) -> None:
    with coordinate_reference_budget(queries_upper=5 * (4 + 4)) as ledger:
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
        ) = _fixture(budget=ledger)
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
                budget=ledger,
            )


def test_original_work_allowance_is_not_reset() -> None:
    with coordinate_reference_budget(queries_upper=5 * (4 + 4)) as ledger:
        args, _ = _fixture(maximum_work=100, budget=ledger)
        with pytest.raises(CoordinateEnclosureResourceError):
            ExactPlcCellGeometryConvexSource(*args, budget=ledger)


def test_changed_support_cannot_reuse_original_binding() -> None:
    with coordinate_reference_budget(queries_upper=6 * (4 + 4)) as ledger:
        args, _ = _fixture(budget=ledger)
        source = ExactPlcCellGeometryConvexSource(*args, budget=ledger)
        changed = eqx.tree_at(
            lambda value: value.cell_parent_ids, source, np.asarray([999999])
        )
        with pytest.raises(
            ValueError, match=r"ExactPlcCellGeometryConvexSource\.original_binding_id"
        ):
            changed.prepare(args[5].coordinates, budget=ledger)


def test_constructor_codec_restores_real_original_owners() -> None:
    with coordinate_reference_budget(queries_upper=6 * (4 + 4)) as ledger:
        from phydrax._model._structure import (
            model_from_array_recipe,
            model_recipe_array_values,
            model_structure_recipe,
        )
        from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts

        register_meshing_source_artifacts()
        args, expected = _fixture(budget=ledger)
        source = ExactPlcCellGeometryConvexSource(*args, budget=ledger)
        recipe = model_structure_recipe(source)
        restored = model_from_array_recipe(
            recipe,
            model_recipe_array_values(source, recipe, prefix="convex-plc"),
            prefix="convex-plc",
        )
        assert isinstance(restored, ExactPlcCellGeometryConvexSource)
        assert restored.source_id == source.source_id
        assert (
            restored.prepare(restored.target_mesh.coordinates, budget=ledger).vertices
            == expected
        )
        assert isinstance(
            restored.parent_geometry.exact_source, ExactPlcCellGeometrySource
        )


def test_q1_whole_law_retains_nonbinary_original_volume() -> None:
    with coordinate_reference_budget(queries_upper=9 * (8 + 8)) as ledger:
        from itertools import product

        from phydrax.discretization._reference_cell import reference_cell_topology

        args, _ = _fixture(budget=ledger)
        parent, geometry = args[:2]
        t = Fraction(1, 3)
        original = (
            (Fraction(0),) * 3,
            (3 * t, t, Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        )
        corners = reference_cell_topology("hexahedron").vertices
        refs = tuple(
            tuple(Fraction(1, 10) + Fraction(float(x)) / 10 for x in row)
            for row in corners
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
            budget=ledger,
        )
        assert source.prepare(target.coordinates, budget=ledger).vertices == points
        target_geometry = CellGeometrySpec.plc(target, source, budget=ledger)
        from phydrax.discretization._cell_geometry_validity import (
            certify_cell_geometry_validity,
        )
        from phydrax.discretization._coordinate_enclosure import (
            coordinate_expressions,
            expression_evaluate,
        )
        from phydrax.discretization._mapped_geometry_measure import (
            certified_cell_measures,
        )

        assert certify_cell_geometry_validity(target_geometry, mesh=target).all_certified
        expressions = coordinate_expressions(
            target_geometry.elements[0],
            target_geometry.source_coordinates(budget=ledger),
            budget=ledger,
        )
        assert expressions is not None
        volumes, errors, exact_measure = certified_cell_measures(
            target, target_geometry, budget=ledger
        )
        assert exact_measure
        expected_volume = (3 * t) / Fraction(10**3)
        assert abs(Fraction(float(volumes[0])) - expected_volume) <= Fraction(
            float(errors[0])
        )
        assert source.cell_parent_reference_corners(budget=ledger) == (refs,)
        # Independently evaluate the entire Q1 law at rational interior points:
        # affine reference coordinates commute with the P1 source law.
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
                tuple(
                    expression_evaluate(value, query, budget=ledger)
                    for value in expressions
                )
                == expected
            )
        assert (3 * t) / Fraction(10**3) != Fraction(float((3 * t) / Fraction(10**3)))


def test_original_parent_inverse_has_real_canonical_receipt_and_refusal() -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.discretization._exact_plc_geometry import (
        convex_parent_inverse,
        ExactPlcBudget,
    )

    matrix = (
        (Fraction(1), Fraction(1, 5), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(1, 7)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    with HostBatchTicket(10000, bytes_upper=1_000_000) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=10000, maximum_memory_bytes=1_000_000
        )
        budget = ExactPlcBudget(10000, 16384, budget=ledger)
        with ledger.activate():
            before = ledger.work_units
            inverse = convex_parent_inverse(matrix, budget, budget=ledger)
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
    with HostBatchTicket(1, bytes_upper=1_000_000) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=1, maximum_memory_bytes=1_000_000
        )
        with (
            ledger.activate(),
            pytest.raises(CoordinateEnclosureResourceError),
        ):
            convex_parent_inverse(
                matrix, ExactPlcBudget(10000, 16384, budget=ledger), budget=ledger
            )


def _coverage_row_occurrence(
    layout: str, budget: CoordinateEnclosureBudget,
) -> tuple[CellMesh, CellGeometrySpec, np.ndarray, CoordinateSourceBank]:
    from phydrax.geometry._global_embedding import coordinate_scope, source_bank_predicate_view
    from phydrax.discretization._coordinate_enclosure import prepared_coordinate_source_bank

    args, _ = _fixture(budget=budget)
    if layout == "convex":
        mesh = CellMesh.from_tetrahedra(
            np.asarray(((0., 0., 0.), (.75, .25, 0.), (0., 1., 0.), (0., 0., 1.))),
            np.asarray(((0, 1, 2, 3),)),
        )
        source = ExactPlcCellGeometryConvexSource(
            args[0], args[1], np.asarray((0, 1, 3, 4, 5)),
            np.asarray((0, 1, 0, 2, 3)),
            (Fraction(1), Fraction(3, 4), Fraction(1, 4), Fraction(1), Fraction(1)),
            mesh, args[6], budget=budget,
        )
        geometry = CellGeometrySpec.plc(mesh, source, budget=budget)
    else:
        mesh, geometry = args[:2]
        if layout == "permuted":
            order = np.asarray((1, 0, 2, 3), dtype=np.int64)
            inverse = np.argsort(order)
            mesh = CellMesh.from_tetrahedra(
                np.asarray(mesh.coordinates)[order],
                inverse[np.asarray(mesh.blocks[0].vertices)],
            )
    prepared = coordinate_scope(mesh, geometry, budget=budget)
    assert prepared.coordinate_scope == "affine"
    bank = prepared.source_values
    points, represented = source_bank_predicate_view(bank, budget=budget)
    assert represented is (layout == "convex")
    if layout == "permuted":
        assert bank is not prepared_coordinate_source_bank(geometry, budget=budget)
    return mesh, geometry, points, bank


@pytest.mark.parametrize("layout", ("root", "permuted", "convex"))
def test_coverage_gathers_actual_nonbinary_plc_mesh_rows_without_reconversion(layout: str) -> None:
    from phydrax.geometry import _domain_coverage as coverage_owner
    from phydrax.geometry._embedding_state import EmbeddingState, MeshCertificateLimits

    values = []
    findings = []
    work = []
    for gather in (False, True):
        with coordinate_reference_budget(queries_upper=256) as budget:
            mesh, geometry, points, bank = _coverage_row_occurrence(layout, budget)
            domain = coverage_owner.PiecewiseLinearDomain(
                np.asarray(((0., 0., 0.), (3., 1., 0.), (0., 1., 0.), (0., 0., 1.))),
                np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))),
                np.tile(np.asarray((0, -1)), (4, 1)), ("material",),
                source_id="exact-plc-bank-coverage",
            )
            state = EmbeddingState([], ["domain_coverage"])
            before = budget.work_units
            phase = coverage_owner._affine_coverage_terms(
                state, mesh, geometry, domain, np.zeros(1, dtype=np.int64),
                points, MeshCertificateLimits(), budget=budget,
                exact_vertex_points=bank if gather else None,
            )
            work.append(budget.work_units - before)
            values.append((phase.requested, phase.achieved, phase.covered, phase.relations))
            findings.append(state.findings)
    assert values[0] == values[1]
    assert eqx.tree_equal(findings[0], findings[1])
    assert work[1] < work[0]


@pytest.mark.parametrize("fault", ("foreign-bank", "coincident-rounded", "foreign-view", "mapped-bank"))
def test_coverage_exact_rows_refuse_foreign_or_rounded_bank_occurrences(fault: str) -> None:
    from phydrax.geometry import _domain_coverage as coverage_owner
    from phydrax.geometry._embedding_state import EmbeddingState, MeshCertificateLimits
    from phydrax.geometry._global_embedding import coordinate_scope, source_bank_predicate_view

    with coordinate_reference_budget(queries_upper=256) as budget:
        if fault == "mapped-bank":
            args, _ = _fixture(budget=budget)
            source = ExactPlcCellGeometryConvexSource(*args, budget=budget)
            mesh = args[5]
            geometry = CellGeometrySpec.plc(mesh, source, budget=budget)
            prepared = coordinate_scope(mesh, geometry, budget=budget)
            assert prepared.coordinate_scope == "mapped"
            bank = prepared.source_values
            points, _ = source_bank_predicate_view(bank, budget=budget)
        else:
            mesh, geometry, points, bank = _coverage_row_occurrence("permuted", budget)
        if fault == "foreign-view":
            points = points.copy()
            value = points[0, 1]
            assert type(value) is Fraction
            points[0, 1] = Fraction(value.numerator, value.denominator)
            assert points[0, 1] is not value
            points.flags.writeable = False
        elif fault != "mapped-bank":
            foreign = tuple(
                tuple(Fraction(float(value)) if fault == "coincident-rounded" else value for value in row)
                for row in bank
            )
            assert foreign is not bank
            if fault == "coincident-rounded":
                assert foreign != bank
                assert np.array_equal(
                    np.asarray(foreign, dtype=np.float64), np.asarray(bank, dtype=np.float64)
                )
            bank = foreign
            points, _ = source_bank_predicate_view(bank, budget=budget)
        domain = coverage_owner.PiecewiseLinearDomain(
            np.asarray(((0., 0., 0.), (3., 1., 0.), (0., 1., 0.), (0., 0., 1.))),
            np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))),
            np.tile(np.asarray((0, -1)), (4, 1)), ("material",),
            source_id="exact-plc-bank-coverage",
        )
        with pytest.raises(ValueError, match="actual prepared source"):
            coverage_owner._affine_coverage_terms(
                EmbeddingState([], ["domain_coverage"]), mesh, geometry, domain,
                np.zeros(1, dtype=np.int64), points, MeshCertificateLimits(),
                budget=budget, exact_vertex_points=bank,
            )
