#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction

import equinox as eqx
import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax._model._structure import (
    model_from_array_recipe,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    ExactPlcCellGeometrySource,
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometrySource,
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization._cell_geometry import (
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import (
    _certified_cell_measures,
    transition_displaced_cell_geometry,
)
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from phydrax.discretization._coordinate_enclosure import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry._supermesh import prepare_common_refinement
from phydrax.lifecycle._meshing_sources import (
    register_meshing_source_artifacts,
    validate_meshing_source_closure,
)


# Binary64 1/3. The Steiner vertex S = (3t, t, 0) on PLC edge (0,0,0)-(3,1,0)
# has x = 1 - 2**-54, so its RNE carrier (1, t, 0) leaves the source edge.
_T = 1.0 / 3.0
_STEINER = (Fraction(_T) * 3, Fraction(_T), Fraction(0))


def _plc_source(
    *, parameters: np.ndarray | None = None, strata: np.ndarray | None = None
) -> ExactPlcCellGeometrySource:
    return ExactPlcCellGeometrySource(
        np.asarray(
            [[0.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        np.asarray([[0, 2, 3]], dtype=np.int64),
        np.asarray([[0, 1]], dtype=np.int64),
        np.asarray([0, 1, 2, 0], dtype=np.int8) if strata is None else strata,
        np.asarray([-1, 0, 0, -1], dtype=np.int64),
        np.asarray([[0.0, 0.0], [_T, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        if parameters is None
        else parameters,
        domain_source_id="exact-source-slot-plc",
        domain_source_revision="exact-source-slot-plc-authored",
        source_triangle_ids=np.asarray([0], dtype=np.int64),
        source_triangle_bounds=np.asarray([1e-15], dtype=np.float64),
        source_segment_ids=np.asarray([0], dtype=np.int64),
        source_segment_bounds=np.asarray([1e-15], dtype=np.float64),
    )


def _plc_mesh() -> CellMesh:
    carrier = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [float(_STEINER[0]), _T, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return CellMesh.from_tetrahedra(carrier, np.asarray([[0, 1, 2, 3]], dtype=np.int64))


def _power_source() -> ExactPowerCellGeometrySource:
    return ExactPowerCellGeometrySource(
        np.asarray([[0.0, 0.0, 0.0], [0.7, 0.6, 0.8]], dtype=np.float64),
        np.asarray([0.03, -0.02], dtype=np.float64),
        np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        np.asarray([0, 1, 2, 3, 4, 6, 8, 10], dtype=np.int64),
        np.asarray([0, 0, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32),
        np.asarray(
            [
                [0, -1, -1, -1],
                [1, -1, -1, -1],
                [2, -1, -1, -1],
                [3, -1, -1, -1],
                [0, 3, -1, -1],
                [1, 3, -1, -1],
                [2, 3, -1, -1],
            ],
            dtype=np.int32,
        ),
    )


def _periodic_power_source() -> ExactPowerCellGeometrySource:
    from phydrax.discretization import PeriodicCell
    from phydrax.geometry._triangulation import PeriodicPowerPreparation

    points = np.asarray([[0.2, 0.2, 0.2]])
    weights = np.zeros(1)
    carrier = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    preparation = PeriodicPowerPreparation(
        points, weights, carrier, PeriodicCell(4 * np.eye(3)), maximum_images=1000
    )
    original_image = int(
        np.flatnonzero(np.all(preparation.image_exponents == 0, axis=1))[0]
    )
    return ExactPowerCellGeometrySource(
        points,
        weights,
        carrier,
        np.asarray([[0, 1, 2, 3]]),
        np.arange(5),
        np.full(4, original_image),
        np.column_stack((np.arange(4), np.full((4, 3), -1))),
        periodic_preparation=preparation,
    )


def test_periodic_exact_power_source_retains_image_axis_and_original_law() -> None:
    source = _periodic_power_source()
    prepared = source.prepare(source.carrier_points)
    assert prepared.vertices == tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(source.carrier_points)
    )
    periodic = source._periodic_owner()
    if periodic is None:
        raise RuntimeError("Periodic source lost its retained preparation.")
    assert periodic.points.shape == (1, 3)
    with pytest.raises(ValueError, match="undeclared sites or images"):
        ExactPowerCellGeometrySource(
            source.site_points,
            source.site_weights,
            source.carrier_points,
            source.carrier_tets,
            source.vertex_site_offsets,
            np.full(4, len(periodic.image_sites)),
            source.vertex_carriers,
            periodic_preparation=periodic,
        )
    moved = eqx.tree_at(
        lambda value: value.site_points, source, source.site_points + 0.01
    )
    with pytest.raises(ValueError, match="retained original preparation"):
        moved.prepare()


def test_exact_power_linear_action_uses_pretransform_exact_parent_coefficients() -> None:
    source = _periodic_power_source()
    owner = ExactPowerCellGeometryLinearActionSource(
        source,
        np.asarray([[0, -1], [0, 1]]),
        ((Fraction(1), Fraction(0)), (Fraction(1, 3), Fraction(2, 3))),
        np.asarray([[0, 0, 0], [1, 0, 0]]),
        periodic_preparation=source.periodic_preparation,
    )
    prepared = owner.prepare()
    assert prepared.vertices == (
        (Fraction(0),) * 3,
        (Fraction(14, 3), Fraction(0), Fraction(0)),
    )
    owner.prepare(prepared.rounded_vertices)
    corrupted = prepared.rounded_vertices.copy()
    corrupted[1, 0] = np.nextafter(corrupted[1, 0], np.inf)
    with pytest.raises(ValueError, match="not the RNE"):
        owner.prepare(corrupted)
    with pytest.raises(ValueError, match="convex unit"):
        ExactPowerCellGeometryLinearActionSource(
            source,
            np.asarray([[0, 1]]),
            ((Fraction(1), Fraction(1)),),
            np.zeros((1, 3), dtype=np.int64),
            periodic_preparation=source.periodic_preparation,
        )
    changed_parent = eqx.tree_at(
        lambda value: value.parent.site_weights, owner, np.ones(1)
    )
    with pytest.raises(ValueError, match="immutable source"):
        changed_parent.prepare()


def test_nonperiodic_exact_convex_source_defines_actual_q1_hexahedron() -> None:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._coordinate_enclosure import coordinate_corner_images
    from phydrax.discretization._reference_cell import reference_cell_topology

    carrier = np.asarray(
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]]
    )
    parent = ExactPowerCellGeometrySource(
        np.zeros((1, 3)),
        np.zeros(1),
        carrier,
        np.asarray([[0, 1, 2, 3]]),
        np.arange(5),
        np.zeros(4, dtype=np.int64),
        np.column_stack((np.arange(4), np.full((4, 3), -1))),
    )
    corners = reference_cell_topology("hexahedron").vertices
    coefficients = tuple(
        (
            1 - sum((Fraction(int(value), 3) for value in corner), Fraction(0)),
            *(Fraction(int(value), 3) for value in corner),
        )
        for corner in corners
    )
    source = ExactPowerCellGeometryLinearActionSource(
        parent,
        np.tile(np.arange(4), (8, 1)),
        coefficients,
        np.empty((8, 0), dtype=np.int64),
    )
    prepared = source.prepare()
    mesh = CellMesh(
        prepared.rounded_vertices,
        (CellBlock("hexes", "hexahedron", np.arange(8)[None, :]),),
    )
    geometry = CellGeometrySpec.power(mesh, source)
    exact = tuple(
        tuple(Fraction(0.3) * int(value) / 3 for value in corner) for corner in corners
    )
    assert geometry.source_coordinates() == exact
    element = geometry.elements[0]
    assert element.element_id == coordinate_lagrange_element("hexahedron", 1).element_id
    assert coordinate_corner_images(element, exact) == exact
    assert any(
        Fraction(float(value)) != ideal
        for row, exact_row in zip(prepared.rounded_vertices, exact, strict=True)
        for value, ideal in zip(row, exact_row, strict=True)
    )
    certificate = certify_cell_geometry_validity(geometry, mesh=mesh)
    assert certificate.all_certified
    volumes, errors, ideal_measure = _certified_cell_measures(mesh, geometry)
    assert ideal_measure
    expected = (Fraction(0.3) / 3) ** 3
    assert abs(Fraction(float(volumes[0])) - expected) <= Fraction(float(errors[0]))
    with pytest.raises(ValueError, match="aligned parent and generator"):
        ExactPowerCellGeometryLinearActionSource(
            parent, np.asarray([[0]]), ((Fraction(1),),), np.zeros((1, 1), dtype=np.int64)
        )


def test_exact_plc_q1_chart_retains_original_bank_and_refuses_outside_simplex() -> None:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_corner_images,
        coordinate_expressions,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology

    original_mesh, source = _plc_mesh(), _plc_source()
    original_geometry = CellGeometrySpec.plc(original_mesh, source)
    bank = original_geometry.source_coordinates()
    reference = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.int64)
    element = PolynomialComposedCellGeometryElement(
        coordinate_lagrange_element("tetrahedron", 1),
        coordinate_lagrange_element("hexahedron", 1),
        reference,
        np.full((8, 3), 4, dtype=np.int64),
    )
    images = coordinate_corner_images(element, bank)
    assert images is not None
    mesh = CellMesh(
        np.asarray([[float(value) for value in row] for row in images]),
        (CellBlock("hexes", "hexahedron", np.arange(8)[None, :]),),
    )
    geometry = CellGeometrySpec(
        {"hexes": element},
        {"hexes": np.arange(4)[None, :]},
        original_geometry.coordinates,
        exact_source=source,
    )
    geometry.resolve(mesh)
    assert geometry.exact_source is source
    assert geometry.source_coordinates() == bank
    expressions = coordinate_expressions(element, bank)
    assert expressions is not None
    middle = (Fraction(1, 2),) * 3
    from phydrax.discretization._coordinate_enclosure import expression_evaluate

    point = tuple(expression_evaluate(expression, middle) for expression in expressions)
    expected = tuple(
        sum((bank[index][axis] for index in (1, 2, 3)), Fraction(0)) / 8
        for axis in range(3)
    )
    assert point == expected
    assert certify_cell_geometry_validity(geometry, mesh=mesh).all_certified
    volumes, errors, exact_measure = _certified_cell_measures(mesh, geometry)
    assert exact_measure
    expected_volume = Fraction(_T) * 3 / 64
    assert abs(Fraction(float(volumes[0])) - expected_volume) <= Fraction(
        float(errors[0])
    )
    for bad_corner in ((-1, 0, 0), (3, 3, 3)):
        invalid = reference.copy()
        invalid[0] = bad_corner
        wrong = PolynomialComposedCellGeometryElement(
            coordinate_lagrange_element("tetrahedron", 1),
            coordinate_lagrange_element("hexahedron", 1),
            invalid,
            np.full((8, 3), 4, dtype=np.int64),
        )
        with pytest.raises(ValueError, match="original source simplex"):
            CellGeometrySpec(
                {"hexes": wrong},
                {"hexes": np.arange(4)[None, :]},
                original_geometry.coordinates,
                exact_source=source,
            )


def test_exact_plc_geometry_keeps_source_coordinates_distinct_from_carrier() -> None:
    mesh, source = _plc_mesh(), _plc_source()
    geometry = CellGeometrySpec.plc(mesh, source)
    assert geometry.exact_source is source
    assert source.prepare(mesh.coordinates).on_entity == (True, False, True, True)
    assert np.array_equal(
        np.asarray(geometry.coordinates).view(np.uint64),
        np.asarray(mesh.coordinates).view(np.uint64),
    )
    exact = geometry.source_coordinates()
    assert exact[1] == _STEINER
    assert exact[1][0] != Fraction(float(np.asarray(geometry.coordinates)[1, 0]))
    assert geometry.source_execution_error(mesh) > 0.0
    # The exact source volume t/2 is representable; the carrier volume 1/6 is not.
    volumes, errors, exact_measure = _certified_cell_measures(mesh, geometry)
    assert volumes[0] == _T / 2 and errors[0] == 0.0 and exact_measure
    finite_volume = UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare(
        cell_geometry=geometry
    )
    assert float(finite_volume.cell_volumes[0]) == _T / 2
    assert float(finite_volume.cell_volume_error_bounds[0]) == 0.0
    affine = CellGeometrySpec.affine(mesh)
    assert affine.exact_source is None
    assert geometry.geometry_layout_id != affine.geometry_layout_id


def test_exact_plc_witness_bank_refuses_before_storage_allocation() -> None:
    mesh, source = _plc_mesh(), _plc_source()
    geometry = CellGeometrySpec.plc(mesh, source)
    budget = CoordinateEnclosureBudget(1_000_000, 1)
    with budget.activate(), pytest.raises(CoordinateEnclosureResourceError) as refusal:
        geometry.source_coordinates()
    assert refusal.value.resource == "polynomial_storage"
    assert refusal.value.limit == 1
    assert refusal.value.completed == 0
    assert refusal.value.requested > refusal.value.limit
    assert budget.retained_basis_bytes == 0


def test_exact_plc_geometry_persistence_roundtrip_renews_source_authority() -> None:
    register_meshing_source_artifacts()
    mesh, source = _plc_mesh(), _plc_source()
    geometry = CellGeometrySpec.plc(mesh, source)
    validate_meshing_source_closure(geometry)
    recipe = model_structure_recipe(geometry)
    restored = model_from_array_recipe(
        recipe, model_recipe_array_values(geometry, recipe, prefix="plc"), prefix="plc"
    )
    validate_meshing_source_closure(restored)
    if not isinstance(restored.exact_source, ExactPlcCellGeometrySource):
        raise AssertionError("The restored geometry lost its exact PLC source kind.")
    assert restored.exact_source.source_id == source.source_id
    assert restored.exact_source.domain_source_id == source.domain_source_id
    assert cell_geometry_id(restored) == cell_geometry_id(geometry)
    assert restored.source_coordinates() == geometry.source_coordinates()
    # A substituted witness is not the RNE ancestry of the published carrier.
    forged = eqx.tree_at(
        lambda value: value.exact_source.vertex_parameters,
        restored,
        np.asarray([[0.0, 0.0], [0.25, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float64),
    )
    with pytest.raises(ValueError):
        validate_meshing_source_closure(forged)


def test_exact_source_slot_refuses_wrong_kind_family_and_foreign_authority() -> None:
    mesh, source = _plc_mesh(), _plc_source()
    element = coordinate_lagrange_element("tetrahedron", 1)
    with pytest.raises(TypeError):
        CellGeometrySpec(
            {"tetrahedra": element},
            {"tetrahedra": mesh.blocks[0].vertices},
            mesh.coordinates,
            exact_source=object(),  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(TypeError):
        CellGeometrySpec.power(mesh, source)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        CellGeometrySpec.plc(mesh, _power_source())  # ty: ignore[invalid-argument-type]
    power = _power_source()
    with pytest.raises(ValueError, match="polyhedral layout"):
        CellGeometrySpec(
            {"tetrahedra": element},
            {"tetrahedra": mesh.blocks[0].vertices},
            power.prepare().rounded_vertices,
            exact_source=power,
        )
    triangles = CellMesh.from_triangles(
        np.asarray(mesh.coordinates)[:3], np.asarray([[0, 1, 2]], dtype=np.int64)
    )
    with pytest.raises(ValueError, match="direct global canonical carrier"):
        CellGeometrySpec.plc(triangles, source)
    with pytest.raises(ValueError, match="affine tetrahedral"):
        CellGeometrySpec(
            {"tetrahedra": coordinate_lagrange_element("tetrahedron", 2)},
            {"tetrahedra": np.arange(10)[None]},
            np.zeros((10, 3), dtype=np.float64),
            exact_source=source,
        )
    # Witnesses for a different carrier vertex count.
    short = ExactPlcCellGeometrySource(
        source.source_points,
        source.source_triangles,
        source.source_segments,
        np.asarray(source.vertex_strata)[:3],
        np.asarray(source.vertex_rows)[:3],
        np.asarray(source.vertex_parameters)[:3],
        domain_source_id=source.domain_source_id,
        domain_source_revision=source.domain_source_revision,
        source_triangle_ids=source.source_triangle_ids,
        source_triangle_bounds=source.source_triangle_bounds,
        source_segment_ids=source.source_segment_ids,
        source_segment_bounds=source.source_segment_bounds,
    )
    with pytest.raises(ValueError):
        CellGeometrySpec.plc(mesh, short)
    geometry = CellGeometrySpec.plc(mesh, source)
    moved = np.asarray(mesh.coordinates).copy()
    moved[1, 0] = np.nextafter(moved[1, 0], 0.0)
    with pytest.raises(ValueError):
        geometry.with_coordinates(moved)
    foreign = mesh.with_coordinates(
        np.asarray(mesh.coordinates) * 2.0, numeric_version="foreign"
    )
    with pytest.raises(ValueError, match="carrier differs"):
        geometry.resolve(foreign)
    with pytest.raises(ValueError, match="quotient"):
        geometry.with_periodic_source(mesh)
    with pytest.raises(ValueError, match="PLC source ancestry"):
        transition_displaced_cell_geometry(mesh, geometry, mesh)
    with pytest.raises(ValueError, match="not power polyhedra"):
        prepare_common_refinement(
            mesh, mesh, source_geometry=geometry, target_geometry=geometry
        )


def test_unbounded_integer_reference_chart_keeps_exact_measure_and_archive_identity() -> (
    None
):
    numerator, denominator = 2**150 + 7, 2**151 + 11
    ratio = Fraction(numerator, denominator)
    basis = coordinate_lagrange_element("tetrahedron", 1)
    numerators = np.asarray(
        [[0, 0, 0], [numerator, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=object
    )
    denominators = np.asarray(
        [[1, 1, 1], [denominator, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=object
    )
    element = PolynomialComposedCellGeometryElement(
        basis, basis, numerators, denominators
    )
    original = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    carrier = original.copy()
    carrier[1, 0] = float(ratio)
    mesh = CellMesh.from_tetrahedra(carrier, np.asarray([[0, 1, 2, 3]], dtype=np.int32))
    name = mesh.blocks[0].name
    geometry = CellGeometrySpec(
        {name: element}, {name: mesh.blocks[0].vertices}, original
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    assert validity.all_certified
    assert (
        Fraction(float(validity.determinant_lower[0]))
        <= ratio
        <= Fraction(float(validity.determinant_upper[0]))
    )
    measures, bounds, exact = _certified_cell_measures(mesh, geometry)
    assert exact
    assert abs(Fraction(float(measures[0])) - ratio / 6) <= Fraction(float(bounds[0]))
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(geometry)
    restored = model_from_array_recipe(
        recipe,
        model_recipe_array_values(geometry, recipe, prefix="rational-chart"),
        prefix="rational-chart",
    )
    validate_meshing_source_closure(restored)
    assert cell_geometry_id(restored) == cell_geometry_id(geometry)
    restored_measures, restored_bounds, restored_exact = _certified_cell_measures(
        mesh, restored
    )
    assert restored_exact
    assert abs(Fraction(float(restored_measures[0])) - ratio / 6) <= Fraction(
        float(restored_bounds[0])
    )


@pytest.mark.parametrize("invalid", (True, Fraction(1, 2), np.int64(1), 1.0))
def test_reference_chart_object_banks_refuse_non_python_integer_entries(
    invalid: object,
) -> None:
    basis = coordinate_lagrange_element("tetrahedron", 1)
    numerator = np.zeros((4, 3), dtype=object)
    denominator = np.ones((4, 3), dtype=object)
    numerator[0, 0] = invalid
    with pytest.raises(TypeError, match="Python integers"):
        PolynomialComposedCellGeometryElement(basis, basis, numerator, denominator)


def test_plc_bank_refuses_tiny_active_memory_before_allocation_and_retains_lifetime() -> (
    None
):
    mesh, source = _plc_mesh(), _plc_source()
    geometry = CellGeometrySpec.plc(mesh, source)
    identity = source.source_id
    tiny = CoordinateEnclosureBudget(1_000_000, 1)
    with tiny.activate(), pytest.raises(CoordinateEnclosureResourceError) as refusal:
        geometry.source_coordinates()
    assert refusal.value.resource == "polynomial_storage"
    assert refusal.value.completed == 0 and refusal.value.requested > 1
    assert tiny.peak_bytes_upper == 0
    assert source.source_id == identity
    assert geometry.source_coordinates()[1] == _STEINER

    ledger = CoordinateEnclosureBudget(1_000_000, 1 << 24)
    with ledger.activate():
        bank = geometry.source_coordinates()
        assert bank[1] == _STEINER
        assert ledger.retained_basis_bytes > 0
        assert ledger.temporary_bytes_upper == 0
        remaining = ledger.maximum_memory_bytes - ledger.retained_basis_bytes
        with pytest.raises(CoordinateEnclosureResourceError):
            ledger.reserve(0, remaining + 1)
    assert ledger.peak_bytes_upper > 0


@pytest.mark.parametrize("resource", ("work", "memory"))
def test_standalone_plc_certificate_scope_shares_request_resource_ledger(
    resource: str,
) -> None:
    from phydrax.geometry._mesh_certificates import (
        certify_domain_coverage,
        certify_global_embedding,
        MeshCertificateLimits,
        PiecewiseLinearDomain,
    )

    mesh, source = _plc_mesh(), _plc_source()
    geometry = CellGeometrySpec.plc(mesh, source)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"
    identity = source.source_id
    limits = MeshCertificateLimits(
        maximum_work_units=1 if resource == "work" else 1_000_000,
        maximum_scratch_bytes=1 if resource == "memory" else 1 << 24,
    )
    refused = certify_global_embedding(mesh, geometry, validity, limits=limits)
    assert refused.status == "unresolved"
    assert any(
        finding.check == "exact_source_resource_budget" for finding in refused.findings
    )
    assert refused.source_expression_work_units <= limits.maximum_work_units
    assert refused.source_expression_peak_bytes <= limits.maximum_scratch_bytes
    domain = PiecewiseLinearDomain(
        mesh.coordinates,
        np.asarray([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int64),
        np.asarray([[0, -1]] * 4, dtype=np.int64),
        ("volume",),
        source_id="plc-ledger-domain",
    )
    coverage = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        np.asarray([0], dtype=np.int64),
        embedding=embedding,
        limits=limits,
    )
    assert coverage.status == "unresolved"
    assert any(
        finding.check == "mapped_coverage_resource_budget"
        for finding in coverage.findings
    )
    assert coverage.source_expression_work_units is not None
    assert coverage.source_expression_work_units <= limits.maximum_work_units
    assert coverage.source_expression_peak_bytes is not None
    assert coverage.source_expression_peak_bytes <= limits.maximum_scratch_bytes
    assert source.source_id == identity
    assert geometry.source_coordinates()[1] == _STEINER


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable")
def test_plc_bank_and_native_buffers_share_one_original_scratch_allowance() -> None:
    from phydrax._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionBudget

    mesh, source = _plc_mesh(), _plc_source()
    geometry = CellGeometrySpec.plc(mesh, source)
    identity = source.source_id
    ledger = CoordinateEnclosureBudget(1_000_000, 1 << 24)
    native = NativeExecutionBudget(
        max_work=1_000_000,
        max_geometry_queries=1_000_000,
        max_cavity_cells=1_000_000,
        max_scratch_bytes=1 << 20,
        max_wall_seconds=np.inf,
    )
    with pytest.raises(MeshcoreError) as final_refusal, native:
        with ledger.activate():
            bank = geometry.source_coordinates()
            assert bank[1] == _STEINER
            available = native.remaining().remaining_scratch_bytes
            assert available < (1 << 20) - ledger.retained_basis_bytes
            with pytest.raises(MeshcoreError) as refusal:
                native.allocate_host_array((available + 1,), np.uint8)
            assert refusal.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
            assert bank[1] == _STEINER
            assert source.source_id == identity
    assert final_refusal.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert native.evidence is not None
    assert native.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert native.evidence.host_storage_live_bytes_upper == 0
    assert native.evidence.host_storage_peak_bytes_upper >= ledger.retained_basis_bytes
    assert native.evidence.memory_evidence[2] <= 1 << 20


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable")
def test_plc_bank_refuses_original_native_tiny_scratch_before_rational_allocation() -> (
    None
):
    from phydrax._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionBudget

    mesh, source = _plc_mesh(), _plc_source()
    geometry = CellGeometrySpec.plc(mesh, source)
    identity = source.source_id
    ledger = CoordinateEnclosureBudget(1_000_000, 1 << 24)
    native = NativeExecutionBudget(
        max_work=1_000_000,
        max_geometry_queries=1_000_000,
        max_cavity_cells=1_000_000,
        max_scratch_bytes=32,
        max_wall_seconds=np.inf,
    )
    with pytest.raises(MeshcoreError) as final_refusal, native:
        with (
            ledger.activate(),
            pytest.raises(CoordinateEnclosureResourceError) as refusal,
        ):
            geometry.source_coordinates()
        assert refusal.value.resource == "polynomial_storage"
        assert refusal.value.requested > 32 and refusal.value.completed == 0
        assert ledger.retained_basis_bytes == 0 and ledger.peak_bytes_upper == 0
    assert final_refusal.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert native.evidence is not None
    assert native.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert native.evidence.host_storage_peak_bytes_upper == 0
    assert source.source_id == identity
    assert geometry.source_coordinates()[1] == _STEINER


def test_coordinate_live_storage_escapes_nested_temporary_scopes_and_shrinks() -> None:
    ledger = CoordinateEnclosureBudget(100, 4096)
    with ledger.live_storage() as live:
        live.set_bound(128)
        with ledger.temporary_scope():
            ledger.reserve(3, 512)
            with ledger.temporary_scope():
                live.set_bound(384)
                ledger.reserve(2, 256)
            assert ledger.temporary_bytes_upper == 896
        assert ledger.temporary_bytes_upper == 384
        assert ledger.work_units == 5
        live.set_bound(64)
        assert ledger.temporary_bytes_upper == 64
    assert ledger.temporary_bytes_upper == 0
    assert ledger.work_units == 5
    with pytest.raises(RuntimeError, match="closed"):
        live.set_bound(1)


def test_coordinate_live_storage_refusal_and_exception_preserve_actual_bound() -> None:
    ledger = CoordinateEnclosureBudget(100, 1024)
    with ledger.live_storage() as live:
        live.set_bound(128)
        with pytest.raises(CoordinateEnclosureResourceError):
            with ledger.temporary_scope():
                ledger.reserve(1, 512)
                live.set_bound(1024)
        assert live.bound == 128
        assert ledger.temporary_bytes_upper == 128
        with pytest.raises(ValueError, match="actual failure"):
            with ledger.temporary_scope():
                live.set_bound(256)
                ledger.reserve(1, 128)
                raise ValueError("actual failure")
        assert live.bound == 256
        assert ledger.temporary_bytes_upper == 256
        assert ledger.work_units == 2
    assert ledger.temporary_bytes_upper == 0


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable")
def test_coordinate_live_escape_resizes_actual_native_host_reservation() -> None:
    from phydrax._meshcore import NativeExecutionBudget

    ledger = CoordinateEnclosureBudget(100, 4096)
    native = NativeExecutionBudget(
        max_work=100,
        max_geometry_queries=100,
        max_cavity_cells=100,
        max_scratch_bytes=4096,
        max_wall_seconds=np.inf,
    )
    with native, ledger.activate(), ledger.live_storage() as live:
        live.set_bound(128)
        before = native.remaining().remaining_scratch_bytes + live.bound
        with ledger.temporary_scope():
            ledger.reserve(1, 512)
            live.set_bound(256)
        assert native.remaining().remaining_scratch_bytes == before - 256
        live.set_bound(64)
        assert native.remaining().remaining_scratch_bytes == before - 64
        live.close()
        assert native.remaining().remaining_scratch_bytes == before
    assert native.evidence is not None
    assert native.evidence.host_storage_live_bytes_upper == 0
    assert native.evidence.host_storage_peak_bytes_upper == 768


def test_coordinate_nested_live_owner_closes_without_erasing_outer_escape() -> None:
    ledger = CoordinateEnclosureBudget(100, 4096)
    with ledger.live_storage() as outer:
        outer.set_bound(128)
        with ledger.temporary_scope():
            ledger.reserve(1, 512)
            with ledger.live_storage() as inner:
                inner.set_bound(256)
                with ledger.temporary_scope():
                    outer.set_bound(384)
                    inner.set_bound(64)
                    ledger.reserve(1, 256)
                assert ledger.temporary_bytes_upper == 960
            assert ledger.temporary_bytes_upper == 896
        assert ledger.temporary_bytes_upper == 384
        assert ledger.work_units == 2
    assert ledger.temporary_bytes_upper == 0


def test_complete_coordinate_maps_reuse_exact_bank_but_not_equal_carriers() -> None:
    from phydrax.discretization._coordinate_enclosure import coordinate_polynomials

    element = coordinate_lagrange_element("tetrahedron", 1)
    bank = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    perturbed = (bank[0], (bank[1][0] + Fraction(1, 2**100), *bank[1][1:]), *bank[2:])
    np.testing.assert_array_equal(
        np.asarray(bank, dtype=np.float64), np.asarray(perturbed, dtype=np.float64)
    )
    ledger = CoordinateEnclosureBudget(100_000, 1 << 20)
    with ledger.activate():
        original = coordinate_polynomials(element, bank)
        assert original is not None
        assert coordinate_polynomials(element, bank) is original
        changed = coordinate_polynomials(element, perturbed)
        assert changed is not None and changed != original
        assert coordinate_polynomials(element, bank) is original
        assert coordinate_polynomials(element, perturbed) is changed


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable")
def test_coordinate_live_native_refusal_rolls_back_scope_without_erasing_status() -> None:
    from phydrax._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionBudget

    ledger = CoordinateEnclosureBudget(100, 8192)
    native = NativeExecutionBudget(
        max_work=100,
        max_geometry_queries=100,
        max_cavity_cells=100,
        max_scratch_bytes=4096,
        max_wall_seconds=np.inf,
    )
    with pytest.raises(MeshcoreError) as final_refusal, native:
        with ledger.activate(), ledger.live_storage() as live:
            live.set_bound(128)
            with pytest.raises(CoordinateEnclosureResourceError) as refusal:
                with ledger.temporary_scope():
                    ledger.reserve(1, 512)
                    live.set_bound(4096)
            assert refusal.value.resource == "polynomial_storage"
            assert live.bound == 128
            assert ledger.temporary_bytes_upper == 128
            assert ledger.work_units == 1
    assert final_refusal.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert native.evidence is not None
    assert native.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert native.evidence.host_storage_live_bytes_upper == 0


def test_coordinate_live_work_and_growth_admission_is_atomic_and_work_first() -> None:
    ledger = CoordinateEnclosureBudget(2, 64)
    with ledger.live_storage() as live:
        with pytest.raises(CoordinateEnclosureResourceError) as both:
            live.set_bound(128, work=3)
        assert both.value.resource == "coefficient_work"
        assert ledger.work_units == 0 and live.bound == 0
        with pytest.raises(CoordinateEnclosureResourceError) as storage:
            live.set_bound(128, work=2)
        assert storage.value.resource == "polynomial_storage"
        assert ledger.work_units == 0 and live.bound == 0
        live.set_bound(32, work=2)
        assert ledger.work_units == 2 and live.bound == 32
        with pytest.raises(CoordinateEnclosureResourceError):
            live.set_bound(64, work=1)
        assert ledger.work_units == 2 and live.bound == 32
        live.set_bound(16)
        assert ledger.work_units == 2
    assert ledger.temporary_bytes_upper == 0


@pytest.mark.parametrize("kind", ("triangle", "tetrahedron"))
def test_full_barycentric_p1_source_preserves_every_offsum_weight(kind: str) -> None:
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_expressions,
        coordinate_polynomials,
        coordinate_reference_chain,
        coordinate_source_signature,
        evaluate,
        source_basis,
        source_expressions,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology

    original = coordinate_lagrange_element(kind, 1)
    width = original.local_dof_count
    weights = np.eye(width, dtype=np.float64)
    delta = Fraction(1, 2**52)
    weights[0, 1] = float(delta)
    action = BarycentricCellGeometryElement(original, weights)
    controls = tuple(
        tuple(
            Fraction(3 + 7 * source + axis)
            for axis in range(original.topological_dimension)
        )
        for source in range(width)
    )
    ledger = CoordinateEnclosureBudget(100_000, 1 << 24)
    with ledger.activate(), ledger.temporary_scope():
        basis = source_basis(action)
        assert basis is not None and source_expressions(action) is basis
        maps = coordinate_polynomials(action, controls)
        assert maps is not None and coordinate_expressions(action, controls) is maps
        for target, vertex in enumerate(reference_cell_topology(kind).vertices):
            point = tuple(Fraction(float(value)) for value in vertex)
            assert tuple(evaluate(value, point) for value in basis) == tuple(
                Fraction(float(value)) for value in weights[target]
            )
            expected = tuple(
                sum(
                    (
                        Fraction(float(weights[target, source])) * controls[source][axis]
                        for source in range(width)
                    ),
                    Fraction(0),
                )
                for axis in range(original.topological_dimension)
            )
            assert tuple(evaluate(value, point) for value in maps) == expected
        origin = (Fraction(0),) * original.topological_dimension
        assert sum((evaluate(value, origin) for value in basis), Fraction(0)) == 1 + delta
        # A Cartesian restriction would silently replace W00 with 1-delta.
        cartesian = controls[0][0] * (1 - delta) + controls[1][0] * delta
        assert evaluate(maps[0], origin) - cartesian == controls[0][0] * delta
        changed_weights = weights.copy()
        changed_weights[0, 0] = np.nextafter(np.float64(1), np.float64(0))
        changed = BarycentricCellGeometryElement(original, changed_weights)
        assert coordinate_source_signature(changed) != coordinate_source_signature(action)
        changed_maps = coordinate_polynomials(changed, controls)
        assert changed_maps is not None and changed_maps != maps
        assert coordinate_polynomials(action, controls) is maps
    with pytest.raises(ValueError, match="Cartesian"):
        coordinate_reference_chain(action)
    with pytest.raises(ValueError, match="Cartesian"):
        coordinate_reference_chain(action, ancestor=original)


@pytest.mark.parametrize("kind", ("triangle", "tetrahedron"))
def test_nested_full_coefficient_action_retains_exact_nonbinary_composition(
    kind: str,
) -> None:
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import evaluate, source_basis

    original = coordinate_lagrange_element(kind, 1)
    width = original.local_dof_count
    third = np.float64(1.0 / 3.0)
    inner = np.eye(width, dtype=np.float64)
    inner[0, :3] = (third, 1.0, 0.0)
    inner[1, :3] = (third, 0.0, 1.0)
    outer = np.eye(width, dtype=np.float64)
    outer[0, :3] = (1.0, 0.5, 0.0)
    retained = BarycentricCellGeometryElement(original, inner)
    action = BarycentricCellGeometryElement(retained, outer)
    ledger = CoordinateEnclosureBudget(100_000, 1 << 24)
    with ledger.activate():
        basis = source_basis(action)
        assert basis is not None
        origin = (Fraction(0),) * original.topological_dimension
        exact = tuple(
            sum(
                (
                    Fraction(float(outer[0, row])) * Fraction(float(inner[row, column]))
                    for row in range(width)
                ),
                Fraction(0),
            )
            for column in range(width)
        )
        assert tuple(evaluate(term, origin) for term in basis) == exact
    # This input genuinely cannot be replaced by its binary64 matrix product.
    assert exact[0] != Fraction(float((outer @ inner)[0, 0]))
    points = np.zeros((1, original.topological_dimension), dtype=np.float64)
    values, gradients = action.tabulate(points)
    root_values, root_gradients = original.tabulate(points)
    np.testing.assert_array_equal(values, (root_values @ outer) @ inner)
    expected = np.swapaxes(
        np.swapaxes(np.asarray(root_gradients), 1, 2) @ outer @ inner, 1, 2
    )
    np.testing.assert_array_equal(gradients, expected)
    np.testing.assert_array_equal(retained.barycentric_weights, inner)
    np.testing.assert_array_equal(action.barycentric_weights, outer)


def test_full_barycentric_action_does_not_extend_to_curved_source() -> None:
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement

    curved = coordinate_lagrange_element("triangle", 2)
    with pytest.raises(ValueError, match="canonical P1"):
        BarycentricCellGeometryElement(
            curved, np.eye(curved.local_dof_count, dtype=np.float64)
        )


def _rational_plc_pyramid() -> tuple[
    CellMesh,
    CellGeometrySpec,
    RationalComposedCellGeometryElement,
    tuple[tuple[Fraction, Fraction, Fraction], ...],
]:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._coordinate_enclosure import coordinate_corner_images

    original_mesh, source = _plc_mesh(), _plc_source()
    original = CellGeometrySpec.plc(original_mesh, source)
    chart = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1, 4), Fraction(0), Fraction(0)),
        (Fraction(1, 5), Fraction(1, 4), Fraction(0)),
        (Fraction(0), Fraction(1, 4), Fraction(0)),
        (Fraction(1, 8), Fraction(1, 8), Fraction(1, 4)),
    )
    element = RationalComposedCellGeometryElement(
        coordinate_lagrange_element("tetrahedron", 1),
        coordinate_lagrange_element("pyramid", 1),
        np.asarray([[x.numerator for x in row] for row in chart]),
        np.asarray([[x.denominator for x in row] for row in chart]),
    )
    images = coordinate_corner_images(element, original.source_coordinates())
    assert images is not None
    mesh = CellMesh(
        np.asarray([[float(value) for value in row] for row in images]),
        (CellBlock("pyramid", "pyramid", np.arange(5)[None, :]),),
    )
    geometry = CellGeometrySpec(
        {"pyramid": element},
        {"pyramid": np.arange(4)[None, :]},
        original.coordinates,
        exact_source=source,
    )
    return mesh, geometry, element, chart


def test_rational_plc_pyramid_exact_map_apex_measure_embedding_and_derivatives() -> None:
    import jax
    import jax.numpy as jnp

    from phydrax.discretization._coordinate_enclosure import (
        _reference_image,
        coordinate_expressions,
        coordinate_reference_chain,
        expression_evaluate,
        RationalPolynomial,
    )
    from phydrax.geometry import certify_global_embedding

    mesh, geometry, element, chart = _rational_plc_pyramid()
    geometry.resolve(mesh)
    bank = geometry.source_coordinates()
    expressions = coordinate_expressions(element, bank)
    assert expressions is not None
    # The collapsed-cube pyramid chart of an exact PLC source is polynomial.
    polynomials = tuple(
        value for value in expressions if not isinstance(value, RationalPolynomial)
    )
    assert len(polynomials) == len(expressions)
    source_reference = (Fraction(19, 160), Fraction(1, 8), Fraction(1, 8))
    expected = tuple(
        sum(
            (source_reference[index - 1] * bank[index][axis] for index in (1, 2, 3)),
            Fraction(0),
        )
        for axis in range(3)
    )
    assert (
        tuple(expression_evaluate(value, (Fraction(1, 2),) * 3) for value in expressions)
        == expected
    )
    root, chain = coordinate_reference_chain(element)
    assert root.element_id == coordinate_lagrange_element("tetrahedron", 1).element_id
    assert (
        tuple(expression_evaluate(value, (Fraction(1, 2),) * 3) for value in chain)
        == source_reference
    )
    apex = tuple(
        sum((chart[4][index - 1] * bank[index][axis] for index in (1, 2, 3)), Fraction(0))
        for axis in range(3)
    )
    assert (
        _reference_image(
            polynomials, "pyramid", (Fraction(1, 2), Fraction(1, 2), Fraction(1))
        )
        == apex
    )
    with pytest.raises(ValueError, match="must be the apex"):
        _reference_image(polynomials, "pyramid", (Fraction(0), Fraction(0), Fraction(1)))
    points = jnp.asarray([[0.5, 0.5, 0.5], [0.5, 0.5, 1.0]])
    np.testing.assert_array_equal(
        np.asarray(element.reference_derivative_status(points)), [True, False]
    )
    values, gradients = element.tabulate(points)
    assert np.all(np.isfinite(np.asarray(values))) and np.all(
        np.isfinite(np.asarray(gradients))
    )
    np.testing.assert_allclose(
        np.asarray(values[1] @ geometry.coordinates),
        np.asarray([float(value) for value in apex]),
        rtol=0.0,
        atol=2e-16,
    )
    for u, v in ((0.1, 0.2), (0.8, 0.9)):
        height = 1 - 2.0**-24
        near = jnp.asarray(
            [[u * (1 - height) + 0.5 * height, v * (1 - height) + 0.5 * height, height]]
        )
        np.testing.assert_allclose(
            np.asarray(element.tabulate(near)[0] @ geometry.coordinates)[0],
            np.asarray([float(value) for value in apex]),
            rtol=0.0,
            atol=1e-7,
        )
    tangent = jnp.arange(12, dtype=jnp.float64).reshape((4, 3)) / 10
    mapping = lambda coefficients: element.tabulate(points)[0] @ coefficients
    _, jvp = jax.jvp(mapping, (geometry.coordinates,), (tangent,))
    np.testing.assert_allclose(
        np.asarray(jvp), np.asarray(values @ tangent), rtol=0.0, atol=2e-15
    )
    cotangent = jnp.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    _, pullback = jax.vjp(mapping, geometry.coordinates)
    np.testing.assert_allclose(
        np.asarray(pullback(cotangent)[0]),
        np.asarray(values.T @ cotangent),
        rtol=0.0,
        atol=2e-15,
    )
    reference_jacobian = jax.jacfwd(
        lambda point: element.tabulate(point[None, :])[0][0] @ geometry.coordinates
    )(points[0])
    np.testing.assert_allclose(
        np.asarray(reference_jacobian),
        np.asarray(geometry.coordinates.T @ gradients[0]),
        rtol=0.0,
        atol=2e-15,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    assert validity.all_certified
    volumes, errors, exact_measure = _certified_cell_measures(mesh, geometry)
    assert exact_measure
    expected_volume = Fraction(9, 640) * Fraction(_T)
    assert abs(Fraction(float(volumes[0])) - expected_volume) <= Fraction(
        float(errors[0])
    )
    embedding = certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"
    assert (
        geometry.source_coordinates()
        == _plc_source().prepare(geometry.coordinates).vertices
    )


def test_rational_plc_pyramid_source_refusals_preserve_original_resource_and_law() -> (
    None
):
    from phydrax.discretization._coordinate_enclosure import source_expressions

    _, _, element, chart = _rational_plc_pyramid()
    with (
        CoordinateEnclosureBudget(0, 1_000_000).activate(),
        pytest.raises(CoordinateEnclosureResourceError),
    ):
        source_expressions(element)
    with (
        CoordinateEnclosureBudget(1_000_000, 1).activate(),
        pytest.raises(CoordinateEnclosureResourceError),
    ):
        source_expressions(element)
    invalid = np.asarray([[value.numerator for value in row] for row in chart])
    denominators = np.asarray([[value.denominator for value in row] for row in chart])
    invalid[0, 0] = -1
    with pytest.raises(ValueError, match="original source simplex"):
        RationalComposedCellGeometryElement(
            element.source_element, element.chart_element, invalid, denominators
        )
    with pytest.raises(ValueError, match="five exact"):
        RationalComposedCellGeometryElement(
            element.source_element,
            element.chart_element,
            invalid,
            np.zeros((5, 3), dtype=np.int64),
        )
    changed = eqx.tree_at(
        lambda value: value.chart_coordinates, element, element.chart_coordinates + 0.01
    )
    with pytest.raises(ValueError, match="not the RNE"):
        source_expressions(changed)
    with pytest.raises(ValueError, match="collapsed rational pyramid"):
        PolynomialComposedCellGeometryElement(
            element.source_element,
            element.chart_element,
            np.zeros((5, 3), dtype=np.int64),
            np.ones((5, 3), dtype=np.int64),
        )


def test_rational_plc_full_p1_action_and_original_chart_cache_are_authoritative() -> None:
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_corner_images,
        reference_composition_arguments,
        source_expressions,
    )

    _, geometry, rational, _ = _rational_plc_pyramid()
    first = np.eye(4)
    first[1] = [0.25, 0.75, 0.0, 0.0]
    second = np.eye(4)
    second[2] = [0.125, 0.0, 0.875, 0.0]
    source = BarycentricCellGeometryElement(
        BarycentricCellGeometryElement(rational.source_element, first), second
    )
    element = RationalComposedCellGeometryElement(
        source,
        rational.chart_element,
        np.asarray([[a for a, _ in row] for row in rational.chart_coefficients]),
        np.asarray([[b for _, b in row] for row in rational.chart_coefficients]),
    )
    bank = geometry.source_coordinates()
    images = coordinate_corner_images(element, bank)
    assert images is not None
    chart_weights = tuple(Fraction(*pair) for pair in element.chart_coefficients[4])
    weights = np.asarray(
        [float(1 - sum(chart_weights)), *(float(value) for value in chart_weights)]
    )
    expected = (weights @ first @ second) @ np.asarray(geometry.coordinates)
    np.testing.assert_allclose(
        element.tabulate(np.asarray([[0.5, 0.5, 1.0]]))[0] @ geometry.coordinates,
        expected[None, :],
        rtol=0.0,
        atol=2e-16,
    )
    with CoordinateEnclosureBudget(1_000_000, 4_000_000).activate() as ledger:
        original = source_expressions(element)
        chart = reference_composition_arguments(element)
        work = ledger.work_units
        assert source_expressions(element) is original
        assert reference_composition_arguments(element) is chart
        assert ledger.work_units == work
        changed = eqx.tree_at(
            lambda value: value.source_element.barycentric_weights,
            element,
            source.barycentric_weights.at[2, 2].set(0.75),
        )
        assert source_expressions(changed) != original


def test_exact_plc_mixed_original_source_actions_have_independent_true_measures() -> None:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._coordinate_enclosure import coordinate_corner_images
    from phydrax.discretization._reference_cell import reference_cell_topology

    _, original, rational, pyramid_chart = _rational_plc_pyramid()
    bank = original.source_coordinates()
    charts = {
        "hex": tuple(
            tuple(Fraction(int(value), 8) for value in corner)
            for corner in reference_cell_topology("hexahedron").vertices
        ),
        "tet": tuple(
            (
                Fraction(1, 2) + Fraction(int(corner[0]), 8),
                Fraction(int(corner[1]), 8),
                Fraction(int(corner[2]), 8),
            )
            for corner in reference_cell_topology("tetrahedron").vertices
        ),
        "pyramid": tuple(
            (row[0], row[1] + Fraction(1, 2), row[2]) for row in pyramid_chart
        ),
    }
    elements, routes, blocks, points = {}, {}, [], []
    for name, kind in (
        ("hex", "hexahedron"),
        ("tet", "tetrahedron"),
        ("pyramid", "pyramid"),
    ):
        chart = charts[name]
        constructor = (
            RationalComposedCellGeometryElement
            if kind == "pyramid"
            else PolynomialComposedCellGeometryElement
        )
        element = constructor(
            rational.source_element,
            coordinate_lagrange_element(kind, 1),
            np.asarray([[x.numerator for x in row] for row in chart]),
            np.asarray([[x.denominator for x in row] for row in chart]),
        )
        images = coordinate_corner_images(element, bank)
        assert images is not None
        offset = len(points)
        points.extend(tuple(float(value) for value in image) for image in images)
        blocks.append(
            CellBlock(
                name,
                kind,
                np.arange(offset, offset + len(images))[None, :],
                global_ids=np.asarray([len(blocks)], dtype=np.int64),
            )
        )
        elements[name], routes[name] = element, np.arange(4)[None, :]
    mesh = CellMesh(np.asarray(points), tuple(blocks))
    geometry = CellGeometrySpec(
        elements, routes, original.coordinates, exact_source=original.exact_source
    )
    geometry.resolve(mesh)
    assert certify_cell_geometry_validity(geometry, mesh=mesh).all_certified
    volumes, errors, exact_measure = _certified_cell_measures(mesh, geometry)
    assert exact_measure
    reference_measures = (Fraction(1, 512), Fraction(1, 3072), Fraction(3, 640))
    for volume, error, reference in zip(volumes, errors, reference_measures, strict=True):
        expected = Fraction(3) * Fraction(_T) * reference
        assert abs(Fraction(float(volume)) - expected) <= Fraction(float(error))
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(geometry)
    restored = model_from_array_recipe(
        recipe,
        model_recipe_array_values(geometry, recipe, prefix="rational-mixed"),
        prefix="rational-mixed",
    )
    assert restored.source_coordinates() == bank
    assert restored.geometry_layout_id == geometry.geometry_layout_id


def test_complete_noncommuting_p1_actions_contract_original_physical_bank_exactly() -> (
    None
):
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_polynomials,
        evaluate,
    )

    root = coordinate_lagrange_element("tetrahedron", 1)
    inner = np.eye(4)
    inner[1] = [0.25, 0.75, 0.0, 0.0]
    outer = np.eye(4)
    outer[0] = [0.5, 0.0, 0.5, 0.0]
    element = BarycentricCellGeometryElement(
        BarycentricCellGeometryElement(root, inner), outer
    )
    bank = (
        (Fraction(0), Fraction(0), Fraction(0)),
        _STEINER,
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    with CoordinateEnclosureBudget(100000, 4_000_000).activate():
        expressions = coordinate_polynomials(element, bank)
        assert expressions is not None
        for point in (
            (Fraction(1, 5), Fraction(1, 7), Fraction(1, 11)),
            (Fraction(0),) * 3,
        ):
            weights = (1 - sum(point), *point)
            first = tuple(
                sum(
                    (
                        weights[row] * Fraction(float(outer[row, column]))
                        for row in range(4)
                    ),
                    Fraction(0),
                )
                for column in range(4)
            )
            complete = tuple(
                sum(
                    (
                        first[row] * Fraction(float(inner[row, column]))
                        for row in range(4)
                    ),
                    Fraction(0),
                )
                for column in range(4)
            )
            expected = tuple(
                sum(
                    (complete[index] * bank[index][axis] for index in range(4)),
                    Fraction(0),
                )
                for axis in range(3)
            )
            assert tuple(evaluate(value, point) for value in expressions) == expected
            assert not np.array_equal(outer @ inner, inner @ outer)


def test_full_p1_affine_embedding_retains_nonbinary_source_and_mapped_identity() -> None:
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import coordinate_corner_images
    from phydrax.geometry import certify_global_embedding

    original_mesh, source = _plc_mesh(), _plc_source()
    original = CellGeometrySpec.plc(original_mesh, source)
    action = np.eye(4)
    action[1] = [0.25, 0.75, 0.0, 0.0]
    element = BarycentricCellGeometryElement(
        coordinate_lagrange_element("tetrahedron", 1), action
    )
    bank = original.source_coordinates()
    corners = coordinate_corner_images(element, bank)
    assert corners is not None
    mesh = CellMesh.from_tetrahedra(
        np.asarray([[float(value) for value in point] for point in corners]),
        np.arange(4)[None, :],
    )
    name = mesh.blocks[0].name
    geometry = CellGeometrySpec(
        {name: element},
        {name: np.arange(4)[None, :]},
        original.coordinates,
        exact_source=source,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    assert validity.all_certified
    embedding = certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"
    assert embedding.binding.coordinate_scope == "mapped"
    assert "mapped_affine_source_maps" in embedding.evaluated_checks
    assert geometry.source_coordinates() == bank
    assert any(Fraction(float(value)) != value for point in corners for value in point)


def test_rational_pyramid_face_restriction_preserves_source_dimension_and_apex() -> None:
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_expressions,
        expression_reference_evaluate,
        restrict_chart_expressions,
    )

    _, geometry, element, chart = _rational_plc_pyramid()
    bank = geometry.source_coordinates()
    source = coordinate_expressions(element, bank)
    assert source is not None
    # Canonical triangular side: base corners 0/1 and apex 4.
    origin = np.zeros(3)
    matrix = np.asarray([[1.0, 0.5], [0.0, 0.5], [0.0, 1.0]])
    face = restrict_chart_expressions(source, "pyramid", "triangle", origin, matrix)
    images = tuple(
        tuple(
            sum((row[index - 1] * bank[index][axis] for index in (1, 2, 3)), Fraction(0))
            for axis in range(3)
        )
        for row in (chart[0], chart[1], chart[4])
    )
    for point in ((Fraction(1, 5), Fraction(1, 3)), (Fraction(0), Fraction(1))):
        weights = (1 - sum(point), *point)
        expected = tuple(
            sum(
                (
                    weight * image[axis]
                    for weight, image in zip(weights, images, strict=True)
                ),
                Fraction(0),
            )
            for axis in range(3)
        )
        assert (
            tuple(
                expression_reference_evaluate(value, point, "simplex") for value in face
            )
            == expected
        )
    assert all(not hasattr(value, "denominator") for value in face)


def test_full_p1_embedded_affine_surface_uses_exact_shared_source_contacts() -> None:
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import coordinate_corner_images
    from phydrax.geometry import certify_global_embedding

    element = BarycentricCellGeometryElement(
        coordinate_lagrange_element("triangle", 1),
        np.asarray([[1.0, 0.0, 0.0], [0.25, 0.75, 0.0], [0.0, 0.0, 1.0]]),
    )
    numeric = np.asarray([[0.0, 0.0, 0.0], [0.7, 0.2, 0.4], [0.0, 0.6, 0.8]])
    bank = tuple(tuple(Fraction(float(value)) for value in row) for row in numeric)
    corners = coordinate_corner_images(element, bank)
    assert corners is not None
    mesh = CellMesh.from_triangles(
        np.asarray([[float(value) for value in point] for point in corners]),
        np.arange(3)[None, :],
    )
    name = mesh.blocks[0].name
    geometry = CellGeometrySpec({name: element}, {name: np.arange(3)[None, :]}, numeric)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    assert validity.all_certified
    embedding = certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"
    assert embedding.binding.coordinate_scope == "mapped"
    assert "mapped_exact_affine_surface_contact" in embedding.evaluated_checks


def test_exact_source_binary_predicate_view_proves_values_and_preserves_nonbinary_bank() -> (
    None
):
    from phydrax.geometry._mesh_certificates import _exact_source_predicate_view

    binary = np.asarray(
        (
            (Fraction(0), Fraction(1, 4), Fraction(1, 2)),
            (Fraction(-1), Fraction(3, 4), Fraction(2)),
        ),
        dtype=object,
    )
    with CoordinateEnclosureBudget(1000, 1_000_000).activate() as ledger:
        exported, exact = _exact_source_predicate_view(binary)
        assert exact
        assert exported.dtype == np.float64
        assert all(
            Fraction(float(value)) == original
            for value, original in zip(exported.flat, binary.flat, strict=True)
        )
        assert ledger.work_units > 0
    nonbinary = binary.copy()
    nonbinary[1, 2] = Fraction(1, 3)
    with CoordinateEnclosureBudget(1000, 1_000_000).activate():
        same, exact = _exact_source_predicate_view(nonbinary)
        assert not exact
        assert same is nonbinary
    with (
        CoordinateEnclosureBudget(1, 1_000_000).activate(),
        pytest.raises(CoordinateEnclosureResourceError),
    ):
        _exact_source_predicate_view(binary)
    with (
        CoordinateEnclosureBudget(1000, 1).activate(),
        pytest.raises(CoordinateEnclosureResourceError),
    ):
        _exact_source_predicate_view(binary)


def test_complete_p1_source_reference_requires_exact_partition_of_unity() -> None:
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_partition_unity_reference_chain,
        coordinate_reference_chain,
        source_basis,
    )

    root = coordinate_lagrange_element("triangle", 1)
    inner = np.eye(3)
    inner[1] = [0.25, 0.75, 0.0]
    outer = np.eye(3)
    outer[0] = [0.5, 0.0, 0.5]
    action = BarycentricCellGeometryElement(
        BarycentricCellGeometryElement(root, inner), outer
    )
    with CoordinateEnclosureBudget(100000, 4_000_000).activate():
        original, arguments = coordinate_partition_unity_reference_chain(
            action, ancestor=root
        )
        assert original is root
        basis = source_basis(action)
        assert basis is not None
        point = (Fraction(1, 5), Fraction(1, 7))
        from phydrax.discretization._coordinate_enclosure import (
            expression_evaluate,
        )

        lambdas = tuple(expression_evaluate(value, point) for value in basis)
        assert sum(lambdas) == 1
        assert lambdas[0] == 1 - sum(lambdas[1:])
        assert (
            tuple(expression_evaluate(value, point) for value in arguments) == lambdas[1:]
        )
        with pytest.raises(ValueError, match="Cartesian"):
            coordinate_reference_chain(action, ancestor=root)
    raw = np.eye(3)
    raw[0] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    nonunit = BarycentricCellGeometryElement(root, raw)
    with (
        CoordinateEnclosureBudget(100000, 4_000_000).activate(),
        pytest.raises(ValueError, match="exact partition"),
    ):
        coordinate_partition_unity_reference_chain(nonunit, ancestor=root)
    np.testing.assert_array_equal(
        np.asarray(nonunit.barycentric_weights).view(np.uint64), raw.view(np.uint64)
    )


def test_full_nonunit_p1_packet_preserves_all_uv_columns_and_physical_remainder() -> None:
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_expressions,
        evaluate,
        expression_evaluate,
        prepare_p1_source_packet_pullback,
        source_basis,
    )

    root = coordinate_lagrange_element("triangle", 1)
    raw = np.eye(3)
    raw[0] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    element = BarycentricCellGeometryElement(root, raw)
    uv = np.asarray([[2.0, 3.0], [4.0, 3.0], [2.0, 6.0]])
    xyz = np.asarray([[10.0, 20.0, 30.0], [12.0, 21.0, 30.0], [10.0, 23.0, 32.0]])
    ids = np.asarray([19, 7, 31], dtype=np.int64)
    with CoordinateEnclosureBudget(100000, 4_000_000).activate():
        original = coordinate_expressions(root, xyz)
        assert original is not None
        actual_root, arguments, remainder = prepare_p1_source_packet_pullback(
            element,
            root,
            uv,
            xyz,
            xyz,
            original,
            root_corner_ids=ids,
            retained_corner_ids=ids,
        )
        assert actual_root is root
        basis = source_basis(element)
        assert basis is not None
        for point in ((Fraction(0), Fraction(0)), (Fraction(1, 5), Fraction(1, 7))):
            weights = tuple(evaluate(value, point) for value in basis)
            actual_uv = tuple(
                sum(
                    (
                        weight * Fraction(float(row[axis]))
                        for weight, row in zip(weights, uv, strict=True)
                    ),
                    Fraction(0),
                )
                for axis in range(2)
            )
            expected_refs = ((actual_uv[0] - 2) / 2, (actual_uv[1] - 3) / 3)
            assert tuple(evaluate(value, point) for value in arguments) == expected_refs
            physical = tuple(
                sum(
                    (
                        weight * Fraction(float(row[axis]))
                        for weight, row in zip(weights, xyz, strict=True)
                    ),
                    Fraction(0),
                )
                for axis in range(3)
            )
            root_image = tuple(
                expression_evaluate(value, expected_refs) for value in original
            )
            assert tuple(
                expression_evaluate(value, point) for value in remainder
            ) == tuple(a - b for a, b in zip(physical, root_image, strict=True))
        assert any(remainder)
        with pytest.raises(ValueError, match="corner SCI"):
            prepare_p1_source_packet_pullback(
                element,
                root,
                uv,
                xyz,
                xyz,
                original,
                root_corner_ids=ids,
                retained_corner_ids=ids[::-1],
            )
        with pytest.raises(ValueError, match="physical coefficient"):
            prepare_p1_source_packet_pullback(
                element,
                root,
                uv,
                xyz,
                xyz + 0.01,
                original,
                root_corner_ids=ids,
                retained_corner_ids=ids,
            )
    np.testing.assert_array_equal(
        np.asarray(element.barycentric_weights).view(np.uint64), raw.view(np.uint64)
    )
