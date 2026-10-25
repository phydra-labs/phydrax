#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import gc
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from typing import NoReturn

import equinox as eqx
import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax._model._structure import (
    model_from_array_recipe,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax._resource_ticket import HostBatchTicket
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
    transition_displaced_cell_geometry,
)
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._mapped_geometry_measure import certified_cell_measures
from phydrax.geometry._supermesh import prepare_common_refinement
from phydrax.lifecycle._meshing_sources import (
    register_meshing_source_artifacts,
    validate_meshing_source_closure,
)
from tests._support.exact_power_geometry import periodic_power_source, power_source


# Empirical `test_rational_plc_pyramid_exact_map_apex_measure_embedding_and_derivatives`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_PLC_PYRAMID_APEX_ATOL = 2e-16
# Empirical `test_rational_plc_pyramid_exact_map_apex_measure_embedding_and_derivatives`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_PLC_PYRAMID_CONTROL_JVP_ATOL = 2e-15
# Empirical `test_rational_plc_pyramid_exact_map_apex_measure_embedding_and_derivatives`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_PLC_PYRAMID_CONTROL_PULLBACK_ATOL = 2e-15
# Empirical `test_rational_plc_pyramid_exact_map_apex_measure_embedding_and_derivatives`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_PLC_PYRAMID_REFERENCE_JACOBIAN_ATOL = 2e-15
# Empirical `test_rational_plc_full_p1_action_and_original_chart_cache_are_authoritative`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_PLC_P1_APEX_ATOL = 2e-16
# Empirical `test_rational_plc_pyramid_exact_map_apex_measure_embedding_and_derivatives`: matched max |delta|=1.49012e-08; joint allowance fraction=0.149012, margin=85.0988%.
_PLC_NEAR_APEX_POSITION_ATOL = 1e-7


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    from phydrax.geometry._embedding_state import MeshCertificateLimits

    limits = MeshCertificateLimits()
    ticket = HostBatchTicket(
        limits.maximum_work_units,
        bytes_upper=limits.maximum_scratch_bytes,
        queries_upper=limits.maximum_work_units,
    )
    with (
        ticket,
        CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        ).activate() as ledger,
    ):
        yield ledger


# Binary64 1/3. The Steiner vertex S = (3t, t, 0) on PLC edge (0,0,0)-(3,1,0)
# has x = 1 - 2**-54, so its RNE carrier (1, t, 0) leaves the source edge.
_T = 1.0 / 3.0
_STEINER = (Fraction(_T) * 3, Fraction(_T), Fraction(0))


def _plc_source(
    *,
    parameters: np.ndarray | None = None,
    strata: np.ndarray | None = None,
    budget: CoordinateEnclosureBudget,
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
        budget=budget,
    )


def test_plain_source_factors_equal_binary64_scalars_without_repricing(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh = CellMesh.from_triangles(
        np.asarray(((0.0, -0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    geometry = CellGeometrySpec.affine(mesh)
    identity = cell_geometry_id(geometry)
    before = budget.work_units
    bank = geometry.source_coordinates(budget=budget)
    assert budget.work_units - before == geometry.coordinates.size
    assert bank == (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    assert bank[0][0] is bank[1][1] is bank[2][0]
    assert bank[1][0] is bank[2][1]
    assert bank[0][0] is not bank[0][1]
    assert np.signbit(np.asarray(geometry.coordinates)[0, 1])
    assert cell_geometry_id(geometry) == identity


def test_exact_plc_source_host_banks_preserve_identity_without_compiles(
    budget: CoordinateEnclosureBudget,
) -> None:
    from tools.compile_counter import CompileCounter

    reference = _plc_source(budget=budget)
    with CompileCounter() as compiler:
        admitted = _plc_source(budget=budget)
    assert compiler.count == 0
    assert admitted.source_id == reference.source_id
    for name in (
        "source_points",
        "source_triangles",
        "source_segments",
        "source_triangle_ids",
        "source_triangle_bounds",
        "source_segment_ids",
        "source_segment_bounds",
        "vertex_strata",
        "vertex_rows",
        "vertex_parameters",
    ):
        expected = np.asarray(getattr(reference, name))
        observed = np.asarray(getattr(admitted, name))
        assert expected.dtype == observed.dtype
        assert expected.shape == observed.shape
        assert expected.tobytes() == observed.tobytes()


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


def test_plc_identity_transition_selects_exact_source_roots_without_polynomials(
    budget: CoordinateEnclosureBudget, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phydrax.discretization import _cell_geometry_transfer as transfer
    from phydrax.discretization._cell_geometry_transition_types import (
        CellGeometryTransitionPolicy,
        NestedReferenceWitnesses,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology

    mesh, source = _plc_mesh(), _plc_source(budget=budget)
    geometry = CellGeometrySpec.plc(mesh, source)
    blocks = transfer.simplex_blocks(mesh, geometry, "source", budget=budget)
    roots = geometry.source_coordinates(budget=budget)
    assert roots[1] == _STEINER
    assert roots[1] != tuple(Fraction(float(value)) for value in mesh.coordinates[1])
    reference = np.asarray(
        (reference_cell_topology("tetrahedron").vertices,), dtype=np.float64
    )
    witnesses = NestedReferenceWitnesses(
        blocks[0].cell_ids, blocks[0].cell_ids, reference
    )

    def unexpected_polynomials(*args: object, **kwargs: object) -> None:
        pytest.fail("cardinal root selection must not prepare coordinate polynomials")

    monkeypatch.setattr(transfer, "_plc_parent_polynomials", unexpected_polynomials)
    placed, restricted, successor = transfer._plc_nested_coordinates(
        mesh,
        geometry,
        mesh,
        blocks,
        blocks,
        witnesses,
        source,
        CellGeometryTransitionPolicy(),
        budget=budget,
    )
    np.testing.assert_array_equal(
        placed.coordinates.view(np.uint64),
        np.asarray(mesh.coordinates).view(np.uint64),
    )
    assert successor.prepare(placed.coordinates, budget=budget).vertices == roots
    assert restricted.evaluations == 0
    for name in ("vertex_strata", "vertex_rows", "vertex_parameters"):
        np.testing.assert_array_equal(getattr(successor, name), getattr(source, name))
    assert successor.vertex_parameter_ratios == source.vertex_parameter_ratios


@pytest.mark.parametrize("affine_proof", (True, False), ids=("affine", "general"))
def test_plc_noncorner_transition_uses_only_proven_affine_source(
    budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
    affine_proof: bool,
) -> None:
    from phydrax.discretization import _cell_geometry_transfer as transfer
    from phydrax.discretization._cell_geometry_transition_types import (
        CellGeometryTransitionPolicy,
        NestedReferenceWitnesses,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology

    mesh, source = _plc_mesh(), _plc_source(budget=budget)
    geometry = CellGeometrySpec.plc(mesh, source)
    blocks = transfer.simplex_blocks(mesh, geometry, "source", budget=budget)
    roots = geometry.source_coordinates(budget=budget)
    target_coordinates = np.asarray(mesh.coordinates).copy()
    target_coordinates[1] /= 2
    target_ids = np.asarray(mesh.vertex_global_ids).copy()
    target_ids[1] = np.max(target_ids) + 1
    target = CellMesh.from_tetrahedra(
        target_coordinates,
        np.asarray(((0, 1, 2, 3),), dtype=np.int64),
        vertex_global_ids=target_ids,
    )
    target_layout = CellGeometrySpec.affine(target, budget=budget)
    target_blocks = transfer.simplex_blocks(
        target, target_layout, "target", budget=budget
    )
    reference = np.asarray(
        (reference_cell_topology("tetrahedron").vertices,), dtype=np.float64
    )
    reference[0, 1] /= 2
    witnesses = NestedReferenceWitnesses(
        target_blocks[0].cell_ids, blocks[0].cell_ids, reference
    )
    if affine_proof:
        def unexpected_polynomials(*args: object, **kwargs: object) -> None:
            pytest.fail("proven affine evaluation must not prepare parent polynomials")

        monkeypatch.setattr(transfer, "_plc_parent_polynomials", unexpected_polynomials)
    else:
        # Withhold the proof, requiring the unchanged real polynomial fallback.
        monkeypatch.setattr(
            transfer.imported_coordinate_enclosure,
            "affine_simplex_source",
            lambda element, *, budget: False,
        )
    placed, _, successor = transfer._plc_nested_coordinates(
        mesh,
        geometry,
        target,
        blocks,
        target_blocks,
        witnesses,
        source,
        CellGeometryTransitionPolicy(),
        budget=budget,
    )
    expected = (roots[0], tuple(value / 2 for value in roots[1]), *roots[2:])
    assert successor.prepare(placed.coordinates, budget=budget).vertices == expected
    np.testing.assert_array_equal(placed.coordinates, np.asarray(expected, dtype=np.float64))
    np.testing.assert_array_equal(successor.vertex_strata, source.vertex_strata)
    np.testing.assert_array_equal(successor.vertex_rows, source.vertex_rows)
    parameters = np.asarray(source.vertex_parameters).copy()
    parameters[1, 0] /= 2
    np.testing.assert_array_equal(successor.vertex_parameters, parameters)


def test_periodic_exact_power_source_retains_image_axis_and_original_law(
    budget: CoordinateEnclosureBudget,
) -> None:
    source = periodic_power_source()
    prepared = source.prepare(source.carrier_points, budget=budget)
    assert prepared.vertices == tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(source.carrier_points)
    )
    periodic = source.periodic_preparation
    if periodic is None:
        raise RuntimeError("fixture.periodic must retain its preparation; got None")
    assert periodic.points.shape == (1, 3)
    with pytest.raises(ValueError, match=r"ExactPowerCellGeometrySource\.vertex_sites"):
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
    with pytest.raises(
        ValueError, match=r"ExactPowerCellGeometrySource\.periodic_preparation"
    ):
        moved.prepare(budget=budget)


def test_exact_power_linear_action_uses_pretransform_exact_parent_coefficients(
    budget: CoordinateEnclosureBudget,
) -> None:
    source = periodic_power_source()
    owner = ExactPowerCellGeometryLinearActionSource(
        source,
        np.asarray([[0, -1], [0, 1]]),
        ((Fraction(1), Fraction(0)), (Fraction(1, 3), Fraction(2, 3))),
        np.asarray([[0, 0, 0], [1, 0, 0]]),
        periodic_preparation=source.periodic_preparation,
    )
    prepared = owner.prepare(budget=budget)
    assert prepared.vertices == (
        (Fraction(0),) * 3,
        (Fraction(14, 3), Fraction(0), Fraction(0)),
    )
    owner.prepare(prepared.rounded_vertices, budget=budget)
    corrupted = prepared.rounded_vertices.copy()
    corrupted[1, 0] = np.nextafter(corrupted[1, 0], np.inf)
    with pytest.raises(
        ValueError, match=r"ExactPowerCellGeometryLinearActionSource\.rounded_vertices"
    ):
        owner.prepare(corrupted, budget=budget)
    with pytest.raises(
        ValueError, match=r"ExactPowerCellGeometryLinearActionSource\.vertex_coefficients"
    ):
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
    with pytest.raises(
        ValueError, match=r"ExactPowerCellGeometryLinearActionSource\.parent_source_id"
    ):
        changed_parent.prepare(budget=budget)


def test_nonperiodic_exact_convex_source_defines_actual_q1_hexahedron(
    budget: CoordinateEnclosureBudget,
) -> None:
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
    prepared = source.prepare(budget=budget)
    mesh = CellMesh(
        prepared.rounded_vertices,
        (CellBlock("hexes", "hexahedron", np.arange(8)[None, :]),),
    )
    geometry = CellGeometrySpec.power(mesh, source, budget=budget)
    exact = tuple(
        tuple(Fraction(0.3) * int(value) / 3 for value in corner) for corner in corners
    )
    assert geometry.source_coordinates(budget=budget) == exact
    element = geometry.elements[0]
    assert element.element_id == coordinate_lagrange_element("hexahedron", 1).element_id
    assert coordinate_corner_images(element, exact, budget=budget) == exact
    assert any(
        Fraction(float(value)) != ideal
        for row, exact_row in zip(prepared.rounded_vertices, exact, strict=True)
        for value, ideal in zip(row, exact_row, strict=True)
    )
    certificate = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
    assert certificate.all_certified
    volumes, errors, ideal_measure = certified_cell_measures(
        mesh, geometry, budget=budget
    )
    assert ideal_measure
    expected = (Fraction(0.3) / 3) ** 3
    assert abs(Fraction(float(volumes[0])) - expected) <= Fraction(float(errors[0]))
    with pytest.raises(
        ValueError, match=r"ExactPowerCellGeometryLinearActionSource\.vertex_actions"
    ):
        ExactPowerCellGeometryLinearActionSource(
            parent, np.asarray([[0]]), ((Fraction(1),),), np.zeros((1, 1), dtype=np.int64)
        )


def test_exact_plc_q1_chart_retains_original_bank_and_refuses_outside_simplex(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_corner_images,
        coordinate_expressions,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology

    original_mesh, source = _plc_mesh(), _plc_source(budget=budget)
    original_geometry = CellGeometrySpec.plc(original_mesh, source, budget=budget)
    bank = original_geometry.source_coordinates(budget=budget)
    reference = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.int64)
    element = PolynomialComposedCellGeometryElement(
        coordinate_lagrange_element("tetrahedron", 1),
        coordinate_lagrange_element("hexahedron", 1),
        reference,
        np.full((8, 3), 4, dtype=np.int64),
        budget=budget,
    )
    images = coordinate_corner_images(element, bank, budget=budget)
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
        budget=budget,
    )
    geometry.resolve(mesh, budget=budget)
    assert geometry.exact_source is source
    assert geometry.source_coordinates(budget=budget) == bank
    expressions = coordinate_expressions(element, bank, budget=budget)
    assert expressions is not None
    middle = (Fraction(1, 2),) * 3
    from phydrax.discretization._coordinate_enclosure import expression_evaluate

    point = tuple(
        expression_evaluate(expression, middle, budget=budget)
        for expression in expressions
    )
    expected = tuple(
        sum((bank[index][axis] for index in (1, 2, 3)), Fraction(0)) / 8
        for axis in range(3)
    )
    assert point == expected
    assert certify_cell_geometry_validity(
        geometry, mesh=mesh, budget=budget
    ).all_certified
    volumes, errors, exact_measure = certified_cell_measures(
        mesh, geometry, budget=budget
    )
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
            budget=budget,
        )
        with pytest.raises(
            ValueError, match=r"require_plc_source_layout\.inputs.*source simplex"
        ):
            CellGeometrySpec(
                {"hexes": wrong},
                {"hexes": np.arange(4)[None, :]},
                original_geometry.coordinates,
                exact_source=source,
                budget=budget,
            )


def test_exact_plc_geometry_keeps_source_coordinates_distinct_from_carrier(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, source = _plc_mesh(), _plc_source(budget=budget)
    geometry = CellGeometrySpec.plc(mesh, source, budget=budget)
    assert geometry.exact_source is source
    assert source.prepare(mesh.coordinates, budget=budget).on_entity == (
        True,
        False,
        True,
        True,
    )
    assert np.array_equal(
        np.asarray(geometry.coordinates).view(np.uint64),
        np.asarray(mesh.coordinates).view(np.uint64),
    )
    exact = geometry.source_coordinates(budget=budget)
    assert exact[1] == _STEINER
    assert exact[1][0] != Fraction(float(np.asarray(geometry.coordinates)[1, 0]))
    assert geometry.source_execution_error(mesh, budget=budget) > 0.0
    # The source volume t/2 is representable; the carrier volume 1/6 is not.
    volumes, errors, exact_measure = certified_cell_measures(
        mesh, geometry, budget=budget
    )
    assert volumes[0] == _T / 2 and errors[0] == 0.0 and exact_measure
    finite_volume = UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare(
        cell_geometry=geometry
    )
    assert float(finite_volume.cell_volumes[0]) == _T / 2
    assert float(finite_volume.cell_volume_error_bounds[0]) == 0.0
    affine = CellGeometrySpec.affine(mesh, budget=budget)
    assert affine.exact_source is None
    assert geometry.geometry_layout_id != affine.geometry_layout_id


@contextmanager
def _request_ledger() -> Iterator[CoordinateEnclosureBudget]:
    from phydrax.geometry._embedding_state import MeshCertificateLimits

    limits = MeshCertificateLimits()
    with HostBatchTicket(
        limits.maximum_work_units, bytes_upper=limits.maximum_scratch_bytes
    ) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with ledger.activate():
            yield ledger


def test_issued_plc_witness_bank_is_reused_only_while_owned(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, source = _plc_mesh(), _plc_source(budget=budget)
    rounded = np.asarray(mesh.coordinates)
    with _request_ledger() as issuing:
        prepared = source.prepare(rounded, budget=issuing)
        issued_work = issuing.work_units
        with _request_ledger() as reusing:
            assert source.prepare(rounded, budget=reusing) is prepared
            # The reusing request admits and retains the bank, not the proof.
            assert any(value is prepared for value in reusing.plc_witness_cache.values())
            assert 0 < reusing.retained_basis_bytes <= issuing.retained_basis_bytes
            assert 0 < reusing.work_units < issued_work
        # The registry is weakly keyed by the source object, not its content.
        equal = _plc_source(budget=budget)
        with _request_ledger() as other:
            separate = equal.prepare(rounded, budget=other)
            assert separate is not prepared and separate == prepared
            assert other.work_units == issued_work
        vertices = prepared.vertices
    # Once every owner, including each request ledger, releases the bank, the
    # same source proves it again.
    del prepared, separate, issuing, reusing, other
    gc.collect()
    with _request_ledger() as fresh:
        assert source.prepare(rounded, budget=fresh).vertices == vertices
        assert fresh.work_units == issued_work



def test_exact_plc_witness_bank_refuses_before_storage_allocation(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, source = _plc_mesh(), _plc_source(budget=budget)
    geometry = CellGeometrySpec.plc(mesh, source, budget=budget)
    budget = CoordinateEnclosureBudget(
        HostBatchTicket(1_000_000, bytes_upper=1, queries_upper=1_000_000),
        maximum_work_units=1_000_000,
        maximum_memory_bytes=1,
    )
    with budget.activate(), pytest.raises(CoordinateEnclosureResourceError) as refusal:
        geometry.source_coordinates(budget=budget)
    assert refusal.value.resource == "polynomial_storage"
    assert refusal.value.limit == 1
    assert refusal.value.completed == 0
    assert refusal.value.requested > refusal.value.limit
    assert budget.retained_basis_bytes == 0


def test_exact_plc_geometry_persistence_roundtrip_renews_source_authority(
    budget: CoordinateEnclosureBudget,
) -> None:
    register_meshing_source_artifacts()
    mesh, source = _plc_mesh(), _plc_source(budget=budget)
    geometry = CellGeometrySpec.plc(mesh, source, budget=budget)
    validate_meshing_source_closure(geometry)
    recipe = model_structure_recipe(geometry)
    restored = model_from_array_recipe(
        recipe, model_recipe_array_values(geometry, recipe, prefix="plc"), prefix="plc"
    )
    validate_meshing_source_closure(restored)
    if not isinstance(restored.exact_source, ExactPlcCellGeometrySource):
        raise AssertionError(
            f"restored.exact_source must be an ExactPlcCellGeometrySource; got {type(restored.exact_source).__name__}"
        )
    assert restored.exact_source.source_id == source.source_id
    assert restored.exact_source.domain_source_id == source.domain_source_id
    assert cell_geometry_id(restored) == cell_geometry_id(geometry)
    assert restored.source_coordinates(budget=budget) == geometry.source_coordinates(
        budget=budget
    )
    # A substituted parent mapping is not the RNE ancestry of the published carrier.
    forged = eqx.tree_at(
        lambda value: value.exact_source.vertex_parameters,
        restored,
        np.asarray([[0.0, 0.0], [0.25, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float64),
    )
    with pytest.raises(ValueError, match=r"source_closure\.exact_source\.stored_fields"):
        validate_meshing_source_closure(forged)


def test_exact_source_slot_refuses_wrong_kind_family_and_foreign_authority(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, source = _plc_mesh(), _plc_source(budget=budget)
    element = coordinate_lagrange_element("tetrahedron", 1)
    with pytest.raises(TypeError, match="exact_source|require_exact_source_layout"):
        CellGeometrySpec(
            {"tetrahedra": element},
            {"tetrahedra": mesh.blocks[0].vertices},
            mesh.coordinates,
            exact_source=object(),  # ty: ignore[invalid-argument-type]
            budget=budget,
        )
    with pytest.raises(TypeError, match="source"):
        CellGeometrySpec.power(mesh, source, budget=budget)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="source"):
        CellGeometrySpec.plc(mesh, power_source(), budget=budget)  # ty: ignore[invalid-argument-type]
    power = power_source()
    with pytest.raises(
        ValueError, match=r"require_exact_source_layout\.inputs.*polyhedral layouts"
    ):
        CellGeometrySpec(
            {"tetrahedra": element},
            {"tetrahedra": mesh.blocks[0].vertices},
            power.prepare(budget=budget).rounded_vertices,
            exact_source=power,
            budget=budget,
        )
    triangles = CellMesh.from_triangles(
        np.asarray(mesh.coordinates)[:3], np.asarray([[0, 1, 2]], dtype=np.int64)
    )
    with pytest.raises(
        ValueError, match=r"CellGeometrySpec\.plc\.inputs.*direct global carrier"
    ):
        CellGeometrySpec.plc(triangles, source, budget=budget)
    with pytest.raises(
        ValueError, match=r"CellGeometrySpec\.elements.*affine tetrahedral"
    ):
        CellGeometrySpec(
            {"tetrahedra": coordinate_lagrange_element("tetrahedron", 2)},
            {"tetrahedra": np.arange(10)[None]},
            np.zeros((10, 3), dtype=np.float64),
            exact_source=source,
            budget=budget,
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
        budget=budget,
    )
    with pytest.raises(ValueError, match="ExactPlcCellGeometrySource.rounded_vertices"):
        CellGeometrySpec.plc(mesh, short, budget=budget)
    geometry = CellGeometrySpec.plc(mesh, source, budget=budget)
    moved = np.asarray(mesh.coordinates).copy()
    moved[1, 0] = np.nextafter(moved[1, 0], 0.0)
    with pytest.raises(ValueError, match="ExactPlcCellGeometrySource.rounded_vertices"):
        geometry.with_coordinates(moved, budget=budget)
    foreign = mesh.with_coordinates(
        np.asarray(mesh.coordinates) * 2.0, numeric_version="foreign"
    )
    with pytest.raises(ValueError, match=r"CellGeometrySpec\.resolve\.inputs"):
        geometry.resolve(foreign, budget=budget)
    with pytest.raises(ValueError, match="quotient"):
        geometry.with_periodic_source(mesh, budget=budget)


@pytest.mark.parametrize("drift", ("carrier", "routes"))
def test_exact_plc_carrier_proof_reuses_only_current_source_binding(
    budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    import phydrax.discretization._cell_geometry as owner

    mesh = _plc_mesh()
    geometry = CellGeometrySpec.plc(mesh, _plc_source(budget=budget), budget=budget)
    budget.plc_carrier_binding_cache.clear()
    original = owner._cardinal_corner_nodes
    evaluations = 0

    def corners(
        element: owner.CellGeometryElement,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[int, ...] | None:
        nonlocal evaluations
        evaluations += 1
        return original(element, budget=budget)

    monkeypatch.setattr(owner, "_cardinal_corner_nodes", corners)
    geometry.resolve(mesh, budget=budget)
    first = evaluations
    assert first > 0
    geometry.resolve(mesh, budget=budget)
    assert evaluations == first
    if drift == "carrier":
        values = np.asarray(mesh.coordinates).copy()
        values[1, 0] = np.nextafter(values[1, 0], np.inf)
        # Keep the authored IDs stale deliberately: the current numerical
        # carrier, not an unchanged declared ID, must invalidate the proof.
        changed_mesh = eqx.tree_at(lambda value: value.coordinates, mesh, values)
        changed_geometry = geometry
    else:
        changed_mesh = mesh
        changed_geometry = eqx.tree_at(
            lambda value: value.geometry_dofs,
            geometry,
            (geometry.geometry_dofs[0][:, (1, 0, 2, 3)],),
        )
    with pytest.raises(ValueError, match=r"CellGeometrySpec\.resolve\.inputs"):
        changed_geometry.resolve(changed_mesh, budget=budget)


def test_unbounded_integer_reference_chart_keeps_exact_measure_and_archive_identity(
    budget: CoordinateEnclosureBudget,
) -> None:
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
        basis,
        basis,
        numerators,
        denominators,
        budget=budget,
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
        {name: element},
        {name: mesh.blocks[0].vertices},
        original,
        budget=budget,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
    assert validity.all_certified
    assert (
        Fraction(float(validity.determinant_lower[0]))
        <= ratio
        <= Fraction(float(validity.determinant_upper[0]))
    )
    measures, bounds, exact = certified_cell_measures(mesh, geometry, budget=budget)
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
    restored_measures, restored_bounds, restored_exact = certified_cell_measures(
        mesh, restored, budget=budget
    )
    assert restored_exact
    assert abs(Fraction(float(restored_measures[0])) - ratio / 6) <= Fraction(
        float(restored_bounds[0])
    )


@pytest.mark.parametrize("invalid", (True, Fraction(1, 2), np.int64(1), 1.0))
def test_reference_chart_object_banks_refuse_non_python_integer_entries(
    invalid: object,
    budget: CoordinateEnclosureBudget,
) -> None:
    basis = coordinate_lagrange_element("tetrahedron", 1)
    numerator = np.zeros((4, 3), dtype=object)
    denominator = np.ones((4, 3), dtype=object)
    numerator[0, 0] = invalid
    with pytest.raises(TypeError, match="Python integers"):
        PolynomialComposedCellGeometryElement(
            basis, basis, numerator, denominator, budget=budget
        )


def test_plc_bank_refuses_tiny_active_memory_before_allocation_and_retains_lifetime(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, source = _plc_mesh(), _plc_source(budget=budget)
    geometry = CellGeometrySpec.plc(mesh, source, budget=budget)
    identity = source.source_id
    tiny = CoordinateEnclosureBudget(
        HostBatchTicket(1_000_000, bytes_upper=1, queries_upper=1_000_000),
        maximum_work_units=1_000_000,
        maximum_memory_bytes=1,
    )
    with tiny.activate(), pytest.raises(CoordinateEnclosureResourceError) as refusal:
        geometry.source_coordinates(budget=tiny)
    assert refusal.value.resource == "polynomial_storage"
    assert refusal.value.completed == 0 and refusal.value.requested > 1
    assert tiny.peak_bytes_upper == 0
    assert source.source_id == identity
    assert geometry.source_coordinates(budget=budget)[1] == _STEINER

    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=1_000_000),
        maximum_work_units=1_000_000,
        maximum_memory_bytes=1 << 24,
    )
    with ledger.activate():
        bank = geometry.source_coordinates(budget=ledger)
        assert bank[1] == _STEINER
        assert ledger.retained_basis_bytes > 0
        assert ledger.temporary_bytes_upper == 0
        remaining = ledger.maximum_memory_bytes - ledger.retained_basis_bytes
        with pytest.raises(CoordinateEnclosureResourceError):
            ledger.reserve(0, remaining + 1)
    assert ledger.peak_bytes_upper > 0


@pytest.mark.parametrize("resource", ("work", "memory"))
def test_standalone_plc_certificate_scope_shares_request_resource_ledger(
    resource: str, budget: CoordinateEnclosureBudget
) -> None:
    from phydrax.geometry._domain_coverage import (
        certify_domain_coverage,
        PiecewiseLinearDomain,
    )
    from phydrax.geometry._embedding_state import MeshCertificateLimits
    from phydrax.geometry._global_embedding import certify_global_embedding

    mesh, source = _plc_mesh(), _plc_source(budget=budget)
    geometry = CellGeometrySpec.plc(mesh, source, budget=budget)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
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
    assert geometry.source_coordinates(budget=budget)[1] == _STEINER


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable")
def test_plc_bank_and_native_buffers_share_one_original_scratch_allowance(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax._meshcore import (
        MeshcoreError,
        MeshcoreStatus,
        NativeExecutionLimits,
        NativeExecutionScope,
    )

    mesh, source = _plc_mesh(), _plc_source(budget=budget)
    geometry = CellGeometrySpec.plc(mesh, source, budget=budget)
    identity = source.source_id
    native = NativeExecutionScope(
        NativeExecutionLimits(1_000_000, 1_000_000, 1_000_000, 1 << 20, np.inf)
    )
    with pytest.raises(MeshcoreError) as final_refusal, native:
        ticket = native.batch(
            1_000_000,
            queries_upper=1_000_000,
            bytes_upper=native.remaining().remaining_managed_bytes,
        )
        ledger = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=1_000_000,
            maximum_memory_bytes=1 << 24,
        )
        with ticket, ledger.activate():
            bank = geometry.source_coordinates(budget=ledger)
            assert bank[1] == _STEINER
            available = native.remaining().remaining_managed_bytes
            assert available < (1 << 20) - ledger.retained_basis_bytes
            assert ticket.bytes_upper + available <= 1 << 20
            with pytest.raises(MeshcoreError) as refusal:
                native.allocate_host_array((available + 1,), np.uint8)
            assert refusal.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
            assert bank[1] == _STEINER
            assert source.source_id == identity
    assert final_refusal.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert native.evidence is not None
    assert native.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert native.evidence.host_storage_live_bytes_upper == 0
    assert ticket.managed_bytes >= ledger.retained_basis_bytes
    assert native.evidence.memory_counters.peak_bytes <= 1 << 20


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable")
def test_plc_bank_refuses_original_native_tiny_scratch_before_rational_allocation(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax._meshcore import (
        MeshcoreError,
        MeshcoreStatus,
        NativeExecutionLimits,
        NativeExecutionScope,
    )

    mesh, source = _plc_mesh(), _plc_source(budget=budget)
    geometry = CellGeometrySpec.plc(mesh, source, budget=budget)
    identity = source.source_id
    native = NativeExecutionScope(
        NativeExecutionLimits(1_000_000, 1_000_000, 1_000_000, 32, np.inf)
    )
    with pytest.raises(MeshcoreError) as final_refusal, native:
        ticket = native.batch(1_000_000, queries_upper=1_000_000, bytes_upper=32)
        ledger = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=1_000_000,
            maximum_memory_bytes=1 << 24,
        )
        with ticket, ledger.activate(), pytest.raises(MeshcoreError) as refusal:
            geometry.source_coordinates(budget=ledger)
        assert refusal.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
        assert refusal.value.batch_requested_upper is not None
        assert refusal.value.batch_refusal_evidence is not None
        assert refusal.value.batch_requested_upper[3] > 32
        assert refusal.value.batch_refusal_evidence[4] == 0
        assert ledger.retained_basis_bytes == 0 and ledger.peak_bytes_upper == 0
    assert final_refusal.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert native.evidence is not None
    assert native.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert ticket.managed_bytes == 0
    assert native.evidence.memory_counters.peak_bytes <= 32
    assert source.source_id == identity
    assert geometry.source_coordinates(budget=budget)[1] == _STEINER


def test_coordinate_live_storage_escapes_nested_temporary_scopes_and_shrinks() -> None:
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(100, bytes_upper=4096, queries_upper=100),
        maximum_work_units=100,
        maximum_memory_bytes=4096,
    )
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
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(100, bytes_upper=1024, queries_upper=100),
        maximum_work_units=100,
        maximum_memory_bytes=1024,
    )
    with ledger.live_storage() as live:
        live.set_bound(128)
        with pytest.raises(CoordinateEnclosureResourceError):
            with ledger.temporary_scope():
                ledger.reserve(1, 512)
                live.set_bound(1024)
        assert live.bound == 128
        assert ledger.temporary_bytes_upper == 128
        with pytest.raises(ValueError, match=r"test\.coordinate_failure"):
            with ledger.temporary_scope():
                live.set_bound(256)
                ledger.reserve(1, 128)
                raise ValueError(
                    "test.coordinate_failure must propagate; got deliberate failure"
                )
        assert live.bound == 256
        assert ledger.temporary_bytes_upper == 256
        assert ledger.work_units == 2
    assert ledger.temporary_bytes_upper == 0


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable")
def test_coordinate_live_escape_resizes_actual_native_host_reservation() -> None:
    from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope

    native = NativeExecutionScope(NativeExecutionLimits(100, 100, 100, 4096, np.inf))
    with native:
        ticket = native.batch(100, bytes_upper=4096)
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100, maximum_memory_bytes=4096
        )
        with ticket, ledger.activate(), ledger.live_storage() as live:
            live.set_bound(128)
            reserved_remaining = native.remaining().remaining_managed_bytes
            assert reserved_remaining + ticket.bytes_upper <= 4096
            with ledger.temporary_scope():
                ledger.reserve(1, 512)
                live.set_bound(256)
            assert ledger.temporary_bytes_upper == 256
            assert ticket.managed_bytes >= live.bound
            assert native.remaining().remaining_managed_bytes == reserved_remaining
            live.set_bound(64)
            assert ledger.temporary_bytes_upper == 64
            assert native.remaining().remaining_managed_bytes == reserved_remaining
            with pytest.raises(CoordinateEnclosureResourceError):
                live.set_bound(4097, work=1)
            assert live.bound == 64 and ledger.work_units == 1
            live.close()
            assert ledger.temporary_bytes_upper == 0
            assert native.remaining().remaining_managed_bytes == reserved_remaining
        assert ticket.settled
    assert native.evidence is not None
    assert native.evidence.host_storage_live_bytes_upper == 0
    assert ticket.managed_bytes == 768
    assert native.evidence.memory_counters.peak_bytes <= 4096


def test_coordinate_nested_live_owner_closes_without_erasing_outer_escape() -> None:
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(100, bytes_upper=4096, queries_upper=100),
        maximum_work_units=100,
        maximum_memory_bytes=4096,
    )
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


def test_complete_coordinate_maps_reuse_exact_bank_but_not_equal_carriers(
    budget: CoordinateEnclosureBudget,
) -> None:
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
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(100_000, bytes_upper=1 << 20, queries_upper=100_000),
        maximum_work_units=100_000,
        maximum_memory_bytes=1 << 20,
    )
    with ledger.activate():
        original = coordinate_polynomials(element, bank, budget=ledger)
        assert original is not None
        assert coordinate_polynomials(element, bank, budget=ledger) is original
        changed = coordinate_polynomials(element, perturbed, budget=ledger)
        assert changed is not None and changed != original
        assert coordinate_polynomials(element, bank, budget=ledger) is original
        assert coordinate_polynomials(element, perturbed, budget=ledger) is changed


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable")
def test_coordinate_live_native_refusal_rolls_back_scope_without_erasing_status() -> None:
    from phydrax._meshcore import (
        MeshcoreError,
        MeshcoreStatus,
        NativeExecutionLimits,
        NativeExecutionScope,
    )

    native = NativeExecutionScope(NativeExecutionLimits(100, 100, 100, 4096, np.inf))
    with pytest.raises(MeshcoreError) as final_refusal, native:
        ticket = native.batch(100, bytes_upper=4096)
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100, maximum_memory_bytes=8192
        )
        with ticket, ledger.activate(), ledger.live_storage() as live:
            live.set_bound(128)
            with pytest.raises(MeshcoreError) as refusal:
                with ledger.temporary_scope():
                    ledger.reserve(1, 512)
                    live.set_bound(4096)
            assert refusal.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
            assert refusal.value.batch_requested_upper is not None
            assert refusal.value.batch_refusal_evidence is not None
            assert refusal.value.batch_requested_upper[3] > 4096
            assert refusal.value.batch_refusal_evidence[4] == 1
            assert live.bound == 128
            assert ledger.temporary_bytes_upper == 128
            assert ledger.work_units == 1
    assert final_refusal.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert native.evidence is not None
    assert native.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert native.evidence.host_storage_live_bytes_upper == 0


def test_coordinate_live_work_and_growth_admission_is_atomic_and_work_first() -> None:
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(2, bytes_upper=64, queries_upper=2),
        maximum_work_units=2,
        maximum_memory_bytes=64,
    )
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
def test_full_barycentric_p1_source_preserves_every_offsum_weight(
    kind: str, budget: CoordinateEnclosureBudget
) -> None:
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
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(100_000, bytes_upper=1 << 24, queries_upper=100_000),
        maximum_work_units=100_000,
        maximum_memory_bytes=1 << 24,
    )
    with ledger.activate(), ledger.temporary_scope():
        basis = source_basis(action, budget=ledger)
        assert basis is not None and source_expressions(action, budget=ledger) is basis
        maps = coordinate_polynomials(action, controls, budget=ledger)
        assert (
            maps is not None
            and coordinate_expressions(action, controls, budget=ledger) is maps
        )
        for target, vertex in enumerate(reference_cell_topology(kind).vertices):
            point = tuple(Fraction(float(value)) for value in vertex)
            assert tuple(
                evaluate(value, point, budget=ledger) for value in basis
            ) == tuple(Fraction(float(value)) for value in weights[target])
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
            assert (
                tuple(evaluate(value, point, budget=ledger) for value in maps) == expected
            )
        origin = (Fraction(0),) * original.topological_dimension
        assert (
            sum((evaluate(value, origin, budget=ledger) for value in basis), Fraction(0))
            == 1 + delta
        )
        # A Cartesian restriction would silently replace W00 with 1-delta.
        cartesian = controls[0][0] * (1 - delta) + controls[1][0] * delta
        assert (
            evaluate(maps[0], origin, budget=ledger) - cartesian == controls[0][0] * delta
        )
        changed_weights = weights.copy()
        changed_weights[0, 0] = np.nextafter(np.float64(1), np.float64(0))
        changed = BarycentricCellGeometryElement(original, changed_weights)
        assert coordinate_source_signature(changed) != coordinate_source_signature(action)
        changed_maps = coordinate_polynomials(changed, controls, budget=ledger)
        assert changed_maps is not None and changed_maps != maps
        assert coordinate_polynomials(action, controls, budget=ledger) is maps
    with pytest.raises(ValueError, match="Cartesian"):
        coordinate_reference_chain(action, budget=budget)
    with pytest.raises(ValueError, match="Cartesian"):
        coordinate_reference_chain(action, ancestor=original, budget=budget)


@pytest.mark.parametrize("kind", ("triangle", "tetrahedron"))
def test_nested_full_coefficient_action_retains_exact_nonbinary_composition(
    kind: str, budget: CoordinateEnclosureBudget, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_vertex_images,
        evaluate,
        source_basis,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology

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
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(100_000, bytes_upper=1 << 24, queries_upper=100_000),
        maximum_work_units=100_000,
        maximum_memory_bytes=1 << 24,
    )
    with ledger.activate():
        basis = source_basis(action, budget=ledger)
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
        assert tuple(evaluate(term, origin, budget=ledger) for term in basis) == exact
        controls = tuple(
            tuple(Fraction(3 + 7 * node + axis) for axis in range(original.topological_dimension))
            for node in range(width)
        )

        def refuse_polynomial_corner_expansion(*args: object, **kwargs: object) -> None:
            raise AssertionError("Cardinal full-action corners must use their exact action rows.")

        monkeypatch.setattr(algebra, "_reference_basis_values", refuse_polynomial_corner_expansion)
        rows = np.arange(width, dtype=np.int64)[None, :]
        images = coordinate_vertex_images(
            ((action, rows, rows),), controls, width,
            mismatch="nonbinary full-action corner mismatch", budget=ledger,
        )
        assert images is not None
        for vertex, point in enumerate(reference_cell_topology(kind).vertices):
            corner = tuple(Fraction(value) for value in point)
            values = tuple(evaluate(term, corner, budget=ledger) for term in basis)
            assert images[vertex] == tuple(
                sum((value * controls[node][axis] for node, value in enumerate(values)), Fraction(0))
                for axis in range(original.topological_dimension)
            )
    # This input cannot be replaced by its binary64 matrix product.
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
    with pytest.raises(ValueError, match=r"require_p1_cardinal_source\.inputs"):
        BarycentricCellGeometryElement(
            curved, np.eye(curved.local_dof_count, dtype=np.float64)
        )


def _rational_plc_pyramid(
    *, budget: CoordinateEnclosureBudget
) -> tuple[
    CellMesh,
    CellGeometrySpec,
    RationalComposedCellGeometryElement,
    tuple[tuple[Fraction, Fraction, Fraction], ...],
]:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._coordinate_enclosure import coordinate_corner_images

    original_mesh, source = _plc_mesh(), _plc_source(budget=budget)
    original = CellGeometrySpec.plc(original_mesh, source, budget=budget)
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
    images = coordinate_corner_images(
        element, original.source_coordinates(budget=budget), budget=budget
    )
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
        budget=budget,
    )
    return mesh, geometry, element, chart


def test_rational_plc_pyramid_exact_map_apex_measure_embedding_and_derivatives(
    budget: CoordinateEnclosureBudget,
) -> None:
    import jax
    import jax.numpy as jnp

    from phydrax.discretization._coordinate_enclosure import (
        coordinate_expressions,
        coordinate_reference_chain,
        expression_evaluate,
        RationalPolynomial,
        reference_image,
    )
    from phydrax.geometry import certify_global_embedding

    mesh, geometry, element, chart = _rational_plc_pyramid(budget=budget)
    geometry.resolve(mesh, budget=budget)
    bank = geometry.source_coordinates(budget=budget)
    expressions = coordinate_expressions(element, bank, budget=budget)
    assert expressions is not None
    # The collapsed-cube pyramid chart of an PLC source is polynomial.
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
        tuple(
            expression_evaluate(value, (Fraction(1, 2),) * 3, budget=budget)
            for value in expressions
        )
        == expected
    )
    root, chain = coordinate_reference_chain(element, budget=budget)
    assert root.element_id == coordinate_lagrange_element("tetrahedron", 1).element_id
    assert (
        tuple(
            expression_evaluate(value, (Fraction(1, 2),) * 3, budget=budget)
            for value in chain
        )
        == source_reference
    )
    apex = tuple(
        sum((chart[4][index - 1] * bank[index][axis] for index in (1, 2, 3)), Fraction(0))
        for axis in range(3)
    )
    assert (
        reference_image(
            polynomials,
            "pyramid",
            (Fraction(1, 2), Fraction(1, 2), Fraction(1)),
            budget=budget,
        )
        == apex
    )
    with pytest.raises(ValueError, match="must be the apex"):
        reference_image(
            polynomials, "pyramid", (Fraction(0), Fraction(0), Fraction(1)), budget=budget
        )
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
        atol=_PLC_PYRAMID_APEX_ATOL,
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
            atol=_PLC_NEAR_APEX_POSITION_ATOL,
        )
    tangent = jnp.arange(12, dtype=jnp.float64).reshape((4, 3)) / 10
    mapping = lambda coefficients: element.tabulate(points)[0] @ coefficients
    _, jvp = jax.jvp(mapping, (geometry.coordinates,), (tangent,))
    np.testing.assert_allclose(
        np.asarray(jvp),
        np.asarray(values @ tangent),
        rtol=0.0,
        atol=_PLC_PYRAMID_CONTROL_JVP_ATOL,
    )
    cotangent = jnp.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    _, pullback = jax.vjp(mapping, geometry.coordinates)
    np.testing.assert_allclose(
        np.asarray(pullback(cotangent)[0]),
        np.asarray(values.T @ cotangent),
        rtol=0.0,
        atol=_PLC_PYRAMID_CONTROL_PULLBACK_ATOL,
    )
    reference_jacobian = jax.jacfwd(
        lambda point: element.tabulate(point[None, :])[0][0] @ geometry.coordinates
    )(points[0])
    np.testing.assert_allclose(
        np.asarray(reference_jacobian),
        np.asarray(geometry.coordinates.T @ gradients[0]),
        rtol=0.0,
        atol=_PLC_PYRAMID_REFERENCE_JACOBIAN_ATOL,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
    assert validity.all_certified
    volumes, errors, exact_measure = certified_cell_measures(
        mesh, geometry, budget=budget
    )
    assert exact_measure
    expected_volume = Fraction(9, 640) * Fraction(_T)
    assert abs(Fraction(float(volumes[0])) - expected_volume) <= Fraction(
        float(errors[0])
    )
    embedding = certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"
    assert (
        geometry.source_coordinates(budget=budget)
        == _plc_source(budget=budget)
        .prepare(geometry.coordinates, budget=budget)
        .vertices
    )


def test_rational_plc_pyramid_source_refusals_preserve_original_resource_and_law(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization._coordinate_enclosure import source_expressions

    _, _, element, chart = _rational_plc_pyramid(budget=budget)
    with (
        CoordinateEnclosureBudget(
            HostBatchTicket(0, bytes_upper=1_000_000, queries_upper=0),
            maximum_work_units=0,
            maximum_memory_bytes=1_000_000,
        ).activate() as stage_budget,
        pytest.raises(CoordinateEnclosureResourceError),
    ):
        source_expressions(element, budget=stage_budget)
    with (
        CoordinateEnclosureBudget(
            HostBatchTicket(1_000_000, bytes_upper=1, queries_upper=1_000_000),
            maximum_work_units=1_000_000,
            maximum_memory_bytes=1,
        ).activate() as stage_budget,
        pytest.raises(CoordinateEnclosureResourceError),
    ):
        source_expressions(element, budget=stage_budget)
    invalid = np.asarray([[value.numerator for value in row] for row in chart])
    denominators = np.asarray([[value.denominator for value in row] for row in chart])
    invalid[0, 0] = -1
    with pytest.raises(
        ValueError, match=r"RationalComposedCellGeometryElement\.inputs.*source simplex"
    ):
        RationalComposedCellGeometryElement(
            element.source_element, element.chart_element, invalid, denominators
        )
    with pytest.raises(
        ValueError, match=r"RationalComposedCellGeometryElement\.inputs.*five"
    ):
        RationalComposedCellGeometryElement(
            element.source_element,
            element.chart_element,
            invalid,
            np.zeros((5, 3), dtype=np.int64),
        )
    changed = eqx.tree_at(
        lambda value: value.chart_coordinates, element, element.chart_coordinates + 0.01
    )
    with pytest.raises(ValueError, match=r"coordinate_enclosure\.chart_coefficients"):
        source_expressions(changed, budget=budget)
    with pytest.raises(ValueError, match="collapsed rational pyramid"):
        PolynomialComposedCellGeometryElement(
            element.source_element,
            element.chart_element,
            np.zeros((5, 3), dtype=np.int64),
            np.ones((5, 3), dtype=np.int64),
            budget=budget,
        )


def test_rational_plc_full_p1_action_and_original_chart_cache_are_authoritative(
    budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _coordinate_enclosure as enclosure
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_corner_images,
        reference_composition_arguments,
        source_expressions,
    )

    _, geometry, rational, _ = _rational_plc_pyramid(budget=budget)
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
    bank = geometry.source_coordinates(budget=budget)
    images = coordinate_corner_images(element, bank, budget=budget)
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
        atol=_PLC_P1_APEX_ATOL,
    )
    with CoordinateEnclosureBudget(
        HostBatchTicket(1_000_000, bytes_upper=4_000_000, queries_upper=1_000_000),
        maximum_work_units=1_000_000,
        maximum_memory_bytes=4_000_000,
    ).activate() as ledger:
        original = source_expressions(element, budget=ledger)
        chart = reference_composition_arguments(element, budget=ledger)
        work = ledger.work_units
        def reject_repeated_preparation(*args: object, **kwargs: object) -> NoReturn:
            raise AssertionError("An authenticated cache hit must not repeat source/chart preparation")

        with monkeypatch.context() as repeated:
            repeated.setattr(enclosure, "_prepare_source_expressions", reject_repeated_preparation)
            repeated.setattr(
                enclosure, "_prepare_reference_composition_arguments", reject_repeated_preparation
            )
            assert source_expressions(element, budget=ledger) is original
            assert reference_composition_arguments(element, budget=ledger) is chart
        # Complete current-owner/JAX authentication is real paid work; the
        # source expressions and chart arithmetic above must still never rerun.
        assert ledger.work_units > work
        changed = eqx.tree_at(
            lambda value: value.source_element.barycentric_weights,
            element,
            source.barycentric_weights.at[2, 2].set(0.75),
        )
        assert source_expressions(changed, budget=ledger) != original


def test_exact_plc_mixed_original_source_actions_have_independent_true_measures(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._coordinate_enclosure import coordinate_corner_images
    from phydrax.discretization._reference_cell import reference_cell_topology

    _, original, rational, pyramid_chart = _rational_plc_pyramid(budget=budget)
    bank = original.source_coordinates(budget=budget)
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
        chart_element = coordinate_lagrange_element(kind, 1)
        numerators = np.asarray([[x.numerator for x in row] for row in chart])
        denominators = np.asarray([[x.denominator for x in row] for row in chart])
        if kind == "pyramid":
            element = RationalComposedCellGeometryElement(
                rational.source_element, chart_element, numerators, denominators
            )
        else:
            element = PolynomialComposedCellGeometryElement(
                rational.source_element,
                chart_element,
                numerators,
                denominators,
                budget=budget,
            )
        images = coordinate_corner_images(element, bank, budget=budget)
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
        elements,
        routes,
        original.coordinates,
        exact_source=original.exact_source,
        budget=budget,
    )
    geometry.resolve(mesh, budget=budget)
    assert certify_cell_geometry_validity(
        geometry, mesh=mesh, budget=budget
    ).all_certified
    volumes, errors, exact_measure = certified_cell_measures(
        mesh, geometry, budget=budget
    )
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
    assert restored.source_coordinates(budget=budget) == bank
    assert restored.geometry_layout_id == geometry.geometry_layout_id


def test_complete_noncommuting_p1_actions_contract_original_physical_bank_exactly(
    budget: CoordinateEnclosureBudget,
) -> None:
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
    with CoordinateEnclosureBudget(
        HostBatchTicket(100000, bytes_upper=4_000_000, queries_upper=100000),
        maximum_work_units=100000,
        maximum_memory_bytes=4_000_000,
    ).activate() as stage_budget:
        expressions = coordinate_polynomials(element, bank, budget=stage_budget)
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
            assert (
                tuple(
                    evaluate(value, point, budget=stage_budget) for value in expressions
                )
                == expected
            )
            assert not np.array_equal(outer @ inner, inner @ outer)


def test_full_p1_affine_embedding_retains_nonbinary_source_and_mapped_identity(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import coordinate_corner_images
    from phydrax.geometry import certify_global_embedding

    original_mesh, source = _plc_mesh(), _plc_source(budget=budget)
    original = CellGeometrySpec.plc(original_mesh, source, budget=budget)
    action = np.eye(4)
    action[1] = [0.25, 0.75, 0.0, 0.0]
    element = BarycentricCellGeometryElement(
        coordinate_lagrange_element("tetrahedron", 1), action
    )
    bank = original.source_coordinates(budget=budget)
    corners = coordinate_corner_images(element, bank, budget=budget)
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
        budget=budget,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
    assert validity.all_certified
    embedding = certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"
    assert embedding.binding.coordinate_scope == "mapped"
    assert "mapped_affine_source_maps" in embedding.evaluated_checks
    assert geometry.source_coordinates(budget=budget) == bank
    assert any(Fraction(float(value)) != value for point in corners for value in point)


def test_rational_pyramid_face_restriction_preserves_source_dimension_and_apex(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_expressions,
        expression_reference_evaluate,
        restrict_chart_expressions,
    )

    _, geometry, element, chart = _rational_plc_pyramid(budget=budget)
    bank = geometry.source_coordinates(budget=budget)
    source = coordinate_expressions(element, bank, budget=budget)
    assert source is not None
    # triangular side: base corners 0/1 and apex 4.
    origin = np.zeros(3)
    matrix = np.asarray([[1.0, 0.5], [0.0, 0.5], [0.0, 1.0]])
    face = restrict_chart_expressions(
        source, "pyramid", "triangle", origin, matrix, budget=budget
    )
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
                expression_reference_evaluate(value, point, "simplex", budget=budget)
                for value in face
            )
            == expected
        )
    assert all(not hasattr(value, "denominator") for value in face)


def test_full_p1_embedded_affine_surface_uses_exact_shared_source_contacts(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import coordinate_corner_images
    from phydrax.geometry import certify_global_embedding

    element = BarycentricCellGeometryElement(
        coordinate_lagrange_element("triangle", 1),
        np.asarray([[1.0, 0.0, 0.0], [0.25, 0.75, 0.0], [0.0, 0.0, 1.0]]),
    )
    numeric = np.asarray([[0.0, 0.0, 0.0], [0.7, 0.2, 0.4], [0.0, 0.6, 0.8]])
    bank = tuple(tuple(Fraction(float(value)) for value in row) for row in numeric)
    corners = coordinate_corner_images(element, bank, budget=budget)
    assert corners is not None
    mesh = CellMesh.from_triangles(
        np.asarray([[float(value) for value in point] for point in corners]),
        np.arange(3)[None, :],
    )
    name = mesh.blocks[0].name
    geometry = CellGeometrySpec(
        {name: element}, {name: np.arange(3)[None, :]}, numeric, budget=budget
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
    assert validity.all_certified
    embedding = certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"
    assert embedding.binding.coordinate_scope == "mapped"
    assert "mapped_exact_affine_surface_contact" in embedding.evaluated_checks


def test_exact_source_binary_predicate_view_proves_values_and_preserves_nonbinary_bank(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.geometry._global_embedding import exact_source_predicate_view

    binary = np.asarray(
        (
            (Fraction(0), Fraction(1, 4), Fraction(1, 2)),
            (Fraction(-1), Fraction(3, 4), Fraction(2)),
        ),
        dtype=object,
    )
    with CoordinateEnclosureBudget(
        HostBatchTicket(1000, bytes_upper=1_000_000, queries_upper=1000),
        maximum_work_units=1000,
        maximum_memory_bytes=1_000_000,
    ).activate() as ledger:
        exported, exact = exact_source_predicate_view(binary, budget=ledger)
        assert exact
        assert exported.dtype == np.float64
        assert all(
            Fraction(float(value)) == original
            for value, original in zip(exported.flat, binary.flat, strict=True)
        )
        assert ledger.work_units > 0
    nonbinary = binary.copy()
    nonbinary[1, 2] = Fraction(1, 3)
    with CoordinateEnclosureBudget(
        HostBatchTicket(1000, bytes_upper=1_000_000, queries_upper=1000),
        maximum_work_units=1000,
        maximum_memory_bytes=1_000_000,
    ).activate() as stage_budget:
        same, exact = exact_source_predicate_view(nonbinary, budget=stage_budget)
        assert not exact
        assert same is nonbinary
    with (
        CoordinateEnclosureBudget(
            HostBatchTicket(1, bytes_upper=1_000_000, queries_upper=1),
            maximum_work_units=1,
            maximum_memory_bytes=1_000_000,
        ).activate() as stage_budget,
        pytest.raises(CoordinateEnclosureResourceError),
    ):
        exact_source_predicate_view(binary, budget=stage_budget)
    with (
        CoordinateEnclosureBudget(
            HostBatchTicket(1000, bytes_upper=1, queries_upper=1000),
            maximum_work_units=1000,
            maximum_memory_bytes=1,
        ).activate() as stage_budget,
        pytest.raises(CoordinateEnclosureResourceError),
    ):
        exact_source_predicate_view(binary, budget=stage_budget)


def test_complete_p1_source_reference_requires_exact_partition_of_unity(
    budget: CoordinateEnclosureBudget,
) -> None:
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
    with CoordinateEnclosureBudget(
        HostBatchTicket(100000, bytes_upper=4_000_000, queries_upper=100000),
        maximum_work_units=100000,
        maximum_memory_bytes=4_000_000,
    ).activate() as stage_budget:
        original, arguments = coordinate_partition_unity_reference_chain(
            action, ancestor=root, budget=stage_budget
        )
        assert original is root
        basis = source_basis(action, budget=stage_budget)
        assert basis is not None
        point = (Fraction(1, 5), Fraction(1, 7))
        from phydrax.discretization._coordinate_enclosure import (
            expression_evaluate,
        )

        lambdas = tuple(
            expression_evaluate(value, point, budget=stage_budget) for value in basis
        )
        assert sum(lambdas) == 1
        assert lambdas[0] == 1 - sum(lambdas[1:])
        assert (
            tuple(
                expression_evaluate(value, point, budget=stage_budget)
                for value in arguments
            )
            == lambdas[1:]
        )
        with pytest.raises(ValueError, match="Cartesian"):
            coordinate_reference_chain(action, ancestor=root, budget=stage_budget)
    raw = np.eye(3)
    raw[0] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    nonunit = BarycentricCellGeometryElement(root, raw)
    with (
        CoordinateEnclosureBudget(
            HostBatchTicket(100000, bytes_upper=4_000_000, queries_upper=100000),
            maximum_work_units=100000,
            maximum_memory_bytes=4_000_000,
        ).activate() as stage_budget,
        pytest.raises(ValueError, match=r"coordinate_enclosure\.reference_action"),
    ):
        coordinate_partition_unity_reference_chain(
            nonunit, ancestor=root, budget=stage_budget
        )
    np.testing.assert_array_equal(
        np.asarray(nonunit.barycentric_weights).view(np.uint64), raw.view(np.uint64)
    )


def test_full_nonunit_p1_packet_preserves_all_uv_columns_and_physical_remainder(
    budget: CoordinateEnclosureBudget,
) -> None:
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
    with CoordinateEnclosureBudget(
        HostBatchTicket(100000, bytes_upper=4_000_000, queries_upper=100000),
        maximum_work_units=100000,
        maximum_memory_bytes=4_000_000,
    ).activate() as stage_budget:
        original = coordinate_expressions(root, xyz, budget=stage_budget)
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
            budget=stage_budget,
        )
        assert actual_root is root
        basis = source_basis(element, budget=stage_budget)
        assert basis is not None
        for point in ((Fraction(0), Fraction(0)), (Fraction(1, 5), Fraction(1, 7))):
            weights = tuple(
                evaluate(value, point, budget=stage_budget) for value in basis
            )
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
            assert (
                tuple(evaluate(value, point, budget=stage_budget) for value in arguments)
                == expected_refs
            )
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
                expression_evaluate(value, expected_refs, budget=stage_budget)
                for value in original
            )
            assert tuple(
                expression_evaluate(value, point, budget=stage_budget)
                for value in remainder
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
                budget=stage_budget,
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
                budget=stage_budget,
            )
    np.testing.assert_array_equal(
        np.asarray(element.barycentric_weights).view(np.uint64), raw.view(np.uint64)
    )


@pytest.mark.parametrize("operation", ("add", "multiply", "compose", "rational"))
def test_structural_coordinate_estimates_cover_exact_operation_work_and_bytes(
    operation: str,
) -> None:
    from phydrax.discretization._coordinate_enclosure import (
        add,
        compose,
        coordinate_estimate_add,
        coordinate_estimate_compose,
        coordinate_estimate_multiply,
        coordinate_estimate_rational,
        coordinate_expression_estimate,
        expression_evaluate,
        multiply,
        Polynomial,
        rational_expression,
    )

    first: Polynomial = {(1,): Fraction(1), (0,): Fraction(1, 3)}
    second: Polynomial = {(1,): Fraction(2), (0,): Fraction(1, 5)}
    a, b = coordinate_expression_estimate(first), coordinate_expression_estimate(second)
    point = (Fraction(1, 7),)
    first_value, second_value = point[0] + Fraction(1, 3), 2 * point[0] + Fraction(1, 5)
    match operation:
        case "add":
            estimate = coordinate_estimate_add(a, b)
            expected = first_value + second_value
        case "multiply":
            estimate = coordinate_estimate_multiply(a, b)
            expected = first_value * second_value
        case "compose":
            estimate = coordinate_estimate_compose(a, (b,))
            expected = second_value + Fraction(1, 3)
        case "rational":
            estimate = coordinate_estimate_rational(a, b)
            expected = first_value / second_value
        case _:
            raise AssertionError(
                f"fixture.operation must be a supported coordinate estimate operation; got {operation!r}"
            )
    ticket = HostBatchTicket(estimate.work_upper, bytes_upper=estimate.bytes_upper)
    ledger = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=estimate.work_upper,
        maximum_memory_bytes=estimate.bytes_upper,
    )
    with ticket, ledger.activate():
        match operation:
            case "add":
                result = add(first, second, budget=ledger)
            case "multiply":
                result = multiply(first, second, budget=ledger)
            case "compose":
                result = compose(first, (second,), budget=ledger)
            case "rational":
                result = rational_expression(first, second, budget=ledger)
            case _:
                raise AssertionError(
                    f"fixture.operation must be a supported coordinate estimate operation; got {operation!r}"
                )
        assert expression_evaluate(result, point, budget=ledger) == expected
    assert ledger.work_units <= estimate.work_upper
    assert ledger.peak_bytes_upper <= estimate.bytes_upper
    assert ticket.work_units == ledger.work_units
    assert ledger.native_charged_work_units == 0


@pytest.mark.parametrize(
    "kind", ("triangle", "tetrahedron", "quadrilateral", "hexahedron", "prism", "pyramid")
)
def test_structural_lattice_estimate_covers_exact_fallback_preparation(kind: str) -> None:
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_lattice_estimate,
        evaluate_coordinate_lattice_basis,
        prepare_coordinate_lattice_basis,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology

    estimate = coordinate_lattice_estimate(kind, 1)
    ticket = HostBatchTicket(estimate.work_upper, bytes_upper=estimate.bytes_upper)
    ledger = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=estimate.work_upper,
        maximum_memory_bytes=estimate.bytes_upper,
    )
    with ticket, ledger.activate():
        prepared = prepare_coordinate_lattice_basis(kind, 1, budget=ledger)
    points = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    values, _ = evaluate_coordinate_lattice_basis(kind, 1, points, preparation=prepared)
    np.testing.assert_array_equal(values, np.eye(len(points), dtype=np.float64))
    assert ledger.work_units <= estimate.work_upper
    assert ledger.peak_bytes_upper <= estimate.bytes_upper


@pytest.mark.parametrize("kind", ("quadrilateral", "hexahedron", "tetrahedron"))
def test_coordinate_lattice_execution_reuses_capacity_kernel(kind: str) -> None:
    from jax import clear_caches

    from phydrax.discretization._coordinate_enclosure import (
        evaluate_coordinate_lattice_basis,
        prepare_coordinate_lattice_basis,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology
    from tools.compile_counter import CompileCounter

    ticket = HostBatchTicket(100_000, bytes_upper=1 << 24)
    ledger = CoordinateEnclosureBudget(
        ticket, maximum_work_units=100_000, maximum_memory_bytes=1 << 24
    )
    with ticket, ledger.activate():
        preparation = prepare_coordinate_lattice_basis(kind, 1, budget=ledger)
    points = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    clear_caches()
    with CompileCounter() as compiler:
        first, gradient = evaluate_coordinate_lattice_basis(
            kind, 1, points, preparation=preparation
        )
        second, repeated_gradient = evaluate_coordinate_lattice_basis(
            kind, 1, points.copy(), preparation=preparation
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(gradient, repeated_gradient)
    np.testing.assert_array_equal(first, np.eye(points.shape[0], dtype=np.float64))
    assert compiler.count == 1
    assert compiler.events[0].kernel_name == "jit(evaluate_coordinate_lattice_basis)"


def test_coordinate_cache_storage_remains_reserved_between_explicit_work_phases() -> None:
    parent = HostBatchTicket(1000, bytes_upper=1 << 20)
    first = parent.batch(100, bytes_upper=1 << 20)
    ledger = CoordinateEnclosureBudget(
        first, maximum_work_units=1000, maximum_memory_bytes=1 << 20
    )
    source = (Fraction(1, 3),)
    with parent:
        with first, ledger.activate():
            ledger.retain_basis(source)
        with ledger.retained_storage(lambda size: parent.batch(0, bytes_upper=size)):
            storage = ledger.storage_ticket
            assert storage is not None and not storage.settled
            assert (
                storage.bytes_upper
                == ledger.retained_basis_bytes + ledger.temporary_bytes_upper
            )
            with pytest.raises(RuntimeError):
                parent.batch(0, bytes_upper=parent.bytes_upper - storage.bytes_upper + 1)
            second = parent.batch(
                100, bytes_upper=parent.bytes_upper - storage.bytes_upper
            )
            with ledger.bind_ticket(second):
                ledger.retain_basis((Fraction(2, 5),))
            assert second.settled
            renewed = ledger.storage_ticket
            assert renewed is not None and renewed.bytes_upper >= storage.bytes_upper
            assert storage.settled
            assert source[0] == Fraction(1, 3)
        assert ledger.storage_ticket is None
        assert parent.remaining_work_units == 1000 - ledger.work_units


def test_host_ticket_managed_contributors_preserve_other_live_storage() -> None:
    from phydrax._resource_ticket import HostBatchEstimateError

    ticket = HostBatchTicket(100, bytes_upper=1024)
    with ticket, ticket.managed_storage() as first, ticket.managed_storage() as second:
        first.set_bound(128)
        second.set_bound(256)
        assert ticket.managed_bytes == 384
        ticket.consume(managed_bytes=64)
        assert ticket.managed_bytes == 448
        first.set_bound(32)
        assert first.bound == 32 and second.bound == 256
        second.close()
        assert second.bound == 0
        with ticket.managed_storage() as excess:
            with pytest.raises(HostBatchEstimateError) as refusal:
                excess.set_bound(1024)
            assert refusal.value.batch_requested_upper[3] == 1120
            assert refusal.value.batch_refusal_evidence[4] == 0
            assert excess.bound == 0
        assert ticket.settled and ticket.work_units == 0
    assert first.bound == 0 and second.bound == 0
    assert ticket.managed_bytes == 448


def test_host_ticket_cannot_claim_a_native_charge_target() -> None:
    ticket = HostBatchTicket(0)
    with pytest.raises(TypeError, match="native ticket"):
        ticket.native_target()


def test_explicit_expression_sequence_equality_preserves_identity_and_prefix_work() -> (
    None
):
    from phydrax.discretization._coordinate_enclosure import (
        expression_sequence_equal,
        RationalPolynomial,
    )

    first = RationalPolynomial(
        {(1,): Fraction(1)}, {(0,): Fraction(1), (1,): Fraction(1)}
    )
    equal = RationalPolynomial(
        {(1,): Fraction(1)}, {(0,): Fraction(1), (1,): Fraction(1)}
    )
    ticket = HostBatchTicket(100_000, bytes_upper=1 << 20)
    ledger = CoordinateEnclosureBudget(
        ticket, maximum_work_units=100_000, maximum_memory_bytes=1 << 20
    )
    with ticket, ledger.activate():
        before = ledger.work_units
        assert expression_sequence_equal((first,), (first,), budget=ledger)
        assert ledger.work_units == before
        assert not expression_sequence_equal((first, {}), (equal,), budget=ledger)
        assert ledger.work_units == before + 16


def test_structural_coordinate_bank_bound_preserves_large_object_array_fractions() -> (
    None
):
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_estimate_bank,
        coordinate_map_expression_estimates,
        coordinate_polynomials,
        evaluate,
    )

    exact = Fraction(2**2048 + 7, 2**2063 + 11)
    coefficients = np.asarray(
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (exact, Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        ),
        dtype=object,
    )
    element = coordinate_lagrange_element("tetrahedron", 1)
    outputs = coordinate_map_expression_estimates(element, coefficients)
    assert outputs is not None
    estimate = coordinate_estimate_bank(outputs)
    ticket = HostBatchTicket(estimate.work_upper, bytes_upper=estimate.bytes_upper)
    ledger = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=estimate.work_upper,
        maximum_memory_bytes=estimate.bytes_upper,
    )
    with ticket, ledger.activate():
        polynomials = coordinate_polynomials(element, coefficients, budget=ledger)
        assert polynomials is not None
        assert (
            evaluate(
                polynomials[0], (Fraction(1), Fraction(0), Fraction(0)), budget=ledger
            )
            == exact
        )
    assert ledger.work_units <= estimate.work_upper
    assert ledger.peak_bytes_upper <= estimate.bytes_upper


@pytest.mark.parametrize("kind", ("triangle", "tetrahedron"))
def test_full_action_corner_rows_retain_exact_homogeneous_source_weights(
    kind: str, budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import coordinate_vertex_images
    from phydrax.discretization._reference_cell import reference_cell_topology

    root = coordinate_lagrange_element(kind, 1)
    width = root.local_dof_count
    weights = np.eye(width)
    weights[0, :3] = 1.0 / 3.0
    homogeneous = np.zeros((width, width))
    homogeneous[0, :3] = 1.0
    action = BarycentricCellGeometryElement(
        root, weights, barycentric_homogeneous=homogeneous
    )
    controls = tuple(
        tuple(Fraction(value) for value in point)
        for point in reference_cell_topology(kind).vertices
    )
    rows = np.arange(width, dtype=np.int64)[None, :]
    images = coordinate_vertex_images(
        ((action, rows, rows),), controls, width,
        mismatch="homogeneous full-action corner mismatch", budget=budget,
    )
    assert images is not None
    assert images[0] == tuple(
        sum((controls[node][axis] for node in range(3)), Fraction(0)) / 3
        for axis in range(root.topological_dimension)
    )
    assert images[0][0] == Fraction(1, 3)
    assert images[0][0] != Fraction(float(1.0 / 3.0))


def test_unbound_plc_entity_preparation_keeps_original_whole_row_tariff(
    budget: CoordinateEnclosureBudget,
) -> None:
    budget.capture_coordinate_operands = False
    before = budget.work_units
    source = _plc_source(budget=budget)
    assert budget.work_units - before == 2 * 64
    before = budget.work_units
    renewed = source.with_witnesses(
        source.vertex_strata, source.vertex_rows, source.vertex_parameters, budget=budget
    )
    assert budget.work_units - before == 2 * 64
    assert renewed.domain_source_id == source.domain_source_id


def test_plc_entity_numeric_frame_seal_binds_actual_source_and_nested_rows() -> None:
    import pickle

    from phydrax.discretization import (
        _coordinate_operands as operands,
        _exact_plc_geometry as owner,
    )
    from phydrax.meshing import MeshingLimits
    from tests._support.coordinate_references import native_meshing_request

    with native_meshing_request(MeshingLimits(
        maximum_work_units=1_000_000, maximum_scratch_bytes=4_000_000
    )) as (scope, budget, _), budget.bind_ticket(scope.ticket_view()):
        budget.capture_coordinate_operands = True
        source = _plc_source(budget=budget)
        frames = operands._ISSUED_PLC_ENTITIES[budget][id(source)].frames
        owner._require_plc_entity_frame_source(frames, source)
        assert frames.triangles[0].normal == (Fraction(1), Fraction(0), Fraction(0))
        assert frames.segments[0].edges == ((Fraction(3), Fraction(1), Fraction(0)),)
        budget.capture_coordinate_operands = False
        equal_valued_other_source = _plc_source(budget=budget)
        with pytest.raises(ValueError, match="actual original source"):
            owner._require_plc_entity_frame_source(frames, equal_valued_other_source)
        forged = owner._PreparedPlcEntityFrames(
            frames.triangles, frames.segments, frames.storage_bytes_upper
        )
        with pytest.raises(ValueError, match="actual producer binding"):
            owner._plc_entity_frame_basis(forged)
        with pytest.raises(TypeError, match="archive and cold"):
            pickle.dumps(frames)
        object.__setattr__(frames.triangles[0], "normal", (Fraction(7), Fraction(0), Fraction(0)))
        with pytest.raises(ValueError, match="numeric rows changed"):
            owner._plc_entity_frame_basis(frames, source=source)


def test_captured_plc_entity_constructor_has_actual_work_r_and_r_minus_one() -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.meshing import MeshingLimits
    from tests._support.coordinate_references import native_meshing_request

    def prepare(work: int) -> int:
        with native_meshing_request(MeshingLimits(
            maximum_work_units=work, maximum_scratch_bytes=4_000_000
        )) as (scope, budget, _), budget.bind_ticket(scope.ticket_view()):
            budget.capture_coordinate_operands = True
            _plc_source(budget=budget)
            return work - int(scope.remaining().remaining_work_units)

    required = prepare(1_000_000)
    assert required >= 2 * 64
    assert prepare(required) == required
    with pytest.raises(CoordinateEnclosureResourceError):
        prepare(required - 1)
