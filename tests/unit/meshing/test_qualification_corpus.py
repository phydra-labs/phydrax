#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Consumer-visible scientific-oracle boundaries for the native corpus tools."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any, cast, Literal, NoReturn, TypedDict, Unpack

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from tools._meshing_cases import corpus_solution_field_degree, meshing_case
from tools.meshing_qualification._campaign import CampaignSample
from tools.meshing_qualification._controls import QualificationControls, SCENARIO_MODULES
from tools.meshing_qualification._physics import (
    _physical_error,
    _physical_inventory,
    _prepare_corpus_solver,
)
from tools.meshing_qualification._runtime import CampaignProcessResult
from tools.meshing_qualification.scenarios.bisection import _bisection_p1_integral
from tools.meshing_qualification.scenarios.native_lifecycle import (
    _corpus_domain,
    _corpus_simplex_rows,
    _material_semantics,
)


# Fixed empirical residual bounds from test_qualification_corpus.py
# at a6827a9e5; each comparison retains its strictness.
# L2/QoI quadrature refinement differences for the independently known affine triangle
# moments.
_TRIANGLE_MOMENT_REFINEMENT_DELTA = 1e-14


# Fixed scalar-reference allowances from test_qualification_corpus.py
# at a6827a9e5; values and relative/absolute roles are unchanged.
# Manufactured quadratic FE error versus analytic simplex moments, including sqrt(1/840).
_QUADRATIC_SIMPLEX_ERROR_RTOL = 1e-10
# Quadratic error at zero scale and oriented solid volumes versus registered analytic
# region inventories.
_CORPUS_ANALYTIC_MEASURE_ATOL = 1e-12
# Independent immersed-triangle moments, all-block tetrahedron inventories and declared
# cavity volumes.
_CORPUS_MOMENT_RTOL = 1e-12


@pytest.fixture(scope="module")
def material_interface_publication() -> tuple[phx.meshing.CellMeshingResult, np.ndarray]:
    # Two triangular pyramids meet on the corpus's unit-square x=1 interface.
    points = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 0.5, 0.5),
            (2.0, 0.5, 0.5),
            (1.0, 0.5, 0.5),
        ),
        dtype=np.float64,
    )
    mesh = phx.meshing.canonicalize_cell_mesh(
        phx.discretization.CellMesh.from_tetrahedra(
            points,
            np.asarray(
                (
                    (0, 6, 1, 4),
                    (1, 6, 2, 4),
                    (2, 6, 3, 4),
                    (3, 6, 0, 4),
                    (0, 1, 6, 5),
                    (1, 2, 6, 5),
                    (2, 3, 6, 5),
                    (3, 0, 6, 5),
                ),
                dtype=np.int32,
            ),
        )
    )
    regions = np.asarray((0, 0, 0, 0, 1, 1, 1, 1), dtype=np.int64)

    def scope(dimension: int, identifiers: np.ndarray) -> phx.meshing.MeshingScope:
        return phx.meshing.MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            phx.meshing.MeshingEntityKind.MESH,
            dimension,
            mesh.entity_set(dimension).entity_set_id,
            identifiers,
        )

    zones = tuple(
        phx.meshing.MeshZone(
            name,
            phx.meshing.MeshZoneRole.REGION,
            scope(3, np.asarray(mesh.entity_set(3).entity_ids)[regions == index]),
            material_id=f"{name}-material",
            region_role=phx.meshing.RegionRole.SOLID,
        )
        for index, name in enumerate(("left", "right"))
    )
    connectivity = mesh.connectivity
    assert isinstance(connectivity, phx.discretization.TetrahedralConnectivity)
    corners = np.asarray(mesh.coordinates)[np.asarray(connectivity.faces)]
    interface = np.asarray(mesh.entity_set(2).entity_ids)[
        np.all(corners[..., 0] == 1.0, axis=1)
    ]
    patch = phx.meshing.MeshPatch(
        "left-right-interface",
        scope(2, interface),
        adjacent_zone_ids=tuple(zone.zone_id for zone in zones),
    )
    result = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        audit_policy=phx.meshing.CellMeshAuditPolicy(
            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
        ),
        zones=zones,
        patches=(patch,),
    )
    return result, regions


def test_material_corpus_solver_preserves_regional_conductivity_and_load(
    material_interface_publication: tuple[phx.meshing.CellMeshingResult, np.ndarray],
) -> None:
    result, _ = material_interface_publication
    _, problem, free = _prepare_corpus_solver(result)
    assert free.size == 1
    system, rhs = problem.linear_system()
    # Each of the eight tetrahedra has volume 1/12 and |grad(phi)|^2=5.
    # Conductivities 1 and 2 give A=5; forcing and boundary lift give b=16/3.
    np.testing.assert_allclose(
        np.asarray(system.operator.mv(jnp.ones((1,), dtype=jnp.float64))),
        np.asarray((5.0,), dtype=np.float64),
        rtol=0.0,
        atol=_CORPUS_ANALYTIC_MEASURE_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(rhs),
        np.asarray((16.0 / 3.0,), dtype=np.float64),
        rtol=0.0,
        atol=_CORPUS_ANALYTIC_MEASURE_ATOL,
    )


@pytest.mark.parametrize("degree", (1, 2), ids=("original-p1-premise", "approved-p2"))
def test_fixed_boundary_corpus_uses_actual_declared_solution_space(
    degree: int, tmp_path: Path
) -> None:
    from tools._meshing_cases import prepare_case
    from tools.meshing_qualification._physics import _solve_corpus_problem

    source, specification, options = prepare_case(
        meshing_case("plc-fixed-boundary"), 4, 20000, 120.0
    )
    result = (
        phx.meshing.NativeMeshingProvider(options)
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    space, problem, free = _prepare_corpus_solver(
        result, material_x=False, solution_degree=degree
    )
    dofs = space.dof_maps[0]
    if degree == 1:
        assert dofs.global_dof_count == result.mesh.coordinates.shape[0]
        assert np.count_nonzero(np.asarray(dofs.boundary_dof_mask)) == 8
        assert free.size == dofs.global_dof_count - 8
        assert free.size > 0
    else:
        assert corpus_solution_field_degree(meshing_case("plc-fixed-boundary")) == degree
        assert dofs.global_dof_count == (
            result.mesh.coordinates.shape[0] + result.mesh.entity_set(1).count
        )
        assert np.count_nonzero(np.asarray(dofs.boundary_dof_mask)) > 8
        assert free.size > 0
    system, rhs = problem.linear_system()
    assert system.operator.source.size == system.operator.target.size == free.size
    assert rhs.shape == free.shape
    values, solved = _solve_corpus_problem(
        problem, jnp.zeros(free.shape, dtype=jnp.float64)
    )
    assert bool(np.asarray(solved.successful))
    # The native matrix-free route retains its uncertified-rank sentinel.
    assert int(np.asarray(solved.diagnostics.rank)) == -1
    assert solved.provenance.backend == "native-krylov"
    if degree == 1:
        # Additional interior stars do not repair the immutable affine P1 trace.
        # The cube Galerkin lower bound from the approved incompatibility proof
        # remains authoritative regardless of the lawful interior refinement.
        error = _physical_error(result.mesh, values, space=space)
        assert error >= 0.5 - 2.0 / np.pi**2 - _CORPUS_ANALYTIC_MEASURE_ATOL
        assert error > 0.05
    else:
        np.testing.assert_allclose(
            values,
            jnp.sum(dofs.dof_coordinates**2, axis=-1),
            rtol=0.0,
            atol=_CORPUS_ANALYTIC_MEASURE_ATOL,
        )
        assert _physical_inventory(space, values) == pytest.approx(
            1.0, rel=0.0, abs=_CORPUS_ANALYTIC_MEASURE_ATOL
        )
        assert (
            _physical_error(result.mesh, values, space=space)
            < _CORPUS_ANALYTIC_MEASURE_ATOL
        )
        from phydrax.lifecycle import (
            read_meshing_source_closure,
            write_meshing_source_closure,
        )

        density = jnp.full((result.mesh.coordinates.shape[0],), 2.0)
        history = dofs.dof_coordinates[:, 0] + 0.5
        receipt = write_meshing_source_closure(
            tmp_path / "fixed-p2.zip", (result, (values, density, history))
        )
        restored, fields = read_meshing_source_closure(
            receipt.path, expected_content_id=receipt.content_id
        )
        for actual, expected in zip(fields, (values, density, history), strict=True):
            np.testing.assert_array_equal(actual, expected)
        reopened_space, reopened_problem, reopened_free = _prepare_corpus_solver(
            restored, material_x=False, solution_degree=degree
        )
        assert reopened_space.dof_maps[0].dof_map_id == dofs.dof_map_id
        continued, continuation = _solve_corpus_problem(
            reopened_problem, fields[0][reopened_free]
        )
        assert bool(np.asarray(continuation.successful))
        np.testing.assert_allclose(
            continued, values, rtol=0.0, atol=_CORPUS_ANALYTIC_MEASURE_ATOL
        )
        assert (
            _physical_error(restored.mesh, continued, space=reopened_space)
            < _CORPUS_ANALYTIC_MEASURE_ATOL
        )


def test_material_form_still_refuses_duplicate_action_identities() -> None:
    with pytest.raises(ValueError, match="action IDs must be unique"):
        phx.equations.FiniteElementForm(
            "duplicate-material-action-identity",
            "u",
            (
                phx.equations.DiffusionAction("u", 1.0),
                phx.equations.DiffusionAction("u", 2.0),
            ),
        )


def test_material_interface_oracle_selects_polygon_vertex_rows(
    material_interface_publication: tuple[phx.meshing.CellMeshingResult, np.ndarray],
) -> None:
    result, regions = material_interface_publication
    evidence = _material_semantics(meshing_case("plc-material-cavity"), result, regions)
    assert evidence["passed"]
    assert evidence["interface_scope_passed"]
    assert evidence["interface_area"] == 1.0


def test_material_interface_oracle_refuses_changed_region_membership(
    material_interface_publication: tuple[phx.meshing.CellMeshingResult, np.ndarray],
) -> None:
    result, regions = material_interface_publication
    changed = regions.copy()
    changed[0] = 1
    evidence = _material_semantics(meshing_case("plc-material-cavity"), result, changed)
    assert not evidence["passed"]
    assert evidence["interface_scope_passed"]


def test_material_interface_oracle_refuses_incomplete_interface_scope(
    material_interface_publication: tuple[phx.meshing.CellMeshingResult, np.ndarray],
) -> None:
    result, regions = material_interface_publication
    before = result.patches[0]
    scope = before.scope
    partial = phx.meshing.MeshPatch(
        before.name,
        phx.meshing.MeshingScope(
            scope.source_id,
            scope.source_revision,
            scope.entity_kind,
            scope.entity_dimension,
            scope.entity_set_id,
            scope.entity_ids[:1],
        ),
        adjacent_zone_ids=before.adjacent_zone_ids,
    )
    changed = phx.meshing.certify_cell_mesh(
        result.mesh,
        result.coordinate_contract,
        geometry=result.geometry,
        patches=(partial,),
        zones=result.zones,
    )
    evidence = _material_semantics(meshing_case("plc-material-cavity"), changed, regions)
    assert not evidence["passed"]
    assert not evidence["interface_scope_passed"]


@pytest.mark.parametrize(
    ("dimension", "scale", "squared_reference_error"),
    (
        (2, 1.0, 11.0 / 180.0),
        (3, 1.0, 1.0 / 28.0),
        (2, 0.25, 11.0 / 180.0),
        (3, 2.0, 1.0 / 28.0),
    ),
    ids=("triangle", "tetrahedron", "scaled-triangle", "scaled-tetrahedron"),
)
def test_quadratic_manufactured_error_has_exact_simplex_integral(
    dimension: int,
    scale: float,
    squared_reference_error: float,
) -> None:
    points = scale * np.concatenate(
        (np.zeros((1, dimension), dtype=np.float64), np.eye(dimension, dtype=np.float64))
    )
    points += np.arange(dimension, dtype=np.float64) + 3.0
    cells = np.arange(dimension + 1, dtype=np.int32)[None, :]
    mesh = (
        phx.discretization.CellMesh.from_triangles(points, cells)
        if dimension == 2
        else phx.discretization.CellMesh.from_tetrahedra(points, cells)
    )
    nodal_values = np.sum(points * points, axis=1)
    # On the unit simplex, u_h=sum(x_i) and u=sum(x_i^2). The analytic
    # monomial integral is prod(a_i!)/(sum(a_i)+d)!. Translation adds only
    # reproduced linear terms; scaling contributes scale^(d+4) to ||e||².
    expected = np.sqrt(squared_reference_error * scale ** (dimension + 4))
    assert _physical_error(mesh, nodal_values) == pytest.approx(
        expected, rel=_QUADRATIC_SIMPLEX_ERROR_RTOL, abs=_CORPUS_ANALYTIC_MEASURE_ATOL
    )


@pytest.mark.parametrize("dimension", (2, 3), ids=("triangle", "tetrahedron"))
def test_quadratic_oracle_integrates_edge_dofs_with_independent_exact_moments(
    dimension: int,
) -> None:
    points = np.concatenate((np.zeros((1, dimension)), np.eye(dimension)))
    # Physical vertex IDs deliberately do not follow the reference vertex order.
    cells = np.asarray([(1, 2, 0, *range(3, dimension + 1))], dtype=np.int32)
    family = "triangle" if dimension == 2 else "tetrahedron"
    mesh = (
        phx.discretization.CellMesh.from_triangles(points, cells)
        if dimension == 2
        else phx.discretization.CellMesh.from_tetrahedra(points, cells)
    )
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element(family, 2)
        ),
    ).prepare()
    dofs = space.dof_maps[0]
    values = np.sum(np.asarray(dofs.dof_coordinates) ** 2, axis=-1)
    assert _physical_error(mesh, values, space=space) < _CORPUS_ANALYTIC_MEASURE_ATOL
    exact_inventory = 2.0 * dimension / (24.0 if dimension == 2 else 120.0)
    assert _physical_inventory(space, values) == pytest.approx(
        exact_inventory, rel=0.0, abs=_CORPUS_ANALYTIC_MEASURE_ATOL
    )
    element = space.elements[0][0]
    edge_dof = int(np.asarray(dofs.cell_dofs[0])[0, element.entity_dofs[1][0][0]])
    perturbed = values.copy()
    perturbed[edge_dof] += 1.0
    assert _physical_inventory(space, perturbed) == pytest.approx(
        exact_inventory + 4.0 / (24.0 if dimension == 2 else 120.0),
        rel=0.0,
        abs=_CORPUS_ANALYTIC_MEASURE_ATOL,
    )
    # An edge basis is 4 lambda_i lambda_j; its squared integral is
    # 16 * (2! * 2!) / (d+4)! on this unit reference simplex.
    expected = np.sqrt(64.0 / (720.0 if dimension == 2 else 5040.0))
    assert _physical_error(mesh, perturbed, space=space) == pytest.approx(
        expected, rel=_QUADRATIC_SIMPLEX_ERROR_RTOL, abs=_CORPUS_ANALYTIC_MEASURE_ATOL
    )
    with pytest.raises(ValueError, match="actual FE space"):
        _physical_error(mesh, perturbed)


def test_only_fixed_boundary_corpus_changes_solution_and_history_degree() -> None:
    from phydrax.meshing.qualification import envelope_transport_field_templates
    from tools._meshing_cases import CORPUS

    for case in CORPUS:
        degree = corpus_solution_field_degree(case)
        assert degree == (2 if case.name == "plc-fixed-boundary" else 1)
        metadata = case.to_record()
        assert metadata["solution_field_degree"] == degree
        assert metadata["history_field_degree"] == degree
        assert metadata["density_field_degree"] == 1
        templates = envelope_transport_field_templates(
            "tetrahedron", solution_degree=degree
        )
        assert tuple(field.elements[0].degree for _, field in templates) == (
            degree,
            1,
            degree,
        )


def test_immersed_triangle_p1_inventory_uses_surface_measure() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    values = np.asarray((1.0, 2.0, 4.0), dtype=np.float64)
    assert _bisection_p1_integral(mesh, values) == pytest.approx(
        7.0 * np.sqrt(3.0) / 6.0, rel=_CORPUS_MOMENT_RTOL
    )


@pytest.mark.parametrize("reverse_blocks", (False, True))
def test_multiblock_scalar_inventory_integrates_every_routed_tetrahedron(
    reverse_blocks: bool,
) -> None:
    simplex = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    points = np.concatenate(
        (simplex, 2.0 * simplex + np.asarray((3.0, 0.0, 0.0), dtype=np.float64))
    )
    blocks = (
        phx.discretization.CellBlock(
            "first",
            "tetrahedron",
            np.asarray(((0, 1, 2, 3),), dtype=np.int64),
            global_ids=np.asarray((101,), dtype=np.int64),
        ),
        phx.discretization.CellBlock(
            "second",
            "tetrahedron",
            np.asarray(((4, 5, 6, 7),), dtype=np.int64),
            global_ids=np.asarray((205,), dtype=np.int64),
        ),
    )
    mesh = phx.discretization.CellMesh(
        points,
        blocks[::-1] if reverse_blocks else blocks,
        vertex_global_ids=np.asarray((90, 12, 41, 8, 73, 62, 24, 35), dtype=np.int64),
    )
    # Integral of x+y+z: (1/6)*(3/4) + (8/6)*(18/4) = 49/8.
    values = np.sum(points, axis=1)
    assert _bisection_p1_integral(mesh, values) == pytest.approx(
        49.0 / 8.0, rel=_CORPUS_MOMENT_RTOL
    )


@pytest.mark.parametrize("dimension", (2, 3), ids=("triangle", "tetrahedron"))
@pytest.mark.parametrize("reverse_blocks", (False, True))
def test_corpus_simplex_oracle_reads_all_packed_blocks_in_topology_order(
    dimension: int, reverse_blocks: bool
) -> None:
    points = np.concatenate(
        (np.zeros((1, dimension)), np.eye(dimension), -np.eye(dimension)[-1:])
    )
    first = tuple(range(dimension + 1))
    second = (1, 0, *range(2, dimension), dimension + 1)
    kind = "triangle" if dimension == 2 else "tetrahedron"
    blocks = (
        phx.discretization.CellBlock(
            "first",
            kind,
            np.asarray((first,), dtype=np.int64),
            global_ids=np.asarray((101,), dtype=np.int64),
        ),
        phx.discretization.CellBlock(
            "second",
            kind,
            np.asarray((second,), dtype=np.int64),
            global_ids=np.asarray((205,), dtype=np.int64),
        ),
    )
    mesh = phx.discretization.CellMesh(
        points,
        blocks[::-1] if reverse_blocks else blocks,
        vertex_global_ids=np.arange(len(points), dtype=np.int64)[::-1] + 70,
    )
    if dimension == 3:
        assert isinstance(mesh.connectivity, phx.discretization.PolyhedralConnectivity)
    rows = _corpus_simplex_rows(mesh)
    assert len(rows) == dimension + 1
    for degree, entities in enumerate(rows):
        expected = {
            tuple(sorted(entity))
            for cell in (first, second)
            for entity in combinations(cell, degree + 1)
        }
        assert {tuple(entity) for entity in entities.tolist()} == expected
    np.testing.assert_array_equal(
        rows[-1],
        np.sort(
            np.concatenate([np.asarray(block.vertices) for block in mesh.blocks]), axis=1
        ),
    )


@pytest.mark.parametrize("kind", ("quadrilateral", "pyramid", "prism", "hexahedron"))
def test_corpus_simplex_oracle_refuses_other_polygonal_or_polyhedral_cells(
    kind: Literal["quadrilateral", "pyramid", "prism", "hexahedron"],
) -> None:
    points = np.asarray(phx.discretization.reference_cell_topology(kind).vertices)
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "not-simplex",
                kind,
                np.arange(len(points), dtype=np.int64)[None],
            ),
        ),
    )
    with pytest.raises(TypeError, match="must all declare one simplex kind"):
        _corpus_simplex_rows(mesh)


def test_material_quadratic_error_respects_conductivity_and_continuous_trace() -> None:
    points = np.asarray(
        ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        points, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    values = 0.5 * points[:, 0] ** 2 + 0.5
    # On this right-material tetrahedron the interpolation error is
    # (xi-xi^2)/2. Integrating xi²-2xi³+xi⁴ gives 1/210, hence 1/840.
    assert _physical_error(mesh, values, material_x=True) == pytest.approx(
        np.sqrt(1.0 / 840.0),
        rel=_QUADRATIC_SIMPLEX_ERROR_RTOL,
        abs=_CORPUS_ANALYTIC_MEASURE_ATOL,
    )


def test_independent_material_cavity_domain_has_correct_oriented_volumes() -> None:
    domain = _corpus_domain(meshing_case("plc-material-cavity"))
    corners = np.asarray(domain.vertices)[np.asarray(domain.facets)]
    contributions = (
        np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2]), axis=1) / 6.0
    )
    adjacency = np.asarray(domain.facet_regions)
    volumes = {
        name: float(
            np.sum(contributions[adjacency[:, 0] == index])
            - np.sum(contributions[adjacency[:, 1] == index])
        )
        for index, name in enumerate(domain.region_ids)
    }
    assert volumes == pytest.approx(
        {"left": 1.0, "right": 1.0 - 0.5**3}, rel=_CORPUS_MOMENT_RTOL
    )


def test_design_surface_qoi_and_error_have_independent_triangle_moments() -> None:
    from tools.meshing_qualification.scenarios.design import (
        _design_qualification_physical,
    )

    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    values = space.dof_maps[0].dof_coordinates[:, 0]
    physical = _design_qualification_physical(result, space, values, 4)
    # The surface Jacobian is sqrt(3); ∫_T x^k = 1/((k+1)(k+2)).
    assert physical["field_L2_error"] == pytest.approx(
        np.sqrt(np.sqrt(3.0) / 60.0), rel=_CORPUS_MOMENT_RTOL
    )
    assert physical["exact_integral"] == pytest.approx(
        np.sqrt(3.0) / 12.0, rel=_CORPUS_MOMENT_RTOL
    )
    assert physical["QoI_absolute_error"] == pytest.approx(
        np.sqrt(3.0) / 12.0, rel=_CORPUS_MOMENT_RTOL
    )
    assert physical["L2_quadrature_refinement_delta"] < _TRIANGLE_MOMENT_REFINEMENT_DELTA
    assert physical["QoI_quadrature_refinement_delta"] < _TRIANGLE_MOMENT_REFINEMENT_DELTA


@pytest.mark.strict_jax
@pytest.mark.parametrize("action", ("barycentric", "restricted", "polynomial"))
def test_design_surface_oracle_uses_affine_coordinate_source_actions(
    action: Literal["barycentric", "restricted", "polynomial"],
) -> None:
    from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
    from tests._support.coordinate_references import coordinate_reference_budget
    from tools.meshing_qualification.scenarios.design import (
        _design_qualification_physical,
    )

    source_coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)), dtype=np.float64
    )
    mesh = phx.discretization.CellMesh(
        np.asarray(
            ((0.0, 0.0, 0.0), (0.5, 0.0, 0.5), (0.0, 1.0, 1.0)),
            dtype=np.float64,
        ),
        (
            phx.discretization.CellBlock(
                "half", "triangle", np.asarray(((0, 1, 2),), dtype=np.int32)
            ),
        ),
    )
    root = phx.discretization.coordinate_lagrange_element("triangle", 1)
    with coordinate_reference_budget() as budget:
        match action:
            case "barycentric":
                element = BarycentricCellGeometryElement(
                    root,
                    np.asarray(
                        ((1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.0, 0.0, 1.0)),
                        dtype=np.float64,
                    ),
                )
            case "restricted":
                element = phx.discretization.RestrictedCellGeometryElement(
                    root,
                    "triangle",
                    np.asarray(((0.5, 0.0), (0.0, 1.0)), dtype=np.float64),
                    np.zeros((2,), dtype=np.float64),
                )
            case "polynomial":
                element = phx.discretization.PolynomialComposedCellGeometryElement(
                    root,
                    root,
                    np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.int64),
                    np.asarray(((1, 1), (2, 1), (1, 1)), dtype=np.int64),
                    budget=budget,
                )
        geometry = phx.discretization.CellGeometrySpec(
            {"half": element},
            {"half": np.asarray(((0, 1, 2),), dtype=np.int32)},
            source_coordinates,
            budget=budget,
        )
    result = phx.meshing.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    values = space.dof_maps[0].dof_coordinates[:, 0]
    physical = _design_qualification_physical(result, space, values, 4)
    # The source maps xi -> x=xi/2, with constant surface Jacobian
    # sqrt(3)/2. Thus integral(x^k)=sqrt(3)*2^(-k-1)/((k+1)(k+2)).
    assert physical["field_L2_error"] == pytest.approx(
        np.sqrt(np.sqrt(3.0) / 192.0), rel=_CORPUS_MOMENT_RTOL
    )
    assert physical["integral"] == pytest.approx(
        np.sqrt(3.0) / 24.0, rel=_CORPUS_MOMENT_RTOL
    )
    assert physical["exact_integral"] == pytest.approx(
        np.sqrt(3.0) / 96.0, rel=_CORPUS_MOMENT_RTOL
    )
    assert physical["QoI_absolute_error"] == pytest.approx(
        np.sqrt(3.0) / 32.0, rel=_CORPUS_MOMENT_RTOL
    )
    assert physical["L2_quadrature_refinement_delta"] < _TRIANGLE_MOMENT_REFINEMENT_DELTA
    assert physical["QoI_quadrature_refinement_delta"] < _TRIANGLE_MOMENT_REFINEMENT_DELTA


@pytest.mark.strict_jax
def test_design_surface_oracle_refuses_nonaffine_coordinate_source() -> None:
    from tests._support.coordinate_references import coordinate_reference_budget
    from tools.meshing_qualification.scenarios.design import (
        _design_qualification_physical,
    )

    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    element = phx.discretization.coordinate_lagrange_element("triangle", 2)
    reference = np.asarray(element.reference_nodes)
    coordinates = np.column_stack(
        (
            reference,
            0.125 * reference[:, 0] * (1.0 - reference[:, 0] - reference[:, 1]),
        )
    )
    with coordinate_reference_budget() as budget:
        geometry = phx.discretization.CellGeometrySpec(
            {"triangles": element},
            {"triangles": np.arange(element.local_dof_count, dtype=np.int32)[None]},
            coordinates,
            budget=budget,
        )
    result = phx.meshing.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    values = space.dof_maps[0].dof_coordinates[:, 0]
    with pytest.raises(ValueError, match="P1 cardinal"):
        _design_qualification_physical(result, space, values, 4)


@pytest.mark.parametrize(
    "name",
    ("plc-nonconvex", "plc-disconnected", "plc-thin-channel", "plc-triple-junction"),
)
def test_declared_solid_oracle_and_oriented_boundary_agree(name: str) -> None:
    case = meshing_case(name)
    domain = _corpus_domain(case)
    corners = np.asarray(domain.vertices)[np.asarray(domain.facets)]
    contributions = (
        np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])) / 6.0
    )
    pairs = np.asarray(domain.facet_regions)
    measured = {
        region: float(
            contributions[pairs[:, 0] == index].sum()
            - contributions[pairs[:, 1] == index].sum()
        )
        for index, (region, _) in enumerate(case.expected_regions)
    }
    assert measured == pytest.approx(
        dict(case.expected_regions), abs=_CORPUS_ANALYTIC_MEASURE_ATOL
    )


def test_declared_material_oracle_does_not_use_x_threshold() -> None:
    from tools._meshing_cases import analytic_region_indices

    points = np.asarray(
        (
            (0.5, 0.5, 0.5),
            (1.5, 0.5, 0.5),
            (0.5, 1.5, 0.5),
            (1.5, 1.5, 0.5),
            (3.0, 3.0, 3.0),
        )
    )
    assert np.array_equal(
        analytic_region_indices(meshing_case("plc-triple-junction"), points),
        np.asarray((0, 1, 2, 2, -1)),
    )


def test_disconnected_oracle_rejects_gap_and_cavity_oracle_rejects_void() -> None:
    from tools._meshing_cases import analytic_region_indices

    assert (
        analytic_region_indices(
            meshing_case("plc-disconnected"), np.asarray(((1.5, 0.5, 0.5),))
        )[0]
        == -1
    )
    assert (
        analytic_region_indices(
            meshing_case("plc-material-cavity"), np.asarray(((1.5, 0.5, 0.5),))
        )[0]
        == -1
    )


@pytest.mark.parametrize(
    ("kind", "field"),
    (
        ("work", "maximum_work_units"),
        ("queries", "maximum_geometry_queries"),
        ("cavity", "maximum_cavity_cells"),
        ("entities", "maximum_vertices"),
        ("storage", "maximum_data_bytes"),
    ),
)
def test_refusal_corpus_changes_only_its_declared_allowance(
    kind: str, field: str
) -> None:
    from tools._meshing_cases import prepare_case, source_payload

    positive = meshing_case("plc-material-cavity")
    refusal = meshing_case(f"plc-{kind}-refusal")
    assert source_payload(positive) == source_payload(refusal)
    _, original, _ = prepare_case(positive, 4, 20000, 120.0)
    _, limited, _ = prepare_case(refusal, 4, 20000, 120.0)
    fields = (
        "maximum_vertices",
        "maximum_edges",
        "maximum_faces",
        "maximum_cells",
        "maximum_connectivity_entries",
        "maximum_data_bytes",
        "maximum_work_units",
        "maximum_cavity_cells",
        "maximum_geometry_queries",
        "maximum_scratch_bytes",
        "maximum_wall_seconds",
    )
    for name in fields:
        assert getattr(limited.limits, name) == (
            1 if name == field else getattr(original.limits, name)
        )

@pytest.mark.parametrize("work, queries", ((None, None), (731, None), (None, 419), (731, 419)))
@pytest.mark.parametrize("scenario", ("planar", "sphere", "distributed", "overset", "cad"))
def test_scenario_exact_resource_overrides_preserve_other_limits(
    scenario: str, work: int | None, queries: int | None
) -> None:
    from tools._meshing_cases import prepare_case, prepare_sphere_case
    from tools.meshing_qualification.scenarios.cad import _cad_native_limits
    from tools.meshing_qualification.scenarios.distributed import (
        _distributed_qualification_source,
    )
    from tools.meshing_qualification.scenarios.overset import (
        _BACKGROUND,
        _overset_qualification_source,
    )

    def limits(work: int | None, queries: int | None) -> Any:
        overrides = {
            "maximum_work_units": work,
            "maximum_geometry_queries": queries,
        }
        match scenario:
            case "planar":
                return prepare_case(
                    meshing_case("planar-hole-feature"), 4, 20000, 120.0, **overrides
                )[1].limits
            case "sphere":
                return prepare_sphere_case(4, 20000, 120.0, **overrides)[1].limits
            case "distributed":
                return _distributed_qualification_source(
                    QualificationControls(
                        maximum_work_units=work, maximum_geometry_queries=queries,
                    )
                )[1].limits
            case "overset":
                return _overset_qualification_source(
                    _BACKGROUND, 4, 20000, 120.0, None, **overrides
                )[1].limits
            case "cad":
                return _cad_native_limits(QualificationControls(
                    maximum_work_units=work, maximum_geometry_queries=queries,
                ))
            case _:
                raise AssertionError(scenario)

    original = limits(None, None)
    overridden = limits(work, queries)
    for name in (
        "maximum_vertices", "maximum_edges", "maximum_faces", "maximum_cells",
        "maximum_connectivity_entries", "maximum_data_bytes", "maximum_cavity_cells",
        "maximum_scratch_bytes", "maximum_wall_seconds",
    ):
        assert getattr(overridden, name) == getattr(original, name)
    assert overridden.maximum_work_units == (
        original.maximum_work_units if work is None else work
    )
    assert overridden.maximum_geometry_queries == (
        original.maximum_geometry_queries if queries is None else queries
    )
    native = overridden.native_limits()
    assert native.maximum_work_units == overridden.maximum_work_units
    assert native.maximum_geometry_queries == overridden.maximum_geometry_queries



def test_stale_plc_request_retains_one_consistent_old_revision() -> None:
    from tools._meshing_cases import corpus_identity, prepare_case, source_payload

    case = meshing_case("plc-stale-revision")
    source, specification, _ = prepare_case(case, 4, 20000, 120.0)
    assert isinstance(source, phx.meshing.NativePlcSource)
    assert isinstance(specification, phx.meshing.VolumeMeshingSpec)
    assert source.source_revision == corpus_identity(case)["source_revision"]
    boundary = specification.boundary_scope
    assert boundary.source_id == source.source_id
    assert boundary.source_revision == "stale"
    assert boundary.source_revision != source.source_revision
    controls = (
        *(control.scope for control in specification.size_controls),
        *(control.scope for control in specification.region_controls),
        *(control.scope for control in specification.patch_controls),
    )
    assert controls
    assert specification.region_controls and specification.patch_controls
    assert all(
        (scope.source_id, scope.source_revision)
        == (boundary.source_id, boundary.source_revision)
        for scope in controls
    )
    assert source_payload(case) == source_payload(meshing_case("plc-material-cavity"))


@pytest.mark.parametrize(
    ("case_name", "category", "phase"),
    (
        (
            "plc-entities-refusal",
            "resource_exhausted",
            "generation_publication_certification",
        ),
        (
            "plc-stale-revision",
            "scope_resolution_failed",
            "geometry_constraint_preparation",
        ),
    ),
)
def test_original_plc_negative_corpus_retains_typed_refusal_without_publication(
    case_name: str, category: str, phase: str
) -> None:
    from tools.meshing_qualification.scenarios.native_lifecycle import (
        qualify_native_lifecycle,
    )

    record = qualify_native_lifecycle(
        case_name,
        resolution=4,
        capacity=20000,
        timeout=120.0,
        repeats=3,
        target_error=0.05,
        adaptation_rounds=1,
    )
    assert record["status"] == "expected-refusal"
    assert record["failure_type"] == "MeshingFailure"
    assert record["failure_category"] == category
    assert record["failure_phase"] == phase
    assert record["source_bits_preserved"] is True
    assert record["epochs"] == []
    assert "checkpoint" not in record
    if case_name == "plc-entities-refusal":
        from tools._meshing_cases import source_payload

        required = len(source_payload(meshing_case(case_name))["vertices"])
        assert record["provider_code"] == "vertex_budget"
        requested = record["requested"]
        achieved = record["achieved"]
        assert isinstance(requested, dict) and isinstance(achieved, dict)
        assert requested["maximum_vertices"] == 1
        assert achieved["required_vertices"] == required
        assert achieved["available_vertices"] == 1


@pytest.mark.parametrize("case_name", ("plc-entities-refusal", "plc-stale-revision"))
def test_plc_negative_corpus_does_not_credit_unrelated_value_errors(
    monkeypatch: pytest.MonkeyPatch, case_name: str
) -> None:
    from tools.meshing_qualification.scenarios import native_lifecycle

    def unrelated_failure(*args: object, **kwargs: object) -> NoReturn:
        raise ValueError(
            "qualification.source must satisfy its publication contract; got an unrelated violation"
        )

    monkeypatch.setattr(native_lifecycle, "_publish_native_lifecycle", unrelated_failure)
    record = native_lifecycle.qualify_native_lifecycle(
        case_name,
        resolution=4,
        capacity=20000,
        timeout=120.0,
        repeats=3,
        target_error=0.05,
        adaptation_rounds=1,
    )
    assert record["status"] == "failed"
    assert record["failure_type"] == "ValueError"
    assert "failure_category" not in record
    assert record["epochs"] == []
    assert "checkpoint" not in record


def test_phase_campaign_keeps_unmeasured_phases_unmeasured() -> None:
    from tools.meshing_qualification._campaign import corpus_phase_evidence

    refused = corpus_phase_evidence({"status": "expected-refusal"}, 20000)
    assert refused["status"] == "unmeasured"
    assert refused["compiler_epochs"] == []
    measured = corpus_phase_evidence(
        {
            "logical_retained_bytes": 321,
            "epochs": [{"epoch": 0, "compiler": {}}],
            "stages_seconds": {
                "solver_lowering_epoch_0": 0.4,
                "solver_compile_epoch_0": 0.5,
            },
        },
        20000,
    )["compiler_epochs"][0]
    assert measured["temporary_bytes"] is None
    assert measured["logical_retained_bytes"] is None
    assert measured["first_execution_seconds"] is None


def test_overset_compiler_phase_projects_actual_counts_and_requires_authored_warms() -> None:
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_phase_gaps,
    )

    controls = QualificationControls(
        scenario="native-overset-lifecycle", repeats=2
    ).to_record()
    compiler = {
        "temporary_bytes": 64,
        "output_bytes": 64,
        "generated_code_bytes": 64,
        "lowering_seconds": 0.01,
        "compile_seconds": 0.02,
        "first_execution_seconds": 0.03,
        "warm_samples_seconds": [0.01, 0.02],
        "compilation_count": 1,
        "compilation_count_scope": "explicit-AOT-compile-invocations",
        "first_execution_compilation_count": 0,
        "warm_recompilation_count": 0,
    }
    payload: dict[str, Any] = {
        "scenario": "native-overset-lifecycle",
        "qualification_controls": controls,
        "epochs": [{"epoch": 0, "compiler": compiler, "retained_bytes": 128}],
    }
    sample = _compiler_consumer_sample(payload)
    evidence = sample.phase_evidence["compiler_epochs"][0]
    assert evidence["compilation_count_scope"] == "explicit-AOT-compile-invocations"
    assert evidence["first_execution_compilation_count"] == 0
    assert evidence["warm_recompilation_count"] == 0
    assert _corpus_phase_gaps(
        sample, full_workflow=True, refusal=False
    ) == []

    from copy import deepcopy

    refused_payload = deepcopy(payload)
    refused_compiler = refused_payload["epochs"][0]["compiler"]
    refused_compiler["warm_samples_seconds"] = [0.01]
    refused_compiler["first_execution_compilation_count"] = 1
    refused = _compiler_consumer_sample(refused_payload)
    gaps = _corpus_phase_gaps(refused, full_workflow=True, refusal=False)
    assert "epochs[0].compiler:warm-samples-count" in gaps
    assert "epochs[0].compiler:first_execution_compilation_count:runtime-compilation" in gaps


def test_distributed_compiler_phases_bind_each_owner_solve_not_topology_epochs() -> None:
    from tools.meshing_qualification._campaign import corpus_phase_evidence

    def solve(seconds: float) -> dict[str, object]:
        return {"compiler": {"temporary_bytes": 64, "warm_samples_seconds": [seconds]}}

    records = corpus_phase_evidence(
        {
            "scenario": "native-distributed-lifecycle",
            # Topology epochs transfer fields exactly and compile no solve.
            "epochs": [{"epoch": 0}, {"epoch": 1}, {"epoch": 2}],
            "initial_fe": [solve(0.1), solve(0.2)],
            "initial_fv": [solve(0.3), solve(0.4)],
            "uninterrupted_fv": [solve(0.5), solve(0.6)],
            "continued_fe": [solve(0.7)],
            "continued_fv": [{}],
        },
        20000,
    )["compiler_epochs"]
    assert [
        (item["evidence_path"], item["solver"], item["phase"], item["epoch"])
        for item in records
    ] == [
        ("initial_fe[0].compiler", "fe", "warm", 0),
        ("initial_fe[1].compiler", "fe", "warm", 0),
        ("initial_fv[0].compiler", "fv", "warm", 0),
        ("initial_fv[1].compiler", "fv", "warm", 0),
        ("uninterrupted_fv[0].compiler", "fv", "warm", 3),
        ("uninterrupted_fv[1].compiler", "fv", "warm", 3),
        ("continued_fe[0].compiler", "fe", "cold", 3),
        ("continued_fv[0].compiler", "fv", "cold", 3),
    ]
    assert [item["warm_samples_seconds"] for item in records[:-1]] == [
        [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7],
    ]
    # An owner whose solve emitted no compiler evidence stays unmeasured.
    assert records[-1]["temporary_bytes"] is None
    assert records[-1]["memory_status"] == "unmeasured"


@pytest.mark.parametrize(
    ("name", "inside", "outside"),
    (
        ("plc-small-angle", (8.0, 0.25, 0.5), (8.0, 1.0, 0.5)),
        ("plc-fixed-boundary", (0.5, 0.5, 0.5), (1.5, 0.5, 0.5)),
        ("plc-sliver-rich", (0.5, 0.5, 0.5), (0.5, 0.5, 1.5)),
    ),
)
def test_original_positive_declared_geometry_classification(
    name: str,
    inside: tuple[float, ...],
    outside: tuple[float, ...],
) -> None:
    from tools._meshing_cases import analytic_region_indices

    assert np.array_equal(
        analytic_region_indices(meshing_case(name), np.asarray((inside, outside))),
        np.asarray((0, -1)),
    )


def test_periodic_polyhedral_source_orbits_close_exactly_on_declared_cube() -> None:
    from tools._meshing_cases import prepare_polyhedral_case

    source, request, _ = prepare_polyhedral_case(
        "polyhedral-periodic-cube", 4, 20000, 120.0
    )
    points = np.asarray(source.complex.vertices)
    offsets = np.asarray(source.complex.polygon_offsets)
    values = np.asarray(source.complex.polygon_vertices)
    for constraint in request.periodic_constraints:
        source_face = int(np.asarray(constraint.source_scope.entity_ids)[0])
        target_face = int(np.asarray(constraint.target_scope.entity_ids)[0])
        first = points[values[offsets[source_face] : offsets[source_face + 1]]]
        second = points[values[offsets[target_face] : offsets[target_face + 1]]]
        matrix = np.asarray(constraint.transform)
        mapped = first @ matrix[:3, :3].T + matrix[:3, 3]
        assert set(map(tuple, mapped)) == set(map(tuple, second))
        normal_first = np.cross(first[1] - first[0], first[2] - first[0])
        normal_second = np.cross(second[1] - second[0], second[2] - second[0])
        assert np.dot(normal_first, normal_second) < 0.0


def test_periodic_prescribed_source_is_not_replaced_by_resolution_sites() -> None:
    from tools._meshing_cases import prepare_polyhedral_case

    original, original_request, _ = prepare_polyhedral_case(
        "polyhedral-periodic-cube", 4, 20000, 120.0, source=True
    )
    later, later_request, _ = prepare_polyhedral_case(
        "polyhedral-periodic-cube", 8, 20000, 120.0, source=False
    )
    assert original.binding_id == later.binding_id
    assert original_request.specification_id == later_request.specification_id
    assert np.array_equal(
        np.asarray(original.sites), np.asarray(((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)))
    )
    assert np.array_equal(np.asarray(original.weights), np.asarray((0.0, 0.0)))


def test_thin_solid_retains_original_planar_source_bits_on_both_caps() -> None:
    from tools._meshing_cases import source_payload

    planar = np.asarray(
        source_payload(meshing_case("planar-thin-channel"))["vertices"], dtype=np.float64
    )
    solid = np.asarray(
        source_payload(meshing_case("plc-thin-channel"))["vertices"], dtype=np.float64
    )
    for height in (0.0, 1.0):
        cap = solid[solid[:, 2] == height, :2].copy()
        assert np.array_equal(cap.view(np.uint64), planar.view(np.uint64))


def test_original_thin_channel_oracle_rejects_the_nearby_bulk_gap() -> None:
    from tools._meshing_cases import analytic_region_indices

    points = np.asarray(
        (
            (0.2, 0.2, 0.5),
            (0.5, 0.2, 0.5),
            (0.8, 0.2, 0.5),
            (0.5, 0.18, 0.5),
            (0.5, 0.22, 0.5),
        )
    )
    assert np.array_equal(
        analytic_region_indices(meshing_case("plc-thin-channel"), points),
        np.asarray((0, 0, 0, -1, -1)),
    )


@pytest.mark.parametrize(
    ("name", "analytic_volume"),
    (
        ("plc-nonconvex", 3.0),
        ("plc-small-angle", 16.0),
        ("plc-fixed-boundary", 1.0),
        ("plc-sliver-rich", 1.0),
        ("plc-thin-channel", 0.324),
    ),
)
def test_feature_free_public_source_preparation_preserves_declared_solid(
    name: str,
    analytic_volume: float,
) -> None:
    from tools._meshing_cases import prepare_case

    source, _, _ = prepare_case(meshing_case(name), 4, 20000, 120.0)
    complex_ = source.complex
    points = np.asarray(complex_.vertices)
    offsets = np.asarray(complex_.polygon_offsets)
    values = np.asarray(complex_.polygon_vertices)
    contributions = []
    for first, last in zip(offsets[:-1], offsets[1:], strict=True):
        polygon = points[values[first:last]]
        contributions.extend(
            np.dot(polygon[0], np.cross(polygon[index], polygon[index + 1])) / 6.0
            for index in range(1, len(polygon) - 1)
        )
    assert sum(contributions) == pytest.approx(
        analytic_volume, abs=_CORPUS_ANALYTIC_MEASURE_ATOL
    )


def test_corpus_registration_is_current_licensed_and_deterministic() -> None:
    from tools._meshing_cases import (
        corpus_registration_record,
        load_corpus_registration,
    )

    stored = load_corpus_registration()
    assert stored == corpus_registration_record()
    assert stored["registration_id"] == corpus_registration_record()["registration_id"]
    cases = stored["cases"]
    assert isinstance(cases, list)
    assert {case["expected_admission"] for case in cases} == {"positive", "refusal"}
    for case in cases:
        assert case["source_provenance"]["license"] == "LicenseRef-PHYDRA-Proprietary"
        assert case["source_artifact"]["sha256"]
        assert case["independent_reference"]["reference_digest"]


def test_cad_fixture_has_reviewed_provenance_and_content_identity() -> None:
    from tools._meshing_cases import load_cad_fixture

    fixture = load_cad_fixture()
    provenance = fixture["provenance"]
    reference = fixture["independent_reference"]
    assert isinstance(provenance, dict)
    assert isinstance(reference, dict)
    assert fixture["license"] == "LicenseRef-PHYDRA-Proprietary"
    assert provenance["method"] == "independently-authored analytic coordinates"
    assert reference["intersection_volume"] == 0.5


def test_frozen_request_record_retains_every_hard_resource_limit() -> None:
    from tools._meshing_cases import frozen_request_record, prepare_case

    case = meshing_case("planar-hole-feature")
    source, request, options = prepare_case(case, 4, 20000, 120.0)
    first = frozen_request_record(source, request, options)
    second = frozen_request_record(source, request, options)
    assert first == second
    assert first["frozen_request_id"] == second["frozen_request_id"]
    assert first["limits"] == {
        "maximum_vertices": 20000,
        "maximum_edges": 160000,
        "maximum_faces": 160000,
        "maximum_cells": 160000,
        "maximum_connectivity_entries": 1280000,
        "maximum_data_bytes": 20480000,
        "maximum_work_units": 640000,
        "maximum_cavity_cells": 20000,
        "maximum_geometry_queries": 1280000,
        "maximum_scratch_bytes": 81920000,
        "maximum_wall_seconds": 120.0,
        "limits_id": request.limits.limits_id,
    }


def test_mandatory_result_never_credits_refusal_as_positive() -> None:
    from tools.meshing_qualification._records import _mandatory_result

    profile = {
        "expected_outcome": "expected-refusal",
        "workflow": "resource-boundary",
        "scenario": None,
        "controls": {},
        "resolution_scale": 1,
        "minimum_adaptation_rounds": 0,
        "source_reference": {"source_digest": "source"},
    }
    assessment = {
        "positive_completion": True,
        "refusal_certified": True,
        "execution": "expected-refusal",
        "scientific_certification": "unassessed",
        "blocking_status": None,
        "release": "unassessed",
        "leadership": "unassessed",
        "sample_binding": None,
    }
    result = _mandatory_result("bounded-refusal", profile, assessment)
    assert result["status"] == "refusal-certified"
    assert result["positive_completion"] is False
    assert result["refusal_certified"] is True


def test_routine_case_certification_does_not_complete_a_workflow() -> None:
    from tools.meshing_qualification._campaign import _complete_campaign_sample
    from tools.meshing_qualification._records import _mandatory_result
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_profile_assessment,
    )

    profile = {
        "expected_outcome": "passed",
        "workflow": "constrained-solid",
        "scenario": None,
        "controls": {},
        "resolution_scale": 1,
        "minimum_adaptation_rounds": 0,
        "full_workflow": False,
        "source_reference": {"source_digest": "source"},
    }
    assessment = _corpus_profile_assessment(
        profile,
        _complete_campaign_sample(
            {
                "status": "passed",
                "case_contract_completion": "complete",
                "mandatory_workflow_completion": "incomplete",
                "independent_scientific_certification": "passed",
            },
            1,
            positive=False,
            accepted=True,
        ),
    )
    assert assessment["qualification_eligible"] is True
    assert assessment["case_certified"] is True
    assert assessment["positive_completion"] is False
    result = _mandatory_result("routine-solid", profile, assessment)
    assert result["status"] == "case-certified"
    assert result["case_certified"] is True
    assert result["positive_completion"] is False


class _LaunchedControlOverrides(TypedDict, total=False):
    family_case: str
    hybrid_profile: str
    surface_case: str


def _launched_controls(
    scenario: str, **controls: Unpack[_LaunchedControlOverrides]
) -> QualificationControls:
    """Controls exactly as the campaign command line hands them to a child."""
    return QualificationControls(
        scenario=scenario,
        corpus_case="planar-hole-feature",
        resolution=2,
        capacity=16,
        timeout=120.0,
        repeats=1,
        target_error=0.05,
        adaptation_rounds=1,
        **controls,
    )


def _launched_expectations(scenario: str) -> dict[str, object]:
    """Controls the real campaign launcher binds a child's verdict to."""
    from tools.meshing_qualification._campaign import _prepare_campaign_attempt

    return _prepare_campaign_attempt(
        scenario, 2, 16, 120.0, 1, 0.05, 1, "native", scenario_override=scenario
    ).expected_controls


def _emitted(record: dict[str, object]) -> dict[str, object]:
    """Round-trip a child record through the JSON edge the campaign consumer reads."""
    return json.loads(json.dumps(record))


def _child_runtime() -> dict[str, object]:
    """The runtime identity a child in this environment loads and records."""
    from tools.meshing_qualification._runtime import _runtime_identity

    return _runtime_identity()


def _exited(record: dict[str, object], returncode: int = 0) -> CampaignProcessResult:
    """A child that printed this record and exited before the scientific deadline."""
    return CampaignProcessResult(
        returncode=returncode,
        stdout=json.dumps(record),
        stderr="",
        deadline_expired=False,
    )


def test_workflow_verdict_cannot_be_moved_to_another_scientific_record() -> None:
    from tools.meshing_qualification._records import (
        _bound_workflow_verdict,
        _finalize_mandatory_cli_record,
    )

    runtime = _child_runtime()
    build = runtime["build_id"]
    assert isinstance(build, str)
    cad = _emitted(
        _finalize_mandatory_cli_record(
            _launched_controls("native-cad-lifecycle"),
            {
                "scenario": "native-cad-lifecycle",
                "runtime": runtime,
                "status": "passed",
                "independent_scientific_certification": "passed",
                "case_contract_completion": "complete",
                "mandatory_workflow_completion": "incomplete",
                "physical_error": 0.01,
            },
        )
    )
    image = _emitted(
        _finalize_mandatory_cli_record(
            _launched_controls("native-image-lifecycle"),
            {
                "scenario": "native-image-lifecycle",
                "runtime": runtime,
                "status": "passed",
                "independent_scientific_certification": "passed",
                "mandatory_workflow_completion": "complete",
                "physical_error": 0.02,
            },
        )
    )
    cad_launch = _launched_expectations("native-cad-lifecycle")
    image_launch = _launched_expectations("native-image-lifecycle")
    genuine_cad = _bound_workflow_verdict(cad, cad_launch, build)
    assert genuine_cad["positive_completion"] is False
    assert genuine_cad["case_accepted"] is True
    assert genuine_cad["status"] == "blocked"
    image_verdict = _bound_workflow_verdict(image, image_launch, build)
    assert image_verdict["positive_completion"] is True

    swapped = {
        **cad,
        "workflow_result": image["workflow_result"],
        "positive_completion": True,
    }
    with pytest.raises(ValueError, match=r"_bound_workflow_verdict\.workflow_result"):
        _bound_workflow_verdict(swapped, cad_launch, build)
    edited = {**cad, "mandatory_workflow_completion": "complete"}
    with pytest.raises(ValueError, match=r"_bound_workflow_verdict\.workflow_result"):
        _bound_workflow_verdict(edited, cad_launch, build)
    with pytest.raises(
        ValueError, match=r"_bound_workflow_verdict\.qualification_controls"
    ):
        _bound_workflow_verdict(image, cad_launch, build)
    foreign = {**cad, "runtime": {**runtime, "build_id": "another-loaded-build"}}
    with pytest.raises(ValueError, match=r"_bound_workflow_verdict\.runtime\.build_id"):
        _bound_workflow_verdict(foreign, cad_launch, build)


def test_distinct_sources_and_records_have_distinct_workflow_verdicts() -> None:
    from tools.meshing_qualification._records import _finalize_mandatory_cli_record

    body = {
        "status": "passed",
        "independent_scientific_certification": "passed",
        "mandatory_workflow_completion": "complete",
    }
    transfinite, multiblock = (
        _finalize_mandatory_cli_record(
            _launched_controls("native-family-lifecycle", family_case=case), dict(body)
        )["workflow_result"]
        for case in ("transfinite-quad", "multiblock-quad")
    )
    assert transfinite["status"] == multiblock["status"] == "positive-complete"
    assert transfinite["controls_id"] != multiblock["controls_id"]
    assert transfinite["result_id"] != multiblock["result_id"]
    other_result = _finalize_mandatory_cli_record(
        _launched_controls("native-family-lifecycle", family_case="transfinite-quad"),
        {**body, "physical_error": 0.03},
    )["workflow_result"]
    assert other_result["evidence_id"] != transfinite["evidence_id"]
    assert other_result["result_id"] != transfinite["result_id"]


def test_accepted_case_timing_is_reported_without_mandatory_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.meshing_qualification._campaign as benchmarks
    from tools.meshing_qualification._records import _finalize_mandatory_cli_record

    controls = _launched_controls("native-lifecycle")
    runtime = _child_runtime()
    accepted = _finalize_mandatory_cli_record(
        controls,
        {
            "scenario": "native-lifecycle",
            "runtime": runtime,
            "status": "passed",
            "independent_scientific_certification": "passed",
            "case_contract_completion": "complete",
            "mandatory_workflow_completion": "incomplete",
            "time_to_target_error_seconds": 3.0,
            "end_to_end_seconds": 7.0,
            "physical_error": 0.01,
            "independent_acceptance": None,
        },
    )
    failed = _finalize_mandatory_cli_record(
        controls,
        {
            "scenario": "native-lifecycle",
            "runtime": runtime,
            "status": "failed",
            "independent_scientific_certification": "passed",
            "case_contract_completion": "incomplete",
            "mandatory_workflow_completion": "incomplete",
            "time_to_target_error_seconds": 1.0,
            "end_to_end_seconds": 2.0,
            "physical_error": 0.2,
            "independent_acceptance": None,
        },
    )
    outputs = iter((accepted, failed))

    def completed(*args: object, **kwargs: object) -> CampaignProcessResult:
        del args, kwargs
        record = next(outputs)
        return _exited(record, 0 if record["status"] == "passed" else 1)

    monkeypatch.setattr(benchmarks, "run_campaign_process", completed)
    samples = [
        benchmarks._native_campaign_sample(
            "planar-hole-feature", 2, 16, 120.0, 1, 0.05, 1, "native"
        )
        for _ in range(2)
    ]
    assert [sample.accepted_case for sample in samples] == [True, False]
    assert [sample.positive_completion for sample in samples] == [False, False]
    summary = benchmarks._campaign_candidate_summary(
        samples, ("passed", "failed"), physics_error=False
    )
    assert summary["mandatory_positive_completions"] == 0
    assert summary["accepted_cases"] == 1
    assert summary["accepted_case_rate"] == 0.5
    target_times = summary["time_to_target_error_seconds"]
    lifecycle_times = summary["end_to_end_seconds"]
    assert isinstance(target_times, dict)
    assert isinstance(lifecycle_times, dict)
    assert target_times["median"] == 3.0
    assert lifecycle_times["median"] == 7.0
    assert summary["worst_accepted_l2_error"] == 0.01
    binding = samples[0].binding
    verdict = accepted["workflow_result"]
    assert isinstance(verdict, dict)
    assert binding["workflow_result_id"] == verdict["result_id"]
    assert binding["evidence_id"] == verdict["evidence_id"]


def test_source_only_accepted_profile_keeps_campaign_without_physical_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.meshing_qualification._campaign as benchmarks
    from tools.meshing_qualification._records import _finalize_mandatory_cli_record

    # A full_workflow=False hybrid source contract publishes no physical error.
    accepted = _finalize_mandatory_cli_record(
        _launched_controls("native-hybrid-lifecycle", hybrid_profile="plate5"),
        {
            "scenario": "native-hybrid-lifecycle",
            "runtime": _child_runtime(),
            "status": "passed",
            "independent_scientific_certification": "passed",
            "case_contract_completion": "complete",
            "mandatory_workflow_completion": "incomplete",
            "end_to_end_seconds": 4.0,
            "independent_acceptance": None,
        },
    )

    def completed(*args: object, **kwargs: object) -> CampaignProcessResult:
        del args, kwargs
        return _exited(accepted)

    monkeypatch.setattr(benchmarks, "run_campaign_process", completed)
    campaign = benchmarks.benchmark_native_lifecycle(
        "native-hybrid-lifecycle",
        2,
        16,
        120.0,
        1,
        0.05,
        1,
        None,
        benchmark_warm_repeats=1,
        scenario_override="native-hybrid-lifecycle",
        controls={"hybrid_profile": "plate5"},
    )
    candidates = campaign["candidates"]
    assert isinstance(candidates, dict)
    summary = candidates["native"]
    assert summary["accepted_cases"] == 1
    assert summary["mandatory_positive_completions"] == 0
    assert summary["worst_accepted_physics_error"] is None
    assert summary["accepted_cases_without_physical_error"] == 1
    assert summary["end_to_end_seconds"]["median"] == 4.0


def test_benchmark_sample_does_not_promote_bare_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.meshing_qualification._campaign as benchmarks
    from tools.meshing_qualification._records import _finalize_mandatory_cli_record

    blocked = _finalize_mandatory_cli_record(
        _launched_controls("native-surface-pde", surface_case="sphere"),
        {
            "status": "passed",
            "runtime": _child_runtime(),
            "mandatory_workflow_completion": "complete",
            "independent_scientific_certification": "unassessed",
        },
    )

    def completed(*args: object, **kwargs: object) -> CampaignProcessResult:
        del args, kwargs
        return _exited(blocked)

    monkeypatch.setattr(benchmarks, "run_campaign_process", completed)
    sample = benchmarks._native_campaign_sample(
        "sphere-feature-surface", 2, 16, 120.0, 1, 0.05, 1, "native"
    ).to_record()
    assert sample["status"] == "passed"
    assert sample["positive_completion"] is False
    assert sample["accepted_case"] is False
    assert sample["phase_evidence"]["status"] == "recorded"
    assert sample["resource_evidence"] == sample["phase_evidence"]["resources"]


def test_runtime_identity_binds_the_loaded_native_artifact() -> None:
    from phydrax._meshcore import (
        load_meshcore,
        meshcore_available,
        meshcore_runtime_identity,
    )
    from tools._meshing_cases import python_source_revision
    from tools.meshing_qualification._runtime import _runtime_identity

    if not meshcore_available():
        pytest.skip("Exact loaded native identity requires the configured meshcore.")
    expected = meshcore_runtime_identity()
    library = load_meshcore()
    runtime = _runtime_identity()
    native = runtime["meshcore_build"]
    build = runtime["build"]
    environment = runtime["environment"]
    assert isinstance(native, dict)
    assert isinstance(build, dict)
    assert isinstance(environment, dict)
    assert {
        name: native[name]
        for name in ("release", "source_digest", "binary_digest", "configuration")
    } == expected
    loaded_path = native["loaded_path"]
    assert isinstance(loaded_path, str)
    assert Path(loaded_path) == library.path.resolve()
    assert sha256(Path(loaded_path).read_bytes()).hexdigest() == native["binary_digest"]
    assert environment["native_configuration"] == native["configuration"]
    assert phx.__file__ is not None
    assert build["loaded_python_source_digest"] == python_source_revision(
        Path(phx.__file__).resolve().parent
    )


@pytest.mark.parametrize(
    ("archive", "content_id", "missing"),
    (
        ("caller-owned-archive", None, "hybrid_content_id"),
        (None, "caller-owned-content", "hybrid_archive"),
    ),
    ids=("archive-without-identity", "identity-without-archive"),
)
def test_registered_benchmark_corpus_preserves_partial_prerequisites(
    archive: str | None, content_id: str | None, missing: str
) -> None:
    from tools.meshing_qualification._campaign import _benchmark_registered_native_corpus
    from tools.meshing_qualification._controls import BenchmarkControls

    profiles = {
        "native-hybrid-lifecycle:archived-layer-core": {
            "case_name": "native-hybrid-lifecycle",
            "workflow": "hybrid-layers",
            "scenario": "native-hybrid-lifecycle",
            "controls": {"hybrid_profile": "archived-layer-core"},
            "prerequisites": ("hybrid_archive", "hybrid_content_id"),
            "source_reference": {
                "kind": "owning-scenario-source-descriptor",
                "source_digest": "frozen-source",
            },
        },
    }
    arguments = BenchmarkControls(
        corpus_profile=None,
        hybrid_archive=archive,
        hybrid_content_id=content_id,
        distributed_checkpoint_root=None,
        resolution=(4,),
        capacity=(20000,),
        timeout=120.0,
        repeats=3,
        target_error=0.05,
        adaptation_rounds=1,
        comparison=None,
    )
    first = _benchmark_registered_native_corpus(arguments, profiles)
    records = first["profiles"]
    assert isinstance(records, dict)
    record = records["native-hybrid-lifecycle:archived-layer-core"]
    assert isinstance(record, dict)
    assert record["status"] == "blocked-missing-input"
    assert record["missing_inputs"] == [missing]
    assert record["campaigns"] == []


def test_distributed_benchmark_attempts_have_fresh_checkpoint_roots() -> None:
    from tools.meshing_qualification._campaign import _native_campaign_sample

    sample = _native_campaign_sample(
        "native-distributed-lifecycle",
        4,
        20000,
        120.0,
        3,
        0.05,
        1,
        "triangle",
        scenario_override="native-distributed-lifecycle",
        controls={"distributed_checkpoint_root": "/shared/frozen-campaign"},
        attempt_index=2,
    ).to_record()
    frozen = sample["frozen_controls"]
    assert isinstance(frozen, dict)
    controls = frozen["scenario_controls"]
    assert isinstance(controls, dict)
    assert controls["distributed_checkpoint_root"] == (
        "/shared/frozen-campaign-resolution-4-capacity-20000-attempt-2"
    )
    assert sample["process_started"] is False
    assert sample["phase_evidence"]["status"] == "unmeasured"


def test_mandatory_benchmark_campaign_registration_is_current() -> None:
    from tools._meshing_cases import MANDATORY_WORKFLOWS
    from tools.meshing_qualification._campaign import (
        campaign_registration_record,
        load_native_campaign_registration,
    )

    record = load_native_campaign_registration()
    assert record == campaign_registration_record()
    workflows = record["workflows"]
    assert isinstance(workflows, list)
    assert [workflow["workflow"] for workflow in workflows] == list(MANDATORY_WORKFLOWS)
    assert all(workflow["expected_admission"] == "positive" for workflow in workflows)
    assert all(
        workflow["refusal_counts_as_positive_completion"] is False
        for workflow in workflows
    )
    assert record["contains_benchmark_results"] is False


@pytest.mark.parametrize("field", ("profiles", "scenarios", "provenance"))
def test_campaign_registration_rejects_rehashed_drift(tmp_path: Path, field: str) -> None:
    from phydrax._fingerprint import canonical_fingerprint
    from tools.meshing_qualification._campaign import (
        campaign_registration_record,
        load_native_campaign_registration,
    )

    record = campaign_registration_record()
    if field == "provenance":
        record[field] = {"origin": "unreviewed"}
    else:
        workflows = record["workflows"]
        assert isinstance(workflows, list)
        workflows[0][field] = ["unreviewed-route"]
    body = {name: value for name, value in record.items() if name != "registration_id"}
    record["registration_id"] = canonical_fingerprint(body)
    path = tmp_path / "campaigns.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=r"load_native_campaign_registration\.value"):
        load_native_campaign_registration(path)


@pytest.mark.strict_jax
def test_design_surface_area_keeps_all_coordinate_blocks() -> None:
    from tools.meshing_qualification.scenarios.design import (
        _design_qualification_surface_area,
    )

    points = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)), dtype=np.float64)
    cells = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32)
    combined = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("whole", "triangle", cells),)
    )
    partitioned = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "first",
                "triangle",
                cells[:1],
                global_ids=np.asarray((0,), dtype=np.int64),
            ),
            phx.discretization.CellBlock(
                "second",
                "triangle",
                cells[1:],
                global_ids=np.asarray((1,), dtype=np.int64),
            ),
        ),
    )
    source, target = (
        phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
        for mesh in (combined, partitioned)
    )
    assert _design_qualification_surface_area(source) == 1.0
    assert _design_qualification_surface_area(target) == 1.0


@pytest.mark.parametrize(
    "configured_root", (False, True), ids=("temporary", "configured")
)
def test_corpus_distributed_checkpoint_destination_spans_child_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured_root: bool,
) -> None:
    from phydrax.meshing import BisectionCompatibility
    from tools.meshing_qualification import _campaign
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification._records import (
        _corpus_profiles,
        _finalize_mandatory_cli_record,
    )
    from tools.meshing_qualification.scenarios.native_corpus import (
        _execute_corpus_profiles,
    )

    profile_name = "native-distributed-lifecycle"
    destination = str(tmp_path / "distributed") if configured_root else None
    arguments = QualificationControls(
        scenario="native-corpus",
        corpus_profile=(profile_name,),
        distributed_checkpoint_root=destination,
    )
    roots: list[Path] = []

    def completed(
        command: list[str], *, cwd: Path, timeout: float
    ) -> CampaignProcessResult:
        del cwd
        assert timeout == 600.0
        assert (
            arguments.distributed_bisection_compatibility is BisectionCompatibility.REJECT
        )
        assert command[command.index("--distributed-bisection-compatibility") + 1] == (
            BisectionCompatibility.UNIFORM_REFINEMENT.value
        )
        root = Path(command[command.index("--distributed-checkpoint-root") + 1])
        assert root.parent.is_dir()
        assert not root.exists()
        repository = root / "repository"
        repository.mkdir(parents=True)
        checkpoint = repository / "checkpoint"
        checkpoint.write_bytes(b"published-for-restore")
        assert checkpoint.read_bytes() == b"published-for-restore"
        roots.append(root)
        # The child echoes the controls it parsed from this command line.
        record = _finalize_mandatory_cli_record(
            QualificationControls(
                scenario=profile_name,
                timeout=600.0,
                maximum_work_units=8_000_000,
                maximum_memory_bytes=24 * 1024**3,
                distributed_parts=4,
                restart_parts=2,
                distributed_processes=4,
                distributed_checkpoint_root=str(root),
                distributed_bisection_compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
            ),
            {
                "status": "passed",
                "runtime": _child_runtime(),
                "mandatory_workflow_completion": "complete",
                "independent_scientific_certification": "passed",
            },
        )
        return _exited(record)

    monkeypatch.setattr(_campaign, "run_campaign_process", completed)
    campaign = _execute_corpus_profiles(arguments, _corpus_profiles())
    assert campaign.missing_inputs == {}
    assert campaign.by_profile[profile_name].positive_completion is True
    frozen = campaign.samples[0]["frozen_controls"]["scenario_controls"]
    assert frozen["distributed_checkpoint_root"] == str(roots[0])
    assert frozen["distributed_bisection_compatibility"] == (
        BisectionCompatibility.UNIFORM_REFINEMENT.value
    )
    if configured_root:
        assert str(roots[0]) == (f"{destination}-resolution-4-capacity-20000-attempt-0")
        assert (roots[0] / "repository" / "checkpoint").is_file()
    else:
        assert not roots[0].parent.exists()
        second = _execute_corpus_profiles(arguments, _corpus_profiles())
        assert second.by_profile[profile_name].positive_completion is True
        assert roots[0] != roots[1]
        assert not roots[1].parent.exists()


@pytest.mark.parametrize("benchmark", (False, True), ids=("qualification", "benchmark"))
def test_distributed_label_control_roundtrip_uses_owner_selector(benchmark: bool) -> None:
    from phydrax.meshing import BisectionCompatibility
    from tools.meshing_qualification._controls import (
        BenchmarkControls,
        QualificationControls,
    )

    control_type = BenchmarkControls if benchmark else QualificationControls
    default = control_type()
    assert default.distributed_bisection_compatibility is BisectionCompatibility.REJECT
    record = {
        **default.to_record(),
        "distributed_bisection_compatibility": BisectionCompatibility.UNIFORM_REFINEMENT.value,
    }
    restored = control_type.from_record(record)
    assert restored.distributed_bisection_compatibility is (
        BisectionCompatibility.UNIFORM_REFINEMENT
    )
    assert restored.to_record() == record
    with pytest.raises(ValueError):
        control_type.from_record(
            {**record, "distributed_bisection_compatibility": "skip"}
        )


def test_surface_qualification_density_preserves_physical_inventory_with_all_roles() -> (
    None
):
    from phydrax.discretization.fem import prepare_surface_chart_field_transfer
    from tests._support.mapped_cells import curved_surface_chart_fv

    _, _, chart, _ = curved_surface_chart_fv()
    fields = tuple(
        field for _, field in phx.meshing.qualification.surface_field_templates()
    )
    source = phx.discretization.FiniteElementPlan(
        chart.source_mesh, fields, coordinate_spec=chart.source_geometry
    ).prepare()
    target = phx.discretization.FiniteElementPlan(
        chart.target_mesh, fields, coordinate_spec=chart.target_geometry
    ).prepare()
    assert tuple(field.name for field in source.field_spaces) == (
        "u",
        "density",
        "history",
        "curl",
        "flux",
    )
    # Density has the same scalar P1 order as the two intensive fields, with
    # independent cell coefficients; compatible circulation and flux stay present.
    assert tuple(
        {element.conformity for element in elements} for elements in source.elements
    ) == ({"H1"}, {"L2"}, {"H1"}, {"Hcurl"}, {"Hdiv"})
    assert all(
        element.degree == 1 for elements in source.elements for element in elements
    )
    assert source.field_spaces[1].layout.size == 3 * source.mesh.entity_set(2).count
    assert target.field_spaces[1].layout.size == 3 * target.mesh.entity_set(2).count
    transfer = prepare_surface_chart_field_transfer(
        source,
        target,
        chart,
        field_name="density",
        source_geometry=chart.source_geometry,
        target_geometry=chart.target_geometry,
        semantics="conservative-density",
    )
    values = 2.0 + source.dof_maps[1].dof_coordinates[:, 0]
    moved = transfer.transfer.apply(values)
    assert transfer.evidence.passed
    assert transfer.transfer.conservative
    assert not transfer.transfer.preserves_constants
    assert transfer.source_measures is not None
    assert transfer.target_measures is not None
    source_terms = np.asarray(transfer.source_measures) * np.asarray(values)
    target_terms = np.asarray(transfer.target_measures) * np.asarray(moved)
    # The owner certifies per-source-column content error. Publishing two
    # weighted inventories adds at most gamma_n times the absolute terms,
    # counting every product, sum, and final subtraction (Higham, Accuracy and
    # Stability of Numerical Algorithms, standard floating-point model).
    operations = 2 * (source_terms.size + target_terms.size) + 1
    epsilon = np.finfo(source_terms.dtype).eps
    publication_bound = (
        operations
        * epsilon
        / (1.0 - operations * epsilon)
        * (np.sum(np.abs(source_terms)) + np.sum(np.abs(target_terms)))
    )
    allowance = (
        transfer.evidence.bound("content") * np.sum(np.abs(np.asarray(values)))
        + publication_bound
    )
    assert abs(np.sum(source_terms) - np.sum(target_terms)) <= allowance


@pytest.mark.parametrize(
    ("scenario", "controls", "names"),
    (
        (
            "native-surface-pde",
            {"surface_case": "folded_plates"},
            {"u", "density", "history", "curl", "flux"},
        ),
        ("native-design-lifecycle", {}, {"u", "density", "history"}),
        ("native-distributed-lifecycle", {}, {"temperature"}),
        (
            "native-periodic-lifecycle",
            {"periodic_case": "skew-lattice"},
            {"u", "vector", "flux"},
        ),
        ("native-family-lifecycle", {"family_case": "dual-quad"}, {"u"}),
        (
            "native-family-lifecycle",
            {"family_case": "curved-quarter-extrusion-quad"},
            {"u", "state"},
        ),
        (
            "native-family-lifecycle",
            {"family_case": "dual-hex"},
            {"u", "current", "history", "state"},
        ),
        (
            "native-family-lifecycle",
            {"family_case": "hex-dominant"},
            {"u", "current", "history", "state"},
        ),
        ("native-family-lifecycle", {"family_case": "transfinite-quad"}, {"u", "state"}),
        (
            "native-hybrid-lifecycle",
            {"hybrid_profile": "unit-box"},
            {"H1", "DG", "Hcurl", "Hdiv", "u", "damage-history", "state", "original-u"},
        ),
        (
            "native-hybrid-lifecycle",
            {"hybrid_profile": "concave-box-core-lifecycle"},
            {"H1", "DG", "Hcurl", "Hdiv", "u", "damage-history", "state"},
        ),
        (
            "native-hybrid-lifecycle",
            {"hybrid_profile": "archived-layer-core"},
            {"H1", "DG", "Hcurl", "Hdiv", "u", "damage-history", "state"},
        ),
        ("native-envelope-lifecycle", {}, {"u", "density", "history"}),
        ("native-overset-lifecycle", {}, {"u"}),
        ("native-implicit-lifecycle", {}, {"u"}),
        ("native-image-lifecycle", {}, {"bulk", "network", "reservoirs"}),
    ),
    ids=(
        "surface",
        "design",
        "distributed",
        "periodic",
        "quad",
        "curved-quad",
        "hex",
        "mixed-hex",
        "structured-quad",
        "hybrid-solve",
        "hybrid-history",
        "archived-history",
        "envelope",
        "overset",
        "implicit",
        "image",
    ),
)
def test_scenario_source_identity_includes_every_authored_field(
    scenario: str,
    controls: dict[str, str | int | float | tuple[float, ...]],
    names: set[str],
) -> None:
    from tools.meshing_qualification._records import _scenario_source_reference

    reference = _scenario_source_reference(scenario, controls)
    templates = reference["field_templates"]
    assert isinstance(templates, list)
    assert {template["name"] for template in templates} == names
    assert [template["name"] for template in templates] == sorted(
        template["name"] for template in templates
    )
    assert all(
        template["role"] and template["space_family"] and template["continuity"]
        for template in templates
    )
    source_only = {
        name: reference[name] for name in ("kind", "scenario", "controls", "driver")
    }
    assert reference["source_digest"] != phx.canonical_fingerprint(source_only)
    assert reference["source_digest"] == phx.canonical_fingerprint(
        {**source_only, "field_templates": templates}
    )


def test_surface_source_identity_tracks_declarations_not_factory_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.meshing_qualification._records import _scenario_source_reference

    qualification = phx.meshing.qualification
    factory = qualification.surface_field_templates
    fields = factory()
    baseline = _scenario_source_reference(
        "native-surface-pde", {"surface_case": "folded_plates"}
    )
    monkeypatch.setattr(qualification, "surface_field_templates", lambda: fields[::-1])
    assert (
        _scenario_source_reference(
            "native-surface-pde", {"surface_case": "folded_plates"}
        )
        == baseline
    )
    changed = tuple(
        (
            role,
            phx.discretization.FiniteElementFieldSpec(
                field.name, phx.discretization.lagrange_element("triangle", 1)
            )
            if field.name == "density"
            else field,
        )
        for role, field in fields
    )
    monkeypatch.setattr(qualification, "surface_field_templates", lambda: changed)
    revised = _scenario_source_reference(
        "native-surface-pde", {"surface_case": "folded_plates"}
    )
    assert revised["source_digest"] != baseline["source_digest"]
    assert revised["source_revision"] != baseline["source_revision"]
    assert revised["source_artifact"] != baseline["source_artifact"]
    original_templates = baseline["field_templates"]
    revised_templates = revised["field_templates"]
    assert isinstance(original_templates, list)
    assert isinstance(revised_templates, list)
    original_fields = {field["name"]: field for field in original_templates}
    revised_fields = {field["name"]: field for field in revised_templates}
    assert (
        original_fields["density"]["field_spec_id"]
        != revised_fields["density"]["field_spec_id"]
    )
    for name in ("u", "history", "curl", "flux"):
        assert original_fields[name] == revised_fields[name]


def test_wall_only_source_identity_does_not_invent_fields() -> None:
    from tools.meshing_qualification._records import _scenario_source_reference

    reference = _scenario_source_reference(
        "native-hybrid-lifecycle", {"hybrid_profile": "plate5"}
    )
    assert "field_templates" not in reference
    descriptor = {
        name: reference[name] for name in ("kind", "scenario", "controls", "driver")
    }
    assert reference["source_digest"] == phx.canonical_fingerprint(descriptor)


@pytest.mark.parametrize("scenario", tuple(SCENARIO_MODULES))
def test_default_qualification_controls_roundtrip_for_every_scenario(
    scenario: str,
) -> None:
    defaults = QualificationControls(scenario=scenario)
    record = defaults.to_record()
    assert set(record) == {field.name for field in fields(QualificationControls)}
    restored = QualificationControls.from_record(record)
    assert restored.scenario == scenario
    assert restored.to_record() == record
    assert restored.distributed_bisection_compatibility is (
        defaults.distributed_bisection_compatibility
    )


@pytest.mark.parametrize("diffusivity", (1.0, 2.0))
def test_compiled_solver_reuses_only_one_epoch_assembly(
    material_interface_publication: tuple[phx.meshing.CellMeshingResult, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    diffusivity: float,
) -> None:
    from tools.meshing_qualification._physics import (
        _compiled_solver_stages,
        _solve_corpus_problem,
    )

    result, _ = material_interface_publication
    D = phx.discretization
    space = D.FiniteElementPlan(
        result.mesh,
        D.FiniteElementFieldSpec("u", D.lagrange_element("tetrahedron", 1)),
        coordinate_spec=result.geometry,
    ).prepare()
    constraint = D.dirichlet_constraint(space, "u")
    form = phx.equations.FiniteElementForm(
        "epoch-assembled-diffusion",
        "u",
        (
            phx.equations.DiffusionAction("u", diffusivity),
            phx.equations.SourceAction("u", 1.0),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint.constraint_map
    )
    seed = jnp.zeros((constraint.free_dofs.size,), dtype=jnp.float64)
    expected, expected_solve = _solve_corpus_problem(problem, seed)
    assert bool(np.asarray(expected_solve.successful))
    prepare = phx.equations.CompiledFiniteElementProblem.linear_system
    calls = 0

    def observe(
        current: phx.equations.CompiledFiniteElementProblem,
        args: object = None,
        /,
        *,
        nullspace_policy: phx.linalg.NullspacePolicy | None = None,
    ) -> tuple[phx.linalg.LinearSystem, object]:
        nonlocal calls
        calls += 1
        return prepare(current, args, nullspace_policy=nullspace_policy)

    monkeypatch.setattr(
        phx.equations.CompiledFiniteElementProblem, "linear_system", observe
    )
    actual, compiler, stages = _compiled_solver_stages(problem, seed, 3)
    assert calls == 1
    np.testing.assert_array_equal(actual, expected)
    assert compiler["assembly_binding_id"] == problem.compilation_id
    assert compiler["assembly_invocations"] == 1
    assert compiler["assembly_compilation_count"] == 1
    assert compiler["assembly_compile_seconds"] >= 0.0
    assert (
        compiler["execution_scope"]
        == "epoch-preassembled-system-residual-solve-and-expand"
    )
    assert compiler["assembly_seconds"] == stages["solver_assembly"]
    assert len(compiler["warm_samples_seconds"]) == 3
    assert compiler["solver"]["successful"]


def _graded_dirichlet_problem(
    diffusivity: float = 1.0,
) -> phx.equations.CompiledFiniteElementProblem:
    # Cubic x-grading makes cell aspect ratios, hence diagonal scales, vary strongly.
    count = 12
    x, y = np.meshgrid(
        np.linspace(0.0, 1.0, count + 1) ** 3, np.linspace(0.0, 1.0, count + 1)
    )
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    cells = []
    for j in range(count):
        for i in range(count):
            a, b = j * (count + 1) + i, j * (count + 1) + i + 1
            c, d = b + count + 1, a + count + 1
            cells.extend(((a, b, c), (a, c, d)))
    D = phx.discretization
    space = D.FiniteElementPlan(
        D.CellMesh.from_triangles(
            jnp.asarray(points), jnp.asarray(cells, dtype=jnp.int32)
        ),
        D.FiniteElementFieldSpec("u", D.lagrange_element("triangle", 1)),
    ).prepare()
    form = phx.equations.FiniteElementForm(
        "graded-exact-jacobi-diffusion",
        "u",
        (
            phx.equations.DiffusionAction("u", diffusivity),
            phx.equations.SourceAction("u", 1.0),
        ),
        # Positive scalar diffusion with full homogeneous Dirichlet data is coercive.
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
    )
    return phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=D.dirichlet_constraint(space, "u"),
        dirichlet_values=lambda points: jnp.zeros((points.shape[0],), dtype=jnp.float64),
    )


def test_exact_jacobi_prepares_once_and_preserves_the_residual_target() -> None:
    from tools.meshing_qualification._physics import (
        _compiled_solver_stages,
        _corpus_solve_policy,
        _prepare_corpus_epoch_system,
        _solve_corpus_problem,
        CorpusPreconditioning,
    )

    problem = _graded_dirichlet_problem()
    seed = jnp.zeros((problem.state_space.size,), dtype=jnp.float64)
    values, compiler, _ = _compiled_solver_stages(
        problem, seed, 2, preconditioning=CorpusPreconditioning.EXACT_JACOBI
    )
    solver = compiler["solver"]
    preconditioner = compiler["preconditioner"]
    assert preconditioner is not None
    assert compiler["preconditioning"] == "exact-jacobi"
    assert preconditioner["positive_definite"] == "verified"
    minimum_diagonal = preconditioner["minimum_diagonal"]
    assert minimum_diagonal is not None and minimum_diagonal > 0.0
    assert solver["method"] == "pcg"
    assert solver["preconditioner_id"] == preconditioner["preconditioner_id"]
    assert solver["preconditioner_refresh"] == "supplied"
    assert solver["successful"]
    assert compiler["assembly_compilation_count"] == 1
    assert compiler["first_execution_compilation_count"] == 0
    assert compiler["warm_recompilation_count"] == 0

    # The unchanged default tolerance targets the original system's true residual.
    system, rhs = problem.linear_system()
    assert problem.constraint is not None
    reduced = problem.constraint.reduce_vector(values)
    rhs_host = np.asarray(rhs)
    residual = rhs_host - np.asarray(system.operator.mv(reduced))
    tolerance = phx.linalg.TolerancePolicy()
    threshold = tolerance.absolute + tolerance.relative * np.linalg.norm(rhs_host)
    assert np.linalg.norm(residual) <= threshold

    # A cold consumer reuses the identical prepared action through its own solve.
    epoch, diagonal = _prepare_corpus_epoch_system(
        problem, CorpusPreconditioning.EXACT_JACOBI
    )
    policy, evidence = _corpus_solve_policy(
        epoch, diagonal, CorpusPreconditioning.EXACT_JACOBI, None
    )
    cold, cold_solve = _solve_corpus_problem(problem, seed, policy)
    assert evidence == preconditioner
    assert int(np.asarray(cold_solve.diagnostics.iterations)) == solver["iterations"]
    np.testing.assert_array_equal(cold, values)


def test_exact_jacobi_preserves_fully_constrained_values_without_fabricated_extrema() -> (
    None
):
    from tools.meshing_qualification._physics import (
        _compiled_solver_stages,
        _prepare_corpus_solver,
        CorpusPreconditioning,
    )

    mesh = phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        np.asarray(((0, 1, 2, 3),), dtype=np.int64),
    )
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    _, problem, free = _prepare_corpus_solver(result, material_x=False)
    assert free.size == 0
    values, compiler, _ = _compiled_solver_stages(
        problem,
        jnp.zeros((0,), dtype=jnp.float64),
        1,
        preconditioning=CorpusPreconditioning.EXACT_JACOBI,
    )
    preconditioner = compiler["preconditioner"]
    assert compiler["solver"]["successful"]
    assert preconditioner is not None
    assert preconditioner["size"] == 0
    assert preconditioner["minimum_diagonal"] is None
    assert preconditioner["maximum_diagonal"] is None
    np.testing.assert_array_equal(
        values, np.asarray((0.0, 1.0, 1.0, 1.0), dtype=np.float64)
    )


def test_exact_jacobi_refuses_invalid_or_unavailable_diagonals(
    material_interface_publication: tuple[phx.meshing.CellMeshingResult, np.ndarray],
) -> None:
    import equinox as eqx

    from tools.meshing_qualification._physics import (
        _compiled_solver_stages,
        _corpus_solve_policy,
        _prepare_corpus_epoch_system,
        CorpusPreconditioning,
    )
    from tools.meshing_qualification._runtime import QualificationFailure

    jacobi = CorpusPreconditioning.EXACT_JACOBI
    problem = _graded_dirichlet_problem()
    seed = jnp.zeros((problem.state_space.size,), dtype=jnp.float64)
    epoch, diagonal = _prepare_corpus_epoch_system(problem, jacobi)
    # The canonical linalg owner admits the diagonal; no local repair or masking.
    for invalid in (
        diagonal.diagonal.at[0].set(0.0),
        diagonal.diagonal.at[0].set(jnp.nan),
    ):
        changed = eqx.tree_at(lambda value: value.diagonal, diagonal, invalid)
        with pytest.raises(ValueError, match="finite and nonzero"):
            _corpus_solve_policy(epoch, changed, jacobi, None)
    negative = eqx.tree_at(lambda value: value.diagonal, diagonal, -diagonal.diagonal)
    with pytest.raises(QualificationFailure, match="positive-definite-claim-unverified"):
        _corpus_solve_policy(epoch, negative, jacobi, None)
    authored = phx.linalg.LinearSolvePolicy(
        preconditioning=phx.linalg.PreconditioningPolicy(
            phx.linalg.JacobiPreconditionerBuilder()
        )
    )
    with pytest.raises(ValueError, match="must not declare preconditioning"):
        _corpus_solve_policy(epoch, diagonal, jacobi, authored)
    limited = phx.linalg.LinearSolvePolicy(
        tolerance=phx.linalg.TolerancePolicy(max_steps=1)
    )
    with pytest.raises(QualificationFailure, match="native-solver-failure") as failed:
        _compiled_solver_stages(problem, seed, 1, policy=limited, preconditioning=jacobi)
    assert '"preconditioning": "exact-jacobi"' in str(failed.value)

    # Region-restricted material diffusion has no structural diagonal: no fallback.
    result, _ = material_interface_publication
    _, material, free = _prepare_corpus_solver(result)
    with pytest.raises(ValueError, match="Exact diagonal lowering is unavailable"):
        _compiled_solver_stages(
            material,
            jnp.zeros((free.size,), dtype=jnp.float64),
            1,
            preconditioning=jacobi,
        )


@pytest.mark.parametrize(
    ("profile", "expected_density", "expected_site"),
    (
        ("unit-box", (2.0,), "fluid-density"),
        ("concave-box-core-lifecycle", (2.0,), "fluid-density"),
        ("curved-periodic-layer-core", (2.0, 3.0), "two-material-density"),
        ("archived-layer-core", (2.0, 3.0), "two-material-density"),
    ),
)
def test_hybrid_omitted_material_controls_resolve_by_authored_profile(
    profile: str, expected_density: tuple[float, ...], expected_site: str
) -> None:
    from tools.meshing_qualification._controls import _hybrid_material_controls

    assert _hybrid_material_controls(profile, None, None) == (
        expected_density,
        expected_site,
    )
    explicit = tuple(7.0 + index for index in range(len(expected_density)))
    assert _hybrid_material_controls(profile, explicit, "explicit-material") == (
        explicit,
        "explicit-material",
    )
    wrong = (2.0, 3.0) if len(expected_density) == 1 else (2.0,)
    with pytest.raises(ValueError, match="source-region values"):
        _hybrid_material_controls(profile, wrong, None)


@pytest.mark.parametrize(
    ("scenario", "profile", "expected"),
    (
        ("native-corpus", "unit-box", (None, None)),
        ("native-hybrid-lifecycle", "unit-box", ((2.0,), "fluid-density")),
        (
            "native-hybrid-lifecycle",
            "curved-periodic-layer-core",
            ((2.0, 3.0), "two-material-density"),
        ),
    ),
)
def test_hybrid_cli_material_sentinel_waits_for_actual_profile(
    scenario: str, profile: str, expected: tuple[object, object]
) -> None:
    import argparse

    namespace = argparse.Namespace(
        **{
            **QualificationControls().to_record(),
            "scenario": scenario,
            "hybrid_profile": profile,
            "distributed_bisection_compatibility": (
                QualificationControls().distributed_bisection_compatibility
            ),
        }
    )
    controls = QualificationControls.from_namespace(namespace)
    assert (controls.hybrid_region_densities, controls.hybrid_material_site) == expected
    explicit = argparse.Namespace(
        **{
            **vars(namespace),
            "hybrid_region_densities": [7.0] if profile == "unit-box" else [7.0, 11.0],
            "hybrid_material_site": "explicit-material",
        }
    )
    controls = QualificationControls.from_namespace(explicit)
    assert controls.hybrid_region_densities == tuple(explicit.hybrid_region_densities)
    assert controls.hybrid_material_site == "explicit-material"


@pytest.fixture(scope="module")
def actual_resource_publication() -> tuple[dict[str, Any], dict[str, Any]]:
    """One real request, not a synthetic status/receipt positive."""
    from phydrax._meshcore import NativeParallelExecution
    from phydrax.meshing.qualification._layer_continuation import (
        _published_result_identity,
    )
    from tools._meshing_cases import prepare_polyhedral_case
    from tools.meshing_qualification._records import _result_identity
    from tools.meshing_qualification._runtime import _runtime_identity

    source, specification, options = prepare_polyhedral_case(
        "polyhedral-periodic-cube",
        4,
        20000,
        120.0,
        source=True,
    )
    with NativeParallelExecution(None, specification.limits.maximum_wall_seconds):
        result = (
            phx.meshing.NativeMeshingProvider(options)
            .plan(
                source,
                specification,
                coordinate_contract=phx.SpatialCoordinateContract.si(),
            )
            .execute()
        )
    publication = _result_identity(result)
    # Both manual and tool serializers use the same actual owner association.
    assert (
        _published_result_identity(result)["resource_binding"]
        == publication["resource_binding"]
    )
    return cast(dict[str, Any], publication), cast(dict[str, Any], _runtime_identity())


def _assert_actual_admission_observation(assessment: Mapping[str, Any]) -> list[str]:
    """A real missing envelope stays unproved; never synthesize a positive case."""
    from phydrax.meshing._measurements import validate_native_execution_interchange

    assert assessment["publications"]
    expected = []
    for index, publication in enumerate(assessment["publications"]):
        observation = validate_native_execution_interchange(
            publication["execution_evidence"]
        )
        if not observation["preparation_complete"]:
            expected.append(
                f"publication-{index}:incomplete-native-preparation-receipt"
            )
        if observation["managed_admission_peak_bytes_upper"] is None:
            expected.append(f"publication-{index}:unobserved-managed-admission-envelope")
    assert assessment["admission_gaps"] == expected, assessment["admission_gaps"]
    if expected:
        assert assessment["admission_coverage"] == "unproved"
    else:
        assert assessment["admission_coverage"] == "proved"
    return expected


def _actual_resource_sample(
    publication: dict[str, Any],
    runtime: dict[str, Any],
    *,
    missing: str | None = None,
    device_shape: Literal["standard", "statistics", "transport"] = "standard",
    device_override: dict[str, Any] | None = None,
    observed_epoch: dict[str, Any] | None = None,
    duplicate_device: bool = False,
) -> CampaignSample:
    """Wrap real observations; intentionally missing observations remain missing."""
    import jax

    from tools.meshing_qualification._campaign import _complete_campaign_sample
    from tools.meshing_qualification._records import _finalize_mandatory_cli_record
    from tools.meshing_qualification._runtime import (
        peak_resident_bytes,
        PEAK_RESIDENT_SCOPE,
    )

    devices = []
    for device in jax.local_devices():
        stats = device.memory_stats()
        devices.append(
            {
                "device_id": device.id,
                "platform": device.platform,
                "device_kind": device.device_kind,
                "peak_bytes": None if stats is None else stats.get("peak_bytes_in_use"),
                "memory_statistics": stats,
                "status": "unavailable" if stats is None else "backend-reported",
            }
        )
        if device_shape == "statistics":
            devices[-1] = {
                "device_id": device.id,
                "status": "unavailable" if stats is None else "backend-reported",
                "statistics": stats,
            }
        elif device_shape == "transport":
            devices[-1] = {
                "device": str(device),
                "peak_bytes": None if stats is None else stats.get("peak_bytes_in_use"),
                "unavailable_reason": (
                    "Backend does not expose device allocation peak."
                    if stats is None or "peak_bytes_in_use" not in stats
                    else None
                ),
            }
    if device_override is not None:
        devices[0] = device_override
    if duplicate_device:
        devices.append(dict(devices[0]))
    controls = QualificationControls(
        scenario="native-polyhedral-lifecycle",
        polyhedral_source="polyhedral-periodic-cube",
        resolution=4,
    )
    record = {
        "status": "passed",
        # This fixture proves resource coverage only, never whole science/workflow.
        "independent_scientific_certification": "unassessed",
        "mandatory_workflow_completion": "incomplete",
        "case_contract_completion": "incomplete",
        "request_id": publication["resource_binding"]["request_id"],
        "generation": publication,
        "runtime": runtime,
        "peak_process_resident_bytes": None
        if missing == "process"
        else peak_resident_bytes(),
        "process_peak_scope": PEAK_RESIDENT_SCOPE,
        "device_memory": [] if missing == "device" else devices,
        "epochs": (
            [{"epoch": 0, "result": publication, "compiler": {}}]
            if missing == "compiler"
            else []
            if observed_epoch is None
            else [observed_epoch]
        ),
    }
    child = _finalize_mandatory_cli_record(controls, record)
    verdict = child["workflow_result"]
    return _complete_campaign_sample(
        {**child, "child_evidence": child},
        20000,
        positive=False,
        accepted=False,
        verdict=verdict,
    )


def test_actual_publication_resource_assessment_separates_admission_from_measurements(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_profile_assessment,
        _corpus_resource_assessment,
    )

    publication, runtime = actual_resource_publication
    sample = _actual_resource_sample(publication, runtime)
    assessment = _corpus_resource_assessment(sample, full_workflow=False, refusal=False)
    _assert_actual_admission_observation(assessment)
    assert assessment["publications"][0] == publication
    assert (
        "scratch_bytes:unmanaged_host_device_compiler"
        in (publication["resource_guarantees"]["unenforced"])
    )
    binding = publication["resource_binding"]
    assert binding["native_managed_claim"] == {
        "scope": "native-managed-request",
        **publication["resource_guarantees"],
    }
    assert (
        "device_allocation_peak_bytes"
        in binding["outside_admission_measurement_requirements"]
    )
    assert (
        "process_resident_peak_bytes"
        in binding["outside_admission_measurement_requirements"]
    )
    assert any(
        "outside-admission:scratch_bytes:" in item for item in assessment["limitations"]
    )
    # On CPU, unavailable backend allocator statistics are not fabricated zeroes.
    if any(
        device["peak_bytes"] is None
        for device in sample.resource_evidence["device_memory"]
    ):
        assert (
            assessment["runtime_measurement_availability"]
            == "complete-with-cpu-device-peak-limitation"
        )
        assert assessment["runtime_measurement_gaps"] == []
        assert assessment["approved_measurement_limitations"]
        assert assessment["approved_measurement_limitations"][0]["peak_bytes"] is None
    scientific = _corpus_profile_assessment(
        {"expected_outcome": "passed", "full_workflow": False},
        sample,
    )
    assert scientific["case_certified"] is False
    assert scientific["positive_completion"] is False


@pytest.mark.parametrize(
    ("defect", "expected"),
    (
        ("missing-axis", "missing-geometry_queries"),
        ("unknown-scope", "unenforced-scratch_bytes:unreviewed_host_scope"),
        ("unmetered-host", "unenforced-work_units:unmetered_host_device_compiler"),
        ("unobserved-envelope", "unobserved-managed-admission-envelope"),
        ("partial-scope", "partial-work_units:native_and_metered_host"),
        ("coordinate-scope", "partial-work_units:native_and_coordinate_batches"),
        ("incomplete-receipt", "invalid-native-receipt"),
        ("foreign-receipt", "actual-publication-resource-binding"),
        ("missing-receipt", "actual-ended-native-receipt"),
        ("missing-binding", "actual-publication-resource-binding"),
        ("build-drift", "publication-native-build-binding"),
        ("claim-drift", "publication-resource-scope-binding"),
        ("runtime-requirement-drift", "publication-resource-scope-binding"),
    ),
)
def test_actual_resource_assessment_keeps_native_gaps_blocking(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
    defect: str,
    expected: str,
) -> None:
    from copy import deepcopy

    from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
    from phydrax.meshing._measurements import NativeExecutionRecord
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_resource_assessment,
    )

    original, runtime = actual_resource_publication
    publication = deepcopy(original)
    if defect == "missing-axis":
        publication["resource_guarantees"]["enforced"].remove("geometry_queries")
    elif defect == "unknown-scope":
        publication["resource_guarantees"]["unenforced"].append(
            "scratch_bytes:unreviewed_host_scope"
        )
    elif defect == "unmetered-host":
        publication["resource_guarantees"]["unenforced"].append(
            "work_units:unmetered_host_device_compiler"
        )
    elif defect in ("partial-scope", "coordinate-scope"):
        publication["resource_guarantees"]["enforced"].remove("work_units")
        publication["resource_guarantees"]["enforced"].append(
            "work_units:native_and_metered_host" if defect == "partial-scope"
            else "work_units:native_and_coordinate_batches"
        )
    elif defect == "incomplete-receipt":
        publication["execution_evidence"]["phases"].pop()
    elif defect in ("foreign-receipt", "unobserved-envelope"):
        with NativeExecutionScope(
            NativeExecutionLimits(10, 10, 10, 100_000, 120.0)
        ) as scope:
            if defect == "foreign-receipt":
                scope.observe_managed_admission()
            retained = scope.allocate_host_array((8,), np.float64)
            retained[:] = 1.0
        foreign = NativeExecutionRecord(
            scope.completed_evidence(), owner_id="foreign-real-scope"
        )
        foreign.require_valid()
        publication["execution_evidence"] = foreign.to_record()
    elif defect == "missing-receipt":
        publication["execution_evidence"] = None
    elif defect == "missing-binding":
        sample = _actual_resource_sample(publication, runtime)
        del publication["resource_binding"]
        assessment = _corpus_resource_assessment(
            sample, full_workflow=False, refusal=False
        )
        assert assessment["admission_coverage"] == "unproved"
        assert any(expected in item for item in assessment["admission_gaps"])
        return
    else:
        binding = publication["resource_binding"]
        if defect == "build-drift":
            binding["native_build"]["binary_digest"] = "f" * 64
        elif defect == "claim-drift":
            binding["native_managed_claim"]["enforced"].remove("scratch_bytes")
        else:
            binding["outside_admission_measurement_requirements"].remove(
                "device_allocation_peak_bytes"
            )
        binding["binding_id"] = phx.canonical_fingerprint(
            {key: value for key, value in binding.items() if key != "binding_id"}
        )
    sample = _actual_resource_sample(publication, runtime)
    assessment = _corpus_resource_assessment(sample, full_workflow=False, refusal=False)
    assert assessment["admission_coverage"] == "unproved"
    assert any(expected in item for item in assessment["admission_gaps"])

def test_canonical_connectivity_does_not_supply_resource_limits(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    from copy import deepcopy

    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_resource_assessment,
        _REQUEST_LIMITS,
    )

    original, runtime = actual_resource_publication
    publication = deepcopy(original)
    publication["resource_guarantees"]["enforced"] = ["canonical_connectivity"]
    assessment = _corpus_resource_assessment(
        _actual_resource_sample(publication, runtime), full_workflow=False, refusal=False
    )
    gaps = assessment["admission_gaps"]
    assert assessment["admission_coverage"] == "unproved"
    assert all(f"publication-0:missing-{name}" in gaps for name in _REQUEST_LIMITS)
    assert not any("unknown-enforced-scope-canonical_connectivity" in gap for gap in gaps)



@pytest.mark.parametrize("missing", ("process", "device", "compiler"))
def test_actual_resource_assessment_missing_runtime_measurement_is_not_admission(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
    missing: str,
) -> None:
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_resource_assessment,
    )

    publication, runtime = actual_resource_publication
    assessment = _corpus_resource_assessment(
        _actual_resource_sample(publication, runtime, missing=missing),
        full_workflow=False,
        refusal=False,
    )
    _assert_actual_admission_observation(assessment)
    assert assessment["runtime_measurement_availability"] == "incomplete"
    assert any(missing in item for item in assessment["runtime_measurement_gaps"])


def test_actual_resource_assessment_rejects_sample_build_drift(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    from copy import deepcopy

    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_resource_assessment,
    )

    publication, runtime = actual_resource_publication
    foreign_runtime = deepcopy(runtime)
    foreign_runtime["build_id"] = "foreign-build"
    assessment = _corpus_resource_assessment(
        _actual_resource_sample(publication, foreign_runtime),
        full_workflow=False,
        refusal=False,
    )
    assert "actual-child-build-binding" in assessment["admission_gaps"]


@pytest.mark.parametrize("device_shape", ("standard", "statistics", "transport"))
def test_actual_resource_assessment_preserves_owner_device_measurement_shapes(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
    device_shape: Literal["standard", "statistics", "transport"],
) -> None:
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_resource_assessment,
    )

    publication, runtime = actual_resource_publication
    sample = _actual_resource_sample(publication, runtime, device_shape=device_shape)
    assessment = _corpus_resource_assessment(sample, full_workflow=False, refusal=False)
    _assert_actual_admission_observation(assessment)
    if any(
        device.get("status") == "unavailable"
        or device.get("unavailable_reason") is not None
        for device in sample.resource_evidence["device_memory"]
    ):
        assert (
            assessment["runtime_measurement_availability"]
            == "complete-with-cpu-device-peak-limitation"
        )
        assert assessment["runtime_measurement_gaps"] == []
        assert assessment["approved_measurement_limitations"][0]["peak_bytes"] is None


@pytest.fixture(scope="module")
def actual_cpu_compiler_epoch(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Actual compiler/host/logical measurements, never a scientific solver claim."""
    from time import perf_counter

    import jax

    from phydrax._execution_sampling import compiled_memory_estimate, sample_host_memory

    publication, _ = actual_resource_publication
    values = jnp.asarray(
        (publication["vertices"], publication["cells"]), dtype=jnp.float64
    )
    executable = jax.jit(lambda x: jnp.sum(x * x)).lower(values).compile()
    started = perf_counter()
    output = executable(values).block_until_ready()
    elapsed = perf_counter() - started
    estimate = compiled_memory_estimate(executable)
    assert estimate is not None
    return {
        "epoch": 0,
        "result": publication,
        "compiler": {
            **estimate.to_payload(),
            "warm_samples_seconds": [elapsed],
            "memory_status": "measured-compiler-analysis",
            "execution_scope": "actual CPU resource-measurement probe; not a physical/PDE solver",
        },
        "retained_bytes": phx.model_array_bytes((values, output)),
        "retained_bytes_scope": "actual probe input/output arrays, not a full physical epoch",
        "host_process_measurement": sample_host_memory().to_payload(),
    }


def test_cpu_peak_disposition_keeps_real_record_and_discloses_in_reports(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
    actual_cpu_compiler_epoch: dict[str, Any],
    tmp_path: Path,
) -> None:
    from tools._meshing_cases import polyhedral_identity
    from tools.meshing_qualification._records import _mandatory_result
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_profile_assessment,
        _corpus_workflow_results,
    )

    publication, runtime = actual_resource_publication
    sample = _actual_resource_sample(
        publication, runtime, observed_epoch=actual_cpu_compiler_epoch
    )
    name = "polyhedral-periodic-cube"
    profile = {
        "case_name": name,
        "workflow": "polyhedral-system",
        "scenario": None,
        "controls": {},
        "expected_outcome": "passed",
        "full_workflow": True,
        "resolution_scale": 1,
        "minimum_adaptation_rounds": 0,
        "source_reference": polyhedral_identity(name),
    }
    assessment = _corpus_profile_assessment(profile, sample)
    resources = assessment["resource_assessment"]
    expected_admission = _assert_actual_admission_observation(resources)
    assert (
        resources["runtime_measurement_availability"]
        == "complete-with-cpu-device-peak-limitation"
    )
    assert resources["runtime_measurement_gaps"] == []
    assert assessment["resource_evidence_gaps"] == expected_admission
    assert assessment["phase_measurement_gaps"] == []
    limitation = assessment["qualification_limitations"][0]
    assert limitation["availability"] == "unavailable"
    assert limitation["peak_bytes"] is None
    assert (
        limitation["device_measurement"] == sample.resource_evidence["device_memory"][0]
    )
    assert limitation["verified_capability"]["probe_evidence_sha256"]
    assert limitation["runtime_id"] == sample.binding["runtime_id"]
    assert limitation["reason"]
    # Resource availability never substitutes for actual science or cold acceptance.
    assert assessment["qualification_eligible"] is False
    assert assessment["positive_completion"] is False
    result = _mandatory_result(name, profile, assessment)
    assert result["qualification_limitations"] == assessment["qualification_limitations"]
    workflows = _corpus_workflow_results({name: profile}, {name: result})
    assert (
        workflows[profile["workflow"]]["qualification_limitations"][name]
        == assessment["qualification_limitations"]
    )
    (tmp_path / "cpu-resource-disposition.json").write_text(
        json.dumps(
            {
                "sample": sample.to_record(),
                "assessment": assessment,
                "mandatory_result": result,
                "workflow_results": workflows,
                "scope": "actual resource probe only; no scientific/full-workflow/corpus positive claim",
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "defect",
    (
        "non-cpu",
        "unknown-backend",
        "missing-capability",
        "foreign-capability",
        "changed-provider",
        "missing-peak-field",
        "missing-unsupported-observation",
        "missing-status",
        "missing-device-record",
        "duplicate-device-record",
        "zero-substitution",
        "rss-substitution",
        "snapshot-substitution",
        "compiler-substitution",
    ),
)
def test_cpu_peak_disposition_rejects_unknown_missing_or_substituted_measurements(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
    actual_cpu_compiler_epoch: dict[str, Any],
    defect: str,
) -> None:
    from copy import deepcopy

    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_resource_assessment,
    )

    publication, original_runtime = actual_resource_publication
    runtime = deepcopy(original_runtime)
    original = _actual_resource_sample(publication, runtime)
    device = deepcopy(original.resource_evidence["device_memory"][0])
    environment = runtime["environment"]
    entry = environment["devices"][0]
    if defect in ("non-cpu", "unknown-backend"):
        backend = "cuda" if defect == "non-cpu" else "unverified-provider"
        runtime["backend"] = backend
        entry["platform"] = backend
        device["platform"] = backend
    elif defect == "missing-capability":
        del entry["device_allocator_peak_capability"]
    elif defect == "foreign-capability":
        capability = entry["device_allocator_peak_capability"]
        capability["probe_evidence_sha256"] = "f" * 64
        capability["capability_id"] = phx.canonical_fingerprint(
            {key: value for key, value in capability.items() if key != "capability_id"}
        )
    elif defect == "changed-provider":
        environment["jax_release"] = "unverified-release"
    elif defect == "missing-peak-field":
        del device["peak_bytes"]
    elif defect == "missing-unsupported-observation":
        del device["memory_statistics"]
    elif defect == "missing-status":
        del device["status"]
    elif defect.endswith("-substitution"):
        device["status"] = "backend-reported"
        device["peak_bytes"] = (
            0
            if defect == "zero-substitution"
            else original.peak_process_resident_bytes
            if defect == "rss-substitution"
            else actual_cpu_compiler_epoch["retained_bytes"]
            if defect == "snapshot-substitution"
            else actual_cpu_compiler_epoch["compiler"]["temporary_bytes"]
        )
    runtime["environment_id"] = phx.canonical_fingerprint(environment)
    sample = _actual_resource_sample(
        publication,
        runtime,
        missing="device" if defect == "missing-device-record" else None,
        device_override=device,
        observed_epoch=actual_cpu_compiler_epoch,
        duplicate_device=defect == "duplicate-device-record",
    )
    assessment = _corpus_resource_assessment(sample, full_workflow=False, refusal=False)
    assert assessment["runtime_measurement_availability"] == "incomplete"
    assert assessment["runtime_measurement_gaps"]


@pytest.mark.parametrize(
    "missing", ("process", "compiler", "logical", "host", "native-axis")
)
def test_cpu_peak_disposition_does_not_exempt_other_hard_requirements(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
    actual_cpu_compiler_epoch: dict[str, Any],
    missing: str,
) -> None:
    from copy import deepcopy

    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_profile_assessment,
    )

    original, runtime = actual_resource_publication
    publication = deepcopy(original)
    epoch = deepcopy(actual_cpu_compiler_epoch)
    if missing == "compiler":
        epoch["compiler"]["temporary_bytes"] = None
    elif missing == "logical":
        epoch["retained_bytes"] = None
    elif missing == "host":
        del publication["execution_evidence"]["phases"][0][
            "host_storage_peak_bytes_upper"
        ]
    elif missing == "native-axis":
        publication["resource_guarantees"]["enforced"].remove("work_units")
    epoch["result"] = publication
    assessment = _corpus_profile_assessment(
        {"expected_outcome": "passed", "full_workflow": False},
        _actual_resource_sample(
            publication,
            runtime,
            observed_epoch=epoch,
            missing="process" if missing == "process" else None,
        ),
    )
    assert assessment["qualification_eligible"] is False
    assert assessment["resource_evidence_gaps"] or assessment["phase_measurement_gaps"]


def _compiler_consumer_sample(payload: dict[str, Any]) -> CampaignSample:
    """Phase-only diagnostic input; never claim accepted scientific epochs."""
    from tools.meshing_qualification._campaign import _complete_campaign_sample

    return _complete_campaign_sample(
        {
            "status": "diagnostic-unassessed",
            "mandatory_workflow_completion": "incomplete",
            "case_contract_completion": "incomplete",
            "independent_scientific_certification": "unassessed",
            **payload,
        },
        20000,
        positive=False,
        accepted=False,
    )


def _unmeasured_polyhedral_phase_sample(
    *,
    periodic: bool,
    **payload: Any,
) -> CampaignSample:
    from tools._meshing_cases import polyhedral_case

    case = "polyhedral-periodic-cube" if periodic else "polyhedral-material-L"
    controls = QualificationControls(
        scenario="native-polyhedral-lifecycle",
        polyhedral_source=case,
        adaptation_rounds=1,
    )
    return _compiler_consumer_sample(
        {
            "scenario": "native-polyhedral-lifecycle",
            "case": polyhedral_case(case).to_record(),
            "qualification_controls": controls.to_record(),
            **payload,
        }
    )


def _bound_polyhedral_generation_diagnostic(
    publication: dict[str, Any],
    runtime: dict[str, Any],
    **payload: Any,
) -> CampaignSample:
    """Bind actual generation evidence; no accepted scientific warm epoch is invented."""
    from tools._meshing_cases import polyhedral_case
    from tools.meshing_qualification._campaign import _complete_campaign_sample
    from tools.meshing_qualification._records import _finalize_mandatory_cli_record

    template = _actual_resource_sample(publication, runtime).to_record()["child_evidence"]
    body = {key: value for key, value in template.items() if key != "workflow_result"}
    body.update(
        scenario="native-polyhedral-lifecycle",
        case=polyhedral_case("polyhedral-periodic-cube").to_record(),
        **payload,
    )
    controls = QualificationControls(
        scenario="native-polyhedral-lifecycle",
        polyhedral_source="polyhedral-periodic-cube",
        resolution=4,
    )
    child = _finalize_mandatory_cli_record(controls, body)
    return _complete_campaign_sample(
        {**child, "child_evidence": child},
        20000,
        positive=False,
        accepted=False,
        verdict=child["workflow_result"],
    )


def test_polyhedral_phase_consumer_keeps_ordinary_actual_measurements(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
    actual_cpu_compiler_epoch: dict[str, Any],
) -> None:
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_compiler_phases,
        _corpus_phase_gaps,
    )

    publication, runtime = actual_resource_publication
    sample = _actual_resource_sample(
        publication, runtime, observed_epoch=actual_cpu_compiler_epoch
    )
    records, gaps = _corpus_compiler_phases(sample)
    assert gaps == []
    assert len(records) == 1 and records[0]["solver"] == "epoch"
    assert records[0]["evidence_path"] == "epochs[0].compiler"
    assert (
        records[0]["temporary_bytes"]
        == actual_cpu_compiler_epoch["compiler"]["temporary_bytes"]
    )
    assert (
        records[0]["logical_retained_bytes"]
        == actual_cpu_compiler_epoch["retained_bytes"]
    )
    assert _corpus_phase_gaps(sample, full_workflow=True, refusal=False) == []


def test_polyhedral_phase_consumer_material_requires_nonvacuous_warm_and_cold() -> None:
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_compiler_phases,
        _corpus_phase_gaps,
    )

    sample = _unmeasured_polyhedral_phase_sample(periodic=False)
    records, binding = _corpus_compiler_phases(sample)
    assert binding == []
    assert len(records) == 16
    assert {(item["phase"], item["epoch"], item["solver"]) for item in records} == {
        (phase, epoch, solver)
        for phase in ("warm", "cold")
        for epoch in (0, 1)
        for solver in ("fe", "vem", "fv", "fv-transferred-update")
    }
    assert all(item["epoch_id"] is None for item in records)
    assert "accepted_epochs" not in sample.to_record()
    assert sample.phase_evidence["status"] == "unmeasured"
    gaps = _corpus_phase_gaps(sample, full_workflow=True, refusal=False)
    for item in records:
        assert f"{item['evidence_path']}:temporary_bytes" in gaps
        assert f"{item['evidence_path']}:logical_retained_bytes" in gaps
        assert f"{item['evidence_path']}:lowering_seconds" in gaps


@pytest.mark.parametrize("count", (0, 1, 3))
def test_polyhedral_phase_consumer_periodic_keeps_warm_history_and_missing_cold(
    count: int,
) -> None:
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_compiler_phases,
        _corpus_phase_gaps,
    )

    # These are explicitly unmeasured physical records, not accepted epochs.
    physical: list[dict[str, Any]] = [{} for _ in range(count)]
    if physical:
        physical[-1]["fv"] = {
            "trace_repreparation": {
                "current_and_history": [
                    {"state": "current", "compiler": None},
                    {"state": "history-0", "compiler": None},
                ]
            },
        }
    sample = _unmeasured_polyhedral_phase_sample(periodic=True, physics=physical)
    records, binding = _corpus_compiler_phases(sample)
    assert binding == []
    cold = [item for item in records if item["phase"] == "cold"]
    assert len(cold) == 3 * max(count, 1)
    assert {item["solver"] for item in cold} == {"fe", "vem", "fv"}
    assert all(item["epoch_id"] is None for item in records)
    trace = [item for item in records if item["solver"].startswith("fv-trace:")]
    assert [item["solver"] for item in trace] == (
        ["fv-trace:current", "fv-trace:history-0"] if count else []
    )
    assert all(".current_and_history[" in item["evidence_path"] for item in trace)
    gaps = _corpus_phase_gaps(sample, full_workflow=True, refusal=False)
    assert all(f"{item['evidence_path']}:warm_samples_seconds" in gaps for item in cold)
    assert "accepted_epochs" not in sample.to_record()


def test_polyhedral_phase_consumer_never_borrows_ordinary_or_global_measurements(
    actual_cpu_compiler_epoch: dict[str, Any],
) -> None:
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_compiler_phases,
        _corpus_phase_gaps,
    )

    sample = _unmeasured_polyhedral_phase_sample(
        periodic=False,
        epochs=[actual_cpu_compiler_epoch],
        logical_retained_bytes=actual_cpu_compiler_epoch["retained_bytes"],
    )
    records, binding = _corpus_compiler_phases(sample)
    assert binding == []
    physical = [item for item in records if item["solver"] != "epoch"]
    assert physical and all(item["logical_retained_bytes"] is None for item in physical)
    assert all(item["temporary_bytes"] is None for item in physical)
    gaps = _corpus_phase_gaps(sample, full_workflow=True, refusal=False)
    assert "source_physics.fe.compiler:logical_retained_bytes" in gaps
    assert (
        "cold.fresh_interpreter.continued_epochs[1].physics.vem.compiler:temporary_bytes"
        in gaps
    )


@pytest.mark.parametrize(
    "defect",
    (
        "drop",
        "foreign-path",
        "foreign-solver",
        "foreign-epoch",
        "negative-clock",
        "bad-capacity",
        "foreign-capacity",
    ),
)
def test_polyhedral_phase_consumer_rejects_flat_measurements_not_bound_to_source(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
    actual_cpu_compiler_epoch: dict[str, Any],
    defect: str,
) -> None:
    from copy import deepcopy

    from tools.meshing_qualification._campaign import _decode_campaign_sample
    from tools.meshing_qualification.scenarios.native_corpus import _corpus_phase_gaps

    publication, runtime = actual_resource_publication
    sample = _actual_resource_sample(
        publication, runtime, observed_epoch=actual_cpu_compiler_epoch
    )
    record = deepcopy(sample.to_record())
    phases = record["phase_evidence"]["compiler_epochs"]
    if defect == "drop":
        phases.clear()
    elif defect == "foreign-path":
        phases[0]["evidence_path"] = (
            "cold.fresh_interpreter.continued_epochs[0].physics.fe.compiler"
        )
    elif defect == "foreign-solver":
        phases[0]["solver"] = "fv"
    elif defect == "foreign-epoch":
        phases[0]["epoch_id"] = "unissued-epoch"
    elif defect == "negative-clock":
        phases[0]["warm_samples_seconds"] = [-1.0]
    elif defect == "foreign-capacity":
        record["phase_evidence"]["controlling_entity_capacity"] = 10000
        phases[0]["controlling_entity_capacity"] = 10000
    else:
        record["phase_evidence"]["controlling_entity_capacity"] = True
    gaps = _corpus_phase_gaps(
        _decode_campaign_sample(record), full_workflow=True, refusal=False
    )
    assert gaps
    assert any("compiler-phase" in gap for gap in gaps)


@pytest.mark.parametrize(
    "defect", ("case-controls", "accepted-epochs", "physics", "cold", "compiler")
)
def test_polyhedral_phase_consumer_fails_closed_on_malformed_owning_payload(
    defect: str,
) -> None:
    from copy import deepcopy

    from tools.meshing_qualification._campaign import _decode_campaign_sample
    from tools.meshing_qualification.scenarios.native_corpus import _corpus_phase_gaps

    record = deepcopy(_unmeasured_polyhedral_phase_sample(periodic=True).to_record())
    if defect == "case-controls":
        record["case"]["controls"] = "periodic-quotient"
    elif defect == "accepted-epochs":
        record["accepted_epochs"] = [None]
    elif defect == "physics":
        record["physics"] = {"fe": None}
    elif defect == "cold":
        record["cold"] = []
    else:
        record["physics"] = [{"fe": {"compiler": []}}]
    gaps = _corpus_phase_gaps(
        _decode_campaign_sample(record), full_workflow=True, refusal=False
    )
    assert any("source-shape" in gap for gap in gaps)


_OVERSET_PUBLICATION_TEMPLATES: dict[str, dict[str, Any]] = {}


def _overset_publication_template(profile_name: str) -> dict[str, Any]:
    cached = _OVERSET_PUBLICATION_TEMPLATES.get(profile_name)
    if cached is not None:
        return cached
    from phydrax._meshcore import current_native_execution_scope, NativeParallelExecution
    from phydrax.meshing._native_preparation import native_preparation_ledger
    from phydrax.meshing.qualification import (
        deform_planar_interior,
        execute_native_consumer,
        execute_native_request,
        rigid_planar_successor,
    )
    from tools.meshing_qualification._controls import (
        _overset_qualification_requirements,
    )
    from tools.meshing_qualification._records import _result_identity
    from tools.meshing_qualification._runtime import qualification_failure
    from tools.meshing_qualification.scenarios.overset import (
        _BACKGROUND,
        _OVERSET_PROFILES,
        _overset_publication_lineage,
        _overset_qualification_source,
        _OversetBody,
        _part_carrier,
        _SCHWARZ_MAXIMUM_ITERATIONS,
    )

    profile = _OVERSET_PROFILES.get(profile_name)
    requirements = _overset_qualification_requirements()
    geometry = requirements.get("geometry")
    profiles = geometry.get("profiles") if isinstance(geometry, dict) else None
    authored_profile = (
        profiles.get(profile_name) if isinstance(profiles, dict) else None
    )
    if (
        profile is None
        or not isinstance(authored_profile, dict)
        or authored_profile != profile.record()
    ):
        raise ValueError(f"Unknown or unauthenticated overset profile {profile_name!r}.")
    work_units = 1_280_000 if profile_name == "routine" else 2_000_000
    controls = QualificationControls(
        scenario="native-overset-lifecycle",
        resolution=4,
        capacity=20000,
        timeout=180.0,
        repeats=3,
        target_error=0.05,
        adaptation_rounds=1,
        maximum_work_units=work_units,
        maximum_geometry_queries=2_560_000,
        maximum_scratch_bytes=128_000_000,
        overset_motion_steps=2,
    ).to_record()
    controls["overset_profile"] = profile_name

    def generate_part(
        name: str, body: _OversetBody | None
    ) -> tuple[phx.meshing.MeshPart, phx.meshing.NativeMeshingPlan]:
        source, specification, options, _scopes = _overset_qualification_source(
            name,
            4,
            20000,
            180.0,
            body,
            maximum_work_units=work_units,
            maximum_geometry_queries=2_560_000,
            maximum_scratch_bytes=128_000_000,
        )
        plan = phx.meshing.NativeMeshingProvider(options).plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        return phx.meshing.MeshPart(name, plan.execute()), plan

    def part_record(
        part: phx.meshing.MeshPart,
        plan: phx.meshing.NativeMeshingPlan,
        body: _OversetBody | None,
    ) -> dict[str, Any]:
        source = plan.source
        if not isinstance(source, phx.meshing.NativePlanarSource):
            raise TypeError("Overset publication fixture requires an actual planar source.")
        carrier = _part_carrier(part)
        return {
            "source_id": source.source_id,
            "source_revision": source.source_revision,
            "source_binding_id": source.binding_id,
            "request_id": plan.specification.specification_id,
            "limits_id": plan.specification.limits.limits_id,
            "prepared_id": plan.plan_id,
            "authored_body": None if body is None else body.record(),
            "generation": _result_identity(carrier),
        }

    parts: dict[str, dict[str, Any]] = {}
    recipes: dict[str, dict[str, str]] = {}
    initial_parts: dict[str, phx.meshing.MeshPart] = {}
    plans: dict[str, phx.meshing.NativeMeshingPlan] = {}
    with NativeParallelExecution(None, 180.0):
        background_part, background_plan = generate_part(_BACKGROUND, None)
        initial_parts[_BACKGROUND] = background_part
        plans[_BACKGROUND] = background_plan
        parts[_BACKGROUND] = part_record(background_part, background_plan, None)
        for body in profile.bodies:
            part, plan = generate_part(body.name, body)
            initial_parts[body.name] = part
            plans[body.name] = plan
            parts[body.name] = part_record(part, plan, body)
        recipes = {
            name: {
                "source_binding_id": part["source_binding_id"],
                "request_id": part["request_id"],
                "plan_id": part["prepared_id"],
            }
            for name, part in parts.items()
        }

        epochs: list[dict[str, Any]] = [{"epoch": 0}]
        motion_sources: list[dict[str, Any]] = []
        current_parts = dict(initial_parts)
        for step in range(2):
            successors: dict[str, dict[str, Any]] = {}
            for body in profile.bodies:
                predecessor = _part_carrier(current_parts[body.name])
                successor, moved_plan = rigid_planar_successor(
                    current_parts[body.name],
                    body.offset(step),
                    motion_id=f"qualification-accepted-motion-{step}-{body.name}",
                    plan=plans[body.name],
                )
                plans[body.name] = moved_plan
                current_parts[body.name] = successor
                carrier = _part_carrier(successor)
                certification = carrier.certification
                if certification is None:
                    raise ValueError("Rigid successor fixture requires actual certification.")
                source = moved_plan.source
                if not isinstance(source, phx.meshing.NativePlanarSource):
                    raise TypeError("Rigid successor fixture must retain a planar source.")
                lineage = _overset_publication_lineage(carrier, moved_plan)
                successors[body.name] = {
                    "result": _result_identity(carrier),
                    "result_id": carrier.result_id,
                    "certification_id": certification.report_id,
                }
                motion_sources.append(
                    {
                        "step": step,
                        "body": body.name,
                        "source_binding_id": source.binding_id,
                        "source_revision": source.source_revision,
                        "request_id": moved_plan.specification.specification_id,
                        "limits_id": moved_plan.specification.limits.limits_id,
                        "prepared_id": moved_plan.plan_id,
                        "predecessor_result_id": predecessor.result_id,
                        "result_id": carrier.result_id,
                        "publication_lineage": lineage,
                    }
                )
            epochs.append(
                {
                    "epoch": step + 1,
                    "step": step,
                    "offsets": {
                        body.name: body.offset(step).tolist()
                        for body in profile.bodies
                    },
                    "source_successors": successors,
                }
            )

        background = _part_carrier(initial_parts[_BACKGROUND])
        background_plan = plans[_BACKGROUND]
        ale_part = deform_planar_interior(
            initial_parts[_BACKGROUND],
            np.asarray(background.mesh.coordinates),
            motion_id=f"qualification-conservative-ale-{profile_name}",
            plan=background_plan,
        )
        ale = _part_carrier(ale_part)
        background_receipt = background.execution_evidence
        ale_receipt = ale.execution_evidence
        if (
            background_receipt is None
            or background_receipt.owner_id is None
            or ale_receipt is None
            or ale_receipt.owner_id is None
        ):
            raise ValueError("Overset archives must retain owner-bound actual receipts.")
        ale_lineage = _overset_publication_lineage(ale, background_plan)
        consumer = execute_native_consumer(
            ale,
            background_plan.specification.limits,
            lambda _budget: None,
            failure_factory=qualification_failure,
            observe_managed_admission=True,
        )
        consumer_record = _part_carrier(
            phx.meshing.MeshPart(_BACKGROUND, consumer.result)
        )
        consumer_receipt = consumer_record.execution_evidence
        if consumer_receipt is None or consumer_receipt.owner_id is None:
            raise ValueError("Consumer fixture must retain its actual execution receipt.")

        def retain_cold_history(_budget: Any) -> None:
            scope = current_native_execution_scope()
            if scope is None:
                raise ValueError("Cold fixture must retain its active native request.")
            native_preparation_ledger(scope).record_history(scope, ale_receipt)

        cold = execute_native_request(
            background,
            background_plan.specification.limits,
            retain_cold_history,
            target=lambda _value: background,
            failure_factory=qualification_failure,
            observe_managed_admission=True,
        )
        cold_receipt = cold.result.execution_evidence
        if cold_receipt is None or cold_receipt.owner_id is None:
            raise ValueError("Cold fixture must retain its actual execution receipt.")

    background_record = parts[_BACKGROUND]
    background_id = background_record["generation"]["result_id"]
    template = {
        "scenario": "native-overset-lifecycle",
        "qualification_controls": controls,
        "profile": profile_name,
        "authored_profile": authored_profile,
        "control_capacities": {
            "overset_profile": profile_name,
            "resolution": controls["resolution"],
            "entity_capacity": controls["capacity"],
            "motion_steps": controls["overset_motion_steps"],
            "fringe_layers": authored_profile["fringe_layers"],
            "maximum_wall_candidate_pairs": controls["overset_wall_pairs"],
            "maximum_donor_candidate_pairs": controls["overset_donor_pairs"],
            "maximum_overlap_candidate_pairs": controls["overset_overlap_pairs"],
            "maximum_overlap_accepted_pairs": controls["overset_overlap_pairs"],
            "maximum_overlap_memory_bytes": 4096 * controls["capacity"],
            "maximum_composite_krylov_iterations": _SCHWARZ_MAXIMUM_ITERATIONS,
        },
        "parts": parts,
        "generation_recipe_bindings": recipes,
        "epochs": epochs,
        "motion_sources": motion_sources,
        "conservative_profile": {
            "source_result_id": background_id,
            "motion_prepared_id": background_plan.plan_id,
            "motion_result_id": ale.result_id,
            "motion_result": _result_identity(ale),
            "motion_provenance": ale_lineage,
        },
        "checkpoint": {
            "conservative_publication_result": _result_identity(consumer_record),
            "conservative_publication_execution": {
                "owner_id": consumer_receipt.owner_id,
                "result_id": consumer_record.result_id,
            },
        },
        "restart": {
            "cold_request_result": _result_identity(cold.result),
            "cold_request": {
                "owner_id": cold_receipt.owner_id,
                "history_owner_id": background_receipt.owner_id,
                "history_receipt_id": background_receipt.receipt_hex(),
                "original_limits_id": background_plan.specification.limits.limits_id,
                "conservative_source_owner_id": background_receipt.owner_id,
                "conservative_source_receipt_id": background_receipt.receipt_hex(),
                "conservative_history_owner_id": ale_receipt.owner_id,
                "conservative_history_receipt_id": ale_receipt.receipt_hex(),
            },
        },
    }
    _OVERSET_PUBLICATION_TEMPLATES[profile_name] = template
    return template


def _overset_publication_record(profile_name: str) -> dict[str, Any]:
    from copy import deepcopy

    return deepcopy(_overset_publication_template(profile_name))


@pytest.mark.parametrize(
    ("profile_name", "expected_endpoints"),
    (("routine", 7), ("larger", 10), ("combined", 10)),
)
def test_overset_publication_discovery_requires_each_authored_native_endpoint(
    profile_name: str, expected_endpoints: int
) -> None:
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_publications,
    )

    sample = _compiler_consumer_sample(_overset_publication_record(profile_name))
    publications, paths, gaps = _corpus_publications(
        sample, full_workflow=True, refusal=False
    )
    assert len(publications) == expected_endpoints
    assert len(paths) == expected_endpoints
    assert paths[0] == "parts.background.generation"
    assert paths[-3:] == [
        "conservative_profile.motion_result",
        "checkpoint.conservative_publication_result",
        "restart.cold_request_result",
    ]
    assert publications[0]["result_id"] == publications[-1]["result_id"]
    assert publications[-3]["result_id"] == publications[-2]["result_id"]
    assert (
        publications[0]["resource_binding"]["root_receipt_id"]
        != publications[-1]["resource_binding"]["root_receipt_id"]
    )
    assert (
        publications[-3]["resource_binding"]["root_receipt_id"]
        != publications[-2]["resource_binding"]["root_receipt_id"]
    )
    assert not any("epochs[0].result" in path for path in paths)
    assert gaps == []


@pytest.mark.parametrize(
    ("defect", "expected_gap"),
    (
        (
            "missing-motion",
            "publication:epochs[1].source_successors.moving.result:actual-native-publication",
        ),
        (
            "missing-conservative-motion",
            "publication:conservative_profile.motion_result:actual-native-publication",
        ),
        (
            "missing-checkpoint-consumer",
            "publication:checkpoint.conservative_publication_result:actual-native-publication",
        ),
        (
            "missing-cold",
            "publication:restart.cold_request_result:actual-native-publication",
        ),
        (
            "tampered-motion-crosslink",
            "publication:epochs[1].source_successors.moving:crosslink-result-id",
        ),
        (
            "tampered-conservative-crosslink",
            "publication:conservative_profile:crosslink-motion-result-id",
        ),
        (
            "tampered-cold-crosslink",
            "publication:restart.cold_request_result:crosslink-result-id",
        ),
        (
            "reassigned-body",
            "publication:epochs[1].source_successors.port.result:crosslink-authored-source-plan-certification",
        ),
        (
            "reassigned-step",
            "publication:epochs[1].source_successors.moving.result:crosslink-authored-source-plan-certification",
        ),
        (
            "omit-motion-lineage",
            "publication:epochs[1].source_successors.moving.result:actual-plan-source-certification-lineage",
        ),
        (
            "fabricated-composition-result",
            "publication:epochs[0].result:non-native-composition",
        ),
    ),
)
def test_overset_publication_consumer_rejects_missing_and_tampered_endpoints(
    defect: str, expected_gap: str
) -> None:
    from copy import deepcopy

    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_publications,
    )

    profile_name = "larger" if defect == "reassigned-body" else "routine"
    record = deepcopy(_overset_publication_record(profile_name))
    if defect == "missing-motion":
        del record["epochs"][1]["source_successors"]["moving"]["result"]
    elif defect == "missing-conservative-motion":
        del record["conservative_profile"]["motion_result"]
    elif defect == "missing-checkpoint-consumer":
        del record["checkpoint"]["conservative_publication_result"]
    elif defect == "missing-cold":
        del record["restart"]["cold_request_result"]
    elif defect == "tampered-motion-crosslink":
        record["motion_sources"][0]["result_id"] = "foreign-motion-result"
    elif defect == "tampered-conservative-crosslink":
        record["conservative_profile"]["motion_result_id"] = "foreign-motion-result"
    elif defect == "tampered-cold-crosslink":
        record["restart"]["cold_request_result"]["result_id"] = "foreign-cold-result"
    elif defect == "reassigned-body":
        names = [
            body["name"] for body in record["authored_profile"]["bodies"]
        ]
        first, second = names[:2]
        successors = record["epochs"][1]["source_successors"]
        successors[first]["result"], successors[second]["result"] = (
            successors[second]["result"],
            successors[first]["result"],
        )
        successors[first]["result_id"], successors[second]["result_id"] = (
            successors[second]["result_id"],
            successors[first]["result_id"],
        )
        successors[first]["certification_id"], successors[second]["certification_id"] = (
            successors[second]["certification_id"],
            successors[first]["certification_id"],
        )
        record["motion_sources"][0]["result_id"] = successors[first]["result_id"]
        record["motion_sources"][1]["result_id"] = successors[second]["result_id"]
    elif defect == "reassigned-step":
        first = record["epochs"][1]["source_successors"]["moving"]
        second = record["epochs"][2]["source_successors"]["moving"]
        first["result"], second["result"] = second["result"], first["result"]
        first["result_id"], second["result_id"] = second["result_id"], first["result_id"]
        first["certification_id"], second["certification_id"] = (
            second["certification_id"],
            first["certification_id"],
        )
        record["motion_sources"][0]["result_id"] = first["result_id"]
        record["motion_sources"][1]["result_id"] = second["result_id"]
    elif defect == "omit-motion-lineage":
        del record["motion_sources"][0]["publication_lineage"]
    else:
        record["epochs"][0]["result"] = {"result_id": "not-a-meshing-publication"}
    publications, paths, gaps = _corpus_publications(
        _compiler_consumer_sample(record), full_workflow=True, refusal=False
    )
    assert expected_gap in gaps
    if defect.startswith("missing-"):
        assert len(publications) == 6
    if defect == "fabricated-composition-result":
        assert "epochs[0].result" not in paths



def test_overset_publication_rejects_same_result_with_a_foreign_valid_root() -> None:
    from copy import deepcopy

    from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
    from phydrax.meshing._measurements import (
        native_publication_resource_binding,
        NativeExecutionRecord,
        validate_native_execution_interchange,
    )
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_publications,
    )

    record = deepcopy(_overset_publication_record("routine"))
    publication = record["conservative_profile"]["motion_result"]
    background = record["parts"]["background"]["generation"]
    old_root = publication["resource_binding"]["root_receipt_id"]
    owner_id = publication["resource_binding"]["request_id"]
    background_owner = background["resource_binding"]["request_id"]
    assert isinstance(owner_id, str) and isinstance(background_owner, str)

    def issue(
        owner: str, history: tuple[NativeExecutionRecord, ...] = ()
    ) -> NativeExecutionRecord:
        with NativeExecutionScope(
            NativeExecutionLimits(10, 10, 10, 100_000, 120.0)
        ) as scope:
            scope.observe_managed_admission()
            retained = scope.allocate_host_array((1,), np.float64)
            retained[0] = 1.0
        receipt = NativeExecutionRecord(
            scope.completed_evidence(),
            owner_id=owner,
            history_evidence=history,
        )
        receipt.require_valid()
        return receipt

    foreign_background = issue(background_owner)
    foreign = issue(owner_id, (foreign_background,))
    foreign_record = foreign.to_record()
    observation = validate_native_execution_interchange(foreign_record)
    assert observation["owner_id"] == owner_id
    assert observation["root_receipt_id"] != old_root
    publication["execution_evidence"] = foreign_record
    replacement = native_publication_resource_binding(
        publication["result_id"],
        publication["resource_binding"]["runtime_id"],
        foreign,
        enforced_limits=tuple(publication["resource_guarantees"]["enforced"]),
        unenforced_limits=tuple(publication["resource_guarantees"]["unenforced"]),
    )
    assert replacement is not None
    publication["resource_binding"] = replacement
    _, _, gaps = _corpus_publications(
        _compiler_consumer_sample(record), full_workflow=True, refusal=False
    )
    assert (
        "publication:conservative_profile.motion_result:crosslink-immediate-predecessor-receipt"
        in gaps
    )


def test_overset_cold_publication_rejects_same_result_with_foreign_history_roots() -> None:
    from copy import deepcopy

    from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
    from phydrax.meshing._measurements import (
        native_publication_resource_binding,
        NativeExecutionRecord,
        validate_native_execution_interchange,
    )
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_publications,
    )

    record = deepcopy(_overset_publication_record("routine"))
    background = record["parts"]["background"]["generation"]
    ale = record["conservative_profile"]["motion_result"]
    publication = record["restart"]["cold_request_result"]
    original_result_id = publication["result_id"]
    original_root = publication["resource_binding"]["root_receipt_id"]
    background_owner = background["resource_binding"]["request_id"]
    ale_owner = ale["resource_binding"]["request_id"]
    cold_owner = publication["resource_binding"]["request_id"]
    if (
        not isinstance(background_owner, str)
        or not isinstance(ale_owner, str)
        or not isinstance(cold_owner, str)
    ):
        raise TypeError("Overset receipt fixtures must retain their actual owners.")

    def issue(
        owner: str, history: tuple[NativeExecutionRecord, ...] = ()
    ) -> NativeExecutionRecord:
        with NativeExecutionScope(
            NativeExecutionLimits(10, 10, 10, 100_000, 120.0)
        ) as scope:
            scope.observe_managed_admission()
            retained = scope.allocate_host_array((1,), np.float64)
            retained[0] = 1.0
        receipt = NativeExecutionRecord(
            scope.completed_evidence(),
            owner_id=owner,
            history_evidence=history,
        )
        receipt.require_valid()
        return receipt

    foreign_background = issue(background_owner)
    foreign_ale = issue(ale_owner, (foreign_background,))
    foreign_cold = issue(cold_owner, (foreign_background, foreign_ale))
    foreign_record = foreign_cold.to_record()
    observation = validate_native_execution_interchange(foreign_record)
    assert observation["owner_id"] == cold_owner
    assert observation["root_receipt_id"] != original_root
    publication["execution_evidence"] = foreign_record
    replacement = native_publication_resource_binding(
        original_result_id,
        publication["resource_binding"]["runtime_id"],
        foreign_cold,
        enforced_limits=tuple(publication["resource_guarantees"]["enforced"]),
        unenforced_limits=tuple(publication["resource_guarantees"]["unenforced"]),
    )
    assert replacement is not None
    publication["resource_binding"] = replacement
    _, _, gaps = _corpus_publications(
        _compiler_consumer_sample(record), full_workflow=True, refusal=False
    )
    assert (
        "publication:restart.cold_request_result:crosslink-archived-source-roots"
        in gaps
    )


def test_overset_publication_outer_fields_must_match_the_authenticated_child(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    from copy import deepcopy

    from tools.meshing_qualification._campaign import (
        _complete_campaign_sample,
        _decode_campaign_sample,
    )
    from tools.meshing_qualification._records import _finalize_mandatory_cli_record
    from tools.meshing_qualification.scenarios.native_corpus import (
        _resource_sample_binding_gaps,
    )

    _, runtime = actual_resource_publication
    controls = QualificationControls(
        scenario="native-overset-lifecycle",
        resolution=4,
        capacity=20000,
        repeats=3,
        timeout=180.0,
        target_error=0.05,
        adaptation_rounds=1,
        maximum_work_units=1_280_000,
        maximum_geometry_queries=2_560_000,
        maximum_scratch_bytes=128_000_000,
        overset_profile="routine",
        overset_motion_steps=2,
    )
    child = _finalize_mandatory_cli_record(
        controls,
        {
            **_overset_publication_record("routine"),
            "runtime": runtime,
            "status": "diagnostic-unassessed",
            "independent_scientific_certification": "unassessed",
            "mandatory_workflow_completion": "incomplete",
            "case_contract_completion": "incomplete",
        },
    )
    frozen: dict[str, Any] = {
        "case_name": "native-overset-lifecycle",
        "scenario": "native-overset-lifecycle",
        "engine": "native",
        "resolution": 4,
        "entity_capacity": 20000,
        "wall_seconds": 180.0,
        "solver_warm_repeats": 3,
        "target_error": 0.05,
        "adaptation_rounds": 1,
        "scenario_controls": {
            "overset_profile": "routine",
            "overset_motion_steps": 2,
            "overset_wall_pairs": 500_000,
            "overset_donor_pairs": 500_000,
            "overset_overlap_pairs": 500_000,
            "maximum_work_units": 1_280_000,
            "maximum_geometry_queries": 2_560_000,
            "maximum_scratch_bytes": 128_000_000,
        },
    }
    outer: dict[str, Any] = {
        **child,
        "child_evidence": child,
        "frozen_controls": frozen,
        "frozen_controls_id": phx.canonical_fingerprint(frozen),
    }
    sample = _complete_campaign_sample(
        outer,
        20000,
        positive=False,
        accepted=False,
        verdict=child["workflow_result"],
    )
    assert _resource_sample_binding_gaps(sample) == []
    for missing in (
        "overset_profile",
        "overset_motion_steps",
        "maximum_work_units",
        "maximum_scratch_bytes",
    ):
        incomplete_outer = deepcopy(outer)
        del incomplete_outer["frozen_controls"]["scenario_controls"][missing]
        incomplete_outer["frozen_controls_id"] = phx.canonical_fingerprint(
            incomplete_outer["frozen_controls"]
        )
        incomplete_sample = _complete_campaign_sample(
            incomplete_outer,
            20000,
            positive=False,
            accepted=False,
            verdict=child["workflow_result"],
        )
        assert _resource_sample_binding_gaps(incomplete_sample) == [
            "actual-child-resource-binding",
        ]
    altered = deepcopy(sample.to_record())
    altered["child_evidence"] = deepcopy(altered["child_evidence"])
    altered["motion_sources"][0]["result_id"] = "tampered-outer-copy"
    assert _resource_sample_binding_gaps(_decode_campaign_sample(altered)) == [
        "actual-child-resource-binding",
    ]
def test_polyhedral_phase_consumer_material_mapping_keeps_each_publication_unproved_without_receipt(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_resource_assessment,
    )

    actual, _ = actual_resource_publication
    # Only the source below has real issued evidence. Other endpoints are
    # deliberately incomplete negative inputs, never fabricated receipts/epochs.
    sample = _unmeasured_polyhedral_phase_sample(
        periodic=True,
        generation={"source": actual, "target": {"result_id": "missing-target-receipt"}},
        independent_target_generation={"result_id": "missing-independent-receipt"},
        warm_publications=[{"epoch": 0, "result": {"result_id": "unproved-warm"}}],
        cold={
            "fresh_interpreter": {
                "cold_request_result": {"result_id": "missing-cold-receipt"},
            }
        },
    )
    assessment = _corpus_resource_assessment(sample, full_workflow=True, refusal=False)
    assert assessment["publications"][0] == actual
    assert assessment["publication_evidence_paths"] == [
        "generation.source",
        "generation.target",
        "independent_target_generation",
        "warm_publications[0].result",
        "cold.fresh_interpreter.cold_request_result",
    ]
    assert assessment["admission_coverage"] == "unproved"
    assert (
        sum("actual-ended-native-receipt" in gap for gap in assessment["admission_gaps"])
        == 4
    )


def test_polyhedral_phase_consumer_excludes_resource_trials_and_archived_history() -> (
    None
):
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_resource_assessment,
    )

    unproved = {"result_id": "not-current-publication", "resource_guarantees": {}}
    sample = _unmeasured_polyhedral_phase_sample(
        periodic=True,
        source_prefix_resources={"admitted_at_observed_envelope": {"result": unproved}},
        resources={"complete_native_requests": [{"receipt_id": "summary-only"}]},
        cold={"fresh_interpreter": {"restored": {"epochs": [{"result": unproved}]}}},
    )
    assessment = _corpus_resource_assessment(sample, full_workflow=True, refusal=False)
    assert assessment["publications"] == []
    assert assessment["publication_evidence_paths"] == []
    assert "accepted-publication-runtime-and-receipt" in assessment["admission_gaps"]
    assert any("warm_publications" in gap for gap in assessment["admission_gaps"])
    assert any("cold_request_result" in gap for gap in assessment["admission_gaps"])


@pytest.mark.parametrize(("periodic", "expected"), ((False, 2), (True, 3)))
def test_polyhedral_phase_consumer_requires_actual_requested_warm_publications(
    periodic: bool,
    expected: int,
) -> None:
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_resource_assessment,
    )

    sample = _unmeasured_polyhedral_phase_sample(periodic=periodic, warm_publications=[])
    assessment = _corpus_resource_assessment(sample, full_workflow=True, refusal=False)
    assert (
        f"publication:warm_publications:expected-{expected}-actual-0"
        in assessment["admission_gaps"]
    )


def test_polyhedral_phase_consumer_never_defaults_unknown_requested_rounds() -> None:
    from copy import deepcopy

    from tools.meshing_qualification._campaign import _decode_campaign_sample
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_resource_assessment,
    )

    raw = deepcopy(
        _unmeasured_polyhedral_phase_sample(
            periodic=True, warm_publications=[]
        ).to_record()
    )
    raw["qualification_controls"]["adaptation_rounds"] = None
    assessment = _corpus_resource_assessment(
        _decode_campaign_sample(raw),
        full_workflow=True,
        refusal=False,
    )
    assert (
        "publication:warm_publications:unknown-requested-rounds"
        in assessment["admission_gaps"]
    )


@pytest.mark.parametrize(
    "defect",
    (
        "missing",
        "status-only",
        "relabeled-whole",
        "claims-whole-R",
        "controls-unproved",
        "copied-summaries",
    ),
)
def test_polyhedral_phase_consumer_requires_genuine_source_prefix_boundary_proof(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
    defect: str,
) -> None:
    from tools.meshing_qualification.scenarios.native_corpus import (
        _corpus_resource_assessment,
    )

    prefix: dict[str, Any] = {
        "status": "passed",
        "mandatory_capacity_gate": "complete",
        "admitted_scope": "SOURCEPREFIX",
        "complete_request_R_claimed": False,
        "original_source_and_controls_preserved": True,
        "unobserved_enforcement_claimed": False,
        "unobserved_or_unenforced_original_bounds": [],
    }
    if defect == "status-only":
        prefix = {"status": "passed", "mandatory_capacity_gate": "complete"}
    elif defect == "relabeled-whole":
        prefix["admitted_scope"] = "COMPLETE_REQUEST"
    elif defect == "claims-whole-R":
        prefix["complete_request_R_claimed"] = True
    elif defect == "controls-unproved":
        prefix["original_source_and_controls_preserved"] = False
    elif defect == "copied-summaries":
        prefix["positive"] = {"bounds": {}, "managed_memory_counters_by_phase": []}
        prefix["refusals"] = []
    payload = {} if defect == "missing" else {"source_prefix_resources": prefix}
    publication, runtime = actual_resource_publication
    # A real ended generation result is an intentionally unsealed warm candidate,
    # not a fabricated accepted epoch or a substitute for a genuine R/R-1 proof.
    assessment = _corpus_resource_assessment(
        _bound_polyhedral_generation_diagnostic(
            publication,
            runtime,
            warm_publications=[{"epoch": 0, "result": publication}],
            **payload,
        ),
        full_workflow=True,
        refusal=False,
    )
    assert any(
        "source-prefix-resource-boundary:invalid-evidence" in gap
        for gap in assessment["admission_gaps"]
    )


@pytest.mark.parametrize(
    "field",
    ("physics", "source_physics", "warm_publications", "cold", "source_prefix_resources"),
)
def test_polyhedral_phase_consumer_binds_payload_paths_to_actual_child(
    actual_resource_publication: tuple[dict[str, Any], dict[str, Any]],
    field: str,
) -> None:
    from copy import deepcopy

    from tools.meshing_qualification._campaign import _decode_campaign_sample
    from tools.meshing_qualification.scenarios.native_corpus import (
        _resource_sample_binding_gaps,
    )

    publication, runtime = actual_resource_publication
    sample = _bound_polyhedral_generation_diagnostic(publication, runtime)
    assert _resource_sample_binding_gaps(sample) == []
    altered = deepcopy(sample.to_record())
    altered[field] = {"unbound-status-label": "passed"}
    assert _resource_sample_binding_gaps(_decode_campaign_sample(altered)) == [
        "actual-child-resource-binding",
    ]


@pytest.mark.parametrize("benchmark", (False, True))
def test_authored_resource_controls_roundtrip(benchmark: bool) -> None:
    import argparse

    from tools.meshing_qualification._controls import (
        add_case_arguments,
        BenchmarkControls,
    )

    control_type = BenchmarkControls if benchmark else QualificationControls
    controls = control_type(
        maximum_work_units=1_280_000,
        maximum_geometry_queries=2_560_000,
        maximum_scratch_bytes=128_000_000,
        maximum_data_bytes=40_960_000,
        maximum_recipe_bytes=2_097_152,
        maximum_memory_bytes=12 * 1024**3,
        timeout=180.0,
    )
    assert control_type.from_record(controls.to_record()).to_record() == controls.to_record()
    namespace = argparse.Namespace(**controls.to_record())
    namespace.distributed_bisection_compatibility = controls.distributed_bisection_compatibility
    assert control_type.from_namespace(namespace).maximum_work_units == 1_280_000
    assert control_type.from_namespace(namespace).maximum_scratch_bytes == 128_000_000
    assert control_type.from_namespace(namespace).maximum_data_bytes == 40_960_000
    assert control_type.from_namespace(namespace).maximum_recipe_bytes == 2_097_152
    parser = argparse.ArgumentParser()
    add_case_arguments(parser)
    parsed = parser.parse_args(
        ["--maximum-work-units", "1280000", "--maximum-geometry-queries", "2560000",
         "--maximum-scratch-bytes", "128000000", "--maximum-data-bytes", "40960000",
         "--maximum-recipe-bytes", "2097152"]
    )
    assert parsed.maximum_work_units == controls.maximum_work_units
    assert parsed.maximum_geometry_queries == controls.maximum_geometry_queries
    assert parsed.maximum_scratch_bytes == controls.maximum_scratch_bytes
    assert parsed.maximum_data_bytes == controls.maximum_data_bytes
    assert parsed.maximum_recipe_bytes == controls.maximum_recipe_bytes
    assert control_type().maximum_work_units is None
    assert control_type().maximum_geometry_queries is None
    assert control_type().maximum_scratch_bytes is None
    assert control_type().maximum_data_bytes is None
    assert control_type().maximum_recipe_bytes is None


@pytest.mark.parametrize("benchmark", (False, True))
@pytest.mark.parametrize(
    "field,value",
    [
        (field, value)
        for field in (
            "maximum_work_units", "maximum_geometry_queries",
            "maximum_memory_bytes", "maximum_scratch_bytes",
            "maximum_data_bytes",
            "maximum_recipe_bytes",
        )
        for value in (0, -1, True, 1.5, float("inf"), float("nan"))
    ] + [("timeout", value) for value in (0.0, -1.0, float("inf"), float("nan"))],
)
def test_authored_resource_controls_refuse_nonpositive_or_nonfinite(
    benchmark: bool, field: str, value: Any
) -> None:
    from tools.meshing_qualification._controls import BenchmarkControls

    control_type = BenchmarkControls if benchmark else QualificationControls
    with pytest.raises((TypeError, ValueError)):
        control_type(**{field: value})


def test_registered_authored_resource_envelopes_are_exact_and_selective() -> None:
    from tools._meshing_cases import POLYHEDRAL_CASE_NAMES
    from tools.meshing_qualification._controls import (
        _periodic_qualification_profile_contracts,
    )
    from tools.meshing_qualification._records import _corpus_profiles

    profiles = _corpus_profiles()
    expected = {
        "planar-hole-feature": (4_000_000, 4_000_000, 300.0, None),
        "planar-hole-feature:larger": (4_000_000, 4_000_000, 300.0, None),
        "sphere-feature-surface": (4_000_000, 4_000_000, 300.0, None),
        "native-surface-pde:capped_cylinder": (4_000_000, 4_000_000, 300.0, None),
        "native-overset-lifecycle": (1_280_000, 2_560_000, 180.0, None),
        "native-overset-lifecycle:larger": (2_000_000, 2_560_000, 180.0, None),
        "native-overset-lifecycle:combined": (2_000_000, 2_560_000, 180.0, None),
        "native-distributed-lifecycle": (8_000_000, None, 600.0, 24 * 1024**3),
        "native-distributed-lifecycle:larger": (32_000_000, None, 600.0, 32 * 1024**3),
        "native-family-lifecycle:dual-hex": (128_000_000, 32_000_000, 2400.0, 32 * 1024**3),
        "native-design-lifecycle": (480_000, None, 180.0, None),
        "native-design-lifecycle:larger-sphere": (480_000, None, 180.0, None),
        "native-cad-lifecycle:step": (None, None, 600.0, None),
        "native-image-lifecycle:oblique-junction-transport": (None, None, 360.0, None),
        "native-implicit-lifecycle:sphere-refinement": (None, None, 240.0, None),
        "native-implicit-lifecycle:offset-ring-torus-anisotropic": (80_000_000, None, 2400.0, None),
    }
    polyhedral_profiles = {
        f"{name}{suffix}" for name in POLYHEDRAL_CASE_NAMES for suffix in ("", ":larger")
    }
    assert len(polyhedral_profiles) == 6
    assert {
        name for name, profile in profiles.items() if profile["workflow"] == "polyhedral-system"
    } == polyhedral_profiles
    expected.update({
        name: (2_000_000, None, 300.0, None) for name in polyhedral_profiles
    })
    periodic_contracts = _periodic_qualification_profile_contracts()
    periodic_positives = {
        f"native-periodic-lifecycle:{name}{suffix}"
        for name, contract in periodic_contracts.items()
        if contract["expected_admission"] == "positive"
        for suffix in ("", ":larger")
    }
    assert len(periodic_positives) == 4
    assert {
        name for name, profile in profiles.items()
        if profile["scenario"] == "native-periodic-lifecycle"
        and profile["expected_outcome"] == "passed"
    } == periodic_positives
    expected.update({
        name: (
            320_000_000 if profiles[name]["controls"]["periodic_case"] == "skew-material-feature-exact-algebraic" else None,
            5_120_000 if profiles[name]["controls"]["periodic_case"] == "skew-material-feature-exact-algebraic" else None,
            1200.0 if profiles[name]["controls"]["periodic_case"] == "skew-material-feature-exact-algebraic" else 300.0,
            None,
        )
        for name in periodic_positives
    })
    for name, contract in periodic_contracts.items():
        if contract["expected_admission"] == "refusal":
            for suffix in ("", ":larger"):
                profile = profiles[f"native-periodic-lifecycle:{name}{suffix}"]
                assert "resource_envelope" not in profile
                assert profile["controls"] == {"periodic_case": name}
    fields = ("maximum_work_units", "maximum_geometry_queries", "timeout", "maximum_memory_bytes")
    for name, values in expected.items():
        profile = profiles[name]
        assert tuple(profile["controls"].get(field) for field in fields) == values
        envelope = profile["resource_envelope"]
        assert envelope["requested"] == envelope["effective"]
        assert envelope["rationale"]
        assert all(value > 0 and np.isfinite(value) for value in envelope["effective"].values())
    for name in ("native-image-lifecycle", "native-implicit-lifecycle", "plc-material-cavity"):
        assert "resource_envelope" not in profiles[name]
        assert not set(fields).intersection(profiles[name]["controls"])
    for name in ("native-distributed-lifecycle", "native-distributed-lifecycle:larger"):
        profile = profiles[name]
        contracts = profile["resource_envelope"]["process_memory_acceptance"]
        for contract in contracts.values():
            assert contract["maximum_bytes"] == profile["controls"]["maximum_memory_bytes"]
        assert contracts["linux"]["required_execution_group_hard_cap"]
        assert not contracts["darwin"]["required_execution_group_hard_cap"]
        assert not contracts["darwin"]["descendants_included"]
    for profile in profiles.values():
        if (
            profile["scenario"] == "native-surface-pde"
            or profile["case_name"] == "planar-hole-feature"
        ):
            assert profile["controls"]["maximum_scratch_bytes"] == 1_024_000_000
        elif profile["scenario"] == "native-overset-lifecycle":
            assert profile["controls"]["maximum_scratch_bytes"] == 128_000_000
        elif profile["scenario"] == "native-family-lifecycle":
            assert profile["controls"]["maximum_scratch_bytes"] == 1_536_000_000
        elif profile["case_name"] in POLYHEDRAL_CASE_NAMES:
            assert profile["controls"]["maximum_scratch_bytes"] == 256_000_000
        elif (
            profile["scenario"] == "native-periodic-lifecycle"
            and profile["expected_outcome"] == "passed"
        ):
            material = profile["controls"]["periodic_case"] == "skew-material-feature-exact-algebraic"
            assert profile["controls"]["maximum_scratch_bytes"] == (512_000_000 if material else 256_000_000)
            if material:
                assert profile["controls"]["maximum_work_units"] == 320_000_000
                assert profile["controls"]["maximum_geometry_queries"] == 5_120_000
            else:
                assert "maximum_work_units" not in profile["controls"]
                assert "maximum_geometry_queries" not in profile["controls"]
        else:
            assert "maximum_scratch_bytes" not in profile["controls"]
        if profile["scenario"] == "native-family-lifecycle":
            assert profile["controls"]["maximum_data_bytes"] == 40_960_000
            assert profile["resource_envelope"]["rerun_prerequisites"] == [
                "durable-incremental-stage-and-failure-evidence",
            ]
        else:
            assert "maximum_data_bytes" not in profile["controls"]
        if profile["controls"].get("periodic_case") == "skew-material-feature-exact-algebraic":
            assert profile["controls"]["maximum_recipe_bytes"] == 2_097_152
        else:
            assert "maximum_recipe_bytes" not in profile["controls"]


@pytest.mark.parametrize(
    "profile_name",
    ("planar-hole-feature", "native-distributed-lifecycle", "native-distributed-lifecycle:larger",
     "native-cad-lifecycle:step", "native-image-lifecycle",
     "native-image-lifecycle:oblique-junction-transport", "native-overset-lifecycle",
     "native-overset-lifecycle:larger", "native-overset-lifecycle:combined",
     "native-family-lifecycle:dual-hex", "native-family-lifecycle:curved-cavity-hex:larger",
     "sphere-feature-surface", "native-surface-pde:capped_cylinder",
     "polyhedral-material-L", "polyhedral-material-L:larger",
     "polyhedral-stretched-L", "polyhedral-stretched-L:larger",
     "polyhedral-periodic-cube", "polyhedral-periodic-cube:larger",
     "native-implicit-lifecycle", "native-implicit-lifecycle:sphere-refinement",
     "native-implicit-lifecycle:offset-ring-torus-anisotropic",
     "native-periodic-lifecycle:skew-lattice-exact-algebraic",
     "native-periodic-lifecycle:skew-lattice-exact-algebraic:larger",
     "native-periodic-lifecycle:skew-material-feature-exact-algebraic",
     "native-periodic-lifecycle:skew-material-feature-exact-algebraic:larger"),
)
def test_registered_campaign_and_corpus_commands_have_identical_envelopes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile_name: str
) -> None:
    from tools.meshing_qualification import _campaign
    from tools.meshing_qualification._controls import BenchmarkControls
    from tools.meshing_qualification._records import _corpus_profiles
    from tools.meshing_qualification.scenarios import native_corpus

    attempts = []

    class Captured(Exception):
        pass

    def capture(*args: Any, **kwargs: Any) -> NoReturn:
        attempts.append(_campaign._prepare_campaign_attempt(*args, **kwargs))
        raise Captured

    monkeypatch.setattr(_campaign, "_native_campaign_sample", capture)
    monkeypatch.setattr(native_corpus, "_native_campaign_sample", capture)
    profiles = _corpus_profiles()
    with pytest.raises(Captured):
        native_corpus._execute_corpus_profiles(
            QualificationControls(
                corpus_profile=(profile_name,),
                distributed_checkpoint_root=str(tmp_path / "checkpoint"),
            ),
            profiles,
        )
    with pytest.raises(Captured):
        _campaign._benchmark_registered_native_corpus(
            BenchmarkControls(
                resolution=(4,), benchmark_warm_repeats=3,
                corpus_profile=(profile_name,),
                distributed_checkpoint_root=str(tmp_path / "checkpoint"),
            ),
            profiles,
        )
    first, second = attempts
    assert first.command == second.command
    assert first.expected_controls == second.expected_controls
    assert first.frozen_controls == second.frozen_controls
    assert first.command.count("--timeout") == 1
    assert float(first.command[first.command.index("--timeout") + 1]) == first.timeout
    for name, value in profiles[profile_name]["controls"].items():
        assert first.expected_controls[name] == value
    if "resource_envelope" in profiles[profile_name]:
        assert first.frozen_controls["resource_envelope"] == profiles[profile_name]["resource_envelope"]


@pytest.mark.parametrize("profile_name,field,required", (
    ("planar-hole-feature", "maximum_work_units", 2_000_000),
    ("native-surface-pde:capped_cylinder", "maximum_work_units", 2_000_001),
    ("planar-hole-feature", "maximum_geometry_queries", 2_560_000),
    ("native-distributed-lifecycle", "maximum_work_units", 4_000_000),
    ("native-distributed-lifecycle:larger", "maximum_work_units", 8_000_000),
    ("native-distributed-lifecycle", "maximum_work_units", 4_002_363),
    ("native-distributed-lifecycle:larger", "maximum_work_units", 4_002_363),
    ("native-family-lifecycle:dual-hex", "maximum_work_units", 64_000_000),
    ("native-family-lifecycle:curved-cavity-hex:larger", "maximum_work_units", 64_000_000),
    ("native-family-lifecycle:dual-hex", "maximum_geometry_queries", 16_000_000),
    ("native-overset-lifecycle", "maximum_work_units", 1_280_000),
    ("native-overset-lifecycle:larger", "maximum_work_units", 1_280_000),
    ("native-overset-lifecycle:combined", "maximum_work_units", 1_280_000),
    ("polyhedral-material-L", "maximum_work_units", 640_000),
    ("polyhedral-periodic-cube:larger", "maximum_work_units", 640_000),
    ("native-implicit-lifecycle:offset-ring-torus-anisotropic", "maximum_work_units", 20_000_000),
))
def test_authored_cap_preserves_exact_native_r_and_r_minus_one(
    profile_name: str, field: str, required: int
) -> None:
    from phydrax._meshcore import (
        meshcore_available,
        MeshcoreError,
        MeshcoreStatus,
        NativeExecutionScope,
    )
    from tools.meshing_qualification._controls import capacity_meshing_limits
    from tools.meshing_qualification._records import _corpus_profiles

    if not meshcore_available():
        pytest.skip("native meshing extension unavailable")
    authored = _corpus_profiles()[profile_name]["controls"]
    for cap in (authored[field], required, required - 1):
        limits = capacity_meshing_limits(
            20000, authored["timeout"], **{field: cap},
        )
        scope = NativeExecutionScope(limits.native_limits())
        work = required if field == "maximum_work_units" else 0
        queries = required if field == "maximum_geometry_queries" else 0
        if cap >= required:
            with scope, scope.batch(work, queries_upper=queries) as ticket:
                ticket.consume(work, geometry_queries=queries)
            evidence = scope.completed_evidence()
            assert evidence.externally_charged_work == work
            assert evidence.externally_charged_geometry_queries == queries
        else:
            with pytest.raises(MeshcoreError), scope:
                with scope.batch(work, queries_upper=queries):
                    pytest.fail("R-1 must refuse before dispatch")
            evidence = scope.completed_evidence()
            assert evidence.status is MeshcoreStatus.CAPACITY_EXCEEDED
            assert evidence.externally_charged_work == 0
            assert evidence.externally_charged_geometry_queries == 0


@pytest.mark.parametrize("profile_name,authored_timeout,elapsed,remaining", (
    ("native-image-lifecycle:oblique-junction-transport", 360.0, 130.352, 229.648),
    ("native-cad-lifecycle:step", 600.0, 240.0, 360.0),
    ("native-family-lifecycle:dual-hex", 2400.0, 1260.0, 1140.0),
    ("native-surface-pde:capped_cylinder", 300.0, 137.25, 162.75),
    ("native-distributed-lifecycle", 600.0, 148.9, 451.1),
    ("native-distributed-lifecycle:larger", 600.0, 148.9, 451.1),
    ("polyhedral-material-L", 300.0, 120.0, 180.0),
    ("native-implicit-lifecycle:sphere-refinement", 240.0, 120.0, 120.0),
    ("native-implicit-lifecycle:offset-ring-torus-anisotropic", 2400.0, 1200.0, 1200.0),
    ("native-periodic-lifecycle:skew-lattice-exact-algebraic", 300.0, 120.0, 180.0),
    ("native-periodic-lifecycle:skew-material-feature-exact-algebraic", 1200.0, 440.0, 760.0),
))
def test_authored_timeout_keeps_one_absolute_cold_deadline(
    monkeypatch: pytest.MonkeyPatch, profile_name: str,
    authored_timeout: float, elapsed: float, remaining: float,
) -> None:
    from tools.meshing_qualification import _runtime
    from tools.meshing_qualification._controls import qualification_timeout
    from tools.meshing_qualification._records import _corpus_profiles

    profiles = _corpus_profiles()
    controls = profiles[profile_name]["controls"]
    timeout = qualification_timeout(controls, 120.0)
    assert timeout == authored_timeout
    assert qualification_timeout(profiles["native-image-lifecycle"]["controls"], 120.0) == 120.0
    launch = 1000.0
    expires = launch + timeout
    now = [launch + elapsed]
    monkeypatch.setattr(_runtime, "perf_counter", lambda: now[0])
    monkeypatch.setattr(_runtime, "_LAUNCH_EXPIRES_AT", expires)
    monkeypatch.setenv(_runtime.LAUNCH_DEADLINE_VARIABLE, str(expires))
    monkeypatch.setattr(_runtime.time, "clock_gettime", lambda _: now[0])
    original = _runtime.Deadline(expires)
    cold = _runtime.Deadline(now[0] + timeout)
    assert original.remaining() == pytest.approx(remaining)
    assert cold.remaining() == original.remaining()
    assert cold.monotonic_expiry() == expires
    now[0] = expires - 1.0
    assert cold.remaining() == 1.0
    now[0] = expires
    with pytest.raises(TimeoutError):
        cold.remaining()
    assert _runtime.REPORTING_GRACE_SECONDS == 10.0


@pytest.mark.parametrize("scenario,authored_scratch,required", (
    ("overset", 128_000_000, 93_128_801),
    ("family", 1_536_000_000, 768_001_807),
    ("planar", 1_024_000_000, 512_001_174),
    ("sphere", 1_024_000_000, 512_001_174),
    ("capped_cylinder", 1_024_000_000, 512_001_174),
))
def test_authored_scratch_preserves_other_limits_and_exact_polynomial_boundary(
    scenario: str, authored_scratch: int, required: int,
) -> None:
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )
    from phydrax.meshing.providers._native_publication import derived_certificate_limits
    from tools._meshing_cases import prepare_case
    from tools.meshing_qualification._controls import meshing_limit_values
    from tools.meshing_qualification.scenarios.family import _family_qualification_fixture
    from tools.meshing_qualification.scenarios.overset import (
        _BACKGROUND,
        _overset_common_refinement_policy,
        _overset_qualification_source,
    )
    from tools.meshing_qualification.scenarios.surface import (
        _surface_qualification_prepare,
    )

    def limits(scratch: int | None) -> phx.meshing.MeshingLimits:
        if scenario == "family":
            return _family_qualification_fixture(QualificationControls(
                family_case="dual-hex", maximum_work_units=128_000_000,
                maximum_geometry_queries=32_000_000,
                maximum_scratch_bytes=scratch, timeout=2400.0,
            ))[1].limits
        if scenario == "planar":
            return prepare_case(
                meshing_case("planar-hole-feature"), 4, 20000, 300.0,
                maximum_work_units=4_000_000,
                maximum_geometry_queries=4_000_000,
                maximum_scratch_bytes=scratch,
            )[1].limits
        if scenario in ("sphere", "capped_cylinder"):
            return _surface_qualification_prepare(QualificationControls(
                surface_case=scenario, maximum_work_units=4_000_000,
                maximum_geometry_queries=4_000_000,
                maximum_scratch_bytes=scratch, timeout=300.0,
            ))[1].limits
        return _overset_qualification_source(
            _BACKGROUND, 4, 20000, 180.0, None,
            maximum_work_units=1_280_000,
            maximum_geometry_queries=2_560_000,
            maximum_scratch_bytes=scratch,
        )[1].limits

    original = limits(None)
    authored = limits(authored_scratch)
    assert original.maximum_scratch_bytes == 81_920_000
    assert authored.maximum_scratch_bytes == authored_scratch

    for name, value in meshing_limit_values(original).items():
        if name != "maximum_scratch_bytes":
            assert getattr(authored, name) == value
    for cap in (authored.maximum_scratch_bytes, required, required - 1):
        exact = limits(cap)
        assert derived_certificate_limits(exact).maximum_scratch_bytes == cap
        if scenario == "overset":
            assert _overset_common_refinement_policy(
                500000, exact.maximum_scratch_bytes,
            ).maximum_memory_bytes == cap
        with NativeExecutionScope(exact.native_limits()) as scope:
            with scope.batch(0, bytes_upper=cap) as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket, maximum_work_units=exact.maximum_work_units,
                    maximum_memory_bytes=exact.maximum_scratch_bytes,
                )
                assert budget.maximum_memory_bytes == cap
                if scenario == "overset":
                    assert _overset_common_refinement_policy(
                        500000, budget.maximum_memory_bytes,
                    ).maximum_memory_bytes == cap
                if cap >= required:
                    budget.reserve(0, required)
                    assert budget.peak_bytes_upper == required
                else:
                    with pytest.raises(CoordinateEnclosureResourceError) as failure:
                        budget.reserve(0, required)
                    assert failure.value.resource == "polynomial_storage"
                    assert budget.peak_bytes_upper == 0


@pytest.mark.parametrize("work", (1_280_000, 2_000_000))
@pytest.mark.parametrize("mismatch", (
    "requested", "main", "conservative", "main-work", "conservative-work",
))
def test_overset_cold_resource_limits_are_inherited_from_both_archives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mismatch: str, work: int,
) -> None:
    from types import SimpleNamespace

    from tools.meshing_qualification._controls import capacity_meshing_limits
    from tools.meshing_qualification._runtime import (
        ColdControlPacket,
        Deadline,
        QualificationFailure,
    )
    from tools.meshing_qualification.scenarios import overset

    scratch = 128_000_000
    archives = [
        SimpleNamespace(
            limits=capacity_meshing_limits(
                20000, 180.0, scratch_bytes_limit=scratch - int(mismatch == name),
                maximum_work_units=work - int(mismatch == f"{name}-work"),
            ),
            specification_id="background-specification",
            source_binding_id="background-source",
            execution_evidence=object(),
        )
        for name in ("main", "conservative")
    ]
    reads = iter(archives)
    monkeypatch.setattr(overset, "read_meshing_source_execution_controls", lambda *a, **k: next(reads))
    packet = ColdControlPacket(
        archive_path=tmp_path / "main",
        content_id="main-content",
        controls={
            "capacity": 20000,
            "maximum_scratch_bytes": scratch - int(mismatch == "requested"),
            "conservative": {"archive_path": str(tmp_path / "conservative"), "content_id": "cold-content"},
            "expected": {
                "specification_ids": {overset._BACKGROUND: "background-specification"},
                "source_binding_ids": {overset._BACKGROUND: "background-source"},
            },
        },
        deadline=Deadline(overset.perf_counter() + 180.0),
    )
    message = "same original background hard request" if mismatch.endswith("-work") else "scratch must equal"
    with pytest.raises(QualificationFailure, match=message):
        overset.cold_worker(packet)


def test_family_cold_scratch_inherits_archived_proof_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from tools.meshing_qualification._controls import capacity_meshing_limits
    from tools.meshing_qualification._runtime import ColdControlPacket, Deadline
    from tools.meshing_qualification.scenarios import family

    archived = SimpleNamespace(
        limits=capacity_meshing_limits(
            20000, 2400.0, maximum_work_units=128_000_000,
            maximum_geometry_queries=32_000_000,
            scratch_bytes_limit=1_536_000_000,
            data_bytes_limit=40_960_000,
        ),
        execution_evidence=object(),
    )
    monkeypatch.setattr(family, "read_meshing_source_execution_controls", lambda *a, **k: archived)
    inherited = []

    class Captured(Exception):
        pass

    def execute(evidence: Any, limits: phx.meshing.MeshingLimits, *args: Any, **kwargs: Any) -> NoReturn:
        assert evidence is archived.execution_evidence
        assert limits is archived.limits
        assert limits.maximum_scratch_bytes == 1_536_000_000
        assert limits.maximum_work_units == 128_000_000
        assert limits.maximum_geometry_queries == 32_000_000
        assert limits.maximum_data_bytes == 40_960_000
        assert kwargs["workflow_remaining_seconds"] <= 2400.0
        inherited.append(limits)
        raise Captured

    monkeypatch.setattr(family, "execute_native_request", execute)
    packet = ColdControlPacket(
        archive_path=tmp_path / "family",
        content_id="family-content",
        controls={
            "oracle_path": str(tmp_path / "oracle"), "oracle_content_id": "oracle",
            "expected": (), "repeats": 1, "target_error": 0.05,
        },
        deadline=Deadline(family.perf_counter() + 2400.0),
    )
    with pytest.raises(Captured):
        family.cold_worker(packet)
    assert inherited == [archived.limits]


@pytest.mark.parametrize("family,original,available,next_batch,authored,required", (
    (True, 768_000_000, 570_621_076, 570_622_883, 1_536_000_000, 768_001_807),
    (False, 512_000_000, 57_885_430, 57_886_604, 1_024_000_000, 512_001_174),
))
def test_authored_managed_scratch_reservation_preserves_exact_r_and_r_minus_one(
    family: bool, original: int, available: int, next_batch: int,
    authored: int, required: int,
) -> None:
    from phydrax._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionScope
    from tools.meshing_qualification._controls import capacity_meshing_limits

    retained = original - available
    assert retained + next_batch == required
    for cap in (authored, required, required - 1):
        limits = capacity_meshing_limits(
            20000, 2400.0 if family else 300.0,
            maximum_work_units=128_000_000 if family else 4_000_000,
            maximum_geometry_queries=32_000_000 if family else 4_000_000,
            scratch_bytes_limit=cap,
        )
        scope = NativeExecutionScope(limits.native_limits())
        if cap >= required:
            with scope, scope.batch(0, bytes_upper=retained):
                with scope.batch(0, bytes_upper=next_batch):
                    pass
        else:
            with pytest.raises(MeshcoreError), scope:
                with scope.batch(0, bytes_upper=retained):
                    with scope.batch(0, bytes_upper=next_batch):
                        pytest.fail("R-1 must refuse the managed batch before dispatch")
            assert scope.completed_evidence().status is MeshcoreStatus.CAPACITY_EXCEEDED


@pytest.mark.parametrize("scenario", (
    "native-lifecycle", "native-polyhedral-lifecycle", "native-periodic-lifecycle",
))
def test_direct_native_benchmark_forwards_explicit_resource_overrides(
    monkeypatch: pytest.MonkeyPatch, scenario: str,
) -> None:
    from tools import meshing_benchmarks
    from tools.meshing_qualification._controls import BenchmarkControls

    captured = []

    def benchmark(*args: Any, **kwargs: Any) -> dict[str, object]:
        captured.append((args, kwargs))
        return {}

    monkeypatch.setattr(meshing_benchmarks, "benchmark_native_lifecycle", benchmark)
    meshing_benchmarks._run_benchmark_cases(
        BenchmarkControls(
            case=(scenario,), resolution=(4,), capacity=(20000,),
            polyhedral_source=("polyhedral-material-L",),
            maximum_work_units=731, maximum_geometry_queries=419,
            maximum_scratch_bytes=1024, timeout=17.0,
            maximum_recipe_bytes=1024 if scenario == "native-periodic-lifecycle" else None,
        ),
        {},
    )
    assert len(captured) == 1
    arguments, keywords = captured[0]
    assert arguments[3] == 17.0
    assert keywords["controls"] == {
        "maximum_work_units": 731,
        "maximum_geometry_queries": 419,
        **(
            {"periodic_case": "skew-lattice-exact-algebraic", "maximum_recipe_bytes": 1024}
            if scenario == "native-periodic-lifecycle" else {}
        ),
        "maximum_scratch_bytes": 1024,
    }


@pytest.mark.parametrize("scenario", ("native-lifecycle", "comparison-triangle-lifecycle"))
def test_direct_planar_scenario_forwards_explicit_resource_overrides(
    monkeypatch: pytest.MonkeyPatch, scenario: str,
) -> None:
    from tools.meshing_qualification.scenarios import native_lifecycle

    captured = []

    def qualify(*args: Any, **kwargs: Any) -> dict[str, object]:
        captured.append(kwargs)
        return {}

    monkeypatch.setattr(native_lifecycle, "qualify_native_lifecycle", qualify)
    native_lifecycle.run(QualificationControls(
        scenario=scenario, maximum_work_units=731, maximum_geometry_queries=419,
        maximum_scratch_bytes=1024, timeout=17.0,
    ))
    assert len(captured) == 1
    assert captured[0]["maximum_work_units"] == 731
    assert captured[0]["maximum_geometry_queries"] == 419
    assert captured[0]["maximum_scratch_bytes"] == 1024
    assert captured[0]["timeout"] == 17.0


def test_family_retained_data_cap_preserves_exact_publication_r_and_r_minus_one() -> None:
    from phydrax.meshing.providers._native_publication import _require_retained_bytes
    from tools.meshing_qualification._controls import meshing_limit_values
    from tools.meshing_qualification.scenarios.family import _family_qualification_fixture

    def limits(data: int | None) -> phx.meshing.MeshingLimits:
        return _family_qualification_fixture(QualificationControls(
            family_case="dual-hex", maximum_work_units=128_000_000,
            maximum_geometry_queries=32_000_000,
            maximum_scratch_bytes=1_536_000_000, maximum_data_bytes=data,
            timeout=2400.0,
        ))[1].limits

    original = limits(None)
    authored = limits(40_960_000)
    assert original.maximum_data_bytes == 20_480_000
    assert authored.maximum_data_bytes == 40_960_000
    for name, value in meshing_limit_values(original).items():
        if name != "maximum_data_bytes":
            assert getattr(authored, name) == value
    required = 27_727_378
    for cap in (authored.maximum_data_bytes, required, required - 1):
        exact = limits(cap)
        if cap >= required:
            _require_retained_bytes(required, exact)
        else:
            with pytest.raises(phx.meshing.MeshingFailure) as refused:
                _require_retained_bytes(required, exact)
            assert refused.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
            assert dict(refused.value.evidence.requested)["maximum_data_bytes"] == cap
            assert dict(refused.value.evidence.achieved)["retained_data_bytes"] == required


@pytest.mark.parametrize("memory_cap", (34_359_738_368, 34_359_738_367))
def test_family_external_memory_r_and_r_minus_one_forward_without_rounding(
    memory_cap: int,
) -> None:
    """Prove exact external cap transport, not kernel RSS admission."""
    from tools.meshing_qualification._campaign import (
        _native_driver_controls,
        _prepare_campaign_attempt,
    )
    from tools.meshing_qualification._controls import meshing_limit_values
    from tools.meshing_qualification.scenarios.family import _family_qualification_fixture

    args = QualificationControls(
        scenario="native-family-lifecycle", maximum_memory_bytes=memory_cap,
        maximum_work_units=128_000_000, maximum_geometry_queries=32_000_000,
        maximum_scratch_bytes=1_536_000_000, maximum_data_bytes=40_960_000,
        timeout=2400.0,
    )
    controls = _native_driver_controls(args.scenario, args)
    attempt = _prepare_campaign_attempt(
        args.scenario, args.resolution, args.capacity, args.timeout,
        args.repeats, args.target_error, args.adaptation_rounds, "native",
        scenario_override=args.scenario, controls=controls,
    )
    flag = attempt.command.index("--maximum-memory-bytes")
    assert attempt.command[flag + 1] == str(memory_cap)
    assert attempt.expected_controls["maximum_memory_bytes"] == memory_cap
    assert attempt.frozen_controls["scenario_controls"]["maximum_memory_bytes"] == memory_cap
    assert args.to_record()["maximum_memory_bytes"] == memory_cap
    old = QualificationControls.from_record({
        **args.to_record(), "maximum_memory_bytes": 16 * 1024**3,
    })
    old_limits = _family_qualification_fixture(old)[1].limits
    new_limits = _family_qualification_fixture(args)[1].limits
    assert meshing_limit_values(old_limits) == meshing_limit_values(new_limits)
    assert new_limits.maximum_scratch_bytes == 1_536_000_000
    assert new_limits.maximum_data_bytes == 40_960_000


@pytest.mark.parametrize("name", (
    "polyhedral-material-L", "polyhedral-stretched-L", "polyhedral-periodic-cube",
))
@pytest.mark.parametrize("sites", (4, 8))
@pytest.mark.parametrize("source", (False, True))
def test_polyhedral_authored_limits_cover_all_sources_and_sizes(
    name: str, sites: int, source: bool,
) -> None:
    from tools._meshing_cases import prepare_polyhedral_case
    from tools.meshing_qualification._controls import meshing_limit_values
    from tools.meshing_qualification.scenarios.polyhedral import (
        _polyhedral_common_refinement_policy,
    )

    original = prepare_polyhedral_case(name, sites, 20000, 120.0, source=source)[1].limits
    authored = prepare_polyhedral_case(
        name, sites, 20000, 300.0, source=source,
        maximum_work_units=2_000_000, maximum_scratch_bytes=256_000_000,
    )[1].limits
    assert authored.maximum_work_units == 2_000_000
    assert authored.maximum_scratch_bytes == 256_000_000
    assert authored.maximum_geometry_queries == 1_280_000
    assert authored.maximum_wall_seconds == 300.0
    for field, value in meshing_limit_values(original).items():
        if field not in ("maximum_work_units", "maximum_scratch_bytes", "maximum_wall_seconds"):
            assert getattr(authored, field) == value
    native = authored.native_limits()
    assert native.maximum_work_units == 2_000_000
    assert native.maximum_geometry_queries == 1_280_000
    assert native.maximum_managed_bytes == 256_000_000
    assert native.maximum_seconds == 300.0
    policy = _polyhedral_common_refinement_policy(20000, authored)
    assert policy.maximum_exact_work == 2_000_000
    assert policy.maximum_memory_bytes == 256_000_000
    assert policy.maximum_candidate_pairs == policy.maximum_accepted_pairs == 640_000
    explicit = prepare_polyhedral_case(
        name, sites, 20000, 17.0, source=source,
        maximum_work_units=731, maximum_geometry_queries=419,
        maximum_scratch_bytes=1024,
    )[1].limits
    assert (explicit.maximum_work_units, explicit.maximum_geometry_queries, explicit.maximum_scratch_bytes) == (
        731, 419, 1024,
    )


def test_polyhedral_authored_scratch_preserves_exact_native_r_and_r_minus_one() -> None:
    from phydrax._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionScope
    from tools._meshing_cases import prepare_polyhedral_case

    required = 81_920_000
    for cap in (256_000_000, required, required - 1):
        limits = prepare_polyhedral_case(
            "polyhedral-material-L", 4, 20000, 300.0,
            maximum_work_units=2_000_000, maximum_scratch_bytes=cap,
        )[1].limits
        scope = NativeExecutionScope(limits.native_limits())
        if cap >= required:
            with scope, scope.batch(0, bytes_upper=required):
                pass
        else:
            with pytest.raises(MeshcoreError), scope:
                with scope.batch(0, bytes_upper=required):
                    pytest.fail("R-1 must refuse native scratch before dispatch")
            evidence = scope.completed_evidence()
            assert evidence.status is MeshcoreStatus.CAPACITY_EXCEEDED
            assert evidence.externally_charged_work == 0


@pytest.mark.parametrize("kind", ("material", "periodic"))
def test_polyhedral_cold_scope_inherits_archived_authored_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str,
) -> None:
    from types import SimpleNamespace

    from tools._meshing_cases import prepare_polyhedral_case
    from tools.meshing_qualification._runtime import ColdControlPacket, Deadline
    from tools.meshing_qualification.scenarios import polyhedral

    case = "polyhedral-material-L" if kind == "material" else "polyhedral-periodic-cube"
    archived = SimpleNamespace(
        limits=prepare_polyhedral_case(
            case, 4, 20000, 300.0, maximum_work_units=2_000_000,
            maximum_scratch_bytes=256_000_000,
        )[1].limits,
        execution_evidence=object(),
    )
    monkeypatch.setattr(polyhedral, "read_meshing_source_execution_controls", lambda *a, **k: archived)
    inherited = []

    class Captured(Exception):
        pass

    def execute(evidence: Any, limits: phx.meshing.MeshingLimits, *args: Any, **kwargs: Any) -> NoReturn:
        assert evidence is archived.execution_evidence
        assert limits is archived.limits
        assert limits.maximum_work_units == 2_000_000
        assert limits.maximum_geometry_queries == 1_280_000
        assert limits.maximum_scratch_bytes == 256_000_000
        assert limits.maximum_data_bytes == 20_480_000
        assert kwargs["workflow_remaining_seconds"] <= 300.0
        inherited.append(limits)
        raise Captured

    monkeypatch.setattr(polyhedral, "execute_native_request", execute)
    packet = ColdControlPacket(
        archive_path=tmp_path / "polyhedral", content_id="polyhedral-content",
        controls={"kind": kind, "expected": {}, "repeats": 1, "target_error": 0.05},
        deadline=Deadline(polyhedral.perf_counter() + 300.0),
    )
    with pytest.raises(Captured):
        polyhedral.cold_worker(packet)
    assert inherited == [archived.limits]


@pytest.mark.parametrize("case_name,expected_timeout,expected_work", (
    ("sphere", 120.0, 5_120_000),
    ("sphere-refinement", 240.0, 5_120_000),
    ("offset-ring-torus-anisotropic", 2400.0, 80_000_000),
))
def test_implicit_authored_envelope_preserves_other_limits_and_exact_work_boundary(
    case_name: str, expected_timeout: float, expected_work: int,
) -> None:
    from dataclasses import replace

    from phydrax._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionScope
    from tools.meshing_qualification._controls import (
        authored_resource_envelope,
        meshing_limit_values,
        MeshingLimitControls,
    )
    from tools.meshing_qualification.scenarios.implicit import (
        _implicit_case,
        _implicit_qualification_source,
        _implicit_volume_request,
    )

    baseline = QualificationControls.from_record({
        **QualificationControls().to_record(),
        "scenario": "native-implicit-lifecycle", "implicit_case": case_name,
    })
    envelope = authored_resource_envelope(
        baseline.scenario, baseline.scenario, {"implicit_case": case_name}, larger=False,
    )
    effective = {} if envelope is None else envelope["effective"]
    expected_controls = {} if case_name == "sphere" else {"timeout": 240.0}
    if case_name == "offset-ring-torus-anisotropic":
        expected_controls["maximum_work_units"] = 80_000_000
        expected_controls["timeout"] = 2400.0
    assert effective == expected_controls
    revised = QualificationControls.from_record({**baseline.to_record(), **effective})
    _, source, _, witness = _implicit_qualification_source(baseline, _implicit_case(baseline))
    old = _implicit_volume_request(baseline, source, witness, baseline.implicit_fidelity)
    new = _implicit_volume_request(revised, source, witness, revised.implicit_fidelity)
    assert new.limits.maximum_wall_seconds == expected_timeout
    assert new.limits.native_limits().maximum_seconds == expected_timeout
    assert new.limits.maximum_work_units == expected_work
    for name, value in meshing_limit_values(old.limits).items():
        if name != "maximum_wall_seconds" and not (
            name == "maximum_work_units" and case_name == "offset-ring-torus-anisotropic"
        ):
            assert getattr(new.limits, name) == value
    required = old.limits.maximum_work_units
    caps = (expected_work, required) if expected_work != required else (required,)
    for cap in caps:
        admitted = replace(
            MeshingLimitControls.from_limits(new.limits), maximum_work_units=cap,
        ).to_limits()
        with NativeExecutionScope(admitted.native_limits()) as scope:
            with scope.batch(required) as ticket:
                ticket.consume(required)
        assert scope.completed_evidence().externally_charged_work == required
    reduced = replace(
        MeshingLimitControls.from_limits(new.limits), maximum_work_units=required - 1,
    ).to_limits()
    refused = NativeExecutionScope(reduced.native_limits())
    with pytest.raises(MeshcoreError), refused:
        with refused.batch(required):
            pytest.fail("R-1 must still refuse under the authored whole deadline")
    assert refused.completed_evidence().status is MeshcoreStatus.CAPACITY_EXCEEDED


def test_implicit_torus_source_record_retains_exact_authored_work_cap() -> None:
    from tools.meshing_qualification._runtime import Deadline
    from tools.meshing_qualification.scenarios import implicit

    args = QualificationControls(
        scenario="native-implicit-lifecycle",
        implicit_case="offset-ring-torus-anisotropic",
        maximum_work_units=80_000_000, timeout=2400.0, adaptation_rounds=2,
    )
    record: dict[str, Any] = {"stages_seconds": {}}
    prepared = implicit._implicit_prepare_source(
        args, record, deadline=Deadline(implicit.perf_counter() + args.timeout),
    )
    assert record["request"]["resource_budgets"]["work_units"] == 80_000_000
    assert prepared.request.limits.maximum_work_units == 80_000_000
    assert all(request.limits.maximum_work_units == 80_000_000 for request in prepared.ladder_requests)
    assert record["request"]["resource_budgets"]["geometry_queries"] == 5_120_000
    assert record["request"]["resource_budgets"]["scratch_bytes"] == 8 * 1024**3
    assert record["request"]["resource_budgets"]["wall_seconds"] == 2400.0


@pytest.mark.parametrize("case_name", (
    "skew-lattice", "skew-material-feature",
    "skew-lattice-exact-algebraic", "skew-material-feature-exact-algebraic",
))
@pytest.mark.parametrize("resolution", (4, 8))
def test_periodic_authored_envelope_preserves_unraised_limits_and_refusals(
    case_name: str, resolution: int,
) -> None:
    from tools.meshing_qualification._controls import (
        authored_resource_envelope,
        meshing_limit_values,
    )
    from tools.meshing_qualification.scenarios.periodic import (
        _periodic_qualification_fixture,
    )

    baseline = QualificationControls(
        scenario="native-periodic-lifecycle", periodic_case=case_name, resolution=resolution,
    )
    envelope = authored_resource_envelope(
        baseline.scenario, baseline.scenario, {"periodic_case": case_name},
        larger=resolution == 8,
    )
    effective = {} if envelope is None else envelope["effective"]
    positive = case_name.endswith("-exact-algebraic")
    material = case_name == "skew-material-feature-exact-algebraic"
    expected = {"maximum_scratch_bytes": 256_000_000, "timeout": 300.0} if positive else {}
    if material:
        expected["maximum_work_units"] = 320_000_000
        expected["maximum_geometry_queries"] = 5_120_000
        expected["maximum_scratch_bytes"] = 512_000_000
        expected["timeout"] = 1200.0
        expected["maximum_recipe_bytes"] = 2_097_152
    assert effective == expected
    revised = QualificationControls.from_record({**baseline.to_record(), **effective})
    old_source, old, _, _ = _periodic_qualification_fixture(baseline)
    source, request, _, record = _periodic_qualification_fixture(revised)
    assert source.binding_id == old_source.binding_id
    assert request.size_controls[0].control_id == old.size_controls[0].control_id
    assert [item.feature_id for item in request.protected_features] == [
        item.feature_id for item in old.protected_features
    ]
    assert old.limits.maximum_work_units == 40_960_000
    assert request.limits.maximum_work_units == (320_000_000 if material else 40_960_000)
    assert request.limits.maximum_geometry_queries == (5_120_000 if material else 1_280_000)
    assert request.limits.maximum_scratch_bytes == (
        512_000_000 if material else 256_000_000 if positive else 81_920_000
    )
    assert request.limits.maximum_wall_seconds == (1200.0 if material else 300.0 if positive else 120.0)
    for name, value in meshing_limit_values(old.limits).items():
        if name not in ("maximum_scratch_bytes", "maximum_wall_seconds") and not (
            name in ("maximum_work_units", "maximum_geometry_queries") and material
        ):
            assert getattr(request.limits, name) == value
    assert record["resource_limits"] == meshing_limit_values(request.limits)
    native = request.limits.native_limits()
    assert native.maximum_work_units == (320_000_000 if material else 40_960_000)
    assert native.maximum_geometry_queries == (5_120_000 if material else 1_280_000)
    assert native.maximum_managed_bytes == request.limits.maximum_scratch_bytes
    assert native.maximum_seconds == request.limits.maximum_wall_seconds


@pytest.mark.parametrize("case_name,work,field,required", (
    ("skew-lattice-exact-algebraic", None, "maximum_work_units", 40_960_000),
    ("skew-lattice-exact-algebraic", None, "maximum_scratch_bytes", 81_920_000),
    ("skew-material-feature-exact-algebraic", 320_000_000, "maximum_work_units", 160_000_000),
    ("skew-material-feature-exact-algebraic", 320_000_000, "maximum_geometry_queries", 2_560_000),
    ("skew-material-feature-exact-algebraic", 320_000_000, "maximum_scratch_bytes", 307_950_571),
))
def test_periodic_authored_limits_preserve_exact_native_r_and_r_minus_one(
    field: str, required: int, case_name: str, work: int | None,
) -> None:
    from dataclasses import replace

    from phydrax._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionScope
    from tools.meshing_qualification._controls import MeshingLimitControls
    from tools.meshing_qualification.scenarios.periodic import (
        _periodic_qualification_fixture,
    )

    material = case_name == "skew-material-feature-exact-algebraic"
    original = _periodic_qualification_fixture(QualificationControls(
        scenario="native-periodic-lifecycle", periodic_case=case_name,
        maximum_work_units=work,
        maximum_geometry_queries=5_120_000 if material else None,
        maximum_scratch_bytes=512_000_000 if material else 256_000_000,
        timeout=1200.0 if material else 300.0,
    ))[1].limits
    declared = getattr(original, field)
    caps = (declared, required, required - 1) if declared != required else (required, required - 1)
    for cap in caps:
        limits = replace(MeshingLimitControls.from_limits(original), **{field: cap}).to_limits()
        scope = NativeExecutionScope(limits.native_limits())
        work = required if field == "maximum_work_units" else 0
        storage = required if field == "maximum_scratch_bytes" else 0
        queries = required if field == "maximum_geometry_queries" else 0
        if cap >= required:
            with scope, scope.batch(work, queries_upper=queries, bytes_upper=storage) as ticket:
                ticket.consume(work, geometry_queries=queries)
            assert scope.completed_evidence().externally_charged_work == work
            assert scope.completed_evidence().externally_charged_geometry_queries == queries
        else:
            with pytest.raises(MeshcoreError), scope:
                with scope.batch(work, queries_upper=queries, bytes_upper=storage):
                    pytest.fail("R-1 must refuse before periodic dispatch")
            assert scope.completed_evidence().status is MeshcoreStatus.CAPACITY_EXCEEDED
            assert scope.completed_evidence().externally_charged_work == 0
            assert scope.completed_evidence().externally_charged_geometry_queries == 0


@pytest.mark.parametrize("policy_drift", (None, -1, 0, 1))
@pytest.mark.parametrize("case_name,work", (
    ("skew-lattice-exact-algebraic", None),
    ("skew-material-feature-exact-algebraic", 320_000_000),
))
def test_periodic_cold_control_packet_preserves_authored_caps_and_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case_name: str,
    work: int | None, policy_drift: int | None,
) -> None:
    from phydrax.lifecycle._meshing_sources import _source_owner_reconstruction_scope
    from tools.meshing_qualification._controls import (
        archive_limit_values,
        qualification_archive_limits,
    )
    from tools.meshing_qualification._runtime import (
        ColdControlPacket,
        Deadline,
        QualificationFailure,
    )
    from tools.meshing_qualification.scenarios import periodic

    material = case_name == "skew-material-feature-exact-algebraic"
    args = QualificationControls(
        scenario="native-periodic-lifecycle", periodic_case=case_name,
        maximum_work_units=work,
        maximum_geometry_queries=5_120_000 if material else None,
        maximum_scratch_bytes=512_000_000 if material else 256_000_000,
        maximum_recipe_bytes=2_097_152 if material else None,
        timeout=1200.0 if material else 300.0,
    )
    deadline = Deadline(periodic.perf_counter() + args.timeout)
    policy = archive_limit_values(qualification_archive_limits(args))
    if policy_drift is not None:
        policy["max_manifest_bytes"] += policy_drift
    packet = ColdControlPacket(
        archive_path=tmp_path / "periodic", content_id="periodic-content",
        controls={
            "expected": {}, "case": {}, "controls": args.to_record(),
            **({} if policy_drift is None else {"archive_limits": policy}),
        },
        deadline=deadline,
    )
    captured = []

    class Captured(Exception):
        pass

    def cold(*values: Any, archive_limits: phx.ArrayArchiveLimits) -> NoReturn:
        restored = values[4]
        assert isinstance(restored, QualificationControls)
        assert restored.maximum_recipe_bytes == args.maximum_recipe_bytes
        assert archive_limits.max_manifest_bytes == (2_097_152 if material else 1_048_576)
        scope = _source_owner_reconstruction_scope.get()
        assert scope is not None
        assert scope.archive_limits is archive_limits
        assert restored.maximum_work_units == work
        assert restored.maximum_geometry_queries == (5_120_000 if material else None)
        assert restored.maximum_scratch_bytes == args.maximum_scratch_bytes
        assert restored.timeout == args.timeout
        assert values[5] is deadline
        assert values[5].remaining() <= args.timeout
        captured.append(restored)
        raise Captured

    monkeypatch.setattr(periodic, "_periodic_qualification_cold_process", cold)
    if policy_drift != 0:
        with pytest.raises(QualificationFailure, match="Cold archive limits must equal"):
            periodic.cold_worker(packet)
        assert captured == []
    else:
        with pytest.raises(Captured):
            periodic.cold_worker(packet)
        assert len(captured) == 1


def test_periodic_authored_fixture_honors_explicit_cli_resource_overrides() -> None:
    from tools.meshing_qualification.scenarios.periodic import (
        _periodic_qualification_fixture,
    )

    _, request, _, record = _periodic_qualification_fixture(QualificationControls(
        scenario="native-periodic-lifecycle", maximum_work_units=731,
        maximum_geometry_queries=419, maximum_scratch_bytes=1024, timeout=17.0,
    ))
    assert request.limits.maximum_work_units == 731
    assert request.limits.maximum_geometry_queries == 419
    assert request.limits.maximum_scratch_bytes == 1024
    assert request.limits.maximum_wall_seconds == 17.0
    assert record["resource_limits"]["maximum_work_units"] == 731
    assert record["resource_limits"]["maximum_geometry_queries"] == 419
    assert record["resource_limits"]["maximum_scratch_bytes"] == 1024
    assert record["resource_limits"]["maximum_wall_seconds"] == 17.0




@pytest.mark.parametrize("recipe_cap", (None, 2_097_152, 8_388_608))
def test_recipe_limit_changes_only_archive_metadata_bound(
    recipe_cap: int | None,
) -> None:
    from dataclasses import asdict

    from tools.meshing_qualification._controls import qualification_archive_limits

    limits = qualification_archive_limits(QualificationControls(maximum_recipe_bytes=recipe_cap))
    assert limits.max_manifest_bytes == (1_048_576 if recipe_cap is None else recipe_cap)
    original = asdict(phx.ArrayArchiveLimits())
    for name, value in asdict(limits).items():
        if name != "max_manifest_bytes":
            assert value == original[name]


def test_periodic_recipe_and_archive_exact_r_and_r_minus_one(tmp_path: Path) -> None:
    """Exercise the real byte guards without fabricating a source certificate."""
    from zipfile import ZipFile

    from phydrax._array_archive import (
        ArrayArchiveCorruptionError,
        read_array_archive,
        write_array_archive,
    )
    from phydrax._model._structure import (
        _canonical_recipe_payload,
        model_recipe_wire_json,
        model_structure_recipe,
        validate_model_structure_recipe,
    )
    from tools.meshing_qualification._controls import qualification_archive_limits

    values = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)
    values.flags.writeable = False
    required_recipe = 1_099_585
    empty_recipe = model_structure_recipe({"payload": "", "values": values})
    overhead = len(_canonical_recipe_payload(empty_recipe))
    records = {"payload": "x" * (required_recipe - overhead), "values": values}
    recipe = model_structure_recipe(records)
    assert len(_canonical_recipe_payload(recipe)) == required_recipe

    def limits(cap: int) -> phx.ArrayArchiveLimits:
        return qualification_archive_limits(QualificationControls(maximum_recipe_bytes=cap))

    def publish(path: Path, cap: int) -> Path:
        policy = limits(cap)
        return write_array_archive(
            path,
            manifest={
                "recipe": model_recipe_wire_json(recipe, limits=policy),
                "metadata": "boundary-probe" * 128,
            },
            arrays={"values": values},
            limits=policy,
        )

    validate_model_structure_recipe(recipe, limits=limits(required_recipe))
    with pytest.raises(ValueError, match=r"recipe\.bytes"):
        validate_model_structure_recipe(recipe, limits=limits(required_recipe - 1))
    with pytest.raises(ValueError, match=r"recipe\.bytes"):
        publish(tmp_path / "old-limit.zip", 1_048_576)
    assert not (tmp_path / "old-limit.zip").exists()
    admitted = publish(tmp_path / "authored.zip", 2_097_152)
    manifest, arrays = read_array_archive(admitted, limits=limits(2_097_152))
    assert manifest["recipe"] == model_recipe_wire_json(recipe, limits=limits(2_097_152))
    np.testing.assert_array_equal(arrays["values"], values)
    with ZipFile(admitted) as archive:
        required_archive = archive.getinfo("manifest.json").file_size
    assert required_archive > required_recipe
    exact = publish(tmp_path / "exact.zip", required_archive)
    exact_manifest, exact_arrays = read_array_archive(exact, limits=limits(required_archive))
    assert exact_manifest == manifest
    np.testing.assert_array_equal(exact_arrays["values"], values)
    before = sha256(exact.read_bytes()).digest()
    with pytest.raises(ValueError, match="manifest byte limit"):
        publish(exact, required_archive - 1)
    assert sha256(exact.read_bytes()).digest() == before
    with pytest.raises(ArrayArchiveCorruptionError, match="manifest byte limit"):
        read_array_archive(exact, limits=limits(required_archive - 1))
