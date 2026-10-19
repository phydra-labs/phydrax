#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Consumer-visible scientific-oracle boundaries for the native corpus tools."""

from __future__ import annotations

import json
import subprocess
from dataclasses import fields
from hashlib import sha256
from pathlib import Path
from typing import Literal, NoReturn

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from tools._meshing_cases import meshing_case
from tools.meshing_qualification._controls import QualificationControls, SCENARIO_MODULES
from tools.meshing_qualification._physics import _physical_error, _prepare_corpus_solver
from tools.meshing_qualification.scenarios.bisection import _bisection_p1_integral
from tools.meshing_qualification.scenarios.native_lifecycle import (
    _corpus_domain,
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


def test_fixed_boundary_corpus_solves_empty_space_without_claiming_exact_pde() -> None:
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
    space, problem, free = _prepare_corpus_solver(result, material_x=False)
    assert space.dof_maps[0].global_dof_count == 8
    assert np.all(np.asarray(space.dof_maps[0].boundary_dof_mask))
    assert free.shape == (0,)
    system, rhs = problem.linear_system()
    assert system.operator.source.size == system.operator.target.size == 0
    assert rhs.shape == (0,)
    values, solved = _solve_corpus_problem(problem, jnp.zeros((0,), dtype=jnp.float64))
    assert bool(np.asarray(solved.successful))
    assert int(np.asarray(solved.diagnostics.iterations)) == 0
    # The native matrix-free route retains its uncertified-rank sentinel.
    assert int(np.asarray(solved.diagnostics.rank)) == -1
    assert solved.provenance.backend == "native-krylov"
    assert float(np.asarray(solved.diagnostics.residual_norm)) == 0.0
    np.testing.assert_array_equal(
        values, jnp.sum(space.dof_maps[0].dof_coordinates, axis=-1)
    )
    # On cube corners |x|²=sum(x), so the unique P1 state is affine, not
    # the quadratic PDE field. Its analytic L2 error is sqrt(4/15), not zero.
    error = _physical_error(result.mesh, values)
    assert error == pytest.approx(np.sqrt(4.0 / 15.0), rel=_CORPUS_MOMENT_RTOL)
    assert error > 0.05


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
            "epochs": [
                {
                    "epoch": 0,
                    "retained_bytes": 321,
                    "compiler": {
                        "temporary_bytes": 17,
                        "output_bytes": 23,
                        "generated_code_bytes": 41,
                        "warm_samples_seconds": [0.2, 0.3],
                        "execution_scope": "epoch-preassembled-system-residual-solve-and-expand",
                        "assembly_seconds": 0.7,
                        "assembly_invocations": 1,
                        "assembly_compilation_count": 1,
                        "assembly_compile_seconds": 0.6,
                        "assembly_binding_id": "measured-problem",
                    },
                }
            ],
            "stages_seconds": {
                "solver_lowering_epoch_0": 0.4,
                "solver_compile_epoch_0": 0.5,
            },
        },
        20000,
    )["compiler_epochs"][0]
    assert measured["temporary_bytes"] == 17
    assert measured["logical_retained_bytes"] == 321
    assert measured["first_execution_seconds"] is None
    assert (
        measured["execution_scope"]
        == "epoch-preassembled-system-residual-solve-and-expand"
    )
    assert measured["assembly_seconds"] == 0.7
    assert measured["assembly_invocations"] == 1
    assert measured["assembly_compilation_count"] == 1
    assert measured["assembly_compile_seconds"] == 0.6
    assert measured["assembly_binding_id"] == "measured-problem"


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
        ),
    )
    assert assessment["qualification_eligible"] is True
    assert assessment["case_certified"] is True
    assert assessment["positive_completion"] is False
    result = _mandatory_result("routine-solid", profile, assessment)
    assert result["status"] == "case-certified"
    assert result["case_certified"] is True
    assert result["positive_completion"] is False


def test_mandatory_cli_result_is_strict_and_requires_scientific_success() -> None:
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification._records import (
        _finalize_mandatory_cli_record,
        _validate_mandatory_workflow_result,
    )

    arguments = QualificationControls(
        scenario="native-surface-pde", corpus_case="planar-hole-feature"
    )
    incomplete = _finalize_mandatory_cli_record(
        arguments,
        {
            "status": "passed",
            "mandatory_workflow_completion": "complete",
            "independent_scientific_certification": "unassessed",
        },
    )
    result = incomplete["workflow_result"]
    assert isinstance(result, dict)
    assert _validate_mandatory_workflow_result(result) == result
    assert result["status"] == "blocked"
    assert result["positive_completion"] is False

    complete = _finalize_mandatory_cli_record(
        arguments,
        {
            "status": "passed",
            "mandatory_workflow_completion": "complete",
            "independent_scientific_certification": "passed",
        },
    )
    qualified = complete["workflow_result"]
    assert isinstance(qualified, dict)
    assert _validate_mandatory_workflow_result(qualified) == qualified
    assert qualified["status"] == "positive-complete"
    assert qualified["positive_completion"] is True

    with pytest.raises(ValueError, match=r"_validate_mandatory_workflow_result\.fields"):
        _validate_mandatory_workflow_result({**qualified, "unexpected": True})
    with pytest.raises(
        ValueError, match=r"_validate_mandatory_workflow_result\.result_id"
    ):
        _validate_mandatory_workflow_result({**qualified, "result_id": "tampered"})


def test_benchmark_sample_does_not_promote_bare_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.meshing_qualification._campaign as benchmarks
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification._records import _finalize_mandatory_cli_record

    blocked = _finalize_mandatory_cli_record(
        QualificationControls(
            scenario="native-surface-pde", corpus_case="planar-hole-feature"
        ),
        {
            "status": "passed",
            "mandatory_workflow_completion": "complete",
            "independent_scientific_certification": "unassessed",
        },
    )

    def completed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(
            args=["qualification"], returncode=0, stdout=json.dumps(blocked), stderr=""
        )

    monkeypatch.setattr(benchmarks, "run_campaign_process", completed)
    sample = benchmarks._native_campaign_sample(
        "sphere-feature-surface", 1, 16, 1.0, 1, 0.05, 0, "native"
    ).to_record()
    assert sample["status"] == "passed"
    assert sample["positive_completion"] is False
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


def test_isolated_campaign_timeout_retains_hard_launcher_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.meshing_qualification._campaign as benchmarks

    def expire(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise subprocess.TimeoutExpired(
            cmd=["qualification"], timeout=0.01, output=b"partial", stderr=b"blocked"
        )

    monkeypatch.setattr(benchmarks, "run_campaign_process", expire)
    sample = benchmarks._native_campaign_sample(
        "planar-hole-feature", 1, 16, 0.01, 1, 0.05, 0, "native"
    ).to_record()
    assert sample["status"] == "timeout"
    assert sample["positive_completion"] is False
    assert sample["hard_timeout"] == {
        "limit_seconds": 0.01,
        "mechanism": "session-process-group-kill",
        "completed_before_deadline": False,
        "terminated_by_launcher": True,
        "returncode": None,
    }
    assert sample["stdout"] == "partial"
    assert sample["stderr"] == "blocked"
    assert sample["phase_evidence"]["status"] == "recorded"
    assert sample["resource_evidence"] == sample["phase_evidence"]["resources"]


def test_registered_benchmark_corpus_preserves_missing_prerequisites() -> None:
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
        hybrid_archive=None,
        hybrid_content_id=None,
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
    second = _benchmark_registered_native_corpus(arguments, profiles)
    assert first == second
    records = first["profiles"]
    assert isinstance(records, dict)
    record = records["native-hybrid-lifecycle:archived-layer-core"]
    assert isinstance(record, dict)
    assert record["status"] == "blocked-missing-input"
    assert record["missing_inputs"] == ["hybrid_archive", "hybrid_content_id"]
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
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        assert timeout == arguments.timeout
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
        record = _finalize_mandatory_cli_record(
            QualificationControls(scenario=profile_name),
            {
                "status": "passed",
                "mandatory_workflow_completion": "complete",
                "independent_scientific_certification": "passed",
            },
        )
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout=json.dumps(record), stderr=""
        )

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
        ("native-hybrid-lifecycle", {"hybrid_profile": "unit-box"}, {"u"}),
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
