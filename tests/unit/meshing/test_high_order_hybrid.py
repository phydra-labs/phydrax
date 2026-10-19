from typing import NamedTuple, TypedDict

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from examples.boundary_layer_core_mesh import mesh_layers_and_core, solve_diffusion
from phydrax._meshcore import meshcore_available, NativeExecutionScope
from phydrax.discretization import CellBlock, CellMesh, reference_cell_topology
from phydrax.discretization._cell_geometry import (
    coordinate_lagrange_element,
    require_scalar_coordinate_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import transition_nested_cell_geometry
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.meshing import HighOrderCurvingPolicy
from phydrax.meshing._curving import (
    admit_geometry_resources,
    node_owners,
    straight_geometry,
)
from phydrax.meshing._domain import compile_surface_domain
from phydrax.meshing._mixed_adaptation import adapt_mixed_mesh
from phydrax.meshing._surface_generation import generate_surface
from phydrax.meshing._topology_edit import assemble_topology_edit
from phydrax.meshing.providers._native_schedules import NativeSurfaceSchedule
from tests._support.coordinate_references import coordinate_reference_budget


# Retained empirical envelopes from a6827a9e5:test_high_order_hybrid.py.
# Degree-10 nodal inversion/gradient/trace/integration and mixed diffusion
# used 2e-8; affine reproduction and refined measure used 2e-9. Lower-degree
# rational, H1 node, and layer checks used 1e-12; the pyramid/tet trace used
# 1e-9 and the curved-prism solve used 1e-11. These are regression limits,
# not derived rounding bounds, and no acceptance value is widened.
HIGH_DEGREE_REGRESSION_ABS = 2e-8
AFFINE_REPRODUCTION_REGRESSION_ABS = 2e-9
LOW_DEGREE_REGRESSION_ABS = 1e-12
PYRAMID_TRACE_REGRESSION_ABS = 1e-9
CURVED_PRISM_SOLVE_REGRESSION_ABS = 1e-11
LAYER_DIFFUSION_REGRESSION_BOUND = 1e-10
# Authored unit-box classification and curving input allowances in the same
# source; these control the request and are not numerical test tolerances.
BOX_CLASSIFICATION_ALLOWANCE = 1e-8
CURVING_FIDELITY_ALLOWANCE = 1e-7


@pytest.mark.parametrize("degree", (2, 3, 4, 6, 10))
@pytest.mark.parametrize(
    "kind", ("triangle", "quadrilateral", "tetrahedron", "prism", "pyramid", "hexahedron")
)
def test_coordinate_map_reproduces_reference_and_nodal_values(
    kind: str, degree: int
) -> None:
    element = coordinate_lagrange_element(kind, degree)
    nodes = np.asarray(element.reference_nodes)
    values, _ = element.tabulate(nodes)
    np.testing.assert_allclose(
        values, np.eye(nodes.shape[0]), atol=HIGH_DEGREE_REGRESSION_ABS, rtol=0
    )
    center = np.mean(np.asarray(reference_cell_topology(kind).vertices), axis=0)[None]
    basis, gradients = element.tabulate(center)
    np.testing.assert_allclose(
        np.asarray(basis) @ nodes, center, atol=AFFINE_REPRODUCTION_REGRESSION_ABS, rtol=0
    )
    np.testing.assert_allclose(
        np.asarray(gradients)[0].T @ nodes,
        np.eye(center.shape[1]),
        atol=HIGH_DEGREE_REGRESSION_ABS,
        rtol=0,
    )


def _hybrid_mesh() -> CellMesh:
    points = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
            (0.5, 0.5, 2),
            (2, 0, 0),
            (2, 0, 1),
            (2, 1, 2),
        ],
        dtype=np.float64,
    )
    return CellMesh(
        points,
        (
            CellBlock(
                "hex",
                "hexahedron",
                np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32),
                global_ids=np.asarray([100], dtype=np.int64),
            ),
            CellBlock(
                "pyramid",
                "pyramid",
                np.asarray([[4, 5, 6, 7, 8]], dtype=np.int32),
                global_ids=np.asarray([101], dtype=np.int64),
            ),
            CellBlock(
                "prism",
                "prism",
                np.asarray([[1, 9, 2, 5, 10, 6]], dtype=np.int32),
                global_ids=np.asarray([102], dtype=np.int64),
            ),
            CellBlock(
                "tet",
                "tetrahedron",
                np.asarray([[5, 10, 6, 11]], dtype=np.int32),
                global_ids=np.asarray([103], dtype=np.int64),
            ),
        ),
    )


@pytest.mark.parametrize("degree", (2, 3, 4, 6, 10))
def test_mixed_faces_share_nodes_and_continuous_trace_under_orientation(
    degree: int,
) -> None:
    mesh = _hybrid_mesh()
    geometry = straight_geometry(mesh, degree)
    dims, rows = node_owners(mesh, geometry)
    elements, routes, coordinates = geometry.resolve(mesh)
    by_name = {
        block.name: (
            require_scalar_coordinate_element(element, "shared trace"),
            np.asarray(route)[0],
        )
        for block, element, route in zip(mesh.blocks, elements, routes, strict=True)
    }
    # Shared hex/pyramid quadrilateral, hex/prism quadrilateral, prism/tet triangle.
    for left, right, axis, position, expected in (
        ("hex", "pyramid", 2, 1.0, (degree + 1) ** 2),
        ("hex", "prism", 0, 1.0, (degree + 1) ** 2),
        ("prism", "tet", 2, 1.0, (degree + 1) * (degree + 2) // 2),
    ):
        shared = np.intersect1d(by_name[left][1], by_name[right][1])
        assert shared.size == expected, (left, right, degree)
        np.testing.assert_allclose(
            np.asarray(coordinates)[shared, axis],
            position,
            atol=LOW_DEGREE_REGRESSION_ABS,
        )
        assert np.all(dims[shared] <= 2)
        assert np.all(rows[shared] >= 0)
    # Arbitrary common face coefficients must give the same continuum trace,
    # not merely coincide at interpolation nodes.
    first, first_route = by_name["hex"]
    second, second_route = by_name["pyramid"]
    reference = np.asarray(
        [[0.17, 0.23, 1], [0.71, 0.49, 1], [0.31, 0.83, 1]], dtype=np.float64
    )
    pyramid_reference = reference.copy()
    pyramid_reference[:, 2] = 0
    coefficients = np.sin(np.arange(coordinates.shape[0], dtype=np.float64))
    left_values = np.asarray(first.tabulate(reference)[0]) @ coefficients[first_route]
    right_values = (
        np.asarray(second.tabulate(pyramid_reference)[0]) @ coefficients[second_route]
    )
    np.testing.assert_allclose(
        left_values, right_values, atol=HIGH_DEGREE_REGRESSION_ABS, rtol=0
    )


@pytest.mark.parametrize("degree", (2, 3, 4, 6, 10))
def test_mixed_high_order_coordinates_are_consumed_by_p1_integration(degree: int) -> None:
    mesh = _hybrid_mesh()
    geometry = straight_geometry(mesh, degree)
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            block.name: phx.discretization.lagrange_element(block.cell_kind, 1)
            for block in mesh.blocks
        },
    )
    space = phx.discretization.FiniteElementPlan(
        mesh, field, coordinate_spec=geometry
    ).prepare()
    blocks = space.evaluate_geometry("u", geometry.coordinates)
    integrated_volume = sum(float(np.sum(np.asarray(block.measure))) for block in blocks)
    # Independent geometric volumes: cube, pyramid, triangular prism, tetrahedron.
    np.testing.assert_allclose(
        integrated_volume,
        1.0 + 1.0 / 3.0 + 1.0 / 2.0 + 1.0 / 6.0,
        atol=HIGH_DEGREE_REGRESSION_ABS,
        rtol=0,
    )


class _CoordinateCapacities(TypedDict, total=False):
    maximum_nodes: int
    maximum_entries: int
    maximum_bernstein_nodes: int


@pytest.mark.parametrize("degree", (2, 3, 4, 6, 10))
@pytest.mark.parametrize(
    ("budget", "message"),
    (
        ({"maximum_nodes": 1}, r"admit_geometry_resources\.maximum_nodes"),
        ({"maximum_entries": 1}, r"admit_geometry_resources\.maximum_entries"),
        (
            {"maximum_bernstein_nodes": 1},
            r"admit_geometry_resources\.maximum_bernstein_nodes",
        ),
    ),
    ids=("coordinate-nodes", "reference-tabulation", "continuous-validity"),
)
def test_resource_admission_refuses_before_coordinate_construction(
    degree: int,
    budget: _CoordinateCapacities,
    message: str,
) -> None:
    mesh = _hybrid_mesh()
    with pytest.raises(ValueError, match=message):
        straight_geometry(mesh, degree, **budget)


def test_tabulation_budget_cannot_be_evaded_by_partitioning_cell_blocks() -> None:
    points = np.asarray(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (2.0, 1.0),
        ]
    )
    blocks = tuple(
        CellBlock(
            name,
            "triangle",
            np.asarray([vertices], dtype=np.int32),
            global_ids=np.asarray([identifier], dtype=np.int64),
        )
        for name, vertices, identifier in (
            ("left", (0, 1, 2), 100),
            ("right", (3, 4, 5), 101),
        )
    )
    policy = HighOrderCurvingPolicy(
        fidelity_tolerance=CURVING_FIDELITY_ALLOWANCE,
        degree=2,
        maximum_tabulation_entries=300,
    )
    # Each independently consumed block fits; their simultaneously retained
    # gradient/inverse tables and construction scratch do not.
    for block in blocks:
        admit_geometry_resources(
            CellMesh(points, (block,)),
            policy.degree,
            policy.maximum_geometry_nodes,
            policy.maximum_tabulation_entries,
            policy.validity.maximum_bernstein_nodes,
        )
    with pytest.raises(ValueError, match=r"admit_geometry_resources\.maximum_entries"):
        admit_geometry_resources(
            CellMesh(points, blocks),
            policy.degree,
            policy.maximum_geometry_nodes,
            policy.maximum_tabulation_entries,
            policy.validity.maximum_bernstein_nodes,
        )


@pytest.mark.parametrize("degree", (1, 5, 11))
def test_unqualified_curving_degree_is_refused(degree: int) -> None:
    with pytest.raises(ValueError, match=r"HighOrderCurvingPolicy\.degree"):
        HighOrderCurvingPolicy(
            fidelity_tolerance=CURVING_FIDELITY_ALLOWANCE, degree=degree
        )


def test_restricted_rational_pyramid_preserves_source_map_and_chain_rule() -> None:
    source = coordinate_lagrange_element("pyramid", 3)
    matrix = np.diag([0.5, 0.5, 1.0])
    offset = np.asarray([0.25, 0.25, 0.0])
    restricted = RestrictedCellGeometryElement(source, "pyramid", matrix, offset)
    points = np.asarray([[0.4, 0.4, 0.2], [0.49, 0.51, 0.8]], dtype=np.float64)
    values, gradients = restricted.tabulate(points)
    # The reference pyramid space contains this non-polynomial function.
    nodes = np.asarray(source.reference_nodes)
    scale = 1.0 - nodes[:, 2]
    coefficients = np.zeros(nodes.shape[0], dtype=np.float64)
    regular = scale > 0
    coefficients[regular] = (
        (nodes[regular, 0] - 0.5 * nodes[regular, 2])
        * (nodes[regular, 1] - 0.5 * nodes[regular, 2])
        / scale[regular]
    )
    mapped = points @ matrix.T + offset
    a = mapped[:, 0] - 0.5 * mapped[:, 2]
    b = mapped[:, 1] - 0.5 * mapped[:, 2]
    s = 1.0 - mapped[:, 2]
    expected = a * b / s
    analytic_gradient = (
        np.column_stack((b / s, a / s, a * b / s**2 - 0.5 * (a + b) / s)) @ matrix
    )
    np.testing.assert_allclose(
        np.asarray(values) @ coefficients, expected, atol=LOW_DEGREE_REGRESSION_ABS
    )
    actual_gradient = np.einsum("qnd,n->qd", np.asarray(gradients), coefficients)
    np.testing.assert_allclose(
        actual_gradient, analytic_gradient, atol=LOW_DEGREE_REGRESSION_ABS
    )


def test_mixed_refinement_preserves_curved_map_in_manufactured_diffusion_solve() -> None:
    mesh = _hybrid_mesh()
    layout = straight_geometry(mesh, 2)
    coordinates = np.asarray(layout.coordinates).copy()
    # An analytic globally injective shear, with determinant exactly one.
    coordinates[:, 1] += 0.05 * coordinates[:, 0] * coordinates[:, 2]
    source = phx.discretization.CellGeometrySpec(
        dict(zip(layout.block_names, layout.elements, strict=True)),
        dict(zip(layout.block_names, layout.geometry_dofs, strict=True)),
        coordinates,
    )
    outcome = adapt_mixed_mesh(
        mesh, refine_cell_ids=np.asarray([100, 101, 102, 103], dtype=np.int64)
    )
    with coordinate_reference_budget() as budget:
        target, _, _ = assemble_topology_edit(
            mesh, outcome.edit, numeric_version="refined", budget=budget
        )
    transition = transition_nested_cell_geometry(
        mesh,
        source,
        target,
        phx.discretization.CellGeometrySpec.affine(target),
        refinement=outcome.edit.refinement,
        coarsening=outcome.edit.coarsening,
    )
    target = target.with_coordinates(
        transition.vertex_coordinates, numeric_version="curved"
    )
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            block.name: phx.discretization.lagrange_element(block.cell_kind, 1)
            for block in target.blocks
        },
    )
    space = phx.discretization.FiniteElementPlan(
        target, field, coordinate_spec=transition.geometry
    ).prepare()
    blocks = space.evaluate_geometry("u", transition.geometry.coordinates)
    volume = sum(float(np.sum(np.asarray(block.measure))) for block in blocks)
    np.testing.assert_allclose(
        volume, 2.0, atol=AFFINE_REPRODUCTION_REGRESSION_ABS, rtol=0
    )
    form = phx.equations.FiniteElementForm(
        "curved-mixed-diffusion", "u", (phx.equations.DiffusionAction("u", 1.0),)
    )

    def exact(points: Array) -> Array:
        return points[..., 0]

    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=exact,
    )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(operator, rhs)
    assert np.all(np.asarray(solved.successful))
    solution = np.asarray(problem.expand(solved.value))
    np.testing.assert_allclose(
        solution,
        np.asarray(target.coordinates)[:, 0],
        atol=HIGH_DEGREE_REGRESSION_ABS,
        rtol=0,
    )


class _HybridBoundary(NamedTuple):
    contract: phx.SpatialCoordinateContract
    projection: phx.geometry.AbstractBRepProjection
    source: phx.geometry.MeshingDomainBoundarySource
    domain: phx.geometry.PiecewiseLinearDomain


class _HybridSource(NamedTuple):
    initial: phx.meshing.CellMeshingResult
    layers: phx.meshing.BoundaryLayerMesh
    boundary: _HybridBoundary
    association: phx.meshing.GeometryAssociation


class _CurvedHybrid(NamedTuple):
    curved: phx.meshing.HighOrderCurvingResult
    published: phx.meshing.CellMeshingResult


@pytest.fixture(scope="module")
def hybrid_boundary() -> _HybridBoundary:
    if not meshcore_available():
        pytest.skip("Native layer/core workflow requires compiled meshcore.")
    contract = phx.SpatialCoordinateContract.si()
    model = phx.geometry.brep_box((0, 0, 0), (1, 1, 1), coordinate_contract=contract)
    projection = phx.geometry.prepare_brep_projection(model)
    source_domain = phx.geometry.MeshingDomain.from_brep(model)
    scope = phx.meshing.MeshingScope(
        source_domain.source_id,
        source_domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        source_domain.entity_set_id(2),
        np.asarray(source_domain.source_indices[2], dtype=np.int64),
    )
    surface_spec = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 1.0, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    limits = surface_spec.limits
    with NativeExecutionScope(limits.native_limits()) as execution:
        with execution.batch(
            limits.maximum_work_units,
            queries_upper=limits.maximum_geometry_queries,
            bytes_upper=limits.maximum_scratch_bytes,
        ) as ticket:
            owner = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with owner.activate():
                compiled = compile_surface_domain(
                    source_domain, surface_spec, budget=owner
                )
        with owner.retained_storage(lambda count: execution.batch(0, bytes_upper=count)):
            construction = generate_surface(
                compiled,
                NativeSurfaceSchedule(),
                limits,
                0.0,
                scope=execution,
                budget=owner,
            )
    source = phx.geometry.MeshingDomainBoundarySource(
        source_domain,
        tuple(compiled.patches.tolist()),
        chart_triangulations=construction.chart_triangulations,
    )
    # Independent declared unit-box boundary, not the generated mesh boundary.
    vertices = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ),
        dtype=np.float64,
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    triangles = np.asarray(
        [row for a, b, c, d in faces for row in ((a, b, c), (a, c, d))], dtype=np.int64
    )
    domain = phx.geometry.PiecewiseLinearDomain(
        vertices,
        triangles,
        np.tile(np.asarray([0, -1], dtype=np.int64), (12, 1)),
        ("fluid",),
        source_id="declared-unit-box",
    )
    return _HybridBoundary(contract, projection, source, domain)


@pytest.fixture(scope="module")
def hybrid_source(hybrid_boundary: _HybridBoundary) -> _HybridSource:
    initial, layers = mesh_layers_and_core()
    association = phx.meshing.associate_mesh_vertices(
        initial.mesh,
        hybrid_boundary.projection,
        policy=phx.meshing.AssociationPropagationPolicy(
            classification_tolerance=BOX_CLASSIFICATION_ALLOWANCE
        ),
    )
    return _HybridSource(initial, layers, hybrid_boundary, association)


def test_native_layer_curving_refuses_foreign_column_coordinates(
    hybrid_source: _HybridSource,
) -> None:
    initial, layers, boundary, association = hybrid_source
    mesh = initial.mesh
    foreign_layers = phx.meshing.prepare_boundary_layers(
        layers.source_wall,
        phx.meshing.BoundaryLayerControl(
            layers.control.wall_scope,
            phx.meshing.LayerSchedule.geometric(2, 0.15, growth_rate=1.0),
            route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        ),
    )
    with pytest.raises(ValueError, match=r"source_layer_cells\.mesh"):
        phx.meshing.curve_cell_mesh(
            mesh,
            association,
            boundary.projection,
            policy=phx.meshing.HighOrderCurvingPolicy(
                fidelity_tolerance=CURVING_FIDELITY_ALLOWANCE,
                degree=2,
                relaxation_rounds=0,
            ),
            source=boundary.source,
            domain=boundary.domain,
            layers=foreign_layers,
            cell_regions=np.zeros(
                sum(block.cell_count for block in mesh.blocks), dtype=np.int64
            ),
        )


@pytest.fixture(scope="module")
def curved_hybrid(hybrid_source: _HybridSource) -> _CurvedHybrid:
    initial, layers, boundary, association = hybrid_source
    mesh = initial.mesh
    curved = phx.meshing.curve_cell_mesh(
        mesh,
        association,
        boundary.projection,
        policy=phx.meshing.HighOrderCurvingPolicy(
            fidelity_tolerance=CURVING_FIDELITY_ALLOWANCE, degree=2, relaxation_rounds=0
        ),
        source=boundary.source,
        domain=boundary.domain,
        layers=layers,
        cell_regions=np.zeros(
            sum(block.cell_count for block in mesh.blocks), dtype=np.int64
        ),
    )
    prepared_source = phx.meshing.certify_cell_mesh(
        mesh,
        boundary.contract,
        geometry=curved.geometry,
        patches=initial.patches,
        zones=initial.zones,
        labels=initial.labels,
        attributes=initial.attributes,
        associations=(association,),
    )
    return _CurvedHybrid(curved, prepared_source)


def _assert_curved_hybrid(curved_hybrid: _CurvedHybrid) -> None:
    curved = curved_hybrid.curved
    assert curved.status is phx.meshing.HighOrderCurvingStatus.CURVED
    assert curved.evidence.accepted and not curved.evidence.certification_failures
    assert curved_hybrid.published.audit.passed
    np.testing.assert_array_equal(
        curved_hybrid.published.geometry.coordinates, curved.geometry.coordinates
    )


def test_native_layer_curving_publishes_certified_complete_geometry(
    curved_hybrid: _CurvedHybrid,
) -> None:
    _assert_curved_hybrid(curved_hybrid)


def test_curved_hybrid_guard_rejects_affine_publication(
    hybrid_source: _HybridSource, curved_hybrid: _CurvedHybrid
) -> None:
    corrupted = _CurvedHybrid(curved_hybrid.curved, hybrid_source.initial)
    with pytest.raises(AssertionError):
        _assert_curved_hybrid(corrupted)


@pytest.fixture(scope="module")
def adapted_hybrid(
    hybrid_source: _HybridSource, curved_hybrid: _CurvedHybrid
) -> phx.meshing.MeshAdaptationResult:
    initial, _, boundary, _ = hybrid_source
    mesh = initial.mesh
    core_ids = np.concatenate(
        [
            np.asarray(block.global_ids)
            for block in mesh.blocks
            if block.cell_kind == "tetrahedron"
        ]
    )
    layer_marker = next(
        attribute for attribute in initial.attributes if attribute.name == "layer_index"
    )
    column_marker = next(
        attribute for attribute in initial.attributes if attribute.name == "layer_column"
    )
    column_by_id = dict(
        zip(
            np.asarray(column_marker.scope.entity_ids).tolist(),
            np.asarray(column_marker.values).tolist(),
            strict=True,
        )
    )
    layer_ids = np.asarray(layer_marker.scope.entity_ids)
    layer_values = np.asarray(layer_marker.values)
    active = layer_values >= 0
    columns = phx.meshing.MixedLayerColumns(
        layer_ids[active],
        np.asarray(
            [column_by_id[identifier] for identifier in layer_ids[active]], dtype=np.int64
        ),
        layer_values[active],
        hard_first_thickness=True,
        axial_refinement=False,
    )
    adapted = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            curved_hybrid.published,
            phx.meshing.MarkedMeshAdaptation(core_ids, layer_columns=columns),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
                association_transfer=phx.meshing.BRepAssociationTransfer(
                    boundary.projection,
                    policy=phx.meshing.AssociationPropagationPolicy(
                        classification_tolerance=BOX_CLASSIFICATION_ALLOWANCE
                    ),
                ),
            ),
        )
    )
    return adapted


def test_native_layer_core_adaptation_consumes_curved_map_in_diffusion(
    adapted_hybrid: phx.meshing.MeshAdaptationResult,
) -> None:
    assert solve_diffusion(adapted_hybrid.target) < LAYER_DIFFUSION_REGRESSION_BOUND


def _assert_layer_intervals(target: phx.meshing.CellMeshingResult) -> None:
    ancestry = next(
        attribute for attribute in target.attributes if attribute.name == "layer_index"
    )
    assert set(np.asarray(ancestry.values).tolist()) == {-1, 0, 1}
    layer_indices = dict(
        zip(
            np.asarray(ancestry.scope.entity_ids).tolist(),
            np.asarray(ancestry.values).tolist(),
            strict=True,
        )
    )
    points = np.asarray(target.mesh.coordinates)
    intervals = []
    starts = []
    expected_starts = []
    for block in target.mesh.blocks:
        for identifier, corners in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        ):
            layer = layer_indices[identifier]
            if layer >= 0:
                heights = points[corners, 2]
                intervals.append(np.max(heights) - np.min(heights))
                starts.append(np.min(heights))
                expected_starts.append(layer * 0.1)
    np.testing.assert_allclose(intervals, 0.1, atol=LOW_DEGREE_REGRESSION_ABS, rtol=0)
    np.testing.assert_allclose(
        starts, expected_starts, atol=LOW_DEGREE_REGRESSION_ABS, rtol=0
    )


def test_native_layer_core_adaptation_preserves_hard_first_intervals(
    adapted_hybrid: phx.meshing.MeshAdaptationResult,
) -> None:
    _assert_layer_intervals(adapted_hybrid.target)


def _assert_unit_box_boundary(boundary: _HybridBoundary) -> None:
    vertices = np.asarray(boundary.domain.vertices)
    np.testing.assert_array_equal(np.min(vertices, axis=0), np.zeros(3, dtype=np.float64))
    np.testing.assert_array_equal(np.max(vertices, axis=0), np.ones(3, dtype=np.float64))
    assert boundary.domain.facets.shape == (12, 3)


def test_native_hybrid_source_has_authored_boundary_and_layer_intervals(
    hybrid_source: _HybridSource,
) -> None:
    _assert_unit_box_boundary(hybrid_source.boundary)
    _assert_layer_intervals(hybrid_source.initial)


def test_hybrid_boundary_guard_rejects_changed_unit_box(
    hybrid_boundary: _HybridBoundary,
) -> None:
    domain = hybrid_boundary.domain
    corrupted_domain = phx.geometry.PiecewiseLinearDomain(
        np.asarray(domain.vertices) * np.asarray((1.0, 1.0, 2.0), dtype=np.float64),
        domain.facets,
        domain.facet_regions,
        domain.region_ids,
        source_id="fault-stretched-box",
    )
    corrupted = hybrid_boundary._replace(domain=corrupted_domain)
    with pytest.raises(AssertionError):
        _assert_unit_box_boundary(corrupted)


def test_hybrid_source_guard_rejects_changed_first_interval(
    hybrid_source: _HybridSource,
) -> None:
    source = hybrid_source.initial
    changed = source.mesh.with_coordinates(
        source.mesh.coordinates * jnp.asarray((1.0, 1.0, 1.5), dtype=jnp.float64),
        numeric_version="fault-layer-source",
    )
    corrupted = eqx.tree_at(lambda result: result.mesh, source, changed)
    with pytest.raises(AssertionError):
        _assert_layer_intervals(corrupted)


def test_adapted_hybrid_guard_rejects_changed_first_interval(
    adapted_hybrid: phx.meshing.MeshAdaptationResult,
) -> None:
    target = adapted_hybrid.target
    changed = target.mesh.with_coordinates(
        target.mesh.coordinates * jnp.asarray((1.0, 1.0, 1.5), dtype=jnp.float64),
        numeric_version="fault-layer-target",
    )
    corrupted = eqx.tree_at(lambda result: result.mesh, target, changed)
    with pytest.raises(AssertionError):
        _assert_layer_intervals(corrupted)


@pytest.mark.parametrize("degree", (2, 3, 4))
def test_prism_h1_shared_face_reproduces_quadratic_under_reversed_orientation(
    degree: int,
) -> None:
    points = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 0),
            (1, 1, 1),
        ],
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "prisms",
                "prism",
                np.asarray([[0, 1, 2, 3, 4, 5], [1, 6, 2, 4, 7, 5]], dtype=np.int32),
            ),
        ),
    )
    element = phx.discretization.lagrange_element("prism", degree)
    space = phx.discretization.FiniteElementPlan(
        mesh, phx.discretization.FiniteElementFieldSpec("u", element)
    ).prepare()
    routes = np.asarray(space.dof_maps[0].cell_dofs[0])
    coordinates = np.asarray(space.dof_maps[0].dof_coordinates)
    assert coordinates.shape[0] == 2 * element.local_dof_count - (degree + 1) ** 2
    assert np.intersect1d(routes[0], routes[1]).size == (degree + 1) ** 2
    coefficients = np.sum(coordinates**2, axis=1)
    first = (
        np.asarray(
            element.tabulate(np.asarray([[0.37, 0.63, 0.42]], dtype=np.float64))[0]
        )
        @ coefficients[routes[0]]
    )
    second = (
        np.asarray(element.tabulate(np.asarray([[0, 0.63, 0.42]], dtype=np.float64))[0])
        @ coefficients[routes[1]]
    )
    np.testing.assert_allclose(
        first, 0.37**2 + 0.63**2 + 0.42**2, atol=LOW_DEGREE_REGRESSION_ABS, rtol=0
    )
    np.testing.assert_allclose(second, first, atol=LOW_DEGREE_REGRESSION_ABS, rtol=0)


@pytest.mark.parametrize("degree", (2, 3, 4, 6, 10))
def test_mixed_h1_consumes_quadratic_in_all_volume_families_without_vertex_downgrade(
    degree: int,
) -> None:
    mesh = _hybrid_mesh()
    elements = {
        block.name: phx.discretization.lagrange_element(block.cell_kind, degree)
        for block in mesh.blocks
    }
    space = phx.discretization.FiniteElementPlan(
        mesh, phx.discretization.FiniteElementFieldSpec("u", elements)
    ).prepare()
    coefficients = np.sum(np.asarray(space.dof_maps[0].dof_coordinates) ** 2, axis=1)
    for index, block in enumerate(mesh.blocks):
        nodes = np.asarray(
            reference_cell_topology(block.cell_kind).vertices, dtype=np.float64
        )
        reference = np.mean(nodes, axis=0)[None]
        basis = np.asarray(elements[block.name].tabulate(reference)[0])
        route = np.asarray(space.dof_maps[0].cell_dofs[index])[0]
        actual = basis @ coefficients[route]
        physical = (
            np.asarray(
                coordinate_lagrange_element(block.cell_kind, 1).tabulate(reference)[0]
            )
            @ np.asarray(mesh.coordinates)[np.asarray(block.vertices)[0]]
        )
        np.testing.assert_allclose(
            actual, np.sum(physical**2, axis=1), atol=LOW_DEGREE_REGRESSION_ABS, rtol=0
        )


@pytest.mark.parametrize("degree", (3, 4, 6, 10))
def test_pyramid_tet_triangular_trace_shares_arbitrary_high_order_coefficients(
    degree: int,
) -> None:
    points = np.asarray(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 1), (0.5, -1, 0.5)],
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "pyramid",
                "pyramid",
                np.asarray([[0, 1, 2, 3, 4]], dtype=np.int32),
                global_ids=np.asarray([71], dtype=np.int64),
            ),
            CellBlock(
                "tet",
                "tetrahedron",
                np.asarray([[0, 1, 4, 5]], dtype=np.int32),
                global_ids=np.asarray([79], dtype=np.int64),
            ),
        ),
    )
    elements = {
        block.name: phx.discretization.lagrange_element(block.cell_kind, degree)
        for block in mesh.blocks
    }
    space = phx.discretization.FiniteElementPlan(
        mesh, phx.discretization.FiniteElementFieldSpec("u", elements)
    ).prepare()
    routes = [np.asarray(route)[0] for route in space.dof_maps[0].cell_dofs]
    assert np.intersect1d(routes[0], routes[1]).size == (degree + 1) * (degree + 2) // 2
    coefficients = np.sin(np.arange(space.dof_maps[0].global_dof_count, dtype=np.float64))
    first = (
        np.asarray(
            elements["pyramid"].tabulate(
                np.asarray([[0.575, 0.225, 0.45]], dtype=np.float64)
            )[0]
        )
        @ coefficients[routes[0]]
    )
    second = (
        np.asarray(
            elements["tet"].tabulate(np.asarray([[0.35, 0.45, 0]], dtype=np.float64))[0]
        )
        @ coefficients[routes[1]]
    )
    np.testing.assert_allclose(first, second, atol=PYRAMID_TRACE_REGRESSION_ABS, rtol=0)


def test_curved_prism_field_nodes_and_physical_dirichlet_solve_use_complete_map() -> None:
    coordinate = coordinate_lagrange_element("prism", 2)
    reference = np.asarray(reference_cell_topology("prism").vertices, dtype=np.float64)
    vertices = reference.copy()
    vertices[:, 1] += 0.2 * vertices[:, 0] ** 2
    mesh = CellMesh(
        vertices,
        (CellBlock("prism", "prism", np.asarray([[0, 1, 2, 3, 4, 5]], dtype=np.int32)),),
    )
    coefficients = np.asarray(coordinate.reference_nodes).copy()
    coefficients[:, 1] += 0.2 * coefficients[:, 0] ** 2
    geometry = phx.discretization.CellGeometrySpec(
        {"prism": coordinate},
        {"prism": np.arange(coordinate.local_dof_count, dtype=np.int32)[None]},
        coefficients,
    )
    element = phx.discretization.lagrange_element("prism", 3)
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec("u", element),
        coordinate_spec=geometry,
    ).prepare()
    dof_map = space.dof_maps[0]
    route = np.asarray(dof_map.cell_dofs[0])[0]
    expected = np.asarray(element.reference_nodes).copy()
    expected[:, 1] += 0.2 * expected[:, 0] ** 2
    np.testing.assert_allclose(
        np.asarray(dof_map.dof_coordinates)[route],
        expected,
        atol=LOW_DEGREE_REGRESSION_ABS,
    )
    np.testing.assert_allclose(
        np.asarray(dof_map.evaluate_coordinates(mesh, geometry.coordinates))[route],
        expected,
        atol=LOW_DEGREE_REGRESSION_ABS,
    )

    def exact(points: Array) -> Array:
        return jnp.sum(points, axis=-1)

    form = phx.equations.FiniteElementForm(
        "curved-prism-harmonic", "u", (phx.equations.DiffusionAction("u", 1.0),)
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=exact,
    )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(operator, rhs)
    assert np.all(np.asarray(solved.successful))
    np.testing.assert_allclose(
        problem.expand(solved.value),
        exact(dof_map.dof_coordinates),
        atol=CURVED_PRISM_SOLVE_REGRESSION_ABS,
        rtol=0,
    )


def test_shared_h1_physical_nodes_refuse_inconsistent_incident_geometry() -> None:
    points = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 0),
            (1, 1, 1),
        ],
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "prisms",
                "prism",
                np.asarray([[0, 1, 2, 3, 4, 5], [1, 6, 2, 4, 7, 5]], dtype=np.int32),
            ),
        ),
    )
    coordinate = coordinate_lagrange_element("prism", 2)
    weights = np.asarray(
        coordinate_lagrange_element("prism", 1).tabulate(coordinate.reference_nodes)[0]
    )
    values = (weights @ points[np.asarray(mesh.blocks[0].vertices)]).reshape((-1, 3))
    routes = np.arange(2 * coordinate.local_dof_count, dtype=np.int32).reshape((2, -1))
    # Alter one incident map at the same authored shared vertex.
    values[coordinate.local_dof_count + coordinate.entity_dofs[0][0][0], 1] += 0.001
    geometry = phx.discretization.CellGeometrySpec(
        {"prisms": coordinate},
        {"prisms": routes},
        values,
    )
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("prism", 2)
    )
    with pytest.raises(ValueError, match=r"finite_element\.dof_coordinates"):
        phx.discretization.FiniteElementPlan(
            mesh, field, coordinate_spec=geometry
        ).prepare()
