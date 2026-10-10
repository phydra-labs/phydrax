import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization import CellBlock, CellMesh, reference_cell_topology
from phydrax.discretization._cell_geometry import (
    _require_scalar_coordinate_element,
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import transition_nested_cell_geometry
from phydrax.meshing import HighOrderCurvingPolicy
from phydrax.meshing._curving import (
    _admit_geometry_resources,
    _node_owners,
    _straight_geometry,
)
from phydrax.meshing._mixed_adaptation import adapt_mixed_mesh
from phydrax.meshing._topology_edit import assemble_topology_edit


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
    np.testing.assert_allclose(values, np.eye(nodes.shape[0]), atol=2e-8, rtol=0)
    center = np.mean(np.asarray(reference_cell_topology(kind).vertices), axis=0)[None]
    basis, gradients = element.tabulate(center)
    np.testing.assert_allclose(np.asarray(basis) @ nodes, center, atol=2e-9, rtol=0)
    np.testing.assert_allclose(
        np.asarray(gradients)[0].T @ nodes, np.eye(center.shape[1]), atol=2e-8, rtol=0
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
    geometry = _straight_geometry(mesh, degree)
    dims, rows = _node_owners(mesh, geometry)
    elements, routes, coordinates = geometry.resolve(mesh)
    by_name = {
        block.name: (
            _require_scalar_coordinate_element(element, "shared trace"),
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
            np.asarray(coordinates)[shared, axis], position, atol=1e-12
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
    np.testing.assert_allclose(left_values, right_values, atol=2e-8, rtol=0)


@pytest.mark.parametrize("degree", (2, 3, 4, 6, 10))
def test_mixed_high_order_coordinates_are_consumed_by_p1_integration(degree: int) -> None:
    mesh = _hybrid_mesh()
    geometry = _straight_geometry(mesh, degree)
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
        atol=2e-8,
        rtol=0,
    )


@pytest.mark.parametrize("degree", (2, 3, 4, 6, 10))
@pytest.mark.parametrize(
    ("budget", "message"),
    (
        ({"maximum_nodes": 1}, "node resource limit"),
        ({"maximum_entries": 1}, "tabulation resource limit"),
        ({"maximum_bernstein_nodes": 1}, "certificate resource limit"),
    ),
    ids=("coordinate-nodes", "reference-tabulation", "continuous-validity"),
)
def test_resource_admission_refuses_before_coordinate_construction(
    degree: int,
    budget: dict[str, int],
    message: str,
) -> None:
    mesh = _hybrid_mesh()
    with pytest.raises(ValueError, match=message):
        _straight_geometry(mesh, degree, **budget)


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
    policy = HighOrderCurvingPolicy(degree=2, maximum_tabulation_entries=300)
    # Each independently consumed block fits; their simultaneously retained
    # gradient/inverse tables and construction scratch do not.
    for block in blocks:
        _admit_geometry_resources(
            CellMesh(points, (block,)),
            policy.degree,
            policy.maximum_geometry_nodes,
            policy.maximum_tabulation_entries,
            policy.validity.maximum_bernstein_nodes,
        )
    with pytest.raises(ValueError, match="tabulation resource limit"):
        _admit_geometry_resources(
            CellMesh(points, blocks),
            policy.degree,
            policy.maximum_geometry_nodes,
            policy.maximum_tabulation_entries,
            policy.validity.maximum_bernstein_nodes,
        )


@pytest.mark.parametrize("degree", (1, 5, 11))
def test_unqualified_curving_degree_is_refused(degree: int) -> None:
    with pytest.raises(ValueError, match="geometry degrees"):
        HighOrderCurvingPolicy(degree=degree)


def test_restricted_rational_pyramid_preserves_source_map_and_chain_rule() -> None:
    source = coordinate_lagrange_element("pyramid", 3)
    matrix = np.diag([0.5, 0.5, 1.0])
    offset = np.asarray([0.25, 0.25, 0.0])
    restricted = RestrictedCellGeometryElement(source, "pyramid", matrix, offset)
    points = np.asarray([[0.4, 0.4, 0.2], [0.49, 0.51, 0.8]], dtype=np.float64)
    values, gradients = restricted.tabulate(points)
    # The canonical pyramid space contains this non-polynomial function.
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
    np.testing.assert_allclose(np.asarray(values) @ coefficients, expected, atol=1e-12)
    actual_gradient = np.einsum("qnd,n->qd", np.asarray(gradients), coefficients)
    np.testing.assert_allclose(actual_gradient, analytic_gradient, atol=1e-12)


def test_mixed_refinement_preserves_curved_map_in_manufactured_diffusion_solve() -> None:
    mesh = _hybrid_mesh()
    layout = _straight_geometry(mesh, 2)
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
    target, _, _ = assemble_topology_edit(mesh, outcome.edit, numeric_version="refined")
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
    np.testing.assert_allclose(volume, 2.0, atol=2e-9, rtol=0)
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
        solution, np.asarray(target.coordinates)[:, 0], atol=2e-8, rtol=0
    )


def test_native_layers_core_curving_mixed_adaptation_and_diffusion() -> None:
    from examples.boundary_layer_core_mesh import mesh_layers_and_core, solve_diffusion
    from phydrax._meshcore import meshcore_available
    from phydrax.geometry._mesh_certificates import PiecewiseLinearDomain
    from phydrax.geometry._meshing_domain import (
        MeshingDomain,
        MeshingDomainBoundarySource,
    )
    from phydrax.meshing._domain import compile_surface_domain
    from phydrax.meshing._surface_generation import generate_surface
    from phydrax.meshing.providers._native_options import NativeSurfaceSchedule

    if not meshcore_available():
        pytest.skip("Native layer/core workflow requires compiled meshcore.")
    initial, layers = mesh_layers_and_core()
    contract = phx.SpatialCoordinateContract.si()
    model = phx.geometry.brep_box((0, 0, 0), (1, 1, 1), coordinate_contract=contract)
    projection = phx.geometry.prepare_brep_projection(model)
    source_domain = MeshingDomain.from_brep(model)
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
    compiled = compile_surface_domain(source_domain, surface_spec)
    construction = generate_surface(
        compiled, NativeSurfaceSchedule(), surface_spec.limits, 0.0
    )
    source = MeshingDomainBoundarySource(
        source_domain,
        tuple(compiled.patches.tolist()),
        chart_triangulations=construction.chart_triangulations,
    )
    # Independent declared unit-box boundary, not the generated mesh boundary.
    vertices = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
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
    domain = PiecewiseLinearDomain(
        vertices,
        triangles,
        np.tile(np.asarray([0, -1], dtype=np.int64), (12, 1)),
        ("fluid",),
        source_id="declared-unit-box",
    )
    mesh = initial.mesh
    association = phx.meshing.associate_mesh_vertices(
        mesh,
        projection,
        policy=phx.meshing.AssociationPropagationPolicy(classification_tolerance=1e-8),
    )
    foreign_layers = phx.meshing.prepare_boundary_layers(
        layers.source_wall,
        phx.meshing.BoundaryLayerControl(
            layers.control.wall_scope,
            phx.meshing.LayerSchedule.geometric(2, 0.15, growth_rate=1.0),
            route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        ),
    )
    with pytest.raises(ValueError, match="prepared source column coordinates"):
        phx.meshing.curve_cell_mesh(
            mesh,
            association,
            projection,
            policy=phx.meshing.HighOrderCurvingPolicy(degree=2, relaxation_rounds=0),
            source=source,
            domain=domain,
            layers=foreign_layers,
            cell_regions=np.zeros(
                sum(block.cell_count for block in mesh.blocks), dtype=np.int64
            ),
        )
    curved = phx.meshing.curve_cell_mesh(
        mesh,
        association,
        projection,
        policy=phx.meshing.HighOrderCurvingPolicy(degree=2, relaxation_rounds=0),
        source=source,
        domain=domain,
        layers=layers,
        cell_regions=np.zeros(
            sum(block.cell_count for block in mesh.blocks), dtype=np.int64
        ),
    )
    assert curved.status is phx.meshing.HighOrderCurvingStatus.CURVED
    assert curved.evidence.accepted and not curved.evidence.certification_failures
    prepared_source = phx.meshing.certify_cell_mesh(
        mesh,
        contract,
        geometry=curved.geometry,
        patches=initial.patches,
        zones=initial.zones,
        labels=initial.labels,
        attributes=initial.attributes,
        associations=(association,),
    )
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
            prepared_source,
            phx.meshing.MarkedMeshAdaptation(core_ids, layer_columns=columns),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
                association_transfer=phx.meshing.BRepAssociationTransfer(
                    projection,
                    policy=phx.meshing.AssociationPropagationPolicy(
                        classification_tolerance=1e-8
                    ),
                ),
            ),
        )
    )
    assert solve_diffusion(adapted.target) < 1e-10
    ancestry = next(
        attribute
        for attribute in adapted.target.attributes
        if attribute.name == "layer_index"
    )
    assert set(np.asarray(ancestry.values).tolist()) == {-1, 0, 1}
    layer_indices = dict(
        zip(
            np.asarray(ancestry.scope.entity_ids).tolist(),
            np.asarray(ancestry.values).tolist(),
            strict=True,
        )
    )
    points = np.asarray(adapted.target.mesh.coordinates)
    intervals = []
    starts = []
    expected_starts = []
    for block in adapted.target.mesh.blocks:
        for identifier, corners in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        ):
            layer = layer_indices[identifier]
            if layer >= 0:
                heights = points[corners, 2]
                intervals.append(np.max(heights) - np.min(heights))
                starts.append(np.min(heights))
                expected_starts.append(layer * 0.1)
    np.testing.assert_allclose(intervals, 0.1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(starts, expected_starts, atol=1e-12, rtol=0)


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
    np.testing.assert_allclose(first, 0.37**2 + 0.63**2 + 0.42**2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(second, first, atol=1e-12, rtol=0)


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
            actual, np.sum(physical**2, axis=1), atol=1e-12, rtol=0
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
    np.testing.assert_allclose(first, second, atol=1e-9, rtol=0)


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
        np.asarray(dof_map.dof_coordinates)[route], expected, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(dof_map.evaluate_coordinates(mesh, geometry.coordinates))[route],
        expected,
        atol=1e-12,
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
        atol=1e-11,
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
    with pytest.raises(ValueError):
        phx.discretization.FiniteElementPlan(
            mesh, field, coordinate_spec=geometry
        ).prepare()
