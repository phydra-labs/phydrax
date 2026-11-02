from __future__ import annotations

from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._interpolation import TMesh2D
from phydrax._interpolation._tspline import TCell2D
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry._surface_source_support import prepare_surface_source_root_atlas
from phydrax.geometry.brep._assembly import (
    BRepAffinePlacement,
    BRepAssemblyDefinition,
    BRepAssemblyGraph,
    BRepAssemblyUse,
)
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_cone,
    brep_cylinder,
    brep_sphere,
    brep_torus,
)
from phydrax.geometry.brep._model import BRepGeometry, BRepImportReport, BRepModel
from phydrax.geometry.brep._patches import (
    BSplineCurve,
    BSplineSurfacePatch,
    LineCurve,
    PlanePatch,
)
from phydrax.geometry.brep._tspline import TSplineSurfacePatch
from phydrax.units import MILLIMETER


def _square_model(
    coordinate_contract: SpatialCoordinateContract, source_digest: str
) -> BRepModel:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    curves = tuple(
        LineCurve(points[first], points[last] - points[first]) for first, last in edges
    )
    pcurves = (
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        LineCurve((1.0, 0.0), (0.0, 1.0)),
        LineCurve((1.0, 1.0), (-1.0, 0.0)),
        LineCurve((0.0, 1.0), (0.0, -1.0)),
    )
    geometry = BRepGeometry(
        vertex_points=points,
        curves=curves,
        edge_curves=(0, 1, 2, 3),
        edge_ranges=np.tile(np.asarray((0.0, 1.0)), (4, 1)),
        edge_vertices=edges,
        pcurves=pcurves,
        coedge_edges=(0, 1, 2, 3),
        coedge_senses=(1, 1, 1, 1),
        face_loops=(((0, 1, 2, 3),),),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    patch = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    report = BRepImportReport(
        source_id="origin:part",
        source_digest=source_digest,
        source_format="native",
        coordinate_contract=coordinate_contract,
        import_policy_id="f" * 64,
        num_solids=0,
        num_faces=1,
        num_edges=4,
        num_vertices=4,
    )
    return BRepModel(
        geometry=geometry,
        patches=(patch,),
        parameter_bounds=np.asarray((((0.0, 0.0), (1.0, 1.0)),)),
        orientation=np.asarray((1.0,)),
        trim_domains=None,
        physical_tags=("profile",),
        coordinate_contract=coordinate_contract,
        report=report,
    )


def _geometry_with_assembly(model: BRepModel, graph: BRepAssemblyGraph) -> BRepGeometry:
    source = model.geometry
    return BRepGeometry(
        vertex_points=source.vertex_points,
        curves=source.curves,
        edge_curves=source.edge_curves,
        edge_ranges=source.edge_ranges,
        edge_vertices=source.edge_vertices,
        pcurves=source.pcurves,
        coedge_edges=source.coedge_edges,
        coedge_senses=source.coedge_senses,
        face_loops=source.face_loops,
        shell_faces=source.shell_faces,
        shell_orientations=source.shell_orientations,
        solid_shells=source.solid_shells,
        assembly_graph=graph,
    )


def test_source_only_brep_identity_excludes_import_origin_and_tessellation() -> None:
    contract = SpatialCoordinateContract.si()
    first = _square_model(contract, "a" * 64)
    reimported = _square_model(contract, "b" * 64)
    millimeter_model = _square_model(SpatialCoordinateContract(MILLIMETER), "a" * 64)

    assert first.tessellation is None
    assert first.tessellation_id is None
    assert first.source_revision == reimported.source_revision
    assert first.report.origin_revision != reimported.report.origin_revision
    assert first.source_revision != millimeter_model.source_revision
    reversed_face = BRepModel(
        geometry=first.geometry,
        patches=first.patches,
        parameter_bounds=first.parameter_bounds,
        orientation=np.asarray((-1.0,)),
        trim_domains=first.trim_domains,
        physical_tags=first.physical_tags,
        coordinate_contract=first.coordinate_contract,
        report=first.report,
    )
    assert reversed_face.source_revision != first.source_revision
    with pytest.raises(ValueError):
        BRepModel(
            geometry=first.geometry,
            patches=first.patches,
            parameter_bounds=first.parameter_bounds,
            orientation=np.asarray((0.0,)),
            trim_domains=first.trim_domains,
            physical_tags=first.physical_tags,
            coordinate_contract=first.coordinate_contract,
            report=first.report,
        )
    evaluated = np.asarray(first.evaluate_face_carrier((0,), ((0.25, 0.75),)))
    np.testing.assert_allclose(evaluated, ((0.25, 0.75, 0.0),))
    with pytest.raises(ValueError):
        first.require_tessellation()
    with pytest.raises(ValueError):
        _ = first.mesh_faces


def test_native_analytic_primitives_publish_without_triangle_arrays() -> None:
    contract = SpatialCoordinateContract.si()
    box = brep_box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), coordinate_contract=contract)
    cylinder = brep_cylinder(0.5, 2.0, coordinate_contract=contract)
    sphere = brep_sphere(1.0, coordinate_contract=contract)
    torus = brep_torus(2.0, 0.5, coordinate_contract=contract)

    assert (box.topology.num_faces, box.topology.num_solids) == (6, 1)
    assert (cylinder.topology.num_faces, cylinder.topology.num_solids) == (3, 1)
    assert (sphere.topology.num_faces, sphere.topology.num_solids) == (1, 1)
    assert (torus.topology.num_faces, torus.topology.num_solids) == (1, 1)
    assert all(model.tessellation is None for model in (box, cylinder, sphere, torus))
    assert len(tuple(box.geometry.assembly_graph.iter_occurrences())) == 1
    cone = brep_cone(0.75, 0.0, 1.5, coordinate_contract=contract)
    assert (cone.topology.num_faces, cone.topology.num_solids) == (2, 1)
    assert cone.tessellation is None


def test_assembly_occurrences_preserve_definition_uses_and_exact_placements() -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    translated = BRepAffinePlacement.from_matrix(
        (
            (1, 0, 0, Fraction(3, 2)),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    graph = BRepAssemblyGraph(
        roots=(BRepAssemblyUse("pair", definition_id="two-copies"),),
        definitions=(
            BRepAssemblyDefinition(
                "two-copies",
                (
                    BRepAssemblyUse("left", solid_index=0),
                    BRepAssemblyUse("right", placement=translated, solid_index=0),
                ),
            ),
        ),
        solid_count=1,
    )
    geometry = _geometry_with_assembly(model, graph)
    placed = BRepModel(
        geometry=geometry,
        patches=model.patches,
        parameter_bounds=model.parameter_bounds,
        orientation=model.orientation,
        trim_domains=model.trim_domains,
        physical_tags=model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        report=model.report,
    )
    occurrences = tuple(placed.geometry.assembly_graph.iter_occurrences())

    assert tuple(item.path for item in occurrences) == (
        ("pair", "left"),
        ("pair", "right"),
    )
    assert occurrences[0].placement.apply((0, 0, 0)) == (Fraction(0),) * 3
    assert occurrences[1].placement.apply((0, 0, 0)) == (
        Fraction(3, 2),
        Fraction(0),
        Fraction(0),
    )
    assert placed.source_revision != model.source_revision


def test_occurrence_placement_copies_shared_ancestry_without_moving_aliases() -> None:
    graph = BRepAssemblyGraph(
        (
            BRepAssemblyUse("first", definition_id="component"),
            BRepAssemblyUse("second", definition_id="component"),
        ),
        (BRepAssemblyDefinition("component", (BRepAssemblyUse("leaf", solid_index=0),)),),
        1,
    )
    moved = BRepAffinePlacement.from_matrix(
        ((1, 0, 0, 2), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    )
    updated = graph.with_occurrence_placement(("first", "leaf"), moved)
    original = tuple(graph.iter_occurrences())
    occurrences = tuple(updated.iter_occurrences())
    assert tuple(item.path for item in occurrences) == (
        ("first", "leaf"),
        ("second", "leaf"),
    )
    assert occurrences[0].placement == moved
    assert occurrences[1].placement == original[1].placement
    assert original[0].placement == BRepAffinePlacement.identity()
    with pytest.raises(ValueError, match="path"):
        graph.with_occurrence_placement(("first", "missing"), moved)


def test_t_spline_patch_keeps_exact_raw_third_weights_and_shared_realization() -> None:
    mesh = TMesh2D.tensor_product((Fraction(0), Fraction(1)), (Fraction(0), Fraction(1)))
    count = mesh.coefficient_count
    exact_points = tuple((Fraction(0), Fraction(0), Fraction(0)) for _ in range(count))
    exact_thirds = tuple(Fraction(1, 3) for _ in range(count))
    raw_controls = np.zeros((count, 3), dtype=np.float64)
    raw_weights = np.full((count,), float(Fraction(1, 3)), dtype=np.float64)
    patch = TSplineSurfacePatch(
        mesh,
        raw_controls,
        raw_weights,
        exact_control_points=exact_points,
        exact_weights=exact_thirds,
    )
    dyadic_weights = tuple(
        Fraction.from_float(float(Fraction(1, 3))) for _ in range(count)
    )
    rounded_source = TSplineSurfacePatch(
        mesh,
        raw_controls,
        raw_weights,
        exact_control_points=exact_points,
        exact_weights=dyadic_weights,
    )

    assert patch.exact_source_weights[0] == Fraction(1, 3)
    assert patch.weights[0] == float(Fraction(1, 3))
    assert patch.source_definition_id != rounded_source.source_definition_id
    values = np.asarray(patch.evaluate(jnp.asarray(((0.25, 0.75),))))
    np.testing.assert_array_equal(values, np.zeros((1, 3), dtype=np.float64))
    patch.validate_exact_transfer_chain()


def test_t_spline_refinement_requires_and_preserves_exact_source_map() -> None:
    mesh = TMesh2D.tensor_product((Fraction(0), Fraction(1)), (Fraction(0), Fraction(1)))
    refinement = mesh.refine((mesh.cells[0].cell_id,))
    assert refinement.certificate.passed
    assert refinement.certificate.partition_of_unity
    assert refinement.certificate.linearly_independent
    assert refinement.certificate.trace_continuous
    assert refinement.certificate.continuity_order == 2
    assert refinement.certificate.maximum_partition_error == 0.0
    count = mesh.coefficient_count
    greville_points = tuple(
        anchor.local_knots.greville_parameter for anchor in mesh.anchors
    )
    exact_points = tuple((u, v, u * v) for u, v in greville_points)
    exact_weights = tuple(Fraction(1, 3) + Fraction(index, 97) for index in range(count))
    controls = np.asarray(
        [[float(value) for value in point] for point in exact_points], dtype=np.float64
    )
    weights = np.asarray([float(value) for value in exact_weights], dtype=np.float64)
    source_patch = TSplineSurfacePatch(
        mesh,
        controls,
        weights,
        exact_control_points=exact_points,
        exact_weights=exact_weights,
    )

    with pytest.raises(ValueError):
        TSplineSurfacePatch(refinement.refined_mesh, controls, weights, source_mesh=mesh)
    refined = TSplineSurfacePatch(
        refinement.refined_mesh,
        controls,
        weights,
        exact_control_points=exact_points,
        exact_weights=exact_weights,
        source_mesh=mesh,
        transfer_plans=(refinement.transfer_plan,),
    )
    refined.validate_exact_transfer_chain()

    expected_target = refinement.transfer_plan.apply_exact_homogeneous_geometry(
        exact_points, exact_weights
    )
    assert refined.exact_target_homogeneous == expected_target
    assert refined.mesh.mesh_id == refinement.refined_mesh.mesh_id
    query = jnp.asarray(((0.3, 0.4), (0.72, 0.15)), dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(refined.evaluate(query)),
        np.asarray(source_patch.evaluate(query)),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_t_spline_repeated_local_refinement_preserves_the_full_chain() -> None:
    mesh = TMesh2D.tensor_product((Fraction(0), Fraction(1)), (Fraction(0), Fraction(1)))
    first = mesh.refine((mesh.cells[0].cell_id,))
    second = first.refined_mesh.refine((first.refined_mesh.cells[0].cell_id,))
    exact_points = tuple(
        (u, v, u * v)
        for u, v in (anchor.local_knots.greville_parameter for anchor in mesh.anchors)
    )
    exact_weights = tuple(
        Fraction(1, 3) + Fraction(index, 97) for index in range(mesh.coefficient_count)
    )
    controls = np.asarray(
        [[float(value) for value in point] for point in exact_points], dtype=np.float64
    )
    weights = np.asarray([float(value) for value in exact_weights], dtype=np.float64)
    source_patch = TSplineSurfacePatch(
        mesh,
        controls,
        weights,
        exact_control_points=exact_points,
        exact_weights=exact_weights,
    )
    refined = TSplineSurfacePatch(
        second.refined_mesh,
        controls,
        weights,
        exact_control_points=exact_points,
        exact_weights=exact_weights,
        source_mesh=mesh,
        transfer_plans=(first.transfer_plan, second.transfer_plan),
        transfer_meshes=(mesh, first.refined_mesh, second.refined_mesh),
    )
    refined.validate_exact_transfer_chain()
    assert tuple(item.mesh_id for item in refined.transfer_meshes) == (
        mesh.mesh_id,
        first.refined_mesh.mesh_id,
        second.refined_mesh.mesh_id,
    )
    query = jnp.asarray(((0.3, 0.4), (0.72, 0.15)), dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(refined.evaluate(query)),
        np.asarray(source_patch.evaluate(query)),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_t_spline_root_atlas_separates_control_cells_and_piece_ids() -> None:
    contract = SpatialCoordinateContract.si()
    source = _square_model(contract, "a" * 64)
    mesh = TMesh2D((TCell2D(44, (Fraction(0), Fraction(1), Fraction(0), Fraction(1))),))
    exact_controls = tuple(
        (*anchor.local_knots.greville_parameter, Fraction(0)) for anchor in mesh.anchors
    )
    controls = np.asarray(exact_controls, dtype=np.float64)
    patch = TSplineSurfacePatch(
        mesh,
        controls,
        np.ones(mesh.coefficient_count, dtype=np.float64),
        exact_control_points=exact_controls,
    )
    model = BRepModel(
        geometry=source.geometry,
        patches=(patch,),
        parameter_bounds=source.parameter_bounds,
        orientation=source.orientation,
        physical_tags=source.physical_tags,
        coordinate_contract=contract,
        report=source.report,
    )
    source_atlas = prepare_surface_source_root_atlas(
        MeshingDomain.from_brep(model), (0,), contract
    )
    source_atlas.require_current()
    source_piece = source_atlas.root_source_cell_ids[0]
    assert source_atlas.root_cell_ids == (0,)
    assert source_piece == (
        0,
        mesh.cells[0].cell_id,
        patch.realization.extractions[0].piece_id,
    )
    assert source_piece[1] != source_piece[2]
    np.testing.assert_allclose(
        model.evaluate_face_carrier((0,), ((0.25, 0.75),)),
        ((0.25, 0.75, 0.0),),
    )
    stale = eqx.tree_at(
        lambda value: value.control_points,
        patch,
        patch.control_points.at[0, 2].set(0.25),
    )
    with pytest.raises(ValueError):
        BRepModel(
            geometry=source.geometry,
            patches=(stale,),
            parameter_bounds=source.parameter_bounds,
            orientation=source.orientation,
            physical_tags=source.physical_tags,
            coordinate_contract=contract,
            report=source.report,
        )


def test_bspline_root_atlas_preserves_exact_control_and_weight_banks() -> None:
    from phydrax.discretization._cell_geometry import SplineCellGeometryElement

    contract = SpatialCoordinateContract.si()
    source = _square_model(contract, "d" * 64)
    root_two = RadicalElement.sqrt(Fraction(2))
    subfloat = RadicalElement.sqrt(Fraction(2, 10**800))
    exact_controls = tuple(
        tuple(
            (
                Fraction(u, 2),
                Fraction(v, 2),
                subfloat if (u, v) == (1, 1) else Fraction(0),
            )
            for v in range(3)
        )
        for u in range(3)
    )
    exact_weights = tuple(
        tuple(
            root_two if (u, v) == (1, 1) else Fraction(1) + Fraction(u + v, 4)
            for v in range(3)
        )
        for u in range(3)
    )
    controls = np.asarray(
        [[[float(value) for value in point] for point in row] for row in exact_controls],
        dtype=np.float64,
    )
    weights = np.asarray(
        [[float(value) for value in row] for row in exact_weights],
        dtype=np.float64,
    )
    control_ids = np.asarray(
        (97, 91, 98, 92, 96, 94, 99, 93, 95), dtype=np.int64
    ).reshape((3, 3))
    knots = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(1),
        Fraction(1),
        Fraction(1),
    )
    patch = BSplineSurfacePatch(
        controls,
        weights,
        np.asarray([float(value) for value in knots]),
        np.asarray([float(value) for value in knots]),
        2,
        2,
        exact_control_points=exact_controls,
        exact_weights=exact_weights,
        exact_u_knots=knots,
        exact_v_knots=knots,
        control_point_ids=control_ids,
    )

    def boundary_curve(
        controls: tuple[tuple[Fraction | RadicalElement, ...], ...],
        weights_: tuple[Fraction | RadicalElement, ...],
        /,
    ) -> BSplineCurve:
        return BSplineCurve(
            np.asarray(
                tuple(tuple(float(value) for value in point) for point in controls),
                dtype=np.float64,
            ),
            np.asarray(tuple(float(value) for value in weights_), dtype=np.float64),
            np.asarray([float(value) for value in knots], dtype=np.float64),
            2,
            exact_control_points=controls,
            exact_weights=weights_,
            exact_knots=knots,
        )

    boundary_curves = (
        boundary_curve(
            tuple(exact_controls[u][0] for u in range(3)),
            tuple(exact_weights[u][0] for u in range(3)),
        ),
        boundary_curve(
            tuple(exact_controls[2][v] for v in range(3)),
            tuple(exact_weights[2][v] for v in range(3)),
        ),
        boundary_curve(
            tuple(exact_controls[u][2] for u in reversed(range(3))),
            tuple(exact_weights[u][2] for u in reversed(range(3))),
        ),
        boundary_curve(
            tuple(exact_controls[0][v] for v in reversed(range(3))),
            tuple(exact_weights[0][v] for v in reversed(range(3))),
        ),
    )
    source_geometry = source.geometry
    geometry = BRepGeometry(
        vertex_points=source_geometry.vertex_points,
        curves=boundary_curves,
        edge_curves=source_geometry.edge_curves,
        edge_ranges=source_geometry.edge_ranges,
        edge_vertices=source_geometry.edge_vertices,
        pcurves=source_geometry.pcurves,
        coedge_edges=source_geometry.coedge_edges,
        coedge_senses=source_geometry.coedge_senses,
        face_loops=source_geometry.face_loops,
        shell_faces=source_geometry.shell_faces,
        shell_orientations=source_geometry.shell_orientations,
        solid_shells=source_geometry.solid_shells,
        assembly_graph=source_geometry.assembly_graph,
    )
    model = BRepModel(
        geometry=geometry,
        patches=(patch,),
        parameter_bounds=source.parameter_bounds,
        orientation=source.orientation,
        physical_tags=source.physical_tags,
        coordinate_contract=contract,
        report=source.report,
    )
    atlas = prepare_surface_source_root_atlas(
        MeshingDomain.from_brep(model), (0,), contract
    )
    atlas.require_current()

    exact_control_rows = tuple(point for row in exact_controls for point in row)
    exact_weight_rows = tuple(value for row in exact_weights for value in row)
    assert float(subfloat) == 0.0
    assert atlas.geometry.source_coordinates() == exact_control_rows
    source_map = atlas.geometry.source_map
    assert source_map is not None
    assert getattr(source_map, "coefficient_keys") == tuple(
        (0, int(identifier)) for identifier in control_ids.reshape((-1,))
    )
    elements, _, _ = atlas.geometry.resolve(atlas.mesh)
    element = elements[0]
    assert isinstance(element, SplineCellGeometryElement)
    control_rows = {
        int(identifier): row for row, identifier in enumerate(control_ids.reshape((-1,)))
    }
    assert element.exact_weights == tuple(
        exact_weight_rows[control_rows[anchor]] for anchor in element.anchor_ids
    )
    assert any(isinstance(value, RadicalElement) for value in element.exact_weights)
