from __future__ import annotations

from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._intersection import TrimIntersectionRoot, TrimRootEndpoint
from phydrax.geometry.brep._model import (
    BRepGeometry,
    BRepImportReport,
    BRepModel,
    BRepTopology,
)
from phydrax.geometry.brep._patches import (
    AbstractSurfacePatch,
    BRepCurve,
    CylinderPatch,
    ExactLineCurve,
    LineCurve,
    PlanePatch,
    RuledSurface,
    SurfaceMappedCurve,
)
from phydrax.geometry.brep._root_bindings import (
    BRepPlacedVertex,
    BRepRootSupport,
    BRepVertexRoot,
)


def _square_geometry(*, exact_lines: bool = False) -> BRepGeometry:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    exact_points = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
    )
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    if exact_lines:
        curves = tuple(
            ExactLineCurve(exact_points[first], exact_points[last])
            for first, last in edges
        )
    else:
        curves = tuple(
            LineCurve(points[first], points[last] - points[first])
            for first, last in edges
        )
    pcurves = (
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        LineCurve((1.0, 0.0), (0.0, 1.0)),
        LineCurve((1.0, 1.0), (-1.0, 0.0)),
        LineCurve((0.0, 1.0), (0.0, -1.0)),
    )
    return BRepGeometry(
        vertex_points=points,
        vertex_exact_points=exact_points,
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


def _square_model(contract: SpatialCoordinateContract, /) -> BRepModel:
    geometry = _square_geometry()
    report = BRepImportReport(
        source_id="authority-square",
        source_digest="a" * 64,
        source_format="native",
        coordinate_contract=contract,
        import_policy_id="b" * 64,
        num_solids=0,
        num_faces=1,
        num_edges=4,
        num_vertices=4,
    )
    return BRepModel(
        geometry=geometry,
        patches=(PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),),
        parameter_bounds=np.asarray((((0.0, 0.0), (1.0, 1.0)),)),
        orientation=np.asarray((1.0,)),
        physical_tags=("profile",),
        coordinate_contract=contract,
        report=report,
    )


def _cylinder_model(contract: SpatialCoordinateContract, /) -> BRepModel:
    patch = CylinderPatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        0.5,
    )
    pcurves = (
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        LineCurve((1.0, 0.0), (0.0, 1.0)),
        LineCurve((0.0, 1.0), (1.0, 0.0)),
        LineCurve((0.0, 0.0), (0.0, 1.0)),
    )
    curves = tuple(SurfaceMappedCurve(patch, pcurve) for pcurve in pcurves)
    corners = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    points = np.asarray(patch.evaluate(corners), dtype=np.float64)
    roots = tuple(
        BRepVertexRoot(
            BRepPlacedVertex(
                point,
                np.eye(3, dtype=np.float64),
                np.zeros((3,), dtype=np.float64),
                f"cylinder-vertex-{index}",
            )
        )
        for index, point in enumerate(points)
    )
    geometry = BRepGeometry(
        vertex_points=points,
        vertex_roots=roots,
        curves=curves,
        edge_curves=(0, 1, 2, 3),
        edge_ranges=np.tile(np.asarray((0.0, 1.0)), (4, 1)),
        edge_vertices=((0, 1), (1, 2), (3, 2), (0, 3)),
        pcurves=pcurves,
        coedge_edges=(0, 1, 2, 3),
        coedge_senses=(1, 1, -1, -1),
        face_loops=(((0, 1, 2, 3),),),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    report = BRepImportReport(
        source_id="authority-cylinder",
        source_digest="c" * 64,
        source_format="native",
        coordinate_contract=contract,
        import_policy_id="d" * 64,
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
        physical_tags=("cylinder-side",),
        coordinate_contract=contract,
        report=report,
    )


def _with_patches(
    model: BRepModel, patches: tuple[AbstractSurfacePatch, ...], /
) -> BRepModel:
    return BRepModel(
        geometry=model.geometry,
        patches=patches,
        parameter_bounds=np.asarray(model.parameter_bounds),
        orientation=np.asarray(model.orientation),
        physical_tags=model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        report=model.report,
        trim_domains=model.trim_domains,
    )


def _with_geometry(model: BRepModel, geometry: BRepGeometry, /) -> BRepModel:
    return eqx.tree_at(lambda value: value.geometry, model, geometry)


def _copy_geometry(
    geometry: BRepGeometry,
    *,
    curves: tuple[BRepCurve, ...],
    vertex_roots: tuple[BRepVertexRoot | None, ...] | None = None,
    edge_endpoint_roots: (
        tuple[tuple[TrimRootEndpoint | None, TrimRootEndpoint | None], ...] | None
    ) = None,
    coedge_endpoint_roots: (
        tuple[tuple[TrimRootEndpoint | None, TrimRootEndpoint | None], ...] | None
    ) = None,
) -> BRepGeometry:
    return BRepGeometry(
        vertex_points=geometry.vertex_points,
        vertex_exact_points=geometry.vertex_exact_points,
        vertex_roots=(
            geometry.vertex_roots if vertex_roots is None else vertex_roots
        ),
        edge_endpoint_roots=(
            geometry.edge_endpoint_roots
            if edge_endpoint_roots is None
            else edge_endpoint_roots
        ),
        coedge_endpoint_roots=(
            geometry.coedge_endpoint_roots
            if coedge_endpoint_roots is None
            else coedge_endpoint_roots
        ),
        edge_vertex_gap_bounds=geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=geometry.edge_face_gap_bounds,
        curves=curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        assembly_graph=geometry.assembly_graph,
    )


def _rooted_square_geometry() -> BRepGeometry:
    geometry = _square_geometry()
    trim_root = TrimIntersectionRoot(
        ExactLineCurve((Fraction(-1), Fraction(-1)), (Fraction(1), Fraction(1))),
        ExactLineCurve((Fraction(-1), Fraction(1)), (Fraction(1), Fraction(-1))),
        parameter_lower=np.asarray((0.4, 0.4)),
        parameter_upper=np.asarray((0.6, 0.6)),
    )
    start_root = TrimRootEndpoint(
        trim_root, "first", affine_parameter_offset=-0.5
    )
    end_root = TrimRootEndpoint(
        trim_root, "first", affine_parameter_offset=0.5
    )
    endpoint_roots = (
        (start_root, None),
        (None, None),
        (None, None),
        (None, end_root),
    )
    root = BRepVertexRoot(
        BRepRootSupport(
            PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            trim_root,
        )
    )
    return _copy_geometry(
        geometry,
        curves=geometry.curves,
        vertex_roots=(root, None, None, None),
        edge_endpoint_roots=endpoint_roots,
        coedge_endpoint_roots=endpoint_roots,
    )


def test_valid_primitive_and_composed_models_remain_current() -> None:
    contract = SpatialCoordinateContract.si()
    primitive = _cylinder_model(contract)
    primitive.require_current()

    planar = _square_model(contract)
    composed_patch = RuledSurface(
        LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        LineCurve((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    )
    composed = _with_patches(planar, (composed_patch,))
    composed.require_current()


def test_cylinder_execution_shadows_refuse_fresh_and_stale_models() -> None:
    source = _cylinder_model(SpatialCoordinateContract.si())
    patch = source.patches[0]
    assert isinstance(patch, CylinderPatch)
    replacements = (
        (lambda value: value.radius, patch.radius * 1.25),
        (lambda value: value.origin, patch.origin.at[0].add(0.125)),
    )
    for select, replacement in replacements:
        changed_patch = eqx.tree_at(select, patch, replacement)
        changed_table = (changed_patch,)
        with pytest.raises(ValueError):
            _with_patches(source, changed_table)

        stale = eqx.tree_at(lambda value: value.patches, source, changed_table)
        with pytest.raises(ValueError):
            stale.require_current()


def test_exact_line_execution_shadow_refuses_fresh_geometry() -> None:
    geometry = _square_geometry(exact_lines=True)
    curve = geometry.curves[0]
    assert isinstance(curve, ExactLineCurve)
    changed = eqx.tree_at(
        lambda value: value.first,
        curve,
        curve.first.at[0].add(0.125),
    )
    curves = (changed, *geometry.curves[1:])
    with pytest.raises(ValueError):
        _copy_geometry(geometry, curves=curves)


def test_nested_root_branch_arrays_refuse_fresh_and_stale_models() -> None:
    contract = SpatialCoordinateContract.si()
    source = _square_model(contract)
    geometry = _rooted_square_geometry()
    rooted_model = BRepModel(
        geometry=geometry,
        patches=source.patches,
        parameter_bounds=np.asarray(source.parameter_bounds),
        orientation=np.asarray(source.orientation),
        physical_tags=source.physical_tags,
        coordinate_contract=source.coordinate_contract,
        report=source.report,
        trim_domains=source.trim_domains,
    )
    rooted_model.require_current()

    vertex = geometry.vertex_roots[0]
    assert vertex is not None
    assert isinstance(vertex.primary, BRepRootSupport)
    root = vertex.primary.root
    assert isinstance(root, TrimIntersectionRoot)
    changed_parameters = root.parameters.at[0].add(0.125)
    changed_root = eqx.tree_at(
        lambda value: value.parameters, root, changed_parameters
    )
    changed_primary = eqx.tree_at(
        lambda value: value.root, vertex.primary, changed_root
    )
    changed_vertex = eqx.tree_at(
        lambda value: value.primary, vertex, changed_primary
    )
    changed_roots = (changed_vertex, None, None, None)
    with pytest.raises(ValueError):
        _copy_geometry(
            geometry, curves=geometry.curves, vertex_roots=changed_roots
        )

    stale_geometry = eqx.tree_at(
        lambda value: value.vertex_roots[0].primary.root.parameters,
        geometry,
        changed_parameters,
    )
    with pytest.raises(ValueError):
        _with_geometry(rooted_model, stale_geometry).require_current()


def test_geometry_currentness_recomputes_vertex_bounds_and_topology() -> None:
    model = _square_model(SpatialCoordinateContract.si())
    geometry = model.geometry
    changed_bounds = geometry.vertex_evaluation_bounds.at[0].set(0.25)
    stale_bounds = eqx.tree_at(
        lambda value: value.vertex_evaluation_bounds, geometry, changed_bounds
    )
    with pytest.raises(ValueError, match="content has changed"):
        _with_geometry(model, stale_bounds).require_current()

    topology = geometry.topology
    changed_topology = BRepTopology(
        face_edges=((0, 1, 2),),
        edge_faces=topology.edge_faces,
        face_wires=topology.face_wires,
        solid_faces=topology.solid_faces,
        solid_face_orientations=topology.solid_face_orientations,
        num_vertices=topology.num_vertices,
    )
    stale_topology = eqx.tree_at(
        lambda value: value.vertex_points, geometry, geometry.vertex_points
    )
    object.__setattr__(stale_topology, "topology", changed_topology)
    with pytest.raises(ValueError, match="content has changed"):
        _with_geometry(model, stale_topology).require_current()
