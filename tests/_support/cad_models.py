"""Native B-Rep fixtures shared by the CAD interchange contract tests."""

import hashlib
from dataclasses import replace
from fractions import Fraction
from functools import cache

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep import (
    assemble_brep_model,
    brep_box,
    brep_cone,
    brep_cylinder,
    brep_extrusion,
    brep_revolution,
    brep_sphere,
    brep_torus,
    BRepGeometry,
    BRepModel,
    BRepOccurrence,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._intersection import (
    NativePeriodEndpoint,
    TrimIntersectionRoot,
    TrimRootEndpoint,
)
from phydrax.geometry.brep._intersection_curve import CurveTrimSegment
from phydrax.geometry.brep._model import BRepAssemblyContainer
from phydrax.geometry.brep._patches import (
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    LineCurve,
    OffsetSurface,
    PlanePatch,
)
from phydrax.geometry.brep._root_bindings import BRepRootSupport, BRepVertexRoot
from phydrax.interchange._cad import CadImportPolicy, CadStage, StagedCoedge


CONTRACT = SpatialCoordinateContract.si()
SOLIDS = ("box", "cylinder", "sphere", "torus", "cone", "plate", "ring")
EXACT_VOLUMES = {
    "box": 6.0,
    "cylinder": np.pi * 2.0,
    "sphere": 4.0 / 3.0 * np.pi * 1.5**3,
    "torus": 2.0 * np.pi**2 * 3.0 * 1.0**2,
    "cone": np.pi * 2.0 / 3.0,
    "plate": (12.0 - np.pi * 0.25) * 0.5,
    "ring": np.pi * (3.0**2 - 2.0**2) * 1.0,
}


@cache
def native_solid(name: str) -> BRepModel:
    match name:
        case "box":
            return brep_box(
                (0.0, 0.0, 0.0), (1.0, 2.0, 3.0), coordinate_contract=CONTRACT
            )
        case "cylinder":
            return brep_cylinder(1.0, 2.0, coordinate_contract=CONTRACT)
        case "sphere":
            return brep_sphere(1.5, coordinate_contract=CONTRACT)
        case "torus":
            return brep_torus(3.0, 1.0, coordinate_contract=CONTRACT)
        case "cone":
            return brep_cone(1.0, 0.0, 2.0, coordinate_contract=CONTRACT)
        case "plate":
            profile = PlanarProfile(
                ProfilePlane((0.0, 0.0, 0.0)),
                ProfileLoop.polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))),
                (ProfileLoop.circle((2.0, 1.5), 0.5),),
            )
            return brep_extrusion(profile, (0.0, 0.0, 0.5), coordinate_contract=CONTRACT)
        case "ring":
            profile = PlanarProfile(
                ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                ProfileLoop.polygon(((2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0))),
            )
            return brep_revolution(
                profile, (0.0, 0.0), (0.0, 1.0), coordinate_contract=CONTRACT
            )
        case _:
            raise ValueError(name)


QUARTER_TURN = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


@cache
def native_assembly() -> BRepModel:
    """A box and a quarter-turned, translated cylinder as two placed solids."""
    parts = (native_solid("box"), native_solid("cylinder"))
    geometries: list[BRepGeometry] = []
    for part in parts:
        if part.geometry is None:
            raise ValueError("Native constructions require exact geometry.")
        geometries.append(part.geometry)
    counts = np.zeros(5, dtype=np.int64)
    fields: dict[str, list] = {
        key: []
        for key in (
            "vertex_points",
            "curves",
            "edge_curves",
            "edge_ranges",
            "edge_vertices",
            "vertex_roots",
            "pcurves",
            "edge_endpoint_roots",
            "coedge_edges",
            "coedge_senses",
            "coedge_endpoint_roots",
            "face_loops",
            "shell_faces",
            "shell_orientations",
            "solid_shells",
        )
    }
    for geometry in geometries:
        vertices, curves, edges, coedges, faces = (int(value) for value in counts)
        shells = len(fields["shell_faces"])
        fields["vertex_points"].append(np.asarray(geometry.vertex_points))
        fields["curves"].extend(geometry.curves)
        fields["edge_curves"].extend(
            -1 if c == -1 else c + curves for c in geometry.edge_curves
        )
        fields["edge_ranges"].append(np.asarray(geometry.edge_ranges))
        fields["edge_vertices"].extend(
            (a + vertices, b + vertices) for a, b in geometry.edge_vertices
        )
        fields["vertex_roots"].extend(geometry.vertex_roots)
        fields["pcurves"].extend(geometry.pcurves)
        fields["edge_endpoint_roots"].extend(geometry.edge_endpoint_roots)
        fields["coedge_edges"].extend(edge + edges for edge in geometry.coedge_edges)
        fields["coedge_senses"].extend(geometry.coedge_senses)
        fields["coedge_endpoint_roots"].extend(geometry.coedge_endpoint_roots)
        fields["face_loops"].extend(
            tuple(tuple(c + coedges for c in loop) for loop in loops)
            for loops in geometry.face_loops
        )
        fields["shell_faces"].extend(
            tuple(f + faces for f in shell) for shell in geometry.shell_faces
        )
        fields["shell_orientations"].extend(geometry.shell_orientations)
        fields["solid_shells"].extend(
            tuple(s + shells for s in solid) for solid in geometry.solid_shells
        )
        counts += (
            geometry.vertex_points.shape[0],
            len(geometry.curves),
            len(geometry.edge_curves),
            len(geometry.coedge_edges),
            len(geometry.face_loops),
        )
    geometry = BRepGeometry(
        vertex_points=np.concatenate(fields["vertex_points"]),
        curves=tuple(fields["curves"]),
        edge_curves=tuple(fields["edge_curves"]),
        edge_ranges=np.concatenate(fields["edge_ranges"]),
        edge_vertices=tuple(fields["edge_vertices"]),
        pcurves=tuple(fields["pcurves"]),
        coedge_edges=tuple(fields["coedge_edges"]),
        coedge_senses=tuple(fields["coedge_senses"]),
        face_loops=tuple(fields["face_loops"]),
        shell_faces=tuple(fields["shell_faces"]),
        shell_orientations=tuple(fields["shell_orientations"]),
        solid_shells=tuple(fields["solid_shells"]),
        vertex_roots=tuple(fields["vertex_roots"]),
        edge_endpoint_roots=tuple(fields["edge_endpoint_roots"]),
        coedge_endpoint_roots=tuple(fields["coedge_endpoint_roots"]),
        occurrences=(
            BRepOccurrence(("base",), 0),
            BRepOccurrence(
                ("post",), 1, np.asarray(QUARTER_TURN), np.asarray((3.0, 0.0, 0.0))
            ),
        ),
        assembly_containers=(BRepAssemblyContainer(("model",), (("base",), ("post",))),),
    )
    return assemble_brep_model(
        geometry,
        tuple(patch for part in parts for patch in part.patches),
        np.concatenate([np.asarray(part.parameter_bounds) for part in parts]),
        np.concatenate([np.asarray(part.orientation) for part in parts]),
        tuple(tag for part in parts for tag in part.physical_tags),
        coordinate_contract=CONTRACT,
        source_id="native-assembly",
        source_format="native",
        source_digest=hashlib.sha256(b"native-assembly").hexdigest(),
        import_policy_id=hashlib.sha256(b"native-assembly-policy").hexdigest(),
    )


@cache
def native_repeated_assembly(*, declared: bool = True) -> BRepModel:
    """One exact box definition used twice under a nested assembly node."""
    source = native_solid("box")
    original = source.geometry
    if original is None:
        raise ValueError("Repeated assembly construction requires exact geometry.")
    geometry = BRepGeometry(
        vertex_points=original.vertex_points,
        curves=original.curves,
        edge_curves=original.edge_curves,
        edge_ranges=original.edge_ranges,
        edge_vertices=original.edge_vertices,
        pcurves=original.pcurves,
        coedge_edges=original.coedge_edges,
        coedge_senses=original.coedge_senses,
        face_loops=original.face_loops,
        shell_faces=original.shell_faces,
        shell_orientations=original.shell_orientations,
        solid_shells=original.solid_shells,
        occurrences=(
            BRepOccurrence(("group", "base"), 0),
            BRepOccurrence(
                ("group", "moved"),
                0,
                np.asarray(QUARTER_TURN),
                np.asarray((4.0, 2.0, 1.0)),
            ),
        ),
        assembly_containers=(
            (
                BRepAssemblyContainer(("model",), (), (("group",),)),
                BRepAssemblyContainer(
                    ("group",), (("group", "base"), ("group", "moved"))
                ),
            )
            if declared
            else ()
        ),
    )
    return assemble_brep_model(
        geometry,
        source.patches,
        source.parameter_bounds,
        source.orientation,
        source.physical_tags,
        coordinate_contract=CONTRACT,
        source_id="native-repeated-assembly",
        source_format="native",
        source_digest=hashlib.sha256(b"native-repeated-assembly").hexdigest(),
        import_policy_id=hashlib.sha256(b"native-repeated-assembly-policy").hexdigest(),
    )


@cache
def native_rational_spline_face() -> BRepModel:
    """Exact quarter-cylinder strip, including rational boundary carriers."""
    poles = np.asarray(
        [[[x, y, z] for z in (0.0, 2.0)] for x, y in ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0))]
    )
    weights = np.asarray([[1.0, 1.0], [np.sqrt(0.5), np.sqrt(0.5)], [1.0, 1.0]])
    u_knots = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    v_knots = np.asarray([0.0, 0.0, 1.0, 1.0])
    patch = BSplineSurfacePatch(poles, weights, u_knots, v_knots, 2, 1)
    geometry = BRepGeometry(
        vertex_points=poles[[0, 2, 2, 0], [0, 0, 1, 1]],
        curves=(
            BSplineCurve(poles[:, 0], weights[:, 0], u_knots, 2),
            LineCurve(poles[-1, 0], poles[-1, 1] - poles[-1, 0]),
            BSplineCurve(poles[:, 1], weights[:, 1], u_knots, 2),
            LineCurve(poles[0, 0], poles[0, 1] - poles[0, 0]),
        ),
        edge_curves=(0, 1, 2, 3),
        edge_ranges=np.asarray([[0.0, 1.0]] * 4),
        edge_vertices=((0, 1), (1, 2), (3, 2), (0, 3)),
        pcurves=(
            LineCurve([0, 0], [1, 0]),
            LineCurve([1, 0], [0, 1]),
            LineCurve([0, 1], [1, 0]),
            LineCurve([0, 0], [0, 1]),
        ),
        coedge_edges=(0, 1, 2, 3),
        coedge_senses=(1, 1, -1, -1),
        face_loops=(((0, 1, 2, 3),),),
        shell_faces=((0,),),
        shell_orientations=((1,),),
        solid_shells=(),
        occurrences=(),
    )
    return assemble_brep_model(
        geometry,
        (patch,),
        np.asarray([[[0.0, 0.0], [1.0, 1.0]]]),
        np.asarray([1.0]),
        ("rational-spline",),
        coordinate_contract=CONTRACT,
        source_id="native-rational-spline",
        source_format="native",
        source_digest=hashlib.sha256(b"native-rational-spline").hexdigest(),
        import_policy_id=hashlib.sha256(b"native-rational-spline-policy").hexdigest(),
    )


@cache
def native_sector_face() -> BRepModel:
    """A planar 270-degree sector with an explicitly trimmed circular edge."""
    curve = CircleCurve([0, 0, 0], [1, 0, 0], [0, 1, 0], 1.0)
    pcurve = CircleCurve([0, 0], [1, 0], [0, 1], 1.0)
    edge_period = (
        NativePeriodEndpoint(curve, turns=0),
        NativePeriodEndpoint(curve, turns=Fraction(3, 4)),
    )
    coedge_period = (
        NativePeriodEndpoint(pcurve, turns=0),
        NativePeriodEndpoint(pcurve, turns=Fraction(3, 4)),
    )
    geometry = BRepGeometry(
        vertex_points=np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]),
        curves=(
            curve,
            LineCurve([0, -1, 0], [0, 1, 0]),
            LineCurve([0, 0, 0], [1, 0, 0]),
        ),
        edge_curves=(0, 1, 2),
        edge_ranges=np.asarray([[0.0, 1.5 * np.pi], [0.0, 1.0], [0.0, 1.0]]),
        edge_vertices=((0, 1), (1, 2), (2, 0)),
        pcurves=(
            pcurve,
            LineCurve([0, -1], [0, 1]),
            LineCurve([0, 0], [1, 0]),
        ),
        coedge_edges=(0, 1, 2),
        coedge_senses=(1, 1, 1),
        face_loops=(((0, 1, 2),),),
        shell_faces=((0,),),
        shell_orientations=((1,),),
        solid_shells=(),
        edge_endpoint_roots=(edge_period, (None, None), (None, None)),
        coedge_endpoint_roots=(coedge_period, (None, None), (None, None)),
        occurrences=(),
    )
    return assemble_brep_model(
        geometry,
        (PlanePatch([0, 0, 0], [1, 0, 0], [0, 1, 0]),),
        np.asarray([[[-1.0, -1.0], [1.0, 1.0]]]),
        np.asarray([1.0]),
        ("plane",),
        coordinate_contract=CONTRACT,
        source_id="native-sector",
        source_format="native",
        source_digest=hashlib.sha256(b"native-sector").hexdigest(),
        import_policy_id=hashlib.sha256(b"native-sector-policy").hexdigest(),
    )


@cache
def native_rooted_sector_face() -> BRepModel:
    """The sector's first junction retained as a unique source UV root."""
    source = native_sector_face()
    original = source.geometry
    if original is None:
        raise ValueError("Rooted sector construction requires exact geometry.")
    root = TrimIntersectionRoot(
        CurveTrimSegment(original.pcurves[2], 0.0, 1.0),
        CurveTrimSegment(original.pcurves[0], 0.0, 1.5 * np.pi),
        parameter_lower=np.asarray([1.0 - 1e-8, -1e-8]),
        parameter_upper=np.asarray([1.0 + 1e-8, 1e-8]),
    )
    vertex = BRepVertexRoot(BRepRootSupport(source.patches[0], root))
    first = TrimRootEndpoint(root, "first")
    second = TrimRootEndpoint(root, "second")
    endpoints = ((second, None), (None, None), (None, first))
    geometry = BRepGeometry(
        vertex_points=original.vertex_points,
        curves=original.curves,
        edge_curves=original.edge_curves,
        edge_ranges=original.edge_ranges,
        edge_vertices=original.edge_vertices,
        pcurves=original.pcurves,
        coedge_edges=original.coedge_edges,
        coedge_senses=original.coedge_senses,
        face_loops=original.face_loops,
        shell_faces=original.shell_faces,
        shell_orientations=original.shell_orientations,
        solid_shells=original.solid_shells,
        occurrences=original.occurrences,
        assembly_containers=original.assembly_containers,
        vertex_roots=(vertex, None, None),
        edge_endpoint_roots=endpoints,
        coedge_endpoint_roots=endpoints,
    )
    report = replace(
        source.report,
        source_id="native-rooted-sector",
        source_digest=hashlib.sha256(root.root_id.encode()).hexdigest(),
        import_policy_id=hashlib.sha256(b"native-rooted-sector-policy").hexdigest(),
    )
    return BRepModel(
        patches=source.patches,
        parameter_bounds=source.parameter_bounds,
        orientation=source.orientation,
        trim_domains=source.trim_domains,
        topology=geometry.topology(),
        coordinate_contract=CONTRACT,
        mesh_vertices=source.mesh_vertices,
        mesh_faces=source.mesh_faces,
        triangle_face_ids=source.triangle_face_ids,
        triangle_parameters=source.triangle_parameters,
        physical_tags=source.physical_tags,
        report=report,
        geometry=geometry,
        tessellation_deviation_bounds=source.tessellation_deviation_bounds,
        tessellation_normal_bounds=source.tessellation_normal_bounds,
        mesh_vertex_source_dimensions=source.mesh_vertex_source_dimensions,
        mesh_vertex_source_indices=source.mesh_vertex_source_indices,
        mesh_vertex_parameters=source.mesh_vertex_parameters,
        mesh_chart_restriction_vertices=source.mesh_chart_restriction_vertices,
        mesh_chart_restriction_edges=source.mesh_chart_restriction_edges,
        mesh_chart_restriction_endpoint_parameters=(
            source.mesh_chart_restriction_endpoint_parameters
        ),
        mesh_chart_restriction_parameters=source.mesh_chart_restriction_parameters,
        triangle_occurrence_ids=source.triangle_occurrence_ids,
        vertex_occurrence_ids=source.vertex_occurrence_ids,
    )


@cache
def native_empty_model() -> BRepModel:
    """A real zero-entity exact model, not a placeholder solid or occurrence."""
    geometry = BRepGeometry(
        vertex_points=np.empty((0, 3)),
        curves=(),
        edge_curves=(),
        edge_ranges=np.empty((0, 2)),
        edge_vertices=(),
        pcurves=(),
        coedge_edges=(),
        coedge_senses=(),
        face_loops=(),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
        occurrences=(),
    )
    return assemble_brep_model(
        geometry,
        (),
        np.empty((0, 2, 2)),
        np.empty((0,)),
        (),
        coordinate_contract=CONTRACT,
        source_id="native-empty",
        source_format="native",
        source_digest=hashlib.sha256(b"native-empty").hexdigest(),
        import_policy_id=hashlib.sha256(b"native-empty-policy").hexdigest(),
    )


@cache
def native_intersection_cap(*, realize: bool = True) -> BRepModel:
    """A real plane cap whose authoritative closed edge is a native branch."""
    brep = phx.geometry.brep
    sphere = brep.SurfaceRegion(
        brep.SpherePatch([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 1.0),
        np.asarray([[0.0, -np.pi / 2], [2 * np.pi, np.pi / 2]]),
    )
    plane = brep.SurfaceRegion(
        PlanePatch([-2, -2, 0.3], [4, 0, 0], [0, 4, 0]),
        np.asarray([[0.0, 0.0], [1.0, 1.0]]),
    )
    result = brep.intersect_surface_regions(sphere, plane)
    assert result.complete and len(result.curves) == 1
    branch = result.curves[0]
    first, last = branch.parameter_interval
    pcurve = branch.p_curve("second")
    parameters = jnp.linspace(first, last, 257)
    uv = np.asarray(pcurve.evaluate(parameters))
    area = np.sum(uv[:-1, 0] * uv[1:, 1] - uv[1:, 0] * uv[:-1, 1])
    sense = 1 if area > 0.0 else -1
    vertex = np.asarray(branch.evaluate(jnp.asarray([first])).point)
    geometry = BRepGeometry(
        vertex_points=vertex,
        curves=(branch,),
        edge_curves=(0,),
        edge_ranges=np.asarray([[first, last]]),
        edge_vertices=((0, 0),),
        pcurves=(pcurve,),
        coedge_edges=(0,),
        coedge_senses=(sense,),
        face_loops=(((0,),),),
        shell_faces=((0,),),
        shell_orientations=((1,),),
        solid_shells=(),
        occurrences=(),
    )
    return assemble_brep_model(
        geometry,
        (plane.patch,),
        np.asarray([plane.parameter_box]),
        np.asarray([1.0]),
        ("plane",),
        coordinate_contract=CONTRACT,
        source_id="native-intersection-cap",
        source_format="native",
        source_digest=hashlib.sha256(branch.branch_id.encode()).hexdigest(),
        import_policy_id=hashlib.sha256(b"native-intersection-cap-policy").hexdigest(),
        tessellation=brep.BRepTessellationPolicy(realize=realize),
    )


def native_offset_plane(policy: CadImportPolicy) -> BRepModel:
    """An authored plane offset with exact boundary carriers and source digest."""
    base = PlanePatch((0, 0, 0.5), (1, 0, 0), (0, 1, 0))
    patch = OffsetSurface(base, 0.25)
    uv = np.asarray(((0, 0), (2, 0), (2, 1), (0, 1)))
    points = np.asarray(patch.evaluate(jnp.asarray(uv)))
    stage = CadStage(2.0, None)
    vertices = [
        stage.vertex(point, f"corner:{index}") for index, point in enumerate(points)
    ]
    loop = []
    for index in range(4):
        following = (index + 1) % 4
        edge = stage.edge(
            LineCurve(points[index], points[following] - points[index]),
            vertices[index],
            vertices[following],
            f"edge:{index}",
            parameter_range=(0, 1),
        )
        loop.append(
            StagedCoedge(
                edge,
                1,
                LineCurve(uv[index], uv[following] - uv[index]),
                f"edge:{index}",
            )
        )
    stage.face(patch, [loop], 1, "offset", "offset-plane", outer_known=True)
    digest = canonical_fingerprint(
        {
            "base_origin": np.asarray(base.origin),
            "base_first_axis": np.asarray(base.first_axis),
            "base_second_axis": np.asarray(base.second_axis),
            "distance": np.asarray(patch.distance),
            "corners": uv,
            "coordinate_contract": policy.coordinate_contract.spatial_id,
        }
    )
    return stage.publish(
        coordinate_contract=policy.coordinate_contract,
        source_id="offset-plane",
        source_format="native",
        source_digest=digest,
        import_policy_id=policy.policy_id,
        tessellation=None,
        default_occurrences=True,
    )


def solid_volumes(model: BRepModel) -> np.ndarray:
    return np.asarray(phx.geometry.brep.prepare_brep_query(model).measures.solid_volumes)
