from __future__ import annotations

from fractions import Fraction
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._constructors import (
    assemble_brep_model,
    brep_box,
    brep_cone,
    brep_cylinder,
    brep_extrusion,
    brep_revolution,
    PlanarProfile,
    ProfileArc,
    ProfileLine,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._differentiable import _FixedTopologyBRepKernel
from phydrax.geometry.brep._edge_classification import brep_edge_classification
from phydrax.geometry.brep._fillet import brep_chamfer, brep_fillet
from phydrax.geometry.brep._model import BRepGeometry, BRepModel
from phydrax.geometry.brep._patches import (
    BSplineCurve,
    ConePatch,
    CylinderPatch,
    LineCurve,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
)


type _Family = Literal[
    "plane",
    "oblique",
    "closed-cylinder",
    "closed-cone",
    "closed-sphere",
    "parallel-cylinders",
]
type _Kind = Literal["fillet", "chamfer"]

_CONTRACT = SpatialCoordinateContract.si()


def _incident_faces(model: BRepModel, edge: int, /) -> tuple[int, ...]:
    geometry = model.geometry
    return tuple(
        face
        for face, loops in enumerate(geometry.face_loops)
        if any(
            geometry.coedge_edges[coedge] == edge
            for loop in loops
            for coedge in loop
        )
    )


def _vertical_edge(model: BRepModel, /) -> int:
    geometry = model.geometry
    points = np.asarray(geometry.vertex_points)
    for edge, (first, last) in enumerate(geometry.edge_vertices):
        direction = points[last] - points[first]
        if np.array_equal(direction[:2], np.zeros((2,), dtype=np.float64)) and abs(
            direction[2]
        ) == 4.0:
            return edge
    raise RuntimeError("The native box fixture has no vertical edge.")


def _oblique_prism() -> tuple[BRepModel, int]:
    profile = PlanarProfile(
        ProfilePlane(),
        ProfileLoop.polygon(((0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0))),
    )
    model = brep_extrusion(
        profile, (1.0, 0.0, 4.0), coordinate_contract=_CONTRACT
    )
    geometry = model.geometry
    points = np.asarray(geometry.vertex_points)
    for edge, (first, last) in enumerate(geometry.edge_vertices):
        direction = points[last] - points[first]
        if np.array_equal(direction, np.asarray((1.0, 0.0, 4.0))):
            return model, edge
        if np.array_equal(direction, np.asarray((-1.0, 0.0, -4.0))):
            return model, edge
    raise RuntimeError("The oblique-prism fixture has no longitudinal edge.")


def _sphere_with_planar_seam() -> BRepModel:
    profile = PlanarProfile(
        ProfilePlane(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        ProfileLoop(
            ((0.0, -5.0), (4.0, 3.0), (0.0, 3.0)),
            (ProfileArc((0.0, 0.0)), ProfileLine(), ProfileLine()),
        ),
    )
    return brep_revolution(
        profile,
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=_CONTRACT,
    )


def _revolution_family(model: BRepModel, face: int, /) -> str | None:
    patch = model.patches[face]
    if not isinstance(patch, RevolutionSurface):
        return None
    bounds = np.asarray(model.parameter_bounds[face])
    witness = patch.exact_quadric_witness(
        (float(bounds[0, 1]), float(bounds[1, 1]))
    )
    return None if witness is None else witness.family


def _closed_seam(model: BRepModel, /) -> int:
    geometry = model.geometry
    points = np.asarray(geometry.vertex_points)
    for edge, (first, last) in enumerate(geometry.edge_vertices):
        faces = _incident_faces(model, edge)
        kinds = tuple(model.patches[face] for face in faces)
        revolution_families = tuple(_revolution_family(model, face) for face in faces)
        if (
            first == last
            and len(faces) == 2
            and (
                (
                    any(isinstance(patch, PlanePatch) for patch in kinds)
                    and any(
                        isinstance(patch, (CylinderPatch, ConePatch, SpherePatch))
                        for patch in kinds
                    )
                )
                or (
                    "plane" in revolution_families
                    and any(
                        family in ("cylinder", "cone", "sphere")
                        for family in revolution_families
                    )
                )
            )
            and points[first, 2] == 3.0
        ):
            return edge
    raise RuntimeError("The quadric fixture has no top closed circular seam.")


def _parallel_cylinder_lens() -> tuple[BRepModel, int]:
    profile = PlanarProfile(
        ProfilePlane(),
        ProfileLoop(
            ((0.0, -4.0), (0.0, 4.0)),
            (ProfileArc((-3.0, 0.0)), ProfileArc((3.0, 0.0))),
        ),
    )
    model = brep_extrusion(
        profile, (0.0, 0.0, 2.0), coordinate_contract=_CONTRACT
    )
    geometry = model.geometry
    points = np.asarray(geometry.vertex_points)
    for edge, (first, last) in enumerate(geometry.edge_vertices):
        faces = _incident_faces(model, edge)
        if (
            first != last
            and len(faces) == 2
            and all(isinstance(model.patches[face], CylinderPatch) for face in faces)
            and points[first, 1] == 4.0
            and points[last, 1] == 4.0
        ):
            return model, edge
    raise RuntimeError("The lens fixture has no parallel-cylinder junction.")


def _fixture(family: _Family, /) -> tuple[BRepModel, int]:
    if family == "plane":
        model = brep_box(
            (0.0, 0.0, 0.0), (2.0, 3.0, 4.0), coordinate_contract=_CONTRACT
        )
        return model, _vertical_edge(model)
    if family == "oblique":
        return _oblique_prism()
    if family == "closed-cylinder":
        model = brep_cylinder(3.0, 3.0, coordinate_contract=_CONTRACT)
        return model, _closed_seam(model)
    if family == "closed-cone":
        model = brep_cone(3.0, 2.0, 3.0, coordinate_contract=_CONTRACT)
        return model, _closed_seam(model)
    if family == "closed-sphere":
        model = _sphere_with_planar_seam()
        return model, _closed_seam(model)
    if family == "parallel-cylinders":
        return _parallel_cylinder_lens()
    raise ValueError(f"Unsupported test blend family: {family!r}")


def _with_rational_spline_cap_pcurve(
    source: BRepModel, edge: int, /
) -> BRepModel:
    geometry = source.geometry
    first, last = geometry.edge_vertices[edge]
    source_curve_index = geometry.edge_curves[edge]
    source_curve = geometry.curves[source_curve_index]
    if not isinstance(source_curve, LineCurve):
        raise RuntimeError("The box cap edge is not a native line.")
    first_parameter, last_parameter = np.asarray(geometry.edge_ranges[edge])
    parameters = (
        first_parameter,
        (first_parameter + last_parameter) / 2.0,
        last_parameter,
    )
    knots = np.asarray(
        (
            first_parameter,
            first_parameter,
            first_parameter,
            last_parameter,
            last_parameter,
            last_parameter,
        ),
        dtype=np.float64,
    )
    weights = np.ones((3,), dtype=np.float64)

    def spline_line(curve: LineCurve, /) -> BSplineCurve:
        exact_parameters = tuple(
            Fraction.from_float(float(parameter)) for parameter in parameters
        )
        exact_controls = tuple(
            tuple(
                origin + parameter * direction
                for origin, direction in zip(
                    curve.origin_exact, curve.direction_exact, strict=True
                )
            )
            for parameter in exact_parameters
        )
        controls = np.asarray(
            tuple(tuple(float(value) for value in point) for point in exact_controls),
            dtype=np.float64,
        )
        return BSplineCurve(controls, weights, knots, 2)

    curves = list(geometry.curves)
    curves[source_curve_index] = spline_line(source_curve)
    pcurves = list(geometry.pcurves)
    for item, used_edge in enumerate(geometry.coedge_edges):
        if used_edge != edge:
            continue
        source_pcurve = geometry.pcurves[item]
        if not isinstance(source_pcurve, LineCurve):
            raise RuntimeError("The box edge p-curve is not a native line.")
        pcurves[item] = spline_line(source_pcurve)
    exact_points = list(geometry.vertex_exact_points)
    for vertex in geometry.edge_vertices[edge]:
        point = np.asarray(geometry.vertex_points[vertex])
        exact_points[vertex] = (
            Fraction.from_float(float(point[0])),
            Fraction.from_float(float(point[1])),
            Fraction.from_float(float(point[2])),
        )
    spline_geometry = BRepGeometry(
        vertex_points=geometry.vertex_points,
        vertex_exact_points=tuple(exact_points),
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        edge_vertex_gap_bounds=geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=geometry.edge_face_gap_bounds,
        curves=tuple(curves),
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=tuple(pcurves),
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        assembly_graph=geometry.assembly_graph,
    )
    return assemble_brep_model(
        spline_geometry,
        source.patches,
        source.parameter_bounds,
        source.orientation,
        source.physical_tags,
        coordinate_contract=source.coordinate_contract,
        source_id="rational-spline-cap-profile",
        trim_domains=source.trim_domains,
        curve_surface_tolerance=source.report.curve_surface_tolerance,
        curve_surface_scale=source.report.curve_surface_scale,
    )


@pytest.mark.parametrize(
    "family",
    (
        "plane",
        "oblique",
        "closed-cylinder",
        "closed-cone",
        "closed-sphere",
        "parallel-cylinders",
    ),
)
@pytest.mark.parametrize("kind", ("fillet", "chamfer"))
def test_supported_blend_families_preserve_exact_source_and_report_lineage(
    family: _Family, kind: _Kind
) -> None:
    source, edge = _fixture(family)
    geometry = source.geometry
    source_identity = (source.source_revision, geometry.geometry_id)
    selected = source.edge_ids[edge]
    convexity = brep_edge_classification(source, geometry)[edge]
    sign_bounds = np.asarray(convexity.sign_enclosure)
    assert convexity.status in ("convex", "concave")
    assert sign_bounds[1] < 0.0 or sign_bounds[0] > 0.0
    assert np.asarray(convexity.change_parameters).shape == (0, 2)

    if kind == "fillet":
        result = brep_fillet(source, (selected,), 0.25, include_parameter_range=True)
    else:
        result = brep_chamfer(
            source, (selected,), (0.25, 0.375), include_parameter_range=True
        )

    assert result.status == "certified", result.reason
    assert result.model is not None
    assert result.source_model_id == source.source_revision
    assert result.source_edges == (selected,)
    assert result.parameter_range is not None
    assert result.parameter_range.parameter == (
        "radius" if kind == "fillet" else "setback_scale"
    )
    assert result.parameter_range.evidence
    assert result.model.source_revision != source.source_revision
    assert result.model.report.curve_surface_tolerance == source.report.curve_surface_tolerance
    endpoint_gaps = np.asarray(result.model.geometry.edge_vertex_gap_bounds)
    assert endpoint_gaps.shape == (len(result.model.geometry.edge_curves), 2)
    assert np.all(np.isfinite(endpoint_gaps))
    assert np.all(endpoint_gaps >= 0.0)
    face_gaps = result.model.geometry.edge_face_gap_bounds
    if face_gaps is not None:
        assert np.all(np.isfinite(face_gaps))
        assert np.all(np.asarray(face_gaps) >= 0.0)
    assert (source.source_revision, geometry.geometry_id) == source_identity
    if kind == "fillet":
        bounds = np.asarray(result.g1_angle_bounds)
        assert bounds.shape == (2, 2)
        assert np.all(np.isfinite(bounds))
        assert np.all(bounds[:, 0] <= bounds[:, 1])
        assert np.all(bounds[:, 1] < 1.0e-6)
        if family == "oblique":
            assert result.sensitivity_status == "unsupported"
            assert result.source is None
        else:
            assert result.sensitivity_status == "available"
            assert result.source is not None
            compiled = result.source.compile()
            if not isinstance(compiled.kernel, _FixedTopologyBRepKernel):
                raise AssertionError("A B-Rep source must compile to its fixed-topology kernel.")
            realized = compiled.kernel.realize(compiled.state)
            assert realized.vertices is None
            assert realized.faces is None
            assert (
                realized.parameter_geometry.source_geometry.geometry_id
                == result.model.geometry.geometry_id
            )
    else:
        assert result.g1_angle_bounds.shape == (0, 2)


def test_closed_cylinder_fillet_radius_derivative_preserves_cap_height() -> None:
    source, edge = _fixture("closed-cylinder")
    result = brep_fillet(source, (source.edge_ids[edge],), 0.25)
    assert result.source is not None
    compiled = result.source.compile()
    kernel = compiled.kernel
    if not isinstance(kernel, _FixedTopologyBRepKernel):
        raise AssertionError("A fillet source must compile to the fixed-topology kernel.")
    parameter = next(
        spec.parameter_id
        for spec in compiled.state.schema.specs
        if spec.trainable and spec.parameter_id.name == "radius"
    )
    radius_index = compiled.state.schema.index(parameter)

    def cap_point(radius: Array) -> Array:
        state = compiled.state.replace_at(radius_index, radius)
        patch = kernel.realize(state).parameter_geometry.patches[-1]
        return patch.evaluate(jnp.asarray((0.0, np.pi / 2), dtype=jnp.float64))

    point, tangent = jax.jvp(
        jax.jit(cap_point),
        (jnp.asarray(0.25, dtype=jnp.float64),),
        (jnp.asarray(1.0, dtype=jnp.float64),),
    )
    np.testing.assert_allclose(point, (0.0, 2.75, 3.0), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(tangent, (0.0, -1.0, 0.0), rtol=0.0, atol=1.0e-14)


@pytest.mark.parametrize("kind", ("fillet", "chamfer"))
def test_planar_blends_preserve_rational_spline_profile_segments(kind: _Kind) -> None:
    source, edge = _fixture("plane")
    source = _with_rational_spline_cap_pcurve(source, edge)
    geometry = source.geometry
    identity = (source.source_revision, geometry.geometry_id)

    if kind == "fillet":
        result = brep_fillet(source, (source.edge_ids[edge],), 0.25)
    else:
        result = brep_chamfer(source, (source.edge_ids[edge],), (0.25, 0.375))

    assert result.status == "certified", result.reason
    assert result.model is not None
    assert any(
        isinstance(curve, BSplineCurve)
        for curve in result.model.geometry.curves
    )
    if kind == "fillet":
        assert result.sensitivity_status == "unsupported"
        assert result.source is None
    assert (source.source_revision, geometry.geometry_id) == identity


def test_corner_network_refusal_rolls_back_without_mutating_source() -> None:
    source = brep_box(
        (0.0, 0.0, 0.0), (2.0, 3.0, 4.0), coordinate_contract=_CONTRACT
    )
    geometry = source.geometry
    edge = _vertical_edge(source)
    endpoints = set(geometry.edge_vertices[edge])
    neighbor = next(
        candidate
        for candidate, vertices in enumerate(geometry.edge_vertices)
        if candidate != edge and endpoints.intersection(vertices)
    )
    snapshot = (source.source_revision, geometry.geometry_id, geometry.edge_vertices)

    result = brep_fillet(
        source, (source.edge_ids[edge], source.edge_ids[neighbor]), 0.25
    )

    assert result.status == "unsupported"
    assert result.model is None
    assert result.source_edges == (source.edge_ids[edge], source.edge_ids[neighbor])
    assert (source.source_revision, geometry.geometry_id, geometry.edge_vertices) == snapshot


def test_fillet_that_exhausts_physical_face_reach_refuses_without_mutation() -> None:
    source = brep_box(
        (0.0, 0.0, 0.0), (2.0, 3.0, 4.0), coordinate_contract=_CONTRACT
    )
    geometry = source.geometry
    identity = (source.source_revision, geometry.geometry_id)

    result = brep_fillet(source, (source.edge_ids[_vertical_edge(source)],), 2.0)

    assert result.status == "unsupported"
    assert result.model is None
    assert result.parameter_range is not None
    assert (source.source_revision, geometry.geometry_id) == identity
