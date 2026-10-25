"""White-box contracts for bounded stationary strata and root realization."""

import pickle

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._interval_enclosure import prepare_interval_function
from phydrax.geometry.brep import (
    brep_planar_face,
    BRepModel,
    BRepTessellationPolicy,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._constructors import brep_trim_domain
from phydrax.geometry.brep._patches import PlanePatch
from phydrax.geometry.brep._query_types import FaceCertificate
from phydrax.geometry.brep._stationary_isolation import (
    ClosestStationaryIsolation,
    isolate_closest_stationary,
    source_distance_bounds,
    stationary_root_representative,
    StationaryRoot,
)
from tests._support.cad_models import native_solid


# The box has maximum coordinate magnitude three and linear carriers: 64
# binary64 ulps per unit cover evaluation, interval endpoints, and the
# final Euclidean error reduction.
# All trim arcs are affine, so this bound also covers their endpoint-rounding
# contribution during trim preparation without a geometric approximation error.
_ROUNDING_BOUND = float(64 * 3 * np.finfo(np.float64).eps)


@pytest.fixture(scope="module")
def box_model() -> BRepModel:
    return native_solid(
        "box",
        catalog="geometry",
        # Linear carriers make tessellation accuracy immaterial to these tests.
        tessellation=BRepTessellationPolicy(
            linear_deflection=0.25, angular_deflection=0.5
        ),
    )


def _plane_certificate(patch: PlanePatch, bounds: np.ndarray) -> FaceCertificate:
    """Prepare independent closed-form squared-distance jets for a plane."""
    basis = jnp.stack((patch.first_axis, patch.second_axis), axis=1)
    gram = basis.T @ basis

    def value(source: Array) -> Array:
        difference = patch.evaluate(source[:2]) - source[2:]
        return jnp.sum(difference * difference)

    def gradient(source: Array) -> Array:
        return 2.0 * (basis.T @ (patch.evaluate(source[:2]) - source[2:]))

    def hessian(source: Array) -> Array:
        return 2.0 * gram

    return FaceCertificate(
        0,
        0,
        (float(bounds[0, 0]), float(bounds[0, 1])),
        (float(bounds[1, 0]), float(bounds[1, 1])),
        prepare_interval_function(value, 5),
        prepare_interval_function(gradient, 5),
        prepare_interval_function(hessian, 5),
    )


@pytest.mark.parametrize("dimension", (0, 1, 2), ids=("vertex", "edge", "face"))
def test_stationary_representative_retains_stratum_enclosures(
    box_model: BRepModel, dimension: int
) -> None:
    geometry = box_model.geometry
    if geometry is None:
        raise RuntimeError("box_model.geometry must contain native geometry; got None")
    face = 0
    edge = geometry.coedge_edges[geometry.face_loops[face][0][0]]
    if dimension == 2:
        entity = face
        uv = np.mean(np.asarray(box_model.parameter_bounds)[face], axis=0)
        parameter_box = np.stack((uv, uv))
        point_box = box_model.patches[face].bounding_box(parameter_box)
    elif dimension == 1:
        entity = edge
        parameter = float(np.mean(np.asarray(geometry.edge_ranges)[edge]))
        parameter_box = np.asarray([[parameter], [parameter]], dtype=np.float64)
        point_box = geometry.curves[geometry.edge_curves[edge]].bounding_box(
            parameter, parameter
        )
    else:
        entity = geometry.edge_vertices[edge][0]
        parameter_box = np.empty((2, 0), dtype=np.float64)
        point = np.asarray(geometry.vertex_points)[entity]
        point_box = np.stack((point, point))
    lower, upper = source_distance_bounds(point_box, np.zeros(3, dtype=np.float64))
    root = StationaryRoot(face, dimension, entity, parameter_box, point_box, lower, upper)

    representative = stationary_root_representative(box_model, root)

    support = np.asarray(
        box_model.patches[face].evaluate(
            jnp.asarray(representative.uv, dtype=jnp.float64)
        )
    )
    np.testing.assert_allclose(
        representative.point, support, rtol=0.0, atol=_ROUNDING_BOUND
    )
    assert np.all(np.isfinite(representative.uv))
    assert representative.regular
    assert representative.uv_error >= 0.0
    assert representative.point_error >= np.linalg.norm(
        np.maximum(
            np.abs(point_box[0] - representative.point),
            np.abs(point_box[1] - representative.point),
        )
    )
    if dimension == 2:
        assert representative.edge == -1
        assert np.isnan(representative.parameter)
    else:
        assert representative.edge >= 0
        assert np.isfinite(representative.parameter)


def test_stationary_edge_search_preserves_vertex_and_edge_identities(
    box_model: BRepModel,
) -> None:
    geometry = box_model.geometry
    if geometry is None:
        raise RuntimeError("box_model.geometry must contain native geometry; got None")
    domains = tuple(
        brep_trim_domain(geometry, face, patch, tolerance=_ROUNDING_BOUND)
        for face, patch in enumerate(box_model.patches)
    )
    isolation = isolate_closest_stationary(
        box_model,
        np.asarray((0.5, 0.5, -1.0), dtype=np.float64),
        certificates=(),
        trim_domains=domains,
        distance_upper=8.0,
        tolerance=_ROUNDING_BOUND,
        parameter_tolerance=_ROUNDING_BOUND,
        max_boxes=1024,
        max_depth=32,
    )

    assert isolation.complete
    assert isolation.unresolved_boxes == 0
    assert np.isinf(isolation.unresolved_distance_lower)
    vertices = tuple(root.entity for root in isolation.roots if root.dimension == 0)
    assert vertices == tuple(range(8))
    edges = tuple(root.entity for root in isolation.roots if root.dimension == 1)
    assert edges
    assert len(edges) == len(set(edges))
    assert all(root.distance_lower <= root.distance_upper for root in isolation.roots)


def test_stationary_work_limit_retains_unresolved_distance_floor(
    box_model: BRepModel,
) -> None:
    geometry = box_model.geometry
    if geometry is None:
        raise RuntimeError("box_model.geometry must contain native geometry; got None")
    domains = tuple(
        brep_trim_domain(geometry, face, patch, tolerance=_ROUNDING_BOUND)
        for face, patch in enumerate(box_model.patches)
    )
    isolation = isolate_closest_stationary(
        box_model,
        np.asarray((0.5, 0.5, -1.0), dtype=np.float64),
        certificates=(),
        trim_domains=domains,
        distance_upper=8.0,
        tolerance=_ROUNDING_BOUND,
        parameter_tolerance=_ROUNDING_BOUND,
        max_boxes=1,
        max_depth=32,
    )

    assert not isolation.complete
    assert isolation.boxes_processed == 1
    assert isolation.unresolved_boxes > 0
    assert 0.0 <= isolation.unresolved_distance_lower <= 8.0
    assert sum(root.dimension == 0 for root in isolation.roots) == 8


def test_stationary_face_search_certifies_an_interior_linear_root(
    box_model: BRepModel,
) -> None:
    geometry = box_model.geometry
    if geometry is None:
        raise RuntimeError("box_model.geometry must contain native geometry; got None")
    patch = box_model.patches[0]
    assert isinstance(patch, PlanePatch)
    bounds = np.asarray(box_model.parameter_bounds)[0]
    uv = np.mean(bounds, axis=0)
    expected = np.asarray(patch.evaluate(jnp.asarray(uv, dtype=jnp.float64)))
    normal = np.cross(np.asarray(patch.first_axis), np.asarray(patch.second_axis))
    point = expected + normal

    certificate = _plane_certificate(patch, bounds)
    domains = tuple(
        brep_trim_domain(geometry, face, source, tolerance=_ROUNDING_BOUND)
        for face, source in enumerate(box_model.patches)
    )
    isolation = isolate_closest_stationary(
        box_model,
        point,
        certificates=(certificate,),
        trim_domains=domains,
        faces=(0,),
        distance_upper=8.0,
        tolerance=_ROUNDING_BOUND,
        parameter_tolerance=_ROUNDING_BOUND,
        max_boxes=1024,
        max_depth=32,
    )

    assert isolation.complete
    roots = tuple(root for root in isolation.roots if root.dimension == 2)
    assert len(roots) == 1
    representative = stationary_root_representative(box_model, roots[0])
    np.testing.assert_allclose(
        representative.point, expected, rtol=0.0, atol=_ROUNDING_BOUND
    )
    assert roots[0].distance_lower <= np.linalg.norm(normal) <= roots[0].distance_upper


def test_stationary_face_search_refines_coarse_sloped_trim_enclosures() -> None:
    height = float(np.sqrt(3.0) * 0.3)
    model = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (0.6, 0.0), (0.3, height))),
        ),
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    geometry = model.geometry
    if geometry is None:
        raise RuntimeError(
            "triangle_model.geometry must contain native geometry; got None"
        )
    patch = model.patches[0]
    assert isinstance(patch, PlanePatch)
    bounds = np.asarray(model.parameter_bounds)[0]
    uv = np.asarray((0.3, height / 3.0), dtype=np.float64)
    expected = np.asarray(patch.evaluate(jnp.asarray(uv, dtype=jnp.float64)))
    normal = np.cross(np.asarray(patch.first_axis), np.asarray(patch.second_axis))
    # Twice the chart width retains each affine edge as one coarse arc. This
    # controls preparation granularity, not a projection acceptance tolerance.
    domain = brep_trim_domain(
        geometry, 0, patch, tolerance=2.0 * float(np.max(bounds[1] - bounds[0]))
    )
    assert bool(domain.in_band(jnp.asarray(uv[None], dtype=jnp.float64))[0])

    isolation = isolate_closest_stationary(
        model,
        expected + normal,
        certificates=(_plane_certificate(patch, bounds),),
        trim_domains=(domain,),
        faces=(0,),
        distance_upper=8.0,
        tolerance=_ROUNDING_BOUND,
        parameter_tolerance=_ROUNDING_BOUND,
        max_boxes=1024,
        max_depth=32,
    )

    assert isolation.complete
    assert isolation.unresolved_boxes == 0
    roots = tuple(root for root in isolation.roots if root.dimension == 2)
    assert len(roots) == 1
    representative = stationary_root_representative(model, roots[0])
    np.testing.assert_allclose(
        representative.point, expected, rtol=0.0, atol=_ROUNDING_BOUND
    )
    assert roots[0].distance_lower <= np.linalg.norm(normal) <= roots[0].distance_upper


@pytest.mark.parametrize(
    ("uv_fraction", "inside"),
    (((0.5, 1.0 / 3.0), True), ((0.92, 0.85), False)),
    ids=("trim-interior-root", "trim-exterior-root"),
)
@pytest.mark.parametrize("max_boxes", (1024, 3), ids=("complete", "box-limit"))
def test_stationary_face_search_keeps_trim_decisions_for_excluded_boxes(
    uv_fraction: tuple[float, float], inside: bool, max_boxes: int
) -> None:
    height = float(np.sqrt(3.0) * 0.3)
    model = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (0.6, 0.0), (0.3, height))),
        ),
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    geometry = model.geometry
    if geometry is None:
        raise RuntimeError(
            "triangle_model.geometry must contain native geometry; got None"
        )
    patch = model.patches[0]
    assert isinstance(patch, PlanePatch)
    bounds = np.asarray(model.parameter_bounds)[0]
    # The squared-distance stationary point of the plane is this chart point;
    # the exterior case lies inside the chart box but outside the triangle.
    uv = bounds[0] + np.asarray(uv_fraction) * (bounds[1] - bounds[0])
    expected = np.asarray(patch.evaluate(jnp.asarray(uv, dtype=jnp.float64)))
    normal = np.cross(np.asarray(patch.first_axis), np.asarray(patch.second_axis))
    domain = brep_trim_domain(geometry, 0, patch, tolerance=_ROUNDING_BOUND)
    classified = domain.classify(uv[None])
    assert bool(classified.inside[0]) is inside

    def isolate() -> ClosestStationaryIsolation:
        return isolate_closest_stationary(
            model,
            expected + normal,
            certificates=(_plane_certificate(patch, bounds),),
            trim_domains=(domain,),
            faces=(0,),
            distance_upper=8.0,
            tolerance=_ROUNDING_BOUND,
            parameter_tolerance=_ROUNDING_BOUND,
            max_boxes=max_boxes,
            max_depth=32,
        )

    isolation = isolate()
    # Request-local trim refinements never change a repeated, fresh query.
    assert pickle.dumps(isolation) == pickle.dumps(isolate())
    roots = tuple(root for root in isolation.roots if root.dimension == 2)
    if max_boxes == 3:
        # The box allowance is consumed only by visited/tightened boxes.
        assert not isolation.complete
        assert isolation.boxes_processed == 3
        assert 0.0 <= isolation.unresolved_distance_lower <= 8.0
        return
    assert isolation.complete
    assert isolation.unresolved_boxes == 0
    if inside:
        assert len(roots) == 1
        representative = stationary_root_representative(model, roots[0])
        np.testing.assert_allclose(
            representative.point, expected, rtol=0.0, atol=_ROUNDING_BOUND
        )
    else:
        # A source-equation root outside the trim never becomes a face root,
        # even though its box is neither interval- nor Krawczyk-excluded.
        assert not roots
