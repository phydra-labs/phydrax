"""Certified analytic face and intersection-edge closest points vs independent oracles."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.brep import (
    _analytic_carrier_closest as carrier_owner,
    _analytic_edge_closest as edge_owner,
    _query as query_owner,
)
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._patches import (
    ConePatch,
    CylinderPatch,
    PlanePatch,
    SpherePatch,
    TorusPatch,
)
from phydrax.geometry.brep._projection_contracts import BRepProjectionStatus


pytestmark = pytest.mark.meshcore

_SI = phx.SpatialCoordinateContract.si()
_LAZY = phx.geometry.BRepTessellationPolicy(realize=False)
_TOLERANCE = 1.0e-9
_X, _Y, _Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


# ----------------------------------------------------------- independent oracles


def _curve_closest(
    curve: Callable[[jax.Array], jax.Array], period: float, point: np.ndarray
) -> tuple[float, np.ndarray]:
    """Dense sampling plus Newton on d/dt |c(t) - p|^2 of an explicit parametrization."""
    target = jnp.asarray(point)

    def squared(t: jax.Array) -> jax.Array:
        return jnp.sum((curve(t) - target) ** 2)

    grid = jnp.linspace(0.0, period, 40001)
    t = grid[int(jnp.argmin(jax.vmap(squared)(grid)))]
    first, second = jax.grad(squared), jax.grad(jax.grad(squared))
    for _ in range(40):
        t = t - first(t) / second(t)
    closest = np.asarray(curve(t))
    return float(np.linalg.norm(closest - point)), closest


def _quartic(t: jax.Array) -> jax.Array:
    y, z = 0.6 * jnp.sin(t), 0.6 * jnp.cos(t)
    return jnp.stack((jnp.sqrt(1.0 - y * y), y, z))


_SPHERE_CENTER = np.asarray((0.25, -0.5, 0.125))
_SPHERE_CIRCLE = math.sqrt(2.25 - (0.5 - 0.125) ** 2)
_TORUS_CIRCLE = 2.0 + math.sqrt(0.25 - 0.0625)
_CONE_SEMI = 0.3
_CONE_CIRCLE = 1.0 + 0.5 * math.tan(_CONE_SEMI)

# (name, carriers, explicit independent curve parametrization over one turn)
_PAIRS: tuple[
    tuple[str, tuple[object, object], Callable[[jax.Array], jax.Array]], ...
] = (
    (
        "cylinder-cylinder quartic",
        (
            CylinderPatch((0.0, 0.0, -1.2), _X, _Y, _Z, 1.0),
            CylinderPatch((-1.5, 0.0, 0.0), _Y, _Z, _X, 0.6),
        ),
        _quartic,
    ),
    (
        "sphere-plane circle",
        (
            SpherePatch(tuple(_SPHERE_CENTER), _Y, _Z, _X, 1.5),
            PlanePatch((0.0, 0.0, 0.5), _X, _Y),
        ),
        lambda t: jnp.stack(
            (
                0.25 + _SPHERE_CIRCLE * jnp.cos(t),
                -0.5 + _SPHERE_CIRCLE * jnp.sin(t),
                jnp.full_like(t, 0.5),
            )
        ),
    ),
    (
        "cylinder-oblique-plane ellipse",
        (
            CylinderPatch((0.0, 0.0, 0.0), _X, _Y, _Z, 1.0),
            PlanePatch((0.0, 0.0, 0.25), (1.0, 0.0, 1.0), _Y),
        ),
        lambda t: jnp.stack((jnp.cos(t), jnp.sin(t), 0.25 + jnp.cos(t))),
    ),
    (
        "torus-plane outer circle",
        (
            TorusPatch((0.0, 0.0, 0.0), _X, _Y, _Z, 2.0, 0.5),
            PlanePatch((0.0, 0.0, 0.25), _X, _Y),
        ),
        lambda t: jnp.stack(
            (
                _TORUS_CIRCLE * jnp.cos(t),
                _TORUS_CIRCLE * jnp.sin(t),
                jnp.full_like(t, 0.25),
            )
        ),
    ),
    (
        "cone-plane circle",
        (
            ConePatch((0.0, 0.0, 0.0), _X, _Y, _Z, 1.0, _CONE_SEMI),
            PlanePatch((0.0, 0.0, 0.5), _Y, _X),
        ),
        lambda t: jnp.stack(
            (
                _CONE_CIRCLE * jnp.cos(t),
                _CONE_CIRCLE * jnp.sin(t),
                jnp.full_like(t, 0.5),
            )
        ),
    ),
)


def _certify(
    carriers: tuple[object, object],
    points: np.ndarray,
    candidates: np.ndarray,
    *,
    maximum_boxes: int = 8192,
) -> edge_owner.EdgeLowerBounds:
    pair = edge_owner.surface_pair_carriers(*carriers)
    assert pair is not None
    distances = np.linalg.norm(candidates - points, axis=1)
    with coordinate_source_request((points, candidates)) as budget:
        return edge_owner.edge_distance_lower_bounds(
            pair,
            points,
            candidates,
            distances,
            _TOLERANCE,
            end_boxes=np.zeros((0, 2, 3), dtype=np.float64),
            maximum_boxes=maximum_boxes,
            maximum_depth=48,
            budget=budget,
        )


# ------------------------------------------------------------- edge certification


@pytest.mark.parametrize(
    ("name", "carriers", "curve"), _PAIRS, ids=[p[0] for p in _PAIRS]
)
def test_edge_lower_bound_certifies_independent_closest_points(
    name: str,
    carriers: tuple[object, object],
    curve: Callable[[jax.Array], jax.Array],
) -> None:
    del name
    rng = np.random.default_rng(7)
    count = 12
    base = np.stack(
        [
            np.asarray(curve(jnp.asarray(t)))
            for t in rng.uniform(0.0, 2.0 * math.pi, count)
        ]
    )
    direction = rng.normal(size=base.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    size = 10.0 ** rng.uniform(-6.0, -1.5, count)
    points = base + size[:, None] * direction
    oracle = [_curve_closest(curve, 2.0 * math.pi, point) for point in points]
    exact = np.asarray([value[0] for value in oracle])
    candidates = np.stack([value[1] for value in oracle])
    certified = _certify(carriers, points, candidates)
    distances = np.linalg.norm(candidates - points, axis=1)
    assert np.all(certified.decided)
    assert np.all(distances - certified.lower <= _TOLERANCE)
    # Soundness: never above the independent minimum distance.
    assert np.all(certified.lower <= exact + 1.0e-12)
    assert np.all(certified.boxes >= 1)


def test_edge_lower_bound_refuses_a_farther_candidate() -> None:
    _, carriers, curve = _PAIRS[0]
    point = np.asarray(curve(jnp.asarray(0.7))) + np.asarray((0.0, 0.01, 0.02))
    exact, _ = _curve_closest(curve, 2.0 * math.pi, point)
    wrong = np.asarray(curve(jnp.asarray(1.0)))
    certified = _certify(carriers, point[None], wrong[None])
    assert not certified.decided[0]
    assert certified.lower[0] <= exact + 1.0e-12


def test_edge_lower_bound_keeps_ties_and_continua_undecided() -> None:
    _, quartic, _ = _PAIRS[0]
    # (1, 0, 0) is equidistant from the stationary points (1, 0, +-0.6).
    tie = _certify(quartic, np.asarray([[1.0, 0.0, 0.0]]), np.asarray([[1.0, 0.0, 0.6]]))
    assert not tie.decided[0]
    _, sphere_plane, circle = _PAIRS[1]
    # The circle center sees a continuum of closest points.
    center = np.asarray([[0.25, -0.5, 0.5]])
    on_circle = np.asarray(circle(jnp.asarray(0.3)))[None]
    continuum = _certify(sphere_plane, center, on_circle, maximum_boxes=256)
    assert not continuum.decided[0]
    assert continuum.lower[0] <= _SPHERE_CIRCLE


def test_edge_carrier_jets_enclose_sampled_system_values() -> None:
    """Interval F and Jacobian enclose independently differentiated samples."""
    rng = np.random.default_rng(3)

    def implicit(patch: object) -> Callable[[jax.Array], jax.Array]:
        match patch:
            case PlanePatch():
                normal = jnp.cross(patch.first_axis, patch.second_axis)
                return lambda q: jnp.dot(q - patch.origin, normal)
            case CylinderPatch():
                return lambda q: (
                    jnp.dot(q - patch.origin, patch.first_axis) ** 2
                    + jnp.dot(q - patch.origin, patch.second_axis) ** 2
                    - patch.radius**2
                )
            case ConePatch():
                return lambda q: (
                    jnp.dot(q - patch.origin, patch.first_axis) ** 2
                    + jnp.dot(q - patch.origin, patch.second_axis) ** 2
                    - (
                        patch.reference_radius
                        + jnp.dot(q - patch.origin, patch.axis)
                        * jnp.tan(patch.semi_angle)
                    )
                    ** 2
                )
            case SpherePatch():
                return lambda q: jnp.sum((q - patch.center) ** 2) - patch.radius**2
            case TorusPatch():
                return lambda q: (
                    (
                        jnp.sum((q - patch.center) ** 2)
                        + patch.major_radius**2
                        - patch.minor_radius**2
                    )
                    ** 2
                    - 4.0
                    * patch.major_radius**2
                    * (
                        jnp.dot(q - patch.center, patch.first_axis) ** 2
                        + jnp.dot(q - patch.center, patch.second_axis) ** 2
                    )
                )
        raise TypeError(type(patch))

    for _, carriers, _ in _PAIRS:
        pair = edge_owner.surface_pair_carriers(*carriers)
        assert pair is not None
        first, second = (implicit(patch) for patch in carriers)
        point = rng.normal(size=3)

        def system(q: jax.Array) -> jax.Array:
            tangent = jnp.cross(jax.grad(first)(q), jax.grad(second)(q))
            return jnp.stack((first(q), second(q), jnp.dot(q - point, tangent)))

        center = rng.normal(size=(6, 3))
        lower, upper = center - 0.05, center + 0.05
        value, jacobian = edge_owner._system(
            pair, np.broadcast_to(point, center.shape).copy(), lower, upper
        )
        samples = lower[:, None, :] + (upper - lower)[:, None, :] * rng.random((6, 32, 3))
        sampled = np.asarray(jax.vmap(jax.vmap(system))(jnp.asarray(samples)))
        derivative = np.asarray(
            jax.vmap(jax.vmap(jax.jacfwd(system)))(jnp.asarray(samples))
        )
        slack = 1.0e-12
        assert np.all(sampled >= value[0][:, None, :] - slack)
        assert np.all(sampled <= value[1][:, None, :] + slack)
        assert np.all(derivative >= jacobian[0][:, None] - slack)
        assert np.all(derivative <= jacobian[1][:, None] + slack)


def test_rounded_frames_are_refused_by_face_and_edge_routes() -> None:
    angle = 0.3
    rotated = CylinderPatch(
        (0.0, 0.0, 0.0),
        (math.cos(angle), math.sin(angle), 0.0),
        (-math.sin(angle), math.cos(angle), 0.0),
        _Z,
        1.0,
    )
    assert not carrier_owner.carrier_supported(rotated)
    assert (
        edge_owner.surface_pair_carriers(rotated, PlanePatch((0, 0, 0), _X, _Y)) is None
    )
    assert edge_owner.edge_carriers(object()) is None


# ------------------------------------------------------------------ face routes


def _cone_oracle(point: np.ndarray) -> np.ndarray:
    radial = math.hypot(point[0], point[1])
    start, end = np.asarray((1.0, 0.0)), np.asarray((0.5, 2.0))
    meridian = np.asarray((radial, point[2]))
    fraction = np.clip(
        np.dot(meridian - start, end - start) / np.sum((end - start) ** 2), 0, 1
    )
    foot = start + fraction * (end - start)
    return np.asarray((foot[0] * point[0] / radial, foot[0] * point[1] / radial, foot[1]))


def _torus_oracle(point: np.ndarray) -> np.ndarray:
    radial = math.hypot(point[0], point[1])
    core = np.asarray((2.0 * point[0] / radial, 2.0 * point[1] / radial, 0.0))
    offset = point - core
    return core + 0.5 * offset / np.linalg.norm(offset)


# (name, model, face, chart sampler (u, v) -> point near the face, oracle)
_FACES = (
    (
        "cylinder",
        lambda: phx.geometry.brep_cylinder(
            1.0, 2.0, coordinate_contract=_SI, tessellation=_LAZY
        ),
        1,
        lambda u, v, r: np.asarray((r * math.cos(u), r * math.sin(u), v)),
        lambda p: (
            np.asarray((p[0], p[1], 0.0)) / math.hypot(p[0], p[1])
            + np.asarray((0, 0, p[2]))
        ),
    ),
    (
        "frustum",
        lambda: phx.geometry.brep_cone(
            1.0, 0.5, 2.0, coordinate_contract=_SI, tessellation=_LAZY
        ),
        1,
        lambda u, v, r: np.asarray(
            ((1.0 - 0.25 * v) * r * math.cos(u), (1.0 - 0.25 * v) * r * math.sin(u), v)
        ),
        _cone_oracle,
    ),
    (
        "sphere",
        lambda: phx.geometry.brep_sphere(
            1.0, coordinate_contract=_SI, tessellation=_LAZY
        ),
        0,
        lambda u, v, r: (
            r
            * np.asarray(
                (math.cos(v) * math.cos(u), math.cos(v) * math.sin(u), math.sin(v))
            )
        ),
        lambda p: p / np.linalg.norm(p),
    ),
    (
        "torus",
        lambda: phx.geometry.brep_torus(
            2.0, 0.5, coordinate_contract=_SI, tessellation=_LAZY
        ),
        0,
        lambda u, v, r: np.asarray(
            (
                (2.0 + 0.5 * r * math.cos(v)) * math.cos(u),
                (2.0 + 0.5 * r * math.cos(v)) * math.sin(u),
                0.5 * r * math.sin(v),
            )
        ),
        _torus_oracle,
    ),
    (
        "plane cap",
        lambda: phx.geometry.brep_cylinder(
            1.0, 2.0, coordinate_contract=_SI, tessellation=_LAZY
        ),
        2,
        lambda u, v, r: np.asarray(
            (
                0.6 * math.cos(u) * (v / 2.0),
                0.6 * math.sin(u) * (v / 2.0),
                2.0 + (r - 1.0),
            )
        ),
        lambda p: np.asarray((p[0], p[1], 2.0)),
    ),
)


@pytest.mark.parametrize(
    ("name", "build", "face", "sample", "oracle"), _FACES, ids=[f[0] for f in _FACES]
)
def test_analytic_face_routes_match_independent_closest_points(
    name: str,
    build: Callable[[], phx.geometry.BRepModel],
    face: int,
    sample: Callable[[float, float, float], np.ndarray],
    oracle: Callable[[np.ndarray], np.ndarray],
) -> None:
    del name
    model = build()
    projection = phx.geometry.prepare_brep_projection(model)
    rng = np.random.default_rng(11)
    count = 24
    points = np.stack(
        [
            sample(float(u), float(v), float(r))
            for u, v, r in zip(
                rng.uniform(0.1, 2.0 * math.pi - 0.1, count),
                rng.uniform(0.2, 1.3, count),
                rng.uniform(0.97, 1.03, count),
                strict=True,
            )
        ]
    )
    result = projection.project(points, np.full(count, 2), np.full(count, face))
    expected = np.stack([oracle(point) for point in points])
    assert np.all(np.asarray(result.status) == int(BRepProjectionStatus.UNIQUE))
    np.testing.assert_allclose(
        np.asarray(result.points), expected, rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        np.asarray(result.residuals),
        np.linalg.norm(expected - points, axis=1),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_full_turn_seam_rows_report_seam_status() -> None:
    model = phx.geometry.brep_cylinder(
        1.0, 2.0, coordinate_contract=_SI, tessellation=_LAZY
    )
    projection = phx.geometry.prepare_brep_projection(model)
    points = np.asarray([[1.05, 0.0, 0.5], [0.95, 0.0, 1.5]])
    result = projection.project(points, np.full(2, 2), np.full(2, 1))
    assert np.all(np.asarray(result.status) == int(BRepProjectionStatus.SEAM))
    np.testing.assert_allclose(
        np.asarray(result.points),
        [[1.0, 0.0, 0.5], [1.0, 0.0, 1.5]],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_degenerate_carrier_minimizers_are_left_to_the_general_route() -> None:
    model = phx.geometry.brep_cylinder(
        1.0, 2.0, coordinate_contract=_SI, tessellation=_LAZY
    )
    projection = phx.geometry.prepare_brep_projection(model)
    query = projection.query
    points = np.asarray([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]])
    with coordinate_source_request((query, points)) as budget:
        batch = carrier_owner.closest_carrier_face(
            model,
            1,
            query.faces[1],
            points,
            query.tolerances,
            maximum_depth=query.policy.maximum_depth,
            preparations=query_owner.TrimRequestPreparations(),
            coordinate_budget=budget,
        )
    assert batch is not None
    # The axis point has a circle of carrier minimizers; the other row is isolated.
    assert bool(batch.unresolved[0]) and not bool(batch.unresolved[1])
    assert int(batch.status[0]) == int(BRepProjectionStatus.FAILED)
    np.testing.assert_allclose(batch.points[1], [1.0, 0.0, 1.0], rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    ("name", "build", "face", "sample", "oracle"),
    _FACES[:4],
    ids=[f[0] for f in _FACES[:4]],
)
def test_analytic_face_route_agrees_with_general_route(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    build: Callable[[], phx.geometry.BRepModel],
    face: int,
    sample: Callable[[float, float, float], np.ndarray],
    oracle: Callable[[np.ndarray], np.ndarray],
) -> None:
    del name, oracle
    model = build()
    projection = phx.geometry.prepare_brep_projection(model)
    point = sample(1.1, 0.7, 1.02)[None]
    fast = projection.project(point, np.full(1, 2), np.full(1, face))
    monkeypatch.setattr(
        query_owner.PreparedBRepQuery, "_carrier_closest", lambda *a, **k: None
    )
    general = projection.project(point, np.full(1, 2), np.full(1, face))
    assert int(np.asarray(fast.status)[0]) == int(np.asarray(general.status)[0])
    np.testing.assert_allclose(
        np.asarray(fast.points), np.asarray(general.points), rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        np.asarray(fast.residuals), np.asarray(general.residuals), rtol=0.0, atol=1.0e-12
    )


# ------------------------------------------------------ Boolean intersection edge


@pytest.fixture(scope="module")
def sphere_lens() -> phx.geometry.BRepModel:
    first = phx.geometry.brep_sphere(1.0, coordinate_contract=_SI, tessellation=_LAZY)
    second = phx.geometry.brep_sphere(
        1.0, center=(0.0, 1.0, 0.0), coordinate_contract=_SI, tessellation=_LAZY
    )
    return phx.geometry.brep.boolean_brep(
        first,
        second,
        "intersection",
        policy=phx.geometry.brep.BRepBooleanPolicy(tessellation=_LAZY),
    ).model


def test_boolean_intersection_edge_projection_is_certified_analytically(
    monkeypatch: pytest.MonkeyPatch, sphere_lens: phx.geometry.BRepModel
) -> None:
    geometry = sphere_lens.require_geometry()
    edges = [
        edge
        for edge, curve in enumerate(geometry.edge_curves)
        if curve != -1
        and isinstance(geometry.curves[curve], query_owner.IntersectionCurve)
    ]
    assert edges
    projection = phx.geometry.prepare_brep_projection(sphere_lens)

    def general(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "analytic intersection rows must not reach interval bisection"
        )

    monkeypatch.setattr(query_owner, "_polish_edge", general)
    rng = np.random.default_rng(5)
    radius = math.sqrt(3.0) / 2.0
    for edge in edges:
        curve = geometry.curves[geometry.edge_curves[edge]]
        assert isinstance(curve, query_owner.IntersectionCurve)
        first, last = (float(value) for value in np.asarray(geometry.edge_ranges)[edge])
        parameters = first + (last - first) * rng.uniform(0.1, 0.9, 8)
        base = np.stack(
            [np.asarray(curve.evaluate(jnp.asarray(value)).point) for value in parameters]
        )
        points = base + 1.0e-2 * rng.normal(size=base.shape)
        result = projection.project(points, np.ones(8, dtype=np.int64), np.full(8, edge))
        # Independent circle oracle: plane y = 1/2, centre (0, 1/2, 0), radius sqrt(3)/2.
        planar = points - np.asarray((0.0, 0.5, 0.0))
        planar[:, 1] = 0.0
        expected = np.asarray((0.0, 0.5, 0.0)) + radius * planar / np.linalg.norm(
            planar, axis=1, keepdims=True
        )
        assert np.all(np.asarray(result.status) == int(BRepProjectionStatus.UNIQUE))
        np.testing.assert_allclose(
            np.asarray(result.points), expected, rtol=0.0, atol=1.0e-9
        )
        np.testing.assert_allclose(
            np.asarray(result.residuals),
            np.linalg.norm(expected - points, axis=1),
            rtol=0.0,
            atol=1.0e-9,
        )


def test_boolean_intersection_domain_binds_placed_extension_roots(
    sphere_lens: phx.geometry.BRepModel,
) -> None:
    domain = phx.geometry.MeshingDomain.from_brep(sphere_lens)
    assert len(domain.patches) == len(sphere_lens.patches)
