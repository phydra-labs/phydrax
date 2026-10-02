#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic identifications, exact coordinate faces, and physical boundaries.

Oracles are the declared box bounds: a face of ``[a, b]`` along component ``c``
is ``x_c = a`` or ``x_c = b`` with transverse measure ``prod_{k != c} (b_k - a_k)``
and outward normal ``-e_c``/``+e_c``; the identification shift is ``(b_c - a_c) e_c``.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


_BOX = phx.domain.HyperRectangle(np.asarray([0.0, -1.0]), np.asarray([2.0, 3.0]))


def _identification_cases() -> dict[str, tuple[Any, str, int | None, float, float]]:
    interval = phx.domain.Interval1d(0.0, 2.0)
    time = phx.domain.TimeInterval(0.5, 1.5)
    return {
        "interval1d": (interval, "x", None, 0.0, 2.0),
        "scalar-time": (time, "t", None, 0.5, 1.5),
        "box-component-1": (_BOX, "x", 1, -1.0, 3.0),
        "product-space": (interval @ time, "x", None, 0.0, 2.0),
    }


@pytest.mark.parametrize(
    "case", tuple(_identification_cases()), ids=tuple(_identification_cases())
)
def test_identification_reads_bounds_period_and_shift_of_its_coordinate(
    case: str,
) -> None:
    domain, label, component, lower, upper = _identification_cases()[case]
    identification = phx.domain.PeriodicIdentification(domain, label, component=component)

    assert (identification.lower, identification.upper) == (lower, upper)
    assert identification.period == upper - lower
    shift = np.asarray(identification.shift())
    if component is None:
        np.testing.assert_allclose(shift, upper - lower)
    else:
        expected = np.zeros(2)
        expected[component] = upper - lower
        np.testing.assert_allclose(shift, expected)


def test_identification_identity_is_content_addressed_by_coordinate() -> None:
    first = phx.domain.PeriodicIdentification(_BOX, "x", component=0)
    rebuilt = phx.domain.PeriodicIdentification(
        phx.domain.HyperRectangle(np.asarray([0.0, -1.0]), np.asarray([2.0, 3.0])),
        "x",
        component=0,
    )
    other_axis = phx.domain.PeriodicIdentification(_BOX, "x", component=1)
    square = phx.domain.HyperRectangle(np.zeros(2), np.full(2, 2.0))
    equal_period_axes = (
        phx.domain.PeriodicIdentification(square, "x", component=0),
        phx.domain.PeriodicIdentification(square, "x", component=1),
    )

    assert first.identification_id == rebuilt.identification_id
    assert first.identification_id != other_axis.identification_id
    assert equal_period_axes[0].period == equal_period_axes[1].period
    assert (
        equal_period_axes[0].identification_id != equal_period_axes[1].identification_id
    )
    explicit = phx.domain.PeriodicIdentification(
        _BOX, "x", component=0, identification_id=" torus-x "
    )
    assert explicit.identification_id == "torus-x"


def test_identification_content_binds_transverse_and_product_support() -> None:
    wider = phx.domain.HyperRectangle(
        jnp.asarray([0.0, -1.0], dtype=jnp.float64),
        jnp.asarray([2.0, 4.0], dtype=jnp.float64),
    )
    original = phx.domain.PeriodicIdentification(_BOX, "x", component=0)
    changed = phx.domain.PeriodicIdentification(wider, "x", component=0)
    assert original.identification_id != changed.identification_id
    declared = phx.domain.PeriodicIdentification(
        _BOX, "x", component=0, identification_id="seam"
    )
    changed_declared = phx.domain.PeriodicIdentification(
        wider, "x", component=0, identification_id="seam"
    )
    assert declared.revision != changed_declared.revision
    first = _BOX @ phx.domain.TimeInterval(0.0, 1.0)
    second = _BOX @ phx.domain.TimeInterval(0.0, 2.0)
    assert (
        phx.domain.PeriodicIdentification(first, "x", component=0).revision
        != phx.domain.PeriodicIdentification(second, "x", component=0).revision
    )


@pytest.mark.parametrize(
    ("build", "error", "match"),
    (
        pytest.param(
            lambda: phx.domain.PeriodicIdentification(_BOX, "x"),
            ValueError,
            "requires an explicit component",
            id="vector-without-component",
        ),
        pytest.param(
            lambda: phx.domain.PeriodicIdentification(_BOX, "x", component=2),
            ValueError,
            "outside coordinate",
            id="component-out-of-range",
        ),
        pytest.param(
            lambda: phx.domain.PeriodicIdentification(
                _BOX,
                "x",
                component=True,
            ),
            TypeError,
            "component must be an integer",
            id="boolean-component",
        ),
        pytest.param(
            lambda: phx.domain.PeriodicIdentification(
                phx.domain.TimeInterval(0.0, 1.0), "t", component=0
            ),
            ValueError,
            "component=None",
            id="scalar-with-component",
        ),
        pytest.param(
            lambda: phx.domain.PeriodicIdentification(_BOX, "y", component=0),
            KeyError,
            "not a coordinate",
            id="unknown-label",
        ),
        pytest.param(
            lambda: phx.domain.PeriodicIdentification(
                _BOX, "x", component=0, identification_id="  "
            ),
            ValueError,
            "non-empty string",
            id="blank-identity",
        ),
        pytest.param(
            lambda: phx.domain.PeriodicIdentification(
                "x",
                "x",
            ),
            TypeError,
            None,
            id="not-a-domain",
        ),
    ),
)
def test_identification_refuses_invalid_coordinates(
    build: Callable[[], Any], error: type[Exception], match: str | None
) -> None:
    with pytest.raises(error, match=match):
        build()


def test_identification_refuses_geometry_without_cartesian_faces() -> None:
    disk = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=1.0).compile()
    )
    with pytest.raises(ValueError, match="does not expose Cartesian coordinate faces"):
        phx.domain.PeriodicIdentification(disk, "x", component=0)


@pytest.mark.parametrize(
    ("axis", "side", "value"),
    (
        pytest.param(0, "lower", 0.0, id="axis0-lower"),
        pytest.param(0, "upper", 2.0, id="axis0-upper"),
        pytest.param(1, "lower", -1.0, id="axis1-lower"),
        pytest.param(1, "upper", 3.0, id="axis1-upper"),
    ),
)
def test_coordinate_face_samples_its_face_with_outward_normal(
    axis: int, side: Any, value: float
) -> None:
    face = _BOX.component({"x": phx.domain.CoordinateFace(axis, side)})
    batch = face.sample(phx.domain.PointSampling(16), key=jr.key(0))
    assert isinstance(batch, phx.domain.PointBatch)
    points = np.asarray(batch.points["x"].data)
    transverse = 1 - axis
    lower = np.asarray([0.0, -1.0])
    upper = np.asarray([2.0, 3.0])

    np.testing.assert_array_equal(points[:, axis], value)
    assert np.all(points[:, transverse] >= lower[transverse])
    assert np.all(points[:, transverse] <= upper[transverse])
    # Transverse samples are spread across the face, not collapsed to a point.
    assert np.ptp(points[:, transverse]) > 0.5 * (upper - lower)[transverse]
    expected = np.zeros(2)
    expected[axis] = -1.0 if side == "lower" else 1.0
    np.testing.assert_array_equal(
        np.asarray(face.normals(batch, var="x").data), np.tile(expected, (16, 1))
    )
    np.testing.assert_array_equal(
        np.asarray(face.normal(var="x")(batch).data), np.tile(expected, (16, 1))
    )


@pytest.mark.parametrize(
    ("domain", "axis", "kind", "mass"),
    (
        pytest.param(phx.domain.Interval1d(0.0, 2.0), 0, "counting", 1.0, id="1d"),
        pytest.param(_BOX, 1, "hausdorff", 2.0, id="2d-edge"),
        pytest.param(_BOX, 0, "hausdorff", 4.0, id="2d-other-edge"),
        pytest.param(
            phx.domain.HyperRectangle(np.zeros(3), np.asarray([2.0, 3.0, 5.0])),
            1,
            "hausdorff",
            10.0,
            id="3d-face",
        ),
    ),
)
def test_coordinate_face_carries_exact_transverse_measure(
    domain: Any, axis: int, kind: str, mass: float
) -> None:
    face = domain.component({"x": phx.domain.CoordinateFace(axis, "upper")})
    (factor,) = face.factor_components
    assert factor.measure.kind == kind
    assert isinstance(face.mass, phx.domain.ExactMass)
    np.testing.assert_allclose(face.mass.value, mass)


def test_coordinate_face_explicit_points_must_lie_on_the_face() -> None:
    face = _BOX.component({"x": phx.domain.CoordinateFace(1, "upper")})
    on_face = face.points({"x": jnp.asarray([[0.5, 3.0], [1.75, 3.0]])})
    np.testing.assert_array_equal(
        np.asarray(on_face.points["x"].data), [[0.5, 3.0], [1.75, 3.0]]
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="selected coordinate face"):
        face.points({"x": jnp.asarray([[0.5, 3.0], [0.5, 2.5]])})


@pytest.mark.parametrize("point", ([0.5, 3.0], [2.0, 3.0], [0.0, 3.0]))
def test_coordinate_face_accepts_transverse_corners(point: list[float]) -> None:
    face = _BOX.component({"x": phx.domain.CoordinateFace(1, "upper")})
    coordinates = np.asarray([point], dtype=np.float64)
    np.testing.assert_array_equal(face.points({"x": coordinates})["x"].data, coordinates)


@pytest.mark.parametrize(
    "point", ([-0.1, 3.0], [2.1, 3.0], [np.nan, 3.0], [0.5, 3.000001])
)
def test_coordinate_face_refuses_outside_or_nonfinite_points(point: list[float]) -> None:
    face = _BOX.component({"x": phx.domain.CoordinateFace(1, "upper")})
    with pytest.raises(eqx.EquinoxRuntimeError, match="coordinate face"):
        face.points({"x": np.asarray([point], dtype=np.float64)})


@pytest.mark.parametrize(
    ("build", "error", "match"),
    (
        pytest.param(
            lambda: phx.domain.CoordinateFace(-1, "lower"),
            ValueError,
            "nonnegative",
            id="negative-axis",
        ),
        pytest.param(
            lambda: phx.domain.CoordinateFace(
                True,
                "lower",
            ),
            TypeError,
            "integer coordinate component",
            id="boolean-axis",
        ),
        pytest.param(
            lambda: phx.domain.CoordinateFace(
                0,
                "middle",  # ty: ignore[invalid-argument-type]
            ),
            ValueError,
            "side",
            id="unknown-side",
        ),
        pytest.param(
            lambda: _BOX.component({"x": phx.domain.CoordinateFace(2, "lower")}),
            ValueError,
            "outside the 2-dimensional coordinate",
            id="axis-beyond-dimension",
        ),
    ),
)
def test_coordinate_face_refuses_invalid_selection(
    build: Callable[[], Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        build()


def _selections(components: tuple[Any, ...]) -> list[tuple[str, Any]]:
    return [
        (label, selection)
        for component in components
        for label, selection in component.spec.by_label.items()
    ]


def test_physical_boundary_keeps_only_unidentified_box_faces_with_normals() -> None:
    identification = phx.domain.PeriodicIdentification(_BOX, "x", component=1)
    walls = phx.domain.physical_boundary(_BOX, (identification,))

    assert _selections(walls) == [
        ("x", phx.domain.CoordinateFace(0, "lower")),
        ("x", phx.domain.CoordinateFace(0, "upper")),
    ]
    for wall, sign in zip(walls, (-1.0, 1.0), strict=True):
        batch = wall.sample(phx.domain.PointSampling(4), key=jr.key(1))
        np.testing.assert_array_equal(
            np.asarray(wall.normals(batch, var="x").data),
            np.tile([sign, 0.0], (4, 1)),
        )


def test_physical_boundary_of_product_keeps_unidentified_factors() -> None:
    domain = phx.domain.Interval1d(0.0, 2.0) @ phx.domain.TimeInterval(0.0, 1.0)

    space_periodic = phx.domain.physical_boundary(
        domain, (phx.domain.PeriodicIdentification(domain, "x"),)
    )
    time_periodic = phx.domain.physical_boundary(
        domain, (phx.domain.PeriodicIdentification(domain, "t"),)
    )

    assert _selections(space_periodic) == [
        ("t", phx.domain.FixedStart()),
        ("t", phx.domain.FixedEnd()),
    ]
    assert _selections(time_periodic) == [("x", phx.domain.Boundary())]


@pytest.mark.parametrize(
    "domain",
    (
        pytest.param(_BOX, id="box"),
        pytest.param(
            phx.domain.Interval1d(0.0, 2.0) @ phx.domain.TimeInterval(0.0, 1.0),
            id="space-time",
        ),
    ),
)
def test_fully_periodic_support_has_no_physical_boundary(domain: Any) -> None:
    identifications = tuple(
        phx.domain.PeriodicIdentification(domain, label, component=component)
        for label in domain.labels
        for component in (
            (None,)
            if not isinstance(domain.factor(label), phx.domain.HyperRectangle)
            else range(domain.factor(label).spatial_dim)
        )
    )
    assert phx.domain.physical_boundary(domain, identifications) == ()


@pytest.mark.parametrize(
    ("identifications", "error", "match"),
    (
        pytest.param(
            (
                phx.domain.PeriodicIdentification(_BOX, "x", component=1),
                phx.domain.PeriodicIdentification(
                    _BOX, "x", component=1, identification_id="again"
                ),
            ),
            ValueError,
            "identified more than once",
            id="duplicate-coordinate",
        ),
        pytest.param(
            (phx.domain.PeriodicIdentification(phx.domain.Interval1d(0.0, 1.0), "x"),),
            ValueError,
            "live on the requested domain",
            id="foreign-domain",
        ),
        pytest.param(
            ("x",), TypeError, "PeriodicIdentification", id="not-identification"
        ),
    ),
)
def test_physical_boundary_refuses_inconsistent_identifications(
    identifications: tuple[Any, ...], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        phx.domain.physical_boundary(_BOX, identifications)


def test_identification_pairing_traces_source_lower_and_target_upper_faces() -> None:
    identification = phx.domain.PeriodicIdentification(_BOX, "x", component=1)
    seam = identification.pairing()
    batch = seam.component.sample(phx.domain.PointSampling(6), key=jr.key(2))
    transverse = np.asarray(batch.points["x"].data)[:, 0]
    field = _BOX.Function("x")(lambda x: x[0] + 10.0 * x[1])

    assert seam.self_seam and seam.topology == "periodic-interface"
    assert seam.identification is identification
    np.testing.assert_allclose(
        seam.trace(field, side=seam.source_side)(batch).data, transverse - 10.0
    )
    np.testing.assert_allclose(
        seam.trace(field, side=seam.target_side)(batch).data, transverse + 30.0
    )
    np.testing.assert_array_equal(
        np.asarray(seam.normal(batch).data), np.tile([0.0, 1.0], (6, 1))
    )


def _probe_seam(
    identification: Any, patch_id: str, target_map: Callable[[Any], Any]
) -> Any:
    return phx.domain.PairedSupport(
        identification.face("lower"),
        {"x": _BOX.Function("x")(target_map)},
        identification.face_map("lower"),
        pairing_id="probe",
        left_patch_id=patch_id,
        right_patch_id=patch_id,
        normal=identification.normal(),
        topology="periodic-interface",
        identification=identification,
    )


@pytest.mark.parametrize(
    ("target_map", "verified"),
    (
        pytest.param(lambda x: jnp.stack((2.0 + 0.0 * x[0], x[1])), True, id="shift"),
        pytest.param(
            lambda x: jnp.stack((2.0 + 0.0 * x[0], x[1] + 2.0)),
            False,
            id="transverse-twist-by-period",
        ),
        pytest.param(
            lambda x: jnp.stack((2.0 + 0.0 * x[0], x[1] + 0.5)),
            False,
            id="transverse-twist",
        ),
        pytest.param(
            lambda x: jnp.stack((1.0 + 0.0 * x[0], x[1])), False, id="half-period"
        ),
    ),
)
def test_periodic_audit_verifies_only_the_identification_shift(
    target_map: Callable[[Any], Any], verified: bool
) -> None:
    # Component 0 has period 2 while the transverse extent is 4, so a
    # transverse offset equal to the period stays inside the box.
    identification = phx.domain.PeriodicIdentification(_BOX, "x", component=0)
    cover = phx.domain.cartesian_subdomain_cover(
        _BOX, "x", (1, 1), periodic=(True, False)
    )
    (patch,) = cover.patches
    seam = _probe_seam(identification, patch.patch_id, target_map)
    batch = seam.component.sample(phx.domain.PointSampling(8), key=jr.key(3))

    assert seam.audit(batch, patch, patch).verified is verified


@pytest.mark.parametrize(
    "domain",
    (
        phx.domain.Interval1d(0.0, 2.0),
        phx.domain.HyperRectangle(
            jnp.asarray([0.0], dtype=jnp.float64), jnp.asarray([2.0], dtype=jnp.float64)
        ),
    ),
)
def test_point_face_has_empty_tangential_grid_and_unit_quadrature_mass(
    domain: Any,
) -> None:
    face = domain.component({"x": phx.domain.CoordinateFace(0, "upper")})
    grid = face.sample(phx.domain.GridSampling({"x": ()}))
    np.testing.assert_array_equal(grid.points["x"][0].data, [2.0])
    estimate = phx.integration.integrate(
        domain.Function("x")(lambda x: x[0] ** 2),
        phx.integration.over(face),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(3)),
    )
    assert estimate.value.data == pytest.approx(4.0)
    transport = phx.domain.reference_transport(
        domain, phx.domain.CoordinateFace(0, "upper")
    )
    assert transport is not None
    np.testing.assert_array_equal(
        transport.map(jnp.asarray([[0.1], [0.9]], dtype=jnp.float64)),
        [[2.0], [2.0]],
    )


def test_empty_tangential_request_is_not_an_empty_interior_grid() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _BOX.component().sample(phx.domain.GridSampling({"x": ()}))


@pytest.mark.parametrize("bounds", ([0.0, np.nan], [-np.inf, 1.0], [0.0, np.inf]))
def test_cartesian_geometries_refuse_nonfinite_bounds(bounds: list[float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        phx.domain.Interval1d(*bounds)
    with pytest.raises(ValueError, match="finite"):
        phx.domain.HyperRectangle(
            np.asarray([bounds[0]], dtype=np.float64),
            np.asarray([bounds[1]], dtype=np.float64),
        )


@pytest.mark.parametrize("grid_request", (0, -1, True, (2, 0), (True,)))
def test_face_grid_refuses_invalid_counts_before_materialization(
    grid_request: Any,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        phx.domain.GridSampling({"x": grid_request})


def test_fractional_face_projection_does_not_truncate_integer_coordinate_batches() -> (
    None
):
    domain = phx.domain.HyperRectangle(
        jnp.asarray([0.5, 0.0], dtype=jnp.float64),
        jnp.asarray([1.5, 2.0], dtype=jnp.float64),
    )
    identification = phx.domain.PeriodicIdentification(domain, "x", component=0)
    batch = domain.component().points({"x": jnp.asarray([[1.0, 1.0]], dtype=jnp.float64)})
    integer_batch = eqx.tree_at(
        lambda points: points.points["x"].data,
        batch,
        jnp.asarray([[1, 1]], dtype=jnp.int32),
    )
    projected = identification.face_map("upper")["x"](integer_batch).data
    np.testing.assert_array_equal(projected, [[1.5, 1.0]])


def test_periodic_identification_requires_representable_period() -> None:
    space = phx.domain.Interval1d(-1e308, 1e308)
    with pytest.raises(ValueError):
        phx.domain.PeriodicIdentification(space, "x")
