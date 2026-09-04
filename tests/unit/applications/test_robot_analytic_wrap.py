#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.robotics._analytic_wrap import (
    AnalyticWrapStatus,
    PlanarCylinderRouteWrapPlan,
    SphereRouteWrapPlan,
)


_SPHERE_START = jnp.asarray((-2.0, 0.35, 0.0))
_SPHERE_END = jnp.asarray((2.1, 0.65, 0.0))


def test_sphere_wrap_satisfies_surface_tangency_and_arc_length():
    prepared = SphereRouteWrapPlan(48).prepare(jnp.zeros(3), 1.0)
    result = prepared.evaluate(_SPHERE_START, _SPHERE_END)

    assert bool(result.evidence.successful)
    assert bool(result.evidence.applied)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(result.tangent_start_m)), 1.0, atol=2.0e-7
    )
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(result.tangent_end_m)), 1.0, atol=2.0e-7
    )
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(result.surface_points_m), axis=-1),
        1.0,
        atol=2.0e-7,
    )
    assert float(result.evidence.endpoint_tangency_residual) < 2.0e-6
    np.testing.assert_allclose(
        result.surface_length_m,
        jnp.abs(result.signed_surface_angle_rad),
        rtol=2.0e-7,
    )
    assert result.total_length_m > jnp.linalg.norm(_SPHERE_END - _SPHERE_START)


def test_sphere_no_wrap_is_a_successful_direct_route_and_inside_fails():
    prepared = SphereRouteWrapPlan().prepare(jnp.zeros(3), 1.0)
    start = jnp.asarray((-2.0, 2.0, 0.0))
    end = jnp.asarray((2.0, 2.2, 0.0))
    direct = prepared.evaluate(start, end)

    assert bool(direct.evidence.successful)
    assert not bool(direct.evidence.applied)
    assert int(direct.evidence.status) & int(AnalyticWrapStatus.NO_WRAP)
    np.testing.assert_allclose(direct.total_length_m, jnp.linalg.norm(end - start))
    assert not bool(jnp.any(direct.surface_mask))

    inside = prepared.evaluate(jnp.asarray((0.5, 0.0, 0.0)), end)
    assert not bool(inside.evidence.successful)
    assert int(inside.evidence.status) & int(AnalyticWrapStatus.ENDPOINT_INSIDE)


def test_sphere_long_route_is_distinct_and_fixed_branch_differentiable():
    short = SphereRouteWrapPlan(sense="short").prepare(jnp.zeros(3), 1.0)
    long = SphereRouteWrapPlan(sense="long").prepare(jnp.zeros(3), 1.0)
    short_result = short.evaluate(_SPHERE_START, _SPHERE_END)
    long_result = long.evaluate(_SPHERE_START, _SPHERE_END)

    assert bool(short_result.evidence.successful)
    assert bool(long_result.evidence.successful)
    assert long_result.surface_length_m > short_result.surface_length_m
    compiled = eqx.filter_jit(short.evaluate)(_SPHERE_START, _SPHERE_END)
    np.testing.assert_allclose(compiled.total_length_m, short_result.total_length_m)
    derivative = jax.grad(
        lambda value: short.evaluate(value, _SPHERE_END).total_length_m
    )(_SPHERE_START)
    assert bool(jnp.all(jnp.isfinite(derivative)))


def test_planar_cylinder_wrap_and_bounded_failures_are_explicit():
    prepared = PlanarCylinderRouteWrapPlan(48).prepare(
        jnp.zeros(3), jnp.asarray((0.0, 0.0, 1.0)), 1.0, 4.0
    )
    result = prepared.evaluate(_SPHERE_START, _SPHERE_END)

    assert bool(result.evidence.successful)
    assert bool(result.evidence.applied)
    radial = np.asarray(result.surface_points_m)[:, :2]
    np.testing.assert_allclose(np.linalg.norm(radial, axis=-1), 1.0, atol=2.0e-7)
    assert float(result.evidence.endpoint_tangency_residual) < 2.0e-6

    nonplanar = prepared.evaluate(
        _SPHERE_START,
        _SPHERE_END + jnp.asarray((0.0, 0.0, 0.1)),
    )
    assert not bool(nonplanar.evidence.successful)
    assert int(nonplanar.evidence.status) & int(
        AnalyticWrapStatus.NONPLANAR_CYLINDER_ROUTE
    )

    outside = prepared.evaluate(
        _SPHERE_START + jnp.asarray((0.0, 0.0, 3.0)),
        _SPHERE_END + jnp.asarray((0.0, 0.0, 3.0)),
    )
    assert bool(outside.evidence.successful)
    assert not bool(outside.evidence.applied)
    assert int(outside.evidence.status) & int(
        AnalyticWrapStatus.OUTSIDE_BOUNDED_LATERAL_SURFACE
    )
