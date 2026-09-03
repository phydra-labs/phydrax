from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _geometry():
    prepared = phx.discretization.SurfelSetPlan(
        jnp.asarray((0, 1, 2)),
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
        jnp.asarray((1.0, 2.0, 3.0)),
    ).prepare()
    return phx.discretization.SurfelGeometryPlan(prepared).materialize(
        prepared.reference_position,
        jnp.tile(jnp.asarray((0.0, 1.0)), (3, 1)),
        jnp.tile(jnp.asarray(((0.5,), (0.0,)))[None, ...], (3, 1, 1)),
    )


def test_surfel_quadrature_integrates_scalar_and_vector_fields() -> None:
    geometry = _geometry()
    scalar = phx.discretization.SurfelQuadraturePlan(
        geometry.discretization, deterministic=True
    ).evaluate(geometry, jnp.asarray((2.0, 2.0, 2.0)))
    assert bool(scalar.successful)
    np.testing.assert_allclose(scalar.total_measure, 6.0)
    np.testing.assert_allclose(scalar.integral, 12.0)
    np.testing.assert_allclose(scalar.average, 2.0)
    vector = phx.discretization.SurfelQuadraturePlan(geometry.discretization).evaluate(
        geometry,
        jnp.asarray(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))),
    )
    np.testing.assert_allclose(vector.integral, [4.0, 5.0])


def test_surfel_quadrature_gradient_uses_physical_measure() -> None:
    geometry = _geometry()
    plan = phx.discretization.SurfelQuadraturePlan(geometry.discretization)
    gradient = jax.grad(lambda value: plan.evaluate(geometry, value).integral)(
        jnp.ones((3,))
    )
    np.testing.assert_allclose(gradient, geometry.physical_surface_weight)
    result = eqx.filter_jit(plan.evaluate)(geometry, jnp.ones((3,)))
    assert bool(result.successful)
