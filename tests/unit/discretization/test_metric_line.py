import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _line():
    return phx.discretization.finite_volume.MetricLinePlan(
        np.asarray([0.0, 0.2, 0.7, 1.0]),
        np.asarray([2.0, 3.0, 5.0]),
        np.asarray([1.0, 1.5, 2.0, 4.0]),
        "fixture-metric-line",
    ).prepare()


def test_metric_line_face_incidence_telescopes_exactly():
    line = _line()
    flux = jnp.asarray([[2.0, -1.0], [3.0, 4.0], [-2.0, 5.0], [1.0, 7.0]])
    source = jnp.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    amount_rate = line.amount_rate(flux, source_density=source)
    evidence = line.conservation_evidence(flux, source_density=source)

    np.testing.assert_allclose(jnp.sum(amount_rate, axis=0), evidence.total_amount_rate)
    np.testing.assert_allclose(evidence.closure_residual, 0.0, atol=1.0e-14)
    assert bool(evidence.successful)


def test_metric_line_total_amount_uses_physical_cell_measures():
    line = _line()
    density = jnp.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    np.testing.assert_allclose(line.total_amount(density), [23.0, 46.0])


def test_metric_line_gradient_and_differentiation_are_consistent():
    line = _line()

    def objective(values):
        gradient = line.gradient(
            values,
            lower_boundary_value=jnp.asarray(0.0),
            upper_boundary_value=jnp.asarray(2.0),
        )
        return jnp.sum(gradient**2)

    values = jnp.asarray([0.2, 0.8, 1.4])
    direction = jnp.asarray([0.3, -0.2, 0.1])
    _, tangent = jax.jvp(objective, (values,), (direction,))
    gradient = jax.grad(objective)(values)
    np.testing.assert_allclose(tangent, jnp.vdot(gradient, direction), rtol=1.0e-12)
