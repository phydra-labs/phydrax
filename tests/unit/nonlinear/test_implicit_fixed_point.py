import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
import phydrax.linalg as la
import phydrax.nonlinear as nl


jax.config.update("jax_enable_x64", True)

_COUPLING = jnp.asarray([[0.6, -0.3], [0.2, 0.5]])


def _termination(*, maximum_steps=200):
    return nl.NonlinearTermination(
        absolute_residual=1e-13,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=maximum_steps,
    )


def _policy():
    return nl.ImplicitRootDerivativePolicy(
        tangent_linear_policy=la.LinearSolvePolicy(
            la.GMRES(),
            tolerance=la.TolerancePolicy(relative=1e-13, absolute=1e-14),
        )
    )


def _contraction(state, theta):
    return 0.5 * jnp.tanh(_COUPLING @ state) + jnp.asarray([theta, theta**2])


@pytest.mark.parametrize(
    "method",
    [
        nl.FixedPointIteration(),
        nl.FixedPointIteration(acceleration=nl.AndersonAcceleration(history=2)),
    ],
)
def test_fixed_point_derivatives_match_central_differences(method):
    problem = nl.FixedPointProblem(_contraction, problem_id="tanh-contraction")

    def fixed_point(theta):
        return nl.implicit_fixed_point_result(
            problem,
            jnp.zeros(2),
            method=method,
            termination=_termination(),
            derivative_policy=_policy(),
            args=theta,
        )

    theta = jnp.asarray(0.3)
    weights = jnp.asarray([1.0, -2.0])
    result = fixed_point(theta)
    _, tangent = jax.jvp(lambda value: fixed_point(value).state, (theta,), (1.0,))
    gradient = jax.grad(lambda value: weights @ fixed_point(value).state)(theta)
    step = 1e-5
    difference = (fixed_point(theta + step).state - fixed_point(theta - step).state) / (
        2 * step
    )

    assert bool(result.successful)
    assert jnp.allclose(result.state, _contraction(result.state, theta), atol=1e-12)
    assert result.provenance.problem_id == "tanh-contraction"
    assert result.component_evidence == (
        "mapping:determinism-undeclared",
        "mapping:regularity-undeclared",
    )
    assert jnp.allclose(tangent, difference, rtol=1e-7, atol=1e-9)
    assert jnp.allclose(gradient, weights @ difference, rtol=1e-7, atol=1e-9)


def test_fixed_point_derivative_refuses_unit_mapping_slope():
    # g(x) = x - (x - 1)^3 + theta has dg/dx = 1 at the fixed point x = 1.
    problem = nl.FixedPointProblem(lambda state, theta: state - (state - 1) ** 3 + theta)

    def fixed_point(theta):
        return nl.implicit_fixed_point_result(
            problem,
            jnp.ones(1),
            termination=_termination(),
            derivative_policy=_policy(),
            args=theta,
        ).state

    theta = jnp.asarray(0.0)
    forward = eqx.filter_jit(lambda value: jax.jvp(fixed_point, (value,), (1.0,))[1])
    reverse = eqx.filter_jit(jax.grad(lambda value: jnp.sum(fixed_point(value))))
    assert jnp.allclose(fixed_point(theta), 1.0)
    with pytest.raises(eqx.EquinoxRuntimeError, match="root derivative solve failed"):
        forward(theta)
    with pytest.raises(eqx.EquinoxRuntimeError, match="root derivative solve failed"):
        reverse(theta)


def test_failed_fixed_point_iteration_is_reported_and_not_differentiated():
    problem = nl.FixedPointProblem(_contraction)

    def fixed_point(theta):
        return nl.implicit_fixed_point_result(
            problem,
            jnp.zeros(2),
            termination=_termination(maximum_steps=2),
            derivative_policy=_policy(),
            args=theta,
        )

    result = fixed_point(jnp.asarray(0.3))
    assert result.status == int(nl.NonlinearStatus.MAXIMUM_STEPS_REACHED)
    reverse = eqx.filter_jit(jax.grad(lambda theta: jnp.sum(fixed_point(theta).state)))
    with pytest.raises(eqx.EquinoxRuntimeError, match="root solve failed"):
        reverse(jnp.asarray(0.3))


class _NetworkMapping(eqx.Module):
    network: eqx.Module

    def __call__(self, state, theta):
        return 0.1 * self.network(state, key=jax.random.key(3)) + theta


def _network(activation):
    return phx.nn.models.MLP(
        in_size=2,
        out_size=2,
        width_size=4,
        depth=1,
        activation=activation,
        key=jax.random.key(0),
    )


def _network_fixed_point(mapping):
    return nl.implicit_fixed_point_result(
        nl.FixedPointProblem(mapping),
        jnp.zeros(2),
        termination=_termination(),
        derivative_policy=_policy(),
        args=jnp.asarray([0.5, -0.25]),
    )


def test_fixed_point_refuses_mapping_components_without_classical_c1_regularity():
    with pytest.raises(ValueError, match="implicit-requires-c1"):
        _network_fixed_point(_NetworkMapping(_network(jax.nn.relu)))

    result = _network_fixed_point(_NetworkMapping(_network(jnp.tanh)))
    assert bool(result.successful)
    assert result.component_evidence == ("mapping.network:deterministic",)


def test_fixed_point_requires_fixed_point_iteration_and_tangent_policy():
    problem = nl.FixedPointProblem(_contraction)
    with pytest.raises(TypeError, match="FixedPointIteration"):
        nl.implicit_fixed_point_result(
            problem,
            jnp.zeros(2),
            method=nl.NewtonKrylov(),
            derivative_policy=_policy(),
            args=0.3,
        )
    with pytest.raises(ValueError, match="tangent linear policy"):
        nl.implicit_fixed_point_result(
            problem,
            jnp.zeros(2),
            derivative_policy=nl.ImplicitRootDerivativePolicy(),
            args=0.3,
        )
