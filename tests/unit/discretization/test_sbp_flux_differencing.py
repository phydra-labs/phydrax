import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _grid(count=24, dimension=1):
    names = tuple("xyz"[:dimension])
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(
                count, periodic=True, endpoint=False
            )
            for _ in range(dimension)
        ),
        axis_names=names,
    )
    bounds = jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,))))
    return grid.prepare(bounds)


@pytest.mark.parametrize("order", (2, 4, 6, 8))
def test_periodic_sbp_derivative_has_skew_norm_identity(order):
    grid = _grid(max(12, order + 3))
    prepared = phx.discretization.SBPDerivativePlan(
        grid, "x", interior_order=order
    ).prepare()
    x = grid.axes[0].nodes
    result = prepared.operator.mv(jnp.sin(2.0 * jnp.pi * x))

    assert prepared.operator.stencil_set.kind == "periodic"
    assert jnp.max(jnp.abs(prepared.identity_residual())) < 1e-12
    assert jnp.max(jnp.abs(result - 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x))) < 0.3


def test_entropy_conservative_two_point_flux_is_symmetric_and_consistent():
    system = phx.equations.EulerSystem(2)
    flux = phx.discretization.EntropyConservativeEulerFluxPlan()
    left = system.primitive_to_conserved(jnp.asarray((1.1, 0.3, -0.2, 1.2)))
    right = system.primitive_to_conserved(jnp.asarray((0.9, -0.1, 0.4, 0.8)))

    for axis in range(2):
        np.testing.assert_allclose(
            flux.two_point_flux(system, left, right, axis),
            flux.two_point_flux(system, right, left, axis),
            atol=2e-12,
        )
        np.testing.assert_allclose(
            flux.two_point_flux(system, left, left, axis),
            system.physical_flux(left, axis),
            atol=2e-12,
        )
    assert flux.symmetric
    assert flux.consistent


def test_sbp_flux_differencing_preserves_constant_state_and_conserved_totals():
    grid = _grid(24)
    system = phx.equations.EulerSystem(1)
    discretization = phx.discretization.TensorSBPPlan(
        grid,
        field_name="state",
        component_names=system.component_names,
        interior_order=6,
    ).prepare()
    method = phx.discretization.SBPFluxDifferencingMethodPlan(
        phx.discretization.EntropyConservativeEulerFluxPlan()
    )
    problem = phx.equations.ConservationProblemIR("euler", "state", system, None)
    compiled = phx.equations.compile_conservation_problem(
        problem, discretization, method
    )
    constant = system.primitive_to_conserved(
        jnp.broadcast_to(jnp.asarray((1.0, 0.2, 1.0)), discretization.state_shape)
    )
    x = grid.axes[0].nodes
    smooth = system.primitive_to_conserved(
        jnp.stack(
            (
                1.0 + 0.05 * jnp.sin(2.0 * jnp.pi * x),
                0.2 + 0.03 * jnp.cos(2.0 * jnp.pi * x),
                jnp.ones_like(x),
            ),
            axis=-1,
        )
    )
    constant_rate = compiled(0.0, constant)
    smooth_rate = jax.jit(lambda value: compiled(0.0, value))(smooth)
    conservation_rate = jnp.sum(
        discretization.quadrature_weights[..., None] * smooth_rate, axis=0
    )

    np.testing.assert_allclose(constant_rate, 0.0, atol=2e-12)
    np.testing.assert_allclose(conservation_rate, 0.0, atol=2e-12)
    assert compiled.dynamics.report.sparse
    assert compiled.dynamics.report.pair_counts[0] < compiled.dynamics.report.dense_pair_count
    assert jnp.isfinite(compiled.stable_step(smooth))


def test_sbp_entropy_diagnostics_and_linearization_are_finite():
    grid = _grid(16)
    system = phx.equations.EulerSystem(1)
    discretization = phx.discretization.TensorSBPPlan(
        grid,
        field_name="state",
        component_names=system.component_names,
        interior_order=4,
    ).prepare()
    pair = phx.equations.ideal_gas_euler_entropy_pair(system)
    method = phx.discretization.SBPFluxDifferencingMethodPlan(
        phx.discretization.EntropyConservativeEulerFluxPlan(),
        entropy_diagnostics=True,
    )
    problem = phx.equations.ConservationProblemIR("euler", "state", system, None)
    compiled = phx.equations.compile_conservation_problem(
        problem, discretization, method, entropy_pair=pair
    )
    x = grid.axes[0].nodes
    state = system.primitive_to_conserved(
        jnp.stack(
            (
                1.0 + 0.05 * jnp.sin(2.0 * jnp.pi * x),
                0.2 * jnp.ones_like(x),
                jnp.ones_like(x),
            ),
            axis=-1,
        )
    )
    residual, diagnostics = compiled.residual_with_diagnostics(0.0, state)
    balance_terms = np.asarray(discretization.quadrature_weights[..., None] * residual)
    expected_rate = np.asarray(
        [math.fsum(balance_terms[:, index].tolist()) for index in range(balance_terms.shape[1])]
    )
    np.testing.assert_array_equal(diagnostics.conservation_rate, expected_rate)
    _, pushforward, pullback = compiled.linearize(0.0, state)
    tangent = jnp.ones_like(state) * 1e-3

    assert diagnostics is not None
    assert diagnostics.admissible
    assert jnp.abs(diagnostics.convective_entropy_rate) < 2e-10
    assert jnp.all(jnp.isfinite(pushforward(tangent)))
    assert jnp.all(jnp.isfinite(pullback(tangent)[0]))
    assert jnp.all(jnp.isfinite(residual))


def test_sbp_flux_differencing_rejects_unsupported_boundaries_and_systems():
    bounded = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(16),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    with pytest.raises(ValueError, match="periodic"):
        phx.discretization.TensorSBPPlan(
            bounded,
            component_names=("u",),
        )
    grid = _grid(16)
    scalar = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="scalar-sbp-rejection",
    )
    discretization = phx.discretization.TensorSBPPlan(
        grid, component_names=scalar.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR("scalar", "state", scalar, None)
    method = phx.discretization.SBPFluxDifferencingMethodPlan(
        phx.discretization.EntropyConservativeEulerFluxPlan()
    )
    with pytest.raises(TypeError, match="EulerSystem"):
        phx.equations.compile_conservation_problem(problem, discretization, method)
