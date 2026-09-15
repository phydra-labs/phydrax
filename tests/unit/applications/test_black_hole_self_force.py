import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.compact_objects._self_force import (
    FirstOrderSelfForceModeSum,
    ModeSumRegularizationParameters,
)


jax.config.update("jax_enable_x64", True)


def test_mode_sum_subtracts_regularization_and_sums_fitted_tail():
    regularization = ModeSumRegularizationParameters(
        jnp.asarray((1.0, -0.5)),
        jnp.asarray((2.0, 0.25)),
        jnp.asarray((3.0, -0.75)),
        jnp.asarray((0.1, -0.2)),
        side="plus",
        gauge="Lorenz",
        worldline_id="circular-r0=10M",
        component_basis="Schwarzschild-coordinate",
    )
    ell_max = 14
    angular_order = jnp.arange(ell_max + 1, dtype=float) + 0.5
    coefficient_2 = jnp.asarray((0.2, -0.05))
    coefficient_4 = jnp.asarray((0.05, 0.025))
    retarded = (
        angular_order[:, None] * regularization.A
        + regularization.B
        + regularization.C / angular_order[:, None]
        + coefficient_2 / angular_order[:, None] ** 2
        + coefficient_4 / angular_order[:, None] ** 4
    )
    calculator = FirstOrderSelfForceModeSum(
        regularization,
        ell_max,
        tail_window=7,
        tail_fit_tolerance=1.0e-9,
        maximum_tail_fraction=1.0,
    )
    result = eqx.filter_jit(calculator.calculate)(retarded)

    expected = (
        coefficient_2 * (np.pi**2 / 2.0)
        + coefficient_4 * (np.pi**4 / 6.0)
        - regularization.D
    )
    np.testing.assert_allclose(result.self_force, expected, rtol=2.0e-11)
    np.testing.assert_allclose(
        result.tail.fitted_coefficients,
        jnp.stack((coefficient_2, coefficient_4)),
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    assert result.regularized_modes.shape == retarded.shape
    assert int(result.tail.fit_start_ell) == ell_max + 1 - calculator.tail_window
    assert bool(result.tail.fitted)
    assert bool(result.tail.qualified)
    assert bool(result.converged)
    assert bool(result.physically_valid)
    assert bool(result.qualified)
    assert bool(result.derivative_valid)


def test_mode_sum_reports_unresolved_tail_instead_of_claiming_convergence():
    regularization = ModeSumRegularizationParameters(
        0.0,
        0.0,
        0.0,
        0.0,
        side="average",
        gauge="radiation",
        worldline_id="eccentric-segment",
        component_basis="orthonormal-tetrad",
    )
    ell_max = 10
    angular_order = jnp.arange(ell_max + 1, dtype=float) + 0.5
    unresolved_modes = (-1.0) ** jnp.arange(ell_max + 1) / angular_order
    calculator = FirstOrderSelfForceModeSum(
        regularization,
        ell_max,
        tail_window=6,
        tail_fit_tolerance=1.0e-5,
        maximum_tail_fraction=0.1,
    )
    result = calculator.calculate(unresolved_modes)

    assert bool(result.finite)
    assert bool(result.tail.fitted)
    assert not bool(result.tail.qualified)
    assert not bool(result.converged)
    assert not bool(result.qualified)
    assert not bool(result.derivative_valid)
    assert int(result.status) in (1, 2)
