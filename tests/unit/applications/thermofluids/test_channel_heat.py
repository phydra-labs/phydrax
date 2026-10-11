from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.thermofluids import LiquidChannelHeatBalance
from phydrax.linalg import DenseLU, LinearSolvePolicy
from phydrax.nonlinear import NewtonKrylov, NonlinearSystemProblem, root


@pytest.mark.parametrize(
    ("wall", "direct", "expected"),
    [
        ((100.0, 200.0), (0.0, 0.0), (100500.0, 101250.0)),
        ((-100.0, 200.0), (200.0, 0.0), (100500.0, 101250.0)),
    ],
    ids=["wall-deposition", "signed-wall-exchange-plus-direct-heating"],
)
def test_coupled_channel_storage_outflow_and_exchange_balance(
    wall: tuple[float, float],
    direct: tuple[float, float],
    expected: tuple[float, float],
) -> None:
    channel = LiquidChannelHeatBalance(
        jnp.asarray([1.0, 1.0], dtype=jnp.float64), channel_id="two-cell-control"
    )
    previous = jnp.asarray([1e5, 1e5], dtype=jnp.float64)
    dt = jnp.asarray(10.0, dtype=jnp.float64)
    mass_flow = jnp.asarray(0.1, dtype=jnp.float64)
    inlet = jnp.asarray(1e5, dtype=jnp.float64)
    wall_power = jnp.asarray(wall, dtype=jnp.float64)
    direct_power = jnp.asarray(direct, dtype=jnp.float64)

    def residual(enthalpy: jax.Array, unused: None) -> jax.Array:
        del unused
        return channel.residual(
            enthalpy,
            previous,
            dt,
            mass_flow=mass_flow,
            inlet_enthalpy=inlet,
            wall_power=wall_power,
            direct_power=direct_power,
        )

    problem = NonlinearSystemProblem(residual, problem_id="channel-energy-control")
    method = NewtonKrylov(linear_policy=LinearSolvePolicy(DenseLU()))
    solved = jax.jit(lambda state: root(problem, state, method=method))(previous)
    assert bool(solved.successful)
    np.testing.assert_allclose(solved.state, expected, rtol=0.0, atol=1e-8)
    stored = channel.stored_energy_change(solved.state, previous)
    outflow = channel.outlet_power(solved.state, mass_flow, inlet)
    np.testing.assert_allclose(
        stored / dt + outflow,
        np.sum(wall) + np.sum(direct),
        rtol=0.0,
        atol=1e-9,
    )
