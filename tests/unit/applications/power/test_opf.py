#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np

from phydrax import optim
from phydrax.applications.power import (
    Branch,
    Bus,
    BusControl,
    compile_ac_opf,
    Generator,
    Load,
    PowerNetwork,
    PowerStudy,
    solve_ac_opf,
    solve_dc_opf,
    solve_dc_power_flow,
)


TWO_BUS_STUDY = PowerStudy((BusControl("r", "reference"), BusControl("d")))


def _congested(*, quadratic: Any = 0.0, q_load: Any = 0.0) -> Any:
    return PowerNetwork(
        (Bus("r", v_min=0.95, v_max=1.05), Bus("d", v_min=0.95, v_max=1.05)),
        (Branch("line", "r", "d", 0.0, 0.1, rate=0.4),),
        (
            Generator(
                "cheap",
                "r",
                p=0.3,
                p_min=0,
                p_max=2,
                q_min=-1,
                q_max=1,
                cost=(quadratic, 1, 0),
            ),
            Generator(
                "local",
                "d",
                p=0.7,
                p_min=0,
                p_max=2,
                q_min=-1,
                q_max=1,
                cost=(quadratic, 3, 0),
            ),
        ),
        (Load("demand", "d", 1.0, q_load),),
    )


def test_opf_scenario_1() -> None:
    network = PowerNetwork(
        (Bus("r"), Bus("d")),
        (Branch("t", "r", "d", 0, 0.1, tap=1.1, phase=0.2),),
        loads=(Load("load", "d", 0.5),),
    )
    result = solve_dc_power_flow(network, study=TWO_BUS_STUDY)
    assert bool(result.converged)
    np.testing.assert_allclose(result.angle, [0, -0.255], atol=1e-7)
    np.testing.assert_allclose(result.branch_from, [0.5], atol=1e-7)
    np.testing.assert_allclose(result.branch_to, [-0.5], atol=1e-7)
    np.testing.assert_allclose(result.reference_power, [0.5, 0], atol=1e-7)
    for quadratic in [0.0, 0.1]:
        result = solve_dc_opf(_congested(quadratic=quadratic), study=TWO_BUS_STUDY)
        assert bool(result.converged)
        np.testing.assert_allclose(result.generator_power, [0.4, 0.6], atol=2e-6)
        np.testing.assert_allclose(result.branch_from, [0.4], atol=2e-6)
        assert float(result.original_feasibility) < 1e-6
    compilation = compile_ac_opf(_congested(q_load=0.1), study=TWO_BUS_STUDY)
    result = solve_ac_opf(
        compilation,
        termination=optim.OptimizationTermination(
            maximum_steps=150,
            absolute_optimality=1e-7,
        ),
    )
    assert bool(result.converged), result.native_result.optimization.status
    assert float(result.original_feasibility) <= 1e-6
    assert float(jnp.max(jnp.abs(result.branch_from))) <= 0.400001
    assert float(jnp.max(jnp.abs(result.branch_to))) <= 0.400001
    assert 0.39 < float(result.generator_power[0].real) <= 0.400001
    np.testing.assert_allclose(result.bus_balance, 0, atol=1e-6)
    np.testing.assert_allclose(result.total_balance, 0, atol=1e-6)
    network = PowerNetwork(
        (Bus("r"),),
        (),
        (Generator("g", "r", p_min=0, p_max=0.2, cost=(0, 1, 0)),),
        (Load("load", "r", 1),),
    )
    result = solve_dc_opf(network, study=PowerStudy((BusControl("r", "reference"),)))
    assert not bool(result.converged)
