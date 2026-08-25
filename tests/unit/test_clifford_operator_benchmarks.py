#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from tools.operator_benchmarks import (
    clifford_benchmark_scenarios,
    DifferentialCliffordOperatorBlock,
    entropy_euler_scenario,
    PeriodicCliffordLaplacian,
    run_clifford_decision_smoke,
)


def test_periodic_clifford_laplacian_recovers_fourier_mode():
    count = 8
    coordinate = 2.0 * jnp.pi * jnp.arange(count) / count
    values = jnp.sin(coordinate)[None, :, None]
    operator = PeriodicCliffordLaplacian((count,))
    assert jnp.allclose(operator(values), -values, atol=1e-12, rtol=1e-12)


def test_all_multigrade_scenarios_are_finite_and_schema_consistent():
    scenarios = clifford_benchmark_scenarios(4)
    assert tuple(value.name for value in scenarios) == (
        "clifford_incompressible_velocity_vorticity_2d",
        "clifford_entropy_euler_2d",
        "clifford_maxwell_plane_wave_3d",
    )
    for scenario in scenarios:
        assert scenario.inputs.shape == scenario.targets.shape
        assert scenario.inputs.shape[-1] == scenario.representation.packed_size
        assert jnp.all(jnp.isfinite(scenario.inputs))
        assert jnp.all(jnp.isfinite(scenario.targets))
        assert scenario.scenario_id


def test_entropy_euler_scenario_retains_pair_evidence_and_admissibility():
    scenario = entropy_euler_scenario(6)
    assert isinstance(scenario.diagnostics["pair_id"], str)
    assert bool(scenario.diagnostics["admissible"])
    assert float(scenario.diagnostics["integrated_relative_entropy"]) >= -1e-12
    assert jnp.isfinite(scenario.diagnostics["total_entropy"])


def test_differential_candidate_and_decision_smoke_execute_every_scenario():
    scenarios = clifford_benchmark_scenarios(4)
    for index, scenario in enumerate(scenarios):
        candidate = DifferentialCliffordOperatorBlock(
            scenario.representation,
            PeriodicCliffordLaplacian(scenario.grid_shape),
            latent_channels=1,
            residual_scale=0.01,
            key=jr.key(index),
        )
        output = candidate(scenario.inputs)
        assert output.shape == scenario.inputs.shape
        assert jnp.all(jnp.isfinite(output))

    report = run_clifford_decision_smoke(4, key=jr.key(9))
    assert report.passed
    assert len(report.scenario_names) == 3
    assert all(jnp.isfinite(value) for value in report.candidate_relative_errors)
    assert report.report_id
