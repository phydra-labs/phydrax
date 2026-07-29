#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tools.operator_benchmarks.models import compatible_architectures
from tools.operator_benchmarks.runner import parameter_count, run_operator_benchmark
from tools.operator_benchmarks.scenarios import polynomial_poisson_scenario


class _ComplexParameters(eqx.Module):
    weight: jax.Array
    real_bias: jax.Array


def _collocation_poisson_solution(source):
    values = np.asarray(source)
    size = int(values.shape[-1])
    forcing = values**2 - np.mean(values**2, axis=(-2, -1), keepdims=True)
    frequency = 2.0 * np.pi * np.fft.fftfreq(size, d=1.0 / size)
    frequency_x, frequency_y = np.meshgrid(frequency, frequency, indexing="ij")
    negative_laplacian = frequency_x**2 + frequency_y**2
    forcing_hat = np.fft.fft2(forcing, axes=(-2, -1))
    solution_hat = np.zeros_like(forcing_hat)
    nonzero = negative_laplacian > 0.0
    solution_hat[..., nonzero] = forcing_hat[..., nonzero] / negative_laplacian[nonzero]
    return np.fft.ifft2(solution_hat, axes=(-2, -1)).real


def test_benchmark_parameter_count_uses_real_degrees_of_freedom():
    model = _ComplexParameters(
        weight=jnp.ones((2, 3), dtype=jnp.complex128),
        real_bias=jnp.ones((4,), dtype=jnp.float64),
    )

    assert parameter_count(model) == 16


def test_polynomial_poisson_targets_are_alias_free_reference_projections():
    scenario = polynomial_poisson_scenario(
        resolution=8,
        num_cases=3,
        polynomial_degree=2,
        maximum_frequency=3,
        seed=7,
    )
    source = scenario.train_batch.input("source")
    collocation_target = _collocation_poisson_solution(source.values)
    relative_gap = np.linalg.norm(
        collocation_target - scenario.train_target
    ) / np.linalg.norm(scenario.train_target)

    assert source.axes[0].periodic and source.axes[1].periodic
    assert source.axes[0].nodes[-1] < 1.0
    assert scenario.reference_evidence is not None
    assert scenario.reference_evidence.passed
    assert relative_gap > 1e-3
    assert scenario.evaluations[-1].shift == "resolution_transfer"
    assert scenario.evaluations[-1].target.shape[-2:] == (16, 16)
    assert scenario.validation is not None
    assert set(scenario.case_ids).isdisjoint(scenario.validation.case_ids)
    assert all(
        set(scenario.case_ids).isdisjoint(evaluation.case_ids)
        for evaluation in scenario.evaluations
    )


def test_controlled_hofno_candidates_train_on_corrected_scenario():
    scenario = polynomial_poisson_scenario(
        resolution=8,
        num_cases=2,
        polynomial_degree=2,
        maximum_frequency=3,
        seed=4,
    )
    architectures = {
        architecture.name: architecture
        for architecture in compatible_architectures(scenario, quick=True)
        if architecture.name.startswith("hofno_")
    }

    assert set(architectures) == {
        "hofno_order1",
        "hofno_order2_collocation",
        "hofno_order2_dealiased",
    }
    assert all(
        report.accepted
        for architecture in architectures.values()
        for report in architecture.capability_reports(scenario)
    )

    architecture = architectures["hofno_order2_dealiased"]
    model = architecture.build(scenario, seed=5)
    _, result = run_operator_benchmark(
        model,
        scenario,
        steps=3,
        learning_rate=1e-3,
        repeats=1,
        architecture=architecture.name,
        family=architecture.family,
        architecture_configuration=architecture.configuration(scenario),
        validation_interval=1,
        run_evaluations=False,
    )

    assert result.training_steps == 3
    assert result.parameter_count == parameter_count(model)
    assert all(math.isfinite(loss) for loss in result.losses)
    assert result.final_loss < result.initial_loss
    assert result.validation_loss is not None
    assert math.isfinite(result.validation_loss)
    assert dict(result.architecture_configuration) == {
        "n_modes": "(4, 4)",
        "width": "6",
        "depth": "1",
        "interaction_order": "2",
        "factor_bias": "False",
        "spectral_channel_mixing": "depthwise",
        "aliasing": "dealiased",
        "ffn_expansion": "2",
        "coordinate_embedding": "False",
    }
