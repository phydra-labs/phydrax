import jax.numpy as jnp

from tools.operator_benchmarks.models import compatible_architectures
from tools.operator_benchmarks.scenarios import (
    add_sensor_dropout_shift,
    periodic_acoustic_wave_scenario,
    periodic_advection_scenario,
    periodic_burgers_scenario,
)
from tools.operator_benchmarks.v2 import (
    scenario_checksum,
    standard_operator_benchmark_ladders,
)


def _architecture_map(scenario):
    return {
        architecture.name: architecture
        for architecture in compatible_architectures(scenario, quick=True)
    }


def test_constant_advection_is_an_exact_periodic_grid_translation():
    scenario = periodic_advection_scenario(
        train_resolution=16,
        test_resolution=16,
        num_cases=4,
        speed_configuration="constant",
        speed=1.0,
        dt=1.0 / 16.0,
        target_steps=1,
        maximum_frequency=5,
        seed=11,
    )
    initial = jnp.asarray(scenario.train_batch.input("state").values)

    assert jnp.allclose(scenario.train_target, jnp.roll(initial, 1, axis=-1), atol=2e-6)
    assert scenario.reference_evidence is not None
    assert scenario.reference_evidence.passed
    assert scenario.reference_evidence.verification == "analytic"
    assert float(dict(scenario.metadata)["maximum_relative_mass_drift"]) < 2e-6
    assert jnp.all(jnp.isfinite(scenario.train_target))


def test_variable_advection_is_reproducible_conservative_and_nonuniform():
    settings = dict(
        train_resolution=24,
        test_resolution=32,
        num_cases=4,
        speed_configuration="variable",
        speed=0.8,
        speed_variation=0.4,
        dt=0.02,
        target_steps=3,
        rollout_steps=3,
        maximum_frequency=5,
        seed=12,
    )
    scenario = periodic_advection_scenario(**settings)
    repeated = periodic_advection_scenario(**settings)
    metadata = dict(scenario.metadata)

    assert scenario_checksum(scenario) == scenario_checksum(repeated)
    assert float(metadata["minimum_speed"]) < 0.8 < float(metadata["maximum_speed"])
    assert float(metadata["maximum_relative_mass_drift"]) < 2e-6
    assert scenario.reference_evidence is not None and scenario.reference_evidence.passed
    assert {evaluation.shift for evaluation in scenario.evaluations} >= {
        "resolution",
        "rollout",
    }
    assert all(jnp.all(jnp.isfinite(evaluation.target)) for evaluation in scenario.evaluations)


def test_acoustic_wave_preserves_characteristic_phase_and_energy():
    scenario = periodic_acoustic_wave_scenario(
        train_resolution=20,
        test_resolution=20,
        num_cases=4,
        sound_speed=2.0,
        density=1.5,
        dt=1.0 / 40.0,
        target_steps=1,
        rollout_steps=2,
        maximum_wavenumber=6,
        seed=13,
    )
    initial = jnp.asarray(scenario.train_batch.input("state").values)
    target = jnp.asarray(scenario.train_target)
    impedance = 3.0
    right_initial = initial[..., 0] + impedance * initial[..., 1]
    left_initial = initial[..., 0] - impedance * initial[..., 1]
    right_target = target[..., 0] + impedance * target[..., 1]
    left_target = target[..., 0] - impedance * target[..., 1]

    assert initial.shape == (4, 20, 2)
    assert jnp.allclose(right_target, jnp.roll(right_initial, 1, axis=-1), atol=3e-6)
    assert jnp.allclose(left_target, jnp.roll(left_initial, -1, axis=-1), atol=3e-6)
    initial_energy = jnp.mean(
        initial[..., 0] ** 2 / 12.0 + 0.75 * initial[..., 1] ** 2,
        axis=-1,
    )
    target_energy = jnp.mean(
        target[..., 0] ** 2 / 12.0 + 0.75 * target[..., 1] ** 2,
        axis=-1,
    )
    assert jnp.allclose(target_energy, initial_energy, rtol=2e-6)
    assert float(dict(scenario.metadata)["maximum_relative_energy_drift"]) < 3e-6
    assert scenario.reference_evidence is not None and scenario.reference_evidence.passed
    assert all(jnp.all(jnp.isfinite(evaluation.target)) for evaluation in scenario.evaluations)


def test_multichannel_acoustic_flower_and_comparators_run_all_resolutions():
    scenario = periodic_acoustic_wave_scenario(
        train_resolution=8,
        test_resolution=12,
        num_cases=2,
        rollout_steps=2,
        maximum_wavenumber=3,
        seed=19,
    )
    architectures = _architecture_map(scenario)

    assert {"fno", "ifno", "cno", "uno", "flower_resolution_consistent"} <= (
        architectures.keys()
    )
    for name in ("fno", "ifno", "cno", "uno", "flower_resolution_consistent"):
        model = architectures[name].build(scenario, seed=5)
        assert jnp.asarray(model(scenario.train_batch)).shape == scenario.train_target.shape
        assert all(
            jnp.asarray(model(evaluation.batch)).shape == evaluation.target.shape
            for evaluation in scenario.evaluations
        )

    configuration = dict(
        architectures["flower_resolution_consistent"].configuration(scenario)
    )
    assert configuration["in_channels"] == "2"
    assert configuration["out_channels"] == "2"


def test_viscous_burgers_shock_rollout_is_finite_and_conservative():
    scenario = periodic_burgers_scenario(
        train_resolution=32,
        test_resolution=48,
        num_cases=4,
        viscosity=0.01,
        dt=0.001,
        target_steps=4,
        rollout_steps=4,
        initial_condition="shock",
        maximum_frequency=6,
        seed=14,
    )
    initial = jnp.asarray(scenario.train_batch.input("state").values)
    target = jnp.asarray(scenario.train_target)
    gradient = jnp.abs(jnp.roll(initial, -1, axis=-1) - initial)

    assert float(jnp.max(gradient)) > 2.0 * float(jnp.mean(gradient))
    assert jnp.allclose(jnp.mean(target, axis=-1), jnp.mean(initial, axis=-1), atol=2e-6)
    assert "shock_formation" in scenario.regimes
    assert "viscous_rollout" in scenario.regimes
    assert scenario.reference_evidence is not None and scenario.reference_evidence.passed
    assert all(jnp.all(jnp.isfinite(evaluation.target)) for evaluation in scenario.evaluations)


def test_flower_factories_and_comparators_obey_tensor_grid_contracts():
    scenario = periodic_advection_scenario(
        train_resolution=8,
        test_resolution=12,
        num_cases=2,
        rollout_steps=2,
        seed=15,
    )
    architectures = _architecture_map(scenario)

    assert {
        "fno",
        "ifno",
        "cno",
        "uno",
        "flower_one_level",
        "flower_multilevel",
        "flower_resolution_consistent",
    } <= architectures.keys()
    assert "wavelet" not in architectures

    for name in (
        "flower_one_level",
        "flower_multilevel",
        "flower_resolution_consistent",
    ):
        architecture = architectures[name]
        model = architecture.build(scenario, seed=3)
        assert jnp.asarray(model(scenario.train_batch)).shape == scenario.train_target.shape
        assert all(
            jnp.asarray(model(evaluation.batch)).shape == evaluation.target.shape
            for evaluation in scenario.evaluations
        )
        configuration = dict(architecture.configuration(scenario))
        assert configuration["query_mode"] == "coincident"
        assert configuration["probabilistic_routing"] == "False"

    assert dict(architectures["flower_one_level"].configuration(scenario))[
        "transition_mode"
    ] == "learned"
    assert dict(architectures["flower_multilevel"].configuration(scenario))[
        "levels"
    ] == "2"
    assert dict(
        architectures["flower_resolution_consistent"].configuration(scenario)
    )["transition_mode"] == "resolution_consistent"


def test_masked_ladder_keeps_only_resolution_consistent_flower():
    unmasked = periodic_advection_scenario(
        train_resolution=8,
        test_resolution=8,
        num_cases=2,
        seed=16,
    )
    assert "wavelet" in _architecture_map(unmasked)

    masked = add_sensor_dropout_shift(unmasked, drop_fraction=0.25, seed=17)
    flower_names = {
        name for name in _architecture_map(masked) if name.startswith("flower_")
    }
    assert flower_names == {"flower_resolution_consistent"}


def test_multilevel_flower_rejects_below_minimum_and_nondivisible_axes():
    for resolution in (8, 18):
        scenario = periodic_burgers_scenario(
            train_resolution=resolution,
            test_resolution=resolution,
            num_cases=2,
            seed=18,
        )
        flower_names = {
            architecture.name
            for architecture in compatible_architectures(scenario, quick=False)
            if architecture.name.startswith("flower_")
        }
        assert flower_names == {"flower_one_level"}


def test_transport_wave_ladders_join_non_smoke_registry_only():
    smoke_names = {
        ladder.name for ladder in standard_operator_benchmark_ladders(quick=True)
    }
    shortlist = standard_operator_benchmark_ladders(profile="shortlist")
    shortlist_names = {ladder.name for ladder in shortlist}
    new_names = {
        "constant_speed_advection",
        "variable_speed_advection",
        "periodic_acoustic_waves",
    }

    assert new_names.isdisjoint(smoke_names)
    assert new_names <= shortlist_names
    for ladder in shortlist:
        if ladder.name in new_names:
            for scenario in ladder.levels:
                assert scenario.provenance is not None
                assert scenario.reference_evidence is not None
                assert scenario.reference_evidence.passed
                assert jnp.all(jnp.isfinite(scenario.train_target))
