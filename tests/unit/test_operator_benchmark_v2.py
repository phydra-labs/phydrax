import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import opt_einsum as oe
import polars as pl
import pytest

import tools.operator_benchmarks.v2 as benchmark_v2
from tools.operator_benchmarks import (
    audit_external_candidate,
    audit_geometry_scenario,
    audit_operator_scenario,
    audit_scenario_difficulty,
    BenchmarkComparisonRecord,
    compatible_architectures,
    ExternalCandidateAudit,
    ExternalOperatorCandidate,
    FamilyParityEvidence,
    flatten_operator_benchmark_ladders,
    HyperparameterTrial,
    KernelParityCheck,
    load_family_parity_evidence,
    native_kernel_parity_checks,
    NearIdentityDiagnostic,
    OperatorBenchmarkAggregate,
    OperatorBenchmarkProtocol,
    parameter_count,
    PromotionCriteria,
    run_operator_benchmark,
    run_operator_benchmark_v2,
    save_benchmark_v2_artifacts,
    scenario_checksum,
    ScenarioDifficultyAudit,
    ScenarioIntegrityAudit,
    select_benchmark_superior_external,
    split_operator_scenario,
    standard_operator_benchmark_ladders,
    train_operator,
)
from tools.operator_benchmarks.scenarios import (
    add_sensor_corruption_shift,
    add_sensor_dropout_shift,
    add_training_sensor_dropout,
    causal_relaxation_scenario,
    cochain_mixed_darcy_scenario,
    conservative_ring_transport_scenario,
    darcy_scenario,
    deformed_elliptic_scenario,
    graph_diffusion_scenario,
    green_function_scenario,
    irregular_poisson_scenario,
    multi_input_diffusion_scenario,
    navier_stokes_scenario,
    periodic_burgers_scenario,
    spherical_diffusion_scenario,
    square_diffusion_symmetry_scenario,
)
from tools.operator_benchmarks.v2 import (
    _pareto_fronts,
    _portfolio_promotions,
    _promotion_reports,
    _target_parameters,
)


@pytest.fixture(scope="module")
def quick_ladders():
    return standard_operator_benchmark_ladders(quick=True)


@pytest.fixture(scope="module")
def conservative_geometry_scenario():
    return conservative_ring_transport_scenario(
        source_points=16,
        query_points=19,
        support_resolution=8,
        num_cases=6,
        seed=7,
    )


def _ladder(ladders, name):
    return next(ladder for ladder in ladders if ladder.name == name)


def _assert_no_duplicate_cases(values):
    flattened = jnp.asarray(values).reshape(values.shape[0], -1)
    exact_matches = jnp.all(
        flattened[:, None, :] == flattened[None, :, :],
        axis=-1,
    )
    assert int(jnp.count_nonzero(exact_matches)) == int(flattened.shape[0])


def _assert_population_rank(values, minimum_rank=6):
    flattened = jnp.asarray(values).reshape(values.shape[0], -1)
    assert int(jnp.linalg.matrix_rank(flattened)) >= int(minimum_rank)


def test_seeded_scenario_populations_are_deterministic_and_diverse():
    populations = (
        (
            lambda seed: navier_stokes_scenario(num_cases=6, seed=seed),
            "vorticity",
        ),
        (
            lambda seed: green_function_scenario(num_cases=6, seed=seed),
            "forcing",
        ),
        (
            lambda seed: spherical_diffusion_scenario(num_cases=6, seed=seed),
            "field",
        ),
    )
    for construct, source_name in populations:
        first = construct(41)
        repeated = construct(41)
        changed = construct(42)
        first_values = jnp.asarray(first.train_batch.input(source_name).values)
        repeated_values = jnp.asarray(repeated.train_batch.input(source_name).values)
        changed_values = jnp.asarray(changed.train_batch.input(source_name).values)
        flattened = first_values.reshape(first_values.shape[0], -1)

        assert jnp.array_equal(first_values, repeated_values)
        assert jnp.array_equal(first.train_target, repeated.train_target)
        assert not jnp.array_equal(first_values, changed_values)
        assert int(jnp.linalg.matrix_rank(flattened)) >= 4

    navier_values = populations[0][0](41).train_batch.input("vorticity").values
    green_values = populations[1][0](41).train_batch.input("forcing").values
    navier_spectrum = jnp.abs(jnp.fft.fftn(navier_values[0]))
    green_spectrum = jnp.abs(jnp.fft.rfft(green_values[0]))
    assert int(jnp.count_nonzero(navier_spectrum > 1e-8 * navier_spectrum.max())) > 8
    assert int(jnp.count_nonzero(green_spectrum > 1e-8 * green_spectrum.max())) > 2


def test_remaining_seeded_physical_populations_have_rank_without_case_aliases():
    populations = (
        (
            lambda seed: periodic_burgers_scenario(
                train_resolution=12,
                test_resolution=24,
                num_cases=16,
                target_steps=2,
                maximum_frequency=5,
                seed=seed,
            ),
            "state",
            "discrete_residual",
        ),
        (
            lambda seed: darcy_scenario(
                resolution=9,
                num_cases=16,
                maximum_frequency=3,
                seed=seed,
            ),
            "coefficient",
            "discrete_residual",
        ),
        (
            lambda seed: multi_input_diffusion_scenario(
                resolution=12,
                num_cases=16,
                parameter_shift_factor=1.7,
                maximum_frequency=5,
                seed=seed,
            ),
            "initial",
            "analytic",
        ),
        (
            lambda seed: irregular_poisson_scenario(
                points=12,
                num_cases=16,
                geometry_shift=True,
                maximum_frequency=3,
                seed=seed,
            ),
            "forcing",
            "analytic",
        ),
        (
            lambda seed: causal_relaxation_scenario(
                source_points=16,
                query_points=16,
                test_query_points=24,
                num_cases=16,
                final_time=1.0,
                maximum_frequency=8.0,
                modes=8,
                seed=seed,
            ),
            "forcing",
            "analytic",
        ),
        (
            lambda seed: graph_diffusion_scenario(
                nodes=12,
                test_nodes=24,
                num_cases=16,
                geometry_shift=True,
                maximum_frequency=5,
                seed=seed,
            ),
            "state",
            "discrete_residual",
        ),
    )

    for construct, source_name, verification in populations:
        first = construct(23)
        repeated = construct(23)
        changed = construct(24)
        first_source = jnp.asarray(first.train_batch.input(source_name).values)
        repeated_source = jnp.asarray(repeated.train_batch.input(source_name).values)
        changed_source = jnp.asarray(changed.train_batch.input(source_name).values)

        assert first.seed == 23
        assert dict(first.metadata)["population_seed"] == "23"
        assert jnp.array_equal(first_source, repeated_source)
        assert jnp.array_equal(first.train_target, repeated.train_target)
        assert not jnp.array_equal(first_source, changed_source)
        _assert_population_rank(first_source)
        _assert_population_rank(first.train_target)
        _assert_no_duplicate_cases(first_source)
        _assert_no_duplicate_cases(first.train_target)

        for evaluation in first.evaluations:
            evaluation_source = evaluation.batch.input(source_name).values
            _assert_population_rank(evaluation_source)
            _assert_population_rank(evaluation.target)
            _assert_no_duplicate_cases(evaluation_source)
            _assert_no_duplicate_cases(evaluation.target)
            assert evaluation.case_ids == first.case_ids

        assert first.reference_evidence is not None
        assert first.reference_evidence.verification == verification
        assert first.reference_evidence.passed


def test_deformed_elliptic_geometry_and_shift_audits_pass():
    scenario = split_operator_scenario(
        deformed_elliptic_scenario(
            points=12,
            query_points=10,
            num_cases=6,
            deformation_amplitude=0.025,
            seed=31,
        ),
        seed=72,
    )
    geometry_audit = audit_geometry_scenario(scenario)
    generic_audit = audit_operator_scenario(scenario, quick=True)
    source = scenario.train_batch.input("forcing")
    coordinates = jnp.asarray(source.coordinates)

    assert dict(scenario.metadata)["population_seed"] == "31"
    assert geometry_audit.passed, geometry_audit.reasons
    assert generic_audit.passed, generic_audit.reasons
    assert geometry_audit.minimum_jacobian > 0.0
    assert coordinates.shape[:2] == (scenario.train_batch.case_shape[0], 12)
    assert not jnp.array_equal(coordinates[0], coordinates[1])
    assert jnp.all(source.quadrature_weights > 0.0)
    assert jnp.allclose(jnp.sum(source.quadrature_weights, axis=-1), 1.0)
    assert {evaluation.name for evaluation in scenario.evaluations} >= {
        "nominal",
        "resolution_transfer",
        "independent_query",
        "geometry_extrapolation",
        "sensor_dropout",
        "boundary_condition_shift",
    }
    architectures = {
        architecture.name: architecture
        for architecture in compatible_architectures(scenario, quick=True)
    }
    gino = architectures["gino"]
    configuration = dict(gino.configuration(scenario))
    prediction = gino.build(scenario, seed=9)(scenario.train_batch)
    geometry_flower = architectures["geometry_informed_flower"]
    geometry_flower_configuration = dict(geometry_flower.configuration(scenario))
    geometry_flower_model = geometry_flower.build(scenario, seed=13)
    geometry_flower_prediction = geometry_flower_model(scenario.train_batch)
    resolution_evaluation = next(
        evaluation
        for evaluation in scenario.evaluations
        if evaluation.name == "resolution_transfer"
    )
    gino_resolution_prediction = gino.build(scenario, seed=9)(resolution_evaluation.batch)
    geometry_flower_resolution_prediction = geometry_flower_model(
        resolution_evaluation.batch
    )
    rigno = architectures["rigno"]
    rigno_configuration = dict(rigno.configuration(scenario))
    rigno_prediction = rigno.build(scenario, seed=10)(scenario.train_batch)
    gaot = architectures["gaot"]
    gaot_configuration = dict(gaot.configuration(scenario))
    gaot_prediction = gaot.build(scenario, seed=11)(scenario.train_batch)
    gnot = architectures["gnot"]
    gnot_prediction = gnot.build(scenario, seed=12)(scenario.train_batch)

    assert gino.family == "geometry_informed"
    assert configuration["latent_shape"] == "(6, 6)"
    assert configuration["bounds_policy"] == "'global'"
    assert prediction.shape == scenario.train_target.shape
    assert jnp.all(jnp.isfinite(prediction))
    assert geometry_flower.family == "geometry_informed_warp"
    assert geometry_flower_configuration["latent_shape"] == "(8, 8)"
    assert geometry_flower_configuration["transition_mode"] == ("'resolution_consistent'")
    assert geometry_flower_prediction.shape == scenario.train_target.shape
    assert jnp.all(jnp.isfinite(geometry_flower_prediction))
    assert gino_resolution_prediction.shape == resolution_evaluation.target.shape
    assert geometry_flower_resolution_prediction.shape == (
        resolution_evaluation.target.shape
    )
    assert jnp.all(jnp.isfinite(gino_resolution_prediction))
    assert jnp.all(jnp.isfinite(geometry_flower_resolution_prediction))
    assert rigno.family == "regional_graph"
    assert rigno_configuration["regional_count"] == "8"
    assert rigno_configuration["regional_mode"] == "'farthest_point'"
    assert rigno_prediction.shape == scenario.train_target.shape
    assert jnp.all(jnp.isfinite(rigno_prediction))
    assert gaot.family == "geometry_transformer"
    assert gaot_configuration["latent_shape"] == "(4, 4)"
    assert gaot_configuration["bounds_policy"] == "'case_bbox'"
    assert gaot_configuration["transfer_scales"] == "(1.0, 2.0)"
    assert gaot_prediction.shape == scenario.train_target.shape
    assert jnp.all(jnp.isfinite(gaot_prediction))
    assert gnot.family == "heterogeneous_geometry_transformer"
    assert gnot.promotion_scope == "general"
    assert gnot_prediction.shape == scenario.train_target.shape
    assert jnp.all(jnp.isfinite(gnot_prediction))
    assert {"transolver", "upt"}.isdisjoint(architectures)


def test_conservative_ring_transport_has_physical_support_and_exact_mass(
    conservative_geometry_scenario,
):
    scenario = conservative_geometry_scenario
    assert scenario.domain_support_key == "domain_sdf"
    assert scenario.domain_support_kind == "sdf"
    assert scenario.domain_support_threshold == 0.0
    assert scenario.conservation_source_key == "density"
    assert scenario.reference_evidence is not None
    assert scenario.reference_evidence.passed
    assert scenario.train_batch.input("domain_sdf").values.shape == (6, 64)
    assert scenario.train_batch.input("domain_sdf").coordinates.shape == (64, 2)
    assert {evaluation.name for evaluation in scenario.evaluations} == {
        "nominal",
        "resolution_transfer",
        "independent_query",
        "geometry_extrapolation",
        "support_extrapolation",
        "speed_extrapolation",
    }

    batches_and_targets = ((scenario.train_batch, scenario.train_target),) + tuple(
        (evaluation.batch, evaluation.target) for evaluation in scenario.evaluations
    )
    for batch, target in batches_and_targets:
        source = batch.input("density")
        source_mass = jnp.sum(
            source.values * source.weights(case_shape=batch.case_shape),
            axis=-1,
        )
        target_mass = jnp.sum(
            target * batch.require_single_query().weights(case_shape=batch.case_shape),
            axis=-1,
        )
        assert jnp.allclose(source_mass, target_mass, rtol=1e-12, atol=1e-12)

    assert scenario_checksum(scenario) != scenario_checksum(
        replace(scenario, domain_support_threshold=0.05)
    )
    assert scenario_checksum(scenario) != scenario_checksum(
        replace(scenario, conservation_source_key="speed")
    )


def test_geometry_informed_flower_ablation_factories_are_controlled_and_finite(
    conservative_geometry_scenario,
):
    scenario = conservative_geometry_scenario
    architectures = {
        architecture.name: architecture
        for architecture in compatible_architectures(scenario, quick=True)
    }
    names = {
        name for name in architectures if name.startswith("geometry_informed_flower")
    }
    assert names == {
        "geometry_informed_flower",
        "geometry_informed_flower_learned",
        "geometry_informed_flower_support",
        "geometry_informed_flower_support_conservative",
    }

    canonical = dict(architectures["geometry_informed_flower"].configuration(scenario))
    learned = dict(
        architectures["geometry_informed_flower_learned"].configuration(scenario)
    )
    supported = dict(
        architectures["geometry_informed_flower_support"].configuration(scenario)
    )
    conservative = dict(
        architectures["geometry_informed_flower_support_conservative"].configuration(
            scenario
        )
    )
    assert canonical["transition_mode"] == "'resolution_consistent'"
    assert "latent_support_key" not in canonical
    assert learned["transition_mode"] == "'learned'"
    assert "latent_support_key" not in learned
    assert supported["latent_support_key"] == "'domain_sdf'"
    assert supported["source_mask_mode"] == "'renormalize'"
    assert "conserve_mass" not in supported
    assert conservative["latent_support_key"] == "'domain_sdf'"
    assert conservative["conserve_mass"] == "True"

    support_model = architectures["geometry_informed_flower_support"].build(
        scenario,
        seed=13,
    )
    support_prediction, support_diagnostics = support_model.evaluate_with_diagnostics(
        scenario.train_batch
    )
    assert support_prediction.shape == scenario.train_target.shape
    assert jnp.all(jnp.isfinite(support_prediction))
    assert support_diagnostics.latent_mask is not None
    assert jnp.all(jnp.any(support_diagnostics.latent_mask, axis=-1))

    conservative_model = architectures[
        "geometry_informed_flower_support_conservative"
    ].build(scenario, seed=17)
    conservative_prediction = conservative_model(scenario.train_batch)
    source = scenario.train_batch.input("density")
    source_mass = jnp.sum(
        source.values * source.weights(case_shape=scenario.train_batch.case_shape),
        axis=-1,
    )
    prediction_mass = jnp.sum(
        conservative_prediction
        * scenario.train_batch.require_single_query().weights(
            case_shape=scenario.train_batch.case_shape
        ),
        axis=-1,
    )
    assert jnp.all(jnp.isfinite(conservative_prediction))
    assert jnp.allclose(source_mass, prediction_mass, rtol=1e-12, atol=1e-12)


def test_sourcewise_normalization_preserves_support_and_physical_mass(
    conservative_geometry_scenario,
):
    scenario = conservative_geometry_scenario
    statistics, target_location, target_scale = benchmark_v2._normalization_statistics(
        scenario, "sourcewise"
    )

    class ConservativeProjector:
        def __call__(self, batch):
            source = batch.input("density")
            source_mass = jnp.sum(
                source.values * source.weights(case_shape=batch.case_shape),
                axis=-1,
            )
            query_measure = batch.require_single_query().weights(
                case_shape=batch.case_shape
            )
            query_total = jnp.sum(query_measure, axis=-1)
            return jnp.broadcast_to(
                (source_mass / query_total)[:, None],
                batch.case_shape + batch.require_single_query().sample_shape,
            )

    normalized = benchmark_v2._NormalizedOperator(
        ConservativeProjector(),
        statistics,
        target_location,
        target_scale,
        "sourcewise",
        domain_support_key=scenario.domain_support_key,
        conservation_source_key=scenario.conservation_source_key,
    )
    prediction = normalized(scenario.train_batch)
    source = scenario.train_batch.input("density")
    source_mass = jnp.sum(
        source.values * source.weights(case_shape=scenario.train_batch.case_shape),
        axis=-1,
    )
    prediction_mass = jnp.sum(
        prediction
        * scenario.train_batch.require_single_query().weights(
            case_shape=scenario.train_batch.case_shape
        ),
        axis=-1,
    )
    assert jnp.allclose(source_mass, prediction_mass, rtol=1e-12, atol=1e-12)

    class SupportProjection:
        def __call__(self, batch):
            count = batch.require_single_query().sample_shape[0]
            return batch.input("domain_sdf").values[:, :count]

    support_passthrough = benchmark_v2._NormalizedOperator(
        SupportProjection(),
        statistics,
        0.0,
        1.0,
        "sourcewise",
        domain_support_key=scenario.domain_support_key,
        conservation_source_key=scenario.conservation_source_key,
    )
    expected = scenario.train_batch.input("domain_sdf").values[
        :, : scenario.train_batch.require_single_query().sample_shape[0]
    ]
    assert jnp.array_equal(support_passthrough(scenario.train_batch), expected)


def test_roadmap_factories_are_geometry_gated_and_finite():
    scenario = darcy_scenario(resolution=4, num_cases=2, seed=91)
    architectures = {
        architecture.name: architecture
        for architecture in compatible_architectures(scenario, quick=True)
    }
    expected = {
        "ifno": ("implicit_spectral", "spectral", "specialized"),
        "axial_factorized_fno": ("axial_spectral", "spectral", "specialized"),
        "poseidon": ("multiscale_transformer", "sourcewise", "specialized"),
        "transolver": ("physics_attention", "sourcewise", "general"),
        "gnot": (
            "heterogeneous_geometry_transformer",
            "sourcewise",
            "general",
        ),
        "upt": ("latent_physics_transformer", "sourcewise", "general"),
    }

    assert set(architectures) >= set(expected)
    assert len({family for family, _, _ in expected.values()}) == len(expected)
    for name, contract in expected.items():
        architecture = architectures[name]
        prediction = architecture.build(scenario, seed=17)(scenario.train_batch)
        assert (
            architecture.family,
            architecture.normalization,
            architecture.promotion_scope,
        ) == contract
        assert prediction.shape == scenario.train_target.shape
        assert jnp.all(jnp.isfinite(prediction))

    odd_grid = darcy_scenario(resolution=5, num_cases=2, seed=92)
    odd_names = {
        architecture.name
        for architecture in compatible_architectures(odd_grid, quick=True)
    }
    assert "poseidon" not in odd_names
    assert {"ifno", "axial_factorized_fno"} <= odd_names

    metadata_only_input = multi_input_diffusion_scenario(
        resolution=6,
        num_cases=3,
        seed=93,
    )
    metadata_names = {
        architecture.name
        for architecture in compatible_architectures(
            metadata_only_input,
            quick=True,
        )
    }
    assert set(expected).isdisjoint(metadata_names)


def test_burgers_reuses_multimode_realizations_and_preserves_sharp_residuals():
    scenarios = (
        periodic_burgers_scenario(
            train_resolution=32,
            test_resolution=64,
            num_cases=16,
            target_steps=3,
            maximum_frequency=6,
            initial_condition="smooth",
            seed=17,
        ),
        periodic_burgers_scenario(
            train_resolution=32,
            test_resolution=64,
            num_cases=16,
            target_steps=3,
            maximum_frequency=6,
            initial_condition="shock",
            seed=17,
        ),
    )
    for scenario in scenarios:
        train_values = jnp.asarray(scenario.train_batch.input("state").values)
        resolution_evaluation = next(
            evaluation
            for evaluation in scenario.evaluations
            if evaluation.shift == "resolution"
        )
        test_values = jnp.asarray(resolution_evaluation.batch.input("state").values)

        assert jnp.array_equal(train_values, test_values[:, ::2])
        _assert_population_rank(train_values)
        _assert_population_rank(scenario.train_target)
        assert dict(scenario.metadata)["resolved_frequency"] == "6"
        assert scenario.reference_evidence is not None
        assert scenario.reference_evidence.verification == "discrete_residual"
        assert scenario.reference_evidence.passed

    sharp_values = jnp.asarray(scenarios[1].train_batch.input("state").values)
    sharp_spectrum = jnp.abs(jnp.fft.rfft(sharp_values[0]))
    spectral_support = jnp.count_nonzero(sharp_spectrum > 1e-8 * sharp_spectrum.max())
    periodic_jump = jnp.max(jnp.abs(sharp_values - jnp.roll(sharp_values, 1, axis=-1)))
    assert int(spectral_support) > 12
    assert float(periodic_jump) > 1.0


def test_darcy_population_is_positive_bounded_heterogeneous_and_direct_solved():
    contrast = 0.35
    scenario = darcy_scenario(
        resolution=10,
        num_cases=16,
        contrast=contrast,
        maximum_frequency=3,
        seed=19,
    )
    permeability = jnp.asarray(scenario.train_batch.input("coefficient").values)

    assert float(permeability.min()) >= 1.0 - contrast
    assert float(permeability.max()) <= 1.0 + contrast
    assert jnp.all(jnp.std(permeability, axis=(-2, -1)) > 0.0)
    _assert_population_rank(permeability)
    _assert_population_rank(scenario.train_target)
    assert scenario.reference_evidence is not None
    assert scenario.reference_evidence.method == "direct finite-volume elliptic solve"
    assert scenario.reference_evidence.verification == "discrete_residual"
    assert scenario.reference_evidence.passed


def test_multi_input_diffusion_components_and_shifted_targets_are_independent():
    dt = 0.05
    shift_factor = 1.7
    scenario = multi_input_diffusion_scenario(
        resolution=16,
        num_cases=16,
        dt=dt,
        parameter_shift_factor=shift_factor,
        maximum_frequency=6,
        seed=29,
    )
    initial = jnp.asarray(scenario.train_batch.input("initial").values)
    forcing = jnp.asarray(scenario.train_batch.input("forcing").values)
    diffusivity = jnp.asarray(scenario.train_batch.input("diffusivity").values)
    component_statistics = jnp.stack(
        (
            jnp.linalg.norm(initial, axis=-1),
            jnp.linalg.norm(forcing, axis=-1),
            diffusivity[:, 0],
            jnp.ones((initial.shape[0],)),
        ),
        axis=-1,
    )

    _assert_population_rank(initial)
    _assert_population_rank(forcing)
    _assert_no_duplicate_cases(initial)
    _assert_no_duplicate_cases(forcing)
    _assert_no_duplicate_cases(diffusivity)
    assert int(jnp.linalg.matrix_rank(component_statistics)) == 4

    squared_wave_number = (
        2.0 * jnp.pi * jnp.fft.rfftfreq(initial.shape[-1], d=1.0 / initial.shape[-1])
    ) ** 2
    for evaluation in scenario.evaluations:
        current_initial = jnp.asarray(evaluation.batch.input("initial").values)
        current_forcing = jnp.asarray(evaluation.batch.input("forcing").values)
        current_diffusivity = jnp.asarray(evaluation.batch.input("diffusivity").values)
        decay_rate = current_diffusivity * squared_wave_number[None, :]
        attenuation = jnp.exp(-decay_rate * dt)
        safe_rate = jnp.where(decay_rate > 0.0, decay_rate, 1.0)
        response = -jnp.expm1(-safe_rate * dt) / safe_rate
        response = jnp.where(decay_rate > 0.0, response, dt)
        expected = jnp.fft.irfft(
            jnp.fft.rfft(current_initial, axis=-1) * attenuation
            + jnp.fft.rfft(current_forcing, axis=-1) * response,
            n=current_initial.shape[-1],
            axis=-1,
        )
        assert jnp.allclose(evaluation.target, expected, rtol=1e-11, atol=1e-11)

    shifted = next(
        evaluation
        for evaluation in scenario.evaluations
        if evaluation.shift == "parameter"
    )
    assert jnp.array_equal(shifted.batch.input("initial").values, initial)
    assert jnp.array_equal(shifted.batch.input("forcing").values, forcing)
    assert jnp.allclose(
        shifted.batch.input("diffusivity").values,
        shift_factor * diffusivity,
    )
    assert scenario.reference_evidence is not None
    assert scenario.reference_evidence.verification == "analytic"
    assert scenario.reference_evidence.passed


def test_irregular_and_graph_shifts_reuse_the_same_physical_realizations():
    maximum_frequency = 2
    irregular = irregular_poisson_scenario(
        points=40,
        num_cases=8,
        geometry_shift=True,
        maximum_frequency=maximum_frequency,
        seed=31,
    )
    original_samples = irregular.train_batch.input("forcing")
    shifted_evaluation = next(
        evaluation
        for evaluation in irregular.evaluations
        if evaluation.shift == "geometry"
    )
    shifted_samples = shifted_evaluation.batch.input("forcing")
    mode_x, mode_y = jnp.meshgrid(
        jnp.arange(maximum_frequency + 1),
        jnp.arange(maximum_frequency + 1),
        indexing="ij",
    )
    mode_x = mode_x.reshape(-1)[1:]
    mode_y = mode_y.reshape(-1)[1:]

    def planar_basis(coordinates):
        phase = (
            2.0
            * jnp.pi
            * (
                mode_x[:, None] * coordinates[None, :, 0]
                + mode_y[:, None] * coordinates[None, :, 1]
            )
        )
        basis = jnp.stack((jnp.sin(phase), jnp.cos(phase)), axis=-1)
        return jnp.transpose(basis, (1, 0, 2)).reshape(coordinates.shape[0], -1)

    original_basis = planar_basis(original_samples.coordinates)
    physical_coefficients = jnp.linalg.lstsq(
        original_basis,
        jnp.asarray(original_samples.values).T,
    )[0]
    predicted_shifted_values = (
        planar_basis(shifted_samples.coordinates) @ physical_coefficients
    ).T
    assert jnp.allclose(
        shifted_samples.values,
        predicted_shifted_values,
        rtol=1e-10,
        atol=1e-10,
    )
    shifted_displacement = (
        shifted_evaluation.batch.require_single_query().coordinates[:, None, :]
        - shifted_samples.coordinates[None, :, :]
    )
    shifted_kernel = -jnp.log(
        jnp.sqrt(jnp.sum(shifted_displacement**2, axis=-1) + 1e-3)
    ) / (2.0 * jnp.pi)
    expected_shifted_target = oe.contract(
        "qs,cs,s->cq",
        shifted_kernel,
        shifted_samples.values,
        shifted_samples.quadrature_weights,
    )
    assert jnp.allclose(
        shifted_evaluation.target,
        expected_shifted_target,
        rtol=1e-12,
        atol=1e-12,
    )
    assert irregular.reference_evidence is not None
    assert irregular.reference_evidence.passed

    graph = graph_diffusion_scenario(
        nodes=12,
        test_nodes=24,
        num_cases=16,
        geometry_shift=True,
        maximum_frequency=5,
        seed=37,
    )
    graph_values = jnp.asarray(graph.train_batch.input("state").values)
    resolution_evaluation = next(
        evaluation for evaluation in graph.evaluations if evaluation.shift == "resolution"
    )
    geometry_evaluation = next(
        evaluation for evaluation in graph.evaluations if evaluation.shift == "geometry"
    )
    assert jnp.array_equal(
        resolution_evaluation.batch.input("state").values[:, ::2],
        graph_values,
    )
    assert jnp.array_equal(
        geometry_evaluation.batch.input("state").values,
        graph_values,
    )
    assert graph.reference_evidence is not None
    assert graph.reference_evidence.verification == "discrete_residual"
    assert graph.reference_evidence.passed


def test_multistep_targets_are_nonidentity_and_spherical_degrees_attenuate():
    navier = navier_stokes_scenario(
        viscosity=0.05,
        dt=0.03,
        target_steps=8,
        seed=7,
    )
    navier_source = jnp.asarray(navier.train_batch.input("vorticity").values)
    navier_change = jnp.linalg.norm(
        navier.train_target - navier_source
    ) / jnp.linalg.norm(navier_source)
    assert float(navier_change) > 0.1
    assert navier.reference_evidence is not None
    assert navier.reference_evidence.passed

    diffusivity = 0.1
    dt = 0.2
    target_steps = 5
    spherical = spherical_diffusion_scenario(
        theta_points=16,
        phi_points=32,
        num_cases=6,
        diffusivity=diffusivity,
        dt=dt,
        target_steps=target_steps,
        maximum_degree=3,
        seed=7,
    )
    spherical_source = jnp.asarray(spherical.train_batch.input("field").values)
    spherical_change = jnp.linalg.norm(
        spherical.train_target - spherical_source
    ) / jnp.linalg.norm(spherical_source)
    assert float(spherical_change) > 0.1

    theta = spherical.train_batch.input("field").axes[0].nodes
    cosine = jnp.cos(theta)
    legendre_modes = jnp.stack(
        (
            cosine,
            0.5 * (3.0 * cosine**2 - 1.0),
            0.5 * (5.0 * cosine**3 - 3.0 * cosine),
        ),
        axis=1,
    )
    source_coefficients = jnp.linalg.lstsq(
        legendre_modes,
        jnp.mean(spherical_source, axis=2).T,
    )[0]
    target_coefficients = jnp.linalg.lstsq(
        legendre_modes,
        jnp.mean(spherical.train_target, axis=2).T,
    )[0]
    degrees = jnp.arange(1, 4)
    expected_attenuation = jnp.exp(
        -diffusivity * dt * target_steps * degrees * (degrees + 1)
    )
    assert jnp.allclose(
        target_coefficients / source_coefficients,
        expected_attenuation[:, None],
        rtol=1e-10,
        atol=1e-10,
    )
    assert spherical.reference_evidence is not None
    assert spherical.reference_evidence.verification == "analytic"
    assert spherical.reference_evidence.passed


def test_decision_long_horizon_rollout_remains_finite():
    ladders = standard_operator_benchmark_ladders(profile="decision")
    scenario = _ladder(ladders, "long_horizon").levels[1]
    targets = (scenario.train_target,) + tuple(
        evaluation.target for evaluation in scenario.evaluations
    )

    assert all(bool(jnp.all(jnp.isfinite(target))) for target in targets)


def test_sensor_corruption_and_dropout_have_distinct_mask_semantics():
    scenario = green_function_scenario(num_cases=8, seed=11)
    corrupted = add_sensor_corruption_shift(
        scenario,
        corruption_fraction=0.4,
        seed=23,
    ).evaluations[-1]
    dropped = add_sensor_dropout_shift(
        scenario,
        drop_fraction=0.4,
        seed=23,
    ).evaluations[-1]
    repeated_corrupted = add_sensor_corruption_shift(
        scenario,
        corruption_fraction=0.4,
        seed=23,
    ).evaluations[-1]
    repeated_dropped = add_sensor_dropout_shift(
        scenario,
        drop_fraction=0.4,
        seed=23,
    ).evaluations[-1]
    original_samples = scenario.evaluations[0].batch.input("forcing")
    corrupted_samples = corrupted.batch.input("forcing")
    dropped_samples = dropped.batch.input("forcing")
    original_mask = original_samples.mask_array(
        case_shape=scenario.evaluations[0].batch.case_shape
    )
    corrupted_mask = corrupted_samples.mask_array(case_shape=corrupted.batch.case_shape)
    dropped_mask = dropped_samples.mask_array(case_shape=dropped.batch.case_shape)

    assert corrupted.shift == "sensor_corruption"
    assert dropped.shift == "sensor_dropout"
    assert jnp.array_equal(corrupted_mask, original_mask)
    assert jnp.any(dropped_mask != original_mask)
    assert jnp.array_equal(corrupted_samples.values, dropped_samples.values)
    assert jnp.array_equal(
        corrupted_samples.values,
        repeated_corrupted.batch.input("forcing").values,
    )
    assert jnp.array_equal(
        dropped_mask,
        repeated_dropped.batch.input("forcing").mask_array(
            case_shape=repeated_dropped.batch.case_shape
        ),
    )
    assert jnp.all(jnp.asarray(corrupted_samples.values)[~dropped_mask] == 0.0)
    assert jnp.array_equal(corrupted.target, scenario.evaluations[0].target)
    assert jnp.array_equal(dropped.target, scenario.evaluations[0].target)


def test_training_sensor_dropout_is_deterministic_and_training_only():
    scenario = split_operator_scenario(
        green_function_scenario(num_cases=10, seed=17),
        seed=19,
    )
    augmented = add_training_sensor_dropout(scenario, drop_fraction=0.4, seed=29)
    repeated = add_training_sensor_dropout(scenario, drop_fraction=0.4, seed=29)
    augmented_samples = augmented.train_batch.input("forcing")
    repeated_samples = repeated.train_batch.input("forcing")
    original_samples = scenario.train_batch.input("forcing")
    augmented_mask = augmented_samples.mask_array(
        case_shape=augmented.train_batch.case_shape
    )
    original_mask = original_samples.mask_array(
        case_shape=scenario.train_batch.case_shape
    )

    assert jnp.any(augmented_mask != original_mask)
    assert jnp.array_equal(
        augmented_mask,
        repeated_samples.mask_array(case_shape=repeated.train_batch.case_shape),
    )
    assert jnp.array_equal(augmented_samples.values, repeated_samples.values)
    assert jnp.array_equal(augmented.train_target, scenario.train_target)
    assert augmented.validation is scenario.validation
    assert augmented.evaluations is scenario.evaluations
    assert dict(augmented.metadata)["training_augmentation"] == "sensor_dropout"


def test_v2_ladders_have_audited_physical_splits(quick_ladders):
    required = {
        "smooth_periodic",
        "polynomial_nonlinearity",
        "shock_discontinuity",
        "elliptic_contrast",
        "irregular_geometry",
        "conservative_geometry",
        "independent_query",
        "multi_input",
        "long_horizon",
        "spherical_field",
        "geometry_extrapolation",
        "parameter_extrapolation",
        "cochain_mixed_darcy",
        "cochain_annulus_harmonic",
    }
    assert required <= {ladder.name for ladder in quick_ladders}
    assert "causal_transient" in {ladder.name for ladder in quick_ladders}

    audits = []
    for scenario in flatten_operator_benchmark_ladders(quick_ladders):
        assert scenario.difficulty in {"easy", "hard"}
        assert scenario.provenance is not None
        assert scenario.provenance.source_uri
        assert scenario.dimensional_parameters
        assert scenario.nondimensional_parameters
        assert scenario.reference_evidence is not None
        assert scenario.reference_evidence.passed
        split = split_operator_scenario(scenario, seed=1729)
        audit = audit_operator_scenario(split, quick=True)
        audits.append(audit)
        assert audit.passed, audit.reasons
        assert audit.physical_split_disjoint
        assert any(
            evaluation.shift == "in_distribution" for evaluation in split.evaluations
        )
    assert any(audit.near_identity.detected for audit in audits)
    assert any(not audit.near_identity.detected for audit in audits)


def test_cochain_benchmarks_preserve_typed_fields_and_matched_architectures(
    quick_ladders,
):
    mixed_ladder = _ladder(quick_ladders, "cochain_mixed_darcy")
    harmonic_ladder = _ladder(quick_ladders, "cochain_annulus_harmonic")
    mixed = split_operator_scenario(mixed_ladder.levels[0], seed=1729)
    harmonic = split_operator_scenario(harmonic_ladder.levels[0], seed=1729)

    assert tuple(mixed.train_target.fields) == ("pressure", "flux")
    assert tuple(harmonic.train_target.fields) == ("harmonic",)
    assert mixed.task is not None
    assert harmonic.task is not None
    assert mixed.task.fields[1].cochain.degree == 0
    assert mixed.task.fields[2].cochain.degree == 1
    assert harmonic.task.fields[0].cochain.cell_orientation == "signed"
    assert mixed.train_batch.input("forcing").topology.graph_fingerprint == (
        mixed.train_batch.query("edges").topology.graph_fingerprint
    )
    assert audit_operator_scenario(mixed, quick=True).passed
    assert audit_operator_scenario(harmonic, quick=True).passed

    mixed_architectures = compatible_architectures(mixed, quick=True)
    assert {architecture.name for architecture in mixed_architectures} == {
        "cochain_pointwise",
        "cochain_neural_operator",
    }
    assert {
        architecture.name
        for architecture in compatible_architectures(harmonic, quick=True)
    } == {
        "cochain_pointwise",
        "cochain_no_harmonic",
        "cochain_neural_operator",
    }
    for architecture in mixed_architectures:
        prediction = architecture.build(mixed, seed=7).predict(mixed.train_batch)
        assert tuple(prediction.fields) == ("pressure", "flux")
        assert all(
            jnp.all(jnp.isfinite(field.values)) for field in prediction.fields.values()
        )


def test_named_multi_field_benchmark_reports_each_physical_field():
    scenario = split_operator_scenario(
        cochain_mixed_darcy_scenario(
            train_points=4,
            test_points=5,
            num_cases=8,
            seed=51,
        ),
        seed=53,
    )
    architecture = next(
        candidate
        for candidate in compatible_architectures(scenario, quick=True)
        if candidate.name == "cochain_neural_operator"
    )
    _, result = run_operator_benchmark(
        architecture.build(scenario, seed=5),
        scenario,
        steps=1,
        repeats=1,
        architecture=architecture.name,
        family=architecture.family,
    )
    evaluation = result.evaluations[0]
    payload = result.to_dict()

    assert {field.name for field in evaluation.field_metrics} == {
        "pressure",
        "flux",
    }
    assert evaluation.relative_l2 == max(
        field.relative_l2 for field in evaluation.field_metrics
    )
    assert {field["name"] for field in payload["evaluations"][0]["field_metrics"]} == {
        "pressure",
        "flux",
    }


def test_scenario_audit_rejects_nonfinite_numerical_data(quick_ladders):
    scenario = split_operator_scenario(
        _ladder(quick_ladders, "independent_query").levels[0],
        seed=1729,
    )
    target = jnp.asarray(scenario.train_target)
    invalid_target = target.at[(0,) * target.ndim].set(jnp.nan)
    invalid = replace(scenario, train_target=invalid_target)

    audit = audit_operator_scenario(invalid, quick=True)

    assert not audit.passed
    assert "scenario contains non-finite numerical data" in audit.reasons


def test_v2_registry_includes_specialized_families_and_fixed_pod_basis(quick_ladders):
    causal = split_operator_scenario(
        _ladder(quick_ladders, "causal_transient").levels[0], seed=1729
    )
    causal_architectures = {
        architecture.name: architecture
        for architecture in compatible_architectures(causal, quick=True)
    }
    assert {
        "constant",
        "nearest_neighbor",
        "linear_interpolation",
        "deeponet",
        "laplace",
    } <= set(causal_architectures)
    laplace = causal_architectures["laplace"].build(causal, 0)
    assert laplace(causal.train_batch).shape == causal.train_target.shape

    spherical = split_operator_scenario(
        _ladder(quick_ladders, "spherical_field").levels[0], seed=1729
    )
    spherical_architectures = {
        architecture.name: architecture
        for architecture in compatible_architectures(spherical, quick=True)
    }
    assert {"fno", "tfno", "cno", "sfno", "pod_linear_rom"} <= set(
        spherical_architectures
    )
    assert "uno" not in spherical_architectures
    sfno = spherical_architectures["sfno"].build(spherical, 0)
    assert sfno(spherical.train_batch).shape == spherical.train_target.shape

    pod = spherical_architectures["pod_deeponet"].build(spherical, 0)
    trained, *_ = train_operator(pod, spherical, steps=1)
    assert jnp.array_equal(pod.trunk.values, trained.trunk.values)
    assert parameter_count(pod) < sum(
        leaf.size
        for leaf in jax.tree_util.tree_leaves(pod)
        if isinstance(leaf, jax.Array)
    )


@pytest.mark.parametrize("comparison", ["capacity", "compute"])
def test_v2_runs_matched_search_and_persists_artifacts(
    quick_ladders, tmp_path, comparison, monkeypatch
):
    evaluation_calls = 0
    evaluate_operator = benchmark_v2.evaluate_operator

    def count_evaluation(*args, **kwargs):
        nonlocal evaluation_calls
        evaluation_calls += 1
        return evaluate_operator(*args, **kwargs)

    monkeypatch.setattr(benchmark_v2, "evaluate_operator", count_evaluation)
    ladder = _ladder(quick_ladders, "independent_query")
    protocol = OperatorBenchmarkProtocol(
        seeds=(0,),
        comparison=comparison,
        steps=1,
        learning_rates=(1e-3, 3e-3),
        repeats=1,
        target_parameters=500 if comparison == "capacity" else None,
        size_scales=(0.5, 1.0),
        quick=True,
        commit_identity="test-v2",
    )
    result = run_operator_benchmark_v2(
        (ladder,),
        protocol=protocol,
        architecture_names=("deeponet",),
        difficulty="easy",
    )
    assert len(result.comparisons) == 1
    record = result.comparisons[0]
    assert record.mode == comparison
    assert record.actual_parameters > 0
    assert record.actual_compute_units > 0
    if comparison == "capacity":
        assert record.capacity_ratio is not None
        assert 0.5 <= record.capacity_ratio <= 2.0
    else:
        assert record.compute_ratio is not None
        assert record.planned_steps > 0
        assert record.compute_measurement == "jax_flops"
        assert record.training_step_flops > 0
        assert record.training_step_bytes > 0
    assert len(result.trials) == 2
    assert sum(trial.selected for trial in result.trials) == 1
    assert len(result.results) == 1
    assert result.results[0].evaluations
    assert evaluation_calls == len(result.results[0].evaluations)
    assert all(
        len(trial.learning_curve) == trial.training_steps for trial in result.trials
    )
    assert all(trial.validation_steps[0] == 0 for trial in result.trials)
    assert all(
        trial.validation_steps[-1] == trial.training_steps for trial in result.trials
    )
    assert all(
        len(trial.validation_curve) == len(trial.validation_steps)
        for trial in result.trials
    )
    assert all(trial.normalization == "sourcewise" for trial in result.trials)
    assert result.portfolio_promotions[0].tier == "experimental"
    assert all(
        "quick smoke profiles are not promotion-eligible" in report.reasons
        for report in result.promotions
    )

    paths = save_benchmark_v2_artifacts(tmp_path / comparison, result)
    assert len(paths) == 9
    assert all(path.exists() for path in paths)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert "schema_version" not in payload
    assert payload["trials"][0]["learning_curve"]
    assert payload["external_audits"] == []
    assert "peak_memory_bytes_mean" in pl.read_parquet(paths[1]).columns
    assert pl.read_parquet(paths[2]).height == 2
    assert pl.read_parquet(paths[3]).height == 1
    assert pl.read_parquet(paths[4]).height == 0
    assert all(pl.read_parquet(paths[index]).height == 1 for index in range(5, 9))


def test_pareto_mode_sweeps_sizes_without_fabricating_missing_metrics(
    quick_ladders,
):
    ladder = _ladder(quick_ladders, "independent_query")
    protocol = OperatorBenchmarkProtocol(
        seeds=(0,),
        comparison="pareto",
        steps=1,
        learning_rates=(1e-3,),
        repeats=1,
        size_scales=(0.75, 1.0),
        quick=True,
        profile="smoke",
        commit_identity="test-pareto",
    )
    result = run_operator_benchmark_v2(
        (ladder,),
        protocol=protocol,
        architecture_names=("deeponet",),
        difficulty="easy",
    )
    assert len(result.comparisons) == 2
    assert {record.size_scale for record in result.comparisons} == {0.75, 1.0}
    assert all(record.compute_measurement == "jax_flops" for record in result.comparisons)
    assert all(record.training_step_flops > 0 for record in result.comparisons)
    assert len(result.pareto_fronts) == 1
    assert len(result.pareto_fronts[0].points) == 2
    for point in result.pareto_fronts[0].points:
        if not point.complete:
            assert point.nondominated is None
    assert all(
        "Pareto reporting is not a promotion matching gate" in report.reasons
        for report in result.promotions
    )


def test_capacity_target_must_lie_in_architecture_feasible_range(quick_ladders):
    scenario = split_operator_scenario(
        _ladder(quick_ladders, "independent_query").levels[0],
        seed=1729,
    )
    architecture = next(
        architecture
        for architecture in compatible_architectures(scenario, quick=True)
        if architecture.name == "deeponet"
    )
    protocol = OperatorBenchmarkProtocol(
        comparison="capacity",
        target_parameters=1,
        size_scales=(0.5, 1.0),
        quick=True,
        profile="smoke",
    )
    with pytest.raises(ValueError, match="outside the common feasible range"):
        _target_parameters((architecture,), scenario, protocol)

    with pytest.raises(ValueError, match="resume requires"):
        OperatorBenchmarkProtocol(resume=True)
    with pytest.raises(ValueError, match="immutable commit identity"):
        OperatorBenchmarkProtocol(
            checkpoint_directory="checkpoints",
            resume=True,
        )


def test_run_rejects_requested_architecture_missing_from_all_scenarios(
    quick_ladders,
):
    protocol = OperatorBenchmarkProtocol(
        seeds=(0,),
        comparison="pareto",
        steps=0,
        learning_rates=(1e-3,),
        repeats=1,
        size_scales=(1.0,),
        quick=True,
        profile="smoke",
    )
    with pytest.raises(ValueError, match="incompatible with every selected scenario"):
        run_operator_benchmark_v2(
            (_ladder(quick_ladders, "independent_query"),),
            protocol=protocol,
            architecture_names=("constant", "pod_linear_rom"),
            difficulty="easy",
        )


def _aggregate(evaluation, shift, relative_l2):
    return OperatorBenchmarkAggregate(
        scenario="scenario",
        architecture="operator",
        family="family",
        evaluation=evaluation,
        split="test",
        shift=shift,
        seeds=(0, 1, 2, 3, 4),
        parameter_count_mean=100.0,
        relative_l2_mean=relative_l2,
        relative_l2_std=0.01,
        absolute_l2_mean=relative_l2,
        h1_mean=None,
        spectral_mean=None,
        conservation_error_mean=0.0,
        maximum_absolute_error_mean=relative_l2,
        compile_seconds_mean=0.01,
        inference_seconds_mean=0.001,
        training_seconds_mean=1.0,
        peak_memory_bytes_mean=1024.0,
        convergence_rate=1.0,
    )


def test_native_kernel_parity_checks_pass():
    checks = native_kernel_parity_checks()
    assert {check.family for check in checks} == {
        "branch_trunk",
        "spectral",
        "laplace_temporal",
        "spherical_spectral",
        "geometry_informed",
        "regional_graph",
        "geometry_transformer",
    }
    assert all(check.passed for check in checks)


def test_pinned_family_parity_evidence_is_loadable():
    path = (
        Path(__file__).parents[2]
        / "tools"
        / "operator_benchmarks"
        / "reference"
        / "family_parity.json"
    )
    evidence = load_family_parity_evidence(path)
    assert {record.family for record in evidence} == {
        "branch_trunk",
        "spectral",
        "laplace_temporal",
        "spherical_spectral",
        "geometry_informed",
        "regional_graph",
        "geometry_transformer",
    }
    assert all(record.verified for record in evidence)
    by_family = {record.family: record for record in evidence}
    assert by_family["geometry_informed"].revision == (
        "86a8bc7812a31b42c4f7895693cf4ac11521c066"
    )
    assert by_family["regional_graph"].revision == (
        "3e4b307c90f34237d0c1e5e497d4301116e9c3db"
    )
    assert by_family["geometry_transformer"].revision == (
        "549c5a5f7113e23ba5e91469f2f8bbb1567fae46"
    )


def test_promotion_requires_all_gates_and_pinned_family_parity():
    aggregates = (
        _aggregate("base", "in_distribution", 0.1),
        _aggregate("noise", "input_noise", 0.15),
    )
    audit = ScenarioIntegrityAudit(
        scenario="scenario",
        checksum="0" * 64,
        train_cases=10,
        validation_cases=3,
        test_cases=3,
        physical_split_disjoint=True,
        provenance_complete=True,
        dimensional_ranges_declared=True,
        nondimensional_ranges_declared=True,
        reference_converged=True,
        reference_relative_error=0.0,
        reference_tolerance=0.0,
        near_identity=NearIdentityDiagnostic(
            "scenario", "nearest_neighbor", 1.0, 0.05, False
        ),
        passed=True,
        reasons=(),
    )
    comparison = BenchmarkComparisonRecord(
        scenario="scenario",
        architecture="operator",
        mode="capacity",
        size_scale=1.0,
        target_parameters=100,
        actual_parameters=100,
        capacity_ratio=1.0,
        planned_steps=10,
        target_compute_units=None,
        actual_compute_units=1000,
        compute_ratio=None,
        normalization="sourcewise",
    )
    parity = FamilyParityEvidence(
        family="family",
        reference_uri="https://example.test/upstream",
        revision="abc123",
        status="verified",
        checks=(
            KernelParityCheck(
                "published-case",
                "family",
                "upstream-checkpoint",
                1e-7,
                1e-6,
                True,
            ),
        ),
    )
    criteria = PromotionCriteria(
        minimum_general_scenarios=1,
        minimum_external_scenarios=1,
    )
    difficulty = ScenarioDifficultyAudit(
        scenario="scenario",
        identity_relative_l2=1.0,
        persistence_relative_change=1.0,
        nearest_realization_relative_distance=1.0,
        source_effective_rank=8.0,
        source_rank_99=8,
        source_rank_fraction=0.8,
        target_effective_rank=8.0,
        target_rank_99=8,
        target_rank_fraction=0.8,
        passed=True,
        reasons=(),
    )
    reports = _promotion_reports(
        aggregates,
        (audit,),
        (comparison,),
        {("scenario", "operator"): "general"},
        (parity,),
        criteria,
        difficulty_audits=(difficulty,),
    )
    assert reports[0].promoted
    portfolio = _portfolio_promotions(reports, criteria)
    assert portfolio[0].promoted
    assert portfolio[0].tier == "validated"

    without_parity = _promotion_reports(
        aggregates,
        (audit,),
        (comparison,),
        {("scenario", "operator"): "general"},
        (),
        criteria,
        difficulty_audits=(difficulty,),
    )
    assert not without_parity[0].promoted
    assert without_parity[0].tier == "experimental"
    assert "parity" in " ".join(without_parity[0].reasons)

    unpinned = _promotion_reports(
        aggregates,
        (audit,),
        (comparison,),
        {("scenario", "operator"): "general"},
        (parity,),
        criteria,
        provenance_pinned=False,
        difficulty_audits=(difficulty,),
    )
    assert not unpinned[0].promoted
    assert "commit identity" in " ".join(unpinned[0].reasons)

    unmatched = _promotion_reports(
        aggregates,
        (audit,),
        (
            replace(
                comparison,
                mode="native",
                target_parameters=None,
                capacity_ratio=None,
            ),
        ),
        {("scenario", "operator"): "general"},
        (parity,),
        criteria,
        difficulty_audits=(difficulty,),
    )
    assert not unmatched[0].promoted
    assert "matching tolerance" in " ".join(unmatched[0].reasons)

    external_without_audit = _promotion_reports(
        aggregates,
        (audit,),
        (comparison,),
        {("scenario", "operator"): "external"},
        (parity,),
        criteria,
        difficulty_audits=(difficulty,),
    )
    assert not external_without_audit[0].promoted
    assert external_without_audit[0].external_audit_passed is False

    external_with_audit = _promotion_reports(
        aggregates,
        (audit,),
        (comparison,),
        {("scenario", "operator"): "external"},
        (parity,),
        criteria,
        external_audits=(ExternalCandidateAudit("operator", True, (), True),),
        difficulty_audits=(difficulty,),
    )
    assert external_with_audit[0].promoted

    failed_hardness = _promotion_reports(
        aggregates,
        (audit,),
        (comparison,),
        {("scenario", "operator"): "general"},
        (parity,),
        criteria,
        difficulty_audits=(replace(difficulty, passed=False, reasons=("low rank",)),),
    )
    assert not failed_hardness[0].promoted
    assert not failed_hardness[0].baseline_hardness_passed

    unconverged = _promotion_reports(
        tuple(replace(row, convergence_rate=0.0) for row in aggregates),
        (audit,),
        (comparison,),
        {("scenario", "operator"): "general"},
        (parity,),
        criteria,
        difficulty_audits=(difficulty,),
    )
    assert not unconverged[0].promoted
    assert not unconverged[0].convergence_passed
    assert external_with_audit[0].tier == "external"


def test_difficulty_audit_detects_persistence_shortcut():
    scenario = split_operator_scenario(
        spherical_diffusion_scenario(
            num_cases=32,
            diffusivity=0.005,
            dt=0.2,
            target_steps=12,
            maximum_degree=5,
            seed=71,
        ),
        seed=73,
    )
    criteria = PromotionCriteria()
    audit = audit_scenario_difficulty(scenario, criteria)
    assert audit.target_effective_rank >= criteria.minimum_target_effective_rank
    assert (
        audit.nearest_realization_relative_distance
        >= criteria.minimum_nearest_realization_distance
    )
    source = jnp.asarray(scenario.train_batch.input("field").values)
    shortcut = replace(scenario, train_target=source)
    shortcut_audit = audit_scenario_difficulty(shortcut, criteria)
    assert not shortcut_audit.passed
    assert shortcut_audit.identity_relative_l2 == 0.0
    assert "identity baseline" in " ".join(shortcut_audit.reasons)


def test_shortlist_physical_ladders_pass_hardness_contracts():
    criteria = PromotionCriteria()
    audits = tuple(
        audit_scenario_difficulty(
            split_operator_scenario(scenario, seed=1729),
            criteria,
        )
        for ladder in standard_operator_benchmark_ladders(profile="shortlist")
        for scenario in ladder.levels
    )
    assert len(audits) == 40
    assert all(audit.passed for audit in audits)
    assert min(audit.source_rank_99 for audit in audits) >= 4
    non_harmonic = tuple(
        audit
        for audit in audits
        if not audit.scenario.startswith("cochain_annulus_harmonic_projection")
    )
    harmonic = tuple(audit for audit in audits if audit not in non_harmonic)
    assert min(audit.target_rank_99 for audit in non_harmonic) >= 4
    assert all(
        audit.target_rank_99 == 1 and audit.target_rank_fraction == 1.0
        for audit in harmonic
    )


def test_pareto_front_reports_dominance_and_missing_metrics():
    first_rows = (
        replace(
            _aggregate("base", "in_distribution", 0.1),
            architecture="a",
            family="a-family",
        ),
        replace(
            _aggregate("shift", "input_noise", 0.15),
            architecture="a",
            family="a-family",
        ),
    )
    second_rows = tuple(
        replace(
            row,
            architecture="b",
            family="b-family",
            relative_l2_mean=row.relative_l2_mean * 2.0,
            inference_seconds_mean=row.inference_seconds_mean * 2.0,
            peak_memory_bytes_mean=2048.0,
        )
        for row in first_rows
    )
    comparisons = (
        BenchmarkComparisonRecord(
            "scenario",
            "a",
            "pareto",
            1.0,
            None,
            100,
            None,
            10,
            None,
            1_000,
            None,
            "sourcewise",
            training_step_flops=100,
            compute_measurement="jax_flops",
        ),
        BenchmarkComparisonRecord(
            "scenario",
            "b",
            "pareto",
            1.0,
            None,
            200,
            None,
            10,
            None,
            2_000,
            None,
            "sourcewise",
            training_step_flops=200,
            compute_measurement="jax_flops",
        ),
    )
    trials = tuple(
        HyperparameterTrial(
            scenario="scenario",
            architecture=architecture,
            family=f"{architecture}-family",
            seed=0,
            learning_rate=1e-3,
            size_scale=1.0,
            normalization="sourcewise",
            parameter_count=100 if architecture == "a" else 200,
            training_steps=10,
            training_seconds=1.0,
            initial_loss=1.0,
            final_loss=validation,
            validation_loss=validation,
            learning_curve=(validation,),
            selected=True,
            converged=True,
        )
        for architecture, validation in (("a", 0.1), ("b", 0.2))
    )
    front = _pareto_fronts(first_rows + second_rows, trials, comparisons)[0]
    lookup = {point.architecture: point for point in front.points}
    assert lookup["a"].nondominated is True
    assert lookup["b"].nondominated is False
    assert lookup["b"].dominated_by == ("a@1",)

    incomplete_rows = tuple(
        replace(row, peak_memory_bytes_mean=None) for row in first_rows + second_rows
    )
    incomplete = _pareto_fronts(incomplete_rows, trials, comparisons)[0]
    assert all(not point.complete for point in incomplete.points)
    assert all(point.nondominated is None for point in incomplete.points)


def test_square_symmetry_contracts_preserve_fno_baselines_and_augmentation():
    d4 = square_diffusion_symmetry_scenario(
        resolution=9,
        num_cases=3,
        seed=11,
    )
    c4 = square_diffusion_symmetry_scenario(
        resolution=9,
        num_cases=3,
        chiral_strength=1e-5,
        seed=11,
    )
    broken = square_diffusion_symmetry_scenario(
        resolution=9,
        num_cases=3,
        diffusivity=(0.01, 0.03),
        seed=11,
    )

    assert d4.symmetry is not None
    assert d4.symmetry.group == "p4m"
    assert max(defect for _, defect in d4.symmetry.reference_defects) < 1e-12
    assert c4.symmetry is not None
    assert c4.symmetry.group == "p4"
    assert max(defect for _, defect in c4.symmetry.reference_defects[:4]) < 1e-12
    assert min(defect for _, defect in c4.symmetry.reference_defects[4:]) > 1e-2
    assert broken.symmetry is not None
    assert broken.symmetry.group is None
    assert broken.symmetry.intentionally_violated
    assert tuple(evaluation.shift for evaluation in d4.evaluations) == (
        "in_distribution",
        "resolution",
        "forcing_spectrum",
    )
    assert d4.evaluations[1].batch.require_single_query().sample_shape == (13, 13)
    assert set(d4.evaluations[0].case_ids) == set(d4.evaluations[1].case_ids)
    assert set(d4.evaluations[0].case_ids).isdisjoint(d4.evaluations[2].case_ids)

    d4_architectures = {
        architecture.name: architecture
        for architecture in compatible_architectures(d4, quick=True)
    }
    c4_architectures = {
        architecture.name: architecture
        for architecture in compatible_architectures(c4, quick=True)
    }
    broken_names = {
        architecture.name for architecture in compatible_architectures(broken, quick=True)
    }
    expected_symmetry_models = {
        "fno",
        "fno_p4_augmented",
        "lattice_equivariant_cno",
    }
    assert expected_symmetry_models <= set(d4_architectures)
    assert expected_symmetry_models <= set(c4_architectures)
    assert "fno_p4_augmented" not in broken_names
    assert "lattice_equivariant_cno" not in broken_names

    augmented = d4_architectures["fno_p4_augmented"].training_scenario(d4)
    assert augmented.train_batch.case_shape == (12,)
    assert augmented.train_target.shape == (12, 9, 9)
    assert jnp.array_equal(augmented.train_target[:3], d4.train_target)
    assert jnp.array_equal(
        augmented.train_target[3:6],
        jnp.rot90(d4.train_target, k=1, axes=(1, 2)),
    )
    assert len(set(augmented.case_ids)) == 12

    output = d4_architectures["fno_p4_augmented"].build(
        d4,
        seed=0,
    )(d4.train_batch)
    assert output.shape == d4.train_target.shape
    assert jnp.all(jnp.isfinite(output))
    equivariant_output = d4_architectures["lattice_equivariant_cno"].build(
        d4,
        seed=0,
    )(d4.train_batch)
    assert equivariant_output.shape == d4.train_target.shape
    assert jnp.all(jnp.isfinite(equivariant_output))


def test_scenario_checksum_includes_structured_symmetry_contract():
    scenario = square_diffusion_symmetry_scenario(
        resolution=9,
        num_cases=2,
    )
    assert scenario.symmetry is not None
    changed = replace(
        scenario,
        symmetry=replace(
            scenario.symmetry,
            action_convention="different convention",
        ),
    )
    assert scenario_checksum(scenario) != scenario_checksum(changed)

    tolerance = scenario.symmetry.reference_tolerance
    first = replace(
        scenario,
        symmetry=replace(
            scenario.symmetry,
            reference_defects=tuple((index, 0.25 * tolerance) for index in range(8)),
        ),
    )
    second = replace(
        scenario,
        symmetry=replace(
            scenario.symmetry,
            reference_defects=tuple((index, 0.75 * tolerance) for index in range(8)),
        ),
    )
    failed = replace(
        scenario,
        symmetry=replace(
            scenario.symmetry,
            reference_defects=((0, 2.0 * tolerance),)
            + tuple((index, 0.25 * tolerance) for index in range(1, 8)),
        ),
    )
    assert scenario_checksum(first) == scenario_checksum(second)
    assert scenario_checksum(first) != scenario_checksum(failed)


def test_symmetry_benchmark_records_fno_defects_and_durable_artifact(tmp_path):
    easy = benchmark_v2._tag_level(
        square_diffusion_symmetry_scenario(
            resolution=9,
            num_cases=6,
            seed=41,
        ),
        "focused_square_symmetry",
        "easy",
    )
    hard = benchmark_v2._tag_level(
        square_diffusion_symmetry_scenario(
            resolution=9,
            num_cases=6,
            chiral_strength=1e-5,
            seed=42,
        ),
        "focused_square_symmetry",
        "hard",
    )
    ladder = benchmark_v2.OperatorBenchmarkLadder(
        "focused_square_symmetry",
        "square_group_equivariance",
        (easy, hard),
    )
    result = run_operator_benchmark_v2(
        (ladder,),
        protocol=OperatorBenchmarkProtocol(
            seeds=(0,),
            comparison="native",
            steps=0,
            learning_rates=(1e-3,),
            sample_fractions=(0.5, 1.0),
            repeats=1,
            validation_interval=1,
            quick=True,
            profile="smoke",
        ),
        architecture_names=(
            "fno",
            "fno_p4_augmented",
            "lattice_equivariant_cno",
        ),
        difficulty="easy",
    )
    records = {record.architecture: record for record in result.symmetry_results}

    assert result.symmetry_audits[0].passed
    assert records["fno"].worst_equivariance_defect > 1e-4
    assert records["lattice_equivariant_cno"].worst_equivariance_defect < 1e-10
    assert len(result.sample_efficiency) == 3
    assert all(
        curve.sample_fractions == (2 / 3, 1.0)
        and jnp.isfinite(curve.area_under_sample_error_curve)
        for curve in result.sample_efficiency
    )

    paths = save_benchmark_v2_artifacts(tmp_path, result)
    symmetry_path = next(
        path for path in paths if path.name == "operator_symmetry_v2.parquet"
    )
    symmetry_frame = pl.read_parquet(symmetry_path)
    assert set(symmetry_frame["architecture"]) == {
        "fno",
        "fno_p4_augmented",
        "lattice_equivariant_cno",
    }
    sample_path = next(
        path for path in paths if path.name == "operator_sample_efficiency_v2.parquet"
    )
    assert set(pl.read_parquet(sample_path)["architecture"]) == {
        "fno",
        "fno_p4_augmented",
        "lattice_equivariant_cno",
    }
    assert all(path.name != "operator_symmetry_decisions_v2.parquet" for path in paths)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["symmetry_audits"][0]["passed"]
    assert len(payload["symmetry_results"]) == 3
    assert "symmetry_decisions" not in payload


def test_scenario_checksum_covers_provenance_and_metadata(quick_ladders):
    scenario = _ladder(quick_ladders, "independent_query").levels[0]
    changed = replace(scenario, metadata=scenario.metadata + (("revision", "changed"),))
    assert scenario_checksum(scenario) != scenario_checksum(changed)


def test_scenario_checksum_uses_reference_verdict_not_roundoff():
    scenario = navier_stokes_scenario(
        resolution=8,
        num_cases=4,
        target_steps=2,
        seed=17,
    )
    evidence = scenario.reference_evidence
    assert evidence is not None
    first = replace(
        scenario,
        reference_evidence=replace(
            evidence,
            relative_error=0.25 * evidence.tolerance,
        ),
    )
    second = replace(
        scenario,
        reference_evidence=replace(
            evidence,
            relative_error=0.75 * evidence.tolerance,
        ),
    )
    failed = replace(
        scenario,
        reference_evidence=replace(
            evidence,
            relative_error=2.0 * evidence.tolerance,
        ),
    )

    assert scenario_checksum(first) == scenario_checksum(second)
    assert scenario_checksum(first) != scenario_checksum(failed)


def test_external_candidate_must_win_robustness_and_complexity():
    candidate = ExternalOperatorCandidate(
        name="external",
        source_uri="https://example.test/source",
        checkpoint_uri="https://example.test/checkpoint",
        revision="abc123",
        code_license="Apache-2.0",
        weights_license="Apache-2.0",
        input_schema_declared=True,
        output_schema_declared=True,
        preprocessing_declared=True,
        normalization_declared=True,
        dataset_provenance_declared=True,
        checkpoint_sha256="0" * 64,
    )
    audit = audit_external_candidate(candidate)
    native = (
        replace(
            _aggregate("base", "in_distribution", 0.2),
            architecture="fno",
            family="spectral",
        ),
        replace(
            _aggregate("noise", "input_noise", 0.3),
            architecture="fno",
            family="spectral",
        ),
    )
    external = (
        replace(
            _aggregate("base", "in_distribution", 0.1),
            architecture="external",
            family="external",
        ),
        replace(
            _aggregate("noise", "input_noise", 0.18),
            architecture="external",
            family="external",
            parameter_count_mean=1000.0,
        ),
    )
    decision = select_benchmark_superior_external(
        candidate,
        audit,
        external,
        native,
        maximum_parameter_ratio=2.0,
    )
    assert not decision.integrated
    assert not decision.robustness_passed
    assert not decision.complexity_passed
