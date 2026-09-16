from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cosmology._dark_matter_inference import (
    ConstantExternalDarkMatterProduct,
    DarkMatterCoordinateContract,
    DarkMatterDiscrepancyPlan,
    DarkMatterEmulatorCalibrationPlan,
    DarkMatterErrorBudget,
    DarkMatterInferenceEvaluation,
    ExternalDarkMatterEmulatorProduct,
    FixedTapeStochasticEvaluation,
    FixedTapeStochasticSensitivityPlan,
    SmoothFixedGridDarkMatterInferencePlan,
)
from phydrax.applications.cosmology._products import CosmologyProductProvenance
from phydrax.artifacts import DifferentiationContract, ScientificArtifactEnvelope
from phydrax.observation import (
    CholeskyCovarianceAction,
    CoordinateLayout,
    TheoryVector,
)
from phydrax.qualification import ReferenceArtifactManifest


jax.config.update("jax_enable_x64", True)


def _provenance(*, source_kind="native", differentiation=None):
    contract = (
        DifferentiationContract.native() if differentiation is None else differentiation
    )
    return CosmologyProductProvenance(
        producer="test-dark-matter-inference",
        producer_version="native",
        model_form_id="dm-model",
        request_id="request",
        numerical_policy_id="numerics",
        physics_policy_id="physics",
        scale_id="code-cosmology",
        source_kind=source_kind,
        differentiation=contract,
    )


def _decode_float64(data):
    return np.frombuffer(data, dtype=np.float64).copy()


def _external_product(values):
    values_host = np.asarray(values, dtype=np.float64)
    payload = values_host.tobytes()
    manifest = ReferenceArtifactManifest(
        "heldout-dark-matter-reference",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="test-license",
        commercial_use_permitted=True,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="EAR99",
        nondimensionalization={"length": 1.0, "mass": 1.0, "time": 1.0},
        uncertainty={"standard_error": 0.1},
        lineage_ids=("independent-measurement",),
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind="dark-matter-reference-vector",
        content_digest=hashlib.sha256(payload).hexdigest(),
        producer="independent-provider",
        producer_version="1",
        build_id="release",
        license_id="test-license",
        resource_id="reference-resource",
        status="complete",
        parent_artifact_ids=(manifest.manifest_id,),
    )
    coordinates = DarkMatterCoordinateContract(
        CoordinateLayout(
            tuple(f"reference:{index}" for index in range(values_host.size))
        ),
        ("dimensionless-observable",) * values_host.size,
    )
    return ConstantExternalDarkMatterProduct(
        values_host,
        payload,
        _decode_float64,
        manifest,
        artifact,
        _provenance(
            source_kind="external",
            differentiation=DifferentiationContract.constant(),
        ),
        coordinates,
        decoder_id="float64-vector-v1",
        product_id="heldout-reference-product",
    )


def _smooth_evaluator(parameters):
    return DarkMatterInferenceEvaluation(
        jnp.asarray((parameters[0] ** 2, parameters[0] * parameters[1])),
        jnp.asarray((3, 7)),
        finite=True,
        successful=True,
        fixed_grid=True,
        topology_fixed=True,
        event_free=True,
        reaction_free=True,
        derivative_valid=True,
        product_id="smooth-product",
        realization_id="fixed-realization",
    )


def test_smooth_fixed_grid_wave_jvp_matches_known_derivative():
    plan = SmoothFixedGridDarkMatterInferencePlan(
        _smooth_evaluator,
        jnp.asarray((3, 7)),
        parameter_count=2,
        output_count=2,
        kind="wave-fixed-grid",
        topology_sensitive=False,
        evaluator_id="known-polynomial",
        realization_id="fixed-realization",
        product_id="smooth-product",
    )

    result = plan.sensitivity(
        jnp.asarray((2.0, 3.0)),
        jnp.asarray((0.5, -1.0)),
        epsilon=1e-4,
        absolute_tolerance=1e-8,
        relative_tolerance=1e-7,
    )

    np.testing.assert_allclose(result.value, jnp.asarray((4.0, 6.0)))
    np.testing.assert_allclose(result.jvp, jnp.asarray((2.0, -0.5)))
    np.testing.assert_allclose(result.finite_difference, result.jvp, atol=1e-8)
    assert bool(result.successful)


def test_soliton_and_vortex_topology_products_refuse_gradients():
    plan = SmoothFixedGridDarkMatterInferencePlan(
        _smooth_evaluator,
        jnp.asarray((3, 7)),
        parameter_count=2,
        output_count=2,
        kind="wave-fixed-grid",
        topology_sensitive=True,
        evaluator_id="topology-product",
        realization_id="fixed-realization",
        product_id="smooth-product",
    )

    with pytest.raises(ValueError, match="Topology-sensitive"):
        plan.sensitivity(jnp.asarray((2.0, 3.0)), jnp.asarray((1.0, 0.0)))


def _stochastic_evaluator(score_multiplier):
    def evaluate(parameters, tape):
        values = (parameters[0] + tape)[:, None]
        score = (score_multiplier * tape)[:, None]
        active = jnp.ones(tape.shape, dtype=bool)
        inactive_event = jnp.zeros(tape.shape, dtype=bool)
        return FixedTapeStochasticEvaluation(
            values,
            score,
            active,
            event_occurred=inactive_event,
            reaction_occurred=inactive_event,
            topology_changed=inactive_event,
            finite=True,
            successful=True,
            tape_id="antithetic-tape",
            product_id="stochastic-mean",
        )

    return evaluate


def _stochastic_plan(score_multiplier, tape=None):
    tape_ = jnp.asarray((-1.0, 1.0, -1.0, 1.0)) if tape is None else jnp.asarray(tape)
    return FixedTapeStochasticSensitivityPlan(
        _stochastic_evaluator(score_multiplier),
        tape_,
        jnp.ones(tape_.shape),
        parameter_count=1,
        output_count=1,
        minimum_effective_sample_size=4.0,
        bias_absolute_tolerance=1e-12,
        bias_standard_error_multiplier=2.0,
        evaluator_id=f"location-family-score-{score_multiplier}",
        tape_id="antithetic-tape",
        product_id="stochastic-mean",
    )


def test_fixed_tape_score_and_common_random_difference_report_uncertainty():
    result = _stochastic_plan(1.0).sensitivity(
        jnp.asarray((2.0,)), jnp.asarray((1.0,)), epsilon=1e-4
    )

    np.testing.assert_allclose(result.value, jnp.asarray((2.0,)), atol=1e-12)
    np.testing.assert_allclose(result.score_estimate, jnp.asarray((1.0,)), atol=1e-12)
    np.testing.assert_allclose(
        result.common_random_finite_difference, jnp.asarray((1.0,)), atol=1e-10
    )
    np.testing.assert_allclose(result.estimator_bias, 0.0, atol=1e-10)
    np.testing.assert_allclose(
        result.score_standard_error, jnp.asarray((2.0 / jnp.sqrt(3.0),))
    )
    np.testing.assert_allclose(
        result.paired_bias_standard_error,
        jnp.asarray((2.0 / jnp.sqrt(3.0),)),
    )
    np.testing.assert_allclose(
        result.combined_standard_error,
        result.paired_bias_standard_error,
    )
    changed_tape = _stochastic_plan(1.0, tape=(-2.0, 2.0, -2.0, 2.0))
    assert changed_tape.tape_id == _stochastic_plan(1.0).tape_id
    assert changed_tape.tape_content_id != _stochastic_plan(1.0).tape_content_id
    assert changed_tape.plan_id != _stochastic_plan(1.0).plan_id
    np.testing.assert_allclose(result.finite_difference_standard_error, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.effective_sample_size, 4.0)
    assert not bool(jnp.any(result.bias_flag))
    assert bool(result.fixed_tape)
    assert bool(result.event_free)
    assert bool(result.reaction_free)
    assert bool(result.topology_fixed)
    assert bool(result.successful)


def test_stochastic_sensitivity_flags_biased_score_estimator():
    result = _stochastic_plan(0.0).sensitivity(
        jnp.asarray((2.0,)), jnp.asarray((1.0,)), epsilon=1e-4
    )

    np.testing.assert_allclose(result.score_estimate, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.common_random_finite_difference, 1.0, atol=1e-10)
    assert bool(jnp.all(result.bias_flag))
    assert not bool(result.successful)


def test_discrepancy_budget_preserves_decomposition_and_exact_standardization():
    layout = CoordinateLayout(("density", "dispersion"))
    coordinates = DarkMatterCoordinateContract(
        layout,
        ("code-mass-density", "code-velocity"),
    )
    prediction = TheoryVector(jnp.asarray((10.0, 30.0)), layout, "prediction")
    target = TheoryVector(jnp.asarray((5.0, 20.0)), layout, "target")
    budget = DarkMatterErrorBudget(
        {
            "numerical": jnp.asarray((3.0, 6.0)),
            "sampling": jnp.asarray((4.0, 8.0)),
        },
        coordinates,
        source_ids=("numerical-study", "sampling-study"),
    )
    covariance = CholeskyCovarianceAction(
        jnp.diag(jnp.asarray((5.0, 10.0))),
        layout,
    )
    product = DarkMatterDiscrepancyPlan(degrees_of_freedom=2).evaluate(
        prediction,
        target,
        budget,
        covariance,
        _provenance(),
        prediction_coordinates=coordinates,
        target_coordinates=coordinates,
    )

    np.testing.assert_allclose(budget.total_standard_error, jnp.asarray((5.0, 10.0)))
    mismatched_coordinates = DarkMatterCoordinateContract(
        layout,
        ("wrong-density-unit", "wrong-velocity-unit"),
    )
    mismatched_budget = DarkMatterErrorBudget(
        {"numerical": jnp.ones((2,))},
        mismatched_coordinates,
        source_ids=("mismatched-unit-study",),
    )
    with pytest.raises(ValueError, match="coordinate/unit contracts"):
        DarkMatterDiscrepancyPlan(degrees_of_freedom=2).evaluate(
            prediction,
            target,
            mismatched_budget,
            covariance,
            _provenance(),
            prediction_coordinates=coordinates,
            target_coordinates=coordinates,
        )
    np.testing.assert_allclose(product.residual, jnp.asarray((5.0, 10.0)))
    np.testing.assert_allclose(product.standardized_residual, jnp.asarray((1.0, 1.0)))
    np.testing.assert_allclose(product.chi_square, 2.0)
    np.testing.assert_allclose(product.reduced_chi_square, 1.0)
    assert budget.component_names == ("numerical", "sampling")
    assert bool(product.successful)


def test_external_reference_is_constant_and_emulator_calibration_keeps_lineage():
    reference = _external_product(jnp.asarray((-1.0, 1.0, -1.0, 1.0)))
    assert reference.as_theory_vector().layout.layout_id == reference.layout.layout_id
    tangent = jax.jvp(
        lambda value: jnp.sum(jax.lax.stop_gradient(value)),
        (reference.values,),
        (jnp.ones(4),),
    )[1]
    np.testing.assert_allclose(tangent, 0.0, atol=0.0)

    payload = np.asarray(reference.values, dtype=np.float64).tobytes()
    with pytest.raises(ValueError, match="decoder output"):
        ConstantExternalDarkMatterProduct(
            2.0 * reference.values,
            payload,
            _decode_float64,
            reference.manifest,
            reference.artifact,
            reference.provenance,
            reference.coordinates,
            decoder_id=reference.decoder_id,
            product_id=reference.declared_product_id,
        )
    commercial = ConstantExternalDarkMatterProduct(
        reference.values,
        payload,
        _decode_float64,
        reference.manifest,
        reference.artifact,
        reference.provenance,
        reference.coordinates,
        decoder_id=reference.decoder_id,
        product_id=reference.declared_product_id,
        commercial_use=True,
    )
    assert commercial.requested_use_id != reference.requested_use_id
    assert commercial.product_id != reference.product_id

    external_budget = DarkMatterErrorBudget(
        {"reference": jnp.ones((4,))},
        reference.coordinates,
        source_ids=(reference.product_id,),
    )
    covariance = CholeskyCovarianceAction(
        jnp.eye(4),
        reference.layout,
    )
    with pytest.raises(ValueError, match="governed"):
        DarkMatterDiscrepancyPlan(degrees_of_freedom=4).evaluate(
            reference.as_theory_vector(),
            reference,
            external_budget,
            covariance,
            reference.provenance,
            prediction_coordinates=reference.coordinates,
        )

    product = DarkMatterEmulatorCalibrationPlan(
        nominal_coverage=0.9,
        maximum_coverage_gap=0.11,
        minimum_calibration_count=4,
    ).fit(
        jnp.zeros((4,)),
        jnp.ones((4,)),
        reference,
        _provenance(),
        emulator_coordinates=reference.coordinates,
        emulator_product_id="native-heldout-emulator-prediction",
        split_id="heldout-four-cases",
    )

    np.testing.assert_allclose(product.calibrator.scale_multiplier, 1.0)
    np.testing.assert_allclose(product.calibrated_scale, 1.0)
    np.testing.assert_allclose(
        product.standardized_residual, reference.values, atol=1e-12
    )
    assert product.reference.manifest.manifest_id == reference.manifest.manifest_id
    assert bool(product.successful)

    mismatched_units = DarkMatterCoordinateContract(
        reference.layout,
        ("wrong-unit",) * reference.layout.size,
    )
    with pytest.raises(ValueError, match="coordinate/unit contracts"):
        DarkMatterEmulatorCalibrationPlan(minimum_calibration_count=4).fit(
            jnp.zeros((4,)),
            jnp.ones((4,)),
            reference,
            _provenance(),
            emulator_coordinates=mismatched_units,
            emulator_product_id="unit-mismatched-native-emulator",
            split_id="unit-mismatched-split",
        )

    with pytest.raises(ValueError, match="effective sample size"):
        DarkMatterEmulatorCalibrationPlan(minimum_calibration_count=4).fit(
            jnp.zeros((4,)),
            jnp.ones((4,)),
            reference,
            _provenance(),
            emulator_coordinates=reference.coordinates,
            emulator_product_id="underweighted-native-emulator",
            split_id="underweighted-split",
            weights=jnp.asarray((1.0, 0.0, 0.0, 0.0)),
        )


def test_external_emulator_calibration_denies_unlicensed_use_and_raw_bypass():
    payload = np.concatenate(
        (np.zeros((4,), dtype=np.float64), np.ones((4,), dtype=np.float64))
    ).tobytes()
    manifest = ReferenceArtifactManifest(
        "restricted-dark-matter-emulator",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="restricted-license",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="EAR99",
        nondimensionalization={"length": 1.0, "mass": 1.0, "time": 1.0},
        uncertainty={"standard_error": 0.2},
        lineage_ids=("restricted-provider",),
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind="external-dark-matter-emulator",
        content_digest=hashlib.sha256(payload).hexdigest(),
        producer="restricted-provider",
        producer_version="1",
        build_id="release",
        license_id="restricted-license",
        resource_id="restricted-emulator",
        status="complete",
        parent_artifact_ids=(manifest.manifest_id,),
    )
    layout = CoordinateLayout(tuple(f"case:{index}" for index in range(4)))
    coordinates = DarkMatterCoordinateContract(
        layout,
        ("dimensionless-observable",) * 4,
    )
    external_provenance = _provenance(
        source_kind="external",
        differentiation=DifferentiationContract.constant(),
    )
    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        ExternalDarkMatterEmulatorProduct(
            jnp.zeros((4,)),
            jnp.ones((4,)),
            payload,
            _decode_float64,
            coordinates,
            manifest,
            artifact,
            external_provenance,
            decoder_id="float64-emulator-location-scale-v1",
            product_id="restricted-emulator-product",
            commercial_use=True,
        )

    reference = _external_product(jnp.asarray((-1.0, 1.0, -1.0, 1.0)))
    plan = DarkMatterEmulatorCalibrationPlan(
        nominal_coverage=0.9,
        maximum_coverage_gap=0.11,
        minimum_calibration_count=4,
    )
    with pytest.raises(ValueError, match="fit_external"):
        plan.fit(
            jnp.zeros((4,)),
            jnp.ones((4,)),
            reference,
            external_provenance,
            emulator_product_id="ungoverned-raw-external",
            emulator_coordinates=reference.coordinates,
            split_id="heldout-four-cases",
        )
