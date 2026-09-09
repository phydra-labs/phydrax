#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Synthetic native transport inference; none of these data are foundry evidence."""

import hashlib
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._array_archive import read_array_archive
from phydrax.applications.semiconductor._calibration import (
    archive_semiconductor_calibration,
    archive_semiconductor_forward_evaluation,
    calibrate_semiconductor,
    evaluate_semiconductor_forward,
    prepare_semiconductor_calibration,
    qualify_semiconductor_calibration,
    semiconductor_identifiability,
    SemiconductorCalibrationCampaign,
    SemiconductorCalibrationCriteria,
    SemiconductorCalibrationDomain,
    SemiconductorForwardResult,
    SemiconductorForwardUnresolved,
    SemiconductorMeasurementCase,
    SemiconductorParameterBinding,
)
from phydrax.applications.semiconductor._quantities import (
    BOLTZMANN_CONSTANT_SI as KB,
    ELEMENTARY_CHARGE_SI as Q,
    MOBILITY_UNIT,
    PER_CUBIC_METER,
    SemiconductorQuantitySpec,
)
from phydrax.discretization import scharfetter_gummel_flux
from phydrax.linalg import DenseLU, LinearSolvePolicy
from phydrax.nonlinear import (
    NewtonKrylov,
    NonlinearSystemProblem,
    NonlinearTermination,
    root,
)
from phydrax.qualification import QualificationEvidence, ReferenceArtifactManifest
from phydrax.units import AMPERE, KELVIN, VOLT
from phydrax.uq import ParameterSpace


AREA_OVER_LENGTH = 1e-6  # m; explicit homogeneous resistor geometry
DENSITY = 1e21  # m^-3; independent synthetic metrology, not inferred from current


def _reference(label, *, training=True, redistribution=False):
    payload = f"Synthetic homogeneous SG resistor, {label}; SI q*n*mu*A*V/L".encode()
    return ReferenceArtifactManifest(
        label,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="test-synthetic-only",
        commercial_use_permitted=False,
        redistribution_permitted=redistribution,
        training_use_permitted=training,
        export_permitted=False,
        export_classification="synthetic-test",
        nondimensionalization={"SI": 1.0},
        uncertainty={"current_standard_error_A": 2e-7},
        lineage_ids=("analytic-ohmic-law",),
    )


def _quantity(name, kind, unit):
    return SemiconductorQuantitySpec(
        name,
        kind,
        unit,
        sign_convention="positive current into left contact"
        if kind == "current"
        else "physical SI",
        support_association="homogeneous-resistor",
        reference_configuration="fixed-SI-geometry",
    )


def _controls(voltage):
    voltage = jnp.asarray(voltage, dtype=float)
    return jnp.stack(
        (voltage, jnp.full_like(voltage, 300.0), jnp.full_like(voltage, DENSITY)), axis=-1
    )


def _current(parameters, case):
    """Genuine native SG number flux in a homogeneous isothermal resistor."""
    voltage, temperature, density = case.controls.T
    thermal_voltage = KB * temperature / Q
    mobility = jnp.sum(parameters)
    number_flux = (
        scharfetter_gummel_flux(
            density, density, -voltage / thermal_voltage, mobility * thermal_voltage
        )
        * AREA_OVER_LENGTH
    )
    current = Q * number_flux
    # Finite SG roundoff bound; equal/opposite terminal scattering closes KCL.
    roundoff = 64 * jnp.finfo(current.dtype).eps * jnp.abs(current)
    return SemiconductorForwardResult(
        current,
        jnp.all(jnp.isfinite(current)),
        roundoff,
        jnp.asarray(0, dtype=jnp.int32),
        "native-SG-homogeneous-resistor",
    )


def _case(
    name,
    voltage=(0.1, 0.3, 0.5),
    *,
    die=None,
    rho=0.0,
    deembed=0.0,
    control_error=0.0,
    observed_shift=0.0,
    training_right=True,
    source="synthetic",
):
    controls = _controls(voltage)
    count = controls.shape[0]
    sigma = 2e-7
    measured = sigma**2 * ((1 - rho) * jnp.eye(count) + rho * jnp.ones((count, count)))
    xcov = jnp.zeros((controls.size, controls.size))
    indices = jnp.arange(count) * 3
    xcov = xcov.at[indices[:, None], indices[None, :]].set(control_error**2)
    return SemiconductorMeasurementCase(
        case_id=name,
        observation_ids=tuple(f"{name}-point-{i}" for i in range(count)),
        process_revision="synthetic-homogeneous-r1",
        geometry_id="resistor-A-over-L-1e-6m",
        lot="synthetic-lot",
        wafer="wafer-1",
        die=name if die is None else die,
        structure="long-resistor",
        condition_group=f"condition-{name}",
        correlation_group=f"noise-{name}",
        source_kind=source,
        source_uri=f"memory://synthetic/{name}",
        instrument_record="analytic-exact-current-no-instrument",
        deembedding_record="explicit-additive-test-offset",
        uncertainty_assumptions="independent measurement, additive deembedding and control errors; cases independent",
        terminal_definitions=(
            "left: positive current into device",
            "right: equal opposite current",
        ),
        observables=tuple(
            _quantity(f"current-{i}", "current", AMPERE) for i in range(count)
        ),
        reference=_reference(name, training=training_right),
        controls=controls,
        observed=Q * DENSITY * 0.1 * AREA_OVER_LENGTH * controls[:, 0] + observed_shift,
        control_scale=jnp.asarray([1.0, 300.0, DENSITY]),
        observation_scale=jnp.full((count,), sigma),
        control_covariance=xcov,
        measurement_covariance=measured,
        deembedding_covariance=deembed**2 * jnp.ones((count, count)),
    )


def _campaign(training=None, heldout=None, *, axes=("die",)):
    training = _case("training") if training is None else training
    heldout = _case("heldout", (0.2, 0.4, 0.6)) if heldout is None else heldout
    domain = SemiconductorCalibrationDomain(
        "synthetic-homogeneous-r1",
        "synthetic independent dies",
        ("resistor-A-over-L-1e-6m",),
        (
            _quantity("bias", "voltage", VOLT),
            _quantity("temperature", "temperature", KELVIN),
            _quantity("density", "number_density", PER_CUBIC_METER),
        ),
        [-1.0, 250.0, 1e20],
        [1.0, 400.0, 1e22],
    )
    return SemiconductorCalibrationCampaign(
        campaign_name="synthetic-SG-inference",
        model_id="homogeneous-SG-resistor",
        source_revision="synthetic-test-source",
        numerical_configuration="exact-fixed-SG-geometry-float64",
        split_lock_record="predeclared-die-holdout-before-fit",
        training_ids=(training.case_id,),
        heldout_ids=(heldout.case_id,),
        holdout_axes=axes,
        domain=domain,
        cases=(training, heldout),
        criteria=SemiconductorCalibrationCriteria(2.0, 0.9, 2.0, 0.1, 1e-7, 1),
    )


def _binding(*, two=False, prior_sigma=0.02):
    initial = jnp.asarray([0.04, 0.06] if two else [0.1])
    quantities = tuple(
        _quantity(f"mobility-{i}", "mobility", MOBILITY_UNIT) for i in range(initial.size)
    )

    def log_prior(p):
        return -0.5 * jnp.sum(
            ((p - initial) / prior_sigma) ** 2
            + 2 * jnp.log(prior_sigma)
            + jnp.log(2 * jnp.pi)
        )

    return SemiconductorParameterBinding(
        ParameterSpace(initial, log_prior=log_prior),
        quantities,
        jnp.full(initial.shape, 0.1),
        prior_record=f"synthetic-Gaussian-mobility-prior-sigma-{prior_sigma}",
        admissible=lambda p: jnp.all(p > 0),
    )


@pytest.mark.parametrize("axis", ["lot", "wafer", "die", "structure", "condition"])
def test_complete_group_holdout_cannot_be_replaced_by_random_curve_points(axis):
    train = _case("train")
    heldout = _case("renamed-heldout", (0.2, 0.4, 0.6), die="train")
    if axis == "condition":
        heldout = eqx.tree_at(lambda c: c.controls, heldout, train.controls)
        # Fresh observations may still be part of the same physical condition.
        values = dict(
            case_id=heldout.case_id,
            observation_ids=heldout.observation_ids,
            process_revision=heldout.process_revision,
            geometry_id=heldout.geometry_id,
            lot=heldout.lot,
            wafer=heldout.wafer,
            die=heldout.die,
            structure=heldout.structure,
            condition_group=train.condition_group,
            correlation_group=heldout.correlation_group,
            source_kind=heldout.source_kind,
            source_uri=heldout.source_uri,
            instrument_record=heldout.instrument_record,
            deembedding_record=heldout.deembedding_record,
            uncertainty_assumptions=heldout.uncertainty_assumptions,
            terminal_definitions=heldout.terminal_definitions,
            observables=heldout.observables,
            reference=heldout.reference,
            **heldout.arrays(),
        )
        heldout = SemiconductorMeasurementCase(**values)
    with pytest.raises(ValueError, match="leakage"):
        _campaign(train, heldout, axes=(axis,))


def test_correlated_measurement_deembedding_and_control_errors_enter_joint_likelihood():
    train = _case("training", (0.2, 0.2), rho=0.5, deembed=1e-7, control_error=1e-3)
    campaign = _campaign(train)
    prepared = prepare_semiconductor_calibration(campaign, _binding(), _current)
    physical = jnp.asarray([0.11])
    # Independent 2x2 formula, including normalization and input derivative;
    # no dense inverse is used by either implementation or reference.
    slope = Q * DENSITY * physical[0] * AREA_OVER_LENGTH
    covariance = (
        train.measurement_covariance + train.deembedding_covariance + slope**2 * 1e-6
    )
    covariance = covariance / (
        train.observation_scale[:, None] * train.observation_scale[None, :]
    )
    residual = (
        _current(physical, train).prediction - train.observed
    ) / train.observation_scale
    a, b, d = covariance[0, 0], covariance[0, 1], covariance[1, 1]
    determinant = a * d - b**2
    quadratic = (
        d * residual[0] ** 2 - 2 * b * residual[0] * residual[1] + a * residual[1] ** 2
    ) / determinant
    expected = -0.5 * (quadratic + jnp.log(determinant) + 2 * jnp.log(2 * jnp.pi))
    np.testing.assert_allclose(
        prepared.posterior.log_likelihood(physical), expected, rtol=1e-10
    )
    # Independent-point scoring overstates information in a common-mode shift.
    independent = -0.5 * jnp.sum(
        residual**2 / jnp.diag(covariance)
        + jnp.log(jnp.diag(covariance))
        + jnp.log(2 * jnp.pi)
    )
    assert abs(float(expected - independent)) > 0.1


def test_heldout_values_never_change_training_posterior():
    original = prepare_semiconductor_calibration(_campaign(), _binding(), _current)
    shifted = prepare_semiconductor_calibration(
        _campaign(heldout=_case("heldout", (0.2, 0.4, 0.6), observed_shift=1e-3)),
        _binding(),
        _current,
    )
    point = jnp.asarray([0.11])
    np.testing.assert_allclose(
        original.posterior.log_density(point), shifted.posterior.log_density(point)
    )
    np.testing.assert_allclose(
        jax.grad(original.posterior.log_density)(point),
        jax.grad(shifted.posterior.log_density)(point),
    )


def _unresolved_diode(parameters, case):
    """Actual series-resistor/diode root truncated before convergence."""
    voltage = case.controls[:, 0]
    vt = KB * case.controls[:, 1] / Q
    saturation_current = Q * DENSITY * jnp.sum(parameters) * AREA_OVER_LENGTH * vt
    resistance = 1e3

    def residual(junction_voltage, args):
        del args
        return (
            junction_voltage
            + resistance * saturation_current * jnp.expm1(junction_voltage / vt)
            - voltage
        )

    result = root(
        NonlinearSystemProblem(residual),
        jnp.zeros_like(voltage),
        method=NewtonKrylov(linear_policy=LinearSolvePolicy(DenseLU())),
        termination=NonlinearTermination(
            maximum_steps=1, absolute_residual=1e-14, relative_residual=0.0
        ),
    )
    current = (voltage - result.state) / resistance
    bound = jnp.abs(residual(result.state, None)) / resistance
    return SemiconductorForwardResult(
        current,
        result.successful,
        bound,
        result.status,
        "native-series-diode-one-newton-step",
    )


def test_unresolved_native_forward_preserves_rejected_evidence_and_stops_inference(
    tmp_path,
):
    campaign, binding = _campaign(), _binding()
    with pytest.raises(SemiconductorForwardUnresolved) as caught:
        prepare_semiconductor_calibration(campaign, binding, _unresolved_diode)
    evidence = caught.value.evaluation
    assert evidence.status == "numerically-unresolved"
    assert not bool(evidence.cases[0].numerical_valid)
    assert bool(jnp.any(evidence.cases[0].numerical_error > 0))
    with pytest.raises(RuntimeError, match="No qualified predictions"):
        _ = evidence.predictions
    path, envelope = archive_semiconductor_forward_evaluation(
        tmp_path / "rejected.pdx",
        campaign,
        evidence,
        producer_version="synthetic-test",
        license_id="local-test-only",
    )
    manifest, arrays = read_array_archive(path)
    assert envelope.status == "failed"
    assert manifest["status"] == "numerically-unresolved"
    np.testing.assert_allclose(
        arrays["training/raw_candidate"], evidence.cases[0].prediction
    )


def test_an_unresolved_likelihood_point_is_not_given_zero_probability():
    initial = jnp.asarray([1e-25])
    binding = SemiconductorParameterBinding(
        ParameterSpace(initial, log_prior=lambda p: -0.5 * jnp.sum((p / 0.1) ** 2)),
        (_quantity("mobility", "mobility", MOBILITY_UNIT),),
        jnp.asarray([0.1]),
        prior_record="synthetic-positive-half-normal-mobility",
        admissible=lambda p: jnp.all(p > 0),
    )
    prepared = prepare_semiconductor_calibration(_campaign(), binding, _unresolved_diode)
    # Vanishing saturation current makes the initial series-resistor solve
    # linear to its actual native tolerance. A conducting diode needs further
    # Newton steps: both physical points are admitted by the same physical prior.
    assert bool(binding.physically_admissible(jnp.asarray([0.1])))
    with pytest.raises(
        RuntimeError, match="Numerically unresolved semiconductor forward"
    ):
        prepared.posterior.log_likelihood(jnp.asarray([0.1])).block_until_ready()


def test_physical_prior_exclusion_is_not_numerical_failure():
    prepared = prepare_semiconductor_calibration(_campaign(), _binding(), _current)
    inadmissible = evaluate_semiconductor_forward(prepared, jnp.asarray([-0.1]))
    assert inadmissible.status == "physical-inadmissible"
    assert not inadmissible.cases
    assert float(prepared.posterior.log_likelihood(jnp.asarray([-0.1]))) == -math.inf


def test_prior_curvature_cannot_identify_two_indistinguishable_mobilities():
    campaign = _campaign()
    weak = prepare_semiconductor_calibration(
        campaign, _binding(two=True, prior_sigma=0.2), _current
    )
    strong = prepare_semiconductor_calibration(
        campaign, _binding(two=True, prior_sigma=0.002), _current
    )
    point = jnp.asarray([0.04, 0.06])
    weak_rank = semiconductor_identifiability(weak, point)
    strong_rank = semiconductor_identifiability(strong, point)
    assert weak_rank.rank == strong_rank.rank == 1
    assert not strong_rank.locally_identifiable
    np.testing.assert_allclose(weak_rank.singular_values, strong_rank.singular_values)
    # A proper prior still yields finite native local posterior uncertainty.
    result = calibrate_semiconductor(strong)
    assert np.all(np.isfinite(np.asarray(result.posterior.covariance)))
    qualification = qualify_semiconductor_calibration(result, (), at_time=100)
    assert not qualification.empirically_calibrated
    assert any("rank deficient" in reason for reason in qualification.empirical_gates)


def test_native_fit_heldout_and_archive_do_not_turn_synthetic_recovery_into_foundry_evidence(
    tmp_path,
):
    prepared = prepare_semiconductor_calibration(_campaign(), _binding(), _current)
    result = calibrate_semiconductor(prepared)
    np.testing.assert_allclose(result.optimization.parameters, [0.1], rtol=1e-8)
    np.testing.assert_allclose(
        result.heldout[0].prediction, prepared.campaign.heldout[0].observed, rtol=1e-9
    )
    assert result.identifiability.locally_identifiable
    evidence = tuple(
        QualificationEvidence(
            "scientific",
            "passed",
            (prepared.campaign.model_id,),
            build_id=prepared.campaign.source_revision,
            environment_id="synthetic-regression",
            backend="jax-cpu",
            topology="homogeneous-edge",
            precision="float64",
            reduction="SG-ohmic-limit",
            replay_id=prepared.campaign.numerical_configuration,
            criteria_ids=(criterion,),
            raw_artifact_ids=(f"synthetic-{criterion}",),
            reviewer_id="synthetic-test-reviewer",
            issued_at=0,
            expires_at=200,
            reason="Synthetic analytic invariant fixture, not empirical evidence.",
        )
        for criterion in (
            "semiconductor-conservation",
            "semiconductor-refinement",
            "semiconductor-solver",
        )
    )
    qualification = qualify_semiconductor_calibration(result, evidence, at_time=100)
    assert qualification.numerically_qualified
    assert not qualification.empirically_calibrated
    assert any("experimental" in reason for reason in qualification.empirical_gates)
    path, envelope = archive_semiconductor_calibration(
        tmp_path / "synthetic.pdx",
        result,
        qualification,
        producer_version="synthetic-test",
        license_id="local-test-only",
    )
    manifest, arrays = read_array_archive(path)
    assert manifest["artifact_id"] == envelope.artifact_id
    assert not manifest["empirically_calibrated"]
    np.testing.assert_allclose(
        arrays["physical_parameters"], result.optimization.parameters
    )
    with pytest.raises(PermissionError, match="redistribution"):
        archive_semiconductor_calibration(
            tmp_path / "unauthorized.pdx",
            result,
            qualification,
            producer_version="synthetic-test",
            license_id="local-test-only",
            redistribution=True,
        )


def test_measurement_training_rights_are_checked_before_forward_use():
    campaign = _campaign(training=_case("training", training_right=False))
    with pytest.raises(PermissionError, match="training-use"):
        prepare_semiconductor_calibration(campaign, _binding(), _current)
