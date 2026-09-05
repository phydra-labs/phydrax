# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.protein_folding.experiments import (
    ChevronKinetics,
    DimerTwoStateUnfolding,
    ExperimentConditions,
    ExperimentParameter,
    fit_protein_experiments,
    FluorescenceExperiment,
    KineticRateExperiment,
    phi_posterior,
    prepare_protein_experiments,
    protein_experiment_identifiability,
    protein_experiment_posterior_problem,
    RepeatTransferUnfolding,
    ThermodynamicConvention,
    TwoStateUnfolding,
)
from phydrax.optim import OptimizationTermination
from phydrax.units import JOULE, MILLISECOND


def _population(t, d):
    # Independently specified synthetic reference (kJ/mol, concentration mol/m^3).
    g = (
        180 * (1 - t / 298.15)
        + 15 * t / 298.15
        + 1.2 * (t - 298.15 - t * np.log(t / 298.15))
        - 0.004 * d
    )
    return 1 / (1 + np.exp(-g / (0.00831446261815324 * t)))


def _joint_problem():
    t, d = np.meshgrid(np.linspace(290, 355, 8), np.linspace(0, 6000, 9), indexing="ij")
    t, d = np.repeat(t.ravel(), 2), np.repeat(d.ravel(), 2)
    group = tuple(
        "channel-a/replicate-1" if i % 2 == 0 else "channel-b/replicate-1"
        for i in range(t.size)
    )
    p = _population(t, d)
    observed = np.where(
        np.arange(t.size) % 2 == 0, 1.2 * p + 0.2 * (1 - p), 0.3 * p + 1.4 * (1 - p)
    )
    model = TwoStateUnfolding()
    plan = FluorescenceExperiment(
        "joint",
        model,
        ExperimentConditions(t, d),
        group,
        observed,
        np.full(t.size, 0.02),
        "synthetic:thermal-denaturant",
        "Synthetic reversible equilibrium at every condition",
        True,
        baseline_terms=("intercept",),
    )
    values = [13.0, 170.0, 1.2, 0.0035, 0.0, 1.1, 0.25, 0.35, 1.3]
    scales = [10.0, 100.0, 1.0, 0.004, 1e-5, 1.0, 1.0, 1.0, 1.0]
    parameters = tuple(
        ExperimentParameter(name, value, unit, scale, free=i not in (2, 4))
        for i, ((name, unit), value, scale) in enumerate(
            zip(plan.parameter_slots(), values, scales, strict=True)
        )
    )
    return prepare_protein_experiments((plan,), parameters), plan, parameters


def test_joint_multichannel_fit_predicts_unseen_conditions_and_reports_covariance():
    problem, _, _ = _joint_problem()
    initial = float(jnp.sum(problem.residual(problem.initial_coordinates) ** 2))
    fit = fit_protein_experiments(
        problem, termination=OptimizationTermination(maximum_steps=100)
    )
    assert bool(fit.optimization.successful)
    assert float(jnp.sum(problem.residual(fit.coordinates) ** 2)) < initial * 1e-8
    assert fit.identifiability.locally_identifiable
    assert fit.covariance is not None
    assert np.all(np.linalg.eigvalsh(np.asarray(fit.covariance)) > 0)
    t = np.repeat(np.linspace(293.0, 352.0, 15), 2)
    d = np.repeat(np.linspace(250.0, 5700.0, 15), 2)
    groups = ("channel-a/replicate-1", "channel-b/replicate-1") * 15
    prediction = problem.observations[0].prepare_prediction(
        ExperimentConditions(t, d), groups=groups
    )
    values = problem.parameters.decode(fit.coordinates)
    actual = jax.jit(lambda parameters: prediction(parameters))(values)
    p = _population(t, d)
    expected = np.where(
        np.arange(t.size) % 2 == 0, 1.2 * p + 0.2 * (1 - p), 0.3 * p + 1.4 * (1 - p)
    )
    np.testing.assert_allclose(actual, expected, atol=2e-5)
    with pytest.raises(ValueError):
        problem.observations[0].prepare_prediction(
            ExperimentConditions(t, d), groups=("new-channel",) * t.size
        )


def test_single_isotherm_does_not_identify_enthalpy_or_manufacture_covariance():
    model = TwoStateUnfolding()
    d = np.linspace(0, 6000, 21)
    t = np.full_like(d, 298.15)
    plan = FluorescenceExperiment(
        "isothermal",
        model,
        ExperimentConditions(t, d),
        ("a",) * d.size,
        _population(t, d),
        np.full_like(d, 0.02),
        "synthetic:isothermal",
        "Exact equilibria",
        True,
        baseline_terms=("intercept",),
    )
    parameters = tuple(
        ExperimentParameter(name, value, unit, scale, free=i in (0, 1))
        for i, ((name, unit), value, scale) in enumerate(
            zip(
                plan.parameter_slots(),
                [13, 120, 1.2, 0.004, 0, 1, 0],
                [10, 100, 1, 0.004, 1e-5, 1, 1],
                strict=True,
            )
        )
    )
    problem = prepare_protein_experiments((plan,), parameters)
    evidence = protein_experiment_identifiability(problem)
    assert evidence.rank == 1
    assert not evidence.locally_identifiable
    np.testing.assert_allclose(
        evidence.sensitivity @ evidence.null_vectors.T, 0, atol=1e-12
    )
    fit = fit_protein_experiments(problem)
    assert fit.covariance is None
    np.testing.assert_allclose(fit.predict()[0], plan.observed, atol=2e-5)


def test_active_mask_correlated_noise_and_explicit_prior_density():
    model = TwoStateUnfolding()
    conditions = ExperimentConditions([298.15] * 3, [0, 2500, 5000])
    root = np.array([[0.2, 0], [0.06, 0.19]])
    error = np.array([0.2, np.nan, np.sqrt(0.06**2 + 0.19**2)])
    plan = FluorescenceExperiment(
        "masked",
        model,
        conditions,
        ("a",) * 3,
        [0.9, np.nan, 0.3],
        error,
        "synthetic:masked",
        "Reversible",
        True,
        baseline_terms=("intercept",),
        mask=[True, False, True],
        covariance_cholesky=root,
    )
    parameters = tuple(
        ExperimentParameter(name, value, unit, 1.0, free=i == 0)
        for i, ((name, unit), value) in enumerate(
            zip(plan.parameter_slots(), [12, 180, 1.2, 0.004, 0, 1, 0], strict=True)
        )
    )
    problem = prepare_protein_experiments((plan,), parameters)
    z = jnp.array([0.7])
    residual = np.asarray(problem.predict(z)[0])[[0, 2]] - np.array([0.9, 0.3])
    whitened = np.linalg.solve(root, residual)
    expected_logp = -0.5 * (
        whitened @ whitened + 2 * np.log(np.diag(root)).sum() + 2 * np.log(2 * np.pi)
    )
    np.testing.assert_allclose(
        jax.jit(lambda coordinates: problem.log_likelihood(coordinates))(z), expected_logp
    )
    assert np.isfinite(np.asarray(jax.grad(problem.log_likelihood)(z))).all()
    posterior = protein_experiment_posterior_problem(
        problem, prior_mean=[0], prior_standard_deviation=[2]
    )
    expected_prior = -0.5 * ((0.7 / 2) ** 2 + 2 * np.log(2) + np.log(2 * np.pi))
    np.testing.assert_allclose(posterior.log_density(z), expected_logp + expected_prior)
    with pytest.raises(ValueError):
        prepare_protein_experiments(
            (replace(plan, standard_errors=[0.2, np.nan, 0.8]),), parameters
        )


def test_real_kinetic_fit_and_time_conversion_predict_unseen_denaturant():
    model = ChevronKinetics()
    d = np.linspace(0, 7000, 31)
    kt = model.convention.thermal_constant * 298.15
    true = np.array([3.0, -2.0, 0.001, 0.002])
    observed_seconds = np.logaddexp(
        true[0] - true[2] * d / kt, true[1] + true[3] * d / kt
    )
    plan = KineticRateExperiment(
        "rates",
        model,
        ExperimentConditions(np.full_like(d, 298.15), d),
        observed_seconds + np.log(0.001),
        np.full_like(d, 0.03),
        "synthetic:chevron",
        time_unit=MILLISECOND,
    )
    parameters = tuple(
        ExperimentParameter(name, value, unit, scale, free=i < 2)
        for i, ((name, unit), value, scale) in enumerate(
            zip(
                plan.parameter_slots(),
                [2.5, -1.7, 0.001, 0.002],
                [1, 1, 0.001, 0.001],
                strict=True,
            )
        )
    )
    problem = prepare_protein_experiments((plan,), parameters)
    fit = fit_protein_experiments(problem)
    assert bool(fit.optimization.successful)
    heldout = np.linspace(150, 6500, 13)
    predict = problem.observations[0].prepare_prediction(
        ExperimentConditions(np.full_like(heldout, 298.15), heldout)
    )
    actual = predict(problem.parameters.decode(fit.coordinates))
    expected = np.logaddexp(
        true[0] - true[2] * heldout / kt, true[1] + true[3] * heldout / kt
    )
    np.testing.assert_allclose(actual, expected, atol=2e-5)
    with pytest.raises(ValueError):
        prepare_protein_experiments(
            (replace(plan, conditions=ExperimentConditions(np.full_like(d, 310), d)),),
            parameters,
        )


def test_preparation_refuses_wrong_basis_zero_dimer_and_irreversible_data():
    _, plan, parameters = _joint_problem()
    with pytest.raises(ValueError):
        prepare_protein_experiments((replace(plan, reversible=False),), parameters)
    with pytest.raises(ValueError):
        prepare_protein_experiments(
            (replace(plan, model=DimerTwoStateUnfolding()),), parameters
        )
    with pytest.raises(ValueError):
        prepare_protein_experiments(
            (plan,), (replace(parameters[0], unit=JOULE),) + parameters[1:]
        )
    with pytest.raises(ValueError):
        prepare_protein_experiments(
            (replace(plan, source_kind="experimental"),), parameters
        )


def test_phi_preserves_paired_draws_and_marks_unresolved_denominators():
    convention = ThermodynamicConvention()
    kt = convention.thermal_constant * convention.reference_temperature
    wt = np.array([[8.0, 9.0, 10.0], [8.5, 9.5, 10.5]])
    loss = np.array([[2.0, 3.0, 4.0], [2.5, 3.5, 4.5]])
    samples = {
        "wt": wt,
        "mutant": wt - loss,
        "log_wt": np.zeros_like(wt),
        "log_mutant": -1.2 * loss / kt,
    }
    kwargs = dict(
        wild_type_stability="wt",
        mutant_stability="mutant",
        wild_type_log_folding_rate="log_wt",
        mutant_log_folding_rate="log_mutant",
        convention=convention,
        source_id="synthetic:paired-posterior-derivation",
        minimum_stability_change=0.1,
    )
    result = phi_posterior(samples, **kwargs)
    np.testing.assert_allclose(result.samples, 1.2)
    np.testing.assert_allclose(result.credible_interval, [1.2, 1.2])
    samples["mutant"] = samples["mutant"].copy()
    samples["mutant"][0, 0] = wt[0, 0]
    result = phi_posterior(samples, **kwargs)
    assert not bool(result.valid[0, 0])
    assert np.isnan(np.asarray(result.credible_interval)).all()
    assert result.valid_fraction == 5 / 6


def test_repeat_fit_uses_explicit_shared_intrinsic_energy_without_duplicate_degrees_of_freedom():
    model = RepeatTransferUnfolding(2)
    kt = model.convention.thermal_constant * 298.15

    def exact_mean(d):
        # States 00, 10, 01, 11 with one favorable -2 kJ/mol interface.
        g = 6.0 - 0.002 * np.asarray(d)
        weights = np.stack(
            (np.ones_like(g), np.exp(g / kt), np.exp(g / kt), np.exp((2 * g + 2) / kt)),
            axis=-1,
        )
        return (weights * np.array([0, 0.5, 0.5, 1])).sum(axis=-1) / weights.sum(axis=-1)

    d = np.linspace(0, 6000, 31)
    bindings = {
        f"repeat.1.{suffix}": f"repeat.0.{suffix}"
        for suffix in ("dg_ref", "dh_ref", "dcp", "m_ref", "dm_dt")
    }
    plan = FluorescenceExperiment(
        "repeats",
        model,
        ExperimentConditions(np.full_like(d, 298.15), d),
        ("a",) * d.size,
        exact_mean(d),
        np.full_like(d, 0.02),
        "synthetic:two-repeat-enumeration",
        "Exact equilibrium enumeration",
        True,
        baseline_terms=("intercept",),
        bindings=bindings,
    )
    unique_slots = tuple(
        slot for slot in plan.parameter_slots() if slot[0] not in bindings
    )
    parameters = tuple(
        ExperimentParameter(name, value, unit, scale, free=i == 0)
        for i, ((name, unit), value, scale) in enumerate(
            zip(
                unique_slots,
                [4, 0, 0, 0.002, 0, -2, 1, 0],
                [5, 1, 1, 0.002, 1e-5, 1, 1, 1],
                strict=True,
            )
        )
    )
    problem = prepare_protein_experiments((plan,), parameters)
    fit = fit_protein_experiments(problem)
    assert bool(fit.optimization.successful)
    heldout = np.linspace(125, 5875, 15)
    predict = problem.observations[0].prepare_prediction(
        ExperimentConditions(np.full_like(heldout, 298.15), heldout),
        groups=("a",) * heldout.size,
    )
    np.testing.assert_allclose(
        predict(problem.parameters.decode(fit.coordinates)),
        exact_mean(heldout),
        atol=2e-5,
    )
