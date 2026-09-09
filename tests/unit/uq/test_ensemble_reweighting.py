#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.observation import (
    CholeskyCovarianceAction,
    CoordinateLayout,
    PrecisionCovarianceAction,
)
from phydrax.units import derived_unit, SECOND, UnitDefinition
from phydrax.uq._ensemble_reweighting import (
    EnsembleSupportPolicy,
    PhysicalEquilibriumSupportProvenance,
    TwoStateKineticRecord,
)


_PER_SECOND = derived_unit("test-s^-1", ((SECOND, -1),))
_MINUTE = UnitDefinition("test-minute", SECOND.dimension, SECOND.reference_system_id, 60)
_PER_MINUTE = derived_unit("test-min^-1", ((_MINUTE, -1),))
_SUPPORT_POLICY = EnsembleSupportPolicy(1.1, 0.01)


def _physical_provenance(source_id, *, converged=True, authorized=True):
    return PhysicalEquilibriumSupportProvenance(
        source_id,
        f"manifest:{source_id}",
        "replica-exchange-equilibrium-v1",
        "condition-25-celsius",
        ("unbound", "bound"),
        f"convergence:{source_id}",
        converged=converged,
        rights_manifest_id=f"rights:{source_id}",
        authorized_use_ids=(
            ("ensemble-reweighting",) if authorized else ("visualization",)
        ),
    )


def _cholesky_plan(
    per_sample,
    observed,
    standard_deviation,
    observation_id,
    *,
    usage="calibration",
    source_ids=None,
    case_ids=None,
    parent_ids=None,
):
    observed_array = np.atleast_1d(np.asarray(observed, dtype=float))
    standard_deviation_array = np.broadcast_to(
        np.asarray(standard_deviation, dtype=float), observed_array.shape
    )
    layout = CoordinateLayout(
        tuple(f"observable-{index}" for index in range(observed_array.size))
    )
    covariance = CholeskyCovarianceAction(np.diag(standard_deviation_array), layout)
    sources = (f"source:{observation_id}",) if source_ids is None else source_ids
    cases = (f"case:{observation_id}",) if case_ids is None else case_ids
    parents = (f"parent:{observation_id}",) if parent_ids is None else parent_ids
    return phx.uq.EnsembleObservablePlan(
        per_sample,
        observed_array,
        covariance,
        observation_id,
        source_ids=sources,
        case_ids=cases,
        parent_ids=parents,
        usage=usage,
    )


def _precision_plan(
    per_sample,
    observed,
    standard_deviation,
    observation_id,
    *,
    source_ids=None,
    case_ids=None,
    parent_ids=None,
):
    observed_array = np.atleast_1d(np.asarray(observed, dtype=float))
    standard_deviation_array = np.broadcast_to(
        np.asarray(standard_deviation, dtype=float), observed_array.shape
    )
    layout = CoordinateLayout(
        tuple(f"observable-{index}" for index in range(observed_array.size))
    )
    covariance = np.diag(standard_deviation_array**2)
    precision = np.diag(standard_deviation_array**-2)
    action = PrecisionCovarianceAction(
        precision, np.linalg.slogdet(covariance)[1], layout
    )
    sources = (f"source:{observation_id}",) if source_ids is None else source_ids
    cases = (f"case:{observation_id}",) if case_ids is None else case_ids
    parents = (f"parent:{observation_id}",) if parent_ids is None else parent_ids
    return phx.uq.EnsembleObservablePlan(
        per_sample,
        observed_array,
        action,
        observation_id,
        source_ids=sources,
        case_ids=cases,
        parent_ids=parents,
    )


def test_two_state_bme_solution_and_reference_normalization_are_known():
    theta = 0.5
    sigma = 0.25
    expected_excited_population = 0.8
    observed = expected_excited_population + theta * sigma**2 * np.log(4.0)
    support = phx.uq.EnsembleSupport([23.0, 23.0], "empirical", "two-state-collection")
    plan = _cholesky_plan([[0.0], [1.0]], [observed], sigma, "calibration-population")

    result = phx.uq.reweight_ensemble(
        support, plan, regularization=theta, support_policy=_SUPPORT_POLICY
    )

    weights = jnp.exp(result.log_weights)
    np.testing.assert_allclose(jnp.sum(weights), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(weights, [0.2, 0.8], atol=2.0e-8)
    assert bool(result.optimization_evidence.successful)
    assert bool(result.support_valid)
    assert not bool(result.physical_equilibrium_valid)


def test_physical_support_requires_typed_converged_authorized_provenance():
    with pytest.raises(TypeError, match="typed physical provenance"):
        phx.uq.EnsembleSupport(
            np.zeros(3), "physical-equilibrium", "renamed-empirical-source"
        )

    unauthorized = phx.uq.EnsembleSupport(
        np.zeros(3),
        "physical-equilibrium",
        _physical_provenance("unauthorized-equilibrium", authorized=False),
    )
    calibration = _cholesky_plan([[0.0], [1.0], [2.0]], [1.0], 0.4, "physical-fit")
    result = phx.uq.reweight_ensemble(
        unauthorized, calibration, regularization=0.5, support_policy=_SUPPORT_POLICY
    )

    assert not unauthorized.physical_equilibrium
    assert not bool(result.physical_equilibrium_valid)
    with pytest.raises(TypeError, match="cannot label empirical"):
        phx.uq.EnsembleSupport(
            np.zeros(3),
            "empirical",
            _physical_provenance("misapplied-physical-record"),
        )


def test_reweighting_is_translation_invariant_and_covariance_whitened():
    support = phx.uq.EnsembleSupport(
        np.log([0.2, 0.5, 0.3]), "empirical", "three-state-collection"
    )
    samples = np.asarray([[0.0], [1.0], [2.0]])
    observed = np.asarray([1.35])
    base = _cholesky_plan(samples, observed, 0.3, "base-calibration")
    translated = _cholesky_plan(
        samples + 50.0, observed + 50.0, 0.3, "translated-calibration"
    )
    scale = 7.0
    scaled_precision = _precision_plan(
        scale * samples,
        scale * observed,
        scale * 0.3,
        "scaled-calibration",
    )

    base_result = phx.uq.reweight_ensemble(
        support, base, regularization=0.4, support_policy=_SUPPORT_POLICY
    )
    translated_result = phx.uq.reweight_ensemble(
        support, translated, regularization=0.4, support_policy=_SUPPORT_POLICY
    )
    precision_result = phx.uq.reweight_ensemble(
        support, scaled_precision, regularization=0.4, support_policy=_SUPPORT_POLICY
    )

    np.testing.assert_allclose(
        base_result.log_weights, translated_result.log_weights, atol=2.0e-10
    )
    np.testing.assert_allclose(
        base_result.log_weights, precision_result.log_weights, atol=2.0e-10
    )


def test_uniform_observable_cannot_change_weights():
    support = phx.uq.EnsembleSupport(
        np.log([0.1, 0.2, 0.7]), "empirical", "uniform-observable-support"
    )
    plan = _cholesky_plan([[4.0], [4.0], [4.0]], [4.0], 0.1, "uniform-observable")

    result = phx.uq.reweight_ensemble(
        support, plan, regularization=0.01, support_policy=_SUPPORT_POLICY
    )

    np.testing.assert_allclose(
        result.log_weights, support.log_reference_weights, atol=1.0e-12
    )
    assert bool(result.convex_support.within_support)


def test_effective_support_policy_is_explicit_fractional_and_inclusive():
    support = phx.uq.EnsembleSupport(np.zeros(3), "empirical", "ess-boundary-support")
    plan = _cholesky_plan([[1.0], [1.0], [1.0]], [1.0], 0.2, "ess-boundary-fit")

    without_policy = phx.uq.reweight_ensemble(support, plan, regularization=0.5)
    boundary = phx.uq.reweight_ensemble(
        support,
        plan,
        regularization=0.5,
        support_policy=EnsembleSupportPolicy(3.0, 0.01),
    )

    assert not bool(without_policy.support_policy_valid)
    assert not bool(without_policy.support_valid)
    assert bool(boundary.support_collapsed)
    assert not bool(boundary.support_valid)
    with pytest.raises(ValueError, match="must exceed one"):
        EnsembleSupportPolicy(1.0, 0.01)


def test_effective_fraction_rejects_large_support_with_only_two_effective_samples():
    support = phx.uq.EnsembleSupport(
        np.zeros(100),
        "empirical",
        "large-correlated-support",
        statistical_inefficiency=50.0,
    )
    plan = _cholesky_plan(np.ones((100, 1)), [1.0], 0.2, "large-support-fit")

    result = phx.uq.reweight_ensemble(
        support,
        plan,
        regularization=0.5,
        support_policy=EnsembleSupportPolicy(1.5, 0.03),
    )

    np.testing.assert_allclose(result.effective_sample_size, 2.0)
    np.testing.assert_allclose(result.effective_sample_fraction, 0.02)
    assert bool(result.support_collapsed)
    assert not bool(result.support_valid)


def test_support_collapse_and_out_of_convex_support_are_explicit():
    support = phx.uq.EnsembleSupport(
        np.zeros(5),
        "physical-equilibrium",
        _physical_provenance("correlated-reference"),
        statistical_inefficiency=2.0,
    )
    samples = np.arange(5.0)[:, None]
    collapse_plan = _cholesky_plan(samples, [3.999], 0.02, "near-boundary-calibration")
    outside_plan = _cholesky_plan(samples, [6.0], 0.1, "outside-support-calibration")

    collapsed = phx.uq.reweight_ensemble(
        support,
        collapse_plan,
        regularization=1.0e-5,
        support_policy=EnsembleSupportPolicy(2.0, 0.01),
    )
    outside = phx.uq.reweight_ensemble(
        support, outside_plan, regularization=0.1, support_policy=_SUPPORT_POLICY
    )

    assert bool(collapsed.support_collapsed)
    np.testing.assert_allclose(
        collapsed.effective_sample_size,
        collapsed.importance_effective_sample_size / 2.0,
    )
    assert not bool(collapsed.support_valid)
    assert not bool(outside.convex_support.within_support)
    assert not bool(outside.support_valid)
    assert bool(jnp.all(jnp.isfinite(outside.log_weights)))
    np.testing.assert_allclose(jnp.sum(jnp.exp(outside.log_weights)), 1.0)


def test_held_out_nonlinear_observable_is_averaged_per_conformation():
    support = phx.uq.EnsembleSupport(
        np.zeros(3), "empirical", "nonlinear-observable-support"
    )
    calibration = _cholesky_plan(
        [[0.0], [1.0], [2.0]], [1.25], 0.25, "linear-calibration"
    )
    result = phx.uq.reweight_ensemble(
        support, calibration, regularization=0.3, support_policy=_SUPPORT_POLICY
    )
    held_out = _cholesky_plan(
        [[0.0], [1.0], [4.0]], [2.0], 0.5, "squared-held-out", usage="held-out"
    )

    prediction = phx.uq.predict_held_out_observables(result, held_out)
    weights = jnp.exp(result.log_weights)

    np.testing.assert_allclose(
        prediction.predicted_observables, weights @ jnp.asarray([[0.0], [1.0], [4.0]])
    )
    assert not jnp.isclose(
        prediction.predicted_observables[0], result.predicted_observables[0] ** 2
    )
    with pytest.raises(ValueError, match="usage='held-out'"):
        phx.uq.predict_held_out_observables(result, calibration)


def test_renaming_an_observation_cannot_relabel_fitted_lineage_as_held_out():
    support = phx.uq.EnsembleSupport(np.zeros(3), "empirical", "lineage-support")
    calibration = _cholesky_plan([[0.0], [1.0], [2.0]], [0.8], 0.3, "fit-observation")
    result = phx.uq.reweight_ensemble(
        support, calibration, regularization=0.4, support_policy=_SUPPORT_POLICY
    )
    renamed = _cholesky_plan(
        calibration.per_sample_observables,
        calibration.observed,
        0.3,
        "renamed-held-out",
        usage="held-out",
        source_ids=calibration.source_ids,
        case_ids=calibration.case_ids,
        parent_ids=calibration.parent_ids,
    )

    with pytest.raises(ValueError, match="lineage must be disjoint"):
        phx.uq.predict_held_out_observables(result, renamed)


def test_regularization_selection_refuses_held_out_outcomes():
    support = phx.uq.EnsembleSupport(np.zeros(3), "empirical", "selection-support")
    calibration = _cholesky_plan([[0.0], [1.0], [2.0]], [0.8], 0.3, "fit-observation")
    selection = _cholesky_plan(
        [[0.0], [1.0], [4.0]],
        [1.1],
        0.5,
        "selection-observation",
        usage="model-selection",
    )
    held_out = _cholesky_plan(
        [[0.0], [1.0], [4.0]],
        [1.1],
        0.5,
        "locked-observation",
        usage="held-out",
    )

    selected = phx.uq.select_ensemble_regularization(
        support,
        calibration,
        selection,
        [0.05, 0.5, 5.0],
        support_policy=_SUPPORT_POLICY,
    )

    assert bool(selected.valid)
    assert float(selected.selected_regularization) in (0.05, 0.5, 5.0)
    with pytest.raises(ValueError, match="model-selection"):
        phx.uq.select_ensemble_regularization(support, calibration, held_out, [0.05, 0.5])

    renamed_selection = _cholesky_plan(
        calibration.per_sample_observables,
        calibration.observed,
        0.3,
        "renamed-selection",
        usage="model-selection",
        source_ids=calibration.source_ids,
        case_ids=calibration.case_ids,
        parent_ids=calibration.parent_ids,
    )
    with pytest.raises(ValueError, match="lineages must be disjoint"):
        phx.uq.select_ensemble_regularization(
            support, calibration, renamed_selection, [0.05, 0.5]
        )

    selected_fit = phx.uq.reweight_ensemble(
        support, calibration, regularization=selected, support_policy=_SUPPORT_POLICY
    )
    renamed_held_out = _cholesky_plan(
        selection.per_sample_observables,
        selection.observed,
        0.5,
        "renamed-selection-as-held-out",
        usage="held-out",
        source_ids=selection.source_ids,
        case_ids=selection.case_ids,
        parent_ids=selection.parent_ids,
    )
    with pytest.raises(ValueError, match="model-selection lineages"):
        phx.uq.predict_held_out_observables(selected_fit, renamed_held_out)


def _closure_inputs(source_kind):
    source = (
        _physical_provenance(f"{source_kind}-closure-support")
        if source_kind == "physical-equilibrium"
        else f"{source_kind}-closure-support"
    )
    support = phx.uq.EnsembleSupport(np.zeros(4), source_kind, source)
    equilibrium_sources = (
        (support.source_id,)
        if support.physical_provenance is None
        else (
            support.physical_provenance.source_id,
            support.physical_provenance.reference_manifest_id,
        )
    )
    calibration = _cholesky_plan(
        [[0.0], [0.0], [1.0], [1.0]],
        [0.6],
        0.3,
        f"{source_kind}-fit",
    )
    result = phx.uq.reweight_ensemble(
        support,
        calibration,
        regularization=0.4,
        support_policy=_SUPPORT_POLICY,
    )
    held_out_values = np.asarray([[0.1], [0.3], [0.7], [0.9]])
    held_out_mean = np.asarray(jnp.exp(result.log_weights) @ held_out_values)
    held_out = _cholesky_plan(
        held_out_values,
        held_out_mean,
        0.2,
        f"{source_kind}-equilibrium-observation",
        usage="held-out",
        source_ids=equilibrium_sources,
    )
    prediction = phx.uq.predict_held_out_observables(result, held_out)
    closure_plan = phx.uq.TwoStateThermodynamicClosurePlan(
        "condition-25-celsius",
        ("unbound", "bound"),
        ("replica-a", "replica-b"),
        held_out.observation_id,
        "independent-rates",
        support_policy_id=result.support_policy_id,
        equilibrium_source_ids=equilibrium_sources,
        kinetic_source_ids=("kinetic-assay",),
        equivalence_margin=0.3,
        confidence_multiplier=1.96,
        maximum_combined_standard_error=0.2,
    )
    assignment = closure_plan.state_assignment_record(
        support,
        ("unbound", "unbound", "bound", "bound"),
        ("replica-a", "replica-b", "replica-a", "replica-b"),
        f"{source_kind}-state-assignment",
    )
    equilibrium = closure_plan.equilibrium_record(result, assignment, 0.08)
    kinetic = closure_plan.kinetic_record(
        3.0,
        2.0,
        0.09,
        forward_rate_unit=_PER_SECOND,
        reverse_rate_unit=_PER_SECOND,
    )
    return result, prediction, closure_plan, equilibrium, kinetic


def _evaluate_closure(result, prediction, plan, equilibrium, kinetic):
    return phx.uq.evaluate_two_state_thermodynamic_closure(
        result,
        prediction,
        plan,
        equilibrium,
        kinetic,
        required_replica_effective_sample_size=1.5,
        required_state_overlap=0.2,
        maximum_replica_population_difference=0.05,
        maximum_held_out_quadratic=1.0,
    )


def _closure_evidence(source_kind):
    return _evaluate_closure(*_closure_inputs(source_kind))


def test_thermodynamic_closure_requires_a_physical_reference_measure():
    physical = _closure_evidence("physical-equilibrium")
    proposal_only = _closure_evidence("proposal-only")

    assert physical.outcome == "passed"
    assert bool(physical.valid)
    assert bool(physical.equilibrium_evidence_valid)
    assert bool(physical.kinetic_evidence_valid)
    assert proposal_only.outcome == "inconclusive"
    assert not bool(proposal_only.equilibrium_evidence_valid)
    assert bool(proposal_only.kinetic_evidence_valid)


def test_thermodynamic_closure_rejects_mixed_record_identity():
    result, prediction, plan, equilibrium, kinetic = _closure_inputs(
        "physical-equilibrium"
    )
    wrong_state_order = TwoStateKineticRecord(
        3.0,
        2.0,
        0.09,
        condition_id=plan.condition_id,
        state_ids=tuple(reversed(plan.state_ids)),
        observation_id=plan.kinetic_observation_id,
        source_ids=plan.kinetic_source_ids,
        forward_rate_unit=_PER_SECOND,
        reverse_rate_unit=_PER_SECOND,
    )
    other_support = phx.uq.EnsembleSupport(
        np.zeros(4),
        "physical-equilibrium",
        _physical_provenance("substituted-support"),
    )
    substituted_assignment = plan.state_assignment_record(
        other_support,
        ("unbound", "unbound", "bound", "bound"),
        ("replica-a", "replica-b", "replica-a", "replica-b"),
        "substituted-support-assignment",
    )

    with pytest.raises(ValueError, match="exactly match result support"):
        plan.equilibrium_record(result, substituted_assignment, 0.08)
    with pytest.raises(ValueError, match="exactly match the closure plan"):
        _evaluate_closure(result, prediction, plan, equilibrium, wrong_state_order)


def test_unknown_or_excessive_closure_uncertainty_is_inconclusive():
    result, prediction, plan, _, _ = _closure_inputs("physical-equilibrium")
    assignment = plan.state_assignment_record(
        result.support,
        ("unbound", "unbound", "bound", "bound"),
        ("replica-a", "replica-b", "replica-a", "replica-b"),
        "uncertainty-check-assignment",
    )
    equilibrium = plan.equilibrium_record(result, assignment, None)
    imprecise_kinetic = plan.kinetic_record(
        3.0,
        2.0,
        1.0,
        forward_rate_unit=_PER_SECOND,
        reverse_rate_unit=_PER_SECOND,
    )

    missing = _evaluate_closure(result, prediction, plan, equilibrium, imprecise_kinetic)
    quantified = plan.equilibrium_record(result, assignment, 0.08)
    excessive = _evaluate_closure(result, prediction, plan, quantified, imprecise_kinetic)

    assert missing.outcome == "inconclusive"
    assert not bool(missing.equilibrium_evidence_valid)
    assert excessive.outcome == "inconclusive"
    assert not bool(excessive.uncertainty_useful)
    assert not bool(excessive.valid)


def test_closure_population_ratio_is_derived_from_bound_state_assignments():
    result, prediction, plan, _, kinetic = _closure_inputs("physical-equilibrium")
    reversed_assignment = plan.state_assignment_record(
        result.support,
        ("bound", "bound", "unbound", "unbound"),
        ("replica-a", "replica-b", "replica-a", "replica-b"),
        "reversed-state-assignment",
    )
    equilibrium = plan.equilibrium_record(result, reversed_assignment, 0.08)

    evidence = _evaluate_closure(result, prediction, plan, equilibrium, kinetic)

    assert equilibrium.support_id == result.support.support_id
    assert equilibrium.support_policy_id == result.support_policy_id
    assert equilibrium.population_aggregation == "replica-effective-sample-size-weighted"
    assert evidence.outcome == "failed"
    assert evidence.equivalence_interval_upper > plan.equivalence_margin
    assert not bool(evidence.valid)


def test_kinetic_rates_convert_compatible_reciprocal_time_units():
    result, prediction, plan, equilibrium, per_second = _closure_inputs(
        "physical-equilibrium"
    )
    mixed_units = plan.kinetic_record(
        3.0,
        120.0,
        0.09,
        forward_rate_unit=_PER_SECOND,
        reverse_rate_unit=_PER_MINUTE,
    )

    evidence = _evaluate_closure(result, prediction, plan, equilibrium, mixed_units)

    np.testing.assert_allclose(mixed_units.forward_rate, 3.0)
    np.testing.assert_allclose(mixed_units.reverse_rate, 2.0)
    assert mixed_units.canonical_rate_unit.symbol == "s^-1"
    assert mixed_units.record_id != per_second.record_id
    assert evidence.outcome == "passed"
    with pytest.raises(ValueError, match="matching dimensions"):
        plan.kinetic_record(
            3.0,
            2.0,
            0.09,
            forward_rate_unit=SECOND,
            reverse_rate_unit=_PER_SECOND,
        )
