#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.applications.compact_objects._inverse import (
    FixedBranchInverseAdapter,
    FixedBranchModelEvaluation,
)
from phydrax.applications.numerical_relativity._uncertainty import (
    admit_learned_closure,
    apply_admitted_learned_closure,
    LearnedClosureAdmissionEvidence,
    LearnedClosureCandidate,
    ModelDiscrepancyRecord,
    NumericalErrorRecord,
    RelativisticMultifidelityPlan,
    smooth_grhd_inverse_adapter,
    smooth_nr_inverse_adapter,
)


def _component(kind, realization, function):
    def evaluate(parameters):
        return FixedBranchModelEvaluation(
            function(parameters),
            jnp.asarray([0], dtype=jnp.int32),
            finite=True,
            converged=True,
            physically_valid=True,
            qualified=True,
            derivative_valid=True,
            shock_free=True,
            event_free=True,
            topology_fixed=True,
            realization_id=realization,
            branch_id="fixed",
        )

    return FixedBranchInverseAdapter(
        evaluate,
        jnp.asarray([0]),
        parameter_count=2,
        output_count=2,
        model_kind=kind,
        realization_id=realization,
        branch_id="fixed",
        adapter_id=f"adapter:{kind}",
        evaluator_semantic_id=f"multifidelity:{kind}",
        evaluator_numeric_id=f"multifidelity:{realization}",
    )


def _error(realization, scale):
    return NumericalErrorRecord(
        jnp.asarray([scale, 2.0 * scale]),
        converged=True,
        physically_valid=True,
        qualified=True,
        method_id="paired-refinement",
        realization_id=realization,
        evidence_id=f"refinement:{realization}",
    )


def test_multifidelity_composition_keeps_levels_and_uncertainties_separate():
    exact = _component("exact-baseline", "exact:r1", lambda p: p)
    perturbative = _component("perturbative-correction", "pert:r2", lambda p: 0.1 * p**2)
    rom = _component("rom-correction", "rom:r3", lambda p: 0.2 * jnp.sin(p))
    full = _component("full-simulation-correction", "full:r4", lambda p: -0.05 * p)
    discrepancy = ModelDiscrepancyRecord(
        jnp.asarray([0.1, -0.2]),
        jnp.asarray([[0.3], [0.4]]),
        converged=True,
        physically_valid=True,
        qualified=True,
        model_id="discrepancy",
        calibration_evidence_id="calibration:held-in",
        validation_evidence_id="validation:held-out",
        support_id="support:two-observables",
    )
    plan = RelativisticMultifidelityPlan(
        exact,
        perturbative,
        rom,
        full,
        numerical_errors=(
            _error("exact:r1", 0.01),
            _error("pert:r2", 0.02),
            _error("rom:r3", 0.03),
            _error("full:r4", 0.04),
        ),
        model_discrepancy=discrepancy,
        plan_id="four-level-composition",
    )
    parameters = jnp.asarray([0.4, -0.3])
    result = plan.evaluate(parameters)
    expected_terms = jnp.stack(
        (
            parameters,
            0.1 * parameters**2,
            0.2 * jnp.sin(parameters),
            -0.05 * parameters,
        )
    )

    assert jnp.allclose(result.term_values, expected_terms)
    assert jnp.allclose(result.native_value, jnp.sum(expected_terms, axis=0))
    assert jnp.allclose(result.value, result.native_value + discrepancy.mean)
    assert jnp.allclose(result.numerical_error_bound, jnp.asarray([0.1, 0.2]))
    assert jnp.allclose(
        result.discrepancy_covariance, jnp.asarray([[0.09, 0.12], [0.12, 0.16]])
    )
    assert bool(result.qualified)
    assert bool(result.derivative_valid)
    assert bool(result.uncertainty_qualified)

    sensitivity = plan.sensitivity(parameters, jnp.asarray([0.25, -0.1]), epsilon=2.0e-4)
    assert bool(sensitivity.derivative_valid)
    assert float(sensitivity.jvp_finite_difference_residual) < 2.0e-3
    assert float(sensitivity.vjp_pairing_residual) < 1.0e-6


def test_unknown_numerical_or_model_error_is_not_silently_zero():
    components = (
        _component("exact-baseline", "exact:r1", lambda p: p),
        _component("perturbative-correction", "pert:r2", lambda p: p * 0.0),
        _component("rom-correction", "rom:r3", lambda p: p * 0.0),
        _component("full-simulation-correction", "full:r4", lambda p: p * 0.0),
    )
    result = RelativisticMultifidelityPlan(
        *components, plan_id="unknown-uncertainty"
    ).evaluate(jnp.asarray([1.0, 2.0]))

    assert not bool(result.numerical_error_available)
    assert not bool(result.model_discrepancy_available)
    assert not bool(result.uncertainty_qualified)
    assert jnp.all(jnp.isinf(result.numerical_error_bound))
    assert jnp.all(jnp.isnan(result.discrepancy_covariance))


def test_smooth_grhd_and_nr_adapters_fail_closed_at_shocks_or_topology_changes():
    def grhd(parameters):
        return FixedBranchModelEvaluation(
            parameters**2,
            jnp.asarray([0]),
            finite=True,
            converged=True,
            physically_valid=True,
            qualified=True,
            derivative_valid=True,
            shock_free=parameters[0] > 0.0,
            event_free=True,
            topology_fixed=True,
            realization_id="grhd:smooth",
            branch_id="primitive-recovery-0",
        )

    grhd_adapter = smooth_grhd_inverse_adapter(
        grhd,
        jnp.asarray([0]),
        parameter_count=2,
        output_count=2,
        realization_id="grhd:smooth",
        branch_id="primitive-recovery-0",
        adapter_id="grhd-parameters",
        evaluator_semantic_id="grhd-smooth-observable",
        evaluator_numeric_id="grhd-smooth-forward:r1",
    )
    smooth = grhd_adapter.sensitivity(jnp.asarray([1.0, 0.5]), jnp.asarray([0.1, -0.2]))
    shocked = grhd_adapter.sensitivity(jnp.asarray([-1.0, 0.5]), jnp.asarray([0.1, -0.2]))
    assert bool(smooth.derivative_valid)
    assert not bool(shocked.derivative_valid)
    assert jnp.all(jnp.isnan(shocked.jvp))

    nr_adapter = smooth_nr_inverse_adapter(
        lambda p: FixedBranchModelEvaluation(
            p,
            jnp.asarray([0]),
            finite=True,
            converged=True,
            physically_valid=True,
            qualified=True,
            derivative_valid=True,
            shock_free=True,
            event_free=True,
            topology_fixed=False,
            realization_id="z4c:grid-1",
            branch_id="fixed-step",
        ),
        jnp.asarray([0]),
        parameter_count=2,
        output_count=2,
        realization_id="z4c:grid-1",
        branch_id="fixed-step",
        adapter_id="z4c-parameters",
        evaluator_semantic_id="z4c-smooth-observable",
        evaluator_numeric_id="z4c-smooth-forward:r1",
    )
    topology_change = nr_adapter.sensitivity(
        jnp.asarray([0.1, 0.2]), jnp.asarray([0.2, 0.1])
    )
    assert not bool(topology_change.derivative_valid)
    assert jnp.all(jnp.isnan(topology_change.finite_difference))


def test_learned_closure_requires_all_evidence_and_only_adds_to_native_physics():
    candidate = LearnedClosureCandidate(
        lambda value: 0.1 * value,
        native_model_id="z4c:native",
        model_id="closure:weights",
        training_realization_id="training:split-7",
        derivative_supported=True,
        differentiation_evidence_id="derivatives:jvp-vjp-fd",
    )
    missing_rights = LearnedClosureAdmissionEvidence(
        jnp.asarray([1.0e-9, -2.0e-9]),
        jnp.asarray([0.2, 0.4]),
        conservation_tolerance=1.0e-8,
        converged=True,
        physically_valid=True,
        qualified=True,
        rights_authorized=False,
        candidate_id=candidate.candidate_id,
        conservation_evidence_id="conservation:campaign",
        admissibility_evidence_id="admissibility:campaign",
        rights_evidence_id="rights:denied",
    )
    refused = admit_learned_closure(candidate, missing_rights)
    assert not refused.admitted
    assert "requested-use-rights-missing" in refused.refusal_reasons
    with pytest.raises(ValueError, match="unadmitted"):
        apply_admitted_learned_closure(
            candidate, refused, jnp.asarray([2.0, 3.0]), jnp.asarray([1.0, 1.0])
        )

    admitted_evidence = LearnedClosureAdmissionEvidence(
        jnp.asarray([1.0e-9, -2.0e-9]),
        jnp.asarray([0.2, 0.4]),
        conservation_tolerance=1.0e-8,
        converged=True,
        physically_valid=True,
        qualified=True,
        rights_authorized=True,
        candidate_id=candidate.candidate_id,
        conservation_evidence_id="conservation:campaign",
        admissibility_evidence_id="admissibility:campaign",
        rights_evidence_id="rights:requested-use-granted",
    )
    admitted = admit_learned_closure(candidate, admitted_evidence)
    native = jnp.asarray([2.0, 3.0])
    correction_inputs = jnp.asarray([1.0, -2.0])
    result = apply_admitted_learned_closure(
        candidate, admitted, native, correction_inputs
    )

    assert admitted.admitted
    assert jnp.allclose(result.native_value, native)
    assert jnp.allclose(result.learned_correction, jnp.asarray([0.1, -0.2]))
    assert jnp.allclose(result.combined_value, jnp.asarray([2.1, 2.8]))
    assert bool(result.qualified)
    assert bool(result.derivative_valid)
