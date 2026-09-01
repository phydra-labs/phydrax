#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _claim():
    return phx.discretization.MPMClaimTuple(
        equation_family="solid-mechanics",
        dimension=2,
        kinematics="plane-strain",
        grid_assignment="quadratic-bspline",
        source_domain="point",
        transfer="apic",
        schedule="usl-minus",
        material="neo-hookean",
        field_contact="single-field-none",
        fracture="none",
        integrator="explicit-fixed",
        storage_backend="dense-cpu-f64-deterministic",
        precision_accumulation="f64-deterministic",
        capacity_envelope="particles-4-grid-10x10",
        derivative_mode="branchwise",
    )


def _intended_use():
    return phx.discretization.MPMIntendedUse(
        "predict small elastic block displacement",
        phenomena=("finite-strain elasticity",),
        target_observables=("particle displacement", "reaction force"),
        prohibited_uses=("fracture", "plasticity"),
        risk_class="commercial-low-consequence",
        geometry_loading_scope="periodic unit square and prescribed loads",
        material_parameter_scope="positive finite shear/bulk calibration domain",
        accuracy_uq_goal="relative displacement error below one percent",
    )


def test_support_matrix_is_exact_and_fail_closed():
    claim = _claim()
    supported = phx.discretization.MPMSupportDecision(
        claim,
        phx.discretization.MPMClaimOutcome.SUPPORTED,
        reason="qualified tuple",
        required_profile="commercial-runtime",
    )
    rejected_claim = phx.discretization.MPMClaimTuple(
        equation_family="solid-mechanics",
        dimension=3,
        kinematics="three-dimensional",
        grid_assignment="cpdi2",
        source_domain="cpdi2",
        transfer="flip",
        schedule="post-advection-musl",
        material="mohr-coulomb",
        field_contact="kway-sharp",
        fracture="cpic",
        integrator="implicit-compact-distributed",
        storage_backend="dynamic-unbounded",
        precision_accumulation="fast",
        capacity_envelope="unbounded",
        derivative_mode="classical-everywhere",
    )
    rejected = phx.discretization.MPMSupportDecision(
        rejected_claim,
        phx.discretization.MPMClaimOutcome.REJECTED,
        reason="unbounded and incompatible derivative claim",
        required_profile="none",
    )
    matrix = phx.discretization.MPMSupportMatrix((supported, rejected))

    assert matrix.decision(claim.claim_id).require_supported()
    with pytest.raises(ValueError, match="REJECTED"):
        matrix.decision(rejected_claim.claim_id).require_supported()
    with pytest.raises(KeyError):
        matrix.decision("missing")


def test_release_bundle_requires_g0_g7_and_independent_review():
    claim = _claim()
    intended = _intended_use()
    gates = tuple(
        phx.discretization.MPMReleaseGateEvidence(
            gate,
            passed=True,
            evidence_ids=(f"evidence-{gate.name.lower()}",),
            reviewer_id=f"reviewer-{int(gate)}",
        )
        for gate in phx.discretization.MPMReleaseGate
    )
    bundle = phx.discretization.MPMReleaseEvidenceBundle(
        claim,
        intended,
        gates,
        independent_approver_id="independent-approver",
    )
    assert bundle.releasable

    standards = phx.discretization.MPMStandardsTraceabilityMatrix(
        (
            phx.discretization.MPMStandardsTrace(
                standard="ASME V&V 10",
                edition="2019 (R2025)",
                applicability="computational solid mechanics",
                requirement="code and solution verification",
                evidence_ids=(gates[1].gate_id, gates[2].gate_id),
                satisfied=True,
            ),
        )
    )
    matrix = phx.discretization.MPMSupportMatrix(
        (
            phx.discretization.MPMSupportDecision(
                claim,
                phx.discretization.MPMClaimOutcome.SUPPORTED,
                reason="qualified tuple",
                required_profile="commercial-runtime",
            ),
        )
    )
    profile = phx.discretization.MPMCommercialProfile(
        "commercial-runtime",
        phx.discretization.MPMCommercialProfileKind.COMMERCIAL_RUNTIME,
        matrix,
        standards,
    )
    review = phx.discretization.MPMIndependentReview(
        author_id="author",
        technical_reviewer_id="reviewer",
        release_approver_id="approver",
    )
    assessment = phx.discretization.assess_release(
        profile,
        claim,
        intended,
        {value.gate: value for value in gates},
        review,
    )
    assert assessment.releasable
    assert not assessment.reasons


def test_derivative_results_distinguish_branch_event_surrogate_and_nondifferentiable():
    objective = lambda value: jnp.sum(value**2)
    primal = jnp.asarray((1.0, -2.0))
    direction = jnp.asarray((0.2, 0.3))
    branch = phx.discretization.branchwise_gradient(
        objective,
        primal,
        direction,
        branch_margin=0.4,
        journal_digest=17,
        evidence_id="branch",
    )
    np.testing.assert_allclose(branch.derivative, -0.8, atol=1e-12)
    assert bool(branch.evidence.valid)
    assert int(branch.evidence.kind) == int(
        phx.discretization.MPMDerivativeKind.BRANCHWISE
    )

    event = phx.discretization.locate_event(
        lambda time: time - 0.25,
        0.0,
        1.0,
    )
    assert bool(event.localized)
    assert abs(float(event.event_time) - 0.25) < 1e-10

    saltation = phx.discretization.saltation_action(
        jnp.asarray((1.0, 0.0)),
        jnp.asarray((0.5, 0.2)),
        jnp.asarray((1.0, 0.0)),
        jnp.eye(2),
        jnp.asarray((0.1, 0.3)),
        evidence_id="saltation",
    )
    assert bool(saltation.evidence.valid)

    nondifferentiable = phx.discretization.nondifferentiable_result(
        primal,
        reason_code=99,
        journal_digest=42,
        evidence_id="topology-change",
    )
    assert nondifferentiable.derivative is None
    assert not bool(nondifferentiable.evidence.valid)
