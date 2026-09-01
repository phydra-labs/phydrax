#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.solid_mechanics._bifurcation import (
    BranchSwitchPolicy,
    EnergyBarrierEvidence,
    ImperfectionFamily,
    ImperfectionStudy,
    MechanicsBifurcationDetector,
    MechanicsBranch,
    MechanicsBranchGraph,
    PhysicalSelectionPolicy,
    select_mechanics_branch,
    switch_mechanics_branch,
)
from phydrax.continuation._bifurcation import (
    BifurcationCertificate,
    BifurcationStatus,
)


sm = phx.applications.solid_mechanics
ct = phx.continuation


def _self_adjoint_properties(*, positive_definite=False):
    evidence = {"self_adjoint": "verified"}
    if positive_definite:
        evidence.update(
            {
                "positive_definite": "verified",
                "positive_semidefinite": "verified",
            }
        )
    return phx.linalg.OperatorProperties(
        self_adjoint=True,
        positive_definite=positive_definite,
        evidence=evidence,
    )


def _static_context():
    dtype = jnp.float32
    space = phx.linalg.ArraySpace((), dtype=dtype, space_id="physical-displacement")
    root = phx.nonlinear.NonlinearSystemProblem(
        lambda state, args: state,
        state_space=space,
        residual_space=space,
        problem_id="mechanics-equilibrium-root",
    )
    equilibrium = sm.MechanicsEquilibriumProblem(
        root,
        realization_id="physical-realization",
        provenance_id="physical-residual-assembly",
    )
    tangent = phx.linalg.DenseLinearOperator(
        jnp.zeros((1, 1), dtype=dtype),
        source=space,
        target=space,
        properties=_self_adjoint_properties(),
        operator_id="physical-critical-tangent",
    )
    stability = sm.PhysicalStaticStabilityProblem(
        equilibrium,
        space,
        tangent,
        tangent_provenance_id="physical-tangent-assembly",
    )
    problem = ct.ParameterContinuationProblem(
        lambda state, coordinate, args: coordinate * state - state**3,
        state_space=space,
        residual_space=space,
        problem_id="mechanics-pitchfork-curve",
    )
    state = jnp.asarray(0.0, dtype=dtype)
    coordinate = jnp.asarray(0.0, dtype=dtype)
    geometry = ct.ContinuationGeometry.resolve(
        state,
        problem.residual(state, coordinate),
        state_space=space,
        residual_space=space,
    )
    return equilibrium, stability, problem, geometry


def _certificate(kind, geometry, *, certificate_id=None):
    state = geometry.public_state_space.zeros()
    mode = jnp.ones_like(state)
    return BifurcationCertificate(
        state=state,
        parameter=jnp.asarray(0.0, dtype=mode.dtype),
        right_nullvector=None if kind == "hopf" else mode,
        left_nullvector=None if kind == "hopf" else mode,
        evidence=None,
        status=BifurcationStatus.CERTIFIED,
        kind=kind,
        assumptions_verified=True,
        certificate_id=certificate_id or f"{kind}-certificate",
        geometry=geometry,
    )


def _primary_graph(problem, geometry):
    state = jnp.asarray(0.0, dtype=jnp.float32)
    point = ct.BranchPoint(
        state=state,
        coordinate=jnp.asarray(0.0, dtype=state.dtype),
        parameters=jnp.asarray(0.0, dtype=state.dtype),
        tangent_state=jnp.asarray(0.0, dtype=state.dtype),
        tangent_coordinate=jnp.asarray(1.0, dtype=state.dtype),
        tangent_parameters=jnp.asarray(1.0, dtype=state.dtype),
        residual_norm=0.0,
        step_size=0.0,
        corrector_iterations=0,
        corrector_retries=0,
        status=0,
        fold_candidate=False,
        point_id="critical-point",
    )
    continuation = ct.ContinuationBranch(
        (point,),
        (),
        0,
        geometry=geometry,
        branch_id="primary-branch",
        problem_id=problem.problem_id,
        method="certified-primary",
        termination_reason="localized critical point",
    )
    primary = MechanicsBranch(
        continuation=continuation,
        branch_id="primary-branch",
        control_protocol="pseudo-arclength",
        realization_id="physical-realization",
        provenance_id="physical-residual-assembly",
    )
    return MechanicsBranchGraph((primary,))


def test_detector_preserves_fold_pitchfork_and_transcritical_semantics():
    equilibrium, stability, _, geometry = _static_context()
    detector = MechanicsBifurcationDetector(equilibrium)

    fold = detector.detect(_certificate("fold", geometry), static_stability=stability)
    pitchfork = detector.detect(
        _certificate("pitchfork", geometry),
        static_stability=stability,
    )
    transcritical = detector.detect(
        _certificate("transcritical", geometry),
        static_stability=stability,
    )
    branch_point = _certificate("branch-point", geometry)
    buckling = detector.detect(
        branch_point,
        static_stability=stability,
        static_interpretation="static-buckling",
        conservative_verified=True,
        proportional_load_verified=True,
    )
    positive_definiteness = detector.detect(
        branch_point,
        static_stability=stability,
        static_interpretation="loss-of-positive-definiteness",
        conservative_verified=True,
    )

    assert fold.classification == "limit-point"
    assert pitchfork.classification == "pitchfork"
    assert transcritical.classification == "transcritical"
    assert buckling.classification == "static-buckling"
    assert positive_definiteness.classification == "loss-of-positive-definiteness"
    assert fold.eigenvalue_quantity == "physical-tangent-curvature"
    assert pitchfork.physical_space_id == "physical-displacement"
    assert pitchfork.mode_provenance_id == "physical-tangent-assembly"
    assert pitchfork.root_mode is not None
    assert pitchfork.physical_mode is not None


def test_hopf_requires_dynamic_evidence_and_refuses_static_claims():
    dtype = jnp.float32
    space = phx.linalg.ArraySpace((2,), dtype=dtype, space_id="physical-dynamics")
    root = phx.nonlinear.NonlinearSystemProblem(
        lambda state, args: state,
        state_space=space,
        residual_space=space,
        problem_id="dynamic-equilibrium-root",
    )
    equilibrium = sm.MechanicsEquilibriumProblem(
        root,
        realization_id="dynamic-realization",
        provenance_id="dynamic-residual-assembly",
    )
    stiffness = phx.linalg.DenseLinearOperator(
        jnp.eye(2, dtype=dtype),
        source=space,
        target=space,
        properties=_self_adjoint_properties(),
        operator_id="dynamic-stiffness",
    )
    mass = phx.linalg.DenseLinearOperator(
        jnp.eye(2, dtype=dtype),
        source=space,
        target=space,
        properties=_self_adjoint_properties(positive_definite=True),
        operator_id="dynamic-mass",
    )
    static = sm.PhysicalStaticStabilityProblem(
        equilibrium,
        space,
        stiffness,
        tangent_provenance_id="static-tangent",
    )
    dynamic = sm.DynamicStabilityProblem(
        equilibrium,
        space,
        stiffness,
        mass,
        stiffness_provenance_id="dynamic-stiffness-assembly",
        mass_provenance_id="dynamic-mass-assembly",
    )
    curve = ct.ParameterContinuationProblem(
        lambda state, coordinate, args: state,
        state_space=space,
        residual_space=space,
        problem_id="dynamic-first-order-curve",
    )
    state = jnp.zeros((2,), dtype=dtype)
    geometry = ct.ContinuationGeometry.resolve(
        state,
        curve.residual(state, jnp.asarray(0.0, dtype=dtype)),
        state_space=space,
        residual_space=space,
    )
    certificate = _certificate("hopf", geometry)
    detector = MechanicsBifurcationDetector(equilibrium)

    with pytest.raises(ValueError, match="static stability contract"):
        detector.detect(certificate, static_stability=static)
    with pytest.raises(ValueError, match="DynamicStabilityProblem"):
        detector.detect(certificate)

    record = detector.detect(certificate, dynamic_stability=dynamic)
    assert record.classification == "hopf-flutter"
    assert record.eigenvalue_quantity == "squared-angular-frequency"
    assert record.dynamic_stability is dynamic


def test_corrected_switching_builds_lineage_and_checks_symmetry_duplicates():
    equilibrium, stability, problem, geometry = _static_context()
    detector = MechanicsBifurcationDetector(equilibrium)
    record = detector.detect(
        _certificate("pitchfork", geometry),
        static_stability=stability,
    )
    graph = _primary_graph(problem, geometry)
    policy = BranchSwitchPolicy(
        amplitude=0.1,
        coordinate_offset=0.02,
        residual_tolerance=1e-6,
        control_protocol="pseudo-arclength",
    )
    switched = switch_mechanics_branch(
        record,
        problem,
        phx.nonlinear.NewtonKrylov(),
        graph,
        lambda physical_mode, state, args: physical_mode,
        physical_mode_lift_id="identity-physical-mode-lift",
        source_branch_id="primary-branch",
        source_point_id="critical-point",
        policy=policy,
        symmetry=lambda state: -state,
    )

    assert len(switched.corrections) == 2
    assert all(bool(correction.successful) for correction in switched.corrections)
    assert len(switched.accepted_branch_ids) == 2
    assert not switched.duplicate_branch_ids
    assert len(switched.graph.branches) == 3
    assert len(switched.graph.edges) == 2
    assert all(edge.parent_branch_id == "primary-branch" for edge in switched.graph.edges)
    corrected_coordinates = [
        float(correction.seed.coordinate) for correction in switched.corrections
    ]
    np.testing.assert_allclose(corrected_coordinates, (0.01, 0.01), atol=1e-5)
    corrected_states = [
        float(correction.seed.state) for correction in switched.corrections
    ]
    np.testing.assert_allclose(corrected_states, (0.1, -0.1), atol=1e-6)
    assert switched.graph.edges[1].symmetry_related

    quotient = switch_mechanics_branch(
        record,
        problem,
        phx.nonlinear.NewtonKrylov(),
        graph,
        lambda physical_mode, state, args: physical_mode,
        physical_mode_lift_id="identity-physical-mode-lift",
        source_branch_id="primary-branch",
        source_point_id="critical-point",
        policy=BranchSwitchPolicy(
            amplitude=0.1,
            coordinate_offset=0.02,
            residual_tolerance=1e-6,
            quotient_symmetry=True,
            control_protocol="pseudo-arclength",
        ),
        symmetry=lambda state: -state,
    )
    assert len(quotient.accepted_branch_ids) == 1
    assert len(quotient.symmetry_rejected_branch_ids) == 1


def test_imperfection_family_and_zero_limit_study_preserve_provenance():
    family = ImperfectionFamily(
        jnp.asarray((0.0, 1.0, -1.0), dtype=jnp.float32),
        units="m",
        orientation="first-antisymmetric-mode",
        discretization_id="mesh-level-2",
        fabrication_provenance_id="survey-2026-08",
    )
    positive = family.realize(jnp.asarray(2e-3, dtype=jnp.float32))
    negative = family.realize(jnp.asarray(-2e-3, dtype=jnp.float32))
    baseline = family.realize(jnp.asarray(0.0, dtype=jnp.float32))
    study = ImperfectionStudy(
        family,
        jnp.asarray((-2e-3, 0.0, 2e-3), dtype=jnp.float32),
        (None, None, None),
        limit_resolved=True,
    )

    np.testing.assert_allclose(positive, -negative)
    np.testing.assert_allclose(baseline, 0.0)
    assert study.family.family_id == family.family_id
    assert bool(study.limit_resolved)
    assert study.family.discretization_id == "mesh-level-2"


def test_explicit_selection_policies_can_disagree_and_refuse_potential_claims():
    _, _, problem, geometry = _static_context()
    graph = _primary_graph(problem, geometry)
    primary = graph.branches[0]
    second_continuation = ct.ContinuationBranch(
        primary.continuation.points,
        (),
        0,
        geometry=geometry,
        branch_id="energy-minimum",
        problem_id=problem.problem_id,
        method="certified-secondary",
        termination_reason="selected branch",
    )
    second = MechanicsBranch(
        continuation=second_continuation,
        branch_id="energy-minimum",
        control_protocol="pseudo-arclength",
        realization_id="physical-realization",
        provenance_id="physical-residual-assembly",
    )
    graph = MechanicsBranchGraph((primary, second))
    connected = select_mechanics_branch(
        graph,
        PhysicalSelectionPolicy("stable-connected"),
        stable_branch_ids=("primary-branch", "energy-minimum"),
        connected_branch_id="primary-branch",
    )
    energetic = select_mechanics_branch(
        graph,
        PhysicalSelectionPolicy("global-energy-minimum"),
        branch_energies={"primary-branch": 2.0, "energy-minimum": 1.0},
        potential_verified=True,
    )
    refused = select_mechanics_branch(
        graph,
        PhysicalSelectionPolicy("global-energy-minimum"),
        branch_energies={"primary-branch": 2.0, "energy-minimum": 1.0},
        potential_verified=False,
    )

    assert connected.selected_branch_id == "primary-branch"
    assert energetic.selected_branch_id == "energy-minimum"
    assert connected.disagrees_with(energetic)
    assert refused.status == "unavailable"
    assert refused.selected_branch_id is None

    with pytest.raises(ValueError, match="verified conservative potential"):
        EnergyBarrierEvidence(
            source_branch_id="primary-branch",
            target_branch_id="energy-minimum",
            source_energy=2.0,
            target_energy=1.0,
            saddle_energy=3.0,
            source_morse_index=0,
            target_morse_index=0,
            saddle_morse_index=1,
            stationary_residual=0.0,
            path_defect=0.0,
            refinement_defect=0.0,
            admissible=True,
            potential_verified=False,
            conservative_verified=False,
            potential_provenance_id="residual-loss-is-not-potential",
        )

    barrier = EnergyBarrierEvidence(
        source_branch_id="primary-branch",
        target_branch_id="energy-minimum",
        source_energy=2.0,
        target_energy=1.0,
        saddle_energy=3.0,
        source_morse_index=0,
        target_morse_index=0,
        saddle_morse_index=1,
        stationary_residual=1e-9,
        path_defect=1e-7,
        refinement_defect=1e-7,
        admissible=True,
        potential_verified=True,
        conservative_verified=True,
        potential_provenance_id="verified-total-potential",
    )
    assert bool(barrier.certified)
    assert barrier.saddle_morse_index == 1
    assert barrier.potential_verified
    assert barrier.conservative_verified
