#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.compact_objects._black_hole_scattering import (
    BlackHoleScatteringPlan,
    BlackHoleScatteringStatus,
    ScatteringQualificationEvidence,
    SchwarzschildScatteringSolvePlan,
    SchwarzschildScatteringStatus,
    solve_black_hole_scattering,
    solve_schwarzschild_scattering,
    SuperradianceStatus,
)
from phydrax.applications.compact_objects._perturbation import (
    RadialBoundaryCondition,
    SeparatedMode,
)
from phydrax.applications.compact_objects._qnm import (
    BoundedContinuedFractionPlan,
    qnm_continuation_problem,
    QnmDerivativeStatus,
    QnmSolvePlan,
    QnmStatus,
    schwarzschild_qnm_reference,
    solve_qnm,
)
from phydrax.applications.compact_objects._radial_perturbation import (
    SchwarzschildRadialPlan,
)
from phydrax.applications.compact_objects._spheroidal import SpheroidalAngularPlan
from phydrax.linalg import DenseLU, LinearSolvePolicy
from phydrax.nonlinear import (
    NewtonKrylov,
    NonlinearTermination,
    SensitivityPolicy,
)


def _schwarzschild_gravitational_plan(
    *, radial_qualification: bool = True
) -> QnmSolvePlan:
    mode = SeparatedMode(
        -2,
        2,
        2,
        overtone=0,
        sector="regge-wheeler",
        family="qnm",
        background_id="test-schwarzschild-M1",
    )
    angular = SpheroidalAngularPlan(
        mode,
        5,
        residual_tolerance=1.0e-9,
        isolation_tolerance=1.0e-9,
        minimum_target_overlap=1.0e-6,
        maximum_condition=1.0e10,
    )
    radial = SchwarzschildRadialPlan(
        mode,
        1.0,
        node_count=65 if radial_qualification else 17,
        outer_radius=30.0 if radial_qualification else 10.0,
        residual_tolerance=1.0e-5 if radial_qualification else 1.0e-8,
        matching_tolerance=1.0e-7 if radial_qualification else 1.0e-10,
        asymptotic_tolerance=5.0e-2,
        maximum_dimension=96,
        integration_substeps=32 if radial_qualification else 1,
        infinity_asymptotic_order=12 if radial_qualification else 1,
    )
    angular_fraction = BoundedContinuedFractionPlan(
        48,
        96,
        0,
        1.0e-9,
        1.0e-9,
    )
    radial_fraction = BoundedContinuedFractionPlan(
        96,
        192,
        0,
        1.0e-9,
        1.0e-9,
    )
    return QnmSolvePlan(
        mode,
        angular,
        radial,
        jnp.asarray(0.0),
        NewtonKrylov(),
        NonlinearTermination(
            absolute_residual=1.0e-10,
            relative_residual=1.0e-10,
            maximum_steps=20,
        ),
        SensitivityPolicy("implicit-forward", condition_limit=1.0e12),
        angular_fraction,
        radial_fraction,
        1.0e-8,
        1.0e12,
        (-0.5, 0.5),
        branch_id="schwarzschild-s-2-l2-n0",
    )


def test_coupled_qnm_root_retains_reference_depth_condition_and_branch_evidence():
    plan = _schwarzschild_gravitational_plan()
    reference = schwarzschild_qnm_reference(plan.mode, 2.0e-8, 2.0e-8)
    result = solve_qnm(
        plan,
        reference.angular_frequency * (1.0 + 1.0e-4),
        jnp.asarray(4.001 + 0.0j),
        qualification=reference,
        continuation_active=True,
    )

    np.testing.assert_allclose(
        np.asarray(result.angular_frequency),
        np.asarray(reference.angular_frequency),
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(result.separation_constant),
        np.asarray(reference.separation_constant),
        rtol=0.0,
        atol=2.0e-9,
    )
    assert result.residuals.shape == (2,)
    assert float(result.residual_norm) < 1.0e-9
    assert bool(result.angular_depth_evidence.resolved)
    assert bool(result.radial_depth_evidence.resolved)
    assert int(result.angular_depth_evidence.maximum_depth) == 96
    assert int(result.radial_depth_evidence.maximum_depth) == 192
    assert bool(result.angular_resolution.qualified)
    assert bool(result.radial_resolution.qualified)
    assert bool(result.radial_resolution.asymptotic.qualified)
    assert result.radial_resolution.residual_evidence.node_count == 65
    assert abs(complex(result.radial_resolution.residual)) < 1.0e-7
    assert (
        float(result.radial_resolution.residual_evidence.relative_residual)
        < plan.radial_plan.residual_tolerance
    )
    assert np.isfinite(float(result.radial_resolution_error))
    assert float(result.minimum_singular_value) > 0.0
    assert 1.0 < float(result.root_condition) < plan.condition_limit
    assert bool(result.successful)
    assert bool(result.qualified)
    assert bool(result.derivative_valid)
    assert int(result.derivative_status) == int(QnmDerivativeStatus.VALID)
    assert int(result.status) == int(QnmStatus.SUCCESS)
    assert bool(result.continuation_active)
    assert result.branch_id == "schwarzschild-s-2-l2-n0"
    assert result.radial_source_id == plan.radial_plan.plan_id

    continuation = qnm_continuation_problem(plan)
    initial_state = result.nonlinear_result.state
    assert continuation.residual(initial_state, jnp.asarray(0.0)).shape == (4,)
    assert continuation.problem_id.endswith(":spin-continuation")


def test_qnm_reference_is_not_implicit_qualification():
    plan = _schwarzschild_gravitational_plan()
    reference = schwarzschild_qnm_reference(plan.mode, 1.0e-7, 1.0e-7)
    result = solve_qnm(
        plan,
        reference.angular_frequency,
        reference.separation_constant,
    )

    assert bool(result.successful)
    assert not bool(result.qualified)
    assert result.qualification_source_id == ""
    assert result.reference_id == ""


def test_qnm_rejects_reference_matching_cf_root_when_radial_check_fails():
    plan = _schwarzschild_gravitational_plan(radial_qualification=False)
    reference = schwarzschild_qnm_reference(plan.mode, 1.0e-7, 1.0e-7)
    result = solve_qnm(
        plan,
        reference.angular_frequency,
        reference.separation_constant,
        qualification=reference,
    )

    assert bool(result.nonlinear_result.successful)
    assert bool(result.angular_depth_evidence.resolved)
    assert bool(result.radial_depth_evidence.resolved)
    assert bool(result.angular_resolution.qualified)
    assert not bool(result.radial_resolution.qualified)
    assert result.radial_resolution.residual_evidence.node_count == 17
    assert not bool(result.radial_resolution.asymptotic.qualified)
    assert np.isfinite(
        float(result.radial_resolution.residual_evidence.relative_residual)
    )
    assert not bool(result.converged)
    assert not bool(result.successful)
    assert not bool(result.qualified)
    assert not bool(result.derivative_valid)
    assert int(result.status) == int(QnmStatus.RADIAL_RESOLUTION_UNRESOLVED)
    assert int(result.derivative_status) == int(
        QnmDerivativeStatus.RADIAL_RESOLUTION_UNRESOLVED
    )


def _computed_scalar_scattering_plan(
    *,
    asymptotically_resolved: bool = True,
    boundary: RadialBoundaryCondition | None = None,
    flux_tolerance: float = 1.0e-7,
) -> SchwarzschildScatteringSolvePlan:
    mode = SeparatedMode(
        0,
        0,
        0,
        family="scattering",
        sector="scalar",
        background_id="computed-schwarzschild-scattering",
    )
    radial = SchwarzschildRadialPlan(
        mode,
        1.0,
        node_count=65 if asymptotically_resolved else 17,
        inner_radius=2.00002 if asymptotically_resolved else 2.0002,
        outer_radius=80.0 if asymptotically_resolved else 10.0,
        residual_tolerance=1.0e-5,
        matching_tolerance=1.0e-7,
        asymptotic_tolerance=5.0e-2,
        maximum_dimension=96,
        integration_substeps=16 if asymptotically_resolved else 1,
        infinity_asymptotic_order=3 if asymptotically_resolved else 1,
        boundary=boundary,
    )
    flux = BlackHoleScatteringPlan(mode, 0.0, flux_tolerance, 1.0e-10)
    return SchwarzschildScatteringSolvePlan(
        radial,
        flux,
        LinearSolvePolicy(DenseLU()),
        refined_integration_substeps=32 if asymptotically_resolved else 2,
        frequency_step=1.0e-3,
        decomposition_tolerance=1.0e-10,
        refinement_tolerance=1.0e-5,
        slope_refinement_tolerance=5.0e-3,
        absolute_flux_tolerance=1.0e-8,
        incident_amplitude_tolerance=1.0e-12,
        low_frequency_maximum=5.0e-2,
        low_frequency_relative_tolerance=2.5e-1,
    )


def test_computed_scalar_schwarzschild_scattering_closes_flux_and_low_frequency_control():
    plan = _computed_scalar_scattering_plan()
    result = solve_schwarzschild_scattering(plan, jnp.asarray(2.0e-2))

    assert bool(result.qualified)
    assert bool(result.derivative_valid)
    assert int(result.status) == int(SchwarzschildScatteringStatus.SUCCESS)
    assert bool(result.asymptotic.qualified)
    assert bool(result.evidence.source_converged)
    assert bool(result.evidence.absolute_wronskian_valid)
    assert bool(result.evidence.absolute_flux_valid)
    assert bool(result.evidence.slope_refinement_valid)
    assert bool(jnp.all(result.evidence.neighboring_ledger_valid))
    assert float(result.evidence.dimensionless_slope_refinement_error) <= float(
        result.evidence.dimensionless_slope_refinement_threshold
    )
    assert bool(jnp.all(result.evidence.neighboring_resolved))
    assert float(result.evidence.decomposition_residual) < 1.0e-10
    assert float(result.evidence.graybody_refinement_error) < plan.refinement_tolerance
    np.testing.assert_allclose(
        np.asarray(result.graybody_factor),
        np.asarray(16.0 * (plan.radial_plan.mass * result.angular_frequency) ** 2),
        rtol=plan.low_frequency_relative_tolerance,
    )
    np.testing.assert_allclose(
        np.asarray(result.flux_residual),
        0.0,
        atol=plan.absolute_flux_tolerance,
    )
    np.testing.assert_allclose(
        np.asarray(result.wronskian_residual),
        0.0,
        atol=plan.absolute_flux_tolerance,
    )
    assert np.isfinite(float(result.corotation_slope))


def test_computed_scattering_rejects_underresolved_asymptotic_source():
    plan = _computed_scalar_scattering_plan(asymptotically_resolved=False)
    result = solve_schwarzschild_scattering(plan, jnp.asarray(2.0e-2))

    assert not bool(result.asymptotic.qualified)
    assert not bool(result.evidence.source_converged)
    assert not bool(result.converged)
    assert not bool(result.qualified)
    assert not bool(result.derivative_valid)
    assert int(result.status) == int(SchwarzschildScatteringStatus.ASYMPTOTIC_UNRESOLVED)


def test_computed_scattering_returns_invalid_frequency_status():
    plan = _computed_scalar_scattering_plan(asymptotically_resolved=False)
    result = solve_schwarzschild_scattering(plan, jnp.asarray(plan.frequency_step))

    assert bool(result.finite)
    assert not bool(result.converged)
    assert not bool(result.physically_valid)
    assert not bool(result.qualified)
    assert int(result.status) == int(SchwarzschildScatteringStatus.INVALID_FREQUENCY)


def test_computed_scattering_propagates_strict_flux_ledger_failure():
    plan = _computed_scalar_scattering_plan(flux_tolerance=1.0e-14)
    result = solve_schwarzschild_scattering(plan, jnp.asarray(2.0e-2))

    assert not bool(result.evidence.scattering_ledger_valid)
    assert not bool(result.converged)
    assert not bool(result.qualified)
    assert int(result.status) == int(
        SchwarzschildScatteringStatus.SCATTERING_LEDGER_UNRESOLVED
    )


def test_computed_scattering_requires_ingoing_horizon_boundary():
    with pytest.raises(ValueError, match="ingoing horizon"):
        _computed_scalar_scattering_plan(
            boundary=RadialBoundaryCondition("outgoing", "outgoing")
        )


def test_real_frequency_scattering_closes_flux_and_requires_explicit_qualification():
    mode = SeparatedMode(
        0,
        1,
        0,
        family="scattering",
        sector="scalar",
        background_id="test-schwarzschild-scattering",
    )
    plan = BlackHoleScatteringPlan(mode, 0.0, 1.0e-10, 1.0e-8)
    source_id = "independent-real-frequency-radial-solve"
    evidence = ScatteringQualificationEvidence(
        mode,
        True,
        radial_source_id=source_id,
        source_id="flux-reference",
    )
    result = solve_black_hole_scattering(
        plan,
        jnp.asarray(0.4),
        jnp.asarray(1.0 + 0.0j),
        jnp.asarray(np.sqrt(0.7) + 0.0j),
        jnp.asarray(np.sqrt(0.3) + 0.0j),
        jnp.asarray(0.0 + 0.0j),
        jnp.asarray(0.5),
        radial_source_id=source_id,
        qualification=evidence,
    )

    np.testing.assert_allclose(np.asarray(result.flux_residual), 0.0, atol=2.0e-8)
    np.testing.assert_allclose(np.asarray(result.graybody_factor), 0.3, atol=2.0e-8)
    assert bool(result.successful)
    assert bool(result.qualified)
    assert bool(result.derivative_valid)
    assert not bool(result.superradiant)
    assert int(result.superradiance_status) == int(SuperradianceStatus.NONSUPERRADIANT)
    assert result.flux_normalization == (
        "scalar-killing-energy-unit-incoming-at-infinity"
    )

    unresolved = solve_black_hole_scattering(
        plan,
        jnp.asarray(0.4),
        jnp.asarray(1.0 + 0.0j),
        jnp.asarray(np.sqrt(0.7) + 0.0j),
        jnp.asarray(np.sqrt(0.3) + 0.0j),
        jnp.asarray(0.0 + 0.1j),
        jnp.asarray(0.5),
        radial_source_id=source_id,
    )
    assert not bool(unresolved.converged)
    assert not bool(unresolved.qualified)
    assert int(unresolved.status) == int(
        BlackHoleScatteringStatus.WRONSKIAN_NOT_CONVERGED
    )


def test_kerr_superradiance_preserves_negative_signed_absorption():
    mode = SeparatedMode(
        0,
        1,
        1,
        family="scattering",
        sector="scalar",
        background_id="test-kerr-scattering",
    )
    plan = BlackHoleScatteringPlan(mode, 0.4, 1.0e-10, 1.0e-8)
    source_id = "independent-superradiant-radial-solve"
    result = solve_black_hole_scattering(
        plan,
        jnp.asarray(0.2),
        jnp.asarray(1.0 + 0.0j),
        jnp.asarray(np.sqrt(1.1) + 0.0j),
        jnp.asarray(np.sqrt(0.1) + 0.0j),
        jnp.asarray(0.0 + 0.0j),
        jnp.asarray(-0.5),
        radial_source_id=source_id,
        qualification=ScatteringQualificationEvidence(
            mode,
            True,
            radial_source_id=source_id,
            source_id="superradiant-flux-reference",
        ),
    )

    np.testing.assert_allclose(np.asarray(result.flux_residual), 0.0, atol=2.0e-8)
    np.testing.assert_allclose(np.asarray(result.horizon_flux), -0.02, atol=2.0e-8)
    np.testing.assert_allclose(np.asarray(result.graybody_factor), -0.1, atol=2.0e-8)
    np.testing.assert_allclose(np.asarray(result.amplification_factor), 0.1, atol=2.0e-8)
    assert bool(result.superradiant)
    assert bool(result.successful)
    assert bool(result.qualified)
    assert int(result.superradiance_status) == int(SuperradianceStatus.SUPERRADIANT)
    assert int(result.status) == int(BlackHoleScatteringStatus.SUCCESS)
