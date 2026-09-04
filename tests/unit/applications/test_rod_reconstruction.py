from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.solid_mechanics._rod_dynamics import prepare_rod, RodPlan
from phydrax.applications.solid_mechanics._rod_reconstruction import (
    compare_reduced_rod_discretizations,
    evaluate_rod_reconstruction,
    prepare_rod_reconstruction,
    RodFrameQueryPlan,
    RodReconstructionPlan,
)
from phydrax.applications.solid_mechanics._rod_reduced_basis import RodStrainBasisPlan
from phydrax.applications.solid_mechanics._rod_reduction import (
    prepare_reduced_rod,
    ReducedRodPlan,
    ReducedRodState,
)
from phydrax.metrix import QuaternionPoseStateGeometry


_POSE = QuaternionPoseStateGeometry(convention="body", tolerance=1.0e-9)


def _rod(segment_count: int = 6):
    dtype = jnp.float64
    nodes = jnp.linspace(0.0, 1.0, segment_count + 1, dtype=dtype)
    positions = jnp.stack((jnp.zeros_like(nodes), jnp.zeros_like(nodes), nodes), axis=-1)
    segment_ids = jnp.stack(
        (
            jnp.arange(segment_count, dtype=jnp.int32),
            jnp.arange(1, segment_count + 1, dtype=jnp.int32),
        ),
        axis=-1,
    )
    return prepare_rod(
        RodPlan(
            segment_ids,
            positions,
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (segment_count, 3, 3)),
            jnp.ones((segment_count + 1,), dtype=dtype),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((0.2, 0.3, 0.4), dtype=dtype)),
                (segment_count, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((20.0, 25.0, 100.0), dtype=dtype)),
                (segment_count, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((4.0, 5.0, 7.0), dtype=dtype)),
                (segment_count - 1, 3, 3),
            ),
        )
    )


def _reduction(basis: RodStrainBasisPlan, *, rod=None):
    prepared_rod = _rod() if rod is None else rod
    return prepare_reduced_rod(
        prepared_rod,
        ReducedRodPlan(
            basis,
            base_policy="fixed",
            fixed_base_position=jnp.asarray((0.3, -0.2, 0.1), dtype=jnp.float64),
            fixed_base_orientation=jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float64),
        ),
    )


def _pcs_reduction(*, breakpoints=(0.0, 1.0)):
    basis = RodStrainBasisPlan.piecewise_constant(
        jnp.asarray(breakpoints, dtype=jnp.float64),
        dimension=3,
        component_scales=jnp.ones((6,), dtype=jnp.float64),
    )
    return _reduction(basis)


def _gvs_reduction():
    basis = RodStrainBasisPlan.shifted_legendre(
        (0, 3),
        dimension=3,
        components=("nu_z", "kappa_y"),
        component_scales=jnp.asarray((0.2, 1.0), dtype=jnp.float64),
        quadrature_order=6,
    )
    return _reduction(basis)


def _prepared(reduction, queries, *, refinement=1, tolerance=1.0e-6, method="auto"):
    return prepare_rod_reconstruction(
        reduction,
        RodReconstructionPlan(
            RodFrameQueryPlan(jnp.asarray(queries, dtype=jnp.float64)),
            method=method,
            refinement=refinement,
            quadrature_tolerance=tolerance,
        ),
    )


def _rotate(quaternion, vector):
    imaginary = quaternion[1:]
    doubled_cross = 2.0 * jnp.cross(imaginary, vector)
    return vector + quaternion[0] * doubled_cross + jnp.cross(imaginary, doubled_cross)


def test_constant_material_strain_is_the_exact_se3_piece_exponential():
    reduction = _pcs_reduction()
    reconstruction = _prepared(reduction, (0.0, 0.37, 1.0))
    coefficients = jnp.asarray((0.08, -0.04, 0.12, 0.17, -0.11, 0.29), dtype=jnp.float64)
    state = ReducedRodState(coefficients, jnp.zeros_like(coefficients))

    evaluation = evaluate_rod_reconstruction(reconstruction, state)
    base_pose = jnp.concatenate((reduction.base_orientation, reduction.base_position))
    total_material_strain = coefficients + jnp.asarray(
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=jnp.float64
    )
    analytic_endpoint = _POSE.retract(base_pose, total_material_strain)

    assert evaluation.valid
    assert evaluation.quadrature_evidence.method == "pcs"
    assert evaluation.quadrature_evidence.maximum_scaled_local_error == 0.0
    assert jnp.allclose(evaluation.poses[-1], analytic_endpoint, atol=2.0e-11)
    assert evaluation.interpretation.startswith("observational/reference")
    assert evaluation.native_discrepancy.finite
    assert evaluation.native_discrepancy.maximum_stretch_shear_error < 2.0e-10
    assert evaluation.native_discrepancy.maximum_bend_twist_error < 2.0e-10


def test_variable_gvs_cf4_reconstruction_converges_under_fixed_refinement():
    reduction = _gvs_reduction()
    queries = (0.0, 0.17, 0.43, 0.79, 1.0)
    state = ReducedRodState(
        jnp.asarray((0.12, 0.31, -0.27, 0.23, -0.18), dtype=jnp.float64),
        jnp.asarray((-0.08, 0.19, 0.11, -0.16, 0.14), dtype=jnp.float64),
    )
    coarse = _prepared(reduction, queries, refinement=1, tolerance=1.0)
    medium = _prepared(reduction, queries, refinement=2, tolerance=1.0)
    fine = _prepared(reduction, queries, refinement=4, tolerance=1.0)

    coarse_evaluation = coarse.evaluate(state)
    medium_evaluation = medium.evaluate(state)
    fine_evaluation = fine.evaluate(state)
    comparison = compare_reduced_rod_discretizations(
        coarse_evaluation, medium_evaluation, fine_evaluation
    )

    assert coarse_evaluation.quadrature_evidence.method == "gvs"
    assert comparison.evidence.valid
    assert comparison.evidence.observed_order_supported
    assert (
        comparison.medium_fine.maximum_scaled_se3_log
        < comparison.coarse_medium.maximum_scaled_se3_log
    )
    assert comparison.observed_order.scaled_se3_log > 3.5
    assert comparison.medium_fine.maximum_scaled_frame_jvp < (
        comparison.coarse_medium.maximum_scaled_frame_jvp
    )


def test_half_open_routes_are_deterministic_and_close_only_the_final_endpoint():
    reduction = _pcs_reduction(breakpoints=(0.0, 0.4, 1.0))
    queries = (1.0, 0.4, 0.0, 0.399, 0.73)
    first = _prepared(reduction, queries)
    second = _prepared(reduction, queries)
    state = ReducedRodState(
        jnp.zeros((12,), dtype=jnp.float64),
        jnp.zeros((12,), dtype=jnp.float64),
    )

    evaluation = first.evaluate(state)

    assert first.route_id == second.route_id
    assert jnp.array_equal(
        evaluation.domain_evidence.route_indices,
        jnp.asarray((1, 1, 0, 0, 1), dtype=jnp.int32),
    )
    assert evaluation.domain_evidence.half_open_routing
    assert evaluation.domain_evidence.final_endpoint_closed
    assert evaluation.domain_evidence.valid
    assert jnp.array_equal(evaluation.arc_lengths, jnp.asarray(queries))


def test_body_world_origin_and_frame_velocities_obey_moment_arm_identity_and_duality():
    reduction = _gvs_reduction()
    reconstruction = _prepared(
        reduction, (0.11, 0.37, 0.82, 1.0), refinement=3, tolerance=1.0
    )
    coefficients = jnp.asarray((0.07, 0.21, -0.17, 0.13, -0.09), dtype=jnp.float64)
    rates = jnp.asarray((-0.12, 0.18, 0.08, -0.14, 0.11), dtype=jnp.float64)
    state = ReducedRodState(coefficients, rates)
    evaluation = reconstruction.evaluate(state)

    moment_arm_velocity = evaluation.world_origin_velocities[:, :3] + jnp.cross(
        evaluation.world_origin_velocities[:, 3:], evaluation.positions
    )
    body_frame_linear = jax.vmap(_rotate)(
        evaluation.orientations, evaluation.body_twists[:, :3]
    )
    body_frame_angular = jax.vmap(_rotate)(
        evaluation.orientations, evaluation.body_twists[:, 3:]
    )

    assert jnp.allclose(
        evaluation.frame_velocities[:, :3], moment_arm_velocity, atol=2.0e-10
    )
    assert jnp.allclose(
        evaluation.frame_velocities[:, :3], body_frame_linear, atol=2.0e-10
    )
    assert jnp.allclose(
        evaluation.frame_velocities[:, 3:], body_frame_angular, atol=2.0e-10
    )

    velocity_operator = reconstruction.frame_velocity_operator(coefficients)
    effort_pullback = reconstruction.frame_effort_pullback(coefficients)
    effort = jnp.asarray(
        (
            (0.4, -0.2, 0.7, -0.1, 0.3, 0.2),
            (-0.3, 0.6, 0.1, 0.5, -0.2, 0.4),
            (0.8, -0.4, 0.2, -0.3, 0.1, -0.5),
            (-0.2, 0.3, -0.6, 0.4, 0.7, -0.1),
        ),
        dtype=jnp.float64,
    )
    frame_velocity = velocity_operator.mv(rates)
    reduced_effort = effort_pullback.mv(effort)

    assert jnp.allclose(frame_velocity, evaluation.frame_velocities, atol=2.0e-10)
    assert jnp.vdot(effort, frame_velocity) == pytest.approx(
        jnp.vdot(reduced_effort, rates), rel=2.0e-11, abs=2.0e-11
    )


def test_domain_chart_quadrature_and_comparison_mismatches_reject():
    pcs_reduction = _pcs_reduction()
    with pytest.raises(ValueError, match="rod domain"):
        _prepared(pcs_reduction, (0.0, 1.01))

    gvs_reduction = _gvs_reduction()
    with pytest.raises(ValueError, match="piecewise-constant"):
        _prepared(gvs_reduction, (0.0, 1.0), method="pcs")

    chart_reconstruction = _prepared(pcs_reduction, (0.0, 1.0))
    chart_coefficients = jnp.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 200.0), dtype=jnp.float64)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
        match="rotation-by-pi|chart",
    ):
        invalid_pose = chart_reconstruction.pose(chart_coefficients)
        jax.block_until_ready(invalid_pose)

    quadrature_reconstruction = _prepared(
        gvs_reduction,
        (0.0, 0.37, 1.0),
        refinement=1,
        tolerance=1.0e-14,
    )
    variable_state = ReducedRodState(
        jnp.asarray((0.17, 0.8, -0.7, 0.6, -0.5), dtype=jnp.float64),
        jnp.zeros((5,), dtype=jnp.float64),
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="fixed-quadrature"):
        invalid_evaluation = quadrature_reconstruction.evaluate(variable_state)
        jax.block_until_ready(invalid_evaluation.poses)

    valid_state = ReducedRodState(
        jnp.asarray((0.05, 0.2, -0.1, 0.07, -0.04), dtype=jnp.float64),
        jnp.zeros((5,), dtype=jnp.float64),
    )
    first = _prepared(
        gvs_reduction, (0.0, 0.5, 1.0), refinement=1, tolerance=1.0
    ).evaluate(valid_state)
    other_queries = _prepared(
        gvs_reduction, (0.0, 0.4, 1.0), refinement=2, tolerance=1.0
    ).evaluate(valid_state)
    with pytest.raises(ValueError, match="query-plan"):
        compare_reduced_rod_discretizations(first, other_queries)


def test_two_level_comparison_marks_observed_order_unsupported():
    reduction = _gvs_reduction()
    state = ReducedRodState(
        jnp.asarray((0.09, 0.26, -0.19, 0.14, -0.08), dtype=jnp.float64),
        jnp.asarray((0.04, -0.12, 0.07, 0.1, -0.06), dtype=jnp.float64),
    )
    coarse = _prepared(
        reduction, (0.0, 0.3, 0.8, 1.0), refinement=1, tolerance=1.0
    ).evaluate(state)
    medium = _prepared(
        reduction, (0.0, 0.3, 0.8, 1.0), refinement=2, tolerance=1.0
    ).evaluate(state)

    comparison = compare_reduced_rod_discretizations(coarse, medium)

    assert comparison.evidence.valid
    assert not comparison.evidence.observed_order_supported
    assert comparison.medium_fine is None
    assert jnp.isnan(comparison.observed_order.scaled_se3_log)
