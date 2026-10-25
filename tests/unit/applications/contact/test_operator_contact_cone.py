#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.contact._articulated import build_delassus_operator
from phydrax.applications.contact._closure_sensitivity import (
    contact_cone_solution_jvp,
    contact_cone_solution_vjp,
    ContactConeTangent,
    prepare_contact_cone_sensitivity,
)
from phydrax.applications.contact._cone import (
    certify_contact_cone_impulse,
    contact_cone_numeric_revision_matches,
    contact_cone_result_is_certified,
    contact_cone_space,
    ContactBlockIdentity,
    ContactBlockLaw,
    ContactBlockLayout,
    ContactCertificatePolicy,
    ContactConeBranch,
    ContactConeProgram,
    ContactConeSolverPlan,
    dense_contact_response,
    local_block_contact_response,
    solve_contact_cone,
)
from phydrax.applications.contact._cone_advanced import (
    SAPContactSolverPlan,
    solve_contact_sap,
)
from phydrax.discretization.particle._articulated_mass import prepare_articulated_mass
from phydrax.linalg import DualSpace, FunctionLinearOperator
from phydrax.linalg.svd import DenseSVD, SVDSolvePolicy


_SOLVER = ContactConeSolverPlan(
    maximum_iterations=2000, absolute_tolerance=1.0e-13, relative_tolerance=0.0
)
_REGULARITY = SVDSolvePolicy(DenseSVD(algorithm="qr"))


def _policy(
    names: tuple[str, ...] = ("contact",),
    *,
    reference: Any = 1.0,
    impulse: Any = 1.0e-8,
    law: Any = 1.0e-8,
    slip: Any = 1.0e-8,
    work: Any = 1.0e-8,
) -> ContactCertificatePolicy:
    return ContactCertificatePolicy(
        names,
        reference_velocity=reference,
        impulse_tolerance=impulse,
        law_velocity_tolerance=law,
        slip_speed_tolerance=slip,
        complementarity_work_tolerance=work,
        friction_work_tolerance=work,
    )


def _program(
    matrix: Any,
    free: Any,
    *,
    dynamic_friction: Any = 0.0,
    static_friction: Any | None = None,
    normal_bias: Any = 0.0,
    compliance: Any = 0.0,
    restitution: Any = 0.0,
    layout: ContactBlockLayout | None = None,
    policy: ContactCertificatePolicy | None = None,
    keys: Any | None = None,
    valid: Any | None = None,
    program_id: str = "operator-cone-test",
) -> ContactConeProgram:
    free_ = jnp.asarray(free, dtype=jnp.float64)
    count, width = free_.shape
    policy_ = _policy() if policy is None else policy
    space = contact_cone_space(count, width, jnp.float64)
    return ContactConeProgram(
        dense_contact_response(jnp.asarray(matrix, dtype=jnp.float64), space),
        free_,
        compliance,
        dynamic_friction,
        jnp.arange(count, dtype=jnp.int64) if keys is None else jnp.asarray(keys),
        jnp.ones((count,), dtype=jnp.bool_) if valid is None else jnp.asarray(valid),
        ContactBlockLayout.coulomb(count, width, family_count=policy_.family_count)
        if layout is None
        else layout,
        policy_,
        program_id,
        static_friction=static_friction,
        normal_bias=normal_bias,
        restitution=restitution,
    )


def _coupled_matrix() -> np.ndarray:
    matrix = np.eye(6)
    matrix[0, 0] = 2.0
    matrix[3, 3] = 1.0
    matrix[0, 3] = matrix[3, 0] = 0.5
    return matrix


def test_dense_and_matrix_free_actions_agree_with_coupled_normals() -> None:
    matrix = _coupled_matrix()
    free = np.asarray(((-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    dense = _program(matrix, free)
    space = contact_cone_space(2, 3, jnp.float64)
    flat = jnp.asarray(matrix)

    def action(impulse: Any) -> Any:
        return (flat @ impulse.reshape((-1,))).reshape(impulse.shape)

    operator = FunctionLinearOperator(
        action,
        source=DualSpace(space),
        target=space,
        transpose_action=action,
        operator_id="coupled-matrix-free-response",
    )
    matrix_free = eqx.tree_at(lambda program: program.response, dense, operator)

    expected_normal = np.linalg.solve(np.asarray(((2.0, 0.5), (0.5, 1.0))), (1.0, 1.0))
    for program in (dense, matrix_free):
        result = solve_contact_cone(program, solver=_SOLVER)
        assert bool(result.successful)
        assert bool(contact_cone_result_is_certified(program, result))
        np.testing.assert_allclose(result.impulse[:, 0], expected_normal, atol=1.0e-10)
        np.testing.assert_allclose(result.impulse[:, 1:], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(result.contact_law_velocity[:, 0], 0.0, atol=1.0e-9)
        np.testing.assert_array_equal(
            result.branch, np.full((2,), int(ContactConeBranch.FRICTIONLESS_BOUNDARY))
        )


def test_static_and_dynamic_coulomb_branches_and_physical_certificates() -> None:
    sticking_program = _program(
        np.eye(2), ((-1.0, 0.2),), dynamic_friction=0.3, static_friction=0.5
    )
    sliding_program = _program(
        np.eye(2), ((-1.0, 1.0),), dynamic_friction=0.3, static_friction=0.5
    )
    sticking = solve_contact_cone(sticking_program, solver=_SOLVER)
    sliding = solve_contact_cone(sliding_program, solver=_SOLVER)

    assert bool(sticking.successful) and bool(sliding.successful)
    np.testing.assert_allclose(sticking.impulse, ((1.0, -0.2),), atol=1.0e-10)
    np.testing.assert_allclose(sliding.impulse, ((1.0, -0.3),), atol=1.0e-10)
    np.testing.assert_allclose(sliding.contact_law_velocity, ((0.0, 0.7),), atol=1.0e-9)
    assert int(sticking.branch[0]) == int(ContactConeBranch.STICKING)
    assert int(sliding.branch[0]) == int(ContactConeBranch.SLIDING)
    certificate = sliding.evidence.certificate
    assert bool(certificate.complementarity_valid)
    assert bool(certificate.friction_law_valid)
    assert float(certificate.physical_tangential_work) < 0.0
    assert float(certificate.complementarity_work[0]) <= 1.0e-8


def test_restitution_bias_separates_law_and_physical_normal_velocity() -> None:
    inverse_effective_mass = 1.0 / 2.0 + 1.0 / 3.0
    restitution = 0.25
    closing = -1.2
    program = _program(
        np.asarray(((inverse_effective_mass, 0.0), (0.0, 0.0))),
        ((closing, 0.0),),
        normal_bias=restitution * min(closing, 0.0),
        restitution=restitution,
    )
    result = solve_contact_cone(program, solver=_SOLVER)

    expected = -(1.0 + restitution) * closing / inverse_effective_mass
    assert bool(result.successful)
    np.testing.assert_allclose(result.impulse[0, 0], expected, rtol=1.0e-10)
    np.testing.assert_allclose(result.contact_law_velocity[0, 0], 0.0, atol=1.0e-9)
    np.testing.assert_allclose(
        result.post_relative_velocity[0, 0], -restitution * closing, rtol=1.0e-9
    )


def test_compliance_enters_the_law_but_not_the_physical_velocity() -> None:
    program = _program(np.eye(2), ((-1.0, 0.0),), compliance=((0.5, 0.0),))
    result = solve_contact_cone(program, solver=_SOLVER)

    assert bool(result.successful)
    np.testing.assert_allclose(result.impulse[0, 0], 1.0 / 1.5, rtol=1.0e-10)
    np.testing.assert_allclose(result.contact_law_velocity[0, 0], 0.0, atol=1.0e-9)
    np.testing.assert_allclose(
        result.post_relative_velocity[0, 0], 1.0 / 1.5 - 1.0, rtol=1.0e-9
    )


def test_rank_deficient_psd_response_is_solved_and_certified() -> None:
    matrix = np.zeros((4, 4))
    matrix[0, 0] = matrix[0, 2] = matrix[2, 0] = matrix[2, 2] = 1.0
    program = _program(matrix, ((-1.0, 0.0), (-1.0, 0.0)))
    result = solve_contact_cone(program, solver=_SOLVER)

    assert bool(result.successful)
    np.testing.assert_allclose(jnp.sum(result.impulse[:, 0]), 1.0, atol=1.0e-9)
    assert bool(jnp.all(result.impulse[:, 0] >= -1.0e-12))
    np.testing.assert_allclose(result.contact_law_velocity[:, 0], 0.0, atol=1.0e-9)


def test_separating_contact_keeps_zero_impulse() -> None:
    program = _program(np.eye(3), ((0.5, 0.1, -0.2),), dynamic_friction=0.4)
    result = solve_contact_cone(program, solver=_SOLVER)

    assert bool(result.successful)
    np.testing.assert_array_equal(result.impulse, 0.0)
    assert int(result.branch[0]) == int(ContactConeBranch.SEPARATING)


def test_solver_continues_past_loose_iteration_tolerance_until_certified() -> None:
    free = np.asarray(((-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    program = _program(_coupled_matrix(), free)
    loose = ContactConeSolverPlan(
        maximum_iterations=2000, absolute_tolerance=1.0e-2, relative_tolerance=0.0
    )
    starved = ContactConeSolverPlan(
        maximum_iterations=1, absolute_tolerance=1.0e-2, relative_tolerance=0.0
    )

    result = solve_contact_cone(program, solver=loose)
    refused = solve_contact_cone(program, solver=starved)

    # The 1e-2 residual control alone would stop far above the 1e-8 law budget.
    assert bool(result.evidence.successful)
    assert float(jnp.max(result.evidence.certificate.law_normal_defect)) <= 1.0e-8
    expected = np.linalg.solve(np.asarray(((2.0, 0.5), (0.5, 1.0))), (1.0, 1.0))
    np.testing.assert_allclose(result.impulse[:, 0], expected)
    assert not bool(refused.evidence.converged)
    assert not bool(refused.evidence.successful)
    np.testing.assert_array_equal(refused.impulse, np.zeros((2, 3)))


def test_large_impulse_with_small_gap_rate_fails_complementarity_work() -> None:
    program = _program(np.eye(2), ((-1.0e6 + 1.0e-6, 0.0),))
    certificate = certify_contact_cone_impulse(program, jnp.asarray(((1.0e6, 0.0),)))

    assert bool(certificate.law_normal_valid)
    assert bool(certificate.normal_impulse_valid)
    assert not bool(certificate.complementarity_valid)
    assert not bool(certificate.successful)


def test_unit_rescaling_preserves_impulses_and_scales_velocities() -> None:
    factor = 1000.0
    base = _program(
        np.asarray(((1.0, 0.3, 0.0), (0.3, 1.0, 0.0), (0.0, 0.0, 1.0))),
        ((-1.0, 0.05, 0.0),),
        dynamic_friction=0.4,
        static_friction=0.6,
    )
    scaled = _program(
        factor * np.asarray(((1.0, 0.3, 0.0), (0.3, 1.0, 0.0), (0.0, 0.0, 1.0))),
        ((-factor, 0.05 * factor, 0.0),),
        dynamic_friction=0.4,
        static_friction=0.6,
        policy=_policy(
            reference=factor,
            law=1.0e-8 * factor,
            slip=1.0e-8 * factor,
            work=1.0e-8 * factor,
        ),
    )
    first = solve_contact_cone(base, solver=_SOLVER)
    second = solve_contact_cone(scaled, solver=_SOLVER)

    assert bool(first.successful) and bool(second.successful)
    np.testing.assert_allclose(second.impulse, first.impulse, rtol=1.0e-9, atol=1.0e-12)
    np.testing.assert_allclose(
        second.post_relative_velocity,
        factor * first.post_relative_velocity,
        rtol=1.0e-9,
        atol=1.0e-9,
    )


def test_stale_operator_law_order_and_tolerance_bindings_refuse() -> None:
    program = _program(
        _coupled_matrix(),
        ((-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        dynamic_friction=0.2,
    )
    result = solve_contact_cone(program, solver=_SOLVER)
    assert bool(contact_cone_numeric_revision_matches(program, result))

    changed_operator = eqx.tree_at(
        lambda value: value.response,
        program,
        dense_contact_response(2.0 * _coupled_matrix(), program.contact_space),
    )
    reordered = eqx.tree_at(lambda value: value.route_keys, program, jnp.asarray((1, 0)))
    friction = eqx.tree_at(
        lambda value: value.dynamic_friction, program, jnp.asarray((0.2, 0.3))
    )
    tolerance = eqx.tree_at(
        lambda value: value.certificate.impulse_tolerance,
        program,
        jnp.asarray((2.0e-8,)),
    )
    for stale in (changed_operator, reordered, friction, tolerance):
        assert not bool(contact_cone_numeric_revision_matches(stale, result))
        assert not bool(contact_cone_result_is_certified(stale, result))


def test_nonfinite_response_fails_closed() -> None:
    matrix = np.eye(2)
    matrix[0, 0] = np.nan
    program = _program(matrix, ((-1.0, 0.0),))
    result = solve_contact_cone(program, solver=_SOLVER)

    assert not bool(result.successful)
    np.testing.assert_array_equal(result.impulse, 0.0)
    np.testing.assert_array_equal(result.post_relative_velocity, program.free_velocity)


def test_unavailable_material_and_invalid_routes() -> None:
    program = _program(np.eye(4), ((-1.0, 0.0), (-1.0, 0.0)), valid=(True, False))
    padded = solve_contact_cone(program, solver=_SOLVER)
    unavailable = eqx.tree_at(
        lambda value: value.mechanical_available, program, jnp.asarray((False, True))
    )
    refused = solve_contact_cone(unavailable, solver=_SOLVER)

    assert bool(padded.successful)
    np.testing.assert_array_equal(padded.impulse[1], 0.0)
    assert not bool(refused.successful)
    assert not bool(refused.evidence.certificate.material_law_complete)


def test_mixed_contact_and_joint_stop_share_one_coupled_solve() -> None:
    policy = _policy(("contact", "hinge-stop"), reference=(1.0, 2.0))
    layout = ContactBlockLayout.mixed(1, (1,), family_count=2)
    matrix = np.zeros((6, 6))
    matrix[:3, :3] = np.eye(3)
    matrix[3, 3] = 2.0
    matrix[0, 3] = matrix[3, 0] = 0.5
    program = _program(
        matrix,
        ((-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        dynamic_friction=(0.5, 0.0),
        layout=layout,
        policy=policy,
    )
    result = solve_contact_cone(program, solver=_SOLVER)

    expected = np.linalg.solve(np.asarray(((1.0, 0.5), (0.5, 2.0))), (1.0, 1.0))
    assert bool(result.successful)
    np.testing.assert_allclose(result.impulse[:, 0], expected, atol=1.0e-10)
    np.testing.assert_array_equal(result.impulse[1, 1:], 0.0)
    assert int(layout.law[1]) == int(ContactBlockLaw.NORMAL_ONLY)
    assert int(layout.identity[1]) == int(ContactBlockIdentity.JOINT_STOP)
    assert int(result.branch[1]) == int(ContactConeBranch.NORMAL_ACTIVE)
    assert bool(result.evidence.certificate.padding_valid)
    with pytest.raises(ValueError, match="Normal-only blocks must pad"):
        ContactBlockLayout(
            (int(ContactBlockLaw.NORMAL_ONLY),),
            (int(ContactBlockIdentity.JOINT_STOP),),
            ((True, True, False),),
            (0,),
            family_count=1,
        )


def test_alternative_solver_shares_the_certificate() -> None:
    program = _program(
        _coupled_matrix(), ((-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), dynamic_friction=0.2
    )
    result = solve_contact_sap(
        program, solver=SAPContactSolverPlan(maximum_iterations=4000, tolerance=1.0e-12)
    )
    reference = solve_contact_cone(program, solver=_SOLVER)

    assert bool(result.successful)
    assert bool(contact_cone_result_is_certified(program, result))
    np.testing.assert_allclose(result.impulse, reference.impulse, atol=1.0e-9)


def test_local_block_response_jit_and_vmap_match_eager() -> None:
    policy = _policy()
    blocks = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    program = ContactConeProgram(
        local_block_contact_response(blocks),
        jnp.asarray(((-1.0, 0.2), (-0.5, 1.0))),
        0.0,
        0.3,
        jnp.asarray((3, 9)),
        jnp.asarray((True, True)),
        ContactBlockLayout.coulomb(2, 2),
        policy,
        "local-block-jit",
        static_friction=0.5,
    )
    eager = solve_contact_cone(program, solver=_SOLVER)
    compiled = eqx.filter_jit(solve_contact_cone)(program, solver=_SOLVER)

    def solve_free(free: Any) -> Any:
        changed = eqx.tree_at(lambda value: value.free_velocity, program, free)
        return solve_contact_cone(changed, solver=_SOLVER).impulse

    batch = jnp.stack((program.free_velocity, 2.0 * program.free_velocity))
    mapped = eqx.filter_vmap(solve_free)(batch)

    np.testing.assert_allclose(compiled.impulse, eager.impulse, atol=1.0e-12)
    np.testing.assert_allclose(mapped[0], eager.impulse, atol=1.0e-12)
    np.testing.assert_allclose(mapped[1], solve_free(batch[1]), atol=1.0e-12)


_STICKING_MATRIX = np.asarray(
    (
        (2.0, 0.1, 0.0, 0.4, 0.0, 0.0),
        (0.1, 1.5, 0.0, 0.0, 0.2, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.1),
        (0.4, 0.0, 0.0, 1.8, 0.0, 0.0),
        (0.0, 0.2, 0.0, 0.0, 1.2, 0.0),
        (0.0, 0.0, 0.1, 0.0, 0.0, 1.1),
    )
)


def _sticking_coupled_program(free: Any) -> ContactConeProgram:
    return _program(_STICKING_MATRIX, free, dynamic_friction=0.6, static_friction=0.8)


def test_fixed_branch_jvp_matches_finite_differences_and_analytic_sliding() -> None:
    sliding = _program(
        np.eye(2), ((-1.0, 1.0),), dynamic_friction=0.3, static_friction=0.5
    )
    result = solve_contact_cone(sliding, solver=_SOLVER)
    tangent = ContactConeTangent(
        sliding,
        free_velocity=((0.1, 0.2),),
        dynamic_friction=(0.1,),
    )
    derivative = contact_cone_solution_jvp(
        sliding, result, tangent, regularity=_REGULARITY
    )

    assert bool(derivative.evidence.successful)
    np.testing.assert_allclose(
        derivative.impulse_tangent, ((-0.1, 0.03 - 0.1),), atol=1.0e-9
    )

    free = jnp.asarray(((-1.0, 0.1, -0.05), (-0.8, -0.05, 0.02)))
    program = _sticking_coupled_program(free)
    solved = solve_contact_cone(program, solver=_SOLVER)
    direction = jnp.asarray(((0.02, -0.01, 0.03), (0.01, 0.02, -0.01)))
    jvp = contact_cone_solution_jvp(
        program,
        solved,
        ContactConeTangent(program, free_velocity=direction),
        regularity=_REGULARITY,
    )
    step = 1.0e-6
    plus = solve_contact_cone(
        _sticking_coupled_program(free + step * direction), solver=_SOLVER
    )
    minus = solve_contact_cone(
        _sticking_coupled_program(free - step * direction), solver=_SOLVER
    )

    assert bool(jvp.evidence.successful)
    np.testing.assert_allclose(
        jvp.impulse_tangent,
        (plus.impulse - minus.impulse) / (2.0 * step),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_response_action_tangent_matches_scaled_operator() -> None:
    free = ((-1.0, 0.2),)
    program = _program(np.eye(2), free, dynamic_friction=0.3, static_friction=0.5)
    result = solve_contact_cone(program, solver=_SOLVER)
    derivative = contact_cone_solution_jvp(
        program,
        result,
        ContactConeTangent(program, response_action=0.1 * result.impulse),
        regularity=_REGULARITY,
    )

    assert bool(derivative.evidence.successful)
    np.testing.assert_allclose(
        derivative.impulse_tangent, -0.1 * result.impulse, atol=1.0e-10
    )


def test_fixed_branch_vjp_is_the_adjoint_of_the_jvp() -> None:
    free = jnp.asarray(((-1.0, 0.1, -0.05), (-0.8, -0.05, 0.02)))
    program = _sticking_coupled_program(free)
    result = solve_contact_cone(program, solver=_SOLVER)
    sensitivity = prepare_contact_cone_sensitivity(
        program, result, regularity=_REGULARITY
    )
    key = jax.random.key(7)
    keys = jax.random.split(key, 6)
    tangent = ContactConeTangent(
        program,
        response_action=jax.random.normal(keys[0], (2, 3), dtype=jnp.float64),
        free_velocity=jax.random.normal(keys[1], (2, 3), dtype=jnp.float64),
        normal_bias=jax.random.normal(keys[2], (2,), dtype=jnp.float64),
        compliance=jnp.abs(jax.random.normal(keys[3], (2, 3), dtype=jnp.float64)),
        dynamic_friction=jax.random.normal(keys[4], (2,), dtype=jnp.float64),
    )
    cotangent = jax.random.normal(keys[5], (2, 3), dtype=jnp.float64)
    jvp = contact_cone_solution_jvp(program, result, tangent, sensitivity=sensitivity)
    vjp = contact_cone_solution_vjp(program, result, cotangent, sensitivity=sensitivity)

    static_cotangent = vjp.cotangent.static_friction
    dynamic_cotangent = vjp.cotangent.dynamic_friction
    assert static_cotangent is not None and dynamic_cotangent is not None
    left = jnp.sum(cotangent * jvp.impulse_tangent)
    right = (
        jnp.sum(vjp.cotangent.response_action * tangent.response_action)
        + jnp.sum(vjp.cotangent.free_velocity * tangent.free_velocity)
        + jnp.sum(vjp.cotangent.normal_bias * tangent.normal_bias)
        + jnp.sum(vjp.cotangent.compliance * tangent.compliance)
        + jnp.sum(static_cotangent * tangent.static_friction)
        + jnp.sum(dynamic_cotangent * tangent.dynamic_friction)
    )
    assert bool(jvp.evidence.successful) and bool(vjp.evidence.successful)
    np.testing.assert_allclose(left, right, rtol=1.0e-8, atol=1.0e-10)


def test_reused_sensitivity_refuses_stale_program_and_result() -> None:
    free = jnp.asarray(((-1.0, 0.1, -0.05), (-0.8, -0.05, 0.02)))
    program = _sticking_coupled_program(free)
    result = solve_contact_cone(program, solver=_SOLVER)
    sensitivity = prepare_contact_cone_sensitivity(
        program, result, regularity=_REGULARITY
    )
    tangent = ContactConeTangent(program, free_velocity=jnp.ones((2, 3)))

    fresh = contact_cone_solution_jvp(program, result, tangent, sensitivity=sensitivity)

    changed_law = eqx.tree_at(
        lambda value: value.dynamic_friction, program, jnp.asarray((0.5, 0.6))
    )
    stale_program = contact_cone_solution_jvp(
        changed_law, result, tangent, sensitivity=sensitivity
    )
    changed_operator = eqx.tree_at(
        lambda value: value.response,
        program,
        dense_contact_response(1.5 * _STICKING_MATRIX, program.contact_space),
    )
    other_result = solve_contact_cone(changed_operator, solver=_SOLVER)
    stale_result = contact_cone_solution_vjp(
        changed_operator,
        other_result,
        jnp.ones((2, 3)),
        sensitivity=sensitivity,
    )
    resolved = contact_cone_solution_jvp(
        changed_operator,
        other_result,
        ContactConeTangent(changed_operator, free_velocity=jnp.ones((2, 3))),
        regularity=_REGULARITY,
    )

    assert bool(fresh.evidence.successful)
    assert not bool(stale_program.evidence.successful)
    assert not bool(stale_program.evidence.primal_successful)
    np.testing.assert_array_equal(stale_program.impulse_tangent, 0.0)
    assert not bool(stale_result.evidence.successful)
    np.testing.assert_array_equal(stale_result.cotangent.free_velocity, 0.0)
    assert bool(resolved.evidence.successful)


def test_derivatives_refuse_apex_stale_and_inadmissible_geometry() -> None:
    apex = _program(np.eye(2), ((0.0, 0.0),), dynamic_friction=0.3)
    apex_result = solve_contact_cone(apex, solver=_SOLVER)
    apex_jvp = contact_cone_solution_jvp(
        apex,
        apex_result,
        ContactConeTangent(apex, free_velocity=((1.0, 1.0),)),
        regularity=_REGULARITY,
    )

    program = _program(
        np.eye(2), ((-1.0, 0.2),), dynamic_friction=0.3, static_friction=0.5
    )
    result = solve_contact_cone(program, solver=_SOLVER)
    stale = eqx.tree_at(
        lambda value: value.free_velocity, program, jnp.asarray(((-2.0, 0.2),))
    )
    stale_jvp = contact_cone_solution_jvp(
        stale,
        result,
        ContactConeTangent(stale, free_velocity=((1.0, 0.0),)),
        regularity=_REGULARITY,
    )
    inadmissible = contact_cone_solution_vjp(
        program,
        result,
        jnp.ones((1, 2)),
        regularity=_REGULARITY,
        geometry_admissible=(False,),
    )

    classification = apex_jvp.evidence.branch_classification
    assert classification is not None
    assert int(classification[0]) == int(ContactConeBranch.UNCLASSIFIED)
    for evidence in (apex_jvp.evidence, stale_jvp.evidence, inadmissible.evidence):
        assert not bool(evidence.successful)
    np.testing.assert_array_equal(apex_jvp.impulse_tangent, 0.0)
    np.testing.assert_array_equal(stale_jvp.impulse_tangent, 0.0)
    np.testing.assert_array_equal(inadmissible.cotangent.free_velocity, 0.0)


def test_derivatives_require_a_native_regularity_proof() -> None:
    program = _program(
        np.diag((1.0, 0.0, 0.0)), ((-1.0, 0.0, 0.0),), dynamic_friction=0.5
    )
    result = solve_contact_cone(program, solver=_SOLVER)
    tangent = ContactConeTangent(program, free_velocity=((0.1, 0.0, 0.0),))
    singular = contact_cone_solution_jvp(program, result, tangent, regularity=_REGULARITY)
    regular = _program(
        np.eye(2), ((-1.0, 0.2),), dynamic_friction=0.3, static_friction=0.5
    )
    regular_result = solve_contact_cone(regular, solver=_SOLVER)
    unproven = contact_cone_solution_jvp(
        regular,
        regular_result,
        ContactConeTangent(regular, free_velocity=((0.1, 0.0),)),
    )

    assert bool(result.successful)
    assert int(result.branch[0]) == int(ContactConeBranch.STICKING)
    assert bool(singular.evidence.branch_qualified)
    assert singular.evidence.jacobian_regular is not None
    assert not bool(singular.evidence.jacobian_regular)
    assert not bool(singular.evidence.successful)
    np.testing.assert_array_equal(singular.impulse_tangent, 0.0)
    assert unproven.evidence.jacobian_regular is not None
    assert not bool(unproven.evidence.jacobian_regular)
    assert not bool(unproven.evidence.successful)


def test_reused_sensitivity_reapplies_geometry_and_margin_admission() -> None:
    program = _program(
        np.eye(2), ((-1.0, 0.2),), dynamic_friction=0.3, static_friction=0.5
    )
    result = solve_contact_cone(program, solver=_SOLVER)
    sensitivity = prepare_contact_cone_sensitivity(
        program, result, regularity=_REGULARITY
    )
    tangent = ContactConeTangent(program, free_velocity=((0.1, 0.0),))

    admitted = contact_cone_solution_jvp(
        program, result, tangent, sensitivity=sensitivity
    )
    inadmissible = contact_cone_solution_jvp(
        program, result, tangent, sensitivity=sensitivity, geometry_admissible=(False,)
    )
    tight = contact_cone_solution_vjp(
        program,
        result,
        jnp.ones((1, 2)),
        sensitivity=sensitivity,
        margin_factor=float(sensitivity.fixed_branch.cone_margin) + 1.0,
    )

    assert bool(admitted.evidence.successful)
    assert not bool(inadmissible.evidence.branch_qualified)
    assert not bool(inadmissible.evidence.successful)
    np.testing.assert_array_equal(inadmissible.impulse_tangent, 0.0)
    assert not bool(tight.evidence.successful)
    with pytest.raises(ValueError, match="fixed when the sensitivity is prepared"):
        contact_cone_solution_jvp(
            program, result, tangent, sensitivity=sensitivity, regularity=_REGULARITY
        )


def test_refused_derivatives_never_execute_the_fixed_branch_solve() -> None:
    program = _program(
        np.eye(2), ((-1.0, 0.2),), dynamic_friction=0.3, static_friction=0.5
    )
    result = solve_contact_cone(program, solver=_SOLVER)
    tangent = ContactConeTangent(program, free_velocity=((0.1, 0.0),))
    proven = prepare_contact_cone_sensitivity(program, result, regularity=_REGULARITY)
    unproven = prepare_contact_cone_sensitivity(program, result)

    def poisoned(sensitivity: Any) -> Any:
        # A Jacobian whose every action fails at run time: any refused call that
        # still reached the GMRES or transpose solve would raise.
        def explode(direction: Any) -> Any:
            return eqx.error_if(
                direction,
                jnp.all(jnp.isfinite(direction)),
                "derivative operator executed",
            )

        jacobian = FunctionLinearOperator(
            explode,
            source=program.contact_space,
            target=program.contact_space,
            transpose_action=explode,
            operator_id="poisoned-fixed-branch-jacobian",
        )
        return eqx.tree_at(lambda value: value.jacobian, sensitivity, jacobian)

    stale = eqx.tree_at(lambda value: value.impulse, result, 1.01 * result.impulse)
    healthy = contact_cone_solution_jvp(program, result, tangent, sensitivity=proven)
    refusals = (
        contact_cone_solution_jvp(
            program, stale, tangent, sensitivity=poisoned(proven)
        ).evidence,
        contact_cone_solution_jvp(
            program, result, tangent, sensitivity=poisoned(unproven)
        ).evidence,
        contact_cone_solution_jvp(
            program,
            result,
            tangent,
            sensitivity=poisoned(proven),
            geometry_admissible=(False,),
        ).evidence,
        contact_cone_solution_vjp(
            program, stale, jnp.ones((1, 2)), sensitivity=poisoned(proven)
        ).evidence,
        contact_cone_solution_vjp(
            program, result, jnp.ones((1, 2)), sensitivity=poisoned(unproven)
        ).evidence,
    )

    assert bool(healthy.evidence.successful)
    assert bool(healthy.evidence.linear_successful)
    for evidence in refusals:
        assert not bool(evidence.linear_successful)
        assert not bool(evidence.successful)
    with pytest.raises(Exception, match="derivative operator executed"):
        contact_cone_solution_jvp(program, result, tangent, sensitivity=poisoned(proven))


def test_frictionless_coulomb_boundary_refuses_friction_derivatives() -> None:
    program = _program(np.eye(2), ((-1.0, 1.0),), dynamic_friction=0.0)
    result = solve_contact_cone(program, solver=_SOLVER)
    sensitivity = prepare_contact_cone_sensitivity(
        program, result, regularity=_REGULARITY
    )
    friction = contact_cone_solution_jvp(
        program,
        result,
        ContactConeTangent(program, dynamic_friction=(1.0,), static_friction=(1.0,)),
        sensitivity=sensitivity,
    )
    velocity = contact_cone_solution_jvp(
        program,
        result,
        ContactConeTangent(program, free_velocity=((-0.1, 0.0),)),
        sensitivity=sensitivity,
    )
    with_friction = contact_cone_solution_vjp(
        program, result, jnp.ones((1, 2)), sensitivity=sensitivity
    )
    without_friction = contact_cone_solution_vjp(
        program,
        result,
        jnp.ones((1, 2)),
        sensitivity=sensitivity,
        friction_cotangent=False,
    )

    assert int(result.branch[0]) == int(ContactConeBranch.FRICTIONLESS_BOUNDARY)
    assert friction.evidence.friction_boundary_refused is not None
    assert bool(friction.evidence.friction_boundary_refused)
    assert not bool(friction.evidence.successful)
    np.testing.assert_array_equal(friction.impulse_tangent, 0.0)
    assert bool(velocity.evidence.successful)
    np.testing.assert_allclose(velocity.impulse_tangent, ((0.1, 0.0),), atol=1.0e-10)
    assert not bool(with_friction.evidence.successful)
    assert bool(without_friction.evidence.successful)
    assert without_friction.cotangent.dynamic_friction is None
    assert without_friction.cotangent.static_friction is None
    np.testing.assert_allclose(
        without_friction.cotangent.free_velocity, ((-1.0, 0.0),), atol=1.0e-10
    )


def _prismatic_slider() -> Any:
    """Root plus one body of mass 2 sliding along the world x axis."""
    body_ids = jnp.asarray([10, 11], dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        body_ids, jnp.asarray([1.0, 2.0]), ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.stack((jnp.eye(3), jnp.eye(3))),
        fixed_mask=jnp.asarray([True, False]),
    ).prepare(particles)
    reference = bodies.kinematics(
        jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
        jnp.zeros((2, 3)),
    )
    graph = phx.discretization.RigidJointGraphPlan(
        prismatic=phx.discretization.PrismaticJointSetPlan(
            jnp.asarray([20]),
            body_ids[:1],
            body_ids[1:],
            jnp.asarray([[0.0, 0.0, 0.0]]),
            jnp.asarray([[1.0, 0.0, 0.0]]),
        )
    ).prepare(bodies, reference)
    return phx.discretization.ReducedArticulationPlan(
        10, jnp.asarray([20]), jnp.asarray([10]), jnp.asarray([11])
    ).prepare(graph, reference)


def test_true_dual_mass_artifact_response_and_binding_refusal() -> None:
    articulation = _prismatic_slider()
    q = jnp.asarray([0.3])
    mass = prepare_articulated_mass(articulation, q)
    space = contact_cone_space(1, 3, jnp.float64)

    def rows(velocity: Any) -> Any:
        return jnp.zeros((1, 3), dtype=velocity.dtype).at[0, 0].set(-velocity[0])

    def pullback(impulse: Any) -> Any:
        return -impulse[0, :1]

    velocity_operator = FunctionLinearOperator(
        rows,
        source=mass.velocity_space,
        target=space,
        transpose_action=pullback,
        operator_id="slider-wall-velocity",
    )
    response = build_delassus_operator(velocity_operator, mass.inverse_mass_operator)
    free_velocity = jnp.asarray([1.5])
    policy = _policy()
    program = ContactConeProgram(
        response,
        velocity_operator.mv(free_velocity),
        0.0,
        0.0,
        jnp.asarray((1,), dtype=jnp.int64),
        jnp.asarray((True,)),
        ContactBlockLayout.coulomb(1, 3),
        policy,
        "slider-wall",
        response_binding=mass.binding,
    )
    result = solve_contact_cone(program, solver=_SOLVER)

    assert bool(mass.successful)
    assert bool(result.successful)
    np.testing.assert_allclose(result.impulse[0, 0], 2.0 * 1.5, rtol=1.0e-10)
    np.testing.assert_allclose(result.post_relative_velocity[0, 0], 0.0, atol=1.0e-9)
    moved = prepare_articulated_mass(articulation, q + 0.1)
    stale = eqx.tree_at(lambda value: value.response_binding, program, moved.binding)
    assert bool(contact_cone_result_is_certified(program, result))
    assert not bool(contact_cone_numeric_revision_matches(stale, result))
