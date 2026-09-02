#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.atomistic._polarizable_force_field import (
    AngleAnglePotential,
    Buffered147Potential,
    ChargeFluxPotential,
    ChargePenetrationPotential,
    ChargeTransferPotential,
    DampedDispersionPotential,
    evaluate_polarizable_term,
    OutOfPlaneBendPotential,
    PauliRepulsionPotential,
    PolarizableForceFieldPlan,
    StretchBendPotential,
)
from phydrax.atomistic._polarization import (
    evaluate_implicit_polarization_jvp,
    evaluate_polarization,
    implicit_polarization_jvp,
    MultipolePMEPlan,
    PermanentMultipoleSiteData,
    polarization_energy,
    PolarizationEvaluation,
    PolarizationPlan,
    PolarizationPreconditionerKind,
    PolarizationPreconditionerPlan,
    PolarizationScaleData,
    PolarizationSolverKind,
    PolarizationState,
)


def _multipoles(charges, polarizabilities, *, damping=4.0):
    count = len(charges)
    return PermanentMultipoleSiteData(
        charges,
        np.zeros((count, 3)),
        np.zeros((count, 3, 3)),
        polarizabilities,
        np.full((count,), damping),
    )


def _two_site_case():
    positions = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    multipoles = _multipoles([1.0, -1.0], [0.1, 0.2], damping=10.0)
    return positions, multipoles


def test_small_analytic_induced_dipoles_and_jit():
    positions, multipoles = _two_site_case()
    plan = PolarizationPlan(maximum_iterations=12, tolerance=1.0e-6)
    prepared = plan.prepare(multipoles)
    state = prepared.solve(positions).state
    damping = 1.0 - np.exp(-10.0 * 2.0**3)
    field = 0.25 * damping
    coupling = 0.25 * damping
    matrix = np.asarray([[10.0, -coupling], [-coupling, 5.0]])
    expected_x = np.linalg.solve(matrix, np.asarray([field, field]))
    expected = np.zeros((2, 3))
    expected[:, 0] = expected_x
    assert bool(state.successful)
    np.testing.assert_allclose(state.induced_dipoles, expected, rtol=2.0e-5, atol=2.0e-7)
    compiled = jax.jit(lambda value: prepared.solve(value).state.induced_dipoles)(
        positions
    )
    np.testing.assert_allclose(compiled, expected, rtol=2.0e-5, atol=2.0e-7)


def test_pcg_tcg_parity_and_fixed_tcg_order():
    positions, multipoles = _two_site_case()
    pcg = (
        PolarizationPlan(
            maximum_iterations=12,
            tolerance=1.0e-6,
            solver_kind=PolarizationSolverKind.PCG,
        )
        .prepare(multipoles)
        .solve(positions)
    )
    tcg = (
        PolarizationPlan(
            maximum_iterations=12,
            tcg_order=4,
            tolerance=1.0e-6,
            solver_kind=PolarizationSolverKind.TCG,
        )
        .prepare(multipoles)
        .solve(positions)
    )
    assert bool(pcg.successful & tcg.successful)
    assert int(tcg.state.iterations) == 4
    np.testing.assert_allclose(
        tcg.state.induced_dipoles,
        pcg.state.induced_dipoles,
        rtol=2.0e-5,
        atol=2.0e-7,
    )


def test_polarizability_preconditioner_is_local_inverse_diagonal():
    positions, multipoles = _two_site_case()
    operator = PolarizationPlan().operator.prepare(multipoles)
    residual = jnp.asarray([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
    local = (
        PolarizationPreconditionerPlan(PolarizationPreconditionerKind.POLARIZABILITY)
        .prepare(operator)
        .apply(residual)
    )
    identity = (
        PolarizationPreconditionerPlan(PolarizationPreconditionerKind.IDENTITY)
        .prepare(operator)
        .apply(residual)
    )
    assert bool(local.successful & identity.successful)
    np.testing.assert_allclose(
        local.value, residual * multipoles.polarizabilities[:, None]
    )
    np.testing.assert_allclose(identity.value, residual)
    local_solve = (
        PolarizationPlan(
            maximum_iterations=12,
            tolerance=1.0e-6,
            preconditioner_kind=PolarizationPreconditionerKind.POLARIZABILITY,
        )
        .prepare(multipoles)
        .solve(positions)
    )
    identity_solve = (
        PolarizationPlan(
            maximum_iterations=12,
            tolerance=1.0e-6,
            preconditioner_kind=PolarizationPreconditionerKind.IDENTITY,
        )
        .prepare(multipoles)
        .solve(positions)
    )
    assert bool(local_solve.successful & identity_solve.successful)
    np.testing.assert_allclose(
        local_solve.state.induced_dipoles,
        identity_solve.state.induced_dipoles,
        rtol=2.0e-5,
        atol=2.0e-7,
    )


def test_solver_convergence_and_force_validity_are_separate_gates():
    positions, multipoles = _two_site_case()
    plan = PolarizationPlan(
        maximum_iterations=1,
        tcg_order=1,
        tolerance=1.0,
        force_tolerance=1.0e-12,
        solver_kind="tcg",
    )
    state = plan.prepare(multipoles).solve(positions).state
    evaluation = evaluate_polarization(plan, positions, multipoles)
    assert bool(state.converged & state.successful)
    assert not bool(state.force_valid)
    assert not bool(evaluation.successful)
    assert bool(jnp.isnan(evaluation.energy))
    assert bool(jnp.all(jnp.isnan(evaluation.forces)))
    unstable_multipoles = _multipoles([1.0, -1.0], [10.0, 10.0], damping=10.0)
    unstable = (
        PolarizationPlan(maximum_iterations=4, tolerance=1.0e-6)
        .prepare(unstable_multipoles)
        .solve(positions)
    )
    assert bool(unstable.breakdown)
    assert not bool(unstable.successful)


def test_retained_result_types_preserve_keyword_construction():
    state = PolarizationState(
        induced_dipoles=jnp.zeros((1, 3)),
        residual=jnp.asarray(0.0),
        iterations=jnp.asarray(0),
        converged=jnp.asarray(True),
        successful=jnp.asarray(True),
        plan_id="manual",
    )
    evaluation = PolarizationEvaluation(
        energy=jnp.asarray(0.0),
        forces=jnp.zeros((1, 3)),
        state=state,
        successful=jnp.asarray(True),
    )
    assert bool(evaluation.successful)


def test_predictor_warm_start_reuses_fixed_shape_history():
    positions, multipoles = _two_site_case()
    prepared = PolarizationPlan(maximum_iterations=12, tolerance=1.0e-6).prepare(
        multipoles
    )
    cold = prepared.solve(positions)
    warm = prepared.solve(positions, predictor_state=cold.predictor_state)
    assert cold.predictor_state.history.shape == (2, 2, 3)
    assert int(cold.predictor_state.valid_count) == 1
    np.testing.assert_allclose(
        warm.initial_dipoles, cold.state.induced_dipoles, rtol=1.0e-6, atol=1.0e-8
    )
    assert int(warm.state.iterations) <= int(cold.state.iterations)
    assert bool(warm.successful)


def test_d_p_u_scaling_semantics_remain_distinct():
    positions, multipoles = _two_site_case()
    off_diagonal = np.ones((2, 2)) - np.eye(2)
    scaling = PolarizationScaleData(
        np.zeros((2, 2)), 0.5 * off_diagonal, 0.25 * off_diagonal
    )
    operator = PolarizationPlan().operator.prepare(multipoles, scaling=scaling)
    trial = jnp.asarray([[0.1, 0.0, 0.0], [-0.2, 0.0, 0.0]])
    result = operator.apply(positions, trial)
    assert bool(result.successful)
    np.testing.assert_allclose(result.d_field, 0.0, atol=1.0e-8)
    assert bool(jnp.max(jnp.abs(result.p_field)) > 0.0)
    assert bool(jnp.max(jnp.abs(result.u_field)) > 0.0)
    full = PolarizationScaleData.unscaled(2)
    full_result = (
        PolarizationPlan()
        .operator.prepare(multipoles, scaling=full)
        .apply(positions, trial)
    )
    np.testing.assert_allclose(result.p_field, 0.5 * full_result.p_field)
    np.testing.assert_allclose(result.u_field, 0.25 * full_result.u_field)


def test_zero_alpha_sources_and_fully_excluded_coincident_pairs_are_masked():
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    multipoles = _multipoles([0.0, 0.0], [0.2, 0.0])
    operator = PolarizationPlan().operator.prepare(multipoles)
    result = operator.apply(positions, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(result.u_field, 0.0, atol=1.0e-8)
    np.testing.assert_allclose(result.action[0], 0.0, atol=1.0e-8)
    residual = jnp.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    identity = (
        PolarizationPreconditionerPlan("identity").prepare(operator).apply(residual)
    )
    local = (
        PolarizationPreconditionerPlan("polarizability").prepare(operator).apply(residual)
    )
    np.testing.assert_allclose(identity.value, residual)
    np.testing.assert_allclose(
        local.value,
        residual * jnp.asarray([[0.2], [1.0]]),
    )

    excluded = PolarizationScaleData(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
    coincident = (
        PolarizationPlan()
        .operator.prepare(multipoles, scaling=excluded)
        .apply(jnp.zeros((2, 3)), jnp.zeros((2, 3)))
    )
    assert bool(coincident.successful)
    np.testing.assert_allclose(coincident.d_field, 0.0)
    np.testing.assert_allclose(coincident.p_field, 0.0)
    np.testing.assert_allclose(coincident.u_field, 0.0)


def test_envelope_force_and_implicit_jvp_match_finite_differences():
    positions, multipoles = _two_site_case()
    plan = PolarizationPlan(
        maximum_iterations=16, tolerance=1.0e-6, force_tolerance=2.0e-6
    )
    evaluation = evaluate_polarization(plan, positions, multipoles)
    assert bool(evaluation.successful)
    step = 2.0e-3
    direction = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    plus = polarization_energy(plan, positions + step * direction, multipoles)[0]
    minus = polarization_energy(plan, positions - step * direction, multipoles)[0]
    finite_difference = (plus - minus) / (2.0 * step)
    analytic = -jnp.sum(evaluation.forces * direction)
    np.testing.assert_allclose(analytic, finite_difference, rtol=3.0e-3, atol=2.0e-5)

    primal, tangent = implicit_polarization_jvp(plan, positions, direction, multipoles)
    plus_state = plan.prepare(multipoles).solve(positions + step * direction).state
    minus_state = plan.prepare(multipoles).solve(positions - step * direction).state
    finite_tangent = (plus_state.induced_dipoles - minus_state.induced_dipoles) / (
        2.0 * step
    )
    assert bool(jnp.all(jnp.isfinite(primal)) & jnp.all(jnp.isfinite(tangent)))
    implicit_result = evaluate_implicit_polarization_jvp(
        plan, positions, direction, multipoles
    )
    assert bool(implicit_result.successful)
    assert implicit_result.evidence.mode == "implicit"
    np.testing.assert_allclose(tangent, finite_tangent, rtol=4.0e-3, atol=2.0e-5)


def test_periodic_multipole_contract_is_bidirectional_and_fail_closed():
    positions, multipoles = _two_site_case()
    nonperiodic = PolarizationPlan().prepare(multipoles)
    with pytest.raises(ValueError, match="cell_vectors require"):
        nonperiodic.solve(positions, cell_vectors=jnp.eye(3) * 8.0)

    periodic_plan = PolarizationPlan(
        maximum_iterations=20,
        tolerance=1.0e-7,
        periodic_plan=MultipolePMEPlan((4, 4, 4), 0.5),
    )
    periodic = periodic_plan.prepare(multipoles)
    with pytest.raises(ValueError, match="requires cell_vectors"):
        periodic.solve(positions)
    valid_operator = periodic.operator.apply(
        positions,
        jnp.zeros_like(multipoles.dipoles),
        cell_vectors=jnp.eye(3) * 8.0,
    )
    assert bool(valid_operator.periodic_contract_valid & valid_operator.finite)
    skew_cell = jnp.asarray([[8.0, 0.0, 0.0], [7.2, 0.8, 0.0], [0.0, 0.0, 8.0]])
    skew_operator = periodic.operator.apply(
        positions,
        jnp.zeros_like(multipoles.dipoles),
        cell_vectors=skew_cell,
    )
    assert not bool(skew_operator.periodic_contract_valid)
    small_skew_operator = periodic.operator.apply(
        positions,
        jnp.zeros_like(multipoles.dipoles),
        cell_vectors=1.0e-5 * skew_cell,
    )
    assert not bool(small_skew_operator.periodic_contract_valid)
    trial = jnp.asarray([[0.1, 0.0, 0.0], [-0.2, 0.0, 0.0]])
    unscaled = PolarizationScaleData.unscaled(2)
    excluded = PolarizationScaleData(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
    periodic_full = periodic_plan.operator.prepare(multipoles, scaling=unscaled).apply(
        positions, trial, cell_vectors=jnp.eye(3) * 8.0
    )
    periodic_excluded = periodic_plan.operator.prepare(
        multipoles, scaling=excluded
    ).apply(positions, trial, cell_vectors=jnp.eye(3) * 8.0)
    real_full = (
        PolarizationPlan()
        .operator.prepare(multipoles, scaling=unscaled)
        .apply(positions, trial)
    )
    np.testing.assert_allclose(
        periodic_excluded.d_field - periodic_full.d_field,
        -real_full.d_field,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        periodic_excluded.p_field - periodic_full.p_field,
        -real_full.p_field,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        periodic_excluded.u_field - periodic_full.u_field,
        -real_full.u_field,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    singular = periodic.solve(positions, cell_vectors=jnp.zeros((3, 3)))
    assert not bool(singular.successful)
    evaluation = evaluate_polarization(
        periodic_plan, positions, multipoles, cell_vectors=jnp.zeros((3, 3))
    )
    assert not bool(evaluation.successful)
    assert bool(jnp.isnan(evaluation.energy))


def _advanced_terms():
    positions = jnp.asarray(
        [
            [1.0, 0.0, 0.2],
            [0.0, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [-0.4, 0.2, 1.1],
        ]
    )
    count = positions.shape[0]
    empty_angles = np.empty((0, 3), dtype=np.int32)
    terms = (
        Buffered147Potential(np.full(count, 0.55), np.full(count, 0.2)),
        ChargePenetrationPotential(
            [0.5, 0.4, 0.3, 0.2],
            [-0.4, -0.3, -0.2, -0.1],
            np.full(count, 1.6),
        ),
        ChargeTransferPotential(np.full(count, 0.3), np.full(count, 1.7)),
        ChargeFluxPotential(
            [0.2, -0.1, -0.1, 0.0],
            [[0, 1]],
            [0.03],
            [1.0],
            empty_angles,
            np.empty((0, 2)),
            np.empty((0,)),
        ),
        DampedDispersionPotential(
            np.full(count, 0.12),
            np.full(count, 1.8),
            c8=np.full(count, 0.04),
            c10=np.full(count, 0.01),
        ),
        PauliRepulsionPotential(np.full(count, 1.2), np.full(count, 2.0)),
        StretchBendPotential(count, [[0, 1, 2]], [[0.2, -0.1]], [[1.0, 1.1]], [1.4]),
        AngleAnglePotential(count, [[0, 1, 2, 3]], [0.15], [[1.4, 1.2]]),
        OutOfPlaneBendPotential(count, [[0, 1, 2, 3]], [0.4]),
    )
    return positions, terms


@pytest.mark.parametrize("term_index", range(9))
def test_advanced_energy_terms_are_invariant_and_forces_are_gradients(term_index):
    positions, terms = _advanced_terms()
    term = terms[term_index]
    evaluation = evaluate_polarizable_term(term, positions)
    translated = term.energy(positions + jnp.asarray([2.0, -1.0, 0.5]))
    rotation = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = term.energy(positions @ rotation.T)
    assert bool(evaluation.successful)
    np.testing.assert_allclose(translated, evaluation.energy, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(rotated, evaluation.energy, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(jnp.sum(evaluation.forces, axis=0), 0.0, atol=3.0e-5)

    direction = jnp.zeros_like(positions).at[0, 0].set(1.0)
    step = 2.0e-3
    finite_difference = (
        term.energy(positions + step * direction)
        - term.energy(positions - step * direction)
    ) / (2.0 * step)
    analytic = -evaluation.forces[0, 0]
    np.testing.assert_allclose(analytic, finite_difference, rtol=5.0e-3, atol=5.0e-5)


def test_tang_toennies_dispersion_is_stable_at_short_range():
    term = DampedDispersionPotential([0.2, 0.3], [2.0, 2.4])
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0e-4, 0.0, 0.0]])
    evaluation = evaluate_polarizable_term(term, positions)
    assert bool(evaluation.successful)
    assert bool(jnp.isfinite(evaluation.energy))
    assert bool(jnp.all(jnp.isfinite(evaluation.forces)))


def test_charge_flux_conserves_total_charge_and_force_field_is_coherent():
    positions, terms = _advanced_terms()
    charge_flux = terms[3]
    charges = charge_flux.charges(positions)
    np.testing.assert_allclose(
        jnp.sum(charges), jnp.sum(charge_flux.reference_charges), atol=1.0e-7
    )
    multipoles = _multipoles([0.2, -0.1, -0.1, 0.0], np.full(4, 0.03))
    polarization = PolarizationPlan(
        maximum_iterations=24, tolerance=1.0e-6, force_tolerance=2.0e-6
    )
    force_field = PolarizableForceFieldPlan(
        terms, polarization=polarization, force_balance_tolerance=2.0e-4
    ).prepare(multipoles=multipoles)
    evaluation = force_field.evaluate(positions)
    assert evaluation.term_energies.shape == (9,)
    assert bool(evaluation.successful)
    assert bool(evaluation.qualification.polarization_force_valid)
    np.testing.assert_allclose(jnp.sum(evaluation.forces, axis=0), 0.0, atol=2.0e-4)
