import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control.games._master_equation import (
    DISCRETE_EMPIRICAL_LAW_NEIGHBOR_TRANSFER_DIFFERENCE,
    FINITE_STATE_DISCRETE_MASTER_EQUATION_REFERENCE,
    FinitePopulationSimplexLattice,
    FiniteStateMasterEquationProblem,
    FiniteStateMasterEquationStatus,
    solve_finite_state_master_equation_reference,
)


def _one_state_problem(
    *,
    horizon=1,
    population_size=3,
    actions=("only",),
    running_cost=None,
    terminal_cost=None,
):
    running = (
        (lambda time, state, action, law, args: 0.0)
        if running_cost is None
        else running_cost
    )
    terminal = (lambda state, law, args: 0.0) if terminal_cost is None else terminal_cost
    return FiniteStateMasterEquationProblem(
        ("state",),
        actions,
        horizon,
        population_size,
        lambda time, state, action, law, args: jnp.asarray([1.0]),
        running,
        terminal,
        selector_id="declared-action-order:first-minimum",
        problem_id="one-state-master-equation",
    )


def test_simplex_enumeration_is_the_complete_exact_count_grid():
    lattice = FinitePopulationSimplexLattice(num_states=3, population_size=2)

    assert lattice.num_laws == 6
    np.testing.assert_array_equal(
        lattice.counts,
        [
            [0, 0, 2],
            [0, 1, 1],
            [0, 2, 0],
            [1, 0, 1],
            [1, 1, 0],
            [2, 0, 0],
        ],
    )
    np.testing.assert_allclose(jnp.sum(lattice.laws, axis=1), 1.0)
    assert lattice.index_of_counts([1, 1, 0]) == 4
    assert lattice.index_of_law([0.5, 0.5, 0.0]) == 4
    assert lattice.num_neighbor_transfers == 18
    with pytest.raises(ValueError, match="not an exact empirical law"):
        lattice.index_of_law([0.25, 0.75, 0.0])


def test_one_state_reduction_is_the_scalar_backward_recursion():
    problem = _one_state_problem(
        horizon=3,
        running_cost=lambda time, state, action, law, args: 2.0,
        terminal_cost=lambda state, law, args: 5.0,
    )

    result = solve_finite_state_master_equation_reference(problem)

    assert result.status == FiniteStateMasterEquationStatus.SUCCESS
    assert result.valid
    assert result.U.shape == (4, 1, 1)
    np.testing.assert_allclose(result.U[:, 0, 0], [11.0, 9.0, 7.0, 5.0])
    np.testing.assert_array_equal(result.selectors, 0)
    np.testing.assert_allclose(result.law_transition_table, 1.0)


def test_two_state_terminal_model_propagates_physical_and_empirical_laws_exactly():
    states = (0, 1)

    def transition(time, state, action, law, args):
        return jnp.asarray([0.0, 1.0]) if state == 0 else jnp.asarray([1.0, 0.0])

    problem = FiniteStateMasterEquationProblem(
        states,
        ("flip",),
        1,
        2,
        transition,
        lambda time, state, action, law, args: 0.0,
        lambda state, law, args: 10.0 * state + law[1],
        selector_id="only-action",
        problem_id="two-state-analytic-terminal-model",
    )

    result = solve_finite_state_master_equation_reference(problem)
    all_zero = problem.lattice.index_of_counts([2, 0])
    all_one = problem.lattice.index_of_counts([0, 2])

    assert result.valid
    assert result.U[0, 0, all_zero] == pytest.approx(11.0)
    assert result.U[0, 1, all_one] == pytest.approx(0.0)
    assert result.law_transition_table[0, all_zero, all_one] == pytest.approx(1.0)
    assert result.law_transition_table[0, all_one, all_zero] == pytest.approx(1.0)


def test_default_population_kernel_is_the_exact_multinomial_law():
    problem = FiniteStateMasterEquationProblem(
        (0, 1),
        ("randomize",),
        1,
        2,
        lambda time, state, action, law, args: jnp.asarray([0.5, 0.5]),
        lambda time, state, action, law, args: 0.0,
        lambda state, law, args: 0.0,
        selector_id="only-action",
        problem_id="two-draw-exact-multinomial",
    )

    result = solve_finite_state_master_equation_reference(problem)

    assert result.valid
    np.testing.assert_allclose(
        result.law_transition_table[0],
        np.tile(np.asarray([0.25, 0.5, 0.25]), (3, 1)),
    )
    assert result.aggregate_transition_mode == "exact-state-wise-multinomial"


def test_population_size_is_explicit_refinement_metadata():
    coarse = solve_finite_state_master_equation_reference(
        FiniteStateMasterEquationProblem(
            (0, 1),
            ("stay",),
            0,
            2,
            lambda time, state, action, law, args: jnp.asarray([1.0, 0.0]),
            lambda time, state, action, law, args: 0.0,
            lambda state, law, args: law[state],
            selector_id="only-action",
            problem_id="coarse-terminal-grid",
        )
    )
    fine = solve_finite_state_master_equation_reference(
        FiniteStateMasterEquationProblem(
            (0, 1),
            ("stay",),
            0,
            4,
            lambda time, state, action, law, args: jnp.asarray([1.0, 0.0]),
            lambda time, state, action, law, args: 0.0,
            lambda state, law, args: law[state],
            selector_id="only-action",
            problem_id="fine-terminal-grid",
        )
    )

    assert coarse.evidence.population_size == 2
    assert coarse.evidence.lattice_size == 3
    assert coarse.evidence.empirical_law_step == 0.5
    assert fine.evidence.population_size == 4
    assert fine.evidence.lattice_size == 5
    assert fine.evidence.empirical_law_step == 0.25
    assert coarse.evidence.refinement_id != fine.evidence.refinement_id


def test_law_dependent_cost_switches_the_deterministic_action():
    def transition(time, state, action, law, args):
        return jnp.asarray([1.0, 0.0]) if state == 0 else jnp.asarray([0.0, 1.0])

    def running(time, state, action, law, args):
        return law[0] if action == "left" else 1.0 - law[0]

    problem = FiniteStateMasterEquationProblem(
        (0, 1),
        ("left", "right"),
        1,
        2,
        transition,
        running,
        lambda state, law, args: 0.0,
        selector_id="left-before-right",
        problem_id="law-dependent-action-switch",
    )

    result = solve_finite_state_master_equation_reference(problem)
    no_zero = problem.lattice.index_of_counts([0, 2])
    balanced = problem.lattice.index_of_counts([1, 1])
    all_zero = problem.lattice.index_of_counts([2, 0])

    assert result.valid
    np.testing.assert_array_equal(result.selectors[0, :, no_zero], 0)
    np.testing.assert_array_equal(result.selectors[0, :, balanced], 0)
    np.testing.assert_array_equal(result.selectors[0, :, all_zero], 1)
    assert result.selected_action(0, 0, all_zero) == "right"


def test_invalid_transition_probability_is_a_stable_failed_result():
    problem = FiniteStateMasterEquationProblem(
        (0, 1),
        ("bad",),
        1,
        2,
        lambda time, state, action, law, args: jnp.asarray([0.4, 0.4]),
        lambda time, state, action, law, args: 0.0,
        lambda state, law, args: 0.0,
        selector_id="only-action",
        problem_id="invalid-transition-simplex",
    )

    result = solve_finite_state_master_equation_reference(problem)

    assert (
        result.status == FiniteStateMasterEquationStatus.INVALID_TRANSITION_PROBABILITIES
    )
    assert not result.valid
    assert result.evidence.simplex_probability_residual == pytest.approx(0.2)
    assert "probability simplex" in result.termination_detail


def test_selector_ties_use_first_declared_action_and_record_its_id():
    problem = _one_state_problem(
        actions=("first", "second"),
        running_cost=lambda time, state, action, law, args: 3.0,
    )

    result = solve_finite_state_master_equation_reference(problem)

    assert result.valid
    np.testing.assert_array_equal(result.selectors, 0)
    assert result.selected_action(0, 0, 0) == "first"
    assert result.selector_id == "declared-action-order:first-minimum"


def test_bellman_action_minimum_terminal_and_probability_residuals_are_returned():
    result = solve_finite_state_master_equation_reference(
        _one_state_problem(
            horizon=2,
            actions=("costly", "cheap"),
            running_cost=lambda time, state, action, law, args: (
                2.0 if action == "costly" else 0.5
            ),
            terminal_cost=lambda state, law, args: 1.0,
        )
    )

    assert result.valid
    assert result.evidence.bellman_residual == pytest.approx(0.0)
    assert result.evidence.action_minimum_residual == pytest.approx(0.0)
    assert result.evidence.terminal_residual == pytest.approx(0.0)
    assert result.evidence.simplex_probability_residual == pytest.approx(0.0)
    np.testing.assert_allclose(result.evidence.bellman_residuals, 0.0)
    np.testing.assert_allclose(result.evidence.action_minimum_residuals, 0.0)
    np.testing.assert_allclose(result.evidence.terminal_residuals, 0.0)
    np.testing.assert_allclose(result.evidence.simplex_probability_residuals, 0.0)


def test_declared_aggregate_transition_is_used_only_as_an_exact_lattice_kernel():
    problem = FiniteStateMasterEquationProblem(
        (0, 1),
        ("hold",),
        1,
        2,
        lambda time, state, action, law, args: (
            jnp.asarray([1.0, 0.0]) if state == 0 else jnp.asarray([0.0, 1.0])
        ),
        lambda time, state, action, law, args: 0.0,
        lambda state, law, args: law[0],
        selector_id="only-action",
        problem_id="declared-aggregate-kernel",
        aggregate_law_transition=lambda time, law, selector, args: {(2, 0): 1.0},
        aggregate_law_transition_id="all-population-to-zero",
    )

    result = solve_finite_state_master_equation_reference(problem)
    all_zero = problem.lattice.index_of_counts([2, 0])

    assert result.valid
    np.testing.assert_allclose(result.law_transition_table[:, :, all_zero], 1.0)
    assert result.aggregate_transition_mode == "declared-lattice-probabilities"


def test_certificate_and_neighbor_evidence_make_no_continuous_or_mfc_claim():
    problem = FiniteStateMasterEquationProblem(
        (0, 1),
        ("stay",),
        0,
        2,
        lambda time, state, action, law, args: jnp.asarray([1.0, 0.0]),
        lambda time, state, action, law, args: 0.0,
        lambda state, law, args: state + law[0],
        selector_id="terminal-only",
        problem_id="scope-explicit-terminal-grid",
    )

    result = solve_finite_state_master_equation_reference(problem)

    assert result.certificate_label == FINITE_STATE_DISCRETE_MASTER_EQUATION_REFERENCE
    assert result.finite_state_discrete_reference
    assert result.exact_population_lattice_evaluated
    assert not result.continuous_state_claimed
    assert not result.continuous_law_claimed
    assert not result.lions_derivative_claimed
    assert not result.lions_derivative_evaluated
    assert not result.continuous_master_equation_claimed
    assert not result.global_master_equation_claimed
    assert not result.mean_field_control_optimum_claimed
    assert not result.mean_field_control_claimed
    assert not result.mean_field_game_equilibrium_claimed
    assert not result.common_noise_equilibrium_claimed
    assert not result.common_noise_claimed
    assert not result.common_noise_supported
    assert not result.finite_common_state_supported
    assert not result.finite_common_state_evaluated
    assert result.evidence.law_sensitivity_label == (
        DISCRETE_EMPIRICAL_LAW_NEIGHBOR_TRANSFER_DIFFERENCE
    )
    assert result.evidence.discrete_empirical_law_difference
    assert not result.evidence.lions_derivative_computed
    assert result.evidence.neighbor_transfer_differences.shape == (1, 2, 4)
