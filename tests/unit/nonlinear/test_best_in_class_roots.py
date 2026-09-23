#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


nl = phx.nonlinear


def _root_termination():
    return nl.NonlinearTermination(
        absolute_residual=1e-8,
        relative_residual=0.0,
        maximum_steps=100,
        maximum_evaluations=2000,
        maximum_linear_iterations=10000,
    )


def test_dynamic_budget_and_fail_fast_nested_evidence_are_jittable():
    problem = nl.NonlinearSystemProblem(lambda state, target: state - target)
    failing = nl.FunctionNonlinearUpdate(
        lambda state, args: jnp.full_like(state, jnp.nan),
        update_id="failing",
    )
    skipped = nl.FunctionNonlinearUpdate(
        lambda state, args: args,
        update_id="skipped",
    )
    update = nl.CompositeNonlinearUpdate(
        (failing, skipped),
        kind="multiplicative",
    )
    prepared = nl.prepare_nonlinear_update(
        problem,
        jnp.asarray([0.0]),
        update,
        args=jnp.asarray([1.0]),
    )

    @eqx.filter_jit
    def apply(current, budget):
        return nl.apply_prepared_nonlinear_update(
            current,
            jnp.asarray([0.0]),
            args=jnp.asarray([1.0]),
            control=nl.NonlinearUpdateControl(maximum_residual_evaluations=budget),
        )[0]

    result = apply(prepared, jnp.asarray(10, dtype=jnp.int32))
    exhausted = apply(prepared, jnp.asarray(1, dtype=jnp.int32))

    assert result.status == int(nl.NonlinearUpdateStatus.INNER_FAILURE)
    assert int(result.components[0].diagnostics.work.residual_evaluations) == 2
    assert int(result.components[1].diagnostics.work.residual_evaluations) == 0
    assert bool(result.components[1].evidence.skipped)
    assert exhausted.status == int(nl.NonlinearUpdateStatus.INNER_FAILURE)
    assert int(exhausted.diagnostics.work.residual_evaluations) == 2


def test_canonical_prepared_newton_step_retains_iteration_state():
    problem = nl.NonlinearSystemProblem(lambda state, target: state * state - target)
    update = nl.NewtonStepUpdate(termination=_root_termination())
    prepared = nl.prepare_nonlinear_update(
        problem,
        jnp.asarray([1.0]),
        update,
        args=jnp.asarray([2.0]),
    )
    first, next_prepared = nl.apply_prepared_nonlinear_update(
        prepared,
        jnp.asarray([1.0]),
        args=jnp.asarray([2.0]),
    )

    assert bool(first.applied)
    assert jnp.allclose(first.state, jnp.asarray([1.5]))
    assert int(next_prepared.internal_state.run.iteration) == 1
    assert int(first.diagnostics.linear_solves) == 1


@pytest.mark.parametrize(
    "method",
    [
        nl.Bisection(),
        nl.Brent(),
        nl.Ridder(),
        nl.TOMS748(),
        nl.SafeguardedNewton(),
        nl.SafeguardedHalley(),
    ],
)
def test_scalar_root_family_preserves_bracket_and_certifies_residual(method):
    problem = nl.ScalarRootProblem(
        lambda state, target: state * state - target,
        bracket=(0.0, 2.0),
        problem_id="sqrt-two",
    )
    result = nl.scalar_root(
        problem,
        method=method,
        termination=_root_termination(),
        args=2.0,
    )

    assert bool(result.successful)
    assert bool(result.bracket_valid)
    assert float(result.lower) <= jnp.sqrt(2.0) <= float(result.upper)
    assert abs(float(result.value)) <= 1e-8


def test_newton_iteration_trace_and_control_stop_at_an_accepted_state():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state * state - target,
        problem_id="observed-sqrt-two",
    )
    iteration = phx.execution.IterationPlan(
        granularity="attempt",
        observers=(phx.execution.IterationTraceObserver(16),),
        stop_rule=phx.execution.CallableIterationStopRule(
            lambda initial: jnp.asarray(0, dtype=jnp.int32),
            lambda state, record: (
                state + 1,
                record.coordinates.ordinal >= 1,
            ),
            "one-newton-step",
        ),
    )
    result = nl.root(
        problem,
        jnp.asarray(1.0),
        args=2.0,
        termination=_root_termination(),
        iteration=iteration,
    )

    assert int(result.status) == int(nl.NonlinearStatus.USER_STOPPED)
    assert jnp.allclose(result.state, 1.5)
    assert result.iteration_evidence is not None
    trace = result.iteration_evidence.observer_outputs[0]
    assert int(trace.stored_count) == 1
    assert float(trace.terminal.metrics.final_residual_norm) == pytest.approx(0.25)


def test_scalar_root_exposes_terminal_iteration_evidence():
    result = nl.scalar_root(
        nl.ScalarRootProblem(
            lambda state, target: state * state - target,
            bracket=(0.0, 2.0),
            problem_id="terminal-sqrt-two",
        ),
        args=2.0,
        iteration=phx.execution.IterationPlan(
            granularity="terminal",
            observers=(phx.execution.IterationTraceObserver(0),),
        ),
    )

    assert result.nonlinear_result.iteration_evidence is not None
    trace = result.nonlinear_result.iteration_evidence.observer_outputs[0]
    assert int(trace.stored_count) == 0
    assert int(trace.terminal.status) == int(nl.NonlinearStatus.SUCCESS)


@pytest.mark.parametrize(
    "method",
    [
        nl.Broyden("good"),
        nl.Broyden("bad"),
        nl.DFSANE(),
        nl.PseudoTransient(initial_step=0.1),
        nl.VectorHalley(),
        nl.RobustRoot(),
    ],
)
def test_vector_root_family_certifies_physical_root(method):
    problem = nl.NonlinearSystemProblem(lambda state, target: state * state - target)
    result = method.solve(
        problem,
        jnp.ones((2,)),
        args=jnp.asarray([4.0, 9.0]),
        termination=_root_termination(),
    )

    assert bool(result.successful)
    assert jnp.allclose(result.state, jnp.asarray([2.0, 3.0]), atol=1e-6)
    assert float(result.diagnostics.final_residual_norm) <= 1e-7


def test_fast_root_defers_initial_residual_to_selected_method():
    calls = 0

    def residual(state, args):
        nonlocal calls
        calls += 1
        return state - 1.0

    result = nl.FastRoot().solve(
        nl.NonlinearSystemProblem(residual),
        jnp.ones(2),
        termination=_root_termination(),
    )

    assert bool(result.successful)
    assert calls == 1
    assert int(result.diagnostics.residual_evaluations) == 1


def test_root_polyalgorithm_hands_newton_model_to_next_attempt():
    result = nl.RootPolyalgorithm((nl.NewtonKrylov(), nl.NewtonTrustRegion())).solve(
        nl.NonlinearSystemProblem(lambda state, args: jnp.ones_like(state)),
        jnp.zeros(2),
        termination=nl.NonlinearTermination(
            absolute_residual=1e-8,
            relative_residual=0.0,
            maximum_steps=6,
            maximum_evaluations=30,
            maximum_linear_iterations=100,
        ),
    )

    assert len(result.attempts) == 2
    assert int(result.attempts[0].work.residual_evaluations) == 1
    assert int(result.attempts[1].work.residual_evaluations) == 0
    assert "residual-reuses=1" in result.provenance.notes
    assert "prepared-handoffs=1" in result.provenance.notes


def test_chord_converges_in_declared_local_basin():
    problem = nl.NonlinearSystemProblem(lambda state, target: state * state - target)
    result = nl.Chord().solve(
        problem,
        jnp.ones((2,)),
        args=jnp.asarray([1.21, 1.44]),
        termination=_root_termination(),
    )
    assert bool(result.successful)
    assert jnp.allclose(result.state, jnp.asarray([1.1, 1.2]), atol=1e-6)


@pytest.mark.parametrize("kind", ["type-i", "type-ii"])
def test_anderson_variants_and_steffensen_converge(kind):
    problem = nl.FixedPointProblem(lambda state, args: jnp.cos(state))
    anderson = nl.FixedPointIteration(
        acceleration=nl.AndersonAcceleration(kind=kind)
    ).solve(problem, jnp.asarray([1.0]), termination=_root_termination())
    steffensen = nl.SteffensenIteration().solve(
        problem,
        jnp.asarray([1.0]),
        termination=_root_termination(),
    )

    assert bool(anderson.successful)
    assert bool(steffensen.successful)
    assert jnp.allclose(anderson.state, steffensen.state, atol=1e-7)


def test_fixed_point_initial_solution_has_exact_success_work():
    result = nl.FixedPointIteration(
        acceleration=nl.AndersonAcceleration(history=4)
    ).solve(
        nl.FixedPointProblem(lambda state, args: state),
        jnp.asarray([1.0, -2.0]),
        termination=nl.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            maximum_steps=1,
            maximum_evaluations=1,
        ),
    )

    assert bool(result.successful)
    assert int(result.diagnostics.iterations) == 0
    assert int(result.diagnostics.residual_evaluations) == 1
    assert int(result.diagnostics.linear_solves) == 0
    assert int(result.diagnostics.accepted_steps) == 0
    assert float(result.diagnostics.final_step_norm) == 0.0
    assert jnp.array_equal(result.residual, jnp.zeros(2))


@pytest.mark.parametrize("kind", ["type-i", "type-ii"])
def test_anderson_damped_complex_two_step_recurrence(kind):
    matrix = jnp.asarray(
        [
            [0.20 + 0.10j, 0.05 - 0.02j],
            [-0.10 + 0.03j, 0.30 - 0.05j],
        ],
        dtype=jnp.complex64,
    )
    offset = jnp.asarray([1.0 + 0.5j, -0.4 + 0.2j], dtype=jnp.complex64)
    initial = jnp.asarray([0.2 - 0.1j, -0.3 + 0.4j], dtype=jnp.complex64)
    damping = 0.4

    def mapping(state, args):
        return matrix @ state + offset

    result = nl.FixedPointIteration(
        damping=damping,
        acceleration=nl.AndersonAcceleration(
            kind=kind,
            history=1,
            regularization=0.0,
            safeguard_factor=1e20,
            restart_condition=1e20,
        ),
    ).solve(
        nl.FixedPointProblem(mapping),
        initial,
        termination=nl.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=2,
            divergence_factor=1e20,
        ),
    )

    residual_0 = mapping(initial, None) - initial
    state_1 = initial + damping * residual_0
    residual_1 = mapping(state_1, None) - state_1
    state_secant = state_1 - initial
    residual_secant = residual_1 - residual_0
    if kind == "type-i":
        coefficient = jnp.vdot(state_secant, residual_1) / jnp.vdot(
            state_secant, residual_secant
        )
    else:
        coefficient = jnp.vdot(residual_secant, residual_1) / jnp.vdot(
            residual_secant, residual_secant
        )
    expected = (
        state_1
        + damping * residual_1
        - (state_secant + damping * residual_secant) * coefficient
    )

    assert jnp.allclose(result.state, expected, rtol=2e-5, atol=2e-5)
    assert int(result.diagnostics.residual_evaluations) == 4
    assert int(result.diagnostics.linear_solves) == 1
    assert bool(result.diagnostics.final_linear_converged)
    assert f"anderson-kind={kind}" in result.provenance.notes


def test_type_ii_regularization_is_direct_and_reports_direct_condition():
    matrix = jnp.diag(jnp.asarray([11.0, 0.0]))
    offset = jnp.asarray([1.0, 0.0])
    result = nl.FixedPointIteration(
        acceleration=nl.AndersonAcceleration(
            kind="type-ii",
            history=2,
            regularization=1.0,
            safeguard_factor=1e6,
            restart_condition=1e6,
        )
    ).solve(
        nl.FixedPointProblem(lambda state, args: matrix @ state + offset),
        jnp.zeros(2),
        termination=nl.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=2,
        ),
    )

    coefficient = 110.0 / 101.0
    expected = jnp.asarray([12.0 - 11.0 * coefficient, 0.0])
    assert jnp.allclose(result.state, expected, rtol=1e-6, atol=1e-6)
    assert int(result.diagnostics.final_linear_rank) == 2
    assert float(result.diagnostics.final_linear_condition_estimate) == pytest.approx(
        101.0**0.5, rel=1e-6
    )


def test_unusable_anderson_solve_reuses_raw_mapping_and_records_restart():
    matrix = jnp.diag(jnp.asarray([11.0, 0.0]))
    offset = jnp.asarray([1.0, 0.0])
    result = nl.FixedPointIteration(
        acceleration=nl.AndersonAcceleration(
            kind="type-ii",
            history=2,
            regularization=1.0,
            safeguard_factor=1e6,
            restart_condition=50.0,
        )
    ).solve(
        nl.FixedPointProblem(lambda state, args: matrix @ state + offset),
        jnp.zeros(2),
        termination=nl.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=2,
            maximum_evaluations=4,
        ),
    )

    assert jnp.array_equal(result.state, jnp.asarray([12.0, 0.0]))
    assert int(result.diagnostics.residual_evaluations) == 3
    assert int(result.diagnostics.linear_solves) == 1
    assert int(result.diagnostics.acceleration_restarts) == 1
    assert float(result.diagnostics.final_linear_condition_estimate) == pytest.approx(
        101.0**0.5, rel=1e-6
    )


def test_anderson_budget_reserves_active_history_mapping_work():
    result = nl.FixedPointIteration(
        acceleration=nl.AndersonAcceleration(history=2)
    ).solve(
        nl.FixedPointProblem(lambda state, args: state + 1.0),
        jnp.zeros(2),
        termination=nl.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=10,
            maximum_evaluations=3,
        ),
    )

    assert int(result.status) == int(nl.NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED)
    assert int(result.diagnostics.iterations) == 1
    assert int(result.diagnostics.residual_evaluations) == 2


def test_anderson_honors_aggregate_iterative_coefficient_budget():
    matrix = jnp.asarray([[0.7, 0.2], [-0.1, 0.8]])
    offset = jnp.asarray([0.3, -0.4])
    method = nl.FixedPointIteration(
        acceleration=nl.AndersonAcceleration(
            kind="type-i",
            history=2,
            regularization=0.0,
            safeguard_factor=1e6,
            restart_condition=1e20,
            linear=phx.linalg.LinearSolvePolicy(
                phx.linalg.GeneralizedLSMR(),
                tolerance=phx.linalg.TolerancePolicy(
                    relative=0.0,
                    absolute=0.0,
                    max_steps=16,
                ),
            ),
        )
    )
    problem = nl.FixedPointProblem(lambda state, args: jnp.tanh(matrix @ state + offset))
    result = method.solve(
        problem,
        jnp.zeros(2),
        termination=nl.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=6,
            maximum_linear_iterations=2,
            divergence_factor=1e20,
        ),
    )

    assert int(result.status) == int(nl.NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED)
    assert int(result.diagnostics.linear_iterations) == 2
    assert int(result.diagnostics.linear_solves) == 2
    wide_budget = method.solve(
        problem,
        jnp.zeros(2),
        termination=nl.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=2,
            maximum_linear_iterations=10_000,
            divergence_factor=1e20,
        ),
    )
    assert jnp.all(jnp.isfinite(wide_budget.state))
    assert int(wide_budget.diagnostics.linear_solves) == 1
    assert int(wide_budget.diagnostics.linear_iterations) <= 16


def test_anderson_safeguard_counts_rejected_accelerated_proposals():
    result = nl.FixedPointIteration(
        acceleration=nl.AndersonAcceleration(
            history=3,
            safeguard_factor=1.000001,
        )
    ).solve(
        nl.FixedPointProblem(lambda state, args: 0.2 + 1.4 * state - 0.6 * state**2),
        jnp.asarray([0.0]),
        termination=nl.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=5,
            divergence_factor=1e20,
        ),
    )

    assert int(result.diagnostics.acceleration_restarts) > 0
    assert int(result.diagnostics.rejected_steps) == int(
        result.diagnostics.acceleration_restarts
    )


def test_zero_coordinate_anderson_is_jittable_and_skips_coefficients():
    method = nl.FixedPointIteration(
        damping=0.5,
        acceleration=nl.AndersonAcceleration(kind="type-i", history=5),
    )
    problem = nl.FixedPointProblem(lambda state, args: state)
    solve = eqx.filter_jit(
        lambda state: method.solve(
            problem,
            state,
            termination=nl.NonlinearTermination(maximum_evaluations=1),
        )
    )

    result = solve(jnp.empty((0,)))

    assert bool(result.successful)
    assert int(result.diagnostics.residual_evaluations) == 1
    assert int(result.diagnostics.linear_solves) == 0
    assert int(result.diagnostics.final_linear_status) == -1
    assert "history-requested=5" in result.provenance.notes
    assert "history-effective=0" in result.provenance.notes


def test_fixed_point_conversion_preserves_identity_sign_and_implicit_derivative():
    fixed_point = nl.FixedPointProblem(
        lambda state, target: target,
        problem_id="converted-fixed-point",
    )
    problem = fixed_point.as_nonlinear_problem()

    def solution(target):
        return nl.implicit_root_result(
            problem,
            jnp.zeros_like(target),
            termination=_root_termination(),
            args=target,
        ).state

    target = jnp.asarray([2.0, -1.0])
    value, tangent = jax.jvp(solution, (target,), (jnp.ones_like(target),))

    assert problem.problem_id == fixed_point.problem_id
    assert jnp.array_equal(problem.residual(jnp.zeros(2), target), target)
    assert jnp.allclose(value, target)
    assert jnp.allclose(tangent, jnp.ones_like(target))


def test_first_second_and_truncated_solution_map_derivatives():
    problem = nl.NonlinearSystemProblem(lambda state, argument: state * state - argument)
    first = nl.root_solution_jvp(
        problem,
        jnp.asarray([2.0]),
        jnp.asarray([4.0]),
        jnp.asarray([1.0]),
    )
    second = nl.root_solution_second_jvp(
        problem,
        jnp.asarray([2.0]),
        jnp.asarray([4.0]),
        jnp.asarray([1.0]),
    )
    truncated = nl.differentiate_iterations_jvp(
        lambda state, argument: 0.5 * (state + argument / state),
        jnp.asarray([1.0]),
        jnp.asarray([4.0]),
        jnp.asarray([1.0]),
        policy=nl.SensitivityPolicy(
            "truncated",
            iterations=8,
            truncation=3,
        ),
    )

    assert bool(first.evidence.successful)
    assert bool(second.evidence.successful)
    assert jnp.allclose(first.value, jnp.asarray([0.25]))
    assert jnp.allclose(second.value, jnp.asarray([-0.03125]))
    assert jnp.allclose(truncated.value, jnp.asarray([0.25]), atol=1e-7)


def test_explicit_solution_map_sensitivities_reject_a_nonroot_state():
    problem = nl.NonlinearSystemProblem(lambda state, argument: state * state - argument)
    state = jnp.asarray([1.5])
    argument = jnp.asarray([4.0])
    policy = nl.SensitivityPolicy(
        "implicit-forward",
        primal_residual_tolerance=1.0e-10,
    )

    forward = nl.root_solution_jvp(
        problem,
        state,
        argument,
        jnp.ones_like(argument),
        policy=policy,
    )
    reverse = nl.root_solution_vjp(
        problem,
        state,
        argument,
        jnp.ones_like(state),
        policy=nl.SensitivityPolicy(
            "implicit-reverse",
            primal_residual_tolerance=1.0e-10,
        ),
    )

    assert forward.evidence.status == int(nl.SensitivityStatus.PRIMAL_FAILED)
    assert reverse.evidence.status == int(nl.SensitivityStatus.PRIMAL_FAILED)
    assert not bool(forward.evidence.primal_valid)
    assert not bool(reverse.evidence.primal_valid)
    assert forward.evidence.primal_residual_norm > 1.0
    assert jnp.all(jnp.isnan(forward.value))
    assert jnp.all(jnp.isnan(reverse.value))


def test_minimizer_solution_sensitivity_rejects_a_nonstationary_point():
    derivative = nl.minimizer_solution_jvp(
        lambda state, target: 0.5 * jnp.sum((state - target) ** 2),
        jnp.asarray([0.0]),
        jnp.asarray([2.0]),
        jnp.asarray([1.0]),
        policy=nl.SensitivityPolicy(
            "implicit-forward",
            primal_residual_tolerance=1.0e-10,
        ),
    )

    assert derivative.evidence.status == int(nl.SensitivityStatus.PRIMAL_FAILED)
    assert not bool(derivative.evidence.primal_valid)
    assert jnp.all(jnp.isnan(derivative.value))


def test_small_batch_mixed_precision_and_sharding_contracts():
    starts = jnp.ones((4, 2))
    arguments = jnp.asarray([[4.0, 9.0], [1.0, 16.0], [0.25, 0.36], [25.0, 36.0]])
    batched = nl.batched_small_root(
        lambda state, target: state * state - target,
        starts,
        arguments,
        maximum_steps=12,
        absolute_tolerance=1e-8,
        relative_tolerance=0.0,
    )
    assert bool(jnp.all(batched.successful))

    problem = nl.NonlinearSystemProblem(lambda state, target: state * state - target)
    mixed = nl.MixedPrecisionRootExecution(
        nl.NonlinearPrecisionPolicy(
            model_dtype="float32",
            direction_dtype="float32",
            certificate_dtype="float64",
        )
    ).solve(
        problem,
        jnp.asarray([1.0], dtype=jnp.float64),
        nl.NewtonKrylov(),
        nl.NonlinearTermination(
            absolute_residual=1e-6,
            relative_residual=0.0,
            maximum_steps=20,
        ),
        args=jnp.asarray([2.0], dtype=jnp.float32),
    )
    assert bool(mixed.successful)
    assert mixed.state.dtype == jnp.float64

    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    policy = nl.ShardedNonlinearPolicy(
        state_sharding=sharding,
        residual_sharding=sharding,
        axis_name=None,
    )
    placed = policy.place_state(jnp.asarray([3.0, 4.0]))
    assert float(policy.residual_norm(placed)) == pytest.approx(5.0)


def test_solver_graduation_and_regression_gates():
    evidence = nl.SolverGraduationEvidence(
        0,
        100,
        100,
        100,
        0.9,
        1e-8,
        True,
        True,
        True,
        True,
        True,
    )
    graduation = nl.evaluate_solver_graduation(evidence)
    regression = nl.evaluate_solver_regression(
        nl.SolverRegressionEvidence(
            0,
            0.0,
            0.0,
            1.0,
            False,
            False,
            False,
        )
    )
    assert bool(graduation.production_ready)
    assert bool(regression.passed)


def test_root_polyalgorithm_replans_when_newton_policies_change():
    result = nl.RootPolyalgorithm(
        (
            nl.NewtonKrylov(),
            nl.NewtonTrustRegion(
                linear_policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
            ),
        )
    ).solve(
        nl.NonlinearSystemProblem(lambda state, args: jnp.ones_like(state)),
        jnp.zeros(2),
        termination=nl.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            maximum_steps=4,
            maximum_evaluations=20,
            maximum_linear_iterations=20,
        ),
    )

    assert len(result.attempts) == 2
    assert "prepared-handoffs=0" in result.provenance.notes
    assert int(result.attempts[1].work.jacobian_preparations) >= 1


def test_picard_and_sensitivity_require_physical_problem_validity():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state - target,
        validity=lambda state, residual, auxiliary, target: jnp.all(state < target),
        problem_id="validity-rejected-root",
    )
    termination = nl.NonlinearTermination(
        absolute_residual=1e-10,
        relative_residual=0.0,
        maximum_steps=4,
        maximum_evaluations=8,
    )

    picard = nl.PicardIteration(lambda residual: residual).solve(
        problem,
        jnp.asarray([0.0]),
        args=jnp.asarray([1.0]),
        termination=termination,
    )
    sensitivity = nl.root_solution_jvp(
        problem,
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
        jnp.ones(1),
    )

    assert not bool(picard.successful)
    assert sensitivity.evidence.status == int(nl.SensitivityStatus.PRIMAL_FAILED)
    assert not bool(sensitivity.evidence.primal_valid)
    assert jnp.all(jnp.isnan(sensitivity.value))


def test_picard_certification_uses_declared_residual_geometry():
    residual_space = phx.linalg.ArraySpace(
        (1,),
        dtype=jnp.float64,
        pairing=phx.linalg.DiagonalPairing(jnp.asarray([10_000.0])),
    )
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state - target,
        residual_space=residual_space,
        problem_id="weighted-picard-residual",
    )
    result = nl.PicardIteration(lambda residual: 0.01 * residual).solve(
        problem,
        jnp.asarray([0.0], dtype=jnp.float64),
        args=jnp.asarray([1.0], dtype=jnp.float64),
        termination=nl.NonlinearTermination(
            absolute_residual=1.0,
            relative_residual=0.0,
            maximum_steps=1,
        ),
    )

    assert not bool(result.successful)
    assert float(result.diagnostics.initial_residual_norm) == pytest.approx(100.0)
    assert float(result.diagnostics.final_residual_norm) == pytest.approx(100.0)


def test_steffensen_exit_reuses_cached_mapping_under_evaluation_limit():
    calls = 0

    def mapping(state, args):
        nonlocal calls
        calls += 1
        return state + 1.0

    result = nl.SteffensenIteration().solve(
        nl.FixedPointProblem(mapping),
        jnp.zeros(1),
        termination=nl.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            maximum_steps=5,
            maximum_evaluations=1,
        ),
    )

    assert calls == 1
    assert int(result.status) == int(nl.NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED)
    assert int(result.diagnostics.residual_evaluations) == 1


@pytest.mark.parametrize("method", (nl.Broyden(), nl.Chord()))
def test_quasi_newton_reserves_final_certification_evaluation(method):
    result = method.solve(
        nl.NonlinearSystemProblem(lambda state, target: state - target),
        jnp.asarray([0.0]),
        args=jnp.asarray([2.0]),
        termination=nl.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            maximum_steps=5,
            maximum_evaluations=2,
        ),
    )

    assert int(result.status) == int(nl.NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED)
    assert int(result.diagnostics.residual_evaluations) == 2
    assert jnp.array_equal(result.state, jnp.asarray([0.0]))


def test_safeguarded_derivative_root_reserves_certification_budget():
    calls = 0

    def residual(value, target):
        nonlocal calls
        calls += 1
        return value * value - target

    problem = nl.ScalarRootProblem(residual, bracket=(0.0, 2.0))
    result = nl.scalar_root(
        problem,
        method=nl.SafeguardedNewton(),
        termination=nl.NonlinearTermination(
            maximum_steps=10,
            maximum_evaluations=3,
        ),
        args=2.0,
    )

    assert calls == 3
    assert int(result.status) == int(nl.NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED)
    assert int(result.nonlinear_result.diagnostics.residual_evaluations) == 3

    calls = 0
    with pytest.raises(ValueError, match="at least three"):
        nl.scalar_root(
            problem,
            method=nl.SafeguardedNewton(),
            termination=nl.NonlinearTermination(maximum_evaluations=2),
            args=2.0,
        )
    assert calls == 0


def test_vector_halley_enforces_residual_and_linear_work_limits():
    problem = nl.NonlinearSystemProblem(lambda state, target: state * state - target)
    evaluation_limited = nl.VectorHalley().solve(
        problem,
        jnp.ones(2),
        args=jnp.asarray([4.0, 9.0]),
        termination=nl.NonlinearTermination(
            maximum_steps=10,
            maximum_evaluations=1,
        ),
    )
    linear_limited = nl.VectorHalley().solve(
        problem,
        jnp.ones(2),
        args=jnp.asarray([4.0, 9.0]),
        termination=nl.NonlinearTermination(
            maximum_steps=10,
            maximum_evaluations=10,
            maximum_linear_iterations=1,
        ),
    )

    assert evaluation_limited.status == int(
        nl.NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED
    )
    assert int(evaluation_limited.diagnostics.residual_evaluations) == 1
    assert linear_limited.status == int(
        nl.NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED
    )
    assert int(linear_limited.diagnostics.linear_iterations) == 0


def test_small_root_damping_floor_controls_trials_and_integer_guesses_fail_early():
    kernel = nl.SmallRootKernel(
        lambda state, target: state * state - target,
        minimum_damping=1.0,
        maximum_steps=1,
    )
    result = kernel.solve(jnp.asarray([[0.1]]), jnp.asarray([[2.0]]))

    assert int(result.residual_evaluations[0]) == 2
    assert int(result.accepted_steps[0]) == 0
    assert jnp.array_equal(result.state, jnp.asarray([[0.1]]))

    with pytest.raises(TypeError, match="inexact dtype"):
        nl.LocalRootPlan(plan_id="integer-scalar").solve(
            lambda value: value - 1,
            jnp.asarray(0, dtype=jnp.int32),
        )
    with pytest.raises(TypeError, match="inexact dtype"):
        nl.VectorLocalRootPlan(1, plan_id="integer-vector").solve(
            lambda value: value - 1,
            jnp.asarray([0], dtype=jnp.int32),
        )


def test_newton_direction_refuses_zero_budget_before_linear_solve():
    problem = nl.NonlinearSystemProblem(lambda state, args: state - 1.0)
    model = nl.RootLinearModelPolicy().prepare(problem, jnp.zeros(2), None)
    result = nl.NewtonDirectionPolicy().compute(
        model,
        nl.NonlinearWorkBudget(
            linear_solves=0,
            linear_iterations=0,
        ),
    )

    assert result.status == int(nl.DirectionStatus.BUDGET_EXHAUSTED)
    assert int(result.work.linear_solves) == 0
    assert int(result.work.linear_iterations) == 0
    assert jnp.array_equal(result.direction, jnp.zeros(2))


def test_mixed_precision_reserves_physical_certification_evaluations():
    calls = 0

    def residual(state, target):
        nonlocal calls
        calls += 1
        return state - target

    execution = nl.MixedPrecisionRootExecution()
    problem = nl.NonlinearSystemProblem(residual)
    result = execution.solve(
        problem,
        jnp.asarray([0.0]),
        nl.NewtonKrylov(),
        nl.NonlinearTermination(
            maximum_steps=4,
            maximum_evaluations=4,
        ),
        args=jnp.asarray([2.0]),
    )

    assert bool(result.successful)
    assert int(result.diagnostics.residual_evaluations) == 4

    calls = 0
    with pytest.raises(ValueError, match="at least three"):
        execution.solve(
            problem,
            jnp.asarray([0.0]),
            nl.NewtonKrylov(),
            nl.NonlinearTermination(maximum_evaluations=2),
            args=jnp.asarray([2.0]),
        )
    assert calls == 0
