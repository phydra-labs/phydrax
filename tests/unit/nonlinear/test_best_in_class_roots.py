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
