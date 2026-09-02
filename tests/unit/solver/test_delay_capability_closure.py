import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax.solver._delay_capabilities import (
    adaptive_stochastic_delay_step_doubling,
    AdaptiveStochasticDelayPolicy,
    backsolve_delay_adjoint,
    BacksolveDelayAdjoint,
    CertifiedTruncatedFunctionalDelay,
    DelayPrimalTape,
    evaluate_certified_truncated_delay,
    ExponentialConvolutionDelay,
    ItoEulerDelayInterpolation,
)


def test_brownian_consistent_step_doubling_retains_only_accepted_states():
    policy = AdaptiveStochasticDelayPolicy(1.0e-3, 1.0e-5, 0.05, 0.25, 16, 8)

    state, evidence, _ = adaptive_stochastic_delay_step_doubling(
        policy,
        ItoEulerDelayInterpolation(),
        0.0,
        0.5,
        jnp.asarray([0.0]),
        0.25,
        lambda left, right, key: jnp.asarray([0.0]),
        lambda left, right, value, increment, args: value + (right - left) + increment,
        jr.key(0),
        causal_maximum_step=0.25,
        path_id="deterministic-brownian-probe",
    )
    assert jnp.allclose(state, jnp.asarray([0.5]))
    assert evidence.accepted_count == 2
    assert not evidence.capacity_exceeded
    assert jnp.all(evidence.accepted_attempts[evidence.attempt_active])


def test_exact_exponential_memory_and_certified_tail_evidence():
    term = ExponentialConvolutionDelay(
        "kernel",
        jnp.asarray([2.0]),
        jnp.asarray([3.0]),
        jnp.asarray([[0.0]]),
    )
    moments = term.advance(term.initial_moments, jnp.asarray([1.0]), 0.5)
    expected = (1.0 - jnp.exp(-1.0)) / 2.0
    assert jnp.allclose(moments[0, 0], expected)
    assert jnp.allclose(term.observation(moments), 3.0 * expected)

    truncated = CertifiedTruncatedFunctionalDelay(
        "tail",
        lambda window, args: jnp.asarray([0.25]),
        4.0,
        lambda time, args: jnp.asarray(1.0e-4),
        1.0e-3,
    )
    value, evidence = evaluate_certified_truncated_delay(
        truncated,
        2.0,
        object(),
        memory_occupancy=16,
    )
    assert evidence.valid & evidence.truncated
    assert jnp.allclose(value, 0.25)


def test_archived_primal_backsolve_reports_advanced_coverage():
    times = jnp.linspace(0.0, 1.0, 5)
    states = jnp.exp(times)[:, None]
    tape = DelayPrimalTape(
        times,
        states,
        jnp.ones((5,), dtype=bool),
        problem_id="linear-retarded",
    )
    gradient, _, evidence = backsolve_delay_adjoint(
        BacksolveDelayAdjoint(8, 2),
        tape,
        lambda time, state, delayed, args: state,
        (0.25,),
        jnp.asarray([1.0]),
    )
    assert evidence.valid
    assert evidence.backward_steps == 4
    assert jnp.all(jnp.isfinite(gradient))


def test_padded_primal_backsolve_matches_compact_active_prefix():
    times = jnp.linspace(0.0, 1.0, 5)
    states = jnp.exp(times)[:, None]
    compact = DelayPrimalTape(
        times,
        states,
        jnp.ones((5,), dtype=bool),
        problem_id="padded-linear-retarded",
    )
    padded = DelayPrimalTape(
        jnp.concatenate((times, jnp.asarray([jnp.inf, -jnp.inf, jnp.nan]))),
        jnp.concatenate(
            (
                states,
                jnp.asarray([[jnp.inf], [-jnp.inf], [jnp.nan]]),
            )
        ),
        jnp.asarray([True, True, True, True, True, False, False, False]),
        problem_id="padded-linear-retarded",
    )
    policy = BacksolveDelayAdjoint(4, 2)

    def drift(time, state, delayed, args):
        return 0.3 * state + 0.2 * delayed[0]

    compact_gradient, _, compact_evidence = backsolve_delay_adjoint(
        policy,
        compact,
        drift,
        (0.25,),
        jnp.asarray([1.0]),
    )
    padded_gradient, _, padded_evidence = backsolve_delay_adjoint(
        policy,
        padded,
        drift,
        (0.25,),
        jnp.asarray([1.0]),
        loss_impulses=jnp.concatenate(
            (jnp.zeros_like(states), jnp.full((3, 1), jnp.nan))
        ),
    )

    assert compact_evidence.valid & padded_evidence.valid
    assert jnp.allclose(padded_gradient, compact_gradient)
    assert padded.tape_id == compact.tape_id
    assert padded_evidence.backward_steps == compact_evidence.backward_steps == 4
    assert jnp.array_equal(
        padded_evidence.backward_active,
        jnp.asarray([True, True, True, True, False, False, False]),
    )
    assert jnp.all(padded_evidence.residual_norms[4:] == 0)
    assert jnp.all(~padded_evidence.advanced_query_covered[4:])


def test_primal_tape_identity_binds_active_primals_discontinuities_and_dtypes():
    active = jnp.asarray([True, True, False])
    reference = DelayPrimalTape(
        jnp.asarray([0.0, 1.0, jnp.nan], dtype=jnp.float32),
        jnp.asarray([[2.0], [3.0], [jnp.inf]], dtype=jnp.float32),
        active,
        jnp.asarray([0.5], dtype=jnp.float32),
        problem_id="identity",
    )
    differently_padded = DelayPrimalTape(
        jnp.asarray([0.0, 1.0, -jnp.inf, jnp.nan], dtype=jnp.float32),
        jnp.asarray([[2.0], [3.0], [jnp.nan], [-jnp.inf]], dtype=jnp.float32),
        jnp.asarray([True, True, False, False]),
        jnp.asarray([0.5], dtype=jnp.float32),
        problem_id="identity",
    )
    changed_time = DelayPrimalTape(
        jnp.asarray([0.0, 1.25, jnp.nan], dtype=jnp.float32),
        reference.states,
        active,
        reference.discontinuities,
        problem_id="identity",
    )
    changed_state = DelayPrimalTape(
        reference.times,
        jnp.asarray([[2.0], [4.0], [jnp.inf]], dtype=jnp.float32),
        active,
        reference.discontinuities,
        problem_id="identity",
    )
    changed_discontinuity = DelayPrimalTape(
        reference.times,
        reference.states,
        active,
        jnp.asarray([0.75], dtype=jnp.float32),
        problem_id="identity",
    )
    changed_dtype = DelayPrimalTape(
        reference.times,
        reference.states.astype(jnp.complex64),
        active,
        reference.discontinuities,
        problem_id="identity",
    )

    assert differently_padded.tape_id == reference.tape_id
    assert changed_time.tape_id != reference.tape_id
    assert changed_state.tape_id != reference.tape_id
    assert changed_discontinuity.tape_id != reference.tape_id
    assert changed_dtype.tape_id != reference.tape_id


def test_primal_backsolve_rejects_nonfinite_active_data_and_masks_parameter_adjoint():
    policy = BacksolveDelayAdjoint(4, 2)
    active = jnp.asarray([True, True, False])
    tapes = (
        DelayPrimalTape(
            jnp.asarray([0.0, jnp.nan, jnp.inf]),
            jnp.asarray([[1.0], [2.0], [jnp.nan]]),
            active,
            problem_id="nonfinite-active-time",
        ),
        DelayPrimalTape(
            jnp.asarray([0.0, 1.0, jnp.nan]),
            jnp.asarray([[1.0], [jnp.inf], [jnp.nan]]),
            active,
            problem_id="nonfinite-active-state",
        ),
    )
    args = {
        "rate": jnp.asarray(0.3),
        "offset": jnp.asarray([1.0, 2.0]),
    }

    for tape in tapes:
        gradient, args_gradient, evidence = backsolve_delay_adjoint(
            policy,
            tape,
            lambda time, state, delayed, parameters: (
                parameters["rate"] * state + 0.0 * jnp.sum(parameters["offset"])
            ),
            (0.25,),
            jnp.asarray([1.0]),
            args=args,
        )

        assert not evidence.valid
        assert evidence.status == policy.failure
        assert jnp.all(jnp.isnan(gradient))
        assert all(jnp.all(jnp.isnan(leaf)) for leaf in jax.tree.leaves(args_gradient))


def test_primal_backsolve_rejects_invalid_active_prefix_with_evidence():
    invalid_masks = (
        jnp.zeros((0,), dtype=bool),
        jnp.asarray([False, False, False]),
        jnp.asarray([True, False, True]),
    )
    policy = BacksolveDelayAdjoint(4, 2)
    args = {"scale": jnp.asarray(1.0)}

    for active in invalid_masks:
        tape = DelayPrimalTape(
            jnp.arange(active.size, dtype=float),
            jnp.zeros((active.size, 1)),
            active,
            problem_id="invalid-active-prefix",
        )
        gradient, args_gradient, evidence = backsolve_delay_adjoint(
            policy,
            tape,
            lambda time, state, delayed, parameters: parameters["scale"] * state,
            (0.25,),
            jnp.asarray([1.0]),
            args=args,
        )

        assert not evidence.valid
        assert evidence.status == policy.failure
        assert evidence.backward_steps == 0
        assert jnp.all(~evidence.backward_active)
        assert jnp.all(jnp.isnan(gradient))
        assert jnp.isnan(args_gradient["scale"])
