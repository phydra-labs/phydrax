#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx


class _Relaxation(phx.AbstractComponentSlot):
    """Accelerator slot: the relaxation factor of a fixed-work Richardson solve."""

    component_authority: ClassVar = phx.ComponentAuthority.ACCELERATOR
    slot_semantic_id: ClassVar[str] = "test.relaxation"
    log_omega: jax.Array = phx.parameter_field()

    def __init__(self, omega):
        self.log_omega = jnp.log(jnp.asarray(omega))


class _Gain(phx.AbstractComponentSlot):
    """Discretization slot with one trainable gain."""

    component_authority: ClassVar = phx.ComponentAuthority.DISCRETIZATION
    slot_semantic_id: ClassVar[str] = "test.gain"
    gain: jax.Array = phx.parameter_field()

    def __init__(self, gain):
        self.gain = jnp.asarray(gain)


class _Response(phx.AbstractArrayModel):
    """Learned response `gain * tanh(x)`, optionally with fresh evaluation noise."""

    gain: jax.Array
    noisy: bool = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, gain, *, noisy=False):
        self.gain = jnp.asarray(gain)
        self.noisy = noisy
        self.in_size = 2
        self.out_size = 2

    def __call__(self, x, /, *, key=None):
        value = self.gain * jnp.tanh(x)
        if self.noisy:
            value = value + 0.05 * jr.normal(key, jnp.shape(x), dtype=x.dtype)
        return value

    def model_execution_contract(self):
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract.smooth(
                (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
            ),
            execution=phx.ExecutionCapabilities("native-jax"),
            randomness=phx.RandomnessContract(
                "resampled" if self.noisy else "deterministic"
            ),
        )


class _Richardson(eqx.Module):
    matrix: jax.Array
    relaxation: _Relaxation | None


def _bind_relaxation(solve, relaxation):
    return _Richardson(solve.matrix, relaxation)


def _richardson_work(owner, rhs):
    omega = jnp.exp(owner.relaxation.log_omega)
    initial = jnp.zeros_like(rhs)
    state = jax.lax.fori_loop(
        0, 4, lambda _, x: x + omega * (rhs - owner.matrix @ x), initial
    )
    return phx.solver.AlgorithmicWorkResult(
        initial_residual=rhs - owner.matrix @ initial,
        final_residual=rhs - owner.matrix @ state,
        iterations=jnp.asarray(4),
        accepted=jnp.all(jnp.isfinite(state)),
    )


def _never_traced(*_):
    raise AssertionError("a refused objective must not bind or measure")


class _Equilibrium(eqx.Module):
    """Residual `x + response(x) - target` of a learned-response balance."""

    response: phx.ComponentBinding

    def __call__(self, state, target):
        return state + self.response.model(state) - target


def _bind_equilibrium(solve, binding):
    del solve
    return _Equilibrium(binding)


def _root_misfit(owner, case):
    target, observed = case
    root = phx.nonlinear.implicit_root_result(
        phx.nonlinear.NonlinearSystemProblem(owner),
        jnp.zeros(2),
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-12, relative_residual=0.0
        ),
        args=target,
    )
    return phx.solver.SolverCaseResult(
        residual=root.state - observed, accepted=root.successful
    )


def test_accelerator_under_a_solution_map_is_refused_before_tracing():
    solve = _Richardson(jnp.eye(2), None)
    refused = phx.solver.SolverObjective(
        solve, _never_traced, _never_traced, objective_id="solution-map"
    )

    with pytest.raises(ValueError, match="no admissible training signal.*accelerator"):
        refused.evaluate(_Relaxation(0.5))
    with pytest.raises(ValueError, match="no admissible training signal.*accelerator"):
        phx.solver.train_components(
            _Relaxation(0.5),
            (refused,),
            optimizer=optax.adam(1e-2),
            steps=1,
            key=jr.key(0),
        )

    work = phx.solver.AlgorithmicWorkObjective(
        solve,
        _bind_relaxation,
        _richardson_work,
        work=4,
        objective_id="fixed-work",
        cases=jnp.asarray([[1.0, 2.0], [-1.0, 0.5]]),
    )
    gradient = jax.grad(lambda tree: work.evaluate(tree).value)(_Relaxation(0.1))
    assert float(jnp.abs(gradient.log_omega)) > 0.0


def test_stochastic_component_needs_one_frozen_realization():
    cases = (
        jnp.asarray([[0.4, -0.2], [0.1, 0.3]]),
        jnp.asarray([[0.2, -0.1], [0.0, 0.2]]),
    )

    def objective():
        return phx.solver.SolverObjective(
            None, _bind_equilibrium, _root_misfit, objective_id="balance", cases=cases
        )

    noisy = phx.bind_component(_Response(0.5, noisy=True), phx.ComponentAuthority.MODEL)
    with pytest.raises(ValueError, match="resampled-randomness-not-admitted"):
        objective().evaluate(noisy)

    def frozen(seed):
        model = phx.FrozenRealization(
            _Response(0.5, noisy=True), jr.key(seed), realization_id=f"draw-{seed}"
        )
        return phx.bind_component(model, phx.ComponentAuthority.MODEL)

    first = objective().evaluate(frozen(3))
    again = objective().evaluate(frozen(3))
    assert first.realization_evidence == ("<root>:fixed-realization",)
    assert bool(jnp.all(first.accepted))
    assert jnp.array_equal(first.value, again.value)
    redrawn = objective().evaluate(frozen(4))
    assert not jnp.allclose(redrawn.value, first.value)


def test_solution_map_gradient_is_the_implicit_derivative_of_accepted_roots():
    cases = (
        jnp.asarray([[0.4, -0.2], [0.1, 0.3]]),
        jnp.asarray([[0.2, -0.1], [0.0, 0.2]]),
    )
    objective = phx.solver.SolverObjective(
        None, _bind_equilibrium, _root_misfit, objective_id="balance", cases=cases
    )

    def value(gain):
        tree = phx.bind_component(_Response(gain), phx.ComponentAuthority.MODEL)
        return objective.evaluate(tree).value

    gradient = jax.grad(value)(jnp.asarray(0.5))
    step = 1e-6
    finite_difference = (value(0.5 + step) - value(0.5 - step)) / (2.0 * step)
    assert gradient == pytest.approx(float(finite_difference), rel=1e-6)

    relu = phx.nn.models.MLP(
        in_size=2,
        out_size=2,
        width_size=4,
        depth=1,
        activation=jax.nn.relu,
        key=jr.key(0),
    )
    kinked = phx.bind_component(relu, phx.ComponentAuthority.MODEL)
    with pytest.raises(ValueError, match="implicit-requires-c1"):
        objective.evaluate(kinked)


def _gain_objective(accepted_results):
    def measure(owner, case):
        # A negative input is outside the owner's support: its loss and its
        # derivative are not numbers.
        loss = (owner.gain * jnp.sqrt(case) - 1.0) ** 2
        return phx.solver.SolverCaseResult(value=loss, accepted=case >= 0.0)

    return phx.solver.RolloutObjective(
        None,
        lambda solve, gain: gain,
        measure,
        objective_id="gain",
        cases=jnp.asarray([1.0, 4.0, -1.0]),
        accepted_results=accepted_results,
    )


def test_failed_cases_reduce_support_with_exact_zero_derivative():
    reduced = _gain_objective("reduce-support")
    evaluation = reduced.evaluate(_Gain(0.5))
    assert evaluation.support == 2.0
    assert int(evaluation.failures) == 1
    assert evaluation.accepted.tolist() == [True, True, False]
    expected = ((0.5 * 1.0 - 1.0) ** 2 + (0.5 * 2.0 - 1.0) ** 2) / 2.0
    assert float(evaluation.value) == pytest.approx(expected)

    gradient = jax.grad(lambda tree: reduced.evaluate(tree).value)(_Gain(0.5))
    # d/dg mean((g sqrt(x) - 1)^2) over the two accepted cases only.
    assert float(gradient.gain) == pytest.approx((2 * (0.5 - 1.0) * 1.0 + 0.0) / 2.0)

    rejected = _gain_objective("reject-attempt").evaluate(_Gain(0.5))
    assert jnp.isnan(rejected.value)
    assert rejected.support == 3.0


class _OffsetGain(phx.AbstractComponentSlot):
    """Discretization slot with a trainable gain and a fixed calibration offset."""

    component_authority: ClassVar = phx.ComponentAuthority.DISCRETIZATION
    slot_semantic_id: ClassVar[str] = "test.offset-gain"
    gain: jax.Array = phx.parameter_field()
    offset: jax.Array = phx.fixed_field()

    def __init__(self, gain, offset):
        self.gain = jnp.asarray(gain)
        self.offset = jnp.asarray(offset)


def test_evaluate_differentiates_only_the_parameter_lane():
    def measure(owner, case):
        x, target = case
        return phx.solver.SolverCaseResult(
            value=(owner.gain * x + owner.offset - target) ** 2,
            accepted=jnp.asarray(True),
        )

    x = jnp.asarray([1.0, 2.0, -1.0])
    target = jnp.asarray([0.5, 3.0, 1.0])
    objective = phx.solver.RolloutObjective(
        None,
        lambda solve, component: component,
        measure,
        objective_id="offset-gain",
        cases=(x, target),
    )
    gradient = jax.grad(lambda tree: objective.evaluate(tree).value)(
        _OffsetGain(2.0, 3.0)
    )
    # The FIXED offset is a constant of the objective: no derivative reaches it.
    assert float(gradient.offset) == 0.0
    expected = jnp.mean(2.0 * (2.0 * x + 3.0 - target) * x)
    assert float(gradient.gain) == pytest.approx(float(expected), rel=1e-12)


def test_algorithmic_work_loss_is_finite_at_exact_convergence_and_saturates():
    def loss(final, dtype):
        return phx.solver.algorithmic_work_loss(
            phx.solver.AlgorithmicWorkResult(
                initial_residual=jnp.asarray([3.0, 4.0], dtype),
                final_residual=final.astype(dtype),
                iterations=jnp.asarray(4),
                accepted=jnp.asarray(True),
            )
        )

    converged = jnp.zeros(2)
    for dtype in (jnp.float32, jnp.float64):
        value, gradient = jax.value_and_grad(lambda final: loss(final, dtype))(converged)
        eps = float(jnp.finfo(dtype).eps)
        assert float(value) == pytest.approx(jnp.log(eps / (1.0 + eps)), rel=1e-5)
        assert bool(jnp.all(gradient == 0.0))
    halved = loss(jnp.asarray([1.5, 2.0]), jnp.float64)
    assert float(halved) == pytest.approx(jnp.log(0.5), rel=1e-12)


def test_fixed_work_objective_fails_cases_that_exit_early():
    def early_exit(owner, rhs):
        result = _richardson_work(owner, rhs)
        return phx.solver.AlgorithmicWorkResult(
            initial_residual=result.initial_residual,
            final_residual=result.final_residual,
            iterations=jnp.asarray(3),
            accepted=result.accepted,
        )

    objective = phx.solver.AlgorithmicWorkObjective(
        _Richardson(jnp.eye(2), None),
        _bind_relaxation,
        early_exit,
        work=4,
        objective_id="early-exit",
        cases=jnp.asarray([[1.0, 2.0]]),
    )
    evaluation = objective.evaluate(_Relaxation(0.5))
    assert evaluation.work.tolist() == [3]
    assert not bool(evaluation.accepted[0])
    assert jnp.isnan(evaluation.value)


def test_mixed_component_trains_its_admitted_group_and_stops_the_rest():
    tree = (
        phx.bind_component(_Response(0.5), phx.ComponentAuthority.MODEL),
        _Relaxation(0.5),
    )
    cases = (jnp.asarray([[0.4, -0.2]]), jnp.asarray([[0.2, -0.1]]))
    objective = phx.solver.SolverObjective(
        None,
        lambda solve, component: _Equilibrium(component[0]),
        _root_misfit,
        objective_id="balance",
        cases=cases,
    )

    evaluation = objective.evaluate(tree)
    assert evaluation.trained == (("[0].model.gain", "model"),)
    assert evaluation.stopped == (("[1].log_omega", "accelerator"),)
    assert "[0]:derivative-supported" in evaluation.derivative_evidence
    gradient = eqx.filter_grad(lambda value: objective.evaluate(value).value)(tree)
    assert float(jnp.abs(gradient[0].model.gain)) > 0.0
    assert float(gradient[1].log_omega) == 0.0
