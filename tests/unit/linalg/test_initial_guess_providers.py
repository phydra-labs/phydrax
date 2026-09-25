#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from tests._ported_models import full_port, in_order, PortedAffine


la = phx.linalg
nl = phx.nonlinear
MATRIX = jnp.asarray([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])


def _diagonal_operator():
    space = la.ArraySpace((2,))
    return la.DenseLinearOperator(
        jnp.asarray([[2.0, 0.0], [0.0, 3.0]]), source=space, target=space
    )


def _operator(matrix=MATRIX):
    space = la.ArraySpace((3,))
    return la.FunctionLinearOperator(
        lambda value: matrix @ value, source=space, target=space
    )


def _policy(mode="mathematical", *, max_steps=None, relative=1e-12):
    return la.LinearSolvePolicy(
        la.FGMRES(restart=3),
        tolerance=la.TolerancePolicy(
            relative=relative, absolute=0.0, max_steps=max_steps
        ),
        differentiation=la.DifferentiationPolicy(mode),
        failure=la.FailurePolicy("status"),
    )


class _Fixed(la.AbstractInitialGuessProvider):
    value: jax.Array
    provider_id: str = eqx.field(static=True)

    def __init__(self, value, provider_id="fixed"):
        self.value = jnp.asarray(value)
        self.provider_id = provider_id

    def propose(self, data, baseline):
        return self.value


class _TreeFixed(la.AbstractInitialGuessProvider):
    value: dict
    provider_id: str = eqx.field(static=True)

    def __init__(self, value, provider_id="tree-fixed"):
        self.value = value
        self.provider_id = provider_id

    def propose(self, data, baseline):
        return self.value


class _Linear(phx.AbstractArrayModel):
    weight: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, weight):
        self.weight = jnp.asarray(weight)
        self.in_size = 3
        self.out_size = 3

    def __call__(self, x, /, *, key=None):
        return self.weight @ x


class _RhsMap(phx.StrictModule):
    model: object

    def __call__(self, data, baseline):
        return self.model(data)


class _BoundRhsMap(phx.StrictModule):
    binding: phx.ComponentBinding

    def __call__(self, data, baseline):
        return self.binding.model(data)


def test_projection_history_exactly_recovers_the_rhs_span():
    operator = _diagonal_operator()
    history = la.HistoryInitialGuess(operator, "shifted-family", capacity=3)
    history = history.update(operator, jnp.asarray([1.0, 0.0]), time=0.0)
    history = history.update(operator, jnp.asarray([0.0, 2.0]), time=1.0)
    rhs = operator.mv(jnp.asarray([2.0, 6.0]))

    result = la.solve(
        la.LinearSystem(operator), rhs, policy=_policy(), initial_guess=history
    )

    assert jnp.allclose(history.propose(rhs, jnp.zeros(2)), jnp.asarray([2.0, 6.0]))
    assert bool(result.initial_guess.accepted)
    assert float(result.initial_guess.proposal_residual_norm) < 1.0e-12
    assert result.initial_guess.provider_id == history.provider_id
    assert jnp.allclose(result.value, jnp.asarray([2.0, 6.0]))


def test_rolling_qr_history_recovers_span_after_eviction():
    operator = _diagonal_operator()
    history = la.HistoryInitialGuess(
        operator, "family", strategy="rolling-qr", capacity=2
    )
    history = history.update(operator, jnp.asarray([1.0, 1.0]), time=0.0)
    history = history.update(operator, jnp.asarray([1.0, 0.0]), time=1.0)
    history = history.update(operator, jnp.asarray([0.0, 2.0]), time=2.0)

    guess = history.propose(operator.mv(jnp.asarray([3.0, 4.0])), jnp.zeros(2))

    assert jnp.allclose(guess, jnp.asarray([3.0, 4.0]), atol=1.0e-10)
    assert history.effective_dimension == 2
    assert history.update_count == 3


def test_rejected_history_update_is_bitwise_inert():
    operator = _diagonal_operator()
    history = la.HistoryInitialGuess(
        operator, "family", strategy="last-solution", capacity=2
    )
    rejected = history.update(operator, jnp.ones((2,)), accepted=False)

    assert rejected.provider_id == history.provider_id
    assert rejected.effective_dimension == 0
    assert rejected.update_count == 0
    assert jnp.array_equal(rejected.solution_basis, history.solution_basis)


def test_stabilized_extrapolation_requires_and_reproduces_a_target_time():
    operator = _diagonal_operator()
    history = la.HistoryInitialGuess(
        operator,
        "family",
        strategy="stabilized-extrapolation",
        capacity=3,
        extrapolation_degree=1,
    )
    history = history.update(operator, jnp.asarray([0.0, 1.0]), time=0.0)
    history = history.update(operator, jnp.asarray([1.0, 3.0]), time=1.0)
    rhs = operator.mv(jnp.zeros((2,)))

    guess = history.at_time(2.0).propose(rhs, jnp.zeros(2))

    assert jnp.allclose(guess, jnp.asarray([2.0, 5.0]), atol=1.0e-12)
    with pytest.raises(ValueError, match="target time"):
        history.propose(rhs, jnp.zeros(2))


def test_worse_or_nonfinite_proposals_visibly_keep_the_native_zero_guess():
    problem = la.LinearSystem(_operator())
    rhs = jnp.asarray([1.0, 2.0, 3.0])
    native = la.solve(problem, rhs, policy=_policy())

    worse = la.solve(problem, rhs, policy=_policy(), initial_guess=_Fixed([50.0, -9, 4]))
    nonfinite = la.solve(
        problem,
        rhs,
        policy=_policy(),
        initial_guess=_Fixed([jnp.nan, 0.0, 0.0]),
    )

    assert not bool(worse.initial_guess.accepted)
    assert bool(worse.initial_guess.proposal_valid)
    assert float(worse.initial_guess.proposal_residual_norm) > float(
        worse.initial_guess.baseline_residual_norm
    )
    assert not bool(nonfinite.initial_guess.accepted)
    assert not bool(nonfinite.initial_guess.proposal_valid)
    for result in (worse, nonfinite):
        assert jnp.array_equal(result.value, native.value)
        assert int(result.diagnostics.iterations) == int(native.diagnostics.iterations)


def test_accepted_proposals_are_selected_per_rhs_under_jit():
    problem = la.LinearSystem(_operator())
    solution = jnp.asarray([1.0, -1.0, 2.0])
    rhs = jnp.stack([MATRIX @ solution, -(MATRIX @ solution)], axis=-1)
    provider = _Fixed(solution + 1.0e-3)

    result = jax.jit(
        lambda target: la.solve(problem, target, policy=_policy(), initial_guess=provider)
    )(rhs)
    native = la.solve(problem, rhs, policy=_policy())

    assert result.initial_guess.accepted.tolist() == [True, False]
    assert bool(jnp.all(result.successful))
    assert jnp.array_equal(result.value[:, 1], native.value[:, 1])
    assert float(result.initial_guess.proposal_residual_norm[0]) < 1.0e-2 * float(
        result.initial_guess.baseline_residual_norm[0]
    )
    assert jnp.allclose(result.value[:, 0], solution, atol=1.0e-10)


def test_production_selection_carries_no_derivative_but_raw_proposals_train():
    problem = la.LinearSystem(_operator())
    rhs = jnp.asarray([1.0, 2.0, 3.0])
    policy = _policy("algorithmic", max_steps=2, relative=0.0)

    def production(weight):
        provider = la.LearnedInitialGuess(_RhsMap(_Linear(weight)))
        return jnp.sum(
            la.solve(problem, rhs, policy=policy, initial_guess=provider).value
        )

    def training(weight):
        provider = la.LearnedInitialGuess(_RhsMap(_Linear(weight)))
        guess = provider.propose(rhs, jnp.zeros(3))
        return jnp.sum(la.solve(problem, rhs, policy=policy, initial_guess=guess).value)

    weight = 0.2 * jnp.eye(3)
    production_result = la.solve(
        problem,
        rhs,
        policy=policy,
        initial_guess=la.LearnedInitialGuess(_RhsMap(_Linear(weight))),
    )

    assert bool(production_result.initial_guess.accepted)
    assert jnp.array_equal(jax.grad(production)(weight), jnp.zeros((3, 3)))
    assert float(jnp.max(jnp.abs(jax.grad(training)(weight)))) > 0.0


def test_learned_provider_binds_its_model_as_an_accelerator():
    model = _Linear(jnp.eye(3))
    provider = la.LearnedInitialGuess(_RhsMap(model))

    ((location, contract),) = provider.component_contracts()
    parameters, _, _ = phx.partition_parameters(provider)

    assert location == "function.model"
    assert contract.authority is phx.ComponentAuthority.ACCELERATOR
    assert contract.slot_semantic_id == la.AbstractInitialGuessProvider.slot_semantic_id
    assert [id(leaf) for leaf in jax.tree.leaves(parameters)] == [id(model.weight)]
    with pytest.raises(TypeError, match="callable module"):
        la.LearnedInitialGuess(model)
    with pytest.raises(ValueError, match="model component"):
        la.LearnedInitialGuess(lambda data, baseline: data)


def test_learned_provider_binds_port_declaring_models_through_its_callable():
    owner = phx.ModelPorts(
        inputs=(full_port("system.rhs", (3,)),),
        outputs=(full_port("system.solution", (3,)),),
    )
    model = PortedAffine(owner, out_size=3, weight=jnp.eye(3))
    with pytest.raises(ValueError, match="initial-guess'.*owner_ports"):
        la.LearnedInitialGuess(_RhsMap(model))

    bound = phx.bind_component(
        model,
        la.LearnedInitialGuess,
        owner_ports=owner,
        port_mapping=in_order(owner, owner),
    )
    provider = la.LearnedInitialGuess(_BoundRhsMap(bound))
    ((location, contract),) = provider.component_contracts()
    assert location == "function.binding"
    assert contract.port_binding.inputs == ((owner.inputs[0].port_id,) * 2,)
    assert contract.port_binding.unverified == ()
    rhs = jnp.asarray([1.0, 2.0, 3.0])
    assert jnp.array_equal(provider.propose(rhs, jnp.zeros(3)), rhs)


def test_nonlinear_initial_state_keeps_the_baseline_for_out_of_domain_proposals():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state**2 - target,
        trial_validity=lambda state, target: jnp.all(state > 0.0),
        trial_validity_id="positive-state",
    )
    target = jnp.asarray([4.0])
    baseline = jnp.asarray([1.0])

    close, close_evidence = nl.select_initial_state(
        problem, baseline, _Fixed([2.1]), args=target
    )
    outside, outside_evidence = nl.select_initial_state(
        problem, baseline, _Fixed([-2.0]), args=target
    )
    result = nl.NewtonKrylov().solve(
        problem, close, termination=nl.NonlinearTermination(), args=target
    )

    assert bool(close_evidence.accepted)
    assert jnp.array_equal(close, jnp.asarray([2.1]))
    assert not bool(outside_evidence.accepted)
    assert not bool(outside_evidence.proposal_valid)
    assert jnp.array_equal(outside, baseline)
    assert bool(result.successful)


def test_nonlinear_selection_is_one_decision_across_every_state_leaf():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: {
            "a": state["a"] - target,
            "b": state["b"] - 2.0 * jnp.sum(target),
        }
    )
    target = jnp.asarray([1.0, 2.0])
    baseline = {"a": jnp.zeros(2), "b": jnp.zeros(3)}
    close = {"a": jnp.asarray([1.0, 2.1]), "b": jnp.full(3, 6.0)}

    selected, evidence = nl.select_initial_state(
        problem, baseline, _TreeFixed(close), args=target
    )

    assert evidence.accepted.shape == ()
    assert bool(evidence.accepted)
    assert jnp.array_equal(selected["a"], close["a"])
    assert jnp.array_equal(selected["b"], close["b"])


def test_selection_evidence_indexes_trailing_rhs_axes_and_rejects_other_layouts():
    from phydrax.linalg._initial_guess import _select_proposal

    proposal = jnp.asarray([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
    baseline = jnp.zeros((3, 2))
    selected, evidence = _select_proposal(
        proposal,
        baseline,
        jnp.asarray([0.1, 9.0]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([True, True]),
        provider_id="columns",
    )
    assert evidence.accepted.tolist() == [True, False]
    assert jnp.array_equal(selected[:, 0], proposal[:, 0])
    assert jnp.array_equal(selected[:, 1], baseline[:, 1])

    with pytest.raises(ValueError, match="trailing evidence axes"):
        _select_proposal(
            proposal,
            baseline,
            jnp.asarray([0.1, 0.2, 0.3]),
            jnp.ones(3),
            jnp.ones(3, dtype=bool),
            provider_id="leading",
        )
    with pytest.raises(ValueError, match="one evidence shape"):
        _select_proposal(
            proposal,
            baseline,
            jnp.asarray([0.1, 0.2]),
            jnp.asarray(1.0),
            jnp.ones(2, dtype=bool),
            provider_id="mixed",
        )
    with pytest.raises(ValueError, match="baseline validity"):
        _select_proposal(
            proposal,
            baseline,
            jnp.asarray([0.1, 0.2]),
            jnp.ones(2),
            jnp.ones(2, dtype=bool),
            provider_id="baseline",
            baseline_valid=jnp.ones(3, dtype=bool),
        )
