#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import io
import json
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

import phydrax as phx
from phydrax._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from phydrax._model._component import (
    AbstractComponentSlot,
    bind_component,
    ComponentBinding,
)
from phydrax._strict import StrictModule
from phydrax._trainable import (
    fixed_field,
    LaneLayout,
    model_state_field,
    parameter_field,
)
from phydrax._training import ExponentialMovingAverageTargetPolicy, TrainingProgress
from phydrax._training_kernel import (
    AbstractKernelUpdateRule,
    AbstractTrialStepRule,
    BacktrackingLineSearchRule,
    build_training_checkpoint,
    KernelObjective,
    OptaxUpdateRule,
    prepare_training_kernel,
    restore_training_checkpoint,
    run_training_attempt,
    training_accepted_site_key,
    training_site_key,
    TrainingAttemptOutcome,
    TrainingCheckpointPayload,
    TrainingKernelSpec,
    TrainingRejectionBudgetError,
)
from phydrax._training_objective import _ObjectiveContribution
from phydrax._tree_math import tree_inner


class _Regressor(StrictModule):
    weight: jax.Array = parameter_field()
    calls: jax.Array = model_state_field()
    target: jax.Array = fixed_field()


def _regressor() -> Any:
    return _Regressor(
        jnp.asarray([2.0, -1.0]), jnp.asarray(0.0), jnp.asarray([0.5, 0.25])
    )


def _count_call(model_state: Any) -> Any:
    return eqx.tree_at(lambda state: state.calls, model_state, model_state.calls + 1.0)


def _squared_error(
    parameters: Any, model_state: Any, fixed: Any, payload: Any, keys: Any
) -> Any:
    del keys
    residual = parameters.weight - payload["scale"] * fixed.target
    contribution = _ObjectiveContribution(jnp.sum(residual**2), payload["support"])
    return contribution, _count_call(model_state), {"residual": residual}


def _noisy_squared_error(
    parameters: Any, model_state: Any, fixed: Any, payload: Any, keys: Any
) -> Any:
    batch = jr.normal(keys.accepted_key("batch"), (2,))
    noise = 0.1 * jr.normal(keys.attempt_key("noise"), (2,))
    residual = parameters.weight - payload["scale"] * fixed.target - batch + noise
    contribution = _ObjectiveContribution(jnp.sum(residual**2), payload["support"])
    return contribution, _count_call(model_state), {"residual": residual}


def _fit(
    fn: Any = _squared_error, *, objective_id: Any = "fit", weight: Any = 1.0
) -> Any:
    return KernelObjective(
        objective_id=objective_id,
        kind=ObjectiveKind.DATA_FIT,
        route=DerivativeRoute.DIRECT,
        fn=fn,
        weight=weight,
    )


def _payload(scale: Any = 1.0, support: Any = 2.0) -> Any:
    return {"scale": jnp.asarray(scale), "support": jnp.asarray(support)}


def _kernel(
    rule: Any,
    *,
    objectives: Any = None,
    budget: Any = 3,
    target_policy: Any = None,
    tree: Any = None,
) -> Any:
    tree = _regressor() if tree is None else tree
    spec = TrainingKernelSpec(
        rule,
        context="kernel test",
        rejection_budget=budget,
        target_policy=target_policy,
    )
    kernel = prepare_training_kernel(
        tree,
        (_fit(),) if objectives is None else objectives,
        spec,
        root_authority=ComponentAuthority.MODEL,
    )
    return kernel, kernel.init(tree, jr.key(0))


def _leaves(tree: Any) -> Any:
    return [
        jr.key_data(leaf)
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
        else leaf
        for leaf in jax.tree_util.tree_leaves(tree)
    ]


def _assert_trees_equal(actual: Any, expected: Any) -> None:
    actual_leaves, expected_leaves = _leaves(actual), _leaves(expected)
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for left, right in zip(actual_leaves, expected_leaves, strict=True):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


class _DampingState(StrictModule):
    damping: jax.Array
    trials: jax.Array


class _DampedNewton(AbstractTrialStepRule):
    """Levenberg-Marquardt-like rule: damping grows tenfold on every rejection."""

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ("damping",)
    rule_id: str = eqx.field(static=True)
    acceptance_ratio: float = eqx.field(static=True)
    initial_damping: float = eqx.field(static=True)

    def __init__(self, initial_damping: Any) -> None:
        self.rule_id = "test-damped-newton"
        self.acceptance_ratio = 0.1
        self.initial_damping = initial_damping

    def init(self, parameters: Any) -> Any:
        del parameters
        return _DampingState(
            jnp.asarray(self.initial_damping), jnp.asarray(0, dtype=jnp.int32)
        )

    def trial_step(
        self, parameters: Any, gradients: Any, value: Any, state: Any, context: Any
    ) -> Any:
        del parameters, value, context
        step = jax.tree.map(lambda gradient: -gradient / state.damping, gradients)
        predicted = tree_inner(gradients, gradients) / (2.0 * state.damping)
        return step, predicted, _DampingState(state.damping, state.trials + 1)

    def adapt(self, state: Any, ratio: Any, /, *, accepted: Any) -> Any:
        del ratio
        factor = 1.0 / 3.0 if accepted else 10.0
        return _DampingState(state.damping * factor, state.trials)


class _CurvatureState(StrictModule):
    curvature: object
    steps: jax.Array


class _CurvaturePreconditioned(AbstractKernelUpdateRule):
    """KFAC-like rule whose curvature estimate survives a rejected step."""

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ("curvature",)
    rule_id: str = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)

    def __init__(self, learning_rate: Any) -> None:
        self.rule_id = "test-curvature"
        self.learning_rate = learning_rate

    def init(self, parameters: Any) -> Any:
        return _CurvatureState(
            jax.tree.map(jnp.ones_like, parameters), jnp.asarray(0, dtype=jnp.int32)
        )

    def propose(
        self, parameters: Any, gradients: Any, value: Any, state: Any, context: Any
    ) -> Any:
        curvature = jax.tree.map(
            lambda old, gradient: 0.5 * old + 0.5 * gradient**2,
            state.curvature,
            gradients,
        )
        candidate = jax.tree.map(
            lambda parameter, gradient, scale: (
                parameter - self.learning_rate * gradient / jnp.sqrt(scale)
            ),
            parameters,
            gradients,
            curvature,
        )
        accepted = context.objective_value(candidate) < value
        next_state = _CurvatureState(curvature, state.steps + 1)
        return candidate, next_state, next_state, accepted


def test_training_kernel_scenario_1() -> None:
    kernel, state = _kernel(
        OptaxUpdateRule(optax.sgd(0.1), rule_id="sgd"),
        target_policy=ExponentialMovingAverageTargetPolicy(decay=0.5),
    )
    calls = []
    next_state, evidence = run_training_attempt(
        kernel, state, _payload(), hooks=(lambda *args: calls.append(args),)
    )

    assert int(evidence.outcome) == TrainingAttemptOutcome.ACCEPTED
    residual = state.parameters.weight - kernel.fixed.target
    np.testing.assert_allclose(
        next_state.parameters.weight, state.parameters.weight - 0.1 * residual
    )
    np.testing.assert_allclose(
        # ty: ignore[unresolved-attribute]
        next_state.targets.target.weight,
        0.5 * state.parameters.weight + 0.5 * next_state.parameters.weight,
    )
    assert float(next_state.model_state.calls) == 1.0
    assert int(next_state.attempt_cursor) == 1
    assert int(next_state.accepted_cursor) == 1
    assert bool(next_state.accepted_boundary)
    assert len(calls) == 1
    assert kernel.tree(next_state).target is kernel.fixed.target
    root = jr.key(3)
    accepted = [
        training_accepted_site_key(
            root, objective_id="fit", site="batch", accepted=4, microstep=0
        )
        for _ in range(2)
    ]
    attempts = [
        training_site_key(root, objective_id="fit", site="batch", attempt=a, microstep=0)
        for a in (4, 5)
    ]
    lanes = [
        training_site_key(
            root, objective_id="fit", site="batch", attempt=4, microstep=0, lane=lane
        )
        for lane in (0, 1)
    ]
    data = [jr.key_data(key) for key in (*accepted, *attempts, *lanes)]
    assert np.array_equal(data[0], data[1])
    assert len({tuple(np.asarray(value)) for value in data[1:]}) == 5
    kernel, state = _kernel(
        _DampedNewton(0.01),
        target_policy=ExponentialMovingAverageTargetPolicy(decay=0.5),
    )
    source = state
    hooks = []
    dampings = []
    outcomes = []
    for _ in range(3):
        state, evidence = run_training_attempt(
            kernel, state, _payload(), hooks=(lambda *args: hooks.append(args),)
        )
        outcomes.append(int(evidence.outcome))
        dampings.append(float(state.rule_state.damping))
        if outcomes[-1] == TrainingAttemptOutcome.REJECTED_FINITE:
            # Only the authorized damping moved: parameters, model state, targets,
            # the unauthorized trial counter, and the accepted cursor are unchanged.
            for select in (
                lambda value: value.parameters,
                lambda value: value.model_state,
                lambda value: value.targets,
                lambda value: value.rule_state.trials,
                lambda value: value.accepted_cursor,
            ):
                _assert_trees_equal(select(state), select(source))

    assert outcomes == [
        TrainingAttemptOutcome.REJECTED_FINITE,
        TrainingAttemptOutcome.REJECTED_FINITE,
        TrainingAttemptOutcome.ACCEPTED,
    ]
    np.testing.assert_allclose(dampings, [0.1, 1.0, 1.0 / 3.0])
    np.testing.assert_allclose(state.parameters.weight, kernel.fixed.target)
    assert int(state.rule_state.trials) == 1
    assert float(state.model_state.calls) == 1.0
    assert int(state.attempt_cursor) == 3
    assert int(state.accepted_cursor) == 1
    assert int(state.finite_rejections) == 2
    assert int(state.consecutive_rejections) == 0
    assert len(hooks) == 1
    kernel, state = _kernel(_CurvaturePreconditioned(learning_rate=10.0))
    next_state, evidence = run_training_attempt(kernel, state, _payload())

    assert int(evidence.outcome) == TrainingAttemptOutcome.REJECTED_FINITE
    gradient = state.parameters.weight - kernel.fixed.target
    np.testing.assert_allclose(
        next_state.rule_state.curvature.weight, 0.5 + 0.5 * gradient**2
    )
    assert int(next_state.rule_state.steps) == 0
    _assert_trees_equal(next_state.parameters, state.parameters)
    _assert_trees_equal(next_state.model_state, state.model_state)


def test_training_kernel_scenario_2() -> None:
    kernel, state = _kernel(
        OptaxUpdateRule(optax.adam(0.1), rule_id="adam"),
        target_policy=ExponentialMovingAverageTargetPolicy(decay=0.5),
    )
    state, _ = run_training_attempt(kernel, state, _payload())
    source = state
    state = kernel.accumulate(state, _payload(scale=jnp.nan))
    next_state, evidence = run_training_attempt(kernel, state, _payload())

    assert int(evidence.outcome) == TrainingAttemptOutcome.NONFINITE
    for select in (
        lambda value: value.parameters,
        lambda value: value.model_state,
        lambda value: value.pending_model_state,
        lambda value: value.rule_state,
        lambda value: value.targets,
        lambda value: value.accumulation,
        lambda value: value.accepted_cursor,
        lambda value: value.microstep,
        lambda value: value.finite_rejections,
    ):
        _assert_trees_equal(select(next_state), select(source))
    assert int(next_state.attempt_cursor) == int(source.attempt_cursor) + 1
    assert int(next_state.nonfinite_rejections) == 1
    assert int(next_state.consecutive_rejections) == 1
    assert not bool(next_state.accepted_boundary)
    for budget in [0, 2]:
        kernel, state = _kernel(
            OptaxUpdateRule(optax.sgd(0.1), rule_id="sgd"), budget=budget
        )
        for _ in range(budget):
            state, _ = run_training_attempt(kernel, state, _payload(scale=jnp.nan))
        with pytest.raises(TrainingRejectionBudgetError) as caught:
            run_training_attempt(kernel, state, _payload(scale=jnp.nan))
        assert caught.value.consecutive_rejections == budget + 1
        assert caught.value.outcome is TrainingAttemptOutcome.NONFINITE
    kernel, initial = _resumable_kernel()
    like = build_training_checkpoint(kernel, initial)
    uninterrupted = _advance(kernel, initial, 4)

    progress = TrainingProgress(update_step=2, best_value=0.5, best_step=1)
    saved = _serialized(
        build_training_checkpoint(
            kernel,
            _advance(kernel, initial, 2),
            selection=progress,
            sharding_identity="single-device",
        ),
        like,
    )
    restored = restore_training_checkpoint(
        kernel, saved, sharding_identity="single-device"
    )
    assert restored.selection == progress
    _assert_trees_equal(_advance(kernel, restored.state, 2), uninterrupted)

    # A mid-window checkpoint keeps the open accumulation window.
    window = kernel.accumulate(_advance(kernel, initial, 3), _payload(support=1.0))
    with pytest.raises(ValueError, match="accepted-update boundary"):
        build_training_checkpoint(kernel, window)
    resumed = restore_training_checkpoint(
        kernel,
        _serialized(
            build_training_checkpoint(kernel, window, allow_intermediate=True), like
        ),
    ).state
    resumed, _ = run_training_attempt(kernel, resumed, _payload(scale=0.5, support=3.0))
    _assert_trees_equal(resumed, uninterrupted)


def _serialized(payload: Any, like: Any) -> Any:
    manifest = json.loads(json.dumps(payload.manifest))
    stream = io.BytesIO()
    eqx.tree_serialise_leaves(stream, payload.arrays)
    stream.seek(0)
    arrays = eqx.tree_deserialise_leaves(stream, like.arrays)
    return TrainingCheckpointPayload(manifest, arrays)


def _resumable_kernel(objective_id: Any = "fit") -> Any:
    return _kernel(
        OptaxUpdateRule(optax.adam(0.05), rule_id="adam"),
        objectives=(_fit(_noisy_squared_error, objective_id=objective_id),),
        target_policy=ExponentialMovingAverageTargetPolicy(decay=0.9),
    )


def _advance(kernel: Any, state: Any, steps: Any) -> Any:
    for _ in range(steps):
        state = kernel.accumulate(state, _payload(support=1.0))
        state, _ = run_training_attempt(kernel, state, _payload(scale=0.5, support=3.0))
    return state


def test_training_kernel_scenario_3() -> None:
    kernel, initial = _resumable_kernel()
    state = _advance(kernel, initial, 1)
    payload = build_training_checkpoint(kernel, state, sharding_identity="mesh-a")

    other, _ = _resumable_kernel(objective_id="other-fit")
    with pytest.raises(ValueError, match="objective_identity"):
        restore_training_checkpoint(other, payload, sharding_identity="mesh-a")
    with pytest.raises(ValueError, match="sharding_identity"):
        restore_training_checkpoint(kernel, payload, sharding_identity="mesh-b")
    tampered = dict(payload.arrays)
    tampered["parameters"] = jax.tree.map(lambda leaf: leaf + 1.0, state.parameters)
    with pytest.raises(ValueError, match="parameter revision"):
        restore_training_checkpoint(
            kernel,
            TrainingCheckpointPayload(payload.manifest, tampered),
            sharding_identity="mesh-a",
        )
    for forge in [
        _forge_cursor,
        _forge_rule_state,
        _forge_model_state,
        _forge_pending_model_state,
        _forge_targets,
        _forge_root_key,
    ]:
        kernel, initial = _resumable_kernel()
        payload = build_training_checkpoint(kernel, _advance(kernel, initial, 2))
        with pytest.raises(ValueError, match="content digest"):
            restore_training_checkpoint(
                kernel, TrainingCheckpointPayload(payload.manifest, forge(payload.arrays))
            )
    rule = OptaxUpdateRule(optax.sgd(0.1), rule_id="sgd")
    with pytest.raises(
        ValueError, match=r"PARAMETER leaves \['\.weight'\].*trained tree"
    ):
        _kernel(rule, objectives=(_fit(_HeldWeight(jnp.asarray([1.0, 1.0]))),))
    with pytest.raises(ValueError, match=r"MODEL_STATE leaves \['\.count'\]"):
        _kernel(rule, objectives=(_fit(_HeldCounter(jnp.asarray(0.0))),))
    kernel, state = _kernel(rule, objectives=(_fit(_HeldTarget(jnp.asarray(1.0))),))
    _, evidence = run_training_attempt(kernel, state, _payload())
    assert int(evidence.outcome) == TrainingAttemptOutcome.ACCEPTED
    weight = jnp.asarray([2.0, -1.0])
    values = {}
    for support_b in (1.0, 100.0):
        kernel, state = _kernel(
            OptaxUpdateRule(optax.sgd(1.0), rule_id="sgd"),
            objectives=_two_objectives(3.0),
        )
        next_state, evidence = run_training_attempt(
            kernel, state, _weighting_payload(support_b)
        )
        first, second = evidence.objective_values
        values[support_b] = float(first)
        np.testing.assert_allclose(first, jnp.sum((weight - 1.0) ** 2))
        np.testing.assert_allclose(second, jnp.sum((weight + 2.0) ** 2) / support_b)
        np.testing.assert_allclose(evidence.value, 0.5 * first + 3.0 * second)
        gradient = 0.5 * 2.0 * (weight - 1.0) + 3.0 * 2.0 * (weight + 2.0) / support_b
        np.testing.assert_allclose(next_state.parameters.weight, weight - gradient)
    assert values[1.0] == values[100.0]


def _forge_cursor(arrays: Any) -> Any:
    accepted = arrays["cursors"]["accepted"] + 7
    return {**arrays, "cursors": {**arrays["cursors"], "accepted": accepted}}


def _forge_rule_state(arrays: Any) -> Any:
    return {
        **arrays,
        "rule_state": jax.tree.map(lambda leaf: leaf * 0, arrays["rule_state"]),
    }


def _forge_calls(arrays: Any, name: Any) -> Any:
    forged = eqx.tree_at(lambda state: state.calls, arrays[name], arrays[name].calls + 9)
    return {**arrays, name: forged}


def _forge_model_state(arrays: Any) -> Any:
    return _forge_calls(arrays, "model_state")


def _forge_pending_model_state(arrays: Any) -> Any:
    return _forge_calls(arrays, "pending_model_state")


def _forge_targets(arrays: Any) -> Any:
    return {**arrays, "targets": jax.tree.map(lambda leaf: leaf + 1, arrays["targets"])}


def _forge_root_key(arrays: Any) -> Any:
    return {**arrays, "root_key_data": arrays["root_key_data"] + 1}


class _HeldWeight(StrictModule):
    weight: jax.Array = parameter_field()

    def __call__(
        self, parameters: Any, model_state: Any, fixed: Any, payload: Any, keys: Any
    ) -> Any:
        return _squared_error(parameters, model_state, fixed, payload, keys)


class _HeldCounter(StrictModule):
    count: jax.Array = model_state_field()

    def __call__(
        self, parameters: Any, model_state: Any, fixed: Any, payload: Any, keys: Any
    ) -> Any:
        return _squared_error(parameters, model_state, fixed, payload, keys)


class _HeldTarget(StrictModule):
    offset: jax.Array = fixed_field()

    def __call__(
        self, parameters: Any, model_state: Any, fixed: Any, payload: Any, keys: Any
    ) -> Any:
        contribution, next_state, diagnostics = _squared_error(
            parameters, model_state, fixed, payload, keys
        )
        return (
            _ObjectiveContribution(
                contribution.numerator + self.offset, contribution.support
            ),
            next_state,
            diagnostics,
        )


def _scaled_error(name: Any, center: Any) -> Any:
    def objective(
        parameters: Any, model_state: Any, fixed: Any, payload: Any, keys: Any
    ) -> Any:
        del fixed, keys
        residual = parameters.weight - center
        contribution = _ObjectiveContribution(
            payload[name]["count"] * jnp.sum(residual**2), payload[name]["support"]
        )
        return contribution, model_state, {}

    return objective


def _two_objectives(weight_b: Any) -> Any:
    return (
        _fit(_scaled_error("first", 1.0), objective_id="first", weight=0.5),
        _fit(_scaled_error("second", -2.0), objective_id="second", weight=weight_b),
    )


def _weighting_payload(support_b: Any) -> Any:
    return {
        "first": {"count": jnp.asarray(4.0), "support": jnp.asarray(4.0)},
        "second": {"count": jnp.asarray(1.0), "support": jnp.asarray(support_b)},
    }


def test_training_kernel_scenario_4() -> None:
    kernel, state = _kernel(OptaxUpdateRule(optax.sgd(1.0), rule_id="sgd"))
    state = kernel.accumulate(state, _payload(scale=1.0, support=1.0))
    _, evidence = run_training_attempt(kernel, state, _payload(scale=2.0, support=3.0))
    weight, target = jnp.asarray([2.0, -1.0]), jnp.asarray([0.5, 0.25])
    total = jnp.sum((weight - target) ** 2) + jnp.sum((weight - 2.0 * target) ** 2)
    np.testing.assert_allclose(evidence.value, total / 4.0)
    tree = _LearnedAccelerator(jnp.asarray(0.5))
    with pytest.raises(
        ValueError,
        match=r"no admissible training signal for accelerator parameters at \('\.gain',\)",
    ):
        prepare_training_kernel(
            tree, (_SOLUTION,), _accelerator_spec(), root_authority=None
        )
    kernel = prepare_training_kernel(
        tree, (_WORK,), _accelerator_spec(), root_authority=None
    )
    assert kernel.parameter_authorities == (ComponentAuthority.ACCELERATOR,)
    network = phx.nn.models.MLP(
        in_size=2, out_size="scalar", width_size=4, depth=1, key=jr.key(0)
    )
    tree = _BoundHolder(
        bind_component(network, ComponentAuthority.ACCELERATOR), jnp.asarray(1.0)
    )
    with pytest.raises(
        ValueError,
        match=r"no admissible training signal for accelerator parameters at "
        r"\('\.binding\.model",
    ):
        prepare_training_kernel(
            tree,
            (_fit(),),
            _accelerator_spec(),
            root_authority=ComponentAuthority.MODEL,
        )


class _LearnedAccelerator(AbstractComponentSlot):
    component_authority: ClassVar[ComponentAuthority] = ComponentAuthority.ACCELERATOR
    slot_semantic_id: ClassVar[str] = "tests.learned-accelerator"
    gain: jax.Array = parameter_field()


class _AcceleratedSolver(StrictModule):
    accelerator: _LearnedAccelerator
    shift: jax.Array = parameter_field()


def _solution_error(
    parameters: Any, model_state: Any, fixed: Any, payload: Any, keys: Any
) -> Any:
    del fixed, payload, keys
    total = parameters.shift + parameters.accelerator.gain
    return _ObjectiveContribution(jnp.sum(total**2), 1.0), model_state, {}


def _work(parameters: Any, model_state: Any, fixed: Any, payload: Any, keys: Any) -> Any:
    del fixed, payload, keys
    total = parameters.shift + parameters.accelerator.gain - 3.0
    return _ObjectiveContribution(jnp.sum(total**2), 1.0), model_state, {}


_SOLUTION = KernelObjective(
    objective_id="solution",
    kind=ObjectiveKind.SOLUTION_MAP,
    route=DerivativeRoute.IMPLICIT,
    fn=_solution_error,
)
_WORK = KernelObjective(
    objective_id="work",
    kind=ObjectiveKind.ALGORITHMIC_WORK,
    route=DerivativeRoute.UNROLLED,
    fn=_work,
)


def _accelerator_spec() -> Any:
    return TrainingKernelSpec(
        OptaxUpdateRule(optax.sgd(1.0), rule_id="sgd"),
        context="authority test",
        rejection_budget=0,
    )


class _BoundHolder(StrictModule):
    binding: ComponentBinding
    shift: jax.Array = parameter_field()


def test_training_kernel_scenario_5() -> None:
    tree = _AcceleratedSolver(_LearnedAccelerator(jnp.asarray(0.5)), jnp.asarray(1.0))
    kernel = prepare_training_kernel(
        tree,
        (_SOLUTION, _WORK),
        _accelerator_spec(),
        root_authority=ComponentAuthority.MODEL,
    )
    state = kernel.init(tree, jr.key(0))
    next_state, _ = run_training_attempt(kernel, state, None)
    # The solution map trains the model-owned shift; the work objective trains
    # the accelerator gain. Neither leaks into the other group.
    np.testing.assert_allclose(next_state.parameters.shift, 1.0 - 2.0 * 1.5)
    np.testing.assert_allclose(
        next_state.parameters.accelerator.gain, 0.5 - 2.0 * (1.5 - 3.0)
    )

    with pytest.raises(ValueError, match="no owning component slot"):
        prepare_training_kernel(
            tree, (_SOLUTION, _WORK), _accelerator_spec(), root_authority=None
        )
    tree = _AcceleratedSolver(_LearnedAccelerator(jnp.asarray(0.5)), jnp.asarray(1.0))
    spec = TrainingKernelSpec(
        _PayloadNewtonRule(), context="authority test", rejection_budget=0
    )
    # The work objective does not admit the model-owned shift, so a direction
    # built from the raw payload could train a group that may not learn from it.
    with pytest.raises(ValueError, match="reads the raw attempt payload"):
        prepare_training_kernel(
            tree, (_SOLUTION, _WORK), spec, root_authority=ComponentAuthority.MODEL
        )
    kernel, state = _kernel(BacktrackingLineSearchRule(initial_step=4.0))
    with pytest.raises(ValueError, match="does not accumulate"):
        kernel.accumulate(state, _payload())
    next_state, evidence = run_training_attempt(kernel, state, _payload())
    assert int(evidence.outcome) == TrainingAttemptOutcome.ACCEPTED
    assert float(evidence.value) > float(
        run_training_attempt(kernel, next_state, _payload())[1].value
    )
    tree = _Regressor(
        jnp.asarray([[2.0, -1.0], [1.0, 1.0]]),
        jnp.zeros((2,)),
        jnp.asarray([0.5, 0.25]),
    )
    spec = TrainingKernelSpec(
        OptaxUpdateRule(optax.sgd(0.1), rule_id="sgd"),
        context="lane test",
        rejection_budget=3,
        lane_layout=LaneLayout("member", (".weight", ".calls")),
    )
    kernel = prepare_training_kernel(
        tree, (_fit(),), spec, root_authority=ComponentAuthority.MODEL
    )
    state = kernel.init(tree, jr.key(0))
    payload = {"scale": jnp.asarray([1.0, jnp.nan]), "support": jnp.asarray([2.0, 2.0])}
    next_state, evidence = run_training_attempt(kernel, state, payload)

    np.testing.assert_array_equal(
        evidence.outcome,
        [TrainingAttemptOutcome.ACCEPTED, TrainingAttemptOutcome.NONFINITE],
    )
    np.testing.assert_allclose(
        next_state.parameters.weight[0],
        tree.weight[0] - 0.1 * (tree.weight[0] - tree.target),
    )
    np.testing.assert_array_equal(next_state.parameters.weight[1], tree.weight[1])
    np.testing.assert_array_equal(next_state.accepted_cursor, [1, 0])
    np.testing.assert_array_equal(next_state.model_state.calls, [1.0, 0.0])

    mixed = LaneLayout("member", (".weight",))
    with pytest.raises(ValueError, match="per-lane training maps every"):
        prepare_training_kernel(
            tree,
            (_fit(),),
            TrainingKernelSpec(
                OptaxUpdateRule(optax.sgd(0.1), rule_id="sgd"),
                context="lane test",
                rejection_budget=3,
                lane_layout=mixed,
            ),
            root_authority=ComponentAuthority.MODEL,
        )


class _PayloadNewtonRule(AbstractKernelUpdateRule):
    """Rule that forms its own step from the raw payload (bypassing admission)."""

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ()
    rule_id: str = "payload-newton"

    @property
    def forms_own_derivatives(self) -> bool:
        return True

    def init(self, parameters: Any, /) -> Any:
        return None

    def propose(
        self,
        parameters: Any,
        gradients: Any,
        value: Any,
        rule_state: Any,
        context: Any,
        /,
    ) -> Any:
        return parameters, rule_state, rule_state, jnp.asarray(True)


def test_zero_support_contracts() -> None:
    kernel, state = _kernel(OptaxUpdateRule(optax.sgd(0.1), rule_id="sgd"), budget=0)
    initial = _leaves((state.parameters, state.rule_state, state.model_state))
    for _ in range(3):
        state, evidence = run_training_attempt(kernel, state, _payload(support=0.0))
        assert int(evidence.outcome) == TrainingAttemptOutcome.REJECTED_FINITE
        assert not bool(evidence.supported)
    assert int(state.consecutive_rejections) == 0
    assert int(state.accepted_cursor) == 0
    assert int(state.attempt_cursor) == 3
    for observed, expected in zip(
        _leaves((state.parameters, state.rule_state, state.model_state)),
        initial,
        strict=True,
    ):
        np.testing.assert_array_equal(observed, expected)

    with pytest.raises(TrainingRejectionBudgetError) as raised:
        run_training_attempt(kernel, state, _payload(scale=jnp.nan))
    assert raised.value.outcome is TrainingAttemptOutcome.NONFINITE
    kernel, state = _kernel(
        OptaxUpdateRule(optax.sgd(0.1), rule_id="sgd"),
        objectives=(_fit(), _fit(_unsupported_counter, objective_id="unsupported")),
    )
    reference, reference_state = _kernel(OptaxUpdateRule(optax.sgd(0.1), rule_id="sgd"))
    state = kernel.accumulate(state, _payload(support=1.0))
    next_state, evidence = run_training_attempt(kernel, state, _payload())
    assert int(evidence.outcome) == TrainingAttemptOutcome.ACCEPTED
    # Each supported microbatch counts once; the unsupported objective's
    # increments never reach the committed model state.
    assert float(next_state.model_state.calls) == 2.0
    reference_state = reference.accumulate(reference_state, _payload(support=1.0))
    expected, _ = run_training_attempt(reference, reference_state, _payload())
    _assert_trees_equal(next_state.parameters, expected.parameters)
    _assert_trees_equal(next_state.model_state, expected.model_state)


def _unsupported_counter(
    parameters: Any, model_state: Any, fixed: Any, payload: Any, keys: Any
) -> Any:
    del fixed, payload, keys
    contribution = _ObjectiveContribution(jnp.sum(parameters.weight**2), 0.0)
    counted = eqx.tree_at(
        lambda state: state.calls, model_state, model_state.calls + 10.0
    )
    return contribution, counted, {}


def test_evaluation_view_drives_evaluation_source_targets_from_the_start() -> None:
    def doubled(optimizer_state: Any, parameters: Any) -> Any:
        del optimizer_state
        return jax.tree.map(lambda value: 2.0 * value, parameters)

    policy = ExponentialMovingAverageTargetPolicy(decay=0.5, source="evaluation")
    kernel, state = _kernel(
        OptaxUpdateRule(optax.sgd(0.1), rule_id="sgd", evaluation_parameters=doubled),
        target_policy=policy,
    )
    np.testing.assert_array_equal(state.targets.target.weight, [4.0, -2.0])

    next_state, _ = run_training_attempt(kernel, state, _payload())
    np.testing.assert_allclose(
        # ty: ignore[unresolved-attribute]
        next_state.targets.target.weight,
        0.5 * state.targets.target.weight + next_state.parameters.weight,
    )
