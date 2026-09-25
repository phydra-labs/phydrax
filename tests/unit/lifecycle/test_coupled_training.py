#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from phydrax._array_tree import ArrayPyTreeSchema
from phydrax._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from phydrax._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from phydrax._strict import StrictModule
from phydrax._trainable import LaneLayout, model_state_field, parameter_field
from phydrax._training_kernel import (
    KernelObjective,
    OptaxUpdateRule,
    prepare_training_kernel,
    TrainingAttemptOutcome,
    TrainingKernelSpec,
)
from phydrax._training_objective import _ObjectiveContribution
from phydrax.dynamics._plant import (
    AbstractDiscretePlant,
    PlantParameters,
    PlantProposal,
    PlantStepContext,
)
from phydrax.lifecycle import coupled_training_step, CoupledTrainingPolicy


class _DriftPlant(AbstractDiscretePlant):
    """x <- x + gain * command; a negative command is a physical rejection."""

    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: dict
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)

    def __init__(self):
        semantic = SemanticProvenance({"kind": "coupled-training-drift-plant"})
        self.state_schema = ArrayPyTreeSchema.from_tree(
            {"x": jnp.zeros((1,))}, case_ndim=1
        )
        self.control_schema = ArrayPyTreeSchema.from_tree(jnp.zeros((1,)), case_ndim=1)
        self.parameter_schema = ArrayPyTreeSchema.from_tree(
            {"gain": jnp.asarray(1.0)}, case_ndim=0
        )
        self.reset_fallback = {"x": jnp.asarray(0.0)}
        self.semantic_provenance = semantic
        self.numeric_revision = NumericRevision(semantic, {"gain": jnp.asarray(2.0)})
        self.execution_signature = ExecutableSignature(
            shapes={"x": ()}, algorithm_facts={"method": "drift"}
        )
        self.require_finite_state = True
        self.require_finite_controls = True
        self.require_finite_parameters = True

    def propose_reset(self, keys, parameters, /, *, case_shape, initial_time):
        del keys, parameters, initial_time
        state = {"x": jnp.zeros(case_shape)}
        ok = jnp.ones(case_shape, dtype=jnp.bool_)
        status = jnp.zeros(case_shape, dtype=jnp.int32)
        return PlantProposal(state, state, ok, ok, status, status, None)

    def propose_step(self, context, source, commands, parameters, keys, /):
        del context, keys
        successful = commands >= 0.0
        status = jnp.where(successful, 0, 37).astype(jnp.int32)
        candidate = {"x": source["x"] + parameters["gain"] * commands}
        attempted = jnp.ones(successful.shape, dtype=jnp.bool_)
        return PlantProposal(
            candidate, candidate, attempted, successful, status, status, None
        )


class _DriftEstimator(StrictModule):
    rate: jax.Array = parameter_field()
    updates: jax.Array = model_state_field()


def _drift_error(parameters, model_state, fixed, payload, keys):
    del fixed, keys
    residual = payload["delta"] - parameters.rate
    next_state = eqx.tree_at(
        lambda state: state.updates, model_state, model_state.updates + 1.0
    )
    contribution = _ObjectiveContribution(jnp.sum(residual**2), residual.size)
    return contribution, next_state, {}


def _observed_drift(source, step):
    return {"delta": step.candidate_state.payload["x"] - source.payload["x"]}


def _poisoned_drift(source, step):
    return {"delta": _observed_drift(source, step)["delta"] * jnp.nan}


def _plant(cases):
    plant = _DriftPlant()
    parameters = PlantParameters(
        {"gain": jnp.asarray(2.0)},
        plant.parameter_schema.schema_id,
        plant.numeric_revision,
    )
    keys = jr.split(jr.key(1), cases)
    state = plant.reset(keys, parameters, case_shape=(cases,)).accepted_state
    return plant, parameters, state


def _context(state):
    return PlantStepContext(state.time, state.time + 1.0, state.step_index)


def _kernel(tree, *, lane_layout=None):
    spec = TrainingKernelSpec(
        OptaxUpdateRule(optax.sgd(0.25), rule_id="sgd"),
        context="coupled training test",
        rejection_budget=3,
        lane_layout=lane_layout,
    )
    objective = KernelObjective(
        objective_id="drift",
        kind=ObjectiveKind.DATA_FIT,
        route=DerivativeRoute.DIRECT,
        fn=_drift_error,
    )
    kernel = prepare_training_kernel(
        tree, (objective,), spec, root_authority=ComponentAuthority.MODEL
    )
    return kernel, kernel.init(tree, jr.key(0))


def _assert_trees_equal(actual, expected):
    def data(leaf):
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            return np.asarray(jr.key_data(leaf))
        return np.asarray(leaf)

    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for left, right in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_array_equal(data(left), data(right))


@pytest.mark.parametrize("policy", list(CoupledTrainingPolicy))
def test_accepted_physical_and_training_steps_commit_together(policy):
    plant, parameters, plant_state = _plant(2)
    kernel, kernel_state = _kernel(_DriftEstimator(jnp.asarray(0.0), jnp.asarray(0.0)))
    hooks = []
    result = coupled_training_step(
        plant,
        plant_state,
        kernel,
        kernel_state,
        _observed_drift,
        policy,
        context=_context(plant_state),
        commands=jnp.asarray([1.0, 0.5]),
        plant_parameters=parameters,
        hooks=(lambda *args: hooks.append(args),),
    )

    np.testing.assert_array_equal(result.plant_state.payload["x"], [2.0, 1.0])
    # d/drate mean((delta - rate)^2) = -3 at rate 0; one sgd step of 0.25.
    np.testing.assert_allclose(result.kernel_state.parameters.rate, 0.75)
    assert float(result.kernel_state.model_state.updates) == 1.0
    assert int(result.kernel_state.accepted_cursor) == 1
    assert bool(result.evidence.training_committed)
    assert len(hooks) == 1


def test_nonfinite_training_on_a_valid_step_rolls_back_only_training():
    plant, parameters, plant_state = _plant(2)
    kernel, kernel_state = _kernel(_DriftEstimator(jnp.asarray(0.0), jnp.asarray(0.0)))
    result = coupled_training_step(
        plant,
        plant_state,
        kernel,
        kernel_state,
        _poisoned_drift,
        CoupledTrainingPolicy.PHYSICAL_MAY_COMMIT,
        context=_context(plant_state),
        commands=jnp.asarray([1.0, 1.0]),
        plant_parameters=parameters,
    )

    assert int(result.evidence.training.outcome) == TrainingAttemptOutcome.NONFINITE
    np.testing.assert_array_equal(result.plant_state.payload["x"], [2.0, 2.0])
    _assert_trees_equal(result.kernel_state.parameters, kernel_state.parameters)
    _assert_trees_equal(result.kernel_state.rule_state, kernel_state.rule_state)
    _assert_trees_equal(result.kernel_state.model_state, kernel_state.model_state)
    assert int(result.kernel_state.nonfinite_rejections) == 1


def test_per_lane_parameters_commit_only_physically_accepted_lanes():
    plant, parameters, plant_state = _plant(2)
    tree = _DriftEstimator(jnp.zeros((2,)), jnp.zeros((2,)))
    kernel, kernel_state = _kernel(
        tree, lane_layout=LaneLayout("case", (".rate", ".updates"))
    )
    result = coupled_training_step(
        plant,
        plant_state,
        kernel,
        kernel_state,
        _observed_drift,
        CoupledTrainingPolicy.PHYSICAL_MAY_COMMIT,
        context=_context(plant_state),
        commands=jnp.asarray([1.0, -1.0]),
        plant_parameters=parameters,
    )

    np.testing.assert_array_equal(result.evidence.training_committed, [True, False])
    # Lane 0 learns from its own step (delta 2); lane 1's derived update is
    # discarded with its rejected physical step.
    np.testing.assert_allclose(result.kernel_state.parameters.rate, [1.0, 0.0])
    np.testing.assert_array_equal(result.kernel_state.accepted_cursor, [1, 0])
    np.testing.assert_array_equal(result.kernel_state.attempt_cursor, [1, 0])
    np.testing.assert_array_equal(result.plant_state.payload["x"], [2.0, 0.0])

    misaligned_plant, misaligned_parameters, misaligned_state = _plant(3)
    with pytest.raises(ValueError, match="aligned with the plant cases"):
        coupled_training_step(
            misaligned_plant,
            misaligned_state,
            kernel,
            kernel_state,
            _observed_drift,
            CoupledTrainingPolicy.PHYSICAL_MAY_COMMIT,
            context=_context(misaligned_state),
            commands=jnp.asarray([1.0, 1.0, 1.0]),
            plant_parameters=misaligned_parameters,
        )


def test_unknown_policy_is_rejected():
    plant, parameters, plant_state = _plant(1)
    kernel, kernel_state = _kernel(_DriftEstimator(jnp.asarray(0.0), jnp.asarray(0.0)))
    with pytest.raises(ValueError, match="not a valid CoupledTrainingPolicy"):
        coupled_training_step(
            plant,
            plant_state,
            kernel,
            kernel_state,
            _observed_drift,
            "physical-first",
            context=_context(plant_state),
            commands=jnp.asarray([1.0]),
            plant_parameters=parameters,
        )
