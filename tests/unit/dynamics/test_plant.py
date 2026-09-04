#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._array_tree import ArrayPyTreeSchema
from phydrax._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from phydrax.dynamics._layout import InputLayout, StateLayout
from phydrax.dynamics._plant import (
    AbstractDiscretePlant,
    ArrayDiscreteSystemPlant,
    PlantCheckpoint,
    PlantParameters,
    PlantProposal,
    PlantRuntimeState,
    PlantStepContext,
)
from phydrax.dynamics._system import (
    DiscreteStepContext,
    DiscreteSystem,
    DiscreteTransitionResult,
)


class _MixedPlant(AbstractDiscretePlant):
    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: dict[str, jax.Array]
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)
    reset_successful: jax.Array

    def __init__(self, reset_successful):
        fallback = {
            "active": jnp.asarray(False),
            "count": jnp.asarray(-1, dtype=jnp.int32),
            "q": jnp.asarray([9.0, 8.0], dtype=jnp.float32),
        }
        batched_template = jax.tree_util.tree_map(
            lambda value: value[jnp.newaxis, ...], fallback
        )
        semantic = SemanticProvenance({"kind": "mixed-plant-test"})
        numeric = NumericRevision(semantic, {"gain": jnp.asarray(1.0, dtype=jnp.float32)})
        self.state_schema = ArrayPyTreeSchema.from_tree(batched_template, case_ndim=1)
        self.control_schema = ArrayPyTreeSchema.from_tree(
            jnp.zeros((1, 2), dtype=jnp.float32), case_ndim=1
        )
        self.parameter_schema = ArrayPyTreeSchema.from_tree(
            {"scale": jnp.asarray(1.0, dtype=jnp.float32)}, case_ndim=0
        )
        self.reset_fallback = fallback
        self.semantic_provenance = semantic
        self.numeric_revision = numeric
        self.execution_signature = ExecutableSignature(
            shapes={"state_q": (2,), "control": (2,)},
            dtypes={"state_q": jnp.float32, "count": jnp.int32},
            algorithm_facts={"method": "mixed-test"},
        )
        self.require_finite_state = True
        self.require_finite_controls = True
        self.require_finite_parameters = True
        self.reset_successful = jnp.asarray(reset_successful, dtype=bool)

    def propose_reset(
        self,
        keys,
        parameters,
        /,
        *,
        case_shape,
        initial_time,
    ):
        del keys, initial_time
        count = jnp.arange(case_shape[0], dtype=jnp.int32)
        q = parameters["scale"] * jnp.stack(
            (count.astype(jnp.float32), count.astype(jnp.float32) + 0.5),
            axis=-1,
        )
        status = jnp.where(
            self.reset_successful,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(19, dtype=jnp.int32),
        )
        return PlantProposal(
            {"active": (count % 2) == 0, "count": count, "q": q},
            {"active": (count % 2) == 0, "count": count, "q": q},
            jnp.ones(case_shape, dtype=bool),
            self.reset_successful,
            status,
            status,
            {"reset_norm": jnp.linalg.norm(q, axis=-1)},
        )

    def propose_step(self, context, source, commands, parameters, keys, /):
        del context, keys
        assert commands is not None
        successful = commands[:, 0] >= 0.0
        status = jnp.where(
            successful,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(37, dtype=jnp.int32),
        )
        candidate = {
            "active": ~source["active"],
            "count": source["count"] + jnp.asarray(1, dtype=jnp.int32),
            "q": source["q"] + parameters["scale"] * commands,
        }
        return PlantProposal(
            candidate,
            candidate,
            jnp.ones(successful.shape, dtype=bool),
            successful,
            status,
            status,
            {"command": commands},
        )


def _parameters(plant):
    return PlantParameters(
        {"scale": jnp.asarray(2.0, dtype=jnp.float32)},
        plant.parameter_schema.schema_id,
        plant.numeric_revision,
    )


def _keys(count=3):
    return jax.random.split(jax.random.key(7), count)


def _allclose_tree(actual, expected):
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        actual_is_key = jax.dtypes.issubdtype(actual_leaf.dtype, jax.dtypes.prng_key)
        expected_is_key = jax.dtypes.issubdtype(expected_leaf.dtype, jax.dtypes.prng_key)
        assert actual_is_key == expected_is_key
        if actual_is_key:
            actual_leaf = jax.random.key_data(actual_leaf)
            expected_leaf = jax.random.key_data(expected_leaf)
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_mixed_pytree_reset_and_step_roll_back_every_failed_case_atomically():
    plant = _MixedPlant(jnp.asarray([True, False, True]))
    keys = _keys()
    reset = plant.reset(keys, _parameters(plant), case_shape=(3,), initial_time=0.25)

    assert reset.accepted_state.payload["q"].dtype == jnp.dtype(jnp.float32)
    assert reset.accepted_state.payload["count"].dtype == jnp.dtype(jnp.int32)
    assert reset.accepted_state.payload["active"].dtype == jnp.dtype(bool)
    np.testing.assert_array_equal(reset.successful, [True, False, True])
    np.testing.assert_array_equal(
        reset.accepted_state.payload["q"][1], plant.reset_fallback["q"]
    )
    assert int(reset.accepted_state.payload["count"][1]) == -1
    assert not bool(reset.accepted_state.payload["active"][1])
    np.testing.assert_array_equal(
        jax.random.key_data(reset.accepted_state.key)[1],
        jax.random.key_data(keys)[1],
    )
    assert not np.array_equal(
        jax.random.key_data(reset.accepted_state.key)[0],
        jax.random.key_data(keys)[0],
    )

    source = reset.accepted_state
    commands = jnp.asarray([[0.5, 1.0], [-3.0, 7.0], [1.5, -1.0]], dtype=jnp.float32)
    context = PlantStepContext(
        source.time,
        jnp.asarray([1.0, 1.0, 1.0], dtype=source.time.dtype),
        source.step_index,
    )
    step = plant.step(context, source, commands, _parameters(plant))

    np.testing.assert_array_equal(step.successful, [True, False, True])
    for leaf in ("q", "count", "active"):
        np.testing.assert_array_equal(
            step.accepted_state.payload[leaf][1], source.payload[leaf][1]
        )
    assert float(step.accepted_state.time[1]) == float(source.time[1])
    assert int(step.accepted_state.step_index[1]) == int(source.step_index[1])
    np.testing.assert_array_equal(
        jax.random.key_data(step.accepted_state.key)[1],
        jax.random.key_data(source.key)[1],
    )
    successful_cases = jnp.asarray((0, 2), dtype=jnp.int32)
    np.testing.assert_array_equal(step.accepted_state.time[successful_cases], [1.0, 1.0])
    np.testing.assert_array_equal(
        step.accepted_state.step_index[successful_cases], [1, 1]
    )
    np.testing.assert_array_equal(step.candidate_state.step_index, [1, 1, 1])


def test_state_parameter_and_executable_identity_mismatches_are_rejected():
    plant = _MixedPlant(jnp.ones((3,), dtype=bool))
    parameters = _parameters(plant)
    state = plant.reset(_keys(), parameters, case_shape=(3,)).accepted_state
    context = PlantStepContext(state.time, state.time + 1.0, state.step_index)
    commands = jnp.ones((3, 2), dtype=jnp.float32)
    values = (
        "different-semantics",
        state.numeric_revision_id,
        state.state_schema_id,
        state.execution_signature_id,
    )
    wrong_semantic = PlantRuntimeState(
        state.payload, state.time, state.step_index, state.key, *values
    )
    with pytest.raises(ValueError, match="semantic provenance"):
        plant.step(context, wrong_semantic, commands, parameters)

    wrong_numeric = PlantRuntimeState(
        state.payload,
        state.time,
        state.step_index,
        state.key,
        state.semantic_provenance_id,
        "different-numeric",
        state.state_schema_id,
        state.execution_signature_id,
    )
    with pytest.raises(ValueError, match="numeric revision"):
        plant.step(context, wrong_numeric, commands, parameters)

    wrong_schema = PlantRuntimeState(
        state.payload,
        state.time,
        state.step_index,
        state.key,
        state.semantic_provenance_id,
        state.numeric_revision_id,
        "different-schema",
        state.execution_signature_id,
    )
    with pytest.raises(ValueError, match="state schema"):
        plant.step(context, wrong_schema, commands, parameters)

    wrong_execution = PlantRuntimeState(
        state.payload,
        state.time,
        state.step_index,
        state.key,
        state.semantic_provenance_id,
        state.numeric_revision_id,
        state.state_schema_id,
        "different-executable",
    )
    with pytest.raises(ValueError, match="execution signature"):
        plant.step(context, wrong_execution, commands, parameters)

    changed_revision = NumericRevision(
        plant.semantic_provenance,
        {"gain": jnp.asarray(3.0, dtype=jnp.float32)},
    )
    stale_parameters = PlantParameters(
        parameters.values, plant.parameter_schema.schema_id, changed_revision
    )
    with pytest.raises(ValueError, match="numeric revision"):
        plant.step(context, state, commands, stale_parameters)


def test_nonfinite_control_is_casewise_failure_and_schema_mismatch_is_rejected():
    plant = _MixedPlant(jnp.ones((3,), dtype=bool))
    parameters = _parameters(plant)
    state = plant.reset(_keys(), parameters, case_shape=(3,)).accepted_state
    context = PlantStepContext(state.time, state.time + 1.0, state.step_index)
    commands = jnp.asarray([[1.0, 0.0], [jnp.nan, 0.0], [1.0, 0.0]], dtype=jnp.float32)

    result = plant.step(context, state, commands, parameters)
    np.testing.assert_array_equal(result.successful, [True, False, True])
    for leaf in jax.tree_util.tree_leaves(state.payload):
        assert leaf.shape[0] == 3
    np.testing.assert_array_equal(
        result.accepted_state.payload["q"][1], state.payload["q"][1]
    )
    np.testing.assert_array_equal(
        jax.random.key_data(result.accepted_state.key)[1],
        jax.random.key_data(state.key)[1],
    )

    with pytest.raises(TypeError, match="dtype"):
        plant.step(context, state, commands.astype(jnp.float16), parameters)


def test_array_discrete_system_adapter_has_legacy_transition_parity():
    layout = StateLayout((2,))
    input_layout = InputLayout((2,), roles="control")

    def transition(context, state, inputs, args):
        candidate = state + context.duration * (inputs + args["bias"])
        return DiscreteTransitionResult(
            candidate,
            0.75 * candidate,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
        )

    system = DiscreteSystem(
        transition,
        state_layout=layout,
        input_layout=input_layout,
        system_id="adapter-parity",
    )
    semantic = SemanticProvenance({"kind": "adapter-parity"})
    numeric = NumericRevision(
        semantic, {"bias": jnp.asarray([0.25, -0.5], dtype=jnp.float32)}
    )
    parameter_schema = ArrayPyTreeSchema.from_tree(
        {"bias": jnp.zeros((2,), dtype=jnp.float32)}, case_ndim=0
    )
    plant = ArrayDiscreteSystemPlant(
        system,
        lambda key: jnp.asarray([1.0, -2.0], dtype=jnp.float32),
        reset_fallback=jnp.zeros((2,), dtype=jnp.float32),
        semantic_provenance=semantic,
        numeric_revision=numeric,
        execution_signature=ExecutableSignature(
            shapes={"state": (2,), "control": (2,)},
            dtypes={"state": jnp.float32, "control": jnp.float32},
        ),
        parameter_schema=parameter_schema,
        control_dtype=jnp.float32,
    )
    parameters = PlantParameters(
        {"bias": jnp.asarray([0.25, -0.5], dtype=jnp.float32)},
        parameter_schema.schema_id,
        numeric,
    )
    reset = plant.reset(jax.random.key(3), parameters)
    commands = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    context = PlantStepContext(0.0, 0.5, 0)
    adapted = plant.step(context, reset.accepted_state, commands, parameters)
    legacy = system.evaluate_result(
        DiscreteStepContext(0.0, 0.5, 0),
        reset.accepted_state.payload,
        parameters.values,
        inputs=commands,
    )

    np.testing.assert_array_equal(adapted.candidate_state.payload, legacy.candidate_state)
    np.testing.assert_array_equal(adapted.accepted_state.payload, legacy.accepted_state)
    np.testing.assert_array_equal(
        adapted.evidence.legacy_candidate_state, legacy.candidate_state
    )
    assert bool(adapted.successful) == bool(legacy.successful)
    assert int(adapted.status) == int(legacy.status)


def test_checkpoint_restore_and_first_replay_digest_mismatch_are_exact():
    plant = _MixedPlant(jnp.ones((3,), dtype=bool))
    parameters = _parameters(plant)
    initial = plant.reset(_keys(), parameters, case_shape=(3,)).accepted_state
    checkpoint = plant.checkpoint(initial)
    assert plant.verify_checkpoint(checkpoint)
    _allclose_tree(plant.restore(checkpoint), initial)

    changed_payload = eqx.tree_at(
        lambda payload: payload["q"],
        initial.payload,
        initial.payload["q"].at[0, 0].add(1.0),
    )
    changed_state = PlantRuntimeState(
        changed_payload,
        initial.time,
        initial.step_index,
        initial.key,
        initial.semantic_provenance_id,
        initial.numeric_revision_id,
        initial.state_schema_id,
        initial.execution_signature_id,
    )
    corrupted = PlantCheckpoint(
        changed_state,
        checkpoint.digest,
        checkpoint.semantic_provenance_id,
        checkpoint.numeric_revision_id,
        checkpoint.state_schema_id,
        checkpoint.execution_signature_id,
    )
    with pytest.raises(ValueError, match="digest verification"):
        plant.restore(corrupted)

    commands = (
        jnp.ones((3, 2), dtype=jnp.float32),
        jnp.full((3, 2), 0.5, dtype=jnp.float32),
    )
    contexts = (
        PlantStepContext(
            jnp.zeros((3,)), jnp.ones((3,)), jnp.zeros((3,), dtype=jnp.int32)
        ),
        PlantStepContext(
            jnp.ones((3,)),
            jnp.full((3,), 2.0),
            jnp.ones((3,), dtype=jnp.int32),
        ),
    )
    first = plant.step(contexts[0], initial, commands[0], parameters)
    second = plant.step(contexts[1], first.accepted_state, commands[1], parameters)
    first_digest = plant.state_digest(first.accepted_state)
    second_digest = plant.state_digest(second.accepted_state)

    matching = plant.replay(
        checkpoint,
        contexts,
        commands,
        parameters,
        expected_digests=(first_digest, second_digest),
    )
    assert matching.matched
    assert matching.first_mismatch_step == -1
    np.testing.assert_array_equal(matching.successful, [True, True, True])
    np.testing.assert_array_equal(matching.first_failure_step, [-1, -1, -1])
    assert len(matching.accepted_states) == 3
    assert len(matching.step_results) == 2

    mismatch = plant.replay(
        checkpoint,
        contexts,
        commands,
        parameters,
        expected_digests=(first_digest, "0" * 64),
    )
    assert not mismatch.matched
    assert mismatch.first_mismatch_step == 1
    assert mismatch.expected_digest == "0" * 64
    assert mismatch.actual_digest == second_digest
