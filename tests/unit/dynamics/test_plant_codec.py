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
from phydrax.dynamics._layout import StateLayout
from phydrax.dynamics._plant_codec import (
    ControlVectorCodec,
    EncodedControl,
    EncodedPlantState,
    EncodedPlantVector,
    PlantStateVectorCodec,
)
from phydrax.linalg import ArraySpace
from phydrax.metrix._quaternion_state_geometry import (
    ScalarFirstQuaternionStateGeometry,
)


def _identities(*, point_shape=(4,), point_dtype=np.float32):
    semantic = SemanticProvenance(
        {"plant": "codec-test", "state_convention": "complete-payload"},
        resource_ids={"model": "codec-test-model"},
    )
    numeric = NumericRevision(
        semantic,
        {"gain": jnp.asarray([0.25, 0.5], dtype=jnp.float32)},
    )
    executable = ExecutableSignature(
        shapes={"point": point_shape},
        dtypes={"point": point_dtype},
        space_ids={"point": "codec-test-point"},
        topology_ids={"plant": "codec-test-topology"},
        algorithm_facts={"codec": "complete-payload"},
        backend_facts={"precision": np.dtype(point_dtype).str},
    )
    return semantic, numeric, executable


def _assert_tree_exact(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert actual_leaf.shape == expected_leaf.shape
        assert actual_leaf.dtype == expected_leaf.dtype
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_mixed_pytree_state_and_control_round_trip_exactly():
    template = {
        "configuration": {"position": jnp.asarray([1.0, -2.0], dtype=jnp.float32)},
        "memory": (
            jnp.asarray(3.0, dtype=jnp.float32),
            {"strain": jnp.asarray([4.0], dtype=jnp.float32)},
        ),
        "mode": jnp.asarray(2, dtype=jnp.int32),
    }
    layout = StateLayout(
        (2, 2),
        local_space=ArraySpace((2, 2), dtype=np.float32),
        tangent_space=ArraySpace((2, 2), dtype=np.float32),
    )
    schema = ArrayPyTreeSchema.from_tree(template, case_ndim=0)
    mode_path = next(path for path in schema.leaf_paths if "mode" in path)
    semantic, numeric, executable = _identities(
        point_shape=layout.shape, point_dtype=np.float32
    )
    codec = PlantStateVectorCodec(
        schema,
        layout,
        template,
        (mode_path,),
        semantic_provenance=semantic,
        numeric_revision=numeric,
        executable_signature=executable,
    )

    state = {
        "configuration": {"position": jnp.asarray([8.0, 7.0], dtype=jnp.float32)},
        "memory": (
            jnp.asarray(6.0, dtype=jnp.float32),
            {"strain": jnp.asarray([5.0], dtype=jnp.float32)},
        ),
        "mode": jnp.asarray(2, dtype=jnp.int32),
    }
    encoded = codec.encode_point(state)

    assert isinstance(encoded, EncodedPlantState)
    assert encoded.vector.shape == (2, 2)
    assert encoded.semantic_id == semantic.semantic_id
    assert encoded.numeric_revision_id == numeric.revision_id
    assert encoded.schema_id == schema.schema_id
    assert encoded.executable_signature_id == executable.signature_id
    _assert_tree_exact(codec.decode_point(encoded), state)

    command = {
        "actuator": jnp.asarray([0.5, -0.25], dtype=jnp.float32),
        "feedforward": (jnp.asarray(0.75, dtype=jnp.float32),),
    }
    command_schema = ArrayPyTreeSchema.from_tree(
        command, case_ndim=0, schema_id="complete-control-v1"
    )
    control_codec = ControlVectorCodec(
        command_schema,
        semantic_provenance=semantic,
        numeric_revision=numeric,
        executable_signature=executable,
    )
    encoded_command = control_codec.encode_command(command)

    assert isinstance(encoded_command, EncodedControl)
    assert encoded_command.vector.shape == (3,)
    _assert_tree_exact(control_codec.decode_command(encoded_command), command)


def test_quaternion_point_four_local_three_and_power_duality():
    template = {
        "orientation": jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        "mode": jnp.asarray(1, dtype=jnp.int32),
    }
    schema = ArrayPyTreeSchema.from_tree(template, case_ndim=0)
    mode_path = next(path for path in schema.leaf_paths if "mode" in path)
    layout = StateLayout(
        (4,),
        geometry=ScalarFirstQuaternionStateGeometry(tolerance=1.0e-6),
        local_space=ArraySpace((3,), dtype=np.float32),
        tangent_space=ArraySpace((3,), dtype=np.float32),
    )
    semantic, numeric, executable = _identities(point_shape=(4,), point_dtype=np.float32)
    codec = PlantStateVectorCodec(
        schema,
        layout,
        template,
        (mode_path,),
        semantic_provenance=semantic,
        numeric_revision=numeric,
        executable_signature=executable,
    )

    state = codec.encode_point(template)
    step = codec.encode_local(jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32))
    velocity = codec.encode_local(jnp.asarray([-0.3, 0.4, 0.15], dtype=jnp.float32))
    cotangent = codec.encode_cotangent(jnp.asarray([0.7, -0.2, 0.6], dtype=jnp.float32))

    point = codec.retract(state, step)
    recovered_step = codec.inverse_retract(state, point)
    tangent = codec.retraction_jvp(state, step, velocity)
    recovered_velocity = codec.retraction_inverse_jvp(state, point, tangent)
    local_cotangent = codec.retraction_vjp(state, step, cotangent)
    evidence = codec.power_evidence(state, step, velocity, cotangent)

    assert state.vector.shape == (4,)
    assert step.vector.shape == (3,)
    assert point.vector.shape == (4,)
    assert tangent.vector.shape == (3,)
    assert local_cotangent.vector.shape == (3,)
    np.testing.assert_allclose(recovered_step.vector, step.vector, atol=2.0e-6)
    np.testing.assert_allclose(recovered_velocity.vector, velocity.vector, atol=2.0e-6)
    np.testing.assert_allclose(evidence.physical_power, evidence.local_power, atol=2.0e-6)
    np.testing.assert_allclose(evidence.absolute_residual, 0.0, atol=2.0e-6)
    assert bool(evidence.finite)
    assert bool(evidence.valid)

    physical_power = layout.cotangent_space.pair(
        codec.decode_cotangent(cotangent), codec.decode_tangent(evidence.tangent)
    )
    local_power = layout.local_cotangent_space.pair(
        codec.decode_local_cotangent(evidence.local_cotangent),
        codec.decode_local(velocity),
    )
    np.testing.assert_allclose(evidence.physical_power, physical_power)
    np.testing.assert_allclose(evidence.local_power, local_power)


def test_dynamic_discrete_modes_round_trip_and_bind_fixed_mode_operations():
    template = {
        "configuration": jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        "memory": jnp.asarray([0.25], dtype=jnp.float32),
        "contact": jnp.asarray(False, dtype=jnp.bool_),
        "mode": jnp.asarray(3, dtype=jnp.int32),
    }
    schema = ArrayPyTreeSchema.from_tree(template, case_ndim=0)
    discrete_paths = tuple(
        leaf.path for leaf in schema.leaves if leaf.dtype.kind in "biu"
    )
    semantic, numeric, executable = _identities(point_shape=(3,), point_dtype=np.float32)

    with pytest.raises(ValueError, match="dynamic inexact state leaf"):
        PlantStateVectorCodec(
            schema,
            StateLayout(
                (2,),
                local_space=ArraySpace((2,), dtype=np.float32),
                tangent_space=ArraySpace((2,), dtype=np.float32),
            ),
            template,
            discrete_paths,
            semantic_provenance=semantic,
            numeric_revision=numeric,
            executable_signature=executable,
        )

    layout = StateLayout(
        (3,),
        local_space=ArraySpace((3,), dtype=np.float32),
        tangent_space=ArraySpace((3,), dtype=np.float32),
    )
    codec = PlantStateVectorCodec(
        schema,
        layout,
        template,
        (),
        semantic_provenance=semantic,
        numeric_revision=numeric,
        executable_signature=executable,
    )
    changed_mode = {
        **template,
        "contact": jnp.asarray(True, dtype=jnp.bool_),
        "mode": jnp.asarray(4, dtype=jnp.int32),
    }
    encoded = codec.encode_point(changed_mode)
    _assert_tree_exact(codec.decode_point(encoded), changed_mode)
    assert encoded.mode_paths == discrete_paths
    assert encoded.vector.dtype == jnp.float32
    assert tuple(value.dtype for value in encoded.mode_values) == (
        jnp.bool_,
        jnp.int32,
    )

    replacement = codec.replace_point_vector(
        encoded, jnp.asarray([8.0, 7.0, 6.0], dtype=jnp.float32)
    )
    replaced = codec.decode_point(replacement)
    assert jnp.array_equal(replaced["contact"], changed_mode["contact"])
    assert jnp.array_equal(replaced["mode"], changed_mode["mode"])

    zero = jnp.zeros((3,), dtype=jnp.float32)
    fixed_local = codec.encode_local(zero, anchor=encoded)
    tangent = codec.retraction_jvp(encoded, fixed_local, fixed_local)
    assert tangent.mode_role == "tangent"
    for actual, expected in zip(tangent.mode_values, encoded.mode_values, strict=True):
        np.testing.assert_array_equal(actual, expected)

    template_encoded = codec.encode_point(template)
    switched_local = codec.encode_local(zero, anchor=template_encoded)
    with pytest.raises(eqx.EquinoxRuntimeError, match="mode change"):
        codec.retraction_jvp(
            encoded, switched_local, switched_local
        ).vector.block_until_ready()
    with pytest.raises(eqx.EquinoxRuntimeError, match="mode change"):
        codec.inverse_retract(encoded, template_encoded).vector.block_until_ready()

    stale_sidecar = eqx.tree_at(lambda point: point.mode, encoded, fixed_local.mode)
    with pytest.raises(ValueError, match="sidecar.*role"):
        codec.decode_point(stale_sidecar)

    mutated_mode = eqx.tree_at(
        lambda sidecar: sidecar.values,
        encoded.mode,
        (
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.asarray(4, dtype=jnp.int32),
        ),
    )
    mutated = eqx.tree_at(lambda point: point.mode, encoded, mutated_mode)
    with pytest.raises(eqx.EquinoxRuntimeError, match="bound identity"):
        codec.decode_point(mutated)["configuration"].block_until_ready()

    immutable_codec = PlantStateVectorCodec(
        schema,
        layout,
        template,
        discrete_paths,
        semantic_provenance=semantic,
        numeric_revision=numeric,
        executable_signature=executable,
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="immutable mode"):
        immutable_codec.encode_point(changed_mode).vector.block_until_ready()


def test_stale_provenance_shape_dtype_and_vector_role_are_rejected():
    template = {
        "state": jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        "mode": jnp.asarray(0, dtype=jnp.int32),
    }
    schema = ArrayPyTreeSchema.from_tree(template, case_ndim=0)
    mode_path = next(path for path in schema.leaf_paths if "mode" in path)
    layout = StateLayout(
        (2,),
        local_space=ArraySpace((2,), dtype=np.float32),
        tangent_space=ArraySpace((2,), dtype=np.float32),
    )
    semantic, numeric, executable = _identities(point_shape=(2,), point_dtype=np.float32)
    codec = PlantStateVectorCodec(
        schema,
        layout,
        template,
        (mode_path,),
        semantic_provenance=semantic,
        numeric_revision=numeric,
        executable_signature=executable,
    )
    encoded = codec.encode_point(template)
    stale_semantic = SemanticProvenance(
        {"plant": "codec-test", "state_convention": "complete-payload"},
        resource_ids={"model": "stale-model"},
    )
    stale = EncodedPlantState(
        encoded.vector,
        semantic_id=stale_semantic.semantic_id,
        numeric_revision_id=encoded.numeric_revision_id,
        schema_id=encoded.schema_id,
        executable_signature_id=encoded.executable_signature_id,
        codec_id=encoded.codec_id,
    )
    with pytest.raises(ValueError, match="provenance"):
        codec.decode_point(stale)

    wrong_shape = EncodedPlantState(
        jnp.zeros((3,), dtype=jnp.float32),
        semantic_id=encoded.semantic_id,
        numeric_revision_id=encoded.numeric_revision_id,
        schema_id=encoded.schema_id,
        executable_signature_id=encoded.executable_signature_id,
        codec_id=encoded.codec_id,
    )
    with pytest.raises(ValueError, match="shape"):
        codec.decode_point(wrong_shape)

    wrong_dtype = EncodedPlantState(
        jnp.zeros((2,), dtype=jnp.complex64),
        semantic_id=encoded.semantic_id,
        numeric_revision_id=encoded.numeric_revision_id,
        schema_id=encoded.schema_id,
        executable_signature_id=encoded.executable_signature_id,
        codec_id=encoded.codec_id,
    )
    with pytest.raises(TypeError, match="dtype"):
        codec.decode_point(wrong_dtype)

    local = codec.encode_local(jnp.asarray([0.1, 0.2], dtype=jnp.float32))
    with pytest.raises(ValueError, match="cannot be used"):
        codec.retraction_vjp(encoded, local, local)

    malformed_cotangent = EncodedPlantVector(
        jnp.ones((1,), dtype=jnp.float32),
        "cotangent",
        semantic_id=local.semantic_id,
        numeric_revision_id=local.numeric_revision_id,
        schema_id=local.schema_id,
        executable_signature_id=local.executable_signature_id,
        codec_id=local.codec_id,
    )
    with pytest.raises(ValueError, match="shape"):
        codec.retraction_vjp(encoded, local, malformed_cotangent)


def test_identity_chain_must_match_before_a_codec_can_exist():
    template = {"state": jnp.asarray([1.0], dtype=jnp.float32)}
    schema = ArrayPyTreeSchema.from_tree(template, case_ndim=0)
    layout = StateLayout(
        (1,),
        local_space=ArraySpace((1,), dtype=np.float32),
        tangent_space=ArraySpace((1,), dtype=np.float32),
    )
    semantic, _, executable = _identities(point_shape=(1,), point_dtype=np.float32)
    other_semantic = SemanticProvenance({"plant": "other"})
    stale_revision = NumericRevision(
        other_semantic, {"gain": jnp.asarray(1.0, dtype=jnp.float32)}
    )

    with pytest.raises(ValueError, match="semantic identity"):
        PlantStateVectorCodec(
            schema,
            layout,
            template,
            (),
            semantic_provenance=semantic,
            numeric_revision=stale_revision,
            executable_signature=executable,
        )
