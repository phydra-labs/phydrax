#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import functools
import sys

import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from phydrax._execution_pool import PoolExecutionSignature
from phydrax._identity import (
    ArtifactBindingIdentity,
    callable_payload,
    ExecutableSignature,
    NumericRevision,
    SemanticProvenance,
    strict_module_payload,
)
from phydrax._strict import StrictModule


class _AffineCallable(StrictModule):
    weight: Array
    enabled: bool = eqx.field(static=True)

    def __init__(self, weight, /, *, enabled: bool = True):
        self.weight = jnp.asarray(weight, dtype=jnp.float32)
        self.enabled = bool(enabled)

    def __call__(self, value):
        return self.weight * value if self.enabled else value


class _ActivatedCallable(StrictModule):
    activation: object = eqx.field(static=True)

    def __init__(self, activation, /):
        self.activation = activation

    def __call__(self, value):
        return self.activation(value)


def _square(value, *, scale=1.0):
    return scale * value * value


def _cube(value, *, scale=1.0):
    return scale * value * value * value


_module_lambda = lambda value: value


def test_plain_function_payload_is_content_addressed():
    payload = callable_payload(_square)

    assert callable_payload(_square) == payload
    assert (
        callable_payload(_cube)["semantic_content_id"] != (payload["semantic_content_id"])
    )
    nested = strict_module_payload(_ActivatedCallable(_square))
    assert nested == strict_module_payload(_ActivatedCallable(_square))
    assert (
        nested["semantic_content_id"]
        != (strict_module_payload(_ActivatedCallable(_cube))["semantic_content_id"])
    )


_GLOBAL_WEIGHTS = jnp.asarray([1.0, 2.0])
_GLOBAL_OFFSETS = (jnp.asarray([3], dtype=jnp.int32),)
_GLOBAL_SCALE = 2.0


def _weighted(value):
    return _GLOBAL_WEIGHTS * value


def _offset(value):
    return value + _GLOBAL_OFFSETS[0]


def _calls_weighted(value):
    return _weighted(value) + 1.0


def _scaled(value):
    return jnp.sin(value) * _GLOBAL_SCALE


def test_functions_reading_global_arrays_are_opaque():
    for reader in (_weighted, _offset, _calls_weighted):
        with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
            callable_payload(reader)
        with pytest.raises(TypeError, match="requires explicit semantic and numeric"):
            strict_module_payload(_ActivatedCallable(reader))
        payload = callable_payload(reader, semantic_id="law", numeric_id="weights")
        assert payload["numeric_content_id"] == "weights"


def test_plain_function_identity_follows_scalar_global_values(monkeypatch):
    payload = callable_payload(_scaled)
    assert callable_payload(_scaled) == payload

    monkeypatch.setattr(sys.modules[__name__], "_GLOBAL_SCALE", 3.0)

    assert (
        callable_payload(_scaled)["semantic_content_id"]
        != (payload["semantic_content_id"])
    )


def test_distinct_lambdas_never_share_an_identity():
    first = lambda value: value + 1.0
    second = lambda value: value + 2.0

    for opaque in (first, second, _module_lambda):
        with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
            callable_payload(opaque)
        with pytest.raises(TypeError, match="requires explicit semantic and numeric"):
            strict_module_payload(_ActivatedCallable(opaque))
    first_payload = callable_payload(first, semantic_id="shift", numeric_id="one")
    second_payload = callable_payload(second, semantic_id="shift", numeric_id="two")
    assert first_payload["numeric_content_id"] != second_payload["numeric_content_id"]


def test_opaque_callables_without_ids_are_refused():
    opaque_callables = (
        functools.partial(_square, scale=2.0),
        _AffineCallable([1.0]).__call__,
        jnp.sin,
        _AffineCallable,
    )
    for opaque in opaque_callables:
        with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
            callable_payload(opaque)
    with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
        callable_payload(_square, semantic_id="square-law")


def test_strict_module_payload_separates_semantics_from_numeric_realization():
    first = _AffineCallable([1.0, 2.0])
    second = _AffineCallable([3.0, 4.0])

    first_payload = strict_module_payload(first)
    second_payload = strict_module_payload(second)

    assert first_payload["semantic_content_id"] == second_payload["semantic_content_id"]
    assert first_payload["numeric_content_id"] != second_payload["numeric_content_id"]
    assert callable_payload(first) == first_payload

    provenance = SemanticProvenance(
        first_payload["semantic_payload"],
        resource_ids={"mesh": "mesh-content-47", "law": "linear-law"},
    )
    first_revision = NumericRevision(provenance, first)
    second_revision = NumericRevision(provenance, second)

    assert first_revision.semantic_id == second_revision.semantic_id
    assert first_revision.content_id != second_revision.content_id
    assert first_revision.revision_id != second_revision.revision_id


def test_semantic_content_and_resource_identity_are_independent():
    content = {"law": "linear-elastic", "state_space": "cartesian"}
    first = SemanticProvenance(content, resource_ids={"mesh": "mesh-a"})
    second = SemanticProvenance(content, resource_ids={"mesh": "mesh-b"})

    assert first.content_id == second.content_id
    assert first.semantic_id != second.semantic_id


def test_executable_signature_rejects_numeric_array_values():
    with pytest.raises(TypeError, match="integer sequences"):
        ExecutableSignature(shapes={"state": jnp.asarray([2])})
    with pytest.raises(TypeError, match="not arrays"):
        ExecutableSignature(dtypes={"state": jnp.zeros((2,))})
    with pytest.raises(TypeError, match="cannot contain a numeric array"):
        ExecutableSignature(algorithm_facts={"coefficients": jnp.asarray([1.0, 2.0])})
    with pytest.raises(TypeError, match="cannot contain a numeric array"):
        ExecutableSignature(backend_facts={"device_state": jnp.asarray(1)})


def test_opaque_callable_requires_explicit_semantic_and_numeric_ids():
    offset = 2.0

    def closure(value):
        return value + offset

    with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
        callable_payload(closure)

    payload = callable_payload(
        closure,
        semantic_id="translation-law",
        numeric_id="translation-offset-two",
    )
    assert payload["semantic_content_id"] == "translation-law"
    assert payload["numeric_content_id"] == "translation-offset-two"


def test_pool_signature_delegates_to_the_generic_executable_signature():
    pool = PoolExecutionSignature(
        topology_id="pool-topology",
        method_id="pool-method",
        precision_id="float32",
        backend_id="cpu",
        shard_count=4,
        static_callables={"response": _AffineCallable([2.0])},
    )
    generic = ExecutableSignature(
        topology_ids={"pool": "pool-topology"},
        capacities={"shards": 4, "processes": 1},
        algorithm_facts={"method_id": "pool-method"},
        backend_facts={"backend_id": "cpu", "precision_id": "float32"},
        static_callables={"response": _AffineCallable([2.0])},
    )

    assert pool.signature_id == generic.signature_id
    assert pool.executable_signature.signature_id == generic.signature_id


def test_static_held_weights_are_part_of_the_executable_signature():
    def signature(**callables):
        return ExecutableSignature(
            shapes={"x": (2,)}, dtypes={"x": "float32"}, static_callables=callables
        ).signature_id

    baseline = signature(response=_AffineCallable([1.0, 2.0]))
    assert signature(response=_AffineCallable([1.0, 2.0])) == baseline
    assert signature(response=_AffineCallable([1.0, 3.0])) != baseline
    assert signature(response=_ActivatedCallable(_square)) != signature(
        response=_ActivatedCallable(_cube)
    )
    assert signature() != baseline
    with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
        signature(response=lambda value: value)


def test_artifact_binding_identity_binds_all_three_identities():
    semantic = SemanticProvenance({"kind": "affine-response"})
    revision = NumericRevision(semantic, {"weight": jnp.asarray([1.0, 2.0])})
    signature = ExecutableSignature(shapes={"weight": (2,)})
    binding = ArtifactBindingIdentity(semantic, revision, signature)

    assert (binding.semantic_id, binding.numeric_revision_id) == (
        semantic.semantic_id,
        revision.revision_id,
    )
    assert binding.executable_signature_id == signature.signature_id
    assert ArtifactBindingIdentity.from_record(binding.to_record()) == binding
    assert (
        ArtifactBindingIdentity(
            semantic.semantic_id, revision.revision_id, signature.signature_id
        )
        == binding
    )
    retrained = NumericRevision(semantic, {"weight": jnp.asarray([1.0, 5.0])})
    assert (
        ArtifactBindingIdentity(semantic, retrained, signature).binding_id
        != binding.binding_id
    )
    foreign = NumericRevision(SemanticProvenance({"kind": "other"}), {})
    with pytest.raises(ValueError, match="another semantic provenance"):
        ArtifactBindingIdentity(semantic, foreign, signature)
    record = binding.to_record()
    with pytest.raises(ValueError, match="corrupt"):
        ArtifactBindingIdentity.from_record(
            {**record, "numeric_revision_id": retrained.revision_id}
        )
    with pytest.raises(ValueError, match="fields do not match"):
        ArtifactBindingIdentity.from_record(
            {key: value for key, value in record.items() if key != "binding_id"}
        )
