#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from phydrax._execution_pool import PoolExecutionSignature
from phydrax._identity import (
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


def _executable_signature() -> ExecutableSignature:
    return ExecutableSignature(
        shapes={"state": (2,), "parameter": (2,)},
        dtypes={"state": jnp.float32, "parameter": jnp.float32},
        space_ids={"state": "cartesian-state"},
        topology_ids={"plant": "two-state-topology"},
        capacities={"cases": 8},
        algorithm_facts={"integrator": "fixed-step"},
        backend_facts={"platform": "cpu", "precision": "float32"},
    )


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


def test_numeric_revisions_do_not_change_executable_signature():
    provenance = SemanticProvenance({"plant": "affine"})
    first = NumericRevision(provenance, {"weight": jnp.asarray([1.0, 2.0])})
    second = NumericRevision(provenance, {"weight": jnp.asarray([3.0, 4.0])})
    first_signature = _executable_signature()
    second_signature = _executable_signature()

    assert first.revision_id != second.revision_id
    assert first_signature.signature_id == second_signature.signature_id


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
    )
    generic = ExecutableSignature(
        topology_ids={"pool": "pool-topology"},
        capacities={"shards": 4},
        algorithm_facts={"method_id": "pool-method"},
        backend_facts={"backend_id": "cpu", "precision_id": "float32"},
    )

    assert pool.signature_id == generic.signature_id
    assert pool.executable_signature.signature_id == generic.signature_id
