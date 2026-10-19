"""Registered replay honors owner hooks and authenticates all stored fields."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

from phydrax._fingerprint import tree_content_equal
from phydrax._identity import SemanticProvenance
from phydrax._model._artifacts import (
    register_artifact_value,
    registered_artifact_value_id,
)
from phydrax._strict import StrictModule
from phydrax.geometry._embedding_state import PeriodicAffineQuotientEvidence
from phydrax.lifecycle._meshing_source_records import reconstruct_stored_fields
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    register_meshing_source_artifacts,
    write_meshing_source_closure,
)


class BoundStoredRecord(StrictModule):
    payload: jax.Array
    authority: str = eqx.field(static=True)

    def __init__(self, content: ArrayLike, *, authority: str) -> None:
        self.payload = jnp.asarray(content, dtype=jnp.float64)
        self.authority = authority

    @classmethod
    def from_stored_fields(
        cls: type[BoundStoredRecord], value: BoundStoredRecord, /, *, authority: str
    ) -> BoundStoredRecord:
        return cls(value.payload, authority=authority)


def test_generic_replay_selects_owner_hook_with_retained_binding() -> None:
    if registered_artifact_value_id(BoundStoredRecord) is None:
        register_artifact_value("test:BoundStoredRecord", BoundStoredRecord)
    value = BoundStoredRecord(np.asarray((1.0, -0.0)), authority="retained-owner")
    rebuilt = reconstruct_stored_fields(value, bindings={"authority": value.authority})
    assert tree_content_equal(value, rebuilt)
    with pytest.raises(ValueError, match=r"stored_fields\.BoundStoredRecord.*authority"):
        reconstruct_stored_fields(value, bindings={"authority": "another-owner"})
    with pytest.raises(TypeError, match=r"BoundStoredRecord.bindings"):
        reconstruct_stored_fields(value, bindings={"content": value.payload})


def test_generic_replay_honors_semantic_provenance_owner_hook() -> None:
    register_meshing_source_artifacts()
    value = SemanticProvenance({"source": "authored", "controls": [1, 2]})
    rebuilt = reconstruct_stored_fields(value)
    assert tree_content_equal(value, rebuilt)


class BoundStoredTuple(NamedTuple):
    payload: jax.Array
    authority: str


def test_generic_replay_authenticates_registered_named_tuple_fields() -> None:
    if registered_artifact_value_id(BoundStoredTuple) is None:
        register_artifact_value("test:BoundStoredTuple", BoundStoredTuple)
    value = BoundStoredTuple(
        jnp.asarray((1.0, -0.0), dtype=jnp.float64), "retained-owner"
    )
    rebuilt = reconstruct_stored_fields(value)
    assert tree_content_equal(value, rebuilt)
    assert rebuilt.payload is value.payload
    with pytest.raises(ValueError, match=r"stored_fields\.BoundStoredTuple.*authority"):
        reconstruct_stored_fields(value, bindings={"authority": "another-owner"})


def test_periodic_quotient_evidence_archive_renews_stored_identity(
    tmp_path: Path,
) -> None:
    evidence = PeriodicAffineQuotientEvidence(
        "mesh", "geometry", "periodic-topology", "validity", "affinity", "facet-traces"
    )
    receipt = write_meshing_source_closure(tmp_path / "periodic-evidence.zip", evidence)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert tree_content_equal(restored, evidence)


def test_periodic_quotient_evidence_replay_refuses_changed_certificate_binding() -> None:
    register_meshing_source_artifacts()
    evidence = PeriodicAffineQuotientEvidence(
        "mesh", "geometry", "periodic-topology", "validity", "affinity", "facet-traces"
    )
    changed = copy(evidence)
    object.__setattr__(changed, "affinity_certificate_id", "foreign-affinity")
    with pytest.raises(ValueError, match=r"PeriodicAffineQuotientEvidence\.evidence_id"):
        reconstruct_stored_fields(changed)
