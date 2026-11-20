"""Unqualified entity IDs of a meshing scope: one validated static host leaf."""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.meshing._model_codec import (
    MODEL_ARCHIVE_LIMITS,
    read_model_artifact,
    restore_model_archive_value,
    write_model_artifact,
)
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope


def _scope() -> MeshingScope:
    return MeshingScope(
        "mesh",
        "revision",
        MeshingEntityKind.MESH,
        3,
        "cells",
        np.asarray((11, 3, 7), dtype=np.int64),
    )


def test_unqualified_ids_are_a_tuple_held_as_one_static_leaf() -> None:
    scope = _scope()
    bank = scope.unqualified_entity_ids
    plain = tuple(bank)
    assert isinstance(bank, tuple)
    assert plain == (3, 7, 11)
    assert bank == plain and hash(bank) == hash(plain)
    assert jax.tree_util.tree_leaves(bank) == [bank]
    restored = pickle.loads(pickle.dumps(bank))
    assert restored == bank and type(restored) is type(bank)
    # Static fields stay out of the dynamic leaves.
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(scope))


def test_explicit_unqualified_ids_keep_the_scope_identity() -> None:
    scope = _scope()
    explicit = MeshingScope(
        "mesh",
        "revision",
        MeshingEntityKind.MESH,
        3,
        "cells",
        np.asarray((3, 7, 11), dtype=np.int64),
        unqualified_entity_ids=(11, 7, 3),
    )
    assert explicit.scope_id == scope.scope_id
    assert explicit.unqualified_entity_ids == scope.unqualified_entity_ids
    with pytest.raises(ValueError, match="projection"):
        MeshingScope(
            "mesh",
            "revision",
            MeshingEntityKind.MESH,
            3,
            "cells",
            np.asarray((3, 7, 11), dtype=np.int64),
            unqualified_entity_ids=(3, 7),
        )


def test_pickled_scope_keeps_its_identity() -> None:
    scope = _scope()
    restored = pickle.loads(pickle.dumps(scope))
    assert restored.scope_id == scope.scope_id
    assert restored.unqualified_entity_ids == scope.unqualified_entity_ids


def test_archived_scope_restores_its_identity(tmp_path: Path) -> None:
    scope = _scope()
    path = write_model_artifact(
        tmp_path / "scope.phx", scope, limits=MODEL_ARCHIVE_LIMITS
    )
    restored = read_model_artifact(
        path,
        expected_type=MeshingScope,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert type(restored) is MeshingScope
    assert restored.scope_id == scope.scope_id
    assert restored.unqualified_entity_ids == scope.unqualified_entity_ids


def test_unqualified_ids_holding_a_jax_array_are_refused() -> None:
    with pytest.raises(ValueError):
        MeshingScope(
            "mesh",
            "revision",
            MeshingEntityKind.MESH,
            3,
            "cells",
            np.asarray((3, 7), dtype=np.int64),
            unqualified_entity_ids=(jnp.asarray(3), 7),  # ty: ignore[invalid-argument-type]
        )
