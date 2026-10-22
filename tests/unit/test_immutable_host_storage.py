# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Numeric host ownership, dynamic leaves and canonical restoration contracts."""

import io

import jax
import numpy as np
import pytest

from phydrax._fingerprint import array_tree_fingerprint
from phydrax._host_storage import immutable_host_array, immutable_host_array_layout
from phydrax._model._structure import (
    deserialize_model_leaf,
    model_from_logical_array_recipe,
    model_from_structure_recipe,
    model_recipe_array_values,
    model_structure_recipe,
    serialize_model_leaf,
)


@pytest.mark.parametrize("dtype", ("?", "i4", "u8", "f4", ">f8", "c16"))
@pytest.mark.parametrize("layout", ("c", "fortran", "strided", "scalar", "empty"))
def test_immutable_host_bank_preserves_exact_dynamic_content(dtype: str, layout: str) -> None:
    values = np.arange(12).astype(dtype).reshape(3, 4)
    if layout == "fortran":
        values = np.asfortranarray(values)
    elif layout == "strided":
        values = values[::-1, ::2]
    elif layout == "scalar":
        values = np.asarray(values[0, 0])
    elif layout == "empty":
        values = values[:0]
    before = array_tree_fingerprint(values)
    sealed = immutable_host_array(values)
    assert sealed.dtype == values.dtype and sealed.shape == values.shape
    assert sealed.tobytes() == values.tobytes()
    assert array_tree_fingerprint(sealed) == before
    assert jax.tree_util.tree_structure(sealed) == jax.tree_util.tree_structure(values)
    assert jax.tree_util.tree_leaves(sealed)[0] is sealed
    assert immutable_host_array_layout(sealed) is not None
    assert immutable_host_array(sealed) is sealed
    assert not np.shares_memory(sealed, values)


def test_immutable_bank_has_no_writable_array_buffer_or_producer_alias() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    expected = values.copy()
    sealed = immutable_host_array(values)
    values[:] = -1
    np.testing.assert_array_equal(sealed, expected)
    aliases = (sealed, sealed.view(), sealed.reshape(-1), np.asarray(sealed), sealed[:, ::2])
    for alias in aliases:
        with pytest.raises(ValueError):
            alias.setflags(write=True)
        with pytest.raises(ValueError):
            alias.flat[0] = 9
        view = memoryview(alias)
        assert view.readonly
        with pytest.raises(TypeError):
            if alias.ndim == 2:
                view[0, 0] = 9
            else:
                view[0] = 9
    owner = sealed
    while isinstance(owner, np.ndarray):
        assert not owner.flags.writeable and not owner.flags.owndata
        with pytest.raises(ValueError):
            owner.setflags(write=True)
        owner = owner.base
    assert type(owner) is bytes
    raw = memoryview(owner)
    with pytest.raises(TypeError):
        raw[0] = 1
    alias = np.frombuffer(raw, dtype=sealed.dtype)
    with pytest.raises(ValueError):
        alias.setflags(write=True)
    with pytest.raises(ValueError):
        alias[0] = 1


@pytest.mark.parametrize("backing", ("owning", "view", "bytearray", "memoryview"))
def test_readonly_flags_never_prove_immutable_storage(backing: str) -> None:
    if backing == "owning":
        values = np.arange(8, dtype=np.int64)
    elif backing == "view":
        values = np.arange(16, dtype=np.int64)[::2]
    elif backing == "bytearray":
        values = np.frombuffer(bytearray(64), dtype=np.int64)
    else:
        values = np.frombuffer(memoryview(bytearray(64)).toreadonly(), dtype=np.int64)
    values.setflags(write=False)
    assert immutable_host_array_layout(values) is None
    sealed = immutable_host_array(values)
    assert sealed is not values
    assert immutable_host_array_layout(sealed) is not None
    np.testing.assert_array_equal(sealed, values)


@pytest.mark.parametrize("mutation", ("shape", "dtype", "strides", "replacement"))
def test_immutable_layout_binds_interpretation_and_bytes_owner(mutation: str) -> None:
    sealed = immutable_host_array(np.arange(12, dtype=np.float64).reshape(3, 4))
    original = immutable_host_array_layout(sealed)
    assert original is not None
    if mutation == "shape":
        object.__setattr__(sealed, "shape", (4, 3))
    elif mutation == "dtype":
        object.__setattr__(sealed, "dtype", np.dtype(np.int64))
    elif mutation == "strides":
        object.__setattr__(sealed, "strides", (0, 8))
    else:
        sealed = immutable_host_array(sealed.copy())
    assert immutable_host_array_layout(sealed) != original


def test_unsupported_object_evidence_is_not_reinterpreted_or_frozen() -> None:
    values = np.asarray([object(), {"failure": "retained"}], dtype=object)
    assert immutable_host_array(values) is values
    assert immutable_host_array_layout(values) is None
    assert values.flags.writeable


@pytest.mark.parametrize("writable", (False, True))
def test_model_recipe_and_leaf_restoration_keep_backend_and_owner_policy(writable: bool) -> None:
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    values.setflags(write=writable)
    recipe = model_structure_recipe((values, values))
    arrays = model_recipe_array_values((values, values), recipe, prefix="host")
    restored = model_from_logical_array_recipe(recipe, arrays, prefix="host")
    assert restored[0] is restored[1]
    assert model_structure_recipe(restored) == recipe
    assert array_tree_fingerprint(restored) == array_tree_fingerprint((values, values))
    assert restored[0].flags.writeable is writable
    assert (immutable_host_array_layout(restored[0]) is None) is writable
    template = model_from_structure_recipe(recipe)
    assert (immutable_host_array_layout(template[0]) is None) is writable
    stream = io.BytesIO()
    serialize_model_leaf(stream, values)
    stream.seek(0)
    leaf = deserialize_model_leaf(stream, values)
    assert leaf.dtype == values.dtype and leaf.shape == values.shape
    np.testing.assert_array_equal(leaf, values)
    assert (immutable_host_array_layout(leaf) is None) is writable
