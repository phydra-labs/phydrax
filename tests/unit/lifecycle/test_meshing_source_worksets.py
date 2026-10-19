# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Scientific source admission worksets do not traverse overlapping model DAGs."""

from dataclasses import fields
from typing import Any

import equinox as eqx
import jax
import numpy as np
import pytest

from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._fingerprint import logical_array_value_collection_digest, tree_content_equal
from phydrax.discretization.fem._restoration import RestorationValidationContext
from phydrax.lifecycle._meshing_sources import (
    _collect_source_values,
    _SourceCollection,
    _SourceVisitContext,
    MeshingSourceValidation,
)
from tests._support.cell_meshes import unit_square
from tools.compile_counter import CompileCounter


def test_source_workset_construction_performs_zero_pytree_leaf_walks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = unit_square()
    original = _collect_source_values(mesh)
    values = {member.name: getattr(original, member.name) for member in fields(original)}
    restoration = RestorationValidationContext()
    original_leaves = jax.tree_util.tree_leaves
    walks: list[object] = []

    def counted_leaves(value: Any, *args: Any, **kwargs: Any) -> Any:
        walks.append(value)
        return original_leaves(value, *args, **kwargs)

    monkeypatch.setattr(jax.tree_util, "tree_leaves", counted_leaves)
    collection = _SourceCollection(**values)
    context = _SourceVisitContext(
        collection, DEFAULT_ARRAY_ARCHIVE_LIMITS, restoration, None, set(), set()
    )
    assert walks == []
    assert context.collection is collection
    assert context.restoration is restoration
    assert collection.meshes == (mesh,)
    for member in fields(collection):
        assert getattr(collection, member.name) is getattr(original, member.name)
    context.active.add(id(mesh))
    context.visited.add((id(mesh), 0, None))
    assert id(mesh) in context.active
    assert (id(mesh), 0, None) in context.visited


class _StaticSourceMetadata(eqx.Module):
    values: jax.Array
    metadata: float = eqx.field(static=True)


def _empty_source_validation() -> MeshingSourceValidation:
    return MeshingSourceValidation(
        (), {}, logical_array_value_collection_digest({}), maximum_chunk_bytes=8
    )


def test_admitted_local_arrays_compare_signed_zero_and_nan_payloads_bitwise() -> None:
    proof = _empty_source_validation()
    bits = np.asarray(
        [0, 0x8000000000000000, 0x7FF8000000000001, 0x7FF8000000000002],
        dtype=np.uint64,
    )
    original = bits.view(np.float64)
    signed_zero = bits.copy()
    signed_zero[0] = np.uint64(0x8000000000000000)
    nan_payload = bits.copy()
    nan_payload[2] = np.uint64(0x7FF8000000000002)
    assert proof.array_equal(original, original.copy())
    assert not proof.array_equal(original, signed_zero.view(np.float64))
    assert not proof.array_equal(original, nan_payload.view(np.float64))
    assert not proof.array_equal(original, original.astype(np.float32))
    assert not proof.array_equal(original, original.reshape((2, 2)))


def test_array_comparison_retains_exact_static_metadata_and_array_metadata() -> None:
    proof = _empty_source_validation()
    values = jax.device_put(np.asarray([1.0, 2.0], dtype=np.float64))
    repeated = jax.device_put(np.asarray([1.0, 2.0], dtype=np.float64))
    nan_one = np.asarray([0x7FF8000000000001], dtype=np.uint64).view(np.float64)[0].item()
    nan_two = np.asarray([0x7FF8000000000002], dtype=np.uint64).view(np.float64)[0].item()
    first = _StaticSourceMetadata(values, nan_one)
    equal = _StaticSourceMetadata(repeated, nan_one)
    changed_nan = _StaticSourceMetadata(repeated, nan_two)
    assert tree_content_equal(first, equal)
    assert tree_content_equal(first, equal, array_equal=proof.array_equal)
    assert not tree_content_equal(first, changed_nan)
    assert not tree_content_equal(first, changed_nan, array_equal=proof.array_equal)
    assert not tree_content_equal(
        _StaticSourceMetadata(values, 0.0),
        _StaticSourceMetadata(repeated, -0.0),
        array_equal=proof.array_equal,
    )
    wrong_shape = jax.device_put(np.asarray([[1.0, 2.0]], dtype=np.float64))
    wrong_dtype = jax.device_put(np.asarray([1.0, 2.0], dtype=np.float32))
    assert not tree_content_equal(values, wrong_shape, array_equal=proof.array_equal)
    assert not tree_content_equal(values, wrong_dtype, array_equal=proof.array_equal)


@pytest.mark.parametrize("shape", ((), (7, 5)))
@pytest.mark.parametrize("dtype", (np.bool_, np.int64, np.float64, np.complex128))
def test_cold_cpu_source_byte_comparison_compiles_zero(
    shape: tuple[int, ...],
    dtype: type,
) -> None:
    proof = _empty_source_validation()
    size = int(np.prod(shape)) if shape else 1
    values = np.arange(size).astype(dtype).reshape(shape)
    changed = values.copy()
    changed.flat[0] = 1
    first = jax.device_put(values)
    equal = jax.device_put(values.copy())
    different = jax.device_put(changed)
    with CompileCounter() as compiler:
        assert proof.array_equal(first, first)
        assert proof.array_equal(first, equal)
        assert not proof.array_equal(first, different)
    assert compiler.count == 0


def test_cold_cpu_source_comparison_preserves_float_bits_without_compilation() -> None:
    proof = _empty_source_validation()
    bits = np.asarray(
        (0, 0x8000000000000000, 0x7FF8000000000001, 0x7FF8000000000002),
        dtype=np.uint64,
    )
    signed_zero = bits.copy()
    signed_zero[0] = np.uint64(0x8000000000000000)
    nan_payload = bits.copy()
    nan_payload[2] = np.uint64(0x7FF8000000000002)
    first, equal, changed_zero, changed_nan = (
        jax.device_put(value.view(np.float64).reshape((2, 2)))
        for value in (bits, bits.copy(), signed_zero, nan_payload)
    )
    with CompileCounter() as compiler:
        assert proof.array_equal(first, equal)
        assert not proof.array_equal(first, changed_zero)
        assert not proof.array_equal(first, changed_nan)
    assert compiler.count == 0
