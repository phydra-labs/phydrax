"""Live catalogs preserve legacy names and indices without expanding aliases."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from typing import Any, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._model import _structure as structure
from phydrax._model._artifacts import register_artifact_value
from phydrax._strict import StrictModule
from phydrax.typing import HostFloat32, HostInt32


@final
class _StaticOwner(StrictModule):
    __strict_contract__ = True

    runtime: HostFloat32[Literal[2]] | HostInt32[Literal[1]]
    static: Any = eqx.field(static=True)


@jax.tree_util.register_pytree_with_keys_class
@dataclasses.dataclass(frozen=True)
class _KeyedOwner:
    first: Any
    second: Any

    def tree_flatten_with_keys(self) -> tuple[tuple[tuple[Any, Any], ...], None]:
        return (
            (jax.tree_util.GetAttrKey("second"), self.second),
            (jax.tree_util.GetAttrKey("first"), self.first),
        ), None

    def tree_flatten(self) -> tuple[tuple[Any, Any], None]:
        return (self.second, self.first), None

    @classmethod
    def tree_unflatten(cls, _aux: None, children: tuple[Any, Any]) -> _KeyedOwner:
        second, first = children
        return cls(first, second)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Branch:
    left: Any
    right: Any


@dataclasses.dataclass(frozen=True)
class _OpaqueOwner:
    array: np.ndarray


for _owner in (_StaticOwner, _KeyedOwner, _Branch, _OpaqueOwner):
    register_artifact_value(f"test:model-live-catalog:{_owner.__name__}", _owner)


def _legacy_inventory(
    recipe: dict[str, Any], prefix: str
) -> tuple[structure.ModelRecipeArray, ...]:
    """Keep the pre-cutover full-flatten inventory as an independent shallow oracle."""
    table = structure._lower_recipe(recipe, DEFAULT_ARRAY_ARCHIVE_LIMITS)
    descriptors: dict[int, int] = {}

    def descriptor(index: int, node: Mapping[str, Any], /) -> Any:
        result = structure._array_descriptor(node)
        descriptors[id(result)] = index
        return result

    restored = structure.restore_nodes(table.nodes, descriptor, template="descriptors")
    ordered: list[tuple[int, int, str]] = []
    visited: set[int] = set()
    for tree_index, (path, leaf) in enumerate(
        jax.tree_util.tree_flatten_with_path(restored[-1])[0]
    ):
        if isinstance(leaf, structure._RecipeArrayDescriptor):
            index = descriptors[id(leaf)]
            if index not in visited:
                ordered.append(
                    (index, tree_index, jax.tree_util.keystr(path) or "<root>")
                )
                visited.add(index)
    paths = structure._first_paths(table.nodes)
    ordered.extend(
        (index, -1, paths[index])
        for index in table.ordinals
        if table.nodes[index]["kind"] in ("array", "prng_key") and index not in visited
    )
    result: list[structure.ModelRecipeArray] = []
    for position, (index, tree_index, path) in enumerate(ordered):
        declared = structure._array_descriptor(table.nodes[index])
        result.append(
            structure.ModelRecipeArray(
                name=f"{prefix}/{position:06d}",
                tree_index=tree_index,
                path=path,
                shape=declared.shape,
                dtype=declared.dtype,
                backend=declared.backend,
                implementation=declared.implementation,
            )
        )
    return tuple(result)


def _canonical(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _mixed_value() -> tuple[Any, ...]:
    shared = np.asarray((1.0, -0.0), dtype=np.float32)
    shared.setflags(write=False)
    static = np.asarray((2, 3), dtype=np.int16)
    owner = structure.restore_registered_fields(
        _StaticOwner, {"runtime": shared, "static": (static, shared)}
    )
    keyed = _KeyedOwner(jnp.asarray((4,), dtype=jnp.int32), ("leaf", shared))
    return (
        owner,
        owner,
        {"z": keyed, "a": (None, (), {}, shared)},
        jax.random.key(7, impl="threefry2x32"),
        _OpaqueOwner(np.asarray((9,), dtype=np.uint8)),
    )


def test_live_catalog_matches_full_flatten_legacy_inventory() -> None:
    value = _mixed_value()
    expected_recipe = structure.model_structure_recipe(value)
    expected_inventory = _legacy_inventory(expected_recipe, "fields")
    expected_values = structure.model_recipe_array_values(
        value, expected_recipe, prefix="fields"
    )
    recipe, inventory, arrays = structure.prepare_model_recipe_catalog(
        value, prefix="fields"
    )
    assert _canonical(recipe) == _canonical(expected_recipe)
    assert tuple(map(dataclasses.asdict, inventory)) == tuple(
        map(dataclasses.asdict, expected_inventory)
    )
    assert structure.model_recipe_array_inventory(recipe, prefix="fields") == inventory
    assert tuple(arrays) == tuple(expected_values)
    for entry in inventory:
        expected, actual = expected_values[entry.name], arrays[entry.name]
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
        if entry.backend != "prng_key":
            assert actual is expected
    assert tuple(entry.backend for entry in inventory) == (
        "numpy",
        "jax",
        "prng_key",
        "numpy",
        "numpy",
    )
    assert tuple(entry.tree_index for entry in inventory[-2:]) == (-1, -1)
    assert arrays[inventory[0].name] is value[0].runtime
    runtime_array = arrays[inventory[0].name]
    assert isinstance(runtime_array, np.ndarray)
    assert not runtime_array.flags.writeable
    assert arrays[inventory[-2].name] is value[0].static[0]
    assert arrays[inventory[-1].name] is value[-1].array
    prepared_recipe, prepared_values = structure.prepare_model_recipe(
        value, prefix="fields"
    )
    assert _canonical(prepared_recipe) == _canonical(recipe)
    assert tuple(prepared_values) == tuple(arrays)
    for name in arrays:
        assert (
            np.asarray(prepared_values[name]).tobytes()
            == np.asarray(arrays[name]).tobytes()
        )


@pytest.mark.parametrize("implementation", ("threefry2x32", "rbg", "unsafe_rbg"))
def test_live_catalog_preserves_prng_payload_and_aliases(implementation: str) -> None:
    key = jax.random.key(11, impl=implementation)
    value = (key, key, np.asarray(5, dtype=np.int32))
    recipe, inventory, arrays = structure.prepare_model_recipe_catalog(
        value, prefix="keys"
    )
    assert inventory == _legacy_inventory(recipe, "keys")
    assert len(arrays) == 2
    assert inventory[0].implementation == implementation
    assert inventory[0].backend == "prng_key"
    assert inventory[1].tree_index == 2
    np.testing.assert_array_equal(arrays[inventory[0].name], jax.random.key_data(key))


@pytest.mark.parametrize("value", (None, (), [], {}, (None, (), {}, "text", 3)))
def test_live_catalog_preserves_empty_and_nonarray_trees(value: Any) -> None:
    recipe, inventory, arrays = structure.prepare_model_recipe_catalog(
        value, prefix="empty"
    )
    assert _canonical(recipe) == _canonical(structure.model_structure_recipe(value))
    assert inventory == ()
    assert arrays == {}


def test_live_catalog_handles_root_array() -> None:
    value = np.asarray((1, 2), dtype=np.int32)
    recipe, inventory, arrays = structure.prepare_model_recipe_catalog(
        value, prefix="root"
    )
    assert inventory == _legacy_inventory(recipe, "root")
    assert inventory[0].tree_index == 0
    assert inventory[0].path == "<root>"
    assert arrays[inventory[0].name] is value


def test_live_catalog_encodes_once_without_descriptor_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _mixed_value()
    encode = structure._structure_recipe
    lower = structure._lower_recipe
    counts = {"root_encode": 0, "lower": 0}

    def counted_encode(current: Any, path: str, shared: Any, /) -> dict[str, Any]:
        if current is value:
            counts["root_encode"] += 1
        return encode(current, path, shared)

    def counted_lower(
        recipe: Any, limits: ArrayArchiveLimits, /
    ) -> structure.RecipeTable:
        counts["lower"] += 1
        return lower(recipe, limits)

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail(
            "Live catalog must not restore descriptors or rebind the recipe graph"
        )

    monkeypatch.setattr(structure, "_structure_recipe", counted_encode)
    monkeypatch.setattr(structure, "_lower_recipe", counted_lower)
    monkeypatch.setattr(structure, "_array_descriptor", refuse)
    monkeypatch.setattr(structure, "restore_nodes", refuse)
    monkeypatch.setattr(structure, "model_array_catalog", refuse)
    monkeypatch.setattr(structure, "_table_objects", refuse)
    _, inventory, arrays = structure.prepare_model_recipe_catalog(value, prefix="live")
    assert len(inventory) == len(arrays) == 5
    assert counts == {"root_encode": 1, "lower": 1}


@pytest.mark.parametrize(
    ("limits", "message"),
    (
        (
            dataclasses.replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_array_rank=1),
            "array-rank",
        ),
        (
            dataclasses.replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_axis_length=1),
            "axis-length",
        ),
        (
            dataclasses.replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_array_elements=3),
            "element limit",
        ),
        (
            dataclasses.replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_total_array_elements=7),
            "recipe.elements",
        ),
        (
            dataclasses.replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_member_bytes=15),
            "array-byte",
        ),
        (
            dataclasses.replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_aggregate_bytes=31),
            "recipe.array_bytes",
        ),
        (
            dataclasses.replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_manifest_bytes=1),
            "recipe.bytes",
        ),
        (
            dataclasses.replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_dtype_itemsize=2),
            "dtype",
        ),
        (
            dataclasses.replace(
                DEFAULT_ARRAY_ARCHIVE_LIMITS, allowed_dtype_kinds=frozenset({"i"})
            ),
            "dtype",
        ),
    ),
)
def test_live_catalog_enforces_supplied_recipe_limits(
    limits: ArrayArchiveLimits, message: str
) -> None:
    value = (np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match=message):
        structure.prepare_model_recipe_catalog(value, prefix="limited", limits=limits)


def test_live_catalog_counts_shared_arrays_once_under_limits() -> None:
    array = np.ones((2, 2), dtype=np.float32)
    limits = dataclasses.replace(
        DEFAULT_ARRAY_ARCHIVE_LIMITS,
        max_member_bytes=16,
        max_aggregate_bytes=16,
        max_total_array_elements=4,
    )
    _, inventory, arrays = structure.prepare_model_recipe_catalog(
        (array, array), prefix="shared", limits=limits
    )
    assert len(inventory) == len(arrays) == 1
    assert arrays[inventory[0].name] is array


@pytest.mark.parametrize("prefix", ("", None, 3))
def test_live_catalog_rejects_invalid_prefix(prefix: Any) -> None:
    with pytest.raises(ValueError, match="recipe.prefix"):
        structure.prepare_model_recipe_catalog(None, prefix=prefix)


def test_shared_dag_preserves_large_conceptual_tree_index_with_bounded_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = 30
    first = np.asarray((1,), dtype=np.int32)
    last = np.asarray((2,), dtype=np.int32)
    branch: Any = ("nonarray", first, None, (), {})
    for _ in range(depth):
        branch = _Branch(branch, branch)
    value = (branch, branch, 7, 7, None, (), {}, last)
    flatten = jax.tree_util.tree_flatten_with_path
    calls = 0

    def bounded_flatten(
        current: Any, is_leaf: Callable[[Any], bool] | None = None
    ) -> Any:
        nonlocal calls
        calls += 1
        assert calls <= 4 * (depth + 10)
        assert is_leaf is not None, "Shared DAG must use one-level public JAX traversal"
        result = flatten(current, is_leaf=is_leaf)
        assert len(result[0]) <= len(value)
        return result

    monkeypatch.setattr(jax.tree_util, "tree_flatten_with_path", bounded_flatten)
    recipe, inventory, arrays = structure.prepare_model_recipe_catalog(
        value, prefix="dag"
    )
    assert len(inventory) == 2
    assert inventory[0].tree_index == 1
    assert inventory[0].path == "[0]" + ".left" * depth + "[1]"
    assert inventory[1].tree_index == 4 * 2**depth + 2
    assert inventory[1].path == "[7]"
    assert arrays[inventory[0].name] is first
    assert arrays[inventory[1].name] is last
    assert structure.model_recipe_array_inventory(recipe, prefix="dag") == inventory
    prepared_recipe, prepared = structure.prepare_model_recipe(value, prefix="dag")
    assert _canonical(prepared_recipe) == _canonical(recipe)
    assert prepared[inventory[0].name] is first
    assert prepared[inventory[1].name] is last


def test_static_prng_key_uses_declared_path_after_runtime_arrays() -> None:
    key = jax.random.key(19)
    runtime = np.asarray((3,), dtype=np.int32)
    value = structure.restore_registered_fields(
        _StaticOwner, {"runtime": runtime, "static": key}
    )
    recipe, inventory, arrays = structure.prepare_model_recipe_catalog(
        value, prefix="static_key"
    )
    assert inventory == _legacy_inventory(recipe, "static_key")
    assert inventory[0].tree_index == 0
    assert inventory[1].tree_index == -1
    assert inventory[1].path == "model.items[1:'static']"
    assert inventory[1].implementation == "threefry2x32"
    np.testing.assert_array_equal(arrays[inventory[1].name], jax.random.key_data(key))


def test_live_catalog_keeps_archive_only_budgets_at_archive_boundary() -> None:
    value = (np.ones((1,), dtype=np.int32), np.zeros((1,), dtype=np.int32))
    limits = dataclasses.replace(
        DEFAULT_ARRAY_ARCHIVE_LIMITS, max_members=1, max_manifest_nesting=1
    )
    recipe, inventory, _ = structure.prepare_model_recipe_catalog(
        value, prefix="fields", limits=limits
    )
    assert inventory == structure.model_recipe_array_inventory(
        recipe, prefix="fields", limits=limits
    )


def test_external_recipe_values_still_reject_changed_alias_structure() -> None:
    shared = np.asarray((1,), dtype=np.int32)
    recipe = structure.model_structure_recipe((shared, shared))
    with pytest.raises(TypeError, match="model.structure"):
        structure.model_recipe_array_values(
            (shared, shared.copy()), recipe, prefix="external"
        )


def test_fresh_array_pairs_encode_each_side_once_and_preserve_alias_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = np.asarray((1.0, -0.0), dtype=np.float32)
    shared.setflags(write=False)
    second_left = np.asarray((2.0, 0.0), dtype=np.float32)
    second_right = np.asarray((3.0, 4.0), dtype=np.float32)
    first = (_Branch(shared, shared), shared)
    second = (_Branch(second_left, second_right), second_left)
    encode = structure._structure_recipe
    counts = [0, 0]

    def counted_encode(value: Any, path: str, retained: Any, /) -> dict[str, Any]:
        if value is first:
            counts[0] += 1
        if value is second:
            counts[1] += 1
        return encode(value, path, retained)

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Fresh paired recipes must reuse their retained shared objects")

    monkeypatch.setattr(structure, "_structure_recipe", counted_encode)
    monkeypatch.setattr(structure, "_table_objects", refuse)
    pairs = structure.model_recipe_array_pairs(first, second)
    assert pairs is not None
    assert len(pairs) == 2
    assert pairs[0][0] is shared
    assert pairs[0][1] is second_left
    assert pairs[1][0] is shared
    assert pairs[1][1] is second_right
    assert counts == [1, 1]


def test_fresh_array_pairs_keep_metadata_and_limit_admission() -> None:
    array = np.asarray((1, 2), dtype=np.int32)
    assert structure.model_recipe_array_pairs((array, "a"), (array, "b")) is None
    assert structure.model_recipe_array_pairs(array, array.astype(np.int16)) is None
    limits = dataclasses.replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_member_bytes=7)
    with pytest.raises(ValueError, match="array-byte"):
        structure.model_recipe_array_pairs(array, array, limits=limits)
