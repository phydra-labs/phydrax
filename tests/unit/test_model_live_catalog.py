"""Live catalogs preserve legacy names and indices without expanding aliases."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from typing import Any, final, Literal, NoReturn

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


def test_fresh_array_pairs_preserve_alias_pairs() -> None:
    shared = np.asarray((1.0, -0.0), dtype=np.float32)
    shared.setflags(write=False)
    second_left = np.asarray((2.0, 0.0), dtype=np.float32)
    second_right = np.asarray((3.0, 4.0), dtype=np.float32)
    first = (_Branch(shared, shared), shared)
    second = (_Branch(second_left, second_right), second_left)
    pairs = structure.model_recipe_array_pairs(first, second)
    assert pairs is not None
    assert len(pairs) == 2
    assert pairs[0][0] is shared
    assert pairs[0][1] is second_left
    assert pairs[1][0] is shared
    assert pairs[1][1] is second_right


def test_fresh_array_pairs_keep_metadata_and_limit_admission() -> None:
    array = np.asarray((1, 2), dtype=np.int32)
    assert structure.model_recipe_array_pairs((array, "a"), (array, "b")) is None
    assert structure.model_recipe_array_pairs(array, array.astype(np.int16)) is None
    limits = dataclasses.replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_member_bytes=7)
    with pytest.raises(ValueError, match="array-byte"):
        structure.model_recipe_array_pairs(array, array, limits=limits)


def test_needed_content_forms_equal_eager_forms_for_every_admitted_node() -> None:
    from fractions import Fraction

    array = np.asarray((1.0, -0.0), dtype=np.float64)
    values = (
        _Branch({"b": array, "a": array}, (Fraction(1, 3), complex(2, -1))),
        (frozenset((("x", 2), ("y", 3))), slice(1, 4, 2), np.dtype("float64")),
        {"nested": [None, True, "λ", float("inf"), float("nan")]},
    )
    for value in values:
        table = structure._lower_recipe(
            structure.model_structure_recipe(value), DEFAULT_ARRAY_ARCHIVE_LIMITS
        )
        eager = structure._content_forms(table.nodes)
        needed = structure._needed_content_forms(table.nodes)
        for index in reversed(range(len(table.nodes))):
            assert needed(index) == eager[index]


def test_lazy_pairs_preserve_eager_schema_alias_key_set_and_literal_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractions import Fraction

    first = np.asarray((1.0, -0.0), dtype=np.float64)
    second = first.copy()
    first.setflags(write=False)
    cases = (
        ((_Branch(first, first), first), (_Branch(second, second.copy()), second)),
        ({"b": first, "a": 3}, {"a": 3, "b": second}),
        ({"a": first}, {"b": second}),
        ((frozenset((Fraction(1, 2), Fraction(2, 3))), first),
         (frozenset((Fraction(2, 3), Fraction(1, 2))), second)),
        ((frozenset(("x", "y")), first), (frozenset(("x", "z")), second)),
        ((float("inf"), float("nan"), first), (float("inf"), float("nan"), second)),
        ((first, "a"), (second, "b")),
        (first, second.astype(np.float32)),
    )

    def eager_lookup(nodes: list[structure.RecipeNode]) -> Callable[[int], bytes]:
        return structure._content_forms(nodes).__getitem__

    for left, right in cases:
        with monkeypatch.context() as patch:
            patch.setattr(structure, "_needed_content_forms", eager_lookup)
            expected = structure.model_recipe_array_pairs(left, right)
        actual = structure.model_recipe_array_pairs(left, right)
        assert (actual is None) == (expected is None)
        if actual is not None and expected is not None:
            assert [(id(a), id(b)) for a, b in actual] == [(id(a), id(b)) for a, b in expected]


def test_pairing_never_computes_unused_composite_content_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = structure._needed_content_forms
    requested = []

    def observe(nodes: list[structure.RecipeNode]) -> Callable[[int], bytes]:
        lookup = original(nodes)

        def read(index: int) -> bytes:
            requested.append(nodes[index]["kind"])
            return lookup(index)

        return read

    monkeypatch.setattr(structure, "_needed_content_forms", observe)
    first = _Branch({"a": np.arange(4, dtype=np.int32)}, (None, "same"))
    second = _Branch({"a": np.arange(4, dtype=np.int32)}, (None, "same"))
    assert structure.model_recipe_array_pairs(first, second) is not None
    assert requested
    assert not set(requested) & {"dataclass", "tuple", "list", "array", "prng_key", "mapping"}


@pytest.mark.parametrize("defect", ("unknown-kind", "literal", "fraction", "duplicate-key", "reference", "array", "type"))
def test_pairing_eagerly_admits_even_a_mismatched_unused_bad_subtree(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    import copy

    left, right = object(), object()
    array = structure.model_structure_recipe(np.arange(2, dtype=np.int32))
    if defect == "unknown-kind":
        invalid = {"kind": "unsupported"}
    elif defect == "literal":
        invalid = {"kind": "literal", "value": float("nan")}
    elif defect == "fraction":
        invalid = {"kind": "fraction", "numerator": 1, "denominator": 0}
    elif defect == "duplicate-key":
        invalid = {"kind": "mapping", "items": [
            [{"kind": "literal", "value": "same"}, {"kind": "literal", "value": 1}],
            [{"kind": "literal", "value": "same"}, {"kind": "literal", "value": 2}],
        ]}
    elif defect == "reference":
        invalid = {"kind": "reference", "target": 10_000}
    elif defect == "array":
        invalid = copy.deepcopy(array)
        invalid["dtype"] = "|O"
    else:
        invalid = {"kind": "type", "value": "test:missing-canonical-type"}
    recipe = {"kind": "tuple", "items": [{"kind": "literal", "value": 0}, invalid]}
    original = structure._structure_recipe

    def encode(
        value: Any, path: str, shared: structure._SharedObjects | None
    ) -> dict[str, Any]:
        if value is left:
            return {"kind": "literal", "value": 0}
        if value is right:
            return copy.deepcopy(recipe)
        return original(value, path, shared)

    def no_form_computation(nodes: list[structure.RecipeNode]) -> Callable[[int], bytes]:
        def refuse(_index: object) -> NoReturn:
            pytest.fail("both full lowerings must finish before any lazy form is requested")
        return refuse

    monkeypatch.setattr(structure, "_structure_recipe", encode)
    with monkeypatch.context() as patch:
        patch.setattr(structure, "_needed_content_forms",
                      lambda nodes: structure._content_forms(nodes).__getitem__)
        with pytest.raises((TypeError, ValueError, KeyError)) as expected:
            structure.model_recipe_array_pairs(left, right)
    monkeypatch.setattr(structure, "_needed_content_forms", no_form_computation)
    with pytest.raises(type(expected.value)) as actual:
        structure.model_recipe_array_pairs(left, right)
    assert str(actual.value) == str(expected.value)


def test_pairing_preserves_cycle_and_size_admission_before_form_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycle = []
    cycle.append(cycle)
    with monkeypatch.context() as patch:
        patch.setattr(structure, "_needed_content_forms",
                      lambda nodes: structure._content_forms(nodes).__getitem__)
        with pytest.raises((TypeError, ValueError, RecursionError)) as expected:
            structure.model_recipe_array_pairs(0, cycle)
    with pytest.raises(type(expected.value)) as actual:
        structure.model_recipe_array_pairs(0, cycle)
    assert str(actual.value) == str(expected.value)
    array = np.arange(4, dtype=np.int32)
    limits = dataclasses.replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_member_bytes=15)
    with pytest.raises(ValueError, match="array-byte"):
        structure.model_recipe_array_pairs(0, array, limits=limits)
