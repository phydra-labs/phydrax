from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import pytest

from phydrax._frozendict import frozendict


def test_frozendict_scenario_1() -> None:
    expected = {"a": 1, "b": 2}
    constructions = (
        frozendict(expected),
        frozendict(a=1, b=2),
        frozendict([("a", 1), ("b", 2)]),
        frozendict(frozendict(a=1, b=2)),
    )
    for value in constructions:
        assert isinstance(value, Mapping)
        assert dict(value) == expected
        assert {**value} == expected
        assert value.get("a") == 1
        assert value.get("missing") is None
        assert value.get("missing", 3) == 3
        assert set(value.keys()) == {"a", "b"}
        assert set(value.values()) == {1, 2}
        assert set(value.items()) == {("a", 1), ("b", 2)}
        assert len(value) == 2
        assert "a" in value and "missing" not in value
        assert set(iter(value)) == {"a", "b"}

    tuple_keys = frozendict({(1, 2): "a", (3, 4): "b"})
    assert tuple_keys[(1, 2)] == "a"
    assert tuple_keys[(3, 4)] == "b"

    empty = frozendict()
    assert not empty
    assert list(empty.keys()) == list(empty.values()) == list(empty.items()) == []
    assert hash(empty) == hash(frozenset())
    value = frozendict(a=1, b=2)
    with pytest.raises(TypeError):
        value["a"] = 3
    with pytest.raises(TypeError):
        del value["a"]
    with pytest.raises(TypeError):
        value.clear()
    with pytest.raises(TypeError):
        value.pop("a")
    with pytest.raises(TypeError):
        value.popitem()
    with pytest.raises(TypeError):
        value.setdefault("c", 3)
    with pytest.raises(TypeError):
        value.update({"c": 3})
    first = frozendict(a=1, b=2)
    equal = frozendict(a=1, b=2)
    different = frozendict(a=1, b=3)
    assert first == equal == {"a": 1, "b": 2}
    assert first != different
    assert first != [("a", 1), ("b", 2)]
    assert first != 42
    assert hash(first) == hash(equal)
    assert hash(first) != hash(different)
    keyed = {first: "first", different: "different"}
    assert keyed[equal] == "first"
    assert keyed[different] == "different"

    nested = frozendict(a=1, b=frozendict(c=2, d=3))
    assert nested["b"] == frozendict(c=2, d=3)
    with pytest.raises(TypeError):
        nested["b"]["c"] = 4

    unhashable = frozendict(a=[1, 2, 3], b=[4, 5, 6])
    assert unhashable["a"] == [1, 2, 3]
    with pytest.raises(TypeError):
        hash(unhashable)


def test_frozendict_scenario_2() -> None:
    left = frozendict({"b": 2, "a": 1})
    right = frozendict({"a": 1, "b": 2})
    assert tuple(left) == ("a", "b")
    assert left == right
    assert jax.tree.structure(left) == jax.tree.structure(right)
    assert jax.tree.leaves(left) == jax.tree.leaves(right)

    boolean_key = frozendict({True: "value"})
    integer_key = frozendict({1: "value"})
    floating_key = frozendict({1.0: "value"})
    assert tuple(boolean_key) == tuple(integer_key) == tuple(floating_key) == (1,)
    assert jax.tree.structure(boolean_key) == jax.tree.structure(integer_key)
    assert jax.tree.structure(integer_key) == jax.tree.structure(floating_key)

    value = frozendict({"direction": 1.0, 3: 2.0, "offset": 3.0})
    paths = [
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(value)[0]
    ]
    assert [path.rsplit("[", 1)[-1] for path in paths] == [
        "3]",
        "'direction']",
        "'offset']",
    ]
    mapped = jax.tree.map(lambda item: 2.0 * item, value)
    assert jax.tree.structure(mapped) == jax.tree.structure(value)
    assert mapped == frozendict({"direction": 2.0, 3: 4.0, "offset": 6.0})

    class CustomKey:
        pass

    with pytest.raises(TypeError, match="canonical"):
        frozendict({CustomKey(): 1})
    with pytest.raises(ValueError, match="finite"):
        frozendict({float("nan"): 1})
    tree = {"metadata": frozendict(), "value": 1.0}
    updated = eqx.tree_at(lambda item: item["value"], tree, 2.0)
    assert jax.tree.structure(updated) == jax.tree.structure(tree)
    assert eqx.tree_equal(updated, {"metadata": frozendict(), "value": 2.0})


def test_generic_annotations_preserve_mapping_and_value_covariance() -> None:
    def total(mapping: Mapping[str, int]) -> int:
        return sum(mapping.values())

    value: frozendict[str, int] = frozendict(a=1, b=2)
    assert total(value) == 3

    class Animal:
        pass

    class Dog(Animal):
        pass

    animals: frozendict[str, Animal] = frozendict(pet=Dog())
    assert isinstance(animals["pet"], Animal)
