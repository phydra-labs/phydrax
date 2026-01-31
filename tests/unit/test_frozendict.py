#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Mapping

import pytest

from phydrax._frozendict import frozendict


class TestFrozenDict:
    def test_init(self):
        # Test initialization with dict
        d = {"a": 1, "b": 2}
        fd = frozendict(d)
        assert fd["a"] == 1
        assert fd["b"] == 2

        # Test initialization with keyword arguments
        fd = frozendict(a=1, b=2)
        assert fd["a"] == 1
        assert fd["b"] == 2

        # Test initialization with items
        fd = frozendict([("a", 1), ("b", 2)])
        assert fd["a"] == 1
        assert fd["b"] == 2

        # Test initialization with another frozendict
        fd1 = frozendict(a=1, b=2)
        fd2 = frozendict(fd1)
        assert fd2["a"] == 1
        assert fd2["b"] == 2

    def test_immutability(self):
        fd = frozendict(a=1, b=2)

        # Test that assignment raises TypeError
        with pytest.raises(TypeError):
            fd["a"] = 3

        # Test that deletion raises TypeError
        with pytest.raises(TypeError):
            del fd["a"]

        # Test that clear raises TypeError
        with pytest.raises(TypeError):
            fd.clear()

        # Test that pop raises TypeError
        with pytest.raises(TypeError):
            fd.pop("a")

        # Test that popitem raises TypeError
        with pytest.raises(TypeError):
            fd.popitem()

        # Test that setdefault raises TypeError
        with pytest.raises(TypeError):
            fd.setdefault("c", 3)

        # Test that update raises TypeError
        with pytest.raises(TypeError):
            fd.update({"c": 3})

    def test_get(self):
        fd = frozendict(a=1, b=2)
        assert fd.get("a") == 1
        assert fd.get("c") is None
        assert fd.get("c", 3) == 3

    def test_keys_values_items(self):
        fd = frozendict(a=1, b=2)
        assert set(fd.keys()) == {"a", "b"}
        assert set(fd.values()) == {1, 2}
        assert set(fd.items()) == {("a", 1), ("b", 2)}

    def test_len(self):
        fd = frozendict(a=1, b=2)
        assert len(fd) == 2

    def test_contains(self):
        fd = frozendict(a=1, b=2)
        assert "a" in fd
        assert "c" not in fd

    def test_iter(self):
        fd = frozendict(a=1, b=2)
        keys = set()
        for key in fd:
            keys.add(key)
        assert keys == {"a", "b"}

    def test_hash(self):
        fd1 = frozendict(a=1, b=2)
        fd2 = frozendict(a=1, b=2)
        fd3 = frozendict(a=1, b=3)

        # Same content should have same hash
        assert hash(fd1) == hash(fd2)

        # Different content should have different hash
        assert hash(fd1) != hash(fd3)

        # Can be used as dict key
        d = {fd1: "value1", fd3: "value2"}
        assert d[fd1] == "value1"
        assert d[fd2] == "value1"  # fd2 has same hash as fd1
        assert d[fd3] == "value2"

    def test_equality(self):
        fd1 = frozendict(a=1, b=2)
        fd2 = frozendict(a=1, b=2)
        fd3 = frozendict(a=1, b=3)
        d = {"a": 1, "b": 2}

        # Same content should be equal
        assert fd1 == fd2

        # Different content should not be equal
        assert fd1 != fd3

        # Should be equal to a regular dict with same content
        assert fd1 == d

        # Should not be equal to non-mapping types
        assert fd1 != [("a", 1), ("b", 2)]
        assert fd1 != 42

    def test_nested_frozendict(self):
        # Test with nested frozendict
        nested = frozendict(a=1, b=frozendict(c=2, d=3))
        assert nested["b"]["c"] == 2
        assert nested["b"]["d"] == 3

        # Ensure nested frozendict is also immutable
        with pytest.raises(TypeError):
            nested["b"]["c"] = 4

    def test_with_unhashable_values(self):
        # Should work with unhashable values like lists
        fd = frozendict(a=[1, 2, 3], b=[4, 5, 6])
        assert fd["a"] == [1, 2, 3]

        # But can't be hashed if values are unhashable
        with pytest.raises(TypeError):
            hash(fd)

    def test_with_complex_keys(self):
        # Test with tuple keys (which are hashable)
        fd = frozendict({(1, 2): "a", (3, 4): "b"})
        assert fd[(1, 2)] == "a"
        assert fd[(3, 4)] == "b"

    def test_mapping_protocol(self):
        # Test that frozendict implements the Mapping protocol
        fd = frozendict(a=1, b=2)
        assert isinstance(fd, Mapping)

        # Test dict methods that should work with any Mapping
        assert dict(fd) == {"a": 1, "b": 2}
        assert {**fd} == {"a": 1, "b": 2}

    def test_empty_frozendict(self):
        fd = frozendict()
        assert len(fd) == 0
        assert list(fd.keys()) == []
        assert list(fd.values()) == []
        assert list(fd.items()) == []
        assert hash(fd) == hash(frozenset())

    def test_type_annotations(self):
        # Test that type annotations work as expected
        def takes_mapping(m: Mapping[str, int]) -> int:
            return sum(m.values())

        fd: frozendict[str, int] = frozendict(a=1, b=2)
        assert takes_mapping(fd) == 3

        # Test covariance of value type
        class Animal:
            pass

        class Dog(Animal):
            pass

        fd_animals: frozendict[str, Animal] = frozendict(pet=Dog())
        assert isinstance(fd_animals["pet"], Animal)
