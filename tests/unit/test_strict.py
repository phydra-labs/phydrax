#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
from equinox._module._prebuilt import BoundMethod

from phydrax._strict import StrictModule


class AbstractValue(StrictModule):
    value: eqx.AbstractVar[int]

    @abstractmethod
    def evaluate(self) -> int:
        raise NotImplementedError


class Value(AbstractValue):
    value: int

    def evaluate(self) -> int:
        return self.value


def test_abstract_var_and_method_resolve_in_one_final_class() -> None:
    with pytest.raises(TypeError, match="abstract"):
        # ty: ignore[call-non-callable, missing-argument]
        AbstractValue()

    value = Value(3)
    assert value.evaluate() == 3
    with pytest.raises(AttributeError):
        value.value = 4
    with pytest.raises(TypeError, match=r"concrete \(final\)"):

        class InvalidChild(Value):
            pass


def test_abstract_property_can_be_implemented() -> None:
    class AbstractPropertyContract(StrictModule):
        @property
        @abstractmethod
        def label(self) -> str:
            raise NotImplementedError

    class PropertyImplementation(AbstractPropertyContract):
        @property
        def label(self) -> str:
            return "implemented"

    assert PropertyImplementation().label == "implemented"


def test_abstract_strict_bases_may_supply_overridable_defaults() -> None:
    class AbstractContract(StrictModule):
        @abstractmethod
        def evaluate(self) -> int:
            raise NotImplementedError

        def identity(self) -> str:
            return "contract"

    class Implementation(AbstractContract):
        def evaluate(self) -> int:
            return 1

        def identity(self) -> str:
            return "replacement"

    assert Implementation().identity() == "replacement"


class _CountedRow:
    """Host-side row whose PyTree flattens are counted."""

    flattens = 0


def _flatten_row(row: _CountedRow, /) -> tuple[tuple[()], None]:
    _CountedRow.flattens += 1
    return (), None


jtu.register_pytree_node(_CountedRow, _flatten_row, lambda _aux, _children: _CountedRow())


class _HostSource(StrictModule):
    rows: tuple[_CountedRow, ...]
    weights: jax.Array

    def row_count(self) -> int:
        return len(self.rows)

    def scaled(self, values: jax.Array, /) -> jax.Array:
        return self.weights * values


def _host_source(size: int, /) -> _HostSource:
    return _HostSource(tuple(_CountedRow() for _ in range(size)), jnp.arange(3.0))


@pytest.mark.parametrize("size", (1, 4096))
def test_method_access_never_flattens_the_receiver(size: int) -> None:
    source = _host_source(size)
    flattens = _CountedRow.flattens
    for _ in range(64):
        method = source.row_count
        assert method() == size
    assert _CountedRow.flattens == flattens
    assert type(method) is BoundMethod
    assert method.__self__ is source


def test_bound_methods_are_equinox_bound_method_pytrees() -> None:
    source = _host_source(2)
    method = source.scaled
    reference = BoundMethod(_HostSource.scaled, source)
    assert jtu.tree_structure(method) == jtu.tree_structure(reference)
    assert vars(method) == vars(reference)
    assert method.__name__ == "scaled"
    assert method == reference


def test_bound_method_passes_through_filter_jit_with_its_receiver() -> None:
    source = _host_source(2)
    flattens = _CountedRow.flattens
    values = jnp.full(3, 2.0)

    def apply(
        method: Callable[[jax.Array], jax.Array], values: jax.Array, /
    ) -> jax.Array:
        return method(values)

    np.testing.assert_array_equal(
        eqx.filter_jit(apply)(source.scaled, values), (0.0, 2.0, 4.0)
    )
    np.testing.assert_array_equal(eqx.filter_jit(source.scaled)(values), (0.0, 2.0, 4.0))
    # The receiver is a PyTree subnode of the bound method, so `jit` flattens it.
    assert _CountedRow.flattens > flattens
