#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Hashable, Iterable, Iterator, Mapping
from math import isfinite
from typing import Any, NoReturn, TypeVar

import equinox as eqx
import jax

from ._strict import StrictModule


_KT = TypeVar("_KT", bound=Hashable)
_VT = TypeVar("_VT", covariant=True)


def _canonical_key(key: Hashable, /) -> Hashable:
    if key is None or isinstance(key, (str, bytes)):
        return key
    if isinstance(key, bool):
        return int(key)
    if isinstance(key, int):
        return key
    if isinstance(key, float):
        if not isfinite(key):
            raise ValueError("frozendict floating keys must be finite")
        return int(key) if key.is_integer() else key
    if isinstance(key, tuple):
        return tuple(_canonical_key(item) for item in key)
    if isinstance(key, frozenset):
        return frozenset(_canonical_key(item) for item in key)
    raise TypeError(
        "frozendict keys must have a canonical primitive, tuple, or frozenset identity"
    )


def _canonical_key_token(key: Hashable, /) -> tuple[Any, ...]:
    if key is None:
        return ("none",)
    if isinstance(key, bool):
        return ("bool", int(key))
    if isinstance(key, int):
        return ("int", key)
    if isinstance(key, float):
        if not isfinite(key):
            raise ValueError("frozendict floating keys must be finite")
        return ("float", key.hex())
    if isinstance(key, str):
        return ("str", key)
    if isinstance(key, bytes):
        return ("bytes", key.hex())
    if isinstance(key, tuple):
        return ("tuple", tuple(_canonical_key_token(item) for item in key))
    if isinstance(key, frozenset):
        return (
            "frozenset",
            tuple(sorted(_canonical_key_token(item) for item in key)),
        )
    raise TypeError(
        "frozendict keys must have a canonical primitive, tuple, or frozenset identity"
    )


class _KeyedValues(tuple):
    """Mapping values in key order whose pytree key paths are the mapping keys.

    A plain tuple to every structural consumer; only its pytree key paths differ.
    """

    mapping_keys: tuple[Any, ...]

    def __new__(cls, keys: tuple[Any, ...], values: Iterable[Any], /):
        instance = super().__new__(cls, values)
        instance.mapping_keys = keys
        return instance

    def __reduce__(self):
        return (type(self), (self.mapping_keys, tuple(self)))


jax.tree_util.register_pytree_with_keys(
    _KeyedValues,
    lambda values: (
        tuple(
            (jax.tree_util.DictKey(key), value)
            for key, value in zip(values.mapping_keys, values, strict=True)
        ),
        values.mapping_keys,
    ),
    lambda keys, values: _KeyedValues(keys, values),
    flatten_func=lambda values: (tuple(values), values.mapping_keys),
)


class frozendict(StrictModule, Mapping[_KT, _VT]):
    _keys: tuple[_KT, ...] = eqx.field(static=True)
    # Leaves flatten under `._values[key]`, so key paths name the mapping keys. An
    # empty mapping holds a plain `()`: there is no key to name, and `eqx.tree_at`
    # rebuilds every empty non-namedtuple tuple as a plain `()`, so an empty
    # `_KeyedValues` would not survive a structural update.
    _values: tuple[_VT, ...]

    def __init__(self, *args, **kwargs):
        supplied = dict(*args, **kwargs)
        mapping = {_canonical_key(key): value for key, value in supplied.items()}
        keys = tuple(sorted(mapping, key=_canonical_key_token))
        self._keys = keys
        self._values = _KeyedValues(keys, (mapping[key] for key in keys)) if keys else ()

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[_KT]:
        return iter(self._keys)

    def __getitem__(self, key: _KT, /) -> _VT:
        for candidate, value in zip(self._keys, self._values, strict=True):
            if candidate == key:
                return value
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def __hash__(self) -> int:
        return hash(frozenset(zip(self._keys, self._values, strict=True)))

    def __repr__(self) -> str:
        return f"frozendict({dict(zip(self._keys, self._values, strict=True))!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, frozendict):
            return dict(zip(self._keys, self._values, strict=True)) == dict(
                zip(other._keys, other._values, strict=True)
            )
        if isinstance(other, Mapping):
            return dict(zip(self._keys, self._values, strict=True)) == dict(other)
        return False

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise TypeError("frozendict is immutable")

    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __setitem__ = _immutable
    __delitem__ = _immutable
