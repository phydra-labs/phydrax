#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Hashable, Iterator, Mapping
from math import isfinite
from typing import Any, NoReturn, TypeVar

import equinox as eqx

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


class frozendict(StrictModule, Mapping[_KT, _VT]):
    _keys: tuple[_KT, ...] = eqx.field(static=True)
    _values: tuple[_VT, ...]

    def __init__(self, *args, **kwargs):
        supplied = dict(*args, **kwargs)
        mapping = {_canonical_key(key): value for key, value in supplied.items()}
        keys = tuple(sorted(mapping, key=_canonical_key_token))
        self._keys = keys
        self._values = tuple(mapping[key] for key in keys)

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
