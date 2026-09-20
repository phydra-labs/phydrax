#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Hashable, Iterator, Mapping
from typing import Any, NoReturn, TypeVar

import equinox as eqx

from ._strict import StrictModule


_KT = TypeVar("_KT", bound=Hashable)
_VT = TypeVar("_VT", covariant=True)


class frozendict(StrictModule, Mapping[_KT, _VT]):
    _keys: tuple[_KT, ...] = eqx.field(static=True)
    _values: tuple[_VT, ...]

    def __init__(self, *args, **kwargs):
        mapping = dict(*args, **kwargs)
        self._keys = tuple(mapping)
        self._values = tuple(mapping.values())

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
