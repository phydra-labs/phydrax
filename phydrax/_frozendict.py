#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Hashable, ItemsView, Iterator, KeysView, Mapping, ValuesView
from typing import Any, NoReturn, TypeVar

from ._strict import StrictModule


_KT = TypeVar("_KT", bound=Hashable)
_VT = TypeVar("_VT", covariant=True)


class frozendict(StrictModule, Mapping[_KT, _VT]):
    _mapping: dict[_KT, _VT]

    def __init__(self, *args, **kwargs):
        self._mapping = dict(*args, **kwargs)  # ty: ignore[invalid-assignment]

    def __len__(self) -> int:
        return self._mapping.__len__()

    def __iter__(self) -> Iterator[_KT]:
        return iter(self._mapping.keys())

    def __getitem__(self, key: _KT, /) -> _VT:
        return self._mapping.__getitem__(key)

    def get(self, key: _KT, default: Any = None, /) -> Any:
        return self._mapping.get(key, default)

    def keys(self) -> KeysView[_KT]:
        return self._mapping.keys()

    def values(self) -> ValuesView[_VT]:
        return self._mapping.values()

    def items(self) -> ItemsView[_KT, _VT]:
        return self._mapping.items()

    def __contains__(self, key: Hashable) -> bool:
        return self._mapping.__contains__(key)

    def __hash__(self) -> int:
        items = self._mapping.items()
        return hash(frozenset(items))

    def __repr__(self) -> str:
        return f"frozendict({self._mapping!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, frozendict):
            return self._mapping == other._mapping
        if isinstance(other, Mapping):
            return self._mapping == dict(other)
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
