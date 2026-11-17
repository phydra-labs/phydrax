#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bounded cyclic-collection scheduling for allocation-heavy host operations.

Scopes restore their entry thresholds only while the installed thresholds remain
unchanged. An observed external change is retained across overlapping scopes.
"""

import gc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock


_YOUNG_THRESHOLD = 50_000
_LOCK = RLock()


@dataclass(slots=True)
class _Entry:
    thresholds: tuple[int, int, int]
    installed: tuple[int, int, int]
    superseded: bool = False
    active: bool = True


_ENTRIES: list[_Entry] = []


def _observe_thresholds(current: tuple[int, int, int], /) -> None:
    if _ENTRIES and current != _ENTRIES[-1].installed:
        for entry in _ENTRIES:
            entry.superseded = True


@contextmanager
def collection_scope() -> Iterator[None]:
    """Raise the young threshold temporarily without disabling cyclic collection.

    The collector is process-wide. Overlapping scopes defer an out-of-order
    restoration until younger scopes have exited, so one thread cannot lower
    the threshold while another still owns a scope.
    """
    with _LOCK:
        if gc.isenabled():
            previous = gc.get_threshold()
            _observe_thresholds(previous)
            installed = (max(previous[0], _YOUNG_THRESHOLD), *previous[1:])
            entry = _Entry(previous, installed)
            _ENTRIES.append(entry)
            gc.set_threshold(*installed)
        else:
            entry = None
    try:
        yield
    finally:
        if entry is not None:
            with _LOCK:
                _observe_thresholds(gc.get_threshold())
                entry.active = False
                while _ENTRIES and not _ENTRIES[-1].active:
                    _observe_thresholds(gc.get_threshold())
                    completed = _ENTRIES.pop()
                    if not completed.superseded:
                        gc.set_threshold(*completed.thresholds)
