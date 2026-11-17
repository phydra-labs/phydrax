#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bounded cyclic-collection scheduling for allocation-heavy host operations."""

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
    active: bool = True


_ENTRIES: list[_Entry] = []


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
            entry = _Entry(previous)
            _ENTRIES.append(entry)
            gc.set_threshold(max(previous[0], _YOUNG_THRESHOLD), *previous[1:])
        else:
            entry = None
    try:
        yield
    finally:
        if entry is not None:
            with _LOCK:
                entry.active = False
                while _ENTRIES and not _ENTRIES[-1].active:
                    previous = _ENTRIES.pop().thresholds
                    gc.set_threshold(*previous)
