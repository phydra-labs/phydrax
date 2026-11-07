#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Proof outcomes bound to the exact live operand objects that earned them.

A ledger records the outcome of one completed proof together with the object
identities of its operands and a content fingerprint of all of them taken
then. A later call on those same objects whose whole content still reproduces
that fingerprint may reuse the outcome. Anything else (a copy, a pytree
rebuild, an archive restore, a mutated array) has a different identity or
content and replays the proof in full. Entries live only in this process, die
with any operand, and are never serialized.
"""

from __future__ import annotations

import hashlib
import threading
import weakref
from typing import Any, Generic, TypeVar

import jax
import numpy as np

from ._fingerprint import iter_array_bytes


_T = TypeVar("_T")
_ARRAY_LEAVES = (np.ndarray, np.generic, jax.Array)


def live_content_fingerprint(value: object, /) -> tuple[object, str]:
    """Tree structure and a digest of every leaf, numeric arrays by content.

    Each leaf is framed by its kind and length: arrays by dtype, shape and
    C-order bytes (object arrays by element ``repr``), other leaves by type
    and ``repr``. The digest is compared only within this process.
    """
    leaves, treedef = jax.tree_util.tree_flatten(value)
    digest = hashlib.sha256()
    scalars: list[str] = []

    def flush() -> None:
        if scalars:
            text = "".join(scalars).encode("utf-8", "surrogatepass")
            digest.update(b"s" + len(text).to_bytes(8, "big"))
            digest.update(text)
            scalars.clear()

    for leaf in leaves:
        if not isinstance(leaf, _ARRAY_LEAVES):
            text = f"{type(leaf).__qualname__}:{leaf!r}"
            scalars.append(f"{len(text)}:{text}")
            continue
        flush()
        array = np.asarray(leaf)
        header = f"{array.dtype.str}{array.shape}".encode("ascii")
        digest.update(b"a" + len(header).to_bytes(8, "big") + header)
        if array.dtype.hasobject:
            text = repr(array.tolist()).encode("utf-8", "surrogatepass")
            digest.update(len(text).to_bytes(8, "big"))
            digest.update(text)
            continue
        digest.update(array.nbytes.to_bytes(8, "big"))
        for chunk in iter_array_bytes(array):
            digest.update(chunk)
    flush()
    return treedef, digest.hexdigest()


def _same_fingerprint(first: tuple[object, str], second: tuple[object, str], /) -> bool:
    if first[1] != second[1]:
        return False
    try:
        return bool(first[0] == second[0])
    except (TypeError, ValueError):
        # Static metadata without exact equality cannot prove itself unchanged.
        return False


def _parameter_key(parameters: tuple[object, ...], /) -> tuple[tuple[str, str], ...]:
    # Type-tagged exact text: 0.0 and -0.0, or 1 and 1.0, are different premises.
    return tuple((type(value).__qualname__, repr(value)) for value in parameters)


class LiveProofLedger(Generic[_T]):
    """Outcomes of completed proofs keyed by their exact live operands.

    ``None`` operands bind as absent; every other operand must support weak
    references, otherwise nothing is recorded and every call replays.
    ``parameters`` are scalar premises with exact ``repr`` (integers, floats,
    strings) that select among outcomes for the same operands.
    """

    __slots__ = ("_entries", "_lock", "__weakref__")

    def __init__(self) -> None:
        self._entries: dict[
            tuple[tuple[int, ...], tuple[tuple[str, str], ...]],
            tuple[tuple[weakref.ref[Any] | None, ...], tuple[object, str], _T],
        ] = {}
        self._lock = threading.Lock()

    def find(
        self,
        operands: tuple[object, ...],
        /,
        *,
        parameters: tuple[object, ...] = (),
    ) -> _T | None:
        """The recorded outcome for exactly these unchanged live operands."""
        key = (tuple(id(operand) for operand in operands), _parameter_key(parameters))
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return None
        references, fingerprint, outcome = entry
        if any(
            (operand is not None if reference is None else reference() is not operand)
            for reference, operand in zip(references, operands, strict=True)
        ):
            return None
        if not _same_fingerprint(live_content_fingerprint(operands), fingerprint):
            return None
        return outcome

    def record(
        self,
        operands: tuple[object, ...],
        fingerprint: tuple[object, str],
        outcome: _T,
        /,
        *,
        parameters: tuple[object, ...] = (),
    ) -> None:
        """Bind ``outcome`` to these operands and their pre-proof ``fingerprint``."""
        key = (tuple(id(operand) for operand in operands), _parameter_key(parameters))
        try:
            references = tuple(
                None if operand is None else weakref.ref(operand) for operand in operands
            )
        except TypeError:
            return
        with self._lock:
            self._entries[key] = (references, fingerprint, outcome)
        for operand in operands:
            if operand is not None:
                weakref.finalize(operand, self._forget, key)

    def _forget(
        self, key: tuple[tuple[int, ...], tuple[tuple[str, str], ...]], /
    ) -> None:
        with self._lock:
            self._entries.pop(key, None)
