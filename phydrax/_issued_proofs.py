#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Process-local lookup of issued proofs that other owners still retain.

Entries are weakly keyed by an input owner object and weakly valued, so the
registry keeps neither alive: a proof is found only while some request or
artifact already owns it, and a fresh process always proves again. Callers
bind every other proof input into the string key and admit the reused proof's
storage on their own request ledger.
"""

from __future__ import annotations

import threading
import weakref
from typing import final


@final
class IssuedProofs[Owner, Proof]:
    """Weak ``(owner object, exact input key) -> proof`` lookup."""

    def __init__(self) -> None:
        # Weakref callbacks may run during collection on any thread, including
        # one that already holds this lock.
        self._lock = threading.RLock()
        self._entries: dict[
            int, tuple[weakref.ref[Owner], dict[str, weakref.ref[Proof]]]
        ] = {}

    def get(self, owner: Owner, key: str, /) -> Proof | None:
        with self._lock:
            entry = self._entries.get(id(owner))
            if entry is None or entry[0]() is not owner:
                return None
            reference = entry[1].get(key)
            return None if reference is None else reference()

    def record(self, owner: Owner, key: str, proof: Proof, /) -> None:
        identifier = id(owner)
        with self._lock:
            entry = self._entries.get(identifier)
            if entry is None or entry[0]() is not owner:
                # The callback captures only the integer identity, never the owner.
                entry = (
                    weakref.ref(
                        owner, lambda reference: self._forget(identifier, reference)
                    ),
                    {},
                )
                self._entries[identifier] = entry
            proofs = entry[1]
            for stale in [name for name, issued in proofs.items() if issued() is None]:
                del proofs[stale]
            proofs[key] = weakref.ref(proof)

    def _forget(self, identifier: int, reference: weakref.ref[Owner], /) -> None:
        with self._lock:
            entry = self._entries.get(identifier)
            if entry is not None and entry[0] is reference:
                del self._entries[identifier]
