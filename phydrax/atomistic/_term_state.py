#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import final

import equinox as eqx
import jax.numpy as jnp

from .._identity import IdentityWords
from .._model import register_artifact_value
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..typing import Bool, Dim, Identifiers, Int32, parse, Scalar


class AtomisticTermStateDim(Dim):
    pass


class AbstractAtomisticTermState(StrictModule, NonTrainableState):
    """Read-only numerical state supplied to one declared scientific term.

    ``owner_identity`` is the exact scientific identity (canonical digest words)
    of the term or force port that owns the state.  It is dynamic, so states of
    different material/parameter revisions share one executable; every consumer
    compares it numerically with its own identity and refuses a foreign or stale
    state through its status.  Owners advance revisions only when accepted
    history or controls change.
    """

    revision: eqx.AbstractVar[Int32[Scalar]]
    successful: eqx.AbstractVar[Bool[Scalar]]
    owner_identity: eqx.AbstractVar[IdentityWords]


@final
class AtomisticTermStateBindings(StrictModule, NonTrainableState):
    """Explicitly named state slots with dynamic owner-identity validation.

    ``slots`` are the declared program-slot names (prepared term names, or the
    declared slot of a force port): layout configuration, sorted canonically,
    never incidental tuple order and never content-derived identities.  The
    scientific owner of each state is its dynamic ``owner_identity``.
    """

    __strict_contract__ = True

    slots: Identifiers[AtomisticTermStateDim] = eqx.field(static=True)
    values: tuple[AbstractAtomisticTermState, ...]

    def __init__(self, states: Mapping[str, AbstractAtomisticTermState], /) -> None:
        if not isinstance(states, Mapping):
            raise TypeError("Term states must be a mapping keyed by declared slot name.")
        admitted: list[tuple[str, AbstractAtomisticTermState]] = []
        for key, state in states.items():
            slot = canonical_identifier(key, "slot")
            if not isinstance(state, AbstractAtomisticTermState):
                raise TypeError("Term states must implement AbstractAtomisticTermState.")
            parse(state.revision, Int32[Scalar], f"{slot}.revision")
            parse(state.successful, Bool[Scalar], f"{slot}.successful")
            if not isinstance(state.owner_identity, IdentityWords):
                raise TypeError(f"{slot}.owner_identity must be IdentityWords.")
            admitted.append((slot, state))
        ordered = tuple(sorted(admitted, key=lambda item: item[0]))
        self.slots = parse(
            tuple(key for key, _ in ordered),
            Identifiers[AtomisticTermStateDim],
            "slots",
        )
        self.values = tuple(state for _, state in ordered)

    def __getitem__(self, slot: str, /) -> AbstractAtomisticTermState:
        identifier = canonical_identifier(slot, "slot")
        if identifier not in self.slots:
            raise ValueError("No state is bound to this declared slot.")
        return self.values[self.slots.index(identifier)]

    def subset(self, slots: tuple[str, ...], /) -> AtomisticTermStateBindings:
        """Select an explicitly named scalar-program or force-port domain."""
        identifiers = parse(slots, Identifiers[AtomisticTermStateDim], "slots")
        return AtomisticTermStateBindings({key: self[key] for key in identifiers})

    def bound_to(self, slot: str, owner: IdentityWords, /) -> Bool[Scalar]:
        """Traced: whether ``slot`` holds a successful state of exactly ``owner``."""
        state = self[slot]
        return (
            state.successful & (state.revision >= 0) & state.owner_identity.equals(owner)
        )

    @property
    def revisions(self) -> Int32[AtomisticTermStateDim]:
        if not self.values:
            return jnp.zeros((0,), dtype=jnp.int32)
        return jnp.stack(tuple(state.revision for state in self.values))


register_artifact_value(
    "phydrax.atomistic:AtomisticTermStateBindings", AtomisticTermStateBindings
)


__all__ = [
    "AbstractAtomisticTermState",
    "AtomisticTermStateBindings",
    "AtomisticTermStateDim",
]
