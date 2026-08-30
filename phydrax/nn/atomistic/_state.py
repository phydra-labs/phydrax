#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import copy
from abc import abstractmethod
from typing import Any, cast, TypeVar

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule


_AtomisticPotentialT = TypeVar("_AtomisticPotentialT", bound="AbstractAtomisticPotential")


class AbstractAtomisticPotential(StrictModule):
    """Atomistic energy model with checkpointable parameter-state provenance."""

    architecture_id: AbstractAttribute[str]
    parameter_state_id: AbstractAttribute[str]
    potential_id: AbstractAttribute[str]

    @abstractmethod
    def parameter_state_tree(self, /) -> Any:
        """Return exactly the trainable arrays defining numerical predictions."""

        raise NotImplementedError


def _parameter_state_id(potential: AbstractAtomisticPotential, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "atomistic-potential-parameter-state",
            "arrays": array_tree_fingerprint(potential.parameter_state_tree()),
        }
    )


def checkpoint_atomistic_potential(
    potential: _AtomisticPotentialT, /
) -> _AtomisticPotentialT:
    """Return an immutable model copy with refreshed content-addressed provenance.

    External Equinox/Optax tree updates preserve static metadata and therefore do
    not constitute a provenance checkpoint. Call this operation after every such
    update and before prediction, persistence, or publication. Native atomistic
    training checkpoints its returned final and selected-best models automatically.
    """

    if not isinstance(potential, AbstractAtomisticPotential):
        raise TypeError("potential must implement AbstractAtomisticPotential.")
    state_id = _parameter_state_id(potential)
    potential_id = canonical_fingerprint(
        {
            "kind": "evaluated-atomistic-potential",
            "architecture": potential.architecture_id,
            "parameter_state": state_id,
        }
    )
    checkpoint = cast(_AtomisticPotentialT, copy.copy(potential))
    object.__setattr__(checkpoint, "parameter_state_id", state_id)
    object.__setattr__(checkpoint, "potential_id", potential_id)
    return checkpoint


def initialize_atomistic_potential_identity(
    potential: AbstractAtomisticPotential, /
) -> tuple[str, str]:
    """Compute constructor-time state and evaluated-potential identities."""

    state_id = _parameter_state_id(potential)
    potential_id = canonical_fingerprint(
        {
            "kind": "evaluated-atomistic-potential",
            "architecture": potential.architecture_id,
            "parameter_state": state_id,
        }
    )
    return state_id, potential_id


__all__ = ["AbstractAtomisticPotential", "checkpoint_atomistic_potential"]
