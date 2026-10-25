#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed nonconservative force ports of atomistic dynamics.

A force port supplies Cartesian forces that are *not* the gradient of any
scalar the program holds.  Dynamics evaluates every port at each conservative
force boundary with the same accepted geometry and adds its forces inside the
integrator's own kick operators, before constraint projection.  A port has no
energy and no strain derivative; ``derivative_contract`` is ``None`` unless a
port admits a derivative of its force map, and no energy Hessian is ever
claimed.  Its trajectory work is booked by dynamics.  Its state is a
read-only term state bound under the port's declared slot ``name``.
"""

from __future__ import annotations

import abc

import equinox as eqx
from jax import Array
from jax.typing import DTypeLike

from .._differentiation import DerivativeContract
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._term_state import AbstractAtomisticTermState


class AtomisticForcePortGeometry(StrictModule):
    """Accepted geometry at one force boundary (no strain chart)."""

    positions: Array
    unwrapped_positions: Array
    cell_vectors: Array | None
    image_counts: Array | None
    species: Array


class AtomisticForcePortEvaluation(StrictModule):
    """Port forces ``(capacity, 3)``, success, status word and typed evidence."""

    forces: Array
    successful: Array
    status: Array
    evidence: tuple[StrictModule, ...] = ()


class AbstractAtomisticForcePort(StrictModule, NonTrainableState):
    """One nonconservative force contribution of a prepared dynamics runtime.

    ``name`` is the declared static slot of the port's read-only state in the
    runtime's term-state bindings (distinct from every stateful term name).
    ``owner_identity`` is the exact canonical digest (dynamic words) its bound
    state carries as ``owner_identity``; compiled steps validate the binding
    with ``bound_to(name, owner_identity)``.  ``system_identity`` binds the
    prepared system whose capacity the forces cover (host ``system_id``).
    ``port_id`` is the host scientific identity string of the port.
    """

    name: eqx.AbstractVar[str]
    owner_identity: eqx.AbstractVar[IdentityWords]
    system_identity: eqx.AbstractVar[IdentityWords]
    port_id: eqx.AbstractVar[str]

    @property
    def system_id(self) -> str:
        """Host decoding of the bound prepared-system digest."""
        return self.system_identity.fingerprint()

    @property
    def derivative_contract(self) -> DerivativeContract | None:
        """Admitted derivatives of the force map; ``None`` admits none."""
        return None

    @abc.abstractmethod
    def initial_term_state(self, dtype: DTypeLike, /) -> AbstractAtomisticTermState:
        raise NotImplementedError

    @abc.abstractmethod
    def forces(
        self,
        geometry: AtomisticForcePortGeometry,
        state: AbstractAtomisticTermState,
        /,
    ) -> AtomisticForcePortEvaluation:
        raise NotImplementedError


__all__ = [
    "AbstractAtomisticForcePort",
    "AtomisticForcePortEvaluation",
    "AtomisticForcePortGeometry",
]
