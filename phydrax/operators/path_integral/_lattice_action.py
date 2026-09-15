#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._sampling import FullMarkovTarget, IncrementalMarkovTarget
from ..._strict import AbstractAttribute, StrictModule
from ...metrix import AbstractStateGeometry
from ...sampling._compact_group_hamiltonian import CompactGeometricTarget


LatticeReferenceMeasure: TypeAlias = Literal[
    "lebesgue",
    "flat-torus",
    "product-haar",
]


class LatticeActionEvidence(StrictModule):
    """Static measure claim carried by one finite lattice action."""

    reference_measure: LatticeReferenceMeasure = eqx.field(static=True)
    real_valued: bool = eqx.field(static=True)
    bounded_below: bool = eqx.field(static=True)
    normalizable: bool = eqx.field(static=True)
    additive_constant: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        reference_measure: LatticeReferenceMeasure,
        real_valued: bool,
        bounded_below: bool,
        normalizable: bool,
        additive_constant: float,
        evidence_id: str,
    ):
        if reference_measure not in ("lebesgue", "flat-torus", "product-haar"):
            raise ValueError("Unknown lattice reference measure.")
        constant = float(additive_constant)
        if not jnp.isfinite(constant):
            raise ValueError("additive_constant must be finite.")
        identifier = str(evidence_id)
        if not identifier:
            raise ValueError("evidence_id must be non-empty.")
        self.reference_measure = reference_measure
        self.real_valued = bool(real_valued)
        self.bounded_below = bool(bounded_below)
        self.normalizable = bool(normalizable)
        self.additive_constant = constant
        self.evidence_id = identifier


class AbstractLatticeEuclideanAction(StrictModule):
    """Real finite lattice action with an explicit coordinate reference measure."""

    topology_id: AbstractAttribute[str]
    field_space_id: AbstractAttribute[str]
    configuration_shape: AbstractAttribute[tuple[int, ...]]
    local_coordinate_shape: AbstractAttribute[tuple[int, ...]]
    geometry: AbstractAttribute[AbstractStateGeometry]
    evidence: AbstractAttribute[LatticeActionEvidence]
    action_id: AbstractAttribute[str]

    @abstractmethod
    def action(self, configuration: PyTree[Any], /) -> Array:
        """Return the reduced Euclidean action as one real scalar."""
        raise NotImplementedError


class AbstractIncrementalLatticeAction(AbstractLatticeEuclideanAction):
    """Lattice action with exact fixed-shape local cache transitions."""

    @abstractmethod
    def initialize_incremental(
        self, configuration: PyTree[Any], /
    ) -> tuple[Array, PyTree[Array]]:
        raise NotImplementedError

    @abstractmethod
    def propose_incremental(
        self,
        current_configuration: PyTree[Any],
        current_cache: PyTree[Any],
        proposed_configuration: PyTree[Any],
        payload: PyTree[Any],
        /,
    ) -> tuple[Array, PyTree[Array], Array]:
        """Return ``delta action``, the candidate cache, and validity."""
        raise NotImplementedError

    @abstractmethod
    def select_incremental(
        self,
        current: PyTree[Any],
        proposed: PyTree[Any],
        accepted: Array,
        /,
    ) -> PyTree[Array]:
        raise NotImplementedError

    @abstractmethod
    def refresh_incremental(
        self, configuration: PyTree[Any], /
    ) -> tuple[Array, PyTree[Array]]:
        raise NotImplementedError


def _validate_action(action: AbstractLatticeEuclideanAction, /) -> None:
    if not isinstance(action, AbstractLatticeEuclideanAction):
        raise TypeError("action must implement AbstractLatticeEuclideanAction.")
    if not action.evidence.real_valued:
        raise ValueError("Positive-weight Markov targets require a real action.")
    if not action.evidence.normalizable:
        raise ValueError("The lattice action does not carry a normalizability claim.")


def _negative_lattice_action(
    action: AbstractLatticeEuclideanAction,
    configuration: Array,
    /,
) -> Array:
    return -action.action(configuration)


def _initialize_incremental_action(
    action: AbstractIncrementalLatticeAction,
    configuration: PyTree[Any],
    /,
):
    value, cache = action.initialize_incremental(configuration)
    return -jnp.asarray(value), cache


def _propose_incremental_action(
    action: AbstractIncrementalLatticeAction,
    current: PyTree[Any],
    cache: PyTree[Any],
    proposed: PyTree[Any],
    payload: PyTree[Any],
    /,
):
    delta, proposed_cache, valid = action.propose_incremental(
        current, cache, proposed, payload
    )
    return -jnp.asarray(delta), proposed_cache, valid


def _select_incremental_action(
    action: AbstractIncrementalLatticeAction,
    current: PyTree[Any],
    proposed: PyTree[Any],
    accepted: Array,
    /,
):
    return action.select_incremental(current, proposed, accepted)


def _refresh_incremental_action(
    action: AbstractIncrementalLatticeAction,
    configuration: PyTree[Any],
    /,
):
    value, cache = action.refresh_incremental(configuration)
    return -jnp.asarray(value), cache


def compact_geometric_target_from_lattice_action(
    action: AbstractLatticeEuclideanAction,
    /,
) -> CompactGeometricTarget:
    """Lower one compact normalizable lattice action to a geometric log target."""
    _validate_action(action)
    if action.evidence.reference_measure not in ("flat-torus", "product-haar"):
        raise ValueError(
            "Compact geometric targets require flat-torus or product-Haar measure."
        )
    return CompactGeometricTarget(
        eqx.Partial(_negative_lattice_action, action),
        action.geometry,
        configuration_shape=action.configuration_shape,
        local_coordinate_shape=action.local_coordinate_shape,
        reference_measure=action.evidence.reference_measure,
        target_id=f"{action.action_id}:boltzmann",
    )


def full_target_from_lattice_action(
    action: AbstractLatticeEuclideanAction,
    /,
) -> FullMarkovTarget:
    """Lower a normalizable real action to a complete Markov log target."""
    _validate_action(action)
    return FullMarkovTarget(
        eqx.Partial(_negative_lattice_action, action),
        target_id=f"{action.action_id}:boltzmann",
    )


def incremental_target_from_lattice_action(
    action: AbstractIncrementalLatticeAction,
    /,
    *,
    refresh_cadence: int,
    cache_tolerance: float = 1e-8,
) -> IncrementalMarkovTarget:
    """Lower an exact local action/cache implementation to one Markov target."""
    _validate_action(action)
    if not isinstance(action, AbstractIncrementalLatticeAction):
        raise TypeError("action must implement AbstractIncrementalLatticeAction.")

    return IncrementalMarkovTarget(
        initialize=eqx.Partial(_initialize_incremental_action, action),
        propose=eqx.Partial(_propose_incremental_action, action),
        select=eqx.Partial(_select_incremental_action, action),
        refresh=eqx.Partial(_refresh_incremental_action, action),
        target_id=f"{action.action_id}:boltzmann:incremental",
        refresh_cadence=refresh_cadence,
        cache_tolerance=cache_tolerance,
    )


def lattice_action_local_gradient(
    action: AbstractLatticeEuclideanAction,
    configuration: PyTree[Any],
    /,
) -> Array:
    """Differentiate the action through real local coordinates at zero."""
    if not isinstance(action, AbstractLatticeEuclideanAction):
        raise TypeError("action must implement AbstractLatticeEuclideanAction.")
    local_zero = jnp.zeros(action.local_coordinate_shape, dtype=float)
    return jax.grad(
        lambda local: action.action(action.geometry.retract(configuration, local))
    )(local_zero)


__all__ = [
    "AbstractIncrementalLatticeAction",
    "AbstractLatticeEuclideanAction",
    "LatticeActionEvidence",
    "LatticeReferenceMeasure",
    "compact_geometric_target_from_lattice_action",
    "full_target_from_lattice_action",
    "incremental_target_from_lattice_action",
    "lattice_action_local_gradient",
]
