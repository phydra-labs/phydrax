#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import copy
from abc import abstractmethod
from typing import Any, cast, TypeVar

import equinox as eqx

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState


_AtomisticPotentialT = TypeVar("_AtomisticPotentialT", bound="AbstractAtomisticPotential")


class AtomisticPotentialCapabilities(StrictModule, NonTrainableState):
    """Static execution capabilities, never a scientific stability claim."""

    conservative_energy: bool = eqx.field(static=True)
    finite_geometry: bool = eqx.field(static=True)
    orthorhombic_periodic: bool = eqx.field(static=True)
    triclinic_periodic: bool = eqx.field(static=True)
    cell_derivative: bool = eqx.field(static=True)
    local_energy: bool = eqx.field(static=True)
    local_energy_delta: bool = eqx.field(static=True)
    dynamic_species: bool = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        conservative_energy: bool = True,
        finite_geometry: bool = True,
        orthorhombic_periodic: bool = False,
        triclinic_periodic: bool = False,
        cell_derivative: bool = False,
        local_energy: bool = True,
        local_energy_delta: bool = False,
        dynamic_species: bool = False,
    ):
        values = {
            "conservative_energy": bool(conservative_energy),
            "finite_geometry": bool(finite_geometry),
            "orthorhombic_periodic": bool(orthorhombic_periodic),
            "triclinic_periodic": bool(triclinic_periodic),
            "cell_derivative": bool(cell_derivative),
            "local_energy": bool(local_energy),
            "local_energy_delta": bool(local_energy_delta),
            "dynamic_species": bool(dynamic_species),
        }
        for name, value in values.items():
            setattr(self, name, value)
        self.capabilities_id = canonical_fingerprint(
            {"kind": "atomistic-potential-capabilities", **values}
        )


class AtomisticPotentialRequirements(StrictModule, NonTrainableState):
    """Prepared-context requirements used to avoid unused runtime allocations."""

    cutoff: float | None = eqx.field(static=True)
    pair_geometry: bool = eqx.field(static=True)
    directed_graph: bool = eqx.field(static=True)
    bonded_geometry: bool = eqx.field(static=True)
    reciprocal_grid: bool = eqx.field(static=True)
    requirements_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        cutoff: float | None = None,
        pair_geometry: bool = False,
        directed_graph: bool = False,
        bonded_geometry: bool = False,
        reciprocal_grid: bool = False,
    ):
        cutoff_ = None if cutoff is None else float(cutoff)
        if cutoff_ is not None and cutoff_ <= 0.0:
            raise ValueError("Potential cutoff must be positive or None.")
        self.cutoff = cutoff_
        self.pair_geometry = bool(pair_geometry)
        self.directed_graph = bool(directed_graph)
        self.bonded_geometry = bool(bonded_geometry)
        self.reciprocal_grid = bool(reciprocal_grid)
        self.requirements_id = canonical_fingerprint(
            {
                "kind": "atomistic-potential-requirements",
                "cutoff": cutoff_,
                "pair_geometry": self.pair_geometry,
                "directed_graph": self.directed_graph,
                "bonded_geometry": self.bonded_geometry,
                "reciprocal_grid": self.reciprocal_grid,
            }
        )


class AbstractPreparedAtomisticPotential(StrictModule):
    """System-bound scalar-energy execution form."""

    prepared_id: AbstractAttribute[str]
    capabilities: AbstractAttribute[AtomisticPotentialCapabilities]
    requirements: AbstractAttribute[AtomisticPotentialRequirements]

    @abstractmethod
    def energy(self, context: Any, /) -> tuple[Any, Any]:
        """Return scalar energy and fixed-schema auxiliary evidence."""
        raise NotImplementedError


class AbstractAtomisticPotential(StrictModule):
    """Atomistic scalar-energy model with checkpointable parameter provenance."""

    configuration: AbstractAttribute[Any]
    scale: AbstractAttribute[Any]
    precision: AbstractAttribute[Any]
    architecture_id: AbstractAttribute[str]
    parameter_state_id: AbstractAttribute[str]
    potential_id: AbstractAttribute[str]
    method_id: AbstractAttribute[str]

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        return AtomisticPotentialCapabilities()

    @property
    def requirements(self) -> AtomisticPotentialRequirements:
        return AtomisticPotentialRequirements(
            cutoff=float(self.configuration.cutoff),
            pair_geometry=True,
            directed_graph=True,
        )

    @abstractmethod
    def _validate_batch(self, batch: Any, /) -> None:
        raise NotImplementedError

    @abstractmethod
    def _energy_unchecked(
        self, batch: Any, positions: Any, execution: Any, /
    ) -> tuple[Any, Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def graph_energy(
        self,
        atomic_numbers: Any,
        atom_mask: Any,
        atom_cases: Any,
        case_count: int,
        atom_capacity: int,
        graph: Any,
        /,
    ) -> tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def parameter_state_tree(self, /) -> Any:
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
    """Return an immutable model copy with refreshed content-addressed provenance."""

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


def _with_atomistic_potential_identity(
    potential: _AtomisticPotentialT,
    identity_source: AbstractAtomisticPotential,
    /,
) -> _AtomisticPotentialT:
    synchronized = cast(_AtomisticPotentialT, copy.copy(potential))
    object.__setattr__(
        synchronized, "parameter_state_id", identity_source.parameter_state_id
    )
    object.__setattr__(synchronized, "potential_id", identity_source.potential_id)
    return synchronized


def initialize_atomistic_potential_identity(
    potential: AbstractAtomisticPotential, /
) -> tuple[str, str]:
    state_id = _parameter_state_id(potential)
    potential_id = canonical_fingerprint(
        {
            "kind": "evaluated-atomistic-potential",
            "architecture": potential.architecture_id,
            "parameter_state": state_id,
        }
    )
    return state_id, potential_id


__all__ = [
    "AbstractAtomisticPotential",
    "AbstractPreparedAtomisticPotential",
    "AtomisticPotentialCapabilities",
    "AtomisticPotentialRequirements",
    "checkpoint_atomistic_potential",
]
