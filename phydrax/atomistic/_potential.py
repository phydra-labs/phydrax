#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from abc import abstractmethod
from enum import StrEnum
from typing import Any

import equinox as eqx
import jax

from .._fingerprint import canonical_fingerprint
from .._identity import NumericRevision, SemanticProvenance
from .._strict import StrictModule
from .._trainable import NonTrainableState, ParameterOwner, partition_parameters


class AtomisticSpeciesKind(StrEnum):
    ATOMIC_NUMBER = "atomic-number"
    ATOM_TYPE_ID = "atom-type-id"


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
    species_kind: AtomisticSpeciesKind = eqx.field(static=True)
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
        species_kind: AtomisticSpeciesKind = AtomisticSpeciesKind.ATOMIC_NUMBER,
    ):
        if not isinstance(species_kind, AtomisticSpeciesKind):
            raise TypeError("species_kind must be AtomisticSpeciesKind.")
        values = {
            "conservative_energy": bool(conservative_energy),
            "finite_geometry": bool(finite_geometry),
            "orthorhombic_periodic": bool(orthorhombic_periodic),
            "triclinic_periodic": bool(triclinic_periodic),
            "cell_derivative": bool(cell_derivative),
            "local_energy": bool(local_energy),
            "local_energy_delta": bool(local_energy_delta),
            "dynamic_species": bool(dynamic_species),
            "species_kind": species_kind,
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
    interaction_site_geometry: bool = eqx.field(static=True)
    directed_graph: bool = eqx.field(static=True)
    bonded_geometry: bool = eqx.field(static=True)
    reciprocal_grid: bool = eqx.field(static=True)
    requirements_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        cutoff: float | None = None,
        pair_geometry: bool = False,
        interaction_site_geometry: bool = False,
        directed_graph: bool = False,
        bonded_geometry: bool = False,
        reciprocal_grid: bool = False,
    ):
        cutoff_ = None if cutoff is None else float(cutoff)
        if cutoff_ is not None and (not math.isfinite(cutoff_) or cutoff_ <= 0.0):
            raise ValueError("Potential cutoff must be finite and positive or None.")
        self.cutoff = cutoff_
        self.pair_geometry = bool(pair_geometry)
        self.interaction_site_geometry = bool(interaction_site_geometry)
        self.directed_graph = bool(directed_graph)
        self.bonded_geometry = bool(bonded_geometry)
        self.reciprocal_grid = bool(reciprocal_grid)
        self.requirements_id = canonical_fingerprint(
            {
                "kind": "atomistic-potential-requirements",
                "cutoff": cutoff_,
                "pair_geometry": self.pair_geometry,
                "interaction_site_geometry": self.interaction_site_geometry,
                "directed_graph": self.directed_graph,
                "bonded_geometry": self.bonded_geometry,
                "reciprocal_grid": self.reciprocal_grid,
            }
        )


class AbstractPreparedAtomisticPotential(StrictModule):
    """System-bound scalar-energy execution form."""

    prepared_id: eqx.AbstractVar[str]
    capabilities: eqx.AbstractVar[AtomisticPotentialCapabilities]
    requirements: eqx.AbstractVar[AtomisticPotentialRequirements]

    @abstractmethod
    def energy(self, context: Any, /) -> tuple[Any, Any]:
        """Return scalar energy and fixed-schema auxiliary evidence."""
        raise NotImplementedError


class AbstractAtomisticPotential(StrictModule, ParameterOwner):
    """Atomistic scalar-energy model whose numeric identity is its PARAMETER lane."""

    configuration: eqx.AbstractVar[Any]
    scale: eqx.AbstractVar[Any]
    precision: eqx.AbstractVar[Any]
    architecture_id: eqx.AbstractVar[str]
    method_id: eqx.AbstractVar[str]

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
        species_ids: Any,
        atom_mask: Any,
        atom_cases: Any,
        case_count: int,
        atom_capacity: int,
        graph: Any,
        /,
    ) -> tuple[Any, Any]:
        raise NotImplementedError


def atomistic_potential_revision(potential: AbstractAtomisticPotential, /) -> NumericRevision:
    """Return the canonical numeric revision of a potential's current parameters.

    The semantic provenance names the architecture and force method; the numeric
    content is every PARAMETER-role leaf, keyed by its tree path. Fixed and
    model-state leaves never enter it. This is a host boundary: the parameters
    must be concrete arrays, so call it outside `jit`, `vmap`, and `grad`.
    """

    if not isinstance(potential, AbstractAtomisticPotential):
        raise TypeError("potential must implement AbstractAtomisticPotential.")
    parameters = partition_parameters(potential)[0]
    return NumericRevision(
        SemanticProvenance(
            {
                "kind": "atomistic-potential",
                "architecture_id": potential.architecture_id,
                "method_id": potential.method_id,
            }
        ),
        {
            jax.tree_util.keystr(path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(parameters)[0]
        },
    )


__all__ = [
    "AbstractAtomisticPotential",
    "AbstractPreparedAtomisticPotential",
    "AtomisticPotentialCapabilities",
    "AtomisticSpeciesKind",
    "AtomisticPotentialRequirements",
    "atomistic_potential_revision",
]
