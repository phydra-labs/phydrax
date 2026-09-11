#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bidirectional adapters between chemistry surfaces and atomistic execution."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..atomistic import (
    AbstractExternalAtomisticProvider,
    AtomisticUnitSystem,
    ExternalAtomisticEvaluation,
    PreparedAtomisticPotentialProgram,
    PreparedAtomisticSystem,
)
from ..discretization import AbstractPreparedParticleNeighborhood
from ._surface import (
    AbstractPreparedPotentialEnergySurface,
    PotentialEnergySurfaceCapabilities,
    PotentialEnergySurfaceEvaluation,
)


class AtomisticPotentialEnergySurface(AbstractPreparedPotentialEnergySurface):
    system: PreparedAtomisticSystem
    potential: PreparedAtomisticPotentialProgram
    neighborhood: AbstractPreparedParticleNeighborhood
    system_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)
    units: AtomisticUnitSystem
    capabilities: PotentialEnergySurfaceCapabilities

    def __init__(
        self,
        system: PreparedAtomisticSystem,
        potential: PreparedAtomisticPotentialProgram,
        neighborhood: AbstractPreparedParticleNeighborhood,
        /,
    ):
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("system must be PreparedAtomisticSystem.")
        if not isinstance(potential, PreparedAtomisticPotentialProgram):
            raise TypeError("potential must be PreparedAtomisticPotentialProgram.")
        if not isinstance(neighborhood, AbstractPreparedParticleNeighborhood):
            raise TypeError("neighborhood must be a prepared particle neighborhood.")
        if potential.system.prepared_id != system.prepared_id:
            raise ValueError("Atomistic potential belongs to another prepared system.")
        if neighborhood.particle_discretization_id != system.particles.prepared_id:
            raise ValueError("Atomistic neighborhood belongs to another particle system.")
        if system.cell is not None and any(system.cell.periodic_axes):
            raise ValueError(
                "The initial atomistic chemistry surface supports finite systems only."
            )
        capabilities = PotentialEnergySurfaceCapabilities(
            forces=True,
            hessian=False,
            conservative=potential.plan.capabilities.conservative_energy,
            differentiable=False,
        )
        self.system = system
        self.potential = potential
        self.neighborhood = neighborhood
        self.system_id = system.plan.system_id
        self.provider_id = potential.prepared_id
        self.units = system.plan.units
        self.capabilities = capabilities
        self.surface_id = canonical_fingerprint(
            {
                "kind": "atomistic-potential-energy-surface",
                "system": system.prepared_id,
                "potential": potential.prepared_id,
                "neighborhood": neighborhood.prepared_id,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> PotentialEnergySurfaceEvaluation:
        if cell_vectors is not None:
            raise ValueError("Finite atomistic chemistry surfaces reject cell vectors.")
        coordinate = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        relation = self.neighborhood.build(
            coordinate, active_mask=self.system.active_mask
        )
        value = self.potential.evaluate(
            coordinate,
            relation,
            species=self.system.plan.atom_type_ids,
        )
        source_id = canonical_fingerprint(
            {
                "kind": "atomistic-surface-evaluation",
                "surface": self.surface_id,
                "arrays": array_tree_fingerprint(
                    {
                        "positions": np.asarray(coordinate),
                        "energy": np.asarray(value.energy),
                        "forces": np.asarray(value.forces),
                    }
                ),
            }
        )
        return PotentialEnergySurfaceEvaluation(
            value.energy,
            value.forces,
            None,
            value.successful,
            provider_id=self.provider_id,
            source_result_id=source_id,
        )


class ExternalAtomisticPotentialEnergySurface(AbstractPreparedPotentialEnergySurface):
    system: PreparedAtomisticSystem
    provider: AbstractExternalAtomisticProvider
    system_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)
    units: AtomisticUnitSystem
    capabilities: PotentialEnergySurfaceCapabilities

    def __init__(
        self,
        system: PreparedAtomisticSystem,
        provider: AbstractExternalAtomisticProvider,
        /,
    ):
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("system must be PreparedAtomisticSystem.")
        if not isinstance(provider, AbstractExternalAtomisticProvider):
            raise TypeError("provider must implement AbstractExternalAtomisticProvider.")
        capabilities = PotentialEnergySurfaceCapabilities(
            forces=True,
            conservative=provider.conservative,
            differentiable=provider.differentiable,
        )
        self.system = system
        self.provider = provider
        self.system_id = system.plan.system_id
        self.provider_id = provider.provider_id
        self.units = system.plan.units
        self.capabilities = capabilities
        self.surface_id = canonical_fingerprint(
            {
                "kind": "external-atomistic-potential-energy-surface",
                "system": system.prepared_id,
                "provider": provider.provider_id,
                "capabilities": capabilities.capabilities_id,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> PotentialEnergySurfaceEvaluation:
        value = self.provider.evaluate(self.system, positions, cell_vectors)
        source_id = canonical_fingerprint(
            {
                "kind": "external-atomistic-surface-evaluation",
                "surface": self.surface_id,
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(value.energy),
                        "forces": np.asarray(value.forces),
                        "stress": None if value.stress is None else np.asarray(value.stress),
                    }
                ),
            }
        )
        return PotentialEnergySurfaceEvaluation(
            value.energy,
            value.forces,
            None,
            value.successful,
            provider_id=value.provider_id,
            source_result_id=source_id,
        )


class SurfaceExternalAtomisticProvider(AbstractExternalAtomisticProvider):
    surface: AbstractPreparedPotentialEnergySurface
    provider_id: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)

    def __init__(self, surface: AbstractPreparedPotentialEnergySurface, /):
        if not isinstance(surface, AbstractPreparedPotentialEnergySurface):
            raise TypeError("surface must be a prepared potential-energy surface.")
        self.surface = surface
        self.provider_id = canonical_fingerprint(
            {
                "kind": "surface-external-atomistic-provider",
                "surface": surface.surface_id,
            }
        )
        self.conservative = surface.capabilities.conservative
        self.differentiable = surface.capabilities.differentiable

    def evaluate(
        self,
        system: PreparedAtomisticSystem,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        /,
    ) -> ExternalAtomisticEvaluation:
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("system must be PreparedAtomisticSystem.")
        if system.plan.system_id != self.surface.system_id:
            raise ValueError("Atomistic dynamics system differs from chemistry surface.")
        result = self.surface.evaluate(positions, cell_vectors)
        return ExternalAtomisticEvaluation(
            result.energy,
            result.forces,
            None,
            result.successful,
            self.provider_id,
        )


__all__ = [
    "AtomisticPotentialEnergySurface",
    "ExternalAtomisticPotentialEnergySurface",
    "SurfaceExternalAtomisticProvider",
]
