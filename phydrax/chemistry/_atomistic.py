#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bidirectional adapters between chemistry surfaces and atomistic execution.

`AtomisticPotentialEnergySurface` is the native PES bridge: it binds a prepared
atomistic Hamiltonian whose program owns its neighborhood resources.  Its host
`evaluate` initializes resources for the given (finite or fixed-cell periodic)
point, grows declared capacities when the point does not fit, and evaluates
energy and forces natively; device-resident consumers (native NEB, relaxation)
use ``potential`` directly through `phydrax.atomistic._relaxation`.
`ExternalAtomisticPotentialEnergySurface` remains the honest host-provider
route; no host callback is ever hidden inside a compiled native path.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import IdentityWords
from ..atomistic import (
    AbstractExternalAtomisticProvider,
    AtomisticUnitSystem,
    ExternalAtomisticEvaluation,
    PreparedAtomisticSystem,
)
from ..atomistic._potential_program import AbstractPreparedAtomisticHamiltonian
from ..atomistic._relaxation import trial_geometry
from ..typing import checked
from ._surface import (
    AbstractPreparedPotentialEnergySurface,
    PotentialEnergySurfaceCapabilities,
    PotentialEnergySurfaceEvaluation,
)


class AtomisticPotentialEnergySurface(AbstractPreparedPotentialEnergySurface):
    """Native prepared atomistic Hamiltonian as a chemistry energy surface.

    Finite and periodic systems are admitted.  ``evaluate(positions,
    cell_vectors)`` takes unwrapped Cartesian positions and, for a periodic
    system, optional cell rows (default: the prepared cell); integer image
    counts are bound outside differentiation.  A capacity-limited point is
    replayed at the next declared capacity level; an exhausted ladder or a
    scientific resource failure reports an unsuccessful evaluation.

    PyTree: the system, provider (prepared Hamiltonian binding) and surface
    digests are dynamic `IdentityWords` (``system_identity``,
    ``provider_identity``, ``surface_identity``); the host properties
    ``system_id``, ``provider_id`` and ``surface_id`` decode the same strings,
    so device NEB/relaxation runtimes holding this surface compile once per
    layout.
    """

    system: PreparedAtomisticSystem
    potential: AbstractPreparedAtomisticHamiltonian
    system_identity: IdentityWords
    provider_identity: IdentityWords
    surface_identity: IdentityWords
    units: AtomisticUnitSystem
    capabilities: PotentialEnergySurfaceCapabilities

    @checked
    def __init__(
        self,
        system: PreparedAtomisticSystem,
        potential: AbstractPreparedAtomisticHamiltonian,
        /,
    ) -> None:
        if potential.system.prepared_id != system.prepared_id:
            raise ValueError("Atomistic potential belongs to another prepared system.")
        capabilities = PotentialEnergySurfaceCapabilities(
            forces=True,
            hessian=False,
            conservative=potential.plan.capabilities.conservative_energy,
            differentiable=False,
        )
        self.system = system
        self.potential = potential
        self.system_identity = system.plan.system_identity
        self.provider_identity = potential.prepared_identity
        self.units = system.plan.units
        self.capabilities = capabilities
        self.surface_identity = IdentityWords.of(
            {
                "kind": "atomistic-potential-energy-surface",
                "system": system.prepared_id,
                "potential": potential.prepared_id,
            }
        )

    @property
    def system_id(self) -> str:
        return self.system_identity.fingerprint()

    @property
    def provider_id(self) -> str:
        return self.provider_identity.fingerprint()

    @property
    def surface_id(self) -> str:
        return self.surface_identity.fingerprint()

    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> PotentialEnergySurfaceEvaluation:
        dtype = self.system.plan.coordinate_dtype
        coordinate = jnp.asarray(positions, dtype=dtype)
        if coordinate.shape != (self.system.capacity, 3):
            raise ValueError(f"positions must have shape {(self.system.capacity, 3)}.")
        cell = self.system.cell
        if cell is None:
            if cell_vectors is not None:
                raise ValueError("Finite atomistic surfaces reject cell vectors.")
            vectors = None
        else:
            vectors = (
                cell.vectors.astype(coordinate.dtype)
                if cell_vectors is None
                else jnp.asarray(cell_vectors, dtype=coordinate.dtype)
            )
            if vectors.shape != cell.vectors.shape:
                raise ValueError(f"cell_vectors must have shape {cell.vectors.shape}.")
        active = jnp.asarray(self.system.active_mask)[:, None]
        point = jnp.where(active, coordinate, 0.0)
        program = self.potential
        bundle = program.neighborhood_bundle
        geometry, wrapped, images = trial_geometry(program, bundle, point, vectors)
        resources = bundle.initialize(geometry)
        exhausted = False
        while bool(np.asarray(resources.capacity_only)[0]):
            try:
                bundle = bundle.grow_capacities(resources)
            except RuntimeError:
                exhausted = True
                break
            geometry, wrapped, images = trial_geometry(program, bundle, point, vectors)
            resources = bundle.initialize(geometry)
        if bundle.prepared_id != program.neighborhood_bundle.prepared_id:
            program = program.with_neighborhood_bundle(bundle)
        if vectors is None:
            value = program.evaluate(wrapped, resources)
        else:
            value = program.evaluate(
                wrapped, resources, cell_vectors=vectors, image_counts=images
            )
        successful = value.successful & resources.successful[0] & (not exhausted)
        energy = jnp.where(successful, value.energy, jnp.nan)
        forces = jnp.where(successful, value.forces, jnp.nan)
        host_coordinate, host_energy, host_forces, host_successful = jax.device_get(
            (coordinate, energy, forces, successful)
        )
        source_id = canonical_fingerprint(
            {
                "kind": "atomistic-surface-evaluation",
                "surface": self.surface_id,
                "arrays": array_tree_fingerprint(
                    {
                        "positions": host_coordinate,
                        "energy": host_energy,
                        "forces": host_forces,
                    }
                ),
            }
        )
        return PotentialEnergySurfaceEvaluation(
            energy,
            forces,
            None,
            successful,
            provider_id=self.provider_id,
            source_result_id=source_id,
            _identity_leaves=(
                np.asarray(host_energy).reshape(()),
                np.asarray(host_forces, dtype=host_energy.dtype),
                None,
                np.asarray(host_successful, dtype=np.bool_).reshape(()),
            ),
        )


class ExternalAtomisticPotentialEnergySurface(AbstractPreparedPotentialEnergySurface):
    """An external atomistic provider as a host chemistry energy surface.

    PyTree: the system and surface digests are dynamic `IdentityWords`
    (``system_identity``, ``surface_identity``); the provider binding stays
    with ``provider``.  The host properties ``system_id``, ``provider_id``
    (the provider's label or decoded digest) and ``surface_id`` decode the
    same strings.
    """

    system: PreparedAtomisticSystem
    provider: AbstractExternalAtomisticProvider
    system_identity: IdentityWords
    surface_identity: IdentityWords
    units: AtomisticUnitSystem
    capabilities: PotentialEnergySurfaceCapabilities

    @checked
    def __init__(
        self,
        system: PreparedAtomisticSystem,
        provider: AbstractExternalAtomisticProvider,
        /,
    ) -> None:
        capabilities = PotentialEnergySurfaceCapabilities(
            forces=True,
            conservative=provider.conservative,
            differentiable=provider.differentiable,
        )
        self.system = system
        self.provider = provider
        self.system_identity = system.plan.system_identity
        self.units = system.plan.units
        self.capabilities = capabilities
        self.surface_identity = IdentityWords.of(
            {
                "kind": "external-atomistic-potential-energy-surface",
                "system": system.prepared_id,
                "provider": provider.provider_id,
                "capabilities": capabilities.capabilities_id,
            }
        )

    @property
    def system_id(self) -> str:
        return self.system_identity.fingerprint()

    @property
    def provider_id(self) -> str:
        return self.provider.provider_id

    @property
    def surface_id(self) -> str:
        return self.surface_identity.fingerprint()

    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> PotentialEnergySurfaceEvaluation:
        value = self.provider.evaluate(self.system, positions, cell_vectors)
        if not bool(value.bound_to(self.provider)):
            raise ValueError("External atomistic provider changed its bound identity.")
        host_energy, host_forces, host_stress, host_successful = jax.device_get(
            (value.energy, value.forces, value.stress, value.successful)
        )
        source_id = canonical_fingerprint(
            {
                "kind": "external-atomistic-surface-evaluation",
                "surface": self.surface_id,
                "arrays": array_tree_fingerprint(
                    {
                        "energy": host_energy,
                        "forces": host_forces,
                        "stress": host_stress,
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
            _identity_leaves=(
                np.asarray(host_energy).reshape(()),
                np.asarray(host_forces, dtype=host_energy.dtype),
                None,
                np.asarray(host_successful, dtype=np.bool_).reshape(()),
            ),
        )


class SurfaceExternalAtomisticProvider(AbstractExternalAtomisticProvider):
    """A chemistry energy surface served through the atomistic provider boundary.

    A digest provider: ``provider_identity`` is the dynamic exact digest of
    the adapted surface (``provider_label`` is ``None``); the host
    ``provider_id`` decodes it.
    """

    surface: AbstractPreparedPotentialEnergySurface
    provider_identity: IdentityWords
    provider_label: str | None = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)

    @checked
    def __init__(self, surface: AbstractPreparedPotentialEnergySurface, /) -> None:
        self.surface = surface
        self.provider_identity = IdentityWords.of(
            {
                "kind": "surface-external-atomistic-provider",
                "surface": surface.surface_id,
            }
        )
        self.provider_label = None
        self.conservative = surface.capabilities.conservative
        self.differentiable = surface.capabilities.differentiable

    @checked
    def evaluate(
        self,
        system: PreparedAtomisticSystem,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        /,
    ) -> ExternalAtomisticEvaluation:
        if system.plan.system_id != self.surface.system_id:
            raise ValueError("Atomistic dynamics system differs from chemistry surface.")
        result = self.surface.evaluate(positions, cell_vectors)
        return ExternalAtomisticEvaluation(
            result.energy,
            result.forces,
            None,
            result.successful,
            provider_identity=self.provider_identity,
        )


__all__ = [
    "AtomisticPotentialEnergySurface",
    "ExternalAtomisticPotentialEnergySurface",
    "SurfaceExternalAtomisticProvider",
]
