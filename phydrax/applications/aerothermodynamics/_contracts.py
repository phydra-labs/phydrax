#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _required_identifier(value: str, name: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _optional_identifier(value: str | None, /) -> str | None:
    if value is None:
        return None
    return _required_identifier(value, "Optional support identifier")


class AerothermodynamicSupportTuple(StrictModule, NonTrainableState):
    """Exact physical, numerical, execution, and data support identity."""

    gas_system_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    thermochemistry_id: str | None = eqx.field(static=True)
    electromagnetic_id: str | None = eqx.field(static=True)
    radiation_id: str | None = eqx.field(static=True)
    surface_id: str | None = eqx.field(static=True)
    material_id: str | None = eqx.field(static=True)
    kinetic_id: str | None = eqx.field(static=True)
    hybrid_id: str | None = eqx.field(static=True)
    turbulence_id: str | None = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    species_count: int = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    radiation_group_count: int = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        gas_system_id: str,
        transport_id: str,
        discretization_id: str,
        topology_id: str,
        backend: str,
        precision: str,
        species_count: int,
        mode_count: int = 0,
        radiation_group_count: int = 0,
        thermochemistry_id: str | None = None,
        electromagnetic_id: str | None = None,
        radiation_id: str | None = None,
        surface_id: str | None = None,
        material_id: str | None = None,
        kinetic_id: str | None = None,
        hybrid_id: str | None = None,
        turbulence_id: str | None = None,
    ):
        species = int(species_count)
        modes = int(mode_count)
        groups = int(radiation_group_count)
        if species <= 0 or modes < 0 or groups < 0:
            raise ValueError("Support dimensions are invalid.")
        self.gas_system_id = _required_identifier(gas_system_id, "Gas system ID")
        self.transport_id = _required_identifier(transport_id, "Transport ID")
        self.discretization_id = _required_identifier(
            discretization_id, "Discretization ID"
        )
        self.topology_id = _required_identifier(topology_id, "Topology ID")
        self.backend = _required_identifier(backend, "Backend")
        self.precision = _required_identifier(precision, "Precision")
        self.thermochemistry_id = _optional_identifier(thermochemistry_id)
        self.electromagnetic_id = _optional_identifier(electromagnetic_id)
        self.radiation_id = _optional_identifier(radiation_id)
        self.surface_id = _optional_identifier(surface_id)
        self.material_id = _optional_identifier(material_id)
        self.kinetic_id = _optional_identifier(kinetic_id)
        self.hybrid_id = _optional_identifier(hybrid_id)
        self.turbulence_id = _optional_identifier(turbulence_id)
        self.species_count = species
        self.mode_count = modes
        self.radiation_group_count = groups
        self.support_id = canonical_fingerprint(
            {
                "kind": "aerothermodynamic-support-tuple",
                "gas_system": self.gas_system_id,
                "transport": self.transport_id,
                "thermochemistry": self.thermochemistry_id,
                "electromagnetic": self.electromagnetic_id,
                "radiation": self.radiation_id,
                "surface": self.surface_id,
                "material": self.material_id,
                "kinetic": self.kinetic_id,
                "hybrid": self.hybrid_id,
                "turbulence": self.turbulence_id,
                "discretization": self.discretization_id,
                "topology": self.topology_id,
                "backend": self.backend,
                "precision": self.precision,
                "species_count": species,
                "mode_count": modes,
                "radiation_group_count": groups,
            }
        )


class AerothermodynamicConservationLedger(StrictModule):
    """Global extensive exchange ledger for one candidate macro-step."""

    mass_defect: Array
    element_defect: Array
    charge_defect: Array
    momentum_defect: Array
    energy_defect: Array
    surface_site_defect: Array
    finite: Array
    successful: Array

    @classmethod
    def from_exchanges(
        cls,
        *,
        mass: ArrayLike,
        elements: ArrayLike,
        charge: ArrayLike,
        momentum: ArrayLike,
        energy: ArrayLike,
        surface_sites: ArrayLike = 0.0,
        tolerance: float = 1.0e-10,
    ) -> AerothermodynamicConservationLedger:
        values = tuple(
            jnp.asarray(value)
            for value in (mass, elements, charge, momentum, energy, surface_sites)
        )
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Conservation tolerance must be finite and positive.")
        finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in values))
        )
        maximum = jnp.max(jnp.stack(tuple(jnp.max(jnp.abs(value)) for value in values)))
        return cls(*values, finite, finite & (maximum <= tolerance_))


class AerothermodynamicCapabilityStatus(StrictModule, NonTrainableState):
    support: AerothermodynamicSupportTuple
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    scientific: bool = eqx.field(static=True)
    performance: bool = eqx.field(static=True)
    operational: bool = eqx.field(static=True)
    security: bool = eqx.field(static=True)
    released: bool = eqx.field(static=True)
    status_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: AerothermodynamicSupportTuple,
        evidence_ids: tuple[str, ...],
        /,
        *,
        scientific: bool,
        performance: bool,
        operational: bool,
        security: bool,
        released: bool = False,
    ):
        evidence = tuple(str(value) for value in evidence_ids)
        gates = tuple(
            bool(value) for value in (scientific, performance, operational, security)
        )
        released_ = bool(released)
        if (
            not isinstance(support, AerothermodynamicSupportTuple)
            or any(not value for value in evidence)
            or len(set(evidence)) != len(evidence)
            or (released_ and (not all(gates) or not evidence))
        ):
            raise ValueError(
                "Capability status cannot release without all independent gates."
            )
        self.support = support
        self.evidence_ids = evidence
        self.scientific, self.performance, self.operational, self.security = gates
        self.released = released_
        self.status_id = canonical_fingerprint(
            {
                "kind": "aerothermodynamic-capability-status",
                "support": support.support_id,
                "evidence": evidence,
                "gates": gates,
                "released": released_,
            }
        )


class AerothermodynamicResourceCaps(StrictModule, NonTrainableState):
    maximum_particles: int = eqx.field(static=True)
    maximum_collision_events: int = eqx.field(static=True)
    maximum_surface_events: int = eqx.field(static=True)
    maximum_topology_events: int = eqx.field(static=True)
    maximum_radiation_groups: int = eqx.field(static=True)
    caps_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_particles: int,
        maximum_collision_events: int,
        maximum_surface_events: int,
        maximum_topology_events: int,
        maximum_radiation_groups: int,
    ):
        values = tuple(
            int(value)
            for value in (
                maximum_particles,
                maximum_collision_events,
                maximum_surface_events,
                maximum_topology_events,
                maximum_radiation_groups,
            )
        )
        if any(value < 0 for value in values):
            raise ValueError("Aerothermodynamic resource capacities must be nonnegative.")
        (
            self.maximum_particles,
            self.maximum_collision_events,
            self.maximum_surface_events,
            self.maximum_topology_events,
            self.maximum_radiation_groups,
        ) = values
        self.caps_id = canonical_fingerprint(
            {"kind": "aerothermodynamic-resource-caps", "values": values}
        )


__all__ = [
    "AerothermodynamicCapabilityStatus",
    "AerothermodynamicConservationLedger",
    "AerothermodynamicResourceCaps",
    "AerothermodynamicSupportTuple",
]
