#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-admitted material/contact semantics lowered to differentiable SI arrays."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...meshing import MeshAttribute, MeshPatch, MeshZone, MeshZoneRole
from ...units import KELVIN, UnitDefinition, VOLT
from ._coupled import (
    MaterialInterface,
    ThermalPort,
    TrapBinding,
    TunnelingChannel,
)
from ._high_field import LocalImpactIonization
from ._materials import DielectricMaterial, SemiconductorMaterial
from ._quantities import (
    _positive_scalar,
    _si,
    _text,
    BOLTZMANN_CONSTANT_SI,
    ELEMENTARY_CHARGE_SI,
    PER_CUBIC_METER,
)
from ._state import SemiconductorStateLayout
from ._support import TransportSupport


Material = SemiconductorMaterial | DielectricMaterial
Region = MeshZone | MeshPatch


def _region(region, role):
    if not isinstance(region, (MeshZone, MeshPatch)):
        raise TypeError("Device regions must be native MeshZone or MeshPatch bindings.")
    if isinstance(region, MeshZone) and region.role != role:
        raise ValueError(f"Device region must have native {role.value} zone semantics.")
    return region


class MaterialBinding(StrictModule):
    """One exclusive material assignment on an exact native mesh region."""

    region: Region
    material: Material

    def __init__(self, region: Region, material: Material, /):
        self.region = _region(region, MeshZoneRole.MATERIAL)
        if not isinstance(material, (SemiconductorMaterial, DielectricMaterial)):
            raise TypeError(
                "Material bindings require a semiconductor or dielectric material."
            )
        self.material = material


class OhmicContact(StrictModule):
    """Neutral majority/minority reservoir; voltage pins both quasi-Fermi energies."""

    name: str = eqx.field(static=True)
    region: Region

    def __init__(self, name: str, region: Region, /):
        self.name = _text(name, "terminal name")
        self.region = _region(region, MeshZoneRole.BOUNDARY)


class GateContact(StrictModule):
    """Insulating metal/dielectric contact pinning only electrostatic potential.

    ``potential_offset`` is an explicit voltage reference/work-function offset;
    it is not inferred from the semiconductor electron affinity.
    """

    name: str = eqx.field(static=True)
    region: Region
    potential_offset: Array

    def __init__(
        self,
        name: str,
        region: Region,
        /,
        *,
        potential_offset=0.0,
        voltage_unit: UnitDefinition = VOLT,
    ):
        self.name = _text(name, "terminal name")
        self.region = _region(region, MeshZoneRole.BOUNDARY)
        offset = _si(potential_offset, voltage_unit, VOLT)
        if np.asarray(offset).shape != () or not np.isfinite(np.asarray(offset)):
            raise ValueError("Gate potential offset must be a finite scalar voltage.")
        self.potential_offset = offset


def _nodal_density(value, unit, count, name):
    array = _si(value, unit, PER_CUBIC_METER)
    if array.shape not in ((), (count,)):
        raise ValueError(f"{name} must be scalar or one value per support node.")
    host = np.asarray(array)
    if not np.all(np.isfinite(host)) or np.any(host < 0):
        raise ValueError(f"{name} must be finite and nonnegative.")
    return jnp.broadcast_to(array, (count,))


def _resolve_dopants(support, baseline, attributes, name):
    count = support.positions.shape[0]
    resolved = baseline
    assigned = np.zeros(count, dtype=bool)
    for attribute in attributes:
        if not isinstance(attribute, MeshAttribute):
            raise TypeError(f"{name} attributes must be native MeshAttribute values.")
        mask = support.resolve_scope(attribute.scope)
        if attribute.scope.entity_dimension != 0 or attribute.component_shape != ():
            raise ValueError(
                "Dopants require scalar attributes on the native node entity set."
            )
        if attribute.unit is None:
            raise ValueError(
                "Dopant attributes require an explicit native number-density unit."
            )
        if np.any(assigned & mask):
            raise ValueError(f"Overlapping {name} attribute assignments are ambiguous.")
        assigned |= mask
        values = _si(attribute.values, attribute.unit, PER_CUBIC_METER)
        if not np.all(np.isfinite(np.asarray(values))) or np.any(np.asarray(values) < 0):
            raise ValueError("Dopant attribute densities must be finite and nonnegative.")
        lookup = {
            int(identifier): i
            for i, identifier in enumerate(np.asarray(support.node_ids))
        }
        indices = np.asarray(
            [
                lookup[int(identifier)]
                for identifier in np.asarray(attribute.scope.entity_ids)
            ]
        )
        resolved = resolved.at[indices].set(values)
    return resolved


def _require_pins(support, semiconductor, potential, ohmic):
    """Reject disconnected gauge modes before a root solve is attempted."""
    count = support.positions.shape[0]
    edges = tuple(zip(np.asarray(support.tail), np.asarray(support.head), strict=True))
    for active, pins, label in (
        (np.ones(count, dtype=bool), potential, "potential"),
        (semiconductor, ohmic, "semiconductor quasi-Fermi"),
    ):
        parent = np.arange(count)

        def root(index, parent=parent):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for tail, head in edges:
            if active[tail] and active[head]:
                parent[root(int(head))] = root(int(tail))
        components = {root(int(index)) for index in np.flatnonzero(active)}
        pinned = {root(int(index)) for index in np.flatnonzero(active & pins)}
        if components - pinned:
            raise ValueError(
                f"Every connected {label} component requires an appropriate contact."
            )


class DevicePlan(StrictModule):
    """Prepared host contract for a declared classical semiconductor model.

    The default is the original isothermal homojunction reduction. Explicit
    DOS thermodynamics admit aligned heterojunction bands. Optional interface,
    trap, thermal, carrier-energy, tunneling and local-ionization owners extend
    the same named physical state; no mechanism is enabled implicitly.

    Bulk solver coordinates are fixed-reference charts ``psi/Vt_ref``,
    ``EFn/(k*T_ref)``, ``EFp/(k*T_ref)`` and optional logarithmic temperatures.
    Conserved storage is always reconstructed as extensive particles/energy.
    Interface Fermi traces are algebraic and trap storage is an actual surface
    or volume population. All electronic energies in one plan share one named
    reference.
    """

    support: TransportSupport
    permittivity: Array
    intrinsic_density: Array
    donor_density: Array
    acceptor_density: Array
    electron_mobility: Array
    hole_mobility: Array
    electron_lifetime: Array
    hole_lifetime: Array
    semiconductor_mask: Array
    temperature: Array
    terminal_index: Array
    ohmic_mask: Array
    potential_mask: Array
    contact_potential_offset: Array
    material_index: Array
    material_models: tuple[Material, ...]
    layout: SemiconductorStateLayout
    edge_lengths: Array
    energy_scale: Array
    interfaces: tuple[MaterialInterface, ...]
    traps: tuple[TrapBinding, ...]
    thermal_ports: tuple[ThermalPort, ...]
    tunneling: tuple[TunnelingChannel, ...]
    ionization: LocalImpactIonization | None
    terminal_names: tuple[str, ...] = eqx.field(static=True)
    material_nodes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    material_edges: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    interface_nodes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    interface_materials: tuple[tuple[int, int], ...] = eqx.field(static=True)
    electrothermal: bool = eqx.field(static=True)
    carrier_energy: bool = eqx.field(static=True)

    def __init__(
        self,
        support: TransportSupport,
        material: Material | None = None,
        /,
        *,
        materials: tuple[MaterialBinding, ...] = (),
        donor_density: ArrayLike = 0.0,
        acceptor_density: ArrayLike = 0.0,
        donor_attributes: tuple[MeshAttribute, ...] = (),
        acceptor_attributes: tuple[MeshAttribute, ...] = (),
        contacts: tuple[OhmicContact | GateContact, ...] = (),
        interfaces: tuple[MaterialInterface, ...] = (),
        traps: tuple[TrapBinding, ...] = (),
        thermal_ports: tuple[ThermalPort, ...] = (),
        tunneling: tuple[TunnelingChannel, ...] = (),
        ionization: LocalImpactIonization | None = None,
        electrothermal: bool = False,
        carrier_energy: bool = False,
        temperature=300.0,
        density_unit: UnitDefinition = PER_CUBIC_METER,
        temperature_unit: UnitDefinition = KELVIN,
    ):
        if not isinstance(support, TransportSupport):
            raise TypeError("DevicePlan requires a prepared TransportSupport.")
        if not isinstance(electrothermal, bool) or not isinstance(carrier_energy, bool):
            raise TypeError(
                "electrothermal and carrier_energy must be Boolean model choices."
            )
        if carrier_energy and not electrothermal:
            raise ValueError(
                "Carrier-energy evolution requires the lattice-energy model."
            )
        for values, kind, label in (
            (interfaces, MaterialInterface, "interfaces"),
            (traps, TrapBinding, "traps"),
            (thermal_ports, ThermalPort, "thermal_ports"),
            (tunneling, TunnelingChannel, "tunneling"),
        ):
            if not isinstance(values, tuple) or not all(
                isinstance(value, kind) for value in values
            ):
                raise TypeError(
                    f"{label} must be an immutable tuple of {kind.__name__} values."
                )
        if ionization is not None and not isinstance(ionization, LocalImpactIonization):
            raise TypeError("ionization must be LocalImpactIonization or None.")
        if thermal_ports and not electrothermal:
            raise ValueError("Thermal ports require the lattice-energy model.")
        if material is not None and materials:
            raise ValueError(
                "Specify either one whole-device material or explicit material bindings."
            )
        if material is None and not materials:
            raise ValueError(
                "Every support node requires an explicit material assignment."
            )
        count = support.positions.shape[0]
        self.support = support
        self.temperature = _positive_scalar(
            temperature, temperature_unit, KELVIN, "device temperature"
        )
        if material is not None:
            if not isinstance(material, (SemiconductorMaterial, DielectricMaterial)):
                raise TypeError(
                    "material must be a semiconductor or dielectric material."
                )
            models, masks = (material,), (np.ones(count, dtype=bool),)
        else:
            if not all(isinstance(binding, MaterialBinding) for binding in materials):
                raise TypeError("materials must contain MaterialBinding values.")
            models = tuple(binding.material for binding in materials)
            masks = tuple(
                support.resolve_scope(binding.region.scope) for binding in materials
            )
        ownership = np.zeros(count, dtype=np.int32)
        for mask in masks:
            ownership += mask
        if np.any(ownership == 0):
            raise ValueError("Uncovered nodes have no material assignment.")
        if np.any(ownership > 1):
            raise ValueError(
                "Overlapping material assignments require explicit exclusive nodal zones."
            )
        self.material_models = models
        self.material_index = jnp.asarray(
            np.argmax(np.stack(masks), axis=0), dtype=jnp.int32
        )
        material_index = np.asarray(self.material_index)
        self.material_nodes = tuple(
            tuple(int(node) for node in np.flatnonzero(mask)) for mask in masks
        )
        tail, head = np.asarray(support.tail), np.asarray(support.head)
        self.material_edges = tuple(
            tuple(
                int(edge)
                for edge in np.flatnonzero(
                    (material_index[tail] == index) & (material_index[head] == index)
                )
            )
            for index in range(len(models))
        )
        lengths = np.linalg.norm(
            np.asarray(support.positions)[head] - np.asarray(support.positions)[tail],
            axis=-1,
        )
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
            raise ValueError(
                "Every transport edge requires a finite positive physical length."
            )
        self.edge_lengths = jnp.asarray(lengths)
        permittivity = jnp.zeros(count)
        ni, tau_n, tau_p = jnp.ones(count), jnp.ones(count), jnp.ones(count)
        semiconductor = np.zeros(count, dtype=bool)
        legacy_band_reference = None
        energy_reference = None
        explicit_bands = False
        legacy_bands = False
        for model, mask in zip(models, masks, strict=True):
            permittivity = jnp.where(mask, model.permittivity, permittivity)
            if not isinstance(model, SemiconductorMaterial):
                continue
            if (
                not model.temperature_range[0]
                <= float(self.temperature)
                <= model.temperature_range[1]
            ):
                raise ValueError(
                    f"Device temperature is outside material {model.name!r} validity."
                )
            intrinsic = model.intrinsic_density_at(self.temperature)
            if not np.isfinite(float(intrinsic)) or float(intrinsic) <= 0:
                raise ValueError(
                    "Resolved intrinsic density must remain finite and positive."
                )
            if model.thermodynamics is None:
                legacy_bands = True
                current_band = (
                    float(intrinsic),
                    float(model.band_gap),
                    float(model.electron_affinity),
                )
                if legacy_band_reference is not None and not np.allclose(
                    current_band, legacy_band_reference, rtol=1e-12, atol=0
                ):
                    raise ValueError(
                        "Unlike phenomenological intrinsic-density materials have no "
                        "shared energy reference; use explicit BandThermodynamics."
                    )
                legacy_band_reference = current_band
            else:
                explicit_bands = True
                reference = model.thermodynamics.energy_reference
                if energy_reference is not None and reference != energy_reference:
                    raise ValueError(
                        "Every explicit semiconductor band must use one electronic energy reference."
                    )
                energy_reference = reference
            ni = jnp.where(mask, intrinsic, ni)
            tau_n = jnp.where(mask, model.electron_lifetime, tau_n)
            tau_p = jnp.where(mask, model.hole_lifetime, tau_p)
            semiconductor |= mask
        if explicit_bands and legacy_bands:
            raise ValueError(
                "Explicit aligned bands cannot be mixed with reference-free intrinsic-density bands."
            )
        self.permittivity, self.intrinsic_density = permittivity, ni
        self.electron_lifetime, self.hole_lifetime = tau_n, tau_p
        self.semiconductor_mask = jnp.asarray(semiconductor)
        donors = _resolve_dopants(
            support,
            _nodal_density(donor_density, density_unit, count, "donor density"),
            donor_attributes,
            "donor",
        )
        acceptors = _resolve_dopants(
            support,
            _nodal_density(acceptor_density, density_unit, count, "acceptor density"),
            acceptor_attributes,
            "acceptor",
        )
        if np.any(np.asarray(donors + acceptors)[~semiconductor] != 0):
            raise ValueError("Dielectric nodes cannot carry semiconductor dopants.")
        self.donor_density, self.acceptor_density = donors, acceptors
        self.electron_mobility, self.hole_mobility = self._mobilities(donors + acceptors)
        self.electrothermal, self.carrier_energy = electrothermal, carrier_energy
        self.interfaces, self.traps = interfaces, traps
        self.thermal_ports, self.tunneling = thermal_ports, tunneling
        self.ionization = ionization

        edge_count = tail.size
        interface_edges = tuple(interface.edge for interface in interfaces)
        if len(set(interface_edges)) != len(interface_edges) or any(
            edge >= edge_count for edge in interface_edges
        ):
            raise ValueError(
                "Material interfaces must bind distinct in-range support edges."
            )
        self.interface_nodes = tuple(
            (int(tail[interface.edge]), int(head[interface.edge]))
            for interface in interfaces
        )
        self.interface_materials = tuple(
            (int(material_index[left]), int(material_index[right]))
            for left, right in self.interface_nodes
        )
        for interface, material_pair in zip(
            interfaces, self.interface_materials, strict=True
        ):
            if interface.electron_law is None:
                continue
            left_material, right_material = (
                models[material_pair[0]],
                models[material_pair[1]],
            )
            if (
                not isinstance(left_material, SemiconductorMaterial)
                or not isinstance(right_material, SemiconductorMaterial)
                or left_material.thermodynamics is None
                or right_material.thermodynamics is None
            ):
                raise ValueError(
                    "Thermionic traces require explicit aligned semiconductor bands on both sides."
                )
            references = (
                left_material.thermodynamics.energy_reference,
                right_material.thermodynamics.energy_reference,
                interface.electron_law.energy_reference,
                interface.hole_law.energy_reference,
            )
            if len(set(references)) != 1:
                raise ValueError(
                    "Thermionic carrier barriers and both materials require one energy reference."
                )

        for binding in traps:
            if binding.trap.population_kind == "bulk":
                if binding.location >= count or not semiconductor[binding.location]:
                    raise ValueError(
                        "A bulk trap must bind an in-range semiconductor node."
                    )
                material_index_for_trap = int(material_index[binding.location])
            else:
                if binding.location >= len(interfaces):
                    raise ValueError(
                        "A surface trap must bind an existing material interface."
                    )
                material_index_for_trap = self.interface_materials[binding.location][0]
            trap_material = models[material_index_for_trap]
            if (
                not isinstance(trap_material, SemiconductorMaterial)
                or trap_material.thermodynamics is None
                or binding.trap.energy_reference
                != trap_material.thermodynamics.energy_reference
            ):
                raise ValueError(
                    "Trap levels require the adjacent explicit semiconductor energy reference."
                )
            if not (
                float(binding.trap.temperature_range[0])
                <= float(self.temperature)
                <= float(binding.trap.temperature_range[1])
            ):
                raise ValueError(
                    "Device temperature is outside a bound trap model domain."
                )

        for port in thermal_ports:
            if port.node >= count or not np.asarray(support.boundary_mask)[port.node]:
                raise ValueError("Thermal ports must bind in-range boundary nodes.")
        if len({port.name for port in thermal_ports}) != len(thermal_ports):
            raise ValueError("Thermal port names must be unique.")

        for channel in tunneling:
            if channel.path.node_count != count:
                raise ValueError(
                    "A tunneling path must be prepared for this device node count."
                )
            if (
                energy_reference is None
                or channel.path.barrier.energy_reference != energy_reference
            ):
                raise ValueError(
                    "Tunneling barriers require the device electronic energy reference."
                )

        capacities = jnp.zeros((count,), dtype=self.temperature.dtype)
        if electrothermal:
            if not explicit_bands:
                raise ValueError(
                    "Electrothermal storage requires explicit DOS band thermodynamics."
                )
            for index, model in enumerate(models):
                if model.lattice_heat_capacity is None:
                    raise ValueError(
                        "Every electrothermal material requires lattice heat capacity."
                    )
                if not bool(
                    model.lattice_heat_capacity.evaluate(self.temperature).successful
                ):
                    raise ValueError(
                        "Device temperature is outside a lattice heat-capacity domain."
                    )
                nodes = jnp.asarray(self.material_nodes[index], dtype=jnp.int32)
                capacities = capacities.at[nodes].set(
                    model.lattice_heat_capacity.volumetric_heat_capacity
                )
                if (
                    isinstance(model, SemiconductorMaterial)
                    and carrier_energy
                    and (
                        model.electron_energy_transport is None
                        or model.hole_energy_transport is None
                        or model.electron_energy_relaxation is None
                        or model.hole_energy_relaxation is None
                    )
                ):
                    raise ValueError(
                        "Carrier-energy evolution requires transport and relaxation for both carriers."
                    )

        self.layout = SemiconductorStateLayout(
            count,
            electrothermal=electrothermal,
            carrier_energy=carrier_energy,
            interfaces=interfaces,
            traps=traps,
        )
        count_reference = support.volumes * jnp.maximum(ni, donors + acceptors)
        carrier_energy_scale = BOLTZMANN_CONSTANT_SI * self.temperature * count_reference
        lattice_energy_scale = (
            support.volumes * capacities * self.temperature
            if electrothermal
            else jnp.zeros_like(carrier_energy_scale)
        )
        self.energy_scale = jnp.maximum(carrier_energy_scale, lattice_energy_scale)
        if not all(
            isinstance(contact, (OhmicContact, GateContact)) for contact in contacts
        ):
            raise TypeError("contacts must contain OhmicContact or GateContact values.")
        names = tuple(contact.name for contact in contacts)
        if len(names) != len(set(names)):
            raise ValueError(
                "Terminal names must be unique; combine same-terminal patches in a native scope."
            )
        terminal = np.full(count, -1, dtype=np.int32)
        ohmic, potential = np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)
        offsets = jnp.zeros(count)
        neutrality = self.neutrality_potential(self.temperature)
        for index, contact in enumerate(contacts):
            mask = support.resolve_scope(contact.region.scope)
            if np.any(mask & ~np.asarray(support.boundary_mask)):
                raise ValueError("Contacts may select boundary nodes only.")
            if np.any(mask & potential):
                raise ValueError(
                    "Overlapping terminal contacts, including shared corners, are ambiguous."
                )
            terminal[mask] = index
            potential |= mask
            if isinstance(contact, OhmicContact):
                if np.any(mask & ~semiconductor):
                    raise ValueError("Ohmic contacts require semiconductor nodes.")
                ohmic |= mask
                offsets = jnp.where(mask, neutrality, offsets)
            else:
                if np.any(mask & semiconductor):
                    raise ValueError(
                        "Insulating gate contacts require explicit dielectric nodes."
                    )
                offsets = jnp.where(mask, contact.potential_offset, offsets)
        _require_pins(support, semiconductor, potential, ohmic)
        self.terminal_names = names
        self.terminal_index, self.ohmic_mask = jnp.asarray(terminal), jnp.asarray(ohmic)
        self.potential_mask, self.contact_potential_offset = (
            jnp.asarray(potential),
            offsets,
        )

    @property
    def thermal_voltage(self) -> Array:
        return BOLTZMANN_CONSTANT_SI * self.temperature / ELEMENTARY_CHARGE_SI

    def neutrality_potential(
        self,
        temperature,
        donor_density=None,
        acceptor_density=None,
    ) -> Array:
        """Potential in V making the local common electronic Fermi energy zero."""
        temperatures = jnp.broadcast_to(
            jnp.asarray(temperature), self.donor_density.shape
        )
        donors = (
            self.donor_density if donor_density is None else jnp.asarray(donor_density)
        )
        acceptors = (
            self.acceptor_density
            if acceptor_density is None
            else jnp.asarray(acceptor_density)
        )
        result = jnp.zeros_like(temperatures)
        for index, model in enumerate(self.material_models):
            if not isinstance(model, SemiconductorMaterial):
                continue
            nodes = jnp.asarray(self.material_nodes[index], dtype=jnp.int32)
            local_temperature = temperatures[nodes]
            if model.thermodynamics is None:
                intrinsic = model.intrinsic_density_at(local_temperature)
                voltage = (
                    BOLTZMANN_CONSTANT_SI
                    * local_temperature
                    / ELEMENTARY_CHARGE_SI
                    * jnp.arcsinh((donors[nodes] - acceptors[nodes]) / (2 * intrinsic))
                )
            else:
                fermi = model.thermodynamics.equilibrium_fermi_energy(
                    jnp.zeros_like(local_temperature),
                    local_temperature,
                    donors[nodes],
                    acceptors[nodes],
                    ionization=model.incomplete_ionization,
                )
                voltage = fermi / ELEMENTARY_CHARGE_SI
            result = result.at[nodes].set(voltage)
        return result

    @property
    def density_scale(self) -> Array:
        return jnp.max(
            jnp.maximum(
                self.intrinsic_density, self.donor_density + self.acceptor_density
            )
        )

    @property
    def length_scale(self) -> Array:
        return jnp.max(
            jnp.max(self.support.positions, axis=0)
            - jnp.min(self.support.positions, axis=0)
        )

    def _mobilities(self, total_density):
        electrons, holes = jnp.zeros_like(total_density), jnp.zeros_like(total_density)
        for index, model in enumerate(self.material_models):
            if isinstance(model, SemiconductorMaterial):
                mask = self.material_index == index
                electrons = jnp.where(
                    mask, model.electron_mobility(total_density), electrons
                )
                holes = jnp.where(mask, model.hole_mobility(total_density), holes)
        return electrons, holes

    def with_doping(
        self, donor_density: ArrayLike, acceptor_density: ArrayLike, /
    ) -> DevicePlan:
        """Pure SI nodal reclosure for conservative transfer and sensitivities.

        Caller must retain the admitted shape, nonnegative finite densities and
        zero dielectric doping. This kernel performs no host validation and is
        differentiable/JIT-safe. Mobility and ohmic neutrality are recomputed.
        """
        donors, acceptors = jnp.asarray(donor_density), jnp.asarray(acceptor_density)
        electrons, holes = self._mobilities(donors + acceptors)
        neutrality = self.neutrality_potential(self.temperature, donors, acceptors)
        offsets = jnp.where(self.ohmic_mask, neutrality, self.contact_potential_offset)
        count_reference = self.support.volumes * jnp.maximum(
            self.intrinsic_density, donors + acceptors
        )
        carrier_scale = BOLTZMANN_CONSTANT_SI * self.temperature * count_reference
        energy_scale = jnp.maximum(self.energy_scale, carrier_scale)
        return eqx.tree_at(
            lambda plan: (
                plan.donor_density,
                plan.acceptor_density,
                plan.electron_mobility,
                plan.hole_mobility,
                plan.contact_potential_offset,
                plan.energy_scale,
            ),
            self,
            (donors, acceptors, electrons, holes, offsets, energy_scale),
        )

    def equilibrium_coordinates(self) -> Array:
        """Charge-neutral seed with common Fermi energy and reference temperatures."""
        potential = self.neutrality_potential(self.temperature) / self.thermal_voltage
        potential = jnp.where(
            self.potential_mask,
            self.contact_potential_offset / self.thermal_voltage,
            potential,
        )
        zeros = jnp.zeros_like(potential)
        result = self.layout.pack(
            potential=potential,
            electron=zeros,
            hole=zeros,
        )
        for index, binding in enumerate(self.traps):
            occupancy = binding.initial_occupancy
            result = self.layout.set(
                result,
                f"trap_{index}",
                jnp.log(occupancy) - jnp.log1p(-occupancy),
            )
        return result


__all__ = [
    "DevicePlan",
    "GateContact",
    "MaterialBinding",
    "OhmicContact",
]
