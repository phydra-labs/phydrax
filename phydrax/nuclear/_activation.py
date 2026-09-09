#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-topology activation, transmutation, and decay evolution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    ArraySpace,
    matrix_exponential_action,
    matrix_phi1_action,
    MatrixFunctionPolicy,
)
from ..sparse import EdgeRelation, SparseCoordinateOperator
from ._energy import EnergyGroupStructure
from ._identity import NuclideKey
from ._provenance import NuclearDataProvenance


AVOGADRO_PER_MOL = 6.02214076e23


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be non-empty canonical text.")
    return normalized


@dataclass(frozen=True, slots=True)
class InventoryTransition:
    """One decay or flux-driven transition on a declared nuclide closure."""

    name: str
    parent: NuclideKey
    products: tuple[tuple[NuclideKey, float], ...]
    decay_rate_s: float
    microscopic_cross_section_m2: np.ndarray
    released_energy_j: float
    incident_baryon_number: float
    incident_charge_number: float
    escaped_baryon_number: float
    escaped_charge_number: float
    escaped_lepton_number: float
    data: NuclearDataProvenance
    transition_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = _text(self.name, "name")
        if not isinstance(self.parent, NuclideKey):
            raise TypeError("parent must be NuclideKey.")
        products = tuple(self.products)
        if not products:
            raise ValueError(
                "Inventory transitions require at least one tracked product."
            )
        normalized_products = []
        for nuclide, product_yield in products:
            if not isinstance(nuclide, NuclideKey):
                raise TypeError(
                    "Inventory transition products must be NuclideKey values."
                )
            value = float(product_yield)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "Inventory product yields must be finite and nonnegative."
                )
            normalized_products.append((nuclide, value))
        if len({value.nuclide_id for value, _ in normalized_products}) != len(
            normalized_products
        ):
            raise ValueError("Inventory transition product nuclides must be unique.")
        decay = float(self.decay_rate_s)
        cross_section = np.array(
            self.microscopic_cross_section_m2, dtype=np.float64, copy=True
        )
        energy = float(self.released_energy_j)
        ledger = tuple(
            float(value)
            for value in (
                self.incident_baryon_number,
                self.incident_charge_number,
                self.escaped_baryon_number,
                self.escaped_charge_number,
                self.escaped_lepton_number,
            )
        )
        if (
            not math.isfinite(decay)
            or decay < 0.0
            or cross_section.ndim != 1
            or np.any(~np.isfinite(cross_section))
            or np.any(cross_section < 0.0)
            or not math.isfinite(energy)
            or energy < 0.0
            or any(not math.isfinite(value) for value in ledger)
        ):
            raise ValueError(
                "Inventory transition rates, energy, and ledger must be finite and nonnegative where physical."
            )
        has_decay = decay > 0.0
        has_flux = bool(np.any(cross_section > 0.0))
        if has_decay == has_flux:
            raise ValueError("A transition must be exactly one of decay or flux driven.")
        if not isinstance(self.data, NuclearDataProvenance):
            raise TypeError("data must be NuclearDataProvenance.")
        product_baryon = sum(
            value.mass_number * product_yield
            for value, product_yield in normalized_products
        )
        product_charge = sum(
            value.proton_number * product_yield
            for value, product_yield in normalized_products
        )
        baryon_defect = product_baryon + ledger[2] - self.parent.mass_number - ledger[0]
        charge_defect = product_charge + ledger[3] - self.parent.proton_number - ledger[1]
        tolerance = 1.0e-12
        if (
            abs(baryon_defect) > tolerance
            or abs(charge_defect) > tolerance
            or abs(ledger[4]) > tolerance
        ):
            raise ValueError(
                "Inventory transition does not close its declared baryon, charge, and lepton ledger."
            )
        cross_section.setflags(write=False)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "products", tuple(normalized_products))
        object.__setattr__(self, "decay_rate_s", decay)
        object.__setattr__(self, "microscopic_cross_section_m2", cross_section)
        object.__setattr__(self, "released_energy_j", energy)
        (
            incident_baryon,
            incident_charge,
            escaped_baryon,
            escaped_charge,
            escaped_lepton,
        ) = ledger
        object.__setattr__(self, "incident_baryon_number", incident_baryon)
        object.__setattr__(self, "incident_charge_number", incident_charge)
        object.__setattr__(self, "escaped_baryon_number", escaped_baryon)
        object.__setattr__(self, "escaped_charge_number", escaped_charge)
        object.__setattr__(self, "escaped_lepton_number", escaped_lepton)
        object.__setattr__(
            self,
            "transition_id",
            canonical_fingerprint(
                {
                    "kind": "inventory-transition",
                    "name": name,
                    "parent": self.parent.nuclide_id,
                    "products": [
                        [value.nuclide_id, product_yield]
                        for value, product_yield in normalized_products
                    ],
                    "decay_rate_s": decay,
                    "cross_section_m2": array_tree_fingerprint(cross_section),
                    "released_energy_j": energy,
                    "ledger": list(ledger),
                    "data": self.data.provenance_id,
                }
            ),
        )

    @property
    def flux_driven(self) -> bool:
        return bool(np.any(self.microscopic_cross_section_m2 > 0.0))


class NuclideInventory(StrictModule):
    amounts_mol: Array
    time_s: Array
    nuclide_ids: tuple[str, ...] = eqx.field(static=True)
    network_id: str = eqx.field(static=True)

    def __init__(
        self,
        amounts_mol: ArrayLike,
        time_s: ArrayLike,
        nuclide_ids: tuple[str, ...],
        network_id: str,
        /,
    ):
        amounts = jnp.asarray(amounts_mol, dtype=jnp.float64)
        time = jnp.asarray(time_s, dtype=amounts.dtype)
        if amounts.ndim != 1 or time.shape != ():
            raise ValueError("Inventory amounts must be rank one and time scalar.")
        if amounts.shape != (len(nuclide_ids),):
            raise ValueError("Inventory amount axis and nuclide identities disagree.")
        self.amounts_mol = amounts
        self.time_s = time
        self.nuclide_ids = tuple(nuclide_ids)
        self.network_id = _text(network_id, "network_id")


class ActivationLedger(StrictModule):
    initial_amount_mol: Array
    candidate_amount_mol: Array
    external_source_amount_mol: Array
    tracked_amount_change_mol: Array
    minimum_amount_mol: Array
    finite: Array
    nonnegative: Array


class ActivationStepResult(StrictModule):
    candidate: NuclideInventory
    accepted: NuclideInventory
    activity_bq: Array
    decay_heat_w: Array
    transition_rates_s: Array
    ledger: ActivationLedger
    exponential_error: Array
    source_error: Array
    finite: Array
    sensitivity_valid: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class ActivationNetworkPlan:
    nuclides: tuple[NuclideKey, ...]
    energy_groups: EnergyGroupStructure
    transitions: tuple[InventoryTransition, ...]
    error_tolerance: float = 1.0e-10
    network_id: str = field(init=False)

    def __post_init__(self) -> None:
        nuclides = tuple(self.nuclides)
        transitions = tuple(self.transitions)
        if not nuclides or any(not isinstance(value, NuclideKey) for value in nuclides):
            raise TypeError("nuclides must contain NuclideKey values.")
        identifiers = tuple(value.nuclide_id for value in nuclides)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Activation nuclides must be unique.")
        if not isinstance(self.energy_groups, EnergyGroupStructure):
            raise TypeError("energy_groups must be EnergyGroupStructure.")
        if not transitions or any(
            not isinstance(value, InventoryTransition) for value in transitions
        ):
            raise TypeError("transitions must contain InventoryTransition values.")
        if len({value.transition_id for value in transitions}) != len(transitions):
            raise ValueError("Activation transitions must be unique.")
        closure = set(identifiers)
        for transition in transitions:
            if transition.parent.nuclide_id not in closure or any(
                value.nuclide_id not in closure for value, _ in transition.products
            ):
                raise ValueError(
                    "Every activation transition nuclide must belong to the fixed closure."
                )
            if transition.microscopic_cross_section_m2.shape != (
                self.energy_groups.group_count,
            ):
                raise ValueError(
                    "Every transition cross section must match the energy groups."
                )
        tolerance = float(self.error_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("error_tolerance must be finite and positive.")
        object.__setattr__(self, "nuclides", nuclides)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "error_tolerance", tolerance)
        object.__setattr__(
            self,
            "network_id",
            canonical_fingerprint(
                {
                    "kind": "activation-network-plan",
                    "nuclides": list(identifiers),
                    "groups": self.energy_groups.group_id,
                    "transitions": [value.transition_id for value in transitions],
                    "error_tolerance": tolerance,
                }
            ),
        )

    def prepare(self) -> PreparedActivationNetwork:
        index = {value.nuclide_id: i for i, value in enumerate(self.nuclides)}
        route_source = []
        route_target = []
        route_transition = []
        route_multiplier = []
        for transition_index, transition in enumerate(self.transitions):
            parent = index[transition.parent.nuclide_id]
            route_source.append(parent)
            route_target.append(parent)
            route_transition.append(transition_index)
            route_multiplier.append(-1.0)
            for product, product_yield in transition.products:
                route_source.append(parent)
                route_target.append(index[product.nuclide_id])
                route_transition.append(transition_index)
                route_multiplier.append(product_yield)
        decay = np.asarray([value.decay_rate_s for value in self.transitions])
        cross_section = np.stack(
            [value.microscopic_cross_section_m2 for value in self.transitions]
        )
        energy = np.asarray([value.released_energy_j for value in self.transitions])
        parents = np.asarray(
            [index[value.parent.nuclide_id] for value in self.transitions], dtype=np.int32
        )
        policy = MatrixFunctionPolicy(
            "arnoldi",
            max_dimension=min(64, max(4, len(self.nuclides))),
            error_tolerance=self.error_tolerance,
        )
        return PreparedActivationNetwork(
            jnp.asarray(route_source, dtype=jnp.int32),
            jnp.asarray(route_target, dtype=jnp.int32),
            jnp.asarray(route_transition, dtype=jnp.int32),
            jnp.asarray(route_multiplier),
            jnp.asarray(decay),
            jnp.asarray(cross_section),
            jnp.asarray(energy),
            jnp.asarray(parents),
            tuple(value.nuclide_id for value in self.nuclides),
            self.energy_groups.group_id,
            self.network_id,
            policy,
        )


class PreparedActivationNetwork(StrictModule, NonTrainableState):
    route_source: Array
    route_target: Array
    route_transition: Array
    route_multiplier: Array
    decay_rates_s: Array
    microscopic_cross_sections_m2: Array
    released_energies_j: Array
    parent_indices: Array
    nuclide_ids: tuple[str, ...] = eqx.field(static=True)
    energy_group_id: str = eqx.field(static=True)
    network_id: str = eqx.field(static=True)
    matrix_function_policy: MatrixFunctionPolicy

    @property
    def nuclide_count(self) -> int:
        return len(self.nuclide_ids)

    @property
    def transition_count(self) -> int:
        return int(self.decay_rates_s.shape[0])

    def inventory(
        self, amounts_mol: ArrayLike, time_s: ArrayLike = 0.0, /
    ) -> NuclideInventory:
        return NuclideInventory(amounts_mol, time_s, self.nuclide_ids, self.network_id)

    def transition_rates(self, scalar_flux_m2_s: ArrayLike, /) -> Array:
        flux = jnp.asarray(scalar_flux_m2_s, dtype=self.decay_rates_s.dtype)
        if flux.shape != (self.microscopic_cross_sections_m2.shape[1],):
            raise ValueError("Scalar flux must match the activation energy-group axis.")
        return self.decay_rates_s + contract(
            "tg,g->t", self.microscopic_cross_sections_m2, flux
        )

    def operator(
        self, scalar_flux_m2_s: ArrayLike, /
    ) -> tuple[SparseCoordinateOperator, Array]:
        rates = self.transition_rates(scalar_flux_m2_s)
        relation = EdgeRelation(
            self.route_source,
            self.route_target,
            source_size=self.nuclide_count,
            target_size=self.nuclide_count,
        )
        space = ArraySpace((self.nuclide_count,), dtype=np.float64)
        coefficients = rates[self.route_transition] * self.route_multiplier
        operator = SparseCoordinateOperator(
            relation,
            coefficients,
            source=space,
            target=space,
            operator_id=self.network_id,
        )
        return operator, rates

    def step(
        self,
        inventory: NuclideInventory,
        scalar_flux_m2_s: ArrayLike,
        dt_s: ArrayLike,
        /,
        *,
        external_source_mol_s: ArrayLike | None = None,
    ) -> ActivationStepResult:
        if not isinstance(inventory, NuclideInventory):
            raise TypeError("inventory must be NuclideInventory.")
        if inventory.network_id != self.network_id:
            raise ValueError("Inventory and activation network identities disagree.")
        dt = jnp.asarray(dt_s, dtype=inventory.amounts_mol.dtype)
        if dt.shape != ():
            raise ValueError("dt_s must be scalar.")
        source = (
            jnp.zeros_like(inventory.amounts_mol)
            if external_source_mol_s is None
            else jnp.asarray(external_source_mol_s, dtype=inventory.amounts_mol.dtype)
        )
        if source.shape != inventory.amounts_mol.shape:
            raise ValueError("external_source_mol_s must match the inventory axis.")
        flux = jnp.asarray(scalar_flux_m2_s, dtype=inventory.amounts_mol.dtype)
        finite_inputs = (
            jnp.all(jnp.isfinite(inventory.amounts_mol))
            & jnp.isfinite(dt)
            & jnp.all(jnp.isfinite(flux))
            & jnp.all(jnp.isfinite(source))
        )
        domain_valid = (
            finite_inputs
            & (dt > 0.0)
            & jnp.all(inventory.amounts_mol >= 0.0)
            & jnp.all(flux >= 0.0)
            & jnp.all(source >= 0.0)
        )
        safe_dt = jnp.where(domain_valid, dt, 1.0)
        safe_flux = jnp.where(jnp.isfinite(flux) & (flux >= 0.0), flux, 0.0)
        safe_source = jnp.where(jnp.isfinite(source) & (source >= 0.0), source, 0.0)
        operator, rates = self.operator(safe_flux)
        exponential = matrix_exponential_action(
            operator,
            inventory.amounts_mol,
            safe_dt,
            policy=self.matrix_function_policy,
        )
        source_action = matrix_phi1_action(
            operator,
            safe_source,
            safe_dt,
            policy=self.matrix_function_policy,
        )
        candidate_amounts = exponential.value + safe_dt * source_action.value
        candidate = self.inventory(candidate_amounts, inventory.time_s + safe_dt)
        minimum = jnp.min(candidate_amounts)
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(inventory.amounts_mol)))
        negativity_tolerance = 512.0 * jnp.finfo(candidate_amounts.dtype).eps * scale
        nonnegative = minimum >= -negativity_tolerance
        finite = jnp.all(jnp.isfinite(candidate_amounts))
        successful = (
            domain_valid
            & exponential.converged
            & source_action.converged
            & finite
            & nonnegative
        )
        accepted_amounts = jnp.where(successful, candidate_amounts, inventory.amounts_mol)
        accepted_time = jnp.where(successful, inventory.time_s + dt, inventory.time_s)
        accepted = self.inventory(accepted_amounts, accepted_time)
        decay_by_nuclide = jnp.zeros((self.nuclide_count,), dtype=candidate_amounts.dtype)
        decay_by_nuclide = decay_by_nuclide.at[self.parent_indices].add(
            self.decay_rates_s
        )
        activity = AVOGADRO_PER_MOL * candidate_amounts * decay_by_nuclide
        parent_amounts = candidate_amounts[self.parent_indices]
        decay_heat = jnp.sum(
            AVOGADRO_PER_MOL
            * parent_amounts
            * self.decay_rates_s
            * self.released_energies_j
        )
        tracked_change = candidate_amounts - inventory.amounts_mol
        ledger = ActivationLedger(
            inventory.amounts_mol,
            candidate_amounts,
            safe_dt * safe_source,
            tracked_change,
            minimum,
            finite,
            nonnegative,
        )
        return ActivationStepResult(
            candidate,
            accepted,
            activity,
            decay_heat,
            rates,
            ledger,
            exponential.error_estimate,
            source_action.error_estimate,
            finite,
            successful,
            successful,
        )


class ActivationScheduleResult(StrictModule):
    amounts_mol: Array
    times_s: Array
    successful_steps: Array
    valid_prefix: Array
    final_inventory: NuclideInventory


@dataclass(frozen=True, slots=True)
class IrradiationSchedulePlan:
    durations_s: np.ndarray
    scalar_flux_m2_s: np.ndarray
    schedule_id: str = field(init=False)

    def __post_init__(self) -> None:
        durations = np.array(self.durations_s, dtype=np.float64, copy=True)
        flux = np.array(self.scalar_flux_m2_s, dtype=np.float64, copy=True)
        if durations.ndim != 1 or durations.size < 1:
            raise ValueError("durations_s must be a non-empty rank-one array.")
        if flux.ndim != 2 or flux.shape[0] != durations.size:
            raise ValueError("scalar_flux_m2_s must have shape (segment, group).")
        if (
            np.any(~np.isfinite(durations))
            or np.any(durations <= 0.0)
            or np.any(~np.isfinite(flux))
            or np.any(flux < 0.0)
        ):
            raise ValueError("Irradiation schedule durations and fluxes are invalid.")
        durations.setflags(write=False)
        flux.setflags(write=False)
        object.__setattr__(self, "durations_s", durations)
        object.__setattr__(self, "scalar_flux_m2_s", flux)
        object.__setattr__(
            self,
            "schedule_id",
            canonical_fingerprint(
                {
                    "kind": "irradiation-schedule",
                    "durations_s": array_tree_fingerprint(durations),
                    "scalar_flux_m2_s": array_tree_fingerprint(flux),
                }
            ),
        )

    def run(
        self,
        network: PreparedActivationNetwork,
        inventory: NuclideInventory,
        /,
    ) -> ActivationScheduleResult:
        if not isinstance(network, PreparedActivationNetwork):
            raise TypeError("network must be PreparedActivationNetwork.")
        current = inventory
        amounts = [inventory.amounts_mol]
        times = [inventory.time_s]
        successful = []
        chain_valid = jnp.asarray(True)
        for duration, flux in zip(self.durations_s, self.scalar_flux_m2_s, strict=True):
            effective_duration = jnp.where(chain_valid, duration, -duration)
            result = network.step(current, flux, effective_duration)
            chain_valid = chain_valid & result.successful
            current = result.accepted
            amounts.append(current.amounts_mol)
            times.append(current.time_s)
            successful.append(chain_valid)
        success = jnp.stack(successful)
        prefix = jnp.cumprod(success.astype(jnp.int32)).astype(bool)
        return ActivationScheduleResult(
            jnp.stack(amounts),
            jnp.stack(times),
            success,
            prefix,
            current,
        )


__all__ = [
    "AVOGADRO_PER_MOL",
    "ActivationLedger",
    "ActivationNetworkPlan",
    "ActivationScheduleResult",
    "ActivationStepResult",
    "InventoryTransition",
    "IrradiationSchedulePlan",
    "NuclideInventory",
    "PreparedActivationNetwork",
]
