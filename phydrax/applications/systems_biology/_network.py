#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compartmental stoichiometric plans and fixed-shape simulation runtimes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum, IntEnum
from math import factorial, isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations import PreparedChemicalMechanism
from ...stochastic import AbstractJumpProcess


ApproximationKind: TypeAlias = Literal["deterministic", "cle"]


class StoichiometricStatus(IntEnum):
    """Fail-closed compartmental network evaluation status."""

    SUCCESS = 0
    INVALID_STATE = 1
    INVALID_PARAMETERS = 2
    NONFINITE = 3


_STOICHIOMETRIC_SUCCESS = StoichiometricStatus.SUCCESS
_STOICHIOMETRIC_INVALID_STATE = StoichiometricStatus.INVALID_STATE
_STOICHIOMETRIC_INVALID_PARAMETERS = StoichiometricStatus.INVALID_PARAMETERS
_STOICHIOMETRIC_NONFINITE = StoichiometricStatus.NONFINITE
_MASS_ACTION = 0
_HILL = 1
_MICHAELIS_MENTEN = 2
_PROMOTER_TRANSITION = 3


def _name(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{owner} must be a non-empty, trimmed string.")
    return value


def _path_name(value: str, owner: str, /) -> str:
    name = _name(value, owner)
    if "." in name:
        raise ValueError(f"{owner} must not contain the reserved '.' delimiter.")
    return name


def _canonical_unit(*factors: tuple[str, int]) -> str:
    """Combine opaque unit symbols into a deterministic reduced product."""
    exponents: dict[str, int] = {}
    for unit, exponent in factors:
        exponents[unit] = exponents.get(unit, 0) + exponent
    terms = []
    for unit in sorted(exponents):
        exponent = exponents[unit]
        if exponent == 0:
            continue
        label = unit if unit.replace("_", "").isalnum() else f"({unit})"
        terms.append(label if exponent == 1 else f"{label}^{exponent}")
    return "dimensionless" if not terms else "*".join(terms)


def _semantic_payload(value: object, /) -> object:
    """Encode all dynamic and static dataclass fields for an exact host identity."""
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _semantic_payload(value.value),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return _semantic_payload(value.item())
    if isinstance(value, (jax.Array, np.ndarray)):
        return {"array": array_tree_fingerprint(np.asarray(value))}
    if isinstance(value, (tuple, list)):
        return [_semantic_payload(item) for item in value]
    if isinstance(value, Mapping):
        items = [
            (_semantic_payload(key), _semantic_payload(item))
            for key, item in value.items()
        ]
        return {"mapping": sorted(items, key=lambda item: canonical_fingerprint(item[0]))}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": [
                (
                    field.name,
                    _semantic_payload(object.__getattribute__(value, field.name)),
                )
                for field in fields(value)
            ],
        }
    raise TypeError(
        f"Unsupported thermochemical identity value {type(value).__qualname__}."
    )


def _scalar(value: ArrayLike, owner: str, /, *, positive: bool = False) -> Array:
    raw = jnp.asarray(value)
    if raw.dtype == jnp.bool_:
        raise TypeError(f"{owner} must be numeric, not boolean.")
    array = raw.astype(float)
    if array.shape != ():
        raise ValueError(f"{owner} must be scalar.")
    host = float(array)
    if not isfinite(host) or (positive and host <= 0.0) or (not positive and host < 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{owner} must be finite and {qualifier}.")
    return array


def _integer_mapping(
    values: Mapping[str, int], owner: str, /
) -> tuple[tuple[str, int], ...]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{owner} must be a mapping.")
    normalized: list[tuple[str, int]] = []
    for species, coefficient in values.items():
        species_name = _name(species, f"{owner} species")
        if isinstance(coefficient, bool) or not isinstance(
            coefficient, (int, np.integer)
        ):
            raise ValueError(f"{owner} coefficients must be integers.")
        integer = int(coefficient)
        if integer == 0:
            raise ValueError(f"{owner} coefficients must be nonzero.")
        normalized.append((species_name, integer))
    if not normalized:
        raise ValueError(f"{owner} must be non-empty.")
    if len({species for species, _ in normalized}) != len(normalized):
        raise ValueError(f"{owner} species must be unique.")
    return tuple(sorted(normalized))


def _order_mapping(
    values: Mapping[str, int], owner: str, /
) -> tuple[tuple[str, int], ...]:
    normalized = _integer_mapping(values, owner)
    if any(order <= 0 for _, order in normalized):
        raise ValueError(f"{owner} orders must be positive.")
    return normalized


class CompartmentSpec(StrictModule, NonTrainableState):
    """Named well-mixed compartment with a positive system measure."""

    name: str = eqx.field(static=True)
    measure: Array
    unit: str = eqx.field(static=True)

    def __init__(self, name: str, measure: ArrayLike, /, *, unit: str = "volume"):
        self.name = _path_name(name, "Compartment name")
        self.measure = _scalar(measure, "Compartment measure", positive=True)
        self.unit = _name(unit, "Compartment measure unit")


class SpeciesSpec(StrictModule, NonTrainableState):
    """Biological species identity, exchange type, and reservoir semantics."""

    name: str = eqx.field(static=True)
    compartment: str = eqx.field(static=True)
    reservoir: bool = eqx.field(static=True)
    quantity: str = eqx.field(static=True)
    unit: str = eqx.field(static=True)
    thermochemical_name: str | None = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        compartment: str,
        /,
        *,
        reservoir: bool = False,
        quantity: str = "count",
        unit: str = "molecule",
        thermochemical_name: str | None = None,
    ):
        self.name = _path_name(name, "Species name")
        self.compartment = _name(compartment, "Species compartment")
        if not isinstance(reservoir, bool):
            raise TypeError("reservoir must be bool.")
        self.reservoir = reservoir
        self.quantity = _name(quantity, "Species quantity")
        self.unit = _name(unit, "Species unit")
        self.thermochemical_name = (
            None
            if thermochemical_name is None
            else _name(thermochemical_name, "Thermochemical species name")
        )


class MassActionPropensity(StrictModule):
    """Discrete combinatorial mass action with explicit kinetic orders."""

    rate: Array
    orders: tuple[tuple[str, int], ...] = eqx.field(static=True)

    def __init__(self, rate: ArrayLike, orders: Mapping[str, int], /):
        self.rate = _scalar(rate, "Mass-action rate")
        self.orders = _order_mapping(orders, "Mass-action orders")


class HillPropensity(StrictModule):
    """Activating or repressing Hill response driven by one concentration."""

    maximum_rate: Array
    half_saturation: Array
    coefficient: Array
    basal_rate: Array
    regulator: str = eqx.field(static=True)
    repression: bool = eqx.field(static=True)

    def __init__(
        self,
        maximum_rate: ArrayLike,
        half_saturation: ArrayLike,
        coefficient: ArrayLike,
        regulator: str,
        /,
        *,
        basal_rate: ArrayLike = 0.0,
        repression: bool = False,
    ):
        self.maximum_rate = _scalar(maximum_rate, "Hill maximum rate")
        self.half_saturation = _scalar(
            half_saturation, "Hill half-saturation", positive=True
        )
        self.coefficient = _scalar(coefficient, "Hill coefficient", positive=True)
        self.basal_rate = _scalar(basal_rate, "Hill basal rate")
        self.regulator = _name(regulator, "Hill regulator")
        if not isinstance(repression, bool):
            raise TypeError("repression must be bool.")
        self.repression = repression


class MichaelisMentenPropensity(StrictModule):
    """Saturating Michaelis--Menten process driven by one substrate."""

    maximum_rate: Array
    michaelis_constant: Array
    substrate: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_rate: ArrayLike,
        michaelis_constant: ArrayLike,
        substrate: str,
        /,
    ):
        self.maximum_rate = _scalar(maximum_rate, "Michaelis--Menten maximum rate")
        self.michaelis_constant = _scalar(
            michaelis_constant, "Michaelis constant", positive=True
        )
        self.substrate = _name(substrate, "Michaelis--Menten substrate")


class PromoterTransitionPropensity(StrictModule):
    """First-order transition intensity gated by a discrete promoter state."""

    rate: Array
    source: str = eqx.field(static=True)

    def __init__(self, rate: ArrayLike, source: str, /):
        self.rate = _scalar(rate, "Promoter-transition rate")
        self.source = _name(source, "Promoter-transition source")


PropensitySpec: TypeAlias = (
    MassActionPropensity
    | HillPropensity
    | MichaelisMentenPropensity
    | PromoterTransitionPropensity
)


class StoichiometricProcessSpec(StrictModule):
    """Sparse integer state change paired with one biological propensity law."""

    name: str = eqx.field(static=True)
    stoichiometry: tuple[tuple[str, int], ...] = eqx.field(static=True)
    propensity: PropensitySpec
    thermochemical_reaction: str | None = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        stoichiometry: Mapping[str, int],
        propensity: PropensitySpec,
        /,
        *,
        thermochemical_reaction: str | None = None,
    ):
        if not isinstance(
            propensity,
            (
                MassActionPropensity,
                HillPropensity,
                MichaelisMentenPropensity,
                PromoterTransitionPropensity,
            ),
        ):
            raise TypeError("propensity must be a supported PropensitySpec.")
        self.name = _path_name(name, "Process name")
        self.stoichiometry = _integer_mapping(stoichiometry, "Process stoichiometry")
        self.propensity = propensity
        self.thermochemical_reaction = (
            None
            if thermochemical_reaction is None
            else _name(thermochemical_reaction, "Thermochemical reaction name")
        )


class StoichiometricRuntime(StrictModule):
    """Invocation-specific kinetic parameter matrix for a prepared network."""

    parameters: Array

    def __init__(self, parameters: ArrayLike, /):
        raw = jnp.asarray(parameters)
        if raw.dtype == jnp.bool_:
            raise TypeError("Runtime parameters must not be boolean.")
        values = raw.astype(float)
        if values.ndim != 2 or values.shape[-1] != 4:
            raise ValueError("Runtime parameters must have shape (process_count, 4).")
        self.parameters = values


class ConservationEvidence(StrictModule):
    """Preparation-time conservation basis and structural rank evidence."""

    basis: Array
    basis_units: tuple[str, ...] = eqx.field(static=True)
    stoichiometric_rank: int = eqx.field(static=True)
    maximum_basis_residual: Array
    valid: Array
    network_id: str = eqx.field(static=True)


class StoichiometricEvaluation(StrictModule):
    """Rates, drift, boundary ledgers, and fail-closed runtime evidence."""

    propensities: Array
    drift: Array
    source_rate: Array
    sink_rate: Array
    conservation_residual: Array
    state_valid: Array
    parameter_valid: Array
    finite: Array
    successful: Array
    status: Array
    network_id: str = eqx.field(static=True)


class ApproximationEvidence(StrictModule):
    """Declared deterministic/CLE regime checks, separate from numerical validity."""

    minimum_copy_number: Array
    minimum_expected_events: Array
    copy_number_valid: Array
    event_frequency_valid: Array
    regime_valid: Array
    numerical_valid: Array
    kind: ApproximationKind = eqx.field(static=True)
    differentiable: Array


class StoichiometricStepResult(StrictModule):
    """Uncommitted fixed-shape state candidate and approximation evidence."""

    candidate: Array
    source_delta: Array
    sink_delta: Array
    evaluation: StoichiometricEvaluation
    evidence: ApproximationEvidence
    accepted: Array
    network_id: str = eqx.field(static=True)

    def commit(self, state: ArrayLike, /) -> Array:
        current = jnp.asarray(state, dtype=self.candidate.dtype)
        if current.shape != self.candidate.shape:
            raise ValueError("Committed state must match candidate shape.")
        return jnp.where(self.accepted, self.candidate, current)


class ThermochemicalInteropEvidence(StrictModule, NonTrainableState):
    """Exact species/reaction mapping to the canonical thermochemical mechanism IR."""

    species_indices: Array
    process_indices: Array
    reaction_indices: Array
    maximum_reaction_residual: Array
    compatible: Array
    network_id: str = eqx.field(static=True)
    mechanism_id: str = eqx.field(static=True)
    mechanism_content_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)


class StoichiometricNetworkPlan(StrictModule, NonTrainableState):
    """Host IR for one fixed-capacity compartmental process network."""

    name: str = eqx.field(static=True)
    compartments: tuple[CompartmentSpec, ...]
    species: tuple[SpeciesSpec, ...]
    processes: tuple[StoichiometricProcessSpec, ...]
    stoichiometry_capacity: int = eqx.field(static=True)
    time_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        compartments: Sequence[CompartmentSpec],
        species: Sequence[SpeciesSpec],
        processes: Sequence[StoichiometricProcessSpec],
        /,
        *,
        stoichiometry_capacity: int | None = None,
        time_unit: str = "s",
    ):
        name_value = _name(name, "Network name")
        time_unit_value = _name(time_unit, "Network time unit")
        compartment_values = tuple(compartments)
        species_values = tuple(species)
        process_values = tuple(processes)
        if not compartment_values or any(
            not isinstance(value, CompartmentSpec) for value in compartment_values
        ):
            raise TypeError("compartments must contain CompartmentSpec objects.")
        if not species_values or any(
            not isinstance(value, SpeciesSpec) for value in species_values
        ):
            raise TypeError("species must contain SpeciesSpec objects.")
        if not process_values or any(
            not isinstance(value, StoichiometricProcessSpec) for value in process_values
        ):
            raise TypeError("processes must contain StoichiometricProcessSpec objects.")
        compartment_names = tuple(item.name for item in compartment_values)
        species_names = tuple(item.name for item in species_values)
        process_names = tuple(item.name for item in process_values)
        if len(set(compartment_names)) != len(compartment_names):
            raise ValueError("Compartment names must be unique.")
        if len(set(species_names)) != len(species_names):
            raise ValueError("Species names must be unique.")
        if len(set(process_names)) != len(process_names):
            raise ValueError("Process names must be unique.")
        compartment_set = set(compartment_names)
        if any(item.compartment not in compartment_set for item in species_values):
            raise ValueError("Every species must reference a declared compartment.")
        known_species = set(species_names)
        species_by_name = {item.name: item for item in species_values}
        for process in process_values:
            referenced = {item for item, _ in process.stoichiometry}
            propensity = process.propensity
            if isinstance(propensity, MassActionPropensity):
                referenced.update(item for item, _ in propensity.orders)
            elif isinstance(propensity, HillPropensity):
                referenced.add(propensity.regulator)
            elif isinstance(propensity, MichaelisMentenPropensity):
                referenced.add(propensity.substrate)
            else:
                referenced.add(propensity.source)
            unknown = referenced - known_species
            if unknown:
                raise ValueError(
                    f"Process {process.name!r} references unknown species "
                    f"{sorted(unknown)!r}."
                )
            changed_types = {
                (
                    species_by_name[species_name].quantity,
                    species_by_name[species_name].unit,
                )
                for species_name, _ in process.stoichiometry
            }
            if len(changed_types) != 1:
                raise ValueError(
                    f"Process {process.name!r} changes species with incompatible "
                    "quantity or unit types."
                )
        maximum_entries = max(len(item.stoichiometry) for item in process_values)
        if stoichiometry_capacity is None:
            capacity = maximum_entries
        else:
            if isinstance(stoichiometry_capacity, bool) or not isinstance(
                stoichiometry_capacity, (int, np.integer)
            ):
                raise ValueError("stoichiometry_capacity must be an integer or None.")
            capacity = int(stoichiometry_capacity)
        if capacity <= 0 or capacity < maximum_entries:
            raise ValueError("stoichiometry_capacity must cover every sparse process.")
        identity = canonical_fingerprint(
            {
                "kind": "systems-biology-stoichiometric-plan",
                "name": name_value,
                "compartments": [
                    (item.name, float(item.measure), item.unit)
                    for item in compartment_values
                ],
                "species": [
                    (
                        item.name,
                        item.compartment,
                        item.reservoir,
                        item.quantity,
                        item.unit,
                        item.thermochemical_name,
                    )
                    for item in species_values
                ],
                "processes": [_process_payload(item) for item in process_values],
                "capacity": capacity,
                "time_unit": time_unit_value,
            }
        )
        self.name = name_value
        self.compartments = compartment_values
        self.species = species_values
        self.processes = process_values
        self.stoichiometry_capacity = capacity
        self.time_unit = time_unit_value
        self.plan_id = identity

    def prepare(self) -> PreparedStoichiometricNetwork:
        return PreparedStoichiometricNetwork(self)


class PreparedStoichiometricNetwork(StrictModule, NonTrainableState):
    """Dense device runtime prepared from sparse biological process semantics."""

    plan: StoichiometricNetworkPlan
    stoichiometry: Array
    dynamic_stoichiometry: Array
    stoichiometric_species: Array
    stoichiometric_values: Array
    stoichiometric_mask: Array
    reservoir_mask: Array
    copy_number_mask: Array
    species_measure: Array
    propensity_kind: Array
    propensity_parameters: Array
    propensity_species: Array
    propensity_orders: Array
    propensity_normalization: Array
    propensity_repression: Array
    availability: Array
    conservation: ConservationEvidence
    species_count: int = eqx.field(static=True)
    process_count: int = eqx.field(static=True)
    maximum_order: int = eqx.field(static=True)
    network_id: str = eqx.field(static=True)

    def __init__(self, plan: StoichiometricNetworkPlan, /):
        if not isinstance(plan, StoichiometricNetworkPlan):
            raise TypeError("plan must be StoichiometricNetworkPlan.")
        species_index = {item.name: index for index, item in enumerate(plan.species)}
        compartment_measure = {
            item.name: float(item.measure) for item in plan.compartments
        }
        reaction_count = len(plan.processes)
        species_count = len(plan.species)
        dense = np.zeros((reaction_count, species_count), dtype=np.int32)
        sparse_species = np.zeros(
            (reaction_count, plan.stoichiometry_capacity), dtype=np.int32
        )
        sparse_values = np.zeros_like(sparse_species)
        sparse_mask = np.zeros_like(sparse_species, dtype=bool)
        orders = np.zeros_like(dense)
        normalization = np.ones((reaction_count, species_count), dtype=float)
        kind = np.zeros(reaction_count, dtype=np.int32)
        parameters = np.zeros((reaction_count, 4), dtype=float)
        propensity_species = np.zeros(reaction_count, dtype=np.int32)
        repression = np.zeros(reaction_count, dtype=bool)
        maximum_order = 0
        for process_index, process in enumerate(plan.processes):
            for slot, (species_name, value) in enumerate(process.stoichiometry):
                index = species_index[species_name]
                dense[process_index, index] = value
                sparse_species[process_index, slot] = index
                sparse_values[process_index, slot] = value
                sparse_mask[process_index, slot] = True
            propensity = process.propensity
            if isinstance(propensity, MassActionPropensity):
                kind[process_index] = _MASS_ACTION
                for species_name, order in propensity.orders:
                    orders[process_index, species_index[species_name]] = order
                    maximum_order = max(maximum_order, order)
                normalization[process_index] = [
                    factorial(int(order)) for order in orders[process_index]
                ]
                total_order = int(np.sum(orders[process_index]))
                ordered = np.flatnonzero(orders[process_index])
                ordered_compartments = {
                    plan.species[index].compartment for index in ordered.tolist()
                }
                if total_order > 1 and len(ordered_compartments) != 1:
                    raise ValueError(
                        f"Mass-action process {process.name!r} spans compartments; "
                        "an explicit transport process is required."
                    )
                measure = (
                    1.0
                    if not ordered_compartments
                    else compartment_measure[next(iter(ordered_compartments))]
                )
                parameters[process_index] = [
                    float(propensity.rate),
                    measure,
                    float(total_order),
                    0.0,
                ]
            elif isinstance(propensity, HillPropensity):
                kind[process_index] = _HILL
                propensity_species[process_index] = species_index[propensity.regulator]
                parameters[process_index] = [
                    float(propensity.maximum_rate),
                    float(propensity.half_saturation),
                    float(propensity.coefficient),
                    float(propensity.basal_rate),
                ]
                repression[process_index] = propensity.repression
            elif isinstance(propensity, MichaelisMentenPropensity):
                kind[process_index] = _MICHAELIS_MENTEN
                propensity_species[process_index] = species_index[propensity.substrate]
                parameters[process_index] = [
                    float(propensity.maximum_rate),
                    float(propensity.michaelis_constant),
                    1.0,
                    0.0,
                ]
            else:
                kind[process_index] = _PROMOTER_TRANSITION
                propensity_species[process_index] = species_index[propensity.source]
                parameters[process_index] = [float(propensity.rate), 1.0, 1.0, 0.0]
        copy_number = np.asarray(
            [item.quantity == "count" for item in plan.species], dtype=bool
        )
        reservoir = np.asarray([item.reservoir for item in plan.species], dtype=bool)
        dynamic = dense.copy()
        dynamic[:, reservoir] = 0
        availability = np.maximum(-dense, 0)
        measures = np.asarray(
            [compartment_measure[item.compartment] for item in plan.species],
            dtype=float,
        )
        basis_rows = []
        basis_units = []
        rank = 0
        species_types = tuple(
            dict.fromkeys((species.quantity, species.unit) for species in plan.species)
        )
        for quantity, unit in species_types:
            indices = np.asarray(
                [
                    index
                    for index, species in enumerate(plan.species)
                    if (species.quantity, species.unit) == (quantity, unit)
                ],
                dtype=np.int32,
            )
            block = dense[:, indices].astype(float)
            singular_values = np.linalg.svd(block, compute_uv=False)
            threshold = (
                max(block.shape)
                * np.finfo(float).eps
                * max(float(np.max(singular_values, initial=0.0)), 1.0)
            )
            block_rank = int(np.sum(singular_values > threshold))
            rank += block_rank
            _, _, right = np.linalg.svd(block, full_matrices=True)
            for local_basis in right[block_rank:]:
                embedded = np.zeros(species_count, dtype=float)
                embedded[indices] = local_basis
                basis_rows.append(embedded)
                basis_units.append(f"{quantity}:{unit}")
        basis = (
            np.asarray(basis_rows)
            if basis_rows
            else np.zeros((0, species_count), dtype=float)
        )
        residual = basis @ dense.T
        maximum_residual = float(np.max(np.abs(residual), initial=0.0))
        network_id = canonical_fingerprint(
            {
                "kind": "prepared-systems-biology-network",
                "plan": plan.plan_id,
                "stoichiometry": array_tree_fingerprint(dense),
                "dynamic": array_tree_fingerprint(dynamic),
                "parameters": array_tree_fingerprint(parameters),
            }
        )
        self.plan = plan
        self.stoichiometry = jnp.asarray(dense)
        self.dynamic_stoichiometry = jnp.asarray(dynamic)
        self.stoichiometric_species = jnp.asarray(sparse_species)
        self.stoichiometric_values = jnp.asarray(sparse_values)
        self.stoichiometric_mask = jnp.asarray(sparse_mask)
        self.reservoir_mask = jnp.asarray(reservoir)
        self.copy_number_mask = jnp.asarray(copy_number)
        self.species_measure = jnp.asarray(measures)
        self.propensity_kind = jnp.asarray(kind)
        self.propensity_parameters = jnp.asarray(parameters)
        self.propensity_species = jnp.asarray(propensity_species)
        self.propensity_orders = jnp.asarray(orders)
        self.propensity_normalization = jnp.asarray(normalization)
        self.propensity_repression = jnp.asarray(repression)
        self.availability = jnp.asarray(availability)
        self.species_count = species_count
        self.process_count = reaction_count
        self.maximum_order = maximum_order
        self.network_id = network_id
        self.conservation = ConservationEvidence(
            jnp.asarray(basis),
            tuple(basis_units),
            rank,
            jnp.asarray(maximum_residual),
            jnp.asarray(maximum_residual <= 1.0e-10),
            network_id,
        )

    def default_runtime(self) -> StoichiometricRuntime:
        return StoichiometricRuntime(self.propensity_parameters)

    def initial_state(self, values: ArrayLike, /) -> Array:
        raw = jnp.asarray(values)
        if raw.dtype == jnp.bool_:
            raise TypeError("Initial state must not be boolean.")
        state = raw.astype(self.propensity_parameters.dtype)
        if state.shape != (self.species_count,):
            raise ValueError("Initial state must have shape (species_count,).")
        host = np.asarray(state)
        if np.any(~np.isfinite(host)) or np.any(host < 0.0):
            raise ValueError("Initial state must be finite and nonnegative.")
        return state

    def evaluate(
        self,
        state: ArrayLike,
        runtime: StoichiometricRuntime | None = None,
        /,
        *,
        mode: Literal["ssa", "deterministic"] = "ssa",
    ) -> StoichiometricEvaluation:
        values = jnp.asarray(state, dtype=self.propensity_parameters.dtype)
        if values.shape != (self.species_count,):
            raise ValueError("State must have shape (species_count,).")
        if mode not in ("ssa", "deterministic"):
            raise ValueError("mode must be 'ssa' or 'deterministic'.")
        runtime_value = self.default_runtime() if runtime is None else runtime
        if not isinstance(runtime_value, StoichiometricRuntime):
            raise TypeError("runtime must be StoichiometricRuntime.")
        if runtime_value.parameters.shape != (self.process_count, 4):
            raise ValueError(
                "Runtime parameter shape does not match the prepared network."
            )
        parameters = runtime_value.parameters
        integer_valid = (
            jnp.asarray(True)
            if mode == "deterministic"
            else jnp.all(values == jnp.floor(values))
        )
        state_valid = jnp.all(jnp.isfinite(values) & (values >= 0.0)) & integer_valid
        finite_parameters = jnp.all(jnp.isfinite(parameters))
        mass_action_valid = (
            (parameters[:, 0] >= 0.0)
            & (parameters[:, 1] == self.propensity_parameters[:, 1])
            & (parameters[:, 2] == self.propensity_parameters[:, 2])
        )
        hill_valid = (
            (parameters[:, 0] >= 0.0)
            & (parameters[:, 1] > 0.0)
            & (parameters[:, 2] > 0.0)
            & (parameters[:, 3] >= 0.0)
        )
        michaelis_valid = (parameters[:, 0] >= 0.0) & (parameters[:, 1] > 0.0)
        promoter_valid = parameters[:, 0] >= 0.0
        per_process_valid = jnp.where(
            self.propensity_kind == _MASS_ACTION,
            mass_action_valid,
            jnp.where(
                self.propensity_kind == _HILL,
                hill_valid,
                jnp.where(
                    self.propensity_kind == _MICHAELIS_MENTEN,
                    michaelis_valid,
                    promoter_valid,
                ),
            ),
        )
        parameter_valid = finite_parameters & jnp.all(per_process_valid)
        selected = values[self.propensity_species]
        concentration = selected / self.species_measure[self.propensity_species]
        mass_action = self._mass_action(values, parameters, exact=mode == "ssa")
        maximum = parameters[:, 0]
        half = jnp.where(parameters[:, 1] > 0.0, parameters[:, 1], 1.0)
        coefficient = jnp.where(parameters[:, 2] > 0.0, parameters[:, 2], 1.0)
        concentration_safe = jnp.maximum(concentration, 0.0)
        power = concentration_safe**coefficient
        half_power = half**coefficient
        hill_fraction = jnp.where(
            self.propensity_repression,
            half_power / (half_power + power),
            power / (half_power + power),
        )
        hill = parameters[:, 3] + maximum * hill_fraction
        michaelis = maximum * concentration_safe / (half + concentration_safe)
        promoter = maximum * selected
        raw = jnp.where(
            self.propensity_kind == _MASS_ACTION,
            mass_action,
            jnp.where(
                self.propensity_kind == _HILL,
                hill,
                jnp.where(
                    self.propensity_kind == _MICHAELIS_MENTEN,
                    michaelis,
                    promoter,
                ),
            ),
        )
        feasible = (
            jnp.ones((self.process_count,), dtype=bool)
            if mode == "deterministic"
            else jnp.all(values[None, :] >= self.availability, axis=-1)
        )
        rates = jnp.where(feasible, raw, 0.0)
        finite = jnp.all(jnp.isfinite(rates))
        successful = state_valid & parameter_valid & finite & jnp.all(rates >= 0.0)
        propensities = jnp.where(successful, rates, jnp.nan)
        drift = contract("r,rs->s", propensities, self.dynamic_stoichiometry)
        boundary_contribution = propensities[:, None] * (
            self.dynamic_stoichiometry - self.stoichiometry
        )
        source = jnp.sum(jnp.maximum(boundary_contribution, 0.0), axis=0)
        sink = jnp.sum(jnp.maximum(-boundary_contribution, 0.0), axis=0)
        residual = contract("ks,s->k", self.conservation.basis, drift + sink - source)
        status = jnp.where(
            ~state_valid,
            _STOICHIOMETRIC_INVALID_STATE,
            jnp.where(
                ~parameter_valid,
                _STOICHIOMETRIC_INVALID_PARAMETERS,
                jnp.where(
                    ~finite,
                    _STOICHIOMETRIC_NONFINITE,
                    _STOICHIOMETRIC_SUCCESS,
                ),
            ),
        )
        return StoichiometricEvaluation(
            propensities,
            drift,
            source,
            sink,
            residual,
            state_valid,
            parameter_valid,
            finite,
            successful,
            jnp.asarray(status, dtype=jnp.int32),
            self.network_id,
        )

    def _mass_action(self, state: Array, parameters: Array, /, *, exact: bool) -> Array:
        if exact:
            factors = jnp.ones(
                (self.process_count, self.species_count), dtype=state.dtype
            )
            for count in range(self.maximum_order):
                factors = factors * jnp.where(
                    self.propensity_orders > count,
                    state[None, :] - float(count),
                    1.0,
                )
            factors = factors / self.propensity_normalization
        else:
            factors = (
                state[None, :] ** self.propensity_orders / self.propensity_normalization
            )
        combinatorial = jnp.prod(factors, axis=-1)
        total_order = parameters[:, 2]
        measure = jnp.where(parameters[:, 1] > 0.0, parameters[:, 1], 1.0)
        return parameters[:, 0] * measure ** (1.0 - total_order) * combinatorial

    def deterministic_step(
        self,
        state: ArrayLike,
        duration: ArrayLike,
        runtime: StoichiometricRuntime | None = None,
        /,
        *,
        minimum_copy_number: float = 20.0,
    ) -> StoichiometricStepResult:
        values = jnp.asarray(state, dtype=self.propensity_parameters.dtype)
        dt = jnp.asarray(duration, dtype=values.dtype)
        if values.shape != (self.species_count,) or dt.shape != ():
            raise ValueError("State and duration have incompatible shapes.")
        if not isfinite(minimum_copy_number) or minimum_copy_number < 0.0:
            raise ValueError("minimum_copy_number must be finite and nonnegative.")
        evaluation = self.evaluate(values, runtime, mode="deterministic")
        candidate = values + dt * evaluation.drift
        minimum = jnp.min(
            jnp.where(
                self.copy_number_mask & ~self.reservoir_mask,
                jnp.minimum(values, candidate),
                jnp.inf,
            ),
            initial=jnp.inf,
        )
        numerical = (
            evaluation.successful
            & jnp.isfinite(dt)
            & (dt >= 0.0)
            & jnp.all(jnp.isfinite(candidate) & (candidate >= 0.0))
        )
        regime = minimum >= minimum_copy_number
        nonlinear_driver = (self.propensity_kind == _HILL) | (
            self.propensity_kind == _MICHAELIS_MENTEN
        )
        driver_interior = jnp.all(
            ~nonlinear_driver | (values[self.propensity_species] > 0.0)
        )
        boundary_structure = self.dynamic_stoichiometry != self.stoichiometry
        boundary_rate = evaluation.propensities[:, None] * (
            self.dynamic_stoichiometry - self.stoichiometry
        )
        boundary_interior = jnp.all(~boundary_structure | (boundary_rate != 0.0))
        differentiable = numerical & (dt > 0.0) & driver_interior & boundary_interior
        evidence = ApproximationEvidence(
            minimum,
            jnp.min(dt * evaluation.propensities, initial=jnp.inf),
            regime,
            jnp.asarray(True),
            regime,
            numerical,
            "deterministic",
            differentiable,
        )
        return StoichiometricStepResult(
            candidate,
            dt * evaluation.source_rate,
            dt * evaluation.sink_rate,
            evaluation,
            evidence,
            numerical,
            self.network_id,
        )

    def cle_step(
        self,
        state: ArrayLike,
        duration: ArrayLike,
        key: Key[Array, ""],
        runtime: StoichiometricRuntime | None = None,
        /,
        *,
        minimum_copy_number: float = 10.0,
        minimum_expected_events: float = 10.0,
    ) -> StoichiometricStepResult:
        if any(item.quantity != "count" for item in self.plan.species):
            raise ValueError("CLE requires every species quantity to be 'count'.")
        values = jnp.asarray(state, dtype=self.propensity_parameters.dtype)
        dt = jnp.asarray(duration, dtype=values.dtype)
        if values.shape != (self.species_count,) or dt.shape != ():
            raise ValueError("State and duration have incompatible shapes.")
        if jr.key_data(key).shape != (2,):
            raise ValueError("CLE requires one scalar JAX PRNG key.")
        if (
            not isfinite(minimum_copy_number)
            or not isfinite(minimum_expected_events)
            or minimum_copy_number < 0.0
            or minimum_expected_events < 0.0
        ):
            raise ValueError("CLE validity thresholds must be finite and nonnegative.")
        evaluation = self.evaluate(values, runtime, mode="deterministic")
        noise = jr.normal(key, (self.process_count,), dtype=values.dtype)
        fluctuations = jnp.sqrt(jnp.maximum(dt * evaluation.propensities, 0.0)) * noise
        reaction_extent = dt * evaluation.propensities + fluctuations
        dynamic_delta = contract("r,rs->s", reaction_extent, self.dynamic_stoichiometry)
        boundary_contribution = reaction_extent[:, None] * (
            self.dynamic_stoichiometry - self.stoichiometry
        )
        source_delta = jnp.sum(jnp.maximum(boundary_contribution, 0.0), axis=0)
        sink_delta = jnp.sum(jnp.maximum(-boundary_contribution, 0.0), axis=0)
        candidate = values + dynamic_delta
        minimum = jnp.min(
            jnp.where(
                self.copy_number_mask & ~self.reservoir_mask,
                jnp.minimum(values, candidate),
                jnp.inf,
            ),
            initial=jnp.inf,
        )
        expected = jnp.min(
            jnp.where(
                evaluation.propensities > 0.0,
                dt * evaluation.propensities,
                jnp.inf,
            ),
            initial=jnp.inf,
        )
        copy_valid = minimum >= minimum_copy_number
        frequency_valid = expected >= minimum_expected_events
        numerical = (
            evaluation.successful
            & jnp.isfinite(dt)
            & (dt >= 0.0)
            & jnp.all(jnp.isfinite(candidate) & (candidate >= 0.0))
        )
        nonlinear_driver = (self.propensity_kind == _HILL) | (
            self.propensity_kind == _MICHAELIS_MENTEN
        )
        driver_interior = jnp.all(
            ~nonlinear_driver | (values[self.propensity_species] > 0.0)
        )
        boundary_structure = self.dynamic_stoichiometry != self.stoichiometry
        boundary_interior = jnp.all(~boundary_structure | (boundary_contribution != 0.0))
        differentiable = (
            numerical
            & (dt > 0.0)
            & jnp.all(evaluation.propensities > 0.0)
            & driver_interior
            & boundary_interior
        )
        evidence = ApproximationEvidence(
            minimum,
            expected,
            copy_valid,
            frequency_valid,
            copy_valid & frequency_valid,
            numerical,
            "cle",
            differentiable,
        )
        return StoichiometricStepResult(
            candidate,
            source_delta,
            sink_delta,
            evaluation,
            evidence,
            numerical,
            self.network_id,
        )

    def exact_jump_process(self) -> CompartmentalJumpProcess:
        return CompartmentalJumpProcess(self)

    def bind_thermochemical(
        self, mechanism: PreparedChemicalMechanism, /
    ) -> ThermochemicalInteropEvidence:
        if not isinstance(mechanism, PreparedChemicalMechanism):
            raise TypeError("mechanism must be PreparedChemicalMechanism.")
        mechanism_species = {
            name: index for index, name in enumerate(mechanism.schema.species_names)
        }
        species_indices = []
        for species in self.plan.species:
            if species.thermochemical_name is None:
                raise ValueError(
                    "Every systems-biology species requires thermochemical_name "
                    "for binding."
                )
            if species.thermochemical_name not in mechanism_species:
                raise ValueError(
                    f"Unknown thermochemical species {species.thermochemical_name!r}."
                )
            species_indices.append(mechanism_species[species.thermochemical_name])
        if len(set(species_indices)) != len(species_indices):
            raise ValueError("Thermochemical species bindings must be one-to-one.")
        reaction_name_values = tuple(item.name for item in mechanism.reactions)
        if len(set(reaction_name_values)) != len(reaction_name_values):
            raise ValueError(
                "Thermochemical reaction names must be unique for biological binding."
            )
        reaction_names = {
            item.name: index for index, item in enumerate(mechanism.reactions)
        }
        process_indices = []
        reaction_indices = []
        residuals = []
        mechanism_reactant = np.asarray(mechanism.reactant_stoichiometry)
        mechanism_product = np.asarray(mechanism.product_stoichiometry)
        mechanism_orders = np.asarray(mechanism.forward_orders)
        biological_net = np.asarray(self.stoichiometry)
        biological_orders = np.asarray(self.propensity_orders)
        for process_index, process in enumerate(self.plan.processes):
            if process.thermochemical_reaction is None:
                continue
            if not isinstance(process.propensity, MassActionPropensity):
                raise ValueError("Thermochemical bindings require mass-action processes.")
            if process.thermochemical_reaction not in reaction_names:
                raise ValueError(
                    "Unknown thermochemical reaction "
                    f"{process.thermochemical_reaction!r}."
                )
            reaction_index = reaction_names[process.thermochemical_reaction]
            if reaction_index in reaction_indices:
                raise ValueError("Thermochemical reaction bindings must be one-to-one.")
            biological_reactant = biological_orders[process_index].astype(float)
            biological_product = biological_reactant + biological_net[process_index]
            if np.any(biological_product < 0.0):
                raise ValueError(
                    "A thermochemical mass-action binding requires kinetic "
                    "orders to cover every consumed species."
                )
            represented_reactant = np.zeros(mechanism.schema.species_count, dtype=float)
            represented_product = np.zeros_like(represented_reactant)
            represented_orders = np.zeros_like(represented_reactant)
            mapped = np.asarray(species_indices)
            represented_reactant[mapped] = biological_reactant
            represented_product[mapped] = biological_product
            represented_orders[mapped] = biological_orders[process_index]
            residuals.extend(
                (
                    represented_reactant - mechanism_reactant[reaction_index],
                    represented_product - mechanism_product[reaction_index],
                    represented_orders - mechanism_orders[reaction_index],
                )
            )
            process_indices.append(process_index)
            reaction_indices.append(reaction_index)
        if not reaction_indices:
            raise ValueError("No process declares a thermochemical reaction binding.")
        residual = np.asarray(residuals)
        maximum = float(np.max(np.abs(residual), initial=0.0))
        if maximum > 1.0e-12:
            raise ValueError(
                "Systems-biology reactants, products, and mass-action orders "
                "do not exactly match the thermochemical reaction."
            )
        mechanism_content_id = canonical_fingerprint(
            {
                "kind": "systems-biology-bound-mechanism-content",
                "content": _semantic_payload(mechanism),
            }
        )
        binding_id = canonical_fingerprint(
            {
                "kind": "systems-biology-thermochemical-binding",
                "network": self.network_id,
                "mechanism": mechanism_content_id,
                "species": species_indices,
                "processes": process_indices,
                "reactions": reaction_indices,
            }
        )
        return ThermochemicalInteropEvidence(
            jnp.asarray(species_indices, dtype=jnp.int32),
            jnp.asarray(process_indices, dtype=jnp.int32),
            jnp.asarray(reaction_indices, dtype=jnp.int32),
            jnp.asarray(maximum),
            jnp.asarray(True),
            self.network_id,
            mechanism.mechanism_id,
            mechanism_content_id,
            binding_id,
        )

    def evidence_fields(self) -> dict[str, object]:
        """Return the closed set of host evidence-addressable plan fields."""
        fields: dict[str, object] = {
            "network.plan_id": self.plan.plan_id,
            "network.prepared_id": self.network_id,
            "network.name": self.plan.name,
            "network.species_count": self.species_count,
            "network.process_count": self.process_count,
            "network.stoichiometry_capacity": self.plan.stoichiometry_capacity,
            "network.time_unit": self.plan.time_unit,
        }
        for compartment in self.plan.compartments:
            prefix = f"network.compartment.{compartment.name}"
            fields[f"{prefix}.measure"] = float(compartment.measure)
            fields[f"{prefix}.unit"] = compartment.unit
        for species in self.plan.species:
            prefix = f"network.species.{species.name}"
            fields[f"{prefix}.compartment"] = species.compartment
            fields[f"{prefix}.reservoir"] = species.reservoir
            fields[f"{prefix}.quantity"] = species.quantity
            fields[f"{prefix}.unit"] = species.unit
            if species.thermochemical_name is not None:
                fields[f"{prefix}.thermochemical_name"] = species.thermochemical_name
        for process in self.plan.processes:
            prefix = f"network.process.{process.name}"
            for species_name, coefficient in process.stoichiometry:
                fields[f"{prefix}.stoichiometry.{species_name}"] = coefficient
            if process.thermochemical_reaction is not None:
                fields[f"{prefix}.thermochemical_reaction"] = (
                    process.thermochemical_reaction
                )
            propensity = process.propensity
            if isinstance(propensity, MassActionPropensity):
                fields[f"{prefix}.propensity.kind"] = "mass_action"
                fields[f"{prefix}.propensity.rate"] = float(propensity.rate)
                for species_name, order in propensity.orders:
                    fields[f"{prefix}.propensity.order.{species_name}"] = order
            elif isinstance(propensity, HillPropensity):
                fields[f"{prefix}.propensity.kind"] = "hill"
                fields[f"{prefix}.propensity.maximum_rate"] = float(
                    propensity.maximum_rate
                )
                fields[f"{prefix}.propensity.half_saturation"] = float(
                    propensity.half_saturation
                )
                fields[f"{prefix}.propensity.coefficient"] = float(propensity.coefficient)
                fields[f"{prefix}.propensity.basal_rate"] = float(propensity.basal_rate)
                fields[f"{prefix}.propensity.regulator"] = propensity.regulator
                fields[f"{prefix}.propensity.repression"] = propensity.repression
            elif isinstance(propensity, MichaelisMentenPropensity):
                fields[f"{prefix}.propensity.kind"] = "michaelis_menten"
                fields[f"{prefix}.propensity.maximum_rate"] = float(
                    propensity.maximum_rate
                )
                fields[f"{prefix}.propensity.michaelis_constant"] = float(
                    propensity.michaelis_constant
                )
                fields[f"{prefix}.propensity.substrate"] = propensity.substrate
            else:
                fields[f"{prefix}.propensity.kind"] = "promoter_transition"
                fields[f"{prefix}.propensity.rate"] = float(propensity.rate)
                fields[f"{prefix}.propensity.source"] = propensity.source
        return fields

    def evidence_units(self) -> dict[str, str]:
        """Return the declared unit for every evidence-addressable field."""
        units = {
            "network.plan_id": "identity",
            "network.prepared_id": "identity",
            "network.name": "label",
            "network.species_count": "count",
            "network.process_count": "count",
            "network.stoichiometry_capacity": "count",
            "network.time_unit": "label",
        }
        compartments_by_name = {item.name: item for item in self.plan.compartments}
        species_by_name = {item.name: item for item in self.plan.species}
        for compartment in self.plan.compartments:
            prefix = f"network.compartment.{compartment.name}"
            units[f"{prefix}.measure"] = compartment.unit
            units[f"{prefix}.unit"] = "label"
        for species in self.plan.species:
            prefix = f"network.species.{species.name}"
            units[f"{prefix}.compartment"] = "label"
            units[f"{prefix}.reservoir"] = "boolean"
            units[f"{prefix}.quantity"] = "label"
            units[f"{prefix}.unit"] = "label"
            if species.thermochemical_name is not None:
                units[f"{prefix}.thermochemical_name"] = "label"
        for process in self.plan.processes:
            prefix = f"network.process.{process.name}"
            for species_name, _ in process.stoichiometry:
                units[f"{prefix}.stoichiometry.{species_name}"] = "dimensionless"
            if process.thermochemical_reaction is not None:
                units[f"{prefix}.thermochemical_reaction"] = "label"
            changed_species = species_by_name[process.stoichiometry[0][0]]
            propensity = process.propensity
            units[f"{prefix}.propensity.kind"] = "label"
            if isinstance(propensity, MassActionPropensity):
                total_order = sum(order for _, order in propensity.orders)
                ordered_species = species_by_name[propensity.orders[0][0]]
                measure_unit = compartments_by_name[ordered_species.compartment].unit
                factors = [
                    (changed_species.unit, 1),
                    (measure_unit, total_order - 1),
                    (self.plan.time_unit, -1),
                ]
                factors.extend(
                    (species_by_name[species_name].unit, -order)
                    for species_name, order in propensity.orders
                )
                units[f"{prefix}.propensity.rate"] = _canonical_unit(*factors)
                for species_name, _ in propensity.orders:
                    units[f"{prefix}.propensity.order.{species_name}"] = "dimensionless"
            elif isinstance(propensity, HillPropensity):
                regulator = species_by_name[propensity.regulator]
                measure_unit = compartments_by_name[regulator.compartment].unit
                units[f"{prefix}.propensity.maximum_rate"] = _canonical_unit(
                    (changed_species.unit, 1),
                    (self.plan.time_unit, -1),
                )
                units[f"{prefix}.propensity.half_saturation"] = _canonical_unit(
                    (regulator.unit, 1),
                    (measure_unit, -1),
                )
                units[f"{prefix}.propensity.coefficient"] = "dimensionless"
                units[f"{prefix}.propensity.basal_rate"] = _canonical_unit(
                    (changed_species.unit, 1),
                    (self.plan.time_unit, -1),
                )
                units[f"{prefix}.propensity.regulator"] = "label"
                units[f"{prefix}.propensity.repression"] = "boolean"
            elif isinstance(propensity, MichaelisMentenPropensity):
                substrate = species_by_name[propensity.substrate]
                measure_unit = compartments_by_name[substrate.compartment].unit
                units[f"{prefix}.propensity.maximum_rate"] = _canonical_unit(
                    (changed_species.unit, 1),
                    (self.plan.time_unit, -1),
                )
                units[f"{prefix}.propensity.michaelis_constant"] = _canonical_unit(
                    (substrate.unit, 1),
                    (measure_unit, -1),
                )
                units[f"{prefix}.propensity.substrate"] = "label"
            else:
                source = species_by_name[propensity.source]
                units[f"{prefix}.propensity.rate"] = _canonical_unit(
                    (changed_species.unit, 1),
                    (source.unit, -1),
                    (self.plan.time_unit, -1),
                )
                units[f"{prefix}.propensity.source"] = "label"
        return units


class CompartmentalJumpProcess(AbstractJumpProcess):
    """Exact SSA adapter over a prepared compartmental stoichiometric network."""

    network: PreparedStoichiometricNetwork
    state_shape: tuple[int, ...] = eqx.field(static=True)
    num_channels: int = eqx.field(static=True)
    mark_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(self, network: PreparedStoichiometricNetwork, /):
        if not isinstance(network, PreparedStoichiometricNetwork):
            raise TypeError("network must be PreparedStoichiometricNetwork.")
        if any(item.quantity != "count" for item in network.plan.species):
            raise ValueError("Exact SSA requires every species quantity to be 'count'.")
        self.network = network
        self.state_shape = (network.species_count,)
        self.num_channels = network.process_count
        self.mark_shape = ()
        self.process_id = canonical_fingerprint(
            {"kind": "systems-biology-exact-jump", "network": network.network_id}
        )

    def intensities(self, time, state, args=None, /):
        del time
        runtime = self.network.default_runtime() if args is None else args
        return self.network.evaluate(state, runtime, mode="ssa").propensities

    def jump(self, state, channel, mark, args=None, /):
        del mark, args
        return (
            jnp.asarray(state)
            + self.network.dynamic_stoichiometry[jnp.asarray(channel, dtype=jnp.int32)]
        )

    def sample_mark(self, key, time, state, channel, args=None, /):
        del key, time, channel, args
        return jnp.asarray(0, dtype=jnp.asarray(state).dtype)


def _process_payload(process: StoichiometricProcessSpec, /) -> dict[str, object]:
    propensity = process.propensity
    if isinstance(propensity, MassActionPropensity):
        payload: tuple[object, ...] = (
            "mass_action",
            float(propensity.rate),
            propensity.orders,
        )
    elif isinstance(propensity, HillPropensity):
        payload = (
            "hill",
            float(propensity.maximum_rate),
            float(propensity.half_saturation),
            float(propensity.coefficient),
            propensity.regulator,
            float(propensity.basal_rate),
            propensity.repression,
        )
    elif isinstance(propensity, MichaelisMentenPropensity):
        payload = (
            "michaelis_menten",
            float(propensity.maximum_rate),
            float(propensity.michaelis_constant),
            propensity.substrate,
        )
    else:
        payload = (
            "promoter_transition",
            float(propensity.rate),
            propensity.source,
        )
    return {
        "name": process.name,
        "stoichiometry": process.stoichiometry,
        "propensity": payload,
        "thermochemical_reaction": process.thermochemical_reaction,
    }


__all__ = [
    "ApproximationEvidence",
    "ApproximationKind",
    "CompartmentalJumpProcess",
    "CompartmentSpec",
    "ConservationEvidence",
    "HillPropensity",
    "PropensitySpec",
    "MassActionPropensity",
    "MichaelisMentenPropensity",
    "PreparedStoichiometricNetwork",
    "PromoterTransitionPropensity",
    "SpeciesSpec",
    "StoichiometricEvaluation",
    "StoichiometricNetworkPlan",
    "StoichiometricProcessSpec",
    "StoichiometricRuntime",
    "StoichiometricStatus",
    "StoichiometricStepResult",
    "ThermochemicalInteropEvidence",
]
