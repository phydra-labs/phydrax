#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be a non-empty string.")
    return value.strip()


def _unique_identifiers(values: Sequence[str], owner: str, /) -> tuple[str, ...]:
    result = tuple(_identifier(value, owner) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{owner} values must be unique.")
    return result


class UnitDimension(StrictModule):
    """SI base-dimension exponents independent of a unit scale or spelling."""

    mass: int = eqx.field(static=True)
    length: int = eqx.field(static=True)
    time: int = eqx.field(static=True)
    amount: int = eqx.field(static=True)
    current: int = eqx.field(static=True)
    temperature: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        mass: int = 0,
        length: int = 0,
        time: int = 0,
        amount: int = 0,
        current: int = 0,
        temperature: int = 0,
    ):
        self.mass = int(mass)
        self.length = int(length)
        self.time = int(time)
        self.amount = int(amount)
        self.current = int(current)
        self.temperature = int(temperature)

    @property
    def exponents(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.mass,
            self.length,
            self.time,
            self.amount,
            self.current,
            self.temperature,
        )

    def multiply(self, other: UnitDimension, /) -> UnitDimension:
        if not isinstance(other, UnitDimension):
            raise TypeError("other must be a UnitDimension.")
        values = tuple(
            left + right for left, right in zip(self.exponents, other.exponents)
        )
        return UnitDimension(
            mass=values[0],
            length=values[1],
            time=values[2],
            amount=values[3],
            current=values[4],
            temperature=values[5],
        )

    def power(self, exponent: int, /) -> UnitDimension:
        value = int(exponent)
        values = tuple(value * item for item in self.exponents)
        return UnitDimension(
            mass=values[0],
            length=values[1],
            time=values[2],
            amount=values[3],
            current=values[4],
            temperature=values[5],
        )


DIMENSIONLESS = UnitDimension()
MASS = UnitDimension(mass=1)
VOLUME = UnitDimension(length=3)
TIME = UnitDimension(time=1)
SUBSTANCE = UnitDimension(amount=1)
SUBSTANCE_FLUX = UnitDimension(time=-1, amount=1)
CONCENTRATION = UnitDimension(length=-3, amount=1)
CONCENTRATION_RATE = UnitDimension(length=-3, time=-1, amount=1)


class ChemicalComposition(StrictModule):
    """Exact integer elemental composition and formal charge of one species."""

    elements: tuple[tuple[str, int], ...] = eqx.field(static=True)
    charge: int = eqx.field(static=True)

    def __init__(
        self,
        elements: Mapping[str, int] | Sequence[tuple[str, int]] = (),
        /,
        *,
        charge: int = 0,
    ):
        items = elements.items() if isinstance(elements, Mapping) else elements
        normalized: list[tuple[str, int]] = []
        for symbol, count in items:
            symbol_ = _identifier(symbol, "element symbol")
            count_ = int(count)
            if count_ <= 0:
                raise ValueError("Element counts must be positive integers.")
            normalized.append((symbol_, count_))
        normalized.sort()
        if len({symbol for symbol, _ in normalized}) != len(normalized):
            raise ValueError("Element symbols must be unique.")
        self.elements = tuple(normalized)
        self.charge = int(charge)

    def count(self, symbol: str, /) -> int:
        symbol_ = _identifier(symbol, "element symbol")
        for candidate, count in self.elements:
            if candidate == symbol_:
                return count
        return 0


class GeneReactionRule(StrictModule):
    """Canonical disjunctive normal form: OR of AND gene clauses."""

    clauses: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    genes: tuple[str, ...] = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(self, clauses: Sequence[Sequence[str]], /):
        normalized = []
        for clause in clauses:
            genes = tuple(sorted(_unique_identifiers(tuple(clause), "gene identifier")))
            normalized.append(genes)
        clauses_ = tuple(sorted(set(normalized)))
        if not clauses_:
            raise ValueError("A gene-reaction rule requires at least one clause.")
        self.clauses = clauses_
        self.genes = tuple(sorted({gene for clause in clauses_ for gene in clause}))
        self.rule_id = canonical_fingerprint(
            {"kind": "gene-reaction-rule", "clauses": [list(item) for item in clauses_]}
        )

    def evaluate(self, active_genes: Iterable[str], /) -> bool:
        active = frozenset(_identifier(gene, "active gene") for gene in active_genes)
        return any(all(gene in active for gene in clause) for clause in self.clauses)


class Compartment(StrictModule):
    """Well-mixed physical compartment with explicit volume dimensions."""

    volume: Array
    compartment_id: str = eqx.field(static=True)
    volume_unit: UnitDimension = eqx.field(static=True)
    spatial_dimensions: int = eqx.field(static=True)
    constant: bool = eqx.field(static=True)

    def __init__(
        self,
        compartment_id: str,
        /,
        *,
        volume: ArrayLike = 1.0,
        volume_unit: UnitDimension = VOLUME,
        spatial_dimensions: int = 3,
        constant: bool = True,
    ):
        if not isinstance(volume_unit, UnitDimension):
            raise TypeError("volume_unit must be a UnitDimension.")
        dimensions = int(spatial_dimensions)
        if dimensions < 0:
            raise ValueError("spatial_dimensions must be non-negative.")
        volume_ = jnp.asarray(volume)
        if volume_.shape != () or not jnp.issubdtype(volume_.dtype, jnp.number):
            raise ValueError("Compartment volume must be one numeric scalar.")
        volume_ = (
            volume_
            if jnp.issubdtype(volume_.dtype, jnp.inexact)
            else volume_.astype(float)
        )
        volume_ = eqx.error_if(
            volume_,
            ~jnp.isfinite(volume_) | (volume_ <= 0.0),
            "Compartment volume must be finite and positive.",
        )
        self.volume = volume_
        self.compartment_id = _identifier(compartment_id, "compartment_id")
        self.volume_unit = volume_unit
        self.spatial_dimensions = dimensions
        self.constant = bool(constant)


class Species(StrictModule):
    """Chemical species identity, location, initial amount, and exact composition."""

    initial_amount: Array
    species_id: str = eqx.field(static=True)
    compartment_id: str = eqx.field(static=True)
    substance_unit: UnitDimension = eqx.field(static=True)
    composition: ChemicalComposition | None = eqx.field(static=True)
    boundary_condition: bool = eqx.field(static=True)
    constant: bool = eqx.field(static=True)

    def __init__(
        self,
        species_id: str,
        compartment_id: str,
        /,
        *,
        initial_amount: ArrayLike = 0.0,
        substance_unit: UnitDimension = SUBSTANCE,
        composition: ChemicalComposition | None = None,
        boundary_condition: bool = False,
        constant: bool = False,
    ):
        if not isinstance(substance_unit, UnitDimension):
            raise TypeError("substance_unit must be a UnitDimension.")
        if composition is not None and not isinstance(composition, ChemicalComposition):
            raise TypeError("composition must be a ChemicalComposition or None.")
        amount = jnp.asarray(initial_amount)
        if amount.shape != () or not jnp.issubdtype(amount.dtype, jnp.number):
            raise ValueError("initial_amount must be one numeric scalar.")
        amount = (
            amount if jnp.issubdtype(amount.dtype, jnp.inexact) else amount.astype(float)
        )
        amount = eqx.error_if(
            amount,
            ~jnp.isfinite(amount) | (amount < 0.0),
            "initial_amount must be finite and non-negative.",
        )
        self.initial_amount = amount
        self.species_id = _identifier(species_id, "species_id")
        self.compartment_id = _identifier(compartment_id, "compartment_id")
        self.substance_unit = substance_unit
        self.composition = composition
        self.boundary_condition = bool(boundary_condition)
        self.constant = bool(constant)


class Reaction(StrictModule):
    """Sparse stoichiometric reaction with bounds, objective, GPR, and units."""

    stoichiometric_coefficients: Array
    lower_bound: Array
    upper_bound: Array
    objective_coefficient: Array
    reaction_id: str = eqx.field(static=True)
    species_ids: tuple[str, ...] = eqx.field(static=True)
    flux_unit: UnitDimension = eqx.field(static=True)
    gene_reaction_rule: GeneReactionRule | None = eqx.field(static=True)
    exchange: bool = eqx.field(static=True)

    def __init__(
        self,
        reaction_id: str,
        species_ids: Sequence[str],
        stoichiometric_coefficients: ArrayLike,
        /,
        *,
        lower_bound: ArrayLike = 0.0,
        upper_bound: ArrayLike = jnp.inf,
        objective_coefficient: ArrayLike = 0.0,
        flux_unit: UnitDimension = SUBSTANCE_FLUX,
        gene_reaction_rule: GeneReactionRule | None = None,
        exchange: bool = False,
    ):
        identifiers = _unique_identifiers(species_ids, "reaction species identifier")
        coefficients = jnp.asarray(stoichiometric_coefficients)
        if coefficients.ndim != 1 or coefficients.shape != (len(identifiers),):
            raise ValueError(
                "stoichiometric_coefficients must have one entry per reaction species."
            )
        if not identifiers:
            raise ValueError("A reaction must involve at least one species.")
        coefficients = (
            coefficients
            if jnp.issubdtype(coefficients.dtype, jnp.inexact)
            else coefficients.astype(float)
        )
        coefficients = eqx.error_if(
            coefficients,
            jnp.any(~jnp.isfinite(coefficients) | (coefficients == 0.0)),
            "Stoichiometric coefficients must be finite and non-zero.",
        )
        lower = jnp.asarray(lower_bound, dtype=coefficients.dtype)
        upper = jnp.asarray(upper_bound, dtype=coefficients.dtype)
        objective = jnp.asarray(objective_coefficient, dtype=coefficients.dtype)
        if lower.shape != () or upper.shape != () or objective.shape != ():
            raise ValueError("Reaction bounds and objective coefficient must be scalar.")
        lower = eqx.error_if(
            lower,
            jnp.isnan(lower) | jnp.isnan(upper) | (lower > upper),
            "Reaction bounds must be ordered and cannot be NaN.",
        )
        objective = eqx.error_if(
            objective,
            ~jnp.isfinite(objective),
            "Reaction objective coefficient must be finite.",
        )
        if not isinstance(flux_unit, UnitDimension):
            raise TypeError("flux_unit must be a UnitDimension.")
        if gene_reaction_rule is not None and not isinstance(
            gene_reaction_rule, GeneReactionRule
        ):
            raise TypeError("gene_reaction_rule must be a GeneReactionRule or None.")
        self.stoichiometric_coefficients = coefficients
        self.lower_bound = lower
        self.upper_bound = upper
        self.objective_coefficient = objective
        self.reaction_id = _identifier(reaction_id, "reaction_id")
        self.species_ids = identifiers
        self.flux_unit = flux_unit
        self.gene_reaction_rule = gene_reaction_rule
        self.exchange = bool(exchange)

    @property
    def reversible(self) -> Array:
        return (self.lower_bound < 0.0) & (self.upper_bound > 0.0)


class StoichiometricNetwork(StrictModule):
    """Compiled compartment/species/reaction topology for flux and kinetic methods."""

    compartments: tuple[Compartment, ...]
    species: tuple[Species, ...]
    reactions: tuple[Reaction, ...]
    stoichiometric_matrix: Array
    reaction_species_indices: tuple[Array, ...]
    internal_species_mask: Array
    boundary_species_mask: Array
    lower_bounds: Array
    upper_bounds: Array
    objective_coefficients: Array
    compartment_volumes: Array
    network_id: str = eqx.field(static=True)
    species_ids: tuple[str, ...] = eqx.field(static=True)
    reaction_ids: tuple[str, ...] = eqx.field(static=True)
    compartment_ids: tuple[str, ...] = eqx.field(static=True)
    objective_sense: str = eqx.field(static=True)

    def __init__(
        self,
        compartments: Sequence[Compartment],
        species: Sequence[Species],
        reactions: Sequence[Reaction],
        /,
        *,
        objective_sense: str = "maximize",
    ):
        compartments_ = tuple(compartments)
        species_ = tuple(species)
        reactions_ = tuple(reactions)
        if not compartments_ or any(
            not isinstance(item, Compartment) for item in compartments_
        ):
            raise ValueError("compartments must contain at least one Compartment.")
        if not species_ or any(not isinstance(item, Species) for item in species_):
            raise ValueError("species must contain at least one Species.")
        if not reactions_ or any(not isinstance(item, Reaction) for item in reactions_):
            raise ValueError("reactions must contain at least one Reaction.")
        compartment_ids = _unique_identifiers(
            tuple(item.compartment_id for item in compartments_), "compartment_id"
        )
        species_ids = _unique_identifiers(
            tuple(item.species_id for item in species_), "species_id"
        )
        reaction_ids = _unique_identifiers(
            tuple(item.reaction_id for item in reactions_), "reaction_id"
        )
        compartment_index = {name: index for index, name in enumerate(compartment_ids)}
        species_index = {name: index for index, name in enumerate(species_ids)}
        for item in species_:
            if item.compartment_id not in compartment_index:
                raise ValueError(
                    f"Species {item.species_id!r} references unknown compartment "
                    f"{item.compartment_id!r}."
                )
        dtype = jnp.result_type(
            *(item.stoichiometric_coefficients.dtype for item in reactions_), jnp.float32
        )
        matrix = jnp.zeros((len(species_), len(reactions_)), dtype=dtype)
        topology = []
        for reaction_index, reaction in enumerate(reactions_):
            unknown = tuple(
                name for name in reaction.species_ids if name not in species_index
            )
            if unknown:
                raise ValueError(
                    f"Reaction {reaction.reaction_id!r} references unknown species {unknown}."
                )
            indices = jnp.asarray(
                [species_index[name] for name in reaction.species_ids], dtype=jnp.int32
            )
            topology.append(indices)
            matrix = matrix.at[indices, reaction_index].set(
                reaction.stoichiometric_coefficients.astype(dtype)
            )
        boundary = jnp.asarray(
            [item.boundary_condition or item.constant for item in species_], dtype=bool
        )
        internal = ~boundary
        lower = jnp.stack(tuple(item.lower_bound.astype(dtype) for item in reactions_))
        upper = jnp.stack(tuple(item.upper_bound.astype(dtype) for item in reactions_))
        objective = jnp.stack(
            tuple(item.objective_coefficient.astype(dtype) for item in reactions_)
        )
        volumes = jnp.stack(
            tuple(
                compartments_[compartment_index[item.compartment_id]].volume.astype(dtype)
                for item in species_
            )
        )
        if objective_sense not in ("maximize", "minimize"):
            raise ValueError("objective_sense must be 'maximize' or 'minimize'.")
        self.compartments = compartments_
        self.species = species_
        self.reactions = reactions_
        self.stoichiometric_matrix = matrix
        self.reaction_species_indices = tuple(topology)
        self.internal_species_mask = internal
        self.boundary_species_mask = boundary
        self.lower_bounds = lower
        self.upper_bounds = upper
        self.objective_coefficients = objective
        self.compartment_volumes = volumes
        self.species_ids = species_ids
        self.reaction_ids = reaction_ids
        self.compartment_ids = compartment_ids
        self.objective_sense = objective_sense
        self.network_id = canonical_fingerprint(
            {
                "kind": "stoichiometric-network",
                "compartments": list(compartment_ids),
                "species": [
                    {
                        "id": item.species_id,
                        "compartment": item.compartment_id,
                        "boundary": item.boundary_condition,
                        "constant": item.constant,
                        "unit": list(item.substance_unit.exponents),
                    }
                    for item in species_
                ],
                "reactions": [
                    {
                        "id": item.reaction_id,
                        "species": list(item.species_ids),
                        "exchange": item.exchange,
                        "unit": list(item.flux_unit.exponents),
                        "gpr": None
                        if item.gene_reaction_rule is None
                        else item.gene_reaction_rule.rule_id,
                    }
                    for item in reactions_
                ],
                "stoichiometry": np.asarray(matrix).tolist(),
                "objective_sense": objective_sense,
            }
        )

    @property
    def num_species(self) -> int:
        return len(self.species)

    @property
    def num_reactions(self) -> int:
        return len(self.reactions)

    @property
    def initial_amounts(self) -> Array:
        return jnp.stack(tuple(item.initial_amount for item in self.species))

    @property
    def initial_concentrations(self) -> Array:
        return self.initial_amounts / self.compartment_volumes

    @property
    def steady_state_matrix(self) -> Array:
        return self.stoichiometric_matrix[self.internal_species_mask]

    def reaction_index(self, reaction_id: str, /) -> int:
        return self.reaction_ids.index(_identifier(reaction_id, "reaction_id"))

    def species_index(self, species_id: str, /) -> int:
        return self.species_ids.index(_identifier(species_id, "species_id"))

    def bounds_for_active_genes(
        self, active_genes: Iterable[str], /
    ) -> tuple[Array, Array]:
        active = tuple(active_genes)
        enabled = jnp.asarray(
            [
                True
                if reaction.gene_reaction_rule is None
                else reaction.gene_reaction_rule.evaluate(active)
                for reaction in self.reactions
            ],
            dtype=bool,
        )
        lower = jnp.where(enabled, self.lower_bounds, 0.0)
        upper = jnp.where(enabled, self.upper_bounds, 0.0)
        return lower, upper


__all__ = [
    "ChemicalComposition",
    "Compartment",
    "CONCENTRATION",
    "CONCENTRATION_RATE",
    "DIMENSIONLESS",
    "GeneReactionRule",
    "MASS",
    "Reaction",
    "Species",
    "StoichiometricNetwork",
    "SUBSTANCE",
    "SUBSTANCE_FLUX",
    "TIME",
    "UnitDimension",
    "VOLUME",
]
