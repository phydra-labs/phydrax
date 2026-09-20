#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import ParticleNeighborhoodState, PeriodicCell
from ._graph import (
    AtomisticGraph,
    AtomisticGraphExecutionPlan,
    realize_particle_atomistic_graph,
)
from ._potential import (
    AbstractAtomisticPotential,
    AtomisticPotentialCapabilities,
    AtomisticPotentialRequirements,
    AtomisticSpeciesKind,
)
from ._sites import AtomisticInteractionSiteState
from ._system import PreparedAtomisticSystem


class AtomisticInteractionScaleState(StrictModule):
    """Numeric route scales supplied by a prepared controlled Hamiltonian."""

    bond: Array
    angle: Array
    torsion: Array
    improper: Array
    lennard_jones: Array
    electrostatic: Array
    lennard_jones_softcore_alpha: Array
    electrostatic_softcore_alpha: Array
    softcore_power: int = eqx.field(static=True)

    @classmethod
    def identity(
        cls,
        system: PreparedAtomisticSystem,
        pair_count: int,
        dtype,
        /,
    ) -> "AtomisticInteractionScaleState":
        one = lambda size: jnp.ones((size,), dtype=dtype)
        return cls(
            bond=one(system.topology.bond_indices.shape[0]),
            angle=one(system.topology.angle_indices.shape[0]),
            torsion=one(system.topology.torsion_indices.shape[0]),
            improper=one(system.topology.improper_indices.shape[0]),
            lennard_jones=one(pair_count),
            electrostatic=one(pair_count),
            lennard_jones_softcore_alpha=jnp.zeros((), dtype=dtype),
            electrostatic_softcore_alpha=jnp.zeros((), dtype=dtype),
            softcore_power=1,
        )


class AtomisticPotentialContext(StrictModule):
    """Requirements-resolved dynamic data shared by one potential program."""

    system: PreparedAtomisticSystem
    positions: Array
    unwrapped_positions: Array
    site_state: AtomisticInteractionSiteState
    site_positions: Array
    site_type_ids: Array
    site_charges: Array
    site_pair_left: Array
    site_pair_right: Array
    site_pair_valid: Array
    site_pair_displacement: Array
    site_pair_distance: Array
    site_lennard_jones_scales: Array
    site_electrostatic_scales: Array
    species: Array
    pair_left: Array
    pair_right: Array
    pair_valid: Array
    pair_keys: Array
    pair_displacement: Array
    pair_distance: Array
    lennard_jones_scales: Array
    electrostatic_scales: Array
    interaction_scales: AtomisticInteractionScaleState
    graph: AtomisticGraph | None
    cell: PeriodicCell | None
    cell_vectors: Array
    neighborhood_successful: Array


class AtomisticTermEvaluation(StrictModule):
    energy: Array
    atom_energy: Array
    successful: Array


class AbstractAtomisticEnergyTerm(StrictModule):
    """One composable scalar-energy term in a prepared atomistic program."""

    name: eqx.AbstractVar[str]
    force_group: eqx.AbstractVar[int]
    term_id: eqx.AbstractVar[str]
    capabilities: eqx.AbstractVar[AtomisticPotentialCapabilities]
    requirements: eqx.AbstractVar[AtomisticPotentialRequirements]

    @abc.abstractmethod
    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "AbstractPreparedAtomisticEnergyTerm":
        raise NotImplementedError


class AbstractPreparedAtomisticEnergyTerm(StrictModule):
    name: eqx.AbstractVar[str]
    force_group: eqx.AbstractVar[int]
    term_id: eqx.AbstractVar[str]
    prepared_id: eqx.AbstractVar[str]
    capabilities: eqx.AbstractVar[AtomisticPotentialCapabilities]
    requirements: eqx.AbstractVar[AtomisticPotentialRequirements]

    @abc.abstractmethod
    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        raise NotImplementedError


class LearnedGraphPotentialTerm(AbstractAtomisticEnergyTerm):
    """Bind a trained graph scalar-energy model into a dynamics potential program."""

    potential: AbstractAtomisticPotential
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    allow_periodic: bool = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        potential: AbstractAtomisticPotential,
        /,
        *,
        name: str = "learned",
        force_group: int = 0,
        allow_periodic: bool = False,
    ):
        if not isinstance(potential, AbstractAtomisticPotential):
            raise TypeError("potential must implement AbstractAtomisticPotential.")
        identifier = str(name).strip()
        group = int(force_group)
        if not identifier or group < 0:
            raise ValueError(
                "Potential term name must be non-empty and force_group non-negative."
            )
        capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True,
            finite_geometry=True,
            orthorhombic_periodic=allow_periodic,
            triclinic_periodic=allow_periodic,
            cell_derivative=False,
            local_energy=True,
        )
        requirements = AtomisticPotentialRequirements(
            cutoff=float(potential.configuration.cutoff),
            pair_geometry=True,
            directed_graph=True,
        )
        self.potential = potential
        self.name = identifier
        self.force_group = group
        self.allow_periodic = bool(allow_periodic)
        self.capabilities = capabilities
        self.requirements = requirements
        self.term_id = canonical_fingerprint(
            {
                "kind": "learned-graph-potential-term",
                "potential": potential.potential_id,
                "name": identifier,
                "force_group": group,
                "allow_periodic": bool(allow_periodic),
                "capabilities": capabilities.capabilities_id,
                "requirements": requirements.requirements_id,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedLearnedGraphPotentialTerm":
        if system.plan.units.scale.scale_id != self.potential.scale.scale_id:
            raise ValueError("Learned potential and atomistic system scales differ.")
        if (
            self.potential.capabilities.species_kind is AtomisticSpeciesKind.ATOMIC_NUMBER
            and not bool(jnp.all(system.plan.element_mask | ~system.active_mask))
        ):
            raise ValueError(
                "Atomic-number learned potentials require element particles."
            )
        if system.cell is not None and not self.allow_periodic:
            raise ValueError(
                "This learned potential term has no periodic execution capability."
            )
        return PreparedLearnedGraphPotentialTerm(self, system)


class PreparedLearnedGraphPotentialTerm(AbstractPreparedAtomisticEnergyTerm):
    plan: LearnedGraphPotentialTerm
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self, plan: LearnedGraphPotentialTerm, system: PreparedAtomisticSystem, /
    ):
        self.plan = plan
        self.system = system
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_id = plan.term_id
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-learned-graph-potential-term",
                "term": plan.term_id,
                "system": system.prepared_id,
            }
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        if context.graph is None:
            raise ValueError("Learned graph potential requires a directed graph context.")
        atom_cases = jnp.zeros((self.system.capacity,), dtype=jnp.int32)
        species = (
            self.system.plan.atomic_numbers
            if self.plan.potential.capabilities.species_kind
            is AtomisticSpeciesKind.ATOMIC_NUMBER
            else context.species
        )
        energy, atom_energy = self.plan.potential.graph_energy(
            species,
            self.system.active_mask,
            atom_cases,
            1,
            self.system.capacity,
            context.graph,
        )
        successful = context.graph.valid[0] & jnp.all(jnp.isfinite(energy))
        return AtomisticTermEvaluation(energy[0], atom_energy[0], successful)


class AtomisticPotentialProgram(StrictModule):
    """Ordered additive scalar-energy program with merged spatial requirements."""

    terms: tuple[AbstractAtomisticEnergyTerm, ...]
    coefficients: Array
    term_names: tuple[str, ...] = eqx.field(static=True)
    force_groups: tuple[int, ...] = eqx.field(static=True)
    requirements: AtomisticPotentialRequirements
    capabilities: AtomisticPotentialCapabilities
    program_id: str = eqx.field(static=True)

    def __init__(
        self,
        terms: Sequence[AbstractAtomisticEnergyTerm],
        /,
        *,
        coefficients: ArrayLike | None = None,
    ):
        values = tuple(terms)
        if not values or any(
            not isinstance(value, AbstractAtomisticEnergyTerm) for value in values
        ):
            raise TypeError(
                "terms must be a non-empty sequence of atomistic energy terms."
            )
        names = tuple(value.name for value in values)
        if len(set(names)) != len(names):
            raise ValueError("Potential program term names must be unique.")
        weights = (
            np.ones((len(values),), dtype=np.float64)
            if coefficients is None
            else np.asarray(coefficients, dtype=np.float64)
        )
        if weights.shape != (len(values),) or np.any(~np.isfinite(weights)):
            raise ValueError("coefficients must be finite with one value per term.")
        cutoffs = tuple(
            value.requirements.cutoff
            for value in values
            if value.requirements.cutoff is not None
        )
        requirements = AtomisticPotentialRequirements(
            cutoff=None if not cutoffs else max(cutoffs),
            pair_geometry=any(value.requirements.pair_geometry for value in values),
            interaction_site_geometry=any(
                value.requirements.interaction_site_geometry for value in values
            ),
            directed_graph=any(value.requirements.directed_graph for value in values),
            bonded_geometry=any(value.requirements.bonded_geometry for value in values),
            reciprocal_grid=any(value.requirements.reciprocal_grid for value in values),
        )
        capabilities = AtomisticPotentialCapabilities(
            conservative_energy=all(
                value.capabilities.conservative_energy for value in values
            ),
            finite_geometry=all(value.capabilities.finite_geometry for value in values),
            orthorhombic_periodic=all(
                value.capabilities.orthorhombic_periodic for value in values
            ),
            triclinic_periodic=all(
                value.capabilities.triclinic_periodic for value in values
            ),
            cell_derivative=all(value.capabilities.cell_derivative for value in values),
            local_energy=all(value.capabilities.local_energy for value in values),
            local_energy_delta=all(
                value.capabilities.local_energy_delta for value in values
            ),
            dynamic_species=all(value.capabilities.dynamic_species for value in values),
            species_kind=(
                AtomisticSpeciesKind.ATOM_TYPE_ID
                if any(
                    value.capabilities.species_kind is AtomisticSpeciesKind.ATOM_TYPE_ID
                    for value in values
                )
                else AtomisticSpeciesKind.ATOMIC_NUMBER
            ),
        )
        self.terms = values
        self.coefficients = jnp.asarray(weights)
        self.term_names = names
        self.force_groups = tuple(value.force_group for value in values)
        self.requirements = requirements
        self.capabilities = capabilities
        self.program_id = canonical_fingerprint(
            {
                "kind": "atomistic-potential-program",
                "terms": [value.term_id for value in values],
                "coefficients": weights.tolist(),
                "requirements": requirements.requirements_id,
                "capabilities": capabilities.capabilities_id,
            }
        )

    def prepare(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        graph_execution: AtomisticGraphExecutionPlan | None = None,
    ) -> "PreparedAtomisticPotentialProgram":
        return PreparedAtomisticPotentialProgram(
            self, system, graph_execution=graph_execution
        )


class AtomisticPotentialEvaluation(StrictModule):
    energy: Array
    term_energies: Array
    atom_energy: Array
    forces: Array
    virial: Array
    successful: Array
    neighborhood_successful: Array
    graph_overflow: Array
    program_id: str = eqx.field(static=True)


class AbstractPreparedAtomisticHamiltonian(StrictModule):
    """Prepared scalar Hamiltonian boundary shared by fixed and controlled programs."""

    system: eqx.AbstractVar[PreparedAtomisticSystem]
    plan: eqx.AbstractVar[Any]
    terms: eqx.AbstractVar[tuple[AbstractPreparedAtomisticEnergyTerm, ...]]
    prepared_id: eqx.AbstractVar[str]
    control_ids: eqx.AbstractVar[tuple[str, ...]]
    control_layout_id: eqx.AbstractVar[str]
    coefficients: eqx.AbstractVar[Array]

    @abc.abstractmethod
    def context(
        self,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState,
        /,
        **kwargs: Any,
    ) -> AtomisticPotentialContext:
        raise NotImplementedError

    @abc.abstractmethod
    def energy(
        self,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState,
        /,
        **kwargs: Any,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState,
        /,
        **kwargs: Any,
    ) -> AtomisticPotentialEvaluation:
        raise NotImplementedError


class PreparedAtomisticPotentialProgram(AbstractPreparedAtomisticHamiltonian):
    plan: AtomisticPotentialProgram
    system: PreparedAtomisticSystem
    terms: tuple[AbstractPreparedAtomisticEnergyTerm, ...]
    graph_execution: AtomisticGraphExecutionPlan | None
    prepared_id: str = eqx.field(static=True)
    control_ids: tuple[str, ...] = eqx.field(static=True)
    control_layout_id: str = eqx.field(static=True)
    coefficients: Array

    def __init__(
        self,
        plan: AtomisticPotentialProgram,
        system: PreparedAtomisticSystem,
        /,
        *,
        graph_execution: AtomisticGraphExecutionPlan | None,
    ):
        if not isinstance(plan, AtomisticPotentialProgram):
            raise TypeError("plan must be an AtomisticPotentialProgram.")
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("system must be a PreparedAtomisticSystem.")
        if plan.requirements.directed_graph:
            if (
                not isinstance(graph_execution, AtomisticGraphExecutionPlan)
                or graph_execution.backend != "particle"
            ):
                raise ValueError(
                    "A particle AtomisticGraphExecutionPlan is required by learned terms."
                )
        elif graph_execution is not None:
            raise ValueError("graph_execution was supplied but no term requires a graph.")
        cutoff = plan.requirements.cutoff
        if cutoff is not None and system.cell is not None:
            system.cell.require_unique_image(cutoff)
        terms = tuple(value.prepare(system) for value in plan.terms)
        self.plan = plan
        self.system = system
        self.terms = terms
        self.coefficients = plan.coefficients
        self.graph_execution = graph_execution
        self.control_ids = ()
        self.control_layout_id = canonical_fingerprint(
            {"kind": "atomistic-control-layout", "control_ids": []}
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-atomistic-potential-program",
                "program": plan.program_id,
                "system": system.prepared_id,
                "terms": [value.prepared_id for value in terms],
                "graph_execution": (
                    None if graph_execution is None else graph_execution.plan_id
                ),
            }
        )

    def context(
        self,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState,
        /,
        *,
        unwrapped_positions: ArrayLike | None = None,
        species: ArrayLike | None = None,
        interaction_scales: AtomisticInteractionScaleState | None = None,
        cell: PeriodicCell | None = None,
        fractional_positions: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
    ) -> AtomisticPotentialContext:
        if not isinstance(neighborhood, ParticleNeighborhoodState):
            raise TypeError("neighborhood must be a ParticleNeighborhoodState.")
        position = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        expected = (self.system.capacity, 3)
        if position.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        unwrapped = (
            position
            if unwrapped_positions is None
            else jnp.asarray(unwrapped_positions, dtype=position.dtype)
        )
        if unwrapped.shape != expected:
            raise ValueError(f"unwrapped_positions must have shape {expected}.")
        species_ = (
            self.system.plan.atom_type_ids
            if species is None
            else jnp.asarray(species, dtype=jnp.int32)
        )
        if species_.shape != (self.system.capacity,):
            raise ValueError("species must match the particle capacity.")
        selected_cell = self.system.cell if cell is None else cell
        if cell_vectors is not None and (
            selected_cell is None or fractional_positions is None
        ):
            raise ValueError(
                "Dynamic cell vectors require a cell and fractional_positions."
            )
        site_state = self.system.coordinate_map.realize(
            position,
            cell=selected_cell,
            fractional_positions=fractional_positions
            if cell_vectors is not None
            else None,
            cell_vectors=cell_vectors,
        )
        site_left = self.system.coordinate_map.pair_left
        site_right = self.system.coordinate_map.pair_right
        site_valid = (
            self.system.coordinate_map.plan.sites.active_mask[site_left]
            & self.system.coordinate_map.plan.sites.active_mask[site_right]
        )
        if self.plan.requirements.interaction_site_geometry:
            site_ids = self.system.coordinate_map.plan.sites.site_ids
            site_left_ids = site_ids[site_left]
            site_right_ids = site_ids[site_right]
            site_zeros = jnp.zeros_like(site_left_ids)
            site_keys = jnp.stack(
                (
                    site_zeros,
                    jnp.minimum(site_left_ids, site_right_ids),
                    jnp.maximum(site_left_ids, site_right_ids),
                    site_zeros,
                    site_zeros,
                ),
                axis=-1,
            )
            site_lj_scales, site_electrostatic_scales = self.system.topology.pair_scales(
                site_keys
            )
        else:
            site_lj_scales = jnp.ones(site_left.shape, dtype=position.dtype)
            site_electrostatic_scales = jnp.ones_like(site_lj_scales)
        site_displacement = (
            site_state.positions[site_left] - site_state.positions[site_right]
        )
        if cell_vectors is not None:
            vectors = jnp.asarray(cell_vectors, dtype=position.dtype)
            determinant = jnp.sum(vectors[0] * jnp.cross(vectors[1], vectors[2]))
            inverse = (
                jnp.stack(
                    (
                        jnp.cross(vectors[1], vectors[2]),
                        jnp.cross(vectors[2], vectors[0]),
                        jnp.cross(vectors[0], vectors[1]),
                    ),
                    axis=1,
                )
                / determinant
            )
            site_fractional = contract("nd,di->ni", site_state.positions, inverse)
            fractional_displacement = (
                site_fractional[site_left] - site_fractional[site_right]
            )
            central = jax.lax.stop_gradient(
                jnp.round(fractional_displacement).astype(jnp.int32)
            )
            central = jnp.where(selected_cell.periodic_mask, central, 0)
            site_displacement = contract(
                "ni,ij->nj",
                fractional_displacement - central.astype(position.dtype),
                vectors,
            )
        elif selected_cell is not None:
            site_displacement = selected_cell.minimum_image(site_displacement)
        site_displacement = jnp.where(site_valid[:, None], site_displacement, 0.0)
        site_squared = jnp.sum(site_displacement * site_displacement, axis=-1)
        site_tiny = jnp.asarray(jnp.finfo(position.dtype).tiny, dtype=position.dtype)
        site_distance = jnp.where(
            site_squared > 0.0,
            jnp.sqrt(jnp.maximum(site_squared, site_tiny)),
            0.0,
        )
        pairs = neighborhood.pair_relation
        left = pairs.left_indices
        right = pairs.right_indices
        pair_valid = pairs.valid
        displacement = position[left] - position[right]
        if cell_vectors is not None:
            fractional = jnp.asarray(fractional_positions, dtype=position.dtype)
            if fractional.shape != expected:
                raise ValueError(f"fractional_positions must have shape {expected}.")
            vectors = jnp.asarray(cell_vectors, dtype=position.dtype)
            if vectors.shape != (3, 3):
                raise ValueError("cell_vectors must have shape (3, 3).")
            fractional_displacement = fractional[left] - fractional[right]
            central = jax.lax.stop_gradient(
                jnp.round(fractional_displacement).astype(jnp.int32)
            )
            central = jnp.where(selected_cell.periodic_mask, central, 0)
            centered = fractional_displacement - central.astype(position.dtype)
            candidates_fractional = (
                centered[:, None, :]
                - selected_cell.image_shifts.astype(position.dtype)[None, :, :]
            )
            candidates = contract("nsi,ij->nsj", candidates_fractional, vectors)
            selected = jax.lax.stop_gradient(
                jnp.argmin(jnp.sum(candidates * candidates, axis=-1), axis=-1)
            )
            displacement = jnp.take_along_axis(
                candidates, selected[:, None, None], axis=1
            )[:, 0, :]
        elif selected_cell is not None:
            displacement = selected_cell.minimum_image(displacement)
        displacement = jnp.where(pair_valid[:, None], displacement, 0.0)
        squared = jnp.sum(displacement * displacement, axis=-1)
        tiny = jnp.asarray(jnp.finfo(position.dtype).tiny, dtype=position.dtype)
        distance = jnp.where(squared > 0.0, jnp.sqrt(jnp.maximum(squared, tiny)), 0.0)
        keys = self.system.pair_key_space.keys(pairs)
        lj_scales, electrostatic_scales = self.system.topology.pair_scales(keys.keys)
        graph = None
        graph_overflow = jnp.asarray(False)
        if self.plan.requirements.directed_graph:
            execution = self.graph_execution
            if execution is None:
                raise RuntimeError("Validated graph execution unexpectedly absent.")
            if cell_vectors is not None:
                raise ValueError(
                    "Dynamic cell graph geometry is not supported by this graph potential."
                )
            graph = realize_particle_atomistic_graph(
                self.system,
                neighborhood,
                execution,
                position,
                cutoff=float(self.plan.requirements.cutoff),
                cell=selected_cell,
            )
            graph_overflow = jnp.any(graph.overflow)
        resolved_cell_vectors = (
            jnp.zeros((0, 0), dtype=position.dtype)
            if selected_cell is None
            else selected_cell.vectors.astype(position.dtype)
            if cell_vectors is None
            else jnp.asarray(cell_vectors, dtype=position.dtype)
        )
        scales = (
            AtomisticInteractionScaleState.identity(
                self.system, left.shape[0], position.dtype
            )
            if interaction_scales is None
            else interaction_scales
        )
        if not isinstance(scales, AtomisticInteractionScaleState):
            raise TypeError(
                "interaction_scales must be AtomisticInteractionScaleState or None."
            )
        expected_scale_shapes = {
            "bond": (self.system.topology.bond_indices.shape[0],),
            "angle": (self.system.topology.angle_indices.shape[0],),
            "torsion": (self.system.topology.torsion_indices.shape[0],),
            "improper": (self.system.topology.improper_indices.shape[0],),
            "lennard_jones": left.shape,
            "electrostatic": left.shape,
        }
        scale_values = (
            scales.bond,
            scales.angle,
            scales.torsion,
            scales.improper,
            scales.lennard_jones,
            scales.electrostatic,
        )
        for (field, shape), value in zip(
            expected_scale_shapes.items(), scale_values, strict=True
        ):
            if value.shape != shape:
                raise ValueError(f"interaction_scales.{field} must have shape {shape}.")
        if (
            scales.lennard_jones_softcore_alpha.shape != ()
            or scales.electrostatic_softcore_alpha.shape != ()
            or scales.softcore_power < 1
        ):
            raise ValueError("Soft-core interaction scales are invalid.")
        return AtomisticPotentialContext(
            system=self.system,
            site_state=site_state,
            site_positions=site_state.positions,
            site_type_ids=self.system.coordinate_map.plan.sites.site_type_ids,
            site_charges=self.system.coordinate_map.plan.sites.charges,
            site_pair_left=site_left,
            site_pair_right=site_right,
            site_pair_valid=site_valid,
            site_pair_displacement=site_displacement,
            site_pair_distance=site_distance,
            site_lennard_jones_scales=site_lj_scales,
            site_electrostatic_scales=site_electrostatic_scales,
            positions=position,
            unwrapped_positions=unwrapped,
            species=species_,
            pair_left=left,
            pair_right=right,
            pair_valid=pair_valid,
            pair_keys=keys.keys,
            pair_displacement=displacement,
            pair_distance=distance,
            lennard_jones_scales=lj_scales,
            electrostatic_scales=electrostatic_scales,
            interaction_scales=scales,
            graph=graph,
            cell=selected_cell,
            cell_vectors=resolved_cell_vectors,
            neighborhood_successful=(
                neighborhood.successful
                & keys.successful
                & ~graph_overflow
                & site_state.successful
            ),
        )

    def energy(
        self,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState,
        /,
        **context_kwargs: Any,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        context = self.context(positions, neighborhood, **context_kwargs)
        evaluations = tuple(term.energy(context) for term in self.terms)
        term_energies = jnp.stack(tuple(value.energy for value in evaluations))
        coefficients = self.plan.coefficients.astype(term_energies.dtype)
        energy = jnp.sum(coefficients * term_energies)
        atom_energy = jnp.sum(
            coefficients[:, None]
            * jnp.stack(tuple(value.atom_energy for value in evaluations)),
            axis=0,
        )
        successful = context.neighborhood_successful & jnp.all(
            jnp.stack(tuple(value.successful for value in evaluations))
        )
        graph_overflow = (
            jnp.asarray(False)
            if context.graph is None
            else jnp.any(context.graph.overflow)
        )
        return energy, (term_energies, atom_energy, successful, graph_overflow)

    def evaluate(
        self,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState,
        /,
        **context_kwargs: Any,
    ) -> AtomisticPotentialEvaluation:
        """Differentiate Cartesian DOFs, retaining fixed coordinate-image choices."""
        position = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        selected_cell = context_kwargs.get("cell")
        if selected_cell is None:
            selected_cell = self.system.cell
        vectors = context_kwargs.get("cell_vectors")
        unwrapped = context_kwargs.get("unwrapped_positions")
        fractional = context_kwargs.get("fractional_positions")
        unwrapped_offset = None
        if unwrapped is not None:
            offset = jnp.asarray(unwrapped, dtype=position.dtype) - position
            if selected_cell is None:
                unwrapped_offset = jax.lax.stop_gradient(offset)
            else:
                image_vectors = (
                    selected_cell.vectors if vectors is None else jnp.asarray(vectors)
                ).astype(position.dtype)
                images = jax.lax.stop_gradient(
                    jnp.where(
                        selected_cell.periodic_mask,
                        jnp.round(
                            contract(
                                "ni,ij->nj",
                                offset,
                                selected_cell.inverse_for_vectors(image_vectors),
                            )
                        ),
                        0.0,
                    )
                )
                translation = contract("ni,ij->nj", images, image_vectors)
                unwrapped_offset = translation + jax.lax.stop_gradient(
                    offset - translation
                )
        fractional_offset = None
        if fractional is not None and vectors is not None and selected_cell is not None:
            fractional_offset = jax.lax.stop_gradient(
                jnp.asarray(fractional, dtype=position.dtype)
                - selected_cell.fractional_with_vectors(position, vectors)
            )

        def closure(value: Array):
            kwargs = dict(context_kwargs)
            if unwrapped_offset is not None:
                kwargs["unwrapped_positions"] = value + unwrapped_offset
            if fractional_offset is not None:
                kwargs["fractional_positions"] = (
                    selected_cell.fractional_with_vectors(value, vectors)
                    + fractional_offset
                )
            return self.energy(value, neighborhood, **kwargs)

        (energy, auxiliary), gradient = jax.value_and_grad(closure, has_aux=True)(
            position
        )
        term_energies, atom_energy, successful, graph_overflow = auxiliary
        forces = -gradient
        active = self.system.active_mask[:, None]
        forces = jnp.where(active, forces, 0.0)
        center = jnp.sum(
            jnp.where(
                active, self.system.plan.masses[:, None] * jnp.asarray(positions), 0.0
            ),
            axis=0,
        ) / jnp.sum(jnp.where(self.system.active_mask, self.system.plan.masses, 0.0))
        lever = jnp.asarray(positions) - center
        virial = -contract("ni,nj->ij", lever, forces)
        finite = (
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(term_energies))
            & jnp.all(jnp.isfinite(forces))
        )
        accepted = successful & finite
        nan = jnp.asarray(jnp.nan, dtype=energy.dtype)
        return AtomisticPotentialEvaluation(
            energy=jnp.where(accepted, energy, nan),
            term_energies=jnp.where(accepted, term_energies, nan),
            atom_energy=jnp.where(accepted, atom_energy, nan),
            forces=jnp.where(accepted, forces, nan),
            virial=jnp.where(accepted, virial, nan),
            successful=accepted,
            neighborhood_successful=successful,
            graph_overflow=graph_overflow,
            program_id=self.prepared_id,
        )


__all__ = [
    "AbstractPreparedAtomisticHamiltonian",
    "AtomisticInteractionScaleState",
    "AbstractAtomisticEnergyTerm",
    "AbstractPreparedAtomisticEnergyTerm",
    "AtomisticPotentialContext",
    "AtomisticPotentialEvaluation",
    "AtomisticPotentialProgram",
    "AtomisticTermEvaluation",
    "LearnedGraphPotentialTerm",
    "PreparedAtomisticPotentialProgram",
]
