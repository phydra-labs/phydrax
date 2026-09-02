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
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
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
)
from ._sites import AtomisticInteractionSiteState
from ._system import PreparedAtomisticSystem


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
    graph: AtomisticGraph | None
    cell: PeriodicCell | None
    cell_vectors: Array
    alchemical_lambda: Array
    neighborhood_successful: Array


class AtomisticTermEvaluation(StrictModule):
    energy: Array
    atom_energy: Array
    successful: Array


class AbstractAtomisticEnergyTerm(StrictModule):
    """One composable scalar-energy term in a prepared atomistic program."""

    name: AbstractAttribute[str]
    force_group: AbstractAttribute[int]
    term_id: AbstractAttribute[str]
    capabilities: AbstractAttribute[AtomisticPotentialCapabilities]
    requirements: AbstractAttribute[AtomisticPotentialRequirements]

    @abc.abstractmethod
    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "AbstractPreparedAtomisticEnergyTerm":
        raise NotImplementedError


class AbstractPreparedAtomisticEnergyTerm(StrictModule):
    name: AbstractAttribute[str]
    force_group: AbstractAttribute[int]
    term_id: AbstractAttribute[str]
    prepared_id: AbstractAttribute[str]
    capabilities: AbstractAttribute[AtomisticPotentialCapabilities]
    requirements: AbstractAttribute[AtomisticPotentialRequirements]

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
        if not np.array_equal(
            np.asarray(system.plan.atom_type_ids),
            np.asarray(system.plan.atomic_numbers),
        ):
            raise ValueError(
                "Learned graph execution requires atom_type_ids to equal atomic numbers."
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
        energy, atom_energy = self.plan.potential.graph_energy(
            context.species,
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
            np.ones((len(values),), dtype=float)
            if coefficients is None
            else np.asarray(coefficients, dtype=float)
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


class PreparedAtomisticPotentialProgram(StrictModule):
    plan: AtomisticPotentialProgram
    system: PreparedAtomisticSystem
    terms: tuple[AbstractPreparedAtomisticEnergyTerm, ...]
    graph_execution: AtomisticGraphExecutionPlan | None
    prepared_id: str = eqx.field(static=True)

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
        self.graph_execution = graph_execution
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
        alchemical_lambda: ArrayLike = 1.0,
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
            graph=graph,
            cell=selected_cell,
            cell_vectors=resolved_cell_vectors,
            alchemical_lambda=jnp.asarray(
                alchemical_lambda, dtype=position.dtype
            ).reshape(()),
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
        def closure(value: Array):
            return self.energy(value, neighborhood, **context_kwargs)

        (energy, auxiliary), gradient = jax.value_and_grad(closure, has_aux=True)(
            jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
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
    "AbstractAtomisticEnergyTerm",
    "AbstractPreparedAtomisticEnergyTerm",
    "AtomisticPotentialContext",
    "AtomisticPotentialEvaluation",
    "AtomisticPotentialProgram",
    "AtomisticTermEvaluation",
    "LearnedGraphPotentialTerm",
    "PreparedAtomisticPotentialProgram",
]
