#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import (
    ParticleImageNeighborhoodState,
    ParticleNeighborhoodState,
    PeriodicCell,
)
from ..linalg import determinant_small_linear, SmallLinearSolvePlan
from ..typing import checked
from ._graph import (
    admit_supplied_topology,
    AtomisticGraph,
    AtomisticGraphExecutionPlan,
    AtomisticGraphTopology,
    bind_atomistic_graph,
    particle_atomistic_graph_topology,
)
from ._potential import (
    AbstractAtomisticPotential,
    atomistic_potential_revision,
    AtomisticPotentialCapabilities,
    AtomisticPotentialRequirements,
    AtomisticSpeciesKind,
    AtomisticStressConvention,
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
        dtype: DTypeLike,
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
    """Bind a trained graph scalar-energy model into a dynamics potential program.

    The term consumes only the model's declared directed graph. Periodic
    execution binds fixed integer image routes to runtime cell vectors, so its
    energy is differentiable in homogeneous cell strain; that is an execution
    capability, not evidence that the fitted model is stable for dynamics.
    """

    potential: AbstractAtomisticPotential
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    allow_periodic: bool = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    @checked
    def __init__(
        self,
        potential: AbstractAtomisticPotential,
        /,
        *,
        name: str = "learned",
        force_group: int = 0,
        allow_periodic: bool = False,
    ) -> None:
        identifier = str(name).strip()
        group = int(force_group)
        if not identifier or group < 0:
            raise ValueError(
                "Potential term name must be non-empty and force_group non-negative."
            )
        periodic = bool(allow_periodic)
        declared = potential.capabilities
        if periodic and not (
            declared.orthorhombic_periodic or declared.triclinic_periodic
        ):
            raise ValueError(
                "allow_periodic requires a learned potential that declares periodic "
                "execution capability."
            )
        capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True,
            finite_geometry=True,
            orthorhombic_periodic=periodic and declared.orthorhombic_periodic,
            triclinic_periodic=periodic and declared.triclinic_periodic,
            cell_derivative=periodic and declared.cell_derivative,
            local_energy=True,
        )
        requirements = potential.requirements
        if not requirements.directed_graph or requirements.cutoff is None:
            raise ValueError(
                "Learned graph potentials must declare a directed graph and cutoff."
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
                "potential_revision": atomistic_potential_revision(potential).revision_id,
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
        if system.cell is not None:
            if not self.allow_periodic:
                raise ValueError(
                    "This learned potential term has no periodic execution capability."
                )
            vectors = np.asarray(system.cell.vectors)
            axis_aligned = vectors.shape == (3, 3) and np.array_equal(
                vectors, np.diag(np.diag(vectors))
            )
            if not (
                self.capabilities.triclinic_periodic
                or (axis_aligned and self.capabilities.orthorhombic_periodic)
            ):
                raise ValueError(
                    "The learned potential does not declare execution for this cell "
                    "geometry."
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
    ) -> None:
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
    ) -> None:
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
    """Energy, conservative forces and optional homogeneous-strain response.

    ``virial`` is the finite position-moment diagnostic ``-sum (r - c) (x) f``
    about the active center of mass; it is not a periodic virial. When stress is
    requested, ``strain_derivative`` is ``dE/dstrain`` from the same scalar
    derivative pass and ``stress`` follows ``stress_convention``; otherwise all
    three are ``None``. Any failed scientific, image, capacity, finite or
    derivative state sets ``successful`` false and every numeric output to NaN.
    """

    energy: Array
    term_energies: Array
    atom_energy: Array
    forces: Array
    virial: Array
    successful: Array
    neighborhood_successful: Array
    graph_overflow: Array
    strain_derivative: Array | None
    stress: Array | None
    program_id: str = eqx.field(static=True)
    stress_convention: AtomisticStressConvention | None = eqx.field(static=True)


class AtomisticHamiltonianEvaluation(Protocol):
    """Evaluation record shared by fixed and controlled prepared Hamiltonians."""

    @property
    def energy(self) -> Array: ...

    @property
    def term_energies(self) -> Array: ...

    @property
    def atom_energy(self) -> Array: ...

    @property
    def forces(self) -> Array: ...

    @property
    def virial(self) -> Array: ...

    @property
    def successful(self) -> Array: ...

    @property
    def neighborhood_successful(self) -> Array: ...

    @property
    def graph_overflow(self) -> Array: ...

    @property
    def strain_derivative(self) -> Array | None: ...

    @property
    def stress(self) -> Array | None: ...

    @property
    def program_id(self) -> str: ...

    @property
    def stress_convention(self) -> AtomisticStressConvention | None: ...


class AtomisticDerivativePoint(StrictModule):
    """Fixed-topology Cartesian and homogeneous-strain chart of one evaluation point.

    Integer images of supplied unwrapped coordinates and the residual of supplied
    fractional coordinates are bound outside differentiation, so no derivative
    moves a route to another image. ``bind(value, strain)`` deforms the point by
    ``F = I + strain`` under ``AtomisticStressConvention``: positions
    ``origin + (r - origin) @ F.T`` and row vectors ``H @ F.T`` at fixed fractional
    coordinates and fixed integer images. The default lattice is the searched
    lattice of a ``ParticleImageNeighborhoodState``, else the prepared cell.
    Stress requires a full 3x3 cell; with nonperiodic axes ``|det H|`` is the
    declared embedding volume, never an invented thickness for a lower-rank
    lattice.
    """

    context_kwargs: dict[str, Any]
    cell: PeriodicCell | None
    vectors: Array | None
    default_vectors: Array | None
    unwrapped_offset: Array | None
    unwrapped_images: Array | None
    unwrapped_residual: Array | None
    fractional_offset: Array | None
    successful: Array
    strained: bool = eqx.field(static=True)

    def __init__(
        self,
        system: PreparedAtomisticSystem,
        position: Array,
        context_kwargs: dict[str, Any],
        /,
        *,
        neighborhood: ParticleNeighborhoodState | ParticleImageNeighborhoodState,
        strained: bool,
        cell_derivative: bool,
    ) -> None:
        kwargs = dict(context_kwargs)
        supplied_cell = kwargs.get("cell")
        if supplied_cell is not None and not isinstance(supplied_cell, PeriodicCell):
            raise TypeError("cell must be a PeriodicCell or None.")
        cell = system.cell if supplied_cell is None else supplied_cell
        if strained:
            if not cell_derivative:
                raise ValueError(
                    "Requested stress requires every potential term to own a cell "
                    "derivative."
                )
            if cell is None or cell.rank != 3 or cell.ambient_dimension != 3:
                raise ValueError(
                    "Requested stress requires a full invertible 3x3 cell; a "
                    "lower-rank or finite system has no physical or embedding volume."
                )
        default_vectors = (
            neighborhood.cell_vectors[0].astype(position.dtype)
            if isinstance(neighborhood, ParticleImageNeighborhoodState)
            else None
            if cell is None
            else cell.vectors.astype(position.dtype)
        )
        supplied_vectors = kwargs.get("cell_vectors")
        vectors = (
            jnp.asarray(supplied_vectors, dtype=position.dtype)
            if supplied_vectors is not None
            else default_vectors
            if strained
            else None
        )
        successful = jnp.asarray(True)
        unwrapped = kwargs.get("unwrapped_positions")
        unwrapped_offset = None
        unwrapped_images = None
        unwrapped_residual = None
        if unwrapped is not None:
            offset = jnp.asarray(unwrapped, dtype=position.dtype) - position
            if cell is None:
                unwrapped_offset = jax.lax.stop_gradient(offset)
            else:
                image_vectors = default_vectors if vectors is None else vectors
                if image_vectors is None:
                    raise RuntimeError(
                        "A periodic derivative point lacks lattice vectors."
                    )
                inverse, inverse_successful = cell.inverse_for_vectors_with_status(
                    image_vectors
                )
                successful = successful & inverse_successful
                unwrapped_images = jax.lax.stop_gradient(
                    jnp.where(
                        cell.periodic_mask,
                        jnp.round(
                            contract("ni,ij->nj", offset, inverse.astype(position.dtype))
                        ),
                        0.0,
                    )
                )
                unwrapped_residual = jax.lax.stop_gradient(
                    offset - contract("ni,ij->nj", unwrapped_images, image_vectors)
                )
        fractional = kwargs.get("fractional_positions")
        fractional_offset = None
        if fractional is not None and vectors is not None and cell is not None:
            derived, derived_successful = cell.fractional_with_vectors_with_status(
                position, vectors
            )
            successful = successful & derived_successful
            fractional_offset = jax.lax.stop_gradient(
                jnp.asarray(fractional, dtype=position.dtype) - derived
            )
        self.context_kwargs = kwargs
        self.cell = cell
        self.vectors = vectors
        self.default_vectors = default_vectors
        self.unwrapped_offset = unwrapped_offset
        self.unwrapped_images = unwrapped_images
        self.unwrapped_residual = unwrapped_residual
        self.fractional_offset = fractional_offset
        self.successful = successful
        self.strained = bool(strained)

    def zero_strain(self) -> Array:
        if self.vectors is None:
            raise RuntimeError("Strain chart requires bound cell vectors.")
        return jnp.zeros((3, 3), dtype=self.vectors.dtype)

    def bind(self, value: Array, strain: Array | None, /) -> tuple[Array, dict[str, Any]]:
        """Return deformed positions and the complete context keywords at ``value``."""
        kwargs = dict(self.context_kwargs)
        position = value
        vectors = self.vectors
        deformation = None
        if strain is not None:
            if self.cell is None or self.vectors is None:
                raise RuntimeError("Strain chart requires a bound periodic cell.")
            deformation = jnp.eye(3, dtype=value.dtype) + strain
            origin = self.cell.origin.astype(value.dtype)
            position = origin + contract("ni,ji->nj", value - origin, deformation)
            vectors = contract("ki,ji->kj", self.vectors, deformation)
        if vectors is not None:
            kwargs["cell_vectors"] = vectors
        if self.unwrapped_offset is not None:
            kwargs["unwrapped_positions"] = position + self.unwrapped_offset
        elif (
            self.cell is not None
            and self.unwrapped_images is not None
            and self.unwrapped_residual is not None
        ):
            image_vectors = self.default_vectors if vectors is None else vectors
            if image_vectors is None:
                raise RuntimeError("Unwrapped image chart requires a bound lattice.")
            residual = (
                self.unwrapped_residual
                if deformation is None
                else contract("ni,ji->nj", self.unwrapped_residual, deformation)
            )
            kwargs["unwrapped_positions"] = (
                position
                + contract("ni,ij->nj", self.unwrapped_images, image_vectors)
                + residual
            )
        if (
            self.fractional_offset is not None
            and self.cell is not None
            and vectors is not None
        ):
            # The context re-admits these same runtime vectors (finite, full
            # rank, conditioned), so the solve status is carried there.
            fractional, _ = self.cell.fractional_with_vectors_with_status(
                position, vectors
            )
            kwargs["fractional_positions"] = fractional + self.fractional_offset
        return position, kwargs

    @property
    def stress_convention(self) -> AtomisticStressConvention | None:
        """Per-result convention of requested stress, ``None`` when not strained."""
        if not self.strained or self.cell is None:
            return None
        return (
            AtomisticStressConvention.CAUCHY_TENSION_POSITIVE
            if self.cell.fully_periodic
            else AtomisticStressConvention.CAUCHY_TENSION_POSITIVE_EMBEDDING_VOLUME
        )

    def stress(self, strain_derivative: Array, /) -> Array:
        """Map ``dE/dstrain`` to the named tension-positive Cauchy stress."""
        if self.vectors is None:
            raise RuntimeError("Stress requires bound cell vectors.")
        volume = jnp.abs(determinant_small_linear(_CELL_DETERMINANT, self.vectors))
        return 0.5 * (strain_derivative + strain_derivative.T) / volume

    def moment_virial(
        self, system: PreparedAtomisticSystem, position: Array, forces: Array, /
    ) -> Array:
        """Finite center-of-mass position-moment diagnostic, never a periodic virial."""
        active = system.active_mask[:, None]
        center = jnp.sum(
            jnp.where(active, system.plan.masses[:, None] * position, 0.0), axis=0
        ) / jnp.sum(jnp.where(system.active_mask, system.plan.masses, 0.0))
        return -contract("ni,nj->ij", position - center, forces)


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
        neighborhood: ParticleNeighborhoodState | ParticleImageNeighborhoodState,
        /,
        *,
        unwrapped_positions: ArrayLike | None = None,
        species: ArrayLike | None = None,
        cell: PeriodicCell | None = None,
        fractional_positions: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
        topology: AtomisticGraphTopology | None = None,
    ) -> AtomisticPotentialContext:
        raise NotImplementedError

    @abc.abstractmethod
    def energy(
        self,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState | ParticleImageNeighborhoodState,
        /,
        **kwargs: Any,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState | ParticleImageNeighborhoodState,
        /,
        *,
        compute_stress: bool = False,
        **kwargs: Any,
    ) -> AtomisticHamiltonianEvaluation:
        raise NotImplementedError


_CELL_DETERMINANT = SmallLinearSolvePlan(3)


def _stencil_complete(cell: PeriodicCell, radius: Array, reach: Array, /) -> Array:
    """Whether every image within ``radius`` lies in the prepared stencil.

    For ``d = (s + n) @ H`` with ``|s_i| <= 1/2`` on periodic axes,
    ``|n_i| <= 1/2 + |d| * reach_i`` where ``reach_i`` is the norm of column ``i``
    of the runtime right inverse, so integer translations reaching ``radius``
    satisfy ``|n_i| <= image_extent`` when ``1/2 + radius * reach_i`` is below
    ``image_extent + 1``.
    """
    required = 0.5 + radius[..., None] * reach
    return jnp.all(~cell.periodic_mask | (required < cell.image_extent + 1.0), axis=-1)


def _runtime_lattice_admissible(
    cell: PeriodicCell,
    vectors: Array,
    inverse: Array,
    unique_image_radius: float | None,
    /,
) -> Array:
    """Traced, LAPACK-free admission of runtime lattice vectors.

    Pair-once classical routes require every nonzero periodic translation to
    exceed twice ``unique_image_radius``. The shortest periodic translation is
    no longer than the shortest periodic lattice row, so it lies in the prepared
    stencil whenever that row length passes ``_stencil_complete``; the stencil
    minimum is then the true shortest translation.
    """
    if unique_image_radius is None:
        return jnp.asarray(True)
    fixed = jax.lax.stop_gradient(vectors)
    reach = jnp.sqrt(jnp.sum(jax.lax.stop_gradient(inverse) ** 2, axis=0))
    row_lengths = jnp.sqrt(jnp.sum(fixed * fixed, axis=-1))
    shortest_row = jnp.min(jnp.where(cell.periodic_mask, row_lengths, jnp.inf))
    translations = contract("si,ij->sj", cell.image_shifts.astype(fixed.dtype), fixed)
    lengths = jnp.sqrt(jnp.sum(translations * translations, axis=-1))
    nonzero = jnp.any(cell.image_shifts != 0, axis=-1)
    shortest = jnp.min(jnp.where(nonzero, lengths, jnp.inf))
    return _stencil_complete(cell, shortest_row, reach) & (
        0.5 * shortest > jnp.asarray(unique_image_radius, dtype=fixed.dtype)
    )


class _RuntimeLattice(StrictModule):
    """Validated cell binding of one context call."""

    cell: PeriodicCell | None
    vectors: Array | None
    inverse: Array | None
    fractional: Array | None
    successful: Array


class _SiteGeometry(StrictModule):
    state: AtomisticInteractionSiteState
    left: Array
    right: Array
    valid: Array
    displacement: Array
    distance: Array
    lennard_jones_scales: Array
    electrostatic_scales: Array


class _PairGeometry(StrictModule):
    left: Array
    right: Array
    valid: Array
    keys: Array
    displacement: Array
    distance: Array
    lennard_jones_scales: Array
    electrostatic_scales: Array
    successful: Array


def _distance(displacement: Array, /) -> Array:
    squared = jnp.sum(displacement * displacement, axis=-1)
    tiny = jnp.asarray(jnp.finfo(displacement.dtype).tiny, dtype=displacement.dtype)
    return jnp.where(squared > 0.0, jnp.sqrt(jnp.maximum(squared, tiny)), 0.0)


class PreparedAtomisticPotentialProgram(AbstractPreparedAtomisticHamiltonian):
    plan: AtomisticPotentialProgram
    system: PreparedAtomisticSystem
    terms: tuple[AbstractPreparedAtomisticEnergyTerm, ...]
    graph_execution: AtomisticGraphExecutionPlan | None
    prepared_id: str = eqx.field(static=True)
    control_ids: tuple[str, ...] = eqx.field(static=True)
    control_layout_id: str = eqx.field(static=True)
    coefficients: Array

    @checked
    def __init__(
        self,
        plan: AtomisticPotentialProgram,
        system: PreparedAtomisticSystem,
        /,
        *,
        graph_execution: AtomisticGraphExecutionPlan | None,
    ) -> None:
        if plan.requirements.directed_graph:
            if (
                not isinstance(graph_execution, AtomisticGraphExecutionPlan)
                or graph_execution.backend != "particle"
            ):
                raise ValueError(
                    "A particle AtomisticGraphExecutionPlan is required by learned terms."
                )
            if plan.requirements.cutoff is None:
                raise ValueError("Directed atomistic graph terms require a cutoff.")
        elif graph_execution is not None:
            raise ValueError("graph_execution was supplied but no term requires a graph.")
        # Pair-once classical geometry owns its unique-image guard here; a directed
        # graph term is guarded where a pair-once neighborhood realizes its graph.
        classical_cutoffs = tuple(
            term.requirements.cutoff
            for term in plan.terms
            if term.requirements.cutoff is not None
            and (
                term.requirements.pair_geometry
                or term.requirements.interaction_site_geometry
            )
        )
        if classical_cutoffs and system.cell is not None:
            system.cell.require_unique_image(max(classical_cutoffs))
        active_virtual_sites = np.asarray(
            system.coordinate_map.plan.sites.active_mask
        ) & ~np.asarray(system.coordinate_map.plan.sites.physical_mask)
        if (
            plan.requirements.interaction_site_geometry
            and np.any(active_virtual_sites)
            and system.topology.exception_keys.shape[0] > 0
        ):
            raise ValueError(
                "Virtual interaction sites require an explicit site-space pair-exception contract."
            )
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

    @checked
    def context(
        self,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState | ParticleImageNeighborhoodState,
        /,
        *,
        unwrapped_positions: ArrayLike | None = None,
        species: ArrayLike | None = None,
        interaction_scales: AtomisticInteractionScaleState | None = None,
        cell: PeriodicCell | None = None,
        fractional_positions: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
        topology: AtomisticGraphTopology | None = None,
    ) -> AtomisticPotentialContext:
        """Bind one fixed neighborhood topology to runtime coordinates and cell.

        A ``ParticleImageNeighborhoodState`` carries directed image routes with
        fixed integer translations; it is admitted only for programs without
        pair-once classical geometry, whose pair arrays are then empty. Runtime
        ``cell_vectors`` may be any full-rank lattice inside the cell condition
        certificate; pair-once neighborhoods also require the runtime unique-image
        radius to exceed the program cutoff. An image neighborhood defaults to its
        own searched lattice, and its stored-and-absent image certificate must
        cover the runtime positions and lattice. Failed certificates are reported
        in ``neighborhood_successful``. ``fractional_positions`` default to the
        runtime-lattice coordinates of ``positions``. ``topology`` is a graph
        topology prepared from ``neighborhood`` outside differentiation (for
        example a reused Verlet schedule) under this program's streamed plan; its
        routes, images, node metadata and build frame must equal the
        neighborhood's numerically, else the case fails. Without it the topology
        is prepared here from ``neighborhood``.
        """
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
        lattice = self._bind_lattice(
            neighborhood, position, cell, fractional_positions, cell_vectors
        )
        sites = self._site_geometry(position, lattice)
        pairs = self._pair_geometry(neighborhood, position, lattice)
        graph = self._graph(neighborhood, position, lattice, topology)
        graph_valid = jnp.asarray(True) if graph is None else jnp.all(graph.valid)
        scales = self._interaction_scales(
            interaction_scales, pairs.left.shape[0], position.dtype
        )
        resolved_cell_vectors = (
            jnp.zeros((0, 0), dtype=position.dtype)
            if lattice.cell is None
            else lattice.vectors
            if lattice.vectors is not None
            else neighborhood.cell_vectors[0].astype(position.dtype)
            if isinstance(neighborhood, ParticleImageNeighborhoodState)
            else lattice.cell.vectors.astype(position.dtype)
        )
        return AtomisticPotentialContext(
            system=self.system,
            site_state=sites.state,
            site_positions=sites.state.positions,
            site_type_ids=self.system.coordinate_map.plan.sites.site_type_ids,
            site_charges=self.system.coordinate_map.plan.sites.charges,
            site_pair_left=sites.left,
            site_pair_right=sites.right,
            site_pair_valid=sites.valid,
            site_pair_displacement=sites.displacement,
            site_pair_distance=sites.distance,
            site_lennard_jones_scales=sites.lennard_jones_scales,
            site_electrostatic_scales=sites.electrostatic_scales,
            positions=position,
            unwrapped_positions=unwrapped,
            species=species_,
            pair_left=pairs.left,
            pair_right=pairs.right,
            pair_valid=pairs.valid,
            pair_keys=pairs.keys,
            pair_displacement=pairs.displacement,
            pair_distance=pairs.distance,
            lennard_jones_scales=pairs.lennard_jones_scales,
            electrostatic_scales=pairs.electrostatic_scales,
            interaction_scales=scales,
            graph=graph,
            cell=lattice.cell,
            cell_vectors=resolved_cell_vectors,
            neighborhood_successful=(
                neighborhood.successful
                & pairs.successful
                & graph_valid
                & sites.state.successful
                & lattice.successful
            ),
        )

    def _bind_lattice(
        self,
        neighborhood: ParticleNeighborhoodState | ParticleImageNeighborhoodState,
        position: Array,
        cell: PeriodicCell | None,
        fractional_positions: ArrayLike | None,
        cell_vectors: ArrayLike | None,
        /,
    ) -> _RuntimeLattice:
        if cell is not None and (
            self.system.cell is None or cell.cell_id != self.system.cell.cell_id
        ):
            raise ValueError("Potential cell must exactly match the prepared system.")
        selected_cell = self.system.cell
        image_routes = isinstance(neighborhood, ParticleImageNeighborhoodState)
        if image_routes:
            self._admit_image_neighborhood(neighborhood, selected_cell)
        elif selected_cell is not None:
            box = neighborhood.box
            if not isinstance(box, PeriodicCell) or box.cell_id != selected_cell.cell_id:
                raise ValueError(
                    "Neighborhood and potential must share the exact periodic cell."
                )
            cutoff = self.plan.requirements.cutoff
            if cutoff is not None:
                selected_cell.require_unique_image(cutoff)
        elif neighborhood.box is not None and any(neighborhood.box.periodic_axes):
            raise ValueError("A finite potential cannot consume a periodic neighborhood.")
        if (
            cell_vectors is None
            and isinstance(neighborhood, ParticleImageNeighborhoodState)
            and selected_cell is not None
            and selected_cell.vectors.shape == (3, 3)
        ):
            # Image routes and their shifts were searched in the state's lattice.
            cell_vectors = neighborhood.cell_vectors[0]
        if cell_vectors is None:
            return _RuntimeLattice(selected_cell, None, None, None, jnp.asarray(True))
        if selected_cell is None:
            raise ValueError("Dynamic cell vectors require a periodic cell.")
        vectors = jnp.asarray(cell_vectors, dtype=position.dtype)
        if vectors.shape != (3, 3) or selected_cell.vectors.shape != (3, 3):
            raise ValueError("cell_vectors must have shape (3, 3) for a rank-3 cell.")
        inverse, solve_successful = selected_cell.inverse_for_vectors_with_status(vectors)
        inverse = inverse.astype(position.dtype)
        fractional = (
            contract(
                "ni,ij->nj",
                position - selected_cell.origin.astype(position.dtype),
                inverse,
            )
            if fractional_positions is None
            else jnp.asarray(fractional_positions, dtype=position.dtype)
        )
        if fractional.shape != position.shape:
            raise ValueError(f"fractional_positions must have shape {position.shape}.")
        successful = jnp.all(solve_successful) & _runtime_lattice_admissible(
            selected_cell,
            vectors,
            inverse,
            None if image_routes else self.plan.requirements.cutoff,
        )
        return _RuntimeLattice(selected_cell, vectors, inverse, fractional, successful)

    def _admit_image_neighborhood(
        self,
        neighborhood: ParticleImageNeighborhoodState,
        cell: PeriodicCell | None,
        /,
    ) -> None:
        requirements = self.plan.requirements
        if cell is None:
            raise ValueError("Image neighborhoods require a periodic prepared system.")
        if requirements.pair_geometry or requirements.interaction_site_geometry:
            raise ValueError(
                "Pair-once classical terms require a ParticleNeighborhoodState; "
                "directed image routes are not a pair-once relation."
            )
        if not requirements.directed_graph:
            raise ValueError("Image neighborhoods are consumed only by graph terms.")
        if (
            neighborhood.relation.support_id != self.system.particles.support.support_id
            or neighborhood.cell_vectors.shape[1:] != cell.vectors.shape
        ):
            raise ValueError(
                "Image neighborhood belongs to another particle support or lattice."
            )

    def _site_geometry(
        self, position: Array, lattice: _RuntimeLattice, /
    ) -> _SiteGeometry:
        coordinate_map = self.system.coordinate_map
        site_plan = coordinate_map.plan
        state = coordinate_map.realize(
            position,
            cell=lattice.cell,
            fractional_positions=lattice.fractional,
            cell_vectors=lattice.vectors,
        )
        left = coordinate_map.pair_left
        right = coordinate_map.pair_right
        valid = site_plan.sites.active_mask[left] & site_plan.sites.active_mask[right]
        if self.plan.requirements.interaction_site_geometry:
            physical_ids = site_plan.dof_particle_ids[
                jnp.maximum(site_plan.physical_dof_indices, 0)
            ]
            left_ids = physical_ids[left]
            right_ids = physical_ids[right]
            exception_supported = (
                site_plan.sites.physical_mask[left] & site_plan.sites.physical_mask[right]
            )
            zeros = jnp.zeros_like(left_ids)
            keys = jnp.stack(
                (
                    zeros,
                    jnp.minimum(left_ids, right_ids),
                    jnp.maximum(left_ids, right_ids),
                    zeros,
                    zeros,
                ),
                axis=-1,
            )
            resolved_lj, resolved_electrostatic = self.system.topology.pair_scales(keys)
            lj_scales = jnp.where(exception_supported, resolved_lj, 1.0)
            electrostatic_scales = jnp.where(
                exception_supported, resolved_electrostatic, 1.0
            )
        else:
            lj_scales = jnp.ones(left.shape, dtype=position.dtype)
            electrostatic_scales = jnp.ones_like(lj_scales)
        displacement = state.positions[left] - state.positions[right]
        if lattice.cell is not None and lattice.vectors is not None:
            vectors = lattice.vectors
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
            site_fractional = contract("nd,di->ni", state.positions, inverse)
            fractional_displacement = site_fractional[left] - site_fractional[right]
            central = jax.lax.stop_gradient(
                jnp.round(fractional_displacement).astype(jnp.int32)
            )
            central = jnp.where(lattice.cell.periodic_mask, central, 0)
            displacement = contract(
                "ni,ij->nj",
                fractional_displacement - central.astype(position.dtype),
                vectors,
            )
        elif lattice.cell is not None:
            displacement = lattice.cell.minimum_image(displacement)
        displacement = jnp.where(valid[:, None], displacement, 0.0)
        return _SiteGeometry(
            state=state,
            left=left,
            right=right,
            valid=valid,
            displacement=displacement,
            distance=_distance(displacement),
            lennard_jones_scales=lj_scales,
            electrostatic_scales=electrostatic_scales,
        )

    def _pair_geometry(
        self,
        neighborhood: ParticleNeighborhoodState | ParticleImageNeighborhoodState,
        position: Array,
        lattice: _RuntimeLattice,
        /,
    ) -> _PairGeometry:
        key_dtype = self.system.pair_key_space.sorted_particle_ids.dtype
        if isinstance(neighborhood, ParticleImageNeighborhoodState):
            index = jnp.zeros((0,), dtype=jnp.int32)
            empty = jnp.zeros((0,), dtype=position.dtype)
            return _PairGeometry(
                left=index,
                right=index,
                valid=jnp.zeros((0,), dtype=jnp.bool_),
                keys=jnp.zeros((0, 5), dtype=key_dtype),
                displacement=jnp.zeros((0, 3), dtype=position.dtype),
                distance=empty,
                lennard_jones_scales=empty,
                electrostatic_scales=empty,
                successful=jnp.asarray(True),
            )
        pairs = neighborhood.pair_relation
        left = pairs.left_indices
        right = pairs.right_indices
        displacement = position[left] - position[right]
        stencil_successful = jnp.asarray(True)
        if (
            lattice.cell is not None
            and lattice.vectors is not None
            and lattice.inverse is not None
            and lattice.fractional is not None
        ):
            fractional_displacement = lattice.fractional[left] - lattice.fractional[right]
            central = jax.lax.stop_gradient(
                jnp.round(fractional_displacement).astype(jnp.int32)
            )
            central = jnp.where(lattice.cell.periodic_mask, central, 0)
            centered = fractional_displacement - central.astype(position.dtype)
            candidates_fractional = (
                centered[:, None, :]
                - lattice.cell.image_shifts.astype(position.dtype)[None, :, :]
            )
            candidates = contract("nsi,ij->nsj", candidates_fractional, lattice.vectors)
            selected = jax.lax.stop_gradient(
                jnp.argmin(jnp.sum(candidates * candidates, axis=-1), axis=-1)
            )
            displacement = jnp.take_along_axis(
                candidates, selected[:, None, None], axis=1
            )[:, 0, :]
            # The minimum image is no longer than the centered candidate and, when
            # it matters to a cutoff term, shorter than the cutoff.
            radius = jax.lax.stop_gradient(
                _distance(contract("ni,ij->nj", centered, lattice.vectors))
            )
            cutoff = self.plan.requirements.cutoff
            if cutoff is not None:
                radius = jnp.minimum(radius, jnp.asarray(cutoff, dtype=radius.dtype))
            reach = jnp.sqrt(jnp.sum(jax.lax.stop_gradient(lattice.inverse) ** 2, axis=0))
            stencil_successful = jnp.all(
                ~pairs.valid | _stencil_complete(lattice.cell, radius, reach)
            )
        elif lattice.cell is not None:
            displacement = lattice.cell.minimum_image(displacement)
        displacement = jnp.where(pairs.valid[:, None], displacement, 0.0)
        if self.plan.requirements.pair_geometry:
            keys = self.system.pair_key_space.keys(pairs)
            key_values = keys.keys
            lj_scales, electrostatic_scales = self.system.topology.pair_scales(key_values)
            keys_successful = keys.successful
        else:
            # No term consumes topology pair identities or exception scales.
            key_values = jnp.zeros((0, 5), dtype=key_dtype)
            lj_scales = jnp.zeros((0,), dtype=position.dtype)
            electrostatic_scales = lj_scales
            keys_successful = jnp.asarray(True)
        return _PairGeometry(
            left=left,
            right=right,
            valid=pairs.valid,
            keys=key_values,
            displacement=displacement,
            distance=_distance(displacement),
            lennard_jones_scales=lj_scales,
            electrostatic_scales=electrostatic_scales,
            successful=keys_successful & stencil_successful,
        )

    def _graph(
        self,
        neighborhood: ParticleNeighborhoodState | ParticleImageNeighborhoodState,
        position: Array,
        lattice: _RuntimeLattice,
        topology: AtomisticGraphTopology | None,
        /,
    ) -> AtomisticGraph | None:
        if not self.plan.requirements.directed_graph:
            if topology is not None:
                raise ValueError("topology was supplied but no term requires a graph.")
            return None
        execution = self.graph_execution
        cutoff = self.plan.requirements.cutoff
        if execution is None or cutoff is None:
            raise RuntimeError("Validated graph execution or cutoff unexpectedly absent.")
        prepared = particle_atomistic_graph_topology(
            self.system,
            neighborhood,
            epoch=0 if topology is None else topology.epoch,
            cell=lattice.cell,
            streamed=execution.streamed if topology is None else topology.streamed,
        )
        if topology is not None:
            prepared = admit_supplied_topology(topology, prepared)
        vectors = lattice.vectors
        if vectors is None and isinstance(neighborhood, ParticleImageNeighborhoodState):
            vectors = neighborhood.cell_vectors[0].astype(position.dtype)
        return bind_atomistic_graph(
            prepared, execution, position, cutoff=cutoff, cell_vectors=vectors
        )

    def _interaction_scales(
        self,
        interaction_scales: AtomisticInteractionScaleState | None,
        pair_count: int,
        dtype: DTypeLike,
        /,
    ) -> AtomisticInteractionScaleState:
        scales = (
            AtomisticInteractionScaleState.identity(self.system, pair_count, dtype)
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
            "lennard_jones": (pair_count,),
            "electrostatic": (pair_count,),
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
        if interaction_scales is None:
            # Identity scales are valid by construction; only supplied scales are
            # runtime-checked.
            return scales
        scale_invalid = jnp.asarray(False)
        for value in scale_values:
            scale_invalid = scale_invalid | jnp.any(
                ~jnp.isfinite(value) | (value < 0.0) | (value > 1.0)
            )
        scale_invalid = (
            scale_invalid
            | ~jnp.isfinite(scales.lennard_jones_softcore_alpha)
            | (scales.lennard_jones_softcore_alpha < 0.0)
            | ~jnp.isfinite(scales.electrostatic_softcore_alpha)
            | (scales.electrostatic_softcore_alpha < 0.0)
        )
        checked_lj_scales = eqx.error_if(
            scales.lennard_jones,
            scale_invalid,
            "Interaction scales must be finite in [0, 1] with non-negative finite soft-core values.",
        )
        return eqx.tree_at(lambda value: value.lennard_jones, scales, checked_lj_scales)

    def energy(
        self,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState | ParticleImageNeighborhoodState,
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
        neighborhood: ParticleNeighborhoodState | ParticleImageNeighborhoodState,
        /,
        *,
        compute_stress: bool = False,
        **context_kwargs: Any,
    ) -> AtomisticPotentialEvaluation:
        """Differentiate one scalar in Cartesian positions and, if requested, strain.

        Topology, integer images and supplied coordinate representations stay
        fixed. ``compute_stress=True`` differentiates the same scalar in
        homogeneous cell strain within one reverse pass and refuses unless every
        term owns a cell derivative and the cell is a full 3x3 lattice. Partially
        periodic cells report stress per declared embedding volume
        (``stress_convention``); it never falls back to zero or to the
        position-moment ``virial``.
        """
        position = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        point = AtomisticDerivativePoint(
            self.system,
            position,
            context_kwargs,
            neighborhood=neighborhood,
            strained=compute_stress,
            cell_derivative=self.plan.capabilities.cell_derivative,
        )

        def closure(
            value: Array, strain: Array | None
        ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
            bound, kwargs = point.bind(value, strain)
            return self.energy(bound, neighborhood, **kwargs)

        strain_derivative = None
        if point.strained:
            (energy, auxiliary), (gradient, strain_derivative) = jax.value_and_grad(
                closure, argnums=(0, 1), has_aux=True
            )(position, point.zero_strain())
        else:
            (energy, auxiliary), gradient = jax.value_and_grad(closure, has_aux=True)(
                position, None
            )
        term_energies, atom_energy, program_successful, graph_overflow = auxiliary
        successful = program_successful & point.successful
        forces = jnp.where(self.system.active_mask[:, None], -gradient, 0.0)
        virial = point.moment_virial(self.system, position, forces)
        stress = None if strain_derivative is None else point.stress(strain_derivative)
        finite = (
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(term_energies))
            & jnp.all(jnp.isfinite(forces))
        )
        if strain_derivative is not None and stress is not None:
            finite = (
                finite
                & jnp.all(jnp.isfinite(strain_derivative))
                & jnp.all(jnp.isfinite(stress))
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
            strain_derivative=(
                None
                if strain_derivative is None
                else jnp.where(accepted, strain_derivative, nan)
            ),
            stress=None if stress is None else jnp.where(accepted, stress, nan),
            program_id=self.prepared_id,
            stress_convention=point.stress_convention,
        )


__all__ = [
    "AbstractPreparedAtomisticHamiltonian",
    "AtomisticHamiltonianEvaluation",
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
