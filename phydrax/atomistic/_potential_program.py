#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, assert_never, Protocol, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from phydrax.ein import contract

from .._admissibility import guard_derivative_validity
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from ..discretization import PeriodicCell
from ..linalg import determinant_small_linear, SmallLinearSolvePlan
from ..typing import checked, PRNGKey
from ..units import conversion_factor
from ._conditioning import (
    AtomisticConditioning,
    AtomisticFieldReference,
    AtomisticObservable,
    AtomisticObservableRequest,
    AtomisticObservableValue,
)
from ._execution_binding import (
    atomistic_hamiltonian_layout_id,
    atomistic_term_layout_id,
)
from ._graph import (
    _neighborhood_graph_topology,
    AtomisticGraph,
    AtomisticGraphExecutionPlan,
    bind_atomistic_graph,
)
from ._model_inputs import AtomisticGraphInputs
from ._neighborhood_bundle import (
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodBundleState,
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticNeighborhoodView,
    AtomisticPairCounting,
    PreparedAtomisticNeighborhoodBundle,
)
from ._potential import (
    AbstractAtomisticModel,
    AbstractPreparedAtomisticModel,
    atomistic_model_revision,
    AtomisticPotentialCapabilities,
    AtomisticPotentialRequirements,
    AtomisticStressConvention,
)
from ._sites import AtomisticInteractionSiteState
from ._system import PreparedAtomisticSystem
from ._term_state import AbstractAtomisticTermState, AtomisticTermStateBindings
from ._types import AtomisticSpeciesKind


def _admitted_scalar(
    value: Array, successful: Array, dependencies: tuple[Array, ...], /
) -> Array:
    """Invalidate the scalar and its coordinate/control tangents together."""
    scale = jnp.where(
        successful,
        jnp.ones((), dtype=value.dtype),
        jnp.asarray(jnp.nan, dtype=value.dtype),
    )
    return guard_derivative_validity(value * scale, successful, dependencies=dependencies)


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


def _resolve_field(
    reference: AtomisticFieldReference,
    producers: tuple[str, ...],
    fields: tuple[AtomisticObservableValue, ...],
    /,
) -> AtomisticObservableValue:
    selected: AtomisticObservableValue | None = None
    for producer, field in zip(producers, fields, strict=True):
        if (
            producer == reference.producer
            and field.observable is reference.observable
            and field.port.port_id == reference.port.port_id
        ):
            if selected is not None:
                raise ValueError(
                    "A producer supplied the same scientific field more than once."
                )
            selected = field
    if selected is None:
        raise ValueError(
            "The requested producer has not supplied its declared scientific field."
        )
    if selected.level is not reference.level:
        raise ValueError(
            "The producer field and consumer reference use different entity levels."
        )
    if reference.unit is None:
        if selected.unit is not None:
            raise ValueError(
                "A field reference cannot discard the producer's physical unit."
            )
        return selected
    if selected.unit is None:
        raise ValueError(
            "A physical field cannot be inferred from an uncalibrated producer output."
        )
    if selected.unit.unit_id == reference.unit.unit_id:
        return selected
    factor = conversion_factor(selected.unit, reference.unit)
    return AtomisticObservableValue(
        selected.observable,
        selected.port,
        selected.values * jnp.asarray(factor, dtype=selected.values.dtype),
        selected.available,
        level=selected.level,
        case_count=selected.case_count,
        atom_capacity=selected.atom_capacity,
        unit=reference.unit,
    )


class AtomisticPotentialContext(StrictModule):
    """Requirements-resolved dynamic data shared by one potential program."""

    __strict_contract__ = True

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
    cell_inverse: Array | None
    neighborhood_successful: Array
    time: Array
    term_states: AtomisticTermStateBindings
    graph_execution: AtomisticGraphExecutionPlan | None = None
    neighborhoods: tuple[AtomisticNeighborhoodView, ...] = ()
    field_values: tuple[AtomisticObservableValue, ...] = ()
    field_producers: tuple[str, ...] = eqx.field(static=True, default=())
    conditioning: AtomisticConditioning | None = None
    key: PRNGKey | None = None
    image_counts: Array | None = None
    scale_values: tuple[AtomisticInteractionScaleState, ...] = ()
    scale_slots: tuple[str, ...] = eqx.field(static=True, default=())

    @checked
    def field(self, reference: AtomisticFieldReference, /) -> AtomisticObservableValue:
        return _resolve_field(reference, self.field_producers, self.field_values)

    @checked
    def neighborhood(
        self, requirement: AtomisticNeighborhoodRequirement, /
    ) -> AtomisticNeighborhoodView:
        for view in self.neighborhoods:
            if view.requirement_id == requirement.requirement_id:
                return view
        raise ValueError(
            "The context has no view for this exact neighborhood requirement."
        )


class AtomisticTermEvaluation(StrictModule):
    energy: Array
    atom_energy: Array
    successful: Array
    atom_energy_complete: bool = eqx.field(static=True, default=True)
    evidence: tuple[StrictModule, ...] = ()
    fields: tuple[AtomisticObservableValue, ...] = ()


class AtomisticProgramAuxiliary(StrictModule):
    """The complete auxiliary state of one scalar program evaluation.

    ``term_slots`` are the evaluated rows' declared term names (layout);
    ``term_identities`` stacks their exact scientific term digests as dynamic
    words, decoded on the host by ``term_ids``.
    """

    term_energies: Array
    contributing_terms: Array
    term_slots: tuple[str, ...] = eqx.field(static=True)
    term_identities: IdentityWords
    atom_energy: Array
    successful: Array
    graph_overflow: Array
    neighborhood_successful: Array
    term_evidence: tuple[tuple[StrictModule, ...], ...]
    fields: tuple[AtomisticObservableValue, ...]
    field_producers: tuple[str, ...] = eqx.field(static=True)
    atom_energy_complete: bool = eqx.field(static=True)

    @property
    def term_ids(self) -> tuple[str, ...]:
        """Host decoding of the evaluated rows' scientific term digests."""
        return tuple(
            IdentityWords(self.term_identities.words[row]).fingerprint()
            for row in range(len(self.term_slots))
        )


def _stacked_identities(values: Sequence[IdentityWords], /) -> IdentityWords:
    if not values:
        return IdentityWords(jnp.zeros((0, 8), dtype=jnp.uint32))
    return IdentityWords(jnp.stack(tuple(value.words for value in values)))


class AbstractAtomisticEnergyTerm(StrictModule):
    """One composable scalar-energy term in a prepared atomistic program.

    ``name`` is the declared program slot; the exact scientific term digest is
    the dynamic ``term_identity`` (``term_id`` decodes it on the host).
    """

    name: eqx.AbstractVar[str]
    force_group: eqx.AbstractVar[int]
    term_identity: eqx.AbstractVar[IdentityWords]
    capabilities: eqx.AbstractVar[AtomisticPotentialCapabilities]
    requirements: eqx.AbstractVar[AtomisticPotentialRequirements]

    @property
    def term_id(self) -> str:
        """Host decoding of the canonical scientific term digest."""
        return self.term_identity.fingerprint()

    @abc.abstractmethod
    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "AbstractPreparedAtomisticEnergyTerm":
        raise NotImplementedError


class AbstractPreparedAtomisticEnergyTerm(StrictModule):
    """One prepared term: layout statics plus dynamic numerics and identities.

    ``term_identity`` (scientific term) and ``prepared_identity`` (numerical
    preparation bound to its system) are exact canonical digests as dynamic
    words; ``term_id``/``prepared_id`` decode them on the host.  ``layout_id``
    is the content-free executable layout of the term.
    """

    name: eqx.AbstractVar[str]
    force_group: eqx.AbstractVar[int]
    term_identity: eqx.AbstractVar[IdentityWords]
    prepared_identity: eqx.AbstractVar[IdentityWords]
    capabilities: eqx.AbstractVar[AtomisticPotentialCapabilities]
    requirements: eqx.AbstractVar[AtomisticPotentialRequirements]

    @property
    def term_id(self) -> str:
        """Host decoding of the canonical scientific term digest."""
        return self.term_identity.fingerprint()

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical prepared-term digest."""
        return self.prepared_identity.fingerprint()

    @property
    def layout_id(self) -> str:
        """Content-free executable layout of this prepared term."""
        return atomistic_term_layout_id(self)

    @abc.abstractmethod
    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        raise NotImplementedError

    def initial_term_state(
        self, dtype: DTypeLike, /
    ) -> AbstractAtomisticTermState | None:
        """Stateless terms have no initial state; stateful owners provide it.

        A provided state's ``owner_identity`` is this term's ``term_identity``.
        """
        if self.requirements.state_required:
            raise TypeError("A stateful term must implement its state initializer.")
        return None


class LearnedGraphPotentialTerm(AbstractAtomisticEnergyTerm):
    """Bind a trained graph scalar-energy model into a dynamics potential program.

    The term consumes only the model's declared directed graph. Periodic
    execution binds fixed integer image routes to runtime cell vectors, so its
    energy is differentiable in homogeneous cell strain; that is an execution
    capability, not evidence that the fitted model is stable for dynamics.
    """

    __strict_contract__ = True

    potential: AbstractAtomisticModel | AbstractPreparedAtomisticModel
    conditioning: AtomisticConditioning | None
    key: PRNGKey | None
    request: AtomisticObservableRequest | None
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    allow_periodic: bool = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    @checked
    def __init__(
        self,
        potential: AbstractAtomisticModel | AbstractPreparedAtomisticModel,
        /,
        *,
        name: str = "learned",
        force_group: int = 0,
        allow_periodic: bool = False,
        conditioning: AtomisticConditioning | None = None,
        key: PRNGKey | None = None,
        request: AtomisticObservableRequest | None = None,
    ) -> None:
        identifier = str(name).strip()
        group = int(force_group)
        if not identifier or group < 0:
            raise ValueError(
                "Potential term name must be non-empty and force_group non-negative."
            )
        periodic = bool(allow_periodic)
        declared = potential.capabilities
        if not declared.conservative_energy:
            raise ValueError("A property-only model cannot be used as an energy term.")
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
            local_energy=declared.local_energy,
            species_kind=declared.species_kind,
        )
        source_requirements = potential.requirements
        source_requirements.neighborhood(
            AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
        ).bounded_cutoff()
        requirements = AtomisticPotentialRequirements(
            bonded_geometry=source_requirements.bonded_geometry,
            reciprocal_grid=source_requirements.reciprocal_grid,
            state_required=source_requirements.state_required,
            field_dependencies=source_requirements.field_dependencies,
            neighborhoods=tuple(
                AtomisticNeighborhoodRequirement(
                    identifier,
                    requirement.role,
                    requirement.cutoff,
                    selection=requirement.selection,
                    receiver_group=requirement.receiver_group,
                    exclusions=requirement.exclusions,
                )
                for requirement in source_requirements.neighborhoods
            ),
        )
        self.potential = potential
        self.conditioning = conditioning
        self.key = key
        self.request = request
        self.name = identifier
        self.force_group = group
        self.allow_periodic = bool(allow_periodic)
        self.capabilities = capabilities
        self.requirements = requirements
        self.term_identity = IdentityWords.of(
            {
                "kind": "learned-graph-potential-term",
                "potential_revision": atomistic_model_revision(potential).revision_id,
                "name": identifier,
                "force_group": group,
                "allow_periodic": bool(allow_periodic),
                "capabilities": capabilities.capabilities_id,
                "requirements": requirements.requirements_id,
                "conditioning": (
                    None
                    if conditioning is None
                    else (conditioning.spec.spec_id, array_tree_fingerprint(conditioning))
                ),
                "random_realization": (
                    None
                    if key is None
                    else (
                        str(jax.random.key_impl(key)),
                        array_tree_fingerprint(jax.random.key_data(key)),
                    )
                ),
                "request": None if request is None else request.request_id,
            }
        )

    def prepare(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        request: AtomisticObservableRequest | None = None,
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
        return PreparedLearnedGraphPotentialTerm(self, system, request=request)


class PreparedLearnedGraphPotentialTerm(AbstractPreparedAtomisticEnergyTerm):
    plan: LearnedGraphPotentialTerm
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements
    request: AtomisticObservableRequest | None

    def __init__(
        self,
        plan: LearnedGraphPotentialTerm,
        system: PreparedAtomisticSystem,
        /,
        *,
        request: AtomisticObservableRequest | None = None,
    ) -> None:
        self.plan = plan
        self.system = system
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.request = plan.request if request is None else request
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-learned-graph-potential-term",
                "term": plan.term_id,
                "system": system.prepared_id,
                "request": None if self.request is None else self.request.request_id,
                "model_realization": (
                    plan.potential.prepared_id
                    if isinstance(plan.potential, AbstractPreparedAtomisticModel)
                    else None
                ),
            }
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        if context.graph is None:
            raise ValueError("Learned graph potential requires a directed graph context.")
        if self.plan.conditioning is not None and context.conditioning is not None:
            raise ValueError("Bind conditioning on the term or the evaluation, not both.")
        if self.plan.key is not None and context.key is not None:
            raise ValueError("Bind a random key on the term or the evaluation, not both.")
        outputs = self.plan.potential.evaluate_graph(
            AtomisticGraphInputs(
                context.graph,
                self.system.plan.units.scale,
                species_kind=self.plan.potential.capabilities.species_kind,
                conditioning=(
                    self.plan.conditioning
                    if self.plan.conditioning is not None
                    else context.conditioning
                ),
                key=self.plan.key if self.plan.key is not None else context.key,
            ),
            self.request,
        )
        return AtomisticTermEvaluation(
            outputs.energy[0],
            outputs.atom_energy[0],
            outputs.successful[0],
            atom_energy_complete=outputs.atom_energy_complete,
            fields=outputs.additional,
            evidence=outputs.evidence,
        )


def _term_execution_order(
    terms: tuple[AbstractAtomisticEnergyTerm, ...], /
) -> tuple[int, ...]:
    indices = {term.name: index for index, term in enumerate(terms)}
    dependencies = []
    for term in terms:
        missing = tuple(
            ref.producer
            for ref in term.requirements.field_dependencies
            if ref.producer not in indices
        )
        if missing:
            raise ValueError(
                f"Potential term {term.name!r} has unknown field producers {missing!r}."
            )
        dependencies.append(
            {indices[ref.producer] for ref in term.requirements.field_dependencies}
        )
    pending = set(range(len(terms)))
    order = []
    while pending:
        ready = tuple(
            index for index in sorted(pending) if not dependencies[index] & pending
        )
        if not ready:
            raise ValueError("Atomistic potential field dependencies contain a cycle.")
        order.extend(ready)
        pending.difference_update(ready)
    return tuple(order)


def _producer_request(
    terms: tuple[AbstractAtomisticEnergyTerm, ...],
    producer: LearnedGraphPotentialTerm,
    /,
) -> AtomisticObservableRequest:
    requested = {AtomisticObservable.ENERGY, AtomisticObservable.ATOM_ENERGY}
    derivatives = ()
    if producer.request is not None:
        requested.update(producer.request.observables)
        derivatives = producer.request.derivatives
    for term in terms:
        requested.update(
            ref.observable
            for ref in term.requirements.field_dependencies
            if ref.producer == producer.name
        )
    return AtomisticObservableRequest(tuple(requested), derivatives=derivatives)


class AtomisticPotentialProgram(StrictModule):
    """Ordered additive scalar-energy program with merged spatial requirements.

    Term slots, force groups and evaluation order are layout; the canonical
    program digest (term digests, coefficients, requirements, capabilities,
    order) is the dynamic ``program_identity``, decoded by ``program_id``.
    """

    terms: tuple[AbstractAtomisticEnergyTerm, ...]
    coefficients: Array
    term_names: tuple[str, ...] = eqx.field(static=True)
    force_groups: tuple[int, ...] = eqx.field(static=True)
    evaluation_order: tuple[int, ...] = eqx.field(static=True)
    requirements: AtomisticPotentialRequirements
    capabilities: AtomisticPotentialCapabilities
    program_identity: IdentityWords

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
        spatial = {
            requirement.requirement_id: requirement
            for term in values
            for requirement in term.requirements.neighborhoods
        }
        dependencies = {
            reference.reference_id: reference
            for term in values
            for reference in term.requirements.field_dependencies
        }
        evaluation_order = _term_execution_order(values)
        requirements = AtomisticPotentialRequirements(
            bonded_geometry=any(value.requirements.bonded_geometry for value in values),
            reciprocal_grid=any(value.requirements.reciprocal_grid for value in values),
            state_required=any(value.requirements.state_required for value in values),
            neighborhoods=tuple(spatial[key] for key in sorted(spatial)),
            field_dependencies=tuple(dependencies[key] for key in sorted(dependencies)),
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
        self.evaluation_order = evaluation_order
        self.requirements = requirements
        self.capabilities = capabilities
        self.program_identity = IdentityWords.of(
            {
                "kind": "atomistic-potential-program",
                "terms": [value.term_id for value in values],
                "coefficients": weights.tolist(),
                "requirements": requirements.requirements_id,
                "capabilities": capabilities.capabilities_id,
                "evaluation_order": evaluation_order,
            }
        )

    @property
    def program_id(self) -> str:
        """Host decoding of the canonical unbound program digest."""
        return self.program_identity.fingerprint()

    def prepare(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        neighborhood_bundle: AtomisticNeighborhoodBundlePlan
        | PreparedAtomisticNeighborhoodBundle,
        graph_execution: AtomisticGraphExecutionPlan | None = None,
    ) -> "PreparedAtomisticPotentialProgram":
        return PreparedAtomisticPotentialProgram(
            self,
            system,
            neighborhood_bundle=neighborhood_bundle,
            graph_execution=graph_execution,
        )


class AtomisticPotentialEvaluation(StrictModule):
    """Energy, conservative forces and optional homogeneous-strain response.

    ``virial`` is the finite position-moment diagnostic ``-sum (r - c) (x) f``
    about the active center of mass; it is not a periodic virial. When stress is
    requested, ``strain_derivative`` is ``dE/dstrain`` from the same scalar
    derivative pass and ``stress`` follows ``stress_convention``; otherwise all
    three are ``None``. Any failed scientific, image, capacity, finite or
    derivative state sets ``successful`` false and every numeric output to NaN.
    ``program_identity`` is the exact prepared-Hamiltonian digest of the
    producing program as dynamic words; ``program_id`` decodes it on the host.
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
    program_identity: IdentityWords
    stress_convention: AtomisticStressConvention | None = eqx.field(static=True)
    term_evidence: tuple[tuple[StrictModule, ...], ...]
    fields: tuple[AtomisticObservableValue, ...]
    field_producers: tuple[str, ...] = eqx.field(static=True)
    atom_energy_complete: bool = eqx.field(static=True)

    @property
    def program_id(self) -> str:
        """Host decoding of the producing prepared Hamiltonian's digest."""
        return self.program_identity.fingerprint()

    @checked
    def field(self, reference: AtomisticFieldReference, /) -> AtomisticObservableValue:
        return _resolve_field(reference, self.field_producers, self.fields)


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
    def program_identity(self) -> IdentityWords: ...

    @property
    def program_id(self) -> str: ...

    @property
    def stress_convention(self) -> AtomisticStressConvention | None: ...

    @property
    def term_evidence(self) -> tuple[tuple[StrictModule, ...], ...]: ...

    @property
    def fields(self) -> tuple[AtomisticObservableValue, ...]: ...

    @property
    def field_producers(self) -> tuple[str, ...]: ...

    @property
    def atom_energy_complete(self) -> bool: ...


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
        neighborhoods: AtomisticNeighborhoodBundleState,
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
            None
            if cell is None
            else neighborhoods.geometry.cell_vectors[0].astype(position.dtype)
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
    """Prepared scalar Hamiltonian boundary shared by fixed and controlled programs.

    The PyTree is layout-only: ``scientific_identity`` (the bound scalar
    Hamiltonian, independent of its resource binding) and ``prepared_identity``
    (the complete numerical binding) are exact canonical digests carried as
    dynamic words and decoded on the host by ``scientific_id``/``prepared_id``.
    Every material and parameter revision of one ``layout_id`` shares one
    executable.
    """

    system: eqx.AbstractVar[PreparedAtomisticSystem]
    plan: eqx.AbstractVar[Any]
    terms: eqx.AbstractVar[tuple[AbstractPreparedAtomisticEnergyTerm, ...]]
    prepared_identity: eqx.AbstractVar[IdentityWords]
    scientific_identity: eqx.AbstractVar[IdentityWords]
    control_ids: eqx.AbstractVar[tuple[str, ...]]
    control_layout_id: eqx.AbstractVar[str]
    coefficients: eqx.AbstractVar[Array]
    neighborhood_bundle: eqx.AbstractVar[PreparedAtomisticNeighborhoodBundle]

    @property
    def scientific_id(self) -> str:
        """Host decoding of the bound scalar Hamiltonian digest."""
        return self.scientific_identity.fingerprint()

    @property
    def prepared_id(self) -> str:
        """Host decoding of the complete prepared-binding digest."""
        return self.prepared_identity.fingerprint()

    @property
    def layout_id(self) -> str:
        """Content-free executable layout shared by all revisions of this program."""
        return atomistic_hamiltonian_layout_id(self)

    @abc.abstractmethod
    def with_neighborhood_bundle(
        self, bundle: PreparedAtomisticNeighborhoodBundle, /
    ) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def context(
        self,
        positions: ArrayLike,
        neighborhoods: AtomisticNeighborhoodBundleState,
        /,
        *,
        term: AbstractPreparedAtomisticEnergyTerm | None = None,
        unwrapped_positions: ArrayLike | None = None,
        species: ArrayLike | None = None,
        cell: PeriodicCell | None = None,
        fractional_positions: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
        image_counts: ArrayLike | None = None,
        conditioning: AtomisticConditioning | None = None,
        key: PRNGKey | None = None,
        term_states: AtomisticTermStateBindings | None = None,
        time: ArrayLike = 0.0,
    ) -> AtomisticPotentialContext:
        raise NotImplementedError

    @abc.abstractmethod
    def energy(
        self,
        positions: ArrayLike,
        neighborhoods: AtomisticNeighborhoodBundleState,
        /,
        **kwargs: Any,
    ) -> tuple[Array, AtomisticProgramAuxiliary]:
        raise NotImplementedError

    @abc.abstractmethod
    def force_group_energy(
        self,
        group: int,
        positions: ArrayLike,
        neighborhoods: AtomisticNeighborhoodBundleState,
        /,
        **kwargs: Any,
    ) -> tuple[Array, AtomisticProgramAuxiliary]:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        positions: ArrayLike,
        neighborhoods: AtomisticNeighborhoodBundleState,
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
    vectors: Array
    inverse: Array | None
    fractional: Array | None
    successful: Array


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
    neighborhood_bundle: PreparedAtomisticNeighborhoodBundle
    unique_image_cutoff: float | None = eqx.field(static=True)
    prepared_identity: IdentityWords
    scientific_identity: IdentityWords
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
        neighborhood_bundle: AtomisticNeighborhoodBundlePlan
        | PreparedAtomisticNeighborhoodBundle,
        graph_execution: AtomisticGraphExecutionPlan | None,
        _reuse: PreparedAtomisticPotentialProgram | None = None,
    ) -> None:
        requested = plan.requirements.neighborhoods
        graph_roles = tuple(
            value
            for value in requested
            if value.role is AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
        )
        if graph_roles:
            if graph_execution is None or graph_execution.backend != "particle":
                raise ValueError(
                    "A particle AtomisticGraphExecutionPlan is required by graph terms."
                )
        elif graph_execution is not None:
            raise ValueError("graph_execution was supplied but no term requires a graph.")
        bundle_plan = (
            neighborhood_bundle
            if isinstance(neighborhood_bundle, AtomisticNeighborhoodBundlePlan)
            else neighborhood_bundle.plan
        )
        if {value.requirement_id for value in bundle_plan.requirements} != {
            value.requirement_id for value in requested
        }:
            raise ValueError(
                "The resource bundle must cover exactly the program's scientific requirements."
            )
        if bundle_plan.case_count != 1:
            raise ValueError(
                "A prepared Hamiltonian binds one system; batch complete system states externally."
            )
        if isinstance(neighborhood_bundle, AtomisticNeighborhoodBundlePlan):
            plans = (
                {}
                if graph_execution is None
                else {
                    value.requirement_id: graph_execution.streamed
                    for value in graph_roles
                }
            )
            bundle = neighborhood_bundle.prepare(system, graph_plans=plans)
        else:
            bundle = neighborhood_bundle
            if bundle.system.prepared_id != system.prepared_id:
                raise ValueError(
                    "The resource bundle belongs to another prepared system."
                )
            for requirement in graph_roles:
                prepared_plan = bundle.graph_plans[bundle.slot(requirement)]
                if (
                    graph_execution is None
                    or prepared_plan is None
                    or (prepared_plan.plan_id != graph_execution.streamed.plan_id)
                ):
                    raise ValueError(
                        "The prepared bundle lacks the program's exact graph schedule plan."
                    )
        classical_cutoffs = tuple(
            value.cutoff
            for value in requested
            if value.counting is AtomisticPairCounting.MINIMUM_IMAGE_HALF
            and value.cutoff is not None
        )
        unique_cutoff = max(classical_cutoffs) if classical_cutoffs else None
        if unique_cutoff is not None and system.cell is not None:
            system.cell.require_unique_image(unique_cutoff)
        if _reuse is None:
            terms = tuple(
                value.prepare(system, request=_producer_request(plan.terms, value))
                if isinstance(value, LearnedGraphPotentialTerm)
                else value.prepare(system)
                for value in plan.terms
            )
        else:
            if (
                _reuse.plan.program_id != plan.program_id
                or _reuse.system.prepared_id != system.prepared_id
            ):
                raise ValueError(
                    "Prepared numerical terms may only be reused for the same program and system."
                )
            terms = _reuse.terms
        if any(
            term.requirements.state_required
            and type(term).initial_term_state
            is AbstractPreparedAtomisticEnergyTerm.initial_term_state
            for term in terms
        ):
            raise TypeError("Every stateful term must own its state initialization.")
        self.plan = plan
        self.system = system
        self.terms = terms
        self.coefficients = plan.coefficients
        self.graph_execution = graph_execution
        self.neighborhood_bundle = bundle
        self.unique_image_cutoff = unique_cutoff
        self.control_ids = ()
        self.control_layout_id = canonical_fingerprint(
            {"kind": "atomistic-control-layout", "control_ids": []}
        )
        self.scientific_identity = IdentityWords.of(
            {
                "kind": "bound-atomistic-hamiltonian",
                "program": plan.program_id,
                "system": system.plan.system_id,
            }
        )
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-atomistic-potential-program",
                "program": plan.program_id,
                "system": system.prepared_id,
                "terms": [value.prepared_id for value in terms],
                "neighborhood_bundle": bundle.prepared_id,
                "graph_execution": None
                if graph_execution is None
                else graph_execution.plan_id,
            }
        )

    def with_neighborhood_bundle(
        self, bundle: PreparedAtomisticNeighborhoodBundle, /
    ) -> PreparedAtomisticPotentialProgram:
        """Rebind resources without rebuilding unchanged numerical term preparations."""
        return PreparedAtomisticPotentialProgram(
            self.plan,
            self.system,
            neighborhood_bundle=bundle,
            graph_execution=self.graph_execution,
            _reuse=self,
        )

    @checked
    def context(
        self,
        positions: ArrayLike,
        neighborhoods: AtomisticNeighborhoodBundleState,
        /,
        *,
        term: AbstractPreparedAtomisticEnergyTerm | None = None,
        unwrapped_positions: ArrayLike | None = None,
        species: ArrayLike | None = None,
        interaction_scales: Mapping[str, AtomisticInteractionScaleState] | None = None,
        cell: PeriodicCell | None = None,
        fractional_positions: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
        image_counts: ArrayLike | None = None,
        conditioning: AtomisticConditioning | None = None,
        key: PRNGKey | None = None,
        term_states: AtomisticTermStateBindings | None = None,
        time: ArrayLike = 0.0,
    ) -> AtomisticPotentialContext:
        """Resolve one term's resources at the bundle's certified point.

        Multiple-term programs require an explicit term; it selects its program
        slot, and the resolved context always belongs to this program's own
        prepared term in that slot.  Refresh the bundle outside this numerical
        boundary; neither discovery nor selection runs inside AD.
        """
        if term is None:
            if len(self.terms) != 1:
                raise ValueError(
                    "A multi-term context requires an explicit prepared term."
                )
            term = self.terms[0]
        matches = tuple(
            value
            for value in self.terms
            if value.name == term.name and type(value) is type(term)
        )
        if len(matches) != 1:
            raise ValueError("The context term does not belong to this program.")
        shared = self._base_context(
            positions,
            neighborhoods,
            unwrapped_positions=unwrapped_positions,
            species=species,
            interaction_scales=interaction_scales,
            cell=cell,
            fractional_positions=fractional_positions,
            cell_vectors=cell_vectors,
            image_counts=image_counts,
            conditioning=conditioning,
            key=key,
            term_states=term_states,
            time=time,
        )
        return self._term_context(shared, neighborhoods, matches[0])

    def _base_context(
        self,
        positions: ArrayLike,
        neighborhoods: AtomisticNeighborhoodBundleState,
        /,
        *,
        unwrapped_positions: ArrayLike | None = None,
        species: ArrayLike | None = None,
        interaction_scales: Mapping[str, AtomisticInteractionScaleState] | None = None,
        cell: PeriodicCell | None = None,
        fractional_positions: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
        image_counts: ArrayLike | None = None,
        conditioning: AtomisticConditioning | None = None,
        key: PRNGKey | None = None,
        term_states: AtomisticTermStateBindings | None = None,
        time: ArrayLike = 0.0,
    ) -> AtomisticPotentialContext:
        position = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        expected = (self.system.capacity, 3)
        if position.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        states = AtomisticTermStateBindings({}) if term_states is None else term_states
        if not isinstance(states, AtomisticTermStateBindings):
            raise TypeError("term_states must use AtomisticTermStateBindings.")
        known_states = {
            term.name for term in self.terms if term.requirements.state_required
        }
        if set(states.slots) - known_states:
            raise ValueError("Term-state bindings name a foreign or stateless slot.")
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
            neighborhoods, position, cell, fractional_positions, cell_vectors
        )
        sites = self.system.coordinate_map.realize(
            position,
            cell=lattice.cell,
            fractional_positions=lattice.fractional,
            cell_vectors=None if lattice.cell is None else lattice.vectors,
        )
        counts = neighborhoods.geometry.image_counts
        counts = None if counts is None else counts[0]
        if image_counts is not None:
            supplied = jnp.asarray(image_counts)
            if supplied.shape != (
                self.system.capacity,
                self.neighborhood_bundle.lattice_rank,
            ) or not jnp.issubdtype(supplied.dtype, jnp.integer):
                raise ValueError(
                    "image_counts must match the physical atom and lattice axes."
                )
            same_images = jnp.all(
                supplied == (jnp.zeros_like(supplied) if counts is None else counts)
            )
            counts = supplied
        else:
            same_images = jnp.asarray(True)
        same_position = jnp.all(
            jnp.where(
                self.system.active_mask[:, None],
                position == neighborhoods.geometry.positions[0],
                True,
            )
        )
        same_cell = jnp.all(lattice.vectors == neighborhoods.geometry.cell_vectors[0])
        same_support = jnp.all(
            neighborhoods.geometry.active[0] == self.system.active_mask
        )
        same_sites = (
            jnp.asarray(True)
            if neighborhoods.geometry.site_positions is None
            else jnp.all(
                jnp.where(
                    sites.active_mask[:, None],
                    sites.positions == neighborhoods.geometry.site_positions[0],
                    True,
                )
            )
        )
        if neighborhoods.geometry.site_active is not None:
            same_support = same_support & jnp.all(
                neighborhoods.geometry.site_active[0] == sites.active_mask
            )
        bindings = (
            ()
            if interaction_scales is None
            else tuple(sorted(interaction_scales.items()))
        )
        known = {term.name for term in self.terms}
        if bindings and {name for name, _ in bindings} != known:
            raise ValueError(
                "Interaction scale bindings must name every program term slot."
            )
        empty = self._pair_geometry(None, position, lattice.vectors)
        timestamp = jnp.asarray(time, dtype=position.dtype)
        if timestamp.shape != ():
            raise ValueError("time must be scalar.")
        return AtomisticPotentialContext(
            system=self.system,
            positions=position,
            unwrapped_positions=unwrapped,
            site_state=sites,
            site_positions=sites.positions,
            site_type_ids=self.system.coordinate_map.plan.sites.site_type_ids,
            site_charges=self.system.coordinate_map.plan.sites.charges,
            site_pair_left=empty.left,
            site_pair_right=empty.right,
            site_pair_valid=empty.valid,
            site_pair_displacement=empty.displacement,
            site_pair_distance=empty.distance,
            site_lennard_jones_scales=empty.lennard_jones_scales,
            site_electrostatic_scales=empty.electrostatic_scales,
            species=species_,
            pair_left=empty.left,
            pair_right=empty.right,
            pair_valid=empty.valid,
            pair_keys=empty.keys,
            pair_displacement=empty.displacement,
            pair_distance=empty.distance,
            lennard_jones_scales=empty.lennard_jones_scales,
            electrostatic_scales=empty.electrostatic_scales,
            interaction_scales=AtomisticInteractionScaleState.identity(
                self.system, 0, position.dtype
            ),
            graph=None,
            cell=lattice.cell,
            cell_vectors=lattice.vectors,
            cell_inverse=lattice.inverse,
            neighborhood_successful=(
                lattice.successful
                & sites.successful
                & same_position
                & same_cell
                & same_sites
                & same_images
                & same_support
                & jnp.isfinite(timestamp)
            ),
            graph_execution=self.graph_execution,
            neighborhoods=neighborhoods.views,
            conditioning=conditioning,
            key=key,
            image_counts=counts,
            time=timestamp,
            term_states=states,
            scale_slots=tuple(name for name, _ in bindings),
            scale_values=tuple(value for _, value in bindings),
        )

    def _term_context(
        self,
        shared: AtomisticPotentialContext,
        neighborhoods: AtomisticNeighborhoodBundleState,
        term: AbstractPreparedAtomisticEnergyTerm,
        /,
    ) -> AtomisticPotentialContext:
        physical = sites = None
        graph = None
        role_successful = shared.neighborhood_successful
        for requirement in term.requirements.neighborhoods:
            view = shared.neighborhood(requirement)
            role_successful = role_successful & jnp.all(view.evidence.successful)
            if (
                requirement.counting is AtomisticPairCounting.MINIMUM_IMAGE_HALF
                and requirement.cutoff is not None
                and shared.cell is not None
            ):
                if shared.cell_inverse is None:
                    raise RuntimeError("A periodic pair context lacks its lattice solve.")
                role_successful = role_successful & _runtime_lattice_admissible(
                    shared.cell,
                    shared.cell_vectors,
                    shared.cell_inverse,
                    requirement.cutoff,
                )
            match requirement.role:
                case AtomisticNeighborhoodRole.CLASSICAL_MINIMUM_IMAGE_PAIRS:
                    if physical is not None:
                        raise ValueError(
                            "A classical kernel must name one physical pair role."
                        )
                    physical = self._pair_geometry(
                        view, shared.positions, shared.cell_vectors
                    )
                case AtomisticNeighborhoodRole.INTERACTION_SITE_PAIRS:
                    if sites is not None:
                        raise ValueError(
                            "A site kernel must name one interaction-site pair role."
                        )
                    sites = self._pair_geometry(
                        view, shared.site_positions, shared.cell_vectors
                    )
                case AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH:
                    if graph is not None:
                        raise ValueError(
                            "A graph kernel must name one directed graph role."
                        )
                    graph = self._graph(neighborhoods, requirement, shared)
                case (
                    AtomisticNeighborhoodRole.IMAGE_PAIR_SUM
                    | AtomisticNeighborhoodRole.COORDINATION
                ):
                    pass
                case _:
                    assert_never(requirement.role)
        supplied = None
        if shared.scale_slots:
            supplied = shared.scale_values[shared.scale_slots.index(term.name)]
        pair_count = 0 if physical is None else physical.left.shape[0]
        scales = self._interaction_scales(supplied, pair_count, shared.positions.dtype)
        result = replace(
            shared,
            graph=graph,
            interaction_scales=scales,
            neighborhood_successful=role_successful,
        )
        if physical is not None:
            result = replace(
                result,
                pair_left=physical.left,
                pair_right=physical.right,
                pair_valid=physical.valid,
                pair_keys=physical.keys,
                pair_displacement=physical.displacement,
                pair_distance=physical.distance,
                lennard_jones_scales=physical.lennard_jones_scales,
                electrostatic_scales=physical.electrostatic_scales,
                neighborhood_successful=result.neighborhood_successful
                & physical.successful,
            )
        if sites is not None:
            result = replace(
                result,
                site_pair_left=sites.left,
                site_pair_right=sites.right,
                site_pair_valid=sites.valid,
                site_pair_displacement=sites.displacement,
                site_pair_distance=sites.distance,
                site_lennard_jones_scales=sites.lennard_jones_scales,
                site_electrostatic_scales=sites.electrostatic_scales,
                neighborhood_successful=result.neighborhood_successful & sites.successful,
            )
        if graph is not None:
            result = replace(
                result,
                neighborhood_successful=result.neighborhood_successful
                & jnp.all(graph.valid),
            )
        return result

    def _bind_lattice(
        self,
        neighborhoods: AtomisticNeighborhoodBundleState,
        position: Array,
        cell: PeriodicCell | None,
        fractional_positions: ArrayLike | None,
        cell_vectors: ArrayLike | None,
        /,
    ) -> _RuntimeLattice:
        if neighborhoods.layout_id != self.neighborhood_bundle.layout_id:
            raise ValueError("Neighborhood state belongs to another resource layout.")
        # Same layout: the exact prepared binding (which includes the material
        # system) is compared numerically, so a foreign or stale state is refused
        # through the status without a separate executable.
        bound = neighborhoods.bundle_identity.equals(
            self.neighborhood_bundle.prepared_identity
        )
        if (
            neighborhoods.geometry.positions.shape != (1, self.system.capacity, 3)
            or neighborhoods.geometry.active.shape != (1, self.system.capacity)
            or neighborhoods.geometry.cell_vectors.shape
            != (1, self.neighborhood_bundle.lattice_rank, 3)
        ):
            raise ValueError("Resource geometry does not match this program's layout.")
        if cell is not None and (
            self.system.cell is None or cell.cell_id != self.system.cell.cell_id
        ):
            raise ValueError("Potential cell must exactly match the prepared system.")
        selected = self.system.cell
        vectors = jnp.asarray(
            neighborhoods.geometry.cell_vectors[0]
            if cell_vectors is None
            else cell_vectors,
            dtype=position.dtype,
        )
        if vectors.shape != (self.neighborhood_bundle.lattice_rank, 3):
            raise ValueError(
                "Current cell vectors do not match the resource lattice axes."
            )
        if selected is None:
            if fractional_positions is not None:
                raise ValueError(
                    "Finite Cartesian systems have no fractional-coordinate chart."
                )
            return _RuntimeLattice(
                None, vectors, None, None, bound & jnp.all(jnp.isfinite(vectors))
            )
        inverse, successful = selected.inverse_for_vectors_with_status(vectors)
        fractional = (
            contract(
                "ni,ij->nj", position - selected.origin.astype(position.dtype), inverse
            )
            if fractional_positions is None
            else jnp.asarray(fractional_positions, dtype=position.dtype)
        )
        if fractional.shape != (self.system.capacity, selected.rank):
            raise ValueError(
                "fractional_positions must match the particle and lattice axes."
            )
        return _RuntimeLattice(selected, vectors, inverse, fractional, bound & successful)

    def _pair_geometry(
        self, view: AtomisticNeighborhoodView | None, position: Array, vectors: Array, /
    ) -> _PairGeometry:
        if view is None:
            index = jnp.zeros((0,), dtype=jnp.int32)
            scalar = jnp.zeros((0,), dtype=position.dtype)
            return _PairGeometry(
                index,
                index,
                jnp.zeros((0,), dtype=jnp.bool_),
                jnp.zeros(
                    (0, 5), dtype=self.system.pair_key_space.sorted_particle_ids.dtype
                ),
                jnp.zeros((0, 3), dtype=position.dtype),
                scalar,
                scalar,
                scalar,
                jnp.asarray(True),
            )
        pairs = view.pair_relation
        if pairs is None:
            raise ValueError("A pair kernel requires a certified pair-once view.")
        valid = view.mask & pairs.valid
        displacement = jnp.where(
            valid[:, None], -view.displacement(position, vectors[None]), 0.0
        )
        distance = _distance(displacement)
        if view.role is AtomisticNeighborhoodRole.CLASSICAL_MINIMUM_IMAGE_PAIRS:
            key_result = self.system.pair_key_space.keys(pairs)
            keys, key_valid = key_result.keys, key_result.successful
        else:
            keys = jnp.zeros(
                (0, 5), dtype=self.system.pair_key_space.sorted_particle_ids.dtype
            )
            key_valid = jnp.asarray(True)
        unit = jnp.where(valid, jnp.ones((), dtype=position.dtype), 0.0)
        lj = (
            unit
            if view.lennard_jones_scales is None
            else view.lennard_jones_scales.astype(position.dtype)
        )
        electrostatic = (
            unit
            if view.electrostatic_scales is None
            else view.electrostatic_scales.astype(position.dtype)
        )
        return _PairGeometry(
            pairs.left_indices,
            pairs.right_indices,
            valid,
            keys,
            displacement,
            distance,
            lj,
            electrostatic,
            jnp.all(view.evidence.successful) & key_valid,
        )

    def _graph(
        self,
        neighborhoods: AtomisticNeighborhoodBundleState,
        requirement: AtomisticNeighborhoodRequirement,
        context: AtomisticPotentialContext,
        /,
    ) -> AtomisticGraph:
        execution = self.graph_execution
        if execution is None:
            raise RuntimeError("A graph role lacks its admitted execution plan.")
        topology = _neighborhood_graph_topology(
            self.system, self.neighborhood_bundle, neighborhoods, requirement, execution
        )
        return bind_atomistic_graph(
            topology,
            execution,
            context.positions,
            cutoff=requirement.bounded_cutoff(),
            cell_vectors=context.cell_vectors,
            atom_type_ids=context.species,
            image_counts=context.image_counts,
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

    def _execute_terms(
        self,
        context: AtomisticPotentialContext,
        neighborhoods: AtomisticNeighborhoodBundleState,
        selected: tuple[int, ...],
        /,
    ) -> tuple[AtomisticPotentialContext, dict[int, AtomisticTermEvaluation], Array]:
        """Execute the selected terms and their transitive scientific field producers."""
        needed = set(selected)
        for index in reversed(self.plan.evaluation_order):
            if index in needed:
                needed.update(
                    self.plan.term_names.index(reference.producer)
                    for reference in self.terms[index].requirements.field_dependencies
                )
        required_states = {
            self.terms[index].name
            for index in needed
            if self.terms[index].requirements.state_required
        }
        if required_states - set(context.term_states.slots):
            raise ValueError("An executed term has no bound numerical state.")
        evaluated: dict[int, AtomisticTermEvaluation] = {}
        published: list[AtomisticObservableValue] = []
        producers: list[str] = []
        graph_overflow = jnp.asarray(False)
        resource_successful = context.neighborhood_successful
        for index in self.plan.evaluation_order:
            if index not in needed:
                continue
            term = self.terms[index]
            scoped = self._term_context(context, neighborhoods, term)
            value = term.energy(scoped)
            resource_successful = resource_successful & scoped.neighborhood_successful
            state_successful = jnp.asarray(True)
            if term.requirements.state_required:
                state_successful = context.term_states.bound_to(
                    term.name, term.term_identity
                )
            value = replace(
                value,
                successful=(
                    value.successful & scoped.neighborhood_successful & state_successful
                ),
                evidence=(
                    *(
                        scoped.neighborhood(requirement).evidence
                        for requirement in term.requirements.neighborhoods
                    ),
                    *value.evidence,
                ),
            )
            evaluated[index] = value
            if scoped.graph is not None:
                graph_overflow = graph_overflow | jnp.any(scoped.graph.overflow)
            for field in value.fields:
                if field.case_count != 1 or field.atom_capacity != self.system.capacity:
                    raise ValueError(
                        "A producer field does not belong to this system layout."
                    )
                published.append(
                    AtomisticObservableValue(
                        field.observable,
                        field.port,
                        field.values,
                        field.available & value.successful,
                        level=field.level,
                        case_count=1,
                        atom_capacity=self.system.capacity,
                        unit=field.unit,
                    )
                )
                producers.append(term.name)
            if value.fields:
                context = replace(
                    context,
                    field_values=tuple(published),
                    field_producers=tuple(producers),
                )
        return (
            replace(context, neighborhood_successful=resource_successful),
            evaluated,
            graph_overflow,
        )

    def _program_result(
        self,
        context: AtomisticPotentialContext,
        evaluated: dict[int, AtomisticTermEvaluation],
        graph_overflow: Array,
        selected: tuple[int, ...],
        /,
    ) -> tuple[Array, AtomisticProgramAuxiliary]:
        """Keep the evaluated ledger distinct from the terms contributing to its scalar."""
        executed = tuple(index for index in range(len(self.terms)) if index in evaluated)
        selected_set = set(selected)
        term_energies = jnp.stack(tuple(evaluated[index].energy for index in executed))
        values = jnp.stack(tuple(evaluated[index].energy for index in selected))
        coefficients = self.coefficients[jnp.asarray(selected, dtype=jnp.int32)].astype(
            values.dtype
        )
        energy = jnp.sum(coefficients * values)
        atom_energy = jnp.sum(
            coefficients[:, None]
            * jnp.stack(tuple(evaluated[index].atom_energy for index in selected)),
            axis=0,
        )
        successful = context.neighborhood_successful & jnp.all(
            jnp.stack(tuple(evaluated[index].successful for index in executed))
        )
        successful = successful & jnp.isfinite(energy)
        energy = _admitted_scalar(
            energy, successful, (context.positions, context.cell_vectors)
        )
        ordered_fields = sorted(
            zip(context.field_producers, context.field_values, strict=True),
            key=lambda item: (item[0], item[1].port.port_id),
        )
        return energy, AtomisticProgramAuxiliary(
            term_energies=term_energies,
            term_slots=tuple(self.terms[index].name for index in executed),
            term_identities=_stacked_identities(
                tuple(self.terms[index].term_identity for index in executed)
            ),
            contributing_terms=jnp.asarray(
                tuple(index in selected_set for index in executed), dtype=jnp.bool_
            ),
            atom_energy=atom_energy,
            successful=successful,
            graph_overflow=graph_overflow,
            neighborhood_successful=context.neighborhood_successful,
            term_evidence=tuple(evaluated[index].evidence for index in executed),
            fields=tuple(value for _, value in ordered_fields),
            field_producers=tuple(name for name, _ in ordered_fields),
            atom_energy_complete=all(
                evaluated[index].atom_energy_complete
                and self.terms[index].capabilities.local_energy
                for index in selected
            ),
        )

    def energy(
        self,
        positions: ArrayLike,
        neighborhoods: AtomisticNeighborhoodBundleState,
        /,
        **context_kwargs: Any,
    ) -> tuple[Array, AtomisticProgramAuxiliary]:
        selected = tuple(range(len(self.terms)))
        context = self._base_context(positions, neighborhoods, **context_kwargs)
        context, evaluated, graph_overflow = self._execute_terms(
            context, neighborhoods, selected
        )
        return self._program_result(context, evaluated, graph_overflow, selected)

    def force_group_energy(
        self,
        group: int,
        positions: ArrayLike,
        neighborhoods: AtomisticNeighborhoodBundleState,
        /,
        **context_kwargs: Any,
    ) -> tuple[Array, AtomisticProgramAuxiliary]:
        selected = tuple(
            index for index, term in enumerate(self.terms) if term.force_group == group
        )
        if not selected:
            raise ValueError(f"Force group {group} is absent from the Hamiltonian.")
        context = self._base_context(positions, neighborhoods, **context_kwargs)
        context, evaluated, graph_overflow = self._execute_terms(
            context, neighborhoods, selected
        )
        energy, auxiliary = self._program_result(
            context, evaluated, graph_overflow, selected
        )
        return energy, auxiliary

    def evaluate(
        self,
        positions: ArrayLike,
        neighborhoods: AtomisticNeighborhoodBundleState,
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
            neighborhoods=neighborhoods,
            strained=compute_stress,
            cell_derivative=self.plan.capabilities.cell_derivative,
        )

        def closure(
            value: Array, strain: Array | None
        ) -> tuple[Array, AtomisticProgramAuxiliary]:
            bound, kwargs = point.bind(value, strain)
            return self.energy(bound, neighborhoods, **kwargs)

        strain_derivative = None
        if point.strained:
            (energy, auxiliary), (gradient, strain_derivative) = jax.value_and_grad(
                closure, argnums=(0, 1), has_aux=True
            )(position, point.zero_strain())
        else:
            (energy, auxiliary), gradient = jax.value_and_grad(closure, has_aux=True)(
                position, None
            )
        term_energies = auxiliary.term_energies
        atom_energy = auxiliary.atom_energy
        program_successful = auxiliary.successful
        graph_overflow = auxiliary.graph_overflow
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
            neighborhood_successful=auxiliary.neighborhood_successful,
            graph_overflow=graph_overflow,
            strain_derivative=(
                None
                if strain_derivative is None
                else jnp.where(accepted, strain_derivative, nan)
            ),
            stress=None if stress is None else jnp.where(accepted, stress, nan),
            program_identity=self.prepared_identity,
            stress_convention=point.stress_convention,
            term_evidence=auxiliary.term_evidence,
            fields=auxiliary.fields,
            field_producers=auxiliary.field_producers,
            atom_energy_complete=auxiliary.atom_energy_complete,
        )


__all__ = [
    "AtomisticProgramAuxiliary",
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
