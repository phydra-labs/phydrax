#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..typing import checked
from ._dynamics import (
    _flag,
    _neighborhood_reasons,
    _strain_reasons,
    advance_atomistic_energy_ledger,
    AtomisticDynamicsState,
    AtomisticKinematics,
    AtomisticStepRejectionReason,
    neighborhood_route_work,
    PreparedAtomisticDynamics,
    VelocityVerletPlan,
)
from ._neighborhood_bundle import AtomisticNeighborhoodBundleState
from ._potential import AtomisticPotentialCapabilities, AtomisticPotentialRequirements
from ._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AbstractPreparedAtomisticHamiltonian,
    AtomisticDerivativePoint,
    AtomisticPotentialContext,
    AtomisticTermEvaluation,
    PreparedAtomisticPotentialProgram,
)
from ._rollout import AtomisticRolloutPlan, AtomisticRolloutResult
from ._system import PreparedAtomisticSystem
from ._thermodynamic import PreparedThermodynamicStateTable


class ForceGroupEvaluation(StrictModule):
    """One force group's scalar, forces and its exact evaluated-term ledger.

    ``term_energies`` rows are the actually evaluated terms (the selected
    group plus transitive field producers), named by their declared program
    slots ``term_slots`` (static) and their exact scientific digests
    ``term_identities`` (dynamic stacked words; ``term_ids`` decodes them on
    the host); ``contributing_terms`` marks the rows summed into ``energy``.
    """

    energy: Array
    forces: Array
    term_energies: Array
    contributing_terms: Array
    successful: Array
    term_slots: tuple[str, ...] = eqx.field(static=True)
    term_identities: IdentityWords
    group: int = eqx.field(static=True)

    @property
    def term_ids(self) -> tuple[str, ...]:
        """Host decoding of the evaluated rows' scientific term digests."""
        return tuple(
            IdentityWords(self.term_identities.words[row]).fingerprint()
            for row in range(len(self.term_slots))
        )


def evaluate_force_group(
    potential: AbstractPreparedAtomisticHamiltonian,
    group: int,
    positions: ArrayLike,
    neighborhoods: AtomisticNeighborhoodBundleState,
    /,
    **context_kwargs: Any,
) -> ForceGroupEvaluation:
    """Energy and conservative forces of one force group of the scalar program.

    The group energy is the Hamiltonian's own ``force_group_energy`` (every
    term's own resource view and transitive field-producer execution), so no
    term is evaluated against a shared foreign context.
    """
    if not any(term.force_group == int(group) for term in potential.terms):
        raise ValueError(f"Potential program has no force group {group}.")
    position = jnp.asarray(positions, dtype=potential.system.plan.coordinate_dtype)
    point = AtomisticDerivativePoint(
        potential.system,
        position,
        context_kwargs,
        neighborhoods=neighborhoods,
        strained=False,
        cell_derivative=False,
    )

    term_slots: list[tuple[str, ...]] = []

    def energy_closure(
        value: Array,
    ) -> tuple[Array, tuple[Array, Array, Array, IdentityWords]]:
        bound, kwargs = point.bind(value, None)
        energy, auxiliary = potential.force_group_energy(
            int(group), bound, neighborhoods, **kwargs
        )
        term_slots.append(auxiliary.term_slots)
        return energy, (
            auxiliary.term_energies,
            jnp.asarray(auxiliary.contributing_terms, dtype=jnp.bool_),
            auxiliary.successful,
            auxiliary.term_identities,
        )

    (energy, auxiliary), gradient = jax.value_and_grad(energy_closure, has_aux=True)(
        position
    )
    terms, contributing, successful, identities = auxiliary
    finite = jnp.isfinite(energy) & jnp.all(jnp.isfinite(gradient))
    return ForceGroupEvaluation(
        energy,
        -gradient,
        terms,
        contributing,
        successful & point.successful & finite,
        term_slots[-1],
        identities,
        int(group),
    )


class RegionMaskedPotential(AbstractAtomisticEnergyTerm):
    """A local term restricted to the atoms of one declared region.

    ``region_id`` is the declared region selector (static); the canonical
    term digest is the dynamic ``term_identity``.
    """

    term: AbstractAtomisticEnergyTerm
    region_id: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    @checked
    def __init__(
        self,
        term: AbstractAtomisticEnergyTerm,
        region_id: int,
        /,
        *,
        name: str | None = None,
    ) -> None:
        if not term.capabilities.local_energy:
            raise ValueError("Region masking requires a local atom-energy decomposition.")
        identifier = (
            f"region-{int(region_id)}-{term.name}" if name is None else str(name).strip()
        )
        if not identifier:
            raise ValueError("Masked potential name must be non-empty.")
        self.term = term
        self.region_id = int(region_id)
        self.name = identifier
        self.force_group = term.force_group
        self.capabilities = term.capabilities
        self.requirements = term.requirements
        self.term_identity = IdentityWords.of(
            {
                "kind": "region-masked-potential",
                "term": term.term_id,
                "region_id": int(region_id),
                "name": identifier,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedRegionMaskedPotential":
        mask = system.plan.region_ids == self.region_id
        if not bool(jnp.any(mask & system.active_mask)):
            raise ValueError("Region mask selects no active atoms.")
        return PreparedRegionMaskedPotential(self, self.term.prepare(system), mask)


class PreparedRegionMaskedPotential(AbstractPreparedAtomisticEnergyTerm):
    """A region-masked term bound to one system.

    The region ``mask`` is a dynamic leaf; ``term_identity`` and
    ``prepared_identity`` are dynamic.
    """

    plan: RegionMaskedPotential
    term: AbstractPreparedAtomisticEnergyTerm
    mask: Array
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        plan: RegionMaskedPotential,
        term: AbstractPreparedAtomisticEnergyTerm,
        mask: Array,
        /,
    ) -> None:
        self.plan = plan
        self.term = term
        self.mask = jnp.asarray(mask, dtype=jnp.bool_)
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-region-masked-potential",
                "plan": plan.term_id,
                "term": term.prepared_id,
            }
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        value = self.term.energy(context)
        atom_energy = jnp.where(self.mask, value.atom_energy, 0.0)
        return AtomisticTermEvaluation(
            jnp.sum(atom_energy), atom_energy, value.successful
        )


class RESPAPlan(StrictModule, NonTrainableState):
    outer_step_size: float = eqx.field(static=True)
    inner_steps: int = eqx.field(static=True)
    fast_group: int = eqx.field(static=True)
    slow_group: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        outer_step_size: float,
        inner_steps: int,
        /,
        *,
        fast_group: int = 0,
        slow_group: int = 1,
    ) -> None:
        step = float(outer_step_size)
        inner = int(inner_steps)
        fast = int(fast_group)
        slow = int(slow_group)
        if (
            not np.isfinite(step)
            or step <= 0.0
            or inner <= 0
            or fast < 0
            or slow < 0
            or fast == slow
        ):
            raise ValueError("RESPA step, groups, or inner_steps are invalid.")
        self.outer_step_size = step
        self.inner_steps = inner
        self.fast_group = fast
        self.slow_group = slow
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-respa",
                "outer_step_size": step,
                "inner_steps": inner,
                "fast_group": fast,
                "slow_group": slow,
            }
        )


class RESPAStepEvaluation(StrictModule):
    state: AtomisticDynamicsState
    successful: Array
    rejection_reasons: Array
    fast_work: Array
    slow_work: Array
    plan_id: str = eqx.field(static=True)


def respa_step(
    dynamics: PreparedAtomisticDynamics,
    state: AtomisticDynamicsState,
    thermodynamic: PreparedThermodynamicStateTable,
    plan: RESPAPlan,
    /,
) -> RESPAStepEvaluation:
    """One reversible RESPA step of a velocity Verlet (NVE) runtime.

    Every inner position update refreshes a pure candidate of the accepted
    neighborhood bundle before its fast-force evaluation; the accepted force
    cache must be valid on entry and the final complete evaluation (with the
    runtime's declared strain response) becomes the committed cache.  Thermostat
    and cell methods have their own schedules and are refused here rather
    than silently dropped.
    """
    if not isinstance(dynamics, PreparedAtomisticDynamics):
        raise TypeError("dynamics must be PreparedAtomisticDynamics.")
    if not isinstance(state, AtomisticDynamicsState):
        raise TypeError("state must be AtomisticDynamicsState.")
    if not isinstance(thermodynamic, PreparedThermodynamicStateTable):
        raise TypeError("thermodynamic must be PreparedThermodynamicStateTable.")
    if not isinstance(plan, RESPAPlan):
        raise TypeError("plan must be RESPAPlan.")
    dynamics._require_state(state, thermodynamic)
    if not isinstance(dynamics.integrator, VelocityVerletPlan):
        raise ValueError(
            "RESPA is a velocity Verlet splitting; thermostat and cell methods "
            "require their own qualified multiple-time-step schedule."
        )
    if dynamics.constraints is not None:
        raise ValueError("RESPA constraints require a separately qualified splitting.")
    dynamics.require_conservative_force_definition("RESPA force-group splitting")
    row = thermodynamic.state_at_replica(state.thermodynamic_state_index)
    force_valid = dynamics.force_cache_valid(state, row.controls)
    outer = jnp.asarray(plan.outer_step_size, dtype=state.kinematics.positions.dtype)
    inner = outer / plan.inner_steps
    force_scale = dynamics.system.plan.units.force_to_momentum_rate
    inverse_mass = dynamics.system.inverse_masses[:, None]
    mobile = dynamics.system.mobile_mask[:, None]
    vectors = state.cell_vectors
    position = state.kinematics.positions
    momentum = state.kinematics.momenta
    images = state.kinematics.image_counts
    neighborhoods = state.neighborhoods
    unwrapped = dynamics._unwrapped(state.kinematics, vectors)

    def kwargs(current_unwrapped: Array, current_images: Array) -> dict[str, Any]:
        return dynamics._potential_kwargs(
            current_unwrapped,
            current_images,
            state.species,
            vectors,
            row.controls,
            state.term_states,
        )

    slow = evaluate_force_group(
        dynamics.potential,
        plan.slow_group,
        position,
        neighborhoods,
        **kwargs(unwrapped, images),
    )
    momentum = jnp.where(mobile, momentum + 0.5 * outer * force_scale * slow.forces, 0.0)
    fast_work = jnp.zeros((), dtype=jnp.int32)
    successful = slow.successful
    reasons = jnp.zeros((), dtype=jnp.int32)
    for _ in range(plan.inner_steps):
        fast = evaluate_force_group(
            dynamics.potential,
            plan.fast_group,
            position,
            neighborhoods,
            **kwargs(unwrapped, images),
        )
        successful = successful & fast.successful
        momentum = momentum + 0.5 * inner * force_scale * fast.forces
        unwrapped = jnp.where(
            mobile, unwrapped + inner * momentum * inverse_mass, unwrapped
        )
        position, images = dynamics._wrap(unwrapped, vectors, images)
        neighborhoods = dynamics.refresh_neighborhoods(
            neighborhoods, position, cell_vectors=vectors, image_counts=images
        )
        reasons = reasons | _neighborhood_reasons(neighborhoods)
        fast = evaluate_force_group(
            dynamics.potential,
            plan.fast_group,
            position,
            neighborhoods,
            **kwargs(unwrapped, images),
        )
        momentum = jnp.where(
            mobile, momentum + 0.5 * inner * force_scale * fast.forces, 0.0
        )
        successful = successful & fast.successful
        fast_work = fast_work + neighborhood_route_work(neighborhoods)
    slow = evaluate_force_group(
        dynamics.potential,
        plan.slow_group,
        position,
        neighborhoods,
        **kwargs(unwrapped, images),
    )
    momentum = jnp.where(mobile, momentum + 0.5 * outer * force_scale * slow.forces, 0.0)
    total = dynamics.evaluate_configuration(
        position,
        neighborhoods,
        unwrapped_positions=unwrapped,
        image_counts=images,
        species=state.species,
        cell_vectors=vectors,
        controls=row.controls,
        term_states=state.term_states,
    )
    revision = state.configuration_revision + 1
    force = dynamics.force_state(
        total,
        neighborhoods,
        revision,
        row.hamiltonian_index,
        row.controls,
        state.random_key,
        term_states=state.term_states,
        ports=None,
    )
    kinetic = dynamics.kinetic_energy(momentum)
    terms = dynamics.extended_terms(state.controllers, vectors, thermodynamic, row)
    ledger = advance_atomistic_energy_ledger(
        state.energy,
        kinetic,
        total.energy,
        terms,
        thermostat_heat=jnp.zeros_like(kinetic),
        reservoir_heat=jnp.zeros_like(kinetic),
        constraint_work=jnp.zeros_like(kinetic),
        barostat_work=jnp.zeros_like(kinetic),
        external_work=jnp.zeros_like(kinetic),
        balanced=True,
        accepted_increment=1,
    )
    reasons = (
        reasons
        | _flag(
            ~(successful & slow.successful & total.successful),
            AtomisticStepRejectionReason.POTENTIAL,
        )
        | _flag(~row.valid, AtomisticStepRejectionReason.THERMODYNAMIC_STATE)
        | _flag(~force_valid, AtomisticStepRejectionReason.STALE_FORCE)
        | _flag(
            ~dynamics.binding_valid(state, thermodynamic),
            AtomisticStepRejectionReason.BINDING,
        )
    )
    reasons = reasons | _strain_reasons(
        dynamics.strain_response, force.strain_available, reasons
    )
    accepted = reasons == 0
    candidate = eqx.tree_at(
        lambda value: (
            value.time,
            value.step_index,
            value.configuration_revision,
            value.kinematics,
            value.neighborhoods,
            value.force,
            value.energy,
        ),
        state,
        (
            state.time + outer,
            state.step_index + 1,
            revision,
            AtomisticKinematics(position, momentum, images),
            neighborhoods,
            force,
            ledger,
        ),
    )
    return RESPAStepEvaluation(
        tree_where(accepted, candidate, state),
        accepted,
        reasons,
        fast_work,
        neighborhood_route_work(neighborhoods),
        plan.plan_id,
    )


class SubtractivePotentialEvaluation(StrictModule):
    energy: Array
    full_forces: Array
    region_forces: Array
    successful: Array


def evaluate_subtractive_potential(
    low_all: PreparedAtomisticPotentialProgram,
    low_region: PreparedAtomisticPotentialProgram,
    high_region: PreparedAtomisticPotentialProgram,
    full_positions: ArrayLike,
    region_positions: ArrayLike,
    full_neighborhood: AtomisticNeighborhoodBundleState,
    region_neighborhood: AtomisticNeighborhoodBundleState,
    region_indices: ArrayLike,
    /,
) -> SubtractivePotentialEvaluation:
    raw_indices = jnp.asarray(region_indices)
    if raw_indices.ndim != 1 or raw_indices.dtype.kind not in "iu":
        raise TypeError("region_indices must be an integer vector.")
    # Both region potentials must bind one prepared system; compared on the
    # identity words so the check is traceable and fails the result status.
    same_region_system = low_region.system.prepared_identity.equals(
        high_region.system.prepared_identity
    )
    full_position = jnp.asarray(full_positions)
    region_position = jnp.asarray(region_positions)
    if (
        full_position.shape != (low_all.system.capacity, 3)
        or region_position.shape != (low_region.system.capacity, 3)
        or raw_indices.shape != (low_region.system.capacity,)
    ):
        raise ValueError("Subtractive region positions and indices are misaligned.")
    indices = raw_indices.astype(jnp.int32)
    full_ids = low_all.system.plan.particle_ids
    region_ids = low_region.system.plan.particle_ids
    indices = eqx.error_if(
        indices,
        jnp.any((indices < 0) | (indices >= low_all.system.capacity))
        | jnp.any(jnp.diff(jnp.sort(indices)) == 0)
        | ~jnp.array_equal(full_ids[indices], region_ids),
        "region_indices must be unique, in range, and match the region particle IDs.",
    )
    low_all_evaluation = low_all.evaluate(full_position, full_neighborhood)
    low_region_evaluation = low_region.evaluate(region_position, region_neighborhood)
    high_region_evaluation = high_region.evaluate(region_position, region_neighborhood)
    region_force = high_region_evaluation.forces - low_region_evaluation.forces
    full_force = low_all_evaluation.forces.at[indices].add(region_force)
    energy = (
        low_all_evaluation.energy
        + high_region_evaluation.energy
        - low_region_evaluation.energy
    )
    successful = (
        same_region_system
        & low_all_evaluation.successful
        & low_region_evaluation.successful
        & high_region_evaluation.successful
        & jnp.all(jnp.isfinite(full_force))
    )
    return SubtractivePotentialEvaluation(energy, full_force, region_force, successful)


def local_species_energy_delta(
    potential: PreparedAtomisticPotentialProgram,
    positions: ArrayLike,
    neighborhood: AtomisticNeighborhoodBundleState,
    species: ArrayLike,
    particle_index: ArrayLike,
    proposed_species: ArrayLike,
    /,
) -> Array:
    if not potential.plan.capabilities.dynamic_species:
        raise ValueError("Potential program does not support dynamic species.")
    current = jnp.asarray(species, dtype=jnp.int32)
    if current.shape != (potential.system.capacity,):
        raise ValueError("species must match the atomistic system capacity.")
    index = jnp.asarray(particle_index)
    replacement = jnp.asarray(proposed_species)
    if index.shape != () or index.dtype.kind not in "iu":
        raise TypeError("particle_index must be an integer scalar.")
    if replacement.shape != () or replacement.dtype.kind not in "iu":
        raise TypeError("proposed_species must be an integer scalar.")
    index = eqx.error_if(
        index.astype(jnp.int32),
        (index < 0)
        | (index >= potential.system.capacity)
        | ~potential.system.active_mask[index],
        "particle_index must identify an active particle.",
    )
    proposed = current.at[index].set(replacement.astype(jnp.int32))
    current_energy = potential.energy(positions, neighborhood, species=current)[0]
    proposed_energy = potential.energy(positions, neighborhood, species=proposed)[0]
    return proposed_energy - current_energy


_IDENTITY_WORD_SHAPE = (8,)


def _provider_binding(
    identity: IdentityWords | None, label: str | None, /
) -> tuple[IdentityWords | None, str | None]:
    """Validate one exclusive provider binding: an owning digest or a declared label."""
    if (identity is None) == (label is None):
        raise ValueError("Supply exactly one of provider_identity or provider_label.")
    if identity is not None:
        if (
            not isinstance(identity, IdentityWords)
            or identity.words.shape != _IDENTITY_WORD_SHAPE
        ):
            raise TypeError("provider_identity must be one unbatched IdentityWords.")
        return identity, None
    if not isinstance(label, str) or not label.strip():
        raise ValueError("provider_label must be a non-empty string.")
    return None, label.strip()


def _provider_id(identity: IdentityWords | None, label: str | None, /) -> str:
    """Host string of one provider binding: the label or the decoded digest."""
    if label is not None:
        return label
    if identity is None:
        raise RuntimeError("Internal invariant failed: provider binding is empty.")
    return identity.fingerprint()


def _same_provider(
    identity: IdentityWords | None,
    label: str | None,
    other_identity: IdentityWords | None,
    other_label: str | None,
    /,
) -> Array:
    """Traced equality of two provider bindings.

    Digest bindings compare their identity words; labelled bindings compare
    their static declared labels; a digest never matches a label.
    """
    if identity is None or other_identity is None:
        return jnp.asarray(label is not None and label == other_label)
    return identity.equals(other_identity)


class ExternalAtomisticEvaluation(StrictModule):
    """One external or native host evaluation in the bound system's units.

    ``energy`` is the scalar system energy in ``units.scale.energy_unit`` and
    ``forces`` the ``(capacity, 3)`` negative position gradient in
    ``units.scale.force_unit``. ``stress`` is the native tensile Cauchy stress
    ``(1 / V) dE/d(strain)`` in ``units.pressure_unit`` for the row-lattice
    deformation ``H' = H @ (I + strain).T``, the same convention as
    ``atomistic_cell_energy_and_stress``; ``None`` declares it unavailable.
    Configurational virials ``W = -V * stress`` exist only at protocol
    boundaries that require them, never in this record.

    PyTree: the producing provider is bound by exactly one of
    ``provider_identity`` (dynamic `IdentityWords` of a provider identified by
    an owning canonical digest, e.g. `NativeAtomisticProvider`) or
    ``provider_label`` (static declared label of a labelled provider, e.g.
    an i-PI transport); the other is ``None``.  Digest-bound evaluations are
    therefore constructible inside compiled code and share one executable per
    layout.  The host property ``provider_id`` returns the label or the
    decoded digest; `bound_to` is the traced binding check.
    """

    energy: Array
    forces: Array
    stress: Array | None
    successful: Array
    provider_identity: IdentityWords | None
    provider_label: str | None = eqx.field(static=True)

    def __init__(
        self,
        energy: ArrayLike,
        forces: ArrayLike,
        stress: ArrayLike | None,
        successful: ArrayLike,
        *,
        provider_identity: IdentityWords | None = None,
        provider_label: str | None = None,
    ) -> None:
        energy_ = jnp.asarray(energy)
        forces_ = jnp.asarray(forces)
        stress_ = None if stress is None else jnp.asarray(stress)
        successful_ = jnp.asarray(successful)
        if energy_.shape != () or forces_.ndim != 2 or forces_.shape[-1] != 3:
            raise ValueError(
                "External atomistic energy must be scalar and forces (capacity, 3)."
            )
        if stress_ is not None and stress_.shape != (3, 3):
            raise ValueError("External atomistic stress must be a (3, 3) tensor.")
        if successful_.shape != () or successful_.dtype != jnp.bool_:
            raise ValueError("External atomistic success must be a boolean scalar.")
        identity, label = _provider_binding(provider_identity, provider_label)
        self.energy = energy_
        self.forces = forces_
        self.stress = stress_
        self.successful = successful_
        self.provider_identity = identity
        self.provider_label = label

    @property
    def provider_id(self) -> str:
        """Host string of the producing provider: its label or decoded digest."""
        return _provider_id(self.provider_identity, self.provider_label)

    def bound_to(self, provider: AbstractExternalAtomisticProvider, /) -> Array:
        """Traced: this evaluation carries exactly ``provider``'s binding."""
        return _same_provider(
            self.provider_identity,
            self.provider_label,
            provider.provider_identity,
            provider.provider_label,
        )


class AbstractExternalAtomisticProvider(StrictModule, NonTrainableState):
    """Provider boundary returning `ExternalAtomisticEvaluation` records.

    A provider declares exactly one binding: dynamic ``provider_identity``
    words when it is identified by an owning canonical digest, or a static
    declared ``provider_label``; the other is ``None``.  The base owns the
    host ``provider_id`` (the label or the decoded digest).
    """

    provider_identity: eqx.AbstractVar[IdentityWords | None]
    provider_label: eqx.AbstractVar[str | None]
    conservative: eqx.AbstractVar[bool]
    differentiable: eqx.AbstractVar[bool]

    def __check_init__(self) -> None:
        _provider_binding(self.provider_identity, self.provider_label)

    @property
    def provider_id(self) -> str:
        """Host string of this provider: its label or decoded digest."""
        return _provider_id(self.provider_identity, self.provider_label)

    @abc.abstractmethod
    def evaluate(
        self,
        system: PreparedAtomisticSystem,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        /,
    ) -> ExternalAtomisticEvaluation:
        raise NotImplementedError


class AtomisticSegmentedResult(StrictModule):
    final_state: AtomisticDynamicsState
    segments: tuple[AtomisticRolloutResult, ...]
    successful: Array
    protocol_id: str = eqx.field(static=True)


def run_atomistic_segments(
    rollout: AtomisticRolloutPlan,
    initial_state: AtomisticDynamicsState,
    segment_count: int,
    /,
) -> AtomisticSegmentedResult:
    if not isinstance(rollout, AtomisticRolloutPlan):
        raise TypeError("rollout must be AtomisticRolloutPlan.")
    count = int(segment_count)
    if count <= 0:
        raise ValueError("segment_count must be positive.")
    state = initial_state
    segments: list[AtomisticRolloutResult] = []
    successful = jnp.asarray(True)
    for _ in range(count):
        result = rollout.rollout(state)
        segments.append(result)
        state = result.final_state
        successful = successful & result.successful
        if not bool(result.successful):
            break
    protocol_id = canonical_fingerprint(
        {
            "kind": "atomistic-segmented-protocol",
            "rollout": rollout.rollout_id,
            "segments": count,
        }
    )
    return AtomisticSegmentedResult(state, tuple(segments), successful, protocol_id)


__all__ = [
    "AbstractExternalAtomisticProvider",
    "AtomisticSegmentedResult",
    "ExternalAtomisticEvaluation",
    "ForceGroupEvaluation",
    "RegionMaskedPotential",
    "RESPAPlan",
    "RESPAStepEvaluation",
    "SubtractivePotentialEvaluation",
    "evaluate_force_group",
    "evaluate_subtractive_potential",
    "local_species_energy_delta",
    "respa_step",
    "run_atomistic_segments",
]
