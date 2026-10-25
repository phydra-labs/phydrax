#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bundle-backed isotropic Monte Carlo volume move (scheduled cell owner).

One move proposes ``ln V' = ln V + u``, ``u ~ U(-a, a)``, scales the declared
entities of the phase-space measure (molecule centers, or every active atom)
at fixed fractional coordinates, refreshes a pure candidate of the accepted
neighborhood bundle at the proposed geometry and evaluates the complete
scalar Hamiltonian (every term, including any embedded bias) there.  With the
symmetric ``ln V`` proposal the scaled-coordinate NPT measure ``V^N dV``
contributes the exact Jacobian ``(N + 1) ln(V'/V)``::

    ln alpha = -beta (U' - U + P (V' - V)) + (N + 1) ln(V' / V).

A hydrostatic scalar pressure enters as ``P V``; tensor targets are refused.
Draws are addressed by run key, realization, the move operator identity, the
typed move counter of the state and the draw index.  A completed
Metropolis decision (accept or reject) advances the counter; a capacity,
scientific, resource or stale-force failure advances nothing, so a
capacity-only replay re-attempts the identical move.  Kinematics, cell,
bundle, force cache, counters and ledger commit together.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_allfinite, tree_where
from ..discretization import PeriodicCell
from ._cell_dynamics import cell_volume
from ._dynamics import (
    _flag,
    _neighborhood_reasons,
    _strain_reasons,
    advance_atomistic_energy_ledger,
    AtomisticDynamicsState,
    AtomisticKinematics,
    AtomisticStepRejectionReason,
    PreparedAtomisticDynamics,
)
from ._thermodynamic import PreparedThermodynamicStateTable


MONTE_CARLO_VOLUME_OPERATOR_ID = 64


class IsotropicMonteCarloBarostatPlan(StrictModule, NonTrainableState):
    """Uniform ``ln V`` proposal width and declared random stream of the move.

    ``maximum_log_volume_change`` is a dynamic ``float64`` scalar (proposal
    width revisions share one executable); ``realization_id`` is a static
    stream label.  The digest of the exact host values is the dynamic
    ``plan_identity`` (host ``plan_id``, unchanged digest).
    """

    maximum_log_volume_change: Array
    realization_id: int = eqx.field(static=True)
    plan_identity: IdentityWords

    def __init__(
        self,
        maximum_log_volume_change: float,
        /,
        *,
        realization_id: int = 0,
    ) -> None:
        maximum = float(maximum_log_volume_change)
        realization = int(realization_id)
        if not math.isfinite(maximum) or maximum <= 0.0 or realization < 0:
            raise ValueError("Monte Carlo barostat parameters are invalid.")
        self.maximum_log_volume_change = jnp.asarray(maximum, dtype=jnp.float64)
        self.realization_id = realization
        self.plan_identity = IdentityWords.of(
            {
                "kind": "isotropic-monte-carlo-barostat",
                "proposal": "uniform-log-volume",
                "jacobian": "scaled-entities-plus-one",
                "maximum_log_volume_change": maximum,
                "realization_id": realization,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical plan digest."""
        return self.plan_identity.fingerprint()


class AtomisticBarostatEvaluation(StrictModule):
    """One Monte Carlo volume move; ``plan_identity`` (host ``plan_id``) is dynamic."""

    candidate_state: AtomisticDynamicsState
    accepted_state: AtomisticDynamicsState
    accepted: Array
    log_acceptance_probability: Array
    volume_before: Array
    volume_after: Array
    energy_before: Array
    energy_after: Array
    successful: Array
    rejection_reasons: Array
    plan_identity: IdentityWords

    @property
    def plan_id(self) -> str:
        return self.plan_identity.fingerprint()


def _admit(
    dynamics: PreparedAtomisticDynamics,
    thermodynamic: PreparedThermodynamicStateTable,
    /,
) -> tuple[PeriodicCell, Array]:
    """Admit the cell and return traced constraint-molecule admission.

    Holonomic constraints must stay within one scaled molecule.  The
    constraint rows are material content, so the check is a traced boolean
    folded into the move's ``BAROSTAT`` rejection, never a host decode.
    """
    from ._cell_dynamics import MTKNPHPlan, MTKNPTPlan

    cell = dynamics.system.cell
    if cell is None or not cell.fully_periodic:
        raise ValueError("Isotropic barostat requires a fully periodic cell.")
    if isinstance(dynamics.integrator, (MTKNPTPlan, MTKNPHPlan)):
        raise ValueError(
            "The integrator already owns the cell; a thermodynamic run has exactly "
            "one cell-evolution owner."
        )
    dynamics.require_conservative_force_definition("A Monte Carlo volume move")
    if not dynamics.integrator.canonical:
        raise ValueError(
            "A Monte Carlo NPT move requires a canonical NVT particle method."
        )
    if dynamics.constraints is None:
        return cell, jnp.asarray(True)
    if thermodynamic.volume_coordinate_convention == "atomic":
        raise ValueError(
            "Atomic volume scaling would deform holonomic constraints; declare "
            "molecular-center scaling."
        )
    indices = dynamics.system.topology.constraint_indices
    if indices.shape[0] == 0:
        return cell, jnp.asarray(True)
    molecules = dynamics.system.plan.molecule_ids
    return cell, jnp.all(molecules[indices[:, 0]] == molecules[indices[:, 1]])


# Proposal, Jacobian/energy ledger, acceptance draw and rollback form one move.
# Keep their source-faithful arithmetic and key/counter order explicit rather than
# fragmenting the numerical transaction solely because its result record is long.
def apply_isotropic_monte_carlo_barostat(
    dynamics: PreparedAtomisticDynamics,
    state: AtomisticDynamicsState,
    thermodynamic: PreparedThermodynamicStateTable,
    plan: IsotropicMonteCarloBarostatPlan,
    /,
) -> AtomisticBarostatEvaluation:
    """Propose and atomically accept or reject the state's next volume move."""
    if not isinstance(dynamics, PreparedAtomisticDynamics):
        raise TypeError("dynamics must be PreparedAtomisticDynamics.")
    if not isinstance(state, AtomisticDynamicsState):
        raise TypeError("state must be AtomisticDynamicsState.")
    if not isinstance(thermodynamic, PreparedThermodynamicStateTable):
        raise TypeError("thermodynamic must be PreparedThermodynamicStateTable.")
    if not isinstance(plan, IsotropicMonteCarloBarostatPlan):
        raise TypeError("plan must be IsotropicMonteCarloBarostatPlan.")
    dynamics._require_state(state, thermodynamic)
    cell, constraints_admitted = _admit(dynamics, thermodynamic)
    bound = dynamics.binding_valid(state, thermodynamic)
    row = thermodynamic.state_at_replica(state.thermodynamic_state_index)
    force_valid = dynamics.force_cache_valid(state, row.controls)
    controllers = state.controllers
    counter = controllers.move_counter
    key = jr.wrap_key_data(state.random_key)
    key = jr.fold_in(key, jnp.asarray(plan.realization_id, dtype=jnp.uint32))
    key = jr.fold_in(key, jnp.uint32(MONTE_CARLO_VOLUME_OPERATOR_ID))
    key = jr.fold_in(key, counter)
    dtype = state.kinematics.positions.dtype
    log_volume_change = jr.uniform(
        jr.fold_in(key, jnp.uint32(0)),
        (),
        minval=-plan.maximum_log_volume_change,
        maxval=plan.maximum_log_volume_change,
        dtype=dtype,
    )
    log_uniform = jnp.log(jr.uniform(jr.fold_in(key, jnp.uint32(1)), (), dtype=dtype))
    linear_scale = jnp.exp(log_volume_change / 3.0)
    old_vectors = state.cell_vectors
    new_vectors = old_vectors * linear_scale
    old_unwrapped = dynamics._unwrapped(state.kinematics, old_vectors)
    active = dynamics.system.active_mask
    if thermodynamic.volume_coordinate_convention == "atomic":
        fractional = cell.fractional_with_vectors(old_unwrapped, old_vectors)
        scaled = cell.cartesian_with_vectors(fractional, new_vectors)
        proposed_unwrapped = jnp.where(active[:, None], scaled, old_unwrapped)
    else:
        # ``molecule_labels`` holds the sorted active labels in its first
        # ``molecule_count`` rows and int32-max padding after; padded rows have
        # no mass and are masked so they never produce 0/0 centers.
        labels = dynamics.system.molecule_labels
        rows = labels.shape[0]
        label_valid = jnp.arange(rows, dtype=jnp.int32) < dynamics.system.molecule_count
        molecule_indices = jnp.searchsorted(labels, dynamics.system.plan.molecule_ids)
        masses = dynamics.system.plan.masses.astype(dtype)
        molecule_mass = (
            jnp.zeros((rows,), dtype=dtype)
            .at[molecule_indices]
            .add(jnp.where(active, masses, 0.0))
        )
        weighted = (
            jnp.zeros((rows, 3), dtype=dtype)
            .at[molecule_indices]
            .add(jnp.where(active[:, None], masses[:, None] * old_unwrapped, 0.0))
        )
        occupied = label_valid & (molecule_mass > 0.0)
        safe_mass = jnp.where(occupied, molecule_mass, 1.0)
        centers = jnp.where(occupied[:, None], weighted / safe_mass[:, None], 0.0)
        proposed_centers = cell.cartesian_with_vectors(
            cell.fractional_with_vectors(centers, old_vectors), new_vectors
        )
        displacement = jnp.where(
            occupied[molecule_indices][:, None],
            proposed_centers[molecule_indices] - centers[molecule_indices],
            0.0,
        )
        proposed_unwrapped = jnp.where(
            active[:, None], old_unwrapped + displacement, old_unwrapped
        )
    proposed_positions, proposed_images = cell.wrap_with_vectors(
        proposed_unwrapped, new_vectors
    )
    neighborhoods = dynamics.refresh_neighborhoods(
        state.neighborhoods,
        proposed_positions,
        cell_vectors=new_vectors,
        image_counts=proposed_images,
    )
    evaluation = dynamics.evaluate_configuration(
        proposed_positions,
        neighborhoods,
        unwrapped_positions=proposed_unwrapped,
        image_counts=proposed_images,
        species=state.species,
        cell_vectors=new_vectors,
        controls=row.controls,
        term_states=state.term_states,
    )
    volume_before = cell_volume(old_vectors)
    volume_after = cell_volume(new_vectors)
    energy_change = evaluation.energy - state.force.potential_energy
    pressure_work = row.pressure * (volume_after - volume_before)
    log_acceptance = -row.beta * (energy_change + pressure_work) + jnp.asarray(
        thermodynamic.scaled_entity_count + 1, dtype=dtype
    ) * jnp.log(volume_after / volume_before)
    revision = state.configuration_revision + 1
    force = dynamics.force_state(
        evaluation,
        neighborhoods,
        revision,
        row.hamiltonian_index,
        row.controls,
        state.random_key,
        term_states=state.term_states,
        ports=None,
    )
    accepted_controllers = eqx.tree_at(
        lambda value: (
            value.move_counter,
            value.accepted_moves,
            value.cell_epoch,
        ),
        controllers,
        (counter + 1, controllers.accepted_moves + 1, controllers.cell_epoch + 1),
    )
    kinetic = dynamics.kinetic_energy(state.kinematics.momenta)
    terms = dynamics.extended_terms(accepted_controllers, new_vectors, thermodynamic, row)
    ledger = advance_atomistic_energy_ledger(
        state.energy,
        kinetic,
        evaluation.energy,
        terms,
        thermostat_heat=jnp.zeros_like(kinetic),
        reservoir_heat=energy_change,
        constraint_work=jnp.zeros_like(kinetic),
        barostat_work=pressure_work,
        external_work=jnp.zeros_like(kinetic),
        balanced=False,
        accepted_increment=0,
    )
    thermodynamic_valid = (
        row.valid & row.temperature_mask & row.pressure_mask & (row.ensemble_code == 2)
    )
    resources = _neighborhood_reasons(neighborhoods)
    reasons = (
        resources
        | _flag(~evaluation.successful, AtomisticStepRejectionReason.POTENTIAL)
        | _flag(~thermodynamic_valid, AtomisticStepRejectionReason.THERMODYNAMIC_STATE)
        | _flag(~force_valid, AtomisticStepRejectionReason.STALE_FORCE)
        | _flag(~bound, AtomisticStepRejectionReason.BINDING)
        | _flag(~constraints_admitted, AtomisticStepRejectionReason.BAROSTAT)
        | _flag(
            ~(
                jnp.isfinite(log_acceptance)
                & dynamics.force_cache_finite(force)
                & tree_allfinite((ledger, new_vectors))
            ),
            AtomisticStepRejectionReason.NONFINITE,
        )
        | _flag(
            counter == jnp.iinfo(jnp.uint32).max, AtomisticStepRejectionReason.BAROSTAT
        )
        | _strain_reasons(dynamics.strain_response, force.strain_available, resources)
    )
    successful = reasons == 0
    accepted = successful & (log_uniform < jnp.minimum(log_acceptance, 0.0))
    candidate = eqx.tree_at(
        lambda value: (
            value.configuration_revision,
            value.kinematics,
            value.cell_vectors,
            value.neighborhoods,
            value.force,
            value.controllers,
            value.energy,
        ),
        state,
        (
            revision,
            AtomisticKinematics(
                proposed_positions, state.kinematics.momenta, proposed_images
            ),
            new_vectors,
            neighborhoods,
            force,
            accepted_controllers,
            ledger,
        ),
    )
    rejected = eqx.tree_at(
        lambda value: value.controllers.move_counter, state, counter + 1
    )
    successor = tree_where(accepted, candidate, tree_where(successful, rejected, state))
    return AtomisticBarostatEvaluation(
        candidate,
        successor,
        accepted,
        log_acceptance,
        volume_before,
        volume_after,
        state.force.potential_energy,
        evaluation.energy,
        successful,
        reasons,
        plan.plan_identity,
    )


__all__ = [
    "AtomisticBarostatEvaluation",
    "IsotropicMonteCarloBarostatPlan",
    "MONTE_CARLO_VOLUME_OPERATOR_ID",
    "apply_isotropic_monte_carlo_barostat",
]
