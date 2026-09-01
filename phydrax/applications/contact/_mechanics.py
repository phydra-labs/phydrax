#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._geometry import ContactQueryResult
from ._laws import (
    AbstractNormalContactLaw,
    CoulombContactLaw,
    FrictionlessPDASContactLaw,
)
from ._state import (
    AcceptedContactState,
    CONTACT_OPEN,
    CONTACT_STICK,
    ContactEpochTransaction,
    ContactEvaluation,
    ContactStateTransaction,
)


class FixedEpochContactOperator(StrictModule, NonTrainableState):
    """Contact mechanics evaluated on one frozen, nondifferentiated search epoch."""

    query: ContactQueryResult
    normal_law: AbstractNormalContactLaw
    friction_law: CoulombContactLaw | None
    active_set_tolerance: Array
    law_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        query: ContactQueryResult,
        normal_law: AbstractNormalContactLaw,
        /,
        *,
        friction_law: CoulombContactLaw | None = None,
        active_set_tolerance: ArrayLike = 1.0e-10,
    ):
        if not isinstance(query, ContactQueryResult):
            raise TypeError("FixedEpochContactOperator requires ContactQueryResult.")
        if not isinstance(normal_law, AbstractNormalContactLaw):
            raise TypeError("normal_law must implement AbstractNormalContactLaw.")
        if friction_law is not None and not isinstance(friction_law, CoulombContactLaw):
            raise TypeError("friction_law must be CoulombContactLaw or None.")
        tolerance = jnp.asarray(active_set_tolerance)
        if (
            tolerance.shape != ()
            or not bool(jnp.isfinite(tolerance))
            or bool(tolerance < 0.0)
        ):
            raise ValueError(
                "active_set_tolerance must be one finite nonnegative scalar."
            )
        law_id = canonical_fingerprint(
            {
                "kind": "contact-law-composition",
                "normal": normal_law.law_id,
                "friction": None if friction_law is None else friction_law.law_id,
            }
        )
        self.query = query
        self.normal_law = normal_law
        self.friction_law = friction_law
        self.active_set_tolerance = tolerance
        self.law_id = law_id
        self.operator_id = canonical_fingerprint(
            {
                "kind": "fixed-epoch-contact-operator",
                "query": query.query_id,
                "law": law_id,
                "active_set_tolerance": float(tolerance),
            }
        )

    def accepted_state(
        self,
        previous: AcceptedContactState | None = None,
        /,
    ) -> AcceptedContactState:
        """Create or transfer accepted history before entering a fixed-epoch solve."""
        patches = self.query.patches
        if previous is None:
            return AcceptedContactState.zeros(patches, law_id=self.law_id)
        if not isinstance(previous, AcceptedContactState):
            raise TypeError("previous must be AcceptedContactState or None.")
        if previous.law_id != self.law_id:
            raise ValueError(
                "Accepted contact state belongs to a different law composition."
            )
        if (
            previous.epoch == patches.epoch
            and previous.pair_ids == patches.pair_ids
            and previous.patch_set_id == patches.patch_set_id
        ):
            return previous
        if previous.epoch == patches.epoch:
            raise ValueError("Contact search topology cannot change within one epoch.")
        return previous.for_patches(patches)

    def _require_state(self, accepted: AcceptedContactState, /) -> None:
        if not isinstance(accepted, AcceptedContactState):
            raise TypeError("accepted must be AcceptedContactState.")
        patches = self.query.patches
        if (
            accepted.law_id != self.law_id
            or accepted.epoch != self.query.epoch
            or accepted.pair_ids != patches.pair_ids
            or accepted.patch_set_id != patches.patch_set_id
        ):
            raise ValueError(
                "Accepted state does not belong to this contact epoch and law."
            )

    def evaluate(
        self,
        accepted: AcceptedContactState,
        plus_coordinates: ArrayLike | None = None,
        minus_coordinates: ArrayLike | None = None,
        /,
        *,
        normal_pressure: ArrayLike | None = None,
    ) -> ContactEvaluation:
        """Evaluate trial forces and evidence without modifying accepted history."""
        self._require_state(accepted)
        configuration = self.query.configuration
        patches = self.query.patches
        plus = (
            configuration.plus.current_coordinates
            if plus_coordinates is None
            else jnp.asarray(plus_coordinates)
        )
        minus = (
            configuration.minus.current_coordinates
            if minus_coordinates is None
            else jnp.asarray(minus_coordinates)
        )
        gap, normals, closest = self.query.current_kinematics(plus, minus)
        normal = self.normal_law.evaluate(
            gap,
            normals,
            accepted.normal_pressure,
            normal_pressure=normal_pressure,
        )

        facet_nodes = configuration.minus.facets[patches.minus_facet_indices]
        plus_increment = (
            plus[patches.plus_node_indices]
            - configuration.plus.current_coordinates[patches.plus_node_indices]
        )
        minus_increment = minus - configuration.minus.current_coordinates
        minus_patch_increment = jnp.sum(
            patches.minus_shape_values[..., None] * minus_increment[facet_nodes], axis=1
        )
        relative_increment = plus_increment - minus_patch_increment

        if self.friction_law is None:
            tangential_traction = jnp.zeros_like(normal.traction)
            accumulated_slip = accepted.accumulated_slip
            mode = jnp.where(normal.active, CONTACT_STICK, CONTACT_OPEN).astype(jnp.int32)
            pair_dissipation = jnp.zeros_like(gap)
            transport_ambiguous = jnp.zeros_like(gap, dtype=bool)
            transport_defect = jnp.zeros_like(gap)
            friction_cone_violation = jnp.zeros_like(gap)
        else:
            friction = self.friction_law.evaluate(
                normal.pressure,
                normals,
                accepted.contact_normals,
                relative_increment,
                accepted.tangential_traction,
                accepted.accumulated_slip,
            )
            tangential_traction = friction.tangential_traction
            accumulated_slip = friction.accumulated_slip
            mode = friction.mode
            pair_dissipation = friction.dissipation
            transport_ambiguous = friction.transport_ambiguous
            transport_defect = friction.transport_defect
            friction_cone_violation = jnp.maximum(
                jnp.linalg.norm(tangential_traction, axis=-1)
                - self.friction_law.coefficient * normal.pressure,
                0.0,
            )

        traction = normal.traction + tangential_traction
        plus_patch_forces = patches.weights[:, None] * traction
        minus_patch_forces = -plus_patch_forces
        plus_nodal_forces = (
            jnp.zeros_like(plus).at[patches.plus_node_indices].add(plus_patch_forces)
        )
        minus_contribution = (
            patches.minus_shape_values[..., None] * minus_patch_forces[:, None, :]
        )
        minus_nodal_forces = (
            jnp.zeros_like(minus)
            .at[facet_nodes.reshape((-1,))]
            .add(minus_contribution.reshape((-1, patches.dimension)))
        )
        patch_action_reaction_defect = plus_patch_forces + minus_patch_forces
        action_reaction_defect = jnp.sum(plus_nodal_forces, axis=0) + jnp.sum(
            minus_nodal_forces, axis=0
        )
        primal_violation = jnp.maximum(-gap, 0.0)
        dual_violation = jnp.maximum(-normal.pressure, 0.0)
        maximum_penetration = jnp.max(primal_violation, initial=0.0)
        total_reaction = jnp.sum(plus_patch_forces, axis=0)
        active_set_ambiguous = (jnp.abs(gap) <= self.active_set_tolerance) & (
            normal.pressure <= self.active_set_tolerance
        )
        dissipation = jnp.sum(patches.weights * pair_dissipation)
        finite = (
            jnp.all(jnp.isfinite(gap))
            & jnp.all(jnp.isfinite(normals))
            & jnp.all(jnp.isfinite(closest))
            & jnp.all(jnp.isfinite(traction))
            & jnp.all(jnp.isfinite(normal.complementarity_residual))
            & jnp.all(jnp.isfinite(action_reaction_defect))
            & jnp.isfinite(dissipation)
        )
        return ContactEvaluation(
            query=self.query,
            gap=gap,
            normals=normals,
            closest_points=closest,
            normal_pressure=normal.pressure,
            trial_accumulated_slip=accumulated_slip,
            tangential_traction=tangential_traction,
            traction=traction,
            normal_tangent=normal.tangent,
            active=normal.active,
            mode=mode,
            relative_displacement_increment=relative_increment,
            transport_ambiguous=transport_ambiguous,
            transport_defect=transport_defect,
            complementarity_residual=normal.complementarity_residual,
            primal_violation=primal_violation,
            dual_violation=dual_violation,
            friction_cone_violation=friction_cone_violation,
            active_set_ambiguous=active_set_ambiguous,
            plus_patch_forces=plus_patch_forces,
            minus_patch_forces=minus_patch_forces,
            plus_nodal_forces=plus_nodal_forces,
            minus_nodal_forces=minus_nodal_forces,
            pair_dissipation=patches.weights * pair_dissipation,
            patch_action_reaction_defect=patch_action_reaction_defect,
            action_reaction_defect=action_reaction_defect,
            dissipation=dissipation,
            maximum_penetration=maximum_penetration,
            total_reaction=total_reaction,
            finite=finite,
            epoch=self.query.epoch,
            law_id=self.law_id,
            operator_id=self.operator_id,
        )

    def attempt(
        self,
        accepted: AcceptedContactState,
        plus_coordinates: ArrayLike | None = None,
        minus_coordinates: ArrayLike | None = None,
        /,
        *,
        normal_pressure: ArrayLike | None = None,
    ) -> ContactStateTransaction:
        """Build an explicit candidate transaction; acceptance remains caller-owned."""
        evaluation = self.evaluate(
            accepted,
            plus_coordinates,
            minus_coordinates,
            normal_pressure=normal_pressure,
        )
        trial = AcceptedContactState(
            accepted.pair_ids,
            evaluation.normal_pressure,
            evaluation.normals,
            evaluation.tangential_traction,
            evaluation.trial_accumulated_slip,
            evaluation.mode,
            epoch=accepted.epoch,
            law_id=accepted.law_id,
            patch_set_id=accepted.patch_set_id,
            state_version=accepted.state_version,
        )
        return ContactStateTransaction(accepted, trial, evaluation)

    def attempt_epoch(
        self,
        previous: AcceptedContactState,
        plus_coordinates: ArrayLike | None = None,
        minus_coordinates: ArrayLike | None = None,
        /,
        *,
        normal_pressure: ArrayLike | None = None,
    ) -> ContactEpochTransaction:
        """Transfer history and trial a new search epoch with exact rollback."""
        candidate_base = self.accepted_state(previous)
        candidate = self.attempt(
            candidate_base,
            plus_coordinates,
            minus_coordinates,
            normal_pressure=normal_pressure,
        )
        return ContactEpochTransaction(previous, candidate)

    @property
    def requires_pressure_unknown(self) -> bool:
        return isinstance(self.normal_law, FrictionlessPDASContactLaw)


__all__ = ["FixedEpochContactOperator"]
