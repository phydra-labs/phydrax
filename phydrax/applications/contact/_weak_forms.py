#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.fem import FiniteElementMortarPlan


class ContactMortarEvidence(StrictModule):
    """Reproduction, transpose-work, and action-reaction evidence for one mortar."""

    plus_action: Array
    minus_action: Array
    relative_trace: Array
    patch_work: Array
    nodal_work: Array
    plus_constant_error: Array
    minus_constant_error: Array
    virtual_work_defect: Array
    action_reaction_defect: Array
    finite: Array
    constant_reproduced: Array
    adjoint_consistent: Array
    conservative: Array
    mortar_id: str = eqx.field(static=True)


class ContactMortarSpace(StrictModule, NonTrainableState):
    """Two-sided contact trace interpolation with an exact dual transpose."""

    plus_interpolation: Array
    minus_interpolation: Array
    quadrature_weights: Array
    mortar_id: str = eqx.field(static=True)

    def __init__(
        self,
        plus_interpolation: ArrayLike,
        minus_interpolation: ArrayLike,
        quadrature_weights: ArrayLike,
        /,
        *,
        mortar_id: str,
    ):
        plus = np.asarray(plus_interpolation)
        minus = np.asarray(minus_interpolation)
        weights = np.asarray(quadrature_weights)
        identifier = str(mortar_id)
        if (
            plus.ndim != 2
            or minus.ndim != 2
            or plus.shape[0] == 0
            or minus.shape[0] != plus.shape[0]
            or weights.shape != (plus.shape[0],)
            or plus.shape[1] == 0
            or minus.shape[1] == 0
            or not identifier
        ):
            raise ValueError(
                "Contact mortar traces and quadrature layouts are inconsistent."
            )
        if (
            not np.issubdtype(plus.dtype, np.inexact)
            or not np.issubdtype(minus.dtype, np.inexact)
            or not np.issubdtype(weights.dtype, np.inexact)
            or np.any(~np.isfinite(plus))
            or np.any(~np.isfinite(minus))
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError(
                "Contact mortar operators must be finite with positive weights."
            )
        self.plus_interpolation = jnp.asarray(plus)
        self.minus_interpolation = jnp.asarray(minus)
        self.quadrature_weights = jnp.asarray(weights)
        self.mortar_id = canonical_fingerprint(
            {
                "kind": "contact-mortar-space",
                "name": identifier,
                "plus": array_tree_fingerprint(plus),
                "minus": array_tree_fingerprint(minus),
                "weights": array_tree_fingerprint(weights),
            }
        )

    @classmethod
    def from_finite_element_mortar(
        cls,
        mortar: FiniteElementMortarPlan,
        /,
    ) -> ContactMortarSpace:
        """Bind the contact dual action to a qualified FE mortar plan."""
        if not isinstance(mortar, FiniteElementMortarPlan):
            raise TypeError("mortar must be FiniteElementMortarPlan.")
        return cls(
            mortar.left_interpolation,
            mortar.right_interpolation,
            mortar.physical_weights,
            mortar_id=mortar.plan_id,
        )

    def evaluate(
        self,
        plus_values: ArrayLike,
        minus_values: ArrayLike,
        traction: ArrayLike,
        /,
    ) -> ContactMortarEvidence:
        plus = jnp.asarray(plus_values)
        minus = jnp.asarray(minus_values)
        traction_ = jnp.asarray(traction)
        quadrature_count = self.quadrature_weights.shape[0]
        if (
            plus.ndim != 2
            or minus.ndim != 2
            or plus.shape[0] != self.plus_interpolation.shape[1]
            or minus.shape[0] != self.minus_interpolation.shape[1]
            or plus.shape[1] != minus.shape[1]
            or traction_.shape != (quadrature_count, plus.shape[1])
        ):
            raise ValueError(
                "Contact mortar values, traction, and trace spaces disagree."
            )
        plus_trace = self.plus_interpolation @ plus
        minus_trace = self.minus_interpolation @ minus
        relative_trace = plus_trace - minus_trace
        weighted_traction = self.quadrature_weights[:, None] * traction_
        plus_action = self.plus_interpolation.T @ weighted_traction
        minus_action = -(self.minus_interpolation.T @ weighted_traction)
        patch_work = jnp.sum(weighted_traction * relative_trace)
        nodal_work = jnp.sum(plus_action * plus) + jnp.sum(minus_action * minus)
        plus_constant_error = jnp.max(
            jnp.abs(jnp.sum(self.plus_interpolation, axis=1) - 1.0)
        )
        minus_constant_error = jnp.max(
            jnp.abs(jnp.sum(self.minus_interpolation, axis=1) - 1.0)
        )
        virtual_work_defect = nodal_work - patch_work
        action_reaction_defect = jnp.sum(plus_action, axis=0) + jnp.sum(
            minus_action, axis=0
        )
        scale = jnp.maximum(
            1.0,
            jnp.linalg.norm(plus_action)
            + jnp.linalg.norm(minus_action)
            + jnp.abs(patch_work),
        )
        tolerance = jnp.finfo(traction_.dtype).eps * max(64, 8 * quadrature_count) * scale
        finite = (
            jnp.all(jnp.isfinite(plus_action))
            & jnp.all(jnp.isfinite(minus_action))
            & jnp.all(jnp.isfinite(relative_trace))
            & jnp.isfinite(patch_work)
            & jnp.isfinite(nodal_work)
        )
        constant_reproduced = (plus_constant_error <= tolerance) & (
            minus_constant_error <= tolerance
        )
        adjoint_consistent = jnp.abs(virtual_work_defect) <= tolerance
        conservative = jnp.linalg.norm(action_reaction_defect) <= tolerance
        return ContactMortarEvidence(
            plus_action=plus_action,
            minus_action=minus_action,
            relative_trace=relative_trace,
            patch_work=patch_work,
            nodal_work=nodal_work,
            plus_constant_error=plus_constant_error,
            minus_constant_error=minus_constant_error,
            virtual_work_defect=virtual_work_defect,
            action_reaction_defect=action_reaction_defect,
            finite=finite,
            constant_reproduced=constant_reproduced,
            adjoint_consistent=adjoint_consistent,
            conservative=conservative,
            mortar_id=self.mortar_id,
        )


class NitscheContactEvidence(StrictModule):
    """Projected contact, adjoint-symmetry, and stabilization evidence."""

    projected_pressure: Array
    active: Array
    complementarity_residual: Array
    symmetry_defect: Array
    stabilization_energy: Array
    coercivity_margin: Array
    finite: Array
    adjoint_consistent: Array
    coercive: Array
    policy_id: str = eqx.field(static=True)


class NitscheContactPolicy(StrictModule, NonTrainableState):
    """Frictionless projected Nitsche policy with explicit coercivity threshold."""

    stabilization: Array
    minimum_stabilization: Array
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        stabilization: ArrayLike,
        minimum_stabilization: ArrayLike,
        /,
    ):
        stabilization_ = jnp.asarray(stabilization)
        minimum = jnp.asarray(minimum_stabilization)
        if (
            stabilization_.shape != ()
            or minimum.shape != ()
            or not bool(jnp.isfinite(stabilization_))
            or not bool(jnp.isfinite(minimum))
            or bool(stabilization_ <= 0.0)
            or bool(minimum <= 0.0)
        ):
            raise ValueError(
                "Nitsche stabilization scales must be positive finite scalars."
            )
        self.stabilization = stabilization_
        self.minimum_stabilization = minimum
        self.policy_id = canonical_fingerprint(
            {
                "kind": "nitsche-contact-policy",
                "stabilization": float(stabilization_),
                "minimum": float(minimum),
            }
        )

    def evidence(
        self,
        gap: ArrayLike,
        consistent_normal_traction: ArrayLike,
        quadrature_weights: ArrayLike,
        primal_test_work: ArrayLike,
        test_primal_work: ArrayLike,
        /,
    ) -> NitscheContactEvidence:
        gap_ = jnp.asarray(gap)
        consistent = jnp.asarray(consistent_normal_traction)
        weights = jnp.asarray(quadrature_weights)
        primal_test = jnp.asarray(primal_test_work)
        test_primal = jnp.asarray(test_primal_work)
        if (
            consistent.shape != gap_.shape
            or weights.shape != gap_.shape
            or primal_test.shape != gap_.shape
            or test_primal.shape != gap_.shape
        ):
            raise ValueError(
                "Nitsche evidence arrays must share the contact quadrature layout."
            )
        projected_pressure = jnp.maximum(-(consistent + self.stabilization * gap_), 0.0)
        active = projected_pressure > 0.0
        complementarity = (
            jnp.sqrt(gap_ * gap_ + projected_pressure * projected_pressure)
            - gap_
            - projected_pressure
        )
        symmetry_defect = jnp.sum(weights * (primal_test - test_primal))
        penetration = jnp.maximum(-gap_, 0.0)
        stabilization_energy = (
            0.5 * self.stabilization * jnp.sum(weights * penetration * penetration)
        )
        coercivity_margin = self.stabilization - self.minimum_stabilization
        scale = jnp.maximum(
            1.0,
            jnp.sum(jnp.abs(weights * primal_test))
            + jnp.sum(jnp.abs(weights * test_primal)),
        )
        tolerance = jnp.finfo(gap_.dtype).eps * max(64, 8 * gap_.size) * scale
        finite = (
            jnp.all(jnp.isfinite(projected_pressure))
            & jnp.all(jnp.isfinite(complementarity))
            & jnp.isfinite(symmetry_defect)
            & jnp.isfinite(stabilization_energy)
        )
        return NitscheContactEvidence(
            projected_pressure=projected_pressure,
            active=active,
            complementarity_residual=complementarity,
            symmetry_defect=symmetry_defect,
            stabilization_energy=stabilization_energy,
            coercivity_margin=coercivity_margin,
            finite=finite,
            adjoint_consistent=jnp.abs(symmetry_defect) <= tolerance,
            coercive=coercivity_margin >= 0.0,
            policy_id=self.policy_id,
        )


__all__ = [
    "ContactMortarEvidence",
    "ContactMortarSpace",
    "NitscheContactEvidence",
    "NitscheContactPolicy",
]
