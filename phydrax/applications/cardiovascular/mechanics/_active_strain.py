#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._contraction import ContractionCandidate


class ActiveStrainState(StrictModule):
    """Accepted multiplicative active distortion on a fixed quadrature layout."""

    fiber_shortening: Array
    active_deformation_gradient: Array

    def __init__(
        self,
        fiber_shortening: ArrayLike,
        active_deformation_gradient: ArrayLike,
        /,
    ):
        shortening = jnp.asarray(fiber_shortening)
        active = jnp.asarray(active_deformation_gradient, dtype=shortening.dtype)
        if active.shape != (*shortening.shape, 3, 3):
            raise ValueError("Active deformation gradient shape must be (..., 3, 3).")
        self.fiber_shortening = shortening
        self.active_deformation_gradient = active


class ActiveStrainEvidence(StrictModule):
    determinant_residual: Array
    frame_orthogonality_residual: Array
    minimum_active_stretch: Array
    reconstruction_residual: Array
    finite: Array
    successful: Array
    formulation_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True, default="active-mechanics-only")


class ActiveStrainCandidate(StrictModule):
    previous_state: ActiveStrainState
    candidate_state: ActiveStrainState
    elastic_deformation_gradient: Array
    evidence: ActiveStrainEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class ActiveStrainPlan(StrictModule, NonTrainableState):
    """Isochoric multiplicative active strain, separate from active stress."""

    maximum_fiber_shortening: float = eqx.field(static=True)
    minimum_fiber_stretch: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True, default="isochoric-fiber-active-strain")

    def __init__(
        self,
        maximum_fiber_shortening: float,
        /,
        *,
        minimum_fiber_stretch: float = 0.2,
    ):
        maximum = float(maximum_fiber_shortening)
        minimum = float(minimum_fiber_stretch)
        if (
            not isfinite(maximum)
            or not isfinite(minimum)
            or maximum <= 0.0
            or maximum >= 1.0
            or minimum <= 0.0
            or minimum > 1.0
            or 1.0 - maximum < minimum
        ):
            raise ValueError("Active-strain shortening/stretch bounds are inconsistent.")
        self.maximum_fiber_shortening = maximum
        self.minimum_fiber_stretch = minimum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiac-active-strain",
                "formulation": self.formulation_id,
                "maximum_fiber_shortening": maximum,
                "minimum_fiber_stretch": minimum,
            }
        )

    def prepare(
        self,
        reference_fiber: ArrayLike,
        reference_sheet: ArrayLike,
        /,
    ) -> PreparedActiveStrain:
        return PreparedActiveStrain(self, reference_fiber, reference_sheet)


class PreparedActiveStrain(StrictModule, NonTrainableState):
    plan: ActiveStrainPlan
    reference_fiber: Array
    reference_sheet: Array
    reference_normal: Array
    field_shape: tuple[int, ...] = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ActiveStrainPlan,
        reference_fiber: ArrayLike,
        reference_sheet: ArrayLike,
        /,
    ):
        if not isinstance(plan, ActiveStrainPlan):
            raise TypeError("Active strain preparation requires ActiveStrainPlan.")
        fiber = jnp.asarray(reference_fiber)
        sheet = jnp.asarray(reference_sheet, dtype=fiber.dtype)
        if fiber.ndim < 1 or fiber.shape[-1] != 3 or sheet.shape != fiber.shape:
            raise ValueError("Reference fiber and sheet fields must have shape (..., 3).")
        dtype = np.dtype(fiber.dtype)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("Active strain directions must use floating-point dtype.")
        fiber_norm = jnp.sqrt(jnp.sum(fiber * fiber, axis=-1, keepdims=True))
        fiber = fiber / fiber_norm
        sheet = sheet - jnp.sum(sheet * fiber, axis=-1, keepdims=True) * fiber
        sheet_norm = jnp.sqrt(jnp.sum(sheet * sheet, axis=-1, keepdims=True))
        sheet = sheet / sheet_norm
        normal = jnp.cross(fiber, sheet)
        valid = bool(
            np.asarray(
                jnp.all(jnp.isfinite(fiber))
                & jnp.all(jnp.isfinite(sheet))
                & jnp.all(jnp.isfinite(normal))
                & jnp.all(fiber_norm > 0.0)
                & jnp.all(sheet_norm > 0.0)
            )
        )
        if not valid:
            raise ValueError(
                "Reference fiber/sheet frame must be finite and non-degenerate."
            )
        self.plan = plan
        self.reference_fiber = fiber
        self.reference_sheet = sheet
        self.reference_normal = normal
        self.field_shape = fiber.shape[:-1]
        self.dtype = dtype
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiac-active-strain",
                "plan": plan.plan_id,
                "shape": list(self.field_shape),
                "dtype": dtype.str,
            }
        )

    def resting_state(self) -> ActiveStrainState:
        identity = jnp.broadcast_to(
            jnp.eye(3, dtype=self.dtype), (*self.field_shape, 3, 3)
        )
        return ActiveStrainState(jnp.zeros(self.field_shape, dtype=self.dtype), identity)

    def candidate(
        self,
        previous_state: ActiveStrainState,
        contraction: ContractionCandidate,
        deformation_gradient: ArrayLike,
        /,
    ) -> ActiveStrainCandidate:
        if not isinstance(previous_state, ActiveStrainState):
            raise TypeError("Active strain candidate requires ActiveStrainState.")
        if not isinstance(contraction, ContractionCandidate):
            raise TypeError("Active strain requires a ContractionCandidate.")
        if contraction.evidence.fidelity_id == "prescribed-tension":
            raise TypeError(
                "Active strain requires a dimensionless activation contraction route."
            )
        deformation = jnp.asarray(deformation_gradient, dtype=self.dtype)
        expected = (*self.field_shape, 3, 3)
        if (
            previous_state.fiber_shortening.shape != self.field_shape
            or contraction.candidate_state.activation.shape != self.field_shape
            or deformation.shape != expected
        ):
            raise ValueError("Active strain candidate violates the prepared field shape.")
        activation = jnp.clip(contraction.candidate_state.activation, 0.0, 1.0)
        shortening = self.plan.maximum_fiber_shortening * activation
        fiber_stretch = jnp.maximum(self.plan.minimum_fiber_stretch, 1.0 - shortening)
        transverse_stretch = jax.lax.rsqrt(fiber_stretch)
        ff = contract("...i,...j->...ij", self.reference_fiber, self.reference_fiber)
        ss = contract("...i,...j->...ij", self.reference_sheet, self.reference_sheet)
        nn = contract("...i,...j->...ij", self.reference_normal, self.reference_normal)
        active = fiber_stretch[..., None, None] * ff + transverse_stretch[
            ..., None, None
        ] * (ss + nn)
        inverse_active = (1.0 / fiber_stretch)[..., None, None] * ff + (
            1.0 / transverse_stretch
        )[..., None, None] * (ss + nn)
        elastic = contract("...ij,...jk->...ik", deformation, inverse_active)
        reconstructed = contract("...ij,...jk->...ik", elastic, active)
        determinant_active = fiber_stretch * transverse_stretch**2
        reconstruction = jnp.sqrt(jnp.sum((reconstructed - deformation) ** 2))
        frame_residual = jnp.maximum(
            jnp.max(
                jnp.abs(jnp.sum(self.reference_fiber * self.reference_sheet, axis=-1))
            ),
            jnp.maximum(
                jnp.max(
                    jnp.abs(
                        jnp.sum(self.reference_fiber * self.reference_normal, axis=-1)
                    )
                ),
                jnp.max(
                    jnp.abs(
                        jnp.sum(self.reference_sheet * self.reference_normal, axis=-1)
                    )
                ),
            ),
        )
        finite = (
            jnp.all(jnp.isfinite(active))
            & jnp.all(jnp.isfinite(elastic))
            & jnp.all(jnp.isfinite(reconstruction))
        )
        successful = contraction.successful & finite & jnp.all(fiber_stretch > 0.0)
        evidence = ActiveStrainEvidence(
            jnp.max(jnp.abs(determinant_active - 1.0)),
            frame_residual,
            jnp.min(fiber_stretch),
            reconstruction,
            finite,
            successful,
            self.plan.formulation_id,
        )
        return ActiveStrainCandidate(
            previous_state,
            ActiveStrainState(shortening, active),
            elastic,
            evidence,
            self.prepared_id,
        )

    def commit(self, candidate: ActiveStrainCandidate, /) -> ActiveStrainState:
        if not isinstance(candidate, ActiveStrainCandidate):
            raise TypeError("commit requires ActiveStrainCandidate.")
        if candidate.prepared_id != self.prepared_id:
            raise ValueError("Active strain candidate belongs to another prepared plan.")
        return jax.tree.map(
            lambda new, old: jnp.where(candidate.successful, new, old),
            candidate.candidate_state,
            candidate.previous_state,
        )


__all__ = [
    "ActiveStrainCandidate",
    "ActiveStrainEvidence",
    "ActiveStrainPlan",
    "ActiveStrainState",
    "PreparedActiveStrain",
]
