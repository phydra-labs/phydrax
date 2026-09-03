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


class ActiveStressState(StrictModule):
    """Accepted active Cauchy stress and tension on a fixed quadrature layout."""

    tension: Array
    cauchy_stress: Array

    def __init__(self, tension: ArrayLike, cauchy_stress: ArrayLike, /):
        tension_ = jnp.asarray(tension)
        stress = jnp.asarray(cauchy_stress, dtype=tension_.dtype)
        if stress.shape != (*tension_.shape, 3, 3):
            raise ValueError("Active stress tensor shape must be tension_shape + (3, 3).")
        self.tension = tension_
        self.cauchy_stress = stress


class ActiveStressEvidence(StrictModule):
    fiber_norm_residual: Array
    sheet_norm_residual: Array
    orthogonality_residual: Array
    symmetry_residual: Array
    active_power: Array
    finite: Array
    successful: Array
    formulation_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True, default="active-mechanics-only")


class ActiveStressCandidate(StrictModule):
    previous_state: ActiveStressState
    candidate_state: ActiveStressState
    first_piola_stress: Array
    evidence: ActiveStressEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class ActiveStressPlan(StrictModule, NonTrainableState):
    """Fiber/sheet active Cauchy-stress formulation, separate from active strain."""

    sheet_tension_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True, default="fiber-sheet-active-stress")
    stress_unit: str = eqx.field(static=True, default="kPa")

    def __init__(self, /, *, sheet_tension_fraction: float = 0.0):
        fraction = float(sheet_tension_fraction)
        if not isfinite(fraction) or fraction < 0.0:
            raise ValueError("sheet_tension_fraction must be finite and non-negative.")
        self.sheet_tension_fraction = fraction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiac-active-stress",
                "formulation": self.formulation_id,
                "sheet_tension_fraction": fraction,
                "stress_unit": self.stress_unit,
            }
        )

    def prepare(
        self,
        reference_fiber: ArrayLike,
        reference_sheet: ArrayLike,
        /,
    ) -> PreparedActiveStress:
        return PreparedActiveStress(self, reference_fiber, reference_sheet)


class PreparedActiveStress(StrictModule, NonTrainableState):
    plan: ActiveStressPlan
    reference_fiber: Array
    reference_sheet: Array
    field_shape: tuple[int, ...] = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ActiveStressPlan,
        reference_fiber: ArrayLike,
        reference_sheet: ArrayLike,
        /,
    ):
        if not isinstance(plan, ActiveStressPlan):
            raise TypeError("Active stress preparation requires ActiveStressPlan.")
        fiber = jnp.asarray(reference_fiber)
        sheet = jnp.asarray(reference_sheet, dtype=fiber.dtype)
        if fiber.ndim < 1 or fiber.shape[-1] != 3 or sheet.shape != fiber.shape:
            raise ValueError("Reference fiber and sheet fields must have shape (..., 3).")
        dtype = np.dtype(fiber.dtype)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("Active stress directions must use floating-point dtype.")
        fiber_norm = jnp.sqrt(jnp.sum(fiber * fiber, axis=-1, keepdims=True))
        fiber = fiber / fiber_norm
        sheet_projection = jnp.sum(sheet * fiber, axis=-1, keepdims=True)
        sheet_orthogonal = sheet - sheet_projection * fiber
        sheet_norm = jnp.sqrt(
            jnp.sum(sheet_orthogonal * sheet_orthogonal, axis=-1, keepdims=True)
        )
        valid = bool(
            np.asarray(
                jnp.all(jnp.isfinite(fiber_norm))
                & jnp.all(jnp.isfinite(sheet_norm))
                & jnp.all(fiber_norm > 0.0)
                & jnp.all(sheet_norm > 0.0)
            )
        )
        if not valid:
            raise ValueError(
                "Reference fiber/sheet frame must be finite and non-degenerate."
            )
        sheet = sheet_orthogonal / sheet_norm
        self.plan = plan
        self.reference_fiber = fiber
        self.reference_sheet = sheet
        self.field_shape = fiber.shape[:-1]
        self.dtype = dtype
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiac-active-stress",
                "plan": plan.plan_id,
                "shape": list(self.field_shape),
                "dtype": dtype.str,
            }
        )

    def resting_state(self) -> ActiveStressState:
        tension = jnp.zeros(self.field_shape, dtype=self.dtype)
        stress = jnp.zeros((*self.field_shape, 3, 3), dtype=self.dtype)
        return ActiveStressState(tension, stress)

    def candidate(
        self,
        previous_state: ActiveStressState,
        contraction: ContractionCandidate,
        deformation_gradient: ArrayLike,
        /,
        *,
        velocity_gradient: ArrayLike | None = None,
    ) -> ActiveStressCandidate:
        if not isinstance(previous_state, ActiveStressState):
            raise TypeError("Active stress candidate requires ActiveStressState.")
        if not isinstance(contraction, ContractionCandidate):
            raise TypeError("Active stress requires a ContractionCandidate.")
        deformation = jnp.asarray(deformation_gradient, dtype=self.dtype)
        expected = (*self.field_shape, 3, 3)
        if (
            previous_state.tension.shape != self.field_shape
            or contraction.active_tension.shape != self.field_shape
            or deformation.shape != expected
        ):
            raise ValueError("Active stress candidate violates the prepared field shape.")
        deformed_fiber = contract("...ij,...j->...i", deformation, self.reference_fiber)
        deformed_sheet = contract("...ij,...j->...i", deformation, self.reference_sheet)
        fiber_norm = jnp.sqrt(jnp.sum(deformed_fiber * deformed_fiber, axis=-1))
        sheet_norm = jnp.sqrt(jnp.sum(deformed_sheet * deformed_sheet, axis=-1))
        tiny = jnp.finfo(deformation.dtype).tiny
        fiber = deformed_fiber / jnp.maximum(fiber_norm[..., None], tiny)
        sheet = deformed_sheet / jnp.maximum(sheet_norm[..., None], tiny)
        tension = contraction.active_tension
        fiber_stress = contract("...,...i,...j->...ij", tension, fiber, fiber)
        sheet_stress = contract(
            "...,...i,...j->...ij",
            self.plan.sheet_tension_fraction * tension,
            sheet,
            sheet,
        )
        cauchy = fiber_stress + sheet_stress
        determinant = (
            deformation[..., 0, 0]
            * (
                deformation[..., 1, 1] * deformation[..., 2, 2]
                - deformation[..., 1, 2] * deformation[..., 2, 1]
            )
            - deformation[..., 0, 1]
            * (
                deformation[..., 1, 0] * deformation[..., 2, 2]
                - deformation[..., 1, 2] * deformation[..., 2, 0]
            )
            + deformation[..., 0, 2]
            * (
                deformation[..., 1, 0] * deformation[..., 2, 1]
                - deformation[..., 1, 1] * deformation[..., 2, 0]
            )
        )
        inverse_transpose = jnp.stack(
            (
                jnp.stack(
                    (
                        deformation[..., 1, 1] * deformation[..., 2, 2]
                        - deformation[..., 1, 2] * deformation[..., 2, 1],
                        deformation[..., 1, 2] * deformation[..., 2, 0]
                        - deformation[..., 1, 0] * deformation[..., 2, 2],
                        deformation[..., 1, 0] * deformation[..., 2, 1]
                        - deformation[..., 1, 1] * deformation[..., 2, 0],
                    ),
                    axis=-1,
                ),
                jnp.stack(
                    (
                        deformation[..., 0, 2] * deformation[..., 2, 1]
                        - deformation[..., 0, 1] * deformation[..., 2, 2],
                        deformation[..., 0, 0] * deformation[..., 2, 2]
                        - deformation[..., 0, 2] * deformation[..., 2, 0],
                        deformation[..., 0, 1] * deformation[..., 2, 0]
                        - deformation[..., 0, 0] * deformation[..., 2, 1],
                    ),
                    axis=-1,
                ),
                jnp.stack(
                    (
                        deformation[..., 0, 1] * deformation[..., 1, 2]
                        - deformation[..., 0, 2] * deformation[..., 1, 1],
                        deformation[..., 0, 2] * deformation[..., 1, 0]
                        - deformation[..., 0, 0] * deformation[..., 1, 2],
                        deformation[..., 0, 0] * deformation[..., 1, 1]
                        - deformation[..., 0, 1] * deformation[..., 1, 0],
                    ),
                    axis=-1,
                ),
            ),
            axis=-2,
        ) / jnp.maximum(determinant[..., None, None], tiny)
        first_piola = determinant[..., None, None] * contract(
            "...ij,...jk->...ik", cauchy, inverse_transpose
        )
        rate = (
            jnp.zeros_like(cauchy)
            if velocity_gradient is None
            else jnp.asarray(velocity_gradient, dtype=self.dtype)
        )
        if rate.shape != expected:
            raise ValueError("Velocity gradient violates the prepared tensor shape.")
        power = jnp.sum(cauchy * rate, axis=(-2, -1))
        frame_dot = jnp.sum(fiber * sheet, axis=-1)
        symmetry = jnp.sqrt(jnp.sum((cauchy - jnp.swapaxes(cauchy, -1, -2)) ** 2))
        finite = (
            jnp.all(jnp.isfinite(cauchy))
            & jnp.all(jnp.isfinite(first_piola))
            & jnp.all(jnp.isfinite(power))
            & jnp.all(determinant > 0.0)
        )
        successful = contraction.successful & finite
        evidence = ActiveStressEvidence(
            jnp.max(jnp.abs(fiber_norm - 1.0)),
            jnp.max(jnp.abs(sheet_norm - 1.0)),
            jnp.max(jnp.abs(frame_dot)),
            symmetry,
            power,
            finite,
            successful,
            self.plan.formulation_id,
        )
        candidate_state = ActiveStressState(tension, cauchy)
        return ActiveStressCandidate(
            previous_state,
            candidate_state,
            first_piola,
            evidence,
            self.prepared_id,
        )

    def commit(self, candidate: ActiveStressCandidate, /) -> ActiveStressState:
        if not isinstance(candidate, ActiveStressCandidate):
            raise TypeError("commit requires ActiveStressCandidate.")
        if candidate.prepared_id != self.prepared_id:
            raise ValueError("Active stress candidate belongs to another prepared plan.")
        return jax.tree.map(
            lambda new, old: jnp.where(candidate.successful, new, old),
            candidate.candidate_state,
            candidate.previous_state,
        )


__all__ = [
    "ActiveStressCandidate",
    "ActiveStressEvidence",
    "ActiveStressPlan",
    "ActiveStressState",
    "PreparedActiveStress",
]
