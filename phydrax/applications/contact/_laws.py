#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._route_state import ContactRouteMode


def _positive_scalar(value: ArrayLike, name: str) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != () or not bool(jnp.isfinite(scalar)) or bool(scalar <= 0.0):
        raise ValueError(f"{name} must be one positive finite scalar.")
    return scalar


def _contact_inputs(gap: ArrayLike, normal: ArrayLike) -> tuple[Array, Array]:
    gap_ = jnp.asarray(gap)
    normal_ = jnp.asarray(normal)
    if normal_.ndim == 0 or normal_.shape[:-1] != gap_.shape:
        raise ValueError("Contact normals must append one coordinate axis to gap.")
    norm = jnp.linalg.norm(normal_, axis=-1)
    return gap_, normal_ / norm[..., None]


def _fischer_burmeister(gap: Array, pressure: Array) -> Array:
    return jnp.sqrt(gap * gap + pressure * pressure) - gap - pressure


def _transport_tangent(
    value: Array,
    accepted_normal: Array,
    current_normal: Array,
    /,
) -> tuple[Array, Array, Array]:
    """Parallel-transport tangent vectors between accepted and current frames."""
    cosine = jnp.sum(accepted_normal * current_normal, axis=-1)
    ambiguous = cosine <= (-1.0 + 64.0 * jnp.finfo(value.dtype).eps)
    if value.shape[-1] == 2:
        sine = (
            accepted_normal[..., 0] * current_normal[..., 1]
            - accepted_normal[..., 1] * current_normal[..., 0]
        )
        transported = jnp.stack(
            (
                cosine * value[..., 0] - sine * value[..., 1],
                sine * value[..., 0] + cosine * value[..., 1],
            ),
            axis=-1,
        )
    else:
        axis = jnp.cross(accepted_normal, current_normal)
        denominator = jnp.where(ambiguous, 1.0, 1.0 + cosine)
        first = jnp.cross(axis, value)
        second = jnp.cross(axis, first)
        transported = value + first + second / denominator[..., None]
    projected = (
        value - jnp.sum(value * current_normal, axis=-1, keepdims=True) * current_normal
    )
    transported = jnp.where(ambiguous[..., None], projected, transported)
    tangency_defect = jnp.abs(jnp.sum(transported * current_normal, axis=-1))
    return transported, ambiguous, tangency_defect


class NormalContactResponse(StrictModule):
    pressure: Array
    traction: Array
    tangent: Array
    active: Array
    complementarity_residual: Array


class PenaltyConvergenceEvidence(StrictModule, NonTrainableState):
    normal_load: Array
    penetration_bound: Array
    maximum_penetration: Array
    satisfies_contract: Array


class AbstractNormalContactLaw(StrictModule, NonTrainableState):
    law_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def evaluate(
        self,
        gap: ArrayLike,
        normal: ArrayLike,
        accepted_pressure: ArrayLike | None = None,
        /,
        *,
        normal_pressure: ArrayLike | None = None,
    ) -> NormalContactResponse:
        raise NotImplementedError


class PenaltyContactLaw(AbstractNormalContactLaw):
    """Frictionless normal penalty law using the positive-open gap convention."""

    penalty: Array

    def __init__(self, penalty: ArrayLike, /):
        penalty_ = _positive_scalar(penalty, "Contact penalty")
        self.penalty = penalty_
        self.law_id = canonical_fingerprint(
            {"kind": "penalty-contact-law", "penalty": float(penalty_)}
        )

    def evaluate(
        self,
        gap: ArrayLike,
        normal: ArrayLike,
        accepted_pressure: ArrayLike | None = None,
        /,
        *,
        normal_pressure: ArrayLike | None = None,
    ) -> NormalContactResponse:
        gap_, normal_ = _contact_inputs(gap, normal)
        active = gap_ < 0.0
        pressure = self.penalty * jnp.maximum(-gap_, 0.0)
        return NormalContactResponse(
            pressure=pressure,
            traction=pressure[..., None] * normal_,
            tangent=self.penalty * active,
            active=active,
            complementarity_residual=_fischer_burmeister(gap_, pressure),
        )

    def penetration_bound(self, normal_load: ArrayLike, /) -> Array:
        return jnp.asarray(normal_load) / self.penalty

    def convergence_evidence(
        self,
        normal_load: ArrayLike,
        maximum_penetration: ArrayLike,
        /,
    ) -> PenaltyConvergenceEvidence:
        load = jnp.asarray(normal_load)
        measured = jnp.asarray(maximum_penetration)
        if measured.shape != load.shape:
            raise ValueError("Measured penetration must match the load layout.")
        bound = self.penetration_bound(load)
        valid = jnp.all(jnp.isfinite(load) & (load >= 0.0)) & jnp.all(
            jnp.isfinite(measured) & (measured >= 0.0)
        )
        return PenaltyConvergenceEvidence(
            normal_load=load,
            penetration_bound=bound,
            maximum_penetration=measured,
            satisfies_contract=valid & jnp.all(measured <= bound),
        )


class FrictionlessPDASContactLaw(AbstractNormalContactLaw):
    """Exact primal-dual active-set map for g >= 0, p >= 0, g p = 0."""

    active_set_scale: Array

    def __init__(self, active_set_scale: ArrayLike, /):
        scale = _positive_scalar(active_set_scale, "PDAS active-set scale")
        self.active_set_scale = scale
        self.law_id = canonical_fingerprint(
            {"kind": "frictionless-pdas-contact-law", "scale": float(scale)}
        )

    def evaluate(
        self,
        gap: ArrayLike,
        normal: ArrayLike,
        accepted_pressure: ArrayLike | None = None,
        /,
        *,
        normal_pressure: ArrayLike | None = None,
    ) -> NormalContactResponse:
        gap_, normal_ = _contact_inputs(gap, normal)
        if normal_pressure is None:
            raise ValueError(
                "PDAS evaluation requires the current normal-pressure unknown."
            )
        multiplier = jnp.asarray(normal_pressure)
        if multiplier.shape != gap_.shape:
            raise ValueError("PDAS normal pressure must match gap.")
        active = multiplier - self.active_set_scale * gap_ > 0.0
        pressure = jnp.where(active, jnp.maximum(multiplier, 0.0), 0.0)
        return NormalContactResponse(
            pressure=pressure,
            traction=pressure[..., None] * normal_,
            tangent=jnp.zeros_like(gap_),
            active=active,
            complementarity_residual=_fischer_burmeister(gap_, pressure),
        )


class AugmentedLagrangianContactLaw(AbstractNormalContactLaw):
    """Frictionless projected multiplier update, committed only by a transaction."""

    augmentation: Array

    def __init__(self, augmentation: ArrayLike, /):
        augmentation = _positive_scalar(augmentation, "Contact augmentation")
        self.augmentation = augmentation
        self.law_id = canonical_fingerprint(
            {
                "kind": "augmented-lagrangian-contact-law",
                "augmentation": float(augmentation),
            }
        )

    def evaluate(
        self,
        gap: ArrayLike,
        normal: ArrayLike,
        accepted_pressure: ArrayLike | None = None,
        /,
        *,
        normal_pressure: ArrayLike | None = None,
    ) -> NormalContactResponse:
        gap_, normal_ = _contact_inputs(gap, normal)
        accepted = (
            jnp.zeros_like(gap_)
            if accepted_pressure is None
            else jnp.asarray(accepted_pressure)
        )
        if accepted.shape != gap_.shape:
            raise ValueError("Accepted augmented multipliers must match gap.")
        pressure = jnp.maximum(accepted - self.augmentation * gap_, 0.0)
        active = pressure > 0.0
        return NormalContactResponse(
            pressure=pressure,
            traction=pressure[..., None] * normal_,
            tangent=self.augmentation * active,
            active=active,
            complementarity_residual=_fischer_burmeister(gap_, pressure),
        )


class CoulombContactResponse(StrictModule):
    tangential_traction: Array
    accumulated_slip: Array
    plastic_slip_increment: Array
    mode: Array
    dissipation: Array
    transport_ambiguous: Array
    transport_defect: Array


class CoulombContactLaw(StrictModule, NonTrainableState):
    """Sharp Coulomb return map with explicit open, stick, and slip history."""

    coefficient: Array
    tangential_penalty: Array
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficient: ArrayLike,
        tangential_penalty: ArrayLike,
        /,
    ):
        coefficient_ = jnp.asarray(coefficient)
        if (
            coefficient_.shape != ()
            or not bool(jnp.isfinite(coefficient_))
            or bool(coefficient_ < 0.0)
        ):
            raise ValueError("Coulomb coefficient must be one finite nonnegative scalar.")
        penalty = _positive_scalar(tangential_penalty, "Tangential contact penalty")
        self.coefficient = coefficient_
        self.tangential_penalty = penalty
        self.law_id = canonical_fingerprint(
            {
                "kind": "coulomb-contact-law",
                "coefficient": float(coefficient_),
                "tangential_penalty": float(penalty),
            }
        )

    def evaluate(
        self,
        normal_pressure: ArrayLike,
        normal: ArrayLike,
        accepted_normal: ArrayLike,
        relative_displacement_increment: ArrayLike,
        accepted_traction: ArrayLike,
        accepted_slip: ArrayLike,
        /,
    ) -> CoulombContactResponse:
        pressure = jnp.asarray(normal_pressure)
        normal_ = jnp.asarray(normal)
        old_normal = jnp.asarray(accepted_normal)
        increment = jnp.asarray(relative_displacement_increment)
        previous = jnp.asarray(accepted_traction)
        slip = jnp.asarray(accepted_slip)
        if (
            normal_.shape[:-1] != pressure.shape
            or old_normal.shape != normal_.shape
            or increment.shape != normal_.shape
            or previous.shape != normal_.shape
            or slip.shape != normal_.shape
        ):
            raise ValueError(
                "Coulomb state, pressure, normal, and increment layouts differ."
            )
        normal_ = normal_ / jnp.linalg.norm(normal_, axis=-1, keepdims=True)
        old_normal = old_normal / jnp.linalg.norm(old_normal, axis=-1, keepdims=True)
        previous_tangent, traction_ambiguous, traction_defect = _transport_tangent(
            previous, old_normal, normal_
        )
        transported_slip, slip_ambiguous, slip_defect = _transport_tangent(
            slip, old_normal, normal_
        )
        tangential_increment = (
            increment - jnp.sum(increment * normal_, axis=-1, keepdims=True) * normal_
        )
        trial = previous_tangent - self.tangential_penalty * tangential_increment
        trial_norm = jnp.linalg.norm(trial, axis=-1)
        limit = self.coefficient * pressure
        open_ = pressure <= 0.0
        stick = ~open_ & (trial_norm <= limit)
        safe_norm = jnp.where(trial_norm > 0.0, trial_norm, 1.0)
        projected = limit[..., None] * trial / safe_norm[..., None]
        traction = jnp.where(
            open_[..., None],
            0.0,
            jnp.where(stick[..., None], trial, projected),
        )
        slip_mode = ~open_ & ~stick
        plastic_increment = (
            tangential_increment + (traction - previous_tangent) / self.tangential_penalty
        )
        plastic_increment = jnp.where(slip_mode[..., None], plastic_increment, 0.0)
        accumulated = transported_slip + plastic_increment
        pair_dissipation = jnp.maximum(
            -jnp.sum(traction * plastic_increment, axis=-1), 0.0
        )
        mode = jnp.where(
            open_,
            int(ContactRouteMode.OPEN),
            jnp.where(
                stick,
                int(ContactRouteMode.STICK),
                int(ContactRouteMode.SLIP),
            ),
        ).astype(jnp.int32)
        return CoulombContactResponse(
            tangential_traction=traction,
            accumulated_slip=accumulated,
            plastic_slip_increment=plastic_increment,
            mode=mode,
            dissipation=pair_dissipation,
            transport_ambiguous=traction_ambiguous | slip_ambiguous,
            transport_defect=jnp.maximum(traction_defect, slip_defect),
        )


__all__ = [
    "AbstractNormalContactLaw",
    "AugmentedLagrangianContactLaw",
    "CoulombContactLaw",
    "CoulombContactResponse",
    "FrictionlessPDASContactLaw",
    "NormalContactResponse",
    "PenaltyContactLaw",
    "PenaltyConvergenceEvidence",
]
