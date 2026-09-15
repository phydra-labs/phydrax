#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class BeamlineElementKind(IntEnum):
    DRIFT = 0
    QUADRUPOLE = 1
    STEERER = 2
    RF_GAP = 3
    CIRCULAR_APERTURE = 4
    SECTOR_BEND = 5


class AcceleratorConvention(StrictModule, NonTrainableState):
    coordinate_order: tuple[str, ...] = eqx.field(static=True)
    longitudinal_sign: str = eqx.field(static=True)
    momentum_normalization: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        longitudinal_sign: str = "positive-late",
        momentum_normalization: str = "px-over-p0,py-over-p0,delta-p-over-p0",
    ):
        sign = str(longitudinal_sign).strip()
        normalization = str(momentum_normalization).strip()
        if not sign or not normalization:
            raise ValueError("Accelerator coordinate conventions must be explicit.")
        order = ("x", "px/p0", "y", "py/p0", "zeta", "delta")
        self.coordinate_order = order
        self.longitudinal_sign = sign
        self.momentum_normalization = normalization
        self.convention_id = canonical_fingerprint(
            {
                "kind": "accelerator-canonical-convention",
                "coordinate_order": list(order),
                "longitudinal_sign": sign,
                "momentum_normalization": normalization,
            }
        )


class AcceleratorBunch(StrictModule, NonTrainableState):
    coordinates: Array
    weights: Array
    particle_ids: Array
    active: Array
    reference_rest_energy: Array
    reference_momentum: Array
    reference_charge: Array
    convention: AcceleratorConvention
    valid: Array
    bunch_id: str = eqx.field(static=True)
    capacity: int = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        weights: ArrayLike,
        particle_ids: ArrayLike,
        /,
        *,
        active: ArrayLike | None = None,
        reference_rest_energy: float,
        reference_momentum: float,
        reference_charge: float,
        convention: AcceleratorConvention | None = None,
        bunch_id: str,
    ):
        coordinates_ = jnp.asarray(coordinates)
        weights_ = jnp.asarray(weights, dtype=coordinates_.dtype)
        identifiers = jnp.asarray(particle_ids, dtype=jnp.int32)
        if (
            coordinates_.ndim != 2
            or coordinates_.shape[1] != 6
            or coordinates_.shape[0] < 1
        ):
            raise ValueError("Bunch coordinates must have shape (capacity, 6).")
        capacity = int(coordinates_.shape[0])
        if weights_.shape != (capacity,) or identifiers.shape != (capacity,):
            raise ValueError("Bunch weights and identities must align with capacity.")
        active_ = (
            jnp.ones((capacity,), dtype=bool)
            if active is None
            else jnp.asarray(active, dtype=bool)
        )
        if active_.shape != (capacity,):
            raise ValueError("active must align with bunch capacity.")
        rest = float(reference_rest_energy)
        momentum = float(reference_momentum)
        charge = float(reference_charge)
        if (
            not np.all(np.isfinite((rest, momentum, charge)))
            or rest <= 0.0
            or momentum <= 0.0
        ):
            raise ValueError(
                "Reference rest energy/momentum must be positive finite values."
            )
        identifier = str(bunch_id).strip()
        if not identifier:
            raise ValueError("bunch_id must be non-empty.")
        convention_ = AcceleratorConvention() if convention is None else convention
        if not isinstance(convention_, AcceleratorConvention):
            raise TypeError("convention must be AcceleratorConvention.")
        valid = (
            jnp.all(jnp.isfinite(coordinates_), axis=1)
            & jnp.isfinite(weights_)
            & (weights_ >= 0.0)
            & (identifiers >= 0)
            & (coordinates_[:, 5] > -1.0)
        )
        self.coordinates = coordinates_
        self.weights = jnp.where(active_, weights_, 0.0)
        self.particle_ids = identifiers
        self.active = active_
        self.reference_rest_energy = jnp.asarray(rest)
        self.reference_momentum = jnp.asarray(momentum)
        self.reference_charge = jnp.asarray(charge)
        self.convention = convention_
        self.valid = jnp.where(active_, valid, True)
        self.bunch_id = identifier
        self.capacity = capacity


class BeamlinePlan(StrictModule, NonTrainableState):
    kinds: Array
    lengths: Array
    strengths: Array
    secondary_strengths: Array
    active: Array
    element_ids: tuple[str, ...] = eqx.field(static=True)
    element_count: int = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kinds: ArrayLike,
        lengths: ArrayLike,
        strengths: ArrayLike,
        secondary_strengths: ArrayLike,
        /,
        *,
        element_ids,
        active: ArrayLike | None = None,
        convention: AcceleratorConvention | None = None,
    ):
        kinds_ = np.asarray(kinds)
        lengths_ = np.asarray(lengths, dtype=float)
        strengths_ = np.asarray(strengths, dtype=float)
        secondary = np.asarray(secondary_strengths, dtype=float)
        if (
            kinds_.ndim != 1
            or kinds_.size < 1
            or not np.issubdtype(kinds_.dtype, np.integer)
        ):
            raise ValueError("Beamline kinds must be a non-empty integer vector.")
        count = int(kinds_.size)
        if (
            lengths_.shape != (count,)
            or strengths_.shape != (count,)
            or secondary.shape != (count,)
        ):
            raise ValueError("Beamline numeric fields must align with element count.")
        if (
            np.any(kinds_ < 0)
            or np.any(kinds_ > int(BeamlineElementKind.SECTOR_BEND))
            or np.any(~np.isfinite(lengths_))
            or np.any(lengths_ < 0.0)
            or np.any(~np.isfinite(strengths_))
            or np.any(~np.isfinite(secondary))
        ):
            raise ValueError("Beamline elements contain invalid kind or numeric values.")
        active_ = (
            np.ones((count,), dtype=bool)
            if active is None
            else np.asarray(active, dtype=bool)
        )
        ids = tuple(str(value).strip() for value in element_ids)
        if (
            active_.shape != (count,)
            or len(ids) != count
            or any(not value for value in ids)
            or len(set(ids)) != count
        ):
            raise ValueError(
                "Beamline activity and stable element identities must align."
            )
        convention_ = AcceleratorConvention() if convention is None else convention
        self.kinds = jnp.asarray(kinds_, dtype=jnp.int32)
        self.lengths = jnp.asarray(lengths_)
        self.strengths = jnp.asarray(strengths_)
        self.secondary_strengths = jnp.asarray(secondary)
        self.active = jnp.asarray(active_)
        self.element_ids = ids
        self.element_count = count
        self.convention_id = convention_.convention_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "accelerator-beamline-plan",
                "elements": array_tree_fingerprint(
                    (kinds_, lengths_, strengths_, secondary, active_)
                ),
                "element_ids": list(ids),
                "convention": convention_.convention_id,
            }
        )


class BeamlineResult(StrictModule, NonTrainableState):
    bunch: AcceleratorBunch
    coordinate_history: Array
    active_history: Array
    loss_element_indices: Array
    finite_steps: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


def track_beamline(plan: BeamlinePlan, bunch: AcceleratorBunch, /) -> BeamlineResult:
    """Track a fixed bunch through symplectic split reference elements."""
    if not isinstance(plan, BeamlinePlan) or not isinstance(bunch, AcceleratorBunch):
        raise TypeError("plan and bunch must use accelerator types.")
    if plan.convention_id != bunch.convention.convention_id:
        raise ValueError("Beamline and bunch coordinate conventions differ.")
    initial_active = bunch.active & bunch.valid
    initial_loss = jnp.full((bunch.capacity,), -1, dtype=jnp.int32)

    def one_element(carry, data):
        coordinates, alive, loss_index, element_index = carry
        kind, length, strength, secondary, enabled = data
        x, px, y, py, zeta, delta = (coordinates[:, index] for index in range(6))
        denominator = jnp.maximum(1.0 + delta, jnp.finfo(coordinates.dtype).tiny)
        half = 0.5 * length
        drifted_x = x + half * px / denominator
        drifted_y = y + half * py / denominator
        quad_px = px - strength * length * drifted_x
        quad_py = py + strength * length * drifted_y
        quad_x = drifted_x + half * quad_px / denominator
        quad_y = drifted_y + half * quad_py / denominator
        drift_x = x + length * px / denominator
        drift_y = y + length * py / denominator
        drift_zeta = zeta + 0.5 * length * (px * px + py * py) / (
            denominator * denominator
        )
        bend_px = px - strength * length
        candidate = coordinates
        candidate = candidate.at[:, 0].set(
            jnp.where(kind == int(BeamlineElementKind.QUADRUPOLE), quad_x, drift_x)
        )
        candidate = candidate.at[:, 2].set(
            jnp.where(kind == int(BeamlineElementKind.QUADRUPOLE), quad_y, drift_y)
        )
        candidate = candidate.at[:, 4].set(drift_zeta)
        candidate = candidate.at[:, 1].set(
            jnp.where(
                kind == int(BeamlineElementKind.QUADRUPOLE),
                quad_px,
                jnp.where(
                    kind == int(BeamlineElementKind.STEERER),
                    px + strength,
                    jnp.where(kind == int(BeamlineElementKind.SECTOR_BEND), bend_px, px),
                ),
            )
        )
        candidate = candidate.at[:, 3].set(
            jnp.where(
                kind == int(BeamlineElementKind.QUADRUPOLE),
                quad_py,
                jnp.where(kind == int(BeamlineElementKind.STEERER), py + secondary, py),
            )
        )
        candidate = candidate.at[:, 5].set(
            jnp.where(
                kind == int(BeamlineElementKind.RF_GAP),
                delta + strength * jnp.cos(secondary),
                delta,
            )
        )
        is_aperture = kind == int(BeamlineElementKind.CIRCULAR_APERTURE)
        inside = x * x + y * y <= strength * strength
        survives = (
            alive
            & (~is_aperture | inside)
            & jnp.all(jnp.isfinite(candidate), axis=1)
            & (candidate[:, 5] > -1.0)
        )
        effective = enabled & alive
        next_coordinates = jnp.where(effective[:, None], candidate, coordinates)
        next_alive = jnp.where(enabled, survives, alive)
        newly_lost = alive & ~next_alive & (loss_index < 0)
        next_loss = jnp.where(newly_lost, element_index, loss_index)
        finite = jnp.all(jnp.where(alive[:, None], jnp.isfinite(next_coordinates), True))
        return (next_coordinates, next_alive, next_loss, element_index + 1), (
            next_coordinates,
            next_alive,
            finite,
        )

    indices = jnp.arange(plan.element_count, dtype=jnp.int32)
    (_, final_active, loss_indices, _), history = jax.lax.scan(
        one_element,
        (
            bunch.coordinates,
            initial_active,
            initial_loss,
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (plan.kinds, plan.lengths, plan.strengths, plan.secondary_strengths, plan.active),
    )
    coordinate_history, active_history, finite_steps = history
    final_coordinates = coordinate_history[-1]
    result_bunch = AcceleratorBunch(
        final_coordinates,
        bunch.weights,
        bunch.particle_ids,
        active=final_active,
        reference_rest_energy=float(bunch.reference_rest_energy),
        reference_momentum=float(bunch.reference_momentum),
        reference_charge=float(bunch.reference_charge),
        convention=bunch.convention,
        bunch_id=f"{bunch.bunch_id}:{plan.plan_id}",
    )
    return BeamlineResult(
        result_bunch,
        coordinate_history,
        active_history,
        loss_indices,
        finite_steps,
        jnp.all(finite_steps),
        plan.plan_id,
    )


class BeamDiagnostics(StrictModule, NonTrainableState):
    mean: Array
    covariance: Array
    geometric_emittance_x: Array
    geometric_emittance_y: Array
    twiss_alpha_x: Array
    twiss_beta_x: Array
    twiss_alpha_y: Array
    twiss_beta_y: Array
    transmission: Array
    valid: Array
    bunch_id: str = eqx.field(static=True)


def beam_diagnostics(bunch: AcceleratorBunch, /) -> BeamDiagnostics:
    if not isinstance(bunch, AcceleratorBunch):
        raise TypeError("bunch must be AcceleratorBunch.")
    weights = jnp.where(bunch.active & bunch.valid, bunch.weights, 0.0)
    total = jnp.sum(weights)
    normalized = weights / jnp.maximum(total, jnp.finfo(weights.dtype).tiny)
    mean = ein.contract("n,ni->i", normalized, bunch.coordinates)
    centered = bunch.coordinates - mean
    covariance = ein.contract("n,ni,nj->ij", normalized, centered, centered)
    x_block = covariance[jnp.ix_(jnp.asarray([0, 1]), jnp.asarray([0, 1]))]
    y_block = covariance[jnp.ix_(jnp.asarray([2, 3]), jnp.asarray([2, 3]))]
    emit_x = jnp.sqrt(jnp.maximum(jnp.linalg.det(x_block), 0.0))
    emit_y = jnp.sqrt(jnp.maximum(jnp.linalg.det(y_block), 0.0))
    tiny = jnp.finfo(covariance.dtype).tiny
    valid = (
        (total > 0.0)
        & (emit_x > 0.0)
        & (emit_y > 0.0)
        & jnp.all(jnp.isfinite(covariance))
    )
    return BeamDiagnostics(
        mean,
        covariance,
        emit_x,
        emit_y,
        -x_block[0, 1] / jnp.maximum(emit_x, tiny),
        x_block[0, 0] / jnp.maximum(emit_x, tiny),
        -y_block[0, 1] / jnp.maximum(emit_y, tiny),
        y_block[0, 0] / jnp.maximum(emit_y, tiny),
        jnp.sum(bunch.active, dtype=bunch.coordinates.dtype) / bunch.capacity,
        valid,
        bunch.bunch_id,
    )


__all__ = [
    "AcceleratorBunch",
    "AcceleratorConvention",
    "BeamDiagnostics",
    "BeamlineElementKind",
    "BeamlinePlan",
    "BeamlineResult",
    "beam_diagnostics",
    "track_beamline",
]
