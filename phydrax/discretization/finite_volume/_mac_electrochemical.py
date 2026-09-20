#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._admissibility import AdmissibilityHeader, AdmissibilityReason
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cochain_electrochemical import scharfetter_gummel_flux
from ._incompressible import FaceVelocity, PreparedMACOperators


class MACElectrochemicalReason(IntFlag):
    NEGATIVE_CONCENTRATION = 1 << 8
    CONTENT_CONSERVATION_FAILED = 1 << 9
    NONZERO_BOUNDARY_ADVECTION = 1 << 10
    FREE_ENERGY_DISSIPATION_FAILED = 1 << 11


def mac_cell_to_faces(
    operators: PreparedMACOperators,
    values: ArrayLike,
    axis: int,
    /,
) -> tuple[Array, Array, Array]:
    """Return oriented tail, head, and center distance on one MAC face family."""

    value = jnp.asarray(values)
    axis_ = int(axis)
    structured_axis = operators.discretization.grid.structured_axes[axis_]
    moved = jnp.moveaxis(value, axis_, 0)
    centers = structured_axis.interval_centers.astype(value.dtype)
    if structured_axis.periodic:
        period = structured_axis.bounds[1] - structured_axis.bounds[0]
        tail = jnp.roll(moved, 1, axis=0)
        head = moved
        previous_centers = jnp.roll(centers, 1).at[0].add(-period)
        distance = centers - previous_centers
    else:
        tail = jnp.concatenate((moved[:1], moved[:-1], moved[-1:]), axis=0)
        head = jnp.concatenate((moved[:1], moved[1:], moved[-1:]), axis=0)
        if centers.size == 1:
            width = structured_axis.interval_widths.astype(value.dtype)
            distance = jnp.asarray((0.5 * width[0], 0.5 * width[0]))
        else:
            interior = centers[1:] - centers[:-1]
            distance = jnp.concatenate(
                (
                    0.5 * structured_axis.interval_widths[:1],
                    interior,
                    0.5 * structured_axis.interval_widths[-1:],
                )
            ).astype(value.dtype)
    shape = (distance.size,) + (1,) * (moved.ndim - 1)
    distance_values = jnp.broadcast_to(distance.reshape(shape), tail.shape)
    return (
        jnp.moveaxis(tail, 0, axis_),
        jnp.moveaxis(head, 0, axis_),
        jnp.moveaxis(distance_values, 0, axis_),
    )


class MACElectrochemicalFluxEvaluation(StrictModule):
    electrochemical_face_flux: tuple[Array, ...]
    advective_face_flux: tuple[Array, ...]
    total_face_flux: tuple[Array, ...]
    concentration_rate: Array
    species_content_defect: Array
    free_energy_dissipation: Array
    explicit_step_restriction: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class PreparedMACElectrochemicalFlux(StrictModule, NonTrainableState):
    """Cell-centered SG transport and conservative MAC advection."""

    operators: PreparedMACOperators
    diffusivities: Array
    species_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        diffusivities: ArrayLike,
        /,
    ) -> None:
        values = np.asarray(diffusivities, dtype=np.float64)
        if (
            not isinstance(operators, PreparedMACOperators)
            or values.ndim != 1
            or values.size == 0
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
        ):
            raise ValueError(
                "MAC electrochemical operators or diffusivities are invalid."
            )
        self.operators = operators
        self.diffusivities = jnp.asarray(values)
        self.species_count = values.size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-electrochemical-flux",
                "operators": operators.prepared_id,
                "diffusivities": array_tree_fingerprint(values),
                "boundary": "periodic-or-zero-normal-flux",
            }
        )

    def evaluate(
        self,
        concentrations: ArrayLike,
        dimensionless_drift_potential: ArrayLike,
        /,
        *,
        face_velocity: FaceVelocity | None = None,
        dimensionless_electrochemical_potential: ArrayLike | None = None,
    ) -> MACElectrochemicalFluxEvaluation:
        concentration = jnp.asarray(concentrations)
        drift = jnp.asarray(dimensionless_drift_potential, dtype=concentration.dtype)
        cell_shape = self.operators.discretization.cell_shape
        expected = cell_shape + (self.species_count,)
        if concentration.shape != expected or drift.shape != expected:
            raise ValueError("MAC concentrations and drift must have cell/species shape.")
        electrochemical = (
            jnp.log(jnp.where(concentration > 0.0, concentration, 1.0)) + drift
            if dimensionless_electrochemical_potential is None
            else jnp.asarray(
                dimensionless_electrochemical_potential, dtype=concentration.dtype
            )
        )
        if electrochemical.shape != expected:
            raise ValueError("MAC electrochemical potential has incompatible shape.")
        velocity = (
            tuple(
                jnp.zeros(layout.shape, dtype=concentration.dtype)
                for layout in self.operators.discretization.face_layouts
            )
            if face_velocity is None
            else self.operators.validate_velocity(face_velocity)
        )
        electrochemical_flux = []
        advective_flux = []
        total_flux = []
        dissipation = jnp.asarray(0.0, dtype=concentration.dtype)
        boundary_velocity_ok = jnp.asarray(True)
        for axis in range(len(cell_shape)):
            tail, head, distance = mac_cell_to_faces(self.operators, concentration, axis)
            tail_drift, head_drift, _ = mac_cell_to_faces(self.operators, drift, axis)
            tail_electrochemical, head_electrochemical, _ = mac_cell_to_faces(
                self.operators, electrochemical, axis
            )
            sg = (
                scharfetter_gummel_flux(
                    tail,
                    head,
                    head_drift - tail_drift,
                    self.diffusivities,
                )
                / distance
            )
            structured_axis = self.operators.discretization.grid.structured_axes[axis]
            if not structured_axis.periodic:
                lower = [slice(None)] * sg.ndim
                upper = [slice(None)] * sg.ndim
                lower[axis] = 0
                upper[axis] = sg.shape[axis] - 1
                sg = sg.at[tuple(lower)].set(0.0)
                sg = sg.at[tuple(upper)].set(0.0)
                velocity_lower = [slice(None)] * velocity[axis].ndim
                velocity_upper = [slice(None)] * velocity[axis].ndim
                velocity_lower[axis] = 0
                velocity_upper[axis] = velocity[axis].shape[axis] - 1
                boundary_velocity_ok = (
                    boundary_velocity_ok
                    & jnp.all(velocity[axis][tuple(velocity_lower)] == 0.0)
                    & jnp.all(velocity[axis][tuple(velocity_upper)] == 0.0)
                )
            upwind = jnp.where(velocity[axis][..., None] >= 0.0, tail, head)
            advective = velocity[axis][..., None] * upwind
            total = sg + advective
            electrochemical_flux.append(sg)
            advective_flux.append(advective)
            total_flux.append(total)
            difference = head_electrochemical - tail_electrochemical
            area = self.operators.discretization.face_measures[axis].astype(
                concentration.dtype
            )
            dissipation = dissipation - jnp.sum(area[..., None] * sg * difference)
        rates = []
        for species in range(self.species_count):
            divergence = self.operators.divergence(
                tuple(value[..., species] for value in total_flux)
            )
            rates.append(-divergence)
        rate = jnp.stack(tuple(rates), axis=-1)
        volumes = self.operators.discretization.cell_volumes.astype(concentration.dtype)
        content_defect = jnp.sum(
            volumes[..., None] * rate, axis=tuple(range(len(cell_shape)))
        )
        scale = jnp.maximum(
            jnp.sum(
                volumes[..., None] * jnp.abs(rate), axis=tuple(range(len(cell_shape)))
            ),
            1.0,
        )
        tolerance = 512.0 * jnp.finfo(rate.dtype).eps * scale
        conservative = jnp.all(jnp.abs(content_defect) <= tolerance)
        consuming = rate < 0.0
        restriction = jnp.min(
            jnp.where(
                consuming,
                concentration / jnp.maximum(-rate, jnp.finfo(rate.dtype).tiny),
                jnp.inf,
            )
        )
        finite = (
            jnp.all(jnp.isfinite(rate))
            & jnp.all(jnp.isfinite(concentration))
            & jnp.isfinite(dissipation)
        )
        positive = jnp.all(concentration > 0.0)
        dissipative = dissipation >= -512.0 * jnp.finfo(rate.dtype).eps
        supported = finite & positive & conservative & boundary_velocity_ok & dissipative
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            positive,
            reasons,
            reasons
            | jnp.asarray(
                int(MACElectrochemicalReason.NEGATIVE_CONCENTRATION), jnp.uint32
            ),
        )
        reasons = jnp.where(
            conservative,
            reasons,
            reasons
            | jnp.asarray(
                int(MACElectrochemicalReason.CONTENT_CONSERVATION_FAILED), jnp.uint32
            ),
        )
        reasons = jnp.where(
            boundary_velocity_ok,
            reasons,
            reasons
            | jnp.asarray(
                int(MACElectrochemicalReason.NONZERO_BOUNDARY_ADVECTION), jnp.uint32
            ),
        )
        reasons = jnp.where(
            dissipative,
            reasons,
            reasons
            | jnp.asarray(
                int(MACElectrochemicalReason.FREE_ENERGY_DISSIPATION_FAILED), jnp.uint32
            ),
        )
        header = AdmissibilityHeader(
            jnp.where(supported, jnp.min(concentration), -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "mac-electrochemical-evidence", "plan": self.plan_id}
            ),
        )
        return MACElectrochemicalFluxEvaluation(
            tuple(electrochemical_flux),
            tuple(advective_flux),
            tuple(total_flux),
            rate,
            content_defect,
            dissipation,
            restriction,
            header,
            self.plan_id,
        )


__all__ = [
    "MACElectrochemicalFluxEvaluation",
    "MACElectrochemicalReason",
    "PreparedMACElectrochemicalFlux",
    "mac_cell_to_faces",
]
