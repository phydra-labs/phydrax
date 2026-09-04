#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._boundary_cascade import prepare_layer_boundary
from ._continuous import (
    continuous_boundary_at,
    PreparedContinuousFourierModalLayer,
)
from ._factorization import _dense_solve
from ._layer import recover_longitudinal_fields
from ._runtime import (
    FourierModalSolveResult,
    PreparedFourierModalLayer,
    PreparedFourierModalMaxwell,
)
from ._scattering import _port_bases, _port_power_data, HomogeneousPortModes


class FourierModalFieldResult(StrictModule):
    electric_harmonics: Array
    magnetic_harmonics: Array
    electric_field: Array
    magnetic_field: Array
    longitudinal_offset: Array
    boundary_solve_residual: Array
    local_constitutive_residual: Array
    continuous_segment_defect: Array
    continuous_segment_index: Array
    continuous_status: Array
    layer_id: str = eqx.field(static=True)


class DiffractionOrderFarField(StrictModule):
    wavevectors: Array
    directions: Array
    polar_angle: Array
    azimuthal_angle: Array
    power: Array
    propagating: Array
    side: str = eqx.field(static=True)


def _prepared_layer_location(
    prepared: PreparedFourierModalMaxwell,
    layer_index: int,
    /,
) -> tuple[int, PreparedFourierModalLayer | PreparedContinuousFourierModalLayer]:
    requested = int(layer_index)
    current = 0
    for element_index, element in enumerate(prepared.elements):
        if not isinstance(
            element,
            PreparedFourierModalLayer | PreparedContinuousFourierModalLayer,
        ):
            continue
        if current == requested:
            return element_index, element
        current += 1
    raise IndexError(f"Layer index {requested} is out of range.")


def fields_in_layer(
    prepared: PreparedFourierModalMaxwell,
    result: FourierModalSolveResult,
    layer_index: int,
    longitudinal_offset: ArrayLike,
    /,
    *,
    coordinates: ArrayLike | None = None,
) -> FourierModalFieldResult:
    if not result.boundary_electric_fields:
        raise ValueError("The prepared solve did not retain boundary fields.")
    element_index, layer = _prepared_layer_location(prepared, layer_index)
    offset = jnp.asarray(
        longitudinal_offset,
        dtype=jnp.result_type(layer.layer.thickness, jnp.float32),
    )
    if offset.ndim > 0:
        raise ValueError("longitudinal_offset must be scalar.")
    offset = eqx.error_if(
        offset,
        (offset < 0.0) | (offset > jnp.real(layer.layer.thickness)),
        "longitudinal_offset must lie within the selected layer.",
    )
    left_electric = result.boundary_electric_fields[element_index]
    left_magnetic = result.boundary_magnetic_fields[element_index]
    if isinstance(layer, PreparedContinuousFourierModalLayer):
        partial, operator, integration_defect, segment_index = continuous_boundary_at(
            layer,
            prepared.problem,
            offset,
            prepared.plan.policy.boundary,
        )
        continuous_status = layer.status
    else:
        operator = layer.operator
        partial = prepare_layer_boundary(operator, offset, prepared.plan.policy.boundary)
        integration_defect = jnp.asarray(0.0, dtype=offset.dtype)
        segment_index = jnp.asarray(-1, dtype=jnp.int32)
        continuous_status = jnp.asarray(-1, dtype=jnp.int32)
    magnetic_rhs = left_magnetic - partial.c @ left_electric
    magnetic = _dense_solve(partial.d, magnetic_rhs)
    electric = partial.a @ left_electric + partial.b @ magnetic
    residual_denominator = jnp.maximum(
        jnp.sqrt(jnp.sum(jnp.abs(magnetic_rhs) ** 2)),
        1.0,
    )
    boundary_solve_residual = (
        jnp.sqrt(jnp.sum(jnp.abs(partial.d @ magnetic - magnetic_rhs) ** 2))
        / residual_denominator
    )
    tangential = jnp.concatenate((electric, magnetic), axis=0)
    electric_z, magnetic_z = recover_longitudinal_fields(operator, tangential)
    count = prepared.problem.harmonics.harmonic_count
    electric_harmonics = jnp.stack(
        (electric[:count], electric[count:], electric_z),
        axis=1,
    )
    magnetic_harmonics = jnp.stack(
        (magnetic[:count], magnetic[count:], magnetic_z),
        axis=1,
    )
    lattice = prepared.problem.harmonics
    if coordinates is None:
        electric_field = lattice.synthesis(electric_harmonics)
        magnetic_field = lattice.synthesis(magnetic_harmonics)
    else:
        electric_field = lattice.evaluate(electric_harmonics, coordinates)
        magnetic_field = lattice.evaluate(magnetic_harmonics, coordinates)
    return FourierModalFieldResult(
        electric_harmonics,
        magnetic_harmonics,
        electric_field,
        magnetic_field,
        offset,
        boundary_solve_residual,
        operator.diagnostics.constitutive_residual,
        integration_defect,
        segment_index,
        continuous_status,
        layer_id=layer.layer.layer_id,
    )


def poynting_flux(
    electric_field: ArrayLike,
    magnetic_field: ArrayLike,
    /,
) -> Array:
    electric = jnp.asarray(electric_field)
    magnetic = jnp.asarray(magnetic_field, dtype=electric.dtype)
    if electric.shape != magnetic.shape or electric.shape[-2] != 3:
        raise ValueError(
            "Fields must have equal shape with the Cartesian component axis second-last."
        )
    return 0.5 * jnp.real(
        electric[..., 0, :] * jnp.conj(magnetic[..., 1, :])
        - electric[..., 1, :] * jnp.conj(magnetic[..., 0, :])
    )


def cell_integrated_poynting_flux(
    prepared: PreparedFourierModalMaxwell,
    field: FourierModalFieldResult,
    /,
) -> Array:
    flux = poynting_flux(field.electric_field, field.magnetic_field)
    physical_axes = tuple(range(prepared.problem.harmonics.periodic_dimension))
    return jnp.mean(flux, axis=physical_axes) * prepared.problem.harmonics.cell_measure


def diffraction_order_far_field(
    prepared: PreparedFourierModalMaxwell,
    result: FourierModalSolveResult,
    /,
    *,
    side: str = "right",
) -> DiffractionOrderFarField:
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")
    modes = prepared.right_modes if side == "right" else prepared.left_modes
    if not isinstance(modes, HomogeneousPortModes):
        raise TypeError(
            "diffraction_order_far_field is the discrete homogeneous-port API."
        )
    amplitudes = result.right_outgoing if side == "right" else result.left_outgoing
    transverse = prepared.problem.harmonics.in_plane_wavevectors(
        prepared.problem.bloch_wavevector
    )
    kz = (
        modes.longitudinal_wavevector
        if side == "right"
        else -modes.longitudinal_wavevector
    )
    wavevectors = jnp.concatenate((transverse, kz[:, None]), axis=-1)
    magnitude = jnp.sqrt(jnp.sum(jnp.abs(wavevectors) ** 2, axis=-1))
    safe_magnitude = jnp.where(magnitude > 0.0, magnitude, 1.0)
    directions = jnp.real(wavevectors / safe_magnitude[:, None])
    polar = jnp.arccos(jnp.clip(directions[:, 2], -1.0, 1.0))
    azimuth = jnp.arctan2(directions[:, 1], directions[:, 0])
    count = prepared.problem.harmonics.harmonic_count
    rhs_count = amplitudes.shape[1]
    power = (jnp.abs(modes.flux_weights)[:, None] * jnp.abs(amplitudes) ** 2).reshape(
        (count, 2, rhs_count)
    )
    power = jnp.where(modes.propagating.reshape((count, 2, 1)), power, 0.0)
    return DiffractionOrderFarField(
        wavevectors,
        directions,
        polar,
        azimuth,
        power,
        modes.propagating.reshape((count, 2)),
        side=side,
    )


class RectangularFiniteAperture(StrictModule, NonTrainableState):
    """Analytic rectangular top-hat aperture."""

    widths: Array
    aperture_id: str = eqx.field(static=True)

    def __init__(self, widths: ArrayLike, /, *, aperture_id: str | None = None):
        value = np.asarray(widths, dtype=float)
        if value.shape != (2,) or np.any(~np.isfinite(value)) or np.any(value <= 0.0):
            raise ValueError("Rectangular aperture widths must be positive shape (2,).")
        identifier = (
            canonical_fingerprint(
                {"kind": "rectangular-finite-aperture", "widths": value.tolist()}
            )
            if aperture_id is None
            else str(aperture_id)
        )
        self.widths = jnp.asarray(value)
        self.aperture_id = identifier


class SampledFiniteAperture(StrictModule, NonTrainableState):
    """Bounded direct-quadrature aperture; no generic NUFFT is implied."""

    points: Array
    weights: Array
    aperture_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        aperture_id: str | None = None,
    ):
        points_ = np.asarray(points, dtype=float)
        weights_ = np.asarray(weights, dtype=float)
        if (
            points_.ndim != 2
            or points_.shape[1] != 2
            or weights_.shape != (points_.shape[0],)
            or np.any(~np.isfinite(points_))
            or np.any(~np.isfinite(weights_))
            or np.any(weights_ < 0.0)
            or np.sum(weights_) <= 0.0
        ):
            raise ValueError("Sampled aperture points/weights are invalid.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "sampled-finite-aperture",
                    "point_count": points_.shape[0],
                    "measure": float(np.sum(weights_)),
                }
            )
            if aperture_id is None
            else str(aperture_id)
        )
        self.points = jnp.asarray(points_)
        self.weights = jnp.asarray(weights_)
        self.aperture_id = identifier


FiniteAperture: TypeAlias = RectangularFiniteAperture | SampledFiniteAperture
FiniteApertureNormalization: TypeAlias = Literal["none", "aperture-area"]


class FiniteApertureFarFieldPlan(StrictModule, NonTrainableState):
    directions: Array
    active: Array
    aperture: FiniteAperture
    query_capacity: int = eqx.field(static=True)
    normalization: FiniteApertureNormalization = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        directions: ArrayLike,
        aperture: FiniteAperture,
        query_capacity: int,
        normalization: FiniteApertureNormalization = "aperture-area",
        /,
    ):
        values = np.asarray(directions, dtype=float)
        capacity = int(query_capacity)
        if (
            values.ndim != 2
            or values.shape[1] != 3
            or values.shape[0] < 1
            or values.shape[0] > capacity
            or np.any(~np.isfinite(values))
        ):
            raise ValueError(
                "Far-field directions must fit query_capacity with shape (*,3)."
            )
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("Far-field directions must be nonzero.")
        if not isinstance(aperture, RectangularFiniteAperture | SampledFiniteAperture):
            raise TypeError("Unknown finite aperture plan.")
        if normalization not in ("none", "aperture-area"):
            raise ValueError("Unknown finite-aperture normalization.")
        padded = np.zeros((capacity, 3), dtype=float)
        padded[:, 2] = 1.0
        padded[: values.shape[0]] = values / norms[:, None]
        active = np.arange(capacity) < values.shape[0]
        self.directions = jnp.asarray(padded)
        self.active = jnp.asarray(active)
        self.aperture = aperture
        self.query_capacity = capacity
        self.normalization = normalization
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-aperture-far-field",
                "directions": padded.tolist(),
                "active": active.tolist(),
                "aperture": aperture.aperture_id,
                "normalization": normalization,
            }
        )


class FiniteApertureFarField(StrictModule):
    electric_amplitudes: Array
    magnetic_amplitudes: Array
    power_density: Array
    active: Array
    aperture_power: Array
    aperture_power_defect: Array
    finite: Array
    side: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _aperture_transform(aperture: FiniteAperture, mismatch: Array, /) -> Array:
    if isinstance(aperture, RectangularFiniteAperture):
        widths = aperture.widths.astype(mismatch.dtype)
        factors = jnp.sinc(mismatch * widths[None, None, :] / (2.0 * jnp.pi))
        return widths[0] * widths[1] * factors[..., 0] * factors[..., 1]
    phase = jnp.exp(-1j * contract("qhd,pd->qhp", mismatch, aperture.points))
    return contract("qhp,p->qh", phase, aperture.weights)


def finite_aperture_far_field(
    prepared: PreparedFourierModalMaxwell,
    result: FourierModalSolveResult,
    plan: FiniteApertureFarFieldPlan,
    /,
    *,
    side: str = "right",
) -> FiniteApertureFarField:
    """Evaluate continuous angles only for a declared finite aperture."""

    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")
    modes = prepared.right_modes if side == "right" else prepared.left_modes
    amplitudes = result.right_outgoing if side == "right" else result.left_outgoing
    _, _, electric_matrix, magnetic_matrix = _port_bases(modes, side)
    tangential_electric = contract("ij,jr->ir", electric_matrix, amplitudes)
    tangential_magnetic = contract("ij,jr->ir", magnetic_matrix, amplitudes)
    count = prepared.problem.harmonics.harmonic_count
    rhs_count = amplitudes.shape[1]
    zeros = jnp.zeros((count, rhs_count), dtype=tangential_electric.dtype)
    electric_coefficients = jnp.stack(
        (
            tangential_electric[:count],
            tangential_electric[count:],
            zeros,
        ),
        axis=1,
    )
    magnetic_coefficients = jnp.stack(
        (
            tangential_magnetic[:count],
            tangential_magnetic[count:],
            zeros,
        ),
        axis=1,
    )
    harmonic_wavevectors = prepared.problem.harmonics.in_plane_wavevectors(
        prepared.problem.bloch_wavevector
    )
    query_wavevectors = (
        jnp.abs(prepared.problem.angular_frequency) * plan.directions[:, :2]
    )
    mismatch = query_wavevectors[:, None, :] - harmonic_wavevectors[None, :, :]
    aperture_transform = _aperture_transform(plan.aperture, mismatch)
    electric = contract("qh,hcr->qcr", aperture_transform, electric_coefficients)
    magnetic = contract("qh,hcr->qcr", aperture_transform, magnetic_coefficients)
    directions = plan.directions.astype(electric.real.dtype)
    electric = (
        electric
        - directions[..., None] * contract("qc,qcr->qr", directions, electric)[:, None, :]
    )
    magnetic = (
        magnetic
        - directions[..., None] * contract("qc,qcr->qr", directions, magnetic)[:, None, :]
    )
    poynting = jnp.cross(
        jnp.moveaxis(electric, 1, -1),
        jnp.conj(jnp.moveaxis(magnetic, 1, -1)),
    )
    power = 0.5 * jnp.real(contract("qc,qrc->qr", directions, poynting))
    if isinstance(plan.aperture, RectangularFiniteAperture):
        aperture_measure = jnp.prod(plan.aperture.widths)
    else:
        aperture_measure = jnp.sum(plan.aperture.weights)
    if plan.normalization == "aperture-area":
        electric = electric / aperture_measure
        magnetic = magnetic / aperture_measure
        power = power / aperture_measure**2
    mask = plan.active[:, None]
    electric = jnp.where(mask[:, :, None], electric, 0.0)
    magnetic = jnp.where(mask[:, :, None], magnetic, 0.0)
    power = jnp.where(mask, power, 0.0)
    _, outgoing_weights, _, outgoing_propagating, _, _ = _port_power_data(modes)
    aperture_power = (
        aperture_measure
        / prepared.problem.harmonics.cell_measure
        * jnp.sum(
            jnp.where(
                outgoing_propagating[:, None],
                outgoing_weights[:, None] * jnp.abs(amplitudes) ** 2,
                0.0,
            ),
            axis=0,
        )
    )
    angular_power = jnp.sum(power, axis=0) / jnp.maximum(jnp.sum(plan.active), 1)
    defect = jnp.abs(angular_power - aperture_power) / jnp.maximum(
        jnp.abs(aperture_power), 1.0
    )
    finite = (
        jnp.all(jnp.isfinite(electric))
        & jnp.all(jnp.isfinite(magnetic))
        & jnp.all(jnp.isfinite(power))
    )
    return FiniteApertureFarField(
        electric,
        magnetic,
        power,
        plan.active,
        aperture_power,
        defect,
        finite,
        side,
        plan.plan_id,
    )


__all__ = [
    "FiniteApertureFarField",
    "FiniteApertureFarFieldPlan",
    "RectangularFiniteAperture",
    "SampledFiniteAperture",
    "DiffractionOrderFarField",
    "FourierModalFieldResult",
    "cell_integrated_poynting_flux",
    "finite_aperture_far_field",
    "diffraction_order_far_field",
    "fields_in_layer",
    "poynting_flux",
]
