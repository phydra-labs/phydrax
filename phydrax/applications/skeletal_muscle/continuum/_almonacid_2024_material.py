#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent material implementation of the pinned Almonacid executable.

The numerical identity is Flexodeal commit
0698e3d87d7261c81437d410dda161fe3b8efcbf, not a substitution of the different
publication parameter table. The source's muscle/aponeurosis laws, pressure
sign, branch endpoints, non-unit reference fibers, and spatial rate are retained.
No tendon, anatomical validation, or conservative active-energy claim is made.
The acquired upstream C++ remains an external numerical oracle, never a runtime
dependency; its provenance and LGPL notice live in the reference fixture.
"""

from __future__ import annotations

from math import log1p

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ....ein import contract
from ....operators.mechanics import finite_strain_kinematics


def _real(value: ArrayLike, shape: tuple[int, ...], name: str) -> Array:
    result = jnp.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real.")
    if not jnp.issubdtype(result.dtype, jnp.inexact):
        result = result.astype(float)
    return result


class Almonacid2024MaterialParameters(StrictModule):
    """Dynamic scalar material leaves, selected for one tissue by the FE owner.

    ``base_coefficients`` are the three dimensionless source Yeoh coefficients;
    ``maximum_base_stress_Pa * base_scale`` supplies their stress normalization.
    Fat's ``fat_c1_Pa`` already has stress units and is NOT multiplied by that
    normalization. Aponeurosis uses the four zero fat inputs of PointHistory.
    Values are checked in the point response so transformed/trainable parameters
    cannot bypass physical-domain evidence. There are no implicit defaults.
    """

    maximum_fiber_stress_Pa: Array
    bulk_modulus_Pa: Array
    maximum_base_stress_Pa: Array
    base_scale: Array
    maximum_strain_rate_per_s: Array
    fat_bulk_modulus_Pa: Array
    fat_scale: Array
    fat_c1_Pa: Array
    fat_fraction: Array
    base_coefficients: Array

    def __init__(
        self,
        maximum_fiber_stress_Pa: ArrayLike,
        bulk_modulus_Pa: ArrayLike,
        maximum_base_stress_Pa: ArrayLike,
        base_scale: ArrayLike,
        maximum_strain_rate_per_s: ArrayLike,
        fat_bulk_modulus_Pa: ArrayLike,
        fat_scale: ArrayLike,
        fat_c1_Pa: ArrayLike,
        fat_fraction: ArrayLike,
        base_coefficients: ArrayLike,
    ):
        self.maximum_fiber_stress_Pa = _real(
            maximum_fiber_stress_Pa, (), "maximum_fiber_stress_Pa"
        )
        self.bulk_modulus_Pa = _real(bulk_modulus_Pa, (), "bulk_modulus_Pa")
        self.maximum_base_stress_Pa = _real(
            maximum_base_stress_Pa, (), "maximum_base_stress_Pa"
        )
        self.base_scale = _real(base_scale, (), "base_scale")
        self.maximum_strain_rate_per_s = _real(
            maximum_strain_rate_per_s, (), "maximum_strain_rate_per_s"
        )
        self.fat_bulk_modulus_Pa = _real(fat_bulk_modulus_Pa, (), "fat_bulk_modulus_Pa")
        self.fat_scale = _real(fat_scale, (), "fat_scale")
        self.fat_c1_Pa = _real(fat_c1_Pa, (), "fat_c1_Pa")
        self.fat_fraction = _real(fat_fraction, (), "fat_fraction")
        self.base_coefficients = _real(base_coefficients, (3,), "base_coefficients")


class Almonacid2024MaterialResponse(StrictModule):
    """Reference/current stresses and mixed residuals at one material point.

    The three isochoric contributions are mixture-weighted and additive; the
    base contribution includes fat. This differs intentionally from upstream
    diagnostic component getters, which report unweighted muscle components.
    Passive energy is W_iso_passive(F) + U(dilation), referenced to fiber stretch
    one and dilation one. It excludes p*(detF-dilation), all active power, and
    kinetic energy. Aponeurosis has source prestress at stretch one; its fiber
    energy can therefore be negative under compression with this energy datum.
    Invalid physical points have non-finite values AND direct stress/energy
    derivatives, rather than a reflected/repaired deformation gradient.
    """

    first_piola_Pa: Array
    kirchhoff_Pa: Array
    kirchhoff_active_Pa: Array
    kirchhoff_passive_fiber_Pa: Array
    kirchhoff_base_Pa: Array
    volume_constraint: Array
    dilation_residual_Pa: Array
    passive_energy_density_J_per_m3: Array
    volume_ratio: Array
    fiber_stretch: Array
    isochoric_fiber_stretch: Array
    normalized_strain_rate: Array
    admissible: Array


def _active_force_length(stretch: Array) -> Array:
    # Seven fitted harmonics, including the source's closed support endpoints.
    fitted = (
        0.642587074375392 * jnp.sin(1.290128342448810 * stretch + 0.629168420414746)
        + 0.325979591577056 * jnp.sin(5.308969899884336 * stretch - 4.520101562237307)
        + 0.328204247867325 * jnp.sin(6.744187042136006 * stretch + 1.689155892259429)
        + 0.015388902741327 * jnp.sin(19.823676877725276 * stretch - 7.386155292116579)
        + 0.139240359517525 * jnp.sin(8.038287396059996 * stretch + 2.543022326676525)
        + 0.001801867529599 * jnp.sin(32.237736486095052 * stretch - 6.454098315528945)
        + 0.012560837549867 * jnp.sin(23.117614057963024 * stretch - 2.643346778503341)
    )
    return jnp.where((stretch >= 0.4) & (stretch <= 1.75), fitted, 0.0)


def _active_force_velocity(rate: Array) -> Array:
    first = rate + 1.2
    second = rate + 0.25
    fourth = rate - 0.05
    return jnp.select(
        (rate < -1.2, rate < -0.25, rate < 0.0, rate < 0.05, rate < 0.75),
        (
            jnp.zeros_like(rate),
            0.25792669408341773 * first**3 + 0.14317485143460784 * first**2,
            29.825565394304522 * second**3
            - 0.9435495605479662 * second**2
            + 0.9703687419567255 * second
            + 0.3503552027590582,
            -3165.6847983144276 * rate**3
            + 186.19612494819665 * rate**2
            + 6.090887473114851 * rate
            + 1.0,
            0.6882206253246714 * fourth**3
            - 1.413963071288272 * fourth**2
            + 0.9678639805763023 * fourth
            + 1.3743240862369306,
        ),
        default=1.5950466421954534,
    )


# (lower stretch, quadratic, linear, constant) for sigma(stretch)/sigma_0.
_MUSCLE_SEGMENTS = (
    (1.0, 2.353844827629192, 0.0, 0.0),
    (1.25, 3.436356700507747, 1.176922413814596, 0.1471153017268245),
    (1.5, 0.4274082856676522, 2.8951007640684696, 0.6561181989622077),
    (1.65, 0.0, 3.023323249768765, 1.1),
)
_APONEUROSIS_SEGMENTS = (
    (1.0, 515.8820342030662, 0.01, 0.01),
    (1.01, 600.5902422602494, 10.327640684061333, 0.06168820342030671),
    (1.02, -9.97532175746044, 22.33944552926633, 0.22502363448694518),
    (1.15, 0.0, 19.745861872326614, 2.9605686155904847),
)


def _fiber_primitive(stretch, lower, quadratic, linear, constant, logarithm):
    """Exact integral from lower to stretch of sigma(s)/s, not sigma(s)."""
    offset = stretch - lower
    log_ratio = logarithm(offset / lower)
    return (
        quadratic * (0.5 * offset**2 - lower * offset + lower**2 * log_ratio)
        + linear * (offset - lower * log_ratio)
        + constant * log_ratio
    )


def _passive_fiber(stretch: Array, *, aponeurosis: bool) -> tuple[Array, Array]:
    segments = _APONEUROSIS_SEGMENTS if aponeurosis else _MUSCLE_SEGMENTS
    # Muscle assigns each internal knot to its left branch; aponeurosis to its
    # right. AD follows those exact inequalities, not a smoothed replacement.
    conditions = tuple(
        stretch < segment[0] if aponeurosis else stretch <= segment[0]
        for segment in segments
    )
    stresses = [0.01 * stretch if aponeurosis else jnp.zeros_like(stretch)]
    energies = [0.01 * (stretch - 1.0) if aponeurosis else jnp.zeros_like(stretch)]
    accumulated = 0.0
    for index, (lower, quadratic, linear, constant) in enumerate(segments):
        offset = stretch - lower
        stresses.append(quadratic * offset**2 + linear * offset + constant)
        # Inactive tensile primitives must remain finite under reverse AD.
        primitive_stretch = jnp.where(stretch < lower, lower, stretch)
        energies.append(
            accumulated
            + _fiber_primitive(
                primitive_stretch, lower, quadratic, linear, constant, jnp.log1p
            )
        )
        if index + 1 < len(segments):
            accumulated += _fiber_primitive(
                segments[index + 1][0], lower, quadratic, linear, constant, log1p
            )
    return (
        jnp.select(conditions, stresses[:-1], default=stresses[-1]),
        jnp.select(conditions, energies[:-1], default=energies[-1]),
    )


def almonacid_2024_material_response(
    parameters: Almonacid2024MaterialParameters,
    F: ArrayLike,
    F_previous: ArrayLike,
    pressure_Pa: ArrayLike,
    dilation: ArrayLike,
    activation: ArrayLike,
    dt_s: ArrayLike,
    reference_direction: ArrayLike,
    tissue_id: ArrayLike,
    *,
    dynamic: bool,
) -> Almonacid2024MaterialResponse:
    """Evaluate one source point; JIT/vmap over points or dynamic parameters.

    ``dynamic`` is a static mode. As in the source update, the diagnostic rate
    uses grad_X(v)=(F-F_previous)/dt in both modes. Only dynamic active stress is
    velocity-dependent. Its normalized rate is a_bar dot dev(sym(grad_X(v)
    F^-1)) a_bar / (lambda_bar * maximum_strain_rate_per_s), not the secant
    change in fiber length. Tangents are AD of these stresses, including that
    full rate dependence, not the source's approximate hand-coded Newton tensor.
    ``reference_direction`` and ``tissue_id`` belong to nontrainable FE topology;
    the source accepts any finite nonzero reference direction without rescaling.
    """
    deformation = _real(F, (3, 3), "F")
    previous = _real(F_previous, (3, 3), "F_previous")
    pressure = _real(pressure_Pa, (), "pressure_Pa")
    dilation_ = _real(dilation, (), "dilation")
    activation_ = _real(activation, (), "activation")
    dt = _real(dt_s, (), "dt_s")
    direction = _real(reference_direction, (3,), "reference_direction")
    tissue = jnp.asarray(tissue_id)
    if tissue.shape != () or jnp.issubdtype(tissue.dtype, jnp.complexfloating):
        raise ValueError("tissue_id must be one real scalar (1 muscle, 2 aponeurosis).")
    if not isinstance(dynamic, bool):
        raise TypeError("dynamic must be a static bool.")

    kinematics = finite_strain_kinematics(deformation)
    jacobian = kinematics.jacobian
    inverse = kinematics.inverse_deformation_gradient
    identity = jnp.eye(3, dtype=deformation.dtype)
    factor = jacobian ** (-1.0 / 3.0)
    isochoric_deformation = factor * deformation
    orientation = contract("ij,j->i", isochoric_deformation, direction)
    stretch_squared = contract("i,i->", orientation, orientation)
    stretch = jnp.sqrt(stretch_squared)
    structure = contract("i,j->ij", orientation, orientation) / stretch_squared
    b_bar = factor**2 * kinematics.left_cauchy_green
    invariant_offset = jnp.trace(b_bar) - 3.0
    spatial_velocity_gradient = contract(
        "ij,jk->ik", (deformation - previous) / dt, inverse
    )
    symmetric_rate = 0.5 * (spatial_velocity_gradient + spatial_velocity_gradient.T)
    deviatoric_rate = symmetric_rate - jnp.trace(symmetric_rate) / 3.0 * identity
    rate = contract("i,ij,j->", orientation, deviatoric_rate, orientation) / (
        stretch * parameters.maximum_strain_rate_per_s
    )

    muscle_stress, muscle_energy = _passive_fiber(stretch, aponeurosis=False)
    aponeurosis_stress, aponeurosis_energy = _passive_fiber(stretch, aponeurosis=True)
    muscle = tissue == 1
    passive_stress = jnp.where(muscle, muscle_stress, aponeurosis_stress)
    fiber_energy = jnp.where(muscle, muscle_energy, aponeurosis_energy)
    muscle_fraction = jnp.where(muscle, 1.0 - parameters.fat_fraction, 1.0)
    fat_fraction = jnp.where(muscle, parameters.fat_fraction, 0.0)
    active_stress = (
        parameters.maximum_fiber_stress_Pa
        * activation_
        * _active_force_length(stretch)
        * (_active_force_velocity(rate) if dynamic else 1.0)
    )
    active_bar = jnp.where(muscle, muscle_fraction * active_stress, 0.0) * structure
    passive_bar = (
        muscle_fraction * parameters.maximum_fiber_stress_Pa * passive_stress * structure
    )
    c1, c2, c3 = parameters.base_coefficients
    base_normalization = parameters.maximum_base_stress_Pa * parameters.base_scale
    fat_normalization = parameters.fat_scale * parameters.fat_c1_Pa
    base_bar = (
        2.0
        * (
            muscle_fraction
            * base_normalization
            * (c1 + 2.0 * c2 * invariant_offset + 3.0 * c3 * invariant_offset**2)
            + fat_fraction * fat_normalization
        )
        * b_bar
    )
    active = active_bar - jnp.trace(active_bar) / 3.0 * identity
    passive = passive_bar - jnp.trace(passive_bar) / 3.0 * identity
    base = base_bar - jnp.trace(base_bar) / 3.0 * identity
    kirchhoff = active + passive + base + pressure * jacobian * identity
    first_piola = contract("ij,kj->ik", kirchhoff, inverse)
    bulk = (
        muscle_fraction * parameters.bulk_modulus_Pa
        + fat_fraction * parameters.fat_bulk_modulus_Pa
    )
    dilation_residual = 0.5 * bulk * (dilation_ - 1.0 / dilation_) - pressure
    passive_energy = (
        muscle_fraction
        * (
            parameters.maximum_fiber_stress_Pa * fiber_energy
            + base_normalization
            * invariant_offset
            * (c1 + invariant_offset * (c2 + invariant_offset * c3))
        )
        + fat_fraction * fat_normalization * invariant_offset
        + 0.25 * bulk * (dilation_**2 - 1.0 - 2.0 * jnp.log(dilation_))
    )

    parameter_values = jnp.stack(
        (
            parameters.maximum_fiber_stress_Pa,
            parameters.bulk_modulus_Pa,
            parameters.maximum_base_stress_Pa,
            parameters.base_scale,
            parameters.maximum_strain_rate_per_s,
            parameters.fat_bulk_modulus_Pa,
            parameters.fat_scale,
            parameters.fat_c1_Pa,
            parameters.fat_fraction,
        )
    )
    # The committed gradient is not inverted: only its orientation is needed.
    previous_jacobian = contract(
        "i,i->", previous[0], jnp.cross(previous[1], previous[2])
    )
    admissible = (
        kinematics.admissible
        & jnp.all(jnp.isfinite(previous))
        & jnp.isfinite(previous_jacobian)
        & (previous_jacobian > 0.0)
        & jnp.all(jnp.isfinite(parameter_values) & (parameter_values >= 0.0))
        & jnp.all(jnp.isfinite(parameters.base_coefficients))
        & (parameters.bulk_modulus_Pa > 0.0)
        & (parameters.maximum_strain_rate_per_s > 0.0)
        & (parameters.fat_fraction <= 1.0)
        & (bulk > 0.0)
        & ((tissue == 1) | (tissue == 2))
        & (
            muscle
            | (
                (parameters.fat_fraction == 0.0)
                & (parameters.fat_bulk_modulus_Pa == 0.0)
                & (parameters.fat_scale == 0.0)
                & (parameters.fat_c1_Pa == 0.0)
            )
        )
        & jnp.all(jnp.isfinite(direction))
        & (stretch > 0.0)
        & jnp.isfinite(pressure)
        & jnp.isfinite(dilation_)
        & (dilation_ > 0.0)
        & jnp.isfinite(activation_)
        & jnp.isfinite(dt)
        & (dt > 0.0)
        & jnp.isfinite(rate)
        & jnp.isfinite(passive_energy)
        & jnp.isfinite(dilation_residual)
        & jnp.all(jnp.isfinite(first_piola))
    )
    # A constant-NaN selection has a zero AD branch. Multiplication preserves
    # invalid JVP/VJP paths instead, including detF<0 with a finite inverse.
    validity = jnp.where(admissible, 1.0, jnp.nan)
    return Almonacid2024MaterialResponse(
        first_piola * validity,
        kirchhoff * validity,
        active * validity,
        passive * validity,
        base * validity,
        (jacobian - dilation_) * validity,
        dilation_residual * validity,
        passive_energy * validity,
        jacobian * validity,
        jacobian ** (1.0 / 3.0) * stretch * validity,
        stretch * validity,
        rate * validity,
        admissible,
    )


__all__ = [
    "Almonacid2024MaterialParameters",
    "Almonacid2024MaterialResponse",
    "almonacid_2024_material_response",
]
