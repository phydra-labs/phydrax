#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stationary Kerr Killing-horizon thermodynamics.

The classical kernels use geometric units: ``mass`` and ``irreducible_mass``
are lengths, angular momentum is length squared, area is length squared, and
surface gravity and angular velocity are inverse lengths. Entropy and
temperature are only produced through an explicit :class:`RelativityScaleContract`.
"""

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class KerrBranchCode(IntEnum):
    """JAX-compatible classification of a geometric Kerr input."""

    INDETERMINATE = 0
    SUBEXTREMAL = 1
    EXTREMAL = 2
    OVEREXTREMAL = 3


class KerrInput(StrictModule, NonTrainableState):
    """One geometric Kerr mass and signed angular momentum.

    Invalid values are retained rather than rejected so that classification and
    scientific status remain explicit. Both values are scalar fixed state;
    differentiable diagnostics accept tangents separately and never reinterpret
    this record as trainable model state.
    """

    mass: Array
    angular_momentum: Array
    input_id: str = eqx.field(static=True)

    def __init__(self, mass: ArrayLike, angular_momentum: ArrayLike, /):
        dtype = jnp.result_type(mass, angular_momentum, 0.0)
        mass_ = jnp.asarray(mass, dtype=dtype)
        angular_momentum_ = jnp.asarray(angular_momentum, dtype=dtype)
        if mass_.shape != () or angular_momentum_.shape != ():
            raise ValueError("Kerr mass and angular momentum must be scalars.")
        self.mass = mass_
        self.angular_momentum = angular_momentum_
        self.input_id = canonical_fingerprint(
            {
                "kind": "geometric-kerr-input",
                "mass": np.asarray(mass_),
                "angular_momentum": np.asarray(angular_momentum_),
            }
        )


class KerrBranch(StrictModule):
    """Mutually exclusive Kerr branch evidence for one input."""

    dimensionless_spin: Array
    extremality_margin: Array
    finite: Array
    subextremal: Array
    extremal: Array
    overextremal: Array
    indeterminate: Array
    code: Array
    input_id: str = eqx.field(static=True)


class StationaryKillingHorizonResult(StrictModule):
    """Classical properties of the stationary Kerr Killing horizon.

    This is deliberately not a universal horizon record: it does not represent
    apparent, dynamical, trapping, or event horizons in evolving spacetimes.
    """

    parameters: KerrInput
    branch: KerrBranch
    spin_parameter: Array
    outer_radius: Array
    inner_radius: Array
    area: Array
    irreducible_mass: Array
    surface_gravity: Array
    angular_velocity: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class KerrEntropyTemperatureResult(StrictModule):
    """Bekenstein--Hawking entropy and temperature in declared physical units."""

    horizon: StationaryKillingHorizonResult
    scale: RelativityScaleContract
    surface_gravity_acceleration: Array
    entropy: Array
    temperature: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class KerrFirstLawSmarrEvidence(StrictModule):
    """Directional first-law JVP and algebraic Smarr identity evidence."""

    horizon: StationaryKillingHorizonResult
    mass_tangent: Array
    angular_momentum_tangent: Array
    area_tangent: Array
    irreducible_mass_tangent: Array
    surface_gravity_tangent: Array
    angular_velocity_tangent: Array
    thermal_term: Array
    rotational_term: Array
    first_law_residual: Array
    normalized_first_law_residual: Array
    smarr_residual: Array
    normalized_smarr_residual: Array
    first_law_satisfied: Array
    smarr_satisfied: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class FixedAngularMomentumResponse(StrictModule):
    """Thermal response along the explicitly fixed-J Kerr ensemble.

    ``mass_temperature_response`` is dM/dT_g and
    ``entropy_temperature_response`` is d(A/4)/dT_g for
    T_g = κ/(2π). ``thermal_mass_response`` is their product
    T_g d(A/4)/dT_g. These ensemble-specific geometric responses are not a
    generic heat-capacity abstraction.
    """

    horizon: StationaryKillingHorizonResult
    geometric_temperature: Array
    temperature_mass_slope: Array
    mass_temperature_response: Array
    entropy_temperature_response: Array
    thermal_mass_response: Array
    conditioning_margin: Array
    condition_number: Array
    singular: Array
    conditioning_tolerance: float = eqx.field(static=True)
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class FixedAngularVelocityResponse(StrictModule):
    """Thermal and rotational response along the fixed-horizon-Ω ensemble.

    The mass response is separated into T_g d(A/4)/dT_g and
    Ω dJ/dT_g, with their first-law residual reported independently. This
    path is a distinct ensemble response, not a universal heat capacity.
    """

    horizon: StationaryKillingHorizonResult
    geometric_temperature: Array
    angular_momentum_mass_slope: Array
    temperature_mass_slope: Array
    mass_temperature_response: Array
    angular_momentum_temperature_response: Array
    entropy_temperature_response: Array
    thermal_mass_response: Array
    rotational_mass_response: Array
    first_law_response_residual: Array
    conditioning_margin: Array
    condition_number: Array
    singular: Array
    conditioning_tolerance: float = eqx.field(static=True)
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


def classify_kerr(parameters: KerrInput, /) -> KerrBranch:
    """Classify subextremal, extremal, overextremal, and indeterminate input."""

    if not isinstance(parameters, KerrInput):
        raise TypeError("parameters must be a KerrInput.")
    mass = parameters.mass
    angular_momentum = parameters.angular_momentum
    mass_squared = mass * mass
    angular_momentum_magnitude = jnp.abs(angular_momentum)
    dimensionless_spin = angular_momentum / mass_squared
    magnitude = jnp.abs(dimensionless_spin)
    finite = (
        jnp.isfinite(mass)
        & jnp.isfinite(angular_momentum)
        & jnp.isfinite(mass_squared)
        & jnp.isfinite(dimensionless_spin)
    )
    determinate = finite & (mass > 0)
    subextremal = determinate & (angular_momentum_magnitude < mass_squared)
    extremal = determinate & (angular_momentum_magnitude == mass_squared)
    overextremal = determinate & (angular_momentum_magnitude > mass_squared)
    indeterminate = ~(subextremal | extremal | overextremal)
    code = jnp.where(
        subextremal,
        int(KerrBranchCode.SUBEXTREMAL),
        jnp.where(
            extremal,
            int(KerrBranchCode.EXTREMAL),
            jnp.where(
                overextremal,
                int(KerrBranchCode.OVEREXTREMAL),
                int(KerrBranchCode.INDETERMINATE),
            ),
        ),
    ).astype(jnp.int32)
    return KerrBranch(
        dimensionless_spin,
        1.0 - magnitude,
        finite,
        subextremal,
        extremal,
        overextremal,
        indeterminate,
        code,
        parameters.input_id,
    )


def _kerr_spin_and_root(mass: Array, angular_momentum: Array, /) -> tuple[Array, Array]:
    mass_squared = mass * mass
    angular_momentum_magnitude = jnp.abs(angular_momentum)
    exactly_extremal = (mass > 0) & (angular_momentum_magnitude == mass_squared)
    raw_spin = angular_momentum / mass_squared
    dimensionless_spin = jnp.where(exactly_extremal, jnp.sign(angular_momentum), raw_spin)
    magnitude = jnp.abs(dimensionless_spin)
    raw_root = jnp.sqrt((1.0 - magnitude) * (1.0 + magnitude))
    extremality_root = jnp.where(exactly_extremal, 0.0, raw_root)
    return dimensionless_spin, extremality_root


def _kerr_quantities(mass: Array, angular_momentum: Array, /) -> tuple[Array, ...]:
    mass_squared = mass * mass
    dimensionless_spin, extremality_root = _kerr_spin_and_root(mass, angular_momentum)
    one_plus_root = 1.0 + extremality_root
    spin_parameter = angular_momentum / mass
    outer_radius = mass * one_plus_root
    # This product form avoids cancellation at small spin.
    inner_radius = mass * dimensionless_spin**2 / one_plus_root
    area = 8.0 * jnp.pi * mass_squared * one_plus_root
    irreducible_mass = mass * jnp.sqrt(0.5 * one_plus_root)
    surface_gravity = extremality_root / (2.0 * mass * one_plus_root)
    angular_velocity = dimensionless_spin / (2.0 * mass * one_plus_root)
    return (
        spin_parameter,
        outer_radius,
        inner_radius,
        area,
        irreducible_mass,
        surface_gravity,
        angular_velocity,
        extremality_root,
    )


def evaluate_stationary_kerr_horizon(
    parameters: KerrInput, /
) -> StationaryKillingHorizonResult:
    """Evaluate the stationary Kerr Killing horizon without repairing input."""

    branch = classify_kerr(parameters)
    raw = _kerr_quantities(parameters.mass, parameters.angular_momentum)
    physical = branch.subextremal | branch.extremal
    invalid = jnp.asarray(jnp.nan, dtype=parameters.mass.dtype)
    values = tuple(jnp.where(physical, value, invalid) for value in raw[:-1])
    finite = branch.finite & jnp.all(
        jnp.stack(tuple(jnp.isfinite(value) for value in values))
    )
    converged = ~branch.indeterminate
    physically_valid = physical
    qualified = finite & converged & physically_valid
    derivative_valid = qualified & branch.subextremal
    return StationaryKillingHorizonResult(
        parameters,
        branch,
        *values,
        finite,
        converged,
        physically_valid,
        qualified,
        derivative_valid,
    )


def evaluate_kerr_entropy_temperature(
    horizon: StationaryKillingHorizonResult,
    scale: RelativityScaleContract,
    /,
) -> KerrEntropyTemperatureResult:
    """Convert classical area and geometric surface gravity using explicit units."""

    if not isinstance(horizon, StationaryKillingHorizonResult):
        raise TypeError("horizon must be a StationaryKillingHorizonResult.")
    if not isinstance(scale, RelativityScaleContract):
        raise TypeError("scale must be a RelativityScaleContract.")
    surface_gravity_acceleration = (
        jnp.asarray(float(scale.speed_of_light), dtype=horizon.surface_gravity.dtype) ** 2
        * horizon.surface_gravity
    )
    entropy = scale.area_to_entropy(horizon.area)
    temperature = scale.surface_gravity_to_temperature(surface_gravity_acceleration)
    finite = (
        horizon.finite
        & jnp.isfinite(surface_gravity_acceleration)
        & jnp.isfinite(entropy)
        & jnp.isfinite(temperature)
    )
    qualified = horizon.qualified & finite
    return KerrEntropyTemperatureResult(
        horizon,
        scale,
        surface_gravity_acceleration,
        entropy,
        temperature,
        finite,
        horizon.converged,
        horizon.physically_valid,
        qualified,
        horizon.derivative_valid & qualified,
    )


def _invalid_tangent_values(dtype) -> tuple[Array, ...]:
    invalid = jnp.asarray(jnp.nan, dtype=dtype)
    return invalid, invalid, invalid, invalid


def evaluate_kerr_first_law(
    parameters: KerrInput,
    mass_tangent: ArrayLike,
    angular_momentum_tangent: ArrayLike,
    /,
) -> KerrFirstLawSmarrEvidence:
    """Evaluate a directional first-law JVP only on the smooth subextremal branch."""

    if not isinstance(parameters, KerrInput):
        raise TypeError("parameters must be a KerrInput.")
    horizon = evaluate_stationary_kerr_horizon(parameters)
    mass_tangent_ = jnp.asarray(mass_tangent, dtype=parameters.mass.dtype)
    angular_momentum_tangent_ = jnp.asarray(
        angular_momentum_tangent, dtype=parameters.mass.dtype
    )
    if mass_tangent_.shape != () or angular_momentum_tangent_.shape != ():
        raise ValueError("Kerr first-law tangents must be scalars.")
    tangent_finite = jnp.isfinite(mass_tangent_) & jnp.isfinite(angular_momentum_tangent_)

    def valid_jvp(_):
        def quantities(mass, angular_momentum):
            values = _kerr_quantities(mass, angular_momentum)
            return values[3], values[4], values[5], values[6]

        _, tangents = jax.jvp(
            quantities,
            (parameters.mass, parameters.angular_momentum),
            (mass_tangent_, angular_momentum_tangent_),
        )
        return tangents

    (
        area_tangent,
        irreducible_tangent,
        surface_gravity_tangent,
        angular_velocity_tangent,
    ) = jax.lax.cond(
        horizon.derivative_valid & tangent_finite,
        valid_jvp,
        lambda _: _invalid_tangent_values(parameters.mass.dtype),
        operand=None,
    )
    thermal_term = horizon.surface_gravity * area_tangent / (8.0 * jnp.pi)
    rotational_term = horizon.angular_velocity * angular_momentum_tangent_
    first_law_residual = mass_tangent_ - thermal_term - rotational_term
    first_scale = (
        jnp.abs(mass_tangent_) + jnp.abs(thermal_term) + jnp.abs(rotational_term)
    )
    normalized_first_law_residual = jnp.where(
        first_scale > 0,
        jnp.abs(first_law_residual) / first_scale,
        jnp.abs(first_law_residual),
    )
    smarr_thermal = horizon.surface_gravity * horizon.area / (4.0 * jnp.pi)
    smarr_rotational = 2.0 * horizon.angular_velocity * parameters.angular_momentum
    smarr_residual = parameters.mass - smarr_thermal - smarr_rotational
    smarr_scale = (
        jnp.abs(parameters.mass) + jnp.abs(smarr_thermal) + jnp.abs(smarr_rotational)
    )
    normalized_smarr_residual = jnp.where(
        smarr_scale > 0,
        jnp.abs(smarr_residual) / smarr_scale,
        jnp.abs(smarr_residual),
    )
    derivative_finite = jnp.all(
        jnp.stack(
            (
                jnp.isfinite(area_tangent),
                jnp.isfinite(irreducible_tangent),
                jnp.isfinite(surface_gravity_tangent),
                jnp.isfinite(angular_velocity_tangent),
                jnp.isfinite(first_law_residual),
                jnp.isfinite(smarr_residual),
            )
        )
    )
    finite = horizon.finite & tangent_finite & derivative_finite
    physically_valid = horizon.physically_valid & tangent_finite
    derivative_valid = horizon.derivative_valid & tangent_finite & derivative_finite
    tolerance = jnp.asarray(
        128.0 * jnp.finfo(parameters.mass.dtype).eps, dtype=parameters.mass.dtype
    )
    first_law_satisfied = derivative_valid & (normalized_first_law_residual <= tolerance)
    smarr_satisfied = (
        horizon.qualified
        & tangent_finite
        & jnp.isfinite(smarr_residual)
        & (normalized_smarr_residual <= tolerance)
    )
    qualified = first_law_satisfied & smarr_satisfied
    return KerrFirstLawSmarrEvidence(
        horizon,
        mass_tangent_,
        angular_momentum_tangent_,
        area_tangent,
        irreducible_tangent,
        surface_gravity_tangent,
        angular_velocity_tangent,
        thermal_term,
        rotational_term,
        first_law_residual,
        normalized_first_law_residual,
        smarr_residual,
        normalized_smarr_residual,
        first_law_satisfied,
        smarr_satisfied,
        finite,
        horizon.converged,
        physically_valid,
        qualified,
        derivative_valid,
    )


def _conditioning_tolerance(value: float, /) -> float:
    tolerance = float(value)
    if not isfinite(tolerance) or tolerance <= 0.0 or tolerance >= 1.0:
        raise ValueError(
            "conditioning_tolerance must be finite and between zero and one."
        )
    return tolerance


def evaluate_fixed_angular_momentum_response(
    parameters: KerrInput,
    /,
    *,
    conditioning_tolerance: float = 1.0e-12,
) -> FixedAngularMomentumResponse:
    """Evaluate the fixed-J response, exposing the Davies singularity."""

    if not isinstance(parameters, KerrInput):
        raise TypeError("parameters must be a KerrInput.")
    tolerance = _conditioning_tolerance(conditioning_tolerance)
    horizon = evaluate_stationary_kerr_horizon(parameters)
    mass = parameters.mass
    _, root = _kerr_spin_and_root(mass, parameters.angular_momentum)
    one_plus_root = 1.0 + root
    davies_denominator = 2.0 - 2.0 * root - root**2
    davies_margin = jnp.abs(davies_denominator) / (2.0 + 2.0 * root + root**2)
    # This minimum combines two diagnostic margins; it never repairs a state.
    conditioning_margin = jnp.minimum(root, davies_margin)
    singular = horizon.physically_valid & (
        ~jnp.isfinite(conditioning_margin) | (conditioning_margin <= tolerance)
    )
    regular = horizon.derivative_valid & ~singular
    geometric_temperature = horizon.surface_gravity / (2.0 * jnp.pi)
    temperature_mass_slope_raw = davies_denominator / (
        4.0 * jnp.pi * mass**2 * root * one_plus_root
    )
    mass_response_raw = 4.0 * jnp.pi * mass**2 * root * one_plus_root / davies_denominator
    entropy_response_raw = (
        16.0 * jnp.pi**2 * mass**3 * one_plus_root**2 / davies_denominator
    )
    invalid = jnp.asarray(jnp.nan, dtype=mass.dtype)
    temperature_mass_slope = jnp.where(regular, temperature_mass_slope_raw, invalid)
    mass_temperature_response = jnp.where(regular, mass_response_raw, invalid)
    entropy_temperature_response = jnp.where(regular, entropy_response_raw, invalid)
    thermal_mass_response = jnp.where(regular, mass_response_raw, invalid)
    condition_number = jnp.where(
        conditioning_margin > 0,
        1.0 / conditioning_margin,
        jnp.asarray(jnp.inf, dtype=mass.dtype),
    )
    response_finite = jnp.all(
        jnp.stack(
            (
                jnp.isfinite(geometric_temperature),
                jnp.isfinite(temperature_mass_slope),
                jnp.isfinite(mass_temperature_response),
                jnp.isfinite(entropy_temperature_response),
                jnp.isfinite(thermal_mass_response),
            )
        )
    )
    finite = horizon.finite & response_finite
    derivative_valid = horizon.derivative_valid & regular & response_finite
    qualified = finite & horizon.physically_valid & ~singular & derivative_valid
    return FixedAngularMomentumResponse(
        horizon,
        geometric_temperature,
        temperature_mass_slope,
        mass_temperature_response,
        entropy_temperature_response,
        thermal_mass_response,
        conditioning_margin,
        condition_number,
        singular,
        tolerance,
        finite,
        horizon.converged,
        horizon.physically_valid,
        qualified,
        derivative_valid,
    )


def evaluate_fixed_angular_velocity_response(
    parameters: KerrInput,
    /,
    *,
    conditioning_tolerance: float = 1.0e-12,
) -> FixedAngularVelocityResponse:
    """Evaluate fixed-Ω thermal/rotational response and extremal conditioning."""

    if not isinstance(parameters, KerrInput):
        raise TypeError("parameters must be a KerrInput.")
    tolerance = _conditioning_tolerance(conditioning_tolerance)
    horizon = evaluate_stationary_kerr_horizon(parameters)
    mass = parameters.mass
    spin = horizon.branch.dimensionless_spin
    _, root = _kerr_spin_and_root(mass, parameters.angular_momentum)
    one_plus_root = 1.0 + root
    conditioning_margin = root
    singular = horizon.physically_valid & (
        ~jnp.isfinite(conditioning_margin) | (conditioning_margin <= tolerance)
    )
    regular = horizon.derivative_valid & ~singular
    geometric_temperature = horizon.surface_gravity / (2.0 * jnp.pi)
    angular_momentum_mass_slope_raw = mass * spin * (2.0 + root)
    temperature_mass_slope_raw = -1.0 / (4.0 * jnp.pi * mass**2 * one_plus_root)
    mass_response_raw = -4.0 * jnp.pi * mass**2 * one_plus_root
    angular_momentum_response_raw = (
        -4.0 * jnp.pi * mass**3 * spin * one_plus_root * (2.0 + root)
    )
    entropy_response_raw = -8.0 * jnp.pi**2 * mass**3 * one_plus_root**3
    thermal_response_raw = -2.0 * jnp.pi * mass**2 * root * one_plus_root**2
    rotational_response_raw = horizon.angular_velocity * angular_momentum_response_raw
    response_residual_raw = (
        mass_response_raw - thermal_response_raw - rotational_response_raw
    )
    invalid = jnp.asarray(jnp.nan, dtype=mass.dtype)
    response_values = tuple(
        jnp.where(regular, value, invalid)
        for value in (
            angular_momentum_mass_slope_raw,
            temperature_mass_slope_raw,
            mass_response_raw,
            angular_momentum_response_raw,
            entropy_response_raw,
            thermal_response_raw,
            rotational_response_raw,
            response_residual_raw,
        )
    )
    condition_number = jnp.where(
        conditioning_margin > 0,
        1.0 / conditioning_margin,
        jnp.asarray(jnp.inf, dtype=mass.dtype),
    )
    response_finite = jnp.isfinite(geometric_temperature) & jnp.all(
        jnp.stack(tuple(jnp.isfinite(value) for value in response_values))
    )
    finite = horizon.finite & response_finite
    derivative_valid = horizon.derivative_valid & regular & response_finite
    tolerance_array = jnp.asarray(128.0 * jnp.finfo(mass.dtype).eps, dtype=mass.dtype)
    response_scale = (
        jnp.abs(response_values[2])
        + jnp.abs(response_values[5])
        + jnp.abs(response_values[6])
    )
    normalized_response_residual = jnp.where(
        response_scale > 0,
        jnp.abs(response_values[7]) / response_scale,
        jnp.abs(response_values[7]),
    )
    qualified = (
        finite
        & horizon.physically_valid
        & ~singular
        & derivative_valid
        & (normalized_response_residual <= tolerance_array)
    )
    return FixedAngularVelocityResponse(
        horizon,
        geometric_temperature,
        *response_values,
        conditioning_margin,
        condition_number,
        singular,
        tolerance,
        finite,
        horizon.converged,
        horizon.physically_valid,
        qualified,
        derivative_valid,
    )
