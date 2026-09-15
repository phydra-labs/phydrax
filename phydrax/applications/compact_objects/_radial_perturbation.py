#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.collocation import ChebyshevCollocation
from ._perturbation import (
    PerturbationStatus,
    RadialBoundaryCondition,
    SeparatedMode,
)


class RadialBoundaryAmplitudes(StrictModule):
    """Endpoint amplitudes in the local unit-phase asymptotic bases."""

    horizon_ingoing: Array
    horizon_outgoing: Array
    infinity_ingoing: Array
    infinity_outgoing: Array


class RadialAsymptoticEvidence(StrictModule):
    """Finite-domain evidence for the declared horizon/infinity asymptotics."""

    horizon_wave_number: Array
    infinity_wave_number: Array
    horizon_log_derivative: Array
    infinity_log_derivative: Array
    horizon_domain_ratio: Array
    infinity_domain_ratio: Array
    horizon_potential_ratio: Array
    infinity_potential_ratio: Array
    horizon_defect: Array
    infinity_defect: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class RadialResidualEvidence(StrictModule):
    """Matching and independent spectral differential-residual evidence."""

    matching_residual: Array
    differential_residual: Array
    residual_norm: Array
    relative_residual: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    node_count: int = eqx.field(static=True)
    match_index: int = eqx.field(static=True)


class KerrTeukolskyRadialCoefficients(StrictModule):
    """Coefficients of ``Delta R'' + B R' + C R = 0`` in Boyer--Lindquist ``r``."""

    delta: Array
    k: Array
    second_derivative: Array
    first_derivative: Array
    zeroth_derivative: Array
    separation_lambda: Array
    finite: Array
    domain_valid: Array


class SchwarzschildRadialPlan(StrictModule, NonTrainableState):
    """Fixed finite-domain two-sided radial matching plan.

    A first-order horizon series and fixed-order infinity
    series initialize Riccati/log-amplitude shooting on
    ``x = log((r - 2M) / M)``.  Every spectral interval uses the declared fixed
    RK4 substep count.  The reconstructed profile is independently checked with
    native Chebyshev derivative matrices; the complex matching residual is
    returned without hiding it behind an absolute value.
    """

    mode: SeparatedMode
    mass: Array
    horizon_radius: Array
    radial_nodes: Array
    collocation: ChebyshevCollocation
    boundary: RadialBoundaryCondition
    match_index: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    matching_tolerance: float = eqx.field(static=True)
    asymptotic_tolerance: float = eqx.field(static=True)
    wave_number_tolerance: float = eqx.field(static=True)
    integration_substeps: int = eqx.field(static=True)
    infinity_asymptotic_order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: SeparatedMode,
        mass: ArrayLike,
        /,
        *,
        node_count: int = 129,
        inner_radius: float | None = None,
        outer_radius: float | None = None,
        match_fraction: float = 0.5,
        boundary: RadialBoundaryCondition | None = None,
        residual_tolerance: float = 1.0e-5,
        matching_tolerance: float = 1.0e-8,
        asymptotic_tolerance: float = 5.0e-2,
        wave_number_tolerance: float = 1.0e-10,
        maximum_dimension: int = 512,
        integration_substeps: int = 32,
        infinity_asymptotic_order: int = 12,
    ):
        if not isinstance(mode, SeparatedMode):
            raise TypeError("mode must be a SeparatedMode.")
        _validate_schwarzschild_sector(mode)
        mass_value = _positive_scalar_host(mass, "mass")
        count, maximum = _collocation_capacity(node_count, maximum_dimension)
        substeps, asymptotic_order = _schwarzschild_integration_capacity(
            integration_substeps,
            infinity_asymptotic_order,
        )
        horizon = 2.0 * mass_value
        inner = horizon * (1.0 + 1.0e-4) if inner_radius is None else float(inner_radius)
        outer = 100.0 * mass_value if outer_radius is None else float(outer_radius)
        fraction = float(match_fraction)
        if (
            not math.isfinite(inner)
            or not math.isfinite(outer)
            or not horizon < inner < outer
            or not math.isfinite(fraction)
            or not 0.0 < fraction < 1.0
        ):
            raise ValueError(
                "Schwarzschild radial bounds must be finite, exterior, increasing, "
                "and have an interior match fraction."
            )
        tolerances = _radial_tolerances(
            residual_tolerance,
            matching_tolerance,
            asymptotic_tolerance,
            wave_number_tolerance,
        )
        boundary_ = (
            RadialBoundaryCondition(convention_id=mode.convention_id)
            if boundary is None
            else boundary
        )
        _validate_boundary(boundary_, mode)
        inner_coordinate = math.log((inner - horizon) / mass_value)
        outer_coordinate = math.log((outer - horizon) / mass_value)
        collocation = ChebyshevCollocation(
            count,
            lower=inner_coordinate,
            upper=outer_coordinate,
            maximum_dimension=maximum,
        )
        radial_nodes_host = horizon + mass_value * np.exp(np.asarray(collocation.nodes))
        target_radius = inner + fraction * (outer - inner)
        match_index = int(np.argmin(np.abs(radial_nodes_host - target_radius)))
        if match_index == 0 or match_index == count - 1:
            raise ValueError("match_fraction must select an interior collocation node.")
        self.mode = mode
        self.mass = jnp.asarray(mass_value)
        self.horizon_radius = jnp.asarray(horizon)
        self.radial_nodes = jnp.asarray(radial_nodes_host)
        self.collocation = collocation
        self.boundary = boundary_
        self.match_index = match_index
        (
            self.residual_tolerance,
            self.matching_tolerance,
            self.asymptotic_tolerance,
            self.wave_number_tolerance,
        ) = tolerances
        self.integration_substeps = substeps
        self.infinity_asymptotic_order = asymptotic_order
        self.plan_id = canonical_fingerprint(
            {
                "kind": "schwarzschild-radial-matching-plan",
                "mode": mode.mode_id,
                "mass": array_tree_fingerprint(np.asarray(mass_value)),
                "collocation": collocation.discretization_id,
                "radial_nodes": array_tree_fingerprint(radial_nodes_host),
                "match_index": match_index,
                "boundary": boundary_.boundary_id,
                "tolerances": list(tolerances),
                "integration_substeps": substeps,
                "infinity_asymptotic_order": asymptotic_order,
            }
        )

    @property
    def node_count(self) -> int:
        return self.collocation.count

    @property
    def coordinate_nodes(self) -> Array:
        return self.collocation.nodes

    def potential(self, radius: ArrayLike, /) -> Array:
        if self.mode.sector in ("polar", "zerilli"):
            return schwarzschild_zerilli_potential(
                radius,
                self.mass,
                self.mode.ell,
            )
        return schwarzschild_regge_wheeler_potential(
            radius,
            self.mass,
            self.mode.ell,
            spin_weight=self.mode.spin_weight,
        )

    def evaluate(
        self,
        frequency: ArrayLike,
        separation_constant: ArrayLike | None = None,
        /,
    ) -> "SchwarzschildRadialResult":
        return evaluate_schwarzschild_radial(self, frequency, separation_constant)

    def outgoing_residual(
        self,
        frequency: ArrayLike,
        separation_constant: ArrayLike | None = None,
        /,
    ) -> Array:
        return schwarzschild_outgoing_residual(self, frequency, separation_constant)


class SchwarzschildRadialResult(StrictModule):
    frequency: Array
    separation_constant: Array
    radial_nodes: Array
    solution: Array
    logarithmic_derivative: Array
    boundary_amplitudes: RadialBoundaryAmplitudes
    asymptotic: RadialAsymptoticEvidence
    residual_evidence: RadialResidualEvidence
    mode_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def residual(self) -> Array:
        return self.residual_evidence.matching_residual

    @property
    def horizon_amplitude(self) -> Array:
        if self.boundary_amplitudes.horizon_ingoing.shape != ():
            raise ValueError("Horizon amplitude evidence must be scalar.")
        return (
            self.boundary_amplitudes.horizon_ingoing
            + self.boundary_amplitudes.horizon_outgoing
        )

    @property
    def infinity_amplitude(self) -> Array:
        return (
            self.boundary_amplitudes.infinity_ingoing
            + self.boundary_amplitudes.infinity_outgoing
        )

    @property
    def finite(self) -> Array:
        return self.residual_evidence.finite

    @property
    def converged(self) -> Array:
        return self.residual_evidence.converged

    @property
    def physically_valid(self) -> Array:
        return self.residual_evidence.physically_valid

    @property
    def qualified(self) -> Array:
        return self.residual_evidence.qualified

    @property
    def valid(self) -> Array:
        return self.qualified

    @property
    def derivative_valid(self) -> Array:
        return self.residual_evidence.derivative_valid

    @property
    def status(self) -> Array:
        return self.residual_evidence.status


class KerrTeukolskyRadialPlan(StrictModule, NonTrainableState):
    """Fixed two-sided matching plan for the untransformed Teukolsky radial field."""

    mode: SeparatedMode
    mass: Array
    spin: Array
    outer_horizon_radius: Array
    inner_horizon_radius: Array
    horizon_angular_velocity: Array
    surface_gravity: Array
    radial_nodes: Array
    collocation: ChebyshevCollocation
    boundary: RadialBoundaryCondition
    match_index: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    matching_tolerance: float = eqx.field(static=True)
    asymptotic_tolerance: float = eqx.field(static=True)
    wave_number_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: SeparatedMode,
        mass: ArrayLike,
        spin: ArrayLike,
        /,
        *,
        node_count: int = 129,
        inner_radius: float | None = None,
        outer_radius: float | None = None,
        match_fraction: float = 0.5,
        boundary: RadialBoundaryCondition | None = None,
        residual_tolerance: float = 5.0e-2,
        matching_tolerance: float = 1.0e-8,
        asymptotic_tolerance: float = 5.0e-2,
        wave_number_tolerance: float = 1.0e-10,
        maximum_dimension: int = 512,
    ):
        if not isinstance(mode, SeparatedMode):
            raise TypeError("mode must be a SeparatedMode.")
        if mode.sector not in ("teukolsky", "scalar"):
            raise ValueError(
                "Kerr Teukolsky plans require sector 'teukolsky' or 'scalar'."
            )
        if mode.sector == "scalar" and mode.spin_weight != 0:
            raise ValueError("The Kerr scalar sector requires spin_weight=0.")
        mass_value = _positive_scalar_host(mass, "mass")
        spin_value = _finite_scalar_host(spin, "spin")
        if abs(spin_value) >= mass_value:
            raise ValueError("Kerr radial matching requires a strictly subextremal spin.")
        count, maximum = _collocation_capacity(node_count, maximum_dimension)
        root = math.sqrt(mass_value * mass_value - spin_value * spin_value)
        outer_horizon = mass_value + root
        inner_horizon = mass_value - root
        sigma_horizon = outer_horizon * outer_horizon + spin_value * spin_value
        angular_velocity = spin_value / sigma_horizon
        surface_gravity = (outer_horizon - inner_horizon) / (2.0 * sigma_horizon)
        inner = (
            outer_horizon + 1.0e-4 * mass_value
            if inner_radius is None
            else float(inner_radius)
        )
        outer = 100.0 * mass_value if outer_radius is None else float(outer_radius)
        fraction = float(match_fraction)
        if (
            not math.isfinite(inner)
            or not math.isfinite(outer)
            or not outer_horizon < inner < outer
            or not math.isfinite(fraction)
            or not 0.0 < fraction < 1.0
        ):
            raise ValueError(
                "Kerr radial bounds must be finite, exterior, increasing, and have "
                "an interior match fraction."
            )
        tolerances = _radial_tolerances(
            residual_tolerance,
            matching_tolerance,
            asymptotic_tolerance,
            wave_number_tolerance,
        )
        boundary_ = (
            RadialBoundaryCondition(convention_id=mode.convention_id)
            if boundary is None
            else boundary
        )
        _validate_boundary(boundary_, mode)
        inner_coordinate = math.log((inner - outer_horizon) / mass_value)
        outer_coordinate = math.log((outer - outer_horizon) / mass_value)
        collocation = ChebyshevCollocation(
            count,
            lower=inner_coordinate,
            upper=outer_coordinate,
            maximum_dimension=maximum,
        )
        radial_nodes_host = outer_horizon + mass_value * np.exp(
            np.asarray(collocation.nodes)
        )
        target_radius = inner + fraction * (outer - inner)
        match_index = int(np.argmin(np.abs(radial_nodes_host - target_radius)))
        if match_index == 0 or match_index == count - 1:
            raise ValueError("match_fraction must select an interior collocation node.")
        self.mode = mode
        self.mass = jnp.asarray(mass_value)
        self.spin = jnp.asarray(spin_value)
        self.outer_horizon_radius = jnp.asarray(outer_horizon)
        self.inner_horizon_radius = jnp.asarray(inner_horizon)
        self.horizon_angular_velocity = jnp.asarray(angular_velocity)
        self.surface_gravity = jnp.asarray(surface_gravity)
        self.radial_nodes = jnp.asarray(radial_nodes_host)
        self.collocation = collocation
        self.boundary = boundary_
        self.match_index = match_index
        (
            self.residual_tolerance,
            self.matching_tolerance,
            self.asymptotic_tolerance,
            self.wave_number_tolerance,
        ) = tolerances
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kerr-teukolsky-radial-matching-plan",
                "mode": mode.mode_id,
                "mass": array_tree_fingerprint(np.asarray(mass_value)),
                "spin": array_tree_fingerprint(np.asarray(spin_value)),
                "collocation": collocation.discretization_id,
                "radial_nodes": array_tree_fingerprint(radial_nodes_host),
                "match_index": match_index,
                "boundary": boundary_.boundary_id,
                "tolerances": list(tolerances),
                "radial_field": "untransformed Kinnersley-tetrad Teukolsky R_s",
            }
        )

    @property
    def node_count(self) -> int:
        return self.collocation.count

    @property
    def coordinate_nodes(self) -> Array:
        return self.collocation.nodes

    def coefficients(
        self,
        radius: ArrayLike,
        frequency: ArrayLike,
        separation_constant: ArrayLike,
        /,
    ) -> KerrTeukolskyRadialCoefficients:
        return kerr_teukolsky_radial_coefficients(
            self,
            radius,
            frequency,
            separation_constant,
        )

    def evaluate(
        self,
        frequency: ArrayLike,
        separation_constant: ArrayLike,
        /,
    ) -> "KerrTeukolskyRadialResult":
        return evaluate_kerr_teukolsky_radial(self, frequency, separation_constant)

    def outgoing_residual(
        self,
        frequency: ArrayLike,
        separation_constant: ArrayLike,
        /,
    ) -> Array:
        return kerr_teukolsky_outgoing_residual(
            self,
            frequency,
            separation_constant,
        )


class KerrTeukolskyRadialResult(StrictModule):
    frequency: Array
    separation_constant: Array
    separation_lambda: Array
    radial_nodes: Array
    solution: Array
    logarithmic_derivative: Array
    coefficients: KerrTeukolskyRadialCoefficients
    boundary_amplitudes: RadialBoundaryAmplitudes
    asymptotic: RadialAsymptoticEvidence
    residual_evidence: RadialResidualEvidence
    mode_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def residual(self) -> Array:
        return self.residual_evidence.matching_residual

    @property
    def horizon_amplitude(self) -> Array:
        return (
            self.boundary_amplitudes.horizon_ingoing
            + self.boundary_amplitudes.horizon_outgoing
        )

    @property
    def infinity_amplitude(self) -> Array:
        return (
            self.boundary_amplitudes.infinity_ingoing
            + self.boundary_amplitudes.infinity_outgoing
        )

    @property
    def finite(self) -> Array:
        return self.residual_evidence.finite

    @property
    def converged(self) -> Array:
        return self.residual_evidence.converged

    @property
    def physically_valid(self) -> Array:
        return self.residual_evidence.physically_valid

    @property
    def qualified(self) -> Array:
        return self.residual_evidence.qualified

    @property
    def valid(self) -> Array:
        return self.qualified

    @property
    def derivative_valid(self) -> Array:
        return self.residual_evidence.derivative_valid

    @property
    def status(self) -> Array:
        return self.residual_evidence.status


def schwarzschild_regge_wheeler_potential(
    radius: ArrayLike,
    mass: ArrayLike,
    ell: int,
    /,
    *,
    spin_weight: int = -2,
) -> Array:
    """Regge--Wheeler family ``f[l(l+1)/r^2 + 2M(1-s^2)/r^3]``."""
    ell_, spin_ = _potential_indices(ell, spin_weight)
    radius_ = jnp.asarray(radius)
    mass_ = jnp.asarray(mass)
    f = 1.0 - 2.0 * mass_ / radius_
    return f * (
        ell_ * (ell_ + 1) / radius_**2 + 2.0 * mass_ * (1 - spin_ * spin_) / radius_**3
    )


def schwarzschild_zerilli_potential(
    radius: ArrayLike,
    mass: ArrayLike,
    ell: int,
    /,
) -> Array:
    """Even-parity gravitational Zerilli potential in areal radius."""
    ell_, _ = _potential_indices(ell, -2)
    if ell_ < 2:
        raise ValueError("The Zerilli potential requires ell >= 2.")
    radius_ = jnp.asarray(radius)
    mass_ = jnp.asarray(mass)
    f = 1.0 - 2.0 * mass_ / radius_
    lam = 0.5 * (ell_ - 1) * (ell_ + 2)
    numerator = (
        lam * lam * (lam + 1.0) * radius_**3
        + 3.0 * lam * lam * mass_ * radius_**2
        + 9.0 * lam * mass_**2 * radius_
        + 9.0 * mass_**3
    )
    denominator = radius_**3 * (lam * radius_ + 3.0 * mass_) ** 2
    return 2.0 * f * numerator / denominator


def kerr_teukolsky_radial_coefficients(
    plan: KerrTeukolskyRadialPlan,
    radius: ArrayLike,
    frequency: ArrayLike,
    separation_constant: ArrayLike,
    /,
) -> KerrTeukolskyRadialCoefficients:
    """Evaluate exact untransformed Teukolsky radial coefficients.

    ``separation_constant`` is the angular ``A`` from :mod:`._spheroidal`; the
    radial equation uses ``lambda = A + (a omega)^2 - 2 a m omega``.
    """
    if not isinstance(plan, KerrTeukolskyRadialPlan):
        raise TypeError("plan must be a KerrTeukolskyRadialPlan.")
    radius_ = jnp.asarray(radius)
    omega = _complex_scalar(frequency, "frequency")
    angular = _complex_scalar(separation_constant, "separation_constant")
    mass = plan.mass
    spin = plan.spin
    delta = radius_**2 - 2.0 * mass * radius_ + spin**2
    delta_prime = 2.0 * (radius_ - mass)
    k = (radius_**2 + spin**2) * omega - spin * plan.mode.m
    c = spin * omega
    separation_lambda = angular + c * c - 2.0 * plan.mode.m * c
    first = (plan.mode.spin_weight + 1) * delta_prime
    zeroth = (
        (k * k - 2.0j * plan.mode.spin_weight * (radius_ - mass) * k) / delta
        + 4.0j * plan.mode.spin_weight * omega * radius_
        - separation_lambda
    )
    finite = (
        jnp.all(jnp.isfinite(jnp.real(delta)))
        & jnp.all(jnp.isfinite(jnp.real(k)))
        & jnp.all(jnp.isfinite(jnp.imag(k)))
        & jnp.all(jnp.isfinite(jnp.real(zeroth)))
        & jnp.all(jnp.isfinite(jnp.imag(zeroth)))
    )
    domain_valid = jnp.all(radius_ > plan.outer_horizon_radius)
    return KerrTeukolskyRadialCoefficients(
        delta,
        k,
        delta,
        first,
        zeroth,
        separation_lambda,
        finite,
        domain_valid,
    )


def _schwarzschild_potential_over_f(
    plan: SchwarzschildRadialPlan,
    radius: Array,
    /,
) -> Array:
    if plan.mode.sector in ("polar", "zerilli"):
        ell = plan.mode.ell
        lam = 0.5 * (ell - 1) * (ell + 2)
        numerator = (
            lam * lam * (lam + 1.0) * radius**3
            + 3.0 * lam * lam * plan.mass * radius**2
            + 9.0 * lam * plan.mass**2 * radius
            + 9.0 * plan.mass**3
        )
        denominator = radius**3 * (lam * radius + 3.0 * plan.mass) ** 2
        return 2.0 * numerator / denominator
    ell = plan.mode.ell
    spin_weight = plan.mode.spin_weight
    return (
        ell * (ell + 1) / radius**2
        + 2.0 * plan.mass * (1 - spin_weight * spin_weight) / radius**3
    )


def _schwarzschild_infinity_potential_series(
    plan: SchwarzschildRadialPlan,
    dtype,
    /,
) -> tuple[Array, ...]:
    count = plan.infinity_asymptotic_order
    zero = jnp.asarray(0.0, dtype=dtype)
    if plan.mode.sector not in ("polar", "zerilli"):
        weights = [zero for _ in range(count)]
        weights[0] = jnp.asarray(
            plan.mode.ell * (plan.mode.ell + 1),
            dtype=dtype,
        )
        if count > 1:
            weights[1] = (2.0 * plan.mass * (1 - plan.mode.spin_weight**2)).astype(dtype)
        return tuple(weights)

    lam = 0.5 * (plan.mode.ell - 1) * (plan.mode.ell + 2)
    numerator = (
        jnp.asarray(2.0 * lam * lam * (lam + 1.0), dtype=dtype),
        jnp.asarray(6.0 * lam * lam, dtype=dtype) * plan.mass,
        jnp.asarray(18.0 * lam, dtype=dtype) * plan.mass**2,
        jnp.asarray(18.0, dtype=dtype) * plan.mass**3,
    )
    weights = []
    for order in range(count):
        source = numerator[order] if order < len(numerator) else zero
        previous = weights[order - 1] if order >= 1 else zero
        second_previous = weights[order - 2] if order >= 2 else zero
        weight = (
            source
            - 6.0 * lam * plan.mass * previous
            - 9.0 * plan.mass**2 * second_previous
        ) / (lam * lam)
        weights.append(weight)
    return tuple(weights)


def _schwarzschild_tortoise_radius(
    plan: SchwarzschildRadialPlan,
    radius: Array,
    /,
) -> Array:
    return radius + 2.0 * plan.mass * jnp.log(
        radius / plan.horizon_radius - 1.0
    )


def _schwarzschild_horizon_series(
    plan: SchwarzschildRadialPlan,
    omega: ArrayLike,
    wave_sign: int,
    /,
) -> tuple[Array, Array]:
    if not isinstance(plan, SchwarzschildRadialPlan):
        raise TypeError("plan must be a SchwarzschildRadialPlan.")
    omega = _complex_scalar(omega, "omega")
    if wave_sign not in (-1, 1):
        raise ValueError("wave_sign must be -1 or 1.")
    inner_radius = plan.radial_nodes[0]
    inner_f = 1.0 - 2.0 * plan.mass / inner_radius
    potential_factor = _schwarzschild_potential_over_f(
        plan,
        plan.horizon_radius,
    )
    coefficient = (
        2.0
        * plan.mass
        * potential_factor
        / (1.0 + 4.0j * wave_sign * plan.mass * omega)
    )
    series = 1.0 + coefficient * (inner_radius - plan.horizon_radius)
    logarithmic_derivative = (
        wave_sign * 1.0j * omega + inner_f * coefficient / series
    )
    value = (
        jnp.exp(
            wave_sign
            * 1.0j
            * omega
            * _schwarzschild_tortoise_radius(plan, inner_radius)
        )
        * series
    )
    return value, logarithmic_derivative


def _schwarzschild_infinity_series(
    plan: SchwarzschildRadialPlan,
    omega: ArrayLike,
    wave_sign: int,
    /,
) -> tuple[Array, Array]:
    """Return the unit-amplitude infinity series value and tortoise log derivative."""
    if not isinstance(plan, SchwarzschildRadialPlan):
        raise TypeError("plan must be a SchwarzschildRadialPlan.")
    omega = _complex_scalar(omega, "omega")
    if wave_sign not in (-1, 1):
        raise ValueError("wave_sign must be -1 or 1.")
    potential_weights = _schwarzschild_infinity_potential_series(
        plan,
        omega.dtype,
    )
    previous = jnp.asarray(0.0, dtype=omega.dtype)
    current = jnp.asarray(1.0, dtype=omega.dtype)
    coefficients = [current]
    for order in range(plan.infinity_asymptotic_order):
        potential_sum = sum(
            potential_weights[degree] * coefficients[order - degree]
            for degree in range(order + 1)
        )
        numerator = (
            order * (order + 1) * current
            - 2.0 * plan.mass * (order * order - 1) * previous
            - potential_sum
        )
        following = numerator / (
            2.0j * wave_sign * omega * (order + 1)
        )
        coefficients.append(following)
        previous, current = current, following
    series_coefficients = jnp.stack(coefficients)
    orders = jnp.arange(
        plan.infinity_asymptotic_order + 1,
        dtype=plan.radial_nodes.dtype,
    )
    outer_radius = plan.radial_nodes[-1]
    inverse_powers = (1.0 / outer_radius) ** orders
    series = contract("n,n->", series_coefficients, inverse_powers)
    series_derivative = contract(
        "n,n,n->",
        -orders,
        series_coefficients,
        inverse_powers / outer_radius,
    )
    outer_f = 1.0 - 2.0 * plan.mass / outer_radius
    logarithmic_derivative = (
        wave_sign * 1.0j * omega + outer_f * series_derivative / series
    )
    value = (
        jnp.exp(
            wave_sign
            * 1.0j
            * omega
            * _schwarzschild_tortoise_radius(plan, outer_radius)
        )
        * series
    )
    return value, logarithmic_derivative


def _schwarzschild_boundary_log_derivatives(
    plan: SchwarzschildRadialPlan,
    omega: Array,
    /,
) -> tuple[Array, Array]:
    horizon_sign = -1 if plan.boundary.horizon == "ingoing" else 1
    infinity_sign = 1 if plan.boundary.infinity == "outgoing" else -1
    _, horizon_q = _schwarzschild_horizon_series(
        plan,
        omega,
        horizon_sign,
    )
    _, infinity_q = _schwarzschild_infinity_series(
        plan,
        omega,
        infinity_sign,
    )
    return horizon_q, infinity_q


def schwarzschild_radial_asymptotics(
    plan: SchwarzschildRadialPlan,
    frequency: ArrayLike,
    /,
) -> RadialAsymptoticEvidence:
    if not isinstance(plan, SchwarzschildRadialPlan):
        raise TypeError("plan must be a SchwarzschildRadialPlan.")
    omega = _complex_scalar(frequency, "frequency")
    horizon_q, infinity_q = _schwarzschild_boundary_log_derivatives(
        plan,
        omega,
    )
    radius = plan.radial_nodes
    potential = plan.potential(radius)
    frequency_scale = jnp.maximum(
        jnp.abs(omega) ** 2,
        jnp.finfo(radius.dtype).eps / plan.mass**2,
    )
    horizon_potential = jnp.abs(potential[0]) / frequency_scale
    infinity_potential = jnp.abs(potential[-1]) / frequency_scale
    horizon_ratio = (radius[0] - plan.horizon_radius) / plan.mass
    infinity_ratio = plan.mass / radius[-1]
    horizon_defect = jnp.maximum(horizon_ratio, horizon_potential)
    infinity_defect = jnp.maximum(infinity_ratio, infinity_potential)
    finite = jnp.all(
        jnp.isfinite(
            jnp.asarray(
                (
                    jnp.real(horizon_q),
                    jnp.imag(horizon_q),
                    jnp.real(infinity_q),
                    jnp.imag(infinity_q),
                    horizon_defect,
                    infinity_defect,
                )
            )
        )
    )
    wave_valid = jnp.abs(omega) * plan.mass > plan.wave_number_tolerance
    physically_valid = wave_valid
    qualified = (
        finite
        & physically_valid
        & (horizon_defect <= plan.asymptotic_tolerance)
        & (infinity_defect <= plan.asymptotic_tolerance)
    )
    return RadialAsymptoticEvidence(
        omega,
        omega,
        horizon_q,
        infinity_q,
        horizon_ratio,
        infinity_ratio,
        horizon_potential,
        infinity_potential,
        horizon_defect,
        infinity_defect,
        finite,
        physically_valid,
        qualified,
        qualified,
    )


def kerr_teukolsky_radial_asymptotics(
    plan: KerrTeukolskyRadialPlan,
    frequency: ArrayLike,
    /,
) -> RadialAsymptoticEvidence:
    if not isinstance(plan, KerrTeukolskyRadialPlan):
        raise TypeError("plan must be a KerrTeukolskyRadialPlan.")
    omega = _complex_scalar(frequency, "frequency")
    horizon_wave_number = omega - plan.mode.m * plan.horizon_angular_velocity
    radius = plan.radial_nodes
    delta = radius**2 - 2.0 * plan.mass * radius + plan.spin**2
    sigma = radius**2 + plan.spin**2
    radial_speed = delta / sigma
    delta_prime = 2.0 * (radius - plan.mass)
    if plan.boundary.horizon == "ingoing":
        horizon_q = (
            -1.0j * horizon_wave_number
            - plan.mode.spin_weight * delta_prime[0] / sigma[0]
        )
    else:
        horizon_q = 1.0j * horizon_wave_number
    if plan.boundary.infinity == "outgoing":
        infinity_q = (
            1.0j * omega - (2 * plan.mode.spin_weight + 1) * radial_speed[-1] / radius[-1]
        )
    else:
        infinity_q = -1.0j * omega - radial_speed[-1] / radius[-1]
    horizon_ratio = (radius[0] - plan.outer_horizon_radius) / plan.mass
    infinity_ratio = plan.mass / radius[-1]
    wave_scale = jnp.maximum(jnp.abs(omega) * radius[-1], plan.wave_number_tolerance)
    infinity_wave_defect = 1.0 / wave_scale
    horizon_defect = horizon_ratio
    infinity_defect = jnp.maximum(infinity_ratio, infinity_wave_defect)
    zero = jnp.asarray(0.0, dtype=radius.dtype)
    finite = jnp.all(
        jnp.isfinite(
            jnp.asarray(
                (
                    jnp.real(horizon_q),
                    jnp.imag(horizon_q),
                    jnp.real(infinity_q),
                    jnp.imag(infinity_q),
                    horizon_defect,
                    infinity_defect,
                )
            )
        )
    )
    wave_valid = (jnp.abs(omega) * plan.mass > plan.wave_number_tolerance) & (
        jnp.abs(horizon_wave_number) * plan.mass > plan.wave_number_tolerance
    )
    physically_valid = wave_valid & (plan.surface_gravity > 0.0)
    qualified = (
        finite
        & physically_valid
        & (horizon_defect <= plan.asymptotic_tolerance)
        & (infinity_defect <= plan.asymptotic_tolerance)
    )
    return RadialAsymptoticEvidence(
        horizon_wave_number,
        omega,
        horizon_q,
        infinity_q,
        horizon_ratio,
        infinity_ratio,
        zero,
        infinity_wave_defect,
        horizon_defect,
        infinity_defect,
        finite,
        physically_valid,
        qualified,
        qualified,
    )


def evaluate_schwarzschild_radial(
    plan: SchwarzschildRadialPlan,
    frequency: ArrayLike,
    separation_constant: ArrayLike | None = None,
    /,
) -> SchwarzschildRadialResult:
    """Evaluate two-sided Regge--Wheeler/Zerilli matching at one frequency."""
    if not isinstance(plan, SchwarzschildRadialPlan):
        raise TypeError("plan must be a SchwarzschildRadialPlan.")
    omega = _complex_scalar(frequency, "frequency")
    spherical = plan.mode.ell * (plan.mode.ell + 1) - plan.mode.spin_weight * (
        plan.mode.spin_weight + 1
    )
    angular = (
        jnp.asarray(spherical, dtype=omega.dtype)
        if separation_constant is None
        else _complex_scalar(separation_constant, "separation_constant")
    )
    asymptotic = schwarzschild_radial_asymptotics(plan, omega)
    nodes = plan.radial_nodes
    coordinates = plan.coordinate_nodes

    horizon_sign = -1 if plan.boundary.horizon == "ingoing" else 1
    infinity_sign = 1 if plan.boundary.infinity == "outgoing" else -1
    left_states = _integrate_schwarzschild_horizon_riccati(
        plan,
        omega,
        horizon_sign,
    )
    infinity_value, infinity_q = _schwarzschild_infinity_series(
        plan,
        omega,
        infinity_sign,
    )
    infinity_state = jnp.stack((jnp.log(infinity_value), infinity_q))
    reverse_states = _integrate_schwarzschild_riccati(
        plan,
        omega,
        coordinates[::-1],
        infinity_state,
    )
    right_states = reverse_states[::-1]
    solution, logarithmic_derivative, matching_residual = _matched_riccati_profile(
        left_states,
        right_states,
        plan.match_index,
        plan.mass,
    )
    radial_jacobian = nodes - plan.horizon_radius
    first_coordinate = contract("ij,j->i", plan.collocation.first_derivative, solution)
    second_coordinate = contract("ij,j->i", plan.collocation.second_derivative, solution)
    first = first_coordinate / radial_jacobian
    second = (second_coordinate - first_coordinate) / radial_jacobian**2
    f = 1.0 - 2.0 * plan.mass / nodes
    f_prime = 2.0 * plan.mass / nodes**2
    potential = plan.potential(nodes)
    differential = (
        f * f * second + f * f_prime * first + (omega * omega - potential) * solution
    )
    scale_terms = (
        jnp.abs(f * f * second)
        + jnp.abs(f * f_prime * first)
        + jnp.abs((omega * omega - potential) * solution)
    )
    angular_scale = jnp.maximum(jnp.abs(angular), 1.0)
    angular_consistent = jnp.abs(angular - spherical) <= (
        jnp.sqrt(jnp.finfo(nodes.dtype).eps) * angular_scale
    )
    amplitudes = _selected_boundary_amplitudes(plan.boundary, solution)
    residual_evidence = _radial_residual_evidence(
        matching_residual,
        differential,
        scale_terms,
        asymptotic,
        _profile_finite(solution, logarithmic_derivative),
        angular_consistent,
        plan.match_index,
        plan.matching_tolerance,
        plan.residual_tolerance,
    )
    return SchwarzschildRadialResult(
        omega,
        angular,
        nodes,
        solution,
        logarithmic_derivative,
        amplitudes,
        asymptotic,
        residual_evidence,
        plan.mode.mode_id,
        plan.plan_id,
    )


def evaluate_kerr_teukolsky_radial(
    plan: KerrTeukolskyRadialPlan,
    frequency: ArrayLike,
    separation_constant: ArrayLike,
    /,
) -> KerrTeukolskyRadialResult:
    """Evaluate two-sided untransformed Teukolsky radial matching."""
    if not isinstance(plan, KerrTeukolskyRadialPlan):
        raise TypeError("plan must be a KerrTeukolskyRadialPlan.")
    omega = _complex_scalar(frequency, "frequency")
    angular = _complex_scalar(separation_constant, "separation_constant")
    nodes = plan.radial_nodes
    coefficients = kerr_teukolsky_radial_coefficients(plan, nodes, omega, angular)
    asymptotic = kerr_teukolsky_radial_asymptotics(plan, omega)
    coordinates = plan.coordinate_nodes

    def state_derivative(coordinate, state):
        radius = plan.outer_horizon_radius + plan.mass * jnp.exp(coordinate)
        radial_jacobian = radius - plan.outer_horizon_radius
        values = kerr_teukolsky_radial_coefficients(plan, radius, omega, angular)
        sigma = radius * radius + plan.spin * plan.spin
        radial_speed = values.delta / sigma
        delta_prime = 2.0 * (radius - plan.mass)
        speed_prime = (delta_prime * sigma - 2.0 * radius * values.delta) / sigma**2
        value, tortoise_derivative = state
        value_derivative_radius = tortoise_derivative / radial_speed
        tortoise_derivative_radius = (
            (speed_prime / radial_speed - values.first_derivative / values.delta)
            * tortoise_derivative
            - values.zeroth_derivative * radial_speed / values.delta * value
        )
        return radial_jacobian * jnp.stack(
            (value_derivative_radius, tortoise_derivative_radius)
        )

    horizon_state = jnp.stack(
        (
            jnp.asarray(1.0, dtype=omega.dtype),
            asymptotic.horizon_log_derivative,
        )
    )
    infinity_state = jnp.stack(
        (
            jnp.asarray(1.0, dtype=omega.dtype),
            asymptotic.infinity_log_derivative,
        )
    )
    left_states, left_scales = _integrate_radial_state(
        coordinates,
        horizon_state,
        state_derivative,
        plan.mass,
    )
    reverse_states, reverse_scales = _integrate_radial_state(
        coordinates[::-1],
        infinity_state,
        state_derivative,
        plan.mass,
    )
    right_states = reverse_states[::-1]
    right_scales = reverse_scales[::-1]
    solution, logarithmic_derivative, matching_residual = _matched_profile(
        left_states,
        left_scales,
        right_states,
        right_scales,
        plan.match_index,
        plan.mass,
    )
    radial_jacobian = nodes - plan.outer_horizon_radius
    first_coordinate = contract("ij,j->i", plan.collocation.first_derivative, solution)
    second_coordinate = contract("ij,j->i", plan.collocation.second_derivative, solution)
    first = first_coordinate / radial_jacobian
    second = (second_coordinate - first_coordinate) / radial_jacobian**2
    differential = (
        coefficients.second_derivative * second
        + coefficients.first_derivative * first
        + coefficients.zeroth_derivative * solution
    )
    scale_terms = (
        jnp.abs(coefficients.second_derivative * second)
        + jnp.abs(coefficients.first_derivative * first)
        + jnp.abs(coefficients.zeroth_derivative * solution)
    )
    amplitudes = _selected_boundary_amplitudes(plan.boundary, solution)
    residual_evidence = _radial_residual_evidence(
        matching_residual,
        differential,
        scale_terms,
        asymptotic,
        _profile_finite(solution, logarithmic_derivative),
        coefficients.domain_valid & coefficients.finite,
        plan.match_index,
        plan.matching_tolerance,
        plan.residual_tolerance,
    )
    return KerrTeukolskyRadialResult(
        omega,
        angular,
        coefficients.separation_lambda,
        nodes,
        solution,
        logarithmic_derivative,
        coefficients,
        amplitudes,
        asymptotic,
        residual_evidence,
        plan.mode.mode_id,
        plan.plan_id,
    )


def schwarzschild_outgoing_residual(
    plan: SchwarzschildRadialPlan,
    frequency: ArrayLike,
    separation_constant: ArrayLike | None = None,
    /,
) -> Array:
    """Complex horizon-to-infinity logarithmic-derivative mismatch."""
    return evaluate_schwarzschild_radial(
        plan,
        frequency,
        separation_constant,
    ).residual


def kerr_teukolsky_outgoing_residual(
    plan: KerrTeukolskyRadialPlan,
    frequency: ArrayLike,
    separation_constant: ArrayLike,
    /,
) -> Array:
    """Complex Teukolsky horizon-to-infinity logarithmic-derivative mismatch."""
    return evaluate_kerr_teukolsky_radial(
        plan,
        frequency,
        separation_constant,
    ).residual


def _integrate_schwarzschild_riccati(
    plan: SchwarzschildRadialPlan,
    omega: Array,
    coordinates: Array,
    initial_state: Array,
    /,
) -> Array:
    def derivative(coordinate, state):
        radius = plan.horizon_radius + plan.mass * jnp.exp(coordinate)
        radial_jacobian = radius - plan.horizon_radius
        f = 1.0 - 2.0 * plan.mass / radius
        factor = radial_jacobian / f
        logarithm, logarithmic_derivative = state
        del logarithm
        potential = plan.potential(radius)
        return jnp.stack(
            (
                factor * logarithmic_derivative,
                factor
                * (
                    potential
                    - omega * omega
                    - logarithmic_derivative * logarithmic_derivative
                ),
            )
        )

    def interval_step(state, interval):
        start, end = interval
        width = (end - start) / plan.integration_substeps

        def substep(_, carry):
            coordinate, current = carry
            k1 = derivative(coordinate, current)
            k2 = derivative(
                coordinate + 0.5 * width,
                current + 0.5 * width * k1,
            )
            k3 = derivative(
                coordinate + 0.5 * width,
                current + 0.5 * width * k2,
            )
            k4 = derivative(coordinate + width, current + width * k3)
            following = current + width * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            return coordinate + width, following

        _, following_state = jax.lax.fori_loop(
            0,
            plan.integration_substeps,
            substep,
            (start, state),
        )
        return following_state, following_state

    intervals = jnp.stack((coordinates[:-1], coordinates[1:]), axis=-1)
    _, history = jax.lax.scan(interval_step, initial_state, intervals)
    return jnp.concatenate((initial_state[None, :], history), axis=0)


def _integrate_schwarzschild_horizon_riccati(
    plan: SchwarzschildRadialPlan,
    frequency: ArrayLike,
    wave_sign: int,
    /,
) -> Array:
    """Integrate one unit-amplitude horizon wave outward on the bounded plan."""
    if not isinstance(plan, SchwarzschildRadialPlan):
        raise TypeError("plan must be a SchwarzschildRadialPlan.")
    omega = _complex_scalar(frequency, "frequency")
    horizon_value, horizon_q = _schwarzschild_horizon_series(
        plan,
        omega,
        wave_sign,
    )
    initial_state = jnp.stack((jnp.log(horizon_value), horizon_q))
    return _integrate_schwarzschild_riccati(
        plan,
        omega,
        plan.coordinate_nodes,
        initial_state,
    )


def _matched_riccati_profile(
    left_states: Array,
    right_states: Array,
    match_index: int,
    mass: Array,
    /,
) -> tuple[Array, Array, Array]:
    left_logarithm = left_states[:, 0]
    right_logarithm = (
        right_states[:, 0] + left_logarithm[match_index] - right_states[match_index, 0]
    )
    indices = jnp.arange(left_states.shape[0])
    use_left = indices <= match_index
    logarithm = jnp.where(use_left, left_logarithm, right_logarithm)
    logarithmic_derivative = jnp.where(
        use_left,
        left_states[:, 1],
        right_states[:, 1],
    )
    solution = jnp.exp(logarithm - jnp.max(jnp.real(logarithm)))
    left_match = left_states[match_index, 1]
    right_match = right_states[match_index, 1]
    denominator = jnp.sqrt(
        (1.0 + jnp.abs(mass * left_match) ** 2) * (1.0 + jnp.abs(mass * right_match) ** 2)
    )
    matching_residual = mass * (left_match - right_match) / denominator
    return solution, logarithmic_derivative, matching_residual


def _integrate_radial_state(
    coordinates,
    initial_state,
    state_derivative,
    mass,
    /,
):
    coordinates_ = jnp.asarray(coordinates)
    state_ = jnp.asarray(initial_state)
    state_scale = jnp.maximum(
        jnp.maximum(jnp.abs(state_[0]), jnp.abs(mass * state_[1])),
        jnp.finfo(state_.real.dtype).tiny,
    )
    initial = (state_ / state_scale, jnp.log(state_scale))

    def step(carry, interval):
        state, logarithmic_scale = carry
        start, end = interval
        width = end - start
        k1 = state_derivative(start, state)
        k2 = state_derivative(start + 0.5 * width, state + 0.5 * width * k1)
        k3 = state_derivative(start + 0.5 * width, state + 0.5 * width * k2)
        k4 = state_derivative(end, state + width * k3)
        candidate = state + width * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(candidate[0]), jnp.abs(mass * candidate[1])),
            jnp.finfo(candidate.real.dtype).tiny,
        )
        next_state = candidate / scale
        next_logarithmic_scale = logarithmic_scale + jnp.log(scale)
        return (
            (next_state, next_logarithmic_scale),
            (next_state, next_logarithmic_scale),
        )

    intervals = jnp.stack((coordinates_[:-1], coordinates_[1:]), axis=-1)
    _, history = jax.lax.scan(step, initial, intervals)
    return (
        jnp.concatenate((initial[0][None, :], history[0]), axis=0),
        jnp.concatenate((initial[1][None], history[1])),
    )


def _matched_profile(
    left_states,
    left_scales,
    right_states,
    right_scales,
    match_index,
    mass,
    /,
):
    left_match = left_states[match_index]
    right_match = right_states[match_index]
    left_scaled = jnp.stack((left_match[0], mass * left_match[1]))
    right_scaled = jnp.stack((right_match[0], mass * right_match[1]))
    right_norm = contract("i,i->", jnp.conj(right_scaled), right_scaled)
    alignment = contract("i,i->", jnp.conj(right_scaled), left_scaled) / jnp.where(
        jnp.abs(right_norm) > 0.0, right_norm, 1.0
    )
    aligned_right_states = right_states * alignment
    aligned_right_scales = (
        right_scales + left_scales[match_index] - right_scales[match_index]
    )
    indices = jnp.arange(left_states.shape[0])
    use_left = indices <= match_index
    states = jnp.where(use_left[:, None], left_states, aligned_right_states)
    scales = jnp.where(use_left, left_scales, aligned_right_scales)
    values = states[:, 0]
    magnitudes = jnp.abs(values)
    safe_magnitudes = jnp.maximum(
        magnitudes,
        jnp.finfo(magnitudes.dtype).tiny,
    )
    phases = values / safe_magnitudes
    logarithmic_magnitudes = jnp.log(safe_magnitudes) + scales
    normalization = jnp.max(logarithmic_magnitudes)
    solution = phases * jnp.exp(logarithmic_magnitudes - normalization)
    safe_values = jnp.where(magnitudes > 0.0, values, 1.0)
    logarithmic_derivative = states[:, 1] / safe_values
    logarithmic_derivative = jnp.where(
        magnitudes > 0.0,
        logarithmic_derivative,
        jnp.asarray(jnp.inf + 0.0j, dtype=states.dtype),
    )
    wronskian = left_match[0] * right_match[1] - left_match[1] * right_match[0]
    denominator = jnp.sqrt(
        jnp.maximum(
            jnp.real(contract("i,i->", jnp.conj(left_scaled), left_scaled)),
            0.0,
        )
        * jnp.maximum(jnp.real(right_norm), 0.0)
    )
    matching_residual = (
        mass
        * wronskian
        / jnp.maximum(
            denominator,
            jnp.finfo(denominator.dtype).tiny,
        )
    )
    return solution, logarithmic_derivative, matching_residual


def _selected_boundary_amplitudes(boundary, solution, /):
    zero = jnp.asarray(0.0, dtype=solution.dtype)
    horizon_ingoing = solution[0] if boundary.horizon == "ingoing" else zero
    horizon_outgoing = solution[0] if boundary.horizon == "outgoing" else zero
    infinity_ingoing = solution[-1] if boundary.infinity == "ingoing" else zero
    infinity_outgoing = solution[-1] if boundary.infinity == "outgoing" else zero
    return RadialBoundaryAmplitudes(
        horizon_ingoing,
        horizon_outgoing,
        infinity_ingoing,
        infinity_outgoing,
    )


def _profile_finite(solution, logarithmic_derivative, /):
    return (
        jnp.all(jnp.isfinite(jnp.real(solution)))
        & jnp.all(jnp.isfinite(jnp.imag(solution)))
        & jnp.all(jnp.isfinite(jnp.real(logarithmic_derivative)))
        & jnp.all(jnp.isfinite(jnp.imag(logarithmic_derivative)))
    )


def _radial_residual_evidence(
    matching_residual,
    differential_residual,
    scale_terms,
    asymptotic,
    profile_finite,
    domain_valid,
    match_index,
    matching_tolerance,
    residual_tolerance,
    /,
):
    size = differential_residual.size
    indices = jnp.arange(size)
    interior = (indices > 0) & (indices < size - 1)
    residual_values = jnp.where(interior, jnp.abs(differential_residual), 0.0)
    scale_values = jnp.where(interior, scale_terms, 0.0)
    residual_norm = jnp.max(residual_values)
    relative_residual = residual_norm / jnp.maximum(
        jnp.max(scale_values),
        jnp.finfo(residual_norm.dtype).tiny,
    )
    finite = (
        asymptotic.finite
        & jnp.asarray(profile_finite, dtype=bool)
        & jnp.isfinite(jnp.real(matching_residual))
        & jnp.isfinite(jnp.imag(matching_residual))
        & jnp.all(jnp.isfinite(jnp.real(differential_residual)))
        & jnp.all(jnp.isfinite(jnp.imag(differential_residual)))
        & jnp.isfinite(relative_residual)
    )
    physically_valid = asymptotic.physically_valid & jnp.asarray(
        domain_valid,
        dtype=bool,
    )
    converged = jnp.abs(matching_residual) <= matching_tolerance
    residual_valid = relative_residual <= residual_tolerance
    qualified = (
        finite & converged & physically_valid & asymptotic.qualified & residual_valid
    )
    derivative_valid = finite & physically_valid & asymptotic.derivative_valid
    status = jnp.where(
        ~physically_valid,
        int(PerturbationStatus.INVALID_DOMAIN),
        jnp.where(
            ~finite,
            int(PerturbationStatus.NONFINITE),
            jnp.where(
                ~asymptotic.qualified,
                int(PerturbationStatus.ASYMPTOTIC_MISMATCH),
                jnp.where(
                    ~converged,
                    int(PerturbationStatus.RADIAL_NONCONVERGENCE),
                    jnp.where(
                        ~residual_valid,
                        int(PerturbationStatus.RESIDUAL_TOLERANCE_NOT_MET),
                        int(PerturbationStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    return RadialResidualEvidence(
        matching_residual,
        differential_residual,
        residual_norm,
        relative_residual,
        finite,
        converged,
        physically_valid,
        qualified,
        derivative_valid,
        status,
        int(size),
        int(match_index),
    )


def _complex_scalar(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return array.astype(jnp.result_type(array, 1.0j))


def _positive_scalar_host(value: ArrayLike, name: str, /) -> float:
    scalar = _finite_scalar_host(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return scalar


def _finite_scalar_host(value: ArrayLike, name: str, /) -> float:
    array = np.asarray(value, dtype=float)
    if array.shape != () or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite scalar.")
    return float(array)


def _collocation_capacity(
    node_count: int,
    maximum_dimension: int,
    /,
) -> tuple[int, int]:
    if (
        isinstance(node_count, bool)
        or not isinstance(node_count, Integral)
        or isinstance(maximum_dimension, bool)
        or not isinstance(maximum_dimension, Integral)
    ):
        raise TypeError("Radial node counts and capacities must be integers.")
    count, maximum = int(node_count), int(maximum_dimension)
    if count < 9 or maximum < count:
        raise ValueError("Radial node_count must be at least 9 and within capacity.")
    return count, maximum


def _schwarzschild_integration_capacity(
    integration_substeps: int,
    infinity_asymptotic_order: int,
    /,
) -> tuple[int, int]:
    if (
        isinstance(integration_substeps, bool)
        or not isinstance(integration_substeps, Integral)
        or isinstance(infinity_asymptotic_order, bool)
        or not isinstance(infinity_asymptotic_order, Integral)
    ):
        raise TypeError(
            "Schwarzschild integration_substeps and infinity_asymptotic_order "
            "must be integers."
        )
    substeps = int(integration_substeps)
    order = int(infinity_asymptotic_order)
    if not 1 <= substeps <= 4096:
        raise ValueError("integration_substeps must lie in [1, 4096].")
    if not 1 <= order <= 64:
        raise ValueError("infinity_asymptotic_order must lie in [1, 64].")
    return substeps, order


def _radial_tolerances(*values: float) -> tuple[float, float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 4:
        raise ValueError("Exactly four radial tolerances are required.")
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError("Radial tolerances must be finite and positive.")
    return result[0], result[1], result[2], result[3]


def _validate_boundary(
    boundary: RadialBoundaryCondition,
    mode: SeparatedMode,
    /,
) -> None:
    if not isinstance(boundary, RadialBoundaryCondition):
        raise TypeError("boundary must be a RadialBoundaryCondition.")
    if boundary.convention_id != mode.convention_id:
        raise ValueError("Radial boundary and separated mode conventions must match.")


def _validate_schwarzschild_sector(mode: SeparatedMode, /) -> None:
    supported = (
        "scalar",
        "electromagnetic",
        "axial",
        "regge-wheeler",
        "polar",
        "zerilli",
        "teukolsky",
    )
    if mode.sector not in supported:
        raise ValueError("Unsupported Schwarzschild radial sector.")
    if mode.sector in ("scalar", "teukolsky") and mode.spin_weight != 0:
        raise ValueError(
            "The Schwarzschild scalar master equation requires spin_weight=0."
        )
    if mode.sector == "electromagnetic" and abs(mode.spin_weight) != 1:
        raise ValueError(
            "The Schwarzschild electromagnetic master equation requires "
            "abs(spin_weight)=1."
        )
    if (
        mode.sector in ("axial", "regge-wheeler", "polar", "zerilli")
        and abs(mode.spin_weight) != 2
    ):
        raise ValueError("Axial/polar Schwarzschild sectors require abs(spin_weight)=2.")
    if mode.sector in ("polar", "zerilli") and mode.ell < 2:
        raise ValueError("The polar Zerilli sector requires ell >= 2.")


def _potential_indices(ell: int, spin_weight: int, /) -> tuple[int, int]:
    if (
        isinstance(ell, bool)
        or not isinstance(ell, Integral)
        or isinstance(spin_weight, bool)
        or not isinstance(spin_weight, Integral)
    ):
        raise TypeError("Potential ell and spin_weight must be integers.")
    ell_, spin_ = int(ell), int(spin_weight)
    if spin_ not in (-2, -1, 0, 1, 2) or ell_ < abs(spin_):
        raise ValueError("Potential indices are outside the supported separated modes.")
    return ell_, spin_


__all__ = [
    "KerrTeukolskyRadialCoefficients",
    "KerrTeukolskyRadialPlan",
    "KerrTeukolskyRadialResult",
    "RadialAsymptoticEvidence",
    "RadialBoundaryAmplitudes",
    "RadialResidualEvidence",
    "SchwarzschildRadialPlan",
    "SchwarzschildRadialResult",
    "evaluate_kerr_teukolsky_radial",
    "evaluate_schwarzschild_radial",
    "kerr_teukolsky_outgoing_residual",
    "kerr_teukolsky_radial_asymptotics",
    "kerr_teukolsky_radial_coefficients",
    "schwarzschild_outgoing_residual",
    "schwarzschild_radial_asymptotics",
    "schwarzschild_regge_wheeler_potential",
    "schwarzschild_zerilli_potential",
]
