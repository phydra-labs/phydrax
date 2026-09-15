#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._chart import ChartTransition, CoordinateChart
from ._metric import LorentzianConvention, LorentzianMetric
from ._metric_domain import MetricDomainEvidence, MetricDomainStatus
from ._utils import _coordinates


def _scalar_parameter(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return array


def _checked_mass(mass: Array, /, *, owner: str) -> Array:
    return eqx.error_if(
        mass,
        ~jnp.isfinite(mass) | (mass <= 0.0),
        f"{owner} mass must be finite and positive.",
    )


def _checked_kerr_parameters(
    mass: Array,
    spin: Array,
    /,
    *,
    owner: str,
) -> tuple[Array, Array]:
    invalid = ~jnp.isfinite(mass) | ~jnp.isfinite(spin) | (mass <= 0.0)
    checked_mass = eqx.error_if(
        mass,
        invalid,
        f"{owner} requires finite M > 0 and finite signed spin a.",
    )
    return checked_mass, spin


def _checked_spherical_coordinates(
    coordinates: Array,
    /,
    *,
    owner: str,
) -> tuple[Array, Array]:
    _, radius, polar, _ = coordinates
    finite = jnp.all(jnp.isfinite(coordinates))
    radius = eqx.error_if(
        radius,
        ~finite | (radius <= 0.0) | ~jnp.isfinite(radius**2),
        f"{owner} requires finite coordinates and radius > 0.",
    )
    polar = eqx.error_if(
        polar,
        (polar <= 0.0) | (polar >= jnp.pi),
        f"{owner} excludes the spherical-coordinate axis (0 < theta < pi).",
    )
    return radius, polar


def _kerr_root_values(mass: Array, spin: Array, /) -> tuple[Array, Array, Array]:
    root = jnp.sqrt((mass - spin) * (mass + spin))
    outer = mass + root
    inner = spin * spin / outer
    return inner, outer, root


def _kerr_delta(mass: Array, spin: Array, radius: Array, /) -> Array:
    def horizon_branch(unused: None) -> Array:
        del unused
        inner, outer, root = _kerr_root_values(mass, spin)
        del root
        return (radius - outer) * (radius - inner)

    def polynomial_branch(_: None) -> Array:
        return radius * radius - 2.0 * mass * radius + spin * spin

    return jax.lax.cond(
        jnp.abs(spin) < mass,
        horizon_branch,
        polynomial_branch,
        operand=None,
    )


def _checked_kerr_coordinates(
    coordinates: Array,
    mass: Array,
    spin: Array,
    /,
    *,
    owner: str,
) -> tuple[Array, Array, Array, Array, Array]:
    mass, spin = _checked_kerr_parameters(mass, spin, owner=owner)
    radius, polar = _checked_spherical_coordinates(coordinates, owner=owner)
    cosine = jnp.cos(polar)
    sigma = radius * radius + spin * spin * cosine * cosine
    sigma = eqx.error_if(
        sigma,
        ~jnp.isfinite(sigma) | (sigma <= 0.0),
        f"{owner} excludes the Kerr ring singularity Sigma = 0.",
    )
    return mass, spin, radius, polar, sigma


class ExactMetricDomainStatus(IntEnum):
    """Provider-specific failure classification for exact black-hole charts."""

    VALID = int(MetricDomainStatus.VALID)
    NEAR_BOUNDARY = int(MetricDomainStatus.NEAR_BOUNDARY)
    OUTSIDE = int(MetricDomainStatus.OUTSIDE)
    NONFINITE = int(MetricDomainStatus.NONFINITE)
    REJECTED = int(MetricDomainStatus.REJECTED)
    AXIS = 5
    RING = 6
    BOYER_LINDQUIST_HORIZON = 7
    INVALID_PARAMETERS = 8


def _minimum_margin(*values: Array) -> Array:
    margin = values[0]
    for value in values[1:]:
        margin = jnp.minimum(margin, value)
    return margin


def _domain_id(kind: str, chart: CoordinateChart, /) -> str:
    return f"{kind}:{chart.name}:{','.join(chart.coordinates)}"


def _classified_domain_evidence(
    margin: Array,
    /,
    *,
    chart: CoordinateChart,
    domain_id: str,
    boundary_tolerance: ArrayLike,
    extra_valid: Array,
    nonfinite: Array,
    invalid_parameters: Array,
    outside: Array,
    axis: Array,
    ring: Array,
    boyer_lindquist_horizon: Array,
) -> MetricDomainEvidence:
    base = MetricDomainEvidence.from_margin(
        margin,
        chart=chart,
        domain_id=domain_id,
        boundary_tolerance=boundary_tolerance,
        extra_valid=extra_valid,
    )
    status = base.status
    status = jnp.where(
        outside,
        int(ExactMetricDomainStatus.OUTSIDE),
        status,
    )
    status = jnp.where(
        invalid_parameters,
        int(ExactMetricDomainStatus.INVALID_PARAMETERS),
        status,
    )
    status = jnp.where(
        boyer_lindquist_horizon,
        int(ExactMetricDomainStatus.BOYER_LINDQUIST_HORIZON),
        status,
    )
    status = jnp.where(ring, int(ExactMetricDomainStatus.RING), status)
    status = jnp.where(axis, int(ExactMetricDomainStatus.AXIS), status)
    status = jnp.where(
        nonfinite,
        int(ExactMetricDomainStatus.NONFINITE),
        status,
    )
    return MetricDomainEvidence(
        base.valid,
        base.near_boundary,
        base.margin,
        status,
        chart=chart,
        domain_id=domain_id,
    )


def ingoing_schwarzschild_domain_evidence(
    mass: ArrayLike,
    coordinates: ArrayLike,
    /,
    *,
    chart: CoordinateChart,
    boundary_tolerance: ArrayLike = 0.0,
) -> MetricDomainEvidence:
    """Classify the future-ingoing ``(v,r,theta,phi)`` chart domain."""

    mass_array = _scalar_parameter(mass, "Ingoing Schwarzschild mass")
    points = _coordinates(coordinates, 4)
    dtype = jnp.result_type(mass_array, points, 1.0)
    mass_array = mass_array.astype(dtype)
    points = points.astype(dtype)
    radius = points[..., 1]
    polar = points[..., 2]
    finite = (
        jnp.isfinite(mass_array)
        & jnp.all(jnp.isfinite(points), axis=-1)
        & jnp.isfinite(radius * radius)
    )
    invalid_parameters = ~jnp.isfinite(mass_array) | (mass_array <= 0.0)
    outside = (radius <= 0.0) | (polar < 0.0) | (polar > jnp.pi)
    axis = (polar == 0.0) | (polar == jnp.pi)
    margin = _minimum_margin(
        jnp.broadcast_to(mass_array, radius.shape),
        radius,
        polar,
        jnp.pi - polar,
    )
    margin = jnp.where(finite, margin, jnp.nan)
    valid = finite & ~invalid_parameters & ~outside & ~axis
    false = jnp.zeros_like(valid)
    return _classified_domain_evidence(
        margin,
        chart=chart,
        domain_id=_domain_id("ingoing-schwarzschild", chart),
        boundary_tolerance=boundary_tolerance,
        extra_valid=valid,
        nonfinite=~finite,
        invalid_parameters=invalid_parameters,
        outside=outside,
        axis=axis,
        ring=false,
        boyer_lindquist_horizon=false,
    )


def _kerr_domain_evidence(
    mass: ArrayLike,
    spin: ArrayLike,
    coordinates: ArrayLike,
    /,
    *,
    chart: CoordinateChart,
    boundary_tolerance: ArrayLike,
    boyer_lindquist: bool,
) -> MetricDomainEvidence:
    mass_array = _scalar_parameter(mass, "Kerr mass")
    spin_array = _scalar_parameter(spin, "Kerr spin")
    points = _coordinates(coordinates, 4)
    dtype = jnp.result_type(mass_array, spin_array, points, 1.0)
    mass_array = mass_array.astype(dtype)
    spin_array = spin_array.astype(dtype)
    points = points.astype(dtype)
    radius = points[..., 1]
    polar = points[..., 2]
    cosine = jnp.cos(polar)
    sigma = radius * radius + spin_array * spin_array * cosine * cosine
    parameters_valid = (
        jnp.isfinite(mass_array) & jnp.isfinite(spin_array) & (mass_array > 0.0)
    )
    has_horizons = parameters_valid & (jnp.abs(spin_array) <= mass_array)
    polynomial_delta = (
        radius * radius - 2.0 * mass_array * radius + spin_array * spin_array
    )
    delta = _kerr_delta(mass_array, spin_array, radius)
    finite = (
        jnp.isfinite(mass_array)
        & jnp.isfinite(spin_array)
        & jnp.all(jnp.isfinite(points), axis=-1)
        & jnp.isfinite(sigma)
        & jnp.isfinite(polynomial_delta)
    )
    invalid_parameters = ~parameters_valid
    outside = (radius <= 0.0) | (polar < 0.0) | (polar > jnp.pi)
    axis = (polar == 0.0) | (polar == jnp.pi)
    ring = (radius == 0.0) & (polar == 0.5 * jnp.pi)
    horizon = has_horizons & (delta == 0.0)
    chart_regular = ~horizon if boyer_lindquist else jnp.ones_like(horizon)
    ring_margin = jnp.where(ring, jnp.zeros_like(sigma), sigma)
    margins = (
        jnp.broadcast_to(mass_array, radius.shape),
        radius,
        polar,
        jnp.pi - polar,
        ring_margin,
    )
    if boyer_lindquist:
        margins = margins + (jnp.abs(delta),)
    margin = _minimum_margin(*margins)
    margin = jnp.where(finite, margin, jnp.nan)
    valid = finite & parameters_valid & ~outside & ~axis & ~ring & chart_regular
    return _classified_domain_evidence(
        margin,
        chart=chart,
        domain_id=_domain_id(
            "kerr-boyer-lindquist" if boyer_lindquist else "ingoing-kerr",
            chart,
        ),
        boundary_tolerance=boundary_tolerance,
        extra_valid=valid,
        nonfinite=~finite,
        invalid_parameters=invalid_parameters,
        outside=outside,
        axis=axis,
        ring=ring,
        boyer_lindquist_horizon=horizon & boyer_lindquist,
    )


def kerr_boyer_lindquist_domain_evidence(
    mass: ArrayLike,
    spin: ArrayLike,
    coordinates: ArrayLike,
    /,
    *,
    chart: CoordinateChart,
    boundary_tolerance: ArrayLike = 0.0,
) -> MetricDomainEvidence:
    """Classify Kerr's ``(t,r,theta,phi)`` Boyer-Lindquist domain."""

    return _kerr_domain_evidence(
        mass,
        spin,
        coordinates,
        chart=chart,
        boundary_tolerance=boundary_tolerance,
        boyer_lindquist=True,
    )


def ingoing_kerr_domain_evidence(
    mass: ArrayLike,
    spin: ArrayLike,
    coordinates: ArrayLike,
    /,
    *,
    chart: CoordinateChart,
    boundary_tolerance: ArrayLike = 0.0,
) -> MetricDomainEvidence:
    """Classify Kerr's future-ingoing ``(v,r,theta,phi_tilde)`` domain."""

    return _kerr_domain_evidence(
        mass,
        spin,
        coordinates,
        chart=chart,
        boundary_tolerance=boundary_tolerance,
        boyer_lindquist=False,
    )


class _IngoingSchwarzschildMetricMap(StrictModule):
    mass: Array
    convention: LorentzianConvention = eqx.field(static=True)

    def __init__(self, mass: ArrayLike, convention: LorentzianConvention, /):
        self.mass = _scalar_parameter(mass, "Ingoing Schwarzschild mass")
        self.convention = convention

    def __call__(self, coordinates: Array, /) -> Array:
        mass = _checked_mass(self.mass, owner="Ingoing Schwarzschild metric")
        radius, polar = _checked_spherical_coordinates(
            coordinates,
            owner="Ingoing Schwarzschild metric",
        )
        sine_squared = jnp.sin(polar) ** 2
        radius_squared = radius * radius
        factor = 1.0 - 2.0 * mass / radius
        dtype = jnp.result_type(coordinates, mass, 1.0)
        matrix = jnp.zeros((4, 4), dtype=dtype)
        matrix = matrix.at[0, 0].set(-factor)
        matrix = matrix.at[0, 1].set(1.0)
        matrix = matrix.at[1, 0].set(1.0)
        matrix = matrix.at[2, 2].set(radius_squared)
        matrix = matrix.at[3, 3].set(radius_squared * sine_squared)
        return matrix if self.convention == "mostly_plus" else -matrix


class _KerrBoyerLindquistMetricMap(StrictModule):
    mass: Array
    spin: Array
    convention: LorentzianConvention = eqx.field(static=True)

    def __init__(
        self,
        mass: ArrayLike,
        spin: ArrayLike,
        convention: LorentzianConvention,
        /,
    ):
        self.mass = _scalar_parameter(mass, "Kerr mass")
        self.spin = _scalar_parameter(spin, "Kerr spin")
        self.convention = convention

    def __call__(self, coordinates: Array, /) -> Array:
        mass, spin, radius, polar, sigma = _checked_kerr_coordinates(
            coordinates,
            self.mass,
            self.spin,
            owner="Boyer-Lindquist Kerr metric",
        )
        delta = _kerr_delta(mass, spin, radius)
        delta = eqx.error_if(
            delta,
            ~jnp.isfinite(delta) | (delta == 0.0),
            "Boyer-Lindquist coordinates exclude both Kerr horizons (Delta = 0).",
        )
        sine_squared = jnp.sin(polar) ** 2
        radius_squared = radius * radius
        spin_squared = spin * spin
        mass_radius_over_sigma = mass * radius / sigma
        time_azimuth = -2.0 * mass_radius_over_sigma * spin * sine_squared
        azimuth_azimuth = (
            radius_squared
            + spin_squared
            + 2.0 * mass_radius_over_sigma * spin_squared * sine_squared
        ) * sine_squared
        dtype = jnp.result_type(coordinates, mass, spin, 1.0)
        matrix = jnp.zeros((4, 4), dtype=dtype)
        matrix = matrix.at[0, 0].set(-(1.0 - 2.0 * mass_radius_over_sigma))
        matrix = matrix.at[0, 3].set(time_azimuth)
        matrix = matrix.at[3, 0].set(time_azimuth)
        matrix = matrix.at[1, 1].set(sigma / delta)
        matrix = matrix.at[2, 2].set(sigma)
        matrix = matrix.at[3, 3].set(azimuth_azimuth)
        return matrix if self.convention == "mostly_plus" else -matrix


class _IngoingKerrMetricMap(StrictModule):
    mass: Array
    spin: Array
    convention: LorentzianConvention = eqx.field(static=True)

    def __init__(
        self,
        mass: ArrayLike,
        spin: ArrayLike,
        convention: LorentzianConvention,
        /,
    ):
        self.mass = _scalar_parameter(mass, "Kerr mass")
        self.spin = _scalar_parameter(spin, "Kerr spin")
        self.convention = convention

    def __call__(self, coordinates: Array, /) -> Array:
        mass, spin, radius, polar, sigma = _checked_kerr_coordinates(
            coordinates,
            self.mass,
            self.spin,
            owner="Ingoing Kerr metric",
        )
        sine_squared = jnp.sin(polar) ** 2
        radius_squared = radius * radius
        spin_squared = spin * spin
        mass_radius_over_sigma = mass * radius / sigma
        time_azimuth = -2.0 * mass_radius_over_sigma * spin * sine_squared
        radial_azimuth = -spin * sine_squared
        azimuth_azimuth = (
            radius_squared
            + spin_squared
            + 2.0 * mass_radius_over_sigma * spin_squared * sine_squared
        ) * sine_squared
        dtype = jnp.result_type(coordinates, mass, spin, 1.0)
        matrix = jnp.zeros((4, 4), dtype=dtype)
        matrix = matrix.at[0, 0].set(-(1.0 - 2.0 * mass_radius_over_sigma))
        matrix = matrix.at[0, 1].set(1.0)
        matrix = matrix.at[1, 0].set(1.0)
        matrix = matrix.at[0, 3].set(time_azimuth)
        matrix = matrix.at[3, 0].set(time_azimuth)
        matrix = matrix.at[1, 3].set(radial_azimuth)
        matrix = matrix.at[3, 1].set(radial_azimuth)
        matrix = matrix.at[2, 2].set(sigma)
        matrix = matrix.at[3, 3].set(azimuth_azimuth)
        return matrix if self.convention == "mostly_plus" else -matrix


def _kerr_radial_shifts(
    mass: Array,
    spin: Array,
    radius: Array,
    delta: Array,
    /,
) -> tuple[Array, Array]:
    _, _, root = _kerr_root_values(mass, spin)
    offset = radius - mass

    def extremal(_: None) -> tuple[Array, Array]:
        logarithm = jnp.log(jnp.abs(offset / mass))
        tortoise = radius + 2.0 * mass * logarithm - 2.0 * mass * mass / offset
        azimuth = -spin / offset
        return tortoise, azimuth

    def nonextremal(_: None) -> tuple[Array, Array]:
        def outside(_: None) -> Array:
            return -2.0 * jnp.arctanh(root / offset)

        def between(_: None) -> Array:
            return -2.0 * jnp.arctanh(offset / root)

        logarithmic_ratio = jax.lax.cond(
            jnp.abs(offset) > root,
            outside,
            between,
            operand=None,
        )
        log_delta = jnp.log(jnp.abs(delta) / (mass * mass))
        tortoise = radius + mass * log_delta + mass * mass * logarithmic_ratio / root
        azimuth = spin * logarithmic_ratio / (2.0 * root)
        return tortoise, azimuth

    return jax.lax.cond(root == 0.0, extremal, nonextremal, operand=None)


class _BoyerLindquistIngoingKerrMap(StrictModule):
    mass: Array
    spin: Array
    direction: int = eqx.field(static=True)

    def __init__(
        self,
        mass: ArrayLike,
        spin: ArrayLike,
        direction: int,
        /,
    ):
        self.mass = _scalar_parameter(mass, "Kerr transition mass")
        self.spin = _scalar_parameter(spin, "Kerr transition spin")
        self.direction = int(direction)

    def __call__(self, coordinates: Array, /) -> Array:
        time, _, _, azimuth = coordinates
        mass, spin, radius, polar, _ = _checked_kerr_coordinates(
            coordinates,
            self.mass,
            self.spin,
            owner="Boyer-Lindquist/ingoing Kerr transition",
        )
        mass = eqx.error_if(
            mass,
            jnp.abs(spin) > mass,
            "Boyer-Lindquist/ingoing Kerr transition requires |a| <= M "
            "because overextremal Kerr has no real horizon-root primitive.",
        )
        inner, outer, _ = _kerr_root_values(mass, spin)
        delta = (radius - outer) * (radius - inner)
        delta = eqx.error_if(
            delta,
            ~jnp.isfinite(delta) | (delta == 0.0),
            "Boyer-Lindquist/ingoing Kerr transition excludes Delta = 0.",
        )
        radial_time_shift, radial_azimuth_shift = _kerr_radial_shifts(
            mass,
            spin,
            radius,
            delta,
        )
        direction = jnp.asarray(self.direction, dtype=radius.dtype)
        return jnp.stack(
            (
                time + direction * radial_time_shift,
                radius,
                polar,
                azimuth + direction * radial_azimuth_shift,
            )
        )


def ingoing_schwarzschild_metric(
    mass: ArrayLike,
    /,
    *,
    chart: CoordinateChart,
    convention: LorentzianConvention = "mostly_plus",
) -> LorentzianMetric:
    """Construct future-ingoing Schwarzschild in ``(v, r, theta, phi)``.

    Unlike static Schwarzschild coordinates, this chart and metric are regular at
    the future horizon. The curvature singularity ``r = 0`` and the spherical
    coordinate axis are outside the metric map's domain.
    """

    if chart.dimension != 4:
        raise ValueError(
            "ingoing_schwarzschild_metric requires a four-dimensional chart."
        )
    return LorentzianMetric(
        _IngoingSchwarzschildMetricMap(mass, convention),
        chart=chart,
        convention=convention,
    )


def kerr_boyer_lindquist_metric(
    mass: ArrayLike,
    spin: ArrayLike,
    /,
    *,
    chart: CoordinateChart,
    convention: LorentzianConvention = "mostly_plus",
) -> LorentzianMetric:
    """Construct Kerr in Boyer-Lindquist ``(t, r, theta, phi)`` coordinates.

    The physical spin parameter ``a`` is signed. Both horizons, the coordinate
    axis, and the ring singularity are excluded from this chart realization.
    """

    if chart.dimension != 4:
        raise ValueError("kerr_boyer_lindquist_metric requires a four-dimensional chart.")
    return LorentzianMetric(
        _KerrBoyerLindquistMetricMap(mass, spin, convention),
        chart=chart,
        convention=convention,
    )


def ingoing_kerr_metric(
    mass: ArrayLike,
    spin: ArrayLike,
    /,
    *,
    chart: CoordinateChart,
    convention: LorentzianConvention = "mostly_plus",
) -> LorentzianMetric:
    """Construct future-ingoing Kerr in ``(v, r, theta, phi_tilde)``.

    This exact advanced chart is regular at the future outer horizon and retains
    the sign of ``a`` in both rotational cross terms. The coordinate axis and the
    Kerr ring singularity remain outside the metric map's domain.
    """

    if chart.dimension != 4:
        raise ValueError("ingoing_kerr_metric requires a four-dimensional chart.")
    return LorentzianMetric(
        _IngoingKerrMetricMap(mass, spin, convention),
        chart=chart,
        convention=convention,
    )


def boyer_lindquist_to_ingoing_kerr_transition(
    mass: ArrayLike,
    spin: ArrayLike,
    /,
    *,
    source: CoordinateChart,
    target: CoordinateChart,
) -> ChartTransition:
    """Map ``(t,r,theta,phi)`` to future-ingoing ``(v,r,theta,phi_tilde)``.

    The radial primitives use cancellation-safe subextremal expressions and their
    exact extremal limits. Their additive logarithmic reference is fixed by ``M``;
    this only chooses the origins of ``v`` and ``phi_tilde``. The transition and
    its inverse are defined away from the Boyer-Lindquist horizons.
    """

    if source.dimension != 4 or target.dimension != 4:
        raise ValueError(
            "boyer_lindquist_to_ingoing_kerr_transition requires two "
            "four-dimensional charts."
        )
    return ChartTransition(
        source,
        target,
        _BoyerLindquistIngoingKerrMap(mass, spin, 1),
        inverse=_BoyerLindquistIngoingKerrMap(mass, spin, -1),
    )


def kerr_horizon_radii(mass: ArrayLike, spin: ArrayLike, /) -> Array:
    """Return ``[r_inner, r_outer]`` using the cancellation-safe Kerr roots.

    Invalid or overextremal black-hole parameters produce NaNs; the discriminant
    is never clipped to manufacture a horizon.
    """

    mass_array, spin_array = jnp.broadcast_arrays(
        jnp.asarray(mass),
        jnp.asarray(spin),
    )
    dtype = jnp.result_type(mass_array, spin_array, 1.0)
    mass_array = mass_array.astype(dtype)
    spin_array = spin_array.astype(dtype)
    inner, outer, _ = _kerr_root_values(mass_array, spin_array)
    valid = (
        jnp.isfinite(mass_array)
        & jnp.isfinite(spin_array)
        & (mass_array > 0.0)
        & (jnp.abs(spin_array) <= mass_array)
    )
    radii = jnp.stack((inner, outer), axis=-1)
    return jnp.where(valid[..., None], radii, jnp.nan)


def kerr_ergosurface_radii(
    mass: ArrayLike,
    spin: ArrayLike,
    polar: ArrayLike,
    /,
) -> Array:
    """Return inner/outer stationary-limit radii at polar angle ``theta``.

    A locally real stationary-limit pair is retained for overextremal Kerr; angles
    without real roots produce NaNs. The discriminant is never clipped.
    """

    mass_array, spin_array, polar_array = jnp.broadcast_arrays(
        jnp.asarray(mass),
        jnp.asarray(spin),
        jnp.asarray(polar),
    )
    dtype = jnp.result_type(mass_array, spin_array, polar_array, 1.0)
    mass_array = mass_array.astype(dtype)
    spin_array = spin_array.astype(dtype)
    polar_array = polar_array.astype(dtype)
    projected_spin = spin_array * jnp.cos(polar_array)
    discriminant = (mass_array - projected_spin) * (mass_array + projected_spin)
    root = jnp.sqrt(discriminant)
    outer = mass_array + root
    inner = projected_spin * projected_spin / outer
    valid = (
        jnp.isfinite(mass_array)
        & jnp.isfinite(spin_array)
        & jnp.isfinite(polar_array)
        & (mass_array > 0.0)
        & (discriminant >= 0.0)
        & (polar_array >= 0.0)
        & (polar_array <= jnp.pi)
    )
    radii = jnp.stack((inner, outer), axis=-1)
    return jnp.where(valid[..., None], radii, jnp.nan)


def kerr_kretschmann_scalar(
    mass: ArrayLike,
    spin: ArrayLike,
    radius: ArrayLike,
    polar: ArrayLike,
    /,
) -> Array:
    """Return ``R_abcd R^abcd`` for stationary Kerr coordinates."""

    mass_array, spin_array, radius_array, polar_array = jnp.broadcast_arrays(
        jnp.asarray(mass),
        jnp.asarray(spin),
        jnp.asarray(radius),
        jnp.asarray(polar),
    )
    dtype = jnp.result_type(
        mass_array,
        spin_array,
        radius_array,
        polar_array,
        1.0,
    )
    mass_array = mass_array.astype(dtype)
    spin_array = spin_array.astype(dtype)
    radius_array = radius_array.astype(dtype)
    polar_array = polar_array.astype(dtype)
    projected_spin = spin_array * jnp.cos(polar_array)
    radius_squared = radius_array * radius_array
    projected_squared = projected_spin * projected_spin
    sigma = radius_squared + projected_squared
    polynomial = (
        (radius_squared - 15.0 * projected_squared) * radius_squared
        + 15.0 * projected_squared * projected_squared
    ) * radius_squared - projected_squared * projected_squared * projected_squared
    return 48.0 * mass_array * mass_array * polynomial / sigma**6


def kerr_pontryagin_scalar(
    mass: ArrayLike,
    spin: ArrayLike,
    radius: ArrayLike,
    polar: ArrayLike,
    /,
) -> Array:
    """Return the right-handed dual invariant ``R_abcd *R^abcd`` for Kerr."""

    mass_array, spin_array, radius_array, polar_array = jnp.broadcast_arrays(
        jnp.asarray(mass),
        jnp.asarray(spin),
        jnp.asarray(radius),
        jnp.asarray(polar),
    )
    dtype = jnp.result_type(
        mass_array,
        spin_array,
        radius_array,
        polar_array,
        1.0,
    )
    mass_array = mass_array.astype(dtype)
    spin_array = spin_array.astype(dtype)
    radius_array = radius_array.astype(dtype)
    polar_array = polar_array.astype(dtype)
    projected_spin = spin_array * jnp.cos(polar_array)
    radius_squared = radius_array * radius_array
    projected_squared = projected_spin * projected_spin
    sigma = radius_squared + projected_squared
    polynomial = (
        3.0 * radius_squared * radius_squared
        - 10.0 * radius_squared * projected_squared
        + 3.0 * projected_squared * projected_squared
    )
    return (
        96.0
        * mass_array
        * mass_array
        * radius_array
        * projected_spin
        * polynomial
        / sigma**6
    )


__all__ = [
    "ExactMetricDomainStatus",
    "ingoing_kerr_domain_evidence",
    "ingoing_schwarzschild_domain_evidence",
    "kerr_boyer_lindquist_domain_evidence",
    "boyer_lindquist_to_ingoing_kerr_transition",
    "ingoing_kerr_metric",
    "ingoing_schwarzschild_metric",
    "kerr_boyer_lindquist_metric",
    "kerr_ergosurface_radii",
    "kerr_horizon_radii",
    "kerr_kretschmann_scalar",
    "kerr_pontryagin_scalar",
]
