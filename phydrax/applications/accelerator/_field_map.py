#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Magnet field maps, insertion devices, and lab-time tracking through them.

Every field element implements the core ``ExternalFieldSource`` capability in
the straight Cartesian map frame: ``z`` is the beamline axis ``s`` and
``(x, y)`` are horizontal and vertical offsets. Analytic magnets use complex
analytic on-axis profiles ``b(ζ)``: the field ``B_y = Re b(z + i y)``,
``B_z = Im b(z + i y)`` (and ``B_x = Re h(z + i x)``, ``B_z += Im h(z + i x)``
for a horizontal component) is curl- and divergence-free wherever ``b`` is
analytic, so every model is an exact vacuum magnetostatic field inside its
declared aperture. Longitudinal envelopes are the smooth window
``g(u) = ½[tanh((u + a)/w) − tanh((u − a)/w)]`` with ``∫ g du = 2a``; its poles
at ``Im u = ±π w / 2`` bound the admissible aperture.

A ``FieldMapBeamline`` binds one ``ElectromagneticScaleContract``: every
element's lengths, fields, and times are in that scale's units.
``track_field_map`` converts an ``AcceleratorBunch`` at the entrance plane into
lab-time initial conditions per ``AcceleratorConvention``, advances each lane
at its own arrival time with ``RelativisticPushPlan``, and converts exit-plane
crossings back to accelerator coordinates. The recorded lanes form a
``ChargedTrajectory`` for trajectory-radiation spectra.
"""

from __future__ import annotations

import math
from enum import IntFlag
from typing import assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import apply_gather_stencil, rectilinear_stencil
from ..._physical import ElectromagneticScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import finite_real_scalar, positive_finite_float, positive_integer
from ...discretization.pic import (
    ExternalFieldSample,
    ExternalFieldSource,
    RelativisticPusher,
    RelativisticPushPlan,
)
from ...electromagnetics import ChargedTrajectory
from ...typing import Bool, Dim, Float64, Identifier, Int32, parse, Scalar, Scope, Size
from ._beam import _late_sign, AcceleratorBunch, AcceleratorConvention


InsertionDevicePolarization: TypeAlias = Literal["planar", "helical"]

# Apertures are limited to half of the window's analyticity strip |Im u| < π w / 2
# so that the tanh poles stay a finite distance from every supported point.
_APERTURE_STRIP_FRACTION = 0.5
_TRAJECTORY_CHANNELS = 10


class _XNodeDim(Dim, minimum=3):
    """Horizontal field-map nodes."""


class _YNodeDim(Dim, minimum=3):
    """Vertical field-map nodes."""


class _ZNodeDim(Dim, minimum=3):
    """Longitudinal field-map nodes."""


class _ParticleDim(Dim, minimum=1):
    """Tracked particle lanes."""


def _window(argument: Array, half_length: float, width: float, /) -> Array:
    """Complex window ``½[tanh((ζ + a)/w) − tanh((ζ − a)/w)]``."""
    return 0.5 * (
        jnp.tanh((argument + half_length) / width)
        - jnp.tanh((argument - half_length) / width)
    )


def _window_derivative(argument: Array, half_length: float, width: float, /) -> Array:
    """``g′(ζ) = [sech²((ζ + a)/w) − sech²((ζ − a)/w)] / (2w)``."""
    leading = jnp.tanh((argument + half_length) / width)
    trailing = jnp.tanh((argument - half_length) / width)
    return (trailing * trailing - leading * leading) / (2.0 * width)


def _aperture(
    half_width: float, half_gap: float, ramp_width: float, *, helical: bool
) -> tuple[float, float]:
    width = positive_finite_float(half_width, "half_width")
    gap = positive_finite_float(half_gap, "half_gap")
    limit = _APERTURE_STRIP_FRACTION * 0.5 * math.pi * ramp_width
    if gap > limit or (helical and width > limit):
        raise ValueError(
            "The magnet aperture exceeds the field model's analytic support: "
            f"transverse offsets in the analytic plane must stay within {limit!r} "
            "(π w / 4 for ramp width w); lengthen the ramps or narrow the aperture."
        )
    return width, gap


def _positions(positions: ArrayLike, /) -> Array:
    array = jnp.asarray(positions, dtype=jnp.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("positions must have shape (particles, 3).")
    return array


def _analytic_sample(
    magnetic: Array, positions: Array, half_width: float, half_gap: float, /
) -> ExternalFieldSample:
    support = (jnp.abs(positions[:, 0]) <= half_width) & (
        jnp.abs(positions[:, 1]) <= half_gap
    )
    return ExternalFieldSample(
        jnp.zeros_like(positions),
        jnp.where(support[:, None], magnetic, 0.0),
        support,
    )


class InsertionDeviceResonance(StrictModule):
    """Textbook spectral scales of an insertion device for one electron energy.

    ``deflection_parameter`` is ``K = e B₀ λ_u / (2π mₑ c)``;
    ``fundamental_angular_frequency`` is the on-axis resonance
    ``2γ² ω_u / (1 + K²/2)`` (planar) or ``2γ² ω_u / (1 + K²)`` (helical), with
    ``ω_u = 2π c / λ_u``; harmonic ``n`` at polar angle ``θ`` resonates at
    ``2nγ² ω_u / (1 + K²/2 + γ²θ²)`` (planar). ``critical_angular_frequency``
    ``(3/2) γ² K ω_u`` is the synchrotron critical frequency of the peak field,
    the spectral scale of a wiggler (``K ≫ 1``).
    """

    lorentz_factor: float = eqx.field(static=True)
    deflection_parameter: float = eqx.field(static=True)
    undulator_angular_frequency: float = eqx.field(static=True)
    fundamental_angular_frequency: float = eqx.field(static=True)
    critical_angular_frequency: float = eqx.field(static=True)


class InsertionDeviceField(StrictModule, NonTrainableState):
    """Planar or helical undulator/wiggler with matched smooth terminations.

    The vertical field has on-axis profile ``b(u) = B₀ g(u) cos(k_u u)`` about
    ``center`` (``u = z − center``) with flat half-length ``a = N λ_u / 2`` and
    ramp width ``w = ramp_periods · λ_u``; ``"helical"`` adds the horizontal
    profile ``h(u) = −k_u⁻¹ b′(u) = B₀ [g(u) sin(k_u u) − g′(u) cos(k_u u)/k_u]``,
    so the on-axis field ``(B₀ sin k_u u, B₀ cos k_u u, 0)`` rotates at constant
    magnitude inside the flat top.

    Matched termination: ``b`` is even about ``center`` and the window transform
    ``ĝ(k_u) = 2 sin(k_u a) S(k_u)/k_u`` vanishes because ``sin(k_u a) = 0``, so
    ``∫ b = ∫ u b = 0``; ``h`` is a total derivative, so ``∫ h = 0`` and
    ``∫ u h = ĝ(k_u) B₀/k_u = 0``. A particle entering on axis far from the
    device therefore exits with zero offset and angle to first order in
    ``K/γ``; in the planar mid-plane the transverse momentum is ``−q ∫ B_y dz``
    exactly and the exit offset vanishes by symmetry to all orders. A wiggler
    is this device at ``K ≫ 1``.

    ``aperture = (half_width, half_gap)`` is the field-support region: samples
    with ``|x| > half_width`` or ``|y| > half_gap`` are unsupported. Offsets in
    an analytic plane (``y``; also ``x`` for helical) are limited to ``π w / 4``.
    """

    peak_field: float = eqx.field(static=True)
    period: float = eqx.field(static=True)
    period_count: int = eqx.field(static=True)
    polarization: InsertionDevicePolarization = eqx.field(static=True)
    center: float = eqx.field(static=True)
    ramp_width: float = eqx.field(static=True)
    half_width: float = eqx.field(static=True)
    half_gap: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        peak_field: float,
        period: float,
        period_count: int,
        /,
        *,
        polarization: InsertionDevicePolarization,
        center: float,
        aperture: tuple[float, float],
        ramp_periods: float = 1.0,
    ) -> None:
        field = finite_real_scalar(peak_field, "peak_field")
        if field == 0.0:
            raise ValueError("peak_field must be nonzero.")
        wavelength = positive_finite_float(period, "period")
        count = positive_integer(period_count, "period_count")
        polarization_ = parse(polarization, InsertionDevicePolarization, "polarization")
        center_ = finite_real_scalar(center, "center")
        ramp = positive_finite_float(ramp_periods, "ramp_periods") * wavelength
        if len(aperture) != 2:
            raise ValueError("aperture must be (half_width, half_gap).")
        half_width, half_gap = _aperture(
            aperture[0], aperture[1], ramp, helical=polarization_ == "helical"
        )
        self.peak_field = field
        self.period = wavelength
        self.period_count = count
        self.polarization = polarization_
        self.center = center_
        self.ramp_width = ramp
        self.half_width = half_width
        self.half_gap = half_gap
        self.source_id = canonical_fingerprint(
            {
                "kind": "insertion-device-field",
                "peak_field": field,
                "period": wavelength,
                "period_count": count,
                "polarization": polarization_,
                "center": center_,
                "ramp_width": ramp,
                "aperture": [half_width, half_gap],
            }
        )

    def external_fields(self, positions: Array, times: Array, /) -> ExternalFieldSample:
        del times
        points = _positions(positions)
        wavenumber = 2.0 * math.pi / self.period
        flat = 0.5 * self.period_count * self.period
        u = points[:, 2] - self.center
        vertical = u + 1j * points[:, 1]
        profile = (
            self.peak_field
            * _window(vertical, flat, self.ramp_width)
            * jnp.cos(wavenumber * vertical)
        )
        horizontal_field = jnp.zeros_like(u)
        longitudinal = jnp.imag(profile)
        match self.polarization:
            case "planar":
                pass
            case "helical":
                horizontal = u + 1j * points[:, 0]
                window = _window(horizontal, flat, self.ramp_width)
                slope = _window_derivative(horizontal, flat, self.ramp_width)
                phase = wavenumber * horizontal
                transverse = self.peak_field * (
                    window * jnp.sin(phase) - slope * jnp.cos(phase) / wavenumber
                )
                horizontal_field = jnp.real(transverse)
                longitudinal = longitudinal + jnp.imag(transverse)
            case _:
                assert_never(self.polarization)
        magnetic = jnp.stack((horizontal_field, jnp.real(profile), longitudinal), axis=-1)
        return _analytic_sample(magnetic, points, self.half_width, self.half_gap)

    def resonance(
        self, scale: ElectromagneticScaleContract, lorentz_factor: float, /
    ) -> InsertionDeviceResonance:
        """Return ``K`` and the resonant and critical frequencies for electrons."""
        if not isinstance(scale, ElectromagneticScaleContract):
            raise TypeError("scale must be an ElectromagneticScaleContract.")
        gamma = positive_finite_float(lorentz_factor, "lorentz_factor")
        if gamma <= 1.0:
            raise ValueError("lorentz_factor must exceed one.")
        light = float(scale.speed_of_light)
        deflection = (
            float(scale.elementary_charge)
            * abs(self.peak_field)
            * self.period
            / (2.0 * math.pi * float(scale.electron_mass) * light)
        )
        undulator = 2.0 * math.pi * light / self.period
        match self.polarization:
            case "planar":
                denominator = 1.0 + 0.5 * deflection**2
            case "helical":
                denominator = 1.0 + deflection**2
            case _:
                assert_never(self.polarization)
        return InsertionDeviceResonance(
            gamma,
            deflection,
            undulator,
            2.0 * gamma**2 * undulator / denominator,
            1.5 * gamma**2 * deflection * undulator,
        )


class DipoleBendField(StrictModule, NonTrainableState):
    """Straight-pole dipole with smooth analytic fringes.

    The vertical on-axis field is ``B₀ g(z − center)`` with flat half-length
    ``length / 2`` and fringe width ``fringe_width``; ``length`` is the exact
    magnetic length ``∫ B_y dz / B₀``. In the mid-plane the exit transverse
    momentum is ``−q B₀ length`` exactly. The pole faces are perpendicular to
    ``z``; the map frame is not rotated with the orbit. ``aperture`` is the
    field-support region; ``half_gap`` is limited to ``π w / 4``.
    """

    field: float = eqx.field(static=True)
    length: float = eqx.field(static=True)
    center: float = eqx.field(static=True)
    fringe_width: float = eqx.field(static=True)
    half_width: float = eqx.field(static=True)
    half_gap: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        field: float,
        length: float,
        /,
        *,
        center: float,
        fringe_width: float,
        aperture: tuple[float, float],
    ) -> None:
        field_ = finite_real_scalar(field, "field")
        length_ = positive_finite_float(length, "length")
        center_ = finite_real_scalar(center, "center")
        fringe = positive_finite_float(fringe_width, "fringe_width")
        if len(aperture) != 2:
            raise ValueError("aperture must be (half_width, half_gap).")
        half_width, half_gap = _aperture(aperture[0], aperture[1], fringe, helical=False)
        self.field = field_
        self.length = length_
        self.center = center_
        self.fringe_width = fringe
        self.half_width = half_width
        self.half_gap = half_gap
        self.source_id = canonical_fingerprint(
            {
                "kind": "dipole-bend-field",
                "field": field_,
                "length": length_,
                "center": center_,
                "fringe_width": fringe,
                "aperture": [half_width, half_gap],
            }
        )

    def external_fields(self, positions: Array, times: Array, /) -> ExternalFieldSample:
        del times
        points = _positions(positions)
        vertical = (points[:, 2] - self.center) + 1j * points[:, 1]
        profile = self.field * _window(vertical, 0.5 * self.length, self.fringe_width)
        magnetic = jnp.stack(
            (jnp.zeros_like(points[:, 0]), jnp.real(profile), jnp.imag(profile)),
            axis=-1,
        )
        return _analytic_sample(magnetic, points, self.half_width, self.half_gap)


def _axis_nodes(values: ArrayLike, name: str, /) -> np.ndarray:
    nodes = np.asarray(values, dtype=np.float64)
    if nodes.ndim != 1 or nodes.shape[0] < 3:
        raise ValueError(f"{name} must be a vector of at least three nodes.")
    if not np.all(np.isfinite(nodes)) or np.any(np.diff(nodes) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing.")
    return nodes


def _interpolation_estimate(
    table: np.ndarray, axes: tuple[np.ndarray, ...], /
) -> tuple[float, float, float]:
    """Multilinear error estimate ``Σ_i h_i² max|∂_i² f| / 8`` per component.

    ``∂_i² f`` is estimated by second divided differences of the table; the
    bound ``|f − I f| ≤ Σ_i h_i² max|∂_i² f| / 8`` holds for ``C²`` fields.
    """
    total = np.zeros((3,), dtype=np.float64)
    for axis, nodes in enumerate(axes):
        spacing = np.diff(nodes)
        slopes = np.diff(table, axis=axis) / np.expand_dims(
            spacing, tuple(index for index in range(4) if index != axis)
        )
        span = spacing[1:] + spacing[:-1]
        curvature = (
            2.0
            * np.diff(slopes, axis=axis)
            / np.expand_dims(span, tuple(index for index in range(4) if index != axis))
        )
        maximum = np.max(np.abs(curvature), axis=tuple(i for i in range(3)))
        total += float(np.max(spacing)) ** 2 * maximum / 8.0
    return (float(total[0]), float(total[1]), float(total[2]))


def _field_table(
    values: ArrayLike, shape: tuple[int, int, int], name: str, /
) -> np.ndarray:
    table = np.asarray(values, dtype=np.float64)
    if table.shape != (*shape, 3):
        raise ValueError(f"{name} must have shape (x_nodes, y_nodes, z_nodes, 3).")
    if not np.all(np.isfinite(table)):
        raise ValueError(f"{name} must be finite.")
    return table


class TabulatedFieldMap(StrictModule, NonTrainableState):
    """Static 3-D field map on a rectilinear grid with multilinear interpolation.

    Fields are interpolated with the native rectilinear gather. Positions with
    ``z`` outside ``[z_nodes[0], z_nodes[-1]]`` lie beyond the element and
    receive zero field with support; positions inside the longitudinal range
    but transversely outside the grid are unsupported. ``*_interpolation_estimate``
    report ``Σ_i h_i² max|∂_i² f| / 8`` per component from second divided
    differences of the table, the multilinear error bound of a ``C²`` field.
    """

    __strict_contract__ = True

    x_nodes: Float64[_XNodeDim]
    y_nodes: Float64[_YNodeDim]
    z_nodes: Float64[_ZNodeDim]
    magnetic: Float64[_XNodeDim, _YNodeDim, _ZNodeDim, Literal[3]]
    electric: Float64[_XNodeDim, _YNodeDim, _ZNodeDim, Literal[3]] | None
    magnetic_interpolation_estimate: tuple[float, float, float] = eqx.field(static=True)
    electric_interpolation_estimate: tuple[float, float, float] | None = eqx.field(
        static=True
    )
    source_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        x_nodes: ArrayLike,
        y_nodes: ArrayLike,
        z_nodes: ArrayLike,
        magnetic: ArrayLike,
        /,
        *,
        electric: ArrayLike | None = None,
    ) -> None:
        axes = (
            _axis_nodes(x_nodes, "x_nodes"),
            _axis_nodes(y_nodes, "y_nodes"),
            _axis_nodes(z_nodes, "z_nodes"),
        )
        shape = (axes[0].shape[0], axes[1].shape[0], axes[2].shape[0])
        magnetic_ = _field_table(magnetic, shape, "magnetic")
        electric_ = (
            None if electric is None else _field_table(electric, shape, "electric")
        )
        scope = Scope()
        self.x_nodes = parse(
            jnp.asarray(axes[0]), Float64[_XNodeDim], "x_nodes", scope=scope
        )
        self.y_nodes = parse(
            jnp.asarray(axes[1]), Float64[_YNodeDim], "y_nodes", scope=scope
        )
        self.z_nodes = parse(
            jnp.asarray(axes[2]), Float64[_ZNodeDim], "z_nodes", scope=scope
        )
        self.magnetic = parse(
            jnp.asarray(magnetic_),
            Float64[_XNodeDim, _YNodeDim, _ZNodeDim, Literal[3]],
            "magnetic",
            scope=scope,
        )
        self.electric = (
            None
            if electric_ is None
            else parse(
                jnp.asarray(electric_),
                Float64[_XNodeDim, _YNodeDim, _ZNodeDim, Literal[3]],
                "electric",
                scope=scope,
            )
        )
        self.magnetic_interpolation_estimate = _interpolation_estimate(magnetic_, axes)
        self.electric_interpolation_estimate = (
            None if electric_ is None else _interpolation_estimate(electric_, axes)
        )
        self.source_id = canonical_fingerprint(
            {
                "kind": "tabulated-field-map",
                "axes": array_tree_fingerprint(axes),
                "magnetic": array_tree_fingerprint(magnetic_),
                "electric": None
                if electric_ is None
                else array_tree_fingerprint(electric_),
            }
        )

    def external_fields(self, positions: Array, times: Array, /) -> ExternalFieldSample:
        del times
        points = _positions(positions)
        stencil = rectilinear_stencil(
            (self.x_nodes, self.y_nodes, self.z_nodes),
            points,
            boundary=("constant", "constant", "constant"),
        )
        magnetic = apply_gather_stencil(self.magnetic.reshape((-1, 3)), stencil)
        electric = (
            jnp.zeros_like(points)
            if self.electric is None
            else apply_gather_stencil(self.electric.reshape((-1, 3)), stencil).values
        )
        beyond = (points[:, 2] < self.z_nodes[0]) | (points[:, 2] > self.z_nodes[-1])
        return ExternalFieldSample(electric, magnetic.values, magnetic.support | beyond)


class FieldMapBeamline(StrictModule, NonTrainableState):
    """Superposed field elements bound to one electromagnetic scale.

    Every element's lengths, times, and fields are in ``scale`` units. The
    beamline field is the element sum; a sample is supported only where every
    element supports it.
    """

    scale: ElectromagneticScaleContract
    elements: tuple[ExternalFieldSource, ...]
    element_ids: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: ElectromagneticScaleContract,
        elements: tuple[ExternalFieldSource, ...],
        /,
        *,
        element_ids: tuple[str, ...],
    ) -> None:
        if not isinstance(scale, ElectromagneticScaleContract):
            raise TypeError("scale must be an ElectromagneticScaleContract.")
        elements_ = tuple(elements)
        if not elements_:
            raise ValueError("A field-map beamline needs at least one element.")
        for element in elements_:
            if not isinstance(element, ExternalFieldSource):
                raise TypeError("Every beamline element must be an ExternalFieldSource.")
        ids = tuple(str(value).strip() for value in element_ids)
        if len(ids) != len(elements_) or any(not value for value in ids):
            raise ValueError("element_ids must name every element.")
        if len(set(ids)) != len(ids):
            raise ValueError("element_ids must be unique.")
        self.scale = scale
        self.elements = elements_
        self.element_ids = ids
        self.source_id = canonical_fingerprint(
            {
                "kind": "field-map-beamline",
                "scale": scale.scale_id,
                "elements": [
                    [identifier, element.source_id]
                    for identifier, element in zip(ids, elements_, strict=True)
                ],
            }
        )

    def external_fields(self, positions: Array, times: Array, /) -> ExternalFieldSample:
        points = _positions(positions)
        electric = jnp.zeros_like(points)
        magnetic = jnp.zeros_like(points)
        support = jnp.ones((points.shape[0],), dtype=jnp.bool_)
        for element in self.elements:
            sample = element.external_fields(points, times)
            electric = electric + sample.electric
            magnetic = magnetic + sample.magnetic
            support = support & sample.support
        return ExternalFieldSample(
            jnp.where(support[:, None], electric, 0.0),
            jnp.where(support[:, None], magnetic, 0.0),
            support,
        )


class FieldMapTrackingStatus(IntFlag):
    SUCCESS = 0
    NONFINITE = 1
    SUPERLUMINAL = 2
    UNSUPPORTED = 4
    NOT_EXITED = 8
    INVALID_ENTRANCE = 16


class FieldMapTrackingResourceError(ValueError):
    """Field-map tracking exceeds its declared trajectory memory."""


class FieldMapTrackingPlan(StrictModule, NonTrainableState):
    """Lab-time tracking of a bunch between two planes of a field-map beamline.

    A particle with coordinates ``(x, px/p₀, y, py/p₀, ζ, δ)`` at the entrance
    plane ``z = entrance_plane`` has momentum ``p = p₀(1 + δ)`` along
    ``(px, py, √((1+δ)² − px² − py²))/(1+δ)`` and arrives at lab time
    ``t = reference_time + σ ζ / (β₀ c)``, with ``σ = +1`` for
    ``"positive-late"`` and ``−1`` for ``"positive-early"``. Each lane advances
    from its own arrival time in ``step_count`` leapfrog steps of
    ``time_step`` with the ``method`` relativistic pusher; the first half kick
    starts the stagger. At the exit plane the crossing of the exact leapfrog
    drift gives ``ζ = σ β₀ c (t_exit − reference_time − (exit − entrance)/(β₀ c))``
    relative to a reference moving at ``β₀ c`` along ``z``, and
    ``δ = |p|/p₀ − 1``. Recorded trajectory memory is refused above
    ``maximum_trajectory_bytes``.
    """

    beamline: FieldMapBeamline
    pusher: RelativisticPushPlan
    convention: AcceleratorConvention
    entrance_plane: float = eqx.field(static=True)
    exit_plane: float = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    reference_time: float = eqx.field(static=True)
    late_sign: float = eqx.field(static=True)
    maximum_trajectory_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        beamline: FieldMapBeamline,
        /,
        *,
        entrance_plane: float,
        exit_plane: float,
        time_step: float,
        step_count: int,
        method: RelativisticPusher = "boris",
        convention: AcceleratorConvention | None = None,
        reference_time: float = 0.0,
        maximum_trajectory_bytes: int = 2**31,
    ) -> None:
        if not isinstance(beamline, FieldMapBeamline):
            raise TypeError("beamline must be a FieldMapBeamline.")
        entrance = finite_real_scalar(entrance_plane, "entrance_plane")
        exit_ = finite_real_scalar(exit_plane, "exit_plane")
        if not exit_ > entrance:
            raise ValueError("exit_plane must lie downstream of entrance_plane.")
        step = positive_finite_float(time_step, "time_step")
        steps = positive_integer(step_count, "step_count")
        if steps < 2:
            raise ValueError("step_count must be at least two.")
        convention_ = AcceleratorConvention() if convention is None else convention
        if not isinstance(convention_, AcceleratorConvention):
            raise TypeError("convention must be an AcceleratorConvention.")
        sign = _late_sign(convention_)
        pusher = RelativisticPushPlan(beamline.scale.relativity, method=method)
        reference = finite_real_scalar(reference_time, "reference_time")
        budget = positive_integer(maximum_trajectory_bytes, "maximum_trajectory_bytes")
        self.beamline = beamline
        self.pusher = pusher
        self.convention = convention_
        self.entrance_plane = entrance
        self.exit_plane = exit_
        self.time_step = step
        self.step_count = steps
        self.reference_time = reference
        self.late_sign = sign
        self.maximum_trajectory_bytes = budget
        self.plan_id = canonical_fingerprint(
            {
                "kind": "field-map-tracking-plan",
                "beamline": beamline.source_id,
                "pusher": pusher.plan_id,
                "convention": convention_.convention_id,
                "entrance_plane": entrance,
                "exit_plane": exit_,
                "time_step": step,
                "step_count": steps,
                "reference_time": reference,
                "maximum_trajectory_bytes": budget,
            }
        )

    def trajectory_bytes(self, particle_count: int, /) -> int:
        """Recorded trajectory bytes: time, position, velocity, acceleration, activity."""
        count = positive_integer(particle_count, "particle_count")
        samples = self.step_count * count
        return samples * (_TRAJECTORY_CHANNELS * 8 + 1)


class FieldMapTrackingEvidence(StrictModule):
    """Support, exit-orbit, and integrity evidence of one tracking run.

    ``status`` holds ``FieldMapTrackingStatus`` bits over particles that were
    active and valid at the entrance. ``supported`` is false for a lane that
    left the beamline field support before crossing the exit plane (the lane
    is deactivated there, never tracked through zero field). ``exited`` marks
    lanes that crossed the exit plane within the step budget; ``exit_time``,
    ``exit_position``, and ``exit_proper_velocity`` hold their crossing state
    in the map frame (NaN otherwise). ``maximum_field`` is the largest sampled
    ``|B|`` on supported nodes.
    """

    __strict_contract__ = True

    status: Int32[Scalar]
    accepted: Bool[Scalar]
    finite: Bool[Scalar]
    subluminal: Bool[Scalar]
    entrance_valid: Bool[_ParticleDim]
    supported: Bool[_ParticleDim]
    exited: Bool[_ParticleDim]
    exit_time: Float64[_ParticleDim]
    exit_position: Float64[_ParticleDim, Literal[3]]
    exit_proper_velocity: Float64[_ParticleDim, Literal[3]]
    maximum_field: Float64[Scalar]
    particle_count: Size[_ParticleDim] = eqx.field(static=True)
    trajectory_bytes: int = eqx.field(static=True)


class FieldMapTrackingResult(StrictModule):
    """Exit-plane bunch, lab-time trajectory, and evidence.

    ``bunch`` holds exit-plane accelerator coordinates; lanes that did not
    exit, left support, or entered invalid are inactive. ``trajectory`` holds
    per-lane lab times, node-centered proper velocities and accelerations of
    the leapfrog, one lane per bunch slot with ``multiplicities = weights``.
    """

    bunch: AcceleratorBunch
    trajectory: ChargedTrajectory
    evidence: FieldMapTrackingEvidence
    plan_id: str = eqx.field(static=True)


class _Entrance(NamedTuple):
    times: Array
    positions: Array
    proper_velocities: Array
    valid: Array


class _Carry(NamedTuple):
    time: Array
    position: Array
    previous_half: Array
    next_half: Array
    alive: Array
    left_support: Array
    exited: Array
    exit_time: Array
    exit_position: Array
    exit_proper_velocity: Array
    finite: Array
    subluminal: Array
    maximum_field: Array


class _Record(NamedTuple):
    time: Array
    position: Array
    proper_velocity: Array
    proper_acceleration: Array
    active: Array


def _entrance(
    plan: FieldMapTrackingPlan,
    bunch: AcceleratorBunch,
    speed_of_light: float,
    reference_proper_velocity: Array,
    reference_beta: Array,
    /,
) -> _Entrance:
    x, px, y, py, zeta, delta = (bunch.coordinates[:, index] for index in range(6))
    longitudinal_sq = (1.0 + delta) ** 2 - px * px - py * py
    valid = bunch.active & bunch.valid & (longitudinal_sq > 0.0)
    longitudinal = jnp.sqrt(jnp.where(valid, longitudinal_sq, 1.0))
    proper = reference_proper_velocity * jnp.stack((px, py, longitudinal), axis=-1)
    positions = jnp.stack((x, y, jnp.full_like(x, plan.entrance_plane)), axis=-1)
    times = plan.reference_time + plan.late_sign * zeta / (
        reference_beta * speed_of_light
    )
    return _Entrance(
        jnp.where(valid, times, plan.reference_time),
        jnp.where(valid[:, None], positions, 0.0),
        jnp.where(valid[:, None], proper, 0.0),
        valid,
    )


def _field_norm(sample: ExternalFieldSample, alive: Array, /) -> Array:
    norm = jnp.sqrt(jnp.sum(sample.magnetic * sample.magnetic, axis=-1))
    return jnp.max(jnp.where(alive, norm, 0.0), initial=0.0)


def _advance(
    plan: FieldMapTrackingPlan, specific_charge: Array, carry: _Carry, /
) -> tuple[_Carry, _Record]:
    step = plan.time_step
    record = _Record(
        carry.time,
        carry.position,
        0.5 * (carry.previous_half + carry.next_half),
        (carry.next_half - carry.previous_half) / step,
        carry.alive,
    )
    velocity = plan.pusher.velocity(carry.next_half)
    position = carry.position + velocity * step
    time = carry.time + step
    start, end = carry.position[:, 2], position[:, 2]
    crossing = (
        carry.alive & ~carry.exited & (start < plan.exit_plane) & (end >= plan.exit_plane)
    )
    fraction = (plan.exit_plane - start) / jnp.where(crossing, end - start, 1.0)
    exit_time = jnp.where(crossing, carry.time + fraction * step, carry.exit_time)
    exit_position = jnp.where(
        crossing[:, None],
        carry.position + fraction[:, None] * (position - carry.position),
        carry.exit_position,
    )
    exit_proper = jnp.where(
        crossing[:, None], carry.next_half, carry.exit_proper_velocity
    )
    exited = carry.exited | crossing
    sample = plan.beamline.external_fields(position, time)
    alive = carry.alive & sample.support
    pushed = plan.pusher.push(
        carry.next_half,
        sample.electric,
        sample.magnetic,
        specific_charge,
        alive,
        step,
    )
    left = carry.left_support | (carry.alive & ~alive & ~exited)
    finite = (
        carry.finite
        & pushed.finite
        & jnp.all(jnp.where(alive[:, None], jnp.isfinite(position), True))
    )
    return (
        _Carry(
            time,
            jnp.where(alive[:, None], position, carry.position),
            carry.next_half,
            pushed.proper_velocity,
            alive,
            left,
            exited,
            exit_time,
            exit_position,
            exit_proper,
            finite,
            carry.subluminal & pushed.subluminal,
            jnp.maximum(carry.maximum_field, _field_norm(sample, alive)),
        ),
        record,
    )


def _status(
    finite: Array,
    subluminal: Array,
    entrance: _Entrance,
    requested: Array,
    supported: Array,
    exited: Array,
    /,
) -> Array:
    tracked = entrance.valid
    flags = (
        jnp.where(finite, 0, int(FieldMapTrackingStatus.NONFINITE))
        | jnp.where(subluminal, 0, int(FieldMapTrackingStatus.SUPERLUMINAL))
        | jnp.where(
            jnp.any(tracked & ~supported), int(FieldMapTrackingStatus.UNSUPPORTED), 0
        )
        | jnp.where(
            jnp.any(tracked & supported & ~exited),
            int(FieldMapTrackingStatus.NOT_EXITED),
            0,
        )
        | jnp.where(
            jnp.any(requested & ~tracked),
            int(FieldMapTrackingStatus.INVALID_ENTRANCE),
            0,
        )
    )
    return flags.astype(jnp.int32)


def _exit_bunch(
    plan: FieldMapTrackingPlan,
    bunch: AcceleratorBunch,
    carry: _Carry,
    arrived: Array,
    reference_proper_velocity: Array,
    reference_beta: Array,
    speed_of_light: float,
    /,
) -> AcceleratorBunch:
    relative = carry.exit_proper_velocity / reference_proper_velocity
    delay = (
        carry.exit_time
        - plan.reference_time
        - (plan.exit_plane - plan.entrance_plane) / (reference_beta * speed_of_light)
    )
    coordinates = jnp.stack(
        (
            carry.exit_position[:, 0],
            relative[:, 0],
            carry.exit_position[:, 1],
            relative[:, 1],
            plan.late_sign * reference_beta * speed_of_light * delay,
            jnp.sqrt(jnp.sum(relative * relative, axis=-1)) - 1.0,
        ),
        axis=-1,
    )
    return AcceleratorBunch(
        jnp.where(arrived[:, None], coordinates, jnp.nan),
        bunch.weights,
        bunch.particle_ids,
        active=arrived,
        reference_rest_energy=float(bunch.reference_rest_energy),
        reference_momentum=float(bunch.reference_momentum),
        reference_charge=float(bunch.reference_charge),
        convention=bunch.convention,
        bunch_id=f"{bunch.bunch_id}:{plan.plan_id}",
    )


def track_field_map(
    plan: FieldMapTrackingPlan, bunch: AcceleratorBunch, /
) -> FieldMapTrackingResult:
    """Track ``bunch`` from the entrance to the exit plane in lab time."""
    if not isinstance(plan, FieldMapTrackingPlan):
        raise TypeError("plan must be a FieldMapTrackingPlan.")
    if not isinstance(bunch, AcceleratorBunch):
        raise TypeError("bunch must be an AcceleratorBunch.")
    if bunch.convention.convention_id != plan.convention.convention_id:
        raise ValueError("Tracking plan and bunch coordinate conventions differ.")
    if bunch.coordinates.dtype != jnp.float64:
        raise TypeError("Field-map tracking requires float64 bunch coordinates.")
    required = plan.trajectory_bytes(bunch.capacity)
    if required > plan.maximum_trajectory_bytes:
        raise FieldMapTrackingResourceError(
            "Field-map tracking exceeds maximum_trajectory_bytes: "
            f"required {required}, allowed {plan.maximum_trajectory_bytes}."
        )
    light = plan.pusher.speed_of_light
    rest_energy = bunch.reference_rest_energy
    momentum = bunch.reference_momentum
    mass = rest_energy / light**2
    reference_proper = momentum / mass
    reference_beta = momentum * light / jnp.sqrt((momentum * light) ** 2 + rest_energy**2)
    specific = jnp.full((bunch.capacity,), bunch.reference_charge / mass)
    entrance = _entrance(plan, bunch, light, reference_proper, reference_beta)
    initial = plan.beamline.external_fields(entrance.positions, entrance.times)
    alive = entrance.valid & initial.support
    first = plan.pusher.push(
        entrance.proper_velocities,
        initial.electric,
        initial.magnetic,
        specific,
        alive,
        0.5 * plan.time_step,
    )
    nan = jnp.full((bunch.capacity,), jnp.nan)
    carry = _Carry(
        entrance.times,
        entrance.positions,
        2.0 * entrance.proper_velocities - first.proper_velocity,
        first.proper_velocity,
        alive,
        entrance.valid & ~initial.support,
        jnp.zeros_like(alive),
        nan,
        jnp.stack((nan, nan, nan), axis=-1),
        jnp.stack((nan, nan, nan), axis=-1),
        first.finite,
        first.subluminal,
        _field_norm(initial, alive),
    )
    final, record = jax.lax.scan(
        lambda state, _: _advance(plan, specific, state),
        carry,
        None,
        length=plan.step_count,
    )
    supported = ~final.left_support
    arrived = entrance.valid & supported & final.exited
    status = _status(
        final.finite,
        final.subluminal,
        entrance,
        bunch.active,
        supported,
        final.exited,
    )
    evidence = FieldMapTrackingEvidence(
        status,
        status == 0,
        final.finite,
        final.subluminal,
        entrance.valid,
        supported,
        final.exited,
        final.exit_time,
        final.exit_position,
        final.exit_proper_velocity,
        final.maximum_field,
        bunch.capacity,
        required,
    )
    trajectory = ChargedTrajectory(
        record.time,
        record.position,
        record.proper_velocity,
        jnp.full((bunch.capacity,), bunch.reference_charge, dtype=jnp.float64),
        bunch.weights.astype(jnp.float64),
        record.active,
        (
            jnp.zeros((bunch.capacity,), dtype=jnp.uint32),
            bunch.particle_ids.astype(jnp.uint32),
        ),
        proper_accelerations=record.proper_acceleration,
    )
    return FieldMapTrackingResult(
        _exit_bunch(
            plan,
            bunch,
            final,
            arrived,
            reference_proper,
            reference_beta,
            light,
        ),
        trajectory,
        evidence,
        plan.plan_id,
    )


__all__ = [
    "DipoleBendField",
    "FieldMapBeamline",
    "FieldMapTrackingEvidence",
    "FieldMapTrackingPlan",
    "FieldMapTrackingResourceError",
    "FieldMapTrackingResult",
    "FieldMapTrackingStatus",
    "InsertionDeviceField",
    "InsertionDevicePolarization",
    "InsertionDeviceResonance",
    "TabulatedFieldMap",
    "track_field_map",
]
