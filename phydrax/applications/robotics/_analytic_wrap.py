#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-branch analytic sphere and planar-cylinder route wrapping.

The geometry is independently derived from OpenSim Core wrap semantics at commit
``86b30588374650fbaf012a345a836a64f6855522``.  A prepared branch is smooth only
while intersection, tangent-pair, and short/long-route choices remain fixed.
"""

from __future__ import annotations

from enum import IntFlag
from math import isfinite
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


WrapSense = Literal["short", "long"]
_OPEN_SIM_REVISION = "86b30588374650fbaf012a345a836a64f6855522"


class AnalyticWrapStatus(IntFlag):
    SUCCESS = 0
    ENDPOINT_INSIDE = 1
    DEGENERATE_GEOMETRY = 2
    NO_WRAP = 4
    TANGENT_PAIR_TIE = 8
    OUTSIDE_BOUNDED_LATERAL_SURFACE = 16
    NONPLANAR_CYLINDER_ROUTE = 32
    NONFINITE = 64


def _vector3(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return array


def _positive(value: ArrayLike, name: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise ValueError(f"{name} must be scalar.")
    if not jnp.issubdtype(scalar.dtype, jnp.inexact):
        scalar = scalar.astype(float)
    concrete = np.asarray(scalar)
    if not np.isfinite(concrete) or concrete <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return scalar


def _dot(left: Array, right: Array, /) -> Array:
    return contract("i,i->", left, right)


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.maximum(_dot(value, value), 0.0))


def _unit(value: Array, tolerance: float, /) -> tuple[Array, Array]:
    magnitude = _norm(value)
    safe = jnp.where(magnitude > tolerance, magnitude, 1.0)
    return value / safe, magnitude


def _rotation(vector: Array, axis: Array, angle: Array, /) -> Array:
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    return (
        vector * cosine
        + jnp.cross(axis, vector) * sine
        + axis * _dot(axis, vector) * (1.0 - cosine)
    )


class AnalyticWrapEvidence(StrictModule, NonTrainableState):
    status: Array
    applied: Array
    finite: Array
    fixed_branch_gradient_supported: Array
    event_margin: Array
    endpoint_tangency_residual: Array
    surface_residual: Array
    source_revision: str = eqx.field(static=True, default=_OPEN_SIM_REVISION)

    @property
    def successful(self) -> Array:
        fatal = int(
            AnalyticWrapStatus.ENDPOINT_INSIDE
            | AnalyticWrapStatus.DEGENERATE_GEOMETRY
            | AnalyticWrapStatus.TANGENT_PAIR_TIE
            | AnalyticWrapStatus.NONPLANAR_CYLINDER_ROUTE
            | AnalyticWrapStatus.NONFINITE
        )
        return jnp.bitwise_and(self.status, fatal) == 0


class AnalyticWrapEvaluation(StrictModule, NonTrainableState):
    tangent_start_m: Array
    tangent_end_m: Array
    surface_points_m: Array
    surface_mask: Array
    surface_length_m: Array
    total_length_m: Array
    signed_surface_angle_rad: Array
    evidence: AnalyticWrapEvidence
    prepared_id: str = eqx.field(static=True)


class SphereRouteWrapPlan(StrictModule, NonTrainableState):
    sample_count: int = eqx.field(static=True)
    sense: WrapSense = eqx.field(static=True)
    mandatory: bool = eqx.field(static=True)
    event_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_count: int = 32,
        /,
        *,
        sense: WrapSense = "short",
        mandatory: bool = False,
        event_tolerance: float = 1.0e-8,
    ):
        count = int(sample_count)
        if count < 2:
            raise ValueError("sample_count must be at least two.")
        if sense not in ("short", "long"):
            raise ValueError("sense must be 'short' or 'long'.")
        tolerance = float(event_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("event_tolerance must be positive and finite.")
        self.sample_count = count
        self.sense = sense
        self.mandatory = bool(mandatory)
        self.event_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "opensim-sphere-route-wrap-plan",
                "source_revision": _OPEN_SIM_REVISION,
                "sample_count": count,
                "sense": sense,
                "mandatory": self.mandatory,
                "event_tolerance": tolerance.hex(),
            }
        )

    def prepare(
        self, center_m: ArrayLike, radius_m: ArrayLike, /
    ) -> PreparedSphereRouteWrap:
        return PreparedSphereRouteWrap(
            self,
            _vector3(center_m, "center_m"),
            _positive(radius_m, "radius_m"),
            canonical_fingerprint(
                {"kind": "prepared-sphere-route-wrap", "plan": self.plan_id}
            ),
        )


class PreparedSphereRouteWrap(StrictModule):
    plan: SphereRouteWrapPlan
    center_m: Array
    radius_m: Array
    prepared_id: str = eqx.field(static=True)

    def evaluate(
        self, endpoint_start_m: ArrayLike, endpoint_end_m: ArrayLike, /
    ) -> AnalyticWrapEvaluation:
        start = _vector3(endpoint_start_m, "endpoint_start_m")
        end = _vector3(endpoint_end_m, "endpoint_end_m")
        tolerance = self.plan.event_tolerance
        first = start - self.center_m
        second = end - self.center_m
        first_distance = _norm(first)
        second_distance = _norm(second)
        inside = (first_distance <= self.radius_m) | (
            second_distance <= self.radius_m
        )
        normal, normal_magnitude = _unit(jnp.cross(first, second), tolerance)
        degenerate = normal_magnitude <= tolerance

        segment = end - start
        segment_square = _dot(segment, segment)
        offset = start - self.center_m
        quadratic_b = 2.0 * _dot(offset, segment)
        quadratic_c = _dot(offset, offset) - self.radius_m**2
        discriminant = quadratic_b**2 - 4.0 * segment_square * quadratic_c
        safe_segment_square = jnp.maximum(segment_square, tolerance**2)
        root_scale = 2.0 * safe_segment_square
        root_offset = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        root_first = (-quadratic_b - root_offset) / root_scale
        root_second = (-quadratic_b + root_offset) / root_scale
        intersects = (
            (discriminant >= 0.0)
            & (root_first > 0.0)
            & (root_first < 1.0)
            & (root_second > 0.0)
            & (root_second < 1.0)
        )
        would_wrap = intersects | self.plan.mandatory
        apply_wrap = would_wrap & ~inside & ~degenerate

        def endpoint_tangents(vector: Array, distance: Array) -> Array:
            safe_square = jnp.maximum(distance**2, tolerance**2)
            base = self.radius_m**2 / safe_square * vector
            transverse = jnp.cross(normal, vector)
            coefficient = (
                self.radius_m
                * jnp.sqrt(jnp.maximum(distance**2 - self.radius_m**2, 0.0))
                / safe_square
            )
            return jnp.stack((base - coefficient * transverse, base + coefficient * transverse))

        first_tangents = endpoint_tangents(first, first_distance)
        second_tangents = endpoint_tangents(second, second_distance)
        first_candidates = jnp.repeat(first_tangents, 2, axis=0)
        second_candidates = jnp.tile(second_tangents, (2, 1))
        first_units = first_candidates / self.radius_m
        second_units = second_candidates / self.radius_m
        cosines = jnp.clip(
            jnp.sum(first_units * second_units, axis=-1), -1.0, 1.0
        )
        angles = jnp.arccos(cosines)
        chosen = jnp.argmin(angles)
        sorted_angles = jnp.sort(angles)
        tie_margin = sorted_angles[1] - sorted_angles[0]
        tangent_start_local = first_candidates[chosen]
        tangent_end_local = second_candidates[chosen]
        short_angle = angles[chosen]
        orientation = jnp.sign(
            _dot(normal, jnp.cross(tangent_start_local, tangent_end_local))
        )
        orientation = jnp.where(orientation == 0.0, 1.0, orientation)
        signed_angle = orientation * short_angle
        if self.plan.sense == "long":
            signed_angle = -orientation * (2.0 * jnp.pi - short_angle)
        parameters = jnp.linspace(
            0.0, 1.0, self.plan.sample_count, dtype=start.dtype
        )
        surface_local = jnp.stack(
            tuple(
                _rotation(tangent_start_local, normal, fraction * signed_angle)
                for fraction in parameters
            )
        )
        surface = surface_local + self.center_m
        tangent_start = tangent_start_local + self.center_m
        tangent_end = tangent_end_local + self.center_m
        surface_length = self.radius_m * jnp.abs(signed_angle)
        total_length = (
            _norm(start - tangent_start)
            + surface_length
            + _norm(end - tangent_end)
        )
        direct_length = _norm(end - start)
        surface_mask = jnp.full((self.plan.sample_count,), apply_wrap)
        tangent_start = jnp.where(apply_wrap, tangent_start, start)
        tangent_end = jnp.where(apply_wrap, tangent_end, end)
        surface = jnp.where(surface_mask[:, None], surface, 0.0)
        surface_length = jnp.where(apply_wrap, surface_length, 0.0)
        total_length = jnp.where(apply_wrap, total_length, direct_length)
        signed_angle = jnp.where(apply_wrap, signed_angle, 0.0)

        finite = (
            jnp.all(jnp.isfinite(start))
            & jnp.all(jnp.isfinite(end))
            & jnp.all(jnp.isfinite(surface))
            & jnp.isfinite(total_length)
        )
        status = jnp.asarray(int(AnalyticWrapStatus.SUCCESS), dtype=jnp.int32)
        status |= jnp.where(
            inside, int(AnalyticWrapStatus.ENDPOINT_INSIDE), 0
        ).astype(jnp.int32)
        status |= jnp.where(
            would_wrap & degenerate,
            int(AnalyticWrapStatus.DEGENERATE_GEOMETRY),
            0,
        ).astype(jnp.int32)
        status |= jnp.where(
            ~would_wrap & ~inside,
            int(AnalyticWrapStatus.NO_WRAP),
            0,
        ).astype(jnp.int32)
        status |= jnp.where(
            apply_wrap & (tie_margin <= tolerance),
            int(AnalyticWrapStatus.TANGENT_PAIR_TIE),
            0,
        ).astype(jnp.int32)
        status |= jnp.where(
            finite, 0, int(AnalyticWrapStatus.NONFINITE)
        ).astype(jnp.int32)
        fatal = int(
            AnalyticWrapStatus.ENDPOINT_INSIDE
            | AnalyticWrapStatus.DEGENERATE_GEOMETRY
            | AnalyticWrapStatus.TANGENT_PAIR_TIE
            | AnalyticWrapStatus.NONPLANAR_CYLINDER_ROUTE
            | AnalyticWrapStatus.NONFINITE
        )
        successful = jnp.bitwise_and(status, fatal) == 0
        radial_start = tangent_start - self.center_m
        radial_end = tangent_end - self.center_m
        tangency = jnp.maximum(
            jnp.abs(_dot(start - tangent_start, radial_start)),
            jnp.abs(_dot(end - tangent_end, radial_end)),
        )
        surface_residual = jnp.max(
            jnp.abs(
                jnp.sqrt(jnp.sum(surface_local * surface_local, axis=-1))
                - self.radius_m
            )
        )
        scale = jnp.maximum(self.radius_m, tolerance)
        event_margin = jnp.minimum(
            jnp.minimum(first_distance - self.radius_m, second_distance - self.radius_m),
            jnp.minimum(
                normal_magnitude / scale,
                jnp.minimum(jnp.abs(discriminant) / (safe_segment_square * scale**2), tie_margin),
            ),
        )
        evidence = AnalyticWrapEvidence(
            status,
            apply_wrap & successful,
            finite,
            successful & (event_margin > 0.0),
            event_margin,
            tangency,
            surface_residual,
        )
        return AnalyticWrapEvaluation(
            tangent_start,
            tangent_end,
            surface,
            surface_mask,
            surface_length,
            total_length,
            signed_angle,
            evidence,
            self.prepared_id,
        )


class PlanarCylinderRouteWrapPlan(StrictModule, NonTrainableState):
    """OpenSim-style lateral-cylinder route restricted to one axial plane."""

    sample_count: int = eqx.field(static=True)
    sense: WrapSense = eqx.field(static=True)
    mandatory: bool = eqx.field(static=True)
    event_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_count: int = 32,
        /,
        *,
        sense: WrapSense = "short",
        mandatory: bool = False,
        event_tolerance: float = 1.0e-8,
    ):
        sphere_policy = SphereRouteWrapPlan(
            sample_count,
            sense=sense,
            mandatory=mandatory,
            event_tolerance=event_tolerance,
        )
        self.sample_count = sphere_policy.sample_count
        self.sense = sphere_policy.sense
        self.mandatory = sphere_policy.mandatory
        self.event_tolerance = sphere_policy.event_tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "opensim-planar-cylinder-route-wrap-plan",
                "source_revision": _OPEN_SIM_REVISION,
                "sample_count": self.sample_count,
                "sense": self.sense,
                "mandatory": self.mandatory,
                "event_tolerance": self.event_tolerance.hex(),
            }
        )

    def prepare(
        self,
        origin_m: ArrayLike,
        axis: ArrayLike,
        radius_m: ArrayLike,
        length_m: ArrayLike,
        /,
    ) -> PreparedPlanarCylinderRouteWrap:
        axis_value, magnitude = _unit(_vector3(axis, "axis"), self.event_tolerance)
        if float(np.asarray(magnitude)) <= self.event_tolerance:
            raise ValueError("axis must be nonzero.")
        return PreparedPlanarCylinderRouteWrap(
            self,
            _vector3(origin_m, "origin_m"),
            axis_value,
            _positive(radius_m, "radius_m"),
            _positive(length_m, "length_m"),
            canonical_fingerprint(
                {"kind": "prepared-planar-cylinder-route-wrap", "plan": self.plan_id}
            ),
        )


class PreparedPlanarCylinderRouteWrap(StrictModule):
    plan: PlanarCylinderRouteWrapPlan
    origin_m: Array
    axis: Array
    radius_m: Array
    length_m: Array
    prepared_id: str = eqx.field(static=True)

    def evaluate(
        self, endpoint_start_m: ArrayLike, endpoint_end_m: ArrayLike, /
    ) -> AnalyticWrapEvaluation:
        start = _vector3(endpoint_start_m, "endpoint_start_m")
        end = _vector3(endpoint_end_m, "endpoint_end_m")
        start_offset = start - self.origin_m
        end_offset = end - self.origin_m
        start_axial = _dot(start_offset, self.axis)
        end_axial = _dot(end_offset, self.axis)
        axial_difference = end_axial - start_axial
        planar = jnp.abs(axial_difference) <= self.plan.event_tolerance
        common_axial = 0.5 * (start_axial + end_axial)
        outside_lateral = jnp.abs(common_axial) > 0.5 * self.length_m

        radial_start = start_offset - start_axial * self.axis
        radial_end = end_offset - end_axial * self.axis
        radial_start_distance = _norm(radial_start)
        radial_end_distance = _norm(radial_end)
        inside = (radial_start_distance <= self.radius_m) | (
            radial_end_distance <= self.radius_m
        )
        tolerance = self.plan.event_tolerance

        chord = radial_end - radial_start
        chord_square = _dot(chord, chord)
        quadratic_b = 2.0 * _dot(radial_start, chord)
        quadratic_c = _dot(radial_start, radial_start) - self.radius_m**2
        discriminant = quadratic_b**2 - 4.0 * chord_square * quadratic_c
        safe_chord_square = jnp.maximum(chord_square, tolerance**2)
        root_offset = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        root_first = (-quadratic_b - root_offset) / (2.0 * safe_chord_square)
        root_second = (-quadratic_b + root_offset) / (2.0 * safe_chord_square)
        intersects = (
            (discriminant >= 0.0)
            & (root_first > 0.0)
            & (root_first < 1.0)
            & (root_second > 0.0)
            & (root_second < 1.0)
        )
        apply_wrap = (
            (intersects | self.plan.mandatory)
            & planar
            & ~outside_lateral
            & ~inside
        )

        def radial_tangents(vector: Array, distance: Array) -> Array:
            unit = vector / jnp.maximum(distance, tolerance)
            transverse = jnp.cross(self.axis, unit)
            cosine = self.radius_m / jnp.maximum(distance, tolerance)
            sine = jnp.sqrt(jnp.maximum(1.0 - cosine**2, 0.0))
            return self.radius_m * jnp.stack(
                (cosine * unit - sine * transverse, cosine * unit + sine * transverse)
            )

        first_tangents = radial_tangents(radial_start, radial_start_distance)
        second_tangents = radial_tangents(radial_end, radial_end_distance)
        first_candidates = jnp.repeat(first_tangents, 2, axis=0)
        second_candidates = jnp.tile(second_tangents, (2, 1))
        signed_candidates = jnp.arctan2(
            jnp.sum(
                jnp.cross(first_candidates, second_candidates) * self.axis,
                axis=-1,
            ),
            jnp.sum(first_candidates * second_candidates, axis=-1),
        )
        chosen = jnp.argmin(jnp.abs(signed_candidates))
        sorted_angles = jnp.sort(jnp.abs(signed_candidates))
        tie_margin = sorted_angles[1] - sorted_angles[0]
        signed_angle = signed_candidates[chosen]
        orientation = jnp.where(signed_angle >= 0.0, 1.0, -1.0)
        if self.plan.sense == "long":
            signed_angle = signed_angle - orientation * 2.0 * jnp.pi
        first_local = first_candidates[chosen] + common_axial * self.axis
        second_local = second_candidates[chosen] + common_axial * self.axis
        parameters = jnp.linspace(
            0.0, 1.0, self.plan.sample_count, dtype=start.dtype
        )
        surface_local = jnp.stack(
            tuple(
                _rotation(first_candidates[chosen], self.axis, fraction * signed_angle)
                + common_axial * self.axis
                for fraction in parameters
            )
        )
        surface = surface_local + self.origin_m
        tangent_start = first_local + self.origin_m
        tangent_end = second_local + self.origin_m
        surface_length = self.radius_m * jnp.abs(signed_angle)
        total_length = (
            _norm(start - tangent_start)
            + surface_length
            + _norm(end - tangent_end)
        )
        direct_length = _norm(end - start)
        mask = jnp.full((self.plan.sample_count,), apply_wrap)
        tangent_start = jnp.where(apply_wrap, tangent_start, start)
        tangent_end = jnp.where(apply_wrap, tangent_end, end)
        surface = jnp.where(mask[:, None], surface, 0.0)
        surface_length = jnp.where(apply_wrap, surface_length, 0.0)
        total_length = jnp.where(apply_wrap, total_length, direct_length)
        signed_angle = jnp.where(apply_wrap, signed_angle, 0.0)

        finite = (
            jnp.all(jnp.isfinite(start))
            & jnp.all(jnp.isfinite(end))
            & jnp.all(jnp.isfinite(surface))
            & jnp.isfinite(total_length)
        )
        status = jnp.asarray(int(AnalyticWrapStatus.SUCCESS), dtype=jnp.int32)
        status |= jnp.where(
            inside, int(AnalyticWrapStatus.ENDPOINT_INSIDE), 0
        ).astype(jnp.int32)
        status |= jnp.where(
            ~planar, int(AnalyticWrapStatus.NONPLANAR_CYLINDER_ROUTE), 0
        ).astype(jnp.int32)
        status |= jnp.where(
            outside_lateral,
            int(AnalyticWrapStatus.OUTSIDE_BOUNDED_LATERAL_SURFACE),
            0,
        ).astype(jnp.int32)
        status |= jnp.where(
            ~apply_wrap & ~inside & planar & ~outside_lateral,
            int(AnalyticWrapStatus.NO_WRAP),
            0,
        ).astype(jnp.int32)
        status |= jnp.where(
            apply_wrap & (tie_margin <= tolerance),
            int(AnalyticWrapStatus.TANGENT_PAIR_TIE),
            0,
        ).astype(jnp.int32)
        status |= jnp.where(
            finite, 0, int(AnalyticWrapStatus.NONFINITE)
        ).astype(jnp.int32)
        fatal = int(
            AnalyticWrapStatus.ENDPOINT_INSIDE
            | AnalyticWrapStatus.DEGENERATE_GEOMETRY
            | AnalyticWrapStatus.TANGENT_PAIR_TIE
            | AnalyticWrapStatus.NONPLANAR_CYLINDER_ROUTE
            | AnalyticWrapStatus.NONFINITE
        )
        successful = jnp.bitwise_and(status, fatal) == 0
        start_radial_tangent = first_local - common_axial * self.axis
        end_radial_tangent = second_local - common_axial * self.axis
        tangency = jnp.maximum(
            jnp.abs(_dot(start - tangent_start, start_radial_tangent)),
            jnp.abs(_dot(end - tangent_end, end_radial_tangent)),
        )
        radial_surface = surface_local - contract(
            "ni,i->n", surface_local, self.axis
        )[:, None] * self.axis
        surface_residual = jnp.max(
            jnp.abs(
                jnp.sqrt(jnp.sum(radial_surface * radial_surface, axis=-1))
                - self.radius_m
            )
        )
        scale = jnp.maximum(self.radius_m, tolerance)
        event_margin = jnp.minimum(
            jnp.minimum(
                radial_start_distance - self.radius_m,
                radial_end_distance - self.radius_m,
            ),
            jnp.minimum(
                jnp.abs(discriminant) / (safe_chord_square * scale**2),
                jnp.minimum(
                    tie_margin,
                    jnp.minimum(
                        tolerance - jnp.abs(axial_difference),
                        0.5 * self.length_m - jnp.abs(common_axial),
                    ),
                ),
            ),
        )
        evidence = AnalyticWrapEvidence(
            status,
            apply_wrap & successful,
            finite,
            successful & (event_margin > 0.0),
            event_margin,
            tangency,
            surface_residual,
        )
        return AnalyticWrapEvaluation(
            tangent_start,
            tangent_end,
            surface,
            mask,
            surface_length,
            total_length,
            signed_angle,
            evidence,
            self.prepared_id,
        )


__all__ = [
    "AnalyticWrapEvaluation",
    "AnalyticWrapEvidence",
    "AnalyticWrapStatus",
    "PlanarCylinderRouteWrapPlan",
    "PreparedPlanarCylinderRouteWrap",
    "PreparedSphereRouteWrap",
    "SphereRouteWrapPlan",
    "WrapSense",
]
