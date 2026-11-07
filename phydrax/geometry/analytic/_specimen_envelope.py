#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite specimens: a smooth region clipped by an analytic Box, Cylinder or Sphere.

A specimen is ``Intersection(region, envelope)`` (``SharpCSG``, exact set
semantics). Each envelope is the image of a parameter box under a smooth map
``Psi`` that is a homeomorphism of the closed box onto the closed envelope and
an orientation-preserving diffeomorphism of the open box:

- ``Box``: the affine map of ``[-1, 1]^3``.
- ``Cylinder`` (full): ``[-1, 1]^2 x [0, 1]`` through the elliptical
  square-to-disc map ``(s sqrt(1 - t^2/2), t sqrt(1 - s^2/2))`` times the
  axial coordinate. Its Jacobian determinant is
  ``(1 - s^2/2 - t^2/2) / sqrt((1 - s^2/2)(1 - t^2/2))``, positive except on
  the four lateral lines ``|s| = |t| = 1``.
- ``Sphere``: ``[-1, 1]^3`` through the cube-to-ball map
  ``x sqrt(1 - y^2/2 - z^2/2 + y^2 z^2/3)`` (and cyclic), which maps each cube
  face onto a sphere patch.

The six parameter-box faces are the envelope's surface charts. ``Psi`` of a
face is on the analytic envelope surface, so discovering the region field
``g o Psi`` clipped to the parameter box gives implicit faces inside the
envelope, the creases ``g o Psi = 0`` on each face chart, and trimmed caps.
The map is elementary (products and square roots), so it carries rigorous
interval enclosures through the traced field program.
"""

from __future__ import annotations

import math
from typing import assert_never, final, Literal, NoReturn, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...typing import PRNGKey
from .._capabilities import GeometryCapability
from .._certificate import DistanceSemantics, FieldCertificate, FieldRegularity
from .._contracts import GeometryKernel, GeometryKind, GeometrySource
from .._sampling import RejectionSamplingPlan
from .._validity import GeometryValidityEvidence, representation_validity
from ..design._schema import _ParameterCollector, DesignState
from ._extended import Cylinder
from ._operations import SharpCSG
from ._primitives import _check_points, _feature_id, Box, Sphere


SpecimenEnvelopeKind: TypeAlias = Literal["box", "cylinder", "sphere"]

_CAPABILITIES = frozenset(
    {GeometryCapability.REGION_QUERY, GeometryCapability.BOUNDARY_NORMAL}
)


def _axis_frame(axis: np.ndarray, /) -> np.ndarray:
    """Right-handed orthonormal rows ``(first, second, direction)`` of an axis."""
    direction = axis / float(np.linalg.norm(axis))
    helper = (
        np.asarray((1.0, 0.0, 0.0))
        if abs(direction[0]) < 0.875
        else np.asarray((0.0, 1.0, 0.0))
    )
    first = np.cross(direction, helper)
    first = first / float(np.linalg.norm(first))
    second = np.cross(direction, first)
    return np.stack((first, second, direction))


@final
class SpecimenEnvelope(StrictModule):
    """Parametrization ``Psi`` of an analytic specimen envelope by a parameter box.

    ``parameter_domain`` is the parameter box. ``face_name(axis, side)`` names
    the physical envelope face carrying that parameter face; parameter-box
    edges in ``physical_edges`` (``(axis, side_a, side_b)``: the edge parallel
    to ``axis`` at the given sides of the other two axes, in increasing axis
    order) are sharp envelope edges, the others lie on a smooth surface.
    """

    kind: SpecimenEnvelopeKind = eqx.field(static=True)
    origin: tuple[float, float, float] = eqx.field(static=True)
    frame: tuple[tuple[float, float, float], ...] = eqx.field(static=True)
    scales: tuple[float, float, float] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(self, envelope: Box | Cylinder | Sphere, /) -> None:
        if isinstance(envelope, Box):
            self.kind = "box"
            origin = np.asarray(envelope.center, dtype=np.float64)
            frame = np.eye(3)
            scales = 0.5 * np.asarray(envelope.size, dtype=np.float64)
        elif isinstance(envelope, Cylinder):
            if not envelope.full:
                raise ValueError("A specimen cylinder must be a full cylinder.")
            self.kind = "cylinder"
            axis = np.asarray(envelope.axis, dtype=np.float64)
            radius = float(np.asarray(envelope.radius))
            origin = np.asarray(envelope.base_center, dtype=np.float64)
            frame = _axis_frame(axis)
            scales = np.asarray((radius, radius, float(np.linalg.norm(axis))))
        elif isinstance(envelope, Sphere):
            self.kind = "sphere"
            origin = np.asarray(envelope.center, dtype=np.float64)
            frame = np.eye(3)
            radius = float(np.asarray(envelope.radius))
            scales = np.asarray((radius, radius, radius))
        else:
            raise TypeError("A specimen envelope must be a Box, Cylinder or Sphere.")
        self.origin = (float(origin[0]), float(origin[1]), float(origin[2]))
        self.frame = tuple((float(row[0]), float(row[1]), float(row[2])) for row in frame)
        self.scales = (float(scales[0]), float(scales[1]), float(scales[2]))
        self.source_id = envelope.feature_id

    @property
    def parameter_domain(self) -> np.ndarray:
        match self.kind:
            case "box" | "sphere":
                return np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
            case "cylinder":
                return np.asarray(((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)))
            case _:
                assert_never(self.kind)

    @property
    def volume(self) -> float:
        """Exact envelope volume (rounded to nearest)."""
        a, b, c = self.scales
        match self.kind:
            case "box":
                return 8.0 * a * b * c
            case "cylinder":
                return math.pi * a * a * c
            case "sphere":
                return 4.0 / 3.0 * math.pi * a * a * a
            case _:
                assert_never(self.kind)

    def face_name(self, axis: int, side: int, /) -> str:
        match self.kind:
            case "box":
                return f"{'xyz'[axis]}{'-+'[side]}"
            case "cylinder":
                return ("base", "top")[side] if axis == 2 else "side"
            case "sphere":
                return "sphere"
            case _:
                assert_never(self.kind)

    @property
    def physical_edges(self) -> tuple[tuple[int, int, int], ...]:
        edges = tuple(
            (axis, side_a, side_b)
            for axis in range(3)
            for side_a in range(2)
            for side_b in range(2)
        )
        match self.kind:
            case "box":
                return edges
            case "cylinder":
                # Edges parallel to s or t at w = 0 or 1 lie on the rims.
                return tuple(edge for edge in edges if edge[0] != 2)
            case "sphere":
                return ()
            case _:
                assert_never(self.kind)

    def map(self, parameters: Array, /) -> Array:
        """``Psi``: parameter points ``(..., 3)`` to Cartesian points."""
        p = jnp.asarray(parameters, dtype=jnp.float64)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        match self.kind:
            case "box":
                local = jnp.stack((x, y, z), axis=-1)
            case "cylinder":
                u = x * jnp.sqrt(1.0 - 0.5 * y * y)
                v = y * jnp.sqrt(1.0 - 0.5 * x * x)
                local = jnp.stack((u, v, z), axis=-1)
            case "sphere":
                xx, yy, zz = x * x, y * y, z * z
                local = jnp.stack(
                    (
                        x * jnp.sqrt(1.0 - 0.5 * yy - 0.5 * zz + yy * zz / 3.0),
                        y * jnp.sqrt(1.0 - 0.5 * zz - 0.5 * xx + zz * xx / 3.0),
                        z * jnp.sqrt(1.0 - 0.5 * xx - 0.5 * yy + xx * yy / 3.0),
                    ),
                    axis=-1,
                )
            case _:
                assert_never(self.kind)
        scaled = local * jnp.asarray(self.scales, dtype=jnp.float64)
        frame = jnp.asarray(self.frame, dtype=jnp.float64)
        return jnp.asarray(self.origin, dtype=jnp.float64) + scaled @ frame

    def map_host(self, parameters: np.ndarray, /) -> np.ndarray:
        values = np.asarray(parameters, dtype=np.float64)
        if values.size == 0:
            return np.empty(values.shape, dtype=np.float64)
        return np.asarray(_host_map(self, values), dtype=np.float64)

    def surface_deviation_bound(
        self, triangles: np.ndarray, faces: np.ndarray, /
    ) -> np.ndarray:
        """Upper bound of each flat triangle's distance to its envelope surface.

        ``triangles`` ``(n, 3, 3)`` have vertices on the surface of the
        parameter faces ``faces`` ``(n, 2)`` rows ``(axis, side)``. A planar
        face bounds by vertex rounding only; a curved face (cylinder side,
        sphere) by the radius minus the triangle's distance to the axis or
        centre, which the flat triangle never undercuts.
        """
        points = np.asarray(triangles, dtype=np.float64)
        rows = np.asarray(faces, dtype=np.int64)
        result = np.zeros(points.shape[0], dtype=np.float64)
        relative = points - np.asarray(self.origin)[None, None, :]
        frame = np.asarray(self.frame)
        local = relative @ frame.T
        spacing = np.spacing(np.max(np.abs(points), initial=1.0))
        curved = np.zeros(points.shape[0], dtype=np.bool_)
        match self.kind:
            case "box":
                pass
            case "cylinder":
                curved = rows[:, 0] != 2
                planar = local[curved][:, :, :2]
                result[curved] = self.scales[0] - _distance_to_origin_2d(planar)
            case "sphere":
                curved[:] = True
                result[:] = self.scales[0] - _distance_to_origin_3d(local)
            case _:
                assert_never(self.kind)
        return np.nextafter(np.maximum(result, 0.0) + 16.0 * spacing, np.inf)


@jax.jit
def _host_map(envelope: SpecimenEnvelope, parameters: Array, /) -> Array:
    return envelope.map(parameters)


def _distance_to_origin_2d(triangles: np.ndarray, /) -> np.ndarray:
    """Distance from the origin to filled planar triangles ``(n, 3, 2)``."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    def cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    s1 = cross(b - a, -a)
    s2 = cross(c - b, -b)
    s3 = cross(a - c, -c)
    inside = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
    edges = np.stack(
        (
            _segment_distance(a, b),
            _segment_distance(b, c),
            _segment_distance(c, a),
        ),
        axis=1,
    )
    return np.where(inside, 0.0, np.min(edges, axis=1))


def _segment_distance(a: np.ndarray, b: np.ndarray, /) -> np.ndarray:
    direction = b - a
    length = np.sum(direction * direction, axis=-1)
    t = np.clip(
        np.sum(-a * direction, axis=-1) / np.where(length > 0, length, 1.0), 0.0, 1.0
    )
    closest = a + t[:, None] * direction
    return np.linalg.norm(closest, axis=-1)


def _distance_to_origin_3d(triangles: np.ndarray, /) -> np.ndarray:
    """Distance from the origin to filled triangles ``(n, 3, 3)``."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    normal = np.cross(b - a, c - a)
    norm = np.linalg.norm(normal, axis=-1)
    unit = normal / np.where(norm > 0, norm, 1.0)[:, None]
    height = np.sum(a * unit, axis=-1)
    foot = height[:, None] * unit

    # Barycentric containment of the foot of the perpendicular.
    def side(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.sum(np.cross(v - u, foot - u) * normal, axis=-1)

    inside = (side(a, b) >= 0) & (side(b, c) >= 0) & (side(c, a) >= 0)
    edges = np.stack(
        (_segment_distance(a, b), _segment_distance(b, c), _segment_distance(c, a)),
        axis=1,
    )
    return np.where(inside & (norm > 0), np.abs(height), np.min(edges, axis=1))


def specimen_parts(specimen: SharpCSG, /) -> tuple[GeometrySource, SpecimenEnvelope]:
    """The region and envelope of ``Intersection(region, envelope)``."""
    if not isinstance(specimen, SharpCSG) or specimen.operation != "intersection":
        raise TypeError("A specimen is a SharpCSG intersection.")
    if len(specimen.children) != 2:
        raise ValueError("A specimen intersects one region with one envelope.")
    envelopes = [
        child for child in specimen.children if isinstance(child, (Box, Cylinder, Sphere))
    ]
    if len(envelopes) != 1:
        raise ValueError("A specimen needs exactly one Box, Cylinder or Sphere envelope.")
    envelope = envelopes[0]
    region = next(child for child in specimen.children if child is not envelope)
    return region, SpecimenEnvelope(envelope)


@final
class SpecimenParameterField(GeometrySource):
    """The region field composed with the envelope map, on the parameter box.

    Its zero set clipped to ``envelope.parameter_domain`` is the preimage of
    the specimen boundary inside the envelope; it is a discovery input, not a
    physical region.
    """

    region: GeometrySource
    envelope: SpecimenEnvelope
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        region: GeometrySource,
        envelope: SpecimenEnvelope,
        /,
        *,
        feature_id: str | None = None,
    ) -> None:
        if not isinstance(region, GeometrySource):
            raise TypeError("region must be a GeometrySource.")
        if not isinstance(envelope, SpecimenEnvelope):
            raise TypeError("envelope must be a SpecimenEnvelope.")
        self.region = region
        self.envelope = envelope
        self.feature_id = _feature_id(feature_id, "specimen-parameter-field")

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        child = self.region._compile(context)
        if child.kind is not GeometryKind.REGION or child.ambient_dimension != 3:
            raise ValueError("A specimen region must be a three-dimensional region.")
        return _SpecimenParameterKernel(child, self.envelope)


class _SpecimenParameterKernel(GeometryKernel):
    child: GeometryKernel
    envelope: SpecimenEnvelope

    def __init__(self, child: GeometryKernel, envelope: SpecimenEnvelope, /) -> None:
        self.child = child
        self.envelope = envelope

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 3

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return _CAPABILITIES

    @property
    def field_certificate(self) -> FieldCertificate:
        child = self.child.field_certificate
        return FieldCertificate(
            zero_set_accuracy=child.zero_set_accuracy,
            sign_reliability=child.sign_reliability,
            distance_semantics=DistanceSemantics.LEVEL_SET,
            regularity=(
                FieldRegularity.NONSMOOTH
                if child.regularity is FieldRegularity.NONSMOOTH
                else child.regularity
            ),
            safe_step_factor=None,
            validity_region=(
                f"specimen parameter box of a {self.envelope.kind} envelope"
            ),
            parameter_differentiable=False,
            provenance=(*child.provenance, f"specimen_{self.envelope.kind}_map"),
        )

    def geometry_validity(self, state: DesignState, /) -> GeometryValidityEvidence:
        return representation_validity(self.child, state)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 3)
        return self.child.boundary_field(state, self.envelope.map(points_))

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 3)

        def field(point: Array) -> Array:
            return self.boundary_field(state, point[None, :])[0]

        gradient = jax.vmap(jax.grad(field))(points_)
        norm = jnp.linalg.norm(gradient, axis=-1, keepdims=True)
        return gradient / jnp.maximum(norm, jnp.finfo(jnp.float64).tiny)

    def bounds(self, state: DesignState, /) -> Array:
        del state
        return jnp.asarray(self.envelope.parameter_domain, dtype=jnp.float64)

    def measure(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("A parameter field has no physical measure.")

    def boundary_measure(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("A parameter field has no physical boundary measure.")

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: PRNGKey,
        plan: RejectionSamplingPlan | None = None,
    ) -> NoReturn:
        del state, num_points, key, plan
        raise NotImplementedError("A parameter field has no interior sampler.")

    def sample_boundary(
        self, state: DesignState, num_points: int, /, *, key: PRNGKey
    ) -> NoReturn:
        del state, num_points, key
        raise NotImplementedError("A parameter field has no boundary sampler.")

    def boundary_atlas(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("A parameter field has no boundary atlas.")


__all__ = [
    "SpecimenEnvelope",
    "SpecimenEnvelopeKind",
    "SpecimenParameterField",
    "specimen_parts",
]
