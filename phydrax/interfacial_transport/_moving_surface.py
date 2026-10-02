#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Moving-surface (ALE) transport of extensive film content on fixed topology.

Three motions are kept apart. The *mesh motion* moves every vertex along the
straight path ``x(t) = x^n + t w`` with ``w = (x^{n+1} - x^n) / dt``; it defines
the discrete surface and its barycentric dual measures at every instant. The
*material motion* ``u`` moves the film; ``material_velocity(u_t)`` builds
``u = u_t + (w.n) n`` from a tangential material velocity with vertex normals
of the midpoint configuration (the ``DDGOperators`` convention). *Surface
deformation* is the change of the discrete surface itself: a tangential mesh
motion that keeps the discrete surface fixed deforms nothing, while normal or
non-rigid motion stretches it.

Discrete geometric conservation law. The barycentric measure
``A_i = sum_f |f| / 3`` obeys the lumped ESFEM transport property
``dA_i/dt = sum_f |f| div_f(w) / 3`` at every instant, with ``div_f`` the P1
tangential divergence on face ``f``. Along straight vertex paths the doubled
face-area vector ``N_f(t) = N_0 + t N_1 + t^2 N_2`` is quadratic, so the rate
integrates in closed form to ``(|N_f(dt)| - |N_f(0)|) / 2``. ``area_rate_m2_s``
is that exact time average, evaluated from the source geometry and ``w``
rather than from the target; the law ``A^{n+1} - A^n = dt * area_rate`` then
holds to roundoff for every motion, non-homothetic and non-planar included.
A residual above the dtype-scaled roundoff policy (for example after
cancellation in far-translated coordinates) refuses the motion with
``SurfaceMotionStatus.GCL_VIOLATED``; nothing is committed silently.

Content. Each barycentric subcell of a face is a material region of the
face's affine mesh motion, so a Lagrangian material (``u = w``) keeps every
cell's content while its measure changes by exactly ``dt * area_rate``.
Material moving relative to the mesh exchanges content through the dual edges
with the in-face part of ``u - w`` at the midpoint geometry (antisymmetric
donor-cell fluxes), so totals are conserved to roundoff. A uniform areal
density stays uniform only under a uniform material dilatation: a tangential
mesh motion of a fixed discrete surface with zero material velocity keeps it
uniform to roundoff, while surface deformation changes it by the physical
dilatation. Geometry is refreshed through ``PreparedFilmSurface.refresh``;
topology is never rebuilt, and fixed-topology motion remains differentiable.
Topology changes use ``SurfaceEpochTransfer``, which carries epoch lineage and
reports no derivative.
"""

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import TopologyEpoch
from ..sparse import EdgeRelation
from ..typing import checked
from ._film_contracts import PreparedFilmSurface
from ._film_evidence import resolve_film_status
from ._film_transport import outflow_courant, upwind_edge_flux


# Roundoff budget of the discrete GCL in units of the dtype epsilon times the
# per-vertex magnitude of the summed face-area terms: a handful of products,
# one square root and at most a dozen incident faces per vertex.
_GCL_ROUNDOFF_FACTOR = 64.0


class SurfaceMotionStatus(IntEnum):
    """Terminal status of a mesh motion or one moving-surface transport.

    Only ``ACCEPTED`` commits. ``GCL_VIOLATED`` means the target dual measures
    differ from the exactly integrated area rate by more than the roundoff
    policy; ``COURANT_LIMIT`` means the donor-cell outflow Courant number
    exceeds one, so positivity is not guaranteed.
    """

    ACCEPTED = 0
    INADMISSIBLE_INPUT = 1
    ORIENTATION_REVERSED = 2
    INADMISSIBLE_MEASURES = 3
    GCL_VIOLATED = 4
    COURANT_LIMIT = 5
    NONFINITE = 6


class SurfaceMotionEvidence(StrictModule):
    """Geometric conservation, orientation and admissibility of one motion.

    ``geometric_conservation_residual_m2`` is ``A^{n+1} - A^n - dt *
    area_rate`` per vertex; ``maximum_relative_gcl_residual`` divides it by the
    per-vertex magnitude of the summed face-area terms, and the motion is
    refused when it exceeds ``relative_gcl_tolerance`` (a fixed multiple of
    the dtype epsilon).
    """

    area_change_m2: Array
    geometric_conservation_residual_m2: Array
    maximum_relative_gcl_residual: Array
    relative_gcl_tolerance: Array
    orientation_preserved: Array
    target_measures_admissible: Array
    target_conductance_admissible: Array
    status: Array

    def __init__(
        self,
        *,
        area_change_m2: Array,
        geometric_conservation_residual_m2: Array,
        maximum_relative_gcl_residual: Array,
        relative_gcl_tolerance: Array,
        orientation_preserved: Array,
        target_measures_admissible: Array,
        target_conductance_admissible: Array,
        status: Array,
    ) -> None:
        self.area_change_m2 = jnp.asarray(area_change_m2)
        self.geometric_conservation_residual_m2 = jnp.asarray(
            geometric_conservation_residual_m2
        )
        self.maximum_relative_gcl_residual = jnp.asarray(maximum_relative_gcl_residual)
        self.relative_gcl_tolerance = jnp.asarray(relative_gcl_tolerance)
        self.orientation_preserved = jnp.asarray(orientation_preserved, dtype=jnp.bool_)
        self.target_measures_admissible = jnp.asarray(
            target_measures_admissible, dtype=jnp.bool_
        )
        self.target_conductance_admissible = jnp.asarray(
            target_conductance_admissible, dtype=jnp.bool_
        )
        self.status = jnp.asarray(status, dtype=jnp.int32)

    @property
    def valid(self) -> Array:
        """Admissible input, orientation, positive measures and exact GCL.

        Transport needs no Delaunay target; diffusive routes on the target
        check ``target_conductance_admissible`` through their own evidence.
        """
        return self.status == SurfaceMotionStatus.ACCEPTED


class SurfaceTransportResult(StrictModule):
    """Moving-surface transport: committed and candidate content with status.

    ``content`` is the candidate when ``status`` is ``ACCEPTED`` and the input
    content otherwise; a refused step leaves the caller on the source
    geometry. ``content_residual`` is the candidate total minus the input
    total.
    """

    content: Array
    candidate_content: Array
    courant_number: Array
    content_residual: Array
    status: Array

    def __init__(
        self,
        content: Array,
        candidate_content: Array,
        courant_number: Array,
        content_residual: Array,
        status: Array,
        /,
    ) -> None:
        self.content = jnp.asarray(content)
        self.candidate_content = jnp.asarray(candidate_content)
        self.courant_number = jnp.asarray(courant_number)
        self.content_residual = jnp.asarray(content_residual)
        self.status = jnp.asarray(status, dtype=jnp.int32)

    @property
    def accepted(self) -> Array:
        return self.status == SurfaceMotionStatus.ACCEPTED


class SurfaceMeshMotion(StrictModule):
    """One fixed-topology mesh motion ``x^n -> x^{n+1}`` over ``dt``.

    ``source`` and ``target`` are prepared film geometries with the same
    topology; ``target.geometry_revision`` is ``source.geometry_revision + 1``.
    ``area_rate_m2_s`` is the exact time-averaged barycentric dual-area rate
    along straight vertex paths (see the module notes).
    """

    source: PreparedFilmSurface
    midpoint: PreparedFilmSurface
    target: PreparedFilmSurface
    step_size: Array
    mesh_velocity: Array
    normal_velocity: Array
    tangential_mesh_velocity: Array
    area_rate_m2_s: Array
    evidence: SurfaceMotionEvidence

    @checked
    def __init__(
        self,
        source: PreparedFilmSurface,
        target_coordinates: ArrayLike,
        step_size_s: ArrayLike,
        /,
    ) -> None:
        coordinates = jnp.asarray(target_coordinates, dtype=jnp.float64)
        if coordinates.shape != source.coordinates.shape:
            raise ValueError("target_coordinates must match the source coordinates.")
        step_size = jnp.asarray(step_size_s, dtype=jnp.float64)
        if step_size.shape != ():
            raise ValueError("step_size_s must be a scalar.")
        velocity = (coordinates - source.coordinates) / step_size
        midpoint = PreparedFilmSurface(
            source.topology,
            0.5 * (source.coordinates + coordinates),
            geometry_revision=source.geometry_revision,
        )
        target = source.refresh(coordinates)
        normal = midpoint.vertex_normal
        normal_speed = jnp.sum(velocity * normal, axis=1)
        face_change, face_scale = _integrated_face_area_change(
            source, velocity, step_size
        )
        faces = source.operators.faces.reshape((-1,))
        integrated = _vertex_thirds(source, faces, face_change)
        scale = _vertex_thirds(source, faces, face_scale)
        change = target.vertex_area - source.vertex_area
        residual = change - integrated
        self.source = source
        self.midpoint = midpoint
        self.target = target
        self.step_size = step_size
        self.mesh_velocity = velocity
        self.normal_velocity = normal_speed
        self.tangential_mesh_velocity = velocity - normal_speed[:, None] * normal
        self.area_rate_m2_s = integrated / step_size
        self.evidence = _motion_evidence(
            source, target, step_size, change, residual, scale
        )

    def material_velocity(self, tangential_velocity_m_s: ArrayLike, /) -> Array:
        """Return ``u_t + (w.n) n``: material moving normally with the surface.

        Use ``mesh_velocity`` itself as the material velocity of a Lagrangian
        mesh whose vertices are material points.
        """
        tangential = jnp.asarray(tangential_velocity_m_s, dtype=jnp.float64)
        if tangential.shape != self.mesh_velocity.shape:
            raise ValueError("Tangential velocity must have shape (num_vertices, 3).")
        normal = self.midpoint.vertex_normal
        tangential = (
            tangential - jnp.sum(tangential * normal, axis=1, keepdims=True) * normal
        )
        return tangential + self.normal_velocity[:, None] * normal

    def relative_area_flux(self, material_velocity_m_s: ArrayLike, /) -> Array:
        """Return dual-edge area fluxes of ``u - w`` at the midpoint geometry.

        Only the in-face components of ``u - w`` cross the barycentric dual
        edges; a vanishing relative velocity (Lagrangian mesh) moves nothing.
        """
        material = jnp.asarray(material_velocity_m_s, dtype=jnp.float64)
        if material.shape != self.mesh_velocity.shape:
            raise ValueError("Material velocity must have shape (num_vertices, 3).")
        return self.midpoint.edge_area_flux(material - self.mesh_velocity)

    def transport(
        self, content: ArrayLike, material_velocity_m_s: ArrayLike, /
    ) -> SurfaceTransportResult:
        """Move extensive vertex content with the relative velocity ``u - w``.

        ``material_velocity_m_s`` is the full material velocity of the surface
        points at the vertices (see ``material_velocity``). Densities are taken
        on the source cells; the candidate lives on the target cells, whose
        measures are ``A^n + dt * area_rate``. The candidate is exactly
        conservative and nonnegative for ``courant_number <= 1``. It is
        committed only when the motion is valid, the Courant number is at
        most one and the result is finite.
        """
        values = jnp.asarray(content, dtype=jnp.float64)
        if values.shape[:1] != (self.source.topology.num_vertices,):
            raise ValueError("content must have a leading vertex axis.")
        area_flux = self.relative_area_flux(material_velocity_m_s)
        area = self.source.vertex_area
        candidate = values - self.step_size * self.midpoint.edge_divergence(
            upwind_edge_flux(self.midpoint, values, area_flux, area)
        )
        courant = outflow_courant(self.midpoint, area_flux, area, self.step_size)
        step_status = resolve_film_status(
            (SurfaceMotionStatus.COURANT_LIMIT, ~(courant <= 1.0)),
            (SurfaceMotionStatus.NONFINITE, ~jnp.all(jnp.isfinite(candidate))),
        )
        motion_status = self.evidence.status
        status = jnp.where(
            motion_status == SurfaceMotionStatus.ACCEPTED, step_status, motion_status
        )
        return SurfaceTransportResult(
            jnp.where(status == SurfaceMotionStatus.ACCEPTED, candidate, values),
            candidate,
            courant,
            jnp.sum(candidate, axis=0) - jnp.sum(values, axis=0),
            status,
        )


def _integrated_face_area_change(
    source: PreparedFilmSurface, velocity: Array, step_size: Array, /
) -> tuple[Array, Array]:
    """Return ``int_0^dt |f| div_f(w) dt`` per face and its roundoff magnitude.

    With edge vectors ``e_k(t) = e_k + t d_k`` the doubled area vector is
    ``N(t) = N_0 + t N_1 + t^2 N_2``; the integral is ``(|N(dt)| - |N_0|) / 2``.
    """
    faces = source.operators.faces
    corners = source.coordinates[faces]
    rates = velocity[faces]
    first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    first_rate, second_rate = rates[:, 1] - rates[:, 0], rates[:, 2] - rates[:, 0]
    constant = jnp.cross(first, second)
    linear = jnp.cross(first, second_rate) + jnp.cross(first_rate, second)
    quadratic = jnp.cross(first_rate, second_rate)
    final = constant + step_size * (linear + step_size * quadratic)
    initial_norm = jnp.linalg.norm(constant, axis=1)
    change = 0.5 * (jnp.linalg.norm(final, axis=1) - initial_norm)
    magnitude = 0.5 * (
        initial_norm
        + jnp.abs(step_size) * jnp.linalg.norm(linear, axis=1)
        + step_size**2 * jnp.linalg.norm(quadratic, axis=1)
    )
    return change, magnitude


def _vertex_thirds(
    surface: PreparedFilmSurface, flat_faces: Array, face_values: Array, /
) -> Array:
    """Return ``sum_f value_f / 3`` over the faces incident to every vertex."""
    return (
        jnp.zeros((surface.topology.num_vertices,), dtype=face_values.dtype)
        .at[flat_faces]
        .add(jnp.repeat(face_values / 3.0, 3))
    )


def _motion_evidence(
    source: PreparedFilmSurface,
    target: PreparedFilmSurface,
    step_size: Array,
    change: Array,
    residual: Array,
    scale: Array,
    /,
) -> SurfaceMotionEvidence:
    """Resolve the motion status from input, orientation, measures and GCL."""
    orientation = jnp.all(
        jnp.sum(source.operators.face_normal * target.operators.face_normal, axis=1) > 0.0
    )
    tolerance = jnp.asarray(
        _GCL_ROUNDOFF_FACTOR * jnp.finfo(residual.dtype).eps,
        dtype=residual.dtype,
    )
    relative = jnp.max(jnp.abs(residual) / (scale + target.vertex_area))
    finite_input = (
        jnp.all(jnp.isfinite(target.coordinates))
        & jnp.isfinite(step_size)
        & (step_size > 0.0)
    )
    status = resolve_film_status(
        (SurfaceMotionStatus.INADMISSIBLE_INPUT, ~finite_input),
        (SurfaceMotionStatus.ORIENTATION_REVERSED, ~orientation),
        (
            SurfaceMotionStatus.INADMISSIBLE_MEASURES,
            ~target.evidence.measures_admissible,
        ),
        (SurfaceMotionStatus.GCL_VIOLATED, ~(relative <= tolerance)),
    )
    return SurfaceMotionEvidence(
        area_change_m2=change,
        geometric_conservation_residual_m2=residual,
        maximum_relative_gcl_residual=relative,
        relative_gcl_tolerance=tolerance,
        orientation_preserved=orientation,
        target_measures_admissible=target.evidence.measures_admissible,
        target_conductance_admissible=target.evidence.conductance_admissible,
        status=status,
    )


class SurfaceEpochTransferResult(StrictModule):
    content: Array
    conservation_residual: Array
    derivative_available: Array

    def __init__(
        self, content: Array, conservation_residual: Array, derivative_available: Array, /
    ) -> None:
        self.content = jnp.asarray(content)
        self.conservation_residual = jnp.asarray(conservation_residual)
        self.derivative_available = jnp.asarray(derivative_available, dtype=jnp.bool_)


class SurfaceEpochTransfer(StrictModule):
    """Conservative sparse transfer of extensive film content across a remesh.

    Routes ``(target_vertex, source_vertex, weight)`` split every source cell's
    content with nonnegative weights summing to one, so totals are conserved
    and nonnegativity is preserved. The transfer is a host epoch transition
    with ``TopologyEpoch`` lineage; it is not differentiable.
    """

    source_epoch: TopologyEpoch
    target_epoch: TopologyEpoch
    relation: EdgeRelation
    weights: Array
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_epoch: TopologyEpoch,
        target_epoch: TopologyEpoch,
        target_indices: ArrayLike,
        source_indices: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        source_size: int,
        target_size: int,
    ) -> None:
        if not isinstance(source_epoch, TopologyEpoch) or not isinstance(
            target_epoch, TopologyEpoch
        ):
            raise TypeError("Transfer endpoints must be TopologyEpoch values.")
        if target_epoch.index != source_epoch.index + 1:
            raise ValueError("Epoch transfers must connect consecutive epochs.")
        target = np.asarray(target_indices, dtype=np.int64).reshape((-1,))
        source = np.asarray(source_indices, dtype=np.int64).reshape((-1,))
        weight = np.asarray(weights, dtype=np.float64).reshape((-1,))
        if not (target.shape == source.shape == weight.shape):
            raise ValueError("Transfer routes must have aligned indices and weights.")
        if (
            np.any(source < 0)
            | np.any(source >= source_size)
            | np.any(target < 0)
            | np.any(target >= target_size)
        ):
            raise ValueError("Transfer routes reference vertices outside the meshes.")
        if not np.all(np.isfinite(weight) & (weight >= 0.0)):
            raise ValueError("Transfer weights must be finite and nonnegative.")
        totals = np.bincount(source, weights=weight, minlength=source_size)
        if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("Every source cell must transfer all of its content.")
        self.source_epoch = source_epoch
        self.target_epoch = target_epoch
        self.relation = EdgeRelation(
            source.astype(np.int32),
            target.astype(np.int32),
            source_size=source_size,
            target_size=target_size,
        )
        self.weights = jnp.asarray(weight)
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "surface-epoch-transfer",
                "source": source_epoch.epoch_id,
                "target": target_epoch.epoch_id,
                "routes": (target, source, weight),
            }
        )

    def apply(self, content: ArrayLike, /) -> SurfaceEpochTransferResult:
        """Transfer extensive content (leading source-vertex axis)."""
        values = jnp.asarray(content, dtype=jnp.float64)
        if values.shape[:1] != (self.relation.source_size,):
            raise ValueError("content must have a leading source-vertex axis.")
        routed = (
            self.weights.reshape((-1,) + (1,) * (values.ndim - 1))
            * values[self.relation.source_indices]
        )
        moved = jnp.zeros((self.relation.target_size, *values.shape[1:]), values.dtype)
        moved = moved.at[self.relation.target_indices].add(routed)
        return SurfaceEpochTransferResult(
            moved,
            jnp.sum(moved, axis=0) - jnp.sum(values, axis=0),
            jnp.asarray(False),
        )


__all__ = [
    "SurfaceEpochTransfer",
    "SurfaceEpochTransferResult",
    "SurfaceMeshMotion",
    "SurfaceMotionStatus",
    "SurfaceMotionEvidence",
    "SurfaceTransportResult",
]
