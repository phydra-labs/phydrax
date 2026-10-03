# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Typed moving-surface geometry laws and the measure-rate source identity.

A surface moving with mesh velocity ``u = V_n n + u_tau`` changes each point
measure at the rate ``dw/dt = w (H V_n + div_G u_tau)`` with ``H`` the trace of
the shape operator (``+2/R`` on an outward sphere). The normal speed is
physical and shared by material and mesh; the tangential mesh component is a
parametrization choice. Every law here moves the mesh with the material
tangential velocity or, for ``"normal-only"`` laws, with none. A mesh shift
adds a tangential mesh velocity that the material does not follow; its
relative velocity is transported by the conservative shift owner.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import AbstractLinearOperator, ScaledLinearOperator
from ...typing import Bool, checked, Float64, parse, Scalar
from .._views import PreparedFieldReconstruction
from ._capacity import ActivePointDim
from ._shifting import ShiftAmbientDim
from ._surface import PreparedSurfacePointCloud, SurfaceRefreshResult
from ._surface_geometry import SurfaceGeometryStatus


SurfaceTangentialMode: TypeAlias = Literal["material", "normal-only"]
"""``material`` moves the mesh with the material tangential velocity;
``normal-only`` declares the physical motion purely normal and discards the
tangential component of the supplied velocity."""

SurfaceRefreshMode: TypeAlias = Literal["fixed-support", "similarity"]
"""``fixed-support`` refreshes the stage points directly; ``similarity``
refreshes them modulo translation and isotropic scale of the support epoch."""


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


@final
class MovingGeometryRefresh(StrictModule):
    """Native surface geometry and operators at one stage point set.

    ``diffusion`` maps concentration to the EXTENSIVE diffusion rate and must
    have zero column sums; it is supplied by a conservative owner, not the
    generally nonconservative strong Laplace--Beltrami stencil.
    ``surface_divergence`` maps an ambient vector field to its strong surface
    divergence and ``laplace_beltrami`` acts on scalar fields. A sampled tube
    estimate is never a certificate.
    """

    __strict_contract__ = True
    points: Float64[ActivePointDim, ShiftAmbientDim]
    measures: Float64[ActivePointDim]
    normals: Float64[ActivePointDim, ShiftAmbientDim]
    mean_curvature: Float64[ActivePointDim]
    diffusion: AbstractLinearOperator
    surface_divergence: AbstractLinearOperator
    laplace_beltrami: AbstractLinearOperator
    trust_valid: Bool[Scalar]
    tube_valid: Bool[Scalar]
    successful: Bool[Scalar]

    @classmethod
    @checked
    def from_surface(
        cls, refresh: SurfaceRefreshResult, diffusion: AbstractLinearOperator, /
    ) -> MovingGeometryRefresh:
        """Consume one native fixed-support surface refresh without a second fit.

        The tube is admitted only when the surface plan does not require it or
        its source certifies every refreshed point.
        """
        prepared = refresh.prepared
        evidence = refresh.geometry_evidence
        tube_valid = jnp.asarray(not prepared.plan.require_tube) | (
            jnp.asarray(evidence.tube_certified) & jnp.all(evidence.tube_valid)
        )
        return cls(
            refresh.points.astype(jnp.float64),
            refresh.measures.astype(jnp.float64),
            refresh.normals.astype(jnp.float64),
            prepared.geometry.mean_curvature.astype(jnp.float64),
            diffusion,
            prepared.strong_surface_divergence,
            prepared.laplace_beltrami,
            jnp.all(refresh.status != int(SurfaceGeometryStatus.SUPPORT_INVALID)),
            tube_valid,
            refresh.accepted,
        )

    @classmethod
    @checked
    def from_similarity_surface(
        cls,
        surface: PreparedSurfacePointCloud,
        points: ArrayLike,
        diffusion: AbstractLinearOperator,
        /,
    ) -> MovingGeometryRefresh:
        """Refresh modulo translation and isotropic scale of the support epoch.

        Points are mapped to the reference centroid and RMS radius, refreshed
        there, and mapped back with the exact similarity weights: measures
        scale with ``rho**d`` for intrinsic dimension ``d``, curvature and
        divergence with ``1/rho`` and Laplace--Beltrami with ``1/rho**2``.
        The fixed-support trust certificate therefore bounds only the
        non-similar deformation; it is not widened.
        """
        current = jnp.asarray(points, dtype=jnp.float64)
        reference = surface.reference_geometry.points.astype(jnp.float64)
        if current.shape != reference.shape:
            raise ValueError("Similarity refresh preserves the support point layout.")
        current_center = jnp.mean(current, axis=0)
        reference_center = jnp.mean(reference, axis=0)
        scale = jnp.sqrt(
            jnp.mean(jnp.sum((current - current_center) ** 2, axis=1))
        ) / jnp.sqrt(jnp.mean(jnp.sum((reference - reference_center) ** 2, axis=1)))
        safe_scale = jnp.where(
            jnp.isfinite(scale) & (scale > 0), scale, jnp.ones_like(scale)
        )
        shape = reference_center + (current - current_center) / safe_scale
        refreshed = cls.from_surface(surface.refresh(shape), diffusion)
        intrinsic_dimension = surface.geometry.tangent_frames.shape[-1]
        return cls(
            current_center + safe_scale * (refreshed.points - reference_center),
            refreshed.measures * safe_scale**intrinsic_dimension,
            refreshed.normals,
            refreshed.mean_curvature / safe_scale,
            diffusion,
            ScaledLinearOperator(refreshed.surface_divergence, 1 / safe_scale),
            ScaledLinearOperator(refreshed.laplace_beltrami, 1 / safe_scale**2),
            refreshed.trust_valid,
            refreshed.tube_valid,
            refreshed.successful & jnp.isfinite(scale) & (scale > 0),
        )


@final
class SurfaceGeometryProvider(StrictModule):
    """Stage geometry of one prepared surface support epoch.

    A callable PyTree: the prepared surface and the conservative diffusion
    operator are dynamic leaves, never captured by a Python closure.
    """

    surface: PreparedSurfacePointCloud
    diffusion: AbstractLinearOperator
    mode: SurfaceRefreshMode = eqx.field(static=True)

    @checked
    def __init__(
        self,
        surface: PreparedSurfacePointCloud,
        diffusion: AbstractLinearOperator,
        /,
        *,
        mode: SurfaceRefreshMode = "fixed-support",
    ) -> None:
        self.surface = surface
        self.diffusion = diffusion
        self.mode = parse(mode, SurfaceRefreshMode, "mode")

    def __call__(self, points: Array, time: Array, args: Any, /) -> MovingGeometryRefresh:
        match self.mode:
            case "fixed-support":
                return MovingGeometryRefresh.from_surface(
                    self.surface.refresh(points), self.diffusion
                )
            case "similarity":
                return MovingGeometryRefresh.from_similarity_surface(
                    self.surface, points, self.diffusion
                )
            case _:
                assert_never(self.mode)


@final
class SurfaceMotion(StrictModule):
    """Normal/tangential split of one stage motion and its measure-rate source.

    ``tangential_velocity`` is the tangential mesh velocity and
    ``relative_velocity`` the material velocity minus the mesh velocity, which
    is tangential and nonzero only under a mesh shift.
    """

    __strict_contract__ = True
    normal_speed: Float64[ActivePointDim]
    tangential_velocity: Float64[ActivePointDim, ShiftAmbientDim]
    mesh_velocity: Float64[ActivePointDim, ShiftAmbientDim]
    relative_velocity: Float64[ActivePointDim, ShiftAmbientDim]
    normal_measure_rate: Float64[ActivePointDim]
    tangential_measure_rate: Float64[ActivePointDim]
    valid: Bool[Scalar]

    @property
    def measure_rate(self) -> Array:
        """``w (H V_n + div_G u_tau)``, the source of the measure evolution."""
        return self.normal_measure_rate + self.tangential_measure_rate

    def with_mesh_shift(
        self, shift: ArrayLike, measure_rate: ArrayLike, /
    ) -> SurfaceMotion:
        """Add a tangential mesh velocity that the material does not follow.

        ``measure_rate`` is the shift's discrete ``w div_G u_s``; the owner of
        the relative content flux supplies it so that both share one discrete
        divergence.
        """
        velocity = jnp.asarray(shift, dtype=jnp.float64)
        rate = jnp.asarray(measure_rate, dtype=jnp.float64)
        if (
            velocity.shape != self.mesh_velocity.shape
            or rate.shape != self.normal_speed.shape
        ):
            raise ValueError(
                "Mesh shift needs one ambient vector and one measure rate per point."
            )
        return SurfaceMotion(
            self.normal_speed,
            self.tangential_velocity + velocity,
            self.mesh_velocity + velocity,
            self.relative_velocity - velocity,
            self.normal_measure_rate,
            self.tangential_measure_rate + rate,
            self.valid & jnp.all(jnp.isfinite(velocity)) & jnp.all(jnp.isfinite(rate)),
        )


class AbstractSurfaceMotionLaw(StrictModule):
    """One typed surface motion law evaluated at a stage geometry and time."""

    tangential: eqx.AbstractVar[SurfaceTangentialMode]
    law_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def velocity(
        self, time: Array, geometry: MovingGeometryRefresh, args: Any, /
    ) -> tuple[Array, Array]:
        """Return the ambient point velocity and the law's own validity."""

    @checked
    def motion(
        self, time: Array, geometry: MovingGeometryRefresh, args: Any, /
    ) -> SurfaceMotion:
        velocity, valid = self.velocity(time, geometry, args)
        velocity = jnp.asarray(velocity, dtype=jnp.float64)
        if velocity.shape != geometry.points.shape:
            raise ValueError("Motion velocity must have one ambient vector per point.")
        normals = geometry.normals
        speed = jnp.sum(velocity * normals, axis=1)
        tangential = velocity - speed[:, None] * normals
        match self.tangential:
            case "material":
                tangential_rate = geometry.measures * geometry.surface_divergence.mv(
                    tangential
                )
            case "normal-only":
                tangential = jnp.zeros_like(tangential)
                tangential_rate = jnp.zeros_like(speed)
            case _:
                assert_never(self.tangential)
        normal_rate = geometry.measures * geometry.mean_curvature * speed
        mesh = speed[:, None] * normals + tangential
        finite = (
            jnp.all(jnp.isfinite(mesh))
            & jnp.all(jnp.isfinite(normal_rate))
            & jnp.all(jnp.isfinite(tangential_rate))
        )
        return SurfaceMotion(
            speed,
            tangential,
            mesh,
            jnp.zeros_like(mesh),
            normal_rate,
            tangential_rate,
            jnp.asarray(valid, dtype=jnp.bool_) & finite,
        )


@final
class PrescribedVelocityMotion(AbstractSurfaceMotionLaw):
    """Ambient velocity ``velocity(time, points, args)`` prescribed by the user."""

    velocity_field: Callable[[Array, Array, Any], ArrayLike]
    tangential: SurfaceTangentialMode = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        velocity: Callable[[Array, Array, Any], ArrayLike],
        /,
        *,
        tangential: SurfaceTangentialMode = "material",
        law_id: str,
    ) -> None:
        mode = parse(tangential, SurfaceTangentialMode, "tangential")
        owner = _identifier(law_id, "law_id")
        self.velocity_field = velocity
        self.tangential = mode
        self.law_id = canonical_fingerprint(
            {"kind": "prescribed-velocity-motion", "owner": owner, "tangential": mode}
        )

    def velocity(
        self, time: Array, geometry: MovingGeometryRefresh, args: Any, /
    ) -> tuple[Array, Array]:
        return jnp.asarray(self.velocity_field(time, geometry.points, args)), jnp.asarray(
            True
        )


@final
class LevelSetMotion(AbstractSurfaceMotionLaw):
    """Authoritative evolving zero level set ``phi(point, time, args) = 0``.

    The normal speed ``-phi_t / |grad phi|`` and its normal are exact
    derivatives of the declared level set at each stage point; the motion is
    normal-only because a level set does not determine a parametrization.
    """

    level_set: Callable[[Array, Array, Any], ArrayLike]
    gradient_floor: float = eqx.field(static=True)
    tangential: SurfaceTangentialMode = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        level_set: Callable[[Array, Array, Any], ArrayLike],
        /,
        *,
        law_id: str,
        gradient_floor: float = 1e-12,
    ) -> None:
        if not np.isfinite(gradient_floor) or gradient_floor <= 0:
            raise ValueError("gradient_floor must be positive and finite.")
        owner = _identifier(law_id, "law_id")
        self.level_set = level_set
        self.gradient_floor = float(gradient_floor)
        self.tangential = "normal-only"
        self.law_id = canonical_fingerprint(
            {
                "kind": "level-set-motion",
                "owner": owner,
                "gradient_floor": self.gradient_floor,
            }
        )

    def velocity(
        self, time: Array, geometry: MovingGeometryRefresh, args: Any, /
    ) -> tuple[Array, Array]:
        def value(point: Array, at: Array) -> Array:
            return jnp.asarray(self.level_set(point, at, args), dtype=jnp.float64)

        gradient = jax.vmap(jax.grad(value, argnums=0), in_axes=(0, None))(
            geometry.points, time
        )
        rate = jax.vmap(jax.grad(value, argnums=1), in_axes=(0, None))(
            geometry.points, time
        )
        squared = jnp.sum(gradient * gradient, axis=1)
        valid = jnp.all(squared > self.gradient_floor**2)
        safe = jnp.where(squared > self.gradient_floor**2, squared, 1)
        return -(rate / safe)[:, None] * gradient, valid


@final
class ChartMotion(AbstractSurfaceMotionLaw):
    """Authoritative point positions ``chart(time)`` of an evolving chart.

    Stage geometry is evaluated at the exact chart positions, never at a
    time-integrated approximation; the velocity is the exact time JVP.
    The chart must move with its full material velocity. Purely normal motion
    without an authoritative chart belongs to ``PrescribedVelocityMotion``.
    """

    chart: Callable[[Array], ArrayLike]
    tangential: SurfaceTangentialMode = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        chart: Callable[[Array], ArrayLike],
        /,
        *,
        tangential: SurfaceTangentialMode = "material",
        law_id: str,
    ) -> None:
        mode = parse(tangential, SurfaceTangentialMode, "tangential")
        if mode == "normal-only":
            raise ValueError(
                "Authoritative charts cannot discard tangential chart velocity; "
                "use tangential='material' or PrescribedVelocityMotion with "
                "tangential='normal-only' for prescribed normal-only motion."
            )
        owner = _identifier(law_id, "law_id")
        self.chart = chart
        self.tangential = mode
        self.law_id = canonical_fingerprint(
            {"kind": "chart-motion", "owner": owner, "tangential": mode}
        )

    def positions(self, time: Array, /) -> Array:
        return jnp.asarray(self.chart(jnp.asarray(time, dtype=jnp.float64)))

    def velocity(
        self, time: Array, geometry: MovingGeometryRefresh, args: Any, /
    ) -> tuple[Array, Array]:
        at = jnp.asarray(time, dtype=jnp.float64)
        _, rate = jax.jvp(self.positions, (at,), (jnp.ones_like(at),))
        return rate, jnp.asarray(True)


@final
class MeanCurvatureMotion(AbstractSurfaceMotionLaw):
    """Mean-curvature flow ``dx/dt = mobility * Laplace--Beltrami(x)``.

    The Laplace--Beltrami operator of the embedding coordinates equals
    ``-H n``; its numerically tangential residue is discarded.
    """

    mobility: float = eqx.field(static=True)
    tangential: SurfaceTangentialMode = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(self, mobility: float, /, *, law_id: str) -> None:
        if not np.isfinite(mobility) or mobility <= 0:
            raise ValueError("Mean-curvature mobility must be positive and finite.")
        owner = _identifier(law_id, "law_id")
        self.mobility = float(mobility)
        self.tangential = "normal-only"
        self.law_id = canonical_fingerprint(
            {
                "kind": "mean-curvature-motion",
                "owner": owner,
                "mobility": self.mobility,
            }
        )

    def velocity(
        self, time: Array, geometry: MovingGeometryRefresh, args: Any, /
    ) -> tuple[Array, Array]:
        embedding = jax.vmap(geometry.laplace_beltrami.mv, in_axes=1, out_axes=1)(
            geometry.points
        )
        return self.mobility * embedding, jnp.asarray(True)


@final
class BulkDrivenMotion(AbstractSurfaceMotionLaw):
    """Velocity queried from a native bulk field reconstruction.

    ``bulk(time, args)`` returns the bulk velocity coefficients valid at the
    stage time; every stage point must be a supported query.
    """

    reconstruction: PreparedFieldReconstruction
    bulk: Callable[[Array, Any], ArrayLike]
    tangential: SurfaceTangentialMode = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        reconstruction: PreparedFieldReconstruction,
        bulk: Callable[[Array, Any], ArrayLike],
        /,
        *,
        tangential: SurfaceTangentialMode = "material",
        law_id: str,
    ) -> None:
        if reconstruction.value_shape != (reconstruction.physical_dimension,):
            raise ValueError(
                "Bulk velocity reconstruction must carry one ambient vector per point."
            )
        mode = parse(tangential, SurfaceTangentialMode, "tangential")
        owner = _identifier(law_id, "law_id")
        self.reconstruction = reconstruction
        self.bulk = bulk
        self.tangential = mode
        self.law_id = canonical_fingerprint(
            {
                "kind": "bulk-driven-motion",
                "owner": owner,
                "tangential": mode,
                "reconstruction": reconstruction.reconstruction_id,
            }
        )

    def velocity(
        self, time: Array, geometry: MovingGeometryRefresh, args: Any, /
    ) -> tuple[Array, Array]:
        query = self.reconstruction.evaluate(self.bulk(time, args), geometry.points)
        return query.values, jnp.all(query.valid)


__all__ = [
    "AbstractSurfaceMotionLaw",
    "BulkDrivenMotion",
    "ChartMotion",
    "LevelSetMotion",
    "MeanCurvatureMotion",
    "MovingGeometryRefresh",
    "PrescribedVelocityMotion",
    "SurfaceGeometryProvider",
    "SurfaceMotion",
    "SurfaceRefreshMode",
    "SurfaceTangentialMode",
]
