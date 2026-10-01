# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Tangential mesh motion and material-minus-mesh conservative upwinding."""

from __future__ import annotations

from collections.abc import Callable
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...sparse import linear_apply, SparseCoordinateOperator
from ...typing import Bool, Dim, Float64, Scalar
from ._capacity import ActivePointDim
from ._exterior_transport import edge_upwind_content


class ShiftAmbientDim(Dim):
    """Ambient Cartesian components."""


class ShiftEdgeDim(Dim):
    """Canonical unoriented surface edges."""


@final
class SurfaceShiftResult(StrictModule):
    __strict_contract__ = True
    points: Float64[ActivePointDim, ShiftAmbientDim]
    mesh_velocity: Float64[ActivePointDim, ShiftAmbientDim]
    successful: Bool[Scalar]
    maximum_displacement: Float64[Scalar]
    projection_residual: Float64[Scalar]


@final
class SurfaceShiftPolicy(StrictModule):
    strength: float = eqx.field(static=True)
    target_separation: float = eqx.field(static=True)
    maximum_displacement: float = eqx.field(static=True)
    projection_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        strength: float = 0.1,
        target_separation: float,
        maximum_displacement: float,
        projection_tolerance: float = 1e-9,
    ) -> None:
        if (
            not np.all(
                np.isfinite(
                    [
                        strength,
                        target_separation,
                        maximum_displacement,
                        projection_tolerance,
                    ]
                )
            )
            or strength < 0
            or min(target_separation, maximum_displacement, projection_tolerance) <= 0
        ):
            raise ValueError(
                "Shift strength must be nonnegative and shift bounds positive finite."
            )
        self.strength, self.target_separation = float(strength), float(target_separation)
        self.maximum_displacement, self.projection_tolerance = (
            float(maximum_displacement),
            float(projection_tolerance),
        )

    def propose(
        self,
        points: ArrayLike,
        normals: ArrayLike,
        pairs: ArrayLike,
        step_size: ArrayLike,
        project: Callable[[Array], Array],
        surface_residual: Callable[[Array], Array],
        /,
    ) -> SurfaceShiftResult:
        x, n = jnp.asarray(points), jnp.asarray(normals)
        edges = jnp.asarray(pairs, dtype=jnp.int32)
        if x.ndim != 2 or n.shape != x.shape or edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Shifting needs compact points/normals and endpoint pairs.")
        dt = jnp.asarray(step_size, dtype=x.dtype)
        delta = x[edges[:, 0]] - x[edges[:, 1]]
        length = jnp.linalg.norm(delta, axis=1)
        safe = jnp.maximum(length, jnp.finfo(x.dtype).tiny)
        force = (
            self.strength
            * jnp.maximum(self.target_separation - length, 0)[:, None]
            * delta
            / safe[:, None]
        )
        velocity = (
            jnp.zeros_like(x).at[edges[:, 0]].add(force).at[edges[:, 1]].add(-force)
        )
        velocity = velocity - jnp.sum(velocity * n, axis=1)[:, None] * n
        proposed = project(x + dt * velocity)
        displacement = jnp.linalg.norm(proposed - x, axis=1)
        residual = jnp.max(jnp.abs(surface_residual(proposed)))
        successful = (
            (dt > 0)
            & jnp.isfinite(dt)
            & jnp.all(jnp.isfinite(proposed))
            & (jnp.max(displacement) <= self.maximum_displacement)
            & (residual <= self.projection_tolerance)
        )
        safe_dt = jnp.where(dt > 0, dt, 1)
        return SurfaceShiftResult(
            jnp.where(successful, proposed, x),
            jnp.where(successful, (proposed - x) / safe_dt, jnp.zeros_like(x)),
            successful,
            jnp.max(displacement),
            residual,
        )


@final
class SurfaceRelativeAdvectionResult(StrictModule):
    __strict_contract__ = True
    concentration: Float64[ActivePointDim]
    content: Float64[ActivePointDim]
    oriented_volume_flux: Float64[ShiftEdgeDim]
    cfl: Float64[Scalar]
    conservation_residual: Float64[Scalar]
    positivity_admitted: Bool[Scalar]
    successful: Bool[Scalar]


def surface_relative_advection(
    concentration: ArrayLike,
    measures: ArrayLike,
    points: ArrayLike,
    incidence: SparseCoordinateOperator,
    edge_metric: ArrayLike,
    material_velocity: ArrayLike,
    mesh_velocity: ArrayLike,
    step_size: ArrayLike,
    /,
    *,
    require_positivity: bool = False,
) -> SurfaceRelativeAdvectionResult:
    """ALE: q=(v_material-v_mesh).tau times edge measure, positive i -> j.

    A pure mesh shift therefore transports material opposite the mesh motion.
    The metric is the supplied nonnegative dual edge measure, not a mass.
    Signed metrics can conserve but can never establish positivity admission.
    """
    c, w, x = jnp.asarray(concentration), jnp.asarray(measures), jnp.asarray(points)
    metric = jnp.asarray(edge_metric)
    vm, vg, dt = (
        jnp.asarray(material_velocity),
        jnp.asarray(mesh_velocity),
        jnp.asarray(step_size),
    )
    if not isinstance(incidence, SparseCoordinateOperator):
        raise TypeError("Surface advection requires a prepared native sparse incidence.")
    if (
        c.ndim != 1
        or w.shape != c.shape
        or x.shape[0] != c.size
        or vm.shape != x.shape
        or vg.shape != x.shape
        or incidence.source.size != c.size
        or metric.shape != (incidence.target.size,)
    ):
        raise ValueError(
            "Relative advection arrays do not match compact incidence spaces."
        )
    delta = linear_apply(incidence.relation, incidence.coefficients, x)
    length = jnp.linalg.norm(delta, axis=1)
    direction = delta / jnp.maximum(length, jnp.finfo(x.dtype).tiny)[:, None]
    relative = linear_apply(
        incidence.relation, 0.5 * jnp.abs(incidence.coefficients), vm - vg
    )
    flux = metric * jnp.sum(relative * direction, axis=1)
    valid_step = jnp.isfinite(dt) & (dt >= 0)
    valid_measure = jnp.isfinite(w) & (w > 0)
    transport = edge_upwind_content(
        c,
        jnp.where(valid_measure, w, 1),
        incidence,
        flux,
        jnp.where(valid_step, dt, 0),
        metric_nonnegative=jnp.all(jnp.isfinite(metric) & (metric >= 0)),
    )
    content = w * c
    proposed = transport.content
    positive = transport.positivity_admitted & valid_step & jnp.all(valid_measure)
    successful = (
        jnp.all(valid_measure)
        & jnp.all(length > 0)
        & valid_step
        & jnp.all(jnp.isfinite(proposed))
    )
    if require_positivity:
        successful = successful & positive
    accepted = jnp.where(successful, proposed, content)
    return SurfaceRelativeAdvectionResult(
        accepted / w,
        accepted,
        flux,
        transport.cfl,
        transport.conservation_residual,
        positive,
        successful,
    )


__all__ = [
    "SurfaceShiftPolicy",
    "SurfaceShiftResult",
    "SurfaceRelativeAdvectionResult",
    "surface_relative_advection",
]
