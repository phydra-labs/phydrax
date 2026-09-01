#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from .._contracts import CompiledGeometry, GeometryKernel, GeometryTolerance
from .._validity import GeometryValidityEvidence
from ..design._schema import DesignState, ParameterSchema
from ._policy import ImplicitProjectionPolicy, ImplicitProjectionStatus


_DEFAULT_PROJECTION_POLICY = ImplicitProjectionPolicy()


def _field_and_gradient(
    kernel: GeometryKernel,
    state: DesignState,
    points: Array,
) -> tuple[Array, Array]:
    def field(point):
        return kernel.boundary_field(state, point[None, :])[0]

    values = kernel.boundary_field(state, points)
    gradients = jax.vmap(jax.grad(field))(points)
    return values, gradients


@eqx.filter_custom_jvp
def _attach_normal_gauge(
    kernel: GeometryKernel,
    state: DesignState,
    points: Array,
    minimum_gradient_norm: float,
    /,
) -> Array:
    del kernel, state, minimum_gradient_norm
    return points


@_attach_normal_gauge.def_jvp
def _attach_normal_gauge_jvp(primals, tangents):
    kernel, state, points, minimum_gradient_norm = primals
    kernel_tangent, state_tangent, _, _ = tangents

    def parameter_field(current_kernel, current_state):
        return current_kernel.boundary_field(current_state, points)

    _, field_tangent = eqx.filter_jvp(
        parameter_field,
        (kernel, state),
        (kernel_tangent, state_tangent),
    )
    if field_tangent is None:
        field_tangent = jnp.zeros(points.shape[:-1], dtype=points.dtype)
    _, gradient = _field_and_gradient(kernel, state, points)
    squared_norm = jnp.sum(gradient * gradient, axis=-1)
    usable = squared_norm >= float(minimum_gradient_norm) ** 2
    tangent = (
        -field_tangent[..., None]
        * gradient
        / jnp.maximum(
            squared_norm[..., None],
            jnp.finfo(points.dtype).tiny,
        )
    )
    tangent = jnp.where(usable[..., None], tangent, jnp.zeros_like(tangent))
    return points, tangent


class ImplicitPointProjectionEvidence(StrictModule):
    """Runtime evidence for a fixed-anchor implicit projection."""

    geometry: GeometryValidityEvidence
    root_residual: Array
    minimum_gradient_norm: Array
    maximum_displacement_ratio: Array
    finite: Array
    status: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry: GeometryValidityEvidence,
        root_residual: Any,
        minimum_gradient_norm: Any,
        maximum_displacement_ratio: Any,
        finite: Any,
        status: Any,
        plan_id: str,
    ):
        if not isinstance(geometry, GeometryValidityEvidence):
            raise TypeError("geometry must be GeometryValidityEvidence.")
        self.geometry = geometry
        self.root_residual = jnp.asarray(root_residual, dtype=float).reshape(())
        self.minimum_gradient_norm = jnp.asarray(
            minimum_gradient_norm, dtype=float
        ).reshape(())
        self.maximum_displacement_ratio = jnp.asarray(
            maximum_displacement_ratio, dtype=float
        ).reshape(())
        self.finite = jnp.asarray(finite, dtype=bool).reshape(())
        self.status = jnp.asarray(status, dtype=jnp.int32).reshape(())
        self.plan_id = str(plan_id)

    @property
    def accepted(self) -> Array:
        return self.status == int(ImplicitProjectionStatus.SUCCESS)

    @property
    def refresh_required(self) -> Array:
        refresh_bits = int(
            ImplicitProjectionStatus.LOST_REGULARITY
            | ImplicitProjectionStatus.TRUST_REGION_EXCEEDED
        )
        return (self.status & refresh_bits) != 0


class ImplicitPointProjectionResult(StrictModule):
    """Proposed and safe coordinates from fixed-anchor implicit projection."""

    proposed_points: Array
    points: Array
    normals: Array
    evidence: ImplicitPointProjectionEvidence

    def __init__(
        self,
        proposed_points: Array,
        points: Array,
        normals: Array,
        evidence: ImplicitPointProjectionEvidence,
        /,
    ):
        proposed = jnp.asarray(proposed_points, dtype=float)
        safe = jnp.asarray(points, dtype=proposed.dtype)
        normals_ = jnp.asarray(normals, dtype=proposed.dtype)
        if proposed.ndim != 2 or safe.shape != proposed.shape:
            raise ValueError("Projection points must have matching shape (points, dim).")
        if normals_.shape != proposed.shape:
            raise ValueError("Projection normals must match the point shape.")
        self.proposed_points = proposed
        self.points = safe
        self.normals = normals_
        self.evidence = evidence

    @property
    def accepted(self) -> Array:
        return self.evidence.accepted

    @property
    def refresh_required(self) -> Array:
        return self.evidence.refresh_required

    @property
    def status(self) -> Array:
        return self.evidence.status


class ImplicitPointProjectionPlan(StrictModule):
    """Fixed-anchor, fixed-shape normal-gauge implicit projection plan."""

    kernel: GeometryKernel
    anchors: Array
    trust_radii: Array
    schema: ParameterSchema = eqx.field(static=True)
    tolerance: GeometryTolerance = eqx.field(static=True)
    policy: ImplicitProjectionPolicy = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: CompiledGeometry,
        anchors: Any,
        trust_radii: Any,
        /,
        *,
        policy: ImplicitProjectionPolicy = _DEFAULT_PROJECTION_POLICY,
        source_id: str,
        plan_id: str | None = None,
    ):
        if not isinstance(geometry, CompiledGeometry):
            raise TypeError("geometry must be CompiledGeometry.")
        if not isinstance(policy, ImplicitProjectionPolicy):
            raise TypeError("policy must be ImplicitProjectionPolicy.")
        if not source_id:
            raise ValueError("source_id must be non-empty.")
        anchors_host = np.asarray(anchors, dtype=float)
        if (
            anchors_host.ndim != 2
            or anchors_host.shape[0] == 0
            or anchors_host.shape[1] != geometry.ambient_dimension
            or not np.all(np.isfinite(anchors_host))
        ):
            raise ValueError(
                "anchors must be a non-empty finite array with shape (points, dim)."
            )
        trust_host = np.asarray(trust_radii, dtype=float)
        if trust_host.shape == ():
            trust_host = np.full((anchors_host.shape[0],), float(trust_host))
        if (
            trust_host.shape != (anchors_host.shape[0],)
            or not np.all(np.isfinite(trust_host))
            or np.any(trust_host <= 0.0)
        ):
            raise ValueError("trust_radii must be positive with one value per anchor.")
        if not bool(np.asarray(geometry.validity().accepted)):
            raise ValueError("Projection discovery geometry must be valid.")
        values, gradients = _field_and_gradient(
            geometry.kernel,
            geometry.state,
            jnp.asarray(anchors_host),
        )
        residual = float(np.max(np.abs(np.asarray(values))))
        minimum_gradient = float(np.min(np.linalg.norm(np.asarray(gradients), axis=-1)))
        if residual > policy.root_tolerance:
            raise ValueError(
                "Projection anchors must lie on the discovery zero set within "
                "root_tolerance."
            )
        if minimum_gradient < policy.minimum_gradient_norm:
            raise ValueError("Projection anchors must be regular field points.")
        identifier = plan_id or canonical_fingerprint(
            {
                "kind": "implicit-point-projection",
                "source_id": source_id,
                "kernel": type(geometry.kernel).__qualname__,
                "schema": [str(item.parameter_id) for item in geometry.schema.specs],
                "anchors": anchors_host.tolist(),
                "trust_radii": trust_host.tolist(),
                "policy": repr(policy),
            }
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.kernel = geometry.kernel
        self.anchors = jnp.asarray(anchors_host)
        self.trust_radii = jnp.asarray(trust_host)
        self.schema = geometry.schema
        self.tolerance = geometry.tolerance
        self.policy = policy
        self.source_id = source_id
        self.plan_id = identifier

    @property
    def mapping_id(self) -> str:
        return self.plan_id

    @property
    def reference_points(self) -> Array:
        return self.anchors

    def realize(self, state: DesignState, /) -> ImplicitPointProjectionResult:
        if not isinstance(state, DesignState) or state.schema != self.schema:
            raise ValueError("Projection state must use the discovery parameter schema.")
        policy = self.policy
        minimum_squared = float(policy.minimum_gradient_norm) ** 2

        def step(_, carry):
            points, trust_hit = carry
            values, gradient = _field_and_gradient(self.kernel, state, points)
            squared_norm = jnp.sum(gradient * gradient, axis=-1)
            usable = squared_norm >= minimum_squared
            increment = (
                -values[..., None]
                * gradient
                / jnp.maximum(
                    squared_norm[..., None],
                    jnp.finfo(points.dtype).tiny,
                )
            )
            increment = jnp.where(usable[..., None], increment, 0.0)
            candidate = points + increment
            delta = candidate - self.anchors
            displacement = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
            ratio = self.trust_radii / jnp.maximum(
                displacement,
                jnp.finfo(points.dtype).tiny,
            )
            clipped = self.anchors + delta * jnp.minimum(ratio, 1.0)[..., None]
            return clipped, trust_hit | (displacement > self.trust_radii)

        projected, trust_hit = jax.lax.fori_loop(
            0,
            int(policy.maximum_steps),
            step,
            (self.anchors, jnp.zeros(self.anchors.shape[:1], dtype=bool)),
        )
        proposed = _attach_normal_gauge(
            self.kernel,
            state,
            jax.lax.stop_gradient(projected),
            policy.minimum_gradient_norm,
        )
        values, gradient = _field_and_gradient(self.kernel, state, proposed)
        gradient_norm = jnp.sqrt(jnp.sum(gradient * gradient, axis=-1))
        displacement = jnp.sqrt(jnp.sum((proposed - self.anchors) ** 2, axis=-1))
        displacement_ratio = displacement / self.trust_radii
        geometry = CompiledGeometry(
            self.kernel,
            state,
            tolerance=self.tolerance,
        ).validity()
        root_residual = jnp.max(jnp.abs(values))
        minimum_gradient = jnp.min(gradient_norm)
        maximum_ratio = jnp.max(displacement_ratio)
        finite = (
            jnp.all(jnp.isfinite(proposed))
            & jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(gradient))
        )
        status = jnp.asarray(int(ImplicitProjectionStatus.SUCCESS), dtype=jnp.int32)
        status = status | jnp.where(
            geometry.accepted,
            0,
            int(ImplicitProjectionStatus.INVALID_GEOMETRY),
        ).astype(jnp.int32)
        status = status | jnp.where(
            finite,
            0,
            int(ImplicitProjectionStatus.NONFINITE),
        ).astype(jnp.int32)
        status = status | jnp.where(
            root_residual <= policy.root_tolerance,
            0,
            int(ImplicitProjectionStatus.ROOT_RESIDUAL),
        ).astype(jnp.int32)
        status = status | jnp.where(
            minimum_gradient >= policy.minimum_gradient_norm,
            0,
            int(ImplicitProjectionStatus.LOST_REGULARITY),
        ).astype(jnp.int32)
        status = status | jnp.where(
            ~jnp.any(trust_hit),
            0,
            int(ImplicitProjectionStatus.TRUST_REGION_EXCEEDED),
        ).astype(jnp.int32)
        evidence = ImplicitPointProjectionEvidence(
            geometry=geometry,
            root_residual=root_residual,
            minimum_gradient_norm=minimum_gradient,
            maximum_displacement_ratio=maximum_ratio,
            finite=finite,
            status=status,
            plan_id=self.plan_id,
        )
        safe_points = jnp.where(evidence.accepted, proposed, self.anchors)
        safe_gradient = jnp.where(evidence.accepted, gradient, jnp.zeros_like(gradient))
        safe_norm = jnp.sqrt(jnp.sum(safe_gradient * safe_gradient, axis=-1))
        normals = safe_gradient / jnp.maximum(
            safe_norm[..., None],
            jnp.finfo(safe_gradient.dtype).eps,
        )
        return ImplicitPointProjectionResult(
            proposed,
            safe_points,
            normals,
            evidence,
        )


__all__ = [
    "ImplicitPointProjectionEvidence",
    "ImplicitPointProjectionPlan",
    "ImplicitPointProjectionResult",
]
