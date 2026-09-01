#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._lifting import LiftingSurfacePlan


class LiftingFrame3D(StrictModule):
    rotation: Array
    translation: Array
    linear_velocity: Array
    angular_velocity: Array

    def __init__(
        self,
        rotation: ArrayLike,
        translation: ArrayLike,
        /,
        *,
        linear_velocity: ArrayLike | None = None,
        angular_velocity: ArrayLike | None = None,
    ):
        rotation_ = jnp.asarray(rotation, dtype=float)
        translation_ = jnp.asarray(translation, dtype=float)
        linear = (
            jnp.zeros((3,), dtype=rotation_.dtype)
            if linear_velocity is None
            else jnp.asarray(linear_velocity, dtype=rotation_.dtype)
        )
        angular = (
            jnp.zeros((3,), dtype=rotation_.dtype)
            if angular_velocity is None
            else jnp.asarray(angular_velocity, dtype=rotation_.dtype)
        )
        if (
            rotation_.shape != (3, 3)
            or translation_.shape != (3,)
            or linear.shape != (3,)
            or angular.shape != (3,)
        ):
            raise ValueError("Lifting frame arrays have invalid shapes.")
        orthogonality = rotation_.T @ rotation_ - jnp.eye(3, dtype=rotation_.dtype)
        rotation_ = eqx.error_if(
            rotation_,
            jnp.max(jnp.abs(orthogonality)) > 1.0e-8,
            "Lifting frame rotation must be orthogonal.",
        )
        self.rotation, self.translation = rotation_, translation_
        self.linear_velocity, self.angular_velocity = linear, angular

    def points(self, value: ArrayLike, /) -> Array:
        return jnp.asarray(value) @ self.rotation.T + self.translation

    def vectors(self, value: ArrayLike, /) -> Array:
        return jnp.asarray(value) @ self.rotation.T

    def velocity(self, world_points: ArrayLike, /) -> Array:
        relative = jnp.asarray(world_points) - self.translation
        return self.linear_velocity + jnp.cross(self.angular_velocity, relative)


class LiftingComponentPlan(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    surface: LiftingSurfacePlan
    frame: LiftingFrame3D
    body_id: int = eqx.field(static=True)
    flap_fraction: Array
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        surface: LiftingSurfacePlan,
        frame: LiftingFrame3D,
        /,
        *,
        body_id: int = 0,
        flap_fraction: ArrayLike | None = None,
    ):
        if (
            not str(name)
            or not isinstance(surface, LiftingSurfacePlan)
            or not isinstance(frame, LiftingFrame3D)
        ):
            raise ValueError("Lifting component requires name, surface, and frame.")
        prepared = surface.prepare()
        flap = (
            jnp.zeros((prepared.panel_count,))
            if flap_fraction is None
            else jnp.asarray(flap_fraction, dtype=prepared.control_point.dtype)
        )
        if flap.shape != (prepared.panel_count,) or jnp.any((flap < 0.0) | (flap > 1.0)):
            raise ValueError("Lifting flap fractions must lie in [0,1] per panel.")
        self.name, self.surface, self.frame, self.body_id, self.flap_fraction = (
            str(name),
            surface,
            frame,
            int(body_id),
            flap,
        )
        self.component_id = canonical_fingerprint(
            {
                "kind": "lifting-component",
                "name": str(name),
                "surface": surface.plan_id,
                "body_id": int(body_id),
            }
        )


class PreparedMultiLiftingSurface(StrictModule, NonTrainableState):
    components: tuple[LiftingComponentPlan, ...]
    bound_start: Array
    bound_end: Array
    trailing_start: Array
    trailing_end: Array
    control_point: Array
    normal: Array
    chord: Array
    span_width: Array
    body_velocity: Array
    component_index: Array
    local_panel_index: Array
    body_id: Array
    flap_fraction: Array
    trailing_edge_owner: Array
    panel_count: int = eqx.field(static=True)
    system_id: str = eqx.field(static=True)


class MultiLiftingSurfacePlan(StrictModule, NonTrainableState):
    components: tuple[LiftingComponentPlan, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(self, components: tuple[LiftingComponentPlan, ...], /):
        if not components or any(
            not isinstance(component, LiftingComponentPlan) for component in components
        ):
            raise ValueError("Multi-surface plan requires lifting components.")
        names = tuple(component.name for component in components)
        if len(set(names)) != len(names):
            raise ValueError("Lifting component names must be unique.")
        self.components = components
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multi-lifting-surface-plan",
                "components": [component.component_id for component in components],
            }
        )

    def prepare(self, /) -> PreparedMultiLiftingSurface:
        prepared = tuple(component.surface.prepare() for component in self.components)

        def points(component, value):
            return component.frame.points(value)

        def vectors(component, value):
            return component.frame.vectors(value)

        bound_start = jnp.concatenate(
            tuple(
                points(component, surface.bound_start)
                for component, surface in zip(self.components, prepared, strict=True)
            ),
            axis=0,
        )
        bound_end = jnp.concatenate(
            tuple(
                points(component, surface.bound_end)
                for component, surface in zip(self.components, prepared, strict=True)
            ),
            axis=0,
        )
        trailing_start = jnp.concatenate(
            tuple(
                points(component, surface.trailing_start)
                for component, surface in zip(self.components, prepared, strict=True)
            ),
            axis=0,
        )
        trailing_end = jnp.concatenate(
            tuple(
                points(component, surface.trailing_end)
                for component, surface in zip(self.components, prepared, strict=True)
            ),
            axis=0,
        )
        control = jnp.concatenate(
            tuple(
                points(component, surface.control_point)
                for component, surface in zip(self.components, prepared, strict=True)
            ),
            axis=0,
        )
        normal = jnp.concatenate(
            tuple(
                vectors(component, surface.normal)
                for component, surface in zip(self.components, prepared, strict=True)
            ),
            axis=0,
        )
        body_velocity = jnp.concatenate(
            tuple(
                component.frame.velocity(points(component, surface.control_point))
                for component, surface in zip(self.components, prepared, strict=True)
            ),
            axis=0,
        )
        panel_counts = tuple(surface.panel_count for surface in prepared)
        component_index = jnp.concatenate(
            tuple(
                jnp.full((count,), index, dtype=jnp.int32)
                for index, count in enumerate(panel_counts)
            )
        )
        local_index = jnp.concatenate(
            tuple(jnp.arange(count, dtype=jnp.int32) for count in panel_counts)
        )
        body_id = jnp.concatenate(
            tuple(
                jnp.full((count,), component.body_id, dtype=jnp.int32)
                for component, count in zip(self.components, panel_counts, strict=True)
            )
        )
        flap = jnp.concatenate(
            tuple(component.flap_fraction for component in self.components)
        )
        owner = jnp.arange(sum(panel_counts), dtype=jnp.int32)
        return PreparedMultiLiftingSurface(
            self.components,
            bound_start,
            bound_end,
            trailing_start,
            trailing_end,
            control,
            normal,
            jnp.concatenate(tuple(surface.chord for surface in prepared)),
            jnp.concatenate(tuple(surface.span_width for surface in prepared)),
            body_velocity,
            component_index,
            local_index,
            body_id,
            flap,
            owner,
            sum(panel_counts),
            canonical_fingerprint(
                {
                    "kind": "prepared-multi-lifting-surface",
                    "plan": self.plan_id,
                    "panel_count": sum(panel_counts),
                }
            ),
        )


__all__ = [
    "LiftingComponentPlan",
    "LiftingFrame3D",
    "MultiLiftingSurfacePlan",
    "PreparedMultiLiftingSurface",
]
