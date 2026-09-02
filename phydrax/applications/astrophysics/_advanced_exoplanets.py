#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...integration import GaussLegendreRule
from ._observation_status import AstrophysicsObservationStatus


class OblateOccultationResult(StrictModule):
    relative_flux: Array
    valid: Array
    contact: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class OblateOccultationPlan(StrictModule, NonTrainableState):
    radial_nodes: Array
    radial_weights: Array
    angular_nodes: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, radial_order=96, angular_order=192):
        radial = GaussLegendreRule(int(radial_order)).data()
        angular = GaussLegendreRule(int(angular_order)).data()
        self.radial_nodes = 0.5 * (radial.nodes + 1.0)
        self.radial_weights = 0.5 * radial.weights
        self.angular_nodes = jnp.pi * (angular.nodes + 1.0)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "oblate-occultation",
                "radial_order": int(radial_order),
                "angular_order": int(angular_order),
            }
        )

    def evaluate(
        self,
        center: ArrayLike,
        semi_major: ArrayLike,
        semi_minor: ArrayLike,
        angle: ArrayLike,
        /,
    ) -> OblateOccultationResult:
        center_ = jnp.asarray(center)
        major = jnp.asarray(semi_major).reshape(())
        minor = jnp.asarray(semi_minor).reshape(())
        rotation = jnp.asarray(
            ((jnp.cos(angle), jnp.sin(angle)), (-jnp.sin(angle), jnp.cos(angle)))
        )
        radius = self.radial_nodes[:, None]
        theta = self.angular_nodes[None, :]
        x = radius * jnp.cos(theta)
        y = radius * jnp.sin(theta)
        points = jnp.stack((x + jnp.zeros_like(y), y + jnp.zeros_like(x)), axis=-1)
        relative = points - center_
        aligned = contract("ij,...j->...i", rotation, relative)
        covered = (aligned[..., 0] / major) ** 2 + (aligned[..., 1] / minor) ** 2 <= 1.0
        angular_fraction = jnp.mean(covered.astype(radius.dtype), axis=1)
        occulted = 2.0 * jnp.sum(
            self.radial_weights * self.radial_nodes * angular_fraction
        )
        relative_flux = jnp.clip(1.0 - occulted, 0.0, 1.0)
        separation = jnp.sqrt(jnp.sum(center_ * center_))
        contact = (
            jnp.abs(separation - (1.0 + major)) <= 64.0 * jnp.finfo(radius.dtype).eps
        )
        valid = jnp.all(jnp.isfinite(center_)) & (major > 0.0) & (minor > 0.0)
        status = jnp.where(
            valid,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.INVALID_GEOMETRY),
        ).astype(jnp.int32)
        return OblateOccultationResult(
            relative_flux, valid, contact, status, self.plan_id
        )


class MicrolensingResult(StrictModule):
    magnification: Array
    topology_id: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class FiniteSourceMicrolensingPlan(StrictModule, NonTrainableState):
    radial_nodes: Array
    radial_weights: Array
    angular_nodes: Array
    source_radius: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, source_radius: ArrayLike, /, *, radial_order=48, angular_order=96):
        radial = GaussLegendreRule(int(radial_order)).data()
        angular = GaussLegendreRule(int(angular_order)).data()
        self.radial_nodes = 0.5 * (radial.nodes + 1.0)
        self.radial_weights = 0.5 * radial.weights
        self.angular_nodes = jnp.pi * (angular.nodes + 1.0)
        self.source_radius = jnp.asarray(source_radius).reshape(())
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-source-microlensing",
                "radius": float(self.source_radius),
                "radial_order": int(radial_order),
                "angular_order": int(angular_order),
            }
        )

    @staticmethod
    def point_magnification(separation: Array, /) -> Array:
        squared = separation * separation
        return (squared + 2.0) / jnp.maximum(
            separation * jnp.sqrt(squared + 4.0), 1.0e-30
        )

    def evaluate(self, lens_source_separation: ArrayLike, /) -> MicrolensingResult:
        center = jnp.asarray(lens_source_separation)
        if center.shape != (2,):
            raise ValueError("Microlensing separation must have shape (2,).")
        radius = self.source_radius * self.radial_nodes[:, None]
        theta = self.angular_nodes[None, :]
        points = center + jnp.stack(
            (radius * jnp.cos(theta), radius * jnp.sin(theta)), axis=-1
        )
        separation = jnp.sqrt(jnp.sum(points * points, axis=-1))
        magnification = self.point_magnification(separation)
        radial_average = jnp.mean(magnification, axis=1)
        integrated = 2.0 * jnp.sum(
            self.radial_weights * self.radial_nodes * radial_average
        )
        topology = jax.lax.stop_gradient(
            jnp.where(jnp.sqrt(jnp.sum(center * center)) < self.source_radius, 1, 0)
        ).astype(jnp.int32)
        valid = jnp.isfinite(integrated) & (self.source_radius > 0.0)
        status = jnp.where(
            valid,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.INVALID_GEOMETRY),
        ).astype(jnp.int32)
        return MicrolensingResult(integrated, topology, valid, status, self.plan_id)


__all__ = [
    "FiniteSourceMicrolensingPlan",
    "MicrolensingResult",
    "OblateOccultationPlan",
    "OblateOccultationResult",
]
