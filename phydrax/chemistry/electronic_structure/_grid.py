#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Moving atom-centered radial–Lebedev molecular integration grids."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._polynomial._cubature import lebedev_rule_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomisticSystemPlan


class AtomicRadialGridKind(StrEnum):
    MURA_KNOWLES = "mura-knowles"
    RATIONAL = "rational"


class AtomicRadialGridPlan(StrictModule, NonTrainableState):
    kind: AtomicRadialGridKind = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    radial_scale: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        point_count: int = 50,
        /,
        *,
        kind: AtomicRadialGridKind = AtomicRadialGridKind.MURA_KNOWLES,
        radial_scale: float = 1.0,
    ):
        count = int(point_count)
        scale = float(radial_scale)
        if not isinstance(kind, AtomicRadialGridKind):
            raise TypeError("kind must be AtomicRadialGridKind.")
        if count < 2 or not isfinite(scale) or scale <= 0.0:
            raise ValueError("Radial point count and scale must be positive.")
        self.kind = kind
        self.point_count = count
        self.radial_scale = scale
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomic-radial-grid-plan",
                "radial_kind": kind.value,
                "point_count": count,
                "radial_scale": scale,
            }
        )

    def rule(self, dtype=np.float64, /) -> tuple[np.ndarray, np.ndarray]:
        x = (np.arange(self.point_count, dtype=dtype) + 0.5) / self.point_count
        if self.kind is AtomicRadialGridKind.MURA_KNOWLES:
            denominator = np.maximum(1.0 - x**3, np.finfo(dtype).tiny)
            radius = -self.radial_scale * np.log(denominator)
            derivative = 3.0 * self.radial_scale * x**2 / denominator
        else:
            denominator = np.maximum(1.0 - x, np.finfo(dtype).tiny)
            radius = self.radial_scale * x**2 / denominator
            derivative = self.radial_scale * x * (2.0 - x) / denominator**2
        weights = radius**2 * derivative / self.point_count
        return radius, weights


class MolecularGridEvaluation(StrictModule):
    points: Array
    weights: Array
    owner_indices: Array
    partition_sum_residual: Array
    minimum_interatomic_distance: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class MolecularDFTGridPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    radial: AtomicRadialGridPlan
    angular_degree: int = eqx.field(static=True)
    becke_iterations: int = eqx.field(static=True)
    maximum_points: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        radial: AtomicRadialGridPlan | None = None,
        /,
        *,
        angular_degree: int = 17,
        becke_iterations: int = 3,
        maximum_points: int = 2_000_000,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        radial_ = AtomicRadialGridPlan() if radial is None else radial
        if not isinstance(radial_, AtomicRadialGridPlan):
            raise TypeError("radial must be AtomicRadialGridPlan or None.")
        degree = int(angular_degree)
        iterations = int(becke_iterations)
        capacity = int(maximum_points)
        if degree < 1 or iterations < 1 or iterations > 5 or capacity <= 0:
            raise ValueError(
                "Molecular DFT angular, Becke, or point capacity is invalid."
            )
        angular = lebedev_rule_data(degree)
        active = np.asarray(system.active_mask) & np.asarray(system.element_mask)
        total = (
            int(np.count_nonzero(active)) * radial_.point_count * angular.points.shape[0]
        )
        if total <= 0 or total > capacity:
            raise ValueError("Molecular DFT grid exceeds its fixed point capacity.")
        self.system = system
        self.radial = radial_
        self.angular_degree = degree
        self.becke_iterations = iterations
        self.maximum_points = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "molecular-dft-grid-plan",
                "system": system.system_id,
                "radial": radial_.plan_id,
                "angular_degree": degree,
                "angular_source": angular.source_id,
                "becke_iterations": iterations,
                "maximum_points": capacity,
            }
        )

    def prepare(self) -> PreparedMolecularDFTGrid:
        return PreparedMolecularDFTGrid(self)


class PreparedMolecularDFTGrid(StrictModule, NonTrainableState):
    plan: MolecularDFTGridPlan
    owner_indices: Array
    local_points: Array
    product_weights: Array
    angular_source_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MolecularDFTGridPlan, /):
        if not isinstance(plan, MolecularDFTGridPlan):
            raise TypeError("plan must be MolecularDFTGridPlan.")
        angular = lebedev_rule_data(plan.angular_degree)
        radial, radial_weights = plan.radial.rule()
        local = (radial[:, None, None] * np.asarray(angular.points)[None, :, :]).reshape(
            (-1, 3)
        )
        weights = (
            radial_weights[:, None] * np.asarray(angular.weights)[None, :]
        ).reshape((-1,))
        active = np.flatnonzero(
            np.asarray(plan.system.active_mask) & np.asarray(plan.system.element_mask)
        )
        self.plan = plan
        self.owner_indices = jnp.repeat(
            jnp.asarray(active, dtype=jnp.int32), local.shape[0]
        )
        self.local_points = jnp.asarray(np.tile(local, (active.size, 1)))
        self.product_weights = jnp.asarray(np.tile(weights, active.size))
        self.angular_source_id = angular.source_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-molecular-dft-grid",
                "plan": plan.plan_id,
                "angular_source": angular.source_id,
                "arrays": array_tree_fingerprint(
                    {
                        "owner_indices": np.repeat(active, local.shape[0]),
                        "local_points": np.tile(local, (active.size, 1)),
                        "product_weights": np.tile(weights, active.size),
                    }
                ),
            }
        )

    @staticmethod
    def _switch(value: Array, iterations: int, /) -> Array:
        result = jnp.clip(value, -1.0, 1.0)
        for _ in range(iterations):
            result = 1.5 * result - 0.5 * result**3
        return result

    def evaluate(self, positions: ArrayLike, /) -> MolecularGridEvaluation:
        coordinate = jnp.asarray(positions, dtype=self.local_points.dtype)
        expected = (self.plan.system.particle_ids.shape[0], 3)
        if coordinate.shape != expected:
            raise ValueError(f"DFT geometry must have shape {expected}.")
        active_indices = jnp.asarray(
            np.flatnonzero(
                np.asarray(self.plan.system.active_mask)
                & np.asarray(self.plan.system.element_mask)
            ),
            dtype=jnp.int32,
        )
        centers = coordinate[active_indices]
        points = coordinate[self.owner_indices] + self.local_points
        distances = jnp.sqrt(
            jnp.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        )
        center_displacement = centers[:, None, :] - centers[None, :, :]
        center_distance = jnp.sqrt(jnp.sum(center_displacement**2, axis=2))
        safe_distance = jnp.where(
            jnp.eye(centers.shape[0], dtype=jnp.bool_), 1.0, center_distance
        )
        products = []
        for atom in range(centers.shape[0]):
            product = jnp.ones((points.shape[0],), dtype=points.dtype)
            for other in range(centers.shape[0]):
                if atom == other:
                    continue
                mu = (distances[:, atom] - distances[:, other]) / safe_distance[
                    atom, other
                ]
                product = (
                    product * 0.5 * (1.0 - self._switch(mu, self.plan.becke_iterations))
                )
            products.append(product)
        raw = jnp.stack(tuple(products), axis=1)
        denominator = jnp.sum(raw, axis=1)
        partition = raw / denominator[:, None]
        owner_ordinals = jnp.searchsorted(active_indices, self.owner_indices)
        owner_partition = jnp.take_along_axis(partition, owner_ordinals[:, None], axis=1)[
            :, 0
        ]
        weights = self.product_weights * owner_partition
        off_diagonal = jnp.where(
            jnp.eye(centers.shape[0], dtype=jnp.bool_), jnp.inf, center_distance
        )
        minimum_distance = jnp.min(off_diagonal, initial=jnp.inf)
        residual = jnp.max(jnp.abs(jnp.sum(partition, axis=1) - 1.0), initial=0.0)
        successful = (
            jnp.all(jnp.isfinite(points))
            & jnp.all(jnp.isfinite(weights))
            & jnp.all(weights >= 0.0)
            & (jnp.sum(weights) > 0.0)
            & jnp.isfinite(residual)
            & (residual <= 1.0e-10)
            & (minimum_distance > 0.0)
        )
        return MolecularGridEvaluation(
            points,
            weights,
            self.owner_indices,
            residual,
            minimum_distance,
            successful,
            self.prepared_id,
        )


__all__ = [
    "AtomicRadialGridKind",
    "AtomicRadialGridPlan",
    "MolecularDFTGridPlan",
    "MolecularGridEvaluation",
    "PreparedMolecularDFTGrid",
]
