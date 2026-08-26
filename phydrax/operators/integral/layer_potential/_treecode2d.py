#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....integration import IntegrationStatus
from ._laplace2d import LaplaceLayerPotential2D


class LaplaceTreecodeEvaluation2D(StrictModule, NonTrainableState):
    """Hierarchical multipole values and truncation evidence."""

    values: Array
    error_estimate: Array
    status: Array
    num_evaluations: Array
    accuracy_supported: Array
    multipole_evaluations: int = eqx.field(static=True)
    direct_evaluations: int = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)


class LaplaceTreecodeBackend2D(StrictModule, NonTrainableState):
    """Fixed-topology 2D Laplace multipole treecode for single layers."""

    source_points: Array
    source_weights: Array
    node_centers: Array
    node_radii: Array
    node_children: Array
    node_indices: tuple[Array, ...]
    expansion_order: int = eqx.field(static=True)
    leaf_size: int = eqx.field(static=True)
    opening_angle: float = eqx.field(static=True)
    source_panelization_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)

    def __init__(
        self,
        potential: LaplaceLayerPotential2D,
        /,
        *,
        expansion_order: int = 8,
        leaf_size: int = 32,
        opening_angle: float = 0.5,
    ):
        if (
            not isinstance(potential, LaplaceLayerPotential2D)
            or potential.kind != "single"
        ):
            raise TypeError("LaplaceTreecodeBackend2D requires a single Laplace layer.")
        order = int(expansion_order)
        leaf = int(leaf_size)
        angle = float(opening_angle)
        if order < 1 or leaf < 1 or not math.isfinite(angle) or not 0.0 < angle < 1.0:
            raise ValueError("Laplace treecode opening_angle must lie in (0, 1).")
        points = np.asarray(potential.panelization.points, dtype=float)
        centers: list[np.ndarray] = []
        radii: list[float] = []
        children: list[list[int]] = []
        index_sets: list[np.ndarray] = []

        def build(indices: np.ndarray) -> int:
            values = points[indices]
            lower = values.min(axis=0)
            upper = values.max(axis=0)
            center = 0.5 * (lower + upper)
            radius = float(np.max(np.linalg.norm(values - center, axis=1)))
            node = len(centers)
            centers.append(center)
            radii.append(max(radius, np.finfo(float).eps))
            children.append([-1, -1, -1, -1])
            index_sets.append(indices)
            if indices.size <= leaf:
                return node
            quadrant = (values[:, 0] >= center[0]).astype(np.int32) + 2 * (
                values[:, 1] >= center[1]
            ).astype(np.int32)
            child_nodes = []
            for quadrant_id in range(4):
                selected = indices[quadrant == quadrant_id]
                if selected.size:
                    child_nodes.append((quadrant_id, build(selected)))
            if len(child_nodes) <= 1:
                return node
            for quadrant_id, child in child_nodes:
                children[node][quadrant_id] = child
            return node

        build(np.arange(points.shape[0], dtype=np.int32))
        self.source_points = jnp.asarray(points)
        self.source_weights = potential.panelization.weights
        self.node_centers = jnp.asarray(np.asarray(centers), dtype=float)
        self.node_radii = jnp.asarray(np.asarray(radii), dtype=float)
        self.node_children = jnp.asarray(np.asarray(children), dtype=jnp.int32)
        self.node_indices = tuple(
            jnp.asarray(indices, dtype=jnp.int32) for indices in index_sets
        )
        self.expansion_order = order
        self.leaf_size = leaf
        self.opening_angle = angle
        self.source_panelization_id = potential.panelization.panelization_id
        self.backend_id = canonical_fingerprint(
            {
                "kind": "laplace-treecode-2d-v1",
                "source_panelization_id": self.source_panelization_id,
                "source_points": array_tree_fingerprint(self.source_points),
                "expansion_order": order,
                "leaf_size": leaf,
                "opening_angle": angle,
            }
        )

    def _moments(self, density: Array) -> tuple[Array, ...]:
        source = self.source_points[:, 0] + 1j * self.source_points[:, 1]
        charges = self.source_weights * density
        moments = []
        for node, indices in enumerate(self.node_indices):
            center = self.node_centers[node, 0] + 1j * self.node_centers[node, 1]
            displacement = source[indices] - center
            moments.append(
                jnp.stack(
                    tuple(
                        jnp.sum(charges[indices] * displacement**degree)
                        for degree in range(self.expansion_order + 1)
                    )
                )
            )
        return tuple(moments)

    def evaluate(
        self,
        potential: LaplaceLayerPotential2D,
        targets: ArrayLike,
        /,
        *,
        absolute_tolerance: float = 1e-6,
    ) -> LaplaceTreecodeEvaluation2D:
        if (
            not isinstance(potential, LaplaceLayerPotential2D)
            or potential.kind != "single"
            or potential.panelization.panelization_id != self.source_panelization_id
        ):
            raise TypeError(
                "LaplaceTreecodeBackend2D requires its bound single-layer source geometry."
            )
        values = jnp.asarray(targets, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError("Treecode targets must have shape (target_count, 2).")
        tolerance = float(absolute_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("absolute_tolerance must be finite and nonnegative.")
        moments = self._moments(potential.density)
        source = self.source_points[:, 0] + 1j * self.source_points[:, 1]
        absolute_charge = jnp.abs(self.source_weights * potential.density)
        outputs = []
        errors = []
        direct_count = [0]
        multipole_count = [0]
        for target in values:
            target_complex = target[0] + 1j * target[1]

            def visit(node: int) -> tuple[Array, Array]:
                center = self.node_centers[node, 0] + 1j * self.node_centers[node, 1]
                displacement = target_complex - center
                distance = max(float(jnp.abs(displacement)), 1e-15)
                children = np.asarray(self.node_children[node])
                is_leaf = bool(np.all(children < 0))
                if is_leaf:
                    indices = self.node_indices[node]
                    direct_count[0] += int(indices.size)
                    direct = jnp.sum(
                        potential.density[indices]
                        * self.source_weights[indices]
                        * (
                            -jnp.log(jnp.abs(target_complex - source[indices]))
                            / (2.0 * jnp.pi)
                        )
                    )
                    return direct, jnp.asarray(0.0)
                if float(self.node_radii[node]) / distance <= self.opening_angle:
                    multipole_count[0] += 1
                    moment = moments[node]
                    high = -jnp.real(moment[0] * jnp.log(displacement)) / (2.0 * jnp.pi)
                    for degree in range(1, self.expansion_order + 1):
                        high = high + jnp.real(
                            moment[degree] / (degree * displacement**degree)
                        ) / (2.0 * jnp.pi)
                    ratio = self.node_radii[node] / distance
                    charge_mass = jnp.sum(absolute_charge[self.node_indices[node]])
                    tail = (
                        charge_mass
                        * ratio ** (self.expansion_order + 1)
                        / (
                            2.0
                            * jnp.pi
                            * (self.expansion_order + 1)
                            * jnp.maximum(1.0 - ratio, 1e-15)
                        )
                    )
                    return high, tail
                value = jnp.asarray(0.0)
                error = jnp.asarray(0.0)
                for child in children:
                    if child >= 0:
                        child_value, child_error = visit(int(child))
                        value = value + child_value
                        error = error + child_error
                return value, error

            value, error = visit(0)
            outputs.append(value)
            errors.append(error)
        values_ = jnp.stack(outputs)
        error_estimate = jnp.max(jnp.stack(errors))
        finite = jnp.all(jnp.isfinite(values_)) & jnp.isfinite(error_estimate)
        supported = finite & (error_estimate <= tolerance)
        return LaplaceTreecodeEvaluation2D(
            values=values_,
            error_estimate=error_estimate,
            status=jnp.where(
                supported,
                int(IntegrationStatus.CONVERGED),
                int(IntegrationStatus.REFINEMENT_STAGNATION),
            ),
            num_evaluations=jnp.asarray(
                direct_count[0] + multipole_count[0],
                dtype=jnp.int32,
            ),
            accuracy_supported=supported,
            multipole_evaluations=multipole_count[0],
            direct_evaluations=direct_count[0],
            evaluation_id=canonical_fingerprint(
                {
                    "kind": "laplace-treecode-evaluation-2d-v1",
                    "backend_id": self.backend_id,
                    "target_count": int(values.shape[0]),
                    "potential": potential.representation_id,
                    "targets": array_tree_fingerprint(values),
                    "density": array_tree_fingerprint(potential.density),
                    "tolerance": tolerance,
                }
            ),
        )


__all__ = ["LaplaceTreecodeBackend2D", "LaplaceTreecodeEvaluation2D"]
