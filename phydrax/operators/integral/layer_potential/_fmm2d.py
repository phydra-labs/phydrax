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


class LaplaceFMMEvaluation2D(StrictModule, NonTrainableState):
    """FMM values, truncation evidence, and translation counts."""

    values: Array
    error_estimate: Array
    status: Array
    num_evaluations: Array
    accuracy_supported: Array
    m2m_translations: int = eqx.field(static=True)
    m2l_translations: int = eqx.field(static=True)
    l2l_translations: int = eqx.field(static=True)
    direct_evaluations: int = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)


def _build_tree(points: np.ndarray, leaf_size: int):
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
        if indices.size <= leaf_size:
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

    root = build(np.arange(points.shape[0], dtype=np.int32))
    return (
        np.asarray(centers),
        np.asarray(radii),
        np.asarray(children, dtype=np.int32),
        tuple(index_sets),
        root,
    )


def _m2m(child_moments: Array, child_center: complex, parent_center: complex) -> Array:
    order = int(child_moments.shape[0] - 1)
    displacement = child_center - parent_center
    return jnp.stack(
        tuple(
            jnp.sum(
                jnp.asarray(
                    [
                        math.comb(degree, lower)
                        * child_moments[lower]
                        * displacement ** (degree - lower)
                        for lower in range(degree + 1)
                    ]
                )
            )
            for degree in range(order + 1)
        )
    )


def _m2l(moment: Array, source_center: complex, target_center: complex) -> Array:
    order = int(moment.shape[0] - 1)
    displacement = target_center - source_center
    local = [
        -(
            moment[0] * jnp.log(displacement)
            - sum(moment[k] / (k * displacement**k) for k in range(1, order + 1))
        )
        / (2.0 * jnp.pi)
    ]
    for degree in range(1, order + 1):
        term = moment[0] * ((-1) ** (degree + 1)) / (degree * displacement**degree)
        for source_degree in range(1, order + 1):
            term = term - moment[source_degree] / source_degree * (
                (-1) ** degree
            ) * math.comb(source_degree + degree - 1, degree) / displacement ** (
                source_degree + degree
            )
        local.append(-term / (2.0 * jnp.pi))
    return jnp.stack(local)


def _l2l(parent_local: Array, parent_center: complex, child_center: complex) -> Array:
    order = int(parent_local.shape[0] - 1)
    displacement = child_center - parent_center
    return jnp.stack(
        tuple(
            sum(
                parent_local[upper]
                * math.comb(upper, degree)
                * displacement ** (upper - degree)
                for upper in range(degree, order + 1)
            )
            for degree in range(order + 1)
        )
    )


class LaplaceFMMBackend2D(StrictModule, NonTrainableState):
    """Two-dimensional Laplace FMM with explicit M2M, M2L, and L2L phases."""

    source_points: Array
    source_weights: Array
    source_centers: Array
    source_radii: Array
    source_children: Array
    source_indices: tuple[Array, ...]
    source_root: int = eqx.field(static=True)
    expansion_order: int = eqx.field(static=True)
    leaf_size: int = eqx.field(static=True)
    opening_angle: float = eqx.field(static=True)
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
            raise TypeError("LaplaceFMMBackend2D requires a single Laplace layer.")
        order = int(expansion_order)
        leaf = int(leaf_size)
        angle = float(opening_angle)
        if order < 1 or leaf < 1 or not math.isfinite(angle) or angle <= 0.0:
            raise ValueError("Invalid Laplace FMM policy.")
        points = np.asarray(potential.panelization.points, dtype=float)
        centers, radii, children, indices, root = _build_tree(points, leaf)
        self.source_points = jnp.asarray(points)
        self.source_weights = potential.panelization.weights
        self.source_centers = jnp.asarray(centers)
        self.source_radii = jnp.asarray(radii)
        self.source_children = jnp.asarray(children, dtype=jnp.int32)
        self.source_indices = tuple(
            jnp.asarray(index, dtype=jnp.int32) for index in indices
        )
        self.source_root = root
        self.expansion_order = order
        self.leaf_size = leaf
        self.opening_angle = angle
        self.backend_id = canonical_fingerprint(
            {
                "kind": "laplace-fmm-2d-v2",
                "source_points": array_tree_fingerprint(self.source_points),
                "expansion_order": order,
                "leaf_size": leaf,
                "opening_angle": angle,
            }
        )

    def _source_moments(self, density: Array) -> tuple[Array, ...]:
        source = self.source_points[:, 0] + 1j * self.source_points[:, 1]
        charges = self.source_weights * density
        moments: list[Array | None] = [None] * int(self.source_centers.shape[0])

        def visit(node: int) -> Array:
            children = np.asarray(self.source_children[node])
            center = self.source_centers[node, 0] + 1j * self.source_centers[node, 1]
            if np.all(children < 0):
                indices = self.source_indices[node]
                displacement = source[indices] - center
                value = jnp.stack(
                    tuple(
                        jnp.sum(charges[indices] * displacement**degree)
                        for degree in range(self.expansion_order + 1)
                    )
                )
            else:
                value = jnp.zeros((self.expansion_order + 1,), dtype=charges.dtype)
                for child in children:
                    if child >= 0:
                        child_center = (
                            self.source_centers[int(child), 0]
                            + 1j * self.source_centers[int(child), 1]
                        )
                        value = value + _m2m(visit(int(child)), child_center, center)
            moments[node] = value
            return value

        visit(self.source_root)
        return tuple(value for value in moments if value is not None)

    def local_expansions(
        self,
        potential: LaplaceLayerPotential2D,
        centers: ArrayLike,
        /,
    ) -> tuple[Array, tuple[int, ...], tuple[Array, ...], tuple[tuple[Array, ...], ...]]:
        """Return far-field local coefficients with M2M/M2L/L2L evidence."""
        values = jnp.asarray(centers, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError("FMM centers must have shape (center_count, 2).")
        moments = self._source_moments(potential.density)
        target_centers, target_radii, target_children, target_indices, target_root = _build_tree(
            np.asarray(values),
            1,
        )
        target_centers_jax = jnp.asarray(target_centers)
        target_radii_jax = jnp.asarray(target_radii)
        local = [
            jnp.zeros((self.expansion_order + 1,), dtype=complex)
            for _ in range(len(target_indices))
        ]
        near_sources: list[list[Array]] = [[] for _ in target_indices]
        m2l_count = [0]
        l2l_count = [0]

        def pair(source_node: int, target_node: int) -> None:
            source_center = self.source_centers[source_node, 0] + 1j * self.source_centers[source_node, 1]
            target_center = target_centers_jax[target_node, 0] + 1j * target_centers_jax[target_node, 1]
            distance = max(float(jnp.abs(target_center - source_center)), 1e-15)
            separated = (
                float(self.source_radii[source_node] + target_radii_jax[target_node])
                <= self.opening_angle * distance
            )
            source_children = np.asarray(self.source_children[source_node])
            target_children_local = np.asarray(target_children_global[target_node])
            if separated:
                local[target_node] = local[target_node] + _m2l(
                    moments[source_node], source_center, target_center
                )
                m2l_count[0] += 1
                return
            source_leaf = bool(np.all(source_children < 0))
            target_leaf = bool(np.all(target_children_local < 0))
            if source_leaf and target_leaf:
                near_sources[target_node].append(self.source_indices[source_node])
                return
            if target_leaf:
                for child in source_children:
                    if child >= 0:
                        pair(int(child), target_node)
                return
            for child in target_children_local:
                if child >= 0:
                    pair(source_node, int(child))

        target_children_global = target_children
        pair(self.source_root, target_root)

        def propagate(parent: int) -> None:
            parent_center = target_centers_jax[parent, 0] + 1j * target_centers_jax[parent, 1]
            for child in np.asarray(target_children_global[parent]):
                if child >= 0:
                    child_center = target_centers_jax[int(child), 0] + 1j * target_centers_jax[int(child), 1]
                    local[int(child)] = local[int(child)] + _l2l(
                        local[parent], parent_center, child_center
                    )
                    l2l_count[0] += 1
                    propagate(int(child))

        propagate(target_root)
        leaf_map = tuple(
            next(
                node
                for node, indices in enumerate(target_indices)
                if int(index) in set(np.asarray(indices).tolist())
                and np.all(target_children[node] < 0)
            )
            for index in range(values.shape[0])
        )
        return (
            target_centers_jax,
            leaf_map,
            tuple(local),
            tuple(tuple(indices) for indices in near_sources),
            (m2l_count[0], l2l_count[0]),
        )

    def evaluate(
        self,
        potential: LaplaceLayerPotential2D,
        targets: ArrayLike,
        /,
        *,
        absolute_tolerance: float = 1e-6,
    ) -> LaplaceFMMEvaluation2D:
        if (
            not isinstance(potential, LaplaceLayerPotential2D)
            or potential.kind != "single"
        ):
            raise TypeError("LaplaceFMMBackend2D requires a matching single layer.")
        values = jnp.asarray(targets, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError("FMM targets must have shape (target_count, 2).")
        tolerance = float(absolute_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("absolute_tolerance must be finite and nonnegative.")
        source = self.source_points[:, 0] + 1j * self.source_points[:, 1]
        moments = self._source_moments(potential.density)
        target_points = np.asarray(values, dtype=float)
        target_centers, target_radii, target_children, target_indices, target_root = (
            _build_tree(
                target_points,
                self.leaf_size,
            )
        )
        target_leaf_map = {}
        for node, indices in enumerate(target_indices):
            if np.all(target_children[node] < 0):
                for index in indices:
                    target_leaf_map[int(index)] = node
        target_centers_jax = jnp.asarray(target_centers)
        target_radii_jax = jnp.asarray(target_radii)
        local = [
            jnp.zeros((self.expansion_order + 1,), dtype=complex)
            for _ in range(len(target_indices))
        ]
        near_sources = [[] for _ in range(len(target_indices))]
        m2l_count = [0]
        l2l_count = [0]
        direct_count = [0]

        def pair(source_node: int, target_node: int) -> None:
            source_center = (
                self.source_centers[source_node, 0]
                + 1j * self.source_centers[source_node, 1]
            )
            target_center = (
                target_centers_jax[target_node, 0]
                + 1j * target_centers_jax[target_node, 1]
            )
            distance = max(float(jnp.abs(target_center - source_center)), 1e-15)
            separated = (
                float(self.source_radii[source_node] + target_radii_jax[target_node])
                <= self.opening_angle * distance
            )
            source_children = np.asarray(self.source_children[source_node])
            target_children = np.asarray(target_children_global[target_node])
            if separated:
                local[target_node] = local[target_node] + _m2l(
                    moments[source_node], source_center, target_center
                )
                m2l_count[0] += 1
                return
            source_leaf = bool(np.all(source_children < 0))
            target_leaf = bool(np.all(target_children < 0))
            if source_leaf and target_leaf:
                near_sources[target_node].append(self.source_indices[source_node])
                direct_count[0] += int(self.source_indices[source_node].size) * int(
                    target_indices[target_node].size
                )
                return
            if target_leaf:
                for child in source_children:
                    if child >= 0:
                        pair(int(child), target_node)
                return
            for child in target_children:
                if child >= 0:
                    pair(source_node, int(child))

        target_children_global = target_children
        pair(self.source_root, target_root)

        def propagate(parent: int) -> None:
            parent_children = np.asarray(target_children_global[parent])
            parent_center = (
                target_centers_jax[parent, 0] + 1j * target_centers_jax[parent, 1]
            )
            for child in parent_children:
                if child >= 0:
                    child_center = (
                        target_centers_jax[int(child), 0]
                        + 1j * target_centers_jax[int(child), 1]
                    )
                    local[int(child)] = local[int(child)] + _l2l(
                        local[parent], parent_center, child_center
                    )
                    l2l_count[0] += 1
                    propagate(int(child))

        propagate(target_root)
        outputs = []
        errors = []
        for target_index, target in enumerate(target_points):
            leaf = target_leaf_map[target_index]
            center = target_centers_jax[leaf, 0] + 1j * target_centers_jax[leaf, 1]
            displacement = target[0] + 1j * target[1] - center
            high = local[leaf][0]
            low = high
            for degree in range(1, self.expansion_order + 1):
                term = local[leaf][degree] * displacement**degree
                high = high + term
                if degree < self.expansion_order:
                    low = low + term
            direct = jnp.asarray(0.0)
            for indices in near_sources[leaf]:
                direct = direct + jnp.sum(
                    potential.density[indices]
                    * self.source_weights[indices]
                    * (
                        -jnp.log(jnp.abs(target[0] + 1j * target[1] - source[indices]))
                        / (2.0 * jnp.pi)
                    )
                )
            outputs.append(jnp.real(high) + direct)
            errors.append(jnp.abs(high - low))
        values_ = jnp.stack(outputs)
        error_estimate = jnp.max(jnp.stack(errors))
        finite = jnp.all(jnp.isfinite(values_)) & jnp.isfinite(error_estimate)
        supported = finite & (error_estimate <= tolerance)
        return LaplaceFMMEvaluation2D(
            values=values_,
            error_estimate=error_estimate,
            status=jnp.where(supported, 0, int(IntegrationStatus.REFINEMENT_STAGNATION)),
            num_evaluations=jnp.asarray(
                direct_count[0] + m2l_count[0] + l2l_count[0], dtype=jnp.int32
            ),
            accuracy_supported=supported,
            m2m_translations=max(len(self.source_indices) - 1, 0),
            m2l_translations=m2l_count[0],
            l2l_translations=l2l_count[0],
            direct_evaluations=direct_count[0],
            evaluation_id=canonical_fingerprint(
                {
                    "kind": "laplace-fmm-evaluation-2d-v2",
                    "backend_id": self.backend_id,
                    "target_count": int(values.shape[0]),
                    "tolerance": tolerance,
                }
            ),
        )


__all__ = ["LaplaceFMMBackend2D", "LaplaceFMMEvaluation2D"]
