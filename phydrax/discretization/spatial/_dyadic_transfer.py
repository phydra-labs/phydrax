#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState

from ._dyadic import DyadicCellTopology


class DyadicFieldTransferResult(StrictModule):
    """Transferred fixed-capacity leaf field and conservation evidence."""

    values: jax.Array
    conservation_residual: jax.Array
    finite: jax.Array
    successful: jax.Array


class DyadicCellTransferPlan(NonTrainableState, StrictModule):
    """Conservative routes between two accepted dyadic leaf partitions."""

    previous: DyadicCellTopology
    current: DyadicCellTopology
    targets: jax.Array
    sources: jax.Array
    average_weights: jax.Array
    content_weights: jax.Array
    active: jax.Array
    route_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        previous: DyadicCellTopology,
        current: DyadicCellTopology,
        /,
    ) -> None:
        if not isinstance(previous, DyadicCellTopology) or not isinstance(
            current, DyadicCellTopology
        ):
            raise TypeError("Dyadic transfer requires dyadic topologies.")
        if previous.address_plan.plan_id != current.address_plan.plan_id:
            raise ValueError("Dyadic transfer topologies use different address plans.")
        previous_leaf = np.flatnonzero(np.asarray(previous.leaf_active))
        current_leaf = np.flatnonzero(np.asarray(current.leaf_active))
        previous_prefix = np.asarray(previous.prefixes, dtype=np.uint64)
        previous_level = np.asarray(previous.levels, dtype=np.int32)
        previous_volume = np.asarray(previous.cell_volumes)
        current_prefix = np.asarray(current.prefixes, dtype=np.uint64)
        current_level = np.asarray(current.levels, dtype=np.int32)
        current_volume = np.asarray(current.cell_volumes)
        dimension = previous.address_plan.dimension
        depth = previous.address_plan.maximum_depth
        routes: list[tuple[int, int, float, float]] = []
        for target in current_leaf:
            target_shift = dimension * (depth - int(current_level[target]))
            target_start = int(current_prefix[target]) << target_shift
            target_end = (int(current_prefix[target]) + 1) << target_shift
            containing_source = None
            descendant_sources: list[int] = []
            for source in previous_leaf:
                source_shift = dimension * (depth - int(previous_level[source]))
                source_start = int(previous_prefix[source]) << source_shift
                source_end = (int(previous_prefix[source]) + 1) << source_shift
                if source_start <= target_start and source_end >= target_end:
                    containing_source = int(source)
                    break
                if target_start <= source_start and target_end >= source_end:
                    descendant_sources.append(int(source))
            if containing_source is not None:
                source = containing_source
                routes.append(
                    (
                        int(target),
                        source,
                        1.0,
                        float(current_volume[target] / previous_volume[source]),
                    )
                )
            else:
                for source in descendant_sources:
                    routes.append(
                        (
                            int(target),
                            source,
                            float(previous_volume[source] / current_volume[target]),
                            1.0,
                        )
                    )
        if not routes and current_leaf.size:
            raise ValueError("Dyadic topologies have no transferable leaf overlap.")
        capacity = max(len(routes), 1)
        targets = np.zeros((capacity,), dtype=np.int32)
        sources = np.zeros((capacity,), dtype=np.int32)
        average_weights = np.zeros((capacity,), dtype=float)
        content_weights = np.zeros((capacity,), dtype=float)
        active = np.zeros((capacity,), dtype=bool)
        for route, (target, source, average, content) in enumerate(routes):
            targets[route] = target
            sources[route] = source
            average_weights[route] = average
            content_weights[route] = content
            active[route] = True
        object.__setattr__(self, "previous", previous)
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "targets", jnp.asarray(targets))
        object.__setattr__(self, "sources", jnp.asarray(sources))
        object.__setattr__(self, "average_weights", jnp.asarray(average_weights))
        object.__setattr__(self, "content_weights", jnp.asarray(content_weights))
        object.__setattr__(self, "active", jnp.asarray(active))
        object.__setattr__(self, "route_count", len(routes))
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "dyadic-cell-transfer-plan",
                    "previous": previous.topology_id,
                    "current": current.topology_id,
                    "routes": len(routes),
                }
            ),
        )

    def apply_cell_averages(self, values: jax.Array) -> DyadicFieldTransferResult:
        value_array = self._validate_values(values)
        transferred = self._apply(value_array, self.average_weights)
        previous_content = self._integral(
            value_array,
            self.previous.cell_volumes,
            self.previous.leaf_active,
        )
        current_content = self._integral(
            transferred,
            self.current.cell_volumes,
            self.current.leaf_active,
        )
        residual = current_content - previous_content
        finite = jnp.all(jnp.isfinite(transferred)) & jnp.all(jnp.isfinite(residual))
        return DyadicFieldTransferResult(
            values=transferred,
            conservation_residual=residual,
            finite=finite,
            successful=(
                finite
                & self.previous.evidence.successful
                & self.current.evidence.successful
            ),
        )

    def apply_cell_contents(self, values: jax.Array) -> DyadicFieldTransferResult:
        value_array = self._validate_values(values)
        transferred = self._apply(value_array, self.content_weights)
        trailing_axes = tuple(range(1, value_array.ndim))
        previous_total = jnp.sum(
            jnp.where(
                self.previous.leaf_active.reshape(
                    self.previous.leaf_active.shape + (1,) * len(trailing_axes)
                ),
                value_array,
                0.0,
            ),
            axis=0,
        )
        current_total = jnp.sum(
            jnp.where(
                self.current.leaf_active.reshape(
                    self.current.leaf_active.shape + (1,) * len(trailing_axes)
                ),
                transferred,
                0.0,
            ),
            axis=0,
        )
        residual = current_total - previous_total
        finite = jnp.all(jnp.isfinite(transferred)) & jnp.all(jnp.isfinite(residual))
        return DyadicFieldTransferResult(
            values=transferred,
            conservation_residual=residual,
            finite=finite,
            successful=(
                finite
                & self.previous.evidence.successful
                & self.current.evidence.successful
            ),
        )

    def _validate_values(self, values: jax.Array) -> jax.Array:
        value_array = jnp.asarray(values)
        if value_array.shape[0] != self.previous.cell_capacity:
            raise ValueError("Dyadic field values must match previous cell capacity.")
        return value_array

    def _apply(self, values: jax.Array, weights: jax.Array) -> jax.Array:
        trailing = (1,) * (values.ndim - 1)
        contributions = values[self.sources] * weights.reshape(weights.shape + trailing)
        contributions = jnp.where(
            self.active.reshape(self.active.shape + trailing), contributions, 0.0
        )
        output = jnp.zeros(
            (self.current.cell_capacity,) + values.shape[1:], dtype=values.dtype
        )
        return output.at[self.targets].add(contributions)

    @staticmethod
    def _integral(values: jax.Array, volumes: jax.Array, active: jax.Array) -> jax.Array:
        trailing = (1,) * (values.ndim - 1)
        weight = volumes.reshape(volumes.shape + trailing)
        mask = active.reshape(active.shape + trailing)
        return jnp.sum(jnp.where(mask, values * weight, 0.0), axis=0)


__all__ = ["DyadicCellTransferPlan", "DyadicFieldTransferResult"]
