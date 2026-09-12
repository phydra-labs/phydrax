#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Static cochain incidence layouts and traceable dynamic metric states."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._cochain import CochainDiscretization
from ._core import DiscretizationKey, DiscretizationRole
from ._topology import CellComplexTopology


class PreparedCochainTopology(StrictModule, NonTrainableState):
    """Immutable incidence and boundary realization shared across metric versions."""

    topology: CellComplexTopology
    key: DiscretizationKey
    boundary_masks: tuple[Array, ...]
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: CellComplexTopology,
        /,
        *,
        boundary_masks: Sequence[ArrayLike] | None = None,
        key: DiscretizationKey | None = None,
    ):
        if not isinstance(topology, CellComplexTopology):
            raise TypeError("Prepared cochain topology requires CellComplexTopology.")
        counts = tuple(entity.count for entity in topology.entity_sets)
        masks = (
            tuple(np.zeros((count,), dtype=bool) for count in counts)
            if boundary_masks is None
            else tuple(np.asarray(mask, dtype=bool) for mask in boundary_masks)
        )
        if len(masks) != len(counts) or any(
            mask.shape != (count,) for mask, count in zip(masks, counts, strict=True)
        ):
            raise ValueError(
                "Cochain topology boundary masks must match entity capacities."
            )
        key_ = (
            DiscretizationKey(
                "cochain",
                DiscretizationRole.PHYSICAL,
                domain_labels=("entity",),
            )
            if key is None
            else key
        )
        if not isinstance(key_, DiscretizationKey):
            raise TypeError("Cochain topology key must be DiscretizationKey.")
        self.topology = topology
        self.key = key_
        self.boundary_masks = tuple(jnp.asarray(mask) for mask in masks)
        self.topology_id = canonical_fingerprint(
            {
                "kind": "prepared-cochain-topology",
                "topology": topology.topology_id,
                "boundary_masks": [array_tree_fingerprint(mask) for mask in masks],
                "key": key_.key_id,
            }
        )


class CochainMetricEvidence(StrictModule, NonTrainableState):
    """Host-admitted geometry family/layout and active-capacity evidence."""

    topology_id: str = eqx.field(static=True)
    geometry_family_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    active_counts: tuple[int, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology_id: str,
        geometry_family_id: str,
        geometry_layout_id: str,
        active_masks: Sequence[ArrayLike],
        /,
    ):
        family = str(geometry_family_id)
        layout = str(geometry_layout_id)
        masks = tuple(np.asarray(mask, dtype=bool) for mask in active_masks)
        if not topology_id or not family or not layout:
            raise ValueError("Cochain metric identities must be non-empty.")
        self.topology_id = topology_id
        self.geometry_family_id = family
        self.geometry_layout_id = layout
        self.active_counts = tuple(int(np.count_nonzero(mask)) for mask in masks)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "cochain-metric-evidence",
                "topology": topology_id,
                "geometry_family": family,
                "geometry_layout": layout,
                "active_masks": [array_tree_fingerprint(mask) for mask in masks],
            }
        )


class CochainMetricPlan(StrictModule, NonTrainableState):
    """Static metric tensor layout; numeric metric values remain runtime arrays."""

    prepared_topology: PreparedCochainTopology
    geometry_family_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    coordinate_shapes: tuple[tuple[int, int] | None, ...] = eqx.field(static=True)
    metric_layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared_topology: PreparedCochainTopology,
        /,
        *,
        geometry_family_id: str,
        geometry_layout_id: str,
        coordinate_shapes: Sequence[tuple[int, int] | None] | None = None,
    ):
        if not isinstance(prepared_topology, PreparedCochainTopology):
            raise TypeError("Cochain metric plan requires PreparedCochainTopology.")
        counts = tuple(entity.count for entity in prepared_topology.topology.entity_sets)
        shapes = (
            (None,) * len(counts)
            if coordinate_shapes is None
            else tuple(coordinate_shapes)
        )
        if len(shapes) != len(counts) or any(
            shape is not None
            and (len(shape) != 2 or int(shape[0]) != count or int(shape[1]) <= 0)
            for shape, count in zip(shapes, counts, strict=True)
        ):
            raise ValueError("Cochain metric coordinate shapes must match capacities.")
        family = str(geometry_family_id)
        layout = str(geometry_layout_id)
        if not family or not layout:
            raise ValueError("Cochain metric geometry identities must be non-empty.")
        self.prepared_topology = prepared_topology
        self.geometry_family_id = family
        self.geometry_layout_id = layout
        self.coordinate_shapes = tuple(
            None if shape is None else (int(shape[0]), int(shape[1])) for shape in shapes
        )
        self.metric_layout_id = canonical_fingerprint(
            {
                "kind": "cochain-metric-plan",
                "topology": prepared_topology.topology_id,
                "geometry_family": family,
                "geometry_layout": layout,
                "coordinate_shapes": self.coordinate_shapes,
            }
        )

    def state(
        self,
        hodge_stars: Sequence[ArrayLike],
        /,
        *,
        primal_measures: Sequence[ArrayLike],
        dual_measures: Sequence[ArrayLike] | None = None,
        coordinates: Sequence[ArrayLike | None] | None = None,
        active_masks: Sequence[ArrayLike] | None = None,
        time: ArrayLike = 0.0,
        revision: ArrayLike = 0,
    ) -> "CochainMetricState":
        return CochainMetricState(
            self,
            hodge_stars,
            primal_measures=primal_measures,
            dual_measures=dual_measures,
            coordinates=coordinates,
            active_masks=active_masks,
            time=time,
            revision=revision,
        )

    def prepare(
        self,
        hodge_stars: Sequence[ArrayLike],
        /,
        *,
        primal_measures: Sequence[ArrayLike],
        dual_measures: Sequence[ArrayLike] | None = None,
        coordinates: Sequence[ArrayLike | None] | None = None,
        active_masks: Sequence[ArrayLike] | None = None,
        time: ArrayLike = 0.0,
        revision: int = 0,
    ) -> "CochainMetricState":
        """Host-validate one accepted metric snapshot before entering a runtime epoch."""
        state = self.state(
            hodge_stars,
            primal_measures=primal_measures,
            dual_measures=dual_measures,
            coordinates=coordinates,
            active_masks=active_masks,
            time=time,
            revision=revision,
        )
        if isinstance(state.valid, jax.core.Tracer) or not bool(state.valid):
            raise ValueError("Cochain metric snapshot is inadmissible.")
        return state


class CochainMetricState(StrictModule):
    """Traceable numeric cochain metrics over one immutable topology layout.

    Arrays are intentionally accepted through ``jnp.asarray`` only.  The static
    layout ID never fingerprints metric contents; stage consumers must bind their
    numeric revision/evidence explicitly rather than confusing shape identity with a
    physical geometry revision.
    """

    plan: CochainMetricPlan
    time: Array
    revision: Array
    coordinates: tuple[Array | None, ...]
    active_masks: tuple[Array, ...]
    primal_measures: tuple[Array, ...]
    dual_measures: tuple[Array, ...]
    hodge_stars: tuple[Array, ...]
    valid: Array

    def __init__(
        self,
        plan: CochainMetricPlan,
        hodge_stars: Sequence[ArrayLike],
        /,
        *,
        primal_measures: Sequence[ArrayLike],
        dual_measures: Sequence[ArrayLike] | None = None,
        coordinates: Sequence[ArrayLike | None] | None = None,
        active_masks: Sequence[ArrayLike] | None = None,
        time: ArrayLike = 0.0,
        revision: ArrayLike = 0,
    ):
        if not isinstance(plan, CochainMetricPlan):
            raise TypeError("Cochain metric state requires CochainMetricPlan.")
        counts = tuple(
            entity.count for entity in plan.prepared_topology.topology.entity_sets
        )
        stars = tuple(jnp.asarray(value) for value in hodge_stars)
        primal = tuple(jnp.asarray(value) for value in primal_measures)
        if (
            len(stars) != len(counts)
            or len(primal) != len(counts)
            or any(
                value.shape != (count,)
                for value, count in (
                    *zip(stars, counts, strict=True),
                    *zip(primal, counts, strict=True),
                )
            )
        ):
            raise ValueError("Cochain metric arrays must match topology capacities.")
        if any(
            not jnp.issubdtype(value.dtype, jnp.inexact) for value in (*stars, *primal)
        ):
            raise TypeError("Cochain metric stars and measures must use inexact dtypes.")
        dual = (
            tuple(
                primal_value * star
                for primal_value, star in zip(primal, stars, strict=True)
            )
            if dual_measures is None
            else tuple(jnp.asarray(value) for value in dual_measures)
        )
        if len(dual) != len(counts) or any(
            value.shape != (count,) for value, count in zip(dual, counts, strict=True)
        ):
            raise ValueError("Cochain dual measures must match topology capacities.")
        if any(not jnp.issubdtype(value.dtype, jnp.inexact) for value in dual):
            raise TypeError("Cochain dual measures must use an inexact dtype.")
        masks = (
            tuple(
                entity.active_mask
                for entity in plan.prepared_topology.topology.entity_sets
            )
            if active_masks is None
            else tuple(jnp.asarray(mask, dtype=bool) for mask in active_masks)
        )
        if len(masks) != len(counts) or any(
            mask.shape != (count,) for mask, count in zip(masks, counts, strict=True)
        ):
            raise ValueError(
                "Cochain metric active masks must match topology capacities."
            )
        topology_active = tuple(
            entity.active_mask for entity in plan.prepared_topology.topology.entity_sets
        )
        coordinates_ = (
            (None,) * len(counts)
            if coordinates is None
            else tuple(
                None if value is None else jnp.asarray(value) for value in coordinates
            )
        )
        if len(coordinates_) != len(counts) or any(
            (expected is None) != (value is None)
            or (
                expected is not None
                and (
                    value.ndim != 2
                    or value.shape != expected
                    or not jnp.issubdtype(value.dtype, jnp.inexact)
                )
            )
            for expected, value in zip(
                plan.coordinate_shapes,
                coordinates_,
                strict=True,
            )
        ):
            raise ValueError("Cochain metric coordinates do not match the metric plan.")
        time_ = jnp.asarray(time)
        revision_ = jnp.asarray(revision)
        if (
            time_.shape != ()
            or revision_.shape != ()
            or not jnp.issubdtype(time_.dtype, jnp.inexact)
            or not jnp.issubdtype(revision_.dtype, jnp.integer)
        ):
            raise TypeError(
                "Cochain metric time must be scalar/inexact and revision scalar/integer."
            )
        valid_terms = [jnp.all(jnp.isfinite(time_))]
        for active, allowed, star, primal_value, dual_value in zip(
            masks,
            topology_active,
            stars,
            primal,
            dual,
            strict=True,
        ):
            valid_terms.extend(
                (
                    jnp.all(~active | allowed),
                    jnp.all(~active | jnp.isfinite(star)),
                    jnp.all(~active | jnp.isfinite(primal_value)),
                    jnp.all(~active | jnp.isfinite(dual_value)),
                    jnp.all(
                        ~active
                        | ((star > 0.0) & (primal_value > 0.0) & (dual_value > 0.0))
                    ),
                )
            )
        for coordinate, active in zip(coordinates_, masks, strict=True):
            if coordinate is not None:
                valid_terms.append(
                    jnp.all(
                        ~active.reshape(active.shape + (1,)) | jnp.isfinite(coordinate)
                    )
                )
        self.plan = plan
        self.time = time_
        self.revision = revision_.astype(jnp.int32)
        self.coordinates = coordinates_
        self.active_masks = masks
        self.primal_measures = primal
        self.dual_measures = dual
        self.hodge_stars = stars
        self.valid = jnp.all(jnp.stack(tuple(valid_terms)))

    @property
    def metric_layout_id(self) -> str:
        return self.plan.metric_layout_id

    def storage_hodge_stars(self, /) -> tuple[Array, ...]:
        return tuple(
            jnp.where(mask, star, jnp.ones((), dtype=star.dtype))
            for mask, star in zip(self.active_masks, self.hodge_stars, strict=True)
        )

    def storage_primal_measures(self, /) -> tuple[Array, ...]:
        return tuple(
            jnp.where(mask, measure, jnp.ones((), dtype=measure.dtype))
            for mask, measure in zip(
                self.active_masks,
                self.primal_measures,
                strict=True,
            )
        )

    def storage_dual_measures(self, /) -> tuple[Array, ...]:
        return tuple(
            jnp.where(mask, measure, jnp.ones((), dtype=measure.dtype))
            for mask, measure in zip(self.active_masks, self.dual_measures, strict=True)
        )

    def storage_coordinates(self, /) -> tuple[Array | None, ...]:
        return tuple(
            None
            if coordinate is None
            else jnp.where(
                mask.reshape(mask.shape + (1,)),
                coordinate,
                jnp.zeros((), dtype=coordinate.dtype),
            )
            for coordinate, mask in zip(
                self.coordinates,
                self.active_masks,
                strict=True,
            )
        )

    def host_snapshot(self, /) -> CochainDiscretization:
        """Create a host-only metric-space snapshot from a valid accepted state."""
        if isinstance(self.valid, jax.core.Tracer) or not bool(self.valid):
            raise ValueError(
                "Only an accepted host cochain metric state may be snapshotted."
            )
        return CochainDiscretization(
            self.plan.prepared_topology.topology,
            self.storage_hodge_stars(),
            primal_measures=self.storage_primal_measures(),
            dual_measures=self.storage_dual_measures(),
            boundary_masks=self.plan.prepared_topology.boundary_masks,
            coordinates=self.storage_coordinates(),
            key=self.plan.prepared_topology.key,
            plan_id=self.metric_layout_id,
            numeric_version=f"metric-revision-{int(self.revision)}",
        )


__all__ = [
    "CochainMetricEvidence",
    "CochainMetricPlan",
    "CochainMetricState",
    "PreparedCochainTopology",
]
