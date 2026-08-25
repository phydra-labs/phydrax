#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_complex import PolygonalConnectivity
from ._flux_ledger import FiniteVolumeStageFluxRateBlock
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_embedded_boundary import (
    EmbeddedBoundaryMetrics,
    EmbeddedBoundaryStabilizationPolicy,
    EmbeddedBoundaryStatus,
)


class ConservativeSmallCellRedistributionEvidence(NamedTuple):
    prepared_geometry_id: str
    topology_id: str
    geometry_id: str
    metrics_id: str
    policy_id: str
    plan_id: str
    small_cell_count: int
    route_count: int


class ConservativeSmallCellRedistributionReport(StrictModule, NonTrainableState):
    small_cell_count: Array
    route_count: Array
    maximum_route_weight_sum_defect: Array
    prepared_geometry_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    metrics_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class ConservativeSmallCellRedistributionResult(StrictModule):
    redistributed_rate: Array
    conservation_defect: Array
    activated: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    evidence: ConservativeSmallCellRedistributionEvidence = eqx.field(static=True)


class ConservativeSmallCellRedistributionPlan(StrictModule, NonTrainableState):
    """Fixed-route conservative redistribution of small-cell content rates.

    Preparation is host-side and deterministic. Execution only applies the frozen
    scatter: it does not inspect state, change connectivity, or clip an average.
    """

    active_cells: Array
    small_cells: Array
    source_cells: Array
    recipient_cells: Array
    recipient_face_ids: Array
    recipient_mask: Array
    weights: Array
    local_retention_fractions: Array
    report: ConservativeSmallCellRedistributionReport
    plan_id: str = eqx.field(static=True)
    evidence: ConservativeSmallCellRedistributionEvidence = eqx.field(static=True)
    redistribution_owner_cells: tuple[int, ...] = eqx.field(static=True)
    redistribution_neighbour_cells: tuple[int, ...] = eqx.field(static=True)
    redistribution_route_indices: tuple[int, ...] = eqx.field(static=True)
    redistribution_block_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        metrics: EmbeddedBoundaryMetrics,
        policy: EmbeddedBoundaryStabilizationPolicy,
        /,
    ):
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError(
                "Small-cell redistribution requires prepared unstructured FV geometry."
            )
        if discretization.cell_dimension != 2 or not isinstance(
            discretization.connectivity, PolygonalConnectivity
        ):
            raise ValueError(
                "Small-cell redistribution currently supports 2-D polygonal geometry."
            )
        if not isinstance(metrics, EmbeddedBoundaryMetrics):
            raise TypeError("metrics must be EmbeddedBoundaryMetrics.")
        if not isinstance(policy, EmbeddedBoundaryStabilizationPolicy):
            raise TypeError("policy must be EmbeddedBoundaryStabilizationPolicy.")
        if metrics.prepared_id != discretization.prepared_id:
            raise ValueError(
                "Embedded-boundary metrics do not bind the prepared geometry."
            )
        if metrics.topology_id != discretization.topology_id:
            raise ValueError("Embedded-boundary metrics do not bind the topology.")
        if metrics.geometry_id != discretization.geometry_id:
            raise ValueError("Embedded-boundary metrics do not bind the geometry.")
        if metrics.stabilization_policy_id != policy.policy_id:
            raise ValueError(
                "Embedded-boundary metrics do not bind the stabilization policy."
            )
        evidence_passed = bool(np.asarray(metrics.evidence.passed))
        evidence_status = int(np.asarray(metrics.evidence.status))
        if not evidence_passed or evidence_status != int(EmbeddedBoundaryStatus.SUCCESS):
            raise ValueError(
                "Small-cell redistribution requires passing embedded-boundary evidence."
            )

        cell_count = discretization.cell_count
        face_count = int(discretization.face_measures.size)
        owner_cells = np.asarray(discretization.owner_cells)
        neighbour_cells = np.asarray(discretization.neighbour_cells)
        stable_cell_ids = np.asarray(discretization.cell_global_ids)
        base_face_measures = np.asarray(discretization.face_measures)
        open_face_measures = np.asarray(metrics.open_face_measures)
        volume_fractions = np.asarray(metrics.volume_fraction)
        active_cells = np.asarray(metrics.active_fluid_cells)

        if (
            owner_cells.shape != (face_count,)
            or neighbour_cells.shape != (face_count,)
            or open_face_measures.shape != (face_count,)
            or base_face_measures.shape != (face_count,)
        ):
            raise ValueError(
                "Face routes and open face measures must contain one entry per face."
            )
        if owner_cells.dtype.kind not in "iu" or neighbour_cells.dtype.kind not in "iu":
            raise TypeError("Unstructured face routes must be integer arrays.")
        if (
            np.any(owner_cells < 0)
            or np.any(owner_cells >= cell_count)
            or np.any((neighbour_cells < -1) | (neighbour_cells >= cell_count))
        ):
            raise ValueError("Unstructured face routes contain an invalid cell index.")
        if (
            stable_cell_ids.shape != (cell_count,)
            or stable_cell_ids.dtype.kind not in "iu"
        ):
            raise ValueError("Stable cell IDs must contain one integer per cell.")
        if np.any(stable_cell_ids < 0) or np.unique(stable_cell_ids).size != cell_count:
            raise ValueError("Stable cell IDs must be unique and nonnegative.")
        if volume_fractions.shape != (cell_count,):
            raise ValueError("volume_fraction must contain one value per cell.")
        if active_cells.shape != (cell_count,) or active_cells.dtype.kind != "b":
            raise ValueError("active_fluid_cells must be one boolean per cell.")
        if np.any(~np.isfinite(volume_fractions)) or np.any(
            (volume_fractions < 0.0) | (volume_fractions > 1.0)
        ):
            raise ValueError("Fluid volume fractions must be finite and lie in [0, 1].")
        if not np.array_equal(active_cells, volume_fractions > 0.0):
            raise ValueError(
                "Active cells must be exactly the positive-volume-fraction cells."
            )
        if np.any(~np.isfinite(open_face_measures)) or np.any(open_face_measures < 0.0):
            raise ValueError("Open face measures must be finite and nonnegative.")
        measure_tolerance = (
            policy.absolute_tolerance
            + policy.relative_tolerance
            * np.maximum(base_face_measures, open_face_measures)
        )
        if np.any(open_face_measures - base_face_measures > measure_tolerance):
            raise ValueError("Open face measures cannot exceed base face measures.")

        minimum_fraction = policy.minimum_volume_fraction
        maximum_recipients = policy.maximum_recipients
        small_cells = active_cells & (volume_fractions < minimum_fraction)
        stable_recipient_cells = active_cells & (volume_fractions >= minimum_fraction)
        source_cells = np.flatnonzero(small_cells).astype(np.int32)
        evidence_small_cell_count = int(np.asarray(metrics.evidence.small_cell_count))
        if evidence_small_cell_count != int(source_cells.size):
            raise ValueError(
                "Embedded-boundary small-cell evidence does not match the policy."
            )
        recipient_cells = np.zeros(
            (source_cells.size, maximum_recipients), dtype=np.int32
        )
        recipient_face_ids = np.full(
            (source_cells.size, maximum_recipients), -1, dtype=np.int32
        )
        recipient_mask = np.zeros((source_cells.size, maximum_recipients), dtype=bool)
        weights = np.zeros(
            (source_cells.size, maximum_recipients), dtype=open_face_measures.dtype
        )

        incident_faces: list[list[int]] = [[] for _ in range(cell_count)]
        for face, (owner, neighbour) in enumerate(
            zip(owner_cells, neighbour_cells, strict=True)
        ):
            if neighbour >= 0:
                incident_faces[int(owner)].append(face)
                incident_faces[int(neighbour)].append(face)

        for row, source in enumerate(source_cells):
            candidates: list[tuple[float, int, int, int]] = []
            for face in incident_faces[int(source)]:
                owner = int(owner_cells[face])
                neighbour = int(neighbour_cells[face])
                recipient = neighbour if owner == source else owner
                measure = float(open_face_measures[face])
                if measure > 0.0 and stable_recipient_cells[recipient]:
                    candidates.append(
                        (measure, int(stable_cell_ids[recipient]), recipient, face)
                    )
            candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
            selected = candidates[:maximum_recipients]
            if not selected:
                raise ValueError(
                    f"Small active cell {int(stable_cell_ids[source])} has no "
                    "non-small open-face recipient."
                )
            measures = np.asarray(
                [candidate[0] for candidate in selected], dtype=open_face_measures.dtype
            )
            normalized = measures / np.sum(measures)
            if normalized.size > 1:
                normalized[-1] = 1.0 - np.sum(normalized[:-1])
            if np.any(~np.isfinite(normalized)) or np.any(normalized <= 0.0):
                raise ValueError(
                    "Small-cell recipient weights must be positive and finite."
                )
            count = len(selected)
            recipient_cells[row, :count] = [candidate[2] for candidate in selected]
            recipient_face_ids[row, :count] = [candidate[3] for candidate in selected]
            recipient_mask[row, :count] = True
            weights[row, :count] = normalized

        local_retention = np.where(active_cells, 1.0, 0.0).astype(volume_fractions.dtype)
        local_retention[source_cells] = volume_fractions[source_cells] / minimum_fraction
        if source_cells.size:
            weight_sum_defects = np.abs(np.sum(weights, axis=1) - 1.0)
            maximum_weight_sum_defect = float(np.max(weight_sum_defects))
        else:
            maximum_weight_sum_defect = 0.0
        weight_tolerance = policy.absolute_tolerance + policy.relative_tolerance
        if maximum_weight_sum_defect > weight_tolerance:
            raise ValueError(
                "Small-cell recipient weights fail the policy conservation tolerance."
            )

        plan_id = canonical_fingerprint(
            {
                "kind": "conservative-small-cell-redistribution",
                "prepared_geometry": discretization.prepared_id,
                "topology": discretization.topology_id,
                "geometry": discretization.geometry_id,
                "metrics": metrics.metrics_id,
                "policy": policy.policy_id,
                "active_cells": array_tree_fingerprint(active_cells),
                "small_cells": array_tree_fingerprint(small_cells),
                "source_cells": array_tree_fingerprint(source_cells),
                "recipient_cells": array_tree_fingerprint(recipient_cells),
                "recipient_face_ids": array_tree_fingerprint(recipient_face_ids),
                "recipient_mask": array_tree_fingerprint(recipient_mask),
                "weights": array_tree_fingerprint(weights),
                "local_retention_fractions": array_tree_fingerprint(local_retention),
            }
        )
        route_count = int(np.sum(recipient_mask))
        route_rows, route_slots = np.nonzero(recipient_mask)
        redistribution_owner_cells = source_cells[route_rows]
        redistribution_neighbour_cells = recipient_cells[route_rows, route_slots]
        redistribution_face_ids = recipient_face_ids[route_rows, route_slots]
        redistribution_weights = weights[route_rows, route_slots]
        if route_count:
            redistribution_route_indices = np.ravel_multi_index(
                (route_rows, route_slots), recipient_mask.shape
            )
        else:
            redistribution_route_indices = np.empty((0,), dtype=np.int64)
        redistribution_block_id = canonical_fingerprint(
            {
                "kind": "small-cell-redistribution-stage-flux-rate-block",
                "prepared_geometry": discretization.prepared_id,
                "topology": discretization.topology_id,
                "geometry": discretization.geometry_id,
                "metrics": metrics.metrics_id,
                "policy": policy.policy_id,
                "plan": plan_id,
                "owner_cells": array_tree_fingerprint(redistribution_owner_cells),
                "neighbour_cells": array_tree_fingerprint(redistribution_neighbour_cells),
                "active_mask": array_tree_fingerprint(
                    np.ones((route_count,), dtype=bool)
                ),
                "recipient_face_ids": array_tree_fingerprint(redistribution_face_ids),
                "weights": array_tree_fingerprint(redistribution_weights),
            }
        )
        evidence = ConservativeSmallCellRedistributionEvidence(
            prepared_geometry_id=discretization.prepared_id,
            topology_id=discretization.topology_id,
            geometry_id=discretization.geometry_id,
            metrics_id=metrics.metrics_id,
            policy_id=policy.policy_id,
            plan_id=plan_id,
            small_cell_count=int(source_cells.size),
            route_count=route_count,
        )
        self.active_cells = jnp.asarray(active_cells)
        self.small_cells = jnp.asarray(small_cells)
        self.source_cells = jnp.asarray(source_cells)
        self.recipient_cells = jnp.asarray(recipient_cells)
        self.recipient_face_ids = jnp.asarray(recipient_face_ids)
        self.recipient_mask = jnp.asarray(recipient_mask)
        self.weights = jnp.asarray(weights)
        self.local_retention_fractions = jnp.asarray(local_retention)
        self.redistribution_owner_cells = tuple(
            int(cell) for cell in redistribution_owner_cells
        )
        self.redistribution_neighbour_cells = tuple(
            int(cell) for cell in redistribution_neighbour_cells
        )
        self.redistribution_route_indices = tuple(
            int(index) for index in redistribution_route_indices
        )
        self.redistribution_block_id = redistribution_block_id
        self.report = ConservativeSmallCellRedistributionReport(
            small_cell_count=jnp.asarray(source_cells.size, dtype=jnp.int32),
            route_count=jnp.asarray(route_count, dtype=jnp.int32),
            maximum_route_weight_sum_defect=jnp.asarray(maximum_weight_sum_defect),
            prepared_geometry_id=discretization.prepared_id,
            topology_id=discretization.topology_id,
            geometry_id=discretization.geometry_id,
            metrics_id=metrics.metrics_id,
            policy_id=policy.policy_id,
            plan_id=plan_id,
        )
        self.plan_id = plan_id
        self.evidence = evidence

    def _redistribution_terms(
        self, content_rate: ArrayLike, /
    ) -> tuple[Array, Array, Array, Array, Array]:
        rate = jnp.asarray(content_rate)
        cell_count = int(self.active_cells.size)
        if rate.ndim < 1 or rate.shape[0] != cell_count:
            raise ValueError(
                "content_rate must have a leading axis with one entry per cell."
            )
        if not jnp.issubdtype(rate.dtype, jnp.inexact):
            raise TypeError("content_rate must have an inexact floating-point dtype.")
        rate = eqx.error_if(
            rate,
            jnp.any(~jnp.isfinite(rate)),
            "Small-cell content rates must be finite.",
        )
        cell_broadcast_shape = (cell_count,) + (1,) * (rate.ndim - 1)
        active = self.active_cells.reshape(cell_broadcast_shape)
        rate = eqx.error_if(
            rate,
            jnp.any(jnp.where(active, False, rate != 0.0)),
            "Inactive-cell content rates must be exactly zero.",
        )
        active_rate = jnp.where(active, rate, jnp.zeros((), dtype=rate.dtype))
        source_rate = active_rate[self.source_cells]
        source_count = int(self.source_cells.size)
        retention_shape = (source_count,) + (1,) * (rate.ndim - 1)
        retained = (
            self.local_retention_fractions[self.source_cells]
            .astype(rate.dtype)
            .reshape(retention_shape)
        )
        excess = (1.0 - retained) * source_rate

        maximum_recipients = self.recipient_cells.shape[1]
        cast_weights = self.weights.astype(rate.dtype)
        recipient_mask = self.recipient_mask
        cast_weight_is_positive_real = (jnp.real(cast_weights) > 0.0) & (
            jnp.imag(cast_weights) == 0.0
        )
        cast_weight_is_valid = jnp.isfinite(cast_weights) & (cast_weight_is_positive_real)
        cast_weights = eqx.error_if(
            cast_weights,
            jnp.any(recipient_mask & ~cast_weight_is_valid),
            "Small-cell recipient weights underflow or are not positive and finite "
            "in the content-rate dtype.",
        )
        cast_weights = jnp.where(
            recipient_mask, cast_weights, jnp.zeros((), dtype=rate.dtype)
        )
        row_weight_sums = jnp.sum(cast_weights, axis=1)
        row_sum_is_positive_real = (jnp.real(row_weight_sums) > 0.0) & (
            jnp.imag(row_weight_sums) == 0.0
        )
        row_sum_is_valid = jnp.isfinite(row_weight_sums) & (row_sum_is_positive_real)
        row_weight_sums = eqx.error_if(
            row_weight_sums,
            jnp.any(~row_sum_is_valid),
            "Small-cell recipient weight rows must have finite positive sums in "
            "the content-rate dtype.",
        )
        normalized_weights = jnp.where(
            recipient_mask,
            cast_weights / row_weight_sums[:, None],
            jnp.zeros((), dtype=rate.dtype),
        )
        closure_route = jnp.argmax(jnp.real(normalized_weights), axis=1)
        closure_mask = recipient_mask & (
            jnp.arange(maximum_recipients)[None, :] == closure_route[:, None]
        )
        other_weight_sum = jnp.sum(
            jnp.where(
                closure_mask,
                jnp.zeros((), dtype=rate.dtype),
                normalized_weights,
            ),
            axis=1,
        )
        normalized_weights = jnp.where(
            closure_mask,
            jnp.ones((), dtype=rate.dtype) - other_weight_sum[:, None],
            normalized_weights,
        )
        normalized_weight_is_positive_real = (jnp.real(normalized_weights) > 0.0) & (
            jnp.imag(normalized_weights) == 0.0
        )
        normalized_weights = eqx.error_if(
            normalized_weights,
            jnp.any(
                recipient_mask
                & cast_weight_is_valid
                & row_sum_is_valid[:, None]
                & (
                    ~jnp.isfinite(normalized_weights)
                    | ~normalized_weight_is_positive_real
                )
            ),
            "Small-cell recipient weight normalization is not representable in "
            "the content-rate dtype.",
        )

        weight_shape = (
            source_count,
            maximum_recipients,
        ) + (1,) * (rate.ndim - 1)
        contributions = normalized_weights.reshape(weight_shape) * excess[:, None, ...]
        return rate, active, active_rate, excess, contributions

    def redistribution_flux_rate_block(
        self, content_rate: ArrayLike, /
    ) -> FiniteVolumeStageFluxRateBlock | None:
        """Expose redistribution as an owner-outward stage flux-rate block."""
        route_count = self.evidence.route_count
        if route_count == 0:
            return None
        rate, _, _, _, contributions = self._redistribution_terms(content_rate)
        flat_contributions = contributions.reshape((-1,) + rate.shape[1:])
        route_flux_rate = flat_contributions[
            jnp.asarray(self.redistribution_route_indices, dtype=jnp.int32)
        ]
        return FiniteVolumeStageFluxRateBlock(
            route_flux_rate,
            self.redistribution_owner_cells,
            self.redistribution_neighbour_cells,
            (True,) * route_count,
            self.redistribution_block_id,
            "small-cell-redistribution",
        )

    def redistribute_rate(
        self, content_rate: ArrayLike, /
    ) -> ConservativeSmallCellRedistributionResult:
        """Redistribute a cell-leading content rate on the prepared fixed routes."""
        rate, active, active_rate, excess, contributions = self._redistribution_terms(
            content_rate
        )
        cell_count = int(self.active_cells.size)
        source_count = int(self.source_cells.size)
        maximum_recipients = self.recipient_cells.shape[1]
        redistributed = active_rate.at[self.source_cells].add(-excess)
        flat_recipients = self.recipient_cells.reshape((-1,))
        flat_contributions = contributions.reshape(
            (source_count * maximum_recipients,) + rate.shape[1:]
        )
        redistributed = redistributed.at[flat_recipients].add(flat_contributions)
        redistributed = jnp.where(
            active, redistributed, jnp.zeros((), dtype=redistributed.dtype)
        )
        conservation_defect = jnp.sum(redistributed, axis=0) - jnp.sum(
            active_rate, axis=0
        )
        real_dtype = jnp.real(rate).dtype
        roundoff_steps = max(1, cell_count + source_count * maximum_recipients)
        conservation_scale = jnp.maximum(
            jnp.sum(jnp.abs(active_rate), axis=0),
            jnp.sum(jnp.abs(redistributed), axis=0),
        )
        conservation_tolerance = (
            jnp.asarray(8 * roundoff_steps, dtype=real_dtype)
            * jnp.asarray(jnp.finfo(real_dtype).eps, dtype=real_dtype)
            * conservation_scale
        )
        conservation_failed = (
            ~jnp.isfinite(conservation_defect)
            | ~jnp.isfinite(conservation_tolerance)
            | (jnp.abs(conservation_defect) > conservation_tolerance)
        )
        redistributed = eqx.error_if(
            redistributed,
            jnp.any(conservation_failed),
            "Small-cell redistribution exceeds the content-rate dtype conservation "
            "tolerance.",
        )
        conservation_defect = jnp.sum(redistributed, axis=0) - jnp.sum(
            active_rate, axis=0
        )
        return ConservativeSmallCellRedistributionResult(
            redistributed_rate=redistributed,
            conservation_defect=conservation_defect,
            activated=source_count > 0,
            plan_id=self.plan_id,
            evidence=self.evidence,
        )


__all__ = [
    "ConservativeSmallCellRedistributionEvidence",
    "ConservativeSmallCellRedistributionPlan",
    "ConservativeSmallCellRedistributionReport",
    "ConservativeSmallCellRedistributionResult",
]
