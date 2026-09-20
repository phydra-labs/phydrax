#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._admissibility import AdmissibilityHeader, AdmissibilityReason
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class DLDOutletClassification(StrictModule):
    terminal_mask: Array
    outlet_code: Array
    ambiguous: Array
    plan_id: str = eqx.field(static=True)


class DLDOutletPlan(StrictModule, NonTrainableState):
    outlet_x: float = eqx.field(static=True)
    transverse_edges: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, outlet_x: float, transverse_edges: ArrayLike, /) -> None:
        location = float(outlet_x)
        edges = np.asarray(transverse_edges, dtype=np.float64)
        if (
            not np.isfinite(location)
            or edges.ndim != 1
            or edges.size < 2
            or np.any(~np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("DLD outlet plane or interval edges are invalid.")
        self.outlet_x = location
        self.transverse_edges = jnp.asarray(edges)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dld-outlet-plan",
                "outlet_x": location,
                "transverse_edges": array_tree_fingerprint(edges),
                "interval_convention": "left-closed-right-open-final-closed",
            }
        )

    @property
    def outlet_count(self) -> int:
        return self.transverse_edges.size - 1

    def classify(
        self, positions: ArrayLike, active: ArrayLike, /
    ) -> DLDOutletClassification:
        position = jnp.asarray(positions)
        active_ = jnp.asarray(active, dtype=jnp.bool_)
        if (
            position.ndim != 2
            or position.shape[1] != 2
            or active_.shape != position.shape[:1]
        ):
            raise ValueError("DLD outlet classification requires planar particle slots.")
        terminal = active_ & (position[:, 0] >= self.outlet_x)
        transverse = position[:, 1]
        raw = jnp.searchsorted(self.transverse_edges, transverse, side="right") - 1
        raw = jnp.where(
            transverse == self.transverse_edges[-1], self.outlet_count - 1, raw
        )
        valid = (raw >= 0) & (raw < self.outlet_count)
        code = jnp.where(terminal & valid, raw, jnp.where(terminal, -3, -1)).astype(
            jnp.int32
        )
        return DLDOutletClassification(
            terminal,
            code,
            terminal & ~valid,
            self.plan_id,
        )


class DLDSeparationMetrics(StrictModule):
    transfer_matrix: Array
    purity: Array
    recovery: Array
    contamination: Array
    throughput_proxy: Array
    pressure_drop: Array
    hydraulic_resistance: Array
    residence_time: Array
    terminal_fraction: Array
    ambiguous_count: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class DLDMetricPlan(StrictModule, NonTrainableState):
    class_count: int = eqx.field(static=True)
    target_outlet_by_class: Array
    outlet_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        class_count: int,
        outlet_count: int,
        target_outlet_by_class: ArrayLike,
        /,
    ) -> None:
        classes = int(class_count)
        outlets = int(outlet_count)
        target = np.asarray(target_outlet_by_class, dtype=np.int32)
        if (
            classes <= 0
            or outlets <= 0
            or target.shape != (classes,)
            or np.any((target < 0) | (target >= outlets))
        ):
            raise ValueError("DLD metric class-to-outlet mapping is invalid.")
        self.class_count = classes
        self.outlet_count = outlets
        self.target_outlet_by_class = jnp.asarray(target)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dld-separation-metrics",
                "class_count": classes,
                "outlet_count": outlets,
                "target_outlet_by_class": tuple(target),
            }
        )

    def evaluate(
        self,
        particle_class: ArrayLike,
        initial_active: ArrayLike,
        terminal_code: ArrayLike,
        terminal_time: ArrayLike,
        volume_flow: ArrayLike,
        pressure_drop: ArrayLike,
        /,
    ) -> DLDSeparationMetrics:
        particle_class_ = jnp.asarray(particle_class, dtype=jnp.int32)
        initial = jnp.asarray(initial_active, dtype=jnp.bool_)
        terminal = jnp.asarray(terminal_code, dtype=jnp.int32)
        time = jnp.asarray(terminal_time)
        flow = jnp.asarray(volume_flow, dtype=time.dtype)
        pressure_drop_ = jnp.asarray(pressure_drop, dtype=time.dtype)
        shape = particle_class_.shape
        if (
            initial.shape != shape
            or terminal.shape != shape
            or time.shape != shape
            or flow.shape != ()
            or pressure_drop_.shape != ()
        ):
            raise ValueError(
                "DLD particle metric arrays and hydraulic scalars are incompatible."
            )
        valid_class = (particle_class_ >= 0) & (particle_class_ < self.class_count)
        valid_outlet = (terminal >= 0) & (terminal < self.outlet_count)
        valid = initial & valid_class & valid_outlet
        safe_class = jnp.clip(particle_class_, 0, self.class_count - 1)
        safe_outlet = jnp.clip(terminal, 0, self.outlet_count - 1)
        matrix = (
            jnp.zeros((self.class_count, self.outlet_count), dtype=time.dtype)
            .at[safe_class, safe_outlet]
            .add(valid.astype(time.dtype))
        )
        desired = self.target_outlet_by_class
        desired_count = matrix[jnp.arange(self.class_count), desired]
        class_total = (
            jnp.zeros((self.class_count,), dtype=time.dtype)
            .at[safe_class]
            .add((initial & valid_class).astype(time.dtype))
        )
        outlet_total = jnp.sum(matrix, axis=0)
        desired_by_outlet = (
            jnp.zeros((self.outlet_count,), dtype=time.dtype)
            .at[desired]
            .add(desired_count)
        )
        purity = desired_by_outlet / jnp.maximum(outlet_total, 1.0)
        recovery = desired_count / jnp.maximum(class_total, 1.0)
        contamination = 1.0 - purity
        terminal_count = jnp.sum(valid)
        initial_count = jnp.sum(initial)
        residence = jnp.sum(jnp.where(valid, time, 0.0)) / jnp.maximum(terminal_count, 1)
        terminal_fraction = terminal_count / jnp.maximum(initial_count, 1)
        ambiguous_count = jnp.sum(initial & (terminal == -3))
        throughput = jnp.abs(flow) * terminal_fraction
        hydraulic_resistance = pressure_drop_ / jnp.abs(flow)
        complete = terminal_count == initial_count
        finite = (
            jnp.all(jnp.isfinite(matrix))
            & jnp.all(jnp.isfinite(purity))
            & jnp.all(jnp.isfinite(recovery))
            & jnp.isfinite(residence)
            & jnp.isfinite(throughput)
            & jnp.isfinite(pressure_drop_)
            & jnp.isfinite(hydraulic_resistance)
        )
        supported = (
            finite
            & complete
            & (ambiguous_count == 0)
            & (jnp.abs(flow) > 0.0)
            & (pressure_drop_ >= 0.0)
        )
        reasons = jnp.where(
            supported,
            jnp.asarray(0, dtype=jnp.uint32),
            jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), dtype=jnp.uint32),
        )
        header = AdmissibilityHeader(
            jnp.where(supported, 1.0, -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint({"kind": "dld-metric-evidence", "plan": self.plan_id}),
        )
        return DLDSeparationMetrics(
            matrix,
            purity,
            recovery,
            contamination,
            throughput,
            pressure_drop_,
            hydraulic_resistance,
            residence,
            terminal_fraction,
            ambiguous_count,
            header,
            self.plan_id,
        )


__all__ = [
    "DLDMetricPlan",
    "DLDOutletClassification",
    "DLDOutletPlan",
    "DLDSeparationMetrics",
]
