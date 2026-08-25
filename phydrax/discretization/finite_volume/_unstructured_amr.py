#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_remap import UnstructuredConservativeRemapPlan


def _amr_identity(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string.")
    return value


def _scalar_time(value: ArrayLike, name: str, /) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or not np.isfinite(array.item()):
        raise ValueError(f"{name} must be a finite scalar.")
    return float(array.item())


def _time_tolerance(first: float, second: float) -> float:
    return 64.0 * np.finfo(float).eps * max(1.0, abs(first), abs(second))


def _interval(
    value: ArrayLike | tuple[ArrayLike, ArrayLike] | None, name: str, /
) -> tuple[float, float] | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape != (2,):
        raise ValueError(f"{name} must be a two-entry (start, end) interval.")
    start, end = (
        _scalar_time(array[0], f"{name}[0]"),
        _scalar_time(array[1], f"{name}[1]"),
    )
    if end <= start:
        raise ValueError(f"{name} must have end greater than start.")
    return start, end


def _required_interval(
    value: ArrayLike | tuple[ArrayLike, ArrayLike] | None, name: str, /
) -> tuple[float, float]:
    interval = _interval(value, name)
    if interval is None:
        raise ValueError(f"{name} must be supplied.")
    return interval


def _mask(value: ArrayLike | None, count: int, name: str, /) -> Array:
    if value is None:
        return jnp.ones((count,), dtype=bool)
    array = jnp.asarray(value)
    if array.shape != (count,) or array.dtype != jnp.dtype(bool):
        raise ValueError(f"{name} must be boolean with one entry per cell.")
    return array


class UnstructuredAMRSelection(StrictModule):
    coarse_refined: Array
    fine_active: Array
    selected_count: Array
    eligible_count: Array
    capacity_overflow: Array
    overflow_count: Array
    overflow_status: Array


class UnstructuredAMRFluxRegister(StrictModule):
    """Accepted coarse/fine content mismatch for one exact time interval.

    The mismatch is formed as ``fine_flux_integral - coarse_flux_integral``;
    both inputs are already accepted (time-integrated) ledger quantities.  No
    additional time-step multiplication is performed by this register.
    """

    integrated_correction: Array
    coarse_flux_integral: Array
    fine_flux_integral: Array
    flux_mismatch: Array
    start_time: Array
    end_time: Array
    fine_interval_starts: Array
    fine_interval_ends: Array
    fine_accepted_steps: Array
    route_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    coarse_topology_id: str = eqx.field(static=True)
    fine_topology_id: str = eqx.field(static=True)
    register_id: str = eqx.field(static=True)

    def __init__(
        self,
        integrated_correction: ArrayLike,
        fine_flux_integral: ArrayLike | None = None,
        /,
        *,
        coarse_flux_integral: ArrayLike | None = None,
        coarse_flux: ArrayLike | None = None,
        fine_flux: ArrayLike | None = None,
        coarse_interval: ArrayLike | None = None,
        fine_intervals: tuple[ArrayLike, ...] | list[ArrayLike] | None = None,
        start_time: ArrayLike | None = None,
        end_time: ArrayLike | None = None,
        coarse_start_time: ArrayLike | None = None,
        coarse_end_time: ArrayLike | None = None,
        fine_start_times: ArrayLike | None = None,
        fine_end_times: ArrayLike | None = None,
        accepted_steps: ArrayLike | None = None,
        route_id: str = "unstructured-amr-interface-route",
        layout_id: str = "unstructured-amr-interface-layout",
        coarse_topology_id: str = "unstructured-amr-coarse-topology",
        fine_topology_id: str = "unstructured-amr-fine-topology",
        topology_id: str | None = None,
    ):
        # The original one-array constructor remains valid.  Supplying two
        # arrays makes the exact coarse/fine mismatch explicit.
        first = jnp.asarray(integrated_correction)
        second = None if fine_flux_integral is None else jnp.asarray(fine_flux_integral)
        if first.ndim < 1:
            raise ValueError("AMR flux-register correction must have a cell axis.")
        if coarse_flux is not None:
            if coarse_flux_integral is not None:
                raise ValueError("Specify only one coarse accepted flux array.")
            coarse_flux_integral = coarse_flux
        if fine_flux is not None:
            if second is not None:
                raise ValueError("Specify fine flux only once.")
            second = jnp.asarray(fine_flux)
        if coarse_flux_integral is not None:
            coarse = jnp.asarray(coarse_flux_integral)
            if second is None:
                raise ValueError("A coarse accepted flux requires a fine accepted flux.")
            if coarse.shape != second.shape:
                raise ValueError("Coarse and fine accepted flux integrals must match.")
            mismatch = second - coarse
        elif second is not None:
            # Positional ``(correction, fine)`` is intentionally interpreted as
            # an already computed correction only when the first array is the
            # same shape as the fine array; otherwise fail loudly.
            if first.shape != second.shape:
                raise ValueError("Flux-register arrays must have identical shapes.")
            coarse = jnp.zeros_like(first)
            mismatch = first
            second = second
        else:
            coarse = jnp.zeros_like(first)
            mismatch = first
            second = first
        if mismatch.shape != first.shape:
            raise ValueError("AMR flux-register correction has an inconsistent shape.")

        topology = (
            None if topology_id is None else _amr_identity(topology_id, "topology_id")
        )
        coarse_topology = _amr_identity(coarse_topology_id, "coarse_topology_id")
        fine_topology = _amr_identity(fine_topology_id, "fine_topology_id")
        if topology is not None:
            coarse_topology = topology
            fine_topology = topology
        route = _amr_identity(route_id, "route_id")
        layout = _amr_identity(layout_id, "layout_id")
        interval = _interval(coarse_interval, "coarse_interval")
        if interval is None:
            start = None if start_time is None else _scalar_time(start_time, "start_time")
            end = None if end_time is None else _scalar_time(end_time, "end_time")
            if start is None and end is None:
                interval = (0.0, 1.0)
            elif start is None or end is None:
                raise ValueError("start_time and end_time must be supplied together.")
            else:
                interval = (start, end)
        else:
            if start_time is not None and not np.isclose(
                _scalar_time(start_time, "start_time"),
                interval[0],
                rtol=0.0,
                atol=_time_tolerance(interval[0], interval[0]),
            ):
                raise ValueError("start_time disagrees with coarse_interval.")
            if end_time is not None and not np.isclose(
                _scalar_time(end_time, "end_time"),
                interval[1],
                rtol=0.0,
                atol=_time_tolerance(interval[1], interval[1]),
            ):
                raise ValueError("end_time disagrees with coarse_interval.")
        if coarse_start_time is not None and not np.isclose(
            _scalar_time(coarse_start_time, "coarse_start_time"),
            interval[0],
            rtol=0.0,
            atol=_time_tolerance(interval[0], interval[0]),
        ):
            raise ValueError("coarse_start_time must match the coarse interval start.")
        if coarse_end_time is not None and not np.isclose(
            _scalar_time(coarse_end_time, "coarse_end_time"),
            interval[1],
            rtol=0.0,
            atol=_time_tolerance(interval[1], interval[1]),
        ):
            raise ValueError("coarse_end_time must match the coarse interval end.")

        if fine_intervals is not None and (
            fine_start_times is not None or fine_end_times is not None
        ):
            raise ValueError("Specify fine_intervals or fine start/end arrays, not both.")
        if fine_intervals is not None:
            intervals = tuple(
                _required_interval(item, f"fine_intervals[{index}]")
                for index, item in enumerate(fine_intervals)
            )
        elif fine_start_times is not None or fine_end_times is not None:
            if fine_start_times is None or fine_end_times is None:
                raise ValueError(
                    "fine_start_times and fine_end_times must be supplied together."
                )
            starts = np.asarray(fine_start_times)
            ends = np.asarray(fine_end_times)
            if starts.ndim != 1 or ends.shape != starts.shape:
                raise ValueError(
                    "Fine interval starts and ends must be matching vectors."
                )
            intervals = tuple(
                _required_interval((left, right), f"fine_intervals[{index}]")
                for index, (left, right) in enumerate(zip(starts, ends, strict=True))
            )
        else:
            intervals = (interval,)
        if not intervals:
            raise ValueError(
                "fine_intervals must contain at least one accepted interval."
            )
        for index, (left, right) in enumerate(intervals):
            if index and left - intervals[index - 1][1] > _time_tolerance(
                left, intervals[index - 1][1]
            ):
                raise ValueError(
                    f"Fine accepted intervals contain a gap before result {index}."
                )
            if index and left - intervals[index - 1][1] < -_time_tolerance(
                left, intervals[index - 1][1]
            ):
                raise ValueError(
                    f"Fine accepted intervals overlap before result {index}."
                )
        if abs(intervals[0][0] - interval[0]) > _time_tolerance(
            intervals[0][0], interval[0]
        ):
            raise ValueError(
                "Fine accepted intervals must start at the coarse interval start."
            )
        if abs(intervals[-1][1] - interval[1]) > _time_tolerance(
            intervals[-1][1], interval[1]
        ):
            raise ValueError(
                "Fine accepted intervals must end at the coarse interval end."
            )
        if accepted_steps is None:
            steps = np.arange(len(intervals), dtype=np.int32)
        else:
            steps = np.asarray(accepted_steps)
            if steps.shape != (len(intervals),) or steps.dtype.kind not in "iu":
                raise ValueError(
                    "accepted_steps must contain one integer ID per fine interval."
                )
            if np.any(np.diff(steps) <= 0):
                raise ValueError("Fine accepted-step IDs must be strictly monotone.")

        self.integrated_correction = mismatch
        self.coarse_flux_integral = coarse
        self.fine_flux_integral = second
        self.flux_mismatch = mismatch
        self.start_time = jnp.asarray(interval[0])
        self.end_time = jnp.asarray(interval[1])
        self.fine_interval_starts = jnp.asarray([item[0] for item in intervals])
        self.fine_interval_ends = jnp.asarray([item[1] for item in intervals])
        self.fine_accepted_steps = jnp.asarray(steps)
        self.route_id = route
        self.layout_id = layout
        self.coarse_topology_id = coarse_topology
        self.fine_topology_id = fine_topology
        self.register_id = canonical_fingerprint(
            {
                "kind": "unstructured-amr-accepted-flux-register",
                "route_id": route,
                "layout_id": layout,
                "coarse_topology_id": coarse_topology,
                "fine_topology_id": fine_topology,
                "coarse_interval": list(interval),
                "fine_intervals": [list(item) for item in intervals],
                "accepted_steps": steps.tolist(),
            }
        )


class UnstructuredAMRHierarchyPlan(StrictModule, NonTrainableState):
    """Fixed two-level nested unstructured AMR on one device."""

    coarse: UnstructuredFiniteVolumeDiscretization
    fine: UnstructuredFiniteVolumeDiscretization
    prolongation: UnstructuredConservativeRemapPlan
    restriction: UnstructuredConservativeRemapPlan
    fine_parent_cells: Array
    coarse_cell_global_ids: Array
    fine_cell_global_ids: Array
    fine_parent_global_ids: Array
    parent_cell_global_ids: Array
    child_cell_global_ids: Array
    coarse_fine_interface_map: Array
    coarse_interface_route_ids: tuple[str, ...] = eqx.field(static=True)
    fine_interface_route_ids: tuple[str, ...] = eqx.field(static=True)
    maximum_refined_cells: int = eqx.field(static=True)
    refinement_ratio: int = eqx.field(static=True)
    coverage_evidence_id: str = eqx.field(static=True)
    interface_route_id: str = eqx.field(static=True)
    interface_layout_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse: UnstructuredFiniteVolumeDiscretization,
        fine: UnstructuredFiniteVolumeDiscretization,
        prolongation: UnstructuredConservativeRemapPlan,
        restriction: UnstructuredConservativeRemapPlan,
        /,
        *,
        maximum_refined_cells: int | None = None,
        refinement_ratio: int = 2,
        coarse_fine_interface_map: ArrayLike | None = None,
        interface_map: ArrayLike | None = None,
        coarse_interface_route_ids: Sequence[str] | None = None,
        fine_interface_route_ids: Sequence[str] | None = None,
    ):
        if not isinstance(
            coarse, UnstructuredFiniteVolumeDiscretization
        ) or not isinstance(fine, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("AMR levels must be unstructured FV discretizations.")
        if not isinstance(
            prolongation, UnstructuredConservativeRemapPlan
        ) or not isinstance(restriction, UnstructuredConservativeRemapPlan):
            raise TypeError("AMR transfers must be conservative remap plans.")
        if (
            prolongation.source_topology_id != coarse.topology_id
            or prolongation.source_geometry_id != coarse.geometry_id
            or prolongation.target_topology_id != fine.topology_id
            or prolongation.target_geometry_id != fine.geometry_id
        ):
            raise ValueError("AMR prolongation does not bind coarse to fine geometry.")
        if (
            restriction.source_topology_id != fine.topology_id
            or restriction.source_geometry_id != fine.geometry_id
            or restriction.target_topology_id != coarse.topology_id
            or restriction.target_geometry_id != coarse.geometry_id
        ):
            raise ValueError("AMR restriction does not bind fine to coarse geometry.")
        offsets = np.asarray(prolongation.target_offsets, dtype=np.int32)
        if np.any(np.diff(offsets) != 1):
            raise ValueError(
                "Nested AMR requires exactly one coarse parent per fine cell."
            )
        parent = np.asarray(prolongation.source_indices, dtype=np.int32)
        overlap = np.asarray(prolongation.intersection_measures)
        if not np.allclose(
            overlap, np.asarray(fine.cell_volumes), rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                "Nested AMR fine cells must lie completely inside one parent."
            )
        if not isinstance(refinement_ratio, (int, np.integer)) or isinstance(
            refinement_ratio, (bool, np.bool_)
        ):
            raise TypeError("refinement_ratio must be a positive integer.")
        ratio = int(refinement_ratio)
        if ratio <= 1:
            raise ValueError("refinement_ratio must be greater than one.")
        coarse_ids = np.asarray(coarse.cell_global_ids)
        fine_ids = np.asarray(fine.cell_global_ids)
        if np.any(coarse_ids < 0) or np.any(fine_ids < 0):
            raise ValueError("AMR stable cell IDs must be nonnegative.")
        if (
            np.unique(coarse_ids).size != coarse_ids.size
            or np.unique(fine_ids).size != fine_ids.size
        ):
            raise ValueError("AMR stable cell IDs must be unique within each level.")
        parent_global_ids = coarse_ids[parent]
        if coarse_fine_interface_map is not None and interface_map is not None:
            raise ValueError("Specify only one coarse/fine interface map.")
        interface_value = (
            coarse_fine_interface_map
            if coarse_fine_interface_map is not None
            else interface_map
        )
        if interface_value is None:
            interface = np.stack(
                (parent, np.arange(fine.cell_count, dtype=np.int32)), axis=1
            )
        else:
            interface = np.asarray(interface_value, dtype=np.int32)
            if interface.ndim != 2 or interface.shape[1] != 2:
                raise ValueError(
                    "coarse_fine_interface_map must have shape (route_count, 2)."
                )
            if (
                np.any(interface[:, 0] < 0)
                or np.any(interface[:, 0] >= coarse.cell_count)
                or np.any(interface[:, 1] < 0)
                or np.any(interface[:, 1] >= fine.cell_count)
            ):
                raise ValueError(
                    "coarse/fine interface map contains an out-of-range cell."
                )
        if (coarse_interface_route_ids is None) != (fine_interface_route_ids is None):
            raise ValueError(
                "Coarse and fine interface route IDs must be supplied together."
            )
        coarse_route_ids = (
            ()
            if coarse_interface_route_ids is None
            else tuple(
                _amr_identity(value, "coarse_interface_route_id")
                for value in coarse_interface_route_ids
            )
        )
        fine_route_ids = (
            ()
            if fine_interface_route_ids is None
            else tuple(
                _amr_identity(value, "fine_interface_route_id")
                for value in fine_interface_route_ids
            )
        )
        if len(coarse_route_ids) != len(fine_route_ids):
            raise ValueError("Coarse/fine interface route-ID counts must match.")
        capacity = (
            coarse.cell_count
            if maximum_refined_cells is None
            else int(maximum_refined_cells)
        )
        if capacity <= 0 or capacity > coarse.cell_count:
            raise ValueError("maximum_refined_cells must lie within coarse capacity.")
        coverage_evidence_id = canonical_fingerprint(
            {
                "kind": "unstructured-amr-coverage-evidence",
                "prolongation": prolongation.coverage_evidence_id,
                "restriction": restriction.coverage_evidence_id,
            }
        )
        interface_route_id = canonical_fingerprint(
            {
                "kind": "unstructured-amr-coarse-fine-route",
                "map": array_tree_fingerprint(interface),
            }
        )
        interface_layout_id = canonical_fingerprint(
            {
                "kind": "unstructured-amr-coarse-fine-layout",
                "coarse_ids": array_tree_fingerprint(coarse_ids),
                "fine_ids": array_tree_fingerprint(fine_ids),
            }
        )
        self.coarse = coarse
        self.fine = fine
        self.prolongation = prolongation
        self.restriction = restriction
        self.fine_parent_cells = jnp.asarray(parent)
        self.coarse_cell_global_ids = coarse.cell_global_ids
        self.fine_cell_global_ids = fine.cell_global_ids
        self.fine_parent_global_ids = jnp.asarray(parent_global_ids)
        self.parent_cell_global_ids = jnp.asarray(parent_global_ids)
        self.child_cell_global_ids = fine.cell_global_ids
        self.coarse_fine_interface_map = jnp.asarray(interface)
        self.coarse_interface_route_ids = coarse_route_ids
        self.fine_interface_route_ids = fine_route_ids
        self.maximum_refined_cells = capacity
        self.refinement_ratio = ratio
        self.coverage_evidence_id = coverage_evidence_id
        self.interface_route_id = interface_route_id
        self.interface_layout_id = interface_layout_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-level-unstructured-amr",
                "coarse": coarse.prepared_id,
                "fine": fine.prepared_id,
                "prolongation": prolongation.plan_id,
                "restriction": restriction.plan_id,
                "maximum_refined_cells": capacity,
                "refinement_ratio": ratio,
                "coarse_fine_interface_map": array_tree_fingerprint(interface),
                "coarse_interface_route_ids": list(coarse_route_ids),
                "fine_interface_route_ids": list(fine_route_ids),
                "coverage_evidence_id": coverage_evidence_id,
                "interface_route_id": interface_route_id,
                "interface_layout_id": interface_layout_id,
            }
        )

    @property
    def coarse_fine_interface_coarse_route_ids(self) -> tuple[str, ...]:
        return self.coarse_interface_route_ids

    @property
    def coarse_fine_interface_fine_route_ids(self) -> tuple[str, ...]:
        return self.fine_interface_route_ids

    @property
    def coarse_fine_interface_coarse_cells(self) -> Array:
        return self.coarse_fine_interface_map[:, 0]

    @property
    def coarse_fine_interface_fine_cells(self) -> Array:
        return self.coarse_fine_interface_map[:, 1]

    def select(
        self,
        indicator: ArrayLike,
        threshold: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ) -> UnstructuredAMRSelection:
        values = jnp.asarray(indicator)
        if values.shape != (self.coarse.cell_count,):
            raise ValueError("AMR indicator must contain one value per coarse cell.")
        active = _mask(active_mask, self.coarse.cell_count, "active_mask")
        threshold_ = jnp.asarray(threshold, dtype=values.dtype).reshape(())
        eligible = active & jnp.isfinite(values) & (values >= threshold_)
        scores = jnp.where(eligible, values, -jnp.inf)
        # ``lexsort`` gives a stable-ID tie break, independent of local mesh
        # ordering or device reduction order.
        order = jnp.lexsort((self.coarse_cell_global_ids, -scores))
        ranks = (
            jnp.empty_like(order)
            .at[order]
            .set(jnp.arange(self.coarse.cell_count, dtype=order.dtype))
        )
        refined = eligible & (ranks < self.maximum_refined_cells)
        fine_active = refined[self.fine_parent_cells]
        selected_count = jnp.sum(refined, dtype=jnp.int32)
        eligible_count = jnp.sum(eligible, dtype=jnp.int32)
        overflow_count = jnp.maximum(eligible_count - self.maximum_refined_cells, 0)
        overflow = eligible_count > self.maximum_refined_cells
        return UnstructuredAMRSelection(
            coarse_refined=refined,
            fine_active=fine_active,
            selected_count=selected_count,
            eligible_count=eligible_count,
            capacity_overflow=overflow,
            overflow_count=overflow_count,
            overflow_status=jnp.where(
                overflow, jnp.asarray(1, dtype=jnp.int8), jnp.asarray(0, dtype=jnp.int8)
            ),
        )

    def prolong(
        self,
        coarse_cell_averages: ArrayLike,
        /,
        *,
        coarse_active_mask: ArrayLike | None = None,
        fine_active_mask: ArrayLike | None = None,
        bounded: bool = False,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> Array:
        if bounded:
            return self.prolongation.apply_bounded(
                coarse_cell_averages,
                lower=lower,
                upper=upper,
                source_active_mask=coarse_active_mask,
                target_active_mask=fine_active_mask,
            )
        return self.prolongation.apply(
            coarse_cell_averages,
            source_active_mask=coarse_active_mask,
            target_active_mask=fine_active_mask,
        )

    def restrict(
        self,
        fine_cell_averages: ArrayLike,
        /,
        *,
        fine_active_mask: ArrayLike | None = None,
        coarse_active_mask: ArrayLike | None = None,
        bounded: bool = False,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> Array:
        if bounded:
            return self.restriction.apply_bounded(
                fine_cell_averages,
                lower=lower,
                upper=upper,
                source_active_mask=fine_active_mask,
                target_active_mask=coarse_active_mask,
            )
        return self.restriction.apply(
            fine_cell_averages,
            source_active_mask=fine_active_mask,
            target_active_mask=coarse_active_mask,
        )

    def prolong_content(
        self,
        coarse_content: ArrayLike,
        /,
        *,
        coarse_active_mask: ArrayLike | None = None,
        fine_active_mask: ArrayLike | None = None,
        coarse_volumes: ArrayLike | None = None,
    ) -> Array:
        return self.prolongation.apply_content(
            coarse_content,
            source_active_mask=coarse_active_mask,
            target_active_mask=fine_active_mask,
            source_volumes=coarse_volumes,
        )

    def restrict_content(
        self,
        fine_content: ArrayLike,
        /,
        *,
        fine_active_mask: ArrayLike | None = None,
        coarse_active_mask: ArrayLike | None = None,
        fine_volumes: ArrayLike | None = None,
    ) -> Array:
        return self.restriction.apply_content(
            fine_content,
            source_active_mask=fine_active_mask,
            target_active_mask=coarse_active_mask,
            source_volumes=fine_volumes,
        )

    def prolong_fluid_volume(self, coarse_fluid_volumes: ArrayLike, /, **kwargs) -> Array:
        return self.prolong_content(coarse_fluid_volumes, **kwargs)

    def restrict_fluid_volume(self, fine_fluid_volumes: ArrayLike, /, **kwargs) -> Array:
        return self.restrict_content(fine_fluid_volumes, **kwargs)

    def synchronize(
        self,
        coarse_cell_averages: ArrayLike,
        fine_cell_averages: ArrayLike,
        selection: UnstructuredAMRSelection,
        /,
        *,
        coarse_active_mask: ArrayLike | None = None,
        fine_active_mask: ArrayLike | None = None,
        bounded: bool = False,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> Array:
        if not isinstance(selection, UnstructuredAMRSelection):
            raise TypeError("selection must be UnstructuredAMRSelection.")
        coarse = jnp.asarray(coarse_cell_averages)
        restricted = self.restrict(
            fine_cell_averages,
            fine_active_mask=fine_active_mask,
            coarse_active_mask=coarse_active_mask,
            bounded=bounded,
            lower=lower,
            upper=upper,
        )
        mask = selection.coarse_refined.reshape(
            selection.coarse_refined.shape + (1,) * (coarse.ndim - 1)
        )
        return jnp.where(mask, restricted, coarse)

    def reflux(
        self,
        coarse_cell_averages: ArrayLike,
        register: UnstructuredAMRFluxRegister,
        /,
        *,
        coarse_active_mask: ArrayLike | None = None,
    ) -> Array:
        if not isinstance(register, UnstructuredAMRFluxRegister):
            raise TypeError("register must be UnstructuredAMRFluxRegister.")
        if register.route_id not in (
            self.interface_route_id,
            "unstructured-amr-interface-route",
        ):
            raise ValueError(
                "AMR flux register route does not match the hierarchy interface map."
            )
        if register.layout_id not in (
            self.interface_layout_id,
            "unstructured-amr-interface-layout",
        ):
            raise ValueError(
                "AMR flux register layout does not match the hierarchy interface map."
            )
        coarse = jnp.asarray(coarse_cell_averages)
        correction = register.integrated_correction.astype(coarse.dtype)
        if correction.shape != coarse.shape:
            raise ValueError("AMR flux-register correction must match coarse state.")
        if coarse_active_mask is not None:
            active = _mask(
                coarse_active_mask, self.coarse.cell_count, "coarse_active_mask"
            )
            correction = jnp.where(
                active.reshape(active.shape + (1,) * (coarse.ndim - 1)), correction, 0.0
            )
        volumes = self.coarse.cell_volumes.astype(coarse.dtype).reshape(
            (-1,) + (1,) * (coarse.ndim - 1)
        )
        return coarse + correction / volumes

    def composite_integral(
        self,
        coarse_cell_averages: ArrayLike,
        fine_cell_averages: ArrayLike,
        selection: UnstructuredAMRSelection,
        /,
    ) -> Array:
        coarse = jnp.asarray(coarse_cell_averages)
        fine = jnp.asarray(fine_cell_averages)
        coarse_mask = ~selection.coarse_refined
        fine_mask = selection.fine_active
        coarse_weight = self.coarse.cell_volumes.astype(coarse.dtype).reshape(
            (-1,) + (1,) * (coarse.ndim - 1)
        )
        fine_weight = self.fine.cell_volumes.astype(fine.dtype).reshape(
            (-1,) + (1,) * (fine.ndim - 1)
        )
        coarse_mask_ = coarse_mask.reshape(coarse_mask.shape + (1,) * (coarse.ndim - 1))
        fine_mask_ = fine_mask.reshape(fine_mask.shape + (1,) * (fine.ndim - 1))
        return jnp.sum(
            jnp.where(coarse_mask_, coarse_weight * coarse, 0.0), axis=0
        ) + jnp.sum(jnp.where(fine_mask_, fine_weight * fine, 0.0), axis=0)


__all__ = [
    "UnstructuredAMRFluxRegister",
    "UnstructuredAMRHierarchyPlan",
    "UnstructuredAMRSelection",
]
