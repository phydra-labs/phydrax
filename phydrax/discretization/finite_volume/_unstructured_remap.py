#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._numerics._compensated import compensated_sum_chunks
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    adjoint,
    FunctionLinearOperator,
    SmallLinearSolvePlan,
    SmallLinearSolveResult,
    solve_small_linear,
    transpose,
)
from ...sparse import (
    EdgeRelation,
    gather_routes,
    route_reduce,
    RowRelation,
    SparseLinearMap,
)
from .._cell_mesh import CellMesh
from .._spaces import DiscreteFieldSpace
from .._topology_epoch import TopologyEpoch, TopologyEpochTransition
from .._transfer import FieldTransfer, TransferGeometryBinding, TransferProperties
from ._unstructured import UnstructuredFiniteVolumeDiscretization


if TYPE_CHECKING:
    from ...geometry._supermesh import PreparedCommonRefinement


class UnstructuredRemapReport(StrictModule):
    """Certified source-content coverage and physical endpoint diagnostics.

    For ``source-physical-chart-content``, target measure defects report bounded
    physical area deformation, not a UV coverage gap. Complete target coverage
    is carried by the actual native chart certificate retained on the plan.
    """

    maximum_target_coverage_defect: Array
    maximum_source_coverage_defect: Array
    uncovered_target_measure: Array
    uncovered_source_measure: Array
    donor_excess_measure: Array
    source_measure: Array
    target_measure: Array
    coverage_complete: Array
    tolerance: Array
    maximum_target_coverage_error_bound: Array
    maximum_source_coverage_error_bound: Array
    total_target_coverage_error_bound: Array
    total_source_coverage_error_bound: Array
    measure_semantics: str = eqx.field(static=True)


def _active_mask(value: ArrayLike | None, count: int, name: str, /) -> Array:
    if value is None:
        return jnp.ones((count,), dtype=jnp.bool_)
    array = jnp.asarray(value)
    if array.shape != (count,) or array.dtype != jnp.dtype(jnp.bool_):
        raise ValueError(f"{name} must be a boolean array with one entry per cell.")
    return array


def _volume_array(
    value: ArrayLike | None, fallback: Array, count: int, name: str, /
) -> Array:
    array = fallback if value is None else jnp.asarray(value)
    if array.shape != (count,):
        raise ValueError(f"{name} must contain one volume per cell.")
    return array


def _mask_values(values: Array, mask: Array, name: str, /) -> Array:
    expanded = mask.reshape(mask.shape + (1,) * (values.ndim - 1))
    values = eqx.error_if(
        values,
        jnp.any((~expanded) & (values != 0.0)),
        f"{name} must be exactly zero on inactive cells.",
    )
    return values


def _coverage_ledger(
    target_routes: np.ndarray,
    source_indices: np.ndarray,
    measures: np.ndarray,
    source_volumes: np.ndarray,
    target_volumes: np.ndarray,
    tolerance: float,
    /,
    *,
    intersection_error_bounds: np.ndarray | None = None,
    source_error_bounds: np.ndarray | None = None,
    target_error_bounds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Signed covered-minus-cell measures and their admissible magnitudes.

    This is the single coverage certificate of a remap ledger: a plan requiring
    complete coverage accepts it exactly when every defect magnitude is at most
    its limit.
    """
    target_defect = (
        np.bincount(target_routes, weights=measures, minlength=target_volumes.size)
        - target_volumes
    )
    source_defect = (
        np.bincount(source_indices, weights=measures, minlength=source_volumes.size)
        - source_volumes
    )
    target_scale = np.maximum(target_volumes, np.max(target_volumes) * 1e-14)
    source_scale = np.maximum(source_volumes, np.max(source_volumes) * 1e-14)
    errors = (
        np.zeros_like(measures)
        if intersection_error_bounds is None
        else intersection_error_bounds
    )
    target_error = np.bincount(
        target_routes, weights=errors, minlength=target_volumes.size
    )
    source_error = np.bincount(
        source_indices, weights=errors, minlength=source_volumes.size
    )
    if target_error_bounds is not None:
        target_error += target_error_bounds
    if source_error_bounds is not None:
        source_error += source_error_bounds
    return (
        target_defect,
        source_defect,
        tolerance * target_scale + target_error,
        tolerance * source_scale + source_error,
    )


class UnstructuredConservativeRemapPlan(StrictModule, NonTrainableState):
    """Canonical CSR conservative content transport between immutable epochs.

    ``physical-cell-overlap`` uses geometric intersections or exact nested cells.
    ``source-physical-chart-content`` uses old physical density integrals over
    actual native UV overlap pieces and divides by actual new physical cell
    areas. It conserves content, not constants under area-changing deformation.
    ``apply`` consumes cell averages; ``apply_content`` consumes extensive content.
    """

    source_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    source_cell_global_ids: Array
    target_cell_global_ids: Array
    target_offsets: Array
    source_indices: Array
    target_routes: Array
    intersection_measures: Array
    intersection_error_bounds: Array
    source_volumes: Array
    target_volumes: Array
    source_volume_error_bounds: Array
    target_volume_error_bounds: Array
    surface_chart_deformation: Any | None
    surface_chart_contents: Any | None
    measure_semantics: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    require_complete: bool = eqx.field(static=True)
    report: UnstructuredRemapReport
    coverage_evidence_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: UnstructuredFiniteVolumeDiscretization,
        target: UnstructuredFiniteVolumeDiscretization,
        target_offsets: ArrayLike,
        source_indices: ArrayLike,
        intersection_measures: ArrayLike,
        /,
        *,
        method: str,
        provenance: str,
        tolerance: float = 1e-10,
        require_complete: bool = True,
        route_id: str | None = None,
        layout_id: str | None = None,
        intersection_error_bounds: ArrayLike | None = None,
        surface_chart_deformation: Any | None = None,
        surface_chart_contents: Any | None = None,
    ) -> None:
        if not isinstance(
            source, UnstructuredFiniteVolumeDiscretization
        ) or not isinstance(target, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("Remap source and target must be unstructured FV geometry.")
        method_ = str(method)
        provenance_ = str(provenance)
        if not method_ or not provenance_:
            raise ValueError("Remap method and provenance must be non-empty.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Remap tolerance must be positive and finite.")
        offsets = np.asarray(target_offsets, dtype=np.int32)
        indices = np.asarray(source_indices, dtype=np.int32)
        measures = np.asarray(intersection_measures, dtype=np.float64)
        if offsets.shape != (target.cell_count + 1,):
            raise ValueError(
                "target_offsets must contain one CSR offset per target cell."
            )
        if (
            offsets[0] != 0
            or np.any(np.diff(offsets) < 0)
            or offsets[-1] != indices.size
            or measures.shape != indices.shape
        ):
            raise ValueError("Remap CSR routes are inconsistent.")
        if np.any(indices < 0) or np.any(indices >= source.cell_count):
            raise ValueError("Remap source_indices are out of range.")
        if np.any(~np.isfinite(measures)) or np.any(measures <= 0.0):
            raise ValueError("Remap intersection measures must be positive and finite.")
        source_ids = np.asarray(source.cell_global_ids)
        target_ids = np.asarray(target.cell_global_ids)
        if source_ids.shape != (source.cell_count,) or target_ids.shape != (
            target.cell_count,
        ):
            raise ValueError("Remap cell global IDs must contain one ID per cell.")
        if source_ids.dtype.kind not in "iu" or target_ids.dtype.kind not in "iu":
            raise TypeError("Remap cell global IDs must be integer arrays.")
        if np.any(source_ids < 0) or np.any(target_ids < 0):
            raise ValueError("Remap cell global IDs must be nonnegative.")
        if (
            np.unique(source_ids).size != source_ids.size
            or np.unique(target_ids).size != target_ids.size
        ):
            raise ValueError("Remap cell global IDs must be unique within each mesh.")
        target_routes = np.repeat(
            np.arange(target.cell_count, dtype=np.int32), np.diff(offsets)
        )
        source_volumes = np.asarray(source.cell_volumes)
        target_volumes = np.asarray(target.cell_volumes)
        errors = (
            np.zeros_like(measures)
            if intersection_error_bounds is None
            else np.asarray(intersection_error_bounds, dtype=np.float64)
        )
        if (
            errors.shape != measures.shape
            or np.any(~np.isfinite(errors))
            or np.any(errors < 0)
            or np.any(errors >= measures)
        ):
            raise ValueError(
                "Remap measure error bounds must certify strictly positive overlaps."
            )
        target_defect, source_defect, target_limit, source_limit = _coverage_ledger(
            target_routes,
            indices,
            measures,
            source_volumes,
            target_volumes,
            tolerance_,
            intersection_error_bounds=errors,
            source_error_bounds=np.asarray(source.cell_volume_error_bounds),
            target_error_bounds=np.asarray(target.cell_volume_error_bounds),
        )
        if (surface_chart_deformation is None) != (surface_chart_contents is None):
            raise ValueError(
                "Chart deformation and its actual physical contents are required together."
            )
        if surface_chart_deformation is not None:
            from .._sphere_chart_deformation import PreparedSphereChartDeformation
            from .._surface_chart_deformation import PreparedSurfaceChartDeformation
            from ..fem._surface_chart_transfer import (
                PreparedSurfaceChartFiniteVolumeContents,
            )

            if not isinstance(
                surface_chart_deformation,
                (PreparedSurfaceChartDeformation, PreparedSphereChartDeformation),
            ) or not isinstance(
                surface_chart_contents, PreparedSurfaceChartFiniteVolumeContents
            ):
                raise TypeError(
                    "Chart remap requires the actual prepared deformation and physical contents."
                )
            if source.cell_geometry is None or target.cell_geometry is None:
                raise ValueError("Chart remap needs the actual prepared coordinate maps.")
            surface_chart_deformation.require_bound(
                source.mesh, source.cell_geometry, target.mesh, target.cell_geometry
            )
            if (
                surface_chart_contents.deformation_id
                != surface_chart_deformation.deformation_id
            ):
                raise ValueError(
                    "Chart physical contents belong to a different deformation."
                )
            if not np.array_equal(
                surface_chart_contents.source_volumes, source_volumes
            ) or not np.array_equal(
                surface_chart_contents.target_volumes, target_volumes
            ):
                raise ValueError(
                    "Chart contents must use authoritative prepared FV physical areas."
                )
            order = np.lexsort(
                (
                    np.asarray(surface_chart_contents.source_rows),
                    np.asarray(surface_chart_contents.target_rows),
                )
            )
            if not (
                np.array_equal(
                    indices, np.asarray(surface_chart_contents.source_rows)[order]
                )
                and np.array_equal(
                    target_routes, np.asarray(surface_chart_contents.target_rows)[order]
                )
                and np.array_equal(
                    measures, np.asarray(surface_chart_contents.source_contents)[order]
                )
                and np.array_equal(
                    errors,
                    np.asarray(surface_chart_contents.source_content_errors)[order],
                )
            ):
                raise ValueError(
                    "Chart CSR does not realize its physical-content certificate."
                )
            source_limit = np.asarray(
                surface_chart_contents.source_content_defect_bounds, dtype=np.float64
            )
            if (
                source_limit.shape != source_volumes.shape
                or np.any(~np.isfinite(source_limit))
                or np.any(source_limit < 0)
            ):
                raise ValueError(
                    "Chart source-content defect bounds must be finite per-cell absolute bounds."
                )
        source_complete = np.all(np.abs(source_defect) <= source_limit)
        # Target coverage is a native UV tiling for chart deformation, not the
        # false assertion that old physical piece areas equal new physical areas.
        complete = source_complete and (
            surface_chart_deformation is not None
            or np.all(np.abs(target_defect) <= target_limit)
        )
        if require_complete and not complete:
            raise ValueError(
                "Conservative remap does not completely cover source and target."
            )
        target_errors = np.bincount(
            target_routes, weights=errors, minlength=target_volumes.size
        ) + np.asarray(target.cell_volume_error_bounds)
        source_errors = np.bincount(
            indices, weights=errors, minlength=source_volumes.size
        ) + np.asarray(source.cell_volume_error_bounds)
        if surface_chart_deformation is not None:
            source_errors = np.maximum(source_errors, source_limit)
        report = UnstructuredRemapReport(
            maximum_target_coverage_defect=jnp.asarray(np.max(np.abs(target_defect))),
            maximum_source_coverage_defect=jnp.asarray(np.max(np.abs(source_defect))),
            uncovered_target_measure=jnp.asarray(np.sum(np.maximum(-target_defect, 0.0))),
            uncovered_source_measure=jnp.asarray(np.sum(np.maximum(-source_defect, 0.0))),
            donor_excess_measure=jnp.asarray(np.sum(np.maximum(source_defect, 0.0))),
            source_measure=jnp.asarray(np.sum(source_volumes)),
            target_measure=jnp.asarray(np.sum(target_volumes)),
            coverage_complete=jnp.asarray(complete),
            tolerance=jnp.asarray(tolerance_),
            maximum_target_coverage_error_bound=jnp.asarray(np.max(target_errors)),
            maximum_source_coverage_error_bound=jnp.asarray(np.max(source_errors)),
            total_target_coverage_error_bound=jnp.asarray(np.sum(target_errors)),
            total_source_coverage_error_bound=jnp.asarray(np.sum(source_errors)),
            measure_semantics="source-physical-chart-content"
            if surface_chart_deformation is not None
            else "physical-cell-overlap",
        )
        route_ = (
            str(route_id)
            if route_id is not None
            else canonical_fingerprint(
                {
                    "kind": "unstructured-remap-route",
                    "target_routes": array_tree_fingerprint(target_routes),
                    "source_indices": array_tree_fingerprint(indices),
                }
            )
        )
        layout_ = (
            str(layout_id)
            if layout_id is not None
            else canonical_fingerprint(
                {
                    "kind": "unstructured-remap-layout",
                    "target_offsets": array_tree_fingerprint(offsets),
                    "component": "cell-content",
                }
            )
        )
        if not route_ or not layout_:
            raise ValueError("Remap route_id and layout_id must be non-empty.")
        coverage_id = canonical_fingerprint(
            {
                "kind": "unstructured-remap-coverage-evidence",
                "source_topology": source.topology_id,
                "target_topology": target.topology_id,
                "maximum_target_defect": float(np.max(np.abs(target_defect))),
                "maximum_source_defect": float(np.max(np.abs(source_defect))),
                "require_complete": bool(require_complete),
                "tolerance": tolerance_,
            }
        )
        self.source_topology_id = source.topology_id
        self.source_geometry_id = source.geometry_id
        self.target_topology_id = target.topology_id
        self.target_geometry_id = target.geometry_id
        self.source_cell_global_ids = source.cell_global_ids
        self.target_cell_global_ids = target.cell_global_ids
        self.target_offsets = jnp.asarray(offsets)
        self.source_indices = jnp.asarray(indices)
        self.target_routes = jnp.asarray(target_routes)
        self.intersection_measures = jnp.asarray(measures)
        self.intersection_error_bounds = jnp.asarray(errors)
        self.source_volumes = source.cell_volumes
        self.target_volumes = target.cell_volumes
        self.source_volume_error_bounds = source.cell_volume_error_bounds
        self.target_volume_error_bounds = target.cell_volume_error_bounds
        self.surface_chart_deformation = surface_chart_deformation
        self.surface_chart_contents = surface_chart_contents
        self.measure_semantics = (
            "source-physical-chart-content"
            if surface_chart_deformation is not None
            else "physical-cell-overlap"
        )
        self.method = method_
        self.provenance = provenance_
        self.require_complete = bool(require_complete)
        self.report = report
        self.coverage_evidence_id = coverage_id
        self.route_id = route_
        self.layout_id = layout_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-conservative-remap",
                "source_topology": source.topology_id,
                "source_geometry": source.geometry_id,
                "target_topology": target.topology_id,
                "target_geometry": target.geometry_id,
                "surface_chart_deformation": None
                if surface_chart_deformation is None
                else surface_chart_deformation.deformation_id,
                "surface_chart_contents": None
                if surface_chart_contents is None
                else surface_chart_contents.contents_id,
                "source_cell_global_ids": array_tree_fingerprint(source.cell_global_ids),
                "target_cell_global_ids": array_tree_fingerprint(target.cell_global_ids),
                "target_offsets": array_tree_fingerprint(offsets),
                "source_indices": array_tree_fingerprint(indices),
                "intersection_measures": array_tree_fingerprint(measures),
                "intersection_error_bounds": array_tree_fingerprint(errors),
                "source_volume_error_bounds": array_tree_fingerprint(
                    source.cell_volume_error_bounds
                ),
                "target_volume_error_bounds": array_tree_fingerprint(
                    target.cell_volume_error_bounds
                ),
                "method": method_,
                "provenance": provenance_,
                "require_complete": bool(require_complete),
                "coverage_evidence_id": coverage_id,
                "route_id": route_,
                "layout_id": layout_,
            }
        )

    def _validate_values(self, values: ArrayLike, name: str, /) -> Array:
        value = jnp.asarray(values)
        if value.ndim == 0 or value.shape[0] != self.source_volumes.size:
            raise ValueError(f"{name} must begin with source cell count.")
        return value

    def apply(
        self,
        source_cell_averages: ArrayLike,
        /,
        *,
        source_active_mask: ArrayLike | None = None,
        target_active_mask: ArrayLike | None = None,
        target_volumes: ArrayLike | None = None,
    ) -> Array:
        """Transfer cell averages using volume-weighted common refinement."""
        value = self._validate_values(source_cell_averages, "Remap values")
        source_active = _active_mask(
            source_active_mask, self.source_volumes.size, "source_active_mask"
        )
        target_active = _active_mask(
            target_active_mask, self.target_volumes.size, "target_active_mask"
        )
        value = _mask_values(value, source_active, "Remap values")
        trailing = (1,) * (value.ndim - 1)
        denominator = _volume_array(
            target_volumes,
            self.target_volumes,
            self.target_volumes.size,
            "target_volumes",
        )
        denominator = eqx.error_if(
            denominator,
            jnp.any(target_active & (~jnp.isfinite(denominator) | (denominator <= 0.0))),
            "Active target remap volumes must be positive and finite.",
        )
        denominator = jnp.where(
            target_active,
            denominator,
            jnp.ones_like(denominator),
        )
        # Each route carries its normalized overlap fraction m_e / V_t, so a
        # route covering its whole target cell has weight exactly one and the
        # identity common refinement reproduces the averages bit for bit.
        weights = (
            self.intersection_measures.astype(value.dtype)
            / denominator.astype(value.dtype)[self.target_routes]
        )
        weighted = value[self.source_indices] * weights.reshape((-1,) + trailing)
        target = jnp.zeros(
            (self.target_volumes.size,) + value.shape[1:], dtype=value.dtype
        )
        target = target.at[self.target_routes].add(weighted)
        return jnp.where(
            target_active.reshape(target_active.shape + trailing),
            target,
            jnp.zeros((), dtype=target.dtype),
        )

    def apply_content(
        self,
        source_content: ArrayLike,
        /,
        *,
        source_volumes: ArrayLike | None = None,
        source_active_mask: ArrayLike | None = None,
        target_active_mask: ArrayLike | None = None,
    ) -> Array:
        """Transfer extensive content without converting it to a target average.

        Each source content is distributed in proportion to its covered geometric
        measure.  Thus a complete map preserves the global extensive integral,
        including when source and target cell volumes differ.
        """
        content = self._validate_values(source_content, "Remap content")
        source_active = _active_mask(
            source_active_mask, self.source_volumes.size, "source_active_mask"
        )
        target_active = _active_mask(
            target_active_mask, self.target_volumes.size, "target_active_mask"
        )
        content = _mask_values(content, source_active, "Remap content")
        source_measure = _volume_array(
            source_volumes,
            self.source_volumes,
            self.source_volumes.size,
            "source_volumes",
        )
        source_measure = eqx.error_if(
            source_measure,
            jnp.any(~jnp.isfinite(source_measure) | (source_measure <= 0.0)),
            "Source remap volumes must be positive and finite.",
        )
        trailing = (1,) * (content.ndim - 1)
        density = content / source_measure.astype(content.dtype).reshape((-1,) + trailing)
        weighted = density[self.source_indices] * self.intersection_measures.astype(
            content.dtype
        ).reshape((-1,) + trailing)
        target = jnp.zeros(
            (self.target_volumes.size,) + content.shape[1:], dtype=content.dtype
        )
        target = target.at[self.target_routes].add(weighted)
        return jnp.where(
            target_active.reshape(target_active.shape + trailing),
            target,
            jnp.zeros((), dtype=target.dtype),
        )

    def apply_extensive(self, source_content: ArrayLike, /, **kwargs: Any) -> Array:
        """Explicit extensive-transfer entry point used by AMR callers."""
        return self.apply_content(source_content, **kwargs)

    def apply_bounded(
        self,
        source_cell_averages: ArrayLike,
        /,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        source_active_mask: ArrayLike | None = None,
        target_active_mask: ArrayLike | None = None,
    ) -> Array:
        """Transfer a bounded scalar, failing for invalid source fractions."""
        if self.surface_chart_deformation is not None:
            raise ValueError(
                "Conservative chart density transfer does not preserve density bounds under area change."
            )
        lower_ = float(lower)
        upper_ = float(upper)
        if not np.isfinite(lower_) or not np.isfinite(upper_) or lower_ > upper_:
            raise ValueError("Bounded remap requires finite lower <= upper bounds.")
        source = self._validate_values(source_cell_averages, "Remap values")
        source = eqx.error_if(
            source,
            jnp.any(~jnp.isfinite(source) | (source < lower_) | (source > upper_)),
            "Bounded remap source values violate the supplied bounds.",
        )
        transferred = self.apply(
            source,
            source_active_mask=source_active_mask,
            target_active_mask=target_active_mask,
        )
        # Positive overlap weights make this a convex transfer.  Clipping only
        # removes roundoff excursions; the explicit check catches a genuine
        # violation instead of silently repairing nonphysical input.
        repaired = jnp.clip(transferred, lower_, upper_)
        return eqx.error_if(
            repaired,
            jnp.any(~jnp.isfinite(repaired) | (repaired < lower_) | (repaired > upper_)),
            "Bounded remap could not produce values in the requested interval.",
        )

    def apply_fixed_combinatorics(
        self,
        source_cell_averages: ArrayLike,
        intersection_measures: ArrayLike,
        source_volumes: ArrayLike,
        target_volumes: ArrayLike,
        /,
    ) -> Array:
        """Differentiable remap for one frozen common-refinement route graph."""
        source = self._validate_values(
            source_cell_averages, "Fixed-combinatorics remap values"
        )
        measures = jnp.asarray(intersection_measures, dtype=source.dtype)
        source_measure = jnp.asarray(source_volumes, dtype=source.dtype)
        target_measure = jnp.asarray(target_volumes, dtype=source.dtype)
        if measures.shape != self.intersection_measures.shape:
            raise ValueError(
                "Dynamic intersection measures must preserve remap capacity."
            )
        if source_measure.shape != self.source_volumes.shape:
            raise ValueError("Dynamic source volumes must preserve source capacity.")
        if target_measure.shape != self.target_volumes.shape:
            raise ValueError("Dynamic target volumes must preserve target capacity.")
        measures = eqx.error_if(
            measures,
            jnp.any(~jnp.isfinite(measures) | (measures <= 0.0)),
            "Fixed-combinatorics intersections must remain positive and finite.",
        )
        source_measure = eqx.error_if(
            source_measure,
            jnp.any(~jnp.isfinite(source_measure) | (source_measure <= 0.0)),
            "Fixed-combinatorics source volumes must remain positive and finite.",
        )
        target_measure = eqx.error_if(
            target_measure,
            jnp.any(~jnp.isfinite(target_measure) | (target_measure <= 0.0)),
            "Fixed-combinatorics target volumes must remain positive and finite.",
        )
        target_coverage = (
            jnp.zeros_like(target_measure).at[self.target_routes].add(measures)
        )
        source_coverage = (
            jnp.zeros_like(source_measure).at[self.source_indices].add(measures)
        )
        tolerance = self.report.tolerance.astype(source.dtype)
        coverage_valid = jnp.all(
            jnp.abs(target_coverage - target_measure)
            <= tolerance * jnp.maximum(target_measure, 1.0)
        ) & jnp.all(
            jnp.abs(source_coverage - source_measure)
            <= tolerance * jnp.maximum(source_measure, 1.0)
        )
        source = eqx.error_if(
            source,
            ~coverage_valid,
            "Fixed-combinatorics remap lost complete geometric coverage.",
        )
        trailing = (1,) * (source.ndim - 1)
        weighted = source[self.source_indices] * measures.reshape((-1,) + trailing)
        target = (
            jnp.zeros((target_measure.size,) + source.shape[1:], dtype=source.dtype)
            .at[self.target_routes]
            .add(weighted)
        )
        return target / target_measure.reshape((-1,) + trailing)

    def conservation_defect(
        self, source_cell_averages: ArrayLike, target_cell_averages: ArrayLike, /
    ) -> Array:
        source = jnp.asarray(source_cell_averages)
        target = jnp.asarray(target_cell_averages)
        if source.ndim == 0 or source.shape[0] != self.source_volumes.size:
            raise ValueError("source_cell_averages must begin with source cell count.")
        if target.ndim == 0 or target.shape[0] != self.target_volumes.size:
            raise ValueError("target_cell_averages must begin with target cell count.")
        source_terms = (
            self.source_volumes.astype(source.dtype).reshape(
                (-1,) + (1,) * (source.ndim - 1)
            )
            * source
        )
        target_terms = (
            self.target_volumes.astype(target.dtype).reshape(
                (-1,) + (1,) * (target.ndim - 1)
            )
            * target
        )
        return compensated_sum_chunks(
            (target_terms, -source_terms),
            output_ndim=source.ndim - 1,
        )

    def conservation_defect_content(
        self, source_content: ArrayLike, target_content: ArrayLike, /
    ) -> Array:
        source = jnp.asarray(source_content)
        target = jnp.asarray(target_content)
        if source.ndim == 0 or source.shape[0] != self.source_volumes.size:
            raise ValueError("source_content must begin with source cell count.")
        if target.ndim == 0 or target.shape[0] != self.target_volumes.size:
            raise ValueError("target_content must begin with target cell count.")
        return compensated_sum_chunks(
            (target, -source),
            output_ndim=source.ndim - 1,
        )

    def _transpose_averages(self, target_cotangent: Array, /) -> Array:
        """Algebraic transpose of `apply` for fully active cells."""
        value = jnp.asarray(target_cotangent)
        trailing = (1,) * (value.ndim - 1)
        weights = (
            self.intersection_measures.astype(value.dtype)
            / self.target_volumes.astype(value.dtype)[self.target_routes]
        )
        weighted = value[self.target_routes] * weights.reshape((-1,) + trailing)
        source = jnp.zeros(
            (self.source_volumes.size,) + value.shape[1:], dtype=value.dtype
        )
        return source.at[self.source_indices].add(weighted)

    def epoch_transition(
        self,
        source_field: DiscreteFieldSpace,
        target_field: DiscreteFieldSpace,
        source_epoch: TopologyEpoch,
        target_epoch: TopologyEpoch,
        /,
    ) -> TopologyEpochTransition:
        """Bind this complete cell remap as an explicit topology-epoch transition.

        The epochs must realize both actual topology and geometry identities.
        Positivity and conservative content have the certified source-piece
        bounds; constant preservation is claimed only for geometric-overlap
        routes, not an area-changing chart deformation. Transpose is exact CSR
        algebra and Hilbert adjoint uses the actual FV physical-area pairings.
        """

        if not isinstance(source_epoch, TopologyEpoch) or not isinstance(
            target_epoch, TopologyEpoch
        ):
            raise TypeError("Epoch transitions need TopologyEpoch endpoints.")
        if (
            source_epoch.topology_id != self.source_topology_id
            or target_epoch.topology_id != self.target_topology_id
            or source_epoch.geometry_id != self.source_geometry_id
            or target_epoch.geometry_id != self.target_geometry_id
        ):
            raise ValueError(
                "Topology epochs do not realize this remap's topologies and geometry."
            )
        if not self.require_complete:
            raise ValueError(
                "Only a complete-coverage remap is conservative enough for a "
                "topology transition."
            )
        primal = FunctionLinearOperator(
            lambda values: self.apply(values),
            source=source_field.vector_space,
            target=target_field.vector_space,
            transpose_action=self._transpose_averages,
            operator_id=f"{self.plan_id}:cell-averages",
        )
        exact_restriction = (
            self.method == "mapped-nested" and self.surface_chart_deformation is None
        )
        geometry_defect = 0.0
        if not exact_restriction:
            covered = np.bincount(
                np.asarray(self.target_routes),
                weights=np.asarray(self.intersection_measures),
                minlength=self.target_volumes.size,
            )
            uncertainty = np.bincount(
                np.asarray(self.target_routes),
                weights=np.asarray(self.intersection_error_bounds),
                minlength=self.target_volumes.size,
            ) + np.asarray(self.target_volume_error_bounds)
            lower_measure = np.asarray(self.target_volumes) - np.asarray(
                self.target_volume_error_bounds
            )
            geometry_defect = float(
                np.nextafter(
                    np.max(
                        (np.abs(covered - np.asarray(self.target_volumes)) + uncertainty)
                        / lower_measure,
                    ),
                    np.inf,
                )
            )
        transfer = FieldTransfer(
            source_field,
            target_field,
            primal,
            dual_pullback_operator=transpose(primal),
            hilbert_adjoint_operator=adjoint(primal),
            geometry=TransferGeometryBinding(
                self.source_geometry_id,
                self.target_geometry_id,
                "exact-restriction" if exact_restriction else "bounded-reconstruction",
                source_topology_id=self.source_topology_id,
                target_topology_id=self.target_topology_id,
                coverage_defect=geometry_defect,
            ),
            properties=TransferProperties(
                constant_preserving=self.surface_chart_deformation is None,
                conservative=True,
                positivity_preserving=True,
                adjoint_paired=True,
                differentiable_geometry=False,
                exact_on=("constants",) if self.surface_chart_deformation is None else (),
            ),
        )
        source_volumes = np.asarray(self.source_volumes, dtype=np.float64)
        target_volumes = np.asarray(self.target_volumes, dtype=np.float64)
        components = primal.source.size // source_volumes.size
        _, _, _, source_limit = _coverage_ledger(
            np.asarray(self.target_routes),
            np.asarray(self.source_indices),
            np.asarray(self.intersection_measures, dtype=np.float64),
            source_volumes,
            target_volumes,
            float(np.asarray(self.report.tolerance)),
            intersection_error_bounds=np.asarray(self.intersection_error_bounds),
            source_error_bounds=np.asarray(self.source_volume_error_bounds),
            target_error_bounds=np.asarray(self.target_volume_error_bounds),
        )
        if self.surface_chart_contents is not None:
            source_limit = np.asarray(
                self.surface_chart_contents.source_content_defect_bounds, dtype=np.float64
            )
        return TopologyEpochTransition(
            source_epoch,
            target_epoch,
            transfer,
            np.repeat(source_volumes, components),
            np.repeat(target_volumes, primal.target.size // target_volumes.size),
            measure_defect_bound=np.repeat(source_limit, components),
        )


class UnstructuredRemapLimiter(StrEnum):
    """Slope limiter of the second-order unstructured remap."""

    NONE = "none"
    BARTH_JESPERSEN = "barth-jespersen"


class UnstructuredSecondOrderRemapResult(StrictModule):
    """Target averages of one second-order remap with limiter and ledger evidence.

    ``limiter_factors`` holds the slope factor of every source cell and payload
    component; ``limited_count`` counts the factors below one.
    ``conservation_residual_before`` is source content minus target content of
    the limited reconstruction.  That residual is redistributed into target cells
    in proportion to their volume (no limiter) or to their slack against the
    neighborhood bounds (Barth-Jespersen), and ``conservation_residual_after`` is
    the measured residual of the returned values.  ``restored`` is false where
    the bounded slack could not absorb the whole residual.
    """

    values: Array
    limiter_factors: Array
    limited_count: Array
    limited_fraction: Array
    minimum_limiter_factor: Array
    conservation_residual_before: Array
    conservation_residual_after: Array
    redistributed_content: Array
    restored: Array


def _vertex_neighbor_stencils(mesh: CellMesh, /) -> tuple[np.ndarray, np.ndarray]:
    """Ascending distinct cells sharing a vertex with each cell, padded by row.

    Vertex stars reach across corners, so boundary and corner cells keep a
    full-rank gradient stencil on meshes where face neighbors do not.
    """
    counts = np.asarray(tuple(block.cell_count for block in mesh.blocks), np.int64)
    starts = np.cumsum(counts) - counts
    cell_count = np.sum(counts)
    cells = np.concatenate(
        tuple(
            np.broadcast_to(
                np.arange(start, start + block.cell_count, dtype=np.int64)[:, None],
                block.vertices.shape,
            )[np.asarray(block.vertex_valid)]
            for start, block in zip(starts, mesh.blocks, strict=True)
        )
    )
    vertices = np.concatenate(
        tuple(
            np.asarray(block.vertices, dtype=np.int64)[np.asarray(block.vertex_valid)]
            for block in mesh.blocks
        )
    )
    star_cells = cells[np.lexsort((cells, vertices))]
    star_counts = np.bincount(vertices, minlength=mesh.coordinates.shape[0])
    star_starts = np.cumsum(star_counts) - star_counts
    degree = star_counts[vertices]
    pair_rows = np.repeat(cells, degree)
    local = np.arange(pair_rows.size) - np.repeat(np.cumsum(degree) - degree, degree)
    pair_columns = star_cells[np.repeat(star_starts[vertices], degree) + local]
    distinct = pair_rows != pair_columns
    keys = np.unique(pair_rows[distinct] * cell_count + pair_columns[distinct])
    rows = keys // cell_count
    row_counts = np.bincount(rows, minlength=cell_count)
    slots = np.arange(keys.size) - np.repeat(
        np.cumsum(row_counts) - row_counts, row_counts
    )
    width = np.max(row_counts, initial=1)
    neighbors = np.zeros((cell_count, width), dtype=np.int32)
    valid = np.zeros((cell_count, width), dtype=np.bool_)
    neighbors[rows, slots] = keys % cell_count
    valid[rows, slots] = True
    return neighbors, valid


def _least_squares_gradient(
    centroids: np.ndarray,
    measures: np.ndarray,
    neighbors: np.ndarray,
    valid: np.ndarray,
    maximum_condition: float,
    /,
) -> SmallLinearSolveResult:
    """Weights of ``u_n - u_s`` in each cell's weighted least-squares gradient.

    Solves the scaled normal equations
    ``sum_n w_n d_n d_n^T g = sum_n w_n d_n (u_n - u_s)`` with
    ``d_n = (c_n - c_s) / h_s``, ``h_s = |s|^(1/d)``, and inverse-square-distance
    weights ``w_n = |d_n|^-2`` for all cells in one batched small solve whose
    rank and condition evidence is returned.
    """
    dimension = centroids.shape[1]
    lengths = measures ** (1.0 / dimension)
    offsets = (centroids[neighbors] - centroids[:, None, :]) / lengths[:, None, None]
    distance_squared = np.sum(offsets * offsets, axis=-1)
    if np.any(valid & ~(distance_squared > 0.0)):
        raise ValueError("Second-order remap stencil cells must have distinct centroids.")
    weights = np.where(valid, 1.0 / np.where(valid, distance_squared, 1.0), 0.0)
    normal = ein.contract("cw,cwi,cwj->cij", weights, offsets, offsets)
    right = ein.contract("cw,cwi->ciw", weights, offsets) / lengths[:, None, None]
    return solve_small_linear(
        SmallLinearSolvePlan(dimension, maximum_condition=maximum_condition),
        normal,
        right,
    )


def _barth_jespersen_factors(
    overlap_sources: EdgeRelation,
    entry_values: Array,
    increments: Array,
    entry_lower: Array,
    entry_upper: Array,
    /,
) -> Array:
    """Largest slope factor per source keeping every overlap average in bounds."""
    headroom = jnp.where(
        increments > 0.0, entry_upper - entry_values, entry_lower - entry_values
    )
    active = increments != 0.0
    ratio = jnp.where(active, headroom / jnp.where(active, increments, 1.0), 1.0)
    return route_reduce(overlap_sources, jnp.clip(ratio, 0.0, 1.0), reduction="min")


def _content_residual(source_content: Array, target_content: Array, /) -> Array:
    return compensated_sum_chunks((source_content, -target_content), output_ndim=1)


class UnstructuredSecondOrderRemapPlan(StrictModule, NonTrainableState):
    """Bound-preserving second-order conservative remap on a common refinement.

    Source cell ``s`` carries the linear reconstruction
    ``u_s + phi_s g_s . (x - c_s)`` about its certified decomposition centroid
    ``c_s``, where ``g_s = G u`` is the weighted least-squares gradient over the
    cells sharing a vertex with ``s`` (a prepared sparse map).  Its exact integral
    over overlap ``e = (s, t)`` with measure ``V_e`` and first moment ``m_e`` is
    ``I_e = V_e u_s + phi_s g_s . (m_e - c_s V_e)``, so linear fields are remapped
    exactly and every source cell keeps its content for any ``phi_s``.

    ``BARTH_JESPERSEN`` chooses the largest ``phi_s`` in ``[0, 1]`` that keeps
    every overlap average of ``s`` within the extrema of ``s`` and its stencil,
    so every target average is a convex combination of bounded overlap averages.
    ``apply`` is pure JAX in the field values and in the stored overlap and
    centroid geometry for the frozen combinatorics.
    """

    remap: UnstructuredConservativeRemapPlan
    overlap_first_moments: Array
    source_centroids: Array
    gradient: SparseLinearMap
    neighborhood: RowRelation
    overlap: EdgeRelation
    overlap_sources: EdgeRelation
    gradient_rank: Array
    gradient_condition: Array
    limiter: UnstructuredRemapLimiter = eqx.field(static=True)
    refinement_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        remap: UnstructuredConservativeRemapPlan,
        refinement: PreparedCommonRefinement,
        source: UnstructuredFiniteVolumeDiscretization,
        /,
        *,
        limiter: UnstructuredRemapLimiter = UnstructuredRemapLimiter.BARTH_JESPERSEN,
        maximum_condition: float = 1.0e12,
    ) -> None:
        from ...geometry._supermesh import PreparedCommonRefinement

        if not isinstance(remap, UnstructuredConservativeRemapPlan):
            raise TypeError("remap must be an UnstructuredConservativeRemapPlan.")
        if not isinstance(refinement, PreparedCommonRefinement):
            raise TypeError("refinement must be a PreparedCommonRefinement.")
        if not isinstance(source, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("source must be an unstructured FV discretization.")
        if not isinstance(limiter, UnstructuredRemapLimiter):
            raise TypeError("limiter must be an UnstructuredRemapLimiter.")
        condition_limit = float(maximum_condition)
        if not np.isfinite(condition_limit) or condition_limit <= 1.0:
            raise ValueError("maximum_condition must be finite and greater than one.")
        if not refinement.succeeded:
            raise ValueError(
                "Second-order remap requires a successful common refinement; got "
                f"{refinement.status.name}: {refinement.evidence.reason}"
            )
        if not remap.require_complete:
            raise ValueError("Second-order remap requires a complete-coverage remap.")
        if (
            source.topology_id != remap.source_topology_id
            or source.geometry_id != remap.source_geometry_id
        ):
            raise ValueError("source is not the source geometry of the remap.")
        if (
            refinement.dimension != source.cell_dimension
            or refinement.source_cell_count != source.cell_count
            or refinement.target_cell_count != remap.target_volumes.size
            or not np.array_equal(
                np.asarray(refinement.target_offsets), np.asarray(remap.target_offsets)
            )
            or not np.array_equal(
                np.asarray(refinement.source_cells), np.asarray(remap.source_indices)
            )
            or not np.array_equal(
                np.asarray(refinement.volumes),
                np.asarray(remap.intersection_measures),
            )
        ):
            raise ValueError("remap was not built from this common refinement.")
        measures = np.asarray(refinement.source_measures, dtype=np.float64)
        centroids = (
            np.asarray(refinement.source_first_moments, dtype=np.float64)
            / measures[:, None]
        )
        neighbors, valid = _vertex_neighbor_stencils(source.mesh)
        solution = _least_squares_gradient(
            centroids, measures, neighbors, valid, condition_limit
        )
        successful = np.asarray(solution.successful)
        if not np.all(successful):
            failed = np.flatnonzero(~successful)
            raise ValueError(
                "Second-order remap gradient stencils are rank deficient or exceed "
                f"maximum_condition for {failed.size} source cells (first: cell "
                f"{failed[0]})."
            )
        count, dimension = centroids.shape
        width = neighbors.shape[1] + 1
        columns = np.concatenate(
            (neighbors, np.arange(count, dtype=np.int32)[:, None]), axis=1
        )
        columns_valid = np.concatenate(
            (valid, np.ones((count, 1), dtype=np.bool_)), axis=1
        )
        # g_s = sum_n a_n (u_n - u_s): the owner column carries -sum_n a_n.
        neighbor_weights = np.where(valid[:, None, :], np.asarray(solution.value), 0.0)
        gradient_weights = np.concatenate(
            (neighbor_weights, -np.sum(neighbor_weights, axis=-1, keepdims=True)),
            axis=-1,
        )
        gradient_id = canonical_fingerprint(
            {
                "kind": "unstructured-remap-least-squares-gradient",
                "source": source.prepared_id,
                "centroids": array_tree_fingerprint(centroids),
                "stencil": array_tree_fingerprint(columns),
                "stencil_valid": array_tree_fingerprint(columns_valid),
                "weights": array_tree_fingerprint(gradient_weights),
            }
        )
        gradient = SparseLinearMap(
            RowRelation(
                np.broadcast_to(columns[:, None, :], (count, dimension, width)),
                source_size=count,
                valid=np.broadcast_to(
                    columns_valid[:, None, :], (count, dimension, width)
                ),
            ),
            gradient_weights,
            operator_id=gradient_id,
        )
        overlap = EdgeRelation(
            remap.source_indices,
            remap.target_routes,
            source_size=count,
            target_size=remap.target_volumes.size,
        )
        self.remap = remap
        self.overlap_first_moments = jnp.asarray(
            refinement.first_moments, dtype=jnp.float64
        )
        self.source_centroids = jnp.asarray(centroids)
        self.gradient = gradient
        self.neighborhood = RowRelation(columns, source_size=count, valid=columns_valid)
        self.overlap = overlap
        self.overlap_sources = overlap.transpose()
        self.gradient_rank = solution.rank
        self.gradient_condition = solution.condition_estimate
        self.limiter = limiter
        self.refinement_id = refinement.refinement_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-second-order-remap",
                "remap": remap.plan_id,
                "refinement": refinement.refinement_id,
                "source": source.prepared_id,
                "limiter": limiter.value,
                "maximum_condition": condition_limit,
                "gradient": gradient_id,
            }
        )

    def apply(
        self, source_cell_averages: ArrayLike, /
    ) -> UnstructuredSecondOrderRemapResult:
        """Remap cell averages through the limited linear reconstruction."""
        remap = self.remap
        value = jnp.asarray(source_cell_averages)
        if value.ndim == 0 or value.shape[0] != remap.source_volumes.size:
            raise ValueError(
                "Second-order remap values must begin with source cell count."
            )
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("Second-order remap values must be real floating point.")
        payload_shape = value.shape[1:]
        source = value.reshape((value.shape[0], -1))
        dtype = source.dtype
        measures = remap.intersection_measures.astype(dtype)[:, None]
        target_volumes = remap.target_volumes.astype(dtype)[:, None]
        source_content = remap.source_volumes.astype(dtype)[:, None] * source
        centroid_offsets = self.overlap_first_moments.astype(
            dtype
        ) / measures - gather_routes(self.overlap, self.source_centroids.astype(dtype))
        increments = ein.contract(
            "ed,edc->ec",
            centroid_offsets,
            gather_routes(self.overlap, self.gradient.mv(source)),
        )
        entry_values = gather_routes(self.overlap, source)
        match self.limiter:
            case UnstructuredRemapLimiter.NONE:
                factors = jnp.ones_like(source)
                target_content = route_reduce(
                    self.overlap, measures * (entry_values + increments)
                )
                residual = _content_residual(source_content, target_content)
                capacity = jnp.broadcast_to(target_volumes, target_content.shape)
                amount = residual
            case UnstructuredRemapLimiter.BARTH_JESPERSEN:
                stencil_values = gather_routes(self.neighborhood, source)
                lower = route_reduce(self.neighborhood, stencil_values, reduction="min")
                upper = route_reduce(self.neighborhood, stencil_values, reduction="max")
                entry_lower = gather_routes(self.overlap, lower)
                entry_upper = gather_routes(self.overlap, upper)
                factors = _barth_jespersen_factors(
                    self.overlap_sources,
                    entry_values,
                    increments,
                    entry_lower,
                    entry_upper,
                )
                target_content = route_reduce(
                    self.overlap,
                    measures
                    * (entry_values + gather_routes(self.overlap, factors) * increments),
                )
                residual = _content_residual(source_content, target_content)
                target_lower = route_reduce(self.overlap, entry_lower, reduction="min")
                target_upper = route_reduce(self.overlap, entry_upper, reduction="max")
                capacity = jnp.maximum(
                    jnp.where(
                        residual > 0.0,
                        target_volumes * target_upper - target_content,
                        target_content - target_volumes * target_lower,
                    ),
                    0.0,
                )
                amount = jnp.sign(residual) * jnp.minimum(
                    jnp.abs(residual), jnp.sum(capacity, axis=0)
                )
            case _:
                raise ValueError(f"Unknown remap limiter {self.limiter!r}.")
        available = jnp.sum(capacity, axis=0)
        restored_content = (
            target_content
            + capacity / jnp.where(available > 0.0, available, 1.0) * amount
        )
        limited_count = jnp.sum(factors < 1.0, dtype=jnp.int32)
        return UnstructuredSecondOrderRemapResult(
            values=(restored_content / target_volumes).reshape(
                (remap.target_volumes.size,) + payload_shape
            ),
            limiter_factors=factors.reshape((source.shape[0],) + payload_shape),
            limited_count=limited_count,
            limited_fraction=limited_count.astype(dtype) / factors.size,
            minimum_limiter_factor=jnp.min(factors),
            conservation_residual_before=residual.reshape(payload_shape),
            conservation_residual_after=_content_residual(
                source_content, restored_content
            ).reshape(payload_shape),
            redistributed_content=amount.reshape(payload_shape),
            restored=(amount == residual).reshape(payload_shape),
        )


__all__ = [
    "UnstructuredConservativeRemapPlan",
    "UnstructuredRemapLimiter",
    "UnstructuredRemapReport",
    "UnstructuredSecondOrderRemapPlan",
    "UnstructuredSecondOrderRemapResult",
]
