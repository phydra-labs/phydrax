#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._unstructured import UnstructuredFiniteVolumeDiscretization


class UnstructuredRemapReport(StrictModule):
    """Host-certified coverage evidence for one common-refinement map."""

    maximum_target_coverage_defect: Array
    maximum_source_coverage_defect: Array
    uncovered_target_measure: Array
    uncovered_source_measure: Array
    donor_excess_measure: Array
    source_measure: Array
    target_measure: Array
    coverage_complete: Array
    tolerance: Array


def _active_mask(value: ArrayLike | None, count: int, name: str, /) -> Array:
    if value is None:
        return jnp.ones((count,), dtype=bool)
    array = jnp.asarray(value)
    if array.shape != (count,) or array.dtype != jnp.dtype(bool):
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


class UnstructuredConservativeRemapPlan(StrictModule, NonTrainableState):
    """Explicit CSR common-refinement artifact between two immutable mesh epochs.

    The CSR measures describe overlap of *geometric* cells.  ``apply`` consumes
    cell averages, while ``apply_content`` consumes extensive conserved content
    and never divides by a source or target measure.  The latter is the path
    used by AMR transfer of runtime conservative state and fluid volume.
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
    source_volumes: Array
    target_volumes: Array
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
    ):
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
        measures = np.asarray(intersection_measures, dtype=float)
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
        target_coverage = np.bincount(
            target_routes,
            weights=measures,
            minlength=target.cell_count,
        )
        source_coverage = np.bincount(
            indices,
            weights=measures,
            minlength=source.cell_count,
        )
        target_defect = target_coverage - target_volumes
        source_defect = source_coverage - source_volumes
        target_scale = np.maximum(target_volumes, np.max(target_volumes) * 1e-14)
        source_scale = np.maximum(source_volumes, np.max(source_volumes) * 1e-14)
        complete = np.all(np.abs(target_defect) <= tolerance_ * target_scale) and np.all(
            np.abs(source_defect) <= tolerance_ * source_scale
        )
        if require_complete and not complete:
            raise ValueError(
                "Conservative remap does not completely cover source and target."
            )
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
        self.source_volumes = source.cell_volumes
        self.target_volumes = target.cell_volumes
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
                "source_cell_global_ids": array_tree_fingerprint(source.cell_global_ids),
                "target_cell_global_ids": array_tree_fingerprint(target.cell_global_ids),
                "target_offsets": array_tree_fingerprint(offsets),
                "source_indices": array_tree_fingerprint(indices),
                "intersection_measures": array_tree_fingerprint(measures),
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
        weighted = value[self.source_indices] * self.intersection_measures.astype(
            value.dtype
        ).reshape((-1,) + trailing)
        target = jnp.zeros(
            (self.target_volumes.size,) + value.shape[1:], dtype=value.dtype
        )
        target = target.at[self.target_routes].add(weighted)
        denominator = _volume_array(
            target_volumes,
            self.target_volumes,
            self.target_volumes.size,
            "target_volumes",
        )
        denominator = denominator.astype(value.dtype).reshape((-1,) + trailing)
        target = target / denominator
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

    def apply_extensive(self, source_content: ArrayLike, /, **kwargs) -> Array:
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

    def conservation_defect(
        self, source_cell_averages: ArrayLike, target_cell_averages: ArrayLike, /
    ) -> Array:
        source = jnp.asarray(source_cell_averages)
        target = jnp.asarray(target_cell_averages)
        if source.ndim == 0 or source.shape[0] != self.source_volumes.size:
            raise ValueError("source_cell_averages must begin with source cell count.")
        if target.ndim == 0 or target.shape[0] != self.target_volumes.size:
            raise ValueError("target_cell_averages must begin with target cell count.")
        source_integral = jnp.sum(
            self.source_volumes.astype(source.dtype).reshape(
                (-1,) + (1,) * (source.ndim - 1)
            )
            * source,
            axis=0,
        )
        target_integral = jnp.sum(
            self.target_volumes.astype(target.dtype).reshape(
                (-1,) + (1,) * (target.ndim - 1)
            )
            * target,
            axis=0,
        )
        return target_integral - source_integral

    def conservation_defect_content(
        self, source_content: ArrayLike, target_content: ArrayLike, /
    ) -> Array:
        source = jnp.asarray(source_content)
        target = jnp.asarray(target_content)
        if source.ndim == 0 or source.shape[0] != self.source_volumes.size:
            raise ValueError("source_content must begin with source cell count.")
        if target.ndim == 0 or target.shape[0] != self.target_volumes.size:
            raise ValueError("target_content must begin with target cell count.")
        return jnp.sum(target, axis=0) - jnp.sum(source, axis=0)


__all__ = ["UnstructuredConservativeRemapPlan", "UnstructuredRemapReport"]
