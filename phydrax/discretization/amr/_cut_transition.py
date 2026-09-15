#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical common-refinement transfer between multivalued block-AMR epochs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._cut_complex import MultivaluedCutCellComplex


if TYPE_CHECKING:
    from ..finite_volume._unstructured_remap import UnstructuredConservativeRemapPlan


def _point_in_tetrahedron(
    point: np.ndarray,
    tetrahedron: np.ndarray,
    tolerance: float,
    /,
) -> bool:
    matrix = np.stack(
        (
            tetrahedron[1] - tetrahedron[0],
            tetrahedron[2] - tetrahedron[0],
            tetrahedron[3] - tetrahedron[0],
        ),
        axis=1,
    )
    barycentric = np.linalg.solve(matrix, point - tetrahedron[0])
    coordinates = np.concatenate((np.asarray((1.0 - np.sum(barycentric),)), barycentric))
    return bool(
        np.all(coordinates >= -tolerance) and np.all(coordinates <= 1.0 + tolerance)
    )


def _component_contains(
    container: tuple[np.ndarray, ...],
    candidate: tuple[np.ndarray, ...],
    tolerance: float,
    /,
) -> bool:
    points = {
        tuple(float(value) for value in point)
        for tetrahedron in candidate
        for point in tetrahedron
    }
    return all(
        any(
            _point_in_tetrahedron(
                np.asarray(point),
                tetrahedron,
                tolerance,
            )
            for tetrahedron in container
        )
        for point in points
    )


class MultivaluedCutCellTransitionResult(StrictModule):
    """Padded target content and exact global conservation evidence."""

    target_content: Array
    source_total: Array
    target_total: Array
    conservation_residual: Array
    successful: Array
    transition_id: str = eqx.field(static=True)


class MultivaluedCutCellTransition(StrictModule, NonTrainableState):
    """Conservative physical-overlap transfer across regrid/rebox epochs."""

    source: MultivaluedCutCellComplex
    target: MultivaluedCutCellComplex
    remap: UnstructuredConservativeRemapPlan
    source_coverage: Array
    target_coverage: Array
    coverage_complete: bool = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: MultivaluedCutCellComplex,
        target: MultivaluedCutCellComplex,
        /,
        *,
        tolerance: float = 1.0e-10,
        require_complete: bool = True,
    ):
        if not isinstance(source, MultivaluedCutCellComplex) or not isinstance(
            target, MultivaluedCutCellComplex
        ):
            raise TypeError("Cut-cell transitions require source and target complexes.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Cut-cell transition tolerance must be positive and finite.")
        from ...geometry._tetra_intersections import (
            intersect_tetrahedra,
            TetraIntersectionStatus,
            TetraIntersectionTolerance,
        )
        from ..finite_volume._unstructured_remap import (
            UnstructuredConservativeRemapPlan,
        )

        source_geometry = source.finite_volume_plan().prepare(
            numeric_version="cut-transition-source"
        )
        target_geometry = target.finite_volume_plan().prepare(
            numeric_version="cut-transition-target"
        )
        source_tetrahedra = tuple(
            tuple(np.asarray(tetra, dtype=float) for tetra in component)
            for component in source.component_tetrahedra
        )
        target_tetrahedra = tuple(
            tuple(np.asarray(tetra, dtype=float) for tetra in component)
            for component in target.component_tetrahedra
        )
        source_lower = np.asarray(
            [
                np.min(np.concatenate(component, axis=0), axis=0)
                for component in source_tetrahedra
            ]
        )
        source_upper = np.asarray(
            [
                np.max(np.concatenate(component, axis=0), axis=0)
                for component in source_tetrahedra
            ]
        )
        target_lower = np.asarray(
            [
                np.min(np.concatenate(component, axis=0), axis=0)
                for component in target_tetrahedra
            ]
        )
        target_upper = np.asarray(
            [
                np.max(np.concatenate(component, axis=0), axis=0)
                for component in target_tetrahedra
            ]
        )
        predicate = TetraIntersectionTolerance(
            absolute=0.0,
            relative=min(tolerance_, 1.0e-10),
        )
        records: list[tuple[int, int, float]] = []
        if (
            source.mesh.geometry_id == target.mesh.geometry_id
            and source.component_count == target.component_count
        ):
            records.extend(
                (index, index, float(source_geometry.cell_volumes[index]))
                for index in range(source.component_count)
            )
        else:
            for target_index in range(target.component_count):
                candidates = np.flatnonzero(
                    np.all(source_lower <= target_upper[target_index], axis=1)
                    & np.all(source_upper >= target_lower[target_index], axis=1)
                )
                for source_index in candidates:
                    containment_tolerance = tolerance_ * max(
                        1.0,
                        float(np.max(np.abs(source_upper[int(source_index)]))),
                        float(np.max(np.abs(target_upper[target_index]))),
                    )
                    if _component_contains(
                        source_tetrahedra[int(source_index)],
                        target_tetrahedra[target_index],
                        containment_tolerance,
                    ):
                        overlap = float(target_geometry.cell_volumes[target_index])
                        records.append((target_index, int(source_index), overlap))
                        continue
                    if _component_contains(
                        target_tetrahedra[target_index],
                        source_tetrahedra[int(source_index)],
                        containment_tolerance,
                    ):
                        overlap = float(source_geometry.cell_volumes[int(source_index)])
                        records.append((target_index, int(source_index), overlap))
                        continue
                    overlap = 0.0
                    for source_tetrahedron in source_tetrahedra[int(source_index)]:
                        for target_tetrahedron in target_tetrahedra[target_index]:
                            overlap_width = np.minimum(
                                np.max(source_tetrahedron, axis=0),
                                np.max(target_tetrahedron, axis=0),
                            ) - np.maximum(
                                np.min(source_tetrahedron, axis=0),
                                np.min(target_tetrahedron, axis=0),
                            )
                            pair_scale = max(
                                1.0,
                                float(np.max(np.abs(source_tetrahedron))),
                                float(np.max(np.abs(target_tetrahedron))),
                            )
                            if np.any(overlap_width <= tolerance_ * pair_scale):
                                continue
                            result = intersect_tetrahedra(
                                source_tetrahedron,
                                target_tetrahedron,
                                source_id=int(source_index),
                                target_id=target_index,
                                tolerance=predicate,
                                volume_only=True,
                            )
                            if result.status is TetraIntersectionStatus.SUCCESS:
                                overlap += result.volume
                            elif result.status not in {
                                TetraIntersectionStatus.DISJOINT,
                                TetraIntersectionStatus.ZERO_MEASURE_CONTACT,
                            }:
                                raise ValueError(
                                    "Cut-cell common refinement has an unresolved "
                                    f"tetrahedron predicate: {result.status.value}."
                                )
                    if overlap > 0.0:
                        records.append((target_index, int(source_index), overlap))
        records.sort(key=lambda value: (value[0], value[1]))
        source_coverage = np.zeros((source.component_count,), dtype=float)
        target_coverage = np.zeros((target.component_count,), dtype=float)
        for target_index, source_index, overlap in records:
            source_coverage[source_index] += overlap
            target_coverage[target_index] += overlap
        source_volumes = np.asarray(source_geometry.cell_volumes, dtype=float)
        target_volumes = np.asarray(target_geometry.cell_volumes, dtype=float)
        source_complete = np.allclose(
            source_coverage,
            source_volumes,
            rtol=tolerance_,
            atol=tolerance_ * max(1.0, float(np.max(source_volumes))),
        )
        target_complete = np.allclose(
            target_coverage,
            target_volumes,
            rtol=tolerance_,
            atol=tolerance_ * max(1.0, float(np.max(target_volumes))),
        )
        coverage_complete = bool(source_complete and target_complete)
        if bool(require_complete) and not coverage_complete:
            raise ValueError(
                "Cut-cell source or target common-refinement coverage is incomplete."
            )
        offsets = np.zeros((target.component_count + 1,), dtype=np.int32)
        for target_index, _, _ in records:
            offsets[target_index + 1] += 1
        np.cumsum(offsets, out=offsets)
        source_indices = np.asarray(
            [source_index for _, source_index, _ in records], dtype=np.int32
        )
        measures = np.asarray([value for _, _, value in records], dtype=float)
        remap = UnstructuredConservativeRemapPlan(
            source_geometry,
            target_geometry,
            offsets,
            source_indices,
            measures,
            method="block-amr-cut-component-common-refinement",
            provenance="block-amr-multivalued-common-refinement",
            tolerance=tolerance_,
            require_complete=bool(require_complete),
        )
        self.source = source
        self.target = target
        self.remap = remap
        self.source_coverage = jnp.asarray(source_coverage)
        self.target_coverage = jnp.asarray(target_coverage)
        self.coverage_complete = coverage_complete
        self.transition_id = canonical_fingerprint(
            {
                "kind": "multivalued-cut-cell-transition",
                "source": source.topology_id,
                "source_geometry": source.geometry_id,
                "target": target.topology_id,
                "target_geometry": target.geometry_id,
                "remap": remap.plan_id,
                "coverage_complete": coverage_complete,
            }
        )

    def _source_values(self, values: ArrayLike, name: str, /) -> Array:
        array = jnp.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.source.component_capacity:
            raise ValueError(f"{name} must begin with padded source component capacity.")
        source_count = self.source.component_count
        inactive = ~self.source.component_active
        trailing = (1,) * (array.ndim - 1)
        array = eqx.error_if(
            array,
            jnp.any(array * inactive.reshape(inactive.shape + trailing) != 0.0),
            f"{name} must be zero on inactive source component slots.",
        )
        return array[:source_count]

    def _pad_target(self, values: Array, /) -> Array:
        target = jnp.zeros(
            (self.target.component_capacity,) + values.shape[1:], dtype=values.dtype
        )
        return target.at[: self.target.component_count].set(values)

    def apply_content(
        self,
        source_content: ArrayLike,
        /,
    ) -> MultivaluedCutCellTransitionResult:
        source = self._source_values(source_content, "Source cut-cell content")
        target_active = jnp.ones((self.target.component_count,), dtype=bool)
        transferred = self.remap.apply_content(
            source,
            target_active_mask=target_active,
        )
        padded = self._pad_target(transferred)
        source_total = jnp.sum(source, axis=0)
        target_total = jnp.sum(transferred, axis=0)
        residual = target_total - source_total
        scale = jnp.maximum(jnp.abs(source_total), jnp.asarray(1.0, dtype=residual.dtype))
        successful = jnp.all(
            jnp.abs(residual) <= 512.0 * jnp.finfo(residual.dtype).eps * scale
        )
        return MultivaluedCutCellTransitionResult(
            target_content=padded,
            source_total=source_total,
            target_total=target_total,
            conservation_residual=residual,
            successful=successful,
            transition_id=self.transition_id,
        )

    def apply_average(self, source_average: ArrayLike, /) -> Array:
        source = self._source_values(source_average, "Source cut-cell average")
        transferred = self.remap.apply(source)
        return self._pad_target(transferred)

    def transpose_content(self, target_cotangent: ArrayLike, /) -> Array:
        """Apply the exact algebraic transpose of extensive remap."""

        target = jnp.asarray(target_cotangent)
        if target.ndim == 0 or target.shape[0] != self.target.component_capacity:
            raise ValueError(
                "Target cotangent must begin with padded target component capacity."
            )
        target = target[: self.target.component_count]
        trailing = (1,) * (target.ndim - 1)
        weights = (
            self.remap.intersection_measures
            / self.remap.source_volumes[self.remap.source_indices]
        )
        contributions = target[self.remap.target_routes] * weights.reshape(
            weights.shape + trailing
        )
        source = (
            jnp.zeros(
                (self.source.component_count,) + target.shape[1:], dtype=target.dtype
            )
            .at[self.remap.source_indices]
            .add(contributions)
        )
        padded = jnp.zeros(
            (self.source.component_capacity,) + source.shape[1:], dtype=source.dtype
        )
        return padded.at[: self.source.component_count].set(source)

    def hilbert_adjoint_average(self, target_value: ArrayLike, /) -> Array:
        """Apply the volume-paired adjoint of average transfer."""

        target = jnp.asarray(target_value)
        if target.ndim == 0 or target.shape[0] != self.target.component_capacity:
            raise ValueError(
                "Target adjoint value must begin with padded target component capacity."
            )
        target_active = target[: self.target.component_count]
        trailing = (1,) * (target.ndim - 1)
        weighted_target = target_active * self.remap.target_volumes.reshape(
            self.remap.target_volumes.shape + trailing
        )
        algebraic = self.transpose_content(self._pad_target(weighted_target))
        source_volumes = jnp.asarray(self.source.component_volumes)
        safe = jnp.where(self.source.component_active, source_volumes, 1.0)
        result = algebraic / safe.reshape(safe.shape + trailing)
        return jnp.where(
            self.source.component_active.reshape(
                self.source.component_active.shape + trailing
            ),
            result,
            jnp.zeros((), dtype=result.dtype),
        )


__all__ = [
    "MultivaluedCutCellTransition",
    "MultivaluedCutCellTransitionResult",
]
