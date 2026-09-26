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
from ..._geometry_predicates import orient3d, PredicateMode, PredicateSign
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_mesh import CellMesh
from ._cut_complex import MultivaluedCutCellComplex


if TYPE_CHECKING:
    from ...geometry._supermesh import CommonRefinementEvidence
    from ..finite_volume._unstructured_remap import UnstructuredConservativeRemapPlan


def _component_tetrahedra_mesh(
    complex_: MultivaluedCutCellComplex,
    /,
) -> tuple[CellMesh, np.ndarray]:
    """Tetrahedral cell mesh of the active cut components and each cell's component."""
    tetrahedra = np.concatenate(
        tuple(
            np.asarray(component, dtype=np.float64)
            for component in complex_.component_tetrahedra
        ),
        axis=0,
    )
    owners = np.repeat(
        np.arange(complex_.component_count, dtype=np.int32),
        np.asarray(
            tuple(len(component) for component in complex_.component_tetrahedra),
            dtype=np.int64,
        ),
    )
    signs = np.asarray(
        orient3d(
            tetrahedra[:, 0],
            tetrahedra[:, 1],
            tetrahedra[:, 2],
            tetrahedra[:, 3],
            mode=PredicateMode.EXACT,
        ).signs
    )
    if np.any(signs == PredicateSign.NEGATIVE):
        raise ValueError("Cut component tetrahedra must be positively oriented.")
    # An exactly degenerate tetrahedron has zero measure: excluding it leaves every
    # component measure and every overlap measure unchanged, and the remaining cells
    # reference four distinct points as a cell mesh requires.
    positive = signs == PredicateSign.POSITIVE
    if np.any(np.bincount(owners[positive], minlength=complex_.component_count) == 0):
        raise ValueError("Every cut component requires a positive-measure tetrahedron.")
    coordinates, inverse = np.unique(
        tetrahedra[positive].reshape(-1, 3), axis=0, return_inverse=True
    )
    mesh = CellMesh.from_tetrahedra(
        coordinates,
        inverse.reshape(-1, 4).astype(np.int32),
        block_name="cut-component-tetrahedra",
        numeric_version=complex_.geometry_id,
    )
    return mesh, owners[positive]


class MultivaluedCutCellTransitionResult(StrictModule):
    """Padded target content and exact global conservation evidence."""

    target_content: Array
    source_total: Array
    target_total: Array
    conservation_residual: Array
    successful: Array
    transition_id: str = eqx.field(static=True)


class MultivaluedCutCellTransition(StrictModule, NonTrainableState):
    """Conservative physical-overlap transfer across regrid/rebox epochs.

    Each active cut component is the union of its positively oriented
    tetrahedra.  The tetrahedral meshes of both complexes are intersected by the
    canonical common refinement (:func:`phydrax.geometry.prepare_common_refinement`)
    and the overlap volumes are summed per ``(target, source)`` component pair.
    ``require_complete`` requests complete source and target coverage; otherwise
    uncovered measure is reported by ``source_coverage``/``target_coverage`` and
    ``refinement_evidence``.  Any refinement failure raises ``ValueError``.
    """

    source: MultivaluedCutCellComplex
    target: MultivaluedCutCellComplex
    remap: UnstructuredConservativeRemapPlan
    source_coverage: Array
    target_coverage: Array
    refinement_evidence: CommonRefinementEvidence
    coverage_complete: bool = eqx.field(static=True)
    refinement_id: str = eqx.field(static=True)
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
        if not isinstance(require_complete, bool):
            raise TypeError("Cut-cell transition require_complete must be a bool.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Cut-cell transition tolerance must be positive and finite.")
        from ...geometry._supermesh import (
            CommonRefinementCoverage,
            CommonRefinementPolicy,
            CommonRefinementStatus,
            prepare_common_refinement,
        )
        from ..finite_volume._unstructured_remap import (
            UnstructuredConservativeRemapPlan,
        )

        source_mesh, source_owners = _component_tetrahedra_mesh(source)
        target_mesh, target_owners = _component_tetrahedra_mesh(target)
        refinement = prepare_common_refinement(
            source_mesh,
            target_mesh,
            policy=CommonRefinementPolicy(
                coverage=(
                    CommonRefinementCoverage.COMPLETE
                    if require_complete
                    else CommonRefinementCoverage.PARTIAL
                ),
                coverage_tolerance=tolerance_,
            ),
        )
        match refinement.status:
            case CommonRefinementStatus.SUCCESS:
                pass
            case CommonRefinementStatus.COVERAGE_GAP:
                raise ValueError(
                    "Cut-cell source or target common-refinement coverage is "
                    f"incomplete (COVERAGE_GAP): {refinement.evidence.reason}"
                )
            case (
                CommonRefinementStatus.INVALID_GEOMETRY
                | CommonRefinementStatus.PREDICATE_UNCERTAIN
                | CommonRefinementStatus.INTERSECTION_FAILURE
                | CommonRefinementStatus.DOUBLE_COVERAGE
                | CommonRefinementStatus.RESOURCE_LIMIT
            ):
                raise ValueError(
                    "Cut-cell common refinement failed with status "
                    f"{refinement.status.name}: {refinement.evidence.reason}"
                )
            case _:
                raise ValueError(
                    f"Unknown common-refinement status {refinement.status!r}."
                )
        source_count = source.component_count
        target_count = target.component_count
        # Canonical component-pair keys sort by target component, then source
        # component, which is exactly the CSR row order of the remap plan.
        pair_keys = target_owners[np.asarray(refinement.target_cells)].astype(
            np.int64
        ) * source_count + source_owners[np.asarray(refinement.source_cells)].astype(
            np.int64
        )
        unique_keys, pair_inverse = np.unique(pair_keys, return_inverse=True)
        measures = np.bincount(
            pair_inverse.reshape(-1),
            weights=np.asarray(refinement.volumes, dtype=np.float64),
            minlength=unique_keys.size,
        )
        pair_targets = (unique_keys // source_count).astype(np.int32)
        pair_sources = (unique_keys % source_count).astype(np.int32)
        offsets = np.zeros((target_count + 1,), dtype=np.int32)
        offsets[1:] = np.cumsum(np.bincount(pair_targets, minlength=target_count))
        source_geometry = source.finite_volume_plan().prepare(
            numeric_version="cut-transition-source"
        )
        target_geometry = target.finite_volume_plan().prepare(
            numeric_version="cut-transition-target"
        )
        source_volumes = np.asarray(source_geometry.cell_volumes, dtype=np.float64)
        target_volumes = np.asarray(target_geometry.cell_volumes, dtype=np.float64)
        source_coverage = np.bincount(
            pair_sources, weights=measures, minlength=source_count
        )
        target_coverage = np.bincount(
            pair_targets, weights=measures, minlength=target_count
        )
        # A component tolerates the summed coverage tolerances of its tetrahedra,
        # so component completeness is the aggregate of the refinement's own test.
        source_tolerances = np.bincount(
            source_owners,
            weights=np.asarray(
                refinement.evidence.source_coverage_tolerances, dtype=np.float64
            ),
            minlength=source_count,
        )
        target_tolerances = np.bincount(
            target_owners,
            weights=np.asarray(
                refinement.evidence.target_coverage_tolerances, dtype=np.float64
            ),
            minlength=target_count,
        )
        coverage_complete = bool(
            np.all(np.abs(source_coverage - source_volumes) <= source_tolerances)
            and np.all(np.abs(target_coverage - target_volumes) <= target_tolerances)
        )
        if require_complete and not coverage_complete:
            raise ValueError(
                "Cut-cell source or target common-refinement coverage is incomplete."
            )
        remap = UnstructuredConservativeRemapPlan(
            source_geometry,
            target_geometry,
            offsets,
            pair_sources,
            measures,
            method="block-amr-cut-component-common-refinement",
            provenance=refinement.refinement_id,
            tolerance=tolerance_,
            require_complete=require_complete,
        )
        self.source = source
        self.target = target
        self.remap = remap
        self.source_coverage = jnp.asarray(source_coverage, dtype=jnp.float64)
        self.target_coverage = jnp.asarray(target_coverage, dtype=jnp.float64)
        self.refinement_evidence = refinement.evidence
        self.coverage_complete = coverage_complete
        self.refinement_id = refinement.refinement_id
        self.transition_id = canonical_fingerprint(
            {
                "kind": "multivalued-cut-cell-transition",
                "source": source.topology_id,
                "source_geometry": source.geometry_id,
                "target": target.topology_id,
                "target_geometry": target.geometry_id,
                "refinement": refinement.refinement_id,
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
        target_active = jnp.ones((self.target.component_count,), dtype=jnp.bool_)
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
