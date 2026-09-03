#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._state import ClosureSnapshot


AlignmentKind = Literal["identity", "restriction", "prolongation"]


class ConservativeAlignmentPlan(StrictModule, NonTrainableState):
    """Symbolic cell-average alignment between nested Cartesian index spaces."""

    conservation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, conservation_tolerance: float = 1e-10):
        tolerance = float(conservation_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("conservation_tolerance must be finite and nonnegative.")
        self.conservation_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-alignment-plan",
                "conservation_tolerance": tolerance,
            }
        )

    def prepare(
        self,
        source_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
        /,
        *,
        source_cell_volumes: ArrayLike | None = None,
        target_cell_volumes: ArrayLike | None = None,
    ) -> PreparedConservativeAlignment:
        return PreparedConservativeAlignment(
            self,
            source_shape,
            target_shape,
            source_cell_volumes=source_cell_volumes,
            target_cell_volumes=target_cell_volumes,
        )


class ConservativeAlignmentResult(StrictModule, NonTrainableState):
    values: Array
    source_integral: Array
    target_integral: Array
    conservation_defect: Array
    alignment_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        values: ArrayLike,
        source_integral: ArrayLike,
        target_integral: ArrayLike,
        alignment_id: str,
    ):
        source = jnp.asarray(source_integral)
        target = jnp.asarray(target_integral)
        if source.shape != target.shape:
            raise ValueError("Source and target integrals must share shape.")
        identifier = str(alignment_id).strip()
        if not identifier:
            raise ValueError("alignment_id must be non-empty.")
        self.values = jnp.asarray(values)
        self.source_integral = source
        self.target_integral = target
        self.conservation_defect = target - source
        self.alignment_id = identifier


class PreparedConservativeAlignment(StrictModule, NonTrainableState):
    """Prepared conservative restriction or piecewise-constant prolongation."""

    plan: ConservativeAlignmentPlan
    source_cell_volumes: Array
    target_cell_volumes: Array
    source_shape: tuple[int, ...] = eqx.field(static=True)
    target_shape: tuple[int, ...] = eqx.field(static=True)
    refinement_ratio: tuple[int, ...] = eqx.field(static=True)
    kind: AlignmentKind = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ConservativeAlignmentPlan,
        source_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
        /,
        *,
        source_cell_volumes: ArrayLike | None = None,
        target_cell_volumes: ArrayLike | None = None,
    ):
        if not isinstance(plan, ConservativeAlignmentPlan):
            raise TypeError("plan must be a ConservativeAlignmentPlan.")
        source = tuple(int(value) for value in source_shape)
        target = tuple(int(value) for value in target_shape)
        if (
            not source
            or len(source) != len(target)
            or any(value <= 0 for value in (*source, *target))
        ):
            raise ValueError("Alignment shapes must be positive and have equal rank.")
        if source == target:
            kind: AlignmentKind = "identity"
            ratio = (1,) * len(source)
        elif all(left % right == 0 for left, right in zip(source, target, strict=True)):
            kind = "restriction"
            ratio = tuple(
                left // right for left, right in zip(source, target, strict=True)
            )
        elif all(right % left == 0 for left, right in zip(source, target, strict=True)):
            kind = "prolongation"
            ratio = tuple(
                right // left for left, right in zip(source, target, strict=True)
            )
        else:
            raise ValueError(
                "Conservative alignment requires uniformly nested source and target shapes."
            )
        source_volumes = _volumes(source_cell_volumes, source)
        target_volumes = _volumes(target_cell_volumes, target)
        if source_cell_volumes is None:
            source_volumes = jnp.full(source, 1.0 / np.prod(source), dtype=jnp.float64)
        if target_cell_volumes is None:
            target_volumes = jnp.full(target, 1.0 / np.prod(target), dtype=jnp.float64)
        if kind == "restriction":
            block_target = target_volumes
            block_source = _block_sum(source_volumes, ratio)
        elif kind == "prolongation":
            block_target = _block_sum(target_volumes, ratio)
            block_source = source_volumes
        else:
            block_target = target_volumes
            block_source = source_volumes
        volume_defect = np.max(
            np.abs(np.asarray(block_target) - np.asarray(block_source))
        )
        volume_scale = max(1.0, float(np.max(np.abs(np.asarray(block_source)))))
        if volume_defect > plan.conservation_tolerance * volume_scale:
            raise ValueError(
                "Source and target cell volumes do not describe the same nested control volumes."
            )
        self.plan = plan
        self.source_cell_volumes = source_volumes
        self.target_cell_volumes = target_volumes
        self.source_shape = source
        self.target_shape = target
        self.refinement_ratio = ratio
        self.kind = kind
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-conservative-alignment",
                "plan": plan.plan_id,
                "source_shape": list(source),
                "target_shape": list(target),
                "refinement_ratio": list(ratio),
                "mode": kind,
                "source_volumes": array_tree_fingerprint(source_volumes),
                "target_volumes": array_tree_fingerprint(target_volumes),
            }
        )

    def execute(self, values: ArrayLike, /) -> ConservativeAlignmentResult:
        source = jnp.asarray(values)
        rank = len(self.source_shape)
        if source.ndim < rank or tuple(source.shape[:rank]) != self.source_shape:
            raise ValueError(
                f"Alignment values must begin with source shape {self.source_shape}; "
                f"got {source.shape}."
            )
        if self.kind == "identity":
            aligned = source
        elif self.kind == "restriction":
            aligned = conservative_restrict(
                source,
                self.refinement_ratio,
                fine_cell_volumes=self.source_cell_volumes,
            )
        else:
            aligned = conservative_prolong(source, self.refinement_ratio)
        source_integral = _integral(source, self.source_cell_volumes)
        target_integral = _integral(aligned, self.target_cell_volumes)
        defect = jnp.max(jnp.abs(target_integral - source_integral), initial=0.0)
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(source_integral), initial=0.0))
        aligned = eqx.error_if(
            aligned,
            defect > self.plan.conservation_tolerance * scale,
            "Prepared grid alignment violated its conservation tolerance.",
        )
        return ConservativeAlignmentResult(
            values=aligned,
            source_integral=source_integral,
            target_integral=target_integral,
            alignment_id=self.prepared_id,
        )

    def align_snapshot(
        self, snapshot: ClosureSnapshot, /, *, target_mesh_id: str
    ) -> ClosureSnapshot:
        if not isinstance(snapshot, ClosureSnapshot):
            raise TypeError("snapshot must be a ClosureSnapshot.")
        result = self.execute(snapshot.values)
        mesh_id = str(target_mesh_id).strip()
        if not mesh_id:
            raise ValueError("target_mesh_id must be non-empty.")
        return ClosureSnapshot(
            result.values,
            snapshot.schema,
            time=snapshot.time,
            case_id=snapshot.case_id,
            trajectory_id=snapshot.trajectory_id,
            realization_id=snapshot.realization_id,
            time_block_id=snapshot.time_block_id,
            mesh_id=mesh_id,
            representation=snapshot.representation,
            parent_ids=(*snapshot.parent_ids, snapshot.snapshot_id, self.prepared_id),
        )


def conservative_restrict(
    fine_values: ArrayLike,
    refinement_ratio: tuple[int, ...],
    /,
    *,
    fine_cell_volumes: ArrayLike | None = None,
) -> Array:
    values = jnp.asarray(fine_values)
    ratio = tuple(int(value) for value in refinement_ratio)
    rank = len(ratio)
    if (
        not ratio
        or values.ndim < rank
        or any(value <= 0 for value in ratio)
        or any(values.shape[axis] % ratio[axis] != 0 for axis in range(rank))
    ):
        raise ValueError("Fine values and refinement ratio are incompatible.")
    fine_shape = tuple(values.shape[:rank])
    volumes = (
        jnp.ones(fine_shape, dtype=values.real.dtype)
        if fine_cell_volumes is None
        else _volumes(fine_cell_volumes, fine_shape).astype(values.real.dtype)
    )
    trailing = values.shape[rank:]
    reshape = tuple(
        item
        for size, factor in zip(fine_shape, ratio, strict=True)
        for item in (size // factor, factor)
    )
    reduction_axes = tuple(2 * axis + 1 for axis in range(rank))
    expanded_volumes = volumes.reshape(fine_shape + (1,) * len(trailing))
    numerator = jnp.sum(
        (values * expanded_volumes).reshape(reshape + trailing), axis=reduction_axes
    )
    denominator = jnp.sum(volumes.reshape(reshape), axis=reduction_axes)
    return numerator / denominator.reshape(denominator.shape + (1,) * len(trailing))


def conservative_prolong(
    coarse_values: ArrayLike, refinement_ratio: tuple[int, ...], /
) -> Array:
    values = jnp.asarray(coarse_values)
    ratio = tuple(int(value) for value in refinement_ratio)
    if not ratio or values.ndim < len(ratio) or any(value <= 0 for value in ratio):
        raise ValueError("Coarse values and refinement ratio are incompatible.")
    result = values
    for axis, factor in enumerate(ratio):
        result = jnp.repeat(result, factor, axis=axis)
    return result


def _block_sum(values: Array, ratio: tuple[int, ...]) -> Array:
    shape = tuple(values.shape)
    if all(shape[axis] % ratio[axis] == 0 for axis in range(len(ratio))):
        reshape = tuple(
            item
            for size, factor in zip(shape, ratio, strict=True)
            for item in (size // factor, factor)
        )
        return jnp.sum(
            values.reshape(reshape),
            axis=tuple(2 * axis + 1 for axis in range(len(ratio))),
        )
    raise ValueError("Cell volumes and refinement ratio are incompatible.")


def _volumes(values: ArrayLike | None, shape: tuple[int, ...]) -> Array:
    if values is None:
        return jnp.ones(shape, dtype=jnp.float64)
    volumes = jnp.asarray(values)
    if volumes.shape != shape:
        raise ValueError(f"Cell volumes must have shape {shape}; got {volumes.shape}.")
    host = np.asarray(volumes)
    if np.any(~np.isfinite(host)) or np.any(host <= 0.0):
        raise ValueError("Cell volumes must be finite and strictly positive.")
    return volumes


def _integral(values: Array, volumes: Array) -> Array:
    rank = volumes.ndim
    weighted = values * volumes.reshape(volumes.shape + (1,) * (values.ndim - rank))
    return jnp.sum(weighted, axis=tuple(range(rank)))


__all__ = [
    "AlignmentKind",
    "ConservativeAlignmentPlan",
    "ConservativeAlignmentResult",
    "PreparedConservativeAlignment",
    "conservative_prolong",
    "conservative_restrict",
]
