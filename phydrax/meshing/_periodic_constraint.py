#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np
from jax.typing import ArrayLike

from ..typing import checked
from ._controls import PeriodicConstraint


@checked
def bind_periodic_entity_pairs(
    constraint: PeriodicConstraint,
    source_entity_ids: ArrayLike,
    target_entity_ids: ArrayLike,
    target_to_source_ids: ArrayLike,
    /,
    *,
    source_coordinates: ArrayLike,
    target_coordinates: ArrayLike,
    orientations: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Bind an authored ID bijection and orientation witness under one isometry.

    Coordinate proximity never creates correspondence. IDs select the paired
    rows; coordinates only verify the declared transform against those rows.
    """
    if (
        constraint.source_scope.source_id != constraint.target_scope.source_id
        or constraint.source_scope.source_revision
        != constraint.target_scope.source_revision
        or constraint.source_scope.entity_kind != constraint.target_scope.entity_kind
        or constraint.source_scope.entity_dimension
        != constraint.target_scope.entity_dimension
    ):
        raise ValueError(
            "Periodic source and target scopes must share source identity and entity degree."
        )
    source_ids = np.asarray(source_entity_ids)
    target_ids = np.asarray(target_entity_ids)
    paired_ids = np.asarray(target_to_source_ids)
    if any(
        value.ndim != 1 or value.dtype.kind not in "iu"
        for value in (source_ids, target_ids, paired_ids)
    ):
        raise TypeError("Periodic entity IDs and pairing rows must be integer vectors.")
    source_ids = source_ids.astype(np.int64, copy=False)
    target_ids = target_ids.astype(np.int64, copy=False)
    paired_ids = paired_ids.astype(np.int64, copy=False)
    source_scope = np.asarray(constraint.source_scope.entity_ids, dtype=np.int64)
    target_scope = np.asarray(constraint.target_scope.entity_ids, dtype=np.int64)
    if (
        constraint.source_scope.entity_dimension
        != constraint.target_scope.entity_dimension
        or source_scope.size != target_scope.size
        or paired_ids.shape != target_ids.shape
        or not np.array_equal(np.sort(source_ids), np.sort(source_scope))
        or not np.array_equal(np.sort(target_ids), np.sort(target_scope))
        or not np.array_equal(np.sort(paired_ids), np.sort(source_scope))
        or np.unique(source_ids).size != source_ids.size
        or np.unique(target_ids).size != target_ids.size
        or np.unique(paired_ids).size != paired_ids.size
    ):
        raise ValueError(
            "Periodic pair IDs must provide a bijection of the two declared scopes."
        )
    source_order = np.argsort(source_ids, kind="stable")
    target_order = np.argsort(target_ids, kind="stable")
    source_sorted = source_ids[source_order]
    source_rows_for_target = np.searchsorted(source_sorted, paired_ids[target_order])
    if np.any(source_sorted[source_rows_for_target] != paired_ids[target_order]):
        raise ValueError("Periodic pair references an ID outside the source entity bank.")
    source_rows = source_order[source_rows_for_target]
    target_rows = target_order
    dimension = constraint.source_scope.entity_dimension
    if dimension == 0:
        if orientations is not None:
            raise ValueError(
                "Vertex periodic pairs do not take orientation coefficients."
            )
        signs = None
    else:
        if orientations is None:
            raise ValueError(
                "Edge and face periodic pairs require explicit ±1 orientations."
            )
        signs = np.asarray(orientations)
        if (
            signs.shape != target_scope.shape
            or signs.dtype.kind not in "iu"
            or np.any(~np.isin(signs, (-1, 1)))
        ):
            raise ValueError(
                "Periodic orientations must contain one ±1 value per target entity."
            )
        signs = signs[target_order].astype(np.int32, copy=False)
    source = np.asarray(source_coordinates, dtype=np.float64)
    target = np.asarray(target_coordinates, dtype=np.float64)
    matrix = np.asarray(constraint.transform, dtype=np.float64)
    ambient = matrix.shape[0] - 1
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1] != ambient
        or target.shape[1] != ambient
        or source.shape[0] != source_ids.size
        or target.shape[0] != target_ids.size
        or np.any(~np.isfinite(source))
        or np.any(~np.isfinite(target))
    ):
        raise ValueError(
            "Periodic source/target coordinate rows must match their ID banks and transform dimension."
        )
    predicted = source[source_rows] @ matrix[:-1, :-1].T + matrix[:-1, -1]
    residual = np.linalg.norm(predicted - target[target_rows], axis=1)
    if np.any(residual > constraint.tolerance):
        raise ValueError(
            "Authored periodic entity pairs violate the declared transform tolerance."
        )
    return source_rows.astype(np.int32), target_rows.astype(np.int32), signs


__all__ = ["bind_periodic_entity_pairs"]
