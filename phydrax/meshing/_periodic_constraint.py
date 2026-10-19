#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Periodic source-scope identifications and bound entity orbits."""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from . import _scope as scope_owner


class _BoundPeriodicOrbits(StrictModule):
    source_entity_ids: np.ndarray | None
    orientations: np.ndarray | None


def _bound_periodic_orbits(
    source_scope: scope_owner.MeshingScope,
    target_scope: scope_owner.MeshingScope,
    source_entity_ids: ArrayLike | None,
    orientations: ArrayLike | None,
    identity: bool,
    /,
) -> _BoundPeriodicOrbits:
    """Validate target-to-source entity orbits and their orientations."""
    if source_entity_ids is None:
        if orientations is not None:
            raise ValueError(
                "PeriodicConstraint.orientations must have explicit source_entity_ids; got None"
            )
        return _BoundPeriodicOrbits(None, None)
    targets = np.asarray(target_scope.entity_ids)
    paired = np.asarray(source_entity_ids)
    if not np.issubdtype(paired.dtype, np.integer) or paired.shape != targets.shape:
        raise ValueError(
            "PeriodicConstraint.source_entity_ids must hold one integer source entity per target entity; got incompatible shape or dtype"
        )
    paired = paired.astype(np.int64)
    if (
        source_scope.entity_dimension != target_scope.entity_dimension
        or not np.array_equal(np.sort(paired), np.asarray(source_scope.entity_ids))
    ):
        raise ValueError(
            "PeriodicConstraint.source_entity_ids must pair equal-dimension scopes bijectively; got incompatible entity orbits"
        )
    if (
        not identity
        and source_scope.scope_id == target_scope.scope_id
        and np.any(paired == targets)
    ):
        raise ValueError(
            "PeriodicConstraint.source_entity_ids must avoid self-images under a nonidentity transform; got unchanged target entities"
        )
    if source_scope.entity_dimension == 0:
        if orientations is not None:
            raise ValueError(
                "PeriodicConstraint.orientations must be absent for point orbits; got supplied orientations"
            )
        return _BoundPeriodicOrbits(paired, None)
    if orientations is None:
        raise ValueError(
            "PeriodicConstraint.orientations must provide orientation witnesses for edge and face orbits; got None"
        )
    signs = np.asarray(orientations)
    if (
        not np.issubdtype(signs.dtype, np.integer)
        or signs.shape != targets.shape
        or np.any(np.abs(signs) != 1)
    ):
        raise ValueError(
            "PeriodicConstraint.orientations must be ±1 per bound pair; got incompatible shape, dtype, or signs"
        )
    return _BoundPeriodicOrbits(paired, signs.astype(np.int32))


class PeriodicConstraint(StrictModule, NonTrainableState):
    """Identify two source scopes under one homogeneous Euclidean isometry.

    ``transform`` maps source to target coordinates. ``source_entity_ids``
    optionally binds one source entity per target entity, in target order;
    edges and faces require ``orientations`` of ±1. Unbound constraints leave
    the pairing to the provider's geometric correspondence.
    """

    source_scope: scope_owner.MeshingScope
    target_scope: scope_owner.MeshingScope
    transform: Array
    source_entity_ids: Array | None
    orientations: Array | None
    tolerance: float = eqx.field(static=True)
    conforming_required: bool = eqx.field(static=True)
    orientation_preserving: bool = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_scope: scope_owner.MeshingScope,
        target_scope: scope_owner.MeshingScope,
        transform: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-10,
        conforming_required: bool = True,
        source_entity_ids: ArrayLike | None = None,
        orientations: ArrayLike | None = None,
    ) -> None:
        if source_scope.source_revision != target_scope.source_revision:
            raise ValueError(
                "PeriodicConstraint.scopes must share one source revision; got different revisions"
            )
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
            raise ValueError(
                "PeriodicConstraint.transform must be one square homogeneous matrix; got incompatible matrix shape"
            )
        if not np.all(np.isfinite(matrix)) or not np.allclose(
            matrix[-1],
            np.eye(matrix.shape[0], dtype=np.float64)[-1],
        ):
            raise ValueError(
                "PeriodicConstraint.transform must be finite and homogeneous; got nonfinite entries or an invalid final row"
            )
        linear = matrix[:-1, :-1]
        determinant = np.linalg.det(linear)
        if abs(determinant) <= np.finfo(np.float64).eps:
            raise ValueError(
                "PeriodicConstraint.transform must be invertible; got a singular linear block"
            )
        # Isometry tolerance covers rounding of trigonometric matrix entries.
        if not np.allclose(
            linear.T @ linear,
            np.eye(linear.shape[0], dtype=np.float64),
            rtol=0.0,
            atol=128.0 * np.finfo(np.float64).eps,
        ):
            raise ValueError(
                "PeriodicConstraint.transform must be a Euclidean isometry; got a nonorthonormal linear block"
            )
        threshold = float(tolerance)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "PeriodicConstraint.tolerance must be finite and nonnegative; got an invalid tolerance"
            )
        orbits = _bound_periodic_orbits(
            source_scope,
            target_scope,
            source_entity_ids,
            orientations,
            bool(np.array_equal(matrix, np.eye(matrix.shape[0], dtype=np.float64))),
        )
        paired, signs = orbits.source_entity_ids, orbits.orientations
        self.source_scope = source_scope
        self.target_scope = target_scope
        self.transform = jax.device_put(np.asarray(matrix))
        self.source_entity_ids = (
            None if paired is None else jax.device_put(np.asarray(paired))
        )
        self.orientations = None if signs is None else jax.device_put(np.asarray(signs))
        self.tolerance = threshold
        self.conforming_required = bool(conforming_required)
        self.orientation_preserving = bool(determinant > 0.0)
        identity = {
            "kind": "periodic-meshing-constraint",
            "source_scope": source_scope.scope_id,
            "target_scope": target_scope.scope_id,
            "transform": array_tree_fingerprint(matrix),
            "tolerance": threshold,
            "conforming_required": bool(conforming_required),
        }
        if paired is not None:
            identity["orbits"] = array_tree_fingerprint(
                {"source_entity_ids": paired, "orientations": signs}
            )
        self.constraint_id = canonical_fingerprint(identity)


__all__ = ["PeriodicConstraint"]
