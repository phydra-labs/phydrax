#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._assembly import MeshPart
from ._scope import MeshingScope


class MeshCouplingKind(StrEnum):
    CONFORMAL = "conformal"
    PERIODIC = "periodic"
    CONTACT = "contact"
    OVERSET = "overset"


def _tolerance(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("Coupling tolerance must be finite and positive.")
    return result


def _endpoints(
    source: MeshPart,
    target: MeshPart,
    source_scope: MeshingScope,
    target_scope: MeshingScope,
) -> None:
    if not isinstance(source, MeshPart) or not isinstance(target, MeshPart):
        raise TypeError("Coupling endpoints must be MeshPart values.")
    source.require_scope(source_scope)
    target.require_scope(target_scope)
    if (
        source.coordinate_contract.spatial_id != target.coordinate_contract.spatial_id
        or source.ambient_dimension != target.ambient_dimension
    ):
        raise ValueError(
            "Coupling endpoints require one coordinate contract and ambient dimension."
        )
    if source.name == target.name and source.part_id != target.part_id:
        raise ValueError("Coupling cannot mix revisions of one part.")
    if source_scope.entity_dimension != target_scope.entity_dimension:
        raise ValueError("Coupling endpoints must have matching entity dimensions.")


def _point_pairs(
    source: MeshPart,
    target: MeshPart,
    source_scope: MeshingScope,
    target_scope: MeshingScope,
    source_ids: ArrayLike | None,
):
    _endpoints(source, target, source_scope, target_scope)
    ids = np.asarray(source_scope.entity_ids)
    paired = ids if source_ids is None else np.asarray(source_ids)
    if paired.shape != np.asarray(target_scope.entity_ids).shape or not np.issubdtype(
        paired.dtype, np.integer
    ):
        raise ValueError("One integer source point ID is required per target point.")
    if not np.array_equal(np.sort(paired), ids):
        raise ValueError(
            "Conformal, periodic and node-contact pairs must be a complete bijection."
        )
    rows = np.searchsorted(ids, paired).astype(np.int32)
    return (
        rows,
        np.asarray(source.point_coordinates(source_scope))[rows],
        np.asarray(target.point_coordinates(target_scope)),
    )


class MeshCoupling(StrictModule, NonTrainableState):
    """Exact endpoint-scoped coupling; field arrays use sorted scope-global-ID order."""

    __strict_abstract__ = True

    source_scope: MeshingScope
    target_scope: MeshingScope
    kind: MeshCouplingKind = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)

    def require_current(self, source: MeshPart, target: MeshPart, /) -> None:
        _endpoints(source, target, self.source_scope, self.target_scope)

    @abstractmethod
    def transfer(self, source_values: ArrayLike, /) -> Array:
        """Evaluate this overlay's source-to-target field trace."""
        raise NotImplementedError


class _PointPairCoupling(MeshCoupling):
    __strict_abstract__ = True

    source_rows: Array

    def transfer(self, source_values: ArrayLike, /) -> Array:
        values = jnp.asarray(source_values)
        if values.ndim == 0 or values.shape[0] != self.source_scope.entity_ids.size:
            raise ValueError("Source field must follow source scope entity order.")
        return values[self.source_rows]

    def transpose(self, target_values: ArrayLike, /) -> Array:
        values = jnp.asarray(target_values)
        if values.ndim == 0 or values.shape[0] != self.target_scope.entity_ids.size:
            raise ValueError("Target field must follow target scope entity order.")
        return (
            jnp.zeros(
                (self.source_scope.entity_ids.size,) + values.shape[1:],
                dtype=values.dtype,
            )
            .at[self.source_rows]
            .add(values)
        )


class ConformalCoupling(_PointPairCoupling):
    """A geometry-checked point bijection; topology remains owned by each part."""

    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        source: MeshPart,
        target: MeshPart,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        /,
        *,
        source_ids: ArrayLike | None = None,
        tolerance: float = 1e-10,
    ):
        tol = _tolerance(tolerance)
        rows, left, right = _point_pairs(
            source, target, source_scope, target_scope, source_ids
        )
        if (
            not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
            or np.any(np.linalg.norm(left - right, axis=-1) > tol)
        ):
            raise ValueError("Conformal point pairs do not coincide within tolerance.")
        self.source_scope, self.target_scope = source_scope, target_scope
        self.source_rows = jnp.asarray(rows)
        self.kind = MeshCouplingKind.CONFORMAL
        self.tolerance = tol
        self.coupling_id = canonical_fingerprint(
            {
                "kind": self.kind.value,
                "source": source_scope.scope_id,
                "target": target_scope.scope_id,
                "rows": array_tree_fingerprint(rows),
                "tolerance": tol,
            }
        )


class PeriodicCoupling(_PointPairCoupling):
    """Point bijection under an explicitly checked Euclidean isometry."""

    rotation: Array
    translation: Array
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        source: MeshPart,
        target: MeshPart,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        rotation: ArrayLike,
        translation: ArrayLike,
        /,
        *,
        source_ids: ArrayLike | None = None,
        tolerance: float = 1e-10,
    ):
        tol = _tolerance(tolerance)
        rows, left, right = _point_pairs(
            source, target, source_scope, target_scope, source_ids
        )
        matrix, offset = (
            np.asarray(rotation, dtype=float),
            np.asarray(translation, dtype=float),
        )
        dimension = source.ambient_dimension
        if (
            matrix.shape != (dimension, dimension)
            or offset.shape != (dimension,)
            or not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(offset))
        ):
            raise ValueError("Periodic transform shape or finite values are invalid.")
        if not np.allclose(matrix.T @ matrix, np.eye(dimension), rtol=0, atol=tol):
            raise ValueError("Periodic transform must be an isometry.")
        if (
            not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
            or np.any(np.linalg.norm(left @ matrix.T + offset - right, axis=-1) > tol)
        ):
            raise ValueError("Periodic point pairs do not match the transform.")
        self.source_scope, self.target_scope = source_scope, target_scope
        self.source_rows = jnp.asarray(rows)
        self.rotation, self.translation = jnp.asarray(matrix), jnp.asarray(offset)
        self.kind = MeshCouplingKind.PERIODIC
        self.tolerance = tol
        self.coupling_id = canonical_fingerprint(
            {
                "kind": self.kind.value,
                "source": source_scope.scope_id,
                "target": target_scope.scope_id,
                "rows": array_tree_fingerprint(rows),
                "rotation": array_tree_fingerprint(matrix),
                "translation": array_tree_fingerprint(offset),
                "tolerance": tol,
            }
        )

    def transfer_vectors(self, source_values: ArrayLike, /) -> Array:
        values = self.transfer(source_values)
        if values.shape[-1] != self.rotation.shape[0]:
            raise ValueError("Periodic vectors must match the ambient dimension.")
        return values @ self.rotation.T


class ContactCoupling(_PointPairCoupling):
    """Frozen node-to-node, frictionless normal contact (not a collision search).

    Normals point from source to target; negative signed gaps mean penetration.
    Displacement-dependent gaps and equal/opposite penalty forces remain differentiable.
    """

    normals: Array
    reference_gap: Array
    clearance: float = eqx.field(static=True)

    def __init__(
        self,
        source: MeshPart,
        target: MeshPart,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        normals: ArrayLike,
        /,
        *,
        source_ids: ArrayLike | None = None,
        clearance: float = 0.0,
        tolerance: float = 1e-10,
    ):
        tol = _tolerance(tolerance)
        rows, left, right = _point_pairs(
            source, target, source_scope, target_scope, source_ids
        )
        normal = np.asarray(normals, dtype=float)
        distance = float(clearance)
        if (
            normal.shape != right.shape
            or not np.all(np.isfinite(normal))
            or not np.allclose(np.linalg.norm(normal, axis=-1), 1.0, rtol=0, atol=tol)
        ):
            raise ValueError("Contact requires one finite unit normal per point pair.")
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("Contact clearance must be finite and non-negative.")
        gap = np.sum((right - left) * normal, axis=-1) - distance
        if not np.all(np.isfinite(gap)):
            raise ValueError("Contact point geometry must be finite.")
        self.source_scope, self.target_scope = source_scope, target_scope
        self.source_rows = jnp.asarray(rows)
        self.normals, self.reference_gap = jnp.asarray(normal), jnp.asarray(gap)
        self.clearance = distance
        self.kind = MeshCouplingKind.CONTACT
        self.coupling_id = canonical_fingerprint(
            {
                "kind": self.kind.value,
                "source": source_scope.scope_id,
                "target": target_scope.scope_id,
                "rows": array_tree_fingerprint(rows),
                "normals": array_tree_fingerprint(normal),
                "gap": array_tree_fingerprint(gap),
                "clearance": distance,
            }
        )

    def gaps(
        self, source_displacement: ArrayLike, target_displacement: ArrayLike, /
    ) -> Array:
        left, right = self.transfer(source_displacement), jnp.asarray(target_displacement)
        if left.shape != self.normals.shape or right.shape != self.normals.shape:
            raise ValueError("Contact displacements must match point-normal shape.")
        return self.reference_gap + jnp.sum((right - left) * self.normals, axis=-1)

    def penalty_forces(
        self,
        source_displacement: ArrayLike,
        target_displacement: ArrayLike,
        stiffness: float,
        /,
    ) -> tuple[Array, Array]:
        stiffness_ = float(stiffness)
        if not np.isfinite(stiffness_) or stiffness_ <= 0:
            raise ValueError("Contact penalty stiffness must be finite and positive.")
        gap = self.gaps(source_displacement, target_displacement)
        target_force = (stiffness_ * jnp.maximum(-gap, 0))[:, None] * self.normals
        return -self.transpose(target_force), target_force


class OversetCoupling(MeshCoupling):
    """Positive partition-of-unity donor stencil, with explicit receptor/hole roles.

    This is interpolation, not a claim of conservative overlap remapping. Donor
    rows contain global IDs from source_scope; -1 padding has exactly zero weight.
    Multiple donor-part overlays must have disjoint receptor ownership.
    """

    donor_ids: Array
    donor_weights: Array
    donor_rows: Array
    hole_scope: MeshingScope | None
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        source: MeshPart,
        target: MeshPart,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        donor_ids: ArrayLike,
        donor_weights: ArrayLike,
        /,
        *,
        hole_scope: MeshingScope | None = None,
        tolerance: float = 1e-10,
    ):
        _endpoints(source, target, source_scope, target_scope)
        tol = _tolerance(tolerance)
        if tol >= 1:
            raise ValueError("Overset normalization tolerance must be smaller than one.")
        donors, weights = np.asarray(donor_ids), np.asarray(donor_weights, dtype=float)
        if (
            donors.ndim != 2
            or donors.shape[0] != target_scope.entity_ids.size
            or donors.shape[1] == 0
            or not np.issubdtype(donors.dtype, np.integer)
            or weights.shape != donors.shape
        ):
            raise ValueError(
                "Overset donors and weights must have shape (receptors, stencil_width)."
            )
        valid = donors >= 0
        source_ids = np.asarray(source_scope.entity_ids)
        if (
            np.any(donors < -1)
            or not np.all(np.isin(donors[valid], source_ids))
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0)
            or np.any(weights[~valid] != 0)
        ):
            raise ValueError(
                "Overset donor IDs, padding or non-negative weights are invalid."
            )
        if np.any(np.abs(np.sum(weights, axis=1) - 1.0) > tol):
            raise ValueError(
                "Every overset receptor must have donor weights summing to one."
            )
        for row, row_valid in zip(donors, valid, strict=True):
            if np.unique(row[row_valid]).size != np.count_nonzero(row_valid):
                raise ValueError("A receptor stencil cannot repeat a donor entity.")
        if hole_scope is not None:
            target.require_scope(hole_scope)
            if (
                hole_scope.entity_set_id != target_scope.entity_set_id
                or hole_scope.entity_dimension != target_scope.entity_dimension
                or np.intersect1d(
                    np.asarray(hole_scope.entity_ids), np.asarray(target_scope.entity_ids)
                ).size
            ):
                raise ValueError(
                    "Overset holes must be disjoint from receptors in the same entity set."
                )
        if (
            source.name == target.name
            and source_scope.entity_set_id == target_scope.entity_set_id
        ):
            forbidden = np.asarray(target_scope.entity_ids)
            if hole_scope is not None:
                forbidden = np.union1d(forbidden, np.asarray(hole_scope.entity_ids))
            if np.any(np.isin(donors[valid], forbidden)):
                raise ValueError("Overset receptor or hole entities cannot be donors.")
        rows = np.where(
            valid, np.searchsorted(source_ids, np.maximum(donors, 0)), 0
        ).astype(np.int32)
        self.source_scope, self.target_scope = source_scope, target_scope
        self.donor_ids, self.donor_weights, self.donor_rows = (
            jnp.asarray(donors, dtype=jnp.int64),
            jnp.asarray(weights),
            jnp.asarray(rows),
        )
        self.hole_scope = hole_scope
        self.tolerance = tol
        self.kind = MeshCouplingKind.OVERSET
        self.coupling_id = canonical_fingerprint(
            {
                "kind": self.kind.value,
                "source": source_scope.scope_id,
                "target": target_scope.scope_id,
                "donors": array_tree_fingerprint(donors.astype(np.int64)),
                "weights": array_tree_fingerprint(weights),
                "holes": None if hole_scope is None else hole_scope.scope_id,
                "tolerance": tol,
            }
        )

    def transfer(self, source_values: ArrayLike, /) -> Array:
        values = jnp.asarray(source_values)
        if values.ndim == 0 or values.shape[0] != self.source_scope.entity_ids.size:
            raise ValueError("Source field must follow source scope entity order.")
        weights = self.donor_weights.reshape(
            self.donor_weights.shape + (1,) * (values.ndim - 1)
        )
        gathered = jnp.where(weights > 0, values[self.donor_rows], 0)
        return jnp.sum(gathered * weights, axis=1)

    def transpose(self, target_values: ArrayLike, /) -> Array:
        values = jnp.asarray(target_values)
        if values.ndim == 0 or values.shape[0] != self.target_scope.entity_ids.size:
            raise ValueError("Target field must follow target scope entity order.")
        weights = self.donor_weights.reshape(
            self.donor_weights.shape + (1,) * (values.ndim - 1)
        )
        weighted = jnp.where(weights > 0, values[:, None, ...], 0) * weights
        shape = (self.source_scope.entity_ids.size,) + values.shape[1:]
        return jnp.zeros(shape, dtype=weighted.dtype).at[self.donor_rows].add(weighted)


__all__ = [
    "ConformalCoupling",
    "ContactCoupling",
    "MeshCoupling",
    "MeshCouplingKind",
    "OversetCoupling",
    "PeriodicCoupling",
]
