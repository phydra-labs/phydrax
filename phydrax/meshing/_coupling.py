#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum, StrEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._bvh import bvh_nearest_items, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    CellLocationStatus,
    PreparedSimplicialCellLocator,
    SimplicialLocationPolicy,
)
from ..discretization.fem import (
    discontinuous_element,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    PreparedFiniteElementCellMap,
)
from ._assembly import MeshPart
from ._result import CellMeshingResult
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


class CouplingSearchStatus(IntEnum):
    """Outcome of the donor search for one receptor entity."""

    FOUND = 0
    OUTSIDE = 1
    EXCLUDED_DONOR = 2
    AMBIGUOUS = 3
    NONFINITE = 4
    UNRESOLVED = 5


class CouplingSearchEvidence(StrictModule, NonTrainableState):
    """Per-receptor donor search outcome in target scope order.

    ``donor_ids`` are source-scope global IDs (-1 padding), ``source_cells``
    the containing source cell row (-1 for node searches or misses), and
    ``distances`` the donor distance (node search) or inverse-map residual
    (cell location). ``method`` names the search and its interpolation.
    """

    status: Array
    donor_ids: Array
    source_cells: Array
    distances: Array
    kind: MeshCouplingKind = eqx.field(static=True)
    method: str = eqx.field(static=True)
    source_scope_id: str = eqx.field(static=True)
    target_scope_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MeshCouplingKind,
        method: str,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        status: ArrayLike,
        donor_ids: ArrayLike,
        source_cells: ArrayLike,
        distances: ArrayLike,
        /,
    ) -> None:
        if not isinstance(kind, MeshCouplingKind):
            raise TypeError("kind must be MeshCouplingKind.")
        codes = np.asarray(status, dtype=np.int32)
        donors = np.asarray(donor_ids, dtype=np.int64)
        cells = np.asarray(source_cells, dtype=np.int32)
        distance = np.asarray(distances, dtype=np.float64)
        count = np.asarray(target_scope.entity_ids).size
        if (
            codes.shape != (count,)
            or donors.ndim != 2
            or donors.shape[0] != count
            or cells.shape != (count,)
            or distance.shape != (count,)
            or not np.all(np.isin(codes, [int(value) for value in CouplingSearchStatus]))
        ):
            raise ValueError(
                "Coupling search evidence must hold one record per receptor."
            )
        self.status = jnp.asarray(codes)
        self.donor_ids = jnp.asarray(donors)
        self.source_cells = jnp.asarray(cells)
        self.distances = jnp.asarray(distance)
        self.kind = kind
        self.method = str(method)
        self.source_scope_id = source_scope.scope_id
        self.target_scope_id = target_scope.scope_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "coupling-search-evidence",
                "coupling": kind.value,
                "method": self.method,
                "source": source_scope.scope_id,
                "target": target_scope.scope_id,
                "arrays": array_tree_fingerprint((codes, donors, cells, distance)),
            }
        )

    @property
    def found(self) -> Array:
        return self.status == int(CouplingSearchStatus.FOUND)


class CouplingSearchError(ValueError):
    """A donor search left receptors without admissible donors; see ``evidence``."""

    def __init__(self, message: str, evidence: CouplingSearchEvidence, /) -> None:
        super().__init__(message)
        self.evidence = evidence


def _search_failure(evidence: CouplingSearchEvidence, /) -> CouplingSearchError:
    counts = np.bincount(np.asarray(evidence.status), minlength=len(CouplingSearchStatus))
    summary = ", ".join(
        f"{status.name.lower()}={counts[status]}"
        for status in CouplingSearchStatus
        if status is not CouplingSearchStatus.FOUND and counts[status]
    )
    return CouplingSearchError(
        f"{evidence.kind.value.capitalize()} donor search failed for receptors "
        f"({summary}).",
        evidence,
    )


def _bound_search_evidence(
    evidence: CouplingSearchEvidence | None,
    kind: MeshCouplingKind,
    source_scope: MeshingScope,
    target_scope: MeshingScope,
    donors: np.ndarray,
    /,
) -> None:
    """Require supplied search evidence to describe exactly this coupling."""
    if evidence is None:
        return
    if not isinstance(evidence, CouplingSearchEvidence):
        raise TypeError("search_evidence must be CouplingSearchEvidence or None.")
    if (
        evidence.kind is not kind
        or evidence.source_scope_id != source_scope.scope_id
        or evidence.target_scope_id != target_scope.scope_id
        or not np.all(np.asarray(evidence.found))
        or not np.array_equal(np.asarray(evidence.donor_ids), donors)
    ):
        raise ValueError("Coupling search evidence does not describe these donors.")


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
) -> Any:
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
    ) -> None:
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
    ) -> None:
        tol = _tolerance(tolerance)
        rows, left, right = _point_pairs(
            source, target, source_scope, target_scope, source_ids
        )
        matrix, offset = (
            np.asarray(rotation, dtype=np.float64),
            np.asarray(translation, dtype=np.float64),
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

    def match_points(
        self, source_points: ArrayLike, target_points: ArrayLike, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pair two point sets (e.g. high-order geometry nodes) through the isometry.

        Returns ``(source_rows, target_rows)`` in ascending target order such that
        ``source @ rotation.T + translation`` lies within ``tolerance`` of its
        target. Raises unless the correspondence is a bijection.
        """
        left = np.asarray(source_points, dtype=np.float64)
        right = np.asarray(target_points, dtype=np.float64)
        dimension = self.rotation.shape[0]
        if (
            left.ndim != 2
            or right.ndim != 2
            or left.shape[1] != dimension
            or right.shape != left.shape
            or not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
        ):
            raise ValueError("Periodic point sets must be finite and equally sized.")
        if left.shape[0] == 0:
            empty = np.zeros((0,), dtype=np.int64)
            return empty, empty
        mapped = left @ np.asarray(self.rotation).T + np.asarray(self.translation)
        nearest = bvh_nearest_items(
            prepare_bvh(mapped, mapped, dtype=jnp.float64),
            jnp.asarray(right),
        )
        source_rows = np.asarray(nearest.items, dtype=np.int64)[:, 0]
        distances = np.linalg.norm(mapped[source_rows] - right, axis=1)
        if (
            np.any(distances > self.tolerance)
            or np.unique(source_rows).size != right.shape[0]
        ):
            raise ValueError("Periodic point sets do not correspond under the transform.")
        return source_rows, np.arange(right.shape[0], dtype=np.int64)


class ContactCoupling(_PointPairCoupling):
    """Frozen node-to-node, frictionless normal contact.

    Pairs are explicit or found once by :meth:`search` (exact BVH nearest-node
    matching); no collision detection runs during evaluation. Normals point
    from source to target; negative signed gaps mean penetration.
    Displacement-dependent gaps and equal/opposite penalty forces remain differentiable.
    """

    normals: Array
    reference_gap: Array
    search_evidence: CouplingSearchEvidence | None
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
        search_evidence: CouplingSearchEvidence | None = None,
    ) -> None:
        tol = _tolerance(tolerance)
        rows, left, right = _point_pairs(
            source, target, source_scope, target_scope, source_ids
        )
        normal = np.asarray(normals, dtype=np.float64)
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
        _bound_search_evidence(
            search_evidence,
            MeshCouplingKind.CONTACT,
            source_scope,
            target_scope,
            np.asarray(source_scope.entity_ids, dtype=np.int64)[rows][:, None],
        )
        self.source_scope, self.target_scope = source_scope, target_scope
        self.source_rows = jnp.asarray(rows)
        self.normals, self.reference_gap = jnp.asarray(normal), jnp.asarray(gap)
        self.search_evidence = search_evidence
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
                "search": None
                if search_evidence is None
                else search_evidence.evidence_id,
            }
        )

    @classmethod
    def search(
        cls,
        source: MeshPart,
        target: MeshPart,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        normals: ArrayLike,
        /,
        *,
        capture_radius: float,
        clearance: float = 0.0,
        tolerance: float = 1e-10,
    ) -> ContactCoupling:
        """Pair every target node with its exact nearest source node.

        Ties resolve to the lowest source global ID. Receptors beyond
        ``capture_radius``, with non-finite coordinates, or sharing a nearest
        donor raise :class:`CouplingSearchError` carrying per-receptor evidence.
        """
        evidence = _contact_donor_search(
            source, target, source_scope, target_scope, capture_radius
        )
        if not np.all(np.asarray(evidence.found)):
            raise _search_failure(evidence)
        return cls(
            source,
            target,
            source_scope,
            target_scope,
            normals,
            source_ids=np.asarray(evidence.donor_ids)[:, 0],
            clearance=clearance,
            tolerance=tolerance,
            search_evidence=evidence,
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

    This is interpolation, not a conservative overlap remap (``conservative``
    is always false). Donor rows contain global IDs from source_scope; -1
    padding has exactly zero weight. Stencils are explicit or found by
    :meth:`search`. Multiple donor-part overlays must have disjoint receptors.
    """

    donor_ids: Array
    donor_weights: Array
    donor_rows: Array
    hole_scope: MeshingScope | None
    search_evidence: CouplingSearchEvidence | None
    tolerance: float = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)

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
        search_evidence: CouplingSearchEvidence | None = None,
    ) -> None:
        _endpoints(source, target, source_scope, target_scope)
        tol = _tolerance(tolerance)
        if tol >= 1:
            raise ValueError("Overset normalization tolerance must be smaller than one.")
        donors, weights = (
            np.asarray(donor_ids),
            np.asarray(donor_weights, dtype=np.float64),
        )
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
        # Distinct negative sentinels per column keep padding out of the check.
        stencil = np.sort(
            np.where(valid, donors, -1 - np.arange(donors.shape[1])), axis=1
        )
        if np.any(stencil[:, 1:] == stencil[:, :-1]):
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
        _bound_search_evidence(
            search_evidence,
            MeshCouplingKind.OVERSET,
            source_scope,
            target_scope,
            donors.astype(np.int64),
        )
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
        self.search_evidence = search_evidence
        self.tolerance = tol
        self.conservative = False
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
                "search": None
                if search_evidence is None
                else search_evidence.evidence_id,
            }
        )

    @classmethod
    def search(
        cls,
        source: MeshPart,
        target: MeshPart,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        /,
        *,
        hole_scope: MeshingScope | None = None,
        location_policy: SimplicialLocationPolicy | None = None,
        tolerance: float = 1e-10,
    ) -> OversetCoupling:
        """Locate every receptor in the donor simplex mesh and interpolate linearly.

        Candidate cells come from the locator's BVH; weights are the clipped,
        renormalized barycentric coordinates of the containing cell (lowest
        cell index on shared faces). Donor vertices outside ``source_scope``
        or claimed as receptors/holes of the same entity set are excluded.
        Receptors without an admissible donor cell raise
        :class:`CouplingSearchError` carrying per-receptor evidence.
        """
        evidence, weights = _overset_donor_search(
            source, target, source_scope, target_scope, hole_scope, location_policy
        )
        if not np.all(np.asarray(evidence.found)):
            raise _search_failure(evidence)
        return cls(
            source,
            target,
            source_scope,
            target_scope,
            np.asarray(evidence.donor_ids),
            weights,
            hole_scope=hole_scope,
            tolerance=tolerance,
            search_evidence=evidence,
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


def _contact_donor_search(
    source: MeshPart,
    target: MeshPart,
    source_scope: MeshingScope,
    target_scope: MeshingScope,
    capture_radius: float,
    /,
) -> CouplingSearchEvidence:
    _endpoints(source, target, source_scope, target_scope)
    radius = float(capture_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Contact capture radius must be finite and positive.")
    if source_scope.entity_dimension != 0:
        raise ValueError("Node contact search pairs point entities.")
    source_ids = np.asarray(source_scope.entity_ids, dtype=np.int64)
    if source_ids.size != np.asarray(target_scope.entity_ids).size:
        raise ValueError("Node contact search requires equally sized point scopes.")
    donors = np.asarray(source.point_coordinates(source_scope), dtype=np.float64)
    receptors = np.asarray(target.point_coordinates(target_scope), dtype=np.float64)
    if not np.all(np.isfinite(donors)):
        raise ValueError("Contact donor points must be finite.")
    finite = np.all(np.isfinite(receptors), axis=1)
    nearest = bvh_nearest_items(
        prepare_bvh(donors, donors, dtype=jnp.float64),
        np.where(finite[:, None], receptors, 0.0),
        k=1,
    )
    items = np.asarray(nearest.items)[:, 0]
    distance = np.where(
        finite, np.sqrt(np.asarray(nearest.distance_squared)[:, 0]), np.inf
    )
    captured = finite & (distance <= radius)
    claims = np.bincount(items[captured], minlength=source_ids.size)
    status = np.select(
        (~finite, ~captured, claims[items] > 1),
        (
            int(CouplingSearchStatus.NONFINITE),
            int(CouplingSearchStatus.OUTSIDE),
            int(CouplingSearchStatus.AMBIGUOUS),
        ),
        int(CouplingSearchStatus.FOUND),
    )
    found = status == int(CouplingSearchStatus.FOUND)
    return CouplingSearchEvidence(
        MeshCouplingKind.CONTACT,
        "bvh-exact-nearest-node-pairing",
        source_scope,
        target_scope,
        status,
        np.where(found, source_ids[items], -1)[:, None],
        np.full(items.shape, -1, dtype=np.int32),
        distance,
    )


def _donor_cell_map(
    source: MeshPart, /
) -> tuple[PreparedFiniteElementCellMap, Array, np.ndarray]:
    """Affine simplex cell map of the donor part with vertex-aligned geometry."""
    carrier = source.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise TypeError("Overset donor search requires a certified cell donor part.")
    mesh = carrier.mesh
    if (
        len(mesh.blocks) != 1
        or mesh.blocks[0].cell_kind not in ("triangle", "tetrahedron")
        or mesh.topological_dimension != mesh.ambient_dimension
    ):
        raise ValueError(
            "Overset donor search requires one full-dimensional simplex block."
        )
    block = mesh.blocks[0]
    geometry = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec("donor", discontinuous_element(block.cell_kind, 0)),
        coordinate_spec=carrier.geometry,
    ).prepare()
    cell_map = PreparedFiniteElementCellMap(geometry, 0)
    coordinates = geometry.default_runtime.coordinates
    vertex_rows = np.asarray(cell_map.coordinate_dofs)
    if not np.array_equal(
        np.asarray(coordinates), np.asarray(mesh.coordinates)
    ) or not np.array_equal(vertex_rows, np.asarray(block.vertices)):
        raise ValueError(
            "Overset donor search requires vertex-aligned affine donor geometry."
        )
    return cell_map, coordinates, vertex_rows


def _overset_donor_search(
    source: MeshPart,
    target: MeshPart,
    source_scope: MeshingScope,
    target_scope: MeshingScope,
    hole_scope: MeshingScope | None,
    location_policy: SimplicialLocationPolicy | None,
    /,
) -> tuple[CouplingSearchEvidence, np.ndarray]:
    _endpoints(source, target, source_scope, target_scope)
    if source_scope.entity_dimension != 0:
        raise ValueError("Overset donor search interpolates point entities.")
    if hole_scope is not None:
        target.require_scope(hole_scope)
    cell_map, coordinates, vertex_rows = _donor_cell_map(source)
    policy = (
        SimplicialLocationPolicy(min(cell_map.cell_count, 64), 16, 1)
        if location_policy is None
        else location_policy
    )
    if not isinstance(policy, SimplicialLocationPolicy):
        raise TypeError("location_policy must be SimplicialLocationPolicy or None.")
    located = PreparedSimplicialCellLocator(cell_map, coordinates, policy).locate(
        target.point_coordinates(target_scope)
    )
    cells = np.asarray(located.cell_ids)
    inside = cells >= 0
    barycentric = np.where(
        inside[:, None], np.maximum(np.asarray(located.barycentric), 0.0), 0.0
    )
    total = np.sum(barycentric, axis=1, keepdims=True)
    weights = barycentric / np.where(total > 0.0, total, 1.0)
    # ty: ignore[unresolved-attribute]
    vertex_ids = np.asarray(source.carrier.mesh.vertex_global_ids, dtype=np.int64)
    donors = np.where(weights > 0.0, vertex_ids[vertex_rows[np.maximum(cells, 0)]], -1)
    admissible = np.isin(donors, np.asarray(source_scope.entity_ids))
    if (
        source.name == target.name
        and source_scope.entity_set_id == target_scope.entity_set_id
    ):
        forbidden = np.asarray(target_scope.entity_ids)
        if hole_scope is not None:
            forbidden = np.union1d(forbidden, np.asarray(hole_scope.entity_ids))
        admissible &= ~np.isin(donors, forbidden)
    blocked = np.any((weights > 0.0) & ~admissible, axis=1)
    location = np.asarray(located.status)
    status = np.select(
        (
            inside & ~blocked,
            inside,
            location == int(CellLocationStatus.OUTSIDE),
            location == int(CellLocationStatus.NONFINITE),
        ),
        (
            int(CouplingSearchStatus.FOUND),
            int(CouplingSearchStatus.EXCLUDED_DONOR),
            int(CouplingSearchStatus.OUTSIDE),
            int(CouplingSearchStatus.NONFINITE),
        ),
        int(CouplingSearchStatus.UNRESOLVED),
    )
    found = status == int(CouplingSearchStatus.FOUND)
    evidence = CouplingSearchEvidence(
        MeshCouplingKind.OVERSET,
        "bvh-simplicial-locator-barycentric-interpolation",
        source_scope,
        target_scope,
        status,
        np.where(found[:, None], donors, -1),
        np.where(inside, cells, -1),
        np.asarray(located.geometry_residual),
    )
    return evidence, np.where(found[:, None], weights, 0.0)


__all__ = [
    "ConformalCoupling",
    "ContactCoupling",
    "CouplingSearchError",
    "CouplingSearchEvidence",
    "CouplingSearchStatus",
    "MeshCoupling",
    "MeshCouplingKind",
    "OversetCoupling",
    "PeriodicCoupling",
]
