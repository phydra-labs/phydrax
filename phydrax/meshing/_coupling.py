#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum, StrEnum
from fractions import Fraction
from typing import Any, Literal, TypeAlias

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
    CellMesh,
    PreparedSimplicialCellLocator,
    SimplicialLocationPolicy,
)
from ..discretization._field_query import PreparedFieldQuery
from ..discretization._periodic_topology import _lifted_loops
from ..discretization.fem import (
    discontinuous_element,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    PreparedFiniteElementCellMap,
)
from ..ein import contract
from ..typing import parse
from ._assembly import MeshPart
from ._result import CellMeshingResult
from ._scope import MeshingScope


OversetValueAction: TypeAlias = Literal[
    "invariant", "polar-vector", "contravariant-vector"
]


def _encoded_image_isometry(matrix: np.ndarray, /) -> bool:
    """Prove Euclidean Gram identity of the original binary64 coefficients."""
    dimension = matrix.shape[0]
    coefficients = tuple(tuple(Fraction(float(value)) for value in row) for row in matrix)
    return all(
        sum(
            (
                coefficients[row][first] * coefficients[row][second]
                for row in range(dimension)
            ),
            Fraction(0),
        )
        == Fraction(1 if first == second else 0)
        for first in range(dimension)
        for second in range(dimension)
    )


def _query_image_action(
    query: PreparedFieldQuery,
    target_points: Array,
    dimension: int,
    rotation: ArrayLike | None,
    translation: ArrayLike | None,
    source_image_rotation: ArrayLike | None,
    target_image_rotation: ArrayLike | None,
    action: OversetValueAction,
    tolerance: float,
    /,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if (rotation is None) != (translation is None):
        raise ValueError("Overset isometries require both rotation and translation.")
    images = tuple(
        None if image is None else np.asarray(image, dtype=np.float64)
        for image in (source_image_rotation, target_image_rotation)
    )
    for image in images:
        if image is not None:
            if image.shape != (dimension, dimension) or not np.all(np.isfinite(image)):
                raise ValueError(
                    "Original source image maps must retain finite ambient coordinate axes."
                )
            if action == "polar-vector" and not _encoded_image_isometry(image):
                raise ValueError(
                    "Polar-vector actions require exact encoded source-image isometries; use contravariant-vector for an affine map."
                )
    matrix, offset = None, None
    points = np.asarray(query.points)
    target_sites = np.asarray(target_points)
    if rotation is not None and translation is not None:
        matrix, offset = (
            np.asarray(rotation, dtype=np.float64),
            np.asarray(translation, dtype=np.float64),
        )
        if (
            matrix.shape != (dimension, dimension)
            or offset.shape != (dimension,)
            or not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(offset))
        ):
            raise ValueError("Overset transform shape or finite values are invalid.")
        if not np.allclose(
            matrix.T @ matrix, np.eye(dimension, dtype=np.float64), rtol=0, atol=tolerance
        ):
            raise ValueError(
                "Overset transform must satisfy its near-isometry admission premise."
            )
        if action == "polar-vector" and not _encoded_image_isometry(matrix):
            raise ValueError(
                "Polar-vector actions require an exact encoded isometry; use contravariant-vector for an affine map."
            )
        matched = np.allclose(
            points @ matrix.T + offset, target_sites, rtol=0, atol=tolerance
        )
    else:
        matched = np.array_equal(points, target_sites)
    if matrix is not None or any(image is not None for image in images):
        if any(query.derivative):
            raise ValueError(
                "Overset images require value queries, not derivative components."
            )
        if action != "invariant" and query.value_shape != (dimension,):
            raise ValueError(f"Overset images require ambient {action} value queries.")
    if not matched:
        raise ValueError(
            "Field-query points must follow the target vertex scope order under its image map."
        )
    return matrix, offset, images[0], images[1]


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
    RESOURCE_EXCEEDED = 6


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


def _paired_rows(
    source_scope: MeshingScope,
    target_scope: MeshingScope,
    source_ids: ArrayLike | None,
    /,
) -> np.ndarray:
    """Source-scope rows paired with target-scope entities, in target order."""

    ids = np.asarray(source_scope.entity_ids)
    paired = ids if source_ids is None else np.asarray(source_ids)
    if paired.shape != np.asarray(target_scope.entity_ids).shape or not np.issubdtype(
        paired.dtype, np.integer
    ):
        raise ValueError("One integer source entity ID is required per target entity.")
    if not np.array_equal(np.sort(paired), ids):
        raise ValueError("Periodic entity pairs must be a complete bijection.")
    return np.searchsorted(ids, paired).astype(np.int32)


def _cell_mesh(part: MeshPart, /) -> CellMesh:
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise TypeError("Periodic entity pairs require certified cell mesh parts.")
    return carrier.mesh


def _lifted_rows(mesh: CellMesh, scope: MeshingScope, /) -> np.ndarray:
    identifiers = np.asarray(mesh.entity_set(scope.entity_dimension).entity_ids)
    order = np.argsort(identifiers, kind="stable")
    return order[np.searchsorted(identifiers[order], np.asarray(scope.entity_ids))]


def _oriented_corners(mesh: CellMesh, dimension: int, /) -> dict[int, np.ndarray]:
    """Oriented corner loop of every lifted edge or face, by lifted row."""

    return {
        int(row): loop
        for rows, loops in _lifted_loops(mesh, dimension)
        for row, loop in zip(rows, loops, strict=True)
    }


def _loop_orientations(
    mapped: np.ndarray, target: np.ndarray, tolerance: float, /
) -> np.ndarray:
    """Orientation witnesses of mapped source loops against target loops."""

    arity = target.shape[1]
    distances = np.linalg.norm(target[:, :, None, :] - mapped[:, None, :, :], axis=-1)
    matched = np.argmin(distances, axis=2)
    if np.any(np.min(distances, axis=2) > tolerance) or np.any(
        np.sort(matched, axis=1) != np.arange(arity)
    ):
        raise ValueError("Periodic entity corners do not match the transform.")
    if arity == 2:
        return np.where(matched[:, 0] == 0, 1, -1).astype(np.int32)
    step = (np.roll(matched, -1, axis=1) - matched) % arity
    forward = np.all(step == 1, axis=1)
    backward = np.all(step == arity - 1, axis=1)
    if not np.all(forward | backward):
        raise ValueError("Periodic entity corners are not an oriented loop image.")
    return np.where(forward, 1, -1).astype(np.int32)


def _require_distinct_quotient_orbits(
    source: MeshPart,
    target: MeshPart,
    dimension: int,
    source_scope: MeshingScope,
    target_scope: MeshingScope,
    rows: np.ndarray,
    /,
) -> None:
    """Keep boundary pairing distinct from identifications a quotient already owns."""

    if source.part_id != target.part_id:
        return
    mesh = _cell_mesh(source)
    periodic = mesh.periodic_topology
    if periodic is None:
        return
    orbit = np.asarray(periodic.orbits(dimension)[0])
    source_rows = _lifted_rows(mesh, source_scope)[rows]
    target_rows = _lifted_rows(mesh, target_scope)
    if np.any(orbit[source_rows] == orbit[target_rows]):
        raise ValueError(
            "Periodic coupling pairs entities that the quotient topology already "
            "identifies; boundary-paired coupling cannot re-identify quotient orbits."
        )


def _entity_pairs(
    source: MeshPart,
    target: MeshPart,
    source_scope: MeshingScope,
    target_scope: MeshingScope,
    source_ids: ArrayLike | None,
    matrix: np.ndarray,
    offset: np.ndarray,
    tolerance: float,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair oriented entities through the isometry and return their witnesses."""

    dimension = source_scope.entity_dimension
    if dimension == 0:
        rows, left, right = _point_pairs(
            source, target, source_scope, target_scope, source_ids
        )
        if (
            not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
            or np.any(
                np.linalg.norm(left @ matrix.T + offset - right, axis=-1) > tolerance
            )
        ):
            raise ValueError("Periodic point pairs do not match the transform.")
        if isinstance(source.carrier, CellMeshingResult):
            _require_distinct_quotient_orbits(
                source, target, 0, source_scope, target_scope, rows
            )
        return rows, np.ones(rows.shape, dtype=np.int32)
    _endpoints(source, target, source_scope, target_scope)
    rows = _paired_rows(source_scope, target_scope, source_ids)
    source_mesh, target_mesh = _cell_mesh(source), _cell_mesh(target)
    if dimension >= min(
        source_mesh.topological_dimension, target_mesh.topological_dimension
    ):
        raise ValueError(
            "Periodic coupling pairs vertices, edges or faces below the cell dimension."
        )
    source_loops = _oriented_corners(source_mesh, dimension)
    target_loops = _oriented_corners(target_mesh, dimension)
    source_corners = [
        source_loops[int(row)] for row in _lifted_rows(source_mesh, source_scope)[rows]
    ]
    target_corners = [
        target_loops[int(row)] for row in _lifted_rows(target_mesh, target_scope)
    ]
    arities = np.asarray([loop.size for loop in target_corners])
    if np.any(arities != [loop.size for loop in source_corners]):
        raise ValueError("Periodic entity pairs must have equal corner counts.")
    source_points = np.asarray(source_mesh.coordinates, dtype=np.float64)
    target_points = np.asarray(target_mesh.coordinates, dtype=np.float64)
    orientations = np.empty(rows.shape, dtype=np.int32)
    for arity in np.unique(arities):
        selected = np.flatnonzero(arities == arity)
        mapped = (
            source_points[np.stack([source_corners[i] for i in selected])] @ matrix.T
            + offset
        )
        orientations[selected] = _loop_orientations(
            mapped,
            target_points[np.stack([target_corners[i] for i in selected])],
            tolerance,
        )
    _require_distinct_quotient_orbits(
        source, target, dimension, source_scope, target_scope, rows
    )
    return rows, orientations


class PeriodicCoupling(_PointPairCoupling):
    """Boundary-paired entity bijection under an explicitly checked isometry.

    Pairs are explicit entity orbits between two boundary-paired carriers:
    ``source_rows`` names the source-scope entity of each target entity and
    ``orientations`` the ``±1`` witness of its oriented corner loop under the
    isometry (always ``+1`` for points). A quotient mesh already identifies its
    seam entities, so pairs inside one quotient orbit are refused.
    """

    rotation: Array
    translation: Array
    orientations: Array
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
        rows, orientations = _entity_pairs(
            source, target, source_scope, target_scope, source_ids, matrix, offset, tol
        )
        self.source_scope, self.target_scope = source_scope, target_scope
        self.source_rows = jnp.asarray(rows)
        self.rotation, self.translation = jnp.asarray(matrix), jnp.asarray(offset)
        self.orientations = jnp.asarray(orientations)
        self.kind = MeshCouplingKind.PERIODIC
        self.tolerance = tol
        identity = {
            "kind": self.kind.value,
            "source": source_scope.scope_id,
            "target": target_scope.scope_id,
            "rows": array_tree_fingerprint(rows),
            "rotation": array_tree_fingerprint(matrix),
            "translation": array_tree_fingerprint(offset),
            "tolerance": tol,
        }
        if source_scope.entity_dimension > 0:
            identity["orientations"] = array_tree_fingerprint(orientations)
        self.coupling_id = canonical_fingerprint(identity)

    def transfer_oriented(self, source_values: ArrayLike, /) -> Array:
        """Transfer oriented entity values (circulations, fluxes) with witnesses."""

        values = self.transfer(source_values)
        signs = self.orientations.astype(values.dtype)
        return values * signs.reshape(signs.shape + (1,) * (values.ndim - 1))

    def transfer_tensors(self, source_values: ArrayLike, /) -> Array:
        """Transfer rank-two tensor values as ``R A R^T``."""

        values = self.transfer(source_values)
        dimension = self.rotation.shape[0]
        if values.shape[-2:] != (dimension, dimension):
            raise ValueError("Periodic tensors must be square in the ambient dimension.")
        rotation = self.rotation.astype(values.dtype)
        return contract("ij,...jk,lk->...il", rotation, values, rotation, backend="jax")

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
    tolerance: float = eqx.field(static=True)

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
        self.tolerance = tol
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
    """Explicit vertex stencils or an owning canonical field-query route.

    Query mode scopes source support *cells*, not coefficients: coefficients
    retain the query's full true field layout, including signed/Piola weights.
    An explicit query-mode map binds donor query points to target points.
    ``value_action`` declares invariant values, polar vectors under exact encoded
    source isometries, or contravariant components under authored affine maps.
    Vector transposes reverse the declared linear action before query transpose.
    Neither mode is a conservative overlap remap.
    """

    donor_ids: Array
    donor_weights: Array | None
    donor_rows: Array | None
    field_query: PreparedFieldQuery | None
    rotation: Array | None
    translation: Array | None
    source_image_rotation: Array | None
    target_image_rotation: Array | None
    hole_scope: MeshingScope | None
    search_evidence: CouplingSearchEvidence | None
    tolerance: float = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    value_action: OversetValueAction = eqx.field(static=True)

    def __init__(
        self,
        source: MeshPart,
        target: MeshPart,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        donor_ids: ArrayLike | None = None,
        donor_weights: ArrayLike | None = None,
        /,
        *,
        hole_scope: MeshingScope | None = None,
        tolerance: float = 1e-10,
        search_evidence: CouplingSearchEvidence | None = None,
        field_query: PreparedFieldQuery | None = None,
        rotation: ArrayLike | None = None,
        translation: ArrayLike | None = None,
        value_action: OversetValueAction = "polar-vector",
        source_image_rotation: ArrayLike | None = None,
        target_image_rotation: ArrayLike | None = None,
    ) -> None:
        if field_query is not None:
            if (
                donor_ids is not None
                or donor_weights is not None
                or search_evidence is not None
            ):
                raise ValueError(
                    "Field queries and explicit vertex donor stencils are distinct modes."
                )
            self._bind_field_query(
                source,
                target,
                source_scope,
                target_scope,
                field_query,
                hole_scope,
                rotation,
                translation,
                tolerance,
                value_action,
                source_image_rotation,
                target_image_rotation,
            )
            return
        if any(
            image is not None
            for image in (
                rotation,
                translation,
                source_image_rotation,
                target_image_rotation,
            )
        ):
            raise ValueError("Overset isometries require an actual field-query route.")
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
        self.field_query = None
        self.rotation, self.translation = None, None
        self.source_image_rotation, self.target_image_rotation = None, None
        self.value_action = parse(value_action, OversetValueAction, "value_action")
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
    def from_field_query(
        cls,
        source: MeshPart,
        target: MeshPart,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        query: PreparedFieldQuery,
        /,
        *,
        hole_scope: MeshingScope | None = None,
        rotation: ArrayLike | None = None,
        translation: ArrayLike | None = None,
        tolerance: float = 1e-10,
        value_action: OversetValueAction = "polar-vector",
        source_image_rotation: ArrayLike | None = None,
        target_image_rotation: ArrayLike | None = None,
    ) -> OversetCoupling:
        """Bind actual field evidence and an optional ambient-vector isometry."""
        return cls(
            source,
            target,
            source_scope,
            target_scope,
            field_query=query,
            hole_scope=hole_scope,
            rotation=rotation,
            translation=translation,
            tolerance=tolerance,
            value_action=value_action,
            source_image_rotation=source_image_rotation,
            target_image_rotation=target_image_rotation,
        )

    def _bind_field_query(
        self,
        source: MeshPart,
        target: MeshPart,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        query: PreparedFieldQuery,
        hole_scope: MeshingScope | None,
        /,
        rotation: ArrayLike | None,
        translation: ArrayLike | None,
        tolerance: float,
        value_action: OversetValueAction,
        source_image_rotation: ArrayLike | None,
        target_image_rotation: ArrayLike | None,
    ) -> None:
        from ..discretization.fem import FiniteElementDiscretization

        if not isinstance(query, PreparedFieldQuery) or not query.complete:
            raise TypeError("Query-mode overset requires a complete PreparedFieldQuery.")
        if not isinstance(source, MeshPart) or not isinstance(target, MeshPart):
            raise TypeError("Coupling endpoints must be MeshPart values.")
        source.require_scope(source_scope)
        target.require_scope(target_scope)
        if (
            source.coordinate_contract.spatial_id != target.coordinate_contract.spatial_id
            or source.ambient_dimension != target.ambient_dimension
            or source_scope.entity_dimension != source.intrinsic_dimension
            or target_scope.entity_dimension != 0
            or (source.name == target.name and source.part_id != target.part_id)
        ):
            raise ValueError(
                "Field-query endpoints require source cells and target vertices."
            )
        tol = _tolerance(tolerance)
        action = parse(value_action, OversetValueAction, "value_action")
        matrix, offset, source_image, target_image = _query_image_action(
            query,
            target.point_coordinates(target_scope),
            source.ambient_dimension,
            rotation,
            translation,
            source_image_rotation,
            target_image_rotation,
            action,
            tol,
        )
        kernel = query.reconstruction.kernel
        owner = getattr(kernel, "discretization", None)
        locator = getattr(kernel, "locator", None)
        if (
            owner is None
            or locator is None
            or not isinstance(source.carrier, CellMeshingResult)
        ):
            raise ValueError(
                "Field-query donor must retain its actual mesh and coordinate owner."
            )
        mesh = source.carrier.mesh
        if owner.mesh.mesh_id != mesh.mesh_id:
            raise ValueError(
                "Field-query donor is bound to another source mesh revision."
            )
        if isinstance(owner, FiniteElementDiscretization):
            coordinates = source.carrier.geometry.coordinates
        else:
            geometry_binding = getattr(kernel, "require_source_geometry", None)
            if geometry_binding is None:
                raise TypeError(
                    "Field-query owner must retain its actual FV coordinate binding."
                )
            geometry_binding(source.carrier.geometry)
            owning_geometry = getattr(owner, "cell_geometry", None)
            coordinates = (
                mesh.coordinates
                if owning_geometry is None
                else owning_geometry.coordinates
            )
        if not np.array_equal(np.asarray(locator.coordinates), np.asarray(coordinates)):
            raise ValueError(
                "Field-query donor is bound to another source coordinate revision."
            )
        if isinstance(owner, FiniteElementDiscretization):
            if (
                owner.mesh.mesh_id != mesh.mesh_id
                or owner.default_runtime.geometry_layout_id
                != source.carrier.geometry.geometry_layout_id
            ):
                raise ValueError("Field-query donor has a stale FE mesh/geometry layout.")
            selected = getattr(locator, "selected_cells", None)
            if selected is None:
                location = locator.locate(query.points)
                selected = np.asarray(location.candidate_cells)
                selected = selected[selected >= 0]
            support_ids = np.asarray(mesh.block(locator.cell_map.block_name).global_ids)[
                np.unique(selected)
            ]
        else:
            support_factory = getattr(kernel, "query_support_rows", None)
            if support_factory is None:
                raise TypeError(
                    "Field-query owner must publish its actual reconstruction support cells."
                )
            identifiers = np.concatenate(
                [np.asarray(block.global_ids) for block in mesh.blocks]
            )
            support_ids = identifiers[support_factory(query)]
        if not np.array_equal(
            np.sort(np.unique(support_ids)), np.asarray(source_scope.entity_ids)
        ):
            raise ValueError("Source scope must retain the query's actual support cells.")
        if hole_scope is not None:
            target.require_scope(hole_scope)
            if (
                hole_scope.entity_dimension != 0
                or hole_scope.entity_set_id != target_scope.entity_set_id
                or np.intersect1d(hole_scope.entity_ids, target_scope.entity_ids).size
            ):
                raise ValueError("Field-query holes must be disjoint target vertices.")
        self.source_scope, self.target_scope = source_scope, target_scope
        self.donor_ids = source_scope.entity_ids[None, :]
        self.donor_weights, self.donor_rows = None, None
        self.field_query = query
        self.rotation = None if matrix is None else jnp.asarray(matrix)
        self.translation = None if offset is None else jnp.asarray(offset)
        self.source_image_rotation = (
            None if source_image is None else jnp.asarray(source_image)
        )
        self.target_image_rotation = (
            None if target_image is None else jnp.asarray(target_image)
        )
        self.hole_scope, self.search_evidence = hole_scope, None
        self.tolerance, self.conservative = tol, False
        self.value_action = action
        self.kind = MeshCouplingKind.OVERSET
        self.coupling_id = canonical_fingerprint(
            {
                "kind": "overset-field-query",
                "source": source_scope.scope_id,
                "target": target_scope.scope_id,
                "query": query.query_id,
                "holes": None if hole_scope is None else hole_scope.scope_id,
                "rotation": None if matrix is None else array_tree_fingerprint(matrix),
                "translation": None if offset is None else array_tree_fingerprint(offset),
                "tolerance": tol,
                "value_action": action,
                "source_images": array_tree_fingerprint((source_image, target_image)),
            }
        )

    @property
    def support_cell_scope(self) -> MeshingScope | None:
        """Actual query-read geometric cell identities, never coefficient DOF IDs.

        The field owner validates this scope against its complete selected-cell
        and reconstruction-stencil support before query-mode publication.
        Coefficients retain the independent PreparedFieldQuery field layout.
        Vertex-stencil mode has no field-query support-cell scope.
        """
        return self.source_scope if self.field_query is not None else None

    def require_current(self, source: MeshPart, target: MeshPart, /) -> None:
        if self.field_query is None:
            super().require_current(source, target)
        else:
            source.require_scope(self.source_scope)
            target.require_scope(self.target_scope)

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
        if self.field_query is not None:
            values = self.field_query.apply(source_values)
            if self.rotation is not None and self.value_action != "invariant":
                values = values @ self.rotation.astype(values.dtype).T
            return values
        values = jnp.asarray(source_values)
        if values.ndim == 0 or values.shape[0] != self.source_scope.entity_ids.size:
            raise ValueError("Source field must follow source scope entity order.")
        if self.donor_weights is None or self.donor_rows is None:
            raise RuntimeError(
                "An overset stencil route requires prepared donor weights and rows."
            )
        weights = self.donor_weights.reshape(
            self.donor_weights.shape + (1,) * (values.ndim - 1)
        )
        gathered = jnp.where(weights > 0, values[self.donor_rows], 0)
        return jnp.sum(gathered * weights, axis=1)

    def transpose(self, target_values: ArrayLike, /) -> Array:
        if self.field_query is not None:
            values = jnp.asarray(target_values)
            if values.shape != self.field_query.output_shape:
                raise ValueError("Target field must follow the field query output shape.")
            if self.rotation is not None and self.value_action != "invariant":
                values = values @ self.rotation.astype(values.dtype)
            return self.field_query.transpose(values)
        values = jnp.asarray(target_values)
        if values.ndim == 0 or values.shape[0] != self.target_scope.entity_ids.size:
            raise ValueError("Target field must follow target scope entity order.")
        if self.donor_weights is None or self.donor_rows is None:
            raise RuntimeError(
                "An overset stencil route requires prepared donor weights and rows."
            )
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
            location == int(CellLocationStatus.RESOURCE_EXCEEDED),
        ),
        (
            int(CouplingSearchStatus.FOUND),
            int(CouplingSearchStatus.EXCLUDED_DONOR),
            int(CouplingSearchStatus.OUTSIDE),
            int(CouplingSearchStatus.NONFINITE),
            int(CouplingSearchStatus.RESOURCE_EXCEEDED),
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
