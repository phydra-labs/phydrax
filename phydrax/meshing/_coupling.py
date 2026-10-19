#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum, StrEnum
from fractions import Fraction
from typing import assert_never, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._simplicial_locator as simplicial_locator_owner

from .._bvh import bvh_nearest_items, prepare_bvh
from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    tree_content_equal,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_finite_float
from ..discretization._field_query import PreparedFieldQuery
from ..discretization._periodic_topology import lifted_loops
from ..discretization.fem import (
    discontinuous_element,
    FiniteElementFieldReconstructionKernel,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    PreparedFiniteElementCellMap,
)
from ..discretization.finite_volume._field_view import (
    UnstructuredFiniteVolumeFieldReconstructionKernel,
)
from ..discretization.finite_volume._triangle_fv import TriangleFiniteVolumeDiscretization
from ..discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
)
from ..ein import contract
from ..typing import checked, parse
from ._mesh_part import MeshPart
from ._result import CellMeshingResult
from ._scope import MeshingScope


type OversetValueAction = Literal["invariant", "polar-vector", "contravariant-vector"]


def encoded_image_isometry(matrix: np.ndarray, /) -> bool:
    """Proves the Euclidean Gram identity using rational arithmetic over the encoded
    binary64 coefficients.
    """
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


@final
class _QueryImageAction(StrictModule, NonTrainableState):
    """Validated source-to-target image action and authored source images."""

    matrix: np.ndarray | None
    offset: np.ndarray | None
    source_image: np.ndarray | None
    target_image: np.ndarray | None


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
) -> _QueryImageAction:
    if (rotation is None) != (translation is None):
        raise ValueError(
            "OversetCoupling.rotation must be provided together with translation or both must be None; got only one image component."
        )
    images = tuple(
        None if image is None else np.asarray(image, dtype=np.float64)
        for image in (source_image_rotation, target_image_rotation)
    )
    for image in images:
        if image is not None:
            if image.shape != (dimension, dimension) or not np.all(np.isfinite(image)):
                raise ValueError(
                    "OversetCoupling.source_image_rotation must contain finite ambient-coordinate matrices; got an incompatible shape or nonfinite coefficient."
                )
            if action == "polar-vector" and not encoded_image_isometry(image):
                raise ValueError(
                    "OversetCoupling.source_image_rotation must pass encoded_image_isometry on encoded coefficients for a polar-vector action; got coefficients failing the isometry predicate."
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
            raise ValueError(
                "OversetCoupling.rotation must have ambient-dimensional finite coefficients together with translation; got invalid shapes or non-finite coefficients."
            )
        if not np.allclose(
            matrix.T @ matrix, np.eye(dimension, dtype=np.float64), rtol=0, atol=tolerance
        ):
            raise ValueError(
                "OversetCoupling.rotation must satisfy the Euclidean Gram identity within tolerance; got a matrix outside the near-isometry tolerance."
            )
        if action == "polar-vector" and not encoded_image_isometry(matrix):
            raise ValueError(
                "OversetCoupling.rotation must pass encoded_image_isometry on encoded coefficients for a polar-vector action; got coefficients failing the isometry predicate."
            )
        matched = np.allclose(
            points @ matrix.T + offset, target_sites, rtol=0, atol=tolerance
        )
    else:
        matched = np.array_equal(points, target_sites)
    if matrix is not None or any(image is not None for image in images):
        if any(query.derivative):
            raise ValueError(
                "OversetCoupling.field_query.derivative must select value components only; got a derivative component request."
            )
        if action != "invariant" and query.value_shape != (dimension,):
            raise ValueError(
                f"OversetCoupling.field_query.value_shape must be an ambient-dimension vector shape for {action}; got a non-vector query shape."
            )
    if not matched:
        raise ValueError(
            "OversetCoupling.field_query.points must follow target vertex scope order under the image map; got a different point order or image."
        )
    return _QueryImageAction(matrix, offset, images[0], images[1])


class MeshCouplingKind(StrEnum):
    CONFORMAL = "conformal"
    PERIODIC = "periodic"
    CONTACT = "contact"
    OVERSET = "overset"


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

    @checked
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
            raise TypeError(
                "CouplingSearchEvidence.kind must be a MeshCouplingKind; got another type."
            )
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
                "CouplingSearchEvidence.records must provide one status code, donor row, cell, and distance per receptor with every code in CouplingSearchStatus; got mismatched record arrays or an unsupported status code."
            )
        self.status = jax.device_put(np.asarray(codes))
        self.donor_ids = jax.device_put(np.asarray(donors))
        self.source_cells = jax.device_put(np.asarray(cells))
        self.distances = jax.device_put(np.asarray(distance))
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

    @checked
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
        f"CouplingSearchEvidence.status must find admissible donors for every receptor in {evidence.kind.value.capitalize()} search; got {summary}.",
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
    if (
        evidence.kind is not kind
        or evidence.source_scope_id != source_scope.scope_id
        or evidence.target_scope_id != target_scope.scope_id
        or not np.all(np.asarray(evidence.found))
        or not np.array_equal(np.asarray(evidence.donor_ids), donors)
    ):
        raise ValueError(
            "CouplingSearchEvidence.binding must match the requested kind, source scope, target scope, and donor IDs with every receptor found; got stale or unresolved search evidence."
        )


def _endpoints(
    source: MeshPart,
    target: MeshPart,
    source_scope: MeshingScope,
    target_scope: MeshingScope,
) -> None:
    source.require_scope(source_scope)
    target.require_scope(target_scope)
    if (
        source.coordinate_contract.spatial_id != target.coordinate_contract.spatial_id
        or source.ambient_dimension != target.ambient_dimension
    ):
        raise ValueError(
            "MeshCoupling.endpoints must share a coordinate-contract spatial ID and ambient dimension; got mismatched endpoint coordinates or dimensions."
        )
    if source.name == target.name and source.part_id != target.part_id:
        raise ValueError(
            "MeshCoupling.endpoints must use one part revision when endpoint names match; got matching names with different part IDs."
        )
    if source_scope.entity_dimension != target_scope.entity_dimension:
        raise ValueError(
            "MeshCoupling.endpoint_scopes must have matching entity dimensions; got different source and target entity dimensions."
        )


def _point_pairs(
    source: MeshPart,
    target: MeshPart,
    source_scope: MeshingScope,
    target_scope: MeshingScope,
    source_ids: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _endpoints(source, target, source_scope, target_scope)
    ids = np.asarray(source_scope.entity_ids)
    paired = ids if source_ids is None else np.asarray(source_ids)
    if paired.shape != np.asarray(target_scope.entity_ids).shape or not np.issubdtype(
        paired.dtype, np.integer
    ):
        raise ValueError(
            "MeshCoupling.source_point_ids must provide one integer ID per target point; got a shape or dtype mismatch."
        )
    if not np.array_equal(np.sort(paired), ids):
        raise ValueError(
            "MeshCoupling.point_pairs must form a bijection covering every source and target point; got source IDs that are not a bijection."
        )
    rows = np.searchsorted(ids, paired).astype(np.int32)
    return (
        rows,
        np.asarray(source.point_coordinates(source_scope))[rows],
        np.asarray(target.point_coordinates(target_scope)),
    )


class MeshCoupling(StrictModule, NonTrainableState):
    """Represents endpoint-scoped coupling with field arrays in sorted scope-global-ID
    order.
    """

    __strict_abstract__ = True

    source_scope: MeshingScope
    target_scope: MeshingScope
    kind: MeshCouplingKind = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)

    @checked
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
            raise ValueError(
                "MeshCoupling.source_values must have one leading-axis value per source-scope entity; got a scalar or mismatched leading axis."
            )
        return values[self.source_rows]

    def transpose(self, target_values: ArrayLike, /) -> Array:
        values = jnp.asarray(target_values)
        if values.ndim == 0 or values.shape[0] != self.target_scope.entity_ids.size:
            raise ValueError(
                "MeshCoupling.target_values must have one leading-axis value per target-scope entity; got a scalar or mismatched leading axis."
            )
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

    @checked
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
        tol = positive_finite_float(tolerance, "tolerance")
        rows, left, right = _point_pairs(
            source, target, source_scope, target_scope, source_ids
        )
        if (
            not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
            or np.any(np.linalg.norm(left - right, axis=-1) > tol)
        ):
            raise ValueError(
                "ConformalCoupling.point_pairs must have finite coordinates that coincide within tolerance; got non-finite or separated paired points."
            )
        self.source_scope, self.target_scope = source_scope, target_scope
        self.source_rows = jax.device_put(np.asarray(rows))
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
        raise ValueError(
            "PeriodicCoupling.source_entity_ids must provide one integer ID per target entity; got a shape or dtype mismatch."
        )
    if not np.array_equal(np.sort(paired), ids):
        raise ValueError(
            "PeriodicCoupling.entity_pairs must form a bijection covering every source and target entity; got source IDs that are not a bijection."
        )
    return np.searchsorted(ids, paired).astype(np.int32)


def _cell_mesh(part: MeshPart, /) -> cell_mesh_owner.CellMesh:
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise TypeError(
            "PeriodicCoupling.endpoint.carrier must be a certified CellMeshingResult; got a carrier without a certified cell mesh."
        )
    return carrier.mesh


def _lifted_rows(mesh: cell_mesh_owner.CellMesh, scope: MeshingScope, /) -> np.ndarray:
    identifiers = np.asarray(mesh.entity_set(scope.entity_dimension).entity_ids)
    order = np.argsort(identifiers, kind="stable")
    return order[np.searchsorted(identifiers[order], np.asarray(scope.entity_ids))]


def _oriented_corners(
    mesh: cell_mesh_owner.CellMesh, dimension: int, /
) -> dict[int, np.ndarray]:
    """Oriented corner loop of every lifted edge or face, by lifted row."""

    return {
        int(row): loop
        for rows, loops in lifted_loops(mesh, dimension)
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
        raise ValueError(
            "PeriodicCoupling.entity_pair_corners must match the transform within tolerance as a corner bijection; got unmatched or repeated transformed corners."
        )
    if arity == 2:
        return np.where(matched[:, 0] == 0, 1, -1).astype(np.int32)
    step = (np.roll(matched, -1, axis=1) - matched) % arity
    forward = np.all(step == 1, axis=1)
    backward = np.all(step == arity - 1, axis=1)
    if not np.all(forward | backward):
        raise ValueError(
            "PeriodicCoupling.entity_pair_corners must form a forward or backward oriented loop image; got neither loop orientation."
        )
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
            "PeriodicCoupling.entity_pairs must belong to distinct quotient orbits; got entities already identified by quotient topology."
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
            raise ValueError(
                "PeriodicCoupling.point_pairs must have finite coordinates matching the transform within tolerance; got non-finite or mismatched paired points."
            )
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
            "PeriodicCoupling.entity_dimension must be below both endpoint cell dimensions; got a cell-dimension entity request."
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
        raise ValueError(
            "PeriodicCoupling.entity_pairs must have matching corner counts; got different source and target arities."
        )
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
    """Bind a boundary-paired entity bijection under the declared isometry tolerance.

    Pairs are explicit entity orbits between two boundary-paired carriers.
    ``source_rows`` names the source-scope entity for every target entity, and
    ``orientations`` gives the ``±1`` sign of its oriented corner loop under the
    isometry; the sign is always ``+1`` for points. Pairs within one quotient orbit are
    refused because a quotient mesh already identifies its seam entities.
    """

    rotation: Array
    translation: Array
    orientations: Array
    tolerance: float = eqx.field(static=True)

    @checked
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
        tol = positive_finite_float(tolerance, "tolerance")
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
            raise ValueError(
                "PeriodicCoupling.rotation must have ambient-dimensional finite coefficients together with translation; got invalid shapes or non-finite coefficients."
            )
        if not np.allclose(matrix.T @ matrix, np.eye(dimension), rtol=0, atol=tol):
            raise ValueError(
                "PeriodicCoupling.rotation must satisfy the Euclidean Gram identity within tolerance; got a matrix that is not an admitted isometry."
            )
        rows, orientations = _entity_pairs(
            source, target, source_scope, target_scope, source_ids, matrix, offset, tol
        )
        self.source_scope, self.target_scope = source_scope, target_scope
        self.source_rows = jax.device_put(np.asarray(rows))
        self.rotation, self.translation = (
            jax.device_put(np.asarray(matrix)),
            jax.device_put(np.asarray(offset)),
        )
        self.orientations = jax.device_put(np.asarray(orientations))
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
            raise ValueError(
                "PeriodicCoupling.transfer_tensors.values must have two trailing ambient-dimension axes; got a non-square or dimension-mismatched tensor shape."
            )
        rotation = self.rotation.astype(values.dtype)
        return contract("ij,...jk,lk->...il", rotation, values, rotation, backend="jax")

    def transfer_vectors(self, source_values: ArrayLike, /) -> Array:
        values = self.transfer(source_values)
        if values.shape[-1] != self.rotation.shape[0]:
            raise ValueError(
                "PeriodicCoupling.transfer_vectors.values must have an ambient-dimension trailing axis; got a mismatched vector dimension."
            )
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
            raise ValueError(
                "PeriodicCoupling.match_points.inputs must be finite two-dimensional point arrays of equal shape in the ambient dimension; got invalid ranks, shapes, dimensions, or coordinates."
            )
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
            raise ValueError(
                "PeriodicCoupling.match_points.correspondence must be a bijection under the transform within tolerance; got an unmatched point or repeated source match."
            )
        return source_rows, np.arange(right.shape[0], dtype=np.int64)


class ContactCoupling(_PointPairCoupling):
    """Represents frozen node-to-node, frictionless normal contact.

    Pairs are explicit or are found once by :meth:`search` using BVH nearest-node
    matching; collision detection does not run during evaluation. Normals point from
    source to target, and negative signed gaps mean penetration. Displacement-dependent
    gaps and equal-and-opposite penalty forces remain differentiable.
    """

    normals: Array
    reference_gap: Array
    search_evidence: CouplingSearchEvidence | None
    clearance: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    @checked
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
        tol = positive_finite_float(tolerance, "tolerance")
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
            raise ValueError(
                "ContactCoupling.normals must provide one finite unit normal per point pair; got a shape mismatch, non-finite component, or non-unit norm."
            )
        if not np.isfinite(distance) or distance < 0:
            raise ValueError(
                "ContactCoupling.clearance must be finite and non-negative; got a non-finite or negative value."
            )
        gap = np.sum((right - left) * normal, axis=-1) - distance
        if not np.all(np.isfinite(gap)):
            raise ValueError(
                "ContactCoupling.point_geometry must yield finite normal gaps; got non-finite gap values."
            )
        _bound_search_evidence(
            search_evidence,
            MeshCouplingKind.CONTACT,
            source_scope,
            target_scope,
            np.asarray(source_scope.entity_ids, dtype=np.int64)[rows][:, None],
        )
        self.source_scope, self.target_scope = source_scope, target_scope
        self.source_rows = jax.device_put(np.asarray(rows))
        self.normals, self.reference_gap = (
            jax.device_put(np.asarray(normal)),
            jax.device_put(np.asarray(gap)),
        )
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
    @checked
    def from_stored_fields(
        cls: type[ContactCoupling],
        value: ContactCoupling,
        /,
        *,
        source: MeshPart,
        target: MeshPart,
    ) -> ContactCoupling:
        """Revalidate stored contact fields against retained endpoint revisions."""
        if cls is not ContactCoupling or type(value) is not ContactCoupling:
            raise TypeError(
                "ContactCoupling.restore_type must be the registered ContactCoupling; "
                f"got {type(value).__qualname__}."
            )
        rows = np.asarray(value.source_rows)
        identifiers = np.asarray(value.source_scope.entity_ids)
        if (
            rows.shape != (value.target_scope.entity_ids.size,)
            or not np.issubdtype(rows.dtype, np.integer)
            or np.any(rows < 0)
            or np.any(rows >= identifiers.size)
        ):
            raise ValueError(
                "ContactCoupling.source_rows must index its source scope once per target entity; "
                "got invalid row shape, dtype, or range."
            )
        rebuilt = cls(
            source,
            target,
            value.source_scope,
            value.target_scope,
            value.normals,
            source_ids=identifiers[rows],
            clearance=value.clearance,
            tolerance=value.tolerance,
            search_evidence=value.search_evidence,
        )
        if not tree_content_equal(rebuilt, value):
            raise ValueError(
                "ContactCoupling.stored_fields must match reconstructed contact state; "
                "got changed scopes, rows, normals, gap, evidence, or identity."
            )
        return rebuilt

    @classmethod
    @checked
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
        """Pairs every target node with its nearest source node using the bounded BVH
        search.

        Ties resolve to the lowest source global ID. Receptors beyond
        ``capture_radius``, receptors with non-finite coordinates, and receptors sharing
        a nearest donor raise :class:`CouplingSearchError` with evidence for every
        affected receptor.
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
            raise ValueError(
                "ContactCoupling.gaps.displacements must match the point-normal shape; got a left or right displacement shape mismatch."
            )
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
            raise ValueError(
                "ContactCoupling.penalty_stiffness must be finite and positive; got a non-finite or non-positive value."
            )
        gap = self.gaps(source_displacement, target_displacement)
        target_force = (stiffness_ * jnp.maximum(-gap, 0))[:, None] * self.normals
        return -self.transpose(target_force), target_force


class OversetCoupling(MeshCoupling):
    """Represents explicit vertex stencils or a field-query route tied to its field
    provider.

    Query mode scopes source support *cells*, not coefficients. Coefficients retain the
    query's full field layout, including signed/Piola weights. An explicit query-mode
    map binds every donor query point to a target point. ``value_action`` declares
    invariant values, polar vectors transformed by source isometries checked through
    rational arithmetic over encoded coefficients, or contravariant components
    transformed by source-bound coordinate maps under authored affine maps. Vector
    transposes reverse the declared linear action before the query transpose. Both modes
    perform interpolation; conservation uses a separate overlap remap.
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

    @checked
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
                    "OversetCoupling.mode must select either a field query or an explicit vertex donor stencil; got both modes at once."
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
            raise ValueError(
                "OversetCoupling.image_transforms must accompany a field-query route; got image transforms on a route without a field query."
            )
        _endpoints(source, target, source_scope, target_scope)
        tol = positive_finite_float(tolerance, "tolerance")
        if tol >= 1:
            raise ValueError(
                "OversetCoupling.normalization_tolerance must be smaller than one; got a value at least one."
            )
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
                "OversetCoupling.donor_ids must form a receptor-by-stencil integer matrix paired with donor_weights and nonzero stencil width; got an invalid rank, shape, dtype, or empty stencil."
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
                "OversetCoupling.donor_ids must use source IDs or -1 padding, paired with finite non-negative weights and zero padding weights; got invalid donor IDs, weights, or padding."
            )
        if np.any(np.abs(np.sum(weights, axis=1) - 1.0) > tol):
            raise ValueError(
                "OversetCoupling.donor_weights must provide non-negative weights summing to one for every receptor; got a receptor sum outside normalization tolerance."
            )
        # Distinct negative sentinels per column keep padding out of the check.
        stencil = np.sort(
            np.where(valid, donors, -1 - np.arange(donors.shape[1])), axis=1
        )
        if np.any(stencil[:, 1:] == stencil[:, :-1]):
            raise ValueError(
                "OversetCoupling.donor_ids must be distinct within each receptor stencil; got a repeated donor entity."
            )
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
                    "OversetCoupling.hole_scope must share the receptor entity set and dimension while remaining disjoint from receptors; got a scope mismatch or receptor overlap refused."
                )
        if (
            source.name == target.name
            and source_scope.entity_set_id == target_scope.entity_set_id
        ):
            forbidden = np.asarray(target_scope.entity_ids)
            if hole_scope is not None:
                forbidden = np.union1d(forbidden, np.asarray(hole_scope.entity_ids))
            if np.any(np.isin(donors[valid], forbidden)):
                raise ValueError(
                    "OversetCoupling.donor_ids must exclude receptor and hole entities; got forbidden donor overlap refused."
                )
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
            jax.device_put(np.asarray(donors, dtype=np.int64)),
            jax.device_put(np.asarray(weights)),
            jax.device_put(np.asarray(rows)),
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
    @checked
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
        """Bind field evidence and an optional authored image map."""
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

        if not query.complete:
            raise TypeError(
                "OversetCoupling.field_query must admit every requested point; got an incomplete field query."
            )
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
                "OversetCoupling.field_query.endpoints must share a coordinate-contract spatial ID and ambient dimension, use source cells and target vertices, and use one part revision per name; got incompatible endpoint scopes or parts."
            )
        tol = positive_finite_float(tolerance, "tolerance")
        action = parse(value_action, OversetValueAction, "value_action")
        image = _query_image_action(
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
        matrix, offset = image.matrix, image.offset
        source_image, target_image = image.source_image, image.target_image
        kernel = query.reconstruction.kernel
        match kernel:
            case (
                FiniteElementFieldReconstructionKernel()
                | UnstructuredFiniteVolumeFieldReconstructionKernel()
            ):
                owner, locator = kernel.discretization, kernel.locator
            case _:
                raise ValueError(
                    "OversetCoupling.field_query must retain its donor mesh and coordinate owner; got an unsupported kernel or carrier kind."
                )
        match source.carrier:
            case CellMeshingResult() as carrier:
                mesh = carrier.mesh
            case _:
                raise ValueError(
                    "OversetCoupling.field_query must retain its donor mesh and coordinate owner; got an unsupported kernel or carrier kind."
                )
        if owner.mesh.mesh_id != mesh.mesh_id:
            raise ValueError(
                "OversetCoupling.field_query must bind its source mesh revision; got a different donor mesh."
            )
        match kernel:
            case FiniteElementFieldReconstructionKernel():
                coordinates = carrier.geometry.coordinates
            case UnstructuredFiniteVolumeFieldReconstructionKernel():
                kernel.require_source_geometry(carrier.geometry)
                match kernel.discretization:
                    case UnstructuredFiniteVolumeDiscretization() as fv:
                        geometry = fv.cell_geometry
                        coordinates = (
                            mesh.coordinates if geometry is None else geometry.coordinates
                        )
                    case TriangleFiniteVolumeDiscretization():
                        coordinates = mesh.coordinates
                    case _ as unknown:
                        assert_never(unknown)
            case _ as unknown:
                assert_never(unknown)
        if not np.array_equal(np.asarray(locator.coordinates), np.asarray(coordinates)):
            raise ValueError(
                "OversetCoupling.field_query must bind its source coordinate revision; got a different donor coordinate map."
            )
        match kernel:
            case FiniteElementFieldReconstructionKernel():
                if (
                    kernel.discretization.mesh.mesh_id != mesh.mesh_id
                    or kernel.discretization.default_runtime.geometry_layout_id
                    != carrier.geometry.geometry_layout_id
                ):
                    raise ValueError(
                        "OversetCoupling.field_query must retain the FE mesh and geometry layout; got a stale donor layout."
                    )
                selected = np.asarray(locator.support_cells(query.points))
                selected = selected[selected >= 0]
                support_ids = np.asarray(
                    mesh.block(locator.cell_map.block_name).global_ids
                )[np.unique(selected)]
            case UnstructuredFiniteVolumeFieldReconstructionKernel():
                identifiers = np.concatenate(
                    [np.asarray(block.global_ids) for block in mesh.blocks]
                )
                support_ids = identifiers[kernel.query_support_rows(query)]
            case _ as unknown:
                assert_never(unknown)
        if not np.array_equal(
            np.sort(np.unique(support_ids)), np.asarray(source_scope.entity_ids)
        ):
            raise ValueError(
                "OversetCoupling.source_scope must retain the query support cells; got different support identities."
            )
        if hole_scope is not None:
            target.require_scope(hole_scope)
            if (
                hole_scope.entity_dimension != 0
                or hole_scope.entity_set_id != target_scope.entity_set_id
                or np.intersect1d(hole_scope.entity_ids, target_scope.entity_ids).size
            ):
                raise ValueError(
                    "OversetCoupling.field_query.hole_scope must contain target vertices in the receptor entity set while remaining disjoint from receptors; got a non-vertex scope, entity-set mismatch, or receptor overlap refused."
                )
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
        """Returns every geometric cell identity read by the query, excluding coefficient
        DOF IDs.

        The field provider validates this scope against every selected cell and every
        reconstruction-stencil support cell before query-mode publication. Coefficients
        retain the independent ``PreparedFieldQuery`` field layout. Vertex-stencil mode
        has no field-query support-cell scope.
        """
        return self.source_scope if self.field_query is not None else None

    @checked
    def require_current(self, source: MeshPart, target: MeshPart, /) -> None:
        if self.field_query is None:
            super().require_current(source, target)
        else:
            source.require_scope(self.source_scope)
            target.require_scope(self.target_scope)

    @classmethod
    @checked
    def search(
        cls,
        source: MeshPart,
        target: MeshPart,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        /,
        *,
        hole_scope: MeshingScope | None = None,
        location_policy: simplicial_locator_owner.SimplicialLocationPolicy | None = None,
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
            raise ValueError(
                "MeshCoupling.source_values must have one leading-axis value per source-scope entity; got a scalar or mismatched leading axis."
            )
        if self.donor_weights is None or self.donor_rows is None:
            raise RuntimeError(
                "OversetCoupling.stencil_route must provide prepared donor weights and donor rows; got missing stencil state."
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
                raise ValueError(
                    "OversetCoupling.transpose.target_field must match the field-query output shape; got a different value shape."
                )
            if self.rotation is not None and self.value_action != "invariant":
                values = values @ self.rotation.astype(values.dtype)
            return self.field_query.transpose(values)
        values = jnp.asarray(target_values)
        if values.ndim == 0 or values.shape[0] != self.target_scope.entity_ids.size:
            raise ValueError(
                "MeshCoupling.target_values must have one leading-axis value per target-scope entity; got a scalar or mismatched leading axis."
            )
        if self.donor_weights is None or self.donor_rows is None:
            raise RuntimeError(
                "OversetCoupling.stencil_route must provide prepared donor weights and donor rows; got missing stencil state."
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
        raise ValueError(
            "ContactCoupling.capture_radius must be finite and positive; got a non-finite or non-positive radius."
        )
    if source_scope.entity_dimension != 0:
        raise ValueError(
            "ContactCoupling.source_scope.entity_dimension must be zero for node contact search; got a non-point source scope."
        )
    source_ids = np.asarray(source_scope.entity_ids, dtype=np.int64)
    if source_ids.size != np.asarray(target_scope.entity_ids).size:
        raise ValueError(
            "ContactCoupling.point_scopes must contain equal numbers of source and target points; got different scope sizes."
        )
    donors = np.asarray(source.point_coordinates(source_scope), dtype=np.float64)
    receptors = np.asarray(target.point_coordinates(target_scope), dtype=np.float64)
    if not np.all(np.isfinite(donors)):
        raise ValueError(
            "ContactCoupling.donor_points must contain finite coordinates; got non-finite donor coordinates."
        )
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
        raise TypeError(
            "OversetCoupling.source.carrier must be a certified CellMeshingResult; got a donor carrier without a certified cell mesh."
        )
    mesh = carrier.mesh
    if (
        len(mesh.blocks) != 1
        or mesh.blocks[0].cell_kind not in ("triangle", "tetrahedron")
        or mesh.topological_dimension != mesh.ambient_dimension
    ):
        raise ValueError(
            "OversetCoupling.donor_mesh must contain one full-dimensional triangle or tetrahedron block; got an unsupported block count, cell kind, or dimension."
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
            "OversetCoupling.donor_geometry must use the mesh coordinates and block vertex rows; got coordinates or connectivity not aligned with the donor mesh."
        )
    return cell_map, coordinates, vertex_rows


def _overset_donor_search(
    source: MeshPart,
    target: MeshPart,
    source_scope: MeshingScope,
    target_scope: MeshingScope,
    hole_scope: MeshingScope | None,
    location_policy: simplicial_locator_owner.SimplicialLocationPolicy | None,
    /,
) -> tuple[CouplingSearchEvidence, np.ndarray]:
    _endpoints(source, target, source_scope, target_scope)
    if source_scope.entity_dimension != 0:
        raise ValueError(
            "OversetCoupling.source_scope.entity_dimension must be zero for donor interpolation; got a non-point source scope."
        )
    if hole_scope is not None:
        target.require_scope(hole_scope)
    cell_map, coordinates, vertex_rows = _donor_cell_map(source)
    policy = (
        simplicial_locator_owner.SimplicialLocationPolicy(
            min(cell_map.cell_count, 64), 16, 1
        )
        if location_policy is None
        else location_policy
    )
    if not isinstance(policy, simplicial_locator_owner.SimplicialLocationPolicy):
        raise TypeError(
            "OversetCoupling.location_policy must be a SimplicialLocationPolicy or None; got an unsupported policy type."
        )
    located = simplicial_locator_owner.PreparedSimplicialCellLocator(
        cell_map, coordinates, policy
    ).locate(target.point_coordinates(target_scope))
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
            location == int(simplicial_locator_owner.CellLocationStatus.OUTSIDE),
            location == int(simplicial_locator_owner.CellLocationStatus.NONFINITE),
            location
            == int(simplicial_locator_owner.CellLocationStatus.RESOURCE_EXCEEDED),
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
