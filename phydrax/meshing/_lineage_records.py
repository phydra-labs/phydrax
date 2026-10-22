#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import fields
from enum import IntEnum, StrEnum
from fractions import Fraction
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike as JaxArrayLike
from numpy.typing import ArrayLike

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    logical_array_value_collection_digest,
    tree_content_equal,
)
from .._host_storage import declared_integer_storage_bytes
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..sparse import RowRelation, SparseLinearMap
from ..typing import checked


if TYPE_CHECKING:
    from ..discretization.fem._topology_transfer import FiniteElementTopologyTransfer


class EntityLineageKind(IntEnum):
    PRESERVED = 0
    REFINED_FROM = 1
    COARSENED_INTO = 2
    SPLIT_FROM = 3
    MERGED_INTO = 4
    COLLAPSED_INTO = 5
    SWAPPED_FROM = 6
    RELOCATED = 7
    GENERATED_ON_GEOMETRY = 8
    UNKNOWN = 9


class MeshTransitionKind(StrEnum):
    REFINE = "refine"
    COARSEN = "coarsen"
    REMESH = "remesh"
    REPAIR = "repair"
    PARTITION = "partition"
    GEOMETRY_REALIZATION = "geometry_realization"
    MOTION = "motion"


class EntityLineage(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    source_entity_set_id: str = eqx.field(static=True)
    target_entity_set_id: str = eqx.field(static=True)
    source_global_ids: Array
    target_global_ids: Array
    relation_kinds: Array
    created_target_ids: Array
    deleted_source_ids: Array
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        source_entity_set_id: str,
        target_entity_set_id: str,
        source_global_ids: ArrayLike,
        target_global_ids: ArrayLike,
        relation_kinds: ArrayLike,
        /,
        *,
        created_target_ids: ArrayLike = (),
        deleted_source_ids: ArrayLike = (),
    ) -> None:
        source_set = str(source_entity_set_id).strip()
        target_set = str(target_entity_set_id).strip()
        if not source_set or not target_set:
            raise ValueError(
                f"EntityLineage.entity_set_ids must be nonempty; got {source_set!r}/{target_set!r}"
            )
        dimension_ = int(dimension)
        if dimension_ < 0:
            raise ValueError(
                f"EntityLineage.dimension must be nonnegative; got {dimension_}"
            )
        source_raw, target_raw, kinds_raw, created_raw, deleted_raw = jax.device_get(
            (
                source_global_ids,
                target_global_ids,
                relation_kinds,
                created_target_ids,
                deleted_source_ids,
            )
        )
        source = np.asarray(source_raw, dtype=np.int64)
        target = np.asarray(target_raw, dtype=np.int64)
        kinds = np.asarray(kinds_raw, dtype=np.int32)
        created = np.asarray(created_raw, dtype=np.int64)
        deleted = np.asarray(deleted_raw, dtype=np.int64)
        if (
            source.ndim != 1
            or target.shape != source.shape
            or kinds.shape != source.shape
        ):
            raise ValueError(
                f"EntityLineage.relations must be aligned rank-one arrays; got shapes {source.shape}/{target.shape}/{kinds.shape}"
            )
        if created.ndim != 1 or deleted.ndim != 1:
            raise ValueError(
                f"EntityLineage.created_deleted_ids must be rank-one arrays; got shapes {created.shape}/{deleted.shape}"
            )
        if (
            np.any(source < 0)
            or np.any(target < 0)
            or np.any(created < 0)
            or np.any(deleted < 0)
        ):
            raise ValueError(
                "EntityLineage.global_ids must be nonnegative; got negative relation or created/deleted IDs"
            )
        supported = np.asarray(
            [int(value) for value in EntityLineageKind], dtype=np.int32
        )
        if np.any(~np.isin(kinds, supported)):
            raise ValueError(
                "EntityLineage.relation_kinds must use EntityLineageKind values; got an unsupported kind"
            )
        if np.any(np.isin(target, created)) or np.any(np.isin(source, deleted)):
            raise ValueError(
                "EntityLineage.created_deleted_ids must be disjoint from relation IDs; got overlap"
            )
        self.dimension = dimension_
        self.source_entity_set_id = source_set
        self.target_entity_set_id = target_set
        (
            self.source_global_ids,
            self.target_global_ids,
            self.relation_kinds,
            self.created_target_ids,
            self.deleted_source_ids,
        ) = jax.device_put((source, target, kinds, created, deleted))
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "entity-lineage",
                "dimension": dimension_,
                "source_entity_set_id": source_set,
                "target_entity_set_id": target_set,
                "relations": logical_array_value_collection_digest(
                    {
                        "source_global_ids": source,
                        "target_global_ids": target,
                        "relation_kinds": kinds,
                        "created_target_ids": created,
                        "deleted_source_ids": deleted,
                    }
                ),
            }
        )


@final
class EntityLineageAccounting(StrictModule):
    """Host counts for one completely accounted entity dimension."""

    dimension: int = eqx.field(static=True)
    relations: int = eqx.field(static=True)
    created: int = eqx.field(static=True)
    deleted: int = eqx.field(static=True)

    def __init__(
        self, dimension: int, relations: int, created: int, deleted: int
    ) -> None:
        if min(dimension, relations, created, deleted) < 0:
            raise ValueError(
                f"EntityLineageAccounting.counts must be nonnegative; got {(dimension, relations, created, deleted)!r}"
            )
        self.dimension = dimension
        self.relations = relations
        self.created = created
        self.deleted = deleted

    def to_record(self) -> dict[str, int]:
        return {
            "relations": self.relations,
            "created": self.created,
            "deleted": self.deleted,
        }


@final
class LineageAccounting(StrictModule):
    """Source/target entity coverage, without retaining either mesh."""

    lineage_id: str = eqx.field(static=True)
    dimensions: tuple[EntityLineageAccounting, ...] = eqx.field(static=True)

    def __init__(
        self, lineage_id: str, dimensions: tuple[EntityLineageAccounting, ...]
    ) -> None:
        if not lineage_id:
            raise ValueError(
                f"LineageAccounting.lineage_id must be nonempty; got {lineage_id!r}"
            )
        observed = tuple(item.dimension for item in dimensions)
        if not dimensions or observed != tuple(range(len(dimensions))):
            raise ValueError(
                f"LineageAccounting.dimensions must cover ordered consecutive dimensions; got {observed!r}"
            )
        self.lineage_id = lineage_id
        self.dimensions = dimensions

    def to_record(self) -> dict[str, object]:
        return {
            "lineage_id": self.lineage_id,
            "complete": True,
            "dimensions": {
                str(item.dimension): item.to_record() for item in self.dimensions
            },
        }


def _entity_lineage_accounting(
    item: EntityLineage, source: CellMesh, target: CellMesh
) -> EntityLineageAccounting:
    before = source.entity_set(item.dimension)
    after = target.entity_set(item.dimension)
    if (item.source_entity_set_id, item.target_entity_set_id) != (
        before.entity_set_id,
        after.entity_set_id,
    ):
        raise ValueError(
            f"MeshLineage.entity_set_ids must bind the supplied source and target dimension {item.dimension}; got {(item.source_entity_set_id, item.target_entity_set_id)!r}"
        )
    source_ids = np.asarray(before.entity_ids)
    target_ids = np.asarray(after.entity_ids)
    created = np.asarray(item.created_target_ids)
    deleted = np.asarray(item.deleted_source_ids)
    if (
        not np.all(
            np.isin(
                target_ids,
                np.concatenate((source_ids, np.asarray(item.target_global_ids), created)),
            )
        )
        or not np.all(
            np.isin(
                source_ids,
                np.concatenate((target_ids, np.asarray(item.source_global_ids), deleted)),
            )
        )
        or not np.all(np.isin(created, target_ids) & ~np.isin(created, source_ids))
        or not np.all(np.isin(deleted, source_ids) & ~np.isin(deleted, target_ids))
    ):
        raise ValueError(
            f"MeshLineage.entities must account for every source and target entity in dimension {item.dimension}; got source={source_ids.size}, target={target_ids.size}, relations={item.relation_kinds.shape[0]}, created={created.size}, deleted={deleted.size}"
        )
    return EntityLineageAccounting(
        item.dimension, item.relation_kinds.shape[0], created.size, deleted.size
    )


class MeshLineage(StrictModule, NonTrainableState):
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    entities: tuple[EntityLineage, ...]
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_topology_id: str,
        target_topology_id: str,
        entities: tuple[EntityLineage, ...],
        /,
    ) -> None:
        source = str(source_topology_id).strip()
        target = str(target_topology_id).strip()
        records = tuple(entities)
        if not source or not target:
            raise ValueError(
                f"MeshLineage.topology_ids must be nonempty; got {source!r}/{target!r}"
            )
        if not records or not all(isinstance(value, EntityLineage) for value in records):
            raise ValueError(
                f"MeshLineage.entities must contain EntityLineage records; got {tuple(type(value).__name__ for value in records)!r}"
            )
        dimensions = tuple(value.dimension for value in records)
        if dimensions != tuple(sorted(set(dimensions))):
            raise ValueError(
                f"MeshLineage.entities must contain one ordered record per dimension; got {dimensions!r}"
            )
        self.source_topology_id = source
        self.target_topology_id = target
        self.entities = records
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "mesh-lineage",
                "source_topology": source,
                "target_topology": target,
                "entities": [value.lineage_id for value in records],
            }
        )

    def entity_lineage(self, dimension: int, /) -> EntityLineage:
        target = int(dimension)
        for value in self.entities:
            if value.dimension == target:
                return value
        raise KeyError(
            f"MeshLineage.dimension must select a retained record; got {target}"
        )

    @checked
    def validate_complete(
        self, source_mesh: CellMesh, target_mesh: CellMesh
    ) -> LineageAccounting:
        """Validate topology bindings and account for every retained entity."""
        if (self.source_topology_id, self.target_topology_id) != (
            source_mesh.topology_id,
            target_mesh.topology_id,
        ):
            raise ValueError(
                f"MeshLineage.topology_ids must bind the supplied source and target meshes; got {(self.source_topology_id, self.target_topology_id)!r}"
            )
        dimensions = tuple(item.dimension for item in self.entities)
        expected = tuple(range(source_mesh.topological_dimension + 1))
        if (
            target_mesh.topological_dimension != source_mesh.topological_dimension
            or dimensions != expected
        ):
            raise ValueError(
                f"MeshLineage.dimensions must cover source and target dimensions {expected!r}; got {dimensions!r}, target_dimension={target_mesh.topological_dimension}"
            )
        return LineageAccounting(
            self.lineage_id,
            tuple(
                _entity_lineage_accounting(item, source_mesh, target_mesh)
                for item in self.entities
            ),
        )


def _exact_row_integer_bits(width: int, /) -> int:
    """Bound every integer formed while rationally rounding binary64 rows.

    A finite binary64 value is ``m * 2**e`` with ``|m| < 2**53`` and
    ``-1074 <= e <= 971``, so its Fraction has at most 1024 numerator and 1075
    denominator bits. A sum of ``width`` such values stays below
    ``2**(2098 + width.bit_length())`` over a ``2**1074`` denominator, and the
    unreduced terms of one quotient by that sum below
    ``3 * 1075 + width.bit_length()`` bits.
    """
    return 3 * 1075 + width.bit_length()


def exact_fraction_bytes(count: int, width: int, /) -> int:
    """Bound ``count`` live Fractions formed from binary64 rows of ``width`` terms."""
    return count * (
        256 + 2 * declared_integer_storage_bytes(_exact_row_integer_bits(width))
    )


def _rounding_array_bytes(rows: int, width: int, /) -> int:
    # Positive mask, rounded output, deduplication copy, unique and rounded banks,
    # plus per-row counts, selection and inverse indices.
    return 40 * rows * width + 32 * rows


def homogeneous_rounding_bounds(rows: int, width: int, /) -> tuple[int, int]:
    """Upper bound the work and bytes of rounding ``rows`` homogeneous rows.

    Each distinct row with several positive weights costs ``4 * width`` rational
    operations (conversion, summation, division, rounding); two rows of live
    Fractions and the vectorized deduplication banks bound storage.
    """
    return 4 * width * rows, _rounding_array_bytes(rows, width) + exact_fraction_bytes(
        2 * (2 * width + 1), width
    )


def rounded_homogeneous_barycentrics(
    homogeneous: ArrayLike,
    valid: ArrayLike,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> np.ndarray:
    """Round exact homogeneous stencil rows to their binary64 barycentrics.

    Along the last axis, a row holds finite sign-bit-free binary64 weights ``h`` on
    its ``valid`` support, ``+0.0`` elsewhere, and a positive exact sum. Its rational
    barycentrics are the rationals ``h_j / sum(h)``; each is rounded to nearest
    with ties to even, so the result is deterministic for the exact row. A row with one
    positive weight is a unit row that rounding leaves unchanged; every distinct row with several positive
    weights is rounded once in rational arithmetic.

    A ``budget`` admits ``homogeneous_rounding_bounds`` before validation, then
    charges the banks and the rational work of the distinct rows before rounding
    them; ``None`` declares unbounded host validation of a standalone record.
    """

    weights = np.asarray(homogeneous, dtype=np.float64)
    support = np.asarray(valid, dtype=np.bool_)
    if weights.ndim == 0 or support.shape != weights.shape:
        raise ValueError(
            f"VertexInterpolationStencil.homogeneous_weights must align with the stencil support; got shapes {weights.shape}/{support.shape}"
        )
    rows = weights.reshape(-1, weights.shape[-1])
    count, width = rows.shape
    with nullcontext() if budget is None else budget.temporary_scope():
        if budget is not None:
            work_upper, bytes_upper = homogeneous_rounding_bounds(count, width)
            budget.admit_work_bound(work_upper, storage_upper=bytes_upper)
            budget.reserve(0, _rounding_array_bytes(count, width))
        if (
            not np.all(np.isfinite(weights))
            or np.any(np.signbit(weights))
            or np.any(weights[~support] != 0.0)
        ):
            raise ValueError(
                "VertexInterpolationStencil.homogeneous_weights must be finite sign-bit-free weights on the valid support and +0.0 elsewhere; got invalid weights"
            )
        positive = rows > 0.0
        counts = np.count_nonzero(positive, axis=1)
        if np.any(counts == 0):
            raise ValueError(
                "VertexInterpolationStencil.homogeneous_weights must have a positive exact sum in every row before binary64 rounding; got a zero row"
            )
        rounded = positive.astype(np.float64)
        mixed = np.flatnonzero(counts > 1)
        if mixed.size:
            unique, inverse = np.unique(rows[mixed], axis=0, return_inverse=True)
            if budget is not None:
                budget.reserve(
                    4 * width * unique.shape[0],
                    exact_fraction_bytes(2 * (2 * width + 1), width),
                )
            barycentrics = np.empty_like(unique)
            for index, row in enumerate(unique.tolist()):
                exact = [Fraction(value) for value in row]
                total = sum(exact, Fraction(0))
                barycentrics[index] = [float(value / total) for value in exact]
            rounded[mixed] = barycentrics[inverse.reshape(-1)]
    return rounded.reshape(weights.shape)


class VertexInterpolationStencil(StrictModule, NonTrainableState):
    """Sparse rows interpolating target vertices from source vertex global IDs.

    ``weights`` are the execution coefficients. An optional rational
    ``homogeneous_weights`` bank on the same support retains source-owned
    nondyadic rows: each row's exact barycentrics are ``h_j / sum(h)`` in rational
    arithmetic, and ``weights`` must be bitwise their correctly rounded values
    (``+0.0`` off the valid support). Such a stencil preserves constants exactly.
    Construction charges the rational validation to ``budget``; ``None`` declares
    unbounded host validation of a standalone record. Archive replay binds a
    ledger sized by ``reconstruction_preparation_bounds``.
    """

    source_entity_set_id: str = eqx.field(static=True)
    target_entity_set_id: str = eqx.field(static=True)
    target_global_ids: Array
    source_global_ids: Array
    weights: Array
    valid: Array
    homogeneous_weights: Array | None
    preserves_constants: bool = eqx.field(static=True)
    stencil_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_entity_set_id: str,
        target_entity_set_id: str,
        target_global_ids: ArrayLike,
        source_global_ids: ArrayLike,
        weights: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        preserves_constants: bool = True,
        homogeneous_weights: ArrayLike | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        source_set = str(source_entity_set_id).strip()
        target_set = str(target_entity_set_id).strip()
        targets_raw, sources_raw, weights_raw, valid_raw, exact_raw = jax.device_get(
            (target_global_ids, source_global_ids, weights, valid, homogeneous_weights)
        )
        targets = np.asarray(targets_raw, dtype=np.int64)
        sources = np.asarray(sources_raw, dtype=np.int64)
        coefficients = np.asarray(weights_raw, dtype=np.float64)
        valid_ = np.asarray(valid_raw, dtype=np.bool_)
        exact = None if exact_raw is None else np.asarray(exact_raw, dtype=np.float64)
        if not source_set or not target_set:
            raise ValueError(
                f"VertexInterpolationStencil.entity_set_ids must be nonempty; got {source_set!r}/{target_set!r}"
            )
        if targets.ndim != 1 or sources.ndim != 2:
            raise ValueError(
                f"VertexInterpolationStencil.global_ids must have target/source ranks one/two; got shapes {targets.shape}/{sources.shape}"
            )
        if sources.shape != coefficients.shape or valid_.shape != sources.shape:
            raise ValueError(
                f"VertexInterpolationStencil.weights must match source and validity shapes; got {sources.shape}/{coefficients.shape}/{valid_.shape}"
            )
        if sources.shape[0] != targets.shape[0] or targets.size == 0:
            raise ValueError(
                f"VertexInterpolationStencil.target_global_ids must be nonempty with one source row each; got shapes {targets.shape}/{sources.shape}"
            )
        if np.any(targets < 0) or np.any(sources[valid_] < 0):
            raise ValueError(
                "VertexInterpolationStencil.global_ids must be nonnegative on valid rows; got negative IDs"
            )
        if np.any(~np.isfinite(coefficients[valid_])):
            raise ValueError(
                "VertexInterpolationStencil.weights must be finite on valid rows; got nonfinite values"
            )
        if exact is not None:
            if exact.shape != sources.shape or not preserves_constants:
                raise ValueError(
                    f"VertexInterpolationStencil.homogeneous_weights must align with the source support of a constant-preserving stencil; got shape {exact.shape} for {sources.shape}, preserves_constants={preserves_constants}"
                )
            if not np.array_equal(
                coefficients.view(np.uint64),
                rounded_homogeneous_barycentrics(exact, valid_, budget=budget).view(
                    np.uint64
                ),
            ):
                raise ValueError(
                    "VertexInterpolationStencil.weights must be the correctly rounded exact barycentrics of homogeneous_weights; got different execution weights"
                )
        row_sums = np.sum(np.where(valid_, coefficients, 0.0), axis=1)
        # Admission tolerance retained from d2dfcb941, VertexInterpolationStencil;
        # it is a public row-normalization policy, not a numerical oracle tolerance.
        if preserves_constants and not np.allclose(row_sums, 1.0, atol=1.0e-12, rtol=0.0):
            raise ValueError(
                "VertexInterpolationStencil.weights must sum to one when preserves_constants is true; got nonunit row sums"
            )
        self.source_entity_set_id = source_set
        self.target_entity_set_id = target_set
        self.target_global_ids = jax.device_put(np.asarray(targets))
        self.source_global_ids = jax.device_put(np.asarray(sources))
        self.weights = jax.device_put(np.asarray(coefficients))
        self.valid = jax.device_put(np.asarray(valid_))
        self.homogeneous_weights = None if exact is None else jax.device_put(exact)
        if (
            self.homogeneous_weights is not None
            and self.homogeneous_weights.dtype != np.float64
        ):
            raise ValueError(
                f"VertexInterpolationStencil.homogeneous_weights must retain binary64 device storage; got {self.homogeneous_weights.dtype}"
            )
        self.preserves_constants = bool(preserves_constants)
        payload: dict[str, object] = {
            "kind": "vertex-interpolation-stencil",
            "source_entity_set_id": source_set,
            "target_entity_set_id": target_set,
            "target_global_ids": array_tree_fingerprint(targets),
            "source_global_ids": array_tree_fingerprint(sources),
            "weights": array_tree_fingerprint(coefficients),
            "valid": array_tree_fingerprint(valid_),
            "preserves_constants": bool(preserves_constants),
        }
        if exact is not None:
            payload["homogeneous_weights"] = array_tree_fingerprint(exact)
        self.stencil_id = canonical_fingerprint(payload)

    def reconstruction_preparation_bounds(self) -> tuple[int, int, int]:
        """Bound replay work, bytes and source queries of the rational-bank validation."""
        if self.homogeneous_weights is None:
            return 0, 0, 0
        if self.homogeneous_weights.ndim != 2:
            raise ValueError(
                f"VertexInterpolationStencil.homogeneous_weights must be a rank-two bank; got shape {self.homogeneous_weights.shape}"
            )
        rows, width = self.homogeneous_weights.shape
        work, storage = homogeneous_rounding_bounds(int(rows), int(width))
        return work, storage, 0

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[VertexInterpolationStencil],
        value: VertexInterpolationStencil,
        /,
        *,
        budget: CoordinateEnclosureBudget | None,
    ) -> VertexInterpolationStencil:
        """Revalidate retained rows, charging the rational bank to the replay ledger."""
        if (
            cls is not VertexInterpolationStencil
            or type(value) is not VertexInterpolationStencil
        ):
            raise TypeError(
                "VertexInterpolationStencil.restore_type must be the registered VertexInterpolationStencil; "
                f"got {type(value).__qualname__}"
            )
        rebuilt = cls(
            value.source_entity_set_id,
            value.target_entity_set_id,
            value.target_global_ids,
            value.source_global_ids,
            value.weights,
            value.valid,
            preserves_constants=value.preserves_constants,
            homogeneous_weights=value.homogeneous_weights,
            budget=budget,
        )
        for member in fields(value):
            if not tree_content_equal(
                getattr(rebuilt, member.name), getattr(value, member.name)
            ):
                raise ValueError(
                    f"stored_fields.VertexInterpolationStencil.{member.name} must match reconstructed retained content; got changed {member.name}"
                )
        return rebuilt

    def _source_routes(self, source_global_ids: ArrayLike, /) -> np.ndarray:
        """Resolve stencil source IDs to positions in one caller-supplied ID order."""

        identifiers = np.asarray(source_global_ids, dtype=np.int64)
        if identifiers.ndim != 1 or identifiers.size == 0:
            raise ValueError(
                f"VertexInterpolationStencil.source_global_ids must be a nonempty rank-one array; got {identifiers.shape}"
            )
        order = np.argsort(identifiers, kind="stable")
        ordered = identifiers[order]
        if np.any(ordered[1:] == ordered[:-1]):
            raise ValueError(
                "VertexInterpolationStencil.source_global_ids must be unique; got duplicate IDs"
            )
        references_raw, valid_raw = jax.device_get((self.source_global_ids, self.valid))
        references = np.asarray(references_raw)
        valid = np.asarray(valid_raw)
        position = np.minimum(np.searchsorted(ordered, references), identifiers.size - 1)
        if np.any(valid & (ordered[position] != references)):
            raise ValueError(
                "VertexInterpolationStencil.source_global_ids must include all valid stencil references; got unavailable IDs"
            )
        return np.where(valid, order[position], 0).astype(np.int32)

    def as_transfer(
        self,
        source_global_ids: ArrayLike,
        /,
        *,
        source_topology_id: str,
        target_topology_id: str,
        preserves_linear: bool = False,
        conservative: bool = False,
        source_coordinates: ArrayLike | None = None,
        target_coordinates: ArrayLike | None = None,
        source_measures: ArrayLike | None = None,
        target_measures: ArrayLike | None = None,
    ) -> FiniteElementTopologyTransfer:
        """Resolve this stencil into a sparse transfer over one source DOF order.

        Linear-preservation and conservation claims are certified by the transfer
        owner from the supplied row-aligned coordinates and P1 DOF measures.
        """

        # FEM transfer preparation consumes coordinate-transfer owners that import
        # lineage through surface source support; resolve only at transfer execution.
        from ..discretization.fem._topology_transfer import vertex_interpolation_transfer

        identifiers = np.asarray(jax.device_get(source_global_ids), dtype=np.int64)
        weights, valid = jax.device_get((self.weights, self.valid))
        return vertex_interpolation_transfer(
            self._source_routes(identifiers),
            np.asarray(weights),
            np.asarray(valid),
            source_size=identifiers.size,
            source_topology_id=source_topology_id,
            target_topology_id=target_topology_id,
            preserves_linear=preserves_linear,
            conservative=conservative,
            # ty: ignore[invalid-argument-type]
            source_coordinates=source_coordinates,
            # ty: ignore[invalid-argument-type]
            target_coordinates=target_coordinates,
            # ty: ignore[invalid-argument-type]
            source_measures=source_measures,
            # ty: ignore[invalid-argument-type]
            target_measures=target_measures,
        )

    def apply(
        self,
        source_global_ids: ArrayLike,
        values: JaxArrayLike,
        /,
    ) -> Array:
        """Interpolate source values given in ``source_global_ids`` order in one gather."""

        identifiers = np.asarray(source_global_ids, dtype=np.int64)
        source_values = jnp.asarray(values)
        routes = self._source_routes(identifiers)
        if source_values.ndim == 0 or source_values.shape[0] != identifiers.size:
            raise ValueError(
                f"VertexInterpolationStencil.values must align with source_global_ids; got {source_values.shape} for {identifiers.size} IDs"
            )
        relation = RowRelation(routes, source_size=identifiers.size, valid=self.valid)
        return SparseLinearMap(
            relation,
            self.weights,
            operator_id=canonical_fingerprint(
                {
                    "kind": "vertex-interpolation-stencil-map",
                    "stencil": self.stencil_id,
                    "source_global_ids": array_tree_fingerprint(identifiers),
                }
            ),
        ).mv(source_values)


__all__ = [
    "EntityLineage",
    "EntityLineageKind",
    "MeshLineage",
    "MeshTransitionKind",
    "VertexInterpolationStencil",
    "exact_fraction_bytes",
    "homogeneous_rounding_bounds",
    "rounded_homogeneous_barycentrics",
]
