#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified common refinement (supermesh) of two cell meshes.

The common refinement of a source and a target mesh is the set of
positive-measure intersections of their cells.  Preparation is host-only NumPy
interchange in three certified phases:

1. Decomposition.  Triangles, tetrahedra, and convex polygons are single convex
   pieces.  Every other cell is the cone of its canonically triangulated
   boundary (faces fanned from their smallest mesh vertex, so neighbors agree on
   shared faces) from one of its vertices.  A cone is accepted only when every
   nondegenerate cone simplex has one exact orientation: the signed cone
   indicators sum to the winding number of the boundary, so equal signs certify
   a partition of the cell.  Apexes are tried in ascending vertex order.
2. Broad phase.  Packed float64 BVHs over the cell boxes enumerate every cell
   pair whose boxes overlap with positive extent, which any positive-measure
   intersection requires.
3. Narrow phase.  meshcore clips every piece pair with exactly classified
   vertices, in batches with a bounded working set, and the moments (optionally
   simplex partitions) are summed per cell pair.

Uncertified cells, unresolved predicates, native item
failures, coverage gaps, double coverage, and resource refusals are reported
through :class:`CommonRefinementStatus` with evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager, nullcontext
from dataclasses import fields
from enum import IntEnum, StrEnum
from fractions import Fraction
from typing import cast, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax._meshcore as meshcore_owner
import phydrax.discretization._cell_complex as cell_complex_owner

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    tree_content_equal,
)
from .._geometry_predicates import orient2d_host, orient3d_host, PredicateMode
from .._host_storage import declared_integer_storage_bytes
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    CoordinateLiveStorage,
)
from ..discretization._reference_cell_types import CellBlock, PolyhedralBlock
from ..ein import contract
from ..typing import checked, parse
from ._exact_polyhedral_geometry import (
    exact_vertices,
    ExactOverlapRefusal,
    ExactPolyhedralRefusal,
    star_determinant,
    Tetrahedron,
    tetrahedron_overlap,
    TetrahedronPlanePreparation,
)


# Narrow-phase working set per native batch (inputs plus outputs).
_BATCH_BYTES = 64 * 1024 * 1024
# Simplex capacity of one tetrahedron-pair intersection (2 V - 4 with V <= 12).
_TETRAHEDRON_SIMPLICES = 20
# Coverage roundoff floor in units of eps * diameter**dimension of the cell box:
# measures are computed relative to local origins, so their absolute rounding
# scales with the cell size, not with its (possibly sliver) measure.
_COVERAGE_ROUNDOFF_ULPS = 256.0


class CommonRefinementStatus(IntEnum):
    """Fail-closed outcome of common-refinement preparation."""

    SUCCESS = 0
    INVALID_GEOMETRY = 1
    PREDICATE_UNCERTAIN = 2
    INTERSECTION_FAILURE = 3
    DOUBLE_COVERAGE = 4
    COVERAGE_GAP = 5
    RESOURCE_LIMIT = 6


class CommonRefinementCoverage(StrEnum):
    """Select which cell sets the overlap partition must cover.

    Over-coverage (double coverage) fails for every requirement.
    """

    COMPLETE = "complete"
    TARGET = "target"
    SOURCE = "source"
    PARTIAL = "partial"


class CommonRefinementPolicy(StrictModule):
    """Certification mode, coverage requirement, and resource limits.

    ``coverage_tolerance`` is relative to the certified cell measure: cell ``i``
    with measure ``m_i``, covered measure ``c_i``, and box diagonal ``h_i`` in
    dimension ``d`` has tolerance ``t_i = coverage_tolerance * m_i + 256 eps
    h_i**d`` (the roundoff floor keeps sliver cells, whose rounding scales with
    ``h_i**d`` rather than with ``m_i``, from failing spuriously).  It is covered
    when ``|c_i - m_i| <= t_i``, has a gap when ``m_i - c_i > t_i``, and is
    double covered when ``c_i - m_i > t_i``.

    ``maximum_candidate_pairs`` bounds broad-phase cell pairs,
    ``maximum_accepted_pairs`` bounds positive-measure entries, and
    ``maximum_memory_bytes`` bounds admitted host decomposition, broad-phase,
    retained artifact and narrow-phase storage. Memory refusal raises before
    allocation, including when even failure evidence cannot fit.
    ``second_moments`` fills the overlap second moments and
    ``overlap_simplices`` retains the entire simplex partition of every overlap,
    with intersection classifications determined by exact predicates.
    """

    predicate_mode: PredicateMode = eqx.field(static=True)
    coverage: CommonRefinementCoverage = eqx.field(static=True)
    coverage_tolerance: float = eqx.field(static=True)
    maximum_candidate_pairs: int = eqx.field(static=True)
    maximum_accepted_pairs: int = eqx.field(static=True)
    maximum_memory_bytes: int = eqx.field(static=True)
    maximum_exact_work: int = eqx.field(static=True)
    second_moments: bool = eqx.field(static=True)
    overlap_simplices: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        predicate_mode: PredicateMode = PredicateMode.EXACT,
        coverage: CommonRefinementCoverage = CommonRefinementCoverage.COMPLETE,
        coverage_tolerance: float = 1.0e-10,
        maximum_candidate_pairs: int = 50_000_000,
        maximum_accepted_pairs: int = 50_000_000,
        maximum_memory_bytes: int = 4 * 1024**3,
        maximum_exact_work: int = 100_000_000,
        second_moments: bool = False,
        overlap_simplices: bool = False,
    ) -> None:
        predicate_mode = parse(
            predicate_mode, PredicateMode, "CommonRefinementPolicy.predicate_mode"
        )
        if predicate_mode is PredicateMode.FILTERED_DEVICE:
            raise ValueError(
                "CommonRefinementPolicy.predicate_mode must be FILTERED or EXACT for host certification; got FILTERED_DEVICE."
            )
        coverage = parse(
            coverage, CommonRefinementCoverage, "CommonRefinementPolicy.coverage"
        )
        if not isinstance(second_moments, bool) or not isinstance(
            overlap_simplices, bool
        ):
            raise TypeError(
                "CommonRefinementPolicy.moment_flags must be bools; got a different scalar kind."
            )
        tolerance = float(coverage_tolerance)
        if not np.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
            raise ValueError(
                f"CommonRefinementPolicy.coverage_tolerance must be finite and in [0, 1); got {tolerance}."
            )
        candidates = positive_integer(maximum_candidate_pairs, "maximum_candidate_pairs")
        accepted = positive_integer(maximum_accepted_pairs, "maximum_accepted_pairs")
        memory = positive_integer(maximum_memory_bytes, "maximum_memory_bytes")
        exact_work = positive_integer(maximum_exact_work, "maximum_exact_work")
        self.predicate_mode = predicate_mode
        self.coverage = coverage
        self.coverage_tolerance = tolerance
        self.maximum_exact_work = exact_work
        self.maximum_candidate_pairs = candidates
        self.maximum_accepted_pairs = accepted
        self.maximum_memory_bytes = memory
        self.second_moments = second_moments
        self.overlap_simplices = overlap_simplices
        self.policy_id = canonical_fingerprint(
            {
                "kind": "common-refinement-policy",
                "predicate_mode": predicate_mode.value,
                "coverage": coverage.value,
                "coverage_tolerance": tolerance,
                "maximum_candidate_pairs": candidates,
                "maximum_accepted_pairs": accepted,
                "maximum_memory_bytes": memory,
                "maximum_exact_work": exact_work,
                "second_moments": second_moments,
                "overlap_simplices": overlap_simplices,
            }
        )

    def native_limits(self) -> meshcore_owner.NativeExecutionLimits:
        """Lower maximum_exact_work and managed-storage allowances; queries consume work."""
        return meshcore_owner.NativeExecutionLimits(
            self.maximum_exact_work,
            self.maximum_exact_work,
            0,
            self.maximum_memory_bytes,
            math.inf,
        )


class CommonRefinementEvidence(StrictModule, NonTrainableState):
    """Coverage, certification, and resource evidence of one refinement.

    Coverage defects are ``covered - measure`` per cell (negative for gaps,
    positive for double coverage) and the tolerances are the per-cell ``t_i`` of
    :class:`CommonRefinementPolicy`.  ``piece_pair_count`` counts native clips.
    ``working_bytes`` is the historical producing request's observed working
    storage, not a reproducible geometric invariant. Restoration retains that
    observation; its new certification is charged to the current caller budget.
    """

    status: CommonRefinementStatus = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    source_coverage_defects: Array
    target_coverage_defects: Array
    source_coverage_tolerances: Array
    target_coverage_tolerances: Array
    source_gap_count: int = eqx.field(static=True)
    target_gap_count: int = eqx.field(static=True)
    source_double_count: int = eqx.field(static=True)
    target_double_count: int = eqx.field(static=True)
    maximum_relative_source_defect: float = eqx.field(static=True)
    maximum_relative_target_defect: float = eqx.field(static=True)
    candidate_pair_count: int = eqx.field(static=True)
    accepted_pair_count: int = eqx.field(static=True)
    piece_pair_count: int = eqx.field(static=True)
    uncertain_predicate_count: int = eqx.field(static=True)
    invalid_cell_count: int = eqx.field(static=True)
    intersection_failure_count: int = eqx.field(static=True)
    retained_bytes: int = eqx.field(static=True)
    working_bytes: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        status: CommonRefinementStatus,
        reason: str,
        source_coverage_defects: np.ndarray,
        target_coverage_defects: np.ndarray,
        source_measures: np.ndarray,
        target_measures: np.ndarray,
        source_tolerances: np.ndarray,
        target_tolerances: np.ndarray,
        counts: Counts,
        retained_bytes: int,
        working_bytes: int,
    ) -> None:
        status = parse(status, CommonRefinementStatus, "CommonRefinementEvidence.status")
        source_defects = np.asarray(source_coverage_defects, dtype=np.float64)
        target_defects = np.asarray(target_coverage_defects, dtype=np.float64)
        if (
            source_defects.shape != source_measures.shape
            or target_defects.shape != target_measures.shape
            or source_tolerances.shape != source_measures.shape
            or target_tolerances.shape != target_measures.shape
        ):
            raise ValueError(
                "CommonRefinementEvidence.coverage_arrays must align with the cell measures; got incompatible shapes."
            )
        source_tolerance = np.asarray(source_tolerances, dtype=np.float64)
        target_tolerance = np.asarray(target_tolerances, dtype=np.float64)
        self.status = status
        self.reason = str(reason)
        self.source_coverage_defects = jax.device_put(np.asarray(source_defects))
        self.target_coverage_defects = jax.device_put(np.asarray(target_defects))
        self.source_coverage_tolerances = jax.device_put(np.asarray(source_tolerance))
        self.target_coverage_tolerances = jax.device_put(np.asarray(target_tolerance))
        self.source_gap_count = int(np.sum(-source_defects > source_tolerance))
        self.target_gap_count = int(np.sum(-target_defects > target_tolerance))
        self.source_double_count = int(np.sum(source_defects > source_tolerance))
        self.target_double_count = int(np.sum(target_defects > target_tolerance))
        self.maximum_relative_source_defect = _relative_maximum(
            source_defects, source_measures
        )
        self.maximum_relative_target_defect = _relative_maximum(
            target_defects, target_measures
        )
        self.candidate_pair_count = counts.candidates
        self.accepted_pair_count = counts.accepted
        self.piece_pair_count = counts.piece_pairs
        self.uncertain_predicate_count = counts.uncertain
        self.invalid_cell_count = counts.invalid
        self.intersection_failure_count = counts.failures
        self.retained_bytes = int(retained_bytes)
        self.working_bytes = int(working_bytes)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "common-refinement-evidence",
                "status": int(status),
                "reason": self.reason,
                "source_defects": array_tree_fingerprint(source_defects),
                "target_defects": array_tree_fingerprint(target_defects),
                "counts": list(counts),
                "retained_bytes": self.retained_bytes,
                "working_bytes": self.working_bytes,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[CommonRefinementEvidence],
        value: CommonRefinementEvidence,
        /,
        *,
        source_measures: Array,
        target_measures: Array,
    ) -> CommonRefinementEvidence:
        """Rebuild evidence with the measures retained by its refinement owner.

        Stored tolerances and execution counts are constructor inputs, while gap
        counts, relative defects, and the evidence identity are recomputed.
        Registered replay authenticates every reconstructed stored field.
        """
        if type(value) is not cls:
            raise TypeError(
                "CommonRefinementEvidence.from_stored_fields.value must have the registered evidence type; "
                f"got {type(value).__name__}."
            )
        rebuilt = cls(
            status=value.status,
            reason=value.reason,
            source_coverage_defects=np.asarray(value.source_coverage_defects),
            target_coverage_defects=np.asarray(value.target_coverage_defects),
            source_measures=np.asarray(source_measures),
            target_measures=np.asarray(target_measures),
            source_tolerances=np.asarray(value.source_coverage_tolerances),
            target_tolerances=np.asarray(value.target_coverage_tolerances),
            counts=Counts(
                value.candidate_pair_count,
                value.accepted_pair_count,
                value.piece_pair_count,
                value.uncertain_predicate_count,
                value.invalid_cell_count,
                value.intersection_failure_count,
            ),
            retained_bytes=value.retained_bytes,
            working_bytes=value.working_bytes,
        )
        if not tree_content_equal(value, rebuilt):
            raise ValueError(
                "CommonRefinementEvidence.stored_fields must reproduce its historical "
                "observations and derived evidence; got changed evidence."
            )
        return rebuilt


def _relative_maximum(defects: np.ndarray, measures: np.ndarray, /) -> float:
    # Uncertified cells carry no pieces and hence no measure.
    measured = measures > 0.0
    if not np.any(measured):
        return 0.0
    return float(np.max(np.abs(defects[measured]) / measures[measured]))


def _canonical_cell_indices(
    values: np.ndarray, cell_count: int, name: str, /
) -> np.ndarray:
    """Validate cell indices before narrowing to their stored representation."""
    indices = np.asarray(values)
    if indices.dtype.kind not in "iu":
        raise TypeError(
            f"PreparedCommonRefinement.entries.{name} must have an integer dtype; got {indices.dtype}."
        )
    if (
        np.any(indices < 0)
        or np.any(indices > np.iinfo(np.int32).max)
        or np.any(indices >= cell_count)
    ):
        raise ValueError(
            f"PreparedCommonRefinement.entries.{name} must contain int32 cell indices in [0, {cell_count}); got out-of-range indices."
        )
    return np.asarray(indices, dtype=np.int32)


class PreparedCommonRefinement(StrictModule, NonTrainableState):
    """CSR common refinement grouped by target cell.

    Entry ``k`` of row ``t`` (``target_offsets[t] <= k < target_offsets[t + 1]``)
    is the positive-measure overlap of source cell ``source_cells[k]`` with
    target cell ``t``, with measure ``volumes[k]``, first moment
    ``first_moments[k]`` (the integral of ``x``, i.e. centroid times measure),
    optional second moment ``second_moments[k]`` (the integral of ``x x^T``), and
    optional simplex partition ``simplices[simplex_offsets[k]:simplex_offsets[k +
    1]]`` whose signed measures sum to ``volumes[k]``.  Rows list source cells in
    ascending order.  Cells are indexed by their position in the concatenated
    mesh blocks.  Entries exist whenever the narrow phase ran; consumers must
    check :attr:`status`.

    Identity fingerprints the canonical stored arrays: cell indices and CSR
    offsets use int32, while overlap volumes and first moments use float64.
    Cell indices must have an integer dtype and lie within the mesh and int32
    range before conversion; replay requires the exact stored dtypes and values.
    """

    status: CommonRefinementStatus = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    source_cell_count: int = eqx.field(static=True)
    target_cell_count: int = eqx.field(static=True)
    target_offsets: Array
    target_cells: Array
    source_cells: Array
    volumes: Array
    first_moments: Array
    second_moments: Array | None
    simplex_offsets: Array | None
    simplices: Array | None
    source_measures: Array
    source_first_moments: Array
    target_measures: Array
    target_first_moments: Array
    source_cell_global_ids: Array
    target_cell_global_ids: Array
    source_mesh_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    volume_error_bounds: Array | None
    target_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    policy: CommonRefinementPolicy
    evidence: CommonRefinementEvidence
    refinement_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        source: Decomposition,
        target: Decomposition,
        entries: Entries,
        identities: tuple[str, str, str, str],
        policy: CommonRefinementPolicy,
        evidence: CommonRefinementEvidence,
        geometry_ids: tuple[str, str] | None = None,
        volume_error_bounds: np.ndarray | None = None,
    ) -> None:
        dimension = source.first_moments.shape[1]
        count = entries.volumes.shape[0]
        if count > np.iinfo(np.int32).max:
            raise ValueError(
                "PreparedCommonRefinement.entries must fit int32 CSR offsets; got too many entries."
            )
        source_cells = _canonical_cell_indices(
            entries.source_cells, source.cell_count, "source_cells"
        )
        target_cells = _canonical_cell_indices(
            entries.target_cells, target.cell_count, "target_cells"
        )
        volumes = np.asarray(entries.volumes, dtype=np.float64)
        first_moments = np.asarray(entries.first_moments, dtype=np.float64)
        if (
            source_cells.shape != (count,)
            or target_cells.shape != (count,)
            or first_moments.shape != (count, dimension)
            or np.any(np.diff(target_cells) < 0)
        ):
            raise ValueError(
                "PreparedCommonRefinement.entries must be target-grouped CSR; got incompatible arrays or row ordering."
            )
        offsets = np.zeros((target.cell_count + 1,), dtype=np.int32)
        np.cumsum(
            np.bincount(target_cells, minlength=target.cell_count),
            out=offsets[1:],
        )
        self.status = evidence.status
        self.dimension = dimension
        self.source_cell_count = source.cell_count
        self.target_cell_count = target.cell_count
        self.target_offsets = jax.device_put(offsets)
        self.target_cells = jax.device_put(target_cells)
        self.source_cells = jax.device_put(source_cells)
        self.volumes = jax.device_put(volumes)
        self.first_moments = jax.device_put(first_moments)
        self.second_moments = (
            None
            if entries.second_moments is None
            else jax.device_put(np.asarray(entries.second_moments, dtype=np.float64))
        )
        self.simplex_offsets = (
            None
            if entries.simplex_offsets is None
            else jax.device_put(np.asarray(entries.simplex_offsets, dtype=np.int32))
        )
        self.simplices = (
            None
            if entries.simplices is None
            else jax.device_put(np.asarray(entries.simplices, dtype=np.float64))
        )
        self.source_measures = jax.device_put(np.asarray(source.measures))
        self.source_first_moments = jax.device_put(np.asarray(source.first_moments))
        self.target_measures = jax.device_put(np.asarray(target.measures))
        self.target_first_moments = jax.device_put(np.asarray(target.first_moments))
        self.source_cell_global_ids = jax.device_put(
            np.asarray(source.global_ids, dtype=np.int64)
        )
        self.target_cell_global_ids = jax.device_put(
            np.asarray(target.global_ids, dtype=np.int64)
        )
        (
            self.source_mesh_id,
            self.target_mesh_id,
            self.source_topology_id,
            self.target_topology_id,
        ) = identities
        self.source_geometry_id, self.target_geometry_id = (
            (identities[0], identities[1]) if geometry_ids is None else geometry_ids
        )
        errors = (
            None
            if volume_error_bounds is None
            else np.asarray(volume_error_bounds, dtype=np.float64)
        )
        if errors is not None and (
            errors.shape != (count,)
            or np.any(errors < 0)
            or not np.all(np.isfinite(errors))
        ):
            raise ValueError(
                "PreparedCommonRefinement.volume_error_bounds must be finite, nonnegative, and align with overlap entries; got incompatible bounds."
            )
        self.volume_error_bounds = (
            None if errors is None else jax.device_put(np.asarray(errors))
        )
        self.policy = policy
        self.evidence = evidence
        self.refinement_id = canonical_fingerprint(
            {
                "kind": "prepared-common-refinement",
                "source_mesh": identities[0],
                "target_mesh": identities[1],
                "source_geometry": self.source_geometry_id,
                "target_geometry": self.target_geometry_id,
                "volume_errors": array_tree_fingerprint(errors),
                "policy": policy.policy_id,
                "evidence": evidence.evidence_id,
                "target_offsets": array_tree_fingerprint(offsets),
                "source_cells": array_tree_fingerprint(source_cells),
                "volumes": array_tree_fingerprint(volumes),
                "first_moments": array_tree_fingerprint(first_moments),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PreparedCommonRefinement],
        value: PreparedCommonRefinement,
        /,
        *,
        source_mesh: CellMesh,
        target_mesh: CellMesh,
        source_geometry: CellGeometrySpec | None = None,
        target_geometry: CellGeometrySpec | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> PreparedCommonRefinement:
        """Reprove science, then canonically replay the historical public record.

        Every field must reproduce bitwise except the fresh phase's observed
        ``evidence.working_bytes`` and its dependent evidence/refinement IDs.
        Those historical fields are reconstructed, never overwritten with new
        observations. A supplied request retains the independently earned fresh
        evidence through sealing; standalone validation is invocation-local.
        """
        if type(value) is not cls:
            raise TypeError(
                "PreparedCommonRefinement.from_stored_fields.value must have the registered refinement type; "
                f"got {type(value).__name__}."
            )
        if (
            source_mesh.mesh_id != value.source_mesh_id
            or target_mesh.mesh_id != value.target_mesh_id
            or source_mesh.topology_id != value.source_topology_id
            or target_mesh.topology_id != value.target_topology_id
        ):
            raise ValueError(
                "PreparedCommonRefinement.meshes must retain its source and target identities; got different mesh bindings."
            )
        if source_geometry is None and value.source_geometry_id != source_mesh.mesh_id:
            raise ValueError(
                "PreparedCommonRefinement.source_geometry must retain its declared construction input; got missing geometry."
            )
        if target_geometry is None and value.target_geometry_id != target_mesh.mesh_id:
            raise ValueError(
                "PreparedCommonRefinement.target_geometry must retain its declared construction input; got missing geometry."
            )
        policy = CommonRefinementPolicy(
            **{
                member.name: getattr(value.policy, member.name)
                for member in fields(value.policy)
                if member.name != "policy_id"
            }
        )
        if not tree_content_equal(value.policy, policy):
            raise ValueError(
                "stored_fields.PreparedCommonRefinement must reproduce its stored "
                "content; got changed policy."
            )
        if budget is not None and budget.ticket.settled:
            scope = meshcore_owner.current_native_execution_scope()
            if scope is None:
                ticket = HostBatchTicket(
                    policy.maximum_exact_work,
                    queries_upper=policy.maximum_candidate_pairs,
                    bytes_upper=policy.maximum_memory_bytes,
                )
            else:
                remaining = scope.remaining()
                ticket = scope.batch(
                    min(policy.maximum_exact_work, remaining.remaining_work_units),
                    queries_upper=min(
                        policy.maximum_exact_work, remaining.remaining_geometry_queries
                    ),
                    bytes_upper=min(
                        policy.maximum_memory_bytes, remaining.remaining_managed_bytes
                    ),
                )
            with budget.bind_ticket(ticket), budget.activate():
                return cls.from_stored_fields(
                    value,
                    source_mesh=source_mesh,
                    target_mesh=target_mesh,
                    source_geometry=source_geometry,
                    target_geometry=target_geometry,
                    budget=budget,
                )
        fresh = prepare_common_refinement(
            source_mesh,
            target_mesh,
            policy=policy,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            budget=budget,
        )
        historical_evidence = CommonRefinementEvidence.from_stored_fields(
            value.evidence,
            source_measures=fresh.source_measures,
            target_measures=fresh.target_measures,
        )
        for member in fields(fresh.evidence):
            if member.name in ("working_bytes", "evidence_id"):
                continue
            if not tree_content_equal(
                getattr(historical_evidence, member.name),
                getattr(fresh.evidence, member.name),
            ):
                raise ValueError(
                    "CommonRefinementEvidence.stored_fields must reproduce fresh "
                    f"scientific evidence; got changed {member.name}."
                )
        for member in fields(fresh):
            if member.name in ("evidence", "refinement_id"):
                continue
            if not tree_content_equal(
                getattr(value, member.name), getattr(fresh, member.name)
            ):
                raise ValueError(
                    "stored_fields.PreparedCommonRefinement must reproduce its "
                    f"stored content; got changed {member.name}."
                )
        historical_id = canonical_fingerprint(
            {
                "kind": "prepared-common-refinement",
                "source_mesh": fresh.source_mesh_id,
                "target_mesh": fresh.target_mesh_id,
                "source_geometry": fresh.source_geometry_id,
                "target_geometry": fresh.target_geometry_id,
                "volume_errors": array_tree_fingerprint(fresh.volume_error_bounds),
                "policy": policy.policy_id,
                "evidence": historical_evidence.evidence_id,
                "target_offsets": array_tree_fingerprint(fresh.target_offsets),
                "source_cells": array_tree_fingerprint(fresh.source_cells),
                "volumes": array_tree_fingerprint(fresh.volumes),
                "first_moments": array_tree_fingerprint(fresh.first_moments),
            }
        )
        if value.refinement_id != historical_id:
            raise ValueError(
                "stored_fields.PreparedCommonRefinement must reproduce its stored "
                "content; got changed refinement_id."
            )
        historical = eqx.tree_at(
            lambda record: record.evidence, value, historical_evidence
        )
        if not tree_content_equal(value, historical):
            raise ValueError(
                "stored_fields.PreparedCommonRefinement must reproduce its stored "
                "content; got changed refinement_id."
            )
        if budget is not None:
            budget.retain_owner(historical)
        return historical

    @property
    def succeeded(self) -> bool:
        return self.status is CommonRefinementStatus.SUCCESS

    @property
    def entry_count(self) -> int:
        return self.volumes.shape[0]


# ---------------------------------------------------------------- host records


class Counts(NamedTuple):
    candidates: int
    accepted: int
    piece_pairs: int
    uncertain: int
    invalid: int
    failures: int


class Decomposition(NamedTuple):
    """Certified convex pieces of every cell of one mesh (host arrays)."""

    cell_count: int
    global_ids: np.ndarray
    # 2-D: (P, K, 2) padded convex polygons with `piece_counts`; 3-D: (P, 4, 3).
    pieces: np.ndarray
    piece_counts: np.ndarray | None
    piece_offsets: np.ndarray
    measures: np.ndarray
    first_moments: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    invalid: int
    uncertain: int


class Entries(NamedTuple):
    target_cells: np.ndarray
    source_cells: np.ndarray
    volumes: np.ndarray
    first_moments: np.ndarray
    second_moments: np.ndarray | None
    simplex_offsets: np.ndarray | None
    simplices: np.ndarray | None


class _PieceSigns(NamedTuple):
    signs: np.ndarray
    certain: np.ndarray


class _RefinementStorage:
    """Preadmit host banks on one live request lease, including metadata."""

    def __init__(
        self, budget: CoordinateEnclosureBudget, lease: CoordinateLiveStorage, /
    ) -> None:
        lease.set_bound(8192)
        self.budget = budget
        self.lease = lease
        self.counts = Counts(0, 0, 0, 0, 0, 0)
        self.banks: dict[str, int] = {}
        self.peak = 8192

    def set(self, name: str, size: int, /) -> None:
        total = 8192 + sum(self.banks.values()) - self.banks.get(name, 0) + size
        self.lease.set_bound(total)
        self.banks[name] = size
        self.peak = max(self.peak, total)

    def transfer(self, source: str, target: str, size: int, /) -> None:
        """Move live aliases between banks without releasing their admission."""
        available = self.banks[source]
        if size < 0 or size > available:
            raise RuntimeError(
                "CommonRefinement.storage_transfer must fit its admitted source; "
                f"got requested={size}, available={available}."
            )
        self.banks[source] = available - size
        self.banks[target] = self.banks.get(target, 0) + size

    @contextmanager
    def scratch(self, name: str, size: int, /) -> Iterator[None]:
        previous = self.banks.get(name, 0)
        self.set(name, size)
        try:
            yield
        finally:
            self.set(name, previous)

    def publish(self, result: PreparedCommonRefinement, /) -> PreparedCommonRefinement:
        self.budget.retain_owner(result, temporary_storage_credit=self.lease)
        return result


def _array_bank_bytes(arrays: Iterable[np.ndarray | None], /) -> int:
    """Release-only measurement of banks whose construction was preadmitted."""
    return sum(256 + array.nbytes for array in arrays if array is not None)


def _decomposition_bytes(part: Decomposition, /) -> int:
    return 1024 + _array_bank_bytes(part[1:9])


def _decomposition_bound(mesh: CellMesh, storage: _RefinementStorage, /) -> int:
    """Derive cone/padding bounds from actual row and face incidences."""
    connectivity = mesh.connectivity
    polyhedral = isinstance(connectivity, cell_complex_owner.PolyhedralConnectivity)
    scan = 2048
    if polyhedral:
        scan += 16 * (
            connectivity.cell_face_offsets.size
            + connectivity.cell_face_values.size
            + connectivity.face_vertex_offsets.size
        )
    with storage.scratch("incidence-scan", scan):
        if polyhedral:
            cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
            cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
            face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        dimension = mesh.topological_dimension
        cells = pieces = padded_facets = vertices = width = first = 0
        for block in mesh.blocks:
            count, arity = block.vertices.shape
            cells += count
            vertices += count * arity
            if dimension == 2:
                pieces += count * arity
                padded_facets += count * arity
                width = max(width, arity, 3)
            elif block.cell_kind == "polyhedron":
                maximum = total = 0
                for row in range(first, first + count):
                    triangles = 0
                    for slot in range(int(cell_offsets[row]), int(cell_offsets[row + 1])):
                        face = int(cell_faces[slot])
                        triangles += int(face_offsets[face + 1] - face_offsets[face]) - 2
                    maximum = max(maximum, triangles)
                    total += triangles
                padded_facets += count * maximum
                pieces += total
            else:
                triangles = sum(
                    len(route) - 2
                    for route in cell_complex_owner.POLYHEDRAL_FACE_ROUTES[
                        block.cell_kind
                    ]
                )
                padded_facets += count * triangles
                pieces += count * triangles
            first += count
        # Block parts coexist with joined/sorted banks. Cone workspace has
        # <= padded_facets rows: 128 float64-equivalent lanes cover masks,
        # nonzero indices, gathered points and orientation-filter temporaries.
        # Moments and at most twelve joined/indexed piece banks coexist.
        piece_width = width * dimension if dimension == 2 else 12
        scalar = declared_integer_storage_bytes(
            max(
                cells, pieces, padded_facets, vertices, mesh.coordinates.size
            ).bit_length()
        )
        metadata = 16384 + len(mesh.blocks) * (128 * 256 + 64 * scalar)
        return (
            metadata
            + 32 * mesh.coordinates.size
            + 8 * (12 * vertices + 128 * padded_facets + 32 * cells)
            + 8 * pieces * (12 * piece_width + 32 + 16 * dimension)
        )


def _bvh_storage_bound(count: int, dimension: int, /) -> int:
    # Median tree: <= 2N-1 nodes and <= N*leaf_size payload slots.
    # Levels, sort keys, casts, packed logical JAX banks and traversal host
    # conversions coexist until the candidate stream closes.
    nodes = 2 * count - 1
    depth = max(1, count.bit_length())
    return (
        16384
        + 256 * (64 + 8 * depth)
        + 8 * (count * (48 + 16 * dimension) + nodes * (40 + 8 * dimension))
        + 2 * count * 16 * 8
    )


# ---------------------------------------------------------------- decomposition


def _orientation(
    points: tuple[np.ndarray, ...],
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> _PieceSigns:
    match len(points):
        case 3:
            result = orient2d_host(
                *points, mode=mode, budget=budget, memoize_native=False
            )
        case 4:
            result = orient3d_host(
                *points, mode=mode, budget=budget, memoize_native=False
            )
        case _:
            raise ValueError(
                f"CommonRefinement.orientation_points must contain three or four points; got {len(points)}."
            )
    return _PieceSigns(np.asarray(result.signs), np.asarray(result.certain))


def cone_decomposition(
    coordinates: np.ndarray,
    facets: np.ndarray,
    facet_valid: np.ndarray,
    candidates: np.ndarray,
    candidate_valid: np.ndarray,
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Certify cones of oriented boundary facets ``(n, F, d)`` from cell vertices.

    Returns the chosen apex per cell, the kept (nondegenerate) facet mask, and
    the invalid and uncertain cell masks.  Candidates are tried in column order;
    facets containing the apex are degenerate and skipped.
    """

    cells = facets.shape[0]
    apex = np.full((cells,), -1, dtype=np.int64)
    keep = np.zeros(facets.shape[:2], dtype=np.bool_)
    uncertain_seen = np.zeros((cells,), dtype=np.bool_)
    unresolved = np.arange(cells, dtype=np.int64)
    # Bounded static loop over the candidate width (cell vertex count).
    for column in range(candidates.shape[1]):
        if unresolved.size == 0:
            break
        tip = candidates[unresolved, column]
        local = facets[unresolved]
        live = facet_valid[unresolved] & ~np.any(local == tip[:, None, None], axis=2)
        rows, slots = np.nonzero(live)
        signs = np.zeros(live.shape, dtype=np.int8)
        certain = np.ones(live.shape, dtype=np.bool_)
        if rows.size:
            evaluated = _orientation(
                (coordinates[tip[rows]],)
                + tuple(
                    coordinates[local[rows, slots, k]] for k in range(facets.shape[2])
                ),
                mode,
                budget=budget,
            )
            signs[rows, slots] = evaluated.signs
            certain[rows, slots] = evaluated.certain
        known = live & certain
        positive = np.any(known & (signs == 1), axis=1)
        negative = np.any(known & (signs == -1), axis=1)
        undecided = np.any(live & ~certain, axis=1)
        conflict = positive & negative
        accepted = (
            candidate_valid[unresolved, column]
            & ~conflict
            & ~undecided
            & (positive | negative)
        )
        uncertain_seen[unresolved] |= undecided & ~conflict
        chosen = unresolved[accepted]
        apex[chosen] = tip[accepted]
        keep[chosen] = known[accepted] & (signs[accepted] != 0)
        unresolved = unresolved[~accepted]
    failed = np.zeros((cells,), dtype=np.bool_)
    failed[unresolved] = True
    return apex, keep, failed & ~uncertain_seen, failed & uncertain_seen


def _polygon_moments(
    vertices: np.ndarray, counts: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Unsigned area and first moment of padded convex polygons ``(P, K, 2)``."""

    width = vertices.shape[1]
    slots = np.arange(width, dtype=np.int64)
    present = slots[None, :] < counts[:, None]
    origin = np.sum(np.where(present[..., None], vertices, 0.0), axis=1) / counts[
        :, None
    ].astype(np.float64)
    relative = vertices - origin[:, None, :]
    following = np.take_along_axis(
        relative, ((slots[None, :] + 1) % counts[:, None])[..., None], axis=1
    )
    cross = np.where(
        present,
        relative[..., 0] * following[..., 1] - following[..., 0] * relative[..., 1],
        0.0,
    )
    area = 0.5 * np.sum(cross, axis=1)
    moment = (
        np.sum((relative + following) * cross[..., None], axis=1) / 6.0
        + origin * area[:, None]
    )
    sign = np.sign(area)
    return np.abs(area), moment * sign[:, None]


def _tetrahedron_moments(tetrahedra: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    edges = tetrahedra[:, 1:] - tetrahedra[:, :1]
    six = contract("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))
    volume = np.abs(six) / 6.0
    return volume, volume[:, None] * np.mean(tetrahedra, axis=1)


def _cell_bounds(
    coordinates: np.ndarray, cells: np.ndarray, valid: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    points = coordinates[np.where(valid, cells, cells[:, :1])]
    return np.min(points, axis=1), np.max(points, axis=1)


class _BlockPieces(NamedTuple):
    cells: np.ndarray
    vertices: np.ndarray
    counts: np.ndarray | None
    invalid: np.ndarray
    uncertain: np.ndarray


def _polygon_block_pieces(
    coordinates: np.ndarray,
    cells: np.ndarray,
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> _BlockPieces:
    """Convex cells stay whole; other polygons become certified triangle cones."""

    count, arity = cells.shape
    previous = np.roll(cells, 1, axis=1)
    following = np.roll(cells, -1, axis=1)
    turns = _orientation(
        (coordinates[previous], coordinates[cells], coordinates[following]),
        mode,
        budget=budget,
    )
    known = turns.certain
    positive = np.any(known & (turns.signs == 1), axis=1)
    negative = np.any(known & (turns.signs == -1), axis=1)
    undecided = np.any(~known, axis=1)
    convex = (positive ^ negative) & ~undecided
    rows = np.arange(count, dtype=np.int64)
    whole = rows[convex]
    concave = rows[~convex]
    edges = np.stack((cells[concave], np.roll(cells[concave], -1, axis=1)), axis=2)
    apex, keep, invalid, uncertain = cone_decomposition(
        coordinates,
        edges,
        np.ones(edges.shape[:2], dtype=np.bool_),
        np.sort(cells[concave], axis=1),
        np.ones((concave.size, arity), dtype=np.bool_),
        mode,
        budget=budget,
    )
    cone_rows, cone_slots = np.nonzero(keep)
    triangles = np.stack(
        (
            apex[cone_rows],
            edges[cone_rows, cone_slots, 0],
            edges[cone_rows, cone_slots, 1],
        ),
        axis=1,
    )
    width = max(arity, 3)
    padded = np.zeros((whole.size + triangles.shape[0], width, 2), dtype=np.float64)
    padded[: whole.size, :arity] = coordinates[cells[whole]]
    padded[whole.size :, :3] = coordinates[triangles]
    piece_cells = np.concatenate((whole, concave[cone_rows]))
    piece_counts = np.concatenate(
        (
            np.full((whole.size,), arity, dtype=np.int32),
            np.full((triangles.shape[0],), 3, dtype=np.int32),
        )
    )
    failed_invalid = np.zeros((count,), dtype=np.bool_)
    failed_uncertain = np.zeros((count,), dtype=np.bool_)
    failed_invalid[concave] = invalid
    failed_uncertain[concave] = uncertain
    return _BlockPieces(
        piece_cells, padded, piece_counts, failed_invalid, failed_uncertain
    )


def _standard_face_triangles(cells: np.ndarray, kind: str, /) -> np.ndarray:
    """Oriented face triangles ``(n, T, 3)`` of standard volume cells."""

    triangles = []
    for route in cell_complex_owner.POLYHEDRAL_FACE_ROUTES[kind]:
        loops = cells[:, route]
        size = len(route)
        start = np.argmin(loops, axis=1)
        rotated = np.take_along_axis(
            loops, (start[:, None] + np.arange(size, dtype=np.int64)) % size, axis=1
        )
        triangles.append(
            np.stack(
                (
                    np.repeat(rotated[:, :1], size - 2, axis=1),
                    rotated[:, 1:-1],
                    rotated[:, 2:],
                ),
                axis=2,
            )
        )
    return np.concatenate(triangles, axis=1)


@checked
def polyhedral_face_triangles(
    connectivity: cell_complex_owner.PolyhedralConnectivity,
    first_cell: int,
    cell_count: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-oriented face triangles of explicit polyhedra from packed incidence."""

    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    begin = cell_face_offsets[first_cell]
    end = cell_face_offsets[first_cell + cell_count]
    faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)[begin:end]
    signs = np.asarray(connectivity.cell_face_sign_values)[begin:end]
    owner = np.repeat(
        np.arange(cell_count, dtype=np.int64),
        np.diff(cell_face_offsets[first_cell : first_cell + cell_count + 1]),
    )
    sizes = face_offsets[faces + 1] - face_offsets[faces]
    starts = face_offsets[faces]
    incidence = np.repeat(np.arange(faces.size, dtype=np.int64), sizes)
    local = np.arange(incidence.size, dtype=np.int64) - np.repeat(
        np.cumsum(sizes) - sizes, sizes
    )
    values = face_values[starts[incidence] + local]
    smallest = np.minimum.reduceat(values, np.cumsum(sizes) - sizes)
    anchor = local[values == smallest[incidence]]
    triangle_incidence = np.repeat(np.arange(faces.size, dtype=np.int64), sizes - 2)
    step = (
        np.arange(triangle_incidence.size, dtype=np.int64)
        - np.repeat(np.cumsum(sizes - 2) - (sizes - 2), sizes - 2)
        + 1
    )
    direction = np.where(signs[triangle_incidence] < 0, -1, 1)
    size = sizes[triangle_incidence]
    origin = starts[triangle_incidence]
    corner = anchor[triangle_incidence]

    def vertex(offset: np.ndarray) -> np.ndarray:
        return face_values[origin + (corner + direction * offset) % size]

    triangles = np.stack(
        (vertex(np.zeros_like(step)), vertex(step), vertex(step + 1)), axis=1
    )
    triangle_cells = owner[triangle_incidence]
    per_cell = np.bincount(triangle_cells, minlength=cell_count)
    width = int(np.max(per_cell))
    slot = np.arange(triangle_cells.size, dtype=np.int64) - np.repeat(
        np.cumsum(per_cell) - per_cell, per_cell
    )
    padded = np.zeros((cell_count, width, 3), dtype=np.int64)
    valid = np.zeros((cell_count, width), dtype=np.bool_)
    padded[triangle_cells, slot] = triangles
    valid[triangle_cells, slot] = True
    return padded, valid


def _volume_block_pieces(
    coordinates: np.ndarray,
    cells: np.ndarray,
    vertex_valid: np.ndarray,
    triangles: np.ndarray,
    triangle_valid: np.ndarray,
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> _BlockPieces:
    ordered = np.sort(np.where(vertex_valid, cells, np.iinfo(np.int64).max), axis=1)
    candidate_valid = ordered != np.iinfo(np.int64).max
    apex, keep, invalid, uncertain = cone_decomposition(
        coordinates,
        triangles,
        triangle_valid,
        np.where(candidate_valid, ordered, 0),
        candidate_valid,
        mode,
        budget=budget,
    )
    rows, slots = np.nonzero(keep)
    tetrahedra = np.concatenate((apex[rows, None], triangles[rows, slots]), axis=1)
    return _BlockPieces(rows, coordinates[tetrahedra], None, invalid, uncertain)


def _block_pieces(
    mesh: CellMesh,
    block: CellBlock | PolyhedralBlock,
    first_cell: int,
    coordinates: np.ndarray,
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> _BlockPieces:
    cells = np.asarray(block.vertices, dtype=np.int64)
    match block.cell_kind:
        case "triangle" | "quadrilateral" | "polygon":
            return _polygon_block_pieces(coordinates, cells, mode, budget=budget)
        case "tetrahedron" | "hexahedron" | "prism" | "pyramid":
            triangles = _standard_face_triangles(cells, block.cell_kind)
            triangle_valid = np.ones(triangles.shape[:2], dtype=np.bool_)
        case "polyhedron":
            if not isinstance(
                mesh.connectivity, cell_complex_owner.PolyhedralConnectivity
            ):
                raise ValueError(
                    "CommonRefinement.connectivity must be polyhedral for polyhedral blocks; got a different connectivity kind."
                )
            identifiers = np.asarray(mesh.connectivity.cell_global_ids, dtype=np.int64)
            if not np.array_equal(
                identifiers[first_cell : first_cell + block.cell_count],
                np.asarray(block.global_ids, dtype=np.int64),
            ):
                raise ValueError(
                    "CommonRefinement.connectivity.cell_global_ids must follow the mesh block order; got different identifiers."
                )
            triangles, triangle_valid = polyhedral_face_triangles(
                mesh.connectivity, first_cell, block.cell_count
            )
        case _:
            raise ValueError(
                f"CommonRefinement.cell_kind must support convex decomposition; got {block.cell_kind!r}."
            )
    return _volume_block_pieces(
        coordinates,
        cells,
        np.asarray(block.vertex_valid, dtype=np.bool_),
        triangles,
        triangle_valid,
        mode,
        budget=budget,
    )


def _decompose(
    mesh: CellMesh,
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> Decomposition:
    coordinates = np.asarray(jax.device_get(mesh.coordinates), dtype=np.float64)
    dimension = mesh.topological_dimension
    cell_parts, vertex_parts, count_parts = [], [], []
    minima, maxima, invalid, uncertain, identifiers = [], [], [], [], []
    first_cell = 0
    for block in mesh.blocks:
        pieces = _block_pieces(mesh, block, first_cell, coordinates, mode, budget=budget)
        cells = np.asarray(block.vertices, dtype=np.int64)
        lower, upper = _cell_bounds(
            coordinates, cells, np.asarray(block.vertex_valid, dtype=np.bool_)
        )
        cell_parts.append(pieces.cells + first_cell)
        vertex_parts.append(pieces.vertices)
        count_parts.append(pieces.counts)
        minima.append(lower)
        maxima.append(upper)
        invalid.append(pieces.invalid)
        uncertain.append(pieces.uncertain)
        identifiers.append(np.asarray(block.global_ids, dtype=np.int64))
        first_cell += block.cell_count
    piece_cells = np.concatenate(cell_parts)
    order = np.argsort(piece_cells, kind="stable")
    piece_cells = piece_cells[order]
    if dimension == 2:
        width = max(part.shape[1] for part in vertex_parts)
        padded = np.zeros((piece_cells.size, width, 2), dtype=np.float64)
        start = 0
        for part in vertex_parts:
            padded[start : start + part.shape[0], : part.shape[1]] = part
            start += part.shape[0]
        pieces = padded[order]
        piece_counts = np.concatenate(count_parts)[order]
        piece_measures, piece_moments = _polygon_moments(pieces, piece_counts)
    else:
        pieces = np.concatenate(vertex_parts)[order]
        piece_counts = None
        piece_measures, piece_moments = _tetrahedron_moments(pieces)
    cell_count = first_cell
    offsets = np.zeros((cell_count + 1,), dtype=np.int64)
    np.cumsum(np.bincount(piece_cells, minlength=cell_count), out=offsets[1:])
    measures = np.bincount(piece_cells, weights=piece_measures, minlength=cell_count)
    moments = np.stack(
        [
            np.bincount(piece_cells, weights=piece_moments[:, k], minlength=cell_count)
            for k in range(dimension)
        ],
        axis=1,
    )
    return Decomposition(
        cell_count=cell_count,
        global_ids=np.concatenate(identifiers),
        pieces=pieces,
        piece_counts=piece_counts,
        piece_offsets=offsets,
        measures=measures,
        first_moments=moments,
        bbox_min=np.concatenate(minima),
        bbox_max=np.concatenate(maxima),
        invalid=int(np.sum(np.concatenate(invalid))),
        uncertain=int(np.sum(np.concatenate(uncertain))),
    )


def _outside_exact_domain(mesh: CellMesh, /) -> bool:
    minimum, maximum = meshcore_owner.meshcore_exact_domain()
    magnitude = np.abs(np.asarray(jax.device_get(mesh.coordinates), dtype=np.float64))
    nonzero = magnitude[magnitude != 0.0]
    return bool(
        np.any(nonzero < np.ldexp(1.0, minimum))
        or np.any(nonzero > np.ldexp(1.0, maximum))
    )


# ---------------------------------------------------------------- broad phase


def _candidate_pairs(
    source: Decomposition,
    target: Decomposition,
    limit: int,
    storage: _RefinementStorage,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Preadmit BVHs and stream blocks, then sort only actual candidate rows."""
    dimension = source.first_moments.shape[1]
    bvh_bytes = _bvh_storage_bound(source.cell_count, dimension) + _bvh_storage_bound(
        target.cell_count, dimension
    )
    with storage.scratch("bvh", bvh_bytes), storage.scratch("traversal", 0):
        target_bvh = prepare_bvh(target.bbox_min, target.bbox_max, dtype=jnp.float64)
        source_bvh = prepare_bvh(source.bbox_min, source.bbox_max, dtype=jnp.float64)
        targets, sources = [], []
        total = 0
        storage.set("candidates", 2048)
        with closing(
            bvh_overlap_pair_blocks(
                target_bvh,
                source_bvh,
                traversal_storage=lambda size: storage.set("traversal", size),
            )
        ) as blocks:
            for target_items, source_items in blocks:
                total += target_items.size
                storage.counts = storage.counts._replace(candidates=total)
                if total > limit:
                    del target_items, source_items
                    return None
                # The yielded copies were already admitted by traversal. Move
                # their ownership without charging the same live aliases twice,
                # then admit retained list/array metadata before appending.
                storage.transfer(
                    "traversal",
                    "candidates",
                    _array_bank_bytes((target_items, source_items)),
                )
                storage.set("candidates", 2048 + 16 * total + 1024 * (len(targets) + 1))
                targets.append(target_items)
                sources.append(source_items)
        # The iterator is closed, so no frontier or leaf scratch survives while
        # joins, sort keys and indexed sorted copies are allocated.
        storage.set("traversal", 0)
        storage.set("candidates", 2048 + 80 * total + 1024 * len(targets))
        if not targets:
            empty = np.empty((0,), dtype=np.int64)
            return empty, empty.copy()
        target_cells = np.concatenate(targets)
        source_cells = np.concatenate(sources)
        order = np.lexsort((source_cells, target_cells))
        return target_cells[order], source_cells[order]


# ---------------------------------------------------------------- narrow phase


class _SimplexPartition(StrictModule):
    simplices: np.ndarray
    counts: np.ndarray


class _Batch(StrictModule):
    pair: np.ndarray
    volume: np.ndarray
    moment: np.ndarray
    status: np.ndarray
    partition: _SimplexPartition | None


def _clip_batch(
    source: Decomposition,
    target: Decomposition,
    target_cells: np.ndarray,
    source_cells: np.ndarray,
    simplices: bool,
    /,
) -> _Batch:
    """Clip every piece pair of the given cell pairs with meshcore."""

    source_counts = np.diff(source.piece_offsets)[source_cells]
    target_counts = np.diff(target.piece_offsets)[target_cells]
    per_pair = source_counts * target_counts
    pair = np.repeat(np.arange(target_cells.size, dtype=np.int64), per_pair)
    local = np.arange(pair.size, dtype=np.int64) - np.repeat(
        np.cumsum(per_pair) - per_pair, per_pair
    )
    source_piece = source.piece_offsets[source_cells][pair] + local // target_counts[pair]
    target_piece = target.piece_offsets[target_cells][pair] + local % target_counts[pair]
    match (source.piece_counts, target.piece_counts):
        case (None, None):
            first = source.pieces[source_piece]
            second = target.pieces[target_piece]
            if simplices:
                native = meshcore_owner.tetrahedron_intersection_simplices(first, second)
                return _Batch(
                    pair,
                    native.volume,
                    native.first_moment,
                    native.status,
                    _SimplexPartition(native.simplices, native.simplex_counts),
                )
            volume, moment, status = meshcore_owner.tetrahedron_intersection_moments(
                first, second
            )
            return _Batch(pair, volume, moment, status, None)
        case (np.ndarray() as source_counts, np.ndarray() as target_counts):
            arguments = (
                source.pieces[source_piece],
                source_counts[source_piece],
                target.pieces[target_piece],
                target_counts[target_piece],
            )
        case _:
            raise ValueError(
                "CommonRefinement.piece_counts must describe matching decomposition dimensions; got different banks."
            )
    if simplices:
        native = meshcore_owner.polygon_intersection_simplices(*arguments)
        return _Batch(
            pair,
            native.area,
            native.first_moment,
            native.status,
            _SimplexPartition(native.simplices, native.simplex_counts),
        )
    volume, moment, status = meshcore_owner.polygon_intersection_moments(*arguments)
    return _Batch(pair, volume, moment, status, None)


def _simplex_second_moments(simplices: np.ndarray, /) -> np.ndarray:
    """Integral of ``x x^T`` over signed simplices ``(S, d + 1, d)``."""

    dimension = simplices.shape[2]
    edges = simplices[:, 1:] - simplices[:, :1]
    match dimension:
        case 2:
            measure = 0.5 * (
                edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
            )
            scale = measure / 12.0
        case 3:
            measure = (
                contract("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))
                / 6.0
            )
            scale = measure / 20.0
        case _:
            raise ValueError(
                f"CommonRefinement.simplex_dimension must be 2 or 3; got {dimension}."
            )
    total = np.sum(simplices, axis=1)
    products = contract("skd,ske->sde", simplices, simplices) + contract(
        "sd,se->sde", total, total
    )
    return scale[:, None, None] * products


class _BatchResult(StrictModule):
    keep: np.ndarray
    volumes: np.ndarray
    moments: np.ndarray
    second: np.ndarray | None
    partition: _SimplexPartition | None
    uncertain: int = eqx.field(static=True)
    failures: int = eqx.field(static=True)


def _reduce_batch(
    batch: _Batch, pair_count: int, policy: CommonRefinementPolicy, dimension: int, /
) -> _BatchResult:
    failed = batch.status != int(meshcore_owner.MeshcoreStatus.OK)
    outside = (batch.status == int(meshcore_owner.MeshcoreStatus.RANGE_ERROR)) | (
        batch.status == int(meshcore_owner.MeshcoreStatus.NONFINITE_INPUT)
    )
    volumes = np.bincount(batch.pair, weights=batch.volume, minlength=pair_count)
    moments = np.stack(
        [
            np.bincount(batch.pair, weights=batch.moment[:, k], minlength=pair_count)
            for k in range(dimension)
        ],
        axis=1,
    )
    keep = volumes > 0.0
    second = None
    kept_partition = None
    partition = batch.partition
    if partition is not None:
        slots = np.arange(partition.simplices.shape[1], dtype=np.int64)
        present = slots[None, :] < partition.counts[:, None]
        simplex_pair = np.broadcast_to(batch.pair[:, None], present.shape)[present]
        flat = partition.simplices[present]
        if policy.second_moments:
            products = _simplex_second_moments(flat).reshape(
                (flat.shape[0], dimension * dimension)
            )
            second = np.stack(
                [
                    np.bincount(
                        simplex_pair, weights=products[:, k], minlength=pair_count
                    )
                    for k in range(products.shape[1])
                ],
                axis=1,
            ).reshape((pair_count, dimension, dimension))[keep]
        if policy.overlap_simplices:
            retained = keep[simplex_pair]
            kept_partition = _SimplexPartition(
                flat[retained], np.bincount(simplex_pair, minlength=pair_count)[keep]
            )
    return _BatchResult(
        keep,
        volumes[keep],
        moments[keep],
        second,
        kept_partition,
        int(np.sum(outside)),
        int(np.sum(failed & ~outside)),
    )


def _batch_bounds(
    source: Decomposition,
    target: Decomposition,
    target_cells: np.ndarray,
    source_cells: np.ndarray,
    pair_bytes: int,
    /,
) -> np.ndarray:
    """Candidate boundaries with at most `_BATCH_BYTES` of piece pairs per batch."""

    per_pair = (
        np.diff(source.piece_offsets)[source_cells]
        * np.diff(target.piece_offsets)[target_cells]
    )
    cumulative = np.cumsum(per_pair * pair_bytes)
    capacity = max(_BATCH_BYTES, pair_bytes)
    marks = np.arange(
        capacity, cumulative[-1] if cumulative.size else 0, capacity, dtype=np.int64
    )
    bounds = np.searchsorted(cumulative, marks, side="right")
    return np.unique(np.concatenate(([0], bounds, [target_cells.size])))


def _piece_pair_bytes(
    source: Decomposition, target: Decomposition, policy: CommonRefinementPolicy, /
) -> int:
    simplices = policy.second_moments or policy.overlap_simplices
    if source.piece_counts is None:
        inputs = 2 * 12 * 8
        outputs = 8 + 3 * 8 + 4
        partition = _TETRAHEDRON_SIMPLICES * 12 * 8 + 4 if simplices else 0
    else:
        width = source.pieces.shape[1] + target.pieces.shape[1]
        inputs = width * 2 * 8 + 8
        outputs = 8 + 2 * 8 + 4
        partition = width * 6 * 8 + 4 if simplices else 0
    # Index expansion (pair, local, piece ids) accompanies every piece pair.
    return inputs + outputs + partition + 5 * 8


class _NarrowResult(NamedTuple):
    entries: Entries | None
    counts: Counts
    retained_bytes: int
    working_bytes: int


def _entries_bytes(parts: Iterable[_BatchResult], dimension: int, /) -> int:
    total = 0
    for part in parts:
        total += part.volumes.size * (4 + 4 + 8 + 8 * dimension)
        if part.second is not None:
            total += part.second.nbytes
        if part.partition is not None:
            total += part.partition.simplices.nbytes + 4 * part.partition.counts.size
    return total


def _piece_pairs_in_rows(
    source: Decomposition,
    target: Decomposition,
    target_cells: np.ndarray,
    source_cells: np.ndarray,
    start: int,
    stop: int,
    /,
) -> int:
    return sum(
        int(
            source.piece_offsets[int(source_cells[row]) + 1]
            - source.piece_offsets[int(source_cells[row])]
        )
        * int(
            target.piece_offsets[int(target_cells[row]) + 1]
            - target.piece_offsets[int(target_cells[row])]
        )
        for row in range(start, stop)
    )


def _narrow_phase(
    source: Decomposition,
    target: Decomposition,
    target_cells: np.ndarray,
    source_cells: np.ndarray,
    policy: CommonRefinementPolicy,
    base_bytes: int,
    /,
    *,
    storage: _RefinementStorage,
) -> _NarrowResult:
    dimension = source.first_moments.shape[1]
    pair_bytes = _piece_pair_bytes(source, target, policy)
    simplices = policy.second_moments or policy.overlap_simplices
    total_clips = _piece_pairs_in_rows(
        source, target, target_cells, source_cells, 0, target_cells.size
    )
    marks = max(0, (total_clips * pair_bytes - 1) // max(_BATCH_BYTES, pair_bytes))
    storage.set(
        "batch-boundaries",
        8192
        + 96 * target_cells.size
        + 16 * marks
        + 16 * (source.cell_count + target.cell_count),
    )
    bounds = _batch_bounds(source, target, target_cells, source_cells, pair_bytes)
    parts: list[_BatchResult] = []
    kept_pairs: list[np.ndarray] = []
    accepted = piece_pairs = uncertain = failures = 0
    working = base_bytes
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        clips = _piece_pairs_in_rows(
            source, target, target_cells, source_cells, int(start), int(stop)
        )
        # Admit gathered inputs and index expansion before clipping. Native
        # outputs stay on the original native owner; reduction does not reserve
        # nonexistent maximum-capacity simplex products before their counts exist.
        input_bytes = (
            2 * 12 * 8
            if dimension == 3
            else (source.pieces.shape[1] + target.pieces.shape[1]) * 2 * 8 + 8
        )
        rows = int(stop - start)
        batch_bytes = (
            65536
            + 8 * (source.cell_count + target.cell_count)
            + clips * (input_bytes + 192)
            + 128 * rows
        )
        with storage.scratch("clip-batch", batch_bytes):
            batch = _clip_batch(
                source,
                target,
                target_cells[start:stop],
                source_cells[start:stop],
                simplices,
            )
            piece_pairs += batch.pair.size
            working = max(working, base_bytes + batch.pair.size * pair_bytes)
            # Gathered inputs have expired. The only ordinary NumPy payload in
            # _Batch is pair; meshcore owns moment/status/simplex buffers.
            storage.set("clip-batch", 8192 + batch.pair.nbytes)
            partition = batch.partition
            capacity = 0 if partition is None else partition.simplices.shape[1]
            flat_count = (
                0 if partition is None else sum(int(count) for count in partition.counts)
            )
            reduction_bytes = (
                65536
                + 8 * capacity
                + 4 * clips
                + clips * capacity
                + 8 * rows * (8 + 4 * dimension)
                + flat_count * (8 * (2 + 3 * (dimension + 1) * dimension) + 1)
            )
            if policy.second_moments:
                reduction_bytes += 32 * rows * dimension**2 + 8 * flat_count * (
                    8 * dimension**2 + 8 * dimension + 16
                )
            with storage.scratch("clip-reduction", reduction_bytes):
                result = _reduce_batch(batch, rows, policy, dimension)
                accepted += result.volumes.size
                uncertain += result.uncertain
                failures += result.failures
                counts = Counts(
                    target_cells.size, accepted, piece_pairs, uncertain, 0, failures
                )
                storage.counts = counts
                if accepted > policy.maximum_accepted_pairs:
                    return _NarrowResult(
                        None,
                        counts,
                        _entries_bytes(parts, dimension)
                        + _entries_bytes((result,), dimension),
                        working,
                    )
                retained_part = 2048 + _array_bank_bytes(
                    (result.keep, result.volumes, result.moments, result.second)
                )
                if result.partition is not None:
                    retained_part += _array_bank_bytes(
                        (result.partition.counts, result.partition.simplices)
                    )
                retained_part += 512 + 8 * result.volumes.size
                storage.transfer("clip-reduction", "accepted-parts", retained_part)
                parts.append(result)
                kept_pairs.append(np.flatnonzero(result.keep) + start)
                del result
            del batch, partition
    counts = Counts(target_cells.size, accepted, piece_pairs, uncertain, 0, failures)
    # Exact retained sizes are now known; joins coexist with all accepted parts.
    retained = _entries_bytes(parts, dimension)
    storage.set(
        "entries",
        16384 + 2 * retained + 64 * accepted + 1024 * len(parts),
    )
    selected = np.concatenate(kept_pairs) if kept_pairs else np.empty((0,), np.int64)
    entries = Entries(
        target_cells=target_cells[selected],
        source_cells=source_cells[selected],
        volumes=_joined([part.volumes for part in parts], (0,)),
        first_moments=_joined([part.moments for part in parts], (0, dimension)),
        second_moments=(
            _joined(
                [part.second for part in parts if part.second is not None],
                (0, dimension, dimension),
            )
            if policy.second_moments
            else None
        ),
        simplex_offsets=(
            _offsets(
                _joined(
                    [
                        part.partition.counts
                        for part in parts
                        if part.partition is not None
                    ],
                    (0,),
                )
            )
            if policy.overlap_simplices
            else None
        ),
        simplices=(
            _joined(
                [
                    part.partition.simplices
                    for part in parts
                    if part.partition is not None
                ],
                (0, dimension + 1, dimension),
            )
            if policy.overlap_simplices
            else None
        ),
    )
    return _NarrowResult(entries, counts, _entries_bytes(parts, dimension), working)


def _joined(parts: list[np.ndarray], empty_shape: tuple[int, ...], /) -> np.ndarray:
    if not parts:
        return np.zeros(empty_shape, dtype=np.float64)
    return np.concatenate(parts)


def _offsets(counts: np.ndarray, /) -> np.ndarray:
    offsets = np.zeros((counts.size + 1,), dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


# ---------------------------------------------------------------- assembly


def _empty_entries(dimension: int, policy: CommonRefinementPolicy, /) -> Entries:
    index = np.empty((0,), dtype=np.int64)
    return Entries(
        target_cells=index,
        source_cells=index.copy(),
        volumes=np.empty((0,), dtype=np.float64),
        first_moments=np.empty((0, dimension), dtype=np.float64),
        second_moments=(
            np.empty((0, dimension, dimension), dtype=np.float64)
            if policy.second_moments
            else None
        ),
        simplex_offsets=(
            np.zeros((1,), dtype=np.int64) if policy.overlap_simplices else None
        ),
        simplices=(
            np.empty((0, dimension + 1, dimension), dtype=np.float64)
            if policy.overlap_simplices
            else None
        ),
    )


def _coverage_tolerances(
    part: Decomposition, policy: CommonRefinementPolicy, /
) -> np.ndarray:
    diameter = np.sqrt(np.sum((part.bbox_max - part.bbox_min) ** 2, axis=1))
    dimension = part.bbox_min.shape[1]
    floor = _COVERAGE_ROUNDOFF_ULPS * np.finfo(np.float64).eps * diameter**dimension
    return policy.coverage_tolerance * part.measures + floor


def _coverage_status(
    source_defects: np.ndarray,
    target_defects: np.ndarray,
    source_tolerances: np.ndarray,
    target_tolerances: np.ndarray,
    policy: CommonRefinementPolicy,
    /,
) -> tuple[CommonRefinementStatus, str]:
    source_gap = bool(np.any(-source_defects > source_tolerances))
    target_gap = bool(np.any(-target_defects > target_tolerances))
    if np.any(source_defects > source_tolerances) or np.any(
        target_defects > target_tolerances
    ):
        return (
            CommonRefinementStatus.DOUBLE_COVERAGE,
            "cells are covered more than once",
        )
    match policy.coverage:
        case CommonRefinementCoverage.COMPLETE:
            gap = source_gap or target_gap
        case CommonRefinementCoverage.TARGET:
            gap = target_gap
        case CommonRefinementCoverage.SOURCE:
            gap = source_gap
        case CommonRefinementCoverage.PARTIAL:
            gap = False
        case _:
            raise ValueError(
                f"CommonRefinementPolicy.coverage must be a supported coverage requirement; got {policy.coverage!r}."
            )
    if gap:
        return (
            CommonRefinementStatus.COVERAGE_GAP,
            f"{policy.coverage.value} coverage requirement is not met",
        )
    return CommonRefinementStatus.SUCCESS, "certified common refinement"


def _validate_meshes(
    source: CellMesh, target: CellMesh, policy: CommonRefinementPolicy, /
) -> None:

    dimension = source.topological_dimension
    if dimension not in (2, 3):
        raise ValueError(
            f"CommonRefinement.topological_dimension must be 2 or 3; got {dimension}."
        )
    if (
        target.topological_dimension != dimension
        or source.ambient_dimension != dimension
        or target.ambient_dimension != dimension
    ):
        raise ValueError(
            "CommonRefinement.mesh_dimensions must share one topological dimension "
            "equal to the ambient dimension; got incompatible source and target dimensions."
        )


def _admit_minimum_publication(
    source: CellMesh,
    target: CellMesh,
    policy: CommonRefinementPolicy,
    budget: CoordinateEnclosureBudget | None,
    /,
) -> None:
    """Refuse an impossible canonical negative artifact before any host bank.

    Even empty overlaps retain real per-cell measures, first moments, IDs,
    coverage defects/tolerances and the target CSR offsets. This lower bound
    excludes headers and construction scratch; it never invents zero measures.
    """
    source_count = sum(block.cell_count for block in source.blocks)
    target_count = sum(block.cell_count for block in target.blocks)
    payload = (
        8 * (source.topological_dimension + 4) * (source_count + target_count)
        + 4 * (target_count + 1)
        + (4 if policy.overlap_simplices else 0)
    )
    completed = (
        0
        if budget is None
        else (budget.retained_basis_bytes + budget.temporary_bytes_upper)
    )
    limit = policy.maximum_memory_bytes
    if budget is not None:
        limit = min(limit, budget.maximum_memory_bytes)
    if completed + payload > limit:
        raise CoordinateEnclosureResourceError(
            "polynomial_storage",
            limit,
            completed + payload,
            completed,
            admission=completed > limit,
        )


def _decomposition_failure(
    outside: bool, counts: Counts, /
) -> tuple[CommonRefinementStatus, str] | None:
    if outside:
        return (
            CommonRefinementStatus.PREDICATE_UNCERTAIN,
            "coordinates lie outside the meshcore exact domain",
        )
    if counts.invalid:
        return (
            CommonRefinementStatus.INVALID_GEOMETRY,
            "cells without a certified convex decomposition",
        )
    if counts.uncertain:
        return (
            CommonRefinementStatus.PREDICATE_UNCERTAIN,
            "unresolved decomposition predicates",
        )
    return None


@checked
def prepare_common_refinement(
    source: CellMesh,
    target: CellMesh,
    /,
    *,
    policy: CommonRefinementPolicy = CommonRefinementPolicy(),
    source_geometry: CellGeometrySpec | None = None,
    target_geometry: CellGeometrySpec | None = None,
    scope: meshcore_owner.NativeExecutionScope | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> PreparedCommonRefinement:
    """Certify the common refinement of two cell meshes of one dimension.

    Requires meshcore (raises :class:`MeshcoreUnavailableError` otherwise).
    Geometric failures are reported by the returned status, with empty entries
    when the narrow phase could not run. Storage admission refuses before
    allocation, raising ``CoordinateEnclosureResourceError`` if no result fits.
    A live native budget owns host and native banks on its original ticket.
    A standalone host budget remains supported, but its native coverage is
    explicitly partial: native work uses the finite policy/supplied scope,
    not a newly minted allowance on the host ticket.
    """

    # _exact_power_refinement consumes these policy and result declarations; import after they load.
    import phydrax.geometry._exact_power_refinement as exact_power_refinement_owner

    _validate_meshes(source, target, policy)
    if (
        source_geometry is not None
        and target_geometry is not None
        and all(block.cell_kind == "tetrahedron" for mesh in (source, target) for block in mesh.blocks)
    ):
        if budget is None or budget.ticket.settled:
            scope = scope or meshcore_owner.current_native_execution_scope()
            ticket = (
                HostBatchTicket(
                    policy.maximum_exact_work, queries_upper=policy.maximum_candidate_pairs,
                    bytes_upper=policy.maximum_memory_bytes,
                )
                if scope is None else scope.batch(
                    min(policy.maximum_exact_work, scope.remaining().remaining_work_units),
                    queries_upper=min(policy.maximum_candidate_pairs, scope.remaining().remaining_geometry_queries),
                    bytes_upper=min(policy.maximum_memory_bytes, scope.remaining().remaining_managed_bytes),
                )
            )
            if budget is not None:
                with budget.bind_ticket(ticket), budget.activate():
                    return prepare_exact_tetrahedral_refinement(
                        source, target, source_geometry, target_geometry, policy, budget,
                    ).refinement
            with ticket:
                ledger = CoordinateEnclosureBudget(
                    ticket, maximum_work_units=policy.maximum_exact_work,
                    maximum_memory_bytes=policy.maximum_memory_bytes,
                )
                with ledger.activate():
                    return prepare_exact_tetrahedral_refinement(
                        source, target, source_geometry, target_geometry, policy, ledger,
                    ).refinement
        return prepare_exact_tetrahedral_refinement(
            source, target, source_geometry, target_geometry, policy, budget,
        ).refinement
    if source_geometry is not None or target_geometry is not None:
        if budget is not None and not budget.ticket.settled:
            with budget.activate():
                return exact_power_refinement_owner.prepare_exact_power_refinement(
                    source,
                    target,
                    source_geometry,
                    target_geometry,
                    policy,
                    budget=budget,
                )
        if scope is None:
            ticket = HostBatchTicket(
                policy.maximum_exact_work,
                queries_upper=policy.maximum_candidate_pairs,
                bytes_upper=policy.maximum_memory_bytes,
            )
        else:
            remaining = scope.remaining()
            ticket = scope.batch(
                min(policy.maximum_exact_work, remaining.remaining_work_units),
                queries_upper=min(
                    policy.maximum_candidate_pairs, remaining.remaining_geometry_queries
                ),
                bytes_upper=min(
                    policy.maximum_memory_bytes, remaining.remaining_managed_bytes
                ),
            )
        if budget is not None:
            with budget.bind_ticket(ticket):
                return exact_power_refinement_owner.prepare_exact_power_refinement(
                    source,
                    target,
                    source_geometry,
                    target_geometry,
                    policy,
                    budget=budget,
                )
        view = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=policy.maximum_exact_work,
            maximum_memory_bytes=policy.maximum_memory_bytes,
        )
        with ticket, view.activate():
            return exact_power_refinement_owner.prepare_exact_power_refinement(
                source, target, source_geometry, target_geometry, policy, budget=view
            )
    _admit_minimum_publication(source, target, policy, budget)
    if (
        budget is not None
        and not budget.ticket.settled
        and budget.ticket.native_accounting
    ):
        execution = scope or meshcore_owner.current_native_execution_scope()
        if execution is None:
            raise RuntimeError(
                "prepare_common_refinement.budget must execute inside its native "
                "request scope; got no active scope."
            )
        return prepare_common_refinement_active(
            source, target, policy=policy, scope=execution, budget=budget
        )
    with (
        meshcore_owner.NativeExecutionScope(policy.native_limits())
        if scope is None
        else nullcontext(scope)
    ) as execution:
        return prepare_common_refinement_active(
            source, target, policy=policy, scope=execution, budget=budget
        )


def _refinement_native_limits(
    policy: CommonRefinementPolicy,
    budget: CoordinateEnclosureBudget,
    /,
) -> meshcore_owner.NativeExecutionLimits:
    """Tighten native residuals without reserving or replenishing the request."""
    work = queries = policy.maximum_exact_work
    if budget.ticket.native_accounting:
        # Scope.remaining() at a reserved root would subtract this ticket's
        # reservation again. The original ticket exposes its true residual;
        # the borrowed native window also enforces every ancestor's limits.
        work = min(work, budget.ticket.remaining_work_units)
        queries = min(queries, budget.ticket.remaining_geometry_queries)
    # Host q=0 says nothing about standalone native coverage; that API remains
    # explicitly partial and uses the existing finite native policy/scope.
    memory = max(
        0,
        min(policy.maximum_memory_bytes, budget.maximum_memory_bytes)
        - budget.retained_basis_bytes
        - budget.temporary_bytes_upper,
    )
    return meshcore_owner.NativeExecutionLimits(work, queries, 0, memory, math.inf)


@checked
def prepare_common_refinement_active(
    source: CellMesh,
    target: CellMesh,
    /,
    *,
    policy: CommonRefinementPolicy,
    scope: meshcore_owner.NativeExecutionScope,
    budget: CoordinateEnclosureBudget | None = None,
) -> PreparedCommonRefinement:
    """Prepare bounded banks, sharing the native ledger when supplied.

    Host-only tickets retain their standalone API and account host storage and
    preparation work only (PARTIAL native coverage). They are not promoted to a
    qualified whole-native request by this helper.
    """
    _validate_meshes(source, target, policy)
    _admit_minimum_publication(source, target, policy, budget)
    meshcore_owner.load_meshcore()
    if budget is None or budget.ticket.settled:
        remaining = scope.remaining()
        ticket = scope.batch(
            min(policy.maximum_exact_work, remaining.remaining_work_units),
            queries_upper=min(
                policy.maximum_exact_work, remaining.remaining_geometry_queries
            ),
            bytes_upper=min(
                policy.maximum_memory_bytes, remaining.remaining_managed_bytes
            ),
        )
        if budget is None:
            view = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=policy.maximum_exact_work,
                maximum_memory_bytes=policy.maximum_memory_bytes,
            )
            with ticket, view.activate():
                return prepare_common_refinement_active(
                    source, target, policy=policy, scope=scope, budget=view
                )
        with budget.bind_ticket(ticket):
            return prepare_common_refinement_active(
                source, target, policy=policy, scope=scope, budget=budget
            )
    with (
        budget.activate(),
        budget.bound_stage(policy.maximum_exact_work, policy.maximum_memory_bytes),
        budget.ticket.native_target()
        if budget.ticket.native_accounting
        else nullcontext(),
        # Enter after the original target: this is a tightening of its existing
        # pool/ledger, not another native reservation or a replacement ticket.
        # A memory window excludes its entry host baseline, so subtract that
        # live baseline from the allowed additional bytes.
        meshcore_owner.NativeExecutionScope(_refinement_native_limits(policy, budget)),
        budget.live_storage() as lease,
    ):
        storage = _RefinementStorage(budget, lease)
        result = _prepare_common_refinement_host(source, target, policy, storage)
        return storage.publish(result)


def _prepare_common_refinement_host(
    source: CellMesh,
    target: CellMesh,
    policy: CommonRefinementPolicy,
    storage: _RefinementStorage,
    /,
) -> PreparedCommonRefinement:
    identities = (
        source.mesh_id,
        target.mesh_id,
        source.topology_id,
        target.topology_id,
    )
    with storage.scratch(
        "exact-domain-scan",
        8192 + 32 * max(source.coordinates.size, target.coordinates.size),
    ):
        outside = _outside_exact_domain(source) or _outside_exact_domain(target)
    mode = PredicateMode.FILTERED if outside else policy.predicate_mode
    storage.set("source-decomposition", _decomposition_bound(source, storage))
    first = _decompose(source, mode, budget=storage.budget)
    storage.set("source-decomposition", _decomposition_bytes(first))
    storage.set("target-decomposition", _decomposition_bound(target, storage))
    second = _decompose(target, mode, budget=storage.budget)
    storage.set("target-decomposition", _decomposition_bytes(second))
    base_bytes = sum(
        array.nbytes
        for part in (first, second)
        for array in (part.pieces, part.piece_offsets, part.bbox_min, part.bbox_max)
    )
    counts = Counts(
        0, 0, 0, first.uncertain + second.uncertain, first.invalid + second.invalid, 0
    )
    storage.counts = counts
    failure = _decomposition_failure(outside, counts)
    if failure is not None:
        return _prepared(
            first,
            second,
            None,
            counts,
            *failure,
            identities,
            policy,
            0,
            base_bytes,
            storage=storage,
        )
    try:
        return _prepare_common_refinement_pairs(
            first, second, counts, identities, policy, base_bytes, storage
        )
    except CoordinateEnclosureResourceError as error:
        if error.resource == "coefficient_work":
            raise
        failure_reason = str(error)
    # The failed helper and its traceback have expired. Release only retired
    # broad/narrow banks; real cell measures/bounds remain live for evidence.
    for name in (
        "candidates",
        "accepted-parts",
        "batch-boundaries",
        "entries",
        "publication",
    ):
        storage.set(name, 0)
    return _prepared(
        first,
        second,
        None,
        storage.counts,
        CommonRefinementStatus.RESOURCE_LIMIT,
        f"accepted pairs or memory exceed the policy limits; {failure_reason}",
        identities,
        policy,
        0,
        storage.peak,
        storage=storage,
    )


def _prepare_common_refinement_pairs(
    first: Decomposition,
    second: Decomposition,
    counts: Counts,
    identities: tuple[str, str, str, str],
    policy: CommonRefinementPolicy,
    base_bytes: int,
    storage: _RefinementStorage,
    /,
) -> PreparedCommonRefinement:
    candidates = _candidate_pairs(first, second, policy.maximum_candidate_pairs, storage)
    if candidates is None:
        counts = counts._replace(candidates=policy.maximum_candidate_pairs + 1)
        return _prepared(
            first,
            second,
            None,
            counts,
            CommonRefinementStatus.RESOURCE_LIMIT,
            "broad-phase candidate pairs exceed maximum_candidate_pairs",
            identities,
            policy,
            0,
            base_bytes,
            storage=storage,
        )
    target_cells, source_cells = candidates
    storage.set("candidates", _array_bank_bytes(candidates) + 1024)
    base_bytes += target_cells.nbytes + source_cells.nbytes
    narrow = _narrow_phase(
        first, second, target_cells, source_cells, policy, base_bytes, storage=storage
    )
    # _narrow_phase's local parts, masks and boundary banks have expired.
    storage.set("accepted-parts", 0)
    storage.set("batch-boundaries", 0)
    if narrow.entries is None:
        return _prepared(
            first,
            second,
            None,
            narrow.counts,
            CommonRefinementStatus.RESOURCE_LIMIT,
            "accepted pairs or memory exceed the policy limits",
            identities,
            policy,
            narrow.retained_bytes,
            narrow.working_bytes,
            storage=storage,
        )
    storage.set("entries", _array_bank_bytes(narrow.entries) + 1024)
    return _prepared(
        first,
        second,
        narrow.entries,
        narrow.counts,
        None,
        "",
        identities,
        policy,
        narrow.retained_bytes,
        narrow.working_bytes,
        storage=storage,
    )


def _prepared(
    source: Decomposition,
    target: Decomposition,
    entries: Entries | None,
    counts: Counts,
    status: CommonRefinementStatus | None,
    reason: str,
    identities: tuple[str, str, str, str],
    policy: CommonRefinementPolicy,
    retained_bytes: int,
    working_bytes: int,
    /,
    *,
    storage: _RefinementStorage,
) -> PreparedCommonRefinement:
    dimension = source.first_moments.shape[1]
    record_bytes = (
        0
        if entries is None
        else sum(array.nbytes for array in entries if array is not None)
    )
    cells = source.cell_count + target.cell_count
    accepted = 0 if entries is None else entries.volumes.size
    # Coverage vectors, CSR count/cumsum/diff buffers, dtype conversions,
    # logical published arrays, hashing byte images and object metadata.
    storage.set(
        "publication",
        65536
        + 4 * record_bytes
        + 256 * cells
        + 64 * accepted
        + 4 * (target.cell_count + 1),
    )
    records = _empty_entries(dimension, policy) if entries is None else entries
    source_defects = (
        np.bincount(
            records.source_cells, weights=records.volumes, minlength=source.cell_count
        )
        - source.measures
    )
    target_defects = (
        np.bincount(
            records.target_cells, weights=records.volumes, minlength=target.cell_count
        )
        - target.measures
    )
    source_tolerances = _coverage_tolerances(source, policy)
    target_tolerances = _coverage_tolerances(target, policy)
    if status is None:
        if counts.uncertain:
            status, reason = (
                CommonRefinementStatus.PREDICATE_UNCERTAIN,
                "native clipping could not certify piece pairs",
            )
        elif counts.failures:
            status, reason = (
                CommonRefinementStatus.INTERSECTION_FAILURE,
                "native clipping failed for certified piece pairs",
            )
        else:
            status, reason = _coverage_status(
                source_defects,
                target_defects,
                source_tolerances,
                target_tolerances,
                policy,
            )
    evidence = CommonRefinementEvidence(
        status=status,
        reason=reason,
        source_coverage_defects=source_defects,
        target_coverage_defects=target_defects,
        source_measures=source.measures,
        target_measures=target.measures,
        source_tolerances=source_tolerances,
        target_tolerances=target_tolerances,
        counts=counts,
        retained_bytes=retained_bytes,
        working_bytes=working_bytes,
    )
    return PreparedCommonRefinement(
        source=source,
        target=target,
        entries=records,
        identities=identities,
        policy=policy,
        evidence=evidence,
    )


__all__ = [
    "CommonRefinementCoverage",
    "CommonRefinementEvidence",
    "CommonRefinementPolicy",
    "CommonRefinementStatus",
    "PreparedCommonRefinement",
    "prepare_common_refinement",
]


class ExactTetrahedralRefinementError(ValueError):
    """Typed exact simplex refusal without a fabricated successful partition."""

    def __init__(self, status: CommonRefinementStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class ExactTetrahedralRefinement(NamedTuple):
    refinement: PreparedCommonRefinement
    exact_volumes: tuple[Fraction, ...]
    source_measures: tuple[Fraction, ...]
    target_measures: tuple[Fraction, ...]


def _material_decomposition(
    mesh: CellMesh, geometry: CellGeometrySpec, budget: CoordinateEnclosureBudget,
    storage: CoordinateLiveStorage,
) -> tuple[Decomposition, tuple[Tetrahedron, ...], tuple[Fraction, ...]]:
    """Read actual rational nodal rows, never their rounded carrier vertices."""
    from ..discretization._cell_geometry import BarycentricCellGeometryElement
    from ..discretization._exact_algebraic_geometry import rational_coefficient
    from ..discretization.fem._reference_types import FiniteElementSpec

    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    if any(
        not isinstance(element, (FiniteElementSpec, BarycentricCellGeometryElement))
        or element.cell_kind != "tetrahedron"
        or element.degree != 1
        for element in elements
    ):
        raise ValueError("exact tetrahedral refinement requires affine simplex coordinates")
    count = sum(block.cell_count for block in mesh.blocks)
    storage.set_bound(storage.bound + 4096 + count * 4096)
    points = exact_vertices(mesh, geometry, budget=budget)
    rational_points = tuple(
        tuple(rational_coefficient(value, role="exact tetrahedral coordinate") for value in row)
        for row in points
    )
    bits = max(
        (max(value.numerator.bit_length(), value.denominator.bit_length())
         for row in rational_points for value in row), default=1,
    )
    storage.set_bound(storage.bound + count * (1024 + 64 * (24 * bits // 8 + 1)))
    tetrahedra = tuple(
        cast(Tetrahedron, tuple(rational_points[index] for index in row))
        for block in routes for row in np.asarray(block)
    )
    measures = tuple(star_determinant(tet, budget=budget) / 6 for tet in tetrahedra)
    if any(value <= 0 for value in measures):
        raise ExactTetrahedralRefinementError(CommonRefinementStatus.INVALID_GEOMETRY, "nonpositive exact tetrahedron")
    budget.reserve(count * 18)
    moments = tuple(
        tuple(volume * sum((point[axis] for point in tet), Fraction(0)) / 4 for axis in range(3))
        for volume, tet in zip(measures, tetrahedra, strict=True)
    )
    floating = np.asarray(tetrahedra, dtype=np.float64)
    carrier = Decomposition(
        count, np.concatenate(tuple(np.asarray(block.global_ids) for block in mesh.blocks)),
        floating, None, np.arange(count + 1, dtype=np.int64),
        np.asarray(measures, dtype=np.float64), np.asarray(moments, dtype=np.float64),
        np.nextafter(floating.min(axis=1), -np.inf),
        np.nextafter(floating.max(axis=1), np.inf), 0, 0,
    )
    return carrier, tetrahedra, measures


def prepare_exact_tetrahedral_refinement(
    source: CellMesh, target: CellMesh, source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec, policy: CommonRefinementPolicy,
    budget: CoordinateEnclosureBudget,
) -> ExactTetrahedralRefinement:
    """Assemble the canonical refinement from exact rational tetra intersections."""
    from ._exact_power_refinement import _rne_error

    with (
        budget.bound_stage(policy.maximum_exact_work, policy.maximum_memory_bytes),
        budget.measurement() as measurement,
        budget.temporary_scope(),
        budget.live_storage() as storage,
        budget.live_storage() as candidates_storage,
        budget.live_storage() as pair_storage,
        budget.live_storage() as workspace,
    ):
        first, first_tets, first_measures = _material_decomposition(
            source, source_geometry, budget, storage,
        )
        second, second_tets, second_measures = _material_decomposition(
            target, target_geometry, budget, storage,
        )
        if policy.coverage is CommonRefinementCoverage.COMPLETE:
            # Both row sets must be partitions of the same domain. Their exact
            # total measures must therefore agree before any pair clipping can
            # possibly certify complete coverage. This is only a refusal test:
            # equal totals still require all local intersection/coverage proofs.
            with budget.temporary_scope():
                count = len(first_measures) + len(second_measures)
                denominator_bits = sum(
                    value.denominator.bit_length()
                    for group in (first_measures, second_measures) for value in group
                )
                numerator_bits = max(
                    (value.numerator.bit_length()
                     for group in (first_measures, second_measures) for value in group),
                    default=1,
                )
                sum_bits = denominator_bits + numerator_bits + count.bit_length() + 2
                budget.reserve(count + 1, 1024 + 8 * declared_integer_storage_bytes(sum_bits))
                if sum(first_measures, Fraction(0)) != sum(second_measures, Fraction(0)):
                    raise ExactTetrahedralRefinementError(
                        CommonRefinementStatus.COVERAGE_GAP,
                        "exact tetrahedral complete coverage requires equal total volumes",
                    )
        candidates = _candidate_pairs(
            first, second, policy.maximum_candidate_pairs,
            _RefinementStorage(budget, candidates_storage),
        )
        if candidates is None:
            raise ExactTetrahedralRefinementError(CommonRefinementStatus.RESOURCE_LIMIT, "exact tetrahedral candidate limit")
        targets, sources = candidates
        covered_source = [Fraction(0)] * first.cell_count
        covered_target = [Fraction(0)] * second.cell_count
        rows, pieces, offsets, errors = [], [], [0], []
        exact_volumes = []
        preparation = TetrahedronPlanePreparation(budget)
        for target_row, source_row in zip(targets, sources, strict=True):
            overlap = tetrahedron_overlap(
                first_tets[source_row], second_tets[target_row],
                second_moments=policy.second_moments, retain_simplices=policy.overlap_simplices,
                budget=budget, preparation=preparation,
                _output_storage=pair_storage, _workspace=workspace,
            )
            if isinstance(overlap, ExactOverlapRefusal):
                raise ExactPolyhedralRefusal(overlap)
            if overlap.volume == 0:
                del overlap
                pair_storage.set_bound(0)
                continue
            if len(rows) >= policy.maximum_accepted_pairs:
                raise ExactTetrahedralRefinementError(CommonRefinementStatus.RESOURCE_LIMIT, "exact tetrahedral accepted-pair limit")
            simplices = () if overlap.simplices is None else overlap.simplices
            storage.set_bound(storage.bound + 4096 + len(simplices) * 384)
            budget.reserve(2)
            covered_source[source_row] += overlap.volume
            covered_target[target_row] += overlap.volume
            rows.append((
                target_row, source_row, float(overlap.volume),
                tuple(map(float, overlap.first_moment)),
                None if overlap.second_moment is None else tuple(
                    tuple(map(float, row)) for row in overlap.second_moment
                ),
            ))
            errors.append(_rne_error(overlap.volume))
            exact_volumes.append(overlap.volume)
            pieces.extend(np.asarray(simplices, dtype=np.float64))
            offsets.append(len(pieces))
            del overlap, simplices
            pair_storage.set_bound(0)
        budget.reserve(first.cell_count + second.cell_count)
        source_defects = tuple(a - b for a, b in zip(covered_source, first_measures, strict=True))
        target_defects = tuple(a - b for a, b in zip(covered_target, second_measures, strict=True))
        defects = source_defects + target_defects
        if any(value > 0 for value in defects):
            raise ExactTetrahedralRefinementError(CommonRefinementStatus.DOUBLE_COVERAGE, "exact tetrahedral double coverage")
        if (
            policy.coverage in (CommonRefinementCoverage.COMPLETE, CommonRefinementCoverage.SOURCE)
            and any(source_defects)
        ) or (
            policy.coverage in (CommonRefinementCoverage.COMPLETE, CommonRefinementCoverage.TARGET)
            and any(target_defects)
        ):
            raise ExactTetrahedralRefinementError(CommonRefinementStatus.COVERAGE_GAP, "exact tetrahedral coverage gap")
        entries = Entries(
            np.asarray([row[0] for row in rows], dtype=np.int32),
            np.asarray([row[1] for row in rows], dtype=np.int32),
            np.asarray([row[2] for row in rows], dtype=np.float64),
            np.asarray([row[3] for row in rows], dtype=np.float64).reshape((-1, 3)),
            np.asarray([row[4] for row in rows], dtype=np.float64).reshape((-1, 3, 3))
            if policy.second_moments else None,
            np.asarray(offsets, dtype=np.int32) if policy.overlap_simplices else None,
            np.asarray(pieces, dtype=np.float64).reshape((-1, 4, 3)) if policy.overlap_simplices else None,
        )
        evidence = CommonRefinementEvidence(
            status=CommonRefinementStatus.SUCCESS, reason="exact tetrahedral coverage",
            source_coverage_defects=np.asarray(source_defects, dtype=np.float64),
            target_coverage_defects=np.asarray(target_defects, dtype=np.float64),
            source_measures=first.measures, target_measures=second.measures,
            source_tolerances=np.asarray([_rne_error(value) for value in first_measures]),
            target_tolerances=np.asarray([_rne_error(value) for value in second_measures]),
            counts=Counts(len(targets), len(rows), len(targets), 0, 0, 0),
            retained_bytes=storage.bound,
            working_bytes=measurement.peak_bytes_upper,
        )
        result = PreparedCommonRefinement(
            source=first, target=second, entries=entries,
            identities=(source.mesh_id, target.mesh_id, source.topology_id, target.topology_id),
            geometry_ids=(cell_geometry_id(source_geometry), cell_geometry_id(target_geometry)),
            volume_error_bounds=np.asarray(errors), policy=policy, evidence=evidence,
        )
        candidates_storage.set_bound(0)
        workspace.set_bound(0)
        result = ExactTetrahedralRefinement(result, tuple(exact_volumes), first_measures, second_measures)
        budget.retain_owner(result, temporary_storage_credit=storage)
        return result
